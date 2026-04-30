"""Low-rank kernel approximations for scalable attention kernel regression."""

import logging
from typing import Optional

import torch

from laker.kernels import _exp_safe

logger = logging.getLogger(__name__)


class NystromAttentionKernelOperator:
    """Nyström low-rank approximation of the exponential attention kernel.

    Approximates ``G = exp(E E^T)`` using ``m`` landmark points:

    .. math::
        G \\approx G_{nm} G_{mm}^{-1} G_{nm}^T

    where ``G_{nm}`` is the kernel between all ``n`` points and ``m`` landmarks,
    and ``G_{mm}`` is the kernel among landmarks. This reduces matvec cost
    from ``O(n^2)`` to ``O(n*m)``.

    Args:
        embeddings: Tensor of shape ``(n, embedding_dim)``.
        lambda_reg: Tikhonov regularisation ``lambda``.
        num_landmarks: Number of landmark points ``m``. If ``None``, defaults to
            ``max(50, int(sqrt(n)))``.
        chunk_size: Chunk size for landmark kernel evaluation.
        device: torch device.
        dtype: torch dtype.

    Raises:
        ValueError: If ``embeddings`` is not 2-D.
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        lambda_reg: float = 1e-2,
        num_landmarks: Optional[int] = None,
        chunk_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if embeddings.dim() != 2:
            raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
        self.n = embeddings.shape[0]
        self.embedding_dim = embeddings.shape[1]
        self.lambda_reg = float(lambda_reg)
        self.chunk_size = chunk_size

        if device is None:
            device = embeddings.device
        if dtype is None:
            dtype = embeddings.dtype

        self.device = device
        self.dtype = dtype
        self.embeddings = embeddings.to(device=device, dtype=dtype)

        m = num_landmarks if num_landmarks is not None else max(50, int(self.n ** 0.5))
        self.m = min(m, self.n)

        # Sample landmarks via k-means++ style greedy selection
        self.landmark_indices = self._select_landmarks()
        self.landmark_embeddings = self.embeddings[self.landmark_indices]

        # Compute K_nm (n, m) and K_mm (m, m)
        self.k_nm = self._compute_kernel_matrix(self.embeddings, self.landmark_embeddings)
        self.k_mm = self._compute_kernel_matrix(self.landmark_embeddings, self.landmark_embeddings)

        # Regularised Cholesky of K_mm for stable solves
        k_mm_reg = self.k_mm + 1e-6 * torch.eye(self.m, device=device, dtype=dtype)
        self.k_mm_chol = torch.linalg.cholesky(k_mm_reg)

        # Precompute K_nm @ K_mm^{-1} for fast matvecs via Cholesky solve
        self.k_nm_kmm_inv = torch.linalg.solve_triangular(
            self.k_mm_chol.T,
            torch.linalg.solve_triangular(self.k_mm_chol, self.k_nm.T, upper=False),
            upper=True,
        ).T  # (n, m)

        self.shape = (self.n, self.n)

    def _select_landmarks(self) -> torch.Tensor:
        """Greedy landmark selection (k-means++ style)."""
        indices = torch.zeros(self.m, dtype=torch.long, device=self.device)
        # First landmark: random
        indices[0] = torch.randint(0, self.n, (1,), device=self.device)

        for i in range(1, self.m):
            # Compute squared distances to nearest landmark
            selected = self.embeddings[indices[:i]]
            dists = torch.cdist(self.embeddings, selected) ** 2
            min_dists = dists.min(dim=1).values
            # Pick point with max distance
            indices[i] = min_dists.argmax()

        return indices

    def _compute_kernel_matrix(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Compute exponential attention kernel matrix."""
        gram = x @ y.T
        return _exp_safe(gram)

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``(lambda I + G_approx)`` to vector(s)."""
        out = self.lambda_reg * x
        if x.dim() == 1:
            # G @ x = K_nm @ K_mm^{-1} @ K_nm^T @ x
            temp = self.k_nm_kmm_inv.T @ x  # (m,)
            out = out + self.k_nm_kmm_inv @ temp
        else:
            temp = self.k_nm_kmm_inv.T @ x  # (m, k)
            out = out + self.k_nm_kmm_inv @ temp
        return out

    def diagonal(self) -> torch.Tensor:
        """Return diagonal of ``lambda I + G_approx``."""
        # Diagonal of G_approx = sum over j of (K_nm @ K_mm^{-1})_{ij} * (K_nm)_{ij}
        diag_approx = torch.sum(self.k_nm_kmm_inv * self.k_nm, dim=1)
        return self.lambda_reg + diag_approx

    def to_dense(self) -> torch.Tensor:
        """Materialise full approximate kernel matrix."""
        k_approx = self.k_nm_kmm_inv @ self.k_nm.T
        k_approx.diagonal().add_(self.lambda_reg)
        return k_approx

    def kernel_eval(self, x: torch.Tensor, y: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Evaluate kernel between queries and training points."""
        if y is None:
            y = self.embeddings
        gram = x @ y.T
        return _exp_safe(gram)


class RandomFeatureAttentionKernelOperator:
    """Random Fourier Feature (RFF) approximation of the exponential kernel.

    Uses random Fourier features to approximate the Gaussian-like kernel
    induced by the exponential of inner products. For embeddings with bounded
    norm, the exponential kernel is approximated by a finite-dimensional
    feature map.

    Args:
        embeddings: Tensor of shape ``(n, embedding_dim)``.
        lambda_reg: Tikhonov regularisation ``lambda``.
        num_features: Number of random Fourier features. If ``None``, defaults
            to ``max(100, int(sqrt(n) * 2))``.
        sigma: Bandwidth for random Fourier features.
        device: torch device.
        dtype: torch dtype.

    Raises:
        ValueError: If ``embeddings`` is not 2-D.
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        lambda_reg: float = 1e-2,
        num_features: Optional[int] = None,
        sigma: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if embeddings.dim() != 2:
            raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
        self.n = embeddings.shape[0]
        self.embedding_dim = embeddings.shape[1]
        self.lambda_reg = float(lambda_reg)
        self.sigma = float(sigma)

        if device is None:
            device = embeddings.device
        if dtype is None:
            dtype = embeddings.dtype

        self.device = device
        self.dtype = dtype
        self.embeddings = embeddings.to(device=device, dtype=dtype)

        r = num_features if num_features is not None else max(100, int(self.n ** 0.5 * 2))
        self.num_features = r

        # Generate random Fourier frequencies
        gen = torch.Generator(device=device).manual_seed(42)
        self.freq = torch.randn(
            self.embedding_dim, r, generator=gen, device=device, dtype=dtype
        ) / sigma
        self.phase = torch.rand(r, generator=gen, device=device, dtype=dtype) * 2.0 * 3.141592653589793

        # Compute feature map Phi: (n, 2*r) [cos, sin]
        proj = self.embeddings @ self.freq
        phi = torch.cat([torch.cos(proj + self.phase), torch.sin(proj + self.phase)], dim=1)
        # Normalise so that Phi @ Phi.T approximates the kernel
        self.phi = phi / (r ** 0.5)

        self.shape = (self.n, self.n)

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``(lambda I + G_approx)`` to vector(s)."""
        out = self.lambda_reg * x
        if x.dim() == 1:
            temp = self.phi.T @ x  # (2*r,)
            out = out + self.phi @ temp
        else:
            temp = self.phi.T @ x  # (2*r, k)
            out = out + self.phi @ temp
        return out

    def diagonal(self) -> torch.Tensor:
        """Return diagonal of ``lambda I + G_approx``."""
        sq_norms = torch.sum(self.phi ** 2, dim=1)
        return self.lambda_reg + sq_norms

    def to_dense(self) -> torch.Tensor:
        """Materialise full approximate kernel matrix."""
        k_approx = self.phi @ self.phi.T
        k_approx.diagonal().add_(self.lambda_reg)
        return k_approx

    def kernel_eval(self, x: torch.Tensor, y: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Evaluate approximate kernel between queries and training points."""
        if y is None:
            y = self.embeddings
        # Build RFF feature maps for both sets using the same random frequencies
        proj_x = x @ self.freq
        phi_x = torch.cat([torch.cos(proj_x + self.phase), torch.sin(proj_x + self.phase)], dim=1)
        proj_y = y @ self.freq
        phi_y = torch.cat([torch.cos(proj_y + self.phase), torch.sin(proj_y + self.phase)], dim=1)
        return (phi_x @ phi_y.T) / self.num_features
