"""Structured Kernel Interpolation (SKI) approximation for attention kernels.

SKI approximates ``K ≈ W K_grid W^T`` where ``W`` contains interpolation
weights from data points to a regular product grid in the embedding space.
Because the grid is regular, the full ``K_grid`` can be materialised explicitly
(since ``grid_size`` is small) and matvecs are ``O(n * grid_size)``.

Warning:
    The current implementation builds a product grid in the *embedding* space.
    For ``embedding_dim=10`` and 2 points per dimension the grid has
    ``2^10 = 1024`` points, which is practical.  For much larger
    ``embedding_dim`` the grid size grows exponentially; in that case use
    Nyström or RFF instead.
"""

import logging
import math
from typing import Optional

import torch

from laker.kernels import _exp_safe

logger = logging.getLogger(__name__)


def _multilinear_weights(x: torch.Tensor, grid_1d: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Multilinear interpolation weights for a product grid.

    Args:
        x: Tensor of shape ``(n, d)`` with coordinates in [0, 1] per dim.
        grid_1d: List of ``d`` tensors, each of shape ``(g_i,)`` with the
            1-D grid coordinates (sorted ascending).

    Returns:
        indices: Long tensor of shape ``(n, num_vertices)`` with grid-point
            linear indices.
        weights: Tensor of shape ``(n, num_vertices)`` with interpolation
            weights (sum to 1 per row).
    """
    n, d = x.shape
    g_per_dim = [g.shape[0] for g in grid_1d]
    vertices = 2 ** d

    # For each dimension find the lower bin index and fractional offset
    low_idx = torch.zeros(n, d, dtype=torch.long, device=x.device)
    frac = torch.zeros(n, d, dtype=x.dtype, device=x.device)
    for dim, g in enumerate(grid_1d):
        g = g.to(x.device, x.dtype)
        # Clamp x to grid bounds
        xc = x[:, dim].clamp(min=g[0], max=g[-1])
        # Find the rightmost grid point <= xc (searchsorted is not on all PyTorch versions)
        # Use broadcasting approach
        diff = xc.unsqueeze(1) - g.unsqueeze(0)  # (n, g_i)
        # Find last non-positive difference
        mask = diff <= 0
        # For values exactly at grid points, this gives the next index; handle separately
        idx = (diff > 0).sum(dim=1) - 1
        idx = idx.clamp(min=0, max=g.shape[0] - 2)
        low = g[idx]
        high = g[idx + 1]
        denom = high - low
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        low_idx[:, dim] = idx
        frac[:, dim] = (xc - low) / denom

    # Build all 2^d vertex combinations
    vertex_offsets = torch.arange(vertices, device=x.device)
    bits = ((vertex_offsets.unsqueeze(1) >> torch.arange(d, device=x.device)) & 1).to(torch.bool)

    # Grid strides for linear index
    strides = [1]
    for g in reversed(g_per_dim[1:]):
        strides.append(strides[-1] * g)
    strides = list(reversed(strides))
    strides_t = torch.tensor(strides, dtype=torch.long, device=x.device)

    indices = torch.zeros(n, vertices, dtype=torch.long, device=x.device)
    weights = torch.ones(n, vertices, dtype=x.dtype, device=x.device)
    for dim in range(d):
        dim_idx = low_idx[:, dim].unsqueeze(1) + bits[:, dim].unsqueeze(0).long()  # (n, vertices)
        indices += dim_idx * strides_t[dim]
        dim_weight = torch.where(
            bits[:, dim].unsqueeze(0),
            frac[:, dim].unsqueeze(1),
            1.0 - frac[:, dim].unsqueeze(1),
        )
        weights *= dim_weight

    return indices, weights


class SKIAttentionKernelOperator:
    """SKI approximation of the exponential attention kernel.

    Builds a regular product grid in the embedding space and uses multilinear
    interpolation weights ``W`` so that ``K ≈ W K_grid W^T``.

    Args:
        embeddings: Tensor of shape ``(n, embedding_dim)``.
        lambda_reg: Tikhonov regularisation ``lambda``.
        grid_size: Maximum number of grid points.  The actual grid is a
            product grid with ``floor(grid_size**(1/d))`` points per
            dimension, capped so the product does not exceed ``grid_size``.
        grid_bounds: Optional ``(d, 2)`` tensor with ``[min, max]`` per
            dimension.  If ``None``, inferred from data.
        device: torch device.
        dtype: torch dtype.

    Raises:
        ValueError: If ``embeddings`` is not 2-D or ``grid_size`` is too small.
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        lambda_reg: float = 1e-2,
        grid_size: Optional[int] = None,
        grid_bounds: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if embeddings.dim() != 2:
            raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
        self.n = embeddings.shape[0]
        self.embedding_dim = embeddings.shape[1]
        self.lambda_reg = float(lambda_reg)

        if device is None:
            device = embeddings.device
        if dtype is None:
            dtype = embeddings.dtype

        self.device = device
        self.dtype = dtype
        self.embeddings = embeddings.to(device=device, dtype=dtype)

        d = self.embedding_dim
        if grid_size is None:
            grid_size = min(4096, max(64, 2 ** d))
        self.grid_size = grid_size

        if grid_size < 2:
            raise ValueError("grid_size must be at least 2")

        # Determine points per dimension for product grid
        gpd = max(2, int(grid_size ** (1.0 / d)))
        while gpd ** d > grid_size and gpd > 2:
            gpd -= 1
        self.grid_points_per_dim = gpd
        actual_grid = gpd ** d
        if actual_grid > grid_size:
            raise ValueError(
                f"Cannot build product grid: {gpd}^{d}={actual_grid} > {grid_size}. "
                "Use a larger grid_size or lower embedding_dim."
            )

        if self.dtype == torch.float32 and actual_grid > 8192:
            logger.warning(
                "SKI grid has %d points; exact kernel evaluation on the grid may be slow. "
                "Consider reducing grid_size for embedding_dim=%d.",
                actual_grid, d,
            )

        self._build_grid(grid_bounds)
        self.shape = (self.n, self.n)

    def _build_grid(self, grid_bounds: Optional[torch.Tensor]) -> None:
        """Construct product grid and interpolation weights."""
        d = self.embedding_dim
        gpd = self.grid_points_per_dim
        n = self.n

        if grid_bounds is None:
            mins = self.embeddings.min(dim=0).values
            maxs = self.embeddings.max(dim=0).values
            # Add small padding
            pad = (maxs - mins) * 0.05 + 1e-6
            mins = mins - pad
            maxs = maxs + pad
        else:
            mins = grid_bounds[:, 0]
            maxs = grid_bounds[:, 1]

        # 1-D grids per dimension
        grid_1d = [
            torch.linspace(mins[i].item(), maxs[i].item(), gpd, device=self.device, dtype=self.dtype)
            for i in range(d)
        ]
        self.grid_1d = grid_1d

        # Normalise embeddings to [0, 1] for interpolation
        norm_embed = torch.zeros_like(self.embeddings)
        for i in range(d):
            denom = maxs[i] - mins[i]
            denom = denom if denom > 0 else 1.0
            norm_embed[:, i] = (self.embeddings[:, i] - mins[i]) / denom

        # Normalise grid to [0, 1] as well (so grid_1d_norm[i][j] = j/(gpd-1))
        grid_1d_norm = [
            torch.linspace(0.0, 1.0, gpd, device=self.device, dtype=self.dtype)
            for _ in range(d)
        ]

        indices, weights = _multilinear_weights(norm_embed, grid_1d_norm)
        self.interp_indices = indices  # (n, vertices)
        self.interp_weights = weights  # (n, vertices)

        # Evaluate exact kernel on grid points
        grid_coords = torch.stack([
            g.to(self.dtype) for g in grid_1d
        ], dim=1)  # (gpd, d) - wait, this is wrong for product grid

        # Build full product grid coordinates
        mesh = torch.meshgrid(*grid_1d, indexing="ij")
        grid_points = torch.stack([m.flatten() for m in mesh], dim=1).to(self.dtype)  # (actual_grid, d)
        self.grid_points = grid_points

        gram_grid = grid_points @ grid_points.T
        self.k_grid = _exp_safe(gram_grid)  # (actual_grid, actual_grid)

        # Precompute W @ K_grid for fast matvec: W @ K_grid @ (W^T @ x)
        # W is sparse n x actual_grid; we store it as indices + weights
        # For matvec we compute v = W^T @ x, then u = K_grid @ v, then W @ u
        # W^T @ x can be done with index_add

    def _wt_x(self, x: torch.Tensor) -> torch.Tensor:
        """Compute W^T @ x efficiently via index_add."""
        g = self.grid_points.shape[0]
        out = torch.zeros(g, *x.shape[1:], device=self.device, dtype=self.dtype)
        # For each data point i, distribute x[i] * weight[i,j] to grid point interp_indices[i,j]
        n = x.shape[0]
        vertices = self.interp_indices.shape[1]
        for v in range(vertices):
            idx = self.interp_indices[:, v]
            w = self.interp_weights[:, v]
            if x.dim() == 1:
                out.index_add_(0, idx, w * x)
            else:
                out.index_add_(0, idx, w.unsqueeze(1) * x)
        return out

    def _w_u(self, u: torch.Tensor) -> torch.Tensor:
        """Compute W @ u via gathering."""
        # u is (actual_grid,) or (actual_grid, k)
        # result[i] = sum_j weight[i,j] * u[indices[i,j]]
        gathered = u[self.interp_indices]  # (n, vertices, [k])
        if u.dim() == 1:
            return (gathered * self.interp_weights).sum(dim=1)
        else:
            return (gathered * self.interp_weights.unsqueeze(-1)).sum(dim=1)

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``(lambda I + G_approx)`` to vector(s)."""
        out = self.lambda_reg * x
        v = self._wt_x(x)
        u = self.k_grid @ v
        out = out + self._w_u(u)
        return out

    def diagonal(self) -> torch.Tensor:
        """Return diagonal of ``lambda I + G_approx``."""
        # Diagonal approx = sum_j (W_{ij})^2 * K_grid[j,j]
        k_diag = self.k_grid.diagonal()
        gathered = k_diag[self.interp_indices]  # (n, vertices)
        diag_approx = (gathered * (self.interp_weights ** 2)).sum(dim=1)
        return self.lambda_reg + diag_approx

    def to_dense(self) -> torch.Tensor:
        """Materialise full approximate kernel matrix."""
        n = self.n
        g = self.grid_points.shape[0]
        w_dense = torch.zeros(n, g, device=self.device, dtype=self.dtype)
        vertices = self.interp_indices.shape[1]
        for v in range(vertices):
            w_dense.index_add_(
                0,
                torch.arange(n, device=self.device),
                self.interp_weights[:, v].unsqueeze(1)
                * torch.eye(n, device=self.device, dtype=self.dtype)[self.interp_indices[:, v]],
            )
        k_approx = w_dense @ self.k_grid @ w_dense.T
        k_approx.diagonal().add_(self.lambda_reg)
        return k_approx

    def _build_interp_matrix(
        self, points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return interpolation indices and weights for arbitrary points."""
        d = self.embedding_dim
        gpd = self.grid_points_per_dim
        grid_1d_norm = [
            torch.linspace(0.0, 1.0, gpd, device=self.device, dtype=self.dtype)
            for _ in range(d)
        ]
        mins = torch.stack([g[0] for g in self.grid_1d])
        maxs = torch.stack([g[-1] for g in self.grid_1d])
        denom = maxs - mins
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        norm_pts = (points - mins) / denom
        return _multilinear_weights(norm_pts, grid_1d_norm)

    def kernel_eval(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Evaluate approximate kernel between queries and training points.

        For SKI this builds interpolation weights for ``x`` and returns
        ``W_x @ K_grid @ W_y^T``.
        """
        if y is None:
            y = self.embeddings
        m = x.shape[0]
        p = y.shape[0]

        idx_x, w_x = self._build_interp_matrix(x)
        idx_y, w_y = self._build_interp_matrix(y)

        g = self.grid_points.shape[0]
        # Compute v = K_grid @ W_y^T  -- shape (g, p)
        v = torch.zeros(g, p, device=self.device, dtype=self.dtype)
        vertices = idx_y.shape[1]
        for vt in range(vertices):
            idx = idx_y[:, vt]  # (p,)
            w = w_y[:, vt]      # (p,)
            # v[:, j] += K_grid[:, idx[j]] * w[j]
            v += self.k_grid[:, idx] * w.unsqueeze(0)

        # result = W_x @ v via gather
        gathered = v[idx_x]  # (m, vertices, p)
        out = (gathered * w_x.unsqueeze(-1)).sum(dim=1)  # (m, p)
        return out
