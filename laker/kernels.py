"""Attention kernel operators for matrix-free linear algebra."""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def _exp_safe(
    gram: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    skip_clamp: bool = False,
) -> torch.Tensor:
    """Element-wise exp with dtype-aware overflow guard.

    Clamps to ``80.0`` for float32 and ``700.0`` for float64 before
    exponentiation to prevent silent overflow to ``inf``.

    When ``gram.requires_grad`` is ``True`` (e.g. during learned-embedding
    training) the ``out=`` form is skipped because PyTorch does not support
    autodiff through in-place ``torch.exp``.

    If ``skip_clamp`` is ``True`` the clamp step is bypassed.  Callers that
    pre-verify their data never exceeds the safe threshold (e.g. via
    ``max_sq_norm``) should set this to recover the original speed.
    """
    if not skip_clamp:
        max_val = 80.0 if gram.dtype == torch.float32 else 700.0
        if gram.requires_grad:
            return torch.exp(gram.clamp(max=max_val))
        if out is None:
            out = gram
        if out is not gram:
            out.copy_(gram)
        out.clamp_(max=max_val)
        return torch.exp(out)
    # Fast path: no clamp needed
    if out is None:
        return torch.exp(gram)
    return torch.exp(gram, out=out)


def _dense_attention_matvec(
    embeddings: torch.Tensor, lambda_reg: float, x: torch.Tensor
) -> torch.Tensor:
    """Helper for the non-chunked attention matvec."""
    gram = embeddings @ embeddings.T
    _exp_safe(gram, out=gram)
    return lambda_reg * x + gram @ x


class AttentionKernelOperator:
    """Matrix-free operator for the exponential attention kernel ``G = exp(E E^T)``.

    The exponential is applied **element-wise**. For large ``n`` the dense matrix
    is never materialised; instead matvecs are computed in optionally chunked
    blocks to respect GPU memory limits.

    Args:
        embeddings: Tensor of shape ``(n, embedding_dim)``.
        lambda_reg: Tikhonov regularisation ``lambda`` added to the diagonal.
        chunk_size: If ``None``, the full kernel block is formed for matvecs.
            If an integer, chunked evaluation is used so that peak memory is
            ``O(chunk_size * n)``.
        device: torch device (inferred from ``embeddings`` if omitted).
        dtype: torch dtype (inferred from ``embeddings`` if omitted).

    Raises:
        ValueError: If ``embeddings`` is not 2-D.
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        lambda_reg: float = 1e-2,
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

        self.shape = (self.n, self.n)
        self.lambda_vec = None  # cached diagonal scaling for Jacobi preconditioning

        # Pre-compute whether gram values can ever overflow; if not we skip the
        # clamp in ``_exp_safe`` and recover the original single-kernel speed.
        max_sq_norm = torch.sum(self.embeddings ** 2, dim=1).max().item()
        safe_limit = 80.0 if self.dtype == torch.float32 else 700.0
        self._skip_clamp = max_sq_norm < safe_limit

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``(lambda I + G)`` to vector(s) ``x``.

        Automatically dispatches to the 1-D or 2-D implementation based on
        the dimensionality of ``x``.

        Args:
            x: Tensor of shape ``(n,)`` or ``(n, k)``.

        Returns:
            Tensor of the same shape as ``x``.

        Raises:
            ValueError: If ``x`` is not 1-D or 2-D.
        """
        if x.dim() == 1:
            return self._matvec_impl(x)
        if x.dim() == 2:
            return self._matvec_impl(x)
        raise ValueError(f"x must be 1-D or 2-D, got shape {x.shape}")

    def _matvec_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Shared implementation for 1-D and 2-D matvecs.

        Uses 2-D tiling when chunked so that peak memory is
        ``O(chunk_size^2)`` rather than ``O(chunk_size * n)``.

        Args:
            x: Tensor of shape ``(n,)`` or ``(n, k)``.

        Returns:
            Tensor of the same shape as ``x``.
        """
        if self.chunk_size is None or self.n <= self.chunk_size:
            return _dense_attention_matvec(self.embeddings, self.lambda_reg, x)

        out = self.lambda_reg * x

        cs = self.chunk_size
        n = self.n
        # Heuristic: if a single output chunk against all inputs fits comfortably
        # in memory (≤ 64 MB), use fast 1-D chunking; otherwise use 2-D tiling.
        element_size = 4 if self.dtype == torch.float32 else 8
        mem_per_chunk = cs * n * element_size
        if mem_per_chunk <= 64 * 1024 * 1024:
            for start in range(0, n, cs):
                end = min(start + cs, n)
                gram_chunk = self.embeddings[start:end] @ self.embeddings.T
                _exp_safe(gram_chunk, out=gram_chunk, skip_clamp=self._skip_clamp)
                if x.dim() == 1:
                    out[start:end].addmv_(gram_chunk, x)
                else:
                    out[start:end].addmm_(gram_chunk, x)
            return out

        # 2-D tiling: chunk both the output and reduction dimensions
        if x.dim() == 1:
            for i_start in range(0, n, cs):
                i_end = min(i_start + cs, n)
                accum = torch.zeros(i_end - i_start, device=self.device, dtype=self.dtype)
                e_i = self.embeddings[i_start:i_end]
                for j_start in range(0, n, cs):
                    j_end = min(j_start + cs, n)
                    gram_block = e_i @ self.embeddings[j_start:j_end].T
                    _exp_safe(gram_block, out=gram_block, skip_clamp=self._skip_clamp)
                    accum.addmv_(gram_block, x[j_start:j_end])
                out[i_start:i_end].add_(accum)
        else:
            k = x.shape[1]
            for i_start in range(0, n, cs):
                i_end = min(i_start + cs, n)
                accum = torch.zeros(i_end - i_start, k, device=self.device, dtype=self.dtype)
                e_i = self.embeddings[i_start:i_end]
                for j_start in range(0, n, cs):
                    j_end = min(j_start + cs, n)
                    gram_block = e_i @ self.embeddings[j_start:j_end].T
                    _exp_safe(gram_block, out=gram_block, skip_clamp=self._skip_clamp)
                    accum.addmm_(gram_block, x[j_start:j_end])
                out[i_start:i_end].add_(accum)
        return out

    def matvec_1d(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``(lambda I + G)`` to a single vector.

        Args:
            x: Tensor of shape ``(n,)``.

        Returns:
            Tensor of shape ``(n,)``.
        """
        return self._matvec_impl(x)

    def matvec_2d(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``(lambda I + G)`` to a batch of vectors.

        Args:
            x: Tensor of shape ``(n, k)``.

        Returns:
            Tensor of shape ``(n, k)``.
        """
        return self._matvec_impl(x)

    def diagonal(self) -> torch.Tensor:
        """Return the diagonal of ``lambda I + G``.

        Since ``G_{ii} = exp(||e_i||^2)``, the diagonal is
        ``lambda + exp(||e_i||^2)``.

        Returns:
            Tensor of shape ``(n,)``.
        """
        sq_norms = torch.sum(self.embeddings**2, dim=1)
        return self.lambda_reg + torch.exp(sq_norms)

    def to_dense(self) -> torch.Tensor:
        """Materialise the full dense ``(lambda I + G)`` matrix.

        Warning:
            This costs ``O(n^2)`` memory. Use only for small ``n``
            or debugging.

        Returns:
            Dense tensor of shape ``(n, n)``.
        """
        gram = self.embeddings @ self.embeddings.T
        torch.exp(gram, out=gram)
        gram.diagonal().add_(self.lambda_reg)
        return gram

    def kernel_eval(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Evaluate the attention kernel between two sets of points.

        Supports 2-D tiling via ``chunk_size`` so that peak memory stays
        ``O(chunk_size^2)`` even for large query/reference sets.

        Args:
            x: Query embeddings of shape ``(m, embedding_dim)``.
            y: Reference embeddings of shape ``(p, embedding_dim)``. If ``None``,
                uses ``self.embeddings``.
            chunk_size: Tile size. Defaults to ``self.chunk_size``.

        Returns:
            Kernel matrix of shape ``(m, p)`` or ``(m, n)``.
        """
        if y is None:
            y = self.embeddings
        if chunk_size is None:
            chunk_size = self.chunk_size
        m = x.shape[0]
        p = y.shape[0]
        if chunk_size is None or m <= chunk_size:
            gram = x @ y.T
            _exp_safe(gram, out=gram, skip_clamp=self._skip_clamp)
            return gram

        # Use 1-D chunking over the query dimension when memory is moderate,
        # otherwise fall back to full 2-D tiling.
        element_size = 4 if self.dtype == torch.float32 else 8
        mem_per_chunk = chunk_size * p * element_size
        if mem_per_chunk <= 64 * 1024 * 1024:
            out = torch.empty(m, p, device=self.device, dtype=self.dtype)
            for start in range(0, m, chunk_size):
                end = min(start + chunk_size, m)
                gram_chunk = x[start:end] @ y.T
                _exp_safe(gram_chunk, out=gram_chunk, skip_clamp=self._skip_clamp)
                out[start:end] = gram_chunk
            return out

        out = torch.empty(m, p, device=self.device, dtype=self.dtype)
        for i_start in range(0, m, chunk_size):
            i_end = min(i_start + chunk_size, m)
            for j_start in range(0, p, chunk_size):
                j_end = min(j_start + chunk_size, p)
                gram_block = x[i_start:i_end] @ y[j_start:j_end].T
                _exp_safe(gram_block, out=gram_block, skip_clamp=self._skip_clamp)
                out[i_start:i_end, j_start:j_end] = gram_block
        return out
