"""Sparse k-NN attention kernel for memory-bounded large-scale regression."""

import logging
from typing import Optional

import torch

from laker.kernels import _exp_safe

logger = logging.getLogger(__name__)


class SparseKNNAttentionKernelOperator:
    """Sparse k-NN approximation of the exponential attention kernel.

    For each point, retains only the ``k`` largest kernel values (nearest
    neighbours in inner-product space).  This reduces storage from ``O(n^2)``
    to ``O(n*k)`` and matvec cost from ``O(n^2)`` to ``O(n*k)``.

    Args:
        embeddings: Tensor of shape ``(n, embedding_dim)``.
        lambda_reg: Tikhonov regularisation ``lambda``.
        k_neighbors: Number of neighbours to retain.  If ``None``, defaults
            to ``min(50, n)``.
        chunk_size: Chunk size for distance computations.
        device: torch device.
        dtype: torch dtype.

    Raises:
        ValueError: If ``embeddings`` is not 2-D.
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        lambda_reg: float = 1e-2,
        k_neighbors: Optional[int] = None,
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

        k = k_neighbors if k_neighbors is not None else min(50, self.n)
        self.k_neighbors = min(k, self.n)

        # Precompute sparse k-NN structure
        self._build_sparse_knn()
        self.shape = (self.n, self.n)

    def _build_sparse_knn(self) -> None:
        """Compute top-k Euclidean neighbours, symmetrise, and store as COO."""
        n = self.n
        k = self.k_neighbors
        cs = self.chunk_size or n

        row_list = []
        col_list = []
        val_list = []

        for i_start in range(0, n, cs):
            i_end = min(i_start + cs, n)
            # Euclidean distance in embedding space; self is always distance 0
            dists = torch.cdist(self.embeddings[i_start:i_end], self.embeddings)  # (cs, n)
            # k-1 nearest neighbours + self (distance 0)
            topk = torch.topk(dists, min(k, n), largest=False, dim=1)
            row_idx = torch.arange(i_start, i_end, device=self.device).unsqueeze(1).expand(-1, min(k, n))
            row_list.append(row_idx.flatten())
            col_list.append(topk.indices.flatten())
            # Compute exact kernel values for the selected pairs
            gram_vals = torch.sum(
                self.embeddings[i_start:i_end].unsqueeze(1) * self.embeddings[topk.indices],
                dim=2,
            ).flatten()
            val_list.append(_exp_safe(gram_vals))

        rows = torch.cat(row_list)
        cols = torch.cat(col_list)
        vals = torch.cat(val_list)

        # Symmetrise: if (i,j) is an edge, ensure (j,i) is also present.
        all_rows = torch.cat([rows, cols])
        all_cols = torch.cat([cols, rows])
        all_vals = torch.cat([vals, vals])

        # Deduplicate by sorting on a composite key
        sort_key = all_rows.to(torch.int64) * (n + 1) + all_cols.to(torch.int64)
        order = torch.argsort(sort_key)
        sorted_rows = all_rows[order]
        sorted_cols = all_cols[order]
        sorted_vals = all_vals[order]

        diff = torch.diff(sorted_rows.to(torch.int64) * (n + 1) + sorted_cols.to(torch.int64))
        is_new = torch.cat([torch.tensor([True], device=self.device), diff != 0])

        coo_indices = torch.stack([sorted_rows[is_new], sorted_cols[is_new]], dim=0)
        coo_values = sorted_vals[is_new]

        # Ensure every row has a diagonal entry and enforce strict diagonal
        # dominance so the symmetrised matrix is guaranteed positive definite.
        # When k >= n the matrix is already exact and PSD, so skip this step.
        if k < n:
            diag_mask = coo_indices[0] == coo_indices[1]
            diag_rows = coo_indices[0, diag_mask]
            diag_vals = coo_values[diag_mask]

            off_mask = ~diag_mask
            off_rows = coo_indices[0, off_mask]
            off_vals = coo_values[off_mask]

            row_sums = torch.zeros(n, device=self.device, dtype=self.dtype)
            row_sums.index_add_(0, off_rows, off_vals.abs())

            diag_map = torch.full((n,), float("-inf"), device=self.device, dtype=self.dtype)
            diag_map[diag_rows] = diag_vals

            min_diag = row_sums * 1.01 + 1e-8
            new_diag = torch.maximum(diag_map, min_diag)

            # Update existing diagonals
            coo_values = coo_values.clone()
            coo_values[diag_mask] = new_diag[diag_rows]

            # Add missing diagonal entries as new COO elements
            missing = torch.where(~torch.isfinite(diag_map))[0]
            if missing.numel() > 0:
                missing_vals = min_diag[missing]
                coo_indices = torch.cat([coo_indices, torch.stack([missing, missing], dim=0)], dim=1)
                coo_values = torch.cat([coo_values, missing_vals])

        self.coo_indices = coo_indices
        self.coo_values = coo_values

    def _sparse_mat(self, m: int, n: int) -> torch.Tensor:
        """Build a sparse COO tensor of shape (m, n)."""
        return torch.sparse_coo_tensor(
            self.coo_indices, self.coo_values, (m, n), device=self.device, dtype=self.dtype
        ).coalesce()

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``(lambda I + G_approx)`` to vector(s)."""
        out = self.lambda_reg * x
        sparse_mat = self._sparse_mat(self.n, self.n)
        if x.dim() == 1:
            out = out + torch.sparse.mm(sparse_mat, x.unsqueeze(1)).squeeze(1)
        else:
            out = out + torch.sparse.mm(sparse_mat, x)
        return out

    def diagonal(self) -> torch.Tensor:
        """Return diagonal of ``lambda I + G_approx``."""
        sq_norms = torch.sum(self.embeddings**2, dim=1)
        return self.lambda_reg + torch.exp(sq_norms)

    def to_dense(self) -> torch.Tensor:
        """Materialise full dense matrix (for debugging only)."""
        dense = self._sparse_mat(self.n, self.n).to_dense()
        dense.diagonal().add_(self.lambda_reg)
        return dense

    def kernel_eval(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Evaluate sparse kernel between queries and training points.

        Returns a **sparse** tensor of shape ``(m, p)`` where each row
        contains the top-``k`` kernel values.  This enables memory-efficient
        ``predict()`` via ``torch.sparse.mm``.
        """
        if y is None:
            y = self.embeddings
        m = x.shape[0]
        p = y.shape[0]
        k = min(self.k_neighbors, p)
        cs = chunk_size or m

        row_list = []
        col_list = []
        val_list = []

        for i_start in range(0, m, cs):
            i_end = min(i_start + cs, m)
            gram_chunk = x[i_start:i_end] @ y.T  # (cs, p)
            topk = torch.topk(gram_chunk, k, largest=True, dim=1)
            row_idx = torch.arange(i_start, i_end, device=self.device).unsqueeze(1).expand(-1, k)
            row_list.append(row_idx.flatten())
            col_list.append(topk.indices.flatten())
            val_list.append(_exp_safe(topk.values).flatten())

        rows = torch.cat(row_list)
        cols = torch.cat(col_list)
        vals = torch.cat(val_list)

        return torch.sparse_coo_tensor(
            torch.stack([rows, cols], dim=0),
            vals,
            (m, p),
            device=self.device,
            dtype=self.dtype,
        ).coalesce()
