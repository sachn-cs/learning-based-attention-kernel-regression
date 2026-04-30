"""Multi-GPU distributed attention kernel operator.

Shards embeddings across available CUDA devices and computes matvecs in
parallel.  Falls back gracefully to single-device execution when only one GPU
is available (or none).
"""

import logging
from typing import Optional

import torch

from laker.kernels import AttentionKernelOperator

logger = logging.getLogger(__name__)


class DistributedAttentionKernelOperator:
    """Wrapper that distributes a dense attention kernel across multiple GPUs.

    Embeddings are split evenly among devices.  Each GPU computes its local
    chunk of the matvec; results are gathered back to the master device.

    Args:
        embeddings: Full embeddings of shape ``(n, embedding_dim)``.
        lambda_reg: Tikhonov regularisation.
        master_device: Device where the input/output vectors live.
        dtype: torch dtype.

    Note:
        This is **data parallelism** (shard vectors, replicate full embeddings
        across devices), not model parallelism.  Peak memory per device is
        ``O(n_local * embedding_dim)`` for the local embedding shard plus
        ``O(n)`` for the full vector.
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        lambda_reg: float = 1e-2,
        master_device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if dtype is None:
            dtype = embeddings.dtype

        if master_device is None:
            master_device = embeddings.device

        self.n = embeddings.shape[0]
        self.embedding_dim = embeddings.shape[1]
        self.lambda_reg = float(lambda_reg)
        self.master_device = master_device
        self.dtype = dtype
        self.shape = (self.n, self.n)

        # Detect CUDA devices
        if torch.cuda.is_available():
            self.devices = [
                torch.device(f"cuda:{i}")
                for i in range(torch.cuda.device_count())
            ]
        else:
            self.devices = [master_device]

        if len(self.devices) == 1:
            # Single device: just wrap a normal operator
            self._single_device = True
            self._local_op = AttentionKernelOperator(
                embeddings=embeddings.to(device=master_device, dtype=dtype),
                lambda_reg=lambda_reg,
                chunk_size=None,
                device=master_device,
                dtype=dtype,
            )
            return

        self._single_device = False
        # Shard embeddings across devices
        self._shard_embeddings(embeddings.to(dtype=dtype))

    def _shard_embeddings(self, embeddings: torch.Tensor) -> None:
        """Split embeddings evenly among available devices."""
        n = embeddings.shape[0]
        num_dev = len(self.devices)
        chunk_sizes = [n // num_dev] * num_dev
        for i in range(n % num_dev):
            chunk_sizes[i] += 1

        self._chunk_sizes = chunk_sizes
        self._operators = []
        start = 0
        for dev, cs in zip(self.devices, chunk_sizes):
            end = start + cs
            local_embed = embeddings[start:end].to(device=dev)
            op = AttentionKernelOperator(
                embeddings=local_embed,
                lambda_reg=self.lambda_reg,
                chunk_size=None,
                device=dev,
                dtype=self.dtype,
            )
            self._operators.append(op)
            start = end

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``(lambda I + G)`` to vector(s) ``x``.

        Automatically handles device movement and gathers results back to
        ``master_device``.
        """
        if self._single_device:
            return self._local_op.matvec(x)

        # Gather full embeddings to master, then replicate to each device
        # Each device computes its local output chunk:
        #   out[start:end] = lambda*x[start:end] + exp(local_embed @ full_embed.T) @ x
        x_master = x.to(self.master_device)
        full_embed = torch.cat([
            op.embeddings.to(self.master_device) for op in self._operators
        ], dim=0)

        outputs = []
        start = 0
        for op in self._operators:
            dev = op.embeddings.device
            local_embed = op.embeddings
            ln = local_embed.shape[0]
            end = start + ln

            # Move full vector and full embeddings to device
            x_dev = x_master.to(dev)
            full_dev = full_embed.to(dev)

            # Compute local chunk with 1-D chunking over the reduction dim
            # to keep peak memory bounded
            cs = 8192  # chunk size for gram blocks
            local_out = self.lambda_reg * x_dev
            for j_start in range(0, self.n, cs):
                j_end = min(j_start + cs, self.n)
                gram_block = local_embed @ full_dev[j_start:j_end].T
                torch.exp(gram_block, out=gram_block)
                if x_dev.dim() == 1:
                    local_out[start:end].addmv_(gram_block, x_dev[j_start:j_end])
                else:
                    local_out[start:end].addmm_(gram_block, x_dev[j_start:j_end])

            outputs.append(local_out[start:end].to(self.master_device))
            start = end

        return torch.cat(outputs, dim=0)

    def diagonal(self) -> torch.Tensor:
        """Return diagonal of ``lambda I + G``."""
        if self._single_device:
            return self._local_op.diagonal()
        diags = []
        for op in self._operators:
            diags.append(op.diagonal().to(self.master_device))
        return torch.cat(diags)

    def to_dense(self) -> torch.Tensor:
        """Materialise full dense matrix on ``master_device``."""
        if self._single_device:
            return self._local_op.to_dense()
        # Gather all embeddings to master device
        full_embed = torch.cat([
            op.embeddings.to(self.master_device) for op in self._operators
        ], dim=0)
        gram = full_embed @ full_embed.T
        torch.exp(gram, out=gram)
        gram.diagonal().add_(self.lambda_reg)
        return gram

    def kernel_eval(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Evaluate exact kernel between queries and training points."""
        if self._single_device:
            return self._local_op.kernel_eval(x, y, chunk_size=chunk_size)
        # Gather all training embeddings
        full_y = torch.cat([
            op.embeddings.to(self.master_device) for op in self._operators
        ], dim=0) if y is None else y
        gram = x @ full_y.T
        torch.exp(gram, out=gram)
        return gram
