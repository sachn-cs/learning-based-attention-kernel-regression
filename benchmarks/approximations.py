"""Compare exact vs low-rank kernel approximations."""

import logging
from typing import Optional

import torch

from benchmarks.executor import BenchmarkExecutor
from laker.kernels import AttentionKernelOperator
from laker.kernels import (
    NystromAttentionKernelOperator,
    RandomFeatureAttentionKernelOperator,
)
from laker.models import LAKERRegressor

logger = logging.getLogger(__name__)


class ApproximationBenchmarkSuite:
    """Suite of benchmarks comparing exact and approximate kernel operators."""

    def __init__(self, executor: Optional[BenchmarkExecutor] = None):
        self.executor = executor if executor is not None else BenchmarkExecutor()
        self.dtype = torch.float64
        self.lambda_reg = 1e-2

    @classmethod
    def benchmark_kernel_speed(cls, n: int = 2000, dim: int = 10) -> dict:
        """Benchmark matvec speed for exact and approximate kernels.

        Args:
            n: Problem size.
            dim: Embedding dimension.

        Returns:
            Dictionary with timing and error results.
        """
        suite = cls()
        return suite.kernel_speed(n, dim)

    @classmethod
    def benchmark_full_fit(cls, n: int = 500) -> dict:
        """Benchmark full model fit with exact and approximate kernels.

        Args:
            n: Number of training samples.

        Returns:
            Dictionary with timing results per approximation.
        """
        suite = cls()
        return suite.full_fit(n)

    def kernel_speed(self, n: int = 2000, dim: int = 10) -> dict:
        """Benchmark matvec speed for exact and approximate kernels."""
        torch.manual_seed(42)
        embeddings = torch.randn(n, dim, dtype=self.dtype)
        vector = torch.randn(n, dtype=self.dtype)

        # Exact
        exact = AttentionKernelOperator(embeddings, lambda_reg=self.lambda_reg, dtype=self.dtype)
        exact_result = self.executor.run_repeated(
            "exact_matvec",
            lambda: exact.matvec(vector),
            repetitions=20,
        )
        exact_ms = exact_result["mean_ms"]

        # Nyström
        nystrom = NystromAttentionKernelOperator(
            embeddings,
            lambda_reg=self.lambda_reg,
            num_landmarks=200,
            dtype=self.dtype,
        )
        nystrom_result = self.executor.run_repeated(
            "nystrom_matvec",
            lambda: nystrom.matvec(vector),
            repetitions=20,
        )
        nystrom_ms = nystrom_result["mean_ms"]

        # RFF
        rff = RandomFeatureAttentionKernelOperator(
            embeddings,
            lambda_reg=self.lambda_reg,
            num_features=400,
            dtype=self.dtype,
        )
        rff_result = self.executor.run_repeated(
            "rff_matvec", lambda: rff.matvec(vector), repetitions=20
        )
        rff_ms = rff_result["mean_ms"]

        # Approx error
        y_exact = exact.matvec(vector)
        y_nystrom = nystrom.matvec(vector)
        y_rff = rff.matvec(vector)
        nystrom_error = torch.norm(y_exact - y_nystrom) / torch.norm(y_exact)
        rff_error = torch.norm(y_exact - y_rff) / torch.norm(y_exact)

        logger.info("n=%d", n)
        logger.info("  Exact matvec:   %.3f ms", exact_ms)
        logger.info("  Nyström matvec: %.3f ms  (rel_err=%.3f)", nystrom_ms, nystrom_error)
        logger.info("  RFF matvec:     %.3f ms  (rel_err=%.3f)", rff_ms, rff_error)

        return {
            "n": n,
            "exact_ms": exact_ms,
            "nystrom_ms": nystrom_ms,
            "rff_ms": rff_ms,
            "nystrom_error": nystrom_error.item(),
            "rff_error": rff_error.item(),
        }

    def full_fit(self, n: int = 500) -> dict:
        """Benchmark full model fit with exact and approximate kernels."""
        torch.manual_seed(42)
        x = torch.rand(n, 2, dtype=self.dtype) * 100.0
        y = torch.randn(n, dtype=self.dtype)

        results = {}
        for label, approx in [("exact", None), ("nystrom", "nystrom"), ("rff", "rff")]:
            model = LAKERRegressor(
                embedding_dim=10,
                lambda_reg=self.lambda_reg,
                gamma=1e-1,
                num_probes=50,
                cccp_max_iter=20,
                cccp_tol=1e-4,
                pcg_tol=1e-6,
                pcg_max_iter=500,
                kernel_approx=approx,
                num_landmarks=100 if approx == "nystrom" else None,
                num_features=200 if approx == "rff" else None,
                dtype=self.dtype,
                verbose=False,
            )
            result = self.executor.run_once(
                f"{label}_fit_n{n}",
                lambda: model.fit(x, y),
            )
            fit_ms = result["mean_ms"]
            logger.info(
                "  %8s: %7.2f ms  pcg_iters=%s",
                label,
                fit_ms,
                model.pcg_iterations_,
            )
            results[label] = {"fit_ms": fit_ms, "pcg_iters": model.pcg_iterations_}

        return {"n": n, **results}

    def run_all(self) -> None:
        """Run all approximation benchmarks."""
        logger.info("=== Kernel Matvec Speed ===")
        self.kernel_speed(n=2000)
        self.kernel_speed(n=5000)

        logger.info("=== Full Fit Speed ===")
        self.full_fit(n=500)
        self.full_fit(n=1000)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    suite = ApproximationBenchmarkSuite()
    suite.run_all()
