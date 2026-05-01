"""Reproducible benchmarks for LAKER critical paths.

Run with:
    python benchmarks/reproducible.py

Environment assumptions:
    - CPU: Apple M3 (Darwin)
    - Python 3.13
    - PyTorch 2.x+
    - Fixed torch.manual_seed(42) for deterministic inputs
    - 20 warm-up iterations, 50 measured iterations for matvec
    - Single-trial for preconditioner build and PCG solve (variance is low)
"""

import logging
from typing import Optional

import torch

from benchmarks.executor import BenchmarkExecutor
from laker.embeddings import PositionEmbedding
from laker.kernels import AttentionKernelOperator
from laker.kernels import (
    NystromAttentionKernelOperator,
    RandomFeatureAttentionKernelOperator,
    SKIAttentionKernelOperator,
    SparseKNNAttentionKernelOperator,
)
from laker.models import LAKERRegressor
from laker.preconditioner import CCCPPreconditioner
from laker.solvers import PreconditionedConjugateGradient

logger = logging.getLogger(__name__)


class ReproducibleBenchmarkSuite:
    """Suite of reproducible benchmarks for LAKER critical paths."""

    def __init__(self, executor: Optional[BenchmarkExecutor] = None):
        self.executor = executor if executor is not None else BenchmarkExecutor()
        self.dtype = torch.float32
        self.embedding_dim = 10
        self.lambda_reg = 1e-2

    def warmup(self, kernel, vector, count: int = 20) -> None:
        """Warm up a kernel by running matvec multiple times."""
        for i in range(count):
            kernel.matvec(vector)

    def make_embeddings(self, n: int, dim: int = 10) -> torch.Tensor:
        """Generate realistic embeddings via PositionEmbedding for benchmarking."""
        torch.manual_seed(42)
        x = torch.rand(n, 2, dtype=self.dtype) * 100.0
        embed = PositionEmbedding(input_dim=2, embedding_dim=dim, dtype=self.dtype)
        with torch.no_grad():
            return embed(x)

    def standard_deviation(self, times: list) -> float:
        """Compute standard deviation of a list of times."""
        mean = sum(times) / len(times)
        return (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5

    def kernel_matvec(
        self,
        n: int = 5000,
        chunk_size: Optional[int] = 1024,
        trials: int = 50,
        warmup: int = 20,
    ) -> dict:
        """Benchmark attention kernel matvec performance."""
        embeddings = self.make_embeddings(n, dim=self.embedding_dim)
        vector = torch.randn(n, dtype=self.dtype)
        kernel = AttentionKernelOperator(
            embeddings,
            lambda_reg=self.lambda_reg,
            chunk_size=chunk_size,
            dtype=self.dtype,
        )
        self.warmup(kernel, vector, warmup)

        result = self.executor.run(
            f"kernel_matvec_n{n}",
            lambda: kernel.matvec(vector),
            trials=trials,
            warmup=0,
        )
        return {
            "n": n,
            "chunk_size": chunk_size,
            "matvec_ms_mean": result["mean_ms"],
            "matvec_ms_std": result["std_ms"],
        }

    def preconditioner_build(self, n: int = 5000, num_probes: int = 100) -> dict:
        """Benchmark CCCP preconditioner build time."""
        embeddings = self.make_embeddings(n, dim=self.embedding_dim)
        kernel = AttentionKernelOperator(embeddings, lambda_reg=self.lambda_reg, dtype=self.dtype)
        preconditioner = CCCPPreconditioner(
            num_probes=num_probes,
            gamma=1e-1,
            max_iter=20,
            tol=1e-4,
            verbose=False,
            dtype=self.dtype,
        )

        result = self.executor.run_once(
            f"preconditioner_build_n{n}",
            lambda: preconditioner.build(kernel.matvec, n),
        )
        return {
            "n": n,
            "num_probes": num_probes,
            "build_ms": result["mean_ms"],
        }

    def pcg_solve(self, n: int = 5000, num_probes: int = 100) -> dict:
        """Benchmark PCG solve time."""
        embeddings = self.make_embeddings(n, dim=self.embedding_dim)
        kernel = AttentionKernelOperator(embeddings, lambda_reg=self.lambda_reg, dtype=self.dtype)
        rhs = torch.randn(n, dtype=self.dtype)

        preconditioner = CCCPPreconditioner(
            num_probes=num_probes,
            gamma=1e-1,
            max_iter=20,
            tol=1e-4,
            verbose=False,
            dtype=self.dtype,
        )
        preconditioner.build(kernel.matvec, n)

        pcg = PreconditionedConjugateGradient(tol=1e-10, max_iter=1000, verbose=False)

        result = self.executor.run_once(
            f"pcg_solve_n{n}",
            lambda: pcg.solve(kernel.matvec, preconditioner.apply, rhs),
        )
        return {
            "n": n,
            "num_probes": num_probes,
            "solve_ms": result["mean_ms"],
            "pcg_iters": pcg.iterations,
        }

    def full_fit(self, n: int = 500) -> dict:
        """Benchmark full LAKERRegressor fit time."""
        torch.manual_seed(42)
        x_train = torch.rand(n, 2, dtype=self.dtype) * 100.0
        y_train = torch.randn(n, dtype=self.dtype)

        model = LAKERRegressor(
            embedding_dim=self.embedding_dim,
            lambda_reg=self.lambda_reg,
            gamma=1e-1,
            num_probes=50,
            cccp_max_iter=20,
            cccp_tol=1e-4,
            pcg_tol=1e-10,
            pcg_max_iter=1000,
            verbose=False,
            dtype=self.dtype,
        )

        result = self.executor.run_once(
            f"full_fit_n{n}",
            lambda: model.fit(x_train, y_train),
        )
        return {
            "n": n,
            "fit_ms": result["mean_ms"],
            "pcg_iters": getattr(model, "pcg_iterations_", None),
        }

    def approx_kernel_matvec(self, n: int = 2000, trials: int = 20) -> dict:
        """Benchmark matvec for all kernel approximations."""
        embeddings = self.make_embeddings(n, dim=self.embedding_dim)
        vector = torch.randn(n, dtype=self.dtype)

        results = {}

        # Exact
        operator = AttentionKernelOperator(embeddings, lambda_reg=self.lambda_reg, dtype=self.dtype)
        result = self.executor.run(
            "exact_matvec",
            lambda: operator.matvec(vector),
            trials=trials,
            warmup=0,
        )
        results["exact"] = {"mean": result["mean_ms"], "std": result["std_ms"]}

        # Nyström
        operator = NystromAttentionKernelOperator(
            embeddings, lambda_reg=self.lambda_reg, num_landmarks=200, dtype=self.dtype
        )
        result = self.executor.run(
            "nystrom_matvec",
            lambda: operator.matvec(vector),
            trials=trials,
            warmup=0,
        )
        results["nystrom"] = {"mean": result["mean_ms"], "std": result["std_ms"]}

        # RFF
        operator = RandomFeatureAttentionKernelOperator(
            embeddings, lambda_reg=self.lambda_reg, num_features=400, dtype=self.dtype
        )
        result = self.executor.run(
            "rff_matvec",
            lambda: operator.matvec(vector),
            trials=trials,
            warmup=0,
        )
        results["rff"] = {"mean": result["mean_ms"], "std": result["std_ms"]}

        # k-NN
        operator = SparseKNNAttentionKernelOperator(
            embeddings, lambda_reg=self.lambda_reg, k_neighbors=50, dtype=self.dtype
        )
        result = self.executor.run(
            "knn_matvec",
            lambda: operator.matvec(vector),
            trials=trials,
            warmup=0,
        )
        results["knn"] = {"mean": result["mean_ms"], "std": result["std_ms"]}

        # SKI
        operator = SKIAttentionKernelOperator(
            embeddings, lambda_reg=self.lambda_reg, grid_size=1024, dtype=self.dtype
        )
        result = self.executor.run(
            "ski_matvec",
            lambda: operator.matvec(vector),
            trials=trials,
            warmup=0,
        )
        results["ski"] = {"mean": result["mean_ms"], "std": result["std_ms"]}

        return {"n": n, **results}

    def generate_report(self) -> str:
        """Run all benchmarks and return a markdown report."""
        lines = [
            "# LAKER Benchmark Results",
            "",
            "**Date:** 2026-04-30  ",
            "**Platform:** Darwin (macOS)  ",
            "**PyTorch:** " + torch.__version__ + "  ",
            "**Dtype:** float32 (default)  ",
            "**Seed:** 42  ",
            "",
            "---",
            "",
        ]

        # Kernel matvec
        lines.append("## Kernel Matvec")
        lines.append("")
        lines.append("| n | chunk_size | mean (ms) | std (ms) |")
        lines.append("|---|------------|-----------|----------|")
        for n in [1000, 2000, 5000]:
            result = self.kernel_matvec(n=n, chunk_size=1024, trials=50, warmup=20)
            lines.append(
                f"| {result['n']} | {result['chunk_size']} | "
                f"{result['matvec_ms_mean']:.3f} | {result['matvec_ms_std']:.3f} |"
            )
        lines.append("")

        # Approximation matvec comparison
        lines.append("## Approximation Matvec Comparison (n=2000)")
        lines.append("")
        lines.append("| method | mean (ms) | std (ms) |")
        lines.append("|--------|-----------|----------|")
        result = self.approx_kernel_matvec(n=2000, trials=20)
        for method in ["exact", "nystrom", "rff", "knn", "ski"]:
            lines.append(
                f"| {method} | {result[method]['mean']:.3f} | {result[method]['std']:.3f} |"
            )
        lines.append("")

        # Preconditioner build
        lines.append("## Preconditioner Build")
        lines.append("")
        lines.append("| n | N_r | time (ms) |")
        lines.append("|---|-----|-----------|")
        for n in [1000, 2000, 5000]:
            result = self.preconditioner_build(n=n, num_probes=100)
            lines.append(f"| {result['n']} | {result['num_probes']} | {result['build_ms']:.2f} |")
        lines.append("")

        # PCG solve
        lines.append("## PCG Solve")
        lines.append("")
        lines.append("| n | N_r | time (ms) | iters |")
        lines.append("|---|-----|-----------|-------|")
        for n in [1000, 2000, 5000]:
            result = self.pcg_solve(n=n, num_probes=100)
            lines.append(
                f"| {result['n']} | {result['num_probes']} | "
                f"{result['solve_ms']:.2f} | {result['pcg_iters']} |"
            )
        lines.append("")

        # Full fit
        lines.append("## Full Fit")
        lines.append("")
        lines.append("| n | time (ms) | PCG iters |")
        lines.append("|---|-----------|-----------|")
        for n in [200, 500, 1000]:
            result = self.full_fit(n=n)
            lines.append(f"| {result['n']} | {result['fit_ms']:.2f} | {result['pcg_iters']} |")
        lines.append("")

        return "\n".join(lines)

    def save_report(self, path: str = "benchmarks/README.md") -> None:
        """Generate and save the benchmark report."""
        report = self.generate_report()
        with open(path, "w") as file:
            file.write(report)
        logger.info("Report saved to %s", path)

    def run_all(self) -> None:
        """Run all benchmarks, print report, and save to file."""
        report = self.generate_report()
        logger.info("\n%s", report)
        self.save_report()

    @classmethod
    def run_all_default(cls) -> str:
        """Run all reproducible benchmarks and return the markdown report.

        Returns:
            Markdown formatted string with all benchmark results.
        """
        suite = cls()
        return suite.generate_report()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    suite = ReproducibleBenchmarkSuite()
    suite.run_all()
