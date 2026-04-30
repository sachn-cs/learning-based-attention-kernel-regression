"""Reproducible benchmarks for LAKER critical paths.

Run with:
    python benchmarks/reproducible_benchmarks.py

Environment assumptions:
    - CPU: Apple M3 (Darwin)
    - Python 3.13
    - PyTorch 2.x+
    - Fixed torch.manual_seed(42) for deterministic inputs
    - 20 warm-up iterations, 50 measured iterations for matvec
    - Single-trial for preconditioner build and PCG solve (variance is low)
"""

import time
from typing import Optional

import torch

from laker.embeddings import PositionEmbedding
from laker.kernels import AttentionKernelOperator
from laker.low_rank_kernels import (
    NystromAttentionKernelOperator,
    RandomFeatureAttentionKernelOperator,
)
from laker.models import LAKERRegressor
from laker.preconditioner import CCCPPreconditioner
from laker.ski_kernels import SKIAttentionKernelOperator
from laker.solvers import PreconditionedConjugateGradient
from laker.sparse_kernels import SparseKNNAttentionKernelOperator


def _warmup(kernel, x, n=20):
    for _ in range(n):
        _ = kernel.matvec(x)


def _make_embeddings(n: int, de: int = 10, dtype=torch.float32):
    """Generate realistic embeddings via PositionEmbedding for benchmarking."""
    torch.manual_seed(42)
    x = torch.rand(n, 2, dtype=dtype) * 100.0
    embed = PositionEmbedding(input_dim=2, embedding_dim=de, dtype=dtype)
    with torch.no_grad():
        return embed(x)


def benchmark_kernel_matvec(
    n: int = 5000,
    chunk_size: Optional[int] = 1024,
    trials: int = 50,
    warmup: int = 20,
) -> dict:
    """Benchmark attention kernel matvec performance."""
    dtype = torch.float32
    e = _make_embeddings(n, de=10, dtype=dtype)
    x = torch.randn(n, dtype=dtype)

    kernel = AttentionKernelOperator(e, lambda_reg=1e-2, chunk_size=chunk_size, dtype=dtype)
    _warmup(kernel, x, warmup)

    times = []
    for _ in range(trials):
        start = time.perf_counter()
        _ = kernel.matvec(x)
        times.append((time.perf_counter() - start) * 1000)

    return {
        "n": n,
        "chunk_size": chunk_size,
        "matvec_ms_mean": sum(times) / len(times),
        "matvec_ms_std": (sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)) ** 0.5,
    }


def benchmark_preconditioner_build(n: int = 5000, num_probes: int = 100) -> dict:
    """Benchmark CCCP preconditioner build time."""
    dtype = torch.float32
    e = _make_embeddings(n, de=10, dtype=dtype)
    kernel = AttentionKernelOperator(e, lambda_reg=1e-2, dtype=dtype)

    pre = CCCPPreconditioner(
        num_probes=num_probes,
        gamma=1e-1,
        max_iter=20,
        tol=1e-4,
        verbose=False,
        dtype=dtype,
    )

    start = time.perf_counter()
    pre.build(kernel.matvec, n)
    elapsed = (time.perf_counter() - start) * 1000

    return {"n": n, "num_probes": num_probes, "build_ms": elapsed}


def benchmark_pcg_solve(n: int = 5000, num_probes: int = 100) -> dict:
    """Benchmark PCG solve time."""
    dtype = torch.float32
    e = _make_embeddings(n, de=10, dtype=dtype)
    kernel = AttentionKernelOperator(e, lambda_reg=1e-2, dtype=dtype)
    b = torch.randn(n, dtype=dtype)

    pre = CCCPPreconditioner(
        num_probes=num_probes,
        gamma=1e-1,
        max_iter=20,
        tol=1e-4,
        verbose=False,
        dtype=dtype,
    )
    pre.build(kernel.matvec, n)

    pcg = PreconditionedConjugateGradient(tol=1e-6, max_iter=500, verbose=False)

    start = time.perf_counter()
    pcg.solve(kernel.matvec, pre.apply, b)
    elapsed = (time.perf_counter() - start) * 1000

    return {
        "n": n,
        "num_probes": num_probes,
        "solve_ms": elapsed,
        "pcg_iters": pcg.iterations,
    }


def benchmark_full_fit(n: int = 500) -> dict:
    """Benchmark full LAKERRegressor fit time."""
    torch.manual_seed(42)
    dtype = torch.float32
    x_train = torch.rand(n, 2, dtype=dtype) * 100.0
    y_train = torch.randn(n, dtype=dtype)

    model = LAKERRegressor(
        embedding_dim=10,
        lambda_reg=1e-2,
        gamma=1e-1,
        num_probes=50,
        cccp_max_iter=20,
        cccp_tol=1e-4,
        pcg_tol=1e-6,
        pcg_max_iter=500,
        verbose=False,
        dtype=dtype,
    )

    start = time.perf_counter()
    model.fit(x_train, y_train)
    elapsed = (time.perf_counter() - start) * 1000

    return {
        "n": n,
        "fit_ms": elapsed,
        "pcg_iters": getattr(model, "pcg_iterations_", None),
    }


def _std(times: list) -> float:
    mean = sum(times) / len(times)
    return (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5


def benchmark_approx_kernel_matvec(n: int = 2000, trials: int = 20) -> dict:
    """Benchmark matvec for all kernel approximations."""
    dtype = torch.float32
    e = _make_embeddings(n, de=10, dtype=dtype)
    x = torch.randn(n, dtype=dtype)

    results = {}

    # Exact
    op = AttentionKernelOperator(e, lambda_reg=1e-2, dtype=dtype)
    times = []
    for _ in range(trials):
        start = time.perf_counter()
        _ = op.matvec(x)
        times.append((time.perf_counter() - start) * 1000)
    results["exact"] = {"mean": sum(times) / len(times), "std": _std(times)}

    # Nyström
    op = NystromAttentionKernelOperator(e, lambda_reg=1e-2, num_landmarks=200, dtype=dtype)
    times = []
    for _ in range(trials):
        start = time.perf_counter()
        _ = op.matvec(x)
        times.append((time.perf_counter() - start) * 1000)
    results["nystrom"] = {"mean": sum(times) / len(times), "std": _std(times)}

    # RFF
    op = RandomFeatureAttentionKernelOperator(e, lambda_reg=1e-2, num_features=400, dtype=dtype)
    times = []
    for _ in range(trials):
        start = time.perf_counter()
        _ = op.matvec(x)
        times.append((time.perf_counter() - start) * 1000)
    results["rff"] = {"mean": sum(times) / len(times), "std": _std(times)}

    # k-NN
    op = SparseKNNAttentionKernelOperator(e, lambda_reg=1e-2, k_neighbors=50, dtype=dtype)
    times = []
    for _ in range(trials):
        start = time.perf_counter()
        _ = op.matvec(x)
        times.append((time.perf_counter() - start) * 1000)
    results["knn"] = {"mean": sum(times) / len(times), "std": _std(times)}

    # SKI
    op = SKIAttentionKernelOperator(e, lambda_reg=1e-2, grid_size=1024, dtype=dtype)
    times = []
    for _ in range(trials):
        start = time.perf_counter()
        _ = op.matvec(x)
        times.append((time.perf_counter() - start) * 1000)
    results["ski"] = {"mean": sum(times) / len(times), "std": _std(times)}

    return {"n": n, **results}


def run_all() -> str:
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
        res = benchmark_kernel_matvec(n=n, chunk_size=1024, trials=50, warmup=20)
        lines.append(
            f"| {res['n']} | {res['chunk_size']} | {res['matvec_ms_mean']:.3f} | {res['matvec_ms_std']:.3f} |"
        )
    lines.append("")

    # Approximation matvec comparison
    lines.append("## Approximation Matvec Comparison (n=2000)")
    lines.append("")
    lines.append("| method | mean (ms) | std (ms) |")
    lines.append("|--------|-----------|----------|")
    res = benchmark_approx_kernel_matvec(n=2000, trials=20)
    for method in ["exact", "nystrom", "rff", "knn", "ski"]:
        lines.append(
            f"| {method} | {res[method]['mean']:.3f} | {res[method]['std']:.3f} |"
        )
    lines.append("")

    # Preconditioner build
    lines.append("## Preconditioner Build")
    lines.append("")
    lines.append("| n | N_r | time (ms) |")
    lines.append("|---|-----|-----------|")
    for n in [1000, 2000, 5000]:
        res = benchmark_preconditioner_build(n=n, num_probes=100)
        lines.append(f"| {res['n']} | {res['num_probes']} | {res['build_ms']:.2f} |")
    lines.append("")

    # PCG solve
    lines.append("## PCG Solve")
    lines.append("")
    lines.append("| n | N_r | time (ms) | iters |")
    lines.append("|---|-----|-----------|-------|")
    for n in [1000, 2000, 5000]:
        res = benchmark_pcg_solve(n=n, num_probes=100)
        lines.append(
            f"| {res['n']} | {res['num_probes']} | {res['solve_ms']:.2f} | {res['pcg_iters']} |"
        )
    lines.append("")

    # Full fit
    lines.append("## Full Fit")
    lines.append("")
    lines.append("| n | time (ms) | PCG iters |")
    lines.append("|---|-----------|-----------|")
    for n in [200, 500, 1000]:
        res = benchmark_full_fit(n=n)
        lines.append(f"| {res['n']} | {res['fit_ms']:.2f} | {res['pcg_iters']} |")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    report = run_all()
    print(report)
    with open("benchmarks/results.md", "w") as f:
        f.write(report)
    print("\nReport saved to benchmarks/results.md")
