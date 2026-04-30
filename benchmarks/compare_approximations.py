"""Compare exact vs low-rank kernel approximations."""

import time

import torch

from laker.kernels import AttentionKernelOperator
from laker.low_rank_kernels import (
    NystromAttentionKernelOperator,
    RandomFeatureAttentionKernelOperator,
)
from laker.models import LAKERRegressor


def benchmark_kernel_speed(n=2000, de=10):
    torch.manual_seed(42)
    dtype = torch.float64
    e = torch.randn(n, de, dtype=dtype)
    x = torch.randn(n, dtype=dtype)
    lam = 1e-2

    # Exact
    exact = AttentionKernelOperator(e, lambda_reg=lam, dtype=dtype)
    t0 = time.perf_counter()
    for _ in range(20):
        _ = exact.matvec(x)
    exact_ms = (time.perf_counter() - t0) / 20 * 1000

    # Nyström
    nys = NystromAttentionKernelOperator(e, lambda_reg=lam, num_landmarks=200, dtype=dtype)
    t0 = time.perf_counter()
    for _ in range(20):
        _ = nys.matvec(x)
    nys_ms = (time.perf_counter() - t0) / 20 * 1000

    # RFF
    rff = RandomFeatureAttentionKernelOperator(e, lambda_reg=lam, num_features=400, dtype=dtype)
    t0 = time.perf_counter()
    for _ in range(20):
        _ = rff.matvec(x)
    rff_ms = (time.perf_counter() - t0) / 20 * 1000

    # Approx error
    y_exact = exact.matvec(x)
    y_nys = nys.matvec(x)
    y_rff = rff.matvec(x)
    nys_err = torch.norm(y_exact - y_nys) / torch.norm(y_exact)
    rff_err = torch.norm(y_exact - y_rff) / torch.norm(y_exact)

    print(f"n={n}")
    print(f"  Exact matvec:  {exact_ms:.3f} ms")
    print(f"  Nyström matvec: {nys_ms:.3f} ms  (rel_err={nys_err:.3f})")
    print(f"  RFF matvec:    {rff_ms:.3f} ms  (rel_err={rff_err:.3f})")
    print()


def benchmark_full_fit(n=500):
    torch.manual_seed(42)
    dtype = torch.float64
    x = torch.rand(n, 2, dtype=dtype) * 100.0
    y = torch.randn(n, dtype=dtype)

    for label, approx in [("exact", None), ("nystrom", "nystrom"), ("rff", "rff")]:
        model = LAKERRegressor(
            embedding_dim=10,
            lambda_reg=1e-2,
            gamma=1e-1,
            num_probes=50,
            cccp_max_iter=20,
            cccp_tol=1e-4,
            pcg_tol=1e-6,
            pcg_max_iter=500,
            kernel_approx=approx,
            num_landmarks=100 if approx == "nystrom" else None,
            num_features=200 if approx == "rff" else None,
            dtype=dtype,
            verbose=False,
        )
        t0 = time.perf_counter()
        model.fit(x, y)
        fit_ms = (time.perf_counter() - t0) * 1000
        print(f"  {label:8s}: {fit_ms:7.2f} ms  pcg_iters={model.pcg_iterations_}")
    print()


if __name__ == "__main__":
    print("=== Kernel Matvec Speed ===")
    benchmark_kernel_speed(n=2000)
    benchmark_kernel_speed(n=5000)

    print("=== Full Fit Speed ===")
    benchmark_full_fit(n=500)
    benchmark_full_fit(n=1000)
