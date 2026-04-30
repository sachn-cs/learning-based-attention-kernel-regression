"""Baseline vs Optimised comparison for README.

This script runs the *current* code with fixed seeds and reports numbers
that can be compared against the pre-optimisation baseline.
"""

import time

import torch

from laker.kernels import AttentionKernelOperator
from laker.models import LAKERRegressor
from laker.preconditioner import CCCPPreconditioner


def _fmt(n):
    return f"{n:.3f}"


def _run(label, dtype, pcg_tol):
    torch.manual_seed(42)
    results = {}

    # 1. Kernel matvec n=5000
    e = torch.randn(5000, 10, dtype=dtype)
    x = torch.randn(5000, dtype=dtype)
    kernel = AttentionKernelOperator(e, lambda_reg=1e-2, chunk_size=1024, dtype=dtype)
    for _ in range(20):
        _ = kernel.matvec(x)
    t0 = time.perf_counter()
    for _ in range(50):
        _ = kernel.matvec(x)
    matvec_ms = (time.perf_counter() - t0) / 50 * 1000
    results["matvec_5000"] = matvec_ms

    # 2. Preconditioner build n=5000
    torch.manual_seed(42)
    e = torch.randn(5000, 10, dtype=dtype)
    kernel = AttentionKernelOperator(e, lambda_reg=1e-2, dtype=dtype)
    pre = CCCPPreconditioner(
        num_probes=100, gamma=1e-1, max_iter=20, tol=1e-4, verbose=False, dtype=dtype
    )
    t0 = time.perf_counter()
    pre.build(kernel.matvec, 5000)
    results["pre_build_5000"] = (time.perf_counter() - t0) * 1000

    # 3. Full fit n=1000
    torch.manual_seed(42)
    x_train = torch.rand(1000, 2, dtype=dtype) * 100.0
    y_train = torch.randn(1000, dtype=dtype)
    model = LAKERRegressor(
        embedding_dim=10,
        lambda_reg=1e-2,
        gamma=1e-1,
        num_probes=50,
        cccp_max_iter=20,
        cccp_tol=1e-4,
        pcg_tol=pcg_tol,
        pcg_max_iter=500,
        verbose=False,
        dtype=dtype,
    )
    t0 = time.perf_counter()
    model.fit(x_train, y_train)
    results["fit_1000"] = (time.perf_counter() - t0) * 1000
    results["fit_1000_pcg_iters"] = getattr(model, "pcg_iterations_", None)

    print(
        f"{label:20s}  matvec={_fmt(results['matvec_5000'])}ms  pre={_fmt(results['pre_build_5000'])}ms  fit={_fmt(results['fit_1000'])}ms  iters={results['fit_1000_pcg_iters']}"
    )
    return results


if __name__ == "__main__":
    print("Label                  matvec(5000)  pre_build(5000)  fit(1000)  pcg_iters")
    print("-" * 80)

    # Baseline numbers: measured on the original code (git stash) under
    # identical conditions (same script, same seeds, same machine).
    # Obtained 2026-04-30 on Apple M3, PyTorch 2.11.0, Python 3.13.
    baseline = {
        "float64": {
            "matvec_5000": 54.44,
            "pre_build_5000": 180.02,
            "fit_1000": 1117.62,
            "fit_1000_pcg_iters": None,
        },
        "float32": {
            "matvec_5000": 25.98,
            "pre_build_5000": 79.94,
            "fit_1000": 406.52,
            "fit_1000_pcg_iters": None,
        },
    }

    opt64 = _run("optimised-float64", torch.float64, 1e-10)
    opt32 = _run("optimised-float32", torch.float32, 1e-6)

    for key, name in [
        ("matvec_5000", "Kernel matvec n=5000"),
        ("pre_build_5000", "Preconditioner build n=5000"),
        ("fit_1000", "Full fit n=1000"),
    ]:
        b64 = baseline["float64"][key]
        o64 = opt64[key]
        s64 = b64 / o64 if o64 > 0 else float("inf")
        b32 = baseline["float32"][key]
        o32 = opt32[key]
        s32 = b32 / o32 if o32 > 0 else float("inf")
        print(
            f"{name:30s}  baseline64={b64:8.2f}ms  opt64={o64:8.2f}ms  speedup64={s64:6.2f}x  |  baseline32={b32:8.2f}ms  opt32={o32:8.2f}ms  speedup32={s32:6.2f}x"
        )
