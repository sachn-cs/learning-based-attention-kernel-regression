"""Tests for laker.benchmark module."""

import torch

from laker.benchmark import BenchmarkResult, benchmark_laker_vs_baselines, benchmark_solver
from laker.kernels import AttentionKernelOperator
from laker.preconditioner import CCCPPreconditioner


def test_benchmark_result_creation():
    """BenchmarkResult dataclass should accept all fields."""
    result = BenchmarkResult(
        name="test",
        n=100,
        solve_time_seconds=0.1,
        iterations=10,
        final_residual=1e-5,
        condition_number=10.0,
        objective_gap=1e-3,
    )
    assert result.name == "test"
    assert result.n == 100
    assert result.condition_number == 10.0
    assert result.objective_gap == 1e-3


def test_benchmark_solver_with_preconditioner():
    """benchmark_solver with a preconditioner should return valid results."""
    torch.manual_seed(42)
    n = 50
    e = torch.randn(n, 8, dtype=torch.float64)
    op = AttentionKernelOperator(e, lambda_reg=1e-2, dtype=torch.float64)
    rhs = torch.randn(n, dtype=torch.float64)

    pre = CCCPPreconditioner(
        num_probes=30, gamma=1e-1, max_iter=20, tol=1e-4, verbose=False, dtype=torch.float64
    )
    pre.build(op.matvec, n)

    result = benchmark_solver(
        name="PCG",
        operator=op.matvec,
        preconditioner=pre.apply,
        rhs=rhs,
        tol=1e-6,
        max_iter=500,
        lambda_reg=1e-2,
    )
    assert result.name == "PCG"
    assert result.n == n
    assert result.iterations > 0
    assert result.solve_time_seconds >= 0
    assert result.final_residual < 1e-3


def test_benchmark_solver_without_preconditioner():
    """benchmark_solver without a preconditioner should return valid results."""
    torch.manual_seed(42)
    n = 50
    e = torch.randn(n, 8, dtype=torch.float64)
    op = AttentionKernelOperator(e, lambda_reg=1e-2, dtype=torch.float64)
    rhs = torch.randn(n, dtype=torch.float64)

    result = benchmark_solver(
        name="CG",
        operator=op.matvec,
        preconditioner=None,
        rhs=rhs,
        tol=1e-6,
        max_iter=500,
        lambda_reg=1e-2,
    )
    assert result.name == "CG"
    assert result.n == n
    assert result.iterations > 0
    assert result.solve_time_seconds >= 0


def test_benchmark_solver_with_reference():
    """benchmark_solver with reference_solution should compute objective_gap."""
    torch.manual_seed(42)
    n = 50
    e = torch.randn(n, 8, dtype=torch.float64)
    op = AttentionKernelOperator(e, lambda_reg=1e-2, dtype=torch.float64)
    rhs = torch.randn(n, dtype=torch.float64)

    # Exact solution for reference
    exact = torch.linalg.solve(op.to_dense(), rhs)

    pre = CCCPPreconditioner(
        num_probes=30, gamma=1e-1, max_iter=20, tol=1e-4, verbose=False, dtype=torch.float64
    )
    pre.build(op.matvec, n)

    result = benchmark_solver(
        name="PCG",
        operator=op.matvec,
        preconditioner=pre.apply,
        rhs=rhs,
        reference_solution=exact,
        tol=1e-6,
        max_iter=500,
        lambda_reg=1e-2,
    )
    assert result.objective_gap is not None
    assert result.objective_gap >= 0


def test_benchmark_laker_vs_baselines():
    """benchmark_laker_vs_baselines should return results for all solvers."""
    torch.manual_seed(42)
    n = 50
    e = torch.randn(n, 6, dtype=torch.float64)
    y = torch.randn(n, dtype=torch.float64)

    results = benchmark_laker_vs_baselines(
        embeddings=e,
        measurements=y,
        lambda_reg=1e-2,
        pcg_tol=1e-6,
        pcg_max_iter=500,
    )
    assert len(results) == 4
    names = [r.name for r in results]
    assert "LAKER" in names
    assert "Jacobi PCG" in names
    assert "CG (no precond)" in names
    assert "Gradient Descent" in names
    for r in results:
        assert r.n == n
        assert r.iterations >= 0
        assert r.solve_time_seconds >= 0


def test_benchmark_solver_zero_reference():
    """benchmark_solver with tiny reference objective should set gap to None."""
    torch.manual_seed(42)
    n = 30
    e = torch.randn(n, 6, dtype=torch.float64)
    op = AttentionKernelOperator(e, lambda_reg=1e-2, dtype=torch.float64)
    rhs = torch.randn(n, dtype=torch.float64)

    pre = CCCPPreconditioner(
        num_probes=20, gamma=1e-1, max_iter=20, tol=1e-4, verbose=False, dtype=torch.float64
    )
    pre.build(op.matvec, n)

    # Force reference solution to zero so obj_ref = lambda * y^T 0 = 0
    exact = torch.zeros(n, dtype=torch.float64)

    result = benchmark_solver(
        name="PCG",
        operator=op.matvec,
        preconditioner=pre.apply,
        rhs=rhs,
        reference_solution=exact,
        tol=1e-6,
        max_iter=500,
        lambda_reg=1e-2,
    )
    assert result.objective_gap is None
