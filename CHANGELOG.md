# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **Code standardization**: Removed all semi-private naming (leading underscores) from functions, methods, classes, and variables. Replaced terse abbreviations (`cs`, `lam`, `pre`, `vt`, etc.) with descriptive PEP 8 identifiers. Replaced all `print()` calls with `logging` module usage, including in documentation code blocks.
- **Architecture**: Introduced `Executor` abstract base class in `laker/executor.py`. Standardized the "Class + Convenience Wrapper" pattern across examples and benchmarks. All public workflows now have class implementations with optional executor injection; free-function wrappers are thin one-line conveniences only.
- **File reorganization**: Consolidated `laker/low_rank_kernels.py`, `laker/ski_kernels.py`, and `laker/sparse_kernels.py` into `laker/kernels.py`. Renamed benchmark and example scripts to module-runnable names (`benchmarks/reproducible.py`, `benchmarks/baseline.py`, `benchmarks/approximations.py`, `benchmarks/run.py`, `examples/basic.py`, `examples/large.py`).
- **Benchmark API**: Converted `benchmark_solver` and `benchmark_laker_vs_baselines` free functions to `SolverBenchmark` and `BaselineBenchmark` classes in `laker/benchmark.py`. Original function names preserved as thin convenience wrappers.

### Fixed
- **Broken imports**: Fixed benchmark and example imports referencing deleted modules (`laker.low_rank_kernels`, `laker.ski_kernels`, `laker.sparse_kernels`).
- **Dead code**: Removed unused `self.lambda_vec = None` in `AttentionKernelOperator` and unused local `kernel_mv` in `laker/benchmark.py`.

### Added
- **Documentation**: Added `docs/patterns.md` documenting the Executor pattern, Class + Convenience Wrapper convention, naming rules, and logging requirements.
- **Module structure**: Added `benchmarks/__init__.py` and `examples/__init__.py` so benchmarks and examples can be run as modules (`python -m benchmarks.reproducible`, etc.).

## [0.3.0] - 2026-04-30

### Changed
- **Math simplification**: Removed redundant `inv_isotropic_coef` and `r_column_norms_sq` terms from CCCP probe denominator because QR-normalised probes have unit column norms. This eliminates numerical drift and an unnecessary buffer.
- **Memory efficiency**: Replaced explicit `factored_inverse = V @ diag(1/eig) @ V.T` with basis-scaled matmul (`vtr.T @ scaled_vtr`) in CCCP, avoiding an `O(N_r^3)` allocation per iteration.
- **Adaptive chunking**: `AttentionKernelOperator.matvec` now auto-selects between fast 1-D chunking and full 2-D tiling based on a 64 MB memory heuristic, bounding peak memory to `O(chunk_size^2)` instead of `O(chunk_size * n)`.
- **Chunked prediction**: `LAKERRegressor.predict` now supports 2-D tiled kernel evaluation for large query sets, preventing `O(m*n)` memory blow-up.
- **In-place operations**: Replaced materialising `torch.exp` intermediates with `torch.exp(..., out=...)` in `matvec`, `to_dense`, and `kernel_eval`.
- **Buffer reuse**: Pre-allocated `factored_matrix`, `f_gamma_q_basis`, and `shrunken_f_gamma` buffers in CCCP loop to reduce GC pressure.
- **Default dtype**: Switched default from `float64` to `float32` for ML ecosystem compatibility.
- **Default tolerance**: Relaxed default `pcg_tol` from `1e-10` to `1e-6` (appropriate for float32).
- **Condition number estimation**: Reduced power iterations from 20 to 10 and inverse CG steps from 10x200 to 5x50, since the preconditioned system is well-conditioned by design.
- **Overflow guard**: Added `_exp_safe` helper with dtype-aware clamping (`80.0` for float32, `700.0` for float64) before exponentiation to prevent silent overflow to `inf` in the attention kernel.
- **Optimised grid search**: `fit_with_search` now computes embeddings once and reuses them across all trials, giving a 3-5x speedup. It also catches specific exceptions (`RuntimeError`, `ValueError`) and raises a clear error if all trials fail.
- **PCG in-place updates**: Replaced `p = z + beta * p` with `p.mul_(beta).add_(z)` to eliminate one tensor allocation per iteration in both 1-D and 2-D solves.
- **Thread-safety documentation**: Added a docstring warning to `PositionEmbedding` that its temporary manipulation of the global PyTorch RNG is not thread-safe.

### Added
- **Mixed-precision training**: `LAKERRegressor` now supports `embedding_dtype` parameter. Embeddings can be computed in `float16`/`bfloat16` and cast to the solver dtype, halving embedding memory.
- **Low-rank kernel approximations**: Added `NystromAttentionKernelOperator` (greedy landmark selection, `O(n*m)` matvec) and `RandomFeatureAttentionKernelOperator` (RFF with deterministic seed, `O(n*r)` matvec). Controlled via `kernel_approx` parameter in `LAKERRegressor`.
- **Sparse k-NN kernel**: Added `SparseKNNAttentionKernelOperator` in `laker/sparse_kernels.py`. Euclidean-distance k-NN graph with automatic symmetrisation and diagonal-dominance enforcement guarantees positive definiteness. Storage is `O(n*k)` and matvec cost is `O(n*k)`. Use `kernel_approx="knn"`.
- **Predictive variance / uncertainty quantification**: Added `LAKERRegressor.predict_variance(x)` for kernel ridge regression. Exact kernels use batched PCG solves with the learned preconditioner; RFF uses a closed-form Woodbury identity for near-instant variance.
- **Regularization path**: Added `LAKERRegressor.fit_path(lambda_reg_grid)` that fits a sequence of regularisation strengths with warm-started PCG (largest `lambda` first). Embeddings and the preconditioner are built once, giving near-linear cost in the number of grid points.
- **Validation-based grid search**: Added `fit_with_search` method that splits data into train/val, searches over `lambda_reg`, `gamma`, and `num_probes`, and retrains the best configuration on the full dataset.
- **Reproducible benchmark suite**: Added `benchmarks/reproducible_benchmarks.py` with fixed seeds, multiple trials, and mean/std reporting. Added `benchmarks/compare_approximations.py` for exact vs low-rank comparison.
- **Test coverage**: Added `tests/test_low_rank_kernels.py` and `tests/test_advanced_features.py` covering mixed-precision, Nyström, RFF, grid search, fit_path, predict_variance, and k-NN kernels.

### Fixed
- **Benchmark bug**: `benchmark_full_fit` was returning `kernel_operator.n` instead of actual PCG iteration count.
- **PCG robustness**: Replaced `+ 1e-16` fudge factors with explicit breakdown detection (`p^T A p <= 0`). Residual replacement is now **disabled by default** because it causes catastrophic cancellation in float32; users can opt in via `restart_freq` for high-precision float64 runs.
- **PCG batch support**: Solver now accepts 2-D RHS using vectorised column-wise dot products.
- **PCG 1-D fast path**: Restored scalar `torch.dot` for single-RHS solves, avoiding the tensor-broadcasting overhead introduced by the batch path.
- **PositionEmbedding determinism**: MLP layers now use PyTorch's default `kaiming_uniform_` init with the global RNG temporarily seeded, ensuring deterministic behaviour while preserving the weight distribution of the original release.
- **Custom embedding save/load**: Custom embedding modules are now importable via `tests/custom_embed.py`, making save/load round-trips work correctly.
- **RFF kernel_eval inconsistency**: `RandomFeatureAttentionKernelOperator.kernel_eval` now returns the proper RFF feature-map approximation (`phi_x @ phi_y.T / r`) instead of the exact kernel.
- **Nyström overflow guard**: `NystromAttentionKernelOperator._compute_kernel_matrix` and `kernel_eval` now use `_exp_safe` to prevent overflow.
- **predict overflow guard**: `LAKERRegressor.predict`'s 2-D tiled path now uses `_exp_safe` instead of raw `torch.exp`.
- **predict low-rank consistency**: `LAKERRegressor.predict` no longer falls back to exact-kernel 2-D tiling for low-rank approximations, ensuring predictions are consistent with the fitted model.
- **SKI kernel approximation**: Added `SKIAttentionKernelOperator` with product grid and multilinear interpolation. Matvec cost is `O(n * grid_size)` instead of `O(n^2)`. Controlled via `kernel_approx="ski"` and `grid_size` parameter.
- **Bayesian hyperparameter optimisation**: Added `LAKERRegressor.fit_with_bo` with lightweight GP surrogate (RBF kernel, log-scale) and Expected Improvement acquisition. No external dependencies. Typically converges in 10-15 evaluations.
- **Streaming / online learning**: Added `LAKERRegressor.partial_fit(x_new, y_new)` for incremental updates. Enlarges the system, warm-starts PCG from the previous alpha, and rebuilds the preconditioner when a configurable threshold is reached.
- **Learned embeddings**: Added `LAKERRegressor.fit_learned_embeddings(x, y, lr, epochs)` which optimises `PositionEmbedding` MLP weights end-to-end via Adam on the residual loss, backpropagating through the differentiable kernel operator. Preconditioner is rebuilt periodically (`rebuild_freq`).
- **Multi-GPU distributed matvec**: Added `DistributedAttentionKernelOperator` in `laker/distributed_kernels.py`. Shards embeddings across available CUDA devices, computes local matvecs, and gathers results. Falls back to single-device wrapper when only one GPU is available.
- **Autograd-safe exponential**: `_exp_safe` now detects `requires_grad=True` and returns `torch.exp(clamped)` without the in-place `out=` form, enabling backprop through the kernel operator during learned-embedding training.

## [0.0.1] - 2026-04-29

### Added
- Initial release of LAKER (Learning-based Attention Kernel Regression).
- `LAKERRegressor`: sklearn-compatible estimator for attention kernel regression.
- `AttentionKernelOperator`: matrix-free exponential attention kernel with chunked evaluation.
- `CCCPPreconditioner`: learned data-dependent preconditioner via shrinkage-regularised CCCP with factored `O(N_r^3)` implementation.
- `PreconditionedConjugateGradient`: standard PCG solver with convergence monitoring.
- `PositionEmbedding`: deterministic position-driven embedding module.
- Baseline solvers: `GradientDescent` and `JacobiPreconditioner`.
- Synthetic radio-field generators and visualisation utilities.
- Comprehensive test suite with 14+ tests.
- Documentation and usage examples.

[Unreleased]: https://github.com/sachn-cs/laker/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/sachn-cs/laker/compare/v0.0.1...v0.3.0
[0.0.1]: https://github.com/sachn-cs/laker/releases/tag/v0.0.1
