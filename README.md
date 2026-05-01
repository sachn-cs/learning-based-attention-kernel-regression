# LAKER: Learning-based Attention Kernel Regression

Production-ready PyTorch implementation of the LAKER algorithm from
*[Accelerating Regularized Attention Kernel Regression for Spectrum Cartography](https://arxiv.org/html/2604.25138v1)* (Tao & Tan, 2026).

> **Disclaimer:** This repository is an independent implementation of the LAKER algorithm. I am not an author of the original paper.

LAKER solves large-scale regularized attention kernel regression problems of the form

```
min_alpha ||G alpha - y||_2^2 + lambda alpha^T G alpha
```

where `G = exp(E E^T)` is an exponential attention kernel induced by learned
embeddings `E`. The key innovation is a **learned data-dependent preconditioner**
obtained via a shrinkage-regularized Convex-Concave Procedure (CCCP), which
reduces the condition number of the system by up to three orders of magnitude
and enables near size-independent Preconditioned Conjugate Gradient (PCG)
convergence.

## Requirements

- Python >= 3.9
- PyTorch >= 2.0.0
- NumPy >= 1.23.0

## Features

- **Scalable to 100k+ samples**: Matrix-free attention kernel with adaptive 1-D/2-D
tiling and optional explicit mode for small problems.
- **Low-rank kernel approximations**: Nyström, random Fourier features (RFF),
sparse k-NN, and SKI (Structured Kernel Interpolation) reduce matvec cost
from `O(n^2)` to `O(n*r)` or `O(n log n)`.
- **Predictive variance / uncertainty quantification**: Exact variance via batched
PCG; closed-form variance for RFF via the Woodbury identity.
- **Mixed-precision training**: Compute embeddings in `float16`/`bfloat16` while
keeping the kernel solver in `float32`/`float64`.
- **Automatic hyperparameter search**: `fit_with_search` performs validation-based
grid search, and `fit_with_bo` uses Bayesian optimization with a lightweight
GP surrogate.
- **Regularization path**: `fit_path` fits a sequence of `lambda_reg` values with
warm-started PCG.
- **Streaming / online learning**: `partial_fit` incrementally updates the model
with new data using warm-start and optional preconditioner rebuild.
- **Learned embeddings**: `fit_learned_embeddings` optimises the `PositionEmbedding`
MLP weights end-to-end via backprop through the kernel operator.
- **Multi-GPU distributed matvec**: `DistributedAttentionKernelOperator` shards
embeddings across CUDA devices and gathers results automatically.
- **GPU acceleration**: First-class PyTorch backend with CPU/CUDA/MPS support.
- **Efficient preconditioner**: Factored representation exploits fixed random-probe
structure to achieve `O(N_r^3)` CCCP iterations independent of problem size `n`.
- **Production API**: `sklearn`-compatible `LAKERRegressor` with standard
`fit`/`predict` interface.
- **Exact replicability**: Faithful implementation of Algorithm 1 from the paper
with numerically stable shrinkage and trace normalization.
- **Modular design**: Swap embeddings, kernels, solvers, and preconditioners
independently.

## What Changed

This release includes algorithmic, numerical, and memory-efficiency improvements
over the original implementation. All changes preserve exact mathematical
correctness while reducing overhead and memory footprint.

### Math Optimizations

1. **Simplified CCCP denominator.** The original code computed probe
denominators as `inv_iso + diag(R^T M^{-1} R) - inv_iso * ||r_k||^2 + eps`.
Because the QR-normalised probes have unit column norms, the `inv_iso` and
`||r_k||^2` terms cancel exactly. Removing them eliminates numerical drift and
an unnecessary buffer.

2. **Avoided explicit matrix inverse in CCCP.** The loop previously formed the
full `M^{-1}` via `V @ diag(1/eig) @ V.T`. We now compute `R^T M^{-1} R`
directly in the eigenbasis (`vtr.T @ scaled_vtr`) without materialising the
inverse, saving an `O(N_r^3)` allocation per iteration.

3. **In-place tensor operations.** Replaced intermediate allocations with
`torch.add(out=...)`, `torch.mul(out=...)`, `torch.lerp(out=...)`, and
`torch.exp(..., out=...)` throughout the kernel and preconditioner hot paths.

4. **Overflow-safe exponential.** All `torch.exp` calls on gram matrices now
clamp to `80.0` (float32) or `700.0` (float64) before exponentiation, preventing
silent overflow to `inf` on large norms.

### Algorithm / Code Optimizations

5. **Adaptive 1-D / 2-D tiling.** `AttentionKernelOperator.matvec` now
auto-selects between fast 1-D chunking and full 2-D tiling based on a 64 MB
memory heuristic. For large `n`, peak memory drops from `O(chunk_size * n)` to
`O(chunk_size^2)`.

6. **Chunked prediction.** `LAKERRegressor.predict` now supports 2-D tiled
evaluation for large query sets, preventing `O(m*n)` memory blow-up.

7. **Pre-allocated buffers in CCCP.** `factored_matrix`, `f_gamma_q_basis`, and
`shrunken_f_gamma` are reused across iterations instead of being reallocated,
reducing GC pressure.

8. **Robust PCG solver.** Explicit breakdown detection (`p^T A p <= 0`), optional
residual replacement (off by default for float32 stability), and support for 2-D
batch RHS. Removed `1e-16` fudge factors. In-place `p` updates eliminate
per-iteration allocations.

9. **Warm-start grid search.** `fit_with_search` now computes embeddings once
and reuses them across all hyperparameter trials, yielding a 3-5x speedup.

### New Features

10. **Mixed-precision training.** Added `embedding_dtype` parameter to
`LAKERRegressor`. Embeddings can be generated in `float16` or `bfloat16` and
cast to the solver dtype, cutting memory for the embedding forward pass in half.

11. **Low-rank kernel approximations.** Added `NystromAttentionKernelOperator`
and `RandomFeatureAttentionKernelOperator`. The Nyström variant uses greedy
landmark selection and reduces matvec cost to `O(n*m)`. The RFF variant uses
random Fourier features with a deterministic seed for reproducibility.

12. **Sparse k-NN attention kernel.** Added `SparseKNNAttentionKernelOperator`
with Euclidean k-NN graph construction, symmetrisation, and diagonal-dominance
enforcement for guaranteed positive definiteness.

13. **SKI (Structured Kernel Interpolation).** Added
`SKIAttentionKernelOperator` with product grid and multilinear interpolation.
Matvec cost is `O(n * grid_size)` instead of `O(n^2)`.

14. **Validation-based grid search.** Added `fit_with_search` method that
splits data into train/val, searches over `lambda_reg`, `gamma`, and
`num_probes`, and retrains the best configuration on the full dataset.

15. **Bayesian hyperparameter optimisation.** Added `fit_with_bo` with a
lightweight GP surrogate (RBF kernel, log-scale) and Expected Improvement
acquisition. Typically converges in 10-15 evaluations vs 27 for grid search.

16. **Regularization path.** Added `fit_path(lambda_reg_grid, reuse_precond=True)`
which fits a descending sequence of `lambda_reg` values, warm-starting PCG from
the previous solution. The preconditioner is reused because it is independent
of `lambda_reg`.

17. **Predictive variance.** Added `predict_variance(x)` for exact and RFF
kernels. Exact variance uses batched PCG solves; RFF variance uses the
Woodbury identity for a closed-form solution.

18. **Streaming / online learning.** Added `partial_fit(x_new, y_new)` for
incremental updates with warm-start and configurable preconditioner rebuild
threshold.

19. **Learned embeddings.** Added `fit_learned_embeddings(x, y, lr, epochs)`
which optimises the `PositionEmbedding` MLP weights via Adam on the residual
loss, backpropagating through the differentiable kernel operator.

20. **Multi-GPU distributed matvec.** Added `DistributedAttentionKernelOperator`
which shards embeddings across available CUDA devices, computes local matvecs,
and gathers results back to the master device. Falls back gracefully to
single-device execution when only one GPU is available.

## Performance

All numbers below were measured on an Apple M3 (Darwin) with PyTorch 2.11.0,
fixed seed `42`, and 50 measurement trials for matvec. The **baseline** is the
pre-optimisation code run under identical conditions (`float64`, `pcg_tol=1e-10`).
The **optimised** column is the new code with the same settings, isolating
algorithmic changes from the float32 switch.

### Speedup vs Baseline (float64)

| Metric | Baseline | Optimised | Speedup |
|---|---|---|---|
| Kernel matvec n=5000 | 30.40 ms | 25.96 ms | **1.17x** |
| Preconditioner build n=5000 | 103.84 ms | 84.78 ms | **1.22x** |
| Full fit n=1000 | 346.98 ms | 322.38 ms | **1.08x** |

The full-fit time is within ~8 % of baseline (well inside run-to-run variance on
a laptop). The real wins are in **memory scaling**: for `n = 100,000` and
`chunk_size = 8192`, peak block memory drops from **3.2 GB** (original 1-D
chunking) to **256 MB** (2-D tiling), a **12x memory reduction**.

### Optimised Defaults (float32)

| n | Matvec (ms) | Pre build (ms) | Full fit (ms) | PCG iters |
|---|---|---|---|---|
| 1000 | 1.91 | 10.4 | 503.8 | 191 |
| 2000 | 5.37 | 18.8 | — | 216 |
| 5000 | 33.48 | 121.7 | — | 260 |

### Approximation Matvec Comparison (n=2000)

| method | mean (ms) | speedup vs exact |
|---|---|---|
| exact | 6.82 | 1.0x |
| nystrom | 0.04 | **170x** |
| rff | 0.09 | **76x** |
| knn | 4.31 | 1.6x |
| ski | 60.41 | 0.11x |

Nyström and RFF provide massive speedups for large `n`. SKI is currently
optimised for very large `n` with small `embedding_dim` and pays interpolation
overhead at `n=2000`.

## How It Works

### Adaptive Tiling

For `n <= chunk_size` or when a single chunk against the full input fits in a
64 MB budget, we use 1-D chunking (fastest path). Otherwise we tile over both
the output and reduction dimensions, keeping peak memory bounded.

### Factored Preconditioner

The learned covariance `Sigma` is maintained as `a*I + Q*C*Q^T` where `Q` is an
orthonormal basis for the random-probe span. This reduces each CCCP iteration
to `O(N_r^3)` instead of `O(n^3)`.

### PCG Solver

The solver uses standard preconditioned conjugate gradient with explicit
breakdown detection (`p^T A p <= 0`). Optional residual replacement
(`restart_freq`) can suppress round-off drift in very long float64 runs, but it
is **disabled by default** because it causes catastrophic cancellation in
float32.

## Design Patterns

The codebase follows a documented set of conventions described in
[`docs/patterns.md`](docs/patterns.md). Key patterns include the **Executor**
abstraction for structured logging and timing, and the **Class + Convenience
Wrapper** rule that requires every public workflow to have a class
implementation.

## How to Run

### Installation

```bash
pip install -e ".[dev]"
pytest tests/
```

### Quick Start

```python
import torch
from laker import LAKERRegressor

n = 1000
x_train = torch.rand(n, 2) * 100.0
y_train = torch.randn(n)

model = LAKERRegressor(
    embedding_dim=10,
    lambda_reg=1e-2,
    gamma=1e-1,
    num_probes=None,
    cccp_max_iter=100,
    pcg_tol=1e-6,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
model.fit(x_train, y_train)

x_test = torch.rand(2000, 2) * 100.0
y_pred = model.predict(x_test)
```

### Mixed-precision training

```python
model = LAKERRegressor(
    embedding_dim=10,
    embedding_dtype=torch.float16,  # half-precision embeddings
    dtype=torch.float32,            # full-precision solver
    pcg_tol=1e-6,
)
model.fit(x_train, y_train)
```

### Low-rank kernel approximation

```python
model = LAKERRegressor(
    embedding_dim=10,
    kernel_approx="nystrom",       # or "rff", "knn", "ski"
    num_landmarks=200,             # for Nyström
    k_neighbors=50,                # for k-NN
    grid_size=1024,                # for SKI
    dtype=torch.float64,
)
model.fit(x_train, y_train)
```

### Automatic hyperparameter search

```python
model = LAKERRegressor(embedding_dim=10, dtype=torch.float64, verbose=True)
model.fit_with_search(
    x_train, y_train,
    val_fraction=0.2,
    lambda_reg_grid=[1e-3, 1e-2, 1e-1],
    gamma_grid=[0.0, 1e-1, 1.0],
    num_probes_grid=[50, 100, 200],
)
```

### Bayesian hyperparameter optimisation

```python
model = LAKERRegressor(embedding_dim=10, dtype=torch.float64, verbose=True)
model.fit_with_bo(
    x_train, y_train,
    val_fraction=0.2,
    n_calls=12,
    n_initial_points=4,
    lambda_reg_bounds=(1e-3, 1e-1),
    gamma_bounds=(0.0, 1.0),
    num_probes_bounds=(50, 200),
)
```

### Regularization path

```python
path = model.fit_path(
    x_train, y_train,
    lambda_reg_grid=[1e-1, 1e-2, 1e-3],
    reuse_precond=True,
)
# path["alphas"] is a list of alpha vectors, one per lambda_reg
```

### Predictive variance

```python
model.fit(x_train, y_train)
var = model.predict_variance(x_test)  # shape (m,)
```

### Streaming / online learning

```python
model.fit(x_train, y_train)
model.partial_fit(x_new, y_new, forgetting_factor=1.0, rebuild_threshold=100)
```

### Learned embeddings

```python
model.fit(x_train, y_train)
model.fit_learned_embeddings(
    x_train, y_train,
    lr=1e-2, epochs=20, rebuild_freq=5, patience=10
)
```

### Multi-GPU distributed matvec

```python
from laker.distributed_kernels import DistributedAttentionKernelOperator

# Inside LAKERRegressor, set distributed=True
model = LAKERRegressor(
    embedding_dim=10,
    lambda_reg=1e-2,
    distributed=True,
    device="cuda",
)
model.fit(x_train, y_train)
```

### Reproducing Benchmarks

```bash
# Full reproducible benchmark suite (fixed seeds, statistics)
python -m benchmarks.reproducible

# Baseline vs optimised comparison
python -m benchmarks.baseline

# Approximation speed comparison
python -m benchmarks.approximations

# Legacy quick benchmarks
python -m benchmarks.run
```

Results are written to `benchmarks/README.md`.

### Command-line interface

```bash
laker fit --locations x_train.pt --measurements y_train.pt --output model.pt
laker predict --model model.pt --locations x_test.pt --output y_pred.pt
```

### Save and load

```python
model.save("laker_model.pt")
loaded = LAKERRegressor.load("laker_model.pt")
```

## Limitations

1. **PCG does not always converge within max_iter.** On very ill-conditioned
problems or with `float32`, the solver may hit the iteration cap. Using
`dtype=torch.float64` and `pcg_tol=1e-10` usually fixes this at a ~2x runtime
cost.

2. **Default float32 trades accuracy for speed.** The float32 path is suitable
for most ML workloads but can struggle when `lambda_reg` is very small
(`< 1e-4`) or when the kernel matrix has entries near the float32 dynamic
range.

3. **CPU-only overhead for small problems.** On CPU, the matrix-free kernel
matvec overhead dominates for `n < 500`; explicit mode (chunk_size=None) is
faster but costs `O(n^2)` memory.

4. **Random-probe preconditioner quality depends on `N_r`.** If `N_r` is too
small relative to the cluster count `Q`, the learned preconditioner may be
suboptimal and PCG convergence degrades.

5. **Low-rank approximations are rough for exponential kernels.** The Nyström
and RFF approximations reduce matvec cost but can have high relative error on
the fast-growing exponential kernel. They are best used for very large `n`
where exact evaluation is infeasible, or when speed dominates accuracy.

6. **Custom embeddings must be importable for save/load.** If you pass a custom
`embedding_module` to `LAKERRegressor`, the module and class must be importable
when calling `LAKERRegressor.load()`. Inline or lambda-defined modules will
fall back to `PositionEmbedding` on load.

7. **SKI grid grows exponentially with embedding_dim.** Because SKI builds a
product grid in the embedding space, the grid size scales as `gpd^d`. For
`embedding_dim > 10`, the grid becomes impractical; use Nyström or RFF instead.

8. **Distributed kernel is data-parallel with replicated full embeddings.**
Each GPU stores its local embedding shard, but the current matvec
implementation gathers the full vector to each device. True model parallelism
(all-reduce over partial contributions) is not yet implemented.

## Future Improvements

- **Warm-start preconditioner:** Re-use a previously learned preconditioner as
an initial guess when fitting on incrementally growing datasets.
- **Distributed model parallelism:** Implement all-reduce over partial
contributions instead of replicating the full vector on each device.
- **Kernel approximation accuracy:** The exponential attention kernel grows very
fast, so low-rank approximations (Nyström, RFF) are currently rough. Exploring
landmark-selection heuristics tailored to exponential kernels would improve
accuracy.
- **Batch prediction:** Vectorise `predict` over multiple independent query sets
for downstream batch-processing pipelines.
- **Sparse tensor backends:** Use `torch.sparse_csr` or `scipy.sparse` for
k-NN matvec on CPU to avoid dense intermediate allocations.

## Citation

```bibtex
@article{tao2026laker,
  title={Accelerating Regularized Attention Kernel Regression for Spectrum Cartography},
  author={Tao, Liping and Tan, Chee Wei},
  journal={arXiv preprint arXiv:2604.25138},
  year={2026}
}
```
