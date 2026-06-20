# LAKER: Learning-based Attention Kernel Regression

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/sachn-cs/laker/actions/workflows/ci.yml/badge.svg)](https://github.com/sachn-cs/laker/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PyPI version](https://img.shields.io/pypi/v/laker.svg)](https://pypi.org/project/laker/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Production-ready PyTorch implementation of the LAKER algorithm from
*[Accelerating Regularized Attention Kernel Regression for Spectrum Cartography](https://arxiv.org/html/2604.25138v1)* (Tao & Tan, 2026).

> **Disclaimer:** This repository is an independent implementation of the LAKER algorithm. The author is not one of the paper's authors.

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

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Feature Demonstrations](#feature-demonstrations)
- [CLI Usage](#cli-usage)
- [Performance](#performance)
- [How It Works](#how-it-works)
- [Design Patterns](#design-patterns)
- [Development](#development)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Code of Conduct](#code-of-conduct)
- [Security](#security)
- [Citation](#citation)
- [License](#license)

---

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
- **Spectral-shaped kernel**: Learned monotone spline reshapes the spectrum of
  the embedding Gram matrix for better conditioning.
- **Bilevel hyperparameter learning**: Implicit differentiation through the PCG
  fixed-point optimises `lambda_reg` and embedding weights jointly against a
  validation loss via `fit_bilevel()`.
- **Uncertainty-aware training**: NLL + calibration penalty objective trains
  embeddings to produce well-calibrated predictive variances.
- **Residual corrector**: A tiny MLP trained on `y - y_hat_laker` captures local
  misspecification without destabilising the core solver.
- **Two-scale kernel**: Combines global Nyström + local sparse k-NN for both
  coherence and sharpness.
- **Continuation schedule**: Solve a decreasing `lambda_reg` path with warm-started
  PCG for stable tracking to sharper solutions.
- **Leverage-score landmarks**: Nyström landmark selection via ridge leverage
  scores (`landmark_method="leverage"`).
- **Modular design**: Swap embeddings, kernels, solvers, and preconditioners
  independently.

## Requirements

- Python >= 3.9
- PyTorch >= 2.0.0
- NumPy >= 1.23.0

## Installation

### From Source (Recommended)

```bash
git clone https://github.com/sachn-cs/laker.git
cd laker
pip install -e ".[dev]"
```

### Optional: Visualization Support

```bash
pip install -e ".[viz]"
```

### Verify Installation

```bash
pytest tests/ -v
```

## Quick Start

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

## Project Structure

```
laker/
├── laker/                     # Main package
│   ├── __init__.py            # Public API and version
│   ├── __main__.py            # CLI interface (laker fit, laker predict)
│   ├── models.py              # LAKERRegressor (sklearn-compatible API)
│   ├── core.py                # Core pipeline: embeddings, kernels, solvers
│   ├── kernels.py             # All kernel operators (exact, Nyström, RFF, etc.)
│   ├── preconditioner.py      # CCCP and adaptive preconditioners
│   ├── solvers.py             # PCG solver and baselines
│   ├── training.py            # Embedding training, residual correctors
│   ├── embeddings.py          # PositionEmbedding (random Fourier features + MLP)
│   ├── search.py              # Grid search and Bayesian optimization
│   ├── streaming.py           # Partial fit, regularization path, continuation
│   ├── distributed_kernels.py # Multi-GPU distributed matvec
│   ├── data.py                # Synthetic radio field generators
│   ├── visualize.py           # Radio map and convergence plotting
│   ├── benchmark.py           # Benchmarking utilities
│   ├── persistence.py         # Save/load models
│   ├── utils.py               # Numerical stability helpers
│   ├── backend.py             # Device/dtype management
│   ├── bilevel.py             # Bilevel hyperparameter learning
│   ├── implicit_diff.py       # Adjoint method for hypergradients
│   ├── correctors.py          # ResidualCorrector MLP
│   └── executor.py            # Abstract Executor base class
├── tests/                     # Test suite (19 files)
├── examples/                  # Worked examples
├── benchmarks/                # Benchmark suite
├── docs/                      # Sphinx documentation
├── .github/                   # CI/CD, issue templates, dependabot
├── pyproject.toml             # Build config and project metadata
├── CHANGELOG.md               # Release history
├── CONTRIBUTING.md            # Contribution guidelines
├── CODE_OF_CONDUCT.md         # Community standards
├── SECURITY.md                # Security policy
└── LICENSE                    # MIT License
```

## Feature Demonstrations

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

### Spectral-shaped kernel

```python
model = LAKERRegressor(
    embedding_dim=10,
    kernel_approx="spectral",
    spectral_knots=5,
    dtype=torch.float64,
)
model.fit(x_train, y_train)
```

### Two-scale kernel

```python
model = LAKERRegressor(
    embedding_dim=10,
    kernel_approx="twoscale",
    num_landmarks=100,
    k_neighbors=30,
    dtype=torch.float64,
)
model.fit(x_train, y_train)
```

### Bilevel hyperparameter learning

```python
model = LAKERRegressor(embedding_dim=10, dtype=torch.float64, verbose=True)
model.fit(x_train, y_train)
model.fit_bilevel(
    x_train, y_train, x_val, y_val,
    lr=1e-3, epochs=20, patience=5,
)
```

### Uncertainty-aware training

```python
model = LAKERRegressor(embedding_dim=10, dtype=torch.float64, verbose=True)
model.fit(x_train, y_train)
model.fit_uncertainty_aware(
    x_train, y_train,
    lr=1e-3, epochs=50, beta=0.1, patience=5,
)
var = model.predict_variance(x_test)
```

### Residual corrector

```python
model = LAKERRegressor(embedding_dim=10, verbose=True)
model.fit(x_train, y_train)
model.fit_residual_corrector(
    x_train, y_train,
    epochs=200, patience=10, weight_decay=1e-2,
)
```

### Continuation schedule

```python
model = LAKERRegressor(embedding_dim=10, dtype=torch.float64, verbose=True)
model.fit_continuation(
    x_train, y_train,
    lambda_max=1.0, lambda_min=1e-2, n_stages=5,
)
```

### Leverage-score Nyström landmarks

```python
model = LAKERRegressor(
    embedding_dim=10,
    kernel_approx="nystrom",
    num_landmarks=100,
    landmark_method="leverage",
    landmark_pilot_size=200,
)
model.fit(x_train, y_train)
```

### Save and load

```python
model.save("laker_model.pt")
loaded = LAKERRegressor.load("laker_model.pt")
```

## CLI Usage

For batch workflows you can use the bundled command-line tool:

```bash
laker fit --locations x_train.pt --measurements y_train.pt --output model.pt
laker predict --model model.pt --locations x_test.pt --output y_pred.pt
```

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

### Memory Reduction

For `n = 100,000` and `chunk_size = 8192`, peak block memory drops from
**3.2 GB** (original 1-D chunking) to **256 MB** (2-D tiling), a **12x memory
reduction**.

### Optimised Defaults (float32)

| n | Matvec (ms) | Pre build (ms) | Full fit (ms) | PCG iters |
|---|---|---|---|---|
| 1000 | 1.91 | 10.4 | 503.8 | 191 |
| 2000 | 5.37 | 18.8 | — | 216 |
| 5000 | 33.48 | 121.7 | — | 260 |

### Approximation Matvec Comparison (n=2000)

| Method | Mean (ms) | Speedup vs Exact |
|---|---|---|
| exact | 6.82 | 1.0x |
| nystrom | 0.04 | **170x** |
| rff | 0.09 | **76x** |
| knn | 4.31 | 1.6x |
| ski | 60.41 | 0.11x |

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

## Development

### Setup

```bash
git clone https://github.com/sachn-cs/laker.git
cd laker
pip install -e ".[dev]"
```

### Commands

| Command | Description |
|---------|-------------|
| `pytest tests/ -v` | Run all tests |
| `pytest tests/ --cov=laker` | Run tests with coverage |
| `black laker/ tests/` | Format code |
| `isort laker/ tests/` | Sort imports |
| `flake8 laker/ tests/` | Lint code |
| `mypy laker/` | Type check |

### Running Benchmarks

```bash
python -m benchmarks.reproducible    # Full reproducible benchmark suite
python -m benchmarks.baseline        # Baseline vs optimised comparison
python -m benchmarks.approximations  # Approximation speed comparison
python -m benchmarks.run             # Legacy quick benchmarks
```

Results are written to `benchmarks/README.md`.

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

## Roadmap

- [ ] Warm-start preconditioner for incremental datasets
- [ ] Distributed model parallelism (all-reduce over partial contributions)
- [ ] Improved landmark-selection heuristics for exponential kernels
- [ ] Batch prediction for multiple independent query sets
- [ ] Sparse tensor backends (`torch.sparse_csr`, `scipy.sparse`)
- [ ] Python 3.13 support
- [ ] GPU-accelerated Nyström landmark selection
- [ ] ONNX export for deployed models

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for
guidelines on how to get started.

## Code of Conduct

This project follows the [Contributor Covenant v2.1](CODE_OF_CONDUCT.md). By
participating, you are expected to uphold this code.

## Security

For reporting security vulnerabilities, please see [SECURITY.md](SECURITY.md).

## Citation

If you use LAKER in your research, please cite:

```bibtex
@article{tao2026laker,
  title={Accelerating Regularized Attention Kernel Regression for Spectrum Cartography},
  author={Tao, Liping and Tan, Chee Wei},
  journal={arXiv preprint arXiv:2604.25138},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file
for details.
