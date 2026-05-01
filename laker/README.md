# LAKER Package

Learning-based Attention Kernel Regression for large-scale spectrum cartography.

## Package Structure

| Module | Purpose |
|--------|---------|
| `backend.py` | Device/dtype management and tensor conversion utilities. |
| `benchmark.py` | Solver benchmarking utilities (PCG, Jacobi, gradient descent). |
| `data.py` | Synthetic radio-field generation and regular grid creation. |
| `embeddings.py` | `PositionEmbedding` module: maps spatial coordinates to learned feature vectors. |
| `kernels.py` | **Exact and approximate kernel operators** (exact, Nyström, RFF, sparse k-NN, SKI). |
| `models.py` | `LAKERRegressor`: high-level sklearn-compatible estimator with PCG solver and CCCP preconditioner. |
| `preconditioner.py` | `CCCPPreconditioner`: learned data-dependent preconditioner via shrinkage-regularised CCCP. |
| `solvers.py` | `PreconditionedConjugateGradient`, `GradientDescent`, and `JacobiPreconditioner`. |
| `utils.py` | Numerical stability helpers (`trace_normalize`, `eigh_stable`, `adaptive_shrinkage_rho`) and Bayesian Optimisation surrogate (`GPSurrogate`). |
| `visualize.py` | Plotting utilities for radio maps and convergence curves. |

## Quick Start

```python
import torch
from laker import LAKERRegressor

# Generate synthetic data
locations = torch.rand(200, 2) * 100.0
measurements = torch.randn(200)

# Fit
model = LAKERRegressor(embedding_dim=10, lambda_reg=1e-2, verbose=True)
model.fit(locations, measurements)

# Predict on a grid
query = torch.rand(1000, 2) * 100.0
predictions = model.predict(query)
```

## Kernel Approximations

The `kernel_approx` argument controls the operator used inside `LAKERRegressor`:

- `None` — exact attention kernel (default, most accurate)
- `"nystrom"` — Nyström low-rank approximation
- `"rff"` — Random Fourier Features
- `"knn"` — sparse k-NN approximation
- `"ski"` — Structured Kernel Interpolation

Example:

```python
model = LAKERRegressor(kernel_approx="nystrom", num_landmarks=100)
model.fit(locations, measurements)
```

## Design Principles

- **Class-based executors**: All major components are implemented as classes with public APIs.
- **No semi-private naming**: All functions, methods, and variables use standard public naming (no leading underscores).
- **Logging over prints**: All diagnostic output goes through the standard `logging` module.
- **Combined modules**: Related functionality is co-located (e.g., all kernel variants live in `kernels.py`, all utilities in `utils.py`).
