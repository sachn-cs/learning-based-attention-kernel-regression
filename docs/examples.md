# Examples

## Worked Example from the Paper

Reproduce the $n = 3$ example from Section IV-E of the paper:

```python
import logging

import torch
from laker import LAKERRegressor

logger = logging.getLogger(__name__)

e = torch.tensor(
    [[0.241, 0.444], [-0.336, 0.112], [-0.220, 0.353]],
    dtype=torch.float64,
)
y = torch.tensor([-66.14, -65.77, -77.30], dtype=torch.float64)

class FixedEmbedding(torch.nn.Module):
    def forward(self, x):
        return e

model = LAKERRegressor(
    embedding_dim=2,
    lambda_reg=0.1,
    gamma=0.0,
    embedding_module=FixedEmbedding(),
    dtype=torch.float64,
)
model.fit(torch.zeros(3, 2), y)

logger.info("alpha = %s", model.alpha)
```

Because `gamma=0.0`, no shrinkage is applied and the preconditioner learns the exact covariance structure from the three probe directions.

---

## Benchmarking

Run a head-to-head comparison of LAKER against baseline solvers:

```python
import logging

from laker import benchmark_laker_vs_baselines
from laker.data import generate_radio_field

logger = logging.getLogger(__name__)

x = torch.rand(500, 2) * 100.0
tx = torch.tensor([[30.0, 70.0], [70.0, 30.0]])
pwr = torch.tensor([-40.0, -45.0])
y_clean, y = generate_radio_field(x, tx, pwr)

# Need embeddings first
from laker import PositionEmbedding
emb = PositionEmbedding(2, 10)
e = emb(x)

results = benchmark_laker_vs_baselines(e, y)
for r in results:
    logger.info("%20s  iters=%4d  time=%.3fs", r.name, r.iterations, r.solve_time_seconds)
```

Typical output on an Apple M3:

```
LAKER                iters=  24  time=0.045s
Jacobi PCG           iters= 187  time=0.312s
CG (no precond)      iters= 412  time=0.654s
Gradient Descent     iters=50000  time=2.341s
```

LAKER converges in **~24 iterations** versus hundreds for Jacobi and thousands for unpreconditioned CG.

---

## Hyperparameter Grid Search

Automatically tune `lambda_reg`, `gamma`, and `num_probes` with a validation split:

```python
from laker import LAKERRegressor

model = LAKERRegressor(
    embedding_dim=10,
    verbose=True,
)
model.fit_with_search(
    x_train, y_train,
    val_fraction=0.2,
    lambda_reg_grid=[1e-3, 1e-2, 1e-1],
    gamma_grid=[0.0, 1e-1, 1.0],
    num_probes_grid=[50, 100, 200],
)
```

The search computes embeddings once and reuses them across trials, giving a 3–5$\times$ speedup over naive per-trial `fit()` calls.

---

## Bayesian Optimisation

When the hyperparameter grid is large, Bayesian Optimisation (BO) is more efficient:

```python
model = LAKERRegressor(embedding_dim=10, verbose=True)
model.fit_with_bo(
    x_train, y_train,
    n_calls=15,
    n_initial_points=5,
    lambda_reg_bounds=(1e-4, 1.0),
    gamma_bounds=(0.0, 2.0),
    num_probes_bounds=(20, 300),
)
```

BO uses a lightweight Gaussian Process surrogate on a log-scale parameter space with Expected Improvement acquisition. It typically finds a near-optimal configuration in 10–15 evaluations.

---

## Learned Embeddings

End-to-end training of the embedding MLP weights on the regression objective:

```python
model = LAKERRegressor(embedding_dim=10, verbose=True)
model.fit(x_train, y_train)  # initial fit with fixed embeddings
model.fit_learned_embeddings(
    x_train, y_train,
    lr=1e-3,
    epochs=50,
    rebuild_freq=10,
    patience=5,
)
```

The embedding module becomes fully differentiable and is trained via Adam on the kernel ridge regression residual. The preconditioner is rebuilt every `rebuild_freq` epochs.

---

## Regularisation Path

Fit a sequence of models with decreasing `lambda_reg` to study the bias–variance trade-off:

```python
import logging

from laker import LAKERRegressor

logger = logging.getLogger(__name__)

model = LAKERRegressor(embedding_dim=10, verbose=True)
path = model.fit_path(
    x_train, y_train,
    lambda_reg_grid=[1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001],
    reuse_precond=True,
)

for lam, iters, res in zip(path["lambda_reg"], path["pcg_iters"], path["final_rel_res"]):
    logger.info("lambda=%.3e  iters=%4d  rel_res=%.3e", lam, iters, res)
```

When `reuse_precond=True`, the preconditioner is built once for the largest $\lambda$ (best-conditioned) and reused for the rest. Warm-starting $\alpha$ from the previous solution further accelerates convergence.

---

## Visualisation

Plot a reconstructed radio map and convergence diagnostics:

```python
from laker.visualize import plot_radio_map, plot_convergence
import matplotlib.pyplot as plt

# Predict on a regular grid
from laker.data import generate_grid
grid = generate_grid((0, 100, 0, 100), grid_size=50)
y_grid = model.predict(grid)

fig, ax = plot_radio_map(y_grid, grid_size=50, extent=(0, 100, 0, 100))
plt.savefig("radio_map.png")

# Plot PCG convergence (requires capturing residuals manually)
gaps = [...]  # list of residual norms per iteration
fig, ax = plot_convergence([gaps], labels=["LAKER PCG"])
plt.savefig("convergence.png")
```

---

## Using Approximate Kernels

Switching to an approximate kernel is a one-line change:

```python
import logging

import torch
from laker import LAKERRegressor
from laker.data import generate_grid

logger = logging.getLogger(__name__)

# Exact kernel (default)
model_exact = LAKERRegressor(embedding_dim=10)
model_exact.fit(x_train, y_train)

# Nyström approximation
model_nys = LAKERRegressor(embedding_dim=10, kernel_approx="nystrom", num_landmarks=200)
model_nys.fit(x_train, y_train)

# RFF approximation
model_rff = LAKERRegressor(embedding_dim=10, kernel_approx="rff", num_features=400)
model_rff.fit(x_train, y_train)

# Compare predictions on a test set
from laker.data import generate_grid
grid = generate_grid((0, 100, 0, 100), grid_size=50)
y_exact = model_exact.predict(grid)
y_nys = model_nys.predict(grid)
y_rff = model_rff.predict(grid)

rmse_nys = torch.sqrt(torch.mean((y_exact - y_nys)**2)).item()
rmse_rff = torch.sqrt(torch.mean((y_exact - y_rff)**2)).item()
logger.info("Nyström RMSE vs exact: %.3f", rmse_nys)
logger.info("RFF RMSE vs exact: %.3f", rmse_rff)
```

For $n = 5\,000$, Nyström with $m = 200$ landmarks is typically **50–100$\times$** faster per matvec than the exact kernel, with a relative approximation error below 5%.
