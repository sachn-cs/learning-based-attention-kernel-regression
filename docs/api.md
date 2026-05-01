# API Reference

This page documents the public API of LAKER. All modules are imported from the top-level `laker` package.

---

## Core Model

### `LAKERRegressor`

```python
class LAKERRegressor(
    embedding_dim: int = 10,
    lambda_reg: float = 1e-2,
    gamma: float = 1e-1,
    num_probes: Optional[int] = None,
    epsilon: float = 1e-8,
    base_rho: float = 0.05,
    cccp_max_iter: int = 200,
    cccp_tol: float = 1e-6,
    pcg_tol: float = 1e-10,
    pcg_max_iter: int = 1000,
    chunk_size: Optional[int] = None,
    embedding_module: Optional[nn.Module] = None,
    kernel_approx: Optional[str] = None,
    num_landmarks: Optional[int] = None,
    num_features: Optional[int] = None,
    k_neighbors: Optional[int] = None,
    grid_size: Optional[int] = None,
    distributed: bool = False,
    embedding_dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
    verbose: bool = True,
)
```

Learning-based Attention Kernel Regression estimator. Fits the regularised attention kernel regression problem and solves it efficiently using a learned CCCP preconditioner inside PCG.

**Key methods:**

- `fit(x, y, x0=None)` — Fit the model to sparse measurements.
- `predict(x)` — Reconstruct the radio field at query locations.
- `predict_variance(x)` — Predictive variance (uncertainty) at query locations.
- `fit_with_search(x, y, ...)` — Fit with validation-based grid search over hyperparameters.
- `fit_with_bo(x, y, ...)` — Fit with Bayesian Optimisation over hyperparameters.
- `partial_fit(x_new, y_new, ...)` — Incremental update with new observations.
- `fit_path(x, y, lambda_reg_grid, ...)` — Fit a regularisation path over multiple $\lambda$ values.
- `fit_learned_embeddings(x, y, lr=1e-3, epochs=50, ...)` — End-to-end optimisation of embedding MLP weights.
- `save(path)` — Serialize the fitted model to disk.
- `load(path)` — Deserialize a model from disk (class method).
- `score(x, y)` — Negative RMSE for sklearn compatibility.
- `condition_number()` — Estimated condition number of the preconditioned system.
- `get_params(deep=True)` — Return estimator parameters for sklearn compatibility.
- `set_params(**params)` — Set estimator parameters for sklearn compatibility.

---

## Kernels

All kernel operators share the interface `matvec(x)`, `diagonal()`, `to_dense()`, and `kernel_eval(x, y, chunk_size)`.

### `AttentionKernelOperator`

Exact exponential attention kernel $G = \exp(E E^{\top})$ with optional chunked evaluation.

```python
class AttentionKernelOperator(
    embeddings: torch.Tensor,
    lambda_reg: float = 1e-2,
    chunk_size: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
)
```

### `NystromAttentionKernelOperator`

Nyström low-rank approximation using $m$ landmark points.

```python
class NystromAttentionKernelOperator(
    embeddings: torch.Tensor,
    lambda_reg: float = 1e-2,
    num_landmarks: Optional[int] = None,
    chunk_size: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
)
```

### `RandomFeatureAttentionKernelOperator`

Random Fourier Feature (RFF) approximation.

```python
class RandomFeatureAttentionKernelOperator(
    embeddings: torch.Tensor,
    lambda_reg: float = 1e-2,
    num_features: Optional[int] = None,
    sigma: float = 1.0,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
)
```

### `SparseKNNAttentionKernelOperator`

Sparse k-NN approximation stored as a COO tensor.

```python
class SparseKNNAttentionKernelOperator(
    embeddings: torch.Tensor,
    lambda_reg: float = 1e-2,
    k_neighbors: Optional[int] = None,
    chunk_size: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
)
```

### `SKIAttentionKernelOperator`

Structured Kernel Interpolation on a product grid.

```python
class SKIAttentionKernelOperator(
    embeddings: torch.Tensor,
    lambda_reg: float = 1e-2,
    grid_size: Optional[int] = None,
    grid_bounds: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
)
```

### `DistributedAttentionKernelOperator`

Multi-GPU wrapper that shards embeddings across CUDA devices.

```python
class DistributedAttentionKernelOperator(
    embeddings: torch.Tensor,
    lambda_reg: float = 1e-2,
    master_device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
)
```

---

## Preconditioner

### `CCCPPreconditioner`

Learned data-dependent preconditioner $P = \Sigma^{-1/2}$ via shrinkage-regularised CCCP.

```python
class CCCPPreconditioner(
    num_probes: Optional[int] = None,
    gamma: float = 1e-1,
    epsilon: float = 1e-8,
    base_rho: float = 0.05,
    max_iter: int = 200,
    tol: float = 1e-6,
    verbose: bool = True,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
)
```

**Key methods:**

- `build(operator, n)` — Learn the preconditioner for an $n \times n$ operator. Returns `self`.
- `apply(x)` — Apply $P$ to a vector or batch of vectors.
- `apply_1d(x)` — Apply $P$ to a single vector.
- `apply_2d(x)` — Apply $P$ to a matrix.
- `to_dense()` — Materialise the full dense preconditioner (for debugging).

---

## Solvers

### `PreconditionedConjugateGradient`

PCG solver for symmetric positive-definite systems.

```python
class PreconditionedConjugateGradient(
    tol: float = 1e-10,
    max_iter: Optional[int] = None,
    verbose: bool = True,
    restart_freq: Optional[int] = None,
    breakdown_eps: Optional[float] = None,
)
```

**Key methods:**

- `solve(operator, preconditioner, rhs, x0=None)` — Solve $A x = b$.

**Attributes:**

- `iterations` — Number of iterations performed.
- `residual_norm` — Final residual norm.

### `GradientDescent`

Unpreconditioned gradient descent baseline.

```python
class GradientDescent(
    step_size: Optional[float] = None,
    tol: float = 1e-3,
    max_iter: int = 50000,
    verbose: bool = False,
)
```

- `solve(operator, rhs, x0=None)` — Solve $A x = b$ via gradient descent.

### `JacobiPreconditioner`

Diagonal (Jacobi) preconditioner baseline.

```python
class JacobiPreconditioner(diagonal: torch.Tensor)
```

- `apply(x)` — Element-wise multiply by the inverse diagonal.

---

## Embeddings

### `PositionEmbedding`

Deterministic position-driven embedding module.

```python
class PositionEmbedding(
    input_dim: int,
    embedding_dim: int,
    num_fourier: Optional[int] = None,
    sigma: float = 10.0,
    seed: int = 42,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
)
```

Maps spatial coordinates to embeddings via random Fourier features followed by a deterministic MLP. Inherits from `torch.nn.Module`.

- `forward(x)` — Map locations to embeddings.

---

## Data Generation

### `RadioFieldGenerator`

Generator for synthetic radio propagation fields.

```python
class RadioFieldGenerator(
    path_loss_exponent: float = 2.0,
    reference_distance: float = 1.0,
    shadow_sigma: float = 1.5,
)
```

- `generate(locations, transmitters, powers, seed=None)` — Return `(rss_clean, rss_noisy)`.

### `generate_radio_field`

Thin functional wrapper around `RadioFieldGenerator`.

```python
def generate_radio_field(
    locations: torch.Tensor,
    transmitters: torch.Tensor,
    powers: torch.Tensor,
    path_loss_exponent: float = 2.0,
    reference_distance: float = 1.0,
    shadow_sigma: float = 1.5,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]
```

### `generate_grid`

Generate a regular 2-D evaluation grid.

```python
def generate_grid(
    bounds: Tuple[float, float, float, float],
    grid_size: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor
```

---

## Visualisation

### `Visualizer`

High-level visualiser for LAKER regression outputs and diagnostics.

```python
class Visualizer(figsize: Tuple[int, int] = (6, 5))
```

- `radio_map_to_image(predictions, grid_size, extent=None)` — Convert flat predictions to a 2-D image.
- `plot_radio_map(predictions, grid_size, title, extent, colorbar_label, vmin, vmax)` — Plot a 2-D radio map.
- `plot_convergence(objective_gaps, labels, title, xlabel, ylabel, figsize)` — Plot convergence curves.

### Thin functional wrappers

- `radio_map_to_image(...)`
- `plot_radio_map(...)`
- `plot_convergence(...)`

---

## Utilities

### Numerical Stability

- `trace_normalize(mat: torch.Tensor) -> torch.Tensor` — Normalise a PD matrix so $\operatorname{tr}(M) = n$.
- `adaptive_shrinkage_rho(num_probes, problem_size, gamma, base_rho=0.05) -> float` — Compute adaptive shrinkage $\rho$.
- `eigh_stable(mat, eps=1e-10) -> Tuple[torch.Tensor, torch.Tensor]` — Stable symmetric eigendecomposition with eigenvalue clamping.

### Normal Distribution Helpers

- `scipy_norm_pdf(x: np.ndarray) -> np.ndarray` — Standard normal PDF.
- `scipy_norm_cdf(x: np.ndarray) -> np.ndarray` — Abramowitz and Stegun approximation of the normal CDF.

### Bayesian Optimisation

- `GPSurrogate(bounds, log_indices=None, sigma_f=1.0, length_scale=0.2, sigma_n=1e-4)` — Gaussian Process surrogate on a normalised parameter space.
  - `fit(X, y)` — Fit to observations.
  - `predict(X_new)` — Posterior mean and variance.
  - `expected_improvement(X_new, xi=0.01)` — Expected Improvement acquisition function.

---

## Benchmarking

### `BenchmarkResult`

```python
@dataclass
class BenchmarkResult:
    name: str
    n: int
    solve_time_seconds: float
    iterations: int
    final_residual: float
    condition_number: Optional[float] = None
    objective_gap: Optional[float] = None
```

### `SolverBenchmark`

Class-based benchmark for a single solver configuration.

```python
class SolverBenchmark(
    name: str,
    operator: Callable[[torch.Tensor], torch.Tensor],
    preconditioner: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    rhs: Optional[torch.Tensor] = None,
    reference_solution: Optional[torch.Tensor] = None,
    tol: float = 1e-10,
    max_iter: int = 1000,
    lambda_reg: float = 1e-2,
)
```

- `run() -> BenchmarkResult` — Execute the benchmark and return results.

### `BaselineBenchmark`

Class-based benchmark comparing LAKER against baseline solvers.

```python
class BaselineBenchmark(
    embeddings: torch.Tensor,
    measurements: torch.Tensor,
    lambda_reg: float = 1e-2,
    reference_solution: Optional[torch.Tensor] = None,
    pcg_tol: float = 1e-10,
    pcg_max_iter: int = 1000,
)
```

- `run() -> List[BenchmarkResult]` — Execute head-to-head comparison.

### Convenience wrappers

Thin one-line wrappers around the benchmark classes:

- `benchmark_solver(...)` — Convenience wrapper around `SolverBenchmark.run()`.
- `benchmark_laker_vs_baselines(...)` — Convenience wrapper around `BaselineBenchmark.run()`.

---

## Backend

- `get_default_device() -> torch.device`
- `set_default_device(device=None) -> torch.device`
- `get_default_dtype() -> torch.dtype`
- `set_default_dtype(dtype) -> None`
- `maybe_compile(func, mode="reduce-overhead")`
- `to_tensor(data, device=None, dtype=None) -> torch.Tensor`
