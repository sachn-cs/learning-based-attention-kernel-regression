# LAKER Code Review

**Repository:** `sachn-cs/laker` · v0.3.0  
**Scope:** Full source review — `laker/`, `tests/`, `benchmarks/`, `pyproject.toml`, `README.md`

---

## Overall Score

| Category | Score | Notes |
|---|---|---|
| Architecture & Design | 8 / 10 | Clean separation of concerns; minor leakage at the models layer |
| Numerical Correctness | 7 / 10 | Several silent precision / correctness traps remain |
| API & Usability | 7 / 10 | sklearn-compatible surface but incomplete CLI and rough edges |
| Code Quality & Style | 8 / 10 | Consistent, well-documented; a few PEP-8 and type-hint gaps |
| Testing | 6 / 10 | Good coverage breadth, but unit tests are shallow on math |
| Performance | 8 / 10 | Adaptive tiling, pre-alloc buffers; distributed path is naive |
| Documentation | 8 / 10 | README is excellent; inline math comments are high-quality |
| **Overall** | **7.4 / 10** | Solid research codebase; several production gaps to close |

---

## Issues

Issues are ordered by severity: **🔴 Bug / Correctness → 🟠 Design Flaw → 🟡 Code Quality → 🔵 Gap / Missing**.

---

### 🔴 Bugs & Correctness Issues

**1. `build_kernel_operator` — dead `distributed` branch (logic bug)**  
`models.py` lines 286–297: the `elif self.distributed` branch is placed _after_ all the `kernel_approx` branches, but the `else` clause immediately before it has already consumed the fall-through. The distributed operator is therefore **never constructed** when `kernel_approx=None` and `distributed=True`; the code falls into `raise ValueError(f"Unknown kernel_approx=...")`. The test `test_distributed.py` doesn't catch this because it patches the operator directly.

```python
# Current (broken)
elif self.kernel_approx == "ski":
    ...
elif self.distributed and self.kernel_approx is None:   # ← never reached
    ...
else:
    raise ValueError(...)
```

Fix: move the `distributed` check before the `kernel_approx` ladder or restructure to a separate guard at the top.

---

**2. `partial_fit` builds an incorrect RHS (mathematical bug)**  
`models.py` lines 766–779: the extended RHS is `[zeros(old_n), y_new]`, but the correct formulation for the enlarged kernel-ridge system `(K_new + λI) α = y_all` requires `y_all = [y_old, y_new]`. The old observations are zeroed out, so the re-solve forgets all previous targets. This makes `partial_fit` mathematically incorrect for any `forgetting_factor = 1.0` run.

---

**3. `partial_fit` raises `RuntimeError` at its own threshold — silently swallowed**  
Lines 738–748: when `total_new >= rebuild_threshold` the method raises a `RuntimeError` telling the caller to use `fit()`. However, `partial_fit_count` is already incremented _before_ the check, so after the first threshold breach every subsequent call also raises immediately, even if the user resets the data. There is also no documented way to reset this counter.

---

**4. `distributed_kernels.py` — silent overflow in `matvec`**  
Lines 141–142: `torch.exp(gram_block, out=gram_block)` bypasses `exp_safe`, so there is no clamping guard for the multi-GPU path. Large embedding norms will produce `inf` silently.

---

**5. `SparseKNNAttentionKernelOperator.diagonal` — returns exact diagonal, not sparse approximation**  
`kernels.py` lines 726–728: `diagonal()` computes `exp(||e_i||²)` (i.e. the _exact_ kernel diagonal) rather than the sum of sparse off-diagonal contributions, which is what `NystromAttentionKernelOperator.diagonal` and the `SKI` variant correctly do. The Jacobi preconditioner built from this diagonal is therefore inconsistent with the approximated operator, which can hurt PCG convergence.

---

**6. `LAKERRegressor.load` — incomplete reconstruction of `knn` and `ski` kernel operators**  
`models.py` lines 1469–1495: the `load()` classmethod reconstructs only `None`, `"nystrom"`, and `"rff"` kernels. Loading a model trained with `kernel_approx="knn"` or `kernel_approx="ski"` leaves `model.kernel_operator = None`, causing a silent `AttributeError` on the first `predict()` call.

---

**7. `PositionEmbedding.__init__` — global RNG mutation is thread-unsafe**  
`embeddings.py` lines 82–89: `torch.manual_seed(seed)` / `torch.set_rng_state(saved_state)` is not atomic under multi-threading. The module's own docstring warns about this, but the fix (using a local `Generator`) is not applied; only the Fourier part uses a local generator. This makes multi-threaded grid search or BO non-deterministic.

---

**8. `GPSurrogate.transform` — log-space clipping uses raw bound values**  
`utils.py` lines 113–116: the log clipping is `bounds[i,0] * 0.1` and `bounds[i,1] * 10`, which produces `0.0` for any bound that starts at zero (e.g. `gamma_bounds=(0.0, 2.0)`). `numpy.log10(0.0) = -inf`, corrupting the GP surrogate for gamma. The BO default bounds include `gamma_bounds=(0.0, 2.0)` and `log_indices=[0,1]` covers gamma.

---

### 🟠 Design Flaws

**9. `LAKERRegressor` is a 1500-line god class**  
`models.py` contains the regressor, grid search, Bayesian optimisation, regularisation path, streaming updates, learned embeddings, variance computation, serialisation, and condition number estimation. These are distinct responsibilities that should be separate classes or modules (e.g. `HyperparameterSearch`, `ModelPersistence`). The current structure makes unit testing, extension, and maintenance unnecessarily hard.

---

**10. `fit_with_search` / `fit_with_bo` mutate `self` during search then call `self.fit()` on full data**  
Lines 542–545 and 687–690: both methods overwrite `self.lambda_reg`, `self.gamma`, and `self.num_probes` as a side effect even if the caller wants to inspect the original hyperparameters. The returned `self` is fitted on full data, but the original model state is destroyed, breaking idempotency.

---

**11. Kernel operators lack a shared protocol / ABC**  
`KernelOperator` in `models.py` is defined as a `Union` type alias. There is no base class or `Protocol` enforcing that every operator implements `matvec`, `diagonal`, `to_dense`, `kernel_eval`, and `shape`. Adding a new operator is error-prone and cannot be type-checked statically. `mypy` cannot verify duck-typing correctness here.

---

**12. `exp_safe` in-place mutation of the input when `out is None`**  
`kernels.py` lines 43–47: when `out is None` and `requires_grad=False`, the function sets `out = gram` (the input itself) and calls `gram.clamp_()` in-place, mutating the caller's tensor. This is a silent aliasing bug — any code that passes `gram` and later reads it gets the clamped version.

---

**13. `backend.py` — module-level side effects on import**  
Lines 104–111: `set_default_device` and `set_default_dtype` are called at import time if `LAKER_DEVICE` / `LAKER_DTYPE` environment variables are set. These mutate global state, which interferes with test isolation (`conftest.py` sets no env cleanup) and violates the "no import-time side effects" principle for libraries.

---

**14. `matvec` dispatch in `AttentionKernelOperator` is redundant**  
`kernels.py` lines 140–144: both `x.dim() == 1` and `x.dim() == 2` call `self.matvec_impl(x)` identically. The `matvec_1d` and `matvec_2d` public methods then also just delegate to `matvec_impl`. These three public methods expose the same implementation and are dead surface area.

---

**15. `partial_fit` does not store training data (`x_old`, `y_old`)**  
The rebuild-threshold logic raises a `RuntimeError` telling the user to concatenate data externally (line 745). But since the model never stores `x_train` or `y_train`, there is no way for the caller to reconstruct the full dataset without external bookkeeping. This makes streaming truly unusable beyond a single batch.

---

### 🟡 Code Quality Issues

**16. Bare `except (RuntimeError, ValueError)` in `bo_eval` swallows tracebacks**  
`models.py` line 834: the BO evaluation silently returns `inf` for any error. Legitimate bugs (e.g. OOM, device mismatches) are suppressed and impossible to debug. At minimum log the exception at `DEBUG` level.

---

**17. `solve_1d` references `relative_residual` after the loop (undefined if `max_iter=0`)**  
`solvers.py` line 157: `relative_residual` is only assigned inside the loop. If `max_iter=0`, the `logger.warning` call will raise `UnboundLocalError`. This is an edge case but crashes production.

---

**18. `rhs_norm` type inconsistency in `PCG.solve`**  
`solvers.py` line 86: `rhs_norm` is a `torch.Tensor`, but on line 87 it is compared with `== 0` (integer). This will not raise but evaluates element-wise and the result is a 0-d boolean tensor, not a Python `bool`. Should be `rhs_norm.item() == 0`.

---

**19. `mypy` is configured with `disallow_untyped_defs = false` and `strict_optional = false`**  
`pyproject.toml` lines 76–78: these settings disable the most valuable mypy checks. Functions like `cmd_fit(args)` and `cmd_predict(args)` have untyped `args`, and `set_params(**params)` uses untyped dict values. The configuration hides real typing debt.

---

**20. `NystromAttentionKernelOperator.select_landmarks` is O(m·n) per iteration**  
`kernels.py` lines 400–408: the greedy landmark selection calls `torch.cdist(self.embeddings, selected)` inside a loop over `m` iterations, each computing distances to a growing set. For `m=200` and `n=100k` this is 200 full distance computations — slower than a single k-means++ pass. A vectorised version (maintain a `min_dist` buffer and update it) would be O(m·n) total.

---

**21. `data.py` uses `Tuple` from `typing` instead of the built-in `tuple`**  
`data.py` lines 4–5: `from typing import Tuple` is used while the project targets Python ≥ 3.9, which supports `tuple[...]` natively. Inconsistent with the rest of the codebase (e.g. `solvers.py` uses `tuple[...]` directly).

---

**22. `__main__.py` uses `argparse`, not `typer` or `click`**  
The project rules prefer `typer`. The `argparse` CLI lacks subcommand help for `--kernel-approx`, `--embedding-dtype`, and all the approximation-specific params (`--num-landmarks`, `--k-neighbors`, etc.), making the fit command a subset of what the Python API exposes.

---

**23. `GradientDescent.solve` uses `1e-16` fudge in `rel_res`**  
`solvers.py` line 281: `rel_res = self.residual_norm / (b_norm + 1e-16)` — the README says fudge factors were removed, but this one remains, and it's inconsistent with the PCG implementation.

---

**24. Coverage XML committed to git**  
`coverage.xml` (69 KB) is checked into the repository root. This is a build artifact that belongs in `.gitignore` and should not be versioned.

---

### 🔵 Gaps & Missing Features

**25. No `__all__` guard or kernel-operator ABC prevents accidental imports**  

**26. No cross-validation support** — `fit_with_search` uses a single random train/val split, not k-fold, so the selected hyperparameters can be highly variance-prone for small datasets.

**27. Preconditioner serialisation is missing** — `save()`/`load()` stores `embeddings` and `alpha` but not the built preconditioner (`q_basis`, `isotropic_coef`, etc.). Every `load()` call requires a full preconditioner rebuild before `predict_variance()` works.

**28. No `score_r2` or standard regression metrics** — `score()` returns negative RMSE. sklearn expects `R²` by default; the current signature will silently produce wrong results in sklearn's `GridSearchCV`.

**29. No seed control in `fit`** — random probes in CCCP (`torch.randn` on line 119 of `preconditioner.py`) are not seeded, so two calls to `fit()` on identical data may not converge to the same preconditioner.

---

## Feature Suggestions

Ordered by estimated impact vs. effort.

| # | Feature | Impact | Effort |
|---|---|---|---|
| 1 | **K-fold cross-validation in `fit_with_search`** | High | Low |
| 2 | **Kernel operator ABC / Protocol** with mypy enforcement | High | Low |
| 3 | **Seed parameter for `fit()`** (propagated to CCCP probes) | High | Low |
| 4 | **`score_r2()` and `score_rmse()` metrics** for full sklearn compatibility | Medium | Low |
| 5 | **Preconditioner persistence** in `save`/`load` | High | Medium |
| 6 | **Fix `knn` / `ski` kernel reconstruction in `load()`** | High | Low |
| 7 | **Streaming `partial_fit` with stored data buffer** (ring buffer or user-managed store) | High | Medium |
| 8 | **`torch.compile` integration** via `maybe_compile` (already in `backend.py` but unused) | Medium | Low |
| 9 | **Sparse CSR backend for k-NN on CPU** (`torch.sparse_csr`) instead of COO | Medium | Medium |
| 10 | **Multi-output regression** (`y` of shape `(n, k)`) — PCG already handles 2-D RHS | High | Medium |
| 11 | **Automatic embedding dimension selection** via elbow on validation RMSE | Medium | Medium |
| 12 | **`torchscript`-compatible kernel operators** for deployment | Medium | High |
| 13 | **Async / concurrent grid search** using `concurrent.futures` or `torch.multiprocessing` | Medium | Medium |
| 14 | **All-reduce distributed matvec** (true model parallelism, not data replication) | High | High |
| 15 | **Nyström landmark caching across `fit_path` iterations** | Medium | Low |
| 16 | **`fit_with_cv` using sklearn's `KFold` integration** for drop-in pipeline support | Medium | Medium |
| 17 | **Confidence-interval prediction via conformal prediction** wrappers | Medium | High |
| 18 | **ONNX export** for the embedding + kernel predict path | Low | High |
| 19 | **Callback / hook API for training progress** (replace verbose logging) | Low | Medium |
| 20 | **`laker evaluate` CLI subcommand** (load model + data, print metrics) | Low | Low |
