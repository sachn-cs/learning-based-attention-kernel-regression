# LAKER Deferred Work — TODO

This document captures the items that were **deferred** from the `Review.md` and `upgrades.md` roadmaps. Each entry explains what it is, why it matters, and what unblocks it.

---

## 1. Bilevel Hyperparameter Learning via Implicit Differentiation

**What:** Instead of grid search or Bayesian optimisation, compute hypergradients of the validation loss with respect to hyperparameters (`lambda`, `gamma`, temperature, etc.) by differentiating through the linear solve `(K + λI) α = y`.

**Why required:** Grid/BO are sample-inefficient and scale poorly with the number of hyperparameters. Bilevel learning can jointly optimise regularisation, kernel bandwidth, embedding dimension, and even preconditioner parameters with far fewer objective evaluations. For LAKER this is especially important because the CCCP preconditioner itself has hyperparameters (`gamma`, `num_probes`) that currently require expensive outer loops.

**Blocked by / complexity:**
- Requires implementing implicit Jacobian-vector products through PCG.
- PyTorch `torch.autograd.functional.jvp` does not support iterative solvers natively; we need a custom backward that uses the conjugate gradient theorem ( differentiate through the fixed-point `α*`).
- Need to ensure kernel operators support autograd through `matvec` for all approximation paths (Nyström, SKI, sparse k-NN).
- Estimated effort: high (2–3 weeks).

---

## 2. Spectral-Shaped Attention Kernel

**What:** Replace the plain `exp(EE^T)` kernel with a spectral decomposition:
```
K = U exp(g(Λ)) U^T
```
where `EE^T = U Λ U^T` and `g` is a learned monotone spline that reshapes the spectrum.

**Why required:** Direct control over the spectrum improves conditioning (which directly impacts PCG iteration count) and injects an inductive bias that can improve both fit quality and solver behaviour. A badly conditioned kernel is the primary reason LAKER needs a learned preconditioner; shaping the spectrum could reduce or eliminate that need.

**Blocked by / complexity:**
- Exact eigendecomposition of `EE^T` is `O(n^3)` — only feasible for `n < ~5k`.
- For large `n`, need a randomized or Nyström-based spectral approximation first, then learn `g` on the approximate spectrum. This couples tightly with the Nyström path.
- The spline `g` must be guaranteed monotone to keep `K` PSD; implementing a monotone spline in PyTorch with correct gradients is non-trivial.
- Estimated effort: high (1–2 weeks).

---

## 3. Uncertainty-Aware Training Objective

**What:** Train embeddings and kernel parameters with a negative log-likelihood loss that includes a calibration penalty:
```
L = NLL(y | μ, σ^2) + β · calibration_penalty
```
using the existing `predict_variance` path.

**Why required:** LAKER currently minimises RMSE, which can produce overconfident predictions far from training data. In radio-map estimation (the primary application), knowing *where* the model is uncertain is critical for active sensing and sensor placement. The calibration penalty ensures that predicted variance actually matches empirical error.

**Blocked by / complexity:**
- `predict_variance` is already implemented but relies on the preconditioner being invertible; the variance formula `σ^2 = λ + k^T (K + λI)^{-1} k` requires a second linear solve per prediction point.
- The calibration penalty (e.g., expected calibration error or a proper scoring rule) needs batch-wise variance computation, which is expensive for large `n`.
- Need to modify `PositionEmbedding` training loop to backprop through the kernel and the variance head.
- Estimated effort: high (1–2 weeks).

---

## 4. Residual Corrector on Top of Kernel Solve

**What:** After the base LAKER prediction `ŷ_LAKER`, learn a small residual model:
```
ŷ = ŷ_LAKER + r_ψ(x)
```
where `r_ψ` is a tiny MLP (e.g., 2 layers, 32 hidden units) with strong regularisation.

**Why required:** Kernel ridge regression has a structural bias towards smooth functions. A residual corrector captures local misspecification (sharp edges, non-stationary clutter in radio maps) without destabilising the core linear solver. This is a cheap way to boost accuracy on structured errors.

**Blocked by / complexity:**
- Requires storing `(x_train, y_train, ŷ_train)` and training a second network.
- Must be careful not to overfit the residual; needs early stopping and L2 regularisation.
- The training loop for `r_ψ` is independent of LAKER but should share the embedding model to avoid redundant forward passes.
- Estimated effort: medium (3–5 days).

---

## 5. Adaptive Probe Distribution for CCCP

**What:** Replace the current Gaussian i.i.d. probe vectors in `CCCPPreconditioner.build()` with spectrum-aware probes: power-iteration-biased directions plus orthogonalised blocks.

**Why required:** CCCP preconditioner quality depends on how well the probe set captures the dominant eigenspaces of `K + λI`. Random Gaussian probes waste directions on already-well-conditioned subspaces. Power-iteration-biased probes (a few steps of Lanczos on random starts) concentrate mass on the ill-conditioned subspace, giving stronger preconditioners with the same `num_probes`.

**Blocked by / complexity:**
- Power iteration requires repeated `matvec` calls, which is already available but adds overhead during preconditioner construction.
- Orthogonalisation (e.g., Gram-Schmidt) is needed to prevent probe collapse; implementing stable blocked orthogonalisation in PyTorch is straightforward but needs careful memory management.
- The current CCCP update assumes isotropic Gaussian probes; biasing them breaks some of the analytical expectations in the CCCP derivation, so the step-size logic may need re-derivation.
- Estimated effort: medium (3–5 days).

---

## 6. Refactor `LAKERRegressor` into Smaller Classes

**What:** `models.py` is a 1500-line god class that currently handles fitting, grid search, Bayesian optimisation, regularisation paths, streaming updates, learned embeddings, variance computation, serialisation, and condition-number estimation. Split into focused classes: `HyperparameterSearch`, `ModelPersistence`, `StreamingUpdater`, etc.

**Why required:** The current structure makes unit testing, extension, and maintenance unnecessarily hard. Any change to hyperparameter search risks touching the core solver logic. Separating concerns would allow independent testing of the BO surrogate, the streaming update policy, and the save/load format.

**Blocked by / complexity:**
- This is a major structural refactor that touches every public method and all tests.
- Must preserve the scikit-learn API (`fit`, `predict`, `score`, `save`, `load`) while moving internals into helper classes.
- Risk of merge conflicts with any other active work on `models.py`.
- Estimated effort: medium–high (1 week).

---

## How to prioritise

If capacity opens up, the recommended order is:

1. **Adaptive Probe Distribution** — highest ROI for preconditioner quality; relatively self-contained.
2. **Residual Corrector** — quick accuracy win; additive and safe.
3. **Refactor `LAKERRegressor`** — unlocks velocity for everything else.
4. **Bilevel Hyperparameter Learning** — biggest long-term win but highest cost.
5. **Spectral-Shaped Kernel** — depends on refactor and Nyström improvements.
6. **Uncertainty-Aware Training** — depends on variance path being solid and fast.
