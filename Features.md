If these remain, improvements will be noisy and hard to trust.

  High-impact algorithm upgrades

  1. Learnable embedding network upgrade.

  - Replace static embedding with small residual MLP + Fourier features + normalization.
  - Add contrastive neighborhood loss to preserve local geometry before kernel regression loss.
  - Train with two-stage schedule: embedding pretrain, then joint fine-tune with implicit gradients.

  2. Adaptive kernel temperature.

  - Add learnable inverse temperature tau: K = exp((E E^T)/tau).
  - Use per-layer or per-region temperature (mixture of experts) to avoid over-smoothing.
  - This often sharply improves bias-variance tradeoff.

  3. Multi-kernel mixture.

  - Combine exact attention kernel with local kernels (RBF/Matérn-like surrogate in embedding space).
  - Learn convex mixture weights on validation loss.
  - Helps both global structure and local fidelity.

  4. Better preconditioner family.

  - Keep CCCP, but add fallback/ensemble preconditioners: low-rank deflation + Jacobi + block diagonal.
  - Auto-select preconditioner by predicted condition number and runtime budget.
  - Cache/reuse preconditioners along hyperparameter path (lambda continuation).

  5. Local-global hybrid operator.

  - Use sparse local graph kernel for high-frequency detail + low-rank/global component for long-range correlations.
  - This is usually much better than pure exact or pure low-rank on real spatial fields.

  6. Hyperparameter optimization that matters.

  - Optimize embedding_dim, tau, lambda, gamma, num_probes, and kernel mix weights jointly.
  - Use log-scale BO with multi-fidelity early stopping.
  - Track Pareto frontier: RMSE vs latency vs memory.

  Numerical and optimization improvements

  1. Use mixed precision only where safe.

  - Keep solver/preconditioner in float64 when condition number is high.
  - Use autocast for embedding forward only.

  2. Add continuation training.

  - Start with high lambda, gradually reduce.
  - Warm-start PCG and preconditioner each stage.

  3. Add robust loss option.

  - Huber or Tukey loss for noisy/outlier measurements.
  - Can materially improve field reconstruction quality.

  Evaluation protocol for real gains

  1. Add hard benchmarks.

  - OOD spatial splits, missing-region interpolation, sensor dropout stress tests.

  2. Measure more than RMSE.

  - MAE, worst-cell error, calibration error for variance, condition number, solve time.

  3. Run ablations.

  - One change at a time; keep a reproducible leaderboard.