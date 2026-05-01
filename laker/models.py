"""High-level sklearn-compatible LAKER estimator."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Union, cast

import numpy
import torch
import torch.nn as nn

from laker.backend import get_default_device, get_default_dtype, to_tensor
from laker.distributed_kernels import DistributedAttentionKernelOperator
from laker.embeddings import PositionEmbedding
from laker.kernels import (
    AttentionKernelOperator,
    NystromAttentionKernelOperator,
    RandomFeatureAttentionKernelOperator,
    SKIAttentionKernelOperator,
    SparseKNNAttentionKernelOperator,
    exp_safe,
)
from laker.preconditioner import CCCPPreconditioner
from laker.solvers import PreconditionedConjugateGradient

logger = logging.getLogger(__name__)

KernelOperator = Union[
    AttentionKernelOperator,
    NystromAttentionKernelOperator,
    RandomFeatureAttentionKernelOperator,
    SparseKNNAttentionKernelOperator,
    SKIAttentionKernelOperator,
    DistributedAttentionKernelOperator,
]


class LAKERRegressor:
    """Learning-based Attention Kernel Regression estimator.

    Fits the regularised attention kernel regression problem

    .. math::
        \\min_\\alpha \\|G \\alpha - y\\|_2^2 + \\lambda \\alpha^\\top G \\alpha

    where ``G = exp(E E^T)`` is an exponential attention kernel induced by
    learned embeddings ``E``. The linear system ``(lambda I + G) alpha = y``
    is solved efficiently using a learned CCCP preconditioner inside PCG.

    The estimator follows the ``scikit-learn`` API with ``fit`` and
    ``predict`` methods.

    Args:
        embedding_dim: Dimension ``d_e`` of the learned embeddings (default 10).
        lambda_reg: Tikhonov regularisation ``lambda`` (default 1e-2).
        gamma: CCCP regularisation parameter (default 1e-1).
        num_probes: Number of random directions ``N_r`` for preconditioner
            learning. ``None`` selects an adaptive heuristic.
        epsilon: Numerical safeguard in CCCP update (default 1e-8).
        base_rho: Base shrinkage parameter (default 0.05).
        cccp_max_iter: Maximum CCCP iterations (default 200).
        cccp_tol: CCCP convergence tolerance (default 1e-6).
        pcg_tol: PCG residual tolerance (default 1e-10).
        pcg_max_iter: Maximum PCG iterations (default 1000).
        chunk_size: Chunk size for matrix-free kernel matvecs. ``None``
            materialises the full kernel for ``n <= 5000``.
        embedding_module: Optional custom ``nn.Module`` mapping locations to
            embeddings. If ``None``, ``PositionEmbedding`` is used.
        kernel_approx: Low-rank kernel approximation. ``None`` uses exact kernel.
            ``"nystrom"`` uses Nyström approximation. ``"rff"`` uses random
            Fourier features.
        num_landmarks: Number of landmarks for Nyström (default adaptive).
        num_features: Number of random Fourier features (default adaptive).
        embedding_dtype: Dtype for embedding generation. If different from
            ``dtype``, embeddings are computed in ``embedding_dtype`` then cast
            to ``dtype`` for the kernel. This enables mixed-precision training.
        device: torch device (``"cuda"``, ``"cpu"``, etc.).
        dtype: torch dtype (``torch.float32`` or ``torch.float64``).
        verbose: Whether to log training progress.

    Raises:
        ValueError: If any hyperparameter is out of its valid range.
    """

    def __init__(
        self,
        embedding_dim: int = 10,
        lambda_reg: float = 1e-2,
        gamma: float = 1e-1,
        num_probes: Optional[int] = None,
        epsilon: float = 1e-8,
        base_rho: float = 0.05,
        cccp_max_iter: int = 200,
        cccp_tol: float = 1e-6,
        pcg_tol: float = 1e-6,
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
    ) -> None:
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if lambda_reg <= 0:
            raise ValueError(f"lambda_reg must be positive, got {lambda_reg}")
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if not 0 <= base_rho <= 1:
            raise ValueError(f"base_rho must be in [0, 1], got {base_rho}")
        if cccp_max_iter <= 0:
            raise ValueError(f"cccp_max_iter must be positive, got {cccp_max_iter}")
        if cccp_tol <= 0:
            raise ValueError(f"cccp_tol must be positive, got {cccp_tol}")
        if pcg_tol <= 0:
            raise ValueError(f"pcg_tol must be positive, got {pcg_tol}")
        if pcg_max_iter <= 0:
            raise ValueError(f"pcg_max_iter must be positive, got {pcg_max_iter}")
        if kernel_approx not in (None, "nystrom", "rff", "knn", "ski"):
            raise ValueError(
                "kernel_approx must be None, 'nystrom', 'rff', 'knn', or 'ski', "
                f"got {kernel_approx}"
            )
        if k_neighbors is not None and k_neighbors <= 0:
            raise ValueError(f"k_neighbors must be positive, got {k_neighbors}")
        if grid_size is not None and grid_size < 2:
            raise ValueError(f"grid_size must be at least 2, got {grid_size}")

        self.embedding_dim = embedding_dim
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.num_probes = num_probes
        self.epsilon = epsilon
        self.base_rho = base_rho
        self.cccp_max_iter = cccp_max_iter
        self.cccp_tol = cccp_tol
        self.pcg_tol = pcg_tol
        self.pcg_max_iter = pcg_max_iter
        self.chunk_size = chunk_size
        self.embedding_module = embedding_module
        self.kernel_approx = kernel_approx
        self.num_landmarks = num_landmarks
        self.num_features = num_features
        self.k_neighbors = k_neighbors
        self.grid_size = grid_size
        self.distributed = distributed
        self.verbose = verbose

        if device is None:
            device = get_default_device()
        elif isinstance(device, str):
            device = torch.device(device)
        if dtype is None:
            dtype = get_default_dtype()
        self.device = device
        self.dtype = dtype
        self.embedding_dtype = embedding_dtype if embedding_dtype is not None else dtype

        # Fitted attributes
        self.embeddings: Optional[torch.Tensor] = None
        self.alpha: Optional[torch.Tensor] = None
        self.preconditioner: Optional[CCCPPreconditioner] = None
        self.kernel_operator: Optional[KernelOperator] = None
        self.embedding_model: Optional[nn.Module] = None

    def compute_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Build or reuse embeddings from spatial locations."""
        n = x.shape[0]
        input_dim = x.shape[1]

        if self.verbose:
            logger.info("Fitting LAKER on n=%d, dx=%d", n, input_dim)

        embedded_input = x.to(dtype=self.embedding_dtype)
        if self.embedding_module is not None:
            self.embedding_model = self.embedding_module.to(self.device)
            with torch.no_grad():
                embeddings = self.embedding_model(embedded_input)
        else:
            self.embedding_model = PositionEmbedding(
                input_dim=input_dim,
                embedding_dim=self.embedding_dim,
                device=self.device,
                dtype=self.embedding_dtype,
            )
            with torch.no_grad():
                embeddings = self.embedding_model(embedded_input)

        if self.embedding_dtype != self.dtype:
            embeddings = embeddings.to(dtype=self.dtype)
            if self.verbose:
                logger.info(
                    "Mixed-precision: embeddings computed in %s, cast to %s for solver",
                    self.embedding_dtype,
                    self.dtype,
                )
        return embeddings

    def build_kernel_operator(
        self,
        embeddings: torch.Tensor,
        lambda_reg: Optional[float] = None,
        chunk_size: Optional[int] = None,
    ) -> KernelOperator:
        """Construct the kernel operator for given embeddings."""
        n = embeddings.shape[0]
        lambda_value = float(lambda_reg) if lambda_reg is not None else self.lambda_reg
        chunk_size_local = chunk_size
        if chunk_size_local is None and n > 5000:
            chunk_size_local = max(1024, min(n // 10, 8192))
            if self.verbose:
                logger.info("Auto-selected chunk_size=%d for n=%d", chunk_size_local, n)

        operator: KernelOperator
        if self.kernel_approx is None:
            operator = AttentionKernelOperator(
                embeddings=embeddings,
                lambda_reg=lambda_value,
                chunk_size=chunk_size_local,
                device=self.device,
                dtype=self.dtype,
            )
        elif self.kernel_approx == "nystrom":
            operator = NystromAttentionKernelOperator(
                embeddings=embeddings,
                lambda_reg=lambda_value,
                num_landmarks=self.num_landmarks,
                chunk_size=chunk_size_local,
                device=self.device,
                dtype=self.dtype,
            )
            if self.verbose:
                logger.info(
                    "Using Nyström approximation with m=%d landmarks",
                    cast(NystromAttentionKernelOperator, operator).m,
                )
        elif self.kernel_approx == "rff":
            operator = RandomFeatureAttentionKernelOperator(
                embeddings=embeddings,
                lambda_reg=lambda_value,
                num_features=self.num_features,
                device=self.device,
                dtype=self.dtype,
            )
            if self.verbose:
                logger.info(
                    "Using RFF approximation with r=%d features",
                    cast(RandomFeatureAttentionKernelOperator, operator).num_features,
                )
        elif self.kernel_approx == "knn":
            operator = SparseKNNAttentionKernelOperator(
                embeddings=embeddings,
                lambda_reg=lambda_value,
                k_neighbors=self.k_neighbors,
                chunk_size=chunk_size_local,
                device=self.device,
                dtype=self.dtype,
            )
            if self.verbose:
                logger.info(
                    "Using sparse k-NN approximation with k=%d neighbours",
                    cast(SparseKNNAttentionKernelOperator, operator).k_neighbors,
                )
        elif self.kernel_approx == "ski":
            operator = SKIAttentionKernelOperator(
                embeddings=embeddings,
                lambda_reg=lambda_value,
                grid_size=self.grid_size,
                device=self.device,
                dtype=self.dtype,
            )
            if self.verbose:
                logger.info(
                    "Using SKI approximation with %d grid points",
                    cast(SKIAttentionKernelOperator, operator).grid_points.shape[0],
                )
        elif self.distributed and self.kernel_approx is None:
            operator = DistributedAttentionKernelOperator(
                embeddings=embeddings,
                lambda_reg=lambda_value,
                master_device=self.device,
                dtype=self.dtype,
            )
            if self.verbose:
                logger.info(
                    "Using distributed kernel on %d device(s)",
                    len(cast(DistributedAttentionKernelOperator, operator).devices),
                )
        else:
            raise ValueError(f"Unknown kernel_approx={self.kernel_approx}")
        return operator

    def build_preconditioner(
        self,
        matvec: Callable[[torch.Tensor], torch.Tensor],
        n: int,
        gamma: Optional[float] = None,
        num_probes: Optional[int] = None,
    ) -> CCCPPreconditioner:
        """Learn the CCCP preconditioner for a given matvec."""
        preconditioner = CCCPPreconditioner(
            num_probes=num_probes if num_probes is not None else self.num_probes,
            gamma=gamma if gamma is not None else self.gamma,
            epsilon=self.epsilon,
            base_rho=self.base_rho,
            max_iter=self.cccp_max_iter,
            tol=self.cccp_tol,
            verbose=self.verbose,
            device=self.device,
            dtype=self.dtype,
        )
        preconditioner.build(matvec, n)
        return preconditioner

    def solve_pcg(
        self,
        kernel_operator: KernelOperator,
        preconditioner: CCCPPreconditioner,
        rhs: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, int]:
        """Solve (K + lambda I) alpha = rhs with PCG."""
        pcg = PreconditionedConjugateGradient(
            tol=self.pcg_tol,
            max_iter=self.pcg_max_iter,
            verbose=self.verbose,
        )
        alpha = pcg.solve(
            operator=kernel_operator.matvec,
            preconditioner=preconditioner.apply,
            rhs=rhs,
            x0=x0,
        )
        if self.verbose:
            final_res = (
                torch.linalg.norm(kernel_operator.matvec(alpha) - rhs).item()
                / torch.linalg.norm(rhs).item()
            )
            logger.info(
                "LAKER fit complete: PCG iters=%d, final rel_res=%.3e",
                pcg.iterations,
                final_res,
            )
        return alpha, pcg.iterations

    def fit(
        self,
        x: Union[torch.Tensor, "numpy.ndarray"],
        y: Union[torch.Tensor, "numpy.ndarray"],
        x0: Optional[torch.Tensor] = None,
    ) -> "LAKERRegressor":
        """Fit the LAKER model to sparse measurements.

        Args:
            x: Measurement locations of shape ``(n, dx)``.
            y: Noisy observations of shape ``(n,)`` or ``(n, 1)``.
            x0: Optional warm-start for PCG initial guess.

        Returns:
            ``self`` for method chaining.

        Raises:
            ValueError: If ``x`` is not 2-D or ``y`` is not 1-D.
        """
        x = to_tensor(x, device=self.device, dtype=self.dtype)
        y = to_tensor(y, device=self.device, dtype=self.dtype).squeeze()

        if x.dim() != 2:
            raise ValueError(f"x must be 2-D, got shape {x.shape}")
        if y.dim() != 1:
            raise ValueError(f"y must be 1-D, got shape {y.shape}")

        self.embeddings = self.compute_embeddings(x)
        self.kernel_operator = self.build_kernel_operator(self.embeddings)
        self.preconditioner = self.build_preconditioner(
            self.kernel_operator.matvec, self.embeddings.shape[0]
        )
        self.alpha, self.pcg_iterations_ = self.solve_pcg(
            self.kernel_operator, self.preconditioner, y, x0=x0
        )
        return self

    def fit_with_search(
        self,
        x: Union[torch.Tensor, "numpy.ndarray"],
        y: Union[torch.Tensor, "numpy.ndarray"],
        val_fraction: float = 0.2,
        lambda_reg_grid: Optional[list[float]] = None,
        gamma_grid: Optional[list[float]] = None,
        num_probes_grid: Optional[list[int]] = None,
        warm_start: bool = True,
    ) -> "LAKERRegressor":
        """Fit with validation-based grid search over key hyperparameters.

        Splits ``(x, y)`` into train/validation, trains a model for each
        hyperparameter combination, and selects the one with the lowest
        validation RMSE. The selected model is then retrained on the full
        dataset.

        Embeddings are computed once and reused across trials, giving a 3-5x
        speedup over naive per-trial ``fit()`` calls.

        Args:
            x: Measurement locations of shape ``(n, dx)``.
            y: Noisy observations of shape ``(n,)``.
            val_fraction: Fraction held out for validation.
            lambda_reg_grid: Grid values for ``lambda_reg``. Defaults to
                ``[1e-3, 1e-2, 1e-1]``.
            gamma_grid: Grid values for ``gamma``. Defaults to
                ``[0.0, 1e-1, 1.0]``.
            num_probes_grid: Grid values for ``num_probes``. Defaults to
                ``[50, 100, 200]``.
            warm_start: If ``True``, pass the previous best ``alpha`` as the
                PCG initial guess for the next trial.

        Returns:
            ``self`` fitted with the best hyperparameters.
        """
        x = to_tensor(x, device=self.device, dtype=self.dtype)
        y = to_tensor(y, device=self.device, dtype=self.dtype).squeeze()

        if x.dim() != 2:
            raise ValueError(f"x must be 2-D, got shape {x.shape}")
        if y.dim() != 1:
            raise ValueError(f"y must be 1-D, got shape {y.shape}")

        n = x.shape[0]
        n_val = max(1, int(n * val_fraction))
        indices = torch.randperm(n, device=self.device)
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]

        if lambda_reg_grid is None:
            lambda_reg_grid = [1e-3, 1e-2, 1e-1]
        if gamma_grid is None:
            gamma_grid = [0.0, 1e-1, 1.0]
        if num_probes_grid is None:
            num_probes_grid = [50, 100, 200]

        # Compute embeddings once from all x (deterministic, no y leakage)
        full_embeddings = self.compute_embeddings(x)
        train_embeddings = full_embeddings[train_idx]

        best_rmse = float("inf")
        best_params = {}
        best_alpha: Optional[torch.Tensor] = None

        if self.verbose:
            logger.info(
                "Starting grid search: %d lambda x %d gamma x %d probes = %d configs",
                len(lambda_reg_grid),
                len(gamma_grid),
                len(num_probes_grid),
                len(lambda_reg_grid) * len(gamma_grid) * len(num_probes_grid),
            )

        x0 = None
        for lambda_reg_value in lambda_reg_grid:
            for gamma_value in gamma_grid:
                for num_probes_value in num_probes_grid:
                    try:
                        kernel_op = self.build_kernel_operator(
                            train_embeddings, lambda_reg=lambda_reg_value
                        )
                        precond = self.build_preconditioner(
                            kernel_op.matvec,
                            train_embeddings.shape[0],
                            gamma=gamma_value,
                            num_probes=num_probes_value,
                        )
                        alpha, _ = self.solve_pcg(
                            kernel_op,
                            precond,
                            y[train_idx],
                            x0=x0 if warm_start else None,
                        )
                        # Validation prediction
                        x_val_embed = x[val_idx].to(dtype=self.embedding_dtype)
                        with torch.no_grad():
                            val_embeddings = self.embedding_model(x_val_embed)
                        if self.embedding_dtype != self.dtype:
                            val_embeddings = val_embeddings.to(dtype=self.dtype)
                        k_val = kernel_op.kernel_eval(val_embeddings, train_embeddings)
                        y_val_pred = k_val @ alpha
                        rmse = torch.sqrt(torch.mean((y_val_pred - y[val_idx]) ** 2)).item()
                    except (RuntimeError, ValueError) as exc:
                        rmse = float("inf")
                        if self.verbose:
                            logger.warning(
                                "Trial failed: lambda=%.3e gamma=%.3e probes=%d (%s)",
                                lambda_reg_value,
                                gamma_value,
                                num_probes_value,
                                exc,
                            )

                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_params = {
                            "lambda_reg": lambda_reg_value,
                            "gamma": gamma_value,
                            "num_probes": num_probes_value,
                        }
                        best_alpha = alpha
                        if self.verbose:
                            logger.info(
                                "New best: lambda=%.3e gamma=%.3e probes=%d val_rmse=%.4f",
                                lambda_reg_value,
                                gamma_value,
                                num_probes_value,
                                rmse,
                            )

                    if warm_start and best_alpha is not None:
                        x0 = best_alpha.clone()

        if not best_params:
            raise RuntimeError(
                "Grid search failed: all parameter combinations diverged or raised errors. "
                "Try widening lambda_reg_grid, increasing pcg_max_iter, "
                "or using dtype=torch.float64."
            )

        if self.verbose:
            logger.info(
                "Best hyperparameters: lambda_reg=%.3e gamma=%.3e num_probes=%d",
                best_params["lambda_reg"],
                best_params["gamma"],
                best_params["num_probes"],
            )

        # Apply best params and fit on full data
        self.lambda_reg = float(best_params["lambda_reg"])
        self.gamma = float(best_params["gamma"])
        self.num_probes = int(best_params["num_probes"])
        return self.fit(x, y)

    def fit_with_bo(
        self,
        x: Union[torch.Tensor, "numpy.ndarray"],
        y: Union[torch.Tensor, "numpy.ndarray"],
        val_fraction: float = 0.2,
        n_calls: int = 15,
        n_initial_points: int = 5,
        lambda_reg_bounds: tuple[float, float] = (1e-4, 1.0),
        gamma_bounds: tuple[float, float] = (0.0, 2.0),
        num_probes_bounds: tuple[int, int] = (20, 300),
    ) -> "LAKERRegressor":
        """Fit with Bayesian Optimisation over key hyperparameters.

        Uses a lightweight Gaussian Process surrogate on a log-scale search
        space with Expected Improvement acquisition.  Typically requires
        10--15 evaluations vs 27 for a full 3x3x3 grid search.

        Args:
            x: Measurement locations of shape ``(n, dx)``.
            y: Noisy observations of shape ``(n,)``.
            val_fraction: Fraction held out for validation.
            n_calls: Total BO iterations (including initial random points).
            n_initial_points: Number of random Latin-hypercube samples before
                using the GP surrogate.
            lambda_reg_bounds: ``(min, max)`` for ``lambda_reg``.
            gamma_bounds: ``(min, max)`` for ``gamma``.
            num_probes_bounds: ``(min, max)`` for ``num_probes``.

        Returns:
            ``self`` fitted with the best hyperparameters.
        """
        import numpy as np

        from laker.utils import GPSurrogate

        x = to_tensor(x, device=self.device, dtype=self.dtype)
        y = to_tensor(y, device=self.device, dtype=self.dtype).squeeze()

        if x.dim() != 2:
            raise ValueError(f"x must be 2-D, got shape {x.shape}")
        if y.dim() != 1:
            raise ValueError(f"y must be 1-D, got shape {y.shape}")

        n = x.shape[0]
        n_val = max(1, int(n * val_fraction))
        indices = torch.randperm(n, device=self.device)
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]

        # Compute embeddings once
        self.embeddings = self.compute_embeddings(x)
        train_embeddings = self.embeddings[train_idx]

        bounds = np.array([lambda_reg_bounds, gamma_bounds, num_probes_bounds], dtype=np.float64)

        # Latin-hypercube initial design
        def lh_sample(n_samp: int) -> np.ndarray:
            """Latin-hypercube sampling in [0,1]^d, then map to bounds."""
            d = bounds.shape[0]
            samples = np.zeros((n_samp, d), dtype=np.float64)
            for i in range(d):
                perm = np.random.permutation(n_samp)
                samples[:, i] = (perm + np.random.uniform(size=n_samp)) / n_samp
            # Map to actual bounds
            return samples * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

        X_obs = []
        y_obs = []

        for _ in range(n_initial_points):
            point = lh_sample(1)[0]
            lambda_reg_value, gamma_value, num_probes_value = (
                float(point[0]),
                float(point[1]),
                int(round(float(point[2]))),
            )
            rmse = self.bo_eval(
                train_embeddings,
                x[val_idx],
                y,
                train_idx,
                val_idx,
                lambda_reg_value,
                gamma_value,
                num_probes_value,
            )
            X_obs.append(point)
            y_obs.append(rmse)

        gp = GPSurrogate(bounds, log_indices=[0, 1])
        best_rmse = min(y_obs)
        best_arr = X_obs[numpy.argmin(y_obs)]
        best_params = {
            "lambda_reg": float(best_arr[0]),
            "gamma": float(best_arr[1]),
            "num_probes": int(round(float(best_arr[2]))),
        }

        for _ in range(n_calls - n_initial_points):
            gp.fit(np.vstack(X_obs), np.array(y_obs, dtype=np.float64))

            # Acquisition: random search over 500 candidates
            candidates = lh_sample(500)
            ei = gp.expected_improvement(candidates)
            next_point = candidates[np.argmax(ei)]

            lambda_reg_value, gamma_value, num_probes_value = (
                next_point[0],
                next_point[1],
                int(round(next_point[2])),
            )
            rmse = self.bo_eval(
                train_embeddings,
                x[val_idx],
                y,
                train_idx,
                val_idx,
                lambda_reg_value,
                gamma_value,
                num_probes_value,
            )
            X_obs.append(next_point)
            y_obs.append(rmse)

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = {
                    "lambda_reg": lambda_reg_value,
                    "gamma": gamma_value,
                    "num_probes": num_probes_value,
                }

        if self.verbose:
            logger.info(
                "Best BO hyperparameters: lambda_reg=%.3e gamma=%.3e num_probes=%d",
                best_params["lambda_reg"],
                best_params["gamma"],
                best_params["num_probes"],
            )

        self.lambda_reg = float(best_params["lambda_reg"])
        self.gamma = float(best_params["gamma"])
        self.num_probes = int(best_params["num_probes"])
        return self.fit(x, y)

    def partial_fit(
        self,
        x_new: Union[torch.Tensor, "numpy.ndarray"],
        y_new: Union[torch.Tensor, "numpy.ndarray"],
        forgetting_factor: float = 1.0,
        rebuild_threshold: int = 100,
    ) -> "LAKERRegressor":
        """Update the model with one or more new observations.

        Uses an incremental update: the new ``alpha`` is obtained by solving
        the enlarged system with the previous ``alpha`` as a warm-start.
        For very large ``n`` this is cheaper than a full refit because PCG
        converges quickly from a good initial guess.

        Args:
            x_new: New measurement locations of shape ``(m, dx)``.
            y_new: New observations of shape ``(m,)``.
            forgetting_factor: Multiplier applied to previous ``alpha`` before
                the update.  ``< 1.0`` downweights old data (exponential forgetting).
            rebuild_threshold: If the cumulative number of new points exceeds
                this threshold, rebuild embeddings, preconditioner, and solve
                from scratch.

        Returns:
            ``self`` for method chaining.

        Raises:
            RuntimeError: If the model has not been previously fitted.
        """
        if self.alpha is None or self.embeddings is None:
            raise RuntimeError("Model has not been fitted. Call fit() before partial_fit().")

        x_new = to_tensor(x_new, device=self.device, dtype=self.dtype)
        y_new = to_tensor(y_new, device=self.device, dtype=self.dtype).squeeze()

        if x_new.dim() != 2:
            raise ValueError(f"x_new must be 2-D, got shape {x_new.shape}")
        if y_new.dim() != 1:
            raise ValueError(f"y_new must be 1-D, got shape {y_new.shape}")

        m = x_new.shape[0]

        # Track total new points since last full rebuild
        total_new = getattr(self, "partial_fit_count", 0) + m
        self.partial_fit_count = total_new

        if total_new >= rebuild_threshold:
            # Full rebuild: concatenate all data and refit
            if self.verbose:
                logger.info("partial_fit: rebuilding after %d new points", total_new)
            # We don't store the full historical dataset, so this requires
            # the user to manage data externally.  Instead we just note
            # that a threshold was hit and recommend ``fit()``.
            raise RuntimeError(
                "partial_fit rebuild threshold exceeded. "
                "Please concatenate all data and call fit() for a full refit."
            )

        # Compute embeddings for new points
        embedded_input = x_new.to(dtype=self.embedding_dtype)
        with torch.no_grad():
            new_embeddings = self.embedding_model(embedded_input)
        if self.embedding_dtype != self.dtype:
            new_embeddings = new_embeddings.to(dtype=self.dtype)

        # Concatenate embeddings
        old_n = self.embeddings.shape[0]
        self.embeddings = torch.cat([self.embeddings, new_embeddings], dim=0)

        # Build enlarged kernel operator (same lambda)
        self.kernel_operator = self.build_kernel_operator(self.embeddings)

        # For the enlarged RHS, warm-start from old alpha (scaled by forgetting factor)
        old_alpha = self.alpha * forgetting_factor
        y_extended = torch.cat(
            [
                torch.zeros(old_n, device=self.device, dtype=self.dtype),
                y_new,
            ]
        )

        # We need to solve (K_new + lambda I) alpha_new = y_extended.
        # Warm-start with [old_alpha * forgetting_factor, 0]
        x0 = torch.cat(
            [
                old_alpha,
                torch.zeros(m, device=self.device, dtype=self.dtype),
            ]
        )

        # Preconditioner: rebuild for enlarged system (cheap relative to solve)
        self.preconditioner = self.build_preconditioner(
            self.kernel_operator.matvec,
            self.embeddings.shape[0],
        )

        self.alpha, self.pcg_iterations_ = self.solve_pcg(
            self.kernel_operator,
            self.preconditioner,
            y_extended,
            x0=x0,
        )

        if self.verbose:
            logger.info(
                "partial_fit: added %d points, total=%d, PCG iters=%d",
                m,
                self.embeddings.shape[0],
                self.pcg_iterations_,
            )
        return self

    def bo_eval(
        self,
        train_embeddings: torch.Tensor,
        x_val: torch.Tensor,
        y: torch.Tensor,
        train_idx: torch.Tensor,
        val_idx: torch.Tensor,
        lambda_reg_value: float,
        gamma_value: float,
        num_probes_value: int,
    ) -> float:
        """Single BO evaluation: fit on train, predict on val, return RMSE."""
        try:
            kernel_op = self.build_kernel_operator(train_embeddings, lambda_reg=lambda_reg_value)
            precond = self.build_preconditioner(
                kernel_op.matvec,
                train_embeddings.shape[0],
                gamma=gamma_value,
                num_probes=num_probes_value,
            )
            alpha, _ = self.solve_pcg(kernel_op, precond, y[train_idx])

            x_val_embed = x_val.to(dtype=self.embedding_dtype)
            with torch.no_grad():
                val_embeddings = self.embedding_model(x_val_embed)
            if self.embedding_dtype != self.dtype:
                val_embeddings = val_embeddings.to(dtype=self.dtype)
            k_val = kernel_op.kernel_eval(val_embeddings, train_embeddings)
            y_val_pred = k_val @ alpha
            return torch.sqrt(torch.mean((y_val_pred - y[val_idx]) ** 2)).item()
        except (RuntimeError, ValueError):
            return float("inf")

    def fit_path(
        self,
        x: Union[torch.Tensor, "numpy.ndarray"],
        y: Union[torch.Tensor, "numpy.ndarray"],
        lambda_reg_grid: list[float],
        reuse_precond: bool = True,
    ) -> dict:
        """Fit a regularization path over a sequence of ``lambda_reg`` values.

        Embeddings and the preconditioner are computed once; only the
        kernel operator and PCG solve are repeated for each ``lambda``.
        Solves are warm-started from the previous ``alpha`` (largest
        ``lambda`` first, since it is best-conditioned).

        Args:
            x: Measurement locations of shape ``(n, dx)``.
            y: Noisy observations of shape ``(n,)``.
            lambda_reg_grid: List of ``lambda_reg`` values.  Should be sorted
                descending (largest first) for best warm-start behaviour.
            reuse_precond: If ``True``, build the preconditioner once for the
                first ``lambda`` and reuse it for the rest.

        Returns:
            Dictionary with keys ``"alphas"``, ``"lambda_reg"``, ``"pcg_iters"``,
            ``"final_rel_res"``.
        """
        x = to_tensor(x, device=self.device, dtype=self.dtype)
        y = to_tensor(y, device=self.device, dtype=self.dtype).squeeze()

        if x.dim() != 2:
            raise ValueError(f"x must be 2-D, got shape {x.shape}")
        if y.dim() != 1:
            raise ValueError(f"y must be 1-D, got shape {y.shape}")
        if not lambda_reg_grid:
            raise ValueError("lambda_reg_grid must not be empty")

        embeddings = self.compute_embeddings(x)
        n = embeddings.shape[0]

        # Sort descending for warm-start stability
        sorted_lambdas = sorted(lambda_reg_grid, reverse=True)

        alphas = []
        pcg_iters = []
        rel_reses = []
        x0 = None
        precond = None

        for lambda_reg_value in sorted_lambdas:
            kernel_op = self.build_kernel_operator(embeddings, lambda_reg=lambda_reg_value)
            if precond is None or not reuse_precond:
                precond = self.build_preconditioner(
                    kernel_op.matvec, n, gamma=self.gamma, num_probes=self.num_probes
                )
            alpha, iters = self.solve_pcg(kernel_op, precond, y, x0=x0)
            alphas.append(alpha)
            pcg_iters.append(iters)
            final_res = (
                torch.linalg.norm(kernel_op.matvec(alpha) - y).item() / torch.linalg.norm(y).item()
            )
            rel_reses.append(final_res)
            x0 = alpha.clone()

        return {
            "lambda_reg": sorted_lambdas,
            "alphas": alphas,
            "pcg_iters": pcg_iters,
            "final_rel_res": rel_reses,
        }

    def fit_learned_embeddings(
        self,
        x: Union[torch.Tensor, "numpy.ndarray"],
        y: Union[torch.Tensor, "numpy.ndarray"],
        lr: float = 1e-3,
        epochs: int = 50,
        rebuild_freq: int = 10,
        patience: int = 5,
    ) -> "LAKERRegressor":
        """Optimise the embedding MLP weights end-to-end on the regression objective.

        This makes the fixed ``PositionEmbedding`` (or a custom module) fully
        differentiable and trains it via Adam on the kernel ridge regression
        residual.  The preconditioner is rebuilt every ``rebuild_freq`` epochs.

        Args:
            x: Measurement locations of shape ``(n, dx)``.
            y: Noisy observations of shape ``(n,)``.
            lr: Learning rate for Adam.
            epochs: Number of optimisation epochs.
            rebuild_freq: Rebuild preconditioner and recompute ``alpha`` every
                ``rebuild_freq`` epochs.
            patience: Early-stopping patience on training loss.

        Returns:
            ``self`` for method chaining.

        Raises:
            RuntimeError: If no embedding model is available or it has no
                trainable parameters.
        """
        x = to_tensor(x, device=self.device, dtype=self.dtype)
        y = to_tensor(y, device=self.device, dtype=self.dtype).squeeze()

        if x.dim() != 2:
            raise ValueError(f"x must be 2-D, got shape {x.shape}")
        if y.dim() != 1:
            raise ValueError(f"y must be 1-D, got shape {y.shape}")

        if self.embedding_module is None and not hasattr(self, "embedding_model"):
            raise RuntimeError(
                "fit_learned_embeddings requires an embedding model. "
                "Call fit() first or pass embedding_module to __init__."
            )

        model = self.embedding_model
        if model is None:
            raise RuntimeError("No embedding model available.")

        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            for p in model.parameters():
                p.requires_grad = True
            trainable = list(model.parameters())
            if not trainable:
                raise RuntimeError("Embedding model has no trainable parameters.")

        optimizer = torch.optim.Adam(trainable, lr=lr)
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward pass: compute embeddings with grad
            embedded_input = x.to(dtype=self.embedding_dtype)
            embeddings = model(embedded_input)
            if self.embedding_dtype != self.dtype:
                embeddings = embeddings.to(dtype=self.dtype)

            # Build DIFFERENTIABLE kernel operator for loss
            kernel_op = self.build_kernel_operator(embeddings)

            # Recompute alpha and preconditioner periodically using DETACHED embeddings
            if epoch % rebuild_freq == 0 or getattr(self, "alpha", None) is None:
                with torch.no_grad():
                    kernel_op_detached = self.build_kernel_operator(embeddings.detach())
                    precond = self.build_preconditioner(
                        kernel_op_detached.matvec, embeddings.shape[0]
                    )
                    alpha = self.solve_pcg(kernel_op_detached, precond, y)[0]
                self.preconditioner = precond
            else:
                alpha = self.alpha.detach()

            # Compute residual loss with fixed alpha but differentiable kernel
            residual = kernel_op.matvec(alpha) - y
            loss = 0.5 * torch.dot(residual, residual)

            loss.backward()
            optimizer.step()

            loss_item = loss.item()
            if self.verbose and (epoch + 1) % 10 == 0:
                logger.info(
                    "Learned embeddings epoch %d/%d, loss=%.4e", epoch + 1, epochs, loss_item
                )

            if loss_item < best_loss:
                best_loss = loss_item
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if self.verbose:
                        logger.info("Early stopping at epoch %d (loss=%.4e)", epoch + 1, loss_item)
                    break

        # Final fit with optimised embeddings
        self.embeddings = embeddings.detach()
        self.kernel_operator = kernel_op
        self.alpha, self.pcg_iterations_ = self.solve_pcg(
            self.kernel_operator, self.preconditioner, y
        )
        return self

    def predict(
        self,
        x: Union[torch.Tensor, "numpy.ndarray"],
    ) -> torch.Tensor:
        """Reconstruct the radio field at query locations.

        Evaluates ``\\hat{r}(x) = \\sum_i G(x, x_i) \\alpha_i``.

        Args:
            x: Query locations of shape ``(m, dx)``.

        Returns:
            Predictions of shape ``(m,)``.

        Raises:
            RuntimeError: If the model has not been fitted or no embedding model
                is available.
            ValueError: If ``x`` is not 2-D.
        """
        if self.alpha is None or self.embeddings is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        x = to_tensor(x, device=self.device, dtype=self.dtype)
        if x.dim() != 2:
            raise ValueError(f"x must be 2-D, got shape {x.shape}")

        with torch.no_grad():
            if self.embedding_model is None:
                raise RuntimeError(
                    "No embedding model available. Ensure the model was fitted with "
                    "an embedding_module or the default PositionEmbedding."
                )
            embedded_input = x.to(dtype=self.embedding_dtype)
            query_embeddings = self.embedding_model(embedded_input)
            if self.embedding_dtype != self.dtype:
                query_embeddings = query_embeddings.to(dtype=self.dtype)
            m = query_embeddings.shape[0]
            n = self.embeddings.shape[0]

            # Chunked evaluation for large query sets to avoid O(m*n) memory blow-up.
            # We 2-D tile over both query and training dimensions when either is large.
            chunk_size = self.chunk_size
            if chunk_size is None and max(m, n) > 5000:
                chunk_size = max(1024, min(max(m, n) // 10, 8192))

            element_size = 4 if self.dtype == torch.float32 else 8
            mem_per_chunk = (
                (chunk_size or m) * n * element_size if chunk_size else m * n * element_size
            )
            if chunk_size is None or mem_per_chunk <= 64 * 1024 * 1024:
                k_query = self.kernel_operator.kernel_eval(
                    query_embeddings, self.embeddings, chunk_size=chunk_size
                )
                return k_query @ self.alpha

            # 2-D tiled prediction for very large m or n (exact kernel only).
            # Low-rank approximations are already memory-bounded, so we always
            # use the kernel_operator path above for them.
            if self.kernel_approx is not None:
                k_query = self.kernel_operator.kernel_eval(
                    query_embeddings, self.embeddings, chunk_size=chunk_size
                )
                return k_query @ self.alpha

            out = torch.empty(m, device=self.device, dtype=self.dtype)
            chunk_size_local = chunk_size
            for i_start in range(0, m, chunk_size_local):
                i_end = min(i_start + chunk_size_local, m)
                accum = torch.zeros(i_end - i_start, device=self.device, dtype=self.dtype)
                e_i = query_embeddings[i_start:i_end]
                for j_start in range(0, n, chunk_size_local):
                    j_end = min(j_start + chunk_size_local, n)
                    gram_block = e_i @ self.embeddings[j_start:j_end].T
                    exp_safe(gram_block, out=gram_block)
                    accum.addmv_(gram_block, self.alpha[j_start:j_end])
                out[i_start:i_end] = accum
            return out

    def predict_variance(
        self,
        x: Union[torch.Tensor, "numpy.ndarray"],
    ) -> torch.Tensor:
        """Predictive variance (uncertainty) at query locations.

        For exact and Nyström kernels this solves one linear system per query
        point using the learned preconditioner.  For RFF it uses the closed-form
        Woodbury identity, which is orders of magnitude faster.

        Args:
            x: Query locations of shape ``(m, dx)``.

        Returns:
            Variance of shape ``(m,)`` (non-negative).

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If ``x`` is not 2-D.
        """
        if self.alpha is None or self.embeddings is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        x = to_tensor(x, device=self.device, dtype=self.dtype)
        if x.dim() != 2:
            raise ValueError(f"x must be 2-D, got shape {x.shape}")

        with torch.no_grad():
            if self.embedding_model is None:
                raise RuntimeError(
                    "No embedding model available. Ensure the model was fitted with "
                    "an embedding_module or the default PositionEmbedding."
                )
            embedded_input = x.to(dtype=self.embedding_dtype)
            query_embeddings = self.embedding_model(embedded_input)
            if self.embedding_dtype != self.dtype:
                query_embeddings = query_embeddings.to(dtype=self.dtype)
            m = query_embeddings.shape[0]
            n = self.embeddings.shape[0]

            # RFF: closed-form via Woodbury (fast path)
            if self.kernel_approx == "rff" and hasattr(self.kernel_operator, "phi"):
                ko = cast(RandomFeatureAttentionKernelOperator, self.kernel_operator)
                proj = query_embeddings @ ko.freq
                phi_q = torch.cat(
                    [torch.cos(proj + ko.phase), torch.sin(proj + ko.phase)], dim=1
                ) / (ko.num_features**0.5)
                a = ko.phi.T @ ko.phi  # (2r, 2r)
                a_reg = a + self.lambda_reg * torch.eye(
                    a.shape[0], device=self.device, dtype=self.dtype
                )
                # Cholesky solve for stability
                chol = torch.linalg.cholesky(a_reg)
                m_solve = torch.cholesky_solve(
                    torch.eye(a.shape[0], device=self.device, dtype=self.dtype), chol
                )
                var = self.lambda_reg * torch.sum(phi_q @ m_solve * phi_q, dim=1)
                return var.clamp(min=0.0)

            # Exact / Nyström: solve (K + lambda I) V = k(X, x*) with PCG
            element_size = 4 if self.dtype == torch.float32 else 8
            chunk_size = self.chunk_size
            if chunk_size is None:
                mem_needed = m * n * element_size
                if mem_needed > 64 * 1024 * 1024:
                    chunk_size = max(1024, min(n // 10, 8192))

            var = torch.empty(m, device=self.device, dtype=self.dtype)
            pcg = PreconditionedConjugateGradient(
                tol=self.pcg_tol,
                max_iter=self.pcg_max_iter,
                verbose=False,
            )

            if chunk_size is None or m <= chunk_size:
                k_train_query = self.kernel_operator.kernel_eval(
                    self.embeddings, query_embeddings
                )  # (n, m)
                if k_train_query.is_sparse:
                    k_train_query = k_train_query.to_dense()
                v = pcg.solve(
                    operator=self.kernel_operator.matvec,
                    preconditioner=self.preconditioner.apply,
                    rhs=k_train_query,
                )  # (n, m)
                k_diag_mat = self.kernel_operator.kernel_eval(query_embeddings, query_embeddings)
                if k_diag_mat.is_sparse:
                    k_diag_mat = k_diag_mat.to_dense()
                k_diag = k_diag_mat.diagonal()
                var[:] = k_diag - torch.sum(k_train_query * v, dim=0)
            else:
                for start in range(0, m, chunk_size):
                    end = min(start + chunk_size, m)
                    q_chunk = query_embeddings[start:end]
                    k_train_chunk = self.kernel_operator.kernel_eval(
                        self.embeddings, q_chunk
                    )  # (n, end-start)
                    if k_train_chunk.is_sparse:
                        k_train_chunk = k_train_chunk.to_dense()
                    v_chunk = pcg.solve(
                        operator=self.kernel_operator.matvec,
                        preconditioner=self.preconditioner.apply,
                        rhs=k_train_chunk,
                    )  # (n, end-start)
                    k_diag_mat = self.kernel_operator.kernel_eval(q_chunk, q_chunk)
                    if k_diag_mat.is_sparse:
                        k_diag_mat = k_diag_mat.to_dense()
                    k_diag_chunk = k_diag_mat.diagonal()
                    var[start:end] = k_diag_chunk - torch.sum(k_train_chunk * v_chunk, dim=0)

            return var.clamp(min=0.0)

    def score(
        self,
        x: Union[torch.Tensor, "numpy.ndarray"],
        y: Union[torch.Tensor, "numpy.ndarray"],
    ) -> float:
        """Compute negative RMSE as a sklearn-style score.

        Args:
            x: Test locations of shape ``(m, dx)``.
            y: Ground-truth values of shape ``(m,)``.

        Returns:
            Negative RMSE (higher is better).
        """
        y_pred = self.predict(x)
        y_true = to_tensor(y, device=self.device, dtype=self.dtype).squeeze()
        rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()
        return -rmse

    def condition_number(self) -> float:
        """Return the condition number of the preconditioned system.

        Estimates the condition number ``kappa(P (lambda I + G))`` via
        power iteration for the largest eigenvalue and inverse power
        iteration for the smallest.

        Requires the preconditioner to have been built.

        Returns:
            Estimated condition number.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if self.preconditioner is None or self.kernel_operator is None:
            raise RuntimeError("Model has not been fitted.")
        # Power iteration on P A and (P A)^{-1} to estimate condition number
        n = self.kernel_operator.n
        apply_operator = self.kernel_operator.matvec
        apply_precond = self.preconditioner.apply

        def preconditioned_operator(v: torch.Tensor) -> torch.Tensor:
            """Apply the preconditioned operator P A to a vector."""
            return apply_precond(apply_operator(v))

        # Largest eigenvalue
        v = torch.randn(n, device=self.device, dtype=self.dtype)
        v = v / torch.linalg.norm(v)
        for _ in range(10):
            v = preconditioned_operator(v)
            v = v / torch.linalg.norm(v)
        lam_max = torch.dot(v, preconditioned_operator(v)).item()

        # Smallest eigenvalue via inverse iteration (fewer steps since PA is
        # well-conditioned by design)
        v = torch.randn(n, device=self.device, dtype=self.dtype)
        v = v / torch.linalg.norm(v)

        pcg = PreconditionedConjugateGradient(tol=1e-6, max_iter=50, verbose=False)
        for _ in range(5):
            v = pcg.solve(
                operator=preconditioned_operator,
                preconditioner=lambda x: x,
                rhs=v,
            )
            v = v / torch.linalg.norm(v)
        lam_min = max(torch.dot(v, preconditioned_operator(v)).item(), torch.finfo(self.dtype).eps)

        return lam_max / lam_min

    def get_params(self, deep: bool = True) -> dict:
        """Return estimator parameters for sklearn compatibility.

        Args:
            deep: Whether to return parameters of nested objects.

        Returns:
            Dictionary mapping parameter names to values.
        """
        return {
            "embedding_dim": self.embedding_dim,
            "lambda_reg": self.lambda_reg,
            "gamma": self.gamma,
            "num_probes": self.num_probes,
            "epsilon": self.epsilon,
            "base_rho": self.base_rho,
            "cccp_max_iter": self.cccp_max_iter,
            "cccp_tol": self.cccp_tol,
            "pcg_tol": self.pcg_tol,
            "pcg_max_iter": self.pcg_max_iter,
            "chunk_size": self.chunk_size,
            "embedding_module": self.embedding_module,
            "kernel_approx": self.kernel_approx,
            "num_landmarks": self.num_landmarks,
            "num_features": self.num_features,
            "embedding_dtype": (
                str(self.embedding_dtype).replace("torch.", "") if self.embedding_dtype else None
            ),
            "device": str(self.device),
            "dtype": str(self.dtype).replace("torch.", ""),
            "verbose": self.verbose,
        }

    def set_params(self, **params) -> "LAKERRegressor":
        """Set estimator parameters for sklearn compatibility.

        Args:
            **params: Parameter names and values.

        Returns:
            ``self`` for method chaining.

        Raises:
            ValueError: If an unknown parameter is provided.
        """
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid parameter {key!r} for LAKERRegressor")
            setattr(self, key, value)
        return self

    def save(self, path: str) -> None:
        """Serialize the fitted model to disk.

        Saves all hyperparameters and fitted tensors so that ``load()``
        can reconstruct an identical model.

        Args:
            path: File path ending in ``.pt`` or ``.pth``.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if self.alpha is None:
            raise RuntimeError("Model has not been fitted. Call fit() before saving.")
        state = {
            "embedding_dim": self.embedding_dim,
            "lambda_reg": self.lambda_reg,
            "gamma": self.gamma,
            "num_probes": self.num_probes,
            "epsilon": self.epsilon,
            "base_rho": self.base_rho,
            "cccp_max_iter": self.cccp_max_iter,
            "cccp_tol": self.cccp_tol,
            "pcg_tol": self.pcg_tol,
            "pcg_max_iter": self.pcg_max_iter,
            "chunk_size": self.chunk_size,
            "kernel_approx": self.kernel_approx,
            "num_landmarks": self.num_landmarks,
            "num_features": self.num_features,
            "k_neighbors": self.k_neighbors,
            "grid_size": self.grid_size,
            "distributed": self.distributed,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "embedding_dtype": str(self.embedding_dtype) if self.embedding_dtype else None,
            "verbose": self.verbose,
            "embeddings": self.embeddings,
            "alpha": self.alpha,
        }
        if self.embedding_model is not None:
            state["embedding_model_state"] = self.embedding_model.state_dict()
            state["embedding_model_class"] = self.embedding_model.__class__.__name__
            state["embedding_model_module"] = self.embedding_model.__class__.__module__
            if hasattr(self.embedding_model, "input_dim"):
                state["input_dim"] = self.embedding_model.input_dim
        torch.save(state, path)

    @classmethod
    def load(cls, path: str) -> "LAKERRegressor":
        """Deserialize a model from disk.

        Args:
            path: File path previously passed to ``save()``.

        Returns:
            Reconstructed ``LAKERRegressor`` instance ready for ``predict()``.
        """
        state = torch.load(path, weights_only=False)
        dtype = torch.float32 if "float32" in state["dtype"] else torch.float64
        embedding_dtype = (
            torch.float32
            if state.get("embedding_dtype") and "float32" in state["embedding_dtype"]
            else (
                torch.float64
                if state.get("embedding_dtype") and "float64" in state["embedding_dtype"]
                else None
            )
        )
        model = cls(
            embedding_dim=state["embedding_dim"],
            lambda_reg=state["lambda_reg"],
            gamma=state["gamma"],
            num_probes=state["num_probes"],
            epsilon=state["epsilon"],
            base_rho=state["base_rho"],
            cccp_max_iter=state["cccp_max_iter"],
            cccp_tol=state["cccp_tol"],
            pcg_tol=state["pcg_tol"],
            pcg_max_iter=state["pcg_max_iter"],
            chunk_size=state.get("chunk_size"),
            kernel_approx=state.get("kernel_approx"),
            num_landmarks=state.get("num_landmarks"),
            num_features=state.get("num_features"),
            k_neighbors=state.get("k_neighbors"),
            grid_size=state.get("grid_size"),
            distributed=state.get("distributed", False),
            embedding_dtype=embedding_dtype,
            device=state["device"],
            dtype=dtype,
            verbose=state["verbose"],
        )
        model.embeddings = state["embeddings"].to(model.device)
        model.alpha = state["alpha"].to(model.device)
        if "embedding_model_state" in state:
            class_name = state["embedding_model_class"]
            module_name = state.get("embedding_model_module", "laker.embeddings")
            try:
                import importlib

                module = importlib.import_module(module_name)
                cls = getattr(module, class_name)
            except (ImportError, AttributeError):
                logger.warning(
                    "Could not import %s.%s for loading; falling back to PositionEmbedding. "
                    "Save/load of custom embedding modules requires the module to be importable.",
                    module_name,
                    class_name,
                )
                from laker.embeddings import PositionEmbedding as cls

                class_name = "PositionEmbedding"

            input_dim = state.get("input_dim", 2)
            embed_dtype = embedding_dtype if embedding_dtype else dtype
            embed_cls: Callable[..., Any] = cls
            if class_name == "PositionEmbedding":
                model.embedding_model = embed_cls(
                    input_dim=input_dim,
                    embedding_dim=model.embedding_dim,
                    device=model.device,
                    dtype=embed_dtype,
                )
            else:
                try:
                    model.embedding_model = embed_cls(
                        input_dim=input_dim,
                        embedding_dim=model.embedding_dim,
                        device=model.device,
                        dtype=embed_dtype,
                    )
                except TypeError:
                    model.embedding_model = cls()
                    model.embedding_model.to(device=model.device, dtype=embed_dtype)
            model.embedding_model.load_state_dict(state["embedding_model_state"])

        # Reconstruct kernel operator with the correct approximation
        if model.kernel_approx is None:
            model.kernel_operator = AttentionKernelOperator(
                embeddings=model.embeddings,
                lambda_reg=model.lambda_reg,
                chunk_size=model.chunk_size,
                device=model.device,
                dtype=dtype,
            )
        elif model.kernel_approx == "nystrom":
            model.kernel_operator = NystromAttentionKernelOperator(
                embeddings=model.embeddings,
                lambda_reg=model.lambda_reg,
                num_landmarks=model.num_landmarks,
                chunk_size=model.chunk_size,
                device=model.device,
                dtype=dtype,
            )
        elif model.kernel_approx == "rff":
            model.kernel_operator = RandomFeatureAttentionKernelOperator(
                embeddings=model.embeddings,
                lambda_reg=model.lambda_reg,
                num_features=model.num_features,
                device=model.device,
                dtype=dtype,
            )
        return model
