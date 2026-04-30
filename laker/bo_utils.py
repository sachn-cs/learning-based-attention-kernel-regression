"""Lightweight Bayesian Optimisation utilities for LAKER hyperparameter search.

Implements a Gaussian Process surrogate with an RBF kernel on a log-scale
parameter space, Expected Improvement acquisition, and simple L-BFGS-B
optimisation of GP hyperparameters.
"""

import logging
import math
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class _GPSurrogate:
    """Gaussian Process surrogate for Bayesian Optimisation.

    Operates on a **normalised** parameter space where each dimension is
    mapped to ``[0, 1]`` via log-transformation where appropriate.

    Kernel: ``k(x, x') = sigma_f^2 * exp(-0.5 * r^2 / l^2)`` with fixed
    small observation noise ``sigma_n^2``.
    """

    def __init__(
        self,
        bounds: np.ndarray,
        log_indices: Optional[list[int]] = None,
        sigma_f: float = 1.0,
        l: float = 0.2,
        sigma_n: float = 1e-4,
    ):
        """Args:
            bounds: ``(d, 2)`` array with ``[min, max]`` per dimension.
            log_indices: Dimensions that should be searched on log-scale.
            sigma_f: Signal variance.
            l: Initial length scale (same for all dims).
            sigma_n: Observation noise (fixed, small).
        """
        self.bounds = bounds.astype(np.float64)
        self.d = bounds.shape[0]
        self.log_indices = log_indices or []
        self.sigma_f = float(sigma_f)
        self.l = float(l)
        self.sigma_n = float(sigma_n)

        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.K_inv: Optional[np.ndarray] = None
        self.alpha_vec: Optional[np.ndarray] = None

    def _transform(self, x: np.ndarray) -> np.ndarray:
        """Map raw parameters to normalised [0, 1] space."""
        x = np.atleast_2d(x).astype(np.float64)
        z = x.copy()
        for i in self.log_indices:
            z[:, i] = np.log10(np.clip(z[:, i], self.bounds[i, 0] * 0.1, self.bounds[i, 1] * 10))
        # Normalise each dimension
        z = (z - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
        return np.clip(z, 0.0, 1.0)

    def _kernel(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """RBF kernel in normalised space."""
        sqdist = np.sum(x1 ** 2, axis=1).reshape(-1, 1) + np.sum(x2 ** 2, axis=1) - 2 * np.dot(x1, x2.T)
        return self.sigma_f ** 2 * np.exp(-0.5 * sqdist / (self.l ** 2 + 1e-12))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit GP to observations."""
        self.X = self._transform(X)
        # Normalise y to zero mean, unit variance for numerical stability
        self.y_mean = y.mean()
        self.y_std = y.std() + 1e-8
        self.y = (y - self.y_mean) / self.y_std

        # Optimise length scale via simple random search on marginal likelihood
        best_l = self.l
        best_ml = float("-inf")
        for cand_l in np.logspace(-2, 0, 20):
            ml = self._marginal_likelihood(cand_l)
            if ml > best_ml:
                best_ml = ml
                best_l = cand_l
        self.l = best_l

        K = self._kernel(self.X, self.X)
        K[np.diag_indices_from(K)] += self.sigma_n ** 2
        self.L = np.linalg.cholesky(K + 1e-8 * np.eye(K.shape[0]))
        self.alpha_vec = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))

    def _marginal_likelihood(self, l: float) -> float:
        """Compute log marginal likelihood for a candidate length scale."""
        old_l = self.l
        self.l = l
        K = self._kernel(self.X, self.X)
        K[np.diag_indices_from(K)] += self.sigma_n ** 2
        try:
            L = np.linalg.cholesky(K + 1e-8 * np.eye(K.shape[0]))
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y))
            ml = -0.5 * np.dot(self.y, alpha) - np.sum(np.log(np.diag(L))) - 0.5 * K.shape[0] * np.log(2 * math.pi)
        except np.linalg.LinAlgError:
            ml = float("-inf")
        self.l = old_l
        return ml

    def predict(self, X_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return posterior mean and variance."""
        X_new_t = self._transform(X_new)
        K_s = self._kernel(self.X, X_new_t)
        K_ss = self._kernel(X_new_t, X_new_t)
        K_ss[np.diag_indices_from(K_ss)] += self.sigma_n ** 2

        v = np.linalg.solve(self.L, K_s)
        mu = np.dot(K_s.T, self.alpha_vec)
        var = np.diag(K_ss) - np.sum(v ** 2, axis=0)
        var = np.clip(var, 1e-12, None)

        # Undo normalisation
        mu = mu * self.y_std + self.y_mean
        var = var * (self.y_std ** 2)
        return mu, var

    def expected_improvement(self, X_new: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """Expected Improvement acquisition function."""
        mu, var = self.predict(X_new)
        sigma = np.sqrt(var)
        y_best = self.y.min() * self.y_std + self.y_mean  # minimise RMSE

        with np.errstate(divide="warn", invalid="warn"):
            z = (y_best - mu - xi) / (sigma + 1e-12)
        ei = (y_best - mu - xi) * scipy_norm_cdf(z) + sigma * scipy_norm_pdf(z)
        ei[sigma < 1e-12] = 0.0
        return ei


def scipy_norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x ** 2) / np.sqrt(2.0 * math.pi)


def scipy_norm_cdf(x: np.ndarray) -> np.ndarray:
    """Abramowitz and Stegun approximation of the normal CDF."""
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = np.sign(x)
    x = np.abs(x) / np.sqrt(2.0)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return 0.5 * (1.0 + sign * y)
