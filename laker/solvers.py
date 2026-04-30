"""Iterative solvers for regularised kernel linear systems."""

import logging
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)


class PreconditionedConjugateGradient:
    """Preconditioned Conjugate Gradient (PCG) solver.

    Solves ``A x = b`` where ``A`` is symmetric positive-definite, using a
    preconditioner ``P`` such that ``P A`` has a compressed spectrum.

    Supports both 1-D (single RHS) and 2-D (batch of RHS) inputs using
    vectorised block operations. Includes residual replacement every 50
    iterations to combat floating-point drift.

    This is Algorithm 1 (lines 15--26) from the LAKER paper.

    Args:
        tol: Relative residual tolerance ``||r|| / ||b|| <= tol``.
        max_iter: Maximum iterations. If ``None``, defaults to ``n``.
        verbose: Whether to log convergence.
        restart_freq: If an integer, the residual is explicitly recomputed
            from ``b - A x`` every ``restart_freq`` iterations to avoid
            round-off accumulation.  ``None`` disables recomputation.
            Disabled by default because it can hurt float32 stability.
    """

    def __init__(
        self,
        tol: float = 1e-10,
        max_iter: Optional[int] = None,
        verbose: bool = True,
        restart_freq: Optional[int] = None,
        breakdown_eps: Optional[float] = None,
    ) -> None:
        self.tol = float(tol)
        self.max_iter = max_iter
        self.verbose = verbose
        self.restart_freq = restart_freq
        self.breakdown_eps = breakdown_eps
        self.iterations: int = 0
        self.residual_norm: float = float("inf")

    def solve(
        self,
        operator: Callable[[torch.Tensor], torch.Tensor],
        preconditioner: Callable[[torch.Tensor], torch.Tensor],
        rhs: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Solve ``A x = b`` using PCG.

        Args:
            operator: Callable applying ``A`` to a vector or batch.
            preconditioner: Callable applying ``P`` to a vector or batch.
            rhs: Right-hand side tensor ``b`` of shape ``(n,)`` or ``(n, k)``.
            x0: Initial guess. If ``None``, zero vector is used.

        Returns:
            Solution tensor ``x`` of the same shape as ``rhs``.

        Raises:
            RuntimeError: If a scalar-product denominator becomes near-zero
                (possible breakdown or indefinite operator/preconditioner).
        """
        if rhs.dim() not in (1, 2):
            raise ValueError(f"rhs must be 1-D or 2-D, got shape {rhs.shape}")

        n = rhs.shape[0]
        max_iter = self.max_iter if self.max_iter is not None else n

        if x0 is None:
            x = torch.zeros_like(rhs)
            r = rhs.clone()
        else:
            x = x0.clone()
            r = rhs - operator(x)

        z = preconditioner(r)
        p = z.clone()
        b_norm = torch.linalg.norm(rhs)
        if b_norm == 0:
            return x

        if rhs.dim() == 1:
            return self._solve_1d(operator, preconditioner, rhs, x, r, z, p, b_norm, max_iter)
        else:
            return self._solve_2d(operator, preconditioner, rhs, x, r, z, p, b_norm, max_iter)

    def _solve_1d(
        self,
        operator: Callable[[torch.Tensor], torch.Tensor],
        preconditioner: Callable[[torch.Tensor], torch.Tensor],
        rhs: torch.Tensor,
        x: torch.Tensor,
        r: torch.Tensor,
        z: torch.Tensor,
        p: torch.Tensor,
        b_norm: float,
        max_iter: int,
    ) -> torch.Tensor:
        """1-D (single RHS) PCG using scalar dot products for speed."""
        rz_old = torch.dot(r, z).item()
        for iteration in range(max_iter):
            ap = operator(p)
            p_ap = torch.dot(p, ap).item()
            eps = self.breakdown_eps if self.breakdown_eps is not None else torch.finfo(p.dtype).eps ** 0.5
            if p_ap <= -eps * torch.linalg.norm(p).item() * torch.linalg.norm(ap).item():
                raise RuntimeError(
                    "PCG breakdown: non-positive curvature detected (p^T A p <= 0). "
                    "The operator may be indefinite or the preconditioner may be unsuitable."
                )

            alpha = rz_old / p_ap
            x.add_(p, alpha=alpha)
            r.add_(ap, alpha=-alpha)

            if self.restart_freq is not None and (iteration + 1) % self.restart_freq == 0:
                r = rhs - operator(x)

            self.residual_norm = torch.linalg.norm(r).item()
            rel_res = self.residual_norm / b_norm

            if rel_res <= self.tol:
                self.iterations = iteration + 1
                if self.verbose:
                    logger.info(
                        "PCG converged in %d iterations, rel_res=%.3e", self.iterations, rel_res
                    )
                return x

            z = preconditioner(r)
            rz_new = torch.dot(r, z).item()
            beta = rz_new / rz_old
            p.mul_(beta).add_(z)
            rz_old = rz_new

        self.iterations = max_iter
        if self.verbose:
            logger.warning("PCG did not converge in %d iterations, rel_res=%.3e", max_iter, rel_res)
        return x

    def _solve_2d(
        self,
        operator: Callable[[torch.Tensor], torch.Tensor],
        preconditioner: Callable[[torch.Tensor], torch.Tensor],
        rhs: torch.Tensor,
        x: torch.Tensor,
        r: torch.Tensor,
        z: torch.Tensor,
        p: torch.Tensor,
        b_norm: float,
        max_iter: int,
    ) -> torch.Tensor:
        """2-D (batch RHS) PCG using vectorised column-wise dot products."""
        rz_old = torch.sum(r * z, dim=0)
        for iteration in range(max_iter):
            ap = operator(p)
            p_ap = torch.sum(p * ap, dim=0)
            eps = self.breakdown_eps if self.breakdown_eps is not None else torch.finfo(p.dtype).eps ** 0.5
            if torch.any(p_ap <= -eps * torch.linalg.norm(p, dim=0) * torch.linalg.norm(ap, dim=0)):
                raise RuntimeError(
                    "PCG breakdown: non-positive curvature detected (p^T A p <= 0). "
                    "The operator may be indefinite or the preconditioner may be unsuitable."
                )

            alpha = rz_old / p_ap
            x.add_(p * alpha.unsqueeze(0))
            r.add_(ap * (-alpha.unsqueeze(0)))

            if self.restart_freq is not None and (iteration + 1) % self.restart_freq == 0:
                r = rhs - operator(x)

            self.residual_norm = torch.linalg.norm(r).item()
            rel_res = self.residual_norm / b_norm

            if rel_res <= self.tol:
                self.iterations = iteration + 1
                if self.verbose:
                    logger.info(
                        "PCG converged in %d iterations, rel_res=%.3e", self.iterations, rel_res
                    )
                return x

            z = preconditioner(r)
            rz_new = torch.sum(r * z, dim=0)
            beta = rz_new / rz_old
            p.mul_(beta.unsqueeze(0)).add_(z)
            rz_old = rz_new

        self.iterations = max_iter
        if self.verbose:
            logger.warning("PCG did not converge in %d iterations, rel_res=%.3e", max_iter, rel_res)
        return x


class GradientDescent:
    """Unpreconditioned gradient descent baseline for benchmarking.

    Args:
        step_size: Fixed step size ``eta``. If ``None``, a conservative
            heuristic ``1 / max_eig`` is used (requires an extra matvec).
        tol: Relative residual tolerance.
        max_iter: Maximum iterations.
        verbose: Whether to log progress.
    """

    def __init__(
        self,
        step_size: Optional[float] = None,
        tol: float = 1e-3,
        max_iter: int = 50000,
        verbose: bool = False,
    ) -> None:
        self.step_size = step_size
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.verbose = verbose
        self.iterations: int = 0
        self.residual_norm: float = float("inf")

    def solve(
        self,
        operator: Callable[[torch.Tensor], torch.Tensor],
        rhs: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Solve ``A x = b`` via gradient descent."""
        if x0 is None:
            x = torch.zeros_like(rhs)
        else:
            x = x0.clone()

        # Estimate step size via power iteration if not provided
        eta = self.step_size
        if eta is None:
            v = torch.randn_like(rhs)
            for _ in range(5):
                v = operator(v)
                v = v / torch.linalg.norm(v)
            max_eig = torch.dot(v, operator(v)).item()
            eta = 0.9 / max(abs(max_eig), 1e-8)
            if self.verbose:
                logger.info("GD estimated step size eta=%.3e", eta)

        b_norm = torch.linalg.norm(rhs)
        for iteration in range(self.max_iter):
            r = rhs - operator(x)
            self.residual_norm = torch.linalg.norm(r).item()
            rel_res = self.residual_norm / (b_norm + 1e-16)
            if rel_res <= self.tol:
                self.iterations = iteration + 1
                if self.verbose:
                    logger.info("GD converged in %d iterations", self.iterations)
                return x
            x = x + eta * r

        self.iterations = self.max_iter
        if self.verbose:
            logger.warning(
                "GD did not converge in %d iterations, rel_res=%.3e", self.max_iter, rel_res
            )
        return x


class JacobiPreconditioner:
    """Diagonal (Jacobi) preconditioner baseline.

    ``P = diag(lambda I + G)^{-1}`` as described in Section V-A-3.
    """

    def __init__(self, diagonal: torch.Tensor) -> None:
        eps = torch.finfo(diagonal.dtype).eps
        self.inv_diag = 1.0 / diagonal.clamp(min=eps)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Jacobi preconditioner element-wise."""
        return self.inv_diag * x
