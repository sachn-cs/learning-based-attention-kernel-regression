"""Implicit differentiation through PCG fixed-point for hypergradients."""

from typing import Callable, List

import torch

from laker.solvers import PreconditionedConjugateGradient


def hypergradient(
    operator_fn: Callable[[torch.Tensor], torch.Tensor],
    preconditioner_fn: Callable[[torch.Tensor], torch.Tensor],
    alpha: torch.Tensor,
    dL_dalpha: torch.Tensor,
    param_list: List[torch.Tensor],
    pcg_tol: float = 1e-6,
    pcg_max_iter: int = 500,
    verbose: bool = False,
) -> List[torch.Tensor]:
    """Compute hypergradients via the adjoint method (implicit differentiation).

    Given the fixed-point ``alpha`` satisfying ``A(alpha, theta) alpha = y``,
    and a scalar loss ``L(alpha)``, we compute ``dL/d theta`` for each
    parameter ``theta`` in ``param_list``.

    The adjoint method:
        1. Solve ``A v = dL_dalpha`` for the adjoint vector ``v``.
        2. For each ``theta``, compute ``-v^T (dA/dtheta) alpha``.

    Args:
        operator_fn: Callable ``A(v)`` that applies the operator to a vector.
        preconditioner_fn: Callable ``P(v)`` for the preconditioner.
        alpha: Fixed-point solution ``alpha*`` (detached).
        dL_dalpha: Gradient of the outer loss w.r.t. ``alpha``.
        param_list: List of parameter tensors to differentiate w.r.t.
        pcg_tol: Tolerance for the adjoint linear solve.
        pcg_max_iter: Maximum iterations for the adjoint solve.
        verbose: Whether to log adjoint solve progress.

    Returns:
        List of hypergradients, one per parameter in ``param_list``.

    """
    # Adjoint solve: A v = dL_dalpha
    pcg = PreconditionedConjugateGradient(
        tol=pcg_tol,
        max_iter=pcg_max_iter,
        verbose=verbose,
    )
    v = pcg.solve(
        operator=operator_fn,
        preconditioner=preconditioner_fn,
        rhs=dL_dalpha,
    )

    # For each parameter theta, compute -v^T (dA/dtheta) alpha
    # via torch.autograd.grad on the scalar v^T A(alpha) alpha.
    # We recompute A(alpha) in a differentiable context.
    alpha_const = alpha.detach()
    v_const = v.detach()

    with torch.enable_grad():
        # Re-enable grad for the operator evaluation
        alpha_for_grad = alpha_const.clone().requires_grad_(True)
        A_alpha = operator_fn(alpha_for_grad)
        scalar = torch.dot(v_const, A_alpha)

    hypergrads = []
    for param in param_list:
        if param.requires_grad:
            # d(scalar)/d(param) = v^T (dA/dparam) alpha
            # We want -v^T (dA/dparam) alpha, so negate.
            grad = torch.autograd.grad(
                scalar,
                param,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if grad is None:
                grad = torch.zeros_like(param)
            hypergrads.append(-grad)
        else:
            hypergrads.append(torch.zeros_like(param))

    return hypergrads
