r"""One-step TMLE via the universal least-favorable submodel.

The iterative TMLE fits one (or a few) fluctuation coefficients and stops.  That
works, but the submodel it uses is only *locally* least favorable -- least
favorable at :math:`\epsilon = 0`.  When the fluctuation has to travel a long way
(a large :math:`\hat\epsilon`, typically because the initial fit is poor or a
positivity violation makes the clever covariate extreme), the fitted submodel is
no longer aligned with the efficient influence function at the endpoint, and the
score need not be solved.

The *universal least-favorable submodel* (van der Laan & Gruber, 2016) fixes this
by rebuilding the direction at every point along the path:

.. math::

    \operatorname{logit} \bar Q_{t + dt}
      = \operatorname{logit} \bar Q_t + dt \cdot h^\top d_t,
    \qquad
    d_t = \frac{P_n D^*(\bar Q_t)}{\lVert P_n D^*(\bar Q_t) \rVert}

so the path's score equals the canonical gradient at *every* :math:`t`, not just at
the start.  Walking until :math:`P_n D^*(\bar Q_t) = 0` therefore solves the
efficient score equation exactly, in one pass, and the log-likelihood increases
monotonically along the way.

This matters most for multidimensional targets, where an iterative fluctuation can
oscillate between components instead of converging.
"""

from __future__ import annotations

import warnings

import numpy as np

from .._typing import BoolArray, FloatArray
from ..exceptions import ConvergenceWarning
from ._score import (
    STOP_CONTEXT_MISSING,
    StopContext,
    absolute_score_weights,
    quasi_loglik,
    relative_score,
    ruled_warning,
    score_columns,
    score_scale,
)
from .iterative import (
    _SEPARATION_EPSILON as SEPARATION_EPSILON,
)
from .iterative import (
    Fluctuation,
    InitialFit,
    TargetingFailure,
    _rule_state,
    apply_logistic,
)
from .submodel import Submodel, weighted_form

__all__ = ["solve_one_step"]


def solve_one_step(
    outcome: FloatArray,
    initial: InitialFit,
    submodel: Submodel,
    weights: FloatArray,
    observed: BoolArray | None = None,
    *,
    target_weights: bool = False,
    alpha: float = 0.9995,
    step_size: float = 1e-3,
    max_steps: int = 20_000,
    tol: float = 1e-10,
    warn: bool = True,
    stopping: str = "relative",
    stop: StopContext | None = None,
) -> Fluctuation:
    """Walk the universal least-favorable submodel until the score vanishes.

    Parameters
    ----------
    step_size:
        Length of each step in ``epsilon`` units.  Smaller is more faithful to the
        continuous path and slower; the step is halved automatically whenever a
        step would overshoot (the score norm increases).
    max_steps:
        Safety cap.  Exceeding it warns rather than raising, and the partially
        targeted fit is returned so the caller can inspect the trace.
    stopping : {"relative", "scaled_var", "sample_size"}
        The rule that ends the walk.  ``"relative"`` runs the shipped walk below, untouched.
        A declared rule runs :func:`_solve_one_step_ruled`, which takes the same steps and
        ends at the first step the rule is met.
    stop : StopContext, optional
        The units and outcome range a declared rule computes its threshold from.  A
        declared rule without one raises ``ValueError``.
    """
    if stopping != "relative":
        return _solve_one_step_ruled(
            outcome,
            initial,
            submodel,
            weights,
            observed,
            target_weights=target_weights,
            alpha=alpha,
            step_size=step_size,
            max_steps=max_steps,
            tol=tol,
            warn=warn,
            stopping=stopping,
            stop=stop,
        )
    y = np.asarray(outcome, dtype=float).reshape(-1)
    n = y.shape[0]
    mask = np.ones(n, dtype=bool) if observed is None else np.asarray(observed, dtype=bool)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if step_size <= 0:
        raise ValueError(f"step_size must be positive; got {step_size}")

    scoring_h = submodel.observed
    fit_submodel, fit_weights = weighted_form(submodel, w) if target_weights else (submodel, w)

    current = initial.shrunk(alpha)
    epsilon = np.zeros(fit_submodel.dim)
    dx = float(step_size)
    scale = score_scale(scoring_h, w, mask)
    trace: list[float] = []
    loglik = quasi_loglik(y[mask], current.observed[mask], fit_weights[mask])
    steps = 0

    score = score_columns(y, current.observed, scoring_h, w, mask)
    score_before = score
    norm = float(np.linalg.norm(score))
    trace.append(relative_score(score, scale))

    while steps < max_steps and relative_score(score, scale) > tol:
        if norm == 0.0:
            break
        direction = score / norm
        candidate_epsilon = epsilon + dx * direction
        candidate = apply_logistic(current, fit_submodel, dx * direction, alpha)
        candidate_score = score_columns(y, candidate.observed, scoring_h, w, mask)
        candidate_norm = float(np.linalg.norm(candidate_score))

        if candidate_norm > norm:
            # Overshot the root: shorten the step rather than oscillating around it.
            dx *= 0.5
            if dx < 1e-14:
                break
            continue

        epsilon = candidate_epsilon
        current = candidate
        score = candidate_score
        norm = candidate_norm
        loglik = quasi_loglik(y[mask], current.observed[mask], fit_weights[mask])
        trace.append(relative_score(score, scale))
        steps += 1

    relative = relative_score(score, scale)
    converged = bool(relative <= tol)
    if not converged and warn:
        warnings.warn(
            f"one-step targeting stopped after {steps} step(s) with a relative score of "
            f"{relative:.3g} > {tol:g}. Try a smaller step_size, or check "
            "res.diagnostics.support() for a positivity violation.",
            ConvergenceWarning,
            stacklevel=2,
        )
    return Fluctuation(
        epsilon=epsilon,
        targeted=current,
        score=score,
        converged=converged,
        n_iter=steps,
        trace=tuple(trace),
        method="one_step",
        names=submodel.names,
        score_scale=scale,
        score_initial=score_before,
        failure=(
            None
            if converged
            else _classify_one_step(epsilon, current, alpha, steps, max_steps, mask)
        ),
        loglik=loglik,
        absolute_score_weights=absolute_score_weights(scoring_h, w, mask),
    )


def _solve_one_step_ruled(
    outcome: FloatArray,
    initial: InitialFit,
    submodel: Submodel,
    weights: FloatArray,
    observed: BoolArray | None,
    *,
    target_weights: bool,
    alpha: float,
    step_size: float,
    max_steps: int,
    tol: float,
    warn: bool,
    stopping: str,
    stop: StopContext | None,
) -> Fluctuation:
    """:func:`solve_one_step` with the stop test replaced by a declared stopping rule.

    The walk is the shipped one, step for step, including the halving on an overshoot.  It
    ends at the first accepted step where the largest score over its threshold is at most
    one, so a rule that is looser than ``tol`` stops the same walk earlier.  ``tol`` is not
    read.
    """
    if stop is None:
        raise ValueError(STOP_CONTEXT_MISSING)
    y = np.asarray(outcome, dtype=float).reshape(-1)
    n = y.shape[0]
    mask = np.ones(n, dtype=bool) if observed is None else np.asarray(observed, dtype=bool)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if step_size <= 0:
        raise ValueError(f"step_size must be positive; got {step_size}")

    scoring_h = submodel.observed
    fit_submodel, fit_weights = weighted_form(submodel, w) if target_weights else (submodel, w)

    current = initial.shrunk(alpha)
    epsilon = np.zeros(fit_submodel.dim)
    dx = float(step_size)
    scale = score_scale(scoring_h, w, mask)
    trace: list[float] = []
    loglik = quasi_loglik(y[mask], current.observed[mask], fit_weights[mask])
    steps = 0

    score = score_columns(y, current.observed, scoring_h, w, mask)
    score_before = score
    norm = float(np.linalg.norm(score))
    trace.append(relative_score(score, scale))
    thresholds, ratio = _rule_state(y, current, submodel, w, mask, score, stop, stopping)

    while steps < max_steps and ratio > 1.0:
        if norm == 0.0:
            break
        direction = score / norm
        candidate_epsilon = epsilon + dx * direction
        candidate = apply_logistic(current, fit_submodel, dx * direction, alpha)
        candidate_score = score_columns(y, candidate.observed, scoring_h, w, mask)
        candidate_norm = float(np.linalg.norm(candidate_score))

        if candidate_norm > norm:
            dx *= 0.5
            if dx < 1e-14:
                break
            continue

        epsilon = candidate_epsilon
        current = candidate
        score = candidate_score
        norm = candidate_norm
        loglik = quasi_loglik(y[mask], current.observed[mask], fit_weights[mask])
        trace.append(relative_score(score, scale))
        thresholds, ratio = _rule_state(y, current, submodel, w, mask, score, stop, stopping)
        steps += 1

    converged = bool(ratio <= 1.0)
    if not converged and warn:
        warnings.warn(
            ruled_warning(
                stopping,
                steps,
                ratio,
                f"the walk took steps of at most {step_size:g}. Try a smaller step_size, or "
                "check res.diagnostics.support() for a positivity violation.",
            ),
            ConvergenceWarning,
            stacklevel=3,
        )
    return Fluctuation(
        epsilon=epsilon,
        targeted=current,
        score=score,
        converged=converged,
        n_iter=steps,
        trace=tuple(trace),
        method="one_step",
        names=submodel.names,
        score_scale=scale,
        score_initial=score_before,
        failure=(
            None
            if converged
            else _classify_one_step(epsilon, current, alpha, steps, max_steps, mask)
        ),
        loglik=loglik,
        absolute_score_weights=absolute_score_weights(scoring_h, w, mask),
        stopping=stopping,
        thresholds=thresholds,
    )


def _classify_one_step(
    epsilon: FloatArray,
    current: InitialFit,
    alpha: float,
    steps: int,
    max_steps: int,
    mask: BoolArray,
) -> TargetingFailure:
    """Why the walk stopped short of the root.

    The universal least-favorable submodel has no Hessian to be singular -- it walks
    a normalised direction -- so the modes here are the endpoint ones: epsilon
    running away, predictions pinned on their bounds, or the step cap.
    """
    if epsilon.size and np.max(np.abs(epsilon)) >= SEPARATION_EPSILON:
        return "separation_suspected"
    edge = 1.0 - alpha
    scored = current.observed[mask]
    pinned = np.mean((scored <= edge * 1.000001) | (scored >= alpha * 0.999999))
    if pinned > 0.01:
        return "bounds_pinned"
    if steps >= max_steps:
        return "max_iter_reached"
    # The walk halves dx on every overshoot and bails at 1e-14; reaching that without
    # solving the equation is the same stall as an exhausted line search.
    return "line_search_exhausted"
