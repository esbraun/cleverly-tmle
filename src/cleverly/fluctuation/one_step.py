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
from ..utils.bounds import expit, logit
from ._score import quasi_loglik, relative_score, score_columns, score_scale
from .iterative import Fluctuation, InitialFit
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
    """
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
    norm = float(np.linalg.norm(score))
    trace.append(relative_score(score, scale))

    while steps < max_steps and relative_score(score, scale) > tol:
        if norm == 0.0:
            break
        direction = score / norm
        candidate_epsilon = epsilon + dx * direction
        candidate = _move(current, fit_submodel, dx * direction, alpha)
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
            "res.sensitivity.positivity() for a positivity violation.",
            ConvergenceWarning,
            stacklevel=2,
        )
    del loglik  # tracked for monotonicity during development; not part of the result
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
    )


def _move(fit: InitialFit, submodel: Submodel, delta: FloatArray, alpha: float) -> InitialFit:
    """One step along the submodel, at the observed and both counterfactual arms."""
    return InitialFit(
        expit(logit(fit.observed) + submodel.observed @ delta),
        expit(logit(fit.at_one) + submodel.at_one @ delta),
        expit(logit(fit.at_zero) + submodel.at_zero @ delta),
    ).shrunk(alpha)
