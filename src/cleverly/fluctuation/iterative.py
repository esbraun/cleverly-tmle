r"""Solving the fluctuation: the targeting step.

Given an initial outcome regression and a clever covariate, find the
:math:`\epsilon` that maximises the (quasi-)likelihood along the submodel.  For a
logistic fluctuation this is a weighted logistic regression with the initial
prediction as an offset, solved here by Newton--Raphson with an explicit Hessian
and a backtracking line search.

Why not hand this to :mod:`statsmodels` or scikit-learn?  Neither exposes a
regression with an *offset* and no intercept, which is exactly what the submodel
requires; and the problem is one- or two-dimensional with a closed-form Hessian,
so a direct Newton solve is both faster and more accurate than a general-purpose
optimiser.  A brute-force grid search is used as a reference in the tests.

Because the solution is a maximum-likelihood estimate *within* the submodel, its
score is exactly zero -- meaning the resulting estimator solves the efficient
influence-function equation.  When the targeted predictions hit the ``[0, 1]``
boundary the score can fail to vanish; the outer loop then re-fluctuates from the
updated fit, which is what makes this the "iterative" TMLE.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .._typing import BoolArray, FloatArray, FluctuationKind, IntArray
from ..exceptions import ConvergenceWarning
from ..utils.bounds import expit, logit, shrink_probabilities
from .submodel import Submodel, weighted_form

__all__ = ["Fluctuation", "FoldFluctuation", "InitialFit", "solve_fluctuation"]

TargetingLabel = Literal["iterative", "one_step", "linear"]

#: Relative slack allowed when checking that a Newton step did not reduce the
#: quasi-log-likelihood.  See :func:`_newton_logistic` for why it must be relative.
_LINE_SEARCH_SLACK = 1e-11


@dataclass(frozen=True)
class InitialFit:
    """The initial outcome regression, evaluated at the observed and both arms.

    All three arrays live on the ``[0, 1]`` scale: for a binary outcome that is the
    natural scale, and for a continuous one it is the scaled outcome (see
    :class:`cleverly.utils.bounds.OutcomeScaler`).
    """

    observed: FloatArray
    at_one: FloatArray
    at_zero: FloatArray

    def shrunk(self, alpha: float) -> InitialFit:
        """Pull predictions away from 0 and 1 so ``logit`` stays finite."""
        return InitialFit(
            shrink_probabilities(self.observed, alpha),
            shrink_probabilities(self.at_one, alpha),
            shrink_probabilities(self.at_zero, alpha),
        )

    @property
    def n(self) -> int:
        return int(self.observed.shape[0])


@dataclass(frozen=True)
class FoldFluctuation:
    """One validation fold's contribution to a cross-validated targeting step.

    Recorded so a CV-TMLE fit can be inspected fold by fold.  A fluctuation
    coefficient that swings wildly across folds is the signature of an unstable
    clever covariate -- something the pooled ``epsilon`` averages away and hides.
    """

    index: IntArray
    epsilon: FloatArray
    score: FloatArray
    converged: bool
    n_iter: int

    @property
    def n(self) -> int:
        return int(self.index.shape[0])


@dataclass(frozen=True)
class Fluctuation:
    """The result of a targeting step.

    Attributes
    ----------
    epsilon:
        Fitted fluctuation coefficients, one per clever-covariate column.
    targeted:
        Targeted predictions at the observed treatment, and at ``A = 1`` / ``A = 0``.
    score:
        Mean of ``w * h * (Y - Q*)`` per column -- the estimating equation the
        targeting step is meant to zero out.  Reported rather than asserted so
        :mod:`cleverly.validation.score` can check it against the standard error.
    score_scale:
        Per-column ``mean(|w * h|)``, the largest the score could possibly be given
        that the residual is bounded by one on the ``[0, 1]`` outcome scale.  Dividing
        by it turns the score into a dimensionless quantity, which is what makes a
        single default tolerance meaningful across problems whose clever covariates
        differ by orders of magnitude.
    converged:
        Whether the *relative* score norm reached ``tol``.
    trace:
        Relative score norm after each outer iteration, for diagnosing a targeting step
        that stalls against the prediction bounds.
    folds:
        Per-fold detail, populated only by the cross-validated (``targeting_scheme=
        "fold"``) targeting step and empty otherwise.
    """

    epsilon: FloatArray
    targeted: InitialFit
    score: FloatArray
    converged: bool
    n_iter: int
    trace: tuple[float, ...]
    method: TargetingLabel
    names: tuple[str, ...]
    score_scale: FloatArray | None = None
    folds: tuple[FoldFluctuation, ...] = ()

    @property
    def score_norm(self) -> float:
        """Largest absolute score component, in the outcome's own units."""
        return float(np.max(np.abs(self.score))) if self.score.size else 0.0

    @property
    def relative_score_norm(self) -> float:
        """Largest score component relative to its maximum possible magnitude."""
        if self.score.size == 0:
            return 0.0
        if self.score_scale is None:
            return self.score_norm
        return float(np.max(np.abs(self.score) / np.maximum(self.score_scale, 1e-300)))

    def coefficients(self) -> dict[str, float]:
        return dict(zip(self.names, self.epsilon.tolist(), strict=True))


def solve_fluctuation(
    outcome: FloatArray,
    initial: InitialFit,
    submodel: Submodel,
    weights: FloatArray,
    observed: BoolArray | None = None,
    *,
    kind: FluctuationKind = "logistic",
    target_weights: bool = False,
    alpha: float = 0.9995,
    max_iter: int = 20,
    tol: float = 1e-10,
    warn: bool = True,
) -> Fluctuation:
    """Run the targeting step.

    Parameters
    ----------
    outcome:
        Outcome on the ``[0, 1]`` scale.  Values at unobserved rows are ignored.
    initial:
        Initial outcome regression predictions.
    submodel:
        Clever covariates from :mod:`cleverly.fluctuation.submodel`.
    weights:
        Observation weights, normalised to mean one.
    observed:
        Mask of rows with an observed outcome; ``None`` means all rows.  Rows with a
        missing outcome contribute nothing to the fluctuation regression -- their
        clever covariate is multiplied by ``Delta = 0`` in the estimating equation.
    kind:
        ``"logistic"`` keeps the targeted predictions inside ``[0, 1]``;
        ``"linear"`` fluctuates on the identity scale, matching R's
        ``fluctuation="linear"``, and can leave the unit interval.
    target_weights:
        Use the weighted form of the fluctuation (R's ``target.gwt``).
    """
    y = np.asarray(outcome, dtype=float).reshape(-1)
    n = y.shape[0]
    mask = np.ones(n, dtype=bool) if observed is None else np.asarray(observed, dtype=bool)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if submodel.n != n or w.shape[0] != n or initial.n != n:
        raise ValueError(
            "outcome, initial fit, submodel and weights must all have the same length; got "
            f"{n}, {initial.n}, {submodel.n}, {w.shape[0]}"
        )
    if not mask.any():
        raise ValueError("no observed outcomes: the fluctuation has nothing to fit")

    scoring_submodel = submodel
    if kind == "linear":
        return _solve_linear(y, initial, submodel, w, mask)

    fit_submodel, fit_weights = weighted_form(submodel, w) if target_weights else (submodel, w)

    current = initial.shrunk(alpha)
    epsilon = np.zeros(fit_submodel.dim)
    scale = _score_scale(scoring_submodel.observed, w, mask)
    trace: list[float] = []
    iterations = 0

    for iterations in range(1, max_iter + 1):  # noqa: B007 - reported after the loop
        step, step_converged = _newton_logistic(
            fit_submodel.observed[mask],
            y[mask],
            logit(current.observed[mask]),
            fit_weights[mask],
            tol=min(tol, 1e-12),
        )
        epsilon = epsilon + step
        current = _apply_logistic(current, fit_submodel, step, alpha)
        score = _score(y, current.observed, scoring_submodel.observed, w, mask)
        trace.append(_relative(score, scale))
        if trace[-1] <= tol or (step_converged and np.max(np.abs(step)) <= tol):
            break

    score = _score(y, current.observed, scoring_submodel.observed, w, mask)
    relative = _relative(score, scale)
    converged = bool(relative <= tol)
    if not converged and warn:
        warnings.warn(
            f"targeting step did not drive the relative score below {tol:g} after "
            f"{iterations} iteration(s) (relative score = {relative:.3g}). This usually means "
            "the targeted predictions are pinned against their bounds because of a "
            "positivity violation; check res.sensitivity.positivity().",
            ConvergenceWarning,
            stacklevel=2,
        )
    return Fluctuation(
        epsilon=epsilon,
        targeted=current,
        score=score,
        converged=converged,
        n_iter=iterations,
        trace=tuple(trace),
        method="iterative",
        names=submodel.names,
        score_scale=scale,
    )


def _score_scale(h: FloatArray, weights: FloatArray, mask: BoolArray) -> FloatArray:
    """Per-column ``mean(|w * h|)``: the largest the score could be.

    The residual ``Y - Q*`` is bounded by one on the ``[0, 1]`` outcome scale, so this
    bounds ``|score|`` and makes the ratio dimensionless.
    """
    contribution = np.zeros_like(h)
    contribution[mask] = np.abs(weights[mask])[:, None] * np.abs(h[mask])
    return np.asarray(contribution.mean(axis=0), dtype=float)


def _relative(score: FloatArray, scale: FloatArray) -> float:
    """Largest score component relative to its maximum possible magnitude."""
    if score.size == 0:
        return 0.0
    return float(np.max(np.abs(score) / np.maximum(scale, 1e-300)))


def _apply_logistic(
    fit: InitialFit, submodel: Submodel, epsilon: FloatArray, alpha: float
) -> InitialFit:
    """Move the predictions along the logistic submodel by ``epsilon``."""
    return InitialFit(
        expit(logit(fit.observed) + submodel.observed @ epsilon),
        expit(logit(fit.at_one) + submodel.at_one @ epsilon),
        expit(logit(fit.at_zero) + submodel.at_zero @ epsilon),
    ).shrunk(alpha)


def _score(
    y: FloatArray,
    q_star: FloatArray,
    h: FloatArray,
    weights: FloatArray,
    mask: BoolArray,
) -> FloatArray:
    """``mean(w * h * (Y - Q*))`` over observed rows, scaled by the full sample.

    The mean is taken over *all* ``n`` rows, not just the observed ones, because the
    estimating equation carries a ``Delta`` factor: unobserved rows contribute a
    genuine zero rather than being excluded from the average.
    """
    residual = np.zeros_like(y)
    residual[mask] = y[mask] - q_star[mask]
    contribution = (weights * residual)[:, None] * h
    return np.asarray(contribution.mean(axis=0), dtype=float)


def _newton_logistic(
    x: FloatArray,
    y: FloatArray,
    offset: FloatArray,
    weights: FloatArray,
    *,
    max_iter: int = 50,
    tol: float = 1e-12,
) -> tuple[FloatArray, bool]:
    """Weighted logistic MLE with an offset and no intercept.

    Returns the coefficient vector and whether the gradient reached ``tol``.  The
    quasi-binomial log-likelihood is valid for any ``y`` in ``[0, 1]``, which is what
    lets the same routine target a scaled continuous outcome (Gruber & van der
    Laan, 2010).
    """
    k = x.shape[1]
    epsilon = np.zeros(k)
    if x.size == 0 or np.allclose(x, 0.0):
        return epsilon, True

    total_weight = weights.sum()
    if total_weight <= 0:
        return epsilon, True

    loglik = _quasi_loglik(y, expit(offset), weights)
    for _ in range(max_iter):
        eta = offset + x @ epsilon
        p = expit(eta)
        gradient = x.T @ (weights * (y - p))
        if np.max(np.abs(gradient)) / total_weight <= tol:
            return epsilon, True

        variance = weights * p * (1.0 - p)
        hessian = x.T @ (x * variance[:, None])
        step = _solve_step(hessian, gradient)
        if step is None:
            return epsilon, False

        # Backtracking: a Newton step can overshoot when the clever covariate has
        # extreme values, and the quasi-likelihood must never decrease.  The slack is
        # *relative* to the magnitude of the log-likelihood: near the optimum the
        # improvement per step falls below the floating-point granularity of a number
        # of size |loglik|, and an absolute slack would reject every remaining step
        # and stall the solver short of the root.
        slack = _LINE_SEARCH_SLACK * max(1.0, abs(loglik))
        scale = 1.0
        for _ in range(30):
            candidate = epsilon + scale * step
            candidate_loglik = _quasi_loglik(y, expit(offset + x @ candidate), weights)
            if candidate_loglik >= loglik - slack:
                epsilon, loglik = candidate, candidate_loglik
                break
            scale *= 0.5
        else:
            return epsilon, False

        if np.max(np.abs(scale * step)) <= tol:
            return epsilon, True
    return epsilon, False


def _solve_step(hessian: FloatArray, gradient: FloatArray) -> FloatArray | None:
    """Newton step, falling back to a pseudo-inverse for a singular Hessian."""
    try:
        step = np.linalg.solve(hessian, gradient)
    except np.linalg.LinAlgError:
        step = np.linalg.pinv(hessian) @ gradient
    if not np.all(np.isfinite(step)):
        return None
    return np.asarray(step, dtype=float)


def _quasi_loglik(y: FloatArray, p: FloatArray, weights: FloatArray) -> float:
    """Weighted binomial quasi-log-likelihood, valid for ``y`` in ``[0, 1]``."""
    q = np.clip(p, 1e-15, 1.0 - 1e-15)
    return float(np.sum(weights * (y * np.log(q) + (1.0 - y) * np.log(1.0 - q))))


def _solve_linear(
    y: FloatArray,
    initial: InitialFit,
    submodel: Submodel,
    weights: FloatArray,
    mask: BoolArray,
) -> Fluctuation:
    """Fluctuate on the identity scale: a single weighted least-squares solve.

    This is R's ``fluctuation="linear"``.  The normal equations *are* the estimating
    equation, so the score is solved exactly in one step -- at the cost of targeted
    predictions that may fall outside the outcome's range.
    """
    x = submodel.observed[mask]
    residual = y[mask] - initial.observed[mask]
    w = weights[mask]
    lhs = x.T @ (x * w[:, None])
    rhs = x.T @ (w * residual)
    step = _solve_step(lhs, rhs)
    epsilon = np.zeros(submodel.dim) if step is None else step

    targeted = InitialFit(
        initial.observed + submodel.observed @ epsilon,
        initial.at_one + submodel.at_one @ epsilon,
        initial.at_zero + submodel.at_zero @ epsilon,
    )
    escaped = (
        targeted.at_one.min() < 0.0
        or targeted.at_one.max() > 1.0
        or targeted.at_zero.min() < 0.0
        or targeted.at_zero.max() > 1.0
    )
    if escaped:
        warnings.warn(
            "the linear fluctuation produced targeted predictions outside the outcome's "
            "range. fluctuation='logistic' (the default) is bounded by construction and is "
            "preferred unless you specifically need the linear submodel.",
            UserWarning,
            stacklevel=3,
        )
    score = _score(y, targeted.observed, submodel.observed, weights, mask)
    scale = _score_scale(submodel.observed, weights, mask)
    return Fluctuation(
        epsilon=epsilon,
        targeted=targeted,
        score=score,
        converged=bool(_relative(score, scale) <= 1e-8),
        n_iter=1,
        trace=(_relative(score, scale),),
        method="linear",
        names=submodel.names,
        score_scale=scale,
    )
