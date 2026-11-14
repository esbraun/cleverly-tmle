"""The quasibinomial learner every study with a fractional target fits.

A fractional outcome is one that lies in ``[0, 1]`` without being a class label: a
longitudinal pseudo-outcome carried back from a later node, or a proportion drawn from a
beta law.  Scikit-learn's logistic classifier refuses such a target, so the studies carry
the score solver instead of silently comparing a different regression family.

The class moved here from :mod:`tests.studies.canonical_ltmle`, unchanged, because the
bounded point-treatment laws in :mod:`tests.studies.bounded_cv_laws` need the same learner
and the longitudinal study is not where a point-treatment study should reach for it.  The
move is result-neutral: the bytes of :class:`QuasiBinomialGLM` are identical, so every
study that already fitted it fits the same numbers.  ``tests/canonical/provenance-revisions.md``
records that judgement for each study whose manifest predates the move.

A fit whose coefficients do not converge in ``max_iter`` steps used to raise.  It now returns
its last iterate with :class:`QuasiBinomialSeparationWarning` when the deviance has converged
by R's ``glm.control`` rule, as R's ``glm`` returns its fit with a warning.  That is the
quasi-complete separation case: a covariate cell with no event, whose coefficient diverges
while the fitted values and the deviance settle.  Every fit that converged before takes the
same iterates and stops at the same step, because only the branch that raised changed.  A fit
whose deviance has not settled still raises.

A fit whose binary target is completely separated also keeps its last iterate, with the same
warning.  There the maximum-likelihood estimate does not exist: the coefficients grow without
bound, the fitted values tend to 0 and 1 on the separated rows, and the deviance tends to zero
geometrically, so its relative change never meets R's rule.  The learner detects the case
directly and scale-free, by Albert and Anderson's (1984) characterization: the target is binary
on the rows of positive weight and some linear predictor puts every such row strictly on the
side of its class.  A small linear program decides that (:func:`completely_separated`), so the
test does not depend on how far IRLS has run.  A fractional target never meets it, and neither
does a quasi-complete separation, where some row must lie on the hyperplane; that case keeps the
deviance rule above.  The last iterate approximates the limiting predictions, and a fit that
converged before takes the same iterates and stops at the same step.

A third case is a fractional target that is binary to within
:data:`NUMERICAL_BINARY_TOLERANCE`: a pseudo-outcome carried back from a separated later fit,
with values such as ``1e-289`` and ``1 - 5e-8``. Its deviance tends to zero as under complete
separation, and the relative rule is not met in ``max_iter`` steps, so the learner raised on
about 1% of the cross-fitted draws of ``boundary-node-ltmle``'s pilot. The farthest such value
from 0 or 1 in that pilot was ``2.5e-6``, and the tolerance is ``1e-4``. With
``numerical_separation=True``, when the target rounded to 0 and 1 is completely separated
(:func:`numerically_separated`), the fit keeps its last iterate with the same warning. The
branch is reached only where the learner raised or already kept the last iterate, so no fit
that converged, and no last iterate it kept, changes. It is opt-in, and only
``boundary-node-ltmle`` opts in: the long-horizon studies refit this learner inside a
bootstrap, which drops a replicate that raises, so turning it on for every study could move a
committed bootstrap interval.

No committed failure record carries the text the learner raised, so no committed run reached
the branch that raised in a fit whose error would stop it. No registered study puts this
learner inside a ``SuperLearner`` library. The long-horizon studies
(:mod:`tests.studies.long_horizon_common`) do refit it inside a bootstrap with
``n_bootstrap=199``, where a replicate that raises is dropped rather than failing the run, so
the first two changes are not shown to leave those bootstrap rows unchanged.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
from scipy.optimize import linprog
from scipy.special import expit, xlogy
from sklearn.base import BaseEstimator

#: R's ``glm.control(epsilon=)``: the relative deviance change at which ``glm`` stops.
DEVIANCE_EPSILON = 1e-8


#: How far from 0 or 1 a fractional target may lie and still be read as binary by
#: :func:`numerically_separated`.
NUMERICAL_BINARY_TOLERANCE = 1e-4


def numerically_separated(design: np.ndarray, target: np.ndarray, weights: np.ndarray) -> bool:
    """Whether a target within :data:`NUMERICAL_BINARY_TOLERANCE` of binary is separated.

    A pseudo-outcome carried back from a completely separated later fit holds values such as
    ``1e-289`` and ``1 - 5e-8`` where the limit holds 0 and 1. Its deviance then tends to zero
    as in complete separation, and :func:`completely_separated` misses it only because the
    values are not exactly 0 and 1. This reads the target rounded to 0 and 1 when every
    positive-weight value lies within the tolerance of one of them.

    Parameters
    ----------
    design : ndarray
        The ``(n, p)`` design, intercept column included.
    target : ndarray
        The ``(n,)`` target.
    weights : ndarray
        The ``(n,)`` row weights. Rows of zero weight do not count.

    Returns
    -------
    bool
        Whether the rounded positive-weight rows are completely separated.
    """
    observed = target[weights > 0]
    if np.max(np.minimum(np.abs(observed), np.abs(1.0 - observed))) > NUMERICAL_BINARY_TOLERANCE:
        return False
    return completely_separated(design, np.round(target), weights)


class QuasiBinomialSeparationWarning(RuntimeWarning):
    """The coefficients diverged under separation, so the last iterate is kept."""


def deviance(target: np.ndarray, fitted: np.ndarray, weights: np.ndarray) -> float:
    """The weighted binomial deviance of a fractional target, with ``0 log 0 = 0``."""
    terms = xlogy(target, target) - xlogy(target, fitted)
    terms += xlogy(1.0 - target, 1.0 - target) - xlogy(1.0 - target, 1.0 - fitted)
    return float(2.0 * np.sum(weights * terms))


def completely_separated(design: np.ndarray, target: np.ndarray, weights: np.ndarray) -> bool:
    """Whether a hyperplane puts every positive-weight row strictly on the side of its class.

    That is complete separation of a binary target (Albert and Anderson, 1984), the case in
    which the maximum-likelihood estimate does not exist.  A strict separator can be scaled to a
    margin of one, so the test is the feasibility of ``s_i x_i' b >= 1`` for every row with
    positive weight, where ``s_i`` is ``+1`` for ``y_i = 1`` and ``-1`` for ``y_i = 0`` and
    ``x_i`` includes the intercept column.  It reads the design and the target, not an iterate.
    A target with a fractional value on such a row is never separated.

    Parameters
    ----------
    design : ndarray
        The ``(n, p)`` design, intercept column included.
    target : ndarray
        The ``(n,)`` target.
    weights : ndarray
        The ``(n,)`` row weights. Rows of zero weight do not count.

    Returns
    -------
    bool
        Whether the positive-weight rows are completely separated.
    """
    informative = weights > 0
    observed = target[informative]
    if not np.all((observed == 0.0) | (observed == 1.0)):
        return False
    signs = np.where(observed == 1.0, 1.0, -1.0)
    rows = design[informative] * signs[:, None]
    solution = linprog(
        np.zeros(design.shape[1]),
        A_ub=-rows,
        b_ub=-np.ones(rows.shape[0]),
        bounds=[(None, None)] * design.shape[1],
        method="highs",
    )
    return bool(solution.status == 0)


class QuasiBinomialGLM(BaseEstimator):
    """Small scikit-compatible unpenalized quasibinomial IRLS learner.

    R ``ltmle`` uses ``glm(..., family=quasibinomial())`` for both the binary final
    outcome and fractional earlier pseudo-outcomes.  Scikit-learn's logistic classifier
    refuses fractional targets, so the canonical study carries the corresponding score
    solver rather than silently comparing different regression families.

    Under complete separation of a binary target the maximum-likelihood estimate does not
    exist, and the predictions tend to 0 and 1 on the separated rows.  The fit then keeps the
    last iterate, which approximates those limits, with
    :class:`QuasiBinomialSeparationWarning` (:func:`completely_separated`).
    """

    #: Unpenalized, and started from the weighted mean, so a row of integer weight ``c``
    #: fits as ``c`` copies of the row (:mod:`cleverly.longitudinal.frequency`).
    frequency_exact = True

    def __init__(
        self, *, max_iter: int = 100, tol: float = 1e-10, numerical_separation: bool = False
    ) -> None:
        self.max_iter = max_iter
        self.tol = tol
        self.numerical_separation = numerical_separation

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> QuasiBinomialGLM:
        matrix = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float).reshape(-1)
        weights = (
            np.ones_like(target)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float).reshape(-1)
        )
        design = np.column_stack([np.ones(len(matrix)), matrix])
        mean = float(np.average(target, weights=weights))
        coefficient = np.zeros(design.shape[1], dtype=float)
        coefficient[0] = math.log(np.clip(mean, 1e-8, 1.0 - 1e-8) / np.clip(1.0 - mean, 1e-8, 1.0))
        previous = coefficient
        for _ in range(self.max_iter):
            previous = coefficient
            fitted = expit(design @ coefficient)
            variance = np.clip(fitted * (1.0 - fitted), 1e-10, None)
            working = design @ coefficient + (target - fitted) / variance
            root_weight = np.sqrt(weights * variance)
            updated = np.linalg.lstsq(
                design * root_weight[:, None], working * root_weight, rcond=None
            )[0]
            if np.max(np.abs(updated - coefficient)) <= self.tol:
                coefficient = updated
                break
            coefficient = updated
        else:
            last = expit(design @ coefficient)
            before = deviance(target, expit(design @ previous), weights)
            after = deviance(target, last, weights)
            if after <= before and (
                completely_separated(design, target, weights)
                or (self.numerical_separation and numerically_separated(design, target, weights))
            ):
                warnings.warn(
                    f"quasibinomial IRLS coefficients did not converge in {self.max_iter} "
                    "steps: complete separation, last iterate kept",
                    QuasiBinomialSeparationWarning,
                    stacklevel=2,
                )
            else:
                if abs(after - before) / (abs(after) + 0.1) >= DEVIANCE_EPSILON:
                    raise RuntimeError("quasibinomial IRLS did not converge")
                warnings.warn(
                    f"quasibinomial IRLS coefficients did not converge in {self.max_iter} "
                    "steps, but the deviance did: quasi-complete separation, last iterate kept",
                    QuasiBinomialSeparationWarning,
                    stacklevel=2,
                )
        self.coef_ = coefficient[1:][None, :]
        self.intercept_ = coefficient[:1]
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict(self, X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        return expit(self.intercept_[0] + matrix @ self.coef_[0])

    def predict_proba(self, X: Any) -> np.ndarray:
        probability = self.predict(X)
        return np.column_stack([1.0 - probability, probability])
