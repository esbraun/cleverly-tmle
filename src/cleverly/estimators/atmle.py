r"""Adaptive TMLE of a trial's effect, with external data pooled in.

The estimator of van der Laan, Qiu, Tarp and van der Laan (arXiv:2405.07186v2). The data are
``O = (S, W, A, Y)``, with ``S = 1`` on trial rows and ``S = 0`` on external rows, and with
an observation indicator ``Delta`` when outcomes can be missing. Write ``g(W) = P(A = 1 | W)``
and ``theta(W) = E(Y | W)`` on the pooled rows, ``Pi(W, a) = P(S = 1 | W, A = a)``,
``Qt(W, a) = E(Y | W, A = a)`` and ``pi(W) = P(S = 1 | W)``. The target is the trial's
conditional effect averaged over the pooled covariates (``population="pooled"``, the paper's
``Psi``) or over the trial's (``population="trial"``, ``Psi_2``). Lemma 2 writes it as

.. math::

    \Psi = E[\tau_A(W)] - E[\Pi(0 \mid W, 0)\,\tau_S(W, 0) - \Pi(0 \mid W, 1)\,\tau_S(W, 1)],

the pooled effect minus the bias of the external rows. Two lasso fits over a dictionary
(:mod:`cleverly.estimators._working_basis`) select working models for ``tau_A`` and
``tau_S``, a least-squares refit solves their score equations, and a logistic fluctuation of
``Pi`` alternates with the refit of the bias model until the remaining score is below
``sd(D) / (sqrt(n) log n)``.

Every quantity is computed under a measure: the empirical measure of a fold's rows in the
estimator, or the masses of a finite law in the exact-law tests. :class:`WorkingUnit` holds
one measure's rows and :func:`evaluate` returns the estimate and its influence function
there, so the two uses run the same arithmetic.

``form="pooled"`` is Algorithm 1 of the paper (Section 7) with cross-fitted nuisances.
``form="cv"`` is the adaptive CV-TMLE of Appendix C, SR(1) step 4(b). Fold ``v``'s
nuisances, dictionary, selection and least-squares refit read only its training rows ``T_v``.
Each refitted coefficient vector then moves along its target direction ``I^{-1} m``, taken on
``T_v``, by one step pooled over the validation folds, and one fluctuation of ``Pi`` is pooled
over the validation folds the same way. The estimate is the average of the validation-fold
plug-ins.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import scipy.linalg
from scipy.special import expit, logit

from .._typing import BoolArray, FloatArray, IntArray, Learner
from .._warnings import warn as _warn_at
from ..exceptions import (
    CapabilityError,
    ConvergenceWarning,
    DataError,
    MethodConfigurationError,
    PositivityWarning,
    RefusalReason,
)
from ..learners._fitting import Task, fit_learner, predict_mean
from ..learners.crossfit import Folds, make_folds, random_partition
from ._working_basis import WorkingDictionary

__all__ = [
    "KKT_TOLERANCE",
    "MAX_PASSES",
    "SOURCE_FLOOR",
    "VARIANCE_ROWS_PER_COLUMN",
    "AdaptiveTMLEDiagnostics",
    "AdaptiveTMLEFit",
    "InfluenceParts",
    "LassoSolution",
    "Nuisances",
    "Refit",
    "ScoreResiduals",
    "Selection",
    "WorkingModel",
    "WorkingUnit",
    "arm_hierarchy",
    "bias_term",
    "evaluate",
    "fit_adaptive_tmle",
    "kkt_residual",
    "lasso_solutions",
    "penalty_grid",
    "pooled_epsilon",
    "pseudo_responses",
    "refit_unit",
    "relaxed_refit",
    "residuals",
    "select_working_model",
    "source_covariates",
    "stratified_partition",
    "target_directions",
    "target_source",
    "variance_directions",
]

Form = Literal["cv", "pooled"]
Population = Literal["pooled", "trial"]

#: The largest accepted Karush-Kuhn-Tucker residual of a lasso solution, relative to the
#: smallest penalty at which every penalized coefficient is zero.
KKT_TOLERANCE = 1e-8

#: The most passes of the alternation of the ``Pi`` fluctuation and the bias-model refit.
MAX_PASSES = 50

#: An external row's ``pi(W)`` below this value is reported as a trial-enrollment
#: positivity concern (condition A3).
SOURCE_FLOOR = 0.01

#: The fewest rows per selected column at which the CV form's variance estimates a working
#: model's ``I`` and ``m`` (:func:`variance_directions`). Below it the fold's variance uses
#: the training rows' direction, which is consistent but adds that direction's estimation
#: error to the variance.
VARIANCE_ROWS_PER_COLUMN = 10


def _warn(message: str, category: type[Warning]) -> None:
    """Warn at the first frame outside the ``cleverly`` package, which is the user's call."""
    package = str(Path(__file__).resolve().parent.parent) + os.sep
    frame: Any = sys._getframe(0)
    level = 1
    while frame is not None and os.path.abspath(frame.f_code.co_filename).startswith(package):
        frame = frame.f_back
        level += 1
    _warn_at(message, category, stacklevel=level)


#: The number of log-spaced penalties of the grid, as ``glmnet``'s ``nlambda``.
_N_LAMBDA = 100

#: The relative pivot below which a refit's design is rank deficient.
_RANK_TOLERANCE = 1e-10

#: The score of the ``Pi`` fluctuation that counts as solved, relative to ``sum w |H|``.
_SOURCE_TOLERANCE = 1e-13


# ----------------------------------------------------------------------- the lasso


@dataclass(frozen=True)
class LassoSolution:
    """One solution of the R-loss lasso.

    Parameters
    ----------
    penalty : float
        The penalty ``lambda``.
    coef : ndarray of float
        The coefficients of the dictionary columns on their original scale.
    kkt : float
        The Karush-Kuhn-Tucker residual of the solution (:func:`kkt_residual`).
    """

    penalty: float
    coef: FloatArray
    kkt: float


@dataclass(frozen=True)
class Selection:
    """A working model chosen by cross-validated R-loss lasso.

    Parameters
    ----------
    penalty : float
        The chosen penalty, the largest that attains the smallest cross-validated risk.
    selected : ndarray of int
        The unpenalized columns and the columns with a nonzero coefficient, ascending. In
        the bias model it also holds ``A`` when another column carrying ``A`` is selected
        (:func:`arm_hierarchy`).
    coef : ndarray of float
        The lasso coefficients at the chosen penalty.
    grid : ndarray of float
        The penalty grid, strictly decreasing.
    risk : ndarray of float
        The cross-validated risk at each penalty.
    kkt : float
        The Karush-Kuhn-Tucker residual of the solution at the chosen penalty.
    """

    penalty: float
    selected: IntArray
    coef: FloatArray
    grid: FloatArray
    risk: FloatArray
    kkt: float


class _RLoss:
    r"""The R-loss lasso on one set of rows, with its leading columns unpenalized.

    The objective in ``b`` is

    .. math::

        \frac{1}{2N} \sum_i \{r_i - m_i \phi(W_i)^\top b\}^2
        + \lambda \sum_{j \ge k} s_j |b_j|, \qquad N = \sum_i m_i^2,

    with ``m`` the multiplier (``A - g`` or ``S - Pi``), ``r`` the response and ``s_j`` the
    standard deviation of column ``j`` under the weights ``m^2``. It is ``cv.glmnet``'s
    weighted least squares of ``r / m`` on ``phi`` with weights ``m^2`` and standardized
    columns, written without the division. The ``k`` unpenalized columns are partialled out
    exactly (Frisch-Waugh) after the scaling, so the penalty scales do not change.
    """

    def __init__(
        self, basis: FloatArray, multiplier: FloatArray, response: FloatArray, n_unpenalized: int
    ) -> None:
        self.basis = np.asarray(basis, dtype=float)
        self.multiplier = np.asarray(multiplier, dtype=float)
        self.response = np.asarray(response, dtype=float)
        self.k = int(n_unpenalized)
        self.x = self.multiplier[:, None] * self.basis
        weights = self.multiplier**2
        self.total = float(np.sum(weights))
        if not self.total > 0.0:
            raise ValueError("the R-loss lasso needs a nonzero multiplier on some row")
        penalized = self.basis[:, self.k :]
        center = (weights @ penalized) / self.total
        scale = np.sqrt((weights @ (penalized - center) ** 2) / self.total)
        informative = weights > 0.0
        constant = np.ptp(penalized[informative], axis=0) == 0.0
        scale = np.where(constant | ~(scale > 0.0), 0.0, scale)
        self.usable = np.flatnonzero(scale > 0.0) + self.k
        self.scale = scale[self.usable - self.k]
        q, r = np.linalg.qr(self.x[:, : self.k])
        diagonal = np.abs(np.diag(r))
        if self.k and diagonal.min() <= _RANK_TOLERANCE * max(diagonal.max(), 1e-300):
            raise ValueError("the unpenalized working-model columns are collinear on these rows")
        self.q, self.r = q, r
        self.y = self.response - q @ (q.T @ self.response)
        z = self.x[:, self.usable]
        self.z = (z - q @ (q.T @ z)) / self.scale
        self.lambda_max = (
            float(np.max(np.abs(self.z.T @ self.y))) / self.total if self.usable.size else 0.0
        )

    def grid(self) -> FloatArray:
        n, p = self.basis.shape
        if self.lambda_max == 0.0:
            return np.array([0.0])
        ratio = 1e-4 if n >= p else 1e-2
        grid = self.lambda_max * np.exp(np.linspace(0.0, np.log(ratio), _N_LAMBDA))
        grid[0] = self.lambda_max
        return np.asarray(grid, dtype=float)

    def coefficients(self, standardized: FloatArray) -> FloatArray:
        """Original-scale coefficients from the penalized standardized ones."""
        coef = np.zeros(self.basis.shape[1])
        coef[self.usable] = standardized / self.scale
        if self.k:
            rest = self.response - self.x[:, self.usable] @ coef[self.usable]
            coef[: self.k] = scipy.linalg.solve_triangular(self.r, self.q.T @ rest)
        return coef

    def kkt(self, coef: FloatArray, penalty: float) -> float:
        gradient = self.x.T @ (self.x @ coef - self.response) / self.total
        values = [float(np.max(np.abs(gradient[: self.k]))) if self.k else 0.0]
        if self.usable.size:
            standardized = gradient[self.usable] / self.scale
            beta = coef[self.usable]
            nonzero = beta != 0.0
            if np.any(nonzero):
                values.append(
                    float(np.max(np.abs(standardized[nonzero] + penalty * np.sign(beta[nonzero]))))
                )
            if np.any(~nonzero):
                values.append(
                    float(np.max(np.maximum(np.abs(standardized[~nonzero]) - penalty, 0)))
                )
        return max(values)

    def tolerance(self) -> float:
        return KKT_TOLERANCE * (self.lambda_max if self.lambda_max > 0.0 else 1.0)

    def solve(self, grid: FloatArray) -> list[LassoSolution]:
        values = np.asarray(grid, dtype=float)
        if values.size > 1 and np.any(np.diff(values) >= 0.0):
            raise ValueError("the penalty grid must be strictly decreasing")
        if np.any(values < 0.0):
            raise ValueError("penalties must be non-negative")
        out: list[LassoSolution] = []
        gram = self.z.T @ self.z / self.total
        linear = self.z.T @ self.y / self.total
        standardized = np.zeros(self.usable.size)
        for lam in values:
            if self.usable.size and lam > 0.0:
                standardized = _lasso(gram, linear, float(lam), standardized)
            elif self.usable.size:
                standardized = np.linalg.lstsq(self.z, self.y, rcond=None)[0]
            coef = self.coefficients(standardized)
            out.append(LassoSolution(float(lam), coef, self.kkt(coef, float(lam))))
        return out


def _lasso_objective(gram: FloatArray, linear: FloatArray, penalty: float, c: FloatArray) -> float:
    return float(0.5 * c @ gram @ c - linear @ c + penalty * np.sum(np.abs(c)))


def _slack(linear: FloatArray, penalty: float) -> float:
    """The optimality slack of the inner solvers, ``1e-12`` of the problem's scale."""
    return 1e-12 * max(penalty, float(np.max(np.abs(linear))), 1e-300)


def _optimality(gram: FloatArray, linear: FloatArray, penalty: float, c: FloatArray) -> float:
    """The Karush-Kuhn-Tucker residual of ``c'Gc / 2 - b'c + lambda |c|_1`` at ``c``."""
    gradient = gram @ c - linear
    nonzero = c != 0.0
    residual = np.where(
        nonzero,
        np.abs(gradient + penalty * np.sign(c)),
        np.maximum(np.abs(gradient) - penalty, 0.0),
    )
    return float(np.max(residual)) if residual.size else 0.0


def _lasso(gram: FloatArray, linear: FloatArray, penalty: float, start: FloatArray) -> FloatArray:
    """Minimize ``c'Gc / 2 - b'c + lambda |c|_1`` to the slack of :func:`_slack`.

    Feature-sign search (:func:`_feature_sign`) runs first. In floating point it can stop
    short of the optimality conditions, for example when it can no longer lower the
    objective. The problem is convex and always has a solution for ``lambda > 0``, so a
    stall is a solver failure: the search then continues by coordinate descent with an
    exact solve on its support and signs (:func:`_coordinate_descent`), and returns the
    point with the smaller optimality residual.
    """
    c = _feature_sign(gram, linear, penalty, start)
    slack = _slack(linear, penalty)
    residual = _optimality(gram, linear, penalty, c)
    if residual <= slack:
        return c
    fallback = _coordinate_descent(gram, linear, penalty, c)
    return fallback if _optimality(gram, linear, penalty, fallback) < residual else c


def _feature_sign(
    gram: FloatArray, linear: FloatArray, penalty: float, start: FloatArray
) -> FloatArray:
    """Minimize ``c'Gc / 2 - b'c + lambda |c|_1`` exactly by feature-sign search.

    The active-set method of Lee, Battle, Raina and Ng (2007, Algorithm 1), started from
    ``start``, the solution at the previous penalty of the grid. Each step solves the
    quadratic on the active set with its signs fixed and moves to the lowest objective among
    that solution and the points where an active coefficient crosses zero. A zero
    coefficient whose gradient exceeds the penalty joins the active set, with the sign that
    lowers the objective, only once the active coefficients meet their optimality
    conditions, as the algorithm's step 4 requires. Entering a coefficient while the active
    set is still off its optimum lets two near-collinear columns trade places forever, each
    step lowering the objective by less than the last. When the entering coefficient's sign
    flips in the active-set solution and no point of the segment descends, an exact
    coordinate step on the entering coefficient is taken instead, and when a coefficient
    crosses zero too early in the segment to lower the objective in floating point, the
    search moves to that crossing and drops it. The search ends when the optimality
    conditions hold to the slack of :func:`_slack`, or when it can make no progress; the
    caller checks them again.
    """
    c = np.array(start, dtype=float)
    slack = _slack(linear, penalty)
    for _ in range(20 * c.size + 100):
        gradient = gram @ c - linear
        signs = np.sign(c)
        active = np.flatnonzero(signs)
        settled = active.size == 0 or bool(
            np.max(np.abs(gradient[active] + penalty * signs[active])) <= slack
        )
        entering = -1
        zero = c == 0.0
        if settled and np.any(zero):
            candidates = np.where(zero, np.abs(gradient), -np.inf)
            j = int(np.argmax(candidates))
            if candidates[j] > penalty + slack:
                signs[j] = -np.sign(gradient[j])
                entering = j
        if settled and entering < 0:
            return c
        active = np.flatnonzero(signs)
        sub = gram[np.ix_(active, active)]
        rhs = linear[active] - penalty * signs[active]
        try:
            target = np.linalg.solve(sub, rhs)
        except np.linalg.LinAlgError:
            target = np.linalg.lstsq(sub, rhs, rcond=None)[0]
        current = c[active]
        direction = target - current
        moving = (current != 0.0) & (direction != 0.0)
        crossing = np.full(current.size, np.inf)
        crossing[moving] = -current[moving] / direction[moving]
        steps = np.unique(np.append(crossing[(crossing > 0.0) & (crossing < 1.0)], 1.0))
        best, best_value = c, _lasso_objective(gram, linear, penalty, c)
        for step in steps:
            point = current + step * direction
            point[crossing == step] = 0.0
            trial = np.zeros_like(c)
            trial[active] = point
            value = _lasso_objective(gram, linear, penalty, trial)
            if value < best_value:
                best, best_value = trial, value
        if best is c and entering < 0:
            if steps[0] >= 1.0:
                return c
            # A coefficient crosses zero at a step too short to lower the objective in
            # floating point. Moving to that crossing drops it from the active set.
            best = np.zeros_like(c)
            point = current + steps[0] * direction
            point[crossing == steps[0]] = 0.0
            best[active] = point
        elif best is c:
            # The entering coefficient's sign flipped in the active-set solution, so the
            # segment holds no descent. An exact coordinate step on it always descends,
            # because its gradient exceeds the penalty.
            best = c.copy()
            excess = abs(gradient[entering]) - penalty
            best[entering] = -np.sign(gradient[entering]) * excess / gram[entering, entering]
        c = best
    return c


#: The most sweeps of the coordinate-descent fallback of the lasso.
_MAX_SWEEPS = 100_000


def _coordinate_descent(
    gram: FloatArray, linear: FloatArray, penalty: float, start: FloatArray
) -> FloatArray:
    """Minimize ``c'Gc / 2 - b'c + lambda |c|_1`` by cyclic coordinate descent.

    Coordinate descent converges to a minimizer of this convex objective from any start
    when every diagonal entry of ``G`` is positive (Tseng 2001), which holds for the
    standardized columns. Every ten sweeps the quadratic is solved exactly on the current
    support with its signs fixed; that point is returned once it meets the optimality
    conditions to the slack of :func:`_slack`, so the result is the exact solution and not
    an approximation to the descent's own tolerance. Without such a point the last iterate
    is returned.
    """
    c = np.array(start, dtype=float)
    slack = _slack(linear, penalty)
    diagonal = np.diag(gram).copy()
    gradient = gram @ c - linear
    for sweep in range(_MAX_SWEEPS):
        for j in range(c.size):
            rho = diagonal[j] * c[j] - gradient[j]
            new = np.sign(rho) * max(abs(rho) - penalty, 0.0) / diagonal[j]
            change = new - c[j]
            if change != 0.0:
                gradient += change * gram[:, j]
                c[j] = new
        if sweep % 10 == 9:
            active = np.flatnonzero(c)
            polished = np.zeros_like(c)
            if active.size:
                sub = gram[np.ix_(active, active)]
                rhs = linear[active] - penalty * np.sign(c[active])
                polished[active] = np.linalg.lstsq(sub, rhs, rcond=None)[0]
            consistent = np.all(np.sign(polished[active]) == np.sign(c[active]))
            if consistent and _optimality(gram, linear, penalty, polished) <= slack:
                return polished
            gradient = gram @ c - linear
    return c


def penalty_grid(
    basis: FloatArray, multiplier: FloatArray, response: FloatArray, n_unpenalized: int
) -> FloatArray:
    """``cv.glmnet``'s default penalty grid for the R-loss lasso.

    ``lambda_max`` is the smallest penalty at which every penalized coefficient is zero.
    The grid holds 100 log-spaced values from ``lambda_max`` down to ``lambda_max * ratio``,
    with ``ratio = 1e-4`` when the rows are at least as many as the columns and ``1e-2``
    otherwise. With no penalized column that varies, or a response orthogonal to every
    penalized column, the grid is ``[0]``.

    Parameters
    ----------
    basis : ndarray of float
        The ``(n, J)`` dictionary columns.
    multiplier : ndarray of float
        ``A - g`` or ``S - Pi`` on each row.
    response : ndarray of float
        The pseudo-response on each row, such as ``Y - theta``.
    n_unpenalized : int
        The number of leading columns that carry no penalty.

    Returns
    -------
    ndarray of float
        The penalties, strictly decreasing.
    """
    return _RLoss(basis, multiplier, response, n_unpenalized).grid()


def lasso_solutions(
    basis: FloatArray,
    multiplier: FloatArray,
    response: FloatArray,
    n_unpenalized: int,
    grid: Sequence[float] | FloatArray,
) -> list[LassoSolution]:
    r"""Solve the R-loss lasso at each penalty of a grid.

    The objective at penalty ``lambda`` is

    .. math::

        \frac{1}{2N} \sum_i \{r_i - m_i \phi_i^\top b\}^2 + \lambda \sum_{j \ge k} s_j |b_j|,
        \qquad N = \sum_i m_i^2,

    with ``s_j`` the standard deviation of column ``j`` under the weights ``m^2``. A penalized
    column that is constant on the rows takes no coefficient. The leading ``k`` columns are
    partialled out exactly, the rest are solved by feature-sign search, an
    active-set method that solves exactly on the active columns and their signs, along the
    penalty grid. Where the search stalls short of the optimality conditions, coordinate
    descent with an exact solve on its support continues from it, and the solution is
    checked against the optimality conditions. A penalty of 0 gives least squares.

    Parameters
    ----------
    basis : ndarray of float
        The ``(n, J)`` dictionary columns.
    multiplier : ndarray of float
        ``A - g`` or ``S - Pi`` on each row.
    response : ndarray of float
        The pseudo-response on each row.
    n_unpenalized : int
        The number of leading columns that carry no penalty.
    grid : sequence of float
        The penalties, strictly decreasing.

    Returns
    -------
    list of LassoSolution
        One solution per penalty, with its Karush-Kuhn-Tucker residual.
    """
    return _RLoss(basis, multiplier, response, n_unpenalized).solve(np.asarray(grid, float))


def kkt_residual(
    basis: FloatArray,
    multiplier: FloatArray,
    response: FloatArray,
    n_unpenalized: int,
    solution: LassoSolution,
) -> float:
    """The Karush-Kuhn-Tucker residual of a solution of the R-loss lasso.

    The largest of: the absolute gradient in an unpenalized coefficient; for a nonzero
    penalized coefficient on the standardized scale, ``|gradient + lambda sign(b)|``; for a
    zero one, ``max(0, |gradient| - lambda)``.

    Parameters
    ----------
    basis : ndarray of float
        The ``(n, J)`` dictionary columns.
    multiplier : ndarray of float
        ``A - g`` or ``S - Pi`` on each row.
    response : ndarray of float
        The pseudo-response on each row.
    n_unpenalized : int
        The number of leading columns that carry no penalty.
    solution : LassoSolution
        The solution to check, on the original scale of ``basis``.

    Returns
    -------
    float
        The residual.
    """
    problem = _RLoss(basis, multiplier, response, n_unpenalized)
    return problem.kkt(np.asarray(solution.coef, dtype=float), solution.penalty)


def select_working_model(
    basis: FloatArray,
    multiplier: FloatArray,
    response: FloatArray,
    n_unpenalized: int,
    inner: IntArray,
) -> Selection:
    """Choose a working model by the cross-validated R-loss lasso.

    The grid comes from :func:`penalty_grid` on every row. Inner fold ``k`` solves the
    lasso of :func:`lasso_solutions` on the rows outside it and scores the rows inside it by
    ``sum (r - m phi'b)^2``. The risk at a penalty is that sum over the folds divided by
    ``sum m^2``, and the chosen penalty is the largest that attains the smallest risk, as
    ``cv.glmnet``'s ``lambda.min``.

    Parameters
    ----------
    basis : ndarray of float
        The ``(n, J)`` dictionary columns on the selection rows.
    multiplier : ndarray of float
        ``A - g`` or ``S - Pi`` on each selection row.
    response : ndarray of float
        The pseudo-response on each selection row.
    n_unpenalized : int
        The number of leading columns that carry no penalty.
    inner : ndarray of int
        The inner fold of each selection row.

    Returns
    -------
    Selection
        The chosen penalty, the selected columns and the cross-validated risk.

    Raises
    ------
    CapabilityError
        When the solution at the chosen penalty misses the Karush-Kuhn-Tucker tolerance.
    """
    problem = _RLoss(basis, multiplier, response, n_unpenalized)
    grid = problem.grid()
    assignment = np.asarray(inner, dtype=np.int64)
    loss = np.zeros(grid.size)
    if grid.size > 1:
        for fold in np.unique(assignment):
            held = assignment == fold
            sub = _RLoss(basis[~held], multiplier[~held], response[~held], n_unpenalized)
            for k, solution in enumerate(sub.solve(grid)):
                fitted = multiplier[held] * (basis[held] @ solution.coef)
                loss[k] += float(np.sum((response[held] - fitted) ** 2))
    risk = loss / problem.total
    index = int(np.flatnonzero(risk == risk.min())[0])
    solution = problem.solve(grid[: index + 1])[-1]
    if not solution.kkt <= problem.tolerance():
        raise CapabilityError(
            f"The working-model lasso did not reach its optimality tolerance at penalty "
            f"{solution.penalty:.3g}: the Karush-Kuhn-Tucker residual is {solution.kkt:.3g}, "
            f"above {problem.tolerance():.3g}. The declared dictionary is too collinear or too "
            "large for these rows.",
            reason=RefusalReason.INVALID_DECLARATION,
            alternative=(
                "Pass a smaller or less collinear basis=, for example a transformer with fewer "
                "knots and no pairwise products."
            ),
        )
    selected = np.union1d(
        np.arange(problem.k, dtype=np.int64),
        np.flatnonzero(solution.coef[problem.k :] != 0.0).astype(np.int64) + problem.k,
    )
    return Selection(solution.penalty, selected, solution.coef, grid, risk, solution.kkt)


@dataclass(frozen=True)
class Refit:
    """A least-squares refit of a working model.

    Parameters
    ----------
    beta : ndarray of float
        The coefficients.
    score : ndarray of float
        The normal-equation residual ``sum_i w_i m_i phi_i (r_i - m_i phi_i'beta)``.
    relative : float
        ``max |score|`` divided by ``max_j sum_i w_i |m_i phi_ij r_i|``.
    """

    beta: FloatArray
    score: FloatArray
    relative: float


def relaxed_refit(
    basis: FloatArray,
    multiplier: FloatArray,
    response: FloatArray,
    weights: FloatArray,
    *,
    where: str = "The refit",
) -> Refit:
    """Refit a working model by weighted least squares of the response on ``m phi``.

    This solves the working model's score equations: ``P U = 0`` for the influence function
    ``U`` of its coefficients, under the measure the weights define.

    Parameters
    ----------
    basis : ndarray of float
        The ``(n, J)`` selected columns.
    multiplier : ndarray of float
        ``A - g`` or ``S - Pi`` on each row.
    response : ndarray of float
        The pseudo-response on each row.
    weights : ndarray of float
        The measure on the rows.
    where : str, default="The refit"
        The subject of the refusal when the columns are collinear.

    Returns
    -------
    Refit
        The coefficients and the score residual.

    Raises
    ------
    CapabilityError
        When the columns are collinear on the rows with positive weight.
    """
    x = np.asarray(multiplier, dtype=float)[:, None] * np.asarray(basis, dtype=float)
    w = np.asarray(weights, dtype=float)
    r = np.asarray(response, dtype=float)
    root = np.sqrt(w)
    if x.shape[1] > int(np.sum((w > 0.0) & np.any(x != 0.0, axis=1))):
        _collinear(where)
    q, factor = np.linalg.qr(root[:, None] * x)
    diagonal = np.abs(np.diag(factor))
    if diagonal.min() <= _RANK_TOLERANCE * max(diagonal.max(), 1e-300):
        _collinear(where)
    beta = scipy.linalg.solve_triangular(factor, q.T @ (root * r))
    score = x.T @ (w * (r - x @ beta))
    scale = float(np.max(np.abs(x).T @ (w * np.abs(r))))
    relative = float(np.max(np.abs(score))) / scale if scale > 0.0 else 0.0
    return Refit(np.asarray(beta, dtype=float), np.asarray(score, dtype=float), relative)


def _collinear(where: str) -> None:
    raise CapabilityError(
        f"{where}: the selected working-model columns are collinear on the rows they are "
        "refitted on, so their coefficients are not identified.",
        reason=RefusalReason.INVALID_DECLARATION,
        alternative="Use fewer folds, or form='pooled'.",
    )


# ----------------------------------------------------------------------- the functional


@dataclass(frozen=True)
class Nuisances:
    """The nuisance values on a set of rows.

    Parameters
    ----------
    g : ndarray of float
        ``P(A = 1 | W)`` on the pooled rows.
    theta : ndarray of float
        ``E(Y | W)``.
    source0 : ndarray of float
        ``Pi(W, 0) = P(S = 1 | W, A = 0)``.
    source1 : ndarray of float
        ``Pi(W, 1) = P(S = 1 | W, A = 1)``; 1 when the external data are controls only.
    qt : ndarray of float
        ``Qt(W, A) = E(Y | W, A)`` at the row's own treatment.
    pibar : ndarray of float
        ``pi(W) = P(S = 1 | W)``.
    q : ndarray of float or None
        ``Q(S, W, A) = E(Y | S, W, A, Delta = 1)`` at the row's own ``(S, A)``, when outcomes
        can be missing.
    g_delta : ndarray of float or None
        ``P(Delta = 1 | S, W, A)``, when outcomes can be missing.
    """

    g: FloatArray
    theta: FloatArray
    source0: FloatArray
    source1: FloatArray
    qt: FloatArray
    pibar: FloatArray
    q: FloatArray | None = None
    g_delta: FloatArray | None = None

    def source_at(self, treatment: FloatArray) -> FloatArray:
        """``Pi(W, A)`` at each row's own treatment.

        Parameters
        ----------
        treatment : ndarray of float
            ``A`` on each row.

        Returns
        -------
        ndarray of float
            ``Pi(W_i, A_i)``.
        """
        return np.where(np.asarray(treatment) == 1.0, self.source1, self.source0)


@dataclass(frozen=True)
class WorkingUnit:
    """One measure's rows, nuisances and selected working-model columns.

    Parameters
    ----------
    outcome : ndarray of float
        ``Y``; any value where it is not observed.
    treatment : ndarray of float
        ``A`` in ``{0, 1}``.
    source : ndarray of float
        ``S`` in ``{0, 1}``.
    observed : ndarray of bool or None
        ``Delta``, or ``None`` when every outcome is observed.
    phi : ndarray of float
        The ``(n, J)`` selected effect columns.
    psi : ndarray of float
        The ``(n, K)`` selected bias columns at the row's own treatment.
    psi0 : ndarray of float
        The bias columns at ``A = 0``.
    psi1 : ndarray of float
        The bias columns at ``A = 1``.
    nuisances : Nuisances
        The nuisance values on the rows.
    weights : ndarray of float
        The measure: the empirical ``1 / n`` of the rows, or a finite law's masses. They
        sum to 1.
    controls_only : bool
        Whether the external data hold controls only.
    """

    outcome: FloatArray
    treatment: FloatArray
    source: FloatArray
    observed: BoolArray | None
    phi: FloatArray
    psi: FloatArray
    psi0: FloatArray
    psi1: FloatArray
    nuisances: Nuisances
    weights: FloatArray
    controls_only: bool


@dataclass(frozen=True)
class InfluenceParts:
    """The components of the influence function of the projection parameter.

    Each array holds one value per row, and :attr:`total` is their sum.

    Parameters
    ----------
    covariate : ndarray of float
        ``tau_1,beta(W) - Psi_w``, or ``S / p (tau_1,beta(W) - Psi_2,w)`` for the trial
        population, with ``tau_1,beta = beta_A'phi - tau#_beta``.
    effect_model : ndarray of float
        ``m_A'U_A``, the effect-model coefficients' term (Lemma 5).
    bias_model : ndarray of float
        ``-m_S'U_S``, the bias-model coefficients' term (Lemma 8).
    source : ndarray of float
        ``-(S - Pi(W, A)) C(W, A)``, times ``pi(W) / p`` for the trial population.
    """

    covariate: FloatArray
    effect_model: FloatArray
    bias_model: FloatArray
    source: FloatArray

    @property
    def total(self) -> FloatArray:
        """The influence function: the sum of the four components."""
        return self.covariate + self.effect_model + self.bias_model + self.source


def pseudo_responses(
    outcome: FloatArray, observed: BoolArray | None, nuisances: Nuisances
) -> tuple[FloatArray, FloatArray]:
    """The responses of the two R-loss regressions.

    Without missing outcomes they are ``Y - theta`` and ``Y - Qt``. With missing outcomes,
    ``Y`` is replaced by its augmented inverse-weighted form
    ``Delta / g_delta (Y - Q) + Q``, whose mean given ``(S, W, A)`` is ``Q``.

    Parameters
    ----------
    outcome : ndarray of float
        ``Y``; any value where it is not observed.
    observed : ndarray of bool or None
        ``Delta``, or ``None``.
    nuisances : Nuisances
        The nuisance values on the rows.

    Returns
    -------
    tuple of ndarray
        The effect-model and bias-model responses.
    """
    y = _augmented(outcome, observed, nuisances)
    return y - nuisances.theta, y - nuisances.qt


def _augmented(outcome: FloatArray, observed: BoolArray | None, nuisances: Nuisances) -> FloatArray:
    y = np.asarray(outcome, dtype=float)
    if observed is None:
        return y
    if nuisances.q is None or nuisances.g_delta is None:
        raise ValueError("missing outcomes need Q(S, W, A) and P(Delta = 1 | S, W, A)")
    seen = np.asarray(observed, dtype=bool)
    correction = np.where(seen, (np.where(seen, y, 0.0) - nuisances.q) / nuisances.g_delta, 0.0)
    return np.asarray(correction + nuisances.q, dtype=float)


def bias_term(
    source0: FloatArray, source1: FloatArray, bias0: FloatArray, bias1: FloatArray
) -> FloatArray:
    """``tau#(W) = Pi(0 | W, 0) tau_S(W, 0) - Pi(0 | W, 1) tau_S(W, 1)`` (Lemma 2).

    With external controls only, ``Pi(0 | W, 1) = 0``, so ``tau_S(W, 1)`` drops out.

    Parameters
    ----------
    source0 : ndarray of float
        ``Pi(1 | W, 0)``.
    source1 : ndarray of float
        ``Pi(1 | W, 1)``.
    bias0 : ndarray of float
        ``tau_S(W, 0)``.
    bias1 : ndarray of float
        ``tau_S(W, 1)``.

    Returns
    -------
    ndarray of float
        ``tau#(W)``.
    """
    return (1.0 - source0) * bias0 - (1.0 - source1) * bias1


def source_covariates(
    g: FloatArray, bias0: FloatArray, bias1: FloatArray, controls_only: bool
) -> tuple[FloatArray, FloatArray]:
    """The arm coefficients of the clever covariate ``C(W, A)`` of the ``Pi`` fluctuation.

    ``C(W, A) = A k_1(W) + (1 - A) k_0(W)`` with ``k_1 = tau_S(W, 1) / g(1 | W)`` and
    ``k_0 = -tau_S(W, 0) / g(0 | W)``, ``g`` the pooled treatment mechanism (Lemma 8). With
    external controls only, ``Pi(W, 1) = 1`` is not fluctuated and ``k_1 = 0``.

    Parameters
    ----------
    g : ndarray of float
        ``g(1 | W)`` on the pooled rows.
    bias0 : ndarray of float
        ``tau_S,beta(W, 0)``.
    bias1 : ndarray of float
        ``tau_S,beta(W, 1)``.
    controls_only : bool
        Whether the external data hold controls only.

    Returns
    -------
    tuple of ndarray
        ``k_0`` and ``k_1``.
    """
    k0 = -bias0 / (1.0 - g)
    k1 = np.zeros_like(k0) if controls_only else bias1 / g
    return k0, k1


def residuals(
    unit: WorkingUnit, beta_a: FloatArray, beta_s: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """The residuals of the two working models in their influence functions.

    ``r_A = Y* - theta - (A - g) beta_A'phi`` and ``r_S = Y* - Qt - (S - Pi) beta_S'psi``,
    with ``Y*`` the augmented outcome of :func:`pseudo_responses`.

    Parameters
    ----------
    unit : WorkingUnit
        The rows.
    beta_a : ndarray of float
        The effect-model coefficients.
    beta_s : ndarray of float
        The bias-model coefficients.

    Returns
    -------
    tuple of ndarray
        ``r_A`` and ``r_S``.
    """
    nu = unit.nuisances
    response_a, response_s = pseudo_responses(unit.outcome, unit.observed, nu)
    multiplier_s = unit.source - nu.source_at(unit.treatment)
    return (
        response_a - (unit.treatment - nu.g) * (unit.phi @ beta_a),
        response_s - multiplier_s * (unit.psi @ beta_s),
    )


def _population_weights(unit: WorkingUnit, population: Population) -> tuple[FloatArray, FloatArray]:
    """The weight of the covariate mean and the factor of the source term, ``(1, 1)`` or
    ``(S / p, pi / p)``."""
    w, s = unit.weights, unit.source
    if population == "pooled":
        return np.ones_like(w), np.ones_like(w)
    if population == "trial":
        share = float(np.sum(w * s))
        return s / share, unit.nuisances.pibar / share
    raise ValueError(f"population must be 'pooled' or 'trial'; got {population!r}")


def target_directions(unit: WorkingUnit, population: Population) -> tuple[FloatArray, FloatArray]:
    """The target directions ``I_A^{-1} m_A`` and ``I_S^{-1} m_S`` under a unit's measure.

    ``I`` and ``m`` are as in :func:`evaluate`: the Hessians ``E[(A - g)^2 phi phi']`` and
    ``E[(S - Pi)^2 psi psi']``, ``m_A = E[phi]`` and
    ``m_S = E[psi(W, 0) Pi(0 | W, 0) - psi(W, 1) Pi(0 | W, 1)]``, with the means taken as
    ``E[S .] / p`` for the trial population. ``m_A' U_A`` is then
    ``(A - g) r_A phi' I_A^{-1} m_A``.

    Parameters
    ----------
    unit : WorkingUnit
        The rows and their measure.
    population : {"pooled", "trial"}
        The covariate population the effect is averaged over.

    Returns
    -------
    tuple of ndarray
        The effect-model and bias-model directions.
    """
    (hessian_a, mean_a), (hessian_s, mean_s) = _information(unit, population)
    return np.linalg.solve(hessian_a, mean_a), np.linalg.solve(hessian_s, mean_s)


def _information(
    unit: WorkingUnit, population: Population
) -> tuple[tuple[FloatArray, FloatArray], tuple[FloatArray, FloatArray]]:
    """The Hessians ``I`` and the means ``m`` of :func:`target_directions`, effect model
    first."""
    nu = unit.nuisances
    w = unit.weights
    weight, _ = _population_weights(unit, population)
    multiplier_a = unit.treatment - nu.g
    multiplier_s = unit.source - nu.source_at(unit.treatment)
    hessian_a = (unit.phi * (w * multiplier_a**2)[:, None]).T @ unit.phi
    hessian_s = (unit.psi * (w * multiplier_s**2)[:, None]).T @ unit.psi
    gap = unit.psi0 * (1.0 - nu.source0)[:, None] - unit.psi1 * (1.0 - nu.source1)[:, None]
    return (hessian_a, (w * weight) @ unit.phi), (hessian_s, (w * weight) @ gap)


def _estimable(basis: FloatArray, multiplier: FloatArray, weights: FloatArray) -> bool:
    """Whether a working model's Hessian ``E[m^2 phi phi']`` is estimable on a unit's rows.

    It is when the rows that carry the model (positive weight, nonzero ``m phi``) number at
    least :data:`VARIANCE_ROWS_PER_COLUMN` per selected column, and the columns are not
    collinear on them by the refit's rank rule (:func:`relaxed_refit`).
    """
    x = multiplier[:, None] * basis
    carried = int(np.sum((weights > 0.0) & np.any(x != 0.0, axis=1)))
    if VARIANCE_ROWS_PER_COLUMN * x.shape[1] > carried:
        return False
    factor = np.linalg.qr(np.sqrt(weights)[:, None] * x, mode="r")
    diagonal = np.abs(np.diag(factor))
    return bool(diagonal.min() > _RANK_TOLERANCE * max(diagonal.max(), 1e-300))


def variance_directions(
    unit: WorkingUnit, population: Population
) -> tuple[FloatArray | None, FloatArray | None]:
    """The directions ``I^{-1} m`` of the CV form's variance under a unit's measure.

    Theorem 2 of arXiv:2405.07186v2 (Appendix C) normalizes the adaptive CV-TMLE by
    ``sigma^2_v = P_0 D_v^2``, the variance of the influence function of fold ``v``'s
    working model at ``P_0``, whose coefficient terms carry ``d_0 = I_0^{-1} m_0``. The fit
    estimates ``d_0`` on every row, with each row's targeted cross-fitted nuisances and the
    fold's selected columns: the largest sample whose nuisances match the ones in ``D_v``'s
    residuals. A direction is ``None`` when its Hessian is not estimable on the rows
    (:data:`VARIANCE_ROWS_PER_COLUMN`, or collinear columns there).

    Parameters
    ----------
    unit : WorkingUnit
        The rows the direction is estimated on, under their measure.
    population : {"pooled", "trial"}
        The covariate population the effect is averaged over.

    Returns
    -------
    tuple of (ndarray or None, ndarray or None)
        The effect-model and bias-model directions.
    """
    nu = unit.nuisances
    (hessian_a, mean_a), (hessian_s, mean_s) = _information(unit, population)
    estimable_a = _estimable(unit.phi, unit.treatment - nu.g, unit.weights)
    estimable_s = _estimable(unit.psi, unit.source - nu.source_at(unit.treatment), unit.weights)
    return (
        np.linalg.solve(hessian_a, mean_a) if estimable_a else None,
        np.linalg.solve(hessian_s, mean_s) if estimable_s else None,
    )


def evaluate(
    unit: WorkingUnit,
    beta_a: FloatArray,
    beta_s: FloatArray,
    population: Population,
    *,
    directions: tuple[FloatArray, FloatArray] | None = None,
) -> tuple[float, InfluenceParts]:
    r"""The projection parameter and its influence function under a unit's measure.

    With ``tau_1,beta(W) = beta_A'phi(W) - tau#_beta(W)``, the estimate is
    ``E[tau_1,beta(W)]`` for ``population="pooled"`` and ``E[S tau_1,beta(W)] / p`` for
    ``"trial"``. The influence function is

    .. math::

        D = c(O) + m_A^\top U_A - m_S^\top U_S - f(W) (S - \Pi(W, A)) C(W, A),

    with ``U_A = I_A^{-1} (A - g) phi r_A`` and ``U_S = I_S^{-1} (S - Pi) psi r_S`` the
    coefficients' influence functions, ``I`` the Hessians ``E[(A - g)^2 phi phi']`` and
    ``E[(S - Pi)^2 psi psi']`` of the two R-losses, and, for the pooled population,
    ``c = tau_1,beta - Psi_w``, ``m_A = E[phi]``,
    ``m_S = E[psi(W, 0) Pi(0 | W, 0) - psi(W, 1) Pi(0 | W, 1)]`` and ``f = 1``. For the
    trial population ``c = S / p (tau_1,beta - Psi_2,w)``, the means are ``E[S .] / p`` and
    ``f = pi(W) / p``.

    Parameters
    ----------
    unit : WorkingUnit
        The rows and their measure.
    beta_a : ndarray of float
        The effect-model coefficients.
    beta_s : ndarray of float
        The bias-model coefficients.
    population : {"pooled", "trial"}
        The covariate population the effect is averaged over.
    directions : tuple of ndarray or None, default=None
        The directions ``I_A^{-1} m_A`` and ``I_S^{-1} m_S``. ``None`` takes them from
        :func:`target_directions` under this unit's measure; ``form="cv"`` passes the
        training rows' directions.

    Returns
    -------
    tuple of (float, InfluenceParts)
        The estimate and the components of its influence function.
    """
    nu = unit.nuisances
    w = unit.weights
    a, s = unit.treatment, unit.source
    observed_source = nu.source_at(a)
    bias0, bias1 = unit.psi0 @ beta_s, unit.psi1 @ beta_s
    effect = unit.phi @ beta_a - bias_term(nu.source0, nu.source1, bias0, bias1)
    residual_a, residual_s = residuals(unit, beta_a, beta_s)
    weight, factor = _population_weights(unit, population)
    direction_a, direction_s = (
        target_directions(unit, population) if directions is None else directions
    )
    k0, k1 = source_covariates(nu.g, bias0, bias1, unit.controls_only)
    clever = np.where(a == 1.0, k1, k0)
    estimate = float(np.sum(w * weight * effect))
    parts = InfluenceParts(
        weight * (effect - estimate),
        (a - nu.g) * residual_a * (unit.phi @ direction_a),
        -(s - observed_source) * residual_s * (unit.psi @ direction_s),
        -factor * (s - observed_source) * clever,
    )
    return estimate, parts


def _bias_rows(unit: WorkingUnit) -> BoolArray:
    """The rows the bias model is fitted on: the controls when the external data are."""
    if unit.controls_only:
        return np.asarray(unit.treatment == 0.0, dtype=bool)
    return np.ones(unit.treatment.shape[0], dtype=bool)


def refit_unit(
    unit: WorkingUnit, *, where: str = "The refit", effect: bool = True
) -> tuple[Refit | None, Refit]:
    """Refit both working models, or the bias model only, under a unit's measure.

    Parameters
    ----------
    unit : WorkingUnit
        The rows, their nuisances and the selected columns.
    where : str, default="The refit"
        The subject of the refusal when the columns are collinear.
    effect : bool, default=True
        Whether to refit the effect model too.

    Returns
    -------
    tuple of (Refit or None, Refit)
        The effect-model refit (``None`` when ``effect`` is false) and the bias-model refit.
    """
    nu = unit.nuisances
    response_a, response_s = pseudo_responses(unit.outcome, unit.observed, nu)
    effect_fit = (
        relaxed_refit(unit.phi, unit.treatment - nu.g, response_a, unit.weights, where=where)
        if effect
        else None
    )
    rows = _bias_rows(unit)
    multiplier_s = unit.source - nu.source_at(unit.treatment)
    bias_fit = relaxed_refit(
        unit.psi[rows], multiplier_s[rows], response_s[rows], unit.weights[rows], where=where
    )
    return effect_fit, bias_fit


def pooled_epsilon(
    units: Sequence[WorkingUnit],
    beta_a: Sequence[FloatArray],
    beta_s: Sequence[FloatArray],
    directions: Sequence[FloatArray],
    model: Literal["effect", "bias"],
) -> float:
    """The one step along the target directions that solves the pooled working-model score.

    For the effect model, ``beta_A,v(e) = beta_A,v + e d_v`` with ``d_v`` fold ``v``'s
    direction, ``H = (A - g) phi' d_v`` and ``r = r_A`` at ``beta_A,v``; for the bias model,
    ``S - Pi``, ``psi`` and ``r_S`` in their place, on the bias model's rows. The score
    ``sum_v P_v H (r - e H) / V`` is linear in ``e``, so
    ``e = sum_v P_v H r / sum_v P_v H^2``.

    Parameters
    ----------
    units : sequence of WorkingUnit
        The validation folds, each under its own measure.
    beta_a : sequence of ndarray of float
        Each fold's effect-model coefficients.
    beta_s : sequence of ndarray of float
        Each fold's bias-model coefficients.
    directions : sequence of ndarray of float
        Each fold's direction for ``model``.
    model : {"effect", "bias"}
        The working model to step.

    Returns
    -------
    float
        The step ``e``.
    """
    numerator = denominator = 0.0
    for unit, ba, bs, direction in zip(units, beta_a, beta_s, directions, strict=True):
        nu = unit.nuisances
        residual_a, residual_s = residuals(unit, ba, bs)
        if model == "effect":
            h = (unit.treatment - nu.g) * (unit.phi @ direction)
            residual, rows = residual_a, np.ones(h.size, dtype=bool)
        else:
            h = (unit.source - nu.source_at(unit.treatment)) * (unit.psi @ direction)
            residual, rows = residual_s, _bias_rows(unit)
        w = unit.weights[rows]
        numerator += float(np.sum(w * h[rows] * residual[rows]))
        denominator += float(np.sum(w * h[rows] ** 2))
    if not denominator > 0.0:
        return 0.0
    return numerator / denominator


def target_source(
    source: FloatArray, covariate: FloatArray, membership: FloatArray, weights: FloatArray
) -> float:
    """Solve the score of the logistic fluctuation of ``Pi`` along a clever covariate.

    The submodel is ``logit Pi_e = logit Pi + e H``, fitted by the weighted log likelihood
    of ``S``. Its score at ``e`` is ``sum_i w_i H_i (S_i - Pi_e,i)``, which decreases in
    ``e``; a Newton step with a halving line search on the convex loss solves it to
    ``1e-13 sum_i w_i |H_i|``.

    Parameters
    ----------
    source : ndarray of float
        ``Pi(W, A)`` at each row's own treatment, strictly inside ``(0, 1)``.
    covariate : ndarray of float
        ``H`` on each row.
    membership : ndarray of float
        ``S`` on each row.
    weights : ndarray of float
        The measure on the rows.

    Returns
    -------
    float
        The fluctuation ``e``.
    """
    offset = logit(np.asarray(source, dtype=float))
    h = np.asarray(covariate, dtype=float)
    s = np.asarray(membership, dtype=float)
    w = np.asarray(weights, dtype=float)
    tolerance = _SOURCE_TOLERANCE * max(float(np.sum(w * np.abs(h))), 1e-300)

    def loss(e: float) -> float:
        eta = offset + e * h
        return float(np.sum(w * (np.logaddexp(0.0, eta) - s * eta)))

    e = 0.0
    for _ in range(200):
        p = expit(offset + e * h)
        score = float(np.sum(w * h * (s - p)))
        if abs(score) <= tolerance:
            return e
        curvature = float(np.sum(w * h**2 * p * (1.0 - p)))
        if not curvature > 0.0:
            break
        step = score / curvature
        current = loss(e)
        while step != 0.0 and loss(e + step) > current + 1e-15 * max(abs(current), 1.0):
            step /= 2.0
            if abs(step) < 1e-300:
                break
        if e + step == e:
            break
        e += step
    p = expit(offset + e * h)
    score = float(np.sum(w * h * (s - p)))
    if abs(score) > tolerance * 1e3:
        _warn(
            f"the fluctuation of P(S = 1 | W, A) stopped with score {score:.3g}, above its "
            f"tolerance {tolerance:.3g}",
            ConvergenceWarning,
        )
    return e


def _fluctuated(unit: WorkingUnit, epsilon: float, k0: FloatArray, k1: FloatArray) -> WorkingUnit:
    nu = unit.nuisances
    source0 = expit(logit(nu.source0) + epsilon * k0)
    source1 = nu.source1 if unit.controls_only else expit(logit(nu.source1) + epsilon * k1)
    return replace(unit, nuisances=replace(nu, source0=source0, source1=source1))


def _source_step(
    unit: WorkingUnit, beta_s: FloatArray, population: Population
) -> tuple[FloatArray, FloatArray, FloatArray, BoolArray]:
    """The fluctuation's arm coefficients, its covariate at the own arm, and its rows."""
    nu = unit.nuisances
    k0, k1 = source_covariates(nu.g, unit.psi0 @ beta_s, unit.psi1 @ beta_s, unit.controls_only)
    if population == "trial":
        factor = nu.pibar / float(np.sum(unit.weights * unit.source))
        k0, k1 = factor * k0, factor * k1
    covariate = np.where(unit.treatment == 1.0, k1, k0)
    return k0, k1, covariate, _bias_rows(unit)


# ----------------------------------------------------------------------- the fit


@dataclass(frozen=True)
class WorkingModel:
    """One selection's working models, as fitted and as targeted.

    Parameters
    ----------
    fold : int or None
        The outer fold whose training rows selected the models, or ``None`` for
        ``form="pooled"``.
    dictionary : WorkingDictionary
        The fitted transformer and the columns the pivoted QR kept.
    effect : Selection
        The lasso selection of the effect model, over ``dictionary.phi``.
    bias : Selection
        The lasso selection of the bias model, over ``dictionary.psi``.
    beta_a : ndarray of float
        The effect-model coefficients of the selected columns: the refit for
        ``form="pooled"``, the targeted refit for ``form="cv"``.
    beta_s : ndarray of float
        The bias-model coefficients of the selected columns: the last refit for
        ``form="pooled"``, the targeted refit for ``form="cv"``.
    directions : tuple of ndarray or None
        For ``form="cv"``, the target directions ``I_A^{-1} m_A`` and ``I_S^{-1} m_S`` of
        the last pass, taken on the training rows; ``None`` for ``form="pooled"``.
    variance_directions : tuple of ndarray or None
        For ``form="cv"``, the directions of the influence function whose variance is
        ``variance``: estimated on every row with the targeted nuisances
        (:func:`variance_directions`), or the training rows' direction where
        ``variance_fallback`` is true; ``None`` for ``form="pooled"``, whose variance uses the
        directions of its one measure.
    variance_fallback : tuple of bool
        For the effect and the bias model, whether ``variance`` used the training rows'
        direction because the model's Hessian was not estimable on every row.
    inner : ndarray of int
        The inner fold of each selection row.
    selection_rows : int
        The number of rows the models were selected on.
    estimate : float
        The estimate under this selection's measure.
    variance : float
        The variance of the influence function under this selection's measure: centred
        for ``form="cv"``, ``P_n D^2`` for ``form="pooled"``.
    """

    fold: int | None
    dictionary: WorkingDictionary
    effect: Selection
    bias: Selection
    beta_a: FloatArray
    beta_s: FloatArray
    directions: tuple[FloatArray, FloatArray] | None
    inner: IntArray
    selection_rows: int
    estimate: float
    variance: float
    variance_directions: tuple[FloatArray, FloatArray] | None
    variance_fallback: tuple[bool, bool]

    @property
    def effect_columns(self) -> tuple[str, ...]:
        """The names of the selected effect-model columns."""
        names = self.dictionary.phi_names
        return tuple(names[j] for j in self.effect.selected)

    @property
    def bias_columns(self) -> tuple[str, ...]:
        """The names of the selected bias-model columns."""
        names = self.dictionary.psi_names
        return tuple(names[j] for j in self.bias.selected)

    @property
    def unsupported_columns(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """The effect and bias columns that the support screen dropped."""
        return self.dictionary.unsupported_phi_names, self.dictionary.unsupported_psi_names


@dataclass(frozen=True)
class ScoreResiduals:
    """The score equations the fit solves, as left at its end.

    Each is the mean of one component of :class:`InfluenceParts`, pooled over the
    selections as ``mean_v P_v``.

    Parameters
    ----------
    effect_model : float
        ``|P_n m_A' U_A|``.
    bias_model : float
        ``|P_n m_S' U_S|``.
    source : float
        ``|P_n f (S - Pi*) C|``.
    influence : float
        ``|P_n D|``.
    threshold : float
        The stopping rule's bound on ``influence``, ``sd(D) / (sqrt(n) log n)``. For
        ``form="cv"`` the four scores and the bound read the targeting's ``D``, with the
        training rows' directions, which differs from the reported influence curve.
    """

    effect_model: float
    bias_model: float
    source: float
    influence: float
    threshold: float


@dataclass(frozen=True)
class AdaptiveTMLEDiagnostics:
    """What the fit records about the data and the nuisance fits.

    Parameters
    ----------
    n : int
        Rows.
    n_trial : int
        Trial rows.
    n_external : int
        External rows.
    n_external_treated : int
        External rows with ``A = 1``.
    controls_only : bool
        Whether the external data hold controls only.
    external_source_min : float
        The smallest cross-fitted ``pi(W) = P(S = 1 | W)`` on an external row, before bounds.
    external_source_quantiles : tuple of float
        Its 5%, 25% and 50% quantiles on the external rows.
    external_source_below_floor : float
        The share of external rows with ``pi(W)`` below :data:`SOURCE_FLOOR`.
    trial_row_treatment_range : tuple of float
        The smallest and largest bounded combined-data ``g(1 | W)``, the treatment
        probability the fit estimates on every row, read on the trial rows.
    """

    n: int
    n_trial: int
    n_external: int
    n_external_treated: int
    controls_only: bool
    external_source_min: float
    external_source_quantiles: tuple[float, float, float]
    external_source_below_floor: float
    trial_row_treatment_range: tuple[float, float]


@dataclass(frozen=True)
class AdaptiveTMLEFit:
    """An adaptive TMLE fit.

    Parameters
    ----------
    estimate : float
        The estimate of the working-model projection of the effect.
    std_error : float
        Its standard error: ``sqrt(P_n D^2 / n)`` for ``form="pooled"``, and
        ``sqrt(mean_v var_v(D_v) / n)`` over the validation folds for ``form="cv"``, with
        ``D_v`` the fold's influence function on its validation rows, its ``I`` and ``m``
        estimated on every row (:func:`variance_directions`).
    influence_curve : ndarray of float
        ``D`` on each row; for ``form="cv"``, the row's validation fold's ``D_v`` of
        ``std_error``.
    population : {"pooled", "trial"}
        The covariate population the effect is averaged over.
    form : {"cv", "pooled"}
        The estimator.
    working_models : tuple of WorkingModel
        One per outer fold for ``form="cv"``, one for ``form="pooled"``.
    epsilon : tuple of float
        The fluctuation of ``Pi`` at each pass.
    epsilon_effect : float or None
        For ``form="cv"``, the step of the effect-model coefficients along their target
        directions; ``None`` for ``form="pooled"``.
    epsilon_bias : tuple of float
        For ``form="cv"``, the step of the bias-model coefficients from their refit along
        their target directions, before the first pass and after each; empty for
        ``form="pooled"``.
    passes : int
        The passes of the alternation of the fluctuation and the bias-model refit.
    converged : bool
        Whether the stopping rule ``|P_n D| <= sd(D) / (sqrt(n) log n)`` was met.
    scores : ScoreResiduals
        The score equations as left at the end.
    diagnostics : AdaptiveTMLEDiagnostics
        The data and nuisance diagnostics.
    folds : Folds
        The outer split.
    """

    estimate: float
    std_error: float
    influence_curve: FloatArray
    population: Population
    form: Form
    force_arm: bool
    working_models: tuple[WorkingModel, ...]
    epsilon: tuple[float, ...]
    epsilon_effect: float | None
    epsilon_bias: tuple[float, ...]
    passes: int
    converged: bool
    scores: ScoreResiduals
    diagnostics: AdaptiveTMLEDiagnostics
    folds: Folds


@dataclass(frozen=True)
class _Data:
    y: FloatArray
    a: FloatArray
    s: FloatArray
    w: FloatArray
    observed: BoolArray | None
    family: Literal["gaussian", "binomial"]
    names: tuple[str, ...] | None = None


@dataclass(frozen=True)
class _Learners:
    outcome: Learner
    treatment: Learner
    source: Learner
    missingness: Learner | None


def _fit(learner: Learner, x: FloatArray, y: FloatArray, task: Task) -> Learner:
    target = y.astype(int) if task == "classification" else y
    return fit_learner(learner, x, target, warn_unweighted=False)


def _predict(model: Learner, x: FloatArray, task: Task) -> FloatArray:
    return np.asarray(predict_mean(model, x, task), dtype=float)


def _nuisances_on(
    data: _Data,
    learners: _Learners,
    train: IntArray,
    rows: IntArray,
    *,
    controls_only: bool,
    bounds: tuple[float, float],
    with_pibar: bool,
) -> tuple[Nuisances, FloatArray]:
    """Fit every nuisance on ``train`` and predict it on ``rows``; also the raw ``pi``."""
    low, high = bounds
    w, a, s = data.w, data.a, data.s
    outcome_task: Task = "classification" if data.family == "binomial" else "regression"
    at = w[rows]

    def with_arm(x: FloatArray, arm: FloatArray | float) -> FloatArray:
        return np.column_stack([x, np.broadcast_to(np.asarray(arm, float), (x.shape[0],))])

    g_model = _fit(learners.treatment, w[train], a[train], "classification")
    g = np.clip(_predict(g_model, at, "classification"), low, high)
    if controls_only:
        controls = train[a[train] == 0.0]
        pi_model = _fit(learners.source, w[controls], s[controls], "classification")
        source0 = np.clip(_predict(pi_model, at, "classification"), low, high)
        source1 = np.ones(rows.size)
    else:
        pi_model = _fit(learners.source, with_arm(w[train], a[train]), s[train], "classification")
        source0 = np.clip(_predict(pi_model, with_arm(at, 0.0), "classification"), low, high)
        source1 = np.clip(_predict(pi_model, with_arm(at, 1.0), "classification"), low, high)
    if with_pibar:
        bar_model = _fit(learners.source, w[train], s[train], "classification")
        raw_pibar = _predict(bar_model, at, "classification")
    else:
        raw_pibar = np.full(rows.size, np.nan)
    pibar = np.clip(raw_pibar, low, high)
    own = a[rows]
    if data.observed is None:
        theta_model = _fit(learners.outcome, w[train], data.y[train], outcome_task)
        qt_model = _fit(learners.outcome, with_arm(w[train], a[train]), data.y[train], outcome_task)
        theta = _predict(theta_model, at, outcome_task)
        qt = _predict(qt_model, with_arm(at, own), outcome_task)
        return Nuisances(g, theta, source0, source1, qt, pibar), raw_pibar
    if learners.missingness is None:
        raise ValueError("missing outcomes need a missingness learner")
    seen = train[data.observed[train]]

    def design(x: FloatArray, member: FloatArray | float, arm: FloatArray | float) -> FloatArray:
        size = x.shape[0]
        return np.column_stack(
            [
                x,
                np.broadcast_to(np.asarray(member, float), (size,)),
                np.broadcast_to(np.asarray(arm, float), (size,)),
            ]
        )

    q_model = _fit(learners.outcome, design(w[seen], s[seen], a[seen]), data.y[seen], outcome_task)
    delta_model = _fit(
        learners.missingness,
        design(w[train], s[train], a[train]),
        data.observed[train].astype(float),
        "classification",
    )
    q = {
        (member, arm): _predict(q_model, design(at, member, arm), outcome_task)
        for member in (0.0, 1.0)
        for arm in (0.0, 1.0)
    }
    qt0 = source0 * q[(1.0, 0.0)] + (1.0 - source0) * q[(0.0, 0.0)]
    qt1 = source1 * q[(1.0, 1.0)] + (1.0 - source1) * q[(0.0, 1.0)]
    theta = g * qt1 + (1.0 - g) * qt0
    qt = np.where(own == 1.0, qt1, qt0)
    q_own = _predict(q_model, design(at, s[rows], own), outcome_task)
    g_delta = np.clip(_predict(delta_model, design(at, s[rows], own), "classification"), low, 1.0)
    return Nuisances(g, theta, source0, source1, qt, pibar, q_own, g_delta), raw_pibar


def _stack(pieces: Sequence[tuple[IntArray, Nuisances]], n: int) -> Nuisances:
    """Assemble nuisance values predicted on disjoint row sets into one row order."""
    values: dict[str, Any] = {}
    for name in vars(pieces[0][1]):
        if getattr(pieces[0][1], name) is None:
            values[name] = None
            continue
        column = np.full(n, np.nan)
        for rows, nuisances in pieces:
            column[rows] = getattr(nuisances, name)
        values[name] = column
    return Nuisances(**values)


def _take(nuisances: Nuisances, rows: IntArray) -> Nuisances:
    values: dict[str, Any] = {
        name: None if value is None else np.asarray(value)[rows]
        for name, value in vars(nuisances).items()
    }
    return Nuisances(**values)


def _check_inputs(
    outcome: Any, treatment: Any, source: Any, covariates: Any, observed: Any
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, BoolArray | None]:
    y = np.asarray(outcome, dtype=float).reshape(-1)
    a = np.asarray(treatment, dtype=float).reshape(-1)
    s = np.asarray(source, dtype=float).reshape(-1)
    w = np.asarray(covariates, dtype=float)
    if w.ndim == 1:
        w = w[:, None]
    n = y.size
    if a.size != n or s.size != n or w.shape[0] != n:
        raise DataError("outcome, treatment, source and covariates must have the same rows")
    if not np.all(np.isin(a, (0.0, 1.0))):
        raise DataError("adaptive TMLE needs a treatment coded 0 or 1")
    if not np.all(np.isin(s, (0.0, 1.0))):
        raise DataError("the source must be 1 on trial rows and 0 on external rows")
    if not np.all(np.isfinite(w)):
        raise DataError("the covariates must be finite")
    trial = s == 1.0
    if not np.any(~trial):
        raise DataError("the pooled data hold no external row")
    if np.unique(a[trial]).size < 2:
        raise DataError("the trial must hold rows in both arms")
    seen: BoolArray | None = None
    if observed is not None:
        seen = np.asarray(observed, dtype=bool).reshape(-1)
        if seen.size != n:
            raise DataError("the observation indicator must have one value per row")
        if not np.all(np.isfinite(y[seen])):
            raise DataError("every observed outcome must be finite")
        for label, rows in (("trial", trial), ("external", ~trial)):
            for arm in (0.0, 1.0):
                group = rows & (a == arm)
                if np.any(group) and not np.any(seen & group):
                    raise DataError(
                        f"every {label} row in arm {arm:g} has a missing outcome, so the "
                        f"outcome regression Q({arm:g}, W) for those rows would be pure "
                        "extrapolation from the other rows. Drop that arm's rows, or "
                        "recover some of their outcomes."
                    )
    elif not np.all(np.isfinite(y)):
        raise DataError("every outcome must be finite when no observation indicator is given")
    return y, a, s, w, seen


def fit_adaptive_tmle(
    outcome: Any,
    treatment: Any,
    source: Any,
    covariates: Any,
    *,
    outcome_learner: Learner,
    treatment_learner: Learner,
    source_learner: Learner,
    g_bounds: tuple[float, float],
    observed: Any = None,
    missingness_learner: Learner | None = None,
    basis: Any = None,
    form: Form = "cv",
    population: Population = "pooled",
    force_arm: bool = False,
    folds: Folds | None = None,
    n_folds: int = 5,
    inner_folds: int = 5,
    random_state: int = 0,
    outcome_family: Literal["gaussian", "binomial"] = "gaussian",
    covariate_names: Sequence[str] | None = None,
) -> AdaptiveTMLEFit:
    """Fit adaptive TMLE of a trial's effect with external data pooled in.

    Parameters
    ----------
    outcome : array-like of float
        ``Y``; any value where ``observed`` is false.
    treatment : array-like of float
        ``A`` in ``{0, 1}``.
    source : array-like of float
        ``S``: 1 on trial rows, 0 on external rows.
    covariates : array-like of float
        ``W``, one row per observation.
    outcome_learner : Learner
        Fits ``theta(W) = E(Y | W)`` and ``Qt(W, A)``, or ``Q(S, W, A)`` with missing
        outcomes.
    treatment_learner : Learner
        Fits ``g(1 | W)`` on the pooled rows.
    source_learner : Learner
        Fits ``Pi(1 | W, A)`` and ``pi(W) = P(S = 1 | W)``.
    g_bounds : tuple of float
        The bounds of ``g``, ``Pi`` and ``pi``, and the lower bound of the observation
        probability.
    observed : array-like of bool or None, default=None
        ``Delta``, true where the outcome is recorded.
    missingness_learner : Learner or None, default=None
        Fits ``P(Delta = 1 | S, W, A)``; required with ``observed``.
    basis : Any, default=None
        A scikit-learn transformer of ``W`` into the dictionary columns, or ``None`` for
        :class:`~cleverly.estimators._working_basis.WorkingBasis`.
    form : {"cv", "pooled"}, default="cv"
        The adaptive CV-TMLE (Appendix C) or Algorithm 1 with cross-fitted nuisances.
    population : {"pooled", "trial"}, default="pooled"
        Average the trial's conditional effect over the pooled covariates or the trial's.
    force_arm : bool, default=False
        Leave the bias model's arm column ``A`` unpenalized, as ``atmle``'s default
        ``force_A = TRUE`` does, and make the arm hierarchy a no-op. ``False`` penalizes
        ``A`` like every other column except the constant. Forcing ``A`` gives each external
        arm a free mean shift, so the external rows carry no information on the effect's
        level and the efficiency bound is the trial-only one. With controls-only external
        data the bias dictionary has no arm column, and the flag has no effect.
    folds : Folds or None, default=None
        The outer split. ``None`` draws a split of ``n_folds`` folds stratified by ``S``,
        and by ``S`` and ``Y`` for a binary outcome (:func:`stratified_partition`).
    n_folds : int, default=5
        The outer folds drawn when ``folds`` is ``None``.
    inner_folds : int, default=5
        The folds of the cross-validation that chooses each lasso penalty, stratified like
        the outer split.
    random_state : int, default=0
        The seed of the outer split, when drawn, of the inner splits, and of every
        ``random_state`` of ``basis`` that is ``None``.
    outcome_family : {"gaussian", "binomial"}, default="gaussian"
        Whether the outcome regressions are regressions or classifiers.
    covariate_names : sequence of str or None, default=None
        The names of the columns of ``covariates``, which name the working-model columns
        (``age``, ``A*age``). ``None`` names them ``x0``, ``x1``, and so on.

    Returns
    -------
    AdaptiveTMLEFit
        The estimate, its standard error, the influence curve and the working models.

    Raises
    ------
    MethodConfigurationError
        When ``form="cv"`` has fewer than three outer folds.
    CapabilityError
        When a refit's selected columns are collinear on the rows it reads.
    """
    y, a, s, w, seen = _check_inputs(outcome, treatment, source, covariates, observed)
    if form not in ("cv", "pooled"):
        raise ValueError(f"form must be 'cv' or 'pooled'; got {form!r}")
    if population not in ("pooled", "trial"):
        raise ValueError(f"population must be 'pooled' or 'trial'; got {population!r}")
    if seen is not None and missingness_learner is None:
        raise ValueError("observed= needs a missingness_learner")
    n = y.size
    if not isinstance(force_arm, (bool, np.bool_)):
        raise ValueError(f"force_arm must be a bool; got {force_arm!r}")
    split = (
        folds
        if folds is not None
        else stratified_partition(s, y, seen, outcome_family, n_folds, random_state)
    )
    if split.n != n:
        raise ValueError(f"the folds cover {split.n} rows; the data hold {n}")
    if form == "cv" and split.n_folds < 3:
        raise MethodConfigurationError(
            "The adaptive CV-TMLE selects each fold's working model with nuisances that read "
            "neither that fold nor the row's own fold, which needs at least three folds. Use "
            "n_folds=3 or more, or form='pooled'."
        )
    names = None if covariate_names is None else tuple(str(name) for name in covariate_names)
    if names is not None and len(names) != w.shape[1]:
        raise ValueError(f"covariate_names has {len(names)} names for {w.shape[1]} columns")
    data = _Data(y, a, s, w, seen, outcome_family, names)
    learners = _Learners(outcome_learner, treatment_learner, source_learner, missingness_learner)
    external = s == 0.0
    n_external_treated = int(np.sum(external & (a == 1.0)))
    controls_only = n_external_treated == 0
    force_arm = bool(force_arm) and not controls_only
    few = max(20, int(0.01 * int(np.sum(external))))
    if 0 < n_external_treated <= few:
        _warn(
            f"the bias of the external treated rows is learned from only {n_external_treated} rows",
            PositivityWarning,
        )
    assignment = split.assignment
    every = np.arange(n, dtype=np.int64)
    out_of_fold: list[tuple[IntArray, Nuisances]] = []
    raw_pibar = np.full(n, np.nan)
    for v in range(split.n_folds):
        held = np.flatnonzero(assignment == v)
        train = every if split.n_folds == 1 else np.flatnonzero(assignment != v)
        values, raw = _nuisances_on(
            data,
            learners,
            train,
            held,
            controls_only=controls_only,
            bounds=g_bounds,
            with_pibar=True,
        )
        out_of_fold.append((held, values))
        raw_pibar[held] = raw
    cross_fitted = _stack(out_of_fold, n)
    seeds = np.random.default_rng(random_state).integers(0, 2**32 - 1, size=split.n_folds + 1)
    selections: list[tuple[int | None, IntArray, Nuisances, IntArray]] = []
    if form == "cv":
        pairs: dict[tuple[int, int], tuple[IntArray, Nuisances]] = {}
        for u in range(split.n_folds):
            for v in range(u + 1, split.n_folds):
                both = np.flatnonzero((assignment == u) | (assignment == v))
                train = np.flatnonzero((assignment != u) & (assignment != v))
                values, _ = _nuisances_on(
                    data,
                    learners,
                    train,
                    both,
                    controls_only=controls_only,
                    bounds=g_bounds,
                    with_pibar=True,
                )
                pairs[(u, v)] = (both, values)
        for v in range(split.n_folds):
            pieces: list[tuple[IntArray, Nuisances]] = []
            for u in range(split.n_folds):
                if u == v:
                    continue
                both, values = pairs[(min(u, v), max(u, v))]
                local = np.flatnonzero(assignment[both] == u)
                pieces.append((both[local], _take(values, local)))
            selection_rows = np.flatnonzero(assignment != v)
            stacked = _take(_stack(pieces, n), selection_rows)
            evaluation_rows = np.flatnonzero(assignment == v)
            selections.append((v, selection_rows, stacked, evaluation_rows))
    else:
        selections.append((None, every, cross_fitted, every))
    units: list[WorkingUnit] = []
    trains: list[WorkingUnit] = []
    models: list[tuple[int | None, WorkingDictionary, Selection, Selection, IntArray, int]] = []
    for index, (fold, rows, values, evaluation) in enumerate(selections):
        inner = stratified_partition(
            data.s[rows],
            data.y[rows],
            None if data.observed is None else data.observed[rows],
            data.family,
            inner_folds,
            int(seeds[index]),
        ).assignment
        dictionary, effect, bias = _select(
            data, rows, values, inner, controls_only, basis, random_state, force_arm
        )
        units.append(
            _unit(
                data,
                evaluation,
                _take(cross_fitted, evaluation),
                dictionary,
                effect,
                bias,
                controls_only,
            )
        )
        if fold is not None:
            trains.append(_unit(data, rows, values, dictionary, effect, bias, controls_only))
        models.append((fold, dictionary, effect, bias, inner, int(rows.size)))
    if form == "cv":
        targeted = _target_cv(units, trains, models, population, data.y.size)
    else:
        targeted = _target_pooled(units, population, data.y.size)
    return _assemble(
        data,
        targeted,
        models,
        population,
        form,
        force_arm,
        split,
        raw_pibar,
        controls_only,
        n_external_treated,
    )


def stratified_partition(
    source: FloatArray,
    outcome: FloatArray,
    observed: BoolArray | None,
    family: Literal["gaussian", "binomial"],
    n_folds: int,
    seed: int,
) -> Folds:
    """Draw a split that holds trial and external rows in proportion in every fold.

    The strata are ``S``, and ``S`` by ``Y`` for a binary outcome, with an unrecorded
    outcome as its own level, as ``atmle`` stratifies its folds. Each stratum is spread over
    the folds to within one row. When a stratum has fewer rows than folds the split
    stratifies by ``S`` alone, and when ``S`` has such a stratum too it is the plain
    :func:`~cleverly.learners.crossfit.random_partition`.

    Parameters
    ----------
    source : ndarray of float
        ``S``.
    outcome : ndarray of float
        ``Y``; any value where ``observed`` is false.
    observed : ndarray of bool or None
        ``Delta``, or ``None`` when every outcome is recorded.
    family : {"gaussian", "binomial"}
        The outcome family; only a binary outcome enters the strata.
    n_folds : int
        The folds requested.
    seed : int
        The seed of the draw.

    Returns
    -------
    Folds
        The split.
    """
    trial = (np.asarray(source) == 1.0).astype(np.int64)
    candidates: list[IntArray] = []
    if family == "binomial":
        level = np.asarray(outcome, dtype=float) == 1.0
        if observed is not None:
            level = np.where(observed, level, 2)
        candidates.append(trial + 2 * level.astype(np.int64))
    candidates.append(trial)
    for strata in candidates:
        if n_folds >= 2 and np.unique(strata, return_counts=True)[1].min() >= n_folds:
            return make_folds(
                strata.size, n_folds, stratify=strata.astype(float), random_state=seed
            )
    return random_partition(strata.size, n_folds, seed=seed)


def _select(
    data: _Data,
    rows: IntArray,
    values: Nuisances,
    inner: IntArray,
    controls_only: bool,
    basis: Any,
    random_state: int,
    force_arm: bool,
) -> tuple[WorkingDictionary, Selection, Selection]:
    """Fit the dictionary and select both working models on the selection rows."""
    a, s = data.a[rows], data.s[rows]
    bias_rows = np.flatnonzero(a == 0.0) if controls_only else np.arange(rows.size)
    dictionary = WorkingDictionary.fit(
        basis,
        data.w[rows],
        a,
        s,
        bias_rows,
        controls_only=controls_only,
        covariate_names=data.names,
        random_state=random_state,
    )
    observed = None if data.observed is None else data.observed[rows]
    response_a, response_s = pseudo_responses(data.y[rows], observed, values)
    phi = dictionary.phi(data.w[rows])
    effect = select_working_model(phi, a - values.g, response_a, 1, inner)
    psi = dictionary.psi(data.w[rows], a)
    multiplier_s = s - values.source_at(a)
    # By default only the constant of the bias model is unpenalized, as the intercept of the
    # paper's lasso (Algorithm 1). The arm column A is penalized like every other column:
    # forced in, it gives the external arms free mean shifts, so they carry no information on
    # the effect's level and the fit's efficiency bound is the trial-only one. force_arm
    # leaves A unpenalized, as atmle's force_A = TRUE does; A is then always selected, and
    # the arm hierarchy has nothing to add.
    if force_arm and tuple(dictionary.psi_keep[:2]) != (0, 1):
        raise CapabilityError(
            "force_arm needs the bias dictionary's constant and arm column, and the "
            "selection rows leave the arm column out as collinear with the constant.",
            reason=RefusalReason.INVALID_DECLARATION,
            alternative="Use force_arm=False.",
        )
    bias = select_working_model(
        psi[bias_rows],
        multiplier_s[bias_rows],
        response_s[bias_rows],
        2 if force_arm else 1,
        inner[bias_rows],
    )
    return dictionary, effect, bias if force_arm else arm_hierarchy(bias, dictionary.psi_arm)


def arm_hierarchy(selection: Selection, arm: BoolArray) -> Selection:
    """Add the arm column ``A`` to a bias selection that holds a column carrying ``A``.

    A selection without ``A`` but with ``A * f(W)`` states that the two external arms' biases
    differ by ``f(W)``, which is zero where ``f`` is. The dictionary centres its columns at the
    selection rows' means and puts its knots at their quantiles, so that zero is a sample
    quantity: the bias the selection then assumes away is of order ``n^{-1/2}`` at every ``n``,
    and the working model's projection misses the effect by that order, which the oracle
    approximation condition (arXiv:2405.07186v2, Theorem 1) forbids. With ``A`` in whenever an
    ``A`` column is, the only arm restriction a selection can make is that the external bias
    is the same in both arms, ``tau_S(W, 1) = tau_S(W, 0)``, which does not depend on how the
    columns are centred.

    Parameters
    ----------
    selection : Selection
        The lasso selection of the bias model.
    arm : ndarray of bool
        Marks the bias columns that carry ``A``; position 1 is ``A`` itself. All false when
        the external data are controls only.

    Returns
    -------
    Selection
        ``selection``, with ``A`` added to ``selected`` when another arm column is selected.
    """
    carries = np.asarray(arm, dtype=bool)
    if not carries.any() or 1 in selection.selected or not carries[selection.selected].any():
        return selection
    return replace(selection, selected=np.union1d(selection.selected, [1]).astype(np.int64))


def _unit(
    data: _Data,
    rows: IntArray,
    values: Nuisances,
    dictionary: WorkingDictionary,
    effect: Selection,
    bias: Selection,
    controls_only: bool,
) -> WorkingUnit:
    w, a = data.w[rows], data.a[rows]
    return WorkingUnit(
        outcome=data.y[rows],
        treatment=a,
        source=data.s[rows],
        observed=None if data.observed is None else data.observed[rows],
        phi=dictionary.phi(w)[:, effect.selected],
        psi=dictionary.psi(w, a)[:, bias.selected],
        psi0=dictionary.psi(w, 0.0)[:, bias.selected],
        psi1=dictionary.psi(w, 1.0)[:, bias.selected],
        nuisances=values,
        weights=np.full(rows.size, 1.0 / rows.size),
        controls_only=controls_only,
    )


@dataclass(frozen=True)
class _Targeted:
    units: list[WorkingUnit]
    beta_a: list[FloatArray]
    beta_s: list[FloatArray]
    directions: list[tuple[FloatArray, FloatArray] | None]
    evaluated: list[tuple[float, InfluenceParts]]
    epsilon: list[float]
    epsilon_effect: float | None
    epsilon_bias: list[float]
    converged: bool


def _pooled_mean(units: Sequence[WorkingUnit], values: Sequence[FloatArray]) -> float:
    """``mean_v P_v x`` over the selections."""
    return sum(float(np.sum(u.weights * x)) for u, x in zip(units, values, strict=True)) / len(
        units
    )


def _source_epsilon(
    units: Sequence[WorkingUnit], beta_s: Sequence[FloatArray], population: Population
) -> tuple[float, list[tuple[FloatArray, FloatArray, FloatArray, BoolArray]]]:
    """The fluctuation of ``Pi`` pooled over the units, and each unit's step."""
    steps = [_source_step(unit, beta, population) for unit, beta in zip(units, beta_s, strict=True)]
    rows = [step[3] for step in steps]
    count = len(units)
    epsilon = target_source(
        np.concatenate(
            [u.nuisances.source_at(u.treatment)[r] for u, r in zip(units, rows, strict=True)]
        ),
        np.concatenate([step[2][r] for step, r in zip(steps, rows, strict=True)]),
        np.concatenate([u.source[r] for u, r in zip(units, rows, strict=True)]),
        np.concatenate([u.weights[r] / count for u, r in zip(units, rows, strict=True)]),
    )
    return epsilon, steps


def _stopped(
    units: Sequence[WorkingUnit], evaluated: Sequence[tuple[float, InfluenceParts]], n: int
) -> tuple[bool, float, float]:
    """The stopping rule ``|P_n D| <= sd(D) / (sqrt(n) log n)``."""
    mean = _pooled_mean(units, [parts.total for _, parts in evaluated])
    spread = float(np.std(np.concatenate([parts.total for _, parts in evaluated])))
    threshold = spread / (np.sqrt(n) * np.log(n))
    return abs(mean) <= threshold, mean, threshold


def _not_converged(mean: float, threshold: float) -> None:
    _warn(
        f"the fluctuation of P(S = 1 | W, A) and the bias-model update did not meet the "
        f"stopping rule in {MAX_PASSES} passes; |P_n D| is {abs(mean):.3g}, above "
        f"{threshold:.3g}",
        ConvergenceWarning,
    )


def _target_pooled(units: list[WorkingUnit], population: Population, n: int) -> _Targeted:
    """Algorithm 1: refit, then alternate the ``Pi`` fluctuation with the bias-model refit."""
    (unit,) = units
    effect_fit, bias_fit = refit_unit(unit)
    assert effect_fit is not None
    beta_a, beta_s = effect_fit.beta, bias_fit.beta
    epsilons: list[float] = []
    converged = False
    for _ in range(MAX_PASSES):
        epsilon, ((k0, k1, *_),) = _source_epsilon([unit], [beta_s], population)
        epsilons.append(epsilon)
        unit = _fluctuated(unit, epsilon, k0, k1)
        beta_s = refit_unit(unit, effect=False)[1].beta
        evaluated = [evaluate(unit, beta_a, beta_s, population)]
        converged, mean, threshold = _stopped([unit], evaluated, n)
        if converged:
            break
    if not converged:
        _not_converged(mean, threshold)
    return _Targeted([unit], [beta_a], [beta_s], [None], evaluated, epsilons, None, [], converged)


def _target_cv(
    units: list[WorkingUnit],
    trains: list[WorkingUnit],
    models: list[tuple[int | None, WorkingDictionary, Selection, Selection, IntArray, int]],
    population: Population,
    n: int,
) -> _Targeted:
    """Appendix C, SR(1) step 4(b): refit on ``T_v``, then step along the target directions.

    The directions ``I^{-1} m`` are taken on each fold's training rows. The effect model's
    step is solved once. The bias model's direction depends on ``Pi``, so each pass
    fluctuates ``Pi`` on the validation and the training rows by the one ``epsilon`` solved
    on the validation rows, recomputes the training direction, and solves the bias step from
    the refit again.
    """
    fits = [
        refit_unit(train, where=f"Fold {model[0]}")
        for train, model in zip(trains, models, strict=True)
    ]
    refit_a = [fit[0].beta for fit in fits if fit[0] is not None]
    refit_s = [fit[1].beta for fit in fits]
    effect_directions = [target_directions(train, population)[0] for train in trains]
    epsilon_effect = pooled_epsilon(units, refit_a, refit_s, effect_directions, "effect")
    beta_a = [b + epsilon_effect * d for b, d in zip(refit_a, effect_directions, strict=True)]

    def bias_step(
        trains: list[WorkingUnit], units: list[WorkingUnit]
    ) -> tuple[float, list[FloatArray], list[FloatArray]]:
        directions = [target_directions(train, population)[1] for train in trains]
        step = pooled_epsilon(units, beta_a, refit_s, directions, "bias")
        return step, [b + step * d for b, d in zip(refit_s, directions, strict=True)], directions

    epsilon_bias, beta_s, bias_directions = bias_step(trains, units)
    epsilons: list[float] = []
    steps_bias = [epsilon_bias]
    converged = False
    for _ in range(MAX_PASSES):
        epsilon, steps = _source_epsilon(units, beta_s, population)
        epsilons.append(epsilon)
        units = [
            _fluctuated(unit, epsilon, step[0], step[1])
            for unit, step in zip(units, steps, strict=True)
        ]
        trains = [
            _fluctuated(train, epsilon, *_source_step(train, beta, population)[:2])
            for train, beta in zip(trains, beta_s, strict=True)
        ]
        epsilon_bias, beta_s, bias_directions = bias_step(trains, units)
        steps_bias.append(epsilon_bias)
        directions = list(zip(effect_directions, bias_directions, strict=True))
        evaluated = [
            evaluate(unit, ba, bs, population, directions=d)
            for unit, ba, bs, d in zip(units, beta_a, beta_s, directions, strict=True)
        ]
        converged, mean, threshold = _stopped(units, evaluated, n)
        if converged:
            break
    if not converged:
        _not_converged(mean, threshold)
    return _Targeted(
        units,
        beta_a,
        beta_s,
        list(directions),
        evaluated,
        epsilons,
        epsilon_effect,
        steps_bias,
        converged,
    )


def _assemble(
    data: _Data,
    targeted: _Targeted,
    models: list[tuple[int | None, WorkingDictionary, Selection, Selection, IntArray, int]],
    population: Population,
    form: Form,
    force_arm: bool,
    split: Folds,
    raw_pibar: FloatArray,
    controls_only: bool,
    n_external_treated: int,
) -> AdaptiveTMLEFit:
    """The estimate, its standard error and the record of the targeting.

    The CV form's variance re-evaluates each fold's influence function on its validation
    rows with the directions of :func:`variance_directions`, estimated on every row with the
    targeted nuisances stacked over the folds; the estimate, the scores and the stopping rule
    keep the targeting's own evaluation.
    """
    n = data.y.size
    units, evaluated = targeted.units, targeted.evaluated
    every = np.arange(n, dtype=np.int64)
    stacked = (
        _stack(
            [
                (rows, u.nuisances)
                for rows, u in zip(_evaluation_rows(split, form), units, strict=True)
            ],
            n,
        )
        if form == "cv"
        else None
    )
    influence = np.full(n, np.nan)
    working: list[WorkingModel] = []
    variances: list[float] = []
    fallbacks: list[int] = []
    for unit, (estimate, parts), ba, bs, d, model, fold_rows in zip(
        units,
        evaluated,
        targeted.beta_a,
        targeted.beta_s,
        targeted.directions,
        models,
        _evaluation_rows(split, form),
        strict=True,
    ):
        fold, dictionary, effect, bias, inner, selection_rows = model
        chosen: tuple[FloatArray, FloatArray] | None = None
        fallback = (False, False)
        if d is None:
            total = parts.total
            variance = float(np.mean(total**2))
        else:
            assert stacked is not None
            measure = _unit(data, every, stacked, dictionary, effect, bias, controls_only)
            direction_a, direction_s = variance_directions(measure, population)
            fallback = (direction_a is None, direction_s is None)
            chosen = (
                d[0] if direction_a is None else direction_a,
                d[1] if direction_s is None else direction_s,
            )
            total = evaluate(unit, ba, bs, population, directions=chosen)[1].total
            variance = float(np.var(total))
            if any(fallback):
                fallbacks.append(int(fold) if fold is not None else 0)
        influence[fold_rows] = total
        variances.append(variance)
        working.append(
            WorkingModel(
                fold,
                dictionary,
                effect,
                bias,
                ba,
                bs,
                d,
                inner,
                selection_rows,
                estimate,
                variance,
                chosen,
                fallback,
            )
        )
    if fallbacks:
        _warn(
            f"the working models of fold(s) {', '.join(str(v) for v in fallbacks)} have fewer "
            f"than {VARIANCE_ROWS_PER_COLUMN} rows per selected column, or collinear columns, "
            "so their variance takes the working model's direction from the fold's training "
            "rows; that standard error is consistent but carries the training direction's "
            "estimation error, so it can be too large",
            UserWarning,
        )
    _, _, threshold = _stopped(units, evaluated, n)
    scores = ScoreResiduals(
        effect_model=abs(_pooled_mean(units, [parts.effect_model for _, parts in evaluated])),
        bias_model=abs(_pooled_mean(units, [parts.bias_model for _, parts in evaluated])),
        source=abs(_pooled_mean(units, [parts.source for _, parts in evaluated])),
        influence=abs(_pooled_mean(units, [parts.total for _, parts in evaluated])),
        threshold=threshold,
    )
    return AdaptiveTMLEFit(
        estimate=float(np.mean([estimate for estimate, _ in evaluated])),
        std_error=float(np.sqrt(np.mean(variances) / n)),
        influence_curve=influence,
        population=population,
        form=form,
        force_arm=force_arm,
        working_models=tuple(working),
        epsilon=tuple(targeted.epsilon),
        epsilon_effect=targeted.epsilon_effect,
        epsilon_bias=tuple(targeted.epsilon_bias),
        passes=len(targeted.epsilon),
        converged=targeted.converged,
        scores=scores,
        diagnostics=_diagnostics(
            data, units, split, form, raw_pibar, controls_only, n_external_treated, population
        ),
        folds=split,
    )


def _evaluation_rows(split: Folds, form: Form) -> list[IntArray]:
    if form == "pooled":
        return [np.arange(split.n, dtype=np.int64)]
    return [np.flatnonzero(split.assignment == v) for v in range(split.n_folds)]


def _diagnostics(
    data: _Data,
    units: list[WorkingUnit],
    split: Folds,
    form: Form,
    raw_pibar: FloatArray,
    controls_only: bool,
    n_external_treated: int,
    population: Population,
) -> AdaptiveTMLEDiagnostics:
    external = data.s == 0.0
    trial = ~external
    values = raw_pibar[external]
    g = np.full(data.y.size, np.nan)
    for unit, rows in zip(units, _evaluation_rows(split, form), strict=True):
        g[rows] = unit.nuisances.g
    below = float(np.mean(values < SOURCE_FLOOR))
    # Assumption A3 (the trial covers the external rows) is needed only when the effect is
    # averaged over every row; the trial population never evaluates Q at an uncovered row.
    if below > 0.0 and population == "pooled":
        _warn(
            f"{below:.1%} of the external rows have an estimated trial-enrollment probability "
            f"P(S = 1 | W) below {SOURCE_FLOOR} (smallest {values.min():.3g}); the effect "
            "averaged over the pooled covariates relies on trial rows like them",
            PositivityWarning,
        )
    q05, q25, q50 = (float(v) for v in np.quantile(values, [0.05, 0.25, 0.5]))
    return AdaptiveTMLEDiagnostics(
        n=int(data.y.size),
        n_trial=int(np.sum(trial)),
        n_external=int(np.sum(external)),
        n_external_treated=n_external_treated,
        controls_only=controls_only,
        external_source_min=float(values.min()),
        external_source_quantiles=(q05, q25, q50),
        external_source_below_floor=below,
        trial_row_treatment_range=(float(np.min(g[trial])), float(np.max(g[trial]))),
    )
