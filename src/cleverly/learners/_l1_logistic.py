"""Weighted L1-penalized logistic regression on the ``glmnet`` objective.

The ``strategy="lasso"`` candidates of :class:`~cleverly.estimators.CTMLE` are the
propensity models on an L1 penalty path. This module fits that path. It states the
objective of R ``glmnet`` for ``family="binomial"`` and ``family="multinomial"`` with
``type.multinomial="ungrouped"``, and solves it to a stated accuracy.

The objective at penalty ``lambda`` is

.. math::

    \\frac{1}{\\sum_i w_i} \\sum_i w_i\\, \\ell(a_i, b_0 + \\tilde x_i^\\top b)
    + \\lambda \\lVert b \\rVert_1 ,

where :math:`\\ell` is the binomial or multinomial negative log likelihood and
:math:`\\tilde x` holds the covariates standardized with the weighted mean and the weighted
population standard deviation of the rows being fitted. The intercept is not penalized. A
column that is constant on those rows takes no coefficient. Under ``K > 2`` classes the
parameterization is symmetric: one intercept and one coefficient vector for each class, with
the penalty on every coefficient. A common shift of the coefficients of one covariate across
the classes leaves the predictions unchanged, so only predictions are compared.

The solver is coordinate descent on a quadratic approximation, warm-started along the grid,
followed by a Newton polish on the active set. A solution is accepted when its
Karush-Kuhn-Tucker residual (:func:`kkt_residual`) is at most ``1e-10``. At penalty 0 the
binomial solution is the unpenalized maximum likelihood fit, and the multinomial solution
is the reference-coded unpenalized maximum likelihood fit, which has the same predictions
as any symmetric solution. When the training rows separate the classes that fit has no
maximum, and the solver raises :class:`L1ConvergenceError`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin

from .._typing import FloatArray, IntArray
from ..exceptions import CapabilityError

__all__ = [
    "KKT_TOLERANCE",
    "L1ConvergenceError",
    "L1Fit",
    "L1Logistic",
    "default_penalty_grid",
    "kkt_residual",
    "penalty_path",
]

#: The largest accepted Karush-Kuhn-Tucker residual of a solution, on the objective's scale.
KKT_TOLERANCE = 1e-10

#: The number of log-spaced values of the default grid, as in ``glmnet``'s ``nlambda``.
_N_LAMBDA = 100

#: Bounds on the curvature ``p (1 - p)`` of the quadratic approximation.
_CURVATURE_FLOOR = 1e-5

#: The largest standardized coefficient or intercept of an unpenalized fit. A larger one
#: means the training rows separate the classes, where the fit has no maximum: the loss
#: keeps falling while the coefficients grow, so a gradient test alone would accept a
#: point on the way to infinity.
_SEPARATION_BOUND = 1e3

_MAX_OUTER = 500
_MAX_SWEEPS = 10_000
_MAX_POLISH = 100
_MAX_ROUNDS = 50
#: A relative objective rise that rounding alone explains: 64 ulps of 1.
_ROUNDING = 64 * float(np.finfo(float).eps)


class L1ConvergenceError(CapabilityError):
    """The penalized fit at one penalty did not reach the stated accuracy."""


@dataclass(frozen=True)
class L1Fit:
    """One solution on the penalty path.

    Parameters
    ----------
    penalty : float
        The penalty ``lambda``.
    intercept : ndarray of float
        One intercept for each class. Binary fits hold ``[0, b0]``.
    coef : ndarray of float
        The ``(p, K)`` coefficients on the standardized columns. Binary fits hold the
        class-1 column and a zero class-0 column.
    center : ndarray of float
        The weighted column means of the training rows.
    scale : ndarray of float
        The weighted population standard deviations, with 0 for a constant column.
    constant : ndarray of float or None
        The class proportions when the solution is the intercept-only fit, else ``None``.
    kkt : float
        The Karush-Kuhn-Tucker residual of the solution.
    """

    penalty: float
    intercept: FloatArray
    coef: FloatArray
    center: FloatArray
    scale: FloatArray
    constant: FloatArray | None
    kkt: float

    @property
    def active(self) -> tuple[int, ...]:
        """The columns with a nonzero coefficient in some class."""
        if self.constant is not None:
            return ()
        return tuple(int(j) for j in np.flatnonzero(np.any(self.coef != 0.0, axis=1)))

    def linear_predictor(self, x: FloatArray) -> FloatArray:
        """The ``(m, K)`` linear predictor at the rows of ``x``."""
        matrix = np.asarray(x, dtype=float)
        used = self.scale > 0.0
        z = (matrix[:, used] - self.center[used]) / self.scale[used]
        return np.asarray(self.intercept + z @ self.coef[used], dtype=float)

    def predict_proba(self, x: FloatArray) -> FloatArray:
        """The ``(m, K)`` class probabilities at the rows of ``x``."""
        matrix = np.asarray(x, dtype=float)
        if self.constant is not None:
            return np.tile(self.constant, (matrix.shape[0], 1))
        eta = self.linear_predictor(matrix)
        if eta.shape[1] == 2:
            positive = expit(eta[:, 1] - eta[:, 0])
            return np.column_stack([1.0 - positive, positive])
        return _probabilities(eta)


def _normalized(weights: FloatArray) -> FloatArray:
    w = np.asarray(weights, dtype=float)
    total = float(np.sum(w))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("the L1-penalized propensity needs positive total weight")
    return w * (w.size / total)


def _standardization(x: FloatArray, w: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Weighted means and population standard deviations; 0 marks a constant column."""
    total = float(np.sum(w))
    center = (w @ x) / total
    scale = np.sqrt((w @ (x - center) ** 2) / total)
    informative = w > 0.0
    constant = (
        np.ptp(x[informative], axis=0) == 0.0
        if informative.any()
        else np.ones(x.shape[1], dtype=bool)
    )
    scale = np.where(constant | ~(scale > 0.0), 0.0, scale)
    return center, scale


def _proportions(y: IntArray, weights: FloatArray, n_classes: int) -> FloatArray:
    """Weighted class proportions, by the arithmetic of the intercept-only candidate."""
    total = float(np.sum(weights))
    return np.array(
        [np.sum(weights * (y == k)) / total for k in range(n_classes)],
        dtype=float,
    )


def _one_hot(y: IntArray, n_classes: int) -> FloatArray:
    out = np.zeros((y.size, n_classes))
    out[np.arange(y.size), y] = 1.0
    return out


def _logsumexp(eta: FloatArray) -> FloatArray:
    top = np.max(eta, axis=1)
    return np.asarray(top + np.log(np.sum(np.exp(eta - top[:, None]), axis=1)), dtype=float)


def _probabilities(eta: FloatArray) -> FloatArray:
    return np.asarray(np.exp(eta - _logsumexp(eta)[:, None]), dtype=float)


def _smooth_gradient(
    z: FloatArray, y: FloatArray, w: FloatArray, intercept: FloatArray, coef: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """The gradient of the unpenalized loss in the intercepts and the coefficients."""
    n = z.shape[0]
    eta = intercept + z @ coef
    if coef.shape[1] == 2:
        p1 = expit(eta[:, 1] - eta[:, 0])
        residual = np.column_stack([y[:, 0] - (1.0 - p1), p1 - y[:, 1]])
        residual[:, 0] = 0.0
    else:
        residual = _probabilities(eta) - y
    weighted = w[:, None] * residual / n
    return weighted.sum(axis=0), z.T @ weighted


def kkt_residual(
    x: FloatArray,
    y: Sequence[Any] | FloatArray,
    weights: FloatArray | None,
    fit: L1Fit,
    classes: Sequence[Any] | None = None,
) -> float:
    """The Karush-Kuhn-Tucker residual of a solution, on the stated objective.

    The residual is the largest of three quantities: the absolute intercept gradient; for
    a nonzero coefficient, ``|gradient + lambda sign(b)|``; for a zero coefficient,
    ``max(0, |gradient| - lambda)``. The objective is restated on ``x``, ``y`` and
    ``weights`` with their own standardization, so a coefficient vector from another
    solver can be checked against it.

    Parameters
    ----------
    x : ndarray of float
        The ``(n, p)`` covariates on their original scale.
    y : array-like
        The class of each row.
    weights : ndarray of float or None
        The row weights. ``None`` gives each row weight 1.
    fit : L1Fit
        The solution to check. Its coefficients are on the standardized scale of ``x``.
    classes : sequence or None
        The class order of ``fit``. ``None`` uses the sorted classes of ``y``.

    Returns
    -------
    float
        The residual.
    """
    matrix = np.asarray(x, dtype=float)
    labels = np.asarray(y)
    order = np.unique(labels) if classes is None else np.asarray(classes)
    codes = np.searchsorted(order, labels)
    w = _normalized(np.ones(labels.size) if weights is None else np.asarray(weights, float))
    center, scale = _standardization(matrix, w)
    used = scale > 0.0
    z = (matrix[:, used] - center[used]) / scale[used]
    coef = fit.coef[used]
    intercept = fit.intercept
    if fit.constant is not None:
        coef = np.zeros_like(coef)
        intercept = np.log(np.clip(fit.constant, 1e-300, None))
    return _kkt(z, _one_hot(codes, order.size), w, intercept, coef, fit.penalty)


def _kkt(
    z: FloatArray,
    y: FloatArray,
    w: FloatArray,
    intercept: FloatArray,
    coef: FloatArray,
    penalty: float,
) -> float:
    grad0, grad = _smooth_gradient(z, y, w, intercept, coef)
    binary = coef.shape[1] == 2
    if binary:
        grad0 = grad0[1:]
        grad = grad[:, 1:]
        coef = coef[:, 1:]
    values = [float(np.max(np.abs(grad0))) if grad0.size else 0.0]
    nonzero = coef != 0.0
    if np.any(nonzero):
        values.append(float(np.max(np.abs(grad[nonzero] + penalty * np.sign(coef[nonzero])))))
    if np.any(~nonzero):
        values.append(float(np.max(np.maximum(np.abs(grad[~nonzero]) - penalty, 0.0))))
    return max(values)


def _null_gradient(z: FloatArray, w: FloatArray, y: FloatArray) -> float:
    """The largest coefficient gradient at the intercept-only fit: ``lambda_max``."""
    indicators = y[:, 1:] if y.shape[1] == 2 else y
    centered = indicators - (w @ indicators) / float(np.sum(w))
    return float(np.max(np.abs(z.T @ (w[:, None] * centered)))) / float(np.sum(w))


def default_penalty_grid(
    x: FloatArray, y: Sequence[Any] | FloatArray, weights: FloatArray | None = None
) -> FloatArray:
    """``glmnet``'s default penalty grid without early stopping, with 0 appended.

    ``lambda_max`` is ``max_j |sum_i w_i x~_ij (a_i - abar_w)| / sum_i w_i``, the smallest
    penalty at which every coefficient is zero. With ``K > 2`` classes it is the largest
    such value over the columns and the class indicators. The grid holds 100 log-spaced
    values from ``lambda_max`` down to ``lambda_max * ratio``, with ``ratio = 1e-4`` when
    ``n >= p`` and ``1e-2`` otherwise, then the penalty 0.

    Parameters
    ----------
    x : ndarray of float
        The ``(n, p)`` covariates.
    y : array-like
        The class of each row.
    weights : ndarray of float or None
        The row weights. ``None`` gives each row weight 1.

    Returns
    -------
    ndarray of float
        The 101 penalties, strictly decreasing.
    """
    matrix = np.asarray(x, dtype=float)
    labels = np.asarray(y)
    order = np.unique(labels)
    codes = np.searchsorted(order, labels)
    w = _normalized(np.ones(labels.size) if weights is None else np.asarray(weights, float))
    center, scale = _standardization(matrix, w)
    used = scale > 0.0
    if not np.any(used):
        raise ValueError("the L1-penalized propensity needs at least one non-constant covariate")
    z = (matrix[:, used] - center[used]) / scale[used]
    top = _null_gradient(z, w, _one_hot(codes, order.size))
    n, p = matrix.shape
    ratio = 1e-4 if n >= p else 1e-2
    grid = top * np.exp(np.linspace(0.0, np.log(ratio), _N_LAMBDA))
    grid[0] = top
    return np.append(grid, 0.0)


def check_penalty_grid(lambdas: Sequence[float] | FloatArray) -> FloatArray:
    """A user grid as an array, or ``ValueError`` when it breaks the grid rule."""
    try:
        values = np.asarray(lambdas, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        values = np.array([np.nan])
    valid = (
        values.size > 0
        and bool(np.all(np.isfinite(values)))
        and bool(np.all(values >= 0.0))
        and bool(np.all(np.diff(values) < 0.0))
        and bool(np.all(values[:-1] > 0.0))
    )
    if not valid:
        raise ValueError(
            "lambdas must be a non-empty sequence of finite, non-negative penalties in "
            "strictly decreasing order, with 0 allowed only as the last value; got "
            f"{list(lambdas) if not isinstance(lambdas, np.ndarray) else lambdas.tolist()!r}"
        )
    return values


# ----------------------------------------------------------------------- solvers


class _Problem:
    """The standardized training problem of one penalty path."""

    def __init__(self, x: FloatArray, codes: IntArray, weights: FloatArray, n_classes: int):
        self.raw_weights = np.asarray(weights, dtype=float)
        self.w = _normalized(self.raw_weights)
        self.center, self.scale = _standardization(x, self.w)
        self.used = self.scale > 0.0
        self.z = (x[:, self.used] - self.center[self.used]) / self.scale[self.used]
        self.n, self.p = self.z.shape
        self.codes = codes
        self.n_classes = n_classes
        self.y = _one_hot(codes, n_classes)
        self.binary = n_classes == 2
        self.proportions = _proportions(codes, self.raw_weights, n_classes)

    # The binary problem works on (b0, b) for class 1; the multinomial one on (K,), (p, K).

    def objective(self, intercept: FloatArray, coef: FloatArray, penalty: float) -> float:
        eta = intercept + self.z @ coef
        if self.binary:
            margin = eta[:, 1] - eta[:, 0]
            loss = np.logaddexp(0.0, margin) - self.y[:, 1] * margin
            return float(self.w @ loss) / self.n + penalty * float(np.sum(np.abs(coef[:, 1])))
        loss = _logsumexp(eta) - np.sum(self.y * eta, axis=1)
        return float(self.w @ loss) / self.n + penalty * float(np.sum(np.abs(coef)))

    def kkt(self, intercept: FloatArray, coef: FloatArray, penalty: float) -> float:
        return _kkt(self.z, self.y, self.w, intercept, coef, penalty)

    def null_gradient(self) -> float:
        """The largest coefficient gradient at the intercept-only fit."""
        return _null_gradient(self.z, self.w, self.y)

    def null_start(self) -> tuple[FloatArray, FloatArray]:
        p = np.clip(self.proportions, 1e-12, None)
        intercept = np.log(p) - np.mean(np.log(p))
        if self.binary:
            intercept = np.array([0.0, np.log(p[1] / p[0])])
        return intercept, np.zeros((self.p, self.n_classes))


def _soft(value: float, threshold: float) -> float:
    if value > threshold:
        return value - threshold
    if value < -threshold:
        return value + threshold
    return 0.0


def _cd_class(
    problem: _Problem,
    eta: FloatArray,
    intercept: FloatArray,
    coef: FloatArray,
    k: int,
    penalty: float,
) -> None:
    """Coordinate descent on class ``k``'s quadratic approximation, in place."""
    z, w, n = problem.z, problem.w, problem.n
    if problem.binary:
        margin = eta[:, 1] - eta[:, 0]
        p = expit(margin)
        target = problem.y[:, 1]
    else:
        probabilities = _probabilities(eta)
        p = probabilities[:, k]
        target = problem.y[:, k]
    curvature = np.clip(p * (1.0 - p), _CURVATURE_FLOOR, None)
    v = w * curvature / n
    residual = (target - p) / curvature
    column_scale = (v[:, None] * z**2).sum(axis=0)
    for _ in range(_MAX_SWEEPS):
        largest = 0.0
        shift = float(v @ residual) / float(np.sum(v))
        intercept[k] += shift
        residual -= shift
        largest = max(largest, abs(shift) * np.sqrt(float(np.sum(v))))
        for j in range(problem.p):
            if column_scale[j] <= 0.0:
                continue
            old = coef[j, k]
            update = float(z[:, j] @ (v * residual)) + column_scale[j] * old
            new = _soft(update, penalty) / column_scale[j]
            if new != old:
                residual -= z[:, j] * (new - old)
                coef[j, k] = new
                largest = max(largest, abs(new - old) * np.sqrt(column_scale[j]))
        if largest < 1e-9:
            break


def _descend(
    problem: _Problem, intercept: FloatArray, coef: FloatArray, penalty: float
) -> tuple[FloatArray, FloatArray]:
    """Proximal Newton steps with a backtracking line search, from a warm start."""
    current = problem.objective(intercept, coef, penalty)
    classes = [1] if problem.binary else list(range(problem.n_classes))
    for _ in range(_MAX_OUTER):
        trial_intercept = intercept.copy()
        trial_coef = coef.copy()
        for k in classes:
            eta = trial_intercept + problem.z @ trial_coef
            _cd_class(problem, eta, trial_intercept, trial_coef, k, penalty)
        step = 1.0
        accepted = False
        while step > 1e-12:
            new_intercept = intercept + step * (trial_intercept - intercept)
            new_coef = coef + step * (trial_coef - coef)
            value = problem.objective(new_intercept, new_coef, penalty)
            if value <= current + 1e-16 * max(1.0, abs(current)):
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
        change = current - value
        intercept, coef, current = new_intercept, new_coef, value
        if change <= 1e-12 * max(1.0, abs(current)):
            break
    return intercept, coef


def _hessian_parts(
    problem: _Problem, intercept: FloatArray, coef: FloatArray, active: list[tuple[int, int]]
) -> tuple[FloatArray, FloatArray]:
    """The smooth gradient and Hessian in the free parameters (intercepts and ``active``)."""
    z, w, n = problem.z, problem.w, problem.n
    eta = intercept + z @ coef
    if problem.binary:
        p = expit(eta[:, 1] - eta[:, 0])
        design = np.column_stack([np.ones(n), *(z[:, j] for j, _ in active)])
        gradient = design.T @ (w * (p - problem.y[:, 1])) / n
        hessian = (design * (w * p * (1.0 - p))[:, None]).T @ design / n
        return gradient, hessian
    probabilities = _probabilities(eta)
    owner = np.array(list(range(problem.n_classes)) + [k for _, k in active], dtype=np.int64)
    design = np.column_stack([np.ones((n, problem.n_classes)), *(z[:, j] for j, _ in active)])
    residual = probabilities - problem.y
    gradient = np.einsum("ia,ia->a", design, w[:, None] * residual[:, owner]) / n
    hessian = np.empty((owner.size, owner.size))
    for k in range(problem.n_classes):
        rows_k = owner == k
        for m in range(k, problem.n_classes):
            cols_m = owner == m
            curvature = probabilities[:, k] * ((k == m) - probabilities[:, m])
            block = (design[:, rows_k] * (w * curvature)[:, None]).T @ design[:, cols_m] / n
            hessian[np.ix_(rows_k, cols_m)] = block
            hessian[np.ix_(cols_m, rows_k)] = block.T
    return gradient, hessian


def _polish(
    problem: _Problem, intercept: FloatArray, coef: FloatArray, penalty: float
) -> tuple[FloatArray, FloatArray]:
    """Newton on the active set with signs held fixed; a crossing coordinate leaves it."""
    n_intercepts = 1 if problem.binary else problem.n_classes
    for _ in range(_MAX_POLISH):
        classes = [1] if problem.binary else range(problem.n_classes)
        active = [(j, k) for j in range(problem.p) for k in classes if coef[j, k] != 0.0]
        signs = np.array([np.sign(coef[j, k]) for j, k in active])
        gradient, hessian = _hessian_parts(problem, intercept, coef, active)
        gradient[n_intercepts:] += penalty * signs
        if float(np.max(np.abs(gradient))) < 1e-15:
            break
        step = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        values = np.array([coef[j, k] for j, k in active])
        proposed = values + step[n_intercepts:]
        fraction = 1.0
        leaving = -1
        if active:
            crossing = np.flatnonzero(np.sign(proposed) != signs)
            if crossing.size:
                ratios = values[crossing] / (values[crossing] - proposed[crossing])
                leaving = int(crossing[int(np.argmin(ratios))])
                fraction = float(np.min(ratios))
        intercept = intercept.copy()
        if problem.binary:
            intercept[1] += fraction * step[0]
        else:
            intercept += fraction * step[:n_intercepts]
        coef = coef.copy()
        for index, (j, k) in enumerate(active):
            coef[j, k] = values[index] + fraction * step[n_intercepts + index]
        if leaving >= 0:
            j, k = active[leaving]
            coef[j, k] = 0.0
    return intercept, coef


def _bounded(intercept: FloatArray, coef: FloatArray) -> bool:
    largest = max(float(np.max(np.abs(intercept))), float(np.max(np.abs(coef), initial=0.0)))
    return bool(np.isfinite(largest) and largest <= _SEPARATION_BOUND)


def _moved(
    problem: _Problem,
    intercept: FloatArray,
    coef: FloatArray,
    active: list[tuple[int, int]],
    step: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """The unpenalized fit moved by ``step`` in the free intercepts and ``active``."""
    n_free = 1 if problem.binary else problem.n_classes - 1
    trial_intercept = intercept.copy()
    if problem.binary:
        trial_intercept[1] += step[0]
    else:
        trial_intercept[1:] += step[:n_free]
    trial_coef = coef.copy()
    for position, (j, k) in enumerate(active):
        trial_coef[j, k] += step[n_free + position]
    return trial_intercept, trial_coef


def _mle(problem: _Problem) -> tuple[FloatArray, FloatArray, bool]:
    """The unpenalized fit by Newton: binomial, or reference-coded multinomial."""
    intercept, coef = problem.null_start()
    if not problem.binary:
        intercept = intercept - intercept[0]
    free_classes = [1] if problem.binary else list(range(1, problem.n_classes))
    current = problem.objective(intercept, coef, 0.0)
    for _ in range(_MAX_POLISH):
        active = [(j, k) for j in range(problem.p) for k in free_classes]
        gradient, hessian = _hessian_parts(problem, intercept, coef, active)
        if not problem.binary:
            keep = [k for k in range(problem.n_classes) if k != 0]
            index = keep + list(range(problem.n_classes, gradient.size))
            gradient = gradient[index]
            hessian = hessian[np.ix_(index, index)]
        if float(np.max(np.abs(gradient))) < 1e-15:
            return intercept, coef, _bounded(intercept, coef)
        step = -np.asarray(np.linalg.lstsq(hessian, gradient, rcond=None)[0], dtype=np.float64)
        scale = max(1.0, abs(current))
        trial_intercept, trial_coef = _moved(problem, intercept, coef, active, step)
        value = problem.objective(trial_intercept, trial_coef, 0.0)
        # Near the optimum the objective is flat to rounding: the full Newton step can raise
        # it by a few ulps while it still cuts the gradient. The full step is kept there when
        # it lowers the KKT residual, the quantity the convergence test reads. A shrunken
        # step would only tie the objective and leave the residual where it is.
        flat = value <= current + _ROUNDING * scale and problem.kkt(
            trial_intercept, trial_coef, 0.0
        ) < problem.kkt(intercept, coef, 0.0)
        size = 1.0
        while not flat and size > 1e-12:
            trial_intercept, trial_coef = _moved(problem, intercept, coef, active, size * step)
            value = problem.objective(trial_intercept, trial_coef, 0.0)
            if value <= current + 1e-16 * scale:
                break
            size *= 0.5
        else:
            if not flat:
                break
        intercept, coef = trial_intercept, trial_coef
        if not np.all(np.isfinite(coef)) or float(np.max(np.abs(coef), initial=0.0)) > 1e8:
            return intercept, coef, False
        current = value
    converged = problem.kkt(intercept, coef, 0.0) <= KKT_TOLERANCE
    return intercept, coef, converged and _bounded(intercept, coef)


def _solve(
    problem: _Problem, intercept: FloatArray, coef: FloatArray, penalty: float
) -> tuple[FloatArray, FloatArray, float]:
    """Descent, then polish, until the residual meets the tolerance."""
    residual = np.inf
    for _ in range(_MAX_ROUNDS):
        intercept, coef = _descend(problem, intercept, coef, penalty)
        intercept, coef = _polish(problem, intercept, coef, penalty)
        residual = problem.kkt(intercept, coef, penalty)
        if residual <= KKT_TOLERANCE:
            break
    return intercept, coef, residual


def _expand(problem: _Problem, values: FloatArray) -> FloatArray:
    full = np.zeros((problem.used.size, problem.n_classes))
    full[problem.used] = values
    return full


def penalty_path(
    x: FloatArray,
    y: Sequence[Any] | FloatArray,
    weights: FloatArray | None,
    lambdas: Sequence[float] | FloatArray,
    classes: Sequence[Any] | None = None,
    *,
    layer: str = "the fit",
) -> list[L1Fit]:
    """Solve the objective at each penalty of a grid, warm-started along it.

    Parameters
    ----------
    x : ndarray of float
        The ``(n, p)`` training covariates on their original scale.
    y : array-like
        The class of each training row.
    weights : ndarray of float or None
        The row weights. ``None`` gives each row weight 1.
    lambdas : sequence of float
        The penalties, strictly decreasing, with 0 allowed only as the last value.
    classes : sequence or None
        The class order. ``None`` uses the sorted classes of ``y``. A class with no
        training row keeps probability 0 only through the intercept-only fit, so every
        class must appear.
    layer : str
        The layer named in a non-convergence error, for example ``"selection fold 2"``.

    Returns
    -------
    list of L1Fit
        One solution for each penalty, in grid order.

    Raises
    ------
    L1ConvergenceError
        When a penalty's solution does not reach the residual :data:`KKT_TOLERANCE`.
    """
    grid = check_penalty_grid(lambdas)
    matrix = np.asarray(x, dtype=float)
    labels = np.asarray(y)
    order = np.unique(labels) if classes is None else np.asarray(classes)
    codes = np.searchsorted(order, labels).astype(np.int64)
    if order.size < 2 or not np.array_equal(order[codes], labels.astype(order.dtype)):
        raise ValueError("every training label must be one of at least two classes")
    w = np.ones(labels.size) if weights is None else np.asarray(weights, dtype=float)
    problem = _Problem(matrix, codes, w, order.size)
    intercept, coef = problem.null_start()
    null_gradient = problem.null_gradient() if problem.p else 0.0
    fits: list[L1Fit] = []
    for penalty in grid:
        penalty = float(penalty)
        if problem.p == 0 or (penalty > 0.0 and null_gradient <= penalty):
            fits.append(
                L1Fit(
                    penalty=penalty,
                    intercept=np.zeros(order.size),
                    coef=np.zeros((problem.used.size, order.size)),
                    center=problem.center,
                    scale=problem.scale,
                    constant=problem.proportions.copy(),
                    kkt=problem.kkt(*problem.null_start(), penalty),
                )
            )
            continue
        if penalty == 0.0:
            intercept, coef, converged = _mle(problem)
            residual = problem.kkt(intercept, coef, 0.0)
            if not converged or residual > KKT_TOLERANCE:
                residual = np.inf
        else:
            intercept, coef, residual = _solve(problem, intercept, coef, penalty)
        if not residual <= KKT_TOLERANCE:
            shown = "0" if penalty == 0.0 else f"{penalty:.6g}"
            raise L1ConvergenceError(
                f"the L1-penalized propensity at penalty {shown} did not converge in "
                f"{layer}. The unpenalized end of the path has no maximum when the "
                "training rows separate the arms. Pass lambdas= that ends above 0."
            )
        fits.append(
            L1Fit(
                penalty=penalty,
                intercept=intercept.copy(),
                coef=_expand(problem, coef),
                center=problem.center,
                scale=problem.scale,
                constant=None,
                kkt=float(residual),
            )
        )
    return fits


class L1Logistic(ClassifierMixin, BaseEstimator):
    """A weighted L1-penalized logistic classifier at one penalty.

    It states the objective of the module docstring and solves it with
    :func:`penalty_path` on a one-value grid, so a fit at a penalty and the matching
    entry of a path agree up to the solver tolerance.

    Parameters
    ----------
    penalty : float
        The penalty ``lambda``, at least 0.
    """

    def __init__(self, penalty: float = 0.0) -> None:
        self.penalty = penalty

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> L1Logistic:
        """Fit at :attr:`penalty`.

        Parameters
        ----------
        X : array-like
            The ``(n, p)`` covariates.
        y : array-like
            The class of each row.
        sample_weight : array-like or None
            The row weights.

        Returns
        -------
        L1Logistic
            The fitted classifier.
        """
        labels = np.asarray(y)
        self.classes_ = np.unique(labels)
        self.fit_ = penalty_path(
            np.asarray(X, dtype=float), labels, sample_weight, [float(self.penalty)]
        )[0]
        return self

    def predict_proba(self, X: Any) -> FloatArray:
        """The class probabilities, in :attr:`classes_` order.

        Parameters
        ----------
        X : array-like
            The ``(m, p)`` covariates.

        Returns
        -------
        ndarray of float
            The ``(m, K)`` probabilities.
        """
        return self.fit_.predict_proba(np.asarray(X, dtype=float))

    def predict(self, X: Any) -> Any:
        """The most probable class.

        Parameters
        ----------
        X : array-like
            The ``(m, p)`` covariates.

        Returns
        -------
        ndarray
            One class for each row.
        """
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
