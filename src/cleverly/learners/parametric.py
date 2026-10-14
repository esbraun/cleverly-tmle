"""Unpenalized logistic GLMs on a declared, fixed feature map.

The two learners here fit the nuisance models whose coefficients a stacked estimating
equation can carry. Each fit is the maximum-likelihood solution of a logistic model on a
feature map fixed before the fit, so the fitted coefficients solve one known score
equation. The per-arm outcome-adaptive C-TMLE reads those coefficients from each model's
``diagnostics_``.

The feature maps are a fixed vocabulary, not a callable: a callable can depend on the data
(knots at sample quantiles), and then the score equation no longer has a fixed form.

Both learners solve Newton steps on internally equilibrated columns: each column is scaled to
unit root mean square, and the coefficients are mapped back. The convergence and conditioning
tests are therefore invariant to the scale of a column. A fit that does not converge, or whose
equilibrated information matrix has a reciprocal condition number below
:data:`MIN_RECIPROCAL_CONDITION`, raises :class:`~cleverly.exceptions.DataError`. Separation
is the usual cause.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.special import expit, logit
from sklearn.base import BaseEstimator

from .._typing import FloatArray
from ..exceptions import DataError

__all__ = [
    "MIN_RECIPROCAL_CONDITION",
    "OutcomeFeatures",
    "ParametricFitDiagnostics",
    "ParametricMechanism",
    "ParametricOutcome",
    "logistic_newton",
]

MIN_RECIPROCAL_CONDITION = 1e-12
"""The smallest reciprocal condition number of the equilibrated information matrix."""

OutcomeFeatures = Literal["main_terms", "arm_interactions", "no_arm_main_effect"]
"""The outcome feature maps.

``D`` is the outcome design ``[I, W]`` with ``arm_columns`` treatment columns ``I`` first.

* ``"main_terms"``: ``[1, I, W]``.
* ``"arm_interactions"``: ``[1, I, W, I x W]``.
* ``"no_arm_main_effect"``: ``[1, W, I x W]``.
"""

_OUTCOME_FEATURES: tuple[str, ...] = ("main_terms", "arm_interactions", "no_arm_main_effect")
_MAX_ITERATIONS = 100
_STEP_TOLERANCE = 1e-12
_MAX_HALVINGS = 50


@dataclass(frozen=True)
class ParametricFitDiagnostics:
    """What one parametric fit reports.

    Parameters
    ----------
    learner : str
        The learner class name.
    coef : tuple of float
        The fitted coefficients on the original feature scale, intercept first.
    converged : bool
        Whether Newton's method met its step tolerance. A fit that did not converge raises,
        so a stored entry always reads ``True``.
    score_norm : float
        The largest absolute weighted mean score at the fitted coefficients.
    iterations : int
        The Newton steps taken.
    reciprocal_condition : float
        The reciprocal condition number of the equilibrated information matrix at the fit.
    """

    learner: str
    coef: tuple[float, ...]
    converged: bool
    score_norm: float
    iterations: int
    reciprocal_condition: float


def _deviance(z: FloatArray, y: FloatArray, w: FloatArray, beta: FloatArray) -> float:
    """The weighted binomial deviance of a logistic fit, finite at fitted values of 0 or 1."""
    eta = z @ beta
    # log(1 + exp(eta)) - y eta, the negative log likelihood of a quasi-binomial row.
    return float(np.sum(w * (np.logaddexp(0.0, eta) - y * eta)))


def logistic_newton(
    z: FloatArray, y: FloatArray, w: FloatArray, *, label: str
) -> tuple[FloatArray, ParametricFitDiagnostics]:
    """The weighted maximum-likelihood logistic fit of ``y`` on ``z``.

    ``y`` may be fractional in ``[0, 1]``: the score ``z (y - expit(z' beta))`` is the
    quasi-binomial one. Newton's method runs on columns scaled to unit root mean square, with
    step halving when a step raises the deviance.

    Parameters
    ----------
    z : ndarray
        ``(n, p)``, the feature matrix, with its intercept column if any.
    y : ndarray
        ``(n,)``, the target in ``[0, 1]``.
    w : ndarray
        ``(n,)``, nonnegative row weights.
    label : str
        The learner name the diagnostics and any error report.

    Returns
    -------
    tuple
        The coefficients on the original scale, and the fit's diagnostics.

    Raises
    ------
    DataError
        If Newton's method does not converge, or the equilibrated information matrix is
        numerically singular.
    """
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    w = np.asarray(w, dtype=float).reshape(-1)
    total = float(np.sum(w))
    if total <= 0.0:
        raise DataError(f"{label} has no row with positive weight to fit")
    if np.any((y < 0.0) | (y > 1.0)) or not np.all(np.isfinite(y)):
        raise DataError(f"{label} needs a target in [0, 1]")
    scale = np.sqrt(np.sum(w[:, None] * z**2, axis=0) / total)
    scale = np.where(scale > 0.0, scale, 1.0)
    zs = z / scale
    beta = np.zeros(zs.shape[1])
    deviance = _deviance(zs, y, w, beta)
    converged = False
    iterations = 0
    rcond = 0.0
    for iterations in range(1, _MAX_ITERATIONS + 1):  # noqa: B007 - read after the loop
        mean = expit(zs @ beta)
        score = zs.T @ (w * (y - mean)) / total
        information = (zs * (w * mean * (1.0 - mean))[:, None]).T @ zs / total
        rcond = _reciprocal_condition(information)
        if rcond < MIN_RECIPROCAL_CONDITION:
            break
        step = np.linalg.solve(information, score)
        candidate = beta + step
        candidate_deviance = _deviance(zs, y, w, candidate)
        halvings = 0
        while candidate_deviance > deviance + 1e-12 * max(1.0, abs(deviance)):
            halvings += 1
            if halvings > _MAX_HALVINGS:
                break
            step = step / 2.0
            candidate = beta + step
            candidate_deviance = _deviance(zs, y, w, candidate)
        beta, deviance = candidate, candidate_deviance
        if np.max(np.abs(step)) < _STEP_TOLERANCE * max(1.0, float(np.max(np.abs(beta)))):
            converged = True
            break
    mean = expit(zs @ beta)
    score = zs.T @ (w * (y - mean)) / total
    information = (zs * (w * mean * (1.0 - mean))[:, None]).T @ zs / total
    rcond = _reciprocal_condition(information)
    if rcond < MIN_RECIPROCAL_CONDITION:
        raise DataError(
            f"{label}: the information matrix is numerically singular (reciprocal condition "
            f"{rcond:.3g} below {MIN_RECIPROCAL_CONDITION:g}). The usual cause is separation: "
            "a feature combination predicts the target exactly, so the maximum-likelihood "
            "fit does not exist. A collinear feature map has the same effect."
        )
    if not converged:
        raise DataError(
            f"{label}: Newton's method did not converge in {_MAX_ITERATIONS} steps. The usual "
            "cause is separation: a feature combination predicts the target exactly, so the "
            "maximum-likelihood fit does not exist."
        )
    coef = beta / scale
    raw_score = z.T @ (w * (y - expit(z @ coef))) / total
    diagnostics = ParametricFitDiagnostics(
        learner=label,
        coef=tuple(float(value) for value in coef),
        converged=True,
        score_norm=float(np.max(np.abs(raw_score))),
        iterations=iterations,
        reciprocal_condition=rcond,
    )
    return coef, diagnostics


def _reciprocal_condition(matrix: FloatArray) -> float:
    """``min / max`` eigenvalue of a symmetric positive semidefinite matrix, 0 if degenerate."""
    if not np.all(np.isfinite(matrix)):
        return 0.0
    values = np.linalg.eigvalsh(matrix)
    top = float(values[-1])
    if top <= 0.0:
        return 0.0
    return max(float(values[0]), 0.0) / top


def _fit_target(y: Any) -> FloatArray:
    return np.asarray(y, dtype=float).reshape(-1)


def _fit_weights(sample_weight: Any, n: int) -> FloatArray:
    if sample_weight is None:
        return np.ones(n)
    weights = np.asarray(sample_weight, dtype=float).reshape(-1)
    if weights.shape != (n,) or np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
        raise DataError("sample_weight must hold one finite, nonnegative weight per row")
    return weights


class _ParametricLogistic(BaseEstimator):
    """The shared fit and prediction of the two parametric learners."""

    coef_: FloatArray
    classes_: FloatArray
    diagnostics_: ParametricFitDiagnostics

    def design_matrix(self, x: Any) -> FloatArray:  # pragma: no cover - overridden
        """The feature matrix of ``x``, intercept first."""
        raise NotImplementedError

    def fit(self, x: Any, y: Any, sample_weight: Any = None) -> _ParametricLogistic:
        """Fit the logistic model by Newton's method.

        Parameters
        ----------
        x : array-like
            ``(n, d)``, the design.
        y : array-like
            ``(n,)``, a binary target or a target in ``[0, 1]``.
        sample_weight : array-like, optional
            ``(n,)``, nonnegative row weights.

        Returns
        -------
        _ParametricLogistic
            The fitted learner.

        Raises
        ------
        DataError
            If the fit does not converge or its information matrix is singular.
        """
        z = self.design_matrix(x)
        target = _fit_target(y)
        weights = _fit_weights(sample_weight, z.shape[0])
        coef, diagnostics = logistic_newton(z, target, weights, label=type(self).__name__)
        self.coef_ = coef
        self.classes_ = np.array([0, 1])
        self.converged_ = diagnostics.converged
        self.score_norm_ = diagnostics.score_norm
        self.diagnostics_ = diagnostics
        return self

    def predict(self, x: Any) -> FloatArray:
        """The fitted mean ``expit(features(x)' coef)``.

        Parameters
        ----------
        x : array-like
            ``(n, d)``, the design.

        Returns
        -------
        ndarray
            ``(n,)``, the fitted mean.
        """
        return np.asarray(expit(self.design_matrix(x) @ self.coef_), dtype=float)

    def predict_proba(self, x: Any) -> FloatArray:
        """``[1 - mean, mean]`` for the classes ``[0, 1]``.

        Parameters
        ----------
        x : array-like
            ``(n, d)``, the design.

        Returns
        -------
        ndarray
            ``(n, 2)``, the class probabilities.
        """
        mean = self.predict(x)
        return np.column_stack([1.0 - mean, mean])


class ParametricOutcome(_ParametricLogistic):
    """An unpenalized logistic GLM of the outcome on a declared feature map.

    The target is binary, or an outcome already scaled to ``[0, 1]`` (a quasi-binomial fit).
    ``predict`` returns the fitted mean, and ``predict_proba`` returns it as the class 1
    column.

    Parameters
    ----------
    features : {"main_terms", "arm_interactions", "no_arm_main_effect"}
        The feature map of :data:`OutcomeFeatures`.
    arm_columns : int
        The number of treatment columns that lead the outcome design: 1 for a binary
        treatment, ``K - 1`` for ``K`` arms.

    Attributes
    ----------
    coef_ : ndarray
        The fitted coefficients, in the column order of :meth:`design_matrix`.
    diagnostics_ : ParametricFitDiagnostics
        The fit's coefficients, convergence and conditioning.
    """

    def __init__(self, features: OutcomeFeatures = "arm_interactions", arm_columns: int = 1):
        self.features = features
        self.arm_columns = arm_columns

    def design_matrix(self, x: Any) -> FloatArray:
        """The outcome feature matrix of the design ``x``, intercept first.

        Parameters
        ----------
        x : array-like
            ``(n, d)``, the outcome design ``[I, W]`` with :attr:`arm_columns` leading
            treatment columns.

        Returns
        -------
        ndarray
            The feature matrix of :data:`OutcomeFeatures`.

        Raises
        ------
        ValueError
            If the feature map is not in the vocabulary, or the design has no covariate
            column.
        """
        if self.features not in _OUTCOME_FEATURES:
            raise ValueError(
                f"features must be one of {', '.join(map(repr, _OUTCOME_FEATURES))}; "
                f"got {self.features!r}"
            )
        k = int(self.arm_columns)
        matrix = np.asarray(x, dtype=float)
        if matrix.ndim != 2 or k < 1 or matrix.shape[1] <= k:
            raise ValueError(
                f"the outcome design needs {k} treatment column(s) followed by at least one "
                f"covariate; got shape {matrix.shape}"
            )
        arms = matrix[:, :k]
        covariates = matrix[:, k:]
        ones = np.ones((matrix.shape[0], 1))
        if self.features == "main_terms":
            return np.column_stack([ones, arms, covariates])
        products = (arms[:, :, None] * covariates[:, None, :]).reshape(matrix.shape[0], -1)
        if self.features == "arm_interactions":
            return np.column_stack([ones, arms, covariates, products])
        return np.column_stack([ones, covariates, products])


class ParametricMechanism(_ParametricLogistic):
    """An unpenalized logistic regression of an indicator on a polynomial in ``logit q``.

    The design is one column ``q``, the outcome model's prediction at one arm. The basis is
    ``phi(q) = (1, l, l^2, ..., l^degree)`` with ``l = logit(clip(q, clip, 1 - clip))``.

    Parameters
    ----------
    degree : int
        The polynomial degree in ``logit q``.
    clip : float
        The basis clip of ``q`` before the logit.

    Attributes
    ----------
    coef_ : ndarray
        The fitted coefficients, in the column order of :meth:`design_matrix`.
    diagnostics_ : ParametricFitDiagnostics
        The fit's coefficients, convergence and conditioning.
    """

    def __init__(self, degree: int = 3, clip: float = 1e-6):
        self.degree = degree
        self.clip = clip

    def _column(self, x: Any) -> FloatArray:
        matrix = np.asarray(x, dtype=float)
        if matrix.ndim == 1:
            return matrix
        if matrix.ndim != 2 or matrix.shape[1] != 1:
            raise ValueError(
                "ParametricMechanism reads one column, the outcome prediction at one arm; "
                f"got shape {matrix.shape}"
            )
        return matrix[:, 0]

    def _check(self) -> None:
        if int(self.degree) != self.degree or self.degree < 1:
            raise ValueError(f"degree must be a positive integer; got {self.degree!r}")
        if not 0.0 < self.clip < 0.5:
            raise ValueError(f"clip must lie in (0, 0.5); got {self.clip!r}")

    def design_matrix(self, x: Any) -> FloatArray:
        """The basis ``phi(q)``.

        Parameters
        ----------
        x : array-like
            ``(n,)`` or ``(n, 1)``, the prediction ``q``.

        Returns
        -------
        ndarray
            ``(n, degree + 1)``, the basis.
        """
        self._check()
        q = self._column(x)
        level = logit(np.clip(q, self.clip, 1.0 - self.clip))
        return np.column_stack([level**power for power in range(int(self.degree) + 1)])

    def derivative(self, x: Any) -> FloatArray:
        """``d phi / d logit(q)``, zero on a row where the basis clip binds.

        Parameters
        ----------
        x : array-like
            ``(n,)`` or ``(n, 1)``, the prediction ``q``.

        Returns
        -------
        ndarray
            ``(n, degree + 1)``, the derivative of each basis column.
        """
        self._check()
        q = self._column(x)
        level = logit(np.clip(q, self.clip, 1.0 - self.clip))
        inside = ((q > self.clip) & (q < 1.0 - self.clip)).astype(float)
        columns = [np.zeros_like(level)] + [
            power * level ** (power - 1) for power in range(1, int(self.degree) + 1)
        ]
        return np.column_stack(columns) * inside[:, None]
