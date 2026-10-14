r"""Data-tuned caps of a modified treatment policy (X40 part (e)).

A :class:`~cleverly.interventions.Shift` or :class:`~cleverly.interventions.Scale` can take
its cap from the treatment law instead of a declared number.  The cap is then a *tuner*:
:class:`DoseQuantile` (a quantile of the natural dose among the units at risk) or
:class:`SupportBoundary` (a learned conditional quantile of the dose).  The map depends on
the sample, so the analyst declares which parameter the fit estimates:

.. list-table::
   :header-rows: 1

   * - ``target``
     - parameter
     - tuners
   * - ``"population"``
     - :math:`\Psi(P) = \Phi(P, \theta(P))`, the mean under the map at the population tuner
     - :class:`DoseQuantile`, and :class:`SupportBoundary` with
       :class:`LinearQuantileRegression`
   * - ``"fold_average"``
     - :math:`(1/V) \sum_v \Phi(P_0, \theta_{n,v})`, the average over folds of the means under
       each fold's map, fitted on that fold's training rows (Hubbard, Kherad-Pajouh and van
       der Laan 2016, Section 2)
     - every tuner

The population target has the influence curve
:math:`D_\Psi = D_\Phi(\theta_0) + \sum_t \partial_t\Phi(\theta_0)' \mathrm{IF}_t` (the chain
rule).  This module holds the tuners, their fits, their influence functions
:math:`\mathrm{IF}_t`, and the kernel estimator of :math:`\partial_t\Phi`.

* A dose quantile has :math:`\mathrm{IF} = -(1\{A \le \theta\} - q) 1\{r\} / (\pi f(\theta))`,
  with :math:`\pi` the at-risk share and :math:`f` the at-risk density, read by a normal
  kernel at Silverman's bandwidth.
* A linear quantile regression :math:`\beta(P) = \arg\min_\beta E[\rho_q(A - \beta'\tilde H)
  \mid r]` has :math:`\mathrm{IF} = J^{-1} \tilde H (q - 1\{A \le \beta'\tilde H\}) 1\{r\} / \pi`,
  :math:`J = E[f(\beta'\tilde H \mid H) \tilde H \tilde H' \mid r]`.  :math:`\beta(P)` is the
  pseudo-true value, so the curve holds under misspecification (Angrist, Chernozhukov and
  Fernández-Val 2006, Theorem 3 and its linear representation, read in the December 2004
  working paper).  :math:`J` is Powell's box-kernel estimator
  :math:`(2 n h)^{-1} \sum_i 1\{|A_i - \hat\beta'\tilde H_i| \le h\} \tilde H_i \tilde H_i'` with
  :math:`h = C n^{-1/3}` and :math:`C` Silverman's scale of the at-risk residuals, the
  estimator that source proves consistent under misspecification.

The derivative of the mean under a capped map with boundary :math:`c_t(h)` and last moved
dose :math:`e_t(h)` (:math:`c - \delta` for a shift, :math:`c / f` for a scale) is

.. math::

    \partial\Phi / \partial\theta_t = E\bigl[W_{1:t-1}\, g_t(e_t \mid H_t)\, |e_t'|\,
        \{m_t(c_t, H_t) - m_t(e_t, H_t)\}\, \partial c_t / \partial\theta_t\bigr],

which :func:`cap_derivative` estimates by a normal kernel average in :math:`A_t - e_t(H_t)`,
so it needs no density value and runs on both ratio routes.

``bandwidth="undersmoothed"`` is the declared fallback of X40 plan section 2.5: half of
Silverman's bandwidth for :math:`f` and :math:`\partial\Phi`, and :math:`(C/2) n^{-1/3}` for
:math:`J`.  Both bandwidths keep :math:`h \to 0` and :math:`n h \to \infty` (and
:math:`n h^2 \to \infty` for :math:`J`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np

from .._typing import BoolArray, FloatArray
from ..exceptions import CapabilityError, DataError

__all__ = [
    "TARGETS",
    "DoseQuantile",
    "FittedTuner",
    "LinearQuantileRegression",
    "SupportBoundary",
    "cap_derivative",
    "fit_tuner",
    "is_tuner",
    "kernel_bandwidth",
    "tuner_influence",
]

#: The declared targets of a tuned map.
TARGETS = ("population", "fold_average")
#: The declared bandwidth rules.
BANDWIDTHS = ("silverman", "undersmoothed")
Bandwidth = Literal["silverman", "undersmoothed"]
Target = Literal["population", "fold_average"]


def _check_level(q: object, what: str) -> float:
    level = float(q)  # type: ignore[arg-type]
    if not 0.0 < level < 1.0:
        raise DataError(f"{what} needs a level q in (0, 1); got {q!r}")
    return level


def _check_bandwidth(bandwidth: object) -> None:
    if bandwidth not in BANDWIDTHS:
        raise DataError(f"bandwidth must be 'silverman' or 'undersmoothed'; got {bandwidth!r}")


def _check_features(features: object, what: str) -> tuple[str, ...]:
    if isinstance(features, str):
        features = (features,)
    items: tuple[object, ...] = tuple(cast(Iterable[object], features))
    names = tuple(name for name in items if isinstance(name, str) and name)
    if len(names) != len(items):
        raise DataError(f"{what} names its features as column names; got {features!r}")
    if len(set(names)) != len(names):
        raise DataError(f"{what} names a feature twice: {list(names)}")
    return names


@dataclass(frozen=True)
class DoseQuantile:
    """The cap at a quantile of the natural dose among the units at risk.

    ``Shift(1.0, cap=DoseQuantile(0.5), target="population")`` moves every dose up by one,
    except a dose whose move would pass the at-risk median.  The quantile is a
    finite-dimensional tuner, so both declared targets apply.

    Parameters
    ----------
    q : float
        The level, in ``(0, 1)``.
    bandwidth : {"silverman", "undersmoothed"}
        The kernel bandwidth of the at-risk density and of the derivative of the mean:
        Silverman's rule, or half of it.
    """

    q: float
    bandwidth: Bandwidth = "silverman"

    def __post_init__(self) -> None:
        object.__setattr__(self, "q", _check_level(self.q, "DoseQuantile"))
        _check_bandwidth(self.bandwidth)


@dataclass(frozen=True)
class LinearQuantileRegression:
    """A declared linear quantile regression: the boundary ``beta' (1, features)``.

    The finite-dimensional boundary learner of :class:`SupportBoundary`.  It fits the
    level-``q`` check loss of the dose on an intercept and the named columns of the policy
    frame, on the units at risk.  :math:`\\beta` solves an estimating equation, so the
    population target has an influence curve, whether or not the regression is correctly
    specified (the pseudo-true :math:`\\beta`).

    Parameters
    ----------
    features : tuple of str
        Columns of the policy frame.  An intercept is added.
    """

    features: tuple[str, ...]

    def __post_init__(self) -> None:
        names = _check_features(self.features, "LinearQuantileRegression")
        if not names:
            raise DataError("LinearQuantileRegression needs at least one feature")
        object.__setattr__(self, "features", names)


@dataclass(frozen=True)
class SupportBoundary:
    """The cap at a learned conditional quantile of the dose: ``c(h)``.

    The learner is :class:`LinearQuantileRegression` or an unfitted scikit-learn regressor
    that estimates the level-``q`` conditional quantile, for example
    ``GradientBoostingRegressor(loss="quantile", alpha=q)``.  The fit clones it and fits it
    on the units at risk.  Only :class:`LinearQuantileRegression` has a finite-dimensional
    parameter, so a different learner supports ``target="fold_average"`` only.

    Parameters
    ----------
    learner : LinearQuantileRegression or estimator
        The boundary learner.
    q : float
        The quantile level, in ``(0, 1)``.
    features : tuple of str
        The policy-frame columns a learner other than :class:`LinearQuantileRegression`
        reads.  Empty for :class:`LinearQuantileRegression`, which names its own.
    bandwidth : {"silverman", "undersmoothed"}
        The bandwidth rule of the derivative and of :math:`J`.

    Attributes
    ----------
    feature_names : tuple of str
    finite_dimensional : bool
    """

    learner: Any
    q: float
    features: tuple[str, ...] = ()
    bandwidth: Bandwidth = "silverman"

    def __post_init__(self) -> None:
        object.__setattr__(self, "q", _check_level(self.q, "SupportBoundary"))
        _check_bandwidth(self.bandwidth)
        features = _check_features(self.features, "SupportBoundary")
        object.__setattr__(self, "features", features)
        if isinstance(self.learner, LinearQuantileRegression):
            if features and features != self.learner.features:
                raise DataError(
                    "SupportBoundary with LinearQuantileRegression reads the regression's own "
                    "features; leave features= empty"
                )
            return
        if not (hasattr(self.learner, "fit") and hasattr(self.learner, "predict")):
            raise DataError(
                "SupportBoundary's learner is LinearQuantileRegression or an estimator with "
                f"fit and predict; got {type(self.learner).__name__}"
            )
        if not features:
            raise DataError(
                "SupportBoundary with a learner other than LinearQuantileRegression names the "
                "policy-frame columns it reads with features="
            )

    @property
    def feature_names(self) -> tuple[str, ...]:
        """The policy-frame columns the boundary reads."""
        if isinstance(self.learner, LinearQuantileRegression):
            return self.learner.features
        return self.features

    @property
    def finite_dimensional(self) -> bool:
        """Whether the boundary is indexed by a finite vector that solves an estimating equation."""
        return isinstance(self.learner, LinearQuantileRegression)


def is_tuner(value: object) -> bool:
    """Whether a cap is a tuner and not a declared number.

    Parameters
    ----------
    value : object
        A ``cap``.

    Returns
    -------
    bool
        ``True`` for :class:`DoseQuantile` and :class:`SupportBoundary`.
    """
    return isinstance(value, (DoseQuantile, SupportBoundary))


def tuner_reads(tuner: DoseQuantile | SupportBoundary) -> tuple[str, ...]:
    """The policy-frame columns a tuner reads beside the dose.

    Parameters
    ----------
    tuner : DoseQuantile or SupportBoundary
        The tuner.

    Returns
    -------
    tuple of str
        Empty for a dose quantile.
    """
    return () if isinstance(tuner, DoseQuantile) else tuner.feature_names


# ------------------------------------------------------------------ fits


def _normalised(weights: FloatArray | None, n: int) -> FloatArray:
    if weights is None:
        return np.ones(n)
    w = np.asarray(weights, dtype=float).reshape(-1)
    total = float(np.sum(w))
    return w * (n / total) if total > 0.0 else np.ones(n)


def weighted_quantile(values: FloatArray, weights: FloatArray, q: float) -> float:
    """The smallest value whose weighted distribution function reaches ``q``.

    Parameters
    ----------
    values : ndarray
        The sample.
    weights : ndarray
        Nonnegative weights.
    q : float
        The level.

    Returns
    -------
    float
        The inverse of the weighted empirical distribution function at ``q``.
    """
    order = np.argsort(values, kind="stable")
    sorted_values = np.asarray(values, dtype=float)[order]
    cumulative = np.cumsum(np.asarray(weights, dtype=float)[order])
    total = cumulative[-1]
    index = int(np.searchsorted(cumulative, q * total - 1e-12 * total, side="left"))
    return float(sorted_values[min(index, sorted_values.size - 1)])


def _scale(values: FloatArray, weights: FloatArray) -> float:
    """Silverman's scale ``0.9 min(sd, IQR / 1.34)`` of a weighted sample."""
    total = float(np.sum(weights))
    mean = float(np.sum(weights * values) / total)
    sd = float(np.sqrt(np.sum(weights * (values - mean) ** 2) / total))
    iqr = weighted_quantile(values, weights, 0.75) - weighted_quantile(values, weights, 0.25)
    spread = min(sd, iqr / 1.34) if iqr > 0.0 else sd
    return 0.9 * spread


def kernel_bandwidth(
    values: FloatArray, weights: FloatArray, *, rule: Bandwidth, power: float = 0.2
) -> float:
    """Silverman's bandwidth ``0.9 min(sd, IQR / 1.34) n^-power``, halved when undersmoothed.

    Parameters
    ----------
    values : ndarray
        The at-risk sample the kernel averages over.
    weights : ndarray
        Its observation weights.
    rule : {"silverman", "undersmoothed"}
        The declared rule.
    power : float
        The exponent: ``0.2`` for a density and a derivative, ``1/3`` for Powell's ``J``.

    Returns
    -------
    float
        The bandwidth.
    """
    h = _scale(values, weights) * values.size ** (-power)
    if rule == "undersmoothed":
        h = 0.5 * h
    if not np.isfinite(h) or h <= 0.0:
        raise DataError(
            "the tuned cap's kernel bandwidth is zero: the at-risk doses do not vary, so the "
            "density at the cap has no estimate"
        )
    return float(h)


def _normal(z: FloatArray) -> FloatArray:
    return np.asarray(np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi), dtype=float)


@dataclass(frozen=True)
class FittedTuner:
    """A tuner fitted on some rows: :math:`\\theta`, or a fitted boundary learner.

    Parameters
    ----------
    tuner : DoseQuantile or SupportBoundary
        The declared tuner.
    theta : ndarray
        ``(p,)`` the fitted parameter: the quantile, or the intercept and slopes of a linear
        quantile regression.  Empty for a learner of growing dimension.
    model : object or None
        The fitted learner of a :class:`SupportBoundary` that is not finite-dimensional.
    n_fitted : int
        The rows the tuner was fitted on.
    """

    tuner: DoseQuantile | SupportBoundary
    theta: FloatArray
    model: Any = None
    n_fitted: int = 0

    def design(self, frame: Any, n: int) -> FloatArray:
        """``(n, p)`` :math:`\\tilde H`: the intercept and features, or ones for a quantile.

        Parameters
        ----------
        frame : object
            The policy frame.
        n : int
            The rows.

        Returns
        -------
        ndarray
            The design of the boundary, which is also :math:`\\partial c / \\partial\\theta`.
        """
        if isinstance(self.tuner, DoseQuantile):
            return np.ones((n, 1))
        columns = [np.ones(n)] + [
            np.asarray(frame[name], dtype=float).reshape(-1) for name in self.tuner.feature_names
        ]
        return np.column_stack(columns)

    def boundary(self, frame: Any, n: int) -> FloatArray:
        """``(n,)`` the cap :math:`c(h)` on every row of ``frame``.

        Parameters
        ----------
        frame : object
            The policy frame.
        n : int
            The rows.

        Returns
        -------
        ndarray
            The boundary.
        """
        if isinstance(self.tuner, DoseQuantile):
            return np.full(n, float(self.theta[0]))
        if self.tuner.finite_dimensional:
            return np.asarray(self.design(frame, n) @ self.theta, dtype=float)
        features = self.design(frame, n)[:, 1:]
        return np.asarray(self.model.predict(features), dtype=float).reshape(-1)

    def describe(self) -> str:
        """A stable text of the fitted tuner."""
        if self.theta.size:
            values = ", ".join(f"{value:.6g}" for value in self.theta)
            return f"{type(self.tuner).__name__}(q={self.tuner.q:g}) at theta=({values})"
        learner = getattr(self.tuner, "learner", None)
        return f"SupportBoundary({type(learner).__name__}, q={self.tuner.q:g})"


def fit_tuner(
    tuner: DoseQuantile | SupportBoundary,
    doses: FloatArray,
    frame: Callable[[], Any],
    rows: BoolArray,
    weights: FloatArray | None = None,
) -> FittedTuner:
    """Fit a tuner on the given rows: the at-risk rows, or one fold's training rows of them.

    Parameters
    ----------
    tuner : DoseQuantile or SupportBoundary
        The declared tuner.
    doses : ndarray
        ``(n,)`` observed doses.
    frame : callable
        Returns the policy frame of every row.  Called only by a boundary.
    rows : ndarray
        ``(n,)`` the rows the tuner is fitted on.
    weights : ndarray or None
        ``(n,)`` observation weights.

    Returns
    -------
    FittedTuner
        The fit.
    """
    a = np.asarray(doses, dtype=float).reshape(-1)
    n = a.size
    keep = np.asarray(rows, dtype=bool)
    if not keep.any():
        raise DataError("a tuned cap has no rows at risk to fit its tuner on")
    w = _normalised(weights, n)
    if isinstance(tuner, DoseQuantile):
        theta = np.array([weighted_quantile(a[keep], w[keep], tuner.q)])
        return FittedTuner(tuner, theta, n_fitted=int(keep.sum()))
    placeholder = FittedTuner(tuner, np.zeros(0))
    design = placeholder.design(frame(), n)
    if tuner.finite_dimensional:
        theta = _linear_quantile(design[keep], a[keep], w[keep], tuner.q)
        return FittedTuner(tuner, theta, n_fitted=int(keep.sum()))
    from sklearn.base import clone

    model = clone(tuner.learner)
    try:
        model.fit(design[keep, 1:], a[keep], sample_weight=w[keep])
    except TypeError:
        if weights is not None and not np.allclose(w, 1.0):
            raise CapabilityError(
                f"the boundary learner {type(tuner.learner).__name__} takes no sample_weight, "
                "and the fit carries observation weights"
            ) from None
        model.fit(design[keep, 1:], a[keep])
    return FittedTuner(tuner, np.zeros(0), model=model, n_fitted=int(keep.sum()))


def _linear_quantile(
    design: FloatArray, target: FloatArray, weights: FloatArray, q: float
) -> FloatArray:
    """The weighted level-``q`` check-loss minimizer, by the HiGHS linear program."""
    from scipy.optimize import linprog
    from scipy.sparse import eye, hstack

    n, p = design.shape
    # min sum_i w_i (q u_i + (1 - q) v_i) subject to X beta + u - v = y, u, v >= 0.
    cost = np.concatenate([np.zeros(2 * p), q * weights, (1.0 - q) * weights])
    identity = eye(n, format="csr")
    equality = hstack([design, -design, identity, -identity], format="csr")
    solved = linprog(cost, A_eq=equality, b_eq=target, bounds=(0.0, None), method="highs")
    if not solved.success:
        raise DataError(f"the linear quantile regression of the tuned cap failed: {solved.message}")
    return np.asarray(solved.x[:p] - solved.x[p : 2 * p], dtype=float)


def tuner_influence(
    fitted: FittedTuner,
    doses: FloatArray,
    frame: Callable[[], Any],
    at_risk: BoolArray,
    weights: FloatArray | None = None,
) -> FloatArray:
    """``(n, p)`` the influence function of a finite-dimensional tuner at every row.

    Parameters
    ----------
    fitted : FittedTuner
        The tuner fitted on the at-risk rows.
    doses : ndarray
        ``(n,)`` observed doses.
    frame : callable
        Returns the policy frame.
    at_risk : ndarray
        ``(n,)`` the rows the tuner reads.
    weights : ndarray or None
        ``(n,)`` observation weights.

    Returns
    -------
    ndarray
        Zero off the at-risk rows.  Its weighted mean is :math:`\\hat\\theta - \\theta` to
        first order.
    """
    tuner = fitted.tuner
    if isinstance(tuner, SupportBoundary) and not tuner.finite_dimensional:
        raise CapabilityError("a boundary of growing dimension has no influence function here")
    a = np.asarray(doses, dtype=float).reshape(-1)
    n = a.size
    risk = np.asarray(at_risk, dtype=bool)
    w = _normalised(weights, n)
    share = float(np.sum(w[risk]) / np.sum(w))
    if isinstance(tuner, DoseQuantile):
        theta = float(fitted.theta[0])
        h = kernel_bandwidth(a[risk], w[risk], rule=tuner.bandwidth)
        density = float(np.sum(w[risk] * _normal((theta - a[risk]) / h)) / (h * np.sum(w[risk])))
        below = (a <= theta).astype(float)
        out = np.where(risk, -(below - tuner.q) / (share * density), 0.0)
        return out.reshape(-1, 1)
    design = fitted.design(frame(), n)
    residual = a - design @ fitted.theta
    jacobian = powell_jacobian(design[risk], residual[risk], w[risk], rule=tuner.bandwidth)
    score = design * (tuner.q - (residual <= 0.0).astype(float))[:, None]
    out = np.linalg.solve(jacobian, score.T).T / share
    return np.where(risk[:, None], out, 0.0)


def powell_jacobian(
    design: FloatArray, residual: FloatArray, weights: FloatArray, *, rule: Bandwidth
) -> FloatArray:
    """Powell's box-kernel estimate of the Jacobian of the quantile estimating equation.

    The estimate is
    :math:`\\hat J = (2 n h)^{-1} \\sum 1\\{|e_i| \\le h\\} \\tilde H_i \\tilde H_i'`.
    The bandwidth is :math:`h = C n^{-1/3}`, :math:`C` Silverman's scale of the residuals,
    halved when undersmoothed.

    Parameters
    ----------
    design : ndarray
        ``(n, p)`` the at-risk :math:`\\tilde H`.
    residual : ndarray
        ``(n,)`` :math:`A - \\hat\\beta'\\tilde H`.
    weights : ndarray
        ``(n,)`` observation weights.
    rule : {"silverman", "undersmoothed"}
        The declared rule.

    Returns
    -------
    ndarray
        ``(p, p)``.
    """
    h = kernel_bandwidth(residual, weights, rule=rule, power=1.0 / 3.0)
    inside = (np.abs(residual) <= h).astype(float) * weights
    jacobian = (design * inside[:, None]).T @ design / (2.0 * h * float(np.sum(weights)))
    if np.linalg.matrix_rank(jacobian) < jacobian.shape[0]:
        raise DataError(
            "the linear quantile regression's J is singular at its bandwidth: too few at-risk "
            "rows sit near the boundary to identify every coefficient"
        )
    return np.asarray(jacobian, dtype=float)


def cap_derivative(
    *,
    weights: FloatArray,
    prefix: FloatArray,
    dose: FloatArray,
    edge: FloatArray,
    high: FloatArray,
    low: FloatArray,
    gradient: FloatArray,
    slope: float,
    at_risk: BoolArray,
    rule: Bandwidth,
) -> FloatArray:
    """The kernel estimate of :math:`\\partial\\Phi / \\partial\\theta_t`.

    The estimate is :math:`P_n[w\\, W_{1:t-1} |e'| K_h(A_t - e_t)
    \\{m_t(c_t) - m_t(e_t)\\} \\partial c_t/\\partial\\theta_t]`.
    It uses a normal kernel at Silverman's bandwidth of the at-risk :math:`A_t - e_t(H_t)`.

    Parameters
    ----------
    weights : ndarray
        ``(n,)`` observation weights, normalised to mean one.
    prefix : ndarray
        ``(n,)`` :math:`W_{1:t-1}`, the cumulative ratio through the node before; zero off
        the rows that reach the node.
    dose : ndarray
        ``(n,)`` :math:`A_t`.
    edge : ndarray
        ``(n,)`` :math:`e_t(H_t)`, the last dose the cap lets move.
    high : ndarray
        ``(n,)`` :math:`m_t(c_t(H_t), H_t)`.
    low : ndarray
        ``(n,)`` :math:`m_t(e_t(H_t), H_t)`.
    gradient : ndarray
        ``(n, p)`` :math:`\\partial c_t / \\partial\\theta_t`.
    slope : float
        :math:`|e_t'|`: one for a shift and ``1 / factor`` for a scale.
    at_risk : ndarray
        ``(n,)`` the rows at risk at the node.
    rule : {"silverman", "undersmoothed"}
        The declared bandwidth rule.

    Returns
    -------
    ndarray
        ``(p,)``.
    """
    risk = np.asarray(at_risk, dtype=bool)
    w = np.asarray(weights, dtype=float)
    gap = np.asarray(dose, dtype=float) - np.asarray(edge, dtype=float)
    h = kernel_bandwidth(gap[risk], w[risk], rule=rule)
    kernel = _normal(gap / h) / h
    term = np.where(risk, w * prefix * slope * kernel * (high - low), 0.0)
    return np.asarray(term @ gradient / w.size, dtype=float)


def check_target(policy_name: str, target: object, tuned: bool) -> None:
    """Refuse an unknown ``target``, a target on an untuned map, or a tuned map with none.

    Parameters
    ----------
    policy_name : str
        The policy's report label.
    target : object
        The declared ``target``.
    tuned : bool
        Whether the cap is a tuner.

    Raises
    ------
    DataError
        If ``target`` is not ``"population"``, ``"fold_average"`` or ``None``, or a map with
        a declared cap declares one.
    CapabilityError
        If a tuned map declares no target (R-e1).
    """
    if target is not None and not (isinstance(target, str) and target in TARGETS):
        raise DataError(
            f"target must be 'population', 'fold_average' or None; got {target!r}. {TARGET_MEANING}"
        )
    if target is not None and not tuned:
        raise DataError(
            f"policy {policy_name!r} declares target={target!r}, but its cap is a declared "
            "number; a target declares the parameter of a data-tuned map only"
        )
    if tuned and target is None:
        raise CapabilityError(f"policy {policy_name!r}: {TARGET_UNDECLARED}")


#: The sentence the refusal of an unknown ``target`` appends.
TARGET_MEANING = "It declares which parameter a data-tuned map estimates."
#: R-e1: a tuned map with no declared target.
TARGET_UNDECLARED = (
    "a data-tuned map reads the sample, so which parameter it estimates is a declaration. "
    "Declare target='population', the mean under the map at the population value of the "
    "tuner (a dose quantile, or LinearQuantileRegression), or target='fold_average', the "
    "average over folds of the means under each fold's own map, fitted on that fold's "
    "training rows"
)


def population_refusal(policy_name: str, tuner: SupportBoundary) -> str:
    """R-e2: the population target with a boundary of growing dimension.

    Parameters
    ----------
    policy_name : str
        The policy's report label.
    tuner : SupportBoundary
        The tuner.

    Returns
    -------
    str
        The refusal text.
    """
    learner = type(tuner.learner).__name__
    return (
        f"policy {policy_name!r} tunes its map with SupportBoundary, whose learner "
        f"{learner!r} has no declared finite-dimensional parameter. The population target "
        "needs the influence function of the tuned boundary, and only a boundary indexed by a "
        "finite vector that solves an estimating equation has one here (a dose quantile, or "
        "LinearQuantileRegression with its estimating equation). A learner of growing "
        "dimension has no such curve in any result this package uses. Declare "
        "target='fold_average', which estimates the average over folds of the means under "
        "each fold's own map, or tune the boundary with DoseQuantile or "
        "LinearQuantileRegression"
    )


def outcome_refusal(policy_name: str) -> str:
    """R-e3: a tuner that reads the outcome.

    Parameters
    ----------
    policy_name : str
        The policy's report label.

    Returns
    -------
    str
        The refusal text.
    """
    return (
        f"policy {policy_name!r} tunes its map with the outcome. A map chosen to optimize the "
        "outcome is a learned policy, whose value needs a limiting-policy condition and its "
        "own contract (docs/roadmap.md X11). A data-tuned map in this release reads the "
        "treatment law only: a dose quantile or a learned support boundary"
    )


#: R-e4: the fold-average target with one fold.
FOLD_AVERAGE_ONE_FOLD = (
    "target='fold_average' builds each fold's map on that fold's training rows and estimates "
    "its mean on the held-out rows, so it needs n_folds of at least 2. For an in-sample fit, "
    "tune the map with a finite-dimensional tuner and declare target='population'"
)


def refuse_outcome_tuners(
    tuners: Sequence[tuple[str, DoseQuantile | SupportBoundary]], outcomes: Sequence[str]
) -> None:
    """Refuse a tuner whose features name an outcome column (R-e3).

    Parameters
    ----------
    tuners : sequence of tuple
        ``(policy name, tuner)`` pairs.
    outcomes : sequence of str
        The outcome columns of the fit.

    Raises
    ------
    CapabilityError
        If a tuner reads an outcome column.
    """
    named = set(outcomes)
    for name, tuner in tuners:
        if named.intersection(tuner_reads(tuner)):
            raise CapabilityError(outcome_refusal(name))
