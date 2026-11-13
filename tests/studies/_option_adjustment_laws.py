r"""Laws, learners, fits and exact-law limits of ``option-treatment-adjustment`` (X76 part (d)).

``treatment_adjustment=`` (``TMLE.fit(treatment_covariates=)``) fits the treatment mechanism on a
subset :math:`W_g` of the adjustment set :math:`W`.  It declares the model restriction that the
mechanism depends on :math:`W` only through :math:`W_g`.  With :math:`Z = W \setminus W_g` and
:math:`g_\infty` the limit of the mechanism fit on :math:`W_g`, the arm-1 remainder is
:math:`P_0[(1 - g_0/g_\infty)(Q^* - Q_0)]` (``docs/technical-reference/point-treatment-tmle.md``,
"The remainder when the restriction fails").  It vanishes when the restriction holds, and it is
linear in the outcome error when the restriction fails (Fix-8 plan section 2.4).

Two laws (Fix-8 plan section 5.4), with :math:`W_1, W_2, Z` iid standard normal and
:math:`W_g = (W_1, W_2)`:

======  ==========================================  ===============================================
law     treatment mechanism                         outcome regression
======  ==========================================  ===============================================
``LDA`` :math:`\operatorname{expit}(0.5 W_1 -        :math:`\operatorname{logit} Q_0 = -0.3 + 0.6 A
        0.5 W_2)`: the restriction holds             + 0.7 W_1 + 0.4 W_2 + 0.8 Z`
``LDB`` :math:`\operatorname{expit}(0.5 W_1 -        :math:`\operatorname{logit} Q_0 = -0.3 + 0.6 A
        0.5 W_2 + Z)`: :math:`Z` is an instrument    + 0.7 W_1 + 0.4 W_2`
======  ==========================================  ===============================================

A logistic GLM on :math:`(A, W)` is correct for both outcome laws, and a logistic GLM on
:math:`W_g` is correct for the ``LDA`` mechanism.  On ``LDB`` the GLM on :math:`W_g` converges to
the population logistic projection :math:`g_\infty`, which :func:`mechanism_limit` computes.

**The constructed error.**  :class:`PerturbedLogistic` fits the correct GLM and adds the logit
offset :math:`k n^{-1/4} (Z - E[Z \mid W_g])` at prediction, with :math:`n` the full sample size.
It is a deliberate, deterministic error, not a fitted learner: it shrinks slower than root n and
varies only in the direction of :math:`Z` given :math:`W_g`, so the first part of the remainder
sees all of it.  ``Z`` is independent of :math:`W_g` with mean 0, so the centring is 0.

**Exact-law limits.**  :func:`targeted_limit` solves the targeting step on a weighted finite set
of covariate points: for each arm it solves
:math:`E[\pi_a H_a (Q_{0,a} - Q^*_a(\epsilon))] = 0` with :math:`H_a = 1/g_{\infty,a}` and returns
the targeted arm means.  The design test runs it on a Gauss-Hermite grid; the fast tier
(``tests/unit/test_option_adjustment_witness.py``) runs the same function on an exact finite law
and checks it against the package fit at 1e-10.  :func:`stacked_variance` gives the asymptotic
variance of the ATE from the stacked estimating equations of the two GLMs, the fluctuation and
the arm means, and the limit of the plug-in variance the package reports.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin

#: The adjustment set and the declared subset the mechanism reads.
COVARIATES = ("W1", "W2", "Z")
SUBSET = ("W1", "W2")
#: The outcome learner's design is ``(A, W1, W2, Z)``; ``Z`` is its last column.
INSTRUMENT_COLUMN = 3
#: ``E[Z | W_g]`` under both laws: ``Z`` is independent of ``W_g`` with mean 0.
INSTRUMENT_CENTRE = 0.0

#: The fit settings: ``canonical-tmle``'s treatment bounds (they never bind on ``LDA``, the
#: design test checks), and plan section 5.0's ``max_iter`` and ``tol``.
G_BOUNDS = (0.01, 0.99)
MAX_ITER = 100
TOL = 1e-10
ESTIMANDS = ("ey1", "ey0", "ate")

#: ``(intercept, A, W1, W2, Z)`` of each law's outcome logit.
OUTCOME = {"LDA": (-0.3, 0.6, 0.7, 0.4, 0.8), "LDB": (-0.3, 0.6, 0.7, 0.4, 0.0)}
#: ``(intercept, W1, W2, Z)`` of each law's treatment logit.
TREATMENT = {"LDA": (0.0, 0.5, -0.5, 0.0), "LDB": (0.0, 0.5, -0.5, 1.0)}
LAWS = tuple(OUTCOME)

#: The constructed error's multiplier ``k`` of ``k n^(-1/4) Z`` (plan 5.4).  Plan 5.4 asks
#: that the exact-law bias at n = 16,000 be at least :data:`CONTROL_FLOOR_SD` of the estimator's
#: SD.  At ``k = 1`` it is 1.91 SD there and 0.95 SD at n = 1,000 (the design test computes
#: both), so every rung of the ladder reads beyond the bias margin and the slope of its log
#: absolute bias has a Monte Carlo error of a few percent.  ``k = 0.25`` would just reach the
#: floor at n = 16,000 and leave the smaller rungs inside the margin.
SLOW_OFFSET = 1.0
#: The least exact-law limit of a control, in SDs (plan 5.0).
CONTROL_FLOOR_SD = 0.45

#: Gauss-Hermite nodes per axis: 1-D truths and the 3-D grid of the exact-law limits.
TRUTH_NODES = 200
GRID_NODES = 40


# ------------------------------------------------------------------------------- the laws
def g0(law: str, w1: Any, w2: Any, z: Any) -> Any:
    """``P(A = 1 | W)`` of ``law``."""
    c0, c1, c2, cz = TREATMENT[law]
    return expit(c0 + c1 * np.asarray(w1) + c2 * np.asarray(w2) + cz * np.asarray(z))


def qbar0(law: str, a: Any, w1: Any, w2: Any, z: Any) -> Any:
    """``E[Y | A = a, W]`` of ``law``."""
    b0, ba, b1, b2, bz = OUTCOME[law]
    return expit(
        b0 + ba * np.asarray(a) + b1 * np.asarray(w1) + b2 * np.asarray(w2) + bz * np.asarray(z)
    )


def draw(law: str, n: int, seed: int) -> pd.DataFrame:
    """One sample of ``law``: columns ``Y, A, W1, W2, Z``."""
    rng = np.random.default_rng(seed)
    w1, w2, z = rng.standard_normal((3, n))
    a = rng.binomial(1, g0(law, w1, w2, z)).astype(float)
    y = rng.binomial(1, qbar0(law, a, w1, w2, z)).astype(float)
    return pd.DataFrame({"Y": y, "A": a, "W1": w1, "W2": w2, "Z": z})


def gauss_hermite(nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Standard-normal Gauss-Hermite nodes and weights."""
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    return x, w / np.sqrt(2.0 * np.pi)


def truth(law: str) -> dict[str, float]:
    """The arm means and the ATE of ``law``, by 1-D Gauss-Hermite quadrature.

    The outcome logit depends on ``W`` through ``b1 W1 + b2 W2 + bz Z``, a centred normal with
    variance ``b1^2 + b2^2 + bz^2``, so one dimension integrates it exactly.
    """
    b0, ba, b1, b2, bz = OUTCOME[law]
    x, w = gauss_hermite(TRUTH_NODES)
    index = np.sqrt(b1**2 + b2**2 + bz**2) * x
    one = float(np.sum(w * expit(b0 + ba + index)))
    zero = float(np.sum(w * expit(b0 + index)))
    return {"ey1": one, "ey0": zero, "ate": one - zero}


@dataclass(frozen=True)
class Points:
    """A weighted finite set of covariate points: a quadrature grid or an exact finite law.

    Parameters
    ----------
    w1 : numpy.ndarray
        ``W1`` at each point.
    w2 : numpy.ndarray
        ``W2`` at each point.
    z : numpy.ndarray
        ``Z`` at each point.
    weight : numpy.ndarray
        The probability of each point; the weights sum to one.
    """

    w1: np.ndarray
    w2: np.ndarray
    z: np.ndarray
    weight: np.ndarray

    def mean(self, values: Any) -> float:
        """The weighted mean of ``values`` over the points."""
        return float(np.sum(self.weight * np.asarray(values, dtype=float)))

    def columns(self, names: tuple[str, ...]) -> np.ndarray:
        """The named covariates as a design matrix, without an intercept."""
        lookup = {"W1": self.w1, "W2": self.w2, "Z": self.z}
        return (
            np.column_stack([lookup[name] for name in names])
            if names
            else np.empty((len(self.weight), 0))
        )


def grid(nodes: int = GRID_NODES) -> Points:
    """The 3-D Gauss-Hermite grid of ``(W1, W2, Z)``."""
    x, w = gauss_hermite(nodes)
    w1, w2, z = (axis.reshape(-1) for axis in np.meshgrid(x, x, x, indexing="ij"))
    weight = np.einsum("i,j,k->ijk", w, w, w).reshape(-1)
    return Points(w1, w2, z, weight)


# ---------------------------------------------------------------------------- the learners
def newton_logistic(design: np.ndarray, target: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Weighted logistic maximum likelihood with an intercept, by Newton, ridge ``1e-12``.

    ``target`` may be a probability: a weighted finite law passes ``Q_0`` itself, and the
    population fit solves the same score equation.
    """
    x = np.column_stack([np.ones(len(target)), design])
    beta = np.zeros(x.shape[1])
    beta[0] = logit(np.clip(np.average(target, weights=weight), 1e-6, 1.0 - 1e-6))
    for _ in range(100):
        p = expit(x @ beta)
        hessian = (x * (weight * p * (1.0 - p))[:, None]).T @ x
        step = np.linalg.solve(hessian + 1e-12 * np.eye(len(beta)), x.T @ (weight * (target - p)))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-12:
            return beta
    raise RuntimeError("the logistic fit did not converge")


class PerturbedLogistic(BaseEstimator, ClassifierMixin):
    """A logistic GLM on ``columns`` plus a fixed logit offset on one column, at prediction.

    ``columns=None`` reads every design column and ``()`` none (the intercept-only fit).  The
    fit ignores the offset, so its coefficients converge to the GLM limit; the offset
    ``offset * (x[:, offset_column] - centre)`` is added to the logit at prediction.  A zero
    offset is the plain GLM.  A nonzero offset is a deliberate constructed error, not a fitted
    learner.

    Parameters
    ----------
    columns : tuple of int or None
        The design columns the GLM reads.
    offset_column : int
        The design column the offset multiplies.
    offset : float
        The offset's coefficient.
    centre : float
        The value subtracted from the offset column.
    """

    def __init__(
        self,
        columns: tuple[int, ...] | None = None,
        offset_column: int = INSTRUMENT_COLUMN,
        offset: float = 0.0,
        centre: float = INSTRUMENT_CENTRE,
    ) -> None:
        self.columns = columns
        self.offset_column = offset_column
        self.offset = offset
        self.centre = centre

    def _design(self, design: Any) -> np.ndarray:
        values = np.asarray(design, dtype=float)
        if self.columns is None:
            return values
        return values[:, list(self.columns)]

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> PerturbedLogistic:
        y = np.asarray(target, dtype=float)
        weight = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        self.beta_ = newton_logistic(self._design(design), y, weight)
        self.classes_ = np.array([0.0, 1.0])
        return self

    def decision_function(self, design: Any) -> np.ndarray:
        values = np.asarray(design, dtype=float)
        eta = self.beta_[0] + self._design(values) @ self.beta_[1:]
        if self.offset != 0.0:
            eta = eta + self.offset * (values[:, self.offset_column] - self.centre)
        return np.asarray(eta)

    def predict_proba(self, design: Any) -> np.ndarray:
        p = expit(self.decision_function(design))
        return np.column_stack([1.0 - p, p])


def slow_offset(n: int, k: float = SLOW_OFFSET) -> float:
    """``k n^(-1/4)``, the coefficient of the constructed error at size ``n``."""
    return float(k * n ** (-0.25))


# --------------------------------------------------------------------------- the settings
@dataclass(frozen=True)
class Setting:
    """One fit configuration of the study.

    Parameters
    ----------
    law : str
        ``"LDA"`` or ``"LDB"``.
    outcome : str
        ``"glm"`` (logistic GLM on ``A, W``), ``"intercept"`` (intercept only) or ``"slow"``
        (the GLM plus the constructed error ``k n^(-1/4) Z``).
    mechanism : str
        ``"subset"`` (logistic GLM on ``W_g``, declared by ``treatment_covariates``),
        ``"intercept"`` (intercept only, with the declaration) or ``"full"`` (logistic GLM on
        the whole ``W``, without the declaration).
    cross_fit : bool
        Whether the nuisances are cross-fitted over the shipped default folds.
    """

    law: str
    outcome: str
    mechanism: str
    cross_fit: bool = False

    @property
    def declares(self) -> bool:
        """Whether the fit declares ``treatment_adjustment``."""
        return self.mechanism != "full"


SETTINGS: dict[str, Setting] = {
    "lda_in_sample": Setting("LDA", "glm", "subset"),
    "lda_stacked": Setting("LDA", "glm", "subset", cross_fit=True),
    "lda_q_intercept": Setting("LDA", "intercept", "subset"),
    "lda_both_intercept": Setting("LDA", "intercept", "intercept"),
    "ldb_slow": Setting("LDB", "slow", "subset"),
    "ldb_slow_full": Setting("LDB", "slow", "full"),
    "ldb_parametric": Setting("LDB", "glm", "subset"),
}


def outcome_learner(setting: Setting, n: int, k: float = SLOW_OFFSET) -> PerturbedLogistic:
    """The outcome learner on the design ``(A, W1, W2, Z)``."""
    if setting.outcome == "intercept":
        return PerturbedLogistic(columns=())
    if setting.outcome == "slow":
        return PerturbedLogistic(offset=slow_offset(n, k))
    return PerturbedLogistic()


def mechanism_learner(setting: Setting) -> PerturbedLogistic:
    """The treatment learner; it reads ``W_g`` under the declaration and ``W`` without it."""
    if setting.mechanism == "intercept":
        return PerturbedLogistic(columns=())
    return PerturbedLogistic()


def estimator(setting: Setting, n: int, *, k: float = SLOW_OFFSET, **overrides: Any) -> Any:
    """The ``TMLE`` of one setting at sample size ``n``."""
    from cleverly.estimators import TMLE

    options: dict[str, Any] = {
        "outcome_learner": outcome_learner(setting, n, k),
        "treatment_learner": mechanism_learner(setting),
        "cross_fit": setting.cross_fit,
        "estimands": ESTIMANDS,
        "simultaneous": False,
        "g_bounds": G_BOUNDS,
        "max_iter": MAX_ITER,
        "tol": TOL,
        "random_state": 0,
    }
    options.update(overrides)
    return TMLE(**options)


def fit(frame: pd.DataFrame, name: str, *, k: float = SLOW_OFFSET, **overrides: Any) -> Any:
    """Fit the named setting on one sample."""
    setting = SETTINGS[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            estimator(setting, len(frame), k=k, **overrides)
            .fit(
                frame,
                outcome="Y",
                treatment="A",
                covariates=list(COVARIATES),
                treatment_covariates=list(SUBSET) if setting.declares else None,
            )
            .single()
        )


# ------------------------------------------------------------------------ exact-law limits
def mechanism_limit(law: str, names: tuple[str, ...], points: Points | None = None) -> np.ndarray:
    """The population logistic fit of ``A`` on the named covariates, under ``law``."""
    at = grid() if points is None else points
    return newton_logistic(at.columns(names), g0(law, at.w1, at.w2, at.z), at.weight)


def mechanism_at(beta: np.ndarray, names: tuple[str, ...], points: Points) -> np.ndarray:
    """A logistic mechanism with coefficients ``beta`` at ``points``, bounded by the study."""
    eta = beta[0] + points.columns(names) @ beta[1:]
    return np.clip(expit(eta), *G_BOUNDS)


def solve_fluctuation(
    points: Points,
    pi: np.ndarray,
    clever: np.ndarray,
    q0: np.ndarray,
    initial: np.ndarray,
    fluctuation: str,
) -> float:
    """The population fluctuation coefficient of one arm.

    It solves ``E[pi H (Q_0 - Q*(eps))] = 0`` with ``Q*(eps) = expit(logit Q + eps H)`` under
    the logistic fluctuation and ``Q + eps H`` under the linear one.
    """

    def targeted(eps: float) -> np.ndarray:
        if fluctuation == "linear":
            return initial + eps * clever
        return np.asarray(expit(logit(initial) + eps * clever))

    def score(eps: float) -> float:
        return points.mean(pi * clever * (q0 - targeted(eps)))

    if fluctuation == "linear":
        return float(points.mean(pi * clever * (q0 - initial)) / points.mean(pi * clever**2))
    return float(brentq(score, -50.0, 50.0, xtol=1e-15, rtol=1e-15, maxiter=500))


@dataclass(frozen=True)
class TargetedLimit:
    """The population targeting step of both arms.

    Parameters
    ----------
    epsilon : dict
        Each arm's fluctuation coefficient.
    means : dict
        Each arm's targeted mean, ``E[Q*_a]``.
    remainder : dict
        Each arm's ``E[(1 - pi_a / g_a)(Q*_a - Q_0a)]``, which equals the arm's error.
    """

    epsilon: dict[float, float]
    means: dict[float, float]
    remainder: dict[float, float]

    @property
    def ate(self) -> float:
        """The targeted ATE."""
        return self.means[1.0] - self.means[0.0]

    @property
    def ate_remainder(self) -> float:
        """The ATE remainder."""
        return self.remainder[1.0] - self.remainder[0.0]


def targeted_limit(
    points: Points,
    *,
    g_true: np.ndarray,
    g_fit: np.ndarray,
    q0: Callable[[float], np.ndarray],
    initial: Callable[[float], np.ndarray],
    fluctuation: str = "logistic",
) -> TargetedLimit:
    """Solve the population targeting step on a weighted finite set of points.

    ``g_true`` is the true ``P(A = 1 | W)``, ``g_fit`` the mechanism the fit uses and
    ``q0(a)`` and ``initial(a)`` the true and initial outcome regressions at arm ``a``.  The
    clever covariate of arm ``a`` is ``1 / g_a``, so the targeted regression of arm 1 is
    ``expit(logit Q(1, W) + eps_1 / g(W))`` under the logistic fluctuation.
    """
    epsilon: dict[float, float] = {}
    means: dict[float, float] = {}
    remainder: dict[float, float] = {}
    for arm in (1.0, 0.0):
        pi = g_true if arm == 1.0 else 1.0 - g_true
        fitted = g_fit if arm == 1.0 else 1.0 - g_fit
        clever = 1.0 / fitted
        start = initial(arm)
        eps = solve_fluctuation(points, pi, clever, q0(arm), start, fluctuation)
        star = (
            start + eps * clever if fluctuation == "linear" else expit(logit(start) + eps * clever)
        )
        epsilon[arm] = eps
        means[arm] = points.mean(star)
        remainder[arm] = points.mean((1.0 - pi / fitted) * (star - q0(arm)))
    return TargetedLimit(epsilon, means, remainder)


def setting_limit(
    name: str, n: int, *, k: float = SLOW_OFFSET, points: Points | None = None
) -> TargetedLimit:
    """The exact-law targeting limit of a setting at size ``n``, on the Gauss-Hermite grid.

    The outcome GLM is correct for both laws, so its limit is ``Q_0``; the constructed error
    adds its offset at ``n``.  The mechanism is the population logistic fit on the covariates
    the setting reads.
    """
    setting = SETTINGS[name]
    at = grid() if points is None else points
    law = setting.law
    names = SUBSET if setting.mechanism == "subset" else COVARIATES
    if setting.outcome == "intercept" or setting.mechanism == "intercept":
        raise ValueError(f"{name} has no exact-law limit here: an intercept fit is not a GLM limit")
    g_fit = mechanism_at(mechanism_limit(law, names, at), names, at)
    offset = slow_offset(n, k) if setting.outcome == "slow" else 0.0

    def q0(arm: float) -> np.ndarray:
        return np.asarray(qbar0(law, arm, at.w1, at.w2, at.z))

    def initial(arm: float) -> np.ndarray:
        return np.asarray(expit(logit(q0(arm)) + offset * (at.z - INSTRUMENT_CENTRE)))

    return targeted_limit(
        at, g_true=np.asarray(g0(law, at.w1, at.w2, at.z)), g_fit=g_fit, q0=q0, initial=initial
    )


@dataclass(frozen=True)
class StackedVariance:
    """The asymptotic variance of the ATE and the limit of its plug-in variance.

    Parameters
    ----------
    true : float
        ``n Var(psi_hat)`` from the stacked estimating equations.
    plugin : float
        The limit of the reported ``n se^2``, the variance of the efficient influence curve at
        the fit's limits.
    residual : float
        ``max |E[U(theta_0)]|``, the largest population mean of the stacked equations at the
        limits.  It is zero when the equations are the estimator's own.
    """

    true: float
    plugin: float
    residual: float

    @property
    def se_ratio(self) -> float:
        """The limit of the reported SE over the true SD."""
        return float(np.sqrt(self.plugin / self.true))

    def sd(self, n: int) -> float:
        """The estimator's SD at size ``n``."""
        return float(np.sqrt(self.true / n))


def stacked_variance(
    name: str, n: int | None = None, *, k: float = SLOW_OFFSET, points: Points | None = None
) -> StackedVariance:
    """The asymptotic variance of a GLM setting's ATE, from its stacked estimating equations.

    The parameter is ``theta = (beta, alpha, eps_1, eps_0, psi_1, psi_0)``: the outcome GLM on
    ``(1, A, W)``, the mechanism GLM on the covariates the setting reads, the two fluctuation
    coefficients and the two targeted arm means.  The equations are the GLM scores, the
    fluctuation scores ``I(A = a) H_a (Y - Q*_a)`` and ``Q*_a(W) - psi_a``.  The influence
    function is ``-M^{-1} U`` with ``M = E[dU/dtheta]``; ``M`` is the central difference of the
    population mean of ``U`` (step ``1e-6``) and ``E[U U']`` sums over ``A`` and ``Y``
    exactly.  A constructed error enters at its offset at ``n`` and is not estimated: the
    outcome GLM's score omits it, as :class:`PerturbedLogistic` fits its coefficients without
    the offset, and the offset enters only the predictions that the targeting starts from.
    """
    setting = SETTINGS[name]
    if setting.outcome == "intercept" or setting.mechanism == "intercept":
        raise ValueError(f"{name} is not a GLM setting")
    at = grid() if points is None else points
    law = setting.law
    names = SUBSET if setting.mechanism == "subset" else COVARIATES
    if setting.outcome == "slow" and n is None:
        raise ValueError("a constructed error needs its sample size")
    offset = slow_offset(n, k) if setting.outcome == "slow" and n is not None else 0.0
    beta0 = np.array(OUTCOME[law])
    alpha0 = mechanism_limit(law, names, at)
    limit = setting_limit(name, 1 if n is None else n, k=k, points=at)
    theta0 = np.concatenate(
        [
            beta0,
            alpha0,
            [limit.epsilon[1.0], limit.epsilon[0.0], limit.means[1.0], limit.means[0.0]],
        ]
    )
    n_beta, n_alpha = len(beta0), len(alpha0)
    covariates = np.column_stack([at.w1, at.w2, at.z])
    v = np.column_stack([np.ones(len(at.weight)), at.columns(names)])
    g_true = np.asarray(g0(law, at.w1, at.w2, at.z))

    def pieces(theta: np.ndarray, a: float) -> tuple[Any, ...]:
        beta = theta[:n_beta]
        alpha = theta[n_beta : n_beta + n_alpha]
        eps1, eps0, psi1, psi0 = theta[n_beta + n_alpha :]
        x = np.column_stack([np.ones(len(at.weight)), np.full(len(at.weight), a), covariates])
        g = np.clip(expit(v @ alpha), *G_BOUNDS)
        q = expit(x @ beta)
        arm_stars = {}
        for arm, eps in ((1.0, eps1), (0.0, eps0)):
            xa = x.copy()
            xa[:, 1] = arm
            fitted = g if arm == 1.0 else 1.0 - g
            base = expit(xa @ beta + offset * (at.z - INSTRUMENT_CENTRE))
            arm_stars[arm] = (expit(logit(base) + eps / fitted), 1.0 / fitted)
        return x, g, q, arm_stars[1.0], arm_stars[0.0], np.array([psi1, psi0])

    def score(theta: np.ndarray, a: float, y: float) -> np.ndarray:
        x, g, q, (star1, h1), (star0, h0), psi = pieces(theta, a)
        observed_star = star1 if a == 1.0 else star0
        clever = h1 if a == 1.0 else h0
        columns = [
            x * (y - q)[:, None],
            v * (a - g)[:, None],
            ((a == 1.0) * clever * (y - observed_star))[:, None],
            ((a == 0.0) * clever * (y - observed_star))[:, None],
            (star1 - psi[0])[:, None],
            (star0 - psi[1])[:, None],
        ]
        return np.hstack(columns)

    def probabilities(a: float, y: float) -> np.ndarray:
        q = np.asarray(qbar0(law, a, at.w1, at.w2, at.z))
        pa = g_true if a == 1.0 else 1.0 - g_true
        return at.weight * pa * (q if y == 1.0 else 1.0 - q)

    cells = [(a, y) for a in (0.0, 1.0) for y in (0.0, 1.0)]

    def mean_score(theta: np.ndarray) -> np.ndarray:
        return sum(probabilities(a, y) @ score(theta, a, y) for a, y in cells)

    step = 1e-6
    jacobian = np.empty((len(theta0), len(theta0)))
    for j in range(len(theta0)):
        up, down = theta0.copy(), theta0.copy()
        up[j] += step
        down[j] -= step
        jacobian[:, j] = (mean_score(up) - mean_score(down)) / (2.0 * step)
    meat = sum(
        (score(theta0, a, y) * probabilities(a, y)[:, None]).T @ score(theta0, a, y)
        for a, y in cells
    )
    inverse = np.linalg.inv(jacobian)
    covariance = inverse @ meat @ inverse.T
    contrast = np.zeros(len(theta0))
    contrast[-2], contrast[-1] = 1.0, -1.0
    true = float(contrast @ covariance @ contrast)

    # The reported variance: the efficient influence curve at the fit's limits, with the
    # targeted regressions in place of Q.
    plugin = 0.0
    ate = limit.ate
    for a, y in cells:
        _x, _g, _q, (star1, h1), (star0, h0), _psi = pieces(theta0, a)
        residual = y - (star1 if a == 1.0 else star0)
        weight = h1 if a == 1.0 else -h0
        curve = weight * residual + star1 - star0 - ate
        plugin += float(probabilities(a, y) @ curve**2)
    residual = float(np.max(np.abs(mean_score(theta0))))
    return StackedVariance(true=true, plugin=plugin, residual=residual)


def efficiency_sd(law: str = "LDA", points: Points | None = None) -> float:
    """``sqrt(E[D*^2])`` of the ATE under ``law``, the nonparametric efficient influence curve.

    Under the declared restriction ``D*`` lies in the restricted tangent space, so this is the
    efficiency bound of the restricted model too (Fix-8 plan section 2.4).
    """
    at = grid() if points is None else points
    g = np.asarray(g0(law, at.w1, at.w2, at.z))
    one = np.asarray(qbar0(law, 1.0, at.w1, at.w2, at.z))
    zero = np.asarray(qbar0(law, 0.0, at.w1, at.w2, at.z))
    ate = truth(law)["ate"]
    second = g * one * (1.0 - one) / g**2 + (1.0 - g) * zero * (1.0 - zero) / (1.0 - g) ** 2
    return float(np.sqrt(at.mean(second + (one - zero - ate) ** 2)))


def bias_over_sd(name: str, n: int, *, k: float = SLOW_OFFSET) -> float:
    """The exact-law ATE bias of a setting at ``n``, in units of its SD at ``n``."""
    bias = setting_limit(name, n, k=k).ate - truth(SETTINGS[name].law)["ate"]
    return bias / stacked_variance(name, n, k=k).sd(n)
