r"""Laws, learners, fits and exact-law limits of ``option-outcome-by-arm`` (X76 part (b)).

``outcome_by_arm=True`` fits one outcome regression per treatment arm, on the arm's rows and the
covariates alone.  It changes the outcome model class and nothing else: the remainder is the
ordinary product :math:`P_0[(g_0 - g_n)/g_n (Q_0 - Q_n)]`, and double robustness is unchanged
(Fix-8 plan section 2.2).  The option helps where a pooled regression misses an arm-by-covariate
interaction, and the study's laws put one there.

Three laws share :math:`W_1, W_2` iid standard normal, the mechanism
:math:`g_0 = \operatorname{expit}(0.4 W_1 + 0.4 W_2)` and the arm indices
:math:`\eta_1 = -0.2 + 0.9 W_1 + 0.2 W_2` and :math:`\eta_0 = -0.5 + 0.1 W_1 + 0.7 W_2`
(Fix-8 plan section 5.3):

=======  =============================================================  ======================
law      outcome given :math:`A = a, W`                                 per-arm regression
=======  =============================================================  ======================
``LB``   Bernoulli(:math:`\operatorname{expit} \eta_a`)                 logistic GLM on W
``LB-c`` :math:`\eta_a + N(0, 1)`                                       linear regression on W
``LB-b`` Beta(:math:`24 m, 24 (1 - m)`), :math:`m = \operatorname{expit}  fractional logistic
         \eta_a`, in (0, 1)                                             GLM on W
=======  =============================================================  ======================

Each per-arm regression is correct by construction, and a pooled main-terms regression on
:math:`(A, W)` is not.  ``LB-b`` is the bounded law of the stacked continuous cell: the ``T02``
key holds a continuous outcome with a declared support (``q_bounds``), and an outcome with
normal noise has none.  It keeps ``LB``'s means, so the two laws have one truth.

**Exact-law limits.**  :func:`outcome_limit` and :func:`mechanism_limit` give the population
limit of each learner; a pooled GLM converges to the population logistic (or least-squares)
projection.  :func:`targeted_limit` solves the population targeting step on a weighted finite
set of covariate points, and :func:`stacked_variance` gives the asymptotic variance of the ATE
from the stacked estimating equations of the outcome GLM, the mechanism GLM, the fluctuation and
the arm means, with the limit of the plug-in variance the package reports.  Every equation is
affine in ``Y``, so the first two conditional moments of ``Y`` give the means and the meat
exactly.  The limits cover the laws whose outcome scale is fixed (``LB`` and ``LB-b``); the
fluctuation of an unbounded outcome runs on the sample range, which has no fixed limit.
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
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LinearRegression

COVARIATES = ("W1", "W2")
#: ``(intercept, W1, W2)`` of the treatment logit.  Plan 5.3 wrote ``0.4 W1 - 0.4 W2``; under
#: that sign the treated arm's excess of ``W1`` and the control arm's excess of ``W2`` raise both
#: arm means alike, and the crude difference misses the ATE by 0.16 SD at n = 2,000, below the
#: control floor.  With ``+0.4 W2`` the confounding adds, and the crude difference misses by
#: 3.4 SD (the design test computes both).
TREATMENT = (0.0, 0.4, 0.4)
#: ``(intercept, W1, W2)`` of each arm's outcome index.
ARM_INDEX = {1.0: (-0.2, 0.9, 0.2), 0.0: (-0.5, 0.1, 0.7)}
#: The Beta concentration of ``LB-b``: ``Var(Y | A, W) = m (1 - m) / (1 + 24)``.
CONCENTRATION = 24.0
#: The noise SD of ``LB-c``.
NOISE_SD = 1.0
LAWS = ("LB", "LB-c", "LB-b")

#: The fit settings: ``canonical-tmle``'s treatment bounds, plan section 5.0's ``max_iter`` and
#: ``tol``, the stacked cells' five folds and ``LB-b``'s declared outcome support.
G_BOUNDS = (0.01, 0.99)
MAX_ITER = 100
TOL = 1e-10
N_FOLDS = 5
Q_BOUNDS = (0.0, 1.0)
ESTIMANDS = ("ey1", "ey0", "ate")
#: The least exact-law limit of a control, in SDs (plan 5.0).
CONTROL_FLOOR_SD = 0.45

#: Gauss-Hermite nodes per axis of the truths (plan 5.3: at least 80) and of the limits.
TRUTH_NODES = 80
GRID_NODES = 40


# ------------------------------------------------------------------------------- the laws
def g0(w1: Any, w2: Any) -> Any:
    """``P(A = 1 | W)``."""
    c0, c1, c2 = TREATMENT
    return expit(c0 + c1 * np.asarray(w1) + c2 * np.asarray(w2))


def index(a: float, w1: Any, w2: Any) -> Any:
    """The outcome index of arm ``a``."""
    b0, b1, b2 = ARM_INDEX[float(a)]
    return b0 + b1 * np.asarray(w1) + b2 * np.asarray(w2)


def qbar0(law: str, a: float, w1: Any, w2: Any) -> Any:
    """``E[Y | A = a, W]`` of ``law``."""
    eta = index(a, w1, w2)
    return eta if law == "LB-c" else expit(eta)


def second_moment(law: str, a: float, w1: Any, w2: Any) -> Any:
    """``E[Y^2 | A = a, W]`` of ``law``."""
    m = qbar0(law, a, w1, w2)
    if law == "LB":
        return m
    if law == "LB-c":
        return NOISE_SD**2 + m**2
    return m * (1.0 - m) / (1.0 + CONCENTRATION) + m**2


def draw(law: str, n: int, seed: int) -> pd.DataFrame:
    """One sample of ``law``: columns ``Y, A, W1, W2``."""
    if law not in LAWS:
        raise KeyError(law)
    rng = np.random.default_rng(seed)
    w1, w2 = rng.standard_normal((2, n))
    a = rng.binomial(1, g0(w1, w2)).astype(float)
    m = np.where(a == 1.0, qbar0(law, 1.0, w1, w2), qbar0(law, 0.0, w1, w2))
    if law == "LB":
        y = rng.binomial(1, m).astype(float)
    elif law == "LB-c":
        y = m + NOISE_SD * rng.standard_normal(n)
    else:
        y = rng.beta(CONCENTRATION * m, CONCENTRATION * (1.0 - m))
    return pd.DataFrame({"Y": y, "A": a, "W1": w1, "W2": w2})


def gauss_hermite(nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Standard-normal Gauss-Hermite nodes and weights."""
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    return x, w / np.sqrt(2.0 * np.pi)


@dataclass(frozen=True)
class Points:
    """A weighted finite set of covariate points: a quadrature grid or an exact finite law.

    Parameters
    ----------
    w1 : numpy.ndarray
        ``W1`` at each point.
    w2 : numpy.ndarray
        ``W2`` at each point.
    weight : numpy.ndarray
        The probability of each point; the weights sum to one.
    """

    w1: np.ndarray
    w2: np.ndarray
    weight: np.ndarray

    def mean(self, values: Any) -> float:
        """The weighted mean of ``values`` over the points."""
        return float(np.sum(self.weight * np.asarray(values, dtype=float)))

    def design(self) -> np.ndarray:
        """The covariates as a design matrix, without an intercept."""
        return np.column_stack([self.w1, self.w2])


def grid(nodes: int = GRID_NODES) -> Points:
    """The 2-D Gauss-Hermite grid of ``(W1, W2)``."""
    x, w = gauss_hermite(nodes)
    w1, w2 = (axis.reshape(-1) for axis in np.meshgrid(x, x, indexing="ij"))
    return Points(w1, w2, np.outer(w, w).reshape(-1))


def truth(law: str, nodes: int = TRUTH_NODES) -> dict[str, float]:
    """The arm means and the ATE of ``law``, on the 2-D Gauss-Hermite grid of ``nodes``.

    ``LB-c`` is linear in ``W``, so its truth is the difference of the intercepts; the grid
    reproduces it.  ``LB-b`` has ``LB``'s means.
    """
    at = grid(nodes)
    one = at.mean(qbar0(law, 1.0, at.w1, at.w2))
    zero = at.mean(qbar0(law, 0.0, at.w1, at.w2))
    return {"ey1": one, "ey0": zero, "ate": one - zero}


def efficiency_sd(law: str, points: Points | None = None) -> float:
    """``sqrt(E[D*^2])`` of the ATE under ``law``, the nonparametric efficiency bound."""
    at = grid() if points is None else points
    g = g0(at.w1, at.w2)
    one, zero = qbar0(law, 1.0, at.w1, at.w2), qbar0(law, 0.0, at.w1, at.w2)
    var1 = second_moment(law, 1.0, at.w1, at.w2) - one**2
    var0 = second_moment(law, 0.0, at.w1, at.w2) - zero**2
    ate = at.mean(one - zero)
    return float(np.sqrt(at.mean(var1 / g + var0 / (1.0 - g) + (one - zero - ate) ** 2)))


# ---------------------------------------------------------------------------- the learners
def newton_logistic(design: np.ndarray, target: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Weighted logistic quasi-likelihood with an intercept, by Newton, ridge ``1e-12``.

    ``target`` may be a proportion or a probability: a fractional outcome and a weighted finite
    law that passes ``Q_0`` itself solve the same score equation.
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


class _NewtonGLM(BaseEstimator):
    """A logistic GLM with an intercept on ``columns`` of the design, fitted by Newton.

    Parameters
    ----------
    columns : tuple of int or None
        The design columns the GLM reads; ``None`` reads all and ``()`` none.
    """

    def __init__(self, columns: tuple[int, ...] | None = None) -> None:
        self.columns = columns

    def _design(self, design: Any) -> np.ndarray:
        values = np.asarray(design, dtype=float)
        return values if self.columns is None else values[:, list(self.columns)]

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> Any:
        y = np.asarray(target, dtype=float)
        weight = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        self.beta_ = newton_logistic(self._design(design), y, weight)
        self.n_features_in_ = np.asarray(design).shape[1]
        return self

    def _mean(self, design: Any) -> np.ndarray:
        return np.asarray(expit(self.beta_[0] + self._design(design) @ self.beta_[1:]))


class Logistic(_NewtonGLM, ClassifierMixin):
    """The study's logistic classifier: maximum likelihood by Newton, without a penalty.

    Parameters
    ----------
    columns : tuple of int or None
        The design columns the GLM reads; ``None`` reads all and ``()`` none.
    """

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> Logistic:
        super().fit(design, target, sample_weight)
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, design: Any) -> np.ndarray:
        p = self._mean(design)
        return np.column_stack([1.0 - p, p])


class FractionalLogit(_NewtonGLM, RegressorMixin):
    """A logistic-mean regression of an outcome in (0, 1), by quasi-likelihood.

    Parameters
    ----------
    columns : tuple of int or None
        The design columns the GLM reads; ``None`` reads all and ``()`` none.
    """

    def predict(self, design: Any) -> np.ndarray:
        return self._mean(design)


# --------------------------------------------------------------------------- the settings
@dataclass(frozen=True)
class Setting:
    """One fit configuration of the study.

    Parameters
    ----------
    law : str
        ``"LB"``, ``"LB-c"`` or ``"LB-b"``.
    outcome : str
        ``"by_arm"`` (one main-terms regression per arm on ``W``, ``outcome_by_arm=True``),
        ``"by_arm_intercept"`` (one intercept per arm, ``outcome_by_arm=True``) or ``"pooled"``
        (one main-terms regression on ``(A, W)``, ``outcome_by_arm=False``).
    mechanism : str
        ``"glm"`` (the correct logistic GLM on ``W``) or ``"intercept"`` (intercept only).
    cross_fit : bool
        Whether the nuisances are cross-fitted over :data:`N_FOLDS` folds (stacked).
    """

    law: str
    outcome: str
    mechanism: str = "glm"
    cross_fit: bool = False

    @property
    def by_arm(self) -> bool:
        """Whether the fit declares ``outcome_by_arm=True``."""
        return self.outcome != "pooled"


SETTINGS: dict[str, Setting] = {
    "binary_in_sample": Setting("LB", "by_arm"),
    "binary_stacked": Setting("LB", "by_arm", cross_fit=True),
    "continuous_in_sample": Setting("LB-c", "by_arm"),
    "bounded_stacked": Setting("LB-b", "by_arm", cross_fit=True),
    "pooled_binary_in_sample": Setting("LB", "pooled"),
    "pooled_binary_stacked": Setting("LB", "pooled", cross_fit=True),
    "pooled_continuous_in_sample": Setting("LB-c", "pooled"),
    "pooled_bounded_stacked": Setting("LB-b", "pooled", cross_fit=True),
    "q_intercept": Setting("LB", "by_arm_intercept"),
    "both_intercept": Setting("LB", "by_arm_intercept", "intercept"),
}


def outcome_learner(setting: Setting) -> Any:
    """The outcome learner: a per-arm learner reads ``W``, a pooled one ``(A, W)``."""
    columns = () if setting.outcome == "by_arm_intercept" else None
    if setting.law == "LB":
        return Logistic(columns=columns)
    if setting.law == "LB-b":
        return FractionalLogit(columns=columns)
    if columns is not None:
        raise ValueError("the continuous law has no intercept-only cell")
    return LinearRegression()


def mechanism_learner(setting: Setting) -> Logistic:
    """The treatment learner on ``W``."""
    return Logistic(columns=() if setting.mechanism == "intercept" else None)


def estimator(setting: Setting, **overrides: Any) -> Any:
    """The ``TMLE`` of one setting."""
    from cleverly.estimators import TMLE

    options: dict[str, Any] = {
        "outcome_learner": outcome_learner(setting),
        "treatment_learner": mechanism_learner(setting),
        "outcome_by_arm": setting.by_arm,
        "cross_fit": setting.cross_fit,
        "estimands": ESTIMANDS,
        "simultaneous": False,
        "g_bounds": G_BOUNDS,
        "max_iter": MAX_ITER,
        "tol": TOL,
        "random_state": 0,
    }
    if setting.cross_fit:
        options["n_folds"] = N_FOLDS
    if setting.law == "LB-b":
        options["q_bounds"] = Q_BOUNDS
    options.update(overrides)
    return TMLE(**options)


def fit(frame: pd.DataFrame, name: str, **overrides: Any) -> Any:
    """Fit the named setting on one sample."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            estimator(SETTINGS[name], **overrides)
            .fit(frame, outcome="Y", treatment="A", covariates=list(COVARIATES))
            .single()
        )


# ------------------------------------------------------------------------ exact-law limits
def _arm_points(points: Points) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(design (1, A, W), weight, arm)`` of the joint law of ``(A, W)`` over ``points``."""
    g = g0(points.w1, points.w2)
    rows = len(points.weight)
    design = np.vstack(
        [
            np.column_stack([np.ones(rows), points.w1, points.w2]),
            np.column_stack([np.zeros(rows), points.w1, points.w2]),
        ]
    )
    weight = np.concatenate([points.weight * g, points.weight * (1.0 - g)])
    return design, weight, design[:, 0]


def outcome_coefficients(name: str, points: Points | None = None) -> np.ndarray:
    """The population limit of a setting's outcome-GLM coefficients.

    A per-arm fit returns ``(beta_1, beta_0)``, each on ``(1, W)`` (or the intercept alone); a
    pooled fit returns one vector on ``(1, A, W)``.  Each is the root of the GLM score under the
    law, so a correct model returns the law's own coefficients.
    """
    setting = SETTINGS[name]
    at = grid() if points is None else points
    law = setting.law
    design, weight, arm = _arm_points(at)
    target = np.concatenate([qbar0(law, 1.0, at.w1, at.w2), qbar0(law, 0.0, at.w1, at.w2)])
    if setting.outcome == "pooled":
        return newton_logistic(design, target, weight)
    columns = design[:, 1:] if setting.outcome == "by_arm" else design[:, :0]
    pieces = []
    for value in (1.0, 0.0):
        mask = arm == value
        pieces.append(newton_logistic(columns[mask], target[mask], weight[mask]))
    return np.concatenate(pieces)


def outcome_limit(name: str, points: Points | None = None) -> Callable[[float], np.ndarray]:
    """``a -> Q_inf(a, W)`` at ``points``, the limit of a setting's outcome fit."""
    setting = SETTINGS[name]
    at = grid() if points is None else points
    if setting.law == "LB-c":
        raise ValueError("the continuous law's targeting runs on the sample range")
    beta = outcome_coefficients(name, at)
    width = len(beta) // 2

    def q(a: float) -> np.ndarray:
        if setting.outcome == "pooled":
            return np.asarray(expit(beta[0] + beta[1] * a + beta[2] * at.w1 + beta[3] * at.w2))
        b = beta[:width] if a == 1.0 else beta[width:]
        eta = b[0] + (b[1] * at.w1 + b[2] * at.w2 if width == 3 else 0.0)
        return np.asarray(expit(eta) * np.ones(len(at.weight)))

    return q


def mechanism_coefficients(name: str, points: Points | None = None) -> np.ndarray:
    """The population limit of a setting's mechanism GLM, on ``(1, W)`` or the intercept."""
    at = grid() if points is None else points
    design = at.design() if SETTINGS[name].mechanism == "glm" else at.design()[:, :0]
    return newton_logistic(design, g0(at.w1, at.w2), at.weight)


def mechanism_limit(name: str, points: Points | None = None) -> np.ndarray:
    """The limit of a setting's mechanism fit at ``points``, bounded as the fit bounds it."""
    at = grid() if points is None else points
    alpha = mechanism_coefficients(name, at)
    design = at.design() if SETTINGS[name].mechanism == "glm" else at.design()[:, :0]
    return np.clip(expit(alpha[0] + design @ alpha[1:]), *G_BOUNDS)


def solve_fluctuation(
    points: Points, pi: np.ndarray, clever: np.ndarray, q0: np.ndarray, initial: np.ndarray
) -> float:
    """The population root of one arm's logistic fluctuation score.

    It solves ``E[pi H (Q_0 - Q*(eps))] = 0`` with ``Q*(eps) = expit(logit Q + eps H)``.
    """

    def score(eps: float) -> float:
        return points.mean(pi * clever * (q0 - expit(logit(initial) + eps * clever)))

    return float(brentq(score, -50.0, 50.0, xtol=1e-14, rtol=1e-15))


@dataclass(frozen=True)
class TargetedLimit:
    """The population targeting step of one fit.

    Parameters
    ----------
    epsilon : dict
        The fluctuation coefficient of each arm.
    means : dict
        The targeted arm mean of each arm.
    """

    epsilon: dict[float, float]
    means: dict[float, float]

    @property
    def ate(self) -> float:
        """The targeted ATE."""
        return self.means[1.0] - self.means[0.0]


def targeted_limit(
    points: Points,
    *,
    g_true: np.ndarray,
    g_fit: np.ndarray,
    q0: Callable[[float], np.ndarray],
    initial: Callable[[float], np.ndarray],
) -> TargetedLimit:
    """Solve the population targeting step on a weighted finite set of points.

    The clever covariate of arm ``a`` is ``1 / g_a``, and arm ``a``'s targeted regression is
    ``expit(logit Q(a, W) + eps_a / g_a(W))``.
    """
    epsilon: dict[float, float] = {}
    means: dict[float, float] = {}
    for arm in (1.0, 0.0):
        pi = g_true if arm == 1.0 else 1.0 - g_true
        clever = 1.0 / (g_fit if arm == 1.0 else 1.0 - g_fit)
        eps = solve_fluctuation(points, pi, clever, q0(arm), initial(arm))
        epsilon[arm] = eps
        means[arm] = points.mean(expit(logit(initial(arm)) + eps * clever))
    return TargetedLimit(epsilon, means)


def setting_limit(name: str, points: Points | None = None) -> TargetedLimit:
    """The exact-law targeting limit of a setting with a fixed outcome scale."""
    setting = SETTINGS[name]
    at = grid() if points is None else points

    def q0(arm: float) -> np.ndarray:
        return np.asarray(qbar0(setting.law, arm, at.w1, at.w2))

    return targeted_limit(
        at,
        g_true=np.asarray(g0(at.w1, at.w2)),
        g_fit=mechanism_limit(name, at),
        q0=q0,
        initial=outcome_limit(name, at),
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


def stacked_variance(name: str, points: Points | None = None) -> StackedVariance:
    """The asymptotic variance of a setting's ATE, from its stacked estimating equations.

    The parameter is ``theta = (beta, alpha, eps_1, eps_0, psi_1, psi_0)``: the outcome GLM
    (one vector per arm on ``(1, W)`` or the intercept, or one on ``(1, A, W)``), the mechanism
    GLM, the two fluctuation coefficients and the two targeted arm means.  The equations are the
    GLM scores, the fluctuation scores ``I(A = a) H_a (Y - Q*_a)`` and ``Q*_a(W) - psi_a``.  The
    influence function is ``-M^{-1} U`` with ``M = E[dU/dtheta]``, the central difference of
    the population mean of ``U`` (step ``1e-6``).  Each equation is ``U_0 + Y U_1``, so
    ``E[U U']`` reads the first two conditional moments of ``Y`` and is exact.
    """
    setting = SETTINGS[name]
    if setting.law == "LB-c":
        raise ValueError("the continuous law's targeting runs on the sample range")
    at = grid() if points is None else points
    law = setting.law
    beta0 = outcome_coefficients(name, at)
    alpha0 = mechanism_coefficients(name, at)
    limit = setting_limit(name, at)
    theta0 = np.concatenate(
        [
            beta0,
            alpha0,
            [limit.epsilon[1.0], limit.epsilon[0.0], limit.means[1.0], limit.means[0.0]],
        ]
    )
    n_beta, n_alpha = len(beta0), len(alpha0)
    rows = len(at.weight)
    ones = np.ones(rows)
    w_design = np.column_stack([ones, at.w1, at.w2])
    v = w_design if setting.mechanism == "glm" else w_design[:, :1]
    x_arm = w_design if setting.outcome != "by_arm_intercept" else w_design[:, :1]
    g_true = np.asarray(g0(at.w1, at.w2))

    def regression(beta: np.ndarray, arm: float) -> np.ndarray:
        if setting.outcome == "pooled":
            return np.asarray(expit(beta[0] + beta[1] * arm + beta[2] * at.w1 + beta[3] * at.w2))
        width = n_beta // 2
        return np.asarray(expit(x_arm @ (beta[:width] if arm == 1.0 else beta[width:])))

    def score(theta: np.ndarray, a: float) -> tuple[np.ndarray, np.ndarray]:
        """``(U_0, U_1)`` with ``U = U_0 + Y U_1`` at ``A = a``, one row per point."""
        beta = theta[:n_beta]
        alpha = theta[n_beta : n_beta + n_alpha]
        eps1, eps0, psi1, psi0 = theta[n_beta + n_alpha :]
        g = np.clip(expit(v @ alpha), *G_BOUNDS)
        q = regression(beta, a)
        if setting.outcome == "pooled":
            x = np.column_stack([ones, np.full(rows, a), at.w1, at.w2])
            beta_zero, beta_one = -x * q[:, None], x
        else:
            width = x_arm.shape[1]
            x = np.zeros((rows, n_beta))
            block = slice(0, width) if a == 1.0 else slice(width, 2 * width)
            x[:, block] = x_arm
            beta_zero, beta_one = -x * q[:, None], x
        stars, clevers = {}, {}
        for arm, eps in ((1.0, eps1), (0.0, eps0)):
            clevers[arm] = 1.0 / (g if arm == 1.0 else 1.0 - g)
            stars[arm] = expit(logit(regression(beta, arm)) + eps * clevers[arm])
        indicator = {1.0: float(a == 1.0), 0.0: float(a == 0.0)}
        zero_columns = [
            beta_zero,
            v * (a - g)[:, None],
            (-indicator[1.0] * clevers[1.0] * stars[1.0])[:, None],
            (-indicator[0.0] * clevers[0.0] * stars[0.0])[:, None],
            (stars[1.0] - psi1)[:, None],
            (stars[0.0] - psi0)[:, None],
        ]
        one_columns = [
            beta_one,
            np.zeros((rows, n_alpha)),
            (indicator[1.0] * clevers[1.0])[:, None],
            (indicator[0.0] * clevers[0.0])[:, None],
            np.zeros((rows, 2)),
        ]
        return np.hstack(zero_columns), np.hstack(one_columns)

    def arm_weight(a: float) -> np.ndarray:
        return at.weight * (g_true if a == 1.0 else 1.0 - g_true)

    def mean_score(theta: np.ndarray) -> np.ndarray:
        total = np.zeros(len(theta))
        for a in (0.0, 1.0):
            u0, u1 = score(theta, a)
            m = np.asarray(qbar0(law, a, at.w1, at.w2))
            total += arm_weight(a) @ (u0 + m[:, None] * u1)
        return total

    step = 1e-6
    jacobian = np.empty((len(theta0), len(theta0)))
    for j in range(len(theta0)):
        up, down = theta0.copy(), theta0.copy()
        up[j] += step
        down[j] -= step
        jacobian[:, j] = (mean_score(up) - mean_score(down)) / (2.0 * step)
    meat = np.zeros((len(theta0), len(theta0)))
    for a in (0.0, 1.0):
        u0, u1 = score(theta0, a)
        m = np.asarray(qbar0(law, a, at.w1, at.w2))[:, None]
        s2 = np.asarray(second_moment(law, a, at.w1, at.w2))[:, None]
        p = arm_weight(a)[:, None]
        meat += (u0 * p).T @ u0 + (u0 * p * m).T @ u1 + (u1 * p * m).T @ u0 + (u1 * p * s2).T @ u1
    inverse = np.linalg.inv(jacobian)
    covariance = inverse @ meat @ inverse.T
    contrast = np.zeros(len(theta0))
    contrast[-2], contrast[-1] = 1.0, -1.0
    true = float(contrast @ covariance @ contrast)

    # The reported variance: the efficient influence curve at the fit's limits, with the
    # targeted regressions in place of Q, integrated over Y by its first two moments.
    g_fit = mechanism_limit(name, at)
    initial = outcome_limit(name, at)
    stars = {
        arm: expit(logit(initial(arm)) + limit.epsilon[arm] / (g_fit if arm else 1.0 - g_fit))
        for arm in (1.0, 0.0)
    }
    centre = stars[1.0] - stars[0.0] - limit.ate
    plugin = 0.0
    for a in (1.0, 0.0):
        h = 1.0 / g_fit if a == 1.0 else -1.0 / (1.0 - g_fit)
        m = np.asarray(qbar0(law, a, at.w1, at.w2))
        s2 = np.asarray(second_moment(law, a, at.w1, at.w2))
        star = stars[a]
        residual_square = s2 - 2.0 * star * m + star**2
        plugin += float(
            arm_weight(a) @ (h**2 * residual_square + 2.0 * h * (m - star) * centre + centre**2)
        )
    residual = float(np.max(np.abs(mean_score(theta0))))
    return StackedVariance(true=true, plugin=plugin, residual=residual)


def bias_over_sd(name: str, n: int) -> float:
    """The exact-law ATE bias of a setting at ``n``, in units of its SD at ``n``."""
    bias = setting_limit(name).ate - truth(SETTINGS[name].law)["ate"]
    return bias / stacked_variance(name).sd(n)
