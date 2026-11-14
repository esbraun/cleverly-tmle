r"""Laws, learners, fits and exact truths of ``known-score-att`` (X35 plan section 5, rev 2.1).

The subject is ``TMLE(conditional_form="known_score")`` on data that declares its treatment
mechanism: the ATT is the g-weighted plug-in :math:`P_n\{g_0(W)\tau^*(W)\}/P_n g_0(W)` with
the curve :math:`D_{KS}`, and the ATC weights by :math:`1-g_0` (``tests/unit/
test_known_score_att.py``).

**Law S** (the score carries information).  :math:`W_1\sim U(-\sqrt3,\sqrt3)`,
:math:`W_2\sim\mathrm{Bernoulli}(0.5)`, :math:`g_0=\operatorname{expit}(-0.3+W_1+0.5W_2)` and

.. math::

    Y = W_1 + 0.5 W_2 + A\,\tau(W) + \varepsilon,\qquad \tau(W) = 1 + 1.5 W_1,
    \qquad \varepsilon\sim U(-\sqrt3,\sqrt3),

so the noise has unit variance and :math:`Y\in[-5.07, 7.57]`, inside the stacked cells'
declared ``q_bounds=(-8, 8)``.  The correct outcome regression is least squares on
:math:`(1, A, W_1, W_2, AW_1)`; the wrong one is least squares on :math:`(1, A, W_2)`.

**Law S0** (the zero-gain variant) keeps Law S and sets :math:`\tau\equiv1`.  The gain
:math:`E\{g_0(1-g_0)(\tau-\psi)^2\}/p_0^2` is then zero, so the known-score and arm-mean
curves coincide at the truth and the paired gate has nothing to find.

**Law B** is the binary law of :mod:`tests.studies.known_mechanism_law` (two and three arms)
with its wrong main-terms logistic regression on :math:`(A, W_1, W_3)`.  Its three-arm ATT and
ATC truths weight by :math:`g_0(c\mid W)`, the conditioning arm's declared probability
(:func:`law_b_truth`).

Every truth is a quadrature: Gauss-Legendre with :data:`LEGENDRE_NODES` nodes on :math:`W_1`,
summed over :math:`W_2` (Law S), and the Law B grid of
:func:`tests.studies.known_mechanism_law.truth`.  :func:`efficiency_sd` gives the exact SD of
:math:`D_{KS}` and of the shipped :math:`D_{AM}` at the truth (plan section 2.2, with
:math:`\sigma^2=1`), and :func:`control_bias` the population bias of the control that
declares the constant :data:`CONTROL_CONSTANT`.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LinearRegression, LogisticRegression

from tests.studies import known_mechanism_law as law_b

ROOT3 = float(np.sqrt(3.0))
COVARIATES = ("W1", "W2")
#: ``(intercept, W1, W2)`` of the Law S treatment logit.
TREATMENT = (-0.3, 1.0, 0.5)
#: ``(intercept, W1, W2)`` of the Law S outcome under control.
BASELINE = (0.0, 1.0, 0.5)
#: ``(intercept, W1)`` of the effect ``tau(W)`` of each law.
EFFECT = {"S": (1.0, 1.5), "S0": (1.0, 0.0)}
LAWS = tuple(EFFECT)
#: The noise half-width: ``U(-sqrt 3, sqrt 3)`` has unit variance.
NOISE_HALF_WIDTH = ROOT3
NOISE_VARIANCE = 1.0

#: The stacked cells' folds and declared outcome support (plan section 5, rev 2 finding 11).
N_FOLDS = 5
Q_BOUNDS = (-8.0, 8.0)
MAX_ITER = 100
TOL = 1e-10
#: The control's declared constant mechanism: the population ``P(A = 1)`` of Law S to two
#: places (:func:`truth` gives 0.489987).  Any constant makes the known-score plug-in tend to
#: the ATE, so the control misses the ATT by ``ATE - ATT``.
CONTROL_CONSTANT = 0.49

#: Gauss-Legendre nodes on ``W1``.
LEGENDRE_NODES = 200

#: The Law B treatment columns, by arm count.
LAW_B_PROBABILITIES = {2: {0.0: "p0", 1.0: "p1"}, 3: {0.0: "p0", 1.0: "p1", 2.0: "p2"}}
LAW_B_COVARIATES = ("W1", "W2", "W3")
#: The Law S treatment columns.
PROBABILITIES = {0.0: "p0", 1.0: "p1"}


# ------------------------------------------------------------------------------- Law S
def g0(w1: Any, w2: Any) -> Any:
    """``P(A = 1 | W)`` of Law S."""
    c0, c1, c2 = TREATMENT
    return expit(c0 + c1 * np.asarray(w1, dtype=float) + c2 * np.asarray(w2, dtype=float))


def tau(name: str, w1: Any) -> Any:
    """The effect ``Qbar(1, W) - Qbar(0, W)`` of a Law S variant."""
    b0, b1 = EFFECT[name]
    return b0 + b1 * np.asarray(w1, dtype=float)


def qbar0(name: str, a: Any, w1: Any, w2: Any) -> Any:
    """``E[Y | A = a, W]`` of a Law S variant."""
    c0, c1, c2 = BASELINE
    w1 = np.asarray(w1, dtype=float)
    return c0 + c1 * w1 + c2 * np.asarray(w2, dtype=float) + np.asarray(a) * tau(name, w1)


def draw(name: str, n: int, seed: int, *, declared: float | None = None) -> pd.DataFrame:
    """One sample of a Law S variant: ``Y, A, W1, W2`` and the declared columns ``p0, p1``.

    ``declared`` replaces the declared mechanism by a constant (the control); the sample
    itself is unchanged.
    """
    if name not in EFFECT:
        raise KeyError(name)
    rng = np.random.default_rng(seed)
    w1 = rng.uniform(-ROOT3, ROOT3, size=n)
    w2 = rng.binomial(1, 0.5, size=n).astype(float)
    g = g0(w1, w2)
    a = (rng.uniform(size=n) < g).astype(float)
    noise = rng.uniform(-NOISE_HALF_WIDTH, NOISE_HALF_WIDTH, size=n)
    y = qbar0(name, a, w1, w2) + noise
    p1 = g if declared is None else np.full(n, float(declared))
    return pd.DataFrame({"Y": y, "A": a, "W1": w1, "W2": w2, "p0": 1.0 - p1, "p1": p1})


@dataclass(frozen=True)
class Grid:
    """The Law S quadrature grid over ``(W1, W2)``.

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
        """The weighted mean of ``values`` over the grid."""
        return float(np.sum(self.weight * np.asarray(values, dtype=float)))


def grid(nodes: int = LEGENDRE_NODES) -> Grid:
    """Gauss-Legendre on ``W1`` (uniform on ``(-sqrt 3, sqrt 3)``), crossed with ``W2``."""
    x, w = leggauss(nodes)
    w1 = ROOT3 * x
    weight = w / w.sum()
    return Grid(
        np.concatenate([w1, w1]),
        np.concatenate([np.zeros(nodes), np.ones(nodes)]),
        np.concatenate([0.5 * weight, 0.5 * weight]),
    )


def truth(name: str, nodes: int = LEGENDRE_NODES) -> dict[str, float]:
    """``P(A = 1)``, the ATE, the ATT and the ATC of a Law S variant."""
    at = grid(nodes)
    g = g0(at.w1, at.w2)
    t = tau(name, at.w1)
    share = at.mean(g)
    return {
        "p1": share,
        "ate": at.mean(t),
        "att": at.mean(g * t) / share,
        "atc": at.mean((1.0 - g) * t) / (1.0 - share),
    }


def efficiency_sd(name: str, estimand: str, form: str, nodes: int = LEGENDRE_NODES) -> float:
    r"""The exact SD of the ATT or ATC curve of a Law S variant at the truth.

    ``form="known_score"`` gives :math:`D_{KS}` and ``form="arm_mean"`` the shipped
    :math:`D_{AM}` (plan section 2.2).  For the ATT, with :math:`p=E g_0`,

    .. math::

        p^2\operatorname{Var}D_{AM} = E\Big[g\sigma^2 + \frac{g^2}{1-g}\sigma^2
            + g(\tau-\psi)^2\Big],\qquad
        p^2\operatorname{Var}D_{KS} = E\Big[g\sigma^2 + \frac{g^2}{1-g}\sigma^2
            + g^2(\tau-\psi)^2\Big],

    and the ATC replaces :math:`g` by :math:`1-g`.  The summands are uncorrelated.
    """
    at = grid(nodes)
    g = g0(at.w1, at.w2)
    weight = g if estimand == "att" else 1.0 - g
    other = 1.0 - weight
    share = at.mean(weight)
    psi = truth(name, nodes)[estimand]
    gap = (tau(name, at.w1) - psi) ** 2
    residual = weight * NOISE_VARIANCE + weight**2 / other * NOISE_VARIANCE
    last = weight * gap if form == "arm_mean" else weight**2 * gap
    if form not in ("arm_mean", "known_score"):
        raise ValueError(form)
    return float(np.sqrt(at.mean(residual + last)) / share)


def curve_correlation(name: str, estimand: str) -> float:
    """``Corr(D_AM, D_KS)``: ``D_KS`` is a projection of ``D_AM``, so it is ``SD_KS / SD_AM``."""
    return efficiency_sd(name, estimand, "known_score") / efficiency_sd(name, estimand, "arm_mean")


#: The curves :func:`curve_correlations` relates, in its row order.
CURVES = ("att_known_score", "att_arm_mean", "atc_known_score", "atc_arm_mean")


def curve_correlations(name: str, nodes: int = LEGENDRE_NODES) -> np.ndarray:
    """The correlation matrix of the :data:`CURVES` of a Law S variant at the truth.

    Each curve is ``c(A, W) eps + d(A, W)`` with ``E[eps | A, W] = 0`` and unit variance, so
    the covariance of two curves is ``E[c_i c_j + d_i d_j]``, summed over ``A`` given ``W``.
    """
    at = grid(nodes)
    g = g0(at.w1, at.w2)
    exact = truth(name, nodes)
    t = tau(name, at.w1)
    p = exact["p1"]
    pieces: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {curve: [] for curve in CURVES}
    for a in (0.0, 1.0):
        res_att = (a - (1.0 - a) * g / (1.0 - g)) / p
        res_atc = (a * (1.0 - g) / g - (1.0 - a)) / (1.0 - p)
        gap_att = (t - exact["att"]) / p
        gap_atc = (t - exact["atc"]) / (1.0 - p)
        pieces["att_known_score"].append((res_att, g * gap_att))
        pieces["att_arm_mean"].append((res_att, a * gap_att))
        pieces["atc_known_score"].append((res_atc, (1.0 - g) * gap_atc))
        pieces["atc_arm_mean"].append((res_atc, (1.0 - a) * gap_atc))
    covariance = np.empty((len(CURVES), len(CURVES)))
    for i, first in enumerate(CURVES):
        for j, second in enumerate(CURVES):
            total = 0.0
            for index, a in enumerate((0.0, 1.0)):
                share = g if a == 1.0 else 1.0 - g
                c_i, d_i = pieces[first][index]
                c_j, d_j = pieces[second][index]
                total += at.mean(share * (NOISE_VARIANCE * c_i * c_j + d_i * d_j))
            covariance[i, j] = total
    scale = np.sqrt(np.diag(covariance))
    return np.asarray(covariance / np.outer(scale, scale))


def control_bias(nodes: int = LEGENDRE_NODES) -> float:
    """The population bias of the known-score ATT under the declared constant mechanism.

    With a constant declaration the weight :math:`c` cancels and the plug-in tends to
    :math:`E\\tau = ` ATE at a consistent outcome regression, so the bias is ATE minus ATT.
    """
    exact = truth("S", nodes)
    return exact["ate"] - exact["att"]


def outcome_support(name: str) -> tuple[float, float]:
    """The range of ``Y`` under a Law S variant, from its bounded parts."""
    corners = [
        float(qbar0(name, a, w1, w2))
        for a in (0.0, 1.0)
        for w1 in (-ROOT3, ROOT3)
        for w2 in (0.0, 1.0)
    ]
    return min(corners) - NOISE_HALF_WIDTH, max(corners) + NOISE_HALF_WIDTH


# ------------------------------------------------------------------------------- Law B
def law_b_truth(arms: int) -> dict[str, float]:
    """The ATT and ATC of each non-reference arm of Law B, by its quadrature grid.

    ``att[a vs 0]`` weights :math:`\\bar Q_a-\\bar Q_0` by :math:`g_0(a\\mid W)` and
    ``atc[a vs 0]`` by :math:`g_0(0\\mid W)`.  At two arms the keys are ``att`` and ``atc``,
    which equal :func:`tests.studies.known_mechanism_law.truth`'s.
    """
    w1, w2, w3, mass = law_b._grid()
    g = law_b.mechanism(w2, arms)
    q = np.column_stack([law_b.outcome_probability(arm, w1, w2, w3) for arm in range(arms)])
    out: dict[str, float] = {}
    for arm in range(1, arms):
        blip = q[:, arm] - q[:, 0]
        att = float(np.sum(mass * g[:, arm] * blip) / np.sum(mass * g[:, arm]))
        atc = float(np.sum(mass * g[:, 0] * blip) / np.sum(mass * g[:, 0]))
        if arms == 2:
            out.update(att=att, atc=atc)
        else:
            out[f"att[{float(arm)} vs 0.0]"] = att
            out[f"atc[{float(arm)} vs 0.0]"] = atc
    return out


# ---------------------------------------------------------------------------- learners
class LawSOutcome(BaseEstimator, RegressorMixin):
    """Least squares on the Law S outcome features of the ``[A, W1, W2]`` design.

    Parameters
    ----------
    kind : str
        ``"correct"``: :math:`(1, A, W_1, W_2, AW_1)`; ``"wrong"``: :math:`(1, A, W_2)`.
    """

    def __init__(self, kind: str = "correct") -> None:
        self.kind = kind

    def _features(self, design: Any) -> np.ndarray:
        values = np.asarray(design, dtype=float)
        a, w1, w2 = values[:, 0], values[:, 1], values[:, 2]
        if self.kind == "correct":
            return np.column_stack([a, w1, w2, a * w1])
        if self.kind == "wrong":
            return np.column_stack([a, w2])
        raise ValueError(self.kind)

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> LawSOutcome:
        self.model_ = LinearRegression().fit(
            self._features(design), np.asarray(target, dtype=float), sample_weight=sample_weight
        )
        self.n_features_in_ = np.asarray(design).shape[1]
        return self

    def predict(self, design: Any) -> np.ndarray:
        return np.asarray(self.model_.predict(self._features(design)), dtype=float)


class LawBOutcome(BaseEstimator, ClassifierMixin):
    """Unpenalized logistic regression on the Law B outcome features.

    Parameters
    ----------
    kind : str
        ``"wrong"`` (main terms in ``(A, W1, W3)``) or ``"correct"`` (the true terms).
    arms : int
        2 or 3.
    """

    def __init__(self, kind: str = "wrong", arms: int = 2) -> None:
        self.kind = kind
        self.arms = arms

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> LawBOutcome:
        self.model_ = LogisticRegression(penalty=None, max_iter=10_000, tol=1e-10).fit(
            law_b.outcome_features(design, self.kind, self.arms),
            target,
            sample_weight=sample_weight,
        )
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, design: Any) -> np.ndarray:
        features = law_b.outcome_features(design, self.kind, self.arms)
        return np.asarray(self.model_.predict_proba(features), dtype=float)


class DeclaredOnly(BaseEstimator, ClassifierMixin):
    """A treatment learner that must never run: the declared mechanism replaces it."""

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> DeclaredOnly:
        raise AssertionError("a fit on a declared mechanism fitted a treatment learner")

    def predict_proba(self, design: Any) -> Any:
        raise AssertionError("a fit on a declared mechanism fitted a treatment learner")


# ---------------------------------------------------------------------------- settings
@dataclass(frozen=True)
class Setting:
    """One fit configuration of the study.

    Parameters
    ----------
    law : str
        ``"S"``, ``"S0"``, ``"B2"`` (Law B, two arms) or ``"B3"`` (three arms).
    outcome : str
        ``"correct"`` or ``"wrong"``.
    form : str
        ``conditional_form``: ``"known_score"`` or ``"arm_mean"``.
    cross_fit : bool
        Whether the outcome regression is cross-fitted over :data:`N_FOLDS` folds (stacked),
        with the declared ``q_bounds``.
    declared : float or None
        A constant declared mechanism in place of the law's own (the control), or ``None``.
    """

    law: str
    outcome: str
    form: str = "known_score"
    cross_fit: bool = False
    declared: float | None = None

    @property
    def arms(self) -> int:
        """The number of treatment arms."""
        return 3 if self.law == "B3" else 2


SETTINGS: dict[str, Setting] = {
    "s_known_score": Setting("S", "correct"),
    "s_arm_mean": Setting("S", "correct", form="arm_mean"),
    "s_known_score_stacked": Setting("S", "correct", cross_fit=True),
    "s_wrong_known_score": Setting("S", "wrong"),
    "s0_known_score": Setting("S0", "correct"),
    "s0_arm_mean": Setting("S0", "correct", form="arm_mean"),
    "s_declared_constant": Setting("S", "correct", declared=CONTROL_CONSTANT),
    "b2_wrong_known_score": Setting("B2", "wrong"),
    "b3_wrong_known_score": Setting("B3", "wrong"),
}

#: The estimands each law's fits report.
ESTIMANDS: dict[str, tuple[str, ...]] = {
    "S": ("att", "atc"),
    "S0": ("att", "atc"),
    "B2": ("att", "atc"),
    "B3": ("att", "atc"),
}


def sample(setting: Setting, n: int, seed: int) -> pd.DataFrame:
    """One draw of a setting's law, with its declared columns."""
    if setting.law in EFFECT:
        return draw(setting.law, n, seed, declared=setting.declared)
    return law_b.sample(n, seed, arms=setting.arms)


def estimator(setting: Setting, **overrides: Any) -> Any:
    """The ``TMLE`` of one setting."""
    from cleverly.estimators import TMLE

    law_s = setting.law in EFFECT
    options: dict[str, Any] = {
        "outcome_learner": (
            LawSOutcome(setting.outcome) if law_s else LawBOutcome(setting.outcome, setting.arms)
        ),
        "treatment_learner": DeclaredOnly(),
        "cross_fit": setting.cross_fit,
        "estimands": ESTIMANDS[setting.law],
        "conditional_form": setting.form,
        "simultaneous": False,
        "max_iter": MAX_ITER,
        "tol": TOL,
        "random_state": 0,
    }
    if setting.law == "B3":
        options["reference"] = 0.0
    if setting.cross_fit:
        options["n_folds"] = N_FOLDS
        options["q_bounds"] = Q_BOUNDS
    options.update(overrides)
    return TMLE(**options)


def fit(frame: pd.DataFrame, name: str, **overrides: Any) -> Any:
    """Fit the named setting on one sample of its law."""
    setting = SETTINGS[name]
    law_s = setting.law in EFFECT
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            estimator(setting, **overrides)
            .fit(
                frame,
                outcome="Y",
                treatment="A",
                covariates=list(COVARIATES if law_s else LAW_B_COVARIATES),
                treatment_probabilities=(
                    PROBABILITIES if law_s else LAW_B_PROBABILITIES[setting.arms]
                ),
            )
            .single()
        )


def setting_truth(name: str) -> dict[str, float]:
    """The truth of every estimand a setting's fit reports, by reported name."""
    setting = SETTINGS[name]
    if setting.law in EFFECT:
        exact = truth(setting.law)
        return {key: exact[key] for key in ("att", "atc")}
    return law_b_truth(setting.arms)


def reported_names(name: str) -> Sequence[str]:
    """The reported estimand names of a setting's fit, in a fixed order."""
    return tuple(setting_truth(name))
