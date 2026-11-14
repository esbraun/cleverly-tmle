r"""Laws, learners, fits and exact-law quantities of ``known-mechanism-sample-weights`` (Fix-16).

The study measures ``mechanism_law="sample"``: a fit on weighted data that declares the
treatment mechanism of the rows *as drawn* (``treatment_probabilities=``) estimates the
mechanism of the weight-tilted law :math:`dP_w = w\,dP / E w` as
:math:`\hat g_w \propto g_0 \hat r`, where :math:`\hat r(a \mid W)` is the treatment learner's
classification of :math:`A` on :math:`W` with sample weights :math:`v = w / g_0(A \mid W)`
(Fix-16 plan section 2.2).  By Bayes' rule :math:`P_w(A = a \mid W) = g_0 m / \sum g_0 m` with
:math:`m(a, W) = E[w \mid A = a, W]`, and :math:`P_v(A = a \mid W) \propto m(a, W)`.

**Law SW** (plan section 7 with the rev 2.1 amendment 1; the weight reads :math:`(A, W)` only,
so the tilted parameter keeps its causal reading):

======================  ===========================================================
quantity                law
======================  ===========================================================
covariates              :math:`W_1 \sim N(0, 1)`, :math:`W_2 \sim \mathrm{Bern}(0.5)`
design mechanism        :math:`g_0(1 \mid W) = 0.3 + 0.4 W_2`, declared by column
outcome                 :math:`Y \sim \mathrm{Bern}(\operatorname{expit}(-1 + A + W_1
                        + 1.5 A W_1 - W_1^2 / 2 + 0.5 W_2))`
weight                  :math:`w = \exp(0.4 A + 0.5 A W_1 + 0.15 W_1)`
======================  ===========================================================

The weight reads :math:`(A, W)` only, so :math:`m = w` and
:math:`r(1 \mid W) = w(1, W) / (w(0, W) + w(1, W)) = \operatorname{expit}(0.4 + 0.5 W_1)`.  A
main-terms logistic classifier on :math:`(W_1, W_2)` is therefore correct for the tilt, and a
logistic outcome regression on :math:`(A, W_1, A W_1, W_1^2, W_2)` is correct for
:math:`\bar Q_0`.  Because the weight does not read :math:`Y`, :math:`\bar Q_w = \bar Q_0` and the
truth is :math:`\Psi(P_w) = E[E[w \mid W] (\bar Q_0(1, W) - \bar Q_0(0, W))] / E[w]`
(:func:`truth`).  :func:`tilted_truth` computes the general form from :math:`\bar Q_w` and
:math:`P_w(W)` for any weight function, so the design test checks that both forms agree on SW and
differ on a variant whose weight reads :math:`Y` (plan review m6).

**Law LW** (plan rev 2.1 amendment 2), two treatment nodes, both declared, no censoring:

======================  ===========================================================
node                    law
======================  ===========================================================
:math:`L_0`             :math:`N(0, 1)`
:math:`A_1`             :math:`\mathrm{Bern}(\operatorname{expit}(0.4 L_0))`, declared
:math:`L_1`             :math:`N(0.5 L_0 + 0.5 A_1, 1)`
:math:`A_2`             :math:`\mathrm{Bern}(\operatorname{expit}(-0.2 + 0.6 L_1))`, declared
:math:`Y`               :math:`0.5 L_0 + 0.4 A_1 + 0.5 A_1 L_0 + 0.6 L_1 + 0.5 A_2
                        + 0.3 A_1 A_2 + N(0, 1)`
weight                  :math:`w = \exp(0.4 A_1 + 0.5 A_1 L_0 + 0.15 L_0)`
======================  ===========================================================

The weight reads :math:`(L_0, A_1)`, so under :math:`P_w` the first treatment factor is tilted to
:math:`\operatorname{expit}(0.4 L_0) \cdot r_1 / \sum` with
:math:`r_1(1 \mid L_0) = \operatorname{expit}(0.4 + 0.5 L_0)`, the second factor is the
declaration (:math:`r_2 = 1/2`), and :math:`P_w(L_0) \propto \phi(L_0) E[w \mid L_0]`.  Every
regression the LTMLE fits is linear within the regimen's followers (the node-two regression of
:math:`Y` on :math:`(L_0, L_1)`, the node-one regression of its prediction on :math:`L_0`), and
every tilt classifier is a correct main-terms logistic on the node's history.  The regimen
means under :math:`P_w` are closed-form given :math:`E_w[L_0]` (:func:`longitudinal_truth`).

Law LW's weight does not read :math:`A_2`, so its tilt sits at node one only
(:math:`r_2 = 1/2`); an error in how the node-two factor composes shows in this study only
through the SE ratio, and the fast tier's unit tests cover node two.  The LTMLE fits cross-fit
in five folds with the declared outcome bounds :data:`LW_Q_BOUNDS`, which no draw reaches
(:func:`lw_outcome_tail_bound`), so the bounds leave the estimand unchanged.

**Misspecified fits.**  The point fits misspecify the outcome regression by its main terms
(``outcome="main"``) and the tilt classifier by dropping :math:`W_1` (``tilt="w2"``,
:class:`W2OnlyLogistic`, which misses the slope of :math:`r`).  The ``"sample"`` reading is
doubly robust: either correct nuisance gives a consistent estimate, and both wrong do not.  The
LW fits misspecify the node-one regression as an intercept (:class:`InterceptOnly`).

**Exact-law quantities.**  :func:`control_limit` solves the population targeting step of the
``"target"`` fit with the main-terms outcome regression on :math:`P_w`, which gives the control's
exact bias; :func:`both_wrong_limit` solves the same step with the population W2-only tilt, and
:func:`lw_control_limit` gives the LW control's limit in closed form.  :func:`point_influence_sd` is the standard deviation of the shipped curve
:math:`(w / \bar w) D^*(\bar Q_0, g_w)` at the truth, the asymptotic SD of the correct
``"sample"`` fit, and :func:`longitudinal_influence_sd` is the same for the LTMLE contrast, by a
fixed-seed Monte Carlo.
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

# ------------------------------------------------------------------------------- law SW
#: The adjustment set.
COVARIATES = ("W1", "W2")
#: ``(A, A W1, W1)`` coefficients of ``log w`` (plan rev 2.1 amendment 1: plan 7's halved).
WEIGHT_COEFFICIENTS = (0.4, 0.5, 0.15)
#: ``(intercept, A, W1, A W1, W1^2, W2)`` of the outcome logit.
OUTCOME_COEFFICIENTS = (-1.0, 1.0, 1.0, 1.5, -0.5, 0.5)
#: The declared mechanism, by column.
PROBABILITIES = {0.0: "p0", 1.0: "p1"}
#: The estimands every point fit reports; the cells read the ATE.
ESTIMANDS = ("ate", "ey1", "ey0")
#: The treatment bounds of every point fit.  ``g_w`` lies in about [0.06, 0.97] on the support
#: that carries mass, so they never bind (the design test checks).
G_BOUNDS = (0.01, 0.99)
MAX_ITER = 100
TOL = 1e-10
N_FOLDS = 5
#: Gauss-Hermite nodes of ``W1``.
NODES = 200


def g0(w2: Any) -> Any:
    """``P(A = 1 | W)`` of the rows as drawn."""
    return 0.3 + 0.4 * np.asarray(w2, dtype=float)


def qbar0(a: Any, w1: Any, w2: Any) -> Any:
    """``E[Y | A = a, W]``."""
    b0, ba, b1, baw, b11, b2 = OUTCOME_COEFFICIENTS
    a, w1, w2 = (np.asarray(x, dtype=float) for x in (a, w1, w2))
    return expit(b0 + ba * a + b1 * w1 + baw * a * w1 + b11 * w1**2 + b2 * w2)


def weight(a: Any, w1: Any) -> Any:
    """The observation weight ``w(A, W)``."""
    ca, caw, c1 = WEIGHT_COEFFICIENTS
    a, w1 = np.asarray(a, dtype=float), np.asarray(w1, dtype=float)
    return np.exp(ca * a + caw * a * w1 + c1 * w1)


def tilt(w1: Any) -> Any:
    """``r(1 | W) = P_v(A = 1 | W)`` with ``v = w / g0(A | W)``: ``w(1, W) / sum_a w(a, W)``."""
    ca, caw, _ = WEIGHT_COEFFICIENTS
    return expit(ca + caw * np.asarray(w1, dtype=float))


def tilted_mechanism(w1: Any, w2: Any) -> Any:
    """``P_w(A = 1 | W) = g0 r / (g0 r + (1 - g0)(1 - r))``."""
    g, r = g0(w2), tilt(w1)
    return g * r / (g * r + (1.0 - g) * (1.0 - r))


def draw(n: int, seed: int) -> pd.DataFrame:
    """One sample of law SW: ``W1, W2, A, Y``, the weight ``wt`` and the declared ``p0, p1``."""
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal(n)
    w2 = (rng.random(n) < 0.5).astype(float)
    a = (rng.random(n) < g0(w2)).astype(float)
    y = (rng.random(n) < qbar0(a, w1, w2)).astype(float)
    p1 = g0(w2)
    return pd.DataFrame(
        {"W1": w1, "W2": w2, "A": a, "Y": y, "wt": weight(a, w1), "p0": 1.0 - p1, "p1": p1}
    )


@dataclass(frozen=True)
class Points:
    """A weighted finite set of ``(W1, W2)`` points: the quadrature grid of law SW.

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


def grid(nodes: int = NODES) -> Points:
    """Gauss-Hermite nodes of ``W1`` crossed with the two values of ``W2``."""
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    w = w / w.sum()
    return Points(
        np.concatenate([x, x]), np.repeat([0.0, 1.0], nodes), np.concatenate([w, w]) / 2.0
    )


#: A weight function of ``(a, W1, W2, y)``: the general form :func:`tilted_truth` reads.
WeightFunction = Callable[[Any, Any, Any, Any], Any]


def sw_weight(a: Any, w1: Any, w2: Any, y: Any) -> Any:
    """Law SW's weight in the general form: it ignores ``W2`` and ``Y``."""
    return weight(a, w1) * np.ones_like(np.asarray(y, dtype=float))


def y_reading_weight(a: Any, w1: Any, w2: Any, y: Any) -> Any:
    """A variant that also reads ``Y``: ``w(A, W) (1 + Y)``, the guard of plan review m6."""
    return weight(a, w1) * (1.0 + np.asarray(y, dtype=float))


def tilted_truth(weight_of: WeightFunction = sw_weight, points: Points | None = None) -> dict:
    r"""``Psi(P_w)`` from :math:`\bar Q_w(a, W) = E[w Y | a, W] / E[w | a, W]` and :math:`P_w(W)`.

    The general form for a weight that may read ``Y``: :math:`P_w(W) \propto p(W) E[w \mid W]`
    and the arm mean is :math:`E_{P_w}[\bar Q_w(a, W)]`.
    """
    at = grid() if points is None else points
    out: dict[str, float] = {}
    mass = 0.0
    q_w: dict[float, np.ndarray] = {}
    for arm in (0.0, 1.0):
        q = qbar0(arm, at.w1, at.w2)
        w_one = weight_of(arm, at.w1, at.w2, 1.0)
        w_zero = weight_of(arm, at.w1, at.w2, 0.0)
        m = q * w_one + (1.0 - q) * w_zero  # E[w | A = arm, W]
        q_w[arm] = q * w_one / m
        g = g0(at.w2) if arm == 1.0 else 1.0 - g0(at.w2)
        mass = mass + g * m  # accumulates E[w | W]
    total = at.mean(mass)
    for arm, name in ((1.0, "ey1"), (0.0, "ey0")):
        out[name] = at.mean(mass * q_w[arm]) / total
    out["ate"] = out["ey1"] - out["ey0"]
    return out


def q0_formula(weight_of: WeightFunction = sw_weight, points: Points | None = None) -> dict:
    r"""``Psi`` computed with :math:`\bar Q_0` in place of :math:`\bar Q_w`, tilting ``W`` only.

    Equal to :func:`tilted_truth` exactly when the weight does not read ``Y``.
    """
    at = grid() if points is None else points
    mass = 0.0
    for arm in (0.0, 1.0):
        q = qbar0(arm, at.w1, at.w2)
        m = q * weight_of(arm, at.w1, at.w2, 1.0) + (1.0 - q) * weight_of(arm, at.w1, at.w2, 0.0)
        mass = mass + (g0(at.w2) if arm == 1.0 else 1.0 - g0(at.w2)) * m
    total = at.mean(mass)
    one = at.mean(mass * qbar0(1.0, at.w1, at.w2)) / total
    zero = at.mean(mass * qbar0(0.0, at.w1, at.w2)) / total
    return {"ey1": one, "ey0": zero, "ate": one - zero}


def truth() -> dict[str, float]:
    """The tilted arm means and ATE of law SW: :math:`\\bar Q_w = \\bar Q_0` here."""
    return q0_formula(sw_weight)


def unweighted_truth() -> dict[str, float]:
    """The arm means and ATE of the rows as drawn, which the guard sets apart from the truth."""
    at = grid()
    one = at.mean(qbar0(1.0, at.w1, at.w2))
    zero = at.mean(qbar0(0.0, at.w1, at.w2))
    return {"ey1": one, "ey0": zero, "ate": one - zero}


def kish_ratio() -> float:
    """The Kish effective sample size over n, ``E[w]^2 / E[w^2]``, of law SW."""
    at = grid()
    first = second = 0.0
    for arm in (0.0, 1.0):
        g = g0(at.w2) if arm == 1.0 else 1.0 - g0(at.w2)
        first += at.mean(g * weight(arm, at.w1))
        second += at.mean(g * weight(arm, at.w1) ** 2)
    return first**2 / second


def point_influence_sd() -> float:
    r"""The SD of :math:`(w / E w) D^*(\bar Q_0, g_w)` for the ATE under the sampling law.

    The asymptotic SD of the ``"sample"`` fit with both learners correct, times
    :math:`\sqrt n`: with :math:`\bar Q` correct the estimated tilt adds nothing to the curve,
    and the normalization of the weights gives the centring at :math:`\psi`.
    """
    at = grid()
    psi = truth()["ate"]
    blip = qbar0(1.0, at.w1, at.w2) - qbar0(0.0, at.w1, at.w2)
    g_w = tilted_mechanism(at.w1, at.w2)
    mean_w = 0.0
    second = 0.0
    for arm in (0.0, 1.0):
        g = g0(at.w2) if arm == 1.0 else 1.0 - g0(at.w2)
        fitted = g_w if arm == 1.0 else 1.0 - g_w
        w = weight(arm, at.w1)
        q = qbar0(arm, at.w1, at.w2)
        mean_w += at.mean(g * w)
        second += at.mean(g * w**2 * (q * (1.0 - q) / fitted**2 + (blip - psi) ** 2))
    return float(np.sqrt(second) / mean_w)


# ---------------------------------------------------------------------------- the learners
def newton_logistic(design: np.ndarray, target: np.ndarray, weight_: np.ndarray) -> np.ndarray:
    """Weighted logistic maximum likelihood with an intercept, by Newton, ridge ``1e-12``.

    ``target`` may be a probability: a weighted finite law passes ``Q_0`` itself, and the
    population fit solves the same score equation.
    """
    x = np.column_stack([np.ones(len(target)), design])
    beta = np.zeros(x.shape[1])
    beta[0] = logit(np.clip(np.average(target, weights=weight_), 1e-6, 1.0 - 1e-6))
    for _ in range(100):
        p = expit(x @ beta)
        hessian = (x * (weight_ * p * (1.0 - p))[:, None]).T @ x
        step = np.linalg.solve(hessian + 1e-12 * np.eye(len(beta)), x.T @ (weight_ * (target - p)))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            return beta
    raise RuntimeError("the logistic fit did not converge")


def outcome_features(design: Any, kind: str) -> np.ndarray:
    """The outcome features from the ``[A, W1, W2]`` design.

    ``kind="correct"`` gives the true terms ``(A, W1, A W1, W1^2, W2)`` and ``kind="main"`` the
    main terms ``(A, W1, W2)``, which omit ``A W1`` and ``W1^2``.
    """
    values = np.asarray(design, dtype=float)
    a, w1, w2 = values[:, 0], values[:, 1], values[:, 2]
    if kind == "main":
        return np.column_stack([a, w1, w2])
    if kind != "correct":
        raise ValueError(kind)
    return np.column_stack([a, w1, a * w1, w1**2, w2])


class OutcomeGLM(BaseEstimator, ClassifierMixin):
    """Unpenalized logistic regression on the declared outcome features, by Newton.

    Parameters
    ----------
    kind : str
        ``"correct"`` or ``"main"`` (:func:`outcome_features`).
    """

    def __init__(self, kind: str = "correct") -> None:
        self.kind = kind

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> OutcomeGLM:
        y = np.asarray(target, dtype=float)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        self.beta_ = newton_logistic(outcome_features(design, self.kind), y, w)
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, design: Any) -> np.ndarray:
        features = outcome_features(design, self.kind)
        p = expit(self.beta_[0] + features @ self.beta_[1:])
        return np.column_stack([1.0 - p, p])


class MainTermsLogistic(BaseEstimator, ClassifierMixin):
    """Unpenalized main-terms logistic regression on every design column, by Newton.

    The tilt classifier of every ``"sample"`` cell: its ``fit`` takes ``sample_weight``, which
    the ``"sample"`` route sets to ``v = w / g0(A | W)`` (normalized to mean one per fold).
    """

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> MainTermsLogistic:
        y = np.asarray(target, dtype=float)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        self.beta_ = newton_logistic(np.asarray(design, dtype=float), y, w)
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, design: Any) -> np.ndarray:
        values = np.asarray(design, dtype=float)
        p = expit(self.beta_[0] + values @ self.beta_[1:])
        return np.column_stack([1.0 - p, p])


class DeclaredOnly(BaseEstimator, ClassifierMixin):
    """A treatment learner that must never run: the ``"target"`` fit divides by the declaration."""

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> DeclaredOnly:
        raise AssertionError("a target fit on a declared mechanism fitted a treatment learner")

    def predict_proba(self, design: Any) -> Any:
        raise AssertionError("a target fit on a declared mechanism fitted a treatment learner")


class LinearOLS(BaseEstimator, RegressorMixin):
    """Weighted least squares with an intercept on every design column (LW's regressions)."""

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> LinearOLS:
        x = np.column_stack([np.ones(len(target)), np.asarray(design, dtype=float)])
        y = np.asarray(target, dtype=float)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        root = np.sqrt(w)
        self.beta_ = np.linalg.lstsq(x * root[:, None], y * root, rcond=None)[0]
        return self

    def predict(self, design: Any) -> np.ndarray:
        x = np.asarray(design, dtype=float)
        return np.asarray(self.beta_[0] + x @ self.beta_[1:])


class W2OnlyLogistic(MainTermsLogistic):
    """The misspecified tilt classifier: a main-terms logistic on ``W2`` alone.

    It reads the last design column only.  The tilt ``r`` has a ``W1`` slope and no ``W2``
    term, so its population fit is the constant ``P_v(A = 1)`` (:func:`w2_only_tilt`).
    """

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> W2OnlyLogistic:
        super().fit(np.asarray(design, dtype=float)[:, -1:], target, sample_weight)
        return self

    def predict_proba(self, design: Any) -> np.ndarray:
        return super().predict_proba(np.asarray(design, dtype=float)[:, -1:])


class InterceptOnly(BaseEstimator, RegressorMixin):
    """The misspecified LW node-one regression: the weighted mean of its target."""

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> InterceptOnly:
        y = np.asarray(target, dtype=float)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        self.mean_ = float(np.average(y, weights=w))
        return self

    def predict(self, design: Any) -> np.ndarray:
        return np.full(len(np.asarray(design)), self.mean_)


# ---------------------------------------------------------------------------- point fits
@dataclass(frozen=True)
class Setting:
    """One point fit: the reading of the declaration and the outcome regression.

    Parameters
    ----------
    reading : str
        ``"sample"`` or ``"target"``, passed as ``mechanism_law=``.
    outcome : str
        ``"correct"`` or ``"main"`` (:func:`outcome_features`).
    tilt : str
        ``"correct"`` (:class:`MainTermsLogistic`) or ``"w2"`` (:class:`W2OnlyLogistic`); a
        ``"target"`` fit fits no tilt and reads ``"correct"``.
    """

    reading: str
    outcome: str
    tilt: str = "correct"


#: The point fits.  ``sample_correct`` has both nuisances correct; ``target_main`` is the old
#: reading with the main-terms outcome regression (the control); ``sample_main`` misspecifies
#: the outcome regression, ``tilt_wrong`` the tilt classifier, and ``both_wrong`` both (the
#: control of ``tilt_wrong``).
SETTINGS = {
    "sample_correct": Setting("sample", "correct"),
    "target_main": Setting("target", "main"),
    "sample_main": Setting("sample", "main"),
    "tilt_wrong": Setting("sample", "correct", "w2"),
    "both_wrong": Setting("sample", "main", "w2"),
}


def fit(frame: pd.DataFrame, name: str, *, random_state: int = 0) -> Any:
    """Fit the named point setting on one sample of law SW (five-fold cross-fit)."""
    from cleverly.estimators import TMLE

    setting = SETTINGS[name]
    if setting.reading == "target":
        treatment: Any = DeclaredOnly()
    else:
        treatment = W2OnlyLogistic() if setting.tilt == "w2" else MainTermsLogistic()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            TMLE(
                outcome_learner=OutcomeGLM(setting.outcome),
                treatment_learner=treatment,
                cross_fit=True,
                n_folds=N_FOLDS,
                estimands=ESTIMANDS,
                simultaneous=False,
                g_bounds=G_BOUNDS,
                max_iter=MAX_ITER,
                tol=TOL,
                random_state=random_state,
            )
            .fit(
                frame,
                outcome="Y",
                treatment="A",
                covariates=list(COVARIATES),
                weights="wt",
                treatment_probabilities=PROBABILITIES,
                mechanism_law=setting.reading,
            )
            .single()
        )


# ------------------------------------------------------------------- the control's limit
def main_terms_limit(points: Points | None = None) -> np.ndarray:
    """The population main-terms logistic fit of ``Y`` on ``(A, W1, W2)`` under ``P_w``."""
    at = grid() if points is None else points
    design, target, mass = [], [], []
    for arm in (0.0, 1.0):
        g = g0(at.w2) if arm == 1.0 else 1.0 - g0(at.w2)
        design.append(np.column_stack([np.full_like(at.w1, arm), at.w1, at.w2]))
        target.append(qbar0(arm, at.w1, at.w2))
        mass.append(at.weight * g * weight(arm, at.w1))
    return newton_logistic(np.vstack(design), np.concatenate(target), np.concatenate(mass))


def w2_only_tilt() -> float:
    """The population fit of :class:`W2OnlyLogistic`: ``P_v(A = 1) = E w(1, W) / sum_a``.

    Under the ``v``-weighted law ``P_v(A = a, W) \\propto p(W) w(a, W)``, which reads
    ``W2`` nowhere, so the ``W2`` coefficient is zero and the fit is this constant.
    """
    at = grid()
    one, zero = at.mean(weight(1.0, at.w1)), at.mean(weight(0.0, at.w1))
    return one / (one + zero)


def misspecified_limit(mechanism: Callable[[Points], Any], points: Points | None = None) -> dict:
    """The limit of a fit with the main-terms outcome regression and a fixed mechanism, on ``P_w``.

    For each arm the targeting step solves
    :math:`E_{P_w}[1\\{A = a\\} H_a (Y - \\operatorname{expit}(\\operatorname{logit} \\bar Q
    + \\epsilon H_a))] = 0` with :math:`H_a = 1 / m(a \\mid W)`, ``m`` the mechanism the fit
    divides by, and the arm mean is the :math:`P_w`-mean of the targeted regression.
    ``mechanism`` maps the points to ``m(1 | W)``.  Returns the arm means, the ATE and its bias.
    """
    at = grid() if points is None else points
    beta = main_terms_limit(at)
    fitted_one = np.asarray(mechanism(at), dtype=float)
    mass_w = 0.0
    for arm in (0.0, 1.0):
        mass_w = mass_w + (g0(at.w2) if arm == 1.0 else 1.0 - g0(at.w2)) * weight(arm, at.w1)
    means: dict[float, float] = {}
    for arm in (0.0, 1.0):
        g = g0(at.w2) if arm == 1.0 else 1.0 - g0(at.w2)
        clever = 1.0 / (fitted_one if arm == 1.0 else 1.0 - fitted_one)
        start = expit(beta[0] + beta[1] * arm + beta[2] * at.w1 + beta[3] * at.w2)
        q0 = qbar0(arm, at.w1, at.w2)
        mass = g * weight(arm, at.w1)

        def score(
            eps: float, start: Any = start, q0: Any = q0, mass: Any = mass, clever: Any = clever
        ) -> float:
            return at.mean(mass * clever * (q0 - expit(logit(start) + eps * clever)))

        eps = brentq(score, -50.0, 50.0, xtol=1e-15, rtol=1e-15, maxiter=500)
        star = expit(logit(start) + eps * clever)
        means[arm] = at.mean(mass_w * star) / at.mean(mass_w)
    ate = means[1.0] - means[0.0]
    return {"ey1": means[1.0], "ey0": means[0.0], "ate": ate, "bias": ate - truth()["ate"]}


def control_limit(points: Points | None = None) -> dict[str, float]:
    """The ``"target"`` fit's limit with the main-terms outcome regression: it divides by ``g0``."""
    return misspecified_limit(lambda at: g0(at.w2), points)


def both_wrong_limit(points: Points | None = None) -> dict[str, float]:
    """The ``both_wrong`` fit's limit: main-terms outcome, tilt ``g0 r~ / sum`` with constant ``r~``."""
    r = w2_only_tilt()

    def tilted(at: Points) -> Any:
        g = g0(at.w2)
        return g * r / (g * r + (1.0 - g) * (1.0 - r))

    return misspecified_limit(tilted, points)


# ------------------------------------------------------------------------------- law LW
#: The node columns of law LW.
LONGITUDINAL_COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["L0"],
    "time_varying": [[], ["L1"]],
}
#: The declared node factors, by column.
NODE_PROBABILITIES = {"A1": {0.0: "g1_0", 1.0: "g1_1"}, "A2": {0.0: "g2_0", 1.0: "g2_1"}}
#: The regimens and their reference.
PLANS: dict[str, int] = {"always": 1, "never": 0}
REFERENCE_PLAN = "never"
CONTRAST = "ate_regimen[always vs never]"
LONGITUDINAL_ESTIMANDS = ("ey_regimen[always]", "ey_regimen[never]", CONTRAST)
#: ``(A1, A1 L0, L0)`` coefficients of ``log w``: law SW's, with ``L0`` for ``W1``.
LW_WEIGHT_COEFFICIENTS = WEIGHT_COEFFICIENTS
#: ``(L0, A1, A1 L0, L1, A2, A1 A2)`` coefficients of ``E[Y | history]`` (no intercept).
LW_OUTCOME_COEFFICIENTS = (0.5, 0.4, 0.5, 0.6, 0.5, 0.3)
#: ``(L0, A1)`` coefficients of ``E[L1 | L0, A1]``.
LW_COVARIATE_COEFFICIENTS = (0.5, 0.5)
#: The LTMLE fits cross-fit in five folds.  A cross-fitted fit of a continuous outcome needs a
#: declared outcome support: :data:`LW_Q_BOUNDS` sits about 13 SD from the mean of ``Y``, no
#: draw reaches it (:func:`lw_outcome_tail_bound`), so it leaves the estimand unchanged.
LW_FOLDS = 5
LW_Q_BOUNDS = (-25.0, 25.0)
#: The treatment bounds of the LTMLE fits.  They bound each cumulative product of node factors,
#: and the declared ones reach below 0.01 on about one follower in 10^7 (the ``"target"`` fit
#: refuses a declared product outside the bounds), the tilted ones below 0.001; the lower bound
#: sits under both, so it never binds (the design test checks).
LW_G_BOUNDS = (1e-4, 1.0)
#: The Monte Carlo draws and seed of :func:`longitudinal_influence_sd`.
INFLUENCE_DRAWS = 4_000_000
INFLUENCE_SEED = 2026111416


def lw_g1(l0: Any) -> Any:
    """``P(A1 = 1 | L0)`` of the rows as drawn."""
    return expit(0.4 * np.asarray(l0, dtype=float))


def lw_g2(l1: Any) -> Any:
    """``P(A2 = 1 | L1)`` of the rows as drawn."""
    return expit(-0.2 + 0.6 * np.asarray(l1, dtype=float))


def lw_weight(l0: Any, a1: Any) -> Any:
    """Law LW's weight ``w(L0, A1)``."""
    ca, caw, c1 = LW_WEIGHT_COEFFICIENTS
    l0, a1 = np.asarray(l0, dtype=float), np.asarray(a1, dtype=float)
    return np.exp(ca * a1 + caw * a1 * l0 + c1 * l0)


def lw_l1_mean(l0: Any, a1: Any) -> Any:
    """``E[L1 | L0, A1]``."""
    b0, b1 = LW_COVARIATE_COEFFICIENTS
    return b0 * np.asarray(l0, dtype=float) + b1 * np.asarray(a1, dtype=float)


def lw_q2(l0: Any, a1: Any, l1: Any, a2: Any) -> Any:
    """``E[Y | L0, A1, L1, A2]``."""
    c0, ca1, ca1l0, c1, ca2, ca12 = LW_OUTCOME_COEFFICIENTS
    l0, a1, l1, a2 = (np.asarray(x, dtype=float) for x in (l0, a1, l1, a2))
    return c0 * l0 + ca1 * a1 + ca1l0 * a1 * l0 + c1 * l1 + ca2 * a2 + ca12 * a1 * a2


def lw_q1(l0: Any, a1: Any, a2: Any) -> Any:
    """``E[Q2(L0, a1, L1, a2) | L0, A1 = a1]``: linear in ``L0`` within the followers."""
    return lw_q2(l0, a1, lw_l1_mean(l0, a1), a2)


def lw_tilted_first(l0: Any) -> Any:
    """``P_w(A1 = 1 | L0)``: the declaration times ``r1 = expit(0.4 + 0.5 L0)``, normalized."""
    ca, caw, _ = LW_WEIGHT_COEFFICIENTS
    g, r = lw_g1(l0), expit(ca + caw * np.asarray(l0, dtype=float))
    return g * r / (g * r + (1.0 - g) * (1.0 - r))


def draw_longitudinal(n: int, seed: int) -> pd.DataFrame:
    """One sample of law LW, with its weight ``wt`` and declared factors."""
    rng = np.random.default_rng(seed)
    l0 = rng.standard_normal(n)
    a1 = (rng.random(n) < lw_g1(l0)).astype(float)
    l1 = lw_l1_mean(l0, a1) + rng.standard_normal(n)
    a2 = (rng.random(n) < lw_g2(l1)).astype(float)
    y = lw_q2(l0, a1, l1, a2) + rng.standard_normal(n)
    g1, g2 = lw_g1(l0), lw_g2(l1)
    return pd.DataFrame(
        {
            "L0": l0,
            "A1": a1,
            "L1": l1,
            "A2": a2,
            "Y": y,
            "wt": lw_weight(l0, a1),
            "g1_0": 1.0 - g1,
            "g1_1": g1,
            "g2_0": 1.0 - g2,
            "g2_1": g2,
        }
    )


def lw_tilted_baseline_moments(nodes: int = NODES) -> tuple[float, float]:
    """``(E_w[L0], E[w])``: the tilted baseline mean and the normalizer, by quadrature."""
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    w = w / w.sum()
    g = lw_g1(x)
    mass = (1.0 - g) * lw_weight(x, 0.0) + g * lw_weight(x, 1.0)  # E[w | L0]
    total = float(np.sum(w * mass))
    return float(np.sum(w * mass * x)) / total, total


def longitudinal_truth() -> dict[str, float]:
    """The regimen means and contrast under ``P_w``: a longhand g-formula.

    ``P_w`` keeps ``L1 | L0, A1`` and ``Y | history`` (the weight reads ``(L0, A1)``, which they
    condition on) and tilts the baseline to ``phi(L0) E[w | L0] / E[w]``.  The g-formula of a
    static plan ``(a1, a2)`` is ``E_w[Q1(L0, a1, a2)]``, linear in ``L0``.
    """
    mean_l0, _ = lw_tilted_baseline_moments()
    out = {}
    for name, arm in PLANS.items():
        out[f"ey_regimen[{name}]"] = float(lw_q1(mean_l0, float(arm), float(arm)))
    out[CONTRAST] = out["ey_regimen[always]"] - out["ey_regimen[never]"]
    return out


def longitudinal_unweighted_truth() -> dict[str, float]:
    """The regimen means and contrast of the rows as drawn (``E[L0] = 0``)."""
    out = {f"ey_regimen[{name}]": float(lw_q1(0.0, arm, arm)) for name, arm in PLANS.items()}
    out[CONTRAST] = out["ey_regimen[always]"] - out["ey_regimen[never]"]
    return out


def longitudinal_kish_ratio(nodes: int = NODES) -> float:
    """``E[w]^2 / E[w^2]`` of law LW, by quadrature."""
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    w = w / w.sum()
    g = lw_g1(x)
    first = float(np.sum(w * ((1.0 - g) * lw_weight(x, 0.0) + g * lw_weight(x, 1.0))))
    second = float(np.sum(w * ((1.0 - g) * lw_weight(x, 0.0) ** 2 + g * lw_weight(x, 1.0) ** 2)))
    return first**2 / second


def longitudinal_influence_sd(draws: int = INFLUENCE_DRAWS, seed: int = INFLUENCE_SEED) -> float:
    r"""The SD of the weighted LTMLE curve of the contrast at the truth, by Monte Carlo.

    Each plan's curve on ``P_w`` is
    :math:`D_d = \frac{I_1}{g_{1w}} \frac{I_2}{g_2} (Y - Q_2) + \frac{I_1}{g_{1w}} (Q_2 - Q_1)
    + Q_1 - \psi_d`, with :math:`I_t` the indicator of following the plan at node ``t``; the
    shipped curve is :math:`(w / \bar w) (D_{always} - D_{never})`.  A fixed-seed draw of
    :data:`INFLUENCE_DRAWS` rows gives the SD to about 0.1%.
    """
    frame = draw_longitudinal(draws, seed)
    l0, a1, l1, a2, y, w = (frame[c].to_numpy() for c in ("L0", "A1", "L1", "A2", "Y", "wt"))
    truth_ = longitudinal_truth()
    g1w = lw_tilted_first(l0)
    g2 = lw_g2(l1)
    curve = np.zeros(draws)
    for name, arm in PLANS.items():
        a = float(arm)
        first = (a1 == a) / np.where(a == 1.0, g1w, 1.0 - g1w)
        second = (a2 == a) / np.where(a == 1.0, g2, 1.0 - g2)
        q2 = lw_q2(l0, a, l1, a)
        q1 = lw_q1(l0, a, a)
        piece = first * second * (y - q2) + first * (q2 - q1) + q1 - truth_[f"ey_regimen[{name}]"]
        curve += piece if name == "always" else -piece
    curve *= w / w.mean()
    return float(curve.std())


def lw_outcome_tail_bound() -> float:
    r"""An upper bound on :math:`P(|Y| > \max |q|)` under law LW, ``q`` :data:`LW_Q_BOUNDS`.

    With :math:`L_1 = 0.5 L_0 + 0.5 A_1 + \eta` and the outcome noise :math:`\varepsilon`,
    :math:`|Y| \le 1.3 |L_0| + 1.5 + 0.6 |\eta| + |\varepsilon|`, and by Cauchy-Schwarz
    :math:`1.3 |L_0| + 0.6 |\eta| + |\varepsilon| \le \sqrt{3.05}\, \chi_3` for the
    independent standard normals, so the bound is a :math:`\chi^2_3` tail.
    """
    from scipy.stats import chi2

    reach = min(abs(LW_Q_BOUNDS[0]), abs(LW_Q_BOUNDS[1])) - 1.5
    return float(chi2.sf((reach / np.sqrt(3.05)) ** 2, 3))


def lw_control_limit() -> dict[str, float]:
    r"""The limit of ``ltmle_target_flat``: ``"target"`` with the intercept-only node-one regression.

    The node-two regression is correct, so its fluctuation vanishes in the limit.  At node one
    an intercept-only initial fit makes the logistic intercept shift a constant, which solves the
    weighted score :math:`\sum w (1\{A_1 = a\} / g_1(a \mid L_0)) (Q_2 - Q_1^*) = 0` over the
    followers: a Hajek mean of :math:`Q_2` with weight :math:`w / g_1`.  Its limit is
    :math:`E[w(L_0, a) Q_1(L_0, a, a)] / E[w(L_0, a)]` with :math:`L_0 \sim N(0, 1)`: the mean
    of a function linear in :math:`L_0` under :math:`L_0 \sim N(c_a, 1)`, with :math:`c_a` the
    ``L0`` slope of :math:`\log w(L_0, a)`.  Under ``"sample"`` the same weight is
    :math:`w / g_{1w} \propto E[w \mid L_0] / g_1`, which gives the truth instead.
    """
    _, caw, c1 = LW_WEIGHT_COEFFICIENTS
    out = {}
    for name, arm in PLANS.items():
        shift = c1 + caw * float(arm)
        out[f"ey_regimen[{name}]"] = float(lw_q1(shift, float(arm), float(arm)))
    out[CONTRAST] = out["ey_regimen[always]"] - out["ey_regimen[never]"]
    out["bias"] = out[CONTRAST] - longitudinal_truth()[CONTRAST]
    return out


#: The LTMLE fits: ``(reading, node-one regression)``.  ``ltmle_sample`` has every nuisance
#: correct; ``ltmle_target_flat`` is the old reading with the intercept-only node-one
#: regression (the control) and ``ltmle_sample_flat`` the ``"sample"`` reading with it.
LW_SETTINGS: dict[str, tuple[str, str]] = {
    "ltmle_sample": ("sample", "linear"),
    "ltmle_target_flat": ("target", "intercept"),
    "ltmle_sample_flat": ("sample", "intercept"),
}


def fit_longitudinal(
    frame: pd.DataFrame, name: str = "ltmle_sample", *, random_state: int = 0
) -> Any:
    """Fit the named LTMLE setting on one sample of law LW (five-fold cross-fit)."""
    from cleverly.longitudinal import LTMLE

    reading, node_one = LW_SETTINGS[name]
    treatment = MainTermsLogistic() if reading == "sample" else DeclaredOnly()
    pseudo = LinearOLS() if node_one == "linear" else InterceptOnly()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LTMLE(
            PLANS,
            reference=REFERENCE_PLAN,
            outcome_learner=LinearOLS(),
            pseudo_learner=pseudo,
            treatment_learner=treatment,
            n_folds=LW_FOLDS,
            q_bounds=LW_Q_BOUNDS,
            g_bounds=LW_G_BOUNDS,
            simultaneous=False,
            max_iter=MAX_ITER,
            tol=TOL,
            random_state=random_state,
        ).fit(
            frame,
            **LONGITUDINAL_COLUMNS,
            weights="wt",
            treatment_probabilities=NODE_PROBABILITIES,
            mechanism_law=reading,
        )


def fit_named(frame: pd.DataFrame, name: str) -> Any:
    """Fit any named setting: a point fit (:data:`SETTINGS`) or an LTMLE fit (:data:`LW_SETTINGS`)."""
    return fit_longitudinal(frame, name) if name in LW_SETTINGS else fit(frame, name)
