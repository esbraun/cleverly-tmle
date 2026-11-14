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

**Exact-law quantities.**  :func:`control_limit` solves the population targeting step of the
``"target"`` fit with the main-terms outcome regression on :math:`P_w`, which gives the control's
exact bias.  :func:`point_influence_sd` is the standard deviation of the shipped curve
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
    """

    reading: str
    outcome: str


#: The point fits.  ``sample_correct`` is the gated cell, ``target_main`` the control (the old
#: reading with the misspecified outcome regression) and ``sample_main`` the reporting cell.
SETTINGS = {
    "sample_correct": Setting("sample", "correct"),
    "target_main": Setting("target", "main"),
    "sample_main": Setting("sample", "main"),
}


def fit(frame: pd.DataFrame, name: str, *, random_state: int = 0) -> Any:
    """Fit the named point setting on one sample of law SW (five-fold cross-fit)."""
    from cleverly.estimators import TMLE

    setting = SETTINGS[name]
    treatment = MainTermsLogistic() if setting.reading == "sample" else DeclaredOnly()
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


def control_limit(points: Points | None = None) -> dict[str, float]:
    """The ``"target"`` fit's limit with the main-terms outcome regression, on ``P_w``.

    The fit divides by ``g0``.  For each arm the targeting step solves
    :math:`E_{P_w}[1\\{A = a\\} H_a (Y - \\operatorname{expit}(\\operatorname{logit} \\bar Q
    + \\epsilon H_a))] = 0` with :math:`H_a = 1 / g_0(a \\mid W)`, and the arm mean is the
    :math:`P_w`-mean of the targeted regression.  Returns the arm means, the ATE and its bias.
    """
    at = grid() if points is None else points
    beta = main_terms_limit(at)
    mass_w = 0.0
    for arm in (0.0, 1.0):
        mass_w = mass_w + (g0(at.w2) if arm == 1.0 else 1.0 - g0(at.w2)) * weight(arm, at.w1)
    means: dict[float, float] = {}
    for arm in (0.0, 1.0):
        g = g0(at.w2) if arm == 1.0 else 1.0 - g0(at.w2)
        clever = 1.0 / g
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
#: The LTMLE fits in sample (``n_folds=1``): a cross-fitted fit of a continuous outcome needs a
#: declared outcome support, which a normal outcome does not have.  Every learner is parametric.
LW_FOLDS = 1
LW_G_BOUNDS = (0.01, 1.0)
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


def fit_longitudinal(frame: pd.DataFrame, *, random_state: int = 0) -> Any:
    """The gated LTMLE ``"sample"`` fit of law LW: correct regressions and tilt classifiers."""
    from cleverly.longitudinal import LTMLE

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LTMLE(
            PLANS,
            reference=REFERENCE_PLAN,
            outcome_learner=LinearOLS(),
            pseudo_learner=LinearOLS(),
            treatment_learner=MainTermsLogistic(),
            n_folds=LW_FOLDS,
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
            mechanism_law="sample",
        )
