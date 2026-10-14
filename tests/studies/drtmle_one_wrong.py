"""Registered evidence for DR-TMLE with one inconsistent nuisance (RM40b).

The subject is complete-data ``DRTMLE`` with ``guard=("Q", "g")`` and the univariate
reduction, on law L of the RM40 plan, where neither conditioning term of Benkeser, Carone,
van der Laan and Gilbert (2017) vanishes:

* ``W1, W2 ~ N(0, 1)``, ``g0(W) = expit(0.6 W1 + 0.3 W2)``;
* ``Qbar0(a, W) = expit(-0.2 + 0.5 a - W1 + 2 W2)``.

The outcome regression a fit uses is the logistic GLM on ``(A, W1)``.  Its limit is not
``Qbar0``, and ``Qbar0 - Qbar`` is not a function of ``g0``, so ``R_{3,n}`` is linear in the
error of the mechanism estimate (``docs/technical-reference/dr-tmle/theorem.md``).  The
mirror control fits the mechanism wrong instead.

The law-M control of :mod:`tests.studies.drtmle_one_wrong_properties` is the randomized
missing-outcome route of Díaz and van der Laan (2017): a known ``g_A = 1/2``, an outcome
recorded with probability ``pi0(a, W) = expit(0.4 + 0.2 a + 0.6 W1 + 0.3 W2)``, the same
outcome law and the same truths.

**The primary scenario** is ``q_wrong_g_known``: the declared mechanism (the H2 case of the
case table), ``ey0``, ``ey1`` and ``ate`` at n = 4,000 over 4,000 replications.  Every other
setting is a property cell, so no conservative or failing setting enters the accuracy grid,
whose 0.8-1.2 sanity band would read a conservative ratio as red.

**Learners** are the investigation's (``reviews/rm40/drdr_sim.py``), moved here so that the
probe is a valid pilot.  The offset learners are deliberate constructed errors, not fitted
learners: each fits the correct GLM and adds a fixed logit offset that shrinks like
``n^(-1/3)`` in a direction that breaks the level sets of its truth.  The reduced regressions
are cubic polynomials in the raw scalar design, which lies in ``[0, 1]``, so nothing
extrapolates.

**Declared before any verdict:** the budgets and seeds of this module and of
:mod:`tests.studies.drtmle_one_wrong_properties`; ``publication_policy="reporting"``; the
red-row reading of the RM40 plan section 5.9, which the properties module states.
"""

from __future__ import annotations

import math
import time
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from cleverly.estimators import DRTMLE
from tests.parallel import STUDY_JOBS
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed
from tests.studies.point_study_helpers import initial_estimates, primary_rows

PRIMARY_REPLICATES = 4_000
PRIMARY_N = 4_000
SEED = 20264001
RESAMPLING_SEED = 2026400101
IMPLEMENTATION = "cleverly-drtmle-one-wrong"
PRIMARY_SCENARIO = "q_wrong_g_known"
ESTIMANDS = ("ey0", "ey1", "ate")
COVARIATES = ["W1", "W2"]
G_BOUNDS = (0.01, 0.99)
#: The known treatment probability of law M, passed with ``delta=``.
LAW_M_TREATMENT = 0.5

#: Law L and law M share the outcome law.
OUTCOME_COEFFICIENTS = (-0.2, 0.5, -1.0, 2.0)  # intercept, A, W1, W2
#: ``logit g0 = 0.6 W1 + 0.3 W2`` (law L), and the covariate part of ``logit pi0`` (law M).
MECHANISM_INDEX = (0.6, 0.3)
#: ``logit pi0(a, W) = 0.4 + 0.2 a + 0.6 W1 + 0.3 W2`` (law M).
RESPONSE_COEFFICIENTS = (0.4, 0.2)  # intercept, A
#: The covariate direction of the perpendicular outcome offset of the mirror control.
PERPENDICULAR = (2.0 / math.sqrt(5.0), 1.0 / math.sqrt(5.0))
#: Logit-offset multipliers of the constructed errors, each times ``n^(-1/3)``.
SLOW_G_OFFSET = 1.0
SLOW_Q_OFFSET = 3.0
SLOW_PI_OFFSET = 2.0
GAUSS_HERMITE_NODES = 120


# ------------------------------------------------------------------------------- the laws
def g0(w1: Any, w2: Any) -> Any:
    """``P(A = 1 | W)`` of law L."""
    return expit(MECHANISM_INDEX[0] * np.asarray(w1) + MECHANISM_INDEX[1] * np.asarray(w2))


def qbar0(a: Any, w1: Any, w2: Any) -> Any:
    """``E[Y | A = a, W]`` of laws L and M."""
    b0, ba, b1, b2 = OUTCOME_COEFFICIENTS
    return expit(b0 + ba * np.asarray(a) + b1 * np.asarray(w1) + b2 * np.asarray(w2))


def pi0(a: Any, w1: Any, w2: Any) -> Any:
    """``P(Delta = 1 | A = a, W)`` of law M."""
    c0, ca = RESPONSE_COEFFICIENTS
    return expit(
        c0
        + ca * np.asarray(a)
        + MECHANISM_INDEX[0] * np.asarray(w1)
        + MECHANISM_INDEX[1] * np.asarray(w2)
    )


def draw_law_l(n: int, seed: int) -> pd.DataFrame:
    """One sample of law L."""
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal(n)
    w2 = rng.standard_normal(n)
    a = rng.binomial(1, g0(w1, w2)).astype(float)
    y = rng.binomial(1, qbar0(a, w1, w2)).astype(float)
    frame = pd.DataFrame({"Y": y, "A": a, "W1": w1, "W2": w2})
    probability = g0(w1, w2)
    frame["p0"] = 1.0 - probability
    frame["p1"] = probability
    return frame


def draw_law_m(n: int, seed: int) -> pd.DataFrame:
    """One sample of law M: a trial with ``A ~ Bernoulli(1/2)`` and a missing outcome."""
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal(n)
    w2 = rng.standard_normal(n)
    a = rng.binomial(1, LAW_M_TREATMENT, n).astype(float)
    delta = rng.binomial(1, pi0(a, w1, w2)).astype(float)
    y = rng.binomial(1, qbar0(a, w1, w2)).astype(float)
    return pd.DataFrame(
        {"Y": np.where(delta == 1.0, y, np.nan), "A": a, "W1": w1, "W2": w2, "Delta": delta}
    )


def gauss_hermite(nodes: int = GAUSS_HERMITE_NODES) -> tuple[np.ndarray, np.ndarray]:
    """Standard-normal Gauss-Hermite nodes and weights."""
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    return x, w / np.sqrt(2.0 * np.pi)


def _grid(nodes: int = GAUSS_HERMITE_NODES) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, w = gauss_hermite(nodes)
    w1, w2 = np.meshgrid(x, x, indexing="ij")
    return w1, w2, np.outer(w, w)


def truth() -> dict[str, float]:
    """The arm means and their difference, by 2-D Gauss-Hermite quadrature."""
    w1, w2, weights = _grid()
    one = float((weights * qbar0(1.0, w1, w2)).sum())
    zero = float((weights * qbar0(0.0, w1, w2)).sum())
    return {"ey0": zero, "ey1": one, "ate": one - zero}


def _logistic_limit(design: list[np.ndarray], target: np.ndarray, weights: np.ndarray) -> Any:
    """The population logistic fit: solve ``E[w x (target - expit(x'b))] = 0`` by Newton."""
    x = np.stack([np.ones_like(target), *design], axis=-1).reshape(-1, len(design) + 1)
    y = np.asarray(target, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    beta = np.zeros(x.shape[1])
    for _ in range(100):
        p = expit(x @ beta)
        hessian = (x * (w * p * (1.0 - p))[:, None]).T @ x
        step = np.linalg.solve(hessian, x.T @ (w * (y - p)))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-13:
            return beta
    raise RuntimeError("the population logistic fit did not converge")


def outcome_limit_l() -> np.ndarray:
    """The limit ``(b0, bA, b1)`` of the wrong outcome GLM on ``(A, W1)`` under law L."""
    w1, w2, weights = _grid()
    g = g0(w1, w2)
    design = [np.stack([np.zeros_like(w1), np.ones_like(w1)]), np.stack([w1, w1])]
    target = np.stack([qbar0(0.0, w1, w2), qbar0(1.0, w1, w2)])
    return np.asarray(_logistic_limit(design, target, np.stack([(1.0 - g), g]) * weights))


def outcome_limit_m() -> np.ndarray:
    """The limit of the wrong outcome GLM on ``(A, W1)`` over the recorded rows of law M."""
    w1, w2, weights = _grid()
    design = [np.stack([np.zeros_like(w1), np.ones_like(w1)]), np.stack([w1, w1])]
    target = np.stack([qbar0(0.0, w1, w2), qbar0(1.0, w1, w2)])
    response = np.stack([pi0(0.0, w1, w2), pi0(1.0, w1, w2)])
    return np.asarray(_logistic_limit(design, target, LAW_M_TREATMENT * response * weights))


def mechanism_limit_mirror() -> np.ndarray:
    """The limit ``(c0, c2)`` of the mirror control's wrong mechanism GLM on ``W2``."""
    w1, w2, weights = _grid()
    return np.asarray(_logistic_limit([w2], g0(w1, w2), weights))


# ---------------------------------------------------------------------- level-set averages
def level_set_mean(direction: tuple[float, float], function: Any, w1: Any, w2: Any) -> Any:
    """``E[function(W) | u(W)]`` at each point, with ``u`` the projection on ``direction``.

    The covariates are independent standard normals, so conditioning on a linear index
    leaves the orthogonal coordinate standard normal; the mean is a 1-D Gauss-Hermite sum
    along the level set.
    """
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    perpendicular = np.array([-d[1], d[0]])
    u = np.asarray(w1) * d[0] + np.asarray(w2) * d[1]
    x, weight = gauss_hermite()
    line1 = u[..., None] * d[0] + x * perpendicular[0]
    line2 = u[..., None] * d[1] + x * perpendicular[1]
    return (function(line1, line2) * weight).sum(axis=-1)


def reduced_outcome_limit(arm: float, beta: np.ndarray, w1: Any, w2: Any) -> Any:
    """``E[Qbar0(a, W) - Qbar(a, W) | index(W)]``, the limit reduced outcome regression.

    The index is ``0.6 W1 + 0.3 W2``: ``g0`` under law L and the covariate part of ``pi0``
    under law M.  The conditioning on ``A = a`` (and ``Delta = 1``) leaves the law of ``W``
    unchanged on a level set of the index, so the conditional mean is the level-set mean.
    """

    def residual(v1: Any, v2: Any) -> Any:
        return qbar0(arm, v1, v2) - expit(beta[0] + beta[1] * arm + beta[2] * v1)

    return level_set_mean(MECHANISM_INDEX, residual, w1, w2)


def mirror_weight(arm: float, gbar: np.ndarray, w1: Any, w2: Any) -> Any:
    """The mirror branch's ``h_a = r_a - pi_a E[r_a | Qbar0] / E[pi_a | Qbar0]``.

    ``pi_1 = g0``, ``pi_0 = 1 - g0`` and ``r_a = (pi_a - pibar_a) / pibar_a``, with ``pibar``
    the limit of the wrong mechanism GLM on ``W2``.  ``Qbar0(a, .)`` is a function of
    ``-W1 + 2 W2`` at either arm.
    """

    def propensity(v1: Any, v2: Any) -> Any:
        g = g0(v1, v2)
        return g if arm == 1.0 else 1.0 - g

    def ratio(v1: Any, v2: Any) -> Any:
        bar = expit(gbar[0] + gbar[1] * v2)
        bar = bar if arm == 1.0 else 1.0 - bar
        return (propensity(v1, v2) - bar) / bar

    index = (OUTCOME_COEFFICIENTS[2], OUTCOME_COEFFICIENTS[3])
    r_mean = level_set_mean(index, ratio, w1, w2)
    p_mean = level_set_mean(index, propensity, w1, w2)
    return ratio(w1, w2) - propensity(w1, w2) * r_mean / p_mean


# ---------------------------------------------------------------------------- the learners
def newton_logistic(x: np.ndarray, y: np.ndarray, ridge: float = 1e-12) -> np.ndarray:
    """Unpenalized logistic maximum likelihood with an intercept, by Newton."""
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(design.shape[1])
    for _ in range(100):
        p = expit(design @ beta)
        hessian = (design * (p * (1.0 - p))[:, None]).T @ design
        step = np.linalg.solve(hessian + ridge * np.eye(len(beta)), design.T @ (y - p))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-11:
            break
    return beta


class OffsetLogistic(BaseEstimator, ClassifierMixin):
    """A logistic GLM on ``columns`` plus a fixed logit offset added at prediction.

    ``offset`` maps a design column to its offset coefficient.  An empty offset is the plain
    GLM.  A non-empty offset is a deliberate constructed error, not a fitted learner.
    """

    def __init__(
        self, columns: tuple[int, ...] = (), offset: tuple[tuple[int, float], ...] = ()
    ) -> None:
        self.columns = columns
        self.offset = offset

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> OffsetLogistic:
        values = np.asarray(design, dtype=float)
        self.beta_ = newton_logistic(values[:, list(self.columns)], np.asarray(target, float))
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, design: Any) -> np.ndarray:
        values = np.asarray(design, dtype=float)
        eta = self.beta_[0] + values[:, list(self.columns)] @ self.beta_[1:]
        for column, coefficient in self.offset:
            eta = eta + coefficient * values[:, column]
        p = expit(eta)
        return np.column_stack([1.0 - p, p])


def _cubic(design: Any) -> np.ndarray:
    x = np.asarray(design, dtype=float).reshape(len(design), -1)[:, 0]
    return np.column_stack([np.ones_like(x), x, x**2, x**3])


class CubicLeastSquares(BaseEstimator, RegressorMixin):
    """Weighted least squares on ``(1, x, x^2, x^3)`` of the scalar design."""

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> CubicLeastSquares:
        basis = _cubic(design)
        y = np.asarray(target, dtype=float)
        weight = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        root = np.sqrt(weight)
        self.coef_ = np.linalg.lstsq(basis * root[:, None], y * root, rcond=None)[0]
        return self

    def predict(self, design: Any) -> np.ndarray:
        return np.asarray(_cubic(design) @ self.coef_)


class CubicLogistic(BaseEstimator, ClassifierMixin):
    """Logistic regression on ``(1, x, x^2, x^3)`` of the scalar design, ridge ``1e-8``."""

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> CubicLogistic:
        basis = _cubic(design)
        y = np.asarray(target, dtype=float)
        weight = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        beta = np.zeros(basis.shape[1])
        beta[0] = logit(np.clip(np.average(y, weights=weight), 1e-6, 1.0 - 1e-6))
        for _ in range(100):
            p = expit(basis @ beta)
            hessian = (basis * (weight * p * (1.0 - p))[:, None]).T @ basis
            step = np.linalg.solve(
                hessian + 1e-8 * np.eye(basis.shape[1]), basis.T @ (weight * (y - p))
            )
            beta = beta + step
            if np.max(np.abs(step)) < 1e-10:
                break
        self.coef_ = beta
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, design: Any) -> np.ndarray:
        p = expit(_cubic(design) @ self.coef_)
        return np.column_stack([1.0 - p, p])


class DeclaredOnly(BaseEstimator, ClassifierMixin):
    """A treatment learner that must never run: the declared mechanism replaces it."""

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> DeclaredOnly:
        raise AssertionError("a fit on a declared mechanism fitted a treatment learner")

    def predict_proba(self, design: Any) -> Any:
        raise AssertionError("a fit on a declared mechanism fitted a treatment learner")


def slow(n: int) -> float:
    """``n^(-1/3)``, the rate of every constructed error."""
    return float(n ** (-1.0 / 3.0))


# --------------------------------------------------------------------------- the settings
@dataclass(frozen=True)
class Setting:
    """One estimator configuration.

    Parameters
    ----------
    law : str
        ``"L"`` (complete outcome) or ``"M"`` (randomized missing outcome).
    outcome : str
        ``"wrong"`` (GLM on ``A, W1``) or ``"slow"`` (correct GLM plus the perpendicular
        offset).
    mechanism : str
        Law L: ``"glm"`` (correct GLM), ``"known"`` (declared), ``"slow"`` (correct GLM plus
        the ``W2`` offset) or ``"wrong"`` (GLM on ``W2``).  Law M: ``"slow"`` is the response
        mechanism, correct GLM plus ``2 n^(-1/3) W2``.
    cross_fit : bool
        Whether the nuisances are cross-fitted over the shipped default folds.
    branch : str
        Which conditioning term the remainder proxy reads: ``"outcome"``, ``"mirror"`` or
        ``"response"``.
    """

    law: str
    outcome: str
    mechanism: str
    cross_fit: bool = False
    branch: str = "outcome"


SETTINGS: dict[str, Setting] = {
    "q_wrong_g_glm": Setting("L", "wrong", "glm"),
    "q_wrong_g_known": Setting("L", "wrong", "known"),
    "q_wrong_g_glm_cf": Setting("L", "wrong", "glm", cross_fit=True),
    "q_wrong_g_slow": Setting("L", "wrong", "slow"),
    "q_slow_g_wrong": Setting("L", "slow", "wrong", branch="mirror"),
    "mar_q_wrong_pi_slow": Setting("M", "wrong", "slow", branch="response"),
}


def outcome_learner(setting: Setting, n: int) -> OffsetLogistic:
    """The outcome learner on the design ``[A, W1, W2]``."""
    if setting.outcome == "wrong":
        return OffsetLogistic(columns=(0, 1))
    delta = SLOW_Q_OFFSET * slow(n)
    return OffsetLogistic(
        columns=(0, 1, 2), offset=((1, delta * PERPENDICULAR[0]), (2, delta * PERPENDICULAR[1]))
    )


def mechanism_learner(setting: Setting, n: int) -> BaseEstimator:
    """The treatment learner on ``[W1, W2]``, or law M's response learner on ``[A, W1, W2]``."""
    if setting.law == "M":
        return OffsetLogistic(columns=(0, 1, 2), offset=((2, SLOW_PI_OFFSET * slow(n)),))
    if setting.mechanism == "known":
        return DeclaredOnly()
    if setting.mechanism == "glm":
        return OffsetLogistic(columns=(0, 1))
    if setting.mechanism == "slow":
        return OffsetLogistic(columns=(0, 1), offset=((1, SLOW_G_OFFSET * slow(n)),))
    return OffsetLogistic(columns=(1,))


def fit(frame: pd.DataFrame, name: str) -> Any:
    """Fit the named setting on one sample."""
    setting = SETTINGS[name]
    n = len(frame)
    common: dict[str, Any] = {
        "guard": ("Q", "g"),
        "reduction": "univariate",
        "outcome_learner": outcome_learner(setting, n),
        "reduced_outcome_learner": CubicLeastSquares(),
        "reduced_treatment_learner": CubicLogistic(),
        "cross_fit": setting.cross_fit,
        "estimands": ESTIMANDS,
        "simultaneous": False,
        "g_bounds": G_BOUNDS,
        "random_state": 0,
    }
    with warnings.catch_warnings():
        # Non-converged fits stay in their cells (RM40 plan 5.8); the warning is the shipped
        # estimator's, and ``fit_record`` publishes the exit of every fit.
        warnings.simplefilter("ignore")
        if setting.law == "M":
            estimator = DRTMLE(
                randomized=True, missingness_learner=mechanism_learner(setting, n), **common
            )
            return estimator.fit(
                frame,
                outcome="Y",
                treatment="A",
                covariates=COVARIATES,
                delta="Delta",
                treatment_probabilities=np.full(n, LAW_M_TREATMENT),
            ).single()
        known = setting.mechanism == "known"
        estimator = DRTMLE(treatment_learner=mechanism_learner(setting, n), **common)
        return estimator.fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=COVARIATES,
            treatment_probabilities={0.0: "p0", 1.0: "p1"} if known else None,
        ).single()


# --------------------------------------------------------------------------- diagnostics
@dataclass(frozen=True)
class FitRecord:
    """What one fit gives every row the study writes, beyond the estimate."""

    remainder: dict[str, float]
    projected_std_error: dict[str, float]
    converged: bool
    rounds: int
    exit_reason: str
    seconds: float


_LIMITS: dict[str, np.ndarray] = {}


def limits() -> dict[str, np.ndarray]:
    """The population limits the remainder proxies read, computed once per process."""
    if not _LIMITS:
        _LIMITS.update(
            outcome_l=outcome_limit_l(),
            outcome_m=outcome_limit_m(),
            mechanism_mirror=mechanism_limit_mirror(),
        )
    return _LIMITS


def remainder_proxy(result: Any, frame: pd.DataFrame, setting: Setting) -> dict[str, float]:
    """The in-sample proxy of the conditioning term this setting exposes, by estimand.

    ``outcome`` (``R_{3,n}``, Appendix A): ``mean_i[F_a (pi_a - pihat_a) / pi_a]`` with
    ``F_a = Qbar0(a, .) - Qhat(a, .) - Qr0_a`` and ``Qr0_a`` the limit reduced regression.
    ``mirror`` (``R~_{5,n}``, Appendix B): ``mean_i[h_a (Qbar0(a, .) - Qhat(a, .))]``.
    ``response`` (eq. (21) of Díaz and van der Laan 2017): ``mean_i[F_a (pi0_a - pihat_a) /
    pihat_a]`` with the response mechanism in place of the treatment mechanism.  Each term
    enters the estimate with a plus sign, so its mean predicts the bias.
    """
    w1 = frame["W1"].to_numpy(dtype=float)
    w2 = frame["W2"].to_numpy(dtype=float)
    nuisance = result.nuisance
    out: dict[str, float] = {}
    for arm, name in ((1.0, "ey1"), (0.0, "ey0")):
        q_hat = np.asarray(nuisance.outcome.arms[arm], dtype=float)
        if setting.branch == "mirror":
            weight = mirror_weight(arm, limits()["mechanism_mirror"], w1, w2)
            out[name] = float(np.mean(weight * (qbar0(arm, w1, w2) - q_hat)))
            continue
        if setting.branch == "response":
            column = list(nuisance.arms).index(arm)
            estimate = np.asarray(nuisance.missingness, dtype=float)[:, column]
            truth_ = pi0(arm, w1, w2)
            beta = limits()["outcome_m"]
            residual = qbar0(arm, w1, w2) - q_hat - reduced_outcome_limit(arm, beta, w1, w2)
            out[name] = float(np.mean(residual * (truth_ - estimate) / estimate))
            continue
        g_true = g0(w1, w2)
        g_hat = np.asarray(nuisance.propensity.arm(1.0), dtype=float)
        if arm == 0.0:
            g_true, g_hat = 1.0 - g_true, 1.0 - g_hat
        beta = limits()["outcome_l"]
        residual = qbar0(arm, w1, w2) - q_hat - reduced_outcome_limit(arm, beta, w1, w2)
        out[name] = float(np.mean(residual * (g_true - g_hat) / g_true))
    out["ate"] = out["ey1"] - out["ey0"]
    return out


def projected_std_errors(result: Any, frame: pd.DataFrame, setting: Setting) -> dict[str, float]:
    """The standard error of each curve's residual on the mechanism GLM's score columns.

    Defined for a correctly specified parametric mechanism only (``"glm"``), where RM40 plan
    2.4 predicts the estimator's curve is that residual.  ``nan`` elsewhere.
    """
    if setting.law != "L" or setting.mechanism != "glm":
        return dict.fromkeys(ESTIMANDS, math.nan)
    n = len(frame)
    g_hat = np.asarray(result.nuisance.propensity.arm(1.0), dtype=float)
    design = np.column_stack(
        [np.ones(n), frame["W1"].to_numpy(dtype=float), frame["W2"].to_numpy(dtype=float)]
    )
    score = design * (frame["A"].to_numpy(dtype=float) - g_hat)[:, None]
    out = {}
    for name in ESTIMANDS:
        curve = np.asarray(result.estimates[name].influence_curve, dtype=float)
        centred = curve - curve.mean()
        coefficients = np.linalg.lstsq(score, centred, rcond=None)[0]
        out[name] = float(np.std(centred - score @ coefficients) / np.sqrt(n))
    return out


def fit_record(name: str, frame: pd.DataFrame) -> tuple[Any, FitRecord]:
    """Fit one setting and read its diagnostics."""
    setting = SETTINGS[name]
    started = time.perf_counter()
    result = fit(frame, name)
    seconds = time.perf_counter() - started
    reduction = result.repeats[0].fluctuations["mean"].reduction
    return result, FitRecord(
        remainder=remainder_proxy(result, frame, setting),
        projected_std_error=projected_std_errors(result, frame, setting),
        converged=bool(reduction.converged),
        rounds=int(reduction.n_outer),
        exit_reason=str(reduction.exit_reason),
        seconds=float(seconds),
    )


# ---------------------------------------------------------------------- the registration
#: Filled by :mod:`tests.studies.drtmle_one_wrong_properties`, which this module cannot
#: import; ``tests/unit/test_drtmle_one_wrong_design.py`` pins the two equal.
PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "one_wrong_inference": (
        "q_wrong_g_glm__ate__conservative",
        "q_wrong_g_glm__ey0__valid",
        "q_wrong_g_glm__ey1__valid",
        "q_wrong_g_glm_cf__ate__conservative",
        "q_wrong_g_glm_cf__ey0__valid",
        "q_wrong_g_glm_cf__ey1__valid",
        "q_wrong_g_known__ate__valid",
        "q_wrong_g_known__ey0__valid",
        "q_wrong_g_known__ey1__valid",
        *(f"q_wrong_g_slow_n{n}__ate__first_order" for n in (1_000, 4_000, 16_000)),
        "q_wrong_g_slow__ate__growth",
        *(f"q_slow_g_wrong_n{n}__ate__first_order" for n in (1_000, 4_000, 16_000)),
        "q_slow_g_wrong__ate__growth",
        "mar_q_wrong_pi_slow__ey0__first_order",
        "mar_q_wrong_pi_slow__ey1__first_order",
    ),
    "one_wrong_reported": (
        *(
            f"{label}_n{n}__{arm}__reported"
            for label in ("q_wrong_g_slow", "q_slow_g_wrong")
            for n in (1_000, 4_000, 16_000)
            for arm in ("ey0", "ey1")
        ),
        "mar_q_wrong_pi_slow__ate__reported",
    ),
}

STUDY = StudyRecord(
    name="DR-TMLE with one inconsistent nuisance",
    slug="drtmle-one-wrong",
    artifacts=ROOT / "tests" / "canonical" / "drtmle_one_wrong",
    document="docs/technical-reference/method-evidence/dr-tmle-one-inconsistent-nuisance.md",
    anchor="dr-tmle-with-one-inconsistent-nuisance",
    scenarios={PRIMARY_SCENARIO: ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation=IMPLEMENTATION,
    reference=None,
    modules=(
        "tests/studies/drtmle_one_wrong.py",
        "tests/studies/drtmle_one_wrong_properties.py",
        "tests/studies/point_study_helpers.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.drtmle_one_wrong",
    properties_module="tests.studies.drtmle_one_wrong_properties",
    property_cells=PROPERTY_CELLS,
    calibration_efficiency_ratio=False,
    publication_policy="reporting",
)

CONFIGURATION = {
    "construction": "complete-data and randomized missing-outcome DR-TMLE",
    "guard": ["Q", "g"],
    "reduction": "univariate",
    "g_bounds": list(G_BOUNDS),
    "simultaneous_intervals": False,
    "law_L": "W1, W2 ~ N(0, 1); g0 = expit(0.6 W1 + 0.3 W2); "
    "Qbar0(a, W) = expit(-0.2 + 0.5 a - W1 + 2 W2)",
    "law_M": "law L's covariates and outcome; A ~ Bernoulli(0.5), known; "
    "pi0(a, W) = expit(0.4 + 0.2 a + 0.6 W1 + 0.3 W2); Y recorded when Delta = 1",
    "outcome_learners": {
        "wrong": "logistic GLM on (A, W1)",
        "slow": "logistic GLM on (A, W1, W2) plus the logit offset "
        "3 n^(-1/3) (2 W1 + W2) / sqrt(5), a constructed error",
    },
    "mechanism_learners": {
        "glm": "logistic GLM on (W1, W2)",
        "known": "the declared g0, by treatment_probabilities",
        "slow": "logistic GLM on (W1, W2) plus the logit offset n^(-1/3) W2, a constructed error",
        "wrong": "logistic GLM on W2",
        "law_M_response": "logistic GLM on (A, W1, W2) plus the logit offset 2 n^(-1/3) W2, "
        "a constructed error",
    },
    "reduced_learners": "cubic polynomials in the raw scalar design: least squares for the "
    "reduced outcome regression, Newton logistic with ridge 1e-8 for the reduced mechanism",
    "cross_fit": "False, except the q_wrong_g_glm_cf cells (the shipped default folds)",
    "settings": {name: vars(setting) for name, setting in SETTINGS.items()},
}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one primary sample from an explicit seed for the published-seed audit."""
    if scenario not in STUDY.scenarios:
        raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")
    return draw_law_l(n, seed), truth()


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one replication from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def cleverly_rows(
    result: Any, truth_: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    """Convert one fit to the shared replication schema."""
    return primary_rows(
        result=result,
        truth=truth_,
        implementation=IMPLEMENTATION,
        scenario=scenario,
        replicate=replicate,
        estimands=ESTIMANDS,
        initials=initial_estimates(result, ESTIMANDS),
    )


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> list[dict[str, Any]]:
    scenario, replicate, _ = key
    frame, truth_ = drawn
    return cleverly_rows(fit(frame, scenario), truth_, scenario, replicate)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Draw and fit every declared primary replication."""
    outcomes = replication.fit_primary(
        [(PRIMARY_SCENARIO, replicate, n) for replicate in range(replicates)],
        _replicate,
        draw=_draw,
        rows_at=None,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    built = pd.DataFrame([row for rows in outcomes.kept for row in rows])
    return replication.result(outcomes, built.loc[:, list(REPLICATE_COLUMNS)])
