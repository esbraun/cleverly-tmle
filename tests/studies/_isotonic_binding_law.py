r"""Law, learners, fits and quadrature truths of ``isotonic-binding`` (X76 part (g)).

``LTMLE(isotonic=True)`` with ``survival_model="iterated_mean"`` projects each regimen's risk
curve onto a non-decreasing one by unit-weight isotonic regression over the horizons
(``cleverly.longitudinal.estimator._project_fits``).  Only the levels move: the influence curve,
and so every standard error, stays the fit's own.  With two horizons and
:math:`D = \hat\theta_2 - \hat\theta_1`, the projection moves each level by :math:`D^-/2`.

**When the projection binds** (Fix-8 plan section 2.7).  Horizon :math:`k`'s level is the mean
over all rows of its targeted node-1 regression :math:`Q^*_{k1}`, and the node-1 fluctuation
solves :math:`P_n[H_1 (\tilde Y_k - Q^*_{k1})] = 0`, with :math:`H_1` the node-1 clever covariate
(zero off the followers) and :math:`\tilde Y_k` the node-1 pseudo-outcome.  Differencing the
horizons gives, up to the two node-1 score residuals,

.. math::

    D = P_n[H_1\, dY] + P_n[(1 - H_1)\, dQ^*], \qquad dY = (1 - Y_1) Q^*_{22} \ge 0,
    \qquad dQ^* = Q^*_{21} - Q^*_{11}.

:func:`score_identity` computes both terms from a fit.  The first term is nonnegative.  When the
node regressions are saturated the second term is zero, so ``D >= 0`` and nothing moves,
whatever the event count.  Here the node-1 regression of horizon 1 is the logistic outcome
learner and that of horizon 2 the linear pseudo learner, so :math:`dQ^*` varies with ``W``.  The
second term is then noise of order :math:`n^{-1/2}` and the projection binds when the true
increment is small against it.

Law ``LG2`` (plan section 5.8), two nodes, :math:`W \sim N(0, 1)`:

* :math:`A_1 \sim \mathrm{Bern}(\mathrm{expit}(g_s W))`, retention
  :math:`C_1 \sim \mathrm{Bern}(\mathrm{expit}(2.5 - 0.5 W))`;
* :math:`Y_1 \sim \mathrm{Bern}(\mathrm{expit}(-1.5 + 1.6 W - 0.5 A_1))`;
* :math:`L_2 \sim \mathrm{Bern}(\mathrm{expit}(-0.3 + 0.6 W + 0.4 A_1))`;
* :math:`A_2 \sim \mathrm{Bern}(\mathrm{expit}(g_s W + 0.8 L_2 + 0.5 A_1))`, retention
  :math:`C_2 \sim \mathrm{Bern}(\mathrm{expit}(2.0 - 0.5 W + 0.5 L_2))`;
* :math:`Y_2 \sim \mathrm{Bern}(0.01\, \mathrm{expit}(-0.5 + 1.2 W^2 - 0.4 A_2))`.

``g_s = 1.5`` is the declared law; ``g_s = 2.0`` (``LG2-gs2``) is the reported larger-move law.
Later columns are missing after censoring or an event.  The truths come from 120-node
Gauss-Hermite quadrature: never (0.26074, 0.26493) and always (0.19717, 0.20106), strictly
increasing.  Horizon 1 is correctly specified: its outcome, treatment and censoring
regressions are logistic in ``W`` and the learner is the logistic class.

**The learner.**  :func:`logistic` is the one pinned logistic learner, unpenalized and solved
to ``tol=1e-13`` (plan section 3.8).  At sklearn's default ``tol=1e-4`` the node-1 score is left
unsolved by enough to move a level on a law whose saturated fit cannot bind (Fix-8
``progress.md``, Part G entry), so the study gates nothing that changes between ``1e-10`` and
``1e-13`` (the design test).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import expit
from sklearn.linear_model import LinearRegression, LogisticRegression

#: The declared law and the reported larger-move law.
LAW = "LG2"
LARGE_MOVE_LAW = "LG2-gs2"
#: The node-2 hazard scale, the outcome slope in ``W`` at node 1, and each law's treatment slope.
HAZARD_SCALE = 0.01
OUTCOME_SLOPE = 1.6
TREATMENT_SLOPE = {LAW: 1.5, LARGE_MOVE_LAW: 2.0}

#: The regimens, the reference and the horizons of every fit.
REGIMENS = {"never": 0, "always": 1}
REFERENCE = "never"
HORIZONS = (1, 2)
#: The fit's settings (plan sections 5.0 and 5.8).
G_BOUNDS = (1e-8, 1.0)
MAX_ITER = 100
TOL = 1e-10
#: The pinned logistic learner's tolerance, and the tolerance the design test refits at.
LOGISTIC_TOL = 1e-13
REFIT_TOL = 1e-10
LOGISTIC_MAX_ITER = 100_000
QUADRATURE_NODES = 120

COLUMNS = ("W", "A1", "C1", "Y1", "L2", "A2", "C2", "Y2")


def name(regimen: str, horizon: int) -> str:
    """The reported name of a regimen's risk at a horizon."""
    return f"risk_regimen[{regimen} @ t={horizon}]"


# ---------------------------------------------------------------------------- the law
def p_event_1(w: Any, a: Any) -> Any:
    """``P(Y_1 = 1 | W, A_1, C_1 = 1)``."""
    return expit(-1.5 + OUTCOME_SLOPE * np.asarray(w) - 0.5 * np.asarray(a))


def p_event_2(w: Any, a: Any) -> Any:
    """``P(Y_2 = 1 | W, A_2, at risk and observed at node 2)``."""
    w = np.asarray(w)
    return HAZARD_SCALE * expit(-0.5 + 1.2 * w * w - 0.4 * np.asarray(a))


def p_treatment_1(w: Any, law: str = LAW) -> Any:
    """``P(A_1 = 1 | W)``."""
    return expit(TREATMENT_SLOPE[law] * np.asarray(w))


def p_retention_1(w: Any) -> Any:
    """``P(C_1 = 1 | W)``, with ``C = 1`` observed."""
    return expit(2.5 - 0.5 * np.asarray(w))


def draw(n: int, seed: int, law: str = LAW) -> pd.DataFrame:
    """One sample of ``n`` rows of ``law``, drawn node by node from one generator."""
    rng = np.random.default_rng(seed)
    slope = TREATMENT_SLOPE[law]
    w = rng.normal(size=n)
    a1 = rng.binomial(1, p_treatment_1(w, law))
    c1 = rng.binomial(1, p_retention_1(w))
    y1 = rng.binomial(1, p_event_1(w, a1)).astype(float)
    l2 = rng.binomial(1, expit(-0.3 + 0.6 * w + 0.4 * a1)).astype(float)
    a2 = rng.binomial(1, expit(slope * w + 0.8 * l2 + 0.5 * a1))
    c2 = rng.binomial(1, expit(2.0 - 0.5 * w + 0.5 * l2))
    y2 = rng.binomial(1, p_event_2(w, a2)).astype(float)
    frame = pd.DataFrame(
        {
            "W": w,
            "A1": a1.astype(float),
            "C1": c1.astype(float),
            "Y1": y1,
            "L2": l2,
            "A2": a2.astype(float),
            "C2": c2.astype(float),
            "Y2": y2,
        }
    )
    frame.loc[frame["C1"] == 0, ["Y1", "L2", "A2", "C2", "Y2"]] = np.nan
    frame.loc[frame["Y1"] == 1, ["L2", "A2", "C2", "Y2"]] = np.nan
    frame.loc[frame["C2"] == 0, ["Y2"]] = np.nan
    return frame


def truths(nodes: int = QUADRATURE_NODES) -> dict[str, float]:
    """Every regimen's risk at both horizons, by Gauss-Hermite quadrature.

    The truths do not depend on the treatment slope, so both laws share them.
    """
    x, weight = np.polynomial.hermite_e.hermegauss(nodes)
    weight = weight / weight.sum()
    out: dict[str, float] = {}
    for regimen, arm in REGIMENS.items():
        p1 = p_event_1(x, arm)
        first = float(np.sum(weight * p1))
        out[name(regimen, 1)] = first
        out[name(regimen, 2)] = first + float(np.sum(weight * (1.0 - p1) * p_event_2(x, arm)))
    return out


TRUTH: dict[str, float] = truths()
#: The gated cell's estimand.
ESTIMAND = name("never", 1)


@cache
def efficiency_sd(regimen: str = "never") -> float:
    r"""``sqrt(E[D*^2])`` of a regimen's horizon-1 risk, by adaptive quadrature.

    :math:`D^* = \frac{1\{A_1 = a, C_1 = 1\}}{g_1(a | W) c_1(W)} (Y_1 - p_1(W, a)) + p_1(W, a)
    - \theta_1(a)`, so :math:`E[D^{*2}] = E[p_1 (1 - p_1) / (g_1 c_1)] + \mathrm{Var}\, p_1`.
    """
    arm = REGIMENS[regimen]
    # The normal density is below 1e-100 beyond this, where the ratio would overflow.
    BOUND = 20.0

    def density(w: float) -> float:
        return float(np.exp(-0.5 * w * w) / np.sqrt(2.0 * np.pi))

    def treated(w: float) -> float:
        g = float(p_treatment_1(w))
        return g if arm == 1 else 1.0 - g

    def residual(w: float) -> float:
        p = float(p_event_1(w, arm))
        return density(w) * p * (1.0 - p) / (treated(w) * float(p_retention_1(w)))

    first = TRUTH[name(regimen, 1)]

    def spread(w: float) -> float:
        return density(w) * (float(p_event_1(w, arm)) - first) ** 2

    total = quad(residual, -BOUND, BOUND, epsabs=1e-13, epsrel=1e-12, limit=400)[0]
    total += quad(spread, -BOUND, BOUND, epsabs=1e-13, epsrel=1e-12, limit=400)[0]
    return float(np.sqrt(total))


# ---------------------------------------------------------------------------- the learners
def logistic(tol: float = LOGISTIC_TOL) -> LogisticRegression:
    """The study's one logistic learner: unpenalized, solved to ``tol`` (plan section 3.8)."""
    return LogisticRegression(C=np.inf, max_iter=LOGISTIC_MAX_ITER, tol=tol)


@dataclass(frozen=True)
class Setting:
    """One fit configuration.

    Parameters
    ----------
    law : str
        :data:`LAW` or :data:`LARGE_MOVE_LAW`.
    tol : float
        The logistic learner's tolerance.
    isotonic : bool
        Whether the fit projects its curves.
    """

    law: str = LAW
    tol: float = LOGISTIC_TOL
    isotonic: bool = True


SETTINGS: dict[str, Setting] = {
    "lg2": Setting(),
    "lg2_gs2": Setting(law=LARGE_MOVE_LAW),
}


def estimator(tol: float = LOGISTIC_TOL, *, isotonic: bool = True, **overrides: Any) -> Any:
    """The study's ``LTMLE``: in sample, iterated mean, pinned logistic and linear learners."""
    from cleverly.longitudinal import LTMLE

    options: dict[str, Any] = {
        "reference": REFERENCE,
        "n_folds": 1,
        "random_state": 0,
        "isotonic": isotonic,
        "simultaneous": False,
        "survival_model": "iterated_mean",
        "g_bounds": G_BOUNDS,
        "max_iter": MAX_ITER,
        "tol": TOL,
        "outcome_learner": logistic(tol),
        "pseudo_learner": LinearRegression(),
        "treatment_learner": logistic(tol),
        "censoring_learner": logistic(tol),
    }
    options.update(overrides)
    return LTMLE(dict(REGIMENS), **options)


def fit(
    frame: pd.DataFrame,
    tol: float = LOGISTIC_TOL,
    *,
    isotonic: bool = True,
    quiet: bool = True,
    **overrides: Any,
) -> Any:
    """Fit one sample; ``quiet=False`` leaves every warning to the caller's filters.

    A quiet fit silences the package's data warnings (a constant node-2 target, a sparse
    influence curve), which the study reads as columns rather than as messages.
    """
    with warnings.catch_warnings():
        if quiet:
            warnings.simplefilter("ignore")
        return estimator(tol, isotonic=isotonic, **overrides).fit(
            frame,
            outcome=["Y1", "Y2"],
            treatment=["A1", "A2"],
            baseline=["W"],
            time_varying=[[], ["L2"]],
            censoring=["C1", "C2"],
        )


def fit_counted(frame: pd.DataFrame, tol: float = LOGISTIC_TOL) -> tuple[Any, int]:
    """Fit one sample and count the logistic fits that stopped at ``max_iter``."""
    from sklearn.exceptions import ConvergenceWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("ignore")
        warnings.simplefilter("always", ConvergenceWarning)
        result = fit(frame, tol, quiet=False)
    return result, sum(issubclass(item.category, ConvergenceWarning) for item in caught)


# ---------------------------------------------------------------------------- reading a fit
@dataclass(frozen=True)
class Level:
    """One regimen's level at one horizon, before and after the projection.

    Parameters
    ----------
    projected : float
        The reported level.
    unprojected : float
        The level before the projection; equal to ``projected`` when nothing moved.
    std_error : float
        The plug-in standard error, which the projection does not change.
    has_se : bool
        Whether the estimate reports ``std_error`` (no constant node-2 target).
    """

    projected: float
    unprojected: float
    std_error: float
    has_se: bool

    @property
    def move(self) -> float:
        """``abs(projected - unprojected)`` in standard errors of the level."""
        return abs(self.projected - self.unprojected) / self.std_error


def levels(result: Any) -> dict[tuple[str, int], Level]:
    """Every regimen's level at both horizons, read from one fit."""
    moved = {(regimen, horizon): before for regimen, _, horizon, before, _ in result.isotonic_moves}
    out: dict[tuple[str, int], Level] = {}
    for regimen in REGIMENS:
        for horizon in HORIZONS:
            estimate = result.estimates[name(regimen, horizon)]
            try:
                _ = estimate.std_error
                has_se = True
            except Exception:
                has_se = False
            after = float(estimate.psi)
            out[(regimen, horizon)] = Level(
                projected=after,
                unprojected=float(moved.get((regimen, horizon), after)),
                std_error=float(estimate.plugin_std_error),
                has_se=has_se,
            )
    return out


@dataclass(frozen=True)
class ScoreIdentity:
    """The two terms of the plan 2.7 identity for one regimen, and the levels they explain.

    Parameters
    ----------
    followers : float
        ``P_n[H_1 dY]``, nonnegative.
    others : float
        ``P_n[(1 - H_1) dQ*]``.
    residual : float
        The difference of the two horizons' node-1 score residuals,
        ``P_n[H_1 (Y~_2 - Q*_21)] - P_n[H_1 (Y~_1 - Q*_11)]``, so that
        ``difference = followers - residual + others`` exactly.
    difference : float
        ``theta_2 - theta_1`` of the unprojected levels (:func:`levels`).
    """

    followers: float
    others: float
    residual: float
    difference: float


def _first_step(result: Any, regimen: str, horizon: int) -> Any:
    from cleverly.longitudinal.estimator import _fit_key

    fit_ = result.fits[_fit_key(regimen, None, horizon, True)]
    return fit_, next(step for step in fit_.steps if step.time == 1)


def initial_level(result: Any, regimen: str = "never", horizon: int = 1) -> float:
    """The untargeted plug-in of a level: the mean of its node-1 initial regression."""
    fit_, step = _first_step(result, regimen, horizon)
    return float(np.average(step.initial, weights=fit_.obs_weights))


def score_identity(
    result: Any, regimen: str = "never", *, indicator: bool = False
) -> ScoreIdentity:
    """The terms of ``D = P_n[H_1 dY] + P_n[(1 - H_1) dQ*]`` from a fit's node-1 steps.

    ``H_1`` is the fit's node-1 clever covariate, zero off the node-1 followers, and
    ``dY = Y~_2 - Y~_1`` is the difference of the two horizons' node-1 pseudo-outcomes on the
    followers, ``(1 - Y_1) Q*_22``.  The projection moves only the reported levels, not the
    steps, so a projected fit reads the same terms as the unprojected one; ``difference`` reads
    the unprojected levels.  ``indicator=True`` replaces ``H_1`` by the follower indicator, the
    mutation the witness test shows breaks the identity.
    """
    level = levels(result)
    fit1, step1 = _first_step(result, regimen, 1)
    _, step2 = _first_step(result, regimen, 2)
    weight = fit1.obs_weights
    clever = np.where(step1.trained_on, step1.clever, 0.0)
    if not np.array_equal(clever, np.where(step2.trained_on, step2.clever, 0.0)):
        raise ValueError("the two horizons' node-1 clever covariates differ")
    if indicator:
        clever = step1.trained_on.astype(float)
    on = step1.trained_on
    pseudo1 = np.where(on, np.nan_to_num(step1.pseudo_outcome), 0.0)
    pseudo2 = np.where(on, np.nan_to_num(step2.pseudo_outcome), 0.0)
    delta_q = step2.targeted - step1.targeted
    n = len(clever)
    followers = float(np.sum(weight * clever * (pseudo2 - pseudo1))) / n
    others = float(np.sum(weight * (1.0 - clever) * delta_q)) / n
    residual = (
        float(np.sum(weight * clever * (pseudo2 - step2.targeted)))
        - float(np.sum(weight * clever * (pseudo1 - step1.targeted)))
    ) / n
    return ScoreIdentity(
        followers=followers,
        others=others,
        residual=residual,
        difference=level[(regimen, 2)].unprojected - level[(regimen, 1)].unprojected,
    )
