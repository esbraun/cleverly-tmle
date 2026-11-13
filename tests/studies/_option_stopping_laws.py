r"""Laws, learners and fits shared by the two declared stopping-rule studies (X76 part (a)).

``option-stopping-point`` (:mod:`tests.studies.canonical_option_stopping_point`) and
``option-stopping-ltmle`` (:mod:`tests.studies.canonical_option_stopping_ltmle`) measure
``stopping="scaled_var"`` and ``stopping="sample_size"``.  Under either rule the targeting walk
stops before the score equation is solved, and the reported interval adds the remaining score
to its half-width (``ParameterEstimate.rule_widened``; Fix-8 plan section 2.1).  The laws:

=======  =====================================================================================
law      construction
=======  =====================================================================================
``L1``   the ``canonical-tmle`` binary and continuous laws
         (:func:`tests.studies.canonical_tmle.scenario_dgp`), with that study's learners
``L2``   contrast residual: :math:`W_1, W_2 \sim N(0, 1)`,
         :math:`A \sim \mathrm{Bern}(\mathrm{expit}(0.4 W_1 - 0.3 W_2))`,
         :math:`Y = 4 W_1 + 0.5 W_2 + 0.1 A + N(0, 1)`.  EY1 = ATE = 0.1.  Linear outcome and
         unpenalized logistic treatment regressions, both correct
``LN``   necessity: :math:`W_1, W_2 \sim N(0, 1)`, :math:`V = W_1^2 - 1`,
         :math:`A \sim \mathrm{Bern}(\mathrm{clip}(\mathrm{expit}(0.2 + 0.5 W_1 - 0.3 W_2 +
         0.5 V), 0.03, 0.97))`, :math:`Y = W_1 + 0.5 W_2 + V + 0.1 A + 3 N(0, 1)`.  EY1 = 0.1
         exactly.  The outcome regression is linear in :math:`(A, W_1, W_2)` and omits the
         confounding term :math:`V`, so the initial score is :math:`O(1)` with a fixed sign; the
         treatment regression on :math:`(W_1, W_2, V)` is correct
``L3``   the ``canonical-ltmle`` censored two-node law, with that study's learners
``L4``   the ``point-survival-curve`` five-visit survival law, one-step hazard walk, with that
         study's learners; its ``mechanism_correct`` configuration (intercept-only hazard,
         correct mechanisms) for the necessity cell
=======  =====================================================================================

**The threshold scale.**  The necessity cells multiply every rule threshold by ``K`` for one fit
(:func:`threshold_scale`), which patches
:func:`cleverly.fluctuation._score.rule_thresholds`, the one function that computes them (its
only caller is :func:`~cleverly.fluctuation._score.rule_ratio_at`).  A walk that starts outside
``K`` times the threshold stops near it, so the residual is close to ``K c_n`` standard errors,
with :math:`c_n = 1/\min(\log n, 10)` (:func:`unit_cap`).  The design test checks that the patch
reaches the fit.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline

import cleverly.fluctuation._score as score_module
from cleverly.utils.bounds import expit
from tests.studies import canonical_tmle

#: The size of every point and longitudinal cell (plan 5.1, 5.2).
N = 2_000
#: Targeting settings (plan 5.0): ``max_iter=100``, 200 for a one-step walk, ``tol=1e-10``.
TOL = 1e-10
MAX_ITER = 100
ONE_STEP_MAX_ITER = 200

RULES = ("scaled_var", "sample_size")
TARGETINGS = ("iterative", "one_step")
RULE_LABELS = {"scaled_var": "sv", "sample_size": "ss", "relative": "rel"}
TARGETING_LABELS = {"iterative": "newton", "one_step": "one_step"}

#: The necessity cells' threshold scale (plan 5.1: K = 1, 4, 6, 8 gave plain coverage 0.940,
#: 0.9275, 0.895, 0.8425 on 400 pilot draws of LN).
NECESSITY_SCALE = 8.0

#: The truth of every L2 and LN parameter, exact by construction.
L2_TRUTH = {"ey1": 0.1, "ate": 0.1}
LN_TRUTH = {"ey1": 0.1}
#: LN's outcome noise SD.
LN_SIGMA = 3.0
#: LN's propensity clip.
LN_CLIP = (0.03, 0.97)


def unit_cap(n: int) -> float:
    """``c_n = 1 / min(log n, 10)``: a unit-weight arm mean's residual cap under ``scaled_var``."""
    return 1.0 / min(math.log(n), 10.0)


@contextmanager
def threshold_scale(factor: float) -> Iterator[None]:
    """Multiply every declared-rule threshold by ``factor`` inside the block.

    The patch replaces :func:`cleverly.fluctuation._score.rule_thresholds` on its module, where
    :func:`~cleverly.fluctuation._score.rule_ratio_at` reads it, and restores it on exit.  At
    ``factor == 1`` it changes nothing.
    """
    original = score_module.rule_thresholds
    if factor == 1.0:
        yield
        return

    def scaled(contributions: Any, stop: Any, stopping: str) -> Any:
        return factor * original(contributions, stop, stopping)

    score_module.rule_thresholds = scaled  # type: ignore[assignment]
    try:
        yield
    finally:
        score_module.rule_thresholds = original  # type: ignore[assignment]


# ---------------------------------------------------------------------------- point laws


def draw_l1(outcome: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One sample of the ``canonical-tmle`` law ``outcome`` (``"binary"`` or ``"continuous"``)."""
    return canonical_tmle.draw_from_seed(outcome, n, seed)


def draw_l2(n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One sample of the contrast-residual law ``L2``."""
    rng = np.random.default_rng(seed)
    w1 = rng.normal(size=n)
    w2 = rng.normal(size=n)
    a = rng.binomial(1, expit(0.4 * w1 - 0.3 * w2)).astype(float)
    y = 4.0 * w1 + 0.5 * w2 + 0.1 * a + rng.normal(size=n)
    return pd.DataFrame({"W1": w1, "W2": w2, "A": a, "Y": y}), dict(L2_TRUTH)


def ln_propensity(w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """LN's treatment mechanism ``clip(expit(0.2 + 0.5 W1 - 0.3 W2 + 0.5 V), 0.03, 0.97)``."""
    v = w1**2 - 1.0
    return np.clip(expit(0.2 + 0.5 * w1 - 0.3 * w2 + 0.5 * v), *LN_CLIP)


def ln_outcome_mean(a: float | np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """LN's outcome regression ``W1 + 0.5 W2 + V + 0.1 A``."""
    return w1 + 0.5 * w2 + (w1**2 - 1.0) + 0.1 * np.asarray(a, dtype=float)


def draw_ln(n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One sample of the necessity law ``LN``, with covariates ``W1, W2, V``."""
    rng = np.random.default_rng(seed)
    w1 = rng.normal(size=n)
    w2 = rng.normal(size=n)
    a = rng.binomial(1, ln_propensity(w1, w2)).astype(float)
    y = ln_outcome_mean(a, w1, w2) + LN_SIGMA * rng.normal(size=n)
    frame = pd.DataFrame({"W1": w1, "W2": w2, "V": w1**2 - 1.0, "A": a, "Y": y})
    return frame, dict(LN_TRUTH)


def logistic() -> LogisticRegression:
    """The unpenalized logistic regression of every point treatment mechanism."""
    return LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs")


def ln_outcome_learner() -> Pipeline:
    """LN's misspecified outcome regression: least squares on ``(A, W1, W2)``, without ``V``.

    The outcome design ``TMLE`` passes is ``[A, W1, W2, V]`` in that order
    (``tests/unit/test_option_stopping_point_design.py`` checks it), so the first three columns
    are kept.
    """
    keep = ColumnTransformer([("kept", "passthrough", [0, 1, 2])])
    return Pipeline([("select", keep), ("ols", LinearRegression())])


def point_learners(law: str) -> dict[str, Any]:
    """The learners of a point law: ``binary``, ``continuous``, ``L2`` or ``LN``."""
    if law == "binary":
        return {"outcome_learner": logistic(), "treatment_learner": logistic()}
    if law in ("continuous", "L2"):
        return {"outcome_learner": LinearRegression(), "treatment_learner": logistic()}
    if law == "LN":
        return {"outcome_learner": ln_outcome_learner(), "treatment_learner": logistic()}
    raise KeyError(law)


def fit_point(
    law: str,
    frame: pd.DataFrame,
    *,
    stopping: str,
    targeting: str = "iterative",
    estimands: tuple[str, ...] = ("ey1", "ey0", "ate"),
    scale: float = 1.0,
    **settings: Any,
) -> Any:
    """One in-sample point TMLE fit of ``law`` under ``stopping`` and ``targeting``.

    ``scale`` multiplies the rule's thresholds (:func:`threshold_scale`).  ``settings`` passes
    further ``TMLE`` keywords, such as ``step_bound``.
    """
    from cleverly.estimators import TMLE

    covariates = [column for column in frame.columns if column not in ("A", "Y")]
    estimator = TMLE(
        cross_fit=False,
        estimands=estimands,
        simultaneous=False,
        g_bounds=canonical_tmle.G_BOUNDS,
        max_iter=ONE_STEP_MAX_ITER if targeting == "one_step" else MAX_ITER,
        tol=TOL,
        random_state=0,
        targeting=targeting,
        stopping=stopping,
        **point_learners(law),
        **settings,
    )
    with threshold_scale(scale):
        return estimator.fit(frame, outcome="Y", treatment="A", covariates=covariates).single()


def point_rule_met(result: Any) -> bool:
    """Whether every outcome fluctuation of a point fit met its declared rule."""
    return all(
        fluctuation.rule_ratio is None or fluctuation.rule_ratio <= 1.0
        for fluctuation in result.fluctuations.values()
    )


# ---------------------------------------------------------------------------- LN exact law

#: Gauss-Hermite nodes per axis of the LN quadrature.
QUADRATURE_NODES = 80


def _ln_grid(nodes: int = QUADRATURE_NODES) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(W1, W2, mass)`` of a product Gauss-Hermite rule for two standard normals."""
    x, weights = np.polynomial.hermite_e.hermegauss(nodes)
    weights = weights / weights.sum()
    w1, w2 = np.meshgrid(x, x, indexing="ij")
    return w1.ravel(), w2.ravel(), np.outer(weights, weights).ravel()


def ln_initial_limit(nodes: int = QUADRATURE_NODES) -> dict[str, float]:
    """The exact-law limit of LN's initial fit and of its EY1 curve.

    The outcome regression's limit is the population least squares of ``Y`` on
    ``(1, A, W1, W2)``.  Returns its EY1 plug-in ``psi_initial``, the initial bias
    ``psi_initial - 0.1`` (which is the limit of minus the initial score
    :math:`P_0 D^*(\\bar Q_{\\mathrm{init}}, g_0)`), the efficient SD of the EY1 curve at the
    truth, and the SD of the EY1 curve at the initial regression.
    """
    w1, w2, mass = _ln_grid(nodes)
    g = ln_propensity(w1, w2)
    design_one = np.column_stack([np.ones_like(w1), np.ones_like(w1), w1, w2])
    design_zero = np.column_stack([np.ones_like(w1), np.zeros_like(w1), w1, w2])
    gram = (design_one.T * (mass * g)) @ design_one + (design_zero.T * (mass * (1.0 - g))) @ (
        design_zero
    )
    q1 = ln_outcome_mean(1.0, w1, w2)
    q0 = ln_outcome_mean(0.0, w1, w2)
    moment = design_one.T @ (mass * g * q1) + design_zero.T @ (mass * (1.0 - g) * q0)
    beta = np.linalg.solve(gram, moment)
    initial = design_one @ beta
    psi_initial = float(mass @ initial)
    truth = LN_TRUTH["ey1"]

    def curve_sd(regression: np.ndarray) -> float:
        # D = A/g (Y - Q(1, W)) + Q(1, W) - psi, so E[D | W] = q1(W) - psi and
        # Var(D | W) = ((q1 - Q)^2 + sigma^2) / g - (q1 - Q)^2.
        gap = q1 - regression
        within = float(mass @ ((gap**2 + LN_SIGMA**2) / g - gap**2))
        between = float(mass @ (q1 - float(mass @ q1)) ** 2)
        return math.sqrt(within + between)

    return {
        "psi_initial": psi_initial,
        "initial_bias": psi_initial - truth,
        "efficient_sd": curve_sd(q1),
        "initial_sd": curve_sd(initial),
    }


def predicted_plain_coverage(residual: float, critical: float) -> float:
    """``Phi(z - b) - Phi(-z - b)``: the plain interval's coverage at a constant residual ``b``.

    Exact when the residual is a constant ``b`` standard errors and the standardized error is
    standard normal, which is the misspecified regime of LN (plan 2.1).
    """
    from scipy.stats import norm

    return float(norm.cdf(critical - residual) - norm.cdf(-critical - residual))


# ---------------------------------------------------------------------------- longitudinal

#: L3's gated parameter and L4's gated, witness and necessity parameters.
L3_CONTRAST = "ate_regimen[always vs never]"
L4_CONTRAST = "ate_regimen[arm1 vs arm0 @ t=5]"
L4_RISK = "risk_regimen[arm1 @ t=5]"
#: L4's one-step step size, that of the ``point-survival-curve`` study.
L4_STEP_SIZE = 1e-3


def draw_l3(n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One sample of the ``canonical-ltmle`` censored two-node law, with its exact truths."""
    from tests.studies import canonical_ltmle

    return canonical_ltmle.draw_from_seed(canonical_ltmle.SCENARIO, n, seed)


def draw_l4(n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One sample of the ``point-survival-curve`` survival law, with its exact truths."""
    from tests.studies import canonical_point_survival_curve as curve

    return curve.draw_from_seed(curve.SCENARIO, n, seed)


def fit_l3(frame: pd.DataFrame, *, stopping: str, scale: float = 1.0, **settings: Any) -> Any:
    """The ``canonical-ltmle`` fit of one L3 sample under ``stopping`` (Newton walk per node)."""
    from cleverly.longitudinal import LTMLE
    from tests.studies import canonical_ltmle as ltmle
    from tests.studies.fractional_glm import QuasiBinomialGLM

    estimator = LTMLE(
        ltmle.REGIMENS,
        reference=ltmle.REFERENCE,
        outcome_learner=QuasiBinomialGLM(),
        pseudo_learner=QuasiBinomialGLM(),
        treatment_learner=ltmle.KnownLongitudinalMechanism("treatment"),
        censoring_learner=ltmle.KnownLongitudinalMechanism("censoring"),
        n_folds=1,
        g_bounds=ltmle.G_BOUNDS,
        simultaneous=False,
        max_iter=MAX_ITER,
        tol=TOL,
        random_state=0,
        stopping=stopping,
        **settings,
    )
    with threshold_scale(scale):
        return estimator.fit(
            frame,
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W1", "W2"],
            time_varying=[[], ["L2"]],
            censoring=["C1", "C2"],
        )


def l4_learners(configuration: str) -> dict[str, Any]:
    """L4's learners: ``"primary"`` (the study's glm learners) or ``"mechanism_correct"``."""
    if configuration == "primary":
        from tests.studies.canonical_point_survival_hazard import primary_learners

        return primary_learners()
    from tests.studies import point_survival_hazard_properties

    return point_survival_hazard_properties.learners(configuration)


def fit_l4(
    frame: pd.DataFrame,
    *,
    stopping: str,
    configuration: str = "primary",
    scale: float = 1.0,
    **settings: Any,
) -> Any:
    """One-step hazard fit of one L4 sample under ``stopping``."""
    from cleverly.longitudinal import LTMLE
    from tests.studies import canonical_point_survival_curve as curve

    estimator = LTMLE(
        curve.REGIMENS,
        reference=curve.REFERENCE,
        n_folds=1,
        simultaneous=False,
        survival_model="hazard",
        targeting="one_step",
        step_size=L4_STEP_SIZE,
        tol=TOL,
        max_iter=ONE_STEP_MAX_ITER,
        random_state=0,
        stopping=stopping,
        **l4_learners(configuration),
        **settings,
    )
    with threshold_scale(scale):
        return estimator.fit(curve.LAW.container(frame))


def longitudinal_rule_met(result: Any) -> bool:
    """Whether every hazard walk or node fluctuation of a longitudinal fit met its rule."""
    for fit in result.fits.values():
        if fit.hazard is not None:
            ratio = fit.hazard.rule_ratio
            if ratio is not None and not ratio <= 1.0:
                return False
        for step in fit.steps:
            if step.fluctuation is not None:
                ratio = step.fluctuation.rule_ratio
                if ratio is not None and not ratio <= 1.0:
                    return False
    return True
