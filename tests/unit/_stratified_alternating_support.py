"""Shared fits for the stratified alternating-targeting tests.

:func:`golden_fits` builds every fit whose numbers the stratified alternating targeting must
leave unchanged: the unstratified incremental, link-MSM, continuous-MSM, natural-course and
DR-TMLE fits, and the stratified arm, regime, shift and identity-MSM fits that shipped before
it.  :func:`summary` reduces each fit to the numbers
``tests/unit/test_stratified_alternating_bit_identity.py`` compares against the values in
:mod:`tests.unit._stratified_alternating_golden`, which were recorded before any source edit.

Every learner converges to its unique optimum: the logistic regressions run at
``tol=1e-12``, so the golden values do not depend on the platform's floating-point path.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.estimators import DRTMLE, TMLE
from cleverly.interventions import Incremental, Shift, Static
from cleverly.msm import MSM
from tests.studies import stratified_law as law

#: The rows of every golden fit.
N = 400
SEED = 20261101
#: The doses of the continuous working models.
GRID = (-1.0, -0.5, 0.0, 0.5, 1.0)


def logistic() -> LogisticRegression:
    """A logistic regression solved to its unique optimum."""
    return LogisticRegression(tol=1e-12, max_iter=10_000)


def frame(n: int = N, seed: int = SEED) -> pd.DataFrame:
    """The baseline-strata law, with the columns each composition reads.

    ``Delta`` is a missing-outcome indicator and ``Yobs`` the outcome with the unobserved
    rows set to ``NaN``.  ``A3`` is a three-arm treatment and ``D`` a continuous dose that
    depends on ``(W, V)``.  ``Yc`` is a continuous outcome and ``DeltaA`` a missing-treatment
    indicator, with ``Amiss`` the treatment set to ``NaN`` where it is unrecorded.
    """
    rows = law.sample(n, seed)
    rng = np.random.default_rng(seed + 1)
    w = rows["W"].to_numpy(dtype=float)
    v = rows["V"].to_numpy(dtype=float)
    a = rows["A"].to_numpy(dtype=float)
    observed = rng.random(n) < 1.0 / (1.0 + np.exp(-(1.2 - 0.3 * a + 0.2 * w - 0.3 * v)))
    third = rng.random(n) < 0.3
    dose = 0.5 * w - 0.4 * v + rng.normal(size=n)
    recorded = rng.random(n) < 0.85
    return rows.assign(
        Delta=observed.astype(float),
        Yobs=np.where(observed, rows["Y"].to_numpy(dtype=float), np.nan),
        A3=np.where(third, 2.0, a),
        D=dose,
        Yc=1.0 + dose + 0.5 * w + 0.5 * v + 0.5 * dose * v + rng.normal(size=n),
        Yd=(rng.random(n) < 1.0 / (1.0 + np.exp(-(-0.5 + 0.6 * dose + 0.4 * w)))).astype(float),
        DeltaA=recorded.astype(float),
        Amiss=np.where(recorded, a, np.nan),
    )


def settings(**overrides: Any) -> dict[str, Any]:
    """In-sample settings with explicit learners and no bands."""
    return {
        "outcome_learner": logistic(),
        "treatment_learner": logistic(),
        "cross_fit": False,
        "simultaneous": False,
        "random_state": 0,
        "max_iter": 100,
        "tol": 1e-10,
        **overrides,
    }


def drtmle_settings(**overrides: Any) -> dict[str, Any]:
    """:func:`settings` with linear reduced regressions for ``DRTMLE``."""
    return settings(
        reduced_outcome_learner=LinearRegression(),
        reduced_treatment_learner=logistic(),
        **overrides,
    )


def _fit(estimator: Any, rows: pd.DataFrame, *, strata: bool = False, **roles: Any) -> Any:
    roles = {"outcome": "Y", "treatment": "A", "covariates": ["W", "V"], **roles}
    if strata:
        roles["strata"] = ["V"]
    return estimator.fit(rows, **roles).single()


def _dose(estimator: Any, rows: pd.DataFrame, outcome: str, *, strata: bool = False) -> Any:
    return _fit(
        estimator,
        rows,
        strata=strata,
        outcome=outcome,
        treatment="D",
        treatment_kind="continuous",
    )


#: Each golden fit, by name.  A builder takes the shared frame.
GOLDEN: dict[str, Callable[[pd.DataFrame], Any]] = {
    "ipsi": lambda f: _fit(TMLE(incremental=[Incremental(2.0), Incremental(0.5)], **settings()), f),
    "msm_logit": lambda f: _fit(
        TMLE(msm=MSM.linear(modifiers=("W",), link="logit"), **settings()), f
    ),
    "msm_log": lambda f: _fit(TMLE(msm=MSM.linear(link="log"), **settings()), f),
    "continuous_msm_identity": lambda f: _dose(
        TMLE(
            msm=MSM.linear(doses=GRID),
            density_bins=6,
            **settings(outcome_learner=LinearRegression()),
        ),
        f,
        "Yc",
    ),
    "continuous_msm_logit": lambda f: _dose(
        TMLE(msm=MSM.linear(doses=GRID, link="logit"), density_bins=6, **settings()), f, "Yd"
    ),
    "natural_course": lambda f: _fit(
        TMLE(estimands=("ey_obs",), **settings()), f, outcome="Yobs", delta="Delta"
    ),
    "drtmle_binary": lambda f: _fit(DRTMLE(estimands=("ey", "ate"), **drtmle_settings()), f),
    "drtmle_three_arm": lambda f: _fit(
        DRTMLE(estimands=("ey", "ate"), **drtmle_settings()), f, treatment="A3"
    ),
    "drtmle_missing_randomized": lambda f: _fit(
        DRTMLE(randomized=True, estimands=("ey", "ate"), **drtmle_settings()),
        f,
        outcome="Yobs",
        delta="Delta",
    ),
    "drtmle_missing_three_arm": lambda f: _fit(
        DRTMLE(randomized=True, estimands=("ey", "ate"), **drtmle_settings()),
        f,
        outcome="Yobs",
        treatment="A3",
        delta="Delta",
    ),
    "drtmle_composite": lambda f: _fit(
        DRTMLE(estimands=("ey", "ate"), **drtmle_settings()), f, outcome="Yobs", delta="Delta"
    ),
    "drtmle_composite_missing_treatment": lambda f: _fit(
        DRTMLE(estimands=("ey", "ate"), **drtmle_settings()),
        f,
        treatment="Amiss",
        treatment_delta="DeltaA",
    ),
    "strata_arms": lambda f: _fit(
        TMLE(estimands=("ey", "ate", "att", "atc"), **settings()), f, strata=True
    ),
    "strata_regimes": lambda f: _fit(
        TMLE(interventions=(Static(1), Static(0)), **settings()), f, strata=True
    ),
    "strata_shifts": lambda f: _dose(
        TMLE(shifts=[Shift(0.5, cap=None)], density_bins=6, **settings()), f, "Y", strata=True
    ),
    "strata_msm_identity": lambda f: _fit(
        TMLE(msm=MSM.linear(modifiers=("W",)), **settings()), f, strata=True
    ),
}


def summary(result: Any) -> dict[str, float]:
    """The numbers of one fit that the golden values pin.

    For each estimate: the point estimate, the mean square of its curve, and two linear
    functionals of the curve.  For each fluctuation: every coefficient.
    """
    probe = np.random.default_rng(0).normal(size=result.data.n)
    out: dict[str, float] = {}
    for name, estimate in result.estimates.items():
        curve = np.asarray(estimate.influence_curve, dtype=float)
        out[f"{name}|psi"] = float(estimate.psi)
        out[f"{name}|square"] = float(np.mean(curve**2))
        out[f"{name}|sum"] = float(np.sum(curve))
        out[f"{name}|probe"] = float(curve @ probe)
    for group, fluctuation in result.fluctuations.items():
        for j, value in enumerate(np.ravel(fluctuation.epsilon)):
            out[f"{group}|epsilon{j}"] = float(value)
    return out


def golden_summaries() -> dict[str, dict[str, float]]:
    """:func:`summary` of every golden fit on the shared frame."""
    rows = frame()
    return {name: summary(build(rows)) for name, build in GOLDEN.items()}
