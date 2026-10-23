"""Registered evidence for the confidence limits of the marginal sensitivity model bounds (X82 SH).

:func:`cleverly.sensitivity.marginal_sensitivity_bounds` reports the ATE bounds at each
``Gamma`` with a standard error for each end. Its confidence limits stay descriptive until a
registered cell measures the coverage of each end's interval (``X82-H-plan.md`` section 8 in
the plans repository). This study holds one cell per ``msm_bound_limit`` rule:

====  =====================  ==========================================  =================
cell  ``msm_bound_limit``    law (:mod:`tests.studies.marginal_bound_law`)  rows
====  =====================  ==========================================  =================
SH1   ``continuous``         Gaussian linear outcome, 20 covariates       ``ate_lower``, ``ate_upper``
SH2   ``binary``             logistic outcome, prevalence near 0.33       ``ate_lower``, ``ate_upper``
====  =====================  ==========================================  =================

Each row reads one end of the bound at ``Gamma = 1.5`` as an estimate of the sharp end, with
the reported standard error of that end. Every truth is the sharp set of the law, by
quadrature. Every nuisance GLM is correctly specified. SH1 calls the bounds with the default
quantile and tail learners, which the ``continuous`` rule names.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.estimators import TMLE
from cleverly.sensitivity import marginal_sensitivity_bounds
from cleverly.validation._interpretation import fit_configuration
from tests.parallel import STUDY_JOBS
from tests.studies import marginal_bound_law as law
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed, stream_seed

#: 2,000 is the smallest multiple of 400 whose joint primary and property pass probability
#: reaches 0.95 (0.960, ``X82-H-plan.md`` rev 2.5, section 8).
PRIMARY_REPLICATES = 2_000

#: An estimated quantile biases each end outward at second order, by about ``0.29`` SD at
#: ``n = 2000`` in SH1 (the pilot of ``X82-H-plan.md`` section 8). The bias falls like
#: ``1 / sqrt(n)``, so ``n = 8000`` holds it near ``0.15`` SD.
PRIMARY_N = 8_000
SEED = 20262321
RESAMPLING_SEED = 20262322

#: The declared sensitivity parameter, and the smaller one the wrong-gamma control reads.
GAMMA = 1.5
CONTROL_GAMMA = 1.25

#: The two-sided 95% multiplier of each row's interval.
TWO_SIDED = float(stats.norm.ppf(0.975))

ENDS = ("lower", "upper")

#: The ``msm_bound_limit`` rule of each cell.
RULES = {"SH1": "continuous", "SH2": "binary"}

SCENARIOS = {cell: tuple(f"ate_{end}" for end in ENDS) for cell in law.CELLS}

#: The file that records the fit configuration of each cell.
CELLS_FILE = "cells.csv"

#: The calibration kinds of each row: the declared reading and its two controls.
KINDS = ("correctly_specified", "inflated_se_control", "wrong_gamma_control")

PROPERTY_CELLS = {
    "interval_calibration": tuple(
        f"{cell}_{row}__{kind}"
        for kind in KINDS
        for cell, rows in SCENARIOS.items()
        for row in rows
    ),
}

STUDY = StudyRecord(
    name="marginal sensitivity bound limit coverage",
    slug="marginal-bound-limit-coverage",
    artifacts=ROOT / "tests" / "canonical" / "marginal_bound_limit_coverage",
    document="docs/technical-reference/method-evidence/marginal-bound-limit-coverage.md",
    anchor="marginal-sensitivity-bound-limit-coverage",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-msm-bound-limit",
    reference=None,
    failure_bound={"cleverly-msm-bound-limit": 0.01},
    modules=(
        "tests/studies/marginal_bound_coverage.py",
        "tests/studies/marginal_bound_coverage_properties.py",
        "tests/studies/marginal_bound_law.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.marginal_bound_coverage",
    properties_module="tests.studies.marginal_bound_coverage_properties",
    property_artifacts=(CELLS_FILE,),
    property_cells=PROPERTY_CELLS,
    publication_policy="gated",
)

CONFIGURATION = {
    "law": "tests.studies.marginal_bound_law: 20 standard normal covariates, logistic "
    "propensity; SH1 Gaussian linear outcome, SH2 logistic outcome",
    "fit": "TMLE, 5-fold cross-fitting, random_state=0; treatment LogisticRegression(C=1e6); "
    "outcome LinearRegression with q_bounds=(-12, 12) (SH1) or LogisticRegression(C=1e6) (SH2)",
    "bounds": f"marginal_sensitivity_bounds(result, ({CONTROL_GAMMA}, {GAMMA})), default "
    "quantile and tail learners",
    "primary_row": "estimate = one end of the ATE bound at Gamma = 1.5; std_error = that end's "
    "reported standard error; interval = estimate +/- 1.959964 std_error",
    "truth": "the sharp set of the law at Gamma = 1.5, by tensor Gauss-Legendre quadrature",
}


@functools.cache
def _sharp(scenario: str, gamma: float) -> tuple[float, float]:
    return law.sharp_set(scenario, gamma)


def truths(scenario: str, gamma: float = GAMMA) -> dict[str, float]:
    """The sharp ATE set of ``scenario`` at ``gamma``, by row."""
    if scenario not in SCENARIOS:
        raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")
    lower, upper = _sharp(scenario, gamma)
    return {"ate_lower": lower, "ate_upper": upper}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one sample from an explicit seed for the published-seed audit."""
    return law.sample(scenario, n, seed), truths(scenario)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one replication from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def fit(scenario: str, frame: pd.DataFrame) -> Any:
    """The declared fit of ``scenario``."""
    if scenario == "SH1":
        estimator = TMLE(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(C=1e6, max_iter=2000),
            q_bounds=(-12.0, 12.0),
            n_folds=5,
            random_state=0,
        )
    elif scenario == "SH2":
        estimator = TMLE(
            outcome_learner=LogisticRegression(C=1e6, max_iter=2000),
            treatment_learner=LogisticRegression(C=1e6, max_iter=2000),
            n_folds=5,
            random_state=0,
        )
    else:
        raise KeyError(scenario)
    return estimator.fit(
        frame, outcome="Y", treatment="A", covariates=list(law.COVARIATES)
    ).single()


def bounds(scenario: str, frame: pd.DataFrame) -> Any:
    """The declared bounds of one draw, at the control and the declared ``Gamma``."""
    return marginal_sensitivity_bounds(fit(scenario, frame), (CONTROL_GAMMA, GAMMA))


def end_values(result: Any, gamma: float) -> dict[str, tuple[float, float]]:
    """``(estimate, std_error)`` of each end of the ATE bound at ``gamma``."""
    (row,) = [row for row in result.rows if row.gamma == gamma]
    return {
        "ate_lower": (float(row.lower), float(row.lower_std_error)),
        "ate_upper": (float(row.upper), float(row.upper_std_error)),
    }


def _row(
    scenario: str, replicate: int, n: int, estimand: str, truth: float, estimate: float, se: float
) -> dict[str, Any]:
    low = estimate - TWO_SIDED * se
    high = estimate + TWO_SIDED * se
    return {
        "implementation": STUDY.implementation,
        "scenario": scenario,
        "replicate": replicate,
        "n": n,
        "estimand": estimand,
        "truth": float(truth),
        "estimate": float(estimate),
        "inference_estimate": float(estimate),
        "std_error": float(se),
        "ci_lower": float(low),
        "ci_upper": float(high),
        "inference_scale": "identity",
        "covered": int(low <= truth <= high),
        "initial_estimate": math.nan,
    }


def cleverly_rows(
    frame: pd.DataFrame, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    """Convert one draw's bounds to the shared replication schema, one row per end."""
    values = end_values(bounds(scenario, frame), GAMMA)
    return [
        _row(scenario, replicate, len(frame), name, truth[name], *values[name])
        for name in SCENARIOS[scenario]
    ]


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, index, n = key
    return draw_scenario(scenario, n, index)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> list[dict[str, Any]]:
    scenario, index, _ = key
    frame, truth = drawn
    return cleverly_rows(frame, truth, scenario, index)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Draw and fit every declared primary replication."""
    outcomes = replication.fit_primary(
        [(scenario, index, n) for scenario in STUDY.scenarios for index in range(replicates)],
        _replicate,
        draw=_draw,
        rows_at=None,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    built = pd.DataFrame([row for records in outcomes.kept for row in records])
    return replication.result(outcomes, built.loc[:, list(REPLICATE_COLUMNS)])


def cell_configurations() -> pd.DataFrame:
    """The fit configuration of each row, read off one fit per cell.

    The draw comes from a stream of its own. Every nuisance learner is fixed, so one draw
    records the configuration that every replication of the cell fits.
    """
    records = []
    for scenario, rows in SCENARIOS.items():
        frame, _ = draw_from_seed(
            scenario, PRIMARY_N, stream_seed(STUDY, "configuration", scenario)
        )
        result = fit(scenario, frame)
        configuration = asdict(fit_configuration(result, "ate"))
        configuration["learners"] = " | ".join(configuration["learners"])
        for row in rows:
            records.append({"scenario": scenario, "parameter": row, **configuration})
    return pd.DataFrame(records)
