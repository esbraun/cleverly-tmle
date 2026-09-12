"""Registered evidence study for the natural-course mean under MAR."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from cleverly.estimators import TMLE
from cleverly.utils.parallel import map_parallel
from tests import discrete_law_mar as mar
from tests.conftest import OracleMissingness, OracleOutcome, OracleOutcomeContinuous
from tests.parallel import STUDY_JOBS
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate
from tests.studies.missing_outcome_study_helpers import (
    FailTreatment,
    NaturalCourseLaw,
    efficiency_sd,
    sample_discrete,
)
from tests.studies.point_study_helpers import primary_rows

PRIMARY_REPLICATES = 800
PRIMARY_N = 2_000
SEED = 20261007
SCENARIOS = ("binary_mar_natural_course", "continuous_mar_natural_course")
ESTIMANDS = ("ey_obs",)
Q_BOUNDS = (0.0, 1.0)
NUISANCE_BOUND = 0.01
BETA_CONCENTRATION = 24.0
TRUTH = float(mar.functional(mar.PROBS, "ey_obs"))
EFFICIENCY_SD = efficiency_sd(mar.PROBS, "ey_obs")


STUDY = StudyRecord(
    name="ordinary missing-outcome natural-course TMLE",
    slug="mar-natural-course-tmle",
    artifacts=ROOT / "tests" / "canonical" / "tmle_mar_natural_course",
    document=(
        "docs/technical-reference/method-evidence/ordinary-missing-outcome-natural-course-tmle.md"
    ),
    anchor="ordinary-missing-outcome-natural-course-tmle",
    scenarios=dict.fromkeys(SCENARIOS, ESTIMANDS),
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=20261008,
    margins=Margins(),
    implementation="cleverly-mar-natural-course-tmle",
    reference=None,
    modules=(
        "tests/studies/canonical_mar_natural_course.py",
        "tests/studies/mar_natural_course_properties.py",
        "tests/studies/missing_outcome_study_helpers.py",
        "tests/discrete_law_mar.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_mar_natural_course",
    properties_module="tests.studies.mar_natural_course_properties",
    property_cells={
        "double_robustness": (
            "both_correct",
            "outcome_correct",
            "response_correct",
            "both_wrong",
        ),
        "root_n_and_efficiency": ("n_500", "n_2000", "n_8000"),
        "root_n_rate": ("empirical_sd", "reported_se"),
        "interval_calibration": (
            "ey_obs__correctly_specified",
            "ey_obs__shrunken_se_control",
            "ey_obs__noise_control",
        ),
        "targeting_necessity": ("ey_obs__targeted", "ey_obs__untargeted"),
        "missingness_necessity": ("ey_obs__declared", "ey_obs__complete_case_control"),
    },
    efficiency_bounds={"ey_obs": EFFICIENCY_SD},
)

CONFIGURATION = {
    "construction": "ordinary MAR natural-course TMLE",
    "source": "Diaz, Carone, and van der Laan (2016), Section 2, equations 1-5",
    "cross_fit": False,
    "fluctuation": "logistic",
    "targeting": "iterative",
    "target_weights": False,
    "simultaneous_intervals": False,
    "q_bounds": list(Q_BOUNDS),
    "missingness_bound": NUISANCE_BOUND,
    "treatment_mechanism": "deliberately unavailable and asserted unused",
    "nuisance_predictions": (
        "exact response probabilities and binary outcome means; correctly specified "
        "affine outcome learner for the continuous law"
    ),
    "continuous_outcome": f"beta with concentration {BETA_CONCENTRATION:g}",
    "comparator_search": (
        "The repository's pinned R tmle 2.1.1 adapter reports MAR intervention-arm means, "
        "not this natural-course target; the cited paper specifies the estimator but no "
        "maintained public implementation with a matching callable target was identified."
    ),
}


def _sample_continuous(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.choice(len(mar.P_W), size=n, p=mar.P_W)
    a = rng.binomial(1, mar.G[w])
    delta = rng.binomial(1, mar.PI[w, a])
    mean = mar.Q[w, a]
    y = rng.beta(mean * BETA_CONCENTRATION, (1.0 - mean) * BETA_CONCENTRATION)
    return pd.DataFrame(
        {
            "W": w.astype(float),
            "A": a.astype(float),
            "Y": np.where(delta == 1, y, np.nan),
            "Delta": delta.astype(float),
        }
    )


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    if scenario == "binary_mar_natural_course":
        frame = sample_discrete(mar.PROBS, n, seed)
    elif scenario == "continuous_mar_natural_course":
        frame = _sample_continuous(n, seed)
    else:
        raise KeyError(scenario)
    return frame, {"ey_obs": TRUTH}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def fit_cleverly(frame: pd.DataFrame, scenario: str, *, law: NaturalCourseLaw | None = None) -> Any:
    declared = NaturalCourseLaw() if law is None else law
    outcome = (
        OracleOutcome(declared)
        if scenario == "binary_mar_natural_course"
        else OracleOutcomeContinuous(declared)
    )
    return (
        TMLE(
            estimands=ESTIMANDS,
            outcome_learner=outcome,
            treatment_learner=FailTreatment(),
            missingness_learner=OracleMissingness(declared),
            cross_fit=False,
            fluctuation="logistic",
            targeting="iterative",
            target_weights=False,
            simultaneous=False,
            **({"q_bounds": Q_BOUNDS} if scenario == "continuous_mar_natural_course" else {}),
            nuisance_bound=NUISANCE_BOUND,
            max_iter=100,
            tol=1e-10,
            random_state=0,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W"], delta="Delta")
        .single()
    )


def initial_estimate(result: Any) -> float:
    values = np.asarray(
        result.nuisance.scaler.unscale_levels(result.nuisance.outcome.observed), dtype=float
    )
    return float(np.average(values, weights=np.asarray(result.data.weights, dtype=float)))


def cleverly_rows(
    frame: pd.DataFrame,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    result = fit_cleverly(frame, scenario)
    return primary_rows(
        result=result,
        truth=truth,
        implementation=STUDY.implementation,
        scenario=scenario,
        replicate=replicate,
        estimands=ESTIMANDS,
        initials={"ey_obs": initial_estimate(result)},
    )


def _replicate(payload: tuple[str, int, int]) -> list[dict[str, Any]]:
    scenario, replicate, n = payload
    frame, truth = draw_scenario(scenario, n, replicate)
    return cleverly_rows(frame, truth, scenario, replicate)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    outcomes = map_parallel(
        _replicate,
        [((scenario, replicate, n),) for scenario in SCENARIOS for replicate in range(replicates)],
        n_jobs=n_jobs,
    )
    rows = pd.DataFrame([row for result in outcomes for row in result])
    return rows.loc[:, list(REPLICATE_COLUMNS)]
