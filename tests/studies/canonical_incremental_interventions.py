"""Registered evidence study for incremental propensity-score interventions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression

from cleverly.estimators import TMLE
from cleverly.utils.parallel import map_parallel
from tests import discrete_law as law
from tests import incrementals
from tests.conftest import OracleTreatment
from tests.parallel import STUDY_JOBS
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import replicate_seed
from tests.studies.intervention_study_helpers import (
    incremental_estimates,
    primary_rows,
    sample_discrete,
    truths,
)

IMTP_COMMIT = "d4b5204f9505147a54ac415180e84b86f005c8b2"
R_BASE_IMAGE = (
    "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
)
PRIMARY_REPLICATES = 1_600
PRIMARY_N = 2_000
SEED = 20260904
SCENARIO = "binary_incremental_odds"
ESTIMANDS = tuple(law.PER_ARM_NAMES["ey_ipsi"]) + tuple(law.PER_ARM_NAMES["ate_ipsi"])


STUDY = StudyRecord(
    name="incremental propensity-score interventions",
    slug="incremental-interventions",
    artifacts=ROOT / "tests" / "canonical" / "imtp_incremental",
    document="docs/technical-reference/method-evidence/incremental-propensity-interventions.md",
    anchor="incremental-propensity-interventions",
    scenarios={SCENARIO: ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=20260914,
    margins=Margins(),
    implementation="cleverly",
    reference="imtp",
    accepted_reference_failure=(
        "Pinned imtp supplies a point-curve witness only: its reported influence curve omits "
        "the derivative through the treatment mechanism, so its independent inference gate "
        "is expected to fail and is not evidence for cleverly's inference."
    ),
    point_only_reference=frozenset(ESTIMANDS),
    modules=(
        "tests/studies/canonical_incremental_interventions.py",
        "tests/studies/incremental_intervention_properties.py",
        "tests/studies/intervention_study_helpers.py",
        "tests/studies/regime_property_helpers.py",
        "tests/discrete_law.py",
        "tests/incrementals.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_incremental_interventions",
    properties_module="tests.studies.incremental_intervention_properties",
    property_cells={
        "mechanism_requirement": ("both_correct", "outcome_wrong", "mechanism_wrong"),
        "root_n_and_efficiency": (
            "contrast__n_500",
            "contrast__n_2000",
            "contrast__n_8000",
        ),
        "root_n_rate": ("contrast__empirical_sd", "contrast__reported_se"),
        "interval_calibration": (
            "contrast__correctly_specified",
            "contrast__shrunken_se_control",
            "contrast__noise_control",
        ),
        "type_i_error": ("sharp_null",),
        "power": ("alternative",),
        "targeting_necessity": (
            "outcome__targeted",
            "outcome__untargeted",
            "mechanism__targeted",
            "mechanism__untargeted",
        ),
        "treatment_score_necessity": ("odds_x2__full_eif", "odds_x2__regime_curve_control"),
        "natural_course_identity": ("natural__ipsi", "natural__mean"),
    },
)

REFERENCE_METADATA = {
    "imtp_commit": IMTP_COMMIT,
    "r_base_image": R_BASE_IMAGE,
    "reference_parameter": "point-treatment incremental odds curve",
    "reference_scope": "point estimates only; influence-curve inference is not canonical",
}

CONFIGURATION = {
    "construction": "ordinary",
    "cross_fit": False,
    "simultaneous_intervals": False,
    "deltas": dict(law.IPSI_DELTAS),
    "external_comparator": "imtp point estimates only",
}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    if scenario != SCENARIO:
        raise KeyError(scenario)
    return sample_discrete(law.PROBS, n, seed), truths(law.PROBS, ESTIMANDS)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_from_seed(scenario, n, replicate_seed(STUDY, scenario, replicate))


def fit_cleverly(frame: pd.DataFrame) -> Any:
    return (
        TMLE(
            incremental=incrementals.interventions(),
            outcome_learner=LogisticRegression(C=1e6, max_iter=2_000),
            treatment_learner=OracleTreatment(law.DiscreteLaw()),
            cross_fit=False,
            simultaneous=False,
            max_iter=100,
            tol=1e-10,
            random_state=0,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W"])
        .single()
    )


def cleverly_rows(
    frame: pd.DataFrame,
    reference: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    result = fit_cleverly(frame)
    return primary_rows(
        result=result,
        reference=reference,
        implementation=STUDY.implementation,
        scenario=scenario,
        replicate=replicate,
        initials=incremental_estimates(result),
        estimands=ESTIMANDS,
    )


def _replicate(
    payload: tuple[str, int, int],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, n = payload
    frame, reference = draw_scenario(scenario, n, replicate)
    sample = frame.copy()
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    truth_rows = [
        {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": value}
        for name, value in reference.items()
    ]
    return sample, truth_rows, cleverly_rows(frame, reference, scenario, replicate)


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outcomes = map_parallel(
        _replicate,
        [((SCENARIO, replicate, n),) for replicate in range(replicates)],
        n_jobs=n_jobs,
    )
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truth_rows = pd.DataFrame([row for _, rows, _ in outcomes for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes for row in rows])
    return samples, truth_rows, estimates.loc[:, list(REPLICATE_COLUMNS)]
