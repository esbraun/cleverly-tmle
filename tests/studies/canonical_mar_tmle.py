"""Registered evidence study for ordinary point-treatment TMLE under MAR."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from cleverly.estimators import TMLE
from cleverly.utils.parallel import map_parallel
from tests import discrete_law_mar as mar
from tests.conftest import OracleMissingness, OracleOutcome, OracleTreatment
from tests.parallel import STUDY_JOBS
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate
from tests.studies.missing_outcome_study_helpers import primary_rows, sample_discrete, truths

TMLE_VERSION = "2.1.1"
TMLE_SOURCE_SHA256 = "5e1fccaea7bf923456b8197d3eca5314db074dcbec8ca0510a15cb837883b133"
R_BASE_IMAGE = (
    "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
)
PRIMARY_REPLICATES = 800
PRIMARY_N = 2_000
SEED = 20260921
SCENARIO = "binary_mar_observational"
ESTIMANDS = ("ey0", "ey1", "ate")
G_BOUNDS = (0.01, 0.99)
NUISANCE_BOUND = 0.01

STUDY = StudyRecord(
    name="ordinary point-treatment TMLE with MAR outcomes",
    slug="mar-tmle",
    artifacts=ROOT / "tests" / "canonical" / "tmle_mar",
    document="docs/technical-reference/method-evidence/ordinary-missing-outcome-tmle.md",
    anchor="ordinary-missing-outcome-tmle",
    scenarios={SCENARIO: ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=20260922,
    margins=Margins(),
    implementation="cleverly-mar-tmle",
    reference="tmle-r",
    modules=(
        "tests/studies/canonical_mar_tmle.py",
        "tests/studies/mar_tmle_properties.py",
        "tests/studies/missing_outcome_study_helpers.py",
        "tests/discrete_law_mar.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_mar_tmle",
    properties_module="tests.studies.mar_tmle_properties",
    property_cells={
        "mar_robustness": (
            "both_correct",
            "outcome_correct",
            "mechanisms_correct",
            "treatment_wrong",
            "observation_wrong",
        ),
        "root_n_and_efficiency": ("n_500", "n_2000", "n_8000"),
        "root_n_rate": ("empirical_sd", "reported_se"),
        "interval_calibration": (
            "ate__correctly_specified",
            "ate__shrunken_se_control",
            "ate__noise_control",
        ),
        "type_i_error": ("sharp_null",),
        "power": ("alternative",),
        "targeting_necessity": ("ate__targeted", "ate__untargeted"),
        "missingness_necessity": ("ate__declared", "ate__complete_case_control"),
    },
)

REFERENCE_METADATA = {
    "tmle_version": TMLE_VERSION,
    "tmle_source_sha256": TMLE_SOURCE_SHA256,
    "r_base_image": R_BASE_IMAGE,
}

CONFIGURATION = {
    "construction": "ordinary MAR point-treatment TMLE",
    "cross_fit": False,
    "simultaneous_intervals": False,
    "g_bounds": list(G_BOUNDS),
    "missingness_bound": NUISANCE_BOUND,
    "nuisance_predictions": "exact finite-law values supplied to both implementations",
}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    if scenario != SCENARIO:
        raise KeyError(scenario)
    return sample_discrete(mar.PROBS, n, seed), truths(mar.PROBS, ESTIMANDS)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def fit_cleverly(frame: pd.DataFrame) -> Any:
    law = mar.DiscreteLaw()
    return (
        TMLE(
            estimands=ESTIMANDS,
            outcome_learner=OracleOutcome(law),
            treatment_learner=OracleTreatment(law),
            missingness_learner=OracleMissingness(law),
            cross_fit=False,
            simultaneous=False,
            g_bounds=G_BOUNDS,
            nuisance_bound=NUISANCE_BOUND,
            max_iter=100,
            tol=1e-10,
            random_state=0,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W"], delta="Delta")
        .single()
    )


def cleverly_rows(
    frame: pd.DataFrame,
    reference: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    return primary_rows(
        result=fit_cleverly(frame),
        reference=reference,
        implementation=STUDY.implementation,
        scenario=scenario,
        replicate=replicate,
        estimands=ESTIMANDS,
    )


def _replicate(
    payload: tuple[str, int, int],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, n = payload
    frame, reference = draw_scenario(scenario, n, replicate)
    law = mar.DiscreteLaw()
    levels = np.rint(frame["W"].to_numpy(dtype=float)).astype(int)
    sample = frame.copy()
    sample["qn0"] = law.q[levels, 0]
    sample["qn1"] = law.q[levels, 1]
    sample["gn1"] = law.g[levels]
    sample["pin0"] = law.pi[levels, 0]
    sample["pin1"] = law.pi[levels, 1]
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
