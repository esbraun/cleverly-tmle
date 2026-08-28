"""Registered evidence study for deterministic point-treatment regimes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from cleverly.estimators import TMLE
from cleverly.interventions import Rule, Static
from cleverly.utils.parallel import map_parallel
from tests import discrete_law as law
from tests.conftest import OracleTreatment
from tests.parallel import STUDY_JOBS
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate
from tests.studies.intervention_study_helpers import (
    initial_regime_estimates,
    primary_rows,
    sample_discrete,
    truths,
)

LMTP_COMMIT = "f04a2b47f46debc515ce4ae778e05ebfde922c44"
R_BASE_IMAGE = (
    "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
)
PRIMARY_REPLICATES = 800
PRIMARY_N = 2_000
SEED = 20260901
SCENARIO = "binary_dynamic_rule"
ESTIMANDS = (
    "ey_regime[never]",
    "ey_regime[rule]",
    "ate_regime[rule vs never]",
)
G_BOUNDS = (0.01, 0.99)


def _levels(frame: Any) -> np.ndarray:
    return np.rint(np.asarray(frame["W"], dtype=float)).astype(int)


def interventions() -> tuple[Any, ...]:
    return (
        Static(0.0, name="never"),
        Rule(lambda frame: np.where(_levels(frame) == 1, 0, 1), name="rule"),
    )


STUDY = StudyRecord(
    name="ordinary deterministic point-treatment regimes",
    slug="deterministic-regimes",
    artifacts=ROOT / "tests" / "canonical" / "lmtp_regimes",
    document="docs/technical-reference/method-evidence/deterministic-point-treatment-regimes.md",
    anchor="deterministic-point-treatment-regimes",
    scenarios={SCENARIO: ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=20260911,
    margins=Margins(),
    implementation="cleverly",
    reference="lmtp",
    modules=(
        "tests/studies/canonical_deterministic_regimes.py",
        "tests/studies/deterministic_regime_properties.py",
        "tests/studies/intervention_study_helpers.py",
        "tests/studies/regime_property_helpers.py",
        "tests/discrete_law.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_deterministic_regimes",
    properties_module="tests.studies.deterministic_regime_properties",
    property_cells={
        "double_robustness": (
            "both_correct",
            "outcome_correct",
            "treatment_correct",
            "both_wrong",
        ),
        "root_n_and_efficiency": ("n_500", "n_2000", "n_8000"),
        "root_n_rate": ("empirical_sd", "reported_se"),
        "interval_calibration": (
            "rule__correctly_specified",
            "rule__shrunken_se_control",
            "rule__noise_control",
        ),
        "type_i_error": ("sharp_null",),
        "power": ("alternative",),
        "targeting_necessity": ("rule__targeted", "rule__untargeted"),
        "rule_necessity": ("rule__declared", "rule__static_control"),
        "static_reduction": ("never__regime", "never__arm"),
    },
)

REFERENCE_METADATA = {
    "lmtp_commit": LMTP_COMMIT,
    "r_base_image": R_BASE_IMAGE,
    "reference_parameter": "one-node deterministic modified policy with mtp=FALSE",
}

CONFIGURATION = {
    "construction": "ordinary",
    "cross_fit": False,
    "simultaneous_intervals": False,
    "g_bounds": list(G_BOUNDS),
    "regimes": ["never", "rule"],
}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    if scenario != SCENARIO:
        raise KeyError(scenario)
    return sample_discrete(law.PROBS, n, seed), truths(law.PROBS, ESTIMANDS)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def fit_cleverly(frame: pd.DataFrame) -> Any:
    return (
        TMLE(
            interventions=interventions(),
            outcome_learner=LogisticRegression(C=1e6, max_iter=2_000),
            treatment_learner=OracleTreatment(law.DiscreteLaw()),
            cross_fit=False,
            simultaneous=False,
            g_bounds=G_BOUNDS,
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
        initials=initial_regime_estimates(result),
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
