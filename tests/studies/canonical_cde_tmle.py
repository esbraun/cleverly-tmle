"""Registered evidence study for ordinary controlled direct-effect TMLE."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cleverly.estimators import TMLE
from cleverly.utils.parallel import map_parallel
from tests import discrete_law_cde as cde
from tests.conftest import (
    OracleDirectOutcome,
    OracleIntermediate,
    OracleMissingness,
    OracleTreatment,
)
from tests.parallel import STUDY_JOBS
from tests.studies.cde_study_helpers import efficiency_sd, level_rows, sample_discrete, truths
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate
from tests.studies.point_study_helpers import initial_estimates, primary_rows

TMLE_VERSION = "2.1.1"
TMLE_SOURCE_SHA256 = "5e1fccaea7bf923456b8197d3eca5314db074dcbec8ca0510a15cb837883b133"
R_BASE_IMAGE = (
    "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
)
PRIMARY_REPLICATES = 3_200
PRIMARY_N = 2_000
SEED = 20261021
SCENARIOS = {0: "binary_cde_z0_mar", 1: "binary_cde_z1_mar"}
ESTIMANDS = ("ey0", "ey1", "ate", "rr", "or")
G_BOUNDS = (0.01, 0.99)
NUISANCE_BOUND = 0.01
EFFICIENCY_SD = {f"z{level}": efficiency_sd("ate", level) for level in cde.LEVELS}

STUDY = StudyRecord(
    name="ordinary controlled direct-effect TMLE",
    slug="cde-tmle",
    artifacts=ROOT / "tests" / "canonical" / "tmle_cde",
    document="docs/technical-reference/method-evidence/controlled-direct-effect-tmle.md",
    anchor="ordinary-controlled-direct-effect-tmle",
    scenarios=dict.fromkeys(SCENARIOS.values(), ESTIMANDS),
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    nuisance_count=4,
    scenario_seed_owners={SCENARIOS[1]: SCENARIOS[0]},
    resampling_seed=20261022,
    margins=Margins(),
    implementation="cleverly-cde-tmle",
    reference="tmle-r-cde",
    modules=(
        "tests/studies/canonical_cde_tmle.py",
        "tests/studies/cde_tmle_properties.py",
        "tests/studies/cde_study_helpers.py",
        "tests/discrete_law_cde.py",
        "tests/conftest.py",
        "tests/studies/point_study_helpers.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/registry.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/studies/evidence/manifest.py",
        "tests/canonical/regenerate.py",
        "tests/canonical/tmle_cde/regenerate.py",
    ),
    runner_module="tests.studies.canonical_cde_tmle",
    properties_module="tests.studies.cde_tmle_properties",
    property_cells={
        "cde_robustness": level_rows(
            cde.LEVELS,
            (
                "all_correct",
                "outcome_correct",
                "mechanisms_correct",
                "treatment_wrong",
                "intermediate_wrong",
                "observation_wrong",
            ),
        ),
        "root_n_and_efficiency": level_rows(
            cde.LEVELS, tuple(f"n_{size}" for size in (500, 2_000, 8_000))
        ),
        "root_n_rate": level_rows(cde.LEVELS, ("empirical_sd", "reported_se")),
        "interval_calibration": level_rows(
            cde.LEVELS,
            ("correctly_specified", "shrunken_se_control", "noise_control"),
        ),
        "type_i_error": level_rows(cde.LEVELS, ("sharp_null",)),
        "power": level_rows(cde.LEVELS, ("alternative",)),
        "targeting_necessity": level_rows(cde.LEVELS, ("targeted", "untargeted")),
    },
    efficiency_bounds=EFFICIENCY_SD,
    extra_artifacts=("native-result2-defect.csv",),
)

REFERENCE_METADATA = {
    "tmle_version": TMLE_VERSION,
    "tmle_source_sha256": TMLE_SOURCE_SHA256,
    "r_base_image": R_BASE_IMAGE,
}

CONFIGURATION = {
    "construction": "ordinary controlled direct-effect TMLE with MAR outcomes",
    "cross_fit": False,
    "simultaneous_intervals": False,
    "g_bounds": list(G_BOUNDS),
    "nuisance_bound": NUISANCE_BOUND,
    "nuisance_predictions": "exact outcome, treatment, intermediate, and observation values",
    "primary_sample_contract": "one observed sample per replicate, shared by both intervention levels",
}


def _level(scenario: str) -> int:
    for level, name in SCENARIOS.items():
        if scenario == name:
            return level
    raise KeyError(scenario)


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    level = _level(scenario)
    return sample_discrete(cde.PROBS, n, seed), truths(cde.PROBS, ESTIMANDS, level)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def fit_cleverly(frame: pd.DataFrame) -> Any:
    law = cde.DiscreteLaw()
    return TMLE(
        estimands=ESTIMANDS,
        outcome_learner=OracleDirectOutcome(law),
        treatment_learner=OracleTreatment(law),
        intermediate_learner=OracleIntermediate(law),
        missingness_learner=OracleMissingness(law),
        cross_fit=False,
        simultaneous=False,
        g_bounds=G_BOUNDS,
        nuisance_bound=NUISANCE_BOUND,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    ).fit(
        frame,
        outcome="Y",
        treatment="A",
        covariates=["W"],
        intermediate="Z",
        delta="Delta",
    )


def cleverly_rows(
    frame: pd.DataFrame,
    reference: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    return _cleverly_rows(fit_cleverly(frame), reference, scenario, replicate)


def _cleverly_rows(
    fit: Any,
    reference: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    result = fit[float(_level(scenario))]
    return primary_rows(
        result=result,
        truth=reference,
        implementation=STUDY.implementation,
        scenario=scenario,
        replicate=replicate,
        estimands=ESTIMANDS,
        initials=initial_estimates(result, ESTIMANDS),
    )


def _replicate(
    payload: tuple[int, int],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    replicate, n = payload
    frame, _ = draw_scenario(SCENARIOS[0], n, replicate)
    fit = fit_cleverly(frame)
    law = cde.DiscreteLaw()
    w = np.rint(frame["W"].to_numpy(dtype=float)).astype(int)
    base_sample = frame.copy()
    for level in cde.LEVELS:
        base_sample[f"qn_z{level}_a0"] = law.q[w, 0, level]
        base_sample[f"qn_z{level}_a1"] = law.q[w, 1, level]
    base_sample["gn1"] = law.g[w]
    base_sample["pzn_a0"] = law.qz[w, 0]
    base_sample["pzn_a1"] = law.qz[w, 1]
    base_sample["pin_a0"] = law.pi[w, 0]
    base_sample["pin_a1"] = law.pi[w, 1]

    sample = base_sample.copy()
    sample.insert(0, "replicate", replicate)
    truth_rows: list[dict[str, Any]] = []
    estimate_rows: list[dict[str, Any]] = []
    for level, scenario in SCENARIOS.items():
        reference = truths(cde.PROBS, ESTIMANDS, level)
        truth_rows.extend(
            {
                "scenario": scenario,
                "replicate": replicate,
                "estimand": name,
                "truth": value,
            }
            for name, value in reference.items()
        )
        estimate_rows.extend(_cleverly_rows(fit, reference, scenario, replicate))
    return sample, truth_rows, estimate_rows


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outcomes = map_parallel(
        _replicate,
        [((replicate, n),) for replicate in range(replicates)],
        n_jobs=n_jobs,
    )
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truth_rows = pd.DataFrame([row for _, rows, _ in outcomes for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes for row in rows])
    return samples, truth_rows, estimates.loc[:, list(REPLICATE_COLUMNS)]


def reference_artifacts(
    *,
    reference: Any,
    here: Path,
    samples: Path,
    truths_path: Path,
    output: Path,
    cores: int,
) -> dict[str, pd.DataFrame]:
    """Reproduce the frozen native second-result defect from the generated replication zero."""
    path = output / "native-result2-defect.csv"
    reference.run(
        here,
        samples,
        truths_path,
        path,
        cores=cores,
        runner="tmle_cde/probe_native_result2.R",
    )
    return {path.name: pd.read_csv(path)}
