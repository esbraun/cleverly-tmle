"""Canonical cross-fitted competing-risk longitudinal TMLE evidence study."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import canonical_ltmle_competing as ordinary
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate

PRIMARY_REPLICATES = ordinary.PRIMARY_REPLICATES
PRIMARY_N = ordinary.PRIMARY_N
SEED = 20260827
SCENARIO = ordinary.SCENARIO
REGIMENS = ordinary.REGIMENS
REFERENCE = ordinary.REFERENCE
HORIZONS = ordinary.HORIZONS
CAUSES = ordinary.CAUSES
MEAN_NAMES = ordinary.MEAN_NAMES
CONTRAST_NAMES = ordinary.CONTRAST_NAMES
ESTIMANDS = ordinary.ESTIMANDS
PROPERTY_LABELS = ordinary.PROPERTY_LABELS
REFERENCE_METADATA = ordinary.REFERENCE_METADATA

STUDY = StudyRecord(
    name="cross-fitted competing-risk longitudinal TMLE",
    slug="canonical-ltmle-competing-crossfit",
    artifacts=ROOT / "tests" / "canonical" / "lmtp_ltmle_competing_crossfit",
    document=(
        "docs/technical-reference/method-evidence/cross-fitted-competing-risk-longitudinal-tmle.md"
    ),
    anchor="cross-fitted-competing-risk-longitudinal-tmle",
    scenarios={SCENARIO: ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    margins=Margins(),
    implementation="cleverly-cross-fitted-competing-ltmle",
    reference="lmtp",
    modules=(
        "tests/studies/canonical_ltmle_competing_crossfit.py",
        "tests/studies/ltmle_competing_crossfit_properties.py",
        "tests/studies/canonical_ltmle_competing.py",
        "tests/studies/ltmle_competing_properties.py",
        "tests/studies/canonical_ltmle_crossfit.py",
        "tests/studies/canonical_ltmle.py",
        "tests/discrete_law_competing.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/canonical/lmtp_crossfit/Dockerfile",
        "tests/canonical/lmtp_crossfit/audit.py",
        "tests/canonical/lmtp_crossfit_adapter.R",
        "tests/canonical/lmtp_competing_adapter.R",
        "tests/canonical/lmtp_ltmle_competing/run_study.R",
    ),
    runner_module="tests.studies.canonical_ltmle_competing_crossfit",
    properties_module="tests.studies.ltmle_competing_crossfit_properties",
    property_cells=ordinary.property_cells(crossfit=True),
)

CONFIGURATION = ordinary.manifest_configuration(crossfit=True)


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return ordinary.draw_from_seed(scenario, n, seed)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def fit_cleverly(frame: pd.DataFrame) -> Any:
    return ordinary.fit_cleverly(frame, n_folds=5)


def cleverly_rows(
    frame: pd.DataFrame,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    if scenario != SCENARIO:
        raise KeyError(scenario)
    return ordinary.rows_from_result(fit_cleverly(frame), truth, scenario, replicate, study=STUDY)


def _replicate(
    payload: tuple[str, int, int],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, n = payload
    frame, truth = draw_scenario(scenario, n, replicate)
    result = fit_cleverly(frame)
    sample = frame.copy()
    sample.insert(0, "fold", result.folds.assignment)
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    truths = [
        {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": value}
        for name, value in truth.items()
    ]
    rows = ordinary.rows_from_result(result, truth, scenario, replicate, study=STUDY)
    return sample, truths, rows


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    payloads = [((SCENARIO, replicate, n),) for replicate in range(replicates)]
    outcomes = map_parallel(_replicate, payloads, n_jobs=n_jobs)
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truths = pd.DataFrame([row for _, rows, _ in outcomes for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes for row in rows])
    return samples, truths, estimates.loc[:, list(REPLICATE_COLUMNS)]
