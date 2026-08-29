"""Registered evidence study for clustered point-treatment CV-TMLE."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from cleverly.datasets import clustered_dgp
from cleverly.estimators import TMLE
from cleverly.learners.crossfit import check_integrity
from cleverly.utils.parallel import map_parallel
from tests.conftest import OracleTreatment
from tests.parallel import STUDY_JOBS
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate
from tests.studies.missing_outcome_study_helpers import primary_rows

LMTP_COMMIT = "f04a2b47f46debc515ce4ae778e05ebfde922c44"
IFE_VERSION = "0.2.3"
IFE_SHA256 = "b6be1e9ba514db95118e425d2f78deabb2c9f745f44f35a301ff9b5f266d5ed2"
R_BASE_IMAGE = (
    "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
)
PRIMARY_REPLICATES = 800
PRIMARY_N = 2_000
PROPERTY_REPLICATES = 2_400
N_FOLDS = 5
CLUSTER_SIZE = 10
SEED = 20260929
RESAMPLING_SEED = 20260930
SCENARIO = "clustered_continuous"
ESTIMANDS = ("ey0", "ey1", "ate")
G_BOUNDS = (1e-9, 1.0 - 1e-9)

STUDY = StudyRecord(
    name="clustered point-treatment CV-TMLE",
    slug="clustered-tmle",
    artifacts=ROOT / "tests" / "canonical" / "lmtp_clustered_tmle",
    document="docs/technical-reference/method-evidence/clustered-point-treatment-cv-tmle.md",
    anchor="clustered-point-treatment-cv-tmle",
    scenarios={SCENARIO: ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-clustered-cvtmle",
    reference="lmtp",
    modules=(
        "tests/studies/canonical_clustered_tmle.py",
        "tests/studies/clustered_tmle_properties.py",
        "tests/studies/missing_outcome_study_helpers.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_clustered_tmle",
    properties_module="tests.studies.clustered_tmle_properties",
    property_cells={
        "clustered_inference": ("cluster_robust", "iid_control"),
    },
)

REFERENCE_METADATA = {
    "ife_sha256": IFE_SHA256,
    "ife_version": IFE_VERSION,
    "lmtp_commit": LMTP_COMMIT,
    "r_base_image": R_BASE_IMAGE,
    "reference_parameter": "five-fold deterministic point-treatment TMLE with cluster identifiers",
}

CONFIGURATION = {
    "cluster_size": CLUSTER_SIZE,
    "cross_fit": True,
    "n_folds": N_FOLDS,
    "folds": "identical grouped assignments supplied to both implementations",
    "targeting_scheme": "pooled",
    "outcome_type": "continuous",
    "treatment_mechanism": "exact",
    "simultaneous_intervals": False,
    "g_bounds": list(G_BOUNDS),
}


def law() -> Any:
    """Return the declared clustered law."""
    return clustered_dgp(cluster_size=CLUSTER_SIZE)


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one clustered sample from an explicit seed."""
    if scenario != SCENARIO:
        raise KeyError(scenario)
    frame, truth = law().sample(n, seed=seed, backend="pandas")
    return frame, {name: float(truth[name]) for name in ESTIMANDS}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one replication from this study's seed stream."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def fit_cleverly(frame: pd.DataFrame) -> Any:
    """Fit the grouped five-fold estimator with the exact treatment mechanism."""
    dgp = law()
    result = (
        TMLE(
            outcome_learner=LinearRegression(),
            treatment_learner=OracleTreatment(dgp),
            cross_fit=True,
            n_folds=N_FOLDS,
            targeting_scheme="pooled",
            estimands=ESTIMANDS,
            simultaneous=False,
            g_bounds=G_BOUNDS,
            max_iter=100,
            tol=1e-10,
            random_state=0,
        )
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2"],
            id="cluster",
        )
        .single()
    )
    check_integrity(result.nuisance.folds, cluster=np.asarray(frame["cluster"]))
    if not np.array_equal(result.data.cluster, np.asarray(frame["cluster"])):
        raise AssertionError("the clustered fit did not retain the supplied identifier")
    return result


def rows_from_result(
    result: Any,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    """Convert one fit to the shared primary-replication schema."""
    return primary_rows(
        result=result,
        reference=truth,
        implementation=STUDY.implementation,
        scenario=scenario,
        replicate=replicate,
        estimands=ESTIMANDS,
    )


def cleverly_rows(
    frame: pd.DataFrame,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    """Fit and return one replication's subject rows."""
    return rows_from_result(fit_cleverly(frame), truth, scenario, replicate)


def _replicate(
    payload: tuple[str, int, int],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, n = payload
    frame, truth = draw_scenario(scenario, n, replicate)
    result = fit_cleverly(frame)
    sample = frame.copy()
    sample.insert(0, "fold", result.nuisance.folds.assignment)
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    truth_rows = [
        {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": value}
        for name, value in truth.items()
    ]
    return sample, truth_rows, rows_from_result(result, truth, scenario, replicate)


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Draw, fit, and retain the exact grouped folds for the R comparator."""
    outcomes = map_parallel(
        _replicate,
        [((SCENARIO, replicate, n),) for replicate in range(replicates)],
        n_jobs=n_jobs,
    )
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truths = pd.DataFrame([row for _, rows, _ in outcomes for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes for row in rows])
    return samples, truths, estimates.loc[:, list(REPLICATE_COLUMNS)]
