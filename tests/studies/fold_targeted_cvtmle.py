"""Paired evidence for fold-targeted, fold-evaluated point-treatment CV-TMLE."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from numpy.random import RandomState

from cleverly.data import CausalData
from cleverly.estimators import TMLE
from cleverly.learners.crossfit import Folds, check_integrity
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_cvtmle import G_BOUNDS, cv_fit, rows_from_result
from tests.studies.canonical_tmle import draw_from_seed as canonical_tmle_draw_from_seed
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate

ZEPID_COMMIT = "16a0f96f8b2c65df8715085801f21757d1478e1e"
PYTHON_BASE_IMAGE = (
    "python:3.10-slim-bookworm@sha256:"
    "673f009e3763f8d03953b525c89b03a9ee8ca315bf6b8006979b5c4a4e2e4d68"
)
PRIMARY_REPLICATES = 1_600
PRIMARY_N = 1_000
SEED = 20240822
N_FOLDS = 2
SUPPORTED = ("ate",)
SCENARIO_ESTIMANDS = {"binary": SUPPORTED}

PROPERTY_CELLS = {
    "double_robustness": (
        "both_correct",
        "outcome_correct",
        "treatment_correct",
        "both_wrong",
    ),
    "root_n_and_efficiency": ("n_500", "n_2000", "n_8000"),
    "root_n_rate": ("empirical_sd", "reported_se"),
    "interval_calibration": ("correctly_specified",),
    "type_i_error": ("sharp_null",),
    "power": ("alternative",),
    "crossfit_overfitting": ("fold_targeted_cvtmle", "in_sample_control"),
}

STUDY = StudyRecord(
    name="fold-targeted point-treatment CV-TMLE",
    slug="fold-targeted-cvtmle",
    artifacts=ROOT / "tests" / "canonical" / "zepid_cvtmle",
    document=("docs/technical-reference/method-evidence/fold-targeted-point-treatment-cv-tmle.md"),
    anchor="fold-targeted-point-treatment-cv-tmle",
    scenarios=SCENARIO_ESTIMANDS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=20261026,
    margins=Margins(),
    implementation="cleverly-fold-targeted-cvtmle",
    reference="zepid-single-crossfit-tmle",
    modules=(
        "tests/studies/fold_targeted_cvtmle.py",
        "tests/studies/canonical_cvtmle.py",
        "tests/studies/canonical_tmle.py",
        "tests/studies/canonical_properties.py",
        "tests/studies/cvtmle_properties.py",
        "tests/studies/fold_targeted_cvtmle_properties.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.fold_targeted_cvtmle",
    properties_module="tests.studies.fold_targeted_cvtmle_properties",
    property_cells=PROPERTY_CELLS,
)

REFERENCE_METADATA = {
    "zepid_commit": ZEPID_COMMIT,
    "zepid_version": "0.9.1",
    "python_base_image": PYTHON_BASE_IMAGE,
}

CONFIGURATION = {
    "cross_fit": True,
    "n_folds": N_FOLDS,
    "partitions": 1,
    "repeats": 1,
    "targeting_scheme": "fold",
    "cv_evaluation": True,
    "simultaneous_intervals": False,
    "g_bounds": list(G_BOUNDS),
    "q_bounds": "binary outcome",
    "folds": "identical equal-size row assignments supplied to both implementations",
    "fold_aggregation": (
        "equal 1/V plug-in average; identical to size weighting here only because folds are equal"
    ),
}


class FixedFoldTMLE(TMLE):
    """Study-only TMLE whose outer fold assignment is supplied by the sample."""

    def __init__(self, assignment: np.ndarray, **kwargs: Any) -> None:
        self._assignment = np.asarray(assignment, dtype=np.int64)
        super().__init__(**kwargs)

    def _folds(self, data: CausalData, seed: int | None = None) -> Folds:
        del seed
        if data.n != self._assignment.size:
            raise ValueError(
                f"fold assignment has {self._assignment.size} rows for a {data.n}-row fit"
            )
        folds = Folds(self._assignment.copy(), N_FOLDS)
        check_integrity(folds, cluster=data.cluster)
        return folds


def zepid_partition(n: int, seed: int) -> tuple[np.ndarray, int]:
    """Reproduce zEpid's one-partition two-split draw from a labelled study seed.

    The second element is the *study* seed, not the partition seed.  ``run_zepid_cvtmle.py``
    reads it out of the ``partition_random_state`` column and hands it to zEpid, which then
    re-derives ``partition_seed`` by the same draw below.  Passing the derived seed instead
    would make zEpid draw a third one and split the rows differently.  The column name is
    read by that runner, whose bytes the manifest hashes, so it is not renamable here.
    """
    if n % N_FOLDS:
        raise ValueError(f"the fold-targeted study needs two equal folds; got n={n}")
    partition_seed = int(RandomState(seed).choice(range(5_000_000), size=1, replace=False)[0])
    positions = pd.DataFrame({"row_id": np.arange(n, dtype=np.int64)})
    first = positions.sample(n=n // N_FOLDS, random_state=RandomState(partition_seed)).index
    assignment = np.ones(n, dtype=np.int64)
    assignment[first.to_numpy()] = 0
    return assignment, seed


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw the binary law and its exact zEpid-native fold partition from one seed."""
    if scenario != "binary":
        raise KeyError(scenario)
    frame, truth = canonical_tmle_draw_from_seed(scenario, n, seed)
    assignment, partition_random_state = zepid_partition(n, seed)
    frame = frame.copy()
    frame.insert(0, "partition_random_state", partition_random_state)
    frame.insert(0, "fold", assignment)
    frame.insert(0, "row_id", np.arange(n, dtype=np.int64))
    return frame, truth


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One fixed replication from this study's scenario-and-replication seed stream."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def fit_cleverly(frame: pd.DataFrame, scenario: str = "binary") -> Any:
    """Fit the registered fold-targeted construction on the stored assignment."""
    if scenario != "binary":
        raise KeyError(scenario)
    assignment = frame["fold"].to_numpy(dtype=np.int64)
    return cv_fit(
        frame,
        binary=True,
        estimands=SUPPORTED,
        n_folds=N_FOLDS,
        targeting_scheme="fold",
        cv_evaluation=True,
        estimator_factory=partial(FixedFoldTMLE, assignment),
    )


def cleverly_rows(
    frame: pd.DataFrame,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    result = fit_cleverly(frame, scenario)
    return rows_from_result(
        STUDY,
        result,
        truth,
        scenario,
        replicate,
        initial_estimates={"ate": initial_fold_ate(result)},
    )


def initial_fold_ate(result: Any) -> float:
    """Equal ``1/V`` average of the exposed pre-targeting fold ATE plug-ins."""
    q_difference = result.nuisance.outcome.arms[1.0] - result.nuisance.outcome.arms[0.0]
    fold_values = [
        result.nuisance.scaler.unscale_difference(float(np.mean(q_difference[test])))
        for _, test in result.nuisance.folds
    ]
    return float(np.mean(fold_values))


def _replicate(
    payload: tuple[str, int, int],
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    scenario, replicate, n = payload
    frame, truth = draw_scenario(scenario, n, replicate)
    result = fit_cleverly(frame, scenario)
    if not np.array_equal(result.nuisance.folds.assignment, frame["fold"].to_numpy()):
        raise RuntimeError("the fixed-fold cleverly fit changed the stored assignment")
    payload_frame = frame.copy()
    payload_frame.insert(0, "replicate", replicate)
    payload_frame.insert(0, "scenario", scenario)
    truth_row = {
        "scenario": scenario,
        "replicate": replicate,
        **{f"truth_{name}": value for name, value in truth.items()},
    }
    rows = rows_from_result(
        STUDY,
        result,
        truth,
        scenario,
        replicate,
        initial_estimates={"ate": initial_fold_ate(result)},
    )
    return payload_frame, truth_row, rows


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Draw, store, and fit every declared paired replication."""
    payloads = [
        ((scenario, replicate, n),)
        for scenario in STUDY.scenarios
        for replicate in range(replicates)
    ]
    outcomes = map_parallel(_replicate, payloads, n_jobs=n_jobs)
    samples = pd.concat([frame for frame, _, _ in outcomes], ignore_index=True)
    truths = pd.DataFrame([truth for _, truth, _ in outcomes])
    rows = pd.DataFrame([row for _, _, records in outcomes for row in records])
    return samples, truths, rows.loc[:, list(REPLICATE_COLUMNS)]
