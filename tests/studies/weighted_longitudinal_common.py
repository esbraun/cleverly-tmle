"""Shared implementation for weighted end-of-study longitudinal evidence rows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from cleverly.datasets import RULE_LABEL, make_longitudinal, rule_arm_at_node_two
from cleverly.longitudinal import LTMLE
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_ltmle import (
    KnownLongitudinalMechanism,
    QuasiBinomialGLM,
    regimen_initials,
    regimen_rows,
)
from tests.studies.evidence.registry import StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate

PRIMARY_REPLICATES = 800
PRIMARY_N = 2_000
SCENARIO = "selected_censored_end_of_study"
G_BOUNDS = (1e-8, 1.0)
WEIGHT_COLUMN = "obs_weight"
SELECTION_LOW = 0.3
SELECTION_HIGH = 0.9

REGIMENS: dict[str, Any] = {
    "never": 0,
    "always": 1,
    RULE_LABEL: (1, lambda history: rule_arm_at_node_two(history["L2"])),
}
REFERENCE = "never"
MEAN_NAMES = tuple(f"ey_regimen[{label}]" for label in REGIMENS)
CONTRAST_NAMES = (
    "ate_regimen[always vs never]",
    f"ate_regimen[{RULE_LABEL} vs never]",
)
ESTIMANDS = (*MEAN_NAMES, *CONTRAST_NAMES)


def selection_probability(w1: Any) -> np.ndarray:
    """Return the declared row-selection probability from baseline ``W1``."""
    values = np.asarray(w1, dtype=float)
    return np.where(values > 0.0, SELECTION_LOW, SELECTION_HIGH)


def sample_selected_exact(n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw exactly ``n`` accepted rows with one explicit generator.

    Each source-law batch and each acceptance draw consumes the same generator. Batches
    continue until the accepted sample reaches ``n``; the final batch is truncated only
    after its accept/reject decisions are realized.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    accepted: list[pd.DataFrame] = []
    remaining = n
    truth: dict[str, float] | None = None
    while remaining:
        # The expected retention is 0.6. Two source rows per needed selected row makes
        # another iteration uncommon while keeping small smoke draws inexpensive.
        source_n = max(64, 2 * remaining)
        source, source_truth = make_longitudinal(
            n=source_n, seed=rng, censoring=True, backend="pandas"
        )
        truth = {name: float(source_truth[name]) for name in ESTIMANDS}
        probability = selection_probability(source["W1"].to_numpy())
        keep = rng.random(source_n) < probability
        selected = source.loc[keep].copy()
        selected[WEIGHT_COLUMN] = 1.0 / probability[keep]
        accepted.append(selected.iloc[:remaining])
        remaining -= min(remaining, len(selected))
    if truth is None:  # pragma: no cover - guarded by n > 0
        raise AssertionError("the selected sampler produced no source batch")
    frame = pd.concat(accepted, ignore_index=True)
    if len(frame) != n:  # pragma: no cover - declaration guard
        raise AssertionError(f"selected sampler returned {len(frame)} rows, expected {n}")
    return frame, truth


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    if scenario != SCENARIO:
        raise KeyError(scenario)
    return sample_selected_exact(n, seed)


def draw_scenario(
    record: StudyRecord, scenario: str, n: int, replicate: int
) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(record, draw_from_seed, scenario, n, replicate)


def fit_cleverly(frame: pd.DataFrame, *, cross_fit: bool) -> Any:
    """Fit the shared weighted GLM construction with one or five outer folds."""
    return LTMLE(
        REGIMENS,
        reference=REFERENCE,
        outcome_learner=QuasiBinomialGLM(),
        pseudo_learner=QuasiBinomialGLM(),
        treatment_learner=KnownLongitudinalMechanism("treatment"),
        censoring_learner=KnownLongitudinalMechanism("censoring"),
        n_folds=5 if cross_fit else 1,
        learner_folds=2 if cross_fit else 5,
        g_bounds=G_BOUNDS,
        simultaneous=False,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    ).fit(
        frame,
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["W1", "W2"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
        weights=WEIGHT_COLUMN,
    )


def rows_from_result(
    record: StudyRecord,
    result: Any,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    return regimen_rows(
        record,
        result,
        truth,
        regimen_initials(result, REGIMENS, CONTRAST_NAMES),
        ESTIMANDS,
        scenario,
        replicate,
        n=result.n,
    )


def replicate(
    record: StudyRecord,
    payload: tuple[str, int, int],
    *,
    cross_fit: bool,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate_index, n = payload
    frame, truth = draw_scenario(record, scenario, n, replicate_index)
    result = fit_cleverly(frame, cross_fit=cross_fit)
    sample = frame.copy()
    if cross_fit:
        sample.insert(0, "fold", result.folds.assignment)
    else:
        sample.insert(0, "row", np.arange(len(sample)))
    sample.insert(0, "replicate", replicate_index)
    sample.insert(0, "scenario", scenario)
    truths = [
        {
            "scenario": scenario,
            "replicate": replicate_index,
            "estimand": name,
            "truth": value,
        }
        for name, value in truth.items()
    ]
    return sample, truths, rows_from_result(record, result, truth, scenario, replicate_index)


def _replicate_dispatch(
    record: StudyRecord,
    payload: tuple[str, int, int],
    cross_fit: bool,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    return replicate(record, payload, cross_fit=cross_fit)


def draw_and_fit(
    record: StudyRecord,
    *,
    replicates: int,
    n: int,
    cross_fit: bool,
    n_jobs: int = STUDY_JOBS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    payloads = [(record, (SCENARIO, index, n), cross_fit) for index in range(replicates)]
    outcomes = map_parallel(_replicate_dispatch, payloads, n_jobs=n_jobs)
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truths = pd.DataFrame([row for _, rows, _ in outcomes for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes for row in rows])
    return samples, truths, estimates.loc[:, list(REPLICATE_COLUMNS)]
