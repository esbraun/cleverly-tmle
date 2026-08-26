"""Repeated-sampling properties for cross-fitted competing-risk LTMLE."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from cleverly.datasets import make_longitudinal_survival
from cleverly.longitudinal import LTMLE
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import canonical_ltmle_competing_crossfit as canonical
from tests.studies import ltmle_competing_properties as base
from tests.studies.canonical_ltmle import KnownLongitudinalMechanism
from tests.studies.evidence.properties import REPLICATE_COLUMNS, replicate_row
from tests.studies.evidence.seeds import stream_seed

STUDY = canonical.STUDY
PROPERTY_LABELS = base.PROPERTY_LABELS
CONTRASTS = base.CONTRASTS
EFFICIENCY_SD = base.EFFICIENCY_SD
OVERFIT_REPLICATES = 8_000
OVERFIT_N = 1_000
OVERFIT_POSITIVE = "cross_fitted_competing_ltmle"


def _balanced_competing_panel(n: int, seed: int) -> tuple[pd.DataFrame, float]:
    """Split each all-cause event independently between two equally likely causes."""
    frame, truth = make_longitudinal_survival(n=n, seed=seed, censoring=True, backend="pandas")
    rng = np.random.default_rng(seed + 1_000_003)
    first_relapse = rng.binomial(1, 0.5, size=n).astype(float)
    second_relapse = rng.binomial(1, 0.5, size=n).astype(float)
    y1 = frame.pop("Y1").to_numpy()
    y2 = frame.pop("Y2").to_numpy()
    first_event = y1 == 1.0
    second_event = (y1 == 0.0) & (y2 == 1.0)
    observed1 = np.isfinite(y1)
    observed2 = np.isfinite(y2)

    r1 = np.where(observed1, first_event * first_relapse, np.nan)
    d1 = np.where(observed1, first_event * (1.0 - first_relapse), np.nan)
    r2 = np.where(first_event, r1, np.where(observed2, second_event * second_relapse, np.nan))
    d2 = np.where(
        first_event,
        d1,
        np.where(observed2, second_event * (1.0 - second_relapse), np.nan),
    )
    frame[["R1", "D1", "R2", "D2"]] = np.column_stack([r1, d1, r2, d2])
    source = "ate_regimen[always vs never @ t=2]"
    return frame, 0.5 * float(truth[source])


def fit(frame: pd.DataFrame, configuration: str = "both_correct") -> Any:
    return base.fit(frame, configuration, n_folds=5)


def _overfit(frame: pd.DataFrame, *, cross_fit: bool) -> Any:
    return LTMLE(
        {"never": 0, "always": 1},
        reference="never",
        outcome_learner=DecisionTreeClassifier(min_samples_leaf=1, random_state=0),
        pseudo_learner=DecisionTreeRegressor(min_samples_leaf=1, random_state=0),
        treatment_learner=KnownLongitudinalMechanism("treatment"),
        censoring_learner=KnownLongitudinalMechanism("censoring"),
        n_folds=5 if cross_fit else 1,
        learner_folds=5,
        g_bounds=base.canonical.G_BOUNDS,
        simultaneous=False,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    ).fit(
        frame,
        outcome={"relapse": ["R1", "R2"], "death": ["D1", "D2"]},
        treatment=["A1", "A2"],
        baseline=["W1", "W2"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
    )


def _overfit_replication(payload: tuple[str, int, int, int, int]) -> list[dict[str, Any]]:
    cell, replicate, n, requested, seed = payload
    frame, truth = _balanced_competing_panel(n, seed)
    result = _overfit(frame, cross_fit=cell == OVERFIT_POSITIVE)
    source_name = "ate_regimen[always vs never, relapse @ t=2]"
    name = source_name
    return [
        replicate_row(
            property_name="crossfit_overfitting",
            cell=cell,
            role="positive" if cell == OVERFIT_POSITIVE else "control",
            replicate=replicate,
            n=n,
            requested=requested,
            truth=truth,
            estimate=result[name],
            alpha=STUDY.margins.alpha,
        )
    ]


def _overfit_payloads() -> list[tuple[tuple[str, int, int, int, int]]]:
    return [
        (
            (
                cell,
                replicate,
                OVERFIT_N,
                OVERFIT_REPLICATES,
                stream_seed(STUDY, "property_sample", "crossfit_overfitting", "paired", replicate),
            ),
        )
        for cell in (OVERFIT_POSITIVE, "in_sample_control")
        for replicate in range(OVERFIT_REPLICATES)
    ]


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    rows = base.generate_for(
        STUDY, lambda frame, configuration: fit(frame, configuration), n_jobs=n_jobs
    )
    overfit = map_parallel(_overfit_replication, _overfit_payloads(), n_jobs=n_jobs)
    combined = pd.concat(
        [rows, pd.DataFrame([row for result in overfit for row in result])], ignore_index=True
    )
    return combined.loc[:, list(REPLICATE_COLUMNS)].sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    return base.summarize_for(rows, STUDY, crossfit_positive_cell=OVERFIT_POSITIVE)
