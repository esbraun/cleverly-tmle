"""Independent properties for repeated point-treatment cross-fitting.

The inherited families come from :mod:`tests.studies.cvtmle_properties`, so they sample the
bounded cross-fitted laws and declare ``q_bounds``.  The repeat-stability family is this
study's own and stays on the binary law: it holds one sample fixed and varies the fold seed,
so a bounded outcome would change what the cells measure without changing what they answer,
and a binary outcome already has the identity scaler.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from cleverly.datasets import binary_outcome_dgp
from cleverly.learners.crossfit import CrossFitPlan
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_cvtmle import cv_fit
from tests.studies.cvtmle_properties import cells, generate, summarize
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    PropertyCell,
    paired_spread_ratio_interval,
    replicate_row,
    require_complete,
)
from tests.studies.evidence.property_verdicts import finish
from tests.studies.evidence.seeds import stream_seed
from tests.studies.repeated_crossfit import (
    FOLD_SEED_TRIALS,
    MAX_REPEAT_SPREAD_RATIO,
    N_FOLDS,
    REPEAT_STABILITY_N,
    REPEATS,
    STUDY,
    draw_from_seed,
)

#: The variant name the shared module labels this row's cells with.
VARIANT = "repeated"

REPEAT_STABILITY_COLUMNS = (
    "spread_ratio",
    "spread_ratio_ci_lower",
    "spread_ratio_ci_upper",
    "spread_ratio_boundary",
)


def _unpenalized_logistic() -> LogisticRegression:
    """The learner :func:`~tests.studies.canonical_cvtmle.cv_fit` builds for a binary law.

    Restated here because :func:`declared_cells` has to name a learner and the stability
    arms are fitted through ``cv_fit`` rather than through
    :func:`~tests.studies.evidence.properties.run_cells`.  Nothing reads this copy during a
    run, so a drift between the two would change no published row.

    Returns
    -------
    LogisticRegression
        An unpenalized main-effects logistic regression.
    """
    return LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs")


def repeat_stability_cells() -> tuple[PropertyCell, ...]:
    """The paired three-draw and one-draw arms, as declarations of the law they sample.

    Both arms read one sample of :func:`~cleverly.datasets.binary_outcome_dgp` and vary only
    the fold seed, so ``seed`` here names the stream that sample comes from rather than a
    per-replication stream, and ``replicates`` is the number of fold-seed trials.

    Returns
    -------
    tuple of PropertyCell
        The positive three-repeat arm, then its one-repeat control.
    """
    dgp = binary_outcome_dgp()
    sample_seed = stream_seed(STUDY, "repeat_stability", "sample")
    return tuple(
        PropertyCell(
            property="repeat_stability",
            cell=cell,
            dgp=dgp,
            outcome_learner=_unpenalized_logistic,
            treatment_learner=_unpenalized_logistic,
            n=REPEAT_STABILITY_N,
            replicates=FOLD_SEED_TRIALS,
            seed=sample_seed,
            role=role,
        )
        for cell, role in (("three_repeats", "positive"), ("one_repeat_control", "control"))
    )


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every cell this study runs, shared and study-specific alike.

    Returns
    -------
    tuple of PropertyCell
        The cross-fitted families from :func:`tests.studies.cvtmle_properties.cells`,
        followed by the two repeat-stability arms.
    """
    return (*cells(VARIANT, include_overfitting=False), *repeat_stability_cells())


def _first_repeat_seed(base_seed: int) -> int:
    """The first fold seed that the estimator derives from a repeated fit's seed."""
    plan = CrossFitPlan(n_folds=N_FOLDS, random_state=base_seed, repeats=REPEATS)
    seed = plan.seeds()[0]
    if seed is None:
        raise RuntimeError("the labelled repeat-stability seed unexpectedly derived None")
    return seed


def _stability_fit(frame: pd.DataFrame, *, repeats: int, random_state: int) -> Any:
    return cv_fit(
        frame,
        binary=True,
        estimands=("ate",),
        n_folds=N_FOLDS,
        repeats=repeats,
        cv_evaluation=False,
        random_state=random_state,
    )


def _stability_trial(
    frame: pd.DataFrame,
    truth: float,
    replicate: int,
    requested: int,
    base_seed: int,
) -> tuple[dict[str, object], dict[str, object]]:
    repeated = _stability_fit(frame, repeats=REPEATS, random_state=base_seed)
    first_seed = _first_repeat_seed(base_seed)
    control = _stability_fit(frame, repeats=1, random_state=first_seed)

    repeated_first = repeated.repeats[0].nuisance.folds.assignment
    control_folds = control.nuisance.folds.assignment
    if not np.array_equal(repeated_first, control_folds):
        raise RuntimeError("the one-repeat control is not the repeated fit's first fold draw")
    if repeated.repeats[0].psi["ate"] != control.psi("ate"):
        raise RuntimeError("the one-repeat control does not reproduce the first draw's ATE")

    return (
        replicate_row(
            property_name="repeat_stability",
            cell="three_repeats",
            role="positive",
            replicate=replicate,
            n=REPEAT_STABILITY_N,
            requested=requested,
            truth=truth,
            estimate=repeated["ate"],
            alpha=STUDY.margins.alpha,
        ),
        replicate_row(
            property_name="repeat_stability",
            cell="one_repeat_control",
            role="control",
            replicate=replicate,
            n=REPEAT_STABILITY_N,
            requested=requested,
            truth=truth,
            estimate=control["ate"],
            alpha=STUDY.margins.alpha,
        ),
    )


def generate_repeat_stability_rows(
    *, trials: int = FOLD_SEED_TRIALS, n_jobs: int = STUDY_JOBS
) -> pd.DataFrame:
    """Fit paired one- and three-repeat reports on one fixed binary sample."""
    sample_seed = stream_seed(STUDY, "repeat_stability", "sample")
    frame, truth = draw_from_seed("binary", REPEAT_STABILITY_N, sample_seed)
    payloads = [
        (
            frame,
            float(truth["ate"]),
            replicate,
            trials,
            stream_seed(STUDY, "repeat_stability", "fold_seed", replicate),
        )
        for replicate in range(trials)
    ]
    paired = map_parallel(_stability_trial, payloads, n_jobs=n_jobs)
    rows = pd.DataFrame([row for pair in paired for row in pair])
    return rows.loc[:, list(REPLICATE_COLUMNS)]


def _summarize_repeat_stability(rows: pd.DataFrame, columns: pd.Index) -> pd.DataFrame:
    """Build fixed-sample stability rows without population-sampling endpoints."""
    require_complete(rows)
    repeated = rows.loc[rows["cell"] == "three_repeats"]
    control = rows.loc[rows["cell"] == "one_repeat_control"]
    result = paired_spread_ratio_interval(
        repeated,
        control,
        replicates=STUDY.margins.bootstrap_replicates,
        confidence_level=STUDY.margins.confidence_level,
        seed=stream_seed(STUDY, "repeat_stability", "bootstrap"),
    )
    passed = result.interval.high < MAX_REPEAT_SPREAD_RATIO
    records: list[dict[str, object]] = []
    for cell, group in rows.groupby("cell", sort=True):
        record = dict.fromkeys(columns, np.nan)
        record.update(
            {
                "property": "repeat_stability",
                "cell": cell,
                "role": group["role"].iloc[0],
                "n": int(group["n"].iloc[0]),
                "replicates": len(group),
                "failed_replicates": int(group["failed_replicates"].iloc[0]),
                "spread_ratio": result.ratio,
                "spread_ratio_ci_lower": result.interval.low,
                "spread_ratio_ci_upper": result.interval.high,
                "spread_ratio_boundary": MAX_REPEAT_SPREAD_RATIO,
                "passed": passed,
                "property_passed": passed,
            }
        )
        records.append(record)
    return pd.DataFrame(records, columns=columns)


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    inherited = generate(
        VARIANT,
        repeats=REPEATS,
        n_folds=N_FOLDS,
        include_overfitting=False,
        n_jobs=n_jobs,
    )
    stability = generate_repeat_stability_rows(n_jobs=n_jobs)
    return pd.concat([inherited, stability], ignore_index=True)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    stability_mask = rows["property"] == "repeat_stability"
    inherited = rows.loc[~stability_mask]
    stability = rows.loc[stability_mask]
    summary, rates = summarize(
        inherited,
        STUDY,
        VARIANT,
        include_overfitting=False,
        extra_columns=REPEAT_STABILITY_COLUMNS,
        return_parts=True,
    )
    summary = pd.concat(
        [summary, _summarize_repeat_stability(stability, summary.columns)], ignore_index=True
    )
    return finish(summary, rates)
