"""Independent properties for repeated point-treatment cross-fitting.

The inherited families come from :mod:`tests.studies.cvtmle_properties`, so they sample the
bounded cross-fitted laws and declare ``q_bounds``.  The repeat-stability family is this
study's own and stays on the binary law: it holds one sample fixed and varies the fold seed,
so a bounded outcome would change what the cells measure without changing what they answer,
and a binary outcome already has the identity scaler.

**Repeat stability.**  Every trial fits the three-draw median and two one-draw fits on the
fixed sample.  The two one-draw fits are ``repeats=1`` refits at the first and the second fold
seed that the three-draw plan derives, so each equals draw 0 and draw 1 of the three-draw fit.

==========================  ==========  =====================================================
cell                        role        what it reads
==========================  ==========  =====================================================
``three_repeats``           positive    the three-draw median
``one_repeat``              positive    draw 0 alone, the base of both ratios
``second_draw_control``     control     draw 1 alone
==========================  ==========  =====================================================

The positive rule reads ``SD(three_repeats) / SD(one_repeat)`` and passes when the 99%
paired-bootstrap upper endpoint is below :data:`~tests.studies.repeated_crossfit.MAX_REPEAT_SPREAD_RATIO`.
The control reads ``SD(second_draw_control) / SD(one_repeat)``.  Draws 0 and 1 are independent
fold draws on one sample with deterministic learners, so their spreads are equal in law and the
ratio is exactly 1 under the truth.  The control passes when its upper endpoint is at or above
the boundary, which a downward-biased ratio, a biased bootstrap or draws that are not
exchangeable would fail.  At a true ratio of 1 the 99% two-sided interval's upper endpoint
reaches 1 with probability about 0.995, so a red control is a defect suspect and never an owner
row.  The control resamples with its own stream, so its interval is not the positive's index
matrix.  The three rows of the family share one ``property_passed``: the positive and the
control verdicts together.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from cleverly.datasets import binary_outcome_dgp
from cleverly.learners.crossfit import CrossFitPlan
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_cvtmle import cv_fit
from tests.studies.cvtmle_properties import cells, generate, sampling_batch, summarize
from tests.studies.evidence import property_failures, property_verdicts
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    PropertyBatch,
    PropertyCell,
    paired_spread_ratio_interval,
    replicate_row,
    require_accounted,
)
from tests.studies.evidence.property_verdicts import finish
from tests.studies.evidence.replication import fit_properties, splat
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


#: The stability arms in publication order, each with its role.  ``one_repeat`` is the base
#: of both spread ratios and shares the positive's verdict; ``second_draw_control`` is the
#: control that must read no variance reduction.
STABILITY_ARMS = (
    ("three_repeats", "positive"),
    ("one_repeat", "positive"),
    ("second_draw_control", "control"),
)
POSITIVE_ARMS = ("three_repeats", "one_repeat")
CONTROL_ARM = "second_draw_control"


def repeat_stability_cells() -> tuple[PropertyCell, ...]:
    """The three-draw, first-draw and second-draw arms, as declarations of the law they sample.

    Every arm reads one sample of :func:`~cleverly.datasets.binary_outcome_dgp` and varies only
    the fold seed, so ``seed`` here names the stream that sample comes from rather than a
    per-replication stream, and ``replicates`` is the number of fold-seed trials.

    Returns
    -------
    tuple of PropertyCell
        The positive three-repeat arm, the positive one-repeat base arm, then the
        second-draw control.
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
        for cell, role in STABILITY_ARMS
    )


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every cell this study runs, shared and study-specific alike.

    Returns
    -------
    tuple of PropertyCell
        The cross-fitted families from :func:`tests.studies.cvtmle_properties.cells`,
        followed by the three repeat-stability arms.
    """
    return (*cells(VARIANT, include_overfitting=False), *repeat_stability_cells())


def _repeat_seeds(base_seed: int) -> tuple[int, int]:
    """The first two fold seeds that the estimator derives from a repeated fit's seed."""
    plan = CrossFitPlan(n_folds=N_FOLDS, random_state=base_seed, repeats=REPEATS)
    first, second = plan.seeds()[:2]
    if first is None or second is None:
        raise RuntimeError("the labelled repeat-stability seed unexpectedly derived None")
    return first, second


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
) -> tuple[dict[str, object], ...]:
    repeated = _stability_fit(frame, repeats=REPEATS, random_state=base_seed)
    first_seed, second_seed = _repeat_seeds(base_seed)
    first = _stability_fit(frame, repeats=1, random_state=first_seed)
    second = _stability_fit(frame, repeats=1, random_state=second_seed)

    draws = repeated.repeats
    if np.array_equal(draws[0].nuisance.folds.assignment, draws[1].nuisance.folds.assignment):
        raise RuntimeError(
            "draws 0 and 1 share one fold assignment: the second-draw control would be the "
            "first draw"
        )
    for index, one in enumerate((first, second)):
        draw = draws[index]
        if not np.array_equal(draw.nuisance.folds.assignment, one.nuisance.folds.assignment):
            raise RuntimeError(f"the one-repeat fit {index} is not the repeated fit's draw {index}")
        if draw.psi["ate"] != one.psi("ate"):
            raise RuntimeError(f"the one-repeat fit {index} does not reproduce draw {index}'s ATE")

    return tuple(
        replicate_row(
            property_name="repeat_stability",
            cell=cell,
            role=role,
            replicate=replicate,
            n=REPEAT_STABILITY_N,
            requested=requested,
            truth=truth,
            estimate=fit["ate"],
            alpha=STUDY.margins.alpha,
        )
        for (cell, role), fit in zip(STABILITY_ARMS, (repeated, first, second), strict=True)
    )


STABILITY_CELLS = tuple(f"repeat_stability/{cell}" for cell, _ in STABILITY_ARMS)


def _stability_cells_of(payload: tuple[Any, ...]) -> tuple[str, ...]:
    return STABILITY_CELLS


def _stability_cell_of(payload: tuple[Any, ...]) -> str:
    return "repeat_stability"


def _stability_replicate_of(payload: tuple[Any, ...]) -> int:
    return int(payload[2])


def generate_repeat_stability_rows(
    *, trials: int = FOLD_SEED_TRIALS, n_jobs: int = STUDY_JOBS
) -> pd.DataFrame:
    """Fit the three-draw report and draws 0 and 1 alone on one fixed binary sample."""
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
    trials_rows = fit_properties(
        payloads,
        splat(_stability_trial),
        cells_of=_stability_cells_of,
        cell_of=_stability_cell_of,
        replicate_of=_stability_replicate_of,
        n_jobs=n_jobs,
        label="properties/repeat_stability",
    )
    rows = pd.DataFrame([row for trial in trials_rows for row in trial])
    return rows.loc[:, list(REPLICATE_COLUMNS)]


def _bootstrap_seed(*labels: str) -> int:
    return stream_seed(STUDY, "repeat_stability", "bootstrap", *labels)


#: The control's own bootstrap stream, so its interval does not resample the positive's index
#: matrix.
CONTROL_STREAM = ("second_draw",)


def _summarize_repeat_stability(rows: pd.DataFrame, columns: pd.Index) -> pd.DataFrame:
    """Build fixed-sample stability rows without population-sampling endpoints.

    The positive rows publish the three-draw ratio and pass when its upper endpoint is below
    the boundary.  The control row publishes the second-draw ratio and passes when its upper
    endpoint is at or above it.  The three rows share the joint verdict.
    """
    require_accounted(rows)
    arms = {cell: rows.loc[rows["cell"] == cell] for cell, _ in STABILITY_ARMS}
    margins = STUDY.margins
    positive = paired_spread_ratio_interval(
        arms["three_repeats"],
        arms["one_repeat"],
        replicates=margins.bootstrap_replicates,
        confidence_level=margins.confidence_level,
        seed=_bootstrap_seed(),
    )
    control = paired_spread_ratio_interval(
        arms[CONTROL_ARM],
        arms["one_repeat"],
        replicates=margins.bootstrap_replicates,
        confidence_level=margins.confidence_level,
        seed=_bootstrap_seed(*CONTROL_STREAM),
    )
    positive_passed = bool(positive.interval.high < MAX_REPEAT_SPREAD_RATIO)
    control_passed = bool(control.interval.high >= MAX_REPEAT_SPREAD_RATIO)
    records: list[dict[str, object]] = []
    for cell, group in rows.groupby("cell", sort=True):
        is_control = cell == CONTROL_ARM
        result = control if is_control else positive
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
                "passed": control_passed if is_control else positive_passed,
                "property_passed": positive_passed and control_passed,
            }
        )
        records.append(record)
    return pd.DataFrame(records, columns=columns)


#: The share of a cell's requested replications that the accounting lets a run lose, declared
#: for the study's next property run (RM39b2 rev 3 section 6, confirmed 2026-10-14).  Up to
#: ``floor(0.01 R)`` lost replicates are judged by the worst case of each clause; one more is
#: ``bound exceeded``.
FAILURE_BOUND = 0.01

#: The draw-level statistic the accounting reports for each lost replicate (rev 3 D3'f).
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


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


def sampling_batches() -> tuple[PropertyBatch, ...]:
    """The allocated sample call; fixed-sample repeat stability is carried unchanged."""
    return (sampling_batch(VARIANT, repeats=REPEATS, n_folds=N_FOLDS, include_overfitting=False),)


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


def _stability_rule() -> property_failures.FamilyRule:
    """``repeat_stability``: S9 on the two paired spread ratios.

    A replicate is a fold seed on one fixed sample, so the survivor range is the range over
    fold seeds.  The positive clause reads ``three_repeats`` over ``one_repeat`` against
    ``U < MAX_REPEAT_SPREAD_RATIO``, and a lost estimate raises the ratio when the numerator
    sits at each vertex split and the denominator at the survivor mean.  The control clause
    reads ``second_draw_control`` over ``one_repeat`` against ``U >= MAX_REPEAT_SPREAD_RATIO``
    and reverses the fills.  The three rows share one joint verdict (``group="family"``).
    """

    def boundary(ctx: property_failures.Context) -> float:
        return MAX_REPEAT_SPREAD_RATIO

    return property_failures.FamilyRule(
        families=frozenset({"repeat_stability"}),
        shared="_summarize_repeat_stability",
        clauses=(
            property_failures.spread_ratio_clause(
                "repeat spread ratio under its boundary",
                cells=("three_repeats", "one_repeat"),
                side="upper",
                judge=property_failures.below_judge(boundary),
                seed=lambda ctx, row: stream_seed(ctx.record, "repeat_stability", "bootstrap"),
                applies=property_failures.cells(*POSITIVE_ARMS),
            ),
            property_failures.spread_ratio_clause(
                "control spread ratio reaches its boundary",
                cells=(CONTROL_ARM, "one_repeat"),
                side="lower",
                judge=property_failures.upper_at_or_above_judge(boundary),
                seed=lambda ctx, row: stream_seed(
                    ctx.record, "repeat_stability", "bootstrap", *CONTROL_STREAM
                ),
                applies=property_failures.cells(CONTROL_ARM),
            ),
        ),
        group="family",
    )


#: The failure accounting's declared rules (RM39b2 rev 3, D3'a).
PROPERTY_RULES = (_stability_rule(),)
