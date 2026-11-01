"""Repeated-sampling properties of the cross-fitted learned rule over three arms.

Four families, in the order the declaration reads them. Every fit is at n = 2,000 on the one law
``three_arms``, fold-evaluated, with ten folds and pooled targeting.

``interval_calibration``
    The ``declared`` (correct) outcome learner (``three_arms__declared``), with the two derived
    controls of :func:`~tests.studies.evidence.property_verdicts.calibration_controls`: the SE
    shrunken by :data:`SHRUNKEN_SE_FACTOR`, and noise of one efficiency-bound unit added to each
    estimate.
``conservative_coverage``
    The ``missing_quadratic`` learner, held to the bias leg and to a coverage interval whose lower
    bound is at or above the calibration floor, with no SE-ratio gate (Corollary 3's route). The
    control is the same estimate with its SE times
    :data:`~tests.studies.learned_rule_arms.COVERAGE_SHRINK`, whose coverage interval must lie
    wholly below the floor.
``targeting_necessity``
    The ``drop_w1`` learner: the targeted value against the untargeted fold average of the
    initial plug-ins.
``rule_margin_diagnostic``
    For each law cell, the same estimate as its positive with the fit's margin columns
    (:func:`~tests.studies._learned_rule_arms_law.margin_columns`), role ``diagnostic``, no
    verdict.

Every statistic reads the error ``estimate - truth`` of each row, because the record declares
``truth_varies_by_replicate``. A property draw's sample seed is
``stream_seed(record, "property_sample", family, label, replicate, *suffix)`` and its fold seed
the same tuple with ``"fold_partition"`` first; rule L3 of the RM30 studies sets each suffix.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import _learned_rule_arms_law as law
from tests.studies.evidence import property_failures as pf
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    cell_groups,
    control_row,
    payload_cell,
    payload_replicate,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    DIAGNOSTIC_ROLE,
    apply_shared_verdicts,
    calibration_controls,
    calibration_derived,
    calibration_verdicts,
    finish,
    necessity_verdicts,
)
from tests.studies.evidence.registry import StudyRecord
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.seeds import stream_seed
from tests.studies.learned_rule_arms import (
    CALIBRATION_LABELS,
    COVERAGE_CONTROL,
    COVERAGE_FAMILY,
    COVERAGE_LABEL,
    COVERAGE_SHRINK,
    EFFICIENCY_BOUNDS,
    LABEL_LEARNERS,
    MARGIN,
    MARGIN_FAMILY,
    NECESSITY_FAMILY,
    NECESSITY_LABEL,
    POSITIVE,
    PRIMARY_LAW,
    PRIMARY_N,
    PRIMARY_REPLICATES,
    PROPERTY_CELLS,
    STUDY,
    harness_columns,
)

base = law.base

CALIBRATION_N = PRIMARY_N
#: The ``interval_calibration`` draws: the per-arm rule of
#: :func:`~tests.studies.mtp_study_tools.per_arm_replicates` on the one positive (the arm-pooled
#: rule needs two arms), read from the 4,000-draw pilot at the declared size
#: (``test_learned_rule_arms_design``).
CALIBRATION_REPLICATES = 3_200
#: The ``conservative_coverage`` draws: the smallest multiple of 400, at least 1,000, whose
#: predictive probability that both cells pass (the positive's bias leg and coverage floor, the
#: control's coverage ceiling) is at least 0.95 on the 4,000-draw pilot
#: (``test_learned_rule_arms_design``).
COVERAGE_REPLICATES = 1_200
#: The ``targeting_necessity`` draws: the smallest multiple of 400, at least 1,000, whose
#: predictive probability that the positive arm's bias leg passes is at least 0.95 on the
#: 4,000-draw pilot; the control's bias and the displacement sit far past their thresholds
#: (``test_learned_rule_arms_design``).
NECESSITY_REPLICATES = 1_200
#: The plan's shrunken-SE control of the calibration family: the declared SE times 0.8.
SHRUNKEN_SE_FACTOR = 0.80
#: The ``targeting_necessity`` displacement floor, in positive-arm SDs (the RM30 value).
TARGETING_DISPLACEMENT = 0.5

PROPERTY_DRAWS: tuple[base.PropertyDraw, ...] = (
    base.PropertyDraw("interval_calibration", PRIMARY_LAW, CALIBRATION_REPLICATES, PRIMARY_LAW),
    base.PropertyDraw(COVERAGE_FAMILY, COVERAGE_LABEL, COVERAGE_REPLICATES, PRIMARY_LAW),
    base.PropertyDraw(
        NECESSITY_FAMILY, f"{PRIMARY_LAW}_{NECESSITY_LABEL}", NECESSITY_REPLICATES, PRIMARY_LAW
    ),
)

#: The outcome learner of each property family's draws.
FAMILY_LEARNERS = {
    "interval_calibration": "declared",
    COVERAGE_FAMILY: "missing_quadratic",
    NECESSITY_FAMILY: "drop_w1",
}

#: The label each law family files its positive, control and diagnostic under.
FAMILY_LABELS = {"interval_calibration": PRIMARY_LAW, COVERAGE_FAMILY: COVERAGE_LABEL}


def suffixes(record: StudyRecord) -> dict[tuple[str, str], tuple[Any, ...]]:
    """The rule-L3 suffix of every property draw of ``record``."""
    return base.seed_labels(record, PRIMARY_REPLICATES, PROPERTY_DRAWS)


#: The calibration positive arm to the controls derived from its rows.
DERIVED_CELLS = calibration_derived(CALIBRATION_LABELS, positive_suffix=POSITIVE)

#: The margin columns of a ``rule_margin_diagnostic`` row. Every other row leaves them empty.
MARGIN_COLUMNS = ("limit_disagreement", "near_tie_share", "min_arm_share")

#: The columns every property row carries beside the shared schema.
ROW_COLUMNS = (*law.HARNESS_COLUMNS, *MARGIN_COLUMNS)

#: The descriptive summary columns of a ``rule_margin_diagnostic`` cell: the mean of each margin
#: column, and the share of draws whose realized rule assigns some arm to no row. No verdict
#: reads them.
MARGIN_SUMMARY = (
    "mean_limit_disagreement",
    "mean_near_tie_share",
    "mean_min_arm_share",
    "lost_arm_share",
)

#: The integer harness columns, which a derived control must keep integer.
INTEGER_COLUMNS = ("rule_rows_checked", "solver_warnings")

_Key = tuple[tuple[str, str, int, int, int, int, int, str]]


def _draw(key: _Key) -> pd.DataFrame:
    _, _, _, n, _, sample_seed, _, _ = key[0]
    return law.draw(n, sample_seed)


def law_cells(family: str) -> list[str]:
    """Every ``property/cell`` one draw of a law family writes, its derived controls included."""
    label = FAMILY_LABELS[family]
    kinds = (POSITIVE,) if family == "interval_calibration" else (POSITIVE, COVERAGE_CONTROL)
    written = [f"{family}/{label}__{kind}" for kind in kinds]
    written.append(f"{MARGIN_FAMILY}/{label}__{MARGIN}")
    for cell in list(written):
        written += DERIVED_CELLS.get(cell, ())
    return written


def _cells_of(key: _Key) -> list[str]:
    family = key[0][0]
    if family == NECESSITY_FAMILY:
        return [f"{NECESSITY_FAMILY}/{cell}" for cell in PROPERTY_CELLS[NECESSITY_FAMILY]]
    return law_cells(family)


def _row(
    family: str,
    cell: str,
    role: str,
    where: tuple[int, int, int],
    truth: float,
    estimate: Any,
    columns: dict[str, Any],
) -> dict[str, Any]:
    replicate, n, requested = where
    row = replicate_row(
        property_name=family,
        cell=cell,
        role=role,
        replicate=replicate,
        n=n,
        requested=requested,
        truth=truth,
        estimate=estimate,
        alpha=STUDY.margins.alpha,
    )
    row.update(columns)
    return row


def _control(
    family: str,
    cell: str,
    where: tuple[int, int, int],
    truth: float,
    estimate: float,
    standard_error: float,
    columns: dict[str, Any],
) -> dict[str, Any]:
    replicate, n, requested = where
    row = control_row(
        property_name=family,
        cell=cell,
        replicate=replicate,
        n=n,
        requested=requested,
        truth=truth,
        estimate=estimate,
        standard_error=standard_error,
        critical=law.CRITICAL,
        role="control",
    )
    row.update(columns)
    return row


def law_rows(fitted: law.Measured, family: str, where: tuple[int, int, int]) -> list[dict]:
    """The rows one law draw writes: its positive, its direct control and its margin row."""
    label = FAMILY_LABELS[family]
    estimate = fitted.result.estimates[law.ESTIMAND]
    truth = fitted.truth.truth
    columns = harness_columns(fitted)
    rows = [_row(family, f"{label}__{POSITIVE}", "positive", where, truth, estimate, columns)]
    if family == COVERAGE_FAMILY:
        rows.append(
            _control(
                family,
                f"{label}__{COVERAGE_CONTROL}",
                where,
                truth,
                float(estimate.psi),
                COVERAGE_SHRINK * float(estimate.std_error),
                columns,
            )
        )
    margin = {**columns, **law.margin_columns(fitted.result, LABEL_LEARNERS[label])}
    rows.append(
        _row(MARGIN_FAMILY, f"{label}__{MARGIN}", DIAGNOSTIC_ROLE, where, truth, estimate, margin)
    )
    return rows


def necessity_rows(fitted: law.Measured, where: tuple[int, int, int]) -> list[dict]:
    """The targeted and the untargeted value of one ``drop_w1`` fit."""
    estimate = fitted.result.estimates[law.ESTIMAND]
    truth = fitted.truth.truth
    columns = harness_columns(fitted)
    targeted, untargeted = PROPERTY_CELLS[NECESSITY_FAMILY]
    return [
        _row(NECESSITY_FAMILY, targeted, "positive", where, truth, estimate, columns),
        _control(
            NECESSITY_FAMILY,
            untargeted,
            where,
            truth,
            law.untargeted_estimate(fitted.result),
            float(estimate.std_error),
            columns,
        ),
    ]


def _replication(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    family, _, replicate, n, requested, _, fold_seed, _ = key[0]
    where = (replicate, n, requested)
    if family not in FAMILY_LEARNERS:
        raise ValueError(f"unknown property family {family!r}")
    fitted = law.measured(frame, law.outcome_learner(FAMILY_LEARNERS[family]), fold_seed)
    if family == NECESSITY_FAMILY:
        return necessity_rows(fitted, where)
    return law_rows(fitted, family, where)


def _payloads(record: StudyRecord, cap: int | None) -> list[_Key]:
    resolved = suffixes(record)
    out = []
    for draw in PROPERTY_DRAWS:
        suffix = resolved[draw.family, draw.label]
        replicates = draw.replicates if cap is None else min(cap, draw.replicates)
        for replicate in range(replicates):
            sample_seed, fold_seed = (
                stream_seed(record, stream, draw.family, draw.label, replicate, *suffix)
                for stream in ("property_sample", "fold_partition")
            )
            out.append(
                (
                    (
                        draw.family,
                        draw.label,
                        replicate,
                        CALIBRATION_N,
                        replicates,
                        sample_seed,
                        fold_seed,
                        draw.law,
                    ),
                )
            )
    return out


FAILURE_BOUND = 0.01
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def generate_property_rows(
    *, n_jobs: int = STUDY_JOBS, record: StudyRecord = STUDY, cap: int | None = None
) -> pd.DataFrame:
    """Fit every property replication and return one row per cell and replication.

    ``record`` and ``cap`` exist for a harness check on the throwaway record of rule L9 alone.
    The declared run passes neither.
    """
    outcomes = fit_properties(
        _payloads(record, cap),
        _replication,
        cells_of=_cells_of,
        cell_of=payload_cell,
        replicate_of=payload_replicate,
        n_jobs=n_jobs,
        draw=_draw,
        derived=DERIVED_CELLS,
    )
    rows = pd.DataFrame([row for records in outcomes for row in records])
    for column in ROW_COLUMNS:
        if column not in rows:
            rows[column] = np.nan
    controls = calibration_controls(
        rows,
        record,
        labels=CALIBRATION_LABELS,
        efficiency_bounds=EFFICIENCY_BOUNDS,
        calibration_n=CALIBRATION_N,
        shrunken_se_factor=SHRUNKEN_SE_FACTOR,
        critical=law.CRITICAL,
        positive_suffix=POSITIVE,
    )
    for column in INTEGER_COLUMNS:
        controls[column] = controls[column].astype(int)
    rows = pd.concat([rows, controls], ignore_index=True)
    for family, declared in PROPERTY_CELLS.items():
        cells = set(rows.loc[rows["property"] == family, "cell"])
        if cells != set(declared):
            raise RuntimeError(f"{family} cells {sorted(cells ^ set(declared))} differ")
    return rows.loc[:, [*REPLICATE_COLUMNS, *ROW_COLUMNS]]


#: The ``targeting_necessity`` family's declaration, read by its verdict and its rule.
TARGETING_NECESSITY = property_verdicts.Necessity(
    family=NECESSITY_FAMILY,
    labels=(NECESSITY_LABEL,),
    arms=("targeted", "untargeted"),
    column="targeting_displacement",
    threshold=TARGETING_DISPLACEMENT,
    truth_varies=STUDY.truth_varies_by_replicate,
)


def coverage_verdicts(summary: pd.DataFrame, *, floor: float) -> None:
    """The ``conservative_coverage`` verdicts: a one-sided coverage claim, and its control.

    A positive passes when its bias is established inside the margin and the lower bound of its
    exact coverage interval is at or above ``floor``; its SE ratio is published and read by no
    verdict, because the claim admits a conservative interval. A control passes when its coverage
    interval lies wholly below ``floor``.
    """
    family = summary["property"] == COVERAGE_FAMILY
    positive = family & (summary["role"] == "positive")
    control = family & (summary["role"] == "control")
    summary.loc[positive, "passed"] = summary.loc[positive, "bias_equivalent"].astype(bool) & (
        summary.loc[positive, "coverage_ci_lower"] >= floor
    )
    summary.loc[control, "passed"] = summary.loc[control, "coverage_ci_upper"] < floor


def report_margin(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """Publish the ``rule_margin_diagnostic`` cells: ``passed`` with no margin read.

    The summary already holds each cell's bias, coverage and SE ratio, which equal its
    positive's. This adds the mean of each margin column over the cell's draws and the share of
    draws whose realized rule assigns some arm to no row (``min_arm_share`` of 0). ``passed`` and
    ``property_passed`` read ``True``, meaning the numbers were computed and are published; the
    published role says that no margin was read.
    """
    family = summary["property"] == MARGIN_FAMILY
    if not family.any():
        return
    roles = set(summary.loc[family, "role"].astype(str))
    if roles != {DIAGNOSTIC_ROLE}:
        raise ValueError(f"{MARGIN_FAMILY} publishes roles {sorted(roles)}, not diagnostic")
    groups = cell_groups(rows, MARGIN_FAMILY)
    for cell in summary.loc[family, "cell"]:
        group = groups[cell]
        mask = family & (summary["cell"] == cell)
        for column in MARGIN_COLUMNS:
            summary.loc[mask, f"mean_{column}"] = float(group[column].astype(float).mean())
        summary.loc[mask, "lost_arm_share"] = float(
            (group["min_arm_share"].astype(float) == 0.0).mean()
        )
    summary.loc[family, "passed"] = True
    summary.loc[family, "property_passed"] = True


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The verdict of each cell, and the report of the diagnostic family.

    The study has one size, so it publishes no rate rows (``rate_labels=()``).
    """
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=("targeting_displacement", *MARGIN_SUMMARY),
        rate_labels=(),
    )
    calibration_verdicts(summary, margins=STUDY.margins, positive_suffix=POSITIVE)
    coverage_verdicts(summary, floor=STUDY.margins.calibration_coverage[0])
    necessity_verdicts(summary, rows, TARGETING_NECESSITY)
    report_margin(summary, rows)
    return finish(summary, rates)


def _coverage_floor(ctx: pf.Context) -> float:
    return ctx.margins.calibration_coverage[0]


#: The ``conservative_coverage`` rule of :func:`coverage_verdicts`.
COVERAGE_RULE = pf.FamilyRule(
    families=frozenset({COVERAGE_FAMILY}),
    shared="coverage_verdicts",
    clauses=(
        pf.bias_clause("bias equivalent", kind="equivalent", applies=pf.roles("positive")),
        pf.proportion_clause(
            "coverage at or above the calibration floor",
            statistic="S2",
            judge=pf.floor_judge(_coverage_floor),
            applies=pf.roles("positive"),
        ),
        pf.proportion_clause(
            "shrunken coverage below the calibration floor",
            statistic="S2",
            judge=pf.below_judge(_coverage_floor),
            applies=pf.roles("control"),
        ),
    ),
)

#: The declared rules of the failure accounting: the shared calibration rule, the coverage rule
#: and the necessity rule. The diagnostic family writes its verdict columns without reading a
#: margin, so it has none.
PROPERTY_RULES = (
    property_verdicts.calibration_rule(efficiency_band=None),
    COVERAGE_RULE,
    *property_verdicts.necessity_rules(TARGETING_NECESSITY),
)
