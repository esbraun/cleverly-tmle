"""Repeated-sampling properties of the stacked report of the learned-rule value.

Two families, in the order the declaration reads them.

``interval_calibration``
    The declared stacked fit of each setting at nominal n = 2,000 (``<setting>__declared``),
    with the two derived controls of
    :func:`~tests.studies.evidence.property_verdicts.calibration_controls`: the SE shrunken by
    :data:`SHRUNKEN_SE_FACTOR` and noise of one efficiency-bound unit added to each estimate.
    Each clustered draw also writes ``clustered__row_variance_control``, the same estimate with
    the row-level centred SE of the same curve, which ignores the clusters and must under-cover.
``fold_evaluated_reporting``
    The fold-evaluated fit (``cv_evaluation=True``) of every clustered calibration draw, with
    the same fold seed, so the same folds and the same fold rules (checked on every draw). Its
    truth weights the same fold values by ``1/V``. The cell carries the role ``diagnostic``: no
    margin is read and no verdict forms. Each row carries the paired stacked estimate, SE and
    truth, and the summary publishes the bias of each report on the same draws, in each report's
    own empirical SD, and the paired difference of the two errors with its Monte Carlo SE. That
    is the ratio-bias difference that motivates the stacked report at outcome-dependent cluster
    sizes.

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
from tests.studies import _learned_rule_law as base
from tests.studies import _learned_rule_stacked_law as law
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
)
from tests.studies.evidence.registry import StudyRecord
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.seeds import stream_seed
from tests.studies.learned_rule_stacked import (
    EFFICIENCY_BOUNDS,
    POSITIVE,
    PRIMARY_N,
    PRIMARY_REPLICATES,
    PROPERTY_CELLS,
    REPORTING_CELL,
    REPORTING_FAMILY,
    ROW_VARIANCE,
    STUDY,
)

CALIBRATION_LABELS = law.SCENARIOS
CALIBRATION_N = PRIMARY_N
#: The per-arm rule of :func:`~tests.studies.mtp_study_tools.per_arm_replicates` over the three
#: positives, on the 4,000-draw pilot read at the declared size (``test_learned_rule_stacked_
#: design``).
CALIBRATION_REPLICATES = 8_800
#: The plan's shrunken-SE control: the declared SE times 0.8.
SHRUNKEN_SE_FACTOR = 0.80

PROPERTY_DRAWS: tuple[base.PropertyDraw, ...] = tuple(
    base.PropertyDraw("interval_calibration", f"{label}__{POSITIVE}", CALIBRATION_REPLICATES, label)
    for label in CALIBRATION_LABELS
)


def suffixes(record: StudyRecord) -> dict[tuple[str, str], tuple[Any, ...]]:
    """The rule-L3 suffix of every property draw of ``record``."""
    return base.seed_labels(record, PRIMARY_REPLICATES, PROPERTY_DRAWS)


#: The calibration positive arms to the controls derived from their rows.
DERIVED_CELLS = calibration_derived(CALIBRATION_LABELS, positive_suffix=POSITIVE)

#: The cells a clustered calibration draw writes beside its positive arm.
ROW_VARIANCE_CELL = f"interval_calibration/clustered__{ROW_VARIANCE}"
REPORTING = f"{REPORTING_FAMILY}/{REPORTING_CELL}"

#: The paired stacked columns of a reporting row. Every other row leaves them empty.
PAIRED_COLUMNS = ("stacked_estimate", "stacked_std_error", "stacked_truth")

#: The columns every property row carries beside the shared schema.
ROW_COLUMNS = (*law.HARNESS_COLUMNS, *PAIRED_COLUMNS)

#: The descriptive summary columns of the reporting cell. No verdict reads them.
REPORTING_SUMMARY = (
    "stacked_bias",
    "stacked_standardized_bias",
    "paired_error_difference",
    "paired_error_difference_se",
)

_Key = tuple[tuple[str, str, int, int, int, int, int, str]]


def _draw(key: _Key) -> pd.DataFrame:
    _, _, _, n, _, sample_seed, _, scenario = key[0]
    return law.draw(scenario, n, sample_seed)


def _cells_of(key: _Key) -> list[str]:
    family, label = key[0][:2]
    name = f"{family}/{label}"
    written = [name, ROW_VARIANCE_CELL, REPORTING] if label == f"clustered__{POSITIVE}" else [name]
    return [*written, *DERIVED_CELLS.get(name, ())]


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


def reporting_row(
    stacked: tuple[Any, law.Truth, int],
    frame: pd.DataFrame,
    fold_seed: int,
    where: tuple[int, int, int],
) -> dict[str, Any]:
    """The fold-evaluated fit of a clustered draw, with the paired stacked columns.

    Raises
    ------
    RuntimeError
        When the fold-evaluated fit drew other folds, or used another rule, than the stacked
        fit of the same draw and fold seed.
    """
    result, truth, warnings = stacked
    other, other_truth, other_warnings = law.measured(
        frame, "clustered", base.outcome_learner(), fold_seed, cv_evaluation=True
    )
    if not law.same_partition(result, other):
        raise RuntimeError("the fold-evaluated fit drew other folds or rules than the stacked one")
    estimate = result.estimates[law.ESTIMAND]
    columns = law.diagnostic_columns(other, other_truth, warnings + other_warnings)
    # The identity terms belong to the stacked curve; the fold-evaluated fit has none.
    columns.update(cross_term=np.nan, between_term=np.nan)
    columns.update(
        stacked_estimate=float(estimate.psi),
        stacked_std_error=float(estimate.std_error),
        stacked_truth=truth.truth,
    )
    return _row(
        REPORTING_FAMILY,
        REPORTING_CELL,
        DIAGNOSTIC_ROLE,
        where,
        other_truth.truth,
        other.estimates[law.ESTIMAND],
        columns,
    )


def _replication(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    family, label, replicate, n, requested, _, fold_seed, scenario = key[0]
    if family != "interval_calibration":
        raise ValueError(f"unknown property family {family!r}")
    where = (replicate, n, requested)
    fitted = law.measured(frame, scenario, base.outcome_learner(), fold_seed)
    result, truth, warnings = fitted
    estimate = result.estimates[law.ESTIMAND]
    columns = law.diagnostic_columns(result, truth, warnings)
    rows = [_row(family, label, "positive", where, truth.truth, estimate, columns)]
    if scenario == "clustered":
        control = control_row(
            property_name=family,
            cell=f"clustered__{ROW_VARIANCE}",
            replicate=replicate,
            n=n,
            requested=requested,
            truth=truth.truth,
            estimate=float(estimate.psi),
            standard_error=law.row_variance_se(result),
            critical=base.CRITICAL,
        )
        control.update(columns)
        rows.append(control)
        rows.append(reporting_row(fitted, frame, fold_seed, where))
    return rows


def _payloads(record: StudyRecord, cap: int | None) -> list[_Key]:
    resolved = suffixes(record)
    out = []
    for draw in PROPERTY_DRAWS:
        suffix = resolved[draw.family, draw.label]
        replicates = draw.replicates if cap is None else min(cap, draw.replicates)
        for replicate in range(replicates):
            sample_seed = stream_seed(
                record, "property_sample", draw.family, draw.label, replicate, *suffix
            )
            fold_seed = stream_seed(
                record, "fold_partition", draw.family, draw.label, replicate, *suffix
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
    """Fit every property replication and return one row per cell and replication."""
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
    for column in PAIRED_COLUMNS:
        if column not in rows:
            rows[column] = np.nan
    controls = calibration_controls(
        rows,
        record,
        labels=CALIBRATION_LABELS,
        efficiency_bounds=EFFICIENCY_BOUNDS,
        calibration_n=CALIBRATION_N,
        shrunken_se_factor=SHRUNKEN_SE_FACTOR,
        critical=base.CRITICAL,
        positive_suffix=POSITIVE,
    )
    for column in ("rule_rows_checked", "solver_warnings", "rows"):
        controls[column] = controls[column].astype(int)
    rows = pd.concat([rows, controls], ignore_index=True)
    for family, declared in PROPERTY_CELLS.items():
        cells = set(rows.loc[rows["property"] == family, "cell"])
        if cells != set(declared):
            raise RuntimeError(f"{family} cells {sorted(cells ^ set(declared))} differ")
    return rows.loc[:, [*REPLICATE_COLUMNS, *ROW_COLUMNS]]


def report_fold_evaluated(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """Report the fold-evaluated reporting cell beside the stacked fit of the same draws.

    The summary already holds the fold-evaluated bias, coverage and SE ratio. This adds the
    stacked bias on the same draws, in the stacked report's own empirical SD, and the mean of
    the paired error difference ``(fold-evaluated - truth) - (stacked - truth)`` with its Monte
    Carlo SE. It writes ``passed`` and ``property_passed`` ``True``, meaning the numbers were
    computed and are published; the published role says that no margin was read.
    """
    family = summary["property"] == REPORTING_FAMILY
    roles = set(summary.loc[family, "role"].astype(str))
    if family.any() and roles != {DIAGNOSTIC_ROLE}:
        raise ValueError(f"{REPORTING_FAMILY} publishes roles {sorted(roles)}, not diagnostic")
    groups = cell_groups(rows, REPORTING_FAMILY)
    for cell in sorted(summary.loc[family, "cell"]):
        group = groups[cell]
        stacked = (group["stacked_estimate"] - group["stacked_truth"]).to_numpy(float)
        evaluated = (group["estimate"] - group["truth"]).to_numpy(float)
        difference = evaluated - stacked
        mask = family & (summary["cell"] == cell)
        summary.loc[mask, "stacked_bias"] = float(np.mean(stacked))
        summary.loc[mask, "stacked_standardized_bias"] = float(
            np.mean(stacked) / np.std(stacked, ddof=1)
        )
        summary.loc[mask, "paired_error_difference"] = float(np.mean(difference))
        summary.loc[mask, "paired_error_difference_se"] = float(
            np.std(difference, ddof=1) / np.sqrt(difference.size)
        )
        summary.loc[mask, "passed"] = True
        summary.loc[mask, "property_passed"] = True


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The verdict of each calibration cell, and the report of the reporting cell.

    The study has one size, so it publishes no rate rows (``rate_labels=()``).
    """
    summary, rates = apply_shared_verdicts(
        rows, STUDY, extra_columns=REPORTING_SUMMARY, rate_labels=()
    )
    calibration_verdicts(summary, margins=STUDY.margins, positive_suffix=POSITIVE)
    report_fold_evaluated(summary, rows)
    return finish(summary, rates)


#: The declared rules of the failure accounting: the shared calibration rule, named here
#: because the module declares its own rules. The reporting cell writes its verdict columns
#: without reading a margin, so it has none.
PROPERTY_RULES = (property_verdicts.calibration_rule(efficiency_band=None),)
