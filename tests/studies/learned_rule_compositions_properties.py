"""Repeated-sampling properties of the learned-rule value with fold targeting, weights and clusters.

Two families, in the order the declaration reads them.

``interval_calibration``
    The declared fit of each primary scenario at n = 2,000 (``<scenario>__declared``), with the
    two derived controls of :func:`~tests.studies.evidence.property_verdicts.calibration_controls`:
    the SE shrunken by 0.70, and noise of one efficiency-bound unit added to each estimate. The
    ``clustered`` draws also carry ``clustered__row_variance_control``, the same estimate with the
    row-level SE that ignores the clusters, which must under-cover. ``clustered_few__declared``
    fits 30 clusters of 34 rows (n = 1,020), where the fit takes the few-cluster t reference.
``targeting_necessity``
    Fold targeting with the outcome learner that drops ``W1``. The positive arm is the fit; the
    control is the same fit without its fluctuations, the fold average of the initial plug-ins at
    the fit's own rule.

Every statistic reads the error ``estimate - truth`` of each row, because the record declares
``truth_varies_by_replicate``. A property draw's sample seed is
``stream_seed(record, "property_sample", family, label, replicate, *suffix)`` and its fold seed
the same tuple with ``"fold_partition"`` first; rule L3 of the RM30 studies sets each suffix.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import _learned_rule_compositions_law as law
from tests.studies import _learned_rule_law as base
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    control_row,
    payload_cell,
    payload_replicate,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
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
from tests.studies.learned_rule_compositions import (
    EFFICIENCY_BOUNDS,
    PRIMARY_N,
    PRIMARY_REPLICATES,
    PROPERTY_CELLS,
    STUDY,
)

CALIBRATION_LABELS = ("fold_targeted", "weighted", "clustered")
CALIBRATION_REPLICATES = 8_400
FEW_CLUSTER_N = 1_020
FEW_CLUSTER_REPLICATES = 8_400
NECESSITY_REPLICATES = 2_655
SHRUNKEN_SE_FACTOR = 0.70
TARGETING_DISPLACEMENT = 0.5
POSITIVE = "declared"

PROPERTY_DRAWS: tuple[base.PropertyDraw, ...] = (
    *(
        base.PropertyDraw(
            "interval_calibration", f"{label}__{POSITIVE}", CALIBRATION_REPLICATES, label
        )
        for label in CALIBRATION_LABELS
    ),
    base.PropertyDraw(
        "interval_calibration",
        f"clustered_few__{POSITIVE}",
        FEW_CLUSTER_REPLICATES,
        "clustered_few",
    ),
    base.PropertyDraw(
        "targeting_necessity", "fold_targeted", NECESSITY_REPLICATES, "fold_targeted"
    ),
)


def suffixes(record: StudyRecord) -> dict[tuple[str, str], tuple[Any, ...]]:
    """The rule-L3 suffix of every property draw of ``record``."""
    return base.seed_labels(record, PRIMARY_REPLICATES, PROPERTY_DRAWS)


#: The calibration positive arms to the controls derived from their rows.
DERIVED_CELLS = calibration_derived(CALIBRATION_LABELS, positive_suffix=POSITIVE)

#: The control a ``clustered`` calibration draw writes beside its positive arm.
ROW_VARIANCE_CELL = "interval_calibration/clustered__row_variance_control"

_Key = tuple[tuple[str, str, int, int, int, int, int, str]]


def _row(
    family: str,
    cell: str,
    role: str,
    replicate: int,
    n: int,
    requested: int,
    truth: law.Truth,
    estimate: Any,
    solver_warnings: int,
) -> dict[str, Any]:
    row = replicate_row(
        property_name=family,
        cell=cell,
        role=role,
        replicate=replicate,
        n=n,
        requested=requested,
        truth=truth.truth,
        estimate=estimate,
        alpha=STUDY.margins.alpha,
    )
    row.update(
        {
            "oracle_se": truth.oracle_se,
            "rule_rows_checked": truth.rows_checked,
            "solver_warnings": solver_warnings,
        }
    )
    return row


def _draw(key: _Key) -> pd.DataFrame:
    _, _, _, n, _, sample_seed, _, scenario = key[0]
    return law.draw(scenario, n, sample_seed)


def _cells_of(key: _Key) -> list[str]:
    family, label = key[0][:2]
    if family == "targeting_necessity":
        return [f"{family}/{label}__targeted", f"{family}/{label}__untargeted"]
    name = f"{family}/{label}"
    written = [name, ROW_VARIANCE_CELL] if label == f"clustered__{POSITIVE}" else [name]
    return [*written, *DERIVED_CELLS.get(name, ())]


def _replication(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    family, label, replicate, n, requested, _, fold_seed, scenario = key[0]
    if family == "interval_calibration":
        result, truth, warnings = law.measured(frame, scenario, base.outcome_learner(), fold_seed)
        estimate = result.estimates[law.ESTIMAND]
        rows = [_row(family, label, "positive", replicate, n, requested, truth, estimate, warnings)]
        if scenario == "clustered":
            control = control_row(
                property_name=family,
                cell="clustered__row_variance_control",
                replicate=replicate,
                n=n,
                requested=requested,
                truth=truth.truth,
                estimate=float(estimate.psi),
                standard_error=law.row_variance_se(result),
                critical=base.CRITICAL,
            )
            control.update({name: rows[0][name] for name in base.HARNESS_COLUMNS})
            rows.append(control)
        return rows
    if family == "targeting_necessity":
        result, truth, warnings = law.measured(frame, scenario, base.drop_w1_learner(), fold_seed)
        estimate = result.estimates[law.ESTIMAND]
        targeted = _row(
            family,
            f"{label}__targeted",
            "positive",
            replicate,
            n,
            requested,
            truth,
            estimate,
            warnings,
        )
        untargeted = control_row(
            property_name=family,
            cell=f"{label}__untargeted",
            replicate=replicate,
            n=n,
            requested=requested,
            truth=truth.truth,
            estimate=base.untargeted_estimate(result),
            standard_error=float(estimate.std_error),
            critical=base.CRITICAL,
        )
        untargeted.update({name: targeted[name] for name in base.HARNESS_COLUMNS})
        return [targeted, untargeted]
    raise ValueError(f"unknown property family {family!r}")


def _payloads(record: StudyRecord, cap: int | None) -> list[_Key]:
    resolved = suffixes(record)
    out = []
    for draw in PROPERTY_DRAWS:
        suffix = resolved[draw.family, draw.label]
        n = FEW_CLUSTER_N if draw.law == "clustered_few" else PRIMARY_N
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
                        n,
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
    controls = calibration_controls(
        rows,
        record,
        labels=CALIBRATION_LABELS,
        efficiency_bounds=EFFICIENCY_BOUNDS,
        calibration_n=PRIMARY_N,
        shrunken_se_factor=SHRUNKEN_SE_FACTOR,
        critical=base.CRITICAL,
        positive_suffix=POSITIVE,
    )
    rows = pd.concat([rows, controls], ignore_index=True)
    for family, declared in PROPERTY_CELLS.items():
        cells = set(rows.loc[rows["property"] == family, "cell"])
        if cells != set(declared):
            raise RuntimeError(f"{family} cells {sorted(cells ^ set(declared))} differ")
    return rows.loc[:, [*REPLICATE_COLUMNS, *base.HARNESS_COLUMNS]]


TARGETING_NECESSITY = property_verdicts.Necessity(
    family="targeting_necessity",
    labels=("fold_targeted",),
    arms=("targeted", "untargeted"),
    column="targeting_displacement",
    threshold=TARGETING_DISPLACEMENT,
    truth_varies=STUDY.truth_varies_by_replicate,
)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The verdict of each cell, every statistic on the error of each row."""
    summary, rates = apply_shared_verdicts(rows, STUDY, extra_columns=("targeting_displacement",))
    calibration_verdicts(summary, margins=STUDY.margins, positive_suffix=POSITIVE)
    necessity_verdicts(summary, rows, TARGETING_NECESSITY)
    return finish(summary, rates)


PROPERTY_RULES = (*property_verdicts.necessity_rules(TARGETING_NECESSITY),)
