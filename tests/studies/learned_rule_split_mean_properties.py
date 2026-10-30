"""Repeated-sampling properties of the learned-rule value as the mean over five splits.

One family, ``interval_calibration``, on each RM30 law at n = 2,000.

``<law>__declared``
    The declared fit with ``repeats=5``. Its SE-ratio interval must lie inside 0.93 to 1.07 and
    its coverage interval inside 0.92 to 0.98, both read on the error ``estimate - truth``,
    where the truth is the mean of the five split targets.
``<law>__shrunken_se_control`` and ``<law>__noise_control``
    The two derived controls of
    :func:`~tests.studies.evidence.property_verdicts.calibration_controls`: the SE shrunken by
    0.70, and noise of one efficiency-bound unit added to each estimate. Each must fall below
    the SE-ratio band. They check the instrument, not the mean over splits.
``<law>__independent_splits_control``
    The declared fit's estimate with the deliberately invalid SE that treats the five splits
    as independent samples, the mean single-split SE over ``sqrt(5)``. The splits reuse every
    row, so it is too small by about ``sqrt(5)`` and must fall below the SE-ratio band
    (``SE_CONTROL_SIDES``). It fails by construction: its SE is about 0.45 of the reported SE
    on every draw, so it can reach 0.93 only where the positive has already failed. It is a
    sanity control that shows the size of the independent-splits mistake. The evidence that the
    reported variance handles the correlation between splits is the positive passing its band.

The median of the five split estimates is descriptive only. Each row carries it in
``median_estimate``, and the summary publishes ``median_aggregate_bias`` and
``median_aggregate_empirical_se`` on each positive cell, both on the error ``median - truth``.
No verdict reads them: the median estimates no defined parameter, so no band could be declared
for it. They are what the refused median would have reported against the mean target. Each
row also carries the descriptive ``first_split_estimate``, ``first_split_truth`` and
``split_estimate_sd``, and the summary publishes ``first_split_error_sd`` on each positive cell,
the SD of the first split's own error; no verdict reads those either.

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
from tests.studies import _learned_rule_split_mean_law as law
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
)
from tests.studies.evidence.registry import StudyRecord
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.seeds import stream_seed
from tests.studies.learned_rule_split_mean import (
    EFFICIENCY_BOUNDS,
    PRIMARY_N,
    PRIMARY_REPLICATES,
    PROPERTY_CELLS,
    STUDY,
)

CALIBRATION_LABELS = law.LAWS
CALIBRATION_REPLICATES = 6_400
SHRUNKEN_SE_FACTOR = 0.70
POSITIVE = "declared"
INDEPENDENT_SPLITS = "independent_splits_control"

#: The descriptive summary columns of the median aggregate. No verdict reads them.
MEDIAN_COLUMNS = ("median_aggregate_bias", "median_aggregate_empirical_se")

#: The descriptive summary column of the first split alone, the SD of ``first_split_estimate -
#: first_split_truth``. No verdict reads it.
FIRST_SPLIT_SUMMARY = "first_split_error_sd"

#: Every descriptive summary column of a positive cell.
DESCRIPTIVE_COLUMNS = (*MEDIAN_COLUMNS, FIRST_SPLIT_SUMMARY)

PROPERTY_DRAWS: tuple[base.PropertyDraw, ...] = tuple(
    base.PropertyDraw("interval_calibration", f"{label}__{POSITIVE}", CALIBRATION_REPLICATES, label)
    for label in CALIBRATION_LABELS
)


def suffixes(record: StudyRecord) -> dict[tuple[str, str], tuple[Any, ...]]:
    """The rule-L3 suffix of every property draw of ``record``."""
    return base.seed_labels(record, PRIMARY_REPLICATES, PROPERTY_DRAWS)


#: The calibration positive arms to the controls derived from their rows.
DERIVED_CELLS = calibration_derived(CALIBRATION_LABELS, positive_suffix=POSITIVE)

#: How many times the harness checks each row of a fit: once per split. The run form's
#: ``--property-harness-check`` reads it.
RULE_CHECKS_PER_ROW = law.SPLITS

#: The columns every property row carries beside the shared schema.
ROW_COLUMNS = (*base.HARNESS_COLUMNS, *law.SPLIT_COLUMNS)

_Key = tuple[tuple[str, str, int, int, int, int, int, str]]


def _draw(key: _Key) -> pd.DataFrame:
    _, _, _, n, _, sample_seed, _, scenario = key[0]
    return base.draw(scenario, n, sample_seed)


def _cells_of(key: _Key) -> list[str]:
    family, label = key[0][:2]
    name = f"{family}/{label}"
    stem = label.removesuffix(f"__{POSITIVE}")
    return [name, f"{family}/{stem}__{INDEPENDENT_SPLITS}", *DERIVED_CELLS.get(name, ())]


def _replication(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    family, label, replicate, n, requested, _, fold_seed, scenario = key[0]
    if family != "interval_calibration":
        raise ValueError(f"unknown property family {family!r}")
    result, truth, warnings = law.measured(frame, scenario, base.outcome_learner(), fold_seed)
    estimate = result.estimates[law.ESTIMAND]
    extra = {
        "oracle_se": truth.oracle_se,
        "rule_rows_checked": truth.rows_checked,
        "solver_warnings": warnings,
        **law.split_columns(result, truth),
    }
    positive = replicate_row(
        property_name=family,
        cell=label,
        role="positive",
        replicate=replicate,
        n=n,
        requested=requested,
        truth=truth.truth,
        estimate=estimate,
        alpha=STUDY.margins.alpha,
    )
    control = control_row(
        property_name=family,
        cell=f"{scenario}__{INDEPENDENT_SPLITS}",
        replicate=replicate,
        n=n,
        requested=requested,
        truth=truth.truth,
        estimate=float(estimate.psi),
        standard_error=law.independent_splits_se(result),
        critical=base.CRITICAL,
    )
    positive.update(extra)
    control.update(extra)
    return [positive, control]


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
                        PRIMARY_N,
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
    return rows.loc[:, [*REPLICATE_COLUMNS, *ROW_COLUMNS]]


def median_aggregate(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """Fill the descriptive :data:`DESCRIPTIVE_COLUMNS` of each positive cell; no verdict reads them.

    The two median columns read the error ``median_estimate - truth``: its mean, and its SD.
    ``first_split_error_sd`` is the SD of the first split's own error.
    """
    for label in CALIBRATION_LABELS:
        cell = f"{label}__{POSITIVE}"
        group = rows.loc[(rows["property"] == "interval_calibration") & (rows["cell"] == cell)]
        error = group["median_estimate"].to_numpy(float) - group["truth"].to_numpy(float)
        mask = (summary["property"] == "interval_calibration") & (summary["cell"] == cell)
        summary.loc[mask, "median_aggregate_bias"] = float(np.mean(error))
        summary.loc[mask, "median_aggregate_empirical_se"] = float(np.std(error, ddof=1))
        first = group["first_split_estimate"].to_numpy(float) - group["first_split_truth"].to_numpy(
            float
        )
        summary.loc[mask, FIRST_SPLIT_SUMMARY] = float(np.std(first, ddof=1))


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The verdict of each cell, every statistic on the error of each row.

    The study has one size, so it publishes no rate rows (``rate_labels=()``).
    """
    summary, rates = apply_shared_verdicts(
        rows, STUDY, extra_columns=DESCRIPTIVE_COLUMNS, rate_labels=()
    )
    calibration_verdicts(summary, margins=STUDY.margins, positive_suffix=POSITIVE)
    median_aggregate(summary, rows)
    return finish(summary, rates)
