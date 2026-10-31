"""Repeated-sampling properties of the learned-rule value with a rule learner.

One family, ``interval_calibration``, for each loss on the RM30 ``misspecified_limit`` law at
n = 4,000 (:data:`CALIBRATION_N`), twice the primary size. At n = 2,000 the error of a rule
learner's fit spreads 3% to 5% more than its reported SE, and than its oracle SE (pilot SE ratios
0.954 and 0.971): each fold's rule is learned from a noisy pseudo-outcome, and its own sampling
variability, which the fixed-rule variance omits, has not yet vanished. That ratio leaves the
band's lower end within the pilot's noise, so the band is read where the rule has settled (pilot
ratio 1.009 at n = 4,000), and the primary phase at n = 2,000 publishes the smaller size's ratio
under the framework's SE sanity band.

``<loss>__declared``
    The declared fit. Its SE-ratio interval must lie inside 0.93 to 1.07 and its coverage
    interval inside 0.92 to 0.98, both read on the error ``estimate - truth``, where the truth
    is the fold-average value of the rules the fit learned.
``<loss>__shrunken_se_control`` and ``<loss>__noise_control``
    The two derived controls of
    :func:`~tests.studies.evidence.property_verdicts.calibration_controls`: the SE shrunken by
    0.70, and noise of one efficiency-bound unit added to each estimate. Each must fall below
    the SE-ratio band. They check the instrument, not the rule learner.
``<loss>__validation_rule_control``
    The fold-locality control: the deliberate mutation
    :class:`~tests.studies._learned_rule_law.ValidationRowRule` on the same sample and fold
    seed, which fits fold ``v``'s rule learner on fold ``v``'s validation rows to the
    pseudo-outcome of the fit's outer cross-fitted nuisances. Its truth is the value of the
    rules it learned (rule L4 on the validation rows). The rows that evaluate each rule also
    chose it, so its estimate is optimistic, and its coverage interval must lie wholly below
    the band floor 0.92 (``COVERAGE_CONTROLS``). A validation fold whose weighted rows carry one
    sign gives the constant rule of that sign, the package's own convention.

Each positive row and each control row also carries the descriptive ``plug_in_estimate`` and
``plug_in_truth``: the plug-in rule's fit of the same sample with the same fold seed, which
draws the same outer folds, and its own harness truth. The summary publishes, on each positive
cell, ``value_gain_mean`` and ``value_gain_sd``, the mean and SD of ``truth - plug_in_truth``,
and ``value_gain_positive_share``, the share of replications where it is positive. No verdict
reads them: the population gain is about 0.0054, near half an SE at n = 4,000, so the study
reports the gain on the oracle truths and claims no test of it.

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
from tests.studies import _learned_rule_learners_law as law
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
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
from tests.studies.learned_rule_learners import (
    EFFICIENCY_BOUNDS,
    PRIMARY_REPLICATES,
    PROPERTY_CELLS,
    STUDY,
)

CALIBRATION_LABELS = law.LOSSES
CALIBRATION_N = 4_000
CALIBRATION_REPLICATES = 7_200
SHRUNKEN_SE_FACTOR = 0.70
POSITIVE = "declared"
VALIDATION_RULE = "validation_rule_control"

#: The descriptive summary columns of the paired comparison with the plug-in rule, on each
#: positive cell. No verdict reads them.
VALUE_GAIN_SUMMARY = ("value_gain_mean", "value_gain_sd", "value_gain_positive_share")

PROPERTY_DRAWS: tuple[base.PropertyDraw, ...] = tuple(
    base.PropertyDraw("interval_calibration", f"{label}__{POSITIVE}", CALIBRATION_REPLICATES, label)
    for label in CALIBRATION_LABELS
)


def suffixes(record: StudyRecord) -> dict[tuple[str, str], tuple[Any, ...]]:
    """The rule-L3 suffix of every property draw of ``record``."""
    return base.seed_labels(record, PRIMARY_REPLICATES, PROPERTY_DRAWS)


#: The calibration positive arms to the controls derived from their rows.
DERIVED_CELLS = calibration_derived(CALIBRATION_LABELS, positive_suffix=POSITIVE)

#: The columns every property row carries beside the shared schema.
ROW_COLUMNS = (*base.HARNESS_COLUMNS, *law.VALUE_GAIN_COLUMNS)

_Key = tuple[tuple[str, str, int, int, int, int, int, str]]


def _draw(key: _Key) -> pd.DataFrame:
    _, _, _, n, _, sample_seed, _, _ = key[0]
    return base.draw(law.LAW, n, sample_seed)


def _cells_of(key: _Key) -> list[str]:
    family, label = key[0][:2]
    name = f"{family}/{label}"
    stem = label.removesuffix(f"__{POSITIVE}")
    return [name, f"{family}/{stem}__{VALIDATION_RULE}", *DERIVED_CELLS.get(name, ())]


def _row(
    family: str,
    cell: str,
    role: str,
    key: tuple[int, int, int],
    truth: base.Truth,
    estimate: Any,
    warnings: int,
    gain: dict[str, float],
) -> dict[str, Any]:
    replicate, n, requested = key
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
            "solver_warnings": warnings,
            **gain,
        }
    )
    return row


def _replication(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    family, label, replicate, n, requested, _, fold_seed, scenario = key[0]
    if family != "interval_calibration":
        raise ValueError(f"unknown property family {family!r}")
    fitted = law.measured(frame, scenario, fold_seed, control=True)
    assert fitted.control is not None and fitted.control_truth is not None
    gain = law.value_gain_columns(fitted)
    where = (replicate, n, requested)
    return [
        _row(
            family,
            label,
            "positive",
            where,
            fitted.truth,
            fitted.result.estimates[law.ESTIMAND],
            fitted.warnings,
            gain,
        ),
        _row(
            family,
            f"{scenario}__{VALIDATION_RULE}",
            "control",
            where,
            fitted.control_truth,
            fitted.control.estimates[law.ESTIMAND],
            fitted.control_warnings,
            gain,
        ),
    ]


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
    rows = pd.concat([rows, controls], ignore_index=True)
    for family, declared in PROPERTY_CELLS.items():
        cells = set(rows.loc[rows["property"] == family, "cell"])
        if cells != set(declared):
            raise RuntimeError(f"{family} cells {sorted(cells ^ set(declared))} differ")
    return rows.loc[:, [*REPLICATE_COLUMNS, *ROW_COLUMNS]]


def value_gain(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """Fill the descriptive :data:`VALUE_GAIN_SUMMARY` of each positive cell; no verdict reads them.

    Each reads the paired difference ``truth - plug_in_truth`` of the declared fit's rules and
    the plug-in fit's rules on the same sample and outer folds.
    """
    for label in CALIBRATION_LABELS:
        cell = f"{label}__{POSITIVE}"
        group = rows.loc[(rows["property"] == "interval_calibration") & (rows["cell"] == cell)]
        gain = group["truth"].to_numpy(float) - group["plug_in_truth"].to_numpy(float)
        mask = (summary["property"] == "interval_calibration") & (summary["cell"] == cell)
        summary.loc[mask, "value_gain_mean"] = float(np.mean(gain))
        summary.loc[mask, "value_gain_sd"] = float(np.std(gain, ddof=1))
        summary.loc[mask, "value_gain_positive_share"] = float(np.mean(gain > 0.0))


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The verdict of each cell, every statistic on the error of each row.

    The study has one size, so it publishes no rate rows (``rate_labels=()``).
    """
    summary, rates = apply_shared_verdicts(
        rows, STUDY, extra_columns=VALUE_GAIN_SUMMARY, rate_labels=()
    )
    calibration_verdicts(summary, margins=STUDY.margins, positive_suffix=POSITIVE)
    value_gain(summary, rows)
    return finish(summary, rates)
