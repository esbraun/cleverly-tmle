"""Size and power of the point-treatment refutation rules (X82 follow-up SP).

Two families. ``refuter_size`` holds one cell per rule and configuration: 200 trials, and the
cell passes when the rule rejects in at most 17. ``refuter_power`` holds one cell per defect:
trials 0 to 49 of the same configuration, and the cell passes when the rule detects the defect
in at least 38. A power cell reads the same fit as its size trial, so the two differ only in
the defect.

Each trial draws its own sample from ``stream_seed(STUDY, "property_sample", cell, trial)``.

Each trial also records the configuration of its fit. :func:`generate_property_artifacts`
stops the run if a cell's configuration varies, and writes the one configuration of each cell
to ``cells.csv``.

The verdicts are the plan's count rules. The failure accounting reads them through the exact
99% Clopper-Pearson interval of the shared S3 clause: :data:`SIZE_CEILING` lies strictly
between the upper endpoints at 17 and 18 rejections of 200, and :data:`POWER_FLOOR` strictly
between the lower endpoints at 37 and 38 detections of 50. With no lost trial the interval rule
and the count rule agree, and :func:`count_rule_agrees` proves it for every count.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies.evidence import property_failures as pf
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.inference import clopper_pearson
from tests.studies.evidence.properties import REPLICATE_COLUMNS
from tests.studies.evidence.property_verdicts import apply_shared_verdicts, finish
from tests.studies.evidence.replication import fit_properties, splat_draw, splat_drawn
from tests.studies.refuter_calibration_point import (
    CELLS,
    POWER_FAMILY,
    POWER_MIN_DETECTIONS,
    POWER_TRIALS,
    PROPERTY_CELLS,
    SIZE_FAMILY,
    SIZE_MAX_REJECTIONS,
    SIZE_TRIALS,
    STUDY,
    N,
    draw_law,
    trial_readings,
    trial_seed,
)

#: The committed file that records each cell's fit configuration.
CELLS_FILE = "cells.csv"
_CONFIGURATION = "configuration"


_LEVEL = STUDY.margins.confidence_level


def _midpoint(low: float, high: float) -> float:
    return 0.5 * (low + high)


#: Between the 99% Clopper-Pearson upper endpoints at 17 and 18 rejections of 200.
SIZE_CEILING = _midpoint(
    clopper_pearson(SIZE_MAX_REJECTIONS, SIZE_TRIALS, confidence_level=_LEVEL).high,
    clopper_pearson(SIZE_MAX_REJECTIONS + 1, SIZE_TRIALS, confidence_level=_LEVEL).high,
)
#: Between the 99% Clopper-Pearson lower endpoints at 37 and 38 detections of 50.
POWER_FLOOR = _midpoint(
    clopper_pearson(POWER_MIN_DETECTIONS - 1, POWER_TRIALS, confidence_level=_LEVEL).low,
    clopper_pearson(POWER_MIN_DETECTIONS, POWER_TRIALS, confidence_level=_LEVEL).low,
)


def count_rule_agrees() -> bool:
    """Whether the interval thresholds reproduce the count rules at every count."""
    size = all(
        (clopper_pearson(k, SIZE_TRIALS, confidence_level=_LEVEL).high <= SIZE_CEILING)
        == (k <= SIZE_MAX_REJECTIONS)
        for k in range(SIZE_TRIALS + 1)
    )
    power = all(
        (clopper_pearson(k, POWER_TRIALS, confidence_level=_LEVEL).low >= POWER_FLOOR)
        == (k >= POWER_MIN_DETECTIONS)
        for k in range(POWER_TRIALS + 1)
    )
    return size and power


#: The truth column of a row: the statistic's value under the rule's reference law.
_REFERENCE = {"placebo": 0.0, "subset": 1.0, "negative_control": 0.0}


def _reference(reading: str) -> float:
    for prefix, value in _REFERENCE.items():
        if reading.startswith(prefix):
            return value
    raise KeyError(reading)


def _row(
    family: str,
    cell: str,
    role: str,
    trial: int,
    requested: int,
    reading: Any,
    configuration: str,
) -> dict[str, Any]:
    reading_name = cell.split("__", 1)[1]
    reference = _reference(reading_name)
    # The placebo's constant-zero case has no statistic and does not reject; its row records
    # the reference value, since the framework requires a finite estimate.
    statistic = reading.statistic if math.isfinite(reading.statistic) else reference
    return {
        "property": family,
        "cell": cell,
        "role": role,
        "replicate": trial,
        "n": N,
        "requested_replicates": requested,
        "failed_replicates": 0,
        "truth": reference,
        "estimate": statistic,
        # Each statistic is already on its standardized scale (``t``, ``rho`` or ``z``), so
        # the row's scale is 1. No verdict reads it; the framework requires a positive value.
        "std_error": 1.0,
        "covered": int(not reading.rejected),
        "rejected": int(reading.rejected),
        _CONFIGURATION: configuration,
    }


def _draw(name: str, trial: int) -> Any:
    return draw_law(CELLS[name].family, N, trial_seed("property_sample", name, trial))


def _replicate(name: str, trial: int, drawn: Any) -> list[dict[str, Any]]:
    cell = CELLS[name]
    result = trial_readings(name, drawn, trial, power=trial < POWER_TRIALS)
    readings = result.readings
    configuration = json.dumps(result.configuration, sort_keys=True)
    rows = [
        _row(
            SIZE_FAMILY,
            f"{name}__{reading}",
            "positive",
            trial,
            SIZE_TRIALS,
            readings[reading],
            configuration,
        )
        for reading in cell.readings
    ]
    if trial < POWER_TRIALS:
        rows += [
            _row(
                POWER_FAMILY,
                f"{name}__{defect}",
                "control",
                trial,
                POWER_TRIALS,
                readings[defect],
                configuration,
            )
            for defect in cell.defects
        ]
    return rows


def cell_configurations(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per cell: the configuration every trial of the cell fitted.

    The smallest arm event count differs from one sample to the next, so the row records its
    range over the cell's trials (a calibration scope covers event counts at or above its
    floor). Every other axis must be the same on every trial.

    Parameters
    ----------
    rows : pandas.DataFrame
        The property rows with their configuration column.

    Returns
    -------
    pandas.DataFrame
        One row per cell of :data:`CELLS`.

    Raises
    ------
    RuntimeError
        If a cell's trials fitted more than one configuration.
    """
    names = rows["cell"].str.split("__", n=1).str[0]
    records = []
    for name in CELLS:
        fits = [json.loads(text) for text in set(rows.loc[names == name, _CONFIGURATION])]
        events = [fit.pop("min_event_count") for fit in fits]
        axes = {json.dumps(fit, sort_keys=True) for fit in fits}
        if len(axes) != 1:
            raise RuntimeError(f"cell {name} fitted {len(axes)} configurations, expected one")
        counted = [count for count in events if count is not None]
        records.append(
            {
                "cell": name,
                **json.loads(axes.pop()),
                "min_event_count_min": min(counted) if counted else None,
                "min_event_count_max": max(counted) if counted else None,
            }
        )
    table = pd.DataFrame(records)
    for column in ("min_event_count_min", "min_event_count_max"):
        table[column] = table[column].astype("Int64")
    return table


def _cells_of(key: tuple[str, int]) -> list[str]:
    name, trial = key
    cell = CELLS[name]
    cells = [f"{SIZE_FAMILY}/{name}__{reading}" for reading in cell.readings]
    if trial < POWER_TRIALS:
        cells += [f"{POWER_FAMILY}/{name}__{defect}" for defect in cell.defects]
    return cells


def _cell_of(key: tuple[str, int]) -> str:
    return f"{SIZE_FAMILY}/{key[0]}__placebo"


def _replicate_of(key: tuple[str, int]) -> int:
    return key[1]


#: The share of a cell's requested trials that the accounting lets a run lose.
FAILURE_BOUND = 0.01

#: The draw-level statistic the accounting reports for each lost trial.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def payloads(
    cells: tuple[str, ...] = tuple(CELLS), trials: int = SIZE_TRIALS
) -> list[tuple[str, int]]:
    """Every property key, cell-major."""
    return [(name, trial) for name in cells for trial in range(trials)]


def generate_property_artifacts(
    *, n_jobs: int = STUDY_JOBS, replicates: int = SIZE_TRIALS
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Fit every trial and return the rows and the cell configurations.

    Parameters
    ----------
    n_jobs : int
        Worker processes.
    replicates : int
        Trials per size cell.

    Returns
    -------
    tuple
        The property rows, one per cell and trial, and :data:`CELLS_FILE` by file name.
    """
    outcomes = fit_properties(
        payloads(trials=replicates),
        splat_drawn(_replicate),
        cells_of=_cells_of,
        draw=splat_draw(_draw),
        cell_of=_cell_of,
        replicate_of=_replicate_of,
        n_jobs=n_jobs,
    )
    rows = pd.DataFrame([row for records in outcomes for row in records])
    for family, declared in PROPERTY_CELLS.items():
        cells = set(rows.loc[rows["property"] == family, "cell"])
        if cells != set(declared):
            raise RuntimeError(f"{family} cells {sorted(cells ^ set(declared))} differ")
    return rows.loc[:, list(REPLICATE_COLUMNS)], {CELLS_FILE: cell_configurations(rows)}


def generate_property_rows(
    *, n_jobs: int = STUDY_JOBS, replicates: int = SIZE_TRIALS
) -> pd.DataFrame:
    """Fit every trial and return one row per cell and trial.

    Parameters
    ----------
    n_jobs : int
        Worker processes.
    replicates : int
        Trials per size cell.

    Returns
    -------
    pandas.DataFrame
        The property rows.
    """
    rows, _ = generate_property_artifacts(n_jobs=n_jobs, replicates=replicates)
    return rows


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """Each cell's count rule, read through its exact interval."""
    summary, rates = apply_shared_verdicts(rows, STUDY, rate_labels=())
    size = summary["property"] == SIZE_FAMILY
    summary.loc[size, "passed"] = summary.loc[size, "rejection_ci_upper"] <= SIZE_CEILING
    power = summary["property"] == POWER_FAMILY
    summary.loc[power, "passed"] = summary.loc[power, "rejection_ci_lower"] >= POWER_FLOOR
    return finish(summary, rates)


def _rules() -> tuple[pf.FamilyRule, ...]:
    def every(ctx: pf.Context, row: pd.Series) -> bool:
        return True

    return (
        pf.FamilyRule(
            families=frozenset({SIZE_FAMILY}),
            shared="summarize_properties",
            clauses=(
                pf.proportion_clause(
                    "rejections at most 17 of 200",
                    statistic="S3",
                    judge=pf.ceiling_judge(lambda ctx: SIZE_CEILING),
                    applies=every,
                ),
            ),
        ),
        pf.FamilyRule(
            families=frozenset({POWER_FAMILY}),
            shared="summarize_properties",
            clauses=(
                pf.proportion_clause(
                    "detections at least 38 of 50",
                    statistic="S3",
                    judge=pf.floor_judge(lambda ctx: POWER_FLOOR),
                    applies=every,
                ),
            ),
        ),
    )


#: The failure accounting's declared rules (RM39b2 rev 3, D3'a).
PROPERTY_RULES = _rules()
