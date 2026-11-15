"""Size and power of the continuous-time survival refutation rules (X101 PR 2).

Two families. ``refuter_size`` holds one cell per rule and estimand: 200 trials on the size
law, and the cell passes when the rule rejects in at most 17. ``refuter_power`` holds one cell
per defect and estimand, and passes when the rule detects the defect in at least 38 of 50
trials. The stale-row placebo cells read 50 trials of their own on the power law; the rescaled
subset cells read size trials 0 to 49, so they differ from their size cell only in the defect.

Each trial draws its own sample from
``trial_seed(stream, "sample", group, trial)`` (:func:`~tests.studies.refuter_calibration_continuous_survival.trial_seed`),
on the stream ``x101-refuter`` for size trials and ``x101-refuter-power`` for power trials.

Each trial also records the configuration of its fit for each estimand.
:func:`generate_property_artifacts` stops the run if a cell's configuration varies, and writes
the one configuration of each estimand and law to ``cells.csv``. It also runs the two reporting
cells (no verdict) through a checkpointed loop of their own and writes
``reporting-cells.csv`` and ``reporting-replicates.csv``.

The verdicts are the plan's count rules. The failure accounting reads them through the exact
99% Clopper-Pearson interval of the shared S3 clause: :data:`SIZE_CEILING` lies strictly
between the upper endpoints at 17 and 18 rejections of 200, and :data:`POWER_FLOOR` strictly
between the lower endpoints at 37 and 38 detections of 50. With no lost trial the interval rule
and the count rule agree, and :func:`count_rule_agrees` proves it for every count.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies.evidence import property_failures as pf
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.inference import clopper_pearson
from tests.studies.evidence.properties import REPLICATE_COLUMNS, PropertyCell
from tests.studies.evidence.property_verdicts import apply_shared_verdicts, finish
from tests.studies.evidence.replication import (
    fit_properties,
    fit_replicates,
    splat_draw,
    splat_drawn,
)
from tests.studies.refuter_calibration_continuous_survival import (
    ESTIMANDS,
    GROUPS,
    POWER_FAMILY,
    POWER_MIN_DETECTIONS,
    POWER_TRIALS,
    PROPERTY_CELLS,
    REPORTING_CELLS,
    REPORTING_STREAM,
    REPORTING_TRIALS,
    SIZE_FAMILY,
    SIZE_MAX_REJECTIONS,
    SIZE_READINGS,
    SIZE_TRIALS,
    STUDY,
    SUBSET_DEFECTS,
    N,
    draw_trial,
    trial_readings,
    trial_seed,
)

#: The committed file that records each cell's fit configuration.
CELLS_FILE = "cells.csv"
#: The reporting cells' summary and rows.
REPORTING_CELLS_FILE = "reporting-cells.csv"
REPORTING_REPLICATES_FILE = "reporting-replicates.csv"
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
_REFERENCE = {"placebo": 0.0, "subset": 1.0}


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
        # Each statistic is already on its standardized scale (``t`` or ``rho``), so the
        # row's scale is 1. No verdict reads it; the framework requires a positive value.
        "std_error": 1.0,
        "covered": int(not reading.rejected),
        "rejected": int(reading.rejected),
        _CONFIGURATION: configuration,
    }


def _draw(group: str, trial: int) -> Any:
    return draw_trial(group, trial)


def _configuration(group: str, configurations: dict[str, dict[str, Any]], key: str) -> str:
    return json.dumps({"law": GROUPS[group][1], **configurations[key]}, sort_keys=True)


def _replicate(group: str, trial: int, drawn: Any) -> list[dict[str, Any]]:
    outcome, _, _ = trial_readings(group, drawn, trial)
    rows = []
    for reading, value in outcome.readings.items():
        key, kind = reading.split("__", 1)
        size = group == "size" and kind in SIZE_READINGS
        rows.append(
            _row(
                SIZE_FAMILY if size else POWER_FAMILY,
                reading,
                "positive" if size else "control",
                trial,
                SIZE_TRIALS if size else POWER_TRIALS,
                value,
                _configuration(group, outcome.configurations, key),
            )
        )
    return rows


def cell_configurations(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per estimand and law: the configuration every trial of its cells fitted.

    The smallest event count and the event share differ from one sample to the next, so the
    row records their ranges over the trials (a calibration scope covers event counts at or
    above its floor and the measured event-share range). Every other axis must be the same on
    every trial.

    Parameters
    ----------
    rows : pandas.DataFrame
        The property rows with their configuration column.

    Returns
    -------
    pandas.DataFrame
        One row per estimand of :data:`ESTIMANDS` and law that a cell read.

    Raises
    ------
    RuntimeError
        If an estimand's trials on one law fitted more than one configuration.
    """
    names = rows["cell"].str.split("__", n=1).str[0]
    records = []
    for key in ESTIMANDS:
        fits = [json.loads(text) for text in set(rows.loc[names == key, _CONFIGURATION])]
        for law_name in sorted({fit["law"] for fit in fits}):
            chosen = [dict(fit) for fit in fits if fit["law"] == law_name]
            events = [fit.pop("min_event_count") for fit in chosen]
            shares = [fit.pop("event_share") for fit in chosen]
            axes = {json.dumps(fit, sort_keys=True) for fit in chosen}
            if len(axes) != 1:
                raise RuntimeError(
                    f"estimand {key} on the {law_name} law fitted {len(axes)} configurations, "
                    "expected one"
                )
            records.append(
                {
                    "cell": key,
                    **json.loads(axes.pop()),
                    "min_event_count_min": min(events),
                    "min_event_count_max": max(events),
                    "event_share_min": min(shares),
                    "event_share_max": max(shares),
                }
            )
    table = pd.DataFrame(records)
    for column in ("min_event_count_min", "min_event_count_max"):
        table[column] = table[column].astype("Int64")
    return table


def _cells_of(key: tuple[str, int]) -> list[str]:
    group, trial = key
    if group == "power":
        return [f"{POWER_FAMILY}/{name}__placebo_stale_rows" for name in ESTIMANDS]
    cells = [f"{SIZE_FAMILY}/{name}__{reading}" for name in ESTIMANDS for reading in SIZE_READINGS]
    if trial < POWER_TRIALS:
        cells += [
            f"{POWER_FAMILY}/{name}__{defect}" for name in ESTIMANDS for defect in SUBSET_DEFECTS
        ]
    return cells


def _cell_of(key: tuple[str, int]) -> str:
    first = next(iter(ESTIMANDS))
    if key[0] == "power":
        return f"{POWER_FAMILY}/{first}__placebo_stale_rows"
    return f"{SIZE_FAMILY}/{first}__placebo"


def _replicate_of(key: tuple[str, int]) -> int:
    return key[1]


#: The share of a cell's requested trials that the accounting lets a run lose.
FAILURE_BOUND = 0.01

#: The draw-level statistic the accounting reports for each lost trial.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def payloads(trials: int = SIZE_TRIALS) -> list[tuple[str, int]]:
    """Every property key: the size trials, then the power trials."""
    power = min(trials, POWER_TRIALS)
    return [("size", trial) for trial in range(trials)] + [
        ("power", trial) for trial in range(power)
    ]


# ---- the reporting cells -------------------------------------------------------------------


def _reporting_draw(key: tuple[str, int]) -> Any:
    return draw_trial(*key)


def _reporting_fit(key: tuple[str, int], drawn: Any) -> list[dict[str, Any]]:
    cell, trial = key
    outcome, _, _ = trial_readings(cell, drawn, trial)
    rows = []
    for name in ESTIMANDS:
        reading = outcome.readings[f"{name}__placebo"]
        configuration = outcome.configurations[name]
        rows.append(
            {
                "cell": cell,
                "estimand": name,
                "trial": trial,
                "statistic": reading.statistic,
                "pvalue": reading.pvalue,
                "rejected": int(reading.rejected),
                "event_share": configuration["event_share"],
                "min_event_count": configuration["min_event_count"],
                "failure": "",
            }
        )
    return rows


def _reporting_failed(key: tuple[str, int], drawn: Any) -> list[dict[str, Any]]:
    cell, trial = key
    return [
        {
            "cell": cell,
            "estimand": name,
            "trial": trial,
            "statistic": np.nan,
            "pvalue": np.nan,
            "rejected": np.nan,
            "event_share": np.nan,
            "min_event_count": np.nan,
            "failure": "fitting_failure",
        }
        for name in ESTIMANDS
    ]


def reporting_rows(*, n_jobs: int = STUDY_JOBS, trials: int = REPORTING_TRIALS) -> pd.DataFrame:
    """Every reporting trial's placebo readings, one row per estimand, in a fixed order.

    The loop is checkpointed like the property loop, so a restarted run reuses its finished
    trials. A trial that raises a recordable error publishes rows with ``failure`` set.
    """
    keys = [(cell, trial) for cell in REPORTING_CELLS for trial in range(trials)]
    outcomes = fit_replicates(
        keys,
        _reporting_fit,
        n_jobs=n_jobs,
        label="properties/reporting",
        implementation=STUDY.implementation,
        cell_of=lambda key: key[0],
        replicate_of=lambda key: key[1],
        draw=_reporting_draw,
        on_failure=_reporting_failed,
        seed_of=lambda key: trial_seed(REPORTING_STREAM, "sample", key[0], key[1]),
    )
    rows = pd.DataFrame([row for values in outcomes.fitted for row in values])
    return rows.sort_values(["cell", "estimand", "trial"], ignore_index=True)


def summarize_reporting(rows: pd.DataFrame) -> pd.DataFrame:
    """Per reporting cell and estimand: trials, failures, rejections and the rejection rate.

    The 99% Clopper-Pearson interval of the rate is printed beside it. No verdict reads it.
    """
    records = []
    for (cell, estimand), group in rows.groupby(["cell", "estimand"], sort=True):
        kept = group.loc[group["failure"].fillna("").eq("")]
        rejections = int(kept["rejected"].sum())
        interval = clopper_pearson(rejections, len(kept), confidence_level=_LEVEL)
        records.append(
            {
                "cell": cell,
                "estimand": estimand,
                "trials": len(group),
                "failures": len(group) - len(kept),
                "rejections": rejections,
                "rejection_rate": rejections / len(kept),
                "rejection_ci_lower": interval.low,
                "rejection_ci_upper": interval.high,
                "mean_statistic": float(kept["statistic"].mean()),
                "event_share_min": float(kept["event_share"].min()),
                "event_share_max": float(kept["event_share"].max()),
            }
        )
    return pd.DataFrame(records)


def _property_rows(*, n_jobs: int, replicates: int) -> pd.DataFrame:
    """The size and power trials' rows, with each row's configuration column."""
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
    return rows


def generate_property_artifacts(
    *, n_jobs: int = STUDY_JOBS, replicates: int = SIZE_TRIALS
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Fit every trial and return the rows, the cell configurations and the reporting cells.

    Parameters
    ----------
    n_jobs : int
        Worker processes.
    replicates : int
        Trials per size cell; the power and reporting cells run at most their declared
        counts and at most this many.

    Returns
    -------
    tuple
        The property rows, one per cell and trial, and :data:`CELLS_FILE`,
        :data:`REPORTING_CELLS_FILE` and :data:`REPORTING_REPLICATES_FILE` by file name.
    """
    rows = _property_rows(n_jobs=n_jobs, replicates=replicates)
    reporting = reporting_rows(n_jobs=n_jobs, trials=min(replicates, REPORTING_TRIALS))
    return rows.loc[:, list(REPLICATE_COLUMNS)], {
        CELLS_FILE: cell_configurations(rows),
        REPORTING_CELLS_FILE: summarize_reporting(reporting),
        REPORTING_REPLICATES_FILE: reporting,
    }


def generate_property_rows(
    *, n_jobs: int = STUDY_JOBS, replicates: int = SIZE_TRIALS
) -> pd.DataFrame:
    """Fit the size and power trials and return one row per cell and trial.

    The reporting cells carry no verdict and do not run here.

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
    rows = _property_rows(n_jobs=n_jobs, replicates=replicates)
    return rows.loc[:, list(REPLICATE_COLUMNS)]


@dataclass(frozen=True)
class DeclaredLaw:
    """The law a declared cell reads, by name (:func:`~tests.studies.refuter_calibration_continuous_survival.draw_law`).

    Parameters
    ----------
    name : str
        ``"size"`` or ``"power"``.
    """

    name: str


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every size and power cell, with its law, trial count, first seed and role."""
    cells = []
    for name in PROPERTY_CELLS[SIZE_FAMILY]:
        cells.append(_declared(SIZE_FAMILY, name, "size", SIZE_TRIALS, "positive"))
    for name in PROPERTY_CELLS[POWER_FAMILY]:
        group = "power" if name.endswith("placebo_stale_rows") else "size"
        cells.append(_declared(POWER_FAMILY, name, group, POWER_TRIALS, "control"))
    return tuple(cells)


def _declared(family: str, cell: str, group: str, trials: int, role: str) -> PropertyCell:
    stream = GROUPS[group][0]
    return PropertyCell(
        property=family,
        cell=cell,
        dgp=DeclaredLaw(GROUPS[group][1]),
        outcome_learner=lambda: None,
        treatment_learner=lambda: None,
        n=N,
        replicates=trials,
        seed=trial_seed(stream, "sample", group, 0),
        role=role,
        estimand=ESTIMANDS[cell.split("__", 1)[0]],
    )


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
