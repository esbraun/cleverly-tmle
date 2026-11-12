"""The property families of the ``score-assessment`` study, and their verdicts.

Every group of :data:`tests.studies.score_assessment.GROUPS` is one fit per replication, and
its rows fill the cells :func:`~tests.studies.score_assessment.group_cells` names:

- ``interval_calibration``: each gated parameter's ``__correctly_specified`` row (the shipped
  99% SE-ratio and coverage bands), and on the paired row the two SE controls, the same
  estimate with the ``unpaired_se`` SE (``__inflated_se_control``, which must lie above the
  band) or the ``no_residual_se`` SE (``__term_dropped_control``, which must lie below it);
- ``bias``: the TMLE rows of the two declared-error groups, each inside +-0.25 of its own SD,
  and the ``untargeted`` plug-in rows, each wholly outside it
  (:func:`~tests.studies.evidence.property_verdicts.bias_role_verdicts`);
- ``type_i_error`` and ``power``: the paired row's Wald test of 0, its rejection read where the
  two risks are equal and at ``c = 0.13``.

A bias control is a number, not a fit with an interval, so its row carries a unit standard
error and only its bias is read.  The Wald rows carry the paired estimate and its SE, and the
test's decision as the rejection.

Besides the rows, the run writes ``comparison.csv`` (:func:`comparison`): for every parameter
of every group, the TMLE's and the one-step's mean, SD and standardized bias at the initial
fit, and the TMLE's coverage; and each group's count of fits that settled only on a reduced
span (``cleverly.effect_curves._alternate_log_ratio``, plan rev 2.2, S2) with the TMLE's
coverage among them.  It is published ungated.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import score_assessment as study
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import REPLICATE_COLUMNS, control_row, replicate_row
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    bias_role_rule,
    bias_role_verdicts,
    calibration_verdicts,
    finish,
)
from tests.studies.evidence.replication import fit_properties, splat_draw, splat_drawn
from tests.studies.evidence.schema import reported_inference
from tests.studies.evidence.seeds import stream_seed

STUDY = study.STUDY
PROPERTY_CELLS = study.PROPERTY_CELLS

#: The share of a cell's requested replications that the accounting lets a run lose.
FAILURE_BOUND = 0.01

#: The draw-level statistic the accounting reports for each lost replicate.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS

#: The families whose rows each answer to their own standardized-bias interval.
BIAS_FAMILIES = (study.BIAS,)

#: The ungated comparison table's file name.
COMPARISON_FILE = "comparison.csv"


def _draw(label: str, replicate: int) -> Any:
    """The draw of replication ``replicate`` of a group, from the stream keyed by its label."""
    group = study.GROUP_BY_LABEL[label]
    seed = stream_seed(STUDY, "property_sample", label, replicate)
    return study.draw_from_seed(label, group.n, seed)


def _replicate(
    label: str, replicate: int, drawn: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Every property row of one replication of one group, and its comparison records."""
    group = study.GROUP_BY_LABEL[label]
    frame, truths = drawn
    fitted = study.fit_group(group, frame, random_state=replicate)
    estimates = fitted["estimates"]
    alpha = STUDY.margins.alpha
    common = {"replicate": replicate, "n": group.n, "requested": group.replicates}
    rows: list[dict[str, Any]] = []
    for name in group.calibration:
        rows.append(
            replicate_row(
                property_name=study.CALIBRATION,
                cell=study.calibration_cell(group, name),
                role="positive",
                truth=float(truths[name]),
                estimate=estimates[name],
                alpha=alpha,
                **common,
            )
        )
    for control in group.controls:
        rows.append(
            control_row(
                property_name=study.CALIBRATION,
                cell=study.control_cell(group, control),
                truth=float(truths["paired"]),
                estimate=float(estimates["paired"].psi),
                standard_error=float(fitted[control]),
                critical=study.CRITICAL,
                **common,
            )
        )
    for name in group.bias:
        rows.append(
            replicate_row(
                property_name=study.BIAS,
                cell=study.bias_cell(group, name),
                role="positive",
                truth=float(truths[name]),
                estimate=estimates[name],
                alpha=alpha,
                **common,
            )
        )
        rows.append(
            control_row(
                property_name=study.BIAS,
                cell=study.bias_cell(group, name, control=True),
                truth=float(truths[name]),
                estimate=float(fitted["plug_in"][name]),
                standard_error=1.0,
                critical=study.CRITICAL,
                **common,
            )
        )
    if group.wald is not None:
        paired = estimates["paired"]
        rejected = study.wald_rejects(paired)
        rows.append(
            {
                "property": study.TYPE_I if group.wald == "size" else study.POWER,
                "cell": study.SIZE_CELL if group.wald == "size" else study.POWER_CELL,
                "role": "positive",
                "replicate": replicate,
                "n": group.n,
                "requested_replicates": group.replicates,
                "failed_replicates": 0,
                "truth": float(truths["paired"]),
                "estimate": float(paired.psi),
                "std_error": reported_inference(paired)[0],
                "covered": 1 - rejected,
                "rejected": rejected,
            }
        )
    records = []
    for name, estimate in estimates.items():
        _, low, high = reported_inference(estimate)
        records.append(
            {
                "group": group.label,
                "parameter": name,
                "replicate": replicate,
                "truth": float(truths[name]),
                "tmle": float(estimate.psi),
                "one_step": float(fitted["one_step"][name]),
                "covered": int(low <= float(truths[name]) <= high),
                "reduced": int(fitted["reduced"]),
            }
        )
    return rows, records


def _cells_of(key: tuple[str, int]) -> list[str]:
    group = study.GROUP_BY_LABEL[key[0]]
    return [
        f"{family}/{cell}" for family, cells in study.group_cells(group).items() for cell in cells
    ]


def _cell_of(key: tuple[str, int]) -> str:
    return f"group/{key[0]}"


def _replicate_of(key: tuple[str, int]) -> int:
    return key[1]


def payloads(budget: int | None = None) -> list[tuple[str, int]]:
    """Every ``(group, replicate)`` key, in group order; ``budget`` caps each group."""
    return [
        (group.label, index)
        for group in study.GROUPS
        for index in range(group.replicates if budget is None else min(budget, group.replicates))
    ]


def _fit_all(n_jobs: int, budget: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcomes = fit_properties(
        payloads(budget),
        splat_drawn(_replicate),
        draw=splat_draw(_draw),
        cells_of=_cells_of,
        cell_of=_cell_of,
        replicate_of=_replicate_of,
        n_jobs=n_jobs,
        rows_at=0,
        seed_of=lambda key: stream_seed(STUDY, "property_sample", key[0], key[1]),
    )
    rows = pd.DataFrame([row for value in outcomes for row in value[0]])
    records = pd.DataFrame([record for value in outcomes for record in value[1]])
    return rows, records


def _property_rows(rows: pd.DataFrame, budget: int | None) -> pd.DataFrame:
    rows = rows.loc[:, list(REPLICATE_COLUMNS)]
    if budget is not None:
        rows["requested_replicates"] = rows["requested_replicates"].clip(upper=budget)
    declared = {(family, cell) for family, cells in PROPERTY_CELLS.items() for cell in cells}
    found = set(zip(rows["property"], rows["cell"], strict=True))
    if found != declared:
        raise RuntimeError(f"cells {sorted(found ^ declared)} differ from the declaration")
    return rows


def comparison(records: pd.DataFrame) -> pd.DataFrame:
    """The ungated comparison of the TMLE with the one-step at the initial fit.

    Parameters
    ----------
    records : pandas.DataFrame
        The comparison records: ``group``, ``parameter``, ``truth``, ``tmle``, ``one_step``,
        ``covered`` and ``reduced`` (the directions the fit dropped), one per replicate.

    Returns
    -------
    pandas.DataFrame
        One row per group and parameter: the replicate count, the truth, each estimator's mean,
        SD and standardized bias, the TMLE's coverage, the count of reduced-span fits and the
        TMLE's coverage among them (NaN when there is none).
    """
    out = []
    for (group, parameter), block in records.groupby(["group", "parameter"], sort=False):
        truth = float(block["truth"].iloc[0])
        row: dict[str, Any] = {
            "group": group,
            "parameter": parameter,
            "replicates": len(block),
            "truth": truth,
        }
        for column in ("tmle", "one_step"):
            values = block[column].to_numpy(dtype=float)
            sd = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_sd"] = sd
            row[f"{column}_standardized_bias"] = (float(values.mean()) - truth) / sd
        row["tmle_coverage"] = float(np.mean(block["covered"]))
        reduced = block.loc[block["reduced"] > 0, "covered"]
        row["reduced_fits"] = len(reduced)
        row["reduced_coverage"] = float(np.mean(reduced)) if len(reduced) else float("nan")
        out.append(row)
    return pd.DataFrame(out)


def generate_property_artifacts(
    *, n_jobs: int = STUDY_JOBS, budget: int | None = None
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Fit every property replication, and tabulate the ungated comparison from the same fits.

    Parameters
    ----------
    n_jobs : int
        Worker processes.
    budget : int or None
        A cap on each group's replications, for a pre-run check; ``None`` is the declaration.

    Returns
    -------
    tuple
        The property rows, and the comparison table by file name.
    """
    rows, records = _fit_all(n_jobs, budget)
    return _property_rows(rows, budget), {COMPARISON_FILE: comparison(records)}


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each group for a pre-run check."""
    return _property_rows(_fit_all(n_jobs, budget)[0], budget)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts, the calibration kinds and the bias rows."""
    summary, rates = apply_shared_verdicts(rows, STUDY, rate_labels=())
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=None)
    bias_role_verdicts(summary, families=BIAS_FAMILIES)
    return finish(summary, rates)


#: The failure accounting's declared rules.  The calibration, type-I and power families read
#: the shared rules (:data:`~tests.studies.evidence.property_verdicts.FAMILY_RULES`).
PROPERTY_RULES = (bias_role_rule(BIAS_FAMILIES),)

__all__ = [
    "FAILURE_BOUND",
    "PROPERTY_RULES",
    "PROPERTY_WITNESS",
    "comparison",
    "generate_property_artifacts",
    "generate_property_rows",
    "summarize_properties",
]
