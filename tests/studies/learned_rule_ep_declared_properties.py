"""The property families of ``learned-rule-ep-declared``, and their verdicts (P1 plan 5.2).

One family, ``interval_calibration``, at n = 16,000 (:data:`~tests.studies.learned_rule_ep_declared.CALIBRATION_N`):

``<law>__S`` for ``nonexc``, ``misspec`` and ``optimal``
    The declared sieve.  Its SE-ratio interval must lie inside 0.93 to 1.07 and its coverage
    interval inside 0.92 to 0.98, read on the error ``estimate - truth``.  The truth of a
    learned-value cell is the fold-average value of the rules the fit learned (rule L4); the
    optimal cell's is :math:`\\Psi_{d_0}`.  Each row carries the paired limit-rule fit
    (:data:`LIMIT_COLUMNS`) on the same sample and outer folds, which the red-cell readings read.
``nonexc__L`` and ``misspec__L``
    The declared final learner, with the same band.  No limit is claimed, so no pair.
``optimal__L``
    Diagnostic (design review S4): its truth is :math:`\\Psi_{d_0}`, so it reads the greedy
    learner's regret, which no claim covers.  It publishes its SE ratio with its bootstrap
    interval, its coverage and the mean of the per-replicate ``regret``, and states no verdict.
``nonexc_S__shrunken_se_control``
    The ``nonexc__S`` fits with the SE times 0.70.  It must fall below the SE-ratio band.
``nonexc_L__validation_rule_control``
    Each fold's ``L`` curve fitted on its own validation rows, at n = 2,000 for 500 draws of its
    own.  Its coverage interval must lie wholly below 0.92.

Every calibration row carries the harness columns ``oracle_se`` (the fixed-rule SE of the
realized rules), ``rule_rows_checked`` and ``solver_warnings``, ``regret``
(:math:`\\Psi_{d_0}` minus the value of the fit's fold rules) and ``harness_verified`` (1 when the
harness refitted every fold's rule and found it equal to the recorded one; design review B4).

Every draw names its stream (:data:`tests.studies.learned_rule_ep_declared.STREAMS`): the
declared run reads ``property``, the pilot ``pilot`` and a smoke ``smoke``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import _learned_rule_law as base
from tests.studies import learned_rule_ep_declared as study
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import REPLICATE_COLUMNS, ratio_intervals
from tests.studies.evidence.property_verdicts import (
    DIAGNOSTIC_ROLE,
    apply_shared_verdicts,
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

#: The paired limit-rule fit's columns (``S`` rows only; NaN elsewhere).
LIMIT_COLUMNS = ("limit_estimate", "limit_std_error", "limit_truth")

#: The columns every property row carries beside the shared schema.
ROW_COLUMNS = (*base.HARNESS_COLUMNS, "regret", "harness_verified", *LIMIT_COLUMNS)

#: The summary columns the readings add.
SUMMARY_COLUMNS = ("regret_mean", "limit_se_ratio", "oracle_se_ratio")

_NO_LIMIT = dict.fromkeys(LIMIT_COLUMNS, float("nan"))


def replicates_of(cell: str) -> int:
    """The declared replications of a property cell."""
    if cell == study.LEAK_CONTROL:
        return study.CONTROL_REPLICATES
    if cell in study.DIAGNOSTIC_CELLS:
        return study.DIAGNOSTIC_REPLICATES
    return study.BUDGETED_R[cell.rsplit("__", 1)[-1].split("_")[0]]


def _law(cell: str) -> tuple[str, str]:
    """``(curve, law)`` of a calibration cell such as ``"misspec__L"``."""
    cell_law, curve = cell.split("__")
    return curve, study.CELL_LAWS[cell_law]


def _draw(cell: str, replicate: int, stream: str) -> pd.DataFrame:
    seed = study.sample_seed(cell, replicate, stream)
    if cell == study.LEAK_CONTROL:
        return base.draw("non_exceptional", study.PRIMARY_N, seed)
    _, law = _law(cell)
    name = "non_exceptional" if law == study.OPTIMAL_LABEL else law
    return base.draw(name, study.CALIBRATION_N, seed)


def _base_row(
    cell: str, role: str, replicate: int, requested: int, n: int, truth: float, estimate: Any
) -> dict[str, Any]:
    std_error, low, high = reported_inference(estimate)
    return {
        "property": "interval_calibration",
        "cell": cell,
        "role": role,
        "replicate": replicate,
        "n": n,
        "requested_replicates": requested,
        "failed_replicates": 0,
        "truth": truth,
        "estimate": float(estimate.psi),
        "std_error": std_error,
        "covered": int(low <= truth <= high),
        "rejected": int(not low <= 0.0 <= high),
    }


def _replicate(cell: str, replicate: int, stream: str, frame: pd.DataFrame) -> list[dict]:
    """Every row of one replication of one property cell."""
    seed = study.fold_seed(cell, replicate, stream)
    requested = replicates_of(cell)
    if cell == study.LEAK_CONTROL:
        result, harness, warnings = study.control(frame, seed)
        row = _base_row(
            cell,
            "control",
            replicate,
            requested,
            len(frame),
            harness.truth,
            result.estimates[study.ESTIMAND],
        )
        regret = study.d0_value("non_exceptional") - harness.truth
        return [
            {
                **row,
                "oracle_se": harness.oracle_se,
                "rule_rows_checked": harness.rows_checked,
                "solver_warnings": warnings,
                "regret": regret,
                "harness_verified": 1,
                **_NO_LIMIT,
            }
        ]
    curve, law = _law(cell)
    optimal = law == study.OPTIMAL_LABEL
    fitted = study.measured(
        frame,
        curve,
        "non_exceptional" if optimal else law,
        seed,
        check=study.verified(replicate, stream),
        optimal=optimal,
    )
    estimate = fitted.result.estimates[study.OPTIMAL_ESTIMAND if optimal else study.ESTIMAND]
    truth = study.d0_value(law) if optimal else fitted.truth.truth
    role = DIAGNOSTIC_ROLE if cell in study.DIAGNOSTIC_CELLS else "positive"
    row = _base_row(cell, role, replicate, requested, len(frame), truth, estimate)
    limit = dict(_NO_LIMIT)
    if curve == "S":
        paired = study.paired_fit(frame, law, seed, fitted.result, optimal=optimal)
        limit = dict(
            zip(LIMIT_COLUMNS, (paired.estimate, paired.std_error, paired.truth), strict=True)
        )
    row.update(
        {
            "oracle_se": fitted.truth.oracle_se,
            "rule_rows_checked": fitted.truth.rows_checked,
            "solver_warnings": fitted.warnings,
            "regret": study.d0_value(law) - fitted.truth.truth,
            "harness_verified": int(fitted.verified),
            **limit,
        }
    )
    rows = [row]
    if cell == study.calibration_cell("nonexc", "S"):
        std_error = row["std_error"] * study.SHRUNKEN_SE_FACTOR
        half = base.CRITICAL * std_error
        rows.append(
            {
                **row,
                "cell": study.SHRUNKEN_CONTROL,
                "role": "control",
                "std_error": std_error,
                "covered": int(abs(row["estimate"] - row["truth"]) <= half),
                "rejected": int(abs(row["estimate"]) > half),
            }
        )
    return rows


def _cells_of(key: tuple[str, int, str]) -> list[str]:
    names = [f"interval_calibration/{key[0]}"]
    if key[0] == study.calibration_cell("nonexc", "S"):
        names.append(f"interval_calibration/{study.SHRUNKEN_CONTROL}")
    return names


def payloads(budget: int | None = None, stream: str = "property") -> list[tuple[str, int, str]]:
    """Every ``(cell, replicate, stream)`` key; ``budget`` caps each cell."""
    cells = [
        cell for cell in PROPERTY_CELLS["interval_calibration"] if cell != study.SHRUNKEN_CONTROL
    ]
    return [
        (cell, replicate, stream)
        for cell in cells
        for replicate in range(
            replicates_of(cell) if budget is None else min(budget, replicates_of(cell))
        )
    ]


def generate_property_rows(
    *,
    n_jobs: int = STUDY_JOBS,
    budget: int | None = None,
    stream: str = "property",
    cells: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Fit every calibration and control replication; ``budget`` caps each cell.

    ``stream`` names the draws: the declared run is ``property``.  A run of every cell must
    fill exactly the declared cells.
    """
    keys = [key for key in payloads(budget, stream) if cells is None or key[0] in cells]
    outcomes = fit_properties(
        keys,
        splat_drawn(_replicate),
        draw=splat_draw(_draw),
        cells_of=_cells_of,
        cell_of=lambda key: key[0],
        replicate_of=lambda key: key[1],
        n_jobs=n_jobs,
        seed_of=lambda key: study.sample_seed(*key),
    )
    rows = pd.DataFrame([row for records in outcomes for row in records])
    if budget is not None:
        rows["requested_replicates"] = rows["requested_replicates"].clip(upper=budget)
    declared = set(PROPERTY_CELLS["interval_calibration"])
    found = set(rows["cell"])
    if found - declared or (cells is None and found != declared):
        raise RuntimeError(f"cells {sorted(found ^ declared)} differ from the declaration")
    return rows.loc[:, [*REPLICATE_COLUMNS, *ROW_COLUMNS]]


def _ratio(rows: pd.DataFrame, se: str, estimate: str, truth: str) -> float:
    errors = rows[estimate].to_numpy(float) - rows[truth].to_numpy(float)
    return float(rows[se].to_numpy(float).mean() / errors.std(ddof=1))


def readings(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """Add the reading columns: mean regret, the limit-rule fit's and the oracle SE's ratios.

    ``limit_se_ratio`` is the paired fit's mean SE over the SD of its error, on each ``S``
    cell; ``oracle_se_ratio`` is the mean ``oracle_se`` over the SD of the cell's own error, on
    each learned-value cell; ``regret_mean`` is the mean regret on every calibration row.  The
    diagnostic row also gets the bootstrap interval of its SE ratio, drawn as the summary's.
    """
    for column in SUMMARY_COLUMNS:
        summary[column] = np.nan
    margins = STUDY.margins
    for index, row in summary.iterrows():
        group = rows.loc[(rows["property"] == row["property"]) & (rows["cell"] == row["cell"])]
        if len(group) < 2:
            continue
        summary.loc[index, "regret_mean"] = float(group["regret"].mean())
        if group["limit_estimate"].notna().all():
            summary.loc[index, "limit_se_ratio"] = _ratio(
                group, "limit_std_error", "limit_estimate", "limit_truth"
            )
        if row["role"] == "positive" and not str(row["cell"]).startswith("optimal"):
            summary.loc[index, "oracle_se_ratio"] = _ratio(group, "oracle_se", "estimate", "truth")
        if row["role"] == DIAGNOSTIC_ROLE:
            interval = ratio_intervals(
                group,
                replicates=margins.bootstrap_replicates,
                confidence_level=margins.confidence_level,
                seed=stream_seed(STUDY, "interval_calibration", str(row["cell"])),
                truth_varies=STUDY.truth_varies_by_replicate,
            )["se_ratio"]
            summary.loc[index, "se_ratio_ci_lower"] = interval.low
            summary.loc[index, "se_ratio_ci_upper"] = interval.high


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The verdict of each cell, every statistic on the error of each row."""
    summary, rates = apply_shared_verdicts(
        rows.loc[:, list(REPLICATE_COLUMNS)], STUDY, rate_labels=()
    )
    # ``optimal__L`` carries the diagnostic role: it publishes its intervals and no verdict.
    calibration_verdicts(summary, margins=STUDY.margins, positive_suffix=study.CURVES)
    readings(summary, rows)
    return finish(summary, rates)


#: The declared rules of the failure accounting: the shared calibration rule.
PROPERTY_RULES = (property_verdicts.calibration_rule(efficiency_band=None),)

__all__ = [
    "FAILURE_BOUND",
    "PROPERTY_RULES",
    "PROPERTY_WITNESS",
    "generate_property_rows",
    "summarize_properties",
]
