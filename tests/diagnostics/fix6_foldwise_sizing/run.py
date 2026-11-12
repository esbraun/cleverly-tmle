"""The sizing pilot of the Fix-6 fold-wise DR-TMLE records, before their declaration.

Each group fits the configuration of one record on fresh draws and reads the planning values
the sizing rules take: each estimand's standardized bias ``mean(error) / sd(error)``, its SE
ratio and its coverage, the per-replication error correlation, and the seconds each fit takes.

=========================  ===================================================================
group                      fit
=========================  ===================================================================
``b_outcome_correct``      unit B: the paper law at n = 1,500, ``outcome_correct``, fold-wise,
                           every primary row; its ``ate`` also plans the paired cell
``b_treatment_correct``    unit B: the same, ``treatment_correct``
``m_primary``              unit M: the multi-arm primary at n = 2,000, every estimand
``m_outcome_correct``      unit M: the ``double_robustness/outcome_correct`` cell, fold-wise
``s_primary``              unit S: the stratified primary at n = 2,000, ten rows
``ladder_tc_2000``         unit S ladder: stratified ``treatment_correct`` at n = 2,000
``ladder_tc_8000``         unit S ladder: the same at n = 8,000, which times the costliest fit
=========================  ===================================================================

Replication ``r`` of a group draws ``stream_seed(<record>, "sizing_pilot", <group>, r)`` with
the record's fresh resampling entropy, a label no study stream uses, so the pilot shares no
draw with a declared run and cannot change a verdict.  The summary goes to ``--output``, one
row per group and estimand, and the per-draw rows beside it as ``<stem>-rows.csv.gz``.  The
script prints what each record's sizing rule returns from the pilot's values.

    python -m tests.diagnostics.fix6_foldwise_sizing.run --output /scratch/pilot.csv --jobs 4
    python -m tests.diagnostics.fix6_foldwise_sizing.run --output /scratch/p.csv \\
        --replicates 2 --group ladder_tc_8000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cleverly.estimators import DRTMLE
from cleverly.utils.parallel import map_parallel
from cleverly.validation import CoverageStudy, ReplicationFailure
from tests.studies import canonical_drtmle, multi_arm_common
from tests.studies import canonical_stratified_drtmle as stratified
from tests.studies import drtmle_foldwise as foldwise
from tests.studies import drtmle_foldwise_n1500 as unit_b
from tests.studies import multi_arm_drtmle_foldwise as unit_m
from tests.studies import stratified_drtmle_foldwise as unit_s
from tests.studies import stratified_drtmle_foldwise_ladder as ladder
from tests.studies.drtmle_foldwise_sizing import as_rows, pilot_from_rows
from tests.studies.evidence.seeds import stream_seed
from tests.studies.point_study_helpers import primary_rows

#: ``group -> (record module, n)``.
GROUPS: dict[str, tuple[Any, int]] = {
    "b_outcome_correct": (unit_b, unit_b.PRIMARY_N),
    "b_treatment_correct": (unit_b, unit_b.PRIMARY_N),
    "m_primary": (unit_m, unit_m.PRIMARY_N),
    "m_outcome_correct": (unit_m, unit_m.registered_cell().n),
    "s_primary": (unit_s, unit_s.PRIMARY_N),
    "ladder_tc_2000": (unit_s, 2_000),
    "ladder_tc_8000": (unit_s, 8_000),
}
#: The ladder's rows, by package name.
LADDER_ESTIMANDS = tuple(stratified.ate_name(label) for label in unit_s.LABELS)
#: The pilot's default draws per group.
REPLICATES = 200


def seed(group: str, replicate: int) -> int:
    """The sample seed of one pilot replication: a stream no study reads."""
    module, _ = GROUPS[group]
    return stream_seed(module.STUDY, "sizing_pilot", group, replicate)


def fit(group: str, replicate: int) -> list[dict[str, Any]]:
    """Draw and fit one pilot replication; each row carries the fit's seconds.

    Parameters
    ----------
    group : str
        A key of :data:`GROUPS`.
    replicate : int
        The replication.

    Returns
    -------
    list of dict
        One row per estimand in the primary replication schema, with ``group`` and
        ``seconds``.
    """
    _, n = GROUPS[group]
    drawn = seed(group, replicate)
    if group.startswith("b_"):
        scenario = group.removeprefix("b_")
        frame, truth = canonical_drtmle.draw_from_seed(scenario, n, drawn)
        start = time.perf_counter()
        result = foldwise.fit_cleverly(frame, scenario, foldwise.PRIMARY_FIT)
        seconds = time.perf_counter() - start
        rows = primary_rows(
            result=result,
            truth=truth,
            implementation=group,
            scenario=scenario,
            replicate=replicate,
            estimands=unit_b.ESTIMANDS,
        )
    elif group == "m_outcome_correct":
        cell = unit_m.property_cell()
        study = CoverageStudy(
            dgp=cell.dgp,
            estimator=lambda: DRTMLE(**unit_m.cell_settings(cell)),
            n=n,
            # A budget that holds the replication asked for; only that one is fitted.
            n_replicates=max(2, replicate + 1),
            estimands=(cell.estimand,),
            fit_kwargs=dict(cell.fit_kwargs),
            seed=drawn,
            n_jobs=1,
        )
        start = time.perf_counter()
        outcome = study.run_replication(replicate, drawn)
        if isinstance(outcome, ReplicationFailure):
            raise RuntimeError(f"{group} replication {replicate} failed: {outcome}")
        (record,) = [r for r in outcome if r.estimand == cell.estimand]
        seconds = time.perf_counter() - start
        rows = [
            {
                "implementation": group,
                "scenario": unit_m.CELL,
                "replicate": replicate,
                "estimand": unit_m.CELL,
                "truth": record.truth,
                "estimate": record.estimate,
                "std_error": record.std_error,
                "covered": int(record.covered),
            }
        ]
    elif group == "m_primary":
        frame, truth = unit_m.draw_from_seed(unit_m.SCENARIO, n, drawn)
        start = time.perf_counter()
        result = unit_m.fit_cleverly(frame)
        seconds = time.perf_counter() - start
        rows = multi_arm_common.rows_from_result(
            result,
            truth,
            implementation=group,
            scenario=unit_m.SCENARIO,
            replicate=replicate,
            n=n,
            estimands=unit_m.ESTIMANDS,
        )
    else:
        frame, truth = stratified.draw_from_seed(stratified.SCENARIO, n, drawn)
        configuration = "both_correct" if group == "s_primary" else unit_s.POSITIVE
        estimands = unit_s.ESTIMANDS if group == "s_primary" else LADDER_ESTIMANDS
        start = time.perf_counter()
        result = unit_s.fit_cleverly(frame, configuration)
        seconds = time.perf_counter() - start
        rows = primary_rows(
            result=result,
            truth=truth,
            implementation=group,
            scenario=configuration,
            replicate=replicate,
            estimands=estimands,
        )
    for row in rows:
        row.update(group=group, n=n, seconds=seconds)
    return rows


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per group and estimand: its planning values and the group's seconds per fit.

    Parameters
    ----------
    rows : pandas.DataFrame
        The per-draw rows :func:`fit` returns.

    Returns
    -------
    pandas.DataFrame
        The summary.
    """
    out = []
    for group, grouped in rows.groupby("group", sort=False):
        # A property group's rows carry no inference-scale columns of the primary groups.
        frame = grouped.dropna(axis=1, how="all")
        pilot = pilot_from_rows(frame, source=f"Fix-6 sizing pilot, {group}")
        seconds = frame.drop_duplicates("replicate")["seconds"]
        for row in as_rows(pilot):
            row.update(
                group=group,
                n=int(frame["n"].iloc[0]),
                seconds_per_fit=float(seconds.mean()),
                seconds_per_fit_max=float(seconds.max()),
            )
            out.append(row)
    columns = ["group", "n", "estimand", "replicates", "standardized_bias", "se_ratio"]
    columns += ["coverage", "seconds_per_fit", "seconds_per_fit_max", "source"]
    return pd.DataFrame(out).loc[:, columns]


def sizing(rows: pd.DataFrame) -> list[str]:
    """What each record's sizing rule returns from the pilot's values.

    Parameters
    ----------
    rows : pandas.DataFrame
        The per-draw rows.

    Returns
    -------
    list of str
        One line per rule whose groups the pilot ran.
    """
    groups = {
        str(name): frame.dropna(axis=1, how="all")
        for name, frame in rows.groupby("group", sort=False)
    }
    lines = []
    if {"b_outcome_correct", "b_treatment_correct"} <= set(groups):
        pilots = {}
        for scenario in unit_b.SCENARIOS:
            frame = groups[f"b_{scenario}"]
            present = [name for name in unit_b.ESTIMANDS if name in set(frame["estimand"])]
            pilots[f"primary_{scenario}"] = pilot_from_rows(
                frame, source=f"pilot b_{scenario}", estimands=present
            )
            pilots[f"cell_{scenario}"] = pilot_from_rows(
                frame, source=f"pilot b_{scenario}", estimands=("ate",), calibration=False
            )
        rule = unit_b.rule(pilots)
        lines.append(f"unit B primary: {rule.budget('primary')}, cells: {rule.budget('cells')}")
    if groups.keys() & {"m_primary", "m_outcome_correct"}:
        primary = (
            pilot_from_rows(groups["m_primary"], source="pilot m_primary")
            if "m_primary" in groups
            else None
        )
        cell = (
            pilot_from_rows(
                groups["m_outcome_correct"], source="pilot m_outcome_correct", calibration=False
            )
            if "m_outcome_correct" in groups
            else None
        )
        rule = unit_m.rule(primary, cell)
        lines.append(
            f"unit M primary: {rule.budget('primary')}, "
            f"{unit_m.CELL}: {rule.budget(unit_m.CELL)} (a group not piloted keeps its "
            "registered planning value)"
        )
    if "s_primary" in groups:
        pilot = pilot_from_rows(groups["s_primary"], source="pilot s_primary")
        lines.append(
            f"unit S primary (less {', '.join(unit_s.EXPECTED_FINITE_SAMPLE)}): "
            f"{unit_s.rule({'primary': pilot}).budget('primary')}"
        )
    if {"ladder_tc_2000", "ladder_tc_8000"} <= set(groups):
        low = pilot_from_rows(groups["ladder_tc_2000"], source="pilot ladder 2,000")
        high = pilot_from_rows(groups["ladder_tc_8000"], source="pilot ladder 8,000")
        standardized = {}
        for label in unit_s.READ_LABELS:
            name = stratified.ate_name(label)
            z_low = abs(low.bias[low.rows.index(name)])
            z_high = abs(high.bias[high.rows.index(name)])
            # The middle rung carries no slope weight; its value only fills the triple.
            standardized[label] = (z_low, float(np.sqrt(z_low * z_high)), z_high)
        lines.append(f"ladder outer rungs: {ladder.planned_outer_budget(standardized)}")
    return lines


def main(argv: list[str] | None = None) -> None:
    """Run the pilot and write its summary and rows.

    Parameters
    ----------
    argv : list of str or None
        The command line, without the program name.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, required=True, help="the summary CSV")
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--group", action="append", choices=sorted(GROUPS), help="a group to run (repeatable)"
    )
    args = parser.parse_args(argv)
    chosen = args.group or list(GROUPS)
    payloads = [(group, r) for group in chosen for r in range(args.replicates)]
    outcomes = map_parallel(fit, payloads, n_jobs=args.jobs)
    rows = pd.DataFrame([row for rows in outcomes for row in rows])
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output.with_name(f"{output.stem}-rows.csv.gz"), index=False)
    summary = summarize(rows)
    summary.to_csv(output, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.drop(columns="source").to_string(index=False))
    if args.replicates >= 3:
        for line in sizing(rows):
            print(line)


if __name__ == "__main__":
    main()
