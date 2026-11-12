"""The sizing pilot of the Fix-6 fold-wise DR-TMLE records, before their declaration.

Each group fits the configuration of one record on fresh draws and reads the planning values
the sizing rules take: each estimand's standardized bias ``mean(error) / sd(error)``, its SE
ratio and its coverage, the per-replication error correlation, and the seconds each fit takes.

=========================  ===================================================================
group                      fit
=========================  ===================================================================
``b_both_correct``         unit B primary: the paper law at n = 1,500, ``both_correct``,
                           fold-wise, every primary row
``b_outcome_correct``      unit B: the same law, ``outcome_correct``; its ``ate`` plans the
                           paired cell
``b_treatment_correct``    unit B: the same, ``treatment_correct``
``m_primary``              unit M: the multi-arm primary at n = 2,000, every estimand
``m_outcome_correct``      unit M: the ``double_robustness/outcome_correct`` cell, fold-wise
``s_primary``              unit S: the stratified primary at n = 2,000, ten rows
``ladder_tc_2000``         unit S ladder: stratified ``treatment_correct`` at n = 2,000
``ladder_tc_8000``         unit S ladder: the same at n = 8,000, which times the costliest fit
=========================  ===================================================================

Replication ``r`` of a group draws ``stream_seed(<record>, "sizing_pilot", <group>, r)`` with
the record's fresh resampling entropy (:func:`seed`), a label no study stream uses, so the pilot
shares no draw with a declared run and cannot change a verdict.  The ladder groups keep the
entropy their first 200 replications drew with, the retired ladder record's
(:data:`LADDER_PILOT_ENTROPY`).  ``--start`` sets the first replicate index, so a later run
continues a group's stream on new indices, and ``--rows`` (repeatable) adds earlier per-draw
rows to the summary and the sizing.  The summary goes to ``--output``, one row per group and
estimand, and this run's per-draw rows beside it as ``<stem>-rows.csv.gz``.  The script prints
what each record's sizing rule returns, and the rows each rule would set aside.

    python -m tests.diagnostics.fix6_foldwise_sizing.run --output /scratch/pilot.csv --jobs 4
    python -m tests.diagnostics.fix6_foldwise_sizing.run --output /scratch/p.csv \\
        --start 200 --replicates 800 --group b_both_correct
    python -m tests.diagnostics.fix6_foldwise_sizing.run --output /scratch/all.csv \\
        --replicates 0 --rows /scratch/p-rows.csv.gz --rows /scratch/q-rows.csv.gz
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path
from typing import Any

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
from tests.studies.drtmle_foldwise_sizing import (
    SET_ASIDE_THRESHOLD,
    Rule,
    as_rows,
    ladder_plan_from_rows,
    pilot_from_rows,
    set_aside_rows,
)
from tests.studies.evidence.seeds import stream_seed
from tests.studies.point_study_helpers import primary_rows

#: ``group -> (record module, n)``.
GROUPS: dict[str, tuple[Any, int]] = {
    "b_both_correct": (unit_b, unit_b.PRIMARY_N),
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


#: The entropy the ladder groups' streams draw with: the retired
#: ``stratified-drtmle-foldwise-ladder`` record's resampling seed, which their first 200
#: replications used, so a continued run extends the same streams.
LADDER_PILOT_ENTROPY = 2026111214


def seed(group: str, replicate: int) -> int:
    """The sample seed of one pilot replication: a stream no study reads."""
    module, _ = GROUPS[group]
    record = module.STUDY
    if group.startswith("ladder_"):
        record = dataclasses.replace(record, resampling_seed=LADDER_PILOT_ENTROPY)
    return stream_seed(record, "sizing_pilot", group, replicate)


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

    def report(name: str, rule: Rule) -> None:
        lines.append(f"{name}: " + ", ".join(f"{u} {rule.budget(u)}" for u in rule.units))
        aside, chosen = set_aside_rows(rule)
        if chosen:
            named = "; ".join(f"{p}/{r} {c} {v:.3f}" for p, r, c, v in chosen)
            lines.append(f"  set aside below {SET_ASIDE_THRESHOLD} at the ceiling: {named}")
            lines.append(
                f"  {name} without them: "
                + ", ".join(f"{u} {aside.budget(u)}" for u in aside.units)
            )

    if {"b_outcome_correct", "b_treatment_correct"} <= set(groups):
        pilots = dict(unit_b.planning_pilots())
        for scenario in unit_b.SCENARIOS:
            pilots[f"cell_{scenario}"] = pilot_from_rows(
                groups[f"b_{scenario}"],
                source=f"pilot b_{scenario}",
                estimands=("ate",),
                calibration=False,
            )
        if "b_both_correct" in groups:
            pilots["primary_both_correct"] = pilot_from_rows(
                groups["b_both_correct"], source="pilot b_both_correct", estimands=unit_b.ESTIMANDS
            )
        report("unit B (a primary not piloted keeps its planning value)", unit_b.rule(pilots))
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
        report(
            "unit M (a group not piloted keeps its registered planning value)",
            unit_m.rule(primary, cell),
        )
    if "s_primary" in groups:
        pilot = pilot_from_rows(groups["s_primary"], source="pilot s_primary")
        report(
            f"unit S primary (less {', '.join(unit_s.EXPECTED_FINITE_SAMPLE)})",
            unit_s.rule({"primary": pilot}),
        )
    if "ladder_tc_8000" in groups:
        plan = ladder_plan_from_rows(
            groups["ladder_tc_8000"],
            source="pilot ladder_tc_8000",
            labels={label: stratified.ate_name(label) for label in unit_s.READ_LABELS},
            size=8_000,
        )
        lines.append(f"ladder outer rungs: {ladder.planned_outer_budget(plan)}")
        lines.append(f"ladder control per rung: {ladder.planned_control_budget()}")
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
    parser.add_argument("--start", type=int, default=0, help="the first replicate index")
    parser.add_argument(
        "--rows", type=Path, action="append", default=[], help="earlier per-draw rows (repeatable)"
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--group", action="append", choices=sorted(GROUPS), help="a group to run (repeatable)"
    )
    args = parser.parse_args(argv)
    chosen = args.group or list(GROUPS)
    stop = args.start + args.replicates
    payloads = [(group, r) for group in chosen for r in range(args.start, stop)]
    outcomes = map_parallel(fit, payloads, n_jobs=args.jobs) if payloads else []
    fresh = pd.DataFrame([row for rows in outcomes for row in rows])
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not fresh.empty:
        fresh.to_csv(output.with_name(f"{output.stem}-rows.csv.gz"), index=False)
    rows = pd.concat([*(pd.read_csv(path) for path in args.rows), fresh], ignore_index=True)
    if rows.empty:
        raise SystemExit("no rows: pass --replicates above 0 or --rows")
    repeated = rows.duplicated(["group", "replicate", "estimand"])
    if repeated.any():
        raise SystemExit(f"{int(repeated.sum())} rows repeat a (group, replicate, estimand)")
    summary = summarize(rows)
    summary.to_csv(output, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.drop(columns="source").to_string(index=False))
    if rows.groupby("group")["replicate"].nunique().min() >= 3:
        for line in sizing(rows):
            print(line)


if __name__ == "__main__":
    main()
