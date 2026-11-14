"""The pilot of the few-cluster floor studies, on its own ``"pilot"`` seed stream (plan 5, M5).

Run before the declaration is final (review M4): 400 draws of every cell of one study, through
the study's own property loop and summary. The pilot reads the ``"pilot"`` stream label, so no
pilot draw is a property draw. It changes no rule, margin, law or ladder; it sets each family's
R (steps of 400 from 4,000, capped at 6,000) and records the predicted reds. It writes, beside
the study's declaration (``STUDY.artifacts``, or ``--output``):

=========================  ===================================================================
file                       content
=========================  ===================================================================
``pilot.csv.gz``           the rows
``pilot-failures.csv``     one row per lost key: fit, law, J, replicate, which key (main fit
                           or (C)), exception class, call and message. Every lost main key is
                           a main-fit failure: (C) and the package-GLM refit never raise out
                           of a key (review B1)
``pilot-summary.csv``      the study's own summary of the rows, which the design test reads
``pilot-cells.csv``        per cell: withheld share by trigger, lost share by cause, shown-only
                           and all-draw coverage, the SE ratio, the control's paired ratio,
                           the package-GLM raised share and (C)'s failed-refit share
``pilot-sizing.csv``       per floor family: the expected floor, the R the M5 rule picks, the
                           plug-in family figure with every control clause, the predictive
                           figure; then each predicted red with its reading
                           (:data:`~tests.studies.few_cluster_floor.READINGS`), and for S-a
                           the predicted reds of the reported d5 reading
``pilot-candidates.csv``   per part (c) family: (C)'s discordance with (A) and the family-wise
                           probability that a (C) as good as (A) passes clause (i); below 0.95
                           is a predicted shortfall
``pilot-timing.csv``       CPU seconds per key by fit and J (main fit, (C), package GLM), and
                           the projected CPU hours at the R the sizing picks (read against
                           1.25 times the declared budget) and at R = 6,000
=========================  ===================================================================

.. code-block:: bash

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
      python -m tests.studies.few_cluster_floor_pilot tmle --jobs 4
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
      python -m tests.studies.few_cluster_floor_pilot ltmle --jobs 4
"""

from __future__ import annotations

import argparse
import importlib
import math
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import numpy as np
import pandas as pd

from tests.studies import few_cluster_floor as floor
from tests.studies.evidence import property_failures
from tests.studies.evidence.replication import recording

#: The pilot's draws per cell (plan 5, general rules).
PILOT_DRAWS: Final[int] = 400
MODULES: Final[dict[str, str]] = {
    "tmle": "tests.studies.few_cluster_floor_tmle_properties",
    "ltmle": "tests.studies.few_cluster_floor_ltmle_properties",
}
#: The declared property-phase budgets (plan 5.9 and the declarations), in CPU hours.
DECLARED_CPU_HOURS: Final[dict[str, float]] = {"tmle": 60.0, "ltmle": 22.0}
#: A measured total above this multiple of the declared budget goes back to the row thread.
BUDGET_TOLERANCE: Final[float] = 1.25
#: The clause-1 power a control cell needs at the chosen R not to be a predicted red.
CONTROL_POWER: Final[float] = 0.90
#: The single-cell plug-in pass probability below which a cell is a predicted red.
RED_PROBABILITY: Final[float] = 0.90
#: The family-wise clause (i) probability below which (C) has a predicted shortfall.
JACKKNIFE_TARGET: Final[float] = 0.95


def _families(which: str, module: ModuleType) -> dict[str, Any]:
    """Each floor family of a study: its anchor and its positive and control rows."""
    if which == "tmle":
        study = module.study
        return {
            family: (anchor, lambda summary, f=family: module.floor_cells(summary, f))
            for family, anchor in study.ANCHORS.items()
        }
    return {module.study.FIT: (module.study.ANCHOR, module.floor_cells)}


def failure_table(records: pd.DataFrame | None) -> pd.DataFrame:
    """One row per lost key, with the key's fit, law, count, kind and exception."""
    columns = ["fit", "law", "clusters", "replicate", "key", "class", "call", "message"]
    if records is None or records.empty:
        return pd.DataFrame(columns=columns)
    out = []
    for record in records.to_dict("records"):
        cells = [part.split("/", 1) for part in str(record["scenarios"]).split(";") if part]
        family, cell = cells[0]
        fit, law, count, arm = cell.split("__")
        jackknife = family == floor.CANDIDATE_FAMILY and arm == floor.JACKKNIFE_ARM
        for replicate in (int(part) for part in str(record["replicates"]).split(";") if part):
            out.append(
                [
                    fit,
                    law,
                    int(count.removeprefix("j")),
                    replicate,
                    "jackknife" if jackknife else "main",
                    str(record["class"]),
                    str(record["call"]),
                    str(record["message"])[:200],
                ]
            )
    return pd.DataFrame(out, columns=columns).sort_values(
        ["fit", "law", "clusters", "key", "replicate"], ignore_index=True
    )


def cell_table(summary: pd.DataFrame, failures: pd.DataFrame) -> pd.DataFrame:
    """Per cell: availability, coverage, SE ratios, the package-GLM and (C)'s refit shares."""
    rows = summary.loc[summary["property"].isin([floor.FAMILY, floor.CANDIDATE_FAMILY])].copy()
    causes: dict[tuple[str, str, int, str], str] = {}
    for (fit, law, clusters, key), group in failures.groupby(["fit", "law", "clusters", "key"]):
        counts = group["class"].value_counts().sort_index()
        causes[(fit, law, int(clusters), key)] = ";".join(f"{k}={v}" for k, v in counts.items())
    lost_by_cause = []
    for _, row in rows.iterrows():
        fit, law, count, arm = str(row["cell"]).split("__")
        key = "jackknife" if arm == floor.JACKKNIFE_ARM else "main"
        lost_by_cause.append(causes.get((fit, law, int(count.removeprefix("j")), key), ""))
    rows["lost_by_cause"] = lost_by_cause
    requested = rows["requested_draws"].astype(float)
    rows["lost_share"] = (requested - rows["replicates"].astype(float)) / requested
    rows["withheld_share"] = rows["withheld_draws"].astype(float) / requested
    columns = [
        "property",
        "cell",
        "role",
        "requested_draws",
        "replicates",
        "lost_share",
        "lost_by_cause",
        "withheld_share",
        "withheld_by_trigger",
        "shown_coverage",
        "shown_ci_lower",
        "shown_ci_upper",
        "all_draw_coverage",
        "all_draw_ci_lower",
        "all_draw_ci_upper",
        "se_ratio",
        "se_ratio_ci_lower",
        "se_ratio_ci_upper",
        "paired_se_ratio",
        "paired_se_ratio_ci_upper",
        "package_glm_raised_share",
        "shown_coverage_package_raised",
        "shown_coverage_package_ran",
        "failed_refit_share",
    ]
    return rows.reindex(columns=columns).reset_index(drop=True)


def control_power(inputs: dict[str, float], replicates: int) -> float:
    """Clause 1's plug-in power at ``replicates``: the paired ratio's 99% upper end below 0.80."""
    from scipy import stats

    sd = inputs["ratio_sd"] * math.sqrt(inputs["pilot_draws"] / replicates)
    z = float(stats.norm.ppf(0.995))
    if sd <= 0:
        return float(inputs["ratio"] < 0.80)
    return float(stats.norm.cdf((0.80 - inputs["ratio"] - z * sd) / sd))


def predicted_reds(
    cells: dict[tuple[str, str, int], tuple[pd.Series, pd.Series]],
    sizing: floor.FamilySizing,
    *,
    anchor: int,
) -> list[tuple[str, str]]:
    """Each predicted red ``family/cell`` with the reading the declared route would give it.

    A positive cell whose plug-in G1 and G2 figure at the chosen R is below
    :data:`RED_PROBABILITY`, and a control cell whose clause-1 power is below
    :data:`CONTROL_POWER`. Below the anchor a red sits below the floor it forces
    (:func:`~tests.studies.few_cluster_floor.red_reading`, or the control reading); at or
    above it, it is a chance red that needs a recorded diagnosis.
    """
    out = []
    for (_, _, count), (positive, control) in sorted(cells.items()):
        inputs = floor.sizing_inputs(positive, control)
        replicates = sizing.replicates
        positive_pass = floor.g1_pass_probability(
            inputs["coverage"], replicates
        ) * floor.bias_pass_probability(inputs["standardized_bias"], replicates)
        if positive_pass < RED_PROBABILITY:
            reading = floor.CHANCE_READING if count >= anchor else floor.red_reading(positive)
            out.append((f"{floor.FAMILY}/{positive['cell']}", reading))
        if control_power(inputs, replicates) < CONTROL_POWER:
            reading = floor.CHANCE_READING if count >= anchor else floor.CONTROL_READING
            out.append((f"{floor.FAMILY}/{control['cell']}", reading))
    return out


def sizing_table(which: str, module: ModuleType, summary: pd.DataFrame) -> pd.DataFrame:
    """Per floor family the M5 sizing, then each predicted red and its reading."""
    sized = module.sizing(summary)
    sized = sized if isinstance(sized, dict) else {module.study.FIT: sized}
    out = []
    for family, (anchor, cells_of) in _families(which, module).items():
        result = sized[family]
        out.append(
            {
                "family": family,
                "kind": "family",
                "cell": "",
                "reading": "",
                "expected_floor": result.expected_floor,
                "replicates": result.replicates,
                "plugin": result.plugin,
                "predictive": result.predictive,
                "target_met": result.plugin >= floor.FAMILY_PASS_TARGET,
            }
        )
        for cell, reading in predicted_reds(cells_of(summary), result, anchor=anchor):
            out.append(
                {"family": family, "kind": "predicted red", "cell": cell, "reading": reading}
            )
    if which == "tmle":
        # The d5 configurations' reported floor reading (review M2): its cells are published
        # verdict rows, run at the stacked family's R, so their predicted reds are listed too.
        study = module.study
        result = sized["tmle_stacked"]
        out.append(
            {
                "family": study.D5_REPORTED,
                "kind": "reported",
                "cell": "",
                "reading": "",
                "replicates": result.replicates,
            }
        )
        cells = module.floor_cells(summary, study.D5_REPORTED)
        for cell, reading in predicted_reds(cells, result, anchor=study.ANCHORS["tmle_stacked"]):
            out.append(
                {
                    "family": study.D5_REPORTED,
                    "kind": "predicted red",
                    "cell": cell,
                    "reading": reading,
                }
            )
    return pd.DataFrame(out)


def candidate_table(which: str, module: ModuleType, rows: pd.DataFrame) -> pd.DataFrame:
    """Per part (c) family, (C)'s pilot discordance with (A) and its clause (i) probability."""
    study = module.study
    pairs = rows.loc[rows["property"] == floor.CANDIDATE_FAMILY]
    groups = dict(tuple(pairs.groupby("cell", sort=False)))
    by_family: dict[str, list[floor.PairedCounts]] = {}
    for name, frame in groups.items():
        fit, law, count, arm = name.split("__")
        if arm != floor.JACKKNIFE_ARM:
            continue
        control = groups.get("__".join((fit, law, count, floor.reference_for(arm))))
        if control is None:
            continue
        family = study.FITS[fit] if which == "tmle" else study.FIT
        draws = int(frame["requested_replicates"].iloc[0])
        by_family.setdefault(family, []).append(floor.paired_counts(frame, control, draws=draws))
    out = []
    for family, counts in sorted(by_family.items()):
        draws = sum(pair.draws for pair in counts)
        discordant = sum(pair.candidate_only + pair.reference_only for pair in counts)
        discordance = discordant / draws if draws else float("nan")
        probability = floor.admissibility_probability(discordance, len(counts))
        out.append(
            {
                "family": family,
                "cells": len(counts),
                "discordance": discordance,
                "admissibility_probability": probability,
                "shortfall": probability < JACKKNIFE_TARGET,
            }
        )
    return pd.DataFrame(out)


def timing_table(
    which: str, module: ModuleType, rows: pd.DataFrame, picked: Mapping[str, int]
) -> pd.DataFrame:
    """CPU seconds per key by fit and J, and the projection at the R the sizing picks.

    ``picked`` maps each floor family to the R of ``pilot-sizing.csv``. The table ends with two
    total rows: the projection at that R, which the 1.25 rule reads against the declared
    budget, and the projection with every family at :data:`~tests.studies.few_cluster_floor.R_CAP`.
    """
    study = module.study
    main = rows.loc[
        (rows["property"] == floor.FAMILY)
        & rows["cell"].astype(str).str.endswith(f"__{floor.REFERENCE_ARM}")
    ].copy()
    jackknife = rows.loc[
        (rows["property"] == floor.CANDIDATE_FAMILY)
        & rows["cell"].astype(str).str.endswith(f"__{floor.JACKKNIFE_ARM}")
    ].copy()
    for frame in (main, jackknife):
        parts = frame["cell"].astype(str).str.split("__", expand=True)
        frame["fit"], frame["clusters"] = parts[0], parts[2].str.removeprefix("j").astype(int)
    package = pd.to_numeric(main["package_glm_cpu_seconds"], errors="coerce").fillna(0.0)
    main["main_seconds"] = main["cpu_seconds"].astype(float) - package
    main["package_seconds"] = package
    table = main.groupby(["fit", "clusters"]).agg(
        main_seconds=("main_seconds", "mean"), package_seconds=("package_seconds", "mean")
    )
    table["jackknife_seconds"] = jackknife.groupby(["fit", "clusters"])["cpu_seconds"].mean()
    table = table.fillna(0.0).reset_index()
    laws = main.groupby(["fit", "clusters"])["cell"].nunique().rename("laws")
    table = table.merge(laws.reset_index(), on=["fit", "clusters"])

    def family(fit: str) -> str:
        return str(study.FITS[fit]) if which == "tmle" else str(study.FIT)

    def projected(replicates: Any) -> float:
        seconds = sum(
            row.laws
            * (
                replicates(row.fit) * (row.main_seconds + row.package_seconds)
                + min(replicates(row.fit), floor.CANDIDATE_DRAWS) * row.jackknife_seconds
            )
            for row in table.itertuples(index=False)
        )
        return float(seconds) / 3600.0

    limit = BUDGET_TOLERANCE * DECLARED_CPU_HOURS[which]
    totals = []
    for name, replicates in (
        ("total at the sized R", lambda fit: int(picked[family(fit)])),
        (f"total at R = {floor.R_CAP}", lambda fit: floor.R_CAP),
    ):
        hours = projected(replicates)
        totals.append(
            {
                "fit": name,
                "clusters": np.nan,
                "main_seconds": np.nan,
                "package_seconds": np.nan,
                "jackknife_seconds": np.nan,
                "laws": np.nan,
                "projected_cpu_hours": hours,
                "declared_cpu_hours": DECLARED_CPU_HOURS[which],
                "within_tolerance": hours <= limit,
            }
        )
    table.attrs["projected_cpu_hours"] = totals[0]["projected_cpu_hours"]
    return pd.concat([table, pd.DataFrame(totals)], ignore_index=True)


def run(which: str, output: Path | None = None, jobs: int = 1) -> dict[str, pd.DataFrame]:
    """Fit the pilot of one study and write every table the module docstring lists."""
    module = importlib.import_module(MODULES[which])
    with recording("cleverly", witness=module.PROPERTY_WITNESS) as recorded:
        rows = module.generate_property_rows(
            n_jobs=jobs, budget=PILOT_DRAWS, stream="pilot", timed=True
        )
    records = property_failures.frame(recorded)
    stamped = property_failures.stamp(rows, records)
    summary = module.summarize_properties(stamped.drop(columns=list(floor.TIMING_COLUMNS)))
    output = module.STUDY.artifacts if output is None else output
    output.mkdir(parents=True, exist_ok=True)
    failures = failure_table(records)
    sized = sizing_table(which, module, summary)
    family_rows = sized.loc[sized["kind"] == "family"]
    picked = dict(zip(family_rows["family"], family_rows["replicates"].astype(int), strict=True))
    tables = {
        "pilot-failures.csv": failures,
        "pilot-summary.csv": summary,
        "pilot-cells.csv": cell_table(summary, failures),
        "pilot-sizing.csv": sized,
        "pilot-candidates.csv": candidate_table(which, module, stamped),
        "pilot-timing.csv": timing_table(which, module, stamped, picked),
    }
    stamped.drop(columns=list(floor.TIMING_COLUMNS)).to_csv(output / "pilot.csv.gz", index=False)
    for name, table in tables.items():
        table.to_csv(output / name, index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(tables["pilot-sizing.csv"].to_string())
        print(tables["pilot-candidates.csv"].to_string())
        print(tables["pilot-timing.csv"].tail(2).to_string())
    return tables


def main() -> None:
    """The command line of :func:`run`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=sorted(MODULES))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    run(args.study, args.output, args.jobs)


if __name__ == "__main__":
    main()
