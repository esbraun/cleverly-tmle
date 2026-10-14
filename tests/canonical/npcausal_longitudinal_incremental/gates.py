"""The pre-declaration pairing gates of ``longitudinal-incremental`` (``X19-plan.md`` 3.4 and 6.1).

**G1.**  Three replicates of the primary scenario at the declared ``n`` (five folds),
plus one forced replicate per fold count: a small sample of law ``A`` whose cross-fitted
treatment fit meets a ``(node, history)`` cell that a training fold lacks
(:data:`FORCED_N`, :data:`FORCED_SEED`).  That replicate exercises the fallback both sides
take, the training mean of the same node.  **G2.**  The same comparison on law ``Ax``, whose
propensities are exactly 0 or 1 in some cells, at two and five folds on three draws of the
gate stream.

Every gated quantity must agree with ``npcausal`` below
:data:`~tests.studies.canonical_longitudinal_incremental.PAIRING_GATE`: each plan's estimate,
every unit's centred influence value, and the standard error.  ``npcausal`` reports
``sd(ifvals)`` and this package reports the same standard deviation over :math:`\\sqrt n`, so
no factor separates them.  ``calpha``, the ``npcausal`` uniform-band critical value over the
three means, is printed beside the critical value of this package's band, which covers the
means and the two contrasts.  It is not gated, since both draw multipliers at random.

The run also times each scenario single-process on both sides, for the declared budget.  It
writes ``gates.csv`` (one row per paired plan) and ``timing.csv`` beside this file, and exits
non-zero when a gate fails.  It needs Docker and the ``cleverly-npcausal:0.1.0-56a5ac1`` image,
so it runs on the local machine.

Usage::

    python -m tests.canonical.npcausal_longitudinal_incremental.gates [--replicates 3]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tests import discrete_law_longitudinal_incremental as law
from tests.studies import canonical_longitudinal_incremental as study
from tests.studies import longitudinal_incremental_common as common
from tests.studies.evidence.registry import ROOT
from tests.studies.evidence.seeds import stream_seed

HERE = Path(__file__).resolve().parent
IMAGE = "cleverly-npcausal:0.1.0-56a5ac1"
#: The forced replicate's size and seed: at 120 rows every fold of either fold count leaves a
#: treatment cell of some held-out row without a training row (asserted by :func:`forced`).
FORCED_N = 120
FORCED_SEED = 1000
AX_PLANS = common.constant_plans(law.DELTAS, n_times=2)
DELTA_GRID = ",".join(f"{delta:g}" for delta in law.DELTAS.values())


def _paired(frame: pd.DataFrame, result: Any, scenario: str, replicate: int) -> pd.DataFrame:
    sample = frame.loc[:, [c for c in study.SAMPLE_COLUMNS if c in frame]].copy()
    sample.insert(0, "row", np.arange(len(sample)))
    sample.insert(0, "fold", np.asarray(result.folds.assignment, dtype=int))
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    return sample


def empty_training_cell(frame: pd.DataFrame, assignment: np.ndarray, n_times: int) -> bool:
    """Whether some fold's training rows lack a ``(node, history)`` cell its held-out rows hold."""
    for t in range(1, n_times + 1):
        columns = ["W"] + [name for s in range(2, t + 1) for name in (f"A{s - 1}", f"L{s}")]
        keys = frame[columns].astype(int).astype(str).agg("|".join, axis=1).to_numpy()
        for fold in np.unique(assignment):
            held = assignment == fold
            if set(keys[held]) - set(keys[~held]):
                return True
    return False


def forced(n_folds: int) -> tuple[pd.DataFrame, Any]:
    frame = law.sample(law.LAW_A, FORCED_N, FORCED_SEED)
    result = common.fit(frame, law.LAW_A, study.PLANS, n_folds=n_folds, reference=study.REFERENCE)
    if not empty_training_cell(frame, np.asarray(result.folds.assignment), law.LAW_A.n_times):
        raise AssertionError(f"the forced replicate at {n_folds} folds meets no empty cell")
    return frame, result


def collect(replicates: int) -> tuple[pd.DataFrame, dict[tuple[str, int], Any], list]:
    samples, fits, timing = [], {}, []
    for replicate in range(replicates):
        for scenario in study.FOLDS:
            start = time.perf_counter()
            frame, _ = study.draw_scenario(scenario, study.PRIMARY_N, replicate)
            result = study.fit(frame, scenario)
            timing.append(
                {"side": "cleverly", "scenario": scenario, "seconds": time.perf_counter() - start}
            )
            samples.append(_paired(frame, result, scenario, replicate))
            fits[(scenario, replicate)] = result
        for n_folds in (2, 5):
            scenario = f"gate_ax_folds_{n_folds}"
            seed = stream_seed(study.STUDY, "pairing_gate", scenario, replicate)
            frame = law.sample(law.LAW_AX, study.PRIMARY_N, seed)
            result = common.fit(
                frame, law.LAW_AX, AX_PLANS, n_folds=n_folds, reference=study.REFERENCE
            )
            samples.append(_paired(frame, result, scenario, replicate))
            fits[(scenario, replicate)] = result
    for n_folds in (2, 5):
        scenario = f"forced_empty_folds_{n_folds}"
        frame, result = forced(n_folds)
        samples.append(_paired(frame, result, scenario, 0))
        fits[(scenario, 0)] = result
    return pd.concat(samples, ignore_index=True), fits, timing


def run_reference(work: Path) -> float:
    start = time.perf_counter()
    subprocess.run(
        [
            "docker", "run", "--rm",
            "-v", f"{work}:/work",
            "-v", f"{ROOT / 'tests' / 'canonical'}:/fixture:ro",
            "--entrypoint", "Rscript", IMAGE,
            "/fixture/npcausal_longitudinal_incremental/gates.R",
            "/work/samples.csv.gz", DELTA_GRID, "/work/estimates.csv", "/work/curves.csv.gz",
        ],
        check=True,
    )  # fmt: skip
    return time.perf_counter() - start


def compare(
    fits: dict[tuple[str, int], Any], estimates: pd.DataFrame, curves: pd.DataFrame
) -> pd.DataFrame:
    labels = {float(delta): label for label, delta in law.DELTAS.items()}
    rows = []
    for (scenario, replicate), result in fits.items():
        mine = estimates[
            (estimates["scenario"] == scenario) & (estimates["replicate"] == replicate)
        ]
        theirs = curves[(curves["scenario"] == scenario) & (curves["replicate"] == replicate)]
        if len(mine) != len(labels):
            raise AssertionError(f"{scenario} replicate {replicate}: npcausal rows are missing")
        n = int(mine["n"].iloc[0])
        for _, row in mine.iterrows():
            label = labels[float(row["delta"])]
            name = f"ey_regimen[{label}]"
            curve = theirs[theirs["delta"] == row["delta"]].sort_values("row")["ic"].to_numpy()
            estimate = result[name]
            gate = "G2" if scenario.startswith("gate_ax") else "G1"
            gaps = {
                "estimate_gap": abs(float(estimate.psi) - float(row["estimate"])),
                "curve_gap": float(np.max(np.abs(result.influence_curves[name] - curve))),
                "std_error_gap": abs(float(estimate.std_error) - float(row["sigma"]) / np.sqrt(n)),
            }
            rows.append(
                {
                    "gate": gate,
                    "scenario": scenario,
                    "replicate": replicate,
                    "estimand": name,
                    "estimate_cleverly": float(estimate.psi),
                    "estimate_npcausal": float(row["estimate"]),
                    **gaps,
                    "calpha_npcausal": float(row["calpha"]),
                    "passes": max(gaps.values()) < study.PAIRING_GATE,
                }
            )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replicates", type=int, default=3)
    arguments = parser.parse_args(argv)
    samples, fits, timing = collect(arguments.replicates)
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        samples.to_csv(
            work / "samples.csv.gz", index=False, compression="gzip", lineterminator="\n"
        )
        seconds = run_reference(work)
        estimates = pd.read_csv(work / "estimates.csv", float_precision="round_trip")
        curves = pd.read_csv(work / "curves.csv.gz", float_precision="round_trip")
    timing.append({"side": "npcausal", "scenario": "all", "seconds": seconds})
    table = compare(fits, estimates, curves)
    table.to_csv(HERE / "gates.csv", index=False, lineterminator="\n", float_format="%.17g")
    pd.DataFrame(timing).groupby(["side", "scenario"], as_index=False)["seconds"].mean().to_csv(
        HERE / "timing.csv", index=False, lineterminator="\n", float_format="%.3f"
    )
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(
            table.groupby(["gate", "scenario"])[
                ["estimate_gap", "curve_gap", "std_error_gap"]
            ].max()
        )
    failed = table[~table["passes"]]
    if len(failed):
        print(f"{len(failed)} gated rows fail:\n{failed.to_string()}", file=sys.stderr)
        return 1
    print("every G1 and G2 row passes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
