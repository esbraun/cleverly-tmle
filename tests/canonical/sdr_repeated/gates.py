"""The registration gates of ``longitudinal-repeated-sdr`` (X32b plan, sections 5.3 and 10).

**P0.**  Five replicates at the declared ``n``: the lmtp wall time on the owner's machine, which
sets the parallel plan of the R phase.  ``--timing-only`` writes ``timing.csv`` and states no
verdict.

**G1r.**  Twenty replicates of each scenario on the declared seeds.  In every cell
``(scenario, estimand)``, ``|psi_cleverly - psi_lmtp|`` and ``max |IC_cleverly - IC_lmtp|`` are
at most :data:`G1R_GATE`.  The runner writes lmtp's unit-level influence curves when
``CLEVERLY_IC_DIR`` is set, and a declared run leaves it unset.  The run writes ``gates.csv``
(one row per replicate and estimand) and exits non-zero when a cell fails.  If G1r fails, PR C
diagnoses the pairing and re-declares it on fresh seeds for both sides.

**Parity ratio.**  On the committed rows (:func:`parity_table`), per cell, the largest
``|cleverly - lmtp|`` over the replicates is at most :data:`PARITY_SHARE` of the RMS of lmtp's
correction, ``estimate - initial_estimate``.  **N8** (:func:`mutate_correction` and
:func:`mutate_plug_in`) reports the first-node plug-in, or a correction shortened by 10%, on
the cleverly side.  Both mutations must fail the parity ratio in every cell.

**PR B rows.**  :func:`pr_b_differences` compares the Python rows of the queued run with the rows
PR B committed (``pr-b-replicates.parquet``), to :data:`PR_B_GATE`.  The queued run draws the same
seeds, so a difference means the Python side changed after the merged verdicts.

Everything but :func:`main` runs without Docker.  :func:`main` needs Docker and the
``cleverly-lmtp-crossfit:1.5.4`` image (``tests/canonical/lmtp_crossfit``).

**Outputs.**  ``gates.csv`` (G1r only) and ``timing.csv`` land in this directory.  They are
study files, so the queue commits them after step 2 and before the regeneration, which then
records a clean worktree.  The run builds the image first (the driver's own ``docker build``
command), and refuses before that if the Python rows differ from PR B's.

Usage::

    python -m tests.canonical.sdr_repeated.gates [--replicates 20] [--timing-only] [--cores 1]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
import warnings
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from tests.canonical.sdr_repeated import regenerate
from tests.studies import canonical_sdr_repeated as study
from tests.studies.evidence.registry import ROOT

HERE = Path(__file__).resolve().parent
IMAGE = "cleverly-lmtp-crossfit:1.5.4"
RUNNER = "sdr_repeated/run_study.R"

#: Plan X32b section 5.3: G1r reads 20 replicates of each scenario.
G1R_REPLICATES = 20
#: The G1r bound on the estimate and on the largest influence-curve difference.
G1R_GATE = 1e-10
#: The parity ratio: the largest paired gap, as a share of the RMS of lmtp's correction.
PARITY_SHARE = 0.01
#: The bound on the Python rows of the queued run against PR B's.
PR_B_GATE = 1e-10
#: The rows PR B committed, and the digest that pins them.
PR_B_ROWS = HERE / "pr-b-replicates.parquet"
PR_B_ROWS_SHA256 = "02a0f0654c7e62b4a2920358ce189da4533a6f3db66ab212ec90e8e81e1c8ff1"
#: The shortening of the correction that N8 applies.
N8_SHRINK = 0.9

CELL = ["scenario", "estimand"]
PAIR = ["scenario", "replicate", "estimand"]


# ---------------------------------------------------------------------------- the parity ratio


def _paired(rows: pd.DataFrame) -> pd.DataFrame:
    """Each (scenario, replicate, estimand) with both implementations' estimate and plug-in."""
    subject = str(study.STUDY.implementation)
    reference = str(study.STUDY.reference)
    wide = rows.pivot_table(
        index=PAIR,
        columns="implementation",
        values=["estimate", "initial_estimate"],
        aggfunc="first",
    )
    missing = {subject, reference} - set(wide["estimate"].columns)
    if missing:
        raise ValueError(f"the rows hold no {sorted(missing)} rows to pair")
    paired = pd.DataFrame(
        {
            "cleverly": wide["estimate"][subject],
            "lmtp": wide["estimate"][reference],
            "lmtp_initial": wide["initial_estimate"][reference],
        }
    ).dropna()
    if paired.empty:
        raise ValueError("no replicate has both implementations")
    return paired.reset_index()


def parity_table(rows: pd.DataFrame, *, share: float = PARITY_SHARE) -> pd.DataFrame:
    """The parity ratio per cell: the largest paired gap over the RMS of lmtp's correction.

    Parameters
    ----------
    rows : pandas.DataFrame
        Replicate rows of both implementations, with ``initial_estimate``.
    share : float
        The share of the RMS correction that the gap may reach.

    Returns
    -------
    pandas.DataFrame
        One row per cell with ``max_gap``, ``rms_correction``, ``ratio`` and ``passed``.
    """
    paired = _paired(rows)
    paired["gap"] = (paired["cleverly"] - paired["lmtp"]).abs()
    paired["correction"] = paired["lmtp"] - paired["lmtp_initial"]
    grouped = paired.groupby(CELL)
    table = pd.DataFrame(
        {
            "replicates": grouped.size(),
            "max_gap": grouped["gap"].max(),
            "rms_correction": grouped["correction"].apply(lambda c: float(np.sqrt(np.mean(c**2)))),
        }
    ).reset_index()
    # A cell whose lmtp correction vanishes cannot be read: the ratio is infinite, not passed.
    with np.errstate(divide="ignore", invalid="ignore"):
        table["ratio"] = table["max_gap"] / table["rms_correction"]
    table["passed"] = (table["max_gap"] <= share * table["rms_correction"]) & (
        table["rms_correction"] > 0.0
    )
    return table


def mutate_correction(rows: pd.DataFrame, factor: float = N8_SHRINK) -> pd.DataFrame:
    """N8: the cleverly estimate keeps ``factor`` of its correction from the first-node plug-in."""
    out = rows.copy()
    mine = out["implementation"] == study.STUDY.implementation
    out.loc[mine, "estimate"] = out.loc[mine, "initial_estimate"] + factor * (
        out.loc[mine, "estimate"] - out.loc[mine, "initial_estimate"]
    )
    return out


def mutate_plug_in(rows: pd.DataFrame) -> pd.DataFrame:
    """N8: the cleverly estimate is its first-node plug-in."""
    return mutate_correction(rows, 0.0)


# ---------------------------------------------------------------------------- PR B's rows


def pr_b_rows() -> pd.DataFrame:
    """The Python rows PR B committed, after the digest check."""
    import hashlib

    digest = hashlib.sha256(PR_B_ROWS.read_bytes()).hexdigest()
    if digest != PR_B_ROWS_SHA256:
        raise AssertionError(f"{PR_B_ROWS.name} is not the file PR B committed ({digest})")
    from tests.studies.evidence.manifest import read_rows

    return read_rows(PR_B_ROWS)


def pr_b_differences(
    current: pd.DataFrame, baseline: pd.DataFrame, *, gate: float = PR_B_GATE
) -> pd.DataFrame:
    """The rows of ``current`` that differ from ``baseline`` by more than ``gate``.

    Both frames are cut to the Python implementation's rows.  A row present on one side only is
    a difference.  The columns compared are every numeric column of the published row.

    Returns
    -------
    pandas.DataFrame
        One row per difference, with the column and the absolute gap (``inf`` for a missing
        row); empty when the Python rows agree.
    """
    subject = study.STUDY.implementation
    keys = [*PAIR, "implementation"]
    left = current.loc[current["implementation"] == subject].set_index(keys).sort_index()
    right = baseline.loc[baseline["implementation"] == subject].set_index(keys).sort_index()
    out: list[dict[str, object]] = []
    for key in left.index.symmetric_difference(right.index):
        out.append({"key": key, "column": "(row)", "gap": float("inf")})
    both = left.index.intersection(right.index)
    numeric = [
        column
        for column in right.columns
        if column not in {"inference_scale"} and pd.api.types.is_numeric_dtype(right[column])
    ]
    for column in numeric:
        gap = (left.loc[both, column] - right.loc[both, column]).abs()
        gap = gap.where(~(left.loc[both, column].isna() & right.loc[both, column].isna()), 0.0)
        for key in gap.index[~(gap <= gate)]:
            out.append({"key": key, "column": column, "gap": float(gap[key])})
    return pd.DataFrame(out, columns=["key", "column", "gap"])


# ---------------------------------------------------------------------------- G1r


def g1r_table(
    cleverly: pd.DataFrame,
    reference: pd.DataFrame,
    curves: Mapping[tuple[str, int, str], tuple[np.ndarray, np.ndarray]],
    *,
    gate: float = G1R_GATE,
) -> pd.DataFrame:
    """G1r per replicate and estimand: the estimate gap and the largest influence-curve gap.

    Parameters
    ----------
    cleverly, reference : pandas.DataFrame
        The two sides' rows for the G1r replicates.
    curves : mapping
        ``(scenario, replicate, estimand)`` to the pair ``(cleverly IC, lmtp IC)`` of unit-level
        curves in row order.  Each curve is centred before the comparison.
    gate : float
        The bound on both gaps.
    """
    merged = cleverly.merge(reference, on=PAIR, suffixes=("_cleverly", "_lmtp"))
    if len(merged) != len(cleverly) or len(merged) != len(reference):
        raise AssertionError(
            f"G1r pairs {len(merged)} rows of {len(cleverly)} cleverly and {len(reference)} lmtp"
        )
    merged["estimate_gap"] = (merged["estimate_cleverly"] - merged["estimate_lmtp"]).abs()
    gaps = []
    for scenario, replicate, estimand in zip(
        merged["scenario"], merged["replicate"], merged["estimand"], strict=True
    ):
        mine, theirs = curves[(scenario, int(replicate), estimand)]
        if mine.shape != theirs.shape:
            raise AssertionError(f"{scenario}/{replicate}/{estimand}: the curves differ in length")
        # lmtp's eif has mean equal to the estimate and cleverly's curve has mean zero, so both
        # are centred; the same rule serves a contrast, whose curve is a difference of two.
        gaps.append(float(np.max(np.abs((mine - mine.mean()) - (theirs - theirs.mean())))))
    merged["ic_gap"] = gaps
    merged["passes"] = (merged["estimate_gap"] <= gate) & (merged["ic_gap"] <= gate)
    return merged.loc[:, [*PAIR, "estimate_gap", "ic_gap", "passes"]]


def g1r_cells(table: pd.DataFrame) -> pd.DataFrame:
    """The largest gaps per cell, and whether every replicate of the cell passes."""
    grouped = table.groupby(CELL)
    return pd.DataFrame(
        {
            "max_estimate_gap": grouped["estimate_gap"].max(),
            "max_ic_gap": grouped["ic_gap"].max(),
            "passes": grouped["passes"].all(),
        }
    ).reset_index()


# ---------------------------------------------------------------------------- the run


def _collect(
    replicates: int, n: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, list]:
    samples, truths, rows, curves, timing = [], [], [], {}, []
    for replicate in range(replicates):
        for scenario in study.STUDY.scenarios:
            start = time.perf_counter()
            frame, truth = study.draw_scenario(scenario, n, replicate)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = study.fit_cleverly(frame, folds=study.SCENARIO_FOLDS[scenario])
            samples.append(study.pairing_sample(frame, result, scenario, replicate))
            truths.extend(
                {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": value}
                for name, value in truth.items()
            )
            rows.extend(study.cleverly_rows(result, truth, scenario, replicate, n=len(frame)))
            for name in study.ESTIMANDS:
                curves[(scenario, replicate, name)] = np.asarray(
                    result[name].influence_curve, dtype=float
                )
            timing.append(
                {"side": "cleverly", "scenario": scenario, "seconds": time.perf_counter() - start}
            )
    return (
        pd.concat(samples, ignore_index=True),
        pd.DataFrame(truths),
        pd.DataFrame(rows),
        curves,
        timing,
    )


def _ensure_image() -> None:
    """Build the image as the regeneration driver does (``Reference.run``): same command."""
    reference = regenerate.REFERENCE
    context = reference.build_context or HERE
    subprocess.run(["docker", "build", "-t", reference.image, str(context)], check=True)


def _run_reference(work: Path, cores: int) -> float:
    start = time.perf_counter()
    subprocess.run(
        [
            "docker", "run", "--rm",
            "-e", f"CLEVERLY_R_CORES={cores}",
            "-e", "CLEVERLY_IC_DIR=/work/ic",
            "-v", f"{work}:/work",
            "-v", f"{ROOT / 'tests' / 'canonical'}:/fixture:ro",
            "--entrypoint", "Rscript", IMAGE,
            f"/fixture/{RUNNER}",
            "/work/samples.csv.gz", "/work/truth.csv", "/work/lmtp.csv",
        ],
        check=True,
    )  # fmt: skip
    return time.perf_counter() - start


def _read_curves(work: Path, rows: pd.DataFrame) -> dict[tuple[str, int, str], np.ndarray]:
    out: dict[tuple[str, int, str], np.ndarray] = {}
    for scenario, replicate in (
        rows[["scenario", "replicate"]].drop_duplicates().itertuples(index=False)
    ):
        frame = pd.read_csv(
            work / "ic" / f"ic-{scenario}-{int(replicate)}.csv.gz", float_precision="round_trip"
        ).sort_values("row")
        for name in study.ESTIMANDS:
            out[(scenario, int(replicate), name)] = frame[name].to_numpy(dtype=float)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replicates", type=int, default=G1R_REPLICATES)
    parser.add_argument("--n", type=int, default=study.PRIMARY_N)
    parser.add_argument("--cores", type=int, default=1, help="R workers; 1 times a single process")
    parser.add_argument("--timing-only", action="store_true", help="P0: write timing.csv only")
    arguments = parser.parse_args(argv)
    samples, truths, rows, mine, timing = _collect(arguments.replicates, arguments.n)
    if arguments.n == study.PRIMARY_N:
        # The G1r refits are PR B's draws, so a Python drift on this host shows before Docker.
        baseline = pr_b_rows()
        baseline = baseline.loc[baseline["replicate"] < arguments.replicates]
        drift = pr_b_differences(rows, baseline)
        if len(drift):
            print(f"the Python rows differ from PR B's:\n{drift.head(20)}", file=sys.stderr)
            return 1
    _ensure_image()
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        samples.to_csv(
            work / "samples.csv.gz", index=False, compression="gzip", lineterminator="\n"
        )
        truths.to_csv(work / "truth.csv", index=False, lineterminator="\n")
        seconds = _run_reference(work, arguments.cores)
        reference = pd.read_csv(work / "lmtp.csv", float_precision="round_trip")
        theirs = _read_curves(work, rows)
    timing.append({"side": "lmtp", "scenario": "all", "seconds": seconds})
    pd.DataFrame(timing).groupby(["side", "scenario"], as_index=False)["seconds"].mean().to_csv(
        HERE / "timing.csv", index=False, lineterminator="\n", float_format="%.3f"
    )
    print(f"lmtp: {seconds:.0f} s for {arguments.replicates} replicates on {arguments.cores} cores")
    if arguments.timing_only:
        return 0
    table = g1r_table(rows, reference, {key: (mine[key], theirs[key]) for key in mine})
    table.to_csv(HERE / "gates.csv", index=False, lineterminator="\n", float_format="%.17g")
    cells = g1r_cells(table)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(cells)
    failed = cells[~cells["passes"]]
    if len(failed):
        print(f"{len(failed)} G1r cells fail:\n{failed.to_string()}", file=sys.stderr)
        return 1
    print("every G1r cell passes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
