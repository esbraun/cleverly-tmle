"""Seed the declared run's scratch with the committed rows of every unchanged fit set.

The X79 rerun changes one family.  ``repeat_stability`` renames its base arm
``one_repeat_control`` to ``one_repeat`` (a positive row that shares the pair's verdict) and adds
the ``second_draw_control`` arm, which reads draw 1 of each three-draw fit.  Every trial of the
two stability laws (``stability_d`` and ``stability_i`` of
:data:`tests.studies.longitudinal_repeated_crossfit_properties.FIT_SETS`) is therefore fitted
again, and every other fit set is unchanged.  An unchanged fit set draws the same samples, fits
them with the same code, and writes the same rows.  This entry checkpoints those rows from the
committed ``property-replicates.csv.gz`` as if the declared run had fitted them, so the declared
run fits only the two stability laws and the primary phase.

    python -m tests.canonical.longitudinal_repeated_crossfit.inherit
    python -m tests.canonical.longitudinal_repeated_crossfit.inherit --check <baseline commit>

``--check`` runs after the declared run.  It compares the committed files with those of the
baseline commit and exits non-zero on a difference.  Every property row outside
``repeat_stability`` must be unchanged, bit for bit.  The declared run refits the primary phase
and the stability trials, so the primary rows and the two stability arms that the rename keeps
(``three_repeats`` and ``one_repeat``, read in the baseline as ``one_repeat_control``) must agree
to ``PRIMARY_RTOL`` and ``PRIMARY_ATOL``, and their published ratios, intervals and verdicts must
agree to the same tolerance.  Only ``second_draw_control`` is new and is not compared.  The
summary rows outside ``repeat_stability`` must agree to the same tolerance, because the same
rows give the same summary up to the last bits of a platform's reductions.

The entry refuses what a declared run refuses: a dirty tree, ``HEAD`` not equal to its
upstream, or a thread variable other than 1.  It refuses a committed file that its manifest
does not describe, a fit set with a missing or short replicate, and a committed row that
records a lost replicate.  Run it from the declared commit and under the declared run's
``LOCALAPPDATA``, as :mod:`tests.canonical.longitudinal_repeated_crossfit.part` does.
``tests/unit/test_longitudinal_repeated_crossfit_study.py`` checks that a declared run reuses
what this entry writes.  ``tests/canonical/provenance-revisions.md`` records why the rows are
sound under the new code.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import pickle
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from tests.canonical.declared_run import thread_refusals
from tests.canonical.longitudinal_repeated_crossfit.part import scratch
from tests.canonical.regenerate import CHECKPOINTS, _claim_scratch, host_lock
from tests.diagnostics.rm18_shared import refusals
from tests.studies import longitudinal_repeated_crossfit_properties as properties
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT

HERE = Path(__file__).resolve().parent
ROWS = "property-replicates.csv.gz"
PRIMARY_ROWS = "replicates.csv.gz"
SUMMARY = "properties.csv"

#: The family the rerun fits again.  Every other property row is inherited.
REFITTED = ("repeat_stability",)

#: The baseline's name of the base arm that the rename keeps, and its new name.
RENAMED_ARM = ("__one_repeat_control", "__one_repeat")
#: The arm that the rerun adds, which the baseline has no row for.
NEW_ARM = "__second_draw_control"

#: The tolerance on the refitted primary and stability rows.  The 38ad7d05 run moved the
#: primary rows by at most 1.8e-15.
PRIMARY_RTOL = 1e-12
PRIMARY_ATOL = 1e-14

#: The columns of a stability summary row that the rename keeps comparable.
STABILITY_SUMMARY = (
    "n",
    "replicates",
    "failed_replicates",
    "spread_ratio",
    "spread_ratio_ci_lower",
    "spread_ratio_ci_upper",
    "spread_ratio_boundary",
    "one_draw_spread_share",
    "passed",
)

#: The fit sets that the rerun fits: the two stability laws.
FITTED = tuple(label for label in properties.plan() if label.startswith("stability_"))

#: The fit sets and stability laws whose rows the committed file supplies.
INHERITED = tuple(label for label in properties.plan() if label not in FITTED)


def committed_rows(here: Path = HERE) -> pd.DataFrame:
    """The committed property rows, checked against the hash that the committed manifest records."""
    manifest = json.loads((here / "manifest.json").read_text(encoding="utf-8"))
    recorded = manifest["sha256"][ROWS]
    if hashlib.sha256((here / ROWS).read_bytes()).hexdigest() != recorded:
        raise RuntimeError(f"refused: {ROWS} no longer matches its manifest hash")
    print(f"source {ROWS} sha256 {recorded}", flush=True)
    rows = pd.read_csv(here / ROWS, float_precision="round_trip")
    if rows["failed_replicates"].ne(0).any():
        raise RuntimeError("refused: a committed row records a lost replicate")
    return rows


def write_checkpoint(directory: Path, ident: str, rows: list[dict[str, object]]) -> None:
    """One finished key, in the format ``fit_replicates`` reads: ``(rows, no failure record)``."""
    temporary = directory / f"{ident}.tmp"
    with gzip.open(temporary, "wb", compresslevel=1) as handle:
        pickle.dump((rows, None), handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, directory / f"{ident}.pkl.gz")


def inherit_part(label: str, committed: pd.DataFrame, root: Path) -> int:
    """Checkpoint every key of ``label`` from ``committed`` under ``root``; return the key count.

    A key's rows are the committed rows of its replicate in the cells it declares, without the
    derived controls (the declared run derives those from the base rows).
    """
    keys = properties.plan()[label]
    controls = {name for names in properties.DERIVED_CELLS.values() for name in names}
    names = [name for name in properties.cells_of(keys[0]) if name not in controls]
    requested = keys[0][2]
    columns: dict[str, list[dict[str, object]]] = {}
    for name in names:
        family, cell = name.split("/", 1)
        frame = committed.loc[(committed["property"] == family) & (committed["cell"] == cell)]
        frame = frame.sort_values("replicate", ignore_index=True)
        if (
            len(frame) != requested
            or frame["replicate"].tolist() != list(range(requested))
            or set(frame["requested_replicates"]) != {requested}
        ):
            raise RuntimeError(f"refused: {name} does not hold replicates 0..{requested - 1}")
        columns[name] = frame.to_dict("records")
    directory = root / properties.STUDY.implementation / f"properties/{label}_r{requested}"
    directory.mkdir(parents=True, exist_ok=True)
    for key in keys:
        ident = replication._digest(
            (properties.STUDY.implementation, f"properties/{label}_r{requested}", key[0], key[1])
        )
        write_checkpoint(directory, ident, [columns[name][key[1]] for name in names])
    return len(keys)


def baseline(commit: str, name: str) -> bytes:
    """The bytes of a study file at ``commit``."""
    path = (HERE / name).relative_to(ROOT).as_posix()
    return subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout


def _baseline_frame(commit: str, name: str) -> pd.DataFrame:
    """A study file of ``commit`` as a frame, parsed so floats round-trip."""
    compression = "gzip" if name.endswith(".gz") else None
    return pd.read_csv(
        io.BytesIO(baseline(commit, name)), compression=compression, float_precision="round_trip"
    )


def _stability_arms(frame: pd.DataFrame, *, renamed: bool) -> pd.DataFrame:
    """The ``repeat_stability`` rows of the arms both runs hold, under the new arm names.

    ``renamed`` maps the baseline's ``one_repeat_control`` to ``one_repeat``, and its role
    from ``control`` to ``positive``.  The new ``second_draw_control`` rows are dropped.
    """
    out = frame.loc[frame["property"] == "repeat_stability"].copy()
    if renamed:
        arm = out["cell"].str.endswith(RENAMED_ARM[0])
        out["cell"] = out["cell"].str.replace(*RENAMED_ARM, regex=False)
        out.loc[arm, "role"] = "positive"
    return out.loc[~out["cell"].str.endswith(NEW_ARM)]


def check(commit: str) -> list[str]:
    """What differs from ``commit`` outside the family that the rerun fits again.

    The kept stability arms are compared at the primary tolerance, with identical verdicts.
    """
    differences: list[str] = []
    keys = ["implementation", "scenario", "replicate", "estimand"]
    primary_old = _baseline_frame(commit, PRIMARY_ROWS).sort_values(keys, ignore_index=True)
    primary_new = pd.read_csv(HERE / PRIMARY_ROWS, float_precision="round_trip").sort_values(
        keys, ignore_index=True
    )
    try:
        # The declared run refits the primary phase, so its floats can move in the last bits.
        pd.testing.assert_frame_equal(
            primary_old, primary_new, check_exact=False, rtol=PRIMARY_RTOL, atol=PRIMARY_ATOL
        )
    except AssertionError as error:
        differences.append(f"{PRIMARY_ROWS} differs from {commit}: {error}")
    old = _baseline_frame(commit, ROWS)
    new = pd.read_csv(HERE / ROWS, float_precision="round_trip")
    order = ["property", "cell", "replicate"]
    old_rest = old.loc[~old["property"].isin(REFITTED)].sort_values(order, ignore_index=True)
    new_rest = new.loc[~new["property"].isin(REFITTED)].sort_values(order, ignore_index=True)
    try:
        pd.testing.assert_frame_equal(old_rest, new_rest, check_exact=True)
    except AssertionError as error:
        differences.append(f"{ROWS} differs from {commit} outside {REFITTED}: {error}")
    old_arms = _stability_arms(old, renamed=True).sort_values(order, ignore_index=True)
    new_arms = _stability_arms(new, renamed=False).sort_values(order, ignore_index=True)
    try:
        pd.testing.assert_frame_equal(
            old_arms, new_arms, check_exact=False, rtol=PRIMARY_RTOL, atol=PRIMARY_ATOL
        )
    except AssertionError as error:
        differences.append(f"{ROWS} stability arms differ from {commit}: {error}")
    differences.extend(_summary_differences(commit))
    return differences


def _summary_differences(commit: str) -> list[str]:
    """The ``properties.csv`` differences: other families, then the kept stability arms."""
    old = _baseline_frame(commit, SUMMARY)
    new = pd.read_csv(HERE / SUMMARY, float_precision="round_trip")
    order = ["property", "cell"]
    out: list[str] = []
    rest_old = old.loc[~old["property"].isin(REFITTED)].sort_values(order, ignore_index=True)
    rest_new = new.loc[~new["property"].isin(REFITTED)].sort_values(order, ignore_index=True)
    try:
        pd.testing.assert_frame_equal(
            rest_old, rest_new, check_exact=False, rtol=PRIMARY_RTOL, atol=PRIMARY_ATOL
        )
    except AssertionError as error:
        out.append(f"{SUMMARY} differs from {commit} outside {REFITTED}: {error}")
    columns = ["property", "cell", *STABILITY_SUMMARY]
    kept_old = _stability_arms(old, renamed=True).sort_values(order, ignore_index=True)
    kept_new = _stability_arms(new, renamed=False).sort_values(order, ignore_index=True)
    try:
        pd.testing.assert_frame_equal(
            kept_old.loc[:, columns],
            kept_new.loc[:, columns],
            check_exact=False,
            rtol=PRIMARY_RTOL,
            atol=PRIMARY_ATOL,
        )
    except AssertionError as error:
        out.append(f"{SUMMARY} stability arms differ from {commit}: {error}")
    return out


def main(arguments: Sequence[str] | None = None) -> None:
    """Apply the declared run's refusals and checkpoint every inherited fit set."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", metavar="COMMIT", help="compare the committed files with those of COMMIT"
    )
    known = parser.parse_args(arguments)
    if known.check is not None:
        differences = check(known.check)
        if differences:
            raise SystemExit("differs:\n  " + "\n  ".join(differences))
        print(f"matches {known.check} outside {REFITTED}", flush=True)
        return
    refused = refusals(False) + thread_refusals()
    if refused:
        raise SystemExit("refused:\n  " + "\n  ".join(refused))
    committed = committed_rows()
    directory, key = scratch()
    directory.mkdir(parents=True, exist_ok=True)
    print(f"run scratch: {directory}", flush=True)
    start = time.monotonic()
    with host_lock(directory):
        _claim_scratch(directory, key, fresh=False)
        for label in INHERITED:
            count = inherit_part(label, committed, directory / CHECKPOINTS)
            print(f"{label}: {count} keys inherited", flush=True)
    print(f"inherited {len(INHERITED)} fit sets in {time.monotonic() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
