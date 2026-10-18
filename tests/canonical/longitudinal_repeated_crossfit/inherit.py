"""Seed the declared run's scratch with the committed rows of every unchanged fit set.

The first declared run (``cleverly_commit`` of the committed manifest) read the
``simultaneous_coverage`` band and the ``repeat_increment`` pair off the calibration fits.  The
rerun draws each of those families its own samples (``band_*`` and ``increment_s`` of
:data:`tests.studies.longitudinal_repeated_crossfit_properties.FIT_SETS`) and leaves every other
fit set as it was.  An unchanged fit set draws the same samples, fits them with the same code,
and writes the same rows.  This entry checkpoints those rows from the committed
``property-replicates.csv.gz`` as if the declared run had fitted them, so the declared run fits
only the new fit sets and the primary phase.

    python -m tests.canonical.longitudinal_repeated_crossfit.inherit
    python -m tests.canonical.longitudinal_repeated_crossfit.inherit --check <baseline commit>

``--check`` runs after the declared run.  It compares the committed files with those of the
baseline commit and exits non-zero on a difference: every property row outside the two
families that the rerun redraws, and the primary rows, must be unchanged.

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

#: The families whose samples the rerun redraws.  Every other property row is inherited.
REDRAWN = ("simultaneous_coverage", "repeat_increment")

#: The fit sets that the rerun fits: each family that has its own samples now.
FITTED = tuple(label for label in properties.plan() if label.startswith(("band_", "increment_")))

#: The fit sets and stability laws whose rows the committed file supplies.
INHERITED = tuple(label for label in properties.plan() if label not in FITTED)


def committed_rows(here: Path = HERE) -> pd.DataFrame:
    """The committed property rows, checked against the hash that the committed manifest records."""
    manifest = json.loads((here / "manifest.json").read_text(encoding="utf-8"))
    recorded = manifest["sha256"][ROWS]
    if hashlib.sha256((here / ROWS).read_bytes()).hexdigest() != recorded:
        raise RuntimeError(f"refused: {ROWS} no longer matches its manifest hash")
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


def check(commit: str) -> list[str]:
    """What differs from ``commit`` outside the families that the rerun redraws."""
    differences: list[str] = []
    if baseline(commit, PRIMARY_ROWS) != (HERE / PRIMARY_ROWS).read_bytes():
        differences.append(f"{PRIMARY_ROWS} differs from {commit}")
    old = pd.read_csv(
        io.BytesIO(baseline(commit, ROWS)), compression="gzip", float_precision="round_trip"
    )
    new = pd.read_csv(HERE / ROWS, float_precision="round_trip")
    order = ["property", "cell", "replicate"]
    old = old.loc[~old["property"].isin(REDRAWN)].sort_values(order, ignore_index=True)
    new = new.loc[~new["property"].isin(REDRAWN)].sort_values(order, ignore_index=True)
    try:
        pd.testing.assert_frame_equal(old, new, check_exact=True)
    except AssertionError as error:
        differences.append(f"{ROWS} differs from {commit} outside {REDRAWN}: {error}")
    return differences


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
        print(f"identical to {known.check} outside {REDRAWN}", flush=True)
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
