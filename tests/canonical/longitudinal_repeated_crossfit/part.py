"""Run one part of the declared ``longitudinal-repeated-crossfit`` run into its scratch.

The declared run (``regenerate.py`` without ``--replicates``) fits the primary phase and every
property fit set in one process, about 360 CPU-hours.  This entry fits one part of it, or the
first replicates of one part, into the scratch and checkpoints the declared run will use.  The
declared run then reuses every finished replicate and fits only the rest.

A part is ``primary`` or a key of
:func:`tests.studies.longitudinal_repeated_crossfit_properties.plan`.  ``--first K`` fits the
first ``K`` replicates of the part, so a part can run in chunks.  A part refuses what a
declared run refuses: a dirty tree, ``HEAD`` not equal to its upstream, or a thread variable
other than 1.  The scratch is keyed by the commit, so every part and the declared run must run
from the same commit and under the same ``LOCALAPPDATA``.

    python -m tests.canonical.longitudinal_repeated_crossfit.part --part calibration_d --jobs 4
    python -m tests.canonical.longitudinal_repeated_crossfit.part --part primary --first 400
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from collections.abc import Sequence
from pathlib import Path

from tests.canonical.declared_run import thread_refusals
from tests.canonical.regenerate import (
    CHECKPOINTS,
    _claim_scratch,
    digest,
    host_lock,
    run_key,
    runs_root,
)
from tests.diagnostics.rm18_shared import refusals
from tests.parallel import available_cores
from tests.studies import longitudinal_repeated_crossfit as study
from tests.studies import longitudinal_repeated_crossfit_properties as properties
from tests.studies.evidence import replication

#: Every part this entry runs, in the declared run's order.
PARTS = ("primary", *properties.plan())


def scratch() -> tuple[Path, dict[str, object]]:
    """The scratch directory and resume key of the declared run from this commit."""
    declared = dataclasses.replace(
        study.STUDY, replicates=study.PRIMARY_REPLICATES, n=study.PRIMARY_N
    )
    key = run_key(declared)
    return runs_root() / declared.slug / digest(key)[:12], key


def fit(part: str, *, n_jobs: int, first: int | None, root: Path) -> int:
    """Fit ``part`` under the checkpoint root ``root``; return the number of keys it covered."""
    with replication.checkpoints(root):
        if part == "primary":
            count = study.PRIMARY_REPLICATES if first is None else first
            result = study.draw_and_fit(replicates=count, n=study.PRIMARY_N, n_jobs=n_jobs)
            if result.failures is not None and not result.failures.empty:
                print(f"primary: {len(result.failures)} recorded fitting failures", flush=True)
            return count * len(study.SCENARIOS)
        return len(properties.fit_part(part, n_jobs=n_jobs, first=first))


def main(arguments: Sequence[str] | None = None) -> None:
    """Parse the part, apply the declared run's refusals and fit the part into its scratch."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--part", required=True, choices=PARTS)
    parser.add_argument("--first", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=available_cores())
    known = parser.parse_args(arguments)
    refused = refusals(False) + thread_refusals()
    if refused:
        raise SystemExit("refused:\n  " + "\n  ".join(refused))
    directory, key = scratch()
    directory.mkdir(parents=True, exist_ok=True)
    print(f"run scratch: {directory}", flush=True)
    start = time.monotonic()
    with host_lock(directory):
        _claim_scratch(directory, key, fresh=False)
        covered = fit(
            known.part, n_jobs=known.jobs, first=known.first, root=directory / CHECKPOINTS
        )
    print(f"{known.part}: {covered} fits or rows in {time.monotonic() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
