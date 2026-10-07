"""Replay committed study rows, and check a regenerated study against its committed rows.

Two checks for a change that must not move a deterministic path.

``replay`` refits the first replications of each named study and compares every estimate and
standard error with the committed ``replicates.csv.gz``, read with ``float_precision=
"round_trip"`` so the comparison is exact.  A label map renames estimands first, for a study
whose estimand names a change renamed.

``relabel`` compares every file of a regenerated study directory with the same file at a git
ref, after the label map.  Every numeric column must be equal, and every text column equal after
the map.  It is the declared assertion that a regeneration only relabels.

Committed rows may come from another platform, where the last bit of a float can differ.  On
2026-10-06 the first five replications of every longitudinal study and of ``shift-policies``
replayed here within 1.7e-15 of the committed rows, while ``origin/main`` and this branch
agreed bit for bit on the same machine.  So ``--rtol`` admits a relative difference of that
order; ``0`` asks for exact equality.

Usage::

    python -m tests.canonical.replay_committed replay --studies canonical-ltmle shift-policies
    python -m tests.canonical.replay_committed relabel --study shift-policies --ref f71548ce

The exit status is 1 on any difference.
"""

from __future__ import annotations

import argparse
import importlib
import io
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from tests.studies.evidence.registry import ROOT, StudyRecord, registered

#: The estimand renames of the modified-treatment-policy change: ``shifts=`` became
#: ``policies=``, so ``ey_shift`` and ``ate_shift`` became ``ey_policy`` and ``ate_policy``.
POLICY_LABELS: dict[str, str] = {"ey_shift[": "ey_policy[", "ate_shift[": "ate_policy["}

#: The columns a replay compares.
REPLAYED = ("estimate", "std_error")


def relabel(frame: pd.DataFrame, labels: Mapping[str, str]) -> pd.DataFrame:
    """Every text column with each ``old`` prefix or substring replaced by its ``new`` one."""
    out = frame.copy()
    for column in out.columns:
        if not pd.api.types.is_numeric_dtype(out[column]):
            values = out[column].astype(object)
            for old, new in labels.items():
                values = values.map(
                    lambda v, o=old, n=new: v.replace(o, n) if isinstance(v, str) else v
                )
            out[column] = values
    return out


def _read(source: Path | bytes, name: str) -> pd.DataFrame:
    handle = io.BytesIO(source) if isinstance(source, bytes) else source
    compression = "gzip" if name.endswith(".gz") else None
    return pd.read_csv(handle, compression=compression, float_precision="round_trip")


def _study(slug: str) -> StudyRecord:
    for record in registered():
        if record.slug == slug:
            return record
    raise KeyError(f"no registered study {slug!r}")


def replay(slugs: Sequence[str], replicates: int, n_jobs: int, labels: Mapping[str, str]) -> float:
    """The largest absolute difference over the replayed rows of every study."""
    worst = 0.0
    for slug in slugs:
        record = _study(slug)
        module = importlib.import_module(record.runner_module)
        _, _, rows = module.draw_and_fit(replicates=replicates, n=record.n, n_jobs=n_jobs)
        committed = relabel(_read(record.artifacts / "replicates.csv.gz", ".gz"), labels)
        committed = committed.loc[
            (committed["implementation"] == record.implementation)
            & (committed["replicate"] < replicates)
        ]
        keys = ["scenario", "replicate", "estimand"]
        merged = rows.merge(committed, on=keys, suffixes=("_now", "_committed"), how="outer")
        missing = merged[[f"{c}_now" for c in REPLAYED] + [f"{c}_committed" for c in REPLAYED]]
        if missing.isna().any().any():
            print(f"{slug}: rows do not join on {keys}", flush=True)
            worst = float("inf")
            continue
        gap = max(
            float(np.max(np.abs(merged[f"{c}_now"] - merged[f"{c}_committed"]))) for c in REPLAYED
        )
        print(f"{slug}: {len(merged)} rows, max abs diff {gap!r}", flush=True)
        worst = max(worst, gap)
    return worst


def relabel_check(slug: str, ref: str, labels: Mapping[str, str], rtol: float = 0.0) -> list[str]:
    """Every difference between a regenerated study directory and the same files at ``ref``."""
    record = _study(slug)
    problems: list[str] = []
    for path in sorted(record.artifacts.glob("*.csv*")):
        relative = path.relative_to(ROOT).as_posix()
        shown = subprocess.run(
            ["git", "show", f"{ref}:{relative}"], cwd=ROOT, capture_output=True, check=False
        )
        if shown.returncode != 0:
            problems.append(f"{relative}: not present at {ref}")
            continue
        before = relabel(_read(shown.stdout, path.name), labels)
        after = _read(path, path.name)
        if list(before.columns) != list(after.columns) or len(before) != len(after):
            problems.append(f"{relative}: columns or row count differ")
            continue
        for column in after.columns:
            left, right = before[column], after[column]
            if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
                a, b = left.to_numpy(dtype=float), right.to_numpy(dtype=float)
                close = np.abs(a - b) <= rtol * np.maximum(np.abs(a), np.abs(b))
                equal = (a == b) | close | (np.isnan(a) & np.isnan(b))
            else:
                equal = (left.astype(str) == right.astype(str)).to_numpy()
            if not np.all(equal):
                problems.append(
                    f"{relative}: column {column!r} differs on {int((~equal).sum())} rows"
                )
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    replayer = sub.add_parser("replay")
    replayer.add_argument("--studies", nargs="+", required=True)
    replayer.add_argument("--replicates", type=int, default=5)
    replayer.add_argument("--jobs", type=int, default=1)
    checker = sub.add_parser("relabel")
    checker.add_argument("--study", required=True)
    checker.add_argument("--ref", required=True)
    checker.add_argument("--rtol", type=float, default=0.0)
    arguments = parser.parse_args(argv)
    if arguments.command == "replay":
        worst = replay(arguments.studies, arguments.replicates, arguments.jobs, POLICY_LABELS)
        print(f"max abs diff over every study: {worst!r}", flush=True)
        return 0 if worst == 0.0 else 1
    problems = relabel_check(arguments.study, arguments.ref, POLICY_LABELS, arguments.rtol)
    for problem in problems:
        print(problem, flush=True)
    print("relabel check:", "passed" if not problems else f"{len(problems)} differences")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
