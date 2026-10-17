"""Replay committed study rows, and check a regenerated study against its committed rows.

Two checks for a change that must not move a deterministic path.

``replay`` refits the first replications of each named study and compares every estimate and
standard error with the committed ``replicates.parquet``, which holds every float64 exactly,
so the comparison is exact.  A label map renames estimands first, for a study
whose estimand names a change renamed.

``relabel`` compares every file of a regenerated study directory with the same file at a git
ref, after the label map.  Every numeric column must be equal, and every text column equal after
the map, with two missing values equal.  It is the declared assertion that a regeneration only
relabels.

Committed rows may come from another platform, where the last bit of a float can differ.  On
2026-10-06 the first five replications of every longitudinal study and of ``shift-policies``
replayed here within 1.7e-15 of the committed rows, while ``origin/main`` and this branch
agreed bit for bit on the same machine.  So ``--rtol`` admits a difference of that order,
relative to the largest magnitude in the column: a last-bit change in the inputs moves a
derived difference near zero, such as a bias, by the inputs' scale and not its own.  ``0``
asks for exact equality.

A rename moves one thing that is not a relabel.  The bootstrap bounds of
``performance-tests.csv`` and ``equivalence.csv`` draw from a stream whose seed hashes the
estimand label (``evidence.seeds.stream_seed``), so a renamed estimand draws a new resample.
``RESAMPLED`` names those columns.  The check reports them and does not fail on them; every
verdict column that reads them must still be equal.

``replay --all`` names every registered study, minus ``REPLAY_EXCLUDED``.  ``replay --write DIR``
also stores the replayed rows of each study under ``DIR/rows/`` and a ``meta.json`` that names
the source tree the run imported.  ``compare DIR_A DIR_B`` then compares two such directories
row by row, so a change that must not move a default path is judged against the same machine's
own baseline and not against rows another platform wrote.  Run the base side from a git
worktree of the base commit, with ``PYTHONPATH=<worktree>/src:<worktree>`` so that the worktree's
``src`` precedes the editable install of the main checkout.  ``meta.json`` records
``cleverly.__file__``, the commit, the hash of ``git diff HEAD`` and the hash of the untracked
files under ``src/``.  ``compare`` exits 2 when both directories name one tree, commit and diff,
because it would compare a source with itself; ``--same-source`` declares an A/A control.

Usage::

    python -m tests.canonical.replay_committed replay --studies canonical-ltmle shift-policies
    python -m tests.canonical.replay_committed replay --all --replicates 20 --jobs 4 --write A
    python -m tests.canonical.replay_committed compare A B --rtol 0
    python -m tests.canonical.replay_committed relabel --study shift-policies --ref f71548ce

The exit status of ``replay`` and ``relabel`` is 1 on any difference.  ``replay --write`` reports
the gap to the committed rows and exits 0 when every study ran, because the committed rows drift
across platforms; it exits 3 when a study raised.  The exit status of ``compare`` is 0 for equal
sides, 1 for a difference and 2 for one source.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tests.studies.evidence import failures
from tests.studies.evidence.manifest import read_rows
from tests.studies.evidence.registry import ROOT, StudyRecord, registered

#: The estimand renames of the modified-treatment-policy change: ``shifts=`` became
#: ``policies=``, so ``ey_shift`` and ``ate_shift`` became ``ey_policy`` and ``ate_policy``.
POLICY_LABELS: dict[str, str] = {"ey_shift[": "ey_policy[", "ate_shift[": "ate_policy["}

#: The columns a replay compares.
REPLAYED = ("estimate", "std_error")

#: The keys that identify a replayed row.
ROW_KEYS = ["scenario", "replicate", "estimand"]

#: Registered studies that ``replay --all`` skips, each with its reason.  A study stays out only
#: when its runner has no ``draw_and_fit`` or the replay cannot judge it.  The set is empty:
#: every registered study has a Python runner.
REPLAY_EXCLUDED: tuple[tuple[str, str], ...] = ()

#: The thread variables that a replay side records beside its rows.
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")

#: Per artifact, the bootstrap bounds whose stream seed hashes the estimand label.
RESAMPLED: dict[str, frozenset[str]] = {
    "performance-tests.csv": frozenset(
        {"se_ratio_ci_lower", "se_ratio_ci_upper", "se_ratio_resolution"}
    ),
    "equivalence.csv": frozenset(
        {
            "rmse_ratio_upper",
            "coverage_difference_lower",
            "calibration_excess_upper",
            "calibration_excess_resolution",
        }
    ),
}


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
    if name.endswith(".parquet"):
        return read_rows(handle)
    compression = "gzip" if name.endswith(".gz") else None
    return pd.read_csv(handle, compression=compression, float_precision="round_trip")


def _study(slug: str) -> StudyRecord:
    for record in registered():
        if record.slug == slug:
            return record
    raise KeyError(f"no registered study {slug!r}")


def replayable() -> tuple[str, ...]:
    """The slug of every registered study that ``replay --all`` runs, in registry order."""
    excluded = {slug for slug, _ in REPLAY_EXCLUDED}
    return tuple(record.slug for record in registered() if record.slug not in excluded)


def source_meta() -> dict[str, Any]:
    """Where the imported ``cleverly`` came from, and whether that tree is a clean commit.

    ``diff_sha256`` hashes ``git diff HEAD`` of the tree and ``untracked_src_sha256`` hashes the
    path and bytes of every untracked file under ``src/``.  Two sides built from one commit
    with different local edits therefore differ in at least one of them.
    """
    import cleverly

    file = Path(cleverly.__file__).resolve()

    def git(*args: str, cwd: Path = file.parent) -> bytes:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, check=False).stdout

    tree = Path(git("rev-parse", "--show-toplevel").decode().strip() or file.parents[2])
    untracked = git("ls-files", "--others", "--exclude-standard", "--", "src", cwd=tree)
    hasher = hashlib.sha256()
    for name in sorted(untracked.decode().splitlines()):
        hasher.update(name.encode() + b"\0" + (tree / name).read_bytes() + b"\0")
    return {
        "cleverly_file": str(file),
        "tree": str(tree),
        "head": git("rev-parse", "HEAD", cwd=tree).decode().strip(),
        "diff_sha256": hashlib.sha256(git("diff", "HEAD", cwd=tree)).hexdigest(),
        "untracked_src_sha256": hasher.hexdigest(),
        "threads": {name: os.environ.get(name) for name in THREAD_VARIABLES},
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


#: The ``meta.json`` fields that say whether two sides are one source.
SOURCE_FIELDS = ("tree", "head", "diff_sha256", "untracked_src_sha256")


def replay(
    slugs: Sequence[str],
    replicates: int,
    n_jobs: int,
    labels: Mapping[str, str],
    rtol: float = 0.0,
    write: Path | None = None,
) -> float:
    """The largest absolute difference over the replayed rows of every study.

    A study whose every estimate and standard error agrees with the committed row within
    ``rtol`` (:func:`equal_entries`) counts as 0.0, and its raw gap is still printed.  With
    ``write``, the replayed rows of each study go to ``write/rows/<slug>.csv`` and
    ``write/meta.json`` names the source (:func:`source_meta`).  A study that raises is
    recorded in ``meta.json``, and the return value is then ``nan``.
    """
    worst = 0.0
    errors: dict[str, str] = {}
    started = time.time()
    if write is not None:
        write.mkdir(parents=True, exist_ok=True)
    for slug in slugs:
        try:
            worst = max(worst, _replay_one(slug, replicates, n_jobs, labels, rtol, write))
        except Exception as error:
            if write is None:
                raise
            errors[slug] = f"{type(error).__name__}: {error}"[:300]
            print(f"{slug}: raised {errors[slug]}", flush=True)
    if write is not None:
        meta = {
            **source_meta(),
            "jobs": n_jobs,
            "replicates": replicates,
            "rtol_to_committed": rtol,
            "studies": list(slugs),
            "errors": errors,
            "seconds": round(time.time() - started, 1),
        }
        (write / "meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    return float("nan") if errors else worst


def _replay_one(
    slug: str,
    replicates: int,
    n_jobs: int,
    labels: Mapping[str, str],
    rtol: float,
    write: Path | None,
) -> float:
    record = _study(slug)
    module = importlib.import_module(record.runner_module)
    result = module.draw_and_fit(replicates=replicates, n=record.n, n_jobs=n_jobs)
    if write is not None:
        (write / "rows").mkdir(parents=True, exist_ok=True)
        frame = result.rows.loc[:, [*ROW_KEYS, *REPLAYED]].sort_values(ROW_KEYS)
        frame.to_csv(write / "rows" / f"{slug}.csv", index=False)
    committed = relabel(_read(record.artifacts / "replicates.parquet", ".parquet"), labels)
    committed = committed.loc[
        (committed["implementation"] == record.implementation)
        & (committed["replicate"] < replicates)
    ]
    # A replicate the subject failed has no row to compare (RM39b).  The replay must fail
    # on the same replicates as the committed run, or the failure itself is the drift.
    lost = _lost(result.failures, record)
    recorded = {
        pair
        for pair in _lost(failures.read(record.artifacts / failures.FAILURES), record)
        if pair[1] < replicates
    }
    if lost != recorded:
        print(
            f"{slug}: the replay failed on {sorted(lost)}, the committed run on {sorted(recorded)}",
            flush=True,
        )
        return float("inf")
    if lost:
        print(f"{slug}: skipping the failed replicates {sorted(lost)}", flush=True)
    rows = result.rows.loc[~_among(result.rows, lost)]
    committed = committed.loc[~_among(committed, lost)]
    merged = rows.merge(committed, on=ROW_KEYS, suffixes=("_now", "_committed"), how="outer")
    missing = merged[[f"{c}_now" for c in REPLAYED] + [f"{c}_committed" for c in REPLAYED]]
    if missing.isna().any().any():
        print(f"{slug}: rows do not join on {ROW_KEYS}", flush=True)
        return float("inf")
    gap = max(
        float(np.max(np.abs(merged[f"{c}_now"] - merged[f"{c}_committed"]))) for c in REPLAYED
    )
    agree = all(
        equal_entries(merged[f"{c}_now"], merged[f"{c}_committed"], rtol).all() for c in REPLAYED
    )
    print(f"{slug}: {len(merged)} rows, max abs diff {gap!r}", flush=True)
    return 0.0 if agree else gap


def _lost(recorded: pd.DataFrame | None, record: StudyRecord) -> set[tuple[str, int]]:
    """The (scenario, replicate) pairs the subject's records cover."""
    lost = failures.expand(recorded, record)
    lost = lost.loc[lost["implementation"] == record.implementation]
    return {
        (str(scenario), int(replicate))
        for scenario, replicate in lost[["scenario", "replicate"]].itertuples(False)
    }


def _among(rows: pd.DataFrame, lost: set[tuple[str, int]]) -> pd.Series:
    pairs = zip(rows["scenario"].astype(str), rows["replicate"].astype(int), strict=True)
    return pd.Series([pair in lost for pair in pairs], index=rows.index, dtype=bool)


def equal_entries(left: pd.Series, right: pd.Series, rtol: float = 0.0) -> np.ndarray:
    """Per row, whether two columns agree.

    Numbers agree when equal, both missing, or within ``rtol`` of the largest finite magnitude
    in either column.  An infinite entry sets no scale, so it cannot widen the tolerance of
    every other row; it agrees only with the same infinity.  Text agrees when equal or both
    missing.
    """
    if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
        a, b = left.to_numpy(dtype=float), right.to_numpy(dtype=float)
        both = np.abs(np.concatenate([a, b]))
        finite = both[np.isfinite(both)]
        scale = float(finite.max()) if finite.size else 0.0
        with np.errstate(invalid="ignore"):
            close = np.abs(a - b) <= rtol * scale
        return np.asarray((a == b) | close | (np.isnan(a) & np.isnan(b)), dtype=bool)
    missing = (left.isna() & right.isna()).to_numpy()
    return np.asarray(missing | (left.astype(str) == right.astype(str)).to_numpy(), dtype=bool)


def relabel_check(
    slug: str, ref: str, labels: Mapping[str, str], rtol: float = 0.0
) -> tuple[list[str], list[str]]:
    """The differences between a regenerated study and ``ref``, and the resampled moves."""
    record = _study(slug)
    problems: list[str] = []
    resampled: list[str] = []
    for path in sorted([*record.artifacts.glob("*.csv*"), *record.artifacts.glob("*.parquet")]):
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
            equal = equal_entries(before[column], after[column], rtol)
            if not np.all(equal):
                message = f"{relative}: column {column!r} differs on {int((~equal).sum())} rows"
                if column in RESAMPLED.get(path.name, frozenset()):
                    resampled.append(message)
                else:
                    problems.append(message)
    return problems, resampled


def _read_side(side: Path) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """The ``meta.json`` and the replayed rows, by slug, of a directory ``replay --write`` made."""
    meta = json.loads((side / "meta.json").read_text())
    rows = {
        path.stem: pd.read_csv(path, float_precision="round_trip")
        for path in sorted((side / "rows").glob("*.csv"))
    }
    return meta, rows


def one_source(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Whether two ``meta.json`` records name the same tree, commit and local edits."""
    return all(left.get(field) == right.get(field) for field in SOURCE_FIELDS)


def compare(
    side_a: Path, side_b: Path, rtol: float = 0.0, same_source: bool = False
) -> tuple[int, list[str]]:
    """The exit status and the report of two ``replay --write`` directories.

    The status is 2 when both name one source and ``same_source`` is not set, 1 when any
    study differs (or one side lacks a study, or a side recorded an error), and 0 otherwise.
    Rows join on ``scenario``, ``replicate`` and ``estimand``.  A numeric column agrees under
    :func:`equal_entries` with ``rtol``.
    """
    meta_a, rows_a = _read_side(side_a)
    meta_b, rows_b = _read_side(side_b)
    if one_source(meta_a, meta_b) and not same_source:
        return 2, [
            "both sides name one source tree, commit and diff "
            f"({meta_a.get('tree')} at {str(meta_a.get('head'))[:10]}); comparing it with "
            "itself shows nothing.  Pass --same-source for an A/A control."
        ]
    report: list[str] = []
    problems = 0
    for side, meta in (("A", meta_a), ("B", meta_b)):
        for slug, message in meta.get("errors", {}).items():
            report.append(f"side {side} recorded an error in {slug}: {message}")
            problems += 1
    for slug in sorted(set(rows_a) | set(rows_b)):
        if slug not in rows_a or slug not in rows_b:
            report.append(f"{slug}: present on side {'A' if slug in rows_a else 'B'} only")
            problems += 1
            continue
        merged = rows_a[slug].merge(
            rows_b[slug], on=ROW_KEYS, suffixes=("_a", "_b"), how="outer", indicator=True
        )
        unjoined = int((merged["_merge"] != "both").sum())
        differing = 0
        for column in REPLAYED:
            both = merged.loc[merged["_merge"] == "both"]
            differing += int((~equal_entries(both[f"{column}_a"], both[f"{column}_b"], rtol)).sum())
        if unjoined or differing:
            report.append(
                f"{slug}: {unjoined} rows on one side only, {differing} differing entries"
            )
            problems += 1
        else:
            report.append(f"{slug}: {len(merged)} rows equal")
    report.append(f"compare: {'equal' if not problems else f'{problems} studies differ'}")
    return (1 if problems else 0), report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    replayer = sub.add_parser("replay")
    which = replayer.add_mutually_exclusive_group(required=True)
    which.add_argument("--studies", nargs="+")
    which.add_argument("--all", action="store_true", help="every registered study")
    replayer.add_argument("--replicates", type=int, default=5)
    replayer.add_argument("--jobs", type=int, default=1)
    replayer.add_argument("--rtol", type=float, default=0.0)
    replayer.add_argument("--write", type=Path, default=None, metavar="DIR")
    comparer = sub.add_parser("compare")
    comparer.add_argument("side_a", type=Path)
    comparer.add_argument("side_b", type=Path)
    comparer.add_argument("--rtol", type=float, default=0.0)
    comparer.add_argument("--same-source", action="store_true")
    checker = sub.add_parser("relabel")
    checker.add_argument("--study", required=True)
    checker.add_argument("--ref", required=True)
    checker.add_argument("--rtol", type=float, default=0.0)
    arguments = parser.parse_args(argv)
    if arguments.command == "replay":
        slugs = replayable() if arguments.all else arguments.studies
        worst = replay(
            slugs,
            arguments.replicates,
            arguments.jobs,
            POLICY_LABELS,
            arguments.rtol,
            arguments.write,
        )
        print(f"max abs diff over every study: {worst!r}", flush=True)
        if arguments.write is not None:
            return 3 if np.isnan(worst) else 0
        return 0 if worst == 0.0 else 1
    if arguments.command == "compare":
        status, report = compare(
            arguments.side_a, arguments.side_b, arguments.rtol, arguments.same_source
        )
        for line in report:
            print(line, flush=True)
        return status
    problems, resampled = relabel_check(
        arguments.study, arguments.ref, POLICY_LABELS, arguments.rtol
    )
    for move in resampled:
        print(f"resampled under the new label: {move}", flush=True)
    for problem in problems:
        print(problem, flush=True)
    print("relabel check:", "passed" if not problems else f"{len(problems)} differences")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
