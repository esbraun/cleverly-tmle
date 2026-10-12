"""Check that the RM39b migration left every study's Python phase byte for byte unchanged.

RM39 part b moved every primary ``draw_and_fit`` loop onto
:func:`tests.studies.evidence.replication.fit_replicates`.  With no failure, the samples, truth
rows and subject rows must equal the ones the loop wrote before.  This script runs each
registered study's ``draw_and_fit`` twice in fresh interpreters: once in a worktree of the base
commit and once in this checkout.  It writes each side's three frames as CSV and compares the
bytes.

In this checkout the systematic prefix is set to one replicate, so the two-call split of
``fit_replicates`` runs at two replicates.  ``--replicates 21`` with the default prefix runs
the split the way a study runs it.

Usage::

    git worktree add /tmp/base <base commit>
    python -m tests.canonical.python_phase_identity --base /tmp/base --out /tmp/identity
"""

from __future__ import annotations

import argparse
import filecmp
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: Runs in either checkout.  It accepts the base's triple or bare frame and the head's
#: ``PhaseResult``, and writes whichever frames exist.
_DUMP = r"""
import sys
from pathlib import Path

import pandas as pd

from tests.studies.evidence.registry import registered

slug, replicates, jobs, prefix, out = sys.argv[1:6]
if prefix != "default":
    from tests.studies.evidence import replication

    replication._PREFIX = int(prefix)
(record,) = [record for record in registered() if record.slug == slug]
module = record.runner()
result = module.draw_and_fit(replicates=int(replicates), n=module.PRIMARY_N, n_jobs=int(jobs))
if isinstance(result, tuple):
    frames = dict(zip(("samples", "truths", "rows"), result))
elif isinstance(result, pd.DataFrame):
    frames = {"rows": result}
else:
    if result.failures is not None:
        raise SystemExit(f"{slug}: a fit failed: {result.failures.to_dict('records')}")
    frames = {"rows": result.rows, "samples": result.samples, "truths": result.truths}
out = Path(out)
out.mkdir(parents=True, exist_ok=True)
for name, frame in frames.items():
    if frame is not None:
        frame.to_csv(out / f"{name}.csv", index=False)
"""


def _slugs() -> list[str]:
    sys.path.insert(0, str(ROOT))
    from tests.studies.evidence.registry import registered

    return [record.slug for record in registered()]


def _dump(checkout: Path, slug: str, replicates: int, jobs: int, prefix: str, out: Path) -> None:
    environment = {**os.environ, "PYTHONPATH": f"{checkout / 'src'}{os.pathsep}{checkout}"}
    subprocess.run(
        [sys.executable, "-c", _DUMP, slug, str(replicates), str(jobs), prefix, str(out)],
        cwd=checkout,
        env=environment,
        check=True,
    )


def compare(base: Path, head: Path) -> list[str]:
    """The differences between two dumps, empty when they agree byte for byte."""
    names = sorted({path.name for path in (*base.glob("*.csv"), *head.glob("*.csv"))})
    out = []
    for name in names:
        left, right = base / name, head / name
        if not left.exists() or not right.exists():
            out.append(f"{name}: present on one side only")
        elif not filecmp.cmp(left, right, shallow=False):
            out.append(f"{name}: bytes differ")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", type=Path, required=True, help="a worktree of the base commit")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--slugs", nargs="*", help="the studies to check (default: all)")
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument(
        "--prefix", default="1", help="the head's systematic prefix, or 'default' for 20"
    )
    arguments = parser.parse_args()
    failed = []
    for slug in arguments.slugs or _slugs():
        started = time.monotonic()
        target = arguments.out / f"{slug}-r{arguments.replicates}"
        try:
            _dump(
                arguments.base,
                slug,
                arguments.replicates,
                arguments.jobs,
                "default",
                target / "base",
            )
            _dump(
                ROOT,
                slug,
                arguments.replicates,
                arguments.jobs,
                arguments.prefix,
                target / "head",
            )
        except subprocess.CalledProcessError as error:
            failed.append(slug)
            print(f"{slug}: ERROR ({error})", flush=True)
            continue
        differences = compare(target / "base", target / "head")
        verdict = "identical" if not differences else "; ".join(differences)
        if differences:
            failed.append(slug)
        print(f"{slug}: {verdict} ({time.monotonic() - started:.0f} s)", flush=True)
    print(f"{len(failed)} studies differ or failed: {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
