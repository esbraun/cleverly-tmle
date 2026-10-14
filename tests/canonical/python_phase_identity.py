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

``--fail R`` checks the failure path.  In this checkout every primary fit of replicate ``R``
raises ``LinAlgError``.  The subject rows must then equal the base's rows without replicate
``R``.  An independent study's samples and truth rows must equal the base's in full, because
its reference input is redrawn without the fit.  A dependent study's samples and truth rows
lose replicate ``R`` as well.

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

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

#: Runs in either checkout.  It accepts the base's triple or bare frame and the head's
#: ``PhaseResult``, and writes whichever frames exist.
_DUMP = r"""
import sys
from pathlib import Path

import pandas as pd

from tests.studies.evidence.registry import registered

slug, replicates, jobs, prefix, out, fail = sys.argv[1:7]
if prefix != "default":
    from tests.studies.evidence import replication

    replication._PREFIX = int(prefix)
if fail != "none":
    import numpy as np

    from tests.studies.evidence import replication

    class _Raise:
        def __init__(self, fit_one, replicate_of):
            self.fit_one, self.replicate_of = fit_one, replicate_of

        def __call__(self, key, drawn):
            if self.replicate_of(key) == int(fail):
                np.linalg.inv(np.zeros((2, 2)))
            return self.fit_one(key, drawn)

    original = replication.fit_replicates

    def _failing(keys, fit_one, *, label, replicate_of, **options):
        if label == "primary":
            fit_one = _Raise(fit_one, replicate_of)
        return original(keys, fit_one, label=label, replicate_of=replicate_of, **options)

    replication.fit_replicates = _failing
(record,) = [record for record in registered() if record.slug == slug]
module = record.runner()
result = module.draw_and_fit(replicates=int(replicates), n=module.PRIMARY_N, n_jobs=int(jobs))
if isinstance(result, tuple):
    frames = dict(zip(("samples", "truths", "rows"), result))
elif isinstance(result, pd.DataFrame):
    frames = {"rows": result}
else:
    if fail == "none" and result.failures is not None:
        raise SystemExit(f"{slug}: a fit failed: {result.failures.to_dict('records')}")
    if fail != "none" and result.failures is None:
        raise SystemExit(f"{slug}: the forced failure was not recorded")
    frames = {"rows": result.rows, "samples": result.samples, "truths": result.truths}
out = Path(out)
out.mkdir(parents=True, exist_ok=True)
if fail != "none" and result.reference_input_from_fit:
    (out / "dependent").touch()
for name, frame in frames.items():
    if frame is not None:
        frame.to_csv(out / f"{name}.csv", index=False)
"""


def _slugs() -> list[str]:
    sys.path.insert(0, str(ROOT))
    from tests.studies.evidence.registry import registered

    return [record.slug for record in registered()]


def _dump(
    checkout: Path,
    slug: str,
    replicates: int,
    jobs: int,
    prefix: str,
    out: Path,
    fail: int | None = None,
) -> None:
    environment = {**os.environ, "PYTHONPATH": f"{checkout / 'src'}{os.pathsep}{checkout}"}
    failing = "none" if fail is None else str(fail)
    subprocess.run(
        [sys.executable, "-c", _DUMP, slug, str(replicates), str(jobs), prefix, str(out), failing],
        cwd=checkout,
        env=environment,
        check=True,
    )


def compare(base: Path, head: Path, *, fail: int | None = None) -> list[str]:
    """The differences between two dumps, empty when they agree byte for byte.

    With ``fail``, the base frames the failure removes lose replicate ``fail`` first, and both
    sides are compared as text cells.
    """
    names = sorted({path.name for path in (*base.glob("*.csv"), *head.glob("*.csv"))})
    dependent = (head / "dependent").exists()
    out = []
    for name in names:
        left, right = base / name, head / name
        if not left.exists() or not right.exists():
            out.append(f"{name}: present on one side only")
        elif fail is None:
            if not filecmp.cmp(left, right, shallow=False):
                out.append(f"{name}: bytes differ")
        else:
            expected = _cells(left)
            if name == "rows.csv" or dependent:
                expected = expected.loc[expected["replicate"] != str(fail)]
            if not expected.reset_index(drop=True).equals(_cells(right)):
                out.append(f"{name}: cells differ")
    return out


def _cells(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


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
    parser.add_argument(
        "--fail", type=int, help="the replicate whose primary fits raise in this checkout"
    )
    arguments = parser.parse_args()
    failed = []
    for slug in arguments.slugs or _slugs():
        started = time.monotonic()
        suffix = "" if arguments.fail is None else f"-fail{arguments.fail}"
        target = arguments.out / f"{slug}-r{arguments.replicates}{suffix}"
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
                arguments.fail,
            )
        except subprocess.CalledProcessError as error:
            failed.append(slug)
            print(f"{slug}: ERROR ({error})", flush=True)
            continue
        differences = compare(target / "base", target / "head", fail=arguments.fail)
        verdict = "identical" if not differences else "; ".join(differences)
        if differences:
            failed.append(slug)
        print(f"{slug}: {verdict} ({time.monotonic() - started:.0f} s)", flush=True)
    print(f"{len(failed)} studies differ or failed: {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
