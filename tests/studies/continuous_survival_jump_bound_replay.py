"""A/B replay: which committed continuous-time fits the engine's jump bound moves.

Exploratory, not a study.  The jump bound (``cleverly.survival._chunked.bound_jumps``) scales a
combined cause hazard jump of one or more below one where the engine used to refuse the fit.
It must leave every fit that ran before bit for bit and only add the fits that refused.  This
module refits chosen replicates of the four registered continuous-time studies, through each
study's own draw and fit, and writes each estimate and standard error, or the failure.  Run it
on the base tree and on the branch, then compare the two at ``rtol 0``::

    python -m tests.studies.continuous_survival_jump_bound_replay run --write A --jobs 4
    python -m tests.studies.continuous_survival_jump_bound_replay compare A B

Run the base side from an export of the base commit, with this file copied into it and
``PYTHONPATH=<export>/src:<export>``.  The keys are, per primary cell, ``--per-cell`` replicates
spread evenly over the cell's committed replicates; every fit the committed records lost to the
combined-jump refusal (:data:`MOVED`); and, for ``continuous-time-survival``, replicates of its
reported cells, ``positivity_4k`` included.  ``compare`` sorts every fit into identical, now
fitting (failed on A, fits on B), changed, newly failing and failing on both.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cleverly.utils.parallel import map_parallel

#: slug -> runner module, for the four registered continuous-time studies.
STUDIES: dict[str, str] = {
    "continuous-time-survival": "tests.studies.continuous_survival",
    "continuous-survival-variance": "tests.studies.continuous_survival_variance",
    "continuous-survival-functionals": "tests.studies.continuous_survival_functionals",
    "continuous-survival-causes": "tests.studies.continuous_survival_causes",
}

#: The fits the committed records lost to the combined-jump refusal: ``(slug, cell, replicate)``.
#: ``continuous-time-survival``'s is a reported cell; the others are primary.
MOVED: tuple[tuple[str, str, int], ...] = (
    *(
        ("continuous-survival-variance", "v3_half_n", r)
        for r in (657, 1209, 2184, 3179, 4272, 4433, 4586, 5932)
    ),
    ("continuous-survival-functionals", "functionals_cf_1k", 10023),
    ("continuous-time-survival", "positivity_4k", 292),
)

#: The reported cell's survival-underflow failure, which the bound must not move.
UNMOVED: tuple[tuple[str, str, int], ...] = (("continuous-time-survival", "positivity_4k", 283),)

#: Replicates per reported cell of ``continuous-time-survival``.
REPORTED_PER_CELL = 6

COLUMNS = ["study", "scenario", "replicate", "estimand", "estimate", "std_error", "failure"]


def _committed_replicates(slug: str, cell: str) -> int:
    from tests.studies.evidence.registry import registered

    record = next(r for r in registered() if r.slug == slug)
    readings = pd.read_csv(record.artifacts / "targeting-readings.csv")
    fits = int(readings.loc[readings["scenario"] == cell, "fits"].iloc[0])
    # A cell's replicates run from 0; a lost fit is not in ``fits``, so round up to hundreds.
    return int(np.ceil(fits / 100.0) * 100)


def keys(per_cell: int) -> list[tuple[str, str, int, str]]:
    """Every ``(slug, cell, replicate, kind)`` to refit, ``kind`` "primary" or "reported"."""
    from tests.studies.evidence.registry import registered

    out: list[tuple[str, str, int, str]] = []
    records = {r.slug: r for r in registered()}
    for slug in STUDIES:
        for cell in records[slug].scenarios:
            total = _committed_replicates(slug, cell)
            spread = np.unique(np.linspace(0, total - 1, per_cell).round().astype(int))
            out.extend((slug, cell, int(r), "primary") for r in spread)
    from tests.studies import continuous_survival_properties as properties

    for cell, (_, _, count) in properties.REPORTED_CELLS.items():
        spread = np.unique(np.linspace(0, count - 1, REPORTED_PER_CELL).round().astype(int))
        out.extend(("continuous-time-survival", cell, int(r), "reported") for r in spread)
    reported = set(properties.REPORTED_CELLS)
    for slug, cell, r in (*MOVED, *UNMOVED):
        kind = "reported" if cell in reported else "primary"
        if (slug, cell, r, kind) not in out:
            out.append((slug, cell, r, kind))
    return out


def refit(slug: str, cell: str, replicate: int, kind: str) -> list[dict[str, Any]]:
    """One fit's rows: each estimate and standard error, or one row naming the failure."""
    from tests.studies.evidence.registry import registered

    base = {"study": slug, "scenario": cell, "replicate": replicate}
    if kind == "reported":
        from tests.studies import continuous_survival_properties as properties

        rows = properties._reported(cell, replicate)
        failure = str(rows[0]["failure"] or "")
        if failure:
            return [{**base, "estimand": "", "estimate": np.nan, "std_error": np.nan,
                     "failure": failure}]  # fmt: skip
        return [
            {**base, "estimand": row["estimand"], "estimate": row["estimate"],
             "std_error": row["std_error"], "failure": ""}
            for row in rows
        ]  # fmt: skip
    module = importlib.import_module(STUDIES[slug])
    record = next(r for r in registered() if r.slug == slug)
    full = (cell, replicate, record.sizes(record.n)[cell])
    try:
        rows = module._replicate(full, module._draw(full))
    except Exception as error:
        from cleverly.validation._failures import recordable

        if not recordable(error):
            raise
        message = f"{type(error).__name__}: {str(error).split('. ')[0]}"
        return [{**base, "estimand": "", "estimate": np.nan, "std_error": np.nan,
                 "failure": message}]  # fmt: skip
    return [
        {**base, "estimand": row["estimand"], "estimate": row["estimate"],
         "std_error": row["std_error"], "failure": ""}
        for row in rows
    ]  # fmt: skip


def run(write: Path, per_cell: int, n_jobs: int) -> None:
    """Refit every key and write ``rows.csv`` and ``meta.json`` under ``write``."""
    import cleverly

    write.mkdir(parents=True, exist_ok=True)
    chosen = keys(per_cell)
    started = time.time()
    outcomes = map_parallel(refit, chosen, n_jobs=n_jobs)
    rows = pd.DataFrame([row for rows in outcomes for row in rows], columns=COLUMNS)
    rows.to_csv(write / "rows.csv", index=False)
    meta = {
        "cleverly_file": cleverly.__file__,
        "keys": len(chosen),
        "per_cell": per_cell,
        "seconds": round(time.time() - started, 1),
        "python": sys.version.split()[0],
    }
    (write / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")


def compare(side_a: Path, side_b: Path) -> pd.DataFrame:
    """Per fit, its outcome on each side and whether its rows are equal at ``rtol 0``."""
    sides = [
        pd.read_csv(side / "rows.csv", float_precision="round_trip") for side in (side_a, side_b)
    ]
    fit_keys = ["study", "scenario", "replicate"]
    out = []
    for key, a in sides[0].groupby(fit_keys, sort=True):
        b = sides[1].loc[(sides[1][fit_keys] == pd.Series(key, index=fit_keys)).all(axis=1)]
        failed_a = bool(a["failure"].fillna("").ne("").any())
        failed_b = bool(b["failure"].fillna("").ne("").any())
        if b.empty:
            kind = "missing on B"
        elif failed_a and failed_b:
            kind = "failing on both"
        elif failed_a:
            kind = "now fitting"
        elif failed_b:
            kind = "newly failing"
        else:
            merged = a.merge(b, on=[*fit_keys, "estimand"], how="outer", suffixes=("_a", "_b"))
            equal = all(
                np.array_equal(merged[f"{c}_a"], merged[f"{c}_b"], equal_nan=True)
                for c in ("estimate", "std_error")
            )
            kind = "identical" if equal else "changed"
        out.append(
            {
                **dict(zip(fit_keys, key, strict=True)),
                "outcome": kind,
                "failure_a": a["failure"].fillna("").iloc[0],
                "failure_b": "" if b.empty else b["failure"].fillna("").iloc[0],
            }
        )
    return pd.DataFrame(out)


def main(argv: Sequence[str] | None = None) -> None:
    """Run one side, or compare two."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    side = commands.add_parser("run")
    side.add_argument("--write", type=Path, required=True)
    side.add_argument("--per-cell", type=int, default=40)
    side.add_argument("--jobs", type=int, default=1)
    both = commands.add_parser("compare")
    both.add_argument("side_a", type=Path)
    both.add_argument("side_b", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        run(args.write, args.per_cell, args.jobs)
        return
    table = compare(args.side_a, args.side_b)
    pd.set_option("display.width", 200)
    print(table.groupby(["study", "outcome"]).size().to_string())
    unusual = table.loc[table["outcome"] != "identical"]
    print(unusual.to_string(index=False))


if __name__ == "__main__":
    main()
