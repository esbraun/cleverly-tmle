"""The first declared run of ``projection-methods``, and the check that the second reproduces it.

The first run was declared at ``32f9d8ac`` and ran at :data:`FIRST_RUN`.  Plan rev 3 (R3.5)
re-declares ``dr_glm_4k`` and the ladder on the ``property_r3`` stream and keeps every other
group, and the primary, on the streams the first run drew.  Their seeds depend only on the
record's seed, the stream, the group and the replicate, so their rows must come out as the first
run's.  :func:`check` compares them while the second run is in progress, through the shared
driver's ``check_rows`` hook (:func:`tests.studies.projection_methods.check_rows`):

- the primary rows (``replicates.parquet``) and the property rows of the unchanged groups
  (``property-replicates.parquet``): estimate, standard error, covered and rejected, per
  replicate;
- their summary rows (``summary.csv`` and ``properties.csv``): the verdict, the intervals and the
  SE ratios.

Exact equality is expected.  A relative difference up to :data:`TOLERANCE` is recorded with the
difference between the two runs' library stacks (the first run's is in its manifest, this run's
in ``run.log``).  A larger difference, a missing or extra row, or any changed indicator or
verdict raises, which stops the run before its manifest is written, so nothing publishes.

The first run's files are read with ``git show`` from the commit, which the run's checkout must
reach.  :mod:`tests.studies.projection_methods_limit` reads the same files for the rows it copies.
"""

from __future__ import annotations

import io
import json
import math
import platform
import subprocess
from collections.abc import Mapping
from importlib import metadata

import pandas as pd

from tests.studies.evidence.registry import ROOT

#: The commit of the first declared run, and the commit that declared it.
FIRST_RUN = "c566a4c27b2469f46699d5a54c58f54edec32429"
DECLARED_AT = "32f9d8acc6df602869ed380ee508913646b32846"
DIRECTORY = "tests/canonical/projection_methods"

#: The largest relative difference recorded as library drift rather than a defect.
TOLERANCE = 1e-9

#: Each compared file: its key columns, its numeric columns, and the columns that must be equal.
FRAMES: Mapping[str, tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = {
    "replicates.parquet": (
        ("implementation", "scenario", "replicate", "estimand"),
        ("estimate", "std_error", "ci_lower", "ci_upper"),
        ("covered",),
    ),
    "property-replicates.parquet": (
        ("property", "cell", "replicate"),
        ("estimate", "std_error"),
        ("covered", "rejected"),
    ),
    "summary.csv": (
        ("implementation", "scenario", "estimand"),
        ("mean_estimate", "empirical_se", "mean_std_error", "se_ratio", "coverage"),
        (),
    ),
    "properties.csv": (
        ("property", "cell"),
        (
            "bias_ci_lower",
            "bias_ci_upper",
            "standardized_bias",
            "se_ratio",
            "se_ratio_ci_lower",
            "se_ratio_ci_upper",
            "coverage",
            "coverage_ci_lower",
            "coverage_ci_upper",
        ),
        ("passed",),
    ),
}

#: The library versions a stack difference names, as the manifest keys them.
STACK = ("python", "numpy", "scipy", "pandas", "scikit_learn", "joblib")


def read(name: str, commit: str = FIRST_RUN) -> pd.DataFrame:
    """The first run's committed ``name`` (a CSV or a parquet file), read from ``commit``."""
    blob = subprocess.run(
        ["git", "show", f"{commit}:{DIRECTORY}/{name}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    if name.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(blob))
    return pd.read_csv(io.BytesIO(blob))


def first_run_stack(commit: str = FIRST_RUN) -> dict[str, str]:
    """The library versions the first run's manifest records."""
    blob = subprocess.run(
        ["git", "show", f"{commit}:{DIRECTORY}/manifest.json"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    subject = json.loads(blob)["generated_with"]["subject"]
    return {name: str(subject.get(name)) for name in STACK}


def current_stack() -> dict[str, str]:
    """This process's library versions, keyed as :func:`first_run_stack` keys them."""
    out = {"python": platform.python_version()}
    for name in STACK[1:]:
        out[name] = metadata.version(name.replace("_", "-"))
    return out


def unchanged_groups() -> tuple[str, ...]:
    """The groups the second run draws on the first run's stream."""
    from tests.studies import projection_methods as study

    return tuple(g.label for g in study.GROUPS if g.stream == study.PROPERTY_STREAM)


def unchanged_cells() -> frozenset[tuple[str, str]]:
    """The ``(family, cell)`` pairs the unchanged groups fill."""
    from tests.studies import projection_methods as study

    return frozenset(
        (family, cell)
        for label in unchanged_groups()
        for family, cells in study.group_cells(study.GROUP_BY_LABEL[label]).items()
        for cell in cells
    )


def _relative(first: pd.Series, second: pd.Series) -> float:
    a = pd.to_numeric(first, errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(second, errors="coerce").to_numpy(dtype=float)
    worst = 0.0
    for x, y in zip(a, b, strict=True):
        if math.isnan(x) and math.isnan(y):
            continue
        if math.isnan(x) or math.isnan(y):
            return math.inf
        scale = max(abs(x), abs(y))
        if scale > 0.0:
            worst = max(worst, abs(x - y) / scale)
    return worst


def compare(name: str, first: pd.DataFrame, second: pd.DataFrame) -> tuple[float, list[str]]:
    """The largest relative difference between two frames of ``name``, and each mismatch.

    A mismatch is a key present in one frame only, or a changed indicator or verdict.
    """
    keys, numeric, exact = FRAMES[name]
    merged = first.loc[:, [*keys, *numeric, *exact]].merge(
        second.loc[:, [*keys, *numeric, *exact]],
        on=list(keys),
        how="outer",
        suffixes=("_first", "_second"),
        indicator=True,
    )
    problems = [
        f"{name}: row {tuple(row)} is in one run only"
        for row in merged.loc[merged["_merge"] != "both", list(keys)].itertuples(index=False)
    ]
    both = merged.loc[merged["_merge"] == "both"]
    for column in exact:
        changed = both.loc[
            both[f"{column}_first"].astype(bool) != both[f"{column}_second"].astype(bool)
        ]
        problems += [
            f"{name}: {column} changed at {tuple(row)}"
            for row in changed.loc[:, list(keys)].itertuples(index=False)
        ]
    worst = max((_relative(both[f"{c}_first"], both[f"{c}_second"]) for c in numeric), default=0.0)
    return worst, problems


def _select(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    """The rows of ``frame`` the check compares: every primary row, the unchanged cells."""
    if name in ("replicates.parquet", "summary.csv"):
        return frame
    cells = unchanged_cells()
    pairs = list(zip(frame["property"], frame["cell"], strict=True))
    return frame.loc[[pair in cells for pair in pairs]]


def check(
    rows: pd.DataFrame,
    property_rows: pd.DataFrame,
    *,
    summary: pd.DataFrame,
    property_summary: pd.DataFrame,
    first: Mapping[str, pd.DataFrame] | None = None,
    stacks: tuple[Mapping[str, str], Mapping[str, str]] | None = None,
) -> list[str]:
    """Compare this run's unchanged rows with the first run's; the last line is the outcome.

    ``first`` replaces the files read from :data:`FIRST_RUN`, and ``stacks`` the first run's and
    this process's library versions (a test's fixture).

    Raises
    ------
    RuntimeError
        On a row in one run only, a changed indicator or verdict, or a relative difference
        above :data:`TOLERANCE`.
    """
    second = {
        "replicates.parquet": rows,
        "property-replicates.parquet": property_rows,
        "summary.csv": summary,
        "properties.csv": property_summary,
    }
    if first is None:
        first = {name: read(name) for name in FRAMES}
    lines: list[str] = []
    problems: list[str] = []
    worst = 0.0
    for name in FRAMES:
        a, b = _select(name, first[name]), _select(name, second[name])
        difference, found = compare(name, a, b)
        lines.append(
            f"first-run check {name}: {len(b)} rows, largest relative difference {difference:.3g}"
        )
        problems += found
        worst = max(worst, difference)
    if problems or worst > TOLERANCE:
        detail = "; ".join(problems[:10]) or f"relative difference {worst:.3g} > {TOLERANCE:g}"
        raise RuntimeError(
            f"the unchanged groups {list(unchanged_groups())} do not reproduce the first run "
            f"{FIRST_RUN[:8]} (defect suspect; nothing publishes): {detail}"
        )
    if worst == 0.0:
        lines.append(
            f"first-run check: the unchanged groups {list(unchanged_groups())} and the primary "
            f"equal the first run {FIRST_RUN[:8]} exactly"
        )
        return lines
    before, now = stacks if stacks is not None else (first_run_stack(), current_stack())
    drift = {k: f"{before.get(k)} -> {now.get(k)}" for k in STACK if before.get(k) != now.get(k)}
    lines.append(
        f"first-run check: the unchanged groups {list(unchanged_groups())} and the primary match "
        f"the first run {FIRST_RUN[:8]} within relative {worst:.3g} (tolerance {TOLERANCE:g}); "
        f"stack differences {drift or 'none'}"
    )
    return lines
