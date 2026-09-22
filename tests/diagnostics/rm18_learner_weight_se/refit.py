"""Refit the selected and comparison replicates, and read the five declared predictions.

RM18 of ``docs/roadmap.md`` declares this diagnostic in "The learner-weight standard-error
diagnostic, declared before it runs".  Each refit goes through ``_fit_replication`` in
``tests/studies/weighted_longitudinal_properties_common.py`` with ``cross_fit=True``, on the
payload ``_payloads`` builds for ``weighted-ltmle-crossfit``.  The module wraps that file's
``fit`` for the duration of one call, so the statistics come from the very ``LTMLE`` result
whose row the reproduction control checks.

A follower is a unit with a nonzero final-node clever covariate for the regimen.  A floored
follower is a follower whose ``cumulative_unbounded`` prefix at the final node is below the
lower ``g_bounds`` limit.  P1 to P4 read the two regimens of the reported contrast, ``always``
and ``never``, and P2 also reports every fitted regimen beside it.  P5 reads every fitted
regimen, because the declaration states that scope for it.  The README records each choice.

``--output`` is required and names a directory.  The run writes ``refit.csv`` and
``reading.csv`` there, so a bare run cannot overwrite the committed record.  ``--jobs`` sets
the one process pool.  Limit every numerical library to one thread in the environment first.

    python -m tests.diagnostics.rm18_learner_weight_se.refit --output <scratch> --jobs 16

``--read-only`` skips the refit and reads the predictions from an existing ``refit.csv``.
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cleverly.utils.parallel import map_parallel
from tests.diagnostics.rm18_learner_weight_se.select import (
    ARMS,
    PROPERTY,
    comparison,
    learner_weight_rows,
    selected,
)
from tests.studies import weighted_longitudinal_properties_common as common
from tests.studies.canonical_weighted_ltmle_crossfit import STUDY
from tests.studies.evidence.manifest import write_csv

HERE = Path(__file__).resolve().parent

CONTRAST = common.CONTRASTS["static"]
#: The two regimens of :data:`CONTRAST`.  The predictions read these two.
CONTRAST_REGIMENS = ("always", "never")
FLOOR = common.G_BOUNDS[0]

#: The reproduction control's tolerance on the committed ``estimate`` and ``std_error``.
REPRODUCTION_TOLERANCE = 1e-9
#: P3: the least share of the squared contrast influence curve floored followers carry.
SHARE_THRESHOLD = 0.9
#: P5: the ``max_weight`` a comparison fit stays below.
MAX_WEIGHT_THRESHOLD = 2_000.0

SELECTED = "selected"
COMPARISON = "comparison"

HOLDS = "holds"
FAILS = "fails"
NOT_READ = "not read"

POSITIVITY = "finite-sample empty-cell instability in the out-of-fold mechanism estimate"
FLOOR_NOT_FIT = "the floor, not the saturated fit"
UNRESOLVED = "unresolved"
NOT_THE_FLOOR = "not the floor"
NOT_VALIDATED = "harness not validated, no reading"

Payload = tuple[str, str, int, int, int, int, str]

REFIT_COLUMNS = (
    "group",
    "replicate",
    "cell",
    "regimen",
    "committed_estimate",
    "refit_estimate",
    "estimate_relative_difference",
    "committed_std_error",
    "refit_std_error",
    "std_error_relative_difference",
    "floored_share",
    "followers",
    "floored_followers",
    "zero_prefix_followers",
    "min_follower_prefix",
    "max_weight",
    "effective_n",
)


def payloads(replicates: Iterable[int]) -> list[Payload]:
    """The ``_payloads`` entries of both arms for ``replicates``, in payload order."""
    wanted = set(replicates)
    calls = [
        payload
        for payload in common._payloads(STUDY)
        if payload[0] == PROPERTY and payload[2] in wanted
    ]
    expected = {(cell.removeprefix("static__"), replicate) for cell in ARMS for replicate in wanted}
    observed = [(payload[1], payload[2]) for payload in calls]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        missing = sorted(expected - set(observed))
        extra = sorted(set(observed) - expected)
        raise RuntimeError(f"the refit payloads are incomplete; missing={missing}, extra={extra}")
    return calls


def _regimen_statistics(fit: Any) -> dict[str, Any]:
    final = fit.steps[-1]
    followers = np.asarray(final.clever) != 0.0
    prefix = np.asarray(fit.cumulative_unbounded, dtype=float)[:, -1]
    floored = followers & (prefix < FLOOR)
    return {
        "followers": int(followers.sum()),
        "floored_followers": int(floored.sum()),
        "zero_prefix_followers": int((followers & (prefix == 0.0)).sum()),
        "min_follower_prefix": float(prefix[followers].min()) if followers.any() else math.nan,
        "max_weight": float(fit.max_weight),
        "effective_n": float(fit.effective_n),
        "floored": floored,
    }


def refit_one(payload: Payload) -> list[dict[str, Any]]:
    """One declared refit, and the statistics of the result it produced.

    Returns one record per fitted regimen.  The replicate-and-arm statistics repeat on each.
    """
    captured: list[Any] = []
    original: Callable[..., Any] = common.fit

    def capturing(*arguments: Any, **options: Any) -> Any:
        result = original(*arguments, **options)
        captured.append(result)
        return result

    common.fit = capturing  # type: ignore[assignment]
    try:
        started = time.perf_counter()
        rows = common._fit_replication(STUDY, True, payload)
        seconds = time.perf_counter() - started
    finally:
        common.fit = original  # type: ignore[assignment]
    if len(captured) != 1 or len(rows) != 1:
        raise RuntimeError(f"expected one fit and one row, got {len(captured)} and {len(rows)}")
    result, row = captured[0], rows[0]
    fits = {fit.regimen.label: fit for fit in result.fits.values()}
    if set(CONTRAST_REGIMENS) - set(fits):
        raise RuntimeError(f"the fit has regimens {sorted(fits)}, not {CONTRAST_REGIMENS}")
    statistics = {label: _regimen_statistics(fit) for label, fit in fits.items()}
    influence = np.asarray(
        fits["always"].influence_curve_scaled - fits["never"].influence_curve_scaled, dtype=float
    )
    floored = statistics["always"]["floored"] | statistics["never"]["floored"]
    total = float(np.sum(np.square(influence)))
    share = float(np.sum(np.square(influence[floored])) / total)
    contrast_std_error = float(result[CONTRAST].std_error)
    if contrast_std_error != float(row["std_error"]):
        raise RuntimeError("the captured result is not the one the row reports")
    return [
        {
            "replicate": int(row["replicate"]),
            "cell": str(row["cell"]),
            "regimen": label,
            "refit_estimate": float(row["estimate"]),
            "refit_std_error": float(row["std_error"]),
            "floored_share": share,
            **{key: value for key, value in values.items() if key != "floored"},
            "seconds": seconds,
        }
        for label, values in statistics.items()
    ]


def _relative(refit: pd.Series, committed: pd.Series) -> pd.Series:
    return (refit - committed).abs() / committed.abs()


def assemble(
    records: Sequence[dict[str, Any]],
    committed: pd.DataFrame,
    chosen: Sequence[int],
    compared: Sequence[int],
) -> pd.DataFrame:
    """The refit records beside the committed rows, in the published column order."""
    frame = pd.DataFrame(list(records)).drop(columns="seconds")
    keys = committed.loc[:, ["replicate", "cell", "estimate", "std_error"]].rename(
        columns={"estimate": "committed_estimate", "std_error": "committed_std_error"}
    )
    frame = frame.merge(keys, on=["replicate", "cell"], how="left", validate="many_to_one")
    if frame[["committed_estimate", "committed_std_error"]].isna().any().any():
        raise RuntimeError("a refit has no committed row")
    frame["estimate_relative_difference"] = _relative(
        frame["refit_estimate"], frame["committed_estimate"]
    )
    frame["std_error_relative_difference"] = _relative(
        frame["refit_std_error"], frame["committed_std_error"]
    )
    group = dict.fromkeys(chosen, SELECTED)
    group.update(dict.fromkeys(compared, COMPARISON))
    frame["group"] = frame["replicate"].map(group)
    frame["cell"] = pd.Categorical(frame["cell"], categories=list(ARMS), ordered=True)
    frame = frame.sort_values(
        ["group", "replicate", "cell", "regimen"], ascending=[False, True, True, True]
    )
    frame["cell"] = frame["cell"].astype(str)
    frame = frame.loc[:, list(REFIT_COLUMNS)].reset_index(drop=True)
    validate_refit_frame(frame, require_declared=True)
    return frame


def validate_refit_frame(frame: pd.DataFrame, *, require_declared: bool = False) -> None:
    """Refuse an incomplete, duplicated, or non-finite refit table before reading it."""
    missing_columns = sorted(set(REFIT_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise RuntimeError(f"the refit table lacks columns {missing_columns}")
    groups = set(frame["group"])
    if groups != {SELECTED, COMPARISON}:
        raise RuntimeError(f"the refit table has unexpected groups {sorted(groups)}")
    if require_declared:
        committed = learner_weight_rows()
        chosen = selected(committed)
        compared = comparison(committed, chosen)
        group_replicates = ((SELECTED, chosen), (COMPARISON, compared))
    else:
        group_replicates = tuple(
            (group, tuple(sorted(set(frame.loc[frame["group"] == group, "replicate"]))))
            for group in (SELECTED, COMPARISON)
        )
    expected = {
        (group, replicate, cell, regimen)
        for group, replicates in group_replicates
        for replicate in replicates
        for cell in ARMS
        for regimen in common.REGIMENS
    }
    key_columns = ["group", "replicate", "cell", "regimen"]
    keys = [tuple(row) for row in frame[key_columns].itertuples(index=False, name=None)]
    observed = set(keys)
    if len(keys) != len(observed) or observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise RuntimeError(f"the refit table is incomplete; missing={missing}, extra={extra}")
    numeric = frame.loc[:, [column for column in REFIT_COLUMNS if column not in key_columns]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise RuntimeError("the refit table contains a non-finite statistic")


def reproduced(frame: pd.DataFrame) -> bool:
    """The reproduction control: every refit within the tolerance on both columns."""
    differences = frame[["estimate_relative_difference", "std_error_relative_difference"]]
    return bool((differences <= REPRODUCTION_TOLERANCE).all().all())


def _contrast(frame: pd.DataFrame, group: str, regimens: Sequence[str]) -> pd.DataFrame:
    return frame.loc[(frame["group"] == group) & frame["regimen"].isin(regimens)]


def _per_arm(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby(["replicate", "cell"])[column].sum()


def _no_floor(frame: pd.DataFrame, regimens: Sequence[str]) -> tuple[bool, str]:
    floored = _per_arm(_contrast(frame, COMPARISON, regimens), "floored_followers")
    return (
        bool((floored == 0).all()),
        f"comparison replicate and arm pairs with a floored follower: "
        f"{int((floored > 0).sum())} of {len(floored)}",
    )


def _bounded_weight(frame: pd.DataFrame, regimens: Sequence[str]) -> tuple[bool, str]:
    weights = _contrast(frame, COMPARISON, regimens)["max_weight"]
    return (
        bool((weights < MAX_WEIGHT_THRESHOLD).all()),
        f"comparison max_weight: min {weights.min():.6f}, max {weights.max():.6f}; "
        f"at or above {MAX_WEIGHT_THRESHOLD:g}: "
        f"{int((weights >= MAX_WEIGHT_THRESHOLD).sum())} of {len(weights)}",
    )


def predictions(frame: pd.DataFrame) -> dict[str, tuple[bool, str]]:
    """Each declared prediction: whether it holds, and its values.

    P1 to P4 read the contrast regimens.  P5 reads every fitted regimen.
    """
    chosen = _contrast(frame, SELECTED, CONTRAST_REGIMENS)
    floored = _per_arm(chosen, "floored_followers")
    shares = chosen.groupby(["replicate", "cell"])["floored_share"].first()
    mismatched = chosen.loc[chosen["floored_followers"] != chosen["zero_prefix_followers"]]
    return {
        "P1": (
            bool((floored >= 1).all()),
            f"floored followers per selected replicate and arm: min {int(floored.min())}, "
            f"max {int(floored.max())}; arms with none: {int((floored < 1).sum())} "
            f"of {len(floored)}",
        ),
        "P2": _no_floor(frame, CONTRAST_REGIMENS),
        "P3": (
            bool((shares >= SHARE_THRESHOLD).all()),
            f"floored share per selected replicate and arm: min {shares.min():.6f}, "
            f"max {shares.max():.6f}; below {SHARE_THRESHOLD:g}: "
            f"{int((shares < SHARE_THRESHOLD).sum())} of {len(shares)}",
        ),
        "P4": (
            mismatched.empty,
            f"selected regimen fits with a floored follower whose raw prefix is not zero: "
            f"{len(mismatched)} of {len(chosen)}",
        ),
        # The declaration: "in every comparison replicate, arm and regimen".
        "P5": _bounded_weight(frame, _every(frame)),
    }


def _every(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(sorted(set(frame["regimen"])))


def supplementary(frame: pd.DataFrame) -> dict[str, tuple[bool, str]]:
    """P2 over every fitted regimen, and P5 over the contrast regimens.  Neither is read."""
    return {
        "P2 over every fitted regimen, not read": _no_floor(frame, _every(frame)),
        "P5 over the contrast regimens, not read": _bounded_weight(frame, CONTRAST_REGIMENS),
    }


def reading(control: bool, read: dict[str, tuple[bool, str]]) -> str:
    """The declared reading rule.  P2 and P5 do not enter it."""
    if not control:
        return NOT_VALIDATED
    if not read["P1"][0]:
        return NOT_THE_FLOOR
    if not read["P3"][0]:
        return UNRESOLVED
    return POSITIVITY if read["P4"][0] else FLOOR_NOT_FIT


def _control_detail(frame: pd.DataFrame) -> str:
    columns = ["estimate_relative_difference", "std_error_relative_difference"]
    per_refit = frame.groupby(["replicate", "cell"])[columns].max()
    outside = int((per_refit > REPRODUCTION_TOLERANCE).any(axis=1).sum())
    return (
        f"largest relative difference: estimate {per_refit[columns[0]].max():.3e}, "
        f"std_error {per_refit[columns[1]].max():.3e}; tolerance {REPRODUCTION_TOLERANCE:g}; "
        f"refits outside it: {outside} of {len(per_refit)}"
    )


def reading_table(frame: pd.DataFrame) -> pd.DataFrame:
    """The reproduction control, the five predictions and the reading, one row each."""
    validate_refit_frame(frame)
    control = reproduced(frame)
    rows = [
        {
            "item": "reproduction control",
            "result": HOLDS if control else FAILS,
            "detail": _control_detail(frame),
        }
    ]
    read = predictions(frame)
    for name, (holds, detail) in (*read.items(), *supplementary(frame).items()):
        rows.append(
            {
                "item": name,
                "result": (HOLDS if holds else FAILS) if control else NOT_READ,
                "detail": detail if control else "",
            }
        )
    rows.append({"item": "reading", "result": reading(control, read), "detail": ""})
    return pd.DataFrame(rows, columns=["item", "result", "detail"])


def refit_all(calls: Sequence[Payload], jobs: int) -> list[dict[str, Any]]:
    """Every refit in one pool, with each payload passed whole.

    ``map_parallel`` splats a tuple into positional arguments, and a payload is a tuple, so
    each one travels inside a one-element tuple.
    """
    outcomes = map_parallel(refit_one, [(payload,) for payload in calls], n_jobs=jobs)
    return [record for outcome in outcomes for record in outcome]


def run(jobs: int) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    committed = learner_weight_rows()
    chosen = selected(committed)
    compared = comparison(committed, chosen)
    records = refit_all(payloads((*chosen, *compared)), jobs)
    return assemble(records, committed, chosen, compared), records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=f"a directory for refit.csv and reading.csv; the committed record is {HERE}",
    )
    parser.add_argument("--jobs", type=int, default=1, help="workers in the one process pool")
    parser.add_argument(
        "--read-only", action="store_true", help="read an existing refit.csv; fit nothing"
    )
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    if arguments.read_only:
        frame = pd.read_csv(arguments.output / "refit.csv")
    else:
        started = time.perf_counter()
        frame, records = run(arguments.jobs)
        wall = time.perf_counter() - started
        write_csv(frame, arguments.output / "refit.csv")
        seconds = pd.DataFrame(records).groupby(["replicate", "cell"])["seconds"].first()
        print(
            f"refits: {len(seconds)}; wall seconds {wall:.1f}; per-refit seconds "
            f"min {seconds.min():.1f}, median {seconds.median():.1f}, max {seconds.max():.1f}, "
            f"sum {seconds.sum():.1f}"
        )
    validate_refit_frame(frame, require_declared=True)
    table = reading_table(frame)
    write_csv(table, arguments.output / "reading.csv")
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(frame.to_string(index=False))
        print(table.to_string(index=False))
    print(f"wrote {len(frame)} refit rows and {len(table)} reading rows to {arguments.output}")


if __name__ == "__main__":
    main()
