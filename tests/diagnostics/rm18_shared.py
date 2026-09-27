"""What the four RM18 follow-up diagnostics share.

``docs/roadmap.md`` declares the designs under RM18, in "The five follow-up designs, declared
before they run".  Its rules R2 (fresh streams), R4 (harness validation) and R5 (intervals), and
its closing rules for bootstrap streams, two-arm differences and seed collisions, read the same
way in every diagnostic.  This module holds that reading once, together with the two weighted
longitudinal pieces that more than one design fits: the ladder replicate of FW-A, OW-A and OW-B,
and the exact-law frame of OW-C and BD-3.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import importlib.metadata
import os
import platform
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cleverly.utils.parallel import map_parallel
from tests import discrete_law_longitudinal as law
from tests.parallel import available_cores
from tests.studies import weighted_longitudinal_common as primary
from tests.studies import weighted_longitudinal_properties_common as weighted
from tests.studies.evidence.inference import Interval, clopper_pearson, percentile_interval
from tests.studies.evidence.manifest import ROOT, write_csv
from tests.studies.evidence.properties import ratio_draws, replicate_row
from tests.studies.evidence.registry import StudyRecord
from tests.studies.evidence.seeds import replicate_seed, stream_seed

#: R4: ``abs(new - committed) <= TOLERANCE * max(1, abs(committed))``.
TOLERANCE = 1e-9
#: R5: every interval is 99%.
CONFIDENCE = 0.99
NOT_VALIDATED = "harness not validated, no reading"
HOLDS = "holds"
FAILS = "fails"

#: One published reading row.  ``value`` is a point statistic, the interval is its 99%
#: interval where it has one, and ``result`` carries a label or a verdict.
READING_COLUMNS = ("part", "scope", "statistic", "value", "ci_lower", "ci_upper", "result")
#: One harness-validation record, written by the run and read by ``--read-only``.
VALIDATION_COLUMNS = ("part", "check", "compared", "largest_difference", "result")

#: The static contrast every weighted design reads.
STATIC = weighted.CONTRASTS["static"]


def read_rows(path: Path) -> pd.DataFrame:
    """A committed table, parsed so that every float round-trips (R4)."""
    return pd.read_csv(path, float_precision="round_trip")


def scaled_difference(new: Any, committed: Any) -> np.ndarray:
    """``abs(new - committed) / max(1, abs(committed))``, elementwise (R4)."""
    new_values = np.asarray(new, dtype=float)
    old_values = np.asarray(committed, dtype=float)
    return np.abs(new_values - old_values) / np.maximum(1.0, np.abs(old_values))


# ------------------------------------------------------------------------------ seeds (R2)


def fresh_seeds(
    record: StudyRecord, labels: Sequence[tuple[Any, ...]], registered: set[int]
) -> list[int]:
    """The fresh seed of each label, in order, under the declared seed-collision rule.

    A seed that equals a registered seed of the study, or a seed this call assigned earlier,
    moves to the seed of the same label followed by ``"retry"`` and the smallest counter that
    clears both sets.
    """
    assigned: set[int] = set()
    out: list[int] = []
    for label in labels:
        seed = stream_seed(record, *label)
        counter = 0
        while seed in registered or seed in assigned:
            counter += 1
            seed = stream_seed(record, *label, "retry", counter)
        assigned.add(seed)
        out.append(seed)
    return out


def weighted_registered_seeds(record: StudyRecord) -> set[int]:
    """Every sample seed a registered weighted longitudinal study drew with."""
    seeds = {int(payload[5]) for payload in weighted._payloads(record)}
    seeds.update(
        replicate_seed(record, primary.SCENARIO, index) for index in range(record.replicates)
    )
    return seeds


def bootstrap_seed(record: StudyRecord, design: str, label: str) -> int:
    """A diagnostic bootstrap seed: ``stream_seed(record, "rm18", design, "bootstrap", label)``."""
    return stream_seed(record, "rm18", design, "bootstrap", label)


def bootstrap_record(record: StudyRecord, design: str, label: str) -> StudyRecord:
    """The record a framework function derives its own streams from, for one diagnostic label."""
    return replace(record, resampling_seed=bootstrap_seed(record, design, label))


# ------------------------------------------------------------------ fits and validation (R4)


def pool(function: Callable[..., Any], payloads: Iterable[Any], jobs: int) -> list[Any]:
    """One process pool, each payload passed whole.

    ``map_parallel`` splats a tuple into positional arguments, so each payload travels inside
    a one-element tuple.
    """
    return map_parallel(function, [(payload,) for payload in payloads], n_jobs=jobs)


def compare_rows(
    refit: pd.DataFrame,
    committed: pd.DataFrame,
    keys: Sequence[str],
    columns: Sequence[str] = ("estimate", "std_error"),
) -> tuple[bool, float, int]:
    """Whether every refit row reproduces its committed row, the largest difference, the count.

    The two frames must hold the same keys, each once.
    """
    left = refit.loc[:, [*keys, *columns]]
    right = committed.loc[:, [*keys, *columns]]
    if left.duplicated(list(keys)).any() or right.duplicated(list(keys)).any():
        raise RuntimeError("a refit or committed key appears twice")
    merged = left.merge(right, on=list(keys), how="outer", suffixes=("", "_committed"))
    if len(merged) != len(left) or len(merged) != len(right):
        raise RuntimeError(
            f"the refit holds {len(left)} rows and the committed selection {len(right)}; "
            f"they share {len(left.merge(right, on=list(keys)))} keys"
        )
    largest = max(
        float(np.max(scaled_difference(merged[column], merged[f"{column}_committed"])))
        for column in columns
    )
    return bool(largest <= TOLERANCE), largest, len(merged)


def summary_check(
    record: StudyRecord,
    committed: pd.DataFrame,
    refit: pd.DataFrame,
    cells: Sequence[tuple[str, str]],
) -> tuple[bool, float, int]:
    """Whether the study's own summary, over the refit rows, reproduces the committed rows.

    The refit replaces the committed ``estimate``, ``std_error``, ``covered`` and ``rejected``
    of the rows it covers, and the study's ``summarize_properties`` runs on the result.  Every
    numeric column of each named ``(property, cell)`` row must then match ``properties.csv``
    to the R4 tolerance, and every other column exactly.
    """
    keys = ["property", "cell", "replicate"]
    columns = ["estimate", "std_error", "covered", "rejected"]
    rows = committed.copy()
    patch = refit.set_index(keys)[columns]
    index = rows.set_index(keys).index
    mask = index.isin(patch.index)
    if int(mask.sum()) != len(patch):
        raise RuntimeError("a refit row has no committed property row")
    rows.loc[mask, columns] = patch.loc[index[mask], columns].to_numpy()
    summary = record.properties().summarize_properties(rows).set_index(["property", "cell"])
    published = read_rows(record.artifact("properties.csv")).set_index(["property", "cell"])
    largest = 0.0
    same = True
    for cell in cells:
        new, old = summary.loc[cell], published.loc[cell]
        for column in published.columns:
            before, after = old[column], new[column]
            if pd.isna(before) and pd.isna(after):
                continue
            if isinstance(before, (bool, np.bool_, str)) or isinstance(after, (bool, np.bool_)):
                same = same and bool(before == after)
                continue
            difference = float(scaled_difference(after, before))
            largest = max(largest, difference if np.isfinite(difference) else np.inf)
    return bool(same and largest <= TOLERANCE), largest, len(cells)


def validation_row(part: str, check: str, outcome: tuple[bool, float, int]) -> dict[str, Any]:
    """One :data:`VALIDATION_COLUMNS` row from a :func:`compare_rows` or
    :func:`summary_check` outcome."""
    held, largest, compared = outcome
    return {
        "part": part,
        "check": check,
        "compared": compared,
        "largest_difference": largest,
        "result": HOLDS if held else FAILS,
    }


def validated(validation: pd.DataFrame, part: str) -> bool:
    """Whether every harness check of one part held."""
    rows = validation.loc[validation["part"] == part]
    return bool(len(rows)) and bool((rows["result"] == HOLDS).all())


def validation_readings(validation: pd.DataFrame, part: str) -> list[dict[str, Any]]:
    """The harness checks of one part, as reading rows."""
    return [
        reading(
            part,
            str(row.check),
            "largest scaled difference",
            value=float(row.largest_difference),
            result=f"{row.result} over {int(row.compared)}",
        )
        for row in validation.loc[validation["part"] == part].itertuples(index=False)
    ]


def reading(
    part: str,
    scope: str,
    statistic: str,
    *,
    value: float = float("nan"),
    interval: Interval | None = None,
    result: str = "",
) -> dict[str, Any]:
    """One :data:`READING_COLUMNS` row."""
    return {
        "part": part,
        "scope": scope,
        "statistic": statistic,
        "value": value,
        "ci_lower": np.nan if interval is None else interval.low,
        "ci_upper": np.nan if interval is None else interval.high,
        "result": result,
    }


def reading_frame(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=list(READING_COLUMNS))


def difference_interval(left: np.ndarray, right: np.ndarray) -> Interval:
    """The 99% interval of a difference between two independent arms, draw by draw."""
    return percentile_interval(np.asarray(left) - np.asarray(right), confidence_level=CONFIDENCE)


def rms_se_ratio(group: pd.DataFrame) -> float:
    """Root-mean-square reported SE over the empirical SD: supplementary, read by no rule."""
    errors = group["std_error"].to_numpy(dtype=float)
    return float(np.sqrt(np.mean(errors**2)) / group["estimate"].std(ddof=1))


# ------------------------------------------------------------- weighted longitudinal pieces


#: The weighted ladder arms: W is the registered selected law with weights ``1/pi(W)``, and U
#: its unweighted twin, drawn with selection probability 1 and weight 1.
ARM_SELECTION = {"W": weighted.SELECTION, "U": np.ones_like(weighted.SELECTION)}
#: The laws a ladder arm draws from, by name.
LADDER_LAWS = {"selected": (law.PROBS, float(law.TRUTH[STATIC]))}


def _null_law() -> tuple[np.ndarray, float]:
    from tests.studies.ltmle_properties import NULL_PROBS, NULL_TRUTH

    return NULL_PROBS, float(NULL_TRUTH)


def ladder_replicate(
    payload: tuple[StudyRecord, bool, str, str, str, int, int, int, int],
) -> dict[str, Any]:
    """One ladder replicate: draw, fit the registered configuration, read the static contrast.

    ``payload`` is ``(record, cross_fit, part, law, arm, n, replicate, requested, seed)``.
    """
    record, cross_fit, part, law_name, arm, n, index, requested, seed = payload
    probs, truth = _null_law() if law_name == "null" else LADDER_LAWS[law_name]
    frame = weighted.sample(probs, n, seed, selection=ARM_SELECTION[arm])
    result = weighted.fit(frame, "both_correct", cross_fit=cross_fit)
    row = replicate_row(
        property_name=part,
        cell=f"{arm}__n{n}",
        role="positive",
        replicate=index,
        n=n,
        requested=requested,
        truth=truth,
        estimate=result[STATIC],
        alpha=record.margins.alpha,
    )
    return {**row, "arm": arm, "seed": seed}


def ladder_payloads(
    record: StudyRecord,
    *,
    cross_fit: bool,
    part: str,
    law_name: str,
    prefix: tuple[str, ...],
    arms: Sequence[str],
    sizes: Sequence[int],
    replicates: int,
) -> list[tuple[Any, ...]]:
    """One :func:`ladder_replicate` payload per fresh draw of a declared ladder.

    The seed label is ``(*prefix, arm, n, replicate)``, assigned arm by arm, size by size and
    replicate by replicate under the collision rule.
    """
    order = [(*prefix, arm, n, index) for arm in arms for n in sizes for index in range(replicates)]
    seeds = fresh_seeds(record, order, weighted_registered_seeds(record))
    return [
        (record, cross_fit, part, law_name, label[-3], label[-2], label[-1], replicates, seed)
        for label, seed in zip(order, seeds, strict=True)
    ]


def ladder_statistics(
    group: pd.DataFrame, record: StudyRecord, *, design: str, part: str, bound: float | None
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, np.ndarray]]:
    """The declared statistics of one ladder arm and size, with its ratio points and draws.

    The bootstrap label is ``f"{arm}__n{n}"``.  Without a ``bound`` the two efficiency ratios
    are absent, and the rejection rate is reported in their place.
    """
    arm, n = str(group["arm"].iloc[0]), int(group["n"].iloc[0])
    scope = f"{arm}, n = {n}"
    draws = ratio_draws(
        group,
        replicates=record.margins.bootstrap_replicates,
        seed=bootstrap_seed(record, design, f"{arm}__n{n}"),
        bound=bound,
    )
    spread = float(group["estimate"].std(ddof=1))
    reported = float(group["std_error"].mean())
    points = {"se_ratio": reported / spread}
    names = {"se_ratio": "SE ratio"}
    if bound is not None:
        scale = float(np.sqrt(n)) / bound
        points["efficiency_empirical"] = spread * scale
        points["efficiency_reported"] = reported * scale
        names = {
            "efficiency_empirical": "empirical efficiency ratio",
            "efficiency_reported": "reported efficiency ratio",
            **names,
        }
    rows = [
        reading(part, scope, names[key], value=points[key], interval=interval_of(draws[key]))
        for key in names
    ]
    counts = [("coverage", "covered")]
    if bound is None:
        counts.append(("rejection rate", "rejected"))
    for statistic, column in counts:
        hits = int(group[column].sum())
        rows.append(
            reading(
                part,
                scope,
                statistic,
                value=hits / len(group),
                interval=clopper_pearson(hits, len(group), confidence_level=CONFIDENCE),
            )
        )
    rows.append(
        reading(part, scope, "RMS-SE ratio", value=rms_se_ratio(group), result="supplementary")
    )
    return rows, points, draws


def interval_of(draws: np.ndarray) -> Interval:
    """The 99% percentile interval of bootstrap draws."""
    return percentile_interval(draws, confidence_level=CONFIDENCE)


def validate_weighted(
    record: StudyRecord,
    *,
    cross_fit: bool,
    part: str,
    payload: tuple[str, str],
    cells: Sequence[tuple[str, str]],
    cap: int | None,
    jobs: int,
) -> pd.DataFrame:
    """R4 on the committed replicates of one registered weighted payload family.

    ``payload`` names the ``(property, suffix)`` of the registered ``_payloads`` entries to
    refit, and ``cells`` the summary rows the study's own summary must reproduce.
    """
    committed = read_rows(record.artifact("property-replicates.csv.gz"))
    calls = [call for call in weighted._payloads(record) if (call[0], call[1]) == payload]
    calls = calls[: budget(len(calls), cap)]
    fitted = pool(_registered_weighted, [(record, cross_fit, call) for call in calls], jobs)
    refit = pd.DataFrame([row for rows in fitted for row in rows])
    keys = ["property", "cell", "replicate"]
    selection = committed.merge(refit[keys], on=keys)
    return pd.DataFrame(
        [
            validation_row(part, "refit rows", compare_rows(refit, selection, keys)),
            validation_row(part, "summary rows", summary_check(record, committed, refit, cells)),
        ],
        columns=list(VALIDATION_COLUMNS),
    )


def _registered_weighted(
    payload: tuple[StudyRecord, bool, tuple[Any, ...]],
) -> list[dict[str, Any]]:
    record, cross_fit, call = payload
    return weighted._fit_replication(record, cross_fit, call)


def exact_selected_frame(*, use_weights: bool = True) -> pd.DataFrame:
    """A frame whose weighted empirical law is exactly the selected law with weights ``1/pi(W)``.

    ``law.frame()`` realises the unselected law.  Each of its rows enters in proportion to its
    selection probability, 0.3 when ``W > 0`` and 0.9 otherwise, so once or three times.
    ``use_weights=False`` sets every weight to one, which is the mutation a test uses.
    """
    base = law.frame()
    cells = np.repeat(np.arange(len(law.SUPPORT)), law.COUNTS)
    selection = weighted.SELECTION[cells]
    multiplicity = np.rint(selection / weighted.SELECTION_LOW).astype(int)
    realised = np.bincount(np.repeat(cells, multiplicity), minlength=len(law.SUPPORT))
    if not np.allclose(realised / realised.sum(), weighted.SELECTED_PROBS, rtol=0.0, atol=1e-15):
        raise AssertionError("the exact-law frame does not realise the selected law")
    frame = base.loc[np.repeat(np.arange(len(base)), multiplicity)].reset_index(drop=True)
    frame["obs_weight"] = (1.0 / np.repeat(selection, multiplicity)) if use_weights else 1.0
    return frame


def follower_mean(label: str, *, use_weights: bool = True) -> float:
    """The weighted mean outcome of the units that follow static regimen ``label``, longhand.

    Read off the support and ``SELECTED_PROBS`` alone: no learner, no fit and no frame.
    """
    arm = {"always": 1.0, "never": 0.0}[label]
    followed = np.array(
        [
            point[2] == 1 and point[1] == arm and point[5] == 1 and point[4] == arm
            for point in law.SUPPORT
        ]
    )
    outcome = np.array([0.0 if point[6] is None else float(point[6]) for point in law.SUPPORT])
    weights = weighted.OBS_WEIGHTS if use_weights else np.ones_like(weighted.OBS_WEIGHTS)
    mass = weighted.SELECTED_PROBS * weights * followed
    return float(np.sum(mass * outcome) / np.sum(mass))


def exact_untargeted(configuration: str, *, use_weights: bool = True) -> dict[str, float]:
    """The ordinary fit and the untargeted plug-in of each static regimen on the exact frame."""
    frame = exact_selected_frame(use_weights=use_weights)
    result = weighted.fit(frame, configuration, cross_fit=False)
    return {
        label: weighted.untargeted(frame, label, configuration, cross_fit=False, folds=result.folds)
        for label in ("always", "never")
    }


# --------------------------------------------------------------------------- the run record


def _installed_digest() -> tuple[str, int]:
    lines = sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
    )
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest(), len(lines)


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=ROOT, capture_output=True, text=True, check=False, timeout=30
    )
    return completed.stdout.strip()


THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@contextlib.contextmanager
def run_log(output: Path, title: str) -> Iterator[None]:
    """Append one block to ``output/run.log``: the code, the runtime, the wall time, the exit.

    R6 asks for the commit, the command, the installed-package digest, the thread limits, the
    wall time and the exit code.  The block also records the SHA-256 of every file the run
    left in ``output``.
    """
    import cleverly

    digest, packages = _installed_digest()
    started = datetime.datetime.now(datetime.UTC)
    lines = [
        f"=== {title}",
        f"command: {shlex.join([sys.executable, *sys.argv])}",
        f"commit: {_git('rev-parse', 'HEAD')}",
        f"clean tree: {_git('status', '--porcelain') == ''}",
        f"python {platform.python_version()}; cleverly {cleverly.__file__}",
        f"installed distributions: {packages}, sha256 {digest}",
        "threads: "
        + " ".join(f"{name}={os.environ.get(name, 'unset')}" for name in THREAD_VARIABLES),
        f"cores: {available_cores()}",
        f"started: {started.isoformat(timespec='seconds')}",
    ]
    clock = time.perf_counter()
    code = 1
    try:
        yield
        code = 0
    finally:
        finished = datetime.datetime.now(datetime.UTC)
        lines += [
            f"finished: {finished.isoformat(timespec='seconds')}",
            f"wall seconds: {time.perf_counter() - clock:.1f}",
            f"exit code: {code}",
        ]
        lines += [
            f"sha256 {path.name}: {hashlib.sha256(path.read_bytes()).hexdigest()}"
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "run.log"
        ]
        with (output / "run.log").open("a", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(lines) + "\n\n")


def write_table(frame: pd.DataFrame, path: Path) -> None:
    """Write one diagnostic table with LF endings, and a gzip one with ``mtime=0``."""
    compression = {"method": "gzip", "mtime": 0} if path.name.endswith(".gz") else None
    write_csv(frame, path, compression=compression)


def arguments(description: str, parts: Sequence[str], here: Path) -> argparse.Namespace:
    """The command line every RM18 follow-up diagnostic takes.

    ``--output`` is required, so a bare run cannot overwrite the committed record.
    ``--read-only`` rebuilds ``reading.csv`` from the rows and the validation record already in
    ``--output`` and fits nothing.  ``--replicates`` caps every budget, for a disposable smoke
    run only.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=f"a directory for the rows and reading.csv; the committed record is {here}",
    )
    parser.add_argument("--part", choices=parts, required=True)
    parser.add_argument("--read-only", action="store_true", help="fit nothing; rebuild reading.csv")
    parser.add_argument(
        "--replicates", type=int, help="cap every budget at this many replicates (smoke only)"
    )
    parser.add_argument("--jobs", type=int, default=available_cores())
    parsed = parser.parse_args()
    parsed.output.mkdir(parents=True, exist_ok=True)
    return parsed


def budget(declared: int, cap: int | None) -> int:
    """The declared budget, or the smoke cap when one is given."""
    return declared if cap is None else min(declared, cap)


def optional_rows(path: Path) -> pd.DataFrame:
    """The rows at ``path``, or an empty frame when a part stopped before it drew any."""
    return read_rows(path) if path.exists() else pd.DataFrame()


def show(path: Path) -> None:
    """Print one written table in full."""
    with pd.option_context("display.width", 220, "display.max_rows", None):
        print(pd.read_csv(path).to_string(index=False))


def part_paths(output: Path, part: str) -> tuple[Path, Path, Path]:
    """The rows, validation and reading files of one part, named after it."""
    stem = part.lower()
    return (
        output / f"{stem}-rows.csv.gz",
        output / f"{stem}-validation.csv",
        output / f"{stem}-reading.csv",
    )
