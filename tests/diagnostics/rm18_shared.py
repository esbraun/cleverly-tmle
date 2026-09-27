"""What the four RM18 follow-up diagnostics share.

``docs/roadmap.md`` declares the designs under RM18, in "The five follow-up designs, declared
before they run".  Its rules R2 (fresh streams), R4 (harness validation), R5 (intervals) and R6
(one run), and its closing rules for failed fits, seed collisions, bootstrap streams and two-arm
differences, read the same way in every diagnostic.  This module holds that reading once,
together with the weighted longitudinal pieces that more than one design fits, and the one
command-line driver every diagnostic runs through.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import importlib.metadata
import os
import shlex
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
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
from tests.studies.evidence.manifest import ROOT, UNKNOWN, _git, provenance, write_csv
from tests.studies.evidence.properties import ratio_draws, replicate_row
from tests.studies.evidence.registry import StudyRecord
from tests.studies.evidence.seeds import replicate_seed, stream_seed

#: R4: ``abs(new - committed) <= TOLERANCE * max(1, abs(committed))``.
TOLERANCE = 1e-9
#: R5: every interval is 99%.
CONFIDENCE = 0.99
NOT_VALIDATED = "harness not validated, no reading"
#: The result of every reading row of a run capped with ``--replicates``.
SMOKE = "smoke run, not the declared budget"
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


def optional_rows(path: Path) -> pd.DataFrame:
    """The rows at ``path``, or an empty frame when a part stopped before it drew any."""
    return read_rows(path) if path.exists() else pd.DataFrame()


def scaled_difference(new: Any, committed: Any) -> np.ndarray:
    """``abs(new - committed) / max(1, abs(committed))``, elementwise (R4)."""
    new_values = np.asarray(new, dtype=float)
    old_values = np.asarray(committed, dtype=float)
    return np.abs(new_values - old_values) / np.maximum(1.0, np.abs(old_values))


# ------------------------------------------------------------------------------ seeds (R2)


def fresh_seeds(
    record: StudyRecord, labels: Sequence[tuple[Any, ...]], taken: set[int]
) -> list[int]:
    """The fresh seed of each label, in order, under the declared seed-collision rule.

    ``taken`` holds every seed the label may not reuse: the study's registered seeds and the
    seeds of the parts earlier in the declared assignment order.  A seed in it moves to the seed
    of the same label followed by ``"retry"`` and the smallest counter that leaves it.  Each
    assigned seed joins ``taken``, so a later label and a later part see it.
    """
    out: list[int] = []
    for label in labels:
        seed = stream_seed(record, *label)
        counter = 0
        while seed in taken:
            counter += 1
            seed = stream_seed(record, *label, "retry", counter)
        taken.add(seed)
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


# ------------------------------------------------------------------ fits and validation (R4)


def pool(function: Callable[..., Any], payloads: Iterable[Any], jobs: int) -> list[Any]:
    """One process pool, each payload passed whole.

    ``map_parallel`` splats a tuple into positional arguments, so each payload travels inside
    a one-element tuple.
    """
    return map_parallel(function, [(payload,) for payload in payloads], n_jobs=jobs)


def require_finite(
    rows: pd.DataFrame, columns: Sequence[str] = ("estimate", "std_error")
) -> pd.DataFrame:
    """Stop the part on a fit that returned a non-finite estimate or standard error.

    The declared failed-fit rule: such a fit stops the part, which draws no replacement.
    """
    values = rows.loc[:, list(columns)].to_numpy(dtype=float)
    bad = ~np.isfinite(values).all(axis=1)
    if bad.any():
        raise RuntimeError(
            f"{int(bad.sum())} fresh rows carry a non-finite {' or '.join(columns)}; first "
            f"{rows.loc[bad].head(3).to_dict('records')}. A failed fit stops the part"
        )
    return rows


def compare_rows(
    refit: pd.DataFrame,
    committed: pd.DataFrame,
    keys: Sequence[str],
    columns: Sequence[str] = ("estimate", "std_error"),
) -> tuple[bool, float, int]:
    """Whether every refit row reproduces its committed row, the largest difference, the count.

    The two frames must hold the same keys, each once.  A structural miss raises, which ends
    the run with a nonzero exit and no reading.
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
    of the rows it covers, and the study's ``summarize_properties`` runs once on the result.
    Every numeric column of each named ``(property, cell)`` row must then match
    ``properties.csv`` to the R4 tolerance, and every other column exactly.
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
    """One :data:`VALIDATION_COLUMNS` row from a check's ``(held, largest, compared)``."""
    held, largest, compared = outcome
    return {
        "part": part,
        "check": check,
        "compared": compared,
        "largest_difference": largest,
        "result": HOLDS if held else FAILS,
    }


def validation_frame(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=list(VALIDATION_COLUMNS))


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


# ------------------------------------------------------------------------------- readings


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


def label(result: str, smoke: bool) -> str:
    """A reading's result, or :data:`SMOKE` when the rows fall short of the declared budget."""
    return SMOKE if smoke else result


def reading_frame(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=list(READING_COLUMNS))


def interval_of(draws: np.ndarray) -> Interval:
    """The 99% percentile interval of bootstrap draws."""
    return percentile_interval(draws, confidence_level=CONFIDENCE)


def difference_interval(left: np.ndarray, right: np.ndarray) -> Interval:
    """The 99% interval of a difference between two independent arms, draw by draw."""
    return interval_of(np.asarray(left) - np.asarray(right))


def rms_se_ratio(group: pd.DataFrame) -> float:
    """Root-mean-square reported SE over the empirical SD: supplementary, read by no rule."""
    errors = group["std_error"].to_numpy(dtype=float)
    return float(np.sqrt(np.mean(errors**2)) / group["estimate"].std(ddof=1))


# ------------------------------------------------------------- weighted longitudinal pieces

#: The weighted ladder arms: W is the registered selected law with weights ``1/pi(W)``, and U
#: its unweighted twin, drawn with selection probability 1 and weight 1.
ARM_SELECTION = {"W": weighted.SELECTION, "U": np.ones_like(weighted.SELECTION)}


def ladder_replicate(
    payload: tuple[StudyRecord, bool, str, bool, str, int, int, int, int],
) -> dict[str, Any]:
    """One ladder replicate: draw, fit the registered configuration, read the static contrast.

    ``payload`` is ``(record, cross_fit, part, null, arm, n, replicate, requested, seed)``.
    ``null`` draws the sharp-null law of ``ltmle_properties`` instead of the selected law.
    """
    record, cross_fit, part, null, arm, n, index, requested, seed = payload
    if null:
        from tests.studies.ltmle_properties import NULL_PROBS, NULL_TRUTH

        probs, truth = NULL_PROBS, float(NULL_TRUTH)
    else:
        probs, truth = law.PROBS, float(law.TRUTH[STATIC])
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


def ladder_labels(
    prefix: tuple[str, ...], arms: Sequence[str], sizes: Sequence[int], replicates: int
) -> list[tuple[Any, ...]]:
    """A ladder's declared seed labels ``(*prefix, arm, n, replicate)``, in assignment order."""
    return [(*prefix, arm, n, index) for arm in arms for n in sizes for index in range(replicates)]


def ladder_payloads(
    record: StudyRecord,
    *,
    cross_fit: bool,
    part: str,
    null: bool,
    labels: Sequence[tuple[Any, ...]],
    seeds: Sequence[int],
    replicates: int,
    cap: int | None,
) -> list[tuple[Any, ...]]:
    """One :func:`ladder_replicate` payload per fresh draw, the first ``cap`` of each rung."""
    return [
        (
            record,
            cross_fit,
            part,
            null,
            label[-3],
            label[-2],
            label[-1],
            budget(replicates, cap),
            seed,
        )
        for label, seed in zip(labels, seeds, strict=True)
        if label[-1] < budget(replicates, cap)
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


def ladder_table(
    rows: pd.DataFrame,
    record: StudyRecord,
    *,
    design: str,
    part: str,
    arms: Sequence[str],
    sizes: Sequence[int],
    bounds: dict[str, float] | None,
    declared: int,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]], bool]:
    """Every arm and size of one ladder, what each reading needs, and whether it is a smoke run.

    A rung with fewer rows than ``declared`` marks the run as a smoke run.
    """
    out: list[dict[str, Any]] = []
    kept: dict[tuple[str, int], dict[str, Any]] = {}
    smoke = False
    for arm in arms:
        for n in sizes:
            group = rows.loc[(rows["arm"] == arm) & (rows["n"] == n)]
            smoke = smoke or len(group) != declared
            statistics, points, draws = ladder_statistics(
                group,
                record,
                design=design,
                part=part,
                bound=None if bounds is None else bounds[arm],
            )
            out.extend(statistics)
            kept[arm, n] = {"points": points, "draws": draws, "rows": statistics}
    return out, kept, smoke


def row_interval(rows: Sequence[dict[str, Any]], statistic: str) -> Interval:
    """The interval of the one reading row that carries ``statistic``."""
    row = next(row for row in rows if row["statistic"] == statistic)
    return Interval(float(row["ci_lower"]), float(row["ci_upper"]))


def weighted_draw(payload: tuple[StudyRecord, bool, tuple[Any, ...]]) -> list[dict[str, Any]]:
    """One registered-shape weighted payload through ``_fit_replication``, each row with its seed.

    Both the R4 refit of registered payloads and the fresh OW-C and BD-3 draws go through it.
    """
    record, cross_fit, call = payload
    return [
        {**row, "seed": int(call[5])} for row in weighted._fit_replication(record, cross_fit, call)
    ]


def validate_weighted(
    record: StudyRecord,
    *,
    cross_fit: bool,
    part: str,
    payload: tuple[str, str],
    cells: Sequence[tuple[str, str]],
    cap: int | None,
    jobs: int,
) -> list[dict[str, Any]]:
    """R4 on the committed replicates of one registered weighted payload family.

    ``payload`` names the ``(property, suffix)`` of the registered ``_payloads`` entries to
    refit, and ``cells`` the summary rows the study's own summary must reproduce.
    """
    committed = read_rows(record.artifact("property-replicates.csv.gz"))
    calls = [call for call in weighted._payloads(record) if (call[0], call[1]) == payload]
    calls = calls[: budget(len(calls), cap)]
    fitted = pool(weighted_draw, [(record, cross_fit, call) for call in calls], jobs)
    refit = pd.DataFrame([row for rows in fitted for row in rows])
    keys = ["property", "cell", "replicate"]
    selection = committed.merge(refit[keys], on=keys)
    return [
        validation_row(part, "refit rows", compare_rows(refit, selection, keys)),
        validation_row(part, "summary rows", summary_check(record, committed, refit, cells)),
    ]


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


# --------------------------------------------------------------- the run guard and record


def refusals(smoke: bool, pushed: Sequence[str] = ()) -> list[str]:
    """Why a run may not start, if it may not (R6).

    A declared run needs ``cleverly`` imported from this tree's ``src``, a clean tree, and a
    ``HEAD`` equal to its upstream.  ``pushed`` names repository files the upstream must hold,
    such as the BD-P pilot that step 2 reads.  A smoke run is refused nothing.
    """
    if smoke:
        return []
    import cleverly

    out = []
    source = Path(cleverly.__file__).resolve()
    if not source.is_relative_to((ROOT / "src").resolve()):
        out.append(f"cleverly is imported from {source}, not from {ROOT / 'src'}")
    status = _git("status", "--porcelain")
    if status:
        out.append(f"the tree has changes: {status.splitlines()[:5]}")
    head, upstream = _git("rev-parse", "HEAD"), _git("rev-parse", "@{u}")
    if head == UNKNOWN or head != upstream:
        out.append(f"HEAD {head} is not its pushed upstream {upstream}")
    out.extend(
        f"{name} is not in the pushed upstream"
        for name in pushed
        if _git("cat-file", "-e", f"@{{u}}:{name}") == UNKNOWN
    )
    return out


def _installed_digest() -> tuple[str, int]:
    lines = sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
    )
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest(), len(lines)


THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


#: Lines a part asks :func:`run_log` to record in its block, as ``note: <text>``.
_NOTES: list[str] = []


def note(text: str) -> None:
    """Record one line in the ``run.log`` block of the running part."""
    _NOTES.append(text)


@contextlib.contextmanager
def run_log(output: Path, title: str) -> Iterator[None]:
    """Append one block to ``output/run.log``: the code, the runtime, the wall time, the exit.

    R6 names what it holds.  The block also records the SHA-256 of every file the run left in
    ``output``; ``manifest.hashes`` keys its digests by a path under the repository, and the
    output directory sits outside it.
    """
    import joblib

    import cleverly

    digest, packages = _installed_digest()
    record = provenance()
    started = datetime.datetime.now(datetime.UTC)
    lines = [
        f"=== {title}",
        f"command: {shlex.join([sys.executable, *sys.argv])}",
        f"commit: {record['cleverly_commit']}; upstream {_git('rev-parse', '@{u}')}",
        f"clean tree: {record['cleverly_worktree_clean']}",
        f"cleverly {record['cleverly_version']} from {cleverly.__file__}",
        f"python {record['python']}; numpy {record['numpy']}; scipy {record['scipy']}; "
        f"pandas {record['pandas']}; scikit-learn {record['scikit_learn']}; joblib {joblib.__version__}",
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
        lines += [f"note: {text}" for text in _NOTES]
        _NOTES.clear()
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


def as_committed(frame: pd.DataFrame) -> pd.DataFrame:
    """``frame`` as its committed file parses, through :func:`write_table` and :func:`read_rows`.

    A rebuilt reading and a committed one compare like with like only through the same writer
    and reader; an empty ``result`` label, for example, reads back as a missing value.
    """
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "reading.csv"
        write_table(frame, path)
        return read_rows(path)


def budget(declared: int, cap: int | None) -> int:
    """The declared budget, or the smoke cap when one is given."""
    return declared if cap is None else min(declared, cap)


def part_paths(output: Path, part: str) -> tuple[Path, Path, Path]:
    """The rows, validation and reading files of one part, named after it."""
    stem = part.lower()
    return (
        output / f"{stem}-rows.csv.gz",
        output / f"{stem}-validation.csv",
        output / f"{stem}-reading.csv",
    )


def main(
    description: str,
    parts: Sequence[str],
    here: Path,
    run_part: Callable[[str, Path, int | None, int], None],
    table: Callable[[str, Path, int], pd.DataFrame],
    pushed: Callable[[str], Sequence[str]] = lambda part: (),
) -> None:
    """The command line every RM18 follow-up diagnostic runs through.

    ``--output`` is required, so a bare run cannot overwrite the committed record.  ``--part``
    picks one declared part.  ``--replicates`` caps every budget for a disposable smoke run.
    ``--read-only`` rebuilds the part's reading from what ``--output`` holds and fits nothing.
    A run that is neither passes :func:`refusals` first.  ``run_part`` writes the part's rows
    and validation record, and ``table`` builds its reading.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=f"a directory for the part's files; the committed record is {here}",
    )
    parser.add_argument("--part", choices=parts, required=True)
    parser.add_argument("--read-only", action="store_true", help="fit nothing; rebuild the reading")
    parser.add_argument(
        "--replicates", type=int, help="cap every budget at this many replicates (smoke only)"
    )
    parser.add_argument("--jobs", type=int, default=available_cores())
    arguments = parser.parse_args()
    output, part = arguments.output, arguments.part
    _, _, reading_path = part_paths(output, part)
    if not arguments.read_only:
        refused = refusals(arguments.replicates is not None, pushed(part))
        if refused:
            parser.exit(2, "refused:\n  " + "\n  ".join(refused) + "\n")
        output.mkdir(parents=True, exist_ok=True)
        with run_log(output, f"{part}, cap {arguments.replicates}"):
            run_part(part, output, arguments.replicates, arguments.jobs)
            write_table(table(part, output, arguments.jobs), reading_path)
    else:
        write_table(table(part, output, arguments.jobs), reading_path)
    with pd.option_context("display.width", 220, "display.max_rows", None):
        print(pd.read_csv(reading_path).to_string(index=False))
