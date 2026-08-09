"""Writing the run out: ``results.jsonl``, ``results.csv``, ``summary.md``, ``environment.json``.

Three things the summary must do, and each of them is a decision rather than formatting.

**Report negatives as loudly as positives.**  The summary's per-kernel verdict is one of
*adopt serial* / *adopt parallel* / *retain numpy* / *fix the algorithm instead* / *defer,
share too small*, and the last three are the majority of them.  A suite whose report is a
list of wins cannot be used to check that the wins are real.

**Never present a row without the box it came from.**  Every row carries the git sha, the
CPU model and the core counts, and ``environment.json`` carries the rest.  Two runs from
different machines are two different measurements and the report says so rather than
letting a reader diff the numbers.

**Say what was not run.**  A skipped implementation (numba absent), a refused core count,
a failed correctness gate: each is a row with a reason, not a gap.  A gap reads as
"covered and equal".
"""

from __future__ import annotations

import csv
import dataclasses
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "MixedEnvironment",
    "Row",
    "fingerprint",
    "load_rows",
    "merge",
    "mixed_fields",
    "write_all",
]

#: The fields that identify *which box and which build* produced a row.  Not the core
#: counts or the thread plan: those are the axes a sweep deliberately varies, and a table
#: comparing four core counts is the point rather than a mixture.  These six are the ones
#: no sweep varies on purpose, so a set of rows that disagrees on any of them is two
#: measurements being presented as one.
FINGERPRINT: tuple[str, ...] = (
    "git_sha",
    "cpu_model",
    "python_version",
    "numpy_version",
    "numba_version",
    "blas_backend",
)


@dataclass
class Row:
    """One measurement.  The schema the plan specifies, plus what the summary needs."""

    scenario: str
    operation: str
    implementation: str
    n: int
    num_cores_requested: int
    #: Threads numba was actually running with, read *inside* the applied plan -- numba
    #: caps a request at ``NUMBA_NUM_THREADS`` rather than refusing it, and the point of
    #: the column is to catch that.  ``None`` on a skipped row, which entered no plan and
    #: so has no effective count: JSON writes ``null`` and the CSV an empty cell.
    num_cores_effective: int | None
    blas_threads: int
    numba_threads: int
    workers: int
    repeat_count: int
    warm_seconds: float
    warm_iqr_seconds: float
    warm_min_seconds: float
    warm_max_seconds: float
    cpu_seconds: float
    peak_rss_bytes: int
    rss_delta_bytes: int
    #: Per-call peak Python-level allocation. The memory column reports *this*, not
    #: `peak_rss_bytes`: a process high-water mark never falls, so after the first
    #: implementation has touched the pages every later one reads a delta of zero.
    peak_alloc_bytes: int
    correct: bool
    max_abs_error: float
    max_rel_error: float
    git_sha: str
    python_version: str
    numpy_version: str
    numba_version: str | None
    blas_backend: str
    cpu_model: str
    #: Calls behind each sample.  ``1`` is one call timed directly; above that a sample is
    #: a per-call mean over a batch, which is how ``min_total_seconds`` is met without one
    #: arm taking more samples than another and breaking the rotation's pairing.  Recorded
    #: because ``warm_iqr_seconds`` is uninterpretable without it -- a batch mean's spread
    #: is narrower than a single call's by roughly ``sqrt`` of this.  Defaulted, so a
    #: `results.jsonl` written before the rotation still loads.
    calls_per_sample: int = 1
    cold_compile_seconds: float | None = None
    result_digest: str = ""
    parallel_axis: str | None = None
    negative_control: bool = False
    dimensions: dict[str, Any] = field(default_factory=dict)
    amortised: dict[str, float] = field(default_factory=dict)
    skipped_reason: str = ""
    oversubscribed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_rows(output: Path) -> list[Row]:
    """Rows from a previous run, for ``--append``.

    A full sweep is hours and a single kernel is minutes, so running it kernel by kernel
    has to be possible without each run erasing the last one's summary.  Rows are keyed by
    ``(kernel, implementation, n, cores, dimensions)``; a re-run of the same configuration
    replaces the old row rather than sitting beside it, because two rows claiming the same
    measurement with different numbers is worse than either of them alone.

    **The environment is *not* merged.**  ``environment.json`` is overwritten by the run
    that wrote last, and the per-row ``git_sha`` and ``cpu_model`` are what a reader has to
    check: appending a row from another machine is exactly the mistake this package refuses
    to make silently, so the evidence stays on every row.
    """
    path = Path(output) / "latest" / "results.jsonl"
    if not path.exists():
        return []
    fields = {f.name for f in dataclasses.fields(Row)}
    rows: list[Row] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        rows.append(Row(**{k: v for k, v in payload.items() if k in fields}))
    return rows


class MixedEnvironment(RuntimeError):
    """Raised rather than rendering a table whose cells came from different boxes."""


def fingerprint(row: Row) -> tuple[Any, ...]:
    """The box-and-build identity of one row."""
    return tuple(getattr(row, field_name) for field_name in FINGERPRINT)


def mixed_fields(rows: Sequence[Row]) -> dict[str, set[Any]]:
    """The fingerprint fields on which ``rows`` disagree, with the values seen.

    Empty when every row came from one box and one build, which is the only state in which
    a summary table means anything.  A *skipped* row is excluded: it entered no plan, ran
    no code and records the build it would have run on, so a sweep on a machine without
    numba would otherwise report itself as mixed against its own skips.
    """
    measured = [row for row in rows if not row.skipped_reason]
    seen: dict[str, set[Any]] = {}
    for field_name in FINGERPRINT:
        values = {getattr(row, field_name) for row in measured}
        if len(values) > 1:
            seen[field_name] = values
    return seen


def refuse_mixed_environment(rows: Sequence[Row]) -> None:
    """Fail closed before a mixed set is rendered as a table.

    ``load_rows`` carries the evidence on every row precisely so that appending a run from
    another machine is not silent -- but "not silent" was a column a reader had to diff by
    eye, and a summary table with a verdict at the bottom of it is read long before anyone
    does that. The benchmark evidence rule is recorded in
    ``docs/architecture-invariants.md``. The cost of the softer version is concrete: every committed
    benchmark number here predates the timing harness's rotation, so a rerun is a different
    instrument rather than a replication, and a ``1.02x`` against a ``0.98x`` was never
    resolved by the numbers on record.  This is that warning made mechanical.

    There is deliberately no override.  A mixed set has two honest resolutions -- rerun the
    sweep, or point ``--output`` at a fresh directory so the two stay two tables -- and a
    flag that suppressed this would be chosen under exactly the deadline that makes the
    mixture a mistake.
    """
    mixed = mixed_fields(rows)
    if not mixed:
        return
    detail = "; ".join(
        f"{name}: {sorted(str(value) for value in values)}" for name, values in mixed.items()
    )
    raise MixedEnvironment(
        f"these rows did not all come from one box and one build, so they cannot be one "
        f"table: {detail}. A partial re-run leaves the rows it did not touch behind, and a "
        f"ratio taken across the two is a comparison of instruments rather than of "
        f"implementations. Re-run the sweep, or write to a fresh --output so the two stay "
        f"two measurements"
    )


def _key(row: Row) -> tuple[Any, ...]:
    return (
        row.operation,
        row.implementation,
        row.n,
        row.num_cores_requested,
        json.dumps(row.dimensions, sort_keys=True, default=str),
    )


def merge(previous: Sequence[Row], current: Sequence[Row]) -> list[Row]:
    """``current`` wins on a collision; everything else is carried forward."""
    merged = {_key(row): row for row in previous}
    merged.update({_key(row): row for row in current})
    return list(merged.values())


def write_all(rows: Sequence[Row], environment: Any, output: Path) -> Path:
    """Write the four artefacts under ``output/latest`` and return that directory.

    Refuses outright on a mixed set: the raw rows are not written either, because a
    ``results.jsonl`` on disk is what the next ``--append`` reads and half-writing one is
    how the mixture would survive the refusal.
    """
    refuse_mixed_environment(rows)
    latest = Path(output) / "latest"
    latest.mkdir(parents=True, exist_ok=True)

    with (latest / "results.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), default=str) + "\n")

    flat = [_flatten(row.to_dict()) for row in rows]
    if flat:
        columns = sorted({key for record in flat for key in record})
        with (latest / "results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(flat)

    (latest / "environment.json").write_text(
        json.dumps(environment.to_dict(), indent=2, default=str) + "\n"
    )
    (latest / "summary.md").write_text(summarise(rows, environment))
    return latest


def _flatten(record: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, dict):
            for inner, inner_value in value.items():
                out[f"{key}.{inner}"] = inner_value
        else:
            out[key] = value
    return out


# ------------------------------------------------------------------- the summary

#: Thresholds the plan sets for continuing an investigation.  A kernel that clears none of
#: them is reported as "retain numpy" however interesting its shape looked.
_SERIAL_BAR = 1.25
_PARALLEL_BAR = 1.5
_MEMORY_BAR = 0.75


def summarise(rows: Sequence[Row], environment: Any) -> str:
    """The markdown report."""
    lines: list[str] = []
    lines.append("# Numba and parallelism after the nuisances are fitted\n")
    lines.append(
        "Every number below is *post-nuisance*: the learner fits are outside the timed "
        "region, so a share here is a share of the half of a fit this package owns and "
        "not of a fit.  See `environment.json` for the box; results from different "
        "machines are different measurements and must not be read as one series.\n"
    )
    lines.append(
        f"- **CPU**: {environment.cpu_model} "
        f"({environment.physical_cores} physical / {environment.logical_cores} logical cores)\n"
        f"- **BLAS**: {environment.blas_backend} ({environment.blas_threading_layer})\n"
        f"- **numpy** {environment.numpy_version}, "
        f"**numba** {environment.numba_version or 'absent'}\n"
        f"- **commit**: `{environment.git_sha[:12]}`"
        f"{' (working tree dirty)' if environment.git_dirty else ''}\n"
    )

    # Grouped by *configuration*, not by kernel. A kernel swept over two sizes and three
    # fold counts is six different measurements, and pooling them would take the maximum
    # speed-up across sizes and the ratio of a 10,000-row reference to a 100,000-row
    # candidate. That is not a speed-up; it is a size difference wearing one's clothes,
    # and it is exactly the mistake that makes a benchmark suite quietly useless.
    configurations = _by_configuration(rows)
    lines.append("\n## Verdicts\n")
    lines.append(
        "| kernel | configuration | estimator | best implementation | serial | parallel "
        "| memory | verdict |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for (name, label), group in configurations.items():
        verdict = _verdict(group)
        lines.append(
            f"| `{name}` | {label} | {verdict['estimator']} | {verdict['best']} | "
            f"{verdict['serial']} | {verdict['parallel']} | {verdict['memory']} | "
            f"{verdict['decision']} |"
        )

    lines.append("\n## Parallel scaling\n")
    lines.append(
        "One table per kernel per configuration.  Speed-up is against *that* "
        "implementation at one core, so it isolates what the added cores bought; the "
        "verdict table above compares against the numpy reference instead.\n"
    )
    for (name, label), group in configurations.items():
        scaling = _scaling_table(group)
        if not scaling:
            continue
        lines.append(f"\n### `{name}` -- {label}\n")
        lines.append(
            "| implementation | cores | seconds | speed-up | efficiency | alloc/call (MB) |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        lines.extend(scaling)

    lines.append("\n## Algorithmic headroom, before any compilation\n")
    lines.append(
        "Several kernels carry a numpy arm that is not a compiled one -- masks carried "
        "rather than rebuilt, counterfactual arms updated once at the end of a walk "
        "rather than on every trial step, horizon-independent quantities shared across "
        "horizons.  Where the column below is large, the answer is *fix the algorithm*, "
        "and a compiler would be optimising work that need not be done.\n"
    )
    lines.append("| kernel | configuration | best numpy variant | gain over the shipped shape |")
    lines.append("| --- | --- | --- | ---: |")
    any_headroom = False
    for (name, label), group in configurations.items():
        variant, gain = _algorithmic_headroom(group)
        if variant is None:
            continue
        any_headroom = True
        lines.append(f"| `{name}` | {label} | `{variant}` | {gain:.2f}x |")
    if not any_headroom:
        lines.append("| - | - | none measured | - |")

    amortised = [row for row in rows if row.amortised]
    if amortised:
        lines.append("\n## Amortisation: seconds per call at k repeated calls\n")
        lines.append(
            "The curve a repeated workload sees.  It converges on the warm time from "
            "above, and how fast says how much of the first call was setup -- for a "
            "compiled kernel, the compilation.  Only kernels a repeated workload actually "
            "calls repeatedly are measured this way (`KernelSpec.amortise`), and only "
            "under `--amortise`, because the 1,000-call column costs a thousand calls.\n"
        )
        counts = sorted({int(k) for row in amortised for k in row.amortised})
        header = " | ".join(f"k={c}" for c in counts)
        lines.append(f"| kernel | implementation | n | {header} |")
        lines.append("| --- | --- | ---: |" + " ---: |" * len(counts))
        for row in amortised:
            cells = " | ".join(
                f"{row.amortised[str(c)] * 1e3:.3f} ms" if str(c) in row.amortised else "-"
                for c in counts
            )
            lines.append(f"| `{row.operation}` | {row.implementation} | {row.n} | {cells} |")

    failures = [row for row in rows if not row.correct and not row.skipped_reason]
    lines.append("\n## Correctness\n")
    if failures:
        lines.append(
            "The following implementations did **not** match the numpy reference to the "
            "tolerance their kernel declares.  Their timings are recorded but are not "
            "summarised above, because a faster wrong answer is not a result.\n"
        )
        lines.append("| kernel | implementation | n | max abs | max rel |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for row in failures:
            lines.append(
                f"| `{row.operation}` | {row.implementation} | {row.n} | "
                f"{row.max_abs_error:.3g} | {row.max_rel_error:.3g} |"
            )
    else:
        lines.append("Every measured implementation matched the numpy reference.\n")

    skipped = [row for row in rows if row.skipped_reason]
    if skipped:
        lines.append("\n## Not run\n")
        lines.append("| kernel | implementation | reason |")
        lines.append("| --- | --- | --- |")
        seen = set()
        for row in skipped:
            key = (row.operation, row.implementation, row.skipped_reason)
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"| `{row.operation}` | {row.implementation} | {row.skipped_reason} |")

    controls = [row for row in rows if row.negative_control]
    if controls:
        lines.append("\n## Negative controls\n")
        lines.append(
            "Kept deliberately.  A suite that drops the kernels it expected to reject "
            "cannot be used to check that its accepted ones are not an artefact of the "
            "harness.\n"
        )
    return "\n".join(lines) + "\n"


def _by_kernel(rows: Sequence[Row]) -> dict[str, list[Row]]:
    out: dict[str, list[Row]] = {}
    for row in rows:
        out.setdefault(row.operation, []).append(row)
    return out


#: Dimensions that are not worth putting in a configuration label: they are the same for
#: every row of a run and would make every label longer without distinguishing anything.
_UNINTERESTING = {"seed"}


def _label(row: Row, varying: set[str]) -> str:
    """A short name for one configuration: ``n`` plus whatever else was swept.

    Only the dimensions that actually *vary* within the kernel are named. A kernel run at
    one fold count does not need "n_folds=10" in every label, and a label that lists every
    dimension is one a reader stops reading.
    """
    parts = [f"n={row.n:,}"] if row.n else []
    parts += [
        f"{key}={row.dimensions[key]}"
        for key in sorted(varying)
        if key in row.dimensions and key not in ({"n"} | _UNINTERESTING)
    ]
    return ", ".join(parts) or "default"


def _by_configuration(rows: Sequence[Row]) -> dict[tuple[str, str], list[Row]]:
    """``(kernel, configuration label) -> rows``, in first-seen order."""
    out: dict[tuple[str, str], list[Row]] = {}
    for kernel, group in _by_kernel(rows).items():
        seen: dict[str, set[Any]] = {}
        for row in group:
            for key, value in row.dimensions.items():
                seen.setdefault(key, set()).add(_hashable(value))
        varying = {key for key, values in seen.items() if len(values) > 1}
        for row in group:
            out.setdefault((kernel, _label(row, varying)), []).append(row)
    return out


def _hashable(value: Any) -> Any:
    return tuple(value) if isinstance(value, list) else value


def _reference(group: Iterable[Row]) -> Row | None:
    candidates = [
        row
        for row in group
        if row.implementation == "numpy" and row.num_cores_requested == 1 and not row.skipped_reason
    ]
    return min(candidates, key=lambda row: row.warm_seconds) if candidates else None


def _verdict(group: Sequence[Row]) -> dict[str, str]:
    reference = _reference(group)
    usable = [row for row in group if row.correct and not row.skipped_reason]
    estimator = group[0].scenario
    if reference is None or not usable:
        return {
            "estimator": estimator,
            "best": "-",
            "serial": "-",
            "parallel": "-",
            "memory": "-",
            "decision": "not measured",
        }
    best = min(usable, key=lambda row: row.warm_seconds)
    # By *implementation*, not by the thread count it happened to run at. A `prange`
    # kernel pinned to one thread is running serially, but crediting its ratio to the
    # "serial" column would answer "what does compilation alone buy" with a number taken
    # from the parallel kernel -- which is a different kernel, with a different loop.
    serial_rows = [
        row
        for row in usable
        if "parallel" not in row.implementation
        and row.implementation not in ("numpy", "numpy_threads", "numpy_threaded_blas")
    ]
    parallel_rows = [
        row
        for row in usable
        if ("parallel" in row.implementation or row.implementation == "numpy_threads")
        and (row.numba_threads > 1 or row.workers > 1)
    ]
    serial_gain = (
        max(reference.warm_seconds / row.warm_seconds for row in serial_rows)
        if serial_rows
        else float("nan")
    )
    parallel_gain = (
        max(reference.warm_seconds / row.warm_seconds for row in parallel_rows)
        if parallel_rows
        else float("nan")
    )
    # Per-call allocation, not process RSS: see `Row.peak_alloc_bytes`.
    memory_ratio = (
        min(row.peak_alloc_bytes for row in usable) / reference.peak_alloc_bytes
        if reference.peak_alloc_bytes
        else float("nan")
    )

    cleared = serial_gain >= _SERIAL_BAR or parallel_gain >= _PARALLEL_BAR
    if group[0].negative_control and not cleared:
        decision = "retain numpy (control, as expected)"
    elif parallel_gain >= _PARALLEL_BAR and parallel_gain > serial_gain * 1.2:
        decision = "adopt numba parallel"
    elif serial_gain >= _SERIAL_BAR:
        decision = "adopt numba serial"
    elif memory_ratio == memory_ratio and memory_ratio <= _MEMORY_BAR:
        decision = "adopt for memory"
    else:
        decision = "retain numpy"
    # A numpy *variant* winning means the answer is an algorithm rather than a compiler --
    # deferring the arm updates, carrying the masks, sharing them across horizons. That is
    # the plan's "improve the numpy instead" outcome and it overrides the verdict above.
    #
    # `numpy_threads` is excluded by name: it is the *same* numpy running on more cores,
    # which is a statement about parallelism and not about the algorithm, and filing it as
    # an algorithmic improvement would tell a reader to go looking for a rewrite that does
    # not exist.
    if best.implementation.startswith("numpy") and best.implementation not in (
        "numpy",
        "numpy_threads",
        "numpy_threaded_blas",
    ):
        decision = f"improve numpy instead ({best.implementation})"
    return {
        "estimator": estimator,
        "best": f"`{best.implementation}` @ {best.num_cores_requested} core(s)",
        "serial": f"{serial_gain:.2f}x" if serial_gain == serial_gain else "-",
        "parallel": f"{parallel_gain:.2f}x" if parallel_gain == parallel_gain else "-",
        "memory": f"{memory_ratio:.2f}x" if memory_ratio == memory_ratio else "-",
        "decision": decision,
    }


def _algorithmic_headroom(group: Sequence[Row]) -> tuple[str | None, float]:
    """The best *non-compiled, non-threaded* numpy variant, and its gain over `numpy`.

    ``numpy_threads`` is excluded: it is the same algorithm on more cores, and reporting
    it here would tell a reader to look for a rewrite that does not exist.
    """
    reference = _reference(group)
    if reference is None:
        return None, float("nan")
    variants = [
        row
        for row in group
        if row.implementation.startswith("numpy")
        and row.implementation not in ("numpy", "numpy_threads", "numpy_threaded_blas")
        and row.correct
        and not row.skipped_reason
    ]
    if not variants:
        return None, float("nan")
    best = min(variants, key=lambda row: row.warm_seconds)
    return best.implementation, reference.warm_seconds / best.warm_seconds


def _scaling_table(group: Sequence[Row]) -> list[str]:
    """One row per (implementation, core count), within a single configuration.

    The caller has already partitioned by configuration, so every row here differs only in
    its implementation and its core count -- which is what makes a speed-up column mean
    what it says.
    """
    lines: list[str] = []
    by_implementation: dict[str, list[Row]] = {}
    for row in group:
        if row.skipped_reason or not row.correct:
            continue
        by_implementation.setdefault(row.implementation, []).append(row)
    for name, rows in sorted(by_implementation.items()):
        rows = sorted(rows, key=lambda row: row.num_cores_requested)
        if len(rows) < 2:
            continue
        baseline = rows[0].warm_seconds
        for row in rows:
            speedup = baseline / row.warm_seconds if row.warm_seconds else float("nan")
            flag = " ⚠ slower" if row.warm_seconds > 1.05 * baseline else ""
            lines.append(
                f"| {name} | {row.num_cores_requested} | {row.warm_seconds:.4f} | "
                f"{speedup:.2f}x | {speedup / row.num_cores_requested:.2f} | "
                f"{row.peak_alloc_bytes / 1e6:.2f}{flag} |"
            )
    return lines
