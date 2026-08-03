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
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Row", "write_all"]


@dataclass
class Row:
    """One measurement.  The schema the plan specifies, plus what the summary needs."""

    scenario: str
    operation: str
    implementation: str
    n: int
    num_cores_requested: int
    num_cores_effective: int
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
    correct: bool
    max_abs_error: float
    max_rel_error: float
    git_sha: str
    python_version: str
    numpy_version: str
    numba_version: str | None
    blas_backend: str
    cpu_model: str
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


def write_all(rows: Sequence[Row], environment: Any, output: Path) -> Path:
    """Write the four artefacts under ``output/latest`` and return that directory."""
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

    kernels = _by_kernel(rows)
    lines.append("\n## Verdicts\n")
    lines.append(
        "| kernel | estimator | best implementation | serial | parallel | memory | verdict |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for name, group in kernels.items():
        verdict = _verdict(group)
        lines.append(
            f"| `{name}` | {verdict['estimator']} | {verdict['best']} | "
            f"{verdict['serial']} | {verdict['parallel']} | {verdict['memory']} | "
            f"{verdict['decision']} |"
        )

    lines.append("\n## Parallel scaling\n")
    for name, group in kernels.items():
        scaling = _scaling_table(group)
        if not scaling:
            continue
        lines.append(f"\n### `{name}`\n")
        lines.append("| implementation | cores | seconds | speed-up | efficiency | peak RSS (MB) |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        lines.extend(scaling)

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
    serial_rows = [
        row
        for row in usable
        if row.numba_threads == 1 and row.workers == 1 and row.implementation != "numpy"
    ]
    parallel_rows = [row for row in usable if row.numba_threads > 1 or row.workers > 1]
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
    memory_ratio = (
        min(row.rss_delta_bytes for row in usable) / reference.rss_delta_bytes
        if reference.rss_delta_bytes
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
    if best.implementation.startswith("numpy") and best.implementation != "numpy":
        decision = f"improve numpy instead ({best.implementation})"
    return {
        "estimator": estimator,
        "best": f"`{best.implementation}` @ {best.num_cores_requested} core(s)",
        "serial": f"{serial_gain:.2f}x" if serial_gain == serial_gain else "-",
        "parallel": f"{parallel_gain:.2f}x" if parallel_gain == parallel_gain else "-",
        "memory": f"{memory_ratio:.2f}x" if memory_ratio == memory_ratio else "-",
        "decision": decision,
    }


def _scaling_table(group: Sequence[Row]) -> list[str]:
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
                f"{row.peak_rss_bytes / 1e6:.0f}{flag} |"
            )
    return lines
