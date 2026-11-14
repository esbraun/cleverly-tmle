"""The refit audit of PR 3: which committed rows take MOVER or the skew-widened t, and by how much.

The screen (:mod:`tests.studies.evidence.asymmetric_audit`) recomputes every committed difference
whose arms its replicate records, and reads every other row under symmetric widenings.  It cannot
tell a continuous estimate from a binary one, and it cannot bound a MOVER difference whose arms
were not recorded, because MOVER narrows one side.  This module refits, under the shipped package,
the keys the committed rows came from, and records how each estimate's interval is formed.  The
screen then reads the records (:func:`tests.studies.evidence.asymmetric_audit.screen_row`).

**Scope refit.**  For every study in :func:`targets` it captures the study's own keys and committed
seeds without fitting, as the band rerun does (:func:`tests.studies.evidence.band_rerun.capture`
for the property phase; ``fit_primary`` swapped for a recorder for the primary loop), and refits
replicate 0 of every property cell and primary scenario.  A recorder inside each worker logs every
:class:`~cleverly.inference.ParameterEstimate` whose interval or test is read, and every call of
:func:`~cleverly.inference.skew_widened_ci` (the working-model and effect-curve paths), with the
interval scale, the degrees of freedom, the acceleration, MOVER's arms and the shipped and normal
limits.  Each written row is matched to the logged estimates by its estimate and standard error,
and its estimate and standard error are held to the committed row at :data:`REPLAY_TOLERANCE`
(``replay`` = ``ok``, ``mismatch`` or ``absent``).  A row that matches no logged estimate is
``own``: the study formed that interval itself, so the shipped default does not reach its
committed verdict.  A mismatch is recorded, not a stop: the screen judges each one
(:func:`tests.studies.evidence.asymmetric_audit.replay_shifts`) and applies the R_a bound to the
committed row when the refit sits within a quarter of a sampling deviation of it.  The R_a
refits read the scope records whatever their replay, so a scope run with mismatches resumes at
the R_a stage.

**R_a refits.**  Every group whose replicate-0 estimate takes ``mover`` or ``skew_t``, and that the
screen does not already recompute exactly, is refitted on its first :data:`REFIT_REPLICATES`
replicates.  Each refit gives, per side, the shipped half-width over the normal half-width.  The
screen widens the observed spread of each side's ratio by :data:`INFLATION` (plan 9.8's factor
``c``), takes the :data:`QUANTILE` quantiles, and judges each committed verdict at both ends of
both sides (narrowing and widening), with the study's own rule.

Outputs, one directory per study under :data:`AUDIT_DIR`: ``scope.csv``, ``refits.csv`` and
``capture.json``.  Commands, each resumable from ``--cache`` and skipping a study whose output
exists::

    python -m tests.studies.evidence.asymmetric_refit scope --jobs N --cache DIR [slug ...]
    python -m tests.studies.evidence.asymmetric_refit project [slug ...]
    python -m tests.studies.evidence.asymmetric_refit refits --jobs N --cache DIR [slug ...]

``project`` fits nothing: it prints each study's R_a keys and their CPU-hours projected from the
scope stage's seconds per key.  A fitting command exits 1 when a study crashed or a key failed;
replay mismatches are counted in ``capture.json`` and judged by the screen.

Every fitting command refuses to start unless each of
:data:`~tests.studies.evidence.band_rerun.THREAD_VARIABLES` is ``1``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tests.studies.evidence import band_rerun, replication
from tests.studies.evidence.manifest import read_rows
from tests.studies.evidence.registry import ROOT, StudyRecord

#: One directory per audited study.
AUDIT_DIR = ROOT / "tests" / "canonical" / "asymmetric-refit"
SCOPE_FILE = "scope.csv"
REFITS_FILE = "refits.csv"
CAPTURE_FILE = "capture.json"

#: Replicates refitted per remaining group (plan 9.8's ``R_a``).
REFIT_REPLICATES = 20
#: The inflation of each side's observed spread (plan 9.8's ``c``).
INFLATION = 1.5
#: The quantile of each side's ratio that bounds it, on either end.
QUANTILE = 0.99
#: The interval scales PR 3 introduced.
NEW_SCALES = frozenset({"mover", "skew_t"})
#: The replay rule of the refits, as the band rerun's.
REPLAY_TOLERANCE = band_rerun.REPLAY_TOLERANCE
#: The matching tolerance of a row to a logged estimate, relative.
MATCH_TOLERANCE = 1e-10

RECORD_COLUMNS = (
    "stage",
    "phase",
    "group",
    "replicate",
    "estimate",
    "std_error",
    "committed_estimate",
    "committed_std_error",
    "replay",
    "scale",
    "effective_df",
    "acceleration",
    "alpha",
    "shipped_lower",
    "shipped_upper",
    "normal_lower",
    "normal_upper",
    "minuend",
    "minuend_se",
    "subtrahend",
    "subtrahend_se",
    "correlation",
)


# ---------------------------------------------------------------------------- the recorder

_LOG: list[dict[str, Any]] = []
_GUARD = [False]
_INSTALLED = [False]


def _z(alpha: float) -> float:
    from scipy import stats

    return float(stats.norm.ppf(1.0 - alpha / 2.0))


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _estimate_entry(estimate: Any, original: Callable[..., Any]) -> dict[str, Any]:
    """The log entry of one :class:`~cleverly.inference.ParameterEstimate`."""
    psi, alpha = _float(estimate.psi), float(estimate.alpha)
    entry: dict[str, Any] = {
        "psi": psi,
        "std_error": _float(estimate.std_error),
        "scale": estimate.interval_scale,
        "effective_df": _float(estimate.effective_df),
        "acceleration": _float(estimate.acceleration),
        "alpha": alpha,
    }
    try:
        se = float(estimate._plugin_std_error())
        margin = float(estimate._residual_margin(estimate.rule_widened))
        low, high = original(estimate, widen=estimate.rule_widened)
    except Exception:  # an estimate that cannot form an interval still records its scale
        se, margin, low, high = (float("nan"),) * 4
    half = _z(alpha) * se + margin
    entry.update(
        plugin_se=se,
        shipped_lower=_float(low),
        shipped_upper=_float(high),
        normal_lower=psi - half,
        normal_upper=psi + half,
    )
    arms = estimate._mover_arms() if entry["scale"] == "mover" else None
    if arms is not None:
        from cleverly.inference import implied_correlation

        minuend, subtrahend = arms
        entry.update(
            minuend=_float(minuend.psi),
            minuend_se=float(minuend._plugin_std_error()),
            subtrahend=_float(subtrahend.psi),
            subtrahend_se=float(subtrahend._plugin_std_error()),
            correlation=implied_correlation(
                float(estimate.variance), float(minuend.variance), float(subtrahend.variance)
            ),
        )
    return entry


def install() -> None:
    """Log every interval or test the package forms, in this process (idempotent)."""
    if _INSTALLED[0]:
        return
    from cleverly.inference import delta
    from cleverly.inference.influence import ParameterEstimate

    interval = ParameterEstimate._wald_interval
    test = ParameterEstimate._plugin_wald_test
    skew = delta.skew_widened_ci

    def log_estimate(estimate: Any) -> None:
        if _GUARD[0]:
            return
        _GUARD[0] = True
        try:
            _LOG.append(_estimate_entry(estimate, interval))
        except Exception:  # the recorder never changes a fit
            pass
        finally:
            _GUARD[0] = False

    def logged_interval(self: Any, *, widen: bool) -> tuple[float, float]:
        log_estimate(self)
        return interval(self, widen=widen)

    def logged_test(self: Any, null: float | None) -> Any:
        log_estimate(self)
        return test(self, null)

    def logged_skew(psi: float, se: float, alpha: float, **options: Any) -> tuple[float, float]:
        low, high = skew(psi, se, alpha, **options)
        if not _GUARD[0]:
            half = _z(alpha) * float(se) + float(options.get("margin", 0.0))
            _LOG.append(
                {
                    "psi": float(psi),
                    "std_error": float(se),
                    "plugin_se": float(se),
                    "scale": "skew_t",
                    "effective_df": _float(options.get("df")),
                    "acceleration": _float(options.get("acceleration")),
                    "alpha": float(alpha),
                    "shipped_lower": float(low),
                    "shipped_upper": float(high),
                    "normal_lower": float(psi) - half,
                    "normal_upper": float(psi) + half,
                }
            )
        return low, high

    ParameterEstimate._wald_interval = logged_interval  # type: ignore[method-assign]
    ParameterEstimate._plugin_wald_test = logged_test  # type: ignore[method-assign]
    delta.skew_widened_ci = logged_skew  # type: ignore[assignment]
    _INSTALLED[0] = True


@contextlib.contextmanager
def logging() -> Iterator[list[dict[str, Any]]]:
    """The entries logged inside the block (the recorder installed first)."""
    install()
    start = len(_LOG)
    held: list[dict[str, Any]] = []
    try:
        yield held
    finally:
        held.extend(_LOG[start:])
        del _LOG[start:]


@dataclass(frozen=True)
class Logged:
    """A study worker whose value is its rows and the log of the intervals it read.

    Parameters
    ----------
    fit_one : Callable
        The study's ``fit_one(key, drawn)``.
    rows_at : int or None
        The position of the rows in the worker's value, as the study declares it.
    """

    fit_one: Callable[[Any, Any], Any]
    rows_at: int | None

    def __call__(self, key: Any, drawn: Any) -> dict[str, Any]:
        with logging() as log:
            try:
                value = self.fit_one(key, drawn)
            except Exception as error:  # every failure is reported, none stops the audit
                return {"rows": [], "log": [], "error": f"{type(error).__name__}: {error}",
                        "trace": traceback.format_exc(limit=8)}  # fmt: skip
        try:
            rows = band_rerun.rows_of(value, self.rows_at)
        except Exception as error:
            return {"rows": [], "log": list(log), "error": f"rows: {error}", "trace": ""}
        return {"rows": rows, "log": list(log), "error": None, "trace": ""}


# ---------------------------------------------------------------------------- capture


@dataclass
class Job:
    """One fitting call: its keys, the positions selected, and how to fit them."""

    phase: str
    label: str
    keys: list[Any]
    selected: list[int]
    fit_one: Callable[[Any, Any], Any]
    rows_at: int | None
    cell_of: Callable[[Any], str]
    replicate_of: Callable[[Any], int]
    draw: Callable[[Any], Any] | None
    seed_of: Callable[[Any], int | None] | None
    names_of: Callable[[Any], set[str]]


class _Captured(Exception):
    """Stops a study's primary loop once its ``fit_primary`` call is recorded."""


def _first(key: Sequence[Any]) -> str:
    return str(key[0])


def _second(key: Sequence[Any]) -> int:
    return int(key[1])


@dataclass(frozen=True)
class _ScenarioGroups:
    """The ``scenario/estimand`` groups a primary key writes: each estimand its scenario
    reports."""

    cell_of: Callable[[Any], str]
    scenarios: Mapping[str, tuple[str, ...]]

    def __call__(self, key: Any) -> set[str]:
        scenario = self.cell_of(key)
        return {f"{scenario}/{estimand}" for estimand in self.scenarios.get(scenario, ())}


def capture_primary(record: StudyRecord) -> list[Job]:
    """The study's ``fit_primary`` call, recorded without a fit; empty when it makes none."""
    module = record.runner()
    jobs: list[Job] = []
    scenarios = {str(s): tuple(e) for s, e in record.scenarios.items()}

    def recorder(keys: Sequence[Any], fit_one: Any, **options: Any) -> Any:
        cell_of = options.get("cell_of") or _first
        jobs.append(
            Job(
                "primary",
                "primary",
                list(keys),
                [],
                fit_one,
                options.get("rows_at", 2),
                cell_of,
                options.get("replicate_of") or _second,
                options.get("draw"),
                options.get("seed_of"),
                _ScenarioGroups(cell_of, scenarios),
            )
        )
        raise _Captured

    replicates = getattr(module, "PRIMARY_REPLICATES", record.replicates)
    n = getattr(module, "PRIMARY_N", record.n)
    with (
        band_rerun.swapped("fit_primary", replication.fit_primary, recorder),
        contextlib.suppress(_Captured),
    ):
        module.draw_and_fit(replicates=replicates, n=n, n_jobs=1)
    return jobs


def capture_properties(record: StudyRecord) -> list[Job]:
    """Every ``fit_properties`` call of the study's property phase, without a fit."""
    path = record.artifacts / "property-replicates.parquet"
    if not path.exists():
        return []
    rows = read_rows(path)
    index = band_rerun.committed_index(rows)
    cells = {f"{p}/{c}" for p, c in zip(rows["property"], rows["cell"], strict=True)}
    jobs = []
    for call in band_rerun.capture(record, cells, index):
        jobs.append(
            Job(
                "property",
                call.label,
                call.keys,
                [],
                call.options["fit_one"],
                call.options.get("rows_at"),
                call.options["cell_of"],
                call.replicate_of,
                call.options.get("draw"),
                call.options.get("seed_of"),
                call.written,
            )
        )
    return jobs


def select(job: Job, *, replicates: int, names: set[str] | None = None) -> list[int]:
    """The positions of the keys that hold the first ``replicates`` replicates of each named
    group (every group the job writes when ``names`` is ``None``)."""
    by_name: dict[str, list[tuple[int, int]]] = {}
    for position, key in enumerate(job.keys):
        replicate = int(job.replicate_of(key))
        for name in job.names_of(key):
            if names is None or name in names:
                by_name.setdefault(name, []).append((replicate, position))
    chosen: set[int] = set()
    for items in by_name.values():
        first = set(sorted({replicate for replicate, _ in items})[:replicates])
        chosen.update(position for replicate, position in items if replicate in first)
    return sorted(chosen)


# ---------------------------------------------------------------------------- matching


def _close(a: float, b: float, tolerance: float) -> bool:
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= tolerance * max(1.0, abs(b))


def match(row: Mapping[str, Any], log: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The logged estimate a written row reports, by its estimate and standard error.

    Returns the entry, with ``scale`` the sorted ``+``-joined scales when several entries of
    different scales match, ``own`` when none does, and ``failed`` for a row with no estimate.
    """
    estimate, se = _float(row.get("estimate")), _float(row.get("std_error"))
    if not (math.isfinite(estimate) and math.isfinite(se)):
        return {"scale": "failed"}
    hits = [
        entry
        for entry in log
        if _close(estimate, entry["psi"], MATCH_TOLERANCE)
        and (
            _close(se, entry["std_error"], MATCH_TOLERANCE)
            or _close(se, entry.get("plugin_se", float("nan")), MATCH_TOLERANCE)
        )
    ]
    if not hits:
        return {"scale": "own"}
    scales = sorted({str(entry["scale"]) for entry in hits})
    chosen = next((entry for entry in hits if entry["scale"] in NEW_SCALES), hits[0])
    return {**chosen, "scale": "+".join(scales)}


def _group(phase: str, row: Mapping[str, Any]) -> str:
    if phase == "property":
        return f"{row['property']}/{row['cell']}"
    return f"{row['scenario']}/{row['estimand']}"


def committed_rows(record: StudyRecord) -> dict[tuple[str, str, int], tuple[float, float]]:
    """``(phase, group, replicate)`` to the committed ``(estimate, std_error)``."""
    out: dict[tuple[str, str, int], tuple[float, float]] = {}
    path = record.artifacts / "replicates.parquet"
    if path.exists():
        rows = read_rows(path)
        rows = rows.loc[rows["implementation"] == record.implementation]
        for row in rows.itertuples():
            out[("primary", f"{row.scenario}/{row.estimand}", int(row.replicate))] = (
                float(row.estimate),
                float(row.std_error),
            )
    path = record.artifacts / "property-replicates.parquet"
    if path.exists():
        rows = read_rows(path)
        rows = rows.loc[rows["role"] == "positive"]
        for row in rows.itertuples():
            out[("property", f"{row.property}/{row.cell}", int(row.replicate))] = (
                float(row.estimate),
                float(row.std_error),
            )
    return out


def records(
    stage: str,
    job: Job,
    values: Sequence[Mapping[str, Any]],
    committed: Mapping[tuple[str, str, int], tuple[float, float]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """The audit records of one job's fitted keys, and the messages of its failed keys."""
    out: list[dict[str, Any]] = []
    errors: list[str] = []
    for value in values:
        if value.get("error"):
            errors.append(f"{job.phase}/{job.label}: {value['error']}\n{value.get('trace', '')}")
            continue
        for row in value["rows"]:
            if job.phase == "property" and row.get("role", "positive") != "positive":
                continue
            group = _group(job.phase, row)
            replicate = int(row["replicate"])
            entry = match(row, value["log"])
            estimate, se = _float(row.get("estimate")), _float(row.get("std_error"))
            before = committed.get((job.phase, group, replicate))
            if before is None:
                replay = "absent"
            else:
                same = _close(estimate, before[0], REPLAY_TOLERANCE) and _close(
                    se, before[1], REPLAY_TOLERANCE
                )
                replay = "ok" if same else "mismatch"
            out.append(
                {
                    "stage": stage,
                    "phase": job.phase,
                    "group": group,
                    "replicate": replicate,
                    "estimate": estimate,
                    "std_error": se,
                    "committed_estimate": np.nan if before is None else before[0],
                    "committed_std_error": np.nan if before is None else before[1],
                    "replay": replay,
                    **{
                        name: entry.get(name, np.nan)
                        for name in RECORD_COLUMNS[9:]
                        if name != "scale"
                    },
                    "scale": entry["scale"],
                }
            )
    return out, errors


# ---------------------------------------------------------------------------- driver


def targets() -> list[str]:
    """The studies the audit refits: every screen row the shipped intervals may move and the
    screen cannot recompute (``widening_flips`` or ``unpaired`` not empty)."""
    from tests.studies.evidence import asymmetric_audit

    screen = asymmetric_audit.read_screen()
    chosen = screen.loc[
        (screen["disposition"] != "current")
        & ((screen["widening_flips"] != "") | (screen["unpaired"] != ""))
    ]
    return sorted(chosen["study"])


def _fit(record: StudyRecord, job: Job, *, jobs: int, cache: Path | None, stage: str) -> list[Any]:
    outcomes = replication.fit_replicates(
        [job.keys[i] for i in job.selected],
        Logged(job.fit_one, job.rows_at),
        n_jobs=jobs,
        label=f"{stage}-{job.phase}-{job.label}",
        implementation=record.implementation,
        cell_of=job.cell_of,
        replicate_of=job.replicate_of,
        draw=job.draw,
        seed_of=job.seed_of,
        checkpoint=None if cache is None else cache / record.slug / stage,
    )
    return [value for value in outcomes.fitted if value is not None]


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n", float_format="%.17g")
    temporary.replace(path)


def read_records(slug_or_dir: str | Path, name: str) -> pd.DataFrame | None:
    """A study's committed scope or refit records, or ``None``."""
    directory = AUDIT_DIR / slug_or_dir if isinstance(slug_or_dir, str) else slug_or_dir
    path = directory / name
    if not path.exists():
        return None
    return pd.read_csv(path, keep_default_na=False, na_values=[""], dtype={"scale": str})


def remaining_groups(slug: str, scope: pd.DataFrame | None = None) -> set[str]:
    """The groups the R_a refits read: each group whose scope record takes a new scale, less
    the differences the screen recomputes exactly."""
    from tests.studies.evidence import asymmetric_audit

    scope = read_records(slug, SCOPE_FILE) if scope is None else scope
    if scope is None:
        return set()
    taking = {
        str(group)
        for group, scale in zip(scope["group"], scope["scale"], strict=True)
        if set(str(scale).split("+")) & NEW_SCALES
    }
    return taking - asymmetric_audit.exact_groups(slug)


def run_study(
    slug: str, *, stage: str, jobs: int, cache: Path | None, output: Path = AUDIT_DIR
) -> dict[str, Any]:
    """Refit one study's scope (``stage="scope"``) or remaining groups (``"refits"``)."""
    band_rerun.require_threads()
    record = band_rerun.record_for(slug)
    started = time.perf_counter()
    property_jobs = capture_properties(record)
    primary_jobs = capture_primary(record)
    captured = time.perf_counter() - started
    if stage == "scope":
        names: set[str] | None = None
        replicates = 1
    else:
        names = remaining_groups(slug, read_records(output / slug, SCOPE_FILE))
        replicates = REFIT_REPLICATES
    committed = committed_rows(record)
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    fitted = 0
    for job in [*property_jobs, *primary_jobs]:
        job.selected = select(job, replicates=replicates, names=names)
        if not job.selected:
            continue
        values = _fit(record, job, jobs=jobs, cache=cache, stage=stage)
        fitted += len(job.selected)
        out, failed = records(stage, job, values, committed)
        rows.extend(out)
        errors.extend(failed)
    frame = pd.DataFrame(rows, columns=list(RECORD_COLUMNS))
    frame = frame.sort_values(["phase", "group", "replicate"], kind="stable")
    name = SCOPE_FILE if stage == "scope" else REFITS_FILE
    summary = {
        "study": slug,
        "stage": stage,
        "property_calls": [job.label for job in property_jobs],
        "primary_captured": bool(primary_jobs),
        "keys_fitted": fitted,
        "rows": len(frame),
        "replay_mismatch": int((frame["replay"] == "mismatch").sum()),
        "replay_absent": int((frame["replay"] == "absent").sum()),
        "failed_keys": len(errors),
        "jobs": jobs,
        "seconds_capture": round(captured, 2),
        "seconds_fit": round(time.perf_counter() - started - captured, 2),
    }
    if errors:
        (output / slug).mkdir(parents=True, exist_ok=True)
        (output / slug / f"{stage}-errors.txt").write_text("\n".join(errors), encoding="utf-8")
    if stage == "refits" and not names:
        summary["rows"] = 0
    _write(frame, output / slug / name)
    capture_path = output / slug / CAPTURE_FILE
    previous = json.loads(capture_path.read_text()) if capture_path.exists() else {}
    previous[stage] = summary
    capture_path.write_text(json.dumps(previous, indent=2, sort_keys=True) + "\n")
    return summary


def project(slug: str, *, output: Path = AUDIT_DIR) -> dict[str, Any]:
    """The R_a keys of one study and their projected CPU seconds, from its scope records and
    the scope stage's seconds per key; nothing is fitted."""
    record = band_rerun.record_for(slug)
    scope = json.loads((output / slug / CAPTURE_FILE).read_text())["scope"]
    names = remaining_groups(slug, read_records(output / slug, SCOPE_FILE))
    keys = 0
    if names:
        for job in [*capture_properties(record), *capture_primary(record)]:
            keys += len(select(job, replicates=REFIT_REPLICATES, names=names))
    per_key = scope["seconds_fit"] * scope["jobs"] / max(1, scope["keys_fitted"])
    return {
        "study": slug,
        "remaining_groups": len(names),
        "refit_keys": keys,
        "cpu_seconds_per_key": round(per_key, 2),
        "projected_cpu_hours": round(keys * per_key / 3600.0, 3),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run a stage over the chosen studies, skipping studies whose output exists."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("scope", "refits", "project", "targets"))
    parser.add_argument("studies", nargs="*")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=AUDIT_DIR)
    parser.add_argument("--force", action="store_true", help="refit a study whose output exists")
    arguments = parser.parse_args(argv)
    chosen = arguments.studies or targets()
    if arguments.stage == "targets":
        print("\n".join(chosen))
        return 0
    if arguments.stage == "project":
        rows = [project(slug, output=arguments.output) for slug in chosen]
        for row in rows:
            print(json.dumps(row, sort_keys=True), flush=True)
        total = sum(row["projected_cpu_hours"] for row in rows)
        print(f"projected R_a CPU-hours: {total:.2f}", flush=True)
        return 0
    name = SCOPE_FILE if arguments.stage == "scope" else REFITS_FILE
    status = 0
    for slug in chosen:
        if (arguments.output / slug / name).exists() and not arguments.force:
            print(f"{slug}: {name} exists, skipped", flush=True)
            continue
        if (
            arguments.stage == "refits"
            and read_records(arguments.output / slug, SCOPE_FILE) is None
        ):
            print(f"{slug}: no scope records; run the scope stage first", flush=True)
            status = 1
            continue
        try:
            summary = run_study(
                slug,
                stage=arguments.stage,
                jobs=arguments.jobs,
                cache=arguments.cache,
                output=arguments.output,
            )
        except Exception:
            print(f"{slug}: CRASH\n{traceback.format_exc()}", flush=True)
            status = 1
            continue
        print(json.dumps(summary, sort_keys=True), flush=True)
        if summary["failed_keys"]:
            status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
