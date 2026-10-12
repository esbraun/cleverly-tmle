"""Fit a study's replicates so that one failed fit costs one replicate.

RM39 part b.  Every Python replicate loop of a registered study goes through
:func:`fit_replicates`.  Each replicate runs in two steps.  ``draw`` makes the sample; an error
there is a defect of the study's law and stops the run.  ``fit_one`` fits it; an error there is
judged by :func:`cleverly.validation._failures.recordable`.  A recordable error becomes a
failure record with the fields of the reference phase's records
(:data:`tests.studies.evidence.failures.FAILURE_COLUMNS`), and the loop goes on.  Any other
error stops the run.  So does a row that ``validate`` rejects, which is recorded with the class
``invalid_result``, as the reference phase records one.

Under the regeneration driver, each finished replicate is checkpointed in the run's scratch,
which the driver keys by the RM39a1 resume key.  Every fit then runs in a worker process,
never in the driver.  A rerun reuses the checkpoints.  A worker that dies mid-fit, or runs out
of memory, costs the replicates in flight: each runs once more alone, and a second death of
one replicate stops the run.

With no failure, the values are exactly those
:func:`cleverly.utils.parallel.map_parallel` returns for the same worker: each replicate
depends on its own key only, and the values come back in key order.
"""

from __future__ import annotations

import contextlib
import contextvars
import gzip
import hashlib
import json
import os
import pickle
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from cleverly.utils.parallel import map_parallel
from cleverly.validation._failures import failure_call, recordable

#: The class of a recorded fitting failure, and of a fit whose rows broke a row rule.  The
#: reference phase's records use the same words.
FITTING_FAILURE = "fitting_failure"
INVALID_RESULT = "invalid_result"

#: The stage every Python record names: the call that fitted ``cleverly``.
FIT_STAGE = "cleverly fit"
ROW_STAGE = "row check"

#: How many leading replicates of a cell run first.  When every one of them fails, the cause
#: is constant, so the run stops rather than recording it thousands of times (RM39 Q2).
SYSTEMATIC_PREFIX = 20

#: The exit status a worker takes when it runs out of memory, so the run treats it as a kill
#: (RM39a2 Q2: memory exhaustion is a kill, never a recorded failure).
MEMORY_EXIT = 137

#: The prefix length every call uses unless it passes ``_prefix``.  Only
#: ``tests/canonical/python_phase_identity.py`` changes it, to run the two-call split at two
#: replicates for every study.
_PREFIX = SYSTEMATIC_PREFIX

FAILURE_FIELDS = (
    "implementation",
    "group_id",
    "scenarios",
    "replicates",
    "attempt",
    "class",
    "stage",
    "call",
    "message",
    "seed",
)

#: Where the driver keeps per-replicate checkpoints for the current run, or ``None``.  It is
#: read in the driver process and passed to the workers as a path.
_CHECKPOINT_ROOT: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "cleverly_replication_checkpoints", default=None
)
_LABELS: contextvars.ContextVar[set[tuple[str, str]] | None] = contextvars.ContextVar(
    "cleverly_replication_labels", default=None
)


@dataclass(frozen=True)
class PhaseResult:
    """What a study's ``draw_and_fit`` returns to the regeneration driver.

    Parameters
    ----------
    rows : pandas.DataFrame
        The subject's replicate rows, from the fits that succeeded.
    samples : pandas.DataFrame or None
        The samples the reference fits, or ``None`` for a study with no reference.
    truths : pandas.DataFrame or None
        The truth rows the reference reads, or ``None`` with ``samples``.
    failures : pandas.DataFrame or None
        The subject's failure records (:data:`FAILURE_FIELDS`), or ``None`` when none failed.
    reference_input_from_fit : bool
        True when a sample is built from the subject's fit, so a failed subject fit leaves its
        replicate without a sample.  The driver then writes one ``paired_input_missing``
        reference record per subject record.  False when every sample is written, so the
        reference fits the replicates the subject lost.
    """

    rows: pd.DataFrame
    samples: pd.DataFrame | None = None
    truths: pd.DataFrame | None = None
    failures: pd.DataFrame | None = None
    reference_input_from_fit: bool = False


def records(outcomes: Outcomes) -> pd.DataFrame | None:
    """The failure records of ``outcomes``, or ``None`` when every fit succeeded."""
    return None if outcomes.failures.empty else outcomes.failures.reset_index(drop=True)


def invalid_rows(rows: pd.DataFrame | Sequence[Mapping[str, Any]]) -> str | None:
    """The first row rule that ``rows`` break, or ``None``.

    The rules of the reference harness's ``study_invalid`` (RM39a2): a finite ``estimate``,
    ``inference_estimate`` and ``std_error``, a positive ``std_error``, ``ci_lower`` at most
    ``ci_upper``, and ``covered`` equal to the interval's coverage of ``truth``.  Only the
    columns present are read.  A missing value breaks a rule.
    """
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    if frame.empty:
        return None
    columns = set(frame.columns)

    def number(column: str) -> np.ndarray:
        return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)

    def where(broken: np.ndarray) -> str:
        row = int(np.flatnonzero(broken)[0])
        estimand = str(frame["estimand"].iloc[row]) if "estimand" in columns else "row"
        scenario = f" of {frame['scenario'].iloc[row]}" if "scenario" in columns else ""
        return f"{estimand}{scenario} (row {row + 1})"

    for column in ("estimate", "inference_estimate", "std_error"):
        if column in columns and not np.isfinite(number(column)).all():
            return f"non-finite {column} for {where(~np.isfinite(number(column)))}"
    with np.errstate(invalid="ignore"):
        if "std_error" in columns:
            positive = number("std_error") > 0
            if not positive.all():
                return f"non-positive std_error for {where(~positive)}"
        if {"ci_lower", "ci_upper"} <= columns:
            ordered = number("ci_lower") <= number("ci_upper")
            if not ordered.all():
                return f"ci_lower above ci_upper for {where(~ordered)}"
        if {"covered", "truth", "ci_lower", "ci_upper"} <= columns:
            truth = number("truth")
            inside = (number("ci_lower") <= truth) & (truth <= number("ci_upper"))
            agrees = number("covered") == inside.astype(float)
            if not agrees.all():
                return f"covered disagrees with the interval and truth for {where(~agrees)}"
    return None


class SystematicFailure(RuntimeError):
    """Every leading replicate of one cell failed, so the run stops."""


class KilledReplicate(RuntimeError):
    """A replicate's worker died, and died again when the replicate ran alone."""


@contextlib.contextmanager
def checkpoints(root: Path | None) -> Iterator[None]:
    """Keep the checkpoints of every :func:`fit_replicates` call inside under ``root``."""
    token = _CHECKPOINT_ROOT.set(root)
    used = _LABELS.set(set())
    try:
        yield
    finally:
        _LABELS.reset(used)
        _CHECKPOINT_ROOT.reset(token)


def checkpoint_root() -> Path | None:
    """The checkpoint root of the current run, or ``None`` outside the driver."""
    return _CHECKPOINT_ROOT.get()


@dataclass
class Outcomes:
    """What :func:`fit_replicates` returns, in key order.

    Parameters
    ----------
    keys : list
        The keys, in the order given.
    fitted : list
        Per key, the fit's value, or the value of ``on_failure`` where the fit failed.
    failed : list of int
        The positions of the keys whose fit failed.
    failures : pandas.DataFrame
        One record per failed key, in key order, with the columns of :data:`FAILURE_FIELDS`.
    reused : int
        How many keys came from a checkpoint.
    """

    keys: list[Hashable]
    fitted: list[Any]
    failed: list[int] = field(default_factory=list)
    failures: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=list(FAILURE_FIELDS))
    )
    reused: int = 0

    @property
    def values(self) -> list[Any]:
        """The fit values of the keys that succeeded, in key order."""
        failed = set(self.failed)
        return [value for index, value in enumerate(self.fitted) if index not in failed]


def _digest(parts: Sequence[Any]) -> str:
    text = json.dumps([str(part) for part in parts], separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class _Task:
    key: Any
    replicate: int
    cell: str
    seed: int | None
    ident: str


@dataclass(frozen=True)
class _Plan:
    fit_one: Callable[[Any, Any], Any]
    draw: Callable[[Any], Any] | None
    on_failure: Callable[[Any, Any], Any] | None
    validate: Callable[[Any], str | None] | None
    label: str
    implementation: str
    directory: Path | None


#: One replicate's result: its value (or the ``on_failure`` value) and its record, if any.
_Outcome = tuple[Any, dict[str, Any] | None]


def _text(message: str) -> str:
    return message.replace("\r", " ").replace("\n", " ")


def _record(plan: _Plan, task: _Task, kind: str, stage: str, call: str, message: str) -> dict:
    return {
        "implementation": plan.implementation,
        "group_id": f"{plan.label}/{task.cell}/r{task.replicate:09d}",
        "scenarios": task.cell,
        "replicates": str(task.replicate),
        "attempt": 1 + _kills(plan, task),
        "class": kind,
        "stage": stage,
        "call": call,
        "message": _text(message),
        "seed": "NA" if task.seed is None else str(task.seed),
    }


def _path(plan: _Plan, task: _Task, suffix: str) -> Path:
    assert plan.directory is not None
    return plan.directory / f"{task.ident}{suffix}"


def _kills(plan: _Plan, task: _Task) -> int:
    if plan.directory is None:
        return 0
    path = _path(plan, task, ".kills")
    return int(path.read_text(encoding="utf-8")) if path.exists() else 0


def _load(plan: _Plan, task: _Task) -> _Outcome | None:
    if plan.directory is None:
        return None
    path = _path(plan, task, ".pkl.gz")
    if not path.exists():
        return None
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)  # the run's own scratch


def _fit(plan: _Plan, task: _Task) -> _Outcome:
    drawn = None if plan.draw is None else plan.draw(task.key)
    try:
        value = plan.fit_one(task.key, drawn)
    except MemoryError:
        if plan.directory is not None:
            # A kill, as the reference phase treats memory exhaustion: no checkpoint, the
            # marker stays, and the driver runs the replicate again alone.
            os._exit(MEMORY_EXIT)
        raise
    except Exception as error:
        if not recordable(error):
            raise
        record = _record(
            plan,
            task,
            FITTING_FAILURE,
            FIT_STAGE,
            failure_call(error),
            f"{type(error).__name__}: {error}",
        )
        return (None if plan.on_failure is None else plan.on_failure(task.key, drawn)), record
    broken = None if plan.validate is None else plan.validate(value)
    if broken is not None:
        record = _record(plan, task, INVALID_RESULT, ROW_STAGE, "", broken)
        return (None if plan.on_failure is None else plan.on_failure(task.key, drawn)), record
    return value, None


def _work(plan: _Plan, task: _Task) -> _Outcome:
    """Fit one replicate in a worker; checkpoint it when the run keeps checkpoints."""
    stored = _load(plan, task)
    if stored is not None:
        return stored
    if plan.directory is None:
        return _fit(plan, task)
    marker = _path(plan, task, ".started")
    marker.write_text(f"{os.getpid()}\n", encoding="utf-8")
    outcome = _fit(plan, task)
    temporary = _path(plan, task, ".tmp")
    with gzip.open(temporary, "wb", compresslevel=1) as handle:
        pickle.dump(outcome, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, _path(plan, task, ".pkl.gz"))
    marker.unlink(missing_ok=True)
    return outcome


def _count_kills(plan: _Plan, tasks: Sequence[_Task]) -> None:
    """Count one kill for every task that started and never finished, and clear its marker.

    A pool that loses a worker aborts every task in flight, and a driver killed whole leaves
    the markers of its last tasks, so a counted kill is not proof that this task killed its
    worker.  It earns the task a run alone, where only its own death can stop it.
    """
    assert plan.directory is not None
    for task in tasks:
        marker = _path(plan, task, ".started")
        if marker.exists() and not _path(plan, task, ".pkl.gz").exists():
            _path(plan, task, ".kills").write_text(f"{_kills(plan, task) + 1}\n", encoding="utf-8")
        marker.unlink(missing_ok=True)
        _path(plan, task, ".tmp").unlink(missing_ok=True)


def _pool(plan: _Plan, tasks: Sequence[_Task], jobs: int) -> list[_Outcome]:
    # Two workers at least, so no fit runs in the driver process.  Results do not depend on
    # the worker count (``tests/unit/test_parallel_invariance.py``).
    runner = Parallel(n_jobs=max(2, jobs))
    return list(runner(delayed(_work)(plan, task) for task in tasks))


def _run(plan: _Plan, tasks: Sequence[_Task], n_jobs: int) -> dict[str, _Outcome]:
    """Every task's outcome, surviving killed workers when the run keeps checkpoints."""
    if plan.directory is None:
        outcomes = map_parallel(_work, [(plan, task) for task in tasks], n_jobs=n_jobs)
        return {task.ident: outcome for task, outcome in zip(tasks, outcomes, strict=True)}
    done: dict[str, _Outcome] = {}
    jobs = n_jobs
    while True:
        remaining = [task for task in tasks if task.ident not in done]
        if not remaining:
            return done
        for task in [task for task in remaining if _kills(plan, task)]:
            try:
                done[task.ident] = _pool(plan, [task], 1)[0]
            except BrokenProcessPool as error:
                raise KilledReplicate(
                    f"replicate {task.replicate} of {plan.label} ({task.cell}) was killed "
                    f"again when it ran alone; attempts: {_kills(plan, task) + 1}"
                ) from error
        remaining = [task for task in remaining if task.ident not in done]
        if not remaining:
            return done
        try:
            outcomes = _pool(plan, remaining, jobs)
        except BrokenProcessPool:
            _count_kills(plan, remaining)
            for task in remaining:
                stored = _load(plan, task)
                if stored is not None:
                    done[task.ident] = stored
            jobs = max(1, jobs // 2)
            continue
        done.update(
            {task.ident: outcome for task, outcome in zip(remaining, outcomes, strict=True)}
        )
        return done


def fit_replicates(
    keys: Sequence[Hashable],
    fit_one: Callable[[Any, Any], Any],
    *,
    n_jobs: int,
    label: str,
    implementation: str,
    cell_of: Callable[[Any], str],
    replicate_of: Callable[[Any], int],
    draw: Callable[[Any], Any] | None = None,
    on_failure: Callable[[Any, Any], Any] | None = None,
    validate: Callable[[Any], str | None] | None = None,
    seed_of: Callable[[Any], int | None] | None = None,
    checkpoint: Path | Literal["ambient"] | None = "ambient",
    _prefix: int | None = None,
) -> Outcomes:
    """Draw and fit every key, recording each recordable fitting failure.

    ``fit_one(key, draw(key))`` fits one key (``draw`` defaults to returning ``None``).  When
    it raises a recordable error, or ``validate`` names a broken rule in its value, the key's
    value is ``on_failure(key, draw(key))`` (``None`` without it) and a record is kept.

    The first ``SYSTEMATIC_PREFIX`` keys of every cell run first.  When each of them failed in
    one cell, the run stops with :class:`SystematicFailure`.  ``_prefix`` lets a test run that
    split at two replicates; studies never pass it.
    """
    prefix = _PREFIX if _prefix is None else _prefix
    if not keys:
        raise ValueError(f"{label}: no replicates to fit")
    tasks = [
        _Task(
            key=key,
            replicate=int(replicate_of(key)),
            cell=str(cell_of(key)),
            seed=None if seed_of is None else seed_of(key),
            ident=_digest((implementation, label, cell_of(key), replicate_of(key))),
        )
        for key in keys
    ]
    if len({task.ident for task in tasks}) != len(tasks):
        raise ValueError(f"{label}: two keys share a cell and a replicate")
    root = checkpoint_root() if checkpoint == "ambient" else checkpoint
    directory = None if root is None else Path(root) / implementation / label
    used = _LABELS.get()
    if directory is not None and used is not None:
        if (implementation, label) in used:
            raise ValueError(f"{implementation}/{label}: the label is used twice in one run")
        used.add((implementation, label))
    plan = _Plan(fit_one, draw, on_failure, validate, label, implementation, directory)
    reused = 0
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
        _count_kills(plan, tasks)
        reused = sum(_path(plan, task, ".pkl.gz").exists() for task in tasks)
        if reused:
            print(f"{implementation}/{label}: reusing {reused} checkpointed replicates", flush=True)
    by_cell: dict[str, list[_Task]] = {}
    for task in tasks:
        by_cell.setdefault(task.cell, []).append(task)
    done = _run(plan, [task for cell in by_cell.values() for task in cell[:prefix]], n_jobs)
    for cell, members in by_cell.items():
        head = members[:prefix]
        records = [done[task.ident][1] for task in head]
        if len(head) == SYSTEMATIC_PREFIX and all(records):
            first = records[0]
            assert first is not None
            raise SystematicFailure(
                f"{label}: the first {SYSTEMATIC_PREFIX} replicates of {cell} failed; "
                f"first: {first['message']}"
            )
    rest = [task for task in tasks if task.ident not in done]
    if rest:
        done.update(_run(plan, rest, n_jobs))
    ordered = [done[task.ident] for task in tasks]
    records = [(index, record) for index, (_, record) in enumerate(ordered) if record]
    if len(records) == len(ordered):
        raise RuntimeError(f"every replicate of {label} failed; first: {records[0][1]['message']}")
    return Outcomes(
        keys=list(keys),
        fitted=[value for value, _ in ordered],
        failed=[index for index, _ in records],
        failures=pd.DataFrame([record for _, record in records], columns=list(FAILURE_FIELDS)),
        reused=reused,
    )
