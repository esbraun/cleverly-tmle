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

RM39 part b2 runs every property loop through :func:`fit_properties`.  A property key may
write several cells, and its record names each of them.  The regeneration driver collects the
records with :func:`recording`.  Outside it, a recorded property failure stops the run.
"""

from __future__ import annotations

import contextlib
import contextvars
import gzip
import hashlib
import json
import os
import pickle
import tempfile
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
    samples : pandas.DataFrame, pathlib.Path or None
        The samples the reference fits, or ``None`` for a study with no reference.  A path names
        an assembled ``samples.csv.gz`` (:func:`assembled_samples`), which the driver moves into
        place instead of writing a frame, so a study need not hold every sample in memory.
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
    samples: pd.DataFrame | Path | None = None
    truths: pd.DataFrame | None = None
    failures: pd.DataFrame | None = None
    reference_input_from_fit: bool = False


def records(outcomes: Outcomes) -> pd.DataFrame | None:
    """The failure records of ``outcomes``, or ``None`` when every fit succeeded."""
    return None if outcomes.failures.empty else outcomes.failures.reset_index(drop=True)


def invalid_rows(
    rows: pd.DataFrame | Sequence[Mapping[str, Any]], *, coverage: bool = True
) -> str | None:
    """The first row rule that ``rows`` break, or ``None``.

    The rules of the reference harness's ``study_invalid`` (RM39a2): a finite ``estimate``,
    ``inference_estimate`` and ``std_error``, a positive ``std_error``, ``ci_lower`` at most
    ``ci_upper``, and ``covered`` equal to the interval's coverage of ``truth``.  Only the
    columns present are read.  A missing value breaks a rule.  ``coverage=False`` drops the
    last rule: the study computes a subject row's ``covered`` itself, so a disagreement there
    is a study defect, and ``schema.validate_replicates`` stops the run on it.
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
        if coverage and {"covered", "truth", "ci_lower", "ci_upper"} <= columns:
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
    substituted : bool
        True when an ``on_failure`` value stands in for each failed key.
    spill_directory : pathlib.Path or None
        Where :func:`fit_primary` spilled the samples (``spill_samples_at``), or ``None``.
    """

    keys: list[Hashable]
    fitted: list[Any]
    failed: list[int] = field(default_factory=list)
    failures: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=list(FAILURE_FIELDS))
    )
    reused: int = 0
    substituted: bool = False
    spill_directory: Path | None = None

    @property
    def values(self) -> list[Any]:
        """The fit values of the keys that succeeded, in key order."""
        failed = set(self.failed)
        return [value for index, value in enumerate(self.fitted) if index not in failed]

    @property
    def kept(self) -> list[Any]:
        """``fitted`` when an ``on_failure`` value stands in for each failed key, else ``values``.

        A study builds its samples, truths and rows from these.  A failed key then keeps its
        sample when the study can build it without the fit, and loses it otherwise.
        """
        return self.fitted if self.substituted else self.values


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
    scenarios: str


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
        "scenarios": task.scenarios,
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
    except _Recorded as recorded:
        # The fit's own handler already judged the error recordable; keep its call and words.
        failure = recorded.failure
        record = _record(
            plan,
            task,
            FITTING_FAILURE,
            FIT_STAGE,
            str(failure.call),
            f"{failure.error_type}: {failure.message}",
        )
        return _substitute(plan, task), record
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
        return _substitute(plan, task), record
    broken = None if plan.validate is None else plan.validate(value)
    if broken is not None:
        record = _record(plan, task, INVALID_RESULT, ROW_STAGE, "", broken)
        return _substitute(plan, task), record
    return value, None


def _substitute(plan: _Plan, task: _Task) -> Any:
    """The ``on_failure`` value of a failed key, from a fresh draw.

    The failed fit may have changed the drawn objects in place, and the draw is a function of
    the key alone, so the key is drawn again.
    """
    if plan.on_failure is None:
        return None
    return plan.on_failure(task.key, None if plan.draw is None else plan.draw(task.key))


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
    scenarios_of: Callable[[Any], str] | None = None,
    _prefix: int | None = None,
) -> Outcomes:
    """Draw and fit every key, recording each recordable fitting failure.

    ``fit_one(key, draw(key))`` fits one key (``draw`` defaults to returning ``None``).  When
    it raises a recordable error, or ``validate`` names a broken rule in its value, the key's
    value is ``on_failure(key, draw(key))`` (``None`` without it) and a record is kept.  A
    record's ``scenarios`` is ``scenarios_of(key)``, which defaults to ``cell_of(key)``.

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
            scenarios=str((scenarios_of or cell_of)(key)),
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
        substituted=on_failure is not None,
    )


def fit_primary(
    keys: Sequence[Sequence[Any]],
    fit_one: Callable[[Any, Any], Any],
    *,
    draw: Callable[[Any], Any],
    n_jobs: int,
    implementation: str,
    reference_input: Callable[[Any, Any], Any] | None = None,
    rows_at: int | None = 2,
    seed_of: Callable[[Any], int | None] | None = None,
    cell_of: Callable[[Any], str] | None = None,
    replicate_of: Callable[[Any], int] | None = None,
    spill_samples_at: int | None = None,
) -> Outcomes:
    """:func:`fit_replicates` for a study's primary loop, with the defaults most studies share.

    A key is ``(scenario, replicate, ...)`` unless ``cell_of`` and ``replicate_of`` say
    otherwise.  ``fit_one`` returns the study's value for one key; ``rows_at`` is the position
    of the subject's rows in it (``None`` when the value is the rows).  The rows are checked by
    :func:`invalid_rows` without the coverage rule.  ``reference_input`` builds a failed key's value without the fit, so
    the reference still fits that replicate.  A study whose reference input needs the fit
    passes none.

    ``spill_samples_at`` is the position, in the value, of a sample frame.  Each worker writes
    that frame to disk when its fit ends and keeps a :class:`SpilledSample` in its place, so
    neither the checkpoint nor the driver holds a frame, and memory does not grow with the
    number of replicates.  Under the driver the files sit beside the checkpoints, so a rerun
    that reuses a checkpoint reuses its file.  Elsewhere they sit in a new temporary directory.
    :func:`assembled_samples` joins them.
    """
    directory = None
    if spill_samples_at is not None:
        root = checkpoint_root()
        if root is None:
            directory = Path(tempfile.mkdtemp(prefix="cleverly-samples-"))
        else:
            directory = Path(root) / implementation / "primary-samples"
            directory.mkdir(parents=True, exist_ok=True)
        fit_one = _Spilling(fit_one, spill_samples_at, directory)
        if reference_input is not None:
            reference_input = _Spilling(reference_input, spill_samples_at, directory)
    outcomes = fit_replicates(
        keys,
        fit_one,
        draw=draw,
        on_failure=reference_input,
        validate=_RowCheck(rows_at),
        n_jobs=n_jobs,
        label="primary",
        implementation=implementation,
        cell_of=cell_of or _first,
        replicate_of=replicate_of or _second,
        seed_of=seed_of,
    )
    outcomes.spill_directory = directory
    return outcomes


@dataclass(frozen=True)
class SpilledSample:
    """A sample frame that a worker wrote to disk, in the place of the frame.

    Parameters
    ----------
    path : pathlib.Path
        The gzip pickle that holds the frame.
    """

    path: Path

    def load(self) -> pd.DataFrame:
        """The frame."""
        with gzip.open(self.path, "rb") as handle:
            frame = pickle.load(handle)  # written by this run's own workers
        assert isinstance(frame, pd.DataFrame)
        return frame


@dataclass(frozen=True)
class _Spilling:
    """Wrap a fit so that the sample at position ``at`` of its value goes to ``directory``."""

    fit_one: Callable[[Any, Any], Any]
    at: int
    directory: Path

    def __call__(self, key: Any, drawn: Any) -> Any:
        value = list(self.fit_one(key, drawn))
        path = self.directory / f"{_digest(('samples', repr(key)))}.pkl.gz"
        temporary = path.with_suffix(".tmp")
        with gzip.open(temporary, "wb", compresslevel=1) as handle:
            pickle.dump(value[self.at], handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)  # a checkpoint never names a half-written file
        value[self.at] = SpilledSample(path)
        return tuple(value)


def _representative(frame: pd.DataFrame) -> pd.DataFrame:
    """One row of ``frame`` with each column's first non-missing value, and its dtype.

    ``pd.concat`` picks a column's dtype from the dtypes and the missing-ness of the pieces,
    never from the number of rows, so this row stands in for the whole frame.
    """
    columns = {}
    for name in frame.columns:
        column = frame[name]
        valid = column.dropna()
        columns[name] = (valid if len(valid) else column).iloc[:1].reset_index(drop=True)
    return pd.DataFrame(columns)


def assembled_samples(outcomes: Outcomes, *, at: int = 0) -> Path:
    """Join the spilled samples of ``outcomes`` into one ``samples.csv.gz`` and return its path.

    The file holds what ``write_csv(pd.concat(frames, ignore_index=True), path,
    compression={"method": "gzip", "mtime": 0})`` writes for the frames in key order, and the
    same bytes, but only one frame is in memory at a time.  The columns are the union in order
    of first appearance, and each column takes the dtype ``pd.concat`` gives it.  The file is
    in the spill directory; the driver moves it into place.

    Parameters
    ----------
    outcomes : Outcomes
        The result of :func:`fit_primary` called with ``spill_samples_at``.
    at : int
        The position of the :class:`SpilledSample` in each kept value.

    Returns
    -------
    pathlib.Path
        The assembled file.
    """
    from tests.studies.evidence.manifest import NEWLINE

    if outcomes.spill_directory is None:
        raise ValueError("these outcomes spilled no samples; pass spill_samples_at to fit_primary")
    chunks = [value[at] for value in outcomes.kept]
    representatives: dict[tuple[Any, ...], pd.DataFrame] = {}
    for chunk in chunks:
        frame = chunk.load()
        signature = tuple(
            (name, str(frame[name].dtype), frame[name].isna().all()) for name in frame
        )
        representatives.setdefault(signature, _representative(frame))
        del frame
    template = pd.concat(list(representatives.values()), ignore_index=True)
    destination = outcomes.spill_directory / "samples.csv.gz"
    temporary = destination.with_name("assembling.csv.gz")
    # pandas opens a gzip file the same way, so the header names the file as it does.
    with temporary.open("wb") as raw:
        stream = gzip.GzipFile(filename=str(destination), fileobj=raw, mode="wb", mtime=0)
        with stream:
            for index, chunk in enumerate(chunks):
                frame = chunk.load().reindex(columns=template.columns)
                frame = frame.astype(template.dtypes.to_dict())
                # Text, not a handle: pandas flushes a handle after each call, which would
                # add a sync-flush marker per chunk to the stream.
                text = frame.to_csv(None, index=False, lineterminator=NEWLINE, header=index == 0)
                stream.write(text.encode("utf-8"))
                del frame
            stream.flush()  # pandas ends its gzip stream with this sync flush too
    os.replace(temporary, destination)
    return destination


def load_samples(samples: pd.DataFrame | Path) -> pd.DataFrame:
    """The sample frame of a :class:`PhaseResult`, read from disk when it is a path."""
    return samples if isinstance(samples, pd.DataFrame) else pd.read_csv(samples)


def result(
    outcomes: Outcomes,
    rows: pd.DataFrame,
    *,
    samples: pd.DataFrame | Path | None = None,
    truths: pd.DataFrame | None = None,
) -> PhaseResult:
    """The :class:`PhaseResult` of a primary loop whose frames were built from ``outcomes.kept``."""
    return PhaseResult(
        rows=rows,
        samples=samples,
        truths=truths,
        failures=records(outcomes),
        reference_input_from_fit=samples is not None and not outcomes.substituted,
    )


def _first(key: Sequence[Any]) -> str:
    return str(key[0])


def _second(key: Sequence[Any]) -> int:
    return int(key[1])


@dataclass(frozen=True)
class _RowCheck:
    rows_at: int | None

    def __call__(self, value: Any) -> str | None:
        rows = value if self.rows_at is None else value[self.rows_at]
        return invalid_rows(rows, coverage=False)


# ---- the property phase (RM39 part b2) --------------------------------------------------------

#: The implementation name a property record carries when no driver collector names the subject.
PROPERTY_IMPLEMENTATION = "properties"

#: What a property failure raises when no driver collector is open: tests, diagnostics and the
#: drivers outside the regeneration run keep the all-fatal behaviour they always had.
NO_COLLECTOR = "a property failure outside the regeneration driver has no record file"


class _Recorded(Exception):
    """A fit that its own handler already recorded, carried to :func:`fit_replicates`.

    ``CoverageStudy.run_replication`` returns a :class:`~cleverly.validation.ReplicationFailure`
    for a recordable error instead of raising it.  A property worker raises this with that
    record, and the record keeps the call and the message the classifier saw.  The class is
    not recordable itself, so it never reaches a record any other way.
    """

    def __init__(self, failure: Any) -> None:
        super().__init__(f"{failure.error_type}: {failure.message}")
        self.failure = failure

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (self.failure,))


class CellsOfMismatch(RuntimeError):
    """A property key wrote cells other than the ones its ``cells_of`` names.

    A subclass, so :func:`cleverly.validation._failures.recordable` never records it: a wrong
    ``cells_of`` would charge a failure to the wrong cells, which is a defect of the study.
    """


#: What a witness row reports when the loop's sample is drawn inside its fit, so no draw
#: exists to compute a statistic of without fitting (RM39b2 rev 3, D3'f).
DRAW_NOT_SEPARABLE = "draw not separable"

#: The columns of the witness values a run with a property failure publishes: one row per
#: key of every cell that lost a replicate, survivors included, so the accounting can place
#: each lost draw among the draws that survived.
WITNESS_COLUMNS = ("property", "cell", "replicate", "statistic", "value")


@dataclass(frozen=True)
class Witness:
    """A draw-level statistic of a property sample, computed without a fit (D3'f).

    A fit that raises is plausibly fitting an extreme draw.  The draw is a function of the key
    alone, so the accounting can regenerate the sample of every lost replicate and place its
    statistic among the survivors'.  The statistic is reported, not gated.

    Parameters
    ----------
    name : str
        What the statistic measures, as the accounting publishes it.
    statistic : Callable
        ``statistic(drawn)`` of one key's drawn sample, a float.
    """

    name: str
    statistic: Callable[[Any], float]


@dataclass
class Recording:
    """The property records of one regeneration run, in the order their loops finished.

    Parameters
    ----------
    implementation : str
        The subject the records name.
    frames : list of pandas.DataFrame
        One frame of records per property loop that recorded a failure.
    labels : list of str
        Every label a property loop used in this run, in order.
    witness : Witness or None
        The study's ``PROPERTY_WITNESS``; ``None`` records no witness values.
    witnesses : list of pandas.DataFrame
        One frame of :data:`WITNESS_COLUMNS` per loop that recorded a failure.
    """

    implementation: str
    frames: list[pd.DataFrame] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    witness: Witness | None = None
    witnesses: list[pd.DataFrame] = field(default_factory=list)

    @property
    def failures(self) -> pd.DataFrame | None:
        """Every record of the run, or ``None`` when no property fit failed."""
        present = [frame for frame in self.frames if not frame.empty]
        if not present:
            return None
        return pd.concat(present, ignore_index=True).loc[:, list(FAILURE_FIELDS)]

    @property
    def witness_values(self) -> pd.DataFrame | None:
        """Every witness value of the run, or ``None`` when no property fit failed."""
        present = [frame for frame in self.witnesses if not frame.empty]
        if not present:
            return None
        return (
            pd.concat(present, ignore_index=True)
            .loc[:, list(WITNESS_COLUMNS)]
            .sort_values(["property", "cell", "replicate"], ignore_index=True)
        )


_RECORDING: contextvars.ContextVar[Recording | None] = contextvars.ContextVar(
    "cleverly_property_recording", default=None
)


@contextlib.contextmanager
def recording(implementation: str, *, witness: Witness | None = None) -> Iterator[Recording]:
    """Collect the records of every :func:`fit_properties` call inside.

    The regeneration driver opens one around its property phase.  Outside it, a recorded
    property failure stops the run (:data:`NO_COLLECTOR`).

    Parameters
    ----------
    implementation : str
        The subject the records name.
    witness : Witness or None
        The study's ``PROPERTY_WITNESS``.  A loop that recorded a failure computes it on the
        draw of every key of each cell that lost a replicate.

    Yields
    ------
    Recording
        The run's records, filled as each property loop finishes.
    """
    collected = Recording(implementation, witness=witness)
    token = _RECORDING.set(collected)
    try:
        yield collected
    finally:
        _RECORDING.reset(token)


def recorded_failures() -> pd.DataFrame | None:
    """The failure records of the property loops that have finished inside the open recording.

    A study's ``generate_property_artifacts`` hook reads it after its :func:`fit_properties` call
    returns, to publish a reading of the refusals beside the rows.

    Returns
    -------
    pandas.DataFrame or None
        The records in :data:`FAILURE_FIELDS`, or ``None`` when no recording is open or no
        property fit failed.
    """
    collected = _RECORDING.get()
    return None if collected is None else collected.failures


def _rows_of(value: Any, rows_at: int | None) -> pd.DataFrame:
    """The rows a property worker returned, as one frame.

    A worker returns a frame, a row dictionary, or a sequence of row dictionaries; ``rows_at``
    is their position in a tuple return.
    """
    rows = value if rows_at is None else value[rows_at]
    if isinstance(rows, pd.DataFrame):
        return rows
    if isinstance(rows, Mapping):
        return pd.DataFrame([rows])
    return pd.DataFrame(list(rows))


def _written(rows: pd.DataFrame) -> set[str]:
    if rows.empty:
        return set()
    if not {"property", "cell"} <= set(rows.columns):
        raise CellsOfMismatch("a property worker returned rows without property and cell")
    return {f"{prop}/{cell}" for prop, cell in zip(rows["property"], rows["cell"], strict=True)}


@dataclass(frozen=True)
class _PropertyFit:
    """A property worker whose rows must be exactly the cells its key declares (F4)."""

    fit_one: Callable[[Any, Any], Any]
    rows_at: int | None
    cells_of: Callable[[Any], Sequence[str]]
    derived: frozenset[str]

    def __call__(self, key: Any, drawn: Any) -> Any:
        value = self.fit_one(key, drawn)
        written = _written(_rows_of(value, self.rows_at))
        declared = set(self.cells_of(key)) - self.derived
        if written != declared:
            raise CellsOfMismatch(
                f"key {key!r} wrote {sorted(written)}, but its cells_of names "
                f"{sorted(declared)} besides its derived controls"
            )
        return value


@dataclass(frozen=True)
class _PropertyRowCheck:
    """The OQ5 row rules on a property worker's rows, plus a study's ``extra`` rule."""

    rows_at: int | None
    extra: Callable[[pd.DataFrame], str | None] | None

    def __call__(self, value: Any) -> str | None:
        rows = _rows_of(value, self.rows_at)
        broken = invalid_rows(rows, coverage=False)
        if broken is None and self.extra is not None:
            broken = self.extra(rows)
        return broken


@dataclass(frozen=True)
class _Splat:
    """A ``map_parallel`` worker as a ``fit_one``: the key is its argument list."""

    worker: Callable[..., Any]

    def __call__(self, key: Any, drawn: Any) -> Any:
        return self.worker(*key) if isinstance(key, tuple) else self.worker(key)


def splat(worker: Callable[..., Any]) -> Callable[[Any, Any], Any]:
    """``worker`` as a ``fit_one`` for :func:`fit_properties`, with ``map_parallel``'s calling rule.

    A tuple key is splatted as positional arguments and any other key is the one argument, so
    a loop that passed ``payloads`` to ``map_parallel`` passes the same list as its keys.  The
    drawn value is ignored.

    Parameters
    ----------
    worker : Callable
        The per-replicate worker, a module-level function so that it pickles by name.

    Returns
    -------
    Callable
        A ``fit_one(key, drawn)``.
    """
    return _Splat(worker)


@dataclass(frozen=True)
class _SplatDrawn:
    """A worker that takes its key's arguments and then the drawn sample, as a ``fit_one``."""

    worker: Callable[..., Any]

    def __call__(self, key: Any, drawn: Any) -> Any:
        return self.worker(*key, drawn) if isinstance(key, tuple) else self.worker(key, drawn)


@dataclass(frozen=True)
class _SplatDraw:
    """A sampler that takes its key's arguments, as a ``draw``."""

    sampler: Callable[..., Any]

    def __call__(self, key: Any) -> Any:
        return self.sampler(*key) if isinstance(key, tuple) else self.sampler(key)


def splat_drawn(worker: Callable[..., Any]) -> Callable[[Any, Any], Any]:
    """A split worker as a ``fit_one``: the key's arguments, then the drawn sample (F5).

    The calling rule is :func:`splat`'s.  Pair it with :func:`splat_draw` as ``draw=``, so the
    draw runs outside the classifier: an error in it stops the run, and the accounting can
    draw a lost key's sample again for its witness (D3'f).

    Parameters
    ----------
    worker : Callable
        The per-replicate fit, a module-level function so that it pickles by name.

    Returns
    -------
    Callable
        A ``fit_one(key, drawn)``.
    """
    return _SplatDrawn(worker)


def splat_draw(sampler: Callable[..., Any]) -> Callable[[Any], Any]:
    """A sampler that takes the key's arguments, as a ``draw`` for :func:`fit_properties`.

    Parameters
    ----------
    sampler : Callable
        The per-replicate draw, a module-level function so that it pickles by name.

    Returns
    -------
    Callable
        A ``draw(key)``.
    """
    return _SplatDraw(sampler)


def _cell_names(cells: Sequence[str], derived: Mapping[str, Sequence[str]], key: Any) -> list[str]:
    """``cells`` checked as ``property/cell`` names, each derived control beside its base."""
    names = [str(name) for name in cells]
    for name in names:
        if name.count("/") != 1 or ";" in name:
            raise ValueError(f"key {key!r}: {name!r} is not a property/cell name")
    if len(set(names)) != len(names):
        raise ValueError(f"key {key!r}: cells_of names a cell twice: {names}")
    for base, controls in derived.items():
        named = [control in names for control in controls]
        if any(named) and not (base in names and all(named)):
            raise ValueError(
                f"key {key!r}: cells_of names a derived control of {base!r} without the base "
                f"and every control {list(controls)}"
            )
        if base in names and not all(named):
            raise ValueError(
                f"key {key!r}: cells_of names {base!r} without its derived controls "
                f"{list(controls)}"
            )
    return names


def fit_properties(
    keys: Sequence[Hashable],
    fit_one: Callable[[Any, Any], Any],
    *,
    cells_of: Callable[[Any], Sequence[str]],
    cell_of: Callable[[Any], str],
    replicate_of: Callable[[Any], int],
    n_jobs: int,
    label: str = "properties",
    draw: Callable[[Any], Any] | None = None,
    rows_at: int | None = None,
    derived: Mapping[str, Sequence[str]] | None = None,
    extra: Callable[[pd.DataFrame], str | None] | None = None,
    seed_of: Callable[[Any], int | None] | None = None,
    checkpoint: Path | Literal["ambient"] | None = "ambient",
) -> list[Any]:
    """Fit a property loop so that one failed fit costs the cells its key writes, not the run.

    :func:`fit_replicates` with the property phase's rules.  ``cell_of(key)`` is the cell the
    systematic prefix reads: the fit set, or the property cell.  ``cells_of(key)`` lists the
    ``property/cell`` names whose rows the key writes, its derived controls included, and a
    record names all of them.  Each successful key's rows must name exactly ``cells_of(key)``
    without the controls ``derived`` lists (base cell to its controls).  Any other set stops
    the run with :class:`CellsOfMismatch`.  The rows pass :func:`invalid_rows` without the
    coverage rule, then ``extra``.  A broken rule is an ``invalid_result`` record.

    The records go to the collector that :func:`recording` opens.  Without one, a record
    stops the run with ``RuntimeError`` (:data:`NO_COLLECTOR`).

    Parameters
    ----------
    keys : Sequence of Hashable
        One key per fit, in the order the rows are published.
    fit_one : Callable
        ``fit_one(key, draw(key))`` returns the key's rows, or a tuple that holds them.
    cells_of : Callable
        The ``property/cell`` names that a key's rows and derived controls fill.
    cell_of : Callable
        The cell whose leading replicates the systematic prefix reads.
    replicate_of : Callable
        The key's replicate index.
    n_jobs : int
        Worker processes.
    label : str
        ``properties`` for a study's main loop, ``properties/<name>`` for any other.
    draw : Callable or None
        Draws a key's sample outside classification, so an error there stops the run.
    rows_at : int or None
        The position of the rows in a tuple return; ``None`` when the value is the rows.
    derived : Mapping or None
        Each base cell's derived controls, which a key names but its worker never returns.
    extra : Callable or None
        A further row rule, which returns the broken rule's text or ``None``.
    seed_of : Callable or None
        The seed a record reports.
    checkpoint : Path, "ambient" or None
        Where finished keys are kept; ``"ambient"`` is the driver's run scratch.

    Returns
    -------
    list
        The values of the keys that succeeded, in key order.
    """
    declared = dict(derived or {})
    controls = frozenset(name for names in declared.values() for name in names)
    collected = _RECORDING.get()
    implementation = PROPERTY_IMPLEMENTATION if collected is None else collected.implementation
    if collected is not None:
        if label in collected.labels:
            raise ValueError(f"{implementation}/{label}: the label is used twice in one run")
        collected.labels.append(label)

    def scenarios(key: Any) -> str:
        return ";".join(_cell_names(cells_of(key), declared, key))

    outcomes = fit_replicates(
        keys,
        _PropertyFit(fit_one, rows_at, cells_of, controls),
        n_jobs=n_jobs,
        label=label,
        implementation=implementation,
        cell_of=cell_of,
        replicate_of=replicate_of,
        draw=draw,
        validate=_PropertyRowCheck(rows_at, extra),
        seed_of=seed_of,
        checkpoint=checkpoint,
        scenarios_of=scenarios,
    )
    if not outcomes.failures.empty:
        if collected is None:
            first = outcomes.failures.iloc[0]
            raise RuntimeError(f"{NO_COLLECTOR}: {first['group_id']}: {first['message']}")
        collected.frames.append(outcomes.failures)
        collected.witnesses.append(
            _witness_values(
                keys,
                outcomes.failures,
                cells_of=lambda key: _cell_names(cells_of(key), declared, key),
                replicate_of=replicate_of,
                draw=draw,
                witness=collected.witness,
                n_jobs=n_jobs,
            )
        )
    return outcomes.values


@dataclass(frozen=True)
class _WitnessOf:
    """One key's witness value: the statistic of its regenerated draw."""

    draw: Callable[[Any], Any]
    statistic: Callable[[Any], float]

    def __call__(self, key: Any) -> float:
        return float(self.statistic(self.draw(key)))


def _witness_values(
    keys: Sequence[Hashable],
    failures: pd.DataFrame,
    *,
    cells_of: Callable[[Any], Sequence[str]],
    replicate_of: Callable[[Any], int],
    draw: Callable[[Any], Any] | None,
    witness: Witness | None,
    n_jobs: int,
) -> pd.DataFrame:
    """The witness value of every key of each cell that a record names (D3'f).

    The draw is a function of the key alone, so a lost key's sample is drawn again, without a
    fit, beside the samples of the keys that survived.  A loop whose sample is drawn inside its
    fit has no draw to read, and every value is ``nan`` with :data:`DRAW_NOT_SEPARABLE`.
    """
    lost = {
        name
        for scenarios in failures["scenarios"].astype(str)
        for name in scenarios.split(";")
        if name
    }
    affected = [key for key in keys if lost.intersection(cells_of(key))]
    if witness is None or draw is None:
        name = "no witness declared" if witness is None else DRAW_NOT_SEPARABLE
        values = [float("nan")] * len(affected)
    else:
        name = witness.name
        # Wrapped, because ``map_parallel`` splats a tuple payload and a key may be a tuple.
        values = map_parallel(
            _WitnessOf(draw, witness.statistic), [(key,) for key in affected], n_jobs=n_jobs
        )
    out = [
        (*cell.split("/"), int(replicate_of(key)), name, float(value))
        for key, value in zip(affected, values, strict=True)
        for cell in cells_of(key)
        if cell in lost
    ]
    return pd.DataFrame(out, columns=list(WITNESS_COLUMNS))
