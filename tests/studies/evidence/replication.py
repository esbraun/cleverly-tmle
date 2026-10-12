"""Fit a study's replicates so that one failed fit costs one replicate.

RM39 part b.  Every Python replicate loop of a registered study goes through
:func:`fit_replicates`.  Each replicate runs in two steps.  ``draw`` builds the sample and its
truth; an error there is a defect of the study's law and stops the run.  ``fit_one`` fits it;
an error there is judged by :func:`cleverly.validation._failures.recordable`.  A recordable
error becomes a failure record with the fields of the reference phase's records
(:data:`tests.studies.evidence.failures.FAILURE_COLUMNS`), the draw is kept, and the loop goes
on.  Any other error stops the run.

Under the regeneration driver, each finished replicate is checkpointed beside the run's
scratch, which the driver keys by the RM39a1 resume key.  A rerun reuses the checkpoints, and a
worker killed mid-fit is fitted once more, alone.  With no failure, the outcome is exactly what
:func:`cleverly.utils.parallel.map_parallel` returns for the same worker.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import os
import pickle
from collections.abc import Callable, Hashable, Iterator, Sequence
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from joblib import Parallel, delayed

from cleverly.utils.parallel import map_parallel
from cleverly.validation._failures import failure_call, recordable

#: The class a recorded fitting failure carries, as in the reference phase's records.
FITTING_FAILURE = "fitting_failure"

#: How many leading replicates of a cell run first.  When every one of them fails, the cause
#: is constant, so the run stops rather than recording it thousands of times (RM39 Q2).
SYSTEMATIC_PREFIX = 20

#: Where the driver keeps per-replicate checkpoints for the current run, or ``None``.
CHECKPOINT_ROOT: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "cleverly_replication_checkpoints", default=None
)

_FAILURE_FIELDS = (
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


class SystematicFailure(RuntimeError):
    """Every leading replicate of one cell failed, so the run stops."""


class KilledReplicate(RuntimeError):
    """A replicate's worker died in a pool and again when it ran alone."""


@contextlib.contextmanager
def checkpoints(root: Path | None) -> Iterator[None]:
    """Make ``root`` the checkpoint root of every :func:`fit_replicates` call inside."""
    token = CHECKPOINT_ROOT.set(root)
    used = _LABELS.set(set())
    try:
        yield
    finally:
        _LABELS.reset(used)
        CHECKPOINT_ROOT.reset(token)


_LABELS: contextvars.ContextVar[set[tuple[str, str]] | None] = contextvars.ContextVar(
    "cleverly_replication_labels", default=None
)


@dataclass
class Outcomes:
    """What :func:`fit_replicates` returns, in key order.

    Parameters
    ----------
    keys : list
        The keys, in the order given.
    drawn : list
        One draw per key, ``None`` when the call had no ``draw``.
    fitted : list
        One fit value per key, ``None`` where the fit failed.
    failed : list of int
        The positions of the keys whose fit failed.
    failures : pandas.DataFrame
        One record per failed key, in key order, with the columns of
        ``failures.FAILURE_COLUMNS``.
    """

    keys: list[Hashable]
    drawn: list[Any]
    fitted: list[Any]
    failed: list[int] = field(default_factory=list)
    failures: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=list(_FAILURE_FIELDS))
    )

    @property
    def values(self) -> list[Any]:
        """The fit values of the keys that succeeded, in key order."""
        failed = set(self.failed)
        return [value for index, value in enumerate(self.fitted) if index not in failed]


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    if hasattr(value, "item") and not isinstance(value, type):
        return _plain(value.item())
    raise TypeError(f"a replicate key holds a {type(value).__name__}: {value!r}")


def key_id(key: Hashable) -> str:
    """A file-safe identifier of ``key``, equal in every process."""
    text = json.dumps(_plain(key), separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


@dataclass(frozen=True)
class _Task:
    key: Hashable
    replicate: int
    cell: str
    seed: int | None
    attempt: int


@dataclass(frozen=True)
class _Plan:
    fit_one: Callable[[Any, Any], Any]
    draw: Callable[[Any], Any] | None
    label: str
    implementation: str
    directory: Path | None


def _message(error: BaseException) -> str:
    return str(error).replace("\r", " ").replace("\n", " ")


def _work(plan: _Plan, task: _Task) -> tuple[Any, Any, dict[str, Any] | None]:
    """Draw and fit one replicate; checkpoint it when the run keeps checkpoints."""
    store = None if plan.directory is None else plan.directory / f"{key_id(task.key)}.pkl"
    if store is not None and store.exists():
        with store.open("rb") as handle:
            return pickle.load(handle)
    marker = None if plan.directory is None else plan.directory / f"{key_id(task.key)}.started"
    if marker is not None:
        marker.write_text(f"{os.getpid()}\n", encoding="utf-8")
    drawn = None if plan.draw is None else plan.draw(task.key)
    failure: dict[str, Any] | None = None
    fitted: Any = None
    try:
        fitted = plan.fit_one(task.key, drawn)
    except Exception as error:
        if not recordable(error):
            raise
        failure = {
            "implementation": plan.implementation,
            "group_id": f"r{task.replicate:09d}",
            "scenarios": task.cell,
            "replicates": str(task.replicate),
            "attempt": task.attempt,
            "class": FITTING_FAILURE,
            "stage": plan.label,
            "call": failure_call(error),
            "message": f"{type(error).__name__}: {_message(error)}",
            "seed": "NA" if task.seed is None else str(task.seed),
        }
    outcome = (drawn, fitted, failure)
    if store is not None and marker is not None:
        temporary = store.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(outcome, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, store)
        marker.unlink(missing_ok=True)
    return outcome


def _alone(plan: _Plan, task: _Task) -> tuple[Any, Any, dict[str, Any] | None]:
    """Run one replicate in a worker process, never in the parent, with nothing beside it.

    joblib rather than :class:`concurrent.futures.ProcessPoolExecutor`, because its pickler
    carries the closures some studies pass as ``fit_one``.  Two workers with one task keep it
    out of the parent process, which ``n_jobs=1`` would not.
    """
    try:
        return Parallel(n_jobs=2)([delayed(_work)(plan, task)])[0]
    except BrokenProcessPool as error:
        raise KilledReplicate(
            f"replicate {task.key!r} of {plan.label} was killed in a pool and again "
            "when it ran alone"
        ) from error


def _started(plan: _Plan, tasks: Sequence[_Task]) -> list[_Task]:
    """The tasks whose worker started and never finished: a kill, or a pool aborted by one."""
    if plan.directory is None:
        return []
    return [
        task
        for task in tasks
        if (plan.directory / f"{key_id(task.key)}.started").exists()
        and not (plan.directory / f"{key_id(task.key)}.pkl").exists()
    ]


def _run(
    plan: _Plan, tasks: Sequence[_Task], n_jobs: int
) -> dict[Hashable, tuple[Any, Any, dict[str, Any] | None]]:
    """Every task's outcome, surviving killed workers when the run keeps checkpoints.

    A pool that loses a worker aborts every task in flight, so each task that started and did
    not finish runs once more alone.  Only a kill of a task running alone stops the run: that
    one is the task's own.  The rest resume at half the workers.
    """
    done: dict[Hashable, tuple[Any, Any, dict[str, Any] | None]] = {}
    for task in _started(plan, tasks):
        done[task.key] = _alone(plan, task)
    jobs = n_jobs
    while True:
        remaining = [task for task in tasks if task.key not in done]
        if not remaining:
            return done
        try:
            outcomes = map_parallel(_work, [(plan, task) for task in remaining], n_jobs=jobs)
        except BrokenProcessPool:
            if plan.directory is None:
                raise
            for task in _started(plan, remaining):
                done[task.key] = _alone(plan, task)
            jobs = max(1, jobs // 2)
            continue
        done.update(zip((task.key for task in remaining), outcomes, strict=True))
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
    seed_of: Callable[[Any], int | None] | None = None,
    checkpoint: Path | Literal["ambient"] | None = "ambient",
) -> Outcomes:
    """Draw and fit every key, recording each recordable fitting failure.

    The first :data:`SYSTEMATIC_PREFIX` keys of every cell run first.  When each of them failed
    in one cell, the run stops with :class:`SystematicFailure`.  Each replicate depends on its
    own seed only, so the order in which keys run does not change any outcome.
    """
    if not keys:
        raise ValueError(f"{label}: no replicates to fit")
    if len({key_id(key) for key in keys}) != len(keys):
        raise ValueError(f"{label}: two replicate keys are equal")
    root = CHECKPOINT_ROOT.get() if checkpoint == "ambient" else checkpoint
    directory = None if root is None else Path(root) / implementation / label
    used = _LABELS.get()
    if directory is not None and used is not None:
        if (implementation, label) in used:
            raise ValueError(f"{implementation}/{label}: the label is used twice in one run")
        used.add((implementation, label))
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
    plan = _Plan(fit_one, draw, label, implementation, directory)
    tasks = [
        _Task(
            key=key,
            replicate=int(replicate_of(key)),
            cell=str(cell_of(key)),
            seed=None if seed_of is None else seed_of(key),
            attempt=1,
        )
        for key in keys
    ]
    by_cell: dict[str, list[_Task]] = {}
    for task in tasks:
        by_cell.setdefault(task.cell, []).append(task)
    prefix = [task for cell in by_cell.values() for task in cell[:SYSTEMATIC_PREFIX]]
    done = _run(plan, prefix, n_jobs)
    for cell, members in by_cell.items():
        head = members[:SYSTEMATIC_PREFIX]
        if len(head) == SYSTEMATIC_PREFIX and all(done[task.key][2] for task in head):
            first = done[head[0].key][2]
            assert first is not None
            raise SystematicFailure(
                f"{label}: the first {SYSTEMATIC_PREFIX} replicates of {cell} failed; "
                f"first: {first['message']}"
            )
    rest = [task for task in tasks if task.key not in done]
    if rest:
        done.update(_run(plan, rest, n_jobs))
    ordered = [done[key] for key in keys]
    records = [(index, outcome[2]) for index, outcome in enumerate(ordered) if outcome[2]]
    if len(records) == len(ordered):
        raise RuntimeError(f"every replicate of {label} failed; first: {records[0][1]['message']}")
    return Outcomes(
        keys=list(keys),
        drawn=[outcome[0] for outcome in ordered],
        fitted=[outcome[1] for outcome in ordered],
        failed=[index for index, _ in records],
        failures=pd.DataFrame([record for _, record in records], columns=list(_FAILURE_FIELDS)),
    )
