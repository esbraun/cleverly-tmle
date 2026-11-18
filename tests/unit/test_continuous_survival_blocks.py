"""The block walk and the event-support grid of the continuous-time engine (X106 part C2).

The engine walks blocks of units over the times where some hazard can jump.  Its contract is
the dense walk's results: bit for bit for risks with the shipped learners, within a tolerance
for restricted means and user learners, and bit for bit across every scheduling choice (blocks
kept, ladder length, memory budget).  The dense walk is transcribed in
``tests/unit/_continuous_dense_reference.py``, which reads the same nuisance fits; the golden
numbers in ``tests/unit/_continuous_dense_golden.py`` came from the engine before blocks.
Every fixture asserts the branch it exists to reach.
"""

from __future__ import annotations

import contextlib
import itertools
import threading
import time
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pytest
from sklearn.base import BaseEstimator

from cleverly.exceptions import DataError, MethodConfigurationError
from cleverly.survival import CoxHazard, _chunked, _memory, engine
from cleverly.survival import _targeting as T
from cleverly.survival.diagnostics import continuous_truncation_curve

from . import _continuous_dense_reference as dense
from ._continuous_blocks_support import (
    ALL,
    FIXTURES,
    TOLERANT,
    estimator,
    law,
    summary,
    walk_floor,
)
from ._continuous_dense_golden import GOLDEN

#: Rows per block where a test forces several blocks: four to five blocks at these sizes.
ROWS = 150
#: Contract section 3: relative tolerance and the floor on the absolute one, times the scale.
RTOL, FLOOR = 1e-12, 1e-14


@dataclass
class Fit:
    """A fit with what the tests read from inside it."""

    result: Any
    initial: Any
    floor: float
    run: Any
    vectors: tuple[np.ndarray, ...] | None
    overflowed: int


@contextmanager
def _patched(
    *,
    rows: int | None = None,
    resident: float | None = None,
    ladder: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Capture a fit's internals, with its block size, residency and ladder forced."""
    seen: dict[str, Any] = {"overflowed": 0}
    with pytest.MonkeyPatch.context() as patch:
        target = engine.ContinuousSurvivalTMLE._target

        def capture_target(self: Any, initial: Any, floor: float, *, settings: Any) -> Any:
            seen["initial"], seen["floor"] = initial, floor
            return target(self, initial, floor, settings=settings)

        run_walk = _chunked.run_walk

        def capture_run(plan: Any, **kwargs: Any) -> Any:
            if resident is not None:
                kwargs["resident"] = round(resident * len(plan.bounds))
            if ladder is not None:
                kwargs["ladder"] = ladder
            seen["run"] = run_walk(plan, **kwargs)
            return seen["run"]

        share = engine._OffGrid.share

        def capture_share(self: Any, cutoff: float) -> float:
            value = share(self, cutoff)
            seen["vectors"] = getattr(self, "vectors", None)
            return value

        overflow = engine._FitPlan.overflow

        def count_overflow(self: Any, *args: Any) -> Any:
            out = overflow(self, *args)
            if out is not None:
                seen["overflowed"] += int(out.sum())
            return out

        patch.setattr(engine.ContinuousSurvivalTMLE, "_target", capture_target)
        patch.setattr(_chunked, "run_walk", capture_run)
        patch.setattr(engine._OffGrid, "share", capture_share)
        patch.setattr(engine._FitPlan, "overflow", count_overflow)
        if rows is not None:
            patch.setattr(engine, "chunk_size", lambda n, J, M: min(rows, n))
        yield seen


def _fit(
    name: str,
    *,
    rows: int | None = None,
    resident: float | None = None,
    ladder: int | None = None,
    data: Any = None,
    **extra: Any,
) -> Fit:
    n, causes, _, floor = ALL[name]
    data = law(n, causes) if data is None else data
    with (
        _patched(rows=rows, resident=resident, ladder=ladder) as seen,
        walk_floor(floor),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        result = estimator(name, **extra).fit_data(data)
    return Fit(
        result,
        seen["initial"],
        seen["floor"],
        seen["run"],
        seen.get("vectors"),
        seen["overflowed"],
    )


def _reference(fit: Fit, name: str, adjust_weight: Any = None) -> dense.Reference:
    """The dense walk from the fit's own initial fit, with the fixture's walk floor."""
    est = fit.result.estimator
    with walk_floor(ALL[name][3]), warnings.catch_warnings():
        # The dense tilt overflows where a test forces it to; the engine's fit is silenced too.
        warnings.simplefilter("ignore")
        return dense.dense_target(
            fit.initial,
            fit.floor,
            cutoff=fit.result.record.cutoff,
            norm=est.norm,
            step=est.step_size,
            max_steps=est.max_iter,
            tilt=est.tilt,
            adjust_weight=adjust_weight,
        )


def _walk_fields(result: Any) -> dict[str, Any]:
    with np.errstate(divide="ignore", invalid="ignore"):
        standardized = result.standardized_score
    return {
        "standardized_score": standardized,
        "psi": result.evaluation.psi,
        "score": result.evaluation.score,
        "influence": result.evaluation.influence,
        "initial psi": result.initial.psi,
        "initial score": result.initial.score,
        "initial influence": result.initial.influence,
        "steps": result.steps,
        "halvings": result.halvings,
        "converged": result.converged,
        "final step": result.final_step,
        "frozen share": result.frozen_share,
        "history": np.asarray(result.history),
    }


def _record_fields(fit: Fit) -> dict[str, Any]:
    record = fit.result.record
    out: dict[str, Any] = {
        name: getattr(record, name)
        for name in (
            "steps",
            "halvings",
            "converged",
            "standardized_score",
            "initial_psi",
            "bounded_share",
            "last_time_floor_share",
            "frozen_share",
            "off_grid_share",
            "grid_size",
        )
    }
    for name, estimate in fit.result.estimates.items():
        out[name] = (estimate.psi, estimate.std_error)
    out["influence"] = fit.run.result.evaluation.influence
    if fit.result.simultaneous is not None:
        out["bands"] = fit.result.simultaneous.bands
    return out


def _bitwise(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    """The names whose values differ in any bit (NaN equal to NaN)."""
    assert left.keys() == right.keys()
    differ = []
    for name in left:
        a, b = left[name], right[name]
        if isinstance(a, tuple) and a and isinstance(a[0], (float, np.floating, np.ndarray)):
            same = all(
                np.array_equal(np.asarray(x), np.asarray(y), equal_nan=True)
                for x, y in zip(a, b, strict=True)
            )
        elif isinstance(a, (np.ndarray, float, np.floating)):
            same = np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
        else:
            same = a == b
        if not same:
            differ.append(name)
    return differ


#: Scores are means of influence-curve terms that nearly cancel, so their scale is the terms'
#: (plan 2.1's floor); a standardized score's is the terms' over the smallest spread.
SCORES = {"score": 1, "initial score": 1, "standardized_score": 2, "history": 2}


def _close(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    """The names outside contract section 3's tolerance; discrete values must be equal."""
    influence = np.asarray(right["influence"], dtype=float)
    finite = influence[np.isfinite(influence)]
    terms = float(np.max(np.abs(finite))) if finite.size else 0.0
    with np.errstate(invalid="ignore"):
        spreads = np.sqrt(np.mean(influence**2, axis=0))
    spreads = spreads[np.isfinite(spreads) & (spreads > 0)]
    spread = float(spreads.min()) if spreads.size else 1.0
    differ = []
    for name in left:
        a, b = left[name], right[name]
        if isinstance(a, (bool, int, np.bool_)) and not isinstance(a, float):
            if a != b:
                differ.append(name)
            continue
        if isinstance(a, tuple) and a and not isinstance(a[0], (float, np.floating, np.ndarray)):
            if a != b:
                differ.append(name)
            continue
        if name in SCORES:
            pairs = [(a, b)]
        else:
            pairs = list(zip(a, b, strict=True)) if isinstance(a, tuple) else [(a, b)]
        for x, y in pairs:
            x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
            scale = float(np.max(np.abs(y[np.isfinite(y)]))) if np.isfinite(y).any() else 0.0
            if name in SCORES:
                scale = terms if SCORES[name] == 1 else terms / spread
            if x.shape != y.shape or not np.allclose(
                x, y, rtol=RTOL, atol=FLOOR * scale, equal_nan=True
            ):
                differ.append(name)
                break
    return differ


def _reduced(fit: Fit) -> bool:
    return fit.initial.kept.size < fit.initial.grid.size


# ------------------------------------------------------------------- dense reference


@pytest.mark.parametrize("regime", ["canonical", "blocks, none kept"])
@pytest.mark.parametrize("name", list(FIXTURES))
def test_the_block_walk_equals_the_dense_walk_bit_for_bit(name: str, regime: str) -> None:
    forced = {} if regime == "canonical" else {"rows": ROWS, "resident": 0.0}
    fit = _fit(name, **forced)
    record = fit.result.record
    assert record.steps >= 2 and record.halvings >= 1, "the fixture must accept and halve"
    assert _reduced(fit), "the fixture must drop some grid times"
    if name == "frozen pair":
        assert record.frozen_share > 0
    if forced:
        assert len(engine.blocks(fit.initial.n, fit.initial.block)) >= 4
        assert fit.run.replayed > 0
    reference = _reference(fit, name)
    assert _bitwise(_walk_fields(fit.run.result), _walk_fields(reference.result)) == []
    assert record.bounded_share == reference.bounded_share
    assert record.last_time_floor_share == reference.last_time_floor_share
    assert record.off_grid_share == reference.off_grid.share
    # R1: the off-grid score, plug-in and spread vectors, not only the share they give.
    assert fit.vectors is not None
    for ours, theirs in zip(
        fit.vectors,
        (reference.off_grid.score, reference.off_grid.psi, reference.off_grid.sigma),
        strict=True,
    ):
        assert np.array_equal(ours, theirs)
    # The dense engine's own numbers, anchored to what it printed before blocks.
    golden = GOLDEN[name]
    assert summary(fit.result)["estimates"].keys() == golden["estimates"].keys()
    for key, (psi, se) in summary(fit.result)["estimates"].items():
        np.testing.assert_allclose((psi, se), golden["estimates"][key], rtol=1e-10)
    for key in ("steps", "halvings", "frozen_share", "off_grid_share"):
        assert summary(fit.result)[key] == golden[key]


def test_the_off_grid_sums_carry_block_totals_in_row_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation control (R1): adding each block's own sum breaks the bitwise vectors."""
    name = "two causes"
    fit = _fit(name, rows=ROWS, resident=0.0)
    reference = _reference(fit, name)
    assert np.array_equal(fit.vectors[0], reference.off_grid.score)

    def per_block(total: Any, rows: Any) -> Any:
        return rows.sum(axis=0) if total is None else total + rows.sum(axis=0)

    monkeypatch.setattr(engine, "_carried", per_block)
    mutated = _fit(name, rows=ROWS, resident=0.0)
    assert not all(
        np.array_equal(ours, theirs)
        for ours, theirs in zip(
            mutated.vectors[:2], (reference.off_grid.score, reference.off_grid.psi), strict=True
        )
    )


class Wrapped(BaseEstimator):  # type: ignore[misc]
    """A user hazard learner that wraps a Cox model by composition, with a call counter.

    ``mode`` changes its predictions: ``"smooth"`` adds a small linear cumulative hazard, so
    every time jumps; ``"negative"`` puts a negative jump at ``at``; ``"nan"`` makes the
    cumulative hazard NaN at ``at`` for rows with ``w0 > 2``; ``"noise"`` adds call-dependent
    noise; ``"refuse"`` gives rows with ``w0 > 1.8`` a jump of one at ``at`` and rows with
    ``w0 < -1.8`` a survival near zero.
    """

    calls = 0

    def __init__(self, mode: str = "plain", at: float = 0.0) -> None:
        self.mode = mode
        self.at = at

    def fit(self, X: Any, time: Any, event: Any) -> Wrapped:
        self.model_ = CoxHazard("main").fit(X, time, event)
        self.noise_ = 0
        return self

    def cumulative_hazard(self, X: Any, times: Any) -> np.ndarray:
        type(self).calls += 1
        X, times = np.asarray(X, dtype=float), np.asarray(times, dtype=float)
        out = np.array(self.model_.cumulative_hazard(X, times), dtype=float)
        at = times == self.at
        if self.mode == "smooth":
            out += 1e-4 * times[None, :]
        elif self.mode == "flat":
            # No jump at ``at``: the cumulative hazard there repeats the time before.
            position = int(np.flatnonzero(at)[0]) if at.any() else 0
            if position:
                out[:, position] = out[:, position - 1]
        elif self.mode == "negative":
            out[:, at] -= 1e-3
        elif self.mode == "nan":
            out[np.ix_(X[:, 1] > 2.0, at)] = np.nan
        elif self.mode == "noise":
            self.noise_ += 1
            out *= 1.0 + 1e-9 * self.noise_
        elif self.mode == "refuse":
            later = times >= self.at
            out[np.ix_(X[:, 1] > 1.8, later)] += 1.0
            out[X[:, 1] < -1.8] = 30.0 * np.arange(times.size)[None, :]
        return out


def _times(name: str) -> np.ndarray:
    n, causes, _, _ = ALL[name]
    data = law(n, causes)
    return np.unique(np.asarray(data.time))


def _event_free_time(name: str) -> float:
    """A censoring time before the first target, where no cause has an event."""
    n, causes, _, _ = ALL[name]
    data = law(n, causes)
    t, d = np.asarray(data.time), np.asarray(data.event)
    censored = np.setdiff1d(t[(d == 0) & (t < 0.9)], t[d > 0])
    return float(censored[len(censored) // 2])


@pytest.mark.parametrize("name", ["restricted mean", "user learner"])
def test_restricted_means_and_user_learners_stay_within_the_tolerance(name: str) -> None:
    Wrapped.calls = 0
    base = "restricted mean" if name == "restricted mean" else "two causes"
    extra: dict[str, Any] = {}
    if name == "user learner":
        extra = {"hazard_library": [Wrapped()], "censoring_library": [Wrapped()]}
    fit = _fit(base, rows=ROWS, resident=0.0, **extra)
    record = fit.result.record
    assert record.steps >= 2 and record.halvings >= 1
    assert _reduced(fit)
    if name == "user learner":
        assert Wrapped.calls > 0
    reference = _reference(fit, base)
    assert _close(_walk_fields(fit.run.result), _walk_fields(reference.result)) == []
    assert record.off_grid_share == reference.off_grid.share
    if name == "restricted mean":
        golden = GOLDEN[base]
        for key, (psi, se) in summary(fit.result)["estimates"].items():
            np.testing.assert_allclose((psi, se), golden["estimates"][key], rtol=1e-10)


def test_a_restricted_direction_on_the_kept_grid_equals_the_dense_one() -> None:
    fit = _fit("restricted mean")
    initial = fit.initial
    assert _reduced(fit)
    plan = engine._FitPlan(initial, fit.floor)
    assert plan.expand is not None
    problem = plan.problem(0)
    jumps = plan.initial(0)
    survival, incidence = T.curves(jumps)
    v = np.random.default_rng(0).normal(size=plan.n_targets)
    ours = T._direction(problem, _chunked._light(survival, incidence), v)

    lo, hi = plan.bounds[0]
    full_jumps = initial.hazards.jumps(lo, hi)
    full = T.Problem(
        initial.A[lo:hi],
        initial.T[lo:hi, None] >= initial.grid[None, :],
        np.zeros((len(initial.causes), hi - lo, initial.grid.size), dtype=bool),
        plan._weight(0),
        initial.target_weights,
    )
    theirs = dense._direction(full, dense.evaluate(full, full_jumps), v)
    assert np.array_equal(ours, np.take(theirs, initial.kept, axis=-1))


# ------------------------------------------------------------------- scheduling


def _schedules() -> list[dict[str, Any]]:
    return [
        {"rows": ROWS, "resident": 1.0},
        {"rows": ROWS, "resident": 0.0},
        {"rows": ROWS, "resident": 0.5},
        {"rows": ROWS, "resident": 0.0, "ladder": 1},
        {"rows": ROWS, "resident": 0.0, "ladder": 2},
        {"rows": ROWS, "resident": 0.5, "ladder": 4},
    ]


@pytest.mark.parametrize("name", ["two causes", "restricted mean", "user learner"])
def test_no_schedule_changes_a_bit(name: str) -> None:
    base = "two causes" if name == "user learner" else name
    extra: dict[str, Any] = {}
    if name == "user learner":
        extra = {"hazard_library": [Wrapped()], "censoring_library": [Wrapped()]}
    fits = [_fit(base, **schedule, **extra) for schedule in _schedules()]
    first = _record_fields(fits[0])
    for fit in fits[1:]:
        assert _bitwise(_record_fields(fit), first) == []
    assert any(fit.run.replayed > 0 for fit in fits)


def test_the_memory_budget_changes_no_bit() -> None:
    name = "two causes"
    data = law(*FIXTURES[name][:2])
    with pytest.raises(_memory.ContinuousSurvivalMemoryError) as caught:
        estimator(name, max_memory_bytes=1).fit_data(data)
    smallest = caught.value.predicted
    assert smallest is not None
    tight = _fit(name, max_memory_bytes=smallest)
    loose = _fit(name, max_memory_bytes=2**62)
    assert tight.run.replayed > 0 or len(engine.blocks(tight.initial.n, tight.initial.block)) == 1
    assert _bitwise(_record_fields(tight), _record_fields(loose)) == []
    assert tight.result.to_frame().equals(loose.result.to_frame())


@pytest.mark.parametrize("rows", [1, 3, 7])
def test_the_block_size_changes_no_bit(rows: int) -> None:
    whole = _fit("small, two causes")
    blocked = _fit("small, two causes", rows=rows, resident=0.0)
    assert whole.result.record.steps >= 2
    assert _bitwise(_record_fields(blocked), _record_fields(whole)) == []


@pytest.mark.parametrize("rows", [1, 3, 7])
def test_the_block_size_keeps_a_restricted_mean_within_the_tolerance(rows: int) -> None:
    whole = _fit("small, restricted mean")
    blocked = _fit("small, restricted mean", rows=rows, resident=0.0)
    assert whole.result.record.steps >= 2
    assert _close(_record_fields(blocked), _record_fields(whole)) == []


# ------------------------------------------------------------------- threads (C2c)


class Exclusive(Wrapped):
    """A user learner that records whether two threads were ever inside it at once."""

    overlapped = False
    _inside = 0
    _guard = threading.Lock()

    def cumulative_hazard(self, X: Any, times: Any) -> np.ndarray:
        cls = type(self)
        with cls._guard:
            cls._inside += 1
            if cls._inside > 1:
                cls.overlapped = True
        try:
            # Long enough that a second thread arrives while the first is inside.
            time.sleep(0.002)
            return super().cumulative_hazard(X, times)
        finally:
            with cls._guard:
                cls._inside -= 1


@pytest.mark.parametrize("name", ["two causes", "three causes", "restricted mean", "accepted hold"])
@pytest.mark.parametrize("resident", [0.0, 0.5, 1.0])
def test_threads_change_no_bit(name: str, resident: float) -> None:
    """DR7, R2: n_jobs 1 and 2 (and 4) give the same record, estimates and off-grid vectors."""
    one = _fit(name, rows=37, resident=resident)
    assert len(engine.blocks(one.initial.n, one.initial.block)) >= 8
    for jobs in (2, 4):
        threaded = _fit(name, rows=37, resident=resident, n_jobs=jobs)
        assert _bitwise(_record_fields(threaded), _record_fields(one)) == []
        assert one.vectors is not None and threaded.vectors is not None
        for a, b in zip(one.vectors, threaded.vectors, strict=True):
            assert np.array_equal(a, b, equal_nan=True)
        assert threaded.run.replayed == one.run.replayed


def test_a_process_backend_around_the_fit_keeps_the_walk_on_threads() -> None:
    """The blocks write shared arrays: a loky backend set by the caller must not take them."""
    import joblib

    one = _fit("two causes", rows=37, resident=0.0, n_jobs=1)
    with joblib.parallel_config(backend="loky"):
        threaded = _fit("two causes", rows=37, resident=0.0, n_jobs=2)
    assert _bitwise(_record_fields(threaded), _record_fields(one)) == []


def test_threads_change_no_bit_with_two_blas_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    """DR15: the walk leaves BLAS settings alone, so threads match at any one setting."""
    threadpoolctl = pytest.importorskip("threadpoolctl")
    for_blocks = _chunked.for_blocks
    seen: list[Any] = []

    def blas() -> list[tuple[str, int]]:
        return sorted(
            (info["filepath"], info["num_threads"])
            for info in threadpoolctl.threadpool_info()
            if info["user_api"] == "blas"
        )

    def watched(count: int, work: Any, commit: Any, n_jobs: int = 1) -> None:
        def observed(k: int) -> Any:
            out = work(k)
            # Read after the block's work: a worker that changed the settings shows it here.
            seen.append(blas())
            return out

        for_blocks(count, observed, commit, n_jobs)

    monkeypatch.setattr(_chunked, "for_blocks", watched)
    with threadpoolctl.threadpool_limits(2, user_api="blas"):
        caller = blas()
        one = _fit("restricted mean", rows=37, resident=0.0)
        two = _fit("restricted mean", rows=37, resident=0.0, n_jobs=2)
    assert seen and all(settings == caller for settings in seen)
    assert _bitwise(_record_fields(two), _record_fields(one)) == []


@pytest.mark.parametrize("locked", [True, False])
def test_a_user_learner_is_never_entered_by_two_threads(
    locked: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A user learner is called under a lock; without it (mutation control) threads overlap."""
    if not locked:
        monkeypatch.setattr(_chunked, "_LEARNER_LOCK", contextlib.nullcontext())
    extra = {"hazard_library": [Exclusive()], "censoring_library": [Exclusive()]}
    one = _fit("two causes", rows=37, resident=0.0, **extra)
    Exclusive.overlapped = False
    two = _fit("two causes", rows=37, resident=0.0, n_jobs=2, **extra)
    assert Exclusive.overlapped is not locked
    if locked:
        assert _bitwise(_record_fields(two), _record_fields(one)) == []


def test_an_overflow_under_threads_signals_as_the_dense_walk_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R-2.2f with threads: the overflow's signals are raised in block order on the caller."""
    plain = _fit("two causes")
    _overflowing(monkeypatch, plain)
    one = _fit("two causes", rows=37, resident=0.0)
    assert one.overflowed > 0
    two = _fit("two causes", rows=37, resident=0.0, n_jobs=2)
    assert _bitwise(_record_fields(two), _record_fields(one)) == []
    data = law(*ALL["two causes"][:2])
    for jobs in (1, 2):
        with (
            walk_floor(ALL["two causes"][3]),
            _patched(rows=37, resident=0.0),
            warnings.catch_warnings(),
            np.errstate(over="raise"),
            pytest.raises(FloatingPointError),
        ):
            warnings.simplefilter("ignore")
            estimator("two causes", n_jobs=jobs).fit_data(data)


def test_a_block_error_under_threads_reaches_the_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first error a block raises is raised by the fit, after the running blocks finish."""
    trial = _chunked._Walker.trial

    def failing(self: Any, k: int, *args: Any) -> Any:
        if k == 3:
            raise RuntimeError("block three")
        return trial(self, k, *args)

    monkeypatch.setattr(_chunked._Walker, "trial", failing)
    with pytest.raises(RuntimeError, match="block three"):
        _fit("two causes", rows=37, resident=0.0, n_jobs=2)


def test_for_blocks_commits_in_block_order_with_a_bounded_lead() -> None:
    """R2: commits run in block order on the caller, at most 2 n_jobs blocks ahead."""
    started: list[int] = []
    committed: list[int] = []
    lead: list[int] = []
    caller = threading.get_ident()
    lock = threading.Lock()

    def work(k: int) -> int:
        with lock:
            started.append(k)
            lead.append(k - len(committed))
        time.sleep(0.001 * ((7 * k) % 5))
        return k * k

    def commit(k: int, value: int) -> None:
        assert threading.get_ident() == caller
        assert value == k * k
        committed.append(k)

    _chunked.for_blocks(40, work, commit, n_jobs=3)
    assert committed == list(range(40))
    assert sorted(started) == list(range(40))
    assert max(lead) < 2 * 3


@pytest.mark.parametrize("n_jobs", [1, 3])
def test_each_block_s_warnings_reach_the_caller_in_block_order(n_jobs: int) -> None:
    """Threads finish out of order; their package warnings are merged in block order."""
    from cleverly._warnings import recording
    from cleverly._warnings import warn as package_warn

    def work(k: int) -> int:
        time.sleep(0.002 * (11 - k))  # later blocks finish first
        package_warn(f"block {k}", UserWarning)
        return k

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with recording() as records:
            _chunked.for_blocks(12, work, lambda k, value: None, n_jobs=n_jobs)
    assert [record.message for record in records] == [f"block {k}" for k in range(12)]


@pytest.mark.parametrize("value", [0, -2, 1.5, True, "2"])
def test_n_jobs_is_a_positive_integer_or_minus_one(value: Any) -> None:
    with pytest.raises(ValueError, match="n_jobs"):
        engine.ContinuousSurvivalTMLE([1.0], n_jobs=value)


def test_n_jobs_minus_one_reads_the_available_cpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine, "available_cpus", lambda: 3)
    assert engine.ContinuousSurvivalTMLE([1.0], n_jobs=-1)._threads() == 3
    assert engine.ContinuousSurvivalTMLE([1.0], n_jobs=None)._threads() == 1


def test_the_facade_passes_runtime_n_jobs() -> None:
    from cleverly import Runtime, TMLEMethod
    from cleverly.survival.data import continuous_kwargs

    assert continuous_kwargs(TMLEMethod(runtime=Runtime(n_jobs=3)))["n_jobs"] == 3
    assert continuous_kwargs(TMLEMethod())["n_jobs"] == 1


# ------------------------------------------------------------------- the ladder


def _dense_ladder(
    script: list[dict[str, Any]],
    *,
    length: float,
    steps: int,
    halvings: int,
    step: float,
    initial_step: float,
    max_steps: int,
    halving_cap: int,
    min_step: float,
    n: int,
) -> tuple[int, int | None, int, int, float, np.ndarray]:
    """The dense loop's trials of one direction, each read from ``script`` in order."""
    frozen = np.zeros((2, n), dtype=bool)
    ever = np.zeros((2, n), dtype=bool)
    entry = 0
    while steps < max_steps and halvings < halving_cap and step >= min_step:
        if entry == len(script):
            break
        trial = script[entry]
        held = frozen.any()
        low = trial["retry low"] if held else trial["low"]
        if low.any():
            new = low & ~frozen
            if new.any():
                frozen |= new
                ever |= new
                continue
            step /= 2
            halvings += 1
            frozen[:] = False
            entry += 1
            continue
        trial_length = trial["retry length"] if held else trial["length"]
        if not np.isfinite(trial_length) or trial_length >= length:
            step /= 2
            halvings += 1
            frozen[:] = False
            entry += 1
            continue
        return entry + 1, entry, steps + 1, halvings, min(2.0 * step, initial_step), ever
    return entry, None, steps, halvings, step, ever


def _scripts(count: int, seed: int) -> Iterator[tuple[list[dict[str, Any]], dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    n = 5
    for _ in range(count):
        script = []
        for _ in range(int(rng.integers(1, 5))):
            low = rng.random((2, n)) < rng.choice([0.0, 0.2])
            # A pair's trial reads only its own increment, so holding the low pairs leaves the
            # others as they were; a held pair can still read low by an ulp.
            retry = low & (rng.random((2, n)) < rng.choice([0.0, 0.0, 0.5]))
            lengths = rng.choice(
                [0.5, 1.0, 1.5, np.inf, np.nan], size=2, p=[0.4, 0.2, 0.2, 0.1, 0.1]
            )
            script.append(
                {"low": low, "retry low": retry, "length": lengths[0], "retry length": lengths[1]}
            )
        state = {
            "length": float(rng.choice([0.0, 1.0])),
            "steps": int(rng.integers(0, 4)),
            "halvings": int(rng.integers(0, 6)),
            "step": float(rng.choice([0.1, 4e-8, 1e-8, 2e-8])),
            "initial_step": 0.1,
            "max_steps": int(rng.integers(1, 5)),
            "halving_cap": int(rng.integers(1, 8)),
            "min_step": 1e-8,
        }
        yield script, state


def _outcomes(script: list[dict[str, Any]]) -> list[_chunked.Outcome]:
    out = []
    for trial in script:
        if trial["low"].any():
            still = bool(trial["retry low"].any())
            out.append(_chunked.Outcome(trial["low"], still, float(trial["retry length"])))
        else:
            out.append(_chunked.Outcome(None, False, float(trial["length"])))
    return out


def _disagreements(decide: Any) -> int:
    count = 0
    for script, state in _scripts(4000, 11):
        expected = _dense_ladder(script, n=5, **state)
        decision = decide(_outcomes(script), **state)
        ever = np.zeros((2, 5), dtype=bool)
        for entry in decision.marked:
            ever |= script[entry]["low"]
        got = (
            decision.consumed,
            decision.accepted,
            decision.steps,
            decision.halvings,
            decision.step,
        )
        if got != expected[:5] or not np.array_equal(ever, expected[5]):
            count += 1
    return count


def test_the_ladder_takes_its_trials_as_the_dense_loop_does() -> None:
    assert _disagreements(_chunked.ladder_decision) == 0


def test_the_ladder_witness_sees_a_hold_from_an_unconsumed_trial() -> None:
    """Mutation control: marking every low trial, consumed or not, is caught."""

    def every_low(outcomes: Any, **state: Any) -> _chunked.Decision:
        decision = _chunked.ladder_decision(outcomes, **state)
        marked = tuple(e for e, o in enumerate(outcomes) if o.low is not None)
        return replace(decision, marked=marked)

    assert _disagreements(every_low) > 0


# ------------------------------------------------------------------- the event-support grid


def test_the_grid_keeps_every_event_time_and_every_jump() -> None:
    fit = _fit("one cause, cross-fitted")
    initial = fit.initial
    kept = np.zeros(initial.grid.size, dtype=bool)
    kept[initial.kept] = True
    events = np.isin(initial.grid, initial.T[initial.D > 0])
    jumping = (initial.hazards.jumps(0, initial.n) != 0).any(axis=(0, 1, 2))
    assert _reduced(fit)
    assert kept[events].all() and kept[jumping].all() and kept[initial.target_index].all()
    assert np.array_equal(
        kept, events | jumping | np.isin(np.arange(kept.size), initial.target_index)
    )


def test_a_smooth_learner_keeps_every_time_and_stays_bitwise() -> None:
    extra = {"hazard_library": [Wrapped("smooth")], "censoring_library": [Wrapped()]}
    fit = _fit("two causes", rows=ROWS, resident=0.0, **extra)
    assert not _reduced(fit)
    reference = _reference(fit, "two causes")
    assert _bitwise(_walk_fields(fit.run.result), _walk_fields(reference.result)) == []


def test_a_negative_jump_keeps_its_event_free_time() -> None:
    at = _event_free_time("two causes")
    extra = {"hazard_library": [Wrapped("negative", at)], "censoring_library": [Wrapped()]}
    fit = _fit("two causes", **extra)
    initial = fit.initial
    position = int(np.searchsorted(initial.grid, at))
    assert initial.grid[position] == at and not np.isin(at, initial.T[initial.D > 0])
    assert (initial.hazards.jumps(0, initial.n)[..., position] < 0).any()
    assert position in initial.kept
    reference = _reference(fit, "two causes")
    assert _close(_walk_fields(fit.run.result), _walk_fields(reference.result)) == []


def _event_time(name: str) -> float:
    """An event time before the first target, with a time before it on the grid."""
    n, causes, _, _ = ALL[name]
    data = law(n, causes)
    t, d = np.asarray(data.time), np.asarray(data.event)
    events = np.unique(t[(d > 0) & (t < 0.9)])
    return float(events[len(events) // 2])


@pytest.mark.parametrize("mutated", [False, True])
def test_an_event_time_with_no_jump_stays_on_the_grid(
    mutated: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The grid keeps an event time even where no hazard jumps; dropping it is caught."""
    name = "two causes"
    at = _event_time(name)
    if mutated:
        original = engine.walk_grid

        def without_events(grid: Any, jumping: Any, observed: Any, target_index: Any) -> Any:
            return original(grid, jumping, observed[:0], target_index)

        monkeypatch.setattr(engine, "walk_grid", without_events)
    extra = {"hazard_library": [Wrapped("flat", at)], "censoring_library": [Wrapped()]}
    fit = _fit(name, **extra)
    initial = fit.initial
    position = int(np.searchsorted(initial.grid, at))
    assert initial.grid[position] == at
    assert not (initial.hazards.jumps(0, initial.n)[..., position] != 0).any()
    assert (position in initial.kept) is not mutated
    reference = _reference(fit, name)
    differ = _close(_walk_fields(fit.run.result), _walk_fields(reference.result))
    assert (differ != []) is mutated


def _without(drop: Any) -> Any:
    original = engine.walk_grid

    def walk_grid(grid: Any, jumping: Any, observed: Any, target_index: Any) -> Any:
        kept = original(grid, jumping, observed, target_index)
        return np.setdiff1d(kept, drop(grid, jumping, observed, target_index, kept))

    return walk_grid


def _unique_event_in_one_fold(
    grid: Any, jumping: Any, observed: Any, target: Any, kept: Any
) -> Any:
    times, counts = np.unique(observed, return_counts=True)
    return np.searchsorted(grid, times[counts == 1][:1])


def _jump_without_event(grid: Any, jumping: Any, observed: Any, target: Any, kept: Any) -> Any:
    free = np.flatnonzero(jumping & ~np.isin(grid, observed))
    free = np.setdiff1d(free, target)
    return free[: max(1, free.size // 2)]


def _target_column(grid: Any, jumping: Any, observed: Any, target: Any, kept: Any) -> Any:
    return np.asarray(target)


@pytest.mark.parametrize("mutation", ["unique event", "jump without an event", "target time"])
def test_dropping_a_needed_time_breaks_the_bitwise_witness(
    mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation controls (DR17): each needed kind of time is needed."""
    name, extra = "one cause, cross-fitted", {}
    if mutation == "jump without an event":
        # A learner whose hazard jumps at every time, events or not.
        name = "two causes"
        extra = {"hazard_library": [Wrapped("smooth")], "censoring_library": [Wrapped()]}
    drop = {
        "unique event": _unique_event_in_one_fold,
        "jump without an event": _jump_without_event,
        "target time": _target_column,
    }[mutation]
    monkeypatch.setattr(engine, "walk_grid", _without(drop))
    fit = _fit(name, **extra)
    assert _reduced(fit)
    reference = _reference(fit, name)
    assert _bitwise(_walk_fields(fit.run.result), _walk_fields(reference.result)) != []


# ------------------------------------------------------------------- overflow and refusals


#: A weight that makes the step overflow ``exp`` at the dropped time for the first steps tried.
HEAVY = 1e5


def _overflowing(monkeypatch: pytest.MonkeyPatch, fit: Fit) -> tuple[int, np.ndarray]:
    """Patch a huge weight at one dropped time for a few units, in the engine and the oracle."""
    initial = fit.initial
    dropped = np.setdiff1d(np.arange(initial.grid.size), initial.kept)
    column = int(dropped[(initial.grid[dropped] < 0.9).sum() // 2])
    units = np.flatnonzero(initial.grid[column] < initial.T)[:4]
    inverse = engine._FitPlan._inverse

    def heavy(self: Any, lo: int, hi: int, product: Any) -> Any:
        out = inverse(self, lo, hi, product).copy()
        inside = units[(units >= lo) & (units < hi)] - lo
        out[:, inside, column] = HEAVY
        return out

    monkeypatch.setattr(engine._FitPlan, "_inverse", heavy)
    return column, units


@pytest.mark.parametrize("name", ["two causes", "frozen pair"])
def test_an_overflow_at_a_dropped_time_is_the_dense_walk_s(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DR12: the dense tilt of a zero jump by an infinite factor; with held pairs elsewhere too."""
    plain = _fit(name)
    column, units = _overflowing(monkeypatch, plain)
    fit = _fit(name, rows=ROWS, resident=0.0)
    assert fit.overflowed > 0, "the fixture must overflow at a dropped time"
    record = fit.result.record
    assert (record.steps, record.halvings) != (
        plain.result.record.steps,
        plain.result.record.halvings,
    )
    if name == "frozen pair":
        assert fit.result.record.frozen_share > 0

    def heavy(weight: np.ndarray) -> np.ndarray:
        weight = weight.copy()
        weight[:, units, column] = HEAVY
        return weight

    reference = _reference(fit, name, adjust_weight=heavy)
    assert _bitwise(_walk_fields(fit.run.result), _walk_fields(reference.result)) == []
    assert fit.result.record.off_grid_share == reference.off_grid.share


def test_a_nan_initial_jump_is_kept_and_walked_as_the_dense_walk_walks_it() -> None:
    at = _event_free_time("two causes")
    extra = {"hazard_library": [Wrapped("nan", at)], "censoring_library": [Wrapped()]}
    fit = _fit("two causes", **extra)
    initial = fit.initial
    position = int(np.searchsorted(initial.grid, at))
    assert np.isnan(initial.hazards.jumps(0, initial.n)[..., position]).any()
    assert position in initial.kept
    reference = _reference(fit, "two causes")
    assert _close(_walk_fields(fit.run.result), _walk_fields(reference.result)) == []


def test_skipping_a_replayed_hold_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation control: a replay that forgets an accepted step's held pairs differs."""
    name = "accepted hold"
    kept = _fit(name, rows=ROWS, resident=1.0)
    rebuilt = _fit(name, rows=ROWS, resident=0.0)
    assert any(step.held is not None for step in kept.run.accepted)
    assert _bitwise(_record_fields(rebuilt), _record_fields(kept)) == []

    def forget(self: Any, flat: Any, lo: int, hi: int) -> Any:
        return np.zeros((2, hi - lo), dtype=bool)

    monkeypatch.setattr(_chunked._Walker, "local", forget)
    mutated = _fit(name, rows=ROWS, resident=0.0)
    assert _bitwise(_record_fields(mutated), _record_fields(kept)) != []


def test_the_combined_jump_refusal_comes_before_the_zero_survival_one() -> None:
    """DR13: the dense engine checked every unit's jumps before any survival."""
    name = "two causes"
    data = law(*FIXTURES[name][:2])
    w0 = np.asarray(data.baseline)[:, 0]
    # The zero survival sits in an earlier block than the combined jump.
    assert np.flatnonzero(w0 < -1.8).min() < np.flatnonzero(w0 > 1.8).min()
    extra = {"hazard_library": [Wrapped("refuse", 1.0)], "censoring_library": [Wrapped()]}
    with pytest.raises(DataError, match="combined jump of one or more"):
        _fit(name, rows=ROWS, **extra)


@pytest.mark.parametrize("which", ["hazard", "censoring"])
def test_a_learner_that_changes_between_calls_is_refused_before_the_walk(which: str) -> None:
    extra = {"hazard_library": [Wrapped()], "censoring_library": [Wrapped()]}
    extra[f"{which}_library"] = [Wrapped("noise")]
    with pytest.raises(MethodConfigurationError, match="predictions change between calls"):
        estimator("two causes", **extra).fit_data(law(*FIXTURES["two causes"][:2]))


def test_a_truncation_curve_replays_the_fit_with_no_block_kept() -> None:
    frames = []
    for resident in (0.0, 1.0):
        fit = _fit("two causes", rows=ROWS, resident=resident)
        # The curve refuses unless its replay at the fit's floor reproduces the fit.
        with _patched(rows=ROWS, resident=resident), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frames.append(continuous_truncation_curve(fit.result, bounds=(0.01, 0.02, 0.05)))
    assert len(set(frames[0]["bound"])) >= 3
    assert frames[0].equals(frames[1])


def test_every_kind_of_time_is_named() -> None:
    assert set(itertools.chain(FIXTURES, TOLERANT)) == set(GOLDEN)


def test_a_kept_block_takes_each_step_in_its_own_arrays(monkeypatch: pytest.MonkeyPatch) -> None:
    """Accepted steps are written into a kept block's arrays, so threads never move them.

    New arrays would come from the malloc arena of whichever thread replayed the block, and
    the freed kept arrays would pile up in other threads' arenas (R7, 16k: four-thread peaks
    crept from 1.04 to 1.16 times the prediction over the passes).
    """
    keep = _chunked._Walker.keep
    seen: dict[int, list[tuple[int, int]]] = {}

    def watched(self: Any, k: int, state: Any) -> Any:
        kept = keep(self, k, state)
        if k < self.resident:
            seen.setdefault(k, []).append((kept.applied, kept.jumps.ctypes.data))
        return kept

    monkeypatch.setattr(_chunked._Walker, "keep", watched)
    fit = _fit("two causes", rows=37, resident=0.5, n_jobs=2)
    assert fit.run.result.steps >= 1
    stepped = [calls for calls in seen.values() if max(applied for applied, _ in calls) >= 1]
    assert stepped
    for calls in seen.values():
        assert len({address for _, address in calls}) == 1, calls
