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

import itertools
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
    predictable: np.ndarray | None = None


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

        matrix = engine._Predictable.matrix

        def capture_predictable(self: Any) -> Any:
            seen["predictable"] = matrix(self)
            return seen["predictable"]

        patch.setattr(engine._Predictable, "matrix", capture_predictable)
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
        seen.get("predictable"),
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
    # The switched variance's predictable variation, from the targeted jumps the final pass
    # collects on the full grid.
    assert fit.predictable is not None and reference.predictable is not None
    assert np.array_equal(fit.predictable, reference.predictable)
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
    ours = T._direction(problem, _chunked._light(survival, incidence), v, jumps)

    lo, hi = plan.bounds[0]
    full_jumps = initial.hazards.jumps(lo, hi)
    full = T.Problem(
        initial.A[lo:hi],
        initial.T[lo:hi, None] >= initial.grid[None, :],
        np.zeros((len(initial.causes), hi - lo, initial.grid.size), dtype=bool),
        plan._weight(0),
        initial.target_weights,
    )
    theirs = dense._direction(full, dense.evaluate(full, full_jumps), v, full_jumps)
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
    # The default switched variance: its smallest budget keeps no block, so the walk replays
    # and the predictable variation is added from replayed blocks.
    with pytest.raises(_memory.ContinuousSurvivalMemoryError) as caught:
        estimator(name, max_memory_bytes=1).fit_data(data)
    smallest = caught.value.predicted
    assert smallest is not None
    tight = _fit(name, max_memory_bytes=smallest)
    loose = _fit(name, max_memory_bytes=2**62)
    assert len(engine.blocks(tight.initial.n, tight.initial.block)) > 1
    assert tight.run.replayed > 0
    assert tight.predictable is not None and loose.predictable is not None
    assert np.array_equal(tight.predictable, loose.predictable)
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


class _Dropped:
    """A plan stub for :meth:`_chunked._Walker.reach`: unit weights at every dropped time."""

    def __init__(self, c: int, gaps: np.ndarray) -> None:
        self.c, self.gaps = c, gaps
        self.bounds = [(0, c)]

    def dropped_weight(self, k: int) -> np.ndarray:
        return np.ones((2, self.c))

    def gap_ends(self) -> np.ndarray:
        return self.gaps


def _reach_case(
    weights: np.ndarray,
    kept: np.ndarray,
    size: int,
    jumps: np.ndarray,
    restricted: bool,
    v: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, Any]:
    """The walker's bound and the dense direction's largest positive value at dropped times.

    ``v`` is the direction in target space, one everywhere by default.
    """
    J, c = jumps.shape[1:3]
    arm = np.arange(c) % 2
    survival, incidence = T.curves(jumps)
    G = kept.size
    if restricted:
        expand = np.searchsorted(kept, np.arange(size), side="right") - 1
        walk_weights, extra = weights, (expand, kept)
    else:
        walk_weights, extra = weights[:, kept], ()
    problem = T.Problem(
        arm,
        np.ones((c, G), bool),
        np.zeros((J, c, G), bool),
        np.ones((2, c, G)),
        walk_weights,
        *extra,
    )
    state = _chunked._State(jumps, survival, incidence, 0, problem)
    gaps = np.flatnonzero(kept - np.concatenate([[-1], kept[:-1]]) > 1)
    walker = _chunked._Walker(_Dropped(c, gaps), resident=0, tilt="all")  # type: ignore[arg-type]
    v = np.ones(2 * J * weights.shape[0]) if v is None else v
    bound = walker.reach(0, state, v)
    full = np.zeros((*jumps.shape[:3], size))
    full[..., kept] = jumps
    curves = T.curves(full)
    dense_problem = T.Problem(
        arm, np.ones((c, size), bool), np.zeros((J, c, size), bool), np.ones((2, c, size)), weights
    )
    direction = T._direction(dense_problem, _chunked._light(*curves), v, full)
    dropped = np.ones(size, dtype=bool)
    dropped[kept] = False
    largest = np.maximum(direction[..., dropped].max(axis=(1, 3)), 0.0)
    return bound, largest, (state, v, gaps, direction[..., dropped])


def _difference_bound(state: Any, v: np.ndarray, gaps: np.ndarray) -> np.ndarray:
    """The bound before the conditional risk, as main computed it: (R - F W) / S at each run."""
    problem = state.problem
    J, Q, c = problem.n_causes, problem.n_rows, problem.n
    largest = np.zeros((2, c))
    for a in (0, 1):
        own = v[a * J * Q : (a + 1) * J * Q].reshape(J, Q) @ problem.target_weights
        revs = np.cumsum(own[:, ::-1], axis=1)[:, ::-1]
        W, before = revs[:, gaps], gaps - 1
        first = before < 0
        shared = np.zeros((c, gaps.size))
        for j in range(J):
            weighted = state.incidence[a, j] * own[j][None, :]
            rev = np.cumsum(weighted[:, ::-1], axis=1)[:, ::-1][:, gaps]
            F = np.where(first[None, :], 0.0, state.incidence[a, j][:, before])
            shared += rev - F * W[j][None, :]
        shared /= np.where(first[None, :], 1.0, state.survival[a][:, before])
        largest[a] = np.maximum((W.max(axis=0)[None, :] - shared).max(axis=1), 0.0)
    return largest


def _steep_jumps(c: int, G: int, last: float) -> np.ndarray:
    """One cause whose survival falls to about 1e-16 before a jump of 0.3 to 0.7."""
    rng = np.random.default_rng(0)
    jumps = np.zeros((2, 1, c, G))
    jumps[..., : G - 4] = 0.9
    jumps[..., G - 4 : G - 2] = rng.uniform(0.0, 0.8, size=(2, 1, c, 2))
    jumps[..., G - 2] = rng.uniform(0.3, 0.7, size=(2, 1, c))
    jumps[..., G - 1] = last
    return jumps


def test_the_overflow_bound_is_the_dense_direction_where_the_survival_is_tiny() -> None:
    """The bound at dropped times is the conditional-risk direction's own value, bit for bit.

    One cause and unit weights make the bound exact.  Where the survival before a run of
    dropped times is near 1e-16, the difference form of the quotient loses every digit, and
    the bound read from it falls below the direction by more than the margin between
    ``OVERFLOW_BOUND`` and ``exp``'s overflow (witness of the rederived bound).
    """
    c, G = 400, 20
    weights = np.zeros((1, G + 1))
    kept = np.delete(np.arange(G + 1), G - 2)  # one dropped time, before the steep jump
    weights[0, kept[-1]] = 1.0
    bound, largest, (state, v, gaps, _) = _reach_case(
        weights, kept, G + 1, _steep_jumps(c, G, 5e-4), restricted=False
    )
    assert np.array_equal(bound, largest)
    assert state.survival[..., G - 3].max() < 1e-15
    old = _difference_bound(state, v, gaps)
    margin = 709.78 / _chunked.OVERFLOW_BOUND
    assert (old * margin < largest).sum() >= 3


@pytest.mark.parametrize("where", ["inner runs", "trailing run"])
def test_the_restricted_overflow_bound_reads_every_run_of_dropped_times(where: str) -> None:
    """A restricted mean's weights move within a run; the bound takes each run's largest W.

    Dropped times lead, follow every kept time and trail the last one.  Weights on the whole
    grid put the largest direction inside runs before the last kept time, where W at a dropped
    time exceeds W at the run's end; weights only from the last kept time on put it in the
    trailing run, where nothing follows and the quotient is zero.
    """
    c, G = 400, 20
    size = 3 * G + 2
    kept = np.arange(1, 3 * G, 3)
    grid = np.linspace(0.1, 3.0, size)
    if where == "inner runs":
        weights = T.restricted_weights(grid, 2.95)[None, :]
        jumps = _steep_jumps(c, G, 5e-4)
    else:
        weights = np.zeros((1, size))
        weights[0, kept[-1] :] = 0.1
        jumps = _steep_jumps(c, G, 0.5)
    bound, largest, (_, _, _, dense) = _reach_case(weights, kept, size, jumps, restricted=True)
    assert np.array_equal(bound, largest)
    trailing = dense.shape[-1] - (size - 1 - kept[-1])
    at = dense.argmax(axis=-1)
    assert (at >= trailing).all() if where == "trailing run" else (at < trailing).all()


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
