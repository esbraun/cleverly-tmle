"""The memory model and budget of the continuous-time engine (``survival/_memory.py``).

The engine walks blocks of units and keeps as many blocks between passes as the budget holds.
The model predicts each phase's peak: the nuisance fits, the initial pass, the walk, the final
pass with the off-grid diagnostic, and the estimates with the band.  The prediction must bound
what each phase holds and stay close to it where it binds.  The tests measure the
``tracemalloc`` peak (NumPy reports its data buffers to it) of each phase of a fit whose data
are built outside the window, on fixtures whose walk accepts at least two steps.  The mutation
controls drop one term from a phase's prediction, which must then fall below the measurement.
"""

from __future__ import annotations

import tracemalloc
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly import CausalStudy, RegimeContrast, TimeToEvent
from cleverly.exceptions import DataError, MethodConfigurationError
from cleverly.survival import ContinuousSurvivalTMLE, CoxHazard, _chunked, _memory, engine
from cleverly.survival.data import ContinuousSurvivalData
from cleverly.survival.diagnostics import continuous_truncation_curve
from cleverly.validation import refute

#: Prediction over measurement may not exceed this where a phase binds (comparison plan, 5).
TIGHTNESS = 1.10
N = 2_000


def _law(n: int, causes: int, seed: int = 1, covariates: int = 1) -> tuple[np.ndarray, ...]:
    """Exponential causes and censoring, as the scaling assessment's fixture, with ``causes``."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n, covariates))
    a = rng.binomial(1, 1.0 / (1.0 + np.exp(-0.3 * w[:, 0])))
    times = [rng.exponential(1.0 / np.exp(-1.0 + 0.3 * w[:, 0] - 0.4 * a)) for _ in range(causes)]
    censor = rng.exponential(1.0 / 0.15, n)
    t = np.minimum.reduce([*times, censor])
    d = np.zeros(n, dtype=int)
    for j, tj in enumerate(times):
        d[tj == t] = j + 1
    return t, d, a, w


def _data(n: int, causes: int, seed: int = 1, covariates: int = 1) -> ContinuousSurvivalData:
    t, d, a, w = _law(n, causes, seed, covariates)
    names = ("t", "d", "a", *[f"w{k}" for k in range(covariates)])
    return ContinuousSurvivalData(t, d, a, w, names, None, "pandas")


def _estimator(**extra: Any) -> ContinuousSurvivalTMLE:
    settings: dict[str, Any] = {
        "cross_fit": False,
        "n_folds": 3,
        "learner_folds": 3,
        "random_state": 0,
        # A large first step and a tight cutoff, so every walk accepts two steps and halves.
        "step_size": 0.3,
        "tol": 1e-3,
    }
    settings.update(extra)
    return ContinuousSurvivalTMLE((1.0, 2.0), **settings)


def _quiet(call: Callable[[], Any]) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return call()


@dataclass
class Measured:
    """The traced peak and the prediction of each phase of one fit, with the walk's sizes."""

    peaks: dict[str, int]
    predicted: dict[str, int]
    shape: dict[str, Any]
    block: int
    blocks: int
    resident: int
    result: Any


#: The prediction's phase that bounds each measured window.
WINDOWS = {
    "nuisance": "nuisance",
    "initial": "initial",
    "walk": "walk",
    "final": "off_grid",
    "estimates": "band",
    "predictable": "predictable",
}


def _windows(measured: Measured) -> list[tuple[str, str]]:
    """The windows a fit opened, with their phases: ``predictable`` only under "switched"."""
    return [(window, phase) for window, phase in WINDOWS.items() if window in measured.peaks]


def _measure(
    estimator: ContinuousSurvivalTMLE,
    data: Any,
    *,
    resident: float | None = None,
) -> Measured:
    """Fit once and trace each phase's peak; ``resident`` forces the share of blocks kept."""
    peaks: dict[str, int] = {}
    windows: list[str] = []
    base = [0]
    seen: dict[str, Any] = {}

    def fold() -> None:
        value = tracemalloc.get_traced_memory()[1] - base[0]
        for window in windows:
            peaks[window] = max(peaks.get(window, 0), value)
        tracemalloc.reset_peak()

    def switch(close: str | None, open_: str | None) -> None:
        fold()
        if close in windows:
            windows.remove(close)
        if open_ is not None:
            windows.append(open_)

    initial = engine.ContinuousSurvivalTMLE._initial

    def traced_initial(self: Any, data: Any, seed: int, **kwargs: Any) -> Any:
        switch(None, "nuisance")
        try:
            return initial(self, data, seed, **kwargs)
        finally:
            switch("nuisance", None)
            switch("initial", None)

    hazards = engine.InitialHazards

    def traced_hazards(causes: Any, censoring: Any) -> Any:
        switch("nuisance", "initial")
        return hazards(causes, censoring)

    run_walk = _chunked.run_walk

    # The wrappers name each keyword the engine passes rather than unpacking **kwargs: an
    # unpacking call keeps its arguments on the caller's stack for the whole call.
    def traced_walk(
        plan: Any,
        *,
        cutoff: float,
        norm: Any,
        step: float,
        max_steps: int,
        tilt: Any,
        resident: int | None,
        final: Any,
    ) -> Any:
        kept = resident if forced is None else round(forced * len(plan.bounds))
        rows = sum(hi - lo for lo, hi in plan.bounds[:kept])
        seen.update(shape=plan.shape, block=plan.bounds[0][1], blocks=len(plan.bounds))
        seen["resident"] = kept
        seen["predicted"] = _memory.peak_bytes(
            **plan.shape, resident_rows=rows, resident_blocks=kept
        ).phases

        def traced_final(k: int, lo: int, hi: int, state: Any) -> None:
            if "walk" in windows:
                switch("walk", "final")
            final(k, lo, hi, state)

        switch(None, "walk")
        try:
            return run_walk(
                plan,
                cutoff=cutoff,
                norm=norm,
                step=step,
                max_steps=max_steps,
                tilt=tilt,
                resident=kept,
                final=traced_final if final is not None else None,
            )
        finally:
            switch("walk", None)
            switch("final", None)

    build = engine.build_estimates

    def traced_build(*args: Any, **kwargs: Any) -> Any:
        switch(None, "estimates")
        try:
            return build(*args, **kwargs)
        finally:
            switch("estimates", None)

    dense_predictable = engine.dense_predictable

    def traced_predictable(plan: Any, dlam: Any) -> Any:
        switch(None, "predictable")
        try:
            return dense_predictable(plan, dlam)
        finally:
            switch("predictable", None)

    forced = resident
    started = not tracemalloc.is_tracing()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(engine.ContinuousSurvivalTMLE, "_initial", traced_initial)
        patch.setattr(engine, "InitialHazards", traced_hazards)
        patch.setattr(_chunked, "run_walk", traced_walk)
        patch.setattr(engine, "build_estimates", traced_build)
        patch.setattr(engine, "dense_predictable", traced_predictable)
        if started:
            tracemalloc.start()
        try:
            base[0] = tracemalloc.get_traced_memory()[0]
            tracemalloc.reset_peak()
            result = _quiet(lambda: estimator.fit_data(data))
            fold()
        finally:
            if started:
                tracemalloc.stop()
    return Measured(
        peaks,
        seen["predicted"],
        seen["shape"],
        seen["block"],
        seen["blocks"],
        seen["resident"],
        result,
    )


def _terms(measured: Measured) -> dict[str, int]:
    """One term of each phase's prediction, larger than the prediction's slack there."""
    shape, c = measured.shape, measured.block
    n, J, M, G = shape["n"], shape["causes"], shape["times"], shape["kept"]
    flags = {"factual": shape["factual_tilt"], "restricted": shape["restricted"]}
    resident = _memory.slot_bytes(min(measured.resident * c, n), J, G)
    spread = 8 * n * 2 * J * _memory.OFF_GRID
    return {
        # The hazard fits' (n_train, d, d) arrays.
        "nuisance": _memory.COX_FIT * shape["train_size"] * shape["design_width"] ** 2,
        # One block's working arrays on the full grid: its jumps, combined jump and survival.
        "initial": 16 * J * c * M + 48 * c * M + 2 * J * c * M,
        # The kept blocks, or with none kept one block's state and a trial's arrays.
        "walk": resident
        if measured.resident
        else _memory.slot_bytes(c, J, G) + _memory._trial(c, J, M, G, **flags),
        # The kept blocks, or with none kept the off-grid diagnostic's (n, 2J x 100) spread.
        "final": resident if measured.resident else spread,
    }


FIXTURES: dict[str, tuple[int, dict[str, Any], float | None]] = {
    "one cause, every block kept": (1, {}, 1.0),
    "one cause, no block kept": (1, {}, 0.0),
    "three causes, half the blocks kept": (3, {}, 0.5),
    "restricted mean, cross-fitted, no block kept": (
        1,
        {"cross_fit": True, "restricted_horizons": (2.0,)},
        0.0,
    ),
    "factual tilt, half the blocks kept": (2, {"tilt": "factual"}, 0.5),
    "two causes, influence-curve variance, no block kept": (
        2,
        {"variance": "influence_curve"},
        0.0,
    ),
}


@pytest.fixture(autouse=True)
def _small_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blocks of a few MB, so a 2,000-unit fit spans many and its walk phase binds."""
    monkeypatch.setattr(_chunked, "CHUNK_CELLS", 2**17)


@pytest.mark.parametrize("name", list(FIXTURES))
def test_each_phase_stays_below_its_prediction_and_the_walk_within_ten_percent(
    name: str,
) -> None:
    causes, extra, resident = FIXTURES[name]
    measured = _measure(_estimator(**extra), _data(N, causes), resident=resident)
    assert measured.result.record.steps >= 2, "the fixture must run a trial after an acceptance"
    assert measured.blocks >= 4
    assert measured.resident == round((resident or 0.0) * measured.blocks)
    for window, phase in _windows(measured):
        assert measured.peaks[window] <= measured.predicted[phase], window
    # The walk and the final pass bind; each prediction stays within ten percent of its peak.
    for window in ("walk", "final"):
        assert measured.predicted[WINDOWS[window]] <= TIGHTNESS * measured.peaks[window], window
    # Mutation controls: without one term that each phase holds, its prediction misses it.
    terms = _terms(measured)
    assert measured.predicted["walk"] - terms["walk"] < measured.peaks["walk"]
    assert measured.predicted["off_grid"] - terms["final"] < measured.peaks["final"]
    assert measured.predicted["initial"] - terms["initial"] < measured.peaks["initial"]
    cells = measured.shape["n"] * measured.shape["times"]
    if not measured.shape["switched"]:
        assert "predictable" not in measured.peaks and "predictable" not in measured.predicted
        return
    # The switched variance's dense pass binds too; without the floored ratio
    # min(1, pi S_c / floor), (2, n, M), its prediction misses its peak.
    peak, predicted = measured.peaks["predictable"], measured.predicted["predictable"]
    assert predicted <= TIGHTNESS * peak
    assert predicted - 16 * cells < peak
    # The final pass collects the targeted (2, J, n, M) jumps on the full grid for it.
    collected = 16 * measured.shape["causes"] * cells
    assert measured.predicted["off_grid"] - collected < measured.peaks["final"]


def test_the_nuisance_phase_bounds_wide_cox_fits() -> None:
    """DR1, R7: the interaction Cox model on eight covariates and the treatment selection."""
    data = _data(N, 1, covariates=8)
    estimator = _estimator(hazard_library=[CoxHazard("interaction")])
    measured = _measure(estimator, data, resident=0.0)
    assert measured.shape["design_width"] == 17
    assert measured.result.record.steps >= 1
    peak, predicted = measured.peaks["nuisance"], measured.predicted["nuisance"]
    assert peak <= predicted <= 1.25 * peak
    # Mutation control: without the Cox fits' (n, d, d) arrays the prediction misses the peak.
    assert predicted - _terms(measured)["nuisance"] < peak


def test_the_exact_overflow_check_stays_within_the_walk_s_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Heavy censoring drops most times; huge steps send the exact check over many units."""
    monkeypatch.setattr(_chunked, "CHUNK_CELLS", 2**15)
    checked = [0]
    overflow = engine._FitPlan.overflow

    def counted(self: Any, k: int, units: Any, state: Any, v: Any, step: float) -> Any:
        checked[0] += len(units)
        return overflow(self, k, units, state, v, step)

    monkeypatch.setattr(engine._FitPlan, "overflow", counted)
    t, d, a, w = _law(N, 1)
    censor = np.random.default_rng(7).exponential(0.4, N)
    d = np.where(censor < t, 0, d)
    t = np.minimum(t, censor)
    data = ContinuousSurvivalData(t, d, a, w, ("t", "d", "a", "w0"), None, "pandas")
    measured = _measure(_estimator(step_size=3000.0), data, resident=0.0)
    assert measured.shape["kept"] < measured.shape["times"] / 3
    assert checked[0] > 100, "the fixture must run the exact check"
    assert measured.peaks["walk"] <= measured.predicted["walk"]


def test_a_stratified_learner_s_cell_tables_are_counted() -> None:
    """DR16: a stratum column of 40 levels, cross-fitted, at 4,000 units: each fold keeps its tables."""
    from cleverly.survival import StratifiedNelsonAalen

    t, d, a, _ = _law(4_000, 1)
    # Levels drawn uniformly, so every training fold sees every level.
    w = np.random.default_rng(3).integers(0, 40, size=(4_000, 1)).astype(float)
    data = ContinuousSurvivalData(t, d, a, w, ("t", "d", "a", "w0"), None, "pandas")
    library = [StratifiedNelsonAalen((1,))]
    estimator = _estimator(
        cross_fit=True, hazard_library=library, censoring_library=library, step_size=0.1
    )
    measured = _measure(estimator, data, resident=0.0)
    assert len(np.unique(w)) == 40 and measured.block >= 40
    for window, phase in _windows(measured):
        assert measured.peaks[window] <= measured.predicted[phase], window


def test_the_traced_peak_grows_with_n_not_with_n_times_the_grid() -> None:
    """DR17: with no block kept, doubling n about doubles the peak (the dense walk's is 4x)."""
    peaks = []
    for n in (1_000, 2_000):
        # The switched variance's predictable pass is dense in n M until it runs per block, so
        # the influence-curve variance shows the walk's growth.
        estimator = _estimator(variance="influence_curve")
        measured = _measure(estimator, _data(n, 1), resident=0.0)
        assert measured.blocks >= 4
        peaks.append(max(measured.peaks.values()))
    assert peaks[1] <= 2.5 * peaks[0]


def test_the_guard_reads_the_full_grid_with_no_block_kept() -> None:
    """The check runs before any fit, from M and the walk with nothing resident."""
    shape = {
        "n": 4_000,
        "causes": 2,
        "times": 3_000,
        "rows": 2,
        "covariates": 3,
        "train_size": 4_000,
        "design_width": 4,
        "simultaneous": False,
        "n_multiplier": 1_000,
    }
    full = _memory.peak_bytes(**shape)
    reduced = _memory.peak_bytes(**shape, kept=1_000)
    kept = _memory.peak_bytes(**shape, kept=1_000, resident_rows=4_000, resident_blocks=10**6)
    assert reduced.bytes <= full.bytes <= kept.bytes


@pytest.mark.parametrize("margin", [None, 1.0])
def test_kept_blocks_leave_the_allocator_margin_within_an_explicit_budget(
    margin: float | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R-2.2g: the kept blocks are as many as keep the walk's prediction times the margin
    within the budget; without the margin (mutation control) they overrun it."""
    shape = {
        "n": 4_000,
        "causes": 2,
        "times": 3_000,
        "rows": 2,
        "covariates": 3,
        "train_size": 4_000,
        "design_width": 4,
        "simultaneous": False,
        "n_multiplier": 1_000,
        "kept": 2_000,
    }
    from cleverly.survival._chunked import blocks, chunk_size

    bounds = blocks(4_000, chunk_size(4_000, 2, 3_000))

    def need(count: int) -> int:
        rows = sum(hi - lo for lo, hi in bounds[:count])
        phases = _memory.peak_bytes(**shape, resident_rows=rows, resident_blocks=count).phases
        return max(phases["walk"], phases["off_grid"])

    budget = int(_memory.ALLOCATOR_MARGIN * need(len(bounds) // 2))
    if margin is not None:
        monkeypatch.setattr(_memory, "ALLOCATOR_MARGIN", margin)
    count = _memory.resident_count(shape, bounds, budget)
    assert 0 < count < len(bounds)
    within = 1.15 * need(count) <= budget
    assert within is (margin is None)
    if margin is None:
        assert 1.15 * need(count + 1) > budget


def test_a_truncation_curve_holds_about_what_its_fit_holds() -> None:
    data = _data(1_000, 1)
    # The curve's walks keep the blocks the fit's did, under the same default budget.
    measured = _measure(_estimator(), data)
    result = measured.result
    with _traced() as peak:
        frame = _quiet(lambda: continuous_truncation_curve(result, bounds=(0.01, 0.02, 0.05)))
    assert len(set(frame["bound"])) >= 3
    # Its walks are the fit's; beside them it holds only its rows and each floor's estimates.
    assert peak[0] <= 1.01 * max(measured.predicted.values())


@contextmanager
def _traced() -> Iterator[list[int]]:
    """Yield a one-element list that holds the traced peak above the start, on exit."""
    started = not tracemalloc.is_tracing()
    if started:
        tracemalloc.start()
    out: list[int] = []
    start = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    try:
        yield out
    finally:
        out.append(tracemalloc.get_traced_memory()[1] - start)
        if started:
            tracemalloc.stop()


def _small() -> ContinuousSurvivalData:
    return _data(300, 1)


def _refused_prediction(**extra: Any) -> int:
    with pytest.raises(_memory.ContinuousSurvivalMemoryError) as caught:
        _estimator(max_memory_bytes=1, **extra).fit_data(_small())
    assert caught.value.predicted is not None
    return caught.value.predicted


def test_the_budget_refuses_one_byte_below_the_prediction_and_fits_at_it() -> None:
    predicted = _refused_prediction()
    with pytest.raises(MethodConfigurationError) as caught:
        _estimator(max_memory_bytes=predicted - 1).fit_data(_small())
    text = str(caught.value)
    for part in (
        "300 units",
        "1 cause(s)",
        "M = ",
        "max_memory_bytes=",
        "Nothing was fitted",
        "keeps no block between passes",
    ):
        assert part in text
    for budget in (predicted, predicted + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert _estimator(max_memory_bytes=budget).fit_data(_small())


class _Spy(CoxHazard):
    """A Cox learner that fails the test if it is ever fitted."""

    def fit(self, X: Any, time: Any, event: Any) -> Any:
        raise AssertionError("a refused request fitted a nuisance")


def test_a_refusal_fits_no_nuisance() -> None:
    with pytest.raises(_memory.ContinuousSurvivalMemoryError):
        _estimator(
            max_memory_bytes=1, hazard_library=[_Spy()], censoring_library=[_Spy()]
        ).fit_data(_small())


def test_the_default_budget_reads_the_available_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_memory, "available_memory", lambda: 1)
    with pytest.raises(_memory.ContinuousSurvivalMemoryError, match="the memory available"):
        _estimator().fit_data(_small())
    # An explicit budget overrides a low reading...
    assert _quiet(lambda: _estimator(max_memory_bytes=2**62).fit_data(_small()))
    # ...and a platform that reports nothing is not checked.
    monkeypatch.setattr(_memory, "available_memory", lambda: None)
    assert _quiet(lambda: _estimator().fit_data(_small()))
    # An explicit budget below the prediction refuses even with plenty available.
    monkeypatch.setattr(_memory, "available_memory", lambda: 2**62)
    with pytest.raises(_memory.ContinuousSurvivalMemoryError, match=r"max_memory_bytes=1\b"):
        _estimator(max_memory_bytes=1).fit_data(_small())


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "1"])
def test_the_budget_is_a_positive_integer(value: Any) -> None:
    with pytest.raises(DataError, match="max_memory_bytes"):
        _estimator(max_memory_bytes=value)


def test_the_budget_changes_no_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """DR17: every block kept or none, the same record, warnings and frame."""
    monkeypatch.setattr(_chunked, "CHUNK_CELLS", 2**17)
    data = _data(N, 1)
    runs = []
    for budget in (None, 2**62):
        with pytest.MonkeyPatch.context() as patch:
            if budget is None:
                # The smallest budget the guard accepts keeps no block.
                patch.setattr(engine, "resident_blocks", lambda plan, budget: 0)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = _estimator(max_memory_bytes=2**62).fit_data(data)
        runs.append((result, [(type(w.message), str(w.message)) for w in caught]))
    (plain, plain_warnings), (budgeted, budgeted_warnings) = runs
    assert plain.record.steps >= 2
    assert plain_warnings == budgeted_warnings
    assert plain.to_frame().equals(budgeted.to_frame())
    for name, estimate in plain.estimates.items():
        other = budgeted.estimates[name]
        assert (estimate.psi, estimate.std_error) == (other.psi, other.std_error)
    for field in (
        "standardized_score",
        "steps",
        "halvings",
        "initial_psi",
        "bounded_share",
        "last_time_floor_share",
        "frozen_share",
        "off_grid_share",
    ):
        assert getattr(plain.record, field) == getattr(budgeted.record, field)


def test_the_copied_constants_match_their_sources() -> None:
    from cleverly.inference import multiplier

    assert engine.OFF_GRID_TIMES == _memory.OFF_GRID
    assert 2 * multiplier._BLOCK_BYTES == _memory.BAND_BUFFER
    # A block's state: jumps, survival, incidences and the problem's arrays, per (unit, time).
    assert _memory.slot_bytes(1, 1, 1) == 66


def test_every_replay_reads_the_fit_s_budget() -> None:
    result = _quiet(lambda: _estimator(max_memory_bytes=2**62).fit_data(_small()))
    assert result.estimator.max_memory_bytes == 2**62
    refit = _quiet(lambda: result.estimator.refit(result.data, random_state=0))
    assert refit.estimator.max_memory_bytes == 2**62
    # A stored budget too small for the data reaches each replay.
    result.estimator.max_memory_bytes = 1
    with pytest.raises(_memory.ContinuousSurvivalMemoryError):
        result.estimator.refit(result.data, random_state=0)
    with pytest.raises(_memory.ContinuousSurvivalMemoryError):
        continuous_truncation_curve(result)
    with pytest.raises(_memory.ContinuousSurvivalMemoryError):
        refute(result, estimand="ate_regimen[1 vs 0 @ t=1]", tests=("placebo",), n_replicates=2)


def _frame(n: int = 300) -> pd.DataFrame:
    t, d, a, w = _law(n, 1)
    return pd.DataFrame({"time": t, "event": d, "Z": a, "W": w[:, 0]})


def _effect() -> Any:
    study = CausalStudy(
        _frame(),
        design=TimeToEvent(
            time="time", event="event", treatment="Z", baseline=["W"], grid="continuous"
        ),
    )
    return study, study.identify(RegimeContrast({"t": 1, "c": 0}, reference="c", horizons=[1, 2]))


def test_a_study_refusal_spells_the_estimator_call_that_reproduces_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study, effect = _effect()
    fitted = _quiet(lambda: effect.estimate(random_state=0, cross_fit=False, n_multiplier=500))
    monkeypatch.setattr(_memory, "available_memory", lambda: 1)
    with pytest.raises(_memory.ContinuousSurvivalMemoryError) as caught:
        effect.estimate(random_state=0, cross_fit=False, n_multiplier=500)
    text = str(caught.value)
    assert "cleverly.survival.ContinuousSurvivalTMLE((1.0, 2.0), " in text
    assert "simultaneous=True" in text and "max_memory_bytes=..." in text
    assert "With this seed it returns the result this study would." in text
    call = text.split("cleverly.survival.ContinuousSurvivalTMLE(", 1)[1].split(".fit_data(", 1)[0]
    call = call[: call.rindex(")")].replace("max_memory_bytes=...", "max_memory_bytes=2**62")
    direct = _quiet(lambda: eval(f"ContinuousSurvivalTMLE({call})").fit_data(study.data))
    assert list(direct.estimates) == list(fitted.estimates)
    for name, estimate in fitted.estimates.items():
        other = direct.estimates[name]
        assert (estimate.psi, estimate.std_error) == (other.psi, other.std_error)
    assert fitted.simultaneous.bands == direct.simultaneous.bands


def test_an_unrelated_study_refusal_keeps_its_text() -> None:
    _, effect = _effect()
    with pytest.raises(MethodConfigurationError) as caught:
        effect.estimate(q_bounds=(0.1, 0.9))
    assert not isinstance(caught.value, _memory.ContinuousSurvivalMemoryError)
    assert "fit the same estimator directly" not in str(caught.value)
