"""The allocation preflight of the continuous-time engine (``survival/_memory.py``).

The prediction must bound what a fit holds and stay close to it, so a request is refused only
when it could not have fitted.  The upper-bound and tightness tests measure the ``tracemalloc``
peak (NumPy reports its data buffers to it) of a fit whose data are built outside the window,
on fixtures whose walk accepts at least two steps, so the trial after an acceptance runs.  The
mutation controls drop one array from the prediction and must then fall below the measurement.
"""

from __future__ import annotations

import tracemalloc
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly import CausalStudy, RegimeContrast, TimeToEvent
from cleverly.exceptions import DataError, MethodConfigurationError
from cleverly.survival import ContinuousSurvivalTMLE, CoxHazard, _memory, engine
from cleverly.survival.data import ContinuousSurvivalData
from cleverly.survival.diagnostics import continuous_truncation_curve
from cleverly.validation import refute

#: Prediction over measurement may not exceed this (comparison plan, section 5).
TIGHTNESS = 1.10
N = 1_500


def _law(n: int, causes: int, seed: int = 1) -> tuple[np.ndarray, ...]:
    """Exponential causes and censoring, as the scaling assessment's fixture, with ``causes``."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    a = rng.binomial(1, 1.0 / (1.0 + np.exp(-0.3 * w)))
    times = [rng.exponential(1.0 / np.exp(-1.0 + 0.3 * w - 0.4 * a)) for _ in range(causes)]
    censor = rng.exponential(1.0 / 0.15, n)
    t = np.minimum.reduce([*times, censor])
    d = np.zeros(n, dtype=int)
    for j, tj in enumerate(times):
        d[tj == t] = j + 1
    return t, d, a, w


def _data(n: int, causes: int, seed: int = 1) -> ContinuousSurvivalData:
    t, d, a, w = _law(n, causes, seed)
    return ContinuousSurvivalData(t, d, a, w[:, None], ("t", "d", "a", "w"), None, "pandas")


def _estimator(**extra: Any) -> ContinuousSurvivalTMLE:
    settings: dict[str, Any] = {
        "cross_fit": False,
        "n_folds": 3,
        "learner_folds": 3,
        "random_state": 0,
    }
    settings.update(extra)
    return ContinuousSurvivalTMLE((1.0, 2.0), **settings)


def _prediction(estimator: ContinuousSurvivalTMLE, data: Any, *, retained: bool = False) -> Any:
    """The estimate the fit's own check computes, captured from the call."""
    seen: list[Any] = []
    original = engine.check_memory

    class _Captured(Exception):
        pass

    def capture(estimate: Any, budget: Any, **kwargs: Any) -> None:
        seen.append(estimate)
        raise _Captured

    engine.check_memory = capture  # type: ignore[assignment]
    try:
        estimator._initial(data, int(estimator.random_state or 0), retained=retained)
    except _Captured:
        pass
    finally:
        engine.check_memory = original
    return seen[-1]


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


def _quiet(call: Callable[[], Any]) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return call()


FIXTURES: dict[str, tuple[int, dict[str, Any]]] = {
    "one cause": (1, {}),
    "two causes, cross-fitted, restricted mean": (
        2,
        {"cross_fit": True, "restricted_horizons": (2.0,)},
    ),
    "three causes": (3, {}),
    "factual tilt": (2, {"tilt": "factual"}),
    "two causes, influence-curve variance": (2, {"variance": "influence_curve"}),
}


@pytest.mark.parametrize("name", list(FIXTURES))
def test_the_prediction_bounds_a_fit_and_stays_within_ten_percent(name: str) -> None:
    causes, extra = FIXTURES[name]
    data = _data(N, causes)
    estimator = _estimator(**extra)
    predicted = _prediction(estimator, data)
    with _traced() as peak:
        result = _quiet(lambda: estimator.fit_data(data))
    assert result.record.steps >= 2, "the fixture must run a trial after an accepted step"
    assert peak[0] <= predicted.bytes
    assert predicted.bytes <= TIGHTNESS * peak[0]
    # Mutation control: without one (2, J, n, M) array the prediction misses the peak.
    dropped = predicted.bytes - 16 * causes * N * predicted.times
    assert dropped < peak[0]
    if estimator.variance == "switched":
        # ...and so it does without the floored ratio the switched variance holds.
        assert predicted.bytes - 16 * N * predicted.times < peak[0]


def test_the_prediction_bounds_a_truncation_curve_over_every_floor() -> None:
    data = _data(N, 2)
    result = _quiet(lambda: _estimator().fit_data(data))
    assert result.record.steps >= 2
    predicted = _prediction(result.estimator, data, retained=True)
    with _traced() as peak:
        frame = _quiet(lambda: continuous_truncation_curve(result, bounds=(0.01, 0.02, 0.05)))
    assert len(set(frame["bound"])) >= 3
    assert peak[0] <= predicted.bytes <= TIGHTNESS * peak[0]
    assert predicted.bytes - 16 * 2 * N * predicted.times < peak[0]


def test_each_phase_stays_below_its_own_prediction(monkeypatch: pytest.MonkeyPatch) -> None:
    from cleverly.inference import multiplier

    data = _data(N, 2)
    estimator = _estimator(restricted_horizons=(2.0,), simultaneous=True, n_multiplier=2_000)
    predicted = _prediction(estimator, data).phases
    peaks: dict[str, int] = {}
    open_windows: list[str] = []
    base = [0]

    def fold() -> None:
        value = tracemalloc.get_traced_memory()[1] - base[0]
        for window in open_windows:
            peaks[window] = max(peaks.get(window, 0), value)
        tracemalloc.reset_peak()

    def wrap(owner: Any, attribute: str, name: str) -> None:
        original = getattr(owner, attribute)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            fold()
            open_windows.append(name)
            try:
                return original(*args, **kwargs)
            finally:
                fold()
                open_windows.remove(name)

        monkeypatch.setattr(owner, attribute, wrapped)

    wrap(engine.ContinuousSurvivalTMLE, "_initial", "initial")
    wrap(engine, "_off_grid_share", "off_grid")
    wrap(engine, "build_estimates", "estimates")
    wrap(multiplier, "simultaneous_bands", "band")
    wrap(engine, "predictable_covariance", "predictable")
    original_walk = engine.walk

    def walk(
        problem: Any,
        jumps: Any,
        *,
        cutoff: float,
        norm: Any,
        step: float,
        max_steps: int,
        tilt: Any,
    ) -> Any:
        # Forward the jumps without a name here, so the wrapper does not hold what the
        # engine releases at the first accepted step.  The call names each keyword rather
        # than unpacking **kwargs: on Python 3.11 an unpacking call keeps its positional
        # arguments in a tuple on the caller's stack for the whole call, which would hold
        # the jumps the walk releases.
        box = [jumps]
        del jumps
        fold()
        open_windows.append("walk")
        try:
            return original_walk(
                problem,
                box.pop(),
                cutoff=cutoff,
                norm=norm,
                step=step,
                max_steps=max_steps,
                tilt=tilt,
            )
        finally:
            fold()
            open_windows.remove("walk")

    monkeypatch.setattr(engine, "walk", walk)
    with _traced():
        base[0] = tracemalloc.get_traced_memory()[0]
        result = _quiet(lambda: estimator.fit_data(data))
    assert result.record.steps >= 2
    initial = max(predicted[k] for k in ("nuisance", "curves", "indicators"))
    assert peaks["initial"] <= initial
    assert peaks["walk"] <= predicted["walk"]
    assert peaks["off_grid"] <= predicted["off_grid"]
    assert peaks["estimates"] <= predicted["band"]
    assert peaks["band"] <= predicted["band"]
    assert peaks["predictable"] <= predicted["predictable"]
    # Mutation control on a phase that does not bind: the window that fits the nuisances and
    # builds the curves misses its peak without the censoring jumps.
    cells = N * _prediction(estimator, data).times
    without_censoring = max(
        predicted["nuisance"] - 16 * cells,
        predicted["curves"] - 16 * cells,
        predicted["indicators"],
    )
    assert without_censoring < peaks["initial"]


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
    for part in ("300 units", "1 cause(s)", "M = ", "max_memory_bytes=", "Nothing was fitted"):
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


def test_the_budget_changes_no_result() -> None:
    data = _data(N, 1)
    plain = _quiet(lambda: _estimator().fit_data(data))
    budgeted = _quiet(lambda: _estimator(max_memory_bytes=2**62).fit_data(data))
    assert plain.record.steps >= 2
    for name, estimate in plain.estimates.items():
        other = budgeted.estimates[name]
        assert (estimate.psi, estimate.std_error) == (other.psi, other.std_error)
    for field in (
        "standardized_score",
        "steps",
        "initial_psi",
        "bounded_share",
        "frozen_share",
        "off_grid_share",
    ):
        assert getattr(plain.record, field) == getattr(budgeted.record, field)


def test_the_copied_constants_match_their_sources() -> None:
    from cleverly.inference import multiplier

    assert engine.OFF_GRID_TIMES == _memory.OFF_GRID
    assert 2 * multiplier._BLOCK_BYTES == _memory.BAND_BUFFER


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
    return pd.DataFrame({"time": t, "event": d, "Z": a, "W": w})


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
