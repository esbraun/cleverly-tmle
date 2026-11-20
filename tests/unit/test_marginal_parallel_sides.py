"""X106 G4b: the continuous marginal-sensitivity sides run on the fit's ``n_jobs``.

An equivalent implementation (mode E, bitwise): every bound and interval equals the serial
loop's, and the caller sees the same warnings, in the same order, as the serial loop shows.
"""

from __future__ import annotations

import os
import pickle
import re
import warnings
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, LogisticRegression, QuantileRegressor
from sklearn.neighbors import KNeighborsRegressor

import cleverly.sensitivity.marginal as marginal_module
from cleverly._warnings import recording
from cleverly.estimators import TMLE
from cleverly.sensitivity import marginal_sensitivity_bounds


def _frame(n: int = 500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n, 2))
    a = rng.binomial(1, expit(0.3 * w[:, 0] - 0.2 * w[:, 1]))
    y = 0.4 * a + 0.3 * w[:, 0] + rng.normal(size=n)
    frame = pd.DataFrame({"Y": y, "A": a, "W0": w[:, 0], "W1": w[:, 1]})
    frame["sw"] = rng.uniform(0.5, 2.0, size=n)
    observed = rng.binomial(1, expit(1.0 + 0.2 * w[:, 1]))
    frame["delta"] = observed
    return frame


def _fit(n_jobs: int, *, weights: bool = False, missing: bool = False) -> Any:
    frame = _frame()
    options: dict[str, Any] = {}
    learners: dict[str, Any] = {}
    if weights:
        options["weights"] = "sw"
    if missing:
        frame.loc[frame["delta"] == 0, "Y"] = np.nan
        options["delta"] = "delta"
        learners["missingness_learner"] = LogisticRegression(max_iter=500)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            TMLE(
                outcome_learner=LinearRegression(),
                treatment_learner=LogisticRegression(max_iter=500),
                cross_fit=False,
                n_folds=5,
                simultaneous=False,
                random_state=11,
                n_jobs=n_jobs,
                **learners,
            )
            .fit(frame, outcome="Y", treatment="A", covariates=["W0", "W1"], **options)
            .single()
        )


def _frame_bytes(bounds: Any) -> dict[str, Any]:
    frame = bounds.to_frame()
    frame = frame.to_pandas() if hasattr(frame, "to_pandas") else frame
    out: dict[str, Any] = {}
    for column in frame.columns:
        values = np.asarray(frame[column])
        out[column] = values.tobytes() if values.dtype != object else list(values)
    return out


def _quantile(tau: float) -> QuantileRegressor:
    return QuantileRegressor(quantile=tau, alpha=0.0, solver="highs")


@pytest.mark.parametrize(
    ("options", "call"),
    [
        ({}, {}),
        ({"weights": True}, {}),
        ({"missing": True}, {}),
        ({}, {"estimand": "ey1"}),
        # A closure quantile factory, which only cloudpickle can send to a worker.
        ({}, {"quantile_learner": lambda tau: _quantile(tau)}),
    ],
    ids=["plain", "weighted", "missing", "one-arm", "closure-factory"],
)
def test_parallel_sides_give_the_serial_bounds(
    options: dict[str, Any], call: dict[str, Any]
) -> None:
    serial = marginal_sensitivity_bounds(_fit(1, **options), gamma=(1.0, 1.5, 2.0), **call)
    parallel = marginal_sensitivity_bounds(_fit(2, **options), gamma=(1.0, 1.5, 2.0), **call)
    assert _frame_bytes(serial) == _frame_bytes(parallel)


def test_memory_mapped_inputs_give_the_serial_bounds() -> None:
    # joblib hands workers read-only memory maps of arrays above max_nbytes (1 MB by default);
    # forcing it on small arrays shows no side writes into its inputs.
    serial = marginal_sensitivity_bounds(_fit(1), gamma=(1.0, 2.0))
    with joblib.parallel_config(max_nbytes="1K"):
        mapped = marginal_sensitivity_bounds(_fit(2), gamma=(1.0, 2.0))
    assert _frame_bytes(serial) == _frame_bytes(mapped)


class _NoisyLinear(LinearRegression):
    """A tail learner that raises a third-party warning on every fit."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> _NoisyLinear:
        warnings.warn("tail learner did not converge", ConvergenceWarning, stacklevel=2)
        return super().fit(X, y, sample_weight=sample_weight)


def _seen(
    n_jobs: int, learner: Any, action: str = "always"
) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    """The warnings the caller sees under filter ``action``, and the package's records."""
    result = _fit(n_jobs, weights=True)
    with warnings.catch_warnings(record=True) as caught, recording() as records:
        warnings.simplefilter(action)  # type: ignore[arg-type]
        marginal_sensitivity_bounds(result, gamma=(1.0, 2.0), pseudo_outcome_learner=learner)
    shown = [(w.category, str(w.message), w.filename, w.lineno) for w in caught]
    recorded = [(r.category, r.message, r.count) for r in records]
    return shown, recorded


@pytest.mark.parametrize(
    "learner",
    # A package warning (KNeighbors ignores the sampling weights) and a third-party one.
    [KNeighborsRegressor(), _NoisyLinear()],
    ids=["package-warning", "third-party-warning"],
)
@pytest.mark.parametrize("action", ["always", "default", "module"])
def test_parallel_sides_show_the_serial_warnings(learner: Any, action: str) -> None:
    # Under "default" and "module" the serial loop shows every repeat, because each learner fit
    # resets the once-per-location registry; the replay passes no registry, so it does too.
    serial = _seen(1, learner, action)
    assert len(serial[0]) > 1, "the fixture must repeat a warning for the witness to measure"
    assert _seen(2, learner, action) == serial


def test_dropping_the_replay_loses_the_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    # Mutation control: without the replay the caller sees nothing from the workers.
    serial = _seen(1, KNeighborsRegressor())
    monkeypatch.setattr(marginal_module, "_replay", lambda shown: None)
    assert _seen(2, KNeighborsRegressor())[0] != serial[0]


def test_a_caller_filter_still_applies_to_worker_warnings() -> None:
    result = _fit(2, weights=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        with pytest.raises(ConvergenceWarning, match="tail learner did not converge"):
            marginal_sensitivity_bounds(
                result, gamma=(1.0, 2.0), pseudo_outcome_learner=_NoisyLinear()
            )


def _module_pattern(result: Any) -> str:
    """A ``module=`` filter pattern matching the module that raises the fixture's warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        marginal_sensitivity_bounds(result, gamma=(1.0, 2.0), pseudo_outcome_learner=_NoisyLinear())
    names = {marginal_module._module_of(w.filename) for w in caught}
    assert len(names) == 1, names
    name = names.pop()
    assert name is not None
    return re.escape(name) + r"\Z"


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_a_caller_module_filter_applies_to_worker_warnings(n_jobs: int) -> None:
    # A filter on ``module=`` matches the replayed warning as it matches the serial one.
    result = _fit(n_jobs, weights=True)
    pattern = _module_pattern(_fit(1, weights=True))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=ConvergenceWarning, module=pattern)
        marginal_sensitivity_bounds(result, gamma=(1.0, 2.0), pseudo_outcome_learner=_NoisyLinear())
    assert not [w for w in caught if issubclass(w.category, ConvergenceWarning)]


def test_dropping_the_module_name_bypasses_a_module_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    # Mutation control: replayed without its module name, the warning escapes the filter.
    pattern = _module_pattern(_fit(1, weights=True))
    monkeypatch.setattr(marginal_module, "_module_of", lambda filename: None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=ConvergenceWarning, module=pattern)
        marginal_sensitivity_bounds(
            _fit(2, weights=True), gamma=(1.0, 2.0), pseudo_outcome_learner=_NoisyLinear()
        )
    assert [w for w in caught if issubclass(w.category, ConvergenceWarning)]


def _warning_state() -> tuple[Any, ...]:
    return (list(warnings.filters), warnings.showwarning, warnings._showwarnmsg_impl)  # type: ignore[attr-defined]


class _CountingCatch:
    """Stands in for ``warnings`` in the module and counts ``catch_warnings`` entries."""

    def __init__(self) -> None:
        self.entered = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(warnings, name)

    def catch_warnings(self, *args: Any, **kwargs: Any) -> Any:
        self.entered += 1
        return warnings.catch_warnings(*args, **kwargs)


def _threaded(monkeypatch: pytest.MonkeyPatch, spy: _CountingCatch) -> tuple[Any, ...]:
    result = _fit(2, weights=True)
    monkeypatch.setattr(marginal_module, "warnings", spy)
    before = _warning_state()
    with (
        warnings.catch_warnings(record=True) as caught,
        joblib.parallel_config(backend="threading"),
    ):
        warnings.simplefilter("always")
        bounds = marginal_sensitivity_bounds(
            result, gamma=(1.0, 1.5, 2.0), pseudo_outcome_learner=_NoisyLinear()
        )
    shown = sorted((w.category.__name__, str(w.message), w.filename, w.lineno) for w in caught)
    return bounds, shown, before, _warning_state()


def test_thread_backend_sides_raise_their_warnings_in_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # On threads the sides share the caller's process, where ``catch_warnings`` would swap
    # process-wide state under the other threads: those sides raise in place instead.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        serial_bounds = marginal_sensitivity_bounds(
            _fit(1, weights=True), gamma=(1.0, 1.5, 2.0), pseudo_outcome_learner=_NoisyLinear()
        )
    serial = sorted((w.category.__name__, str(w.message), w.filename, w.lineno) for w in caught)
    spy = _CountingCatch()
    bounds, shown, before, after = _threaded(monkeypatch, spy)
    assert spy.entered == 0
    assert after == before
    assert _frame_bytes(bounds) == _frame_bytes(serial_bounds)
    assert len(serial) > 1
    assert shown == serial


def test_capturing_on_threads_is_what_the_guard_prevents(monkeypatch: pytest.MonkeyPatch) -> None:
    # Mutation control: with the process check defeated, thread sides enter catch_warnings.
    pids = iter(range(10**9, 10**9 + 10**6))

    class _Os:
        def __getattr__(self, name: str) -> Any:
            return getattr(os, name)

        def getpid(self) -> int:
            return next(pids)

    monkeypatch.setattr(marginal_module, "os", _Os())
    spy = _CountingCatch()
    _threaded(monkeypatch, spy)
    assert spy.entered > 0


def test_gamma_one_sides_stay_in_the_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    # Gamma = 1 is a closed form: its sides take the serial loop even at n_jobs > 1.
    calls: list[int] = []
    original = marginal_module.map_parallel

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(marginal_module, "map_parallel", counting)
    result = _fit(2)
    marginal_sensitivity_bounds(result, gamma=(1.0,))
    assert calls == []
    marginal_sensitivity_bounds(result, gamma=(1.0, 2.0))
    assert calls


class _Unpicklable(UserWarning):
    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.hook = lambda: None


def test_a_warning_that_does_not_pickle_travels_as_text() -> None:
    message = _Unpicklable("cannot travel")
    shown = marginal_module._portable(message, _Unpicklable, "f.py", 3)
    assert shown == ("cannot travel", _Unpicklable, "f.py", 3)
    pickle.dumps(shown)

    class Local(UserWarning):
        pass

    shown = marginal_module._portable(Local("local"), Local, "f.py", 4)
    assert shown == ("local", Warning, "f.py", 4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        marginal_module._replay([shown])
    assert [(w.category, str(w.message)) for w in caught] == [(Warning, "local")]
