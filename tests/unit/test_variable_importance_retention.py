"""``variable_importance(keep_fits=False)`` keeps the estimates and drops each candidate's fit.

The two settings report the same numbers, the same summary and the same warnings; only what the
result holds on to differs. Without kept fits no candidate's data or fit outlives its turn, so
memory does not grow with the number of candidates.
"""

from __future__ import annotations

import gc
import importlib
import re
import tracemalloc
import warnings
import weakref
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import variable_importance
from cleverly.estimators import DRTMLE, TMLE
from cleverly.estimators.base import TMLEResult
from cleverly.exceptions import CapabilityError, DataError, RefusalReason
from cleverly.variable_importance import ImportanceFacts, VariableImportanceResult


def _estimator(**overrides: Any) -> TMLE:
    settings: dict[str, Any] = {
        "outcome_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "cross_fit": False,
        "simultaneous": False,
        "random_state": 1,
    }
    settings.update(overrides)
    return TMLE(**settings)


def _frame(n: int = 160, seed: int = 5, levels: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    x1 = rng.integers(0, levels, n) if levels > 2 else rng.binomial(1, 1 / (1 + np.exp(-w)))
    x2 = rng.binomial(1, 0.5, n)
    x3 = rng.binomial(1, 0.4, n)
    y = 1.2 * x1 + 0.3 * x2 + 0.2 * w + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"Y": y, "X1": x1, "X2": x2, "X3": x3, "W": w, "V": w * 0.5 + 1.0})


def _warning_frame() -> pd.DataFrame:
    # A constant covariate is dropped with a warning when each candidate's data is built, and
    # integer-valued weights read as counts, which warns too.
    frame = _frame()
    frame["K"] = 1.0
    frame["wt"] = np.random.default_rng(2).integers(1, 4, len(frame)).astype(float)
    return frame


def _one_candidate_warns() -> pd.DataFrame:
    # X2 depends strongly on W, so only X2's fit has its treatment probabilities limited and
    # warns; X1 before it and X3 after it do not.
    frame = _frame(seed=11)
    frame["X2"] = np.random.default_rng(111).binomial(1, 1 / (1 + np.exp(-5.0 * frame["W"])))
    return frame


def _run(frame: Any, *, keep_fits: bool, **extra: Any) -> VariableImportanceResult:
    settings: dict[str, Any] = {
        "outcome": "Y",
        "candidates": ("X1", "X2", "X3"),
        "covariates": ("W",),
        "estimator": _estimator(),
    }
    settings.update(extra)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return variable_importance(frame, keep_fits=keep_fits, **settings)


FIXTURES: dict[str, Callable[[], tuple[Any, dict[str, Any]]]] = {
    "plain": lambda: (_frame(), {}),
    "several parameters per candidate": lambda: (
        _frame(levels=3),
        {"candidates": ("X1", "X2")},
    ),
    "a fit that warns": lambda: (_one_candidate_warns(), {}),
    "clustered and weighted": lambda: (
        _frame().assign(cid=np.repeat(np.arange(40), 4), wt=np.linspace(0.5, 1.5, 160)),
        {"id": "cid", "weights": "wt"},
    ),
    "polars": lambda: (pl.from_pandas(_frame()), {}),
    "a DR-TMLE template": lambda: (
        _frame(),
        {
            "estimator": DRTMLE(
                outcome_learner=LinearRegression(),
                treatment_learner=LogisticRegression(max_iter=1000),
                reduced_outcome_learner=LinearRegression(),
                reduced_mechanism_residual_learner=LinearRegression(),
                cross_fit=False,
                simultaneous=False,
                random_state=1,
            )
        },
    ),
}


def _frame_bytes(result: VariableImportanceResult) -> list[tuple[str, Any]]:
    frame = result.to_frame()
    if isinstance(frame, pl.DataFrame):
        frame = frame.to_pandas()
    columns = [(name, np.asarray(frame[name])) for name in frame.columns]
    return [
        (name, values.tobytes() if values.dtype.kind in "fiub" else values.tolist())
        for name, values in columns
    ]


def _entries(result: VariableImportanceResult) -> list[tuple[Any, ...]]:
    return [
        (
            entry.candidate,
            entry.parameter,
            entry.adjustment_set,
            np.float64(entry.estimate.psi).tobytes(),
            np.float64(entry.estimate.std_error).tobytes(),
            np.float64(entry.estimate.pvalue).tobytes(),
            np.float64(entry.adjusted_pvalue).tobytes(),
            np.asarray(entry.estimate.influence_curve).tobytes(),
        )
        for entry in result
    ]


@pytest.mark.parametrize("fixture", list(FIXTURES))
def test_both_settings_report_the_same_thing(fixture: str) -> None:
    frame, extra = FIXTURES[fixture]()
    kept = _run(frame, keep_fits=True, **extra)
    dropped = _run(frame, keep_fits=False, **extra)
    assert _entries(dropped) == _entries(kept)
    assert _frame_bytes(dropped) == _frame_bytes(kept)
    for detail in ("short", "full"):
        k, d = kept.summary(detail=detail), dropped.summary(detail=detail)
        if kept.facts.cautious:
            # Without kept fits the fits' warnings are written out in place of the pointer to
            # result.fits; everything else in the summary is the same.
            assert k.note_codes.count("fit_cautions") == 1
            assert d.note_codes == tuple(
                "fit_cautions_listed" if code == "fit_cautions" else code for code in k.note_codes
            )
            assert [n for n in d.notes if n.code != "fit_cautions_listed"] == [
                n for n in k.notes if n.code != "fit_cautions"
            ]
            assert d.facts == k.facts
            listed = next(n.text for n in d.notes if n.code == "fit_cautions_listed")
            for texts in kept.facts.cautions.values():
                assert all(text in listed for text in texts)
        else:
            assert str(d) == str(k)
    assert repr(dropped) == repr(kept)
    assert dropped.facts == kept.facts
    assert kept.keep_fits and not dropped.keep_fits


def test_the_fixtures_reach_what_they_are_named_for() -> None:
    frame, extra = FIXTURES["several parameters per candidate"]()
    result = _run(frame, keep_fits=True, **extra)
    assert sum(entry.candidate == "X1" for entry in result) == 2
    frame, extra = FIXTURES["a fit that warns"]()
    # One candidate in the middle warns, so flagging every candidate, or the first or last
    # one, would not pass.
    assert _run(frame, keep_fits=True, **extra).facts.cautious == ("X2",)


def _facts_from_fits(fits: Mapping[str, TMLEResult]) -> ImportanceFacts:
    """The summary's reading of the fits before the facts were stored: the first fit's
    header, outcome and rows, and each fit whose summary warns, in candidate order."""
    first = next(iter(fits.values()))
    return ImportanceFacts(
        candidates=tuple(fits),
        first_fit=dict(first.summary().facts),
        outcome=first.data.outcome_name,
        n=int(first.data.n),
        cautions={
            candidate: texts
            for candidate, fit in fits.items()
            if (texts := tuple(item.text for item in fit.summary().notes if item.kind == "warning"))
        },
        fitted_method=first.fitted_method,
    )


def _last_fit_facts(fits: Mapping[str, TMLEResult]) -> ImportanceFacts:
    """A deliberate mutation: the header read from the last candidate's fit."""
    reversed_fits = dict(reversed(list(fits.items())))
    facts = _facts_from_fits(reversed_fits)
    return ImportanceFacts(
        tuple(fits),
        facts.first_fit,
        facts.outcome,
        facts.n,
        _facts_from_fits(fits).cautions,
        facts.fitted_method,
    )


@pytest.mark.parametrize(
    "fixture", ["plain", "several parameters per candidate", "a fit that warns"]
)
def test_the_stored_facts_are_what_the_fits_said(fixture: str) -> None:
    frame, extra = FIXTURES[fixture]()
    result = _run(frame, keep_fits=True, **extra)
    assert result.facts == _facts_from_fits(result.fits)


def test_facts_read_from_the_wrong_fit_are_caught() -> None:
    frame, extra = FIXTURES["a fit that warns"]()
    result = _run(frame, keep_fits=True, **extra)
    assert _last_fit_facts(result.fits) != result.facts


def _shown(
    frame: pd.DataFrame, keep_fits: bool, estimator: TMLE | None = None
) -> tuple[list[tuple[str, str]], list[Any]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = variable_importance(
            frame,
            outcome="Y",
            candidates=("X1", "X2"),
            covariates=("W", "K"),
            weights="wt",
            estimator=_estimator() if estimator is None else estimator,
            keep_fits=keep_fits,
        )
    shown = [(item.category.__name__, str(item.message)) for item in caught]
    return shown, [entry.estimate.psi for entry in result]


def test_the_user_sees_each_warning_as_often_under_both_settings() -> None:
    kept, _ = _shown(_warning_frame(), True)
    dropped, _ = _shown(_warning_frame(), False)
    assert any("constant" in message for _, message in kept)
    assert dropped == kept
    # The mute covers the second build of the data only: a warning the fit raises is shown
    # under both settings.
    tight = _estimator(g_bounds=0.45)
    kept, _ = _shown(_warning_frame(), True, tight)
    dropped, _ = _shown(_warning_frame(), False, tight)
    assert any(category == "PositivityWarning" for category, _ in kept)
    assert dropped == kept


def test_a_second_preparation_that_shows_its_warnings_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Deliberate mutation: without the mute, the re-built data warns a second time.
    module = importlib.import_module("cleverly.variable_importance")

    @contextmanager
    def loud() -> Iterator[None]:
        yield

    kept, _ = _shown(_warning_frame(), True)
    monkeypatch.setattr(module, "muted", loud)
    dropped, _ = _shown(_warning_frame(), False)
    assert dropped != kept


def test_the_fits_recorded_warnings_are_unchanged() -> None:
    def recorded(keep_fits: bool) -> list[tuple[Any, ...]]:
        seen: list[tuple[Any, ...]] = []
        original = TMLE.fit

        def fit(self: TMLE, data: Any, **kwargs: Any) -> Any:
            out = original(self, data, **kwargs)
            seen.append(tuple(out.single().fit_warnings))
            return out

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(TMLE, "fit", fit)
            _shown(_warning_frame(), keep_fits)
        return seen

    assert recorded(False) == recorded(True)
    assert any(recorded(True))


def test_without_kept_fits_every_read_of_fits_refuses() -> None:
    result = _run(_frame(), keep_fits=False)
    reads: list[Callable[[], Any]] = [
        lambda: result.fits["X1"],
        lambda: result.fits.get("X1"),
        lambda: "X1" in result.fits,
        lambda: list(result.fits),
        lambda: len(result.fits),
    ]
    for read in reads:
        with pytest.raises(CapabilityError, match="keep_fits=True") as raised:
            read()
        assert raised.value.refusal.reason is RefusalReason.API_REDIRECTION
    assert set(_run(_frame(), keep_fits=True).fits) == {"X1", "X2", "X3"}


class _Counting(LogisticRegression):
    calls = 0

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> Any:
        type(self).calls += 1
        return super().fit(X, y, sample_weight=sample_weight)


def _fold_policy_estimator() -> TMLE:
    estimator = _estimator(cross_fit=True)
    estimator.stratify_folds = "treatment"
    return estimator


def _bounded(frame: pd.DataFrame) -> tuple[float, float]:
    return float(frame["Y"].min()) - 1.0, float(frame["Y"].max()) + 1.0


# Each case reaches its own refusal: the type and a phrase only that refusal writes.
REFUSALS: dict[str, Callable[[], tuple[pd.DataFrame, dict[str, Any], type[Exception], str]]] = {
    "a modified fold policy": lambda: (
        _frame(),
        {"estimator": _fold_policy_estimator()},
        CapabilityError,
        "stratify_folds='none'",
    ),
    "a candidate with missing values": lambda: (
        _frame().assign(X1=lambda f: f["X1"].where(f.index != 3)),
        {},
        DataError,
        "has missing values",
    ),
    "an empty adjustment set": lambda: (
        _frame(),
        {"candidates": ("X1",), "covariates": ()},
        DataError,
        "empty adjustment set",
    ),
    "a cross-fitted outcome on an undeclared scale": lambda: (
        _frame(),
        {},
        CapabilityError,
        "Declare the known outcome support",
    ),
    "a withheld clustered status": lambda: (
        (frame := _frame()).assign(cid=np.arange(len(frame)) % 2),
        {"id": "cid", "estimator": _estimator(cross_fit=True, q_bounds=_bounded(frame))},
        CapabilityError,
        "fewer than 3 clusters",
    ),
}
# The derivative-array refusal (a declared propensity array cannot follow a changed
# treatment) is raised before the pre-pass, independent of keep_fits; its own test is
# tests/unit/test_ctmle_derivative_strategy.py::test_variable_importance_refuses_arrays.


@pytest.mark.parametrize("case", list(REFUSALS))
def test_every_refusal_still_comes_before_the_first_fit(case: str) -> None:
    frame, extra, kind, phrase = REFUSALS[case]()
    messages = []
    for keep_fits in (True, False):
        _Counting.calls = 0
        settings: dict[str, Any] = {
            "outcome": "Y",
            "candidates": ("X1", "X2"),
            "covariates": ("W",),
            "estimator": _estimator(cross_fit=True),
        }
        settings.update(extra)
        settings["estimator"].treatment_learner = _Counting(max_iter=1000)
        with pytest.raises(kind, match=re.escape(phrase)) as raised:
            variable_importance(frame, keep_fits=keep_fits, **settings)
        assert _Counting.calls == 0
        messages.append((type(raised.value), str(raised.value)))
    assert messages[0] == messages[1]


def test_muted_records_and_does_not_show() -> None:
    from cleverly import _warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with _warnings.recording() as recorder:
            with _warnings.muted():
                _warnings.warn("inside", UserWarning)
                with _warnings.muted():
                    pass
                _warnings.warn("still inside", UserWarning)
            _warnings.warn("outside", UserWarning)
            with pytest.raises(RuntimeError), _warnings.muted():
                raise RuntimeError
            _warnings.warn("after an exception", UserWarning)
    assert [str(item.message) for item in caught] == ["outside", "after an exception"]
    assert [record.message for record in recorder] == [
        "inside",
        "still inside",
        "outside",
        "after an exception",
    ]


def _tracked(keep_fits: bool, keep: str | None = None) -> dict[str, list[Any]]:
    """Weak references to every prepared dataset and fit, and which were alive when each fit
    started and after the call returned. ``keep`` holds one kind alive on purpose."""
    module = importlib.import_module("cleverly.variable_importance")

    prepared: list[weakref.ref[Any]] = []
    fitted: list[weakref.ref[Any]] = []
    alive_at_fit: list[int] = []
    held: list[Any] = []
    original_prepare, original_fit = TMLE._prepare, TMLE.fit

    def prepare(self: TMLE, data: Any, **kwargs: Any) -> Any:
        out = original_prepare(self, data, **kwargs)
        prepared.append(weakref.ref(out))
        if keep == "pre-pass data" and len(prepared) <= 3:
            held.append(out)
        return out

    def fit(self: TMLE, data: Any, **kwargs: Any) -> Any:
        # Everything but the data handed to this fit must be gone already.
        alive_at_fit.append(
            sum(ref() is not None for ref in prepared if ref() is not data)
            + sum(ref() is not None for ref in fitted)
        )
        out = original_fit(self, data, **kwargs)
        fitted.append(weakref.ref(out))
        if keep == "previous fit":
            held[:] = [out]
        return out

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(TMLE, "_prepare", prepare)
        patch.setattr(TMLE, "fit", fit)
        result = _run(_frame(), keep_fits=keep_fits, estimator=_estimator())
    del module
    after = sum(ref() is not None for ref in prepared + fitted)
    del result
    return {"alive_at_fit": alive_at_fit, "after": [after]}


def test_without_kept_fits_nothing_outlives_its_turn() -> None:
    tracked = _tracked(keep_fits=False)
    assert tracked["alive_at_fit"] == [0, 0, 0]
    assert tracked["after"] == [0]


@pytest.mark.parametrize("keep", ["pre-pass data", "previous fit"])
def test_holding_a_dataset_or_a_fit_is_caught(keep: str) -> None:
    tracked = _tracked(keep_fits=False, keep=keep)
    assert tracked["alive_at_fit"] != [0, 0, 0]


def test_without_kept_fits_the_result_holds_little() -> None:
    frame = _frame(n=5_000)
    for index in range(5):
        frame[f"X{index + 4}"] = np.random.default_rng(index).binomial(1, 0.5, len(frame))
    candidates = tuple(f"X{index}" for index in range(1, 9))
    run = {"candidates": candidates, "estimator": _estimator()}
    _run(_frame(n=200), keep_fits=False)  # imports and first-call allocations

    def retained(keep_fits: bool) -> int:
        gc.collect()
        tracemalloc.start()
        try:
            before = tracemalloc.get_traced_memory()[0]
            result = _run(frame, keep_fits=keep_fits, **run)
            gc.collect()
            held = tracemalloc.get_traced_memory()[0] - before
        finally:
            tracemalloc.stop()
        curves = sum(np.asarray(entry.estimate.influence_curve).nbytes for entry in result)
        return held - curves

    dropped, kept = retained(False), retained(True)
    assert dropped < 1_000_000
    assert kept > 5 * dropped
