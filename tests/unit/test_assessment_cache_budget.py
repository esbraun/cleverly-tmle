"""The assessment cache's byte budget drops only answers it can compute again exactly (X106 (l)).

A result keeps every assessment answer in :class:`~cleverly.assessment.AssessmentCache`. With a
budget, the cache drops answers that are summaries or retargets of stored artifacts, least
recently used first, and keeps every answer whose recomputation could differ: a refit, a fresh
seed, a learner fit or a parallel section (``_assessment_cache._classify``).

The witnesses below compare each recomputed answer with the dropped one bit for bit, and each
rule carries a mutation control that must make its witness fail.
"""

from __future__ import annotations

import copy
import functools
import gc
import pickle
import re
import sys
import tracemalloc
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cloudpickle
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression, LogisticRegression
from sklearn.model_selection import KFold

import cleverly
from cleverly import ATE, CausalStudy, ConditionalContrast, PointTreatment
from cleverly import _assessment_cache as cache_module
from cleverly.assessment import (
    SENSITIVITY_ROUTES,
    AssessmentCache,
    DiagnosticsFacade,
)
from cleverly.bases import Linear, Modifiers
from cleverly.datasets import make_linear_ate
from cleverly.exceptions import DataError
from cleverly.utils import random as random_module
from tests.unit._capability_sweep_support import KINDS, sweep_arguments

SRC = Path(cleverly.__file__).resolve().parent

#: One kind per family and per entry type that holds rows: point, several arms, a known
#: mechanism (derived risk ratio), DR-TMLE, ordered C-TMLE (nuisance arrays), LTMLE, hazard
#: LTMLE, the structural g-formula, continuous-time survival, an incremental fit and adaptive TMLE.
SUBSET = (
    "ordinary",
    "multi_arm",
    "known_score",
    "drtmle",
    "ctmle_ordered",
    "ltmle",
    "ltmle_hazard",
    "gformula",
    "continuous_survival",
    "incremental",
    "atmle",
)

#: A kind whose full battery fills in seconds, for the tests that need a filled cache but not
#: a particular family (the ordinary kind's battery takes about two minutes).
_QUICK = "drtmle"

#: The answers that hold rows, which a budget exists to drop.
ROW_SCALING = (
    "sensitivity.elements",
    "sensitivity.derived_risk_ratio",
    "diagnostics.nuisance_models",
)


def _quiet(build: Callable[[], Any]) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build()


def _fill(result: Any) -> Callable[[], Any]:
    call = _refill(result)
    call()
    return call


def _refill(result: Any) -> Callable[[], Any]:
    """The call that ran :func:`_fill`, without running it."""
    arguments = sweep_arguments(result)
    return lambda: _quiet(
        lambda: result.assess(
            include_refits=True, include_retargets=True, arguments=arguments, random_state=0
        )
    )


@functools.cache
def _filled_bytes(kind: str) -> bytes:
    result = KINDS[kind].build()
    _fill(result)
    return cloudpickle.dumps(result)


def _filled(kind: str) -> tuple[Any, Callable[[], Any]]:
    """A fresh copy of a filled result of ``kind``, built and filled once per worker.

    Each test gets its own copy, so a budget or an eviction in one test reaches no other.
    The copy comes back through the cache's own persistence path (W7).
    """
    result = _quiet(lambda: pickle.loads(_filled_bytes(kind)))
    return result, _refill(result)


def _snapshot(cache: dict[str, Any]) -> dict[str, bytes]:
    return {key: pickle.dumps(value) for key, value in cache.items()}


@contextmanager
def _filings() -> Iterator[list[tuple[str, bool]]]:
    """Record each key the cache files, with whether it was filed evictable."""
    seen: list[tuple[str, bool]] = []
    original = cache_module._file

    def record(cache: Any, key: str, value: Any, *, evictable: bool) -> None:
        seen.append((key, evictable))
        original(cache, key, value, evictable=evictable)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cache_module, "_file", record)
        import cleverly.assessment as assessment_module

        patch.setattr(assessment_module, "_file", record)
        yield seen


def _operation(key: str) -> str:
    return key.split(":", 1)[0]


def _point_fit(n: int = 400, *, random_state: int | None = 1, **learners: Any) -> Any:
    frame, _ = make_linear_ate(n=n, seed=7)
    settings = {
        "outcome_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=1000),
        **learners,
    }
    study = CausalStudy(
        frame,
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4")),
    )
    return _quiet(
        lambda: study.identify(ATE()).estimate(
            cross_fit=False, random_state=random_state, **settings
        )
    )


# --------------------------------------------------------------------------------- W1


def _assert_recomputes_bit_for_bit(kind: str, result: Any, call: Callable[[], Any]) -> None:
    cache = result.assessment_cache
    before = _snapshot(cache)
    evictable = {entry.key for entry in cache.info() if entry.evictable}
    assert evictable, "a sweep that drops nothing would pass on any recomputation"
    for entry in cache.info():
        operation = _operation(entry.key)
        if operation == "sensitivity.derived_risk_ratio":
            assert entry.evictable is _DERIVED_RATIO_EVICTABLE[kind], entry.key
        elif operation in ROW_SCALING:
            assert entry.evictable, entry.key
    cache.evict()
    assert not evictable & set(cache)
    with _filings() as filed:
        call()
    refiled = {key for key, _ in filed}
    # Every dropped answer was computed again, not served by something that survived.
    assert evictable <= refiled
    changed = [key for key in sorted(evictable) if pickle.dumps(cache[key]) != before[key]]
    assert not changed, f"changed on recomputation: {changed}"
    assert set(cache) == set(before)


@pytest.mark.parametrize("kind", SUBSET)
def test_each_dropped_answer_is_computed_again_bit_for_bit(kind: str) -> None:
    # A fresh fill, not a copy from ``_filled``: pickle bytes record which objects an answer
    # shares, and a restored graph shares differently from a computed one.
    result = KINDS[kind].build()
    _assert_recomputes_bit_for_bit(kind, result, _fill(result))


def test_the_comparison_sees_an_answer_that_changes_on_recomputation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation control for W1: a perturbed recomputation of a summary must fail the witness."""
    import cleverly.assessment as assessment_module

    result = KINDS[_QUICK].build()
    call = _fill(result)
    original = assessment_module._cached

    def perturbed(result: Any, name: str, args: Any, kwargs: Any, compute: Any, **o: Any) -> Any:
        if name == "diagnostics.score_equations":
            inner = compute

            def compute() -> Any:
                return ("perturbed", inner())

        return original(result, name, args, kwargs, compute, **o)

    # The fill is already filed, so only the recomputation after the drop is perturbed.
    monkeypatch.setattr(assessment_module, "_cached", perturbed)
    with pytest.raises(AssertionError, match="score_equations"):
        _assert_recomputes_bit_for_bit(_QUICK, result, call)


def test_combined_reports_are_dropped_after_a_refitting_battery() -> None:
    """A refit inside ``assess`` pins itself, not the combined report that ran it.

    A pinned combined report would hold the omitted-confounder elements by reference, so
    dropping their own entries would free nothing.
    """
    result, _ = _filled(_QUICK)
    cache = result.assessment_cache
    flags = {_operation(entry.key): entry.evictable for entry in cache.info()}
    assert flags["diagnostics.refute"] is False
    assert flags["diagnostics.run_all"] is True
    assert flags["sensitivity.run_all"] is True


def test_marks_credited_to_every_open_computation_would_pin_the_combined_reports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation control: a mark that reached the enclosing computation pins ``run_all``."""
    frames: list[set[str]] = []

    def mark_all(kind: str) -> None:
        for frame in random_module._FRAMES.get():
            frame.add(kind)
        frames.append(set())

    monkeypatch.setattr(random_module, "_mark", mark_all)
    result = KINDS[_QUICK].build()
    _fill(result)
    flags = {_operation(entry.key): entry.evictable for entry in result.assessment_cache.info()}
    assert flags["diagnostics.run_all"] is False


def test_dropping_the_elements_releases_their_rows() -> None:
    result = _point_fit(n=20_000)
    tracemalloc.start()
    try:
        _quiet(lambda: result.sensitivity.elements())
        _quiet(lambda: result.sensitivity.run_all())
        gc.collect()
        held, _ = tracemalloc.get_traced_memory()
        freed = result.assessment_cache.evict()
        gc.collect()
        after, _ = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # Four arrays of n floats: the representer and three score components.
    assert held - after >= 4 * 8 * 20_000
    assert freed >= held - after


def test_an_effect_curve_files_nothing_in_its_own_cache() -> None:
    rng = np.random.default_rng(0)
    n = 300
    w, z = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
    treated = rng.binomial(1, 0.5, n)
    y = rng.binomial(1, 0.3 + 0.2 * treated * (w > 0) + 0.05 * z)
    frame = pd.DataFrame({"Y": y, "A": treated, "W": w, "Z": z})
    curve = _quiet(
        lambda: (
            CausalStudy(
                frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W", "Z"))
            )
            .identify(
                ConditionalContrast(Modifiers(["W"], domains={"W": (-1, 1)}), model=Linear(["W"]))
            )
            .estimate(
                outcome_learner=LogisticRegression(max_iter=1000),
                treatment_learner=LogisticRegression(max_iter=1000),
                cross_fit=False,
                random_state=0,
            )
            .effect_curve()
        )
    )
    assert isinstance(curve.assessment_cache, AssessmentCache)
    _quiet(lambda: curve.diagnostics.support())
    assert curve.assessment_cache == {}


# --------------------------------------------------------------------------------- W2


def test_unseeded_refits_survive_every_eviction() -> None:
    result = _point_fit(
        random_state=None,
        outcome_learner=RandomForestRegressor(n_estimators=5),
        treatment_learner=RandomForestClassifier(n_estimators=5),
    )
    refutation = _quiet(lambda: result.diagnostics.refute(n_replicates=1))
    cache = result.assessment_cache
    refute_keys = [key for key in cache if _operation(key) == "diagnostics.refute"]
    # The request's own key and its resolved-seed alias.
    assert len(refute_keys) == 2
    cache.evict()
    cache.max_bytes = 1
    assert set(refute_keys) <= set(cache)
    assert _quiet(lambda: result.diagnostics.refute(n_replicates=1)) is refutation


def test_an_unseeded_simulated_confounding_refit_survives_every_eviction() -> None:
    result = _point_fit(random_state=None)
    grid = sweep_arguments(result)["simulated_confounding"]["grid"]
    report = _quiet(lambda: result.sensitivity.simulated_confounding(grid=grid))
    cache = result.assessment_cache
    keys = [key for key in cache if _operation(key) == "sensitivity.simulated_confounding"]
    assert keys and not any(e.evictable for e in cache.info() if e.key in keys)
    cache.evict()
    cache.max_bytes = 1
    assert set(keys) <= set(cache)
    assert _quiet(lambda: result.sensitivity.simulated_confounding(grid=grid)) is report


@contextmanager
def _summary_that_draws_a_seed(result: Any) -> Iterator[None]:
    """``score_equations`` made to draw a fresh seed, as an unseeded refit does."""
    original = DiagnosticsFacade.score_equations

    def drawing(self: Any, **kwargs: Any) -> Any:
        def compute() -> Any:
            random_module.resolve_assessment_seed(self._result, None)
            return original(self, **kwargs)

        return cache_module._cached(self._result, "diagnostics.score_equations", (), {}, compute)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(DiagnosticsFacade, "score_equations", drawing)
        yield


def _drawn_summary_flag(result: Any) -> bool:
    with _summary_that_draws_a_seed(result):
        result.diagnostics.score_equations()
    # The drawing wrapper's own entry; the summary it wraps files a second, unmarked one.
    key = cache_module._cache_key("diagnostics.score_equations", (), {})
    (entry,) = [e for e in result.assessment_cache.info() if e.key == key]
    return entry.evictable


def test_a_summary_that_draws_fresh_entropy_is_pinned() -> None:
    result = _point_fit(random_state=None)
    assert result.estimator.random_state is None
    assert _drawn_summary_flag(result) is False
    # Without the draw the same operation may be dropped.
    plain = _point_fit(random_state=None)
    plain.diagnostics.score_equations()
    assert all(e.evictable for e in plain.assessment_cache.info())


def test_ignoring_the_marks_would_drop_the_drawn_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation control for rule 2: with marks discarded, the drawn summary reads evictable."""

    @contextmanager
    def discard() -> Iterator[set[str]]:
        yield set()

    monkeypatch.setattr(cache_module, "recording_marks", discard)
    assert _drawn_summary_flag(_point_fit(random_state=None)) is True


# --------------------------------------------------------------------------------- W3


_REFITS = {
    "diagnostics.refute",
    "sensitivity.benchmark",
    "sensitivity.marginal_bounds",
    "sensitivity.simulated_confounding",
}


def _assert_refits_pinned(kind: str, result: Any) -> None:
    cache = result.assessment_cache
    refits = [
        entry.key
        for entry in cache.info()
        if _operation(entry.key) in _REFITS
        or (_operation(entry.key) == "diagnostics.truncation_curve" and kind != "ordinary")
    ]
    assert refits
    cache.evict()
    cache.max_bytes = 1
    assert set(refits) <= set(cache)


# One kind per family; the point family is DR-TMLE, whose battery fills in seconds.
@pytest.mark.parametrize("kind", ["drtmle", "ltmle", "continuous_survival"])
def test_every_refit_answer_is_pinned(kind: str) -> None:
    _assert_refits_pinned(kind, _filled(kind)[0])


def test_a_rule_that_allows_every_answer_fails_the_refit_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation control for rule 1: the survival engine's refits make no mark, so only the rule
    pins them."""
    monkeypatch.setattr(cache_module, "_EVICTION_RULE", [lambda *args: True])
    result = KINDS["continuous_survival"].build()
    _fill(result)
    with pytest.raises(AssertionError):
        _assert_refits_pinned("continuous_survival", result)


def test_consistent_mechanism_bounds_are_pinned() -> None:
    """On a selecting C-TMLE fit the bound refits the treatment learner on every covariate."""
    result = KINDS["ctmle_ordered"].build()
    _quiet(lambda: result.sensitivity.omitted_confounding(assume_consistent_mechanism=True))
    flags = [e.evictable for e in result.assessment_cache.info() if "omitted_confounding" in e.key]
    assert flags and not any(flags)


def test_a_seeded_refit_can_still_recompute_differently(monkeypatch: pytest.MonkeyPatch) -> None:
    """Why refits are pinned whatever their seed, and the mutation control for rule 1.

    A splitter nested in a learner keeps ``random_state=None`` after the estimator seeds the
    learner, so a seeded refutation reads the global generator. Classified evictable, with the
    marks discarded, its recomputation after a drop differs.
    """
    learner = LassoCV(cv=KFold(3, shuffle=True))
    result = _point_fit(random_state=3, outcome_learner=learner)

    @contextmanager
    def discard() -> Iterator[set[str]]:
        yield set()

    monkeypatch.setattr(cache_module, "recording_marks", discard)
    monkeypatch.setattr(cache_module, "_may_evict", lambda *args: True)
    first = pickle.dumps(_quiet(lambda: result.diagnostics.refute(n_replicates=1)))
    result.assessment_cache.evict()
    np.random.seed(12345)  # noqa: NPY002 - the global state a nested splitter reads
    second = pickle.dumps(_quiet(lambda: result.diagnostics.refute(n_replicates=1)))
    assert first != second


# --------------------------------------------------------------------------------- W4


def _released(build: Callable[[], Any], key_part: str) -> tuple[int, int]:
    """The counted bytes of one fresh answer, and the bytes deleting it releases."""
    result = build()
    tracemalloc.start()
    try:
        build_answer = key_part
        _quiet(lambda: getattr(result.sensitivity, build_answer)())
        gc.collect()
        cache = result.assessment_cache
        (key,) = [key for key in cache if build_answer in key]
        # Deleting the entry also frees its key string, which the answer's count leaves out.
        counted = cache._size(key) + sys.getsizeof(key)
        before, _ = tracemalloc.get_traced_memory()
        del cache[key]
        gc.collect()
        after, _ = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return counted, before - after


#: Allocator slack on the lower bound: deleting an entry can also free or keep a few allocator
#: blocks that are not the answer's (27 bytes on a CI runner). One percent of the answer is far
#: above that noise and far below the 99% an arrayless count misses (the control below).
_LOWER_SLACK = 0.01


def _counted_within_bounds(counted: int, released: int) -> bool:
    return (1 - _LOWER_SLACK) * released <= counted <= 1.25 * released


def test_the_elements_are_counted_within_a_quarter_of_what_they_hold() -> None:
    counted, released = _released(lambda: _point_fit(n=20_000), "elements")
    assert _counted_within_bounds(counted, released), (counted, released)


def test_dropping_arrays_from_the_count_fails_the_lower_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation control for W4: an arrayless count falls below what deleting releases."""
    original = cache_module._payload_nbytes

    def arrayless(value: Any) -> int:
        stripped = {
            name: field for name, field in vars(value).items() if not isinstance(field, np.ndarray)
        }
        return original(stripped)

    monkeypatch.setattr(cache_module, "_payload_nbytes", arrayless)
    counted, released = _released(lambda: _point_fit(n=20_000), "elements")
    assert not _counted_within_bounds(counted, released)
    # Not a near miss: the arrays are almost all of what the entry holds.
    assert counted < 0.1 * released, (counted, released)


def test_the_count_of_a_filled_cache_is_at_most_twice_what_clearing_it_releases() -> None:
    """The over-count from arrays the answers share with the fit stays within a factor of two."""
    tracemalloc.start()
    try:
        result, _ = _filled(_QUICK)
        gc.collect()
        cache = result.assessment_cache
        counted = cache.nbytes
        before, _ = tracemalloc.get_traced_memory()
        cache.clear()
        gc.collect()
        after, _ = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert counted <= 2 * (before - after)


def test_the_count_stops_at_a_result_and_counts_a_shared_array() -> None:
    result = _point_fit()
    array = np.zeros(10_000)
    assert cache_module._payload_nbytes({"result": result}) < 10_000
    # An array the answer shares is counted although dropping the answer frees nothing.
    assert cache_module._payload_nbytes({"view": array[:5_000]}) >= array.nbytes


def test_without_the_stop_rule_a_result_is_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation control for the stop rule."""
    result = _point_fit(n=5_000)
    monkeypatch.setattr(cache_module, "_is_stop", lambda value: False)
    assert cache_module._payload_nbytes({"result": result}) > 5_000 * 8


def test_every_stop_object_of_an_answer_belongs_to_the_fit() -> None:
    """An answer that owned a fitted learner of its own would be under-counted."""
    _assert_stop_objects_belong_to_the_fit(_filled(_QUICK)[0])


def _assert_stop_objects_belong_to_the_fit(result: Any) -> None:
    reachable: set[int] = set()
    stack: list[Any] = [
        getattr(result, name) for name in vars(result) if name != "assessment_cache"
    ]
    while stack:
        item = stack.pop()
        if id(item) in reachable or isinstance(item, (str, int, float, bytes, np.ndarray)):
            continue
        if isinstance(item, AssessmentCache):
            # A facade's back reference leads to the result and its cache; the answers are
            # what is being checked, so they cannot vouch for themselves.
            continue
        reachable.add(id(item))
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple, set, frozenset)):
            stack.extend(item)
        elif hasattr(item, "__dict__") and not isinstance(item, type):
            stack.extend(vars(item).values())
    cache = result.assessment_cache
    for entry in cache.info():
        if not entry.evictable:
            continue
        seen: set[int] = set()
        stack = [dict.__getitem__(cache, entry.key)]
        while stack:
            item = stack.pop()
            if id(item) in seen or isinstance(item, (str, int, float, bytes, np.ndarray)):
                continue
            seen.add(id(item))
            if cache_module._is_stop(item) and not isinstance(item, type):
                if callable(getattr(type(item), "fit", None)) or hasattr(item, "assessment_cache"):
                    assert id(item) in reachable, (entry.key, type(item))
                continue
            if isinstance(item, dict):
                stack.extend(item.values())
            elif isinstance(item, (list, tuple, set, frozenset)):
                stack.extend(item)
            elif hasattr(item, "__dict__"):
                stack.extend(vars(item).values())


# --------------------------------------------------------------------------------- W5


class _Value:
    def __init__(self, size: int) -> None:
        self.payload = np.zeros(size, dtype=np.uint8)


def _budgeted(*sizes: int) -> AssessmentCache:
    cache = AssessmentCache()
    for index, size in enumerate(sizes):
        cache_module._file(cache, f"op{index}:{{}}", _Value(size), evictable=True)
    return cache


def test_the_budget_drops_the_least_recently_used_first() -> None:
    cache = _budgeted(10_000, 10_000, 10_000)
    cache_module._touch(cache, "op0:{}")
    cache.max_bytes = cache._size("op0:{}") * 2 + 200
    assert list(cache) == ["op0:{}", "op2:{}"]
    assert cache.nbytes - cache.pinned_nbytes <= cache.max_bytes


def test_each_filing_keeps_the_cache_within_its_budget() -> None:
    cache = AssessmentCache()
    cache.max_bytes = 25_000
    for index in range(6):
        cache_module._file(cache, f"op{index}:{{}}", _Value(10_000), evictable=True)
        assert cache.nbytes - cache.pinned_nbytes <= 25_000
    assert list(cache) == ["op4:{}", "op5:{}"]


def test_an_answer_above_the_whole_budget_is_dropped_alone() -> None:
    cache = _budgeted(1_000, 1_000)
    cache.max_bytes = 5_000
    cache_module._file(cache, "big:{}", _Value(10_000), evictable=True)
    assert list(cache) == ["op0:{}", "op1:{}"]


def test_a_hit_through_the_filing_path_makes_an_answer_recent() -> None:
    result = _point_fit()
    cache = result.assessment_cache
    calls: list[str] = []

    def ask(name: str) -> Any:
        def compute() -> Any:
            calls.append(name)
            return np.zeros(1_000)

        return cache_module._cached(result, name, (), {}, compute, evictable=True)

    for name in ("first", "second", "third"):
        ask(name)
    ask("first")
    assert calls == ["first", "second", "third"]
    sizes = sorted(cache._size(key) for key in cache)
    cache.max_bytes = sizes[0] + sizes[1] + 1
    assert sorted(_operation(key) for key in cache) == ["first", "third"]


def test_one_byte_below_an_answer_drops_it_and_its_size_keeps_it() -> None:
    cache = _budgeted(10_000)
    size = cache._size("op0:{}")
    cache.max_bytes = size
    assert "op0:{}" in cache
    cache.max_bytes = size - 1
    assert "op0:{}" not in cache


def test_pinned_answers_stay_above_the_budget() -> None:
    cache = AssessmentCache()
    cache_module._file(cache, "pinned:{}", _Value(10_000), evictable=False)
    cache_module._file(cache, "free:{}", _Value(10_000), evictable=True)
    cache.max_bytes = 1
    assert list(cache) == ["pinned:{}"]
    assert cache.pinned_nbytes == cache.nbytes > cache.max_bytes
    assert cache.evict() == 0


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "1"])
def test_the_budget_must_be_a_positive_integer(value: Any) -> None:
    with pytest.raises(DataError, match="max_bytes must be a positive integer"):
        AssessmentCache().max_bytes = value


# --------------------------------------------------------------------------------- W6
# With no budget nothing is sized: ``test_capability_row_sweep.py`` asserts it on every kind.


# --------------------------------------------------------------------------------- W7


def _state(cache: AssessmentCache) -> tuple[Any, ...]:
    return (
        list(cache),
        cache.max_bytes,
        sorted(entry.key for entry in cache.info() if entry.evictable),
    )


@pytest.mark.parametrize(
    "round_trip",
    [
        lambda c: pickle.loads(pickle.dumps(c)),
        lambda c: cloudpickle.loads(cloudpickle.dumps(c)),
        copy.copy,
        copy.deepcopy,
        lambda c: c.copy(),
    ],
    ids=["pickle", "cloudpickle", "copy", "deepcopy", "method"],
)
def test_persistence_and_copies_keep_the_budget_and_the_flags(round_trip: Any) -> None:
    cache = _budgeted(100, 100)
    cache_module._file(cache, "pinned:{}", _Value(100), evictable=False)
    cache.max_bytes = 10**6
    other = round_trip(cache)
    assert isinstance(other, AssessmentCache)
    assert _state(other) == _state(cache)
    state = _state(cache)
    other.evict()
    assert list(other) == ["pinned:{}"]
    assert len(cache) == 3
    assert _state(cache) == state
    other.max_bytes = None
    cache_module._file(other, "late:{}", _Value(100), evictable=True)
    assert _state(cache) == state


def test_a_saved_result_keeps_its_cache_policy(tmp_path: Path) -> None:
    result, _ = _filled(_QUICK)
    result.assessment_cache.max_bytes = 10**9
    path = tmp_path / "result.joblib"
    result.save(path)
    restored = _quiet(lambda: cleverly.load(path))
    assert _state(restored.assessment_cache) == _state(result.assessment_cache)


def test_writes_outside_the_filing_path_are_pinned() -> None:
    cache = _budgeted(100, 100)
    cache.update({"other:{}": _Value(100)})
    cache.setdefault("default:{}", _Value(100))
    cache["op0:{}"] = _Value(100)
    assert {e.key for e in cache.info() if e.evictable} == {"op1:{}"}
    del cache["op1:{}"]
    cache["op1:{}"] = _Value(100)
    assert not any(e.evictable for e in cache.info())
    cache.pop("other:{}")
    assert cache.nbytes == sum(cache._size(key) for key in cache)


def test_a_derived_result_starts_an_empty_cache_with_no_budget() -> None:
    import dataclasses

    result, _ = _filled(_QUICK)
    result.assessment_cache.max_bytes = 10**9
    derived = dataclasses.replace(result)
    assert isinstance(derived.assessment_cache, AssessmentCache)
    assert derived.assessment_cache == {} and derived.assessment_cache.max_bytes is None


# --------------------------------------------------------------------------------- W8


def test_requests_that_differ_in_one_argument_are_filed_apart() -> None:
    result = KINDS["ordinary"].build()
    _quiet(lambda: result.diagnostics.score_equations(tolerance=1e-3))
    _quiet(lambda: result.diagnostics.score_equations(tolerance=1e-4))
    _quiet(lambda: result.diagnostics.refute(n_replicates=1, random_state=1))
    _quiet(lambda: result.diagnostics.refute(n_replicates=1, random_state=2))
    operations = [_operation(key) for key in result.assessment_cache]
    assert operations.count("diagnostics.score_equations") == 2
    assert operations.count("diagnostics.refute") == 2


# --------------------------------------------------------------------------------- W9

#: Every call that can seed a generator under ``src/``, by file and line text, with why it
#: needs no mark or which function marks it. ``marks`` sites must call
#: ``note_fresh_entropy`` in the same function when their argument is ``None``.
_DRAWS = re.compile(r"(SeedSequence|default_rng|RandomState|check_random_state)\(")

_MARKS = {
    "utils/random.py",
    "survival/engine.py",
    "learners/crossfit.py",
    "longitudinal/estimator.py",
    "validation/simulation.py",
    "inference/bootstrap.py",
    "targeting_curves.py",
    "inference/multiplier.py",
}

#: Sites whose argument is never ``None`` there, or whose state is replaced before use.
_SAFE = {
    ("_curve_assessment.py", "default_rng(seed)"): "seed from _seed(), an int",
    ("survival/engine.py", "default_rng(seed)"): "seed resolved by _resolved_seed",
    ("survival/engine.py", "default_rng()"): "state set from the stored rng_state next line",
    (
        "utils/random.py",
        "twin = np.random.default_rng()",
    ): "state set from a recorded state next line (replay)",
    ("sensitivity/_simulated_confounding_survival.py", "SeedSequence([root_seed"): "list seed",
    ("sensitivity/_simulated_confounding_survival.py", "default_rng(seed)"): "spawned int",
    ("sensitivity/_simulated_confounding_continuous.py", "default_rng(seed)"): "coupling int",
    ("sensitivity/simulated_confounding.py", "SeedSequence([root_seed"): "list seed",
    ("sensitivity/simulated_confounding.py", "default_rng(latent_seed)"): "spawned int",
    ("learners/crossfit.py", "RandomState(seed)"): "seed validated as an integer",
    ("learners/crossfit.py", "SeedSequence(self.random_state)"): "only when random_state is set",
    ("targeting_curves.py", "default_rng([0 if random_state"): "list seed",
    ("inference/bootstrap.py", "default_rng(draw.sequence)"): "spawned sequence",
    ("survival/engine.py", "default_rng(sequence)"): "a bootstrap draw's spawned sequence",
    ("validation/_seeds.py", "SeedSequence(root)"): "validated root",
    ("validation/_seeds.py", "SeedSequence([root"): "list seed",
    ("validation/_continuous_survival_refute.py", "default_rng(seed)"): "resolved seed",
    ("validation/outcome_processes.py", "default_rng(seed)"): "child seed",
    ("validation/_point_generated.py", "default_rng(seed)"): "child seed",
    ("validation/_sampled_refute.py", "default_rng(seed)"): "the sample's child seed",
    ("validation/_longitudinal_plasmode.py", "default_rng(seed)"): "child seed",
    ("validation/refute.py", "SeedSequence([root_seed"): "list seed",
    ("validation/_planned_refits.py", "np.random.default_rng(root),"): (
        "refute's resolved root, replayed for the banded preflight"
    ),
    ("validation/_planned_refits.py", "np.random.default_rng(0), 0"): (
        "constant: the noise frame's columns only, for counting"
    ),
    ("validation/refute.py", "default_rng(seed)"): "child seed from the resolved root",
    ("validation/refute.py", "np.random.default_rng(seed), observed"): "child seed",
    ("longitudinal/structural.py", "SeedSequence([int(random_state)"): "list seed",
    ("longitudinal/frequency.py", "default_rng(0)"): "constant",
    ("longitudinal/estimator.py", "SeedSequence([entropy"): "list seed",
    ("interventions/_rule_learners.py", "SeedSequence([int(seed)"): "list seed",
    ("estimators/ep.py", "SeedSequence(entropy)"): "list seed",
    ("estimators/atmle.py", "default_rng(random_state)"): "seed resolved by its only caller",
}


def _draw_sites() -> list[tuple[str, int, str]]:
    sites = []
    for path in sorted(SRC.rglob("*.py")):
        relative = path.relative_to(SRC).as_posix()
        for number, line in enumerate(path.read_text().splitlines(), 1):
            text = line.strip()
            if not _DRAWS.search(text) or text.startswith(("#", ">>>", "...")):
                continue
            if "``" in text or relative.startswith("datasets/"):
                # Prose, and the dataset generators, which no assessment calls.
                continue
            sites.append((relative, number, text))
    return sites


def test_every_generator_seeding_site_is_classified() -> None:
    unclassified = []
    for relative, number, text in _draw_sites():
        if any(relative == file and fragment in text for (file, fragment) in _SAFE):
            continue
        if relative in _MARKS:
            source = (SRC / relative).read_text().splitlines()
            window = "\n".join(source[max(0, number - 6) : number])
            if "note_fresh_entropy()" in window:
                continue
        unclassified.append(f"{relative}:{number}: {text}")
    assert unclassified == []


def test_removing_a_marker_fails_the_site_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation control for W9 (a)."""
    original = Path.read_text

    def unmarked(self: Path, *args: Any, **kwargs: Any) -> str:
        text = original(self, *args, **kwargs)
        if self.name == "multiplier.py":
            return text.replace("note_fresh_entropy()", "pass")
        return text

    monkeypatch.setattr(Path, "read_text", unmarked)
    with pytest.raises(AssertionError):
        test_every_generator_seeding_site_is_classified()


def test_a_refit_marks_and_no_answer_the_rule_allows_does(monkeypatch: pytest.MonkeyPatch) -> None:
    """The marks see a refit; the sweep checks that no allowed answer marks on every kind."""
    seen: list[tuple[str, bool, frozenset[str]]] = []
    original = cache_module._classify

    def record(result: Any, operation: str, args: Any, kwargs: Any, **options: Any) -> bool:
        allowed = cache_module._may_evict(result, operation, args, kwargs)
        if options["evictable"] is not None:
            allowed = options["evictable"]
        seen.append((operation, allowed, frozenset(options["marks"])))
        return original(result, operation, args, kwargs, **options)

    monkeypatch.setattr(cache_module, "_classify", record)
    _fill(KINDS[_QUICK].build())
    allowed_but_marked = [(op, marks) for op, allowed, marks in seen if allowed and marks]
    assert allowed_but_marked == []
    refits = [marks for op, _, marks in seen if op == "diagnostics.refute"]
    assert refits and all(refits)


def test_every_operation_is_classified() -> None:
    """Every operation name that reaches the cache is one the rule knows how to read."""
    names = {f"sensitivity.{name}" for name in SENSITIVITY_ROUTES}
    names |= {
        f"diagnostics.{row}"
        for row in (
            "support",
            "nuisance_models",
            "score_equations",
            "corrections",
            "truncation_curve",
            "refute",
            "calibration",
            "risk_comparison",
            "stability",
            "modifier_stability",
            "targeting_curve",
        )
    }
    names |= {
        "validate",
        "diagnostics.influence",
        "diagnostics.run_all",
        "sensitivity.run_all",
        "sensitivity.derived_risk_ratio",
    }
    literal = re.compile(r'_cached\(\s*[\w.]+,\s*"([^"]+)"')
    found = set()
    for path in SRC.rglob("*.py"):
        found |= set(literal.findall(path.read_text()))
    assert found <= names, found - names


#: Files that fit a model by calling ``.fit(`` directly instead of through ``fit_learner``, with
#: the number of such calls. Each is reached at fit time or from a refit row, which rule 1 pins,
#: so the missing learner-fit mark cannot leave an evictable answer that refits. A new direct
#: call changes a count and must be classified here (design review finding 26).
_DIRECT_FITS = {
    "_curve_assessment.py": (1, "effect-modification refit rows; the curve cache is unused"),
    "estimators/_working_basis.py": (1, "fit time"),
    "estimators/atmle.py": (1, "fit time"),
    "estimators/ep.py": (2, "fit time"),
    "estimators/tmle.py": (1, "fit time"),
    "interventions/tuning.py": (2, "fit time"),
    "learners/super_learner.py": (1, "inside fit_learner's own learner"),
    "longitudinal/drift.py": (8, "fit time"),
    "longitudinal/estimator.py": (10, "fit time and the refit truncation curve"),
    "longitudinal/gformula.py": (3, "fit time"),
    "longitudinal/policy_variance.py": (1, "fit time"),
    "longitudinal/structural.py": (3, "fit time and refit rows"),
    "sensitivity/_simulated_confounding_longitudinal.py": (1, "simulated_confounding, a refit"),
    "sensitivity/_simulated_confounding_survival.py": (2, "simulated_confounding, a refit"),
    "sensitivity/simulated_confounding.py": (3, "simulated_confounding, a refit"),
    "study.py": (5, "fit time"),
    "survival/cox.py": (2, "fit time and the refit truncation curve"),
    "survival/engine.py": (3, "fit time and the refit truncation curve"),
    "validation/_band_placebo.py": (1, "refute, a refit"),
    "validation/_longitudinal_plasmode.py": (6, "refute, a refit"),
    "validation/_planned_refits.py": (
        1,
        "planned_refits: the Markov kernel refute's placebo check reads",
    ),
    "validation/_point_generated.py": (3, "refute, a refit"),
    "validation/refute.py": (5, "refute, a refit"),
    "validation/simulation.py": (1, "simulation studies, not an assessment"),
    "variable_importance.py": (1, "fit time"),
}

_FIT_CALL = re.compile(r"\.fit\(")


def test_every_direct_fit_is_classified() -> None:
    counts: dict[str, int] = {}
    for path in sorted(SRC.rglob("*.py")):
        relative = path.relative_to(SRC).as_posix()
        if relative == "learners/_fitting.py":
            continue
        for line in path.read_text().splitlines():
            text = line.strip()
            if not _FIT_CALL.search(text) or text.startswith(("#", ">>>", "...")):
                continue
            if "``" in text or re.search(r"\"[^\"]*\.fit\(", text):
                continue
            counts[relative] = counts.get(relative, 0) + 1
    assert counts == {name: count for name, (count, _) in _DIRECT_FITS.items()}


def test_a_reused_address_cannot_pass_for_the_filed_answer() -> None:
    cache = _budgeted(100)
    cache.pop("op0:{}")
    cache.update({"op0:{}": _Value(100)})
    assert not any(entry.evictable for entry in cache.info())


def test_a_member_that_raises_leaves_no_marks_frame_behind(monkeypatch: pytest.MonkeyPatch) -> None:
    """The frame stack unwinds on an exception, so later filings are classified on their own."""
    result = KINDS["ordinary"].build()

    def raising() -> Any:
        random_module.note_learner_fit()
        raise RuntimeError("member failed")

    with pytest.raises(RuntimeError):
        cache_module._cached(result, "diagnostics.support", (), {}, raising)
    assert random_module._FRAMES.get() == ()
    _quiet(lambda: result.diagnostics.support())
    (entry,) = [e for e in result.assessment_cache.info() if "support" in e.key]
    assert entry.evictable


#: Whether each fixture's derived risk ratio may be dropped, stated per kind rather than read from
#: the predicate ``sensitivity/_derived.py`` uses. A guarded DR-TMLE retarget refits the reduced
#: regressions; the other two retarget from saved nuisances.
_DERIVED_RATIO_EVICTABLE = {
    "known_score": True,
    "or": True,
    "drtmle": False,
    "drtmle_companion": False,
}


@pytest.mark.parametrize("kind", sorted(_DERIVED_RATIO_EVICTABLE))
def test_a_derived_ratio_is_kept_exactly_where_its_retarget_fits_learners(kind: str) -> None:
    result, _ = _filled(kind)
    flags = [
        e.evictable
        for e in result.assessment_cache.info()
        if _operation(e.key) == "sensitivity.derived_risk_ratio"
    ]
    assert flags
    assert flags == [_DERIVED_RATIO_EVICTABLE[kind]] * len(flags)


@contextmanager
def _classified() -> Iterator[list[tuple[str, frozenset[str], bool]]]:
    """Record each answer the cache classifies: its operation, its marks and the outcome."""
    seen: list[tuple[str, frozenset[str], bool]] = []
    original = cache_module._classify

    def record(result: Any, operation: str, args: Any, kwargs: Any, **options: Any) -> bool:
        outcome = original(result, operation, args, kwargs, **options)
        seen.append((operation, frozenset(options["marks"]), outcome))
        return outcome

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cache_module, "_classify", record)
        yield seen


def _survives_every_drop(result: Any, operation: str) -> None:
    cache = result.assessment_cache
    keys = [key for key in cache if _operation(key) == operation]
    assert keys
    cache.evict()
    cache.max_bytes = 1
    assert all(key in cache for key in keys)


def test_a_guarded_drtmle_derived_ratio_fits_learners_and_is_pinned() -> None:
    """The pinned outcome above agrees with the marks: the retarget really fits learners."""
    with _classified() as seen:
        _fill(KINDS["drtmle"].build())
    ratios = [(marks, outcome) for op, marks, outcome in seen if "derived_risk_ratio" in op]
    assert ratios
    assert all("learner_fit" in marks and outcome is False for marks, outcome in ratios)


def test_an_adaptive_iptw_derived_ratio_is_pinned() -> None:
    from tests.unit._capability_sweep_support import fit_iptw_known

    result = _quiet(fit_iptw_known)
    with _classified() as seen:
        _quiet(lambda: result.sensitivity.evalue("ate"))
    ratios = [(marks, outcome) for op, marks, outcome in seen if "derived_risk_ratio" in op]
    assert ratios and all("learner_fit" in m and o is False for m, o in ratios)
    _survives_every_drop(result, "sensitivity.derived_risk_ratio")


def test_an_evalue_on_the_missing_at_random_scale_is_pinned() -> None:
    """The ``response_sd`` E-value refits the outcome scale, so its answer is kept."""
    from tests.unit.test_evalue_part_g import missing_fit, missing_frame

    frame, _, _ = missing_frame(n=600)
    result = _quiet(lambda: missing_fit(frame))
    with _classified() as seen:
        _quiet(lambda: result.sensitivity.evalue("ate"))
    assert [(op, o) for op, _, o in seen] == [("sensitivity.evalue", False)]
    _survives_every_drop(result, "sensitivity.evalue")


def test_a_consistent_mechanism_request_still_runs_without_include_refits() -> None:
    """The cache's refit classification leaves the capability row, and so the report, alone."""
    result = KINDS["ctmle_ordered"].build()
    report = _quiet(
        lambda: result.sensitivity.run_all(
            arguments={"omitted_confounding": {"assume_consistent_mechanism": True}}
        )
    )
    (item,) = [item for item in report.items if item.name == "omitted_confounding"]
    assert item.status.name not in ("DEFERRED", "SKIPPED"), item


def test_no_thread_or_process_pool_starts_outside_map_parallel() -> None:
    pools = re.compile(r"threading\.Thread\(|ThreadPoolExecutor\(|ProcessPoolExecutor\(")
    found = [
        path.relative_to(SRC).as_posix()
        for path in SRC.rglob("*.py")
        if pools.search(path.read_text())
    ]
    assert found == []


def test_a_budget_drops_a_primary_answer_and_its_alias() -> None:
    """An alias never keeps a report the budget dropped (design review finding 23)."""
    cache = AssessmentCache()
    cache_module._file(cache, "op:primary", _Value(10_000), evictable=True)
    cache_module._file(cache, "op:alias", cache["op:primary"], evictable=True)
    cache.max_bytes = 1
    assert list(cache) == []


def test_an_alias_of_an_answer_the_budget_refused_is_dropped_too() -> None:
    result = KINDS[_QUICK].build()
    result.assessment_cache.max_bytes = 1
    _quiet(lambda: result.sensitivity.evalue())
    assert not any(_operation(key) == "sensitivity.evalue" for key in result.assessment_cache)


def test_a_restored_cache_pins_a_key_overwritten_before_any_read() -> None:
    cache = _budgeted(100)
    restored = pickle.loads(pickle.dumps(cache))
    restored["op0:{}"] = _Value(100)
    assert not any(entry.evictable for entry in restored.info())


def test_a_consistent_mechanism_report_matches_a_plain_mapping() -> None:
    """The budgeted cache files and returns the report a plain mapping would (finding 31)."""
    arguments = {"omitted_confounding": {"assume_consistent_mechanism": True}}
    budgeted = KINDS["ctmle_ordered"].build()
    plain = KINDS["ctmle_ordered"].build()
    object.__setattr__(plain, "assessment_cache", {})
    first = _quiet(lambda: budgeted.sensitivity.run_all(arguments=arguments))
    second = _quiet(lambda: plain.sensitivity.run_all(arguments=arguments))
    assert pickle.dumps(first) == pickle.dumps(second)
    assert list(budgeted.assessment_cache) == list(plain.assessment_cache)


def test_an_answer_owning_a_learner_of_its_own_fails_the_stop_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation control for W4c: a freshly fitted learner inside an answer is caught."""
    original = DiagnosticsFacade.support

    def owning(self: Any) -> Any:
        report = original(self)
        key = next(key for key in self._result.assessment_cache if "support" in key)
        learner = LinearRegression().fit(np.zeros((3, 1)), np.zeros(3))
        self._result.assessment_cache[key] = (report, learner)
        cache_module._file(self._result.assessment_cache, key, (report, learner), evictable=True)
        return report

    monkeypatch.setattr(DiagnosticsFacade, "support", owning)
    result = KINDS[_QUICK].build()
    _fill(result)
    assert any("support" in e.key and e.evictable for e in result.assessment_cache.info())
    with pytest.raises(AssertionError):
        _assert_stop_objects_belong_to_the_fit(result)
