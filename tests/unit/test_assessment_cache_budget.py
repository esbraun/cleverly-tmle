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
import gc
import pickle
import re
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
    arguments = sweep_arguments(result)

    def call() -> Any:
        return _quiet(
            lambda: result.assess(
                include_refits=True, include_retargets=True, arguments=arguments, random_state=0
            )
        )

    call()
    return call


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


@pytest.mark.parametrize("kind", SUBSET)
def test_each_dropped_answer_is_computed_again_bit_for_bit(kind: str) -> None:
    result = KINDS[kind].build()
    call = _fill(result)
    cache = result.assessment_cache
    before = _snapshot(cache)
    evictable = {entry.key for entry in cache.info() if entry.evictable}
    assert evictable, "a sweep that drops nothing would pass on any recomputation"
    for entry in cache.info():
        if _operation(entry.key) == "sensitivity.elements":
            assert entry.evictable, entry.key
    cache.evict()
    assert not evictable & set(cache)
    with _filings() as filed:
        call()
    refiled = {key for key, _ in filed}
    # Every dropped answer was computed again, not served by something that survived.
    assert evictable <= refiled
    for key in evictable:
        assert pickle.dumps(cache[key]) == before[key], key
    assert set(cache) == set(before)


def test_the_comparison_sees_an_answer_that_changes_on_recomputation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation control for W1: a perturbed recomputation of a summary must be caught."""
    result = KINDS["ordinary"].build()
    call = _fill(result)
    cache = result.assessment_cache
    before = _snapshot(cache)
    cache.evict()
    original = DiagnosticsFacade.score_equations

    def perturbed(self: Any, **kwargs: Any) -> Any:
        report = original(self, **kwargs)
        key = next(key for key in self._result.assessment_cache if "score_equations" in key)
        self._result.assessment_cache[key] = ("perturbed", report)
        return report

    monkeypatch.setattr(DiagnosticsFacade, "score_equations", perturbed)
    call()
    changed = [key for key in before if pickle.dumps(cache[key]) != before[key]]
    assert any("score_equations" in key for key in changed)


def test_combined_reports_are_dropped_after_a_refitting_battery() -> None:
    """A refit inside ``assess`` pins itself, not the combined report that ran it.

    A pinned combined report would hold the omitted-confounder elements by reference, so
    dropping their own entries would free nothing.
    """
    result = KINDS["ordinary"].build()
    _fill(result)
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
    result = KINDS["ordinary"].build()
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


@pytest.mark.parametrize("kind", ["ordinary", "ltmle", "continuous_survival", "drtmle"])
def test_every_refit_answer_is_pinned(kind: str) -> None:
    result = KINDS[kind].build()
    _fill(result)
    cache = result.assessment_cache
    refits = [
        entry.key
        for entry in cache.info()
        if _operation(entry.key)
        in {
            "diagnostics.refute",
            "sensitivity.benchmark",
            "sensitivity.marginal_bounds",
            "sensitivity.simulated_confounding",
        }
        or (_operation(entry.key) == "diagnostics.truncation_curve" and kind != "ordinary")
    ]
    assert refits
    cache.evict()
    cache.max_bytes = 1
    assert set(refits) <= set(cache)


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
        counted = cache._size(key)
        before, _ = tracemalloc.get_traced_memory()
        del cache[key]
        gc.collect()
        after, _ = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return counted, before - after


def test_the_elements_are_counted_within_a_quarter_of_what_they_hold() -> None:
    counted, released = _released(lambda: _point_fit(n=20_000), "elements")
    assert released <= counted <= 1.25 * released


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
    assert counted < released


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
    result = KINDS["ordinary"].build()
    _fill(result)
    reachable: set[int] = set()
    stack: list[Any] = [
        getattr(result, name) for name in vars(result) if name != "assessment_cache"
    ]
    while stack:
        item = stack.pop()
        if id(item) in reachable or isinstance(item, (str, int, float, bytes, np.ndarray)):
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


def test_an_answer_above_the_whole_budget_is_not_kept() -> None:
    cache = AssessmentCache()
    cache.max_bytes = 1_000
    cache_module._file(cache, "big:{}", _Value(10_000), evictable=True)
    assert "big:{}" not in cache


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


def test_without_a_budget_nothing_is_sized_or_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Any] = []
    original = cache_module._payload_nbytes
    monkeypatch.setattr(
        cache_module, "_payload_nbytes", lambda value: calls.append(value) or original(value)
    )
    result = KINDS["ordinary"].build()
    with _filings() as filed:
        _fill(result)
    assert calls == []
    assert set(result.assessment_cache) == {key for key, _ in filed}


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
    other.evict()
    assert list(other) == ["pinned:{}"]
    assert len(cache) == 3


def test_a_saved_result_keeps_its_cache_policy(tmp_path: Path) -> None:
    result = KINDS["ordinary"].build()
    _fill(result)
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

    result = KINDS["ordinary"].build()
    _fill(result)
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
    ("sensitivity/_simulated_confounding_survival.py", "SeedSequence([root_seed"): "list seed",
    ("sensitivity/_simulated_confounding_survival.py", "default_rng(seed)"): "spawned int",
    ("sensitivity/simulated_confounding.py", "SeedSequence([root_seed"): "list seed",
    ("sensitivity/simulated_confounding.py", "default_rng(latent_seed)"): "spawned int",
    ("learners/crossfit.py", "RandomState(seed)"): "seed validated as an integer",
    ("learners/crossfit.py", "SeedSequence(self.random_state)"): "only when random_state is set",
    ("targeting_curves.py", "default_rng([0 if random_state"): "list seed",
    ("inference/bootstrap.py", "default_rng(draw.sequence)"): "spawned sequence",
    ("validation/_seeds.py", "SeedSequence(root)"): "validated root",
    ("validation/_seeds.py", "SeedSequence([root"): "list seed",
    ("validation/_continuous_survival_refute.py", "default_rng(seed)"): "resolved seed",
    ("validation/outcome_processes.py", "default_rng(seed)"): "child seed",
    ("validation/_point_generated.py", "default_rng(seed)"): "child seed",
    ("validation/_longitudinal_plasmode.py", "default_rng(seed)"): "child seed",
    ("validation/refute.py", "SeedSequence([root_seed"): "list seed",
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


@pytest.mark.parametrize("kind", ["ordinary", "ctmle_ordered", "ltmle", "continuous_survival"])
def test_no_answer_the_rule_allows_marks_and_the_refits_do(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str, bool, frozenset[str]]] = []
    original = cache_module._classify

    def record(result: Any, operation: str, args: Any, kwargs: Any, **options: Any) -> bool:
        allowed = cache_module._may_evict(result, operation, args, kwargs)
        if options["evictable"] is not None:
            allowed = options["evictable"]
        seen.append((operation, allowed, frozenset(options["marks"])))
        return original(result, operation, args, kwargs, **options)

    monkeypatch.setattr(cache_module, "_classify", record)
    _fill(KINDS[kind].build())
    allowed_but_marked = [(op, marks) for op, allowed, marks in seen if allowed and marks]
    assert allowed_but_marked == []
    refits = [marks for op, _, marks in seen if op == "diagnostics.refute"]
    assert refits
    if kind != "continuous_survival":
        # The survival engine fits its hazards outside ``fit_learner``, so its refutation
        # marks nothing; rule 1 pins it as a refit row (``test_every_refit_answer_is_pinned``).
        assert all(refits)


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
