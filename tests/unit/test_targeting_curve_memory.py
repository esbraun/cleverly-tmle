"""Targeting-curve half-sample draws in bounded blocks, and their memory preflight.

The draws are made and reduced one block at a time (X106 part B).  Every draw must equal the
draw the earlier whole-matrix algorithm made, whatever the block size: the oracle below is
that algorithm, kept here verbatim except that its key line calls the module's ``_keys``
helper, so a patched key draw reaches both.  The preflight's byte model must bound the
kernel's ``tracemalloc`` peak from above, within its declared factor, at both ends of the
block size, and refuse exactly where its prediction crosses the budget.
"""

from __future__ import annotations

import itertools
import tracemalloc
import warnings
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from cleverly import targeting_curves as tc
from cleverly.exceptions import DataError, MethodConfigurationError, RefusalReason
from cleverly.utils import memory

GROUP_EDGES = (0.2, 0.4, 0.6, 0.8, 1.0)
FORTY = tuple(round(0.025 * k, 3) for k in range(1, 41))
MUTATIONS: tuple[dict[str, Any], ...] = (
    {},
    {"cut": "fixed"},
    {"ranks": "fixed"},
    {"tie_rule": "at_or_above"},
    {"scores": "one_step"},
    {"paired": False},
)


# ------------------------------------------------------------------ the oracle


def _oracle_masks(units: Any, n_draws: int, rng: np.random.Generator) -> tuple[Any, Any]:
    labels, inverse = np.unique(units, return_inverse=True)
    count = labels.size
    half = count // 2
    keys = tc._keys(rng, (n_draws, count))
    chosen = np.argsort(keys, axis=1)[:, :half]
    mask = np.zeros((n_draws, count))
    np.put_along_axis(mask, chosen, 1.0, axis=1)
    return inverse.astype(np.int64), mask


def _oracle(
    evaluation: tc.TargetingEvaluation,
    *,
    n_half_samples: int,
    random_state: int | None,
    scores: str = "targeted",
    cut: str = "quantile",
    ranks: str = "recompute",
    tie_rule: Any = "average",
    paired: bool = True,
    masks: Any = None,
    chunk: int | None = None,
) -> dict[str, dict[tuple[str, float], Any]]:
    """``half_sample_draws`` as it was before the blocks (cleverly-tmle ``ace81996``)."""
    gamma = evaluation.targeted if scores == "targeted" else evaluation.one_step
    mass = evaluation.mass
    rng = np.random.default_rng(random_state)
    inverse = np.unique(evaluation.units, return_inverse=True)[1].astype(np.int64)
    shipped = cut == "quantile" and ranks == "recompute" and tie_rule == "average"
    if masks is None:
        _, unit_masks = _oracle_masks(evaluation.units, n_half_samples, rng)
    else:
        unit_masks = np.asarray(masks, dtype=float)
    out: dict[str, dict[tuple[str, float], Any]] = {}
    for index, (name, _score) in enumerate(evaluation.scores.items()):
        layout = evaluation.layout(name)
        order = layout.order
        draws_masks = unit_masks
        if not paired and index > 0 and masks is None:
            stream = np.random.default_rng([0 if random_state is None else random_state, index])
            _, draws_masks = _oracle_masks(evaluation.units, n_half_samples, stream)
        full_m = tc._block_sums(mass[order], layout.starts)
        fixed_shares = None
        fixed_weights = None
        if cut == "fixed":
            b = np.cumsum(full_m)
            fixed_shares = {
                q: tc._shares(full_m, b - full_m, b[-1:], q, tie_rule)
                for q in sorted({*evaluation.grid, *evaluation.edges})
            }
        if ranks == "fixed":
            fixed_weights = tc.block_weights(
                full_m, evaluation.grid, evaluation.edges, tie_rule=tie_rule
            )
        size = chunk or max(1, int(4_000_000 // max(1, gamma.size)))
        pieces: dict[tuple[str, float], list[Any]] = {}
        for start in range(0, draws_masks.shape[0], size):
            row_mask = draws_masks[start : start + size][:, inverse]
            weighted = row_mask[:, order] * mass[order]
            m = tc._block_sums(weighted, layout.starts)
            sums = tc._block_sums(weighted * gamma[order], layout.starts)
            total = m.sum(axis=1)
            if shipped:
                for key, values in tc._shipped_draws(
                    m, sums, evaluation.grid, evaluation.edges
                ).items():
                    pieces.setdefault(key, []).append(values)
                continue
            weights = (
                fixed_weights
                if fixed_weights is not None
                else tc.block_weights(
                    m, evaluation.grid, evaluation.edges, tie_rule=tie_rule, shares=fixed_shares
                )
            )
            for key, c in weights.items():
                pieces.setdefault(key, []).append(np.sum(c * sums, axis=-1) / total)
        out[name] = {key: np.concatenate(parts) for key, parts in pieces.items()}
    return out


# ------------------------------------------------------------------ fixtures


def _evaluation(
    n: int,
    *,
    seed: int = 0,
    ties: bool = False,
    clusters: int | None = None,
    weighted: bool = False,
    two: bool = False,
    grid: tuple[float, ...] = tc.DEFAULT_GRID,
    edges: tuple[float, ...] = GROUP_EDGES,
    blocks: int | None = None,
) -> tc.TargetingEvaluation:
    rng = np.random.default_rng(seed)
    score = rng.normal(size=n)
    if ties:
        score = np.round(score, 1)
    if blocks is not None:
        score = rng.integers(0, blocks, n).astype(float)
    scores = {"S": score}
    if two:
        scores["T"] = score + rng.normal(size=n)
    units = np.arange(n) if clusters is None else rng.integers(0, clusters, n)
    mass = rng.exponential(size=n) if weighted else np.ones(n)
    return tc.TargetingEvaluation(
        scores=scores,
        targeted=rng.normal(size=n),
        one_step=rng.normal(size=n),
        mass=mass / mass.mean(),
        units=units,
        grid=grid,
        edges=edges,
    )


def _assert_same(one: dict[str, Any], two: dict[str, Any]) -> None:
    assert list(one) == list(two)
    for name in one:
        assert list(one[name]) == list(two[name])
        for key in one[name]:
            np.testing.assert_array_equal(one[name][key], two[name][key], err_msg=str(key))


def _matrix() -> Iterator[tuple[str, tc.TargetingEvaluation, dict[str, Any]]]:
    shapes = {
        "rows": {},
        "ties": {"ties": True},
        "clusters": {"clusters": 37},
        "weights": {"weighted": True},
        "two scores": {"two": True},
        "two tied clustered weighted": {
            "two": True,
            "ties": True,
            "clusters": 41,
            "weighted": True,
        },
    }
    for (label, shape), mutation in itertools.product(shapes.items(), MUTATIONS):
        yield label, _evaluation(301, seed=len(label), **shape), mutation


MATRIX = list(_matrix())


def _quiet(function: Any, *args: Any, **kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(all="ignore"):
            return function(*args, **kwargs)


# ------------------------------------------------------------------ B1: bitwise blocks


@pytest.mark.parametrize("chunk", [None, 1, 7, 61])
@pytest.mark.parametrize(("label", "evaluation", "mutation"), MATRIX, ids=lambda v: str(v))
def test_blocked_draws_equal_the_whole_matrix_draws(
    label: str, evaluation: tc.TargetingEvaluation, mutation: dict[str, Any], chunk: int | None
) -> None:
    """Every draw, every route, every block size: the oracle's draws exactly (D = 61, prime)."""
    del label
    expected = _quiet(_oracle, evaluation, n_half_samples=61, random_state=5, **mutation)
    got = _quiet(
        tc.half_sample_draws, evaluation, n_half_samples=61, random_state=5, chunk=chunk, **mutation
    )
    _assert_same(expected, got)


@pytest.mark.parametrize(("n", "draws"), [(1_000, 40), (10_000, 20), (100_000, 4)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_blocked_draws_equal_the_whole_matrix_draws_at_larger_sizes(
    n: int, draws: int, seed: int
) -> None:
    evaluation = _evaluation(n, seed=seed, two=True)
    for mutation in ({}, {"paired": False}):
        expected = _oracle(evaluation, n_half_samples=draws, random_state=seed, **mutation)
        for chunk in (None, 1, 3):
            got = tc.half_sample_draws(
                evaluation, n_half_samples=draws, random_state=seed, chunk=chunk, **mutation
            )
            _assert_same(expected, got)


def test_an_unseeded_call_keeps_the_seeded_unpaired_stream() -> None:
    """Under ``random_state=None`` the second score's own stream is still seeded ``[0, 1]``."""
    evaluation = _evaluation(200, two=True)
    expected = _oracle(evaluation, n_half_samples=30, random_state=None, paired=False)["T"]
    got = tc.half_sample_draws(
        evaluation, n_half_samples=30, random_state=None, paired=False, chunk=4
    )["T"]
    for key in expected:
        np.testing.assert_array_equal(expected[key], got[key])


@pytest.mark.parametrize("chunk", [None, 1, 7])
def test_caller_masks_are_used_as_given(chunk: int | None) -> None:
    evaluation = _evaluation(300, two=True, clusters=60)
    _, float_masks = tc._masks(evaluation.units, 23, np.random.default_rng(3))
    expected = _oracle(evaluation, n_half_samples=999, random_state=0, masks=float_masks)
    for masks in (float_masks, float_masks.astype(np.int8), float_masks.astype(bool)):
        got = tc.half_sample_draws(
            evaluation, n_half_samples=999, random_state=0, masks=masks, chunk=chunk
        )
        _assert_same(expected, got)
        assert all(values.shape == (23,) for values in got["S"].values())


@pytest.mark.parametrize("chunk", [None, 1])
def test_tied_random_keys_keep_their_winner(
    monkeypatch: pytest.MonkeyPatch, chunk: int | None
) -> None:
    """Keys rounded to two decimals tie often; the blocks pick the same units as one matrix."""
    monkeypatch.setattr(tc, "_keys", lambda rng, shape: np.round(rng.random(shape), 2))
    evaluation = _evaluation(400, clusters=300)
    expected = _oracle(evaluation, n_half_samples=31, random_state=2)
    got = tc.half_sample_draws(evaluation, n_half_samples=31, random_state=2, chunk=chunk)
    _assert_same(expected, got)


def test_a_stable_ranking_of_tied_keys_fails_the_tie_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation control: ``argsort(kind="stable")`` breaks key ties differently."""
    monkeypatch.setattr(tc, "_keys", lambda rng, shape: np.round(rng.random(shape), 2))

    def stable(rng: np.random.Generator, draws: int, count: int) -> Any:
        keys = tc._keys(rng, (draws, count))
        chosen = np.argsort(keys, axis=1, kind="stable")[:, : count // 2]
        mask = np.zeros((draws, count), dtype=bool)
        np.put_along_axis(mask, chosen, True, axis=1)
        return mask

    evaluation = _evaluation(400, clusters=300)
    expected = _oracle(evaluation, n_half_samples=31, random_state=2)
    monkeypatch.setattr(tc, "_block_masks", stable)
    got = tc.half_sample_draws(evaluation, n_half_samples=31, random_state=2, chunk=1)
    with pytest.raises(AssertionError):
        _assert_same(expected, got)


def _restarting(original: Any, *, primary: bool) -> Any:
    """A mask draw that puts each generator back where it started, as if made anew per block."""
    first: dict[int, Any] = {}

    def draw(rng: np.random.Generator, draws: int, count: int) -> Any:
        key = id(rng)
        if key not in first:
            first[key] = rng.bit_generator.state
        elif primary or (len(first) > 1 and key != next(iter(first))):
            rng.bit_generator.state = first[key]
        return original(rng, draws, count)

    return draw


@pytest.mark.parametrize("primary", [True, False])
def test_a_generator_restarted_per_block_fails_the_block_witness(
    monkeypatch: pytest.MonkeyPatch, primary: bool
) -> None:
    """Mutation control: re-creating the primary or the unpaired stream per block is caught."""
    evaluation = _evaluation(200, two=True)
    expected = _oracle(evaluation, n_half_samples=21, random_state=4, paired=False)
    monkeypatch.setattr(tc, "_block_masks", _restarting(tc._block_masks, primary=primary))
    one_block = tc.half_sample_draws(
        evaluation, n_half_samples=21, random_state=4, paired=False, chunk=21
    )
    _assert_same(expected, one_block)
    monkeypatch.setattr(tc, "_block_masks", _restarting(tc._block_masks, primary=primary))
    blocks = tc.half_sample_draws(
        evaluation, n_half_samples=21, random_state=4, paired=False, chunk=7
    )
    with pytest.raises(AssertionError):
        _assert_same(expected, blocks)


@pytest.mark.parametrize("units", [9, 10])
def test_each_draw_keeps_half_the_units_whole(units: int) -> None:
    evaluation = _evaluation(60, clusters=units)
    mask = tc._block_masks(np.random.default_rng(0), 40, units)
    assert mask.dtype == bool and np.all(mask.sum(axis=1) == units // 2)
    inverse, float_masks = tc._masks(evaluation.units, 40, np.random.default_rng(0))
    np.testing.assert_array_equal(float_masks, mask.astype(float))
    rows = float_masks[:, inverse]
    for unit in np.unique(inverse):
        block = rows[:, inverse == unit]
        assert np.all(block == block[:, :1])


def test_paired_scores_share_their_masks() -> None:
    evaluation = _evaluation(300, two=True)
    evaluation = tc.TargetingEvaluation(
        scores={"S": evaluation.scores["S"], "T": evaluation.scores["S"].copy()},
        targeted=evaluation.targeted,
        one_step=evaluation.one_step,
        mass=evaluation.mass,
        units=evaluation.units,
        grid=evaluation.grid,
        edges=evaluation.edges,
    )
    paired = tc.half_sample_draws(evaluation, n_half_samples=25, random_state=1, chunk=4)
    unpaired = tc.half_sample_draws(
        evaluation, n_half_samples=25, random_state=1, chunk=4, paired=False
    )
    np.testing.assert_array_equal(paired["S"][("autoc", 1.0)], paired["T"][("autoc", 1.0)])
    assert not np.array_equal(unpaired["S"][("autoc", 1.0)], unpaired["T"][("autoc", 1.0)])


# ------------------------------------------------------------------ B2: the byte model


def _peak(evaluation: tc.TargetingEvaluation, draws: int, **kwargs: Any) -> int:
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        _quiet(tc.half_sample_draws, evaluation, n_half_samples=draws, random_state=0, **kwargs)
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def _predicted(
    evaluation: tc.TargetingEvaluation, draws: int, chunk: int | None, **mutation: Any
) -> tuple[int, int, int]:
    route = tc._route(
        mutation.get("cut", "quantile"),
        mutation.get("ranks", "recompute"),
        mutation.get("tie_rule", "average"),
    )
    fixed, per_draw = tc.draw_bytes(
        n=evaluation.targeted.size,
        units=int(np.unique(evaluation.units).size),
        blocks=max(int(np.unique(score).size) for score in evaluation.scores.values()),
        scores=len(evaluation.scores),
        draws=draws,
        grid=evaluation.grid,
        edges=evaluation.edges,
        route=route,
        unpaired=not mutation.get("paired", True) and len(evaluation.scores) > 1,
    )
    block = chunk or max(1, min(draws, tc.BLOCK_BYTES // per_draw))
    return fixed + block * per_draw, fixed, per_draw


GUARD_CASES: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
    "untied": ({}, {}),
    "tied": ({"blocks": 40}, {}),
    "clustered": ({"clusters": 300}, {}),
    "two paired": ({"two": True}, {}),
    "two unpaired": ({"two": True}, {"paired": False}),
    "two fixed ranks": ({"two": True}, {"ranks": "fixed"}),
    "fixed cut": ({}, {"cut": "fixed"}),
    "fixed cut forty cuts": ({"grid": FORTY}, {"cut": "fixed"}),
    "at or above forty cuts": ({"grid": FORTY, "blocks": 60}, {"tie_rule": "at_or_above"}),
}


@pytest.mark.parametrize("chunk", [None, 1])
@pytest.mark.parametrize("case", list(GUARD_CASES))
def test_the_byte_model_bounds_the_peak_within_its_factor(case: str, chunk: int | None) -> None:
    """Predicted bytes are at least the ``tracemalloc`` peak and at most twice it."""
    shape, mutation = GUARD_CASES[case]
    evaluation = _evaluation(3_000, seed=7, **shape)
    _quiet(tc.half_sample_draws, evaluation, n_half_samples=2, random_state=0, **mutation)
    measured = _peak(evaluation, 40, chunk=chunk, **mutation)
    predicted, _, _ = _predicted(evaluation, 40, chunk, **mutation)
    assert measured <= predicted <= tc.DRAW_BYTES_FACTOR * measured, (predicted, measured)


@pytest.mark.parametrize(
    ("route", "mutation", "shape"),
    [("shipped", {}, {}), ("general", {"cut": "fixed"}, {})],
)
def test_dropping_the_largest_draw_term_falls_below_the_peak(
    monkeypatch: pytest.MonkeyPatch, route: str, mutation: dict[str, Any], shape: dict[str, Any]
) -> None:
    """Mutation control: without the tie-block temporaries the model under-predicts."""
    evaluation = _evaluation(3_000, seed=7, **shape)
    measured = _peak(evaluation, 40, chunk=8, **mutation)
    table = dict(tc._BLOCK_BYTES_PER_TIE)
    table[route] = (0.0, 0.0, 0.0)
    monkeypatch.setattr(tc, "_BLOCK_BYTES_PER_TIE", table)
    predicted, _, _ = _predicted(evaluation, 40, 8, **mutation)
    assert predicted < measured


def test_dropping_the_largest_held_term_falls_below_the_peak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation control: without the held full-sample rank weights the model under-predicts."""
    evaluation = _evaluation(3_000, seed=7, two=True)
    measured = _peak(evaluation, 40, chunk=1, ranks="fixed")
    held = dict(tc._HELD_ARRAYS)
    held["fixed_ranks"] = (0.0, 0.0, 0.0)
    monkeypatch.setattr(tc, "_HELD_ARRAYS", held)
    predicted, _, _ = _predicted(evaluation, 40, 1, ranks="fixed")
    assert predicted < measured


def test_the_budget_refuses_one_byte_short_and_runs_one_draw_at_a_time_one_byte_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluation = _evaluation(2_000, seed=3, two=True)
    _, fixed, per_draw = _predicted(evaluation, 30, None)
    with pytest.raises(MethodConfigurationError, match="max_memory_bytes") as caught:
        tc.half_sample_draws(
            evaluation, n_half_samples=30, random_state=1, max_memory_bytes=fixed + per_draw - 1
        )
    assert caught.value.refusal.reason is RefusalReason.INVALID_DECLARATION
    sizes: list[int] = []
    original = tc._block_masks

    def spy(rng: np.random.Generator, draws: int, count: int) -> Any:
        sizes.append(draws)
        return original(rng, draws, count)

    monkeypatch.setattr(tc, "_block_masks", spy)
    got = tc.half_sample_draws(
        evaluation, n_half_samples=30, random_state=1, max_memory_bytes=fixed + per_draw + 1
    )
    assert sizes == [1] * 30
    _assert_same(_oracle(evaluation, n_half_samples=30, random_state=1), got)


def test_caller_masks_are_never_refused() -> None:
    evaluation = _evaluation(500, seed=3)
    _, masks = tc._masks(evaluation.units, 12, np.random.default_rng(0))
    got = tc.half_sample_draws(
        evaluation, n_half_samples=12, random_state=0, masks=masks, max_memory_bytes=1
    )
    _assert_same(_oracle(evaluation, n_half_samples=12, random_state=0, masks=masks), got)


def test_no_budget_reading_means_no_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory, "available_memory", lambda: None)
    monkeypatch.setattr(tc, "available_memory", lambda: None)
    evaluation = _evaluation(500, seed=3)
    tc.half_sample_draws(evaluation, n_half_samples=12, random_state=0)


def test_the_available_memory_is_the_default_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory, "available_memory", lambda: 1_000)
    evaluation = _evaluation(500, seed=3)
    with pytest.raises(MethodConfigurationError, match="available now"):
        tc.half_sample_draws(evaluation, n_half_samples=12, random_state=0)
    # An explicit budget overrides the reading in either direction.
    tc.half_sample_draws(evaluation, n_half_samples=12, random_state=0, max_memory_bytes=10**9)


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "1000"])
def test_a_budget_must_be_a_positive_integer(value: Any) -> None:
    with pytest.raises(DataError, match="max_memory_bytes"):
        memory.check_max_memory_bytes(value)


def test_a_numpy_integer_budget_is_accepted() -> None:
    assert memory.check_max_memory_bytes(np.int64(10**9)) == 10**9
    assert memory.check_max_memory_bytes(None) is None


def test_the_default_draws_stay_far_below_the_whole_matrix_footprint() -> None:
    """CI memory guard: 5,000 units and 2,000 draws peaked at 405 MB before the blocks."""
    evaluation = _evaluation(5_000, seed=11)
    _quiet(tc.half_sample_draws, evaluation, n_half_samples=2, random_state=0)
    assert _peak(evaluation, 2_000) < 100 * 10**6


# ------------------------------------------------------------------ the facades


def _study_effect(n: int = 600) -> Any:
    import pandas as pd

    from cleverly import CausalStudy, ConditionalContrast, PointTreatment
    from cleverly.bases import Linear, Modifiers

    rng = np.random.default_rng(0)
    w1, w2 = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
    a = rng.binomial(1, 0.5, n)
    y = w2 + a * (1 + w1) + rng.normal(0, 0.5, n)
    frame = pd.DataFrame({"Y": y, "A": a, "W1": w1, "W2": w2, "S": w1})
    study = CausalStudy(
        frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2"))
    )
    return study.identify(ConditionalContrast(Modifiers(["W1"]), model=Linear(["W1"])))


def _learners() -> dict[str, Any]:
    from sklearn.linear_model import LinearRegression, LogisticRegression

    return {
        "outcome_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(),
        "outcome_by_arm": True,
        "q_bounds": (-5.0, 6.0),
    }


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    calls: list[Any] = []
    original = tc.half_sample_draws

    def recording(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs.get("max_memory_bytes"))
        return original(*args, **kwargs)

    monkeypatch.setattr(tc, "half_sample_draws", recording)
    return calls


def _external(**kwargs: Any) -> Any:
    return _quiet(
        _study_effect().targeting_curve,
        "S",
        computed_from=["W1"],
        score_kind="external",
        n_half_samples=20,
        random_state=0,
        **_learners(),
        **kwargs,
    )


def test_max_memory_bytes_reaches_the_draws_from_the_external_route(spy: list[Any]) -> None:
    curve = _external(max_memory_bytes=10**9)
    assert spy == [10**9]
    tc.control_values(curve, random_state=0, max_memory_bytes=10**8)
    assert spy == [10**9, 10**8]


@pytest.fixture(scope="module")
def split_curve() -> Any:
    return _quiet(
        lambda: _study_effect(1_000).estimate(**_learners(), random_state=0).effect_curve()
    )


def test_max_memory_bytes_reaches_the_draws_from_the_split_route(
    spy: list[Any], split_curve: Any
) -> None:
    _quiet(
        split_curve.diagnostics.targeting_curve,
        n_half_samples=20,
        random_state=0,
        max_memory_bytes=10**9,
    )
    assert spy == [10**9]


@pytest.mark.parametrize(("value", "error"), [(1, MethodConfigurationError), (1.5, DataError)])
def test_the_facades_refuse_before_any_fit(
    monkeypatch: pytest.MonkeyPatch, split_curve: Any, value: Any, error: type[Exception]
) -> None:
    from cleverly.estimators import TMLE

    def no_fit(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a fit ran before the memory preflight")

    monkeypatch.setattr(TMLE, "fit", no_fit)
    with pytest.raises(error, match="max_memory_bytes"):
        _external(max_memory_bytes=value)
    monkeypatch.setattr(type(split_curve.diagnostics._estimator(split_curve.fit)), "refit", no_fit)
    with pytest.raises(error, match="max_memory_bytes"):
        split_curve.diagnostics.targeting_curve(n_half_samples=20, max_memory_bytes=value)
