"""What ``SurvivalCTMLE``'s selection keeps in memory, and that keeping less changes nothing.

The search scores every candidate position but keeps a position's row arrays only while the
path can read them again: the first position, each accepted one and the best of the pass in
progress.  A position the rule asks for after its arrays went is targeted again.  These checks
compare the shipped path, byte for byte, with a reference that keeps every position (the path
as it was before the retention change), bound the live positions of a whole fit, and pin the
work counts the selection record reports.
"""

from __future__ import annotations

import dataclasses
import inspect
import textwrap
import warnings
import weakref
from typing import Any

import numpy as np
import pytest

from cleverly.exceptions import ConvergenceWarning
from cleverly.longitudinal import LongitudinalData
from cleverly.longitudinal import collaborative as module
from tests.unit.test_survival_ctmle import Pick, _fit, _frame, _instrument, _plan, _selector


def _reference_path(
    selector: module._HazardSelector, plan: Any, horizon: int, branch: str, train: Any
) -> list[module._Position]:
    """The path that caches every scored position, as the selector did before it kept less."""
    weights = selector.data.weights * (1.0 if train is None else train.astype(float))
    cache: dict[Any, module._Position] = {}
    accepted_at: dict[Any, module._Position] = {}
    starts = {0: selector.start}

    def score(accepted: Any, move: Any, n_steps: int) -> float:
        if accepted and accepted not in accepted_at:
            accepted_at[accepted] = cache[(accepted[:-1], accepted[-1], n_steps)]
        if n_steps not in starts:
            current = accepted_at[accepted] if accepted else cache[((), None, 0)]
            starts[n_steps] = current.state
        key = (accepted, move, n_steps)
        if key not in cache:
            chosen: tuple[set[str], set[str]] = (set(), set())
            for mechanism, name in (*accepted, *(() if move is None else (move,))):
                chosen[0 if mechanism == "g1" else 1].add(name)
            terms = (selector._ordered(0, chosen[0]), selector._ordered(1, chosen[1]))
            cache[key] = selector.position(
                plan, branch, train, terms, move, starts[n_steps], horizon, weights, n_steps
            )
        return cache[key].risk

    rule = module.greedy_rule(selector.pools, score)
    moves = [move for move, _ in rule[1:] if move is not None]
    return [cache[((), None, 0)]] + [
        cache[(tuple(moves[:index]), move, n_steps)]
        for index, (move, n_steps) in enumerate(rule[1:])
    ]


def _arrays(position: module._Position) -> list[tuple[str, np.ndarray]]:
    """Every row array a position holds, named."""
    out = []
    for owner, value in (
        ("state", position.state),
        ("inputs", position.inputs),
        ("candidate", position.candidate),
    ):
        for field in dataclasses.fields(value):
            item = getattr(value, field.name)
            items = item if isinstance(item, tuple) else (item,)
            for index, part in enumerate(items):
                if isinstance(part, np.ndarray):
                    out.append((f"{owner}.{field.name}[{index}]", part))
    return out


def _assert_same_path(shipped: list[module._Position], reference: list[module._Position]) -> None:
    assert [(p.move, p.n_steps) for p in shipped] == [(p.move, p.n_steps) for p in reference]
    for left, right in zip(shipped, reference, strict=True):
        assert (left.loss, left.penalty, left.clipped) == (right.loss, right.penalty, right.clipped)
        assert (left.record.converged, left.record.n_iter) == (
            right.record.converged,
            right.record.n_iter,
        )
        named = _arrays(right)
        assert [name for name, _ in _arrays(left)] == [name for name, _ in named]
        for (name, a), (_, b) in zip(_arrays(left), named, strict=True):
            assert a.dtype == b.dtype and a.shape == b.shape, name
            assert a.tobytes() == b.tobytes(), name


@pytest.fixture(scope="module")
def instrument() -> LongitudinalData:
    return _instrument()


def _fold_train(selector: module._HazardSelector, fold: int) -> np.ndarray:
    fit_rows, _ = list(selector.selection)[fold - 1]
    train = np.zeros(selector.data.n, dtype=bool)
    train[fit_rows] = True
    return train


# ------------------------------------------------------------- the path equals the reference


@pytest.mark.parametrize(("horizon", "branch"), [(1, "full"), (2, "full"), (2, "cv2")])
def test_the_path_equals_the_path_that_keeps_every_position(
    instrument: LongitudinalData, horizon: int, branch: str
) -> None:
    selector = _selector(instrument, Pick((4,), 5))
    plan = _plan(selector, "a1")
    train = None if branch == "full" else _fold_train(selector, int(branch[2:]))
    shipped = selector.path(plan, horizon, branch, train)
    reference = _reference_path(selector, plan, horizon, branch, train)
    _assert_same_path(shipped, reference)
    assert selector.counts[("a1", horizon, branch)].rebuilt == 0


def test_the_witness_law_ties_and_restarts(instrument: LongitudinalData) -> None:
    """Horizon 1 scores every g2 move at one risk, and the search restarts: the shipped rule
    takes the first move of a tie, so the retention must keep it."""
    selector = _selector(instrument, Pick((4,), 5))
    plan = _plan(selector, "a1")
    path = selector.path(plan, 1, "full", None)
    risks = [p.risk for p in path]
    assert any(risks.count(r) > 1 for r in risks), "the law needs an exact tie"
    assert any(p.n_steps > 0 for p in path), "the law needs a restart"


_SOURCES = {
    "path": textwrap.dedent(inspect.getsource(module._HazardSelector.path)),
    "greedy_rule": textwrap.dedent(inspect.getsource(module.greedy_rule)),
}


def _mutated(name: str, old: str, new: str) -> Any:
    """The shipped ``name`` with one edit to its source, compiled in the module's namespace."""
    source = _SOURCES[name]
    assert source.count(old) == 1, old
    namespace = dict(vars(module))
    exec(compile(source.replace(old, new), module.__file__, "exec"), namespace)
    return namespace[name]


def test_a_retention_that_keeps_the_last_of_a_tie_rebuilds(
    instrument: LongitudinalData, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mutation keeps the last minimum of a pass.  The path still equals the reference,
    because the released position is targeted again, but the rebuild count catches it."""
    old = "_finite(position.risk) < _finite(risks[incumbent])"
    path = _mutated("path", old, old.replace(" < ", " <= "))
    monkeypatch.setattr(module._HazardSelector, "path", path)
    selector = _selector(instrument, Pick((4,), 5))
    plan = _plan(selector, "a1")
    _assert_same_path(
        selector.path(plan, 1, "full", None), _reference_path(selector, plan, 1, "full", None)
    )
    assert selector.counts[("a1", 1, "full")].rebuilt > 0
    # The unmutated source compiled the same way keeps the first, so the edit is what fails.
    monkeypatch.setattr(module._HazardSelector, "path", _mutated("path", old, old))
    selector = _selector(instrument, Pick((4,), 5))
    selector.path(plan, 1, "full", None)
    assert selector.counts[("a1", 1, "full")].rebuilt == 0


def test_a_rule_the_retention_does_not_follow_gets_rebuilt_positions(
    instrument: LongitudinalData, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rule that sends ties to the last minimum asks for released positions, restarted ones
    among them; each rebuilt position equals the reference's by bytes."""
    last = _mutated(
        "greedy_rule",
        "best = int(np.argmin(risks))",
        "best = len(risks) - 1 - int(np.argmin(risks[::-1]))",
    )
    monkeypatch.setattr(module, "greedy_rule", last)
    selector = _selector(instrument, Pick((4,), 5))
    plan = _plan(selector, "a1")
    built: list[tuple[Any, int]] = []
    position = selector.position

    def spy(*args: Any) -> Any:
        built.append((args[3], args[8]))
        return position(*args)

    monkeypatch.setattr(selector, "position", spy)
    shipped = selector.path(plan, 1, "full", None)
    twice = {key for key in built if built.count(key) > 1}
    reference = _reference_path(selector, plan, 1, "full", None)
    _assert_same_path(shipped, reference)
    assert selector.counts[("a1", 1, "full")].rebuilt > 0
    assert any(n_steps > 0 for _, n_steps in twice), "no restarted position was rebuilt"


# ---------------------------------------------------------------- what a whole fit keeps


def _live_positions(selector: module._HazardSelector, monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record the live position count after every position the selector builds."""
    live: list[weakref.ref[module._Position]] = []
    counts: list[int] = []
    position = selector.position

    def tracked(*args: Any) -> Any:
        out = position(*args)
        live.append(weakref.ref(out))
        counts.append(sum(ref() is not None for ref in live))
        return out

    monkeypatch.setattr(selector, "position", tracked)
    return counts


def test_a_fit_keeps_at_most_the_positions_its_path_can_read(
    instrument: LongitudinalData, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bound does not grow with the covariates: the latest accepted position (the first
    one before any move), the pass best, the one being built, the full path's chosen position,
    and one chosen position per finished (arm, horizon).  The reference path, which keeps every
    scored position, exceeds it on the same fit."""
    horizons = [1, 2]
    keys = 2 * len(horizons)
    bound = 1 + 1 + 1 + 1 + (keys - 1)
    selector = _selector(instrument, Pick((4,), 5))
    counts = _live_positions(selector, monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module.run_selection(selector, horizons)
    assert max(counts) <= bound
    assert all(run.rebuilt == 0 for run in selector.counts.values())
    # The reference keeps every position of a path.
    reference = _selector(instrument, Pick((4,), 5))

    def keep_all(plan: Any, horizon: int, branch: str, train: Any, visit: Any = None) -> Any:
        out = _reference_path(reference, plan, horizon, branch, train)
        reference.counts[(plan.label, int(horizon), branch)] = module._PathCounts(0, 0, 0)
        for index, position in enumerate(out):
            if visit is not None:
                visit(index, position)
        return out

    monkeypatch.setattr(reference, "path", keep_all)
    counts = _live_positions(reference, monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module.run_selection(reference, horizons)
    assert max(counts) > 2 * bound


def test_the_mechanism_cache_holds_fits_not_rows(instrument: LongitudinalData) -> None:
    selector = _selector(instrument, Pick((4,), 5))
    plan = _plan(selector, "a1")
    selector.path(plan, 2, "full", None)
    mechanisms = selector.mechanisms
    assert mechanisms._g1 and mechanisms._g2
    caches: list[Any] = [*mechanisms._g1.values(), *mechanisms._g2.values()]
    for cached in caches:
        for part in cached.parts:
            # The predicted-row mask is the only row-sized member: no prediction is kept.
            rows = [x for x in part if isinstance(x, np.ndarray) and x.size >= instrument.n]
            assert len(rows) == 1 and rows[0].dtype == bool
    assert not hasattr(mechanisms, "_pairs")


# --------------------------------------------------------------------------- the counts


def test_the_record_counts_the_solves_and_candidates_of_each_path(
    instrument: LongitudinalData, monkeypatch: pytest.MonkeyPatch
) -> None:
    selector = _selector(instrument, Pick((4,), 5))
    plan = _plan(selector, "a1")
    built: list[Any] = []
    position = selector.position

    def spy(*args: Any) -> Any:
        built.append(args[3])
        return position(*args)

    monkeypatch.setattr(selector, "position", spy)
    path = selector.path(plan, 2, "full", None)
    run = selector.counts[("a1", 2, "full")]
    assert run.solves == len(built) > len(path)
    assert run.candidates == len(set(built))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, records, _ = module.run_selection(selector, [2])
    record = records[("a1", 2)]
    assert (
        len(record.target_solves) == len(record.candidate_pairs) == 1 + selector.selection.n_folds
    )
    assert record.target_solves[0] == run.solves
    solves = f"{sum(record.target_solves)}, {record.target_solves[0]} on full data"
    assert record.summary().facts["Solves"] == solves


def test_a_missing_path_count_is_an_error(instrument: LongitudinalData) -> None:
    selector = _selector(instrument, Pick((4,), 5))
    selector.cross_validate = lambda plan, horizon: (np.zeros(10), ())  # type: ignore[method-assign]
    with pytest.raises(KeyError):
        module.run_selection(selector, [2])


def test_regimens_on_one_arm_count_its_unconverged_fits_once_each() -> None:
    """Two regimens on one arm reuse its search, and the warning counts the arm's unconverged
    candidate fits for each regimen, as before the search kept less."""
    frame = _frame(
        1200,
        19,
        3,
        lambda w: 40.0 * w[:, 2],
        lambda w, a, k: -0.6 + 0.5 * w[:, 0] - 0.5 * a,
        lambda w, a, k: -2.0 + 0.4 * w[:, 1],
    )
    from tests.unit.test_survival_ctmle import _data

    data = _data(frame)
    counts = []
    for regimens in ({"a1": 1, "a0": 0}, {"a1": 1, "b1": 1, "a0": 0}):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            module.SurvivalCTMLE(
                regimens,
                reference="a0",
                outcome_learner=Pick((0, 3), 4),
                selection_folds=3,
                random_state=0,
            ).fit(data)
        text = [
            str(w.message)
            for w in caught
            if issubclass(w.category, ConvergenceWarning)
            and "candidate mechanism" in str(w.message)
        ]
        counts.append(int(text[0].split()[0]))
    assert 0 < counts[0] < counts[1]


def test_fit_refits_reproduce_the_counts(instrument: LongitudinalData) -> None:
    first = _fit(instrument, Pick((4,), 5))
    second = _fit(instrument, Pick((4,), 5))
    for key, record in first.selection.items():
        assert record.target_solves == second.selection[key].target_solves
        assert record.candidate_pairs == second.selection[key].candidate_pairs


# ------------------------------------------------- the mechanisms equal the cached predictions


def _reference_treatment(m: module._Mechanisms, branch: str, train: Any, terms: Any) -> Any:
    """``_Mechanisms.treatment`` as it was when it cached row predictions."""
    data = m.data
    target = (m.observed == 1).astype(float)
    columns = [m.covariates.index(name) for name in terms]
    design = np.asarray(data.baseline[:, columns], dtype=float).reshape(data.n, len(columns))
    p = np.empty(data.n)
    converged, dropped = True, 0
    for _, fit_rows, predict in m._splits(branch, train):
        w = data.weights[fit_rows]
        if not terms:
            p[predict] = float(np.sum(w * target[fit_rows]) / np.sum(w))
            continue
        fit = module._logistic(design[fit_rows], target[fit_rows], w)
        converged = converged and fit.converged
        dropped = max(dropped, int(np.count_nonzero(~fit.active[1:])))
        p[predict] = module._predict(fit, design[predict])
    return np.column_stack([1.0 - p, p]), converged, dropped


def _reference_retention(m: module._Mechanisms, branch: str, train: Any, terms: Any) -> Any:
    """``_Mechanisms.retention`` as it was when it cached row predictions."""
    data = m.data
    retention = np.ones((data.n, data.n_times, m.n_levels))
    converged, dropped = True, 0
    held: list[tuple[int, int, str]] = []
    for fold, fit_rows, predict in m._splits(branch, train):
        fitted: list[int] = []
        for k in range(data.n_times):
            rows = m.eligible[:, k] & fit_rows
            if not rows.any() or not m.censored[rows, k].any():
                held.append((fold, k + 1, module.NO_CENSORING))
            elif bool(np.all(m.censored[rows, k])):
                held.append((fold, k + 1, module.ALL_CENSORED))
                retention[predict, k, :] = module._HELD_RETENTION
            else:
                fitted.append(k)
        if not fitted:
            continue
        design_rows, target, weights = [], [], []
        for k in fitted:
            members = np.flatnonzero(m.eligible[:, k] & fit_rows)
            design_rows.append(m._g2_design(members, k, fitted, terms, None))
            target.append(m.censored[members, k].astype(float))
            weights.append(data.weights[members])
        fit = module._logistic(
            np.vstack(design_rows), np.concatenate(target), np.concatenate(weights)
        )
        converged = converged and fit.converged
        dropped = max(dropped, int(np.count_nonzero(~fit.active[len(fitted) :])))
        index = np.flatnonzero(predict)
        for k in fitted:
            for level in range(m.n_levels):
                at = None if m.evaluate_at_observed else level
                hazard = module._predict(fit, m._g2_design(index, k, fitted, terms, at))
                retention[index, k, level] = 1.0 - hazard
    return retention, converged, dropped, tuple(held)


def _heavy() -> LongitudinalData:
    from tests.unit.test_survival_ctmle import _heavy_censoring

    return _heavy_censoring()


def test_the_mechanisms_rebuilt_from_cached_fits_equal_the_cached_predictions() -> None:
    """Every branch, including a training mask whose rows at risk at the last node are all
    censored there, so that node is held at the all-censored retention."""
    data = _heavy()
    selector = _selector(data, Pick((0, 1, 3, 4), 5))
    m = selector.mechanisms
    last = data.n_times - 1
    eligible = m.eligible[:, last]
    all_censored = ~eligible | m.censored[:, last]
    trains: list[tuple[str, Any]] = [("full", None), ("cv1", _fold_train(selector, 1))]
    trains.append(("crafted", all_censored))
    seen_all_censored = False
    for branch, train in trains:
        for g1 in ((), ("W1",), ("W1", "W3")):
            got = m.treatment(branch, train, g1)
            want = _reference_treatment(m, branch, train, g1)
            assert got.probabilities.tobytes() == want[0].tobytes()
            assert (got.converged, got.dropped) == want[1:]
            # A second call rebuilds from the cached fits and is the same.
            assert m.treatment(branch, train, g1).probabilities.tobytes() == want[0].tobytes()
        for g2 in ((), ("A",), ("A", "W3"), ("W1", "W2", "W3")):
            got = m.retention(branch, train, g2)
            want = _reference_retention(m, branch, train, g2)
            assert got.retention.tobytes() == want[0].tobytes()
            assert (got.converged, got.dropped, got.held) == want[1:]
            seen_all_censored |= any(kind == module.ALL_CENSORED for _, _, kind in got.held)
    assert seen_all_censored, "the crafted mask must hold a node at the all-censored retention"
