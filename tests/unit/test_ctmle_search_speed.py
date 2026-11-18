"""The C-TMLE candidate search reads each row's observed-arm column in one pass (X106 K2a).

``Propensity.columns_for`` replaced a per-row Python comprehension over ``column_for``
that took about a fifth of a greedy C-TMLE fit at 100k rows (unprofiled). The change is exact: these
tests compare it with that comprehension element for element, pin the error it raises,
and pin that the search no longer calls ``column_for`` once per row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from cleverly.estimators import CTMLE
from cleverly.estimators._nuisance import Propensity
from cleverly.learners import GLM

UNSORTED = (1.0, 0.0, 2.0)


def _reference(propensity: Propensity, levels: object) -> np.ndarray:
    """The comprehension the search used before K2a, verbatim."""
    return np.array([propensity.column_for(float(arm)) for arm in levels], dtype=int)  # type: ignore[attr-defined]


def _propensity(arms: tuple[float, ...], n: int = 4) -> Propensity:
    return Propensity(np.full((n, len(arms)), 1.0 / len(arms)), arms, simplex=len(arms) == 2)


@pytest.mark.parametrize(
    ("arms", "levels"),
    [
        ((0.0, 1.0), np.array([1, 0, 0, 1, 1])),
        ((0.0, 1.0), np.array([1.0, 0.0, -0.0, 1.0])),
        (UNSORTED, np.array([0.0, 1.0, 2.0, 2.0, 0.0, 1.0])),
        (UNSORTED, np.array([2, 2, 1], dtype=np.int64)),
        (UNSORTED, np.array([], dtype=float)),
    ],
)
def test_columns_match_the_per_row_lookup(arms: tuple[float, ...], levels: np.ndarray) -> None:
    propensity = _propensity(arms)
    got = propensity.columns_for(levels)
    expected = _reference(propensity, levels)
    assert got.dtype == expected.dtype
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("bad", [3.0, float("nan")])
def test_an_unmatched_level_raises_what_the_per_row_lookup_raised(bad: float) -> None:
    propensity = _propensity(UNSORTED)
    levels = np.array([0.0, bad, 1.0, 5.0])
    with pytest.raises(KeyError) as reference:
        _reference(propensity, levels)
    with pytest.raises(KeyError) as got:
        propensity.columns_for(levels)
    assert str(got.value) == str(reference.value)


def test_arm_order_mistakes_are_visible() -> None:
    """Mutation controls: a sorted-position lookup and a last-match loop give other columns."""
    propensity = _propensity(UNSORTED)
    levels = np.array([0.0, 1.0, 2.0])
    reference = _reference(propensity, levels)
    np.testing.assert_array_equal(propensity.columns_for(levels), reference)
    sorted_position = np.searchsorted(np.sort(np.array(UNSORTED)), levels)
    assert not np.array_equal(sorted_position, reference)

    # First match is observable only when an arm value repeats, which `Propensity` allows
    # for more than two arms.
    duplicated = Propensity(np.full((2, 3), 1.0 / 3.0), (1.0, 0.0, 1.0), simplex=False)
    levels = np.array([1.0, 0.0])
    reference = _reference(duplicated, levels)
    np.testing.assert_array_equal(reference, [0, 1])
    np.testing.assert_array_equal(duplicated.columns_for(levels), reference)
    last_match = np.full(levels.shape, -1)
    for j, level in enumerate(duplicated.arms):
        last_match[levels == level] = j
    assert not np.array_equal(last_match, reference)


def _frame(n: int = 600, arms: int = 2, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n, 3))
    if arms == 2:
        a = rng.binomial(1, 1.0 / (1.0 + np.exp(-0.4 * w[:, 0])))
    else:
        a = rng.integers(0, arms, size=n)
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-(0.3 * w[:, 1] + 0.4 * (a > 0)))))
    frame = pd.DataFrame(w, columns=["W1", "W2", "W3"])
    frame["A"] = a
    frame["Y"] = y
    frame["wt"] = rng.uniform(0.5, 2.0, n)
    return frame


def _greedy(frame: pd.DataFrame, **overrides: object) -> CTMLE:
    settings: dict[str, object] = {
        "outcome_learner": GLM(family="binomial"),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "n_folds": 2,
        "selection_folds": 2,
        "selection_inner_folds": 2,
        "random_state": 11,
        "estimands": ("ate",),
        **overrides,
    }
    if settings.get("strategy") in {"derivative", "oat"}:
        # Selector settings are refused by the strategies that select nothing.
        del settings["selection_folds"], settings["selection_inner_folds"]
    return CTMLE(**settings)  # type: ignore[arg-type]


def test_candidate_treatment_risk_equals_the_per_row_formula(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every scored candidate's treatment risk is bitwise the pre-K2a formula, three arms, weights."""
    from cleverly.estimators import ctmle

    frame = _frame(arms=3)
    checked: list[tuple[float, float]] = []
    original = ctmle._Selector._fluctuate

    def spy(self, covariates, propensity, initial, rows, n_steps, label=""):  # type: ignore[no-untyped-def]
        candidate = original(self, covariates, propensity, initial, rows, n_steps, label)
        g = np.clip(propensity.values[rows], ctmle._LOSS_EPS, 1.0 - ctmle._LOSS_EPS)
        columns = _reference(propensity, self.data.treatment[rows])
        w = self.data.weights[rows]
        expected = float(-np.sum(w * np.log(g[np.arange(rows.size), columns])))
        checked.append((candidate.treatment_risk, expected))
        return candidate

    monkeypatch.setattr(ctmle._Selector, "_fluctuate", spy)
    _greedy(frame).fit(
        frame, outcome="Y", treatment="A", covariates=["W1", "W2", "W3"], weights="wt"
    )
    assert checked
    assert all(got == expected for got, expected in checked)


_SCORING = {"_fluctuate", "_derivative_nuisances", "_outcome_adaptive_nuisances"}


def _per_row(self: Propensity, levels: object) -> np.ndarray:
    """The pre-K2a lookup, swapped in as the guard's positive control."""
    return np.array([self.column_for(float(arm)) for arm in levels], dtype=int)  # type: ignore[attr-defined]


def _scoring_lookups(
    monkeypatch: pytest.MonkeyPatch, n: int, strategy: str = "greedy", *, per_row: bool = False
) -> int:
    """``column_for`` calls made on behalf of the three risk-scoring sites during one fit.

    A call counts when a scoring site is within three frames of it, which covers the
    sites calling ``column_for`` themselves and calling it through ``columns_for``. The
    scalar calls of ``Propensity.truncate`` are not counted.
    """
    import sys
    import types

    frame = _frame(n=n)
    calls = 0
    original = Propensity.column_for

    def counting(self: Propensity, arm: float) -> int:
        nonlocal calls
        caller: types.FrameType | None = sys._getframe(1)
        for _ in range(3):
            if caller is None or caller.f_code.co_name == "truncate":
                break
            if caller.f_code.co_name in _SCORING:
                calls += 1
                break
            caller = caller.f_back
        return original(self, arm)

    monkeypatch.setattr(Propensity, "column_for", counting)
    if per_row:
        monkeypatch.setattr(Propensity, "columns_for", _per_row)
    overrides: dict[str, object] = {"strategy": strategy}
    if strategy == "derivative":
        overrides["lambdas"] = [0.1, 0.01]
    if strategy == "oat":
        # The shared design scores the one mechanism with the observed-arm lookup; the
        # per-arm design scores each arm's binary factor instead.
        overrides["oat_design"] = "shared"
    _greedy(frame, **overrides).fit(
        frame, outcome="Y", treatment="A", covariates=["W1", "W2", "W3"]
    )
    return calls


@pytest.mark.parametrize("strategy", ["greedy", "derivative", "oat"])
def test_the_search_no_longer_looks_up_columns_row_by_row(
    monkeypatch: pytest.MonkeyPatch, strategy: str
) -> None:
    """Work guard: the per-row form called ``column_for`` once per scored row per candidate."""
    assert _scoring_lookups(monkeypatch, 1000, strategy) == 0
    assert _scoring_lookups(monkeypatch, 2000, strategy) == 0


@pytest.mark.parametrize("strategy", ["greedy", "derivative", "oat"])
def test_the_guard_sees_a_per_row_lookup(monkeypatch: pytest.MonkeyPatch, strategy: str) -> None:
    """Positive control: with the per-row lookup swapped back in, the guard counts every row."""
    assert _scoring_lookups(monkeypatch, 1000, strategy, per_row=True) >= 1000
