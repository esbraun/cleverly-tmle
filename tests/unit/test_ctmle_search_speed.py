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


def _scoring_lookups(monkeypatch: pytest.MonkeyPatch, n: int) -> int:
    """``column_for`` calls made by the three risk-scoring sites during one greedy fit."""
    import sys
    import types

    frame = _frame(n=n)
    calls = 0
    original = Propensity.column_for
    scoring = {"_fluctuate", "_derivative_nuisances", "_outcome_adaptive_nuisances"}

    def counting(self: Propensity, arm: float) -> int:
        nonlocal calls
        caller: types.FrameType | None = sys._getframe(1)
        # Skip the comprehension's own frame, which is where a per-row form calls from.
        while caller is not None and caller.f_code.co_name in {"<listcomp>", "<genexpr>"}:
            caller = caller.f_back
        if caller is not None and caller.f_code.co_name in scoring:
            calls += 1
        return original(self, arm)

    monkeypatch.setattr(Propensity, "column_for", counting)
    _greedy(frame).fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2", "W3"])
    return calls


def test_the_search_no_longer_looks_up_columns_row_by_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """Work guard: the per-row form called ``column_for`` once per scored row per candidate."""
    assert _scoring_lookups(monkeypatch, 1000) == 0
    assert _scoring_lookups(monkeypatch, 2000) == 0
