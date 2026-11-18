"""The training-support preflight counts units without sorting every label.

The refusals ask whether an arm or an outcome class reaches two independent units and print
the count when it does not. ``_units_up_to_two`` answers that exactly below two and with "at
least two" above, which is all the refusals read, so their decisions and messages are the ones
a full ``np.unique`` count gives.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.estimators import DRTMLE
from cleverly.exceptions import DataError

# The package exports a ``tmle`` function that shadows the module's name.
tmle_module = importlib.import_module("cleverly.estimators.tmle")

Counter = Callable[[np.ndarray, np.ndarray, bool], int]


def _reference(unit_of: np.ndarray, rows: np.ndarray, clustered: bool) -> int:
    """The full count the preflight took before."""
    del clustered
    return int(np.unique(unit_of[rows]).size)


def _rows_only(unit_of: np.ndarray, rows: np.ndarray, clustered: bool) -> int:
    """A deliberate mutation: rows counted as units even when clusters are declared."""
    del unit_of, clustered
    return int(np.count_nonzero(rows))


def _agrees(counter: Counter, unit_of: np.ndarray, rows: np.ndarray, clustered: bool) -> bool:
    want = _reference(unit_of, rows, clustered)
    got = counter(unit_of, rows, clustered)
    return got == want if want < 2 else got >= 2


def _cases() -> list[tuple[np.ndarray, np.ndarray, bool]]:
    rng = np.random.default_rng(20261115)
    cases: list[tuple[np.ndarray, np.ndarray, bool]] = []
    for n in (1, 2, 5, 40):
        unit_of = np.arange(n, dtype=np.int64)
        for keep in (0, 1, 2, n):
            rows = np.zeros(n, dtype=bool)
            rows[rng.permutation(n)[: min(keep, n)]] = True
            cases.append((unit_of, rows, False))
    for clusters in (1, 2, 7):
        unit_of = np.repeat(np.arange(clusters, dtype=np.int64), 4)
        for chosen in ([], [0], [0, 1], [0, 4], [3, 7], list(range(unit_of.size))):
            rows = np.zeros(unit_of.size, dtype=bool)
            rows[[c for c in chosen if c < unit_of.size]] = True
            cases.append((unit_of, rows, True))
    # Labels that are not integer codes take the full count: two NaN labels are one unit.
    unit_of_float = np.array([np.nan, np.nan, 1.0, 1.0])
    for chosen in ([0, 1], [0, 2], [2, 3]):
        rows = np.zeros(4, dtype=bool)
        rows[chosen] = True
        cases.append((unit_of_float, rows, True))
    return cases


@pytest.mark.parametrize("case", range(len(_cases())))
def test_the_count_is_exact_below_two_and_at_least_two_above(case: int) -> None:
    unit_of, rows, clustered = _cases()[case]
    assert _agrees(tmle_module._units_up_to_two, unit_of, rows, clustered)


def test_counting_rows_inside_clusters_is_caught() -> None:
    # The mutation counts two rows of one cluster as two units; the witness sees it.
    assert not all(_agrees(_rows_only, *case) for case in _cases())


def _frame(n: int = 60, clusters: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            "Y": rng.binomial(1, 0.4, n).astype(float),
            "A": rng.integers(0, 3, n).astype(float),
            "W1": rng.normal(size=n),
            "W2": rng.normal(size=n),
        }
    )
    if clusters:
        frame["cid"] = np.repeat(np.arange(n // 5), 5)
    return frame


def _refusal(frame: pd.DataFrame, **fit: Any) -> str | None:
    # DR-TMLE's outcome regression takes no constant-target rule, so it owes two units of
    # each outcome class; a plain TMLE owes the arms only.
    estimator = DRTMLE(
        outcome_learner=LogisticRegression(max_iter=1000),
        treatment_learner=LogisticRegression(max_iter=1000),
        reduced_outcome_learner=LinearRegression(),
        reduced_mechanism_residual_learner=LinearRegression(),
        n_folds=3,
        random_state=0,
        simultaneous=False,
        estimands=["ate"],
    )
    try:
        estimator.fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"], **fit)
    except (DataError, ValueError) as error:  # a later check may refuse a passing count
        return f"{type(error).__name__}: {error}"
    return None


def _refusal_fixtures() -> dict[str, tuple[pd.DataFrame, dict[str, Any]]]:
    one_arm_row = _frame()
    one_arm_row["A"] = np.where(np.arange(60) == 0, 2.0, one_arm_row["A"] % 2)
    one_event = _frame()
    one_event["Y"] = 0.0
    one_event.loc[0, "Y"] = 1.0
    one_cluster_events = _frame(clusters=True)
    one_cluster_events["Y"] = 0.0
    one_cluster_events.loc[[0, 1], "Y"] = 1.0  # two rows, one cluster
    one_cluster_arm = _frame(clusters=True)
    one_cluster_arm["A"] = np.where(one_cluster_arm["cid"] == 0, 2.0, one_cluster_arm["A"] % 2)
    two_clusters_arm = _frame(clusters=True)
    two_clusters_arm["A"] = np.where(two_clusters_arm["cid"] < 2, 2.0, two_clusters_arm["A"] % 2)
    return {
        "an arm in one row": (one_arm_row, {}),
        "an outcome class in one row": (one_event, {}),
        "an outcome class in one cluster": (one_cluster_events, {"id": "cid"}),
        "an arm in one cluster": (one_cluster_arm, {"id": "cid"}),
        "two clusters holding an arm": (two_clusters_arm, {"id": "cid"}),
        "a sample that passes": (_frame(), {}),
        "a clustered sample that passes": (_frame(clusters=True), {"id": "cid"}),
    }


@pytest.mark.parametrize("fixture", list(_refusal_fixtures()))
def test_each_refusal_reads_as_the_full_count_writes_it(
    fixture: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame, fit = _refusal_fixtures()[fixture]
    got = _refusal(frame, **fit)
    monkeypatch.setattr(tmle_module, "_units_up_to_two", _reference)
    assert got == _refusal(frame, **fit)
    if fixture.startswith("an "):
        assert got is not None and "in at least two independent" in got
    else:
        assert got is None or "in at least two independent" not in got


def test_an_unclustered_sample_sorts_no_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    original = np.unique

    def counting(values: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(np.size(values))
        return original(values, *args, **kwargs)

    unit_of = np.arange(200_000, dtype=np.int64)
    rows = np.arange(200_000) % 3 == 0
    monkeypatch.setattr(np, "unique", counting)
    assert tmle_module._units_up_to_two(unit_of, rows, False) == int(rows.sum())
    clustered = np.repeat(np.arange(2_000, dtype=np.int64), 100)
    assert tmle_module._units_up_to_two(clustered, rows, True) == 2
    assert calls == []
