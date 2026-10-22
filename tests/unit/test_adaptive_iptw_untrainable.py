r"""The leave-two-folds-out refusal of cross-fitted AdaptiveIPTW, and its zero-fit guarantee.

When two folds hold every trainable row of a class an estimated mechanism needs, the pair's
refit sees one class. The check runs after the folds are drawn and before any learner.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.estimators import AdaptiveIPTW
from cleverly.estimators.iptw import refuse_untrainable_mechanism_pairs
from cleverly.exceptions import DataError
from cleverly.learners.crossfit import Folds


class Spy(LogisticRegression):
    calls: list[int] = []  # noqa: RUF012 - shared across clones on purpose

    def fit(self, x: Any, y: Any, sample_weight: Any = None) -> Any:
        type(self).calls.append(len(y))
        return super().fit(x, y, sample_weight=sample_weight)


class SpyRegressor(LinearRegression):
    calls: list[int] = []  # noqa: RUF012 - shared across clones on purpose

    def fit(self, x: Any, y: Any, sample_weight: Any = None) -> Any:
        type(self).calls.append(len(y))
        return super().fit(x, y, sample_weight=sample_weight)


@pytest.fixture(autouse=True)
def _reset() -> None:
    Spy.calls = []
    SpyRegressor.calls = []


def _frame(n: int = 90) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "W": rng.normal(size=n),
            "A": rng.binomial(1, 0.5, size=n).astype(float),
            "Y": rng.binomial(1, 0.4, size=n).astype(float),
            "Delta": np.ones(n),
            "DeltaA": np.ones(n),
        }
    )


def _fit(frame: pd.DataFrame, seed: int, **kwargs: Any) -> Any:
    estimator = AdaptiveIPTW(
        treatment_learner=Spy(),
        missingness_learner=Spy(),
        reduced_outcome_learner=SpyRegressor(),
        cross_fit=True,
        n_folds=3,
        random_state=seed,
        simultaneous=False,
        stratify_folds="none",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator.fit(frame, outcome="Y", treatment="A", covariates=["W"], **kwargs)


def _rare(frame: pd.DataFrame, column: str, rows: list[int], rare: float) -> pd.DataFrame:
    out = frame.copy()
    out[column] = 1.0 - rare
    out.loc[rows, column] = rare
    if column == "Delta":
        out.loc[out["Delta"] == 0.0, "Y"] = np.nan
    if column == "DeltaA":
        out.loc[out["DeltaA"] == 0.0, "A"] = np.nan
    return out


def _treated_two(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["A"] = 0.0
    out.loc[[3, 50], "A"] = 1.0
    return out


CASES = {
    "treatment": (_treated_two, {}, "arm 1"),
    "outcome observation": (
        lambda f: _rare(f, "Delta", [3, 50], 1.0),
        {"delta": "Delta"},
        "Delta = 1",
    ),
    "treatment observation": (
        lambda f: _rare(f, "DeltaA", [3, 50], 0.0),
        {"treatment_delta": "DeltaA"},
        "DeltaA = 0",
    ),
}


@pytest.mark.parametrize("factor", list(CASES))
def test_a_pair_that_holds_every_row_of_a_class_is_refused_before_any_fit(factor: str) -> None:
    build, kwargs, label = CASES[factor]
    frame = build(_frame())
    messages = []
    for seed in range(8):
        Spy.calls = []
        SpyRegressor.calls = []
        with pytest.raises(DataError) as raised:
            _fit(frame, seed, **kwargs)
        # Zero fits: a single complement or a pair refused it before the first learner.
        assert Spy.calls == [] and SpyRegressor.calls == []
        messages.append(str(raised.value))
    pair = [m for m in messages if "outside folds" in m]
    assert pair, "no seed placed the rare rows in two different folds"
    assert all(factor in m and label in m for m in pair)
    assert all("at least three" in m and "cross_fit=False" in m for m in pair)


def test_in_sample_fitting_is_not_refused() -> None:
    frame = _treated_two(_frame())
    estimator = AdaptiveIPTW(
        treatment_learner=LogisticRegression(),
        reduced_outcome_learner=LinearRegression(),
        cross_fit=False,
        simultaneous=False,
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = estimator.fit(frame, outcome="Y", treatment="A", covariates=["W"]).single()
    assert fit.estimates


def _stub(**overrides: Any) -> SimpleNamespace:
    n = 6
    settings: dict[str, Any] = {
        "n": n,
        "treatment": np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
        "observed": np.ones(n, dtype=bool),
        "treatment_recorded": np.ones(n, dtype=bool),
        "has_missing_outcome": False,
        "has_missing_treatment": False,
        "known_treatment": None,
    }
    settings.update(overrides)
    return SimpleNamespace(**settings)


def test_the_predicate_reads_the_rows_outside_each_pair() -> None:
    folds = [Folds(np.array([0, 0, 1, 1, 2, 2]), 3)]
    # Arm 1 sits in folds 0, 1 and 2: every pair leaves one of them outside.
    refuse_untrainable_mechanism_pairs(_stub(), folds, treatment=True)
    # Arm 1 only in folds 0 and 1: the pair (0, 1) leaves no treated row.
    one = _stub(treatment=np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0]))
    with pytest.raises(DataError, match=r"outside folds 0 and 1.*arm 1"):
        refuse_untrainable_mechanism_pairs(one, folds, treatment=True)
    # A known mechanism, or the natural course, estimates no treatment factor.
    refuse_untrainable_mechanism_pairs(one, folds, treatment=False)
    refuse_untrainable_mechanism_pairs(
        one.__class__(**{**vars(one), "known_treatment": object()}), folds, treatment=True
    )


def test_an_arm_seen_only_with_a_missing_treatment_does_not_count() -> None:
    folds = [Folds(np.array([0, 0, 1, 1, 2, 2]), 3)]
    data = _stub(
        treatment=np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
        treatment_recorded=np.array([True, True, True, True, True, False]),
        has_missing_treatment=True,
    )
    # Arm 1 also sits at row 5 (fold 2), but that row records no treatment.
    with pytest.raises(DataError, match=r"outside folds 0 and 1.*arm 1"):
        refuse_untrainable_mechanism_pairs(data, folds, treatment=True)
