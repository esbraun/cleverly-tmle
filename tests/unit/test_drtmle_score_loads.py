"""A doubly-robust fit records one absolute score-weight column per reported outcome equation.

The closing pass solves equations (8) and (10) jointly over a stacked submodel, then splits the
joint fluctuation into the outcome block and the extra block. The split used to cut ``epsilon``,
``names`` and the scores but keep the stacked ``absolute_score_weights``. So the outcome block
carried four columns for its two equations, and the support report omitted the group
diagnostic with ``SCORE_LOAD_SHAPE_MISMATCH`` on every complete-data DR-TMLE fit that ran the
``g`` guard, with a fitted or a known treatment mechanism alike. The missing-outcome
construction split its stacked solve the same way.

The witness is nonzero: each fit below runs the stacked solve, so a split that keeps the
stacked block has more columns than names and fails the shape assertion. The ``("Q",)`` guard
skips the stacked solve and is kept as the control that the assertion holds without it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sklearn.linear_model

from cleverly.data.weighting import SCORE_LOAD_SHAPE_MISMATCH
from cleverly.estimators import DRTMLE


def _frame(n: int = 300, seed: int = 7, *, missing: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w1 = rng.normal(size=n)
    w2 = rng.normal(size=n)
    g = 1.0 / (1.0 + np.exp(-(0.2 + 0.6 * w1)))
    a = rng.binomial(1, g).astype(float)
    y = 1.0 / (1.0 + np.exp(-(0.3 + 0.8 * a + 0.5 * w1 - 0.4 * w2 * w1)))
    y = np.clip(y + rng.normal(scale=0.05, size=n), 0.0, 1.0)
    frame = pd.DataFrame({"W1": w1, "W2": w2, "A": a, "Y": y, "g1": g})
    if missing:
        pi = 1.0 / (1.0 + np.exp(-(0.4 + 0.3 * a + 0.3 * w1)))
        observed = rng.binomial(1, pi).astype(float)
        frame["Delta"] = observed
        frame.loc[observed == 0.0, "Y"] = np.nan
    return frame


def _estimator(**settings: object) -> DRTMLE:
    defaults: dict[str, object] = {
        "cross_fit": False,
        "outcome_learner": sklearn.linear_model.LinearRegression(),
        "treatment_learner": sklearn.linear_model.LogisticRegression(max_iter=1000),
        "reduced_outcome_learner": sklearn.linear_model.LinearRegression(),
        "reduced_mechanism_residual_learner": sklearn.linear_model.LinearRegression(),
        "reduced_treatment_learner": sklearn.linear_model.LogisticRegression(max_iter=1000),
        "q_bounds": (0.0, 1.0),
        "simultaneous": False,
        "random_state": 0,
        "estimands": ("ate", "ey1", "ey0"),
    }
    return DRTMLE(**{**defaults, **settings})  # type: ignore[arg-type]


def _assert_one_column_per_equation(result) -> None:
    fluctuation = result.repeats[0].fluctuations["mean"]
    loads = fluctuation.absolute_score_weights
    assert loads is not None
    assert loads.shape[1] == len(fluctuation.names)
    assert np.asarray(fluctuation.epsilon).shape == (len(fluctuation.names),)
    if fluctuation.score_initial is not None:
        assert np.asarray(fluctuation.score_initial).shape == (len(fluctuation.names),)
    if fluctuation.epsilon_std_error is not None:
        assert np.asarray(fluctuation.epsilon_std_error).shape == (len(fluctuation.names),)

    support = result.diagnostics.support()
    assert support.group_score_load_omissions.get("mean") != SCORE_LOAD_SHAPE_MISMATCH
    assert "mean" in support.group_score_load
    assert "No equation row for mean:" not in support.summary()


@pytest.mark.parametrize("guard", [("Q", "g"), ("g",), ("Q",)])
def test_fitted_mechanism_records_outcome_score_loads(guard: tuple[str, ...]) -> None:
    frame = _frame()
    result = (
        _estimator(guard=guard)
        .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"])
        .single()
    )
    _assert_one_column_per_equation(result)


def test_known_mechanism_records_outcome_score_loads() -> None:
    """The case the DR-TMLE tutorial review found: probabilities declared on the data."""
    frame = _frame()
    result = (
        _estimator()
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2"],
            treatment_probabilities=frame["g1"].to_numpy(),
        )
        .single()
    )
    _assert_one_column_per_equation(result)


def test_missing_outcome_records_outcome_score_loads() -> None:
    frame = _frame(missing=True)
    result = (
        _estimator(
            randomized=True,
            missingness_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"], delta="Delta")
        .single()
    )
    _assert_one_column_per_equation(result)
