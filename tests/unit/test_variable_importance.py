"""Exposure-wise orchestration and corrected multiplicity inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from cleverly import variable_importance
from cleverly.estimators import TMLE
from cleverly.variable_importance import _bh_adjust


def test_bh_adjustment_is_monotone_in_rank_and_uses_both_tails() -> None:
    adjusted = _bh_adjust((0.04, 0.01, 0.03))
    np.testing.assert_allclose(adjusted, (0.04, 0.03, 0.04))


def test_each_candidate_gets_its_own_declared_fit() -> None:
    rng = np.random.default_rng(5)
    n = 180
    w = rng.normal(size=n)
    x1 = rng.binomial(1, 1.0 / (1.0 + np.exp(-w)))
    x2 = rng.binomial(1, 0.5, size=n)
    y = 1.2 * x1 + 0.2 * w + rng.normal(scale=0.5, size=n)
    frame = pd.DataFrame({"Y": y, "X1": x1, "X2": x2, "W": w})
    result = variable_importance(
        frame,
        outcome="Y",
        candidates=("X1", "X2"),
        covariates=("W",),
        estimator=TMLE(
            outcome_learner="glm",
            treatment_learner="glm",
            cross_fit=False,
            simultaneous=False,
            random_state=1,
        ),
    )
    assert set(result.fits) == {"X1", "X2"}
    entries = {entry.candidate: entry for entry in result}
    assert entries["X1"].adjustment_set == ("W", "X2")
    assert entries["X2"].adjustment_set == ("W", "X1")
    assert entries["X1"].estimate.pvalue <= entries["X1"].adjusted_pvalue
    assert entries["X1"].estimate.psi > entries["X2"].estimate.psi


def test_the_summary_preserves_a_polars_callers_backend() -> None:
    rng = np.random.default_rng(9)
    n = 80
    frame = pl.DataFrame(
        {
            "Y": rng.normal(size=n),
            "X1": rng.binomial(1, 0.5, size=n),
            "X2": rng.binomial(1, 0.5, size=n),
            "W": rng.normal(size=n),
        }
    )
    result = variable_importance(
        frame,
        outcome="Y",
        candidates=("X1", "X2"),
        covariates=("W",),
        estimator=TMLE(
            outcome_learner="glm",
            treatment_learner="glm",
            cross_fit=False,
            simultaneous=False,
            random_state=4,
        ),
    )
    assert isinstance(result.to_frame(), pl.DataFrame)
