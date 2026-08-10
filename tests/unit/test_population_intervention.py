"""Composition guards for complete-data population-intervention targets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import fast_tmle


def _missing_frame() -> pd.DataFrame:
    rng = np.random.default_rng(31)
    n = 120
    w = rng.normal(size=n)
    a = rng.binomial(1, 1 / (1 + np.exp(-0.3 * w)))
    y = rng.binomial(1, 1 / (1 + np.exp(-(-0.2 + 0.5 * a + 0.2 * w)))).astype(float)
    delta = rng.binomial(1, 0.8, size=n)
    y[delta == 0] = np.nan
    return pd.DataFrame({"Y": y, "A": a, "W": w, "Delta": delta})


def test_an_explicit_observed_mean_target_refuses_missing_outcomes() -> None:
    with pytest.raises(NotImplementedError, match="natural-course mean"):
        fast_tmle(estimands=("ey_obs",)).fit(
            _missing_frame(),
            outcome="Y",
            treatment="A",
            covariates=("W",),
            delta="Delta",
        )


def test_all_means_all_targets_supported_by_the_data_composition() -> None:
    result = (
        fast_tmle(estimands="all")
        .fit(
            _missing_frame(),
            outcome="Y",
            treatment="A",
            covariates=("W",),
            delta="Delta",
        )
        .single()
    )
    assert {"ey_obs", "par", "paf"}.isdisjoint(result.estimates)
    assert {"ey1", "ey0", "ate", "att", "atc", "rr", "or"}.issubset(result.estimates)
