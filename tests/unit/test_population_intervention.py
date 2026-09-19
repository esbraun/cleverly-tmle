"""Composition guards for complete-data population-intervention targets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleverly import CapabilityError
from tests.conftest import fast_tmle
from tests.unit._natural_course_support import NeverFit, never_fit_learners


def _missing_frame() -> pd.DataFrame:
    rng = np.random.default_rng(31)
    n = 120
    w = rng.normal(size=n)
    a = rng.binomial(1, 1 / (1 + np.exp(-0.3 * w)))
    y = rng.binomial(1, 1 / (1 + np.exp(-(-0.2 + 0.5 * a + 0.2 * w)))).astype(float)
    delta = rng.binomial(1, 0.8, size=n)
    y[delta == 0] = np.nan
    return pd.DataFrame({"Y": y, "A": a, "W": w, "Delta": delta})


def test_an_explicit_observed_mean_target_uses_its_missing_outcome_score() -> None:
    result = (
        fast_tmle(estimands=("ey_obs",), cross_fit=False)
        .fit(
            _missing_frame(),
            outcome="Y",
            treatment="A",
            covariates=("W",),
            delta="Delta",
        )
        .single()
    )
    assert tuple(result.estimates) == ("ey_obs",)
    assert tuple(result.fluctuations) == ("natural_course",)
    assert np.isfinite(result.psi("ey_obs"))
    assert not np.isnan(result["ey_obs"].influence_curve).any()


def test_all_means_all_targets_supported_by_the_data_composition() -> None:
    # In sample: the cross-fitted MAR contract refuses the att and atc that "all"
    # includes, which the next test pins.
    result = (
        fast_tmle(estimands="all", cross_fit=False)
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


def test_a_cross_fitted_all_request_names_the_admitted_estimands() -> None:
    """``"all"`` drops the population-intervention targets and keeps att and atc.

    The cross-fitted MAR contract refuses those two by name before any learner fits,
    and it lists what the request can have instead. It drops nothing silently.
    """
    learners = never_fit_learners()
    estimator = fast_tmle(estimands="all", stratify_folds="none", **learners)

    with pytest.raises(CapabilityError) as caught:
        estimator.fit(
            _missing_frame(), outcome="Y", treatment="A", covariates=("W",), delta="Delta"
        )
    message = str(caught.value)
    assert "no audited result covers ['att', 'atc']" in message
    assert "Request estimands from ['ate', 'ey', 'ey1', 'ey0', 'rr', 'or']" in message
    assert NeverFit.calls == 0
