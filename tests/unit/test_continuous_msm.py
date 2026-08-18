"""Continuous-treatment MSM projection and density-ratio targeting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sklearn.linear_model

from cleverly.estimators import TMLE
from cleverly.estimators.serialize import dumps, loads
from cleverly.msm import MSM


def _fit():  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(11)
    n = 180
    w = rng.normal(size=n)
    a = 0.4 * w + rng.normal(size=n)
    y = 1.0 + 2.0 * a + 0.3 * w
    frame = pd.DataFrame({"Y": y, "A": a, "W": w})
    model = MSM.linear(doses=np.linspace(-1.5, 1.5, 9))
    return (
        TMLE(
            msm=model,
            outcome_learner=sklearn.linear_model.LinearRegression(),
            treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
            cross_fit=False,
            density_bins=8,
            simultaneous=False,
            random_state=3,
        )
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=("W",),
            treatment_kind="continuous",
        )
        .single()
    )


def test_continuous_msm_recovers_a_linear_dose_slope_and_solves_its_score() -> None:
    result = _fit()
    assert result["msm[a]"].psi == pytest.approx(2.0, abs=2e-6)
    np.testing.assert_allclose(result.fluctuations["msm"].score, 0.0, atol=1e-10)
    assert abs(result["msm[a]"].score) < 1e-10


def test_continuous_msm_round_trip_preserves_the_integration_measure() -> None:
    result = _fit()
    back = loads(dumps(result))
    assert back.nuisance.msm is not None and result.nuisance.msm is not None
    assert back.nuisance.msm.dose_values == result.nuisance.msm.dose_values
    np.testing.assert_array_equal(back.nuisance.msm.weights, result.nuisance.msm.weights)
    np.testing.assert_array_equal(
        back.nuisance.msm.clever_weights, result.nuisance.msm.clever_weights
    )
    assert back["msm[a]"].psi == result["msm[a]"].psi
