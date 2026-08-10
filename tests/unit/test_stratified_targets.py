"""Finite baseline strata: joint scores, conditional targets and persistence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleverly import TMLE
from cleverly.data import CausalData
from cleverly.estimators.serialize import dumps, loads
from cleverly.exceptions import DataError


def _frame(n: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(18)
    v = np.repeat(["low", "high"], n // 2)
    w = rng.normal(size=n)
    g = 1.0 / (1.0 + np.exp(-0.3 * w + 0.4 * (v == "high")))
    a = rng.binomial(1, g)
    y = 0.2 + (1.0 + 0.8 * (v == "high")) * a + 0.3 * w + rng.normal(scale=0.2, size=n)
    return pd.DataFrame({"Y": y, "A": a, "W": w, "V": v})


def _fit(frame: pd.DataFrame):  # type: ignore[no-untyped-def]
    return (
        TMLE(
            outcome_learner="glm",
            treatment_learner="glm",
            cross_fit=False,
            estimands=("ate", "att", "ey_obs", "par"),
            simultaneous=False,
            random_state=2,
        )
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=("W", "V"),
            strata=("V",),
        )
        .single()
    )


def test_joint_targeting_returns_marginal_and_conditional_parameters() -> None:
    result = _fit(_frame())
    assert {"ate", "ate[V='low']", "ate[V='high']"}.issubset(result.estimates)
    # Two arms x two strata for the mean group; one ATT column x two strata.
    assert result.fluctuations["mean"].epsilon.shape == (4,)
    assert result.fluctuations["att"].epsilon.shape == (2,)
    np.testing.assert_allclose(result.fluctuations["mean"].score, 0.0, atol=1e-10)
    np.testing.assert_allclose(result.fluctuations["att"].score, 0.0, atol=1e-10)
    assert abs(result["ate[V='low']"].score) < 1e-10
    assert abs(result["att[V='high']"].score) < 1e-10


def test_marginal_point_estimate_is_the_empirical_stratum_mixture() -> None:
    result = _fit(_frame())
    data = result.data
    assert data.strata is not None
    mixture = sum(
        np.average(data.strata == code, weights=data.weights)
        * result[f"ate[{data.stratum_label(code)}]"].psi
        for code in range(data.n_strata)
    )
    assert result["ate"].psi == pytest.approx(mixture, abs=1e-12)


def test_strata_must_remain_in_the_adjustment_set() -> None:
    with pytest.raises(DataError, match="also be adjustment covariates"):
        TMLE(cross_fit=False).fit(
            _frame(), outcome="Y", treatment="A", covariates=("W",), strata=("V",)
        )


def test_array_strata_must_define_a_nontrivial_partition() -> None:
    with pytest.raises(DataError, match="only one baseline stratum"):
        CausalData.from_arrays(
            outcome=np.arange(30.0),
            treatment=np.tile((0, 1), 15),
            covariates=np.arange(30.0)[:, None],
            strata=np.zeros(30),
            strata_names=("V",),
        )


def test_stratum_metadata_and_estimates_survive_round_trip() -> None:
    result = _fit(_frame())
    back = loads(dumps(result))
    np.testing.assert_array_equal(back.data.strata, result.data.strata)
    assert back.data.strata_names == result.data.strata_names
    assert back.data.strata_levels == result.data.strata_levels
    for name in result.estimates:
        assert back[name].psi == result[name].psi
        np.testing.assert_array_equal(back[name].influence_curve, result[name].influence_curve)
