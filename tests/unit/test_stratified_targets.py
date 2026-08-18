"""Finite baseline strata: joint scores, conditional targets and persistence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleverly import ATE, CapabilityError, CausalStudy, PointTreatment
from cleverly.data import CausalData
from cleverly.estimators import TMLE
from cleverly.estimators.serialize import dumps, loads
from cleverly.exceptions import DataError
from cleverly.sensitivity._parameters import arm_parameters


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


def _study_fit():  # type: ignore[no-untyped-def]
    """The same design through ``CausalStudy``, which is what fills ``parameter_keys``.

    ``_fit`` above calls the estimator directly and leaves ``parameter_keys`` empty, so
    the structured routing these tests are about is only reachable from here.
    """
    study = CausalStudy(
        _frame(),
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W", "V"),
            strata=("V",),
        ),
    )
    return study.identify(ATE()).estimate(
        outcome_learner="glm",
        treatment_learner="glm",
        n_folds=3,
        learner_folds=2,
        random_state=2,
        simultaneous=False,
    )


def _study_fit_with_missingness():  # type: ignore[no-untyped-def]
    """A stratified fit that also carries an observation mechanism, so the tilt applies."""
    rng = np.random.default_rng(7)
    frame = _frame(320)
    keep = rng.binomial(1, 1.0 / (1.0 + np.exp(-0.5 - 0.3 * frame["W"].to_numpy())))
    frame = frame.assign(Delta=keep.astype(float))
    frame.loc[frame["Delta"] == 0.0, "Y"] = np.nan
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W", "V"),
            strata=("V",),
            missingness="Delta",
        ),
    )
    return study.identify(ATE()).estimate(
        outcome_learner="glm",
        treatment_learner="glm",
        n_folds=3,
        learner_folds=2,
        random_state=2,
        simultaneous=False,
    )


class TestSensitivityRefusesAConditionalParameterRatherThanApproximatingIt:
    """A stratum key copies the marginal one; only ``stratum`` tells them apart.

    ``CausalStudy._point_parameter_keys`` mints ``ate[V='low']`` from the marginal ``ate``
    key with ``replace(alias=..., stratum=...)``, so arm, reference, axis and estimand are
    all identical. A resolver that reads everything *except* ``stratum`` therefore answers
    the marginal question under the conditional name -- and silently, since the alias is a
    real reported parameter rather than a typo.
    """

    def test_the_conditional_aliases_are_not_arm_parameters(self) -> None:
        result = _study_fit()
        assert "ate[V='low']" in result.estimates
        known = arm_parameters(result)
        assert "ate" in known
        assert not [name for name in known if "V=" in name]

    def test_the_bound_refuses_the_conditional_parameter_by_name(self) -> None:
        result = _study_fit()
        with pytest.raises(CapabilityError, match="conditional on a baseline stratum"):
            result.sensitivity.omitted_confounding(estimand="ate[V='low']")

    def test_the_refusal_is_not_the_never_requested_one(self) -> None:
        result = _study_fit()
        with pytest.raises(CapabilityError) as caught:
            result.sensitivity.omitted_confounding(estimand="ate[V='low']")
        # The fit *did* report it, so the coverage message would be a false statement
        # about the sample rather than about what is derived.
        assert "not requested in this fit" not in str(caught.value)
        assert "'ate'" in str(caught.value)

    def test_the_marginal_bound_still_answers(self) -> None:
        result = _study_fit()
        marginal = result.sensitivity.omitted_confounding(estimand="ate")
        assert marginal.max_bias >= 0.0

    def test_the_mnar_tilt_refuses_an_explicitly_named_conditional_parameter(self) -> None:
        result = _study_fit_with_missingness()
        assert "ate[V='low']" in result.estimates
        with pytest.raises(CapabilityError, match="conditional on a baseline stratum"):
            result.sensitivity.missingness(estimands=("ate[V='low']",))

    def test_the_default_tilt_sweep_reports_only_the_marginal_parameters(self) -> None:
        """The default sweep skips what it cannot tilt rather than refusing the call."""
        result = _study_fit_with_missingness()
        frame = result.sensitivity.missingness()
        assert not [name for name in set(frame["estimand"]) if "V=" in name]
        assert "ate" in set(frame["estimand"])

    def test_a_conditional_alias_does_not_borrow_the_marginal_bound(self) -> None:
        """The witness for the defect: the two answers were bit-identical."""
        result = _study_fit()
        marginal = result.sensitivity.omitted_confounding(estimand="ate")
        assert result["ate[V='low']"].psi != pytest.approx(result["ate"].psi, abs=1e-9)
        with pytest.raises(CapabilityError):
            result.sensitivity.omitted_confounding(estimand="ate[V='low']")
        assert marginal.max_bias == result.sensitivity.omitted_confounding(estimand="ate").max_bias


def test_stratum_metadata_and_estimates_survive_round_trip() -> None:
    result = _fit(_frame())
    back = loads(dumps(result))
    np.testing.assert_array_equal(back.data.strata, result.data.strata)
    assert back.data.strata_names == result.data.strata_names
    assert back.data.strata_levels == result.data.strata_levels
    for name in result.estimates:
        assert back[name].psi == result[name].psi
        np.testing.assert_array_equal(back[name].influence_curve, result[name].influence_curve)
