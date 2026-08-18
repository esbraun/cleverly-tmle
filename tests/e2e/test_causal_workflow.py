"""The first causal-workflow vertical slice through the existing analytic engine."""

import numpy as np
import pytest
import sklearn.linear_model
from joblib import hash as joblib_hash

from cleverly import (
    ATE,
    CausalStudy,
    LongitudinalTreatment,
    PointTreatment,
    RegimeMean,
    TMLEMethod,
    load,
)
from cleverly.datasets import make_linear_ate, make_longitudinal, make_multi_arm
from cleverly.estimators import TMLE, TMLEResult
from tests.conftest import FAST_KWARGS


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_the_new_binary_ate_path_is_bit_for_bit_the_existing_fit(backend: str, tmp_path) -> None:
    frame, _ = make_linear_ate(n=500, seed=31, backend=backend)
    adjustment = ["W1", "W2", "W3", "W4"]
    legacy = (
        TMLE(estimands=("ate",), **FAST_KWARGS)
        .fit(frame, outcome="Y", treatment="A", covariates=adjustment)
        .single()
    )

    effect = CausalStudy(
        frame,
        design=PointTreatment(outcome="Y", treatment="A", adjustment=adjustment),
    ).identify(ATE())
    result = effect.estimate(**FAST_KWARGS)

    assert isinstance(result, TMLEResult)
    assert not hasattr(result, "single")
    assert result.psi() == legacy.psi("ate")
    assert result.estimate == result["ate"]
    assert result["ate"].variance == legacy["ate"].variance
    np.testing.assert_array_equal(
        result["ate"].influence_curve,
        legacy["ate"].influence_curve,
    )
    assert result.identified_effect is effect
    assert isinstance(result.method, TMLEMethod)
    assert result.parameter_keys["ate"].treatment == 1
    assert result.parameter_keys["ate"].reference == 0
    assert "identification assumptions" in result.summary()
    saved = result.save(tmp_path / "causal-result.joblib")
    restored = load(saved)
    assert restored.parameter_keys == result.parameter_keys
    assert joblib_hash(restored.method) == joblib_hash(result.method)
    assert restored.identified_effect.summary() == effect.summary()


def test_structured_keys_are_composed_from_multi_arm_labels() -> None:
    frame, _ = make_multi_arm(n=600, seed=32)
    result = (
        CausalStudy(
            frame,
            design=PointTreatment(
                outcome="Y",
                treatment="A",
                adjustment=("W1", "W2", "W3"),
            ),
        )
        .identify(ATE(reference="medium"))
        .estimate(**FAST_KWARGS)
    )

    assert set(result.parameter_keys) == {"ate[high vs medium]", "ate[low vs medium]"}
    high = result.parameter_keys["ate[high vs medium]"]
    assert high.treatment == "high"
    assert high.reference == "medium"


def test_longitudinal_causal_metadata_and_inference_survive_persistence(tmp_path) -> None:
    frame, _ = make_longitudinal(n=240, seed=33)
    effect = CausalStudy(
        frame,
        design=LongitudinalTreatment(
            outcome="Y",
            treatment=("A1", "A2"),
            baseline=("W1", "W2"),
            time_varying=((), ("L2",)),
            censoring=("C1", "C2"),
        ),
    ).identify(RegimeMean({"always": 1, "never": 0}, reference="always"))
    result = effect.estimate(
        outcome_learner=sklearn.linear_model.LinearRegression(),
        pseudo_learner=sklearn.linear_model.LinearRegression(),
        treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
        n_folds=3,
        learner_folds=3,
        random_state=0,
        simultaneous=False,
    )

    restored = load(result.save(tmp_path / "longitudinal-causal.joblib"))

    assert restored.parameter_keys == result.parameter_keys
    assert joblib_hash(restored.method) == joblib_hash(result.method)
    assert restored.identified_effect.summary() == effect.summary()
    assert list(restored.estimates) == list(result.estimates)
    for name in result.estimates:
        assert restored[name].psi == result[name].psi
        assert restored[name].ci == result[name].ci
        np.testing.assert_array_equal(
            restored[name].influence_curve,
            result[name].influence_curve,
        )
