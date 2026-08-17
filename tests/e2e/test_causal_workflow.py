"""The first causal-workflow vertical slice through the existing analytic engine."""

import numpy as np
import pytest

from cleverly import (
    ATE,
    TMLE,
    CapabilityError,
    CausalStudy,
    PointTreatment,
    TMLEMethod,
    TMLEResult,
)
from cleverly.datasets import make_linear_ate, make_multi_arm
from tests.conftest import FAST_KWARGS


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_the_new_binary_ate_path_is_bit_for_bit_the_existing_fit(backend: str) -> None:
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
    with pytest.raises(CapabilityError, match="cannot store structured identification"):
        result.save("not-written.npz")


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
