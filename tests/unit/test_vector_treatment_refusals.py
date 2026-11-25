"""Refusals that name what an all-categorical vector treatment accepts.

An all-categorical vector is one categorical treatment over the observed tuples of its
columns. Two refusals used to point elsewhere: ``MSM.linear`` read a tuple level as a dose and
raised an unrelated length error, and the 20-level limit suggested ``treatment_kind=``, which a
vector refuses. Each test checks the refusal's type, code and the alternative it names.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleverly import ATE, CausalStudy, MSMProjection, PointTreatment
from cleverly.data import CausalData
from cleverly.exceptions import CapabilityError, DataError, RefusalReason
from cleverly.msm import MSM


def _frame(n: int = 300, coupon_levels: int = 2, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    a = rng.integers(0, 2, n)
    coupon = np.arange(n) % coupon_levels
    y = rng.binomial(1, 1 / (1 + np.exp(-(0.3 * a + 0.2 * w))))
    return pd.DataFrame({"Y": y.astype(float), "A": a, "coupon": coupon, "W": w})


def _study(frame: pd.DataFrame) -> CausalStudy:
    return CausalStudy(
        frame, design=PointTreatment(outcome="Y", treatment=("A", "coupon"), adjustment=("W",))
    )


def test_msm_linear_on_tuple_levels_refuses_at_identify() -> None:
    with pytest.raises(CapabilityError) as caught:
        _study(_frame()).identify(MSMProjection(MSM.linear()))
    error = caught.value
    assert error.refusal.reason is RefusalReason.INVALID_DECLARATION
    message = str(error)
    assert "treatment 'A+coupon' is a combination of categorical columns" in message
    assert "no numeric dose" in message
    assert "design_kind='known'" in message
    assert "CounterfactualMean or ATE" in message
    assert "a dose must be finite" not in message


def test_a_written_design_on_tuple_levels_is_not_refused_at_identify() -> None:
    """Deliberate-mutation control: the refusal reads the design of MSM.linear only."""
    model = MSM(
        design=lambda level, frame: np.column_stack(
            [np.ones(len(frame)), np.full(len(frame), float(level[0]))]
        ),
        terms=("(intercept)", "A"),
        design_kind="known",
    )
    _study(_frame()).identify(MSMProjection(model))
    _study(_frame()).identify(ATE())


def test_msm_linear_design_refuses_a_tuple_level_directly() -> None:
    design = MSM.linear().design
    with pytest.raises(CapabilityError) as caught:
        design((1, 0), pd.DataFrame({"W": [0.0, 1.0]}))
    assert caught.value.refusal.reason is RefusalReason.INVALID_DECLARATION
    assert "the treatment with the level (1, 0)" in str(caught.value)
    np.testing.assert_array_equal(design(1.0, pd.DataFrame({"W": [0.0, 1.0]}))[:, 1], [1.0, 1.0])


def test_too_many_tuples_suggest_what_a_vector_accepts() -> None:
    frame = _frame(n=440, coupon_levels=11)
    with pytest.raises(DataError) as caught:
        _study(frame)
    message = str(caught.value)
    assert "22 distinct tuples" in message
    assert "treatment_kind='continuous'" not in message
    assert "fewer combinations" in message
    assert "continuous_treatment=[...]" in message
    assert "ModifiedTreatmentPolicyEffect" in message


def test_the_container_gives_the_same_vector_remedy() -> None:
    frame = _frame(n=440, coupon_levels=11)
    with pytest.raises(DataError, match="fewer combinations") as caught:
        CausalData.from_arrays(
            frame[["W"]].to_numpy(),
            frame[["A", "coupon"]].to_numpy(),
            frame["Y"].to_numpy(),
            treatment_name=["A", "coupon"],
        )
    assert "treatment_kind='continuous'" not in str(caught.value)
