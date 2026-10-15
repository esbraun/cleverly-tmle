"""Repeated cross-fitting on the sequential baselines: the refusal and its witnesses.

``repeats=R`` combines ``R`` complete cross-fitted draws by the median rule of Chernozhukov et
al. (2018, Definition 3.3).  :class:`SequentialGComputation` and :class:`SequentialIPW` fit in
sample, and their interval is the stacked estimating equation of one parametric fit, so no
split exists to draw again.  Both refuse ``repeats`` above 1 with the F35 implementation gap.
Repeated draws on the cross-fitted engines stay in ``test_longitudinal_repeats.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from cleverly import (
    CausalStudy,
    GComputationMethod,
    IPWMethod,
    LongitudinalTreatment,
    RegimeContrast,
)
from cleverly._refusal import RefusalReason
from cleverly.datasets import make_longitudinal
from cleverly.exceptions import CapabilityError
from cleverly.longitudinal import LTMLE, SequentialGComputation, SequentialIPW
from cleverly.longitudinal.estimator import baseline_repeats_refusal

PAIR = {"always": 1, "never": 0}
COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W1", "W2"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
}


@pytest.mark.parametrize("cls", [SequentialGComputation, SequentialIPW])
@pytest.mark.parametrize("n_folds", [1, 2])
def test_a_baseline_refuses_repeats_above_one(cls: Any, n_folds: int) -> None:
    with pytest.raises(CapabilityError) as raised:
        cls(PAIR, repeats=2, n_folds=n_folds)
    assert raised.value.refusal == baseline_repeats_refusal(cls.__name__)
    assert raised.value.refusal.reason is RefusalReason.IMPLEMENTATION_GAP
    assert raised.value.refusal.roadmap == "F35"


def test_the_refusal_names_the_estimator_and_a_working_call() -> None:
    refusal = baseline_repeats_refusal("SequentialIPW")
    assert refusal.condition.startswith("SequentialIPW fits in sample")
    assert "repeats=1" in (refusal.alternative or "")
    assert "LTMLE" in (refusal.alternative or "")


@pytest.mark.parametrize("cls", [SequentialGComputation, SequentialIPW])
def test_a_baseline_accepts_repeats_of_one(cls: Any) -> None:
    assert cls(PAIR, repeats=1).repeats == 1


def test_a_baseline_fit_is_one_draw() -> None:
    frame, _ = make_longitudinal(n=150, seed=3)
    for cls in (SequentialGComputation, SequentialIPW):
        result = cls(PAIR, reference="never", simultaneous=False).fit(frame, **COLUMNS)
        assert result.n_repeats == 1
        assert result.config.repeats == 1
        assert result.config.multiplier_seed is None


def test_the_refusal_does_not_reach_the_cross_fitted_engines() -> None:
    assert LTMLE(PAIR, repeats=2, n_folds=2).repeats == 2


@pytest.mark.parametrize("method_cls", [GComputationMethod, IPWMethod])
def test_a_baseline_method_with_repeats_reaches_the_engine_refusal(method_cls: Any) -> None:
    from cleverly import CrossFitting

    frame, _ = make_longitudinal(n=150, seed=3)
    study = CausalStudy(
        frame, design=LongitudinalTreatment(**{**COLUMNS, "time_varying": ((), ("L2",))})
    )
    effect = study.identify(RegimeContrast(PAIR, reference="never"))
    method = method_cls(cross_fitting=CrossFitting(n_folds=3, repeats=2))
    with pytest.raises(CapabilityError) as raised:
        effect.estimate(method=method)
    name = "SequentialGComputation" if method_cls is GComputationMethod else "SequentialIPW"
    assert raised.value.refusal == baseline_repeats_refusal(name)
