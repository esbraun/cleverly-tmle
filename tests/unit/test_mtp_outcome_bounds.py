"""A continuous outcome under a modified treatment policy declares its support (X40 part (f)).

``X40-plan.md`` section 1.2, R-f1 as amended in review m4.  An ``LTMLE`` or ``SDR`` fit
with a modified-treatment-policy node refuses a continuous outcome with ``q_bounds=None``,
in sample as well as cross-fitted, before any learner, and raises ``LongitudinalError``,
the class of its cross-fitted sibling.  The point ``TMLE`` half of R-f1 is not enforced
yet: the Gaussian outcome of ``make_shift_dose`` has no known support to declare.  With
declared bounds an outcome outside them raises from ``OutcomeScaler.from_outcome`` and is
never clipped, on the longitudinal and the point fits alike (T-f1); M-f1 makes the scaler
clip and shows that T-f1 then fails."""

from __future__ import annotations

import re
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.estimators import TMLE
from cleverly.exceptions import LongitudinalError
from cleverly.interventions import Shift
from cleverly.longitudinal import LTMLE, SDR, DynamicRegimen
from cleverly.utils import bounds as bounds_module
from cleverly.utils.bounds import OutcomeScaler

from .. import law_q_point as lawp
from .. import law_q_tuned as lawq

LONGITUDINAL: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W"],
    "time_varying": [[], ["L2"]],
    "continuous_treatment": ["A1", "A2"],
}
POINT: dict[str, Any] = {
    "outcome": "Y",
    "treatment": "A",
    "covariates": ["W"],
    "treatment_kind": "continuous",
}
MESSAGE = (
    "a continuous outcome ('Y') under a modified treatment policy needs q_bounds equal to its "
    "known support. The estimator of Díaz et al. (2023), Section 5.2, maps the outcome onto "
    "[0, 1] with known bounds, and a policy on a continuous dose moves rows toward doses "
    "where few outcomes were recorded, so the bounds must be the support the outcome can "
    "take, not the range this sample shows. Declare q_bounds=(lower, upper); the fit raises "
    "on an outcome outside them and never clips one"
)


class _Raises(BaseEstimator, RegressorMixin):
    """A learner that fails if a fit reaches it."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> _Raises:
        raise AssertionError("a learner was fitted before the refusal")

    def predict(self, X: Any) -> Any:  # pragma: no cover - never reached
        raise AssertionError("unreachable")


def _regimens() -> list[DynamicRegimen]:
    shift = Shift(0.5, cap=None)
    return [DynamicRegimen("up", (shift, shift))]


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return lawq.sample(400, 3)


@pytest.fixture(scope="module")
def point() -> pd.DataFrame:
    return lawp.sample(400, 3)


def _longitudinal(estimator: type, frame: pd.DataFrame, **settings: Any) -> Any:
    options = {
        "outcome_learner": LinearRegression(),
        "pseudo_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=500),
        "density_bins": 10,
        "random_state": 1,
        "simultaneous": False,
        **settings,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator(_regimens(), **options).fit(frame, **LONGITUDINAL)


def _point(frame: pd.DataFrame, **settings: Any) -> Any:
    options = {
        "outcome_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=500),
        "density_bins": 10,
        "random_state": 1,
        "simultaneous": False,
        "policies": [Shift(0.5, cap=None), Shift(0.0, cap=None)],
        **settings,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TMLE(**options).fit(frame, **POINT)


@pytest.mark.parametrize("estimator", [LTMLE, SDR])
@pytest.mark.parametrize("folds", [1, 3])
def test_t_f1_r_f1_a_longitudinal_policy_fit_declares_the_outcome_support(
    estimator: type, folds: int, panel: Any
) -> None:
    """R-f1 on LTMLE and SDR: the class of the cross-fitted sibling, before any learner."""
    raising = {"outcome_learner": _Raises(), "pseudo_learner": _Raises()}
    with pytest.raises(LongitudinalError, match=f"^{re.escape(MESSAGE)}$"):
        _longitudinal(estimator, panel, n_folds=folds, treatment_learner=_Raises(), **raising)


def test_r_f1_leaves_a_binary_outcome_alone(panel: Any) -> None:
    """R-f1 reads a continuous outcome, so a binary outcome under the same policy fits."""
    binary = panel.assign(Y=(panel["Y"] > 0.5).astype(float))
    assert "ey_regimen[up]" in _longitudinal(LTMLE, binary, n_folds=1).estimates


def _outside(frame: pd.DataFrame) -> pd.DataFrame:
    """The frame with one outcome above the declared upper bound of 1."""
    out = frame.copy()
    out.loc[0, "Y"] = 1.25
    return out


def _raises_outside(fit: Any) -> bool:
    try:
        fit()
    except ValueError as error:
        return "observed outcomes fall outside q_bounds" in str(error)
    return False


def _fits(panel: pd.DataFrame, point: pd.DataFrame) -> list[Any]:
    bounded = {"q_bounds": (0.0, 1.0)}
    return [
        lambda: _longitudinal(LTMLE, _outside(panel), n_folds=1, **bounded),
        lambda: _longitudinal(SDR, _outside(panel), n_folds=3, **bounded),
        lambda: _point(_outside(point), cross_fit=False, **bounded),
        lambda: _point(_outside(point), cross_fit=True, **bounded),
    ]


def test_t_f1_an_outcome_outside_the_declared_bounds_raises(panel: Any, point: Any) -> None:
    """T-f1: with declared bounds, an outcome outside them raises and is never clipped."""
    assert all(_raises_outside(fit) for fit in _fits(panel, point))
    # The same fits inside the bounds run.
    assert "ey_regimen[up]" in _longitudinal(LTMLE, panel, n_folds=1, q_bounds=(0.0, 1.0)).estimates


def test_m_f1_a_clipping_scaler_fails_t_f1(
    panel: Any, point: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M-f1: the scaler clips the outcome into the bounds, and T-f1 catches it."""
    original = OutcomeScaler.from_outcome.__func__  # type: ignore[attr-defined]

    def clipping(cls: Any, y: Any, bounds: Any = None, **kwargs: Any) -> Any:
        if bounds is not None:
            y = np.clip(np.asarray(y, dtype=float), *bounds)
        return original(cls, y, bounds, **kwargs)

    monkeypatch.setattr(bounds_module.OutcomeScaler, "from_outcome", classmethod(clipping))
    assert not any(_raises_outside(fit) for fit in _fits(panel, point))
