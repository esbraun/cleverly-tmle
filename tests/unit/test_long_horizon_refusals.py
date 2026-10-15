"""The two refusals the long-horizon studies rely on, at four and five decisions.

Each refusal fires before any learner is fitted: every learner here raises on ``fit``.  Each has a
control that the same call answers when the refused part is absent.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sklearn.base import BaseEstimator

from cleverly.exceptions import CapabilityError
from cleverly.interventions import Incremental
from cleverly.longitudinal import LTMLE, SDR, DynamicRegimen
from cleverly.longitudinal.regimen import _RULE_DECLARATION
from cleverly.methods import SDR_TILT_REFUSAL
from tests import discrete_law_longitudinal_horizon as H
from tests.studies import long_horizon_common as C


class Exploding(BaseEstimator):
    """A learner that fails the test if a fit reaches it."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> Exploding:
        raise AssertionError("a learner was fitted before the refusal")

    def predict(self, X: Any) -> np.ndarray:  # pragma: no cover - never reached
        raise AssertionError("a learner predicted before the refusal")

    def predict_proba(self, X: Any) -> np.ndarray:  # pragma: no cover - never reached
        raise AssertionError("a learner predicted before the refusal")


def _settings() -> dict[str, Any]:
    return {
        "outcome_learner": Exploding(),
        "pseudo_learner": Exploding(),
        "treatment_learner": Exploding(),
        "censoring_learner": Exploding(),
        "n_folds": 1,
        "simultaneous": False,
    }


def _rule(history: Any) -> np.ndarray:
    return (np.asarray(history["L4"], dtype=float) == 1.0).astype(float)


@pytest.fixture(scope="module")
def frame() -> Any:
    return H.draw(5, 2, 400, 0)


def test_sdr_tilt_at_node_five_refuses_before_any_learner(frame: Any) -> None:
    """An odds tilt at the last of five nodes, beside labels: ``SDR_TILT_REFUSAL``."""
    plan = DynamicRegimen("tilt at five", (1, 1, 1, 1, Incremental(2.0)))
    with pytest.raises(CapabilityError) as raised:
        SDR({"always": 1, "tilt at five": plan}, **_settings()).fit(frame, **H.roles(5, 2))
    assert str(raised.value) == SDR_TILT_REFUSAL


def test_an_undeclared_rule_at_node_four_refuses(frame: Any) -> None:
    """A callable at node 4 with no ``rule_kind``: refused when built and when fitted."""
    spec = (1, 1, 1, _rule, 1)
    with pytest.raises(CapabilityError) as built:
        DynamicRegimen("rule at four", spec)
    assert str(built.value) == _RULE_DECLARATION.undeclared
    with pytest.raises(CapabilityError) as fitted:
        LTMLE({"always": 1, "rule at four": spec}, **_settings()).fit(frame, **H.roles(5, 2))
    assert str(fitted.value) == _RULE_DECLARATION.undeclared


def test_the_declared_rule_at_node_four_fits(frame: Any) -> None:
    """The control: the same rule with ``rule_kind="known"`` passes every pre-fit check."""
    plan = DynamicRegimen("rule at four", (1, 1, 1, _rule, 1), rule_kind="known")
    result = C.fit(frame, 5, 2, (), regimens={"always": 1, "rule at four": plan})
    assert np.isfinite(float(result["ey_regimen[rule at four]"].psi))
