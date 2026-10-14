r"""T6 and T7: SDR regresses its transforms unclipped, and bounds partial path probabilities.

T6.  A transform leaves ``[0, 1]``, and ``lmtp`` predicts it unbounded.  The SDR path passes
no clip to its regressions, and its default pseudo learner is a regression Super Learner
with ``clip=None``.  The witness is a node-1 regression that predicts 1.3 while node 1's
mechanism is wrong.  Double robustness then does not hide the regression, so clipping it to
1 moves the estimate (M8).

T7.  ``g_bounds`` bounds each partial path probability :math:`\prod_{k=t}^{s} p_k`.  At five
folds on an iid sample, with bounds that bind, the engine equals the longhand recursion
under that rule.  Bounding each node's factor before the product (M10) moves the estimate.
With saturated learners a node-2 weight cannot move a training-fold fit, because its
residual sums to zero within each training cell, so the witness is the node-1 weight
:math:`W_{1,2}`.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor

from cleverly.learners import SuperLearner
from cleverly.longitudinal import LTMLE, SDR

from .. import discrete_law_longitudinal as law
from ..longitudinal_sdr import (
    CENSORING_WIDTH,
    COLUMNS,
    TREATMENT_WIDTH,
    NodeLearner,
    longhand_sdr,
    sample,
)
from ..studies.canonical_ltmle import declared_regimens

CAUGHT = 1e-4
PLANS = {"never": (0, 0), "always": (1, 1)}
REGIMENS = declared_regimens({label: law.REGIMEN_SPEC[label] for label in PLANS})


class _Clipped(BaseEstimator):
    """M8: ``base`` with its predictions clipped to ``[0, 1]``."""

    def __init__(self, base: Any = None) -> None:
        self.base = base

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> _Clipped:
        from sklearn.base import clone

        self.model_ = clone(self.base).fit(X, y)
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.clip(self.model_.predict(X), 0.0, 1.0)


def _wrong_first_node(pseudo: Any) -> Any:
    def prior() -> Any:
        return DummyClassifier(strategy="prior")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SDR(
            REGIMENS,
            outcome_learner=law.CellMeans(),
            pseudo_learner=pseudo,
            treatment_learner=NodeLearner(
                {TREATMENT_WIDTH[1]: prior(), TREATMENT_WIDTH[2]: law.CellMeans()}
            ),
            censoring_learner=NodeLearner(
                {CENSORING_WIDTH[1]: prior(), CENSORING_WIDTH[2]: law.CellMeans()}
            ),
            n_folds=1,
            g_bounds=(1e-8, 1.0),
            simultaneous=False,
        ).fit(law.frame(), **COLUMNS)


def test_the_node_regression_is_not_clipped() -> None:
    """A prediction of 1.3 reaches the transform as 1.3."""
    fit = _wrong_first_node(DummyRegressor(strategy="constant", constant=1.3))
    for label in PLANS:
        first = fit.fits[label].steps[0]
        np.testing.assert_array_equal(first.initial, np.full(first.initial.shape, 1.3))


def test_clipping_the_node_regression_moves_the_estimate() -> None:
    """M8: the clip is not neutral where the node's mechanism is wrong."""
    constant = DummyRegressor(strategy="constant", constant=1.3)
    shipped = _wrong_first_node(constant)
    clipped = _wrong_first_node(_Clipped(constant))
    for label in PLANS:
        name = f"ey_regimen[{label}]"
        assert abs(shipped[name].psi - clipped[name].psi) > CAUGHT


def test_the_default_pseudo_learner_is_unclipped() -> None:
    """SDR's default regresses unbounded; LTMLE's default keeps its ``[0, 1]`` clip."""
    sdr = SDR({"always": 1})._pseudo_learner()
    assert isinstance(sdr, SuperLearner)
    assert sdr.task == "regression"
    assert sdr.clip is None
    assert LTMLE({"always": 1})._pseudo_learner().clip == (0.0, 1.0)


def test_an_outcome_learner_still_stands_in_for_the_pseudo_learner() -> None:
    """The fallback order is LTMLE's: ``outcome_learner`` before the default library."""
    outcome = DummyRegressor()
    assert SDR({"always": 1}, outcome_learner=outcome)._pseudo_learner() is outcome


# ------------------------------------------------------------------ T7


@pytest.fixture(scope="module")
def frame() -> Any:
    return sample(600, 3)


@pytest.mark.parametrize("g_bounds", [(0.1, 1.0), (0.2, 1.0)])
def test_partial_path_probabilities_are_bounded(frame: Any, g_bounds: tuple[float, float]) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = SDR(
            REGIMENS,
            outcome_learner=law.CellMeans(),
            pseudo_learner=law.CellMeans(),
            treatment_learner=law.CellMeans(),
            censoring_learner=law.CellMeans(),
            n_folds=5,
            random_state=4,
            g_bounds=g_bounds,
            simultaneous=False,
        ).fit(frame, **COLUMNS)
    for label, arms in PLANS.items():
        assignment = fit.folds.assignment
        name = f"ey_regimen[{label}]"
        shipped = longhand_sdr(frame, arms, assignment, g_bounds=g_bounds)
        per_node = longhand_sdr(frame, arms, assignment, g_bounds=g_bounds, bound_rule="per_node")
        assert fit[name].psi == pytest.approx(shipped.psi, abs=1e-12)
        np.testing.assert_allclose(fit.influence_curves[name], shipped.curve, atol=1e-12, rtol=0)
        assert abs(per_node.psi - shipped.psi) > CAUGHT
        # The bounds bind, and the diagnostic says on what share of paths.
        assert all(step.bound_binding > 0.0 for step in fit.fits[label].steps)
