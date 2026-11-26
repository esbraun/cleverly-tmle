"""Messages for learners a collaborative or DR-TMLE fit cannot use as declared.

* A ``treatment_learner`` declared beside ``strategy="derivative"`` without
  ``treatment_candidates=``, or beside ``strategy="lasso"``, fits no treatment mechanism. The fit
  warns, records the warning on the result, and gives the same estimate as without the learner.
* A ``ParametricOutcome`` in a DR-TMLE reduced regression slot, explicit or through the
  ``outcome_learner=`` fallback, is refused before any learner fits, with a message that names it.
* ``refute`` annotates every declaration its generated tests accept.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.estimators import CTMLE, DRTMLE
from cleverly.exceptions import CapabilityError
from cleverly.learners import ParametricOutcome
from cleverly.validation import BernoulliAdjustmentOutcome, BoundedAdjustmentOutcome, refute
from tests.studies.ctmle_derivative_laws import PRIMARY

COVARIATES = ["W1", "W2", "W3", "W4"]
UNUSED = "fits no treatment mechanism"


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return PRIMARY.sample(400, 3)


def _ctmle(frame: pd.DataFrame, **settings: Any) -> Any:
    estimator = CTMLE(
        outcome_learner=LogisticRegression(C=1e6, max_iter=1000),
        estimands=["ate"],
        cross_fit=False,
        random_state=7,
        **settings,
    )
    return estimator.fit(frame, outcome="Y", treatment="A", covariates=COVARIATES).single()


def _messages(record: list[warnings.WarningMessage]) -> list[str]:
    return [str(item.message) for item in record]


class TestUnusedTreatmentLearner:
    def test_the_derivative_strategy_warns_and_ignores_the_learner(
        self, frame: pd.DataFrame
    ) -> None:
        with pytest.warns(UserWarning, match=UNUSED) as record:
            declared = _ctmle(
                frame,
                strategy="derivative",
                treatment_learner=LogisticRegression(C=np.inf, max_iter=1000),
            )
        (message,) = [text for text in _messages(record) if UNUSED in text]
        assert "strategy='derivative'" in message
        assert "treatment_candidates=(mechanism, partner)" in message
        assert any(UNUSED in item.message for item in declared.fit_warnings)
        # The witness that the learner is unused: the fit without it is the same fit.
        with warnings.catch_warnings(record=True) as plain:
            warnings.simplefilter("always")
            default = _ctmle(frame, strategy="derivative")
        assert not [text for text in _messages(plain) if UNUSED in text]
        assert declared["ate"].psi == default["ate"].psi

    def test_declared_candidates_use_the_learners_and_do_not_warn(
        self, frame: pd.DataFrame
    ) -> None:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            _ctmle(
                frame,
                strategy="derivative",
                treatment_learner=LogisticRegression(max_iter=1000),
                treatment_candidates=(
                    LogisticRegression(C=1.0, max_iter=1000),
                    LogisticRegression(C=0.5, max_iter=1000),
                ),
            )
        assert not [text for text in _messages(record) if UNUSED in text]

    def test_the_lasso_strategy_warns(self, frame: pd.DataFrame) -> None:
        with pytest.warns(UserWarning, match="fits no candidate") as record:
            _ctmle(
                frame,
                strategy="lasso",
                lambdas=(0.05, 0.01, 0.0),
                treatment_learner=LogisticRegression(max_iter=1000),
            )
        assert any("strategy='learners'" in text for text in _messages(record))

    def test_a_strategy_that_uses_the_learner_does_not_warn(self, frame: pd.DataFrame) -> None:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            _ctmle(frame, strategy="greedy", treatment_learner=LogisticRegression(max_iter=1000))
        assert not [
            text for text in _messages(record) if UNUSED in text or "fits no candidate" in text
        ]


class TestParametricOutcomeInReducedSlots:
    def _drtmle(self, **settings: Any) -> DRTMLE:
        return DRTMLE(
            treatment_learner=LogisticRegression(C=np.inf, max_iter=1000),
            estimands=("ate",),
            cross_fit=False,
            random_state=7,
            **settings,
        )

    def test_the_outcome_learner_fallback_is_refused_by_name(self, frame: pd.DataFrame) -> None:
        estimator = self._drtmle(outcome_learner=ParametricOutcome(features="arm_interactions"))
        with pytest.raises(CapabilityError, match="ParametricOutcome") as caught:
            estimator.fit(frame, outcome="Y", treatment="A", covariates=COVARIATES)
        assert "reduced_outcome_learner=None takes outcome_learner=" in str(caught.value)

    def test_an_explicit_reduced_slot_is_refused_by_name(self, frame: pd.DataFrame) -> None:
        estimator = self._drtmle(
            outcome_learner=ParametricOutcome(features="arm_interactions"),
            reduced_outcome_learner=LinearRegression(),
            reduced_mechanism_residual_learner=ParametricOutcome(),
        )
        with pytest.raises(CapabilityError, match="reduced_mechanism_residual_learner="):
            estimator.fit(frame, outcome="Y", treatment="A", covariates=COVARIATES)

    def test_regressors_in_both_slots_keep_the_parametric_outcome_model(
        self, frame: pd.DataFrame
    ) -> None:
        estimator = self._drtmle(
            outcome_learner=ParametricOutcome(features="arm_interactions"),
            reduced_outcome_learner=LinearRegression(),
            reduced_mechanism_residual_learner=LinearRegression(),
        )
        result = estimator.fit(frame, outcome="Y", treatment="A", covariates=COVARIATES).single()
        assert np.isfinite(result["ate"].psi)


@pytest.mark.parametrize("name", ["dummy_outcome", "simulated_outcome"])
def test_refute_annotates_every_accepted_point_declaration(name: str) -> None:
    # The module defers its annotations, so the annotation is the source text of the union.
    annotation = str(inspect.signature(refute).parameters[name].annotation)
    for process in (BernoulliAdjustmentOutcome, BoundedAdjustmentOutcome):
        assert process.__name__ in annotation
