"""The methods table of a clustered point study names the one C-TMLE strategy that fits (X124).

Every C-TMLE strategy but ``"derivative"`` refuses ``id=`` before its first learner, so the
table read "yes" while ``estimate(method="collaborative_tmle")`` refused. The record now
says the method fits only with ``strategy='derivative'``, and that strategy fits.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import ATE, CausalStudy, PointTreatment
from cleverly.datasets import make_clustered
from cleverly.exceptions import CapabilityError
from cleverly.methods import CollaborativeTMLEMethod, CrossFitting, ModelSpec


def _question(cluster: str | None) -> Any:
    frame = make_clustered(n=600, cluster_size=20, seed=11)[0]
    design = PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2"), cluster=cluster)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CausalStudy(frame, design=design).identify(ATE())


def _record(question: Any) -> Any:
    return {record.name: record for record in question.available_methods()}["collaborative_tmle"]


def test_a_clustered_study_lists_only_the_derivative_strategy() -> None:
    question = _question("cluster")
    record = _record(question)
    assert record.available and record.only == "strategy='derivative'"
    assert "only with strategy='derivative'" in str(question.available_methods().summary())
    with warnings.catch_warnings(), pytest.raises(CapabilityError, match="no clustered result"):
        warnings.simplefilter("ignore")
        question.estimate(method="collaborative_tmle")
    models = ModelSpec(outcome_learner=LinearRegression(), treatment_learner=LogisticRegression())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = question.estimate(
            method=CollaborativeTMLEMethod(
                models=models,
                cross_fitting=CrossFitting(enabled=False),
                strategy="derivative",
            )
        )
    assert "ate" in result.estimates


def test_an_unclustered_study_names_no_configuration() -> None:
    assert _record(_question(None)).only is None
