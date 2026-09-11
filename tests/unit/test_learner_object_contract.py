"""Every nuisance-model entry point enforces the estimator-object contract."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import SuperLearner
from cleverly.datasets import make_longitudinal
from cleverly.estimators import DRTMLE, TMLE
from cleverly.exceptions import MethodConfigurationError
from cleverly.learners.super_learner import resolve_learner
from cleverly.longitudinal import LTMLE
from cleverly.methods import DRTMLEMethod, ModelSpec


@pytest.mark.parametrize(
    "slot",
    [
        "outcome_learner",
        "treatment_learner",
        "missingness_learner",
        "intermediate_learner",
        "pseudo_learner",
        "censoring_learner",
    ],
)
def test_model_spec_rejects_string_nuisance_learners(slot: str) -> None:
    with pytest.raises(TypeError, match=slot):
        ModelSpec(**{slot: "glm"})


@pytest.mark.parametrize(
    "slot",
    [
        "outcome_learner",
        "treatment_learner",
        "missingness_learner",
        "intermediate_learner",
    ],
)
def test_tmle_rejects_string_nuisance_learners(slot: str) -> None:
    with pytest.raises(TypeError, match=slot):
        TMLE(**{slot: "glm"})


@pytest.mark.parametrize(
    "slot",
    [
        "outcome_learner",
        "pseudo_learner",
        "treatment_learner",
        "censoring_learner",
    ],
)
def test_ltmle_rejects_string_nuisance_learners(slot: str) -> None:
    with pytest.raises(TypeError, match=slot):
        LTMLE({}, **{slot: "glm"})


@pytest.mark.parametrize("slot", ["reduced_outcome_learner", "reduced_treatment_learner"])
def test_drtmle_rejects_string_reduced_learners(slot: str) -> None:
    with pytest.raises(TypeError, match=slot):
        DRTMLE(**{slot: "glm"})


@pytest.mark.parametrize("slot", ["reduced_outcome_learner", "reduced_treatment_learner"])
def test_drtmle_method_rejects_string_reduced_learners(slot: str) -> None:
    with pytest.raises(TypeError, match=slot):
        DRTMLEMethod(**{slot: "glm"})


def test_super_learner_rejects_a_string_library() -> None:
    with pytest.raises(TypeError, match="estimator objects"):
        SuperLearner(library="default")  # type: ignore[arg-type]


class _Unclonable(BaseEstimator):
    """An estimator whose ``fit`` and ``predict`` work and whose clone raises.

    It satisfies the object contract the tests above enforce, so nothing refuses it before a
    fit asks for a clone of it.
    """

    def __sklearn_clone__(self) -> _Unclonable:
        raise TypeError("refuses to be cloned")

    def fit(self, X: Any, y: Any, **kwargs: Any) -> _Unclonable:
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X: Any) -> Any:
        return np.full(len(X), self.mean_)


def test_a_learner_the_fit_clones_is_refused_by_name_and_not_by_sklearn() -> None:
    """cleverly clones a learner template per fold, so the refusal names the learner.

    The refusal replaces :func:`sklearn.base.clone`'s own ``TypeError``, which ended in
    scikit-learn several frames below the argument that caused it.
    :class:`~cleverly.exceptions.MethodConfigurationError` subclasses ``ValueError``, so the
    two are asserted apart here rather than left to the class name.
    """

    frame, _ = make_longitudinal(n=60, seed=3)
    with pytest.raises(MethodConfigurationError, match="_Unclonable cannot be cloned") as caught:
        LTMLE(
            {"always": 1},
            reference="always",
            outcome_learner=_Unclonable(),
            pseudo_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            censoring_learner=LogisticRegression(max_iter=1000),
            n_folds=1,
            learner_folds=2,
            random_state=3,
            simultaneous=False,
        ).fit(
            frame,
            outcome="Y",
            treatment=("A1", "A2"),
            baseline=("W1", "W2"),
            time_varying=((), ("L2",)),
            censoring=("C1", "C2"),
        )

    assert isinstance(caught.value, ValueError)
    assert not isinstance(caught.value, TypeError)
    # The remedy, which the bare scikit-learn exception did not carry.
    assert "clones it once per fold" in str(caught.value)


def test_an_unclonable_super_learner_member_is_refused_through_resolve_learner() -> None:
    """Cloning a SuperLearner clones its library, so a nested member reaches the same refusal.

    ``resolve_learner`` clones an explicit SuperLearner that carries no seed of its own, to
    give it the enclosing estimator's seed without mutating the caller's template.  That
    clone is the one place a nested member's refusal used to escape as a raw scikit-learn
    exception naming neither the library nor the remedy.
    """

    library = SuperLearner(library=[_Unclonable()], n_folds=2, random_state=None)
    with pytest.raises(MethodConfigurationError, match="SuperLearner cannot be cloned"):
        resolve_learner(library, task="regression", random_state=3)
