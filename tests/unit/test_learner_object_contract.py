"""Every nuisance-model entry point enforces the estimator-object contract."""

from __future__ import annotations

import pytest

from cleverly import SuperLearner
from cleverly.estimators import DRTMLE, TMLE
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
