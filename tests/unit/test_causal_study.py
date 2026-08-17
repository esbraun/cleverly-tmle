"""Typed study, identification, capability, and configuration contracts."""

from dataclasses import FrozenInstanceError

import pytest

from cleverly import (
    ATE,
    CapabilityError,
    CausalStudy,
    CrossFitting,
    ModelSpec,
    PointTreatment,
    Runtime,
    TMLEMethod,
)
from cleverly.datasets import make_linear_ate


def _study() -> CausalStudy:
    frame, _ = make_linear_ate(n=120, seed=19)
    return CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W1", "W2", "W3", "W4"),
        ),
    )


def test_the_design_is_immutable_and_requires_an_identification_declaration() -> None:
    design = _study().design
    assert design.adjustment == ("W1", "W2", "W3", "W4")
    with pytest.raises(FrozenInstanceError):
        design.outcome = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="non-empty adjustment set"):
        PointTreatment(outcome="Y", treatment="A")


def test_identification_is_inspectable_before_estimation() -> None:
    effect = _study().identify(ATE())
    assert effect.functional.adjustment == ("W1", "W2", "W3", "W4")
    assert effect.identification.required_nuisances == (
        "outcome_regression",
        "treatment_mechanism",
    )
    summary = effect.summary()
    assert "E_W[E(Y | A=a, W)" in summary
    assert "no unmeasured confounding" in summary


def test_an_unknown_reference_is_refused_during_identification() -> None:
    with pytest.raises(ValueError, match="is not a treatment level"):
        _study().identify(ATE(reference="not-an-arm"))


def test_stratified_parameter_keys_are_refused_before_estimation() -> None:
    frame, _ = make_linear_ate(n=120, seed=20)
    frame = frame.assign(S=(frame["W1"] > 0).astype(int))
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W1", "W2", "W3", "W4", "S"),
            strata=("S",),
        ),
    )
    with pytest.raises(CapabilityError, match="stratified ATE parameters"):
        study.identify(ATE())


def test_method_availability_is_structured_and_refuses_before_fitting(monkeypatch) -> None:
    effect = _study().identify(ATE())
    methods = {record.name: record for record in effect.available_methods()}
    assert methods["tmle"].available
    assert not methods["riesz_tmle"].available
    assert "representer" in (methods["riesz_tmle"].reason or "")

    class MustNotConstruct:
        def __init__(self, **kwargs) -> None:
            raise AssertionError("nuisance fitting path was reached")

    monkeypatch.setattr("cleverly.study.TMLE", MustNotConstruct)
    with pytest.raises(CapabilityError, match="direct-Riesz engine"):
        effect.estimate(method="riesz_tmle")


def test_keyword_shortcuts_normalize_to_the_same_typed_method() -> None:
    normalized = TMLEMethod().with_overrides(
        outcome_learner="glm",
        treatment_learner="glm",
        n_folds=4,
        random_state=7,
    )
    declared = TMLEMethod(
        models=ModelSpec(outcome_learner="glm", treatment_learner="glm"),
        cross_fitting=CrossFitting(n_folds=4),
        runtime=Runtime(random_state=7),
    )
    assert normalized == declared


def test_estimation_options_cannot_reassign_study_roles() -> None:
    effect = _study().identify(ATE())
    with pytest.raises(TypeError, match="Study-design roles belong on PointTreatment"):
        effect.estimate(covariates=["W1"])
