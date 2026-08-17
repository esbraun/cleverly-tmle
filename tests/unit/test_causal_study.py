"""Typed study, identification, capability, and configuration contracts."""

import dataclasses
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
from cleverly.methods import SHORTCUTS


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


def test_the_study_cannot_be_pointed_at_a_design_it_did_not_prepare() -> None:
    """Freezing the design is not enough while the study can be handed a different one.

    ``_data`` is prepared once, at construction, from the design's column roles; ``identify``
    is the only reader of ``design``.  So a study whose ``design`` had been rebound would
    report an outcome column and adjustment set that no fitted number came from, and nothing
    downstream compares the two.
    """
    study = _study()
    other = PointTreatment(outcome="W1", treatment="A", adjustment=("W2", "W3"))
    with pytest.raises(AttributeError):
        study.design = other  # type: ignore[misc]
    assert study.design.outcome == "Y"


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


def test_a_shortcut_named_like_a_field_sets_that_field() -> None:
    """The rule that stops a keyword from silently moving a different knob.

    ``alpha=`` mapped to ``Targeting.submodel_alpha`` -- the 0.9995 logistic-submodel bound --
    while the field named ``alpha`` was ``Inference.alpha``, the interval level.  Both are
    floats near zero-to-one, so nothing raised and nothing looked wrong; the interval simply
    stayed at 95% while the shrink bound moved.  This checks the general property rather than
    that one pair, so the next shortcut added cannot reintroduce it.
    """
    fields = {
        group: {f.name for f in dataclasses.fields(getattr(TMLEMethod(), group))}
        for group in SHORTCUTS
    }
    for group, mapping in SHORTCUTS.items():
        for shortcut, attribute in mapping.items():
            assert attribute in fields[group], f"{group}.{attribute} is not a field"
            owners = [other for other, names in fields.items() if shortcut in names]
            assert owners in ([], [group]), (
                f"shortcut {shortcut!r} sets {group}.{attribute} but is also the name of a "
                f"field on {owners}; a caller will reasonably expect it to set that one"
            )


def test_the_interval_level_and_the_submodel_bound_are_reachable_separately() -> None:
    method = TMLEMethod().with_overrides(alpha=0.10, submodel_alpha=0.99)
    assert method.inference.alpha == 0.10
    assert method.targeting.submodel_alpha == 0.99
    # And they land on the engine under the names it uses, which are the other way around.
    kwargs = method.estimator_kwargs()
    assert kwargs["alpha_sig"] == 0.10
    assert kwargs["alpha"] == 0.99
    with pytest.raises(TypeError, match="alpha_sig"):
        TMLEMethod().with_overrides(alpha_sig=0.10)


def test_estimation_options_cannot_reassign_study_roles() -> None:
    effect = _study().identify(ATE())
    with pytest.raises(TypeError, match="Study-design roles belong on PointTreatment"):
        effect.estimate(covariates=["W1"])
