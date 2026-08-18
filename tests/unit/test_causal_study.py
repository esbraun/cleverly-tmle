"""Typed study, identification, capability, and configuration contracts."""

import dataclasses
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from cleverly import (
    ATE,
    CapabilityError,
    CausalStudy,
    ControlledDirectEffect,
    CrossFitting,
    DataError,
    ModelSpec,
    MSMProjection,
    PointTreatment,
    RegimeContrast,
    RegimeMean,
    Runtime,
    TMLEMethod,
)
from cleverly.datasets import make_linear_ate
from cleverly.methods import SHORTCUTS
from cleverly.msm import MSM
from tests.conftest import FAST_KWARGS


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


def test_stratified_parameter_keys_are_structured_before_the_alias_is_displayed() -> None:
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
    result = study.identify(ATE()).estimate(**FAST_KWARGS)
    assert result.parameter_keys["ate[S=0]"].stratum == (0,)
    assert result.parameter_keys["ate[S=1]"].stratum == (1,)


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


def test_a_controlled_direct_effect_refuses_the_variants_before_fitting(monkeypatch) -> None:
    """The variant check has to read the intermediate, not only the functional's target.

    A ``ControlledDirectEffect``'s ``functional.target`` is its *contrast's* name -- ``ate`` --
    so a check reading the target alone declared C-TMLE and DR-TMLE available for it, and both
    engines then refused partway through a fit.  Refusing after nuisance fitting has started is
    the thing ``docs/architecture-invariants.md`` puts at the identification boundary.
    """
    frame, _ = make_linear_ate(n=120, seed=21)
    study = CausalStudy(
        frame.assign(Z=(frame["W1"] > 0).astype(int)),
        design=PointTreatment(
            outcome="Y", treatment="A", adjustment=("W1", "W2"), intermediate="Z"
        ),
    )
    effect = study.identify(ControlledDirectEffect(intermediate=1.0))
    methods = {record.name: record for record in effect.available_methods()}
    assert methods["tmle"].available
    for name in ("collaborative_tmle", "drtmle"):
        assert not methods[name].available
        assert "controlled direct effect" in (methods[name].reason or "")

    monkeypatch.setattr("cleverly.study.CTMLE", _MustNotConstruct)
    monkeypatch.setattr("cleverly.study.DRTMLE", _MustNotConstruct)
    for name in ("collaborative_tmle", "drtmle"):
        with pytest.raises(CapabilityError, match="controlled direct effect"):
            effect.estimate(method=name)


class _MustNotConstruct:
    """An engine stand-in proving a refusal happened before nuisance fitting."""

    def __init__(self, *args, **kwargs) -> None:
        raise AssertionError("nuisance fitting path was reached")


@pytest.mark.parametrize(
    ("estimand", "reason"),
    [
        ("ate", "not a typed causal estimand"),
        (None, "not a typed causal estimand"),
        (ATE, "not a typed causal estimand"),
    ],
)
def test_an_untyped_estimand_is_refused_by_name(estimand, reason, monkeypatch) -> None:
    """``identify("ate")`` used to die on ``'str' object has no attribute 'name'``.

    Every legacy call site spelled its estimands as strings, so that is the first thing a
    migrating reader tries.  The provider dereferences ``estimand.name`` as its opening move,
    so the failure surfaced from inside identification with nothing pointing at the typed
    object to pass instead.  ``ATE`` the *class* is here too: it has a ``name`` attribute, so
    it got further than a string did and failed later and less legibly.
    """
    monkeypatch.setattr("cleverly.study.TMLE", _MustNotConstruct)
    with pytest.raises(CapabilityError, match=reason) as raised:
        _study().identify(estimand)
    if estimand == "ate":
        assert "ATE()" in str(raised.value)


def test_a_string_contrast_on_a_controlled_direct_effect_is_refused_at_construction() -> None:
    with pytest.raises(DataError, match="typed arm contrast"):
        ControlledDirectEffect(intermediate=1.0, contrast="ate")


@pytest.mark.parametrize(
    ("estimand", "reason"),
    [
        (RegimeMean(regimens=(), horizons=(1, 2)), "one time point"),
        (RegimeContrast(regimens=(), horizons=(1,)), "one time point"),
        (MSMProjection(MSM.linear(), horizons=(1,)), "one time point"),
        (MSMProjection(MSM.linear(), regimens={"always": 1}), "longitudinal regimen cells"),
    ],
)
def test_a_sequential_declaration_on_a_point_design_is_refused(
    estimand, reason, monkeypatch
) -> None:
    """A declaration that cannot take effect is refused, not dropped.

    ``horizons=`` and ``MSMProjection(regimens=...)`` are read only on the longitudinal path.
    On a point design they were silently discarded, so the fit answered a different question
    from the one written down and reported it under the name of the one asked for.
    """
    monkeypatch.setattr("cleverly.study.TMLE", _MustNotConstruct)
    with pytest.raises(CapabilityError, match=reason):
        _study().identify(estimand)


def test_a_continuous_dose_refuses_by_axis_and_admits_a_working_model() -> None:
    """The dose rule is about the parameter axis, and ``msm`` is not an arm axis.

    The refusal named targets rather than axes, so a continuous-dose ``MSMProjection`` was
    turned away with the message "is arm-indexed" -- which ``TARGETS["msm"].parameter_axis``
    contradicts, and which the engine contradicts too: ``tests/unit/test_continuous_msm.py``
    fits exactly this composition.  The two genuinely arm-shaped axes stay refused, and now
    say which axis they are.
    """
    rng = np.random.default_rng(11)
    n = 180
    w = rng.normal(size=n)
    a = 0.4 * w + rng.normal(size=n)
    frame = pd.DataFrame({"Y": 1.0 + 2.0 * a + 0.3 * w, "A": a, "W": w})
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y", treatment="A", adjustment=("W",), treatment_kind="continuous"
        ),
    )

    result = study.estimate(
        MSMProjection(MSM.linear(doses=np.linspace(-1.5, 1.5, 9))),
        outcome_learner="glm",
        treatment_learner="glm",
        cross_fit=False,
        density_bins=8,
        simultaneous=False,
        random_state=3,
    )
    # The same slope and score the engine-level test pins, reached through the typed API.
    assert result["msm[a]"].psi == pytest.approx(2.0, abs=2e-6)
    assert abs(result["msm[a]"].score) < 1e-10
    assert result.parameter_keys["msm[a]"].term == "a"

    for refused, axis in ((ATE(), "arm"), (RegimeMean(regimens=()), "regime")):
        with pytest.raises(CapabilityError, match=f"indexed by {axis}"):
            study.identify(refused)


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
