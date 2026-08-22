"""Typed study, identification, capability, and configuration contracts."""

import dataclasses
import inspect
import re
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sklearn.linear_model

import cleverly
from cleverly import (
    ATE,
    CapabilityError,
    CausalStudy,
    CleverlyError,
    CollaborativeTMLEMethod,
    ControlledDirectEffect,
    CrossFitting,
    DataError,
    DRTMLEMethod,
    LongitudinalTreatment,
    MethodConfigurationError,
    ModelSpec,
    MSMProjection,
    PointTreatment,
    RegimeContrast,
    RegimeMean,
    Runtime,
    TMLEMethod,
)
from cleverly.data import CausalData, validate
from cleverly.data.validate import RANDOMIZED_INTERCEPT
from cleverly.datasets import make_linear_ate, make_longitudinal
from cleverly.estimators import TMLE
from cleverly.longitudinal import LTMLE, LongitudinalData
from cleverly.longitudinal.estimator import DEFAULT_LTMLE_G_BOUNDS
from cleverly.methods import (
    DEFAULT_LONGITUDINAL_G_BOUNDS,
    DEFAULT_LONGITUDINAL_MULTIPLIER,
    DEFAULT_POINT_MULTIPLIER,
    SHORTCUTS,
)
from cleverly.msm import MSM
from cleverly.study import _STRING_ESTIMANDS
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


class TestAPreparedContainerIsReconciledWithTheDesign:
    """Handing the study a built container used to adopt it without a single check.

    ``prepare`` returned any ``CausalData``/``LongitudinalData`` unchanged, so every role on
    the design became a claim nothing had verified. The design is what
    ``IdentifiedEffect.functional`` records and ``summary()`` prints, so a design naming an
    adjustment set the container never adjusted for reported that set as identification --
    and this PR then persisted it. It is the data-construction half of the hazard
    ``test_the_study_cannot_be_pointed_at_a_design_it_did_not_prepare`` argues for rebinding.
    """

    @staticmethod
    def _prepared():  # type: ignore[no-untyped-def]
        frame, _ = make_linear_ate(n=120, seed=19)
        return frame, CausalData.from_frame(
            frame, outcome="Y", treatment="A", covariates=("W1", "W2", "W3")
        )

    def test_a_matching_container_is_accepted_and_reported_truthfully(self) -> None:
        _, data = self._prepared()
        design = PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3"))
        effect = CausalStudy(data, design=design).identify(ATE())
        assert effect.functional.adjustment == ("W1", "W2", "W3")
        assert "W1" in effect.summary()

    @pytest.mark.parametrize(
        ("design", "reason"),
        [
            (
                PointTreatment(outcome="Y", treatment="A", adjustment=("Z1",)),
                "adjusts for",
            ),
            (
                PointTreatment(outcome="W1", treatment="A", adjustment=("W1", "W2", "W3")),
                "outcome=",
            ),
            (
                PointTreatment(outcome="Y", treatment="W1", adjustment=("W1", "W2", "W3")),
                "treatment=",
            ),
            (
                PointTreatment(
                    outcome="Y", treatment="A", adjustment=("W1", "W2", "W3"), cluster="G"
                ),
                "cluster=",
            ),
            (
                PointTreatment(
                    outcome="Y",
                    treatment="A",
                    adjustment=("W1", "W2", "W3"),
                    treatment_kind="continuous",
                ),
                "treatment_kind=",
            ),
            (
                PointTreatment(
                    outcome="Y",
                    treatment="A",
                    adjustment=("W1", "W2", "W3"),
                    outcome_family="binomial",
                ),
                "outcome_family=",
            ),
        ],
    )
    def test_every_role_the_design_states_is_checked(self, design, reason) -> None:  # type: ignore[no-untyped-def]
        _, data = self._prepared()
        with pytest.raises(DataError, match=reason):
            CausalStudy(data, design=design)

    def test_an_adjustment_set_survives_encoding_and_dropping(self) -> None:
        """``covariate_names`` is post-encoding, so the comparison uses the named columns.

        A categorical adjustment variable becomes several generated columns and a degenerate
        one is dropped outright, so comparing the stored names to the declaration directly
        would reject the very containers this check exists to accept.
        """
        frame, _ = make_linear_ate(n=120, seed=23)
        frame = frame.assign(G=pd.Categorical(np.where(frame["W1"] > 0, "high", "low")), C=1.0)
        data = CausalData.from_frame(frame, outcome="Y", treatment="A", covariates=("W1", "G", "C"))
        assert data.dropped_covariates == ("C",)
        assert set(data.covariate_names) != {"W1", "G", "C"}
        design = PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "G", "C"))
        assert CausalStudy(data, design=design).data is data

    def test_a_longitudinal_container_is_reconciled_node_by_node(self) -> None:
        frame, _ = make_longitudinal(n=100, seed=11)
        columns = {
            "outcome": "Y",
            "treatment": ("A1", "A2"),
            "baseline": ("W1", "W2"),
            "time_varying": ((), ("L2",)),
            "censoring": ("C1", "C2"),
        }
        data = LongitudinalData.from_frame(frame, **columns)
        assert CausalStudy(data, design=LongitudinalTreatment(**columns)).data is data
        for role, changed in (
            ("baseline", {"baseline": ("W1",)}),
            ("outcome event nodes", {"outcome": ("Y1", "Y2")}),
            ("censoring", {"censoring": None}),
            ("time_varying", {"time_varying": (("L2",), ())}),
        ):
            with pytest.raises(DataError, match=role):
                CausalStudy(data, design=LongitudinalTreatment(**{**columns, **changed}))


def test_the_reserved_intercept_column_survives_the_constant_sweep() -> None:
    """One name, two packages, and only one thing standing between them and a bug.

    ``PointTreatment(randomized=True, adjustment=())`` is a claim of *no* adjustment, and the
    reserved constant column is what keeps the design well formed for learners that fit their
    own intercept. ``check_covariates`` drops constant columns, so the exemption keyed on that
    name is the only reason the column survives -- and the name was written out as a bare
    literal at both ends, where renaming one would have left the fit with no covariates.
    """
    written_out = [
        path
        for path in Path(cleverly.__file__).parent.rglob("*.py")
        if f'"{RANDOMIZED_INTERCEPT}"' in path.read_text(encoding="utf-8")
    ]
    assert written_out == [Path(validate.__file__)], (
        "the reserved name is spelled out somewhere other than its definition; import "
        "RANDOMIZED_INTERCEPT instead, so a rename cannot reach one end and not the other"
    )
    frame, _ = make_linear_ate(n=120, seed=29)
    study = CausalStudy(
        frame[["Y", "A"]], design=PointTreatment(outcome="Y", treatment="A", randomized=True)
    )
    assert study.data.covariate_names == (RANDOMIZED_INTERCEPT,)
    assert study.data.dropped_covariates == ()
    result = study.estimate(
        ATE(),
        outcome_learner=sklearn.linear_model.LinearRegression(),
        treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
    )
    assert np.isfinite(result.psi("ate"))
    # The design still says what it claimed: no adjustment variables at all.
    assert result.identified_effect.functional.adjustment == ()


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


def test_the_refusal_recommends_what_the_migration_guide_recommends() -> None:
    """Two places tell a migrating reader what to write instead, and they must agree.

    The refusal names a typed object for the string it was given; ``docs/migration.md`` maps
    the same strings in its argument table. If those drift, one of them is telling somebody to
    write code that does not do what the other says it does. The table is parsed rather than
    restated here for the same reason ``TestEvidenceManifest`` parses the evidence manifest:
    the artefact a reader opens has to be the thing that is checked.
    """
    rows = re.findall(
        r"^\|\s*`estimands=\(\"(\w+)\",\)`\s*\|\s*`([^`]+)`\s*\|",
        (Path(cleverly.__file__).parents[2] / "docs" / "migration.md").read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    assert len(rows) > 5, "the migration table stopped matching; fix the pattern, not the count"
    documented = dict(rows)
    for target, replacement in documented.items():
        assert _STRING_ESTIMANDS[target] == replacement
    assert documented.keys() <= _STRING_ESTIMANDS.keys()


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
        outcome_learner=sklearn.linear_model.LinearRegression(),
        treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
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
        outcome_learner=sklearn.linear_model.LinearRegression(),
        treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
        n_folds=4,
        random_state=7,
    )
    declared = TMLEMethod(
        models=ModelSpec(
            outcome_learner=sklearn.linear_model.LinearRegression(),
            treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
        ),
        cross_fitting=CrossFitting(n_folds=4),
        runtime=Runtime(random_state=7),
    )
    assert dataclasses.replace(normalized, models=ModelSpec()) == dataclasses.replace(
        declared, models=ModelSpec()
    )
    assert type(normalized.models.outcome_learner) is type(declared.models.outcome_learner)
    assert (
        normalized.models.outcome_learner.get_params()
        == declared.models.outcome_learner.get_params()
    )
    assert type(normalized.models.treatment_learner) is type(declared.models.treatment_learner)
    assert (
        normalized.models.treatment_learner.get_params()
        == declared.models.treatment_learner.get_params()
    )


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


def test_auto_resolves_to_each_engines_own_default_and_says_so_when_they_move() -> None:
    """``Inference`` serves two engines whose defaults are not the same number.

    ``n_multiplier`` was a single literal 1000, which matches ``TMLE`` and silently halved
    every study-driven longitudinal fit: ``LTMLE`` draws 2000. Nothing reported the change,
    and the parity suite could not see it because it turns simultaneous bands off. ``"auto"``
    now defers to the engine, exactly as ``g_bounds`` already did.

    The resolved values are restated in ``cleverly.methods`` rather than imported, so this
    reads both engine signatures and fails when a restatement stops matching -- otherwise the
    duplication would be free to drift back apart.
    """
    point = inspect.signature(TMLE).parameters
    sequential = inspect.signature(LTMLE).parameters
    assert point["n_multiplier"].default == DEFAULT_POINT_MULTIPLIER
    assert sequential["n_multiplier"].default == DEFAULT_LONGITUDINAL_MULTIPLIER
    assert DEFAULT_POINT_MULTIPLIER != DEFAULT_LONGITUDINAL_MULTIPLIER, (
        "if the engines ever agree, delete the sentinel rather than keeping a split that "
        "no longer splits anything"
    )
    assert DEFAULT_LONGITUDINAL_G_BOUNDS == DEFAULT_LTMLE_G_BOUNDS

    method = TMLEMethod()
    assert method.inference.n_multiplier == "auto"
    assert method.estimator_kwargs()["n_multiplier"] == DEFAULT_POINT_MULTIPLIER
    assert (
        method.estimator_kwargs(longitudinal=True)["n_multiplier"]
        == DEFAULT_LONGITUDINAL_MULTIPLIER
    )
    # An explicit request still wins on both paths.
    asked = TMLEMethod().with_overrides(n_multiplier=400)
    assert asked.estimator_kwargs()["n_multiplier"] == 400
    assert asked.estimator_kwargs(longitudinal=True)["n_multiplier"] == 400


def test_the_interval_level_and_the_submodel_bound_are_reachable_separately() -> None:
    method = TMLEMethod().with_overrides(alpha=0.10, submodel_alpha=0.99)
    assert method.inference.alpha == 0.10
    assert method.targeting.submodel_alpha == 0.99
    # And they land on the engine under the names it uses, which are the other way around.
    kwargs = method.estimator_kwargs()
    assert kwargs["alpha_sig"] == 0.10
    assert kwargs["alpha"] == 0.99
    with pytest.raises(MethodConfigurationError, match="alpha_sig") as raised:
        TMLEMethod().with_overrides(alpha_sig=0.10)
    assert isinstance(raised.value, CleverlyError)


@pytest.mark.parametrize(
    "option",
    [
        {"missingness_learner": sklearn.linear_model.LogisticRegression(max_iter=1000)},
        {"intermediate_learner": sklearn.linear_model.LogisticRegression(max_iter=1000)},
        {"density_bins": 12},
        {"screen_treatment": True},
        {"screen_threshold": 0.2},
        {"min_retain": 2},
        {"repeats": 2},
        {"stratify_folds": "outcome"},
        {"targeting_scheme": "foldwise"},
        {"cv_evaluation": True},
        {"fluctuation": "linear"},
        {"targeting": "one_step"},
        {"nuisance_bound": 0.02},
        {"target_weights": True},
        {"step_size": 0.002},
        {"n_bootstrap": 500},
        {"bootstrap_resampling": "iid"},
    ],
    ids=lambda option: next(iter(option)),
)
def test_every_point_only_option_is_refused_by_longitudinal_translation(
    option: dict[str, object],
) -> None:
    """A normalized declaration must reach the engine or fail before construction.

    These are the 17 point-only fields in the shared configuration. Sixteen used to be
    accepted and omitted from the longitudinal kwargs; ``repeats`` alone had a bespoke
    refusal. Pinning the whole list prevents a future field from disappearing just because
    the two engine signatures differ.
    """
    name = next(iter(option))
    method = TMLEMethod().with_overrides(**option)
    with pytest.raises(MethodConfigurationError, match=name) as raised:
        method.estimator_kwargs(longitudinal=True)
    assert isinstance(raised.value, CleverlyError)


def test_cross_fit_false_keeps_its_supported_longitudinal_meaning() -> None:
    method = TMLEMethod().with_overrides(cross_fit=False)
    assert method.estimator_kwargs(longitudinal=True)["n_folds"] == 1


@pytest.mark.parametrize("method", [CollaborativeTMLEMethod(), DRTMLEMethod()])
def test_variant_longitudinal_refusals_use_the_library_error_hierarchy(method) -> None:
    with pytest.raises(MethodConfigurationError) as raised:
        method.estimator_kwargs(longitudinal=True)
    assert isinstance(raised.value, CleverlyError)


def test_a_longitudinal_option_refuses_before_engine_construction(monkeypatch) -> None:
    frame, _ = make_longitudinal(n=100, seed=23)
    study = CausalStudy(
        frame,
        design=LongitudinalTreatment(
            outcome="Y",
            treatment=("A1", "A2"),
            baseline=("W1", "W2"),
            time_varying=((), ("L2",)),
            censoring=("C1", "C2"),
        ),
    )
    effect = study.identify(RegimeMean({"always": 1}))
    monkeypatch.setattr("cleverly.study.LTMLE", _MustNotConstruct)
    with pytest.raises(MethodConfigurationError, match="n_bootstrap"):
        effect.estimate(n_bootstrap=500)


def test_estimation_options_cannot_reassign_study_roles() -> None:
    effect = _study().identify(ATE())
    with pytest.raises(
        MethodConfigurationError, match="Study-design roles belong on PointTreatment"
    ):
        effect.estimate(covariates=["W1"])


def test_an_invalid_method_declaration_uses_the_library_error_hierarchy() -> None:
    effect = _study().identify(ATE())
    with pytest.raises(MethodConfigurationError) as raised:
        effect.estimate(method=object())  # type: ignore[arg-type]
    assert isinstance(raised.value, CleverlyError)
