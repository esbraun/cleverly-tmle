"""The shared post-fit assessment contract across scalar result families."""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import json
import re
import types
import typing

import pandas as pd
import pytest
import sklearn.linear_model

from cleverly import (
    ATE,
    ATT,
    AssessmentReport,
    AssessmentStatus,
    CapabilityError,
    CausalResult,
    CausalStudy,
    ExplicitAdjustmentProvider,
    IdentificationProvider,
    LongitudinalTreatment,
    PointTreatment,
    PositivityWarning,
    RegimeMean,
    ValidationReport,
    load,
)
from cleverly._assessment_cache import _CACHE_GENERATIONS
from cleverly.assessment import (
    _ATTENTION,
    _BLOCKING,
    _OMISSIONS,
    _SETTLED,
    _SUMMARY_CHECK_ORDER,
    _SUMMARY_OMISSION_ORDER,
    ASSESSMENT_CAPABILITIES,
    INTERPRETERS,
    SENSITIVITY_ROUTES,
    AssessmentItem,
    DiagnosticReport,
    LongitudinalDiagnostics,
    _defaults_to_ambiguous_estimand,
)
from cleverly.datasets import make_linear_ate, make_longitudinal, make_multi_arm
from cleverly.sensitivity import ConfounderStrengthGrid, PositivityReport, simulated_confounding
from cleverly.sensitivity._parameters import arm_parameters
from cleverly.sensitivity._simulated_confounding_request import (
    _FIT_WIDE_RULES,
    _LONGITUDINAL_REFUSAL,
    _MULTI_ARM_REFUSAL,
    _fit_wide_refusal,
)
from cleverly.sensitivity.positivity import positivity_report
from cleverly.validation.nuisance import nuisance_diagnostics
from tests.unit._confounding_support import forbid_draw_and_refit


@pytest.fixture(scope="module")
def point_result():  # type: ignore[no-untyped-def]
    frame, _ = make_linear_ate(n=350, seed=11)
    return (
        CausalStudy(
            frame,
            design=PointTreatment(
                outcome="Y",
                treatment="A",
                adjustment=["W1", "W2", "W3", "W4"],
            ),
        )
        .identify(ATE())
        .estimate(
            outcome_learner=sklearn.linear_model.LinearRegression(),
            treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
            n_folds=3,
            learner_folds=2,
            random_state=4,
            simultaneous=False,
        )
    )


@pytest.fixture(scope="module")
def longitudinal_result():  # type: ignore[no-untyped-def]
    frame, _ = make_longitudinal(n=400, seed=12)
    study = CausalStudy(
        frame,
        design=LongitudinalTreatment(
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W1", "W2"],
            time_varying=[[], ["L2"]],
            censoring=["C1", "C2"],
        ),
    )
    return study.identify(RegimeMean({"always": 1, "never": 0})).estimate(
        outcome_learner=sklearn.linear_model.LinearRegression(),
        pseudo_learner=sklearn.linear_model.LinearRegression(),
        treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
        censoring_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
        n_folds=3,
        learner_folds=2,
        random_state=5,
        simultaneous=False,
    )


@pytest.fixture(scope="module")
def multi_arm_result():  # type: ignore[no-untyped-def]
    frame, _ = make_multi_arm(n=180, seed=13)
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W1", "W2", "W3"),
        ),
    )
    return study.identify(ATE(reference="low")).estimate(
        outcome_learner=sklearn.linear_model.LinearRegression(),
        treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
        n_folds=2,
        learner_folds=2,
        random_state=13,
        simultaneous=False,
    )


@pytest.fixture(scope="module")
def att_result():  # type: ignore[no-untyped-def]
    """A fit whose one reported parameter is eligible and is not the bare ``"ate"``.

    This is the length-one case, which the multi-arm fixture cannot reach: it reports two
    contrasts, and a fit reporting a bare ``"ate"`` reports none. Every operation that
    defaults to ``estimand="ate"`` therefore faces exactly one eligible name here, and the
    two facades owe the caller different answers for it.
    """
    frame, _ = make_linear_ate(n=350, seed=11)
    return (
        CausalStudy(
            frame,
            design=PointTreatment(
                outcome="Y",
                treatment="A",
                adjustment=["W1", "W2", "W3", "W4"],
            ),
        )
        .identify(ATT())
        .estimate(
            outcome_learner=sklearn.linear_model.LinearRegression(),
            treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
            n_folds=3,
            learner_folds=2,
            random_state=4,
            simultaneous=False,
        )
    )


def test_every_diagnostic_operation_covers_every_result_family() -> None:
    expected = {
        "support",
        "nuisance_models",
        "score_equations",
        "corrections",
        "truncation_curve",
        "refute",
        "stagewise",
    }
    assert {item.result_family for item in ASSESSMENT_CAPABILITIES} == {"point", "longitudinal"}
    for family in ("point", "longitudinal"):
        family_rows = [item for item in ASSESSMENT_CAPABILITIES if item.result_family == family]
        declared = {item.operation for item in family_rows}
        assert declared == expected
        if family == "point":
            corrections = next(item for item in family_rows if item.operation == "corrections")
            assert set(corrections.methods) == {"drtmle"}
            assert all(item.methods for item in family_rows)
    assert len({(item.result_family, item.operation) for item in ASSESSMENT_CAPABILITIES}) == len(
        ASSESSMENT_CAPABILITIES
    )


@pytest.mark.parametrize("fixture_name", ["point_result", "longitudinal_result"])
def test_every_declared_method_is_constructible(request, fixture_name) -> None:  # type: ignore[no-untyped-def]
    result = request.getfixturevalue(fixture_name)
    for surface in (result.diagnostics, result.sensitivity):
        for row in surface.capabilities:
            for method in row.methods:
                constructed = result.identified_effect._method(method, {})
                assert constructed.name == method


def test_capabilities_declare_artifacts_cost_and_replay_semantics(point_result) -> None:  # type: ignore[no-untyped-def]
    for capability in point_result.diagnostics.capabilities:
        assert capability.interpretation
        assert capability.cost in {"cheap", "moderate", "expensive"}
        assert capability.execution in {"summarize", "retarget", "refit"}
        if not capability.available:
            assert capability.reason


def test_point_diagnostics_reuse_the_existing_numbers(point_result) -> None:  # type: ignore[no-untyped-def]
    """Each facade answer must equal what the function it routes to computes.

    Against the independent implementation, never against a second call to the facade:
    every one of these goes through ``_cached``, so ``facade.x() == facade.x()`` compares
    one cached object with itself and holds whatever the facade does.
    """
    score = point_result.diagnostics.score_equations()
    assert score == point_result.score_verdict

    nuisance = point_result.diagnostics.nuisance_models()
    assert nuisance == nuisance_diagnostics(point_result)

    support = point_result.diagnostics.support()
    assert support == positivity_report(point_result)


def test_longitudinal_stagewise_reports_one_row_per_node(longitudinal_result) -> None:  # type: ignore[no-untyped-def]
    frame = longitudinal_result.diagnostics.stagewise().to_frame()
    assert list(frame.columns) == [
        "regimen",
        "time",
        "n_followed",
        "share_assigned_1",
        "max_weight",
        "effective_n",
        "share_truncated",
        "epsilon",
        "converged",
    ]
    assert len(frame) == sum(len(fit.steps) for fit in longitudinal_result.fits.values())
    assert set(frame["share_assigned_1"]) == {0.0, 1.0}  # two static regimens


def test_longitudinal_stagewise_is_a_direct_support_alias(longitudinal_result) -> None:  # type: ignore[no-untyped-def]
    result = dataclasses.replace(longitudinal_result)

    stagewise = result.diagnostics.stagewise()

    assert stagewise is result.diagnostics.support()
    cache_operations = {key.split(":", 1)[0] for key in result.assessment_cache}
    assert "diagnostics.support" in cache_operations
    assert "diagnostics.stagewise" not in cache_operations


def test_only_the_longitudinal_stagewise_alias_is_excluded_from_combined_reports(
    longitudinal_result,
) -> None:  # type: ignore[no-untyped-def]
    excluded = {
        (row.result_family, row.operation)
        for row in ASSESSMENT_CAPABILITIES
        if not row.include_in_combined
    }
    assert excluded == {("longitudinal", "stagewise")}
    assert not longitudinal_result.diagnostics.capability("stagewise").include_in_combined

    combined = dataclasses.replace(longitudinal_result).diagnostics.run_all()
    names = [item.name for item in combined.items]
    assert names.count("support") == 1
    assert "stagewise" not in names
    assert combined.report("support").rows


def test_longitudinal_score_and_nuisance_adapters_cover_every_node(longitudinal_result) -> None:  # type: ignore[no-untyped-def]
    expected = sum(len(fit.steps) for fit in longitudinal_result.fits.values())
    scores = longitudinal_result.diagnostics.score_equations()
    nuisances = longitudinal_result.diagnostics.nuisance_models()
    mechanism_rows = longitudinal_result.data.n_times * 2
    assert len(nuisances.rows) == expected + mechanism_rows
    assert {row.role for row in nuisances.rows} == {
        "treatment",
        "censoring",
        "outcome",
        "pseudo_outcome",
    }
    # One row per node per question the node poses.  A cross-fitted node poses two -- did
    # every fold's solve reach its root, and is the stitched residual where sampling would
    # leave it -- and a single-fold node poses only the first.
    kinds = [row.kind for row in scores.rows]
    assert kinds.count("solver") == expected
    assert kinds.count("stitching") in {0, expected}
    assert len(scores.rows) == len(kinds)
    assert all(row.score >= 0 and row.relative_score >= 0 for row in scores.rows)
    assert all(row.n > 0 and row.reported_loss >= 0 for row in nuisances.rows)


def test_longitudinal_nuisance_capability_names_every_retained_artifact(
    longitudinal_result,
) -> None:  # type: ignore[no-untyped-def]
    capability = longitudinal_result.diagnostics.capability("nuisance_models")
    assert capability.required_artifacts == (
        "observed-law treatment predictions",
        "observed-law censoring predictions",
        "node pseudo-outcomes",
        "initial node predictions",
        "nuisance learner diagnostics",
    )
    assert capability.interpretation == (
        "weighted treatment, censoring, outcome, and pseudo-outcome fit by node and "
        "fitted recursion"
    )


@pytest.mark.parametrize("fixture_name", ["point_result", "longitudinal_result"])
def test_default_validation_is_immutable_cache_only_and_never_refits(
    request: pytest.FixtureRequest, fixture_name: str
) -> None:
    result = request.getfixturevalue(fixture_name)
    before = {name: estimate.psi for name, estimate in result.estimates.items()}
    report = result.validate()
    assert dataclasses.is_dataclass(report)
    assert report == result.validate()
    assert all(item.status in AssessmentStatus for item in report.items)
    assert not any("refute" in key for key in result.assessment_cache)
    assert {name: estimate.psi for name, estimate in result.estimates.items()} == before


def test_a_deferred_required_validation_does_not_pass() -> None:
    """A required check the caller deferred is not evidence that validation passed."""
    report = ValidationReport(
        (AssessmentItem("required", AssessmentStatus.DEFERRED, "waiting for a choice"),)
    )

    assert not report.passed
    assert not bool(report)


def test_combined_reports_distinguish_inapplicable_from_deferred(point_result) -> None:  # type: ignore[no-untyped-def]
    report = point_result.sensitivity.run_all()
    assert report["missingness"].status is AssessmentStatus.NOT_APPLICABLE
    assert report["benchmark"].status is AssessmentStatus.DEFERRED


def test_longitudinal_sensitivity_is_a_capability_aware_facade(longitudinal_result) -> None:  # type: ignore[no-untyped-def]
    report = longitudinal_result.sensitivity.run_all()
    assert {item.status for item in report.items} == {AssessmentStatus.UNAVAILABLE}
    with pytest.raises(CapabilityError, match="no longitudinal sensitivity derivation"):
        longitudinal_result.sensitivity.omitted_confounding()


def test_longitudinal_simulated_confounding_refuses_the_missing_scientific_law(  # type: ignore[no-untyped-def]
    longitudinal_result,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reason = _LONGITUDINAL_REFUSAL
    capability = longitudinal_result.sensitivity.capability("simulated_confounding")
    assert not capability.available
    assert capability.status is AssessmentStatus.UNAVAILABLE
    assert capability.reason == reason
    assert _fit_wide_refusal(longitudinal_result) == reason
    # A ``LongitudinalResult`` stores no replay estimator, so there is no ``refit`` to
    # forbid.  Asserted rather than assumed: the day it gains one, this guard must cover it.
    assert not hasattr(longitudinal_result, "estimator")
    forbid_draw_and_refit(monkeypatch, None)
    grid = ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,))
    with pytest.raises(CapabilityError) as facade_refusal:
        longitudinal_result.sensitivity.simulated_confounding(grid=grid)
    assert str(facade_refusal.value) == (
        "sensitivity 'simulated_confounding' is unavailable: " + reason
    )
    with pytest.raises(CapabilityError) as direct_refusal:
        simulated_confounding(longitudinal_result, grid=grid)
    assert str(direct_refusal.value) == reason


def test_multi_arm_simulated_confounding_capability_matches_direct_refusal(  # type: ignore[no-untyped-def]
    multi_arm_result,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reason = _MULTI_ARM_REFUSAL
    capability = multi_arm_result.sensitivity.capability("simulated_confounding")
    assert not capability.available
    assert capability.status is AssessmentStatus.UNAVAILABLE
    assert capability.reason == reason
    # A multi-arm fit reports one estimate per arm, so the surface cannot pick a default
    # alias. The row must ask for the estimand as well as the grid.
    assert capability.requires_arguments == ("grid", "estimand")
    assert _fit_wide_refusal(multi_arm_result) == reason
    forbid_draw_and_refit(monkeypatch, multi_arm_result.estimator)
    grid = ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,))
    # Both routes, because multi-arm is the case that regressed once. The facade read the
    # capability row and the direct call read the guard, and only the guard refused.
    with pytest.raises(CapabilityError) as facade_refusal:
        multi_arm_result.sensitivity.simulated_confounding(grid=grid)
    assert str(facade_refusal.value) == (
        "sensitivity 'simulated_confounding' is unavailable: " + reason
    )
    with pytest.raises(CapabilityError) as direct_refusal:
        simulated_confounding(multi_arm_result, grid=grid)
    assert str(direct_refusal.value) == reason


def test_an_ambiguous_evalue_row_says_it_is_deferred_rather_than_passed(  # type: ignore[no-untyped-def]
    multi_arm_result,
) -> None:
    """The row a caller reads must describe the call the caller then makes.

    A multi-arm fit reports two contrasts, so the bare E-value default is ambiguous. The
    row published ``available: True | status: passed`` beside the refusal sentence that
    says why it cannot run, and ``result.sensitivity.evalue()`` raised straight after. An
    explicit estimand still runs, because ``_dispatch`` skips the availability gate for it.
    """
    capability = multi_arm_result.sensitivity.capability("evalue")

    assert not capability.available
    assert capability.status is AssessmentStatus.DEFERRED
    assert capability.reason is not None
    assert "choose an explicit estimand" in capability.reason
    assert capability.requires_arguments == ("estimand",)
    with pytest.raises(CapabilityError, match="is deferred"):
        multi_arm_result.sensitivity.evalue()
    chosen = next(iter(multi_arm_result.estimates))
    assert multi_arm_result.sensitivity.evalue(estimand=chosen) is not None


def test_the_public_default_estimand_defers_exactly_as_the_bare_request_does(  # type: ignore[no-untyped-def]
    multi_arm_result,
) -> None:
    """One request, spelled two ways, may not reach two statuses.

    ``estimand=None`` is the documented public default, so ``arguments={"evalue":
    {"estimand": None}}`` asks for what a bare run asks for. The skip gate tested for a
    missing argument *key* while the E-value defers on the argument *value*, so this
    spelling passed both gates, invoked the operation, and returned through the generic
    post-invocation handler as ``unavailable`` with "call ... directly for the refusal in
    full". The status disagreed with the bare run and the next step named no argument.
    """
    bare = multi_arm_result.sensitivity.run_all()["evalue"]
    explicit = multi_arm_result.sensitivity.run_all(arguments={"evalue": {"estimand": None}})[
        "evalue"
    ]

    assert bare.status is AssessmentStatus.DEFERRED
    assert explicit.status is bare.status
    assert explicit.detail == bare.detail
    assert explicit.next_steps == bare.next_steps
    assert explicit.next_steps == ("call result.sensitivity.evalue() directly with estimand",)
    assert "choose an explicit estimand" in explicit.detail
    # The deferral is the caller's to lift, which is what separates it from unavailable.
    chosen = next(iter(multi_arm_result.estimates))
    completed = multi_arm_result.sensitivity.run_all(arguments={"evalue": {"estimand": chosen}})
    assert completed["evalue"].status is AssessmentStatus.COMPLETED


def test_a_saved_facade_carries_no_verdict_it_memoized(multi_arm_result, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A memo is derived state, and an artifact that keeps one pins a stale verdict.

    ``result.sensitivity`` is cached on the result and ``save`` pickles the result whole,
    so a facade the caller has touched wrote ``_evalue_selections`` into the artifact. A
    multi-arm result saved before ``deferred`` existed reloaded with
    ``{None: ("unavailable", ...)}`` inside it, and the row read that tuple instead of
    recomputing. The loaded result reported an unavailable E-value with no next step, past
    the ``sensitivity.run_all`` cache generation that exists to force the recompute.

    A same-version round trip pins the write side. The revived state below pins the read
    side, which is the migration case, and needs no committed binary artifact.
    """
    facade = multi_arm_result.sensitivity
    fresh = facade.run_all()["evalue"]
    assert fresh.status is AssessmentStatus.DEFERRED
    assert "_evalue_selections" in facade.__dict__
    assert "_capability_map" in facade.__dict__

    state = facade.__getstate__()
    assert "_evalue_selections" not in state
    assert "_capability_map" not in state
    assert "_declared" not in state
    assert state["_result"] is multi_arm_result

    restored = load(multi_arm_result.save(tmp_path / "warmed-evalue.joblib"))
    assert "_evalue_selections" not in restored.sensitivity.__dict__
    assert restored.sensitivity.capability("evalue").status is AssessmentStatus.DEFERRED
    assert restored.sensitivity.run_all()["evalue"] == fresh

    stale = dict(state)
    stale["_evalue_selections"] = {None: ("unavailable", "an older verdict for this fit")}
    revived = type(facade).__new__(type(facade))
    revived.__setstate__(stale)

    assert "_evalue_selections" not in revived.__dict__
    assert revived.capability("evalue").status is AssessmentStatus.DEFERRED
    assert revived.run_all()["evalue"] == fresh


#: Every operation that declares ``estimand="ate"`` and answers for one parameter, with
#: the facade it lives on.  ``evalue`` is covered above, and it was the only one of the
#: seven whose ambiguous default reported ``deferred``: the rest refused during invocation
#: and left the generic handler to publish ``unavailable`` with a next step that named no
#: argument.  One report then carried two taxonomies for one cause.
#:
#: ``tipping_gamma`` is the seventh and is missing here on purpose. It needs a fitted
#: missingness mechanism, which this fixture has none of, so
#: ``test_sensitivity_multi_arm.py::TestTheTippingSearchDefersItsChoiceBeforeItsCost``
#: covers it on a fit that does. It is also the only one that sits behind a cost flag as
#: well, which is what that class checks the order of.
_AMBIGUOUS_DEFAULT_OPERATIONS = [
    ("sensitivity", "omitted_confounding"),
    ("sensitivity", "robustness_value"),
    ("sensitivity", "elements"),
    ("sensitivity", "contour"),
    ("diagnostics", "refute"),
]

#: Flags that pay for every row above, so a cost gate can never be what these tests read.
_PAID = {"include_refits": True, "include_retargets": True}


@pytest.mark.parametrize("facade,operation", _AMBIGUOUS_DEFAULT_OPERATIONS)
def test_an_ambiguous_default_estimand_defers_and_runs_once_it_is_named(  # type: ignore[no-untyped-def]
    multi_arm_result, facade: str, operation: str
) -> None:
    """A choice the caller can make is a deferral, and the caller then makes it.

    Both halves matter and only together. The first is the taxonomy: this fit reports two
    contrasts and no bare ``"ate"``, which is a missing choice rather than a missing
    derivation, and the report says which argument settles it. The second is that the
    argument really does settle it. ``_skipped`` defers on the status alone, so a row that
    stayed ``deferred`` once the estimand arrived would defer for ever, and the first half
    would read exactly the same.
    """
    surface = getattr(multi_arm_result, facade)
    item = surface.run_all(**_PAID)[operation]

    assert item.status is AssessmentStatus.DEFERRED
    assert "choose an explicit estimand" in item.detail
    assert sorted(multi_arm_result.estimates)[0] in item.detail
    assert item.next_steps == (f"call result.{facade}.{operation}() directly with estimand",)
    multi_arm_result.assessment_cache.clear()

    chosen = next(iter(multi_arm_result.estimates))
    completed = surface.run_all(**_PAID, arguments={operation: {"estimand": chosen}})[operation]
    assert completed.status not in {
        AssessmentStatus.DEFERRED,
        AssessmentStatus.NOT_APPLICABLE,
        AssessmentStatus.UNAVAILABLE,
    }
    assert completed.arguments["estimand"] == chosen
    multi_arm_result.assessment_cache.clear()


@pytest.mark.parametrize("facade,operation", _AMBIGUOUS_DEFAULT_OPERATIONS)
def test_the_public_default_spelled_out_defers_exactly_as_the_bare_request_does(  # type: ignore[no-untyped-def]
    multi_arm_result, facade: str, operation: str
) -> None:
    """One request, spelled two ways, may not reach two statuses.

    ``estimand=None`` asks for the documented default. The gate therefore reads the
    supplied *value* rather than the key, which is the split verdict the E-value row met
    first: the membership test let this spelling through to the invocation, and the
    refusal came back as ``unavailable`` while the bare request said ``deferred``.
    """
    surface = getattr(multi_arm_result, facade)
    bare = surface.run_all(**_PAID)[operation]
    multi_arm_result.assessment_cache.clear()
    spelled = surface.run_all(**_PAID, arguments={operation: {"estimand": None}})[operation]

    assert spelled.status is bare.status is AssessmentStatus.DEFERRED
    assert spelled.detail == bare.detail
    assert spelled.next_steps == bare.next_steps
    multi_arm_result.assessment_cache.clear()


@pytest.mark.parametrize("facade,operation", _AMBIGUOUS_DEFAULT_OPERATIONS)
def test_an_estimand_the_caller_named_and_the_fit_never_reported_stays_unavailable(  # type: ignore[no-untyped-def]
    multi_arm_result, facade: str, operation: str
) -> None:
    """Only the ambiguity moved. A refused *name* is still an unavailable operation.

    These operations say "estimand 'ate' was not requested in this fit" for both cases,
    which is why they have to be told apart here rather than by the sentence. Reporting a
    name the fit does not report as ``deferred`` would answer the caller's explicit
    argument by asking for that argument again.
    """
    surface = getattr(multi_arm_result, facade)
    item = surface.run_all(**_PAID, arguments={operation: {"estimand": "ate[nope vs low]"}})[
        operation
    ]

    assert item.status is AssessmentStatus.UNAVAILABLE
    assert "declined this request" in item.detail
    multi_arm_result.assessment_cache.clear()


@pytest.mark.parametrize("facade,operation", _AMBIGUOUS_DEFAULT_OPERATIONS)
def test_a_reported_ate_settles_the_choice_and_nothing_defers(  # type: ignore[no-untyped-def]
    point_result, facade: str, operation: str
) -> None:
    """The control: the deferral is the fit's ambiguity, not a new demand on every fit.

    A two-armed fit reports a bare ``"ate"``, which is what every one of these operations
    defaults to, so there is no choice left open and the row runs argument-free. Without
    this the tests above would pass on a change that deferred these rows unconditionally.
    """
    surface = getattr(point_result, facade)
    assert surface._estimand_candidates(operation) == ()
    item = surface.run_all(**_PAID)[operation]

    assert item.status not in {
        AssessmentStatus.DEFERRED,
        AssessmentStatus.NOT_APPLICABLE,
        AssessmentStatus.UNAVAILABLE,
    }
    point_result.assessment_cache.clear()


def test_a_saved_aggregate_from_before_the_deferral_is_not_replayed(  # type: ignore[no-untyped-def]
    multi_arm_result, tmp_path
) -> None:
    """The migration case: the stale row is inside the artifact, not in the code.

    ``run_all`` is cached on the result and ``save`` writes that cache, so a multi-arm
    result saved before this change carries a combined report whose bound says
    ``unavailable`` with no argument to act on. The cache key is what rejects it, so both
    aggregates carry a generation and both had to move. Without the bump the loaded result
    hits the old entry and republishes the old taxonomy.
    """
    result = dataclasses.replace(multi_arm_result)
    fresh = result.sensitivity.run_all()
    assert fresh["omitted_confounding"].status is AssessmentStatus.DEFERRED

    current = next(key for key in result.assessment_cache if key.startswith("sensitivity.run_all:"))
    result.assessment_cache.clear()
    result.assessment_cache[_with_cache_generation(current, 1)] = "a verdict from before"
    restored = load(result.save(tmp_path / "stale-multi-arm-aggregate.joblib"))

    assert restored.sensitivity.run_all() == fresh
    assert restored.diagnostics.run_all()["refute"].status is AssessmentStatus.DEFERRED


def test_one_predicate_answers_the_row_and_the_substitution(multi_arm_result) -> None:  # type: ignore[no-untyped-def]
    """The deferral and the filled-in argument must not be able to disagree.

    ``_with_default_parameter`` declined to guess between two contrasts while the row
    beside it advertised the analysis as runnable, and that split is the whole defect:
    the combined report invoked an operation its own capability had already decided it
    could not choose an argument for. Both now read
    :meth:`_CapabilityFacade._estimand_candidates`, so this asserts the two answers
    against the one predicate rather than against each other.
    """
    facade = multi_arm_result.sensitivity
    candidates = facade._estimand_candidates("omitted_confounding")

    assert len(candidates) > 1
    assert set(candidates) == set(multi_arm_result.estimates)
    # Ambiguous, so nothing is substituted and the row defers.
    assert facade._with_default_parameter("omitted_confounding", (), {}) == ((), {})
    assert facade._capability_for_arguments("omitted_confounding", {}).status is (
        AssessmentStatus.DEFERRED
    )
    # Named, so the row is the declared one again and the operation runs.
    chosen = candidates[0]
    resolved = facade._capability_for_arguments("omitted_confounding", {"estimand": chosen})
    assert resolved == facade.capability("omitted_confounding")
    assert resolved.available and resolved.requires_arguments == ()


def test_a_keyword_only_ambiguous_default_is_gated_like_a_positional_one(  # type: ignore[no-untyped-def]
    multi_arm_result,
) -> None:
    """``benchmark`` declares the same default, so it owes the same answer.

    The gate read ``SENSITIVITY_ROUTES[...].needs_estimand``, which says where the estimand
    goes rather than whether there is one. ``benchmark`` takes ``covariates`` positionally
    and ``estimand="ate"`` by keyword, so that flag is False and the row was never gated:
    one report published six ``deferred`` rows and one ``unavailable`` row for one missing
    choice. The predicate is the signature now, and the two facts are asserted together so
    that reading the flag again fails here rather than in a report.
    """
    facade = multi_arm_result.sensitivity
    function, _ = facade._routed_callable("benchmark")
    assert not SENSITIVITY_ROUTES["benchmark"].needs_estimand
    assert _defaults_to_ambiguous_estimand(function)
    parameter = inspect.signature(function).parameters["estimand"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY

    covariates = {"benchmark": {"covariates": ["W1"]}}
    item = facade.run_all(**_PAID, arguments=covariates)["benchmark"]

    assert item.status is AssessmentStatus.DEFERRED
    assert facade._estimand_candidates("benchmark") == tuple(multi_arm_result.estimates)
    assert all(name in item.detail for name in multi_arm_result.estimates)
    assert item.next_steps == (
        "call result.sensitivity.benchmark() directly with covariates and estimand",
    )
    multi_arm_result.assessment_cache.clear()

    chosen = next(iter(multi_arm_result.estimates))
    named = {"benchmark": {"covariates": ["W1"], "estimand": chosen}}
    ran = facade.run_all(**_PAID, arguments=named)["benchmark"]

    assert ran.status is AssessmentStatus.COMPLETED
    assert ran.arguments["estimand"] == chosen
    multi_arm_result.assessment_cache.clear()


def test_the_ambiguous_default_is_read_from_every_routed_signature(point_result) -> None:  # type: ignore[no-untyped-def]
    """The set the predicate finds is the set three documents describe.

    ``_default_estimand_candidates`` names eight operations, the user guide tables seven
    of them beside ``evalue``, and the technical reference tables the same. A ninth
    operation is gated by the predicate on the day it is written, and this fails until
    the three counts are corrected with it.
    """
    surfaces = {
        "sensitivity": set(SENSITIVITY_ROUTES),
        "diagnostics": {row.operation for row in point_result.diagnostics.capabilities},
    }
    found = {
        name: {
            operation
            for operation in operations
            if _defaults_to_ambiguous_estimand(
                getattr(point_result, name)._routed_callable(operation)[0]
            )
        }
        for name, operations in surfaces.items()
    }

    assert found["sensitivity"] == {
        "omitted_confounding",
        "robustness_value",
        "elements",
        "benchmark",
        "contour",
        "tipping_gamma",
        "simulated_confounding",
    }
    assert found["diagnostics"] == {"refute"}
    assert len(found["sensitivity"] | found["diagnostics"]) == 8
    # ``evalue`` declares an ``estimand`` and selects for itself from a ``None`` default,
    # which is why the predicate reads the default rather than the parameter's name.
    assert "evalue" in surfaces["sensitivity"] - found["sensitivity"]


def test_a_sole_eligible_name_reaches_a_keyword_only_estimand_too(att_result) -> None:  # type: ignore[no-untyped-def]
    """The substitution half of the same defect, on the fit that shows it.

    This fit leaves one eligible parameter, and the sensitivity facade fills it in. It
    filled in a positional estimand alone, so ``omitted_confounding`` reported
    ``completed`` while ``benchmark`` reported ``unavailable`` beside it: two taxonomies,
    one fit, one cause. Both rows now run, and ``benchmark`` records the name that ran.
    """
    facade = att_result.sensitivity
    assert facade._estimand_candidates("benchmark") == ("att",)
    args, kwargs = facade._with_default_parameter("benchmark", (), {"covariates": ["W1"]})
    assert args == ()
    assert kwargs["estimand"] == "att"

    report = facade.run_all(**_PAID, arguments={"benchmark": {"covariates": ["W1"]}})

    assert report["benchmark"].status is AssessmentStatus.COMPLETED
    assert report["omitted_confounding"].status is AssessmentStatus.COMPLETED
    assert report["benchmark"].arguments["estimand"] == "att"
    # A caller who writes the covariates positionally reaches the same substitution. The
    # positional argument is not an estimand, so it settles no choice.
    assert facade.benchmark(["W1"]).estimand == "att"
    att_result.assessment_cache.clear()


def test_a_facade_that_substitutes_nothing_defers_a_sole_eligible_name(att_result) -> None:  # type: ignore[no-untyped-def]
    """One candidate is a substitution here and a deferral there, by facade.

    ``refute`` has no name to fall back on, so nothing fills the gap: the row fell past a
    gate that defers at two candidates, ``refute`` ran on its ambiguous ``"ate"``, and the
    generic handler published ``unavailable`` with a next step that named no argument.
    ``unavailable`` means this fit cannot run the operation, and naming the one reported
    alias runs it. The sensitivity half of the same fit is the paired witness: identical
    count, and a row that runs because that facade supplies the name.
    """
    diagnostics, sensitivity = att_result.diagnostics, att_result.sensitivity
    assert not diagnostics._substitutes_estimand
    assert sensitivity._substitutes_estimand
    assert diagnostics._estimand_candidates("refute") == ("att",)
    assert sensitivity._estimand_candidates("omitted_confounding") == ("att",)

    item = diagnostics.run_all(**_PAID, random_state=3)["refute"]

    assert item.status is AssessmentStatus.DEFERRED
    assert "choose an explicit estimand from ['att']" in item.detail
    assert item.next_steps == ("call result.diagnostics.refute() directly with estimand",)
    assert dict(item.arguments) == {"random_state": 3}
    att_result.assessment_cache.clear()

    named = diagnostics.run_all(**_PAID, random_state=3, arguments={"refute": {"estimand": "att"}})[
        "refute"
    ]

    assert named.status not in _OMISSIONS
    assert named.arguments["estimand"] == "att"
    assert sensitivity.run_all(**_PAID)["omitted_confounding"].status is (
        AssessmentStatus.COMPLETED
    )
    att_result.assessment_cache.clear()


@pytest.mark.parametrize(
    "fixture_name,operation",
    [
        ("longitudinal_result", "benchmark"),
        ("longitudinal_result", "simulated_confounding"),
        ("multi_arm_result", "simulated_confounding"),
    ],
)
def test_a_row_this_fit_refuses_outright_records_no_invocation(  # type: ignore[no-untyped-def]
    request, point_result, fixture_name: str, operation: str
) -> None:
    """A fit-wide refusal describes no call, so it carries no arguments to replay.

    The combined run injects its seed before the gates, which is what a deferred row
    needs. It reached this row too, and "no longitudinal benchmarking derivation is
    registered" arrived carrying ``{'random_state': 7}``: an invocation that never
    happened and that no argument can produce. ``benchmark`` on a point fit is the paired
    witness. Same operation, same seed, and a row that answers a request the caller can
    complete, so it keeps the request.
    """
    result = request.getfixturevalue(fixture_name)
    surface = result.sensitivity
    assert surface.capability(operation).accepts_random_state

    refused = surface.run_all(**_PAID, random_state=7)[operation]
    deferred = point_result.sensitivity.run_all(**_PAID, random_state=7)["benchmark"]

    assert refused.status is AssessmentStatus.UNAVAILABLE
    assert dict(refused.arguments) == {}
    assert deferred.status is AssessmentStatus.DEFERRED
    assert dict(deferred.arguments) == {"random_state": 7}
    result.assessment_cache.clear()
    point_result.assessment_cache.clear()


def _deferred_argument_values(result) -> dict[str, object]:  # type: ignore[no-untyped-def]
    """A value for every argument a deferred capability can declare on this fit.

    The test below supplies the request that lifts a deferral rather than guessing at one.
    A new required argument raises ``KeyError`` here, which is the point: nobody can add
    one without saying what supplying it looks like.
    """
    return {
        "estimand": next(iter(result.estimates)),
        "grid": ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,)),
        # Every fixture in this module adjusts for ``W1``.
        "covariates": ["W1"],
    }


@pytest.mark.parametrize(
    "fixture_name,facade,defers",
    [
        ("point_result", "diagnostics", False),
        ("point_result", "sensitivity", False),
        ("multi_arm_result", "diagnostics", True),
        ("multi_arm_result", "sensitivity", True),
        ("att_result", "diagnostics", True),
        ("att_result", "sensitivity", False),
        ("longitudinal_result", "diagnostics", False),
        ("longitudinal_result", "sensitivity", False),
    ],
)
def test_a_deferred_capability_is_never_deferred_once_its_arguments_arrive(  # type: ignore[no-untyped-def]
    request, fixture_name: str, facade: str, defers: bool
) -> None:
    """The constraint ``_skipped``'s first gate depends on, asserted over real producers.

    That gate defers on the status and the declared arguments alone. It does not check
    whether the caller already supplied them, because no producer can reach it in that
    state. Nothing said so in a test, so a future declaration carrying a static
    ``DEFERRED`` row with ``requires_arguments`` would be unrunnable through ``run_all``
    for ever, and would look like an ordinary deferral while it was.

    Every capability each facade resolves for a request is read here, rather than a
    hand-built row, so a new producer is covered on the day it is written. The ``defers``
    column is the nonzero witness: without it a change that deferred nothing would pass
    this loop by never entering it.
    """
    result = request.getfixturevalue(fixture_name)
    surface = getattr(result, facade)
    values = _deferred_argument_values(result)

    deferred = []
    for declared in surface.capabilities:
        capability = surface._capability_for_arguments(declared.operation, {})
        if capability.status is not AssessmentStatus.DEFERRED:
            continue
        deferred.append(capability.operation)
        assert capability.requires_arguments, f"{capability.operation} names no argument"
        supplied = {name: values[name] for name in capability.requires_arguments}
        resolved = surface._capability_for_arguments(capability.operation, supplied)
        assert resolved.status is not AssessmentStatus.DEFERRED, capability.operation

    assert bool(deferred) is defers, deferred


@dataclasses.dataclass(frozen=True)
class _DelegatingBackdoorProvider:
    """A custom ``IdentificationProvider`` that reuses the built-in backdoor derivation.

    The ``provider=`` argument of :meth:`cleverly.CausalStudy.identify` is documented and
    public, so a reader can reach this state without touching a private name. The provider
    delegates the derivation and then stamps itself, which is what any real third-party
    provider does. ``simulated_confounding`` reads the stamp, and it requires specifically
    an ``ExplicitAdjustmentProvider``.
    """

    name: str = "delegating-backdoor"

    def identify(self, study, estimand):  # type: ignore[no-untyped-def]
        """Delegate the derivation, then record this provider on the identified effect."""
        effect = ExplicitAdjustmentProvider().identify(study, estimand)
        return dataclasses.replace(effect, provider=self)


@pytest.fixture(scope="module")
def custom_provider_result():  # type: ignore[no-untyped-def]
    """Fit an ordinary point ATE whose only unsupported feature is its provider."""
    frame, _ = make_linear_ate(n=200, seed=17)
    study = CausalStudy(
        frame,
        design=PointTreatment(outcome="Y", treatment="A", adjustment=["W1", "W2", "W3", "W4"]),
    )
    provider = _DelegatingBackdoorProvider()
    assert isinstance(provider, IdentificationProvider)
    return study.identify(ATE(), provider=provider).estimate(
        outcome_learner=sklearn.linear_model.LinearRegression(),
        treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
        n_folds=2,
        learner_folds=2,
        random_state=3,
        simultaneous=False,
    )


def test_a_custom_provider_fit_never_advertises_a_surface_that_refuses_it(  # type: ignore[no-untyped-def]
    custom_provider_result,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The capability row and the execution guard must read the same table.

    This is the regression witness. The capability row consulted only the first six
    fit-wide rules, and the execution guard consulted all eighteen. Every boundary past
    the sixth therefore reported ``available=True``, ``PASSED`` and ``reason=None``, and
    then raised when the caller took the row at its word.

    ``provider`` is the last rule in the table, so it is the furthest a public call can
    reach, and ``provider=`` is a documented extension point rather than a tampered field.
    The fit is otherwise ordinary: same design, same learners, same estimand as
    ``point_result``. Only the identification provider differs.
    """
    result = custom_provider_result
    assert type(result.identified_effect.provider) is _DelegatingBackdoorProvider
    # The nonzero control: everything the first six rules test is fine on this fit, so a
    # refusal here can only come from a rule beyond them.
    assert result.assessment_family == "point"
    assert result.data.is_binary_treatment
    assert not result.data.has_missing_outcome
    assert not result.data.has_intermediate
    assert result.data.cluster is None

    capability = result.sensitivity.capability("simulated_confounding")
    forbid_draw_and_refit(monkeypatch, result.estimator)
    grid = ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,))
    with pytest.raises(CapabilityError) as direct_refusal:
        simulated_confounding(result, grid=grid)

    assert not capability.available
    assert capability.status is AssessmentStatus.UNAVAILABLE
    assert capability.reason == str(direct_refusal.value)
    assert _fit_wide_refusal(result) == str(direct_refusal.value)
    with pytest.raises(CapabilityError) as facade_refusal:
        result.sensitivity.simulated_confounding(grid=grid)
    assert str(facade_refusal.value) == (
        "sensitivity 'simulated_confounding' is unavailable: " + capability.reason
    )


#: One ``dataclasses.replace`` per fit-wide rule that the capability row did not read
#: before this change, named by the rule it reaches.  Each edit is minimal, so the rules
#: before it in ``_FIT_WIDE_RULES`` still pass and the named rule is the one that fires.
_LATE_RULE_CASES: dict[str, typing.Callable[[typing.Any], typing.Any]] = {
    "missing_estimator": lambda result: dataclasses.replace(result, estimator=None),
    "outcome_family": lambda result: dataclasses.replace(
        result, data=dataclasses.replace(result.data, family="poisson")
    ),
    "weight_kind": lambda result: dataclasses.replace(
        result,
        data=dataclasses.replace(
            result.data,
            weight_spec=dataclasses.replace(result.data.weight_spec, kind="frequency"),
        ),
    ),
    "weight_provenance": lambda result: dataclasses.replace(
        result,
        data=dataclasses.replace(
            result.data,
            weight_spec=dataclasses.replace(result.data.weight_spec, name="wrong"),
        ),
    ),
    "identification": lambda result: dataclasses.replace(result, identified_effect=None),
    "functional": lambda result: dataclasses.replace(
        result,
        identified_effect=dataclasses.replace(
            result.identified_effect, functional=types.SimpleNamespace()
        ),
    ),
    "provider": lambda result: dataclasses.replace(
        result,
        identified_effect=dataclasses.replace(
            result.identified_effect, provider=types.SimpleNamespace()
        ),
    ),
}


@pytest.mark.parametrize("rule", sorted(_LATE_RULE_CASES))
def test_every_late_fit_wide_rule_reaches_the_capability_row(  # type: ignore[no-untyped-def]
    point_result, monkeypatch: pytest.MonkeyPatch, rule: str
) -> None:
    """Sample the rules the capability row skipped, not the one that a public fit reaches.

    ``test_a_custom_provider_fit_never_advertises_a_surface_that_refuses_it`` reaches the
    last rule through the public API and is the load-bearing witness. It pins one boundary.
    These cases carry a tampered fitted result, which is how the neighbouring refusal tests
    construct a state no supported fit produces, and they check the same agreement across a
    sample of the rules that the row previously never read.
    """
    assert rule in {name for name, _ in _FIT_WIDE_RULES}
    tampered = _LATE_RULE_CASES[rule](point_result)
    capability = tampered.sensitivity.capability("simulated_confounding")
    forbid_draw_and_refit(monkeypatch, point_result.estimator)
    with pytest.raises(CapabilityError) as refusal:
        simulated_confounding(
            tampered, grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,))
        )

    assert not capability.available, rule
    assert capability.status is AssessmentStatus.UNAVAILABLE, rule
    assert capability.reason == str(refusal.value), rule
    assert _fit_wide_refusal(tampered) == str(refusal.value), rule
    # The nonzero control: the untampered fit advertises the row and does not refuse.
    intact = point_result.sensitivity.capability("simulated_confounding")
    assert intact.available and intact.reason is None


def test_a_refusal_gives_the_reason_its_own_capability_declared(  # type: ignore[no-untyped-def]
    longitudinal_result,
) -> None:
    """The reason is per operation, not per family.

    An E-value has no mechanism bound in it, so refusing one with a sentence about
    sequential pseudo-outcome recursion is a true statement about a different analysis.
    """
    for operation in ("evalue", "robustness_value", "missingness"):
        declared = longitudinal_result.sensitivity.capability(operation)
        with pytest.raises(CapabilityError, match=re.escape(declared.reason)):
            getattr(longitudinal_result.sensitivity, operation)()


def test_every_sensitivity_operation_is_declared_and_routed(  # type: ignore[no-untyped-def]
    point_result, longitudinal_result
) -> None:
    """Both directions, as for the diagnostics table.

    The gate that would have caught ``tipping_gamma`` being the one operation of its
    signature shape left out of the default-estimand set.
    """
    for result in (point_result, longitudinal_result):
        declared = {row.operation for row in result.sensitivity.capabilities}
        assert declared == set(SENSITIVITY_ROUTES)
        for operation in declared:
            assert callable(getattr(result.sensitivity, operation))
    # ``needs_estimand`` is a claim about the target's signature, so read the signature.
    # ``evalue`` takes ``estimand`` there too but defaults it to ``None`` and selects for
    # itself, which is why the flag is not simply the parameter's name.
    for route in SENSITIVITY_ROUTES.values():
        module = importlib.import_module(f"cleverly.sensitivity.{route.module}")
        parameter = list(inspect.signature(getattr(module, route.function)).parameters.values())[1]
        assert route.needs_estimand == (parameter.name == "estimand" and parameter.default == "ate")


def test_every_seeded_combined_route_accepts_random_state_in_its_real_signature(
    point_result,
) -> None:  # type: ignore[no-untyped-def]
    for surface in (point_result.diagnostics, point_result.sensitivity):
        for capability in surface.capabilities:
            if not capability.accepts_random_state:
                continue
            function, _ = surface._routed_callable(capability.operation)
            assert "random_state" in inspect.signature(function).parameters


def test_sensitivity_routing_reads_structured_parameter_keys(point_result) -> None:  # type: ignore[no-untyped-def]
    routed = arm_parameters(point_result)
    key = point_result.parameter_keys["ate"]
    assert routed["ate"].stem == key.estimand
    assert point_result.data.arm_label(routed["ate"].arm) == key.value
    assert point_result.data.arm_label(routed["ate"].versus) == key.reference


@pytest.mark.parametrize("fixture_name", ["point_result", "longitudinal_result"])
def test_cached_assessments_replay_after_persistence(
    request: pytest.FixtureRequest, fixture_name: str, tmp_path
) -> None:
    result = request.getfixturevalue(fixture_name)
    validation = result.validate()
    diagnostics = result.diagnostics.run_all()
    cache_keys = set(result.assessment_cache)
    restored = load(result.save(tmp_path / f"{fixture_name}.joblib"))

    assert set(restored.assessment_cache) == cache_keys
    assert restored.validate() == validation
    assert restored.diagnostics.run_all() == diagnostics


def test_a_pre_method_aware_nuisance_report_uses_additive_defaults(point_result, tmp_path) -> None:
    """An artifact pickled before the method-aware fields reads back and still renders.

    Every field this change added carries a class-level default, so a legacy instance
    resolves it through the class rather than through its own ``__dict__``. Asserting the
    six attribute values alone does not test that: they resolve the same way whether or
    not the instance ever lost them.

    What a legacy artifact actually does is get *rendered*, and each renderer reads the
    new fields directly. ``summary``, ``verdict``, ``findings``, the two frames and the
    combined row are therefore the assertions here. An attribute that stops carrying a
    default raises inside one of those, and nowhere else.
    """
    import joblib

    report = nuisance_diagnostics(point_result)
    live_summary = report.summary()
    live_findings = report.findings
    added = (
        "selection",
        "treatment_role",
        "repeat_spread",
        "selection_omission",
        "repeat_spread_omission",
        "reported_repeat",
    )
    for name in added:
        object.__delattr__(report, name)

    path = tmp_path / "legacy-nuisance-report.joblib"
    joblib.dump(report, path)
    restored = joblib.load(path)

    assert restored is not report
    # The round trip reconstructs the legacy shape rather than backfilling it, which is
    # what makes the renderings below run against a genuinely older instance.
    assert not set(added) & set(vars(restored))

    assert restored.selection is None
    assert restored.treatment_role is None
    assert restored.repeat_spread == ()
    assert restored.selection_omission is None
    assert restored.repeat_spread_omission is None
    assert restored.reported_repeat == 1

    summary = restored.summary()
    assert summary == live_summary
    assert restored.verdict() in summary
    assert restored.verdict().startswith("VERDICT:")
    assert "C-TMLE" not in summary
    assert "Repeated-split sensitivity" not in summary
    assert "split spread unavailable" not in summary
    # Non-empty, so the calibration rule runs its ``treatment_role`` gate on a live note
    # rather than on an empty tuple that a broken gate would also produce.
    assert live_findings
    assert restored.findings == live_findings
    assert list(restored.to_frame()["model"]) == [model.name for model in restored.models]
    assert len(restored.repeat_spread_frame()) == 0

    item = INTERPRETERS["nuisance_models"](restored, None)
    assert item.status is AssessmentStatus.WARNING
    assert item.detail == "; ".join(live_findings)

    # The repeated branches read three of the absent fields and are unreachable at one
    # draw, so the draw count is raised on the same legacy instance to reach them.
    object.__setattr__(restored, "n_repeats", 3)
    repeated_summary = restored.summary()
    assert "draw 1 of 3" in repeated_summary
    assert "split spread unavailable" not in repeated_summary
    assert "split spread" not in INTERPRETERS["nuisance_models"](restored, None).detail


def _without_cache_generation(key: str) -> str:
    """Rewrite a current cache key as the unversioned key an older result carries."""
    operation, encoded = key.split(":", 1)
    payload = json.loads(encoded)
    payload.pop("cache_generation")
    return f"{operation}:{json.dumps(payload, sort_keys=True, separators=(',', ':'))}"


def _with_cache_generation(key: str, generation: int) -> str:
    """Rewrite a current cache key as an older versioned result carries it."""
    operation, encoded = key.split(":", 1)
    payload = json.loads(encoded)
    payload["cache_generation"] = generation
    return f"{operation}:{json.dumps(payload, sort_keys=True, separators=(',', ':'))}"


def _with_previous_generation(key: str) -> str:
    """Rewrite a current cache key as the generation directly below it.

    Read from ``_CACHE_GENERATIONS`` rather than written as a literal. A seed one below the
    current number is what puts a bump itself under test, and two bumps in a row each had
    to be chased through hand-edited literals in three tests. A stale literal stops testing
    the bump silently, because a miss on a generation nobody uses passes either way.
    """
    operation = key.split(":", 1)[0]
    return _with_cache_generation(key, _CACHE_GENERATIONS[operation] - 1)


def test_changed_assessment_schemas_ignore_persisted_unversioned_cache_entries(
    point_result, tmp_path
) -> None:  # type: ignore[no-untyped-def]
    """Old support, aggregate, and validation entries are cache misses after loading."""
    result = dataclasses.replace(point_result)
    result.diagnostics.support()
    result.diagnostics.nuisance_models()
    result.diagnostics.run_all()
    result.sensitivity.run_all()
    result.validate()
    versioned = {
        key: value
        for key, value in result.assessment_cache.items()
        if key.split(":", 1)[0]
        in {
            "diagnostics.support",
            "diagnostics.nuisance_models",
            "diagnostics.run_all",
            "sensitivity.run_all",
            "validate",
        }
    }
    assert {key.split(":", 1)[0] for key in versioned} == {
        "diagnostics.support",
        "diagnostics.nuisance_models",
        "diagnostics.run_all",
        "sensitivity.run_all",
        "validate",
    }

    result.assessment_cache.clear()
    legacy_keys = {_without_cache_generation(key) for key in versioned}
    # Generation one covers the sensitivity aggregate's own immediately previous version:
    # both aggregates now report ``deferred`` where they reported ``unavailable`` for an
    # ambiguous default estimand, and a result saved before that carries the old row in
    # its own cache.
    generation_one_keys = {_with_cache_generation(key, 1) for key in versioned}
    previous_diagnostic_aggregate = {
        _with_previous_generation(key)
        for key in versioned
        if key.startswith("diagnostics.run_all:")
    }
    stale_keys = legacy_keys | generation_one_keys | previous_diagnostic_aggregate
    result.assessment_cache.update(dict.fromkeys(stale_keys, "legacy cached report"))
    restored = load(result.save(tmp_path / "legacy-assessment-cache.joblib"))

    assert restored.diagnostics.support().group_leverage
    assert restored.diagnostics.nuisance_models() != "legacy cached report"
    assert restored.diagnostics.run_all() != "legacy cached report"
    assert restored.sensitivity.run_all() != "legacy cached report"
    assert restored.validate() != "legacy cached report"
    assert restored.assess().diagnostics != "legacy cached report"
    assert restored.assess().sensitivity != "legacy cached report"
    assert stale_keys <= set(restored.assessment_cache)
    assert any(
        "cache_generation" in key and key.startswith("diagnostics.support:")
        for key in restored.assessment_cache
    )
    assert any(
        "cache_generation" in key and key.startswith("diagnostics.nuisance_models:")
        for key in restored.assessment_cache
    )
    assert any(
        "cache_generation" in key and key.startswith("sensitivity.run_all:")
        for key in restored.assessment_cache
    )


def test_longitudinal_alias_and_aggregate_ignore_pre_change_cache_entries(
    longitudinal_result, tmp_path
) -> None:  # type: ignore[no-untyped-def]
    result = dataclasses.replace(longitudinal_result)
    result.diagnostics.support()
    result.diagnostics.run_all()
    current = {
        key: value
        for key, value in result.assessment_cache.items()
        if key.split(":", 1)[0] in {"diagnostics.support", "diagnostics.run_all"}
    }
    assert {key.split(":", 1)[0] for key in current} == {
        "diagnostics.support",
        "diagnostics.run_all",
    }

    result.assessment_cache.clear()
    stale = {_with_previous_generation(key) for key in current}
    result.assessment_cache.update(dict.fromkeys(stale, "legacy cached report"))
    result.assessment_cache['diagnostics.stagewise:{"args":[],"kwargs":{}}'] = (
        "legacy stagewise report"
    )
    restored = load(result.save(tmp_path / "legacy-longitudinal-assessment-cache.joblib"))

    # Compared against the value the support key was seeded with. The alias delegates to
    # ``support`` and reads no key of its own, so an inequality against the stagewise
    # sentinel holds however the support generation resolves. The positive assertions say
    # the miss produced a recomputed report rather than any other object.
    stagewise = restored.diagnostics.stagewise()
    assert stagewise != "legacy cached report"
    assert isinstance(stagewise, LongitudinalDiagnostics)
    assert {row.regimen for row in stagewise.rows} == {"always", "never"}
    assert {row.time for row in stagewise.rows} == {1, 2}
    combined = restored.diagnostics.run_all()
    assert combined != "legacy cached report"
    assert [item.name for item in combined.items].count("support") == 1
    assert "stagewise" not in {item.name for item in combined.items}
    assert stale <= set(restored.assessment_cache)


def test_the_validation_report_ignores_its_immediately_previous_generation(
    point_result, tmp_path
) -> None:  # type: ignore[no-untyped-def]
    """The ``validate`` entry one generation below the current one.

    The unversioned and generation-one seeds miss whatever the current number is, so they
    hold while it stays anything above one and say nothing about a bump. Seeding the
    generation directly below the current one is what puts the bump itself under test.
    """
    result = dataclasses.replace(point_result)
    result.validate()
    keys = [key for key in result.assessment_cache if key.split(":", 1)[0] == "validate"]
    assert len(keys) == 1

    result.assessment_cache.clear()
    stale = _with_previous_generation(keys[0])
    result.assessment_cache[stale] = "pre-generation validation report"
    restored = load(result.save(tmp_path / "previous-generation-validate.joblib"))

    report = restored.validate()
    assert report != "pre-generation validation report"
    assert isinstance(report, ValidationReport)
    assert report.items
    assert stale in restored.assessment_cache


def test_positivity_report_preserves_its_pre_leverage_positional_slots() -> None:
    """Appending diagnostics must not reinterpret the former repeat and backend slots."""
    report = PositivityReport(
        {},
        {},
        {},
        {},
        {"fraction": 0.0},
        {},
        (0.01, 0.99),
        10,
        {},
        (),
        0.0,
        0.0,
        3,
        "pandas",
    )

    assert report.n_repeats == 3
    assert report.backend == "pandas"
    assert report.group_leverage == {}


def test_a_cached_frame_replays_in_the_callers_backend(point_result, tmp_path) -> None:  # type: ignore[no-untyped-def]
    before = point_result.diagnostics.truncation_curve(bounds=[0.02, 0.05])
    assert {
        "upper_bound",
        "fitted_lower_bound",
        "fitted_upper_bound",
        "fitted_psi",
        "delta_from_fitted",
    } <= set(before)
    assert not before["is_fitted_bound"].any()
    restored = load(point_result.save(tmp_path / "cached-frame.joblib"))
    after = restored.diagnostics.truncation_curve(bounds=[0.02, 0.05])
    assert isinstance(after, pd.DataFrame)
    pd.testing.assert_frame_equal(after, before, check_exact=True)


def test_a_combined_report_retains_and_replays_a_returned_frame(point_result, tmp_path) -> None:  # type: ignore[no-untyped-def]
    combined = point_result.diagnostics.run_all(include_retargets=True)
    before = combined.report("truncation_curve")
    detail = combined["truncation_curve"].detail
    assert isinstance(before, pd.DataFrame)
    assert before["is_fitted_bound"].sum() == 1
    pd.testing.assert_frame_equal(combined.reports()["truncation_curve"], before, check_exact=True)

    restored = load(point_result.save(tmp_path / "combined-frame.joblib"))
    replayed = restored.diagnostics.run_all(include_retargets=True)
    after = replayed.report("truncation_curve")
    assert isinstance(after, pd.DataFrame)
    pd.testing.assert_frame_equal(after, before, check_exact=True)
    assert replayed["truncation_curve"].detail == detail


def test_replayability_names_the_refit_boundary(point_result, longitudinal_result) -> None:  # type: ignore[no-untyped-def]
    assert point_result.replayability.summarize_existing_artifacts
    assert point_result.replayability.retarget_cached_nuisances
    assert point_result.replayability.refit_nuisances
    assert longitudinal_result.replayability.summarize_existing_artifacts
    assert not longitudinal_result.replayability.retarget_cached_nuisances
    assert not longitudinal_result.replayability.evaluate_new_data


class TestTheCombinedSensitivityReportRunsToCompletion:
    """``run_all`` invokes every operation argument-free, which not all of them accept."""

    def test_including_refits_does_not_raise(self, point_result) -> None:  # type: ignore[no-untyped-def]
        report = point_result.sensitivity.run_all(include_refits=True, include_retargets=True)
        assert {item.name for item in report.items} == set(SENSITIVITY_ROUTES)

    @pytest.mark.parametrize("facade", ["diagnostics", "sensitivity"])
    def test_the_two_cost_classes_are_skipped_by_their_own_flag(  # type: ignore[no-untyped-def]
        self, point_result, facade: str
    ) -> None:
        """Both facades read the same declaration the same way.

        They did not: one skipped everything non-``summarize`` and the other only
        ``refit``, so ``tipping_gamma`` -- a root search over full missingness retargets --
        ran in a bare ``sensitivity.run_all()`` while ``truncation_curve`` did not.

        A row that also needs a caller argument is judged by
        ``test_a_cost_flag_is_never_named_before_a_required_argument`` instead. No flag
        makes such a row run, so naming its cost would be a false instruction.
        """
        surface = getattr(point_result, facade)
        rows = {row.operation: row for row in surface.capabilities if row.available}
        assert {"refit", "retarget"} & {row.execution for row in rows.values()}

        report = surface.run_all()
        for operation, row in rows.items():
            if row.execution == "summarize":
                continue
            assert report[operation].status is AssessmentStatus.DEFERRED
            if row.requires_arguments:
                assert "has no basis to choose" in report[operation].detail
                continue
            flag = "include_refits" if row.execution == "refit" else "include_retargets"
            assert f"pass {flag}=True" in report[operation].detail

        for row in rows.values():
            if row.execution == "retarget" and not row.requires_arguments:
                assert report[row.operation].detail != (
                    surface.run_all(include_retargets=True)[row.operation].detail
                )

    def test_benchmark_says_it_needs_covariates_rather_than_crashing(self, point_result) -> None:  # type: ignore[no-untyped-def]
        """A mapping that supplies something else still names the argument that is missing.

        ``test_a_deferred_row_retains_the_request_it_did_not_run`` pins what the row then
        carries. This one pins the sentence, on a request that is partly filled in.
        """
        report = point_result.sensitivity.run_all(
            include_refits=True,
            arguments={"benchmark": {"random_state": 91}},
        )
        item = report["benchmark"]
        assert item.status is AssessmentStatus.DEFERRED
        assert "covariates" in item.detail
        assert any("benchmark" in step for step in item.next_steps)

    def test_the_requirement_is_declared_on_the_capability_row(self, point_result) -> None:  # type: ignore[no-untyped-def]
        """``run_all`` must learn this from the row, not from the operation's name."""
        rows = {row.operation: row for row in point_result.sensitivity.capabilities}
        assert rows["benchmark"].requires_arguments == ("covariates",)
        assert rows["simulated_confounding"].requires_arguments == ("grid",)
        assert rows["omitted_confounding"].requires_arguments == ()

    def test_simulated_surface_is_never_launched_implicitly(self, point_result) -> None:  # type: ignore[no-untyped-def]
        report = point_result.sensitivity.run_all(include_refits=True)
        item = report["simulated_confounding"]
        assert item.status is AssessmentStatus.DEFERRED
        assert "grid" in item.detail
        assert any("simulated_confounding" in step for step in item.next_steps)

    def test_required_arguments_and_the_common_seed_run_and_are_retained(
        self, point_result
    ) -> None:  # type: ignore[no-untyped-def]
        report = point_result.sensitivity.run_all(
            include_refits=True,
            arguments={"benchmark": {"covariates": ("W1",)}},
            random_state=91,
        )
        item = report["benchmark"]
        assert item.status is AssessmentStatus.COMPLETED
        assert item.arguments["covariates"] == ("W1",)
        assert item.arguments["random_state"] == 91
        assert report.report("benchmark").random_state == 91
        first_keys = set(point_result.assessment_cache)

        second = point_result.sensitivity.run_all(
            include_refits=True,
            arguments={"benchmark": {"covariates": ("W1",)}},
            random_state=92,
        )
        assert second.report("benchmark").random_state == 92
        assert set(point_result.assessment_cache) - first_keys

    def test_seed_conflicts_and_unknown_operations_fail_before_any_operation_runs(
        self, point_result
    ) -> None:  # type: ignore[no-untyped-def]
        before = set(point_result.assessment_cache)
        with pytest.raises(ValueError, match="supplied both"):
            point_result.sensitivity.run_all(
                arguments={"benchmark": {"covariates": ("W1",), "random_state": 2}},
                random_state=1,
            )
        with pytest.raises(KeyError, match="not_an_operation"):
            point_result.assess(arguments={"not_an_operation": {}})
        assert set(point_result.assessment_cache) == before


def test_assess_is_on_the_public_protocol_and_presents_each_owned_row_once(
    point_result, longitudinal_result
) -> None:  # type: ignore[no-untyped-def]
    for result in (point_result, longitudinal_result):
        assert isinstance(result, CausalResult)
        battery = result.assess()
        assert isinstance(battery, AssessmentReport)
        frame = battery.to_frame()
        for owned in ("score_equations", "support", "nuisance_models"):
            assert list(frame["check"]).count(owned) == 1
        assert set(frame["surface"]) == {"validation", "diagnostics", "sensitivity"}


def _runtime_protocol_members(protocol: type) -> frozenset[str]:
    """The names ``isinstance`` presence-checks, on every interpreter this package runs on.

    ``typing.get_protocol_members`` arrived in 3.13 and ``__protocol_attrs__`` in 3.12, so
    read whichever this interpreter has. The set itself is the same on 3.11: it is what
    ``typing._get_protocol_attrs`` returns, and what ``_ProtocolMeta.__instancecheck__``
    loops over there.
    """
    members = getattr(typing, "get_protocol_members", None)
    if members is not None:  # pragma: no branch - one branch per interpreter
        return frozenset(members(protocol))
    return frozenset(  # pragma: no cover - taken on Python 3.11 only
        getattr(protocol, "__protocol_attrs__", None) or typing._get_protocol_attrs(protocol)  # type: ignore[attr-defined]
    )


@pytest.mark.parametrize("fixture_name", ["point_result", "longitudinal_result"])
def test_no_runtime_protocol_member_raises_when_isinstance_reads_it(request, fixture_name) -> None:  # type: ignore[no-untyped-def]
    """``isinstance(result, CausalResult)`` must answer rather than raise.

    Python 3.11 checks a runtime protocol by calling ``hasattr`` for every member, which
    *invokes* a property and swallows only ``AttributeError``. ``CausalResult.estimate``
    was such a member, and ``LongitudinalResult.estimate`` raises ``ValueError`` on a
    multi-parameter fit, so ``isinstance`` raised there on 3.11 while 3.12 and 3.13
    passed. Python 3.12 reads the members with ``inspect.getattr_static`` instead, so this
    test asserts the 3.11 predicate directly rather than calling ``isinstance``.

    The witness for the mechanism, not only for its symptom: the longitudinal fixture
    reports three parameters, so its ``estimate`` still raises. A member added to the
    protocol whose read raises anything at all fails the loop below on every interpreter.
    """
    result = request.getfixturevalue(fixture_name)
    members = _runtime_protocol_members(CausalResult)
    assert members
    assert "estimate" not in members

    # 3.11's ``_ProtocolMeta.__instancecheck__`` body, run here on 3.13.
    assert all(
        hasattr(result, attr)
        and (not callable(getattr(CausalResult, attr, None)) or getattr(result, attr) is not None)
        for attr in members
    )
    assert isinstance(result, CausalResult)


def test_the_multi_parameter_estimate_that_broke_the_protocol_check_still_refuses(
    longitudinal_result,
) -> None:  # type: ignore[no-untyped-def]
    """The nonzero witness for the test above.

    Dropping ``estimate`` from the runtime membership must not weaken the refusal it
    exists for. If this fit ever reported one parameter the loop above would pass for a
    reason that has nothing to do with the fix.
    """
    assert len(longitudinal_result.estimates) > 1
    with pytest.raises(ValueError, match="multiple parameters"):
        _ = longitudinal_result.estimate
    assert "estimate" in CausalResult.__doc__


def test_assess_refuses_an_argument_for_an_operation_the_validation_battery_owns(
    point_result,
) -> None:  # type: ignore[no-untyped-def]
    """A caller's own question must not be answered and then discarded.

    ``assess`` presents the validation row for these three names and hides the diagnostics
    row of the same name. Forwarding ``arguments`` to the diagnostics side therefore ran
    the caller's tolerance, recorded a ``failed`` row, and then showed the argument-free
    ``passed`` row. ``attention`` never named the failure.
    """
    with pytest.raises(CapabilityError, match="score_equations"):
        point_result.assess(arguments={"score_equations": {"tolerance": 1e-30}})

    # The nonzero witness: the refused tolerance really does change the verdict, so the
    # discarded row was a ``failed`` one rather than a second copy of the same answer.
    strict = point_result.diagnostics.score_equations(tolerance=1e-30)
    assert not strict.passed
    assert point_result.diagnostics.score_equations().passed

    # The refusal names a call that answers the question, and that call works.
    direct = point_result.diagnostics.run_all(arguments={"score_equations": {"tolerance": 1e-30}})
    assert direct["score_equations"].status is AssessmentStatus.FAILED

    # Every other operation still accepts its arguments through ``assess``.
    battery = point_result.assess(arguments={"omitted_confounding": {"cf_y": 0.03}})
    assert battery.report("omitted_confounding").cf_y == pytest.approx(0.03)


def test_a_cost_flag_is_never_named_before_a_required_argument(point_result) -> None:  # type: ignore[no-untyped-def]
    """A refusal names the first thing that is wrong, not the first gate in the code.

    The cost gate ran before the required-argument gate, so a combined report told the
    caller to pass ``include_refits=True`` for ``benchmark``. The caller passed it, and the
    next report said ``benchmark`` needs an explicit ``covariates`` argument that no flag
    supplies. The instruction was false when it was given.

    Both facades are read together because only the sensitivity side declares a required
    argument today. Reading them one at a time would skip the diagnostics half, and a
    skipped check reads exactly like a passing one.
    """
    surfaces = [point_result.diagnostics, point_result.sensitivity]
    demanding = [
        (surface, row)
        for surface in surfaces
        for row in surface.capabilities
        if row.available and row.requires_arguments
    ]
    assert demanding, "no available row declares a required argument"

    for flags in ({}, {"include_refits": True, "include_retargets": True}):
        for surface, row in demanding:
            item = surface.run_all(**flags)[row.operation]
            assert item.status is AssessmentStatus.DEFERRED
            assert row.requires_arguments[0] in item.detail
            assert "include_refits" not in item.detail
            assert "include_retargets" not in item.detail

    # The paired witness: a row of the same cost and no required argument still names its
    # flag, so the assertions above are about the argument gate rather than about cost.
    priced = [
        (surface, row)
        for surface in surfaces
        for row in surface.capabilities
        if row.available and not row.requires_arguments and row.execution != "summarize"
    ]
    assert priced
    for surface, row in priced:
        flag = "include_refits" if row.execution == "refit" else "include_retargets"
        assert f"pass {flag}=True" in surface.run_all()[row.operation].detail


#: Every gate that defers a row, on both surfaces, with and without a seed. The three
#: gates are the required argument, the cost flag, and the ambiguous estimand, and each
#: appears once with a seed and once without. An injection that reached every row, or no
#: row, fails on one of the pairs. ``refute`` under no flags is the case the seed defect
#: was first reported on: a diagnostics cost deferral that accepts a seed.
_DEFERRED_REQUESTS = [
    pytest.param(
        "point_result",
        "sensitivity",
        "benchmark",
        {"include_refits": True},
        {},
        {"random_state": 91},
        id="argument-seed",
    ),
    pytest.param(
        "point_result",
        "diagnostics",
        "truncation_curve",
        {},
        {"bounds": (0.02, 0.05)},
        {},
        id="cost-no-seed",
    ),
    pytest.param(
        "point_result", "diagnostics", "refute", {}, {}, {"random_state": 91}, id="cost-seed"
    ),
    pytest.param(
        "multi_arm_result",
        "sensitivity",
        "omitted_confounding",
        _PAID,
        {},
        {},
        id="estimand-no-seed",
    ),
    pytest.param(
        "multi_arm_result",
        "diagnostics",
        "refute",
        _PAID,
        {},
        {"random_state": 91},
        id="estimand-seed",
    ),
]


@pytest.mark.parametrize("fixture_name,facade,operation,flags,supplied,seed", _DEFERRED_REQUESTS)
def test_a_deferred_row_retains_the_request_it_did_not_run(  # type: ignore[no-untyped-def]
    request, fixture_name: str, facade: str, operation: str, flags: dict, supplied: dict, seed: dict
) -> None:
    """One property, three gates, both surfaces: the row carries what a replay needs.

    The seed is the half that was structurally absent. ``run_all`` injected
    ``random_state`` after the gates rather than before them, so a deferred row reported
    the raw supplied mapping and no seed. A caller replaying that row from
    ``item.arguments`` then drew a different sample from the one the report would have
    taken, which is exactly what a non-deterministic operation must not do.
    """
    result = request.getfixturevalue(fixture_name)
    surface = getattr(result, facade)
    assert surface.capability(operation).accepts_random_state == bool(seed)

    item = surface.run_all(**flags, arguments={operation: dict(supplied)}, random_state=91)[
        operation
    ]

    assert item.status is AssessmentStatus.DEFERRED
    assert item.next_steps
    assert dict(item.arguments) == {**supplied, **seed}
    result.assessment_cache.clear()


def test_paying_the_cost_lifts_the_deferral_and_keeps_the_request(point_result) -> None:  # type: ignore[no-untyped-def]
    """The flag runs the same request the deferred row named."""
    arguments = {"truncation_curve": {"bounds": (0.02, 0.05)}}

    deferred = point_result.diagnostics.run_all(arguments=arguments)["truncation_curve"]
    completed = point_result.diagnostics.run_all(include_retargets=True, arguments=arguments)[
        "truncation_curve"
    ]

    assert deferred.status is AssessmentStatus.DEFERRED
    assert completed.status is AssessmentStatus.COMPLETED
    assert completed.arguments["bounds"] == arguments["truncation_curve"]["bounds"]


@pytest.mark.parametrize("facade", ["diagnostics", "sensitivity"])
def test_capabilities_never_claims_a_row_that_replayability_forbids(point_result, facade) -> None:  # type: ignore[no-untyped-def]
    """Availability is authoritative before execution, for every row and not four of them.

    ``truncation_curve``, ``benchmark`` and ``simulated_confounding`` each read their own
    replay slot in their own facade. ``refute`` read the same slot only when it ran, so a
    restored result reported ``refute`` available and then raised. This check derives the
    requirement from the row rather than naming the operations.
    """
    detached = dataclasses.replace(point_result, estimator=None)
    replay = detached.replayability
    assert not replay.refit_nuisances and not replay.retarget_cached_nuisances

    surface = getattr(detached, facade)
    gated = {row.operation for row in surface.capabilities if row.requires_replay}
    assert gated, f"the {facade} facade declares no replay-gated row"

    for row in surface.capabilities:
        if row.requires_replay is None:
            continue
        assert not row.available
        assert row.status is AssessmentStatus.UNAVAILABLE
        assert "estimator configuration" in (row.reason or "")
        # Two wordings are correct here, and the contract is the same under both. A row
        # the replay gate rewrites says the result "no longer carries" its estimator. A
        # row that its own declaration already refused keeps that more specific reason,
        # which for ``simulated_confounding`` says the result has "no estimator
        # configuration". The gate short-circuits rather than overwriting the sharper
        # text. Pinning one wording would forbid the sharper one.
        wordings = r"no longer carries|no estimator configuration"
        with pytest.raises(CapabilityError, match=wordings):
            surface._require(row.operation)

    # The nonzero control: the same rows are available while the estimator is present, so
    # the gate is reading replayability rather than refusing everything.
    intact = {row.operation for row in getattr(point_result, facade).capabilities if row.available}
    assert gated <= intact


def _run_argument_free_routes(result):  # type: ignore[no-untyped-def]
    ran = set()
    for surface in (result.diagnostics, result.sensitivity):
        for declared in surface.capabilities:
            row = surface.capability(declared.operation)
            if row.requires_arguments or not row.available:
                continue
            function, _ = surface._routed_callable(row.operation)
            assert inspect.signature(function)
            # Bind the underlying implementation, not the facade's **kwargs wrapper.
            surface._bind_arguments(row.operation, {}, partial=False)
            getattr(surface, row.operation)()
            ran.add(row.operation)
    return ran


@pytest.mark.parametrize("fixture_name", ["point_result", "longitudinal_result"])
def test_every_argument_free_row_really_binds_and_runs(request, fixture_name) -> None:  # type: ignore[no-untyped-def]
    assert _run_argument_free_routes(request.getfixturevalue(fixture_name))


@pytest.mark.parametrize("facade", ["diagnostics", "sensitivity"])
@pytest.mark.parametrize("error", [KeyError("arm 'high'"), TypeError("unexpected 'subset'")])
def test_a_structural_error_is_raised_rather_than_reported_as_unavailable(  # type: ignore[no-untyped-def]
    point_result, monkeypatch: pytest.MonkeyPatch, facade: str, error: Exception
) -> None:
    """A structural failure must not look like a scientific refusal."""
    surface = getattr(point_result, facade)
    operation = next(
        row.operation
        for row in surface.capabilities
        if row.available and not row.requires_arguments and row.execution == "summarize"
    )

    def broken(*_args: object, **_kwargs: object) -> object:
        raise error

    monkeypatch.setattr(type(surface), operation, broken, raising=True)
    point_result.assessment_cache.clear()
    with pytest.raises(type(error)):
        surface.run_all()
    point_result.assessment_cache.clear()


@pytest.mark.parametrize("facade", ["diagnostics", "sensitivity"])
def test_a_refusal_is_unavailable_and_later_diagnostics_still_run(  # type: ignore[no-untyped-def]
    point_result, monkeypatch: pytest.MonkeyPatch, facade: str
) -> None:
    """One unsupported diagnostic must not discard accepted diagnostic reports."""
    surface = getattr(point_result, facade)
    runnable = [
        row.operation
        for row in surface.capabilities
        if row.available and not row.requires_arguments and row.execution == "summarize"
    ]
    refused, accepted = runnable[:2]

    def decline(*_args: object, **_kwargs: object) -> object:
        raise CapabilityError("the fitted artifacts do not support this requested variant")

    monkeypatch.setattr(type(surface), refused, decline, raising=True)
    point_result.assessment_cache.clear()
    report = surface.run_all()

    assert report[refused].status is AssessmentStatus.UNAVAILABLE
    assert "declined this request" in report[refused].detail
    assert report[accepted].status not in {
        AssessmentStatus.NOT_APPLICABLE,
        AssessmentStatus.UNAVAILABLE,
    }
    assert report.report(accepted) is not None
    point_result.assessment_cache.clear()


def test_a_real_sensitivity_refusal_does_not_prevent_a_later_evalue(point_result, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Median-repeat refusals stay informative while independent analyses continue."""
    repeated = dataclasses.replace(
        point_result,
        repeats=point_result.repeats * 2,
    )

    requested = {"omitted_confounding": {"cf_y": 0.23, "cf_d": 0.17}}
    report = repeated.sensitivity.run_all(arguments=requested)

    assert report["omitted_confounding"].status is AssessmentStatus.UNAVAILABLE
    assert "median-combined repeats" in report["omitted_confounding"].detail
    assert report["evalue"].status is AssessmentStatus.COMPLETED
    assert report.report("evalue").estimand == "ate"
    arguments = report["omitted_confounding"].arguments
    assert arguments["cf_y"] == 0.23
    assert arguments["cf_d"] == 0.17
    assert arguments["rho"] == 1.0
    restored = load(repeated.save(tmp_path / "refused-arguments.joblib"))
    replayed = restored.sensitivity.run_all(arguments=requested)
    assert replayed["omitted_confounding"].arguments == arguments
    assert replayed == report


def test_a_refusal_before_seed_resolution_keeps_the_seed_unspecified(
    point_result, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    module = importlib.import_module("cleverly.validation.refute")

    def unexpected_seed(*args, **kwargs):  # type: ignore[no-untyped-def]
        pytest.fail("the refused request must not resolve a stochastic seed")

    monkeypatch.setattr(module, "resolve_assessment_seed", unexpected_seed)
    result = dataclasses.replace(point_result)
    report = result.diagnostics.run_all(
        include_refits=True,
        arguments={"refute": {"tests": ("bootstrap_measurement_error",)}},
    )
    assert report["refute"].status is AssessmentStatus.UNAVAILABLE
    assert "BootstrapMeasurementError declaration" in report["refute"].detail
    assert report["refute"].arguments["random_state"] is None
    assert report["refute"].arguments["tests"] == ("bootstrap_measurement_error",)


class TestSupportDiagnosticsSeeAPerInterventionReport:
    """A shift or IPSI fit reports a mapping, which the attribute probes cannot read.

    ``support()`` returns one record per declared intervention on these axes rather than
    a single report object, so ``hasattr(report, "truncated")``,
    ``isinstance(report, LongitudinalDiagnostics)`` and ``getattr(report, "regimes")`` all
    miss -- and the battery reported ``passed`` for every shift fit ever made, including
    one that had already warned about extrapolating past the observed dose.
    """

    @pytest.fixture(scope="class")
    def extrapolating_shift(self):  # type: ignore[no-untyped-def]
        from cleverly import ModifiedTreatmentPolicy
        from cleverly.datasets import make_shift_dose
        from cleverly.interventions import Shift

        frame, _ = make_shift_dose(n=300, seed=0)
        study = CausalStudy(
            frame,
            design=PointTreatment(
                outcome="Y",
                treatment="A",
                adjustment=["W1", "W2"],
                treatment_kind="continuous",
            ),
        )
        with pytest.warns(PositivityWarning, match="above the largest one observed"):
            return study.identify(ModifiedTreatmentPolicy(shifts=[Shift(3.0, cap=None)])).estimate(
                outcome_learner=sklearn.linear_model.LinearRegression(),
                treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
                n_folds=3,
                learner_folds=2,
                random_state=0,
                simultaneous=False,
            )

    def test_the_stored_report_really_does_leave_almost_no_effective_sample(
        self, extrapolating_shift
    ) -> None:  # type: ignore[no-untyped-def]
        """The witness: without this the status below would be vacuously right."""
        support = extrapolating_shift.diagnostics.support()
        assert min(item.ess_ratio for item in support.values()) < 0.2

    def test_validate_reports_the_ratio_rather_than_passing(self, extrapolating_shift) -> None:  # type: ignore[no-untyped-def]
        """No pass on a fit this thin, and no invented threshold either.

        This shift retains under 1% of its effective sample and truncates nothing. The
        row therefore grades nothing, and says so: ``COMPLETED`` carries the ratio and
        leaves the judgement to the reader. ``PASSED`` would read as a positivity
        clearance that no threshold in this package is entitled to give.
        """
        item = extrapolating_shift.validate()["support"]
        assert item.status is AssessmentStatus.COMPLETED
        assert item.status is not AssessmentStatus.PASSED
        assert "effective-sample-size ratio" in item.detail

    def test_the_combined_report_agrees_with_validate(self, extrapolating_shift) -> None:  # type: ignore[no-untyped-def]
        item = extrapolating_shift.diagnostics.run_all()["support"]
        assert item.status is AssessmentStatus.COMPLETED

    def test_a_well_supported_tilt_does_not_warn(self) -> None:
        """The control: the new branch must not warn about every mapping it sees."""
        from cleverly import IncrementalMean
        from cleverly.datasets import make_linear_ate
        from cleverly.interventions import Incremental

        frame, _ = make_linear_ate(n=300, seed=11)
        result = (
            CausalStudy(
                frame,
                design=PointTreatment(
                    outcome="Y", treatment="A", adjustment=["W1", "W2", "W3", "W4"]
                ),
            )
            .identify(IncrementalMean(interventions=[Incremental(2.0)]))
            .estimate(
                outcome_learner=sklearn.linear_model.LinearRegression(),
                treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
                n_folds=3,
                learner_folds=2,
                random_state=4,
                simultaneous=False,
            )
        )
        support = result.diagnostics.support()
        assert min(item.ess_ratio for item in support.values()) >= 0.2
        assert result.validate()["support"].status is AssessmentStatus.COMPLETED


class TestCapabilityRowsDoNotContradictThemselves:
    """``available`` and ``status`` are two statements about one cell; they must agree.

    ``_require`` gates on ``available`` while the report surfaces ``status`` and
    ``reason``, so a row that disagrees with itself leaks into user-facing text: a point
    fit with a fitted missingness mechanism published ``available: True | status:
    unavailable | reason: no longitudinal missingness-tilt adapter is implemented`` for an
    operation that works.

    The fixtures are the witnesses. None of the first three reaches an ambiguous E-value
    default, so the row that published ``available: True | status: passed | reason: an
    E-value needs one contrast`` broke the second invariant below while this class passed.
    ``multi_arm_result`` reports two contrasts, which is the case the E-value row turns on.
    """

    @pytest.fixture(scope="class")
    def missing_outcome_result(self):  # type: ignore[no-untyped-def]
        from cleverly.datasets import make_missing_outcome

        frame, _ = make_missing_outcome(n=400, seed=61)
        return (
            CausalStudy(
                frame,
                design=PointTreatment(
                    outcome="Y",
                    treatment="A",
                    adjustment=["W1", "W2", "W3"],
                    missingness="Delta",
                ),
            )
            .identify(ATE())
            .estimate(
                outcome_learner=sklearn.linear_model.LinearRegression(),
                treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
                n_folds=3,
                learner_folds=2,
                random_state=4,
                simultaneous=False,
            )
        )

    @pytest.mark.parametrize(
        "fixture_name",
        ["point_result", "longitudinal_result", "missing_outcome_result", "multi_arm_result"],
    )
    def test_an_available_operation_is_never_reported_unavailable(
        self, fixture_name, request
    ) -> None:  # type: ignore[no-untyped-def]
        result = request.getfixturevalue(fixture_name)
        for row in result.sensitivity.capabilities:
            if not row.available:
                continue
            assert row.status not in {
                AssessmentStatus.UNAVAILABLE,
                AssessmentStatus.NOT_APPLICABLE,
            }, f"{row.operation} is available but reports {row.status}"

    @pytest.mark.parametrize(
        "fixture_name",
        ["point_result", "longitudinal_result", "missing_outcome_result", "multi_arm_result"],
    )
    def test_only_an_unrunnable_operation_carries_a_reason(self, fixture_name, request) -> None:  # type: ignore[no-untyped-def]
        """Both facades, because the invariant is about the field and not about one surface.

        This checked ``sensitivity`` alone, and ``truncation_curve`` lives on
        ``diagnostics``. A per-result cost override that stamped a note onto an available
        diagnostics row therefore breached the documented meaning of ``reason`` without
        failing anything.
        """
        result = request.getfixturevalue(fixture_name)
        for surface in (result.diagnostics, result.sensitivity):
            for row in surface.capabilities:
                if row.available:
                    assert row.reason is None, (
                        f"{row.operation} is available but explains itself away"
                    )
                else:
                    assert row.reason, f"{row.operation} is unavailable and does not say why"

    def test_a_fitted_missingness_mechanism_makes_the_tilt_available(
        self, missing_outcome_result
    ) -> None:  # type: ignore[no-untyped-def]
        rows = {row.operation: row for row in missing_outcome_result.sensitivity.capabilities}
        assert rows["missingness"].available
        assert rows["missingness"].status is AssessmentStatus.PASSED
        assert rows["missingness"].reason is None
        # And the operation really does run, which is what made the old row wrong.
        assert missing_outcome_result.sensitivity.missingness() is not None

    def test_missing_outcome_argument_free_routes_bind_and_run(
        self, missing_outcome_result
    ) -> None:  # type: ignore[no-untyped-def]
        assert {"missingness", "tipping_gamma"} <= _run_argument_free_routes(missing_outcome_result)

    def test_missing_estimator_disables_retargeting_but_not_stored_missingness(
        self, missing_outcome_result
    ) -> None:  # type: ignore[no-untyped-def]
        restored = dataclasses.replace(
            missing_outcome_result,
            estimator=None,
        )

        assert not restored.replayability.retarget_cached_nuisances
        assert not restored.diagnostics.capability("truncation_curve").available
        assert restored.sensitivity.capability("missingness").available
        assert restored.sensitivity.missingness() is not None

    def test_the_longitudinal_reason_stays_on_the_longitudinal_row(
        self, longitudinal_result
    ) -> None:  # type: ignore[no-untyped-def]
        rows = {row.operation: row for row in longitudinal_result.sensitivity.capabilities}
        assert "longitudinal" in rows["missingness"].reason


class TestAttributeAccessAnswersExistenceNotAvailability:
    """``__getattr__`` conflated "no such operation" with "not on this fit".

    Both raised ``CapabilityError``, which subclasses ``ValueError``. ``hasattr`` only
    swallows ``AttributeError``, so on a longitudinal result -- where the legacy analysis
    object is absent entirely -- probing *any* name raised, and a typo was answered with
    the sequential-recursion rationale as though it named a real analysis.
    """

    def test_hasattr_reports_a_real_operation_rather_than_raising(
        self, longitudinal_result
    ) -> None:  # type: ignore[no-untyped-def]
        assert hasattr(longitudinal_result.sensitivity, "evalue")

    def test_getattr_with_a_default_does_not_raise(self, longitudinal_result) -> None:  # type: ignore[no-untyped-def]
        assert getattr(longitudinal_result.sensitivity, "evalue", None) is not None
        assert (
            getattr(longitudinal_result.sensitivity, "no_such_analysis", "fallback") == "fallback"
        )

    def test_a_typo_is_an_attribute_error_naming_what_does_exist(self, longitudinal_result) -> None:  # type: ignore[no-untyped-def]
        typo = "evalu"
        with pytest.raises(AttributeError, match="has no attribute 'evalu'"):
            getattr(longitudinal_result.sensitivity, typo)
        assert not hasattr(longitudinal_result.sensitivity, "evalu")

    def test_a_real_operation_still_refuses_by_name_when_called(self, longitudinal_result) -> None:  # type: ignore[no-untyped-def]
        """Existence is not availability: the refusal moves to the call, it does not go."""
        with pytest.raises(CapabilityError, match="no longitudinal sensitivity derivation"):
            longitudinal_result.sensitivity.evalue()

    def test_the_point_facade_still_delegates_and_caches(self, point_result) -> None:  # type: ignore[no-untyped-def]
        """The control: a fit that *can* serve these must be unaffected."""
        curve = point_result.diagnostics.truncation_curve(bounds=[0.02, 0.05])
        assert curve is not None
        assert hasattr(point_result.sensitivity, "evalue")


@pytest.mark.parametrize(
    "arguments",
    [
        {"simulated_confounding": {"grid": object()}},
        {"simulated_confounding": {"grid": object(), "estimand": "ate"}},
        {"benchmark": {"covariates": ("W1",)}},
        {"evalue": {"estimand": "ate"}},
    ],
)
def test_unavailable_longitudinal_arguments_never_bind_point_data(longitudinal_result, arguments):
    result = dataclasses.replace(longitudinal_result)
    battery = result.assess(arguments=arguments)
    assert battery.sensitivity[next(iter(arguments))].status is AssessmentStatus.UNAVAILABLE
    with pytest.raises(TypeError):
        result.assess(arguments={"simulated_confounding": {"not_a_keyword": 1}})


@pytest.mark.parametrize("name", ["score_equations", "support", "nuisance_models"])
def test_explicit_surface_retrieves_validation_owned_diagnostics(point_result, name):
    battery = point_result.assess()
    assert battery.report(name, surface="diagnostics") is battery.diagnostics.report(name)
    assert battery.report(name) is battery.report(name, surface="validation")
    assert sum(item.name == name for _, item in battery._presented()) == 1
    with pytest.raises(KeyError, match="surface"):
        battery.report(name, surface="wrong")


def test_completed_none_payload_survives_retrieval_and_pickle(point_result, tmp_path):
    import joblib

    item = AssessmentItem(
        "tipping_gamma", AssessmentStatus.COMPLETED, "no tipping point", _report=None
    )
    omitted = AssessmentItem("missingness", AssessmentStatus.UNAVAILABLE, "missing artifacts")
    surface = DiagnosticReport((item, omitted))
    battery = AssessmentReport(ValidationReport(()), DiagnosticReport(()), surface)
    path = tmp_path / "none-report.joblib"
    joblib.dump(battery, path)
    for report in (battery, joblib.load(path)):
        assert report.report("tipping_gamma") is None
        assert report.sensitivity.report("tipping_gamma") is None
        assert report.sensitivity.reports() == {"tipping_gamma": None}
        with pytest.raises(KeyError):
            report.report("missingness")


@pytest.mark.parametrize(
    "mse, expected", [(float("nan"), AssessmentStatus.WARNING), (0.1, AssessmentStatus.COMPLETED)]
)
def test_longitudinal_loss_warning(mse, expected):
    from cleverly.assessment import (
        INTERPRETERS,
        LongitudinalNuisanceDiagnostics,
        LongitudinalNuisanceRow,
    )

    report = LongitudinalNuisanceDiagnostics(
        (LongitudinalNuisanceRow("always", None, None, 1, 12, mse),)
    )
    assert INTERPRETERS["nuisance_models"](report, None).status is expected


@pytest.mark.parametrize(
    "supplied, phrase",
    [
        ({}, "at the default strengths"),
        ({"cf_y": 0.03}, "at the default cf_d strength"),
        ({"cf_d": 0.03}, "at the default cf_y strength"),
        ({"cf_y": 0.03, "cf_d": 0.03}, None),
    ],
)
def test_default_strength_provenance_uses_supplied_arguments(point_result, supplied, phrase):
    row = point_result.sensitivity.run_all(arguments={"omitted_confounding": supplied})[
        "omitted_confounding"
    ]
    if phrase is None:
        assert "default" not in row.detail
    else:
        assert phrase in row.detail
    assert row.arguments["cf_y"] == row.arguments["cf_d"] == 0.03


def test_refute_direct_aggregate_and_seed_replay_share_one_computation(point_result, monkeypatch):
    result = dataclasses.replace(point_result)
    module = importlib.import_module("cleverly.validation.refute")
    calls = []
    original = module.refute

    def tracked(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    tracked.__signature__ = inspect.signature(original)
    monkeypatch.setattr(module, "refute", tracked)
    arguments = {"n_replicates": 1, "tests": ("placebo",)}
    direct = result.diagnostics.refute(**arguments)
    combined = result.diagnostics.run_all(include_refits=True, arguments={"refute": arguments})
    assert combined.report("refute") is direct
    assert result.diagnostics.refute(**combined["refute"].arguments) is direct
    assert calls == [1]


def test_refusals_stay_out_of_attention_while_support_warnings_remain(point_result):
    repeated = dataclasses.replace(point_result, repeats=point_result.repeats * 2)
    sensitivity = repeated.sensitivity.run_all()
    warning = AssessmentItem("support", AssessmentStatus.WARNING, "positivity warning")
    battery = AssessmentReport(ValidationReport((warning,)), DiagnosticReport(()), sensitivity)
    assert "support" in [item.name for item in battery.attention]
    assert "omitted_confounding" not in [item.name for item in battery.attention]
    assert "omitted_confounding" in [item.name for item in battery.omissions]
    assert "benchmark" in [item.name for item in battery.omissions]
    assert sensitivity["benchmark"].status is AssessmentStatus.DEFERRED


def test_every_status_is_presented_by_exactly_one_grouping() -> None:
    """A new member of the taxonomy lands in one bucket, and never in none of them.

    ``assessment.py`` spelled three status groupings inline and nothing tied them
    together, so an added member fell into zero of them silently: it stayed out of
    ``attention``, out of ``omissions``, and out of the blocking set that decides
    ``ValidationReport.passed``. It would then read as a clean row on every surface.
    ``DEFERRED`` reached two of the three by hand.

    The three presentation buckets partition the enum. The blocking set is a separate
    claim and cuts across them, so it is checked by containment rather than by identity.
    """
    assert set(AssessmentStatus) == _ATTENTION | _OMISSIONS | _SETTLED
    assert _ATTENTION.isdisjoint(_OMISSIONS)
    assert _ATTENTION.isdisjoint(_SETTLED)
    assert _OMISSIONS.isdisjoint(_SETTLED)
    # Nothing a report presents as needing no action may fail validation, and a status
    # that blocks has to be one a reader is shown as actionable or as an omission.
    assert _BLOCKING <= _ATTENTION | _OMISSIONS
    assert _BLOCKING.isdisjoint(_SETTLED)
    # The nonzero witnesses, so the containments above cannot hold by being empty.
    assert AssessmentStatus.WARNING in _ATTENTION - _BLOCKING
    assert AssessmentStatus.NOT_APPLICABLE in _OMISSIONS - _BLOCKING
    assert AssessmentStatus.DEFERRED in _OMISSIONS & _BLOCKING
    assert set(_SUMMARY_CHECK_ORDER) == _ATTENTION | {AssessmentStatus.PASSED}
    assert set(_SUMMARY_OMISSION_ORDER) == _OMISSIONS
    summary_order = (*_SUMMARY_CHECK_ORDER, *_SUMMARY_OMISSION_ORDER)
    assert len(summary_order) == len(set(summary_order))
    assert set(AssessmentStatus) == {AssessmentStatus.COMPLETED, *summary_order}


@pytest.mark.parametrize("fixture_name", ["point_result", "longitudinal_result"])
def test_every_cached_item_bearing_report_declares_a_cache_generation(  # type: ignore[no-untyped-def]
    request, fixture_name: str
) -> None:
    """A new aggregate cannot enter the cache without a row in the generation table.

    A cached report that carries assessment items carries an interpretation, and an
    interpretation changes. Without a generation the key never moves, so a result saved
    before the change replays the old verdict for ever. The table is read from the cache a
    real fit writes rather than from a list, so a fourth aggregate is covered on the day it
    is added.

    One family cannot discover an aggregate another family alone writes. A point fit and a
    longitudinal fit are read here because the two declare different capabilities and reach
    different report builders, so an aggregate that only a sequential fit produces is
    covered too.
    """
    result = dataclasses.replace(request.getfixturevalue(fixture_name))
    result.validate()
    result.diagnostics.run_all()
    result.sensitivity.run_all()
    result.assess()

    bearing = {
        key.split(":", 1)[0]
        for key, value in result.assessment_cache.items()
        if isinstance(getattr(value, "items", None), tuple)
        and all(isinstance(item, AssessmentItem) for item in value.items)
        and value.items
    }
    assert bearing == {"diagnostics.run_all", "sensitivity.run_all", "validate"}
    assert bearing <= set(_CACHE_GENERATIONS)


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_frames_pack_once_and_retrieval_cannot_mutate_cached_storage(
    point_result, backend, monkeypatch, tmp_path
):
    import joblib

    from cleverly._assessment_cache import _CachedFrame

    result = dataclasses.replace(
        point_result,
        data=dataclasses.replace(point_result.data, backend=backend),
    )
    calls = []
    original = _CachedFrame.from_frame.__func__

    def tracked(cls, frame, backend):
        calls.append(1)
        return original(cls, frame, backend)

    monkeypatch.setattr(_CachedFrame, "from_frame", classmethod(tracked))
    report = result.sensitivity.run_all()
    assert calls == [1]
    retained = report.report("contour")
    expected = (
        retained.to_dict(as_series=False)
        if backend == "polars"
        else retained.to_dict(orient="list")
    )
    if backend == "pandas":
        retained.iloc[0, 0] = 99
    else:
        retained[0, 0] = 99
    fresh = report.report("contour")
    actual = fresh.to_dict(as_series=False) if backend == "polars" else fresh.to_dict(orient="list")
    assert actual == expected
    path = tmp_path / f"{backend}.joblib"
    joblib.dump(report, path)
    assert type(joblib.load(path).report("contour")) is type(fresh)
    assert calls == [1]
