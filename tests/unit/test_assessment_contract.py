"""The shared post-fit assessment contract across scalar result families."""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import re

import pandas as pd
import pytest
import sklearn.linear_model

from cleverly import (
    ATE,
    AssessmentReport,
    AssessmentStatus,
    CapabilityError,
    CausalResult,
    CausalStudy,
    LongitudinalTreatment,
    PointTreatment,
    PositivityWarning,
    RegimeMean,
    load,
)
from cleverly.assessment import ASSESSMENT_CAPABILITIES, SENSITIVITY_ROUTES
from cleverly.datasets import make_linear_ate, make_longitudinal
from cleverly.sensitivity import ConfounderStrengthGrid
from cleverly.sensitivity._parameters import arm_parameters
from cleverly.sensitivity.positivity import positivity_report
from cleverly.validation.nuisance import nuisance_diagnostics


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


def test_longitudinal_score_and_nuisance_adapters_cover_every_node(longitudinal_result) -> None:  # type: ignore[no-untyped-def]
    expected = sum(len(fit.steps) for fit in longitudinal_result.fits.values())
    scores = longitudinal_result.diagnostics.score_equations()
    nuisances = longitudinal_result.diagnostics.nuisance_models()
    assert len(nuisances.rows) == expected
    # One row per node per question the node poses.  A cross-fitted node poses two -- did
    # every fold's solve reach its root, and is the stitched residual where sampling would
    # leave it -- and a single-fold node poses only the first.
    kinds = [row.kind for row in scores.rows]
    assert kinds.count("solver") == expected
    assert kinds.count("stitching") in {0, expected}
    assert len(scores.rows) == len(kinds)
    assert all(row.score >= 0 and row.relative_score >= 0 for row in scores.rows)
    assert all(row.n > 0 and row.mse >= 0 for row in nuisances.rows)


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


def test_combined_reports_distinguish_inapplicable_from_unavailable(point_result) -> None:  # type: ignore[no-untyped-def]
    report = point_result.sensitivity.run_all()
    assert report["missingness"].status is AssessmentStatus.NOT_APPLICABLE
    assert report["benchmark"].status is AssessmentStatus.UNAVAILABLE


def test_longitudinal_sensitivity_is_a_capability_aware_facade(longitudinal_result) -> None:  # type: ignore[no-untyped-def]
    report = longitudinal_result.sensitivity.run_all()
    assert {item.status for item in report.items} == {AssessmentStatus.UNAVAILABLE}
    with pytest.raises(CapabilityError, match="no longitudinal sensitivity derivation"):
        longitudinal_result.sensitivity.omitted_confounding()


def test_longitudinal_simulated_confounding_refuses_the_missing_scientific_law(  # type: ignore[no-untyped-def]
    longitudinal_result,
) -> None:
    capability = longitudinal_result.sensitivity.capability("simulated_confounding")
    assert capability.reason == (
        "no longitudinal simulated-confounder perturbation law is implemented"
    )
    grid = ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,))
    with pytest.raises(CapabilityError, match=re.escape(capability.reason)):
        longitudinal_result.sensitivity.simulated_confounding(grid=grid)


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


def test_a_cached_frame_replays_in_the_callers_backend(point_result, tmp_path) -> None:  # type: ignore[no-untyped-def]
    before = point_result.diagnostics.truncation_curve(bounds=[0.02, 0.05])
    restored = load(point_result.save(tmp_path / "cached-frame.joblib"))
    after = restored.diagnostics.truncation_curve(bounds=[0.02, 0.05])
    assert isinstance(after, pd.DataFrame)
    pd.testing.assert_frame_equal(after, before, check_exact=True)


def test_a_combined_report_retains_and_replays_a_returned_frame(point_result, tmp_path) -> None:  # type: ignore[no-untyped-def]
    combined = point_result.diagnostics.run_all(include_retargets=True)
    before = combined.report("truncation_curve")
    assert isinstance(before, pd.DataFrame)
    pd.testing.assert_frame_equal(combined.reports()["truncation_curve"], before, check_exact=True)

    restored = load(point_result.save(tmp_path / "combined-frame.joblib"))
    after = restored.diagnostics.run_all(include_retargets=True).report("truncation_curve")
    assert isinstance(after, pd.DataFrame)
    pd.testing.assert_frame_equal(after, before, check_exact=True)


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
        """
        surface = getattr(point_result, facade)
        rows = {row.operation: row for row in surface.capabilities if row.available}
        assert {"refit", "retarget"} & {row.execution for row in rows.values()}

        report = surface.run_all()
        for operation, row in rows.items():
            if row.execution == "summarize":
                continue
            flag = "include_refits" if row.execution == "refit" else "include_retargets"
            assert report[operation].status is AssessmentStatus.UNAVAILABLE
            assert f"pass {flag}=True" in report[operation].detail

        for row in rows.values():
            if row.execution == "retarget":
                assert report[row.operation].detail != (
                    surface.run_all(include_retargets=True)[row.operation].detail
                )

    def test_benchmark_says_it_needs_covariates_rather_than_crashing(self, point_result) -> None:  # type: ignore[no-untyped-def]
        report = point_result.sensitivity.run_all(include_refits=True)
        item = report["benchmark"]
        assert item.status is AssessmentStatus.UNAVAILABLE
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
        assert item.status is AssessmentStatus.UNAVAILABLE
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
        assessment_cache={},
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
    result = dataclasses.replace(point_result, assessment_cache={})
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

    def test_the_stored_report_really_does_breach_a_threshold(self, extrapolating_shift) -> None:  # type: ignore[no-untyped-def]
        """The witness: without this the status below would be vacuously right."""
        support = extrapolating_shift.diagnostics.support()
        assert min(item.ess_ratio for item in support.values()) < 0.2

    def test_validate_reports_the_warning_rather_than_passing(self, extrapolating_shift) -> None:  # type: ignore[no-untyped-def]
        item = extrapolating_shift.validate()["support"]
        assert item.status is AssessmentStatus.WARNING
        assert "+3" in item.detail

    def test_the_combined_report_agrees_with_validate(self, extrapolating_shift) -> None:  # type: ignore[no-untyped-def]
        item = extrapolating_shift.diagnostics.run_all()["support"]
        assert item.status is AssessmentStatus.WARNING

    def test_a_well_supported_tilt_still_passes(self) -> None:
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
        assert result.validate()["support"].status is AssessmentStatus.PASSED


class TestCapabilityRowsDoNotContradictThemselves:
    """``available`` and ``status`` are two statements about one cell; they must agree.

    ``_require`` gates on ``available`` while the report surfaces ``status`` and
    ``reason``, so a row that disagrees with itself leaks into user-facing text: a point
    fit with a fitted missingness mechanism published ``available: True | status:
    unavailable | reason: no longitudinal missingness-tilt adapter is implemented`` for an
    operation that works.
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
        "fixture_name", ["point_result", "longitudinal_result", "missing_outcome_result"]
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
        "fixture_name", ["point_result", "longitudinal_result", "missing_outcome_result"]
    )
    def test_only_an_unrunnable_operation_carries_a_reason(self, fixture_name, request) -> None:  # type: ignore[no-untyped-def]
        result = request.getfixturevalue(fixture_name)
        for row in result.sensitivity.capabilities:
            if row.available:
                assert row.reason is None, f"{row.operation} is available but explains itself away"
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
            assessment_cache={},
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
    result = dataclasses.replace(longitudinal_result, assessment_cache={})
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

    from cleverly.assessment import AssessmentItem, DiagnosticReport, ValidationReport

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
    result = dataclasses.replace(point_result, assessment_cache={})
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
    from cleverly.assessment import AssessmentItem, DiagnosticReport, ValidationReport

    repeated = dataclasses.replace(
        point_result, repeats=point_result.repeats * 2, assessment_cache={}
    )
    sensitivity = repeated.sensitivity.run_all()
    warning = AssessmentItem("support", AssessmentStatus.WARNING, "positivity warning")
    battery = AssessmentReport(ValidationReport((warning,)), DiagnosticReport(()), sensitivity)
    assert "support" in [item.name for item in battery.attention]
    assert "omitted_confounding" not in [item.name for item in battery.attention]
    assert "omitted_confounding" in [item.name for item in battery.omissions]


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_frames_pack_once_and_retrieval_cannot_mutate_cached_storage(
    point_result, backend, monkeypatch, tmp_path
):
    import joblib

    from cleverly.assessment import _CachedFrame

    result = dataclasses.replace(
        point_result,
        data=dataclasses.replace(point_result.data, backend=backend),
        assessment_cache={},
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
