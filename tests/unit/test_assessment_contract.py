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
    AssessmentStatus,
    CapabilityError,
    CausalStudy,
    LongitudinalTreatment,
    PointTreatment,
    PositivityWarning,
    RegimeMean,
    load,
)
from cleverly.assessment import ASSESSMENT_CAPABILITIES, SENSITIVITY_ROUTES
from cleverly.datasets import make_linear_ate, make_longitudinal
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
        methods = {"tmle", "collaborative_tmle", "drtmle"} if family == "point" else {"tmle"}
        assert all(set(item.methods) == methods for item in family_rows)
    assert len({(item.result_family, item.operation) for item in ASSESSMENT_CAPABILITIES}) == len(
        ASSESSMENT_CAPABILITIES
    )


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
    assert len(scores.rows) == len(nuisances.rows) == expected
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
        assert rows["omitted_confounding"].requires_arguments == ()

    def test_every_argument_free_row_really_is_argument_free(self, point_result) -> None:  # type: ignore[no-untyped-def]
        """The gate that would have caught this: call what the report claims it can call."""
        import inspect

        for row in point_result.sensitivity.capabilities:
            if row.requires_arguments or not row.available:
                continue
            signature = inspect.signature(getattr(point_result.sensitivity, row.operation))
            required = [
                name
                for name, parameter in signature.parameters.items()
                if parameter.default is inspect.Parameter.empty
                and parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
            ]
            assert not required, f"{row.operation} needs {required} but declares none"


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
