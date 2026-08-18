"""The shared post-fit assessment contract across scalar result families."""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from cleverly import (
    ATE,
    AssessmentStatus,
    CapabilityError,
    CausalStudy,
    LongitudinalTreatment,
    PointTreatment,
    RegimeMean,
    load,
)
from cleverly.assessment import ASSESSMENT_CAPABILITIES
from cleverly.datasets import make_linear_ate, make_longitudinal
from cleverly.sensitivity._parameters import arm_parameters


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
            outcome_learner="glm",
            treatment_learner="glm",
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
        outcome_learner="glm",
        pseudo_learner="glm",
        treatment_learner="glm",
        censoring_learner="glm",
        n_folds=3,
        learner_folds=2,
        random_state=5,
        simultaneous=False,
    )


def test_every_diagnostic_operation_covers_every_result_family() -> None:
    expected = {"support", "nuisance_models", "score_equations", "refute", "stagewise"}
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
    score = point_result.diagnostics.score_equations()
    legacy_score = point_result.validation.score_check()
    assert score == legacy_score

    nuisance = point_result.diagnostics.nuisance_models()
    legacy_nuisance = point_result.validation.nuisance()
    assert nuisance == legacy_nuisance

    support = point_result.diagnostics.support()
    legacy_support = point_result._legacy_sensitivity.positivity()
    assert support == legacy_support


def test_longitudinal_stagewise_adapter_preserves_the_existing_table(longitudinal_result) -> None:  # type: ignore[no-untyped-def]
    before = longitudinal_result._legacy_diagnostics_frame()
    after = longitudinal_result.diagnostics.stagewise().to_frame()
    pd.testing.assert_frame_equal(after, before, check_exact=True)


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
    with pytest.raises(CapabilityError, match="full evidence-backed recursion/refit adapter"):
        longitudinal_result.sensitivity.omitted_confounding()


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
    restored = load(result.save(tmp_path / f"{fixture_name}.npz"))

    assert set(restored.assessment_cache) == cache_keys
    assert restored.validate() == validation
    assert restored.diagnostics.run_all() == diagnostics


def test_a_cached_frame_replays_in_the_callers_backend(point_result, tmp_path) -> None:  # type: ignore[no-untyped-def]
    before = point_result.sensitivity.truncation_curve(bounds=[0.02, 0.05])
    restored = load(point_result.save(tmp_path / "cached-frame.npz"))
    after = restored.sensitivity.truncation_curve(bounds=[0.02, 0.05])
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
        report = point_result.sensitivity.run_all(include_refits=True)
        assert {item.name for item in report.items} == {
            "omitted_confounding",
            "benchmark",
            "missingness",
        }

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
