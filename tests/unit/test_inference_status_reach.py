"""Every non-inferential status reaches every report a plain clustered fit publishes.

RM20 gives ordinary TMLE fits their first non-inferential statuses. RM12 followed its one
status through the reports a selector-path collaborative fit reaches, and two reports lay
outside that path: the fold-level ``CVTargeting`` report and the omitted-variable bound.
RM11 refuses the bound on collaborative fits, and a collaborative fit builds no
fold-level report, so neither report had met a status before.

Each class here forces one status from :data:`~cleverly._inference_status.NON_INFERENTIAL`
on the ordinary estimator's hook and fits a clustered TMLE with ``cv_evaluation=True``.
The forced hook reaches every status on one fit, the clustered statuses of
``tests/unit/test_cluster_status.py`` included, so a status added to the table is checked
here with no new test. The fit has 40 equal clusters, so its own hook gives it
``"influence_curve"`` and only the forced status applies. Every report must
then publish no inferential name and no confidence-limit text, and every operation must
answer or refuse by the reason of the status.

Two mutation controls prove the checks can fail. A fit whose stamp skips the fold-level
reports must fail the fold-report check. An omitted-variable module that ignores the
status must fail the bound check.
"""

from __future__ import annotations

import pickle
from collections.abc import Iterator
from typing import Any

import pytest

from cleverly._inference_status import NON_INFERENTIAL
from cleverly.assessment import AssessmentStatus
from cleverly.datasets import make_clustered
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError
from cleverly.sensitivity import omitted_variable as omitted_variable_module
from cleverly.sensitivity import omitted_variable_bounds, robustness_value
from tests.conftest import linear_in_sample
from tests.unit._inference_status_support import (
    ROUTES,
    assert_assessment_note,
    assert_evalue_unavailable,
    assert_fold_report_withholds,
    assert_fold_reports_restamped,
    assert_no_inferential_name,
    assert_no_inferential_text,
    assert_refused_by,
    assert_restamped,
    assert_variable_importance_refuses,
    stamp_headline_only,
)
from tests.unit._natural_course_support import never_fit_learners

pytestmark = pytest.mark.xdist_group("inference_status_reach")

#: What the combined report writes when an operation it ran raised ``CapabilityError``
#: under a capability row that said the operation was available.
DECLINED = "the operation declined this request"


@pytest.fixture(scope="module")
def frame() -> Any:
    """40 equal clusters of 10 rows, with a binary outcome, which needs no q_bounds."""
    return make_clustered(n=400, cluster_size=10, seed=7, family="binomial")[0]


def estimator(**overrides: Any) -> TMLE:
    settings = {"cross_fit": True, "n_folds": 5, "cv_evaluation": True, "estimands": ("ate",)}
    return TMLE(**linear_in_sample(**{**settings, **overrides}))


def fit(frame: Any, **overrides: Any) -> Any:
    return estimator(**overrides).fit(frame, outcome="Y", treatment="A", id="cluster").single()


@pytest.fixture(scope="module", params=list(NON_INFERENTIAL))
def status(request: pytest.FixtureRequest) -> Iterator[str]:
    """One status, forced on the ordinary hook for as long as its tests run."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(TMLE, "_inference_status", lambda self, data: request.param)
        yield request.param


@pytest.fixture(scope="module")
def result(status: str, frame: Any) -> Any:
    return fit(frame)


def assert_bound_withholds(result: Any, status: str) -> None:
    """The bound and robustness value expose no inferential limit, even as attributes."""
    bound = omitted_variable_bounds(result, "ate")
    assert bound.inference == status
    for name in ("ci_lower", "ci_upper", "robustness_value_ci"):
        with pytest.raises(CapabilityError) as raised:
            getattr(bound, name)
        assert_refused_by(status, raised)
    row = bound.to_dict()
    assert row["inference"] == status
    assert_no_inferential_name(row)
    assert {"plugin_interval_lower", "plugin_interval_upper"} <= set(row)
    assert bound.plugin_interval_lower == row["plugin_interval_lower"]
    assert bound.plugin_interval_upper == row["plugin_interval_upper"]
    assert bound.robustness_value_plugin_interval == row["robustness_value_plugin_interval"]
    assert_no_inferential_text(bound.summary())
    values = robustness_value(result, "ate")
    assert_no_inferential_name(values)
    assert "rv_plugin_interval" in values
    assert values["inference"] == status


class TestTheEstimatesAndTheirReports:
    def test_the_fit_declares_the_status(self, result: Any, status: str) -> None:
        assert result.inference_status == status
        for accessor in ("ci", "std_error", "pvalue"):
            with pytest.raises(CapabilityError) as raised:
                getattr(result["ate"], accessor)
            assert_refused_by(status, raised)

    def test_the_summary_prints_the_record(self, result: Any, status: str) -> None:
        record = NON_INFERENTIAL[status]
        text = result.summary()
        assert record.summary_label in text
        assert record.reason in text
        assert_no_inferential_text(text)

    def test_the_frame_names_the_status(self, result: Any, status: str) -> None:
        frame = result.to_frame()
        assert_no_inferential_name(frame.columns)
        assert list(frame["inference"]) == [status]


class TestTheFoldLevelReport:
    def test_the_fold_report_withholds(self, result: Any, status: str) -> None:
        assert_fold_report_withholds(result, status)

    def test_a_stamp_that_skips_the_fold_reports_fails_the_check(
        self, status: str, frame: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The mutation control: only the headline estimates carry the status."""
        stamp_headline_only(monkeypatch)
        mutant = fit(frame)
        assert mutant.inference_status == status
        with pytest.raises(AssertionError):
            assert_fold_report_withholds(mutant, status)


class TestTheOmittedVariableBound:
    def test_the_bound_withholds(self, result: Any, status: str) -> None:
        assert_bound_withholds(result, status)

    def test_pickle_keeps_the_guarded_accessors(self, result: Any, status: str) -> None:
        bound = pickle.loads(pickle.dumps(omitted_variable_bounds(result, "ate")))
        assert bound.inference == status
        with pytest.raises(CapabilityError):
            _ = bound.ci_lower
        assert bound.plugin_interval_lower == bound.to_dict()["plugin_interval_lower"]

    def test_old_pickle_fields_migrate_to_guarded_accessors(self, result: Any, status: str) -> None:
        bound = omitted_variable_bounds(result, "ate")
        old_state = dict(bound.__dict__)
        for name in ("ci_lower", "ci_upper", "robustness_value_ci"):
            old_state[name] = old_state.pop(f"_{name}")
        restored = type(bound).__new__(type(bound))
        restored.__setstate__(old_state)
        with pytest.raises(CapabilityError):
            _ = restored.ci_lower
        assert restored.plugin_interval_lower == bound.plugin_interval_lower
        assert restored.inference == status

    def test_a_module_that_ignores_the_status_fails_the_check(
        self, result: Any, status: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The mutation control: every name the bound publishes keeps its inferential form."""
        monkeypatch.setattr(omitted_variable_module, "spread_name", lambda name, status: name)
        with pytest.raises(AssertionError):
            assert_bound_withholds(result, status)


class TestTheCombinedReports:
    def test_the_sensitivity_battery_answers_or_refuses_by_the_reason(
        self, result: Any, status: str
    ) -> None:
        report = result.sensitivity.run_all(include_retargets=True, include_refits=True)
        for item in report.items:
            assert DECLINED not in item.detail, item
            assert_no_inferential_text(item.detail)
        evalue = report["evalue"]
        assert evalue.status is AssessmentStatus.UNAVAILABLE
        assert NON_INFERENTIAL[status].reason in evalue.detail
        assert_no_inferential_text(report.summary())

    def test_the_diagnostics_battery_answers(self, result: Any, status: str) -> None:
        report = result.diagnostics.run_all(include_retargets=True, include_refits=True)
        for item in report.items:
            assert DECLINED not in item.detail, item
            assert_no_inferential_text(item.detail)
        assert_no_inferential_text(report.summary())

    def test_the_nuisance_report_and_the_assessment_carry_the_note(
        self, result: Any, status: str
    ) -> None:
        assert_assessment_note(result, status)

    def test_the_assessment_answers(self, result: Any, status: str) -> None:
        assessment = result.assess()
        for item in assessment.validation.items:
            assert DECLINED not in item.detail, item
            assert_no_inferential_text(item.detail)
        assert_no_inferential_text(assessment.summary())

    def test_the_evalue_row_is_unavailable_with_the_reason(self, result: Any, status: str) -> None:
        assert_evalue_unavailable(result, status)


class TestVariableImportanceRefusesBeforeItFits:
    def test_the_refusal_arrives_before_the_first_learner_is_fitted(
        self, status: str, frame: Any
    ) -> None:
        assert_variable_importance_refuses(
            status,
            frame,
            covariates=["W1", "W2"],
            estimator=estimator(**never_fit_learners()),
            id="cluster",
        )


class TestARestoredArtifactIsReStamped:
    """An artifact saved before its configuration took a status loads under the status.

    The estimates are checked by the shared helper, which each surface's own test also
    calls on a real fit. This class adds the fold-level reports, which only this fit has.
    """

    @pytest.mark.parametrize("route", ROUTES)
    def test_every_report_carries_the_status_again(
        self, result: Any, status: str, route: str
    ) -> None:
        assert_fold_reports_restamped(assert_restamped(result, status, route), result, status)
