"""Each capability row reads the predicate that its call raises from.

RM23 found rows that read ``available`` while the call then refused. Each case was a row
that answered from a flag of its own while the call answered from a check of its own. The
fix gives each pair one predicate: the call raises the sentence the predicate returns, and
the row quotes the same sentence. This module pins that agreement kind by kind.

A predicate that refuses nothing would pass an agreement test on a fit the call admits, so
each section carries two controls. A fit that the predicate admits must run, which is the
nonzero witness against over-refusal. A monkeypatched mutation that restores the old,
independent answer must make the agreement check fail, which proves that the check can see
the defect it exists for.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import importlib
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import ATE, CausalStudy, PointTreatment
from cleverly.assessment import (
    POINT_REPLAY_REFIT_CONFIGURATION,
    AssessmentStatus,
    DiagnosticsFacade,
    SensitivityFacade,
    replayability,
)
from cleverly.data import CausalData
from cleverly.datasets import make_binary_outcome, make_linear_ate
from cleverly.estimators import CTMLE, TMLE
from cleverly.estimators.serialize import dumps, loads
from cleverly.exceptions import CapabilityError, DataError
from cleverly.sensitivity import ConfounderStrengthGrid, simulated_confounding
from cleverly.sensitivity import missingness as missingness_module
from cleverly.sensitivity import omitted_variable as omitted_variable_module
from cleverly.sensitivity._simulated_confounding_request import simulated_confounding_refusal
from cleverly.sensitivity.missingness import (
    _FIT_WIDE_TILT_RULES,
    _fit_wide_tilt_rule,
    fit_wide_tilt_refusal,
    missingness_tilt,
    tipping_gamma,
)
from cleverly.sensitivity.omitted_variable import (
    _EVALUE_POINTER,
    _FIT_WIDE_BOUND_RULES,
    _RESPONSE_BOUND_REFUSAL,
    _RESPONSE_TILT_POINTER,
    OMITTED_VARIABLE_OPERATIONS,
    benchmark,
    benchmark_refusal,
    contour_data,
    fit_wide_bound_refusal,
    omitted_variable_bounds,
    resolve_parameter,
    robustness_value,
    sensitivity_elements,
)
from cleverly.sensitivity.positivity import (
    _TRUNCATION_RULES,
    truncation_axis,
    truncation_curve,
    truncation_refusal,
)
from cleverly.targets.population_intervention import NATURAL_COURSE_TILT_REFUSAL
from cleverly.validation.refute import (
    _REQUEST_RULES,
    DEFAULT_TESTS,
    BootstrapMeasurementError,
    EmpiricalInclusionRule,
    _RefuteRequest,
    refute,
    refute_refusal,
)
from tests.conftest import linear_in_sample
from tests.unit._capability_sweep_support import (
    DECLINED,
    DECLINED_TILT_KINDS,
    INSTRUMENT_ORDERING,
    KINDS,
    MUTATIONS,
    cross_fitted,
    ctmle_ordered,
    ctmle_stratified,
    discrete_fit,
    drtmle_companion_frame,
    fit_drtmle,
    reconfigured,
    reconfigured_ctmle_clustered,
    reconfigured_stratified,
    unbounded_scale_ate,
    without_provenance,
)
from tests.unit._confounding_support import confounding_study
from tests.unit._declaration_support import assert_replay_agrees, replay_disagreements
from tests.unit._natural_course_support import NeverFit, never_fit_learners
from tests.unit._simulated_confounding_support import (
    _estimate,
    _fit_attributable,
    _fit_with_a_support_constant_covariate,
)

# ----------------------------------------------------------------------------- the tilt

#: The tilt rule each kind of fit meets first, or ``None`` for a fit the tilt admits.
#: The five kinds between ``missing`` and ``natural_course`` are the ones RM23 found
#: declining: their rows read ``available`` and both calls refused.
TILT_RULE_OF: dict[str, str | None] = {
    "missing": None,
    "shift+missing": "continuous",
    "incremental+missing": "incremental",
    "regime+missing": "tiltable_parameters",
    "msm+missing": "tiltable_parameters",
    "rr+missing": "tiltable_parameters",
    "natural_course": "natural_course",
    "ordinary": "missing_outcome",
}

TILT_CALLS = {"missingness": missingness_tilt, "tipping_gamma": tipping_gamma}


@pytest.fixture(scope="module")
def tilt_fits() -> dict[str, Any]:
    """One fresh fit of each kind in :data:`TILT_RULE_OF`, shared by this section."""
    return {kind: KINDS[kind].build() for kind in TILT_RULE_OF}


def _raised(call: Callable[[], Any]) -> str | None:
    """The sentence ``call()`` refuses with, or ``None`` when it runs."""
    try:
        call()
    except CapabilityError as error:
        return str(error)
    return None


def _disagreement(label: object, row: Any, raised: str | None) -> list[str]:
    """How ``row`` disagrees with a call that refused with ``raised``, or ``[]``.

    ``raised`` is ``None`` when the call ran. ``label`` names the request in the message,
    and ``None`` names none.
    """
    prefix = "" if label is None else f"{label}: "
    if row.available and raised is not None:
        return [f"{prefix}the row reads available and the call raised {raised}"]
    if not row.available and raised != row.reason:
        return [f"{prefix}the row quotes {row.reason!r}, the call {raised!r}"]
    return []


def tilt_disagreements(result: Any) -> list[str]:
    """Every way the two tilt rows and the bound's pointer disagree with the calls.

    A fresh facade reads the rows, so a monkeypatched seam takes effect even when the
    result already memoized its own facade. An empty list is agreement.
    """
    facade = SensitivityFacade(result)
    problems = []
    for operation, call in TILT_CALLS.items():
        raised = _raised(functools.partial(call, result))
        problems += _disagreement(operation, facade.capability(operation), raised)
    bound = fit_wide_bound_refusal(result) or ""
    points = bound.endswith(_RESPONSE_TILT_POINTER)
    if points and not facade.capability("missingness").available:
        problems.append("the bound points at a tilt row that is not available")
    return problems


class TestTheTiltRowsReadTheCallsPredicate:
    """Both tilt rows and both tilt calls answer from :func:`fit_wide_tilt_refusal`."""

    def test_the_rule_table_is_ordered(self) -> None:
        """The names are the contract the rows read, so the order is pinned by name."""
        assert [name for name, _ in _FIT_WIDE_TILT_RULES] == [
            "longitudinal",
            "natural_course",
            "missing_outcome",
            "continuous",
            "incremental",
            "tiltable_parameters",
        ]

    def test_a_longitudinal_result_hears_the_literal_two_notebooks_print(self) -> None:
        """First, so no later rule reads a ``has_missing_outcome`` that is not there."""
        stand_in = SimpleNamespace(assessment_family="longitudinal")
        assert fit_wide_tilt_refusal(stand_in) == (
            "no longitudinal missingness-tilt adapter is implemented"
        )

    @pytest.mark.parametrize("kind", list(TILT_RULE_OF))
    def test_each_kind_meets_the_rule_the_table_names(
        self, tilt_fits: dict[str, Any], kind: str
    ) -> None:
        rule = _fit_wide_tilt_rule(tilt_fits[kind])
        assert (None if rule is None else rule[0]) == TILT_RULE_OF[kind]

    @pytest.mark.parametrize("kind", list(TILT_RULE_OF))
    @pytest.mark.parametrize("operation", list(TILT_CALLS))
    def test_the_row_and_the_call_agree(
        self, tilt_fits: dict[str, Any], kind: str, operation: str
    ) -> None:
        result = tilt_fits[kind]
        row = result.sensitivity.capability(operation)
        raised = _raised(functools.partial(TILT_CALLS[operation], result))
        rule = TILT_RULE_OF[kind]
        if rule is None:
            # The nonzero witness: a fit the table admits runs both calls.
            assert row.available and row.status is AssessmentStatus.PASSED
            assert row.reason is None
            assert raised is None
            return
        assert not row.available
        expected = (
            AssessmentStatus.NOT_APPLICABLE
            if rule == "missing_outcome"
            else AssessmentStatus.UNAVAILABLE
        )
        assert row.status is expected
        assert row.reason == fit_wide_tilt_refusal(result)
        assert raised == row.reason

    @pytest.mark.parametrize("kind", DECLINED_TILT_KINDS)
    def test_the_facade_refuses_before_it_calls(self, tilt_fits: dict[str, Any], kind: str) -> None:
        """The facade quotes the row, so its refusal carries the call's own sentence."""
        result = tilt_fits[kind]
        reason = fit_wide_tilt_refusal(result)
        assert reason is not None
        for operation in TILT_CALLS:
            with pytest.raises(CapabilityError) as error:
                getattr(result.sensitivity, operation)()
            assert str(error.value) == f"sensitivity {operation!r} is unavailable: {reason}"

    @pytest.mark.parametrize("kind", DECLINED_TILT_KINDS)
    def test_tipping_gamma_refuses_before_it_checks_search(
        self, tilt_fits: dict[str, Any], kind: str
    ) -> None:
        """A refused fit hears its refusal, and not the ``search=`` it could never use."""
        result = tilt_fits[kind]
        with pytest.raises(CapabilityError) as error:
            tipping_gamma(result, search=(1.0, 2.0))
        assert str(error.value) == fit_wide_tilt_refusal(result)
        # The nonzero witness: the fit the table admits checks the same search.
        with pytest.raises(ValueError, match="search must be a finite") as malformed:
            tipping_gamma(tilt_fits["missing"], search=(1.0, 2.0))
        assert not isinstance(malformed.value, CapabilityError)

    def test_a_combined_report_declines_no_tilt_row(self, tilt_fits: dict[str, Any]) -> None:
        """What RM23 saw: ``run_all`` ran the row, and the call declined it."""
        for kind in DECLINED_TILT_KINDS:
            report = tilt_fits[kind].sensitivity.run_all(include_retargets=True)
            for operation in TILT_CALLS:
                item = report[operation]
                assert item.status is AssessmentStatus.UNAVAILABLE, (kind, operation)
                assert "declined this request" not in item.detail, (kind, operation)

    def test_the_natural_course_rule_keeps_its_sentence(self, tilt_fits: dict[str, Any]) -> None:
        reason = fit_wide_tilt_refusal(tilt_fits["natural_course"])
        assert reason == NATURAL_COURSE_TILT_REFUSAL


class TestTheBoundPointsAtTheTiltOnlyWhereItRuns:
    """The response refusal of the bound names the tilt only where the tilt runs."""

    def test_a_missing_outcome_fit_keeps_the_pointer(self, tilt_fits: dict[str, Any]) -> None:
        reason = fit_wide_bound_refusal(tilt_fits["missing"])
        assert reason == _RESPONSE_BOUND_REFUSAL + _RESPONSE_TILT_POINTER

    @pytest.mark.parametrize("kind", [*DECLINED_TILT_KINDS, "natural_course"])
    def test_a_fit_the_tilt_refuses_hears_no_pointer(
        self, tilt_fits: dict[str, Any], kind: str
    ) -> None:
        assert fit_wide_bound_refusal(tilt_fits[kind]) == _RESPONSE_BOUND_REFUSAL


class TestEachTiltMutationRestoresADisagreement:
    """The agreement check sees each defect it exists for."""

    @pytest.mark.parametrize("kind", list(TILT_RULE_OF))
    def test_m0_every_kind_agrees_unmutated(self, tilt_fits: dict[str, Any], kind: str) -> None:
        assert tilt_disagreements(tilt_fits[kind]) == []

    @pytest.mark.parametrize("kind", DECLINED_TILT_KINDS)
    def test_rows_that_ignore_the_predicate_disagree(
        self, tilt_fits: dict[str, Any], kind: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The rows as they were: available on every missing-outcome fit."""
        MUTATIONS["M1"].apply(monkeypatch)
        problems = tilt_disagreements(tilt_fits[kind])
        assert len(problems) == 2
        assert all("reads available" in problem for problem in problems)

    def test_rows_that_ignore_the_predicate_leave_the_admitted_fit_alone(
        self, tilt_fits: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same mutation changes nothing where the table already said ``None``."""
        MUTATIONS["M1"].apply(monkeypatch)
        assert tilt_disagreements(tilt_fits["missing"]) == []

    @pytest.mark.parametrize("kind", DECLINED_TILT_KINDS)
    def test_an_unconditional_pointer_disagrees(
        self, tilt_fits: dict[str, Any], kind: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The pointer as it was: printed on every response fit but the natural course."""
        monkeypatch.setattr(omitted_variable_module, "fit_wide_tilt_refusal", lambda result: None)
        assert tilt_disagreements(tilt_fits[kind]) == [
            "the bound points at a tilt row that is not available"
        ]

    def test_a_call_that_ignores_the_predicate_disagrees(
        self, tilt_fits: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The calls must raise from the table too, or a row quotes a sentence nobody raises.

        The rows read the table through ``_fit_wide_tilt_rule``, so silencing the public
        predicate inside the module reaches the two calls alone. The regime call then falls
        through to the per-request check and raises a sentence the row does not quote.
        """
        monkeypatch.setattr(missingness_module, "fit_wide_tilt_refusal", lambda result: None)
        problems = tilt_disagreements(tilt_fits["regime+missing"])
        assert len(problems) == 2
        assert all("the row quotes" in problem for problem in problems)


# ----------------------------------------------------------------------- the truncation

PASSED = AssessmentStatus.PASSED
DEFERRED = AssessmentStatus.DEFERRED
UNAVAILABLE = AssessmentStatus.UNAVAILABLE

#: The requests a truncation row resolves: the bare one, and each explicit axis.
TRUNCATION_REQUESTS: tuple[dict[str, Any], ...] = ({}, {"mechanism": True}, {"mechanism": False})

#: What each kind's row reads for each request in :data:`TRUNCATION_REQUESTS`, in order.
#: ``missing`` admits every request, which is the nonzero witness for the mutations below.
TRUNCATION_STATUS_OF: dict[str, tuple[AssessmentStatus, ...]] = {
    "ordinary": (PASSED, UNAVAILABLE, PASSED),
    "missing": (PASSED, PASSED, PASSED),
    "incremental": (UNAVAILABLE, UNAVAILABLE, UNAVAILABLE),
    "incremental+missing": (DEFERRED, PASSED, UNAVAILABLE),
    "natural_course": (PASSED, PASSED, UNAVAILABLE),
}


@pytest.fixture(scope="module")
def truncation_fits() -> dict[str, Any]:
    """One fresh fit of each kind in :data:`TRUNCATION_STATUS_OF`, shared by this section."""
    return {kind: KINDS[kind].build() for kind in TRUNCATION_STATUS_OF}


def truncation_disagreements(result: Any) -> list[str]:
    """Every request whose truncation row and module call disagree.

    The module call sweeps one bound, which is enough to reach every refusal. A fresh
    facade resolves each row, so a monkeypatched seam takes effect even when the result
    already memoized its own facade. An empty list is agreement.
    """
    facade = DiagnosticsFacade(result)
    problems = []
    for arguments in TRUNCATION_REQUESTS:
        row = facade._capability_for_arguments("truncation_curve", arguments)
        raised = _raised(functools.partial(truncation_curve, result, [0.05], **arguments))
        problems += _disagreement(arguments, row, raised)
    return problems


class TestTheTruncationRowResolvesEachRequest:
    """The row and the call both read :func:`truncation_refusal`, one request at a time."""

    def test_the_rule_table_is_ordered(self) -> None:
        assert [name for name, _ in _TRUNCATION_RULES] == [
            "observation_axis",
            "incremental",
            "natural_course",
        ]

    @pytest.mark.parametrize("kind", list(TRUNCATION_STATUS_OF))
    def test_each_request_reads_the_status_the_table_names(
        self, truncation_fits: dict[str, Any], kind: str
    ) -> None:
        facade = truncation_fits[kind].diagnostics
        statuses = tuple(
            facade._capability_for_arguments("truncation_curve", arguments).status
            for arguments in TRUNCATION_REQUESTS
        )
        assert statuses == TRUNCATION_STATUS_OF[kind]
        # The bare request is what ``capability()`` reports.
        assert facade.capability("truncation_curve").status is statuses[0]

    def test_the_module_call_refuses_the_incremental_default(
        self, truncation_fits: dict[str, Any]
    ) -> None:
        """RM23 measured a flat curve at 2.96262 here, because ``g`` is inside the estimand."""
        result = truncation_fits["incremental"]
        row = result.diagnostics.capability("truncation_curve")
        with pytest.raises(CapabilityError) as error:
            truncation_curve(result)
        assert str(error.value) == row.reason == truncation_refusal(result)
        assert "*inside* the estimand" in str(error.value)

    def test_an_incremental_fit_with_missing_outcomes_defers_on_the_axis(
        self, truncation_fits: dict[str, Any]
    ) -> None:
        result = truncation_fits["incremental+missing"]
        bare = result.diagnostics.capability("truncation_curve")
        assert bare.status is AssessmentStatus.DEFERRED
        assert bare.requires_arguments == ("mechanism",)
        assert bare.reason == truncation_refusal(result)
        # Naming the axis the row asks for runs the curve, directly and in a report.
        curve = result.diagnostics.truncation_curve([0.01, 0.05], mechanism=True)
        assert len(curve["bound"]) == 2 * len(result.estimates)
        report = result.diagnostics.run_all(
            include_retargets=True, arguments={"truncation_curve": {"mechanism": True}}
        )
        assert report["truncation_curve"].status is AssessmentStatus.COMPLETED
        # The bare report names the argument rather than declining the call.
        skipped = result.diagnostics.run_all(include_retargets=True)["truncation_curve"]
        assert skipped.status is AssessmentStatus.DEFERRED
        assert "declined this request" not in skipped.detail
        # The refused axis reads unavailable, with the call's own sentence.
        refused = result.diagnostics._capability_for_arguments(
            "truncation_curve", {"mechanism": False}
        )
        assert refused.status is AssessmentStatus.UNAVAILABLE
        assert refused.reason == truncation_refusal(result, False) == bare.reason
        with pytest.raises(CapabilityError, match=r"is unavailable: the propensity g"):
            result.diagnostics.truncation_curve(mechanism=False)

    def test_an_ordinary_fit_refuses_the_observation_axis_it_lacks(
        self, truncation_fits: dict[str, Any]
    ) -> None:
        result = truncation_fits["ordinary"]
        row = result.diagnostics._capability_for_arguments("truncation_curve", {"mechanism": True})
        assert row.status is AssessmentStatus.UNAVAILABLE
        assert row.reason == truncation_refusal(result, True)
        assert row.reason is not None
        assert row.reason.startswith("mechanism=True needs a fit with missing outcomes")
        with pytest.raises(CapabilityError, match=r"is unavailable: mechanism=True needs"):
            result.diagnostics.truncation_curve(mechanism=True)
        # The nonzero witness: the default axis of the same fit runs.
        assert result.diagnostics.capability("truncation_curve").available
        assert len(result.diagnostics.truncation_curve([0.05])["bound"]) == len(result.estimates)

    def test_a_natural_course_fit_sweeps_its_only_axis_by_default(
        self, truncation_fits: dict[str, Any]
    ) -> None:
        result = truncation_fits["natural_course"]
        assert truncation_axis(result, None) is True
        assert result.diagnostics.capability("truncation_curve").available
        curve = result.diagnostics.truncation_curve([0.05])
        assert len(curve["bound"]) == 1
        with pytest.raises(CapabilityError, match=r"is unavailable: NaturalCourseMean"):
            result.diagnostics.truncation_curve(mechanism=False)


class TestEachTruncationMutationRestoresADisagreement:
    """The agreement check sees a row that stops reading the predicate."""

    @pytest.mark.parametrize("kind", list(TRUNCATION_STATUS_OF))
    def test_m0_every_kind_agrees_unmutated(
        self, truncation_fits: dict[str, Any], kind: str
    ) -> None:
        assert truncation_disagreements(truncation_fits[kind]) == []

    @pytest.mark.parametrize("kind", ["incremental", "incremental+missing"])
    def test_a_row_that_ignores_the_predicate_disagrees_on_the_bare_request(
        self, truncation_fits: dict[str, Any], kind: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The row as it was: available on either axis, so a report ran the refused default."""
        MUTATIONS["M2"].apply(monkeypatch)
        problems = truncation_disagreements(truncation_fits[kind])
        assert any(problem.startswith("{}: the row reads available") for problem in problems)

    def test_the_same_mutation_leaves_a_fit_that_admits_every_axis_alone(
        self, truncation_fits: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        MUTATIONS["M2"].apply(monkeypatch)
        assert truncation_disagreements(truncation_fits["missing"]) == []


# --------------------------------------------------------------------------- the refute

#: The estimand each refute kind names, and a ``tests=`` value that runs on it. Each value
#: leaves out the one default test the kind refuses. ``ordinary`` refuses none, which is
#: the nonzero witness for the mutation below.
REFUTE_REQUEST_OF: dict[str, tuple[str, tuple[str, ...]]] = {
    "split_plan": ("ate", ("placebo", "random_common_cause")),
    "natural_course": ("ey_obs", ("random_common_cause", "subset")),
    "ordinary": ("ate", ("placebo", "random_common_cause")),
}

#: The kinds whose ``refute`` row read ``available`` before RM23 while the call refused
#: its default tests.
DEFERRED_REFUTE_KINDS = ("split_plan", "natural_course")


@pytest.fixture(scope="module")
def refute_fits() -> dict[str, Any]:
    """One fresh fit of each kind in :data:`REFUTE_REQUEST_OF`, shared by this section."""
    return {kind: KINDS[kind].build() for kind in REFUTE_REQUEST_OF}


def _spied(result: Any) -> Any:
    """A copy of ``result`` whose estimator fails at its first learner fit.

    The call's own preflight is what the row must agree with, and a refit is what that
    preflight precedes. So the copy reaches a :class:`NeverFit` learner exactly when the
    call admits the request, and no test here pays for a refit to find that out.
    """
    estimator = copy.copy(result.estimator)
    vars(estimator).update(never_fit_learners())
    return dataclasses.replace(result, estimator=estimator)


def _spied_raised(call: Callable[[], Any]) -> str | None:
    """The sentence ``call()`` refuses with, or ``None`` when it reaches a learner fit.

    ``call`` builds its :func:`_spied` copy itself, which resets ``NeverFit.calls``. The
    generated and measurement tests retain a failed refit rather than raise, so a call
    that returns must show the spy in the count.
    """
    try:
        call()
    except CapabilityError as error:
        return str(error)
    except AssertionError as error:
        assert "before any learner is fitted" in str(error)
        return None
    assert NeverFit.calls > 0, "the spied call neither refused nor reached a learner fit"
    return None


def refute_requests(kind: str) -> tuple[dict[str, Any], ...]:
    """The bare request, a request that names tests that run, and the default tests."""
    estimand, runs = REFUTE_REQUEST_OF[kind]
    return (
        {"estimand": estimand},
        {"estimand": estimand, "tests": runs},
        {"estimand": estimand, "tests": DEFAULT_TESTS},
    )


def refute_disagreements(result: Any, kind: str) -> list[str]:
    """Every request whose ``refute`` row and module call disagree.

    A fresh facade resolves each row, so a monkeypatched seam takes effect even when the
    result already memoized its own facade. An empty list is agreement.
    """
    facade = DiagnosticsFacade(result)
    problems = []
    for arguments in refute_requests(kind):
        row = facade._capability_for_arguments("refute", arguments)
        raised = _spied_raised(
            lambda request=arguments: refute(_spied(result), n_replicates=1, **request)
        )
        problems += _disagreement(arguments, row, raised)
    return problems


class TestTheRefuteRowResolvesEachRequest:
    """The row and the call both read :func:`refute_refusal`, one request at a time."""

    def test_the_rule_table_is_ordered(self) -> None:
        assert [name for name, _ in _REQUEST_RULES] == [
            "estimator",
            "estimand",
            "no_effect_null",
            "row_set_under_plan",
            "generated_rule",
            "generated_eligibility",
            "measurement_declaration",
            "measurement_rule",
            "measurement_eligibility",
            "measurement_budget",
            "generated_budget",
        ]

    @pytest.mark.parametrize("kind", DEFERRED_REFUTE_KINDS)
    def test_the_bare_request_defers_on_tests(self, refute_fits: dict[str, Any], kind: str) -> None:
        result = refute_fits[kind]
        bare, _, _ = refute_requests(kind)
        row = result.diagnostics._capability_for_arguments("refute", bare)
        assert row.status is AssessmentStatus.DEFERRED
        assert row.requires_arguments == ("tests",)
        assert row.reason == refute_refusal(result, estimand=bare["estimand"], tests=DEFAULT_TESTS)
        with pytest.raises(CapabilityError) as error:
            result.diagnostics.refute(**bare)
        assert str(error.value) == f"diagnostic 'refute' is deferred: {row.reason}"
        # A combined report names the argument rather than declining the call.
        report = result.diagnostics.run_all(include_refits=True, arguments={"refute": bare})
        assert report["refute"].status is AssessmentStatus.DEFERRED
        assert "declined this request" not in report["refute"].detail

    @pytest.mark.parametrize("kind", DEFERRED_REFUTE_KINDS)
    def test_naming_tests_that_run_lifts_the_deferral(
        self, refute_fits: dict[str, Any], kind: str
    ) -> None:
        result = refute_fits[kind]
        _, runs, _ = refute_requests(kind)
        request = {**runs, "n_replicates": 1}
        assert result.diagnostics._capability_for_arguments("refute", request).available
        report = result.diagnostics.run_all(
            include_refits=True, arguments={"refute": request}, random_state=0
        )
        item = report["refute"]
        assert item.report is not None, item.detail
        assert [test.name for test in item.report.tests] == list(runs["tests"])

    @pytest.mark.parametrize("kind", DEFERRED_REFUTE_KINDS)
    def test_the_default_tests_named_read_unavailable(
        self, refute_fits: dict[str, Any], kind: str
    ) -> None:
        result = refute_fits[kind]
        _, _, defaults = refute_requests(kind)
        row = result.diagnostics._capability_for_arguments("refute", defaults)
        assert row.status is AssessmentStatus.UNAVAILABLE
        with pytest.raises(CapabilityError) as error:
            refute(result, **defaults)
        assert str(error.value) == row.reason

    @pytest.mark.parametrize("kind", DEFERRED_REFUTE_KINDS)
    def test_the_call_refuses_before_any_learner_fit(
        self, refute_fits: dict[str, Any], kind: str
    ) -> None:
        result = _spied(refute_fits[kind])
        _, runs, defaults = refute_requests(kind)
        with pytest.raises(CapabilityError):
            refute(result, **defaults)
        assert NeverFit.calls == 0
        # The nonzero witness: the spy is live, so a request the call admits reaches it.
        with pytest.raises(AssertionError, match="before any learner is fitted"):
            refute(result, n_replicates=1, **runs)
        assert NeverFit.calls > 0

    def test_an_unknown_test_name_is_reported_before_any_refusal(
        self, refute_fits: dict[str, Any]
    ) -> None:
        """The malformed argument first, as the omitted-variable calls report it."""
        with pytest.raises(ValueError, match="unknown refutation test 'bogus'") as error:
            refute(refute_fits["split_plan"], estimand="not_reported", tests=("bogus",))
        assert not isinstance(error.value, CapabilityError)

    @pytest.mark.parametrize(
        ("request_", "message"),
        [
            ({"tests": ("typo", "subset")}, "unknown refutation test 'typo'"),
            ({"tests": ("subset",), "n_replicates": 0}, "n_replicates must be positive"),
            (
                {"tests": ("placebo", "negative_control_outcome")},
                "the negative_control_outcome test needs an outcome array",
            ),
        ],
    )
    def test_the_facade_reports_a_malformed_argument_before_the_row(
        self, refute_fits: dict[str, Any], request_: dict[str, Any], message: str
    ) -> None:
        """The facade, the row and the function agree on the order: malformed first.

        Each request names ``subset``, which this fit refuses. The facade used to read the
        row first and raise that refusal, while the function raised ``ValueError``.
        """
        result = refute_fits["split_plan"]
        request = {"estimand": "ate", **request_}
        # The predicate leaves a malformed request to the call, so the row does not refuse.
        assert refute_refusal(result, **request) is None
        assert result.diagnostics._capability_for_arguments("refute", request).available
        for call in (
            lambda: refute(result, **request),
            lambda: result.diagnostics.refute(**request),
        ):
            with pytest.raises(ValueError, match=message) as error:
                call()
            assert not isinstance(error.value, CapabilityError)
        # The nonzero witness: the well-formed request still meets the refusal.
        with pytest.raises(CapabilityError, match="refit on rows this fit did not run"):
            result.diagnostics.refute(estimand="ate", tests=("subset",))

    def test_a_malformed_argument_precedes_the_replay_gate(self) -> None:
        """A reconfigured result whose refit is refused still hears the malformed argument."""
        result = reconfigured_stratified()
        assert not result.diagnostics.capability("refute").available
        with pytest.raises(ValueError, match="unknown refutation test 'typo'") as error:
            result.diagnostics.refute(tests=("typo",))
        assert not isinstance(error.value, CapabilityError)
        with pytest.raises(CapabilityError, match="refitting the nuisance models is unavailable"):
            result.diagnostics.refute(tests=("placebo",))

    @pytest.mark.parametrize("surface", ("module", "facade", "run_all", "assess"))
    def test_missing_negative_control_outcome_precedes_every_refit(
        self, refute_fits: dict[str, Any], surface: str
    ) -> None:
        result = _spied(refute_fits["ordinary"])
        request = {"tests": ("placebo", "negative_control_outcome"), "n_replicates": 1}
        calls = {
            "module": lambda: refute(result, **request),
            "facade": lambda: result.diagnostics.refute(**request),
            "run_all": lambda: result.diagnostics.run_all(
                include_refits=True, arguments={"refute": request}
            ),
            "assess": lambda: result.assess(include_refits=True, arguments={"refute": request}),
        }
        with pytest.raises(ValueError, match="the negative_control_outcome test needs"):
            calls[surface]()
        assert NeverFit.calls == 0
        assert refute_refusal(result, estimand="ate", **request) is None

    def test_supplied_negative_control_outcome_reaches_a_refit(
        self, refute_fits: dict[str, Any]
    ) -> None:
        result = _spied(refute_fits["ordinary"])
        with pytest.raises(AssertionError, match="before any learner is fitted"):
            refute(
                result,
                tests=("placebo", "negative_control_outcome"),
                negative_control_outcome=np.zeros(result.data.n),
                n_replicates=1,
            )
        assert NeverFit.calls > 0


class TestEachRefuteMutationRestoresADisagreement:
    """The agreement check sees a row that stops reading the predicate."""

    @pytest.mark.parametrize("kind", list(REFUTE_REQUEST_OF))
    def test_m0_every_kind_agrees_unmutated(self, refute_fits: dict[str, Any], kind: str) -> None:
        assert refute_disagreements(refute_fits[kind], kind) == []

    @pytest.mark.parametrize("kind", DEFERRED_REFUTE_KINDS)
    def test_a_row_that_ignores_the_predicate_disagrees_on_the_bare_request(
        self, refute_fits: dict[str, Any], kind: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The row as it was: available, so a report ran the refused default tests."""
        MUTATIONS["M3"].apply(monkeypatch)
        problems = refute_disagreements(refute_fits[kind], kind)
        assert any("the row reads available" in problem for problem in problems)

    def test_the_same_mutation_leaves_a_fit_that_admits_every_test_alone(
        self, refute_fits: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        MUTATIONS["M3"].apply(monkeypatch)
        assert refute_disagreements(refute_fits["ordinary"], "ordinary") == []


class TestReportedScaleNoEffectNull:
    """The request gate and both null-effect verdicts use the parameter target."""

    @pytest.mark.parametrize("estimand", ("rr", "or"))
    @pytest.mark.parametrize("test", ("placebo", "negative_control_outcome"))
    def test_ratios_compare_with_one(
        self,
        bound_fits: dict[str, Any],
        estimand: str,
        test: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from cleverly.study import ParameterKey

        fitted = bound_fits["ratio_only"]
        alias = f"reported_{estimand}"
        result = dataclasses.replace(
            fitted,
            estimates={alias: fitted[estimand]},
            parameter_keys={alias: ParameterKey(alias, estimand)},
        )
        request: dict[str, Any] = {"estimand": alias, "tests": (test,), "n_replicates": 1}
        if test == "negative_control_outcome":
            request["negative_control_outcome"] = np.zeros(result.data.n)
        tolerance = 0.8 / result[alias].plugin_std_error
        assert result.parameter_keys[alias].estimand == estimand
        assert result.diagnostics._capability_for_arguments("refute", request).available
        assert refute_refusal(result, **request) is None

        class RatioRefit:
            def refit(self, *_: Any, **__: Any) -> dict[str, Any]:
                return {alias: SimpleNamespace(psi=2.0, plugin_std_error=0.8 / tolerance)}

        controlled = dataclasses.replace(result, estimator=RatioRefit())
        report = refute(controlled, random_state=0, tolerance=tolerance, **request)[test]
        assert report.expectation == "~ 1"
        assert report.passed
        # A raw-scale comparison sees a distance of one and wrongly fails.
        refute_module = importlib.import_module("cleverly.validation.refute")
        monkeypatch.setattr(
            refute_module, "_no_effect_difference", lambda value, null: value - null
        )
        assert not refute(controlled, random_state=0, tolerance=tolerance, **request)[test].passed

    @pytest.mark.parametrize("kind, estimand", (("par", "par"), ("paf", "paf")))
    def test_attributable_parameters_keep_zero(
        self, bound_fits: dict[str, Any], kind: str, estimand: str
    ) -> None:
        result = bound_fits[kind]
        assert refute_refusal(result, estimand=estimand, tests=("placebo",)) is None
        assert result.diagnostics._capability_for_arguments(
            "refute", {"estimand": estimand, "tests": ("placebo",)}
        ).available

    def test_direct_multi_arm_alias_uses_its_stem(self, ambiguous_fits: dict[str, Any]) -> None:
        result = ambiguous_fits["multi_arm"]
        alias = "ate[low vs high]"
        assert result.parameter_keys == {}
        assert alias in result.estimates
        request = {"estimand": alias, "tests": ("placebo",), "n_replicates": 1}
        assert refute_refusal(result, **request) is None
        assert result.diagnostics._capability_for_arguments("refute", request).available
        report = result.diagnostics.refute(random_state=0, **request)
        assert report["placebo"].estimand == alias
        assert report["placebo"].expectation == "~ 0"

    def test_ratio_placebo_reports_the_mean_it_tests(self, bound_fits: dict[str, Any]) -> None:
        result = bound_fits["ratio_only"]

        class RatioRefit:
            calls = 0

            def refit(self, *_: Any, **__: Any) -> dict[str, Any]:
                value = (1.0, 4.0)[self.calls % 2]
                self.calls += 1
                return {"rr": SimpleNamespace(psi=value)}

        controlled = dataclasses.replace(result, estimator=RatioRefit())
        report = refute(
            controlled, estimand="rr", tests=("placebo",), n_replicates=2, random_state=0
        )
        placebo = report["placebo"]
        assert placebo.values == (1.0, 4.0)
        assert placebo.mean == 2.0
        assert "geometric mean placebo estimate +2" in placebo.detail
        assert float(report.to_frame()["refuted_mean"].iloc[0]) == 2.0

    def test_ratio_noise_reports_the_mean_it_tests(self, bound_fits: dict[str, Any]) -> None:
        result = bound_fits["ratio_only"]
        original = result["rr"].psi

        class RatioRefit:
            calls = 0

            def refit(self, *_: Any, **__: Any) -> dict[str, Any]:
                value = (original / 2, original * 2)[self.calls % 2]
                self.calls += 1
                return {"rr": SimpleNamespace(psi=value)}

        controlled = dataclasses.replace(result, estimator=RatioRefit())
        report = refute(
            controlled,
            estimand="rr",
            tests=("random_common_cause",),
            n_replicates=2,
            random_state=0,
        )
        noise = report["random_common_cause"]
        assert noise.passed
        assert noise.mean == pytest.approx(original)
        assert float(report.to_frame()["refuted_mean"].iloc[0]) == pytest.approx(original)

    @pytest.mark.parametrize("test", ("random_common_cause", "subset"))
    def test_ratio_stability_uses_log_scale(
        self, bound_fits: dict[str, Any], test: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = bound_fits["ratio_only"]
        original = result["rr"].psi
        if test == "random_common_cause":
            values = (original * 1.2,)
            threshold = (abs(np.log(1.2)) + abs(original * 0.2)) / 2
        else:
            values = (1.0, 2.0)
            threshold = (np.std(np.log(values), ddof=1) + np.std(values, ddof=1)) / 2

        class RatioRefit:
            calls = 0

            def refit(self, *_: Any, **__: Any) -> dict[str, Any]:
                value = values[self.calls % len(values)]
                self.calls += 1
                return {"rr": SimpleNamespace(psi=value)}

        controlled = dataclasses.replace(result, estimator=RatioRefit())
        tolerance = float(threshold / result["rr"].plugin_std_error)
        if test == "subset":
            tolerance *= np.sqrt(0.7)
        request = {"estimand": "rr", "tests": (test,), "n_replicates": len(values)}
        report = refute(controlled, tolerance=tolerance, random_state=0, **request)[test]
        assert report.values == values
        assert report.passed
        if test == "subset":
            assert report.spread == pytest.approx(np.std(np.log(values), ddof=1))
            assert report.expectation.startswith("log-scale scatter")
        refute_module = importlib.import_module("cleverly.validation.refute")
        monkeypatch.setattr(refute_module, "_stability_value", lambda value, _: value)
        assert not refute(controlled, tolerance=tolerance, random_state=0, **request)[test].passed

    @pytest.mark.parametrize(
        "target, axis",
        (
            ("ate", "arm"),
            ("ate_regime", "regime"),
            ("ate_ipsi", "ipsi"),
            ("ate_shift", "shift"),
        ),
    )
    def test_additive_targets_keep_zero_through_structured_keys(
        self, refute_fits: dict[str, Any], target: str, axis: str
    ) -> None:
        from cleverly.study import ParameterKey
        from cleverly.validation.refute import _no_effect_null

        fitted = refute_fits["ordinary"]
        alias = f"reported_{target}"
        result = dataclasses.replace(
            fitted,
            estimates={alias: fitted["ate"]},
            parameter_keys={alias: ParameterKey(alias, target, axis)},
        )
        assert _no_effect_null(result, alias) == 0.0
        assert refute_refusal(result, estimand=alias, tests=("placebo",)) is None

    @pytest.mark.parametrize("estimand", ("ey1", "ey0", "ey_obs", "ey_shift", "msm"))
    @pytest.mark.parametrize("test", ("placebo", "negative_control_outcome"))
    def test_levels_and_coefficients_refuse_before_refit(
        self, refute_fits: dict[str, Any], estimand: str, test: str
    ) -> None:
        result = refute_fits["ordinary"]
        from cleverly.study import ParameterKey

        alias = "msm[a]" if estimand == "msm" else estimand
        keyed = dataclasses.replace(
            result,
            estimates={alias: result["ate"]},
            parameter_keys={alias: ParameterKey(alias, estimand)},
        )
        request: dict[str, Any] = {"estimand": alias, "tests": (test,)}
        if test == "negative_control_outcome":
            request["negative_control_outcome"] = np.zeros(result.data.n)
        reason = refute_refusal(keyed, **request)
        assert reason is not None and "fixed no-effect value" in reason
        row = keyed.diagnostics._capability_for_arguments("refute", request)
        assert not row.available and row.reason == reason
        with pytest.raises(CapabilityError, match="fixed no-effect value"):
            refute(_spied(keyed), **request)
        assert NeverFit.calls == 0


# ------------------------------------------------------------------ the bare request


@pytest.fixture(scope="module")
def ambiguous_fits() -> dict[str, Any]:
    """Two fits that report no bare ``ate``: one alias, and three."""
    return {kind: KINDS[kind].build() for kind in ("natural_course", "multi_arm")}


class TestThePublicRowIsTheBareRequestsRow:
    """``capability()`` reads the estimand gate, as the bare request does.

    Before, the public row read ``passed`` on these fits, while the bare request deferred
    on ``estimand`` and the bare call refused the default ``ate``.
    """

    @pytest.mark.parametrize("kind", ["natural_course", "multi_arm"])
    def test_the_bare_refute_row_defers_on_the_estimand(
        self, ambiguous_fits: dict[str, Any], kind: str
    ) -> None:
        result = ambiguous_fits[kind]
        row = result.diagnostics.capability("refute")
        assert row.status is AssessmentStatus.DEFERRED
        assert row.requires_arguments == ("estimand",)
        assert row == result.diagnostics._capability_for_arguments("refute", {})
        assert row in result.diagnostics.capabilities
        # The bare call refuses, in the call's sentence, through the same predicate.
        reason = refute_refusal(result, estimand="ate", tests=DEFAULT_TESTS)
        assert reason == "estimand 'ate' was not requested in this fit"
        with pytest.raises(CapabilityError) as error:
            result.diagnostics.refute(n_replicates=1)
        assert str(error.value) == f"diagnostic 'refute' is unavailable: {reason}"

    def test_a_call_that_names_an_estimand_is_not_refused(
        self, ambiguous_fits: dict[str, Any]
    ) -> None:
        """The nonzero witness: a direct call applies no estimand gate, so it runs."""
        result = ambiguous_fits["multi_arm"]
        chosen = next(iter(result.estimates))
        assert result.sensitivity.capability("omitted_confounding").status is (
            AssessmentStatus.DEFERRED
        )
        bounds = result.sensitivity.omitted_confounding(estimand=chosen, cf_y=0.05, cf_d=0.05)
        assert bounds.lower < result[chosen].psi < bounds.upper
        report = result.diagnostics.refute(
            estimand=chosen, tests=("random_common_cause",), n_replicates=1, random_state=0
        )
        assert [test.name for test in report.tests] == ["random_common_cause"]


# --------------------------------------------- the refute refusals before any refit

#: The module, which ``cleverly.validation`` shadows with the function of the same name.
refute_module = importlib.import_module("cleverly.validation.refute")

#: A four-draw rule. The rule refuses ``minimum_draws * alpha < 2``, so four draws need
#: ``alpha=0.5``.
FOUR_DRAWS = EmpiricalInclusionRule(alpha=0.5, minimum_draws=4)
MEASURED = BootstrapMeasurementError(variables=("W1",))


class _UnregisteredRule(EmpiricalInclusionRule):
    """An inclusion rule of another type, which the refute rules refuse by type."""


#: Each rule the call raised before any refit while the row did not read it, as the fit
#: and the request it refuses first. ``binary`` is a direct fit, which records no
#: identification, and ``study`` is a study fit, which records it.
PRE_REFIT_REQUESTS: dict[str, tuple[str, dict[str, Any]]] = {
    "generated_rule": (
        "study",
        {
            "tests": ("dummy_outcome",),
            "n_replicates": 4,
            "outcome_rule": _UnregisteredRule(alpha=0.5, minimum_draws=4),
        },
    ),
    "generated_eligibility": (
        "binary",
        {"tests": ("dummy_outcome",), "n_replicates": 4, "outcome_rule": FOUR_DRAWS},
    ),
    "measurement_declaration": (
        "study",
        {
            "tests": ("bootstrap_measurement_error",),
            "n_replicates": 4,
            "measurement_error_rule": FOUR_DRAWS,
        },
    ),
    "measurement_rule": (
        "study",
        {
            "tests": ("bootstrap_measurement_error",),
            "n_replicates": 4,
            "bootstrap_measurement_error": MEASURED,
            "measurement_error_rule": _UnregisteredRule(alpha=0.5, minimum_draws=4),
        },
    ),
    "measurement_eligibility": (
        "study",
        {
            "tests": ("bootstrap_measurement_error",),
            "n_replicates": 4,
            "bootstrap_measurement_error": BootstrapMeasurementError(
                variables=("W1",), resampling="cluster"
            ),
            "measurement_error_rule": FOUR_DRAWS,
        },
    ),
    "measurement_budget": (
        "study",
        {
            "tests": ("bootstrap_measurement_error",),
            "n_replicates": 1,
            "bootstrap_measurement_error": MEASURED,
        },
    ),
    "generated_budget": ("study", {"tests": ("dummy_outcome",), "n_replicates": 1}),
}

#: A request of each test that the call admits on the study fit: the nonzero witness.
PRE_REFIT_RUNS: dict[str, dict[str, Any]] = {
    "dummy_outcome": {"tests": ("dummy_outcome",), "n_replicates": 4, "outcome_rule": FOUR_DRAWS},
    "simulated_outcome": {
        "tests": ("simulated_outcome",),
        "n_replicates": 4,
        "outcome_rule": FOUR_DRAWS,
    },
    "bootstrap_measurement_error": {
        "tests": ("bootstrap_measurement_error",),
        "n_replicates": 4,
        "bootstrap_measurement_error": MEASURED,
        "measurement_error_rule": FOUR_DRAWS,
    },
}


@pytest.fixture(scope="module")
def pre_refit_fits() -> dict[str, Any]:
    """An in-sample study fit of the ATE with a Gaussian outcome, and the binary kind."""
    frame, _ = make_linear_ate(n=200, seed=0)
    study = CausalStudy(
        frame,
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4")),
    )
    fitted = study.identify(ATE()).estimate(
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        cross_fit=False,
        simultaneous=False,
        random_state=0,
    )
    return {"study": fitted, "binary": KINDS["binary"].build()}


def _first_rule(result: Any, request: dict[str, Any]) -> str | None:
    """The name of the first rule of the table that refuses ``request``."""
    arguments = _RefuteRequest(estimand="ate", **{**request, "tests": tuple(request["tests"])})
    return next(
        (name for name, rule in _REQUEST_RULES if rule(result, arguments) is not None), None
    )


def pre_refit_disagreements(result: Any, request: dict[str, Any]) -> list[str]:
    """Whether the ``refute`` row and the module call disagree on one request.

    The call runs on a spied copy, so a request it admits stops at the first learner fit.
    A fresh facade resolves the row, so a monkeypatched predicate takes effect.
    """
    row = DiagnosticsFacade(result)._capability_for_arguments(
        "refute", {"estimand": "ate", **request}
    )
    raised = _spied_raised(
        lambda: refute(_spied(result), estimand="ate", random_state=0, **request)
    )
    return _disagreement(None, row, raised)


class TestTheRefuteRowReadsEveryRefusalBeforeARefit:
    """Every refusal ``refute`` raises before a refit is a rule the row reads."""

    @pytest.mark.parametrize("rule", list(PRE_REFIT_REQUESTS))
    def test_each_request_meets_the_rule_the_table_names(
        self, pre_refit_fits: dict[str, Any], rule: str
    ) -> None:
        kind, request = PRE_REFIT_REQUESTS[rule]
        assert _first_rule(pre_refit_fits[kind], request) == rule

    @pytest.mark.parametrize("rule", list(PRE_REFIT_REQUESTS))
    def test_the_row_and_every_entry_point_quote_the_refusal(
        self, pre_refit_fits: dict[str, Any], rule: str
    ) -> None:
        """Before the row read these rules, a combined report declined each request."""
        kind, request = PRE_REFIT_REQUESTS[rule]
        result = pre_refit_fits[kind]
        request = {"estimand": "ate", **request}
        reason = refute_refusal(result, **request)
        assert reason is not None
        row = result.diagnostics._capability_for_arguments("refute", request)
        assert row.status is AssessmentStatus.UNAVAILABLE
        assert row.reason == reason
        with pytest.raises(CapabilityError) as error:
            refute(_spied(result), **request)
        assert str(error.value) == reason
        assert NeverFit.calls == 0
        with pytest.raises(CapabilityError) as error:
            result.diagnostics.refute(**request)
        assert str(error.value) == f"diagnostic 'refute' is unavailable: {reason}"
        report = result.diagnostics.run_all(
            include_refits=True, arguments={"refute": request}, random_state=0
        )
        assert report["refute"].status is AssessmentStatus.UNAVAILABLE
        assert report["refute"].detail == reason

    @pytest.mark.parametrize("rule", list(PRE_REFIT_REQUESTS))
    def test_m0_each_request_agrees_unmutated(
        self, pre_refit_fits: dict[str, Any], rule: str
    ) -> None:
        kind, request = PRE_REFIT_REQUESTS[rule]
        assert pre_refit_disagreements(pre_refit_fits[kind], request) == []

    @pytest.mark.parametrize("test", list(PRE_REFIT_RUNS))
    def test_a_request_the_call_admits_runs(
        self, pre_refit_fits: dict[str, Any], test: str
    ) -> None:
        """The nonzero witness: each test runs on a fit that records its identification."""
        result = pre_refit_fits["study"]
        request = {"estimand": "ate", **PRE_REFIT_RUNS[test]}
        assert result.diagnostics._capability_for_arguments("refute", request).available
        assert pre_refit_disagreements(result, PRE_REFIT_RUNS[test]) == []
        report = result.diagnostics.refute(random_state=0, **request)
        assert [record.name for record in report.tests] == [test]

    def test_a_row_that_reads_the_old_four_rules_disagrees(
        self, pre_refit_fits: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The row as it was: the call raised each of these rules and the row read none.

        The call reads the table directly, so replacing the predicate the row reads leaves
        the call as it is.
        """
        old_rules = _REQUEST_RULES[:4]

        def four_rules(result: Any, *, estimand: str, tests: Any, **_: Any) -> str | None:
            request = _RefuteRequest(estimand=estimand, tests=tuple(tests))
            return next(
                (reason for _, rule in old_rules if (reason := rule(result, request))), None
            )

        monkeypatch.setattr(refute_module, "refute_refusal", four_rules)
        for rule, (kind, request) in PRE_REFIT_REQUESTS.items():
            problems = pre_refit_disagreements(pre_refit_fits[kind], request)
            assert problems and problems[0].startswith("the row reads available"), rule
        # The same mutation leaves a request the call admits agreeing.
        run = PRE_REFIT_RUNS["dummy_outcome"]
        assert pre_refit_disagreements(pre_refit_fits["study"], run) == []


# ------------------------------------------------------------------------ the benchmark

#: The fit with the one covariate ``W``, which the sweep found, and a fit with four.
BENCHMARK_KINDS = ("split_plan", "ordinary")


@pytest.fixture(scope="module")
def benchmark_fits() -> dict[str, Any]:
    """One fresh fit of each kind in :data:`BENCHMARK_KINDS`, shared by this section."""
    return {kind: KINDS[kind].build() for kind in BENCHMARK_KINDS}


def _benchmark_raised(result: Any, covariates: list[str]) -> str | None:
    """The sentence ``benchmark`` refuses with, or ``None`` when it reaches a learner fit."""
    return _spied_raised(lambda: benchmark(_spied(result), covariates, random_state=0))


def benchmark_disagreements(result: Any) -> list[str]:
    """Every request whose ``benchmark`` row and module call disagree.

    The bare request names no covariates, so no call stands for it alone. Its row reads
    available exactly when some single covariate runs. The first covariate and every
    covariate are then each one request. A fresh facade resolves each row, so a
    monkeypatched seam takes effect. An empty list is agreement.
    """
    facade = SensitivityFacade(result)
    names = list(result.data.covariate_names)
    problems = []
    bare = facade._capability_for_arguments("benchmark", {})
    singles = {name: _benchmark_raised(result, [name]) for name in names}
    if bare.available != any(raised is None for raised in singles.values()):
        problems.append(f"the bare row reads available={bare.available}, the calls {singles}")
    for covariates in (names[:1], names):
        row = facade._capability_for_arguments("benchmark", {"covariates": covariates})
        problems += _disagreement(covariates, row, _benchmark_raised(result, covariates))
    return problems


class TestTheBenchmarkRowResolvesEachRequest:
    """The row and the call both read :func:`benchmark_refusal`, one request at a time."""

    def test_a_fit_with_one_covariate_reads_unavailable(
        self, benchmark_fits: dict[str, Any]
    ) -> None:
        """No value runs, because the refit cannot drop the only covariate."""
        result = benchmark_fits["split_plan"]
        row = result.sensitivity.capability("benchmark")
        assert row.status is AssessmentStatus.UNAVAILABLE
        assert row.reason == benchmark_refusal(result, ["W"])
        assert row.reason is not None and "['W']" in row.reason
        with pytest.raises(CapabilityError) as error:
            benchmark(result, ["W"])
        assert type(error.value) is CapabilityError
        assert str(error.value) == row.reason
        with pytest.raises(CapabilityError) as error:
            result.sensitivity.benchmark(["W"])
        assert str(error.value) == f"sensitivity 'benchmark' is unavailable: {row.reason}"

    def test_a_combined_report_names_the_refusal(self, benchmark_fits: dict[str, Any]) -> None:
        """Before this row read the predicate, ``assess`` raised ``DataError`` here."""
        result = benchmark_fits["split_plan"]
        report = result.assess(
            include_refits=True,
            arguments={"benchmark": {"covariates": ["W"]}},
            random_state=0,
        )
        item = report.sensitivity["benchmark"]
        assert item.status is AssessmentStatus.UNAVAILABLE
        assert item.detail == result.sensitivity.capability("benchmark").reason

    def test_a_fit_with_more_covariates_keeps_the_declared_argument(
        self, benchmark_fits: dict[str, Any]
    ) -> None:
        result = benchmark_fits["ordinary"]
        row = result.sensitivity.capability("benchmark")
        assert row.available
        assert row.requires_arguments == ("covariates",)
        report = result.sensitivity.run_all(include_refits=True)
        assert report["benchmark"].status is AssessmentStatus.DEFERRED

    def test_naming_every_covariate_reads_unavailable(self, benchmark_fits: dict[str, Any]) -> None:
        result = benchmark_fits["ordinary"]
        every = list(result.data.covariate_names)
        row = result.sensitivity._capability_for_arguments("benchmark", {"covariates": every})
        assert row.status is AssessmentStatus.UNAVAILABLE
        assert row.reason == benchmark_refusal(result, every)
        with pytest.raises(CapabilityError) as error:
            result.sensitivity.benchmark(every, random_state=0)
        assert str(error.value) == row.reason
        report = result.sensitivity.run_all(
            include_refits=True, arguments={"benchmark": {"covariates": every}}, random_state=0
        )
        assert report["benchmark"].status is AssessmentStatus.UNAVAILABLE
        assert report["benchmark"].detail == row.reason
        # The nonzero witness: one covariate fewer runs.
        kept = every[1:]
        assert result.sensitivity._capability_for_arguments(
            "benchmark", {"covariates": kept}
        ).available
        assert benchmark(result, kept, random_state=0).covariates == tuple(kept)

    def test_an_unknown_covariate_is_reported_before_any_refusal(
        self, benchmark_fits: dict[str, Any]
    ) -> None:
        """The malformed argument first, as the omitted-variable calls report it."""
        with pytest.raises(DataError, match=r"unknown covariates \['nope'\]") as error:
            benchmark(benchmark_fits["split_plan"], ["W", "nope"])
        assert not isinstance(error.value, CapabilityError)

    @pytest.mark.parametrize("kind", BENCHMARK_KINDS)
    def test_the_row_leaves_an_unknown_covariate_to_the_call(
        self, benchmark_fits: dict[str, Any], kind: str
    ) -> None:
        """Beside every covariate, an unknown name is malformed before it is refused.

        The row used to read ``unavailable`` with "cannot drop every covariate", while the
        function raised ``DataError``. On the one-covariate fit, the facade read its bare
        row first and raised that refusal too.
        """
        result = benchmark_fits[kind]
        request = [*result.data.covariate_names, "typo"]
        assert benchmark_refusal(result, request) is None
        row = result.sensitivity._capability_for_arguments("benchmark", {"covariates": request})
        assert row.available and row.reason is None
        for call in (
            lambda: benchmark(result, request, random_state=0),
            lambda: result.sensitivity.benchmark(request, random_state=0),
            lambda: result.sensitivity.benchmark(covariates=request, random_state=0),
        ):
            with pytest.raises(DataError, match=r"unknown covariates \['typo'\]") as error:
                call()
            assert not isinstance(error.value, CapabilityError)

    def test_an_unknown_covariate_precedes_the_missing_estimator(
        self, benchmark_fits: dict[str, Any]
    ) -> None:
        """The function checks the malformed name before the estimator it would refit."""
        result = dataclasses.replace(benchmark_fits["ordinary"], estimator=None)
        with pytest.raises(DataError, match=r"unknown covariates \['typo'\]"):
            benchmark(result, ["typo"])
        # The nonzero witness: a known name meets the refusal.
        with pytest.raises(CapabilityError, match="needs the fitted estimator"):
            benchmark(result, ["W1"])

    @pytest.mark.parametrize("kind", BENCHMARK_KINDS)
    @pytest.mark.parametrize("surface", ("run_all", "assess"))
    def test_a_combined_report_checks_unknown_covariates_before_any_operation(
        self, benchmark_fits: dict[str, Any], kind: str, surface: str
    ) -> None:
        result = _spied(benchmark_fits[kind])
        request = {"benchmark": {"covariates": ["typo"]}}
        calls = {
            "run_all": lambda: result.sensitivity.run_all(include_refits=True, arguments=request),
            "assess": lambda: result.assess(include_refits=True, arguments=request),
        }
        with pytest.raises(DataError, match=r"unknown covariates \['typo'\]"):
            calls[surface]()
        assert NeverFit.calls == 0


class TestTheBenchmarkMutationRestoresADisagreement:
    """The agreement check sees a row that stops reading the predicate."""

    @pytest.mark.parametrize("kind", BENCHMARK_KINDS)
    def test_m0_every_kind_agrees_unmutated(
        self, benchmark_fits: dict[str, Any], kind: str
    ) -> None:
        assert benchmark_disagreements(benchmark_fits[kind]) == []

    @pytest.mark.parametrize("kind", BENCHMARK_KINDS)
    def test_a_row_that_ignores_the_predicate_disagrees(
        self, benchmark_fits: dict[str, Any], kind: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """M8: the row as it was, available for every covariate the request names."""
        MUTATIONS["M8"].apply(monkeypatch)
        problems = benchmark_disagreements(benchmark_fits[kind])
        every = list(benchmark_fits[kind].data.covariate_names)
        assert any(problem.startswith(f"{every}: the row reads available") for problem in problems)
        # Only the one-covariate fit has a bare row that no single covariate runs.
        bare = any(problem.startswith("the bare row") for problem in problems)
        assert bare is (kind == "split_plan")


class TestARowItsRequestRefusesCarriesTheRequest:
    """A combined report keeps the invocation of a row that its request refuses.

    Such a row reads available before the request, so the report attaches the request
    with its signature defaults bound, as it does for a call that refused after it was
    invoked. A row refused for the whole fit describes no invocation and carries none.
    Before this rule both kinds carried ``{}``.
    """

    def test_a_refused_axis_carries_the_request(self, truncation_fits: dict[str, Any]) -> None:
        result = truncation_fits["incremental+missing"]
        report = result.diagnostics.run_all(
            include_retargets=True, arguments={"truncation_curve": {"mechanism": False}}
        )
        item = report["truncation_curve"]
        assert item.status is AssessmentStatus.UNAVAILABLE
        assert item.detail == truncation_refusal(result, False)
        assert item.arguments == {"bounds": None, "estimands": None, "mechanism": False}

    def test_a_refused_covariate_set_carries_the_request(
        self, benchmark_fits: dict[str, Any]
    ) -> None:
        result = benchmark_fits["ordinary"]
        every = list(result.data.covariate_names)
        report = result.sensitivity.run_all(
            include_refits=True, arguments={"benchmark": {"covariates": every}}, random_state=0
        )
        item = report["benchmark"]
        assert item.status is AssessmentStatus.UNAVAILABLE
        assert item.detail == benchmark_refusal(result, every)
        assert item.arguments == {
            "covariates": every,
            "estimand": "ate",
            "nu2_estimator": "auto",
            "random_state": 0,
        }

    def test_a_row_refused_for_the_whole_fit_carries_nothing(
        self, truncation_fits: dict[str, Any]
    ) -> None:
        """The control: the tilt rows of a complete-outcome fit are refused fit-wide."""
        result = truncation_fits["ordinary"]
        report = result.sensitivity.run_all(
            include_retargets=True, arguments={"missingness": {"gamma": (0.5,)}}
        )
        item = report["missingness"]
        assert item.status is AssessmentStatus.NOT_APPLICABLE
        assert item.arguments == {}


# ------------------------------------------------------------ the simulated confounding

#: The zero-strength anchor alone. The call calibrates and reads the anchor, and it refits
#: no cell, so a request that runs costs no learner fit.
ANCHOR = ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,))

#: Each study fit, and its requests with whether the call runs them. A study fit records
#: its identification, which every estimator-fitted kind of the sweep lacks, so these are
#: fits whose row no fit-wide rule refuses. ``categorical`` adjusts for ``W`` and the
#: encoded category ``V``. ``constant`` adjusts for ``W4``, which is constant where the
#: weights are positive. ``natural_course`` reports the complete-outcome ``ey_obs`` alone.
#: ``policy_means`` reports a zero-delta and a nonzero policy mean. Each request that runs
#: is the nonzero witness of its fit, and ``natural_course`` has none.
CONFOUNDING_REQUESTS: dict[str, tuple[tuple[dict[str, Any], bool], ...]] = {
    "categorical": (
        ({"benchmark_covariates": ("V",)}, False),
        ({"benchmark_covariates": ("V__small",)}, False),
        ({"benchmark_covariates": ("W",)}, True),
    ),
    "constant": (
        ({"benchmark_covariates": ("W4",)}, False),
        ({"benchmark_covariates": ("W1",)}, True),
    ),
    "natural_course": (({}, False), ({"estimand": "ey_obs"}, False)),
    "policy_means": (
        ({"estimand": "ey_shift[natural course]"}, False),
        ({"estimand": "ey_shift[up half]"}, True),
    ),
}

#: The fit and the covariate of each refused calibration.
REFUSED_COVARIATES = (("categorical", "V"), ("categorical", "V__small"), ("constant", "W4"))


@pytest.fixture(scope="module")
def confounding_fits() -> dict[str, Any]:
    """One fresh fit of each kind in :data:`CONFOUNDING_REQUESTS`, shared by this section."""
    return {
        "categorical": dataclasses.replace(_estimate(confounding_study(), ATE())),
        "constant": _fit_with_a_support_constant_covariate(),
        "natural_course": KINDS["natural_course_study"].build(),
        "policy_means": KINDS["policy_means"].build(),
    }


def _at_anchor(result: Any, request: dict[str, Any]) -> Callable[[], Any]:
    """The ``simulated_confounding`` call of ``request`` at the anchor alone."""
    return functools.partial(simulated_confounding, result, grid=ANCHOR, random_state=0, **request)


def confounding_disagreements(result: Any, kind: str) -> list[str]:
    """Every request of ``kind`` whose ``simulated_confounding`` row and call disagree.

    A fresh facade resolves each row, so a monkeypatched seam takes effect. An empty list
    is agreement.
    """
    facade = SensitivityFacade(result)
    problems = []
    for request, _ in CONFOUNDING_REQUESTS[kind]:
        arguments = {"grid": ANCHOR, **request}
        row = facade._capability_for_arguments("simulated_confounding", arguments)
        problems += _disagreement(request, row, _raised(_at_anchor(result, request)))
    return problems


class TestTheSimulatedConfoundingRowResolvesEachRequest:
    """The row and the call both read :func:`simulated_confounding_refusal`."""

    @pytest.mark.parametrize("kind", list(CONFOUNDING_REQUESTS))
    def test_each_request_runs_as_the_table_says(
        self, confounding_fits: dict[str, Any], kind: str
    ) -> None:
        result = confounding_fits[kind]
        for request, runs in CONFOUNDING_REQUESTS[kind]:
            assert (_raised(_at_anchor(result, request)) is None) is runs, request

    @pytest.mark.parametrize(("kind", "name"), REFUSED_COVARIATES)
    def test_a_refused_covariate_reads_unavailable(
        self, confounding_fits: dict[str, Any], kind: str, name: str
    ) -> None:
        """The request reads the call's sentence, and the bare row stays available."""
        result = confounding_fits[kind]
        request = {"benchmark_covariates": (name,)}
        row = result.sensitivity._capability_for_arguments(
            "simulated_confounding", {"grid": ANCHOR, **request}
        )
        assert row.status is AssessmentStatus.UNAVAILABLE
        assert row.reason == simulated_confounding_refusal(result, "ate", (name,))
        assert row.reason is not None and f"{kind} covariate {name!r}" in row.reason
        with pytest.raises(CapabilityError) as error:
            simulated_confounding(result, grid=ANCHOR, **request)
        assert type(error.value) is CapabilityError
        assert str(error.value) == row.reason
        assert result.sensitivity.capability("simulated_confounding").available
        with pytest.raises(CapabilityError) as error:
            result.sensitivity.simulated_confounding(grid=ANCHOR, **request)
        assert str(error.value) == row.reason

    @pytest.mark.parametrize(("kind", "name"), REFUSED_COVARIATES)
    def test_a_combined_report_names_the_refusal(
        self, confounding_fits: dict[str, Any], kind: str, name: str
    ) -> None:
        """Before this row read the predicate, the report published a declined row here."""
        result = dataclasses.replace(confounding_fits[kind])
        report = result.assess(
            include_refits=True,
            arguments={
                "refute": {"n_replicates": 1},
                "simulated_confounding": {"grid": ANCHOR, "benchmark_covariates": (name,)},
            },
            random_state=0,
        )
        item = report.sensitivity["simulated_confounding"]
        assert item.status is AssessmentStatus.UNAVAILABLE
        assert item.detail == simulated_confounding_refusal(result, "ate", (name,))
        assert DECLINED not in item.detail

    @pytest.mark.parametrize(("kind", "name"), [("categorical", "W"), ("constant", "W1")])
    def test_a_covariate_that_calibrates_runs(
        self, confounding_fits: dict[str, Any], kind: str, name: str
    ) -> None:
        """The nonzero witness: a numeric covariate beside the refused one calibrates."""
        result = confounding_fits[kind]
        request = {"benchmark_covariates": (name,)}
        assert result.sensitivity._capability_for_arguments(
            "simulated_confounding", {"grid": ANCHOR, **request}
        ).available
        surface = simulated_confounding(result, grid=ANCHOR, random_state=0, **request)
        assert {row.covariate for row in surface.calibrations} == {name}

    def test_a_malformed_name_is_reported_before_the_refusal(
        self, confounding_fits: dict[str, Any]
    ) -> None:
        """A duplicate, non-string or unknown name is not a refusal, and it comes first."""
        result = confounding_fits["categorical"]
        malformed: tuple[tuple[tuple[Any, ...], type[Exception], str], ...] = (
            (("V", "V"), ValueError, "contains duplicates"),
            (("V", 1), TypeError, "only column names"),
            (("V", "nope"), ValueError, "'nope' is unavailable"),
        )
        for covariates, error_type, message in malformed:
            with pytest.raises(error_type, match=message) as error:
                simulated_confounding(result, grid=ANCHOR, benchmark_covariates=covariates)
            assert not isinstance(error.value, CapabilityError)
            assert simulated_confounding_refusal(result, "ate", covariates) is None

    def test_the_bare_natural_course_row_reads_unavailable(
        self, confounding_fits: dict[str, Any]
    ) -> None:
        """No reported parameter runs, so the bare row quotes the default's sentence."""
        result = dataclasses.replace(confounding_fits["natural_course"])
        row = result.sensitivity.capability("simulated_confounding")
        assert row.status is AssessmentStatus.UNAVAILABLE
        assert row.reason == simulated_confounding_refusal(result, "ate")
        assert row.reason == simulated_confounding_refusal(result, "ey_obs")
        assert row.reason is not None and "refuses NaturalCourseMean" in row.reason
        with pytest.raises(CapabilityError) as error:
            result.sensitivity.simulated_confounding(grid=ANCHOR)
        assert (
            str(error.value) == f"sensitivity 'simulated_confounding' is unavailable: {row.reason}"
        )
        report = result.sensitivity.run_all(
            include_refits=True, arguments={"simulated_confounding": {"grid": ANCHOR}}
        )
        assert report["simulated_confounding"].status is AssessmentStatus.UNAVAILABLE
        assert report["simulated_confounding"].detail == row.reason

    def test_a_zero_delta_policy_mean_reads_unavailable(
        self, confounding_fits: dict[str, Any]
    ) -> None:
        """The bare row keeps its declared arguments, and the nonzero mean runs."""
        result = confounding_fits["policy_means"]
        bare = result.sensitivity.capability("simulated_confounding")
        assert bare.available
        assert bare.requires_arguments == ("grid", "estimand")
        zero = {"grid": ANCHOR, "estimand": "ey_shift[natural course]"}
        row = result.sensitivity._capability_for_arguments("simulated_confounding", zero)
        assert row.status is AssessmentStatus.UNAVAILABLE
        assert row.reason == simulated_confounding_refusal(result, zero["estimand"])
        assert row.reason is not None and "zero-delta policy" in row.reason
        up = {"grid": ANCHOR, "estimand": "ey_shift[up half]"}
        assert result.sensitivity._capability_for_arguments("simulated_confounding", up).available


class TestTheSimulatedConfoundingMutationRestoresADisagreement:
    """The agreement check sees a row that stops reading the predicate."""

    @pytest.mark.parametrize("kind", list(CONFOUNDING_REQUESTS))
    def test_m0_every_kind_agrees_unmutated(
        self, confounding_fits: dict[str, Any], kind: str
    ) -> None:
        assert confounding_disagreements(confounding_fits[kind], kind) == []

    @pytest.mark.parametrize("kind", list(CONFOUNDING_REQUESTS))
    def test_a_row_that_ignores_the_predicate_disagrees(
        self, confounding_fits: dict[str, Any], kind: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """M9: the row as it was, available for every request the fit-wide rules admit."""
        MUTATIONS["M9"].apply(monkeypatch)
        problems = confounding_disagreements(confounding_fits[kind], kind)
        refused = [request for request, runs in CONFOUNDING_REQUESTS[kind] if not runs]
        assert len(problems) == len(refused)
        for request, problem in zip(refused, problems, strict=True):
            assert problem.startswith(f"{request}: the row reads available and the call raised")


# ------------------------------------------------------------ the omitted-variable bound

#: The entry point of each omitted-variable row, called as a user calls it, at the
#: signature default ``estimand="ate"``. ``benchmark`` drops the first covariate.
BOUND_CALLS: dict[str, Callable[[Any], Any]] = {
    "omitted_confounding": omitted_variable_bounds,
    "robustness_value": robustness_value,
    "elements": sensitivity_elements,
    "benchmark": lambda result: benchmark(
        result, list(result.data.covariate_names[:1]), random_state=0
    ),
    "contour": lambda result: contour_data(result, grid_size=3),
}

#: Each arm-indexed fit that reports no mean and no linear contrast, and whether its
#: refusal carries the E-value pointer. The complete-outcome natural-course mean is the
#: sweep kind that found the defect. The attributable risk and fraction keep their strata.
#: The ratio-only fit is the one whose reported parameters the E-value answers for.
EMPTY_BOUND_FITS: dict[str, bool] = {
    "natural_course_study": False,
    "par": False,
    "paf": False,
    "ratio_only": True,
}


@pytest.fixture(scope="module")
def bound_fits() -> dict[str, Any]:
    """One fresh fit of each kind in :data:`EMPTY_BOUND_FITS`, and the ordinary fit."""
    frame, _ = make_binary_outcome(n=400, seed=17)
    ratio_only = TMLE(**linear_in_sample(estimands=("rr", "or")))
    return {
        "natural_course_study": KINDS["natural_course_study"].build(),
        "par": dataclasses.replace(_fit_attributable("par")),
        "paf": dataclasses.replace(_fit_attributable("paf")),
        "ratio_only": ratio_only.fit(frame, outcome="Y", treatment="A").single(),
        "ordinary": KINDS["ordinary"].build(),
    }


def bound_disagreements(result: Any) -> list[str]:
    """Every omitted-variable row of ``result`` that disagrees with its module call.

    A fresh facade reads the rows, so a monkeypatched rule table takes effect. An empty
    list is agreement.
    """
    facade = SensitivityFacade(result)
    problems = []
    for operation, call in BOUND_CALLS.items():
        raised = _raised(functools.partial(call, result))
        problems += _disagreement(operation, facade.capability(operation), raised)
    return problems


class TestTheBoundRowsReadTheCallsTable:
    """The five rows and the five entry points answer from :func:`fit_wide_bound_refusal`."""

    def test_the_calls_cover_every_row(self) -> None:
        assert set(BOUND_CALLS) == set(OMITTED_VARIABLE_OPERATIONS)

    @pytest.mark.parametrize("kind", list(EMPTY_BOUND_FITS))
    def test_a_fit_with_no_parameter_to_bound_reads_unavailable(
        self, bound_fits: dict[str, Any], kind: str
    ) -> None:
        """Every row quotes the sentence that every call raises."""
        result = bound_fits[kind]
        assert result.config.parameter_axis == "arm"
        assert not result.data.has_missing_outcome
        reason = fit_wide_bound_refusal(result)
        assert reason is not None
        assert reason == dict(_FIT_WIDE_BOUND_RULES)["bound_parameters"](result)
        assert f"it reported {sorted(result.estimates)}" in reason
        assert reason.endswith(_EVALUE_POINTER) is EMPTY_BOUND_FITS[kind]
        for operation, call in BOUND_CALLS.items():
            row = result.sensitivity.capability(operation)
            assert row.status is AssessmentStatus.UNAVAILABLE, operation
            assert row.reason == reason, operation
            with pytest.raises(CapabilityError) as error:
                call(result)
            assert type(error.value) is CapabilityError
            assert str(error.value) == reason, operation

    @pytest.mark.parametrize("kind", list(EMPTY_BOUND_FITS))
    def test_resolving_a_parameter_directly_raises_the_same_sentence(
        self, bound_fits: dict[str, Any], kind: str
    ) -> None:
        """``resolve_parameter`` reads the rule itself, so no second sentence exists."""
        result = bound_fits[kind]
        with pytest.raises(CapabilityError) as error:
            resolve_parameter(result, "ate")
        assert str(error.value) == fit_wide_bound_refusal(result)

    def test_the_facade_and_a_combined_report_name_the_refusal(
        self, bound_fits: dict[str, Any]
    ) -> None:
        """Before this rule, the report published five declined rows on this fit."""
        result = dataclasses.replace(bound_fits["natural_course_study"])
        reason = fit_wide_bound_refusal(result)
        with pytest.raises(CapabilityError) as error:
            result.sensitivity.omitted_confounding()
        assert str(error.value) == f"sensitivity 'omitted_confounding' is unavailable: {reason}"
        covariates = list(result.data.covariate_names[:1])
        report = result.sensitivity.run_all(
            include_refits=True, arguments={"benchmark": {"covariates": covariates}}
        )
        for operation in OMITTED_VARIABLE_OPERATIONS:
            assert report[operation].status is AssessmentStatus.UNAVAILABLE, operation
            assert report[operation].detail == reason, operation

    def test_an_ordinary_fit_still_answers_every_row(self, bound_fits: dict[str, Any]) -> None:
        """The nonzero witness: the rule admits a fit that reports a linear contrast."""
        result = bound_fits["ordinary"]
        assert dict(_FIT_WIDE_BOUND_RULES)["bound_parameters"](result) is None
        assert fit_wide_bound_refusal(result) is None
        for operation, call in BOUND_CALLS.items():
            assert result.sensitivity.capability(operation).available, operation
            assert call(result) is not None, operation


class TestTheBoundMutationRestoresADisagreement:
    """The agreement check sees a rule table that no longer refuses these fits."""

    @pytest.mark.parametrize("kind", [*EMPTY_BOUND_FITS, "ordinary"])
    def test_m0_every_kind_agrees_unmutated(self, bound_fits: dict[str, Any], kind: str) -> None:
        assert bound_disagreements(bound_fits[kind]) == []

    @pytest.mark.parametrize("kind", [*EMPTY_BOUND_FITS, "ordinary"])
    def test_a_table_without_the_rule_disagrees(
        self, bound_fits: dict[str, Any], kind: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """M10: every row reads available, and every call still refuses."""
        MUTATIONS["M10"].apply(monkeypatch)
        problems = bound_disagreements(bound_fits[kind])
        if kind == "ordinary":
            assert problems == []
            return
        assert [problem.split(":", 1)[0] for problem in problems] == list(BOUND_CALLS)
        for problem in problems:
            assert "the row reads available and the call raised" in problem


# ------------------------------------------------------------ the covariate a refit adds


@pytest.fixture(scope="module")
def ordered_fit() -> Any:
    """The ordered collaborative fit with the explicit ordering :data:`INSTRUMENT_ORDERING`."""
    return KINDS["ctmle_ordered"].build()


def _prepend_added(self: CTMLE, data: Any) -> CTMLE:
    """The mutation: an added covariate ranked first rather than last."""
    added = tuple(name for name in data.covariate_names if name not in self.ordering)
    configured = copy.copy(self)
    configured.ordering = (*added, *self.ordering)
    return configured


def ordering_problems(result: Any) -> list[str]:
    """Every way the ``random_common_cause`` refit breaks the ordering contract.

    The contract is that the refit ranks the declared ordering first and the added noise
    column last, and leaves the estimator's own ordering alone. The refute value is
    checked against an independent fit that declares that order, on the same noise draw.
    """
    report = refute(result, tests=("random_common_cause",), n_replicates=1, random_state=0)
    seed = report.random_state
    noise = np.random.default_rng(seed).normal(size=result.data.n)
    noisy = result.data.with_extra_covariate(noise, "_noise_0")
    expected = (*INSTRUMENT_ORDERING, "_noise_0")
    problems = []
    path = result.estimator.refit(noisy, random_state=seed).extra["ctmle"].path
    if path[-1] != expected:
        problems.append(f"the refit path ends with {path[-1]}")
    manual = ctmle_ordered(expected).refit(noisy, random_state=seed)["ate"].psi
    if report["random_common_cause"].values != (manual,):
        problems.append("the refute value is not the fit that orders the noise last")
    if tuple(result.estimator.ordering) != INSTRUMENT_ORDERING:
        problems.append(f"the estimator's ordering became {result.estimator.ordering}")
    return problems


class TestAnAddedCovariateGoesAfterTheDeclaredOrdering:
    """``CTMLE._configured_for_refit`` places a covariate a refit adds after the ordering."""

    def test_the_refit_orders_the_noise_last(self, ordered_fit: Any) -> None:
        assert ordering_problems(ordered_fit) == []

    def test_a_combined_report_runs_the_refutation(self, ordered_fit: Any) -> None:
        """RM23 saw ``assess(include_refits=True)`` raise ``ValueError`` on this fit."""
        report = dataclasses.replace(ordered_fit).assess(
            include_refits=True, arguments={"refute": {"n_replicates": 1}}, random_state=0
        )
        item = report.diagnostics["refute"]
        assert item.report is not None, item.detail
        assert "random_common_cause" in [test.name for test in item.report.tests]

    def test_a_mutation_that_ranks_the_noise_first_is_detected(
        self, ordered_fit: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(CTMLE, "_configured_for_refit", _prepend_added)
        problems = ordering_problems(ordered_fit)
        assert problems[0] == f"the refit path ends with {('_noise_0', *INSTRUMENT_ORDERING)}"
        assert "the refute value is not the fit that orders the noise last" in problems

    def test_an_estimator_that_ignores_the_added_covariate_is_refused(
        self, ordered_fit: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """M4: without the hook the refit meets the ordering's coverage refusal."""
        MUTATIONS["M4"].apply(monkeypatch)
        with pytest.raises(ValueError, match="ordering must cover every covariate"):
            dataclasses.replace(ordered_fit).assess(
                include_refits=True, arguments={"refute": {"n_replicates": 1}}, random_state=0
            )


#: The name the ``random_common_cause`` refit gives its noise covariate.
NOISE = "_noise_0"


def _noise(n: int) -> Any:
    """A standard normal column. A constant one would be dropped as a covariate."""
    return np.random.default_rng(1).standard_normal(n)


@pytest.fixture(scope="module")
def drtmle_fits() -> dict[str, Any]:
    """The DR-TMLE fit under each kind of companion, and the same fit without one.

    ``frame`` is the sweep's companion frame. ``prepared`` is the same rows as a
    :class:`~cleverly.data.CausalData`, which names the fit's covariates alone. ``extended``
    is the frame with a column for the covariate the refit adds, so it follows the refit.
    """
    frame = drtmle_companion_frame()
    return {
        "frame": KINDS["drtmle_companion"].build(),
        "prepared": fit_drtmle(CausalData.from_frame(frame, outcome="Y", treatment="A")),
        "extended": fit_drtmle(frame.assign(**{NOISE: _noise(len(frame))})),
        "plain": KINDS["drtmle"].build(),
    }


def _noise_refit_values(result: Any) -> tuple[float, ...]:
    """The single ``random_common_cause`` value of ``result``, at seed 0."""
    report = refute(result, tests=("random_common_cause",), n_replicates=1, random_state=0)
    return report["random_common_cause"].values


def _noisy(result: Any) -> Any:
    """The data of ``result`` with the covariate the ``random_common_cause`` refit adds."""
    return result.data.with_extra_covariate(_noise(result.data.n), NOISE)


class TestARefitDropsACompanionThatLacksACovariate:
    """``DRTMLE._configured_for_refit`` refits without a companion that cannot follow."""

    @pytest.mark.parametrize("companion", ["frame", "prepared", "extended"])
    def test_the_refute_draws_equal_the_fit_without_a_companion(
        self, drtmle_fits: dict[str, Any], companion: str
    ) -> None:
        paired = drtmle_fits[companion]
        evaluation = paired.estimator.evaluation
        values = _noise_refit_values(paired)
        # Bit for bit: the companion enters no fit, fold or score.
        assert values == _noise_refit_values(drtmle_fits["plain"])
        # The nonzero witness: the refit moved the estimate, so the equality has content.
        assert values[0] != paired["ate"].psi
        # The fit keeps its own companion.
        assert paired.estimator.evaluation is evaluation

    @pytest.mark.parametrize("companion", ["frame", "prepared"])
    def test_a_companion_without_the_added_covariate_is_dropped(
        self, drtmle_fits: dict[str, Any], companion: str
    ) -> None:
        estimator = drtmle_fits[companion].estimator
        configured = estimator._configured_for_refit(_noisy(drtmle_fits[companion]))
        assert configured is not estimator
        assert configured.evaluation is None
        # The companion still follows a refit on the fit's own covariates.
        assert estimator._configured_for_refit(drtmle_fits[companion].data) is estimator

    def test_a_frame_that_holds_the_added_covariate_is_kept(
        self, drtmle_fits: dict[str, Any]
    ) -> None:
        estimator = drtmle_fits["extended"].estimator
        assert estimator._configured_for_refit(_noisy(drtmle_fits["extended"])) is estimator

    def test_an_estimator_that_keeps_its_companion_is_refused(
        self, drtmle_fits: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """M5: without the hook neither the frame nor the prepared companion holds the noise."""
        MUTATIONS["M5"].apply(monkeypatch)
        with pytest.raises(DataError, match=NOISE):
            _noise_refit_values(drtmle_fits["frame"])
        with pytest.raises(ValueError, match=r"the companion carries covariates .* the fit"):
            _noise_refit_values(drtmle_fits["prepared"])
        # The same mutation leaves the fit without a companion running, and the companion
        # that holds the column too.
        assert _noise_refit_values(drtmle_fits["plain"])
        assert _noise_refit_values(drtmle_fits["extended"])


# ---------------------------------------------------------------------- the replay slots


#: Each reconfigured result, and whether its refit runs. A refused refit is the RM23 defect:
#: before RM23 each of these read ``refit_nuisances`` true and its refit raised.
REPLAY_KINDS: dict[str, tuple[Callable[[], Any], bool]] = {
    "stratify=treatment": (reconfigured_stratified, False),
    "stratify=treatment+outcome": (
        lambda: reconfigured(cross_fitted(), stratify_folds="treatment+outcome"),
        False,
    ),
    "n_folds=1": (lambda: reconfigured(cross_fitted(), n_folds=1), False),
    "repeats=0": (lambda: reconfigured(cross_fitted(), repeats=0), False),
    "repeats=2 in sample": (lambda: reconfigured(discrete_fit(), repeats=2), False),
    "plan without provenance": (without_provenance, False),
    "ctmle greedy stratified": (ctmle_stratified, False),
    "ctmle oat clustered": (reconfigured_ctmle_clustered, False),
    "unbounded scale": (unbounded_scale_ate, False),
    "declared cross-fitted": (lambda: loads(dumps(cross_fitted())), True),
    "declared in sample": (lambda: loads(dumps(discrete_fit())), True),
}

#: The reconfigured kinds whose refit this version refuses.
REFUSED_REPLAY_KINDS = [kind for kind, (_, runs) in REPLAY_KINDS.items() if not runs]


@pytest.fixture(scope="module")
def replay_results() -> dict[str, Any]:
    """One result of each kind in :data:`REPLAY_KINDS`."""
    return {kind: build() for kind, (build, _) in REPLAY_KINDS.items()}


class TestTheRefitSlotReadsTheRefitPreflight:
    """``refit_nuisances`` reads false exactly when ``refit()`` meets a preflight refusal."""

    @pytest.mark.parametrize("kind", list(REPLAY_KINDS))
    def test_each_slot_agrees_with_its_call(
        self, replay_results: dict[str, Any], kind: str
    ) -> None:
        result = replay_results[kind]
        assert_replay_agrees(result, ("ate",))
        replay = replayability(result)
        # The nonzero witness: the cached nuisances retarget, and the refit slot is the
        # one the table expects, so agreement is not two refusals or two passes alone.
        assert replay.retarget_cached_nuisances
        assert replay.refit_nuisances is REPLAY_KINDS[kind][1]
        if not replay.refit_nuisances:
            assert replay.unreconstructible == (POINT_REPLAY_REFIT_CONFIGURATION,)

    @pytest.mark.parametrize("kind", REFUSED_REPLAY_KINDS)
    def test_a_combined_report_names_the_refused_refit(
        self, replay_results: dict[str, Any], kind: str
    ) -> None:
        """Before RM23 the collaborative kind raised ``ValueError`` from ``assess``."""
        result = replay_results[kind]
        report = result.assess(
            include_refits=True,
            include_retargets=True,
            arguments={"refute": {"n_replicates": 1}},
            random_state=0,
        )
        item = report.diagnostics["refute"]
        assert item.status is AssessmentStatus.UNAVAILABLE
        assert "refitting the nuisance models is unavailable" in item.detail
        assert POINT_REPLAY_REFIT_CONFIGURATION in item.detail

    def test_a_reconfigured_stratified_refit_raises_the_slot_sentence(
        self, replay_results: dict[str, Any]
    ) -> None:
        result = replay_results["stratify=treatment"]
        with pytest.raises(CapabilityError) as raised:
            result.estimator.refit(result.data)
        assert type(raised.value) is CapabilityError
        assert str(raised.value) == result.estimator._refit_configuration_refusal(result.data)


class TestEachReplayMutationRestoresADisagreement:
    """The agreement check sees a slot that ignores the preflight or a lost default."""

    @pytest.mark.parametrize("kind", REFUSED_REPLAY_KINDS)
    def test_a_slot_that_ignores_the_preflight_disagrees(
        self, replay_results: dict[str, Any], kind: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """M6: the slot as it was, true while the refit raised."""
        MUTATIONS["M6"].apply(monkeypatch)
        problems = replay_disagreements(replay_results[kind], ("ate",))
        assert len(problems) == 1
        assert problems[0].startswith("refit_nuisances reads True, and the call gave Capab")

    def test_the_same_mutation_leaves_a_declared_result_agreeing(
        self, replay_results: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        MUTATIONS["M6"].apply(monkeypatch)
        assert replay_disagreements(replay_results["declared cross-fitted"], ("ate",)) == []
