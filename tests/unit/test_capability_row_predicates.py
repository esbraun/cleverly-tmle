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
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from cleverly import ATE
from cleverly.assessment import (
    POINT_REPLAY_REFIT_CONFIGURATION,
    AssessmentStatus,
    DiagnosticsFacade,
    SensitivityFacade,
    replayability,
)
from cleverly.estimators import CTMLE
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
    _RESPONSE_BOUND_REFUSAL,
    _RESPONSE_TILT_POINTER,
    benchmark,
    benchmark_refusal,
    fit_wide_bound_refusal,
)
from cleverly.sensitivity.positivity import (
    _TRUNCATION_RULES,
    truncation_axis,
    truncation_curve,
    truncation_refusal,
)
from cleverly.targets.population_intervention import NATURAL_COURSE_TILT_REFUSAL
from cleverly.validation.refute import _REQUEST_RULES, DEFAULT_TESTS, refute, refute_refusal
from tests.unit._capability_sweep_support import (
    DECLINED,
    INSTRUMENT_ORDERING,
    KINDS,
    MUTATIONS,
    as_saved_by_v011,
    assert_replay_agrees,
    cross_fitted,
    ctmle_ordered,
    ctmle_stratified,
    discrete_fit,
    fit_drtmle,
    replay_disagreements,
    restored,
    restored_stratified,
    unbounded_scale,
    without_provenance,
)
from tests.unit._confounding_support import confounding_study
from tests.unit._natural_course_support import NeverFit, never_fit_learners
from tests.unit._simulated_confounding_support import _estimate
from tests.unit.test_simulated_confounding import _fit_with_a_support_constant_covariate
from tests.unit.test_simulated_confounding_attributable import _fit_attributable

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

#: The kinds whose tilt rows read ``available`` before RM23 and whose calls refused.
DECLINED_TILT_KINDS = ("shift+missing", "incremental+missing", "regime+missing")
DECLINED_TILT_KINDS += ("msm+missing", "rr+missing")

TILT_CALLS = {"missingness": missingness_tilt, "tipping_gamma": tipping_gamma}


@pytest.fixture(scope="module")
def tilt_fits() -> dict[str, Any]:
    """One fresh fit of each kind in :data:`TILT_RULE_OF`, shared by this section."""
    return {kind: KINDS[kind].build() for kind in TILT_RULE_OF}


def _raised(call: Any, result: Any) -> str | None:
    """The sentence ``call(result)`` refuses with, or ``None`` when it runs."""
    try:
        call(result)
    except CapabilityError as error:
        return str(error)
    return None


def tilt_disagreements(result: Any) -> list[str]:
    """Every way the two tilt rows and the bound's pointer disagree with the calls.

    A fresh facade reads the rows, so a monkeypatched seam takes effect even when the
    result already memoized its own facade. An empty list is agreement.
    """
    facade = SensitivityFacade(result)
    problems = []
    for operation, call in TILT_CALLS.items():
        row = facade.capability(operation)
        raised = _raised(call, result)
        if row.available and raised is not None:
            problems.append(f"{operation}: the row reads available and the call raised {raised}")
        if not row.available and raised != row.reason:
            problems.append(f"{operation}: the row quotes {row.reason!r}, the call {raised!r}")
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
        raised = _raised(TILT_CALLS[operation], result)
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


def _module_curve(result: Any, arguments: dict[str, Any]) -> Any:
    """The module call on one bound, which is enough to reach every refusal."""
    return truncation_curve(result, [0.05], **arguments)


def truncation_disagreements(result: Any) -> list[str]:
    """Every request whose truncation row and module call disagree.

    A fresh facade resolves each row, so a monkeypatched seam takes effect even when the
    result already memoized its own facade. An empty list is agreement.
    """
    facade = DiagnosticsFacade(result)
    problems = []
    for arguments in TRUNCATION_REQUESTS:
        row = facade._capability_for_arguments("truncation_curve", arguments)
        raised = _raised(lambda fitted, request=arguments: _module_curve(fitted, request), result)
        if row.available and raised is not None:
            problems.append(f"{arguments}: the row reads available and the call raised {raised}")
        if not row.available and raised != row.reason:
            problems.append(f"{arguments}: the row quotes {row.reason!r}, the call {raised!r}")
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


def _refute_raised(result: Any, arguments: dict[str, Any]) -> str | None:
    """The sentence ``refute`` refuses with, or ``None`` when it reaches a learner fit."""
    try:
        refute(_spied(result), n_replicates=1, **arguments)
    except CapabilityError as error:
        return str(error)
    except AssertionError as error:
        assert "before any learner is fitted" in str(error)
        return None
    raise AssertionError("the spied refute neither refused nor reached a learner fit")


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
        raised = _refute_raised(result, arguments)
        if row.available and raised is not None:
            problems.append(f"{arguments}: the row reads available and the call raised {raised}")
        if not row.available and raised != row.reason:
            problems.append(f"{arguments}: the row quotes {row.reason!r}, the call {raised!r}")
    return problems


class TestTheRefuteRowResolvesEachRequest:
    """The row and the call both read :func:`refute_refusal`, one request at a time."""

    def test_the_rule_table_is_ordered(self) -> None:
        assert [name for name, _ in _REQUEST_RULES] == [
            "estimator",
            "estimand",
            "placebo_level",
            "row_set_under_plan",
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


# ------------------------------------------------------------------------ the benchmark

#: The fit with the one covariate ``W``, which the sweep found, and a fit with four.
BENCHMARK_KINDS = ("split_plan", "ordinary")


@pytest.fixture(scope="module")
def benchmark_fits() -> dict[str, Any]:
    """One fresh fit of each kind in :data:`BENCHMARK_KINDS`, shared by this section."""
    return {kind: KINDS[kind].build() for kind in BENCHMARK_KINDS}


def _benchmark_raised(result: Any, covariates: list[str]) -> str | None:
    """The sentence ``benchmark`` refuses with, or ``None`` when it reaches a learner fit."""
    try:
        benchmark(_spied(result), covariates, random_state=0)
    except CapabilityError as error:
        return str(error)
    except AssertionError as error:
        assert "before any learner is fitted" in str(error)
        return None
    raise AssertionError("the spied benchmark neither refused nor reached a learner fit")


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
        raised = _benchmark_raised(result, covariates)
        if row.available and raised is not None:
            problems.append(f"{covariates}: the row reads available and the call raised {raised}")
        if not row.available and raised != row.reason:
            problems.append(f"{covariates}: the row quotes {row.reason!r}, the call {raised!r}")
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
        "natural_course": _fit_attributable("ey_obs", family="gaussian", strata=False),
        "policy_means": KINDS["policy_means"].build(),
    }


def _confounding_raised(result: Any, request: dict[str, Any]) -> str | None:
    """The sentence ``simulated_confounding`` refuses with at the anchor, or ``None``."""
    try:
        simulated_confounding(result, grid=ANCHOR, random_state=0, **request)
    except CapabilityError as error:
        return str(error)
    return None


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
        raised = _confounding_raised(result, request)
        if row.available and raised is not None:
            problems.append(f"{request}: the row reads available and the call raised {raised}")
        if not row.available and raised != row.reason:
            problems.append(f"{request}: the row quotes {row.reason!r}, the call {raised!r}")
    return problems


class TestTheSimulatedConfoundingRowResolvesEachRequest:
    """The row and the call both read :func:`simulated_confounding_refusal`."""

    @pytest.mark.parametrize("kind", list(CONFOUNDING_REQUESTS))
    def test_each_request_runs_as_the_table_says(
        self, confounding_fits: dict[str, Any], kind: str
    ) -> None:
        result = confounding_fits[kind]
        for request, runs in CONFOUNDING_REQUESTS[kind]:
            assert (_confounding_raised(result, request) is None) is runs, request

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


@pytest.fixture(scope="module")
def drtmle_pair() -> tuple[Any, Any]:
    """The DR-TMLE fit with an ``evaluation=`` companion, and the same fit without one."""
    return KINDS["drtmle_companion"].build(), fit_drtmle(companion=False)


def _noise_refit_values(result: Any) -> tuple[float, ...]:
    """The single ``random_common_cause`` value of ``result``, at seed 0."""
    report = refute(result, tests=("random_common_cause",), n_replicates=1, random_state=0)
    return report["random_common_cause"].values


class TestARefitDropsACompanionThatLacksACovariate:
    """``DRTMLE._configured_for_refit`` refits without a companion that cannot follow."""

    def test_the_refute_draws_equal_the_fit_without_a_companion(
        self, drtmle_pair: tuple[Any, Any]
    ) -> None:
        paired, plain = drtmle_pair
        evaluation = paired.estimator.evaluation
        values = _noise_refit_values(paired)
        # Bit for bit: the companion enters no fit, fold or score.
        assert values == _noise_refit_values(plain)
        # The nonzero witness: the refit moved the estimate, so the equality has content.
        assert values[0] != paired["ate"].psi
        # The fit keeps its own companion.
        assert paired.estimator.evaluation is evaluation

    def test_an_estimator_that_keeps_its_companion_is_refused(
        self, drtmle_pair: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """M5: without the hook the companion lacks ``_noise_0``."""
        MUTATIONS["M5"].apply(monkeypatch)
        paired, plain = drtmle_pair
        with pytest.raises(DataError, match="_noise_0"):
            _noise_refit_values(paired)
        # The same mutation leaves the fit without a companion running.
        assert _noise_refit_values(plain)


# ---------------------------------------------------------------------- the replay slots


#: Each restored result, and whether its refit runs. A refused refit is the RM23 defect:
#: before RM23 each of these read ``refit_nuisances`` true and its refit raised.
REPLAY_KINDS: dict[str, tuple[Callable[[], Any], bool]] = {
    "stratify=treatment": (restored_stratified, False),
    "stratify=treatment+outcome": (
        lambda: restored(cross_fitted(), stratify_folds="treatment+outcome"),
        False,
    ),
    "n_folds=1": (lambda: restored(cross_fitted(), n_folds=1), False),
    "repeats=0": (lambda: restored(cross_fitted(), repeats=0), False),
    "repeats=2 in sample": (lambda: restored(discrete_fit(), repeats=2), False),
    "plan without provenance": (without_provenance, False),
    "ctmle greedy stratified": (ctmle_stratified, False),
    "unbounded scale": (unbounded_scale, False),
    "v0.1.1 cross-fitted": (lambda: as_saved_by_v011(cross_fitted()), True),
    "v0.1.1 in sample": (lambda: as_saved_by_v011(discrete_fit()), True),
    "declared cross-fitted": (lambda: loads(dumps(cross_fitted())), True),
    "declared in sample": (lambda: loads(dumps(discrete_fit())), True),
}

#: The restored kinds whose refit this version refuses.
REFUSED_REPLAY_KINDS = [kind for kind, (_, runs) in REPLAY_KINDS.items() if not runs]


@pytest.fixture(scope="module")
def replay_results() -> dict[str, Any]:
    """One restored result of each kind in :data:`REPLAY_KINDS`."""
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

    def test_a_restored_stratified_refit_raises_the_slot_sentence(
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

    def test_without_the_class_default_a_v011_result_raises(
        self, replay_results: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """M7: release 0.1.1 wrote no ``split_plan``, and only the class default supplies it."""
        MUTATIONS["M7"].apply(monkeypatch)
        with pytest.raises(AttributeError, match="split_plan"):
            replay_disagreements(replay_results["v0.1.1 cross-fitted"], ("ate",))
        # The control: a result that holds its own attribute needs no default.
        assert replay_disagreements(replay_results["declared cross-fitted"], ("ate",)) == []
