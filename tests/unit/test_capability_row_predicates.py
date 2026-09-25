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

from types import SimpleNamespace
from typing import Any

import pytest

from cleverly.assessment import AssessmentStatus, SensitivityFacade
from cleverly.exceptions import CapabilityError
from cleverly.sensitivity import missingness as missingness_module
from cleverly.sensitivity import omitted_variable as omitted_variable_module
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
    fit_wide_bound_refusal,
)
from cleverly.targets.population_intervention import NATURAL_COURSE_TILT_REFUSAL
from tests.unit._capability_sweep_support import KINDS

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
    return {kind: KINDS[kind]() for kind in TILT_RULE_OF}


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
        monkeypatch.setattr(SensitivityFacade, "_tilt_rule", lambda self: None)
        problems = tilt_disagreements(tilt_fits[kind])
        assert len(problems) == 2
        assert all("reads available" in problem for problem in problems)

    def test_rows_that_ignore_the_predicate_leave_the_admitted_fit_alone(
        self, tilt_fits: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same mutation changes nothing where the table already said ``None``."""
        monkeypatch.setattr(SensitivityFacade, "_tilt_rule", lambda self: None)
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
