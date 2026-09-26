"""Every outcome-adaptive collaborative fit withholds its interval, as RM20 decides.

``CTMLE(strategy="oat")`` fits one treatment mechanism on the estimated outcome predictions
of every arm and targets every arm mean jointly. Theorem 1 of Benkeser, Cai and van der
Laan (2020) proves the ordinary curve for one binary treatment-specific mean with one scalar
design, so no result covers a shipped fit. An ``ey1``-only request uses the same joint
design, and a fit with ``delta=`` is outside the theorem too. Every such fit therefore takes
the ``"generated_design_plugin"`` status, and F19 in the roadmap holds the reopen route.

The cases are the RM20 probes: the binary joint fit, the three-arm fit, the in-sample fit
with missing outcomes, an ``ey1``-only request, and the cross-fitted default. The control is
the ordinary TMLE on the same frames, because the roadmap's "one binary treatment-specific
mean" control has no shipped instance. A mutation that restores the ordinary status on
``oat`` must fail the check the cases pass.
"""

from __future__ import annotations

from typing import Any

import pytest

from cleverly._inference_status import NON_INFERENTIAL
from cleverly.assessment import AssessmentStatus
from cleverly.datasets import make_binary_outcome, make_missing_outcome, make_multi_arm
from cleverly.estimators import CTMLE, TMLE
from cleverly.estimators.ctmle import is_selector_strategy
from cleverly.exceptions import CapabilityError
from tests.conftest import linear_in_sample
from tests.unit._inference_status_support import (
    ROUTES,
    assert_assessment_note,
    assert_evalue_unavailable,
    assert_keeps_inference,
    assert_round_trips,
    assert_variable_importance_refuses,
    assert_withholds,
)
from tests.unit._natural_course_support import never_fit_learners

pytestmark = pytest.mark.xdist_group("outcome_adaptive_status")

STATUS = "generated_design_plugin"
RECORD = NON_INFERENTIAL[STATUS]

#: Each case: the frame builder, the roles the fit reads, and the estimator settings.
CASES: dict[str, tuple[Any, dict[str, str], dict[str, Any]]] = {
    "binary": (lambda: make_binary_outcome(n=500, seed=3)[0], {}, {"estimands": ("ate",)}),
    "multi_arm": (lambda: make_multi_arm(n=600, seed=5)[0], {}, {"estimands": ("ate",)}),
    "missing_outcome": (
        lambda: make_missing_outcome(n=500, seed=4)[0],
        {"delta": "Delta"},
        {"estimands": ("ate",)},
    ),
    "ey1_only": (lambda: make_binary_outcome(n=500, seed=3)[0], {}, {"estimands": ("ey1",)}),
    "cross_fitted": (
        lambda: make_binary_outcome(n=500, seed=3)[0],
        {},
        {"estimands": ("ate",), "cross_fit": True, "n_folds": 3},
    ),
}


def fit(estimator: type, case: str) -> Any:
    build, roles, settings = CASES[case]
    extra = {"strategy": "oat"} if estimator is CTMLE else {}
    return (
        estimator(**extra, **linear_in_sample(**settings))
        .fit(build(), outcome="Y", treatment="A", **roles)
        .single()
    )


@pytest.fixture(scope="module", params=list(CASES))
def case(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture(scope="module")
def result(case: str) -> Any:
    return fit(CTMLE, case)


class TestEveryOutcomeAdaptiveFitReportsNoInterval:
    def test_the_estimates_withhold_their_inference(self, result: Any) -> None:
        assert_withholds(result, STATUS)

    def test_the_nuisance_report_and_the_assessment_carry_the_note(self, result: Any) -> None:
        assert_assessment_note(result, STATUS)

    def test_the_evalue_is_unavailable_with_the_reason(self, result: Any, case: str) -> None:
        contrasts = [name for name in result.estimates if name.startswith("ate")]
        if not contrasts:
            # An arm mean has no two-arm contrast, and that check runs before the status.
            capability = result.sensitivity.capability("evalue")
            assert capability.status is AssessmentStatus.NOT_APPLICABLE
            return
        row = result.sensitivity.run_all(arguments={"evalue": {"estimand": contrasts[0]}})
        assert row["evalue"].status is AssessmentStatus.UNAVAILABLE
        assert RECORD.reason in row["evalue"].detail
        with pytest.raises(CapabilityError) as raised:
            result.sensitivity.evalue(contrasts[0])
        assert RECORD.reason in str(raised.value)
        if case != "multi_arm":
            # Two contrasts defer the bare row to an explicit estimand, so only a
            # single-contrast fit shows the reason on the bare capability row.
            assert_evalue_unavailable(result, STATUS)


class TestTheOrdinaryTMLEKeepsItsInterval:
    """The control: the same frames and learners under ``TMLE`` keep all three accessors."""

    def test_the_same_frame_keeps_its_interval(self, case: str) -> None:
        assert_keeps_inference(fit(TMLE, case))


class TestVariableImportanceRefusesBeforeItFits:
    def test_the_refusal_arrives_before_the_first_learner_is_fitted(self) -> None:
        assert_variable_importance_refuses(
            STATUS,
            make_binary_outcome(n=200, seed=3)[0],
            covariates=["W1", "W2", "W3"],
            estimator=CTMLE(strategy="oat", **linear_in_sample(**never_fit_learners())),
        )


class TestTheStatusIsTheHooksToWithhold:
    def test_a_hook_that_restores_the_ordinary_status_fails_the_check(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The mutation control: the hook as it stood before RM20."""

        def before_rm20(self: CTMLE, data: Any) -> str:
            if is_selector_strategy(self.strategy):
                return "working_mechanism_plugin"
            return "influence_curve"

        monkeypatch.setattr(CTMLE, "_inference_status", before_rm20)
        mutant = fit(CTMLE, "binary")
        with pytest.raises(AssertionError):
            assert_withholds(mutant, STATUS)


class TestASavedResultLoadsAsSaved:
    """An ``oat`` result loads with the status it was saved under."""

    @pytest.mark.parametrize("route", ROUTES)
    def test_the_estimates_keep_the_status(self, result: Any, route: str) -> None:
        assert_round_trips(result, STATUS, route)
