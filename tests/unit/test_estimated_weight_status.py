"""A guarded DR-TMLE fit with weights declared estimated shows a labelled interval.

No result read here gives the reduced-dimension regressions of an estimated weight, so ``DRTMLE`` with a non-empty ``guard`` and ``weights_estimated=True`` takes the
``"estimated_weight_plugin"`` status, and X50 in the roadmap holds the reopen route. The
flag changes no number, so a status and not a refusal records what it declares.

The fit is the RM20 probe: ``make_binary_outcome(n=500, seed=3)`` with weights drawn
uniform on [0.5, 2]. Three controls keep their interval on the same frame and weights: the
weights declared fixed, ``guard=()``, which fits the ordinary TMLE, and the ordinary TMLE
itself. Two mutations must fail: a DR-TMLE hook that is the ordinary one, and a hook that
ignores the guard. A per-arm outcome-adaptive ``CTMLE`` fit with estimated weights takes
``"generated_design_plugin"`` when ``per_arm_parametric_support`` does not admit it, and the
status text names the admitted per-arm fit and the sequential fits beside ``DRTMLE``.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from cleverly import variable_importance
from cleverly._inference_status import STATUS_RECORDS
from cleverly._summary import vocabulary
from cleverly._summary._estimates import plain_support_reason
from cleverly.datasets import make_binary_outcome
from cleverly.estimators import CTMLE, DRTMLE, TMLE
from cleverly.estimators.ctmle import per_arm_parametric_support
from cleverly.exceptions import WeightingWarning
from tests.conftest import linear_in_sample
from tests.unit._inference_status_support import (
    ROUTES,
    assert_assessment_note,
    assert_keeps_inference,
    assert_round_trips,
    assert_shows_interval,
)

pytestmark = pytest.mark.xdist_group("estimated_weight_status")

STATUS = "estimated_weight_plugin"
RECORD = STATUS_RECORDS[STATUS]

#: ``ey0`` beside ``ate`` gives the E-value its reported reference-arm mean, so the
#: control's E-value row is available and the status fit's row is the one that is not.
ESTIMANDS = ("ate", "ey0")


@pytest.fixture(scope="module")
def frame() -> Any:
    data = make_binary_outcome(n=500, seed=3)[0]
    data["w"] = np.random.default_rng(3).uniform(0.5, 2.0, len(data))
    return data


def fit(frame: Any, *, estimator: type = DRTMLE, estimated: bool = True, **settings: Any) -> Any:
    return (
        estimator(**linear_in_sample(estimands=ESTIMANDS, **settings))
        .fit(frame, outcome="Y", treatment="A", weights="w", weights_estimated=estimated)
        .single()
    )


@pytest.fixture(scope="module")
def result(frame: Any) -> Any:
    return fit(frame)


class TestAnEstimatedWeightDRTMLEShowsALabelledInterval:
    def test_the_estimates_show_the_plug_in_wald_interval_under_the_label(
        self, result: Any
    ) -> None:
        assert_shows_interval(result, STATUS, None)
        support = result["ate"].support
        assert RECORD.shown_reason in support.reason
        assert "X50" in support.evidence
        settings = result.summary(detail="full").section("Settings").lines
        assert any(plain_support_reason(RECORD.shown_reason) in line for line in settings)

    def test_the_nuisance_report_and_the_assessment_carry_the_note(self, result: Any) -> None:
        assert_assessment_note(result, STATUS)

    def test_the_evalue_is_available(self, result: Any) -> None:
        assert result.sensitivity.capability("evalue").available

    def test_the_score_verdict_does_not_claim_an_interval(self, result: Any) -> None:
        check = result.score_verdict
        assert check.passed
        summary = check.summary()
        assert "dr_curve" not in summary.note_codes
        assert summary.note("equations_only").text == vocabulary.NOTE_TEXTS[
            "equations_only"
        ].format(reason=RECORD.plain_reason)


class TestTheNeighbouringFitsKeepTheirInterval:
    """The controls: each differs from the status fit in one declaration."""

    @pytest.mark.parametrize(
        "change",
        [
            pytest.param({"estimated": False}, id="weights declared fixed"),
            pytest.param({"guard": ()}, id="guard=() fits the ordinary TMLE"),
            pytest.param({"estimator": TMLE}, id="the ordinary TMLE"),
        ],
    )
    def test_the_control_keeps_its_interval(self, frame: Any, change: dict[str, Any]) -> None:
        control = fit(frame, **change)
        assert_keeps_inference(control)
        # The nonzero witness for the E-value row above: it is available here.
        assert control.sensitivity.capability("evalue").available

    def test_constant_weights_declared_estimated_keep_the_interval(self, frame: Any) -> None:
        """Constant weights fit the unweighted estimator, so the declaration acts on nothing.

        The status, the bootstrap warning and the simulated-confounding refusal all read
        ``CausalData.declares_estimated_weights``. A rule that reads the declaration alone
        withheld this fit's interval.
        """
        constant = frame.assign(w=2.0)
        control = fit(constant)
        assert control.data.weight_spec.estimated
        assert not control.data.declares_estimated_weights
        assert_keeps_inference(control)
        assert control.sensitivity.capability("evalue").available
        with warnings.catch_warnings():
            warnings.simplefilter("error", WeightingWarning)
            fit(constant, n_bootstrap=2)


class TestTheWeightTextAgreesWithTheStatus:
    """The weight report and the bootstrap warning are true of the status fit and a control.

    ``data.weight_report()`` belongs to the data, so it cannot read a fit's status. Its
    estimated-weight line said "the interval conditions on the fitted weights" on every
    fit, which contradicts a guarded DR-TMLE fit that reports no interval. The line now
    names that fit and its status, and the ordinary TMLE is the control whose interval the
    conditioning sentence describes.
    """

    def test_the_status_fit_reads_its_own_status_in_the_weight_report(self, result: Any) -> None:
        assert result.inference_status == STATUS
        text = result.data.weight_report().summary()
        assert "the interval conditions on the fitted weights" not in text
        assert STATUS in text
        assert f"{RECORD.reopened_by} in docs/roadmap.md" in text

    def test_the_ordinary_tmle_has_the_interval_the_report_describes(self, frame: Any) -> None:
        control = fit(frame, estimator=TMLE)
        assert_keeps_inference(control)
        text = control.data.weight_report().summary()
        assert "an interval that a fit reports conditions on the fitted weights" in text
        lower, upper = control.estimates["ate"].ci
        assert np.isfinite(lower) and np.isfinite(upper)

    def test_the_bootstrap_warning_claims_no_influence_curve_interval(self, frame: Any) -> None:
        """The warning once said the bootstrap conditions "just as the influence-curve ones
        do", and the status fit reports no influence-curve interval."""
        with pytest.warns(WeightingWarning, match="n_bootstrap") as caught:
            bootstrapped = fit(frame, n_bootstrap=2)
        assert bootstrapped.inference_status == STATUS
        (message,) = [str(w.message) for w in caught if "n_bootstrap" in str(w.message)]
        assert "condition on the fitted weights" in message
        assert "influence-curve" not in message


class TestVariableImportanceNamesTheStatus:
    def test_the_frame_names_the_shown_status(self, frame: Any) -> None:
        """A shown status does not refuse: the importance table carries the status column."""
        entries = variable_importance(
            frame,
            outcome="Y",
            candidates=["A"],
            covariates=["W1", "W2", "W3"],
            estimator=DRTMLE(**linear_in_sample()),
            weights="w",
            weights_estimated=True,
        )
        table = entries.to_frame()
        assert set(table["inference"]) == {STATUS}
        assert {"std_err", "p_value", "p_value_adjusted"} <= set(table.columns)


class TestTheStatusIsTheHooksToWithhold:
    """The mutation controls: each wrong hook fails the check its surface passes."""

    def test_the_ordinary_hook_fails_the_status_check(
        self, frame: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(DRTMLE, "_inference_status", TMLE._inference_status)
        with pytest.raises(AssertionError):
            assert_shows_interval(fit(frame), STATUS, None)

    def test_a_hook_that_ignores_the_guard_fails_the_guard_control(
        self, frame: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def ignores_guard(self: DRTMLE, data: Any) -> str:
            return STATUS if data.weight_spec.estimated else "influence_curve"

        monkeypatch.setattr(DRTMLE, "_inference_status", ignores_guard)
        # The mutant still passes the status fit, so only the control can catch it.
        assert_shows_interval(fit(frame), STATUS, None)
        with pytest.raises(AssertionError):
            assert_keeps_inference(fit(frame, guard=()))


class TestASavedResultLoadsAsSaved:
    @pytest.mark.parametrize("route", ROUTES)
    def test_a_saved_result_keeps_its_status(self, result: Any, route: str) -> None:
        assert_round_trips(result, STATUS, route)


class TestAPerArmCTMLEFitThatIsNotAdmittedTakesTheGeneratedDesignStatus:
    """A per-arm C-TMLE fit that ``per_arm_parametric_support`` does not admit keeps its status.

    A weighted ``"oat"`` fit with sklearn learners is not admitted, so it takes
    ``"generated_design_plugin"`` and F19 holds that interval. An admitted per-arm
    parametric fit with estimated weights takes ``"estimated_weight_plugin"``, as do the
    sequential fits. X50 reopens the status for ``DRTMLE`` only.
    """

    def test_estimated_weights_report_the_generated_design_status(self, frame: Any) -> None:
        settings = {"strategy": "oat", "oat_design": "per_arm"}
        result = fit(frame, estimator=CTMLE, **settings)
        assert result.data.declares_estimated_weights
        verdict = per_arm_parametric_support(
            CTMLE(**linear_in_sample(estimands=ESTIMANDS, **settings)), result.data
        )
        assert verdict.admitted is False
        assert result.inference_status == "generated_design_plugin"
        assert STATUS_RECORDS["generated_design_plugin"].reopened_by == "F19"

    def test_the_status_text_names_each_row_for_its_estimator(self) -> None:
        assert RECORD.reopened_by == "X50"
        assert "X50 in docs/roadmap.md reopens this for DRTMLE" in RECORD.reason
        assert "A sequential g-computation or IPW fit" in RECORD.reason
        assert "per-arm outcome-adaptive C-TMLE fit" in RECORD.reason
        assert "per_arm_parametric_support admits" in RECORD.reason
        assert "each one's stacked estimating-equation curve conditions on" in RECORD.reason
        assert RECORD.assessment_note.endswith("reopens it for DRTMLE")
