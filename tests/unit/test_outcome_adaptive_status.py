"""The status of each outcome-adaptive collaborative fit, by design.

``CTMLE(strategy="oat")`` has two treatment designs. ``oat_design="shared"`` fits one
treatment mechanism on the estimated outcome predictions of every arm, and no result covers
it, so every shared fit takes the ``"generated_design_plugin"`` status (RM20). F19 in the
roadmap holds its reopen route.

``oat_design="per_arm"``, the default, fits one binary mechanism per arm on that arm's own
prediction. Theorem 1 of Benkeser, Cai and van der Laan (2020) gives its curve for one
treatment-specific mean, and an indicator reduction per arm and a fixed-dimension stack give
the joint curve. A per-arm fit on complete data without baseline strata therefore reports
its interval while :data:`~cleverly.estimators.ctmle.OAT_PER_ARM_INFERENTIAL` is set. That
covers the binary and the three-arm fits, an ``ey1``-only request, the cross-fitted fit,
fixed weights and ``repeats=`` with ``simultaneous=False``. A per-arm fit with missing
outcomes or strata withholds, and X29 in the roadmap holds the missing construction. A
per-arm fit on weights declared estimated takes ``"estimated_weight_plugin"``, as a guarded
DR-TMLE fit does.

The control is the ordinary TMLE on the same frames. Each mutation control restores one
wrong status and must fail the check its cases pass.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest

from cleverly import variable_importance
from cleverly._inference_status import NON_INFERENTIAL
from cleverly.assessment import AssessmentStatus
from cleverly.datasets import make_binary_outcome, make_missing_outcome, make_multi_arm
from cleverly.estimators import CTMLE, TMLE
from cleverly.estimators import ctmle as ctmle_module
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
from tests.unit._natural_course_support import NeverFit, never_fit_learners
from tests.unit._source_mutation import mutate

pytestmark = pytest.mark.xdist_group("outcome_adaptive_status")

STATUS = "generated_design_plugin"
RECORD = NON_INFERENTIAL[STATUS]


def _binary() -> Any:
    return make_binary_outcome(n=500, seed=3)[0]


def _stratified() -> Any:
    frame = _binary()
    return frame.assign(S=(frame["W1"] > 0).astype(float))


def _weighted() -> Any:
    frame = _binary()
    return frame.assign(wt=0.5 + (frame["W1"] > 0).astype(float))


#: Each case: the frame builder, the roles the fit reads, and the estimator settings.
CASES: dict[str, tuple[Any, dict[str, Any], dict[str, Any]]] = {
    "binary": (_binary, {}, {"estimands": ("ate",)}),
    "multi_arm": (lambda: make_multi_arm(n=600, seed=5)[0], {}, {"estimands": ("ate",)}),
    "missing_outcome": (
        lambda: make_missing_outcome(n=500, seed=4)[0],
        {"delta": "Delta"},
        {"estimands": ("ate",)},
    ),
    "ey1_only": (_binary, {}, {"estimands": ("ey1",)}),
    "cross_fitted": (_binary, {}, {"estimands": ("ate",), "cross_fit": True, "n_folds": 3}),
    "weighted": (_weighted, {"weights": "wt"}, {"estimands": ("ate",)}),
    "repeats": (
        _binary,
        {},
        {"estimands": ("ate", "ey"), "cross_fit": True, "n_folds": 3, "repeats": 3},
    ),
    "strata": (
        _stratified,
        {"strata": ["S"], "covariates": ["W1", "W2", "W3", "S"]},
        {"estimands": ("ate",)},
    ),
}

#: The cases the per-arm design admits, and the two it withholds.
ADMITTED = ("binary", "multi_arm", "ey1_only", "cross_fitted", "weighted", "repeats")
WITHHELD = ("missing_outcome", "strata")


def fit(estimator: type, case: str, design: str | None = None) -> Any:
    build, roles, settings = CASES[case]
    extra: dict[str, Any] = {}
    if estimator is CTMLE:
        extra = {"strategy": "oat", "oat_design": design}
    return (
        estimator(**extra, **linear_in_sample(**settings))
        .fit(build(), outcome="Y", treatment="A", **roles)
        .single()
    )


def _prepared(estimator: CTMLE, case: str) -> Any:
    """The prepared data of the fit ``case`` names, with no learner run."""
    build, roles, _ = CASES[case]
    return estimator._prepare(
        build(),
        outcome="Y",
        treatment="A",
        covariates=roles.get("covariates"),
        delta=roles.get("delta"),
        weights=roles.get("weights"),
        id=None,
        intermediate=None,
        strata=roles.get("strata"),
    )


@pytest.fixture(scope="module", params=list(CASES))
def case(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture(scope="module")
def shared_result(case: str) -> Any:
    return fit(CTMLE, case, "shared")


class TestEverySharedDesignFitReportsNoInterval:
    def test_the_estimates_withhold_their_inference(self, shared_result: Any) -> None:
        assert_withholds(shared_result, STATUS)

    def test_the_nuisance_report_and_the_assessment_carry_the_note(
        self, shared_result: Any
    ) -> None:
        assert_assessment_note(shared_result, STATUS)

    def test_the_evalue_is_unavailable_with_the_reason(self, shared_result: Any, case: str) -> None:
        contrasts = [name for name in shared_result.estimates if name.startswith("ate")]
        if not contrasts:
            # An arm mean has no two-arm contrast, and that check runs before the status.
            capability = shared_result.sensitivity.capability("evalue")
            assert capability.status is AssessmentStatus.NOT_APPLICABLE
            return
        row = shared_result.sensitivity.run_all(arguments={"evalue": {"estimand": contrasts[0]}})
        assert row["evalue"].status is AssessmentStatus.UNAVAILABLE
        assert RECORD.reason in row["evalue"].detail
        with pytest.raises(CapabilityError) as raised:
            shared_result.sensitivity.evalue(contrasts[0])
        assert RECORD.reason in str(raised.value)
        if case not in ("multi_arm", "strata"):
            # Two contrasts defer the bare row to an explicit estimand, so only a
            # single-contrast fit shows the reason on the bare capability row.
            assert_evalue_unavailable(shared_result, STATUS)


@pytest.mark.parametrize("admitted", ADMITTED)
class TestAnAdmittedPerArmFitReportsItsInterval:
    def test_the_estimates_keep_their_inference(self, admitted: str) -> None:
        result = fit(CTMLE, admitted)
        assert result.extra["ctmle"].design == "per_arm"
        assert_keeps_inference(result)
        text = result.summary()
        assert "95% CI" in text
        assert RECORD.summary_note() not in text

    def test_the_status_is_the_ordinary_fits(self, admitted: str) -> None:
        """The same frame under ``TMLE`` gives the same status: the control."""
        estimator = CTMLE(strategy="oat", **linear_in_sample(**CASES[admitted][2]))
        ordinary = fit(TMLE, admitted)
        assert ordinary.inference_status == "influence_curve"
        assert estimator._inference_status(ordinary.data) == ordinary.inference_status


class TestTheAdmittedOutputs:
    def test_the_default_band_is_built_at_one_split(self) -> None:
        result = (
            CTMLE(strategy="oat", **linear_in_sample(estimands=("ey", "ate"), simultaneous=True))
            .fit(_binary(), outcome="Y", treatment="A")
            .single()
        )
        assert result.inference_status == "influence_curve"
        assert result.simultaneous is not None
        assert result.simultaneous.critical_value > 1.96

    def test_a_default_band_beside_repeats_raises_the_repeat_refusal(self) -> None:
        estimator = CTMLE(
            strategy="oat",
            **linear_in_sample(
                estimands=("ey", "ate"), simultaneous=True, cross_fit=True, n_folds=3, repeats=3
            ),
        )
        with pytest.raises(CapabilityError, match="simultaneous=True"):
            estimator.fit(_binary(), outcome="Y", treatment="A")

    def test_the_evalue_carries_its_interval(self) -> None:
        # The ATE E-value reads the reported reference-arm mean, so the fit reports it.
        result = (
            CTMLE(strategy="oat", **linear_in_sample(estimands=("ate", "ey0")))
            .fit(_binary(), outcome="Y", treatment="A")
            .single()
        )
        assert result.sensitivity.capability("evalue").available
        report = result.sensitivity.evalue("ate")
        assert np.isfinite(report.limit)

    def test_variable_importance_reports_inference(self) -> None:
        importance = variable_importance(
            _binary(),
            outcome="Y",
            candidates=["A"],
            covariates=["W1", "W2", "W3"],
            estimator=CTMLE(strategy="oat", **linear_in_sample(estimands=("ate",))),
        )
        assert {"std_err", "ci_lower", "ci_upper", "p_value"} <= set(importance.to_frame().columns)

    def test_estimated_weights_withhold_the_interval(self) -> None:
        """The conditioning argument concerns the efficient curve, and this curve is not."""
        estimated = _estimated_weight_fit()
        assert_withholds(estimated, "estimated_weight_plugin")
        # The control: the same frame with the weights declared fixed keeps its interval.
        fixed = fit(CTMLE, "weighted")
        assert_keeps_inference(fixed)

    def test_a_hook_that_ignores_the_declaration_fails_the_check(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The mutation control: estimated weights given the ordinary status."""

        def mutant(self: CTMLE, data: Any) -> str:
            if ctmle_module.per_arm_design_admits(self, data):
                return str(TMLE._inference_status(self, data))
            return STATUS

        monkeypatch.setattr(CTMLE, "_inference_status", mutant)
        with pytest.raises(AssertionError):
            assert_withholds(_estimated_weight_fit(), "estimated_weight_plugin")


def _estimated_weight_fit() -> Any:
    return (
        CTMLE(strategy="oat", **linear_in_sample(estimands=("ate",)))
        .fit(_weighted(), outcome="Y", treatment="A", weights="wt", weights_estimated=True)
        .single()
    )


@pytest.mark.parametrize("withheld", WITHHELD)
class TestAPerArmFitOutsideTheResultWithholds:
    def test_the_estimates_withhold_their_inference(self, withheld: str) -> None:
        result = fit(CTMLE, withheld)
        assert result.extra["ctmle"].design == "per_arm"
        assert_withholds(result, STATUS)

    def test_the_status_is_known_before_any_learner_is_fitted(self, withheld: str) -> None:
        settings = CASES[withheld][2]
        estimator = CTMLE(strategy="oat", **linear_in_sample(**settings, **never_fit_learners()))
        assert estimator._inference_status(_prepared(estimator, withheld)) == STATUS
        assert NeverFit.calls == 0


class TestTheOrdinaryTMLEKeepsItsInterval:
    """The control: the same frames and learners under ``TMLE`` keep all three accessors."""

    def test_the_same_frame_keeps_its_interval(self, case: str) -> None:
        assert_keeps_inference(fit(TMLE, case))


class TestVariableImportanceRefusesBeforeItFits:
    def test_the_shared_design_is_refused_before_the_first_learner_is_fitted(self) -> None:
        assert_variable_importance_refuses(
            STATUS,
            make_binary_outcome(n=200, seed=3)[0],
            covariates=["W1", "W2", "W3"],
            estimator=CTMLE(
                strategy="oat", oat_design="shared", **linear_in_sample(**never_fit_learners())
            ),
        )

    def test_a_per_arm_fit_with_missing_outcomes_is_refused_before_it_fits(self) -> None:
        assert_variable_importance_refuses(
            STATUS,
            make_missing_outcome(n=200, seed=4)[0],
            covariates=["W1", "W2", "W3"],
            estimator=CTMLE(strategy="oat", **linear_in_sample(**never_fit_learners())),
            delta="Delta",
        )


def _without(data: Any, **fields: Any) -> Any:
    """A shallow copy of ``data`` with ``fields`` replaced, for a mutant key to read."""
    clone = copy.copy(data)
    for name, value in fields.items():
        object.__setattr__(clone, name, value)
    return clone


class TestTheStatusIsTheHooksToWithhold:
    def test_a_hook_that_admits_the_shared_design_fails_the_check(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The mutation control: the shared design given the ordinary status."""
        monkeypatch.setattr(ctmle_module, "per_arm_design_admits", lambda estimator, data: True)
        mutant = fit(CTMLE, "binary", "shared")
        with pytest.raises(AssertionError):
            assert_withholds(mutant, STATUS)

    def test_a_hook_that_withholds_the_admitted_per_arm_fit_fails_the_check(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ctmle_module, "per_arm_design_admits", lambda estimator, data: False)
        mutant = fit(CTMLE, "binary")
        with pytest.raises(AssertionError):
            assert_keeps_inference(mutant)

    @pytest.mark.parametrize("ignored", WITHHELD)
    def test_a_key_that_ignores_a_withheld_input_fails_the_check(
        self, monkeypatch: pytest.MonkeyPatch, ignored: str
    ) -> None:
        """The key with one of its data reads dropped admits the withheld case."""
        original = ctmle_module.per_arm_design_admits

        def mutant(estimator: Any, data: Any) -> bool:
            if ignored == "strata":
                data = _without(data, strata=None)
            else:
                data = _without(data, observed=np.ones(data.n, dtype=bool))
            return original(estimator, data)

        monkeypatch.setattr(ctmle_module, "per_arm_design_admits", mutant)
        result = fit(CTMLE, ignored)
        with pytest.raises(AssertionError):
            assert_withholds(result, STATUS)

    def test_a_key_that_ignores_the_revert_flag_fails_the_check(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The key with its read of the flag dropped keeps the interval after a revert."""
        mutate(
            monkeypatch,
            ctmle_module,
            ctmle_module,
            "per_arm_design_admits",
            [("OAT_PER_ARM_INFERENTIAL\n        and ", "")],
        )
        monkeypatch.setattr(ctmle_module, "OAT_PER_ARM_INFERENTIAL", False)
        with pytest.raises(AssertionError):
            assert_withholds(fit(CTMLE, "binary"), STATUS)

    @pytest.mark.parametrize("admitted", ADMITTED)
    def test_the_revert_flag_withholds_every_admitted_case(
        self, monkeypatch: pytest.MonkeyPatch, admitted: str
    ) -> None:
        """With the declared revert applied, every per-arm fit takes the withheld status."""
        monkeypatch.setattr(ctmle_module, "OAT_PER_ARM_INFERENTIAL", False)
        assert_withholds(fit(CTMLE, admitted), STATUS)


class TestASavedResultLoadsAsSaved:
    """An ``oat`` result loads with the status and the design it was saved under."""

    @pytest.mark.parametrize("route", ROUTES)
    def test_the_shared_estimates_keep_the_status(self, shared_result: Any, route: str) -> None:
        restored = assert_round_trips(shared_result, STATUS, route)
        assert restored.estimator.oat_design == "shared"
        assert restored.extra["ctmle"].design == "shared"

    @pytest.mark.parametrize("route", ROUTES)
    def test_a_per_arm_fit_keeps_its_interval(self, route: str) -> None:
        result = fit(CTMLE, "binary")
        restored = assert_round_trips(result, "influence_curve", route)
        assert restored.estimator.oat_design == "per_arm"
        assert restored.extra["ctmle"].design == "per_arm"
        assert restored["ate"].ci == result["ate"].ci
