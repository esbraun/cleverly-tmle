"""The status of each outcome-adaptive collaborative fit, by design.

``CTMLE(strategy="oat")`` has two treatment designs. ``oat_design="shared"`` fits one
treatment mechanism on the estimated outcome predictions of every arm, and no result covers
it, so every shared fit takes the ``"generated_design_plugin"`` status (RM20). F19 in the
roadmap holds its reopen route.

``oat_design="per_arm"``, the default, fits one binary mechanism per arm on that arm's own
prediction, and Theorem 1 of Benkeser, Cai and van der Laan (2020) states its curve. That
theorem needs its condition (v), which fails for an estimated outcome regression, and the
registered study ``ctmle-oat-per-arm`` found the reported standard errors 8 to 24 percent
too small. So a per-arm fit takes ``"generated_design_plugin"`` unless
:func:`~cleverly.estimators.ctmle.per_arm_parametric_support` admits it. The predicate admits
an in-sample fit with ``ParametricOutcome`` and ``ParametricMechanism`` on a binary treatment
with complete, unweighted, unstratified data. Its stacked curve includes the generated-design
term, and the status follows the verdict.

The class marked "parametric" fits that admitted design and checks its outputs. Every
sklearn-learner fit stays withheld, and the predicate names the axis that refuses it.

The control is the ordinary TMLE on the same frames. Each mutation control restores one
wrong status and must fail the check its cases pass.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest

from cleverly._inference_status import STATUS_RECORDS
from cleverly.assessment import AssessmentStatus
from cleverly.datasets import make_binary_outcome, make_missing_outcome, make_multi_arm
from cleverly.estimators import CTMLE, TMLE
from cleverly.estimators import ctmle as ctmle_module
from cleverly.estimators.ctmle import SupportVerdict, per_arm_parametric_support
from cleverly.exceptions import CapabilityError
from cleverly.learners import ParametricMechanism, ParametricOutcome
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
RECORD = STATUS_RECORDS[STATUS]


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

#: The per-arm cases whose data the old key admitted, fitted with sklearn learners, and the two
#: whose data it withheld. The predicate refuses all of them, the first six on the learners.
PER_ARM = ("binary", "multi_arm", "ey1_only", "cross_fitted", "weighted", "repeats")
WITHHELD = ("missing_outcome", "strata")


def _parametric(**settings: Any) -> CTMLE:
    """The per-arm estimator the predicate admits on binary, complete, in-sample data."""
    options: dict[str, Any] = {
        "strategy": "oat",
        "outcome_learner": ParametricOutcome(),
        "treatment_learner": ParametricMechanism(),
        "cross_fit": False,
        "simultaneous": False,
        "random_state": 0,
        **settings,
    }
    return CTMLE(**options)


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


@pytest.mark.parametrize("admitted", PER_ARM)
class TestEveryPerArmFitWithholdsItsInterval:
    def test_the_estimates_withhold_their_inference(self, admitted: str) -> None:
        result = fit(CTMLE, admitted)
        assert result.extra["ctmle"].design == "per_arm"
        assert_withholds(result, STATUS)
        # The reason names the failed condition and the study that measured it.
        assert "condition (v)" in RECORD.reason
        assert "ctmle-oat-per-arm" in RECORD.reason

    def test_the_predicate_refuses_the_sklearn_learners(self, admitted: str) -> None:
        """The status follows the verdict, and the first unmet axis is the outcome learner."""
        result = fit(CTMLE, admitted)
        estimator = CTMLE(strategy="oat", **linear_in_sample(**CASES[admitted][2]))
        verdict = per_arm_parametric_support(estimator, result.data)
        assert verdict.admitted is False
        assert verdict.axis == "outcome_learner"
        assert estimator._inference_status(result.data) == STATUS


class TestTheWithheldOutputs:
    def test_no_band_is_built(self) -> None:
        result = (
            CTMLE(strategy="oat", **linear_in_sample(estimands=("ey", "ate"), simultaneous=True))
            .fit(_binary(), outcome="Y", treatment="A")
            .single()
        )
        assert result.inference_status == STATUS
        assert result.simultaneous is None

    def test_the_evalue_is_unavailable_with_the_reason(self) -> None:
        result = (
            CTMLE(strategy="oat", **linear_in_sample(estimands=("ate", "ey0")))
            .fit(_binary(), outcome="Y", treatment="A")
            .single()
        )
        capability = result.sensitivity.capability("evalue")
        assert capability.status is AssessmentStatus.UNAVAILABLE
        assert RECORD.reason in (capability.reason or "")

    def test_variable_importance_is_refused_before_it_fits(self) -> None:
        assert_variable_importance_refuses(
            STATUS,
            make_binary_outcome(n=200, seed=3)[0],
            covariates=["W1", "W2", "W3"],
            estimator=CTMLE(strategy="oat", **linear_in_sample(**never_fit_learners())),
        )

    def test_estimated_weights_withhold_by_the_same_status(self) -> None:
        assert_withholds(_estimated_weight_fit(), STATUS)


class TestAParametricPerArmFitReportsItsInterval:
    """The admitted fit: the stacked curve gives every accessor under ``influence_curve``."""

    def test_the_verdict_admits_the_fit_and_the_estimates_keep_inference(self) -> None:
        estimator = _parametric(estimands=("ate", "ey0"))
        result = estimator.fit(_binary(), outcome="Y", treatment="A").single()
        assert per_arm_parametric_support(estimator, result.data).admitted is True
        assert_keeps_inference(result)

    def test_the_evalue_carries_its_interval(self) -> None:
        # The ATE E-value reads the reported reference-arm mean, so the fit reports it.
        result = (
            _parametric(estimands=("ate", "ey0"))
            .fit(_binary(), outcome="Y", treatment="A")
            .single()
        )
        assert result.sensitivity.capability("evalue").available
        report = result.sensitivity.evalue("ate")
        assert np.isfinite(report.limit)

    def test_a_default_band_keeps_the_interval_and_is_not_built(self) -> None:
        """``simultaneous=True`` is admitted; the fit reports its intervals and builds no band."""
        estimator = _parametric(estimands=("ey", "ate"), simultaneous=True)
        result = estimator.fit(_binary(), outcome="Y", treatment="A").single()
        assert per_arm_parametric_support(estimator, result.data).admitted
        assert result.inference_status == "influence_curve"
        assert result.simultaneous is None


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

    def test_the_predicate_names_the_data_axis_for_the_parametric_learners(
        self, withheld: str
    ) -> None:
        estimator = _parametric(estimands=("ate",))
        verdict = per_arm_parametric_support(estimator, _prepared(estimator, withheld))
        assert verdict.admitted is False
        assert verdict.axis == withheld
        assert estimator._inference_status(_prepared(estimator, withheld)) == STATUS


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


class TestTheStatusFollowsThePredicate:
    """Each mutant predicate gives a status the check rejects.

    The mutants act on the status hook alone, so no stack runs on a case the predicate
    refuses. The sklearn-learner fit would start the stack with learners it cannot read.
    """

    def test_a_predicate_that_admits_the_shared_design_fails_the_check(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The mutation control: the shared design given the ordinary status."""
        estimator = CTMLE(
            strategy="oat",
            oat_design="shared",
            outcome_learner=ParametricOutcome(),
            treatment_learner=ParametricMechanism(),
            cross_fit=False,
            simultaneous=False,
        )
        data = _prepared(estimator, "binary")
        assert estimator._inference_status(data) == STATUS
        monkeypatch.setattr(
            ctmle_module,
            "per_arm_parametric_support",
            lambda estimator, data: SupportVerdict(True, None, "mutant"),
        )
        assert estimator._inference_status(data) != STATUS

    def test_a_predicate_that_withholds_the_admitted_per_arm_fit_fails_the_check(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            ctmle_module,
            "per_arm_parametric_support",
            lambda estimator, data: SupportVerdict(False, "outcome_learner", "mutant"),
        )
        mutant = _parametric(estimands=("ate",)).fit(_binary(), outcome="Y", treatment="A").single()
        with pytest.raises(AssertionError):
            assert_keeps_inference(mutant)

    @pytest.mark.parametrize("ignored", WITHHELD)
    def test_a_predicate_that_ignores_a_withheld_input_fails_the_check(
        self, monkeypatch: pytest.MonkeyPatch, ignored: str
    ) -> None:
        """The predicate with one of its data reads dropped admits the withheld case."""
        # The missing-outcome frame is continuous, so a declared range keeps that axis out of play.
        estimator = _parametric(estimands=("ate",), q_bounds=(-6.0, 6.0))
        data = _prepared(estimator, ignored)
        assert estimator._inference_status(data) == STATUS
        original = ctmle_module.per_arm_parametric_support

        def mutant(estimator: Any, data: Any) -> SupportVerdict:
            if ignored == "strata":
                data = _without(data, strata=None)
            else:
                data = _without(data, observed=np.ones(data.n, dtype=bool))
            return original(estimator, data)

        monkeypatch.setattr(ctmle_module, "per_arm_parametric_support", mutant)
        assert estimator._inference_status(data) != STATUS

    def test_a_predicate_that_drops_the_option_allow_list_fails_the_check(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The predicate that skips every unmet option admits an sklearn-learner fit."""
        estimator = CTMLE(strategy="oat", **linear_in_sample(estimands=("ate",)))
        data = _prepared(estimator, "binary")
        assert estimator._inference_status(data) == STATUS
        mutate(
            monkeypatch,
            ctmle_module,
            ctmle_module,
            "per_arm_parametric_support",
            [("        if not ok:\n", "        if False:\n")],
        )
        assert estimator._inference_status(data) != STATUS


class TestASavedResultLoadsAsSaved:
    """An ``oat`` result loads with the status and the design it was saved under."""

    @pytest.mark.parametrize("route", ROUTES)
    def test_the_shared_estimates_keep_the_status(self, shared_result: Any, route: str) -> None:
        restored = assert_round_trips(shared_result, STATUS, route)
        assert restored.estimator.oat_design == "shared"
        assert restored.extra["ctmle"].design == "shared"

    @pytest.mark.parametrize("route", ROUTES)
    def test_a_per_arm_fit_keeps_the_status(self, route: str) -> None:
        result = fit(CTMLE, "binary")
        restored = assert_round_trips(result, STATUS, route)
        assert restored.estimator.oat_design == "per_arm"
        assert restored.extra["ctmle"].design == "per_arm"
