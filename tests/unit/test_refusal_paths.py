"""A refusal keeps its reason code on every path it takes.

Two groups of checks. The first group makes one realistic public call per reason code, and
each call is refused before any learner fits. A counting learner proves the "before": it
adds one to a class-level counter at every fit, and the counter reads zero after the
refusal. A control call that the package admits fits the same learners and moves the
counter, so a counter that never moves cannot pass the first group by accident.

The second group follows a refusal through each wrapper that catches one error and raises
or records another: the sensitivity replay wrappers, the stratified wrappers, the split-plan
check, the longitudinal curve, the combined report and the study. Each wrapper test sends an
inner record with a code other than the default through the wrapper. A wrapper that rebuilt
its error with the default code would pass a test whose inner code is the default, so every
inner record here is an ``inference_gap`` or an ``implementation_gap`` with an alternative.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import (
    ATE,
    CausalStudy,
    PointTreatment,
    RegimeMean,
    RiskRatio,
    SDRMethod,
)
from cleverly.assessment import AssessmentStatus, SensitivityFacade
from cleverly.datasets import make_linear_ate, make_longitudinal
from cleverly.estimators import _strata, reduced
from cleverly.exceptions import (
    CapabilityError,
    CleverlyError,
    DataError,
    MethodConfigurationError,
    Refusal,
    RefusalReason,
)
from cleverly.interventions import Incremental, Rule
from cleverly.learners import SplitPlan, random_partition
from cleverly.learners import crossfit as crossfit_module
from cleverly.longitudinal import LTMLE
from cleverly.longitudinal import estimator as longitudinal_estimator
from cleverly.longitudinal.estimator import LONGITUDINAL_REPLAY_RANDOM_STATE_UNSEEDED
from cleverly.msm import MSMSet
from cleverly.sensitivity import _simulated_confounding_common as common_replay
from cleverly.sensitivity import _simulated_confounding_fixed as fixed_replay
from cleverly.sensitivity.evalue import _EValueRefusal
from tests import discrete_law_mar
from tests.unit._natural_course_support import Counting, CountingLinear, CountingLogistic
from tests.unit._simulated_confounding_support import _alias, _fit_msm, _model

INNER_ALTERNATIVE = "Fit the request again with the supported option."


def _inner(reason: RefusalReason = RefusalReason.INFERENCE_GAP) -> Refusal:
    """A record whose code is not the default, so a dropped code cannot go unseen."""
    record = Refusal(reason, "The inner condition failed.", INNER_ALTERNATIVE, "F5")
    assert record.reason is not RefusalReason.INVALID_DECLARATION
    return record


def _assert_carries(outer: CleverlyError, inner: Refusal, *, prefix: str = "") -> None:
    """The outer error keeps the inner code, alternative and roadmap id, and names its own text."""
    assert outer.refusal.reason is inner.reason
    assert outer.refusal.alternative == inner.alternative
    assert outer.refusal.roadmap == inner.roadmap
    assert outer.refusal.condition == prefix + inner.condition
    assert str(outer) == outer.refusal.message()


# --------------------------------------------------------------------------------------
# Group 1: one realistic call per code, refused before any learner fits.
# --------------------------------------------------------------------------------------

COLUMNS = ("W1", "W2", "W3", "W4")


def _study() -> CausalStudy:
    frame, _ = make_linear_ate(n=120, seed=1)
    return CausalStudy(
        frame,
        design=PointTreatment(outcome="Y", treatment="A", adjustment=COLUMNS),
    )


def _learners() -> dict[str, Any]:
    """Counting learners. A refusal that precedes every fit leaves ``Counting.calls`` at 0."""
    return {
        "outcome_learner": CountingLinear(),
        "treatment_learner": CountingLogistic(max_iter=200),
        "cross_fit": False,
        "simultaneous": False,
        "random_state": 0,
    }


def _estimate(estimand: Any, **options: Any) -> Any:
    return _study().estimate(estimand, **{**_learners(), **options})


@dataclass(frozen=True)
class Case:
    """One pre-fit refusal: the call, and the record the call must raise."""

    reason: RefusalReason
    call: Any
    condition: str
    alternative: str | None
    roadmap: str | None = None
    #: A long condition is pinned by its opening sentence.
    whole: bool = True


CASES = {
    RefusalReason.INVALID_DECLARATION: Case(
        RefusalReason.INVALID_DECLARATION,
        lambda: _estimate(RiskRatio()),
        "estimand(s) ['rr'] require a binary outcome (family='binomial'), but this fit has "
        "family='gaussian'. The risk ratio and odds ratio are not defined for a continuous "
        "outcome. Drop them or dichotomise the outcome.",
        None,
    ),
    RefusalReason.DIFFERENT_ESTIMAND: Case(
        RefusalReason.DIFFERENT_ESTIMAND,
        lambda: _estimate(
            _incremental(),
            g_bounds=(0.05, 0.95),
        ),
        "g_bounds= truncates the treatment mechanism, and on an incremental fit the "
        "mechanism is part of the *estimand*: q_delta = delta*g / (delta*g + 1 - g), so "
        "truncating g moves Psi(delta) itself rather than regularising a denominator. It "
        "is also unnecessary: the clever covariate is delta/D at A=1 and 1/D at A=0, both "
        "between min(delta, 1/delta) and max(delta, 1/delta) whatever g is, which is the "
        "point of an incremental intervention.",
        "Leave g_bounds at its default. (nuisance_bound= is a different matter and is "
        "accepted: with delta= the missingness mechanism divides the covariate and is not "
        "in the estimand, so bounding it regularises rather than retargets.)",
    ),
    RefusalReason.IMPLEMENTATION_GAP: Case(
        RefusalReason.IMPLEMENTATION_GAP,
        lambda: _estimate(ATE(), method="collaborative_tmle", cv_evaluation=True),
        "CTMLE does not support cv_evaluation=True: canonical CV-TMLE selection requires a "
        "separate fold-specific collaborative derivation.",
        "Use cv_evaluation=False.",
        roadmap="F18",
    ),
    RefusalReason.INFERENCE_GAP: Case(
        RefusalReason.INFERENCE_GAP,
        lambda: _missing_outcome_study().estimate(
            ATE(),
            missingness_learner=CountingLogistic(max_iter=200),
            n_bootstrap=20,
            **{**_learners(), "cross_fit": True},
        ),
        "Cross-fitted TMLE of arm-indexed means and contrasts with missing outcomes supports "
        "one audited stacked CV-TMLE contract. No audited result covers bootstrap inference.",
        "Set n_bootstrap=0 (Inference(n_bootstrap=0)).",
        "F2",
        whole=False,
    ),
    RefusalReason.API_REDIRECTION: Case(
        RefusalReason.API_REDIRECTION,
        lambda: _estimate(ATE(), method="sdr"),
        "method 'sdr' cannot estimate ATE: the SDR estimator is the sequential regression "
        "of a longitudinal fit.",
        "A point-treatment functional fits tmle, drtmle or collaborative_tmle.",
    ),
}


def _missing_outcome_study() -> CausalStudy:
    return CausalStudy(
        discrete_law_mar.frame(),
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",), missingness="Delta"),
    )


def _incremental() -> Any:
    from cleverly import IncrementalMean

    return IncrementalMean([Incremental(delta=2.0, name="d2")])


def _treat_when_w1_is_high(frame: Any) -> Any:
    return (np.asarray(frame["W1"]) > 0).astype(float)


def _refused(call: Any) -> CleverlyError:
    Counting.calls = 0
    with pytest.raises(CleverlyError) as raised:
        call()
    return raised.value


class TestOneRefusalPerCodeBeforeAnyFit:
    @pytest.mark.parametrize("reason", list(RefusalReason))
    def test_the_call_is_refused_with_its_code_before_any_learner_fits(
        self, reason: RefusalReason
    ) -> None:
        case = CASES[reason]
        error = _refused(case.call)
        assert Counting.calls == 0
        assert isinstance(error, CapabilityError)
        record = error.refusal
        assert record.reason is case.reason
        if case.whole:
            assert record.condition == case.condition
        else:
            assert record.condition.startswith(case.condition)
        if case.whole:
            assert record.alternative == case.alternative
        else:
            assert case.alternative is not None and record.alternative is not None
            assert record.alternative.startswith(case.alternative)
        if case.roadmap is None:
            assert record.roadmap is None
        else:
            assert record.roadmap == case.roadmap
            assert case.roadmap in str(error)
        # The text is the record's message, and the message ends with the alternative.
        assert str(error) == record.message()
        assert str(error).startswith(case.condition)
        if case.alternative is not None and case.whole:
            assert str(error).endswith(case.alternative)
        elif case.alternative is not None:
            assert case.alternative in str(error)

    def test_every_code_is_covered(self) -> None:
        assert set(CASES) == set(RefusalReason)
        # Five records that differ, so a code that every call shared would fail here.
        assert len({case.reason for case in CASES.values()}) == 5

    def test_a_code_other_than_invalid_declaration_names_an_alternative(self) -> None:
        for reason, case in CASES.items():
            if reason is not RefusalReason.INVALID_DECLARATION:
                assert case.alternative, reason

    def test_a_call_the_package_admits_does_fit(self) -> None:
        """The control: the counter moves when a learner fits, so a zero above means no fit."""
        Counting.calls = 0
        result = _estimate(ATE())
        assert Counting.calls > 0
        assert result is not None

    def test_an_estimated_rule_is_an_implementation_gap_that_names_its_roadmap_item(
        self,
    ) -> None:
        """No estimator of a sample-learned regime runs, so the code is a gap, not inference."""
        error = _refused(
            lambda: _estimate(
                RegimeMean((Rule(_treat_when_w1_is_high, name="learned", rule_kind="estimated"),))
            )
        )
        assert Counting.calls == 0
        assert error.refusal.reason is RefusalReason.IMPLEMENTATION_GAP
        assert error.refusal.roadmap == "X11"
        assert error.refusal.condition.startswith(
            "a treatment rule learned from the analysis sample is refused."
        )

    def test_a_stratified_cross_fitted_fit_with_missing_outcomes_names_its_roadmap_item(
        self,
    ) -> None:
        """An implementation gap that names its roadmap item, refused before the nuisance fit."""
        frame = discrete_law_mar.frame().assign(S=(np.arange(discrete_law_mar.N) % 2).astype(float))
        study = CausalStudy(
            frame,
            design=PointTreatment(
                outcome="Y",
                treatment="A",
                adjustment=("W", "S"),
                strata=("S",),
                missingness="Delta",
            ),
        )
        error = _refused(
            lambda: study.estimate(
                ATE(),
                missingness_learner=CountingLogistic(max_iter=200),
                **{**_learners(), "cross_fit": True},
            )
        )
        assert Counting.calls == 0
        assert error.refusal.reason is RefusalReason.IMPLEMENTATION_GAP
        assert error.refusal.roadmap == "X53"
        assert error.refusal.alternative == "Drop strata= from fit (PointTreatment(strata=()))."
        assert str(error).endswith(
            "No audited result covers baseline strata. X53 in docs/roadmap.md tracks this. "
            "Drop strata= from fit (PointTreatment(strata=()))."
        )


# --------------------------------------------------------------------------------------
# Group 2: each wrapper keeps the inner code.
# --------------------------------------------------------------------------------------


class TestTheSensitivityWrappersKeepTheInnerCode:
    def test_the_provenance_wrapper_carries_a_coded_data_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = _fit_msm()
        inner = _inner()

        def refuse(self: Any, data: Any) -> None:
            raise DataError.from_refusal(inner)

        monkeypatch.setattr(PointTreatment, "_check_prepared", refuse)
        with pytest.raises(CapabilityError) as raised:
            common_replay.point_study_or_refuse(result, "the replay metadata disagrees")
        _assert_carries(raised.value, inner, prefix="the replay metadata disagrees: ")
        assert isinstance(raised.value.__cause__, DataError)

    def test_the_provenance_wrapper_takes_the_context_of_a_record(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A record as the context supplies its condition, and the inner code still wins."""
        result = _fit_msm()
        inner = _inner(RefusalReason.IMPLEMENTATION_GAP)
        context = Refusal(RefusalReason.API_REDIRECTION, "The context record.", "Use it.")

        def refuse(self: Any, data: Any) -> None:
            raise DataError.from_refusal(inner)

        monkeypatch.setattr(PointTreatment, "_check_prepared", refuse)
        with pytest.raises(CapabilityError) as raised:
            common_replay.point_study_or_refuse(result, context)
        _assert_carries(raised.value, inner, prefix="The context record: ")

    def test_the_provenance_wrapper_leaves_other_errors_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The control: only a data-contract failure is rewritten."""
        result = _fit_msm()

        def refuse(self: Any, data: Any) -> None:
            raise KeyError("W")

        monkeypatch.setattr(PointTreatment, "_check_prepared", refuse)
        with pytest.raises(KeyError):
            common_replay.point_study_or_refuse(result, "context")

    def test_the_fixed_replay_wrapper_carries_a_coded_data_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = _fit_msm()
        alias = _alias(result)
        key = result.parameter_keys[alias]
        # The control: without the patch the replay validates.
        assert fixed_replay.validate_fixed_replay(result, alias, key) is not None
        inner = _inner()

        def refuse(*args: Any, **kwargs: Any) -> Any:
            raise DataError.from_refusal(inner)

        monkeypatch.setattr(fixed_replay, "_freeze_msm", refuse)
        with pytest.raises(CapabilityError) as raised:
            fixed_replay.validate_fixed_replay(result, alias, key)
        _assert_carries(
            raised.value,
            inner,
            prefix="simulated_confounding found inconsistent fixed-policy parameter metadata: ",
        )


class TestTheStratifiedWrappersKeepTheInnerClassAndCode:
    @staticmethod
    def _data() -> Any:
        strata = np.array([0, 0, 1, 1])
        return SimpleNamespace(
            strata=strata,
            n_strata=2,
            has_strata=True,
            stratum_label=lambda code: f"S={code}",
        )

    @pytest.mark.parametrize("error_class", [CapabilityError, DataError, MethodConfigurationError])
    def test_the_reduced_regression_keeps_the_class_and_code(
        self, error_class: type[CleverlyError], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        inner = _inner()

        def refuse(*args: Any, **kwargs: Any) -> Any:
            raise error_class.from_refusal(inner)

        monkeypatch.setattr(reduced, "_pooled_column", refuse)
        with pytest.raises(CleverlyError) as raised:
            self._call_stratified_column()
        assert type(raised.value) is error_class
        _assert_carries(raised.value, inner, prefix="inside baseline stratum S=0: ")

    def test_the_reduced_regression_names_a_plain_error_without_a_code(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The control: a plain ``ValueError`` has no record, so it stays a plain error."""

        def refuse(*args: Any, **kwargs: Any) -> Any:
            raise ValueError("no trainable rows")

        monkeypatch.setattr(reduced, "_pooled_column", refuse)
        with pytest.raises(ValueError, match="inside baseline stratum S=0: no trainable") as raised:
            self._call_stratified_column()
        assert not isinstance(raised.value, CleverlyError)

    def _call_stratified_column(self) -> Any:
        return reduced._stratified_column(
            None,
            design=np.zeros((4, 1)),
            target=np.zeros(4),
            training=None,
            companion=None,
            fit_mask=None,
            data=self._data(),
            nuisance=cast(Any, None),
            task="regression",
            clip=None,
            n_jobs=1,
            diagnostics=[],
            companion_strata=None,
        )

    def test_the_stratified_msm_check_keeps_the_code(self, monkeypatch: pytest.MonkeyPatch) -> None:
        inner = _inner()

        class Evaluated:
            def subset(self, rows: Any) -> None:
                raise DataError.from_refusal(inner)

        monkeypatch.setattr(MSMSet, "evaluate", classmethod(lambda cls, msm, data: Evaluated()))
        with pytest.raises(DataError) as raised:
            _strata.check_stratified_msm_design(_model(), self._data())
        _assert_carries(raised.value, inner, prefix="inside baseline stratum S=0: ")


class TestTheSplitPlanCheckKeepsTheInnerCode:
    def test_a_coded_data_error_from_the_redraw_keeps_its_code(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plan = SplitPlan.from_folds([random_partition(12, 3, seed=5)])
        plan.verify(n=12)  # The control: the recorded plan verifies without the patch.
        inner = _inner()

        def refuse(*args: Any, **kwargs: Any) -> Any:
            raise DataError.from_refusal(inner)

        monkeypatch.setattr(crossfit_module, "random_partition", refuse)
        with pytest.raises(DataError) as raised:
            plan.verify(n=12)
        _assert_carries(
            raised.value,
            inner,
            prefix="split-plan repeat 0 records a draw that cannot run on these rows: ",
        )

    def test_a_plain_value_error_from_the_redraw_is_an_invalid_declaration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plan = SplitPlan.from_folds([random_partition(12, 3, seed=5)])

        def refuse(*args: Any, **kwargs: Any) -> Any:
            raise ValueError("bad fold count")

        monkeypatch.setattr(crossfit_module, "random_partition", refuse)
        with pytest.raises(DataError) as raised:
            plan.verify(n=12)
        assert raised.value.refusal.reason is RefusalReason.INVALID_DECLARATION
        assert "bad fold count" in str(raised.value)


class TestTheCombinedReportKeepsTheCode:
    @staticmethod
    def _result() -> Any:
        return _study().estimate(ATE(), **{**_learners(), "cross_fit": False})

    def test_a_refusal_the_run_catches_is_an_item_with_the_same_record(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        inner = _inner()

        def refuse(self: Any, *args: Any, **kwargs: Any) -> Any:
            raise CapabilityError.from_refusal(inner)

        result = self._result()
        # The control: the row is available, so the run calls the operation.
        assert result.sensitivity.capability("robustness_value").available
        before = result.sensitivity.run_all()["robustness_value"]
        assert before.status is AssessmentStatus.COMPLETED
        assert before.refusal is None

        monkeypatch.setattr(SensitivityFacade, "robustness_value", refuse)
        item = self._result().sensitivity.run_all()["robustness_value"]
        assert item.status is AssessmentStatus.UNAVAILABLE
        assert item.refusal == inner
        assert item.refusal.reason is RefusalReason.INFERENCE_GAP
        assert str(inner) in item.detail or inner.condition in item.detail

    def test_a_cached_evalue_refusal_is_raised_again_with_the_same_record(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        inner = _inner(RefusalReason.IMPLEMENTATION_GAP)
        calls = []

        def select(result: Any, estimand: Any) -> Any:
            calls.append(estimand)
            raise _EValueRefusal(
                AssessmentStatus.UNAVAILABLE,
                inner.condition,
                reason=inner.reason,
                alternative=inner.alternative,
                roadmap=inner.roadmap,
            )

        monkeypatch.setattr(
            importlib.import_module("cleverly.sensitivity.evalue"), "_select_evalue", select
        )
        facade = self._result().sensitivity
        with pytest.raises(_EValueRefusal) as first:
            facade._evalue_selection(None)
        with pytest.raises(_EValueRefusal) as second:
            facade._evalue_selection(None)
        # The second raise is built from the cache, and it keeps the first one's code.
        assert calls == [None]
        for error in (first.value, second.value):
            assert error.status is AssessmentStatus.UNAVAILABLE
            _assert_carries(error, inner)
        row = facade.capability("evalue")
        assert row.refusal == inner
        assert row.status is AssessmentStatus.UNAVAILABLE


class TestTheStudyKeepsTheRowsCode:
    @pytest.mark.parametrize("by", ["name", "object"])
    def test_a_refused_method_raises_the_availability_record(self, by: str) -> None:
        effect = _study().identify(ATE())
        row = {item.name: item for item in effect.available_methods()}["sdr"]
        assert row.refusal is not None
        assert row.refusal.reason is RefusalReason.API_REDIRECTION
        method: Any = "sdr" if by == "name" else SDRMethod()
        error = _refused(lambda: effect.estimate(method=method, **_learners()))
        assert Counting.calls == 0
        assert error.refusal.reason is row.refusal.reason
        assert error.refusal.alternative == row.refusal.alternative
        assert row.refusal.condition in error.refusal.condition
        assert str(error).startswith("method 'sdr' cannot estimate ATE: ")
        assert str(error).endswith(row.refusal.message())


class TestTheLongitudinalWrappersKeepTheInnerCode:
    @pytest.fixture(scope="class")
    def result(self) -> Any:
        frame, _ = make_longitudinal(n=140, seed=17)
        return LTMLE(
            {"always": 1, "never": 0},
            reference="never",
            outcome_learner=LinearRegression(),
            pseudo_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            censoring_learner=LogisticRegression(max_iter=1000),
            n_folds=2,
            learner_folds=2,
            random_state=17,
            simultaneous=False,
        ).fit(
            frame,
            outcome="Y",
            treatment=("A1", "A2"),
            baseline=("W1", "W2"),
            time_varying=((), ("L2",)),
            censoring=("C1", "C2"),
        )

    def test_the_curve_keeps_the_code_of_a_bound_the_resolver_refuses(
        self, result: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The control: the same call runs with the real resolver.
        assert len(result.diagnostics.truncation_curve([0.2])) > 0
        inner = _inner()

        def refuse(bound: Any) -> Any:
            raise CapabilityError.from_refusal(inner)

        monkeypatch.setattr(longitudinal_estimator, "resolve_cumulative_g_bounds", refuse)
        with pytest.raises(CapabilityError) as raised:
            longitudinal_estimator.longitudinal_truncation_curve(result, [0.2])
        prefix = (
            "a longitudinal truncation curve cannot resolve the requested cumulative bound "
            "grid, which is a sequence of scalars or of (lower, upper) pairs: "
        )
        _assert_carries(raised.value, inner, prefix=prefix)

    def test_the_curve_keeps_the_code_of_an_estimand_the_selector_refuses(
        self, result: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        inner = _inner(RefusalReason.IMPLEMENTATION_GAP)

        def refuse(estimates: Any, estimands: Any) -> Any:
            raise CapabilityError.from_refusal(inner)

        monkeypatch.setattr(longitudinal_estimator, "select_estimates", refuse)
        with pytest.raises(CapabilityError) as raised:
            longitudinal_estimator.longitudinal_truncation_curve(result, [0.2])
        _assert_carries(
            raised.value,
            inner,
            prefix="a longitudinal truncation curve cannot report the requested estimands: ",
        )

    def test_a_plain_bad_grid_is_an_invalid_declaration(self, result: Any) -> None:
        """The control: with no inner record the wrapper's own code is the default."""
        with pytest.raises(CapabilityError) as raised:
            longitudinal_estimator.longitudinal_truncation_curve(result, [(0.9, 0.1)])
        assert raised.value.refusal.reason is RefusalReason.INVALID_DECLARATION

    def test_a_replay_refusal_names_its_omission_and_its_code(self) -> None:
        frame, _ = make_longitudinal(n=90, seed=47)
        unseeded = RandomForestRegressor(n_estimators=2)
        fitted = LTMLE(
            {"always": 1},
            reference="always",
            outcome_learner=unseeded,
            pseudo_learner=unseeded,
            treatment_learner=LogisticRegression(max_iter=1000),
            censoring_learner=LogisticRegression(max_iter=1000),
            n_folds=1,
            random_state=7,
            simultaneous=False,
        ).fit(
            frame,
            outcome="Y",
            treatment=("A1", "A2"),
            baseline=("W1", "W2"),
            time_varying=((), ("L2",)),
            censoring=("C1", "C2"),
        )
        with pytest.raises(CapabilityError) as raised:
            fitted.diagnostics.truncation_curve([0.2])
        assert raised.value.refusal.reason is RefusalReason.INVALID_DECLARATION
        assert LONGITUDINAL_REPLAY_RANDOM_STATE_UNSEEDED in str(raised.value)
        assert str(raised.value) == raised.value.refusal.message()
