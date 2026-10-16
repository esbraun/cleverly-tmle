"""The support predicate, statuses, labels and persistence of the cross-fitted stacked interval.

X58b1. The predicate admits a cross-fitted per-arm fit at any fold count inside the fold-count
cap (at most 20 folds, held-out folds of at least 30 rows), with any ``repeats``, on a binary
outcome or a bounded one (plan 13.1, D6). The support rows ``C11.stacked`` and ``C11.repeated``
cover the binary cells the study ``ctmle-oat-per-arm-parametric-crossfit`` measures: ``n_folds=5``
and ``repeats`` in ``{1, 3}``. Another admitted fit reports its interval labelled
``"experimental"``. A fold or repeat count outside the study meets a preflight condition, because
the support key holds neither count.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

import cleverly
from cleverly.estimators import CTMLE, _oat_stack
from cleverly.estimators.ctmle import (
    MAX_STACK_FOLDS,
    MIN_STACK_FOLD_ROWS,
    CTMLEOutcomeAdaptiveFit,
    _unstudied_stack_conditions,
    per_arm_parametric_support,
)
from cleverly.exceptions import (
    CapabilityError,
    DataError,
    MethodConfigurationError,
    RefusalReason,
)
from cleverly.learners import ParametricMechanism, ParametricOutcome, random_partition
from cleverly.learners.crossfit import SplitPlan
from tests.studies import oat_parametric_quadrature as quad
from tests.studies import oat_per_arm_laws as laws
from tests.unit._inference_status_support import assert_round_trips

pytestmark = pytest.mark.xdist_group("oat_per_arm_crossfit_support")

ESTIMANDS = ("ey0", "ey1", "ate", "rr", "or")


def _options(**overrides: Any) -> dict[str, Any]:
    options: dict[str, Any] = {
        "strategy": "oat",
        "outcome_learner": ParametricOutcome(),
        "treatment_learner": ParametricMechanism(),
        "cross_fit": True,
        "n_folds": 5,
        "simultaneous": False,
        "estimands": ESTIMANDS,
        "g_bounds": laws.G_BOUNDS,
        "max_iter": 100,
        "tol": 1e-10,
        "random_state": 0,
    }
    options.update(overrides)
    return options


def _frame(n: int = 1500, seed: int = 31) -> pd.DataFrame:
    return laws.BINARY_ACTIVE.sample(n, seed)[0]


def _prepared(estimator: CTMLE, frame: pd.DataFrame | None = None) -> Any:
    return estimator._prepare(
        _frame() if frame is None else frame,
        outcome="Y",
        treatment="A",
        covariates=["W1", "W2", "W3"],
        delta=None,
        weights=None,
        id=None,
        intermediate=None,
        strata=None,
    )


def _verdict(frame: pd.DataFrame | None = None, **overrides: Any) -> Any:
    estimator = CTMLE(**_options(**overrides))
    return per_arm_parametric_support(estimator, _prepared(estimator, frame))


def _fit(frame: pd.DataFrame | None = None, **overrides: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            CTMLE(**_options(**overrides))
            .fit(
                _frame() if frame is None else frame,
                outcome="Y",
                treatment="A",
                covariates=["W1", "W2", "W3"],
            )
            .single()
        )


@pytest.fixture(scope="module")
def crossfit() -> Any:
    return _fit()


@pytest.fixture(scope="module")
def repeated() -> Any:
    return _fit(repeats=3)


class TestTheAdmittedValues:
    def test_cross_fitting_at_five_folds(self) -> None:
        verdict = _verdict()
        assert verdict.admitted
        assert "cross-fitted over 5 folds" in verdict.detail

    def test_three_repeats(self) -> None:
        verdict = _verdict(repeats=3)
        assert verdict.admitted
        assert "the per-arm fit, cross-fitted over 5 folds with 3 repeats, with" in verdict.detail

    def test_in_sample_is_unchanged(self) -> None:
        verdict = _verdict(cross_fit=False, n_folds=7)
        assert verdict.admitted
        assert verdict.detail.startswith("the per-arm fit, in sample, with")

    def test_every_estimate_supplies_inference(self, crossfit: Any, repeated: Any) -> None:
        for result in (crossfit, repeated):
            for name in ESTIMANDS:
                assert result[name].inference == "influence_curve", name
                assert result[name].plugin_std_error == result[name].std_error

    def test_the_arm_means_read_the_v_fold_stack(self, crossfit: Any) -> None:
        stack = _oat_stack.fitted_stack(crossfit)
        assert stack.problem.n_folds == 5
        curves = _oat_stack.corrected_curves(stack.problem, stack.theta).curves
        for j, name in enumerate(("ey0", "ey1")):
            expected = np.std(curves[:, j], ddof=1) / np.sqrt(crossfit.n)
            assert crossfit[name].std_error == pytest.approx(expected, rel=1e-9)

    def test_a_pipeline_fit_keeps_the_generated_design_status(self) -> None:
        result = _fit(
            outcome_learner=laws.interaction_outcome(2),
            treatment_learner=laws.cubic_logit_learner(),
        )
        assert result["ate"].inference == "generated_design_plugin"
        assert "stack" not in result.nuisance.diagnostics


class TestTheWidening:
    """D6: the predicate admits what the stacked theory covers, inside the fold-count cap."""

    @pytest.mark.parametrize("folds", [2, 10])
    def test_another_fold_count(self, folds: int) -> None:
        verdict = _verdict(n_folds=folds)
        assert verdict.admitted
        assert f"cross-fitted over {folds} folds" in verdict.detail

    def test_five_repeats(self) -> None:
        verdict = _verdict(repeats=5)
        assert verdict.admitted
        assert "with 5 repeats" in verdict.detail

    @pytest.mark.parametrize("repeats", [0, True, 2.0])
    def test_a_repeat_count_that_is_not_a_positive_integer(self, repeats: Any) -> None:
        estimator = CTMLE(**_options())
        data = _prepared(estimator)
        estimator.repeats = repeats
        assert per_arm_parametric_support(estimator, data).axis == "repeats"

    def test_a_cross_fitted_bounded_outcome(self) -> None:
        frame = quad.sample(quad.BINARY_ACTIVE, 1500, 33, bounded=(2.0, 10.0))
        verdict = _verdict(frame, q_bounds=(2.0, 10.0), estimands=("ey0", "ey1", "ate"))
        assert verdict.admitted

    def test_a_cross_fitted_bounded_outcome_without_bounds(self) -> None:
        frame = quad.sample(quad.BINARY_ACTIVE, 1500, 33, bounded=(2.0, 10.0))
        verdict = _verdict(frame, estimands=("ey0", "ey1", "ate"))
        assert verdict.axis == "q_bounds"


class TestTheSupportLabel:
    """The rows C11.stacked and C11.repeated cover the study's cells; other fits are labelled."""

    def test_the_stacked_fit_is_supported(self, crossfit: Any) -> None:
        supports = [crossfit[name].support for name in ESTIMANDS]
        assert {(support.label, support.row) for support in supports} == {
            ("supported", "C11.stacked")
        }

    def test_the_repeated_fit_is_supported(self) -> None:
        result = _fit(repeats=3, estimands=("ey0", "ey1", "ate"))
        supports = [result[name].support for name in ("ey0", "ey1", "ate")]
        assert {(support.label, support.row) for support in supports} == {
            ("supported", "C11.repeated")
        }

    def test_a_repeated_ratio_is_experimental_with_its_interval(self, repeated: Any) -> None:
        """The study measures no ratio under repeats, so C11.repeated holds only P1."""
        assert repeated["rr"].support.label == "experimental"
        assert repeated["rr"].inference == "influence_curve"
        assert np.isfinite(repeated["rr"].std_error)

    def test_a_cross_fitted_bounded_outcome_is_experimental_with_its_interval(self) -> None:
        frame = quad.sample(quad.BINARY_ACTIVE, 1500, 33, bounded=(2.0, 10.0))
        result = _fit(frame, q_bounds=(2.0, 10.0), estimands=("ey0", "ey1", "ate"))
        for name in ("ey0", "ey1", "ate"):
            assert result[name].inference == "influence_curve"
            assert result[name].support.label == "experimental"
            low, high = result[name].ci
            assert low < result[name].psi < high

    def test_another_fold_count_is_experimental_with_its_interval(self) -> None:
        """The key holds no fold count, so the preflight condition labels the fit."""
        result = _fit(n_folds=10, estimands=("ey0", "ey1", "ate"))
        for name in ("ey0", "ey1", "ate"):
            support = result[name].support
            assert (support.label, support.row) == ("experimental", "C11.stacked")
            assert "at n_folds=5 only. This fit has n_folds=10." in support.reason
            assert support.evidence == ("X58",)
            assert result[name].inference == "influence_curve"
            assert np.isfinite(result[name].std_error)

    def test_another_repeat_count_is_experimental(self) -> None:
        result = _fit(repeats=2, estimands=("ey0", "ey1", "ate"))
        support = result["ate"].support
        assert (support.label, support.row) == ("experimental", "C11.repeated")
        assert "repeats=1 or repeats=3 only. This fit has repeats=2." in support.reason
        assert result["ate"].inference == "influence_curve"

    def test_the_supported_rows_state_the_measured_counts(self, crossfit: Any) -> None:
        assert any("measures n_folds=5 only" in limit for limit in crossfit["ate"].support.limits)


class TestTheUnstudiedConditions:
    """The preflight conditions of an admitted fit outside the study's fold and repeat counts."""

    def _conditions(self, **overrides: Any) -> tuple[Any, ...]:
        estimator = CTMLE(**_options(**overrides))
        return _unstudied_stack_conditions(estimator, _prepared(estimator))

    @pytest.mark.parametrize(
        "overrides",
        [{}, {"repeats": 3}, {"cross_fit": False, "n_folds": 10}],
    )
    def test_a_studied_or_in_sample_fit_meets_none(self, overrides: dict[str, Any]) -> None:
        assert self._conditions(**overrides) == ()

    def test_a_refused_fit_meets_none(self) -> None:
        """A fit outside the predicate keeps the shipped curve, which no condition labels."""
        assert self._conditions(n_folds=10, outcome_learner=laws.interaction_outcome(2)) == ()

    def test_each_unstudied_count_meets_its_own_condition(self) -> None:
        conditions = self._conditions(n_folds=10, repeats=2)
        assert [c.reason for c in conditions] == [RefusalReason.INFERENCE_GAP] * 2
        assert [c.roadmap for c in conditions] == ["X58", "X58"]
        assert "n_folds=10" in conditions[0].condition
        assert "repeats=2" in conditions[1].condition

    def test_the_preflight_carries_the_conditions(self) -> None:
        estimator = CTMLE(**_options(n_folds=10))
        preflight = estimator._preflight_fit_configuration(_prepared(estimator))
        assert [c.roadmap for c in preflight.conditions] == ["X58"]


class TestTheFoldCap:
    """Refused always: the V-fold stack needs a fixed fold count and folds of 30 rows."""

    def test_the_constants(self) -> None:
        assert (MAX_STACK_FOLDS, MIN_STACK_FOLD_ROWS) == (20, 30)

    def test_twenty_one_folds(self) -> None:
        verdict = _verdict(n_folds=21)
        assert verdict.axis == "n_folds"
        assert "fold count fixed as n grows" in verdict.detail
        assert "n_folds=21" in verdict.detail

    def test_held_out_folds_below_thirty_rows(self) -> None:
        verdict = _verdict(_frame(250), n_folds=10)
        assert verdict.axis == "n_folds"
        assert "held-out fold of 25 rows" in verdict.detail

    def test_five_folds_of_twenty_rows(self) -> None:
        verdict = _verdict(_frame(100))
        assert verdict.axis == "n_folds"
        assert "held-out fold of 20 rows" in verdict.detail

    def test_the_capped_fit_keeps_its_point_estimate(self) -> None:
        result = _fit(_frame(250), n_folds=10)
        assert {estimate.inference for estimate in result.estimates.values()} == {
            "generated_design_plugin"
        }
        assert np.isfinite(result["ate"].psi)


class TestTheNamedRefusals:
    """Each refusal names what the stack is missing."""

    @staticmethod
    def _declared(option: str, value: Any) -> Any:
        """The verdict on a value :class:`CTMLE` refuses at construction, set afterwards.

        The engine refusal comes first on a fit; the predicate still names the missing piece.
        """
        estimator = CTMLE(**_options())
        data = _prepared(estimator)
        setattr(estimator, option, value)
        return per_arm_parametric_support(estimator, data)

    def test_cv_evaluation(self) -> None:
        with pytest.raises(CapabilityError, match="cv_evaluation"):
            CTMLE(**_options(cv_evaluation=True))
        verdict = self._declared("cv_evaluation", True)
        assert verdict.axis == "cv_evaluation"
        assert "not implemented in the stack" in verdict.detail

    def test_fold_targeting(self) -> None:
        with pytest.raises(CapabilityError, match="targeting_scheme"):
            CTMLE(**_options(targeting_scheme="fold"))
        verdict = self._declared("targeting_scheme", "fold")
        assert verdict.axis == "targeting_scheme"
        assert "eps_{a,k}" in verdict.detail

    def test_a_split_plan(self) -> None:
        plan = SplitPlan.from_folds([random_partition(1500, 5, seed=3)])
        verdict = _verdict(split_plan=plan)
        assert verdict.axis == "split_plan"
        assert "independent of the data" in verdict.detail

    def test_target_weights(self) -> None:
        verdict = _verdict(target_weights=True)
        assert verdict.axis == "target_weights"
        assert "w H" in verdict.detail


class TestTheDefaultBand:
    """``simultaneous=True`` builds no band on an admitted fit, so repeats do not refuse it."""

    def test_three_repeats_keep_the_intervals_and_build_no_band(self) -> None:
        assert _verdict(repeats=3, simultaneous=True).admitted
        result = _fit(repeats=3, simultaneous=True)
        assert {estimate.inference for estimate in result.estimates.values()} == {"influence_curve"}
        assert result.simultaneous is None
        assert "No simultaneous band" in result.summary()

    def test_a_hook_that_builds_the_band_hits_the_repeat_refusal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The control: a band over median-combined repeats is refused, not built."""
        monkeypatch.setattr(CTMLE, "_builds_default_band", lambda self, data: True)
        with pytest.raises(CapabilityError, match="median-combined repeats"):
            _fit(repeats=3, simultaneous=True)


class TestTheEngineRefusalsComeFirst:
    def test_stratified_folds(self) -> None:
        with pytest.raises((MethodConfigurationError, CapabilityError, ValueError)):
            _fit(stratify_folds="treatment")

    def test_repeats_in_sample(self) -> None:
        with pytest.raises((MethodConfigurationError, CapabilityError, ValueError)):
            _fit(cross_fit=False, repeats=3)


class TestTheEstimandResolution:
    """m5: the predicate reads the pure resolution, never the refusing one."""

    def test_unresolvable_estimands_are_refused_not_raised(self) -> None:
        frame = quad.sample(quad.BINARY_ACTIVE, 1500, 33, bounded=(2.0, 10.0))
        verdict = _verdict(frame, q_bounds=(2.0, 10.0), cross_fit=False)
        assert not verdict.admitted
        assert verdict.axis == "estimands"
        assert "resolve" in verdict.detail

    def test_the_binary_default_is_refused_by_the_estimator_first(self) -> None:
        verdict = _verdict(estimands=None)
        assert verdict.axis == "estimands"
        # m6: the detail names the estimands that fail, and no X58 part holds att or atc.
        assert verdict.detail.endswith("got atc, att")
        assert "part (b)" not in verdict.detail
        with pytest.raises(Exception, match="att"):
            _fit(estimands=None)


class TestTheCaughtErrors:
    """m7: only a record error of the stack becomes a failed arm; any other error propagates."""

    def test_a_missing_record_fails_every_arm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from cleverly.estimators import ctmle as ctmle_module

        def missing(*args: Any, **kwargs: Any) -> Any:
            raise _oat_stack.StackRecordError("the nuisances carry no stack record")

        monkeypatch.setattr(ctmle_module, "stack_from_fit", missing)
        result = _fit(estimands=("ey0", "ey1", "ate"))
        assert {result[name].inference for name in ("ey0", "ey1", "ate")} == {
            "stack_failure_plugin"
        }

    def test_a_defect_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from cleverly.estimators import ctmle as ctmle_module

        def defect(*args: Any, **kwargs: Any) -> Any:
            raise ValueError("a shape defect")

        monkeypatch.setattr(ctmle_module, "stack_from_fit", defect)
        with pytest.raises(ValueError, match="a shape defect"):
            _fit(estimands=("ey0", "ey1", "ate"))


def _separated_frame(seed: int) -> pd.DataFrame:
    """Outcomes separated by ``W1`` everywhere but on three rows of fold 0."""
    frame = _frame(600, 13)
    estimator = CTMLE(**_options())
    folds = estimator._folds(_prepared(estimator, frame), seed)
    frame["Y"] = (frame["W1"] > 0.0).astype(int)
    fold_zero = np.flatnonzero(folds.assignment == 0)
    positive = fold_zero[frame["W1"].to_numpy()[fold_zero] > 0.5][:3]
    frame.loc[frame.index[positive], "Y"] = 0
    return frame


def test_a_separated_fold_refuses_with_the_learners_data_error() -> None:
    frame = _separated_frame(0)
    with pytest.raises(DataError, match="separation"):
        CTMLE(**_options()).fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2", "W3"])


class TestTheRecord:
    def test_the_fit_records_its_folds_and_repeats(self, crossfit: Any, repeated: Any) -> None:
        for result, repeats in ((crossfit, 1), (repeated, 3)):
            fit = result.extra["ctmle"]
            assert isinstance(fit, CTMLEOutcomeAdaptiveFit)
            assert (fit.stacked_folds, fit.stacked_repeats) == (5, repeats)

    def test_the_summary_names_the_folds_and_the_median(self, repeated: Any) -> None:
        text = repeated.summary()
        assert "cross-fitted over 5 folds in the exact V-fold stack" in text
        assert "median of 3 cross-fitting repeats" in text

    def test_each_draw_records_its_fold_coefficients(self, repeated: Any) -> None:
        records = [repeat.nuisance.diagnostics["stack"] for repeat in repeated.repeats]
        assert [record.folds for record in records] == [5, 5, 5]
        assert records[0].outcome != records[1].outcome

    def test_every_fit_writes_a_draw_record(self) -> None:
        result = _fit(
            outcome_learner=laws.interaction_outcome(2),
            treatment_learner=laws.cubic_logit_learner(),
        )
        record = result.repeats[0].inference["ate"]
        assert record.status == "generated_design_plugin"
        assert record.curve is None and record.shipped_se is None
        assert record.se == pytest.approx(result["ate"].plugin_std_error, rel=1e-12)


class TestPersistence:
    @pytest.mark.parametrize("route", ["save", "pickle"])
    def test_a_cross_fitted_fit_round_trips(self, crossfit: Any, route: str, tmp_path: Any) -> None:
        if route == "save":
            restored = cleverly.load(crossfit.save(tmp_path / "fit.pkl"))
        else:
            restored = assert_round_trips(crossfit, "influence_curve", "pickle")
        for name in ESTIMANDS:
            assert restored[name].std_error == crossfit[name].std_error
        assert restored.summary() == crossfit.summary()
        assert restored.nuisance.diagnostics["stack"] == crossfit.nuisance.diagnostics["stack"]

    def test_a_repeated_fit_round_trips_its_draws(self, repeated: Any, tmp_path: Any) -> None:
        restored = cleverly.load(repeated.save(tmp_path / "fit.pkl"))
        for repeat, original in zip(restored.repeats, repeated.repeats, strict=True):
            assert repeat.inference.keys() == original.inference.keys()
            for name, record in repeat.inference.items():
                assert record.status == original.inference[name].status
                assert record.se == original.inference[name].se
                assert record.shipped_se == original.inference[name].shipped_se
        for name in ESTIMANDS:
            assert restored[name].std_error == repeated[name].std_error


class TestRetarget:
    def test_a_retarget_rereads_its_own_fold_record(self) -> None:
        estimator = CTMLE(**_options())
        data = _prepared(estimator)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = estimator.fit(data).single()
            again, _ = estimator.retarget(data, fit.nuisance, estimands=("ate",))
        assert again["ate"].inference == "influence_curve"
        assert again["ate"].std_error == pytest.approx(fit["ate"].std_error, rel=1e-12)

    def test_a_retarget_without_the_record_fails_every_arm(self) -> None:
        from dataclasses import replace

        estimator = CTMLE(**_options())
        data = _prepared(estimator)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = estimator.fit(data).single()
            diagnostics = dict(fit.nuisance.diagnostics)
            del diagnostics["stack"]
            bare = replace(fit.nuisance, diagnostics=diagnostics)
            out, _ = estimator.retarget(data, bare, estimands=("ate",))
        assert out["ate"].inference == "stack_failure_plugin"
