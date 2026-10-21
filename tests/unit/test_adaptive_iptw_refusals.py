r"""Refusals of AdaptiveIPTW: the message, the type, the code and the absence of a learner fit.

The ids follow the plan: R1, R2b, R3, R4, R8 to R13, R16, R17, R19 and R20, and the range rule.
R2, R5 to R7 and R18 are deleted, and their ids are not reused. Each refusal that runs before a
learner leaves zero fits on a spy learner. The tests also pin the shared missing-treatment gate:
``"adaptive_iptw"`` is in its method tuple and is exempt from the ``cross_fit`` branch, and the
TMLE and DRTMLE texts of that gate stay what they were.
"""

from __future__ import annotations

import re
import warnings
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import AdaptiveIPTWMethod, CausalStudy, LongitudinalTreatment, RegimeContrast
from cleverly.datasets import make_longitudinal, make_shift_dose
from cleverly.estimators import DRTMLE, TMLE, AdaptiveIPTW
from cleverly.estimators import iptw as module
from cleverly.estimators.composite import KNOWN_TREATMENT_DELTA_REFUSAL, missing_treatment_refusal
from cleverly.exceptions import (
    CapabilityError,
    DataError,
    MethodConfigurationError,
    RefusalReason,
)
from cleverly.interventions import Incremental, Static, Stochastic
from cleverly.methods import ADAPTIVE_IPTW_LONGITUDINAL_REFUSAL
from tests.unit import _adaptive_iptw_support as s
from tests.unit import _capability_sweep_support as sweep
from tests.unit import _direct_effect_support as direct_effect


class Counting(LogisticRegression):
    """A logistic regression that counts its ``fit`` calls on a class-level list."""

    calls: list[int] = []  # noqa: RUF012 - shared across clones on purpose

    def fit(self, x: Any, y: Any, sample_weight: Any = None) -> Any:
        type(self).calls.append(len(y))
        return super().fit(x, y, sample_weight=sample_weight)


class CountingRegressor(LinearRegression):
    calls: list[int] = []  # noqa: RUF012 - shared across clones on purpose

    def fit(self, x: Any, y: Any, sample_weight: Any = None) -> Any:
        type(self).calls.append(len(y))
        return super().fit(x, y, sample_weight=sample_weight)


def _estimator(**overrides: Any) -> AdaptiveIPTW:
    settings: dict[str, Any] = {
        "treatment_learner": Counting(max_iter=1000),
        "missingness_learner": Counting(max_iter=1000),
        "reduced_outcome_learner": CountingRegressor(),
        "cross_fit": False,
        "simultaneous": False,
        "random_state": 0,
    }
    settings.update(overrides)
    return AdaptiveIPTW(**settings)


def _no_fits() -> None:
    assert Counting.calls == []
    assert CountingRegressor.calls == []


@pytest.fixture(autouse=True)
def _reset_counters() -> None:
    Counting.calls = []
    CountingRegressor.calls = []


def _assert_refusal(
    error: Any,
    refusal: Any,
    *,
    code: RefusalReason | None = None,
    roadmap: str | None = None,
) -> None:
    assert error.refusal == refusal
    assert re.search(re.escape(refusal.condition), str(error))
    if code is not None:
        assert refusal.reason is code
    if roadmap is not None:
        assert refusal.roadmap == roadmap


def _fit(estimator: AdaptiveIPTW, frame: Any, **kwargs: Any) -> Any:
    kwargs.setdefault("outcome", "Y")
    kwargs.setdefault("treatment", "A")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator.fit(frame, **kwargs)


# ----------------------------------------------------------------------------------- R1, R4


class TestDataRefusals:
    def test_r1_a_continuous_dose_is_refused_before_a_learner(self) -> None:
        frame, _ = make_shift_dose(n=200, seed=1)
        with pytest.raises(CapabilityError) as raised:
            _fit(_estimator(), frame, covariates=["W1", "W2", "W3"], treatment_kind="continuous")
        _assert_refusal(raised.value, module.R1, code=RefusalReason.API_REDIRECTION)
        _no_fits()

    def test_r1_a_deterministic_regime_on_a_continuous_dose_meets_it(self) -> None:
        frame, _ = make_shift_dose(n=200, seed=1)
        estimator = _estimator(interventions=[Static(1.0, name="always")])
        with pytest.raises(CapabilityError) as raised:
            _fit(estimator, frame, covariates=["W1", "W2", "W3"], treatment_kind="continuous")
        _assert_refusal(raised.value, module.R1)
        _no_fits()

    def test_r4_an_intermediate_is_refused_with_the_implementation_gap(self) -> None:
        with pytest.raises(CapabilityError) as raised:
            _fit(
                _estimator(),
                direct_effect.cde_frame(),
                covariates=direct_effect.COVARIATES,
                intermediate="Z",
            )
        _assert_refusal(
            raised.value, module.R4, code=RefusalReason.IMPLEMENTATION_GAP, roadmap="F5"
        )
        _no_fits()

    def test_r4_is_the_shipped_blocker_the_study_reports(self) -> None:
        from cleverly import CausalStudy, ControlledDirectEffect, PointTreatment

        study = CausalStudy(
            direct_effect.cde_frame(),
            design=PointTreatment(
                outcome="Y", treatment="A", adjustment=direct_effect.COVARIATES, intermediate="Z"
            ),
        )
        records = {
            m.name: m
            for m in study.identify(ControlledDirectEffect(intermediate=0.0)).available_methods()
        }
        assert not records["adaptive_iptw"].available


# ---------------------------------------------------------------------- R2b, R3, R13 keywords


class TestKeywordRefusals:
    @pytest.mark.parametrize(
        ("keyword", "value"),
        [
            ("incremental", Incremental(0.5)),
            ("policies", [object()]),
        ],
    )
    def test_r2b_an_axis_keyword_is_refused_at_construction(self, keyword: str, value: Any) -> None:
        with pytest.raises(CapabilityError) as raised:
            AdaptiveIPTW(**{keyword: value})
        _assert_refusal(
            raised.value, module.axis_refusal(f"{keyword}="), code=RefusalReason.API_REDIRECTION
        )

    def test_r2b_a_learned_rule_is_refused_at_construction(self) -> None:
        with pytest.raises(CapabilityError) as raised:
            AdaptiveIPTW(learned_rule=object())
        _assert_refusal(raised.value, module.axis_refusal("learned_rule="))

    def test_r3_an_msm_is_refused_at_construction(self) -> None:
        with pytest.raises(CapabilityError) as raised:
            AdaptiveIPTW(msm=object())
        _assert_refusal(
            raised.value, module.target_refusal("an MSM"), code=RefusalReason.API_REDIRECTION
        )

    @pytest.mark.parametrize("target", ["att", "atc"])
    def test_r3_att_and_atc_are_refused_before_a_learner(self, target: str) -> None:
        with pytest.raises(CapabilityError) as raised:
            _fit(_estimator(estimands=[target]), sweep._linear_frame())
        _assert_refusal(raised.value, module.target_refusal(target))
        _no_fits()

    @pytest.mark.parametrize(
        ("keyword", "value"),
        [
            ("fluctuation", "linear"),
            ("q_bounds", (0.0, 1.0)),
            ("alpha", 0.99),
            ("target_weights", True),
            ("step_size", 0.01),
            ("targeting", "one_step"),
            ("outcome_by_arm", True),
            ("iterate", "best"),
            ("stopping", "absolute"),
            ("step_bound", 0.5),
            ("step_search", False),
            ("ratio", "classifier"),
            ("density_bins", 5),
        ],
    )
    def test_r13_an_outcome_fluctuation_keyword_is_refused(self, keyword: str, value: Any) -> None:
        with pytest.raises(MethodConfigurationError) as raised:
            AdaptiveIPTW(**{keyword: value})
        _assert_refusal(
            raised.value, module.keyword_refusal(keyword), code=RefusalReason.INVALID_DECLARATION
        )

    def test_r13_the_default_of_each_keyword_applies(self) -> None:
        estimator = AdaptiveIPTW()
        for keyword in module._OUTCOME_KEYWORDS:
            assert getattr(estimator, keyword) == module._DEFAULTS[keyword]

    def test_r13_two_learners_for_the_one_reduced_regression_are_refused(self) -> None:
        with pytest.raises(MethodConfigurationError) as raised:
            AdaptiveIPTW(
                outcome_learner=LinearRegression(), reduced_outcome_learner=LinearRegression()
            )
        _assert_refusal(raised.value, module.LEARNER_PAIR_REFUSAL)

    def test_a_modified_estimator_meets_the_same_refusal_before_a_learner(self) -> None:
        estimator = _estimator()
        estimator.q_bounds = (0.0, 1.0)
        with pytest.raises(MethodConfigurationError) as raised:
            _fit(estimator, sweep._linear_frame())
        _assert_refusal(raised.value, module.keyword_refusal("q_bounds"))
        _no_fits()


# ------------------------------------------------------------------ R8, R9, R10, R11, R16


class TestFitConfigurationRefusals:
    def test_r8_fold_targeting_is_an_implementation_gap(self) -> None:
        with pytest.raises(MethodConfigurationError) as raised:
            AdaptiveIPTW(targeting_scheme="fold")
        _assert_refusal(
            raised.value, module.R8, code=RefusalReason.IMPLEMENTATION_GAP, roadmap="X61"
        )

    def test_r9_fold_evaluation_is_an_implementation_gap(self) -> None:
        with pytest.raises(MethodConfigurationError) as raised:
            AdaptiveIPTW(cv_evaluation=True)
        _assert_refusal(
            raised.value, module.R9, code=RefusalReason.IMPLEMENTATION_GAP, roadmap="X61"
        )

    def test_r10_a_bootstrap_is_an_inference_gap(self) -> None:
        with pytest.raises(MethodConfigurationError) as raised:
            AdaptiveIPTW(n_bootstrap=50)
        _assert_refusal(raised.value, module.R10, code=RefusalReason.INFERENCE_GAP, roadmap="X55")

    def test_r10_a_non_default_resampling_is_refused(self) -> None:
        with pytest.raises(MethodConfigurationError) as raised:
            AdaptiveIPTW(bootstrap_resampling="cluster")
        _assert_refusal(raised.value, module.R10)

    def test_r11_a_form_outside_the_set_is_a_value_error(self) -> None:
        with pytest.raises(
            ValueError, match=re.escape("form must be 'targeted' or 'one_step'")
        ) as raised:
            AdaptiveIPTW(form="mean")
        assert "got 'mean'" in str(raised.value)
        assert not isinstance(raised.value, (CapabilityError, MethodConfigurationError))

    def test_r16_repeats_are_an_implementation_gap(self) -> None:
        with pytest.raises(MethodConfigurationError) as raised:
            AdaptiveIPTW(repeats=2)
        _assert_refusal(
            raised.value, module.R16, code=RefusalReason.IMPLEMENTATION_GAP, roadmap="X81"
        )


# ------------------------------------------------------------------------------- R12


class TestLongitudinalRefusal:
    def test_r12_a_method_refuses_the_longitudinal_translation(self) -> None:
        with pytest.raises(MethodConfigurationError) as raised:
            AdaptiveIPTWMethod().estimator_kwargs(longitudinal=True)
        _assert_refusal(
            raised.value, ADAPTIVE_IPTW_LONGITUDINAL_REFUSAL, code=RefusalReason.API_REDIRECTION
        )
        assert "SequentialIPW" in raised.value.refusal.alternative

    def test_r12_a_longitudinal_study_lists_the_method_unavailable_and_refuses_it(self) -> None:

        frame, _ = make_longitudinal(n=300, seed=12)
        design = LongitudinalTreatment(
            outcome="Y",
            treatment=("A1", "A2"),
            baseline=("W1", "W2"),
            time_varying=((), ("L2",)),
            censoring=("C1", "C2"),
        )
        effect = CausalStudy(frame, design=design).identify(
            RegimeContrast({"always": 1, "never": 0}, reference="never")
        )
        records = {m.name: m for m in effect.available_methods()}
        assert not records["adaptive_iptw"].available
        assert records["adaptive_iptw"].refusal == ADAPTIVE_IPTW_LONGITUDINAL_REFUSAL
        with pytest.raises(CapabilityError, match="no longitudinal derivation"):
            effect.estimate(method="adaptive_iptw")


# ------------------------------------------------------------------------------- R17, R19


class TestRegimeRefusals:
    def _regimes(self) -> list[Any]:
        return [Static(0, name="never"), Static(1, name="always")]

    def test_r17_a_cross_fitted_regime_with_a_missing_outcome_is_refused(self) -> None:
        estimator = _estimator(interventions=self._regimes(), cross_fit=True, n_folds=2)
        with pytest.raises(CapabilityError) as raised:
            _fit(estimator, sweep._missing_frame(), delta="Delta")
        _assert_refusal(
            raised.value, module.R17, code=RefusalReason.IMPLEMENTATION_GAP, roadmap="X81"
        )
        _no_fits()

    def test_r17_a_cross_fitted_regime_with_a_missing_treatment_is_refused(self) -> None:
        estimator = _estimator(interventions=self._regimes(), cross_fit=True, n_folds=2)
        frame = sweep._with_missing_treatment(sweep._linear_frame())
        with pytest.raises(CapabilityError) as raised:
            _fit(estimator, frame, covariates=["W1", "W2"], treatment_delta="DeltaA")
        _assert_refusal(raised.value, module.R17)
        _no_fits()

    def test_r17_the_same_regimes_fit_in_sample_and_the_cross_fitted_arm_fit_runs(self) -> None:
        frame = sweep._with_missing_treatment(sweep._linear_frame())
        in_sample = _fit(
            _estimator(interventions=self._regimes()),
            frame,
            covariates=["W1", "W2"],
            treatment_delta="DeltaA",
        ).single()
        assert in_sample.estimates
        arms = _fit(
            _estimator(cross_fit=True, n_folds=2),
            frame,
            covariates=["W1", "W2"],
            treatment_delta="DeltaA",
        ).single()
        assert {"ey1", "ey0"} <= set(arms.estimates)

    def _stochastic(self) -> Stochastic:
        return Stochastic(
            lambda data: np.column_stack([np.full(len(data), 0.7), np.full(len(data), 0.3)]),
            name="tilt",
            density_kind="known",
        )

    def test_r19_a_classifier_for_a_stochastic_regime_is_refused_before_a_fit(self) -> None:
        estimator = _estimator(
            interventions=[self._stochastic()], reduced_outcome_learner=Counting(max_iter=1000)
        )
        with pytest.raises(MethodConfigurationError) as raised:
            _fit(estimator, sweep._linear_frame(), covariates=["W1", "W2"])
        _assert_refusal(raised.value, module.R19, code=RefusalReason.INVALID_DECLARATION)
        _no_fits()

    def test_r19_does_not_apply_to_a_deterministic_regime_or_to_the_default_library(self) -> None:
        frame = sweep._linear_frame()
        deterministic = _fit(
            _estimator(interventions=self._regimes(), reduced_outcome_learner=LogisticRegression()),
            frame.assign(Y=(frame["Y"] > frame["Y"].median()).astype(int)),
            covariates=["W1", "W2"],
        ).single()
        assert deterministic.estimates
        default = _fit(
            AdaptiveIPTW(
                treatment_learner=LogisticRegression(max_iter=1000),
                interventions=[self._stochastic()],
                cross_fit=False,
                simultaneous=False,
                n_folds=2,
                random_state=0,
            ),
            frame,
            covariates=["W1", "W2"],
        ).single()
        assert default.estimates


# --------------------------------------------------------------------- R20 and the gate


class TestMissingTreatmentGate:
    @pytest.fixture(scope="class")
    def frame(self) -> Any:
        return sweep._with_missing_treatment(sweep._linear_frame())

    def test_r20_a_known_mechanism_with_a_missing_treatment_meets_the_shipped_refusal(
        self, frame: Any
    ) -> None:
        with pytest.raises(CapabilityError) as raised:
            _fit(
                _estimator(),
                frame,
                covariates=["W1", "W2"],
                treatment_delta="DeltaA",
                treatment_probabilities=np.full((len(frame), 2), 0.5),
            )
        _assert_refusal(
            raised.value, KNOWN_TREATMENT_DELTA_REFUSAL, code=RefusalReason.INVALID_DECLARATION
        )
        _no_fits()

    def test_the_natural_course_with_a_missing_treatment_meets_the_shipped_gate(
        self, frame: Any
    ) -> None:
        with pytest.raises(CapabilityError, match="A missing treatment is supported for"):
            _fit(
                _estimator(estimands=["par"]),
                frame,
                covariates=["W1", "W2"],
                treatment_delta="DeltaA",
            )
        _no_fits()

    def test_the_adaptive_iptw_preamble_names_its_own_compositions(self, frame: Any) -> None:
        with pytest.raises(CapabilityError) as raised:
            _fit(
                _estimator(estimands=["par"]),
                frame,
                covariates=["W1", "W2"],
                treatment_delta="DeltaA",
            )
        text = str(raised.value)
        assert (
            "A missing treatment is supported for AdaptiveIPTW arm means, in sample and "
            "cross-fitted, and for its regime means in sample, as point estimates." in text
        )
        assert "TMLE and DRTMLE fits" not in text

    def test_adaptive_iptw_is_in_the_method_tuple_and_exempt_from_the_cross_fit_branch(
        self, frame: Any
    ) -> None:
        data = _prepared(frame)
        for cross_fit in (False, True):
            estimator = _estimator(cross_fit=cross_fit, n_folds=2)
            assert missing_treatment_refusal(estimator, data, ("ey1", "ey0")) is None
        generic = _estimator()
        generic._assessment_method = "other"  # type: ignore[assignment]
        refusal = missing_treatment_refusal(generic, data, ("ey1",))
        assert refusal is not None
        assert "This fit is not one of those compositions." in str(refusal)

    def test_tmle_and_drtmle_keep_the_cross_fit_text_bit_for_bit(self, frame: Any) -> None:
        head = (
            "A is missing on {n} row(s). A missing treatment is supported for arm means and "
            "their contrasts (ey, ate, rr, or) on in-sample TMLE and DRTMLE fits and, for point "
            "estimates, per-arm outcome-adaptive CTMLE fits, and for regime means and "
            "arm-indexed MSM coefficients on in-sample TMLE fits, through the composite "
            "indicator Delta_A * Delta * 1{{A = a}}. The composite construction uses the "
            "in-sample (Donsker) theorem. The cross-fitted composite construction is open "
            "(roadmap {row}). Pass cross_fit=False (CrossFitting(enabled=False) on a method)."
        )
        unrecorded = int(frame["DeltaA"].eq(0.0).sum())
        shared = {
            "cross_fit": True,
            "n_folds": 2,
            "treatment_learner": LogisticRegression(max_iter=1000),
            "missingness_learner": LogisticRegression(max_iter=1000),
            "simultaneous": False,
        }
        for estimator, row in (
            (TMLE(outcome_learner=LinearRegression(), **shared), "X53"),
            (
                DRTMLE(
                    outcome_learner=LinearRegression(),
                    reduced_outcome_learner=LinearRegression(),
                    reduced_treatment_learner=LogisticRegression(max_iter=1000),
                    **shared,
                ),
                "F21",
            ),
        ):
            with pytest.raises(CapabilityError) as raised:
                _fit(estimator, frame, covariates=["W1", "W2"], treatment_delta="DeltaA")
            assert str(raised.value) == head.format(n=unrecorded, row=row)
            assert raised.value.refusal.roadmap == row

    def test_tmle_still_meets_the_known_mechanism_branch(self, frame: Any) -> None:
        with pytest.raises(CapabilityError) as raised:
            _fit(
                TMLE(
                    outcome_learner=LinearRegression(),
                    treatment_learner=LogisticRegression(max_iter=1000),
                    cross_fit=False,
                    simultaneous=False,
                ),
                frame,
                covariates=["W1", "W2"],
                treatment_delta="DeltaA",
                treatment_probabilities=np.full((len(frame), 2), 0.5),
            )
        assert raised.value.refusal == KNOWN_TREATMENT_DELTA_REFUSAL


def _prepared(frame: Any) -> Any:
    from cleverly.data import CausalData

    return CausalData.from_frame(
        frame, outcome="Y", treatment="A", covariates=["W1", "W2"], treatment_delta="DeltaA"
    )


# ------------------------------------------------------------------------------- range rule


class TestRangeRuleRefusal:
    def test_a_ratio_target_outside_its_range_raises_a_data_error_after_targeting(self) -> None:
        import pandas as pd
        from scipy.special import expit

        rng = np.random.default_rng(105)
        n = 120
        w = rng.normal(size=n)
        a = (rng.random(n) < expit(0.8 * w)).astype(float)
        y = np.where(a == 1.0, (rng.random(n) < 0.6).astype(float), 0.0)
        frame = pd.DataFrame({"W": w, "A": a, "Y": y})
        with pytest.raises(DataError, match="not confined to the outcome's range"):
            s.fit_arm(
                frame,
                weights=None,
                estimands=("rr",),
                g_bounds=(0.001, 0.999),
                treatment_learner=LogisticRegression(C=100),
                regressor=LogisticRegression(),
            )
