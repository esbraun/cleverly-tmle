r"""The AdaptiveIPTW study API, the keyword inventory and the result record.

* ``AdaptiveIPTWMethod`` round-trips through the study API and equals the engine.
* ``available_methods`` agrees with the fit, on the arm axis, the regime axis and the natural
  course beside ``delta=``.
* ``extra["adaptive_iptw"]`` pickles, and the result serializes.
* The default estimands are the mean-group defaults on the arm axis and
  ``("ey_regime", "ate_regime")`` on the regime axis.
* A cross-fitted continuous outcome fits without ``q_bounds``.
* One test per row of the keyword inventory that the estimator applies.
"""

from __future__ import annotations

import pickle
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import (
    ATC,
    ATE,
    ATT,
    AdaptiveIPTWMethod,
    CausalStudy,
    CounterfactualMean,
    CrossFitting,
    Inference,
    ModelSpec,
    NaturalCourseMean,
    PointTreatment,
    PopulationAttributableRisk,
    RegimeContrast,
    RegimeMean,
    Runtime,
)
from cleverly.datasets import make_binary_outcome, make_linear_ate, make_missing_outcome
from cleverly.estimators import AdaptiveIPTW
from cleverly.estimators.iptw import AdaptiveIPTWFit
from cleverly.estimators.serialize import dumps, loads
from cleverly.exceptions import CapabilityError, MethodConfigurationError
from cleverly.interventions import Rule, Static
from tests.unit import _adaptive_iptw_support as s
from tests.unit import _capability_sweep_support as sweep


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    data, _ = make_linear_ate(n=300, seed=2)
    return data


@pytest.fixture(scope="module")
def study(frame: pd.DataFrame) -> CausalStudy:
    return CausalStudy(
        frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2"))
    )


def _method(**fields: Any) -> AdaptiveIPTWMethod:
    outcome = None if "reduced_outcome_learner" in fields else LinearRegression()
    return AdaptiveIPTWMethod(
        models=ModelSpec(
            treatment_learner=LogisticRegression(max_iter=1000), outcome_learner=outcome
        ),
        cross_fitting=CrossFitting(enabled=False),
        inference=Inference(simultaneous=False),
        runtime=Runtime(random_state=3),
        **fields,
    )


def _estimate(effect: Any, method: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return effect.estimate(method=method)


# ----------------------------------------------------------------------- round trip


class TestMethodRoundTrip:
    def test_the_method_equals_the_engine_it_configures(self, study: CausalStudy) -> None:
        method = _method()
        result = _estimate(study.identify(ATE()), method)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            direct = (
                AdaptiveIPTW(**method.estimator_kwargs(), estimands=["ate"])
                .fit(study.data)
                .single()
            )
        assert result.fitted_method == "adaptive_iptw"
        assert result.estimates["ate"].psi == pytest.approx(direct.estimates["ate"].psi, abs=1e-12)

    def test_the_name_alone_fits_with_the_default_library(self) -> None:
        method = AdaptiveIPTWMethod()
        assert method.name == "adaptive_iptw"
        assert method.form == "targeted"
        assert method.reduced_outcome_learner is None

    def test_form_reaches_the_engine_and_changes_the_reported_value(
        self, study: CausalStudy
    ) -> None:
        effect = study.identify(ATE())
        targeted = _estimate(effect, _method())
        one_step = _estimate(effect, _method(form="one_step"))
        record = one_step.extra["adaptive_iptw"]
        assert record.form == "one_step"
        assert targeted.extra["adaptive_iptw"].form == "targeted"
        assert abs(one_step.estimates["ate"].psi - targeted.estimates["ate"].psi) > 1e-9
        recorded = targeted.extra["adaptive_iptw"].one_step
        assert one_step.estimates["ate"].psi == pytest.approx(
            recorded["1"][0] - recorded["0"][0], abs=1e-9
        )

    def test_reduced_outcome_learner_reaches_the_engine(self, study: CausalStudy) -> None:
        s.SpyRegressor.calls = []
        _estimate(study.identify(ATE()), _method(reduced_outcome_learner=s.SpyRegressor()))
        assert len(s.SpyRegressor.calls) > 0

    def test_a_form_outside_the_set_is_refused_by_the_method(self) -> None:
        with pytest.raises(ValueError, match="form must be 'targeted' or 'one_step'"):
            AdaptiveIPTWMethod(form="mean")


# ------------------------------------------------------------ available_methods agreement


def _natural_course_study() -> CausalStudy:
    data, _ = make_missing_outcome(n=300, seed=4)
    return CausalStudy(
        data,
        design=PointTreatment(
            outcome="Y", treatment="A", adjustment=("W1", "W2"), missingness="Delta"
        ),
    )


def _binary_study() -> CausalStudy:
    data, _ = make_binary_outcome(n=300, seed=17)
    return CausalStudy(
        data, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2"))
    )


REGIMES = (Static(1, name="always"), Static(0, name="never"))


def _cells() -> list[tuple[str, Any, bool]]:
    rule = Rule(lambda f: (f["W1"] > 0).astype(int).to_numpy(), name="rule", rule_kind="known")
    return [
        ("ate", ATE(), True),
        ("counterfactual", CounterfactualMean(1), True),
        ("att", ATT(), False),
        ("atc", ATC(), False),
        ("regime_mean", RegimeMean(REGIMES), True),
        ("regime_contrast", RegimeContrast(REGIMES, reference="never"), True),
        ("rule_mean", RegimeMean((rule, Static(0, name="never"))), True),
    ]


class TestAvailableMethodsAgreeWithTheFit:
    @pytest.mark.parametrize(("label", "estimand", "expected"), _cells(), ids=lambda v: str(v)[:20])
    def test_the_point_cells(
        self, study: CausalStudy, label: str, estimand: Any, expected: bool
    ) -> None:
        effect = study.identify(estimand)
        record = {m.name: m for m in effect.available_methods()}["adaptive_iptw"]
        assert record.available is expected, label
        if expected:
            result = _estimate(effect, _method())
            assert result.fitted_method == "adaptive_iptw"
        else:
            assert record.refusal is not None
            with pytest.raises(CapabilityError) as raised:
                _estimate(effect, _method())
            assert record.refusal.condition in str(raised.value)

    def test_the_natural_course_beside_delta_is_available_and_fits(self) -> None:
        effect = _natural_course_study().identify(NaturalCourseMean())
        record = {m.name: m for m in effect.available_methods()}["adaptive_iptw"]
        assert record.available
        result = _estimate(effect, _method())
        assert "ey_obs" in result.estimates
        assert result.fitted_method == "adaptive_iptw"

    def test_the_population_attributable_risk_beside_delta_is_available_and_fits(self) -> None:
        effect = _natural_course_study().identify(PopulationAttributableRisk())
        record = {m.name: m for m in effect.available_methods()}["adaptive_iptw"]
        assert record.available
        result = _estimate(effect, _method())
        assert "par" in result.estimates

    def test_a_binary_outcome_ratio_is_available_and_fits_or_meets_the_range_rule(self) -> None:
        from cleverly import RiskRatio

        effect = _binary_study().identify(RiskRatio())
        record = {m.name: m for m in effect.available_methods()}["adaptive_iptw"]
        assert record.available
        result = _estimate(effect, _method())
        assert "rr" in result.estimates


# ----------------------------------------------------------------------------- the record


class TestResultRecord:
    @pytest.fixture(scope="class")
    def result(self) -> Any:
        return sweep.fit_iptw()

    def test_the_record_is_the_frozen_fit_and_pickles(self, result: Any) -> None:
        record = result.extra["adaptive_iptw"]
        assert isinstance(record, AdaptiveIPTWFit)
        assert pickle.loads(pickle.dumps(record)) == record
        with pytest.raises(AttributeError):
            record.form = "one_step"  # type: ignore[misc]

    def test_the_record_holds_the_documented_fields(self, result: Any) -> None:
        record = result.extra["adaptive_iptw"]
        for name in (
            "form",
            "arms",
            "strata",
            "untargeted",
            "one_step",
            "targeted",
            "epsilon",
            "rounds",
            "stop",
            "score",
            "reduced_learner",
            "constant_design",
            "bound_active",
            "axis",
            "regimes",
            "regime_route",
        ):
            assert hasattr(record, name), name
        assert record.axis == "arm"
        assert record.stop in {"tolerance", "cap"}

    def test_the_result_serializes_and_loads_as_saved(self, result: Any) -> None:
        loaded = loads(dumps(result))
        assert loaded.extra["adaptive_iptw"] == result.extra["adaptive_iptw"]
        for name, estimate in result.estimates.items():
            assert loaded.estimates[name].psi == estimate.psi

    def test_a_regime_record_names_its_axis_and_routes(self) -> None:
        record = sweep.fit_iptw_stochastic().extra["adaptive_iptw"]
        assert record.axis == "regime"
        assert set(record.regime_route.values()) <= {"S9", "S11"}
        assert "S11" in record.regime_route.values()
        assert "S9" in record.regime_route.values()


# ------------------------------------------------------------------- default estimands


class TestDefaultEstimands:
    def test_the_arm_axis_resolves_to_the_mean_group(self, frame: pd.DataFrame) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = (
                AdaptiveIPTW(
                    treatment_learner=LogisticRegression(max_iter=1000),
                    reduced_outcome_learner=LinearRegression(),
                    cross_fit=False,
                    simultaneous=False,
                )
                .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"])
                .single()
            )
        assert set(result.estimates) == {"ate", "ey1", "ey0"}
        assert "att" not in result.estimates and "atc" not in result.estimates

    def test_all_resolves_to_the_mean_group_names_too(self, frame: pd.DataFrame) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = (
                AdaptiveIPTW(
                    treatment_learner=LogisticRegression(max_iter=1000),
                    reduced_outcome_learner=LinearRegression(),
                    estimands="all",
                    cross_fit=False,
                    simultaneous=False,
                )
                .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"])
                .single()
            )
        assert {"ate", "ey1", "ey0"} <= set(result.estimates)
        assert "att" not in result.estimates and "atc" not in result.estimates

    @pytest.mark.parametrize("estimands", [None, "all"])
    def test_the_regime_axis_resolves_to_the_two_regime_targets(self, estimands: Any) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = s.fit_arm(
                interventions=[Static(0, name="never"), Static(1, name="always")],
                estimands=None if estimands is None else estimands,
            )
        stems = {name.partition("[")[0] for name in result.estimates}
        assert stems == {"ey_regime", "ate_regime"}


# --------------------------------------------------------------------- cross-fit scale


class TestCrossFittedContinuousOutcome:
    def test_a_cross_fitted_continuous_outcome_fits_without_q_bounds(
        self, frame: pd.DataFrame
    ) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = (
                AdaptiveIPTW(
                    treatment_learner=LogisticRegression(max_iter=1000),
                    reduced_outcome_learner=LinearRegression(),
                    cross_fit=True,
                    n_folds=2,
                    simultaneous=False,
                    random_state=0,
                )
                .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"])
                .single()
            )
        assert np.isfinite(result.estimates["ate"].psi)


# ------------------------------------------------------------------ keyword inventory


def _base(**overrides: Any) -> AdaptiveIPTW:
    settings: dict[str, Any] = {
        "treatment_learner": LogisticRegression(max_iter=1000),
        "reduced_outcome_learner": LinearRegression(),
        "cross_fit": False,
        "simultaneous": False,
        "random_state": 0,
        "estimands": ["ey1", "ey0", "ate"],
    }
    settings.update(overrides)
    return AdaptiveIPTW(**settings)


def _run(estimator: AdaptiveIPTW, frame: pd.DataFrame, **fit: Any) -> Any:
    fit.setdefault("covariates", ["W1", "W2"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator.fit(frame, outcome="Y", treatment="A", **fit).single()


class TestKeywordInventoryApplies:
    def test_outcome_learner_is_the_default_reduced_learner(self, frame: pd.DataFrame) -> None:
        s.SpyRegressor.calls = []
        _run(_base(reduced_outcome_learner=None, outcome_learner=s.SpyRegressor()), frame)
        assert len(s.SpyRegressor.calls) > 0

    def test_the_treatment_learner_is_the_one_that_fits_the_mechanism(
        self, frame: pd.DataFrame
    ) -> None:
        a = _run(_base(), frame)
        b = _run(_base(treatment_learner=LogisticRegression(C=1e-3, max_iter=1000)), frame)
        assert abs(a.estimates["ey1"].psi - b.estimates["ey1"].psi) > 1e-6

    def test_g_bounds_clip_the_mechanism_and_move_the_estimate(self, frame: pd.DataFrame) -> None:
        wide = _run(_base(g_bounds=(0.001, 0.999)), frame)
        tight = _run(_base(g_bounds=(0.4, 0.6)), frame)
        assert abs(wide.estimates["ey1"].psi - tight.estimates["ey1"].psi) > 1e-3

    def test_max_iter_caps_the_targeting_loop(self, frame: pd.DataFrame) -> None:
        capped = _run(_base(max_iter=1, tol=1e-14), frame)
        free = _run(_base(tol=1e-12), frame)
        assert capped.extra["adaptive_iptw"].stop == "cap"
        assert free.extra["adaptive_iptw"].stop == "tolerance"
        assert capped.extra["adaptive_iptw"].rounds <= 1

    def test_tol_sets_the_stopping_point(self, frame: pd.DataFrame) -> None:
        loose = _run(_base(tol=1e-2), frame)
        tight = _run(_base(tol=1e-12), frame)
        assert loose.extra["adaptive_iptw"].rounds < tight.extra["adaptive_iptw"].rounds

    def test_random_state_fixes_the_cross_fit_split(self, frame: pd.DataFrame) -> None:
        def fit(seed: int) -> Any:
            return _run(_base(cross_fit=True, n_folds=2, random_state=seed), frame)

        same = fit(1), fit(1)
        other = fit(2)
        assert same[0].estimates["ey1"].psi == same[1].estimates["ey1"].psi
        assert abs(same[0].estimates["ey1"].psi - other.estimates["ey1"].psi) > 1e-9

    def test_n_folds_sets_the_split(self, frame: pd.DataFrame) -> None:
        two = _run(_base(cross_fit=True, n_folds=2), frame)
        three = _run(_base(cross_fit=True, n_folds=3), frame)
        assert two.config.n_folds == 2 and three.config.n_folds == 3

    def test_estimands_selects_the_reported_targets(self, frame: pd.DataFrame) -> None:
        result = _run(_base(estimands=["ey1"]), frame)
        assert set(result.estimates) == {"ey1"}

    def test_alpha_sig_sets_the_interval_of_a_known_mechanism(self) -> None:
        from tests.unit import _known_mechanism_support as known

        data = known.point_frame()
        probabilities = known.mechanism(data)

        def width(alpha: float) -> float:
            result = _run(
                _base(alpha_sig=alpha),
                data,
                covariates=["W1", "W2"],
                treatment_probabilities=probabilities,
            )
            low, high = result.estimates["ey1"].ci
            return float(high - low)

        assert width(0.01) > width(0.1) > 0.0

    def test_simultaneous_builds_a_band_when_every_estimate_is_inferential(self) -> None:
        from tests.unit import _known_mechanism_support as known

        data = known.point_frame()
        result = _run(
            _base(simultaneous=True),
            data,
            covariates=["W1", "W2"],
            treatment_probabilities=known.mechanism(data),
        )
        assert result.simultaneous is not None

    def test_a_binary_outcome_regresses_with_a_probability(self) -> None:
        """The binary task clips ``Qr`` to ``[0, 1]``, so a mean cannot leave the unit interval."""
        data, _ = make_binary_outcome(n=300, seed=17)
        result = _run(_base(estimands=["ey1", "ey0"], family="binomial"), data)
        for name in ("ey1", "ey0"):
            assert 0.0 < result.estimates[name].psi < 1.0

    def test_screen_treatment_runs_the_screen(self, frame: pd.DataFrame) -> None:
        result = _run(_base(screen_treatment=True, screen_threshold=0.5), frame)
        assert np.isfinite(result.estimates["ey1"].psi)

    def test_nuisance_bound_applies_to_the_observation_factors(self) -> None:
        data, _ = make_missing_outcome(n=300, seed=4)
        loose = _run(
            _base(
                estimands=["ey1"], nuisance_bound=0.001, missingness_learner=LogisticRegression()
            ),
            data,
            delta="Delta",
        )
        assert np.isfinite(loose.estimates["ey1"].psi)

    def test_run_id_and_n_jobs_do_not_change_the_estimate(self, frame: pd.DataFrame) -> None:
        base = _run(_base(), frame)
        other = _run(_base(run_id="x21", n_jobs=1), frame)
        assert base.estimates["ey1"].psi == other.estimates["ey1"].psi

    def test_stratified_folds_meet_the_shipped_fold_policy_refusal(
        self, frame: pd.DataFrame
    ) -> None:
        with pytest.raises(MethodConfigurationError, match="stratify_folds='treatment'"):
            _run(_base(cross_fit=True, n_folds=2, stratify_folds="treatment"), frame)

    def test_learner_folds_applies_to_a_library_learner(self, frame: pd.DataFrame) -> None:
        result = _run(_base(learner_folds=3), frame)
        assert np.isfinite(result.estimates["ey1"].psi)

    def test_reference_sets_the_contrast_reference(self, frame: pd.DataFrame) -> None:
        result = _run(_base(reference=1), frame)
        assert "ate" in result.estimates

    def test_min_retain_applies_with_the_screen(self, frame: pd.DataFrame) -> None:
        result = _run(_base(screen_treatment=True, min_retain=1), frame)
        assert np.isfinite(result.estimates["ey1"].psi)

    def test_treatment_covariates_restricts_the_mechanism_to_its_subset(
        self, frame: pd.DataFrame
    ) -> None:
        full = _run(_base(), frame)
        restricted = _run(_base(), frame, treatment_covariates=["W1"])
        assert abs(full.estimates["ey1"].psi - restricted.estimates["ey1"].psi) > 1e-9
