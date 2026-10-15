"""The support predicate and the statuses of the stacked per-arm interval (X58).

:func:`~cleverly.estimators.ctmle.per_arm_parametric_support` admits an in-sample per-arm
outcome-adaptive fit with :class:`~cleverly.learners.ParametricOutcome` and
:class:`~cleverly.learners.ParametricMechanism`. An admitted fit reports the stacked curve of
:mod:`cleverly.estimators._oat_stack` under ``"influence_curve"``. A failed arm stack stamps
``"stack_failure_plugin"`` on each estimate that reads the arm. Every other per-arm fit keeps
``"generated_design_plugin"``.
"""

from __future__ import annotations

import inspect
import warnings
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import cleverly
from cleverly.estimators import CTMLE, TMLE, _oat_stack
from cleverly.estimators import ctmle as ctmle_module
from cleverly.estimators.ctmle import (
    PER_ARM_PARAMETRIC_OPTIONS,
    SupportVerdict,
    per_arm_parametric_support,
)
from cleverly.exceptions import CapabilityError, DataWarning
from cleverly.learners import ParametricMechanism, ParametricOutcome
from tests.studies import oat_parametric_quadrature as quad
from tests.studies import oat_per_arm_laws as laws

pytestmark = pytest.mark.xdist_group("oat_per_arm_support")

ESTIMANDS = ("ey0", "ey1", "ate", "rr", "or")


def _options(**overrides: Any) -> dict[str, Any]:
    options: dict[str, Any] = {
        "strategy": "oat",
        "outcome_learner": ParametricOutcome(),
        "treatment_learner": ParametricMechanism(),
        "cross_fit": False,
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


def _prepared(estimator: CTMLE, frame: pd.DataFrame | None = None, **roles: Any) -> Any:
    frame = _frame() if frame is None else frame
    return estimator._prepare(
        frame,
        outcome="Y",
        treatment="A",
        covariates=roles.pop("covariates", ["W1", "W2", "W3"]),
        delta=roles.pop("delta", None),
        weights=roles.pop("weights", None),
        id=roles.pop("id", None),
        intermediate=roles.pop("intermediate", None),
        strata=roles.pop("strata", None),
        **roles,
    )


def _fit(frame: pd.DataFrame | None = None, **overrides: Any) -> Any:
    frame = _frame() if frame is None else frame
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            CTMLE(**_options(**overrides))
            .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2", "W3"])
            .single()
        )


@pytest.fixture(scope="module")
def admitted_fit() -> Any:
    return _fit()


class _SubclassedOutcome(ParametricOutcome):
    """A subclass the predicate refuses, because its ``fit`` could differ."""


class TestTheAllowList:
    def test_every_constructor_option_has_an_entry(self) -> None:
        names = {
            name
            for cls in (TMLE, CTMLE)
            for name, parameter in inspect.signature(cls.__init__).parameters.items()
            if name != "self" and parameter.kind is not inspect.Parameter.VAR_KEYWORD
        }
        assert names == set(PER_ARM_PARAMETRIC_OPTIONS)

    def test_the_default_options_are_admitted(self) -> None:
        estimator = CTMLE(**_options())
        verdict = per_arm_parametric_support(estimator, _prepared(estimator))
        assert verdict == SupportVerdict(True, None, verdict.detail)
        assert "ParametricOutcome" in verdict.detail

    def test_an_unfitted_estimator_reads_only_its_configuration(self) -> None:
        estimator = CTMLE(**_options())
        data = _prepared(estimator)
        before = dict(vars(estimator))
        assert per_arm_parametric_support(estimator, data).admitted
        assert dict(vars(estimator)).keys() == before.keys()

    @pytest.mark.parametrize(
        ("option", "value"),
        [
            ("strategy", "greedy"),
            ("oat_design", "shared"),
            ("outcome_learner", LogisticRegression()),
            ("outcome_learner", _SubclassedOutcome()),
            ("treatment_learner", LogisticRegression()),
            ("fluctuation", "linear"),
            ("n_folds", 2),
            ("repeats", 2),
            ("simultaneous", True),
            ("target_weights", True),
            ("screen_treatment", True),
        ],
        ids=lambda v: v if isinstance(v, str) else type(v).__name__,
    )
    def test_each_refused_option_names_its_axis(self, option: str, value: Any) -> None:
        options = _options(**{option: value})
        if option in ("n_folds", "repeats"):
            options["cross_fit"] = True
        if option == "repeats":
            options["n_folds"] = 5
        estimator = CTMLE(**options)
        verdict = per_arm_parametric_support(estimator, _prepared(estimator))
        assert not verdict.admitted
        assert verdict.axis == option
        assert verdict.detail.startswith("the stacked interval needs")

    def test_the_refused_option_keeps_the_generated_design_status(self) -> None:
        result = _fit(outcome_learner=LogisticRegression())
        assert {estimate.inference for estimate in result.estimates.values()} == {
            "generated_design_plugin"
        }


class TestTheArmColumns:
    def test_a_mismatched_arm_column_count_is_refused(self) -> None:
        estimator = CTMLE(**_options(outcome_learner=ParametricOutcome(arm_columns=2)))
        verdict = per_arm_parametric_support(estimator, _prepared(estimator))
        assert not verdict.admitted
        assert verdict.axis == "outcome_learner"
        assert "arm_columns=1" in verdict.detail
        assert "got arm_columns=2" in verdict.detail

    def test_the_mismatched_fit_keeps_the_generated_design_status(self) -> None:
        result = _fit(outcome_learner=ParametricOutcome(arm_columns=2))
        assert {estimate.inference for estimate in result.estimates.values()} == {
            "generated_design_plugin"
        }


class TestTheDataAxes:
    def test_three_arms(self) -> None:
        frame = _frame()
        frame.loc[frame.index[:300], "A"] = 2
        estimator = CTMLE(**_options(estimands=("ey0", "ey1", "ey2")))
        assert per_arm_parametric_support(estimator, _prepared(estimator, frame)).axis == (
            "treatment"
        )

    def test_weights(self) -> None:
        frame = _frame().assign(w=1.0 + np.arange(1500) % 3)
        estimator = CTMLE(**_options())
        verdict = per_arm_parametric_support(estimator, _prepared(estimator, frame, weights="w"))
        assert verdict.axis == "weights"

    def test_cluster(self) -> None:
        frame = _frame().assign(cid=np.arange(1500) // 5)
        estimator = CTMLE(**_options())
        verdict = per_arm_parametric_support(estimator, _prepared(estimator, frame, id="cid"))
        assert verdict.axis == "cluster"

    def test_strata(self) -> None:
        frame = _frame().assign(S=np.arange(1500) % 2)
        estimator = CTMLE(**_options())
        verdict = per_arm_parametric_support(
            estimator, _prepared(estimator, frame, strata=["S"], covariates=["W1", "W2", "W3", "S"])
        )
        assert verdict.axis == "strata"

    def test_missing_outcome(self) -> None:
        frame = _frame()
        frame["D"] = (np.arange(1500) % 7 != 0).astype(int)
        frame.loc[frame["D"] == 0, "Y"] = np.nan
        estimator = CTMLE(**_options())
        verdict = per_arm_parametric_support(estimator, _prepared(estimator, frame, delta="D"))
        assert verdict.axis == "missing_outcome"

    def test_a_continuous_outcome_needs_q_bounds(self) -> None:
        frame = quad.sample(quad.BINARY_ACTIVE, 1500, 33, bounded=(2.0, 10.0))
        means = ("ey0", "ey1", "ate")
        bare = CTMLE(**_options(estimands=means))
        assert per_arm_parametric_support(bare, _prepared(bare, frame)).axis == "q_bounds"
        bounded = CTMLE(**_options(q_bounds=(2.0, 10.0), estimands=means))
        assert per_arm_parametric_support(bounded, _prepared(bounded, frame)).admitted

    def test_one_estimand_named_by_a_string(self) -> None:
        estimator = CTMLE(**_options(estimands="ate"))
        assert per_arm_parametric_support(estimator, _prepared(estimator)).admitted

    def test_an_unlisted_estimand(self) -> None:
        estimator = CTMLE(**_options(estimands=("ate", "ey_obs")))
        verdict = per_arm_parametric_support(estimator, _prepared(estimator))
        assert verdict.axis == "estimands"


class TestTheAdmittedFit:
    def test_every_estimate_supplies_inference(self, admitted_fit: Any) -> None:
        for name in ESTIMANDS:
            estimate = admitted_fit[name]
            assert estimate.inference == "influence_curve", name
            assert estimate.plugin_std_error == estimate.std_error

    def test_the_arm_means_read_the_stacked_curve(self, admitted_fit: Any) -> None:
        stack = _oat_stack.fitted_stack(admitted_fit)
        curves = _oat_stack.corrected_curves(stack.problem, stack.theta).curves
        n = admitted_fit.n
        for j, arm in enumerate(("ey0", "ey1")):
            expected = np.std(curves[:, j], ddof=1) / np.sqrt(n)
            assert admitted_fit[arm].std_error == pytest.approx(expected, rel=1e-9)

    def test_w6_the_ate_is_the_arm_difference(self, admitted_fit: Any) -> None:
        """W6: the ATE curve is the difference of the arm curves, through every reader."""
        ate = admitted_fit["ate"]
        ey0, ey1 = admitted_fit["ey0"], admitted_fit["ey1"]
        assert ate.psi == pytest.approx(ey1.psi - ey0.psi, abs=1e-12)
        np.testing.assert_allclose(
            ate.influence_curve, ey1.influence_curve - ey0.influence_curve, atol=1e-12
        )
        cov = admitted_fit.covariance(["ey0", "ey1"])
        sd = np.sqrt(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
        assert ate.std_error == pytest.approx(sd, rel=1e-9)
        contrast = admitted_fit.contrast(lambda p: p[1] - p[0], ["ey0", "ey1"])
        assert contrast.std_error == pytest.approx(ate.std_error, rel=1e-9)

    def test_w8_the_ratios_take_the_delta_method(self, admitted_fit: Any) -> None:
        """W8: ``rr`` and ``or`` are delta-method transforms of the stacked arm curves."""
        ic0 = admitted_fit["ey0"].influence_curve
        ic1 = admitted_fit["ey1"].influence_curve
        p0, p1 = admitted_fit["ey0"].psi, admitted_fit["ey1"].psi
        n = admitted_fit.n
        log_rr = ic1 / p1 - ic0 / p0
        log_or = ic1 / (p1 * (1.0 - p1)) - ic0 / (p0 * (1.0 - p0))
        for name, curve in (("rr", log_rr), ("or", log_or)):
            se = np.std(curve, ddof=1) / np.sqrt(n)
            lower, upper = admitted_fit[name].ci
            point = admitted_fit[name].psi
            assert np.log(upper / point) == pytest.approx(1.959963984540054 * se, rel=1e-6)
            assert np.log(point / lower) == pytest.approx(1.959963984540054 * se, rel=1e-6)

    def test_a_pipeline_fit_keeps_the_generated_design_status(self) -> None:
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        result = _fit(
            outcome_learner=make_pipeline(StandardScaler(), LogisticRegression()),
            treatment_learner=make_pipeline(StandardScaler(), LogisticRegression()),
        )
        assert result["ate"].inference == "generated_design_plugin"

    def test_a_bounded_outcome_is_admitted(self) -> None:
        frame = quad.sample(quad.BINARY_ACTIVE, 1500, 33, bounded=(2.0, 10.0))
        result = _fit(frame, q_bounds=(2.0, 10.0), estimands=("ey0", "ey1", "ate"))
        assert result["ate"].inference == "influence_curve"

    def test_a_continuous_outcome_learner_is_refused(self) -> None:
        frame = quad.sample(quad.BINARY_ACTIVE, 1500, 33, bounded=(2.0, 10.0))
        result = _fit(
            frame,
            q_bounds=(2.0, 10.0),
            estimands=("ey0", "ey1", "ate"),
            outcome_learner=LinearRegression(),
        )
        assert result["ate"].inference == "generated_design_plugin"


def _failing(monkeypatch: pytest.MonkeyPatch, arms: set[int], reason: str = "forced") -> None:
    original = _oat_stack._arm_failure

    def mutant(stack: Any, rconds: dict[str, float], j: int) -> str | None:
        return reason if j in arms else original(stack, rconds, j)

    monkeypatch.setattr(_oat_stack, "_arm_failure", mutant)


class TestTheFailureStatus:
    def test_one_failed_arm_withholds_what_reads_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _failing(monkeypatch, {1})
        with pytest.warns(DataWarning, match="forced"):
            result = (
                CTMLE(**_options())
                .fit(_frame(), outcome="Y", treatment="A", covariates=["W1", "W2", "W3"])
                .single()
            )
        assert result["ey0"].inference == "influence_curve"
        for name in ("ey1", "ate", "rr", "or"):
            estimate = result[name]
            assert estimate.inference == "stack_failure_plugin", name
            assert not estimate.supplies_inference
            with pytest.raises(CapabilityError):
                _ = estimate.std_error
            assert np.isfinite(estimate.plugin_std_error)
        with pytest.raises(CapabilityError):
            _ = result.contrast(lambda p: p[1] - p[0], ["ey0", "ey1"]).ci

    def test_the_ey_target_holds_the_sound_arm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _failing(monkeypatch, {1})
        with pytest.warns(DataWarning, match="forced"):
            result = (
                CTMLE(**_options(estimands=("ey", "ate")))
                .fit(_frame(), outcome="Y", treatment="A", covariates=["W1", "W2", "W3"])
                .single()
            )
        assert {name: e.inference for name, e in result.estimates.items()} == {
            "ey[0]": "influence_curve",
            "ey[1]": "stack_failure_plugin",
            "ate": "stack_failure_plugin",
        }

    def test_a_failed_arm_reports_the_shipped_curve(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stack = _oat_stack.fitted_stack(_fit())
        shipped = _oat_stack.stack_report(stack, (0.0, 1.0)).shipped_std_error
        _failing(monkeypatch, {0, 1})
        result = _fit()
        for j, name in enumerate(("ey0", "ey1")):
            assert result[name].plugin_std_error == pytest.approx(shipped[float(j)], rel=1e-9)

    def test_non_convergence(self, admitted_fit: Any) -> None:
        stack = replace(_oat_stack.fitted_stack(admitted_fit), converged=False)
        report = _oat_stack.stack_report(stack, (0.0, 1.0))
        assert set(report.failures) == {0.0, 1.0}
        assert "convergence" in report.failures[0.0]

    def test_a_singular_block(self, admitted_fit: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_oat_stack, "MIN_BLOCK_RCOND", 2.0)
        report = _oat_stack.stack_report(_oat_stack.fitted_stack(admitted_fit), (0.0, 1.0))
        assert set(report.failures) == {0.0, 1.0}
        assert "beta block" in report.failures[1.0]

    def test_a_path_dependent_solve(self, admitted_fit: Any) -> None:
        stack = replace(_oat_stack.fitted_stack(admitted_fit), arm_mismatch=(0.0, 1e-6))
        report = _oat_stack.stack_report(stack, (0.0, 1.0))
        assert set(report.failures) == {1.0}
        assert "outer re-shrink" in report.failures[1.0]
        np.testing.assert_array_equal(report.corrections[1.0], np.zeros(admitted_fit.n))

    def test_a_sound_fit_has_no_failure(self, admitted_fit: Any) -> None:
        report = _oat_stack.stack_report(_oat_stack.fitted_stack(admitted_fit), (0.0, 1.0))
        assert report.failures == {}

    def test_the_status_is_a_parameter_status(self) -> None:
        from cleverly._inference_status import PARAMETER_STATUSES, STACK_FAILURE_STATUS

        assert STACK_FAILURE_STATUS in PARAMETER_STATUSES
        assert ctmle_module.STACK_FAILURE_STATUS == STACK_FAILURE_STATUS


class TestABlockWithNoSolve:
    def test_a_failed_arm_leaves_the_held_arm_finite(self, monkeypatch: pytest.MonkeyPatch) -> None:
        original = _oat_stack._equilibrated_rcond
        calls: list[int] = []

        def singular_gamma_one(diagonal: np.ndarray) -> float:
            calls.append(1)
            # The solve order is beta, gamma[0], eps[0], gamma[1]: the fourth call.
            return 0.0 if len(calls) % 5 == 4 else original(diagonal)

        monkeypatch.setattr(_oat_stack, "_equilibrated_rcond", singular_gamma_one)
        with pytest.warns(DataWarning, match="gamma\\[1\\]"):
            result = _fit_warning()
        assert result["ey0"].inference == "influence_curve"
        assert np.isfinite(result["ey0"].std_error)
        assert result["ate"].inference == "stack_failure_plugin"
        assert np.isfinite(result["ate"].plugin_std_error)


def _fit_warning() -> Any:
    return (
        CTMLE(**_options())
        .fit(_frame(), outcome="Y", treatment="A", covariates=["W1", "W2", "W3"])
        .single()
    )


class TestTheSummary:
    def test_an_admitted_fit_states_what_its_interval_covers(self, admitted_fit: Any) -> None:
        text = admitted_fit.summary()
        assert "stacked parametric curve of ParametricOutcome(features='arm_interactions')" in text
        assert "covers the arm mean when the outcome model is correct" in text
        assert "No simultaneous band" in text

    def test_a_refused_per_arm_fit_names_the_unmet_requirement(self) -> None:
        text = _fit(simultaneous=True).summary()
        assert "stacked parametric curve" not in text
        assert "No stacked interval (X58): the stacked interval needs simultaneous=False" in text

    def test_the_refusal_names_the_parametric_route(self, admitted_fit: Any) -> None:
        result = _fit(outcome_learner=LogisticRegression())
        with pytest.raises(CapabilityError, match="per_arm_parametric_support"):
            _ = result["ate"].std_error

    def test_a_shared_design_fit_states_nothing(self) -> None:
        text = _fit(oat_design="shared", treatment_learner=LogisticRegression()).summary()
        assert "stacked" not in text

    def test_the_note_survives_a_round_trip(self, admitted_fit: Any, tmp_path: Any) -> None:
        loaded = cleverly.load(admitted_fit.save(tmp_path / "fit.pkl"))
        assert loaded.summary() == admitted_fit.summary()


class TestTheOutcomeScale:
    def test_a_bounded_outcome_reports_the_stacked_curve_on_its_scale(self) -> None:
        frame = quad.sample(quad.BINARY_ACTIVE, 1500, 33, bounded=(2.0, 10.0))
        result = _fit(frame, q_bounds=(2.0, 10.0), estimands=("ey0", "ey1", "ate"))
        stack = _oat_stack.fitted_stack(result)
        curves = _oat_stack.corrected_curves(stack.problem, stack.theta).curves
        expected = 8.0 * np.std(curves[:, 1] - curves[:, 0], ddof=1) / np.sqrt(result.n)
        assert result["ate"].std_error == pytest.approx(expected, rel=1e-9)


class TestRecomputation:
    def test_save_and_load_keep_the_stacked_interval(
        self, admitted_fit: Any, tmp_path: Any
    ) -> None:
        path = admitted_fit.save(tmp_path / "fit.pkl")
        loaded = cleverly.load(path)
        for name in ESTIMANDS:
            assert loaded[name].inference == "influence_curve"
            assert loaded[name].std_error == admitted_fit[name].std_error

    def test_a_refit_on_the_same_data_reproduces_the_interval(self, admitted_fit: Any) -> None:
        estimator = CTMLE(**_options())
        data = _prepared(estimator)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            again = estimator.refit(data)
        assert again["ate"].std_error == pytest.approx(admitted_fit["ate"].std_error, rel=1e-9)

    def test_a_retarget_at_binding_bounds_withholds_the_interval(self) -> None:
        """The stack reads the declared bounds, so a retarget that moves them fails the stack."""
        estimator = CTMLE(**_options())
        data = _prepared(estimator)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = estimator.fit(data).single()
            same, _ = estimator.retarget(data, fit.nuisance, estimands=("ate",))
            other, _ = estimator.retarget(
                data, fit.nuisance, estimands=("ate",), g_bounds=(0.2, 0.8)
            )
        assert same["ate"].std_error == pytest.approx(fit["ate"].std_error, rel=1e-9)
        assert other["ate"].inference == "stack_failure_plugin"

    def test_a_failed_fit_round_trips(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
        _failing(monkeypatch, {1})
        with pytest.warns(DataWarning):
            result = _fit_warning()
        loaded = cleverly.load(result.save(tmp_path / "failed.pkl"))
        assert loaded["ate"].inference == "stack_failure_plugin"
        assert loaded["ey0"].inference == "influence_curve"
        text = loaded.summary()
        assert text == result.summary()
        assert "rows withheld (ate, ey1, rr, or): stack_failure_plugin" in text

    @pytest.mark.parametrize("estimand", ["att", "atc"])
    def test_a_retarget_to_another_group_reads_no_stack(
        self, admitted_fit: Any, estimand: str
    ) -> None:
        estimator = CTMLE(**_options())
        data = _prepared(estimator)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = estimator.fit(data).single()
            out, _ = estimator.retarget(data, fit.nuisance, estimands=(estimand,))
        assert out[estimand].inference == "generated_design_plugin"

    def test_a_retarget_to_an_uncovered_estimand_withholds_every_interval(self) -> None:
        estimator = CTMLE(**_options())
        data = _prepared(estimator)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = estimator.fit(data).single()
            out, _ = estimator.retarget(data, fit.nuisance, estimands=("ate", "ey_obs", "par"))
        assert {estimate.inference for estimate in out.values()} == {"generated_design_plugin"}
