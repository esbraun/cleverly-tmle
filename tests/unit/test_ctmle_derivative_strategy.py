r"""C-TMLE0 through ``CTMLE(strategy="derivative")``: the sources, the status and the reuse paths.

The seam itself is tested in ``test_ctmle_derivative.py`` and against R in
``test_ctmle_derivative_witness.py``.  This file covers what the strategy adds around it.

* W6: the internal penalty is the weighted-deviance cross-validation choice, recomputed here
  with independent single-penalty fits, and the partner is its grid neighbour.
* W7: the mechanism and the partner are fitted on the same rows, the training rows of each
  outer fold.
* W8: every refusal of the strategy fires.
* W9: the fit takes the ordinary TMLE status.
* W10: ``retarget`` rebuilds the derivative columns at the requested bounds, and the partner
  survives save and load.
* W11: the omitted-variable bound admits the fit; array entries refuse every refit.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from cleverly import CollaborativeTMLEMethod
from cleverly.estimators import CTMLE, TMLE, CTMLEDerivativeFit, PropensityArrays, logistic_plugin
from cleverly.estimators.serialize import dumps, loads
from cleverly.exceptions import CapabilityError
from cleverly.learners._l1_logistic import (
    L1ConvergenceError,
    L1Logistic,
    cv_penalty,
    default_penalty_grid,
    nearest_neighbour,
)
from cleverly.learners.crossfit import make_folds
from cleverly.sensitivity.omitted_variable import benchmark, omitted_variable_bounds
from cleverly.validation.nuisance import nuisance_diagnostics
from cleverly.variable_importance import variable_importance
from tests.studies.ctmle_derivative_laws import CLUSTERED, PRIMARY, THREE_ARM

COVARIATES = ["W1", "W2", "W3", "W4"]
PAIR = (0.01, 0.03)
OUTCOME = LogisticRegression(C=1e6, max_iter=1000)
TREATMENT = LogisticRegression(max_iter=1000)


def _settings(**overrides: Any) -> dict[str, Any]:
    return {
        "strategy": "derivative",
        "outcome_learner": OUTCOME,
        "treatment_learner": TREATMENT,
        "estimands": ["ate"],
        "cross_fit": False,
        "random_state": 7,
        "lambdas": PAIR,
        **overrides,
    }


def _fit(frame: pd.DataFrame, **overrides: Any) -> Any:
    return (
        CTMLE(**_settings(**overrides))
        .fit(frame, outcome="Y", treatment="A", covariates=COVARIATES)
        .single()
    )


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return PRIMARY.sample(600, 11)


@pytest.fixture(scope="module")
def lasso_fit(frame: pd.DataFrame) -> Any:
    return _fit(frame)


class TestPartnerRule:
    def test_the_partner_is_the_next_smaller_penalty_and_the_last_ones_neighbour(
        self, frame: pd.DataFrame
    ) -> None:
        grid = default_penalty_grid(
            frame[COVARIATES].to_numpy(), frame["A"].to_numpy(), np.ones(len(frame))
        )
        assert grid.size == 101 and grid[-1] == 0.0
        for index in (0, 37, 98):
            assert nearest_neighbour(grid, index) == index + 1
        # The smallest positive penalty is nearer its larger neighbour than the appended 0.
        assert nearest_neighbour(grid, 99) == 98
        assert nearest_neighbour(grid, 100) == 99

    def test_a_tie_goes_to_the_larger_penalty(self) -> None:
        assert nearest_neighbour([3.0, 2.0, 1.0], 1) == 0

    def test_cv_penalty_equals_independent_single_penalty_fits(self, frame: pd.DataFrame) -> None:
        x = frame[COVARIATES].to_numpy()
        a = frame["A"].to_numpy()
        w = np.random.default_rng(0).uniform(0.5, 2.0, len(frame))
        grid = default_penalty_grid(x, a, w)[::10]
        assignment = make_folds(len(frame), 4, random_state=3).assignment
        index, risk = cv_penalty(x, a, w, grid, assignment, classes=[0.0, 1.0])

        expected = np.zeros(grid.size)
        for fold in range(4):
            held = assignment == fold
            for k, penalty in enumerate(grid):
                model = L1Logistic(penalty).fit(x[~held], a[~held], sample_weight=w[~held])
                p1 = model.predict_proba(x[held])[:, 1]
                likelihood = np.where(a[held] == 1.0, p1, 1.0 - p1)
                expected[k] -= np.sum(w[held] * np.log(likelihood))
        expected /= w.sum()
        np.testing.assert_allclose(risk, expected, rtol=1e-6)
        assert index == int(np.argmin(expected))
        # The training deviance would choose the smallest penalty; the held-out one does not.
        assert index < grid.size - 1

    def test_the_fit_uses_the_cross_validated_pair(self, frame: pd.DataFrame) -> None:
        result = _fit(frame, lambdas=None, learner_folds=4)
        fit = result.extra["ctmle"]
        x = frame[COVARIATES].to_numpy()
        a = frame["A"].to_numpy()
        grid = default_penalty_grid(x, a, np.ones(len(frame)))
        seed = result.repeats[0].seed
        assignment = make_folds(len(frame), 4, random_state=seed).assignment
        index, risk = cv_penalty(x, a, None, grid, assignment, classes=[0.0, 1.0])
        assert fit.source == "lasso"
        assert fit.penalties == ((grid[index], grid[nearest_neighbour(grid, index)]),)
        np.testing.assert_array_equal(fit.cv_risk[0], risk)
        assert fit.treatment_risk == pytest.approx(risk[index])

    def test_a_cross_fitted_fit_chooses_per_training_set(self, frame: pd.DataFrame) -> None:
        fit = _fit(frame, lambdas=None, cross_fit=True, n_folds=3, learner_folds=3).extra["ctmle"]
        assert len(fit.penalties) == 3 and len(fit.cv_risk) == 3
        assert "over 3 training sets" in fit.describe()


class _Spy(ClassifierMixin, BaseEstimator):
    """A logistic learner that records the rows it was fitted on."""

    log: ClassVar[list[tuple[str, frozenset[float]]]] = []

    def __init__(self, tag: str = "", c: float = 1.0) -> None:
        self.tag = tag
        self.c = c

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> _Spy:
        self.model_ = LogisticRegression(C=self.c, max_iter=1000).fit(X, y, sample_weight)
        self.classes_ = self.model_.classes_
        _Spy.log.append((self.tag, frozenset(np.round(np.asarray(X)[:, 0], 12))))
        return self

    def predict_proba(self, X: Any) -> Any:
        return self.model_.predict_proba(X)


def test_w7_the_mechanism_and_the_partner_see_the_same_training_rows(
    frame: pd.DataFrame,
) -> None:
    _Spy.log.clear()
    result = _fit(
        frame,
        lambdas=None,
        cross_fit=True,
        n_folds=3,
        treatment_candidates=[_Spy("g", 0.05), _Spy("plus", 1.0)],
    )
    first = frame["W1"].to_numpy()
    training = [frozenset(np.round(first[train], 12)) for train, _ in result.nuisance.folds]
    mechanism = [rows for tag, rows in _Spy.log if tag == "g"]
    partner = [rows for tag, rows in _Spy.log if tag == "plus"]
    assert sorted(map(sorted, mechanism)) == sorted(map(sorted, partner))
    assert sorted(map(sorted, mechanism)) == sorted(map(sorted, training))
    assert result.extra["ctmle"].source == "learners"


class TestRefusals:
    @pytest.mark.parametrize(
        ("settings", "error", "pattern"),
        [
            (
                {"lambdas": None, "strategy": "greedy", "treatment_candidates": [TREATMENT]},
                ValueError,
                "treatment_candidates= only applies to strategy='learners' or "
                "'derivative', not 'greedy'",
            ),
            (
                {"strategy": "ordered", "lambdas": [0.1]},
                ValueError,
                "lambdas= only applies to strategy='lasso' or 'derivative', not 'ordered'",
            ),
            (
                {"treatment_candidates": [TREATMENT, TREATMENT]},
                ValueError,
                "from lambdas= or from treatment_candidates=, not from both",
            ),
            ({"lambdas": (0.1,)}, ValueError, "is the pair (mechanism_penalty, partner_penalty)"),
            ({"lambdas": (0.1, 0.1)}, ValueError, "two finite, non-negative and different"),
            ({"lambdas": (0.1, -0.1)}, ValueError, "Got (0.1, -0.1)"),
            ({"lambdas": (0.1, np.nan)}, ValueError, "Omit it to choose the mechanism penalty"),
            (
                {"lambdas": None, "treatment_candidates": [TREATMENT]},
                ValueError,
                "exactly two entries, each an unfitted learner or a full-sample propensity "
                "array. Got 1 entries",
            ),
            (
                {"lambdas": None, "treatment_candidates": [TREATMENT] * 3},
                ValueError,
                "Got 3 entries",
            ),
            (
                {
                    "lambdas": None,
                    "treatment_candidates": [PropensityArrays([0.5], [0.5]), TREATMENT],
                },
                CapabilityError,
                "entry 0 is a PropensityArrays pair",
            ),
            (
                {
                    "lambdas": None,
                    "treatment_candidates": [np.full(5, 0.5), TREATMENT],
                    "cross_fit": True,
                },
                CapabilityError,
                "needs one fit for each outer fold",
            ),
            (
                {
                    "lambdas": None,
                    "treatment_candidates": [np.full(5, 0.5), TREATMENT],
                    "n_bootstrap": 10,
                },
                CapabilityError,
                "a bootstrap replicate refits",
            ),
            ({"targeting": "one_step"}, CapabilityError, "targeting='one_step' walks"),
            ({"target_weights": True}, CapabilityError, "keeps only one of the two scores"),
            (
                {"selection_folds": 3},
                ValueError,
                "selection_folds= configure selector strategies and do not apply to "
                "strategy='derivative', which fits one mechanism and its partner",
            ),
            (
                {"ctmle_estimand": "ey"},
                ValueError,
                "ctmle_estimand= does not apply to strategy='derivative'",
            ),
            (
                {"oat_design": "shared"},
                ValueError,
                "oat_design= only applies to strategy='oat', not 'derivative'",
            ),
        ],
    )
    def test_construction(self, settings: dict[str, Any], error: type, pattern: str) -> None:
        with pytest.raises(error, match=re.escape(pattern)):
            CTMLE(**_settings(**settings))

    def test_an_array_of_the_wrong_shape(self, frame: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match=re.escape("entry 1: the array must hold 600")):
            _fit(frame, lambdas=None, treatment_candidates=[TREATMENT, np.full(599, 0.5)])

    def test_a_partner_equal_to_the_mechanism(self, frame: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="its derivative equation is empty"):
            _fit(frame, lambdas=None, treatment_candidates=[TREATMENT, clone(TREATMENT)])

    def test_two_constant_learners_give_a_collinear_column(self, frame: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="repeats its clever covariate"):
            _fit(
                frame,
                lambdas=None,
                treatment_candidates=[
                    DummyClassifier(strategy="prior"),
                    DummyClassifier(strategy="uniform"),
                ],
            )

    def test_a_solve_that_does_not_converge(self, frame: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="did not converge in 1 iterations"):
            _fit(frame, max_iter=1, tol=1e-300)

    def test_baseline_strata(self, frame: pd.DataFrame) -> None:
        stratified = frame.assign(S=(frame["W4"] > 0).astype(int))
        with pytest.raises(CapabilityError, match="has no stratified update"):
            CTMLE(**_settings()).fit(
                stratified,
                outcome="Y",
                treatment="A",
                covariates=[*COVARIATES, "S"],
                strata=["S"],
            )

    def test_the_logistic_plugin(self, lasso_fit: Any) -> None:
        with pytest.raises(CapabilityError, match=r"which std_error gives up to n/\(n-1\)"):
            logistic_plugin(lasso_fit)

    def test_l1_non_convergence_names_the_penalty_cv_fold(self) -> None:
        separated = PRIMARY.sample(300, 2)
        separated["A"] = (separated["W1"] > 0).astype(float)
        with pytest.raises(L1ConvergenceError, match="penalty cross-validation fold"):
            _fit(separated, lambdas=None, learner_folds=3)


class TestStatus:
    def test_the_fit_reports_the_tmle_interval(self, lasso_fit: Any) -> None:
        estimate = lasso_fit["ate"]
        assert lasso_fit.inference_status == "influence_curve"
        assert np.isfinite(estimate.std_error) and np.isfinite(estimate.pvalue)
        low, high = estimate.ci
        assert low < estimate.psi < high

    def test_three_arm_contrasts_report_intervals(self) -> None:
        result = (
            CTMLE(**_settings(estimands=["ate"]))
            .fit(THREE_ARM.sample(900, 4), outcome="Y", treatment="A", covariates=COVARIATES)
            .single()
        )
        assert len(result.extra["ctmle"].names) == 6
        for estimate in result.estimates.values():
            assert np.isfinite(estimate.std_error)

    def test_estimated_weights_take_the_tmle_status(self, frame: pd.DataFrame) -> None:
        weighted = frame.assign(wt=np.random.default_rng(1).uniform(0.5, 2.0, len(frame)))
        declared = {"weights": "wt", "weights_estimated": True}
        result = (
            CTMLE(**_settings())
            .fit(weighted, outcome="Y", treatment="A", covariates=COVARIATES, **declared)
            .single()
        )
        plain = (
            TMLE(outcome_learner=OUTCOME, treatment_learner=TREATMENT, cross_fit=False)
            .fit(weighted, outcome="Y", treatment="A", covariates=COVARIATES, **declared)
            .single()
        )
        assert result.inference_status == plain.inference_status

    def test_few_clusters_withhold_the_interval(self) -> None:
        clustered = CLUSTERED.sample(160, 5)
        result = (
            CTMLE(**_settings())
            .fit(clustered, outcome="Y", treatment="A", covariates=COVARIATES, id="cluster")
            .single()
        )
        assert result.inference_status == "few_cluster_plugin"

    def test_clusters_are_admitted(self) -> None:
        clustered = CLUSTERED.sample(400, 5)
        result = (
            CTMLE(**_settings(cross_fit=True, n_folds=3))
            .fit(clustered, outcome="Y", treatment="A", covariates=COVARIATES, id="cluster")
            .single()
        )
        assert result.inference_status == "influence_curve"

    def test_the_nuisance_report_reads_the_mechanism_as_the_treatment_law(
        self, lasso_fit: Any
    ) -> None:
        assert isinstance(lasso_fit.ctmle_selection, CTMLEDerivativeFit)
        report = nuisance_diagnostics(lasso_fit)
        assert report.treatment_role == "estimated_treatment_law"
        assert "penalty 0.01 with partner 0.03" in report.summary()


class TestRetarget:
    def test_a_retarget_equals_a_fresh_fit_at_the_new_bounds(
        self, frame: pd.DataFrame, lasso_fit: Any
    ) -> None:
        bounds = (0.1, 0.9)
        estimator = lasso_fit.estimator
        retargeted, _ = estimator.retarget(
            lasso_fit.data, lasso_fit.nuisance, estimands=["ate"], g_bounds=bounds
        )
        fresh = _fit(frame, g_bounds=bounds)
        assert retargeted["ate"].psi == pytest.approx(fresh["ate"].psi, abs=1e-10)
        # Without the partner the retarget continues from the fitted Q* only.
        continued, _ = estimator.retarget(
            lasso_fit.data,
            replace(lasso_fit.nuisance, treatment_partner=None),
            estimands=["ate"],
            g_bounds=bounds,
        )
        assert abs(continued["ate"].psi - retargeted["ate"].psi) > 1e-6

    def test_the_fit_retargets_to_its_own_estimate(self, lasso_fit: Any) -> None:
        estimates, _ = lasso_fit.estimator.retarget(
            lasso_fit.data,
            lasso_fit.nuisance,
            estimands=["ate"],
            g_bounds=lasso_fit.config.g_bounds,
        )
        assert estimates["ate"].psi == pytest.approx(lasso_fit["ate"].psi, abs=1e-12)

    @pytest.mark.parametrize("source", ["lasso", "learners", "arrays"])
    def test_the_partner_survives_save_and_load(self, frame: pd.DataFrame, source: str) -> None:
        if source == "lasso":
            result = _fit(frame)
        elif source == "learners":
            result = _fit(
                frame,
                lambdas=None,
                treatment_candidates=[LogisticRegression(C=0.05), LogisticRegression(C=1.0)],
            )
        else:
            g = PRIMARY.mechanism(frame[COVARIATES].to_numpy())[:, 1]
            result = _fit(frame, lambdas=None, treatment_candidates=[g, 0.5 + 0.8 * (g - 0.5)])
        loaded = loads(dumps(result))
        np.testing.assert_array_equal(
            loaded.nuisance.treatment_partner.values, result.nuisance.treatment_partner.values
        )
        assert loaded.extra["ctmle"].source == source
        np.testing.assert_array_equal(loaded.extra["ctmle"].epsilon, result.extra["ctmle"].epsilon)
        estimates, _ = loaded.estimator.retarget(
            loaded.data, loaded.nuisance, estimands=["ate"], g_bounds=(0.05, 0.95)
        )
        again, _ = result.estimator.retarget(
            result.data, result.nuisance, estimands=["ate"], g_bounds=(0.05, 0.95)
        )
        assert estimates["ate"].psi == again["ate"].psi


class TestReuse:
    def test_the_omitted_variable_bound_admits_the_fit(self, lasso_fit: Any) -> None:
        bounds = omitted_variable_bounds(lasso_fit, "ate")
        assert np.isfinite(bounds.robustness_value)

    def test_benchmark_refits_the_lasso_source(self, lasso_fit: Any) -> None:
        assert benchmark(lasso_fit, "W2", random_state=0) is not None

    def test_benchmark_refuses_arrays(self, frame: pd.DataFrame) -> None:
        g = PRIMARY.mechanism(frame[COVARIATES].to_numpy())[:, 1]
        result = _fit(frame, lambdas=None, treatment_candidates=[g, 0.5 + 0.8 * (g - 0.5)])
        with pytest.raises(CapabilityError, match="a propensity array entry was fitted"):
            benchmark(result, "W2", random_state=0)

    def test_variable_importance_refuses_arrays(self, frame: pd.DataFrame) -> None:
        g = PRIMARY.mechanism(frame[COVARIATES].to_numpy())[:, 1]
        estimator = CTMLE(
            **_settings(lambdas=None, treatment_candidates=[g, 0.5 + 0.8 * (g - 0.5)])
        )
        with pytest.raises(CapabilityError, match="Pass learners to refit"):
            variable_importance(
                frame,
                outcome="Y",
                candidates=["W1", "W2"],
                covariates=["W3", "W4"],
                estimator=estimator,
            )

    def test_variable_importance_admits_the_lasso_source(self, frame: pd.DataFrame) -> None:
        binary = frame.assign(B=(frame["W1"] > 0).astype(float))
        result = variable_importance(
            binary,
            outcome="Y",
            candidates=["A", "B"],
            covariates=["W2", "W3", "W4"],
            estimator=CTMLE(**_settings()),
        )
        assert len(result.entries) == 2

    def test_the_method_configuration_round_trips(self, frame: pd.DataFrame) -> None:
        method = CollaborativeTMLEMethod(strategy="derivative", lambdas=PAIR)
        kwargs = method.estimator_kwargs()
        assert kwargs["strategy"] == "derivative" and kwargs["lambdas"] == PAIR
        CTMLE(**{**kwargs, "outcome_learner": OUTCOME, "treatment_learner": TREATMENT})

    def test_the_tmle_without_derivative_columns_differs(
        self, frame: pd.DataFrame, lasso_fit: Any
    ) -> None:
        # The same g and Q without the derivative columns: the ordinary TMLE update.
        plain, _ = TMLE.retarget(
            lasso_fit.estimator,
            lasso_fit.data,
            replace(lasso_fit.nuisance, treatment_partner=None, targeting_outcome=None),
            estimands=["ate"],
            g_bounds=lasso_fit.config.g_bounds,
        )
        assert plain["ate"].psi != lasso_fit["ate"].psi


def test_both_mechanisms_are_bounded_before_the_ratio(frame: pd.DataFrame) -> None:
    # The mechanism is clipped at both bounds on some rows, and the partner crosses a bound
    # on rows where the mechanism does not. Each bound changes the derivative columns there.
    from cleverly.estimators._ctmle_derivative import derivative_update
    from cleverly.estimators.targeting import TargetingSpec
    from cleverly.fluctuation.submodel import mean_submodel

    index = 2.0 * frame["W1"].to_numpy() + 0.8 * frame["W2"].to_numpy()
    g = 1.0 / (1.0 + np.exp(-index))
    plus = 1.0 / (1.0 + np.exp(-1.6 * index))
    bounds = (0.05, 0.95)
    inside = (g > bounds[0]) & (g < bounds[1])
    assert np.sum(g < bounds[0]) >= 10 and np.sum(g > bounds[1]) >= 10
    assert np.sum(inside & ((plus < bounds[0]) | (plus > bounds[1]))) >= 10
    result = _fit(frame, lambdas=None, treatment_candidates=[g, plus], g_bounds=bounds)

    a = frame["A"].to_numpy()
    clipped = np.column_stack([1.0 - np.clip(g, *bounds), np.clip(g, *bounds)])
    partner = np.column_stack([1.0 - np.clip(plus, *bounds), np.clip(plus, *bounds)])
    nuisance = result.nuisance
    expected = derivative_update(
        nuisance.scaler.scale(frame["Y"].to_numpy()),
        nuisance.outcome,
        mean_submodel(a, clipped, arms=(0.0, 1.0)),
        clipped,
        partner,
        a,
        np.ones(len(frame), dtype=bool),
        np.ones(len(frame)),
        TargetingSpec(alpha=result.estimator.alpha, tol=result.estimator.tol),
    )
    np.testing.assert_allclose(nuisance.targeting_outcome.observed, expected.targeted.observed)
    np.testing.assert_allclose(result.extra["ctmle"].epsilon, expected.epsilon, rtol=1e-10)
