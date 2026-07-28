"""Clever covariates and the targeting step.

The central claims verified here:

* the clever covariates match the formulas in the literature, checked against values
  computed by hand rather than against the implementation itself;
* the Newton solver finds the true maximiser of the quasi-likelihood, checked against
  a brute-force grid search;
* the covariate and weighted forms of the fluctuation solve the *same* estimating
  equation, which is the property that makes ``target_weights`` a numerical choice
  rather than a statistical one;
* the one-step universal least-favorable submodel reaches the same solution as the
  iterative fluctuation -- two independent implementations agreeing on the same root.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from cleverly.exceptions import ConvergenceWarning
from cleverly.fluctuation import (
    InitialFit,
    Submodel,
    atc_submodel,
    att_submodel,
    mean_submodel,
    solve_fluctuation,
    solve_one_step,
    submodel_for,
    weighted_form,
)
from cleverly.utils.bounds import expit, logit


@pytest.fixture
def setting() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    n = 1200
    w = rng.normal(size=n)
    g1 = expit(0.6 * w)
    a = rng.binomial(1, g1).astype(float)
    q1 = expit(0.4 + w)
    q0 = expit(w - 0.6)
    qa = np.where(a == 1.0, q1, q0)
    y = rng.binomial(1, qa).astype(float)
    return {"w": w, "g1": g1, "a": a, "q1": q1, "q0": q0, "y": y}


def _flat_initial(n: int, value: float = 0.5) -> InitialFit:
    """A deliberately uninformative initial fit, so targeting has work to do."""
    return InitialFit(np.full(n, value), np.full(n, value), np.full(n, value))


class TestCleverCovariates:
    def test_mean_submodel_matches_the_hand_computed_formula(self, setting) -> None:
        a, g1 = setting["a"], setting["g1"]
        submodel = mean_submodel(a, g1)
        expected_one = a / g1
        expected_zero = (1.0 - a) / (1.0 - g1)
        assert np.allclose(submodel.observed[:, 1], expected_one)
        assert np.allclose(submodel.observed[:, 0], expected_zero)
        # Counterfactual columns drop the arm indicator: they are evaluated *at* the arm.
        assert np.allclose(submodel.at_one[:, 1], 1.0 / g1)
        assert np.allclose(submodel.at_zero[:, 0], 1.0 / (1.0 - g1))
        assert np.allclose(submodel.at_one[:, 0], 0.0)
        assert np.allclose(submodel.at_zero[:, 1], 0.0)

    def test_the_two_mean_columns_have_disjoint_support(self, setting) -> None:
        submodel = mean_submodel(setting["a"], setting["g1"])
        # This is what lets the weighted form use a single weight vector for both.
        assert np.all(submodel.observed[:, 0] * submodel.observed[:, 1] == 0.0)

    def test_missingness_divides_the_clever_covariate(self, setting) -> None:
        a, g1 = setting["a"], setting["g1"]
        pi = np.column_stack([np.full(a.shape[0], 0.8), np.full(a.shape[0], 0.6)])
        submodel = mean_submodel(a, g1, missingness=pi)
        assert np.allclose(submodel.observed[:, 1], a / (g1 * 0.6))
        assert np.allclose(submodel.observed[:, 0], (1.0 - a) / ((1.0 - g1) * 0.8))

    def test_att_reweights_controls_by_the_propensity_odds(self, setting) -> None:
        a, g1 = setting["a"], setting["g1"]
        q = float(a.mean())
        submodel = att_submodel(a, g1, q)
        expected = (a - (1.0 - a) * g1 / (1.0 - g1)) / q
        assert np.allclose(submodel.observed[:, 0], expected)
        # The treated arm needs no reweighting: the ATT conditions on A = 1.
        assert np.allclose(submodel.at_one[:, 0], 1.0 / q)

    def test_atc_mirrors_the_att(self, setting) -> None:
        a, g1 = setting["a"], setting["g1"]
        q = float(a.mean())
        submodel = atc_submodel(a, g1, q)
        expected = (a * (1.0 - g1) / g1 - (1.0 - a)) / (1.0 - q)
        assert np.allclose(submodel.observed[:, 0], expected)

    def test_att_covariate_changes_sign_across_arms(self, setting) -> None:
        a, g1 = setting["a"], setting["g1"]
        submodel = att_submodel(a, g1, float(a.mean()))
        values = submodel.observed[:, 0]
        assert np.all(values[a == 1.0] > 0)
        assert np.all(values[a == 0.0] < 0)

    def test_intermediate_selection_zeroes_non_matching_rows(self, setting) -> None:
        a, g1 = setting["a"], setting["g1"]
        n = a.shape[0]
        z = np.resize([0.0, 1.0], n)
        density = np.column_stack([np.full(n, 0.5), np.full(n, 0.4)])
        submodel = mean_submodel(
            a, g1, intermediate_density=density, selection=(z == 1.0).astype(float)
        )
        # Only units whose realised Z equals the targeted value contribute.
        assert np.all(submodel.observed[z == 0.0] == 0.0)
        assert np.allclose(submodel.observed[z == 1.0, 1], (a / (g1 * 0.4))[z == 1.0])

    def test_extreme_propensity_is_refused(self, setting) -> None:
        a = setting["a"]
        with pytest.raises(ValueError, match="strictly inside"):
            mean_submodel(a, np.zeros_like(a))

    def test_dispatch_covers_every_group(self, setting) -> None:
        a, g1 = setting["a"], setting["g1"]
        for group in ("mean", "att", "atc"):
            submodel = submodel_for(group, a, g1, treated_fraction=float(a.mean()))
            assert submodel.group == group
        with pytest.raises(ValueError, match="unknown target group"):
            submodel_for("nope", a, g1)  # type: ignore[arg-type]

    def test_conditional_groups_need_the_treated_fraction(self, setting) -> None:
        with pytest.raises(ValueError, match="needs treated_fraction"):
            submodel_for("att", setting["a"], setting["g1"])

    def test_max_abs_reports_the_worst_weight(self, setting) -> None:
        submodel = mean_submodel(setting["a"], setting["g1"])
        assert submodel.max_abs == pytest.approx(np.abs(submodel.observed).max())

    def test_mismatched_shapes_are_refused(self) -> None:
        with pytest.raises(ValueError, match="mismatched shapes"):
            Submodel(np.zeros((5, 1)), np.zeros((4, 1)), np.zeros((5, 1)), ("h",), "mean")

    def test_name_count_must_match_columns(self) -> None:
        with pytest.raises(ValueError, match="name"):
            Submodel(np.zeros((5, 2)), np.zeros((5, 2)), np.zeros((5, 2)), ("h",), "mean")


class TestWeightedForm:
    def test_signs_and_magnitudes_factor_exactly(self, setting) -> None:
        submodel = mean_submodel(setting["a"], setting["g1"])
        weights = np.ones(submodel.n)
        signed, new_weights = weighted_form(submodel, weights)
        # sign(h) * |h| == h, so the score being solved is unchanged.
        reconstructed = signed.observed * new_weights[:, None]
        assert np.allclose(reconstructed, submodel.observed * weights[:, None])

    def test_att_sign_trick_preserves_the_score(self, setting) -> None:
        a, g1 = setting["a"], setting["g1"]
        submodel = att_submodel(a, g1, float(a.mean()))
        weights = np.ones(a.shape[0])
        signed, new_weights = weighted_form(submodel, weights)
        assert np.allclose(signed.observed[:, 0] * new_weights, submodel.observed[:, 0] * weights)


class TestNewtonSolver:
    def test_matches_a_brute_force_grid_search(self, setting) -> None:
        a, g1, y = setting["a"], setting["g1"], setting["y"]
        n = a.shape[0]
        initial = _flat_initial(n)
        submodel = att_submodel(a, g1, float(a.mean()))
        fitted = solve_fluctuation(y, initial, submodel, np.ones(n))

        offset = logit(np.clip(initial.observed, 1e-9, 1 - 1e-9))
        covariate = submodel.observed[:, 0]

        def quasi_loglik(epsilon: float) -> float:
            p = np.clip(expit(offset + covariate * epsilon), 1e-15, 1 - 1e-15)
            return float(np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

        grid = np.linspace(fitted.epsilon[0] - 0.4, fitted.epsilon[0] + 0.4, 40_001)
        best = grid[int(np.argmax([quasi_loglik(value) for value in grid]))]
        assert fitted.epsilon[0] == pytest.approx(best, abs=2e-5)

    def test_the_score_equation_is_solved(self, setting) -> None:
        a, g1, y = setting["a"], setting["g1"], setting["y"]
        n = a.shape[0]
        fitted = solve_fluctuation(y, _flat_initial(n), mean_submodel(a, g1), np.ones(n))
        assert fitted.converged
        assert fitted.score_norm < 1e-10

    def test_covariate_and_weighted_forms_solve_the_same_equation(self, setting) -> None:
        a, g1, y = setting["a"], setting["g1"], setting["y"]
        n = a.shape[0]
        submodel = mean_submodel(a, g1)
        plain = solve_fluctuation(y, _flat_initial(n), submodel, np.ones(n))
        weighted = solve_fluctuation(y, _flat_initial(n), submodel, np.ones(n), target_weights=True)
        # Different submodels, so different epsilon and slightly different Q*, but both
        # must drive the same estimating equation to zero.
        assert plain.score_norm < 1e-10
        assert weighted.score_norm < 1e-10
        psi_plain = plain.targeted.at_one.mean() - plain.targeted.at_zero.mean()
        psi_weighted = weighted.targeted.at_one.mean() - weighted.targeted.at_zero.mean()
        assert psi_plain == pytest.approx(psi_weighted, abs=5e-3)

    def test_a_correct_initial_fit_needs_almost_no_fluctuation(self, setting) -> None:
        a, g1, y = setting["a"], setting["g1"], setting["y"]
        q1, q0 = setting["q1"], setting["q0"]
        n = a.shape[0]
        initial = InitialFit(np.where(a == 1.0, q1, q0), q1, q0)
        fitted = solve_fluctuation(y, initial, mean_submodel(a, g1), np.ones(n))
        # With the truth plugged in, epsilon is pure sampling noise: O(1/sqrt(n)).
        assert np.max(np.abs(fitted.epsilon)) < 0.15
        assert fitted.score_norm < 1e-10

    def test_weights_shift_the_solution(self, setting) -> None:
        a, g1, y = setting["a"], setting["g1"], setting["y"]
        n = a.shape[0]
        rng = np.random.default_rng(1)
        weights = rng.uniform(0.3, 2.0, n)
        plain = solve_fluctuation(y, _flat_initial(n), mean_submodel(a, g1), np.ones(n))
        weighted = solve_fluctuation(y, _flat_initial(n), mean_submodel(a, g1), weights)
        assert not np.allclose(plain.epsilon, weighted.epsilon)
        assert weighted.score_norm < 1e-10

    def test_unobserved_rows_do_not_contribute(self, setting) -> None:
        a, g1, y = setting["a"], setting["g1"], setting["y"]
        n = a.shape[0]
        observed = np.ones(n, dtype=bool)
        observed[:200] = False
        corrupted = y.copy()
        corrupted[:200] = 999.0  # would wreck the fit if it were used
        fitted = solve_fluctuation(
            corrupted, _flat_initial(n), mean_submodel(a, g1), np.ones(n), observed
        )
        clean = solve_fluctuation(y, _flat_initial(n), mean_submodel(a, g1), np.ones(n), observed)
        assert np.allclose(fitted.epsilon, clean.epsilon)

    def test_all_missing_outcomes_is_refused(self, setting) -> None:
        a, g1, y = setting["a"], setting["g1"], setting["y"]
        n = a.shape[0]
        with pytest.raises(ValueError, match="no observed outcomes"):
            solve_fluctuation(
                y,
                _flat_initial(n),
                mean_submodel(a, g1),
                np.ones(n),
                np.zeros(n, dtype=bool),
            )

    def test_coefficients_are_reported_by_name(self, setting) -> None:
        a, g1, y = setting["a"], setting["g1"], setting["y"]
        n = a.shape[0]
        fitted = solve_fluctuation(y, _flat_initial(n), mean_submodel(a, g1), np.ones(n))
        assert set(fitted.coefficients()) == {"h0", "h1"}


class TestLinearFluctuation:
    def test_least_squares_solves_the_score_exactly(self, setting) -> None:
        a, g1, y = setting["a"], setting["g1"], setting["y"]
        n = a.shape[0]
        with pytest.warns(UserWarning, match="outside the outcome's range"):
            # The unbounded linear submodel is free to leave [0, 1], and does here --
            # which is exactly why the logistic fluctuation is the default.
            fitted = solve_fluctuation(
                y, _flat_initial(n), mean_submodel(a, g1), np.ones(n), kind="linear"
            )
        # The normal equations *are* the estimating equation, so one solve suffices.
        assert fitted.n_iter == 1
        assert fitted.score_norm < 1e-10
        assert fitted.method == "linear"

    def test_a_bounded_linear_fluctuation_does_not_warn(self, setting) -> None:
        a, y = setting["a"], setting["y"]
        n = a.shape[0]
        # Near-randomised treatment keeps the clever covariate close to 2, so the
        # linear update stays inside the unit interval.
        balanced = np.full(n, 0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            fitted = solve_fluctuation(
                y,
                InitialFit(np.full(n, 0.5), np.full(n, 0.5), np.full(n, 0.5)),
                mean_submodel(a, balanced),
                np.ones(n),
                kind="linear",
            )
        assert fitted.score_norm < 1e-10


class TestOneStep:
    def test_reaches_the_same_root_as_the_iterative_solver(self, setting) -> None:
        a, g1, y = setting["a"], setting["g1"], setting["y"]
        n = a.shape[0]
        submodel = mean_submodel(a, g1)
        iterative = solve_fluctuation(y, _flat_initial(n), submodel, np.ones(n))
        one_step = solve_one_step(y, _flat_initial(n), submodel, np.ones(n), step_size=2e-3)
        assert one_step.converged
        assert one_step.score_norm < 1e-9
        # Two independent paths to the same root.
        assert np.allclose(one_step.epsilon, iterative.epsilon, atol=5e-3)
        assert one_step.targeted.at_one.mean() == pytest.approx(
            iterative.targeted.at_one.mean(), abs=1e-4
        )

    def test_the_score_decreases_monotonically_along_the_path(self, setting) -> None:
        a, g1, y = setting["a"], setting["g1"], setting["y"]
        n = a.shape[0]
        fitted = solve_one_step(
            y, _flat_initial(n), mean_submodel(a, g1), np.ones(n), step_size=5e-3
        )
        trace = np.asarray(fitted.trace)
        # The universal least-favorable submodel rebuilds its direction each step, so the
        # score norm never has to increase.
        assert np.all(np.diff(trace) <= 1e-12)

    def test_records_the_method_and_step_count(self, setting) -> None:
        a, g1, y = setting["a"], setting["g1"], setting["y"]
        n = a.shape[0]
        fitted = solve_one_step(
            y, _flat_initial(n), mean_submodel(a, g1), np.ones(n), step_size=5e-3
        )
        assert fitted.method == "one_step"
        assert fitted.n_iter > 1

    def test_a_step_budget_that_is_too_small_warns(self, setting) -> None:
        a, g1, y = setting["a"], setting["g1"], setting["y"]
        n = a.shape[0]
        with pytest.warns(ConvergenceWarning, match="one-step targeting stopped"):
            solve_one_step(
                y,
                _flat_initial(n),
                mean_submodel(a, g1),
                np.ones(n),
                step_size=1e-4,
                max_steps=5,
            )

    def test_a_non_positive_step_is_refused(self, setting) -> None:
        a, g1, y = setting["a"], setting["g1"], setting["y"]
        n = a.shape[0]
        with pytest.raises(ValueError, match="step_size must be positive"):
            solve_one_step(y, _flat_initial(n), mean_submodel(a, g1), np.ones(n), step_size=0.0)


class TestInitialFit:
    def test_shrinking_keeps_the_logit_finite(self) -> None:
        fit = InitialFit(np.array([0.0]), np.array([1.0]), np.array([0.5]))
        shrunk = fit.shrunk(0.9995)
        assert shrunk.observed[0] == pytest.approx(0.0005)
        assert shrunk.at_one[0] == pytest.approx(0.9995)
        assert np.all(np.isfinite(logit(shrunk.observed)))

    def test_length_mismatch_is_refused(self, setting) -> None:
        a, g1, y = setting["a"], setting["g1"], setting["y"]
        with pytest.raises(ValueError, match="same length"):
            solve_fluctuation(y, _flat_initial(10), mean_submodel(a, g1), np.ones(a.shape[0]))
