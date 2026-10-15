"""Witnesses of the cross-fitted per-arm OAT stack (X58 plan, 13.3, CW0 to CW5).

The V-fold stack carries one outcome and one mechanism fit per fold, one pooled fluctuation
per arm and the whole-sample means. Each witness runs where the term it checks is material,
and each has a deliberate-mutation control that fails when that term is wrong. The mutations
patch the stack in memory (``monkeypatch``) or build a mutated input; no source file changes.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from scipy.special import expit

from cleverly.estimators import CTMLE
from cleverly.estimators import ctmle as ctmle_module
from cleverly.estimators._oat_stack import (
    ArmDesign,
    FoldCoefficients,
    FoldStackProblem,
    StackProblem,
    corrected_curves,
    fitted_stack,
)
from cleverly.estimators.targeting import build_submodel
from cleverly.inference.influence import counterfactual_means
from cleverly.learners import ParametricMechanism, ParametricOutcome
from tests.studies import oat_parametric_quadrature as quad
from tests.studies import oat_per_arm_laws as laws

pytestmark = pytest.mark.xdist_group("oat_crossfit_stack")

FLOOR = 1e-9
"""The absolute floor on a solved estimating equation's sample mean (C-solve at tol 1e-10)."""

CLIP_LAW = quad.QuadratureLaw(
    "clip_witness",
    lambda w: np.asarray(expit(0.5 * w[:, 0] + 0.8 * w[:, 2])),
    lambda a, w: np.asarray(expit(-1.0 + (6.0 + 2.0 * a) * w[:, 0] + 0.5 * w[:, 1])),
)
"""An outcome so steep in ``W1`` that the ``alpha`` clip, the outer re-shrink and the basis
clip all bind on a sample, with ``eps* != 0`` under the no-intercept design."""

ESTIMANDS = ("ey0", "ey1", "ate")


def _fit(
    frame: Any,
    *,
    features: str = "arm_interactions",
    g_bounds: tuple[float, float] = laws.G_BOUNDS,
    **options: Any,
) -> Any:
    settings: dict[str, Any] = {
        "strategy": "oat",
        "outcome_learner": ParametricOutcome(features),  # type: ignore[arg-type]
        "treatment_learner": ParametricMechanism(),
        "cross_fit": True,
        "n_folds": 5,
        "simultaneous": False,
        "estimands": ESTIMANDS,
        "g_bounds": g_bounds,
        "max_iter": 100,
        "tol": 1e-10,
        "random_state": 0,
    }
    settings.update(options)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            CTMLE(**settings)
            .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2", "W3"])
            .single()
        )


def _block_means(problem: FoldStackProblem, theta: np.ndarray) -> dict[str, float]:
    means = problem.mass @ problem.rows(theta)
    return {name: float(np.max(np.abs(means[part]))) for name, part in problem.slices.items()}


def _fit_shipped(result: Any, draw: int = 0) -> np.ndarray:
    """The fit's uncorrected curves, from ``counterfactual_means`` without corrections."""
    repeat = result.repeats[draw]
    bounds = tuple(float(value) for value in result.config.g_bounds)
    submodel = build_submodel(
        result.data,
        repeat.nuisance,
        "mean",
        bounds=(bounds[0], bounds[1]),
        nuisance_bound=float(result.estimator.nuisance_bound),
    )
    fluctuation = repeat.fluctuations["mean"]
    means = counterfactual_means(
        repeat.nuisance.scaler.scale(result.data.outcome),
        fluctuation.targeted,
        submodel,
        result.data.weights,
    )
    return np.column_stack([means[float(arm)].influence_curve for arm in result.data.arm_codes])


def _shipped_from_coefficients(stack: Any, fold: int | None) -> np.ndarray:
    """The shipped curve rebuilt from coefficients: each row's own fold, or one fold for all."""
    problem, theta = stack.problem, stack.theta
    targeted = problem.targeted(theta)
    out = np.empty_like(targeted)
    for j, design in enumerate(problem.arms):
        clever = np.zeros(problem.n)
        for k in range(problem.n_folds):
            beta = theta[problem.slices[problem.beta_name(k)]]
            gamma = theta[problem.slices[problem.gamma_name(j, k)]]
            q = expit(design.features @ beta)
            value = 1.0 / np.clip(expit(design.basis(q) @ gamma), *design.bounds)
            rows = problem.labels == k if fold is None else np.ones(problem.n, dtype=bool)
            if fold is None or k == fold:
                clever[rows] = value[rows]
        out[:, j] = (
            design.indicator * clever * (problem.outcome - targeted[:, j])
            + targeted[:, j]
            - float(np.mean(targeted[:, j]))
        )
    return out


@pytest.fixture(scope="module")
def unequal() -> Any:
    """``binary_active`` at ``n = 1,503`` and ``V = 5``: three folds of 301 rows, two of 300."""
    return _fit(laws.BINARY_ACTIVE.sample(1503, 61)[0])


@pytest.fixture(scope="module")
def in_sample() -> Any:
    return _fit(laws.BINARY_ACTIVE.sample(1503, 61)[0], cross_fit=False)


class TestCW0SolvedSystem:
    """CW0: the cross-fitted fit solved the V-fold system, block by block."""

    def test_the_folds_are_unequal(self, unequal: Any) -> None:
        sizes = np.bincount(fitted_stack(unequal).problem.labels)
        assert sorted(sizes.tolist()) == [300, 300, 301, 301, 301]

    def test_every_block_is_solved(self, unequal: Any) -> None:
        stack = fitted_stack(unequal)
        assert stack.problem.n_folds == 5
        means = _block_means(stack.problem, stack.theta)
        assert len(means) == 5 + 2 * (5 + 1) + 2
        assert max(means.values()) < FLOOR, means
        assert stack.targeted_mismatch < 1e-12

    @pytest.mark.parametrize("fixture", ["unequal", "in_sample"])
    def test_the_shipped_curve_is_the_fits(
        self, request: pytest.FixtureRequest, fixture: str
    ) -> None:
        """m2: the curve the correction is taken from is the one the estimate builder reads."""
        result = request.getfixturevalue(fixture)
        stack = fitted_stack(result)
        assert stack.shipped is not None
        assert np.max(np.abs(stack.shipped - _fit_shipped(result))) < 1e-14

    def test_the_coefficients_rebuild_the_shipped_curve(self, unequal: Any) -> None:
        """Each row's own fold coefficients rebuild the fit's clever covariate (W0's check)."""
        stack = fitted_stack(unequal)
        assert stack.shipped is not None
        assert np.max(np.abs(_shipped_from_coefficients(stack, None) - stack.shipped)) < 1e-10

    def test_mutation_whole_sample_beta(self, unequal: Any, in_sample: Any) -> None:
        stack = fitted_stack(unequal)
        problem = stack.problem
        whole = fitted_stack(in_sample).theta[fitted_stack(in_sample).problem.slices["beta"]]
        theta = stack.theta.copy()
        for k in range(problem.n_folds):
            theta[problem.slices[problem.beta_name(k)]] = whole
        means = _block_means(problem, theta)
        assert min(means[problem.beta_name(k)] for k in range(problem.n_folds)) > 1e-4

    def test_mutation_rotated_fold_coefficients(self, unequal: Any) -> None:
        stack = fitted_stack(unequal)
        problem = stack.problem
        theta = stack.theta.copy()
        for k in range(problem.n_folds):
            source = (k + 1) % problem.n_folds
            theta[problem.slices[problem.beta_name(k)]] = stack.theta[
                problem.slices[problem.beta_name(source)]
            ]
            for j in range(len(problem.arms)):
                theta[problem.slices[problem.gamma_name(j, k)]] = stack.theta[
                    problem.slices[problem.gamma_name(j, source)]
                ]
        means = _block_means(problem, theta)
        assert min(means[problem.beta_name(k)] for k in range(problem.n_folds)) > 1e-4

    def test_mutation_shipped_curve_from_fold_zero(self, unequal: Any) -> None:
        stack = fitted_stack(unequal)
        mutant = _shipped_from_coefficients(stack, 0)
        assert np.max(np.abs(mutant - _fit_shipped(unequal))) > 1e-3


def _central(problem: FoldStackProblem, theta: np.ndarray, step: float) -> np.ndarray:
    numeric = np.zeros((problem.dimension, problem.dimension))
    for k in range(theta.size):
        h = step * max(1.0, abs(theta[k]))
        up, down = theta.copy(), theta.copy()
        up[k] += h
        down[k] -= h
        numeric[:, k] = problem.mass @ (problem.rows(up) - problem.rows(down)) / (2.0 * h)
    return numeric


def _richardson(problem: FoldStackProblem, theta: np.ndarray, step: float) -> np.ndarray:
    return (4.0 * _central(problem, theta, step / 2.0) - _central(problem, theta, step)) / 3.0


def _fd_report(problem: FoldStackProblem, theta: np.ndarray) -> tuple[float, float]:
    """The largest block-relative Jacobian gap, and the largest FD entry on an analytic zero.

    The numeric Jacobian is a Richardson-extrapolated central difference at a step of
    ``1e-6``. The cubic basis reaches ``|logit q|^3`` of about 2,600 at the basis clip, so a
    larger step crosses a clip on some row and a smaller one meets rounding.
    """
    analytic = problem.blocks(theta).jacobian
    numeric = _richardson(problem, theta, 1e-6)
    worst, structural = 0.0, 0.0
    for rows in problem.slices.values():
        for cols in problem.slices.values():
            size = float(np.max(np.abs(numeric[rows, cols])))
            if np.all(analytic[rows, cols] == 0.0):
                structural = max(structural, size)
            elif size > 1e-6:
                gap = float(np.max(np.abs(analytic[rows, cols] - numeric[rows, cols])))
                worst = max(worst, gap / size)
    return worst, structural


@pytest.fixture(scope="module")
def clip_fit() -> Any:
    """A cross-fitted fit on which every clip binds, with ``eps_0 != 0``."""
    return _fit(
        quad.sample(CLIP_LAW, 2000, 7),
        features="arm_interactions_no_intercept",
        g_bounds=(0.3, 0.7),
    )


class TestCW1Jacobian:
    """CW1: the analytic V-fold Jacobian equals a finite difference, every clip binding."""

    def test_every_clip_binds(self, clip_fit: Any) -> None:
        stack = fitted_stack(clip_fit)
        curves = corrected_curves(stack.problem, stack.theta)
        for clip in ("g_bounds", "alpha", "basis"):
            assert min(curves.clipped_share[clip]) > 0.01, clip
        assert min(curves.outer_binding) > 100
        eps = stack.theta[stack.problem.slices["eps[0]"]][0]
        assert abs(eps) > 0.1

    def test_analytic_equals_finite_difference(self, clip_fit: Any) -> None:
        stack = fitted_stack(clip_fit)
        worst, structural = _fd_report(stack.problem, stack.theta)
        assert worst < 1e-5
        assert structural == 0.0

    def test_the_fd_curve_is_the_stacked_curve(self, clip_fit: Any) -> None:
        stack = fitted_stack(clip_fit)
        problem, theta = stack.problem, stack.theta
        rows = problem.rows(theta)
        numeric = -np.linalg.solve(_richardson(problem, theta, 1e-6), rows.T).T
        psi = [problem.slices[f"psi[{j}]"].start for j in range(2)]
        curves = corrected_curves(problem, theta).curves
        assert np.max(np.abs(numeric[:, psi] - curves)) < 1e-4 * np.max(np.abs(curves))

    @pytest.mark.parametrize("mask", ["held", "training"])
    def test_mutation_dropped_fold_mask(
        self, clip_fit: Any, monkeypatch: pytest.MonkeyPatch, mask: str
    ) -> None:
        """A fluctuation block over every row, or an outcome block over every row."""
        original = FoldStackProblem._fold_masses

        def mutated(self: FoldStackProblem, k: int) -> tuple[np.ndarray, np.ndarray]:
            trained, own = original(self, k)
            return (trained, self.mass) if mask == "held" else (self.mass, own)

        monkeypatch.setattr(FoldStackProblem, "_fold_masses", mutated)
        stack = fitted_stack(clip_fit)
        worst, _ = _fd_report(stack.problem, stack.theta)
        assert worst > 1e-2


def _replicated(fit: quad.PopulationFit, shares: np.ndarray) -> tuple[FoldStackProblem, Any]:
    """``V`` copies of the grid as folds, copy ``v`` with mass ``pi_v`` times the grid's."""
    problem, theta = fit.problem, fit.theta
    folds = len(shares)
    m = problem.n
    labels = np.repeat(np.arange(folds), m)
    arms = [
        ArmDesign(
            features=np.tile(arm.features, (folds, 1)),
            indicator=np.tile(arm.indicator, folds),
            basis=arm.basis,
            basis_derivative=arm.basis_derivative,
            bounds=arm.bounds,
        )
        for arm in problem.arms
    ]
    copied = FoldStackProblem(
        np.tile(problem.features, (folds, 1)),
        np.tile(problem.outcome, folds),
        arms,
        problem.alpha,
        folds=labels,
        mass=np.concatenate([share * fit.grid.mass for share in shares]),
    )
    beta = theta[problem.slices["beta"]]
    gammas = [theta[problem.slices[f"gamma[{j}]"]] for j in range(len(problem.arms))]
    eps = [float(theta[problem.slices[f"eps[{j}]"]][0]) for j in range(len(problem.arms))]
    psi = [float(theta[problem.slices[f"psi[{j}]"]][0]) for j in range(len(problem.arms))]
    return copied, copied.pack([beta] * folds, [[g] * folds for g in gammas], eps, psi)


def _split_curve(fit: quad.PopulationFit) -> tuple[np.ndarray, np.ndarray]:
    """The in-sample curve's ``eps``/``psi``-row terms, and the nuisance contribution ``N``."""
    problem, theta = fit.problem, fit.theta
    stack = problem.blocks(theta)
    rows = stack.rows.copy()
    psi = [problem.slices[f"psi[{j}]"].start for j in range(len(problem.arms))]
    full = -np.linalg.solve(stack.jacobian, rows.T).T[:, psi]
    rows[:, problem.slices["beta"]] = 0.0
    for j in range(len(problem.arms)):
        rows[:, problem.slices[f"gamma[{j}]"]] = 0.0
    own = -np.linalg.solve(stack.jacobian, rows.T).T[:, psi]
    return own, full - own


SHARES = np.array([0.30, 0.25, 0.20, 0.15, 0.10])


def _fold_weights(shares: np.ndarray) -> np.ndarray:
    """``m_v = sum_{k != v} pi_k / (1 - pi_k)``, per fold."""
    ratio = shares / (1.0 - shares)
    return ratio.sum() - ratio


@pytest.fixture(scope="module", params=["binary_active", "l3"])
def fold_grid(request: pytest.FixtureRequest) -> quad.PopulationFit:
    if request.param == "binary_active":
        return quad.population_fit(quad.BINARY_ACTIVE, nodes=14)
    return quad.population_fit(
        quad.STEEP, nodes=14, bounds=(0.05, 0.95), features="arm_interactions_no_intercept"
    )


class TestCW2FoldMap:
    """CW2: on a fold-replicated grid each row's curve is its own terms plus ``m_v N``."""

    def test_every_fold_fit_is_the_population_fit(self, fold_grid: quad.PopulationFit) -> None:
        problem, theta = _replicated(fold_grid, SHARES)
        assert max(_block_means(problem, theta).values()) < 1e-12

    def test_the_law_has_a_nuisance_term(self, fold_grid: quad.PopulationFit) -> None:
        _, nuisance = _split_curve(fold_grid)
        assert np.max(np.abs(nuisance)) > 0.05

    def test_rows_follow_the_fold_map(self, fold_grid: quad.PopulationFit) -> None:
        problem, theta = _replicated(fold_grid, SHARES)
        curves = corrected_curves(problem, theta).curves
        own, nuisance = _split_curve(fold_grid)
        weights = _fold_weights(SHARES)
        m = fold_grid.problem.n
        for v, weight in enumerate(weights):
            expected = own + weight * nuisance
            assert np.max(np.abs(curves[v * m : (v + 1) * m] - expected)) < 1e-12

    def test_the_fold_weight_departs_from_one(self) -> None:
        assert _fold_weights(SHARES)[0] == pytest.approx(0.8709, abs=1e-4)

    def test_mutation_averaged_form(self, fold_grid: quad.PopulationFit) -> None:
        """2.9's averaged form, ``m_v = 1``, misses the fold with ``pi = 0.30``."""
        problem, theta = _replicated(fold_grid, SHARES)
        curves = corrected_curves(problem, theta).curves
        own, nuisance = _split_curve(fold_grid)
        m = fold_grid.problem.n
        assert np.max(np.abs(curves[:m] - (own + nuisance))) > 1e-3


class TestCW3InSampleReduction:
    """CW3: one fold whose training and held-out rows are the whole sample is X58a's stack."""

    @pytest.fixture(scope="class")
    def problems(self, in_sample: Any) -> tuple[StackProblem, FoldStackProblem, np.ndarray]:
        stack = fitted_stack(in_sample)
        problem = stack.problem
        assert isinstance(problem, FoldStackProblem) and problem.n_folds == 1
        labelled = FoldStackProblem(
            problem.features,
            problem.outcome,
            problem.arms,
            problem.alpha,
            folds=np.zeros(problem.n, dtype=int),
        )
        return problem, labelled, stack.theta

    def test_the_labelled_fold_is_the_in_sample_stack(
        self, problems: tuple[StackProblem, FoldStackProblem, np.ndarray]
    ) -> None:
        problem, labelled, theta = problems
        np.testing.assert_array_equal(labelled.training, labelled.held)
        reference = corrected_curves(problem, theta).curves
        assert np.max(np.abs(corrected_curves(labelled, theta).curves - reference)) < 1e-12

    def test_the_forward_solve_is_the_dense_solve(
        self, problems: tuple[StackProblem, FoldStackProblem, np.ndarray]
    ) -> None:
        problem, _, theta = problems
        stack = problem.blocks(theta)
        dense = -np.linalg.solve(stack.jacobian, stack.rows.T).T[:, -2:]
        curves = corrected_curves(problem, theta).curves
        assert np.max(np.abs(curves - dense)) < 1e-12 * max(1.0, np.max(np.abs(dense)))

    def test_mutation_training_mask_at_one_fold(
        self, problems: tuple[StackProblem, FoldStackProblem, np.ndarray]
    ) -> None:
        """A ``1 - pi_k`` factor at ``V = 1``: no row trains the fold."""
        problem, _, theta = problems
        labelled = FoldStackProblem(
            problem.features,
            problem.outcome,
            problem.arms,
            problem.alpha,
            folds=np.zeros(problem.n, dtype=int),
        )
        labelled.training = 1.0 - labelled.held
        reference = corrected_curves(problem, theta).curves
        assert np.max(np.abs(corrected_curves(labelled, theta).curves - reference)) > 1e-3


@pytest.fixture(scope="module")
def large_crossfit() -> Any:
    return _fit(laws.BINARY_ACTIVE.sample(200_000, 41)[0])


class TestCW4LargeSample:
    """CW4: at one draw of ``n = 200,000`` the cross-fitted SDs match the quadrature to 3%."""

    def test_cross_fitted_sds_match_the_quadrature(self, large_crossfit: Any) -> None:
        stack = fitted_stack(large_crossfit)
        curves = corrected_curves(stack.problem, stack.theta).curves
        population = quad.population_fit(quad.BINARY_ACTIVE)
        covariance = quad.covariance(population)
        sample = np.cov(curves, rowvar=False)
        for j in (0, 1):
            assert np.sqrt(sample[j, j] / covariance[j, j]) == pytest.approx(1.0, abs=0.03)
        ratio = np.std(curves[:, 1] - curves[:, 0], ddof=1) / quad.ate_sd(population)
        assert ratio == pytest.approx(1.0, abs=0.03)

    def test_mutation_shipped_curve(self, large_crossfit: Any) -> None:
        stack = fitted_stack(large_crossfit)
        assert stack.shipped is not None
        shipped = stack.shipped[:, 1] - stack.shipped[:, 0]
        ratio = np.std(shipped, ddof=1) / quad.ate_sd(quad.population_fit(quad.BINARY_ACTIVE))
        assert abs(ratio - 1.0) > 0.1


class _Entry:
    """A learner diagnostics stand-in with coefficients."""

    def __init__(self, coef: tuple[float, ...]) -> None:
        self.coef = coef


def _entries(stack: Any) -> tuple[dict[int, Any], dict[int, list[Any]]]:
    problem, theta = stack.problem, stack.theta
    outcome = {
        k: _Entry(tuple(theta[problem.slices[problem.beta_name(k)]]))
        for k in range(problem.n_folds)
    }
    mechanism = {
        k: [
            _Entry(tuple(theta[problem.slices[problem.gamma_name(j, k)]]))
            for j in range(len(problem.arms))
        ]
        for k in range(problem.n_folds)
    }
    return outcome, mechanism


class TestCW5FoldRecord:
    """CW5: the fold record is keyed by fold, and a missing fold raises."""

    def test_the_record_holds_one_entry_per_fold(self, unequal: Any) -> None:
        record = unequal.nuisance.diagnostics["stack"]
        assert isinstance(record, FoldCoefficients)
        assert record.folds == 5
        assert len(record.outcome) == 5
        assert [len(arm) for arm in record.mechanism] == [5, 5]

    def test_an_in_sample_record_holds_one_fold(self, in_sample: Any) -> None:
        assert in_sample.nuisance.diagnostics["stack"].folds == 1

    def test_a_missing_fold_raises(self, unequal: Any) -> None:
        outcome, mechanism = _entries(fitted_stack(unequal))
        outcome[2] = None
        with pytest.raises(ValueError, match="fold 2 of 5"):
            FoldCoefficients.from_entries(outcome, mechanism, folds=5, arms=2)
        outcome, mechanism = _entries(fitted_stack(unequal))
        mechanism[3][1] = None
        with pytest.raises(ValueError, match="fold 3 of 5"):
            FoldCoefficients.from_entries(outcome, mechanism, folds=5, arms=2)

    def test_a_fit_that_drops_a_fold_entry_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        original = ctmle_module.fit_on_rows
        calls: list[int] = []

        def dropping(learner: Any, *args: Any, **kwargs: Any) -> Any:
            model = original(learner, *args, **kwargs)
            if isinstance(model, ParametricOutcome):
                calls.append(1)
                if len(calls) == 3:
                    del model.diagnostics_
            return model

        monkeypatch.setattr(ctmle_module, "fit_on_rows", dropping)
        with pytest.raises(ValueError, match="fold 2 of 5 has no outcome coefficients"):
            _fit(laws.BINARY_ACTIVE.sample(1503, 61)[0])

    def test_mutation_positional_record(self, unequal: Any) -> None:
        """Today's filter: the remaining entries shift, and fold 2's training score moves."""
        stack = fitted_stack(unequal)
        outcome, mechanism = _entries(stack)
        kept = [k for k in range(5) if k != 2]
        shifted = [outcome[k].coef for k in kept] + [outcome[kept[-1]].coef]
        problem = stack.problem
        theta = stack.theta.copy()
        for k, coef in enumerate(shifted):
            theta[problem.slices[problem.beta_name(k)]] = coef
        assert _block_means(problem, theta)[problem.beta_name(2)] > 1e-4
        assert len(mechanism) == 5
