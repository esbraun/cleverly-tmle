"""Witnesses of the per-arm OAT stacked curve (X58 plan, 4.2).

Each witness runs on a law where the term it checks is material, and each has a
deliberate-mutation control that fails when that term is wrong. The mutations patch the
stack in memory (``monkeypatch``); no source file changes.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from scipy.special import expit, logit

from cleverly.estimators import CTMLE, _oat_stack
from cleverly.estimators._oat_stack import StackProblem, corrected_curves, fitted_stack
from cleverly.learners import ParametricMechanism, ParametricOutcome
from tests.studies import oat_parametric_quadrature as quad
from tests.studies import oat_per_arm_laws as laws

pytestmark = pytest.mark.xdist_group("oat_stack")

FLOOR = 1e-9
"""The absolute floor on a solved estimating equation's sample mean (C-solve at tol 1e-10)."""

CLIP_LAW = quad.QuadratureLaw(
    "clip_witness",
    lambda w: np.asarray(expit(0.5 * w[:, 0] + 0.8 * w[:, 2])),
    lambda a, w: np.asarray(expit(-1.0 + (6.0 + 2.0 * a) * w[:, 0] + 0.5 * w[:, 1])),
)
"""An outcome so steep in ``W1`` that the ``alpha`` clip, the outer re-shrink and the basis
clip all bind on material mass, with ``eps* != 0`` under the no-intercept design."""


def _fit(
    frame: Any,
    *,
    features: str = "arm_interactions",
    covariates: tuple[str, ...] = ("W1", "W2", "W3"),
    g_bounds: tuple[float, float] = laws.G_BOUNDS,
    q_bounds: tuple[float, float] | None = None,
) -> Any:
    options: dict[str, Any] = {} if q_bounds is None else {"q_bounds": q_bounds}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            CTMLE(
                strategy="oat",
                outcome_learner=ParametricOutcome(features),  # type: ignore[arg-type]
                treatment_learner=ParametricMechanism(),
                cross_fit=False,
                simultaneous=False,
                estimands=("ey0", "ey1", "ate"),
                g_bounds=g_bounds,
                max_iter=100,
                tol=1e-10,
                random_state=0,
                **options,
            )
            .fit(frame, outcome="Y", treatment="A", covariates=list(covariates))
            .single()
        )


def _block_means(problem: StackProblem, theta: np.ndarray) -> dict[str, float]:
    means = problem.mass @ problem.rows(theta)
    return {name: float(np.max(np.abs(means[part]))) for name, part in problem.slices.items()}


ADMITTED = {
    "binary": lambda: _fit(laws.BINARY_ACTIVE.sample(2000, 31)[0]),
    "binding_bound": lambda: _fit(quad.sample(quad.STEEP, 2000, 32), g_bounds=(0.05, 0.95)),
    "bounded_outcome": lambda: _fit(
        quad.sample(quad.BINARY_ACTIVE, 2000, 33, bounded=(2.0, 10.0)), q_bounds=(2.0, 10.0)
    ),
    "wrong_outcome": lambda: _fit(laws.BINARY_ACTIVE.sample(2000, 34)[0], covariates=("W2", "W3")),
}


@pytest.fixture(scope="module")
def admitted() -> dict[str, Any]:
    return {name: build() for name, build in ADMITTED.items()}


class TestW0SolvedSystem:
    """W0: the stack's ``U`` is the system the fit solved, block by block."""

    @pytest.mark.parametrize("name", sorted(ADMITTED))
    def test_every_block_is_solved(self, admitted: dict[str, Any], name: str) -> None:
        stack = fitted_stack(admitted[name])
        means = _block_means(stack.problem, stack.theta)
        assert max(means.values()) < FLOOR, means
        assert stack.targeted_mismatch < 1e-12

    def test_binding_bound_law_clips(self, admitted: dict[str, Any]) -> None:
        stack = fitted_stack(admitted["binding_bound"])
        shares = corrected_curves(stack.problem, stack.theta).clipped_share["g_bounds"]
        assert min(shares) > 0.05

    def test_bounded_outcome_scale(self, admitted: dict[str, Any]) -> None:
        result = admitted["bounded_outcome"]
        stack = fitted_stack(result)
        assert stack.scale == pytest.approx(8.0)
        assert result["ey1"].psi == pytest.approx(2.0 + 8.0 * stack.theta[-1], abs=1e-12)

    def test_mutation_swapped_arm_indicator(self, admitted: dict[str, Any]) -> None:
        stack = fitted_stack(admitted["binary"])
        arms = stack.problem.arms
        swapped = [
            replace(arm, indicator=other.indicator)
            for arm, other in zip(arms, arms[::-1], strict=True)
        ]
        mutant = StackProblem(stack.problem.features, stack.problem.outcome, swapped, 0.9995)
        assert max(_block_means(mutant, stack.theta).values()) > 1e-3

    def test_mutation_basis_without_intercept(self, admitted: dict[str, Any]) -> None:
        stack = fitted_stack(admitted["binary"])

        def no_intercept(q: np.ndarray) -> np.ndarray:
            phi = ParametricMechanism().design_matrix(q)
            phi[:, 0] = 0.0
            return phi

        arms = [replace(arm, basis=no_intercept) for arm in stack.problem.arms]
        mutant = StackProblem(stack.problem.features, stack.problem.outcome, arms, 0.9995)
        assert _block_means(mutant, stack.theta)["gamma[0]"] > 1e-3

    def test_mutation_dropped_scaler_range(self, admitted: dict[str, Any]) -> None:
        result = admitted["bounded_outcome"]
        stack = fitted_stack(result)
        raw = np.asarray(result.data.outcome, dtype=float)
        mutant = StackProblem(stack.problem.features, raw, stack.problem.arms, 0.9995)
        assert _block_means(mutant, stack.theta)["beta"] > 1e-3


def _central(problem: StackProblem, theta: np.ndarray, step: float) -> np.ndarray:
    numeric = np.zeros((problem.dimension, problem.dimension))
    for k in range(theta.size):
        h = step * max(1.0, abs(theta[k]))
        up, down = theta.copy(), theta.copy()
        up[k] += h
        down[k] -= h
        numeric[:, k] = problem.mass @ (problem.rows(up) - problem.rows(down)) / (2.0 * h)
    return numeric


def _fd_error(problem: StackProblem, theta: np.ndarray, step: float = 1e-5) -> float:
    """The largest block-relative gap between the analytic and the numeric Jacobian.

    The numeric Jacobian is a Richardson-extrapolated central difference, accurate to
    ``O(step^4)``: the cubic basis in ``logit q`` has a large third derivative where
    ``|logit q|`` is near the basis clip, and a plain central difference leaves an
    ``O(step^2)`` truncation error of about ``1e-6`` there.
    """
    analytic = problem.blocks(theta).jacobian
    numeric = (4.0 * _central(problem, theta, step / 2.0) - _central(problem, theta, step)) / 3.0
    worst = 0.0
    for rows in problem.slices.values():
        for cols in problem.slices.values():
            size = float(np.max(np.abs(numeric[rows, cols])))
            if size > 1e-6:
                gap = float(np.max(np.abs(analytic[rows, cols] - numeric[rows, cols])))
                worst = max(worst, gap / size)
    return worst


@pytest.fixture(scope="module")
def l3() -> quad.PopulationFit:
    return quad.population_fit(
        quad.STEEP, nodes=16, bounds=(0.05, 0.95), features="arm_interactions_no_intercept"
    )


@pytest.fixture(scope="module")
def clip_grid() -> quad.PopulationFit:
    return quad.population_fit(CLIP_LAW, nodes=16, features="arm_interactions_no_intercept")


def _drop_mask(monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    original = StackProblem._arm_state

    def mutated(self: StackProblem, theta: np.ndarray, j: int) -> Any:
        state = original(self, theta, j)
        return replace(state, **{field: np.ones_like(getattr(state, field))})

    monkeypatch.setattr(StackProblem, "_arm_state", mutated)


def _unmasked_basis(fit: quad.PopulationFit) -> quad.PopulationFit:
    arms = [replace(arm, basis_derivative=_raw_derivative) for arm in fit.problem.arms]
    problem = StackProblem(
        fit.problem.features, fit.problem.outcome, arms, fit.problem.alpha, mass=fit.grid.mass
    )
    return replace(fit, problem=problem)


def _raw_derivative(q: np.ndarray) -> np.ndarray:
    """``d phi / d logit(q)`` of the clipped basis, without the clip's indicator."""
    template = ParametricMechanism()
    level = logit(np.clip(q, template.clip, 1.0 - template.clip))
    return np.column_stack([np.zeros_like(level), np.ones_like(level), 2 * level, 3 * level**2])


class TestW1Jacobian:
    """W1: the analytic Jacobian equals a central finite difference, with every clip mask."""

    def test_l3_has_eps_and_a_binding_bound(self, l3: quad.PopulationFit) -> None:
        assert abs(l3.epsilons[0]) > 0.01
        assert min(l3.clipped) > 0.05

    def test_clip_grid_binds_every_other_clip(self, clip_grid: quad.PopulationFit) -> None:
        _, shares = clip_grid.problem.clip_summary(clip_grid.theta)
        assert min(shares["alpha"]) > 0.01
        assert min(shares["basis"]) > 0.01
        assert min(clip_grid.outer_binding) > 0.01
        assert np.max(np.abs(clip_grid.epsilons)) > 0.01

    @pytest.mark.parametrize("fixture", ["l3", "clip_grid"])
    def test_analytic_equals_finite_difference(
        self, request: pytest.FixtureRequest, fixture: str
    ) -> None:
        fit = request.getfixturevalue(fixture)
        assert _fd_error(fit.problem, fit.theta) < 1e-6

    @pytest.mark.parametrize(
        ("fixture", "field"),
        [
            ("l3", "inside_bounds"),
            ("clip_grid", "inside_alpha"),
            ("clip_grid", "inside_outer"),
        ],
    )
    def test_mutation_dropped_mask(
        self,
        request: pytest.FixtureRequest,
        monkeypatch: pytest.MonkeyPatch,
        fixture: str,
        field: str,
    ) -> None:
        fit = request.getfixturevalue(fixture)
        _drop_mask(monkeypatch, field)
        assert _fd_error(fit.problem, fit.theta) > 1e-3

    def test_mutation_dropped_basis_mask(self, clip_grid: quad.PopulationFit) -> None:
        mutant = _unmasked_basis(clip_grid)
        assert _fd_error(mutant.problem, mutant.theta) > 1e-3


@pytest.fixture(scope="module")
def large_binary() -> Any:
    return _fit(laws.BINARY_ACTIVE.sample(200_000, 41)[0])


class TestW2LargeSample:
    """W2: at one draw of ``n = 200,000`` the corrected SDs match the quadrature to 3%."""

    def test_corrected_sds_match_the_quadrature(self, large_binary: Any) -> None:
        stack = fitted_stack(large_binary)
        curves = corrected_curves(stack.problem, stack.theta).curves
        population = quad.covariance(quad.population_fit(quad.BINARY_ACTIVE))
        sample = np.cov(curves, rowvar=False)
        for j in (0, 1):
            assert np.sqrt(sample[j, j] / population[j, j]) == pytest.approx(1.0, abs=0.03)
        ate = curves[:, 1] - curves[:, 0]
        assert np.std(ate, ddof=1) / quad.ate_sd(
            quad.population_fit(quad.BINARY_ACTIVE)
        ) == pytest.approx(1.0, abs=0.03)

    def test_mutation_shipped_curve(self, large_binary: Any) -> None:
        n = large_binary.n
        shipped = large_binary["ate"].plugin_std_error * np.sqrt(n)
        ratio = shipped / quad.ate_sd(quad.population_fit(quad.BINARY_ACTIVE))
        assert abs(ratio - 1.0) > 0.1


@pytest.fixture(scope="module")
def medium_binary() -> tuple[Any, Any]:
    frame, _ = laws.BINARY_ACTIVE.sample(50_000, 42)
    return frame, _fit(frame)


def _oracle_ate_rows(result: Any, frame: Any, sign: float = 1.0) -> np.ndarray:
    """The x17 oracle: the shipped curve plus ``c' IF_beta`` with the true ``g_0``."""
    stack = fitted_stack(result)
    problem, theta = stack.problem, stack.theta
    beta = theta[problem.slices["beta"]]
    x, y = problem.features, problem.outcome
    fitted = expit(x @ beta)
    information = (x * (fitted * (1.0 - fitted))[:, None]).T @ x / y.size
    if_beta = np.linalg.solve(information, (x * (y - fitted)[:, None]).T).T
    g0 = laws.BINARY_ACTIVE.mechanism(frame["W1"].to_numpy(), frame["W3"].to_numpy())
    targeted = problem.targeted(theta)
    rows = []
    for j, arm in enumerate(problem.arms):
        gamma = theta[problem.slices[f"gamma[{j}]"]]
        q = expit(arm.features @ beta)
        clipped = np.clip(expit(arm.basis(q) @ gamma), *arm.bounds)
        plug = (
            arm.indicator / clipped * (y - targeted[:, j]) + targeted[:, j] - targeted[:, j].mean()
        )
        c = np.mean(arm.features * (q * (1.0 - q) * (1.0 - g0[:, j] / clipped))[:, None], axis=0)
        rows.append(plug + sign * (if_beta @ c))
    return rows[1] - rows[0]


class TestW3Oracle:
    """W3: with ``kappa = 1`` the corrected curve is the x17 oracle row by row."""

    def test_corrected_curve_tracks_the_oracle(self, medium_binary: tuple[Any, Any]) -> None:
        frame, result = medium_binary
        stack = fitted_stack(result)
        curves = corrected_curves(stack.problem, stack.theta).curves
        corrected = curves[:, 1] - curves[:, 0]
        assert np.corrcoef(corrected, _oracle_ate_rows(result, frame))[0, 1] > 0.999

    def test_mutation_reversed_generated_design_term(self, medium_binary: tuple[Any, Any]) -> None:
        frame, result = medium_binary
        stack = fitted_stack(result)
        curves = corrected_curves(stack.problem, stack.theta).curves
        corrected = curves[:, 1] - curves[:, 0]
        assert np.corrcoef(corrected, _oracle_ate_rows(result, frame, sign=-1.0))[0, 1] < 0.9


NO_SHRINK = 1.0 - 1e-14
"""An ``alpha`` that never binds on the grids, so ``eps* = 0`` holds to rounding."""


@pytest.fixture(scope="module", params=["u_shaped", "steep"])
def kappa_grid(request: pytest.FixtureRequest) -> quad.PopulationFit:
    if request.param == "u_shaped":
        return quad.population_fit(quad.U_SHAPED, nodes=20, alpha=NO_SHRINK)
    return quad.population_fit(quad.STEEP, nodes=20, bounds=(0.05, 0.95), alpha=NO_SHRINK)


class TestW4ClosedForm:
    """W4: under a correct outcome model each curve row equals the closed form of 2.3."""

    def test_kappa_departs_from_one(self, kappa_grid: quad.PopulationFit) -> None:
        assert np.max(np.abs(quad.kappa(kappa_grid) - 1.0)) > 0.05

    def test_rows_equal_the_closed_form(self, kappa_grid: quad.PopulationFit) -> None:
        gap = np.max(np.abs(quad.arm_curves(kappa_grid) - quad.closed_form_curves(kappa_grid)))
        assert gap < 1e-10

    def test_mutation_kappa_forced_to_one(
        self, kappa_grid: quad.PopulationFit, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(quad, "kappa", lambda fit: np.ones(2))
        gap = np.max(np.abs(quad.arm_curves(kappa_grid) - quad.closed_form_curves(kappa_grid)))
        assert gap > 0.01


@pytest.fixture(scope="module", params=["binary_active", "u_shaped"])
def pinned_grid(request: pytest.FixtureRequest) -> quad.PopulationFit:
    law = quad.BINARY_ACTIVE if request.param == "binary_active" else quad.U_SHAPED
    return quad.population_fit(law, nodes=20, alpha=NO_SHRINK)


def _pinned_closed_form(fit: quad.PopulationFit) -> np.ndarray:
    problem, theta = fit.problem, fit.theta
    beta = theta[problem.slices["beta"]]
    kappas = quad.kappa(fit)
    out = np.empty((problem.n, len(problem.arms)))
    for j, arm in enumerate(problem.arms):
        gamma = theta[problem.slices[f"gamma[{j}]"]]
        q = expit(arm.features @ beta)
        clever = 1.0 / np.clip(expit(arm.basis(q) @ gamma), *arm.bounds)
        out[:, j] = (q - fit.psi_star[j]) + kappas[j] * arm.indicator * clever * (
            problem.outcome - q
        )
    return out


class TestW5FixedDesign:
    """W5: with the outcome pinned, each row is ``(q - psi) + kappa R H (Y - q)``."""

    def test_pinned_rows(self, pinned_grid: quad.PopulationFit) -> None:
        gap = np.max(
            np.abs(quad.fixed_outcome_curves(pinned_grid) - _pinned_closed_form(pinned_grid))
        )
        assert gap < 1e-10

    def test_mutation_beta_block_kept(self, pinned_grid: quad.PopulationFit) -> None:
        gap = np.max(np.abs(quad.arm_curves(pinned_grid) - _pinned_closed_form(pinned_grid)))
        assert gap > 0.01


def test_stack_refuses_other_learners() -> None:
    frame, _ = laws.BINARY_ACTIVE.sample(500, 51)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = (
            CTMLE(
                strategy="oat",
                outcome_learner=laws.interaction_outcome(2),
                treatment_learner=laws.cubic_logit_learner(),
                cross_fit=False,
                simultaneous=False,
                estimands=("ey0", "ey1", "ate"),
                random_state=0,
            )
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
    with pytest.raises(ValueError, match="ParametricOutcome"):
        fitted_stack(result)


def test_module_surface() -> None:
    assert set(_oat_stack.__all__) >= {"StackProblem", "corrected_curves", "fitted_stack"}
