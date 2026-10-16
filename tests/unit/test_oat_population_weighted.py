"""Witnesses of ``ey_obs``, ``par``, ``paf`` and the weighted per-arm OAT stack.

X58 plan, 13.2 (PW1 to PW3) and 13.4 (DW0 to DW4). Each witness runs where the term it
checks is material, and each has a deliberate-mutation control that fails when that term is
wrong. The mutations patch a target, the status hook or the stack in memory
(``monkeypatch``); no source file changes.
"""

from __future__ import annotations

import importlib
import warnings
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly._inference_status import precedent_status
from cleverly.estimators import CTMLE, TMLE
from cleverly.estimators._oat_stack import (
    FoldStackProblem,
    StackProblem,
    corrected_curves,
    fitted_stack,
)
from cleverly.estimators.ctmle import per_arm_parametric_support
from cleverly.inference.influence import counterfactual_means
from cleverly.learners import ParametricMechanism, ParametricOutcome
from cleverly.targets import TargetContext
from tests.studies import oat_multi_arm_quadrature as mq
from tests.studies import oat_per_arm_laws as laws

pytestmark = pytest.mark.xdist_group("oat_population_weighted")

tmle_module = importlib.import_module("cleverly.estimators.tmle")

FLOOR = 1e-9
"""The absolute floor on a solved estimating equation's sample mean (C-solve at tol 1e-10)."""

COVARIATES = ["W1", "W2", "W3"]
POPULATION = ("ey0", "ey1", "ate", "ey_obs", "par", "paf")


def _fit(frame: pd.DataFrame, *, estimands: tuple[str, ...] = POPULATION, **roles: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            CTMLE(
                strategy="oat",
                outcome_learner=ParametricOutcome(),
                treatment_learner=ParametricMechanism(),
                cross_fit=False,
                simultaneous=False,
                estimands=estimands,
                g_bounds=laws.G_BOUNDS,
                max_iter=100,
                tol=1e-10,
                random_state=0,
            )
            .fit(frame, outcome="Y", treatment="A", covariates=COVARIATES, **roles)
            .single()
        )


def _stacked(result: Any) -> np.ndarray:
    """``(n, 2)``: the reported stacked arm curves, ``w_i (-A_w^{-1} U_i)``, built afresh."""
    stack = fitted_stack(result)
    curves = corrected_curves(stack.problem, stack.theta).curves
    return np.asarray(curves * stack.scale * np.asarray(result.data.weights)[:, None])


def _block_means(problem: FoldStackProblem, theta: np.ndarray) -> dict[str, float]:
    means = problem.mass @ problem.rows(theta)
    return {name: float(np.max(np.abs(means[part]))) for name, part in problem.slices.items()}


# --------------------------------------------------------------------------- PW1


@pytest.fixture(scope="module")
def binary() -> Any:
    return _fit(laws.BINARY_ACTIVE.sample(1500, 91)[0])


def _stale_par(ctx: TargetContext) -> list[Any]:
    """``par`` from the reference arm mean of before the stack correction (a seam mutation)."""
    observed = ctx.observed_mean
    stale = counterfactual_means(ctx.scaled, ctx.targeted, ctx.submodel, ctx.weights, ctx.observed)
    reference = stale[ctx.reference]
    return [
        ctx.finish(
            ctx.name_for("par", ctx.reference),
            observed.psi - reference.psi,
            observed.influence_curve - reference.influence_curve,
            "difference",
            reads=(ctx.reference,),
        )
    ]


class TestPW1ExactCurves:
    """PW1: ``ey_obs`` is ``w (Y - P_n^w Y)``, and ``par`` subtracts the stacked reference curve."""

    def test_ey_obs_is_the_empirical_curve(self, binary: Any) -> None:
        y = np.asarray(binary.data.outcome, dtype=float)
        np.testing.assert_allclose(binary["ey_obs"].influence_curve, y - y.mean(), atol=1e-15)

    def test_par_subtracts_the_stacked_reference(self, binary: Any) -> None:
        expected = binary["ey_obs"].influence_curve - _stacked(binary)[:, 0]
        np.testing.assert_allclose(binary["par"].influence_curve, expected, atol=1e-12)

    def test_the_covariance_is_of_the_stacked_rows(self, binary: Any) -> None:
        rows = np.column_stack([binary["ey_obs"].influence_curve, _stacked(binary)[:, 0]])
        expected = np.cov(rows, rowvar=False) / binary.n
        np.testing.assert_allclose(binary.covariance(["ey_obs", "ey0"]), expected, rtol=1e-10)

    def test_mutation_par_from_the_shipped_reference(self, binary: Any) -> None:
        stack = fitted_stack(binary)
        assert stack.shipped is not None
        shipped = binary["ey_obs"].influence_curve - stack.shipped[:, 0]
        assert np.max(np.abs(binary["par"].influence_curve - shipped)) > 1e-3

    def test_mutation_par_reads_the_uncorrected_mean(self, monkeypatch: pytest.MonkeyPatch) -> None:
        original = tmle_module.targets_for

        def stale(group: Any, requested: Any) -> tuple[Any, ...]:
            return tuple(
                replace(target, build=_stale_par) if target.name == "par" else target
                for target in original(group, requested)
            )

        monkeypatch.setattr(tmle_module, "targets_for", stale)
        mutant = _fit(laws.BINARY_ACTIVE.sample(1500, 91)[0])
        expected = mutant["ey_obs"].influence_curve - _stacked(mutant)[:, 0]
        assert np.max(np.abs(mutant["par"].influence_curve - expected)) > 1e-3


# --------------------------------------------------------------------------- PW2, PW3

EXACT_PAR, EXACT_PAF = 0.6605, 1.3164
"""``sqrt(n)`` SDs of ``par`` and ``paf`` on ``binary_active`` (``par_paf_quadrature.log``)."""


@pytest.fixture(scope="module")
def binary_grid() -> mq.PopulationFit:
    return mq.population_fit(mq.BINARY_ACTIVE, *mq.hermite_nodes(30))


@pytest.fixture(scope="module")
def large_binary() -> Any:
    return _fit(
        laws.BINARY_ACTIVE.sample(200_000, 93)[0], estimands=("ey0", "ey_obs", "par", "paf")
    )


class TestPW2LargeSample:
    """PW2: at ``n = 200,000`` the ``par`` and ``paf`` SDs match the quadrature to 3%."""

    def test_the_quadrature_reproduces_the_plan(self, binary_grid: mq.PopulationFit) -> None:
        named = mq.contrast_curves(binary_grid, mq.arm_curves(binary_grid), 0)
        assert mq.sd(binary_grid, named["par[0]"]) == pytest.approx(EXACT_PAR, abs=1e-4)
        assert mq.sd(binary_grid, named["paf[0]"]) == pytest.approx(EXACT_PAF, abs=1e-4)

    def test_the_reported_sds_match(self, large_binary: Any) -> None:
        root = np.sqrt(large_binary.n)
        assert large_binary["par"].std_error * root / EXACT_PAR == pytest.approx(1.0, abs=0.03)
        assert large_binary["paf"].std_error * root / EXACT_PAF == pytest.approx(1.0, abs=0.03)

    def test_mutation_shipped_curve(self, large_binary: Any) -> None:
        stack = fitted_stack(large_binary)
        assert stack.shipped is not None
        observed = large_binary["ey_obs"].influence_curve
        psi_obs, psi_ref = large_binary["ey_obs"].psi, large_binary["ey0"].psi
        par = observed - stack.shipped[:, 0]
        paf = -stack.shipped[:, 0] / psi_obs + psi_ref * observed / psi_obs**2
        assert abs(np.std(par, ddof=1) / EXACT_PAR - 1.0) > 0.1
        assert abs(np.std(paf, ddof=1) / EXACT_PAF - 1.0) > 0.1


def _paf_difference(binary: Any, step: float = 1e-6) -> np.ndarray:
    """Row by row, a central difference of ``1 - psi_ref / psi_obs`` along each row's curve."""
    psi_ref, psi_obs = binary["ey0"].psi, binary["ey_obs"].psi
    ic_ref = binary["ey0"].influence_curve
    ic_obs = binary["ey_obs"].influence_curve

    def paf(t: float) -> np.ndarray:
        return np.asarray(1.0 - (psi_ref + t * ic_ref) / (psi_obs + t * ic_obs))

    return (paf(step) - paf(-step)) / (2.0 * step)


class TestPW3PafGradient:
    """PW3: the ``paf`` curve is the delta method of ``1 - psi_ref / psi_obs``."""

    def test_the_curve_is_the_derivative(self, binary: Any) -> None:
        np.testing.assert_allclose(
            binary["paf"].influence_curve, _paf_difference(binary), rtol=1e-6, atol=1e-9
        )

    def test_mutation_dropped_observed_term(self, binary: Any) -> None:
        dropped = -binary["ey0"].influence_curve / binary["ey_obs"].psi
        assert np.max(np.abs(dropped - _paf_difference(binary))) > 1e-2


# --------------------------------------------------------------------------- DW0


def _weighted_frame(n: int, seed: int) -> pd.DataFrame:
    return laws.BINARY_WEIGHTED.sample(n, seed)[0]


@pytest.fixture(scope="module")
def weighted() -> Any:
    return _fit(_weighted_frame(3000, 95), weights="wt")


class TestDW0WeightedSystem:
    """DW0: the weighted fit solves ``P_n[w U(theta)] = 0`` block by block."""

    def test_every_block_is_solved(self, weighted: Any) -> None:
        stack = fitted_stack(weighted)
        weights = np.asarray(weighted.data.weights)
        assert np.ptp(weights) > 0.5
        np.testing.assert_allclose(stack.problem.mass, weights / weights.sum(), rtol=1e-14)
        means = _block_means(stack.problem, stack.theta)
        assert max(means.values()) < FLOOR, means
        assert stack.targeted_mismatch < 1e-12

    def test_mutation_unweighted_psi(self, weighted: Any) -> None:
        stack = fitted_stack(weighted)
        problem, theta = stack.problem, stack.theta.copy()
        targeted = problem.targeted(theta)
        for j in range(2):
            theta[problem.slices[f"psi[{j}]"]] = np.mean(targeted[:, j])
        assert _block_means(problem, theta)["psi[0]"] > 1e-4


# --------------------------------------------------------------------------- DW1


def _integer_weight(frame: pd.DataFrame) -> np.ndarray:
    return 1.0 + (frame["W1"].to_numpy() > 0.0)


@pytest.fixture(scope="module")
def duplication() -> tuple[Any, Any, int]:
    frame = laws.BINARY_ACTIVE.sample(1500, 97)[0]
    frame = frame.assign(w=_integer_weight(frame))
    doubled = pd.concat([frame, frame[frame["w"] == 2.0]], ignore_index=True)
    return _fit(frame, weights="w"), _fit(doubled), len(frame)


def _rows_of(problem: FoldStackProblem, theta: np.ndarray, jacobian: np.ndarray) -> np.ndarray:
    rows = problem.blocks(theta, jacobian=False).rows
    return np.asarray(np.linalg.solve(jacobian, -rows.T).T)


class TestDW1Duplication:
    """DW1: integer weights equal the unweighted fit on the duplicated rows, exactly."""

    def test_theta_jacobian_and_rows(self, duplication: tuple[Any, Any, int]) -> None:
        weighted_fit, doubled_fit, n = duplication
        weighted_stack, doubled_stack = fitted_stack(weighted_fit), fitted_stack(doubled_fit)
        np.testing.assert_allclose(weighted_stack.theta, doubled_stack.theta, rtol=1e-10)
        a_w = weighted_stack.problem.blocks(weighted_stack.theta).jacobian
        a_d = doubled_stack.problem.blocks(doubled_stack.theta).jacobian
        np.testing.assert_allclose(a_w, a_d, rtol=1e-10, atol=1e-12)
        u_w = _rows_of(weighted_stack.problem, weighted_stack.theta, a_w)
        u_d = _rows_of(doubled_stack.problem, doubled_stack.theta, a_d)
        np.testing.assert_allclose(u_w, u_d[:n], rtol=1e-10, atol=1e-12)

    def test_the_reported_standard_errors(self, duplication: tuple[Any, Any, int]) -> None:
        weighted_fit, doubled_fit, n = duplication
        for name in ("ey0", "ey1", "ate"):
            assert weighted_fit[name].psi == pytest.approx(doubled_fit[name].psi, rel=1e-10)

    def test_mutation_unweighted_mass_in_the_jacobian(
        self, duplication: tuple[Any, Any, int]
    ) -> None:
        weighted_fit, doubled_fit, n = duplication
        weighted_stack, doubled_stack = fitted_stack(weighted_fit), fitted_stack(doubled_fit)
        problem = weighted_stack.problem
        unweighted = StackProblem(problem.features, problem.outcome, problem.arms, problem.alpha)
        a_mutant = unweighted.blocks(weighted_stack.theta).jacobian
        u_mutant = _rows_of(problem, weighted_stack.theta, a_mutant)
        a_d = doubled_stack.problem.blocks(doubled_stack.theta).jacobian
        u_d = _rows_of(doubled_stack.problem, doubled_stack.theta, a_d)
        assert np.max(np.abs(u_mutant - u_d[:n])) > 1e-3


# --------------------------------------------------------------------------- DW2


class _UnnormalizedStack(StackProblem):
    """A stack whose measure keeps the raw masses, with ``A_{psi, psi} = -sum(mass)``.

    The control of DW2. The plan's control keeps ``A_{psi, psi} = -1`` beside the raw masses,
    and that one cannot fail: every other block scales by the total mass ``S``, so the forward
    substitution scales each earlier block's rows by ``1 / S`` and the ``psi`` rows not at all.
    The defect that moves the curve is a measure that is not normalized in every block.
    """

    def __init__(self, problem: FoldStackProblem, mass: np.ndarray) -> None:
        super().__init__(problem.features, problem.outcome, problem.arms, problem.alpha)
        self.mass = np.asarray(mass, dtype=float)

    def blocks(self, theta: np.ndarray, *, jacobian: bool = True) -> Any:
        stack = super().blocks(theta, jacobian=jacobian)
        if jacobian:
            for j in range(len(self.arms)):
                psi = self.slices[f"psi[{j}]"]
                stack.jacobian[psi, psi] = -float(np.sum(self.mass))
        return stack


class TestDW2ScaleInvariance:
    """DW2: raw masses ``w`` and ``7 w`` give the same curves; the data stores one weight."""

    def test_the_stack_is_invariant(self, weighted: Any) -> None:
        stack = fitted_stack(weighted)
        problem, theta = stack.problem, stack.theta
        raw = np.asarray(weighted.data.weights)
        curves = {
            scale: corrected_curves(
                StackProblem(
                    problem.features, problem.outcome, problem.arms, problem.alpha, scale * raw
                ),
                theta,
            ).curves
            for scale in (1.0, 7.0)
        }
        np.testing.assert_allclose(curves[1.0], curves[7.0], rtol=1e-12, atol=1e-14)
        se = {scale: np.std(raw[:, None] * c, axis=0, ddof=1) for scale, c in curves.items()}
        np.testing.assert_allclose(se[1.0], se[7.0], rtol=1e-12)

    def test_the_fit_stores_the_same_weights(self) -> None:
        frame = _weighted_frame(1500, 99)
        frame = frame.assign(wt7=7.0 * frame["wt"])
        one, seven = _fit(frame, weights="wt"), _fit(frame, weights="wt7")
        np.testing.assert_allclose(one.data.weights, seven.data.weights, rtol=1e-14)
        assert one["ate"].std_error == pytest.approx(seven["ate"].std_error, rel=1e-12)

    def test_mutation_unnormalized_mass(self, weighted: Any) -> None:
        stack = fitted_stack(weighted)
        raw = np.asarray(weighted.data.weights)
        one = corrected_curves(_UnnormalizedStack(stack.problem, raw), stack.theta).curves
        seven = corrected_curves(_UnnormalizedStack(stack.problem, 7.0 * raw), stack.theta).curves
        assert np.max(np.abs(one - seven)) > 1e-3


# --------------------------------------------------------------------------- DW3

EXACT_WEIGHTED = {"ey0": 0.9500, "ey1": 0.8134, "ate": 1.2598}
"""``sqrt(n)`` SDs on ``binary_active_weighted`` (``three_arm_weighted_quadrature.log``)."""


@pytest.fixture(scope="module")
def weighted_grid() -> mq.PopulationFit:
    return mq.population_fit(mq.BINARY_ACTIVE_WEIGHTED, *mq.half_line_nodes(60, 20))


@pytest.fixture(scope="module")
def large_weighted() -> Any:
    return _fit(_weighted_frame(200_000, 101), estimands=("ey0", "ey1", "ate"), weights="wt")


def _exact_weighted(grid: mq.PopulationFit) -> dict[str, float]:
    named = mq.contrast_curves(grid, mq.arm_curves(grid), 0)
    return {
        "ey0": mq.sd(grid, named["ey[0]"]),
        "ey1": mq.sd(grid, named["ey[1]"]),
        "ate": mq.sd(grid, named["ate[1 vs 0]"]),
    }


class TestDW3LargeSample:
    """DW3: at ``n = 200,000`` the weighted SDs match the weighted quadrature to 3%."""

    def test_the_quadrature_reproduces_the_plan(self, weighted_grid: mq.PopulationFit) -> None:
        exact = _exact_weighted(weighted_grid)
        for name, value in EXACT_WEIGHTED.items():
            assert exact[name] == pytest.approx(value, abs=1e-4), name

    def test_the_reported_sds_match(self, large_weighted: Any) -> None:
        root = np.sqrt(large_weighted.n)
        for name, value in EXACT_WEIGHTED.items():
            ratio = large_weighted[name].std_error * root / value
            assert ratio == pytest.approx(1.0, abs=0.03), name

    def test_mutation_shipped_curve(self, large_weighted: Any) -> None:
        stack = fitted_stack(large_weighted)
        assert stack.shipped is not None
        weights = np.asarray(large_weighted.data.weights)
        shipped = np.std(weights * stack.shipped[:, 0], ddof=1)
        assert abs(shipped / EXACT_WEIGHTED["ey0"] - 1.0) > 0.1

    def test_mutation_curve_without_the_weight(self, large_weighted: Any) -> None:
        stack = fitted_stack(large_weighted)
        curves = corrected_curves(stack.problem, stack.theta).curves
        assert abs(np.std(curves[:, 0], ddof=1) / EXACT_WEIGHTED["ey0"] - 1.0) > 0.1


# --------------------------------------------------------------------------- DW4


def _fixed_status(self: CTMLE, data: Any) -> str:
    """The status hook without the estimated-weight branch (the mutation of DW4)."""
    if per_arm_parametric_support(self, data).admitted:
        return precedent_status(["influence_curve", TMLE._inference_status(self, data)])
    return "generated_design_plugin"


@pytest.fixture(scope="module")
def estimated_pair() -> tuple[Any, Any]:
    frame = _weighted_frame(1500, 103)
    return (
        _fit(frame, weights="wt"),
        _fit(frame, weights="wt", weights_estimated=True),
    )


class TestDW4EstimatedWeights:
    """DW4: estimated weights keep the fixed-weight curve under ``estimated_weight_plugin``."""

    def test_the_curves_are_the_fixed_weight_curves(self, estimated_pair: tuple[Any, Any]) -> None:
        fixed, estimated = estimated_pair
        for name in POPULATION:
            np.testing.assert_allclose(
                estimated[name].influence_curve, fixed[name].influence_curve, rtol=1e-12, atol=0
            )
            assert estimated[name].std_error == pytest.approx(fixed[name].std_error, rel=1e-12)

    def test_the_status_and_the_label(self, estimated_pair: tuple[Any, Any]) -> None:
        fixed, estimated = estimated_pair
        assert fixed.inference_status == "influence_curve"
        assert estimated.inference_status == "estimated_weight_plugin"
        for estimate in estimated.estimates.values():
            assert estimate.inference == "estimated_weight_plugin"
            assert estimate.supplies_inference
            assert estimate.support.label == "experimental"
            assert "X50" in estimate.support.evidence

    def test_constant_estimated_weights_keep_the_interval(self) -> None:
        frame = _weighted_frame(1500, 103).assign(one=2.0)
        result = _fit(frame, weights="one", weights_estimated=True)
        assert result.inference_status == "influence_curve"

    def test_mutation_no_status_branch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(CTMLE, "_inference_status", _fixed_status)
        result = _fit(_weighted_frame(1500, 103), weights="wt", weights_estimated=True)
        assert result.inference_status != "estimated_weight_plugin"

    def test_mutation_a_branch_that_drops_the_weight(self, estimated_pair: tuple[Any, Any]) -> None:
        fixed, estimated = estimated_pair
        weights = np.asarray(estimated.data.weights)
        dropped = estimated["ey0"].influence_curve / weights
        assert np.max(np.abs(dropped - fixed["ey0"].influence_curve)) > 1e-3


# --------------------------------------------------------------------------- the weighted FD


class TestTheWeightedJacobian:
    """The weighted Jacobian equals a finite difference of ``P_n^w U`` on a weighted grid."""

    def test_analytic_equals_finite_difference(self) -> None:
        from tests.unit.test_oat_multi_arm_stack import fd_error

        grid = mq.population_fit(mq.BINARY_ACTIVE_WEIGHTED, *mq.half_line_nodes(30, 12))
        assert np.ptp(grid.grid.weight) > 0.5
        assert fd_error(grid.problem, grid.theta) < 1e-6
