"""Witnesses of the ``K``-arm per-arm OAT stack and its band (X58 plan, 13.2: KW and BW).

Each witness runs where the term it checks is material, and each has a deliberate-mutation
control that fails when that term is wrong. The mutations patch the stack, a target or the
band in memory (``monkeypatch``); no source file changes.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from cleverly.estimators import CTMLE, _oat_stack
from cleverly.estimators._oat_stack import (
    FoldStackProblem,
    StackProblem,
    corrected_curves,
    fitted_stack,
)
from cleverly.exceptions import CapabilityError, DataWarning
from cleverly.inference.multiplier import multiplier_critical_value, simultaneous_bands
from cleverly.learners import ParametricMechanism, ParametricOutcome
from cleverly.targets import TargetContext
from tests.studies import oat_multi_arm_quadrature as mq
from tests.studies import oat_per_arm_laws as laws

pytestmark = pytest.mark.xdist_group("oat_multi_arm_stack")

FLOOR = 1e-9
"""The absolute floor on a solved estimating equation's sample mean (C-solve at tol 1e-10)."""

NO_SHRINK = 1.0 - 1e-14
"""An ``alpha`` that never binds on the grids, so ``eps* = 0`` holds to rounding."""

COVARIATES = ["W1", "W2", "W3"]


def _fit(
    frame: Any,
    *,
    reference: str = "0",
    estimands: tuple[str, ...] = ("ey", "ate"),
    simultaneous: bool = False,
    **options: Any,
) -> Any:
    settings: dict[str, Any] = {
        "strategy": "oat",
        "outcome_learner": ParametricOutcome(arm_columns=2),
        "treatment_learner": ParametricMechanism(),
        "cross_fit": False,
        "simultaneous": simultaneous,
        "estimands": estimands,
        "reference": reference,
        "g_bounds": laws.G_BOUNDS,
        "max_iter": 100,
        "tol": 1e-10,
        "random_state": 0,
    }
    settings.update(options)
    weights = settings.pop("weights", None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            CTMLE(**settings)
            .fit(frame, outcome="Y", treatment="A", covariates=COVARIATES, weights=weights)
            .single()
        )


def _block_means(problem: FoldStackProblem, theta: np.ndarray) -> dict[str, float]:
    means = problem.mass @ problem.rows(theta)
    return {name: float(np.max(np.abs(means[part]))) for name, part in problem.slices.items()}


def _central(problem: Any, theta: np.ndarray, step: float) -> np.ndarray:
    numeric = np.zeros((theta.size, theta.size))
    for k in range(theta.size):
        h = step * max(1.0, abs(theta[k]))
        up, down = theta.copy(), theta.copy()
        up[k] += h
        down[k] -= h
        numeric[:, k] = problem.mass @ (problem.rows(up) - problem.rows(down)) / (2.0 * h)
    return numeric


def fd_jacobian(problem: Any, theta: np.ndarray, step: float = 1e-5) -> np.ndarray:
    """A Richardson-extrapolated central difference of ``P_n U``, accurate to ``O(step^4)``."""
    return (4.0 * _central(problem, theta, step / 2.0) - _central(problem, theta, step)) / 3.0


def fd_error(problem: Any, theta: np.ndarray, numeric: np.ndarray | None = None) -> float:
    """The largest block-relative gap between the analytic and the numeric Jacobian."""
    analytic = problem.blocks(theta).jacobian
    numeric = fd_jacobian(problem, theta) if numeric is None else numeric
    worst = 0.0
    for rows in problem.slices.values():
        for cols in problem.slices.values():
            size = float(np.max(np.abs(numeric[rows, cols])))
            if size > 1e-6:
                gap = float(np.max(np.abs(analytic[rows, cols] - numeric[rows, cols])))
                worst = max(worst, gap / size)
    return worst


def cross_arm_blocks(problem: FoldStackProblem) -> list[tuple[str, str]]:
    """The structural zeros of 13.2: every block of arm ``a``'s rows in arm ``b``'s columns."""
    blocks = []
    arms = range(len(problem.arms))
    folds = range(problem.n_folds)
    for a in arms:
        for b in arms:
            if a == b:
                continue
            rows = [*(problem.gamma_name(a, k) for k in folds), f"eps[{a}]", f"psi[{a}]"]
            cols = [*(problem.gamma_name(b, k) for k in folds), f"eps[{b}]"]
            blocks += [(row, col) for row in rows for col in cols]
    return blocks


# --------------------------------------------------------------------------- KW0


@pytest.fixture(scope="module")
def three_arm() -> Any:
    return _fit(laws.THREE_ARM_ACTIVE.sample(3000, 71)[0])


class TestKW0SolvedSystem:
    """KW0: at ``K = 3`` the fit solves ``P_n U = 0`` block by block, arm by arm."""

    def test_every_block_is_solved(self, three_arm: Any) -> None:
        stack = fitted_stack(three_arm)
        assert len(stack.problem.arms) == 3
        means = _block_means(stack.problem, stack.theta)
        assert max(means.values()) < FLOOR, means
        assert stack.targeted_mismatch < 1e-12

    def test_the_arm_designs_carry_two_indicators(self, three_arm: Any) -> None:
        stack = fitted_stack(three_arm)
        indicators = [arm.features[:, 1:3] for arm in stack.problem.arms]
        np.testing.assert_array_equal(indicators[0], 0.0)
        np.testing.assert_array_equal(indicators[1][:, 0], 1.0)
        np.testing.assert_array_equal(indicators[2][:, 1], 1.0)

    def test_mutation_arm_one_with_arm_two_features(self, three_arm: Any) -> None:
        stack = fitted_stack(three_arm)
        arms = list(stack.problem.arms)
        arms[1] = replace(arms[1], features=arms[2].features)
        mutant = StackProblem(stack.problem.features, stack.problem.outcome, arms, 0.9995)
        means = _block_means(mutant, stack.theta)
        assert means["gamma[1]"] > 1e-3 or means["eps[1]"] > 1e-3 or means["psi[1]"] > 1e-3


# --------------------------------------------------------------------------- KW1


@pytest.fixture(scope="module")
def steep_grid() -> mq.PopulationFit:
    return mq.population_fit(
        mq.THREE_ARM_STEEP,
        *mq.hermite_nodes(16),
        bounds=(0.05, 0.95),
        features="arm_interactions_no_intercept",
    )


class _CoupledJacobian:
    """A stack whose Jacobian reads a coupled fluctuation score while its rows do not.

    The control of the cross-arm structure: arm 0's fluctuation Jacobian is built with the
    indicator ``R_0 + R_1``, as a defect that mixed the arms' rows into one score would build it.
    """

    def __init__(self, problem: StackProblem) -> None:
        arms = list(problem.arms)
        arms[0] = replace(arms[0], indicator=arms[0].indicator + arms[1].indicator)
        self._coupled = StackProblem(
            problem.features, problem.outcome, arms, problem.alpha, mass=problem.mass
        )
        self._problem = problem
        self.mass = problem.mass
        self.slices = problem.slices

    def rows(self, theta: np.ndarray) -> np.ndarray:
        return self._problem.rows(theta)

    def blocks(self, theta: np.ndarray) -> Any:
        return self._coupled.blocks(theta)


@pytest.fixture(scope="module")
def steep_numeric(steep_grid: mq.PopulationFit) -> np.ndarray:
    return fd_jacobian(steep_grid.problem, steep_grid.theta)


class TestKW1Jacobian:
    """KW1: at ``K = 3`` the analytic Jacobian equals a finite difference, block by block."""

    def test_the_grid_has_eps_and_a_binding_bound(self, steep_grid: mq.PopulationFit) -> None:
        assert np.max(np.abs(steep_grid.epsilons)) > 0.01
        assert np.max(steep_grid.clipped) > 0.05

    def test_analytic_equals_finite_difference(
        self, steep_grid: mq.PopulationFit, steep_numeric: np.ndarray
    ) -> None:
        assert fd_error(steep_grid.problem, steep_grid.theta, steep_numeric) < 1e-6

    def test_every_cross_arm_block_is_exactly_zero(
        self, steep_grid: mq.PopulationFit, steep_numeric: np.ndarray
    ) -> None:
        problem, theta = steep_grid.problem, steep_grid.theta
        analytic = problem.blocks(theta).jacobian
        numeric = steep_numeric
        for row, col in cross_arm_blocks(problem):
            block = analytic[problem.slices[row], problem.slices[col]]
            assert np.all(block == 0.0), (row, col)
            assert np.all(numeric[problem.slices[row], problem.slices[col]] == 0.0), (row, col)

    def test_mutation_arm_index_swap(
        self, steep_grid: mq.PopulationFit, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = FoldStackProblem._arm_state

        def swapped(self: FoldStackProblem, theta: np.ndarray, j: int, k: int = 0) -> Any:
            if j == 1:
                theta = theta.copy()
                one, two = self.slices["gamma[1]"], self.slices["gamma[2]"]
                theta[one], theta[two] = theta[two].copy(), theta[one].copy()
            return original(self, theta, j, k)

        monkeypatch.setattr(FoldStackProblem, "_arm_state", swapped)
        assert fd_error(steep_grid.problem, steep_grid.theta) > 1e-3

    def test_mutation_cross_arm_coupling(
        self, steep_grid: mq.PopulationFit, steep_numeric: np.ndarray
    ) -> None:
        mutant = _CoupledJacobian(steep_grid.problem)
        assert fd_error(mutant, steep_grid.theta, steep_numeric) > 1e-3


# --------------------------------------------------------------------------- KW2, BW2


@pytest.fixture(scope="module")
def three_arm_grid() -> mq.PopulationFit:
    return mq.population_fit(mq.THREE_ARM_ACTIVE, *mq.hermite_nodes(20))


@pytest.fixture(scope="module")
def large_three_arm() -> Any:
    return _fit(laws.THREE_ARM_ACTIVE.sample(200_000, 61)[0], simultaneous=True)


EXACT_SD = {
    "ey[0]": 0.9404,
    "ey[1]": 1.0176,
    "ey[2]": 1.0332,
    "ate[1 vs 0]": 1.3855,
    "ate[2 vs 0]": 1.4033,
}
"""The exact-law ``sqrt(n)`` SDs on ``three_arm_active`` (X58 plan, 13.2;
``three_arm_weighted_quadrature.log``)."""


def _quadrature_sds(fit: mq.PopulationFit, curves: np.ndarray) -> dict[str, float]:
    named = mq.contrast_curves(fit, curves, 0)
    return {name: mq.sd(fit, named[name]) for name in EXACT_SD}


class TestKW2LargeSample:
    """KW2: at one draw of ``n = 200,000`` the stacked SDs match the quadrature to 3%."""

    def test_the_quadrature_reproduces_the_plan(self, three_arm_grid: mq.PopulationFit) -> None:
        sds = _quadrature_sds(three_arm_grid, mq.arm_curves(three_arm_grid))
        for name, value in EXACT_SD.items():
            assert sds[name] == pytest.approx(value, abs=6e-5), name

    def test_the_reported_sds_match(
        self, large_three_arm: Any, three_arm_grid: mq.PopulationFit
    ) -> None:
        exact = _quadrature_sds(three_arm_grid, mq.arm_curves(three_arm_grid))
        root = np.sqrt(large_three_arm.n)
        for name, value in exact.items():
            assert large_three_arm[name].std_error * root / value == pytest.approx(1.0, abs=0.03)

    def test_mutation_shipped_curve(
        self, large_three_arm: Any, three_arm_grid: mq.PopulationFit
    ) -> None:
        stack = fitted_stack(large_three_arm)
        assert stack.shipped is not None
        exact = _quadrature_sds(three_arm_grid, mq.arm_curves(three_arm_grid))
        shipped = np.std(stack.shipped[:, 1], ddof=1)
        assert abs(shipped / exact["ey[1]"] - 1.0) > 0.1


class TestBW2LargeSampleBand:
    """BW2: the band half-widths times ``sqrt(n)`` match ``c* sd*`` of the quadrature to 3%."""

    def test_the_half_widths_match(
        self, large_three_arm: Any, three_arm_grid: mq.PopulationFit
    ) -> None:
        named = mq.contrast_curves(three_arm_grid, mq.arm_curves(three_arm_grid), 0)
        names = list(large_three_arm.simultaneous.bands)
        cov = mq.covariance(three_arm_grid, np.column_stack([named[name] for name in names]))
        exact = mq.critical_value(cov) * np.sqrt(np.diag(cov))
        root = np.sqrt(large_three_arm.n)
        for j, name in enumerate(names):
            low, high = large_three_arm.simultaneous.bands[name]
            assert (high - low) / 2.0 * root / exact[j] == pytest.approx(1.0, abs=0.03), name

    def test_mutation_shipped_curves(
        self, large_three_arm: Any, three_arm_grid: mq.PopulationFit
    ) -> None:
        named = mq.contrast_curves(three_arm_grid, mq.arm_curves(three_arm_grid), 0)
        shipped = mq.contrast_curves(three_arm_grid, mq.shipped_curves(three_arm_grid), 0)
        names = list(large_three_arm.simultaneous.bands)
        cov = mq.covariance(three_arm_grid, np.column_stack([named[name] for name in names]))
        cov_shipped = mq.covariance(
            three_arm_grid, np.column_stack([shipped[name] for name in names])
        )
        exact = mq.critical_value(cov) * np.sqrt(np.diag(cov))
        wrong = mq.critical_value(cov_shipped) * np.sqrt(np.diag(cov_shipped))
        assert abs(wrong[names.index("ey[1]")] / exact[names.index("ey[1]")] - 1.0) > 0.1


# --------------------------------------------------------------------------- KW4


@pytest.fixture(scope="module")
def u_shaped_grid() -> mq.PopulationFit:
    return mq.population_fit(mq.THREE_ARM_U_SHAPED, *mq.hermite_nodes(20), alpha=NO_SHRINK)


class TestKW4ClosedForm:
    """KW4: under a correct outcome model each ``K = 3`` row equals the closed form of 2.3."""

    def test_kappa_departs_from_one(self, u_shaped_grid: mq.PopulationFit) -> None:
        assert np.max(np.abs(mq.kappa(u_shaped_grid) - 1.0)) > 0.05
        assert np.max(np.abs(u_shaped_grid.epsilons)) < 1e-12

    def test_rows_equal_the_closed_form(self, u_shaped_grid: mq.PopulationFit) -> None:
        rows = corrected_curves(u_shaped_grid.problem, u_shaped_grid.theta).curves
        assert np.max(np.abs(rows - mq.closed_form_curves(u_shaped_grid))) < 1e-10

    def test_mutation_kappa_forced_to_one(self, u_shaped_grid: mq.PopulationFit) -> None:
        rows = corrected_curves(u_shaped_grid.problem, u_shaped_grid.theta).curves
        forced = mq.closed_form_curves(u_shaped_grid, kappas=np.ones(3))
        assert np.max(np.abs(rows - forced)) > 0.01


# --------------------------------------------------------------------------- KW6, BW3

KW6_ESTIMANDS = ("ey", "ate", "rr", "ey_obs", "par", "paf")


def _expected_reads(reference: str) -> dict[str, set[str]]:
    """The arms each estimate reads, from the definitions of 13.2 (labels "0" to "2")."""
    labels = ("0", "1", "2")
    out: dict[str, set[str]] = {f"ey[{a}]": {a} for a in labels}
    for a in labels:
        if a != reference:
            out[f"ate[{a} vs {reference}]"] = {a, reference}
            out[f"rr[{a} vs {reference}]"] = {a, reference}
    out["ey_obs"] = set()
    out[f"par[{reference}]"] = {reference}
    out[f"paf[{reference}]"] = {reference}
    return out


def _failing(monkeypatch: pytest.MonkeyPatch, arm: int) -> None:
    original = _oat_stack._arm_failure

    def mutant(stack: Any, rconds: dict[str, float], j: int) -> str | None:
        return "forced" if j == arm else original(stack, rconds, j)

    monkeypatch.setattr(_oat_stack, "_arm_failure", mutant)


@pytest.fixture(scope="module")
def kw6_frame() -> Any:
    return laws.THREE_ARM_ACTIVE.sample(1500, 73)[0]


def _statuses(frame: Any, reference: str, **options: Any) -> tuple[Any, dict[str, str]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DataWarning)
        result = _fit(frame, reference=reference, estimands=KW6_ESTIMANDS, **options)
    return result, {name: estimate.inference for name, estimate in result.estimates.items()}


def _expected_statuses(reference: str, failed: int) -> dict[str, str]:
    return {
        name: "stack_failure_plugin" if str(failed) in arms else "influence_curve"
        for name, arms in _expected_reads(reference).items()
    }


class TestKW6FailureIsolation:
    """KW6: a failed arm withholds exactly the estimates whose target reads it."""

    @pytest.mark.parametrize("reference", ["0", "1"])
    @pytest.mark.parametrize("failed", [0, 1, 2])
    def test_the_statuses_follow_the_arms_read(
        self, kw6_frame: Any, monkeypatch: pytest.MonkeyPatch, reference: str, failed: int
    ) -> None:
        _failing(monkeypatch, failed)
        _, statuses = _statuses(kw6_frame, reference)
        assert statuses == _expected_statuses(reference, failed)

    def test_a_sound_fit_withholds_nothing(self, kw6_frame: Any) -> None:
        _, statuses = _statuses(kw6_frame, "1")
        assert set(statuses.values()) == {"influence_curve"}
        assert set(statuses) == set(_expected_reads("1"))

    def test_mutation_contrasts_read_the_reference_only(
        self, kw6_frame: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = TargetContext._record_reads

        def reference_only(self: TargetContext, name: str, reads: Any) -> None:
            if name.startswith(("ate", "rr", "or")) and reads is not None:
                reads = (self.reference,)
            original(self, name, reads)

        monkeypatch.setattr(TargetContext, "_record_reads", reference_only)
        _failing(monkeypatch, 2)
        _, statuses = _statuses(kw6_frame, "1")
        assert statuses["ate[2 vs 1]"] == "influence_curve"
        assert statuses != _expected_statuses("1", 2)


class TestBW3TheBandAfterAFailure:
    """BW3: with one arm failed the band covers the estimates that supply inference only."""

    @pytest.fixture
    def failed(self, kw6_frame: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
        _failing(monkeypatch, 2)
        result, _ = _statuses(kw6_frame, "1", simultaneous=True)
        return result

    def test_the_band_covers_the_held_estimates(self, failed: Any) -> None:
        held = {
            name
            for name, status in _expected_statuses("1", 2).items()
            if status == "influence_curve"
        }
        assert failed.simultaneous is not None
        assert set(failed.simultaneous.bands) == held

    def test_the_summary_names_the_others(self, failed: Any) -> None:
        text = failed.summary()
        assert "the band is joint over the rows above only; it leaves out" in text
        for name, status in _expected_statuses("1", 2).items():
            if status != "influence_curve":
                assert f"{name} (stack_failure_plugin)" in text

    def test_mutation_unfiltered_band(self, failed: Any) -> None:
        with pytest.raises(CapabilityError):
            simultaneous_bands(failed.estimates, random_state=0)


# --------------------------------------------------------------------------- BW1


@pytest.fixture(scope="module")
def banded() -> Any:
    return _fit(laws.THREE_ARM_ACTIVE.sample(1500, 75)[0], simultaneous=True)


def _stacked_matrix(result: Any, names: list[str]) -> np.ndarray:
    stack = fitted_stack(result)
    curves = corrected_curves(stack.problem, stack.theta).curves * stack.scale
    columns = {f"ey[{a}]": curves[:, a] for a in range(3)}
    columns |= {f"ate[{a} vs 0]": curves[:, a] - curves[:, 0] for a in (1, 2)}
    return np.column_stack([columns[name] for name in names])


class TestBW1TheBandReadsTheStack:
    """BW1: the band's critical value and half-widths are those of the stacked curves."""

    def test_the_critical_value_and_the_half_widths(self, banded: Any) -> None:
        bands = banded.simultaneous
        names = list(bands.bands)
        se = np.array([banded[name].std_error for name in names])
        critical = multiplier_critical_value(
            _stacked_matrix(banded, names),
            se,
            n=banded.n,
            alpha=bands.alpha,
            n_replicates=bands.n_replicates,
            kind=bands.kind,
            random_state=0,
        )
        assert bands.critical_value == pytest.approx(critical, rel=1e-12)
        for name in names:
            low, high = bands.bands[name]
            half = critical * banded[name].std_error
            assert (high - low) / 2.0 == pytest.approx(half, rel=1e-12), name

    def test_mutation_shipped_curves(self, banded: Any) -> None:
        bands = banded.simultaneous
        names = list(bands.bands)
        stack = fitted_stack(banded)
        assert stack.shipped is not None
        shipped = stack.shipped
        columns = {f"ey[{a}]": shipped[:, a] for a in range(3)}
        columns |= {f"ate[{a} vs 0]": shipped[:, a] - shipped[:, 0] for a in (1, 2)}
        matrix = np.column_stack([columns[name] for name in names])
        se = np.std(matrix, axis=0, ddof=1) / np.sqrt(banded.n)
        critical = multiplier_critical_value(
            matrix,
            se,
            n=banded.n,
            alpha=bands.alpha,
            n_replicates=bands.n_replicates,
            kind=bands.kind,
            random_state=0,
        )
        j = names.index("ey[1]")
        low, high = bands.bands["ey[1]"]
        assert abs(critical * se[j] / ((high - low) / 2.0) - 1.0) > 0.1


# --------------------------------------------------------------------------- composition


@pytest.fixture(scope="module")
def composition() -> Any:
    frame, _ = laws.THREE_ARM_ACTIVE.sample(1500, 77)
    frame = frame.assign(wt=laws.weight(frame["W1"].to_numpy()))
    return _fit(frame, cross_fit=True, n_folds=5, weights="wt")


class TestTheComposition:
    """13.5: ``K = 3``, ``V = 5`` and weights in one stack run W0 and W1 on one sample."""

    def test_w0_every_block_is_solved(self, composition: Any) -> None:
        stack = fitted_stack(composition)
        assert stack.problem.n_folds == 5
        assert len(stack.problem.arms) == 3
        assert np.ptp(stack.problem.mass) > 0.0
        means = _block_means(stack.problem, stack.theta)
        assert max(means.values()) < FLOOR, means
        assert stack.targeted_mismatch < 1e-12

    def test_w1_analytic_equals_finite_difference(self, composition: Any) -> None:
        stack = fitted_stack(composition)
        assert fd_error(stack.problem, stack.theta) < 1e-6

    def test_every_cross_arm_block_is_exactly_zero(self, composition: Any) -> None:
        stack = fitted_stack(composition)
        analytic = stack.problem.blocks(stack.theta).jacobian
        for row, col in cross_arm_blocks(stack.problem):
            assert np.all(analytic[stack.problem.slices[row], stack.problem.slices[col]] == 0.0)

    def test_mutation_unweighted_measure(self, composition: Any) -> None:
        stack = fitted_stack(composition)
        problem = stack.problem
        mutant = FoldStackProblem(
            problem.features, problem.outcome, problem.arms, problem.alpha, folds=problem.labels
        )
        assert max(_block_means(mutant, stack.theta).values()) > 1e-4
