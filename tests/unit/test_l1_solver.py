"""Witnesses for the built-in L1 path solver (X106 K2d, plan rev 3.1).

The solver is a working-set proximal Newton method. It converges to the same optimum as the
coordinate-descent solver it replaced, not to the same bits, so its contract is checked against
a frozen copy of that solver (``_l1_reference``) run in the same process: the same library
versions move both sides alike. Each control that guards a branch is shown to fire, and each
mutation that removes a branch is shown to fail.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from scipy.special import expit, softmax

from cleverly.learners import _l1_logistic as L

from . import _l1_reference as R


def _labels(rng: np.random.Generator, eta: np.ndarray) -> np.ndarray:
    probability = softmax(eta, axis=1)
    return (rng.random(eta.shape[0])[:, None] > np.cumsum(probability, axis=1)).sum(axis=1)


def _fixture(
    kind: str, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """``(x, y, weights, grid)`` for one of the contract fixtures."""
    rng = np.random.default_rng(seed)
    n, p = 1500, 6
    if kind == "p_above_n":
        n, p = 60, 120
    x = rng.normal(size=(n, p))
    weights = None
    if kind == "collinear":
        x[:, 1] = 0.999 * x[:, 0] + np.sqrt(1 - 0.999**2) * x[:, 1]
    if kind == "constant":
        x[:, 2] = 1.5
    linear = 0.6 * x[:, 0] - 0.4 * x[:, 1] + 0.3 * x[:, 2]
    if kind == "quasi_separated":
        linear = 6.0 * x[:, 0]
    if kind == "rare":
        linear = linear - 4.0
    if kind in ("three_class", "four_class"):
        columns = [np.zeros(n), linear, -linear + 0.5 * x[:, 3]]
        if kind == "four_class":
            columns.append(0.4 * x[:, 4] - 0.3 * x[:, 0])
        y = _labels(rng, np.column_stack(columns))
    else:
        y = rng.binomial(1, expit(linear))
    if kind == "weighted":
        weights = rng.lognormal(size=n)
    if kind == "zero_weights":
        weights = np.where(rng.random(n) < 0.2, 0.0, 1.0)
    grid = L.default_penalty_grid(x, y, weights)
    if kind != "ends_at_zero":
        grid = grid[:-1]
    return x, y, weights, grid


CONTRACT = (
    "plain",
    "weighted",
    "three_class",
    "four_class",
    "collinear",
    "quasi_separated",
    "rare",
    "constant",
    "zero_weights",
    "p_above_n",
    "ends_at_zero",
)
WELL_CONDITIONED_BINARY = ("plain", "weighted", "rare", "constant", "zero_weights", "ends_at_zero")


def _paths(module, x, y, weights, grid, classes=None):  # type: ignore[no-untyped-def]
    try:
        return module.penalty_path(x, y, weights, grid, classes), None
    except Exception as raised:  # the refusal status is part of the contract
        return None, type(raised).__name__


def _membership(fit: L.L1Fit, by_covariate: bool) -> np.ndarray:
    coef = fit.coef[:, 1:] if fit.coef.shape[1] == 2 else fit.coef
    nonzero = coef != 0.0
    return np.any(nonzero, axis=1) if by_covariate else nonzero


class TestContract:
    """Against the frozen copy of the solver K2d replaced, in this process."""

    @pytest.mark.parametrize("kind", CONTRACT)
    def test_the_new_solver_meets_the_contract_of_the_old_one(self, kind: str) -> None:
        x, y, weights, grid = _fixture(kind)
        new, new_error = _paths(L, x, y, weights, grid)
        old, old_error = _paths(R, x, y, weights, grid)
        assert new_error == old_error
        if new is None:
            return
        assert old is not None
        for fit, reference in zip(new, old, strict=True):
            independent = L.kkt_residual(x, y, weights, fit)
            assert independent <= 1e-12
            # The stored residual is the solver's own; it shares the Newton step's gradient code,
            # so it is held to the independent recomputation on every penalty.
            assert abs(fit.kkt - independent) <= 1e-13
            assert np.max(np.abs(fit.predict_proba(x) - reference.predict_proba(x))) <= 1e-10
            even = fit.coef.shape[1] % 2 == 0 and fit.coef.shape[1] > 2
            assert np.array_equal(_membership(fit, even), _membership(reference, even))
            if kind in WELL_CONDITIONED_BINARY:
                assert np.max(np.abs(fit.coef - reference.coef)) <= 1e-8

    def test_an_absent_class_is_handled_as_before(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(300, 4))
        y = rng.integers(0, 2, 300)
        grid = np.array([0.05, 0.01])
        new, new_error = _paths(L, x, y, None, grid, classes=[0, 1, 2])
        old, old_error = _paths(R, x, y, None, grid, classes=[0, 1, 2])
        assert new_error == old_error
        if new is not None and old is not None:
            for fit, reference in zip(new, old, strict=True):
                assert np.max(np.abs(fit.predict_proba(x) - reference.predict_proba(x))) <= 1e-10

    def test_dropping_the_weights_from_the_gradient_fails_the_residual_check(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deliberate bug: the stored residual and the solve share ``_Point.finish``."""
        x, y, weights, grid = _fixture("weighted")
        original = L._Point.finish

        def unweighted(self: L._Point) -> L._Point:
            saved = self.problem.w
            self.problem.w = np.ones_like(saved)
            try:
                return original(self)
            finally:
                self.problem.w = saved

        monkeypatch.setattr(L._Point, "finish", unweighted)
        # Caught either way: the solve cannot meet its own stop rule and refuses, or a stored
        # residual disagrees with the independent one.
        try:
            fits = L.penalty_path(x, y, weights, grid[:20])
        except L.L1ConvergenceError:
            return
        worst = max(abs(f.kkt - L.kkt_residual(x, y, weights, f)) for f in fits)
        assert worst > 1e-13

    @pytest.mark.parametrize("classes", [2, 3])
    def test_three_runs_are_bitwise_equal(self, classes: int) -> None:
        x, y, weights, grid = _fixture("plain" if classes == 2 else "three_class", seed=5)
        digests = {
            hashlib.sha256(
                b"".join(
                    f.coef.tobytes() + f.intercept.tobytes()
                    for f in L.penalty_path(x, y, weights, grid)
                )
            ).hexdigest()
            for _ in range(3)
        }
        assert len(digests) == 1


class TestInvariants:
    """The grid, the standardization and the unpenalized fit keep their bits (K2c invariants)."""

    @pytest.mark.parametrize("kind", ["plain", "weighted", "three_class"])
    def test_the_grid_starts_at_the_null_gradient_bitwise(self, kind: str) -> None:
        x, y, weights, grid = _fixture(kind)
        labels = np.asarray(y)
        order = np.unique(labels)
        codes = np.searchsorted(order, labels).astype(np.int64)
        w = np.ones(labels.size) if weights is None else weights
        assert grid[0] == L._Problem(x, codes, w, order.size).null_gradient()
        # So the first penalty takes the intercept-only shortcut.
        assert L.penalty_path(x, y, weights, grid[:1])[0].constant is not None

    def test_c_ordering_the_design_breaks_the_shortcut_on_some_fixture(self) -> None:
        """Mutation control: the shortcut holds only because both sides share one layout."""
        broken = 0
        for seed in range(20):
            x, y, _, grid = _fixture("plain", seed)
            codes = np.asarray(y).astype(np.int64)
            problem = L._Problem(x, codes, np.ones(codes.size), 2)
            c_ordered = np.ascontiguousarray(problem.z)
            mutated = L._null_gradient(c_ordered, problem.w, problem.y)
            broken += mutated != grid[0]
        assert broken >= 1

    @pytest.mark.parametrize("kind", ["plain", "weighted", "three_class"])
    def test_the_unpenalized_fit_is_bitwise_the_old_one(self, kind: str) -> None:
        x, y, weights, _ = _fixture(kind)
        labels = np.asarray(y)
        order = np.unique(labels)
        codes = np.searchsorted(order, labels).astype(np.int64)
        w = np.ones(labels.size) if weights is None else weights
        new = L._mle(L._Problem(x, codes, w, order.size))
        old = R._mle(R._Problem(x, codes, w, order.size))
        assert new[2] == old[2]
        assert np.array_equal(new[0], old[0]) and np.array_equal(new[1], old[1])


def _problem(x: np.ndarray, y: np.ndarray) -> L._Problem:
    order = np.unique(y)
    codes = np.searchsorted(order, y).astype(np.int64)
    return L._Problem(np.asarray(x, float), codes, np.ones(y.size), order.size)


def _strong_rule_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Found by search (K2d stage 1, ``controls.py``): the sequential rule at the third penalty
    leaves out coordinate 6, which is active at the solution."""
    rng = np.random.default_rng(191)
    n, p = 1000, 20
    rho = rng.uniform(0.5, 0.95)
    cov = rho ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    x = rng.multivariate_normal(np.zeros(p), cov, size=n)
    beta = np.zeros(p)
    chosen = rng.choice(p, 6, replace=False)
    beta[chosen] = rng.choice([-1, 1], 6) * rng.uniform(0.5, 3, 6)
    y = rng.binomial(1, expit(x @ beta))
    top = L.default_penalty_grid(x, y)[0]
    return x, y, np.array([top, 0.2533346445033245 * top, 0.13572088082974532 * top])


class TestStrongRule:
    def test_the_rule_leaves_out_a_coordinate_the_solution_needs(self) -> None:
        x, y, grid = _strong_rule_fixture()
        problem = _problem(x, y)
        start = L._Point(problem, *problem.null_start()).finish()
        first = L._newton(problem, start, float(grid[1]), float(grid[0]))
        screened = set(L._working_set(first, 2 * grid[2] - grid[1]).tolist())
        second = L._newton(problem, first, float(grid[2]), float(grid[1]))
        active = set(np.flatnonzero(second.coef[:, 1] != 0.0).tolist())
        assert active - screened
        assert second.residual(float(grid[2])) <= L.KKT_TOLERANCE

    def test_without_the_violator_step_the_path_refuses(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        x, y, grid = _strong_rule_fixture()
        monkeypatch.setattr(L, "_grow", lambda work, point, penalty: work)
        with pytest.raises(L.L1ConvergenceError):
            L.penalty_path(x, y, None, grid)


def _model_exits(classes: int, seed: int = 3) -> list[float]:
    """Each quadratic model's exit residual over its threshold, along one default-grid path."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(3000, 6))
    if classes == 2:
        y = rng.binomial(1, expit(x[:, 0] - x[:, 1]))
    else:
        columns = [np.zeros(3000)] + [
            k * 0.4 * x[:, k] - 0.3 * x[:, (k + 1) % 6] for k in range(1, classes)
        ]
        y = _labels(rng, np.column_stack(columns))
    ratios: list[float] = []
    original = L._quadratic

    def recording(hessian, gradient, start, penalized, penalty, threshold):  # type: ignore[no-untyped-def]
        step = original(hessian, gradient, start, penalized, penalty, threshold)
        residual = L._model_residual(gradient + hessian @ step, start + step, penalized, penalty)
        ratios.append(residual / threshold)
        return step

    L._quadratic = recording  # restored below; the test does not share the module across threads
    try:
        L.penalty_path(x, y, None, L.default_penalty_grid(x, y)[:-1])
    finally:
        L._quadratic = original
    return ratios


def _unchecked(hessian, gradient, start, step, penalized, penalty, threshold):  # type: ignore[no-untyped-def]
    """``_sign_fixed`` with every acceptance check removed (mutation)."""
    values = start + step
    free = ~penalized | (values != 0.0)
    signs = np.where(penalized, np.sign(values), 0.0)
    trial = -start.copy()
    trial[free] = 0.0
    right = -(gradient[free] + penalty * signs[free]) - hessian[np.ix_(free, ~free)] @ trial[~free]
    trial[free] = np.linalg.lstsq(hessian[np.ix_(free, free)], right, rcond=None)[0]
    return trial


class TestQuadraticModel:
    @pytest.mark.parametrize("classes", [2, 3, 4])
    def test_every_model_exits_at_its_stop_rule(self, classes: int) -> None:
        ratios = _model_exits(classes)
        assert ratios and max(ratios) <= 1.0

    def test_without_the_acceptance_checks_a_model_exits_above_its_rule(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(L, "_sign_fixed", _unchecked)
        assert max(_model_exits(4)) > 1.0

    def test_an_inconsistent_sign_fixed_system_is_rejected(self) -> None:
        """The free-coordinate check: a singular system whose least-squares point keeps its
        signs and has no zero coordinate, yet is not stationary."""
        hessian = np.array([[1.0, 1.0], [1.0, 1.0]])
        gradient = np.array([-3.0, -2.0])
        start = np.array([1.0, 1.0])
        penalized = np.array([True, True])
        assert L._sign_fixed(hessian, gradient, start, np.zeros(2), penalized, 0.5, 1e-12) is None
        mutant = _unchecked(hessian, gradient, start, np.zeros(2), penalized, 0.5, 1e-12)
        assert L._model_residual(gradient + hessian @ mutant, start + mutant, penalized, 0.5) > 0.1


class TestLineSearch:
    def test_an_overshooting_step_is_halved(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.normal(size=(2000, 8))
        y = rng.binomial(1, expit(x[:, 0] - 0.5 * x[:, 1]))
        problem = _problem(x, y)
        penalty = L._null_gradient(problem.z, problem.w, problem.y) * 0.05
        start = L._Point(problem, *problem.null_start()).finish()
        solution = L._newton(problem, start, penalty, None)
        intercept_step = 8 * (solution.intercept - start.intercept)
        coef_step = 8 * (solution.coef - start.coef)
        trial = L._line_search(problem, start, intercept_step, coef_step, penalty)
        assert trial is not None
        size = (trial.intercept[1] - start.intercept[1]) / intercept_step[1]
        assert size < 1.0
        assert trial.value(penalty) <= start.value(penalty) + L._ROUNDING * max(
            1.0, abs(start.value(penalty))
        )
        # Mutation: the unit step would raise the objective beyond the rounding allowance.
        unit = L._Point(problem, start.intercept + intercept_step, start.coef + coef_step)
        assert unit.value(penalty) > start.value(penalty) + L._ROUNDING * max(
            1.0, abs(start.value(penalty))
        )


class TestWorkGuard:
    """Full-data passes (one evaluation over every row) on a p = 20 default-grid path."""

    BOUND = 978  # 1.25 times the stage-1 count of 782 (K2d stage 1, controls.log)

    def _data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(4)
        x = rng.normal(size=(2000, 20))
        y = rng.binomial(1, expit(0.5 * x[:, 0] - 0.5 * x[:, 1] + 0.3 * x[:, 2]))
        return x, y, L.default_penalty_grid(x, y)[:-1]

    def test_the_path_stays_within_its_pass_bound(self, monkeypatch: pytest.MonkeyPatch) -> None:
        x, y, grid = self._data()
        passes = [0]
        init, finish, gram = L._Point.__init__, L._Point.finish, L._gram

        def counted_init(self, problem, intercept, coef):  # type: ignore[no-untyped-def]
            passes[0] += 1
            init(self, problem, intercept, coef)

        def counted_finish(self):  # type: ignore[no-untyped-def]
            passes[0] += 1
            return finish(self)

        def counted_gram(problem, point, columns):  # type: ignore[no-untyped-def]
            passes[0] += 1
            return gram(problem, point, columns)

        monkeypatch.setattr(L._Point, "__init__", counted_init)
        monkeypatch.setattr(L._Point, "finish", counted_finish)
        monkeypatch.setattr(L, "_gram", counted_gram)
        L.penalty_path(x, y, None, grid)
        assert passes[0] <= self.BOUND

    def test_the_old_solver_took_at_least_ten_times_the_bound(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Positive control: the replaced solver's passes, counted on its objective, residual and
        Hessian calls and its coordinate updates (an instrumented copy of its ``_cd_class``)."""
        x, y, grid = self._data()
        passes = [0]
        objective, kkt, hessian = R._Problem.objective, R._Problem.kkt, R._hessian_parts

        def counted_cd(problem, eta, intercept, coef, k, penalty):  # type: ignore[no-untyped-def]
            z, w, n = problem.z, problem.w, problem.n
            p = expit(eta[:, 1] - eta[:, 0]) if problem.binary else R._probabilities(eta)[:, k]
            target = problem.y[:, 1] if problem.binary else problem.y[:, k]
            curvature = np.clip(p * (1.0 - p), R._CURVATURE_FLOOR, None)
            v = w * curvature / n
            residual = (target - p) / curvature
            column_scale = (v[:, None] * z**2).sum(axis=0)
            passes[0] += 1
            for _ in range(R._MAX_SWEEPS):
                largest = 0.0
                shift = float(v @ residual) / float(np.sum(v))
                intercept[k] += shift
                residual -= shift
                largest = max(largest, abs(shift) * np.sqrt(float(np.sum(v))))
                for j in range(problem.p):
                    if column_scale[j] <= 0.0:
                        continue
                    passes[0] += 1
                    old = coef[j, k]
                    update = float(z[:, j] @ (v * residual)) + column_scale[j] * old
                    new = R._soft(update, penalty) / column_scale[j]
                    if new != old:
                        residual -= z[:, j] * (new - old)
                        coef[j, k] = new
                        largest = max(largest, abs(new - old) * np.sqrt(column_scale[j]))
                if largest < 1e-9:
                    break

        def counted(function):  # type: ignore[no-untyped-def]
            def wrapped(*args, **kwargs):  # type: ignore[no-untyped-def]
                passes[0] += 1
                return function(*args, **kwargs)

            return wrapped

        plain = R.penalty_path(x, y, None, grid)
        monkeypatch.setattr(R, "_cd_class", counted_cd)
        monkeypatch.setattr(R._Problem, "objective", counted(objective))
        monkeypatch.setattr(R._Problem, "kkt", counted(kkt))
        monkeypatch.setattr(R, "_hessian_parts", counted(hessian))
        instrumented = R.penalty_path(x, y, None, grid)
        assert all(np.array_equal(a.coef, b.coef) for a, b in zip(plain, instrumented, strict=True))
        assert passes[0] >= 10 * self.BOUND
