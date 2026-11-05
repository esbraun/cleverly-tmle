"""The adaptive TMLE engine on random draws: solvers, dictionary, fold roles and the fit.

The exact-law checks of the projection parameter are in ``test_atmle_exact.py``. These tests
check the pieces that must hold on every sample: the R-loss lasso against its optimality
conditions and an independent solver, the refit and fluctuation score equations, the support
screen and full rank of the default dictionary, which rows each fit and each pooled step of
the cross-validated form reads, and the bookkeeping of a fit.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.estimators import atmle
from cleverly.estimators._working_basis import (
    WorkingBasis,
    WorkingDictionary,
    independent_columns,
    minimum_support,
)
from cleverly.estimators.atmle import (
    Nuisances,
    WorkingUnit,
    evaluate,
    fit_adaptive_tmle,
    lasso_solutions,
    penalty_grid,
    refit_unit,
    select_working_model,
    target_source,
)
from cleverly.exceptions import DataError, MethodConfigurationError, PositivityWarning
from cleverly.learners.crossfit import random_partition

BOUNDS = (0.01, 0.99)


def lasso_problem(seed: int, n: int = 200, p: int = 8, k: int = 2) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    basis = np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))])
    if k == 2:
        basis[:, 1] = rng.binomial(1, 0.5, n)
    multiplier = rng.uniform(-0.9, 0.9, n)
    beta = np.where(rng.uniform(size=p) < 0.5, rng.normal(size=p), 0.0)
    response = multiplier * (basis @ beta) + rng.normal(size=n)
    return basis, multiplier, response


def direct_lasso(
    basis: np.ndarray, multiplier: np.ndarray, response: np.ndarray, k: int, penalty: float
) -> np.ndarray:
    """Cyclic coordinate descent on the stated objective, penalty factor 0 on the first k."""
    x = multiplier[:, None] * basis
    weights = multiplier**2
    total = weights.sum()
    center = weights @ basis / total
    scale = np.sqrt(weights @ (basis - center) ** 2 / total)
    factor = np.where(np.arange(basis.shape[1]) < k, 0.0, scale)
    curvature = (x**2).sum(axis=0) / total
    b = np.zeros(basis.shape[1])
    residual = response.copy()
    for _ in range(200_000):
        largest = 0.0
        for j in range(b.size):
            rho = x[:, j] @ residual / total + curvature[j] * b[j]
            new = np.sign(rho) * max(abs(rho) - penalty * factor[j], 0.0) / curvature[j]
            change = new - b[j]
            if change:
                residual -= change * x[:, j]
                b[j] = new
                largest = max(largest, abs(change))
        if largest < 1e-14:
            return b
    raise AssertionError("the reference coordinate descent did not converge")


# --------------------------------------------------------------------------- the lasso


@pytest.mark.parametrize("seed", range(4))
def test_the_lasso_at_penalty_zero_is_least_squares(seed: int) -> None:
    basis, m, r = lasso_problem(seed)
    (solution,) = lasso_solutions(basis, m, r, 2, [0.0])
    expected = np.linalg.lstsq(m[:, None] * basis, r, rcond=None)[0]
    np.testing.assert_allclose(solution.coef, expected, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("k", [1, 2])
def test_above_the_largest_penalty_only_unpenalized_columns_are_nonzero(k: int) -> None:
    basis, m, r = lasso_problem(3, k=k)
    top = penalty_grid(basis, m, r, k)[0]
    for solution in lasso_solutions(basis, m, r, k, [2.0 * top, top]):
        assert np.all(solution.coef[k:] == 0.0)
        assert np.any(solution.coef[:k] != 0.0)
        expected = np.linalg.lstsq((m[:, None] * basis)[:, :k], r, rcond=None)[0]
        np.testing.assert_allclose(solution.coef[:k], expected, rtol=1e-10)
    # Witness: just below the largest penalty a penalized column enters.
    (below,) = lasso_solutions(basis, m, r, k, [0.95 * top])
    assert np.any(below.coef[k:] != 0.0)


def _dictionary_problems() -> list[tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    out = []
    for seed in range(6):
        rng = np.random.default_rng(seed)
        n = 300 + 150 * seed
        w = rng.normal(size=(n, 3))
        w[:, 2] = rng.integers(0, 3, n)
        a = rng.binomial(1, 0.4, n).astype(float)
        source = rng.binomial(1, 0.5, n).astype(float)
        dictionary = WorkingDictionary.fit(None, w, a, source, np.arange(n), controls_only=False)
        for basis, k in ((dictionary.phi(w), 1), (dictionary.psi(w, a), 2)):
            m = rng.uniform(-0.9, 0.9, n)
            beta = rng.normal(size=basis.shape[1]) * (rng.uniform(size=basis.shape[1]) < 0.3)
            out.append((basis, m, m * (basis @ beta) + rng.normal(size=n), k))
    return out


@pytest.mark.parametrize("index", range(12))
def test_every_grid_solution_meets_the_optimality_conditions(index: int) -> None:
    basis, m, r, k = _dictionary_problems()[index]
    grid = penalty_grid(basis, m, r, k)
    assert grid.size == 100
    for solution in lasso_solutions(basis, m, r, k, grid):
        assert solution.kkt <= atmle.KKT_TOLERANCE * grid[0]
        assert solution.kkt == pytest.approx(
            atmle.kkt_residual(basis, m, r, k, solution), rel=1e-6, abs=1e-15
        )


@pytest.mark.parametrize(("seed", "k"), [(0, 1), (1, 2), (2, 2), (5, 1)])
def test_the_partialled_lasso_equals_a_direct_solve_with_penalty_factor_zero(
    seed: int, k: int
) -> None:
    basis, m, r = lasso_problem(seed, k=k)
    top = penalty_grid(basis, m, r, k)[0]
    penalties = [0.5 * top, 0.1 * top, 0.01 * top]
    for penalty, solution in zip(
        penalties, lasso_solutions(basis, m, r, k, penalties), strict=True
    ):
        np.testing.assert_allclose(
            solution.coef, direct_lasso(basis, m, r, k, penalty), rtol=1e-6, atol=1e-9
        )


def test_the_kkt_residual_rejects_a_wrong_solution() -> None:
    basis, m, r = lasso_problem(1)
    top = penalty_grid(basis, m, r, 2)[0]
    (solution,) = lasso_solutions(basis, m, r, 2, [0.1 * top])
    shifted = atmle.LassoSolution(solution.penalty, solution.coef * 1.01, 0.0)
    assert atmle.kkt_residual(basis, m, r, 2, shifted) > 1e3 * atmle.KKT_TOLERANCE * top


def test_selection_takes_the_largest_penalty_at_the_smallest_risk() -> None:
    basis, m, r = lasso_problem(2, n=400)
    inner = random_partition(400, 5, seed=3).assignment
    selection = select_working_model(basis, m, r, 2, inner)
    smallest = selection.risk.min()
    assert selection.penalty == selection.grid[np.flatnonzero(selection.risk == smallest)[0]]
    nonzero = np.flatnonzero(selection.coef != 0.0)
    np.testing.assert_array_equal(selection.selected, np.union1d([0, 1], nonzero))
    # The risk at a penalty is the held-out R-loss, recomputed here at the chosen one.
    loss = 0.0
    for fold in range(5):
        held = inner == fold
        path = lasso_solutions(basis[~held], m[~held], r[~held], 2, selection.grid)
        chosen = path[int(np.flatnonzero(selection.grid == selection.penalty)[0])]
        loss += float(np.sum((r[held] - m[held] * (basis[held] @ chosen.coef)) ** 2))
    assert loss / np.sum(m**2) == pytest.approx(smallest, rel=1e-10)


# --------------------------------------------------------------------------- scores


def random_unit(seed: int, *, controls_only: bool = False, missing: bool = False) -> WorkingUnit:
    rng = np.random.default_rng(seed)
    n = 400
    w = rng.normal(size=(n, 2))
    s = rng.binomial(1, 0.5, n).astype(float)
    a = rng.binomial(1, 0.5, n).astype(float)
    if controls_only:
        a[s == 0] = 0.0
    y = 1 + w[:, 0] + a + rng.normal(size=n)
    phi = np.column_stack([np.ones(n), w])
    arm = lambda value: np.broadcast_to(value, (n,))  # noqa: E731

    def psi(at: np.ndarray) -> np.ndarray:
        if controls_only:
            return phi
        return np.column_stack([np.ones(n), at, w[:, 0], at * w[:, 1]])

    source1 = np.ones(n) if controls_only else rng.uniform(0.2, 0.8, n)
    observed = rng.uniform(size=n) < 0.8 if missing else None
    nuisances = Nuisances(
        g=rng.uniform(0.2, 0.8, n),
        theta=rng.normal(size=n),
        source0=rng.uniform(0.2, 0.8, n),
        source1=source1,
        qt=rng.normal(size=n),
        pibar=rng.uniform(0.2, 0.8, n),
        q=rng.normal(size=n) if missing else None,
        g_delta=rng.uniform(0.6, 0.95, n) if missing else None,
    )
    return WorkingUnit(
        outcome=np.where(observed, y, np.nan) if missing else y,
        treatment=a,
        source=s,
        observed=observed,
        phi=phi,
        psi=psi(a),
        psi0=psi(arm(0.0)),
        psi1=psi(arm(1.0)),
        nuisances=nuisances,
        weights=np.full(n, 1.0 / n),
        controls_only=controls_only,
    )


UNITS = [
    {},
    {"controls_only": True},
    {"missing": True},
    {"controls_only": True, "missing": True},
]


@pytest.mark.parametrize("options", UNITS)
@pytest.mark.parametrize("seed", range(3))
def test_the_relaxed_refit_solves_the_working_model_scores(seed: int, options: Any) -> None:
    unit = random_unit(seed, **options)
    effect_fit, bias_fit = refit_unit(unit)
    assert effect_fit is not None
    assert effect_fit.relative < 1e-10
    assert bias_fit.relative < 1e-10
    for population in ("pooled", "trial"):
        _, parts = evaluate(unit, effect_fit.beta, bias_fit.beta, population)
        for term in (parts.effect_model, parts.bias_model):
            scale = float(np.mean(np.abs(term)))
            assert scale > 1e-3
            assert abs(float(np.mean(term))) < 1e-10 * scale


@pytest.mark.parametrize("seed", range(5))
def test_the_source_fluctuation_solves_its_score(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 500
    source = rng.uniform(0.02, 0.98, n)
    membership = rng.binomial(1, np.clip(source + rng.normal(0, 0.2, n), 0.01, 0.99), n)
    covariate = rng.normal(size=n) * rng.uniform(0.1, 20.0)
    weights = rng.uniform(size=n)
    weights /= weights.sum()
    epsilon = target_source(source, covariate, membership.astype(float), weights)
    fluctuated = 1.0 / (1.0 + np.exp(-(np.log(source / (1 - source)) + epsilon * covariate)))
    score = float(np.sum(weights * covariate * (membership - fluctuated)))
    assert abs(score) <= 1e-10 * float(np.sum(weights * np.abs(covariate)))
    # Witness: the starting score is not already zero.
    assert abs(float(np.sum(weights * covariate * (membership - source)))) > 1e-4


# --------------------------------------------------------------------------- the dictionary


def test_the_default_dictionary_has_full_rank_and_indicator_columns_for_few_levels() -> None:
    rng = np.random.default_rng(0)
    n = 500
    w = np.column_stack([rng.normal(size=n), rng.binomial(1, 0.4, n), rng.integers(0, 3, n)])
    a = rng.binomial(1, 0.5, n).astype(float)
    s = rng.binomial(1, 0.5, n).astype(float)
    basis = WorkingBasis().fit(w)
    names = list(basis.get_feature_names_out())
    assert names[:6] == ["x0", *(f"x0_hinge{b}" for b in range(5))]
    assert names[6:] == ["x1==1", "x2==1", "x2==2", "x0*x1", "x0*x2", "x1*x2"]
    assert basis.transform(w).shape[1] == len(names)
    assert basis.unscreened_.tolist() == [True] + [False] * 11
    for controls_only in (False, True):
        rows = np.flatnonzero(a == 0) if controls_only else np.arange(n)
        dictionary = WorkingDictionary.fit(None, w, a, s, rows, controls_only=controls_only)
        phi = dictionary.phi(w)
        psi = dictionary.psi(w[rows], a[rows])
        # Linear plus hinges on a generic continuous covariate is full rank and well
        # supported here: neither the screen nor the pivoted QR drops a column.
        assert dictionary.unsupported_phi_names == dictionary.unsupported_psi_names == ()
        assert np.linalg.matrix_rank(phi) == phi.shape[1] == 1 + len(names)
        assert np.linalg.matrix_rank(psi) == psi.shape[1] == len(dictionary.psi_names)
        assert psi.shape[1] == (1 + len(names) if controls_only else 2 + 2 * len(names))


def test_the_pivoted_qr_drops_exactly_collinear_columns() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(100, 3))
    matrix = np.column_stack(
        [np.ones(100), x, x[:, 0] + 2 * x[:, 1], np.full(100, 3.0), np.zeros(100), x[:, 2] * 1e-6]
    )
    kept = independent_columns(matrix, 1)
    assert kept.tolist()[0] == 0
    assert len(kept) == 4
    assert 5 not in kept and 6 not in kept  # the constant multiple and the zero column
    assert np.linalg.matrix_rank(matrix[:, kept]) == 4
    with pytest.raises(ValueError, match="collinear"):
        independent_columns(np.column_stack([np.ones(100), np.full(100, 2.0), x]), 2)


def test_a_tied_covariate_keeps_only_independent_hinge_columns() -> None:
    w = np.r_[np.zeros(50), np.linspace(0, 1, 9)][:, None]
    a = np.tile([0.0, 1.0], 30)[:59]
    dictionary = WorkingDictionary.fit(
        None, w, a, np.r_[np.zeros(30), np.ones(29)], np.arange(59), controls_only=False
    )
    phi = dictionary.phi(w)
    # The hinges at the tied knots equal the linear term plus a constant.
    assert np.linalg.matrix_rank(phi) == phi.shape[1] < 7


def _screen_draw(k: int) -> tuple[np.ndarray, ...]:
    """400 rows; ``k`` of the 80 external treated rows lie above the top hinge knot."""
    rng = np.random.default_rng(3)
    n = 400
    s = np.r_[np.ones(200), np.zeros(200)]
    a = np.r_[np.tile([0.0, 1.0], 100), np.zeros(120), np.ones(80)]
    x0 = rng.uniform(size=n)
    x0[320:] = rng.uniform(0.0, 0.5, 80)
    x0[320 : 320 + k] = 0.995 + 0.001 * np.arange(k)
    return np.column_stack([x0, rng.normal(size=n)]), a, s


@pytest.mark.parametrize("k", [9, 11])
def test_the_support_screen_drops_a_hinge_few_external_treated_rows_carry(k: int) -> None:
    w, a, s = _screen_draw(k)
    assert minimum_support(400) == 10
    dictionary = WorkingDictionary.fit(None, w, a, s, np.arange(400), controls_only=False)
    top = "x0_hinge4"
    z = dictionary.phi(w)[:, list(dictionary.phi_names).index(top)]
    assert int(np.sum((z != 0) & (s == 0) & (a == 1))) == k
    assert top in dictionary.psi_names
    # The linear term and its A product are never screened.
    assert {"x0", "A*x0"} <= set(dictionary.psi_names)
    if k < 10:
        assert f"A*{top}" in dictionary.unsupported_psi_names
        assert f"A*{top}" not in dictionary.psi_names
    else:
        assert f"A*{top}" not in dictionary.unsupported_psi_names
        assert f"A*{top}" in dictionary.psi_names


def test_the_support_screen_keeps_the_linear_term_with_few_external_treated_rows() -> None:
    w, a, s = _screen_draw(0)
    a[320:395] = 0.0  # 5 external treated rows, below k_min
    dictionary = WorkingDictionary.fit(None, w, a, s, np.arange(400), controls_only=False)
    assert {"A", "A*x0", "A*x1"} <= set(dictionary.psi_names)
    # Every other A column is nonzero on at most 5 external treated rows and is dropped.
    hinges = [f"A*x{j}_hinge{b}" for j in range(2) for b in range(5)]
    assert dictionary.unsupported_psi_names == (*hinges, "A*x0*x1")


# --------------------------------------------------------------------------- fold roles

SPY: list[dict[str, Any]] = []
BASIS_FITS: list[set[int]] = []
SELECTIONS: list[dict[str, np.ndarray]] = []
REFITS: list[dict[str, np.ndarray]] = []
STEPS: list[list[dict[str, np.ndarray]]] = []


def _record(key: int, x: np.ndarray, values: np.ndarray) -> None:
    store = SPY[key]["predictions"]
    for row, value in zip(x[:, 0].astype(int), values, strict=True):
        store.setdefault(int(row), []).append(float(value))


class SpyRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, role: str = "outcome") -> None:
        self.role = role

    def fit(self, X: np.ndarray, y: np.ndarray) -> SpyRegressor:
        self.model_ = LinearRegression().fit(X, y)
        self.key_ = len(SPY)
        SPY.append({"role": self.role, "train": set(X[:, 0].astype(int)), "predictions": {}})
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = self.model_.predict(X)
        _record(self.key_, X, out)
        return out


class SpyClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, role: str = "treatment") -> None:
        self.role = role

    def fit(self, X: np.ndarray, y: np.ndarray) -> SpyClassifier:
        self.model_ = LogisticRegression().fit(X, y)
        self.classes_ = self.model_.classes_
        self.key_ = len(SPY)
        SPY.append({"role": self.role, "train": set(X[:, 0].astype(int)), "predictions": {}})
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        out = self.model_.predict_proba(X)
        _record(self.key_, X, out[:, 1])
        return out


class SpyBasis(TransformerMixin, BaseEstimator):
    def fit(self, X: np.ndarray, y: Any = None) -> SpyBasis:
        BASIS_FITS.append(set(np.asarray(X)[:, 0].astype(int)))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float)


def trial_draw(seed: int, n: int, *, controls_only: bool = False) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    w = np.column_stack([np.arange(n, dtype=float), rng.normal(size=n)])
    s = rng.binomial(1, 1 / (1 + np.exp(-w[:, 1])), n).astype(float)
    a = np.where(s == 1, rng.binomial(1, 0.5, n), rng.binomial(1, 0.3, n)).astype(float)
    if controls_only:
        a[s == 0] = 0.0
    y = 1 + w[:, 1] + 1.5 * a + (1 - s) * (0.5 + 0.5 * w[:, 1]) + rng.normal(size=n)
    return {"outcome": y, "treatment": a, "source": s, "covariates": w}


def _producer_ok(role: str, row: int, value: float, forbidden: set[int]) -> tuple[bool, bool]:
    """Whether some fit of ``role`` predicted ``value`` at ``row``, and none of them read
    a forbidden row."""
    found, clean = False, True
    for model in SPY:
        if model["role"] != role:
            continue
        if any(
            np.isclose(v, value, rtol=1e-10, atol=1e-12) for v in model["predictions"].get(row, [])
        ):
            found = True
            clean = clean and not (model["train"] & forbidden)
    return found, clean


def test_no_validation_row_reaches_its_folds_nuisance_or_selection_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    SPY.clear()
    BASIS_FITS.clear()
    SELECTIONS.clear()
    REFITS.clear()
    STEPS.clear()
    original = atmle.select_working_model
    original_refit = atmle.relaxed_refit
    original_step = atmle.pooled_epsilon
    original_directions = atmle.target_directions
    directions_ids: list[set[int]] = []

    def directions(unit, population):  # type: ignore[no-untyped-def]
        directions_ids.append(set(unit.phi[:, 1].astype(int).tolist()))
        return original_directions(unit, population)

    def recorded(basis, multiplier, response, n_unpenalized, inner):  # type: ignore[no-untyped-def]
        SELECTIONS.append(
            {"basis": basis, "multiplier": multiplier, "response": response, "k": n_unpenalized}
        )
        chosen = original(basis, multiplier, response, n_unpenalized, inner)
        # Keep every column, so the row-id column reaches the refit and the steps.
        return replace(chosen, selected=np.arange(basis.shape[1], dtype=np.int64))

    def refit(basis, multiplier, response, weights, *, where="The refit"):  # type: ignore[no-untyped-def]
        REFITS.append({"basis": basis, "multiplier": multiplier, "response": response})
        return original_refit(basis, multiplier, response, weights, where=where)

    def step(units, beta_a, beta_s, directions, model):  # type: ignore[no-untyped-def]
        STEPS.append(
            [
                {
                    "ids": u.phi[:, 1].astype(int),
                    "g": u.nuisances.g,
                    "source": u.nuisances.source_at(u.treatment),
                    "qt": u.nuisances.qt,
                }
                for u in units
            ]
        )
        return original_step(units, beta_a, beta_s, directions, model)

    monkeypatch.setattr(atmle, "select_working_model", recorded)
    monkeypatch.setattr(atmle, "relaxed_refit", refit)
    monkeypatch.setattr(atmle, "pooled_epsilon", step)
    monkeypatch.setattr(atmle, "target_directions", directions)
    n, folds = 240, 4
    data = trial_draw(4, n)
    fit = fit_adaptive_tmle(
        **data,
        outcome_learner=SpyRegressor("outcome"),
        treatment_learner=SpyClassifier("treatment"),
        source_learner=SpyClassifier("source"),
        g_bounds=(1e-9, 1 - 1e-9),
        basis=SpyBasis(),
        n_folds=folds,
        inner_folds=3,
    )
    assignment = fit.folds.assignment
    fold_rows = [set(np.flatnonzero(assignment == v).tolist()) for v in range(folds)]
    # Every learner fit trains on whole folds, leaves one out, and predicts only rows it
    # did not read.
    for model in SPY:
        read = {v for v in range(folds) if model["train"] & fold_rows[v]}
        assert model["train"] == set().union(*(fold_rows[v] for v in read))
        assert len(read) < folds
        assert not model["train"] & set(model["predictions"])
    # Each fold's dictionary is fitted on its training rows alone.
    assert sorted(map(sorted, BASIS_FITS)) == sorted(
        sorted(set(range(n)) - fold_rows[v]) for v in range(folds)
    )
    # Each fold's selection reads its training rows alone, with nuisances fitted without
    # the fold and without the row's own fold.
    y, a, s = data["outcome"], data["treatment"], data["source"]
    assert len(SELECTIONS) == 2 * folds
    for index, call in enumerate(SELECTIONS):
        # Each fold selects its effect model, then its bias model.
        effect = index % 2 == 0
        assert call["k"] == 1
        ids = call["basis"][:, 1 if effect else 2].astype(int)
        (v,) = [v for v in range(folds) if set(ids.tolist()) == set(range(n)) - fold_rows[v]]
        for position, row in enumerate(ids):
            forbidden = fold_rows[v] | fold_rows[int(assignment[row])]
            if effect:
                checks = [
                    ("treatment", a[row] - call["multiplier"][position]),
                    ("outcome", y[row] - call["response"][position]),
                ]
            else:
                checks = [
                    ("source", s[row] - call["multiplier"][position]),
                    ("outcome", y[row] - call["response"][position]),
                ]
            for role, value in checks:
                found, clean = _producer_ok(role, int(row), float(value), forbidden)
                assert found and clean, (role, v, row)
    # Each fold's refit reads its training rows alone, with the selection's nuisances.
    assert len(REFITS) == 2 * folds
    for call in REFITS:
        effect = call["basis"].shape[1] == 3
        ids = call["basis"][:, 1 if effect else 2].astype(int)
        (v,) = [v for v in range(folds) if set(ids.tolist()) == set(range(n)) - fold_rows[v]]
        for position, row in enumerate(ids):
            forbidden = fold_rows[v] | fold_rows[int(assignment[row])]
            producer, observed = ("treatment", a) if effect else ("source", s)
            for role, value in (
                (producer, observed[row] - call["multiplier"][position]),
                ("outcome", y[row] - call["response"][position]),
            ):
                found, clean = _producer_ok(role, int(row), float(value), forbidden)
                assert found and clean, ("refit", role, v, row)
    # The CV directions I^{-1} m are computed on each fold's training rows T_v, never on the
    # validation rows V_v (a mutation that passes V_v leaves every id set equal to a fold).
    training = [set(range(n)) - fold_rows[v] for v in range(folds)]
    assert directions_ids
    assert all(ids in training for ids in directions_ids)
    assert all(training[v] in directions_ids for v in range(folds))
    # Each pooled step reads the validation folds alone, with nuisances fitted without them.
    assert len(STEPS) == 2 + fit.passes
    for index, units in enumerate(STEPS):
        assert len(units) == folds
        for v, unit in enumerate(units):
            assert set(unit["ids"].tolist()) == fold_rows[v]
            for position, row in enumerate(unit["ids"]):
                checks = [("treatment", unit["g"]), ("outcome", unit["qt"])]
                if index < 2:
                    # Pi is still the cross-fitted nuisance before the first fluctuation.
                    checks.append(("source", unit["source"]))
                for role, value in checks:
                    found, clean = _producer_ok(
                        role, int(row), float(value[position]), fold_rows[v]
                    )
                    assert found and clean, ("step", role, v, row)


def test_the_fold_role_check_sees_a_leak(monkeypatch: pytest.MonkeyPatch) -> None:
    """Witness: selection nuisances taken from the training fits read the fold's own rows."""
    SPY.clear()
    data = trial_draw(5, 240)
    fit = fit_adaptive_tmle(
        **data,
        outcome_learner=SpyRegressor("outcome"),
        treatment_learner=SpyClassifier("treatment"),
        source_learner=SpyClassifier("source"),
        g_bounds=(1e-9, 1 - 1e-9),
        form="pooled",
        n_folds=3,
        inner_folds=3,
    )
    assignment = fit.folds.assignment
    leaks = 0
    for row in range(240):
        own = set(np.flatnonzero(assignment == assignment[row]).tolist())
        other = (int(assignment[row]) + 1) % 3
        forbidden = set(np.flatnonzero(assignment == other).tolist())
        for model in SPY:
            if row in model["predictions"] and not model["train"] & own:
                leaks += bool(model["train"] & forbidden)
    assert leaks > 0


# --------------------------------------------------------------------------- the fit


def _fit(data: dict[str, np.ndarray], **options: Any) -> atmle.AdaptiveTMLEFit:
    return fit_adaptive_tmle(
        **data,
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(),
        source_learner=LogisticRegression(),
        g_bounds=BOUNDS,
        **options,
    )


def _shifted_draw(seed: int, n: int, shift: tuple[float, float]) -> dict[str, np.ndarray]:
    """A trial with external rows in both arms whose outcomes are shifted by ``shift[a]``."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n, 2))
    s = rng.binomial(1, 0.3, n).astype(float)
    external = rng.binomial(1, 1.0 / (1.0 + np.exp(-0.5 * w[:, 0])), n)
    a = np.where(s == 1, rng.binomial(1, 0.5, n), external).astype(float)
    bias = np.where(a == 1.0, shift[1], shift[0])
    y = 1 + w[:, 0] + 0.5 * w[:, 1] + 1.5 * a + (1 - s) * bias + rng.normal(size=n)
    return {"outcome": y, "treatment": a, "source": s, "covariates": w}


def test_the_bias_model_penalizes_the_arm_column_so_compatible_arms_add_information() -> None:
    """Only the bias model's constant is unpenalized (Algorithm 1's lasso intercept).

    A shift common to both external arms leaves their contrast informative: the lasso drops
    ``A`` and the standard error falls well below the trial-only bound
    ``sqrt(4 / n_trial)`` (unit noise, randomization 1/2). Forcing ``A`` in gives the arms
    free shifts and returns that bound (0.081 on this draw against 0.045). An arm-specific
    shift is still found: ``A`` is selected on every fold.
    """
    common = _shifted_draw(0, 2_000, (0.5, 0.5))
    fit = _fit(common)
    bound = float(np.sqrt(4.0 / common["source"].sum()))
    assert all("A" not in model.bias_columns for model in fit.working_models)
    assert fit.std_error < 0.7 * bound
    arm = _fit(_shifted_draw(0, 2_000, (0.0, 3.0)))
    assert all("A" in model.bias_columns for model in arm.working_models)
    assert arm.std_error > 0.9 * bound


FITS = [
    {"form": form, "population": population}
    for form in ("cv", "pooled")
    for population in ("pooled", "trial")
]


@pytest.mark.parametrize("controls_only", [False, True])
@pytest.mark.parametrize("options", FITS)
def test_a_fit_solves_its_scores_and_reports_its_standard_error(
    options: Any, controls_only: bool
) -> None:
    data = trial_draw(1, 600, controls_only=controls_only)
    fit = _fit(data, n_folds=4, **options)
    assert fit.converged
    assert fit.diagnostics.controls_only is controls_only
    assert fit.scores.effect_model < 1e-10
    assert fit.scores.bias_model < 1e-10
    ic = fit.influence_curve
    assert fit.scores.influence <= np.std(ic) / (np.sqrt(600) * np.log(600))
    estimates = [model.estimate for model in fit.working_models]
    assert fit.estimate == pytest.approx(np.mean(estimates), rel=1e-12)
    assert np.all(np.isfinite(ic))
    if options["form"] == "pooled":
        assert len(fit.working_models) == 1
        assert fit.std_error == pytest.approx(np.sqrt(np.mean(ic**2) / 600), rel=1e-12)
    else:
        assert len(fit.working_models) == 4
        variances = [np.var(ic[fit.folds.assignment == v]) for v in range(4)]
        assert fit.std_error == pytest.approx(np.sqrt(np.mean(variances) / 600), rel=1e-12)
    for model in fit.working_models:
        assert model.effect.selected[0] == 0
        assert len(model.effect_columns) == model.beta_a.size
        assert len(model.bias_columns) == model.beta_s.size


def test_a_fit_with_missing_outcomes_runs_both_forms() -> None:
    data = trial_draw(2, 600)
    rng = np.random.default_rng(9)
    observed = rng.uniform(size=600) < 0.85
    data["outcome"] = np.where(observed, data["outcome"], np.nan)
    for form in ("cv", "pooled"):
        fit = _fit(
            data,
            observed=observed,
            missingness_learner=LogisticRegression(),
            form=form,
            n_folds=3,
        )
        assert np.isfinite(fit.estimate) and fit.std_error > 0
        assert fit.scores.effect_model < 1e-10


@pytest.mark.parametrize("group", ["external", "trial"])
@pytest.mark.parametrize("arm", [0.0, 1.0])
def test_an_arm_with_no_observed_outcome_is_a_data_error(group: str, arm: float) -> None:
    """Q(arm, W) on those rows would be pure extrapolation, so the fit refuses."""
    data = trial_draw(2, 600)
    rows = (data["source"] == (1.0 if group == "trial" else 0.0)) & (data["treatment"] == arm)
    assert rows.sum() > 5
    observed = ~rows
    data["outcome"] = np.where(observed, data["outcome"], np.nan)
    with pytest.raises(DataError, match=f"every {group} row in arm {arm:g} has a missing outcome"):
        _fit(data, observed=observed, missingness_learner=LogisticRegression(), form="pooled")
    # One observed row in that arm is enough to pass the check.
    one = observed.copy()
    one[np.flatnonzero(rows)[0]] = True
    data["outcome"][one & rows] = 0.0
    fit = _fit(data, observed=one, missingness_learner=LogisticRegression(), form="pooled")
    assert np.isfinite(fit.estimate)


def test_the_cross_validated_form_needs_three_folds() -> None:
    with pytest.raises(MethodConfigurationError, match="at least three folds"):
        _fit(trial_draw(1, 200), n_folds=2)
    assert _fit(trial_draw(1, 200), n_folds=2, form="pooled").folds.n_folds == 2


@pytest.mark.parametrize("population", ["pooled", "trial"])
@pytest.mark.parametrize("controls_only", [False, True])
def test_the_pooled_steps_solve_the_target_direction_scores(
    monkeypatch: pytest.MonkeyPatch, population: str, controls_only: bool
) -> None:
    calls: list[tuple[Any, ...]] = []
    original = atmle.pooled_epsilon

    def step(units, beta_a, beta_s, directions, model):  # type: ignore[no-untyped-def]
        e = original(units, beta_a, beta_s, directions, model)
        calls.append((units, beta_a, beta_s, directions, model, e))
        return e

    monkeypatch.setattr(atmle, "pooled_epsilon", step)
    fit = _fit(
        trial_draw(6, 1000, controls_only=controls_only),
        n_folds=5,
        population=population,
    )
    assert [call[4] for call in calls] == ["effect"] + ["bias"] * (fit.passes + 1)
    assert fit.epsilon_effect == calls[0][5]
    assert fit.epsilon_bias == tuple(call[5] for call in calls[1:])
    for units, beta_a, beta_s, directions, model, e in calls:
        start = end = scale = 0.0
        for unit, ba, bs, d in zip(units, beta_a, beta_s, directions, strict=True):
            nu = unit.nuisances
            y = unit.outcome
            if model == "effect":
                h = (unit.treatment - nu.g) * (unit.phi @ d)
                r = y - nu.theta - (unit.treatment - nu.g) * (unit.phi @ ba)
            else:
                multiplier = unit.source - np.where(unit.treatment == 1, nu.source1, nu.source0)
                h = multiplier * (unit.psi @ d)
                r = y - nu.qt - multiplier * (unit.psi @ bs)
            start += float(np.mean(h * r)) / len(units)
            end += float(np.mean(h * (r - e * h))) / len(units)
            scale += float(np.mean(np.abs(h * r))) / len(units)
        assert abs(end) <= 1e-10 * scale
        # Witness: the refit alone leaves the pooled score unsolved.
        assert abs(start) > 1e-6 * scale
    # The targeted fit leaves the pooled target-direction scores solved.
    assert fit.scores.effect_model < 1e-10
    assert fit.scores.bias_model < 1e-10


def test_the_trial_population_does_not_warn_about_a_pooled_covariate_gap() -> None:
    """A3 (the trial covers the external rows) is needed only for the combined-rows effect."""
    data = trial_draw(3, 600, controls_only=True)
    data["covariates"][:, 1] = np.where(data["source"] == 0, 4.0, 0.0) + data["covariates"][:, 1]
    with pytest.warns(PositivityWarning, match="below 0.01"):
        pooled = _fit(data, form="pooled", population="pooled")
    with warnings.catch_warnings():
        warnings.simplefilter("error", PositivityWarning)
        trial = _fit(data, form="pooled", population="trial")
    # The reading itself stays on the record, as information.
    assert trial.diagnostics.external_source_below_floor > 0
    assert (
        trial.diagnostics.external_source_below_floor
        == pooled.diagnostics.external_source_below_floor
    )


def test_few_external_treated_rows_and_low_enrollment_probabilities_warn() -> None:
    data = trial_draw(3, 600, controls_only=True)
    external = np.flatnonzero(data["source"] == 0)
    data["treatment"][external[:3]] = 1.0
    data["covariates"][:, 1] = np.where(data["source"] == 0, 4.0, 0.0) + data["covariates"][:, 1]
    with pytest.warns(PositivityWarning) as caught:
        fit = _fit(data, form="pooled")
    messages = [str(item.message) for item in caught]
    assert any("only 3 rows" in message for message in messages)
    assert any("below 0.01" in message for message in messages)
    assert fit.diagnostics.n_external_treated == 3
    assert not fit.diagnostics.controls_only
    assert fit.diagnostics.external_source_below_floor > 0
