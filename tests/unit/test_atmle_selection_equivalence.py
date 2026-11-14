"""The adaptive TMLE working-model selection computes what it computed before X106 E1.

``select_working_model`` used to slice each inner fold's held rows, copy the penalized columns
and compute a Karush-Kuhn-Tucker residual once per penalty. It now does each once. The
arithmetic of every float that feeds a comparison (the risk minimum, the KKT acceptance) is
unchanged, so every comparison here is exact equality, never a tolerance, against the
replaced algorithm written out from the public lasso helpers. The memory test is a bound.
"""

from __future__ import annotations

import dataclasses
import tracemalloc
import weakref
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
import scipy.linalg
from sklearn.linear_model import LinearRegression, LogisticRegression

import cleverly.estimators.atmle as atmle
from cleverly import (
    AdaptiveTMLEMethod,
    CapabilityError,
    CausalStudy,
    CrossFitting,
    ModelSpec,
    TrialATE,
    TrialWithExternalData,
)
from cleverly.estimators.atmle import (
    Selection,
    lasso_solutions,
    penalty_grid,
    select_working_model,
)


def _reference_select(
    basis: Any, multiplier: Any, response: Any, n_unpenalized: int, inner: Any
) -> Selection:
    """The selection as it was computed before X106 E1, from the public lasso helpers.

    Each fold's held rows are sliced afresh for every penalty, and every solution carries its
    KKT residual, as the replaced code did; the comparison is exact.
    """
    grid = penalty_grid(basis, multiplier, response, n_unpenalized)
    total = float(np.sum(np.asarray(multiplier, dtype=float) ** 2))
    assignment = np.asarray(inner, dtype=np.int64)
    loss = np.zeros(grid.size)
    if grid.size > 1:
        for fold in np.unique(assignment):
            held = assignment == fold
            solutions = lasso_solutions(
                basis[~held], multiplier[~held], response[~held], n_unpenalized, grid
            )
            for k, solution in enumerate(solutions):
                fitted = multiplier[held] * (basis[held] @ solution.coef)
                loss[k] += float(np.sum((response[held] - fitted) ** 2))
    risk = loss / total
    index = int(np.flatnonzero(risk == risk.min())[0])
    solution = lasso_solutions(basis, multiplier, response, n_unpenalized, grid[: index + 1])[-1]
    tolerance = atmle.KKT_TOLERANCE * (grid[0] if grid[0] > 0.0 else 1.0)
    if not solution.kkt <= tolerance:
        raise CapabilityError(
            f"The working-model lasso did not reach its optimality tolerance at penalty "
            f"{solution.penalty:.3g}: the Karush-Kuhn-Tucker residual is {solution.kkt:.3g}, "
            f"above {tolerance:.3g}. The declared dictionary is too collinear or too "
            "large for these rows.",
            reason=atmle.RefusalReason.INVALID_DECLARATION,
            alternative=(
                "Pass a smaller or less collinear basis=, for example a transformer with fewer "
                "knots and no pairwise products."
            ),
        )
    selected = np.union1d(
        np.arange(n_unpenalized, dtype=np.int64),
        np.flatnonzero(solution.coef[n_unpenalized:] != 0.0).astype(np.int64) + n_unpenalized,
    )
    return Selection(solution.penalty, selected, solution.coef, grid, risk, solution.kkt)


# --- fixtures ---------------------------------------------------------------------------------

Problem = tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]


def _problem(
    kind: str, k: int, *, folds: int = 5, n: int = 400, columns: int = 12, seed: int = 0
) -> Problem:
    rng = np.random.default_rng([seed, k, folds, sum(map(ord, kind))])
    w = rng.normal(size=(n, columns - 2))
    a = rng.binomial(1, 0.5, n).astype(float)
    basis = np.column_stack([np.ones(n), a, w])
    multiplier = a - 0.5 + 0.1 * rng.normal(size=n)
    signal = 0.8 * w[:, 0] - 0.5 * w[:, 1] + 0.3 * w[:, 2]
    response = multiplier * signal + rng.normal(size=n)
    if kind == "collinear":
        basis[:, 3] = basis[:, 2] + 1e-8 * rng.normal(size=n)
    elif kind == "rare":
        event = rng.random(n) < 0.02
        response = np.where(event, 1.0, 0.0) - 0.02 + 0.01 * multiplier * signal
    elif kind == "constant-column":
        basis[:, 4] = 3.0
    elif kind == "zero-multiplier":
        multiplier[rng.random(n) < 0.3] = 0.0
        response = multiplier * signal + rng.normal(size=n)
    elif kind == "no-usable-column":
        # Every penalized column is constant, so lambda_max is 0 and the grid is [0].
        basis[:, k:] = 3.0
    elif kind == "flat-risk":
        # Pure noise: the held-out risk is flat near lambda_max, so the minimum can tie.
        response = rng.normal(size=n)
        basis[:, 2:] = np.round(basis[:, 2:])
    inner = rng.integers(0, folds, n)
    inner[:folds] = np.arange(folds)
    if kind == "fold-indicator":
        # With two folds, a column that marks fold 1 varies on every row but is constant on
        # each fold's training rows. The fold paths then hold no penalized column, every
        # penalty scores the same, and the risk minimum ties at all 100 penalties.
        indicator = (inner == 1).astype(float)
        basis = np.column_stack([np.ones(n), indicator])
        response = multiplier * (0.5 * indicator) + rng.normal(size=n)
    return basis, multiplier, response, k, inner


KINDS = ("ordinary", "collinear", "rare", "constant-column", "zero-multiplier", "flat-risk")
CASES = [(kind, k, 5) for kind in KINDS for k in (1, 2)] + [
    ("ordinary", 1, 2),
    ("ordinary", 2, 10),
    ("no-usable-column", 1, 5),
    ("fold-indicator", 1, 2),
]


def _assert_same(left: Selection, right: Selection) -> None:
    assert left.penalty == right.penalty
    for name in ("selected", "coef", "grid", "risk"):
        a, b = getattr(left, name), getattr(right, name)
        assert a.dtype == b.dtype and a.shape == b.shape, name
        assert np.array_equal(a, b, equal_nan=True), name
    assert left.kkt == right.kkt


@pytest.mark.parametrize(("kind", "k", "folds"), CASES)
def test_the_selection_equals_the_reference_bit_for_bit(kind: str, k: int, folds: int) -> None:
    problem = _problem(kind, k, folds=folds)
    _assert_same(select_working_model(*problem), _reference_select(*problem))


def test_the_witness_fixtures_choose_an_interior_penalty_on_a_varying_risk() -> None:
    # A defect that scores a fold on the wrong rows only shows where the risk varies and the
    # minimum is interior; at least one fixture must have both.
    interior = 0
    for kind, k, folds in CASES:
        selection = _reference_select(*_problem(kind, k, folds=folds))
        index = int(np.flatnonzero(selection.grid == selection.penalty)[0])
        if np.ptp(selection.risk) > 0.0 and 0 < index < selection.grid.size - 1:
            interior += 1
    assert interior >= 5


def test_one_witness_fixture_ties_the_risk_minimum() -> None:
    selection = _reference_select(*_problem("fold-indicator", 1, folds=2))
    assert selection.grid.size == atmle._N_LAMBDA
    assert np.count_nonzero(selection.risk == selection.risk.min()) > 1


def _mutant_risk(problem: Problem, *, slices_once: bool) -> np.ndarray:
    """The cross-validated risk, with the held slices of fold 0 reused for every fold."""
    basis, multiplier, response, k, inner = problem
    full = atmle._RLoss(basis, multiplier, response, k)
    grid = full.grid()
    loss = np.zeros(grid.size)
    first = inner == 0
    for fold in np.unique(inner):
        held = inner == fold
        scored = first if slices_once else held
        sub = atmle._RLoss(basis[~held], multiplier[~held], response[~held], k)
        for j, coef in enumerate(sub.path(grid)):
            fitted = multiplier[scored] * (basis[scored] @ coef)
            loss[j] += float(np.sum((response[scored] - fitted) ** 2))
    return loss / full.total


def test_the_equivalence_sees_the_defects_hoisting_could_introduce() -> None:
    # Deliberate mutations near the changed code, each of which the exact comparison must
    # detect on an ordinary fixture: held slices taken once before the fold loop, the final
    # path stopped one penalty short, and a path that drops its warm start.
    problem = _problem("ordinary", 1)
    reference = _reference_select(*problem)
    assert np.array_equal(_mutant_risk(problem, slices_once=False), reference.risk)
    assert not np.array_equal(_mutant_risk(problem, slices_once=True), reference.risk)

    basis, multiplier, response, k, _ = problem
    full = atmle._RLoss(basis, multiplier, response, k)
    index = int(np.flatnonzero(reference.grid == reference.penalty)[0])
    assert index > 0
    assert np.array_equal(full.path(reference.grid[: index + 1])[-1], reference.coef)
    assert not np.array_equal(full.path(reference.grid[:index])[-1], reference.coef)

    warm = full.path(reference.grid)
    cold = [full.path(reference.grid[j : j + 1])[-1] for j in range(reference.grid.size)]
    assert any(not np.array_equal(a, b) for a, b in zip(warm, cold, strict=True))


def test_both_refuse_the_same_way_below_the_kkt_tolerance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(atmle, "KKT_TOLERANCE", 0.0)
    problem = _problem("ordinary", 1)
    with pytest.raises(CapabilityError) as shipped:
        select_working_model(*problem)
    with pytest.raises(CapabilityError) as reference:
        _reference_select(*problem)
    assert str(shipped.value) == str(reference.value)
    assert shipped.value.refusal == reference.value.refusal


def test_consecutive_selections_share_no_state() -> None:
    first = _problem("ordinary", 1, seed=1)
    second = _problem("collinear", 2, folds=3, seed=2, n=300, columns=9)
    third = (first[0], first[1], first[2] + 1.0, 1, np.roll(first[4], 7))
    for problem in (first, second, third, first):
        _assert_same(select_working_model(*problem), _reference_select(*problem))


def _reference_path(
    basis: np.ndarray, multiplier: np.ndarray, response: np.ndarray, k: int, grid: np.ndarray
) -> list[np.ndarray]:
    """The warm-started coefficient path as the replaced ``_RLoss.solve`` computed it."""
    problem = atmle._RLoss(basis, multiplier, response, k)
    gram = problem.z.T @ problem.z / problem.total
    linear = problem.z.T @ problem.y / problem.total
    standardized = np.zeros(problem.usable.size)
    out = []
    for lam in grid:
        if problem.usable.size and lam > 0.0:
            standardized = atmle._lasso(gram, linear, float(lam), standardized)
        elif problem.usable.size:
            standardized = np.linalg.lstsq(problem.z, problem.y, rcond=None)[0]
        coef = np.zeros(basis.shape[1])
        coef[problem.usable] = standardized / problem.scale
        if k:
            rest = problem.response - problem.x[:, problem.usable] @ coef[problem.usable]
            coef[:k] = scipy.linalg.solve_triangular(problem.r, problem.q.T @ rest)
        out.append(coef)
    return out


@pytest.mark.parametrize(("kind", "k"), [("ordinary", 1), ("collinear", 2), ("rare", 1)])
def test_the_coefficient_path_equals_the_replaced_loop(kind: str, k: int) -> None:
    # The oracle above shares _RLoss.path with the shipped code; this pins the path itself
    # against the replaced loop, with its warm start and its per-call column copy.
    basis, multiplier, response, k, _ = _problem(kind, k)
    grid = penalty_grid(basis, multiplier, response, k)
    expected = _reference_path(basis, multiplier, response, k, grid)
    shipped = atmle._RLoss(basis, multiplier, response, k).path(grid)
    solutions = lasso_solutions(basis, multiplier, response, k, grid)
    for left, solution, right in zip(shipped, solutions, expected, strict=True):
        assert np.array_equal(left, right)
        assert np.array_equal(solution.coef, right)


# --- deterministic work guards ----------------------------------------------------------------


class _CountingArray(np.ndarray):
    """An array that counts the boolean-mask row slices taken of it."""

    masks: list[int]

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, np.ndarray) and key.dtype == bool:
            self.masks.append(1)
        return np.asarray(super().__getitem__(key))


def test_a_selection_computes_one_kkt_residual_and_slices_each_fold_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    basis, multiplier, response, k, inner = _problem("ordinary", 1)
    calls: list[int] = []
    original = atmle._RLoss.kkt

    def counted(self: Any, coef: np.ndarray, penalty: float) -> float:
        calls.append(1)
        return original(self, coef, penalty)

    monkeypatch.setattr(atmle._RLoss, "kkt", counted)
    counting = basis.view(_CountingArray)
    counting.masks = []
    selection = select_working_model(counting, multiplier, response, k, inner)
    folds = np.unique(inner).size
    assert selection.grid.size == atmle._N_LAMBDA
    # The replaced code computed one residual per solution and sliced each fold's held rows
    # once per penalty: about 100 * (folds + 1) residuals and 100 * folds slices.
    assert len(calls) == 1
    assert len(counting.masks) == 2 * folds


def test_the_penalized_columns_are_copied_once_per_problem() -> None:
    basis, multiplier, response, k, _ = _problem("ordinary", 1)
    problem = atmle._RLoss(basis, multiplier, response, k)
    # Built on the first coefficients() call only, so the full-row problem holds no copy
    # while the fold problems are solved.
    assert "xu" not in vars(problem)
    fresh = problem.x[:, problem.usable]
    # The stored copy has the values and the memory layout of the per-call copy it replaced.
    assert np.array_equal(problem.xu, fresh)
    assert problem.xu.strides == fresh.strides
    # coefficients() reads only the stored copy: poisoning x leaves the path unchanged.
    expected = problem.path(problem.grid()[:10])
    problem.x = np.full_like(problem.x, np.nan)
    for left, right in zip(problem.path(problem.grid()[:10]), expected, strict=True):
        assert np.array_equal(left, right)


def test_a_selection_peaks_below_the_replaced_code() -> None:
    # The traced peak is a fixed multiple of one copy of the dictionary, 8 * n * J bytes. The
    # replaced code peaked at 8.6 copies on this fixture (cleverly-tmle ace81996); the lazy
    # column copy and the release of each fold's state before the next keep it near 6.2.
    # Keeping a fold's problem into the next fold reads 9.6 and fails.
    basis, multiplier, response, k, inner = _problem("ordinary", 1, n=4000, columns=40)
    tracing = tracemalloc.is_tracing()
    if not tracing:
        tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        start = tracemalloc.get_traced_memory()[0]
        select_working_model(basis, multiplier, response, k, inner)
        peak = tracemalloc.get_traced_memory()[1] - start
    finally:
        if not tracing:
            tracemalloc.stop()
    copies = peak / basis.nbytes
    assert copies <= 7.9, f"the selection peaked at {copies:.2f} copies of the dictionary"


def test_the_full_row_problem_holds_no_column_copy_while_the_folds_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A copy built for the full-row problem before the fold loop would stay alive beside
    # every fold's own copy.
    problems: list[weakref.ref[Any]] = []
    holders: list[int] = []
    init, path = atmle._RLoss.__init__, atmle._RLoss.path

    def tracked_init(self: Any, *args: Any) -> None:
        init(self, *args)
        problems.append(weakref.ref(self))

    def tracked_path(self: Any, grid: np.ndarray) -> list[np.ndarray]:
        if self is not problems[0]():
            holders.append(
                sum(
                    "xu" in vars(p)
                    for ref in problems[1:]
                    if (p := ref()) is not None and p is not self
                )
            )
            holders.append(int("xu" in vars(problems[0]())))
        return path(self, grid)

    monkeypatch.setattr(atmle._RLoss, "__init__", tracked_init)
    monkeypatch.setattr(atmle._RLoss, "path", tracked_path)
    select_working_model(*_problem("ordinary", 1))
    assert len(holders) == 2 * 5
    assert not any(holders)


# --- the public fit ---------------------------------------------------------------------------


def _frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    s = rng.binomial(1, 0.5, n)
    w = rng.normal(size=(n, 2))
    a = np.where(s == 1, rng.binomial(1, 0.5, n), rng.binomial(1, 0.3, n))
    y = 1.0 + a + w[:, 0] + 0.5 * (1 - s) * (1 + a) * (1 + w[:, 1]) + rng.normal(size=n)
    return pd.DataFrame({"Y": y, "A": a, "W1": w[:, 0], "W2": w[:, 1], "S": s})


def _public_fit(frame: pd.DataFrame, form: str) -> Any:
    method = AdaptiveTMLEMethod(
        form=form,  # type: ignore[arg-type]
        models=ModelSpec(
            outcome_learner=LinearRegression(), treatment_learner=LogisticRegression()
        ),
        cross_fitting=CrossFitting(n_folds=3, learner_folds=3),
    ).with_overrides(random_state=11)
    design = TrialWithExternalData(outcome="Y", treatment="A", adjustment=["W1", "W2"], source="S")
    return CausalStudy(frame, design=design).identify(TrialATE(population="trial")).estimate(method)


def _same(left: Any, right: Any) -> bool:
    """Exact equality through dataclasses, sequences, mappings and arrays."""
    if dataclasses.is_dataclass(left) and not isinstance(left, type):
        return type(left) is type(right) and all(
            _same(getattr(left, f.name), getattr(right, f.name)) for f in dataclasses.fields(left)
        )
    if isinstance(left, np.ndarray):
        return (
            isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and np.array_equal(left, right, equal_nan=left.dtype.kind in "fc")
        )
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _same(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, float) and np.isnan(left):
        return isinstance(right, float) and np.isnan(right)
    if hasattr(left, "__dict__") and not isinstance(left, type):
        # A fitted transformer or learner: equal when its fitted state is.
        return type(left) is type(right) and _same(vars(left), vars(right))
    return bool(left == right)


@pytest.mark.parametrize("form", ["cv", "pooled"])
def test_a_public_fit_is_unchanged_when_the_reference_selects(
    form: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _frame(600, seed=3)
    shipped = _public_fit(frame, form).fit
    select: Callable[..., Selection] = _reference_select
    monkeypatch.setattr(atmle, "select_working_model", select)
    reference = _public_fit(frame, form).fit
    assert _same(shipped, reference)


def test_the_public_comparison_sees_a_one_column_change(monkeypatch: pytest.MonkeyPatch) -> None:
    # Deliberate mutation: the fit must change when a selection loses its last column (or
    # gains the first penalized one, when it holds only the unpenalized columns), so the
    # exact comparison above has something to detect.
    frame = _frame(600, seed=3)
    shipped = _public_fit(frame, "cv").fit

    def changed(*args: Any) -> Selection:
        selection = _reference_select(*args)
        n_unpenalized = args[3]
        if selection.selected.size > n_unpenalized:
            selected = selection.selected[:-1]
        else:
            selected = np.append(selection.selected, n_unpenalized)
        return dataclasses.replace(selection, selected=selected)

    monkeypatch.setattr(atmle, "select_working_model", changed)
    assert not _same(shipped, _public_fit(frame, "cv").fit)
