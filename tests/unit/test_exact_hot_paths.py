"""Faster forms of four internal computations return exactly what the plain forms return.

The tests of the first three keep the plain form as a reference and compare bitwise; the
duplicate-covariate tests compare with the real digest and under forced digest collisions. The
inputs reach the cases where a shortcut could differ: tie blocks and NaNs for the weighted AUC, ties and
NaN event times for the Breslow counts, pattern counts on either side of the term count for
the constant-cell check, and digest collisions and signed zeros for duplicate covariates.
"""

from __future__ import annotations

import importlib
import tracemalloc
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from cleverly.data import validate
from cleverly.data.validate import check_covariates
from cleverly.estimators.tmle import _coefficients_on_constant_cells
from cleverly.survival.cox import _risk_sets
from cleverly.validation.nuisance import _weighted_auc

# The package exports a function named ``tmle``, which shadows the module attribute.
tmle = importlib.import_module("cleverly.estimators.tmle")

# ----------------------------------------------------------------------------------------
# Weighted AUC


def _auc_reference(predicted: np.ndarray, actual: np.ndarray, weights: np.ndarray) -> float:
    positive = actual == 1.0
    if not positive.any() or positive.all():
        return float("nan")
    order = np.argsort(predicted, kind="stable")
    p_sorted = predicted[order]
    y_sorted = positive[order]
    w_sorted = weights[order]
    ranks = np.empty(p_sorted.shape[0], dtype=float)
    cumulative = np.cumsum(w_sorted)
    start = 0
    while start < p_sorted.shape[0]:
        stop = start
        while stop + 1 < p_sorted.shape[0] and p_sorted[stop + 1] == p_sorted[start]:
            stop += 1
        below = cumulative[start - 1] if start > 0 else 0.0
        block = cumulative[stop] - below
        ranks[start : stop + 1] = below + 0.5 * block
        start = stop + 1
    weight_positive = w_sorted[y_sorted].sum()
    weight_negative = w_sorted[~y_sorted].sum()
    if weight_positive <= 0 or weight_negative <= 0:
        return float("nan")
    concordant = float(np.sum(w_sorted[y_sorted] * ranks[y_sorted]))
    concordant -= 0.5 * weight_positive**2
    return float(concordant / (weight_positive * weight_negative))


def _auc_inputs() -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(20261114)
    n = 2_000
    actual = rng.binomial(1, 0.3, n).astype(float)
    weights = rng.uniform(0.2, 3.0, n)
    zero = weights.copy()
    zero[rng.random(n) < 0.1] = 0.0
    unique = rng.random(n)
    tied = np.round(rng.random(n), 1)
    with_nan = unique.copy()
    with_nan[[3, 17]] = np.nan
    signed = np.where(rng.random(n) < 0.5, 0.0, -0.0)
    return [
        ("unique", unique, actual, weights),
        ("ten levels", tied, actual, weights),
        ("all tied", np.full(n, 0.4), actual, weights),
        ("zero weights", tied, actual, zero),
        ("two NaNs", with_nan, actual, weights),
        ("signed zeros", signed, actual, weights),
        ("one row per block", np.arange(n, dtype=float), actual, weights),
        ("no positives", unique, np.zeros(n), weights),
        ("no negative weight", unique, actual, np.where(actual == 1.0, weights, 0.0)),
    ]


@pytest.mark.parametrize("case", _auc_inputs(), ids=lambda case: case[0])
def test_weighted_auc_is_bitwise_the_tie_block_loop(case) -> None:
    _, predicted, actual, weights = case
    expected = _auc_reference(predicted, actual, weights)
    found = _weighted_auc(predicted, actual, weights)
    assert np.isnan(expected) == np.isnan(found)
    if not np.isnan(expected):
        assert found == expected


def test_weighted_auc_ties_take_half_credit() -> None:
    # One positive tied with one negative of equal weight: AUC 0.5 exactly.
    assert _weighted_auc(np.array([0.3, 0.3]), np.array([1.0, 0.0]), np.ones(2)) == 0.5


# ----------------------------------------------------------------------------------------
# Breslow event counts


def _deaths_reference(time: np.ndarray, event: np.ndarray) -> np.ndarray:
    events = event > 0
    times = np.unique(time[events])
    return np.array([np.count_nonzero(events & (time == t)) for t in times], dtype=float)


@pytest.mark.parametrize(
    "case",
    [
        "continuous",
        "tied",
        "nan event times",
        "signed zero",
        "one event",
        "all events",
        "no events",
    ],
)
def test_breslow_counts_match_the_row_scan(case: str) -> None:
    rng = np.random.default_rng(7)
    n = 500
    time = rng.exponential(2.0, n)
    event = rng.binomial(1, 0.6, n).astype(float)
    if case == "tied":
        time = np.round(time, 1)
    elif case == "nan event times":
        time[np.flatnonzero(event)[:2]] = np.nan
    elif case == "signed zero":
        time[np.flatnonzero(event)[:3]] = [0.0, -0.0, 0.0]
    elif case == "one event":
        event[:] = 0.0
        event[5] = 1.0
    elif case == "all events":
        event[:] = 1.0
    elif case == "no events":
        event[:] = 0.0
    sets = _risk_sets(time, event)
    expected = _deaths_reference(time, event)
    assert sets.deaths.dtype == expected.dtype
    np.testing.assert_array_equal(sets.deaths, expected, strict=True)
    assert np.array_equal(sets.times, np.unique(time[event > 0]), equal_nan=True)


# ----------------------------------------------------------------------------------------
# Constant-cell check of a projection


def _cells_reference(outcome, treatment, qualifying, positive, *, msm, contrast):
    """The check without the early exit, as it stood before."""
    if contrast is not None:
        design = np.asarray(contrast.design, dtype=float)
        signature = design
    elif msm is not None:
        design = np.asarray(msm.design, dtype=float)
        weights = np.asarray(msm.weights, dtype=float)
        signature = np.concatenate([design.reshape(design.shape[0], -1), weights], axis=1)
    else:
        return frozenset()
    rows = np.flatnonzero(positive)
    if rows.size == 0:
        return frozenset()
    _, first, label = np.unique(signature[rows], axis=0, return_index=True, return_inverse=True)
    label = np.asarray(label).reshape(-1)
    terms = design.shape[-1]

    def constant(pattern, arms):
        members = rows[label == pattern]
        for arm in arms:
            cell = members[qualifying[members] & (treatment[members] == arm)]
            if cell.size and np.unique(outcome[cell]).size == 1:
                return True
        return False

    cells, flags = [], []
    for pattern, row in enumerate(rows[first]):
        if contrast is not None:
            cells.append(design[row])
            flags.append(constant(pattern, contrast.pair))
            continue
        for position, arm in enumerate(msm.arms):
            if weights[row, position] > 0.0:
                cells.append(design[row, position])
                flags.append(constant(pattern, (arm,)))
        if len(cells) > terms:
            return frozenset()
    if len(cells) != terms or not any(flags):
        return frozenset()
    try:
        inverse = np.linalg.inv(np.vstack(cells))
    except np.linalg.LinAlgError:
        return frozenset()
    reads = np.abs(inverse) > 1e-10 * np.abs(inverse).max()
    hit = np.asarray(flags, dtype=bool)
    return frozenset(float(j) for j in range(terms) if bool(np.any(reads[j] & hit)))


def _indicator_design(levels: np.ndarray, count: int) -> np.ndarray:
    """An intercept plus indicators of levels 1 .. count - 1, so ``count`` terms."""
    columns = [np.ones(levels.size)] + [(levels == k).astype(float) for k in range(1, count)]
    return np.column_stack(columns)


def test_projection_check_matches_the_full_loop_on_random_designs() -> None:
    rng = np.random.default_rng(20261114)
    nonempty = 0
    for _ in range(600):
        n = int(rng.integers(6, 40))
        terms = int(rng.integers(1, 5))
        patterns = int(rng.integers(1, terms + 3))
        levels = rng.integers(0, patterns, n)
        if rng.random() < 0.5:
            design = _indicator_design(levels, terms)
        else:
            design = rng.choice([0.0, -0.0, 0.5, 1.0], size=(n, terms))
            design[:, 0] = 1.0
        treatment = rng.integers(0, 2, n).astype(float)
        binary = rng.random() < 0.5
        outcome = rng.integers(0, 2, n).astype(float) if binary else rng.normal(size=n)
        positive = rng.random(n) > 0.15
        qualifying = positive & (rng.random(n) > 0.1)
        contrast = SimpleNamespace(design=design, pair=(1.0, 0.0))
        expected = _cells_reference(
            outcome, treatment, qualifying, positive, msm=None, contrast=contrast
        )
        found = _coefficients_on_constant_cells(
            outcome, treatment, qualifying, positive, msm=None, contrast=contrast
        )
        assert found == expected
        nonempty += bool(found)
    assert nonempty > 20  # the comparison reaches saturated designs with a constant cell


def _saturated_case(patterns: int, terms: int):
    """A projection over ``patterns`` levels with the outcome constant in level 0's arm 1."""
    levels = np.repeat(np.arange(patterns), 8)
    treatment = np.tile(np.array([0.0, 1.0]), levels.size // 2)
    outcome = np.where(levels == 0, 1.0, np.tile([0.0, 1.0, 1.0, 0.0], levels.size // 4))
    outcome = np.where((levels == 0) & (treatment == 0.0), np.arange(levels.size) % 2, outcome)
    design = np.column_stack(
        [np.ones(levels.size)] + [(levels == k).astype(float) for k in range(1, terms)]
    )
    contrast = SimpleNamespace(design=design, pair=(1.0, 0.0))
    ones = np.ones(levels.size, dtype=bool)
    return outcome.astype(float), treatment, ones, ones, contrast


def test_a_saturated_projection_still_reports_its_constant_cell() -> None:
    # Mutation control: P == q must run the loop. An exit at P >= q would return no code here.
    outcome, treatment, qualifying, positive, contrast = _saturated_case(3, 3)
    found = _coefficients_on_constant_cells(
        outcome, treatment, qualifying, positive, msm=None, contrast=contrast
    )
    assert found
    assert found == _cells_reference(
        outcome, treatment, qualifying, positive, msm=None, contrast=contrast
    )


def test_more_patterns_than_terms_skips_the_cell_scan(monkeypatch) -> None:
    calls = 0
    original = tmle._pattern_has_constant_cell

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(tmle, "_pattern_has_constant_cell", counted)
    rng = np.random.default_rng(3)
    n = 10_000
    modifier = rng.uniform(-1, 1, n)
    contrast = SimpleNamespace(design=np.column_stack([np.ones(n), modifier]), pair=(1.0, 0.0))
    ones = np.ones(n, dtype=bool)
    found = _coefficients_on_constant_cells(
        rng.integers(0, 2, n).astype(float),
        rng.integers(0, 2, n).astype(float),
        ones,
        ones,
        msm=None,
        contrast=contrast,
    )
    assert found == frozenset()
    assert calls == 0
    # The saturated case does call it, so a zero count above is not vacuous.
    outcome, treatment, qualifying, positive, saturated = _saturated_case(3, 3)
    _coefficients_on_constant_cells(
        outcome, treatment, qualifying, positive, msm=None, contrast=saturated
    )
    assert calls == 3


def test_working_model_check_is_unchanged() -> None:
    rng = np.random.default_rng(11)
    for _ in range(200):
        n = int(rng.integers(8, 40))
        levels = rng.integers(0, 3, n)
        arms = (0.0, 1.0)
        terms = int(rng.integers(2, 4))
        design = np.zeros((n, 2, terms))
        for k, arm in enumerate(arms):
            design[:, k, 0] = 1.0
            design[:, k, 1] = arm
            if terms == 3:
                design[:, k, 2] = arm * (levels == 1)
        weights = (rng.random((n, 2)) > 0.3).astype(float)
        msm = SimpleNamespace(design=design, weights=weights, arms=arms)
        treatment = rng.integers(0, 2, n).astype(float)
        outcome = rng.integers(0, 2, n).astype(float)
        positive = rng.random(n) > 0.1
        qualifying = positive.copy()
        expected = _cells_reference(
            outcome, treatment, qualifying, positive, msm=msm, contrast=None
        )
        found = _coefficients_on_constant_cells(
            outcome, treatment, qualifying, positive, msm=msm, contrast=None
        )
        assert found == expected


def test_a_saturated_working_model_reports_its_constant_cell() -> None:
    # One pattern with both arms weighted gives two cells for two terms: saturated. Arm 1's
    # outcome is constant, so the coefficient on the arm reads a constant cell.
    n = 20
    design = np.zeros((n, 2, 2))
    design[:, :, 0] = 1.0
    design[:, 1, 1] = 1.0
    msm = SimpleNamespace(design=design, weights=np.ones((n, 2)), arms=(0.0, 1.0))
    treatment = np.tile([0.0, 1.0], n // 2)
    outcome = np.where(treatment == 1.0, 1.0, np.arange(n) % 2).astype(float)
    ones = np.ones(n, dtype=bool)
    found = _coefficients_on_constant_cells(outcome, treatment, ones, ones, msm=msm, contrast=None)
    assert found
    assert found == _cells_reference(outcome, treatment, ones, ones, msm=msm, contrast=None)


# ----------------------------------------------------------------------------------------
# Duplicate covariates


def _checked(matrix: np.ndarray, names: list[str]):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = check_covariates(matrix, names)
    return result, [str(w.message) for w in caught]


def test_duplicates_are_found_through_digest_collisions(monkeypatch) -> None:
    rng = np.random.default_rng(5)
    a, b, c = rng.normal(size=(3, 50))
    zero = np.zeros(50)
    zero[1] = 1.0
    negative = zero.copy()
    zero[0], negative[0] = 0.0, -0.0
    matrix = np.column_stack([a, b, b, np.full(50, 2.0), zero, negative, c, a])
    names = ["a", "b", "b2", "const", "zero", "negzero", "c", "a2"]
    reference = _checked(matrix, names)
    monkeypatch.setattr(validate, "_column_digest", lambda column: 0)
    (kept, retained, dropped), messages = _checked(matrix, names)
    assert dropped == ["const", "b2", "a2"]  # constants first, then duplicates in order
    assert retained == ["a", "b", "zero", "negzero", "c"]  # signed zeros differ as bytes
    assert kept.tobytes() == reference[0][0].tobytes()
    assert (retained, dropped, messages) == (reference[0][1], reference[0][2], reference[1])


def test_a_constant_duplicate_is_dropped_as_a_constant() -> None:
    matrix = np.column_stack([np.arange(5.0), np.ones(5), np.ones(5)])
    (_, retained, dropped), _ = _checked(matrix, ["x", "one", "one2"])
    assert retained == ["x"]
    assert dropped == ["one", "one2"]


def test_stored_covariates_do_not_follow_the_callers_array() -> None:
    from cleverly.data import CausalData

    rng = np.random.default_rng(9)
    covariates = rng.normal(size=(200, 3))
    data = CausalData.from_arrays(
        outcome=rng.binomial(1, 0.5, 200).astype(float),
        treatment=rng.binomial(1, 0.5, 200).astype(float),
        covariates=covariates,
    )
    before = np.array(data.covariates, copy=True)
    covariates[:] = 0.0
    np.testing.assert_array_equal(data.covariates, before)


def test_duplicate_detection_keeps_no_copy_of_every_column() -> None:
    rng = np.random.default_rng(13)
    n, p = 200_000, 20
    matrix = rng.normal(size=(n, p))
    tracemalloc.start()
    try:
        check_covariates(matrix, [f"w{j}" for j in range(p)])
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # The retained output is one 8 * n * p copy. Byte keys of every column would add another
    # 8 * n * p on top; the digests keep at most a column or two of scratch.
    assert peak < 8 * n * p * 1.4
