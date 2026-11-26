"""The pooled hazard table is built once, in place, with today's values (X106 D1a)."""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from cleverly.learners import density as density_module


def _stacked(
    covariates: np.ndarray, bins: np.ndarray, n_hazards: int, index_only: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The table as whole blocks stacked side by side: the reference construction."""
    counts = np.minimum(bins, n_hazards - 1) + 1
    rows = np.repeat(np.arange(bins.size, dtype=np.int64), counts)
    offsets = np.repeat(np.cumsum(counts) - counts, counts)
    index = np.arange(rows.size, dtype=np.int64) - offsets
    block = density_module._bin_block(index, n_hazards, index_only)
    design = np.column_stack([covariates[rows], block])
    return design, (index == bins[rows]).astype(float), rows


@pytest.mark.parametrize("n_bins", [3, 4, 30])
@pytest.mark.parametrize("index_only", [False, True])
@pytest.mark.parametrize("n", [1, 9, 400])
def test_table_matches_the_stacked_blocks_bitwise(n_bins: int, index_only: bool, n: int) -> None:
    rng = np.random.default_rng(n * 100 + n_bins)
    covariates = rng.standard_normal((n, 3))
    # Leave bins empty on purpose: the indicator of an empty bin is a column of zeros.
    bins = rng.integers(0, max(n_bins - 1, 1), n)
    expected = _stacked(covariates, bins, n_bins - 1, index_only)
    found = density_module._long_expansion(covariates, bins, n_bins - 1, index_only)
    for want, got in zip(expected, found, strict=True):
        assert got.dtype == want.dtype
        assert got.shape == want.shape
        np.testing.assert_array_equal(got, want)
    assert found[0].flags.c_contiguous


def test_indicator_block_is_not_empty() -> None:
    # A nonzero witness: the indicators carry ones, so a build that left them zero would fail
    # the bitwise test above on this fixture.
    covariates = np.zeros((6, 1))
    bins = np.array([0, 1, 2, 3, 3, 2])
    design, _, _ = density_module._long_expansion(covariates, bins, 3)
    assert design[:, 2:].sum() > 0


@pytest.mark.parametrize(("n_bins", "ceiling"), [(60, 1.3), (10, 1.6)])
def test_table_is_held_once(n_bins: int, ceiling: float) -> None:
    rng = np.random.default_rng(7)
    covariates = rng.standard_normal((5_000, 4))
    bins = rng.integers(0, n_bins, 5_000)
    tracemalloc.start()
    try:
        design, _, _ = density_module._long_expansion(covariates, bins, n_bins - 1)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    # Stacking whole blocks peaked at about two tables. The in-place build holds the table
    # plus the record index vectors, which weigh more when the table is narrow (few bins).
    assert peak < ceiling * design.nbytes


# --- the sparse design (X106 D1b) ------------------------------------------------------------


@pytest.mark.parametrize("n_bins", [3, 4, 30])
@pytest.mark.parametrize("n", [1, 9, 400])
def test_sparse_table_holds_the_dense_tables_nonzeros_in_order(n_bins: int, n: int) -> None:
    rng = np.random.default_rng(n * 10 + n_bins)
    covariates = rng.standard_normal((n, 3))
    # Exact zeros, which liblinear skips in a dense design, and a negative zero.
    covariates[rng.uniform(size=covariates.shape) < 0.3] = 0.0
    covariates[0, 0] = -0.0
    bins = rng.integers(0, max(n_bins - 1, 1), n)
    dense, target, rows = density_module._long_expansion(covariates, bins, n_bins - 1)
    found, sparse_target, sparse_rows = density_module._long_expansion(
        covariates, bins, n_bins - 1, sparse=True
    )
    np.testing.assert_array_equal(sparse_target, target)
    np.testing.assert_array_equal(sparse_rows, rows)
    assert found.shape == dense.shape
    assert found.has_sorted_indices
    expected = [(i, j, dense[i, j]) for i, j in zip(*np.nonzero(dense), strict=True)]
    coo = found.tocoo()
    got = list(zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist(), strict=True))
    assert got == [(int(i), int(j), float(v)) for i, j, v in expected]


def _density(learner: object, sparse: bool, monkeypatch: pytest.MonkeyPatch) -> np.ndarray:
    from cleverly.learners.crossfit import Folds

    if not sparse:
        monkeypatch.setattr(density_module, "_sparse_route", lambda *args: False)
    rng = np.random.default_rng(4)
    n = 900
    w = rng.standard_normal((n, 3))
    w[:, 2] = (w[:, 2] > 0.5).astype(float)
    a = 1.5 + 0.6 * w[:, 0] + rng.standard_normal(n)
    folds = Folds(np.arange(n) % 3, 3)
    fitted, _ = density_module.fit_conditional_density(
        learner, w, a, rng.uniform(0.5, 2.0, n), folds, n_bins=40
    )
    monkeypatch.undo()
    return fitted.density_at(a)


@pytest.mark.parametrize("options", [{}, {"C": 1e10}, {"penalty": "l1"}, {"dual": True}], ids=str)
def test_liblinear_fits_the_same_density_on_either_design(
    monkeypatch: pytest.MonkeyPatch, options: dict[str, object]
) -> None:
    from sklearn.linear_model import LogisticRegression

    learner = LogisticRegression(solver="liblinear", random_state=0, **options)
    assert density_module._sparse_route(learner, 10, 3, 59)
    calls: list[bool] = []
    route = density_module._sparse_route
    monkeypatch.setattr(
        density_module, "_sparse_route", lambda *a: calls.append(route(*a)) or calls[-1]
    )
    sparse = _density(learner, True, monkeypatch)
    assert calls and all(calls)  # the fixture takes the sparse route
    dense = _density(learner, False, monkeypatch)
    assert dense.tobytes() == sparse.tobytes()


def test_only_liblinear_logistic_regression_takes_the_sparse_design() -> None:
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

    class Subclass(LogisticRegression):
        pass

    assert not density_module._sparse_route(LogisticRegression(), 10, 3, 59)
    assert not density_module._sparse_route(LogisticRegression(solver="newton-cholesky"), 10, 3, 59)
    assert not density_module._sparse_route(Subclass(solver="liblinear"), 10, 3, 59)
    assert not density_module._sparse_route(LogisticRegressionCV(solver="liblinear"), 10, 3, 59)
    liblinear = LogisticRegression(solver="liblinear")
    assert density_module._sparse_route(liblinear, 10, 3, 59)
    assert not density_module._sparse_route(liblinear, 2**31 // 4, 3, 59)
    # Few bins: the dense table (3 + 11 columns) is smaller than the sparse build's count.
    assert not density_module._sparse_route(liblinear, 10, 3, 11)
    assert density_module._record_columns(3, 11, False, True) >= 3 + 11


def test_the_guard_counts_the_sparse_bytes() -> None:
    # Nonzero witness: a sparse record is counted at its sparse build, not the dense width.
    assert density_module._record_columns(4, 59, False, True) == 21
    assert density_module._record_columns(4, 59, False, False) == 4 + 59


def _sparse_build_peak(n: int, n_bins: int, p: int) -> tuple[int, int]:
    """(tracemalloc peak of the CSR build, records) on a fixture with a zero in each column."""
    rng = np.random.default_rng(n + n_bins + p)
    covariates = rng.standard_normal((n, p))
    covariates[rng.random((n, p)) < 0.2] = 0.0
    bins = rng.integers(0, n_bins, n)
    tracemalloc.start()
    try:
        base = tracemalloc.get_traced_memory()[0]
        _, _, rows = density_module._long_expansion(covariates, bins, n_bins - 1, False, True)
        peak = tracemalloc.get_traced_memory()[1] - base
    finally:
        tracemalloc.stop()
    return peak, rows.size


@pytest.mark.parametrize(("n", "n_bins", "p"), [(6_000, 60, 4), (6_000, 12, 1), (3_000, 40, 20)])
def test_the_sparse_guard_covers_the_build_it_admits(n: int, n_bins: int, p: int) -> None:
    """W5: the guard's sparse bytes are at least the CSR build's peak and at most 1.5x it.

    The mutation control is the count of the CSR arrays alone, which the build's record
    vectors pass on every fixture.
    """
    peak, records = _sparse_build_peak(n, n_bins, p)
    counted = records * density_module._record_columns(p, n_bins - 1, False, True) * 8
    assert peak <= counted <= 1.5 * peak
    csr_only = records * -(-((p + 2) * 12 + 4) // 8) * 8
    assert csr_only < peak


def test_the_sparse_guard_refuses_one_byte_short(monkeypatch: pytest.MonkeyPatch) -> None:
    """A liblinear density fit completes at exactly the sparse count and refuses one byte below."""
    from sklearn.linear_model import LogisticRegression

    from cleverly.exceptions import MethodConfigurationError
    from cleverly.learners.crossfit import Folds

    learner = LogisticRegression(solver="liblinear", random_state=0)
    rng = np.random.default_rng(5)
    n = 600
    w = rng.standard_normal((n, 3))
    a = rng.standard_normal(n)
    folds = Folds(np.zeros(n, dtype=np.int64), 1)
    grid = density_module._grid(a, None, 40, None, None)
    records, columns, _ = density_module._table_size(learner, 3, grid, None)
    assert columns < 3 + grid.n_hazards  # the sparse count, below the dense width
    needed = records * columns * 8

    def fit(free: int) -> None:
        monkeypatch.setattr(density_module, "available_memory", lambda: free)
        density_module.fit_conditional_density(learner, w, a, np.ones(n), folds, n_bins=40)

    fit(needed)
    with pytest.raises(MethodConfigurationError, match="needs"):
        fit(needed - 1)
