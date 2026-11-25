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
        learner, w, a, rng.uniform(0.5, 2.0, n), folds, n_bins=12
    )
    monkeypatch.undo()
    return fitted.density_at(a)


@pytest.mark.parametrize("options", [{}, {"C": 1e10}, {"penalty": "l1"}, {"dual": True}], ids=str)
def test_liblinear_fits_the_same_density_on_either_design(
    monkeypatch: pytest.MonkeyPatch, options: dict[str, object]
) -> None:
    from sklearn.linear_model import LogisticRegression

    learner = LogisticRegression(solver="liblinear", random_state=0, **options)
    assert density_module._sparse_route(learner, 10, 3)
    dense = _density(learner, False, monkeypatch)
    sparse = _density(learner, True, monkeypatch)
    assert dense.tobytes() == sparse.tobytes()


def test_only_liblinear_logistic_regression_takes_the_sparse_design() -> None:
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

    class Subclass(LogisticRegression):
        pass

    assert not density_module._sparse_route(LogisticRegression(), 10, 3)
    assert not density_module._sparse_route(LogisticRegression(solver="newton-cholesky"), 10, 3)
    assert not density_module._sparse_route(Subclass(solver="liblinear"), 10, 3)
    assert not density_module._sparse_route(LogisticRegressionCV(solver="liblinear"), 10, 3)
    liblinear = LogisticRegression(solver="liblinear")
    assert not density_module._sparse_route(liblinear, 2**31 // 4, 3)


def test_the_guard_counts_the_sparse_bytes() -> None:
    # Nonzero witness: a sparse record is counted at its CSR size, not the dense width.
    assert density_module._record_columns(4, 59, False, True) == 10
    assert density_module._record_columns(4, 59, False, False) == 4 + 59
