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


def test_table_is_held_once() -> None:
    rng = np.random.default_rng(7)
    covariates = rng.standard_normal((5_000, 4))
    bins = rng.integers(0, 60, 5_000)
    tracemalloc.start()
    try:
        design, _, _ = density_module._long_expansion(covariates, bins, 59)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    # Stacking whole blocks peaked at about two tables; the in-place build at about 1.1.
    assert peak < 1.3 * design.nbytes
