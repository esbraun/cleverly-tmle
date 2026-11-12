"""The univariate kernel smoother of the drift-corrected survival TMLE (check 11)."""

from __future__ import annotations

import numpy as np
import pytest

from cleverly.longitudinal import _kernel
from cleverly.longitudinal._kernel import BINS, MIN_ROWS, smooth


def _sample(m: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, m)
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-np.sin(2.0 * x)))).astype(float)
    return x, y


def test_a_constant_outcome_is_returned_unchanged() -> None:
    x = np.linspace(0.0, 1.0, 50)
    fit = smooth(x, np.full(50, 0.3), np.ones(50, dtype=bool))
    assert fit.degenerate
    np.testing.assert_array_equal(fit.values, 0.3)


def test_the_fit_is_invariant_to_row_order_with_tied_indices() -> None:
    rng = np.random.default_rng(1)
    x = np.round(rng.uniform(0.0, 1.0, 300), 1)
    y = rng.binomial(1, 0.3 + 0.4 * x).astype(float)
    rows = rng.uniform(size=300) < 0.8
    permutation = rng.permutation(300)
    one = smooth(x, y, rows)
    two = smooth(x[permutation], y[permutation], rows[permutation])
    assert one.bandwidth == two.bandwidth
    np.testing.assert_array_equal(one.values[permutation], two.values)


def _brute_loo(x: np.ndarray, y: np.ndarray, bandwidth: float) -> float:
    """The LOO error on the binned representation, by removing each row and rebinning."""
    order = np.lexsort((y, x))
    x, y = x[order], y[order]
    low, high = x[0], x[-1]
    width = (high - low) / (BINS - 1)
    grid = np.linspace(low, high, BINS)
    errors = []
    for i in range(x.size):
        keep = np.arange(x.size) != i
        _, _, _, binned = _kernel._bin(x[keep], y[keep], low, width)
        sums = _kernel._convolve(binned, _kernel._kernel(bandwidth, width))
        ratio = sums[1] / sums[0]
        errors.append((y[i] - np.interp(x[i], grid, ratio)) ** 2)
    return float(np.mean(errors))


def test_the_binned_loo_error_is_the_exact_loo_on_the_binned_representation() -> None:
    x, y = _sample(60, 2)
    order = np.lexsort((y, x))
    xs, ys = x[order], y[order]
    width = (xs[-1] - xs[0]) / (BINS - 1)
    left, lw, rw, binned = _kernel._bin(xs, ys, xs[0], width)
    for bandwidth in _kernel.bandwidth_grid(xs, width)[::5]:
        kernel = _kernel._kernel(float(bandwidth), width)
        fast = _kernel._loo_error(ys, left, lw, rw, _kernel._convolve(binned, kernel), kernel)
        assert fast == pytest.approx(_brute_loo(x, y, float(bandwidth)), abs=1e-12)


def test_probability_fits_stay_in_the_unit_interval() -> None:
    x, y = _sample(400, 3)
    fit = smooth(x, y, np.ones(400, dtype=bool))
    assert not fit.degenerate
    assert np.all((fit.values >= 0.0) & (fit.values <= 1.0))


def test_a_two_point_index_gets_a_positive_bandwidth() -> None:
    """The IQR is zero on a two-point index with unequal masses; the sd fallback applies."""
    x = np.r_[np.zeros(80), np.ones(20)]
    y = np.r_[np.zeros(40), np.ones(40), np.ones(15), np.zeros(5)]
    fit = smooth(x, y, np.ones(100, dtype=bool))
    assert fit.bandwidth is not None and fit.bandwidth > 0.0
    assert 0.0 < fit.values[0] < fit.values[-1] < 1.0


def test_fewer_than_the_minimum_rows_give_the_subpopulation_mean() -> None:
    x, y = _sample(100, 4)
    rows = np.zeros(100, dtype=bool)
    rows[: MIN_ROWS - 1] = True
    fit = smooth(x, y, rows)
    assert fit.degenerate and fit.rows == MIN_ROWS - 1
    np.testing.assert_allclose(fit.values, y[rows].mean(), rtol=0, atol=1e-15)


def test_the_error_falls_with_the_sample_size() -> None:
    """The L2 error on a smooth target falls by at least 1.5 per quadrupling of m."""

    def error(m: int) -> float:
        values = []
        for seed in range(20):
            rng = np.random.default_rng(100 + seed)
            x = rng.uniform(-2.0, 2.0, m)
            truth = 1.0 / (1.0 + np.exp(-np.sin(2.0 * x)))
            y = rng.binomial(1, truth).astype(float)
            fit = smooth(x, y, np.ones(m, dtype=bool))
            values.append(np.mean((fit.values - truth) ** 2))
        return float(np.sqrt(np.mean(values)))

    small, large = error(500), error(2_000)
    assert small / large > 1.5
