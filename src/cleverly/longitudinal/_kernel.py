"""The univariate kernel smoother of the drift-corrected survival TMLE.

Díaz (2019), Section 5, fits every auxiliary regression of the drift correction as a regression
of an outcome on one scalar index, among a subpopulation of rows; the paper's example (10) is a
kernel regression.  :func:`smooth` is the one deterministic smoother those regressions use: a
Nadaraya-Watson fit with a Gaussian kernel (local constant), on a linearly binned grid, with the
bandwidth chosen by leave-one-out squared error over a fixed grid.

A local-constant fit is a convex combination of outcomes, so a probability outcome gives a fit in
``[0, 1]``: the products and denominators built from the fits stay in range.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["BINS", "MIN_ROWS", "SmoothFit", "bandwidth_grid", "smooth"]

#: The number of grid points of the linear binning.
BINS = 401
#: Below this many subpopulation rows the regression is the subpopulation mean.
MIN_ROWS = 20
#: The bandwidth multipliers ``2 ** ((j - 10) / 5)``, ``j = 0, ..., 20``.
_MULTIPLIERS = 2.0 ** ((np.arange(21) - 10) / 5.0)


@dataclass(frozen=True)
class SmoothFit:
    """One fitted regression.

    Parameters
    ----------
    values : numpy.ndarray
        The fit evaluated at every row's index.
    bandwidth : float or None
        The chosen bandwidth, ``None`` for the degenerate branch.
    degenerate : bool
        Whether the fit is the subpopulation mean (too few rows, a constant outcome, or an index
        with zero spread).
    rows : int
        The number of subpopulation rows.
    """

    values: NDArray[np.float64]
    bandwidth: float | None
    degenerate: bool
    rows: int


def _scale(x: NDArray[np.float64]) -> float:
    """The spread of the rule-of-thumb bandwidth: ``min(sd, IQR / 1.34)``, or ``sd`` at IQR 0."""
    sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    q75, q25 = np.percentile(x, [75.0, 25.0])
    iqr = float(q75 - q25)
    return min(sd, iqr / 1.34) if iqr > 0.0 else sd


def bandwidth_grid(x: NDArray[np.float64], width: float) -> NDArray[np.float64]:
    """The candidate bandwidths ``c_j h_0``, each floored at the bin width ``width``.

    Parameters
    ----------
    x : numpy.ndarray
        The index values of the subpopulation rows.
    width : float
        The bin width of the grid.

    Returns
    -------
    numpy.ndarray
        Twenty-one bandwidths, increasing.
    """
    base = 0.9 * _scale(x) * x.size ** (-0.2)
    return np.maximum(_MULTIPLIERS * base, width)


def _bin(
    x: NDArray[np.float64], y: NDArray[np.float64], low: float, width: float
) -> tuple[NDArray[np.intp], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Linear binning: each row's left bin, its two weights, and the binned counts and sums."""
    position = (x - low) / width
    left = np.clip(np.floor(position).astype(np.intp), 0, BINS - 2)
    right_weight = np.clip(position - left, 0.0, 1.0)
    left_weight = 1.0 - right_weight
    counts = np.zeros(BINS)
    sums = np.zeros(BINS)
    # Rows were sorted by (index, outcome), so tied indices add in one value order and the
    # result does not depend on the row order of the input.
    np.add.at(counts, left, left_weight)
    np.add.at(counts, left + 1, right_weight)
    np.add.at(sums, left, left_weight * y)
    np.add.at(sums, left + 1, right_weight * y)
    return left, left_weight, right_weight, np.stack([counts, sums])


def _kernel(bandwidth: float, width: float) -> NDArray[np.float64]:
    """The Gaussian kernel at every grid offset ``-(B - 1), ..., B - 1``."""
    offsets = np.arange(-(BINS - 1), BINS) * width
    return np.exp(-0.5 * (offsets / bandwidth) ** 2)


def _convolve(binned: NDArray[np.float64], kernel: NDArray[np.float64]) -> NDArray[np.float64]:
    """Kernel sums on the grid for the binned counts and sums (rows 0 and 1 of ``binned``)."""
    return np.stack([np.convolve(row, kernel)[BINS - 1 : 2 * BINS - 1] for row in binned])


def _loo_error(
    y: NDArray[np.float64],
    left: NDArray[np.intp],
    left_weight: NDArray[np.float64],
    right_weight: NDArray[np.float64],
    sums: NDArray[np.float64],
    kernel: NDArray[np.float64],
) -> float:
    """The leave-one-out squared error on the binned representation.

    Each row's own binned weight is removed from the grid sums at its two bins, and the fit at
    the row is the interpolation of the two reduced grid ratios, as :func:`smooth` evaluates.
    """
    near = kernel[BINS - 1]
    step = kernel[BINS]
    own_left = near * left_weight + step * right_weight
    own_right = step * left_weight + near * right_weight
    fits = np.zeros(y.size)
    for bins, own, weight in ((left, own_left, left_weight), (left + 1, own_right, right_weight)):
        denominator = sums[0, bins] - own
        numerator = sums[1, bins] - own * y
        safe = denominator > 1e-300
        ratio = np.where(safe, numerator / np.where(safe, denominator, 1.0), np.nan)
        fits += weight * ratio
    residual = y - fits
    finite = np.isfinite(residual)
    if not finite.any():
        return float("inf")
    # A row whose removal empties a bin's neighbourhood has no LOO fit; it scores as the worst.
    worst = float(np.max(np.abs(residual[finite]))) + 1.0
    return float(np.mean(np.where(finite, residual, worst) ** 2))


def smooth(
    index: NDArray[np.float64], outcome: NDArray[np.float64], rows: NDArray[np.bool_]
) -> SmoothFit:
    """Regress ``outcome`` on ``index`` among ``rows`` and evaluate the fit at every row.

    Parameters
    ----------
    index : numpy.ndarray
        The scalar index of every row.
    outcome : numpy.ndarray
        The outcome of every row; only the subpopulation rows are read.
    rows : numpy.ndarray
        The subpopulation, a boolean mask.

    Returns
    -------
    SmoothFit
        The fit at every row, the chosen bandwidth and whether the degenerate branch applied.
    """
    index = np.asarray(index, dtype=float)
    mask = np.asarray(rows, dtype=bool)
    m = int(mask.sum())
    if m == 0:
        return SmoothFit(np.zeros(index.size), None, True, 0)
    x = index[mask]
    y = np.asarray(outcome, dtype=float)[mask]
    order = np.lexsort((y, x))
    x, y = x[order], y[order]
    low, high = float(x[0]), float(x[-1])
    if bool(np.all(y == y[0])):
        return SmoothFit(np.full(index.size, float(y[0])), None, True, m)
    if m < MIN_ROWS or high <= low:
        return SmoothFit(np.full(index.size, float(np.mean(y))), None, True, m)
    width = (high - low) / (BINS - 1)
    left, left_weight, right_weight, binned = _bin(x, y, low, width)
    candidates = bandwidth_grid(x, width)
    best, best_error = candidates[-1], float("inf")
    # From the largest bandwidth down with a strict comparison, so a tie keeps the larger one.
    for bandwidth in candidates[::-1]:
        kernel = _kernel(float(bandwidth), width)
        error = _loo_error(y, left, left_weight, right_weight, _convolve(binned, kernel), kernel)
        if error < best_error:
            best, best_error = bandwidth, error
    sums = _convolve(binned, _kernel(float(best), width))
    grid = np.linspace(low, high, BINS)
    ratio = sums[1] / np.maximum(sums[0], 1e-300)
    return SmoothFit(np.interp(index, grid, ratio), float(best), False, m)
