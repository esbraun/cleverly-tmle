"""Input checks applied when a :class:`~cleverly.data.CausalData` is built.

Every check here exists because violating it produces a *silently wrong*
estimate rather than a crash: a non-binary treatment quietly redefines the
estimand, missing outcomes without a missingness indicator bias ``Q``, and
constant covariates make the treatment model's design matrix singular.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np

from .._typing import BoolArray, FloatArray, IntArray
from ..exceptions import DataError

__all__ = [
    "check_binary",
    "check_covariates",
    "check_delta",
    "check_outcome",
    "check_weights",
    "encode_binary",
    "infer_family",
]


def check_binary(values: FloatArray, name: str) -> FloatArray:
    """Validate a 0/1 indicator, returning it as floats."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise DataError(f"{name} contains missing or non-finite values")
    unique = np.unique(arr)
    if not np.all(np.isin(unique, (0.0, 1.0))):
        raise DataError(
            f"{name} must be coded 0/1; found values {unique[:6].tolist()}"
            + (" ..." if unique.size > 6 else "")
        )
    if unique.size < 2:
        raise DataError(f"{name} takes only the value {unique[0]:.0f}; both levels must be present")
    return arr


def encode_binary(values: np.ndarray, name: str) -> tuple[FloatArray, tuple[object, object]]:
    """Map a two-level column onto 0/1, returning the level order used.

    Numeric 0/1 columns pass through untouched.  Anything else with exactly two
    distinct levels is encoded with the *sorted* levels, so the mapping is
    reproducible and reported back to the caller rather than guessed at.
    """
    arr = np.asarray(values).reshape(-1)
    if arr.dtype.kind in "fiu":
        numeric = np.asarray(arr, dtype=float)
        unique = np.unique(numeric[np.isfinite(numeric)])
        if np.all(np.isin(unique, (0.0, 1.0))):
            return check_binary(numeric, name), (0, 1)
        if unique.size == 2:
            low, high = float(unique[0]), float(unique[1])
            encoded = np.where(numeric == high, 1.0, 0.0)
            return check_binary(encoded, name), (low, high)
        raise DataError(
            f"{name} must be binary; found {unique.size} distinct values. "
            "Multi-valued treatments are not supported by the classic TMLE estimator."
        )

    unique_obj = np.unique(arr)
    if unique_obj.size != 2:
        raise DataError(
            f"{name} must be binary; found {unique_obj.size} distinct values "
            f"({unique_obj[:6].tolist()})."
        )
    levels = (unique_obj[0], unique_obj[1])
    encoded = np.where(arr == levels[1], 1.0, 0.0)
    return check_binary(encoded, name), levels


def check_outcome(
    y: np.ndarray,
    name: str,
    observed: BoolArray | None,
) -> FloatArray:
    """Validate the outcome, allowing missing values only where unobserved."""
    arr = np.asarray(y, dtype=float).reshape(-1)
    missing = ~np.isfinite(arr)
    if observed is None:
        if missing.any():
            raise DataError(
                f"{name} has {int(missing.sum())} missing value(s) but no missingness "
                "indicator was supplied. Pass delta=<column> (1 = outcome observed) so the "
                "missingness mechanism is estimated and enters the clever covariate."
            )
        return arr

    unexpected = missing & observed
    if unexpected.any():
        raise DataError(
            f"{name} is missing for {int(unexpected.sum())} row(s) flagged as observed "
            "by the missingness indicator"
        )
    # Values at unobserved rows never enter any regression; normalise them so
    # downstream arithmetic cannot propagate NaN through a multiply-by-zero.
    cleaned = np.where(observed, arr, 0.0)
    return np.asarray(cleaned, dtype=float)


def infer_family(y: FloatArray, observed: BoolArray | None) -> str:
    """``"binomial"`` for a 0/1 outcome, ``"gaussian"`` otherwise."""
    values = y if observed is None else y[observed]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise DataError("outcome has no observed values")
    unique = np.unique(finite)
    if unique.size <= 2 and np.all(np.isin(unique, (0.0, 1.0))):
        return "binomial"
    return "gaussian"


def check_delta(delta: np.ndarray, name: str) -> BoolArray:
    """Validate the observed-outcome indicator."""
    arr = np.asarray(delta, dtype=float).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise DataError(f"{name} contains missing values; it must be 1 (observed) or 0 (missing)")
    unique = np.unique(arr)
    if not np.all(np.isin(unique, (0.0, 1.0))):
        raise DataError(f"{name} must be coded 0/1; found values {unique[:6].tolist()}")
    if not np.any(arr == 1.0):
        raise DataError(f"{name} is 0 everywhere: no outcomes are observed")
    return np.asarray(arr == 1.0, dtype=bool)


def check_weights(weights: np.ndarray | None, n: int, name: str = "weights") -> FloatArray:
    """Validate observation weights and normalise them to mean one.

    Normalising keeps the influence curve, and therefore the variance estimate,
    on the same scale as the unweighted case, so weighted and unweighted fits
    are directly comparable.  It also makes the fit invariant to the scale of the
    supplied weights, which is the right behaviour for the tilt they encode --
    see :mod:`cleverly.data.weighting` for what that tilt means and for the
    readings of "weight" this does *not* implement.

    Zero weights are allowed: such a row contributes nothing to the estimate but
    still counts towards ``n``, which is what the i.i.d.-weights model calls for.
    """
    if weights is None:
        return np.ones(n, dtype=float)
    arr = np.asarray(weights, dtype=float).reshape(-1)
    if arr.size != n:
        raise DataError(f"{name} has length {arr.size}, expected {n}")
    if not np.all(np.isfinite(arr)):
        raise DataError(f"{name} contains missing or non-finite values")
    if np.any(arr < 0):
        raise DataError(f"{name} contains negative values")
    total = arr.sum()
    if total <= 0:
        raise DataError(f"{name} sums to zero")
    return np.asarray(arr * (n / total), dtype=float)


def check_covariates(
    w: FloatArray,
    names: Sequence[str],
    *,
    drop_constant: bool = True,
) -> tuple[FloatArray, list[str], list[str]]:
    """Validate the covariate matrix, dropping degenerate columns.

    Returns the retained matrix, the retained names, and the names dropped.
    Constant and exactly duplicated columns are dropped (with a warning) because
    they make the design matrix singular without carrying information; missing
    values are an error, since silently imputing them would change the estimand.
    """
    matrix = np.asarray(w, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.shape[1] != len(names):
        raise DataError(
            f"covariate matrix has {matrix.shape[1]} columns but {len(names)} names were given"
        )
    if matrix.shape[1] == 0:
        raise DataError(
            "no covariates supplied. TMLE needs at least one covariate to adjust for; "
            "for a randomised trial with no baseline covariates use a constant-plus-noise "
            "design or a simple difference in means instead."
        )

    non_finite = ~np.isfinite(matrix)
    if non_finite.any():
        bad = [names[j] for j in np.unique(np.where(non_finite)[1])]
        raise DataError(
            f"covariates {bad} contain missing or non-finite values. Impute them first, or add "
            "an explicit missingness-indicator column, so the adjustment set stays well defined."
        )

    keep = np.ones(matrix.shape[1], dtype=bool)
    dropped: list[str] = []

    if drop_constant:
        for j in range(matrix.shape[1]):
            if np.ptp(matrix[:, j]) == 0.0:
                keep[j] = False
                dropped.append(names[j])

    seen: dict[bytes, int] = {}
    for j in range(matrix.shape[1]):
        if not keep[j]:
            continue
        digest = np.ascontiguousarray(matrix[:, j]).tobytes()
        if digest in seen:
            keep[j] = False
            dropped.append(names[j])
        else:
            seen[digest] = j

    if dropped:
        warnings.warn(
            f"dropped {len(dropped)} constant or duplicated covariate(s): {dropped}",
            UserWarning,
            stacklevel=3,
        )

    retained = [name for name, flag in zip(names, keep, strict=True) if flag]
    if not retained:
        raise DataError("all covariates were constant or duplicated; nothing left to adjust for")
    return matrix[:, keep], retained, dropped


def encode_clusters(values: np.ndarray, name: str) -> IntArray:
    """Map cluster identifiers onto contiguous integer codes."""
    arr = np.asarray(values).reshape(-1)
    _, codes = np.unique(arr, return_inverse=True)
    if np.unique(codes).size == 1:
        raise DataError(f"{name} identifies a single cluster; cluster-robust variance is undefined")
    return np.asarray(codes, dtype=np.int64)
