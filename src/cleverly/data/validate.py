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
from ..exceptions import DataError, DataWarning

__all__ = [
    "MAX_TREATMENT_LEVELS",
    "MIN_CONTINUOUS_LEVELS",
    "RANDOMIZED_INTERCEPT",
    "arm_indicators",
    "check_binary",
    "check_covariates",
    "check_delta",
    "check_outcome",
    "check_weights",
    "encode_binary",
    "encode_continuous_treatment",
    "encode_treatment",
    "infer_family",
]

#: The one covariate column ``cleverly`` supplies itself, for
#: ``PointTreatment(randomized=True, adjustment=())`` -- an identification claim of *no*
#: adjustment, which still needs a well-formed design for learners that fit their own
#: intercept.  Producer and consumer live in different packages, so the name is defined
#: once here rather than written out at both ends: it was a bare literal in
#: ``cleverly.study`` and another in :func:`check_covariates`, where it is the only thing
#: standing between the column and ``drop_constant``.  Renaming one of the two would have
#: dropped the column and left an unadjusted randomized fit with no covariates at all.
RANDOMIZED_INTERCEPT = "__cleverly_randomized_intercept__"

#: Most treatment levels the estimator will accept.  Each level costs a counterfactual
#: mean, a clever-covariate column and a row/column of the Newton solve, and positivity
#: degrades quickly as the arms multiply, so a treatment with more levels than this is
#: almost always one that should be collapsed or modelled as continuous.  The limit is a
#: guard against a mis-typed column silently becoming a 200-arm fit, not a statistical
#: threshold.
MAX_TREATMENT_LEVELS = 20


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


def encode_treatment(
    values: np.ndarray, name: str, *, min_per_arm: int = 1, remedy: str | None = None
) -> tuple[FloatArray, tuple[object, ...]]:
    """Map a ``K``-level treatment onto codes ``0 .. K-1``, returning the level order.

    The generalisation of :func:`encode_binary` to more than two arms, and a strict
    superset of it: a two-level column is encoded to exactly the same codes and reported
    with exactly the same ``levels`` tuple, including the numeric ``0/1`` pass-through
    that leaves the caller's array untouched.  That equality is what keeps a binary fit
    bit-for-bit identical, and ``tests/unit/test_causal_data.py`` asserts it directly
    rather than trusting the reading.

    The returned ``levels`` are the caller's *original* labels in sorted order, and the
    code for a row is that label's index.  Everything user-facing -- parameter names,
    positivity tables, error messages -- is written in terms of the labels, so a fit on
    ``{"low", "medium", "high"}`` never makes the analyst translate to ``2.0``.

    Parameters
    ----------
    min_per_arm:
        Reject a level with fewer rows than this.  A near-empty arm is not a positivity
        problem to be reported later; its counterfactual mean is not estimable at all,
        and a cross-fit would hand some fold no rows in that arm.
    remedy:
        What to tell the caller to do when the column has more levels than the estimator
        accepts.  The default names ``treatment_kind='continuous'``, which is the answer
        for a :class:`~cleverly.data.CausalData`; a caller that does not offer that
        keyword must pass its own, since an error naming a keyword the entry point does
        not take is worse than no suggestion at all.
    """
    arr = np.asarray(values).reshape(-1)

    if arr.dtype.kind in "fiu":
        numeric = np.asarray(arr, dtype=float)
        if not np.all(np.isfinite(numeric)):
            raise DataError(f"{name} contains missing or non-finite values")
        unique = np.unique(numeric)
        # The binary 0/1 pass-through, preserved exactly: same array, same ``(0, 1)``.
        if unique.size == 2 and np.all(np.isin(unique, (0.0, 1.0))):
            return _check_arms(numeric, (0, 1), name, min_per_arm), (0, 1)
        levels_num: tuple[object, ...] = tuple(float(v) for v in unique)
        _reject_arm_count(len(levels_num), name, remedy)
        codes = np.searchsorted(unique, numeric).astype(float)
        return _check_arms(codes, levels_num, name, min_per_arm), levels_num

    unique_obj = np.unique(arr)
    levels_obj: tuple[object, ...] = tuple(unique_obj.tolist())
    _reject_arm_count(len(levels_obj), name, remedy)
    codes = np.searchsorted(unique_obj, arr).astype(float)
    return _check_arms(codes, levels_obj, name, min_per_arm), levels_obj


#: Below this many distinct values, a treatment declared continuous is *probably* a set of
#: arms the caller forgot to declare -- so it warns.  It does not refuse, because a
#: coarse support is not an obstacle to the estimator: the density on a handful of points
#: is a probability mass function with unit-width bins, and a shift along an ordered
#: discrete dose is a perfectly well-defined modified treatment policy.  (That case is not
#: hypothetical -- it is what ``tests/discrete_law_shift.py`` uses to check the influence
#: curve against a numerically differentiated one, which needs finite support.)  Well below
#: :data:`MAX_TREATMENT_LEVELS`, so the two readings overlap and the choice stays the
#: caller's.
MIN_CONTINUOUS_LEVELS = 10


def encode_continuous_treatment(values: np.ndarray, name: str) -> FloatArray:
    """Validate a numeric treatment that is to be modelled on a continuum.

    Unlike :func:`encode_treatment` this returns the values *themselves* rather than
    codes: there are no arms to code, and the numbers carry the spacing that a shift
    intervention moves along.  Nothing here is sorted, binned or relabelled -- the
    binning that the conditional density estimator does is a property of that estimator
    and of the fold it was fit on, not of the data container.
    """
    arr = np.asarray(values).reshape(-1)
    if arr.dtype.kind not in "fiu":
        raise DataError(
            f"{name} is not numeric ({arr.dtype}), so it cannot be treated as continuous. "
            "A continuous treatment is a quantity a shift can move along; a categorical "
            "column is a set of arms, and needs treatment_kind='discrete'."
        )
    numeric = np.asarray(arr, dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise DataError(f"{name} contains missing or non-finite values")
    distinct = int(np.unique(numeric).size)
    if distinct < 2:
        raise DataError(
            f"{name} takes only one value; a treatment needs at least two for a "
            "counterfactual contrast to be defined"
        )
    if distinct < MIN_CONTINUOUS_LEVELS:
        warnings.warn(
            f"{name} was declared continuous but takes only {distinct} distinct values. "
            "That is estimable -- the density becomes a probability mass function and a "
            "shift moves along the ordered values -- but if those values are the arms of "
            "a categorical treatment, dropping treatment_kind='continuous' gives the "
            "per-arm estimands instead, which are what most analyses of a few levels "
            "want.",
            DataWarning,
            stacklevel=3,
        )
    return numeric


def arm_indicators(codes: np.ndarray, n_levels: int) -> FloatArray:
    r"""Drop-first indicators for arm codes ``0 .. n_levels-1``: ``(n, n_levels-1)``.

    **The one place a treatment enters a design matrix.**  Every design that conditions
    on an arm goes through here -- :meth:`~cleverly.data.CausalData.treatment_block` for
    a point treatment, and
    :meth:`~cleverly.longitudinal.data.LongitudinalData.history_design` for each earlier
    node of a longitudinal mechanism -- so the encoding is one decision with one
    implementation rather than a convention two modules happen to share.  The invariant
    is recorded in ``docs/architecture-invariants.md``.

    **Two arms return the single 0/1 code column itself**, which is exactly the design
    the binary estimators have always been handed.  That is not a shortcut: it is what
    keeps a binary fit bit for bit what it was, and ``K - 1`` is one column at ``K = 2``
    anyway, so the general rule and the compatibility guarantee are the same statement.

    With more than two arms a *single numeric column* would be wrong, not merely crude:
    it would impose a linear dose-response, forcing
    :math:`\bar Q(2, W) - \bar Q(1, W) = \bar Q(1, W) - \bar Q(0, W)` for any learner
    linear in its design, and so shrink the very contrasts the fit exists to estimate.
    On a longitudinal mechanism the same column would force
    :math:`g_2(\cdot \mid H_2, A_1)` to move monotonically in :math:`A_1`.  Indicators
    leave the arms unconstrained.

    The first arm is dropped rather than one-hot encoding all ``K``, so an unregularised
    model with an intercept has a full-rank design.  Which arm is dropped is a property
    of the design only and does not privilege any arm in the estimand: counterfactual
    means are all evaluated by prediction, and the reference used for *contrasts* is a
    separate, caller-chosen thing.
    """
    c = np.asarray(codes, dtype=float).reshape(-1)
    if n_levels < 2:
        raise DataError(
            f"an arm-coded design needs at least two levels; got {n_levels}. A treatment "
            "with one arm has no counterfactual contrast to encode."
        )
    if n_levels == 2:
        return c.reshape(-1, 1)
    return np.column_stack([(c == float(level)).astype(float) for level in range(1, n_levels)])


#: What to suggest when an arm-coded column has more levels than the estimator accepts and
#: the caller offers ``treatment_kind=``.  Kept beside :func:`_reject_arm_count` rather than
#: inlined into it, because a caller without that keyword has to supply its own.
CONTINUOUS_REMEDY = (
    "Collapse the levels into the contrast you actually want to report, or -- if the "
    "treatment is really continuous -- pass treatment_kind='continuous', which models it "
    "with a conditional density and reports shift interventions rather than a mean per arm."
)


def _reject_arm_count(k: int, name: str, remedy: str | None = None) -> None:
    if k < 2:
        raise DataError(
            f"{name} takes only one value; a treatment needs at least two levels for a "
            "counterfactual contrast to be defined"
        )
    if k > MAX_TREATMENT_LEVELS:
        raise DataError(
            f"{name} has {k} distinct levels, above the limit of {MAX_TREATMENT_LEVELS}. "
            + (CONTINUOUS_REMEDY if remedy is None else remedy)
        )


def _check_arms(
    codes: FloatArray, levels: tuple[object, ...], name: str, min_per_arm: int
) -> FloatArray:
    """Every declared level must be present, with enough rows to estimate its mean."""
    counts = np.bincount(np.asarray(codes, dtype=np.int64), minlength=len(levels))
    thin = [
        (levels[i], int(counts[i])) for i in range(len(levels)) if counts[i] < max(min_per_arm, 1)
    ]
    if thin:
        detail = ", ".join(f"{level!r}: {count} row(s)" for level, count in thin)
        raise DataError(
            f"{name} has too few observations in {len(thin)} of its {len(levels)} levels "
            f"({detail}); each level needs at least {max(min_per_arm, 1)} for its "
            "counterfactual mean to be estimable"
        )
    return np.asarray(codes, dtype=float)


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

    # ``PointTreatment(randomized=True, adjustment=())`` supplies one reserved zero
    # column so ordinary learners can fit their own intercept.  It is not an adjustment
    # variable and must remain constant: replacing it by row position would silently
    # introduce a data-order-dependent nuisance model for an unadjusted randomized effect.
    # Kept per column rather than by comparing the whole list, so the exemption still
    # applies -- and still means the same thing -- if the reserved column ever travels
    # alongside another one.
    if drop_constant:
        for j in range(matrix.shape[1]):
            if names[j] == RANDOMIZED_INTERCEPT:
                continue
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
