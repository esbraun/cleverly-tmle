"""Covariate screening for the treatment model.

R's ``tmle`` exposes ``prescreenW.g``: before estimating the treatment mechanism
``g(W)``, keep only the covariates that predict the **outcome**.  The point is the
classic instrument problem.  A covariate that predicts treatment but not the outcome
removes no confounding bias, and in ``g`` it pushes the fitted probabilities towards 0
and 1, which inflates the variance of the clever covariate (Brookhart et al. 2006).
Screening on the outcome drops such covariates; screening on the treatment would keep
exactly them.

The screen here is a lasso of the outcome on the covariates **with the treatment held
fixed**: the outcome and every covariate column are first regressed on a basis of the
treatment, and the lasso, with its penalty chosen by cross-validation (``lambda.min``),
runs on the residuals.  A covariate is kept when its own coefficient or its interaction
with the treatment is nonzero.  Conditioning on the treatment is what drops an
instrument: marginally it predicts the outcome through the treatment, so R's marginal
lasso of ``Y`` on ``W`` can keep it.

``min_retain`` is a floor on the number of covariates kept.  ``None`` takes R's default,
``min(ceil(n_eff / 20), p)``, so the default screens only when the number of covariates
is large next to the effective sample size.  When the lasso keeps fewer than the floor,
covariates are added in the order they enter the lasso path.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.linear_model import LassoCV, lasso_path
from sklearn.preprocessing import SplineTransformer

from .._typing import BoolArray, FloatArray, IntArray

__all__ = [
    "OutcomeScreen",
    "ScreenedLearner",
    "correlation_strength",
    "default_min_retain",
    "learner_for_rows",
    "screen_by_outcome",
]

#: Inner folds of the cross-validated penalty.
SCREEN_FOLDS = 5
#: Interior knots of the spline basis a continuous treatment is partialled on.
_SPLINE_KNOTS = 4
#: Multiplier of the inner-fold hash; any odd constant spreads consecutive rows.
_HASH = 2654435761


def correlation_strength(
    x: FloatArray,
    y: FloatArray,
    *,
    sample_weight: FloatArray | None = None,
) -> FloatArray:
    """(Weighted) Pearson correlation of each column of ``x`` with ``y``.

    :class:`~cleverly.CTMLE` orders covariates by their association with the outcome,
    which is a ranking rather than a selection.

    Parameters
    ----------
    x : ndarray
        ``(n, p)`` columns.
    y : ndarray
        ``(n,)`` target.
    sample_weight : ndarray or None
        ``(n,)`` weights, or ``None`` for equal weights.

    Returns
    -------
    ndarray
        ``(p,)`` correlations, clipped inside ``(-1, 1)``; a constant column gives 0.

    Raises
    ------
    ValueError
        If ``y`` does not have one value per row of ``x``.
    """
    matrix = np.asarray(x, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    target = np.asarray(y, dtype=float).reshape(-1)
    n = matrix.shape[0]
    if target.shape[0] != n:
        raise ValueError(f"y has length {target.shape[0]}, expected {n}")

    weights = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    weights = weights / weights.sum()

    y_centred = target - np.sum(weights * target)
    y_var = np.sum(weights * y_centred**2)
    x_centred = matrix - np.sum(weights[:, None] * matrix, axis=0)
    x_var = np.sum(weights[:, None] * x_centred**2, axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        cov = np.sum(weights[:, None] * x_centred * y_centred[:, None], axis=0)
        r = np.where((x_var > 0) & (y_var > 0), cov / np.sqrt(x_var * y_var), 0.0)
    return np.clip(np.nan_to_num(r), -1.0 + 1e-12, 1.0 - 1e-12)


def default_min_retain(n_covariates: int, n: int, arm_counts: Sequence[int] | None) -> int:
    """R's default floor, ``min(ceil(n_eff / 20), p)``, and at least one.

    ``n_eff`` is ``min(5 * smallest arm count, n)`` for a discrete treatment, as in R's
    ``tmle``.  R applies that rule to a binary treatment only; a continuous or vector
    treatment takes ``n_eff = n`` here, since five times a count of one per distinct dose
    is an artefact of the binary rule.

    Parameters
    ----------
    n_covariates : int
        Number of covariates the screen chooses among.
    n : int
        Number of units.
    arm_counts : sequence of int or None
        Units per arm of a discrete treatment, or ``None`` for a continuous or vector one.

    Returns
    -------
    int
        The floor.
    """
    n_eff = n if arm_counts is None or not len(arm_counts) else min(5 * min(arm_counts), n)
    return max(1, min(math.ceil(n_eff / 20), int(n_covariates)))


def _inner_splits(rows: IntArray, groups: IntArray | None) -> list[tuple[IntArray, IntArray]]:
    """Inner folds of the penalty search, read from neither treatment nor outcome.

    A cluster stays in one fold.  Without clusters a row's fold is a fixed hash of its
    unit index, so data sorted by treatment, time or outcome still gives balanced folds,
    and a unit keeps its inner fold across the nested training sets of one fit.
    """
    if groups is not None:
        labels, codes = np.unique(np.asarray(groups), return_inverse=True)
        k = min(SCREEN_FOLDS, labels.size)
        keys = np.arange(labels.size, dtype=np.uint64)
    else:
        k = min(SCREEN_FOLDS, rows.size)
        codes = np.arange(rows.size)
        keys = np.asarray(rows, dtype=np.uint64)
    hashed = (keys * np.uint64(_HASH)) % np.uint64(2**32)
    # Folds of near-equal size: rank the hashes, then deal the ranks round-robin.
    rank = np.empty(hashed.size, dtype=np.int64)
    rank[np.argsort(hashed, kind="stable")] = np.arange(hashed.size)
    fold = (rank % max(k, 1))[codes]
    splits = []
    for f in range(k):
        test = np.flatnonzero(fold == f)
        train = np.flatnonzero(fold != f)
        if test.size and train.size:
            splits.append((train, test))
    return splits


def _residualise(matrix: FloatArray, basis: FloatArray) -> FloatArray:
    """Residuals of each column of ``matrix`` after least squares on ``basis``."""
    q, _ = np.linalg.qr(basis)
    return matrix - q @ (q.T @ matrix)


class _TreatmentBasis:
    """The treatment terms the screen holds fixed, built on one training set.

    A discrete column enters as drop-first indicators of its levels; a continuous column
    as a natural cubic spline with quantile knots, so a nonlinear dose response leaves no
    treatment signal in the residuals.  ``linear`` is the part the covariates interact
    with: the indicators, and the centred dose.
    """

    def __init__(self, treatment: FloatArray, discrete: Sequence[bool]) -> None:
        self.treatment = treatment
        self.discrete = tuple(discrete)

    def build(self, rows: IntArray) -> tuple[FloatArray, FloatArray]:
        partial: list[FloatArray] = [np.ones((rows.size, 1))]
        linear: list[FloatArray] = []
        for j, is_discrete in enumerate(self.discrete):
            column = self.treatment[rows, j]
            if is_discrete:
                levels = np.unique(column)
                indicators = (column[:, None] == levels[None, 1:]).astype(float)
                partial.append(indicators)
                linear.append(indicators)
                continue
            centred = column - column.mean()
            linear.append(centred[:, None])
            if np.unique(column).size > _SPLINE_KNOTS + 2:
                spline = SplineTransformer(
                    n_knots=_SPLINE_KNOTS + 2,
                    degree=3,
                    knots="quantile",
                    extrapolation="linear",
                    include_bias=False,
                )
                partial.append(spline.fit_transform(column[:, None]))
            else:
                partial.append(centred[:, None])
        return np.column_stack(partial), (
            np.column_stack(linear) if linear else np.zeros((rows.size, 0))
        )


class OutcomeScreen:
    """The unit-level inputs of the screen, and its support on any training set.

    One screen serves every treatment-side fit of a nuisance: the propensity, a dose's
    hazard table, a policy classifier and each factor of a vector's chain rule all read
    :meth:`support` of their training units, so they keep the same covariates.  A
    support is computed once per training set.

    Parameters
    ----------
    covariates : ndarray
        ``(n, p)`` covariates of the treatment model.
    treatment : ndarray
        ``(n, d)`` treatment columns, one per component.
    discrete : sequence of bool
        For each treatment column, whether it is discrete.
    outcome : ndarray
        ``(n,)`` outcome.
    eligible : ndarray
        ``(n,)`` rows the screen may read: outcome observed, treatment recorded, and
        positive weight.
    weights : ndarray
        ``(n,)`` observation weights.
    groups : ndarray or None
        ``(n,)`` cluster codes, or ``None``.
    min_retain : int
        The resolved floor, at least one.
    """

    def __init__(
        self,
        covariates: FloatArray,
        treatment: FloatArray,
        discrete: Sequence[bool],
        outcome: FloatArray,
        eligible: BoolArray,
        weights: FloatArray,
        groups: IntArray | None,
        min_retain: int,
    ) -> None:
        self.covariates = np.asarray(covariates, dtype=float)
        matrix = np.asarray(treatment, dtype=float)
        self.treatment = matrix.reshape(-1, 1) if matrix.ndim == 1 else matrix
        self.basis = _TreatmentBasis(self.treatment, discrete)
        self.outcome = np.asarray(outcome, dtype=float).reshape(-1)
        self.eligible = np.asarray(eligible, dtype=bool)
        self.weights = np.asarray(weights, dtype=float)
        self.groups = None if groups is None else np.asarray(groups)
        self.min_retain = int(min_retain)
        self._cache: dict[bytes, BoolArray] = {}

    @property
    def n_covariates(self) -> int:
        """Number of covariates the screen chooses among."""
        return int(self.covariates.shape[1])

    def support(self, rows: IntArray | None = None) -> BoolArray:
        """The covariates kept on the training set ``rows``.

        Parameters
        ----------
        rows : ndarray or None
            Unit indices of the training set, or ``None`` for every unit.

        Returns
        -------
        ndarray
            ``(p,)`` boolean mask.
        """
        units = (
            np.arange(self.covariates.shape[0], dtype=np.int64)
            if rows is None
            else np.unique(np.asarray(rows, dtype=np.int64))
        )
        key = units.tobytes()
        if key not in self._cache:
            self._cache[key] = self._support(units)
        return self._cache[key].copy()

    def _support(self, units: IntArray) -> BoolArray:
        p = self.n_covariates
        floor = max(1, min(self.min_retain, p))
        if p <= floor:
            return np.ones(p, dtype=bool)
        rows = units[self.eligible[units] & (self.weights[units] > 0.0)]
        if rows.size < 2 * SCREEN_FOLDS:
            return np.ones(p, dtype=bool)
        keep, entry, strength = self._path(rows)
        for index in _entry_order(entry, strength):
            if keep.sum() >= floor:
                break
            keep[index] = True
        return keep

    def _path(self, rows: IntArray) -> tuple[BoolArray, FloatArray, FloatArray]:
        return _lasso_screen(
            self.covariates[rows],
            *self.basis.build(rows),
            self.outcome[rows],
            self.weights[rows],
            rows,
            None if self.groups is None else self.groups[rows],
        )

    def entry_order(self, rows: IntArray | None = None) -> IntArray:
        """Covariate indices in the order the screen admits them on ``rows``.

        The order the floor fills in: entry on the lasso path (largest penalty first),
        then residual correlation with the outcome.

        Parameters
        ----------
        rows : ndarray or None
            Unit indices of the training set, or ``None`` for every unit.

        Returns
        -------
        ndarray
            ``(p,)`` covariate indices.
        """
        units = (
            np.arange(self.covariates.shape[0], dtype=np.int64)
            if rows is None
            else np.unique(np.asarray(rows, dtype=np.int64))
        )
        eligible = units[self.eligible[units] & (self.weights[units] > 0.0)]
        _, entry, strength = self._path(eligible)
        return _entry_order(entry, strength)


def _lasso_screen(
    w: FloatArray,
    partial: FloatArray,
    linear: FloatArray,
    y: FloatArray,
    weights: FloatArray,
    rows: IntArray,
    groups: Any,
) -> tuple[BoolArray, FloatArray, FloatArray]:
    """The lasso of ``y`` on ``[w, w x linear]`` with ``partial`` held fixed.

    Returns, per covariate, whether it is kept at ``lambda.min``, the index on the
    penalty path (largest penalty first) at which its first term turns nonzero, ``inf``
    for never, and its largest absolute residual correlation with the outcome.
    """
    n, p = w.shape
    blocks = [w] + [w * linear[:, [j]] for j in range(linear.shape[1])]
    design = np.column_stack(blocks)
    owner = np.tile(np.arange(p), len(blocks))
    root = np.sqrt(weights / weights.mean())[:, None]
    x = _residualise(design * root, partial * root)
    target = _residualise((y[:, None] * root), partial * root)[:, 0]
    scale = np.sqrt(np.mean(x**2, axis=0))
    live = scale > 1e-10 * max(1.0, float(scale.max(initial=0.0)))
    keep = np.zeros(p, dtype=bool)
    entry = np.full(p, np.inf)
    strength = np.zeros(p)
    splits = _inner_splits(rows, groups)
    if not live.any() or np.std(target) == 0.0 or len(splits) < 2:
        return keep, entry, strength
    x_std = x[:, live] / scale[live]
    owners = owner[live]
    correlation = np.abs(x_std.T @ target) / (np.sqrt(n) * np.linalg.norm(target))
    np.maximum.at(strength, owners, correlation)
    model = LassoCV(fit_intercept=False, cv=splits, max_iter=20_000, tol=1e-6)
    model.fit(x_std, target)
    np.logical_or.at(keep, owners, model.coef_ != 0.0)
    _, path, _ = lasso_path(x_std, target, alphas=model.alphas_, max_iter=20_000, tol=1e-6)
    nonzero = np.abs(path) > 0.0
    first = np.where(nonzero.any(axis=1), nonzero.argmax(axis=1), np.inf)
    np.minimum.at(entry, owners, first)
    return keep, entry, strength


def _entry_order(entry: FloatArray, strength: FloatArray) -> IntArray:
    """Covariates by path entry, then by residual correlation, then by position."""
    p = entry.size
    return np.lexsort((np.arange(p), -strength, entry))


def screen_by_outcome(
    covariates: FloatArray,
    treatment: FloatArray,
    outcome: FloatArray,
    *,
    discrete: bool | Sequence[bool] = True,
    sample_weight: FloatArray | None = None,
    groups: IntArray | None = None,
    min_retain: int | None = None,
) -> BoolArray:
    """Select the covariates that predict the outcome with the treatment held fixed.

    The outcome and each covariate (and its product with the treatment) are regressed on
    a basis of the treatment, and a lasso with its penalty chosen by cross-validation
    (``lambda.min``) runs on the residuals, after weighting and standardising.  A
    covariate is kept when its main term or an interaction is nonzero.  If fewer than
    ``min_retain`` are kept, covariates are added in the order they enter the lasso path,
    then by their residual correlation with the outcome.  This is the selection that
    ``screen_treatment=True`` applies to the treatment model inside each training set.

    Parameters
    ----------
    covariates : ndarray
        ``(n, p)`` covariates.
    treatment : ndarray
        ``(n,)`` treatment, or ``(n, d)`` treatment components.
    outcome : ndarray
        ``(n,)`` outcome, observed on every row passed.
    discrete : bool or sequence of bool
        Whether each treatment column is discrete (indicators) or continuous (a spline).
    sample_weight : ndarray or None
        ``(n,)`` observation weights.
    groups : ndarray or None
        ``(n,)`` cluster codes; an inner fold keeps each cluster whole.
    min_retain : int or None
        The fewest covariates kept.  ``None`` is ``min(ceil(n_eff / 20), p)``
        (:func:`default_min_retain`), which keeps every covariate when ``p`` is small next
        to the effective sample size.

    Returns
    -------
    ndarray
        ``(p,)`` boolean mask of the kept covariates.

    Examples
    --------
    >>> import numpy as np
    >>> from cleverly.learners import screen_by_outcome
    >>> rng = np.random.default_rng(0)
    >>> w = rng.normal(size=(2000, 4))  # confounder, instrument, outcome-only, noise
    >>> a = (rng.uniform(size=2000) < 1 / (1 + np.exp(-w[:, 0] - 2 * w[:, 1]))).astype(float)
    >>> y = a + w[:, 0] + w[:, 2] + rng.normal(size=2000)
    >>> screen_by_outcome(w, a, y, min_retain=1)[[0, 2]].tolist()
    [True, True]
    """
    matrix = np.asarray(covariates, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    n, p = matrix.shape
    columns = np.asarray(treatment, dtype=float)
    columns = columns.reshape(-1, 1) if columns.ndim == 1 else columns
    flags = (discrete,) * columns.shape[1] if isinstance(discrete, bool) else tuple(discrete)
    if columns.shape[0] != n or np.asarray(outcome).reshape(-1).shape[0] != n:
        raise ValueError("covariates, treatment and outcome need one row per unit")
    if min_retain is None:
        counts = (
            np.unique(columns[:, 0], return_counts=True)[1].tolist()
            if len(flags) == 1 and flags[0]
            else None
        )
        floor = default_min_retain(p, n, counts)
    else:
        floor = max(1, min(int(min_retain), p))
    screen = OutcomeScreen(
        matrix,
        columns,
        flags,
        np.asarray(outcome, dtype=float),
        np.ones(n, dtype=bool),
        np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float),
        groups,
        floor,
    )
    return screen.support()


class ScreenMask(BaseEstimator, SelectorMixin):
    """A fixed column selection, the first step of a screened treatment model.

    Parameters
    ----------
    support : ndarray
        ``(m,)`` boolean mask over the design's columns.
    """

    def __init__(self, support: BoolArray) -> None:
        self.support = support

    def fit(self, X: FloatArray, y: Any = None) -> ScreenMask:
        """Record the design width; the selection is fixed.

        Parameters
        ----------
        X : ndarray
            ``(n, m)`` design.
        y : ignored
            Accepted for the scikit-learn contract.

        Returns
        -------
        ScreenMask
            This object.

        Raises
        ------
        ValueError
            If the design does not have one column per mask entry.
        """
        del y
        width = np.asarray(X).shape[1]
        if width != np.asarray(self.support).size:
            raise ValueError(f"the screen mask has {np.asarray(self.support).size} columns")
        self.n_features_in_ = width
        return self

    def _get_support_mask(self) -> BoolArray:
        return np.asarray(self.support, dtype=bool)


class ScreenedLearner:
    """A treatment learner and the screen that selects its covariates per training set.

    It is never fitted itself: each fit loop calls :func:`learner_for_rows` with its
    training units, which returns the learner behind a :class:`ScreenMask` of that
    training set's support.  Fitting it directly raises, so a route that forgets the
    screen fails loudly rather than fitting on every covariate.

    Parameters
    ----------
    learner : Learner
        The treatment learner.
    screen : OutcomeScreen
        The unit-level screen.
    """

    def __init__(self, learner: Any, screen: OutcomeScreen) -> None:
        self.learner = learner
        self.screen = screen

    @property
    def bin_design(self) -> Any:
        """The wrapped learner's ``bin_design``, read by the density route."""
        return getattr(self.learner, "bin_design", None)

    def fit(self, *args: Any, **kwargs: Any) -> ScreenedLearner:  # numpydoc ignore=PR01
        """Refuse: a screened learner is fitted through :func:`learner_for_rows`.

        Raises
        ------
        TypeError
            Always.
        """
        del args, kwargs
        raise TypeError(
            "a screened treatment learner was fitted without its training rows; this is a "
            "defect in cleverly: the fit loop must call learner_for_rows"
        )


def learner_for_rows(
    learner: Any, rows: IntArray | None, width: int, *, covariates_last: bool = False
) -> Any:
    """The learner a fit loop fits on the training units ``rows``.

    A :class:`ScreenedLearner` becomes its learner behind the :class:`ScreenMask` of
    that training set's support; any other learner comes back unchanged.  The design has
    ``width`` columns, and the covariates are its first ``p`` columns, or its last ``p``
    with ``covariates_last=True``.  Every other column (a bin block, a treatment block,
    an earlier treatment component) is always kept.

    Parameters
    ----------
    learner : Learner
        The treatment learner, screened or not.
    rows : ndarray or None
        Unit indices of the training set, or ``None`` for every unit.
    width : int
        Columns of the design the learner is fitted on.
    covariates_last : bool
        Whether the covariates are the design's last columns rather than its first.

    Returns
    -------
    Learner
        The learner to fit.
    """
    if not isinstance(learner, ScreenedLearner):
        return learner
    from sklearn.pipeline import Pipeline

    support = learner.screen.support(rows)
    p = support.size
    mask = np.ones(int(width), dtype=bool)
    if covariates_last:
        mask[width - p :] = support
    else:
        mask[:p] = support
    return Pipeline([("screen", ScreenMask(mask)), ("model", learner.learner)])
