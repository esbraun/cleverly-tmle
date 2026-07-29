"""Covariate screening for the treatment model.

R's ``tmle`` exposes ``prescreenW.g``: before estimating ``g(W) = P(A = 1 | W)``,
drop covariates that show no marginal association with treatment.  The point is
not parsimony for its own sake -- including covariates that predict treatment but
not the outcome inflates the variance of the clever covariate without reducing
bias, which is the classic instrument-inflation problem.

``min_retain`` guarantees a floor on the number of covariates kept, so a
screening step can never leave the treatment model empty.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin

from .._typing import BoolArray, FloatArray

__all__ = ["CorrelationScreener", "correlation_strength", "screen_by_correlation"]

DEFAULT_THRESHOLD = 0.1


def correlation_strength(
    x: FloatArray,
    y: FloatArray,
    *,
    sample_weight: FloatArray | None = None,
) -> FloatArray:
    """(Weighted) Pearson correlation of each column of ``x`` with ``y``.

    Split out from :func:`screen_by_correlation` because the correlations are useful
    on their own -- :class:`~cleverly.CTMLE` orders covariates by their association
    with the outcome, which is a ranking rather than a threshold.
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


def screen_by_correlation(
    x: FloatArray,
    y: FloatArray,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    min_retain: int | None = None,
    sample_weight: FloatArray | None = None,
) -> BoolArray:
    """Select columns of ``x`` marginally associated with ``y``.

    Association is the (weighted) Pearson correlation, whose two-sided p-value
    comes from the usual ``t = r * sqrt((n - 2) / (1 - r^2))`` statistic.  With a
    binary ``y`` this is the point-biserial correlation, which is what R's
    correlation-based pre-screen uses.

    Columns whose p-value falls below ``threshold`` are kept.  If fewer than
    ``min_retain`` survive, the strongest-associated columns are added back.
    """
    matrix = np.asarray(x, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    target = np.asarray(y, dtype=float).reshape(-1)
    n, p = matrix.shape
    if target.shape[0] != n:
        raise ValueError(f"y has length {target.shape[0]}, expected {n}")
    floor = p if min_retain is None else int(min_retain)
    floor = max(1, min(floor, p))

    if n < 3:
        return np.ones(p, dtype=bool)

    r = correlation_strength(matrix, target, sample_weight=sample_weight)

    t_stat = np.abs(r) * np.sqrt((n - 2) / (1.0 - r**2))
    pvalues = 2.0 * stats.t.sf(t_stat, df=n - 2)

    keep = pvalues < threshold
    if keep.sum() < floor:
        order = np.argsort(pvalues, kind="stable")
        keep = np.zeros(p, dtype=bool)
        keep[order[:floor]] = True
    return keep


class CorrelationScreener(BaseEstimator, SelectorMixin):
    """scikit-learn selector wrapping :func:`screen_by_correlation`.

    Usable inside a :class:`~sklearn.pipeline.Pipeline`, which is how the
    treatment-model screen is applied:

    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.pipeline import make_pipeline
    >>> model = make_pipeline(CorrelationScreener(), LogisticRegression())
    """

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        min_retain: int | None = None,
    ) -> None:
        self.threshold = threshold
        self.min_retain = min_retain

    def fit(
        self,
        X: FloatArray,
        y: FloatArray,
        sample_weight: FloatArray | None = None,
    ) -> CorrelationScreener:
        matrix = np.asarray(X, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        self.n_features_in_ = matrix.shape[1]
        self.support_ = screen_by_correlation(
            matrix,
            y,
            threshold=self.threshold,
            min_retain=self.min_retain,
            sample_weight=sample_weight,
        )
        return self

    def _get_support_mask(self) -> BoolArray:
        if not hasattr(self, "support_"):
            raise AttributeError("CorrelationScreener has not been fitted")
        return self.support_
