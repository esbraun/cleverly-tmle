"""Working-model dictionaries for adaptive TMLE of a trial pooled with external data.

Adaptive TMLE (van der Laan, Qiu, Tarp and van der Laan, arXiv:2405.07186v2, Sections 4-5)
selects two working models by a lasso over a dictionary of functions:

``phi(W)``
    the dictionary of the conditional effect ``tau_A(W)`` on the pooled data, with
    ``phi_0 = 1``;
``psi(W, A)``
    the dictionary of the bias ``tau_S(W, A)`` between the trial and the external data,
    ``[1, A, phi_-0(W), A phi_-0(W)]`` with ``phi_-0`` the columns of ``phi`` other than
    the constant. When the external data hold controls only, the bias is learned on the
    control rows, so ``psi(W, A) = phi(W)``.

:class:`WorkingBasis` is the default transformer of ``W`` into the non-constant columns of
``phi``. :class:`WorkingDictionary` adds the constant, builds ``psi`` and drops exactly
collinear columns with a pivoted QR on the selection rows (:func:`independent_columns`).
A declared transformer replaces :class:`WorkingBasis` and goes through the same steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.linalg
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import SplineTransformer

from .._typing import FloatArray, IntArray

__all__ = [
    "COLLINEARITY_TOLERANCE",
    "WorkingBasis",
    "WorkingDictionary",
    "bias_columns",
    "independent_columns",
]

#: The relative size below which a column's part outside the span of the columns kept
#: before it counts as zero, so that the column is dropped as exactly collinear.
COLLINEARITY_TOLERANCE = 1e-10


class WorkingBasis(TransformerMixin, BaseEstimator):
    """The default dictionary of the working models, without its constant.

    A covariate with at least ``n_knots + degree + 1`` distinct values on the fit rows
    gets cubic B-splines on quantile knots without the intercept column. The spline span
    holds the linear function, so no separate main term is added. A covariate with fewer
    distinct values gets one indicator column for each level other than the smallest. Each
    pair of covariates then gets the product of the two covariates, each centred and scaled
    by its mean and standard deviation on the fit rows. A covariate constant on the fit rows
    contributes no column.

    Parameters
    ----------
    n_knots : int, default=5
        The number of quantile knots of each spline.
    degree : int, default=3
        The degree of each spline.
    interactions : bool, default=True
        Whether to add the pairwise products.

    See Also
    --------
    WorkingDictionary : Adds the constant and the bias columns, and drops collinear columns.
    """

    def __init__(self, n_knots: int = 5, degree: int = 3, interactions: bool = True) -> None:
        self.n_knots = n_knots
        self.degree = degree
        self.interactions = interactions

    def fit(self, X: Any, y: Any = None) -> WorkingBasis:
        """Learn the knots, levels and scales on the fit rows.

        Parameters
        ----------
        X : array-like of shape (n, d)
            The covariates of the fit rows.
        y : None
            Ignored.

        Returns
        -------
        WorkingBasis
            The fitted transformer.
        """
        del y
        matrix = _as_matrix(X)
        threshold = self.n_knots + self.degree + 1
        self.splines_: dict[int, SplineTransformer] = {}
        self.levels_: dict[int, FloatArray] = {}
        for j in range(matrix.shape[1]):
            values = np.unique(matrix[:, j])
            if values.size >= threshold:
                self.splines_[j] = SplineTransformer(
                    n_knots=self.n_knots,
                    degree=self.degree,
                    knots="quantile",
                    include_bias=False,
                ).fit(matrix[:, [j]])
            else:
                self.levels_[j] = values[1:]
        self.mean_ = matrix.mean(axis=0)
        self.scale_ = matrix.std(axis=0)
        varying = [j for j in range(matrix.shape[1]) if self.scale_[j] > 0.0]
        self.pairs_ = (
            [(j, k) for i, j in enumerate(varying) for k in varying[i + 1 :]]
            if self.interactions
            else []
        )
        self.n_features_in_ = matrix.shape[1]
        return self

    def transform(self, X: Any) -> FloatArray:
        """Map covariates to the dictionary columns.

        Parameters
        ----------
        X : array-like of shape (m, d)
            The covariates.

        Returns
        -------
        ndarray of float
            The ``(m, q)`` columns, in the order of :meth:`get_feature_names_out`.
        """
        matrix = _as_matrix(X)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                f"the working basis was fitted on {self.n_features_in_} covariates; got "
                f"{matrix.shape[1]}"
            )
        blocks: list[FloatArray] = []
        for j in range(matrix.shape[1]):
            if j in self.splines_:
                blocks.append(np.asarray(self.splines_[j].transform(matrix[:, [j]]), float))
            else:
                levels = self.levels_[j]
                blocks.append((matrix[:, [j]] == levels[None, :]).astype(float))
        for j, k in self.pairs_:
            left = (matrix[:, j] - self.mean_[j]) / self.scale_[j]
            right = (matrix[:, k] - self.mean_[k]) / self.scale_[k]
            blocks.append((left * right)[:, None])
        if not blocks:
            return np.zeros((matrix.shape[0], 0))
        return np.column_stack(blocks)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Name each dictionary column.

        Parameters
        ----------
        input_features : sequence of str or None
            The covariate names. ``None`` names them ``x0``, ``x1``, and so on.

        Returns
        -------
        ndarray of str
            One name for each column of :meth:`transform`.
        """
        names = (
            [f"x{j}" for j in range(self.n_features_in_)]
            if input_features is None
            else [str(name) for name in input_features]
        )
        out: list[str] = []
        for j in range(self.n_features_in_):
            if j in self.splines_:
                count = int(self.splines_[j].n_features_out_)
                out.extend(f"{names[j]}_spline{b}" for b in range(count))
            else:
                out.extend(f"{names[j]}=={level:g}" for level in self.levels_[j])
        out.extend(f"{names[j]}*{names[k]}" for j, k in self.pairs_)
        return np.asarray(out, dtype=object)


def _as_matrix(X: Any) -> FloatArray:
    matrix = np.asarray(X, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2:
        raise ValueError(f"covariates must be a 2-D array; got shape {matrix.shape}")
    return matrix


def independent_columns(
    matrix: FloatArray, n_fixed: int, tol: float = COLLINEARITY_TOLERANCE
) -> IntArray:
    """The columns of a matrix that a pivoted QR keeps, with the leading ones forced in.

    The first ``n_fixed`` columns are kept and must be linearly independent. Each other
    column is scaled to unit norm, its projection on the fixed columns is removed, and a
    QR decomposition with column pivoting orders what is left. A column is kept while its
    pivot exceeds ``tol``, so a column whose part outside the span of the columns kept
    before it is below ``tol`` of its own norm is dropped. A zero column is dropped.

    Parameters
    ----------
    matrix : ndarray of float
        The ``(n, p)`` columns on the selection rows.
    n_fixed : int
        The number of leading columns that are always kept.
    tol : float, default=1e-10
        The relative pivot below which a column is dropped.

    Returns
    -------
    ndarray of int
        The kept column positions, ascending.

    Raises
    ------
    ValueError
        When the fixed columns are collinear on these rows.
    """
    x = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(x, axis=0)
    fixed = x[:, :n_fixed]
    if n_fixed:
        if np.any(norms[:n_fixed] == 0.0):
            raise ValueError("an unpenalized working-model column is zero on the selection rows")
        q, r = np.linalg.qr(fixed / norms[:n_fixed])
        if np.min(np.abs(np.diag(r))) <= tol:
            raise ValueError("the unpenalized working-model columns are collinear")
    else:
        q = np.zeros((x.shape[0], 0))
    rest = np.flatnonzero(norms[n_fixed:] > 0.0) + n_fixed
    if rest.size == 0:
        return np.arange(n_fixed, dtype=np.int64)
    scaled = x[:, rest] / norms[rest]
    residual = scaled - q @ (q.T @ scaled)
    _, factor, pivots = scipy.linalg.qr(residual, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(factor))
    rank = int(np.sum(diagonal > tol))
    kept = np.sort(rest[pivots[:rank]])
    return np.concatenate([np.arange(n_fixed, dtype=np.int64), kept.astype(np.int64)])


def bias_columns(phi: FloatArray, treatment: FloatArray | float, controls_only: bool) -> FloatArray:
    """``[1, A, phi_-0, A phi_-0]``, or ``phi`` itself when the external data are controls.

    Parameters
    ----------
    phi : ndarray of float
        The ``(n, J)`` kept columns of the effect dictionary; column 0 is the constant.
    treatment : ndarray of float or float
        The treatment value at which to build the columns, one per row or one for every row.
    controls_only : bool
        Whether the external data hold controls only.

    Returns
    -------
    ndarray of float
        The ``(n, K)`` raw bias dictionary, before its collinear columns are dropped.
    """
    if controls_only:
        return np.asarray(phi, dtype=float)
    a = np.broadcast_to(np.asarray(treatment, dtype=float), (phi.shape[0],)).reshape(-1, 1)
    rest = phi[:, 1:]
    return np.column_stack([np.ones(phi.shape[0]), a[:, 0], rest, a * rest])


@dataclass(frozen=True)
class WorkingDictionary:
    """The two working-model dictionaries of one selection, fitted on its selection rows.

    Parameters
    ----------
    basis : Any
        The fitted transformer of ``W`` into the non-constant columns of ``phi``.
    phi_keep : ndarray of int
        The positions of ``[1, basis(W)]`` that the pivoted QR kept. Position 0 is the
        constant.
    psi_keep : ndarray of int
        The positions of :func:`bias_columns` that the pivoted QR kept. Position 0 is the
        constant and, with external rows in both arms, position 1 is ``A``.
    controls_only : bool
        Whether the external data hold controls only.
    names : tuple of str
        The names of the columns of ``[1, basis(W)]``.
    """

    basis: Any
    phi_keep: IntArray
    psi_keep: IntArray
    controls_only: bool
    names: tuple[str, ...]

    @classmethod
    def fit(
        cls,
        basis: Any,
        covariates: FloatArray,
        treatment: FloatArray,
        bias_rows: IntArray,
        *,
        controls_only: bool,
    ) -> WorkingDictionary:
        """Fit a clone of ``basis`` on the selection rows and keep the independent columns.

        Parameters
        ----------
        basis : Any
            A scikit-learn transformer of ``W``, or ``None`` for :class:`WorkingBasis`.
        covariates : ndarray of float
            ``W`` on the selection rows.
        treatment : ndarray of float
            ``A`` on the selection rows.
        bias_rows : ndarray of int
            The positions, within the selection rows, that the bias model is selected on.
        controls_only : bool
            Whether the external data hold controls only.

        Returns
        -------
        WorkingDictionary
            The fitted dictionaries.
        """
        fitted = clone(WorkingBasis() if basis is None else basis).fit(covariates)
        raw = _with_constant(fitted, covariates)
        phi_keep = independent_columns(raw, 1)
        phi = raw[:, phi_keep]
        bias = bias_columns(phi[bias_rows], treatment[bias_rows], controls_only)
        psi_keep = independent_columns(bias, 1 if controls_only else 2)
        return cls(fitted, phi_keep, psi_keep, controls_only, _names(fitted, raw.shape[1]))

    def phi(self, covariates: FloatArray) -> FloatArray:
        """The kept effect columns at ``W``.

        Parameters
        ----------
        covariates : ndarray of float
            ``W``, one row per point.

        Returns
        -------
        ndarray of float
            The ``(m, J)`` columns.
        """
        return _with_constant(self.basis, covariates)[:, self.phi_keep]

    def psi(self, covariates: FloatArray, treatment: FloatArray | float) -> FloatArray:
        """The kept bias columns at ``(W, A)``.

        Parameters
        ----------
        covariates : ndarray of float
            ``W``, one row per point.
        treatment : ndarray of float or float
            ``A`` at each point, or one value for every point.

        Returns
        -------
        ndarray of float
            The ``(m, K)`` columns.
        """
        return bias_columns(self.phi(covariates), treatment, self.controls_only)[:, self.psi_keep]

    @property
    def phi_names(self) -> tuple[str, ...]:
        """The names of the kept effect columns."""
        return tuple(self.names[j] for j in self.phi_keep)

    @property
    def psi_names(self) -> tuple[str, ...]:
        """The names of the kept bias columns."""
        rest = [self.names[j] for j in self.phi_keep[1:]]
        raw = (
            ["1", *rest]
            if self.controls_only
            else ["1", "A", *rest, *(f"A*{name}" for name in rest)]
        )
        return tuple(raw[j] for j in self.psi_keep)


def _with_constant(basis: Any, covariates: FloatArray) -> FloatArray:
    columns = np.asarray(basis.transform(covariates), dtype=float)
    if columns.ndim == 1:
        columns = columns[:, None]
    return np.column_stack([np.ones(columns.shape[0]), columns])


def _names(basis: Any, width: int) -> tuple[str, ...]:
    try:
        names = [str(name) for name in basis.get_feature_names_out()]
    except (AttributeError, TypeError, ValueError):
        names = []
    if len(names) != width - 1:
        names = [f"phi{j}" for j in range(1, width)]
    return ("1", *names)
