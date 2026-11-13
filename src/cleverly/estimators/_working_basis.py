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
``phi``. :class:`WorkingDictionary` adds the constant, builds ``psi``, drops the columns
with too little support among the rows that identify them (:func:`supported_columns`) and
drops exactly collinear columns with a pivoted QR on the selection rows
(:func:`independent_columns`). A declared transformer replaces :class:`WorkingBasis` and
goes through the same steps.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.linalg
from sklearn.base import BaseEstimator, TransformerMixin, clone

from .._typing import BoolArray, FloatArray, IntArray
from ..learners._fitting import seed_unset

__all__ = [
    "COLLINEARITY_TOLERANCE",
    "CONTINUOUS_LEVELS",
    "WorkingBasis",
    "WorkingDictionary",
    "bias_columns",
    "independent_columns",
    "minimum_support",
    "supported_columns",
]

#: The relative size below which a column's part outside the span of the columns kept
#: before it counts as zero, so that the column is dropped as exactly collinear.
COLLINEARITY_TOLERANCE = 1e-10

#: The fewest distinct values on the fit rows for which a covariate gets a linear term and
#: hinge columns rather than level indicators.
CONTINUOUS_LEVELS = 9


class WorkingBasis(TransformerMixin, BaseEstimator):
    """The default dictionary of the working models, without its constant.

    A covariate with at least :data:`CONTINUOUS_LEVELS` distinct values on the fit rows gets
    its standardized linear term ``z = (x - mean) / sd`` and the first-order hinges
    ``(z - k_j)_+`` at ``n_knots`` interior quantiles of ``z`` on the fit rows. The linear
    function is then one column, and each hinge is exactly zero below its knot. A covariate
    with fewer distinct values gets one indicator column for each level other than the
    smallest. Each pair of covariates then gets the product of their standardized values. A
    covariate constant on the fit rows contributes no column.

    Parameters
    ----------
    n_knots : int, default=5
        The number of interior quantile knots of each continuous covariate.
    interactions : bool, default=True
        Whether to add the pairwise products.

    Attributes
    ----------
    unscreened_ : ndarray of bool
        Marks the linear terms, which the support screen of :class:`WorkingDictionary`
        always keeps.

    See Also
    --------
    WorkingDictionary : Adds the constant and the bias columns, and drops unsupported and
        collinear columns.
    """

    def __init__(self, n_knots: int = 5, interactions: bool = True) -> None:
        self.n_knots = n_knots
        self.interactions = interactions

    def fit(self, X: Any, y: Any = None) -> WorkingBasis:
        """Learn the scales, knots and levels on the fit rows.

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
        self.mean_ = matrix.mean(axis=0)
        self.scale_ = matrix.std(axis=0)
        self.knots_: dict[int, FloatArray] = {}
        self.levels_: dict[int, FloatArray] = {}
        probabilities = np.arange(1, self.n_knots + 1) / (self.n_knots + 1)
        unscreened: list[bool] = []
        for j in range(matrix.shape[1]):
            values = np.unique(matrix[:, j])
            if values.size >= CONTINUOUS_LEVELS:
                z = (matrix[:, j] - self.mean_[j]) / self.scale_[j]
                self.knots_[j] = np.unique(np.quantile(z, probabilities))
                unscreened.extend([True] + [False] * self.knots_[j].size)
            else:
                self.levels_[j] = values[1:]
                unscreened.extend([False] * self.levels_[j].size)
        varying = [j for j in range(matrix.shape[1]) if self.scale_[j] > 0.0]
        self.pairs_ = (
            [(j, k) for i, j in enumerate(varying) for k in varying[i + 1 :]]
            if self.interactions
            else []
        )
        unscreened.extend([False] * len(self.pairs_))
        self.unscreened_ = np.asarray(unscreened, dtype=bool)
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
        standardized = (matrix - self.mean_) / np.where(self.scale_ > 0.0, self.scale_, 1.0)
        blocks: list[FloatArray] = []
        for j in range(matrix.shape[1]):
            if j in self.knots_:
                z = standardized[:, [j]]
                blocks.append(np.column_stack([z, np.maximum(z - self.knots_[j][None, :], 0.0)]))
            else:
                levels = self.levels_[j]
                blocks.append((matrix[:, [j]] == levels[None, :]).astype(float))
        for j, k in self.pairs_:
            blocks.append((standardized[:, j] * standardized[:, k])[:, None])
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
            if j in self.knots_:
                out.append(names[j])
                out.extend(f"{names[j]}_hinge{b}" for b in range(self.knots_[j].size))
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


def minimum_support(rows: int) -> int:
    """``k_min = max(10, ceil(sqrt(n) / 2))``, the support a column needs on ``n`` rows.

    Parameters
    ----------
    rows : int
        The rows the working model is selected on.

    Returns
    -------
    int
        The fewest nonzero rows a dictionary column needs in each group of
        :func:`supported_columns`.
    """
    return max(10, int(np.ceil(np.sqrt(rows) / 2.0)))


def supported_columns(
    matrix: FloatArray,
    groups: tuple[BoolArray, BoolArray],
    arm_columns: BoolArray,
    arm_rows: BoolArray,
    fixed: BoolArray,
    k_min: int,
) -> IntArray:
    """The columns that are nonzero on at least ``k_min`` rows of each of two groups.

    For a column marked in ``arm_columns`` only the rows marked in ``arm_rows`` count.
    Columns marked in ``fixed`` are always kept.

    Parameters
    ----------
    matrix : ndarray of float
        The ``(n, p)`` columns on the selection rows.
    groups : tuple of ndarray of bool
        The two row groups, such as the external and the trial rows.
    arm_columns : ndarray of bool
        The columns that live in one arm (the ``A`` columns of the bias dictionary).
    arm_rows : ndarray of bool
        The rows of that arm.
    fixed : ndarray of bool
        The columns that are always kept.
    k_min : int
        The fewest nonzero rows a column needs in each group.

    Returns
    -------
    ndarray of int
        The kept column positions, ascending.
    """
    nonzero = np.asarray(matrix, dtype=float) != 0.0
    everywhere = np.ones(matrix.shape[0], dtype=bool)
    kept = [
        j
        for j in range(matrix.shape[1])
        if fixed[j]
        or all(
            int(np.sum(nonzero[(arm_rows if arm_columns[j] else everywhere) & group, j])) >= k_min
            for group in groups
        )
    ]
    return np.asarray(kept, dtype=np.int64)


@dataclass(frozen=True)
class WorkingDictionary:
    """The two working-model dictionaries of one selection, fitted on its selection rows.

    Each dictionary is screened for support (:func:`supported_columns`) and then reduced to
    linearly independent columns (:func:`independent_columns`). An effect column must be
    nonzero on ``k_min`` treated and ``k_min`` control rows. A bias column must be nonzero on
    ``k_min`` external and ``k_min`` trial rows, counting only the treated rows for a column
    that carries ``A``. ``k_min`` is :func:`minimum_support` of the rows the model is selected
    on. A column nonzero on a handful of rows of one group cannot identify its coefficient
    there. The constant, ``A``, and the columns the basis marks in an ``unscreened_``
    attribute (the linear terms of :class:`WorkingBasis`) and their ``A`` products are always
    kept.

    Parameters
    ----------
    basis : Any
        The fitted transformer of ``W`` into the non-constant columns of ``phi``.
    phi_keep : ndarray of int
        The positions of ``[1, basis(W)]`` that were kept. Position 0 is the constant.
    psi_keep : ndarray of int
        The positions of :func:`bias_columns` that were kept. Position 0 is the constant
        and, with external rows in both arms, position 1 is ``A``.
    controls_only : bool
        Whether the external data hold controls only.
    names : tuple of str
        The names of the columns of ``[1, basis(W)]``.
    phi_unsupported : ndarray of int
        The positions of ``[1, basis(W)]`` that the support screen dropped.
    psi_unsupported : ndarray of int
        The positions of :func:`bias_columns` that the support screen dropped.
    """

    basis: Any
    phi_keep: IntArray
    psi_keep: IntArray
    controls_only: bool
    names: tuple[str, ...]
    phi_unsupported: IntArray
    psi_unsupported: IntArray

    @classmethod
    def fit(
        cls,
        basis: Any,
        covariates: FloatArray,
        treatment: FloatArray,
        source: FloatArray,
        bias_rows: IntArray,
        *,
        controls_only: bool,
        covariate_names: Sequence[str] | None = None,
        random_state: int | None = None,
    ) -> WorkingDictionary:
        """Fit a clone of ``basis`` on the selection rows and keep the supported columns.

        Parameters
        ----------
        basis : Any
            A scikit-learn transformer of ``W``, or ``None`` for :class:`WorkingBasis`.
        covariates : ndarray of float
            ``W`` on the selection rows.
        treatment : ndarray of float
            ``A`` on the selection rows.
        source : ndarray of float
            ``S`` on the selection rows.
        bias_rows : ndarray of int
            The positions, within the selection rows, that the bias model is selected on.
        controls_only : bool
            Whether the external data hold controls only.
        covariate_names : sequence of str or None
            The names of the columns of ``covariates``, which name the dictionary's columns.
            ``None`` names them ``x0``, ``x1``, and so on.
        random_state : int or None
            The fit's seed, which every ``random_state`` of ``basis`` that is ``None`` takes,
            so a stochastic basis (random features) fits the same way on every run.
            ``None`` leaves ``basis`` as it is.

        Returns
        -------
        WorkingDictionary
            The fitted dictionaries.
        """
        fitted = clone(WorkingBasis() if basis is None else seed_unset(basis, random_state)).fit(
            covariates
        )
        raw = _with_constant(fitted, covariates)
        marked = getattr(fitted, "unscreened_", None)
        unscreened = np.r_[
            True,
            np.zeros(raw.shape[1] - 1, dtype=bool) if marked is None else np.asarray(marked, bool),
        ]
        treated = np.asarray(treatment, dtype=float) == 1.0
        phi_supported = supported_columns(
            raw,
            (treated, ~treated),
            np.zeros(raw.shape[1], dtype=bool),
            treated,
            unscreened,
            minimum_support(raw.shape[0]),
        )
        phi_keep = phi_supported[independent_columns(raw[:, phi_supported], 1)]
        phi = raw[:, phi_keep]
        bias = bias_columns(phi[bias_rows], treatment[bias_rows], controls_only)
        rest = unscreened[phi_keep][1:]
        if controls_only:
            fixed = np.r_[True, rest]
            arm_columns = np.zeros(bias.shape[1], dtype=bool)
        else:
            fixed = np.r_[True, True, rest, rest]
            arm_columns = np.r_[False, True, np.zeros(rest.size, bool), np.ones(rest.size, bool)]
        external = np.asarray(source, dtype=float)[bias_rows] == 0.0
        psi_supported = supported_columns(
            bias,
            (external, ~external),
            arm_columns,
            treated[bias_rows],
            fixed,
            minimum_support(bias.shape[0]),
        )
        psi_keep = psi_supported[
            independent_columns(bias[:, psi_supported], 1 if controls_only else 2)
        ]
        return cls(
            fitted,
            phi_keep,
            psi_keep,
            controls_only,
            _names(fitted, raw.shape[1], covariate_names),
            np.setdiff1d(np.arange(raw.shape[1]), phi_supported),
            np.setdiff1d(np.arange(bias.shape[1]), psi_supported),
        )

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
        raw = self._bias_names()
        return tuple(raw[j] for j in self.psi_keep)

    @property
    def psi_arm(self) -> BoolArray:
        """Marks the kept bias columns that carry ``A``: ``A`` and its products.

        All false when the external data are controls only.
        """
        if self.controls_only:
            return np.zeros(self.psi_keep.size, dtype=bool)
        return np.asarray((self.psi_keep == 1) | (self.psi_keep >= 1 + self.phi_keep.size))

    @property
    def unsupported_phi_names(self) -> tuple[str, ...]:
        """The names of the effect columns that the support screen dropped."""
        return tuple(self.names[j] for j in self.phi_unsupported)

    @property
    def unsupported_psi_names(self) -> tuple[str, ...]:
        """The names of the bias columns that the support screen dropped."""
        raw = self._bias_names()
        return tuple(raw[j] for j in self.psi_unsupported)

    def _bias_names(self) -> list[str]:
        rest = [self.names[j] for j in self.phi_keep[1:]]
        if self.controls_only:
            return ["1", *rest]
        return ["1", "A", *rest, *(f"A*{name}" for name in rest)]


def _with_constant(basis: Any, covariates: FloatArray) -> FloatArray:
    columns = np.asarray(basis.transform(covariates), dtype=float)
    if columns.ndim == 1:
        columns = columns[:, None]
    return np.column_stack([np.ones(columns.shape[0]), columns])


def _names(basis: Any, width: int, covariate_names: Sequence[str] | None) -> tuple[str, ...]:
    names: list[str] = []
    tries: list[list[str] | None] = [None]
    if covariate_names is not None:
        tries.insert(0, [str(name) for name in covariate_names])
    for given in tries:
        try:
            names = [str(name) for name in basis.get_feature_names_out(given)]
            break
        except (AttributeError, TypeError, ValueError):
            names = []
    if len(names) != width - 1:
        names = [f"phi{j}" for j in range(1, width)]
    return ("1", *names)
