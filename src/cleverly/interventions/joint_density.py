r"""The density a joint policy induces at a vector treatment with a continuous component.

A vector node :math:`A = (C, D)` holds categorical components :math:`C` and doses
:math:`D = (D_1, \ldots, D_K)`.  Its law factors by the chain rule, categorical components
first and then the doses in column order:

.. math::

    g(c, d \mid h) = g_C(c \mid h) \prod_k g_k(d_k \mid d_{<k}, c, h).

:math:`g_C` is one multinomial over the observed cells of the categorical components, and
each :math:`g_k` is a pooled-hazard density
(:func:`~cleverly.learners.density.fit_conditional_density`) whose design holds the history,
the categorical components' indicators and the earlier doses.  Any factorization gives the
same joint density, so the column order serves every policy.

A :class:`~cleverly.interventions.JointPolicy` maps the categorical components by label maps
that read every natural value, and each dose by a piecewise strictly monotone map that reads
the natural categorical values and the natural doses before it in the policy's ``order``.
For a fixed natural cell :math:`c` the dose map :math:`T_c` is then triangular, its inverse
is triangular, and the determinant of the inverse's Jacobian is the product of the declared
derivatives.  The density the policy induces at :math:`(c', b)` is

.. math::

    g^d(c', b \mid h) = \sum_c \sum_{j_1, \ldots, j_K}
        \mathbb 1\{d^C(c, \beta, h) = c'\}\, \mathbb 1\{\beta_k \in I_{j_k}(c, \beta_{<k}, h)\}
        \, g(c, \beta \mid h) \prod_k \Bigl|\frac{\partial \beta_k}{\partial b_k}\Bigr|,

with :math:`\beta = \beta(c, b)` the preimage of :math:`b` along the pieces :math:`j`,
computed one dose at a time in ``order``.  This is the chain rule, a finite sum over the
categorical preimages, and Equation (3) of Díaz, Williams, Hoffman and Schenck (2023) applied
to each conditional factor.  A randomizer mixes the branches with their known probabilities.
The node ratio is :math:`g^d / g` at the observed value.

**Every piece is explicit.**  A scalar branch adds an identity piece as a ratio of exactly
one.  That shortcut is wrong here: the identity piece of dose :math:`k` contributes
:math:`g(c, \ldots, b_k, \ldots) / g(c', b)`, which is not one when the categorical cell or
an earlier dose moved.  So a ``cap`` that holds a row, an identity :class:`Piece` and the
uncovered set of a :class:`Piecewise` are each a piece with the identity inverse and
derivative one.

**The evaluation reads counterfactual conditioning designs.**  The density of dose :math:`k`
is evaluated at :math:`(c, \beta_{<k}, h)` for every candidate cell and every preimage, and
these are not the row's observed design.  The fit predicts them out of fold, by the model of
the fold that held the row out, and keeps the density values only
(:func:`~cleverly.learners.density.fit_conditional_density`'s ``predict_designs``).

**The checks read the same frames.**  Each declared dose map is checked for a partition of
the preimages it counts, for strict monotonicity, for its inverse and for its derivative at
every frame the formula evaluates it on.  Each categorical map is checked to return a level
there.  These frames hold a candidate cell and the preimages of the earlier doses, which the
check of the observed frames does not reach.

The remainder of the estimator is the product of the ratio error and the outcome
regression's error, as for a scalar continuous node.  No regression is fitted on an
estimated index: :math:`\beta` is a known function of observed values.  The ratio error
combines the errors of :math:`K + 1` factors, and the density of dose :math:`k` conditions on
:math:`k - 1` continuous doses, so its learner's rate degrades with :math:`k`.
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .._typing import BoolArray, FloatArray, IntArray, Learner
from ..data.validate import MAX_TREATMENT_LEVELS, arm_indicators
from ..exceptions import CapabilityError, DataError, PositivityWarning
from ..learners.crossfit import Folds
from ..learners.density import (
    ConditionalDensity,
    DensityDiagnostics,
    PredictionMemory,
    bin_edges,
    default_density_bins,
    fit_conditional_density,
    preflight_conditional_densities,
)
from ..learners.super_learner import SuperLearnerDiagnostics
from .policy import (
    FrameSource,
    FrameWith,
    JointComponent,
    ModifiedPolicy,
    Piecewise,
    Scale,
    Shift,
    _call,
    _categorical_component,
    _check_monotone,
    _is_identity_map,
    _member,
    _per_row,
    _PieceBranch,
    dose_order,
    guarded_source,
)

__all__ = [
    "ExplicitPiece",
    "JointBranch",
    "JointDensityParts",
    "JointEvaluation",
    "NodeLayout",
    "bound_cells",
    "evaluate_joint_density",
    "explicit_pieces",
]


# ------------------------------------------------------------------ explicit pieces


@dataclass(frozen=True)
class ExplicitPiece:
    """One piece of a dose map as the joint evaluator reads it.

    ``inverse`` and ``derivative`` are ``None`` on an identity piece, whose inverse is the
    identity and whose derivative is one.  ``member`` says which values lie in the piece,
    given the frame.  ``declared`` holds the :class:`~cleverly.interventions.ModifiedPolicy`
    pieces a check reads, or ``None`` for a closed-form policy, which is monotone by
    construction.

    Parameters
    ----------
    member : callable
        ``member(values, frame)``, a boolean per row.
    inverse : callable or None
        ``inverse(b, frame)``, the value the piece's map sends to ``b``.
    derivative : callable or None
        ``derivative(b, frame)``, the derivative of the inverse.
    """

    member: Callable[[FloatArray, FrameSource], BoolArray]
    inverse: Callable[[FloatArray, FrameSource], FloatArray] | None = None
    derivative: Callable[[FloatArray, FrameSource], FloatArray] | None = None

    @property
    def identity(self) -> bool:
        """Whether the piece leaves its values unchanged."""
        return self.inverse is None


def _finite(values: FloatArray) -> BoolArray:
    return np.asarray(np.isfinite(values), dtype=bool)


_EVERYWHERE = ExplicitPiece(lambda values, frame: _finite(values))


def _capped(
    forward: Callable[[FloatArray], FloatArray],
    inverse: Callable[[FloatArray], FloatArray],
    slope: float,
    cap: float | None,
) -> tuple[ExplicitPiece, ...]:
    """A closed-form map with an optional cap, as a moved piece and a held piece.

    The held set is the one the policy's ``apply`` holds, ``forward(a) > cap``, written with
    the same arithmetic so the two cannot disagree at the cap.
    """

    def held(values: FloatArray) -> BoolArray:
        if cap is None:
            return np.zeros(values.shape, dtype=bool)
        with np.errstate(invalid="ignore"):
            return np.asarray(forward(values) > float(cap), dtype=bool)

    moved = ExplicitPiece(
        member=lambda values, frame: _finite(values) & ~held(values),
        inverse=lambda b, frame: np.asarray(inverse(b), dtype=float),
        derivative=lambda b, frame: np.full(b.shape, slope),
    )
    if cap is None:
        return (moved,)
    stay = ExplicitPiece(member=lambda values, frame: _finite(values) & held(values))
    return (moved, stay)


def _closed_form(mapping: Shift | Scale) -> tuple[Callable[..., Any], Callable[..., Any], float]:
    """``(forward, inverse, derivative of the inverse)`` of an uncapped shift or scale."""
    if isinstance(mapping, Shift):
        delta = float(mapping.delta)
        return (lambda a: a + delta), (lambda b: b - delta), 1.0
    factor = float(mapping.factor)
    return (lambda a: a * factor), (lambda b: b / factor), 1.0 / factor


def _declared_piece(branch: _PieceBranch, index: int) -> ExplicitPiece:
    """One :class:`Piece` of a declared policy, with the identity piece made explicit."""
    piece = branch.pieces[index]

    def member(values: FloatArray, frame: FrameSource) -> BoolArray:
        return _member(values, *branch.bounds(piece, frame, values.size), piece.closed)

    if piece.map is None:
        return ExplicitPiece(member)
    inverse, derivative = piece.inverse, piece.derivative
    assert inverse is not None and derivative is not None

    def invert(b: FloatArray, frame: FrameSource) -> FloatArray:
        with np.errstate(all="ignore"):
            return _per_row(_call(inverse, b, frame()), b.size)

    def slope(b: FloatArray, frame: FrameSource) -> FloatArray:
        with np.errstate(all="ignore"):
            return _per_row(_call(derivative, b, frame()), b.size)

    return ExplicitPiece(member, invert, slope)


def _moved_piece(
    member: Callable[[FloatArray, FrameSource], BoolArray],
    inverse: Callable[[FloatArray], FloatArray],
    slope: float,
) -> ExplicitPiece:
    """A closed-form moved piece with its declared membership."""

    def invert(b: FloatArray, frame: FrameSource) -> FloatArray:
        return np.asarray(inverse(b), dtype=float)

    def derivative(b: FloatArray, frame: FrameSource) -> FloatArray:
        return np.full(b.shape, slope)

    return ExplicitPiece(member, invert, derivative)


def explicit_pieces(policy: object | None) -> tuple[ExplicitPiece, ...]:
    """The pieces of one dose's policy, every identity piece and held set explicit.

    Parameters
    ----------
    policy : object or None
        A :class:`Shift`, :class:`Scale`, :class:`Piecewise` or :class:`ModifiedPolicy` with
        pieces, or ``None`` for a dose the policy leaves unchanged.

    Returns
    -------
    tuple of ExplicitPiece
        Pieces that partition the dose range.
    """
    if policy is None:
        return (_EVERYWHERE,)
    if isinstance(policy, (Shift, Scale)):
        if _is_identity_map(policy):
            return (_EVERYWHERE,)
        forward, inverse, slope = _closed_form(policy)
        return _capped(forward, inverse, slope, policy.cap)
    if isinstance(policy, Piecewise):
        out: list[ExplicitPiece] = []
        closed = policy.closed
        for lower, upper, mapping in policy.pieces:

            def member(
                values: FloatArray, frame: FrameSource, lower: float = lower, upper: float = upper
            ) -> BoolArray:
                return _member(values, lower, upper, closed)

            if _is_identity_map(mapping):
                out.append(ExplicitPiece(member))
                continue
            _, inverse, slope = _closed_form(mapping)
            out.append(_moved_piece(member, inverse, slope))
        bounds = tuple((lower, upper) for lower, upper, _ in policy.pieces)

        def uncovered(values: FloatArray, frame: FrameSource) -> BoolArray:
            covered = np.zeros(values.shape, dtype=bool)
            for lower, upper in bounds:
                covered |= _member(values, lower, upper, closed)
            return _finite(values) & ~covered

        out.append(ExplicitPiece(uncovered))
        return tuple(out)
    if isinstance(policy, ModifiedPolicy) and policy.pieces is not None:
        assert not isinstance(policy.pieces, Mapping)
        branch = _PieceBranch(policy.name, policy.pieces)
        return tuple(_declared_piece(branch, index) for index in range(len(branch.pieces)))
    raise CapabilityError(
        f"policy {getattr(policy, 'name', policy)!r} has no density ratio on a dose: declare "
        "a Shift, Scale, Piecewise or ModifiedPolicy with pieces"
    )


# ------------------------------------------------------------------ the node


@dataclass(frozen=True)
class NodeLayout:
    """A vector node, as the joint evaluator reads it.

    Parameters
    ----------
    components : tuple of JointComponent
        The node's components, in column order.
    natural : ndarray
        ``(n, d)`` natural values: level codes and doses, zero where a row holds no value.
    rows : ndarray
        ``(n,)`` the rows that hold the node.
    frame_with : callable
        Builds the node's policy frame with extra columns.
    conditioning : ndarray
        ``(n, p)`` the history the node's law conditions on.
    where : str
        The node, for the refusals.
    """

    components: tuple[JointComponent, ...]
    natural: FloatArray
    rows: BoolArray
    frame_with: FrameWith
    conditioning: FloatArray
    where: str

    @property
    def n(self) -> int:
        """Return the number of rows."""
        return int(self.natural.shape[0])

    @property
    def categorical(self) -> tuple[int, ...]:
        """The column indices of the categorical components."""
        return tuple(i for i, component in enumerate(self.components) if not component.continuous)

    @property
    def doses(self) -> tuple[int, ...]:
        """The column indices of the doses, in column order."""
        return tuple(i for i, component in enumerate(self.components) if component.continuous)

    @property
    def names(self) -> tuple[str, ...]:
        """The component names."""
        return tuple(component.name for component in self.components)

    def cells(self) -> tuple[tuple[int, ...], ...]:
        """The observed cells of the categorical components, sorted.

        Returns
        -------
        tuple of tuple of int
            Each cell's level codes, one per categorical component.

        Raises
        ------
        CapabilityError
            If more cells are observed than a categorical treatment may hold.
        """
        codes = self.natural[np.ix_(self.rows, self.categorical)].astype(np.int64)
        cells = tuple(sorted({tuple(int(v) for v in row) for row in codes})) or ((),)
        if len(cells) > MAX_TREATMENT_LEVELS:
            raise CapabilityError(
                f"{self.where} holds {len(cells)} observed cells of its categorical "
                f"components, and a categorical treatment holds at most "
                f"{MAX_TREATMENT_LEVELS} levels. The density of the node sums over these "
                "cells; collapse a categorical component"
            )
        return cells

    def cell_codes(self, codes: FloatArray, cells: Sequence[tuple[int, ...]]) -> IntArray:
        """Each row's cell index for ``(n, n_categorical)`` level codes, ``-1`` off the cells.

        Parameters
        ----------
        codes : ndarray
            ``(n, n_categorical)`` level codes.
        cells : sequence of tuple of int
            The cells, from :meth:`cells`.

        Returns
        -------
        ndarray
            ``(n,)`` cell indices.
        """
        lookup = {cell: index for index, cell in enumerate(cells)}
        values = np.asarray(codes, dtype=float).reshape(self.n, -1)
        out = np.full(self.n, -1, dtype=np.int64)
        for row in range(self.n):
            out[row] = lookup.get(tuple(int(v) for v in values[row]), -1)
        return out

    def cell_block(self, codes: FloatArray) -> FloatArray:
        """The categorical components' drop-first indicators, as the node's design reads them.

        Parameters
        ----------
        codes : ndarray
            ``(n, n_categorical)`` level codes.

        Returns
        -------
        ndarray
            ``(n, q)`` indicators.
        """
        values = np.asarray(codes, dtype=float).reshape(self.n, -1)
        blocks = [
            arm_indicators(values[:, j], len(self.components[index].levels))
            for j, index in enumerate(self.categorical)
        ]
        return np.hstack(blocks) if blocks else np.zeros((self.n, 0))

    def dose_design(self, position: int, codes: FloatArray, doses: FloatArray) -> FloatArray:
        """The design of the density of the dose at ``position``: history, cell, earlier doses.

        Parameters
        ----------
        position : int
            The dose's position among the doses, in column order.
        codes : ndarray
            ``(n, n_categorical)`` level codes the design conditions on.
        doses : ndarray
            ``(n, K)`` dose values, of which the first ``position`` are read.

        Returns
        -------
        ndarray
            The design.
        """
        return np.column_stack(
            [self.conditioning, self.cell_block(codes), np.asarray(doses)[:, :position]]
        )

    def labels(self, index: int, values: FloatArray) -> Any:
        """The natural values of one component, as a map reads them.

        Parameters
        ----------
        index : int
            The component's column index.
        values : ndarray
            ``(n,)`` its codes or doses.

        Returns
        -------
        object
            Labels or doses.
        """
        return self.components[index].labels(values)


@dataclass(frozen=True)
class JointBranch:
    """One deterministic branch of a joint policy, as the evaluator reads it.

    Parameters
    ----------
    name : str
        The policy's name.
    declared : mapping
        The component column mapped to its policy, for this branch.
    order : tuple of str or None
        The policy's declared dose order.
    """

    name: str
    declared: Mapping[str, Any]
    order: tuple[str, ...] | None = None


@dataclass(frozen=True)
class JointEvaluation:
    """A value of the node at which the induced and the observed density are evaluated.

    Parameters
    ----------
    values : ndarray
        ``(n, d)`` level codes and doses.
    branches : tuple of int
        The branches whose induced density is evaluated here; empty reads the observed
        density alone.
    """

    values: FloatArray
    branches: tuple[int, ...] = ()


@dataclass(frozen=True)
class JointDensityParts:
    """The induced and the observed density of a vector node, by factor, at each value.

    Kept apart because the categorical factor is bounded where the targeting step bounds a
    categorical mechanism, and the doses' factors are not.  The ratio of branch ``b`` at
    evaluation ``e`` is :meth:`ratio`.

    Parameters
    ----------
    cells : tuple of tuple of int
        The observed cells of the categorical components.
    cell_probabilities : ndarray
        ``(n, K)`` the out-of-fold :math:`g_C(c \\mid h)`, untruncated.
    induced : ndarray
        ``(n, E, B, K)``: at evaluation ``e`` and branch ``b``, the doses' part of the
        induced density from candidate cell ``c``.  The induced density is
        :math:`\\sum_c g_C(c \\mid h)` times this.  Zero where a branch is not evaluated.
    observed : ndarray
        ``(n, E)`` :math:`\\prod_k g_k` at each evaluation's value.
    targets : ndarray
        ``(n, E)`` the cell index of each evaluation's categorical values, ``-1`` off the
        cells.
    densities : dict of str to ConditionalDensity
        Each dose's density at its observed design, for the support report.
    diagnostics : dict of str to DensityDiagnostics
        Each dose's density diagnostics.
    cell_diagnostics : tuple of SuperLearnerDiagnostics
        The categorical factor's learner diagnostics, one per fitted fold.
    """

    cells: tuple[tuple[int, ...], ...]
    cell_probabilities: FloatArray
    induced: FloatArray
    observed: FloatArray
    targets: IntArray
    densities: dict[str, ConditionalDensity]
    diagnostics: dict[str, DensityDiagnostics]
    cell_diagnostics: tuple[SuperLearnerDiagnostics, ...] = ()

    def joint(self, cell_probabilities: FloatArray | None = None) -> FloatArray:
        """``(n, E)`` :math:`g(c', b \\mid h)` at each evaluation's value.

        Parameters
        ----------
        cell_probabilities : ndarray or None
            The categorical factor to use, such as a bounded one; ``None`` reads the fit.

        Returns
        -------
        ndarray
            The observed density, zero where the categorical values are not a cell.
        """
        g = self.cell_probabilities if cell_probabilities is None else cell_probabilities
        rows = np.arange(self.observed.shape[0])[:, None]
        safe = np.where(self.targets >= 0, self.targets, 0)
        cell = np.where(self.targets >= 0, g[rows, safe], 0.0)
        return np.asarray(cell * self.observed, dtype=float)

    def ratio(self, cell_probabilities: FloatArray | None = None) -> FloatArray:
        """``(n, E, B)`` :math:`g^{d_b}(v_e \\mid h) / g(v_e \\mid h)`, zero where ``g`` is zero.

        Parameters
        ----------
        cell_probabilities : ndarray or None
            The categorical factor to use in the numerator and the denominator, such as a
            bounded one; ``None`` reads the fit.

        Returns
        -------
        ndarray
            The ratio of every branch at every evaluation.
        """
        g = self.cell_probabilities if cell_probabilities is None else cell_probabilities
        numerator = np.einsum("iebk,ik->ieb", self.induced, g)
        denominator = self.joint(g)[:, :, None]
        safe = np.where(denominator > 0.0, denominator, 1.0)
        return np.asarray(np.where(denominator > 0.0, numerator / safe, 0.0), dtype=float)

    def subset(self, index: Any) -> JointDensityParts:
        """The same parts on a row subset.

        Parameters
        ----------
        index : array_like
            Row positions or a boolean mask.

        Returns
        -------
        JointDensityParts
            The parts over the selected rows.
        """
        idx = np.asarray(index)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        return replace(
            self,
            cell_probabilities=self.cell_probabilities[idx],
            induced=self.induced[idx],
            observed=self.observed[idx],
            targets=self.targets[idx],
            densities={name: density.subset(idx) for name, density in self.densities.items()},
        )


def bound_cells(probabilities: FloatArray, bounds: tuple[float, float]) -> FloatArray:
    """The categorical factor truncated into ``bounds``, by the categorical treatment's rule.

    Two cells take the complement form: the second is clipped and the first is one minus
    it, as :meth:`~cleverly.estimators._nuisance.Propensity.truncate` does on the simplex.
    More cells are clipped one by one and are not renormalised.  A single cell is one and
    is not bounded.

    Parameters
    ----------
    probabilities : ndarray
        ``(n, K)`` cell probabilities.
    bounds : tuple of float
        ``(lower, upper)``.

    Returns
    -------
    ndarray
        The bounded probabilities.
    """
    lower, upper = float(bounds[0]), float(bounds[1])
    g = np.asarray(probabilities, dtype=float)
    if g.shape[1] == 1:
        return g
    if g.shape[1] == 2:
        one = np.clip(g[:, 1], lower, upper)
        return np.column_stack([1.0 - one, one])
    return np.asarray(np.clip(g, lower, upper), dtype=float)


# ------------------------------------------------------------------ the evaluation


@dataclass(frozen=True)
class _Design:
    """A conditioning design of one dose's density, kept as the columns it changes.

    The design is :meth:`NodeLayout.dose_design`: the history, the categorical indicators
    and the earlier doses.  The history is the fitted design's, so only the level codes and
    the earlier doses are stored, and a fold builds its held-out rows on demand
    (:class:`~cleverly.learners.density.DesignRows`).  A candidate cell's codes are stored as
    one row, which every held-out row repeats, so a design holds no ``(n, q)`` array.
    """

    width: int
    levels: tuple[int, ...]
    codes: FloatArray
    doses: FloatArray

    @property
    def nbytes(self) -> int:
        """Bytes the design stores."""
        return int(self.codes.nbytes + self.doses.nbytes)

    def __call__(self, index: IntArray, fitted: FloatArray) -> FloatArray:
        if self.codes.shape[0] == self.doses.shape[0]:
            codes = self.codes[index]
        else:
            codes = np.repeat(self.codes, len(index), axis=0)
        blocks = [arm_indicators(codes[:, j], count) for j, count in enumerate(self.levels)]
        return np.column_stack([fitted[:, : self.width], *blocks, self.doses[index]])


def _digest(values: FloatArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=float).tobytes()).hexdigest()


@dataclass
class _Requests:
    """The conditioning designs and evaluation values each dose's density is asked for.

    A design is keyed by its content: the candidate cell (or a digest of per-row codes)
    and a digest of the earlier doses.  Two terms that condition on the same values share
    one design, whichever cell, piece path, branch or evaluation asked for it.  A digest
    match is confirmed by comparing the stored arrays.
    """

    layout: NodeLayout
    designs: list[dict[Hashable, _Design]]
    values: list[dict[Hashable, list[FloatArray]]]

    def ask(
        self,
        position: int,
        codes: FloatArray,
        doses: FloatArray,
        at: FloatArray,
        cell: tuple[int, ...] | None = None,
    ) -> tuple[int, Hashable, int]:
        layout = self.layout
        earlier = np.asarray(doses, dtype=float)[:, :position]
        if cell is None:
            stored = np.asarray(codes, dtype=float).reshape(layout.n, -1)
            identity: Hashable = ("codes", _digest(stored))
        else:
            stored = np.asarray(cell, dtype=float).reshape(1, -1)
            identity = ("cell", tuple(cell))
        key: Hashable = (identity, _digest(earlier))
        designs = self.designs[position]
        while key in designs and not (
            np.array_equal(designs[key].codes, stored)
            and np.array_equal(designs[key].doses, earlier, equal_nan=True)
        ):
            key = (key, "collision")
        if key not in designs:
            designs[key] = _Design(
                layout.conditioning.shape[1],
                tuple(len(layout.components[index].levels) for index in layout.categorical),
                stored,
                np.array(earlier, dtype=float),
            )
            self.values[position][key] = []
        self.values[position][key].append(np.asarray(at, dtype=float))
        return position, key, len(self.values[position][key]) - 1

    def memory(self, position: int, n_bins: int, folds: Folds) -> PredictionMemory:
        """What the designs and values of the dose at ``position`` hold while it is fitted."""
        n = self.layout.n
        stored = sum(design.nbytes for design in self.designs[position].values())
        values = sum(8 * n * len(columns) for columns in self.values[position].values())
        # The predictions are allocated up front and, across folds, also arrive at once.
        predictions = values if folds.is_single else 2 * values
        held_out = n if folds.is_single else max(int(test.size) for _, test in folds)
        width = self.layout.conditioning.shape[1] + sum(
            max(len(self.layout.components[index].levels) - 1, 1)
            for index in self.layout.categorical
        )
        rows = 8 * held_out * (width + position + 2 * n_bins)
        return PredictionMemory(
            held=stored + values + predictions,
            shipped=stored + values,
            rows=rows,
            kept=values,
        )


@dataclass(frozen=True)
class _Term:
    """One candidate cell and piece path: its weight and the factors it multiplies."""

    cell: int
    weight: FloatArray
    factors: tuple[tuple[int, Hashable, int], ...]


def _check_preimages(
    name: str,
    pieces: Sequence[ExplicitPiece],
    preimages: Sequence[tuple[FloatArray, BoolArray]],
    frame: FrameSource,
    rows: BoolArray,
    where: str,
) -> None:
    """Every preimage the formula counts lies in exactly one piece of the dose's map."""
    for value, inside in preimages:
        count = np.zeros(value.size, dtype=np.int64)
        for piece in pieces:
            count += piece.member(value, frame).astype(np.int64)
        wrong = rows & inside & (count != 1)
        if wrong.any():
            first = float(value[np.flatnonzero(wrong)[0]])
            raise DataError(
                f"policy {name!r} at {where}: the pieces do not partition the dose at "
                f"{int(wrong.sum())} preimage(s) the density ratio counts (first at {first:g}), "
                "on a frame that holds a candidate categorical cell or the preimages of the "
                "earlier doses. Declare pieces that partition the dose range at every value of "
                "the components they read, as Assumption 4 of Díaz et al. (2023) requires"
            )


#: How the declaration checks read a counterfactual frame, appended to their refusals.
_COUNTERFACTUAL_FRAMES = (
    ". The check reads each frame the density ratio evaluates, which holds a candidate "
    "categorical cell or the preimages of the earlier doses, over the observed range of the "
    "dose, because the estimated density can be positive anywhere in that range before any "
    "learner runs. A map that is monotone only on a narrower true support of the dose on "
    "such a frame is refused: the check is conservative there"
)

#: The most observed values of a dose that the coverage check reads at each frame.
_COVERAGE_POINTS = 64


def _coverage_points(values: FloatArray) -> FloatArray:
    """Observed values of a dose at which every frame's pieces must partition the range.

    Every distinct observed value up to :data:`_COVERAGE_POINTS` of them, else that many
    observed quantiles.  Each is a value the natural dose takes, so a piece set that leaves
    one uncovered drops observed mass from the induced density.
    """
    distinct = np.unique(values[np.isfinite(values)])
    if distinct.size <= _COVERAGE_POINTS:
        return distinct
    levels = np.linspace(0.0, 1.0, _COVERAGE_POINTS)
    return np.unique(np.quantile(distinct, levels, method="inverted_cdf"))


def _check_coverage(
    name: str,
    pieces: Sequence[ExplicitPiece],
    points: FloatArray,
    frame: FrameSource,
    rows: BoolArray,
    where: str,
) -> None:
    """Every observed value of the dose lies in exactly one piece on each counterfactual frame.

    :func:`_check_preimages` sees the values the formula counts, so it finds an overlap
    but not a gap: a natural value in no piece is never a preimage, and its mass silently
    leaves the induced density.  This asks the same partition question at the observed
    values of the dose, on the frames the formula reads.
    """
    for point in points:
        values = np.full(rows.size, float(point))
        count = np.zeros(rows.size, dtype=np.int64)
        for piece in pieces:
            count += piece.member(values, frame).astype(np.int64)
        wrong = rows & (count != 1)
        if wrong.any():
            covered = "no piece" if int(count[np.flatnonzero(wrong)[0]]) == 0 else "two pieces"
            raise DataError(
                f"policy {name!r} at {where}: {covered} holds the observed dose {float(point):g} "
                f"on {int(wrong.sum())} frame(s) that hold a candidate categorical cell or the "
                "preimages of the earlier doses. The density ratio would drop or double that "
                "mass. Declare pieces that partition the dose range at every value of the "
                "components they read, as Assumption 4 of Díaz et al. (2023) requires"
            )


def _enumerate(
    layout: NodeLayout,
    branch: JointBranch,
    target: FloatArray,
    cells: Sequence[tuple[int, ...]],
    requests: _Requests,
    observed_range: Sequence[tuple[float, float]],
    coverage: Sequence[FloatArray] = (),
) -> list[_Term]:
    """The terms of the induced density of one branch at one value of the node.

    For each candidate natural cell, the dose maps are inverted one dose at a time in the
    branch's order, piece by piece, on frames that hold the candidate cell and the earlier
    preimages.  Each path keeps its validity, its Jacobian, and the categorical image of
    ``(c, beta)``, which must equal the value's categorical part.
    """
    components = layout.components
    names = layout.names
    where = layout.where
    doses = layout.doses
    categorical = layout.categorical
    order = dose_order(branch.name, branch.order, components, where)
    pieces = {index: explicit_pieces(branch.declared.get(names[index])) for index in doses}
    target = np.asarray(target, dtype=float)
    target_codes = target[:, categorical]
    terms: list[_Term] = []
    for cell_index, cell in enumerate(cells):
        cell_codes = np.tile(np.asarray(cell, dtype=float), (layout.n, 1))
        cell_labels = {
            names[index]: layout.labels(index, cell_codes[:, j])
            for j, index in enumerate(categorical)
        }
        # (betas by dose column index, valid, Jacobian)
        paths: list[tuple[dict[int, FloatArray], BoolArray, FloatArray]] = [
            ({}, layout.rows.copy(), np.ones(layout.n))
        ]
        for position, index in enumerate(order):
            earlier = order[:position]
            grown: list[tuple[dict[int, FloatArray], BoolArray, FloatArray]] = []
            for betas, valid, jacobian in paths:
                readable = dict(cell_labels)
                readable.update({names[e]: betas[e] for e in earlier})
                frame = guarded_source(layout.frame_with, readable, set(names) - set(readable))
                b = target[:, index]
                policy = branch.declared.get(names[index])
                checked = valid & _inside(betas, earlier, observed_range, layout)
                preimages: list[tuple[FloatArray, BoolArray]] = []
                for piece in pieces[index]:
                    if piece.identity:
                        beta, slope = b.copy(), np.ones(layout.n)
                    else:
                        assert piece.inverse is not None and piece.derivative is not None
                        beta = piece.inverse(b, frame)
                        slope = piece.derivative(b, frame)
                    usable = valid & _finite(beta) & _finite(slope)
                    safe = np.where(usable, beta, layout.natural[:, index])
                    inside = usable & piece.member(safe, frame)
                    preimages.append((safe, inside))
                    grown.append(
                        (
                            {**betas, index: np.where(inside, beta, layout.natural[:, index])},
                            inside,
                            jacobian * np.abs(np.where(inside, slope, 0.0)),
                        )
                    )
                if isinstance(policy, ModifiedPolicy) and checked.any():
                    here = f"component {names[index]!r} of {where}"
                    _check_preimages(branch.name, pieces[index], preimages, frame, checked, here)
                    if coverage:
                        points = coverage[layout.doses.index(index)]
                        _check_coverage(branch.name, pieces[index], points, frame, checked, here)
                    assert policy.pieces is not None and not isinstance(policy.pieces, Mapping)
                    _check_monotone(
                        branch.name,
                        _PieceBranch(policy.name, policy.pieces),
                        layout.natural[:, index],
                        frame,
                        checked,
                        here,
                        note=_COUNTERFACTUAL_FRAMES,
                    )
                # A map that caught its own refused read still refuses.
                frame.check()
            paths = grown
        for betas, valid, jacobian in paths:
            dose_values = np.column_stack([betas[index] for index in doses])
            image = _image(layout, branch, cell_codes, dose_values, valid)
            hit = valid & np.all(image == target_codes, axis=1)
            weight = np.where(hit, jacobian, 0.0)
            factors = []
            for position in range(len(doses)):
                factors.append(
                    requests.ask(
                        position, cell_codes, dose_values, dose_values[:, position], cell=cell
                    )
                )
            terms.append(_Term(cell_index, weight, tuple(factors)))
    return terms


def _inside(
    betas: Mapping[int, FloatArray],
    earlier: Sequence[int],
    observed_range: Sequence[tuple[float, float]],
    layout: NodeLayout,
) -> BoolArray:
    """Rows whose earlier preimages lie inside each dose's bin range.

    A frame outside that range is one where the estimated density is exactly zero, by the
    zero contract of :meth:`~cleverly.learners.density.ConditionalDensity.density_at` outside
    the outer edges, so the formula reads nothing there and the declaration is not checked
    there.  ``observed_range`` must be those outer edges for the skip to be sound.
    """
    out = np.ones(layout.n, dtype=bool)
    positions = {index: k for k, index in enumerate(layout.doses)}
    for index in earlier:
        low, high = observed_range[positions[index]]
        out &= (betas[index] >= low) & (betas[index] <= high)
    return out


def _image(
    layout: NodeLayout,
    branch: JointBranch,
    cell_codes: FloatArray,
    dose_values: FloatArray,
    valid: BoolArray,
) -> FloatArray:
    """The categorical part of the branch's value at natural ``(c, beta)``: ``(n, n_cat)``.

    A categorical map reads the natural value of every other component, which here is the
    candidate cell and the preimage of the doses.  The level check runs on the valid rows.

    The check runs before any learner, so it cannot tell which frames the fitted density
    will put mass on.  A valid row whose ``(c, beta)`` later has estimated density zero
    contributes nothing to the ratio, and a map that returns no level there is still
    refused.  That refusal is conservative and deliberate: a categorical map must return a
    level at every value of the components it reads (Díaz et al. 2023, Assumption 1), and a
    frame of zero estimated density can have positive true density.
    """
    names = layout.names
    columns = []
    natural: dict[str, Any] = {}
    for j, index in enumerate(layout.categorical):
        natural[names[index]] = layout.labels(index, cell_codes[:, j])
    for k, index in enumerate(layout.doses):
        natural[names[index]] = dose_values[:, k]
    for j, index in enumerate(layout.categorical):
        component = layout.components[index]
        policy = branch.declared.get(component.name)
        if policy is None:
            columns.append(cell_codes[:, j])
            continue
        readable = {key: value for key, value in natural.items() if key != component.name}
        source = guarded_source(layout.frame_with, readable, (component.name,))
        columns.append(
            _categorical_component(
                branch.name,
                policy,
                component,
                cell_codes[:, j],
                source,
                valid,
                f"component {component.name!r} of {layout.where}",
            )
        )
        source.check()
    if not columns:
        return np.zeros((layout.n, 0))
    return np.column_stack(columns)


def evaluate_joint_density(
    layout: NodeLayout,
    branches: Sequence[JointBranch],
    evaluations: Sequence[JointEvaluation],
    *,
    learner: Learner,
    weights: FloatArray,
    folds: Folds,
    n_bins: int | None = None,
    groups: IntArray | None = None,
    n_jobs: int = 1,
    fit_rows: BoolArray | None = None,
) -> JointDensityParts:
    """Fit a vector node's law by the chain rule and evaluate each branch's induced density.

    Every design and every value is enumerated before any learner runs, because the
    preimages are known functions of the observed values.  The categorical factor is one
    out-of-fold multinomial over the observed cells.  Each dose's density is fitted once on
    its observed design and predicted out of fold at every counterfactual design.

    Parameters
    ----------
    layout : NodeLayout
        The node.
    branches : sequence of JointBranch
        Every branch of every policy at the node.
    evaluations : sequence of JointEvaluation
        The values to evaluate at, each naming the branches whose induced density it needs.
    learner : Learner
        The classifier of the categorical factor and of every dose's hazards.
    weights : ndarray
        ``(n,)`` observation weights.
    folds : Folds
        The outer split.
    n_bins : int or None
        The bin count of each dose's density.
    groups : ndarray or None
        Cluster codes.
    n_jobs : int
        Parallel workers across the folds.
    fit_rows : ndarray or None
        ``(n,)`` the rows every learner trains on, a subset of ``layout.rows``: one outer
        training fold of a cross-fitted recursion.  The observed cells and every dose's bin
        edges are still read from ``layout.rows``, so at the fold's held-out rows each
        factor is the model the out-of-fold mechanism uses there.  ``None`` trains on
        ``layout.rows``.

    Returns
    -------
    JointDensityParts
        The fitted factors at every value.
    """
    from ..estimators._nuisance import cross_fit_predictions

    n = layout.n
    doses = layout.doses
    names = layout.names
    cells = layout.cells()
    rows = layout.rows
    train = rows if fit_rows is None else rows & np.asarray(fit_rows, dtype=bool)
    natural_doses = layout.natural[:, doses]
    # Each dose's bin edges are read from every row at risk, before any fit, as the fit
    # itself reads them; a restricted fit (``fit_rows``) keeps them, as the scalar node does.
    dose_edges: list[list[float]] = []
    for k in range(len(doses)):
        grid_values = natural_doses[rows, k]
        bins = default_density_bins(grid_values.size) if n_bins is None else n_bins
        dose_edges.append(bin_edges(grid_values, bins, name=names[doses[k]]).tolist())
    # The range a fitted density can be positive on.  The density lookup returns exactly 0
    # outside its outer edges (``learners.density._lookup``), so a frame whose earlier
    # preimages leave this range contributes nothing, and :func:`_inside` skips its checks.
    # Keep the two in step: a lookup that extrapolated past the edges would need checks there.
    observed_range = [(float(edges[0]), float(edges[-1])) for edges in dose_edges]
    coverage = [_coverage_points(natural_doses[rows, k]) for k in range(len(doses))]
    requests = _Requests(layout, [{} for _ in doses], [{} for _ in doses])
    induced_terms: dict[tuple[int, int], list[_Term]] = {}
    for e, evaluation in enumerate(evaluations):
        for b in evaluation.branches:
            induced_terms[(e, b)] = _enumerate(
                layout, branches[b], evaluation.values, cells, requests, observed_range, coverage
            )
    observed_factors: list[tuple[tuple[int, Hashable, int], ...]] = []
    targets = np.zeros((n, len(evaluations)), dtype=np.int64)
    for e, evaluation in enumerate(evaluations):
        values = np.asarray(evaluation.values, dtype=float)
        codes = values[:, layout.categorical]
        targets[:, e] = layout.cell_codes(codes, cells)
        at = values[:, doses]
        factors = []
        for position in range(len(doses)):
            factors.append(requests.ask(position, codes, at, at[:, position]))
        observed_factors.append(tuple(factors))

    # Every dose's long table is checked before the first learner runs.
    fit_designs = [
        layout.dose_design(position, layout.natural[:, layout.categorical], natural_doses)
        for position in range(len(doses))
    ]
    # The counterfactual designs and their values count too, with a copy per parallel
    # worker: a node with many cells, pieces or policies holds more of them than tables.
    bin_counts = [len(edges) - 1 for edges in dose_edges]
    preflight_conditional_densities(
        learner,
        [(fit_designs[k].shape[1], natural_doses[:, k]) for k in range(len(doses))],
        folds,
        n_bins=n_bins,
        fit_mask=rows,
        n_jobs=n_jobs,
        names=[names[index] for index in doses],
        predictions=[requests.memory(k, bin_counts[k], folds) for k in range(len(doses))],
    )
    # The categorical factor.
    observed_cells = layout.cell_codes(layout.natural[:, layout.categorical], cells)
    cell_diagnostics: list[SuperLearnerDiagnostics] = []
    if len(cells) == 1:
        cell_probabilities = np.ones((n, 1))
    else:
        classes = tuple(float(index) for index in range(len(cells)))
        predicted, cell_diagnostics = cross_fit_predictions(
            learner,
            layout.conditioning,
            np.where(rows, observed_cells, 0).astype(float),
            weights,
            folds,
            task="classification",
            predict_designs={"cells": layout.conditioning},
            fit_mask=train,
            groups=groups,
            clip=(0.0, 1.0),
            classes=classes,
            n_jobs=n_jobs,
        )
        cell_probabilities = np.asarray(predicted["cells"], dtype=float)

    # Each dose's density, at its observed design and at every requested design.
    evaluated: list[dict[Hashable, FloatArray]] = []
    densities: dict[str, ConditionalDensity] = {}
    diagnostics: dict[str, DensityDiagnostics] = {}
    retained = 0
    for position, index in enumerate(doses):
        density, report, predicted_values = fit_conditional_density(
            learner,
            fit_designs[position],
            natural_doses[:, position],
            weights,
            folds,
            n_bins=n_bins,
            edges=dose_edges[position],
            groups=groups,
            n_jobs=n_jobs,
            fit_mask=train,
            name=names[index],
            retained=retained,
            predict_designs={
                key: (design, np.column_stack(requests.values[position][key]))
                for key, design in requests.designs[position].items()
            },
        )
        densities[names[index]] = density
        diagnostics[names[index]] = report
        retained += density.bin_probabilities.nbytes + sum(
            values.nbytes for values in predicted_values.values()
        )
        evaluated.append(predicted_values)

    def product(factors: Sequence[tuple[int, Hashable, int]]) -> FloatArray:
        out = np.ones(n)
        for position, key, column in factors:
            out = out * evaluated[position][key][:, column]
        return out

    induced = np.zeros((n, len(evaluations), len(branches), len(cells)))
    for (e, b), terms in induced_terms.items():
        for term in terms:
            induced[:, e, b, term.cell] += term.weight * product(term.factors)
    observed = np.column_stack([product(factors) for factors in observed_factors])
    return JointDensityParts(
        cells=cells,
        cell_probabilities=cell_probabilities,
        induced=induced,
        observed=observed,
        targets=targets,
        densities=densities,
        diagnostics=diagnostics,
        cell_diagnostics=tuple(cell_diagnostics),
    )


def warn_unsupported_values(
    name: str,
    parts: JointDensityParts,
    evaluation: int,
    moved: BoolArray,
    where: str,
) -> None:
    """Warn when a policy sends a row it moves to a value of estimated density zero.

    Positivity at a vector node reads the joint density: the value a policy assigns must lie
    in the conditional support of the node given its history.  For an ordered map that
    includes the support of each dose given the categorical components and the earlier
    doses, which the observed range of a cell does not show.

    Parameters
    ----------
    name : str
        The policy.
    parts : JointDensityParts
        The fitted factors.
    evaluation : int
        The evaluation that holds the branch's assigned values.
    moved : ndarray
        ``(n,)`` the rows the branch moves.
    where : str
        The node.
    """
    zero = moved & (parts.joint()[:, evaluation] <= 0.0)
    if not zero.any():
        return
    warnings.warn(
        f"policy {name!r} sends {int(zero.sum())} of the {int(moved.sum())} rows it moves at "
        f"{where} to a value where the estimated density of the node is zero. "
        "Identification requires the assigned value to lie in the conditional support of the "
        "node given its history: of the categorical cell, and of each dose given the cell and "
        "the earlier doses (Díaz et al. 2023, Assumption 1). Estimated zeros flag model "
        "support failures, not proof of nonidentification.",
        PositivityWarning,
        stacklevel=3,
    )
