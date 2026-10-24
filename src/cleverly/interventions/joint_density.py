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

**A dose with declared atoms has a mixed factor.**  Its factor is taken against Lebesgue
measure plus counting measure on its atoms :math:`a_{k,1}, \ldots, a_{k,J}`: it is
:math:`p_{k,j}(d_{<k}, c, h) = P(D_k = a_{k,j} \mid d_{<k}, c, h)` at an atom and
:math:`(1 - \sum_j p_{k,j})\, f_k(d_k \mid d_{<k}, c, h)` elsewhere, with :math:`f_k` the
density of the dose on the rows that hold no atom.  The dominating measure of the node is
the product of these measures and counting measure on the cells, so the joint density is
still the product of the factors, and the induced density is the same sum over cells and
paths with one more kind of path.  At an atom value :math:`b_k` the paths of dose :math:`k`
are its declared atoms :math:`a_{k,j}` whose image under the dose's map, on the path's frame,
is :math:`b_k`, each with a Jacobian factor of one.  At any other value they are the smooth
and identity pieces, as before.  A smooth piece that sends a dose of no recorded mass onto
an atom adds Lebesgue mass there, which has measure zero at a point.  This is the mixed
ratio of a dose of one column (:func:`~cleverly.interventions.policy.mixed_ratio`) applied to
each conditional factor.  The ratio exists when every atom image on every frame is a
declared atom (R-b3, checked here on the frames the formula reads, and on the observed
frame by :func:`~cleverly.interventions.policy.check_continuous_policy`).  A piece that
sends an interval to an atom (``Piece(..., to_atom=)``) is refused on this route at any dose
but the last: the mass it sends integrates every later factor over the interval, because
those factors condition on the collapsed dose, and that integral is not a finite sum over
preimages.  At the last dose, in the column order and in the policy's order, no later
factor or dose map reads the dose, and the mass is the integral of its binned density over
the interval: a finite sum over the bins (:func:`_collapse_paths`).  A categorical map
that reads that dose is refused.  The classifier route takes such a piece at any dose.

The remainder of the estimator is the product of the ratio error and the outcome
regression's error, as for a scalar continuous node.  No regression is fitted on an
estimated index: :math:`\beta` is a known function of observed values.  The ratio error
combines the errors of :math:`K + 1` factors, and the density of dose :math:`k` conditions on
:math:`k - 1` continuous doses, so its learner's rate degrades with :math:`k`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .._typing import BoolArray, FloatArray, IntArray, Learner
from .._warnings import warn as _warn
from ..data.validate import MAX_TREATMENT_LEVELS, arm_indicators, atom_block
from ..exceptions import CapabilityError, DataError, PositivityWarning, Refusal, RefusalReason
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
    ForbiddenRead,
    FrameSource,
    FrameWith,
    JointComponent,
    ModifiedPolicy,
    Piecewise,
    Scale,
    Shift,
    _atom_index,
    _call,
    _categorical_component,
    _check_monotone,
    _is_identity_map,
    _member,
    _per_row,
    _PieceBranch,
    _refuse_unresolved,
    atom_image_refusal,
    dose_order,
    guarded_source,
    policy_branches,
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
    to_atom : float or None
        The declared atom a collapsing piece sends its whole interval to, ``None`` for a
        piece that maps values one to one.
    bounds : callable or None
        ``bounds(frame, n)``, the ``(lower, upper)`` ends of a collapsing piece's interval
        on every row.
    """

    member: Callable[[FloatArray, FrameSource], BoolArray]
    inverse: Callable[[FloatArray, FrameSource], FloatArray] | None = None
    derivative: Callable[[FloatArray, FrameSource], FloatArray] | None = None
    to_atom: float | None = None
    bounds: Callable[[FrameSource, int], tuple[FloatArray, FloatArray]] | None = None

    @property
    def identity(self) -> bool:
        """Whether the piece leaves its values unchanged."""
        return self.inverse is None and self.to_atom is None


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

    if piece.to_atom is not None:

        def bounds(frame: FrameSource, n: int) -> tuple[FloatArray, FloatArray]:
            return branch.bounds(piece, frame, n)

        return ExplicitPiece(member, to_atom=float(piece.to_atom), bounds=bounds)
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
        return _capped(forward, inverse, slope, _refuse_unresolved(policy))
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
        "a Shift, Scale, Piecewise or ModifiedPolicy with pieces.",
        reason=RefusalReason.INVALID_DECLARATION,
    )


def collapse_refusal(name: str, atom: float) -> Refusal:
    """The refusal of a piece that sends an interval to an atom, on the density route.

    Parameters
    ----------
    name : str
        The component policy's name.
    atom : float
        The atom the piece sends its interval to.

    Returns
    -------
    Refusal
        An :attr:`~cleverly._refusal.RefusalReason.API_REDIRECTION` record: the classifier
        route fits the policy.
    """
    return Refusal(
        RefusalReason.API_REDIRECTION,
        f"policy {name!r} sends an interval of a component of a vector treatment to the "
        f"declared atom {atom:g} (Piece(..., to_atom={atom:g})), which the density route "
        "evaluates only at the last dose of the node. The induced mass at the atom "
        "integrates the factors of the later components over that interval, because each "
        "later factor conditions on the collapsed dose, and that integral is not a finite "
        "sum over preimages. At the last dose, in the node's column order and in the "
        "policy's order, the mass is the integral of one binned density, a finite sum.",
        alternative=(
            "Pass ratio='classifier', which estimates the ratio without a density, declare "
            "the dose last, or map the interval by a smooth piece."
        ),
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
    atoms : tuple of tuple of float
        Per dose, in column order, its declared atoms.  Empty when no dose declares one.
    """

    components: tuple[JointComponent, ...]
    natural: FloatArray
    rows: BoolArray
    frame_with: FrameWith
    conditioning: FloatArray
    where: str
    atoms: tuple[tuple[float, ...], ...] = ()

    def dose_atoms(self, position: int) -> tuple[float, ...]:
        """The declared atoms of the dose at ``position`` among the doses, or ``()``.

        Parameters
        ----------
        position : int
            The dose's position among the doses, in column order.

        Returns
        -------
        tuple of float
            The atoms.
        """
        return tuple(self.atoms[position]) if self.atoms else ()

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
        columns = np.asarray(self.categorical, dtype=np.intp)
        codes = self.natural[np.ix_(self.rows, columns)].astype(np.int64)
        cells = tuple(sorted({tuple(int(v) for v in row) for row in codes})) or ((),)
        if len(cells) > MAX_TREATMENT_LEVELS:
            raise CapabilityError(
                f"{self.where} holds {len(cells)} observed cells of its categorical "
                f"components, and a categorical treatment holds at most "
                f"{MAX_TREATMENT_LEVELS} levels. The density of the node sums over these "
                "cells; collapse a categorical component.",
                reason=RefusalReason.INVALID_DECLARATION,
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
            The design.  An earlier dose with declared atoms adds the indicator of each
            atom beside it, as the outcome designs do.
        """
        values = np.asarray(doses, dtype=float)
        earlier = [atom_block(values[:, k], self.dose_atoms(k)) for k in range(position)]
        return np.column_stack([self.conditioning, self.cell_block(codes), *earlier])

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
    atoms: tuple[tuple[float, ...], ...] = ()

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
        doses = self.doses[index]
        earlier = [
            atom_block(doses[:, k], self.atoms[k] if self.atoms else ())
            for k in range(doses.shape[1])
        ]
        return np.column_stack([fitted[:, : self.width], *blocks, *earlier])


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
                tuple(layout.dose_atoms(k) for k in range(position)),
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
        earlier = sum(1 + len(self.layout.dose_atoms(k)) for k in range(position))
        rows = 8 * held_out * (width + earlier + 2 * n_bins)
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
    edges: Sequence[Sequence[float] | None] = (),
) -> list[_Term]:
    """The terms of the induced density of one branch at one value of the node.

    For each candidate natural cell, the dose maps are inverted one dose at a time in the
    branch's order, piece by piece, on frames that hold the candidate cell and the earlier
    preimages.  Each path keeps its validity, its Jacobian, and the categorical image of
    ``(c, beta)``, which must equal the value's categorical part.

    A piece that sends an interval to an atom adds, at that atom, the continuous mass of the
    interval (:func:`_collapse_paths`).  It is evaluated at the last dose alone, in the
    node's column order and in the policy's order: no later factor conditions on the dose,
    and no later dose map reads it, so the mass is the integral of one binned density.
    ``edges`` holds each dose's bin edges.
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
        # (betas by dose column index, valid, Jacobian, the collapsed dose or None)
        paths: list[tuple[dict[int, FloatArray], BoolArray, FloatArray, str | None]] = [
            ({}, layout.rows.copy(), np.ones(layout.n), None)
        ]
        for position, index in enumerate(order):
            earlier = order[:position]
            grown: list[tuple[dict[int, FloatArray], BoolArray, FloatArray, str | None]] = []
            for betas, valid, jacobian, collapsed in paths:
                readable = dict(cell_labels)
                readable.update({names[e]: betas[e] for e in earlier})
                frame = guarded_source(layout.frame_with, readable, set(names) - set(readable))
                b = target[:, index]
                policy = branch.declared.get(names[index])
                checked = valid & _inside(betas, earlier, observed_range, layout)
                atoms = layout.dose_atoms(doses.index(index))
                # A value at a declared atom is reached by the atoms' own paths alone: a
                # smooth or identity piece adds Lebesgue mass there, of measure zero at a
                # point under the counting part of the dominating measure.
                at_atom = _atom_index(b, atoms) if atoms else np.full(layout.n, -1, dtype=np.int64)
                preimages: list[tuple[FloatArray, BoolArray]] = []
                for piece in pieces[index]:
                    if piece.to_atom is not None:
                        # A collapsing piece has no preimage of a value: its mass at the
                        # atom is added below.
                        continue
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
                    kept = inside & (at_atom < 0)
                    grown.append(
                        (
                            {**betas, index: np.where(kept, beta, layout.natural[:, index])},
                            kept,
                            jacobian * np.abs(np.where(kept, slope, 0.0)),
                            collapsed,
                        )
                    )
                if atoms:
                    grown.extend(
                        (*path, collapsed)
                        for path in _atom_paths(
                            branch.name,
                            policy,
                            atoms,
                            at_atom,
                            (betas, valid, jacobian),
                            index,
                            frame,
                            checked,
                            f"component {names[index]!r} of {where}",
                        )
                    )
                collapsing = [piece for piece in pieces[index] if piece.to_atom is not None]
                if collapsing:
                    if index != doses[-1] or index != order[-1]:
                        raise CapabilityError.from_refusal(
                            collapse_refusal(
                                str(getattr(policy, "name", branch.name)),
                                float(collapsing[0].to_atom or 0.0),
                            )
                        )
                    grown.extend(
                        (*path, names[index])
                        for path in _collapse_paths(
                            collapsing,
                            b,
                            atoms,
                            at_atom,
                            (betas, valid, jacobian),
                            index,
                            frame,
                            edges[doses.index(index)] if edges else None,
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
        for betas, valid, jacobian, collapsed in paths:
            dose_values = np.column_stack([betas[index] for index in doses])
            if collapsed is None:
                image = _image(layout, branch, cell_codes, dose_values, valid)
            else:
                image = _collapsed_image(layout, branch, cell_codes, dose_values, valid, collapsed)
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


def _atom_paths(
    name: str,
    policy: object | None,
    atoms: tuple[float, ...],
    at_atom: IntArray,
    path: tuple[dict[int, FloatArray], BoolArray, FloatArray],
    index: int,
    frame: FrameSource,
    checked: BoolArray,
    where: str,
) -> list[tuple[dict[int, FloatArray], BoolArray, FloatArray]]:
    """The paths of a dose with declared atoms that start at an atom, one per atom.

    Atom :math:`a_j` reaches the evaluated value :math:`b` on a row when :math:`b` is a
    declared atom and the dose's map sends :math:`a_j` to it on the path's frame.  Its
    Jacobian factor is one, because the counting part of the dominating measure has no
    derivative.  The image of each atom must be a declared atom on every frame the formula
    reads (R-b3): an image off the atoms gives the induced law an atom the observed law
    lacks, and no ratio exists.  The check is conservative on a frame of zero estimated
    density, as the other checks of the counterfactual frames are.
    """
    betas, valid, jacobian = path
    n = valid.size
    if policy is None:
        branch = None
    else:
        ((_, branch),) = policy_branches(policy)
    out = []
    for atom in atoms:
        source = np.full(n, float(atom))
        if branch is None:
            image = source
        else:
            with np.errstate(all="ignore"):
                assigned = np.asarray(branch.assign(source, frame), dtype=float).reshape(-1)
            image = np.full(n, float(assigned[0])) if assigned.size == 1 else assigned
        target = _atom_index(image, atoms)
        bad = checked & (target < 0)
        if bad.any():
            first = float(image[np.flatnonzero(bad)[0]])
            raise CapabilityError.from_refusal(
                atom_image_refusal(
                    name, where, float(atom), first, describe=name, note=_COUNTERFACTUAL_ATOMS
                )
            )
        hit = valid & (at_atom >= 0) & (target == at_atom)
        out.append(({**betas, index: source}, hit, np.where(hit, jacobian, 0.0)))
    return out


def _collapsed_image(
    layout: NodeLayout,
    branch: JointBranch,
    cell_codes: FloatArray,
    dose_values: FloatArray,
    valid: BoolArray,
    collapsed: str,
) -> FloatArray:
    """:func:`_image` on a collapse path, where no categorical map may read the dose."""
    try:
        return _image(layout, branch, cell_codes, dose_values, valid, hidden=(collapsed,))
    except ForbiddenRead as error:
        raise CapabilityError(
            f"policy {branch.name!r} at {layout.where} collapses an interval of {collapsed!r} "
            f"to an atom, and a categorical map reads {error}. On the density route the mass "
            "of the interval is one integral of the dose's binned density only when no other "
            "map reads the collapsed dose.",
            reason=RefusalReason.API_REDIRECTION,
            alternative="Pass ratio='classifier', or drop the read.",
        ) from error


def _collapse_paths(
    pieces: Sequence[ExplicitPiece],
    b: FloatArray,
    atoms: tuple[float, ...],
    at_atom: IntArray,
    path: tuple[dict[int, FloatArray], BoolArray, FloatArray],
    index: int,
    frame: FrameSource,
    edges: Sequence[float] | None,
) -> list[tuple[dict[int, FloatArray], BoolArray, FloatArray]]:
    r"""The paths of the last dose that a collapsing piece sends to the evaluated atom.

    At an atom value :math:`b = a` of the last dose, a piece that sends :math:`I(h)` to
    :math:`a` adds :math:`(1 - \sum_j p_j) \int_{I(h)} f(s)\,ds`.  The binned density is
    constant on each bin, so the integral is
    :math:`\sum_m f(s_m) |I(h) \cap B_m|`, exactly: one path per bin :math:`B_m`, read at a
    point :math:`s_m` inside it that is no atom, with weight the overlap length.  The
    atoms inside :math:`I(h)` reach :math:`a` by their own paths (:func:`_atom_paths`).
    """
    betas, valid, jacobian = path
    out: list[tuple[dict[int, FloatArray], BoolArray, FloatArray]] = []
    if edges is None:
        # Every row at risk holds an atom of this dose, so it has no continuous mass.
        return out
    grid = np.asarray(edges, dtype=float)
    left, right = grid[:-1], grid[1:]
    n = b.size
    for piece in pieces:
        assert piece.bounds is not None and piece.to_atom is not None
        lower, upper = piece.bounds(frame, n)
        target = _atom_index(np.array([piece.to_atom]), atoms)[0] if atoms else -1
        hit = valid & (at_atom >= 0) & (at_atom == target)
        for m in range(left.size):
            with np.errstate(invalid="ignore"):
                overlap = np.clip(
                    np.minimum(upper, right[m]) - np.maximum(lower, left[m]), 0.0, None
                )
            reach = hit & (np.nan_to_num(overlap) > 0.0)
            if not reach.any():
                continue
            point = _bin_point(left[m], right[m], atoms)
            out.append(
                (
                    {**betas, index: np.full(n, point)},
                    reach,
                    np.where(reach, jacobian * np.nan_to_num(overlap), 0.0),
                )
            )
    return out


def _bin_point(left: float, right: float, atoms: tuple[float, ...]) -> float:
    """A point strictly inside a bin that is no declared atom."""
    for share in (0.5, 1.0 / 3.0, 2.0 / 3.0):
        point = left + share * (right - left)
        if _atom_index(np.array([point]), atoms)[0] < 0:
            return float(point)
    raise AssertionError("three points of one bin cannot all be atoms")


#: How the atom-image check reads a counterfactual frame, appended to its refusal.
_COUNTERFACTUAL_ATOMS = (
    ". The check reads each frame the density ratio evaluates, which holds a candidate "
    "categorical cell or the preimages of the earlier doses, before any learner runs, so it "
    "is conservative on a frame the data give no mass"
)


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
    hidden: tuple[str, ...] = (),
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

    ``hidden`` names doses a categorical map may not read: a collapsed dose, whose natural
    value runs over an interval on the path.
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
        readable = {
            key: value
            for key, value in natural.items()
            if key != component.name and key not in hidden
        }
        source = guarded_source(layout.frame_with, readable, (component.name, *hidden))
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
    # Per dose, the declared atom each natural dose equals, ``-1`` for none.  A dose with
    # atoms fits its density on the rows that hold no atom, as a dose of one column does.
    natural_atoms = [
        _atom_index(natural_doses[:, k], layout.dose_atoms(k))
        if layout.dose_atoms(k)
        else np.full(n, -1, dtype=np.int64)
        for k in range(len(doses))
    ]
    # Each dose's bin edges are read from every row at risk, before any fit, as the fit
    # itself reads them; a restricted fit (``fit_rows``) keeps them, as the scalar node does.
    # A dose whose every row at risk holds an atom fits no density (``None``).
    dose_edges: list[list[float] | None] = []
    for k in range(len(doses)):
        grid_values = natural_doses[rows & (natural_atoms[k] < 0), k]
        if grid_values.size == 0:
            dose_edges.append(None)
            continue
        bins = default_density_bins(grid_values.size) if n_bins is None else n_bins
        dose_edges.append(bin_edges(grid_values, bins, name=names[doses[k]]).tolist())
    # The range a fitted density can be positive on.  The density lookup returns exactly 0
    # outside its outer edges (``learners.density._lookup``), so a frame whose earlier
    # preimages leave this range contributes nothing, and :func:`_inside` skips its checks.
    # Keep the two in step: a lookup that extrapolated past the edges would need checks there.
    # A declared atom outside the edges has mass, so the range covers the atoms too.
    observed_range = []
    for k, edges in enumerate(dose_edges):
        ends = [] if edges is None else [float(edges[0]), float(edges[-1])]
        ends.extend(float(atom) for atom in layout.dose_atoms(k))
        observed_range.append((min(ends), max(ends)))
    coverage = [_coverage_points(natural_doses[rows, k]) for k in range(len(doses))]
    requests = _Requests(layout, [{} for _ in doses], [{} for _ in doses])
    induced_terms: dict[tuple[int, int], list[_Term]] = {}
    for e, evaluation in enumerate(evaluations):
        for b in evaluation.branches:
            induced_terms[(e, b)] = _enumerate(
                layout,
                branches[b],
                evaluation.values,
                cells,
                requests,
                observed_range,
                coverage,
                dose_edges,
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
    fitted_doses = [k for k in range(len(doses)) if dose_edges[k] is not None]
    if fitted_doses:
        preflight_conditional_densities(
            learner,
            [(fit_designs[k].shape[1], natural_doses[:, k]) for k in fitted_doses],
            folds,
            n_bins=n_bins,
            fit_mask=rows,
            n_jobs=n_jobs,
            names=[names[doses[k]] for k in fitted_doses],
            predictions=[
                requests.memory(k, len(dose_edges[k] or ()) - 1, folds) for k in fitted_doses
            ],
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
        atoms = layout.dose_atoms(position)
        continuous = natural_atoms[position] < 0
        edges = dose_edges[position]
        if edges is None:
            predicted_values = {
                key: np.zeros((n, len(columns)))
                for key, columns in requests.values[position].items()
            }
        else:
            density, report, predicted_values = fit_conditional_density(
                learner,
                fit_designs[position],
                natural_doses[:, position],
                weights,
                folds,
                n_bins=n_bins,
                edges=edges,
                groups=groups,
                n_jobs=n_jobs,
                fit_mask=train & continuous if atoms else train,
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
        if atoms:
            predicted_values = _mixed_factor(
                learner,
                layout,
                position,
                fit_designs[position],
                natural_atoms[position],
                requests,
                predicted_values,
                weights=weights,
                folds=folds,
                train=train,
                groups=groups,
                n_jobs=n_jobs,
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


def _mixed_factor(
    learner: Learner,
    layout: NodeLayout,
    position: int,
    fit_design: FloatArray,
    natural_atom: IntArray,
    requests: _Requests,
    density_values: Mapping[Hashable, FloatArray],
    *,
    weights: FloatArray,
    folds: Folds,
    train: BoolArray,
    groups: IntArray | None,
    n_jobs: int,
) -> dict[Hashable, FloatArray]:
    r"""The factor of a dose with declared atoms at every requested design and value.

    A multinomial classifier of "which declared atom, or none" on the dose's design gives
    :math:`p_{k,j}`, cross-fitted on the folds every factor uses and predicted out of fold
    at each requested design.  The factor is :math:`p_{k,j}` at atom :math:`a_{k,j}` and
    :math:`(1 - \sum_j p_{k,j}) f_k` elsewhere, with ``density_values`` the density
    :math:`f_k` of the rows that hold no atom at the same designs and values.
    """
    from ..estimators._nuisance import cross_fit_predictions

    atoms = layout.dose_atoms(position)
    n = layout.n
    label = np.where(layout.rows & (natural_atom >= 0), natural_atom + 1, 0).astype(float)
    classes = tuple(float(j) for j in range(len(atoms) + 1))
    none = bool(np.any(train & (natural_atom < 0)))
    if not none:
        classes = classes[1:]
    keys = list(requests.designs[position])
    designs = {
        str(number): requests.designs[position][key](np.arange(n), fit_design)
        for number, key in enumerate(keys)
    }
    predicted, _ = cross_fit_predictions(
        learner,
        fit_design,
        label,
        weights,
        folds,
        task="classification",
        predict_designs=designs,
        fit_mask=train,
        groups=groups,
        clip=(0.0, 1.0),
        classes=classes,
        n_jobs=n_jobs,
    )
    rows = np.arange(n)
    out: dict[Hashable, FloatArray] = {}
    for number, key in enumerate(keys):
        probabilities = np.asarray(predicted[str(number)], dtype=float)
        p = probabilities[:, 1:] if none else probabilities
        share = np.clip(1.0 - p.sum(axis=1), 0.0, 1.0)
        values = np.column_stack(requests.values[position][key])
        density = np.asarray(density_values[key], dtype=float)
        columns = []
        for column in range(values.shape[1]):
            which = _atom_index(values[:, column], atoms)
            at_atom = p[rows, np.maximum(which, 0)]
            columns.append(np.where(which >= 0, at_atom, share * density[:, column]))
        out[key] = np.column_stack(columns)
    return out


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
    _warn(
        f"policy {name!r} sends {int(zero.sum())} of the {int(moved.sum())} rows it moves at "
        f"{where} to a value where the estimated density of the node is zero. "
        "Identification requires the assigned value to lie in the conditional support of the "
        "node given its history: of the categorical cell, and of each dose given the cell and "
        "the earlier doses (Díaz et al. 2023, Assumption 1). Estimated zeros flag model "
        "support failures, not proof of nonidentification.",
        PositivityWarning,
        stacklevel=3,
    )
