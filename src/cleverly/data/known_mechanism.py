r"""A treatment mechanism that the design declares known.

A randomized trial fixes :math:`g_0(a \mid W)` by its allocation scheme.  A fit can divide by
that known mechanism instead of an estimate of it.  With :math:`g = g_0` the TMLE remainder
:math:`P_0[(g - g_0)/g \cdot (\bar Q - \bar Q_0)]` is identically zero, so the estimator is
consistent for any outcome-regression limit and its influence curve is
:math:`D^*(\bar Q_\infty, g_0)` exactly (Moore and van der Laan 2009).

The declaration lives on the **data**, not on an estimator.  Every reader that subsets or
resamples rows (the bootstrap, a data-subset refutation) then carries the matching rows of the
mechanism by construction.  Every reader that rewrites the treatment (the placebo refuter,
simulated confounding) drops the declaration, because a permuted or simulated treatment has a
different mechanism.

:class:`KnownMechanism` is the point-treatment carrier.  :func:`known_mechanism` is its one
parser, which reads four forms:

==========================  ===============================================================
form                        reading
==========================  ===============================================================
mapping level -> column     the named frame column holds ``P(A = level | W)``; preferred
mapping level -> array      one ``(n,)`` column per arm, every arm named
``(n, K)`` array            columns in the sorted level order
``(n,)`` array              two arms only, ``P(A = levels[1] | W)``
==========================  ===============================================================

The longitudinal carrier, :class:`KnownNodeMechanisms`, holds one matrix per treatment node
and one retention vector per censoring node.  :func:`known_node_mechanisms` parses it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np

from .._typing import FloatArray
from ..exceptions import CapabilityError, DataError

__all__ = [
    "CONTINUOUS_REFUSAL",
    "KNOWN_CTMLE_REFUSAL",
    "KNOWN_EVALUATION_REFUSAL",
    "KNOWN_POLICIES_REFUSAL",
    "KNOWN_SCREEN_REFUSAL",
    "KNOWN_TREATMENT_DELTA_REFUSAL",
    "KnownMechanism",
    "KnownNodeMechanisms",
    "KnownSource",
    "known_mechanism",
    "known_node_mechanisms",
    "moved_known_values_refusal",
    "probability_columns",
]

#: How a declaration was written.  Recorded because the column form is the one a companion
#: frame or a study design can repeat by name, and the array forms cannot be repeated.
KnownSource = Literal["array", "mapping", "columns"]

#: Tolerance of the row-sum check.  Loose enough for float error in a caller's complement,
#: tight enough that a second mechanism cannot pass as this one.
_ROW_SUM_TOLERANCE = 1e-12

#: The refusal of a known mechanism on a treatment with no arms.
CONTINUOUS_REFUSAL = (
    "treatment_probabilities= gives arm probabilities, and a continuous treatment has a "
    "conditional density, not arms. A known density is not supported; fit without "
    "treatment_probabilities=."
)
#: The refusal of C-TMLE on data that declares the mechanism.
KNOWN_CTMLE_REFUSAL = (
    "CTMLE selects the treatment mechanism, and the data declares it known, so there is "
    "nothing to select. Fit TMLE or DRTMLE on the declared data."
)
#: The refusal of a modified treatment policy beside a declared mechanism.
KNOWN_POLICIES_REFUSAL = (
    "treatment_probabilities= and policies= are not combined. A modified treatment policy "
    "reads the mechanism in its estimand and in its ratio, and the known-mechanism "
    "translation is written for incremental= only. Fit without treatment_probabilities=."
)
#: The refusal of covariate screening beside a declared mechanism.
KNOWN_SCREEN_REFUSAL = (
    "screen_treatment= selects covariates for the treatment learner, and "
    "treatment_probabilities= replaces that learner. Drop one of the two."
)
#: The refusal of a declared mechanism beside a declared missing treatment.
KNOWN_TREATMENT_DELTA_REFUSAL = (
    "treatment_probabilities= and treatment_delta= are not combined. A missing treatment "
    "needs P(A = a | treatment recorded, W), which equals the design mechanism only when "
    "recording is independent of treatment given W, and the fit cannot check that. Fit "
    "without treatment_probabilities=."
)
#: The refusal of a DR-TMLE evaluation companion that declares no mechanism.
KNOWN_EVALUATION_REFUSAL = (
    "evaluation= evaluates every nuisance on an independent draw, and that draw declares no "
    "known treatment mechanism. Declare the same treatment_probabilities columns on the "
    "evaluation data, or drop evaluation=."
)


def moved_known_values_refusal(count: int, bounds: tuple[float, float], values: FloatArray) -> str:
    """The refusal of a truncation bound pair that would move a known value.

    Parameters
    ----------
    count : int
        How many rows the bound pair moves.
    bounds : tuple of float
        The bound pair.
    values : FloatArray
        The declared ``(n, K)`` probabilities.

    Returns
    -------
    str
        The sentence the fit raises.
    """
    known = np.asarray(values, dtype=float)
    return (
        f"treatment_probabilities has {count} values outside the truncation bounds "
        f"[{bounds[0]:.4g}, {bounds[1]:.4g}] (smallest {float(np.min(known)):.4g}, largest "
        f"{float(np.max(known)):.4g}). Truncating a known design mechanism moves the estimate "
        "with no variance reason. Pass g_bounds=(lower, upper) that contains every known "
        "probability."
    )


@dataclass(frozen=True)
class KnownMechanism:
    r"""The declared point-treatment mechanism :math:`g_0(a \mid W)`, one column per arm.

    Parameters
    ----------
    values : FloatArray
        The ``(n, K)`` probabilities, columns in arm-code order, which is the sorted order of
        the treatment levels.  Every row sums to one and every entry lies in ``(0, 1)``.
    source : {"array", "mapping", "columns"}
        The form the caller wrote.
    columns : tuple of tuple
        ``(level, column name)`` pairs when ``source`` is ``"columns"``, in level order, and
        empty otherwise.

    Attributes
    ----------
    n : int
    n_arms : int
    """

    values: FloatArray
    source: KnownSource
    columns: tuple[tuple[Any, str], ...] = ()

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 2:
            raise ValueError(f"a known mechanism is (n, K); got shape {values.shape}")
        if self.source not in ("array", "mapping", "columns"):
            raise ValueError(f"source must be 'array', 'mapping' or 'columns'; got {self.source!r}")

    @property
    def n(self) -> int:
        """The number of rows."""
        return int(np.asarray(self.values).shape[0])

    @property
    def n_arms(self) -> int:
        """The number of arms."""
        return int(np.asarray(self.values).shape[1])

    @property
    def column_names(self) -> tuple[str, ...]:
        """The frame columns of the column form, in level order, or ``()``."""
        return tuple(name for _, name in self.columns)

    def subset(self, index: Any) -> KnownMechanism:
        """Return the declaration on the selected rows, repeats kept.

        Parameters
        ----------
        index : array-like of int or bool
            The selected rows, as :meth:`~cleverly.data.CausalData.subset` receives them.

        Returns
        -------
        KnownMechanism
            The same declaration, its values indexed by ``index``.
        """
        idx = np.asarray(index)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        return replace(self, values=np.asarray(self.values, dtype=float)[idx])


def probability_columns(supplied: Any) -> tuple[tuple[Any, str], ...] | None:
    """The ``(level, column)`` pairs of a column-name declaration, or ``None``.

    Parameters
    ----------
    supplied : Any
        A ``treatment_probabilities=`` value.

    Returns
    -------
    tuple of tuple or None
        The pairs when ``supplied`` is a mapping whose values are all strings, ``None`` for
        every other form.

    Raises
    ------
    DataError
        If a mapping mixes column names with arrays.
    """
    if not isinstance(supplied, Mapping):
        return None
    named = [isinstance(value, str) for value in supplied.values()]
    if not any(named):
        return None
    if not all(named):
        raise DataError(
            "treatment_probabilities mixes column names with arrays. Give every arm a column "
            "name, or give every arm an array."
        )
    return tuple((level, str(column)) for level, column in supplied.items())


def known_mechanism(
    supplied: Any,
    *,
    n: int,
    levels: Sequence[Any],
    treatment_name: str,
    columns: tuple[tuple[Any, str], ...] = (),
) -> KnownMechanism:
    """Parse a ``treatment_probabilities=`` declaration into a :class:`KnownMechanism`.

    The mapping form is the one to reach for.  A trial's arms are named, and the positional
    forms bind to the arm *codes*, which are indices into the sorted levels.  So ``(n,)`` is
    ``P(A = levels[1] | W)``: the probability of ``"placebo"`` in a trial labelled
    ``active``/``placebo``.  A caller who reads it as "the probability of treatment" inverts
    the design, and nothing downstream contradicts them, so each message names the level that
    a positional form resolved to.

    Parameters
    ----------
    supplied : Any
        A mapping level to ``(n,)`` array, an ``(n, K)`` array, or an ``(n,)`` array at two
        arms.  A column-name mapping arrives here already read into arrays, with its names in
        ``columns``.
    n : int
        The number of rows.
    levels : sequence
        The treatment levels in sorted order.  Empty for a continuous treatment, which is
        refused.
    treatment_name : str
        The treatment's name, for the messages.
    columns : tuple of tuple
        The ``(level, column)`` pairs of a column-name declaration, or ``()``.

    Returns
    -------
    KnownMechanism
        The validated ``(n, K)`` declaration in arm-code order.

    Raises
    ------
    CapabilityError
        If the treatment is continuous.
    DataError
        If the shape, a level, the range or a row sum is wrong.
    """
    levels = list(levels)
    if not levels:
        raise CapabilityError(CONTINUOUS_REFUSAL)
    source: KnownSource
    if isinstance(supplied, Mapping):
        values = _probabilities_from_levels(supplied, levels, n=n, treatment_name=treatment_name)
        source = "columns" if columns else "mapping"
    else:
        source = "array"
        values = np.array(supplied, dtype=float, copy=True)
        if values.ndim == 1:
            if len(levels) != 2:
                raise DataError(
                    f"treatment_probabilities as an (n,) vector is "
                    f"P({treatment_name} = {levels[1]!r} | W) and describes two arms "
                    f"only; {treatment_name} has {len(levels)} levels {levels}. Pass "
                    f"an (n, {len(levels)}) array in that level order, or a mapping keyed "
                    "by every level."
                )
            if values.shape[0] != n:
                raise DataError(f"treatment_probabilities has {values.shape[0]} rows; expected {n}")
            values = np.column_stack([1.0 - values, values])
    if values.shape != (n, len(levels)):
        if len(levels) == 2:
            raise DataError(
                f"treatment_probabilities must be (n,) for P({treatment_name} = "
                f"{levels[1]!r} | W), (n, 2) in the level order {levels}, or a mapping "
                f"keyed by those levels; got {values.shape}"
            )
        raise DataError(
            f"treatment_probabilities must be (n, {len(levels)}) in the level order "
            f"{levels}, or a mapping keyed by those levels; got {values.shape}"
        )
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0) or np.any(values >= 1.0):
        raise DataError("treatment_probabilities must be finite and strictly between 0 and 1")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=_ROW_SUM_TOLERANCE):
        raise DataError("each row of treatment_probabilities must sum to one")
    ordered = tuple(sorted(columns, key=lambda pair: _level_index(pair[0], levels)))
    return KnownMechanism(values, source, ordered)


def _level_index(label: Any, levels: list[Any]) -> int:
    matches = [index for index, level in enumerate(levels) if level == label]
    return matches[0] if matches else len(levels)


def _probabilities_from_levels(
    supplied: Mapping[Any, Any], levels: list[Any], *, n: int, treatment_name: str
) -> FloatArray:
    """One ``(n,)`` column per arm, keyed on the way in by the caller's own level.

    Keyed by level in and by arm code out, which is the convention every reported name
    follows.  Every arm must be named, because an arm left out would be filled in by a
    complement the caller never wrote, which is the assumption this form exists to state
    rather than inherit.
    """
    columns: dict[int, FloatArray] = {}
    for label, probabilities in supplied.items():
        index = _level_index(label, levels)
        if index == len(levels):
            raise DataError(
                f"treatment_probabilities names {label!r}, which is not a level of "
                f"{treatment_name}; its levels are {levels}"
            )
        column = np.asarray(probabilities, dtype=float).reshape(-1)
        if column.shape[0] != n:
            raise DataError(
                f"treatment_probabilities[{label!r}] has {column.shape[0]} rows; expected {n}"
            )
        columns[index] = column
    missing = [levels[index] for index in range(len(levels)) if index not in columns]
    if missing:
        raise DataError(
            f"treatment_probabilities must name every arm, and {missing} are missing. "
            "The arms left out would take whatever is left over from the ones named, "
            "which is the design this form exists to state explicitly; give every arm "
            "its own column."
        )
    return np.column_stack([columns[index] for index in range(len(levels))])


# --------------------------------------------------------------------- longitudinal


@dataclass(frozen=True)
class KnownNodeMechanisms:
    r"""Declared longitudinal mechanism factors, node by node.

    ``treatment`` holds, for each treatment node, either ``None`` (the factor is estimated)
    or the ``(n, K_t)`` matrix :math:`P(A_t = a \mid \text{observed past})` in arm-code
    order.  ``censoring`` holds, for each censoring node, either ``None`` or the ``(n,)``
    retention probability :math:`P(C_t = 1 \mid \text{observed past})`.  Values on rows that
    are not at risk at a node are never read and may be ``NaN``.

    Parameters
    ----------
    treatment : tuple of FloatArray or None
        One entry per treatment node.
    censoring : tuple of FloatArray or None
        One entry per censoring node, or ``()`` without censoring.
    source : {"array", "mapping", "columns"}
        The form the caller wrote, for the treatment declaration when there is one.
    treatment_columns : tuple
        By node, the ``(level, column)`` pairs of a column-name declaration, or ``()``.
    censoring_columns : tuple of str or None
        By censoring node, the column of a column-name declaration, or ``None``.

    Attributes
    ----------
    declares_treatment : bool
    declares_censoring : bool
    """

    treatment: tuple[FloatArray | None, ...]
    censoring: tuple[FloatArray | None, ...] = ()
    source: KnownSource = "array"
    treatment_columns: tuple[tuple[tuple[Any, str], ...], ...] = ()
    censoring_columns: tuple[str | None, ...] = ()

    @property
    def declares_treatment(self) -> bool:
        """Whether any treatment node is declared known."""
        return any(values is not None for values in self.treatment)

    @property
    def declares_censoring(self) -> bool:
        """Whether any censoring node is declared known."""
        return any(values is not None for values in self.censoring)

    def treatment_at(self, time: int) -> FloatArray | None:
        """The ``(n, K_t)`` declaration of node ``time``, counted from one, or ``None``.

        Parameters
        ----------
        time : int
            The node, counted from one.

        Returns
        -------
        FloatArray or None
            The declared matrix, or ``None`` when the node is estimated.
        """
        if not self.treatment:
            return None
        return self.treatment[time - 1]

    def censoring_at(self, time: int) -> FloatArray | None:
        """The ``(n,)`` declared retention of censoring node ``time``, or ``None``.

        Parameters
        ----------
        time : int
            The node, counted from one.

        Returns
        -------
        FloatArray or None
            The declared retention, or ``None`` when the node is estimated.
        """
        if not self.censoring:
            return None
        return self.censoring[time - 1]

    def subset(self, index: Any) -> KnownNodeMechanisms:
        """Return the declarations on the selected units, repeats kept.

        Parameters
        ----------
        index : array-like of int or bool
            The selected units.

        Returns
        -------
        KnownNodeMechanisms
            The same declarations, every array indexed by ``index``.
        """
        idx = np.asarray(index)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        return replace(
            self,
            treatment=tuple(None if values is None else values[idx] for values in self.treatment),
            censoring=tuple(None if values is None else values[idx] for values in self.censoring),
        )


def known_node_mechanisms(
    *,
    treatment_probabilities: Any,
    censoring_probabilities: Any,
    n: int,
    treatment_names: Sequence[str],
    treatment_levels: Sequence[Sequence[Any]],
    continuous_nodes: Sequence[bool],
    censoring_names: Sequence[str],
    treatment_columns: Mapping[str, tuple[tuple[Any, str], ...]] | None = None,
    censoring_columns: Mapping[str, str] | None = None,
) -> KnownNodeMechanisms | None:
    """Parse the longitudinal declarations, or return ``None`` when neither is given.

    ``treatment_probabilities`` takes three forms: an ``(n, T)`` array of
    ``P(A_t = levels_t[1] | observed past)`` when every node is binary, an ``(n, T, K)`` array
    when every node shares ``K`` levels, or a mapping node column to a per-node mapping level
    to array.  ``censoring_probabilities`` takes an ``(n, T_c)`` array or a mapping censoring
    column to array.  A column-name form arrives here already read, with its names in
    ``treatment_columns`` and ``censoring_columns``.  ``NaN`` is accepted; the fit refuses a
    ``NaN`` on a row that is at risk at its node, with the row count.

    Parameters
    ----------
    treatment_probabilities : Any
        The treatment declaration, or ``None``.
    censoring_probabilities : Any
        The censoring declaration, or ``None``.
    n : int
        The number of units.
    treatment_names : sequence of str
        The treatment column of each node.
    treatment_levels : sequence of sequence
        The levels of each node, in sorted order.
    continuous_nodes : sequence of bool
        Whether each node holds a continuous dose, which is refused.
    censoring_names : sequence of str
        The censoring column of each node, or ``()`` without censoring.
    treatment_columns : mapping or None
        By node column, the ``(level, column)`` pairs of a column-name declaration.
    censoring_columns : mapping or None
        By censoring column, the column of a column-name declaration.

    Returns
    -------
    KnownNodeMechanisms or None
        The validated declarations.

    Raises
    ------
    CapabilityError
        If a declared node holds a continuous dose.
    DataError
        If a shape, a name, a level or a value is wrong.
    """
    if treatment_probabilities is None and censoring_probabilities is None:
        return None
    n_nodes = len(treatment_names)
    treatment: list[FloatArray | None] = [None] * n_nodes
    source: KnownSource = "array"
    by_node_columns: list[tuple[tuple[Any, str], ...]] = [()] * n_nodes
    if treatment_probabilities is not None:
        if isinstance(treatment_probabilities, Mapping):
            source = "columns" if treatment_columns else "mapping"
            for node, per_node in treatment_probabilities.items():
                if node not in treatment_names:
                    raise DataError(
                        f"treatment_probabilities names {node!r}, which is not a treatment "
                        f"node; the nodes are {list(treatment_names)}"
                    )
                time = list(treatment_names).index(node)
                if not isinstance(per_node, Mapping):
                    raise DataError(
                        f"treatment_probabilities[{node!r}] must map each level of {node} to "
                        "its probability (a column name or an (n,) array)"
                    )
                levels = list(treatment_levels[time])
                if continuous_nodes[time] or not levels:
                    raise CapabilityError(CONTINUOUS_REFUSAL)
                treatment[time] = _node_matrix(
                    _probabilities_from_levels(per_node, levels, n=n, treatment_name=node),
                    node,
                )
                if treatment_columns and node in treatment_columns:
                    by_node_columns[time] = treatment_columns[node]
        else:
            values = np.array(treatment_probabilities, dtype=float, copy=True)
            if any(continuous_nodes):
                raise CapabilityError(CONTINUOUS_REFUSAL)
            if values.ndim == 2:
                if values.shape != (n, n_nodes):
                    raise DataError(
                        f"treatment_probabilities as a 2-D array is (n, T) = ({n}, {n_nodes}); "
                        f"got {values.shape}"
                    )
                binary = [len(levels) == 2 for levels in treatment_levels]
                if not all(binary):
                    raise DataError(
                        "treatment_probabilities as an (n, T) array reads P(A_t = levels_t[1]) "
                        "at binary nodes only; pass an (n, T, K) array or a mapping by node"
                    )
                for time in range(n_nodes):
                    column = values[:, time]
                    treatment[time] = _node_matrix(
                        np.column_stack([1.0 - column, column]), treatment_names[time]
                    )
            elif values.ndim == 3:
                widths = {len(levels) for levels in treatment_levels}
                if len(widths) != 1 or values.shape != (n, n_nodes, widths.pop()):
                    raise DataError(
                        "treatment_probabilities as a 3-D array is (n, T, K) with every node "
                        f"sharing K levels; got {values.shape}. Pass a mapping by node when "
                        "the nodes have different level sets"
                    )
                for time in range(n_nodes):
                    treatment[time] = _node_matrix(values[:, time, :], treatment_names[time])
            else:
                raise DataError(
                    "treatment_probabilities is an (n, T) array, an (n, T, K) array, or a "
                    f"mapping by node; got shape {values.shape}"
                )
    censoring: list[FloatArray | None] = [None] * len(censoring_names)
    censoring_by_node: list[str | None] = [None] * len(censoring_names)
    if censoring_probabilities is not None:
        if not censoring_names:
            raise DataError(
                "censoring_probabilities declares a censoring mechanism, and this fit has no "
                "censoring columns"
            )
        if isinstance(censoring_probabilities, Mapping):
            for node, column in censoring_probabilities.items():
                if node not in censoring_names:
                    raise DataError(
                        f"censoring_probabilities names {node!r}, which is not a censoring "
                        f"column; the censoring columns are {list(censoring_names)}"
                    )
                time = list(censoring_names).index(node)
                censoring[time] = _retention(np.asarray(column, dtype=float), node, n)
                if censoring_columns and node in censoring_columns:
                    censoring_by_node[time] = censoring_columns[node]
        else:
            values = np.array(censoring_probabilities, dtype=float, copy=True)
            if values.shape != (n, len(censoring_names)):
                raise DataError(
                    "censoring_probabilities as an array is (n, T_c) = "
                    f"({n}, {len(censoring_names)}); got {values.shape}"
                )
            for time, name in enumerate(censoring_names):
                censoring[time] = _retention(values[:, time], name, n)
    return KnownNodeMechanisms(
        treatment=tuple(treatment) if treatment_probabilities is not None else (),
        censoring=tuple(censoring) if censoring_probabilities is not None else (),
        source=source,
        treatment_columns=tuple(by_node_columns) if treatment_columns else (),
        censoring_columns=tuple(censoring_by_node) if censoring_columns else (),
    )


def _node_matrix(values: FloatArray, node: str) -> FloatArray:
    """Check one node's ``(n, K)`` matrix where it is finite."""
    matrix = np.asarray(values, dtype=float)
    rows = np.all(np.isfinite(matrix), axis=1)
    partial = np.any(np.isfinite(matrix), axis=1) & ~rows
    if np.any(partial):
        raise DataError(
            f"treatment_probabilities at node {node!r} is NaN in some but not all arms on "
            f"{int(np.count_nonzero(partial))} row(s); a row is either declared or not at risk"
        )
    finite = matrix[rows]
    # A node may assign an arm with certainty, as a SMART does for the units it does not
    # re-randomize.  A zero is then a statement about an arm the unit cannot take, and the
    # fit refuses it where an at-risk unit took that arm or a regimen assigns it.
    if np.any(finite < 0.0) or np.any(finite > 1.0):
        raise DataError(
            f"treatment_probabilities at node {node!r} must lie in [0, 1] where it is given"
        )
    if not np.allclose(finite.sum(axis=1), 1.0, rtol=0.0, atol=_ROW_SUM_TOLERANCE):
        raise DataError(f"each row of treatment_probabilities at node {node!r} must sum to one")
    return matrix


def _retention(values: FloatArray, node: str, n: int) -> FloatArray:
    """Check one censoring node's ``(n,)`` retention probability where it is finite."""
    column = np.asarray(values, dtype=float).reshape(-1)
    if column.shape[0] != n:
        raise DataError(
            f"censoring_probabilities[{node!r}] has {column.shape[0]} rows; expected {n}"
        )
    finite = column[np.isfinite(column)]
    if np.any(finite < 0.0) or np.any(finite > 1.0):
        raise DataError(f"censoring_probabilities at {node!r} must lie in [0, 1] where it is given")
    return column
