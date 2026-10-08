"""Treatment regimens: what a unit would have been given at every time point.

A *regime* (:mod:`cleverly.interventions`) says what one treatment decision would have
been.  A **regimen** says what the whole sequence would have been --
:math:`\\bar a = (a_1, \\ldots, a_T)` -- and it is the thing a longitudinal fit's
parameters are indexed by.  The two words are one letter apart and name different
objects, which is why this module spells the distinction out rather than reusing
:class:`~cleverly.interventions.base.Intervention`: an intervention is a density over
arms at *one* node, and a regimen is a plan across nodes.

Two kinds live here, and the difference between them is the whole reason this module is
not a tuple of floats.  A :class:`Regimen` assigns the same arm to everybody at each
node.  A :class:`DynamicRegimen` assigns node :math:`t`'s arm by a rule
:math:`d_t(H_t)` -- treat once the biomarker crosses a threshold, say -- so its
**followers are a covariate-dependent set that differs at every node**, and the rows each
sequential regression is fitted on move with the data rather than being a fixed slice.

Both answer :meth:`assignment`, which returns the same ``(n, T)`` object: for a static
regimen a *broadcast view* of its plan, costing nothing.  Everything downstream reads
that matrix and nothing reads a scalar arm, which is what keeps the static path
bit-for-bit what it was before rules existed rather than a second implementation of it.

A rule is handed :meth:`~cleverly.longitudinal.data.LongitudinalData.history_frame` --
:math:`[W, L_1, \\ldots, L_t]`, in the backend the data came from, and nothing else.
Not the outcome, because reading it is not an intervention; not the earlier treatments,
because under the regimen those are what the rule itself assigned, and passing them would
let a rule read the treatment of a unit that *deviated*.  The same restriction, enforced
the same way, as :func:`cleverly.interventions.base._covariate_frame` at one time point.

A rule can close over any estimate, such as a threshold at a sample mean, and no code can inspect a
closure.  So a :class:`DynamicRegimen` declares ``rule_kind``, and :func:`refuse_regimen_rules`
refuses a plan with a callable node unless it is ``"known"``. The declaration is the one that
:class:`~cleverly.interventions.Rule` carries.  A callable written inline in a ``regimens=`` mapping
carries no declaration, so a fit refuses it.

A node can also hold a **known policy density** :math:`q_t(\\cdot \\mid H_t)`, written as a
:class:`~cleverly.interventions.Stochastic` node.  Such a node draws the arm rather than
assigning it.  Its density function is handed
:meth:`~cleverly.longitudinal.data.LongitudinalData.policy_frame`, which adds the earlier
treatments to the history frame, because under a policy the earlier arms vary among the
units that remain on the plan.  The density is evaluated once, in :func:`resolve_plans`,
and a :class:`Plan` carries it.  At a policy node the plan's ``values`` column holds the
*observed* arm, so the mechanism is evaluated at the observed history, and
:meth:`Plan.masks` keeps the rows whose observed arm has positive policy probability.  A
policy density that is one-hot on every reachable row is the rule it equals, and
:func:`resolve_plans` resolves it as that rule.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

from .._declarations import FunctionKind
from .._typing import BoolArray, FloatArray, IntArray
from ..exceptions import CapabilityError, DataError
from ..interventions.base import (
    _RULE_DECLARATION,
    Stochastic,
    _as_array,
    check_regime_density,
    refuse_regime_densities,
)
from ..interventions.policy import (
    POLICY_TYPES,
    check_continuous_policy,
    discrete_assignments,
    lazy_frame,
    policy_branches,
    refuse_policy_declarations,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance, types only
    from ..interventions.policy import Policy
    from .data import LongitudinalData, RegimenMasks

__all__ = [
    "DynamicRegimen",
    "Plan",
    "Regimen",
    "RegimenSpec",
    "declares_mtp",
    "declares_policy",
    "refuse_regimen_rules",
    "resolve_plans",
    "resolve_regimens",
]

#: One node of a plan: a categorical label, a rule reading that node's history, a known
#: policy density over the node's levels, or a modified treatment policy that reads the
#: unit's own treatment at the node.
TreatmentLabel: TypeAlias = "str | np.str_ | bool | int | float | np.number"
RuleNode: TypeAlias = "TreatmentLabel | Callable[[Any], Any] | Stochastic | Policy"


def _is_mtp(node: object) -> bool:
    """Whether a plan node is a modified treatment policy."""
    return isinstance(node, POLICY_TYPES)


@dataclass(frozen=True)
class Regimen:
    """A static treatment plan: one arm per time point, under the user's own label.

    Attributes
    ----------
    label:
        What the reported parameter is named by -- ``ey_regimen[always]``.  Part of
        the estimand rather than decoration: two regimens are two different
        parameters, and the report has to be able to say which is which.
    values:
        The treatment label assigned at each time point, one per node.
    """

    label: str
    values: tuple[object, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise DataError(f"regimen {self.label!r} assigns no treatment at any time point")
        if any(isinstance(value, Stochastic) or _is_mtp(value) for value in self.values):
            raise DataError(
                f"regimen {self.label!r} holds a policy node. Write a plan with a "
                "policy node as DynamicRegimen(label, plan), or pass it in regimens= as a "
                "mapping value"
            )

    @property
    def n_times(self) -> int:
        return len(self.values)

    def at(self, time: int) -> object:
        """The arm assigned at ``time``, counted from one."""
        return self.values[time - 1]

    def assignment(self, data: LongitudinalData) -> Any:
        """The ``(n, T)`` matrix of assigned *labels*, as a broadcast view of the plan.

        A view rather than a copy, so reading a static regimen through the same matrix
        interface a rule needs allocates nothing.  ``object`` rather than ``float64``
        because a label is whatever the analyst's treatment column held -- a string as
        readily as a number -- and the dense codes a fit runs on are produced from this
        by :meth:`~cleverly.longitudinal.data.LongitudinalData.encode_assignment`, which
        reproduces the old float path exactly for a 0/1 node.  That is where "a static
        binary fit is unchanged bit for bit" is now delivered.
        """
        return np.broadcast_to(np.asarray(self.values, dtype=object), (data.n, self.n_times))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        plan = "/".join(str(value) for value in self.values)
        return f"Regimen({self.label!r}, {plan})"


@dataclass(frozen=True)
class DynamicRegimen:
    """A plan whose nodes may be *rules* rather than constants.

    ``rule_kind="known"`` declares that every callable node is a fixed rowwise function of
    one unit's available history, chosen independently of the analysis sample.  A rule
    that estimates a threshold, or otherwise aggregates across the sample, defines a
    different, data-adaptive target, and inference for it needs conditions this path does
    not check.  No code can inspect a closure, so the declaration is the check.
    :func:`refuse_regimen_rules` refuses ``None`` and ``"estimated"`` when the regimen is
    built, and ``LTMLE.fit`` refuses them again before any learner.

    .. code-block:: python

        DynamicRegimen("treat once L2 rises", (0, lambda h: h["L2"] > 0), rule_kind="known")

    A node can hold a known policy density instead, written as a
    :class:`~cleverly.interventions.Stochastic` node.  The unit then draws its arm at that
    node with probability :math:`q_t(a \\mid H_t)`:

    .. code-block:: python

        DynamicRegimen("mix", (Stochastic(q1, "q1", density_kind="known"), "low"))

    A rule node is handed the history frame, which holds no earlier treatment.  A rule that
    must read an arm drawn at an earlier policy node, such as "continue the arm drawn at the
    first node", is written as a one-hot :class:`~cleverly.interventions.Stochastic` node.
    That node reads the policy frame, and :func:`resolve_plans` resolves a one-hot density
    as the rule it equals.

    Parameters
    ----------
    label : str
        What the reported parameter is named by, exactly as for :class:`Regimen`.
    plan : tuple
        One entry per time point.  An entry is a categorical treatment label, meaning that
        label for everybody at that node, a callable :math:`d_t(H_t)` handed that node's
        history frame and returning one arm per row, or a
        :class:`~cleverly.interventions.Stochastic` node declared
        ``density_kind="known"``, whose density function is handed that node's policy frame
        and returns one probability per level.  Mixing them is the ordinary case: "treat at
        the first node, then keep treating only while the biomarker stays high" is a
        constant followed by a rule.  The regimen stores the plan as a tuple, so a list or
        an iterator is read once, when it is built.  A single label, callable or policy is
        not a plan and raises :class:`~cleverly.exceptions.DataError`.
    rule_kind : {"known", "estimated"} or None
        The declaration that every callable node of ``plan`` is a known function.  One
        declaration covers every node.  ``"known"`` is the one value a fit accepts for a
        plan with a callable node.  ``None``, the default, and ``"estimated"`` raise
        :class:`~cleverly.exceptions.CapabilityError` for such a plan, and a plan of
        labels and policy nodes alone is exempt.  Each policy node carries its own
        ``density_kind``.  Any other value raises
        :class:`~cleverly.exceptions.DataError`.  It is the last field, so
        ``DynamicRegimen(label, plan)`` keeps its positional order.

    Attributes
    ----------
    n_times : int
    has_policy : bool
    """

    label: str
    plan: tuple[RuleNode, ...]
    rule_kind: FunctionKind | None = None

    def __post_init__(self) -> None:
        # The plan is stored as a tuple before any check, so the check reads the nodes that
        # every later reader reads.  An iterator is read here, once, and never again.
        plan: Any = self.plan
        nodes = _plan_nodes(self.label, tuple(plan) if isinstance(plan, Iterator) else plan)
        if nodes is None:
            raise DataError(
                f"regimen {self.label!r} needs a plan with one entry per treatment node; got "
                f"{plan!r}. Write one rule for every node as (rule,) * T, with T the number "
                "of nodes"
            )
        object.__setattr__(self, "plan", nodes)
        if not self.plan:
            raise DataError(f"regimen {self.label!r} assigns no treatment at any time point")
        refuse_regimen_rules(self)

    @property
    def n_times(self) -> int:
        """The number of treatment nodes, one for each entry of ``plan``."""
        return len(self.plan)

    def is_rule(self, time: int) -> bool:
        """Whether ``time``'s arm is decided by a rule rather than declared.

        Parameters
        ----------
        time : int
            The treatment node, counted from one.

        Returns
        -------
        bool
            ``True`` when the entry of ``plan`` at ``time`` is callable.
        """
        node = self.plan[time - 1]
        return callable(node) and not _is_mtp(node)

    def is_mtp(self, time: int) -> bool:
        """Whether ``time``'s arm is a modified treatment policy of the unit's own treatment.

        Parameters
        ----------
        time : int
            The treatment node, counted from one.

        Returns
        -------
        bool
            ``True`` when the entry of ``plan`` at ``time`` is a policy object, such as
            :class:`~cleverly.interventions.Shift` or
            :class:`~cleverly.interventions.ModifiedPolicy`.
        """
        return _is_mtp(self.plan[time - 1])

    def is_policy(self, time: int) -> bool:
        """Whether ``time``'s arm is drawn from a known policy density.

        Parameters
        ----------
        time : int
            The treatment node, counted from one.

        Returns
        -------
        bool
            ``True`` when the entry of ``plan`` at ``time`` is a
            :class:`~cleverly.interventions.Stochastic` node.
        """
        return isinstance(self.plan[time - 1], Stochastic)

    @property
    def has_policy(self) -> bool:
        """Whether any node of ``plan`` is a stochastic policy or a modified treatment policy."""
        return any(isinstance(node, Stochastic) or _is_mtp(node) for node in self.plan)

    @property
    def has_mtp(self) -> bool:
        """Whether any node of ``plan`` is a modified treatment policy."""
        return any(_is_mtp(node) for node in self.plan)

    def assignment(self, data: LongitudinalData) -> Any:
        """Evaluate every node's rule and return the ``(n, T)`` arm matrix.

        Called **once** per fit, in :meth:`cleverly.longitudinal.LTMLE.fit`, and the
        matrix is what every mask, mechanism design and clever covariate then reads.
        Evaluating it once rather than at each use is not only cheaper: a rule that is
        not a deterministic function of the frame would otherwise let the follower masks
        disagree with the designs the mechanism was evaluated at, and the fit would be
        answering for no single regimen at all. The callable receives a frame for ergonomic
        vectorization, but its supported contract is rowwise: it must not read sample summaries
        or learn a rule from those rows.

        A rule is asked for an arm on every row that is still *in the study* before the
        node -- uncensored through ``t - 1``, and on a survival fit event-free through
        ``t - 1`` as well, since a unit that has had the event has no treatment decision
        at ``t`` for a rule to make.  Off that set the history is the zero fill
        :meth:`~cleverly.longitudinal.data.LongitudinalData.covariate_history` puts there,
        so whatever the rule returns is meaningless; it is replaced by a sentinel rather than
        validated, because such a row is masked out of every regression and every
        influence curve, and the only way it could still matter is by putting a ``nan``
        into a design matrix that a learner is called on.

        A policy node assigns no arm.  Its column holds the label each unit was *observed*
        to receive, because the mechanism of a policy plan is evaluated at the observed
        history.  :meth:`policy_density` evaluates the policy itself.

        It runs :func:`refuse_regimen_rules` before it calls any rule, because a modified
        regimen can reach it directly with a declaration this version refuses.

        Parameters
        ----------
        data : LongitudinalData
            The validated panel, which supplies each node's history frame and the rows
            still in the study.

        Returns
        -------
        ndarray
            ``(n, T)`` object array of treatment labels, one column per node.  A row that
            has left the study before a node holds ``None`` at that node.

        Raises
        ------
        CapabilityError
            If a node is callable and ``rule_kind`` is ``None`` or ``"estimated"``, or a
            policy node does not declare ``density_kind="known"``.
        DataError
            If ``rule_kind`` is not one of the three states, or a rule raises or returns
            other than one arm per row.
        """
        refuse_regimen_rules(self)
        columns = []
        for time, node in enumerate(self.plan, start=1):
            reachable = data.uncensored_through(time - 1) & data.event_free_through(time - 1)
            if isinstance(node, Stochastic) or _is_mtp(node):
                arms = data.observed_labels(time)
            elif callable(node):
                arms = self._evaluate(node, data, time, reachable)
            else:
                arms = np.full(data.n, node, dtype=object)
            columns.append(np.where(reachable, arms, None))
        return np.column_stack(columns)

    def policy_density(self, data: LongitudinalData, time: int) -> FloatArray:
        """Evaluate the policy node at ``time`` and check it before it is used.

        The density function is handed
        :meth:`~cleverly.longitudinal.data.LongitudinalData.policy_frame`.  It returns an
        ``(n, K)`` array whose columns follow ``data.treatment_levels[time - 1]``, or a
        dataframe whose columns are exactly those levels, which this reorders by level.  The
        label route exists because a permutation of the columns of an array is silent.

        The check reads the rows still in the study before the node: uncensored and
        event-free through ``time - 1``.  Every other row is set to a zero row, which no
        regression and no influence curve reads.

        Parameters
        ----------
        data : LongitudinalData
            The validated panel.
        time : int
            The policy node, counted from one.

        Returns
        -------
        ndarray
            ``(n, K)`` policy probabilities, zero on the rows that left the study.

        Raises
        ------
        CapabilityError
            If the node does not declare ``density_kind="known"``.
        DataError
            If ``time`` is not a policy node, if the density function raises, or if it
            returns other than a probability simplex per reachable row over the levels.
        """
        node = self.plan[time - 1]
        if not isinstance(node, Stochastic):
            raise DataError(f"node {time} of regimen {self.label!r} is not a policy node")
        refuse_regime_densities((node,))
        label = f"the policy at time {time} of regimen {self.label!r}"
        levels = tuple(data.treatment_levels[time - 1])
        name = data.treatment_names[time - 1]
        frame = data.policy_frame(time)
        try:
            returned = node.density_fn(frame)
        except Exception as error:
            raise DataError(
                f"{label} raised {type(error).__name__}: {error}. It is handed "
                "[W, L_1, ..., L_t] and the earlier treatments, which at time "
                f"{time} are {list(data.policy_names(time))}. A policy reading a covariate "
                "measured later can only be used from that node on, so pass a plan with one "
                "entry per node."
            ) from error
        if hasattr(returned, "columns") and not isinstance(returned, np.ndarray):
            returned = _frame_by_level(returned, levels, label=label, name=name)
        try:
            values = np.asarray(_as_array(returned), dtype=float)
        except (TypeError, ValueError) as error:
            raise DataError(f"{label} returned a value that is not a probability") from error
        expected = (data.n, len(levels))
        if values.shape != expected:
            raise DataError(
                f"{label} returned shape {values.shape}; expected {expected}. One column per "
                f"level of treatment column {name!r}, in the order {levels!r}."
            )
        reachable = data.uncensored_through(time - 1) & data.event_free_through(time - 1)
        check_regime_density(values[reachable], label=label)
        return np.where(reachable[:, None], values, 0.0)

    def _evaluate(
        self, rule: Callable[[Any], Any], data: LongitudinalData, time: int, reachable: Any
    ) -> Any:
        """One rule, called on its node's history frame and checked before it is used."""
        names = data.history_names(time)
        try:
            returned = rule(data.history_frame(time))
        except Exception as error:
            raise DataError(
                f"the rule at time {time} of regimen {self.label!r} raised "
                f"{type(error).__name__}: {error}. It is handed [W, L_1, ..., L_t], which "
                f"at time {time} is {list(names)} -- a rule reading a covariate measured "
                "later can only be used from that node on, so pass a plan with one entry "
                "per node rather than a single rule for all of them"
            ) from error
        arms = np.asarray(_as_array(returned), dtype=object)
        if arms.ndim != 1 or arms.shape[0] != data.n:
            raise DataError(
                f"the rule at time {time} of regimen {self.label!r} returned "
                f"{arms.shape} assignments for {data.n} rows; it must return one arm per row"
            )
        return arms

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"DynamicRegimen({self.label!r}, {describe_plan(self)})"


def _frame_by_level(frame: Any, levels: tuple[object, ...], *, label: str, name: str) -> Any:
    """A policy's dataframe as an ``(n, K)`` array in level order, or a ``DataError``.

    The columns must be exactly the node's levels, matched by ``==`` on the analyst's own
    labels.  A frame with any other column set is refused, because no order can be read
    from a column the node does not have.
    """
    columns = list(frame.columns)
    order: list[int] = []
    for level in levels:
        matches = [index for index, column in enumerate(columns) if _same_label(column, level)]
        if len(matches) != 1:
            break
        order.append(matches[0])
    if len(columns) != len(levels) or len(order) != len(levels) or len(set(order)) != len(levels):
        raise DataError(
            f"{label} returned columns {columns!r}. A frame must have exactly the levels "
            f"{list(levels)!r} of treatment column {name!r}, or return an array in that "
            "column order."
        )
    return np.column_stack([_as_array(frame[columns[index]]) for index in order])


def _same_label(column: object, level: object) -> bool:
    """Whether a frame column names a treatment level, by the label's own ``==``."""
    try:
        return bool(column == level)
    except (TypeError, ValueError):
        return False


#: A regimen of either kind.  A closed union rather than a Protocol: :func:`resolve_regimens`
#: is the only constructor, so there is no third-party regimen type to admit -- and an open
#: one would invite into this module the very extensions it refuses by name elsewhere.
RegimenSpec: TypeAlias = "Regimen | DynamicRegimen"


def describe_plan(regimen: RegimenSpec) -> str:
    """A plan as ``1/0`` for constants, ``d`` for a rule and ``q`` for a policy."""
    if isinstance(regimen, Regimen):
        return "/".join(str(value) for value in regimen.values)
    return "/".join(_describe_node(node) for node in regimen.plan)


def _describe_node(node: RuleNode) -> str:
    """One node of a plan: its arm, or the rule's or the policy's own name.

    A ``def``\\ -ed rule carries the name the analyst gave it, and reporting ``d:responds``
    rather than a bare ``d`` costs nothing and says which of two rules was run.  A lambda
    has no such name -- ``__name__`` is ``"<lambda>"``, which names nothing -- so it falls
    back to ``d`` and the plan fingerprint on the config is what tells two of them apart.
    A policy node prints ``q:`` and the name its :class:`~cleverly.interventions.Stochastic`
    carries.
    """
    if isinstance(node, Stochastic):
        return f"q:{node.name}" if node.name else "q"
    if _is_mtp(node):
        return f"m:{getattr(node, 'name', '')}"
    if not callable(node):
        return str(node)
    name = getattr(node, "__name__", "<lambda>")
    return "d" if name == "<lambda>" else f"d:{name}"


@dataclass(frozen=True)
class Plan:
    """A regimen together with the ``(n, T)`` arms it assigns *this* sample.

    The pair travels as one object so that nothing downstream can reach a rule and call
    it a second time: :func:`~cleverly.longitudinal.sequential.fit_mechanism` and
    :func:`~cleverly.longitudinal.sequential.fit_regimen` see arms, never callables.

    ``values`` holds the *dense codes* the container assigned, not the analyst's labels.
    The labels are recoverable from
    :attr:`~cleverly.longitudinal.data.LongitudinalData.treatment_levels`, which every
    reader of this plan already holds, and carrying a second ``(n, T)`` object array
    beside the codes would be a copy that only some of them kept in step.

    At a policy node ``values`` holds the code each unit was observed to receive, and
    ``policy`` holds the evaluated density.  So the mechanism is evaluated at the observed
    history, :meth:`arm` is the observed arm, and :meth:`intervention_density` reads the
    policy at that arm.  A plan with no policy node has ``policy=()``, and every reader then
    runs the code a deterministic plan runs.

    Parameters
    ----------
    regimen : Regimen or DynamicRegimen
        The resolved regimen.
    values : FloatArray
        ``(n, T)`` dense codes: the assigned arm at a label or rule node, and the observed
        arm at a policy node.
    policy : tuple of FloatArray or None
        One entry per node: the ``(n, K_t)`` policy density at a policy node and ``None``
        elsewhere.  Empty when the plan has no policy node.
    point_mass_nodes : tuple of int
        The policy nodes whose density was one-hot on every reachable row, and which were
        therefore resolved as the rule they equal.
    mtp_nodes : tuple of int
        The nodes that hold a modified treatment policy.  At such a node ``values`` holds
        the observed code or dose, and ``policy`` holds the ``(n, J)`` weights of the
        predictions the node carries: the levels the policy sends a unit to at a
        categorical node, and the randomizer's probabilities at a continuous one.
    policy_targets : tuple
        Per node, the ``(n,)`` dose of each branch at a continuous policy node, and
        ``None`` elsewhere.  Empty without one.
    mtp_assignments : tuple
        Per node, each branch of a categorical policy node as ``(probability, codes)``,
        with ``codes`` the ``(n, K)`` level map of the discrete formula, and ``None``
        elsewhere.  Empty without one.
    mtp_numerators : tuple
        Per node, the ratio numerator of a policy node once the mechanism is fitted, and
        ``None`` elsewhere.  Empty until :func:`attach_policy_numerators` sets it.

    Attributes
    ----------
    label : str
    has_policy : bool
    has_mtp : bool
    """

    regimen: RegimenSpec
    values: FloatArray
    policy: tuple[FloatArray | None, ...] = ()
    point_mass_nodes: tuple[int, ...] = ()
    #: The nodes, counted from one, that hold a modified treatment policy.
    mtp_nodes: tuple[int, ...] = ()
    #: Per node, the ``(n,)`` dose each branch of a continuous node's policy assigns, and
    #: ``None`` elsewhere.  Empty without a continuous policy node.
    policy_targets: tuple[tuple[FloatArray, ...] | None, ...] = ()
    #: Per node, each branch of a categorical node's policy as ``(probability, codes)``
    #: with ``codes`` the ``(n, K)`` level map of the discrete formula, and ``None``
    #: elsewhere.  Empty without a categorical policy node.
    mtp_assignments: tuple[tuple[tuple[float, IntArray], ...] | None, ...] = ()
    #: Per node, the ``(n,)`` ratio numerator of a policy node once the mechanism is fitted:
    #: :math:`g^d_t(A_t \mid H_t)` at a categorical node and :math:`r_t = g^d_t / g_t` at a
    #: continuous one.  Empty until :func:`attach_policy_numerators` sets it.
    mtp_numerators: tuple[FloatArray | None, ...] = ()

    @property
    def label(self) -> str:
        return self.regimen.label

    @property
    def has_mtp(self) -> bool:
        """Whether any node of this plan is a modified treatment policy."""
        return bool(self.mtp_nodes)

    def is_mtp_node(self, time: int) -> bool:
        """Whether node ``time``, counted from one, is a modified treatment policy.

        Parameters
        ----------
        time : int
            The treatment node, counted from one.

        Returns
        -------
        bool
            ``True`` at a modified treatment policy node.
        """
        return time in self.mtp_nodes

    def carry_targets(self, time: int) -> tuple[FloatArray, ...] | None:
        """The doses a continuous policy node's regression is predicted at, one per branch.

        Parameters
        ----------
        time : int
            The treatment node, counted from one.

        Returns
        -------
        tuple of FloatArray or None
            One ``(n,)`` dose per branch at a continuous policy node, ``None`` elsewhere.
        """
        if not self.policy_targets:
            return None
        return self.policy_targets[time - 1]

    @property
    def has_policy(self) -> bool:
        """Whether any node of this plan draws its arm from an evaluated policy density."""
        return any(entry is not None for entry in self.policy)

    def is_policy_node(self, time: int) -> bool:
        """Whether node ``time``, counted from one, carries an evaluated policy density.

        Parameters
        ----------
        time : int
            The treatment node, counted from one.

        Returns
        -------
        bool
            ``True`` at a policy node that was not resolved as a rule.
        """
        return bool(self.policy) and self.policy[time - 1] is not None

    def policy_at(self, time: int) -> FloatArray:
        """The ``(n, K_t)`` policy density at node ``time``.

        Parameters
        ----------
        time : int
            A policy node, counted from one.

        Returns
        -------
        FloatArray
            The density evaluated in :func:`resolve_plans`.
        """
        density = self.policy[time - 1] if self.policy else None
        if density is None:
            raise ValueError(f"node {time} of regimen {self.label!r} is not a policy node")
        return density

    def arm(self, time: int) -> FloatArray:
        """The arm this plan assigns each unit at ``time``, counted from one."""
        return self.values[:, time - 1]

    def intervention_density(self, data: LongitudinalData) -> FloatArray:
        r"""``(n, T)`` :math:`\pi_t(A_t \mid H_t)`, the plan's density at the observed arm.

        At a label or rule node it is :math:`1\{A_t = d_t(H_t)\}`.  At a policy node it is
        :math:`q_t(A_t \mid H_t)`.  A row with no treatment at a node holds zero there.

        Parameters
        ----------
        data : LongitudinalData
            The panel the plan was resolved against.

        Returns
        -------
        FloatArray
            One column per node.
        """
        return np.column_stack(
            [_node_numerator(self, data, time) for time in range(1, data.n_times + 1)]
        )

    def support(self, data: LongitudinalData) -> BoolArray:
        r"""``(n, T)`` indicator that a row stays on the plan at each node.

        At a label or rule node it is :math:`1\{A_t = d_t(H_t)\}`, at a stochastic policy
        node :math:`q_t(A_t \mid H_t) > 0`, and at a modified treatment policy node every
        row with a treatment there: the policy reads the unit's own treatment, so no row
        leaves the plan, and a row whose ratio is zero keeps a zero term.

        Parameters
        ----------
        data : LongitudinalData
            The panel the plan was resolved against.

        Returns
        -------
        BoolArray
            Booleans as one column per node.
        """
        columns = []
        for time in range(1, data.n_times + 1):
            if self.is_mtp_node(time):
                columns.append(~np.isnan(data.treatment[:, time - 1]))
            else:
                columns.append(_node_numerator(self, data, time) > 0.0)
        return np.column_stack(columns)

    def cumulative_numerator(self, data: LongitudinalData) -> FloatArray | None:
        r"""``(n, T)`` running product :math:`\prod_{s \le t} \pi_s(A_s \mid H_s)`, or ``None``.

        The numerator of the cumulative ratio :math:`R_t`.  ``None`` on a plan without a
        policy node, whose ratio numerator is the indicator that the masks already apply.

        Parameters
        ----------
        data : LongitudinalData
            The panel the plan was resolved against.

        Returns
        -------
        FloatArray or None
            One column per node, or ``None`` without a policy node.
        """
        if not self.has_policy:
            return None
        density = self.intervention_density(data)
        running = np.ones(data.n)
        columns = []
        for time in range(data.n_times):
            running = running * density[:, time]
            columns.append(running)
        return np.column_stack(columns)

    def masks(self, data: LongitudinalData) -> RegimenMasks:
        """The plan's prefix masks: support-consistent, uncensored and event-free.

        Without a policy node this is ``data.regimen_masks(values)`` itself, so a
        deterministic plan keeps its exact arrays.  With one, a row follows the plan
        through ``t`` when its observed arm has positive intervention density at every
        node up to ``t``.  At a label or rule node that is the follower indicator, and at
        a policy node it keeps every row whose observed arm the policy can draw.

        Parameters
        ----------
        data : LongitudinalData
            The panel the plan was resolved against.

        Returns
        -------
        RegimenMasks
            The prefix scans every node of the recursion reads.
        """
        if not self.has_policy:
            return data.regimen_masks(self.values)
        return data.regimen_masks(self.values, support=self.support(data))


def _node_numerator(plan: Plan, data: LongitudinalData, time: int) -> FloatArray:
    r""":math:`\pi_t(A_t \mid H_t)` at the observed arm, the ratio numerator at node ``time``.

    The one place the numerator is read.  At a policy node it reads the evaluated density
    at the observed code, and elsewhere the indicator that the observed arm is the assigned
    one.  A row with no treatment at the node, censored or past an event, holds zero.

    Parameters
    ----------
    plan : Plan
        The resolved plan.
    data : LongitudinalData
        The panel the plan was resolved against.
    time : int
        The treatment node, counted from one.

    Returns
    -------
    FloatArray
        One value per row.
    """
    observed = data.treatment[:, time - 1]
    present = ~np.isnan(observed)
    if plan.is_mtp_node(time):
        numerator = plan.mtp_numerators[time - 1] if plan.mtp_numerators else None
        if numerator is None:
            raise ValueError(
                f"node {time} of regimen {plan.label!r} is a modified treatment policy, whose "
                "ratio numerator reads the fitted mechanism; attach it first"
            )
        return np.where(present, numerator, 0.0)
    if plan.is_policy_node(time):
        codes = np.nan_to_num(observed, nan=0.0).astype(np.int64)
        density = plan.policy_at(time)
        return np.where(present, density[np.arange(data.n), codes], 0.0)
    return np.asarray(observed == plan.values[:, time - 1], dtype=float)


def resolve_plans(
    regimens: Sequence[RegimenSpec],
    data: LongitudinalData,
    *,
    _collapse_point_masses: bool = True,
) -> tuple[Plan, ...]:
    """Evaluate every regimen against ``data``, once, before any nuisance is fitted.

    Each rule and each policy density is called here and nowhere else.  A policy density
    that is one-hot on every reachable row, by exact equality, is resolved as the rule it
    equals: its ``values`` column holds the argmax codes, and its ``policy`` entry is
    ``None``.  The node is recorded on :attr:`Plan.point_mass_nodes`.  A plan whose every
    policy node collapses runs exactly the code its rule regimen runs.

    Parameters
    ----------
    regimens : sequence of Regimen or DynamicRegimen
        The resolved regimens, in report order.
    data : LongitudinalData
        The validated panel.
    _collapse_point_masses : bool, default=True
        Private.  ``False`` keeps a one-hot policy on the policy path, which a test uses to
        check that path against the rule it equals.

    Returns
    -------
    tuple of Plan
        One per regimen, in the order given.
    """
    return tuple(
        _resolve_plan(regimen, data, collapse=_collapse_point_masses) for regimen in regimens
    )


def _resolve_plan(regimen: RegimenSpec, data: LongitudinalData, *, collapse: bool) -> Plan:
    """One regimen's plan, with its policy densities evaluated and point masses collapsed."""
    refuse_continuous_assignments(regimen, data)
    values = data.encode_assignment(regimen.assignment(data), regimen.label)
    if not (isinstance(regimen, DynamicRegimen) and regimen.has_policy):
        return Plan(regimen, values)
    values = np.array(values, dtype=float, copy=True)
    policy: list[FloatArray | None] = []
    collapsed: list[int] = []
    mtp_nodes: list[int] = []
    targets: list[tuple[FloatArray, ...] | None] = []
    assignments: list[tuple[tuple[float, IntArray], ...] | None] = []
    for time in range(1, data.n_times + 1):
        if regimen.is_mtp(time):
            carry, target, assigned = _resolve_mtp_node(regimen, data, time)
            policy.append(carry)
            targets.append(target)
            assignments.append(assigned)
            mtp_nodes.append(time)
            continue
        targets.append(None)
        assignments.append(None)
        if not regimen.is_policy(time):
            policy.append(None)
            continue
        density = regimen.policy_density(data, time)
        reachable = data.uncensored_through(time - 1) & data.event_free_through(time - 1)
        if collapse and _is_point_mass(density[reachable]):
            values[:, time - 1] = np.where(reachable, np.argmax(density, axis=1), 0.0)
            policy.append(None)
            collapsed.append(time)
        else:
            policy.append(density)
    kept = tuple(policy) if any(entry is not None for entry in policy) else ()
    return Plan(
        regimen,
        values,
        kept,
        tuple(collapsed),
        mtp_nodes=tuple(mtp_nodes),
        policy_targets=tuple(targets) if any(t is not None for t in targets) else (),
        mtp_assignments=tuple(assignments) if any(a is not None for a in assignments) else (),
    )


def _resolve_mtp_node(
    regimen: DynamicRegimen, data: LongitudinalData, time: int
) -> tuple[FloatArray, tuple[FloatArray, ...] | None, tuple[tuple[float, IntArray], ...] | None]:
    r"""Evaluate a modified treatment policy at one node, once, before any learner.

    Returns the ``(n, J)`` carry weights, and either the ``(n,)`` dose of each branch at a
    continuous node or each branch's ``(n, K)`` level map at a categorical one.  The policy
    reads the policy frame, :math:`[W, L_1, \ldots, L_t]` and the earlier treatments, which
    hold the observed values on which the regression conditions.  Rows that left the study
    before the node hold zero weight and a zero dose.
    """
    node = regimen.plan[time - 1]
    name = data.treatment_names[time - 1]
    where = f"node {name!r} at time {time} of regimen {regimen.label!r}"
    reachable = data.uncensored_through(time - 1) & data.event_free_through(time - 1)
    frame = lazy_frame(lambda: data.policy_frame(time))
    branches = policy_branches(node)
    probabilities = np.array([probability for probability, _ in branches])
    if data.is_continuous_node(time):
        observed = np.nan_to_num(data.treatment[:, time - 1], nan=0.0)
        check_continuous_policy(node, observed, frame, reachable, where=where)
        doses = []
        for _, branch in branches:
            assigned = np.asarray(branch.assign(observed, frame), dtype=float).reshape(-1)
            doses.append(np.where(reachable, assigned, 0.0))
        carry = np.where(reachable[:, None], np.tile(probabilities, (data.n, 1)), 0.0)
        return carry, tuple(doses), None
    levels = data.treatment_levels[time - 1]
    assigned_codes = discrete_assignments(node, levels, frame, reachable, where=where)
    observed_codes = np.nan_to_num(data.treatment[:, time - 1], nan=0.0).astype(np.int64)
    carry = np.zeros((data.n, len(levels)))
    rows = np.arange(data.n)
    for probability, codes in assigned_codes:
        np.add.at(carry, (rows, codes[rows, observed_codes]), probability)
    carry = np.where(reachable[:, None], carry, 0.0)
    return carry, None, assigned_codes


def refuse_continuous_assignments(regimen: RegimenSpec, data: LongitudinalData) -> None:
    """Refuse a label, rule or known density at a continuous node, before any learner.

    A static dose on a continuous treatment is not pathwise differentiable (Díaz, Williams,
    Hoffman and Schenck 2023, Section 4), and neither is a rule that sets one, so a
    continuous node takes a modified treatment policy only.

    Parameters
    ----------
    regimen : Regimen or DynamicRegimen
        The resolved regimen.
    data : LongitudinalData
        The validated panel.

    Raises
    ------
    CapabilityError
        If a continuous node holds anything other than a policy object.
    """
    if not data.has_continuous_node:
        return
    nodes = regimen.values if isinstance(regimen, Regimen) else regimen.plan
    for time, node in enumerate(nodes, start=1):
        if data.is_continuous_node(time) and not _is_mtp(node):
            raise CapabilityError(
                f"regimen {regimen.label!r} sets the continuous node "
                f"{data.treatment_names[time - 1]!r} to a fixed dose or rule. A static dose on "
                "a continuous treatment is not pathwise differentiable (Díaz et al. 2023, "
                "Section 4). Declare a modified treatment policy at that node"
            )


def attach_policy_numerators(
    plans: Sequence[Plan], numerators: Mapping[str, dict[int, FloatArray]]
) -> tuple[Plan, ...]:
    """Each plan with its policy nodes' ratio numerators, once the mechanism is fitted.

    Parameters
    ----------
    plans : sequence of Plan
        The resolved plans.
    numerators : mapping
        By plan label, by node counted from one, the ``(n,)`` numerator.

    Returns
    -------
    tuple of Plan
        The plans, unchanged where they hold no modified treatment policy.
    """
    from dataclasses import replace

    out = []
    for plan in plans:
        if not plan.has_mtp:
            out.append(plan)
            continue
        by_node = numerators[plan.label]
        n_times = int(plan.values.shape[1])
        out.append(
            replace(
                plan,
                mtp_numerators=tuple(by_node.get(time) for time in range(1, n_times + 1)),
            )
        )
    return tuple(out)


def _is_point_mass(density: FloatArray) -> bool:
    """Whether every row is one-hot, by exact equality and with no tolerance."""
    ones = density == 1.0
    return bool(np.all(ones | (density == 0.0)) and np.all(ones.sum(axis=1) == 1))


def refuse_regimen_rules(regimens: Any) -> None:
    """Raise unless every plan among ``regimens`` with a callable node declares it known.

    A rule can close over any estimate, and no code can inspect a closure, so the status
    of each callable node is the ``rule_kind`` of its :class:`DynamicRegimen`.  The table
    gives the plans this reads from each shape of the ``regimens=`` argument.  Any other
    value holds no plan, and :func:`resolve_regimens` raises its own error for it.

    ==============================================  ============
    ``regimens``                                    plans
    ==============================================  ============
    a :class:`Regimen` or :class:`DynamicRegimen`   that regimen
    a mapping                                       its values
    a sequence other than a ``str``                 its items
    ==============================================  ============

    The declaration of a plan is the ``rule_kind`` of a :class:`DynamicRegimen`, and
    ``None`` for any other plan.  So a callable written inline in a mapping, or held by a
    :class:`Regimen`, is undeclared.  The check of each plan runs in order:

    1. A plan that is an iterator, such as a generator, is a
       :class:`~cleverly.exceptions.DataError`.  Reading it would consume it, and a fit
       reads its ``regimens=`` each time it runs.
    2. A declaration outside ``"known"``, ``"estimated"`` and ``None`` is a
       :class:`~cleverly.exceptions.DataError`, whatever the plan holds.
    3. A plan with a callable node and a declaration of ``None`` or ``"estimated"`` is a
       :class:`~cleverly.exceptions.CapabilityError`.  A plan of labels alone passes.

    This check and :func:`resolve_regimens` read the shape of a plan through one helper,
    so they cannot read one plan two ways.  :class:`DynamicRegimen` runs this when it is
    built and before :meth:`DynamicRegimen.assignment` calls a rule.  ``LTMLE.fit`` runs
    it on the raw ``regimens=`` before any other check of the data and before any learner,
    and :func:`~cleverly.longitudinal.estimator.longitudinal_truncation_curve` runs it on
    the resolved regimens of a result.  A regimen changed with ``object.__setattr__`` can
    carry a declaration this version refuses.

    Parameters
    ----------
    regimens : Any
        The ``regimens=`` argument of a fit, or the resolved regimens of a result.

    Raises
    ------
    DataError
        If a plan is an iterator, or a declaration is not one of the three states.
    CapabilityError
        If a plan with a callable node declares ``None`` or ``"estimated"``.
    """
    for label, plan in _plans(regimens):
        # Typed ``object`` on purpose: this checks what a modified regimen holds at run
        # time, which its annotations do not guarantee.
        kind: object = plan.rule_kind if isinstance(plan, DynamicRegimen) else None
        nodes = _plan_nodes(label, plan)
        entries = (plan,) if nodes is None else nodes
        if any(callable(node) and not _is_mtp(node) for node in entries):
            # ``refuse`` runs ``check`` first, so every refusal of a rule is the shared one.
            _RULE_DECLARATION.refuse(kind)
        else:
            _RULE_DECLARATION.check(kind)
        # Each policy node carries its own declaration, which the point-treatment
        # ``Stochastic`` check reads, so the two cannot refuse one policy differently.
        refuse_regime_densities(node for node in entries if isinstance(node, Stochastic))
        # A modified treatment policy is known by its class, or declares policy_kind.
        refuse_policy_declarations(node for node in entries if _is_mtp(node))


def declares_policy(regimens: Any) -> bool:
    """Whether any plan of a ``regimens=`` argument holds a known policy node.

    Reads the plans by the table of :func:`refuse_regimen_rules`, and refuses nothing: a
    plan this cannot read holds no policy node.

    Parameters
    ----------
    regimens : Any
        The ``regimens=`` argument of a fit or of a longitudinal estimand.

    Returns
    -------
    bool
        ``True`` when some node of some plan is a
        :class:`~cleverly.interventions.Stochastic` node.
    """
    for label, plan in _plans(regimens):
        try:
            nodes = _plan_nodes(label, plan)
        except DataError:
            continue
        if any(isinstance(node, Stochastic) for node in ((plan,) if nodes is None else nodes)):
            return True
    return False


def declares_mtp(regimens: Any) -> bool:
    """Whether any plan of a ``regimens=`` argument holds a modified treatment policy.

    Parameters
    ----------
    regimens : Any
        The ``regimens=`` argument of a fit or of a longitudinal estimand.

    Returns
    -------
    bool
        ``True`` when some node of some plan is a policy object, such as
        :class:`~cleverly.interventions.Shift`.
    """
    for label, plan in _plans(regimens):
        try:
            nodes = _plan_nodes(label, plan)
        except DataError:
            continue
        if any(_is_mtp(node) for node in ((plan,) if nodes is None else nodes)):
            return True
    return False


def _plans(regimens: Any) -> tuple[tuple[object, Any], ...]:
    """The labelled plans of ``regimens=``, by the table of :func:`refuse_regimen_rules`.

    A mapping labels each plan by its key.  Any other plan carries its own label, or
    ``None`` when it is not a regimen, and :func:`resolve_regimens` refuses that plan.
    """
    if isinstance(regimens, (Regimen, DynamicRegimen)):
        return ((regimens.label, regimens),)
    if isinstance(regimens, Mapping):
        return tuple(regimens.items())
    if isinstance(regimens, Sequence) and not isinstance(regimens, (str, bytes)):
        return tuple((getattr(plan, "label", None), plan) for plan in regimens)
    return ()


#: The scalar types that one entry broadcasts across every node, as one treatment label.
_LABEL_TYPES = (bool, int, float, str, np.str_, np.number)


def _plan_nodes(label: object, plan: Any) -> tuple[Any, ...] | None:
    """Read the shape of one plan: its nodes, or ``None`` for a plan of one entry.

    :class:`DynamicRegimen`, :func:`refuse_regimen_rules` and :func:`resolve_regimens`
    read the shape of a plan here and nowhere else, so the check and the resolver cannot
    disagree about the nodes of one plan.

    ======================================================  ===========================
    ``plan``                                                reads as
    ======================================================  ===========================
    a :class:`Regimen`                                      its ``values``
    a :class:`DynamicRegimen`                               its ``plan``
    a callable, a treatment label, or no ``__iter__``       ``None``
    an iterator, such as a generator                        a ``DataError``
    any other iterable                                      a tuple of its items
    ======================================================  ===========================

    ``None`` leaves the one entry to the caller, which broadcasts a callable or a label
    across the nodes and refuses anything else.  An iterator is refused, not read: reading
    it would consume it, and a fit reads its ``regimens=`` again at the next call.
    :class:`DynamicRegimen` reads an iterator into a tuple before it calls this, because it
    stores its plan.
    """
    if isinstance(plan, Regimen):
        return tuple(plan.values)
    if isinstance(plan, DynamicRegimen):
        return tuple(plan.plan)
    if isinstance(plan, Mapping):
        raise DataError(
            f"the plan of regimen {label!r} is a mapping. Pass its treatment nodes as a "
            "tuple in time order, or use DynamicRegimen(label, ordered_nodes, rule_kind='known') "
            "for a plan with a rule"
        )
    if isinstance(plan, np.ndarray) and plan.ndim == 0:
        raise DataError(
            f"the plan of regimen {label!r} is a zero-dimensional array. Pass a treatment "
            "label to assign it at every node, a sequence with one entry per node, or "
            "DynamicRegimen(label, ordered_nodes, rule_kind='known') for a plan with a rule"
        )
    if (
        callable(plan)
        or _is_mtp(plan)
        or isinstance(plan, _LABEL_TYPES)
        or not hasattr(plan, "__iter__")
    ):
        return None
    if isinstance(plan, Iterator):
        raise DataError(
            f"the plan of regimen {label!r} is an iterator, {type(plan).__name__}. A fit reads "
            "regimens= each time it runs, and an iterator is empty after its first read. "
            "Pass the plan as a tuple, or as a DynamicRegimen, which stores its plan"
        )
    return tuple(plan)


def resolve_regimens(spec: Any, n_times: int) -> tuple[RegimenSpec, ...]:
    """Turn a user's ``regimens=`` argument into an ordered tuple of regimens.

    Accepts a mapping from label to plan, where a plan is a single arm meaning "that arm
    at every node", or a sequence of ``n_times`` arms.  A mapping value may also be a
    :class:`Regimen` or a :class:`DynamicRegimen`, and the key is then its label.  A
    sequence of :class:`Regimen` or :class:`DynamicRegimen` objects passes through.  A
    plan that is an iterator, such as a generator, is refused: a fit reads its
    ``regimens=`` each time it runs, and an iterator is empty after its first read.

    A plan with a callable node must be a :class:`DynamicRegimen` declared
    ``rule_kind="known"``.  A callable written inline in a mapping, as one rule for every
    node or as a node of a sequence, carries no declaration.  Its regimen is built with
    ``rule_kind=None``, and :func:`refuse_regimen_rules` refuses it.
    A resolved :class:`DynamicRegimen` keeps the ``rule_kind`` of the one it was given.
    A :class:`~cleverly.interventions.Stochastic` node is a known policy at its node, and a
    mapping value that is one such node is that policy at every node.  Each one carries its own
    ``density_kind``, which :func:`refuse_regimen_rules` checks.

    A plan with no rule and no policy in it comes back a :class:`Regimen`, which is what keeps
    a static fit on exactly the code path it was on before rules existed.

    Order is preserved, because the first regimen is the one contrasts are taken
    against by default and so is part of what the fit reports.

    Parameters
    ----------
    spec : Any
        The ``regimens=`` argument: a mapping from label to plan, one regimen, or a
        sequence of regimens.
    n_times : int
        The number of treatment nodes, which a single arm is broadcast across.

    Returns
    -------
    tuple of Regimen or DynamicRegimen
        The regimens in the order given.

    Raises
    ------
    DataError
        If ``spec`` is missing, empty or malformed, if a label repeats, if a plan is an
        iterator or has the wrong number of nodes, or if a declaration is not one of the
        three states.
    CapabilityError
        If a plan with a callable node is not declared ``rule_kind="known"``.
    """
    if spec is None:
        raise DataError(
            "a longitudinal fit needs regimens= : the parameter is the mean outcome under "
            "a treatment plan, and there is no default plan to fall back on. Pass, for "
            "example, regimens={'always': 1, 'never': 0}"
        )
    if isinstance(spec, (Regimen, DynamicRegimen)):
        spec = (spec,)
    if isinstance(spec, Mapping):
        items: list[tuple[str, Any]] = list(spec.items())
    elif isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
        items = []
        for entry in spec:
            if not isinstance(entry, (Regimen, DynamicRegimen)):
                raise DataError(
                    "a sequence of regimens must hold Regimen or DynamicRegimen objects; "
                    "pass a mapping {label: plan} to name plans inline"
                )
            items.append((entry.label, entry))
    else:
        raise DataError(f"regimens= must be a mapping or a sequence of Regimen; got {spec!r}")

    if not items:
        raise DataError("regimens= is empty; a fit with no regimen reports no parameter")

    resolved: list[RegimenSpec] = []
    seen: set[str] = set()
    for label, plan in items:
        name = str(label)
        if name in seen:
            raise DataError(f"regimen label {name!r} appears twice; labels name parameters")
        seen.add(name)
        resolved.append(_resolve_one(name, plan, n_times))
    return tuple(resolved)


def _resolve_one(label: str, plan: Any, n_times: int) -> RegimenSpec:
    """Read one plan into the regimen kind it describes.

    A rebuilt :class:`DynamicRegimen` carries the ``rule_kind`` of the one it was given,
    and ``None`` for any other plan, so its construction checks the declaration.
    """
    kind: FunctionKind | None = plan.rule_kind if isinstance(plan, DynamicRegimen) else None
    nodes = _nodes(label, plan, n_times)
    if any(callable(node) or isinstance(node, Stochastic) or _is_mtp(node) for node in nodes):
        return DynamicRegimen(label, nodes, rule_kind=kind)
    return Regimen(label, nodes)


def _nodes(label: str, plan: Any, n_times: int) -> tuple[RuleNode, ...]:
    """Read one plan into one entry per node, broadcasting a single entry across them."""
    nodes = _plan_nodes(label, plan)
    if nodes is None:
        # A numpy array and a pandas Series are plans by every reading except
        # ``isinstance(..., Sequence)``, which neither registers for.  ``_plan_nodes`` tests
        # for the iteration protocol instead, so the message about rules is not aimed at
        # an array whose diagnosis it gets wrong.
        if not (callable(plan) or _is_mtp(plan) or isinstance(plan, (Stochastic, *_LABEL_TYPES))):
            raise DataError(
                f"regimen {label!r} must be a treatment label, a rule d_t(H_t), a known "
                f"policy Stochastic(q, name, density_kind='known'), a modified treatment "
                f"policy such as Shift(0.5, cap=4.0), or a sequence of {n_times} of these; "
                f"got {plan!r}"
            )
        # A rule or a policy gets the broadcast that a scalar arm gets.  So a rule that reads a
        # late-measured covariate is diagnosed at evaluation rather than here: whether
        # ``lambda h: h["L2"] > 0`` is usable at node 1 is a question about the data.
        return (plan,) * n_times
    if len(nodes) != n_times:
        raise DataError(
            f"regimen {label!r} assigns {len(nodes)} arm(s) but the data has {n_times} "
            "treatment node(s); a plan must say what happens at every one of them"
        )
    return nodes
