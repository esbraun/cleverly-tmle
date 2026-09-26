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

A rule can close over any estimate, such as a threshold at a sample mean, and no code can
inspect a closure.  So a :class:`DynamicRegimen` declares ``rule_kind``, and
:func:`refuse_regimen_rules` refuses a plan with a callable node unless it is ``"known"``
(roadmap row RM28).  The declaration is the one that :class:`~cleverly.interventions.Rule`
carries.  A callable written inline in a ``regimens=`` mapping carries no declaration, so a
fit refuses it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

from .._declarations import FunctionKind
from .._typing import FloatArray
from ..exceptions import DataError
from ..interventions.base import _RULE_DECLARATION, _as_array

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance, types only
    from .data import LongitudinalData

__all__ = [
    "DynamicRegimen",
    "Plan",
    "Regimen",
    "RegimenSpec",
    "refuse_regimen_rules",
    "resolve_plans",
    "resolve_regimens",
]

#: One node of a plan: a categorical label, or a rule reading that node's history.
TreatmentLabel: TypeAlias = "str | np.str_ | bool | int | float | np.number"
RuleNode: TypeAlias = "TreatmentLabel | Callable[[Any], Any]"


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
    built, and ``LTMLE.fit`` refuses them again before any learner (roadmap row RM28).

    .. code-block:: python

        DynamicRegimen("treat once L2 rises", (0, lambda h: h["L2"] > 0), rule_kind="known")

    Parameters
    ----------
    label : str
        What the reported parameter is named by, exactly as for :class:`Regimen`.
    plan : tuple
        One entry per time point.  An entry is either a categorical treatment label,
        meaning that label for everybody at that node, or a callable :math:`d_t(H_t)`
        handed that node's history frame and returning one arm per row.  Mixing the two
        is the ordinary case: "treat at the first node, then keep treating only while
        the biomarker stays high" is a constant followed by a rule.  The regimen stores
        the plan as a tuple, so a list or an iterator is read once, when it is built.  A
        single label or callable is not a plan and raises
        :class:`~cleverly.exceptions.DataError`.
    rule_kind : {"known", "estimated"} or None
        The declaration that every callable node of ``plan`` is a known function.  One
        declaration covers every node.  ``"known"`` is the one value a fit accepts for a
        plan with a callable node.  ``None``, the default, and ``"estimated"`` raise
        :class:`~cleverly.exceptions.CapabilityError` for such a plan, and a plan of
        labels alone is exempt.  Any other value raises
        :class:`~cleverly.exceptions.DataError`.  It is the last field, so
        ``DynamicRegimen(label, plan)`` keeps its positional order.

    Attributes
    ----------
    n_times : int
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
        return callable(self.plan[time - 1])

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
            If a node is callable and ``rule_kind`` is ``None`` or ``"estimated"``.
        DataError
            If ``rule_kind`` is not one of the three states, or a rule raises or returns
            other than one arm per row.
        """
        refuse_regimen_rules(self)
        columns = []
        for time, node in enumerate(self.plan, start=1):
            reachable = data.uncensored_through(time - 1) & data.event_free_through(time - 1)
            if callable(node):
                arms = self._evaluate(node, data, time, reachable)
            else:
                arms = np.full(data.n, node, dtype=object)
            columns.append(np.where(reachable, arms, None))
        return np.column_stack(columns)

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


#: A regimen of either kind.  A closed union rather than a Protocol: :func:`resolve_regimens`
#: is the only constructor, so there is no third-party regimen type to admit -- and an open
#: one would invite into this module the very extensions it refuses by name elsewhere.
RegimenSpec: TypeAlias = "Regimen | DynamicRegimen"


def describe_plan(regimen: RegimenSpec) -> str:
    """A plan as ``1/0`` for constants and ``d`` for a rule, for the settings report."""
    if isinstance(regimen, Regimen):
        return "/".join(str(value) for value in regimen.values)
    return "/".join(_describe_node(node) for node in regimen.plan)


def _describe_node(node: RuleNode) -> str:
    """One node of a plan: its arm, or the rule's own name when it has one.

    A ``def``\\ -ed rule carries the name the analyst gave it, and reporting ``d:responds``
    rather than a bare ``d`` costs nothing and says which of two rules was run.  A lambda
    has no such name -- ``__name__`` is ``"<lambda>"``, which names nothing -- so it falls
    back to ``d`` and the plan fingerprint on the config is what tells two of them apart.
    """
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
    """

    regimen: RegimenSpec
    values: FloatArray

    @property
    def label(self) -> str:
        return self.regimen.label

    def arm(self, time: int) -> FloatArray:
        """The arm this plan assigns each unit at ``time``, counted from one."""
        return self.values[:, time - 1]


def resolve_plans(regimens: Sequence[RegimenSpec], data: LongitudinalData) -> tuple[Plan, ...]:
    """Evaluate every regimen against ``data``, once, before any nuisance is fitted."""
    return tuple(
        Plan(regimen, data.encode_assignment(regimen.assignment(data), regimen.label))
        for regimen in regimens
    )


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
        if any(callable(node) for node in ((plan,) if nodes is None else nodes)):
            # ``refuse`` runs ``check`` first, so every refusal of a rule is the shared one.
            _RULE_DECLARATION.refuse(kind)
        else:
            _RULE_DECLARATION.check(kind)


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
            "label to assign it at every node, or a sequence with one entry per node"
        )
    if callable(plan) or isinstance(plan, _LABEL_TYPES) or not hasattr(plan, "__iter__"):
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
    ``rule_kind=None``, and :func:`refuse_regimen_rules` refuses it (roadmap row RM28).
    A resolved :class:`DynamicRegimen` keeps the ``rule_kind`` of the one it was given.

    A plan with no rule in it comes back a :class:`Regimen`, which is what keeps a static
    fit on exactly the code path it was on before rules existed.

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
    if any(callable(node) for node in nodes):
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
        if not (callable(plan) or isinstance(plan, _LABEL_TYPES)):
            raise DataError(
                f"regimen {label!r} must be a treatment label, a rule d_t(H_t), or a "
                f"sequence of {n_times} of either; got {plan!r}"
            )
        # A rule gets the broadcast that a scalar arm gets.  So a rule that reads a
        # late-measured covariate is diagnosed at evaluation rather than here: whether
        # ``lambda h: h["L2"] > 0`` is usable at node 1 is a question about the data.
        return (plan,) * n_times
    if len(nodes) != n_times:
        raise DataError(
            f"regimen {label!r} assigns {len(nodes)} arm(s) but the data has {n_times} "
            "treatment node(s); a plan must say what happens at every one of them"
        )
    return nodes
