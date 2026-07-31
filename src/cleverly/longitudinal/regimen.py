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
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

from .._typing import FloatArray
from ..exceptions import DataError
from ..interventions.base import _as_array

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance, types only
    from .data import LongitudinalData

__all__ = [
    "DynamicRegimen",
    "Plan",
    "Regimen",
    "RegimenSpec",
    "resolve_plans",
    "resolve_regimens",
]

#: One node of a plan: a constant arm, or a rule reading that node's history.
RuleNode: TypeAlias = "float | Callable[[Any], Any]"


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
        The arm assigned at each time point, ``0.0`` or ``1.0``, one per node.
    """

    label: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise DataError(f"regimen {self.label!r} assigns no treatment at any time point")
        for time, value in enumerate(self.values, start=1):
            if value not in (0.0, 1.0):
                raise DataError(
                    f"regimen {self.label!r} assigns {value!r} at time {time}; a longitudinal "
                    "fit takes a binary treatment at every node, so each entry must be 0 or 1"
                )

    @property
    def n_times(self) -> int:
        return len(self.values)

    def at(self, time: int) -> float:
        """The arm assigned at ``time``, counted from one."""
        return self.values[time - 1]

    def assignment(self, data: LongitudinalData) -> FloatArray:
        """The ``(n, T)`` arm matrix, as a broadcast view of the plan.

        A view rather than a copy, so reading a static regimen through the same matrix
        interface a rule needs allocates nothing -- and produces the same float64 the
        old scalar path produced, which is why a static fit is unchanged bit for bit.
        """
        return np.broadcast_to(np.asarray(self.values, dtype=float), (data.n, self.n_times))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        plan = "".join(str(int(value)) for value in self.values)
        return f"Regimen({self.label!r}, {plan})"


@dataclass(frozen=True)
class DynamicRegimen:
    """A plan whose nodes may be *rules* rather than constants.

    Attributes
    ----------
    label:
        What the reported parameter is named by, exactly as for :class:`Regimen`.
    plan:
        One entry per time point.  An entry is either an arm -- ``0.0`` or ``1.0``,
        meaning that arm for everybody at that node -- or a callable :math:`d_t(H_t)`
        handed that node's history frame and returning one arm per row.  Mixing the two
        is the ordinary case: "treat at the first node, then keep treating only while
        the biomarker stays high" is a constant followed by a rule.
    """

    label: str
    plan: tuple[RuleNode, ...]

    def __post_init__(self) -> None:
        if not self.plan:
            raise DataError(f"regimen {self.label!r} assigns no treatment at any time point")
        for time, node in enumerate(self.plan, start=1):
            if callable(node):
                continue
            if node not in (0.0, 1.0):
                raise DataError(
                    f"regimen {self.label!r} assigns {node!r} at time {time}; a longitudinal "
                    "fit takes a binary treatment at every node, so each entry must be 0, 1 "
                    "or a rule returning one of them per row"
                )

    @property
    def n_times(self) -> int:
        return len(self.plan)

    def is_rule(self, time: int) -> bool:
        """Whether ``time``'s arm is decided by a rule rather than declared."""
        return callable(self.plan[time - 1])

    def assignment(self, data: LongitudinalData) -> FloatArray:
        """Evaluate every node's rule and return the ``(n, T)`` arm matrix.

        Called **once** per fit, in :meth:`cleverly.longitudinal.LTMLE.fit`, and the
        matrix is what every mask, mechanism design and clever covariate then reads.
        Evaluating it once rather than at each use is not only cheaper: a rule that is
        not a deterministic function of the frame would otherwise let the follower masks
        disagree with the designs the mechanism was evaluated at, and the fit would be
        answering for no single regimen at all.

        A rule is asked for an arm on every row that is still *in the study* before the
        node -- uncensored through ``t - 1``, and on a survival fit event-free through
        ``t - 1`` as well, since a unit that has had the event has no treatment decision
        at ``t`` for a rule to make.  Off that set the history is the zero fill
        :meth:`~cleverly.longitudinal.data.LongitudinalData.covariate_history` puts there,
        so whatever the rule returns is meaningless; it is replaced by zero rather than
        validated, because such a row is masked out of every regression and every
        influence curve, and the only way it could still matter is by putting a ``nan``
        into a design matrix that a learner is called on.
        """
        columns = []
        for time, node in enumerate(self.plan, start=1):
            reachable = data.uncensored_through(time - 1) & data.event_free_through(time - 1)
            if callable(node):
                arms = self._evaluate(node, data, time, reachable)
            else:
                arms = np.full(data.n, float(node))
            columns.append(np.where(reachable, arms, 0.0))
        return np.column_stack(columns)

    def _evaluate(
        self, rule: Callable[[Any], Any], data: LongitudinalData, time: int, reachable: Any
    ) -> FloatArray:
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
        arms = np.asarray(_as_array(returned), dtype=float)
        if arms.ndim != 1 or arms.shape[0] != data.n:
            raise DataError(
                f"the rule at time {time} of regimen {self.label!r} returned "
                f"{arms.shape} assignments for {data.n} rows; it must return one arm per row"
            )
        unreadable = np.asarray(reachable) & ~np.isin(arms, (0.0, 1.0))
        if unreadable.any():
            offending = arms[unreadable]
            raise DataError(
                f"the rule at time {time} of regimen {self.label!r} returned "
                f"{offending[0]!r} for {int(unreadable.sum())} of the {data.n} rows; a "
                "longitudinal fit takes a binary treatment at every node, so a rule must "
                "return 0 or 1 for every unit whose history is recorded"
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
        return "/".join(str(int(value)) for value in regimen.values)
    return "/".join(_describe_node(node) for node in regimen.plan)


def _describe_node(node: RuleNode) -> str:
    """One node of a plan: its arm, or the rule's own name when it has one.

    A ``def``\\ -ed rule carries the name the analyst gave it, and reporting ``d:responds``
    rather than a bare ``d`` costs nothing and says which of two rules was run.  A lambda
    has no such name -- ``__name__`` is ``"<lambda>"``, which names nothing -- so it falls
    back to ``d`` and the plan fingerprint on the config is what tells two of them apart.
    """
    if not callable(node):
        return str(int(node))
    name = getattr(node, "__name__", "<lambda>")
    return "d" if name == "<lambda>" else f"d:{name}"


@dataclass(frozen=True)
class Plan:
    """A regimen together with the ``(n, T)`` arms it assigns *this* sample.

    The pair travels as one object so that nothing downstream can reach a rule and call
    it a second time: :func:`~cleverly.longitudinal.sequential.fit_mechanism` and
    :func:`~cleverly.longitudinal.sequential.fit_regimen` see arms, never callables.
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
    return tuple(Plan(regimen, regimen.assignment(data)) for regimen in regimens)


def resolve_regimens(spec: Any, n_times: int) -> tuple[RegimenSpec, ...]:
    """Turn a user's ``regimens=`` argument into an ordered tuple of regimens.

    Accepts a mapping from label to plan, where a plan is a single arm meaning "that arm
    at every node", a single rule meaning "that rule at every node", or a sequence of
    ``n_times`` entries each of which is an arm or a rule.  A sequence of
    :class:`Regimen` or :class:`DynamicRegimen` objects passes through.

    A plan with no rule in it comes back a :class:`Regimen`, which is what keeps a static
    fit on exactly the code path it was on before rules existed.

    Order is preserved, because the first regimen is the one contrasts are taken
    against by default and so is part of what the fit reports.
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
    """Read one plan into the regimen kind it describes."""
    if isinstance(plan, Regimen):
        plan = plan.values
    elif isinstance(plan, DynamicRegimen):
        plan = plan.plan
    if callable(plan):
        # The same broadcast a scalar arm gets, and the reason a rule reading a
        # late-measured covariate is diagnosed at evaluation rather than here: whether
        # ``lambda h: h["L2"] > 0`` is usable at node 1 is a question about the data.
        return DynamicRegimen(label, (plan,) * n_times)
    nodes = _nodes(label, plan, n_times)
    arms: list[float] = []
    for node in nodes:
        if callable(node):
            return DynamicRegimen(label, nodes)
        arms.append(float(node))
    return Regimen(label, tuple(arms))


def _nodes(label: str, plan: Any, n_times: int) -> tuple[RuleNode, ...]:
    """Read one plan into one entry per node, broadcasting a scalar arm across them."""
    if isinstance(plan, (bool, int, float)):
        return (float(plan),) * n_times
    # A numpy array and a pandas Series are plans by every reading except
    # ``isinstance(..., Sequence)``, which neither registers for.  Testing for the
    # iteration protocol instead keeps the message about rules where it belongs, rather
    # than aiming it at an array whose diagnosis it gets wrong.
    if isinstance(plan, str) or not hasattr(plan, "__iter__"):
        raise DataError(
            f"regimen {label!r} must be an arm (0 or 1), a rule d_t(H_t), or a sequence "
            f"of {n_times} of either; got {plan!r}"
        )
    nodes: tuple[RuleNode, ...] = tuple(
        entry if callable(entry) else float(entry) for entry in plan
    )
    if len(nodes) != n_times:
        raise DataError(
            f"regimen {label!r} assigns {len(nodes)} arm(s) but the data has {n_times} "
            "treatment node(s); a plan must say what happens at every one of them"
        )
    return nodes
