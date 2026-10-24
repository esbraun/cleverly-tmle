r"""The :class:`LongitudinalData` container.

A point-treatment dataset is a row per unit with one treatment on it.  A longitudinal
one is a row per unit with a *time-ordered sequence* of nodes on it, and the ordering is
the whole point: a covariate recorded after the first treatment is a confounder of the
second and a consequence of the first, which is exactly the structure no point-treatment
adjustment can handle.  The nodes are

.. math::

    W,\; L_1,\, A_1,\, C_1,\; L_2,\, A_2,\, C_2,\; \ldots,\; L_T,\, A_T,\, C_T,\; Y

read left to right in time.  ``W`` are the baseline covariates, ``L_t`` the covariates
measured at time ``t`` *before* the treatment decision, ``A_t`` that decision, and
``C_t`` an indicator that the unit is **still under observation after time** ``t``.  The
outcome ``Y`` sits at the end, and is observed exactly for the units that were never
censored.

A **survival** outcome moves ``Y`` into the ordering rather than leaving it at the end:

.. math::

    W,\; L_1,\, A_1,\, C_1,\, Y_1,\; L_2,\, A_2,\, C_2,\, Y_2,\; \ldots

where ``Y_t`` says the event happened at time ``t``, and is *absorbing* -- the unit is
out of the study from then on, so it has no ``L_{t+1}``, no ``A_{t+1}`` and no
``C_{t+1}``.  Declare it by passing one outcome column per node instead of one column:
``outcome=["Y1", "Y2"]``.  The parameter is then a curve, one cumulative risk per
horizon, rather than a number.

A **repeated** outcome is measured at every node and is not absorbing:

.. math::

    W,\; L_1,\, A_1,\, C_1,\, Y_1,\; L_2,\, A_2,\, C_2,\, Y_2,\; \ldots,\; Y_T

Declare it with ``outcome=Repeated(["Y1", ..., "YT"])``.  ``Y_s`` is the first covariate
of the history at node ``s + 1``, so each later treatment and censoring mechanism may read
it.  The parameter is the curve of means ``E[Y_h^d]``, one per horizon ``h``.  Each point
is the end-of-study parameter of the panel truncated after ``h``, with ``Y_h`` as its
final outcome.  A ``Y_t`` missing on a unit still under observation at ``t`` is refused,
with instructions to encode it as censoring at ``t``.

Four conventions are declared rather than inferred, because each one changes the
estimand if it is read the other way:

* **Censoring is monotone.**  ``C_t = 0`` means the unit left for good; a unit that
  returns is not censoring but an intermittent-observation process with its own
  identification.  Checked, not assumed.
* **A censored unit's later nodes are missing.**  Data recorded after ``C_t = 0`` is
  refused rather than ignored, because ignoring it silently is the difference between
  "this estimator did not need it" and "this estimator dropped it".
* **The outcome is missing only through censoring.**  An outcome missing on a unit that
  was never censored is a further node -- ``Delta`` in the point-treatment estimator --
  and is refused with instructions to encode it as a final censoring node, so that the
  factor it contributes to the clever covariate is estimated rather than assumed to be
  one.
* **The event is absorbing.**  ``Y_t = 1`` means the unit had the event and left the
  risk set; a later ``Y_s = 0`` would say it un-happened, which is a recurrent-event
  process with its own identification.  Checked, not assumed -- and, exactly as for
  censoring, a unit that has already had the event may carry ``1`` at the later nodes or
  nothing at all, since either way it says the same absorbing thing.

Observation weights are read exactly as :class:`~cleverly.data.causal_data.CausalData`
reads them, and mean exactly what they mean there: the estimand becomes the causal
parameter of the tilted law :math:`dP_w = w\,dP/E[w]`, and the whole fit runs on the
weighted empirical measure.  A weight is a property of the *unit* rather than of a node,
so it is one column whatever ``T`` is, and it is required at every row.
:mod:`cleverly.data.weighting` states the parameter, the influence function that goes with
it, and the readings of "weight" this container refuses rather than approximates.

The container holds numpy arrays and never branches on the dataframe backend; like
:class:`~cleverly.data.causal_data.CausalData` it records the backend it came from so
results are returned in it.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, TypeAlias

import numpy as np

from .._refusal import NO_ALTERNATIVE
from .._typing import BoolArray, FloatArray, IntArray
from ..data.known_mechanism import (
    KnownNodeMechanisms,
    known_node_mechanisms,
    probability_columns,
)
from ..data.validate import (
    MIN_CONTINUOUS_LEVELS,
    MIN_OBSERVATIONS,
    arm_indicators,
    atom_block,
    check_covariates,
    check_weights,
    encode_clusters,
    encode_treatment,
    encode_vector_treatment,
    resolve_family,
)
from ..data.weighting import (
    WeightReport,
    WeightSpec,
    _prepare_weights,
    describe_weights,
    effective_sample_size,
)
from ..exceptions import CapabilityError, DataError, DataWarning, Refusal, RefusalReason
from ..interventions.policy import JointComponent
from ..utils.frames import (
    as_frame,
    backend_of,
    column_array,
    frame_from_dict,
    is_dataframe,
    matrix_from_columns,
)
from .panel import PanelLayout

__all__ = [
    "Assignment",
    "LongitudinalData",
    "NodeKind",
    "RegimenMasks",
    "Repeated",
    "assignment_matrix",
    "check_markov_order",
    "markov_restriction",
]

#: What a treatment node holds.  A ``"categorical"`` node holds level codes, of one column or
#: of the observed tuples of a vector's components.  A ``"continuous"`` node holds one dose.
#: A ``"mixed"`` node is a vector with at least one continuous component: its components
#: live in :attr:`LongitudinalData.component_values`, and its slot in
#: :attr:`LongitudinalData.treatment` holds only its presence, ``0.0`` or ``nan``.
NodeKind: TypeAlias = Literal["categorical", "continuous", "mixed"]

#: One continuous node's kind, to repeat over the nodes of a panel of one dose column each.
_CONTINUOUS_KIND: tuple[NodeKind] = ("continuous",)

#: The fields :meth:`LongitudinalData.subset` indexes by row.  Every other field is
#: shared by the subset unchanged and is listed in :data:`_SHARED_FIELDS`.  A test checks
#: that the two tuples cover every field, so a new field without a subset rule fails.
_ROW_FIELDS: tuple[str, ...] = (
    "outcome",
    "baseline",
    "treatment",
    "uncensored",
    "time_varying",
    "weights",
    "event",
    "cause_event",
    "repeated",
    "cluster",
    "weight_spec",
    "known_mechanisms",
    "component_values",
    "unit_ids",
)
_SHARED_FIELDS: tuple[str, ...] = (
    "dose_supports",
    "dose_atoms",
    "baseline_names",
    "treatment_names",
    "treatment_levels",
    "node_kinds",
    "treatment_components",
    "component_levels",
    "time_varying_names",
    "family",
    "outcome_name",
    "event_names",
    "cause_labels",
    "censoring_names",
    "cluster_name",
    "weights_name",
    "dropped_covariates",
    "backend",
    "treatment_held",
    "time_grid",
    "time_name",
    "event_name",
    "event_codes",
    "repeated_names",
    "panel",
    "markov_order",
)


@dataclass(frozen=True)
class Repeated:
    """Declare a repeated outcome: one non-absorbing outcome column per time point.

    ``outcome=Repeated(["Y1", "Y2", "Y3"])`` declares that :math:`Y_t` is measured after the
    treatment and censoring nodes of time ``t`` and before the covariates of time ``t + 1``,
    and that it may return to an earlier value.  The fit then reports the mean
    :math:`E[Y_h^d]` of each plan ``d`` at each horizon ``h``.  The shape of ``outcome=``
    declares the parameter: a name is an end-of-study mean, a sequence is a survival curve,
    a mapping is one incidence curve per cause, and a ``Repeated`` is a curve of means.

    Parameters
    ----------
    columns : sequence of str
        The outcome column of each time point, in the same order as ``treatment``.
    """

    columns: tuple[str, ...]

    def __init__(self, columns: Sequence[str]) -> None:
        if isinstance(columns, str):
            raise DataError(
                "Repeated() takes one outcome column per time point, as a list; one column "
                "name is an end-of-study outcome, declared as outcome='Y'"
            )
        names = tuple(str(name) for name in columns)
        if not names:
            raise DataError("Repeated() is empty; name one outcome column per time point")
        if len(set(names)) != len(names):
            raise DataError(f"Repeated() repeats a column: {list(names)}")
        object.__setattr__(self, "columns", names)

    def __iter__(self) -> Iterator[str]:
        return iter(self.columns)

    def __len__(self) -> int:
        return len(self.columns)


#: What arm a regimen assigns at each node: one arm per node for a static plan, or an
#: ``(n, T)`` matrix when a dynamic rule assigns a different arm to different units.
#: Every mask and design here reads it through :func:`assignment_matrix`, so the static
#: case is the broadcast of the dynamic one rather than a second code path.
Assignment: TypeAlias = Sequence[float] | FloatArray


def markov_restriction(markov_order: int) -> str:
    """The statement of the statistical-model restriction that ``markov_order`` declares.

    Parameters
    ----------
    markov_order : int
        The declared order ``k``.

    Returns
    -------
    str
        The restriction, for the identification record of a fit.
    """
    return (
        f"statistical-model restriction, Markov order {markov_order}: at every node the "
        "outcome regression and the treatment and censoring mechanisms depend on the history "
        f"only through W and the last {markov_order + 1} nodes. It restricts the observed-data "
        "model and does not change the estimand. Every nuisance the estimator fits reads the "
        "window, so the estimator's own consistency and interval conditions apply to the "
        "nuisances fitted on the window, and each of them needs the restriction for the "
        "nuisance it reads."
    )


def check_markov_order(markov_order: int | None) -> None:
    """Refuse a ``markov_order`` that is not a non-negative integer.

    Parameters
    ----------
    markov_order : int or None
        The declared order, or ``None`` for the full history.

    Raises
    ------
    ValueError
        When ``markov_order`` is not ``None`` and not a non-negative integer.
    """
    if markov_order is None:
        return
    valid = (
        isinstance(markov_order, (int, np.integer))
        and not isinstance(markov_order, bool)
        and int(markov_order) >= 0
    )
    if not valid:
        raise ValueError(f"markov_order must be a non-negative integer; got {markov_order!r}")


def assignment_matrix(assignment: Assignment, n: int, n_times: int) -> FloatArray:
    """Read a plan as one ``(n, T)`` matrix, whether or not it varies by unit.

    A one-dimensional plan is *broadcast*, not tiled: the result is a read-only view
    costing no memory, so a static regimen pays nothing for the generality.  That the
    two paths then produce bit-identical floats is what lets a static fit stay
    bit-for-bit what it was before dynamic rules existed.
    """
    values = np.asarray(assignment, dtype=float)
    if values.ndim == 1:
        if values.shape[0] != n_times:
            raise DataError(
                f"a plan assigns {values.shape[0]} arm(s) but the data has {n_times} "
                "treatment node(s); it must say what happens at every one of them"
            )
        return np.broadcast_to(values, (n, n_times))
    if values.ndim == 2:
        if values.shape != (n, n_times):
            raise DataError(
                f"a per-unit assignment must be ({n}, {n_times}), a row per unit and a "
                f"column per treatment node, but got {values.shape}"
            )
        return values
    raise DataError(
        f"an assignment must be one arm per node or an (n, T) matrix; got {values.ndim} "
        "dimension(s)"
    )


class _MixedLevels(tuple[object, ...]):
    """The :attr:`LongitudinalData.treatment_levels` entry of a mixed node.

    A mixed node has no arms: its categorical components carry their own labels in
    :attr:`LongitudinalData.component_levels`.  Every read of this entry, by length,
    iteration, index, membership or truth, raises
    :class:`~cleverly.exceptions.CapabilityError`, so a reader that does not first ask
    :meth:`LongitudinalData.node_kind` refuses instead of reading zero levels.

    Parameters
    ----------
    node : str
        The node's name, for the refusal.
    time : int
        The node, counted from one.
    """

    node: str
    time: int

    def __new__(cls, node: str, time: int) -> _MixedLevels:
        out = super().__new__(cls, ())
        out.node = str(node)
        out.time = int(time)
        return out

    def _refuse(self) -> CapabilityError:
        return CapabilityError(
            f"node {self.node!r} at time {self.time} is a vector with a continuous component, "
            "and treatment_levels reads one column's labels. This path does not support a "
            "mixed node; read node_kind() and component_levels.",
            reason=RefusalReason.INVALID_DECLARATION,
        )

    def __len__(self) -> int:
        raise self._refuse()

    def __iter__(self) -> Any:
        raise self._refuse()

    def __getitem__(self, index: Any) -> Any:
        raise self._refuse()

    def __contains__(self, value: object) -> bool:
        raise self._refuse()

    def __bool__(self) -> bool:
        raise self._refuse()

    def __repr__(self) -> str:
        return f"<no levels: mixed node {self.node!r}>"

    def __reduce__(self) -> tuple[Any, ...]:
        return (_MixedLevels, (self.node, self.time))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _MixedLevels) and other.node == self.node and other.time == self.time
        )

    def __hash__(self) -> int:
        return hash((_MixedLevels, self.node, self.time))


@dataclass(frozen=True)
class LongitudinalData:
    """Validated inputs for a longitudinal estimator.

    Build one with :meth:`from_frame`.  Every array is numpy; missing entries -- a
    censored unit's later nodes -- are ``nan`` and are never read, because every
    quantity that reads them is masked by :meth:`at_risk` first.
    """

    outcome: FloatArray
    baseline: FloatArray
    baseline_names: tuple[str, ...]
    #: ``(n, T)`` treatment, ``nan`` where the unit had already been censored.
    treatment: FloatArray
    treatment_names: tuple[str, ...]
    #: Original treatment labels at each node, in the order used by :attr:`treatment`'s
    #: dense codes.  Nodes need not share either their labels or their number of levels.
    treatment_levels: tuple[tuple[object, ...], ...]
    #: ``(n, T)`` "still under observation after time ``t``".  All-true when the fit
    #: declares no censoring nodes, in which case it never enters the clever covariate.
    uncensored: BoolArray
    #: One ``(n, p_t)`` block per time point, holding the covariates measured at that
    #: time *before* the treatment decision.  A block may have no columns.
    time_varying: tuple[FloatArray, ...]
    time_varying_names: tuple[tuple[str, ...], ...]
    family: str
    #: Observation weights, normalised to mean one, all-ones on an unweighted fit.  One
    #: value per *unit* rather than per node: the tilt is of the population the parameter
    #: is defined in, not of any one node's mechanism.
    weights: FloatArray
    outcome_name: str = "Y"
    #: ``(n, T)`` "had the event at or before ``t``" on a survival fit, and ``None`` on a
    #: fit with one end-of-study outcome.  Cumulative, so it is monotone by construction
    #: and :meth:`event_free_through` is a column read rather than a scan.  A unit
    #: censored before the event carries ``False``: it is masked out of everything by
    #: :meth:`uncensored_through` first, and a matrix with no missing entries is one no
    #: caller has to remember to guard.
    event: BoolArray | None = None
    event_names: tuple[str, ...] = ()
    #: ``(n, T, J)`` "had *this* cause's event at or before ``t``", and ``None`` unless the
    #: fit declared competing risks.  :attr:`event` stays the **all-cause** matrix beside
    #: it, which is what every mask reads: leaving the risk set is leaving it, whichever
    #: cause did it, so :meth:`at_risk`, :meth:`following` and the mechanism's fit mask are
    #: the same expressions they were for one event.  Only the pseudo-outcome of the
    #: sequential regression is cause-specific, and it is the one thing that reads this.
    cause_event: BoolArray | None = None
    #: The absorbing causes, in report order; empty unless the fit declared them.
    cause_labels: tuple[str, ...] = ()
    censoring_names: tuple[str, ...] = ()
    cluster: IntArray | None = None
    cluster_name: str | None = None
    weights_name: str | None = None
    weight_spec: WeightSpec = field(default_factory=WeightSpec)
    dropped_covariates: tuple[str, ...] = field(default_factory=tuple)
    #: Name of the dataframe backend the data arrived in; see
    #: :attr:`cleverly.data.CausalData.backend`, which this mirrors exactly.
    backend: str | None = None
    #: The :data:`NodeKind` of each node.  A continuous node keeps the raw dose in
    #: :attr:`treatment`, has no :attr:`treatment_levels`, and accepts only a modified
    #: treatment policy.  A mixed node keeps its components in :attr:`component_values` and
    #: accepts only a :class:`~cleverly.interventions.JointPolicy`.  Empty means that every
    #: node is categorical.
    node_kinds: tuple[NodeKind, ...] = ()
    #: The columns of each node.  A node of one column holds its own name; a vector node,
    #: declared as a list of columns, is one categorical node over the observed tuples of
    #: its components (Díaz et al. 2023, Section 2: :math:`A_t` may be a vector), or a mixed
    #: node when a component is declared continuous.
    treatment_components: tuple[tuple[str, ...], ...] = ()
    #: Per node, the ``(n, d_t)`` components of a mixed node, and ``None`` at every other
    #: node: the level code of a categorical component and the dose of a continuous one,
    #: ``nan`` where the unit holds no treatment at the node.  Empty without a mixed node.
    component_values: tuple[FloatArray | None, ...] = ()
    #: Per node, the labels of each component of a mixed node in code order, ``()`` for a
    #: continuous component, and ``()`` at every other node.  Empty without a mixed node.
    component_levels: tuple[tuple[tuple[object, ...], ...], ...] = ()
    #: Whether the design has **one baseline treatment decision held over every node**.
    #: Declared by passing ``treatment=`` as one column name.  Node 1 is the decision.
    #: Every later node is an *identity node*: :attr:`treatment` carries the observed node-1
    #: value there on the rows still in the study, its mechanism factor is exactly one, it
    #: fits no model, it adds no block to any design, and a regimen is resolved on node 1
    #: alone.  :attr:`treatment_names` then has one entry, for the decision, and
    #: :attr:`treatment_levels` and :attr:`node_kinds` repeat node 1's entry per node.
    treatment_held: bool = False
    #: The grid times :math:`g_1 < \dots < g_K` of a container built from one time and one
    #: event column (:meth:`from_time_to_event`), and ``None`` on the wide path.  Node
    #: :math:`k` covers the interval :math:`(g_{k-1}, g_k]`, with :math:`g_0 = 0`.
    time_grid: tuple[float, ...] | None = None
    #: The time column of a container built by :meth:`from_time_to_event`, else ``None``.
    time_name: str | None = None
    #: The event-code column of a container built by :meth:`from_time_to_event`, else ``None``.
    event_name: str | None = None
    #: The event code of each absorbing cause, in :attr:`cause_labels` order (``(1,)`` for a
    #: single event), on a container built by :meth:`from_time_to_event`.  Empty otherwise.
    event_codes: tuple[int, ...] = ()
    #: The treatment and censoring factors the design declares known, or ``None`` when every
    #: factor is estimated.  A data field, so a bootstrap replicate carries the rows it drew.
    #: See :class:`~cleverly.data.known_mechanism.KnownNodeMechanisms`.
    known_mechanisms: KnownNodeMechanisms | None = None
    #: Per node, the declared conditional support of a continuous dose, a
    #: :class:`~cleverly.interventions.Support`, and ``None`` elsewhere.  Empty when no node
    #: declares one.  Declared by the mapping form of ``continuous_treatment=``.
    dose_supports: tuple[Any, ...] = ()
    #: Per node, the declared atoms of a continuous dose (the doses with a recorded point
    #: mass), and ``()`` elsewhere.  A vector node holds ``(component, atoms)`` pairs in
    #: column order.  Empty when no node declares one.  Declared by ``dose_atoms=``.
    dose_atoms: tuple[tuple[Any, ...], ...] = ()
    #: ``(n, T)`` outcome at each node of a repeated outcome (:class:`Repeated`), ``nan``
    #: where the unit was censored at or before that node, and ``None`` otherwise.  Each
    #: column is the outcome of one horizon.  Column ``s`` is also the first covariate of
    #: block ``s + 1`` of :attr:`time_varying`, where the covariate screen may drop it; this
    #: array keeps every node's outcome whole.
    repeated: FloatArray | None = None
    #: The outcome column of each node of a repeated outcome, in node order; empty otherwise.
    repeated_names: tuple[str, ...] = ()
    #: The long-panel roles of a container built by :meth:`from_long_panel`, else ``None``.
    panel: PanelLayout | None = None
    #: The unit identifier of each row on a container built by :meth:`from_long_panel`, in
    #: sorted order, else ``None``.
    unit_ids: Any = None
    #: The declared Markov order ``k`` of :meth:`with_markov_order`, or ``None`` for the full
    #: history.  Every nuisance design then reads ``W`` and a window of the last nodes, and
    #: the frames a rule or a policy reads keep the full history.
    markov_order: int | None = None

    # ------------------------------------------------------------------ build

    @classmethod
    def from_frame(
        cls,
        data: Any,
        *,
        outcome: str | Sequence[str] | Mapping[str, Sequence[str]] | Repeated,
        treatment: str | Sequence[str | Sequence[str]],
        baseline: Sequence[str],
        time_varying: Sequence[Sequence[str]] | None = None,
        censoring: Sequence[str] | None = None,
        id: str | None = None,
        weights: str | None = None,
        weights_type: str = "probability",
        weights_estimated: bool = False,
        family: str = "auto",
        continuous_treatment: Sequence[str] | Mapping[str, Any] = (),
        treatment_probabilities: Any = None,
        censoring_probabilities: Any = None,
        dose_atoms: Mapping[str, Sequence[float]] | None = None,
    ) -> LongitudinalData:
        """Build from one wide dataframe: a row per unit, a column per node.

        Parameters
        ----------
        outcome:
            One column name for an end-of-study outcome; **one per time point** for a
            survival outcome, in the same order as ``treatment``; or a **mapping of cause
            to one column per time point** for competing risks,
            ``{"relapse": ["R1", "R2"], "death": ["D1", "D2"]}``; or a :class:`Repeated`
            of **one non-absorbing outcome per time point**, ``Repeated(["Y1", "Y2"])``.
            Which of the four is passed declares which parameter the fit answers, so it is
            read off the shape rather than from a further keyword: a number, a risk curve,
            a set of incidence curves and a curve of means are not the same report, and a
            flag saying which would be a second place for them to disagree.  A mapping with
            one cause is still competing risks by declaration, and reports a cumulative
            incidence.  A repeated outcome :math:`Y_s` is observed with the covariates of
            time ``s + 1``, so it is also their first covariate: it enters every history
            from node ``s + 1`` on.
        treatment:
            Treatment column per time point, in time order.  Its length declares ``T``.  A
            node may be a list of columns, ``[["A1a", "A1b"], ["A2a", "A2b"]]``: it is then
            one categorical node whose levels are the observed tuples of its components, or
            a mixed node when ``continuous_treatment`` names one of its components.
            One column name, ``treatment="A"``, declares **one baseline decision held over
            every node** (:attr:`treatment_held`).  ``T`` then comes from the survival
            outcome, else from ``censoring=``, else from ``time_varying=``, else it is one.
        baseline:
            Covariates measured before any treatment.
        time_varying:
            Covariate columns measured at each time point, before that point's
            treatment: one (possibly empty) list per time point.  ``None`` means there
            are none, which makes the fit a sequence of point treatments and is almost
            never what a longitudinal analysis is for -- state it explicitly all the
            same, since "no time-varying confounder" is an assumption.
        censoring:
            One column per time point, ``1`` where the unit is still under observation
            after that point.  ``None`` means nobody was censored.
        weights:
            Observation weights: one column, one value per unit, present at every row.
            The estimand becomes the causal parameter in the tilted population
            ``dP_w = w dP / E[w]`` -- see :mod:`cleverly.data.weighting`, which states it
            for the point-treatment estimator in the same terms it holds here.
        weights_type:
            How to read ``weights``, as for
            :meth:`cleverly.data.CausalData.from_frame`.  ``"frequency"`` -- counts of
            identical units -- is a different experiment and is refused with instructions.
        weights_estimated:
            Declare that the weights came out of a fitted model.  Changes no number; it
            makes the reports state that the intervals condition on them.
        continuous_treatment:
            The treatment columns that hold a continuous dose.  Such a node keeps its raw
            dose, has a conditional density rather than a multinomial mechanism, and takes a
            modified treatment policy at every regimen (Díaz, Williams, Hoffman and Schenck
            2023).  A component of a vector node may be named too.  The node is then mixed:
            it takes a :class:`~cleverly.interventions.JointPolicy` at every regimen, and the
            fit estimates its policy ratio with ``ratio="classifier"``.  A mapping
            ``{name: Support(lower, upper) or None}`` also declares the conditional support
            of each dose (:class:`~cleverly.interventions.Support`).  The fit then refuses,
            before any learner, an observed dose outside it and a policy that assigns a row
            at risk a dose outside it (Díaz et al. 2023, Assumption 1).  The support of a
            component is given the history, the node's categorical components and its
            doses before it in column order, and each branch of a joint policy is
            checked on the components it assigns.
        treatment_probabilities:
            The known treatment mechanism of a sequentially randomized design, which the fit
            then reads in place of a treatment learner.  The preferred form maps each node
            column to a mapping of that node's levels to the column of
            ``P(A_t = level | observed past)``, ``{"A1": {0: "p1_0", 1: "p1_1"}}``; nodes
            left out are estimated.  An ``(n, T)`` array of ``P(A_t = levels_t[1] | observed
            past)`` at binary nodes and an ``(n, T, K)`` array at nodes that share ``K``
            levels are read too.  A value on a row that is not at risk at the node is never
            read and may be missing.  On a held design the one decision is the one node.
        censoring_probabilities:
            The known retention probabilities ``P(C_t = 1 | observed past)``: a mapping of
            censoring column to the column that holds them, or an ``(n, T)`` array.  It may
            be given alone or beside ``treatment_probabilities``.
        dose_atoms:
            A mapping of continuous dose column to its declared atoms, the doses with a
            recorded point mass, such as a list price: ``{"A2": (3.0,)}``.  The policy
            ratio at that node is then taken against Lebesgue plus counting measure on the
            atoms, and a piece may send an interval to one of them with
            ``Piece(lower, upper, to_atom=value)``.  A continuous node with an undeclared
            point mass is refused before any learner.
        """
        if not is_dataframe(data):
            raise DataError("LongitudinalData.from_frame expects a pandas or polars DataFrame")
        supports: dict[str, Any] = {}
        if isinstance(continuous_treatment, Mapping):
            supports = {
                str(name): value
                for name, value in continuous_treatment.items()
                if value is not None
            }
            continuous_treatment = tuple(str(name) for name in continuous_treatment)
        frame = as_frame(data)
        columns = list(frame.columns)

        held = isinstance(treatment, str)
        components: list[tuple[str, ...]] = []
        if held:
            decision = str(treatment)
            components = [(decision,)] * _held_node_count(
                decision, outcome, censoring, time_varying
            )
            treatment_names = [decision]
            flat_treatment = [decision]
        else:
            for node in treatment:
                if isinstance(node, str):
                    components.append((node,))
                else:
                    node_columns = tuple(str(name) for name in node)
                    if not node_columns:
                        raise DataError("a vector treatment node needs at least one column")
                    components.append(node_columns)
            treatment_names = ["+".join(node_block) for node_block in components]
            if not treatment_names:
                raise DataError("treatment= is empty; a longitudinal fit needs at least one node")
            flat_treatment = [name for node_block in components for name in node_block]
        n_times = len(components)
        continuous_names = [str(name) for name in continuous_treatment]
        unknown_continuous = sorted(set(continuous_names) - set(flat_treatment))
        if unknown_continuous:
            raise DataError(
                f"continuous_treatment= names {unknown_continuous}, which are not treatment "
                f"columns; the treatment columns are {flat_treatment}"
            )
        node_kinds = tuple(_node_kind_of(node_block, continuous_names) for node_block in components)

        censor_names = [] if censoring is None else [str(name) for name in censoring]
        if censoring is not None and len(censor_names) != n_times:
            raise DataError(
                f"censoring= names {len(censor_names)} column(s) but there are {n_times} "
                "treatment node(s); pass one censoring column per time point"
            )

        if time_varying is None:
            blocks: list[list[str]] = [[] for _ in range(n_times)]
        else:
            blocks = [[str(name) for name in block] for block in time_varying]
            if len(blocks) != n_times:
                raise DataError(
                    f"time_varying= has {len(blocks)} block(s) but there are {n_times} "
                    "treatment node(s); pass one (possibly empty) list per time point"
                )

        # Four shapes, and which one is passed declares which parameter the fit answers:
        # a name is a number, a sequence is a risk curve, a mapping is one curve per cause,
        # and a ``Repeated`` is a curve of means.  Read off the shape rather than from a
        # further keyword, since a flag saying which would be a second place for the
        # declaration and the columns to disagree.
        cause_labels: tuple[str, ...] = ()
        repeated_names: list[str] = []
        event_blocks: list[list[str]]
        if isinstance(outcome, Repeated):
            event_blocks = []
            outcome_names = list(outcome.columns)
            survival = False
            if len(outcome_names) != n_times:
                raise DataError(
                    f"Repeated() names {len(outcome_names)} column(s) but there are "
                    f"{n_times} treatment node(s); a repeated outcome is one column per time "
                    "point, in the same order as treatment="
                )
            repeated_names = outcome_names
            declared = sorted({name for block in blocks for name in block} & set(repeated_names))
            if declared:
                raise DataError(
                    f"time_varying= names the outcome column(s) {declared}. A repeated outcome "
                    "enters the history of every later node by itself, as the first covariate "
                    "of the next time point; drop it from time_varying="
                )
            # Y_s is measured with L_{s+1} and before A_{s+1}, so it is a covariate of node
            # s + 1.  It joins that block *before* the covariate screen, which makes each
            # horizon's history exactly the history of the panel truncated after it.
            for position in range(1, n_times):
                blocks[position] = [repeated_names[position - 1], *blocks[position]]
        elif isinstance(outcome, str):
            event_blocks = []
            outcome_names = [outcome]
            survival = False
        elif isinstance(outcome, Mapping):
            if not outcome:
                raise DataError(
                    "outcome= is an empty mapping; competing risks are declared by naming "
                    "each absorbing cause and its indicator column per time point, for "
                    "example outcome={'relapse': ['R1', 'R2'], 'death': ['D1', 'D2']}"
                )
            cause_labels = tuple(str(label) for label in outcome)
            event_blocks = [[str(name) for name in block] for block in outcome.values()]
            for label, block in zip(cause_labels, event_blocks, strict=True):
                if len(block) != n_times:
                    raise DataError(
                        f"cause {label!r} names {len(block)} column(s) but there are "
                        f"{n_times} treatment node(s); every cause needs one event "
                        "indicator per time point, in the same order as treatment="
                    )
            outcome_names = [name for block in event_blocks for name in block]
            survival = True
        else:
            outcome_names = [str(name) for name in outcome]
            event_blocks = [outcome_names]
            survival = True
            if len(outcome_names) != n_times:
                raise DataError(
                    f"outcome= names {len(outcome_names)} column(s) but there are "
                    f"{n_times} treatment node(s); a survival outcome is one event "
                    "indicator per time point, and one column is an end-of-study outcome"
                )

        baseline_names = [str(name) for name in baseline]
        wanted = [*outcome_names, *flat_treatment, *censor_names, *baseline_names]
        for block in blocks:
            # A repeated outcome's covariate copy is already in ``outcome_names``.
            wanted.extend(name for name in block if name not in repeated_names)
        if id is not None:
            wanted.append(id)
        if weights is not None:
            wanted.append(str(weights))
        treatment_columns, censoring_columns = _probability_column_names(
            treatment_probabilities, censoring_probabilities
        )
        probability_names = [
            name for pairs in treatment_columns.values() for _, name in pairs
        ] + list(censoring_columns.values())
        missing = [name for name in [*wanted, *probability_names] if name not in columns]
        if missing:
            raise DataError(f"columns not found in the frame: {missing}; available: {columns}")
        _refuse_duplicates(wanted)
        overlap = sorted(set(probability_names) & set(wanted))
        if overlap:
            raise DataError(f"columns {overlap} are used both as a role and as a known probability")

        built = cls._build(
            outcome=None if survival else column_array(frame, outcome_names[-1]),
            repeated=(
                np.column_stack([column_array(frame, name) for name in repeated_names])
                if repeated_names
                else None
            ),
            repeated_names=repeated_names,
            # ``(n, T, J)``: a node axis and a cause axis, with ``J = 1`` for a single
            # absorbing event.  One shape rather than two keeps the validating sweep and
            # every mask below written once.
            event=(
                np.stack(
                    [
                        np.column_stack([column_array(frame, name) for name in block])
                        for block in event_blocks
                    ],
                    axis=2,
                )
                if survival
                else None
            ),
            event_names=event_blocks if survival else [],
            cause_labels=cause_labels,
            baseline=matrix_from_columns(frame, baseline_names) if baseline_names else None,
            baseline_names=baseline_names,
            treatment=np.column_stack([_node_column(frame, block) for block in components]),
            treatment_names=treatment_names,
            node_kinds=node_kinds,
            treatment_components=tuple(components),
            continuous_components=tuple(
                tuple(name in continuous_names for name in node_block) for node_block in components
            ),
            censoring=(
                None
                if censoring is None
                else np.column_stack([column_array(frame, name) for name in censor_names])
            ),
            censoring_names=censor_names,
            time_varying=[matrix_from_columns(frame, block) if block else None for block in blocks],
            time_varying_names=blocks,
            cluster=None if id is None else frame[id].to_numpy(),
            cluster_name=id,
            weights=None if weights is None else column_array(frame, weights),
            weights_type=weights_type,
            weights_estimated=weights_estimated,
            weights_name=weights,
            outcome_name=outcome_names[-1],
            family=family,
            backend=backend_of(frame),
            treatment_held=held,
        )
        built = built.with_dose_declarations(supports, dose_atoms or {})
        if treatment_probabilities is None and censoring_probabilities is None:
            return built
        return built._declare_known(
            treatment_probabilities=(
                treatment_probabilities
                if not treatment_columns
                else {
                    node: (
                        {
                            level: column_array(frame, name)
                            for level, name in treatment_columns[node]
                        }
                        if node in treatment_columns
                        else per_node
                    )
                    for node, per_node in treatment_probabilities.items()
                }
            ),
            censoring_probabilities=(
                censoring_probabilities
                if not censoring_columns
                else {
                    node: (
                        column_array(frame, censoring_columns[node])
                        if node in censoring_columns
                        else column
                    )
                    for node, column in censoring_probabilities.items()
                }
            ),
            treatment_columns=treatment_columns,
            censoring_columns=censoring_columns,
        )

    def with_dose_declarations(
        self,
        supports: Mapping[str, Any],
        atoms: Mapping[str, Sequence[float]],
    ) -> LongitudinalData:
        """Return a copy that declares the support and the atoms of each continuous dose.

        Parameters
        ----------
        supports : mapping of str to Support
            The declared conditional support of each named dose column.
        atoms : mapping of str to sequence of float
            The declared atoms of each named dose column.

        Returns
        -------
        LongitudinalData
            The same panel with the declarations attached, or itself when both are empty.

        Raises
        ------
        DataError
            If a name is not a continuous treatment node or component, an entry is
            malformed, or an atom is declared twice.

        Notes
        -----
        A continuous component of a vector node takes a declared support and declared
        atoms.  The support is given the history, the node's categorical components and its
        doses before the component in column order, the conditioning set of the component's
        factor in the joint density.  The atoms make that factor mixed: it is taken against
        Lebesgue measure plus counting measure on the atoms.
        """
        from ..interventions.policy import Support

        if not supports and not atoms:
            return self
        names = {name: time for time, name in enumerate(self.treatment_names, start=1)}
        components = {
            name: time
            for time, block in enumerate(self.treatment_components, start=1)
            if len(block) > 1
            for name in block
        }
        node_supports: list[Any] = [None] * self.n_times
        node_atoms: list[tuple[Any, ...]] = [()] * self.n_times
        component_supports: dict[int, dict[str, Any]] = {}
        component_atoms: dict[int, dict[str, tuple[float, ...]]] = {}
        for keyword, declared in (("continuous_treatment", supports), ("dose_atoms", atoms)):
            for name, value in declared.items():
                if name in components:
                    node = components[name]
                    index = self.treatment_components[node - 1].index(name)
                    if self.component_levels[node - 1][index]:
                        what = "a support" if keyword == "continuous_treatment" else "atoms"
                        raise DataError(
                            f"{keyword}= declares {what} for {name!r}, a categorical component "
                            "of a vector node. A declaration of the dose applies to a "
                            "continuous component"
                        )
                    if keyword == "dose_atoms":
                        values = tuple(float(atom) for atom in value)
                        if (
                            not values
                            or not all(np.isfinite(atom) for atom in values)
                            or len(set(values)) != len(values)
                        ):
                            raise DataError(
                                f"dose_atoms= for {name!r} holds distinct finite doses; got "
                                f"{list(values)!r}"
                            )
                        component_atoms.setdefault(node, {})[name] = tuple(sorted(values))
                        continue
                    if value is None:
                        continue
                    if not isinstance(value, Support):
                        raise DataError(
                            f"continuous_treatment= maps {name!r} to {value!r}; a declared "
                            "support is Support(lower, upper) or None"
                        )
                    component_supports.setdefault(node, {})[name] = value
                    continue
                time = names.get(name)
                if time is None or self.node_kind(time) != "continuous":
                    raise DataError(
                        f"{keyword}= declares {name!r}, which is not a continuous treatment "
                        "node. Declare the node in continuous_treatment= first"
                    )
                if keyword == "dose_atoms":
                    values = tuple(float(atom) for atom in value)
                    if (
                        not values
                        or not all(np.isfinite(atom) for atom in values)
                        or len(set(values)) != len(values)
                    ):
                        raise DataError(
                            f"dose_atoms= for {name!r} holds distinct finite doses; got "
                            f"{list(values)!r}"
                        )
                    node_atoms[time - 1] = tuple(sorted(values))
                else:
                    if not isinstance(value, Support):
                        raise DataError(
                            f"continuous_treatment= maps {name!r} to {value!r}; a declared "
                            "support is Support(lower, upper) or None"
                        )
                    node_supports[time - 1] = value
        for time, declared in component_supports.items():
            order = self.treatment_components[time - 1]
            node_supports[time - 1] = tuple(
                (name, declared[name]) for name in order if name in declared
            )
        for time, declared_atoms in component_atoms.items():
            order = self.treatment_components[time - 1]
            node_atoms[time - 1] = tuple(
                (name, declared_atoms[name]) for name in order if name in declared_atoms
            )
        if self.treatment_held:
            node_supports = [node_supports[0]] * self.n_times
            node_atoms = [node_atoms[0]] * self.n_times
        return replace(
            self,
            dose_supports=tuple(node_supports) if any(v is not None for v in node_supports) else (),
            dose_atoms=tuple(node_atoms) if any(node_atoms) else (),
        )

    def node_support(self, time: int) -> Any:
        """The declared support of node ``time``, or ``None``.

        Parameters
        ----------
        time : int
            The treatment node, counted from one.

        Returns
        -------
        Support or None
            The declaration.  A vector node returns ``None``; its components' supports are
            :meth:`component_supports`.
        """
        value = self.dose_supports[time - 1] if self.dose_supports else None
        return None if isinstance(value, tuple) else value

    def component_supports(self, time: int) -> dict[str, Any]:
        """The declared support of each continuous component of vector node ``time``.

        Parameters
        ----------
        time : int
            The treatment node, counted from one.

        Returns
        -------
        dict of str to Support
            Empty at a node of one column, or when no component declares one.
        """
        value = self.dose_supports[time - 1] if self.dose_supports else None
        return dict(value) if isinstance(value, tuple) else {}

    def node_atoms(self, time: int) -> tuple[float, ...]:
        """The declared atoms of node ``time``, or ``()``.

        Parameters
        ----------
        time : int
            The treatment node, counted from one.

        Returns
        -------
        tuple of float
            The declared atoms, in increasing order.  A vector node returns ``()``; its
            components' atoms are :meth:`component_atoms`.
        """
        value = self.dose_atoms[time - 1] if self.dose_atoms else ()
        return () if value and isinstance(value[0], tuple) else tuple(value)

    def component_atoms(self, time: int) -> dict[str, tuple[float, ...]]:
        """The declared atoms of each continuous component of vector node ``time``.

        Parameters
        ----------
        time : int
            The treatment node, counted from one.

        Returns
        -------
        dict of str to tuple of float
            Empty at a node of one column, or when no component declares atoms.
        """
        value = self.dose_atoms[time - 1] if self.dose_atoms else ()
        if not value or not isinstance(value[0], tuple):
            return {}
        return {str(name): tuple(atoms) for name, atoms in value}

    def dose_block(self, time: int, dose: FloatArray) -> FloatArray:
        """A continuous node's design block: the dose and the indicator of each declared atom.

        Without declared atoms it is the dose as one column, as before.  With them each
        indicator is read from the dose given, so a counterfactual design reads the
        assigned dose.

        Parameters
        ----------
        time : int
            A continuous node, counted from one.
        dose : ndarray
            ``(n,)`` doses.

        Returns
        -------
        ndarray
            ``(n, 1 + K)`` block, with ``K`` the number of declared atoms.
        """
        return atom_block(dose, self.node_atoms(time))

    @classmethod
    def from_long_panel(
        cls,
        data: Any,
        *,
        unit: str,
        time: str,
        treatment: str | Sequence[str],
        baseline: Sequence[str],
        time_varying: Sequence[str] = (),
        censoring: str | None = None,
        outcome: str | None = None,
        event: str | Mapping[str, str] | None = None,
        times: Sequence[Any] | None = None,
        cluster: str | None = None,
        weights: str | None = None,
        weights_type: str = "probability",
        weights_estimated: bool = False,
        family: str = "auto",
        continuous_treatment: bool = False,
        treatment_probabilities: Mapping[Any, str] | None = None,
        censoring_probabilities: str | None = None,
    ) -> LongitudinalData:
        """Build from a long panel: one row per unit and decision time.

        Each row holds the covariates measured before that row's decision, the decision,
        and whether the unit is still observed after it.  The rows of one unit form one
        trajectory, and the trajectory is one observation: repeated rows add no independent
        observations.  Units that depend on each other, such as the customers of one
        account, declare ``cluster=``.

        The container is the one :meth:`from_frame` builds from the equivalent wide frame.
        Node ``t`` is the ``t``-th decision time.  A column ``c`` measured at node ``t`` is
        named ``"c@t"`` in :meth:`history_frame` and :meth:`policy_frame`, and
        :func:`~cleverly.longitudinal.each_node` writes a rule that reads it.  Rows may
        arrive in any order.  The container's rows follow the sorted unit identifiers.

        Parameters
        ----------
        data : DataFrame
            A pandas or polars frame with one row per unit and decision time.
        unit : str
            The unit identifier column.  Identifiers must sort, so use one type.
        time : str
            The decision-time column.  Times must sort.
        treatment : str or sequence of str
            The decision at each row.  A list of columns is one vector node per time.
        baseline : sequence of str
            Unit-level covariates measured before the first decision.  Each is recorded on
            the unit's first row and holds one value per unit.
        time_varying : sequence of str
            Covariates measured at each row before that row's decision.  A measurement taken
            after the decision belongs to the next row.  A column with no value at the first
            nodes, such as a lag, starts at its first recorded node.
        censoring : str or None
            ``1`` where the unit is still observed after the row's decision, and ``0`` on the
            last row of a unit that left.  ``None`` means every unit has a row at every
            decision time, or leaves through the event.
        outcome : str or None
            An end-of-study outcome: one value per unit, on any of its rows.
        event : str or mapping of str to str or None
            A survival outcome: an event indicator on each row, ``1`` on the row of the
            event.  A mapping of cause label to column declares competing risks.  Declare
            exactly one of ``outcome`` and ``event``.
        times : sequence or None
            Every decision time, in time order; the order given is the node order.  ``None``
            uses the times the panel holds, sorted, and needs numbers, dates or durations.
        cluster : str or None
            The cluster identifier, constant within a unit.
        weights : str or None
            A unit-level probability weight.
        weights_type : {"probability"}
            How to read ``weights``, as for :meth:`from_frame`.
        weights_estimated : bool
            Whether the weights came out of a fitted model.
        family : {"auto", "gaussian", "binomial"}
            The outcome family.
        continuous_treatment : bool
            Whether the treatment column holds a continuous dose at every node.
        treatment_probabilities : mapping of level to str, or None
            The known treatment mechanism: each level mapped to the column that holds
            ``P(A_t = level | past)`` on each row.
        censoring_probabilities : str or None
            The column that holds ``P(C_t = 1 | past)`` on each row.

        Returns
        -------
        LongitudinalData
            The container, with :attr:`panel` and :attr:`unit_ids` set.

        Raises
        ------
        DataError
            For a missing unit or time, a duplicated ``(unit, time)`` row, a unit that
            starts late or skips a decision time, rows after a unit left, a unit whose rows
            stop without leaving, a unit-level column that varies within a unit or is first
            recorded after the first row, a column in two roles, a covariate that copies the
            outcome or the event, an event recorded on a censored row, a time-varying column
            that stops being recorded, and a treatment that never changes within a unit.
            Every check of :meth:`from_frame` applies as well.

        See Also
        --------
        to_long_panel : The inverse.
        cleverly.LongPanel : The same input as a study design.
        cleverly.longitudinal.each_node : One rule for every node.

        Examples
        --------
        A panel of 20 customers over two weeks.  Customer 3 leaves after week 1, so its
        censoring is 0 on its one row and its outcome is missing.

        >>> import numpy as np
        >>> import pandas as pd
        >>> from cleverly.longitudinal import LongitudinalData
        >>> rng = np.random.default_rng(0)
        >>> panel = pd.DataFrame(
        ...     {
        ...         "customer": np.repeat(np.arange(20), 2),
        ...         "week": np.tile([1, 2], 20),
        ...         "age": np.repeat(rng.normal(size=20), 2),
        ...         "eng": rng.normal(size=40),
        ...         "offer": rng.choice(["none", "offer"], size=40),
        ...         "active": 1.0,
        ...         "bought": np.repeat(rng.integers(0, 2, size=20), 2).astype(float),
        ...     }
        ... )
        >>> panel = panel.drop(index=7)
        >>> panel.loc[6, ["active", "bought"]] = [0.0, np.nan]
        >>> data = LongitudinalData.from_long_panel(
        ...     panel,
        ...     unit="customer",
        ...     time="week",
        ...     treatment="offer",
        ...     baseline=["age"],
        ...     time_varying=["eng"],
        ...     censoring="active",
        ...     outcome="bought",
        ... )
        >>> data.n, data.n_times
        (20, 2)
        >>> data.history_names(2)
        ('age', 'eng@1', 'eng@2')
        """
        from .panel import from_long_panel

        return from_long_panel(
            cls,
            data,
            unit=unit,
            time=time,
            treatment=treatment,
            baseline=baseline,
            time_varying=time_varying,
            censoring=censoring,
            outcome=outcome,
            event=event,
            times=times,
            cluster=cluster,
            weights=weights,
            weights_type=weights_type,
            weights_estimated=weights_estimated,
            family=family,
            continuous_treatment=continuous_treatment,
            treatment_probabilities=treatment_probabilities,
            censoring_probabilities=censoring_probabilities,
        )

    def to_long_panel(self) -> Any:
        """Return the long panel this container holds: one row per unit and node.

        A unit has a row at each node at which it is still in the study: uncensored and
        event-free before the node.  Rows are sorted by unit and then time, in the backend
        the data came from.  The outcome is repeated on every row of a unit, and missing for
        a censored unit.  An event column holds ``1`` on the event row, ``0`` before it, and
        a missing value on the row where the unit was censored.

        Three values differ from the input: the weights are normalized to mean one, a
        covariate screened out at a node is missing at that node, and known probability
        columns are not returned.

        Returns
        -------
        DataFrame
            The long panel, with the column names of :attr:`panel`.

        Raises
        ------
        DataError
            If the container was not built by :meth:`from_long_panel`.
        """
        from .panel import to_long_panel

        return to_long_panel(self)

    def with_markov_order(self, markov_order: int | None) -> LongitudinalData:
        """Return a copy whose nuisance designs read a window of the last nodes.

        The order ``k`` is a statistical-model restriction on the observed law.  At node
        ``t`` the treatment and censoring mechanisms read ``W``, ``L_{t-k}, ..., L_t`` and
        ``A_{t-k-1}, ..., A_{t-1}``.  The outcome regression reads ``W`` and
        ``L_{t-k}, ..., L_t``, plus the arms of the rule and policy nodes in ``[t-k-1, t]``.
        The estimand does not change.

        Parameters
        ----------
        markov_order : int or None
            The order ``k``, or ``None`` for the full history.

        Returns
        -------
        LongitudinalData
            A copy that records the order.

        Raises
        ------
        ValueError
            When ``markov_order`` is not a non-negative integer.
        DataError
            When the window keeps every earlier node of the design.
        CapabilityError
            When the design has one baseline treatment and no time-varying covariate.
        """
        check_markov_order(markov_order)
        if markov_order is None:
            return replace(self, markov_order=None)
        if self.treatment_held and not any(names for names in self.time_varying_names):
            raise CapabilityError(
                "markov_order= restricts the time-varying history, and this design has none: "
                "one baseline treatment and no time-varying covariate.",
                reason=RefusalReason.INVALID_DECLARATION,
                alternative="Omit markov_order=.",
            )
        if markov_order >= self.n_times - 1:
            raise DataError(
                f"markov_order={markov_order} keeps every earlier node of this "
                f"{self.n_times}-node design, so it restricts nothing. Omit markov_order= to "
                "use the full history"
            )
        return replace(self, markov_order=int(markov_order))

    def with_known_mechanisms(
        self, *, treatment_probabilities: Any = None, censoring_probabilities: Any = None
    ) -> LongitudinalData:
        """Return a copy that declares treatment or censoring factors known.

        :meth:`LTMLE.fit <cleverly.longitudinal.LTMLE.fit>` calls this when it receives the
        declarations beside a prepared container.  The column-name forms need the frame,
        which a container no longer holds, so only the array forms are read here.

        Parameters
        ----------
        treatment_probabilities : array-like or mapping or None
            The treatment declaration, in an array form :meth:`from_frame` reads.
        censoring_probabilities : array-like or mapping or None
            The censoring declaration, in an array form :meth:`from_frame` reads.

        Returns
        -------
        LongitudinalData
            A copy whose :attr:`known_mechanisms` holds the validated declarations.

        Raises
        ------
        ValueError
            If this data already declares a mechanism.
        DataError
            If a declaration is malformed or names columns.
        """
        if self.known_mechanisms is not None:
            raise ValueError(
                "treatment_probabilities or censoring_probabilities is declared twice: on "
                "the data and at fit. Declare them once."
            )
        treatment_columns, censoring_columns = _probability_column_names(
            treatment_probabilities, censoring_probabilities
        )
        if treatment_columns or censoring_columns:
            raise DataError(
                "the declaration names columns, and this call has no frame to read them "
                "from. Pass the arrays, or declare the columns in "
                "LongitudinalData.from_frame or in the fit on the frame."
            )
        return self._declare_known(
            treatment_probabilities=treatment_probabilities,
            censoring_probabilities=censoring_probabilities,
        )

    def _declare_known(
        self,
        *,
        treatment_probabilities: Any,
        censoring_probabilities: Any,
        treatment_columns: Mapping[str, tuple[tuple[Any, str], ...]] | None = None,
        censoring_columns: Mapping[str, str] | None = None,
    ) -> LongitudinalData:
        decisions = self.n_decisions
        self._refuse_known_mixed_nodes(treatment_probabilities)
        return replace(
            self,
            known_mechanisms=known_node_mechanisms(
                treatment_probabilities=treatment_probabilities,
                censoring_probabilities=censoring_probabilities,
                n=self.n,
                treatment_names=self.treatment_names,
                treatment_levels=self.treatment_levels[:decisions],
                continuous_nodes=tuple(
                    self.node_kind(time) != "categorical" for time in range(1, decisions + 1)
                ),
                censoring_names=self.censoring_names,
                treatment_columns=treatment_columns,
                censoring_columns=censoring_columns,
            ),
        )

    @classmethod
    def from_time_to_event(
        cls,
        data: Any,
        *,
        time: str,
        event: str,
        treatment: str,
        baseline: Sequence[str],
        grid: Sequence[float] | None = None,
        causes: Mapping[int, str] | None = None,
        id: str | None = None,
        weights: str | None = None,
        weights_type: str = "probability",
        weights_estimated: bool = False,
        continuous_treatment: bool = False,
        treatment_probabilities: Mapping[Any, str] | None = None,
    ) -> LongitudinalData:
        r"""Build a point-treatment survival container from one time and one event column.

        The frame holds one row per unit: a follow-up time, an event code (``0`` for
        censored), one baseline treatment and the baseline covariates.  The time is binned
        onto a grid :math:`0 = g_0 < g_1 < \dots < g_K`, and node :math:`k` covers
        :math:`(g_{k-1}, g_k]`.  The treatment is one decision held over every node, as
        ``treatment="A"`` declares on :meth:`from_frame`.

        The rules follow the visit structure of Benkeser, Carone and Gilbert (2018,
        Section 2.1):

        ==========================================  ==========================================
        unit                                        nodes
        ==========================================  ==========================================
        event of cause ``j`` at :math:`T \le g_K`   event at the node :math:`k` with
                                                    :math:`g_{k-1} < T \le g_k`
        censored at a grid time :math:`g_j < g_K`   observed through node :math:`j`, then
                                                    censored
        event or censoring after :math:`g_K`, or    event-free and observed through node
        censored at :math:`g_K`                     :math:`K` (the administrative end)
        ==========================================  ==========================================

        An event and a censoring at one grid time are ordered event first, as ``survtmle``
        orders them.  A censoring time strictly between two grid times is refused: it
        cannot be ordered against the events of its interval.

        A grid time before which no follower of a regimen had the event is not refused.
        The regression of such a node is zero, its maximum-likelihood hazard, so the risk
        there is zero.  That risk, and every contrast that reads it, takes the
        ``"constant_node_plugin"`` status: it reports no interval or p-value, the fit warns,
        and ``summary()`` marks the row.  Declare ``horizons=`` at grid times where each
        arm has events, or a coarser grid, for intervals at every reported time.

        Parameters
        ----------
        data : DataFrame
            A pandas or polars frame with one row per unit.
        time : str
            The follow-up time column.  Every time is a positive finite number.
        event : str
            The event-code column: ``0`` for censored, and a positive integer code for each
            absorbing cause.
        treatment : str
            The baseline treatment column.
        baseline : sequence of str
            The covariates measured before the treatment.
        grid : sequence of float or None
            The grid times :math:`g_1 < \dots < g_K`, which are the visit times.  ``None``
            needs integer times and uses :math:`g_k = k` up to the largest event time.  The
            fit costs :math:`K(K+1)/2` sequential regressions per regimen and cause, so a
            default grid on day-level times can hold thousands of nodes.
        causes : mapping of int to str or None
            The label of each nonzero event code.  ``None`` reads codes ``{0, 1}`` as one
            event and any other codes as competing causes labelled ``str(code)``.  A mapping
            declares competing risks, even with one cause.
        id : str or None
            A cluster column.
        weights : str or None
            A column of observation weights, as for :meth:`from_frame`.
        weights_type : str
            How to read ``weights``, as for :meth:`from_frame`.
        weights_estimated : bool
            Declare that the weights came out of a fitted model.
        continuous_treatment : bool
            Declare the treatment a continuous dose.  Node 1 then has a conditional density,
            and a regimen is a modified treatment policy, as ``continuous_treatment=`` on
            :meth:`from_frame` declares.
        treatment_probabilities : mapping of level to str, or None
            The known baseline treatment mechanism: each treatment level mapped to the
            column that holds :math:`P(A = \text{level} \mid W)`.  The fit then divides by
            these probabilities at the decision node and fits no treatment learner, as
            :attr:`known_mechanisms` describes.  The columns are not covariates.  Arrays go
            to :meth:`cleverly.longitudinal.LTMLE.fit`.

        Returns
        -------
        LongitudinalData
            A held survival container with :attr:`time_grid` set.

        Raises
        ------
        DataError
            If a time is missing, not finite or at or below zero; if ``grid=None`` and a
            time is not an integer; if an event code is missing, negative, not an integer
            or not named by ``causes``; if a cause has no event at or before :math:`g_K`;
            if a censoring time lies strictly between two grid times; or if
            ``treatment_probabilities`` is not a mapping of levels to columns.
        ValueError
            If ``grid`` is not a strictly increasing sequence of positive finite times.

        See Also
        --------
        LongitudinalData.to_time_to_event : The inverse map, in grid values.
        cleverly.TimeToEvent : The same input as a study design.

        Examples
        --------
        >>> import pandas as pd
        >>> from cleverly.longitudinal import LongitudinalData
        >>> frame = pd.DataFrame(
        ...     {
        ...         "ftime": [1, 2, 3, 3, 1, 2, 3, 2, 1, 3] * 2,
        ...         "ftype": [1, 0, 1, 0, 1, 1, 0, 1, 0, 1] * 2,
        ...         "arm": [0, 1] * 10,
        ...         "age": [40, 52, 61, 45, 38, 70, 55, 49, 66, 58] * 2,
        ...     }
        ... )
        >>> data = LongitudinalData.from_time_to_event(
        ...     frame, time="ftime", event="ftype", treatment="arm", baseline=["age"]
        ... )
        >>> data.n_times, data.time_grid, data.treatment_held
        (3, (1.0, 2.0, 3.0), True)
        """
        if not is_dataframe(data):
            raise DataError(
                "LongitudinalData.from_time_to_event expects a pandas or polars DataFrame"
            )
        frame = as_frame(data)
        columns = list(frame.columns)
        baseline_names = [str(name) for name in baseline]
        wanted = [str(time), str(event), str(treatment), *baseline_names]
        if id is not None:
            wanted.append(id)
        if weights is not None:
            wanted.append(str(weights))
        known_columns: tuple[tuple[Any, str], ...] = ()
        if treatment_probabilities is not None:
            pairs = probability_columns(treatment_probabilities)
            if pairs is None:
                raise DataError(
                    "from_time_to_event takes treatment_probabilities as a mapping from each "
                    "treatment level to the column that holds its probability. Put the "
                    "probabilities in the frame, or pass the arrays to "
                    "LTMLE.fit(treatment_probabilities=)."
                )
            known_columns = pairs
        probability_names = [name for _, name in known_columns]
        missing = [name for name in [*wanted, *probability_names] if name not in columns]
        if missing:
            raise DataError(f"columns not found in the frame: {missing}; available: {columns}")
        _refuse_duplicates(wanted)
        overlap = sorted(set(probability_names) & set(wanted))
        if overlap:
            raise DataError(f"columns {overlap} are used both as a role and as a known probability")

        times = column_array(frame, str(time))
        codes = column_array(frame, str(event))
        grid_values = _time_grid(times, codes, str(time), grid)
        labels, code_of_cause, competing = _event_causes(codes, str(event), causes)
        n_times = len(grid_values)
        last = grid_values[-1]
        for label, code in zip(labels, code_of_cause, strict=True):
            if not np.any((codes == code) & (times <= last)):
                raise DataError(
                    f"cause {label!r} has no event at or before the last grid time {last:g}; "
                    "its incidence would be zero at every horizon"
                )
        censored = codes == 0.0
        inside = censored & (times < last)
        off_grid = inside & ~np.isin(times, np.asarray(grid_values))
        if np.any(off_grid):
            example = float(times[np.flatnonzero(off_grid)[0]])
            raise DataError(
                f"time column {time!r} has {int(off_grid.sum())} censoring time(s) between grid "
                f"points, for example {example!r}. A censoring that falls inside an interval "
                "cannot be ordered against that interval's events, so the risk at the grid "
                "times is not identified from these rows. Declare the grid at the visit times "
                "when follow-up is checked, so that every censoring time is a grid value or "
                "later than the last grid time"
            )

        grid_array = np.asarray(grid_values, dtype=float)
        nodes = np.arange(1, n_times + 1)
        never = n_times + 1
        # The node of each unit's event, and the node at which a censored unit leaves; a
        # unit that does neither by the last grid time gets ``K + 1`` for both.
        event_node = np.where(
            (codes > 0.0) & (times <= last), _event_node(grid_array, times), never
        )
        censor_node = np.where(inside, _censor_node(grid_array, times), never)
        before_event = nodes[None, :] <= event_node[:, None]
        present_censoring = before_event & (nodes[None, :] <= censor_node[:, None])
        stays = nodes[None, :] < censor_node[:, None]
        censoring = np.where(present_censoring, stays.astype(float), np.nan)
        present_event = before_event & stays
        fired = nodes[None, :] == event_node[:, None]
        event_blocks = np.stack(
            [
                np.where(present_event, (fired & (codes == code)[:, None]).astype(float), np.nan)
                for code in code_of_cause
            ],
            axis=2,
        )
        stamp = [f"{g:g}" for g in grid_values]
        event_names = (
            [[f"{event}={code}@{g}" for g in stamp] for code in code_of_cause]
            if competing
            else [[f"{event}@{g}" for g in stamp]]
        )
        # Node k's indicator is 1{C >= g_k}: a unit censored at g_k is observed through k.
        censoring_names = [f"{time}>={g}" for g in stamp]
        has_censoring = bool(np.any(inside))
        raw = column_array(frame, str(treatment), dtype=object)
        built = cls._build(
            outcome=None,
            event=event_blocks,
            event_names=event_names,
            cause_labels=labels if competing else (),
            baseline=matrix_from_columns(frame, baseline_names) if baseline_names else None,
            baseline_names=baseline_names,
            treatment=np.column_stack([raw] * n_times),
            treatment_names=[str(treatment)],
            censoring=censoring if has_censoring else None,
            censoring_names=censoring_names if has_censoring else [],
            time_varying=[None] * n_times,
            time_varying_names=[[] for _ in range(n_times)],
            cluster=None if id is None else frame[id].to_numpy(),
            cluster_name=id,
            weights=None if weights is None else column_array(frame, weights),
            weights_type=weights_type,
            weights_estimated=weights_estimated,
            weights_name=weights,
            outcome_name=event_names[-1][-1],
            family="binomial",
            backend=backend_of(frame),
            node_kinds=_CONTINUOUS_KIND * n_times if continuous_treatment else (),
            treatment_held=True,
            time_grid=grid_values,
            time_name=str(time),
            event_name=str(event),
            event_codes=tuple(int(code) for code in code_of_cause),
        )
        if not known_columns:
            return built
        node = str(treatment)
        return built._declare_known(
            treatment_probabilities={
                node: {level: column_array(frame, name) for level, name in known_columns}
            },
            censoring_probabilities=None,
            treatment_columns={node: known_columns},
        )

    def to_time_to_event(self) -> tuple[FloatArray, IntArray]:
        """The follow-up time and event code of each unit, in grid values.

        The inverse of :meth:`from_time_to_event` on a law whose event and censoring times
        sit on the grid.  The wide path reads node :math:`k` as the grid time :math:`k`.

        ======================================  ======================================
        unit                                    returned pair
        ======================================  ======================================
        an event of cause ``j`` at node ``k``   :math:`(g_k, j)`
        a censored unit                         its last observed grid time, and ``0``
        an event-free unit at the end           :math:`(g_K, 0)`
        ======================================  ======================================

        A unit censored before node 1 on the wide path returns time ``0``.

        Returns
        -------
        tuple of ndarray
            ``(time, event)``: the grid time as float and the event code as int.

        Raises
        ------
        DataError
            On a container with one end-of-study outcome.
        """
        if self.repeated is not None:
            raise DataError(
                "a repeated outcome has no event time: it is observed at every node and can "
                "return to a previous value, so the container has no follow-up time and "
                "event code to return"
            )
        if self.event is None:
            raise DataError(
                f"{self.outcome_name!r} is an end-of-study outcome, so the container has no "
                "event time to return"
            )
        grid = np.asarray(
            self.time_grid if self.time_grid is not None else range(1, self.n_times + 1),
            dtype=float,
        )
        padded = np.concatenate([[0.0], grid])
        observed_through = _prefix_all(self.uncensored)[:, 1:].sum(axis=1)
        times = padded[np.minimum(observed_through, self.n_times)]
        codes = np.zeros(self.n, dtype=np.int64)
        had = np.asarray(self.event[:, -1], dtype=bool)
        first = np.argmax(self.event, axis=1)
        times[had] = grid[first[had]]
        if self.cause_event is None:
            codes[had] = self.event_codes[0] if self.event_codes else 1
        else:
            by_cause = self.cause_event[np.arange(self.n), first, :]
            cause = np.argmax(by_cause, axis=1)
            table = (
                np.asarray(self.event_codes, dtype=np.int64)
                if self.event_codes
                else np.arange(1, len(self.cause_labels) + 1)
            )
            codes[had] = table[cause[had]]
        return times, codes

    @classmethod
    def _build(
        cls,
        *,
        outcome: Any,
        repeated: FloatArray | None = None,
        repeated_names: Sequence[str] = (),
        event: FloatArray | None = None,
        event_names: Sequence[Sequence[str]] = (),
        cause_labels: Sequence[str] = (),
        baseline: FloatArray | None,
        baseline_names: Sequence[str],
        treatment: Any,
        treatment_names: Sequence[str],
        censoring: FloatArray | None,
        censoring_names: Sequence[str],
        time_varying: Sequence[FloatArray | None],
        time_varying_names: Sequence[Sequence[str]],
        cluster: Any,
        cluster_name: str | None,
        weights: np.ndarray | None = None,
        weights_type: str = "probability",
        weights_estimated: bool = False,
        weights_name: str | None = None,
        outcome_name: str,
        family: str,
        backend: str | None,
        node_kinds: Sequence[NodeKind] = (),
        treatment_components: Sequence[Sequence[str]] = (),
        continuous_components: Sequence[Sequence[bool]] = (),
        treatment_held: bool = False,
        time_grid: Sequence[float] | None = None,
        time_name: str | None = None,
        event_name: str | None = None,
        event_codes: Sequence[int] = (),
    ) -> LongitudinalData:
        survival = event is not None
        if survival:
            event_values = np.asarray(event, dtype=float)
            n = event_values.shape[0]
        else:
            y = np.asarray(outcome, dtype=float).reshape(-1)
            n = y.shape[0]
        if n < MIN_OBSERVATIONS:
            raise DataError(f"need at least {MIN_OBSERVATIONS} observations; got {n}")
        n_times = treatment.shape[1]

        if baseline is None:
            raise DataError(
                "baseline= is empty. A longitudinal fit needs at least one baseline covariate, "
                "because the package fits no intercept-only nuisance design.",
                reason=RefusalReason.IMPLEMENTATION_GAP,
                alternative=NO_ALTERNATIVE,
                roadmap="X63",
            )
        w, w_names, dropped = check_covariates(
            np.asarray(baseline, dtype=float), list(baseline_names)
        )

        uncensored, failed, failed_by_cause = _read_followup(
            censoring,
            censoring_names,
            event_values if survival else None,
            event_names,
            cause_labels,
            n,
            n_times,
        )
        # "Still in the study *before* the treatment decision at t", which is what every
        # node at time t is required to be present for.  Two ways out of the study, and
        # both close it: a unit that was censored earlier has no node here, and so does
        # one that had the event earlier.
        at_risk = np.column_stack(
            [_through(uncensored, t) & _event_free(failed, t) for t in range(n_times)]
        )

        # A repeated outcome is checked first, by its own name and with the outcome's
        # wording, so that every Y_t -- Y_s for s < T too, which is also the first covariate
        # of block s + 1 -- is told to encode a missing value as censoring.  The covariate
        # screen below may still drop the copy while the column stays the outcome of
        # horizon s.
        repeated_values: FloatArray | None = None
        observed_by_node: np.ndarray | None = None
        if repeated is not None and not survival:
            repeated_values = np.asarray(repeated, dtype=float)
            if repeated_values.shape != (n, n_times):
                raise DataError(
                    f"a repeated outcome must be ({n}, {n_times}), rows by time points, "
                    f"but got {repeated_values.shape}"
                )
            observed_by_node = np.column_stack(
                [_through(uncensored, time) for time in range(1, n_times + 1)]
            )
            for time, name in enumerate(repeated_names, start=1):
                _check_presence(
                    repeated_values[:, time - 1],
                    observed_by_node[:, time - 1],
                    str(name),
                    absent=(
                        f"unit(s) are uncensored through time {time} but have no {name}. A "
                        "repeated outcome missing for any other reason is a further node in "
                        "the likelihood: encode it as censoring at that time point, so that its "
                        "probability is estimated and enters the clever covariate."
                    ),
                )

        blocks: list[FloatArray] = []
        names: list[tuple[str, ...]] = []
        for time, (matrix, block_names) in enumerate(
            zip(time_varying, time_varying_names, strict=True), start=1
        ):
            if matrix is None:
                blocks.append(np.zeros((n, 0)))
                names.append(())
                continue
            values = np.asarray(matrix, dtype=float)
            if values.shape[0] != n:
                raise DataError(
                    f"the time-{time} covariate block has {values.shape[0]} rows, expected {n}"
                )
            present = at_risk[:, time - 1]
            for column, name in enumerate(block_names):
                _check_presence(values[:, column], present, str(name))
            # Screened on the same terms as the baseline block.  A constant or duplicated
            # L_t is not harmless here: ``covariate_history`` stacks every block into one
            # design, so it makes the history matrix singular at that node and at every
            # node after it.
            #
            # The screen runs on the rows still under observation, not on the whole
            # column.  A censored unit's later nodes are ``nan`` by construction, which
            # ``check_covariates`` would refuse as missing data; and the rows that decide
            # whether a covariate varies are the rows any model at this node is fitted on.
            block = [str(name) for name in block_names]
            if block and present.any():
                _, kept_names, block_dropped = check_covariates(values[present], block)
                if block_dropped:
                    keep = [block.index(name) for name in kept_names]
                    values = values[:, keep]
                    block = list(kept_names)
                    dropped = [*dropped, *block_dropped]
            blocks.append(values)
            names.append(tuple(block))

        raw_treatment = np.asarray(treatment)
        if raw_treatment.shape != (n, n_times):
            raise DataError(
                f"treatment must be ({n}, {n_times}), rows by treatment nodes, "
                f"but got {raw_treatment.shape}"
            )
        a = np.full((n, n_times), np.nan)
        level_sets: list[tuple[object, ...]] = []
        mixed_values: list[FloatArray | None] = [None] * n_times
        mixed_levels: list[tuple[tuple[object, ...], ...]] = [()] * n_times
        for time in range(1, n_times + 1):
            name = treatment_names[0 if treatment_held else time - 1]
            # Named apart from the ``column`` index and ``values`` matrix of the block
            # loop above: mypy unifies a name's type across the whole function body, so
            # reusing either here is an error rather than a shadow.
            arms = raw_treatment[:, time - 1]
            at_this_node = at_risk[:, time - 1]
            if treatment_held and time > 1:
                # An identity node: the decision's own value, on the rows still in the
                # study.  Its column is a copy of the decision column, so it has no
                # presence of its own to check.
                a[at_this_node, time - 1] = a[at_this_node, 0]
                level_sets.append(level_sets[0])
                continue
            _check_presence(arms, at_this_node, str(name))
            kind = node_kinds[time - 1] if node_kinds else "categorical"
            if kind == "continuous":
                a[at_this_node, time - 1] = _continuous_dose(arms[at_this_node], str(name))
                level_sets.append(())
                continue
            if kind == "mixed":
                encoded, component_sets = _encode_mixed_node(
                    arms[at_this_node],
                    treatment_components[time - 1],
                    continuous_components[time - 1],
                )
                # The slot holds presence only, so every mask that reads it keeps working.
                a[at_this_node, time - 1] = 0.0
                stored = np.full((n, encoded.shape[1]), np.nan)
                stored[at_this_node] = encoded
                mixed_values[time - 1] = stored
                mixed_levels[time - 1] = component_sets
                level_sets.append(_MixedLevels(str(name), time))
                continue
            codes_at_node, levels = _encode_node_treatment(arms[at_this_node], str(name))
            a[at_this_node, time - 1] = codes_at_node
            level_sets.append(levels)

        if survival:
            # ``_read_event`` has already checked every event column on the rows that
            # were still at risk for it, which is the survival analogue of the presence
            # rule below -- and a stricter one, since being at risk is being uncensored
            # *and* event-free.  What is left is the outcome array the container reports:
            # the cumulative indicator at the last node, which is what "the event had
            # happened by the end" means for a unit that left the risk set at any node.
            assert failed is not None  # survival implies the sweep built one
            y = failed[:, -1].astype(float)
            if family not in ("auto", "binomial"):
                raise DataError(
                    f"family={family!r} cannot apply to a survival outcome: an event "
                    "indicator is binary at every node, and the parameter is a "
                    "probability rather than a mean on some other scale"
                )
            resolved = "binomial"
        elif repeated_values is not None and observed_by_node is not None:
            # One family for the whole curve, resolved over every node's observed values.
            resolved = resolve_family(
                repeated_values.T.reshape(-1), observed_by_node.T.reshape(-1), family
            )
            y = repeated_values[:, -1]
        else:
            observed_outcome = _through(uncensored, n_times)
            _check_presence(
                y,
                observed_outcome,
                outcome_name,
                absent=(
                    f"unit(s) were never censored but have no {outcome_name}. An outcome "
                    "missing for any other reason is a further node in the likelihood, not an "
                    "absence: encode it as a final censoring column, so that its probability "
                    "is estimated and enters the clever covariate rather than being assumed one."
                ),
            )

            resolved = resolve_family(y, observed_outcome, family)

        codes = None if cluster is None else encode_clusters(cluster, cluster_name or "id")

        # Read exactly as the point-treatment container reads them, and by the same
        # functions: the normalisation, the refusals and the warnings are statements about
        # what a weight *means*, and they cannot mean one thing at one time point and
        # another over several.
        obs_weights, spec = _prepare_weights(
            weights,
            n,
            weights_type=weights_type,
            weights_estimated=weights_estimated,
            weights_name=weights_name,
        )

        return cls(
            outcome=y,
            baseline=w,
            baseline_names=tuple(w_names),
            treatment=a,
            treatment_names=tuple(str(name) for name in treatment_names),
            treatment_levels=tuple(level_sets),
            uncensored=uncensored,
            time_varying=tuple(blocks),
            time_varying_names=tuple(names),
            family=resolved,
            weights=obs_weights,
            outcome_name=outcome_name,
            event=failed,
            event_names=tuple(str(name) for block in event_names for name in block),
            # Only when the mapping form was passed, so a one-cause mapping reports a
            # cumulative incidence *by declaration* rather than a fit being classified by
            # how many causes its data happened to contain.
            cause_event=failed_by_cause if cause_labels else None,
            cause_labels=tuple(str(label) for label in cause_labels),
            censoring_names=tuple(str(name) for name in censoring_names),
            cluster=codes,
            cluster_name=cluster_name,
            weights_name=weights_name,
            weight_spec=spec,
            dropped_covariates=tuple(dropped),
            backend=backend,
            node_kinds=(
                tuple(node_kinds) if any(kind != "categorical" for kind in node_kinds) else ()
            ),
            treatment_components=(
                tuple(tuple(block) for block in treatment_components)
                if any(len(block) > 1 for block in treatment_components)
                else ()
            ),
            component_values=(
                tuple(mixed_values) if any(v is not None for v in mixed_values) else ()
            ),
            component_levels=(
                tuple(mixed_levels) if any(v is not None for v in mixed_values) else ()
            ),
            treatment_held=bool(treatment_held),
            time_grid=None if time_grid is None else tuple(float(g) for g in time_grid),
            time_name=time_name,
            event_name=event_name,
            event_codes=tuple(int(code) for code in event_codes),
            repeated=(
                None
                if repeated is None
                else np.where(
                    np.column_stack([_through(uncensored, t) for t in range(1, n_times + 1)]),
                    np.asarray(repeated, dtype=float),
                    np.nan,
                )
            ),
            repeated_names=tuple(str(name) for name in repeated_names),
        )

    # ------------------------------------------------------------- properties

    @property
    def n(self) -> int:
        return int(self.outcome.shape[0])

    @property
    def n_times(self) -> int:
        """Number of treatment nodes, ``T``."""
        return int(self.treatment.shape[1])

    @property
    def n_decisions(self) -> int:
        """Number of treatment decisions: one on a held design, else :attr:`n_times`."""
        return 1 if self.treatment_held else self.n_times

    def decision_name(self, time: int) -> str:
        """The treatment column that decides node ``time``, counted from one.

        Parameters
        ----------
        time : int
            The treatment node, counted from one.

        Returns
        -------
        str
            The node's own column, or the baseline column at every node of a held design.
        """
        return self.treatment_names[0 if self.treatment_held else time - 1]

    def is_held_node(self, time: int) -> bool:
        """Whether node ``time``, counted from one, is an identity node of a held design.

        Parameters
        ----------
        time : int
            The treatment node, counted from one.

        Returns
        -------
        bool
            ``True`` at every node after the first on a held design.
        """
        return self.treatment_held and time > 1

    def node_kind(self, time: int) -> NodeKind:
        """The :data:`NodeKind` of node ``time``, counted from one.

        Parameters
        ----------
        time : int
            The treatment node, counted from one.

        Returns
        -------
        str
            ``"categorical"``, ``"continuous"`` or ``"mixed"``.
        """
        return self.node_kinds[time - 1] if self.node_kinds else "categorical"

    def is_continuous_node(self, time: int) -> bool:
        """Whether node ``time``, counted from one, holds one continuous dose.

        A mixed node has no answer here, so the call raises there rather than let a reader
        of one column take the categorical branch.  Read :meth:`node_kind` first.

        Parameters
        ----------
        time : int
            The treatment node, counted from one.

        Returns
        -------
        bool
            ``True`` for a node of one column declared in ``continuous_treatment=``.

        Raises
        ------
        CapabilityError
            At a mixed node.
        """
        kind = self.node_kind(time)
        if kind == "mixed":
            raise CapabilityError.from_refusal(_mixed_reader(self, time, "is_continuous_node"))
        return kind == "continuous"

    @property
    def has_continuous_node(self) -> bool:
        """Whether any node holds a continuous dose, alone or as a component."""
        return any(kind != "categorical" for kind in self.node_kinds)

    @property
    def continuous_columns(self) -> tuple[str, ...]:
        """The columns declared continuous, in node order: a dose node or a dose component."""
        columns: list[str] = []
        for time in range(1, self.n_decisions + 1):
            kind = self.node_kind(time)
            if kind == "continuous":
                columns.append(self.decision_name(time))
            elif kind == "mixed":
                columns.extend(
                    component.name
                    for component in self.node_components(time)
                    if component.continuous
                )
        return tuple(columns)

    @property
    def has_mixed_node(self) -> bool:
        """Whether any node is a vector with a continuous component."""
        return "mixed" in self.node_kinds

    def node_present(self, time: int) -> BoolArray:
        """``(n,)`` whether each unit holds a treatment at node ``time``, for every kind.

        Parameters
        ----------
        time : int
            The treatment node, counted from one.

        Returns
        -------
        BoolArray
            ``False`` on a row censored before the node or past an absorbing event.
        """
        return np.asarray(~np.isnan(self.treatment[:, time - 1]), dtype=bool)

    def node_codes(self, time: int) -> FloatArray:
        """``(n,)`` the one column of node ``time``: level codes, or a dose.

        Parameters
        ----------
        time : int
            The treatment node, counted from one.

        Returns
        -------
        FloatArray
            The column, ``nan`` where the unit holds no treatment.

        Raises
        ------
        CapabilityError
            At a mixed node, whose components :meth:`node_values` returns.
        """
        if self.node_kind(time) == "mixed":
            raise CapabilityError.from_refusal(_mixed_reader(self, time, "node_codes"))
        return self.treatment[:, time - 1]

    def node_values(self, time: int) -> FloatArray:
        """``(n, d_t)`` the treatment of node ``time``, one column per component.

        A node of one column returns that column as ``(n, 1)``.  A mixed node returns its
        component codes and doses.

        Parameters
        ----------
        time : int
            The treatment node, counted from one.

        Returns
        -------
        FloatArray
            The values, ``nan`` where the unit holds no treatment.
        """
        if self.node_kind(time) == "mixed":
            values = self.component_values[time - 1]
            assert values is not None
            return values
        return self.treatment[:, time - 1 : time]

    def component_names(self, time: int) -> tuple[str, ...]:
        """The columns of node ``time``: its components, or the one column it reads.

        Parameters
        ----------
        time : int
            The treatment node, counted from one.

        Returns
        -------
        tuple of str
            One name per column of :meth:`node_values`.
        """
        if self.treatment_components:
            return self.treatment_components[0 if self.treatment_held else time - 1]
        return (self.decision_name(time),)

    def node_components(self, time: int) -> tuple[JointComponent, ...]:
        """The components of mixed node ``time``, as a joint policy reads them.

        Parameters
        ----------
        time : int
            A mixed node, counted from one.

        Returns
        -------
        tuple of JointComponent
            One per column of :meth:`node_values`, with its labels, ``()`` for a dose.
        """
        if self.node_kind(time) != "mixed":
            raise ValueError(f"node {time} is not a mixed node")
        return tuple(
            JointComponent(name, levels)
            for name, levels in zip(
                self.component_names(time), self.component_levels[time - 1], strict=True
            )
        )

    def component_design(self, time: int, values: FloatArray | None = None) -> FloatArray:
        """The design block of mixed node ``time``: doses as themselves, labels as indicators.

        A continuous component enters as its dose, one column, and the indicator of each of
        its declared atoms (:func:`~cleverly.data.validate.atom_block`).  A categorical
        component enters as :func:`~cleverly.data.validate.arm_indicators` of its own codes,
        drop-first, as a categorical node does in :meth:`history_design`.  The joint level
        code of the components is never a column.  A missing value fills with zero, on rows
        no fit reads.

        Parameters
        ----------
        time : int
            A mixed node, counted from one.
        values : FloatArray or None
            ``(n, d_t)`` component codes and doses, such as the values a policy assigns.
            ``None`` reads the observed ones.

        Returns
        -------
        FloatArray
            The block, one row per unit.
        """
        observed = self.node_values(time) if values is None else np.asarray(values, dtype=float)
        atoms = self.component_atoms(time)
        names = self.component_names(time)
        columns = []
        for index, levels in enumerate(self.component_levels[time - 1]):
            column = np.nan_to_num(observed[:, index], nan=0.0)
            if not levels:
                # A dose with declared atoms adds the indicator of each atom, read from the
                # values given, as a dose of one column does (:meth:`dose_block`).
                columns.append(atom_block(column, atoms.get(names[index], ())))
            else:
                columns.append(arm_indicators(column, len(levels)))
        return np.hstack(columns)

    def _refuse_known_mixed_nodes(self, treatment_probabilities: Any) -> None:
        """Refuse a known treatment mechanism declared at a mixed node, before any learner."""
        if treatment_probabilities is None or not self.has_mixed_node:
            return
        mixed = [
            self.treatment_names[time - 1]
            for time in range(1, self.n_times + 1)
            if self.node_kind(time) == "mixed"
        ]
        if isinstance(treatment_probabilities, Mapping):
            mixed = [name for name in mixed if name in treatment_probabilities]
        if mixed:
            raise CapabilityError(
                f"treatment_probabilities declares the vector node(s) {mixed}, which hold a "
                "continuous component. A known mechanism is a probability per level, and a "
                "vector node with a dose has no levels. A known density is not supported.",
                reason=RefusalReason.IMPLEMENTATION_GAP,
                alternative="Estimate that node's policy ratio with ratio='classifier'.",
                roadmap="X63",
            )

    @property
    def n_clusters(self) -> int:
        if self.cluster is None:
            return self.n
        return int(np.unique(self.cluster).size)

    @property
    def has_censoring(self) -> bool:
        return bool(self.censoring_names) and bool(not self.uncensored.all())

    @property
    def is_weighted(self) -> bool:
        return bool(not np.allclose(self.weights, 1.0))

    @property
    def effective_n(self) -> float:
        """Kish effective sample size of the observation weights, ``(sum w)^2 / sum w^2``.

        The same diagnostic quantity, for the same reason, as
        :attr:`cleverly.data.CausalData.effective_n`.  It does not tune longitudinal
        ``g_bounds``: LTMLE has no automatic bound-selection procedure and its default is
        the fixed cumulative pair ``(0.01, 1.0)``.
        """
        return effective_sample_size(self.weights)

    def weight_report(self) -> WeightReport:
        """Effective sample size, weight concentration and the estimand statement.

        See :mod:`cleverly.data.weighting`; ``print(data.weight_report().summary())``.
        """
        return describe_weights(self.weights, self.weight_spec)

    @property
    def is_survival(self) -> bool:
        """Whether the outcome is an event indicator at every node rather than one at the end.

        Which it is decides what the fit *reports* -- a curve or a number -- so every
        caller that branches on it is branching on the parameter, not on a storage detail.
        """
        return self.event is not None

    @property
    def is_competing(self) -> bool:
        """Whether the fit declared more than one absorbing state per node.

        True when ``outcome=`` was passed as a **mapping** of cause to columns, including
        a mapping with a single cause: what a fit reports is a statement it made, not one
        inferred from how many causes its sample happened to contain.  A competing-risks
        fit is a survival fit, so :attr:`is_survival` is true of it too.
        """
        return bool(self.cause_labels)

    @property
    def is_repeated(self) -> bool:
        """Whether the outcome is a non-absorbing :class:`Repeated` outcome at every node."""
        return self.repeated is not None

    @property
    def reports_curve(self) -> bool:
        """Whether every parameter is indexed by a horizon: a survival or a repeated outcome.

        A caller that names, keys or groups parameters by horizon branches on this.  A
        caller that relies on the absorbing structure -- a risk set, a cumulative risk, a
        survival complement -- branches on :attr:`is_survival`.
        """
        return self.is_survival or self.is_repeated

    def outcome_at(self, time: int) -> FloatArray:
        """The outcome of horizon ``time``: :math:`Y_t` of a repeated outcome, else ``Y``.

        Parameters
        ----------
        time : int
            The horizon, counted from one.  An end-of-study outcome has only ``T``.

        Returns
        -------
        FloatArray
            One value per unit, ``nan`` where the outcome is not observed.

        Raises
        ------
        DataError
            On a survival outcome, which has event indicators rather than an outcome per
            horizon, or at a horizon the outcome does not have.
        """
        if self.is_survival:
            raise DataError(
                "a survival outcome is an event indicator at every node, read through the "
                "risk sets; it has no outcome value per horizon"
            )
        if self.repeated is not None:
            if not 1 <= time <= self.n_times:
                raise DataError(f"time {time} is outside 1..{self.n_times}")
            return np.asarray(self.repeated[:, time - 1], dtype=float)
        if time != self.n_times:
            raise DataError(
                f"{self.outcome_name!r} is an end-of-study outcome, observed only at "
                f"t={self.n_times}; got t={time}"
            )
        return np.asarray(self.outcome, dtype=float)

    def observed_outcomes(self) -> FloatArray:
        """Every observed outcome value: each horizon's on its own uncensored rows.

        One end-of-study outcome gives its values on the rows never censored.  A repeated
        outcome pools every node's, so one scale and one support cover the whole curve.

        Returns
        -------
        FloatArray
            The observed values, in node order.

        Raises
        ------
        DataError
            On a survival outcome, whose observed values are event indicators.
        """
        if self.is_survival:
            raise DataError("a survival outcome has event indicators, not outcome values")
        if self.repeated is None:
            return np.asarray(self.outcome[self.uncensored_through(self.n_times)], dtype=float)
        return np.concatenate(
            [
                self.repeated[self.uncensored_through(time), time - 1]
                for time in range(1, self.n_times + 1)
            ]
        )

    def subset(self, index: Any) -> LongitudinalData:
        """A copy holding only the selected units, each with its whole trajectory.

        The full-refit bootstrap resamples with this. A unit's row carries every node,
        so a censored row keeps its ``nan`` pattern and an event row keeps its absorbing
        structure. Weights are renormalised to mean one, as
        :meth:`cleverly.data.CausalData.subset` renormalises them, and the scale factor
        is folded into :attr:`weight_spec`. Cluster codes are re-derived so they stay
        contiguous. A cluster bootstrap overrides them afterwards with one code per drawn
        occurrence.

        Parameters
        ----------
        index : array-like of int or bool
            The selected rows. Repeats are kept, as a bootstrap resample needs.

        Returns
        -------
        LongitudinalData
            The subset, with every node array indexed by row.

        Raises
        ------
        DataError
            When the subset holds fewer rows than a fit needs.
        """
        idx = np.asarray(index)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        if idx.size < MIN_OBSERVATIONS:
            raise DataError(f"subset has {idx.size} rows; need at least {MIN_OBSERVATIONS}")
        cluster = (
            None
            if self.cluster is None
            else np.asarray(np.unique(self.cluster[idx], return_inverse=True)[1], dtype=np.int64)
        )
        panel = self.panel
        if panel is not None and self.cluster is not None:
            # ``cluster`` is re-coded over the selected codes in sorted order, so the labels
            # it indexes are the selected codes' labels in that order.
            panel = replace(
                panel,
                cluster_labels=tuple(
                    panel.cluster_labels[code] for code in np.unique(self.cluster[idx])
                ),
            )
        selected = self.weights[idx]
        mean = float(selected.mean())
        return replace(
            self,
            outcome=self.outcome[idx],
            baseline=self.baseline[idx],
            treatment=self.treatment[idx],
            component_values=tuple(
                None if values is None else values[idx] for values in self.component_values
            ),
            uncensored=self.uncensored[idx],
            time_varying=tuple(block[idx] for block in self.time_varying),
            weights=check_weights(selected, idx.size),
            event=None if self.event is None else self.event[idx],
            cause_event=None if self.cause_event is None else self.cause_event[idx],
            repeated=None if self.repeated is None else self.repeated[idx],
            cluster=cluster,
            weight_spec=self.weight_spec.rescaled(self.weight_spec.scale * mean),
            known_mechanisms=(
                None if self.known_mechanisms is None else self.known_mechanisms.subset(idx)
            ),
            unit_ids=None if self.unit_ids is None else np.asarray(self.unit_ids)[idx],
            panel=panel,
        )

    def with_treatment(
        self,
        treatment: Any,
        *,
        known_mechanisms: KnownNodeMechanisms | Literal["keep"] | None = "keep",
    ) -> LongitudinalData:
        """A copy whose treatment matrix is replaced, every other array unchanged.

        The refutations of :func:`cleverly.validation.refute` replace the treatment of a
        longitudinal fit with a placebo. A ``nan`` treatment means that the unit is not at
        risk at the node, which the other arrays decide, so the replacement must hold
        ``nan`` exactly where the observed treatment does.

        Parameters
        ----------
        treatment : array-like
            The ``(n, T)`` replacement, in the codes of :attr:`treatment` at a categorical
            node and as a dose at a continuous node.
        known_mechanisms : KnownNodeMechanisms or None or "keep"
            The declared mechanism of the replacement. ``"keep"`` keeps the declaration of
            this container, which is right only when the declaration does not read the
            treatment.

        Returns
        -------
        LongitudinalData
            The copy.

        Raises
        ------
        CapabilityError
            On a container with a mixed node, whose components a matrix of codes cannot
            replace.
        DataError
            When the replacement has another shape, another ``nan`` pattern, a code outside
            a node's levels, or a held node that differs from the decision.
        """
        if self.has_mixed_node:
            raise CapabilityError(
                "a treatment matrix cannot replace a vector node with a continuous component, "
                "whose components it does not hold.",
                reason=RefusalReason.IMPLEMENTATION_GAP,
                alternative=NO_ALTERNATIVE,
                roadmap="X82",
            )
        values = np.asarray(treatment, dtype=float)
        if values.shape != self.treatment.shape:
            raise DataError(
                f"the replacement treatment has shape {values.shape}; the container holds "
                f"{self.treatment.shape}"
            )
        if not np.array_equal(np.isnan(values), np.isnan(self.treatment)):
            raise DataError(
                "the replacement treatment holds nan on other cells than the observed one. A "
                "nan treatment means that the unit is not at risk at the node, which the "
                "censoring and event arrays decide"
            )
        for time in range(1, self.n_times + 1):
            column = values[:, time - 1]
            present = column[~np.isnan(column)]
            if self.is_held_node(time):
                first = values[~np.isnan(column), 0]
                if not np.array_equal(present, first):
                    raise DataError(
                        f"node {time} of a held design must repeat the decision at node 1"
                    )
                continue
            if self.node_kind(time) == "continuous":
                if not np.all(np.isfinite(present)):
                    raise DataError(f"the replacement dose at node {time} is not finite")
                continue
            levels = len(self.treatment_levels[time - 1])
            if not np.all(np.isin(present, np.arange(levels, dtype=float))):
                raise DataError(
                    f"the replacement treatment at node {time} holds a code outside the "
                    f"{levels} level code(s) of the node"
                )
        declared = self.known_mechanisms if known_mechanisms == "keep" else known_mechanisms
        return replace(self, treatment=values, known_mechanisms=declared)

    def with_outcome(self, values: Any, *, name: str = "replacement outcome") -> LongitudinalData:
        """A copy whose outcome is replaced, every other array unchanged.

        The negative-control refutation of :func:`cleverly.validation.refute` refits on a
        control in place of the outcome. The control must be observed wherever the outcome
        is. A value where the outcome is missing is never read and becomes ``nan``. A repeated
        outcome's copies in the covariate blocks keep the original values, so the adjustment
        set of every node is unchanged.

        Parameters
        ----------
        values : array-like
            ``(n,)`` for an end-of-study outcome, ``(n, T)`` for a repeated outcome.
        name : str
            What the caller calls the replacement. Every refusal names it.

        Returns
        -------
        LongitudinalData
            The copy, with the family resolved from the replacement.

        Raises
        ------
        DataError
            On a survival outcome, on the wrong shape, or on a missing or non-finite value
            where the outcome is observed.
        """
        if self.is_survival:
            raise DataError(
                f"{name} cannot replace a survival outcome: the event indicators define the "
                "risk sets, so a replacement would change the population"
            )
        array = np.asarray(values, dtype=float)
        if self.repeated is not None:
            current = np.asarray(self.repeated, dtype=float)
        else:
            current = np.asarray(self.outcome, dtype=float)
        if array.shape != current.shape:
            raise DataError(f"{name} has shape {array.shape}; expected {current.shape}")
        observed = ~np.isnan(current)
        if not np.all(np.isfinite(array[observed])):
            raise DataError(f"{name} must be finite wherever the outcome is observed")
        cleaned = np.where(observed, array, np.nan)
        family = resolve_family(cleaned.T.reshape(-1), observed.T.reshape(-1), "auto")
        if self.repeated is not None:
            return replace(self, repeated=cleaned, outcome=cleaned[:, -1], family=family)
        return replace(self, outcome=cleaned, family=family)

    def with_events(self, event: Any, uncensored: Any, cause_event: Any = None) -> LongitudinalData:
        """A copy whose survival outcome is replaced, with each exit masked.

        The held-history generated outcomes of :func:`cleverly.validation.refute` draw new
        events on the observed treatments and covariates. A unit leaves at its first event
        or at its first censoring node. Its treatment and covariates after that node become
        ``nan``, exactly as :meth:`from_frame` stores them. A fit that declared no censoring
        node gets one per node, because the replacement can censor.

        Parameters
        ----------
        event : array-like
            ``(n, T)`` cumulative "had the event at or before ``t``".
        uncensored : array-like
            ``(n, T)`` "still under observation after ``t``". It must be monotone and must
            not drop a unit that is not present at that node.
        cause_event : array-like or None
            ``(n, T, J)`` cumulative per-cause events of a competing-risk fit.

        Returns
        -------
        LongitudinalData
            The copy.

        Raises
        ------
        DataError
            On a fit without a survival outcome, on the wrong shape, or on an event or a
            censoring node at which the unit is not present.
        """
        if not self.is_survival:
            raise DataError("with_events replaces a survival outcome, and this fit has none")
        events = np.asarray(event, dtype=bool)
        kept = np.asarray(uncensored, dtype=bool)
        shape = (self.n, self.n_times)
        if events.shape != shape or kept.shape != shape:
            raise DataError(f"with_events needs (n, T) = {shape} arrays")
        if np.any(events[:, 1:] < events[:, :-1]) or np.any(kept[:, 1:] > kept[:, :-1]):
            raise DataError("with_events needs a cumulative event and a monotone censoring")
        present = ~np.isnan(np.asarray(self.treatment, dtype=float))
        # A unit is present at node t when it is uncensored and event-free through t - 1.
        reach = np.ones(shape, dtype=bool)
        reach[:, 1:] = kept[:, :-1] & ~events[:, :-1]
        if np.any(reach & ~present):
            raise DataError("with_events reaches a node at which a unit holds no treatment")
        first = events.copy()
        first[:, 1:] &= ~events[:, :-1]
        if np.any(first & ~(kept & reach)):
            raise DataError("with_events records an event on a unit not under observation")
        treatment = np.where(reach, self.treatment, np.nan)
        blocks = tuple(
            np.where(reach[:, [node]], block, np.nan) if block.shape[1] else block
            for node, block in enumerate(self.time_varying)
        )
        causes = None
        if self.cause_event is not None:
            causes = np.asarray(cause_event, dtype=bool)
            if causes.shape != self.cause_event.shape or np.any(causes.any(axis=2) != events):
                raise DataError("with_events needs per-cause events that add up to the event")
        names = self.censoring_names or tuple(
            f"recoded_exit_{node}" for node in range(1, self.n_times + 1)
        )
        return replace(
            self,
            outcome=events[:, -1].astype(float),
            event=events,
            cause_event=causes,
            uncensored=kept & reach,
            treatment=treatment,
            time_varying=blocks,
            censoring_names=names,
        )

    def with_extra_covariate(self, values: Any, name: str) -> LongitudinalData:
        """A copy with one more baseline covariate, appended after the others.

        Parameters
        ----------
        values : array-like
            One finite value per unit.
        name : str
            The new column's name, which no column of the container may hold.

        Returns
        -------
        LongitudinalData
            The copy.

        Raises
        ------
        DataError
            When the values are not one finite value per unit, or the name is taken.
        """
        column = np.asarray(values, dtype=float).reshape(-1)
        if column.shape[0] != self.n or not np.all(np.isfinite(column)):
            raise DataError(f"a new baseline covariate needs {self.n} finite values")
        self._refuse_taken_name(name)
        return replace(
            self,
            baseline=np.column_stack([self.baseline, column]),
            baseline_names=(*self.baseline_names, name),
        )

    def with_node_covariates(self, values: Any, names: Sequence[str]) -> LongitudinalData:
        """A copy with one more time-varying covariate at every node, appended to each block.

        Parameters
        ----------
        values : array-like
            ``(n, T)``: column ``t`` joins the covariates measured before the node-``t``
            treatment. A value on a row that is not at risk at the node is never read.
        names : sequence of str
            One new name per node, which no column of the container may hold.

        Returns
        -------
        LongitudinalData
            The copy.

        Raises
        ------
        DataError
            When the values are not ``(n, T)`` and finite, or a name is taken or repeated.
        """
        block = np.asarray(values, dtype=float)
        if block.shape != (self.n, self.n_times) or not np.all(np.isfinite(block)):
            raise DataError(
                f"new node covariates need finite values of shape {(self.n, self.n_times)}"
            )
        labels = tuple(names)
        if len(labels) != self.n_times or len(set(labels)) != len(labels):
            raise DataError(f"new node covariates need {self.n_times} distinct names")
        for name in labels:
            self._refuse_taken_name(name)
        return replace(
            self,
            time_varying=tuple(
                np.column_stack([existing, block[:, index]])
                for index, existing in enumerate(self.time_varying)
            ),
            time_varying_names=tuple(
                (*existing, labels[index]) for index, existing in enumerate(self.time_varying_names)
            ),
        )

    def _refuse_taken_name(self, name: str) -> None:
        """Refuse a new column name that the container already uses."""
        taken = {
            *self.baseline_names,
            *(item for block in self.time_varying_names for item in block),
            *self.treatment_names,
            *self.censoring_names,
            *self.event_names,
            self.outcome_name,
        }
        if not isinstance(name, str) or not name or name in taken:
            raise DataError(f"the new column name {name!r} is empty or already in use")

    # ------------------------------------------------------------------ masks

    def uncensored_through(self, time: int) -> BoolArray:
        """Still under observation after every node up to and including ``time``.

        ``time=0`` is everybody: no censoring node has been passed yet.
        """
        return _through(self.uncensored, time)

    def followed_through(self, assignment: Assignment, time: int) -> BoolArray:
        """Took the regimen's arm at every node up to and including ``time``.

        ``assignment`` is either one arm per node -- a static plan -- or the ``(n, T)``
        matrix a dynamic rule produces, where the arm a unit was to be given depends on
        its own history.  Both are read through :func:`assignment_matrix`, so this is one
        comparison either way rather than a branch.

        A unit whose treatment is missing at a node it was censored before has not
        followed anything, and comes back false -- which is what the mask is for, since
        such a unit contributes to no regression from that node on.  That holds under a
        rule too: ``nan`` compares false against whatever arm the rule assigned it.
        """
        plan = assignment_matrix(assignment, self.n, self.n_times)
        mask = np.ones(self.n, dtype=bool)
        for t in range(1, time + 1):
            mask &= self.treatment[:, t - 1] == plan[:, t - 1]
        return mask

    def at_risk(self, assignment: Assignment, time: int) -> BoolArray:
        """Rows whose history :math:`H_t` is observed, regimen-consistent and event-free.

        The sequential regression at ``time`` predicts for exactly these rows.  Without a
        survival outcome the regression at ``time - 1`` is *fitted* on exactly these rows
        too -- the two sets are the same object, which is what makes the recursion close.
        With one the identity generalises rather than holds:

        .. code-block:: text

            at_risk(t + 1) == following(t) & event-free at t

        because a unit that has the event at ``t`` **is** in node ``t``'s regression, with
        a pseudo-outcome of one, and is not in node ``t + 1``'s.  That is the one place
        here a reader is likely to get the populations backwards, so
        :func:`tests.unit.test_longitudinal_data.test_masks_line_up_with_the_recursion`
        pins the general form and not the special case.  A dynamic rule does not weaken
        it either: it changes *which* rows, not the relation between the two masks.
        """
        return (
            self.uncensored_through(time - 1)
            & self.followed_through(assignment, time - 1)
            & _event_free(self.event, time - 1)
        )

    def following(self, assignment: Assignment, time: int) -> BoolArray:
        """Rows still on the regimen and under observation *after* ``time``.

        The clever covariate at ``time`` is supported on these rows and zero elsewhere.
        Note which event node this reads: ``time - 1``, not ``time``.  These are the rows
        the node's regression is fitted on, and a unit that had the event *at* ``time`` is
        one of them -- it is the observation that the event happened.
        """
        return (
            self.uncensored_through(time)
            & self.followed_through(assignment, time)
            & _event_free(self.event, time - 1)
        )

    def regimen_masks(
        self, assignment: Assignment, *, support: BoolArray | None = None
    ) -> RegimenMasks:
        """Every node's masks for one regimen, scanned once.

        :meth:`at_risk` and :meth:`following` each rebuild a prefix from scratch --
        ``uncensored[:, :t].all(axis=1)`` and a loop of ``t`` comparisons -- so calling
        them at every node costs :math:`O(T^2 n)` per regimen, and a survival fit pays
        that once per horizon on top.  The masks are *prefix scans of a conjunction*, so
        carrying them down the nodes computes the same arrays in :math:`O(T n)`.

        The two methods stay, answer the same thing, and are what a caller wanting one node
        should use; this is what the recursion uses.

        ``support`` is the ``(n, T)`` indicator that the plan's intervention density is
        positive at the observed arm, which a plan with a policy node passes
        (:meth:`~cleverly.longitudinal.regimen.Plan.masks`).  A row then follows the plan
        through ``t`` when its observed arm has positive density at every node up to ``t``.
        ``None`` reads the comparison alone, and every deterministic plan passes ``None``.
        """
        matches = self.treatment == assignment_matrix(assignment, self.n, self.n_times)
        if support is not None:
            matches = matches & np.asarray(support, dtype=bool)
        return RegimenMasks(
            uncensored=_prefix_all(self.uncensored),
            followed=_prefix_all(matches),
            event_free=self._event_free_prefix(),
        )

    def _event_free_prefix(self) -> BoolArray:
        """``(n, T+1)`` "no event through ``t``", ``t = 0`` being everybody.

        A column read rather than a scan, because :attr:`event` is stored cumulatively --
        so unlike the other two this was never quadratic, and it is assembled here only so
        that one object carries all three.
        """
        if self.event is None:
            return np.ones((self.n, self.n_times + 1), dtype=bool)
        out = np.ones((self.n, self.n_times + 1), dtype=bool)
        out[:, 1:] = ~self.event
        return out

    def event_free_through(self, time: int) -> BoolArray:
        """Had not yet had the event after every node up to and including ``time``.

        ``time=0`` is everybody, and so is every ``time`` on a fit with one end-of-study
        outcome: there is no event node for a unit to have passed.
        """
        free = _event_free(self.event, time)
        if free is True:
            return np.ones(self.n, dtype=bool)
        return np.asarray(free, dtype=bool)

    def event_by(self, time: int, cause: str | None = None) -> FloatArray:
        """``1.0`` where an event had happened at or before ``time``, else ``0.0``.

        The target of the sequential regression at ``time`` on a survival fit, read on
        :meth:`following` -- where "at or before" and "at" coincide, since every row there
        was event-free entering the node.

        ``cause=None`` answers for **any** cause, which is what the recursion's survival
        factor needs and what every mask is built on.  Naming a cause answers for that one
        alone, which is what its pseudo-outcome's *numerator* needs.  The two come apart
        exactly when there is more than one cause, and keeping them one call with one
        argument is deliberate: the composition
        ``event_by(t, cause) + (1 - event_by(t)) * carried`` then reads as the asymmetry it
        is, rather than as two similarly-named quantities a reader must tell apart.
        """
        if self.event is None:
            raise DataError(
                f"{self.outcome_name!r} is an end-of-study outcome, so there is no event "
                f"node at time {time}. Pass one outcome column per time point to fit a "
                "survival outcome"
            )
        if not 1 <= time <= self.n_times:
            raise DataError(f"time {time} is outside 1..{self.n_times}")
        if cause is None:
            return self.event[:, time - 1].astype(float)
        if self.cause_event is None:
            raise DataError(
                f"this fit has one absorbing event ({self.outcome_name!r}), so there is no "
                f"cause {cause!r} to ask about. Declare competing risks by passing a "
                "mapping of cause to its indicator column per time point"
            )
        if cause not in self.cause_labels:
            raise DataError(f"unknown cause {cause!r}; this fit declared {list(self.cause_labels)}")
        return self.cause_event[:, time - 1, self.cause_labels.index(cause)].astype(float)

    # ----------------------------------------------------------------- design

    def covariate_history(self, time: int) -> FloatArray:
        """``[W, L_1, ..., L_t]`` as a finite matrix.

        Rows that were censored before ``time`` have no history to speak of; their
        entries are filled with zeros so that a learner can be *called* on the whole
        matrix in one pass.  Nothing reads the predictions at those rows: every use is
        masked by :meth:`at_risk` first, and
        ``test_longitudinal_data.test_the_fill_cannot_reach_the_estimate`` pins that the
        fill cannot leak by replacing the filled entries with ``1e6`` and checking the
        estimate and every influence curve come back bit-for-bit identical.
        """
        if not 1 <= time <= self.n_times:
            raise DataError(f"time {time} is outside 1..{self.n_times}")
        blocks = [self.baseline, *self.time_varying[:time]]
        matrix = np.hstack([block for block in blocks if block.shape[1]])
        return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    def window_start(self, time: int) -> int:
        """The first covariate node the nuisance designs at ``time`` read.

        Parameters
        ----------
        time : int
            The node, counted from one.

        Returns
        -------
        int
            ``max(1, time - k)`` under :attr:`markov_order` ``k``, and ``1`` without one.
        """
        if self.markov_order is None:
            return 1
        return max(1, time - self.markov_order)

    def window_covariates(self, time: int) -> FloatArray:
        """``[W, L_s, ..., L_t]`` from :meth:`window_start` ``s``, as a finite matrix.

        Without :attr:`markov_order` it is :meth:`covariate_history` itself.

        Parameters
        ----------
        time : int
            The node, counted from one.

        Returns
        -------
        FloatArray
            The baseline columns and the covariate blocks of the window nodes.
        """
        if self.markov_order is None:
            return self.covariate_history(time)
        if not 1 <= time <= self.n_times:
            raise DataError(f"time {time} is outside 1..{self.n_times}")
        start = self.window_start(time)
        blocks = [self.baseline, *self.time_varying[start - 1 : time]]
        present = [block for block in blocks if block.shape[1]]
        if not present:
            return np.zeros((self.n, 0))
        return np.nan_to_num(np.hstack(present), nan=0.0, posinf=0.0, neginf=0.0)

    def history_names(self, time: int) -> tuple[str, ...]:
        """Column names of :meth:`covariate_history`, in its column order.

        Kept beside that method rather than derived at the call site, because the
        correspondence is otherwise an accident of two independent loops: the matrix
        skips a zero-width block and the names of a zero-width block are the empty
        tuple, so flattening happens to drop the same ones.
        :func:`tests.unit.test_longitudinal_data.test_the_history_names_are_the_history_columns`
        pins it, since a rule handed a mislabelled frame reads the wrong covariate and
        still returns a perfectly valid-looking arm.
        """
        if not 1 <= time <= self.n_times:
            raise DataError(f"time {time} is outside 1..{self.n_times}")
        return (
            *self.baseline_names,
            *(name for block in self.time_varying_names[:time] for name in block),
        )

    def history_frame(self, time: int) -> Any:
        """:meth:`covariate_history` as a dataframe, in the backend the data came from.

        This is what a dynamic rule :math:`d_t(H_t)` is handed: ``[W, L_1, ..., L_t]``.
        The final outcome is not in it because reading it is not an intervention.  A repeated
        outcome's earlier nodes ``Y_s``, ``s < t``, are in it, because each is the first
        covariate of block ``s + 1``.  A ``Y_s`` copy that the covariate screen dropped, as
        a constant column, is absent.  The earlier *treatments*
        are not in it because under the regimen they are what the rule itself assigned --
        passing them would let a rule read the treatment of a unit that deviated, which is a
        different object from the one it is assigning.  The same reasoning, and the same omissions,
        as
        :func:`cleverly.interventions.base._covariate_frame` at one time point.
        """
        matrix = self.covariate_history(time)
        names = self.history_names(time)
        return self.frame_like({name: matrix[:, index] for index, name in enumerate(names)})

    def policy_names(self, time: int) -> tuple[str, ...]:
        """Column names of :meth:`policy_frame`: the history, then the earlier treatments.

        A mixed node contributes one column per component, under the component's name.
        """
        earlier: list[str] = []
        for node in range(1, min(time, self.n_decisions + 1)):
            if self.node_kind(node) == "mixed":
                earlier.extend(self.component_names(node))
            else:
                earlier.append(self.treatment_names[node - 1])
        return (*self.history_names(time), *earlier)

    def observed_labels(self, time: int) -> Any:
        """``(n,)`` object array of the label each unit received at ``time``, ``None`` if absent.

        A row with no treatment at the node, censored before it or past an absorbing
        event, holds ``None``.
        """
        if not 1 <= time <= self.n_times:
            raise DataError(f"time {time} is outside 1..{self.n_times}")
        if self.node_kind(time) == "mixed":
            # One tuple of natural values per unit: labels for the categorical components
            # and doses for the continuous ones.
            present_rows = self.node_present(time)
            parts = [
                component.labels(values)
                for component, values in zip(
                    self.node_components(time), self.node_values(time).T, strict=True
                )
            ]
            tuples = np.empty(self.n, dtype=object)
            for row in range(self.n):
                tuples[row] = tuple(part[row] for part in parts) if present_rows[row] else None
            return tuples
        codes = self.treatment[:, time - 1]
        present = ~np.isnan(codes)
        if self.is_continuous_node(time):
            doses = np.asarray(codes, dtype=object)
            doses[~present] = None
            return doses
        levels = np.empty(len(self.treatment_levels[time - 1]), dtype=object)
        for index, level in enumerate(self.treatment_levels[time - 1]):
            levels[index] = level
        labels = levels[np.nan_to_num(codes, nan=0.0).astype(np.int64)]
        labels[~present] = None
        return labels

    def policy_frame(self, time: int, extra: Mapping[str, Any] | None = None) -> Any:
        """:meth:`history_frame` plus the earlier treatments, for a policy node to read.

        This is what a known policy density :math:`q_t(\\cdot \\mid H_t)` is handed:
        ``[W, L_1, ..., L_t]`` followed by one column per earlier treatment
        :math:`A_1, \\ldots, A_{t-1}`, under its treatment column name and holding the
        analyst's labels.  It never holds :math:`A_t`, a censoring indicator, the outcome or
        a later column.

        It differs from a rule's frame on purpose.  Under a deterministic plan the earlier
        arms of a follower are what the plan assigned, so a rule has nothing to learn from
        them.  Under a policy the earlier arms vary among the units that remain on the plan,
        and they are the arms the policy drew, reweighted by the cumulative ratio.  So
        :math:`H_t` holds them, and a policy may read them.

        A row with no earlier treatment, censored or past an absorbing event, has its
        missing label filled with the node's first sorted level, as :meth:`covariate_history`
        fills a missing covariate with zero.  So code such as ``h["A1"].str.startswith(...)``
        cannot raise on a row nobody reads, and the density at such a row is overwritten
        with zero.

        An earlier mixed node contributes one column per component, under the component's
        name: labels for a categorical component and doses for a continuous one.  ``extra``
        appends further columns, which a :class:`~cleverly.interventions.JointPolicy` uses
        to hand a component's map the natural values of the current node's other
        components.
        """
        history = self.covariate_history(time)
        payload: dict[str, Any] = {
            name: history[:, index] for index, name in enumerate(self.history_names(time))
        }
        for node in range(1, min(time, self.n_decisions + 1)):
            if self.node_kind(node) == "mixed":
                values = np.nan_to_num(self.node_values(node), nan=0.0)
                for component, column in zip(self.node_components(node), values.T, strict=True):
                    payload[component.name] = component.labels(column)
                continue
            if self.is_continuous_node(node):
                payload[self.treatment_names[node - 1]] = np.nan_to_num(
                    self.treatment[:, node - 1], nan=0.0
                )
                continue
            codes = np.nan_to_num(self.treatment[:, node - 1], nan=0.0).astype(np.int64)
            levels = np.empty(len(self.treatment_levels[node - 1]), dtype=object)
            for index, level in enumerate(self.treatment_levels[node - 1]):
                levels[index] = level
            payload[self.treatment_names[node - 1]] = levels[codes]
        if extra:
            clashing = sorted(set(extra) & set(payload))
            if clashing:
                raise DataError(
                    f"the component column(s) {clashing} of node {time} share a name with a "
                    "column of the policy frame; rename the component"
                )
            payload.update(extra)
        return self.frame_like(payload)

    def baseline_frame(self) -> Any:
        """``[W]`` and nothing else, in the backend the data came from.

        What a working model's design is handed
        (:mod:`cleverly.longitudinal.msm`).  :math:`V` is a subset of the *baseline*
        covariates by the estimand's own statement: :math:`m(\\bar a, V; \\beta)`
        summarises :math:`E[Y^{\\bar a} \\mid V]`, and a design reading :math:`L_t` would
        not be a model for that -- it would be conditioning on a consequence of
        :math:`\\bar a_1`, which is a different parameter with a different
        identification.  Enforced by what the frame contains rather than by a check,
        exactly as :meth:`history_frame` enforces what a rule may read.

        The columns are :attr:`baseline_names` and not the ``baseline=`` list the caller
        passed: validation may have dropped a column, and a design written against the
        names a fit *reports* is the one that keeps working.
        """
        return self.frame_like(
            {name: self.baseline[:, index] for index, name in enumerate(self.baseline_names)}
        )

    def history_design(
        self,
        time: int,
        *,
        treatment: Assignment | None = None,
        include_current: bool = False,
    ) -> FloatArray:
        """The conditioning set of a mechanism model at ``time``.

        ``[W, L_1, ..., L_t]`` plus a block per earlier treatment, and -- with
        ``include_current``, for the censoring model, which sits *after* the treatment
        decision -- a block for the current one.  ``treatment=None`` uses what each
        unit actually received, which is how the model is fitted; an assignment sets the
        arms a regimen would have assigned, which is where it is evaluated.  A dynamic
        rule assigns a different arm to different units, so that block is per row.

        Each node's block is :func:`~cleverly.data.validate.arm_indicators` of that
        node's codes against *that node's own* level count, which nodes do not have to
        share.  So a two-level node contributes the single 0/1 column it always did --
        a wholly binary panel's design is unchanged, bit for bit -- and a ``K``-level
        node contributes ``K - 1`` drop-first indicators.  One ordinal column would
        force :math:`g_t(\\cdot \\mid H_t, A_{t-1})` to move monotonically in the earlier
        arm for any learner linear in its design, which is a restriction on the
        mechanism nobody asked for; the shared helper's docstring carries the argument
        and ``docs/architecture-invariants.md`` carries the rule.  Note the exact-law
        tests cannot see this choice -- a saturated learner partitions by distinct design
        row, and ordinal codes and indicator tuples are a bijection -- so the witness for
        it is a ``glm`` mechanism on a law whose truth is non-monotone in the earlier
        arm, in ``tests/unit/test_sequential_design.py``.

        A row whose arm is missing -- censored, or past an absorbing event -- fills with
        code zero and so with the all-zero reference block.  Such a row is masked out of
        every fit and every influence curve; the fill exists to keep ``nan`` out of a
        design matrix a learner is called on.

        The *outcome* sequence uses :meth:`covariate_history` instead, with no treatment
        columns at all, and that holds under a rule as well as under a constant plan.
        The reason is not that a follower's past treatment is a constant -- under a rule
        it is not -- but that among the followers :math:`A_s = d_s(W, L_1, \\ldots, L_s)`
        is a *deterministic function of columns this design already carries*, since that
        is precisely the frame :meth:`history_frame` handed the rule.  Adding it would
        buy no information, only a redundant column for a penalised learner to spread a
        coefficient across.

        **No fit in this repository would come out differently if that were changed**,
        which is why the reasoning is written here rather than left to a fixture.  Two
        independent things hide it.  On the exact law the learner is saturated, and it
        partitions by distinct design row: a column that is a function of the others
        leaves the partition, and so every prediction, untouched.  Under ``glm`` the
        natural comparison is a rule that ignores the history against the constant plan it
        equals -- but such a rule assigns a *constant*, and a constant column is
        standardised to zeros and then ignored, on both sides of the comparison.  A
        genuinely dynamic rule would see a difference and has no second answer to be
        checked against.  Change this only with an argument.

        What ``tests/unit/test_sequential_design.py`` adds is the *call site*, not the
        argument: it pins that ``fit_regimen`` is handed
        :meth:`covariate_history` bit for bit, so the decision cannot be reversed by an
        edit that reads as a tidy-up.  It says nothing about which design is right, and a
        test claiming to say that should be mutated and seen to fail before it is trusted.

        Under :attr:`markov_order` ``k`` the design holds ``W``, ``L_{t-k}, ..., L_t`` and the
        blocks of ``A_{t-k-1}, ..., A_{t-1}``, plus ``A_t`` with ``include_current``.
        """
        columns = [self.window_covariates(time)]
        last = time if include_current else time - 1
        plan = None if treatment is None else assignment_matrix(treatment, self.n, self.n_times)
        # Under a Markov order k the window holds A_{t-k-1}, ..., A_{t-1}: the arms that
        # precede the window's covariates L_{t-k}, ..., L_t.
        first = 1 if self.markov_order is None else max(1, self.window_start(time) - 1)
        for t in range(first, last + 1):
            if self.is_held_node(t):
                # One block per *decision*: an identity node repeats node 1, so it adds
                # nothing the node-1 block does not already hold.
                continue
            if self.node_kind(t) == "mixed":
                # Every plan holds a joint policy at a mixed node, whose plan column is the
                # observed treatment, so the block is the observed components: each dose as
                # itself and each categorical component as its own indicators.
                columns.append(self.component_design(t))
                continue
            if plan is None:
                codes = np.nan_to_num(self.treatment[:, t - 1], nan=0.0)
            else:
                codes = np.nan_to_num(plan[:, t - 1], nan=0.0)
            if self.is_continuous_node(t):
                # A dose enters as itself, one column, as a point treatment's dose does, and
                # beside it the indicator of each declared atom.
                columns.append(self.dose_block(t, codes))
                continue
            columns.append(arm_indicators(codes, len(self.treatment_levels[t - 1])))
        return np.hstack(columns)

    def encode_assignment(self, assignment: Any, label: str) -> FloatArray:
        """Encode one regimen's raw labels against each treatment node's level set.

        A rule is evaluated before this method is called.  Only rows that still have a
        treatment decision at the node are validated; assignments after censoring or an
        absorbing event are outside the regimen's observed-data support and are filled.

        One comparison per *level* rather than one lookup per *row*: a node has at most
        :data:`~cleverly.data.validate.MAX_TREATMENT_LEVELS` of them and the data has
        ``n``, so this is the difference between a fixed number of vectorised passes and
        a Python loop that runs once per unit per node per regimen.  Equality is still
        ``==`` on the caller's own objects, which is what makes an ``int`` assignment find
        a ``float`` level and a ``numpy.str_`` find a ``str``.
        """
        raw = np.asarray(assignment, dtype=object)
        width = self.n_decisions
        if raw.ndim == 1:
            if raw.shape[0] != width:
                raise DataError(
                    f"regimen {label!r} assigns {raw.shape[0]} arm(s) but the data has "
                    f"{width} treatment decision(s)"
                )
            raw = np.broadcast_to(raw, (self.n, width))
        if raw.shape != (self.n, width):
            raise DataError(
                f"regimen {label!r} produced assignments with shape {raw.shape}; expected "
                f"({self.n}, {width})"
            )

        encoded = np.zeros((self.n, self.n_times), dtype=float)
        for time in range(self.n_decisions + 1, self.n_times + 1):
            # An identity node's plan value is the observed decision, on the rows still in
            # the study, so a unit follows the plan there exactly when it followed node 1.
            reachable = self.uncensored_through(time - 1) & self.event_free_through(time - 1)
            encoded[reachable, time - 1] = self.treatment[reachable, time - 1]
        for time, levels in enumerate(self.treatment_levels[: self.n_decisions], start=1):
            reachable = self.uncensored_through(time - 1) & self.event_free_through(time - 1)
            column = raw[:, time - 1]
            if self.node_kind(time) == "mixed":
                # A mixed node is assigned by a joint policy, which reads the observed
                # components, so the plan's column is the node's presence slot.
                # ``refuse_continuous_assignments`` refuses every other plan node there.
                encoded[reachable, time - 1] = self.treatment[reachable, time - 1]
                continue
            if self.is_continuous_node(time):
                # A continuous node is assigned by a policy, which reads the observed dose,
                # so the plan's column is that dose.
                encoded[reachable, time - 1] = np.asarray(column[reachable], dtype=float)
                continue
            matched = np.zeros(self.n, dtype=bool)
            for code, level in enumerate(levels):
                # Elementwise ``==`` on the object array: numpy runs the comparison loop
                # in C and calls each object's own ``__eq__``, so the level-vs-row
                # semantics are the dict's and only the loop overhead is gone.
                if isinstance(level, tuple):
                    # A vector node's level is a tuple, which numpy would broadcast as a
                    # sequence; compare it to each row as one label instead.
                    here = np.fromiter((value == level for value in column), bool, self.n)
                else:
                    here = np.asarray(column == level, dtype=bool)
                if here.shape != (self.n,):
                    # Every level comes from ``np.unique`` of one column and so is a
                    # scalar.  One that was not would *broadcast* rather than compare
                    # elementwise, and a wrongly shaped mask marks the wrong rows
                    # silently -- checked rather than assumed, as everywhere here.
                    raise DataError(
                        f"treatment column {self.decision_name(time)!r} has a "
                        f"non-scalar level {level!r}; a treatment arm must be a single "
                        "label per unit"
                    )
                encoded[here & reachable, time - 1] = float(code)
                matched |= here
            unknown = np.flatnonzero(reachable & ~matched)
            if unknown.size:
                value = column[unknown[0]]
                raise DataError(
                    f"regimen {label!r} assigns {value!r} at time {time}; treatment "
                    f"column {self.decision_name(time)!r} has levels "
                    f"{list(levels)!r}"
                )
        return encoded

    # ------------------------------------------------------------------ output

    def frame_like(self, payload: dict[str, Any]) -> Any:
        """Build a dataframe of ``payload`` in the backend this data came from."""
        return frame_from_dict(payload, backend=self.backend)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        parts = [
            f"n={self.n}",
            f"T={self.n_times}",
            f"family={self.family!r}",
            f"baseline={self.baseline.shape[1]}",
        ]
        widths = [block.shape[1] for block in self.time_varying]
        parts.append(f"time_varying={widths}")
        if self.event is not None:
            parts.append(f"events={int(self.event[:, -1].sum())}")
        if self.censoring_names:
            share = float(self.uncensored_through(self.n_times).mean())
            parts.append(f"uncensored={share:.3f}")
        if self.treatment_held:
            parts.append(f"held={self.treatment_names[0]!r}")
        if self.cluster is not None:
            parts.append(f"clusters={self.n_clusters}")
        if self.is_weighted:
            parts.append(f"weighted=yes (effective n={self.effective_n:.0f})")
        return f"LongitudinalData({', '.join(parts)})"


#: What to tell a caller whose node has more levels than the estimator accepts.  Not
#: :data:`~cleverly.data.validate.CONTINUOUS_REMEDY`, which names ``treatment_kind=``: that
#: is a :class:`~cleverly.data.CausalData` keyword and ``LTMLE`` does not take it, so the
#: default suggestion would send the reader to an argument that does not exist here.
_TOO_MANY_LEVELS_REMEDY = (
    "Collapse the levels into the arms you actually want to report, or declare the node a "
    "continuous dose with continuous_treatment=[...] and fit a modified treatment policy at "
    "it, such as Shift(0.5, cap=...)."
)


def _event_node(grid: FloatArray, times: FloatArray) -> IntArray:
    """The node ``k`` of an event at ``T``: the one with ``g_{k-1} < T <= g_k``."""
    return np.asarray(np.searchsorted(grid, times, side="left") + 1, dtype=np.int64)


def _censor_node(grid: FloatArray, times: FloatArray) -> IntArray:
    """The node at which a unit censored at the grid time ``g_j`` leaves: ``j + 1``.

    The unit is observed through node ``j``, so an event at ``g_j`` in another unit is
    ordered first and this unit was at risk for it.
    """
    return np.asarray(np.searchsorted(grid, times, side="left") + 2, dtype=np.int64)


def _time_grid(
    times: FloatArray, codes: FloatArray, name: str, grid: Sequence[float] | None
) -> tuple[float, ...]:
    """Validate the time column and return the grid, declared or default."""
    bad = ~np.isfinite(times) | ~(times > 0.0)
    if np.any(bad):
        example = float(times[np.flatnonzero(bad)[0]])
        raise DataError(
            f"time column {name!r} has {int(bad.sum())} value(s) that are missing, not finite, "
            f"or at or below zero, for example {example!r}. A time is measured from the "
            "baseline treatment, so every time must be a positive number"
        )
    if grid is None:
        fractional = times != np.floor(times)
        if np.any(fractional):
            example = float(times[np.flatnonzero(fractional)[0]])
            raise DataError(
                f"time column {name!r} holds non-integer times, for example {example!r}, so "
                "there is no default grid. Declare grid=[g1, g2, ...] at the visit times. A "
                "grid read off the observed times would change with the sample, and the curve "
                "would have no fixed horizons"
            )
        events = times[np.isfinite(codes) & (codes > 0.0)]
        if events.size == 0:
            raise DataError(
                f"time column {name!r} has no event, so there is no default grid; the "
                "default grid runs from 1 to the largest event time"
            )
        return tuple(float(g) for g in range(1, int(events.max()) + 1))
    try:
        values = np.asarray(list(grid), dtype=float)
    except (TypeError, ValueError):
        values = np.asarray([np.nan])
    if (
        values.ndim != 1
        or values.size == 0
        or not np.all(np.isfinite(values))
        or not np.all(values > 0.0)
        or not np.all(np.diff(values) > 0.0)
    ):
        raise ValueError(
            f"grid must be a strictly increasing sequence of positive finite times; got {grid!r}"
        )
    return tuple(float(g) for g in values)


def _event_causes(
    codes: FloatArray, name: str, causes: Mapping[int, str] | None
) -> tuple[tuple[str, ...], tuple[int, ...], bool]:
    """Validate the event codes; return the cause labels, their codes, and whether competing."""
    absent = ~np.isfinite(codes)
    if np.any(absent):
        raise DataError(f"event column {name!r} has {int(absent.sum())} missing value(s)")
    invalid = (codes < 0.0) | (codes != np.floor(codes))
    if np.any(invalid):
        shown = sorted({float(code) for code in codes[invalid]})[:6]
        raise DataError(
            f"event column {name!r} holds code(s) {shown} that are negative or not integers. "
            "A code is 0 for censored and a positive integer for each absorbing cause"
        )
    nonzero = sorted({int(code) for code in codes if code != 0.0})
    if causes is None:
        if set(nonzero) <= {1}:
            return ("1",), (1,), False
        return tuple(str(code) for code in nonzero), tuple(nonzero), True
    declared = {int(code): str(label) for code, label in causes.items()}
    unnamed = [code for code in nonzero if code not in declared]
    if unnamed:
        raise DataError(
            f"event column {name!r} holds code(s) {unnamed} that causes= does not name. Name "
            "every nonzero code, or recode a code as 0 if it means censored"
        )
    if 0 in declared:
        raise DataError("causes= names code 0, which is the censoring code, not a cause")
    order = sorted(declared)
    return tuple(declared[code] for code in order), tuple(order), True


def _held_node_count(
    name: str,
    outcome: Any,
    censoring: Sequence[str] | None,
    time_varying: Sequence[Sequence[str]] | None,
) -> int:
    """The node count of a held design, read from the per-node arguments.

    A survival or repeated outcome gives one node per outcome column.  An end-of-study
    outcome takes the count from ``censoring=``, else from ``time_varying=``, else it is
    one.  Every per-node argument that is given must agree, because the decision is held over the
    same nodes.
    """
    counts: list[tuple[str, int]] = []
    if isinstance(outcome, Mapping):
        blocks = [list(block) for block in outcome.values()]
        if blocks:
            counts.append(("outcome", len(blocks[0])))
    elif not isinstance(outcome, str):
        counts.append(("outcome", len(list(outcome))))
    if censoring is not None:
        counts.append(("censoring=", len(list(censoring))))
    if time_varying is not None:
        counts.append(("time_varying=", len(list(time_varying))))
    if len({count for _, count in counts}) > 1:
        given = ", ".join(f"{role} gives {count}" for role, count in counts)
        raise DataError(
            f"treatment={name!r} is one baseline decision held over every node, and the nodes "
            f"are counted from the other columns: {given}. Give every per-node argument the "
            "same length"
        )
    return counts[0][1] if counts else 1


def _encode_node_treatment(values: Any, name: str) -> tuple[FloatArray, tuple[object, ...]]:
    """Encode a node while preserving the historical one-observed-binary-arm case.

    A binary treatment's support is declared by its 0/1 convention even when this
    particular sample contains only one arm.  That case reaches
    :func:`~cleverly.longitudinal.sequential.prepare_node`'s "no unit followed regimen"
    refusal, which names the regimen and the node the sample cannot answer for; keeping
    the ``(0, 1)`` support here is what lets it get that far rather than being refused as
    a one-level column.  An arbitrary one-label categorical column has no analogous way to
    infer which unobserved labels belong to its support, so the shared categorical encoder
    correctly refuses it.

    A numeric node with *many* distinct values warns rather than being refused, on the
    reasoning :data:`~cleverly.data.validate.MIN_CONTINUOUS_LEVELS` already records for a
    point treatment: a coarse ordered dose is a perfectly well-defined set of arms, and
    which reading is wanted is the analyst's to declare.  What is worth saying is that
    ``LTMLE`` will treat them as *unordered* labels, since that is the reading a reader who
    typed a dose column is least likely to expect.
    """
    first = np.asarray(values, dtype=object).reshape(-1)
    if first.size and isinstance(first[0], tuple):
        return _encode_vector_node(first, name)
    raw = np.asarray(values).reshape(-1)
    unique = np.unique(raw)
    if unique.size == 1:
        try:
            numeric = float(unique[0])
        except (TypeError, ValueError):
            pass
        else:
            if numeric in (0.0, 1.0):
                return np.asarray(raw, dtype=float), (0, 1)
    # Recognised from the *values* rather than from ``raw.dtype``: the container reads
    # treatment columns with ``dtype=object`` so that a string label survives to here, so
    # the array's own kind is ``O`` however numeric its contents are.
    numeric_levels = unique.dtype.kind in "fiu" or all(
        isinstance(value, (int, float, np.number)) and not isinstance(value, bool)
        for value in unique
    )
    if numeric_levels and unique.size >= MIN_CONTINUOUS_LEVELS:
        warnings.warn(
            f"treatment node {name!r} is numeric and takes {unique.size} distinct values; "
            "it will be modelled as that many unordered arms, with one multinomial "
            "mechanism per node and one counterfactual mean per assigned sequence. If it "
            "is a dose whose spacing matters, declare it with continuous_treatment=[...] "
            "and fit a modified treatment policy at it.",
            DataWarning,
            stacklevel=4,
        )
    return encode_treatment(raw, name, remedy=_TOO_MANY_LEVELS_REMEDY)


def _encode_vector_node(values: Any, name: str) -> tuple[FloatArray, tuple[object, ...]]:
    """Encode a vector node: one level per observed tuple of its components, in sorted order.

    A discrete vector is one categorical node over the product of its components' observed
    values (a finite partition), so the level limit of a categorical node applies to the
    tuples.  :func:`~cleverly.data.validate.encode_vector_treatment` encodes a point vector
    treatment the same way.
    """
    return encode_vector_treatment(
        values, name, what="vector treatment node", remedy=_TOO_MANY_LEVELS_REMEDY
    )


def _node_kind_of(block: Sequence[str], continuous: Sequence[str]) -> NodeKind:
    """The kind of a node from its columns and the declared continuous columns.

    Every component is read: a vector node whose first component is categorical and whose
    second is a dose is mixed, not categorical.
    """
    declared = [name in continuous for name in block]
    if not any(declared):
        return "categorical"
    return "continuous" if len(block) == 1 else "mixed"


def _encode_mixed_node(
    values: Any, names: Sequence[str], continuous: Sequence[bool]
) -> tuple[FloatArray, tuple[tuple[object, ...], ...]]:
    """Encode a mixed node's tuples component by component, on the rows that hold it.

    A continuous component keeps its dose.  A categorical component is encoded on its own,
    as a node of one column is, so its codes index its own labels and not the joint tuples.
    """
    tuples = list(np.asarray(values, dtype=object).reshape(-1))
    columns = []
    level_sets: list[tuple[object, ...]] = []
    for index, (name, dose) in enumerate(zip(names, continuous, strict=True)):
        raw = np.empty(len(tuples), dtype=object)
        for row, value in enumerate(tuples):
            raw[row] = value[index]
        if dose:
            columns.append(_continuous_dose(raw, str(name)))
            level_sets.append(())
            continue
        codes, levels = _encode_node_treatment(raw, str(name))
        columns.append(np.asarray(codes, dtype=float))
        level_sets.append(tuple(levels))
    return np.column_stack(columns), tuple(level_sets)


def _mixed_reader(data: LongitudinalData, time: int, reader: str) -> Refusal:
    """The refusal of a one-column reader at a mixed node."""
    return Refusal(
        RefusalReason.INVALID_DECLARATION,
        f"node {data.treatment_names[time - 1]!r} at time {time} is a vector with a "
        f"continuous component, and {reader} reads one column. This path does not support a "
        "mixed node; read node_kind() and node_values().",
    )


def _node_column(frame: Any, block: Sequence[str]) -> Any:
    """One node's raw values: the column itself, or the tuple of a vector node's columns."""
    if len(block) == 1:
        return column_array(frame, block[0], dtype=object)
    parts = [column_array(frame, name, dtype=object) for name in block]
    out = np.empty(len(parts[0]), dtype=object)
    for row in range(out.size):
        values = tuple(part[row] for part in parts)
        out[row] = None if any(not _categorical_value_is_present(v) for v in values) else values
    return out


def _continuous_dose(values: Any, name: str) -> FloatArray:
    """A continuous node's dose on the rows that hold one, refusing a non-number."""
    try:
        dose = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise DataError(
            f"treatment node {name!r} is declared continuous but holds a value that is not a number"
        ) from error
    if not np.all(np.isfinite(dose)):
        raise DataError(
            f"treatment node {name!r} is declared continuous but holds a non-finite dose"
        )
    return dose


def _prefix_all(indicator: BoolArray) -> BoolArray:
    """``(n, T+1)`` cumulative AND: column ``t`` is "held at every node up to ``t``".

    Column 0 is all-true -- no node has been passed yet -- which is what makes the scan
    the same statement the per-node rebuild made, rather than an off-by-one of it.
    """
    n, times = indicator.shape
    out = np.ones((n, times + 1), dtype=bool)
    np.logical_and.accumulate(indicator, axis=1, out=out[:, 1:])
    return out


@dataclass(frozen=True)
class RegimenMasks:
    """One regimen's node masks, as prefix scans rather than as repeated rebuilds.

    Three ``(n, T+1)`` boolean matrices, each holding "this held at every node up to and
    including ``t``" with ``t = 0`` meaning everybody.  :meth:`at_risk` and
    :meth:`following` read a column each; they answer exactly what
    :meth:`LongitudinalData.at_risk` and :meth:`LongitudinalData.following` answer, and
    ``tests/unit/test_longitudinal_masks.py`` checks that node by node rather than
    trusting the derivation.

    The memory is ``3 n (T + 1)`` bytes -- 78 MB at ``n = 10^6, T = 25``, against the
    ``(n, T)`` float64 arrays the container already holds at eight times that.
    """

    uncensored: BoolArray
    followed: BoolArray
    event_free: BoolArray

    def at_risk(self, time: int) -> BoolArray:
        """:math:`H_t` observed, regimen-consistent and event-free: reads ``time - 1``."""
        index = max(0, time - 1)
        return np.asarray(
            self.uncensored[:, index] & self.followed[:, index] & self.event_free[:, index]
        )

    def following(self, time: int) -> BoolArray:
        """Still on the regimen and observed *after* ``time`` -- and event-free *before*.

        Note the two indices: ``time`` for the censoring and follow factors, ``time - 1``
        for the event.  A unit that had the event at ``time`` **is** in this node's
        regression -- it is the observation that the event happened -- and
        ``at_risk(t + 1) == following(t) & event-free at t`` is the closure identity that
        generalises rather than breaks.  Tidying the asymmetry away is the single easiest
        mistake to make here.
        """
        return np.asarray(
            self.uncensored[:, time] & self.followed[:, time] & self.event_free[:, max(0, time - 1)]
        )


def _through(uncensored: BoolArray, time: int) -> BoolArray:
    """Rows uncensored at every node up to and including ``time``."""
    if time <= 0:
        return np.ones(uncensored.shape[0], dtype=bool)
    return np.asarray(uncensored[:, :time].all(axis=1), dtype=bool)


def _event_free(failed: BoolArray | None, time: int) -> BoolArray | bool:
    """Rows that had not had the event at any node up to and including ``time``.

    Returns ``True`` -- the identity for ``&`` -- rather than an all-true array when the
    fit declares no event nodes, so that a mask written ``a & b & _event_free(...)`` is
    the same expression it was before there were event nodes at all.  One code path with
    a degenerate factor, on the same argument as :func:`assignment_matrix`'s broadcast.
    """
    if failed is None:
        return True
    if time <= 0:
        return np.ones(failed.shape[0], dtype=bool)
    return np.asarray(~failed[:, time - 1], dtype=bool)


def _read_followup(
    censoring: FloatArray | None,
    censoring_names: Sequence[str],
    event: FloatArray | None,
    event_names: Sequence[Sequence[str]],
    cause_labels: Sequence[str],
    n: int,
    n_times: int,
) -> tuple[BoolArray, BoolArray | None, BoolArray | None]:
    """Validate the censoring and event columns, returning both cumulative indicators.

    One left-to-right sweep rather than two, because the two processes interleave and
    each decides what the other is required to hold.  At time ``t`` the ordering is
    ``A_t, C_t, Y_t``: ``C_t`` is asked of the units still in the study *before* that
    node, which means uncensored through ``t - 1`` **and** event-free through ``t - 1``;
    ``Y_t`` is asked of the units still uncensored *after* it, so through ``t``, and
    event-free through ``t - 1``.  Reading the censoring columns first and the event
    columns after would demand ``C_2`` of a unit that had the event at time 1.

    With no event columns declared, ``failed`` is ``None`` and every line below reduces
    to the censoring-only sweep this replaced, so a fit with one end-of-study outcome is
    validated by exactly the arithmetic it always was.

    The event array carries a **cause** axis, of length one for a single absorbing event.
    Leaving the risk set is leaving it whichever cause did it, so ``happened`` -- the thing
    every other node's presence rule is asked against -- is all-cause, and a second cause
    adds two refusals rather than a second sweep: two causes may not fire at one node, and
    a unit that has left may not later be marked as having a *different* cause's event.
    """
    if censoring is not None:
        censor_values = np.asarray(censoring, dtype=float)
        if censor_values.shape != (n, n_times):
            raise DataError(f"censoring has shape {censor_values.shape}, expected {(n, n_times)}")
    n_causes = 1
    if event is not None:
        event_values = np.asarray(event, dtype=float)
        n_causes = event_values.shape[2] if event_values.ndim == 3 else 1
        event_values = event_values.reshape(n, n_times, n_causes)
        if event_values.shape[:2] != (n, n_times):
            raise DataError(f"outcome has shape {event_values.shape[:2]}, expected {(n, n_times)}")

    def _column_name(cause: int, time: int) -> str:
        block = event_names[cause] if cause < len(event_names) else ()
        return str(block[time]) if time < len(block) else f"Y{time + 1}"

    def _cause_name(cause: int) -> str:
        return str(cause_labels[cause]) if cause < len(cause_labels) else "the event"

    uncensored = np.ones((n, n_times), dtype=bool)
    failed = None if event is None else np.zeros((n, n_times), dtype=bool)
    by_cause = None if event is None else np.zeros((n, n_times, n_causes), dtype=bool)
    alive = np.ones(n, dtype=bool)
    happened = np.zeros(n, dtype=bool)
    happened_by_cause = np.zeros((n, n_causes), dtype=bool)
    for time in range(n_times):
        at_risk = alive & ~happened
        if censoring is None:
            uncensored[:, time] = alive
        else:
            column = censor_values[:, time]
            name = str(censoring_names[time]) if time < len(censoring_names) else f"C{time + 1}"
            present = np.isfinite(column)
            bad = at_risk & ~present
            if np.any(bad):
                raise DataError(
                    f"censoring column {name!r} is missing for {int(bad.sum())} unit(s) that "
                    "were still under observation; a unit either remained (1) or left (0)"
                )
            seen = np.unique(column[at_risk])
            if not np.all(np.isin(seen, (0.0, 1.0))):
                raise DataError(
                    f"censoring column {name!r} takes values {seen[:6].tolist()}; it must be "
                    "1 where the unit is still under observation after that time point and 0 "
                    "where it left"
                )
            # A unit that has already left may carry 0 or nothing at all; anything else
            # would say it came back, which is a different observation process.
            returned = ~alive & present & (column == 1.0)
            if np.any(returned):
                raise DataError(
                    f"censoring column {name!r} marks {int(returned.sum())} unit(s) as under "
                    "observation after they had already been censored. Censoring is assumed "
                    "monotone (a unit that leaves does not return) and intermittent "
                    "observation identifies a different parameter, so this is refused rather "
                    "than read as a return to follow-up."
                )
            uncensored[:, time] = at_risk & (column == 1.0)
        alive = uncensored[:, time]

        if failed is None or by_cause is None:
            continue
        # Asked of the units still under observation *after* this node's censoring and
        # not already out of the risk set: those are the ones for whom "did the event
        # happen at time t" is a question the data answers.
        required = alive & ~happened
        fired = np.zeros((n, n_causes), dtype=bool)
        for cause in range(n_causes):
            column = event_values[:, time, cause]
            name = _column_name(cause, time)
            present = np.isfinite(column)
            # The order at a node is ``A_t, C_t, Y_t``, so ``Y_t`` is unobserved once
            # ``C_t = 0``.  A value there is data recorded after the unit left, and reading
            # past it would drop an event that was meant as "had it and left".
            stray = ~alive & ~happened & present
            if np.any(stray):
                raise DataError(
                    f"outcome column {name!r} holds a value for {int(stray.sum())} unit(s) "
                    f"that were censored at or before time {time + 1}. The order at a node is "
                    "A_t, C_t, Y_t, so the event is unobserved once the unit is censored. "
                    "Record an event at a node as censoring 1 and event 1, and a loss after "
                    "an event-free interval as censoring 0 with the event missing"
                )
            bad = required & ~present
            if np.any(bad):
                raise DataError(
                    f"outcome column {name!r} is missing for {int(bad.sum())} unit(s) that were "
                    "still at risk; a unit at risk either had the event (1) or did not (0). A "
                    "unit whose outcome is missing for some other reason is censored, and its "
                    "censoring column is where that belongs"
                )
            seen = np.unique(column[required])
            if not np.all(np.isin(seen, (0.0, 1.0))):
                raise DataError(
                    f"outcome column {name!r} takes values {seen[:6].tolist()}; a survival "
                    "outcome is an event indicator at every node, 1 where the event happened "
                    "at that time point and 0 where it had not yet"
                )
            # Exactly the licence the censoring sweep gives above, and for the same
            # reason: a unit that has already had the event may carry the 1 forward or
            # carry nothing, since either says the same absorbing thing.  A 0 does not.
            recovered = happened_by_cause[:, cause] & present & (column == 0.0)
            if np.any(recovered):
                raise DataError(
                    f"outcome column {name!r} marks {int(recovered.sum())} unit(s) as event-free "
                    "after they had already had the event. The event is assumed absorbing (a "
                    "unit that has it leaves the risk set and does not return to it) and a "
                    "recurrent event identifies a different parameter, so this is refused "
                    "rather than read as a recovery."
                )
            # A unit that has left through one cause cannot later be marked as having had
            # another: it is out of the risk set, and a second absorbing event is not an
            # observation this parameter has a place for.
            struck = happened & ~happened_by_cause[:, cause] & present & (column == 1.0)
            if np.any(struck):
                raise DataError(
                    f"outcome column {name!r} marks {int(struck.sum())} unit(s) as having had "
                    f"{_cause_name(cause)} after they had already left the risk set through "
                    "another cause. The causes are competing and each is absorbing, so a unit "
                    "has at most one of them"
                )
            fired[:, cause] = required & (column == 1.0)

        if n_causes > 1:
            clash = fired.sum(axis=1) > 1
            if np.any(clash):
                row = int(np.flatnonzero(clash)[0])
                both = [_cause_name(int(c)) for c in np.flatnonzero(fired[row])]
                raise DataError(
                    f"{int(clash.sum())} unit(s) have more than one cause marked at time "
                    f"{time + 1}; the first has {both}. Competing causes are mutually "
                    "exclusive: a unit leaves the risk set through exactly one of them, so "
                    "at most one indicator is 1 at any node"
                )

        for cause in range(n_causes):
            by_cause[:, time, cause] = happened_by_cause[:, cause] | fired[:, cause]
        happened_by_cause = by_cause[:, time, :]
        failed[:, time] = happened | fired.any(axis=1)
        happened = failed[:, time]
    return uncensored, failed, by_cause


def _check_presence(
    column: Any, required: BoolArray, name: str, *, absent: str | None = None
) -> None:
    """Every node must be recorded while the unit is under observation, and only then."""
    values = np.asarray(column)
    if values.dtype.kind in "biufc":
        finite = np.isfinite(np.asarray(values, dtype=float))
    else:
        finite = np.asarray([_categorical_value_is_present(value) for value in values], dtype=bool)
    missing = required & ~finite
    if np.any(missing):
        if absent is not None:
            raise DataError(f"{int(missing.sum())} {absent}")
        raise DataError(
            f"{name!r} is missing for {int(missing.sum())} unit(s) that were still in the "
            "study at that time. Every node before a unit leaves has to be recorded; a "
            "value missing for some other reason is a further node in the likelihood."
        )
    stray = ~required & finite
    if np.any(stray):
        raise DataError(
            f"{name!r} is recorded for {int(stray.sum())} unit(s) that had already left the "
            "study (censored, or, on a survival fit, having had the event). The estimator "
            "conditions on the observed history and would ignore those values, so they are "
            "refused rather than silently dropped: set the nodes after a unit leaves to "
            "missing."
        )


def _categorical_value_is_present(value: Any) -> bool:
    """Whether an object-valued categorical cell is a recorded finite label."""
    if value is None:
        return False
    value_type = type(value)
    if value_type.__name__ == "NAType" and value_type.__module__.startswith("pandas"):
        return False
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def _refuse_duplicates(names: Sequence[str]) -> None:
    """A column may hold exactly one node; the same name twice is an ordering error."""
    seen: set[str] = set()
    repeated: set[str] = set()
    for name in names:
        if name in seen:
            repeated.add(name)
        seen.add(name)
    if repeated:
        raise DataError(
            f"columns {sorted(repeated)} are used for more than one node. Each column is one node "
            "in the time ordering, and a column that is both (say) a baseline covariate "
            "and a time-varying one would be adjusted for twice."
        )


def _probability_column_names(
    treatment_probabilities: Any, censoring_probabilities: Any
) -> tuple[dict[str, tuple[tuple[Any, str], ...]], dict[str, str]]:
    """The column names of a column-form longitudinal declaration, by node.

    A treatment node whose per-node mapping holds strings names columns, and a censoring
    entry that is a string names a column.  Array entries are left to the parser.
    """
    treatment_columns: dict[str, tuple[tuple[Any, str], ...]] = {}
    if isinstance(treatment_probabilities, Mapping):
        for node, per_node in treatment_probabilities.items():
            pairs = probability_columns(per_node)
            if pairs is not None:
                treatment_columns[str(node)] = pairs
    censoring_columns: dict[str, str] = {}
    if isinstance(censoring_probabilities, Mapping):
        for node, column in censoring_probabilities.items():
            if isinstance(column, str):
                censoring_columns[str(node)] = column
    return treatment_columns, censoring_columns
