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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import numpy as np

from .._typing import BoolArray, FloatArray, IntArray
from ..data.validate import (
    MIN_CONTINUOUS_LEVELS,
    MIN_OBSERVATIONS,
    arm_indicators,
    check_covariates,
    encode_clusters,
    encode_treatment,
    resolve_family,
)
from ..data.weighting import (
    WeightReport,
    WeightSpec,
    _prepare_weights,
    describe_weights,
    effective_sample_size,
)
from ..exceptions import DataError, DataWarning
from ..utils.frames import (
    as_frame,
    backend_of,
    column_array,
    frame_from_dict,
    is_dataframe,
    matrix_from_columns,
)

__all__ = ["Assignment", "LongitudinalData", "RegimenMasks", "assignment_matrix"]

#: What arm a regimen assigns at each node: one arm per node for a static plan, or an
#: ``(n, T)`` matrix when a dynamic rule assigns a different arm to different units.
#: Every mask and design here reads it through :func:`assignment_matrix`, so the static
#: case is the broadcast of the dynamic one rather than a second code path.
Assignment: TypeAlias = Sequence[float] | FloatArray


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
                f"a per-unit assignment must be ({n}, {n_times}) -- a row per unit and a "
                f"column per treatment node -- but got {values.shape}"
            )
        return values
    raise DataError(
        f"an assignment must be one arm per node or an (n, T) matrix; got {values.ndim} "
        "dimension(s)"
    )


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

    # ------------------------------------------------------------------ build

    @classmethod
    def from_frame(
        cls,
        data: Any,
        *,
        outcome: str | Sequence[str] | Mapping[str, Sequence[str]],
        treatment: Sequence[str],
        baseline: Sequence[str],
        time_varying: Sequence[Sequence[str]] | None = None,
        censoring: Sequence[str] | None = None,
        id: str | None = None,
        weights: str | None = None,
        weights_type: str = "probability",
        weights_estimated: bool = False,
        family: str = "auto",
    ) -> LongitudinalData:
        """Build from one wide dataframe: a row per unit, a column per node.

        Parameters
        ----------
        outcome:
            One column name for an end-of-study outcome; **one per time point** for a
            survival outcome, in the same order as ``treatment``; or a **mapping of cause
            to one column per time point** for competing risks,
            ``{"relapse": ["R1", "R2"], "death": ["D1", "D2"]}``.  Which of the three is
            passed declares which parameter the fit answers, so it is read off the shape
            rather than from a further keyword: a number, a curve and a set of curves are
            not the same report, and a flag saying which would be a second place for them
            to disagree.  A mapping with one cause is still competing risks by
            declaration, and reports a cumulative incidence.
        treatment:
            Treatment column per time point, in time order.  Its length declares ``T``.
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
        """
        if not is_dataframe(data):
            raise DataError("LongitudinalData.from_frame expects a pandas or polars DataFrame")
        frame = as_frame(data)
        columns = list(frame.columns)

        treatment_names = [str(name) for name in treatment]
        if not treatment_names:
            raise DataError("treatment= is empty; a longitudinal fit needs at least one node")
        n_times = len(treatment_names)

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

        # Three shapes, and which one is passed declares which parameter the fit answers:
        # a name is a number, a sequence is a curve, a mapping is one curve per cause.
        # Read off the shape rather than from a further keyword, since a flag saying which
        # would be a second place for the declaration and the columns to disagree.
        cause_labels: tuple[str, ...] = ()
        if isinstance(outcome, str):
            event_blocks: list[list[str]] = []
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
        wanted = [*outcome_names, *treatment_names, *censor_names, *baseline_names]
        for block in blocks:
            wanted.extend(block)
        if id is not None:
            wanted.append(id)
        if weights is not None:
            wanted.append(str(weights))
        missing = [name for name in wanted if name not in columns]
        if missing:
            raise DataError(f"columns not found in the frame: {missing}; available: {columns}")
        _refuse_duplicates(wanted)

        return cls._build(
            outcome=None if survival else column_array(frame, outcome_names[0]),
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
            treatment=np.column_stack(
                [column_array(frame, name, dtype=object) for name in treatment_names]
            ),
            treatment_names=treatment_names,
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
        )

    @classmethod
    def _build(
        cls,
        *,
        outcome: Any,
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
                "baseline= is empty; a longitudinal fit adjusts for the baseline covariates "
                "at every node, and with none of them there is nothing to adjust for"
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
                f"treatment must be ({n}, {n_times}) -- rows by treatment nodes -- "
                f"but got {raw_treatment.shape}"
            )
        a = np.full((n, n_times), np.nan)
        level_sets: list[tuple[object, ...]] = []
        for time, name in enumerate(treatment_names, start=1):
            # Named apart from the ``column`` index and ``values`` matrix of the block
            # loop above: mypy unifies a name's type across the whole function body, so
            # reusing either here is an error rather than a shadow.
            arms = raw_treatment[:, time - 1]
            at_this_node = at_risk[:, time - 1]
            _check_presence(arms, at_this_node, str(name))
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

    def regimen_masks(self, assignment: Assignment) -> RegimenMasks:
        """Every node's masks for one regimen, scanned once.

        :meth:`at_risk` and :meth:`following` each rebuild a prefix from scratch --
        ``uncensored[:, :t].all(axis=1)`` and a loop of ``t`` comparisons -- so calling
        them at every node costs :math:`O(T^2 n)` per regimen, and a survival fit pays
        that once per horizon on top.  The masks are *prefix scans of a conjunction*, so
        carrying them down the nodes computes the same arrays in :math:`O(T n)`.

        The two methods stay, answer the same thing, and are what a caller wanting one node
        should use; this is what the recursion uses.
        """
        matches = self.treatment == assignment_matrix(assignment, self.n, self.n_times)
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

        This is what a dynamic rule :math:`d_t(H_t)` is handed: ``[W, L_1, ..., L_t]``
        and nothing else.  The outcome is not in it because reading it is not an
        intervention, and the earlier *treatments* are not in it because under the
        regimen they are what the rule itself assigned -- passing them would let a rule
        read the treatment of a unit that deviated, which is a different object from the
        one it is assigning.  The same reasoning, and the same omissions, as
        :func:`cleverly.interventions.base._covariate_frame` at one time point.
        """
        matrix = self.covariate_history(time)
        names = self.history_names(time)
        return self.frame_like({name: matrix[:, index] for index, name in enumerate(names)})

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
        """
        columns = [self.covariate_history(time)]
        last = time if include_current else time - 1
        plan = None if treatment is None else assignment_matrix(treatment, self.n, self.n_times)
        for t in range(1, last + 1):
            if plan is None:
                codes = np.nan_to_num(self.treatment[:, t - 1], nan=0.0)
            else:
                codes = plan[:, t - 1]
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
        if raw.ndim == 1:
            if raw.shape[0] != self.n_times:
                raise DataError(
                    f"regimen {label!r} assigns {raw.shape[0]} arm(s) but the data has "
                    f"{self.n_times} treatment node(s)"
                )
            raw = np.broadcast_to(raw, (self.n, self.n_times))
        if raw.shape != (self.n, self.n_times):
            raise DataError(
                f"regimen {label!r} produced assignments with shape {raw.shape}; expected "
                f"({self.n}, {self.n_times})"
            )

        encoded = np.zeros((self.n, self.n_times), dtype=float)
        for time, levels in enumerate(self.treatment_levels, start=1):
            reachable = self.uncensored_through(time - 1) & self.event_free_through(time - 1)
            column = raw[:, time - 1]
            matched = np.zeros(self.n, dtype=bool)
            for code, level in enumerate(levels):
                # Elementwise ``==`` on the object array: numpy runs the comparison loop
                # in C and calls each object's own ``__eq__``, so the level-vs-row
                # semantics are the dict's and only the loop overhead is gone.
                here = np.asarray(column == level, dtype=bool)
                if here.shape != (self.n,):
                    # Every level comes from ``np.unique`` of one column and so is a
                    # scalar.  One that was not would *broadcast* rather than compare
                    # elementwise, and a wrongly shaped mask marks the wrong rows
                    # silently -- checked rather than assumed, as everywhere here.
                    raise DataError(
                        f"treatment column {self.treatment_names[time - 1]!r} has a "
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
                    f"column {self.treatment_names[time - 1]!r} has levels "
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
    "Collapse the levels into the arms you actually want to report. A continuous "
    "longitudinal dose is a different estimand -- the intervention is a shift of a "
    "conditional density at every node rather than an assigned label -- and LTMLE does "
    "not estimate it."
)


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
            "is a dose whose spacing matters, collapse it into the arms you want to "
            "contrast -- a continuous longitudinal dose is a different estimand and LTMLE "
            "does not estimate it.",
            DataWarning,
            stacklevel=4,
        )
    return encode_treatment(raw, name, remedy=_TOO_MANY_LEVELS_REMEDY)


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
                    "monotone -- a unit that leaves does not return -- and intermittent "
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
                    "after they had already had the event. The event is assumed absorbing -- a "
                    "unit that has it leaves the risk set and does not return to it -- and a "
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
                    f"{time + 1} -- the first has {both}. Competing causes are mutually "
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
            "study -- censored, or, on a survival fit, having had the event. The estimator "
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
