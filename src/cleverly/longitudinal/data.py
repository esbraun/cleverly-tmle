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

Three conventions are declared rather than inferred, because each one changes the
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

The container holds numpy arrays and never branches on the dataframe backend; like
:class:`~cleverly.data.causal_data.CausalData` it records the backend it came from so
results are returned in it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import narwhals as nw
import numpy as np

from .._typing import BoolArray, FloatArray, IntArray
from ..data.validate import check_covariates, encode_clusters, infer_family
from ..exceptions import DataError
from ..utils.frames import as_frame, frame_from_dict, is_dataframe, matrix_from_columns

__all__ = ["Assignment", "LongitudinalData", "assignment_matrix"]

_MIN_OBSERVATIONS = 10

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
    #: ``(n, T)`` "still under observation after time ``t``".  All-true when the fit
    #: declares no censoring nodes, in which case it never enters the clever covariate.
    uncensored: BoolArray
    #: One ``(n, p_t)`` block per time point, holding the covariates measured at that
    #: time *before* the treatment decision.  A block may have no columns.
    time_varying: tuple[FloatArray, ...]
    time_varying_names: tuple[tuple[str, ...], ...]
    family: str
    outcome_name: str = "Y"
    censoring_names: tuple[str, ...] = ()
    cluster: IntArray | None = None
    cluster_name: str | None = None
    dropped_covariates: tuple[str, ...] = field(default_factory=tuple)
    _template: Any = None

    # ------------------------------------------------------------------ build

    @classmethod
    def from_frame(
        cls,
        data: Any,
        *,
        outcome: str,
        treatment: Sequence[str],
        baseline: Sequence[str],
        time_varying: Sequence[Sequence[str]] | None = None,
        censoring: Sequence[str] | None = None,
        id: str | None = None,
        family: str = "auto",
    ) -> LongitudinalData:
        """Build from one wide dataframe: a row per unit, a column per node.

        Parameters
        ----------
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

        baseline_names = [str(name) for name in baseline]
        wanted = [outcome, *treatment_names, *censor_names, *baseline_names]
        for block in blocks:
            wanted.extend(block)
        if id is not None:
            wanted.append(id)
        missing = [name for name in wanted if name not in columns]
        if missing:
            raise DataError(f"columns not found in the frame: {missing}; available: {columns}")
        _refuse_duplicates(wanted)

        return cls._build(
            outcome=frame[outcome].to_numpy(),
            baseline=matrix_from_columns(frame, baseline_names) if baseline_names else None,
            baseline_names=baseline_names,
            treatment=np.column_stack(
                [np.asarray(frame[name].to_numpy(), dtype=float) for name in treatment_names]
            ),
            treatment_names=treatment_names,
            censoring=(
                None
                if censoring is None
                else np.column_stack(
                    [np.asarray(frame[name].to_numpy(), dtype=float) for name in censor_names]
                )
            ),
            censoring_names=censor_names,
            time_varying=[matrix_from_columns(frame, block) if block else None for block in blocks],
            time_varying_names=blocks,
            cluster=None if id is None else frame[id].to_numpy(),
            cluster_name=id,
            outcome_name=outcome,
            family=family,
            template=frame,
        )

    @classmethod
    def _build(
        cls,
        *,
        outcome: Any,
        baseline: FloatArray | None,
        baseline_names: Sequence[str],
        treatment: FloatArray,
        treatment_names: Sequence[str],
        censoring: FloatArray | None,
        censoring_names: Sequence[str],
        time_varying: Sequence[FloatArray | None],
        time_varying_names: Sequence[Sequence[str]],
        cluster: Any,
        cluster_name: str | None,
        outcome_name: str,
        family: str,
        template: Any,
    ) -> LongitudinalData:
        y = np.asarray(outcome, dtype=float).reshape(-1)
        n = y.shape[0]
        if n < _MIN_OBSERVATIONS:
            raise DataError(f"need at least {_MIN_OBSERVATIONS} observations; got {n}")
        n_times = treatment.shape[1]

        if baseline is None:
            raise DataError(
                "baseline= is empty; a longitudinal fit adjusts for the baseline covariates "
                "at every node, and with none of them there is nothing to adjust for"
            )
        w, w_names, dropped = check_covariates(
            np.asarray(baseline, dtype=float), list(baseline_names)
        )

        uncensored = _read_censoring(censoring, censoring_names, n, n_times)
        # "Still under observation *before* the treatment decision at t", which is what
        # every node at time t is required to be present for.
        at_risk = np.column_stack([_through(uncensored, t) for t in range(n_times)])

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

        a = np.asarray(treatment, dtype=float)
        if a.shape[0] != n:
            raise DataError(f"treatment has {a.shape[0]} rows, expected {n}")
        for time, name in enumerate(treatment_names, start=1):
            # Named apart from the ``column`` index and ``values`` matrix of the block
            # loop above: mypy unifies a name's type across the whole function body, so
            # reusing either here is an error rather than a shadow.
            arms = a[:, time - 1]
            at_this_node = at_risk[:, time - 1]
            _check_presence(arms, at_this_node, str(name))
            seen_arms = np.unique(arms[at_this_node])
            if not np.all(np.isin(seen_arms, (0.0, 1.0))):
                raise DataError(
                    f"treatment column {name!r} takes values {seen_arms[:6].tolist()}; a "
                    "longitudinal fit takes a binary treatment at every node. A "
                    "multi-valued treatment over time is refused rather than coded, "
                    "since the clever covariate needs one factor per arm per node"
                )

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

        resolved = infer_family(y, observed_outcome) if family == "auto" else family
        if resolved not in ("binomial", "gaussian"):
            raise DataError(f"family must be 'binomial', 'gaussian' or 'auto'; got {family!r}")
        if resolved == "binomial":
            values = np.unique(y[observed_outcome])
            if not np.all(np.isin(values, (0.0, 1.0))):
                raise DataError(
                    f"family='binomial' requires a 0/1 outcome; observed {values[:6].tolist()}"
                )

        codes = None if cluster is None else encode_clusters(cluster, cluster_name or "id")

        return cls(
            outcome=y,
            baseline=w,
            baseline_names=tuple(w_names),
            treatment=a,
            treatment_names=tuple(str(name) for name in treatment_names),
            uncensored=uncensored,
            time_varying=tuple(blocks),
            time_varying_names=tuple(names),
            family=resolved,
            outcome_name=outcome_name,
            censoring_names=tuple(str(name) for name in censoring_names),
            cluster=codes,
            cluster_name=cluster_name,
            dropped_covariates=tuple(dropped),
            _template=template,
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
    def backend(self) -> str | None:
        if self._template is None:
            return None
        return str(nw.get_native_namespace(self._template).__name__)

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
        """Rows whose history :math:`H_t` is observed and regimen-consistent.

        The sequential regression at ``time`` predicts for exactly these rows, and the
        one at ``time - 1`` is *fitted* on exactly these rows -- the two sets are the
        same object, which is what makes the recursion close.  A dynamic rule does not
        weaken that: it changes *which* rows, not the identity between the two masks.
        """
        return self.uncensored_through(time - 1) & self.followed_through(assignment, time - 1)

    def following(self, assignment: Assignment, time: int) -> BoolArray:
        """Rows still on the regimen and under observation *after* ``time``.

        The clever covariate at ``time`` is supported on these rows and zero elsewhere.
        """
        return self.uncensored_through(time) & self.followed_through(assignment, time)

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

    def history_design(
        self,
        time: int,
        *,
        treatment: Assignment | None = None,
        include_current: bool = False,
    ) -> FloatArray:
        """The conditioning set of a mechanism model at ``time``.

        ``[W, L_1, ..., L_t]`` plus a column per earlier treatment, and -- with
        ``include_current``, for the censoring model, which sits *after* the treatment
        decision -- a column for the current one.  ``treatment=None`` uses what each
        unit actually received, which is how the model is fitted; an assignment sets the
        arms a regimen would have assigned, which is where it is evaluated.  A dynamic
        rule assigns a different arm to different units, so that column is per row.

        The *outcome* sequence uses :meth:`covariate_history` instead, with no treatment
        columns at all, and that holds under a rule as well as under a constant plan.
        The reason is not that a follower's past treatment is a constant -- under a rule
        it is not -- but that among the followers :math:`A_s = d_s(W, L_1, \\ldots, L_s)`
        is a *deterministic function of columns this design already carries*, since that
        is precisely the frame :meth:`history_frame` handed the rule.  Adding it would
        buy no information, only a redundant column for a penalised learner to spread a
        coefficient across.

        **No test in this repository would catch that being changed**, which is why the
        reasoning is written here rather than left to a fixture.  Two independent things
        hide it.  On the exact law the learner is saturated, and it partitions by distinct
        design row: a column that is a function of the others leaves the partition, and so
        every prediction, untouched.  Under ``glm`` the natural comparison is a rule that
        ignores the history against the constant plan it equals -- but such a rule assigns
        a *constant*, and a constant column is standardised to zeros and then ignored, on
        both sides of the comparison.  A genuinely dynamic rule would see a difference and
        has no second answer to be checked against.  Change this only with an argument.
        """
        columns = [self.covariate_history(time)]
        last = time if include_current else time - 1
        plan = None if treatment is None else assignment_matrix(treatment, self.n, self.n_times)
        for t in range(1, last + 1):
            if plan is None:
                observed = np.nan_to_num(self.treatment[:, t - 1], nan=0.0)
                columns.append(observed.reshape(-1, 1))
            else:
                columns.append(plan[:, t - 1].reshape(-1, 1))
        return np.hstack(columns)

    # ------------------------------------------------------------------ output

    def frame_like(self, payload: dict[str, Any]) -> Any:
        """Build a dataframe of ``payload`` in the backend this data came from."""
        return frame_from_dict(payload, like=self._template)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        parts = [
            f"n={self.n}",
            f"T={self.n_times}",
            f"family={self.family!r}",
            f"baseline={self.baseline.shape[1]}",
        ]
        widths = [block.shape[1] for block in self.time_varying]
        parts.append(f"time_varying={widths}")
        if self.censoring_names:
            share = float(self.uncensored_through(self.n_times).mean())
            parts.append(f"uncensored={share:.3f}")
        if self.cluster is not None:
            parts.append(f"clusters={self.n_clusters}")
        return f"LongitudinalData({', '.join(parts)})"


def _through(uncensored: BoolArray, time: int) -> BoolArray:
    """Rows uncensored at every node up to and including ``time``."""
    if time <= 0:
        return np.ones(uncensored.shape[0], dtype=bool)
    return np.asarray(uncensored[:, :time].all(axis=1), dtype=bool)


def _read_censoring(
    censoring: FloatArray | None,
    names: Sequence[str],
    n: int,
    n_times: int,
) -> BoolArray:
    """Validate the censoring columns and return the monotone indicator matrix."""
    if censoring is None:
        return np.ones((n, n_times), dtype=bool)
    values = np.asarray(censoring, dtype=float)
    if values.shape != (n, n_times):
        raise DataError(f"censoring has shape {values.shape}, expected {(n, n_times)}")

    uncensored = np.ones((n, n_times), dtype=bool)
    alive = np.ones(n, dtype=bool)
    for time in range(n_times):
        column = values[:, time]
        name = str(names[time]) if time < len(names) else f"C{time + 1}"
        present = np.isfinite(column)
        bad = alive & ~present
        if np.any(bad):
            raise DataError(
                f"censoring column {name!r} is missing for {int(bad.sum())} unit(s) that "
                "were still under observation; a unit either remained (1) or left (0)"
            )
        seen = np.unique(column[alive])
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
        uncensored[:, time] = alive & (column == 1.0)
        alive = uncensored[:, time]
    return uncensored


def _check_presence(
    column: FloatArray, required: BoolArray, name: str, *, absent: str | None = None
) -> None:
    """Every node must be recorded while the unit is under observation, and only then."""
    values = np.asarray(column, dtype=float)
    finite = np.isfinite(values)
    missing = required & ~finite
    if np.any(missing):
        if absent is not None:
            raise DataError(f"{int(missing.sum())} {absent}")
        raise DataError(
            f"{name!r} is missing for {int(missing.sum())} unit(s) that were still under "
            "observation at that time. Every node before censoring has to be recorded; a "
            "value missing for some other reason is a further node in the likelihood."
        )
    stray = ~required & finite
    if np.any(stray):
        raise DataError(
            f"{name!r} is recorded for {int(stray.sum())} unit(s) that had already been "
            "censored. The estimator conditions on the observed history and would ignore "
            "those values, so they are refused rather than silently dropped: set the "
            "nodes after a unit's censoring time to missing."
        )


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
