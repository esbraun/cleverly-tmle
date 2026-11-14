"""Covariate balance over time: how alike the groups look at each treatment decision.

A fit with a treatment at several time points, or with censoring over time, weights each unit
by the inverse of the probability of the path it took. Jackson (2016, *Epidemiology* 27(6),
Tables 1 and 2) states three checks of those weights for a treatment that varies over time,
each a difference in a covariate's mean between two treatment groups within the units who
share the same earlier treatments:

* **confounding** (Jackson's Diagnostics 1 and 3): covariates measured at or before a
  treatment decision, compared across that decision. Before weighting, a difference says the
  decision depended on the covariate given the earlier treatments; after weighting it should
  be close to zero.
* **feedback** (Diagnostic 2a): a covariate measured after an earlier decision, compared across
  that decision, with the weight of that decision alone. A difference says the treatment
  changed the covariate, which is why a method for time-varying treatments is needed.
* **censoring**: leaving the study treated as one more decision (Jackson's censoring check),
  compared across "censored at this time" and "still observed".

A continuous-time survival fit has one treatment at baseline and censoring in continuous time.
Its report compares the baseline covariates across the treatment, across the treatment among
the units whose status at each target time is known (weighted by treatment and censoring), and,
within each arm, among those units against the whole arm (weighted by censoring alone).

The report is a description of the data and the fitted weights, not an estimate: it carries
no interval.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .._refusal import RefusalReason
from .._typing import BoolArray, FloatArray, IntArray
from ..exceptions import CapabilityError, DataError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .._summary import Detail, Summary
    from ..longitudinal.data import LongitudinalData
    from ..longitudinal.sequential import Mechanism

__all__ = ["BalanceOverTime", "TimeBalanceRow"]

#: The conventional size below which a difference counts as small.
DEFAULT_THRESHOLD = 0.1

Diagnostic = Literal["confounding", "feedback", "censoring", "joint"]
#: The diagnostics in report order.
DIAGNOSTICS: tuple[str, ...] = ("confounding", "feedback", "censoring", "joint")
Over = Literal["history", "time", "all"]

#: Why a survival fit with an absorbing event reports no feedback rows.
FEEDBACK_SURVIVAL_REASON = (
    "the covariates after a treatment are recorded only on units still free of the event, so "
    "a difference across that treatment mixes its effect on the covariate with its effect on "
    "survival"
)
#: The note a collaborative fit carries.
COLLABORATIVE_NOTE = (
    "The treatment model is the collaborative selection for each arm at horizon {horizon}; "
    "covariates the selection left out are not expected to balance."
)


@dataclass(frozen=True)
class TimeBalanceRow:
    """One covariate in one comparison of one diagnostic, within one stratum.

    Parameters
    ----------
    diagnostic : {"confounding", "feedback", "censoring", "joint"}
        Which check the row belongs to (see :class:`BalanceOverTime`).
    exposure : str
        What the groups differ by: a treatment column, ``"censoring at <time>"``,
        ``"eliminated causes at <time>"``, or ``"status known at <time>"``.
    exposure_node : int
        The time point of the exposure, counted from one; for the rows read at a target time of
        a continuous-time fit, the target time's position counted from one.
    exposure_time : float
        The exposure's time: the grid time of a node when the data has one, the target time of
        a continuous-time row, else ``exposure_node``. ``0.0`` for a baseline treatment of a
        continuous-time fit.
    covariate : str
        The covariate's column.
    covariate_node : int
        The time point the covariate was measured at; ``0`` for a baseline covariate.
    distance : int
        How many time points separate the covariate from the exposure.
    history : str
        The earlier treatments the stratum shares, such as ``"A1 = 1, A2 = 0"``; empty when
        there are none.
    contrast : str
        The two groups compared, in words.
    levels : str
        The two groups, without the node: ``"1 vs 0"``, which matches rows over time.
    n_units : int
        The rows in the two groups.
    mean_group : float
        The covariate's mean in the first group, with the data's weights only.
    mean_rest : float
        Its mean in the other group, with the data's weights only.
    mean_group_weighted : float
        The first group's mean under the diagnostic's weights.
    mean_rest_weighted : float
        The other group's mean under those weights.
    spread : float
        The standard deviation both differences are divided by: the pooled standard deviation
        of the two groups in the stratum, before the diagnostic's weights. ``nan`` when it is
        zero or a group has fewer than two rows.
    difference_unweighted : float
        ``mean_group - mean_rest``.
    difference_weighted : float
        ``mean_group_weighted - mean_rest_weighted``.
    smd_unweighted : float
        The unweighted difference divided by ``spread``.
    smd_weighted : float
        The weighted difference divided by ``spread``.
    person_time_unweighted : float
        The stratum's size with the data's weights, which the averages weight the unweighted
        column by.
    person_time_weighted : float
        The stratum's size under the stabilized weights, which the averages weight the weighted
        column by.
    """

    diagnostic: str
    exposure: str
    exposure_node: int
    exposure_time: float
    covariate: str
    covariate_node: int
    distance: int
    history: str
    contrast: str
    levels: str
    n_units: int
    mean_group: float
    mean_rest: float
    mean_group_weighted: float
    mean_rest_weighted: float
    spread: float
    difference_unweighted: float
    difference_weighted: float
    smd_unweighted: float
    smd_weighted: float
    person_time_unweighted: float
    person_time_weighted: float


#: The columns of :meth:`BalanceOverTime.to_frame`, in order.
ROW_COLUMNS: tuple[str, ...] = tuple(TimeBalanceRow.__dataclass_fields__)


@dataclass(frozen=True)
class DroppedStrata:
    """The strata of one exposure that had no comparison: one of the two groups was empty.

    Parameters
    ----------
    diagnostic : str
        The diagnostic.
    exposure : str
        The exposure.
    strata : int
        The strata dropped.
    units : int
        The rows in them.
    """

    diagnostic: str
    exposure: str
    strata: int
    units: int


@dataclass(frozen=True)
class BalanceOverTime:
    """Covariate balance over time of a fit with weights that vary over time.

    ``"confounding"`` rows compare the covariates measured at or before a treatment decision
    across that decision, within the units that share the earlier treatments (Jackson 2016,
    Table 1, Diagnostic 1 before weighting and Diagnostic 3 after). ``"feedback"`` rows compare
    a covariate measured after an earlier decision across that decision (Diagnostic 2a).
    ``"censoring"`` rows compare the covariates across leaving the study at a time point and
    staying, within the units that share the treatments so far. A continuous-time survival fit
    has ``"confounding"`` rows at its baseline treatment, ``"joint"`` rows that compare the
    arms among the units whose status at a target time is known, weighted by treatment and
    censoring, and ``"censoring"`` rows that compare those units with their whole arm, weighted
    by censoring alone.

    Parameters
    ----------
    rows : tuple of TimeBalanceRow
        One row per diagnostic, exposure, stratum, comparison and covariate.
    treatments : tuple of str
        The treatment decisions compared, in time order.
    diagnostics : tuple of str
        The diagnostics with rows.
    omitted : tuple of tuple of str
        ``(diagnostic, reason)`` for each diagnostic the fit cannot report.
    dropped : tuple of DroppedStrata
        The strata with no comparison, per diagnostic and exposure.
    threshold : float
        The size an absolute standardized difference is held against.
    n : int
        The units.
    n_repeats : int
        The cross-fitting repeats the fit holds; the first one's probabilities are used.
    cross_fitted : bool
        Whether the probabilities are out-of-fold predictions; in-sample otherwise.
    limits : tuple of float or None
        The limits put on each cumulative probability before it is inverted.
    bounded_share : float
        The share of weighted rows whose cumulative probability those limits moved.
    weighted_data : bool
        Whether the data carried observation weights, which every column uses.
    family : {"longitudinal", "continuous_survival"}
        The kind of fit.
    notes : tuple of str
        Further readings of this fit's report, such as the reading of a collaborative fit.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression, LogisticRegression
    >>> from cleverly import CausalStudy, LongitudinalTreatment, RegimeMean
    >>> from cleverly.datasets import make_longitudinal
    >>> from cleverly.sensitivity import covariate_balance
    >>> frame, _ = make_longitudinal(n=600, seed=0)
    >>> study = CausalStudy(
    ...     frame,
    ...     design=LongitudinalTreatment(
    ...         outcome="Y",
    ...         treatment=["A1", "A2"],
    ...         baseline=["W1", "W2"],
    ...         time_varying=[[], ["L2"]],
    ...         censoring=["C1", "C2"],
    ...     ),
    ... )
    >>> result = study.identify(RegimeMean({"always": 1, "never": 0})).estimate(
    ...     outcome_learner=LinearRegression(),
    ...     treatment_learner=LogisticRegression(max_iter=1000),
    ...     censoring_learner=LogisticRegression(max_iter=1000),
    ...     cross_fit=False,
    ...     random_state=0,
    ... )
    >>> balance = covariate_balance(result)
    >>> balance.treatments
    ('A1', 'A2')
    >>> frame = balance.average(over="history", diagnostic="confounding")
    >>> frame[["exposure", "covariate"]].values.tolist()
    [['A1', 'W1'], ['A1', 'W2'], ['A2', 'W1'], ['A2', 'W2'], ['A2', 'L2']]
    """

    rows: tuple[TimeBalanceRow, ...]
    treatments: tuple[str, ...]
    diagnostics: tuple[str, ...]
    omitted: tuple[tuple[str, str], ...]
    dropped: tuple[DroppedStrata, ...]
    threshold: float
    n: int
    n_repeats: int
    cross_fitted: bool
    limits: tuple[float, float] | None
    bounded_share: float
    weighted_data: bool
    family: str
    notes: tuple[str, ...] = field(default=())

    def to_frame(self, backend: str | None = None) -> Any:
        """Return one row per diagnostic, exposure, stratum, comparison and covariate.

        Parameters
        ----------
        backend : {"pandas", "polars", "pyarrow"} or None, default=None
            Dataframe backend to return. ``None`` uses pandas when installed, then the first
            available backend.

        Returns
        -------
        dataframe
            The fields of :class:`TimeBalanceRow`, one column each.
        """
        from ..utils.frames import frame_from_dict

        return frame_from_dict(
            {name: [getattr(row, name) for row in self.rows] for name in ROW_COLUMNS},
            backend=backend,
        )

    def rows_of(self, diagnostic: str) -> tuple[TimeBalanceRow, ...]:
        """The rows of one diagnostic.

        Parameters
        ----------
        diagnostic : {"confounding", "feedback", "censoring", "joint"}
            The diagnostic.

        Returns
        -------
        tuple of TimeBalanceRow
            Its rows, in report order.

        Raises
        ------
        CapabilityError
            When the fit cannot report that diagnostic.
        ValueError
            When ``diagnostic`` is not one of the four.
        """
        self._check_diagnostic(diagnostic)
        return tuple(row for row in self.rows if row.diagnostic == diagnostic)

    def average(
        self,
        over: Over = "history",
        *,
        diagnostic: str | None = None,
        distances: Sequence[tuple[int, int]] | None = None,
        backend: str | None = None,
    ) -> Any:
        """Average the differences over strata, weighting each by its size.

        Jackson (2016, Table 2) averages a diagnostic over the earlier treatments, then over
        time, then over the distance between the covariate and the exposure, each stratum
        counting by its size: the data's weights for the unweighted column and the stabilized
        weights for the weighted one. The averages are of the signed differences, so strata
        that differ in opposite directions offset each other; ``smd_min`` and ``smd_max`` give
        the range of the strata's weighted standardized differences.

        Covariates are matched over time by name. A long panel's column ``"L@2"`` matches as
        ``"L"``; on a wide frame each column is its own covariate, so ``over="time"`` and
        ``over="all"`` pool a covariate only with itself.

        Parameters
        ----------
        over : {"history", "time", "all"}, default="history"
            ``"history"``: one row per exposure, comparison and covariate, averaged over the
            earlier treatments. ``"time"``: one row per distance and covariate, also averaged
            over the time points. ``"all"``: one row per covariate, also averaged over the
            distances.
        diagnostic : str or None, default=None
            One diagnostic; every diagnostic with rows when omitted.
        distances : sequence of tuple of int or None, default=None
            With ``over="all"``, ranges ``(low, high)`` of distance, both ends included, to
            average within instead of over every distance; one row per range and covariate.
        backend : {"pandas", "polars", "pyarrow"} or None, default=None
            Dataframe backend to return.

        Returns
        -------
        dataframe
            The grouping columns, then ``n_strata``, ``difference_unweighted``,
            ``difference_weighted``, ``smd_unweighted``, ``smd_weighted``, ``smd_min``,
            ``smd_max``, ``person_time_unweighted`` and ``person_time_weighted``.

        Raises
        ------
        CapabilityError
            When ``diagnostic`` is one the fit cannot report.
        ValueError
            When ``over`` or ``diagnostic`` is not one of the accepted values, or
            ``distances`` is passed without ``over="all"``.
        """
        from ..utils.frames import frame_from_dict

        if over not in ("history", "time", "all"):
            raise ValueError(f"over must be 'history', 'time' or 'all'; got {over!r}")
        if distances is not None and over != "all":
            raise ValueError("distances= applies to over='all' only")
        if diagnostic is not None:
            self._check_diagnostic(diagnostic)
        groups = averages(self.rows, over=over, diagnostic=diagnostic, distances=distances)
        if not groups:
            names = _average_keys(over, distances) + AVERAGE_COLUMNS
            return frame_from_dict({name: [] for name in names}, backend=backend)
        names = tuple(groups[0])
        return frame_from_dict(
            {name: [group[name] for group in groups] for name in names}, backend=backend
        )

    def summary(self, detail: Detail = "short") -> Summary:
        """Return the printed summary of the balance over time.

        Parameters
        ----------
        detail : {"short", "full"}, default="short"
            ``"short"`` prints, per diagnostic and exposure, each covariate's difference before
            and after weighting, averaged over the earlier treatments. ``"full"`` adds the
            strata with no comparison.

        Returns
        -------
        Summary
            A ``str`` whose value is the printed text. Its ``facts``, ``tables``, ``notes``
            and ``reading`` hold the parts as data.
        """
        from .._summary import check_detail
        from .._summary._balance_time import balance_time_summary

        return balance_time_summary(self, check_detail(detail))

    def plot(self, **kwargs: Any) -> Any:
        """Plot the balance over time.

        Parameters
        ----------
        **kwargs
            Passed to :func:`cleverly.plotting.balance`: ``diagnostic=``, ``view=``,
            ``nodes=``, ``max_covariates=``, ``labels=`` and the label options.

        Returns
        -------
        Figure
            The figure.
        """
        from ..plotting import balance

        return balance(self, **kwargs)

    def _repr_html_(self) -> str:
        return self.summary().as_html()

    def __repr__(self) -> str:
        from .._summary import short_repr

        return short_repr(
            "BalanceOverTime",
            (
                ("treatments", len(self.treatments)),
                ("diagnostics", len(self.diagnostics)),
                ("rows", len(self.rows)),
            ),
        )

    def _check_diagnostic(self, diagnostic: str) -> None:
        if diagnostic not in DIAGNOSTICS:
            raise ValueError(f"diagnostic must be one of {DIAGNOSTICS}; got {diagnostic!r}")
        reasons = dict(self.omitted)
        if diagnostic in reasons:
            raise CapabilityError(
                f"This fit reports no {diagnostic} balance: {reasons[diagnostic]}.",
                reason=RefusalReason.DIFFERENT_ESTIMAND,
                alternative="Read the fit's confounding and censoring rows instead.",
            )
        if diagnostic not in self.diagnostics:
            allowed = ", ".join(repr(name) for name in self.diagnostics)
            raise ValueError(f"This fit's balance has no {diagnostic!r} rows; it has {allowed}.")


#: The least value a unit's own treatment or censoring factor takes in a weight: it only
#: keeps a fitted probability of zero, for an outcome the unit had, from dividing by zero.
OWN_FLOOR = 1e-9

#: The value columns of :meth:`BalanceOverTime.average`.
AVERAGE_COLUMNS: tuple[str, ...] = (
    "n_strata",
    "difference_unweighted",
    "difference_weighted",
    "smd_unweighted",
    "smd_weighted",
    "smd_min",
    "smd_max",
    "person_time_unweighted",
    "person_time_weighted",
)


def _average_keys(over: str, distances: Sequence[tuple[int, int]] | None) -> tuple[str, ...]:
    if over == "history":
        return (
            "diagnostic",
            "exposure",
            "exposure_node",
            "contrast",
            "covariate",
            "covariate_node",
        )
    if over == "time":
        return ("diagnostic", "levels", "distance", "covariate")
    if distances is not None:
        return ("diagnostic", "levels", "distance_low", "distance_high", "covariate")
    return ("diagnostic", "levels", "covariate")


def covariate_identity(name: str) -> str:
    """The name a covariate is matched by over time: a long panel's ``"L@2"`` reads ``"L"``.

    Parameters
    ----------
    name : str
        The column.

    Returns
    -------
    str
        The name without a panel's time suffix.
    """
    from ..longitudinal.panel import SEPARATOR

    head, separator, tail = name.rpartition(SEPARATOR)
    return head if separator and tail.isdigit() else name


def _mean(values: list[float], weights: list[float]) -> float:
    pairs = [(v, w) for v, w in zip(values, weights, strict=True) if np.isfinite(v) and w > 0]
    total = sum(w for _, w in pairs)
    if not pairs or total <= 0:
        return float("nan")
    return float(sum(v * w for v, w in pairs) / total)


def averages(
    rows: Sequence[TimeBalanceRow],
    *,
    over: str,
    diagnostic: str | None = None,
    distances: Sequence[tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    """The size-weighted averages of :meth:`BalanceOverTime.average`, as dictionaries.

    Parameters
    ----------
    rows : sequence of TimeBalanceRow
        The rows.
    over : {"history", "time", "all"}
        What to average over.
    diagnostic : str or None, default=None
        One diagnostic, or every one.
    distances : sequence of tuple of int or None, default=None
        Ranges of distance for ``over="all"``.

    Returns
    -------
    list of dict
        One dictionary per group, in first-seen order.
    """
    keys = _average_keys(over, distances)
    groups: dict[tuple[Any, ...], list[TimeBalanceRow]] = {}
    for row in rows:
        if diagnostic is not None and row.diagnostic != diagnostic:
            continue
        values: dict[str, Any] = {
            "diagnostic": row.diagnostic,
            "exposure": row.exposure,
            "exposure_node": row.exposure_node,
            "contrast": row.contrast,
            "covariate": row.covariate if over == "history" else covariate_identity(row.covariate),
            "covariate_node": row.covariate_node,
            "levels": row.levels,
            "distance": row.distance,
        }
        if distances is not None:
            ranges = [(lo, hi) for lo, hi in distances if lo <= row.distance <= hi]
            for low, high in ranges:
                key = (row.diagnostic, row.levels, low, high, values["covariate"])
                groups.setdefault(key, []).append(row)
            continue
        groups.setdefault(tuple(values[name] for name in keys), []).append(row)
    out: list[dict[str, Any]] = []
    for key, members in groups.items():
        unweighted = [row.person_time_unweighted for row in members]
        weighted = [row.person_time_weighted for row in members]
        smds = [row.smd_weighted for row in members if np.isfinite(row.smd_weighted)]
        record = dict(zip(keys, key, strict=True))
        record.update(
            n_strata=len(members),
            difference_unweighted=_mean([r.difference_unweighted for r in members], unweighted),
            difference_weighted=_mean([r.difference_weighted for r in members], weighted),
            smd_unweighted=_mean([r.smd_unweighted for r in members], unweighted),
            smd_weighted=_mean([r.smd_weighted for r in members], weighted),
            smd_min=float(min(smds)) if smds else float("nan"),
            smd_max=float(max(smds)) if smds else float("nan"),
            person_time_unweighted=float(sum(unweighted)),
            person_time_weighted=float(sum(weighted)),
        )
        out.append(record)
    return out


# ----------------------------------------------------------------------------- the cells


@dataclass
class _Exposure:
    """One comparison's inputs: the rows, their group, stratum and weights."""

    diagnostic: str
    exposure: str
    exposure_node: int
    exposure_time: float
    rows: BoolArray
    group: IntArray  # code per row; -1 where not read
    reference: int
    labels: dict[int, str]
    strata: IntArray  # stratum code per row
    history_labels: dict[int, str]
    weight: FloatArray  # inverse probability per row (unstabilized)
    stabilized: FloatArray  # the stabilized weight per row
    covariates: list[tuple[str, int, FloatArray]]
    contrast_words: dict[int, str]
    #: The distance of every row, when it does not follow from the nodes.
    fixed_distance: int | None = None


def _moments(values: FloatArray, weights: FloatArray) -> tuple[float, float]:
    """The weighted mean and reliability-weighted variance, as the point report computes."""
    from .balance import _weighted_moments

    if values.size == 0:
        return float("nan"), float("nan")
    return _weighted_moments(values, weights)


def _cells(exposure: _Exposure, base: FloatArray) -> tuple[list[TimeBalanceRow], DroppedStrata]:
    """Every row of one exposure, and the strata it had to drop."""
    rows: list[TimeBalanceRow] = []
    read = exposure.rows & (exposure.group >= 0)
    strata = exposure.strata
    dropped_strata = 0
    dropped_units = 0
    others = sorted(code for code in exposure.labels if code != exposure.reference)
    for stratum in np.unique(strata[read]):
        inside = read & (strata == stratum)
        reference = inside & (exposure.group == exposure.reference)
        compared = False
        for code in others:
            group = inside & (exposure.group == code)
            if not group.any() or not reference.any():
                continue
            compared = True
            for name, node, column in exposure.covariates:
                finite = np.isfinite(column)
                first, rest = group & finite, reference & finite
                if not first.any() or not rest.any():
                    continue
                raw = [_moments(column[m], base[m]) for m in (first, rest)]
                weighted = [
                    _moments(column[m], base[m] * exposure.weight[m]) for m in (first, rest)
                ]
                variances = [raw[0][1], raw[1][1]]
                value = (
                    float(np.mean(variances))
                    if all(np.isfinite(v) for v in variances)
                    else float("nan")
                )
                spread = float(np.sqrt(value)) if np.isfinite(value) and value > 0 else float("nan")
                both = first | rest
                difference = raw[0][0] - raw[1][0]
                difference_weighted = weighted[0][0] - weighted[1][0]
                rows.append(
                    TimeBalanceRow(
                        diagnostic=exposure.diagnostic,
                        exposure=exposure.exposure,
                        exposure_node=exposure.exposure_node,
                        exposure_time=exposure.exposure_time,
                        covariate=name,
                        covariate_node=node,
                        distance=abs(exposure.exposure_node - node)
                        if exposure.fixed_distance is None
                        else exposure.fixed_distance,
                        history=exposure.history_labels.get(int(stratum), ""),
                        contrast=exposure.contrast_words[code],
                        levels=f"{exposure.labels[code]} vs {exposure.labels[exposure.reference]}",
                        n_units=int(both.sum()),
                        mean_group=raw[0][0],
                        mean_rest=raw[1][0],
                        mean_group_weighted=weighted[0][0],
                        mean_rest_weighted=weighted[1][0],
                        spread=spread,
                        difference_unweighted=difference,
                        difference_weighted=difference_weighted,
                        smd_unweighted=difference / spread,
                        smd_weighted=difference_weighted / spread,
                        person_time_unweighted=float(base[both].sum()),
                        person_time_weighted=float((base * exposure.stabilized)[both].sum()),
                    )
                )
        if not compared:
            dropped_strata += 1
            dropped_units += int(inside.sum())
    return rows, DroppedStrata(
        exposure.diagnostic, exposure.exposure, dropped_strata, dropped_units
    )


def _stratum_codes(matrix: IntArray, rows: BoolArray) -> IntArray:
    """One code per distinct row of ``matrix`` among ``rows``; ``-1`` elsewhere."""
    codes = np.full(matrix.shape[0], -1, dtype=np.int64)
    if not rows.any():
        return codes
    if matrix.shape[1] == 0:
        codes[rows] = 0
        return codes
    _, inverse = np.unique(matrix[rows], axis=0, return_inverse=True)
    codes[rows] = np.asarray(inverse).reshape(-1)
    return codes


def _frequency(group: IntArray, strata: IntArray, rows: BoolArray, base: FloatArray) -> FloatArray:
    """Each row's weighted share of its own group within its stratum, among ``rows``."""
    share = np.ones(group.shape[0])
    for stratum in np.unique(strata[rows]):
        inside = rows & (strata == stratum)
        total = float(base[inside].sum())
        if total <= 0:
            continue
        for code in np.unique(group[inside]):
            mask = inside & (group == code)
            share[mask] = float(base[mask].sum()) / total
    return share


def _label(data: LongitudinalData, node: int, code: int) -> str:
    levels = data.treatment_levels[node - 1] if data.treatment_levels else ()
    value: Any = levels[code] if 0 <= code < len(levels) else code
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        value = int(value)
    return str(value)


def _node_time(data: LongitudinalData, node: int) -> float:
    grid = data.time_grid
    return float(grid[node - 1]) if grid is not None else float(node)


def _time_words(data: LongitudinalData, node: int) -> str:
    grid = data.time_grid
    return f"{grid[node - 1]:g}" if grid is not None else str(node)


def refuse_continuous_nodes(data: LongitudinalData) -> None:
    """Refuse a fit with a continuous or mixed treatment node: it has no groups there."""
    for node, kind in enumerate(data.node_kinds, start=1):
        if kind in ("continuous", "mixed"):
            name = data.treatment_names[min(node, len(data.treatment_names)) - 1]
            raise DataError(
                f"{name} is a continuous treatment, so there are no treatment groups to "
                "compare at that node and the balance has no rows to fill.",
                reason=RefusalReason.DIFFERENT_ESTIMAND,
                alternative="Check the density ratio of the policy with diagnostics.support().",
            )


def longitudinal_cells(
    data: LongitudinalData,
    mechanism: Mechanism,
    g_bounds: tuple[float, float],
    *,
    feedback: bool = True,
) -> tuple[list[TimeBalanceRow], list[DroppedStrata], float]:
    """The rows of every diagnostic of a fit over discrete time.

    A function of the data, the mechanism and the bounds rather than of a result, so a test
    can hand it the true probabilities.

    Parameters
    ----------
    data : LongitudinalData
        The panel.
    mechanism : Mechanism
        The treatment and retention probabilities at each unit's observed history.
    g_bounds : tuple of float
        The limits on each cumulative probability.
    feedback : bool, default=True
        Whether to compute the feedback rows.

    Returns
    -------
    rows : list of TimeBalanceRow
        Every row, confounding first, then feedback, then censoring.
    dropped : list of DroppedStrata
        The strata with no comparison.
    bounded_share : float
        The share of weighted rows whose cumulative probability ``g_bounds`` moved.
    """
    refuse_continuous_nodes(data)
    n, horizon = data.n, data.n_times
    lower, upper = float(g_bounds[0]), float(g_bounds[1])
    base = np.asarray(data.weights, dtype=float)
    codes = np.asarray(np.nan_to_num(data.treatment, nan=-1.0), dtype=np.int64)
    exposure_nodes = [
        node
        for node in range(1, horizon + 1)
        if np.asarray(mechanism.treatment_observed[node - 1]).ndim == 2
        and np.asarray(mechanism.treatment_observed[node - 1]).shape[1] > 0
    ]
    if data.treatment_held:
        exposure_nodes = exposure_nodes[:1]
    rows_index = np.arange(n)
    # The treatment factor of each node at each unit's observed arm, one at a node with none.
    g = np.ones((n, horizon))
    for node in exposure_nodes:
        matrix = np.asarray(mechanism.treatment_observed[node - 1], dtype=float)
        valid = codes[:, node - 1] >= 0
        safe = np.where(valid, codes[:, node - 1], 0)
        g[:, node - 1] = np.where(valid, matrix[rows_index, safe], 1.0)
    censored = bool(data.censoring_names)
    composite = np.ones((n, horizon))
    if censored:
        for node in range(1, horizon + 1):
            composite[:, node - 1] = np.asarray(mechanism.censoring_observed[node - 1], dtype=float)
    composite = np.nan_to_num(composite, nan=1.0)
    g = np.nan_to_num(g, nan=1.0)

    def at_risk(node: int) -> BoolArray:
        return np.asarray(data.uncensored_through(node - 1) & data.event_free_through(node - 1))

    def prefix_c(node: int) -> FloatArray:
        """Retention through node - 1."""
        return np.prod(composite[:, : node - 1], axis=1)

    def prefix_g(node: int) -> FloatArray:
        return np.prod(g[:, :node], axis=1)

    counted = 0
    moved = 0

    def bounded(product: FloatArray, rows: BoolArray) -> FloatArray:
        nonlocal counted, moved
        clipped = np.clip(product, lower, upper)
        counted += int(rows.sum())
        moved += int((rows & (clipped != product)).sum())
        return clipped

    def floored(factor: FloatArray, rows: BoolArray) -> FloatArray:
        # The bound clips the earlier path only (plan section 2), so a unit's own factor is
        # left as fitted; a probability of exactly zero on a row that had that outcome is held
        # at OWN_FLOOR so its weight stays finite. Rows outside ``rows`` take 1.
        return np.where(rows, np.maximum(factor, OWN_FLOOR), 1.0)

    # The stabilizing numerators: the weighted share of each unit's arm within its earlier
    # treatments, among the units at risk before the node, and the marginal retention.
    numerator_g = np.ones((n, horizon))
    for index, node in enumerate(exposure_nodes):
        risk = at_risk(node) & (codes[:, node - 1] >= 0)
        history = codes[:, [s - 1 for s in exposure_nodes[:index]]]
        strata = _stratum_codes(history, risk)
        numerator_g[:, node - 1] = _frequency(codes[:, node - 1], strata, risk, base)
    # The composite "still observed" is what every risk set and prefix reads.
    target = data.uncensored
    numerator_c = np.ones((n, horizon))
    if censored:
        for node in range(1, horizon + 1):
            risk = at_risk(node)
            total = float(base[risk].sum())
            if total > 0:
                numerator_c[:, node - 1] = float(base[risk & target[:, node - 1]].sum()) / total

    def covariates_through(node: int) -> list[tuple[str, int, FloatArray]]:
        out = [
            (name, 0, np.asarray(data.baseline[:, index], dtype=float))
            for index, name in enumerate(data.baseline_names)
        ]
        for t in range(1, node + 1):
            block = data.time_varying[t - 1]
            out.extend(
                (name, t, np.asarray(block[:, index], dtype=float))
                for index, name in enumerate(data.time_varying_names[t - 1])
            )
        return out

    def history_of(nodes: list[int], rows: BoolArray) -> tuple[IntArray, dict[int, str]]:
        matrix = codes[:, [s - 1 for s in nodes]]
        strata = _stratum_codes(matrix, rows)
        words: dict[int, str] = {}
        for stratum in np.unique(strata[rows]):
            first = int(np.flatnonzero(rows & (strata == stratum))[0])
            words[int(stratum)] = ", ".join(
                f"{data.treatment_names[min(s, len(data.treatment_names)) - 1]} = "
                f"{_label(data, s, int(codes[first, s - 1]))}"
                for s in nodes
            )
        return strata, words

    def node_name(node: int) -> str:
        return data.treatment_names[min(node, len(data.treatment_names)) - 1]

    def arm_words(node: int) -> tuple[dict[int, str], dict[int, str]]:
        levels = data.treatment_levels[node - 1] if data.treatment_levels else ()
        labels = {code: _label(data, node, code) for code in range(len(levels))}
        name = node_name(node)
        reference = labels.get(0, "0")
        words = {
            code: f"{name} = {label} vs {name} = {reference}"
            for code, label in labels.items()
            if code != 0
        }
        return labels, words

    exposures: list[_Exposure] = []
    for index, node in enumerate(exposure_nodes):
        rows = at_risk(node) & (codes[:, node - 1] >= 0)
        earlier = exposure_nodes[:index]
        strata, words = history_of(earlier, rows)
        weight = 1.0 / bounded(prefix_g(node) * prefix_c(node), rows)
        stabilized = (
            np.prod(numerator_g[:, :node], axis=1) * np.prod(numerator_c[:, : node - 1], axis=1)
        ) * weight
        labels, contrasts = arm_words(node)
        exposures.append(
            _Exposure(
                "confounding",
                node_name(node),
                node,
                _node_time(data, node),
                rows,
                codes[:, node - 1],
                0,
                labels,
                strata,
                words,
                weight,
                stabilized,
                covariates_through(node),
                contrasts,
            )
        )
    if feedback:
        for index, node in enumerate(exposure_nodes):
            for later in range(node + 1, horizon + 1):
                block = data.time_varying_names[later - 1]
                if not block:
                    continue
                rows = at_risk(later) & (codes[:, node - 1] >= 0)
                strata, words = history_of(exposure_nodes[:index], rows)
                # The bound applies to the cumulative retention, as the plan defines; g_s stays out.
                factor_s = np.where(rows, g[:, node - 1], 1.0)
                weight = 1.0 / (floored(factor_s, rows) * bounded(prefix_c(later), rows))
                stabilized = (
                    numerator_g[:, node - 1] * np.prod(numerator_c[:, : later - 1], axis=1)
                ) * weight
                labels, contrasts = arm_words(node)
                exposures.append(
                    _Exposure(
                        "feedback",
                        node_name(node),
                        node,
                        _node_time(data, node),
                        rows,
                        codes[:, node - 1],
                        0,
                        labels,
                        strata,
                        words,
                        weight,
                        stabilized,
                        [
                            (
                                name,
                                later,
                                np.asarray(data.time_varying[later - 1][:, i], dtype=float),
                            )
                            for i, name in enumerate(block)
                        ],
                        contrasts,
                    )
                )
    if censored:
        retained = data.retained if data.eliminated else data.uncensored
        factor_of = (
            mechanism.censoring_factor_observed if data.eliminated else mechanism.censoring_observed
        )
        assert retained is not None
        for node in range(1, horizon + 1):
            earlier = [s for s in exposure_nodes if s <= node]
            risk = at_risk(node)
            strata, words = history_of(earlier, risk)
            factor = np.nan_to_num(np.asarray(factor_of[node - 1], dtype=float), nan=1.0)
            stayed = np.asarray(retained[:, node - 1], dtype=bool)
            # Group code 0: stayed under observation (the reference); 1: censored at the node.
            group = np.where(risk, np.where(stayed, 0, 1), -1)
            # Rows outside the risk set take 1, so their unused weight stays finite.
            own = np.where(risk, np.where(stayed, factor, 1.0 - factor), 1.0)
            # The bound applies to the path through A_t; the unit's own censoring outcome at t
            # multiplies in afterwards, so a small 1 - c_t is not clipped away.
            weight = 1.0 / (bounded(prefix_g(node) * prefix_c(node), risk) * floored(own, risk))
            share = _frequency(group, np.zeros(n, dtype=np.int64), risk, base)
            stabilized = (
                np.prod(numerator_g[:, :node], axis=1)
                * np.prod(numerator_c[:, : node - 1], axis=1)
                * share
            ) * weight
            when = _time_words(data, node)
            exposures.append(
                _Exposure(
                    "censoring",
                    f"censoring at {when}",
                    node,
                    _node_time(data, node),
                    risk,
                    group,
                    0,
                    {0: "observed", 1: "censored"},
                    strata,
                    words,
                    weight,
                    stabilized,
                    covariates_through(node),
                    {1: f"censored at {when} vs observed"},
                )
            )
            if data.eliminated:
                passed = risk & stayed
                competing = np.nan_to_num(
                    np.asarray(mechanism.competing_observed[node - 1], dtype=float), nan=1.0
                )
                free = np.asarray(data.uncensored[:, node - 1], dtype=bool)
                group = np.where(passed, np.where(free, 0, 1), -1)
                own = np.where(passed, np.where(free, competing, 1.0 - competing), 1.0)
                weight = 1.0 / (
                    bounded(prefix_g(node) * prefix_c(node) * factor, passed) * floored(own, passed)
                )
                share = _frequency(group, np.zeros(n, dtype=np.int64), passed, base)
                stabilized = (
                    np.prod(numerator_g[:, :node], axis=1)
                    * np.prod(numerator_c[:, : node - 1], axis=1)
                    * share
                ) * weight
                strata_e, words_e = history_of(earlier, passed)
                exposures.append(
                    _Exposure(
                        "censoring",
                        f"eliminated causes at {when}",
                        node,
                        _node_time(data, node),
                        passed,
                        group,
                        0,
                        {0: "no eliminated event", 1: "eliminated event"},
                        strata_e,
                        words_e,
                        weight,
                        stabilized,
                        covariates_through(node),
                        {1: f"eliminated event at {when} vs none"},
                    )
                )
    rows_out: list[TimeBalanceRow] = []
    dropped: list[DroppedStrata] = []
    for exposure in exposures:
        produced, lost = _cells(exposure, base)
        rows_out.extend(produced)
        if lost.strata:
            dropped.append(lost)
    moved_share = moved / counted if counted else 0.0
    return rows_out, dropped, float(moved_share)


def continuous_cells(
    *,
    time: FloatArray,
    event: FloatArray,
    arm: IntArray,
    baseline: FloatArray,
    names: Sequence[str],
    arm_labels: tuple[str, str],
    treatment: str,
    p1: FloatArray,
    retention_before: FloatArray,
    retention_at_time: FloatArray,
    weight_floor: float,
    horizons: Sequence[float],
) -> tuple[list[TimeBalanceRow], float]:
    """The rows of a continuous-time survival fit.

    Written over arrays rather than a result, so a test can hand it the true probabilities.

    Parameters
    ----------
    time : ndarray
        ``(n,)`` each row's observed time.
    event : ndarray
        ``(n,)`` its event code, ``0`` when censored.
    arm : ndarray
        ``(n,)`` its arm code: ``0`` reference, ``1`` treated.
    baseline : ndarray
        ``(n, p)`` the baseline covariates.
    names : sequence of str
        Their names.
    arm_labels : tuple of str
        ``(reference, treated)``.
    treatment : str
        The treatment's name.
    p1 : ndarray
        ``(n,)`` the probability of the treated arm.
    retention_before : ndarray
        ``(2, n, K)`` the probability of staying uncensored just before each target time.
    retention_at_time : ndarray
        ``(n,)`` that probability just before each row's own time, at its arm.
    weight_floor : float
        The lower bound on each product of probabilities before it is inverted.
    horizons : sequence of float
        The target times.

    Returns
    -------
    rows : list of TimeBalanceRow
        The confounding rows, then per target time the joint and the censoring rows.
    bounded_share : float
        The share of weighted rows whose product the floor raised.
    """
    n = int(time.shape[0])
    base = np.ones(n)
    arm = np.asarray(arm, dtype=np.int64)
    pi = np.where(arm == 1, p1, 1.0 - p1)
    labels = {0: str(arm_labels[0]), 1: str(arm_labels[1])}
    contrast = {1: f"{treatment} = {labels[1]} vs {treatment} = {labels[0]}"}
    covariates = [
        (name, 0, np.asarray(baseline[:, index], dtype=float)) for index, name in enumerate(names)
    ]
    everyone = np.ones(n, dtype=bool)
    zero = np.zeros(n, dtype=np.int64)
    share_arm = _frequency(arm, zero, everyone, base)
    counted = 0
    moved = 0

    def floor(product: FloatArray, rows: BoolArray) -> FloatArray:
        nonlocal counted, moved
        raised = np.maximum(product, weight_floor)
        counted += int(rows.sum())
        moved += int((rows & (raised != product)).sum())
        return raised

    out: list[TimeBalanceRow] = []
    weight = 1.0 / floor(pi, everyone)
    exposures = [
        _Exposure(
            "confounding",
            treatment,
            1,
            0.0,
            everyone,
            arm,
            0,
            labels,
            zero,
            {},
            weight,
            share_arm * weight,
            covariates,
            contrast,
            fixed_distance=0,
        )
    ]
    out.extend(_cells(exposures[0], base)[0])
    for k, horizon in enumerate(horizons):
        known = ((event > 0) & (time <= horizon)) | (time >= horizon)
        followed = time >= horizon
        retention = np.where(followed, retention_before[arm, np.arange(n), k], retention_at_time)
        weight = 1.0 / floor(pi * retention, known)
        joint = _Exposure(
            "joint",
            f"status known at {horizon:g}",
            k + 1,
            float(horizon),
            known,
            arm,
            0,
            labels,
            zero,
            {},
            weight,
            share_arm * weight,
            covariates,
            contrast,
            fixed_distance=0,
        )
        out.extend(_cells(joint, base)[0])
        # Within each arm: the status-known units, weighted by censoring alone, against the
        # whole arm.
        for code in (0, 1):
            inside = arm == code
            group_known = inside & known
            for name, node, column in covariates:
                raw_all = _moments(column[inside], base[inside])
                raw_known = _moments(column[group_known], base[group_known])
                inverse = 1.0 / np.maximum(retention[group_known], weight_floor)
                weighted_known = _moments(column[group_known], inverse)
                spread = (
                    float(np.sqrt(raw_all[1]))
                    if np.isfinite(raw_all[1]) and raw_all[1] > 0
                    else float("nan")
                )
                difference = raw_known[0] - raw_all[0]
                difference_weighted = weighted_known[0] - raw_all[0]
                out.append(
                    TimeBalanceRow(
                        diagnostic="censoring",
                        exposure=f"status known at {horizon:g}",
                        exposure_node=k + 1,
                        exposure_time=float(horizon),
                        covariate=name,
                        covariate_node=node,
                        distance=0,
                        history="",
                        contrast=f"known at {horizon:g} vs all, {treatment} = {labels[code]}",
                        levels=f"known vs all, {labels[code]}",
                        n_units=int(inside.sum()),
                        mean_group=raw_known[0],
                        mean_rest=raw_all[0],
                        mean_group_weighted=weighted_known[0],
                        mean_rest_weighted=raw_all[0],
                        spread=spread,
                        difference_unweighted=difference,
                        difference_weighted=difference_weighted,
                        smd_unweighted=difference / spread,
                        smd_weighted=difference_weighted / spread,
                        person_time_unweighted=float(inside.sum()),
                        person_time_weighted=float(inside.sum()),
                    )
                )
    share = moved / counted if counted else 0.0
    return out, float(share)


# ----------------------------------------------------------------------------- the readers


def balance_over_time(result: Any, *, threshold: float) -> BalanceOverTime:
    """The balance over time of a longitudinal or continuous-time survival fit.

    Parameters
    ----------
    result : LongitudinalResult or ContinuousSurvivalResult
        The fit.
    threshold : float
        The size an absolute standardized difference is held against.

    Returns
    -------
    BalanceOverTime
        The report.
    """
    from ..survival.result import ContinuousSurvivalResult

    if isinstance(result, ContinuousSurvivalResult):
        return _continuous(result, threshold)
    return _longitudinal(result, threshold)


def _diagnostics_present(rows: Sequence[TimeBalanceRow]) -> tuple[str, ...]:
    present = {row.diagnostic for row in rows}
    return tuple(name for name in DIAGNOSTICS if name in present)


def _longitudinal(result: Any, threshold: float) -> BalanceOverTime:
    data = result.data
    config = result.config
    mechanism = result.mechanism
    survival = bool(getattr(data, "is_survival", False))
    rows, dropped, bounded_share = longitudinal_cells(
        data, mechanism, tuple(config.g_bounds), feedback=not survival
    )
    omitted: tuple[tuple[str, str], ...] = (
        (("feedback", FEEDBACK_SURVIVAL_REASON),) if survival else ()
    )
    notes: list[str] = []
    if getattr(result, "selection", None) is not None:
        horizon = max(config.horizons) if getattr(config, "horizons", ()) else data.n_times
        notes.append(COLLABORATIVE_NOTE.format(horizon=horizon))
    folds = getattr(result, "folds", None)
    cross_fitted = bool(folds is not None and not getattr(folds, "is_single", True))
    treatments = tuple(
        dict.fromkeys(row.exposure for row in rows if row.diagnostic == "confounding")
    )
    return BalanceOverTime(
        rows=tuple(rows),
        treatments=treatments,
        diagnostics=_diagnostics_present(rows),
        omitted=omitted,
        dropped=tuple(dropped),
        threshold=float(threshold),
        n=int(data.n),
        n_repeats=len(result.repeats),
        cross_fitted=cross_fitted,
        limits=(float(config.g_bounds[0]), float(config.g_bounds[1])),
        bounded_share=bounded_share,
        weighted_data=bool(np.ptp(np.asarray(data.weights, dtype=float)) > 0.0),
        family="longitudinal",
        notes=tuple(notes),
    )


def _continuous(result: Any, threshold: float) -> BalanceOverTime:
    data = result.data
    if data is None:
        raise CapabilityError(
            "covariate_balance reads the rows the fit read; this continuous-time survival "
            "result was built by hand and holds none.",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    record = result.record
    treatment = str(data.names[2])
    rows, bounded_share = continuous_cells(
        time=np.asarray(data.time, dtype=float),
        event=np.asarray(data.event, dtype=float),
        arm=np.asarray(record.arm),
        baseline=np.asarray(data.baseline, dtype=float),
        names=data.baseline_names,
        arm_labels=(str(result.arms[0]), str(result.arms[1])),
        treatment=treatment,
        p1=np.asarray(record.p1, dtype=float),
        retention_before=np.asarray(record.retention_before, dtype=float),
        retention_at_time=np.asarray(record.retention_at_time, dtype=float),
        weight_floor=float(record.weight_floor),
        horizons=tuple(float(h) for h in result.horizons),
    )
    return BalanceOverTime(
        rows=tuple(rows),
        treatments=(treatment,),
        diagnostics=_diagnostics_present(rows),
        omitted=(),
        dropped=(),
        threshold=float(threshold),
        n=int(data.n),
        n_repeats=1,
        cross_fitted=bool(record.cross_fitted),
        limits=(float(record.weight_floor), 1.0),
        bounded_share=bounded_share,
        weighted_data=False,
        family="continuous_survival",
    )
