r"""Read a long panel: one row per unit and decision time.

Customer data usually arrives long: a row per customer and decision time, holding the
covariates measured before that decision, the decision, whether the customer stays under
observation, and the outcome.  :class:`~cleverly.longitudinal.LongitudinalData` holds one
trajectory per unit.  :meth:`~cleverly.longitudinal.LongitudinalData.from_long_panel`
converts the first into the second, and
:meth:`~cleverly.longitudinal.LongitudinalData.to_long_panel` converts back.

The conversion is a reshape followed by
:meth:`~cleverly.longitudinal.LongitudinalData.from_frame`, so every check of the wide path
runs unchanged.  The checks here are the ones a wide frame cannot express: a duplicated or
missing decision time, late entry, a unit-level column that varies within a unit, and a
column that would put a later value or the outcome into the history a rule reads.

A generated wide column is named ``"{column}@{t}"``, with ``t`` the node index counted from
one.  The original decision-time labels stay in :class:`PanelLayout`.  A rule at node ``t``
reads ``"{column}@{s}"`` for ``s <= t``.  :func:`each_node` writes one rule for every node.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np

from ..exceptions import DataError
from ..utils.frames import as_frame, backend_of, column_array, frame_from_dict, is_dataframe

if TYPE_CHECKING:
    from .data import LongitudinalData

__all__ = ["NodeRule", "PanelLayout", "each_node"]

#: The separator of a generated wide column name, ``"{column}@{t}"``.
SEPARATOR = "@"


@dataclass(frozen=True)
class PanelLayout:
    """The long-panel roles a container was built from.

    :meth:`~cleverly.longitudinal.LongitudinalData.from_long_panel` records it, and
    :meth:`~cleverly.longitudinal.LongitudinalData.to_long_panel` reads it.

    Parameters
    ----------
    unit : str
        The unit identifier column.
    time : str
        The decision-time column.
    decision_times : tuple
        The decision-time labels, in node order.  Node ``t`` is ``decision_times[t - 1]``.
    treatment : tuple of str
        The treatment column, or the components of a vector treatment.
    baseline : tuple of str
        The unit-level covariates.
    time_varying : tuple of str
        The covariates measured at each row before its decision.
    censoring : str or None
        The observation indicator column.
    outcome : str or None
        The end-of-study outcome column.
    event : tuple of str
        The event indicator columns of a survival outcome, one per cause.
    causes : tuple of str
        The cause labels of competing risks, else empty.
    cluster : str or None
        The cluster identifier column.
    cluster_labels : tuple
        The cluster identifiers in the order that the container's cluster codes index.
    weights : str or None
        The unit-level weight column.
    omitted : tuple of tuple of (str, int)
        Each time-varying column and node that the leading run of missing values left out.
    treatment_probabilities : tuple of tuple or None
        The ``(level, column)`` pairs of a known treatment mechanism.
    censoring_probabilities : str or None
        The column of the known retention probabilities.
    """

    unit: str
    time: str
    decision_times: tuple[Any, ...]
    treatment: tuple[str, ...]
    baseline: tuple[str, ...]
    time_varying: tuple[str, ...] = ()
    censoring: str | None = None
    outcome: str | None = None
    event: tuple[str, ...] = ()
    causes: tuple[str, ...] = ()
    cluster: str | None = None
    cluster_labels: tuple[Any, ...] = ()
    weights: str | None = None
    omitted: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    treatment_probabilities: tuple[tuple[Any, str], ...] | None = None
    censoring_probabilities: str | None = None

    def describe(self) -> str:
        """Return the settings line of a fit on an adapted panel.

        Returns
        -------
        str
            The unit and time columns and the decision-time labels.
        """
        times = ", ".join(str(label) for label in self.decision_times)
        line = (
            f"long panel: unit column {self.unit!r}, decision times {times} (column {self.time!r})"
        )
        if self.omitted:
            gaps = ", ".join(f"{name}@{node}" for name, node in self.omitted)
            line += f"; no value recorded, so left out: {gaps}"
        return line


def node_name(column: str, node: int) -> str:
    """The wide name of ``column`` at node ``node``, counted from one."""
    return f"{column}{SEPARATOR}{node}"


@dataclass(frozen=True)
class NodeRule:
    """One node of a rule written for every node: calls ``rule(frame, time)``.

    Build these with :func:`each_node`.

    Parameters
    ----------
    rule : callable
        Maps a node's frame and the node index to one value per row.
    time : int
        The node, counted from one.
    """

    rule: Callable[[Any, int], Any]
    time: int

    def __call__(self, frame: Any) -> Any:
        """Return ``rule(frame, time)``.

        Parameters
        ----------
        frame : DataFrame
            The history or policy frame of the node.

        Returns
        -------
        array-like
            What ``rule`` returns.
        """
        return self.rule(frame, self.time)

    @property
    def __name__(self) -> str:
        """The rule's own name and the node, as ``describe_plan`` prints it."""
        name = getattr(self.rule, "__name__", "<lambda>")
        return f"{'d' if name == '<lambda>' else name}@{self.time}"


def each_node(rule: Callable[[Any, int], Any], n_times: int) -> tuple[NodeRule, ...]:
    """Write one rule for every node, each reading its own node's columns by name.

    Node ``t`` calls ``rule(frame, t)``.  On a container built from a long panel, the
    covariate ``eng`` measured at node ``t`` is the column ``f"eng@{t}"``.  A rule that names
    it reads exactly that node's value.  When that column is absent, for example because it
    was constant at the node and screened out, the fit raises rather than read an earlier
    node.

    Parameters
    ----------
    rule : callable
        ``rule(frame, t)`` returns one arm per row for a rule node, or one probability per
        level for a :class:`~cleverly.interventions.Stochastic` density.
    n_times : int
        The number of nodes.

    Returns
    -------
    tuple of NodeRule
        One callable per node, in node order.

    Raises
    ------
    DataError
        If ``n_times`` is not a positive integer.

    See Also
    --------
    cleverly.longitudinal.DynamicRegimen : The plan that holds the rules.
    cleverly.longitudinal.LongitudinalData.from_long_panel : The names the rules read.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from cleverly.longitudinal import each_node
    >>> rules = each_node(lambda h, t: np.where(h[f"eng@{t}"] > 0, "offer", "none"), 2)
    >>> history = pd.DataFrame({"W": [0.0, 1.0], "eng@1": [0.5, -1.0], "eng@2": [-2.0, 3.0]})
    >>> rules[0](history).tolist(), rules[1](history).tolist()
    (['offer', 'none'], ['none', 'offer'])
    """
    if isinstance(n_times, bool) or not isinstance(n_times, (int, np.integer)) or n_times < 1:
        raise DataError(f"each_node needs a positive number of nodes; got {n_times!r}")
    return tuple(NodeRule(rule, time) for time in range(1, int(n_times) + 1))


# ------------------------------------------------------------------ reading


def _present(values: Any) -> np.ndarray:
    """Whether each cell of an object or float array holds a recorded value."""
    array = np.asarray(values)
    if array.dtype.kind in "biufc":
        return np.isfinite(np.asarray(array, dtype=float))
    from .data import _categorical_value_is_present

    return np.fromiter(
        (_categorical_value_is_present(value) for value in array), dtype=bool, count=array.size
    )


def _sorted_unique(values: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    """``np.unique`` with the inverse, refusing values that do not sort."""
    try:
        labels, codes = np.unique(values, return_inverse=True)
    except TypeError as error:
        raise DataError(
            f"column {name!r} holds values that do not sort against each other, such as "
            "numbers and strings together. Use one type"
        ) from error
    return labels, np.asarray(codes, dtype=np.int64).reshape(-1)


def _orders_in_time(value: Any) -> bool:
    """Whether a decision-time label sorts in time order: a number or a date."""
    if isinstance(value, (bool, np.bool_)):
        return False
    return isinstance(
        value,
        (int, float, np.integer, np.floating, np.datetime64, np.timedelta64, date, timedelta),
    )


def _label(value: Any) -> Any:
    """A numpy scalar as its Python value, so messages and layouts print plainly."""
    return value.item() if isinstance(value, np.generic) else value


@dataclass
class _Panel:
    """The reshaped index of a long panel: unit codes, node positions and row order."""

    unit_labels: np.ndarray
    unit_code: np.ndarray
    node: np.ndarray
    grid: tuple[Any, ...]
    present: np.ndarray
    first_row: np.ndarray
    last_node: np.ndarray

    @property
    def n(self) -> int:
        return int(self.unit_labels.shape[0])

    @property
    def n_times(self) -> int:
        return len(self.grid)

    def unit(self, code: int) -> Any:
        return _label(self.unit_labels[code])

    def time(self, node: int) -> Any:
        return self.grid[node]

    def scatter(self, values: np.ndarray, *, dtype: Any = float) -> np.ndarray:
        """Row values into an ``(n, T)`` array, missing where the unit has no row."""
        if dtype is float:
            out = np.full((self.n, self.n_times), np.nan)
        else:
            out = np.full((self.n, self.n_times), None, dtype=object)
        out[self.unit_code, self.node] = values
        return out


def _index(frame: Any, unit: str, time: str, times: Sequence[Any] | None) -> _Panel:
    """Validate the unit and time columns and map every row onto a unit and a node."""
    ids = column_array(frame, unit, dtype=object)
    stamps = column_array(frame, time, dtype=object)
    for name, values in ((unit, ids), (time, stamps)):
        missing = ~_present(values)
        if np.any(missing):
            raise DataError(
                f"column {name!r} is missing on {int(missing.sum())} row(s); every row needs a "
                "unit and a decision time"
            )
    unit_labels, unit_code = _sorted_unique(ids, unit)
    observed, node = _sorted_unique(stamps, time)
    if times is None:
        # Sorting defines the node order, so it must be the time order.  Numbers, dates and
        # durations sort in time; labels such as "w1", "w10", "w2" do not, and would put a later
        # node's covariates into an earlier node's history.
        unordered = [_label(value) for value in observed if not _orders_in_time(value)]
        if unordered:
            raise DataError(
                f"column {time!r} holds decision time(s) {unordered[:3]} that are not numbers, "
                "dates or durations, so their sort order need not be their time order. "
                "Pass times= with every decision time in time order"
            )
        grid = tuple(_label(value) for value in observed)
    else:
        # A declared grid is taken in the order given: that order is the time order.
        grid = tuple(_label(value) for value in times)
        if len(set(grid)) != len(grid):
            raise DataError("times= lists a decision time more than once")
        if all(_orders_in_time(value) for value in grid) and any(
            later <= earlier for earlier, later in pairwise(grid)
        ):
            raise DataError(
                f"times= lists {list(grid)}, which is not increasing. It gives the decision "
                "times in time order"
            )
        position = {value: index for index, value in enumerate(grid)}
        unknown = [_label(value) for value in observed if _label(value) not in position]
        if unknown:
            raise DataError(
                f"column {time!r} holds decision time(s) {unknown[:5]} that times= does not "
                f"list. times= is {list(grid)}; it names every decision time, and cannot "
                "coarsen them"
            )
        remap = np.asarray([position[_label(value)] for value in observed], dtype=np.int64)
        node = remap[node]
    n_units, n_times = unit_labels.shape[0], len(grid)
    key = unit_code * n_times + node
    counts = np.bincount(key, minlength=n_units * n_times)
    if np.any(counts > 1):
        first = int(np.flatnonzero(counts > 1)[0])
        raise DataError(
            f"{int((counts > 1).sum())} (unit, decision time) pair(s) appear on more than one "
            f"row; the first is unit {_label(unit_labels[first // n_times])!r} at "
            f"{grid[first % n_times]!r}. A long panel holds one row per unit and decision time"
        )
    present = (counts > 0).reshape(n_units, n_times)
    first_node = np.argmax(present, axis=1)
    last_node = n_times - 1 - np.argmax(present[:, ::-1], axis=1)
    late = first_node > 0
    if np.any(late):
        code = int(np.flatnonzero(late)[0])
        raise DataError(
            f"{int(late.sum())} unit(s) have no row at the first decision time "
            f"{grid[0]!r}; the first is unit {_label(unit_labels[code])!r}, whose first row is "
            f"at {grid[first_node[code]]!r}. Every trajectory starts at the first decision; "
            "late entry is a different observation scheme"
        )
    gap = present.sum(axis=1) != last_node + 1
    if np.any(gap):
        code = int(np.flatnonzero(gap)[0])
        skipped = grid[int(np.flatnonzero(~present[code, : last_node[code] + 1])[0])]
        raise DataError(
            f"{int(gap.sum())} unit(s) skip a decision time between two of their rows; the "
            f"first is unit {_label(unit_labels[code])!r}, with no row at {skipped!r}. "
            "Intermittent observation is not censoring: a unit that leaves does not return"
        )
    # The row of each unit at its first node, for the unit-level columns.
    first_row = np.full(n_units, -1, dtype=np.int64)
    starts = np.flatnonzero(node == 0)
    first_row[unit_code[starts]] = starts
    return _Panel(
        unit_labels=unit_labels,
        unit_code=unit_code,
        node=node,
        grid=grid,
        present=present,
        first_row=first_row,
        last_node=last_node,
    )


def _unit_level(
    panel: _Panel, values: np.ndarray, name: str, *, role: str, first_row: bool
) -> np.ndarray:
    """One value per unit from a column that a unit repeats on its rows.

    A unit may leave the column missing on some rows, and must not hold two different
    values.  With ``first_row``, the value must be recorded on the unit's first row: a
    unit-level column is measured before the first decision.
    """
    present = _present(values)
    numeric = np.asarray(values).dtype.kind in "biufc"
    out: np.ndarray = np.full(panel.n, np.nan) if numeric else np.full(panel.n, None, dtype=object)
    seen = np.zeros(panel.n, dtype=bool)
    clash = np.zeros(panel.n, dtype=bool)
    order = np.lexsort((panel.node, panel.unit_code))
    for row in order:
        if not present[row]:
            continue
        code = panel.unit_code[row]
        value = values[row]
        if not seen[code]:
            out[code] = value
            seen[code] = True
        elif not clash[code] and not _same(out[code], value):
            clash[code] = True
    if np.any(clash):
        code = int(np.flatnonzero(clash)[0])
        reason = {
            "baseline": "A column that changes over time is time-varying. Read as a baseline "
            "covariate, a later value would let a rule at an earlier node read the future",
            "outcome": "An end-of-study outcome holds one value per unit. A per-row event "
            "indicator is a survival outcome, declared with event=. A long panel does not "
            "take a repeated outcome in this version; declare it on a wide frame with Repeated",
        }.get(role, f"A {role} column holds one value per unit")
        raise DataError(
            f"column {name!r} holds more than one value for {int(clash.sum())} unit(s); the "
            f"first is unit {panel.unit(code)!r}. {reason}"
        )
    if first_row:
        late = seen & ~present[panel.first_row]
        if np.any(late):
            code = int(np.flatnonzero(late)[0])
            rows = np.flatnonzero((panel.unit_code == code) & present)
            when = panel.time(int(panel.node[rows].min()))
            raise DataError(
                f"column {name!r} is missing on the first row of {int(late.sum())} unit(s) and "
                f"recorded later; the first is unit {panel.unit(code)!r}, first recorded at "
                f"{when!r}. A {role} column is measured before the first decision, so it is "
                "recorded on the unit's first row"
            )
    return out


def _same(left: Any, right: Any) -> bool:
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return left is right


def from_long_panel(
    cls: type[LongitudinalData],
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
    """The body of :meth:`LongitudinalData.from_long_panel`, which documents it."""
    if not is_dataframe(data):
        raise DataError("LongitudinalData.from_long_panel expects a pandas or polars DataFrame")
    frame = as_frame(data)
    columns = [str(name) for name in frame.columns]

    if outcome is not None and not isinstance(outcome, str):
        raise DataError(
            "from_long_panel takes outcome= as one column name, an end-of-study outcome. A "
            "repeated outcome (Repeated([...])) is declared on a wide frame, with one column "
            "per node; a long panel with a repeated outcome on each row is not supported in "
            "this version"
        )
    if (outcome is None) == (event is None):
        raise DataError(
            "declare exactly one of outcome= (one end-of-study value per unit) and event= "
            "(an event indicator on each row, for a survival outcome)"
        )
    components = (str(treatment),) if isinstance(treatment, str) else tuple(map(str, treatment))
    if not components:
        raise DataError("treatment= is empty; a long panel needs a treatment column")
    if isinstance(event, Mapping):
        if not event:
            raise DataError("event= is an empty mapping; name each cause and its column")
        causes = tuple(str(label) for label in event)
        event_columns = tuple(str(name) for name in event.values())
    elif event is not None:
        causes = ()
        event_columns = (str(event),)
    else:
        causes, event_columns = (), ()
    baseline_names = tuple(str(name) for name in baseline)
    varying = tuple(str(name) for name in time_varying)
    roles: list[tuple[str, str]] = [("unit=", unit), ("time=", time)]
    roles += [("treatment=", name) for name in components]
    roles += [("baseline=", name) for name in baseline_names]
    roles += [("time_varying=", name) for name in varying]
    roles += [("event=", name) for name in event_columns]
    for role, name in (
        ("censoring=", censoring),
        ("outcome=", outcome),
        ("cluster=", cluster),
        ("weights=", weights),
    ):
        if name is not None:
            roles.append((role, str(name)))
    probability_columns: list[str] = []
    if treatment_probabilities is not None:
        if not isinstance(treatment_probabilities, Mapping) or not all(
            isinstance(name, str) for name in treatment_probabilities.values()
        ):
            raise DataError(
                "treatment_probabilities= maps each treatment level to the column that holds "
                "P(A_t = level | past) on each row"
            )
        probability_columns += list(treatment_probabilities.values())
    if censoring_probabilities is not None:
        if censoring is None:
            raise DataError("censoring_probabilities= needs censoring=")
        probability_columns.append(str(censoring_probabilities))
    missing = [name for _, name in roles if name not in columns]
    missing += [name for name in probability_columns if name not in columns]
    if missing:
        raise DataError(f"columns not found in the frame: {missing}; available: {columns}")
    owner: dict[str, str] = {}
    for role, name in [*roles, *(("a probability", name) for name in probability_columns)]:
        if name in owner:
            raise DataError(
                f"column {name!r} is declared as {owner[name]} and as {role}. A column holds one "
                "role; an outcome in a second role would let a rule read it"
            )
        owner[name] = role

    panel = _index(frame, unit, time, times)
    n, n_times = panel.n, panel.n_times
    nodes = range(1, n_times + 1)

    generated = {
        node_name(name, t): name
        for name in (*components, *varying, *event_columns, *probability_columns)
        + ((censoring,) if censoring is not None else ())
        for t in nodes
    }
    clash = sorted(set(generated) & set(columns))
    if clash:
        raise DataError(
            f"column {clash[0]!r} is also the generated name of {generated[clash[0]]!r} at node "
            f"{clash[0].rsplit(SEPARATOR, 1)[1]}. Rename it"
        )
    clash = sorted(set(generated) & {*baseline_names, *(n for n in (outcome,) if n)})
    if clash:  # pragma: no cover - covered by the frame-column check above
        raise DataError(f"column {clash[0]!r} collides with a generated name")

    # ---------------------------------------------------- leaving the study
    censor = None if censoring is None else column_array(frame, censoring)
    events = [column_array(frame, name) for name in event_columns]
    after = np.zeros(n, dtype=np.int64) + n_times  # first node after which a unit has left
    if censor is not None:
        left = (censor == 0.0) & np.isfinite(censor)
        for name, values in zip(event_columns, events, strict=True):
            stray = left & np.isfinite(values)
            if np.any(stray):
                row = int(np.flatnonzero(stray)[0])
                raise DataError(
                    f"column {name!r} holds a value on {int(stray.sum())} row(s) whose "
                    f"censoring {censoring!r} is 0; the first is unit "
                    f"{panel.unit(panel.unit_code[row])!r} at "
                    f"{panel.time(int(panel.node[row]))!r}. The order at a node is A_t, C_t, "
                    "Y_t, so the event is unobserved once the unit is censored. Record an event "
                    "as censoring 1 and event 1, and a loss after an event-free interval as "
                    "censoring 0 with the event missing"
                )
        rows = np.flatnonzero(left)
        np.minimum.at(after, panel.unit_code[rows], panel.node[rows] + 1)
    for values in events:
        rows = np.flatnonzero(values == 1.0)
        np.minimum.at(after, panel.unit_code[rows], panel.node[rows] + 1)
    beyond = panel.last_node + 1 > after
    if np.any(beyond):
        code = int(np.flatnonzero(beyond)[0])
        raise DataError(
            f"{int(beyond.sum())} unit(s) have rows after they left the study; the first is "
            f"unit {panel.unit(code)!r}, which left at {panel.time(int(after[code]) - 1)!r} "
            f"and has a row at {panel.time(int(panel.last_node[code]))!r}. A unit that is "
            "censored or has the event leaves for good"
        )
    short = (panel.last_node + 1 < n_times) & (after == n_times)
    if np.any(short):
        code = int(np.flatnonzero(short)[0])
        instruction = (
            f"Its censoring {censoring!r} must be 0 on its last row"
            if censoring is not None
            else "Declare censoring= with 0 on the last row of a unit that left"
        )
        stop = panel.time(int(panel.last_node[code]))
        raise DataError(
            f"{int(short.sum())} unit(s) have no row after {stop!r} but neither left nor had "
            f"the event there; the first is unit {panel.unit(code)!r}. {instruction}"
        )

    # ---------------------------------------------------------- unit level
    payload: dict[str, Any] = {}
    for name in baseline_names:
        payload[name] = _unit_level(
            panel, column_array(frame, name), name, role="baseline", first_row=True
        )
    cluster_labels: tuple[Any, ...] = ()
    if cluster is not None:
        labels = _unit_level(
            panel,
            column_array(frame, cluster, dtype=object),
            cluster,
            role="cluster",
            first_row=True,
        )
        if not np.all(_present(labels)):
            raise DataError(f"cluster column {cluster!r} is missing for some units")
        uniques, codes = _sorted_unique(labels, cluster)
        cluster_labels = tuple(_label(value) for value in uniques)
        payload[cluster] = codes
    if weights is not None:
        payload[weights] = _unit_level(
            panel, column_array(frame, weights), weights, role="weights", first_row=True
        )
    y = None
    if outcome is not None:
        y = _unit_level(
            panel, column_array(frame, outcome), outcome, role="outcome", first_row=False
        )
        payload[outcome] = y

    # ---------------------------------------------------------- per node
    treatment_values = [column_array(frame, name, dtype=object) for name in components]
    if continuous_treatment:
        if len(components) > 1:
            raise DataError(
                "continuous_treatment=True needs one treatment column; a vector node is categorical"
            )
        treatment_values = [column_array(frame, components[0])]
    blocks = {
        name: panel.scatter(values)
        for name, values in ((name, column_array(frame, name)) for name in varying)
    }
    reachable = np.zeros((n, n_times), dtype=bool)
    reachable[panel.unit_code, panel.node] = True
    omitted: list[tuple[str, int]] = []
    kept: dict[str, list[int]] = {}
    for name, matrix in blocks.items():
        recorded = np.isfinite(matrix) & reachable
        any_value = recorded.any(axis=0)
        if not any_value.any():
            raise DataError(f"time-varying column {name!r} holds no value at any node")
        first = int(np.argmax(any_value))
        later = np.flatnonzero(~any_value[first:]) + first
        if later.size:
            raise DataError(
                f"time-varying column {name!r} has no value at decision time "
                f"{panel.time(int(later[0]))!r} after it was first recorded at "
                f"{panel.time(first)!r}. Leaving it out there would remove a confounder from "
                "that node's adjustment set; record it, or drop the column"
            )
        omitted += [(name, node + 1) for node in range(first)]
        kept[name] = list(range(first, n_times))
    _refuse_outcome_copies(
        panel, frame, baseline_names, varying, y, outcome, events, event_columns, reachable
    )
    if n_times > 1:
        _refuse_held_treatment(panel, treatment_values, components)

    for t in nodes:
        for name in varying:
            if t - 1 in kept[name]:
                payload[node_name(name, t)] = blocks[name][:, t - 1]
        for name, values in zip(components, treatment_values, strict=True):
            scattered = panel.scatter(values, dtype=float if continuous_treatment else object)
            payload[node_name(name, t)] = scattered[:, t - 1]
        if censor is not None:
            payload[node_name(str(censoring), t)] = panel.scatter(censor)[:, t - 1]
        for name, values in zip(event_columns, events, strict=True):
            payload[node_name(name, t)] = panel.scatter(values)[:, t - 1]
        for name in probability_columns:
            payload[node_name(name, t)] = panel.scatter(column_array(frame, name))[:, t - 1]

    wide = frame_from_dict(payload, backend=backend_of(frame))
    wide_treatment: list[Any] = [
        node_name(components[0], t)
        if len(components) == 1
        else [node_name(name, t) for name in components]
        for t in nodes
    ]
    wide_outcome: Any
    if outcome is not None:
        wide_outcome = outcome
    elif causes:
        wide_outcome = {
            label: [node_name(name, t) for t in nodes]
            for label, name in zip(causes, event_columns, strict=True)
        }
    else:
        wide_outcome = [node_name(event_columns[0], t) for t in nodes]
    node_labels = ["+".join(node_name(name, t) for name in components) for t in nodes]
    built = cls.from_frame(
        wide,
        outcome=wide_outcome,
        treatment=wide_treatment,
        baseline=list(baseline_names),
        time_varying=[
            [node_name(name, t) for name in varying if t - 1 in kept[name]] for t in nodes
        ],
        censoring=None if censoring is None else [node_name(censoring, t) for t in nodes],
        id=cluster,
        weights=weights,
        weights_type=weights_type,
        weights_estimated=weights_estimated,
        family=family,
        continuous_treatment=(
            [node_name(components[0], t) for t in nodes] if continuous_treatment else ()
        ),
        treatment_probabilities=(
            None
            if treatment_probabilities is None
            else {
                label: {
                    level: node_name(column, t) for level, column in treatment_probabilities.items()
                }
                for t, label in zip(nodes, node_labels, strict=True)
            }
        ),
        censoring_probabilities=(
            None
            if censoring_probabilities is None
            else {
                node_name(str(censoring), t): node_name(str(censoring_probabilities), t)
                for t in nodes
            }
        ),
    )
    layout = PanelLayout(
        unit=unit,
        time=time,
        decision_times=panel.grid,
        treatment=components,
        baseline=baseline_names,
        time_varying=varying,
        censoring=censoring,
        outcome=outcome,
        event=event_columns,
        causes=causes,
        cluster=cluster,
        cluster_labels=cluster_labels,
        weights=weights,
        omitted=tuple(omitted),
        treatment_probabilities=(
            None
            if treatment_probabilities is None
            else tuple((level, str(column)) for level, column in treatment_probabilities.items())
        ),
        censoring_probabilities=(
            None if censoring_probabilities is None else str(censoring_probabilities)
        ),
    )
    return replace(built, panel=layout, unit_ids=panel.unit_labels)


def _refuse_outcome_copies(
    panel: _Panel,
    frame: Any,
    baseline: Sequence[str],
    varying: Sequence[str],
    y: np.ndarray | None,
    outcome: str | None,
    events: Sequence[np.ndarray],
    event_columns: Sequence[str],
    reachable: np.ndarray,
) -> None:
    """Refuse a covariate that holds the outcome, or the row's event, under another name.

    Compared on the units or rows where the outcome is recorded, and only where the outcome
    varies there, so a constant covariate is never mistaken for a copy of a constant outcome.
    """
    if y is not None:
        # Compared on every row where both the covariate and its unit's outcome are
        # recorded, so a copy kept on the last row only, or on the first, is caught too.
        target = y[panel.unit_code]
        # A status column can hold interim values at earlier nodes and the outcome at a
        # later one, so a time-varying column is also compared one node at a time.
        for name in (*baseline, *varying):
            values = column_array(frame, name)
            recorded = np.isfinite(values) & np.isfinite(target)
            subsets = [("", recorded)]
            if name in varying:
                subsets += [
                    (f" at decision time {panel.time(node)!r}", recorded & (panel.node == node))
                    for node in range(panel.n_times)
                ]
            for where, rows in subsets:
                if np.unique(target[rows]).size > 1 and np.array_equal(values[rows], target[rows]):
                    raise DataError(
                        f"column {name!r} equals the outcome {outcome!r}{where} on every row "
                        "where both are recorded. It is the outcome under another name, and a "
                        "rule could read it. Remove it from the covariates"
                    )
    for event_name, values in zip(event_columns, events, strict=True):
        rows = np.isfinite(values)
        if rows.sum() < 2 or np.unique(values[rows]).size < 2:
            continue
        for name in varying:
            covariate = column_array(frame, name)
            if np.array_equal(covariate[rows], values[rows]):
                raise DataError(
                    f"column {name!r} equals the event {event_name!r} on every row where the "
                    "event is recorded. A covariate of a row is measured before its decision, "
                    "and the event after it; remove it from the covariates"
                )


def _refuse_held_treatment(
    panel: _Panel, values: Sequence[np.ndarray], components: Sequence[str]
) -> None:
    """Refuse a treatment that never changes within a unit: that is one held decision."""
    rows_per_unit = np.bincount(panel.unit_code, minlength=panel.n)
    if not np.any(rows_per_unit > 1):
        return
    first = panel.first_row[panel.unit_code]
    for column in values:
        array = np.asarray(column, dtype=object)
        same = np.fromiter(
            (_same(array[row], array[first[row]]) for row in range(array.size)),
            dtype=bool,
            count=array.size,
        )
        if not np.all(same):
            return
    raise DataError(
        f"treatment {list(components)} never changes within a unit, so it is one baseline "
        "decision held over every node rather than a decision at each time. Build the wide "
        "frame with one row per unit and declare the held design, treatment='A' in "
        "LongitudinalData.from_frame or LongitudinalTreatment"
    )


# ------------------------------------------------------------------ writing


def to_long_panel(data: LongitudinalData) -> Any:
    """The body of :meth:`LongitudinalData.to_long_panel`, which documents it."""
    layout = data.panel
    if layout is None or data.unit_ids is None:
        raise DataError(
            "to_long_panel needs a container built by LongitudinalData.from_long_panel; this "
            "one was built from a wide frame"
        )
    n_times = data.n_times
    units: list[int] = []
    nodes: list[int] = []
    for time in range(1, n_times + 1):
        rows = np.flatnonzero(data.uncensored_through(time - 1) & data.event_free_through(time - 1))
        units.extend(rows.tolist())
        nodes.extend([time] * rows.size)
    order = np.lexsort((np.asarray(nodes), np.asarray(units)))
    unit_rows = np.asarray(units, dtype=np.int64)[order]
    node_index = np.asarray(nodes, dtype=np.int64)[order]
    out: dict[str, Any] = {
        layout.unit: np.asarray([_label(v) for v in data.unit_ids[unit_rows]], dtype=object)
        if np.asarray(data.unit_ids).dtype == object
        else np.asarray(data.unit_ids)[unit_rows],
        layout.time: _column_of([layout.decision_times[t - 1] for t in node_index.tolist()]),
    }
    for name in layout.baseline:
        if name in data.baseline_names:
            out[name] = data.baseline[unit_rows, data.baseline_names.index(name)]
        else:
            out[name] = np.full(unit_rows.size, np.nan)
    for name in layout.time_varying:
        values = np.full(unit_rows.size, np.nan)
        for t in range(1, n_times + 1):
            wide = node_name(name, t)
            names = data.time_varying_names[t - 1]
            if wide in names:
                at = node_index == t
                values[at] = data.time_varying[t - 1][unit_rows[at], names.index(wide)]
        out[name] = values
    labels = [np.asarray(data.observed_labels(t), dtype=object) for t in range(1, n_times + 1)]
    for position, name in enumerate(layout.treatment):
        values = np.empty(unit_rows.size, dtype=object)
        for index, (row, t) in enumerate(zip(unit_rows.tolist(), node_index.tolist(), strict=True)):
            label = labels[t - 1][row]
            values[index] = label[position] if len(layout.treatment) > 1 else label
        out[name] = _column_of(list(values))
    if layout.censoring is not None:
        out[layout.censoring] = data.uncensored[unit_rows, node_index - 1].astype(float)
    if layout.outcome is not None:
        out[layout.outcome] = data.outcome[unit_rows]
    elif data.event is not None:
        cumulative = data.event if data.cause_event is None else None
        for cause, name in enumerate(layout.event):
            by_node = data.cause_event[:, :, cause] if data.cause_event is not None else cumulative
            assert by_node is not None
            before = np.column_stack([np.zeros(data.n, dtype=bool), by_node[:, :-1]])
            fired = (by_node & ~before)[unit_rows, node_index - 1].astype(float)
            stayed = data.uncensored[unit_rows, node_index - 1]
            out[name] = np.where(stayed, fired, np.nan)
    if layout.cluster is not None and data.cluster is not None:
        out[layout.cluster] = _column_of(
            [layout.cluster_labels[code] for code in data.cluster[unit_rows].tolist()]
        )
    if layout.weights is not None:
        out[layout.weights] = data.weights[unit_rows]
    return data.frame_like(out)


def _column_of(values: list[Any]) -> np.ndarray:
    """A list of labels as a numpy column: numeric when every label is a number."""
    if all(isinstance(v, (bool, np.bool_)) for v in values):
        return np.asarray(values, dtype=bool)
    if all(
        isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
        for v in values
    ):
        return np.asarray(values)
    out = np.empty(len(values), dtype=object)
    out[:] = values
    return out
