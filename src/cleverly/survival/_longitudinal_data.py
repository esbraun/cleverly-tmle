r"""The container of a continuous-time design with a time-varying treatment (X6 part (e)).

The data arrive in the counting-process long format of ``survival::tmerge``: one row per
interval between monitoring events ("visits").  A row starts at a visit, where the time-varying
covariates :math:`L(s)` are measured and then the treatment :math:`A(s)` is decided; both are
constant on ``(start, stop]``.  A unit's first row starts at 0 (the baseline visit), its rows
are contiguous, and only its last row carries an event code (``0`` censored).

Within one time the engine reads death first, then the visit (:math:`L(s)`, then :math:`A(s)`),
then censoring (Rytgaard, Gerds and van der Laan 2022, supplement S3.2, with one marked visit
process in place of their two monitoring processes).  So a last row of zero length with an event
code is a visit and a death at one time: the death comes first, the visit is dropped, and a
:class:`UserWarning` names the count, which :attr:`ContinuousLongitudinalData.dropped_visits`
keeps.  A last row of zero length with code 0 is a visit, then censoring: the visit is kept.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .._typing import FloatArray, IntArray
from .._warnings import warn as _warn
from ..exceptions import DataError

__all__ = ["ContinuousLongitudinalData"]


def dropped_visit_text(count: int) -> str:
    """The warning of a fit whose data hold a visit and a death at one time.

    Parameters
    ----------
    count : int
        The visits dropped.

    Returns
    -------
    str
        The text.
    """
    noun = "units record" if count != 1 else "unit records"
    return (
        f"{count} {noun} a visit and a death at the same time (a last row of zero length with "
        "an event code). Within one time the engine reads death first, so the death's hazard "
        "does not read the visit's covariates, and the visit is dropped."
    )


@dataclass(frozen=True)
class ContinuousLongitudinalData:
    """Units followed in continuous time, with covariates and a treatment that change at visits.

    Parameters
    ----------
    time : ndarray
        ``(n,)`` each unit's last stop: its event or censoring time.
    event : ndarray
        ``(n,)`` each unit's event code, ``0`` for censored.
    baseline : ndarray
        ``(n, p)`` the baseline covariates :math:`W`.
    unit : ndarray
        ``(V,)`` the unit of each visit, visits of one unit contiguous and in time order.
    visit_time : ndarray
        ``(V,)`` each visit's time; a unit's first visit is at 0.
    visit_index : ndarray
        ``(V,)`` the visit count :math:`k`, 0 at baseline.
    covariates : ndarray
        ``(V, q)`` the time-varying covariates :math:`L(s)` measured at each visit.
    treatment : ndarray
        ``(V,)`` the treatment :math:`A(s)` decided at each visit, as given.
    decision : ndarray
        ``(V,)`` whether the treatment may change at the visit; at any other visit it is
        carried forward.  Every baseline visit is a decision visit.
    names : tuple of str
        ``(id, start, stop, event, treatment, *baseline, *time_varying)``.
    n_baseline : int
        The number of baseline covariates.
    causes : dict of int to str or None
        The label of each nonzero event code, as declared.
    backend : str or None
        The dataframe library the frame arrived in.
    ids : ndarray
        ``(n,)`` each unit's identifier, in unit order.
    dropped_visits : int
        The visits read as coming after a death at the same time and dropped.
    """

    time: FloatArray
    event: IntArray
    baseline: FloatArray
    unit: IntArray
    visit_time: FloatArray
    visit_index: IntArray
    covariates: FloatArray
    treatment: np.ndarray
    decision: np.ndarray
    names: tuple[str, ...]
    n_baseline: int
    causes: dict[int, str] | None = None
    backend: str | None = None
    ids: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    dropped_visits: int = 0
    #: Private: one weight per unit, so a test passes a finite law's support points weighted
    #: by their mass (as X70's tests pass law (M)).  No public argument reaches it.
    _weight: FloatArray | None = field(default=None, repr=False)

    @property
    def n(self) -> int:
        """The number of units."""
        return int(self.time.size)

    @property
    def baseline_names(self) -> tuple[str, ...]:
        """The baseline covariate names."""
        return self.names[5 : 5 + self.n_baseline]

    @property
    def time_varying_names(self) -> tuple[str, ...]:
        """The time-varying covariate names."""
        return self.names[5 + self.n_baseline :]

    @property
    def first_visit(self) -> IntArray:
        """``(n,)`` the position of each unit's baseline visit in the visit arrays."""
        return np.flatnonzero(self.visit_index == 0)

    @property
    def previous_treatment(self) -> np.ndarray:
        """``(V,)`` the treatment at the unit's previous visit; at baseline its own value."""
        out = np.array(self.treatment, copy=True)
        later = self.visit_index > 0
        out[later] = self.treatment[np.flatnonzero(later) - 1]
        return out

    @classmethod
    def from_frame(
        cls,
        data: Any,
        *,
        id: str,
        start: str,
        stop: str,
        event: str,
        treatment: str,
        baseline: Sequence[str],
        time_varying: Sequence[str],
        causes: Mapping[int, str] | None = None,
        decision: str | None = None,
    ) -> ContinuousLongitudinalData:
        """Validate a counting-process frame and hold its unit and visit arrays.

        Parameters
        ----------
        data : DataFrame
            A pandas or polars frame, one row per interval between visits.
        id : str
            The unit identifier column.
        start : str
            The interval's start: the visit time.
        stop : str
            The interval's end: the next visit, or the event or censoring time.
        event : str
            The event code, nonzero on a unit's last row only; ``0`` for censored.
        treatment : str
            The treatment decided at the visit.
        baseline : sequence of str
            The baseline covariates, constant within a unit; at least one.
        time_varying : sequence of str
            The covariates measured at each visit; at least one.
        causes : mapping of int to str, or None
            The label of each nonzero event code.
        decision : str or None
            A boolean column marking the visits where the treatment may change.  ``None``
            makes every visit a decision visit.

        Returns
        -------
        ContinuousLongitudinalData
            The container.
        """
        from ..utils.frames import as_frame, backend_of, is_dataframe

        if not is_dataframe(data):
            raise DataError("ContinuousLongitudinalData expects a pandas or polars DataFrame")
        frame = as_frame(data)
        baseline = tuple(str(b) for b in baseline)
        time_varying = tuple(str(c) for c in time_varying)
        names = (str(id), str(start), str(stop), str(event), str(treatment), *baseline)
        names = (*names, *time_varying)
        needed = (*names, *(() if decision is None else (str(decision),)))
        missing = [name for name in needed if name not in frame.columns]
        if missing:
            raise DataError(f"columns {missing} are not in the data")
        if not baseline:
            raise DataError("ContinuousLongitudinalTMLE names at least one baseline covariate")
        if not time_varying:
            raise DataError(
                "ContinuousLongitudinalTMLE names at least one time-varying covariate; with "
                "none, ContinuousSurvivalTMLE fits a baseline treatment."
            )
        ids_raw = frame[names[0]].to_numpy()
        starts = frame[names[1]].to_numpy().astype(float)
        stops = frame[names[2]].to_numpy().astype(float)
        codes = frame[names[3]].to_numpy().astype(float)
        levels = frame[names[4]].to_numpy()
        W = np.column_stack([frame[b].to_numpy().astype(float) for b in baseline])
        L = np.column_stack([frame[c].to_numpy().astype(float) for c in time_varying])
        marks = (
            np.ones(starts.size, dtype=bool)
            if decision is None
            else frame[str(decision)].to_numpy().astype(bool)
        )
        for what, values in (
            ("start", starts),
            ("stop", stops),
            ("event", codes),
            ("baseline covariate", W),
            ("time-varying covariate", L),
        ):
            if not np.all(np.isfinite(values)):
                raise DataError(f"every {what} value is a finite number")
        try:
            numeric = np.asarray(levels, dtype=float)
        except (TypeError, ValueError):
            numeric = None
        if numeric is not None and not np.all(np.isfinite(numeric)):
            raise DataError("every treatment value is a finite number")
        if np.any(codes < 0) or np.any(codes % 1):
            raise DataError("every event code is a non-negative integer, 0 for censored")
        if np.any(stops < starts):
            raise DataError("every row ends at or after its start")
        # Units in order of first appearance, rows of a unit in time order.
        _, first, inverse = np.unique(ids_raw, return_index=True, return_inverse=True)
        inverse = np.asarray(inverse).reshape(-1)
        rank = np.empty(first.size, dtype=int)
        rank[np.argsort(first, kind="stable")] = np.arange(first.size)
        unit_of_row = rank[inverse]
        order = np.lexsort((starts, unit_of_row))
        unit_of_row, starts, stops = unit_of_row[order], starts[order], stops[order]
        codes, levels, W, L, marks = codes[order], levels[order], W[order], L[order], marks[order]
        ids_sorted = np.asarray(ids_raw)[order]
        n = int(first.size)
        boundaries = np.flatnonzero(np.diff(unit_of_row)) + 1
        heads = np.concatenate([[0], boundaries])
        tails = np.concatenate([boundaries - 1, [unit_of_row.size - 1]])
        last = np.zeros(unit_of_row.size, dtype=bool)
        last[tails] = True
        head = np.zeros(unit_of_row.size, dtype=bool)
        head[heads] = True
        if np.any(starts[heads] != 0):
            raise DataError("every unit's first row starts at 0, its baseline visit")
        following = ~head
        gap = starts[following] != stops[np.flatnonzero(following) - 1]
        if np.any(gap):
            raise DataError(
                "the rows of a unit are contiguous: each row starts where the previous one stops, "
                "with no overlap and no gap"
            )
        if np.any((stops == starts) & ~last):
            raise DataError("only a unit's last row may have zero length")
        if np.any((codes != 0) & ~last):
            raise DataError("an event code is nonzero only on a unit's last row")
        if np.any(stops[tails] <= 0):
            raise DataError("every unit is followed for a positive time")
        if np.any(np.ptp(W, axis=0) > 0):
            for column in range(W.shape[1]):
                for u in range(n):
                    rows = slice(heads[u], tails[u] + 1)
                    if np.ptp(W[rows, column]) > 0:
                        raise DataError(
                            f"the baseline covariate {baseline[column]!r} changes within unit "
                            f"{ids_sorted[heads[u]]!r}; declare it under time_varying="
                        )
        changed = following & np.r_[False, levels[1:] != levels[:-1]] & ~marks
        if np.any(changed):
            raise DataError(
                f"the treatment changes at {int(changed.sum())} visits that decision= marks as "
                "not a decision visit; at such a visit it is carried forward"
            )
        marks = marks | head
        # A visit and a death at one time: death first, the visit dropped.
        dropped = last & (stops == starts) & (codes != 0) & ~head
        count = int(dropped.sum())
        if count:
            previous = np.flatnonzero(dropped) - 1
            codes[previous] = codes[dropped]
            keep = ~dropped
            unit_of_row, starts, stops = unit_of_row[keep], starts[keep], stops[keep]
            codes, levels, W, L, marks = codes[keep], levels[keep], W[keep], L[keep], marks[keep]
            ids_sorted = ids_sorted[keep]
            _warn(dropped_visit_text(count), UserWarning)
        boundaries = np.flatnonzero(np.diff(unit_of_row)) + 1
        heads = np.concatenate([[0], boundaries])
        tails = np.concatenate([boundaries - 1, [unit_of_row.size - 1]])
        index = np.arange(unit_of_row.size) - np.repeat(heads, np.diff(np.r_[heads, starts.size]))
        event_codes = codes[tails].astype(int)
        present = sorted({int(c) for c in event_codes if c != 0})
        if not present:
            raise DataError("the event column holds no event")
        if causes is not None and set(present) - set(causes):
            raise DataError(
                f"event codes {sorted(set(present) - set(causes))} are not named by causes"
            )
        return cls(
            time=stops[tails],
            event=event_codes,
            baseline=W[heads],
            unit=unit_of_row.astype(int),
            visit_time=starts,
            visit_index=index.astype(int),
            covariates=L,
            treatment=levels,
            decision=marks,
            names=names,
            n_baseline=len(baseline),
            causes=None if causes is None else {int(k): str(v) for k, v in causes.items()},
            backend=backend_of(frame),
            ids=ids_sorted[heads],
            dropped_visits=count,
        )

    def baseline_view(self) -> Any:
        """The one-row-per-unit container of the baseline treatment and covariates.

        Returns
        -------
        ContinuousSurvivalData
            The unit's last stop and event code, :math:`A(0)` as the treatment, and
            :math:`(W, L(0))` as the baseline covariates.  On data with no visit after baseline
            it is the same design, which the baseline engine
            (:class:`~cleverly.survival.ContinuousSurvivalTMLE`) fits.
        """
        from .data import ContinuousSurvivalData

        first = self.first_visit
        return ContinuousSurvivalData(
            np.array(self.time, copy=True),
            np.array(self.event, copy=True),
            np.array(self.treatment[first], copy=True),
            np.column_stack([self.baseline, self.covariates[first]]),
            (
                self.names[2],
                self.names[3],
                self.names[4],
                *self.baseline_names,
                *self.time_varying_names,
            ),
            None if self.causes is None else dict(self.causes),
            self.backend,
        )


def weighted(data: ContinuousLongitudinalData, weight: Any) -> ContinuousLongitudinalData:
    """``data`` with one weight per unit, the private seam the exact-law tests use.

    Parameters
    ----------
    data : ContinuousLongitudinalData
        The container.
    weight : array-like
        ``(n,)`` positive weights.

    Returns
    -------
    ContinuousLongitudinalData
        A copy carrying the weights.
    """
    from dataclasses import replace

    w = np.asarray(weight, dtype=float)
    if w.shape != (data.n,) or not np.all(w > 0):
        raise ValueError("the private weight seam takes one positive weight per unit")
    return replace(data, _weight=w)
