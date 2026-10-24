"""The result objects of the sensitivity analyses that used to return a bare dict or frame.

``robustness_value()`` returns :class:`RobustnessValue`, a ``dict`` with the keys it always
had. ``contour_data()`` and ``missingness_tilt()`` return :class:`SensitivityContour` and
:class:`MissingnessTilt`, which hold their rows as Python values and give the frame back
through ``to_frame()``. Each adds a printed ``summary()``, a one-line repr and the notebook
display, so a reader sees what the numbers answer rather than bare columns.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..utils.frames import emit_frame

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .._summary import Detail, Summary
    from ._mechanism_stack import MechanismRefusal

__all__ = ["MissingnessTilt", "RobustnessValue", "SensitivityContour"]


def _python(value: Any) -> Any:
    """A numpy scalar as its Python value; everything else unchanged."""
    return value.item() if isinstance(value, np.generic) else value


def _columns(payload: Mapping[str, Sequence[Any]]) -> tuple[tuple[str, tuple[Any, ...]], ...]:
    return tuple(
        (str(name), tuple(_python(item) for item in values)) for name, values in payload.items()
    )


class RobustnessValue(dict[str, Any]):
    """The confounding strength that would move an estimate to its null.

    A ``dict`` with the keys :func:`~cleverly.sensitivity.robustness_value` documents
    (``"rv"``, the limit value under its status name, ``"max_bias"``, and the keys a
    plug-in or descriptive bound adds), so ``value["rv"]`` and ``json.dumps(value)``
    work as on any mapping. :meth:`summary` prints what the numbers mean.

    Parameters
    ----------
    values : mapping of str to Any
        The keys and values, in order.
    estimand : str
        The estimand the values describe.
    psi : float
        Its reported estimate.
    rho : float
        The alignment the values assume.
    level : float
        The coverage level of the confidence limit the limit value reads.
    null_hypothesis : float
        The value the strengths are measured against.
    refusal : MechanismRefusal or None
        Why the limit value is withheld, when an estimator rule withholds it.
    """

    estimand: str
    psi: float
    rho: float
    level: float
    null_hypothesis: float
    refusal: MechanismRefusal | None

    def __init__(
        self,
        values: Mapping[str, Any],
        *,
        estimand: str,
        psi: float,
        rho: float,
        level: float,
        null_hypothesis: float,
        refusal: MechanismRefusal | None = None,
    ) -> None:
        super().__init__(values)
        self.estimand = estimand
        self.psi = float(psi)
        self.rho = float(rho)
        self.level = float(level)
        self.null_hypothesis = float(null_hypothesis)
        self.refusal = refusal

    def __reduce__(self) -> tuple[Any, ...]:
        context = {
            "estimand": self.estimand,
            "psi": self.psi,
            "rho": self.rho,
            "level": self.level,
            "null_hypothesis": self.null_hypothesis,
            "refusal": self.refusal,
        }
        return (_rebuild_robustness, (dict(self), context))

    def summary(self, *, detail: Detail = "short") -> Summary:
        """Return the printed summary.

        Parameters
        ----------
        detail : {"short", "full"}
            ``"full"`` adds the settings and the definitions of the strengths.

        Returns
        -------
        Summary
            A ``str`` that prints the values and how to read them.
        """
        from ._summaries import robustness_summary

        return robustness_summary(self, detail)

    def __repr__(self) -> str:
        from ._summaries import robustness_repr

        return robustness_repr(self)

    def _repr_html_(self) -> str:
        return self.summary().as_html()


def _rebuild_robustness(values: dict[str, Any], context: dict[str, Any]) -> RobustnessValue:
    return RobustnessValue(values, **context)


@dataclass(frozen=True)
class _Rows:
    """Rows kept as Python values and the backend that :meth:`to_frame` emits.

    A stored backend frame would hold the backend's memory for the life of the result and
    would not survive a save, so the rows are kept as column tuples.
    """

    rows: tuple[tuple[str, tuple[Any, ...]], ...]
    backend: str | None

    def to_frame(self) -> Any:
        """Return the rows as a dataframe, in the backend of the fit's data.

        Returns
        -------
        DataFrame
            One column per field.
        """
        return emit_frame(self._payload(), backend=self.backend)

    def _payload(self) -> dict[str, list[Any]]:
        """The columns as lists, which the assessment rows read without a frame."""
        return {name: list(values) for name, values in self.rows}

    @property
    def columns(self) -> tuple[str, ...]:
        """The column names, in order."""
        return tuple(name for name, _ in self.rows)

    def column(self, name: str) -> tuple[Any, ...]:
        """One column's values, top to bottom.

        Parameters
        ----------
        name : str
            A column name.

        Returns
        -------
        tuple
            The values.
        """
        return dict(self.rows)[name]

    def __getitem__(self, key: Any) -> Any:
        """Index the frame :meth:`to_frame` returns, as the old frame return allowed."""
        return self.to_frame()[key]

    def __len__(self) -> int:
        return len(self.rows[0][1]) if self.rows else 0

    def _repr_html_(self) -> str:
        return self.summary().as_html()  # type: ignore[attr-defined]


@dataclass(frozen=True, repr=False)
class SensitivityContour(_Rows):
    """One end of the bias-adjusted range over a grid of confounder strengths.

    Returned by :func:`~cleverly.sensitivity.contour_data`. :meth:`to_frame` gives the
    long frame of ``cf_d``, ``cf_y`` and ``value`` for a contour plot.

    Parameters
    ----------
    rows : tuple
        The ``cf_d``, ``cf_y`` and ``value`` columns, as ``(name, values)`` pairs.
    backend : str or None
        The dataframe backend :meth:`to_frame` emits.
    estimand : str
        The estimand the grid bounds.
    psi : float
        Its reported estimate.
    bound : {"lower", "upper"}
        Which end of the range each cell holds.
    rho : float
        The alignment every cell assumes.
    """

    estimand: str = ""
    psi: float = float("nan")
    bound: str = "lower"
    rho: float = 1.0

    @classmethod
    def build(
        cls,
        payload: Mapping[str, Sequence[Any]],
        *,
        backend: str | None,
        estimand: str,
        psi: float,
        bound: str,
        rho: float,
    ) -> SensitivityContour:
        """Build the grid from its columns.

        Parameters
        ----------
        payload : mapping of str to sequence
            The ``cf_d``, ``cf_y`` and ``value`` columns.
        backend : str or None
            The backend :meth:`to_frame` emits.
        estimand : str
            The estimand the grid bounds.
        psi : float
            Its reported estimate.
        bound : {"lower", "upper"}
            Which end of the range each cell holds.
        rho : float
            The alignment every cell assumes.

        Returns
        -------
        SensitivityContour
            The grid.
        """
        return cls(_columns(payload), backend, estimand, float(psi), bound, float(rho))

    def summary(self, *, detail: Detail = "short") -> Summary:
        """Return the printed summary.

        Parameters
        ----------
        detail : {"short", "full"}
            ``"full"`` adds the definitions of the strengths.

        Returns
        -------
        Summary
            A ``str`` that prints part of the grid and how to read it.
        """
        from ._summaries import contour_summary

        return contour_summary(self, detail)

    def __repr__(self) -> str:
        from ._summaries import contour_repr

        return contour_repr(self)


@dataclass(frozen=True, repr=False)
class MissingnessTilt(_Rows):
    """Estimates under departures from outcomes missing at random.

    Returned by :func:`~cleverly.sensitivity.missingness_tilt`. :meth:`to_frame` gives
    one row per ``(gamma, estimand)``, with the spread columns the estimate's inference
    status names and one ``gamma[<level>]`` column per arm.

    Parameters
    ----------
    rows : tuple
        The frame's columns, as ``(name, values)`` pairs.
    backend : str or None
        The dataframe backend :meth:`to_frame` emits.
    outcome : str
        The outcome column.
    n_missing : int
        Rows whose outcome is missing.
    n : int
        Rows in the fit.
    arm_gamma : tuple of tuple or None
        The declared ``(arm, multiplier)`` pairs, or ``None`` when every arm is tilted
        alike.
    level : float
        The coverage level of the intervals.
    statuses : tuple of tuple of str
        Each estimand's inference status, as ``(estimand, status)`` pairs.
    ratios : tuple of str
        The estimands on a ratio scale, whose spread is on the log scale.
    """

    outcome: str = ""
    n_missing: int = 0
    n: int = 0
    arm_gamma: tuple[tuple[Any, float], ...] | None = None
    level: float = 0.95
    statuses: tuple[tuple[str, str], ...] = field(default=())
    ratios: tuple[str, ...] = ()

    @classmethod
    def build(
        cls,
        payload: Mapping[str, Sequence[Any]],
        *,
        backend: str | None,
        outcome: str,
        n_missing: int,
        n: int,
        arm_gamma: tuple[tuple[Any, float], ...] | None,
        level: float,
        statuses: tuple[tuple[str, str], ...],
        ratios: tuple[str, ...] = (),
    ) -> MissingnessTilt:
        """Build the tilt from its columns and the fit facts its summary prints.

        Parameters
        ----------
        payload : mapping of str to sequence
            The frame's columns.
        backend : str or None
            The backend :meth:`to_frame` emits.
        outcome : str
            The outcome column.
        n_missing : int
            Rows whose outcome is missing.
        n : int
            Rows in the fit.
        arm_gamma : tuple of tuple or None
            The declared ``(arm, multiplier)`` pairs, or ``None``.
        level : float
            The coverage level of the intervals.
        statuses : tuple of tuple of str
            Each estimand's inference status.
        ratios : tuple of str
            The estimands on a ratio scale.

        Returns
        -------
        MissingnessTilt
            The tilt.
        """
        return cls(
            _columns(payload),
            backend,
            outcome,
            int(n_missing),
            int(n),
            arm_gamma,
            float(level),
            statuses,
            ratios,
        )

    @property
    def estimands(self) -> tuple[str, ...]:
        """The estimands the tilt reports, in order."""
        return tuple(dict.fromkeys(self.column("estimand")))

    def summary(self, *, detail: Detail = "short") -> Summary:
        """Return the printed summary.

        Parameters
        ----------
        detail : {"short", "full"}
            ``"full"`` adds the tilt formula and the per-arm tilts.

        Returns
        -------
        Summary
            A ``str`` with one table per estimand and how to read it.
        """
        from ._summaries import tilt_summary

        return tilt_summary(self, detail)

    def __repr__(self) -> str:
        from ._summaries import tilt_repr

        return tilt_repr(self)
