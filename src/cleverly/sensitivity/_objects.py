"""The result objects of the sensitivity analyses that used to return a bare dict or frame.

``robustness_value()`` returns :class:`RobustnessValue`, a ``dict`` with the keys it always
had. ``contour_data()`` and ``missingness_tilt()`` return :class:`SensitivityContour` and
:class:`MissingnessTilt`, which hold their rows as Python values and give the frame back
through ``to_frame()``. Each adds a printed ``summary()``, a one-line repr and the notebook
display, so a reader sees what the numbers answer rather than bare columns.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

import numpy as np

from ..utils.frames import emit_frame

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .._summary import Detail, Summary
    from ..validation._interpretation import Resolution
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
        """Index the frame :meth:`to_frame` returns, such as ``tilt["psi"]``."""
        return self.to_frame()[key]

    def __len__(self) -> int:
        return len(self.rows[0][1]) if self.rows else 0

    def __eq__(self, other: object) -> bool:
        """Equal when the type and every field match, a NaN matching a NaN.

        A frame that sweeps two spread contracts holds NaN where a row has no value, and
        ``nan != nan``, so the generated comparison would call such an object unequal to its
        own pickle.
        """
        if type(other) is not type(self):
            return NotImplemented
        return self._canonical() == other._canonical()  # type: ignore[attr-defined]

    def __hash__(self) -> int:
        return hash(self._canonical())

    def _canonical(self) -> tuple[Any, ...]:
        """Every field, with each NaN replaced by one marker that compares equal to itself."""
        return tuple(
            _without_nan(getattr(self, item.name)) for item in fields(self) if item.compare
        )

    def _repr_html_(self) -> str:
        return self.summary().as_html()  # type: ignore[attr-defined]


#: The marker :meth:`_Rows._canonical` puts in place of a NaN.
_NAN_MARKER = ("nan",)


def _without_nan(value: Any) -> Any:
    """``value`` with every NaN float, at any depth of tuples and lists, made comparable."""
    if isinstance(value, float) and value != value:
        return _NAN_MARKER
    if isinstance(value, (tuple, list)):
        return tuple(_without_nan(item) for item in value)
    return value


@dataclass(frozen=True, repr=False, eq=False)
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
    scale : {"difference", "level"}
        The scale of the estimate, as the fit reports it.  A difference has a no-effect value
        of zero, and a level (a counterfactual mean) has none.
    max_bias : float
        The bias scale ``sqrt(sigma^2 nu^2)`` every cell multiplies, the median over the
        fit's draws.
    nu2 : float
        The Riesz second moment ``nu^2`` behind it, the median over the fit's draws.
    nu2_estimator : {"doubly_robust", "plugin"}
        The estimator of ``nu^2`` the cells use.
    n_repeats : int
        The fit's cross-fitting draws.  Each cell is the median over them.
    assumption : str or None
        The sentence a point bound under ``assume_consistent_mechanism=True`` carries,
        else ``None``.
    """

    estimand: str = ""
    psi: float = float("nan")
    bound: str = "lower"
    rho: float = 1.0
    scale: str = "level"
    max_bias: float = float("nan")
    nu2: float = float("nan")
    nu2_estimator: str = "doubly_robust"
    n_repeats: int = 1
    assumption: str | None = None

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
        scale: str,
        max_bias: float,
        nu2: float,
        nu2_estimator: str,
        n_repeats: int,
        assumption: str | None,
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
        scale : str
            The scale of the estimate.
        max_bias : float
            The bias scale every cell multiplies.
        nu2 : float
            The Riesz second moment behind it.
        nu2_estimator : str
            The estimator of ``nu^2``.
        n_repeats : int
            The fit's cross-fitting draws.
        assumption : str or None
            The consistent-mechanism sentence, or ``None``.

        Returns
        -------
        SensitivityContour
            The grid.
        """
        return cls(
            _columns(payload),
            backend,
            estimand,
            float(psi),
            bound,
            float(rho),
            str(scale),
            float(max_bias),
            float(nu2),
            str(nu2_estimator),
            int(n_repeats),
            assumption,
        )

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

    def plot(self, **kwargs: Any) -> Any:
        """Plot the bias-adjusted bound over the two confounder strengths.

        Parameters
        ----------
        **kwargs
            Passed to :func:`cleverly.plotting.confounding`: ``robustness=``, ``benchmarks=``,
            ``multiples=``, ``null=``, ``ax=`` and the formatting keywords ``title=``,
            ``subtitle=``, ``xlabel=``, ``ylabel=``, ``legend=`` and ``figsize=``.

        Returns
        -------
        Figure
            A filled contour of the cells with the no-effect contour in bold.

        Examples
        --------
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> import matplotlib.pyplot as plt
        >>> from sklearn.linear_model import LinearRegression, LogisticRegression
        >>> from cleverly import ATE, CausalStudy, PointTreatment
        >>> from cleverly.datasets import make_linear_ate
        >>> from cleverly.sensitivity import contour_data, robustness_value
        >>> frame, _ = make_linear_ate(n=200, seed=0)
        >>> design = PointTreatment(
        ...     outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4")
        ... )
        >>> result = CausalStudy(frame, design=design).identify(ATE()).estimate(
        ...     outcome_learner=LinearRegression(),
        ...     treatment_learner=LogisticRegression(max_iter=1000),
        ...     cross_fit=False,
        ...     random_state=0,
        ... )
        >>> contour = contour_data(result, "ate", grid_size=8)
        >>> fig = contour.plot(robustness=robustness_value(result, "ate"))
        >>> fig.axes[-1].get_ylabel()
        'Bias-adjusted lower bound, no sampling error'
        >>> plt.close(fig)
        """
        from ..plotting import confounding

        return confounding(self, **kwargs)

    def __repr__(self) -> str:
        from ._summaries import contour_repr

        return contour_repr(self)


@dataclass(frozen=True, repr=False, eq=False)
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
    scales : tuple of tuple of str
        Each estimand's scale (``"difference"``, ``"ratio"``, ``"level"`` or
        ``"fraction"``), as ``(estimand, scale)`` pairs, which sets its no-effect value.
    readings : tuple of tuple
        For each estimand whose interval off ``gamma = 0`` comes from the targeted tilt, the
        registry's reading of that interval, as ``(estimand, Resolution)`` pairs.

    Attributes
    ----------
    descriptive : tuple of str
    """

    outcome: str = ""
    n_missing: int = 0
    n: int = 0
    arm_gamma: tuple[tuple[Any, float], ...] | None = None
    level: float = 0.95
    statuses: tuple[tuple[str, str], ...] = field(default=())
    ratios: tuple[str, ...] = ()
    scales: tuple[tuple[str, str], ...] = ()
    readings: tuple[tuple[str, Resolution], ...] = field(default=(), compare=False)

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
        scales: tuple[tuple[str, str], ...] = (),
        readings: tuple[tuple[str, Resolution], ...] = (),
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
        scales : tuple of tuple of str
            Each estimand's scale, as ``(estimand, scale)`` pairs.
        readings : tuple of tuple
            The registry's reading of each targeted interval off ``gamma = 0``.

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
            scales,
            readings,
        )

    @property
    def estimands(self) -> tuple[str, ...]:
        """The estimands the tilt reports, in order."""
        return tuple(dict.fromkeys(self.column("estimand")))

    @property
    def descriptive(self) -> tuple[str, ...]:
        """The estimands whose interval off ``gamma = 0`` no validation study licenses."""
        return tuple(name for name, reading in self.readings if not reading.licensed)

    def plot(self, **kwargs: Any) -> Any:
        """Plot one estimate against the tilt γ, with its interval where one is reported.

        Parameters
        ----------
        **kwargs
            Passed to :func:`cleverly.plotting.sensitivity`: ``estimand=``,
            ``outcome_label=``, ``units=`` and ``title=``.

        Returns
        -------
        Figure
            One panel over the grid of γ.

        Examples
        --------
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> import matplotlib.pyplot as plt
        >>> from sklearn.linear_model import LinearRegression, LogisticRegression
        >>> from cleverly import ATE, CausalStudy, PointTreatment
        >>> from cleverly.datasets import make_missing_outcome
        >>> frame, _ = make_missing_outcome(n=300, seed=61)
        >>> design = PointTreatment(
        ...     outcome="Y", treatment="A", adjustment=["W1", "W2", "W3"], missingness="Delta"
        ... )
        >>> result = CausalStudy(frame, design=design).identify(ATE()).estimate(
        ...     outcome_learner=LinearRegression(),
        ...     treatment_learner=LogisticRegression(max_iter=1000),
        ...     cross_fit=False,
        ...     random_state=4,
        ... )
        >>> fig = result.sensitivity.missingness(gamma=[-1.0, 0.0, 1.0]).plot()
        >>> fig.axes[0].get_ylabel()
        'Average effect'
        >>> plt.close(fig)
        """
        from ..plotting import sensitivity

        return sensitivity(self, **kwargs)

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
