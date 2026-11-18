"""The diagnostics of a continuous-time survival fit.

:class:`~cleverly.assessment.DiagnosticsFacade` returns these reports from
``result.diagnostics.support()``, ``score_equations()`` and ``nuisance_models()`` on a
:class:`~cleverly.survival.ContinuousSurvivalResult`. Each one reads
:class:`~cleverly.survival.ContinuousSurvivalRecord` and refits nothing.
``result.diagnostics.truncation_curve()`` returns :func:`continuous_truncation_curve`, which
refits the initial hazards from the stored estimator and data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Final

import numpy as np

from .._refusal import Refusal, RefusalReason
from .._summary import vocabulary
from ..exceptions import CapabilityError
from .result import level_name

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .result import ContinuousSurvivalResult

__all__ = [
    "ContinuousSurvivalNuisances",
    "ContinuousSurvivalScores",
    "ContinuousSurvivalSupport",
    "NuisanceSelection",
    "continuous_truncation_curve",
    "nuisance_report",
    "score_report",
    "support_report",
]

#: The quantiles each probability is summarized at, as the point support report reads them.
QUANTILES: tuple[float, ...] = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
#: The thresholds of the propensity tail shares, as the point support report reads them.
THRESHOLDS: tuple[float, ...] = (0.01, 0.025, 0.05, 0.1)


@dataclass(frozen=True)
class ContinuousSurvivalSupport:
    """Overlap of the treatment and censoring fits of a continuous-time survival fit.

    The clever covariate of arm :math:`a` divides by
    :math:`\\hat\\pi(a \\mid L)\\hat S_c(u^- \\mid a, L)` at every time :math:`u` up to the
    target, bounded below by :attr:`weight_floor`. The report summarizes both factors as
    fitted and the share of cells the floor raised.

    Parameters
    ----------
    n : int
        The number of units.
    arms : tuple of str
        ``(reference, treated)``.
    horizons : tuple of float
        The target times.
    propensity_quantiles : dict of str to dict of float to float
        Quantiles of the estimated probability of the treated arm,
        :math:`\\hat\\pi(1 \\mid L)`, over every row (``"overall"``) and within each arm,
        keyed by the arm label.
    tail_mass : dict of float to dict of str to float
        The share of rows with :math:`\\hat\\pi(1 \\mid L)` below each threshold
        (``"below"``) and above one minus it (``"above"``).
    effective_sample_size : dict of str to dict of str to float
        For each arm label, its row count (``"n"``), the Kish effective sample size of the
        weights :math:`1 / \\hat\\pi(a \\mid L)` over its rows (``"effective"``), and their
        ratio (``"ratio"``).
    censoring_quantiles : dict of float to dict of float to float
        For each target time, quantiles of each row's estimated probability of remaining
        uncensored just before it under its received arm, :math:`\\hat S_c(t_k^- \\mid A, L)`.
    weight_floor : float
        The lower bound the targeting weights applied to
        :math:`\\hat\\pi(a \\mid L)\\hat S_c(u^- \\mid a, L)`.
    bounded_share : float
        The share of (unit, time) cells of each unit's received arm, while it is followed,
        that the floor raised.
    last_time_floor_share : float
        The share of (unit, arm) pairs, over both arms, below the floor at the last target
        time: the counterfactual reach of the floor. Reported and not graded.
    frozen_share : float
        The share of (unit, arm) pairs the walk held for a step to keep a survival above its
        floor.
    cross_fitted : bool
        Whether the probabilities were predicted out of fold.
    """

    n: int
    arms: tuple[str, str]
    horizons: tuple[float, ...]
    propensity_quantiles: dict[str, dict[float, float]]
    tail_mass: dict[float, dict[str, float]]
    effective_sample_size: dict[str, dict[str, float]]
    censoring_quantiles: dict[float, dict[float, float]]
    weight_floor: float
    bounded_share: float
    last_time_floor_share: float
    frozen_share: float
    cross_fitted: bool

    @property
    def severity(self) -> str:
        """``"adequate"``, ``"strain"`` or ``"serious"`` by :attr:`bounded_share`.

        The tiers are the point support report's truncation tiers
        (:data:`~cleverly.sensitivity.positivity.STRAIN_SHARE` and
        :data:`~cleverly.sensitivity.positivity.SERIOUS_SHARE`), read on the share of cells
        the floor raised. No effective-sample-size ratio is graded.
        """
        from ..sensitivity.positivity import SERIOUS_SHARE, STRAIN_SHARE

        if self.bounded_share > SERIOUS_SHARE:
            return "serious"
        if self.bounded_share > STRAIN_SHARE:
            return "strain"
        return "adequate"


@dataclass(frozen=True)
class ContinuousSurvivalScores:
    """The score equations of the targeting walk, as it ended.

    The walk stops when every target's :math:`|P_n D^*| / \\hat\\sigma` is at or below
    :attr:`cutoff`, the rule of Rytgaard and van der Laan (2024), Section 4.3.

    Parameters
    ----------
    names : tuple of str
        Each targeted risk, in the walk's order.
    standardized_score : tuple of float
        Each target's final :math:`|P_n D^*| / \\hat\\sigma`.
    cutoff : float
        The stopping bound.
    converged : bool
        Whether the walk reached the cutoff.
    steps : int
        Accepted steps.
    halvings : int
        Rejected steps.
    off_grid_share : float
        The share of random times in the window, among those where a score is not
        identically zero, at which the score equation holds to the cutoff.
    """

    names: tuple[str, ...]
    standardized_score: tuple[float, ...]
    cutoff: float
    converged: bool
    steps: int
    halvings: int
    off_grid_share: float

    @property
    def passed(self) -> bool:
        """Whether every standardized score is at or below the cutoff."""
        return bool(all(score <= self.cutoff for score in self.standardized_score))

    @property
    def worst_ratio(self) -> float:
        """The largest standardized score as a multiple of the cutoff."""
        return float(max(self.standardized_score) / self.cutoff)


@dataclass(frozen=True)
class NuisanceSelection:
    """One nuisance model's cross-validated selection.

    Parameters
    ----------
    role : str
        What the model estimates, such as ``"hazard of cause death"``.
    members : tuple of str
        The candidates.
    losses : tuple of float
        Each candidate's cross-validated loss: the negative log partial likelihood for a
        hazard and the log loss for the treatment model. ``nan`` for a one-member library,
        which is not cross-validated, and empty for a supplied treatment learner.
    chosen : str
        The selected candidate.
    """

    role: str
    members: tuple[str, ...]
    losses: tuple[float, ...]
    chosen: str


@dataclass(frozen=True)
class ContinuousSurvivalNuisances:
    """The hazard, censoring and treatment selections of a continuous-time survival fit.

    Parameters
    ----------
    rows : tuple of NuisanceSelection
        Each cause's hazard, the censoring hazard and the treatment model. A cross-fitted
        fit records the first fold's selections.
    cross_fitted : bool
        Whether the nuisances were fitted out of fold.
    n_folds : int
        The number of cross-fitting folds, ``1`` for a full-sample fit.
    grid_size : int
        The number of distinct observed times up to the last target.
    """

    rows: tuple[NuisanceSelection, ...]
    cross_fitted: bool
    n_folds: int
    grid_size: int

    #: The battery's row label: this fit has no outcome regression to name.
    check_label: ClassVar[str] = vocabulary.CONTINUOUS_SURVIVAL_NUISANCE_NAME


def _quantiles(values: Any) -> dict[float, float]:
    return {q: float(np.quantile(values, q)) for q in QUANTILES}


def _kish(weights: Any) -> float:
    from ..sensitivity.positivity import _kish_ess

    return _kish_ess(weights)


def support_report(result: ContinuousSurvivalResult) -> ContinuousSurvivalSupport:
    """The overlap of the treatment and censoring fits, read off the record.

    Parameters
    ----------
    result : ContinuousSurvivalResult
        A continuous-time survival fit.

    Returns
    -------
    ContinuousSurvivalSupport
        The report.
    """
    record = result.record
    p1 = np.asarray(record.p1, dtype=float)
    arm = np.asarray(record.arm)
    # Each row's censoring survival just before each target time, under its received arm.
    received_retention = np.asarray(record.retention_before)[arm, np.arange(arm.size)]
    received = np.where(arm == 1, p1, 1.0 - p1)
    labels = result.arms
    quantiles = {"overall": _quantiles(p1)}
    ess: dict[str, dict[str, float]] = {}
    for code, label in enumerate(labels):
        rows = arm == code
        quantiles[label] = _quantiles(p1[rows])
        effective = _kish(1.0 / received[rows])
        count = float(rows.sum())
        ess[label] = {"n": count, "effective": effective, "ratio": effective / count}
    return ContinuousSurvivalSupport(
        n=result.n,
        arms=labels,
        horizons=result.horizons,
        propensity_quantiles=quantiles,
        tail_mass={
            t: {"below": float(np.mean(p1 < t)), "above": float(np.mean(p1 > 1.0 - t))}
            for t in THRESHOLDS
        },
        effective_sample_size=ess,
        censoring_quantiles={
            time: _quantiles(received_retention[:, k]) for k, time in enumerate(result.horizons)
        },
        weight_floor=float(record.weight_floor),
        bounded_share=float(record.bounded_share),
        last_time_floor_share=float(record.last_time_floor_share),
        frozen_share=float(record.frozen_share),
        cross_fitted=bool(record.cross_fitted),
    )


def score_report(result: ContinuousSurvivalResult) -> ContinuousSurvivalScores:
    """The standardized scores of the targeting walk against its cutoff.

    Parameters
    ----------
    result : ContinuousSurvivalResult
        A continuous-time survival fit.

    Returns
    -------
    ContinuousSurvivalScores
        The report.
    """
    record = result.record
    single = len(result.causes) == 1
    names = tuple(
        level_name(arm, None if single else cause, time)
        for arm in result.arms
        for cause in result.causes
        for time in result.horizons
    )
    return ContinuousSurvivalScores(
        names=names,
        standardized_score=tuple(float(v) for v in record.standardized_score),
        cutoff=float(record.cutoff),
        converged=bool(record.converged),
        steps=int(record.steps),
        halvings=int(record.halvings),
        off_grid_share=float(record.off_grid_share),
    )


def nuisance_report(result: ContinuousSurvivalResult) -> ContinuousSurvivalNuisances:
    """The hazard, censoring and treatment selections on the record.

    Parameters
    ----------
    result : ContinuousSurvivalResult
        A continuous-time survival fit.

    Returns
    -------
    ContinuousSurvivalNuisances
        The report.
    """
    record = result.record
    rows = [
        NuisanceSelection(f"hazard of cause {cause}", s.names, s.losses, s.chosen_name)
        for cause, s in zip(result.causes, record.hazards, strict=True)
    ]
    c = record.censoring
    rows.append(NuisanceSelection("censoring hazard", c.names, c.losses, c.chosen_name))
    p = record.propensity
    if p is None:
        rows.append(NuisanceSelection("treatment", (), (), "the supplied treatment_learner"))
    else:
        rows.append(NuisanceSelection("treatment", p.names, p.losses, p.names[p.chosen]))
    return ContinuousSurvivalNuisances(
        rows=tuple(rows),
        cross_fitted=bool(record.cross_fitted),
        n_folds=int(record.n_folds),
        grid_size=int(record.grid_size),
    )


#: ``mechanism=True`` names a second axis a continuous-time fit does not have.
MECHANISM_REFUSAL: Final = Refusal(
    RefusalReason.INVALID_DECLARATION,
    "a continuous-time survival fit has one floor, on pi(a | L) S_c(u- | a, L); "
    "mechanism=True is a point-treatment option",
)

#: Why a replay that does not reproduce the fit is refused.
REPLAY_REFUSAL: Final = Refusal(
    RefusalReason.INVALID_DECLARATION,
    "the stored estimator and data do not reproduce this fit at its own floor, so no other "
    "floor can be read against it. A refit reproduces a fit bit for bit only under the same "
    "library versions and BLAS thread settings; cleverly.load warns with "
    "VersionMismatchWarning when the artifact was saved by another version.",
)

#: The multiples of the fitted floor in the default grid; the ``1`` point is the floor itself.
DEFAULT_MULTIPLES: Final[tuple[float, ...]] = (0.25, 0.5, 1.0, 2.0, 4.0)


def _invalid(text: str) -> CapabilityError:
    return CapabilityError(text, reason=RefusalReason.INVALID_DECLARATION)


def _grid(bounds: Any, fitted: float) -> list[float]:
    """The floors to evaluate: the default multiples of ``fitted``, or the caller's grid."""
    if bounds is None:
        if fitted <= 0.0:
            raise _invalid(
                "the fit's floor is 0, so the default grid of multiples of it has no points; "
                "pass bounds= with floors in (0, 1)"
            )
        return [fitted * m for m in DEFAULT_MULTIPLES if 0.0 < fitted * m < 1.0]
    if isinstance(bounds, (str, bytes)) or not isinstance(bounds, Sequence):
        bounds = [bounds] if np.isscalar(bounds) else list(bounds)
    grid: list[float] = []
    for value in bounds:
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise _invalid(
                f"a continuous-time truncation floor is one number in (0, 1); got {value!r}"
            )
        floor = float(value)
        if not np.isfinite(floor) or not 0.0 < floor < 1.0:
            raise _invalid(
                f"a continuous-time truncation floor lies in (0, 1); got {floor!r}. The floor "
                "bounds pi(a | L) S_c(u- | a, L) from below"
            )
        if floor not in grid:
            grid.append(floor)
    if not grid:
        raise _invalid("bounds= names no floor; pass at least one in (0, 1)")
    return sorted(grid)


def continuous_truncation_curve(
    result: ContinuousSurvivalResult,
    bounds: Any = None,
    *,
    estimands: Sequence[str] | None = None,
) -> Any:
    r"""Rerun the targeting walk of a continuous-time fit at other floors.

    The clever covariate of arm :math:`a` divides by
    :math:`\max\{\hat\pi(a \mid L)\hat S_c(u^- \mid a, L), b\}` in every cell of both
    arms, so a floor :math:`b` moves the estimate even where it binds only counterfactual
    cells. The estimand does not depend on :math:`b`. The curve refits the initial hazards
    once from the stored estimator, at its stored seed, and refuses unless the walk at the
    fitted floor reproduces every reported estimate, standard error and record share exactly.
    It then reruns the walk at each other floor; each row equals a refit of the stored
    estimator with ``g_bounds`` set to that floor. A row at another floor reads
    ``experimental``, as such a fit does; the row at the fitted floor keeps the fit's
    settings and support.

    Parameters
    ----------
    result : ContinuousSurvivalResult
        A fit that holds its estimator and data.
    bounds : sequence of float or None
        Floors in :math:`(0, 1)`, deduplicated and sorted. ``None`` takes the fitted floor
        :math:`f` times 1/4, 1/2, 1, 2 and 4, inside :math:`(0, 1)`; the ``1`` point is
        :math:`f` exactly.
    estimands : sequence of str or None
        Reported parameters to include. ``None`` takes every reported parameter.

    Returns
    -------
    dataframe
        One row per floor and estimand in the fit's backend: ``bound``, ``estimand``,
        ``psi``, the standard error and interval, ``truncated_fraction`` (the share of
        units with any cell of :math:`\hat\pi \hat S_c(u^-)` on the grid, in either arm,
        below the floor), ``bounded_share`` (received-arm cells while followed),
        ``last_time_floor_share``, ``frozen_share``, ``converged``, ``steps``,
        ``is_fitted_bound``, ``upper_bound`` (1), ``fitted_lower_bound``,
        ``fitted_upper_bound``, ``fitted_psi``, ``delta_from_fitted``, ``scale``,
        ``level``, and ``support`` and ``support_reason`` on an experimental row.

    Raises
    ------
    CapabilityError
        When the result holds no estimator or data, a floor lies outside :math:`(0, 1)`,
        the fit's floor is 0 and ``bounds`` is omitted, an estimand was not reported, or the
        replay at the fitted floor does not reproduce the fit.

    Notes
    -----
    The curve keeps the initial hazard jumps alive through every walk, one extra
    ``(2, J, n, M)`` array over the fit's peak memory, and also keeps the ``(2, n, M)``
    weight product that the fit frees before its walk: about 600 MB at n = 4,000 with two
    causes.  The memory check counts both, against the fit's ``max_memory_bytes``, or the
    memory available when that is ``None``.
    """
    from ..utils.frames import emit_frame
    from .result import MEASURED_DEFAULTS

    estimator, data = result.estimator, result.data
    if estimator is None or data is None:
        raise _invalid(
            "truncation_curve needs the fitted estimator and the data that produced the result"
        )
    fitted = float(result.record.weight_floor)
    grid = _grid(bounds, fitted)
    if estimands is None:
        names = list(result.estimates)
    else:
        names = list(dict.fromkeys(str(name) for name in estimands))
        unknown = [name for name in names if name not in result.estimates]
        if unknown or not names:
            raise _invalid(
                f"estimands {unknown or names!r} name no parameter this fit reported; choose "
                f"from {list(result.estimates)}"
            )
    seed = estimator.random_state
    assert seed is not None  # a stored estimator carries its resolved seed
    initial = estimator._initial(data, int(seed), folds=estimator._fold_override)
    settings = {key: getattr(estimator, key) for key in MEASURED_DEFAULTS}
    replayed, record = estimator._target(initial, fitted, settings=settings)
    if not _reproduces(result, replayed, record):
        raise CapabilityError.from_refusal(REPLAY_REFUSAL)
    lowest = initial.lowest
    rows: list[dict[str, Any]] = []
    for floor in grid:
        if floor == fitted:
            estimates, at = replayed, record
        else:
            estimates, at = estimator._target(
                initial, floor, settings={**settings, "g_bounds": floor}
            )
        truncated = float(np.mean(lowest < floor))
        for name in names:
            estimate = estimates[name]
            reference = float(result[name].psi)
            rows.append(
                {
                    "bound": floor,
                    "estimand": name,
                    "psi": float(estimate.psi),
                    **estimate.spread_columns(pvalue=False),
                    "truncated_fraction": truncated,
                    "bounded_share": float(at.bounded_share),
                    "last_time_floor_share": float(at.last_time_floor_share),
                    "frozen_share": float(at.frozen_share),
                    "converged": bool(at.converged),
                    "steps": int(at.steps),
                    "is_fitted_bound": floor == fitted,
                    "upper_bound": 1.0,
                    "fitted_lower_bound": fitted,
                    "fitted_upper_bound": 1.0,
                    "fitted_psi": reference,
                    "delta_from_fitted": float(estimate.psi) - reference,
                    "scale": str(estimate.scale),
                    "level": 1.0 - float(estimate.alpha),
                    **(
                        {"support": "experimental", "support_reason": estimate.support.reason}
                        if estimate.support.label == "experimental"
                        else {}
                    ),
                }
            )
    del initial
    keys = list(dict.fromkeys(key for row in rows for key in row))
    return emit_frame({key: [row.get(key) for row in rows] for key in keys}, backend=result.backend)


def _reproduces(result: ContinuousSurvivalResult, estimates: Any, record: Any) -> bool:
    """Whether a replay equals the fit exactly, in every compared field."""
    fitted = result.record
    if set(estimates) != set(result.estimates):
        return False
    for name, estimate in result.estimates.items():
        replay = estimates[name]
        if replay.psi != estimate.psi or replay.plugin_std_error != estimate.plugin_std_error:
            return False
    return (
        tuple(record.initial_psi) == tuple(fitted.initial_psi)
        and record.bounded_share == fitted.bounded_share
        and record.last_time_floor_share == fitted.last_time_floor_share
        and record.frozen_share == fitted.frozen_share
        and record.off_grid_share == fitted.off_grid_share
    )
