"""The diagnostics of a continuous-time survival fit, read off its record.

:class:`~cleverly.assessment.DiagnosticsFacade` returns these reports from
``result.diagnostics.support()``, ``score_equations()`` and ``nuisance_models()`` on a
:class:`~cleverly.survival.ContinuousSurvivalResult`. Each one reads
:class:`~cleverly.survival.ContinuousSurvivalRecord` and refits nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .result import level_name

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .result import ContinuousSurvivalResult

__all__ = [
    "ContinuousSurvivalNuisances",
    "ContinuousSurvivalScores",
    "ContinuousSurvivalSupport",
    "NuisanceSelection",
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
    target, bounded below by :attr:`positivity_floor`. The report summarizes both factors as
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
        uncensored through it under its received arm, :math:`\\hat S_c(t_k \\mid A, L)`.
    positivity_floor : float
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
    positivity_floor: float
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
    p1 = np.asarray(record.treatment_probability, dtype=float)
    arm = np.asarray(record.arm)
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
            time: _quantiles(record.censoring_survival[:, k])
            for k, time in enumerate(result.horizons)
        },
        positivity_floor=float(record.positivity_floor),
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
