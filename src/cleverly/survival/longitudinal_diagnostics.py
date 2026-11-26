"""The diagnostics of a continuous-time longitudinal fit (X6 part (e)).

:class:`~cleverly.assessment.DiagnosticsFacade` returns these reports from
``result.diagnostics.support()``, ``score_equations()`` and ``nuisance_models()`` on a
:class:`~cleverly.survival.ContinuousLongitudinalResult`. Each one reads
:class:`~cleverly.survival.ContinuousLongitudinalRecord` and refits nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from .._summary import vocabulary
from .._typing import FloatArray
from .diagnostics import NuisanceSelection
from .result import level_name

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .longitudinal_result import ContinuousLongitudinalResult

__all__ = [
    "ContinuousLongitudinalNuisances",
    "ContinuousLongitudinalScores",
    "ContinuousLongitudinalSupport",
    "nuisance_report",
    "score_report",
    "support_report",
]


@dataclass(frozen=True)
class ContinuousLongitudinalSupport:
    """The weight floor, the bounded jumps and the visit caps of a longitudinal fit.

    The clever covariate of node :math:`k` divides by the product of the treatment
    probabilities at the decision visits and the censoring survival, bounded below by
    :attr:`weight_floor`. The report states how often the floor bound, how often the engine
    bounded a hazard or visit jump, and how many units each regime keeps.

    Parameters
    ----------
    n : int
        The number of units.
    regimes : tuple of str
        The regime labels, the reference first.
    horizons : tuple of float
        The target times.
    caps : dict of str to int
        Each regime's visit cap :math:`K^*`.
    past_cap_share : dict of str to tuple of float
        Per regime and target time, the estimated share of followers with more than
        :math:`K^*` visits before it.
    followers_at_last : dict of str to int
        Per regime, the units that follow it at every decision visit and are at risk at the
        last target time.
    weight_floor : float
        The floor on the product of the treatment probabilities and the censoring survival.
    bounded_share : float
        The share of followers' weighted (row, time) cells at risk that the floor raised.
    jump_bound_share : float
        The share of node rows whose combined death jump the engine bounded below one.
    visit_bound_share : float
        The share of node rows whose visit jump the engine bounded below one.
    cross_fitted : bool
        Whether the probabilities were predicted out of fold.
    """

    n: int
    regimes: tuple[str, ...]
    horizons: tuple[float, ...]
    caps: dict[str, int]
    past_cap_share: dict[str, tuple[float, ...]]
    followers_at_last: dict[str, int]
    weight_floor: float
    bounded_share: float
    jump_bound_share: float
    visit_bound_share: float
    cross_fitted: bool

    @property
    def severity(self) -> str:
        """``"adequate"``, ``"strain"`` or ``"serious"`` by :attr:`bounded_share`.

        The tiers are the point support report's truncation tiers
        (:data:`~cleverly.sensitivity.positivity.STRAIN_SHARE` and
        :data:`~cleverly.sensitivity.positivity.SERIOUS_SHARE`), read on the share of cells
        the floor raised.
        """
        from ..sensitivity.positivity import SERIOUS_SHARE, STRAIN_SHARE

        if self.bounded_share > SERIOUS_SHARE:
            return "serious"
        if self.bounded_share > STRAIN_SHARE:
            return "strain"
        return "adequate"


@dataclass(frozen=True)
class ContinuousLongitudinalScores:
    """The score equations of the node walks, as they ended.

    Each node walk stops at :math:`s_n / N`, with :math:`N` the regime's nodes, so every
    target's total :math:`|P_n D^*| / \\hat\\sigma` is at most :attr:`cutoff` when every walk
    converged.

    Parameters
    ----------
    names : tuple of str
        Each targeted risk, regime-major.
    standardized_score : tuple of float
        Each target's final :math:`|P_n D^*| / \\hat\\sigma`.
    cutoff : float
        :math:`s_n`.
    converged : bool
        Whether every node walk reached its cutoff.
    node_scores : dict of str to ndarray
        Per regime, ``(N, P)`` each node's final :math:`|P_n D^p_k| / \\hat\\sigma_p`, the
        baseline node first.
    steps : dict of str to tuple of int
        Per regime, each node's accepted steps, the baseline node first.
    """

    names: tuple[str, ...]
    standardized_score: tuple[float, ...]
    cutoff: float
    converged: bool
    node_scores: dict[str, FloatArray]
    steps: dict[str, tuple[int, ...]]

    @property
    def passed(self) -> bool:
        """Whether every standardized score is at or below the cutoff."""
        return bool(all(score <= self.cutoff for score in self.standardized_score))

    @property
    def worst_ratio(self) -> float:
        """The largest standardized score as a multiple of the cutoff."""
        return float(max(self.standardized_score) / self.cutoff)


@dataclass(frozen=True)
class ContinuousLongitudinalNuisances:
    """The hazard, visit, censoring and treatment selections of a longitudinal fit.

    Parameters
    ----------
    rows : tuple of NuisanceSelection
        Each cause's hazard, the visit intensity, the censoring hazard and the treatment
        model. A cross-fitted fit records the first fold's selections.
    cross_fitted : bool
        Whether the nuisances were fitted out of fold.
    n_folds : int
        The number of cross-fitting folds, ``1`` for a full-sample fit.
    grid_size : int
        The number of grid times up to the last target.
    history_restriction : str
        The model restriction of the history window.
    """

    rows: tuple[NuisanceSelection, ...]
    cross_fitted: bool
    n_folds: int
    grid_size: int
    history_restriction: str

    #: The battery's row label.
    check_label: ClassVar[str] = vocabulary.CONTINUOUS_LONGITUDINAL_NUISANCE_NAME


def support_report(result: ContinuousLongitudinalResult) -> ContinuousLongitudinalSupport:
    """The weight floor, bounded jumps and visit caps, read off the record.

    Parameters
    ----------
    result : ContinuousLongitudinalResult
        A continuous-time longitudinal fit.

    Returns
    -------
    ContinuousLongitudinalSupport
        The report.
    """
    record = result.record
    regimes = tuple(record.regimes)
    return ContinuousLongitudinalSupport(
        n=int(result.n),
        regimes=regimes,
        horizons=tuple(result.horizons),
        caps={r: int(c) for r, c in zip(regimes, record.caps, strict=True)},
        past_cap_share={
            r: tuple(float(v) for v in shares)
            for r, shares in zip(regimes, record.past_cap_share, strict=True)
        },
        followers_at_last={
            r: int(c) for r, c in zip(regimes, record.followers_at_last, strict=True)
        },
        weight_floor=float(record.weight_floor),
        bounded_share=float(record.bounded_share),
        jump_bound_share=float(record.jump_bound_share),
        visit_bound_share=float(record.visit_bound_share),
        cross_fitted=bool(record.cross_fitted),
    )


def score_report(result: ContinuousLongitudinalResult) -> ContinuousLongitudinalScores:
    """The standardized scores of the node walks against the cutoff.

    Parameters
    ----------
    result : ContinuousLongitudinalResult
        A continuous-time longitudinal fit.

    Returns
    -------
    ContinuousLongitudinalScores
        The report.
    """
    record = result.record
    single = len(result.causes) == 1
    regimes = tuple(record.regimes)
    names = tuple(
        level_name(regime, None if single else cause, time)
        for regime in regimes
        for cause in result.causes
        for time in result.horizons
    )
    return ContinuousLongitudinalScores(
        names=names,
        standardized_score=tuple(float(v) for v in record.standardized_score),
        cutoff=float(record.cutoff),
        converged=bool(record.converged),
        node_scores={
            r: np.asarray(s, dtype=float) for r, s in zip(regimes, record.node_scores, strict=True)
        },
        steps={r: tuple(int(v) for v in s) for r, s in zip(regimes, record.steps, strict=True)},
    )


def nuisance_report(result: ContinuousLongitudinalResult) -> ContinuousLongitudinalNuisances:
    """The hazard, visit, censoring and treatment selections on the record.

    Parameters
    ----------
    result : ContinuousLongitudinalResult
        A continuous-time longitudinal fit.

    Returns
    -------
    ContinuousLongitudinalNuisances
        The report.
    """
    record = result.record
    rows = [
        NuisanceSelection(f"hazard of cause {cause}", s.names, s.losses, s.chosen_name)
        for cause, s in zip(result.causes, record.hazards, strict=True)
    ]
    if record.visit is not None:
        v = record.visit
        rows.append(NuisanceSelection("visit intensity", v.names, v.losses, v.chosen_name))
    c = record.censoring
    rows.append(NuisanceSelection("censoring hazard", c.names, c.losses, c.chosen_name))
    p = record.propensity
    if p is None:
        rows.append(NuisanceSelection("treatment", (), (), "the supplied treatment_learner"))
    else:
        rows.append(NuisanceSelection("treatment", p.names, p.losses, p.names[p.chosen]))
    return ContinuousLongitudinalNuisances(
        rows=tuple(rows),
        cross_fitted=bool(record.cross_fitted),
        n_folds=int(record.n_folds),
        grid_size=int(record.grid_size),
        history_restriction=str(record.history_restriction),
    )
