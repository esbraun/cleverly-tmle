"""The result of a continuous-time survival fit."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np

from .._support import Support
from ..inference.influence import ParameterEstimate, make_estimate
from ..utils.frames import emit_frame

#: The registered study that measures :class:`~cleverly.survival.ContinuousSurvivalTMLE`.
CONTINUOUS_SURVIVAL_STUDY: Final[str] = "continuous-time-survival"

#: The share of the observed events (any cause) at or before a target time that the study's
#: gated times reach.  Its gated times sit at the median and the upper quartile of the observed
#: event times (shares 0.5 and 0.75); at the lower quartile (0.25) the influence-curve standard
#: error read up to 8% below the sampling spread and coverage fell to 0.88, so that time was
#: published without a verdict.  The band leaves at least five sampling standard deviations at
#: n = 1,000 between each gated time and its edge, and so reaches past the measured shares (about
#: 0.44 to 0.80 over the study's draws).  It is calibrated on one law: it reads neither the events
#: of each arm and cause nor n, which :data:`MEASURED_N` bounds.
MEASURED_EVENT_SHARE: Final[tuple[float, float]] = (0.4, 0.85)

#: The sample size the study gates.  A smaller fit has fewer events of each arm and cause; at the
#: early time the cell with the fewest events (about 16 per fit) covered least.
MEASURED_N: Final[int] = 1_000

#: The limits every continuous-time estimate states, supported or not.
CONTINUOUS_LIMITS: Final[tuple[str, ...]] = (
    "One law (van Hage et al. 2025, scenario 1, with an arm-specific term): a three-level factor "
    "and a normal covariate, a binary treatment, two causes with constant hazards and "
    "independent censoring, n = 1,000, the default libraries, cross-fitted. The double "
    "robustness cells add covariate-dependent censoring. At n = 4,000 the same fit covered "
    "0.918 to 0.960 over 400 draws (reported, not gated).",
    "Under near-violations of positivity (treatment probabilities near 0 or 1) the intervals "
    "under-cover badly: a reported cell at n = 4,000 covered 0.37 to 0.97.",
    "The time condition reads the share of observed events, of any cause, by the target time. "
    "It is calibrated on one law, extends past the measured shares (about 0.44 to 0.80), and "
    "does not read how many events each arm and cause has. The study targeted three times "
    "jointly.",
)

#: The constructor settings the study measures at their defaults, each with its default.
MEASURED_DEFAULTS: Final[Mapping[str, Any]] = {
    "norm": "covariance",
    "g_bounds": "auto",
    "max_iter": 500,
    "step_size": 0.1,
    "tol": None,
    "tilt": "all",
    "n_folds": 10,
    "learner_folds": 5,
    "alpha": 0.05,
}


def _experimental(reason: str) -> Support:
    return Support("experimental", reason, ("X91",), limits=CONTINUOUS_LIMITS)


def continuous_survival_support(
    *,
    scale: str,
    event_share: float,
    n: int,
    causes: int,
    cross_fitted: bool,
    default_nuisances: bool,
    settings: Mapping[str, Any],
) -> Support:
    """The support of one continuous-time estimate.

    An estimate reads ``supported`` when the registered ``continuous-time-survival`` study
    measures its configuration: a risk or a risk difference with two causes and at least
    :data:`MEASURED_N` units, at a time by which between 40% and 85% of the observed events have
    happened (:data:`MEASURED_EVENT_SHARE`),
    from a cross-fitted fit with the default libraries and the settings of
    :data:`MEASURED_DEFAULTS`.  Otherwise it reads ``experimental`` with every reason that
    applies.

    Parameters
    ----------
    scale : {"level", "difference", "ratio"}
        The estimate's scale.
    event_share : float
        The share of the observed events, any cause, at or before the estimate's time.
    n : int
        The number of units.
    causes : int
        The number of event types.
    cross_fitted : bool
        Whether the nuisances are fitted out of fold.
    default_nuisances : bool
        Whether the hazard and censoring libraries and the treatment learner are the defaults.
    settings : mapping of str to Any
        The fit's values of the keys of :data:`MEASURED_DEFAULTS`.

    Returns
    -------
    Support
        The estimate's support.

    Examples
    --------
    >>> from cleverly.survival.result import MEASURED_DEFAULTS, continuous_survival_support
    >>> support = continuous_survival_support(
    ...     scale="level", event_share=0.6, n=1_000, causes=2, cross_fitted=True,
    ...     default_nuisances=True, settings=MEASURED_DEFAULTS,
    ... )
    >>> support.label
    'supported'
    >>> continuous_survival_support(
    ...     scale="ratio", event_share=0.6, n=1_000, causes=2, cross_fitted=True,
    ...     default_nuisances=True, settings=MEASURED_DEFAULTS,
    ... ).label
    'experimental'
    """
    reasons: list[str] = []
    if scale == "ratio":
        reasons.append("No registered study measures a continuous-time risk ratio.")
    low, high = MEASURED_EVENT_SHARE
    if event_share < low:
        reasons.append(
            f"Only {event_share:.0%} of the observed events happen by this time. The "
            f"registered study gates times by which {low:.0%} to {high:.0%} have happened; at "
            "its earliest time (25%) the standard error ran low and coverage fell to 0.88."
        )
    elif event_share > high:
        reasons.append(
            f"{event_share:.0%} of the observed events happen by this time, beyond the measured "
            f"band ({low:.0%} to {high:.0%}); the registered study's latest gated time sits at "
            "75%."
        )
    if n < MEASURED_N:
        reasons.append(
            f"The fit has {n:,} units, and the registered study gates n = {MEASURED_N:,}; a "
            "smaller sample has fewer events of each arm and cause."
        )
    if causes != 2:
        reasons.append(
            "The registered study measures two causes; no study measures a single event type "
            "or more than two."
        )
    if not cross_fitted:
        reasons.append(
            "The fit is in-sample. The registered study gates cross-fitted fits; the in-sample "
            "fit covered 0.855 to 0.950 at n = 1,000 (reported, not gated)."
        )
    if not default_nuisances:
        reasons.append(
            "The fit replaces a default library or the default treatment learner, and the "
            "registered study measures the defaults only."
        )
    changed = sorted(key for key, value in MEASURED_DEFAULTS.items() if settings[key] != value)
    if changed:
        reasons.append(
            f"The fit sets {', '.join(changed)} away from the default, and the registered study "
            "measures the defaults only."
        )
    if reasons:
        return _experimental(" ".join(reasons))
    return Support(
        "supported",
        f"The registered {CONTINUOUS_SURVIVAL_STUDY} study measures this configuration.",
        (CONTINUOUS_SURVIVAL_STUDY,),
        limits=CONTINUOUS_LIMITS,
    )


#: The support of a simultaneous band: the study measures pointwise intervals only.
BAND_SUPPORT: Final[Support] = _experimental(
    "No registered study measures the simultaneous band of the continuous-time TMLE; the "
    "study measures pointwise intervals only."
)

if TYPE_CHECKING:
    from ._targeting import WalkResult
    from .engine import ContinuousSurvivalRecord

__all__ = [
    "CONTINUOUS_SURVIVAL_STUDY",
    "ContinuousSurvivalResult",
    "build_estimates",
    "continuous_survival_support",
]


def _t(time: float) -> str:
    return f"{time:g}"


def level_name(arm: str, cause: str | None, time: float) -> str:
    """The parameter name of one arm's risk.

    Parameters
    ----------
    arm : str
        The arm label.
    cause : str or None
        The cause label, ``None`` with one event type.
    time : float
        The target time.

    Returns
    -------
    str
        ``risk_regimen[arm @ t=time]`` or ``cif_regimen[arm, cause @ t=time]``.
    """
    if cause is None:
        return f"risk_regimen[{arm} @ t={_t(time)}]"
    return f"cif_regimen[{arm}, {cause} @ t={_t(time)}]"


def contrast_name(head: str, treated: str, reference: str, cause: str | None, time: float) -> str:
    """The parameter name of a contrast of the two arms.

    Parameters
    ----------
    head : str
        ``"ate_regimen"`` or ``"rr_regimen"``.
    treated : str
        The treated arm's label.
    reference : str
        The reference arm's label.
    cause : str or None
        The cause label, ``None`` with one event type.
    time : float
        The target time.

    Returns
    -------
    str
        The name.
    """
    inner = f"{treated} vs {reference}" + ("" if cause is None else f", {cause}")
    return f"{head}[{inner} @ t={_t(time)}]"


def build_estimates(
    walk: WalkResult,
    arms: tuple[str, str],
    causes: Mapping[int, str],
    horizons: tuple[float, ...],
    *,
    n: int,
    alpha: float,
    support: Callable[[str, int], Support],
) -> dict[str, ParameterEstimate]:
    """Assemble the risks, risk differences and risk ratios from the targeted walk.

    Parameters
    ----------
    walk : WalkResult
        The targeting outcome.
    arms : tuple of str
        ``(reference, treated)`` labels, arm codes 0 and 1.
    causes : mapping of int to str
        The causes in target order.
    horizons : tuple of float
        The target times.
    n : int
        The number of units.
    alpha : float
        Significance level.
    support : callable
        The support of an estimate, given its scale and its time's index.

    Returns
    -------
    dict of str to ParameterEstimate
        Every arm's risk, then each difference and ratio, by time.
    """
    labels = list(causes.values())
    J, K = len(labels), len(horizons)
    single = J == 1
    psi = walk.evaluation.psi
    ic = walk.evaluation.influence
    out: dict[str, ParameterEstimate] = {}
    for a in (0, 1):
        for j, cause in enumerate(labels):
            for k, time in enumerate(horizons):
                p = (a * J + j) * K + k
                name = level_name(arms[a], None if single else cause, time)
                out[name] = make_estimate(
                    name,
                    psi[p],
                    ic[:, p],
                    n=n,
                    scale="level",
                    alpha=alpha,
                    bounds=(0.0, 1.0),
                    rule_widened=False,
                    support=support("level", k),
                )
    for j, cause in enumerate(labels):
        for k, time in enumerate(horizons):
            p0, p1 = j * K + k, (J + j) * K + k
            label = None if single else cause
            name = contrast_name("ate_regimen", arms[1], arms[0], label, time)
            out[name] = make_estimate(
                name,
                psi[p1] - psi[p0],
                ic[:, p1] - ic[:, p0],
                n=n,
                scale="difference",
                alpha=alpha,
                bounds=(-1.0, 1.0),
                rule_widened=False,
                support=support("difference", k),
            )
            name = contrast_name("rr_regimen", arms[1], arms[0], label, time)
            if psi[p0] > 0 and psi[p1] > 0:
                log_ic = ic[:, p1] / psi[p1] - ic[:, p0] / psi[p0]
                out[name] = make_estimate(
                    name,
                    psi[p1] / psi[p0],
                    log_ic,
                    n=n,
                    scale="ratio",
                    alpha=alpha,
                    log_psi=float(np.log(psi[p1] / psi[p0])),
                    rule_widened=False,
                    support=support("ratio", k),
                )
    return out


@dataclass(frozen=True)
class ContinuousSurvivalResult(Mapping[str, ParameterEstimate]):
    """The targeted risks of a continuous-time survival fit.

    Behaves as a mapping from parameter name to
    :class:`~cleverly.inference.ParameterEstimate`.

    Parameters
    ----------
    estimates : dict of str to ParameterEstimate
        Each arm's risk (``risk_regimen[arm @ t=60]``, or ``cif_regimen[arm, cause @ t=60]``
        with competing causes), the risk differences (``ate_regimen[...]``) and the risk ratios
        (``rr_regimen[...]``).
    record : ContinuousSurvivalRecord
        The nuisance selections and the targeting record.
    simultaneous : SimultaneousBands or None
        The sup-t band over every risk and risk difference, when requested.
    arms : tuple of str
        ``(reference, treated)``.
    causes : tuple of str
        The cause labels.
    horizons : tuple of float
        The target times.
    n : int
        The number of units.
    backend : str or None
        The dataframe library the data arrived in; :meth:`to_frame` returns a frame of it.
    """

    estimates: dict[str, ParameterEstimate]
    record: ContinuousSurvivalRecord
    simultaneous: Any
    arms: tuple[str, str]
    causes: tuple[str, ...]
    horizons: tuple[float, ...]
    n: int
    backend: str | None = None

    def __getitem__(self, name: str) -> ParameterEstimate:
        return self.estimates[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.estimates)

    def __len__(self) -> int:
        return len(self.estimates)

    @property
    def initial_estimates(self) -> dict[str, float]:
        """Every risk and risk difference at the initial hazard fits, before any step."""
        psi = self.record.initial_psi
        J, K = len(self.causes), len(self.horizons)
        single = J == 1
        reference, treated = self.arms
        out: dict[str, float] = {}
        for a, arm in enumerate(self.arms):
            for j, cause in enumerate(self.causes):
                for k, time in enumerate(self.horizons):
                    name = level_name(arm, None if single else cause, time)
                    out[name] = psi[(a * J + j) * K + k]
        for j, cause in enumerate(self.causes):
            for k, time in enumerate(self.horizons):
                name = contrast_name(
                    "ate_regimen", treated, reference, None if single else cause, time
                )
                out[name] = psi[(J + j) * K + k] - psi[j * K + k]
        return out

    @property
    def n_folds(self) -> int:
        """The number of cross-fitting folds, ``1`` for a full-sample fit."""
        return int(self.record.n_folds)

    def summary(self, *, detail: Literal["short", "full"] = "short") -> Any:
        """A printable report of the fit.

        Parameters
        ----------
        detail : {"short", "full"}
            ``"full"`` adds the assumptions and how the update ended.

        Returns
        -------
        Summary
            The report.
        """
        from .._summary._continuous_survival import continuous_survival_summary

        return continuous_survival_summary(self, detail=detail)

    def _repr_html_(self) -> str:
        return str(self.summary().as_html())

    def __repr__(self) -> str:
        return (
            f"ContinuousSurvivalResult(parameters={len(self.estimates)}, n={self.n}, "
            f"causes={len(self.causes)}, times={len(self.horizons)})"
        )

    def to_frame(self) -> Any:
        """One row per parameter, in the backend the data arrived in.

        Returns
        -------
        dataframe
            The estimates with their standard errors and intervals.
        """
        rows = [estimate.to_dict() for estimate in self.estimates.values()]
        columns: list[str] = []
        for row in rows:
            columns.extend(key for key in row if key not in columns)
        return emit_frame(
            {key: [row.get(key) for row in rows] for key in columns}, backend=self.backend
        )

    def save(self, path: Any) -> Any:
        """Write this result to a trusted joblib artifact; see :func:`cleverly.load`.

        Parameters
        ----------
        path : path-like
            Destination file.

        Returns
        -------
        Path
            The destination path, as given.
        """
        from ..estimators.serialize import save as _save

        return _save(self, path)
