"""The result of a continuous-time survival fit."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np

from .._refusal import RefusalReason
from .._support import Support, combine_support
from .._typing import FloatArray
from ..exceptions import CapabilityError
from ..inference.delta import Transform
from ..inference.influence import ParameterEstimate, Scale, make_estimate
from ..inference.results import ratio_contrast, smooth_contrast
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


#: Why a ratio of two continuous-time risks reads experimental.
RATIO_REASON: Final[str] = (
    "No registered study measures a continuous-time risk ratio or odds ratio."
)


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
    restricted: bool = False,
    restricted_horizons: int = 0,
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
    restricted : bool
        Whether the estimate is a restricted mean, whose ``event_share`` is read at its
        restriction time.
    restricted_horizons : int
        How many restriction times the fit targets.  They add targets to the walk, which moves
        every risk of the fit, so this is a check of its own rather than a key of
        :data:`MEASURED_DEFAULTS`: no registered study measures such a fit yet.

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
    if restricted:
        reasons.append("No registered study measures a continuous-time restricted mean.")
    elif scale == "ratio":
        reasons.append(RATIO_REASON)
    if restricted_horizons > 1:
        reasons.append(
            f"The fit targets {restricted_horizons} restricted horizons; no registered study "
            "targets more than one."
        )
    elif restricted_horizons and not restricted:
        reasons.append(
            "The fit also targets restricted means, which moves its risks, and no registered "
            "study measures such a fit."
        )
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

#: The share of random times at which the registered study's score equation held, on average
#: over its fits (``targeting-readings.csv``, ``mean_off_grid_share``): the reason a restricted
#: mean must be targeted rather than read off the targeted risks.
OFF_GRID_SHARE: Final[float] = 0.40

#: The support of a ratio of two risks: no registered study measures one.
RATIO_SUPPORT: Final[Support] = _experimental(RATIO_REASON)

#: The support of a smooth function of reported estimates: no registered study measures one.
CONTRAST_SUPPORT: Final[Support] = _experimental(
    "No registered study measures a smooth function of continuous-time estimates."
)

#: A risk's name, ``risk_regimen[arm @ t=...]`` or ``cif_regimen[arm, cause @ t=...]``, with
#: its time; read only to tell a risk at an untargeted time (refusal C2) from another name.
_LEVEL_PATTERN: Final = re.compile(r"^(?:risk|cif)_regimen\[.+ @ t=(?P<time>[^\]]+)\]$")

if TYPE_CHECKING:
    from ._targeting import WalkResult
    from .engine import ContinuousSurvivalRecord

__all__ = [
    "CONTINUOUS_SURVIVAL_STUDY",
    "ContinuousSurvivalResult",
    "build_estimates",
    "continuous_survival_support",
    "risk_names",
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


def restricted_name(head: str, arm: str, cause: str | None, tau: float) -> str:
    """The parameter name of a restricted mean, or of a difference of two.

    Parameters
    ----------
    head : str
        ``"rmtl_regimen"`` or ``"rmst_regimen"``.
    arm : str
        The arm label, or ``"treated vs reference"`` for a difference.
    cause : str or None
        The cause of a restricted mean time lost on a fit with several causes, else ``None``.
    tau : float
        The restriction time.

    Returns
    -------
    str
        ``rmtl_regimen[arm, cause @ t=tau]``, ``rmst_regimen[arm @ t=tau]`` and the like, the
        names :meth:`cleverly.longitudinal.LongitudinalResult.rmtl` composes.
    """
    inner = arm + ("" if cause is None else f", {cause}")
    return f"{head}[{inner} @ t={_t(tau)}]"


def risk_names(
    arms: tuple[str, str], causes: tuple[str, ...], horizons: tuple[float, ...]
) -> list[str]:
    """The names of every risk and risk difference, in report order.

    Parameters
    ----------
    arms : tuple of str
        ``(reference, treated)``.
    causes : tuple of str
        The cause labels.
    horizons : tuple of float
        The target times.

    Returns
    -------
    list of str
        Each arm's risks by cause and time, then the differences by cause and time.
    """
    single = len(causes) == 1
    out = [
        level_name(arm, None if single else cause, time)
        for arm in arms
        for cause in causes
        for time in horizons
    ]
    out.extend(
        contrast_name("ate_regimen", arms[1], arms[0], None if single else cause, time)
        for cause in causes
        for time in horizons
    )
    return out


def build_estimates(
    walk: WalkResult,
    arms: tuple[str, str],
    causes: Mapping[int, str],
    horizons: tuple[float, ...],
    *,
    n: int,
    alpha: float,
    support: Callable[[str, int], Support],
    restricted: tuple[float, ...] = (),
) -> dict[str, ParameterEstimate]:
    """Assemble the risks, their contrasts and the restricted means from the targeted walk.

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
        The support of an estimate, given its scale and the index of its weight row: a
        target time, then a restriction time.
    restricted : tuple of float
        The restriction times, whose targets follow the target times in each arm and cause.

    Returns
    -------
    dict of str to ParameterEstimate
        Every arm's risk, then each difference and ratio, by time; then the restricted means
        time lost of each arm and cause, their differences, and the restricted mean survival
        times of each arm and their differences, by restriction time.
    """
    labels = list(causes.values())
    J, K = len(labels), len(horizons)
    Q = K + len(restricted)
    single = J == 1
    psi = walk.evaluation.psi
    ic = walk.evaluation.influence
    out: dict[str, ParameterEstimate] = {}
    for a in (0, 1):
        for j, cause in enumerate(labels):
            for k, time in enumerate(horizons):
                p = (a * J + j) * Q + k
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
            p0, p1 = j * Q + k, (J + j) * Q + k
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
    pair = f"{arms[1]} vs {arms[0]}"
    for r, tau in enumerate(restricted):
        q = K + r
        bounds = (0.0, float(tau))
        span = (-float(tau), float(tau))
        for a in (0, 1):
            for j, cause in enumerate(labels):
                p = (a * J + j) * Q + q
                name = restricted_name("rmtl_regimen", arms[a], None if single else cause, tau)
                out[name] = make_estimate(
                    name,
                    psi[p],
                    ic[:, p],
                    n=n,
                    scale="level",
                    alpha=alpha,
                    bounds=bounds,
                    rule_widened=False,
                    support=support("level", q),
                )
        for j, cause in enumerate(labels):
            p0, p1 = j * Q + q, (J + j) * Q + q
            name = restricted_name("rmtl_regimen", pair, None if single else cause, tau)
            out[name] = make_estimate(
                name,
                psi[p1] - psi[p0],
                ic[:, p1] - ic[:, p0],
                n=n,
                scale="difference",
                alpha=alpha,
                bounds=span,
                rule_widened=False,
                support=support("difference", q),
            )
        # RMST(tau) = tau - sum_j RMTL_j(tau); its score is solved by linearity.
        lost = [
            (
                sum(psi[(a * J + j) * Q + q] for j in range(J)),
                np.sum([ic[:, (a * J + j) * Q + q] for j in range(J)], axis=0),
            )
            for a in (0, 1)
        ]
        for a in (0, 1):
            name = restricted_name("rmst_regimen", arms[a], None, tau)
            out[name] = make_estimate(
                name,
                float(tau) - lost[a][0],
                -lost[a][1],
                n=n,
                scale="level",
                alpha=alpha,
                bounds=bounds,
                rule_widened=False,
                support=support("level", q),
            )
        name = restricted_name("rmst_regimen", pair, None, tau)
        out[name] = make_estimate(
            name,
            lost[0][0] - lost[1][0],
            lost[0][1] - lost[1][1],
            n=n,
            scale="difference",
            alpha=alpha,
            bounds=span,
            rule_widened=False,
            support=support("difference", q),
        )
    return out


@dataclass(frozen=True)
class ContinuousSurvivalResult(Mapping[str, ParameterEstimate]):
    """The targeted risks of a continuous-time survival fit.

    Behaves as a mapping from parameter name to
    :class:`~cleverly.inference.ParameterEstimate`.  Every method reads targeted estimates, or a
    smooth function of them: a quantity whose score the fit did not solve, such as a risk
    between target times or a restricted mean the fit did not target, is refused.

    Parameters
    ----------
    estimates : dict of str to ParameterEstimate
        Each arm's risk (``risk_regimen[arm @ t=60]``, or ``cif_regimen[arm, cause @ t=60]``
        with competing causes), the risk differences (``ate_regimen[...]``) and the risk ratios
        (``rr_regimen[...]``).  A fit with ``restricted_horizons=`` adds the restricted mean
        time lost to each cause (``rmtl_regimen[arm, cause @ t=tau]``) and the restricted mean
        survival time (``rmst_regimen[arm @ t=tau]``) of each arm, and their differences
        (``rmtl_regimen[treated vs reference, cause @ t=tau]``, ``rmst_regimen[... vs ...]``).
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
    restricted_horizons : tuple of float
        The restriction times the fit targeted.
    """

    estimates: dict[str, ParameterEstimate]
    record: ContinuousSurvivalRecord
    simultaneous: Any
    arms: tuple[str, str]
    causes: tuple[str, ...]
    horizons: tuple[float, ...]
    n: int
    backend: str | None = None
    restricted_horizons: tuple[float, ...] = field(default=())

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
        Q = K + len(self.restricted_horizons)
        single = J == 1
        reference, treated = self.arms
        out: dict[str, float] = {}
        for a, arm in enumerate(self.arms):
            for j, cause in enumerate(self.causes):
                for k, time in enumerate(self.horizons):
                    name = level_name(arm, None if single else cause, time)
                    out[name] = psi[(a * J + j) * Q + k]
        for j, cause in enumerate(self.causes):
            for k, time in enumerate(self.horizons):
                name = contrast_name(
                    "ate_regimen", treated, reference, None if single else cause, time
                )
                out[name] = psi[(J + j) * Q + k] - psi[j * Q + k]
        return out

    @property
    def n_folds(self) -> int:
        """The number of cross-fitting folds, ``1`` for a full-sample fit."""
        return int(self.record.n_folds)

    # ----------------------------------------------------------- reading the fit

    def _single(self) -> bool:
        return len(self.causes) == 1

    def _arm(self, arm: str, what: str) -> int:
        if arm not in self.arms:
            raise ValueError(f"{what}: unknown arm {arm!r}; this fit compares {list(self.arms)}")
        return self.arms.index(arm)

    def _risk_index(self) -> dict[str, tuple[str, str | None, float]]:
        """``name -> (arm, cause, time)`` of every reported risk, composed forward."""
        single = self._single()
        return {
            level_name(arm, None if single else cause, time): (
                arm,
                None if single else cause,
                time,
            )
            for arm in self.arms
            for cause in self.causes
            for time in self.horizons
        }

    def _refuse_untargeted_time(self, time: float) -> None:
        """Refusal C2: a time that is not a target time."""
        raise CapabilityError(
            f"{_t(time)} is not a target time, and a risk between target times solves no "
            f"score equation of the fit; the target times are {list(self.horizons)}.",
            reason=RefusalReason.API_REDIRECTION,
            alternative=f"Add {_t(time)} to horizons= and fit again.",
        )

    def _untargeted_risk(self, name: str) -> float | None:
        """The time of ``name`` when it names a risk of this fit at an untargeted time."""
        match = _LEVEL_PATTERN.match(name)
        if match is None:
            return None
        try:
            time = float(match.group("time"))
        except ValueError:
            return None
        single = self._single()
        for arm in self.arms:
            for cause in self.causes:
                if level_name(arm, None if single else cause, time) == name:
                    return None if time in self.horizons else time
        return None

    def curve(
        self,
        scale: Literal["risk", "survival"] = "risk",
        *,
        times: Sequence[float] | None = None,
    ) -> Any:
        """The targeted risks and their differences as a curve: one row per arm, cause and time.

        Only the target times are reported.  A risk between two target times would be a
        plug-in of the fitted hazards whose score the fit did not solve, so it has no
        first-order guarantee and is refused.  ``scale="survival"`` reports
        :math:`S(t) = 1 - F(t)` for a level, mirroring its interval, and the negated difference
        :math:`S_a - S_b = -(F_a - F_b)` for a contrast, with the interval negated and swapped.
        A fit with several causes refuses it, because event-free survival is one minus the sum
        of the incidences, not one minus one of them.

        Parameters
        ----------
        scale : {"risk", "survival"}
            The view of each row.
        times : sequence of float or None
            The target times to report.  ``None`` reports every target time.

        Returns
        -------
        dataframe
            One row per arm, cause and time, then one per difference, in the input backend.
            The columns are ``estimand`` (the quantity under the view), ``parameter`` (the
            key of the estimate the row reads), ``arm``, ``cause`` (with several causes),
            ``time``, ``psi``, ``std_err``, ``ci_lower``, ``ci_upper``, ``band_lower`` and
            ``band_upper`` (the simultaneous band, when the fit has one), ``scale``
            (``"level"`` or ``"difference"``), ``view`` and ``support``.

        Raises
        ------
        CapabilityError
            When a time is not a target time, or ``scale="survival"`` on a fit with several
            causes.

        See Also
        --------
        cleverly.survival.ContinuousSurvivalResult.to_frame
            Every estimate, with no time column.
        cleverly.survival.ContinuousSurvivalResult.rmst
            The area under the survival curve up to a restriction time.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from cleverly.survival import ContinuousSurvivalTMLE, CoxHazard
        >>> rng = np.random.default_rng(0)
        >>> x = rng.normal(size=300)
        >>> treated = rng.binomial(1, 0.5, size=300)
        >>> event_time = rng.exponential(1 / np.exp(-1.0 + 0.5 * x - 0.5 * treated))
        >>> dropout = rng.exponential(4.0, size=300)
        >>> frame = pd.DataFrame(
        ...     {
        ...         "time": np.minimum(event_time, dropout),
        ...         "event": (event_time <= dropout).astype(int),
        ...         "treated": treated,
        ...         "x": x,
        ...     }
        ... )
        >>> fit = ContinuousSurvivalTMLE(
        ...     [1.0, 2.0],
        ...     cross_fit=False,
        ...     hazard_library=[CoxHazard("main")],
        ...     censoring_library=[CoxHazard("main")],
        ...     random_state=0,
        ... ).fit(frame, time="time", event="event", treatment="treated", baseline=["x"])
        >>> curve = fit.curve(scale="survival")
        >>> list(curve["estimand"][:2])
        ['survival_regimen[0 @ t=1]', 'survival_regimen[0 @ t=2]']
        >>> bool((curve["psi"][:4] == 1 - fit.curve()["psi"][:4]).all())
        True
        """
        if scale not in ("risk", "survival"):
            raise ValueError(f"scale must be 'risk' or 'survival'; got {scale!r}")
        if scale == "survival" and not self._single():
            raise CapabilityError(
                "this fit reports one cumulative incidence per cause, and 1 - one cause's "
                "incidence is not event-free survival, which is 1 - the sum of the incidences.",
                reason=RefusalReason.DIFFERENT_ESTIMAND,
                alternative=(
                    "Use scale='risk', or contrast() of the incidences of every cause for "
                    "their sum."
                ),
            )
        chosen = self.horizons if times is None else tuple(float(t) for t in times)
        for time in chosen:
            if time not in self.horizons:
                self._refuse_untargeted_time(time)
        single = self._single()
        survival = scale == "survival"
        bands = getattr(self.simultaneous, "bands", None)
        reference, treated = self.arms
        rows: list[dict[str, Any]] = []

        def row(name: str, arm: str, cause: str, time: float, level: bool) -> dict[str, Any]:
            estimate = self.estimates[name]
            low, high = estimate.ci
            band = None if bands is None else bands.get(name)
            psi = estimate.psi
            if survival:
                if level:
                    psi, low, high = 1.0 - psi, 1.0 - high, 1.0 - low
                    band = None if band is None else (1.0 - band[1], 1.0 - band[0])
                else:
                    psi, low, high = -psi, -high, -low
                    band = None if band is None else (-band[1], -band[0])
            estimand = name
            if survival and level:
                estimand = name.replace("risk_regimen[", "survival_regimen[", 1)
            out: dict[str, Any] = {"estimand": estimand, "parameter": name, "arm": arm}
            if not single:
                out["cause"] = cause
            out.update(
                time=time,
                psi=float(psi),
                std_err=estimate.std_error,
                ci_lower=float(low),
                ci_upper=float(high),
            )
            if bands is not None:
                out["band_lower"] = None if band is None else float(band[0])
                out["band_upper"] = None if band is None else float(band[1])
            out.update(
                scale="level" if level else "difference",
                view=scale,
                support=estimate.support.label,
            )
            return out

        for arm in self.arms:
            for cause in self.causes:
                for time in chosen:
                    name = level_name(arm, None if single else cause, time)
                    rows.append(row(name, arm, cause, time, True))
        for cause in self.causes:
            for time in chosen:
                name = contrast_name(
                    "ate_regimen", treated, reference, None if single else cause, time
                )
                rows.append(row(name, f"{treated} vs {reference}", cause, time, False))
        columns = list(rows[0])
        return emit_frame({key: [r[key] for r in rows] for key in columns}, backend=self.backend)

    def contrast(
        self,
        function: Callable[[FloatArray], float],
        names: Sequence[str],
        *,
        name: str | None = None,
        scale: Scale = "difference",
        gradient: Callable[[FloatArray], FloatArray] | None = None,
        transform: Transform | None = None,
    ) -> ParameterEstimate:
        """A smooth function of reported estimates, with joint delta-method inference.

        Each reported estimate solves its score equation, so a smooth function of them solves
        it to first order: its influence curve is the gradient times the joint curves
        (:func:`cleverly.inference.results.smooth_contrast`).  No registered study measures
        such a function, so the result reads ``experimental``.

        Parameters
        ----------
        function : callable
            A smooth scalar function of the selected estimates, in the order of ``names``.
        names : sequence of str
            The reported estimates it reads.
        name : str or None
            The name of the derived estimate.  ``None`` gives ``contrast(<names>)``.
        scale : str
            The reported scale of the derived estimate.  With ``transform=`` it must be
            ``"level"`` or ``"difference"``.
        gradient : callable or None
            The analytic gradient.  ``None`` uses central differences.
        transform : Transform or None
            The map the interval and the test are computed on, as for
            :meth:`cleverly.estimators.TMLEResult.contrast`.

        Returns
        -------
        ParameterEstimate
            The derived estimate.
        """
        derived = smooth_contrast(
            self.estimates,
            function,
            names,
            n=self.n,
            alpha=self._alpha,
            name=name,
            scale=scale,
            gradient=gradient,
            transform=transform,
        )
        support = combine_support(
            [CONTRAST_SUPPORT, *(self.estimates[key].support for key in names)]
        )
        return replace(derived, support=support)

    @property
    def _alpha(self) -> float:
        return float(next(iter(self.estimates.values())).alpha)

    def ratio(
        self,
        numerator: str,
        denominator: str,
        *,
        kind: Literal["rr", "or"] = "rr",
        view: Literal["risk", "survival"] = "risk",
        name: str | None = None,
    ) -> ParameterEstimate:
        """The risk ratio or odds ratio of the two arms' risks at one cause and time.

        The interval is formed on the log scale and exponentiated, with curve
        :math:`IC_1/\\psi_1 - IC_0/\\psi_0` for a risk ratio and
        :math:`IC_1/\\{\\psi_1(1-\\psi_1)\\} - IC_0/\\{\\psi_0(1-\\psi_0)\\}` for an odds ratio
        (:func:`cleverly.inference.results.ratio_contrast`).  ``view="survival"`` takes the
        ratio of :math:`1 - F` instead, which is event-free survival on a fit with one event
        type only.  No registered study measures a continuous-time ratio, so it reads
        ``experimental``.

        Parameters
        ----------
        numerator : str
            The name of one arm's risk.
        denominator : str
            The name of the other arm's risk at the same cause and time.
        kind : {"rr", "or"}
            A ratio of the risks, or a ratio of their odds.
        view : {"risk", "survival"}
            Whether to take the ratio of the risks or of their complements.
        name : str or None
            The name of the derived estimate.  ``None`` gives ``rr_regimen[a vs b @ t=...]``,
            with the cause on a fit with several causes; ``or_regimen`` for an odds ratio,
            and a ``survival_`` prefix for the survival view.  The risk ratio of the treated
            arm over the reference arm is the reported ``rr_regimen`` estimate.

        Returns
        -------
        ParameterEstimate
            The ratio, on the ``"ratio"`` scale with ``log_psi`` set.

        Raises
        ------
        CapabilityError
            When the two names are not the two arms' risks at one cause and one time, when a
            name is a risk at a time the fit did not target, or when ``view="survival"`` is
            asked of a fit with several causes.

        See Also
        --------
        cleverly.sensitivity.evalue : The E-value of this ratio.
        cleverly.survival.ContinuousSurvivalResult.contrast
            Any other smooth function of reported estimates.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from cleverly.survival import ContinuousSurvivalTMLE, CoxHazard
        >>> rng = np.random.default_rng(0)
        >>> x = rng.normal(size=300)
        >>> treated = rng.binomial(1, 0.5, size=300)
        >>> event_time = rng.exponential(1 / np.exp(-1.0 + 0.5 * x - 0.5 * treated))
        >>> dropout = rng.exponential(4.0, size=300)
        >>> frame = pd.DataFrame(
        ...     {
        ...         "time": np.minimum(event_time, dropout),
        ...         "event": (event_time <= dropout).astype(int),
        ...         "treated": treated,
        ...         "x": x,
        ...     }
        ... )
        >>> fit = ContinuousSurvivalTMLE(
        ...     [1.0, 2.0],
        ...     cross_fit=False,
        ...     hazard_library=[CoxHazard("main")],
        ...     censoring_library=[CoxHazard("main")],
        ...     random_state=0,
        ... ).fit(frame, time="time", event="event", treatment="treated", baseline=["x"])
        >>> rr = fit.ratio("risk_regimen[1 @ t=2]", "risk_regimen[0 @ t=2]")
        >>> rr.name, rr.scale
        ('rr_regimen[1 vs 0 @ t=2]', 'ratio')
        >>> bool(np.isclose(rr.psi, fit["rr_regimen[1 vs 0 @ t=2]"].psi))
        True
        """
        if kind not in ("rr", "or"):
            raise ValueError(f"kind must be 'rr' or 'or'; got {kind!r}")
        if view not in ("risk", "survival"):
            raise ValueError(f"view must be 'risk' or 'survival'; got {view!r}")
        index = self._risk_index()
        for given in (numerator, denominator):
            if given not in index:
                untargeted = self._untargeted_risk(given)
                if untargeted is not None:
                    self._refuse_untargeted_time(untargeted)
        top, bottom = index.get(numerator), index.get(denominator)
        if top is None or bottom is None or top[1:] != bottom[1:] or top[0] == bottom[0]:
            raise CapabilityError(
                "ratio() compares the two arms' risks at one cause and one time, and "
                f"{numerator!r} and {denominator!r} are not such a pair.",
                reason=RefusalReason.API_REDIRECTION,
                alternative=("For any other smooth function of reported estimates use contrast()."),
            )
        if view == "survival" and not self._single():
            raise CapabilityError(
                "this fit reports one cumulative incidence per cause, and 1 - one cause's "
                "incidence is not event-free survival, which is 1 - the sum of the incidences.",
                reason=RefusalReason.DIFFERENT_ESTIMAND,
                alternative="Drop view=, or use contrast() of the incidences of every cause.",
            )
        prefix = "survival_" if view == "survival" else ""
        label = name or contrast_name(f"{prefix}{kind}_regimen", top[0], bottom[0], top[1], top[2])
        derived = ratio_contrast(
            self.estimates,
            numerator,
            denominator,
            kind=kind,
            complement=view == "survival",
            n=self.n,
            alpha=self._alpha,
            name=label,
        )
        support = combine_support(
            [RATIO_SUPPORT, self.estimates[numerator].support, self.estimates[denominator].support]
        )
        return replace(derived, support=support)

    def _restricted(
        self, head: str, arm: str, tau: float, cause: str | None, versus: str | None
    ) -> ParameterEstimate:
        """Read a targeted restricted mean, or the difference of two arms' (refusal C1)."""
        what = f"{head[:4]}()"
        tau = float(tau)
        if tau not in self.restricted_horizons:
            raise CapabilityError(
                f"{what} reads a restricted mean the fit targeted, and {_t(tau)} is not one of "
                f"restricted_horizons={self.restricted_horizons}. The risks at the target "
                "times do not solve the score of their integral: between them the score "
                f"equation held at only about {OFF_GRID_SHARE:.0%} of random times in the "
                "registered study.",
                reason=RefusalReason.API_REDIRECTION,
                alternative=(
                    f"Refit ContinuousSurvivalTMLE with restricted_horizons=({_t(tau)},)."
                ),
            )
        first = self._arm(arm, what)
        if versus is None:
            return self.estimates[restricted_name(head, arm, cause, tau)]
        second = self._arm(versus, what)
        if second == first:
            raise ValueError(f"{what}: versus= names the same arm as arm=, {arm!r}")
        pair = f"{self.arms[1]} vs {self.arms[0]}"
        stored = self.estimates[restricted_name(head, pair, cause, tau)]
        if first == 1:
            return stored
        a = self.estimates[restricted_name(head, arm, cause, tau)]
        b = self.estimates[restricted_name(head, versus, cause, tau)]
        label = restricted_name(head, f"{arm} vs {versus}", cause, tau)
        return make_estimate(
            label,
            a.psi - b.psi,
            a.influence_curve - b.influence_curve,
            n=self.n,
            scale="difference",
            alpha=self._alpha,
            bounds=(-tau, tau),
            rule_widened=False,
            support=stored.support,
        )

    def rmtl(
        self, arm: str, tau: float, cause: str | None = None, *, versus: str | None = None
    ) -> ParameterEstimate:
        r"""The restricted mean time lost to one cause up to :math:`\tau`.

        .. math::

            \mathrm{RMTL}_{a,j}(\tau) = E \int_0^\tau F_j(u \mid a, L)\, du
            = \sum_{u_m \le \tau} \hat F_j(u_m) \{\min(u_{m+1}, \tau) - u_m\},

        exactly for the fitted step function on the grid of observed times, with the curve
        :math:`\sum_m \{\min(u_{m+1}, \tau) - u_m\} D_{a,j,u_m}`.  The fit targets it only when
        :math:`\tau` is in ``restricted_horizons=``: the walk solves the scores of the risks
        at the target times alone, so the integral of the targeted curve between them has no
        solved score, and that request is refused.

        Parameters
        ----------
        arm : str
            The arm label.
        tau : float
            The restriction time, one of the fit's ``restricted_horizons``.
        cause : str or None
            The cause.  Required on a fit with several causes.
        versus : str or None
            The other arm.  The result is then the difference ``arm`` minus ``versus``.

        Returns
        -------
        ParameterEstimate
            A ``"level"`` estimate, or a ``"difference"`` estimate with ``versus=``.

        Raises
        ------
        CapabilityError
            When ``tau`` is not one of the fit's ``restricted_horizons``.
        ValueError
            When an arm or the cause is unknown, or the cause is missing on a fit with
            several causes.

        See Also
        --------
        cleverly.survival.ContinuousSurvivalResult.rmst
            The restricted mean survival time, tau minus the sum over causes.
        """
        if cause is None:
            if not self._single():
                raise ValueError(
                    f"this fit has the causes {list(self.causes)}; name one with cause="
                )
        elif self._single() or cause not in self.causes:
            raise ValueError(
                f"unknown cause {cause!r}; this fit has the causes {list(self.causes)}"
                if not self._single()
                else "this fit has one event type; leave cause=None"
            )
        return self._restricted("rmtl_regimen", arm, tau, cause, versus)

    def rmst(self, arm: str, tau: float, *, versus: str | None = None) -> ParameterEstimate:
        r"""The restricted mean survival time up to :math:`\tau`.

        :math:`\mathrm{RMST}_a(\tau) = E \min(T^a, \tau) = \tau - \sum_j \mathrm{RMTL}_{a,j}(\tau)`,
        with the curve :math:`-\sum_j` of theirs, so its score is solved by linearity.  The fit
        targets it only when :math:`\tau` is in ``restricted_horizons=``.

        Parameters
        ----------
        arm : str
            The arm label.
        tau : float
            The restriction time, one of the fit's ``restricted_horizons``.
        versus : str or None
            The other arm.  The result is then the difference ``arm`` minus ``versus``.

        Returns
        -------
        ParameterEstimate
            A ``"level"`` estimate, or a ``"difference"`` estimate with ``versus=``.

        Raises
        ------
        CapabilityError
            When ``tau`` is not one of the fit's ``restricted_horizons``.
        ValueError
            When an arm is unknown.

        See Also
        --------
        cleverly.survival.ContinuousSurvivalResult.rmtl
            The restricted mean time lost to one cause.
        cleverly.survival.ContinuousSurvivalResult.curve
            The targeted risks the restricted mean summarizes.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from cleverly.survival import ContinuousSurvivalTMLE, CoxHazard
        >>> rng = np.random.default_rng(0)
        >>> x = rng.normal(size=300)
        >>> treated = rng.binomial(1, 0.5, size=300)
        >>> event_time = rng.exponential(1 / np.exp(-1.0 + 0.5 * x - 0.5 * treated))
        >>> dropout = rng.exponential(4.0, size=300)
        >>> frame = pd.DataFrame(
        ...     {
        ...         "time": np.minimum(event_time, dropout),
        ...         "event": (event_time <= dropout).astype(int),
        ...         "treated": treated,
        ...         "x": x,
        ...     }
        ... )
        >>> fit = ContinuousSurvivalTMLE(
        ...     [1.0, 2.0],
        ...     restricted_horizons=(2.0,),
        ...     cross_fit=False,
        ...     hazard_library=[CoxHazard("main")],
        ...     censoring_library=[CoxHazard("main")],
        ...     random_state=0,
        ... ).fit(frame, time="time", event="event", treatment="treated", baseline=["x"])
        >>> gain = fit.rmst("1", 2.0, versus="0")
        >>> gain.name, gain.scale
        ('rmst_regimen[1 vs 0 @ t=2]', 'difference')
        >>> lost = fit.rmtl("1", 2.0)
        >>> round(fit.rmst("1", 2.0).psi + lost.psi, 12)
        2.0
        """
        return self._restricted("rmst_regimen", arm, tau, None, versus)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready record of the fit.

        Returns
        -------
        dict of str to Any
            ``estimates``, each estimate's
            :meth:`~cleverly.inference.ParameterEstimate.to_dict` by name; ``record``, the
            selected models and how the walk ended; and ``settings``, the arms, causes, target
            times, restriction times, units and significance level.
        """
        record = self.record
        propensity = record.propensity
        return {
            "estimates": {name: estimate.to_dict() for name, estimate in self.estimates.items()},
            "record": {
                "hazards": [selection.chosen_name for selection in record.hazards],
                "censoring": record.censoring.chosen_name,
                "propensity": None if propensity is None else propensity.names[propensity.chosen],
                "steps": int(record.steps),
                "halvings": int(record.halvings),
                "converged": bool(record.converged),
                "cutoff": float(record.cutoff),
                "standardized_score": [float(v) for v in record.standardized_score],
                "bounded_share": float(record.bounded_share),
                "last_time_floor_share": float(record.last_time_floor_share),
                "frozen_share": float(record.frozen_share),
                "off_grid_share": float(record.off_grid_share),
                "cross_fitted": bool(record.cross_fitted),
                "n_folds": int(record.n_folds),
                "grid_size": int(record.grid_size),
            },
            "settings": {
                "arms": list(self.arms),
                "causes": list(self.causes),
                "horizons": [float(t) for t in self.horizons],
                "restricted_horizons": [float(t) for t in self.restricted_horizons],
                "n": int(self.n),
                "alpha": self._alpha,
            },
        }

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
