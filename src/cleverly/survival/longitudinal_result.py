"""The result of a continuous-time longitudinal fit (X6 part (e)).

:class:`ContinuousLongitudinalResult` holds every regime's risk of each cause at each target time,
each regime's difference from the reference regime and their ratio. It reads its curve, smooth
functions, ratios and frame through the code path of
:class:`~cleverly.survival.ContinuousSurvivalResult` (``result._TargetedRisks``), with one arm
per regime. Every estimate reads its influence-curve variance. A risk or risk difference reads
supported inside the configuration the registered study ``continuous-time-longitudinal``
measures (:func:`continuous_longitudinal_support`), and experimental outside it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal

import numpy as np

from .._assessment_cache import AssessmentCache
from .._continuous_longitudinal_family import CONTINUOUS_LONGITUDINAL_FAMILY, replay_refusal
from .._refusal import RefusalReason
from .._support import Support
from .._typing import FloatArray
from ..exceptions import CapabilityError
from ..inference.delta import Transform
from ..inference.influence import ParameterEstimate, Scale, make_estimate
from ..inference.sparse import influence_row
from .result import _json_ready, _TargetedRisks, contrast_name, level_name

if TYPE_CHECKING:
    from ._longitudinal_data import ContinuousLongitudinalData
    from .longitudinal import ContinuousLongitudinalRecord, ContinuousLongitudinalTMLE

__all__ = [
    "CONTINUOUS_LONGITUDINAL_STUDY",
    "ContinuousLongitudinalResult",
    "build_result",
    "continuous_longitudinal_support",
]

#: Below this many units following a regime at every decision visit and at risk at the last
#: target time, the fit warns (W1).
FEW_FOLLOWERS: Final = 30
#: Above this estimated share of a regime's followers with more visits than the cap before a
#: target time, the fit warns (W2).
PAST_CAP_SHARE: Final = 0.05

#: The registered study that measures the continuous-time longitudinal TMLE.
CONTINUOUS_LONGITUDINAL_STUDY: Final[str] = "continuous-time-longitudinal"

#: The smallest sample size at which a risk or risk difference reads supported: the study's one
#: size.  The design test checks it against the study's record.
LONGITUDINAL_N: Final[int] = 1_000

#: The number of causes the study fits.  A fit with another number of causes reads
#: experimental.
LONGITUDINAL_CAUSES: Final[int] = 2

#: The largest relative standard deviation of the influence-curve variance
#: (``variance_rel_sd``) at which a risk or risk difference reads supported: the largest median
#: rung over the fits of a gated primary row such that every gated primary row at or below it
#: passes, read after the run from the study's committed ``floors.csv``.  Every gated row
#: passed, static and rule alike, so the value is the largest median rung of any gated row.
LONGITUDINAL_REL_SD: Final[float] = 0.22353336729101952

#: The share of units with an event of the estimate's cause by its time, inside which a risk or
#: risk difference reads supported: the range of the gated rows' median shares (cause 2 at day
#: 10 to cause 1 at day 30), every one of which passed.
LONGITUDINAL_EVENT_SHARE: Final[tuple[float, float]] = (0.092, 0.244)

#: The fewest units that follow a regime at every decision visit and are at risk at the last
#: target time, over every fit of the gated rows.  Fewer reads experimental; fewer than
#: :data:`FEW_FOLLOWERS` also warns (W1).
LONGITUDINAL_FOLLOWERS: Final[int] = 49

#: The largest estimated share of a regime's followers past its visit cap :math:`K^*` before a
#: target time, over every fit of the gated rows.  An estimate whose regime's share at its time
#: is larger reads experimental; above :data:`PAST_CAP_SHARE` the fit also warns (W2).
LONGITUDINAL_PAST_CAP_SHARE: Final[float] = 0.1776787070035202

#: The largest share of weighted cells at risk whose :math:`\pi S_c` product was raised to the
#: floor, over every fit of the gated rows (``bounded_share``).  A larger share reads
#: experimental.
LONGITUDINAL_BOUNDED_SHARE: Final[float] = 0.001797937787020189

#: The constructor settings the study measures, each at the value it fits.
LONGITUDINAL_DEFAULTS: Final[Mapping[str, Any]] = {
    "norm": "covariance",
    "g_bounds": "auto",
    "max_iter": 500,
    "step_size": 0.1,
    "tol": None,
    "n_folds": 10,
    "learner_folds": 5,
    "alpha": 0.05,
}

#: Why a fit with a user learner never reads supported.
USER_LEARNER_REASON: Final[str] = (
    "The fit uses a library or learner other than the default for some model; a study "
    "measures the default libraries and cannot measure another choice."
)

#: Why a fit whose engine bounded a hazard jump reads experimental.
JUMP_BOUND_REASON: Final[str] = (
    "The initial hazard fits put a jump of one or more at some node row, which the engine "
    "bounded below one."
)

#: Why a fit whose engine bounded a visit jump reads experimental.
VISIT_BOUND_REASON: Final[str] = (
    "The initial visit intensity put a jump of one or more at some node row, which the engine "
    "bounded below one; no fit of the registered study bounded one."
)

#: Why a risk ratio reads experimental.
RATIO_REASON: Final[str] = (
    "No registered study measures a risk ratio of the continuous-time longitudinal TMLE."
)

#: The limits every estimate states.
LONGITUDINAL_LIMITS: Final[tuple[str, ...]] = (
    "The hazards, the visit intensity, the censoring hazard and the treatment model read the "
    "history through the window (A(s_k), W, L(s_k), A(s_{k-1}), k, s_k) of the last visit; the "
    "risk is identified under that restriction, no unmeasured confounding at each treatment "
    "time, positivity and censoring independent of the outcome given the history.",
    "Visits after the cap K* are not modelled: the regime holds at baseline and the first K* "
    "visits, and the record's past_cap_share bounds the gap to the regime at every visit. Past "
    "the cap, censoring and the hazards read the window of visit K*, so the capped risk is "
    "identified only when censoring after visit K* + 1 does not depend on the later L or A "
    "given that window. K* is chosen on the full sample.",
    "The variance is the influence-curve variance P_n D^2. One law measures it: a binary "
    "covariate and treatment that change at about two visits per unit over 30 days, two "
    "causes and censoring that depend on both, in continuous time and on whole days, n = "
    "1,000, the default libraries cross-fitted, always and never treat and one rule (treat "
    "when the covariate is 1), with three target times fitted jointly.",
)

#: The support of a smooth function and a band: no study measures them.
RATIO_SUPPORT: Final[Support] = Support(
    "experimental", RATIO_REASON, ("X6",), limits=LONGITUDINAL_LIMITS
)
CONTRAST_SUPPORT: Final[Support] = Support(
    "experimental",
    "No registered study measures a smooth function of continuous-time longitudinal estimates.",
    ("X6",),
    limits=LONGITUDINAL_LIMITS,
)
BAND_SUPPORT: Final[Support] = Support(
    "experimental",
    "No registered study measures the simultaneous band of the continuous-time longitudinal TMLE.",
    ("X6",),
    limits=LONGITUDINAL_LIMITS,
)


def few_followers_reason(regime: str, count: int, last: float) -> str:
    """The support reason of a regime with fewer followers than the study saw.

    Parameters
    ----------
    regime : str
        The regime's label.
    count : int
        The units that follow it at every decision visit and are at risk at the last target.
    last : float
        The last target time.

    Returns
    -------
    str
        The reason.
    """
    return (
        f"Only {count} units follow regime {regime!r} at every decision visit and are at risk "
        f"at {last:g}; their inverse weights are large, and every fit of the registered "
        f"{CONTINUOUS_LONGITUDINAL_STUDY} study had at least {LONGITUDINAL_FOLLOWERS}."
    )


def past_cap_reason(regime: str, cap: int, share: float) -> str:
    """The support reason of a regime with more followers past its visit cap than the study saw.

    Parameters
    ----------
    regime : str
        The regime's label.
    cap : int
        The visit cap :math:`K^*`.
    share : float
        The estimated share of followers past the cap before the estimate's time.

    Returns
    -------
    str
        The reason.
    """
    return (
        f"An estimated {share:.3f} of regime {regime!r}'s followers visit more than K* = {cap} "
        "times before the target time, which bounds the gap to the risk under the regime at "
        f"every visit; the registered {CONTINUOUS_LONGITUDINAL_STUDY} study measured shares up "
        f"to {LONGITUDINAL_PAST_CAP_SHARE:.3f}."
    )


def continuous_longitudinal_support(
    *,
    scale: str,
    n: int,
    causes: int,
    cross_fitted: bool,
    default_nuisances: bool,
    settings: Mapping[str, Any],
    event_share: float,
    variance_rel_sd: float,
    followers: Mapping[str, int],
    past_cap: Mapping[str, tuple[int, float]],
    last: float,
    bounded_share: float,
    jump_bounded: bool,
    visit_bounded: bool,
) -> Support:
    r"""The support label of one continuous-time longitudinal estimate.

    A risk or a risk difference reads ``supported`` when the registered study
    ``continuous-time-longitudinal`` measures its configuration: a cross-fitted fit with the
    default libraries and the settings of :data:`LONGITUDINAL_DEFAULTS`, two causes, at least
    :data:`LONGITUDINAL_N` units, static regimes or declared rules, a share of units with an
    event of the estimate's cause by its time inside :data:`LONGITUDINAL_EVENT_SHARE`, a
    ``variance_rel_sd`` at most :data:`LONGITUDINAL_REL_SD`, at least
    :data:`LONGITUDINAL_FOLLOWERS` followers of each regime at risk at the last target time,
    each regime's share past its visit cap at most :data:`LONGITUDINAL_PAST_CAP_SHARE`, a
    weight-floor share at most :data:`LONGITUDINAL_BOUNDED_SHARE`, and no bounded hazard or
    visit jump.  A ratio always reads ``experimental``.  Otherwise the estimate reads
    ``experimental`` with every reason that applies; a user learner never reads supported.

    Parameters
    ----------
    scale : {"level", "difference", "ratio"}
        The estimate's scale.
    n : int
        The number of units.
    causes : int
        The number of event types.
    cross_fitted : bool
        Whether the nuisances are fitted out of fold.
    default_nuisances : bool
        Whether every nuisance takes its default library (no library or learner given).
    settings : mapping of str to Any
        The fit's values of the keys of :data:`LONGITUDINAL_DEFAULTS`.
    event_share : float
        The share of units with an observed event of the estimate's cause by its time.
    variance_rel_sd : float
        The relative standard deviation of the estimate's influence-curve variance,
        :math:`\sqrt{(\hat\kappa - 1)/n}`.
    followers : mapping of str to int
        Per regime of the estimate, the units that follow it at every decision visit and are at
        risk at the last target time.
    past_cap : mapping of str to tuple of (int, float)
        Per regime of the estimate, its visit cap :math:`K^*` and the estimated share of its
        followers past the cap before the estimate's time.
    last : float
        The last target time.
    bounded_share : float
        The fit's share of weighted cells at risk raised to the weight floor.
    jump_bounded : bool
        Whether the engine bounded some hazard jump below one.
    visit_bounded : bool
        Whether the engine bounded some visit jump below one.

    Returns
    -------
    Support
        The estimate's support.

    Examples
    --------
    >>> from cleverly.survival.longitudinal_result import (
    ...     LONGITUDINAL_DEFAULTS, continuous_longitudinal_support,
    ... )
    >>> measured = dict(
    ...     n=1_000, causes=2, cross_fitted=True, default_nuisances=True,
    ...     settings=LONGITUDINAL_DEFAULTS, event_share=0.15, variance_rel_sd=0.15,
    ...     followers={"always": 100, "never": 60},
    ...     past_cap={"always": (4, 0.05), "never": (4, 0.1)},
    ...     last=30.0, bounded_share=0.0, jump_bounded=False, visit_bounded=False,
    ... )
    >>> continuous_longitudinal_support(scale="difference", **measured).evidence
    ('continuous-time-longitudinal',)
    >>> continuous_longitudinal_support(scale="ratio", **measured).label
    'experimental'
    """
    reasons: list[str] = []
    if scale == "ratio":
        reasons.append(RATIO_REASON)
    study = CONTINUOUS_LONGITUDINAL_STUDY
    if not default_nuisances:
        reasons.append(USER_LEARNER_REASON)
    changed = sorted(key for key, value in LONGITUDINAL_DEFAULTS.items() if settings[key] != value)
    if changed:
        reasons.append(
            f"The fit sets {', '.join(changed)} away from the default, and the registered "
            f"{study} study measures the defaults only."
        )
    if not cross_fitted:
        reasons.append(f"The fit is in sample, and the registered {study} study cross-fits.")
    if causes != LONGITUDINAL_CAUSES:
        reasons.append(
            f"The fit has {causes} cause{'' if causes == 1 else 's'}, and the registered "
            f"{study} study measures two."
        )
    if n < LONGITUDINAL_N:
        reasons.append(
            f"The fit has {n:,} units, and the registered {study} study measures n = "
            f"{LONGITUDINAL_N:,}."
        )
    low, high = LONGITUDINAL_EVENT_SHARE
    if not low <= event_share <= high:
        reasons.append(
            f"{event_share:.1%} of the units have an event of this cause by this time, outside "
            f"the shares from {low:.1%} to {high:.1%} at which the registered {study} study's "
            "gated rows sit."
        )
    if scale != "ratio" and not variance_rel_sd <= LONGITUDINAL_REL_SD:
        reasons.append(
            "The influence curve of this estimate is sparse: the relative standard deviation of "
            f"its variance is {variance_rel_sd:.3f}, above {LONGITUDINAL_REL_SD:.3f}, the "
            f"largest median at which the registered {study} study's gated rows pass."
        )
    for regime, count in followers.items():
        if count < LONGITUDINAL_FOLLOWERS:
            reasons.append(few_followers_reason(regime, count, last))
    for regime, (cap, share) in past_cap.items():
        if share > LONGITUDINAL_PAST_CAP_SHARE:
            reasons.append(past_cap_reason(regime, cap, share))
    if bounded_share > LONGITUDINAL_BOUNDED_SHARE:
        reasons.append(
            f"The weight floor raised {bounded_share:.2%} of the weighted cells at risk, above "
            f"the {LONGITUDINAL_BOUNDED_SHARE:.2%} of any fit of the registered {study} study."
        )
    if jump_bounded:
        reasons.append(JUMP_BOUND_REASON)
    if visit_bounded:
        reasons.append(VISIT_BOUND_REASON)
    if reasons:
        return Support("experimental", " ".join(reasons), ("X6",), limits=LONGITUDINAL_LIMITS)
    return Support(
        "supported",
        f"The registered {study} study measures this configuration.",
        (study,),
        limits=LONGITUDINAL_LIMITS,
    )


#: Refusal C9: a restricted mean of a longitudinal fit.
RESTRICTED_TEXT: Final[str] = (
    "A continuous-time longitudinal fit targets the risks at its target times only; a "
    "restricted mean is not one of its targets."
)


@dataclass(frozen=True)
class ContinuousLongitudinalResult(_TargetedRisks, Mapping[str, ParameterEstimate]):
    """The targeted risks of a continuous-time longitudinal fit.

    Behaves as a mapping from parameter name to
    :class:`~cleverly.inference.ParameterEstimate`. Every estimate reads its influence-curve
    variance. A risk or risk difference reads supported inside the configuration the registered
    study ``continuous-time-longitudinal`` measures (:func:`continuous_longitudinal_support`); a
    ratio, a smooth function and the band read experimental.

    Parameters
    ----------
    estimates : dict of str to ParameterEstimate
        Each regime's risk (``risk_regimen[always @ t=60]``, or
        ``cif_regimen[always, cause @ t=60]`` with competing causes), each other regime's
        difference from the reference (``ate_regimen[...]``) and their ratio
        (``rr_regimen[...]``).
    record : ContinuousLongitudinalRecord
        The nuisance selections, the visit caps and the targeting record.
    simultaneous : SimultaneousBands or None
        The sup-t band over every risk and risk difference, when requested.
    arms : tuple of str
        The regime labels, the reference first.
    causes : tuple of str
        The cause labels.
    horizons : tuple of float
        The target times.
    n : int
        The number of units.
    backend : str or None
        The dataframe library the data arrived in.
    estimator : ContinuousLongitudinalTMLE or None
        A copy of the estimator, with its seed resolved.
    data : ContinuousLongitudinalData or None
        The container the fit read.
    """

    estimates: dict[str, ParameterEstimate]
    record: ContinuousLongitudinalRecord
    simultaneous: Any
    arms: tuple[str, ...]
    causes: tuple[str, ...]
    horizons: tuple[float, ...]
    n: int
    backend: str | None = None
    estimator: ContinuousLongitudinalTMLE | None = field(default=None, repr=False)
    data: ContinuousLongitudinalData | None = field(default=None, repr=False)

    #: Assessment results keyed by operation and normalized arguments. ``init=False``, so a
    #: mapping handed to the constructor by ``dataclasses.replace`` cannot answer for
    #: another fit.
    assessment_cache: AssessmentCache = field(
        default_factory=AssessmentCache, init=False, repr=False, compare=False
    )

    #: The assessment family: one declaration per assessment operation in
    #: :data:`cleverly.assessment.ASSESSMENT_CAPABILITIES`.
    assessment_family: ClassVar[str] = CONTINUOUS_LONGITUDINAL_FAMILY
    #: The method identity of the result.
    fitted_method: ClassVar[str] = "tmle"
    _contrast_support: ClassVar[Support] = CONTRAST_SUPPORT
    _ratio_support: ClassVar[Support] = RATIO_SUPPORT

    def __getitem__(self, name: str) -> ParameterEstimate:
        return self.estimates[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.estimates)

    def __len__(self) -> int:
        return len(self.estimates)

    @property
    def initial_estimates(self) -> dict[str, float]:
        """Every risk and risk difference at the initial fits, before any step."""
        psi = self.record.initial_psi
        J, K = len(self.causes), len(self.horizons)
        single = J == 1
        # The record is in the order the regimes were given; the result puts the reference
        # first.
        slot = {label: r for r, label in enumerate(self.record.regimes)}

        def value(arm: str, j: int, k: int) -> float:
            return float(psi[(slot[arm] * J + j) * K + k])

        out: dict[str, float] = {}
        for arm in self.arms:
            for j, cause in enumerate(self.causes):
                for k, time in enumerate(self.horizons):
                    out[level_name(arm, None if single else cause, time)] = value(arm, j, k)
        reference = self.arms[0]
        for arm in self.arms[1:]:
            for j, cause in enumerate(self.causes):
                for k, time in enumerate(self.horizons):
                    name = contrast_name(
                        "ate_regimen", arm, reference, None if single else cause, time
                    )
                    out[name] = value(arm, j, k) - value(reference, j, k)
        return out

    @property
    def n_folds(self) -> int:
        """The number of cross-fitting folds, ``1`` for a full-sample fit."""
        return int(self.record.n_folds)

    def curve(
        self,
        scale: Literal["risk", "survival"] = "risk",
        *,
        times: Sequence[float] | None = None,
    ) -> Any:
        """The targeted risks and their differences: one row per regime, cause and time.

        Only the target times are reported, as for
        :meth:`ContinuousSurvivalResult.curve <cleverly.survival.ContinuousSurvivalResult.curve>`.

        Parameters
        ----------
        scale : {"risk", "survival"}
            The view of each row; ``"survival"`` needs a fit with one cause.
        times : sequence of float or None
            The target times to report. ``None`` reports every target time.

        Returns
        -------
        dataframe
            One row per regime, cause and time, then one per difference from the reference,
            in the input backend.

        Raises
        ------
        CapabilityError
            When a time is not a target time, or ``scale="survival"`` on a fit with several
            causes.

        See Also
        --------
        cleverly.survival.ContinuousLongitudinalResult.contrast : A smooth function of them.
        cleverly.survival.ContinuousLongitudinalResult.to_frame : Every estimate, untimed.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from cleverly.survival import ContinuousLongitudinalTMLE, CoxHazard
        >>> rng = np.random.default_rng(0)
        >>> rows = []
        >>> for unit in range(300):
        ...     w, l, a = rng.integers(2), rng.integers(2), rng.integers(2)
        ...     start, visit = 0.0, rng.exponential(3.0)
        ...     end = rng.exponential(1 / (0.1 * np.exp(0.5 * l - 0.5 * a)))
        ...     if visit < min(end, 8.0):
        ...         rows.append((unit, 0.0, visit, 0, w, l, a))
        ...         l = rng.integers(2)
        ...         a = int(rng.random() < 0.3 + 0.4 * l)
        ...         start = visit
        ...         end = visit + rng.exponential(1 / (0.1 * np.exp(0.5 * l - 0.5 * a)))
        ...     rows.append((unit, start, min(end, 8.0), int(end < 8.0), w, l, a))
        >>> frame = pd.DataFrame(rows, columns=["id", "start", "stop", "event", "w", "l", "a"])
        >>> cox = [CoxHazard("main")]
        >>> fit = ContinuousLongitudinalTMLE(
        ...     [2.0, 6.0],
        ...     regimes={"always": 1, "never": 0},
        ...     reference="never",
        ...     cross_fit=False,
        ...     hazard_library=cox,
        ...     visit_library=cox,
        ...     censoring_library=cox,
        ...     random_state=0,
        ... ).fit(
        ...     frame,
        ...     id="id",
        ...     start="start",
        ...     stop="stop",
        ...     event="event",
        ...     treatment="a",
        ...     baseline=["w"],
        ...     time_varying=["l"],
        ... )
        >>> curve = fit.curve()
        >>> list(curve["arm"])
        ['never', 'never', 'always', 'always', 'always vs never', 'always vs never']
        >>> list(curve["time"])
        [2.0, 6.0, 2.0, 6.0, 2.0, 6.0]
        """
        return self._curve(scale, times=times)

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

        Parameters
        ----------
        function : callable
            A smooth scalar function of the selected estimates, in the order of ``names``.
        names : sequence of str
            The reported estimates it reads.
        name : str or None
            The name of the derived estimate. ``None`` gives ``contrast(<names>)``.
        scale : str
            The reported scale of the derived estimate.
        gradient : callable or None
            The analytic gradient. ``None`` uses central differences.
        transform : Transform or None
            The map the interval and the test are computed on.

        Returns
        -------
        ParameterEstimate
            The derived estimate, labelled experimental.

        See Also
        --------
        cleverly.survival.ContinuousLongitudinalResult.ratio : Two regimes' risk ratio.
        cleverly.survival.ContinuousLongitudinalResult.curve : Every risk by regime and time.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from cleverly.survival import ContinuousLongitudinalTMLE, CoxHazard
        >>> rng = np.random.default_rng(0)
        >>> rows = []
        >>> for unit in range(300):
        ...     w, l, a = rng.integers(2), rng.integers(2), rng.integers(2)
        ...     start, visit = 0.0, rng.exponential(3.0)
        ...     end = rng.exponential(1 / (0.1 * np.exp(0.5 * l - 0.5 * a)))
        ...     if visit < min(end, 8.0):
        ...         rows.append((unit, 0.0, visit, 0, w, l, a))
        ...         l = rng.integers(2)
        ...         a = int(rng.random() < 0.3 + 0.4 * l)
        ...         start = visit
        ...         end = visit + rng.exponential(1 / (0.1 * np.exp(0.5 * l - 0.5 * a)))
        ...     rows.append((unit, start, min(end, 8.0), int(end < 8.0), w, l, a))
        >>> frame = pd.DataFrame(rows, columns=["id", "start", "stop", "event", "w", "l", "a"])
        >>> cox = [CoxHazard("main")]
        >>> fit = ContinuousLongitudinalTMLE(
        ...     [2.0, 6.0],
        ...     regimes={"always": 1, "never": 0},
        ...     reference="never",
        ...     cross_fit=False,
        ...     hazard_library=cox,
        ...     visit_library=cox,
        ...     censoring_library=cox,
        ...     random_state=0,
        ... ).fit(
        ...     frame,
        ...     id="id",
        ...     start="start",
        ...     stop="stop",
        ...     event="event",
        ...     treatment="a",
        ...     baseline=["w"],
        ...     time_varying=["l"],
        ... )
        >>> names = ["risk_regimen[always @ t=6]", "risk_regimen[always @ t=2]"]
        >>> later = fit.contrast(lambda x: x[0] - x[1], names, name="always 2 to 6")
        >>> bool(np.isclose(later.psi, fit[names[0]].psi - fit[names[1]].psi))
        True
        >>> later.support.label
        'experimental'
        """
        return self._contrast(
            function, names, name=name, scale=scale, gradient=gradient, transform=transform
        )

    def ratio(
        self,
        numerator: str,
        denominator: str,
        *,
        kind: Literal["rr", "or"] = "rr",
        view: Literal["risk", "survival"] = "risk",
        name: str | None = None,
    ) -> ParameterEstimate:
        """The risk ratio or odds ratio of two regimes' risks at one cause and time.

        Parameters
        ----------
        numerator : str
            The name of one regime's risk.
        denominator : str
            The name of another regime's risk at the same cause and time.
        kind : {"rr", "or"}
            A ratio of the risks, or a ratio of their odds.
        view : {"risk", "survival"}
            Whether to take the ratio of the risks or of their complements.
        name : str or None
            The name of the derived estimate.

        Returns
        -------
        ParameterEstimate
            The ratio, on the ``"ratio"`` scale with ``log_psi`` set, labelled experimental.
        """
        return self._ratio(numerator, denominator, kind=kind, view=view, name=name)

    def rmtl(self, *args: Any, **kwargs: Any) -> ParameterEstimate:
        """Refused (C9): a longitudinal fit targets no restricted mean.

        Parameters
        ----------
        *args : Any
            Ignored.
        **kwargs : Any
            Ignored.

        Returns
        -------
        ParameterEstimate
            Never returns.

        Raises
        ------
        CapabilityError
            Always, ``API_REDIRECTION``.
        """
        raise CapabilityError(
            RESTRICTED_TEXT,
            reason=RefusalReason.API_REDIRECTION,
            alternative="Read the risks at the target times with curve().",
        )

    def rmst(self, *args: Any, **kwargs: Any) -> ParameterEstimate:
        """Refused (C9): a longitudinal fit targets no restricted mean.

        Parameters
        ----------
        *args : Any
            Ignored.
        **kwargs : Any
            Ignored.

        Returns
        -------
        ParameterEstimate
            Never returns.

        Raises
        ------
        CapabilityError
            Always, ``API_REDIRECTION``.
        """
        return self.rmtl(*args, **kwargs)

    def to_frame(self) -> Any:
        """One row per parameter, in the backend the data arrived in.

        Returns
        -------
        dataframe
            The estimates with their standard errors and intervals.
        """
        return self._frame()

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready record of the fit.

        Returns
        -------
        dict of str to Any
            ``estimates``, each estimate's mapping by name; ``record``, the selections, the
            visit caps and how each node walk ended; and ``settings``, the regimes, causes,
            target times, units and significance level. A value that is not a finite number
            reads ``None``.
        """
        record = self.record
        propensity = record.propensity
        return _json_ready(
            {
                "estimates": {
                    name: estimate.to_dict() for name, estimate in self.estimates.items()
                },
                "record": {
                    "hazards": [selection.chosen_name for selection in record.hazards],
                    "visit": None if record.visit is None else record.visit.chosen_name,
                    "censoring": record.censoring.chosen_name,
                    "propensity": None
                    if propensity is None
                    else propensity.names[propensity.chosen],
                    "caps": [int(c) for c in record.caps],
                    "past_cap_share": [[float(v) for v in row] for row in record.past_cap_share],
                    "followers_at_last": [int(c) for c in record.followers_at_last],
                    "steps": [[int(s) for s in row] for row in record.steps],
                    "converged": bool(record.converged),
                    "cutoff": float(record.cutoff),
                    "standardized_score": [float(v) for v in record.standardized_score],
                    "bounded_share": float(record.bounded_share),
                    "jump_bound_share": float(record.jump_bound_share),
                    "visit_bound_share": float(record.visit_bound_share),
                    "history_restriction": record.history_restriction,
                    "cross_fitted": bool(record.cross_fitted),
                    "n_folds": int(record.n_folds),
                    "grid_size": int(record.grid_size),
                    "dropped_visits": int(record.dropped_visits),
                },
                "settings": {
                    "regimes": list(self.arms),
                    "causes": list(self.causes),
                    "horizons": [float(t) for t in self.horizons],
                    "n": int(self.n),
                    "alpha": self._alpha,
                    "variance": "influence_curve",
                },
            }
        )

    def __repr__(self) -> str:
        return (
            f"ContinuousLongitudinalResult(parameters={len(self.estimates)}, n={self.n}, "
            f"regimes={len(self.arms)}, causes={len(self.causes)}, times={len(self.horizons)})"
        )

    @property
    def bootstrap_band(self) -> None:
        """``None``: a longitudinal fit runs no bootstrap."""
        return None

    @property
    def restricted_horizons(self) -> tuple[float, ...]:
        """``()``: a longitudinal fit targets no restricted mean (C9)."""
        return ()

    def summary(self, *, detail: Literal["short", "full"] = "short") -> Any:
        """A printable report of the fit.

        Parameters
        ----------
        detail : {"short", "full"}
            ``"full"`` adds the assumptions and how the node walks ended.

        Returns
        -------
        Summary
            The report.
        """
        from .._summary._continuous_longitudinal import continuous_longitudinal_summary

        return continuous_longitudinal_summary(self, detail=detail)

    def _repr_html_(self) -> str:
        return str(self.summary().as_html())

    def plot(self, **kwargs: Any) -> Any:
        """Plot each regime's risk and its difference from the reference at the target times.

        Each estimate is drawn at its target time with its interval and, when the fit holds
        one, its simultaneous band. Nothing is drawn between target times, where the fit
        estimates no risk.

        Parameters
        ----------
        **kwargs : Any
            Passed to :func:`cleverly.plotting.curve` (``scale=``, ``band=``, ``cause=``,
            ``regimens=``, ``labels=``, ``units=``, ``time_label=``, ``truth=`` and
            ``title=``).

        Returns
        -------
        Figure
            The levels above and the differences below, one column per cause.

        Raises
        ------
        CapabilityError
            When the fit refuses the requested view, such as ``measure="rmst"`` (C9).
        ValueError
            When ``band=``, ``scale=`` or ``cause=`` do not fit together or the fit.
        DataError
            When ``cause=`` or ``regimens=`` names a cause or regime the fit does not hold.
        """
        from ..plotting import curve

        return curve(self, **kwargs)

    # ------------------------------------------------------------- assessment

    @property
    def diagnostics(self) -> Any:
        """The support, score, nuisance and influence diagnostics of the fit.

        The support, score and nuisance reports read :attr:`record`; the refutation and the
        truncation curve refuse (roadmap X105 part (d)).
        """
        from ..assessment import DiagnosticsFacade

        return DiagnosticsFacade(self)

    @property
    def sensitivity(self) -> Any:
        """The sensitivity-analysis facade, whose every row refuses with an inference gap."""
        from ..assessment import SensitivityFacade

        return SensitivityFacade(self)

    @property
    def replayability(self) -> Any:
        """Which assessment operations survive from the stored fit."""
        from ..assessment import replayability

        return replayability(self)

    def validate(self) -> Any:
        """Run the inexpensive default battery without refitting.

        Returns
        -------
        ValidationReport
            The score-equation, support and nuisance checks, read off :attr:`record`.
        """
        from ..assessment import validate_result

        return validate_result(self)

    def assess(
        self,
        *,
        include_refits: bool = False,
        include_retargets: bool = False,
        arguments: Mapping[str, Mapping[str, Any]] | None = None,
        random_state: int | None = None,
    ) -> Any:
        """Run the applicable post-fit assessment battery.

        The validation rows read :attr:`record`. The refutation and the truncation curve
        report their refusal (roadmap X105 part (d)), and so does every sensitivity row.

        Parameters
        ----------
        include_refits : bool
            Run requested operations that refit nuisance models; none runs on this fit.
        include_retargets : bool
            Include moderate retargets; none runs on this fit.
        arguments : mapping or None
            Per-operation keyword arguments.
        random_state : int or None
            Common seed for stochastic refit operations.

        Returns
        -------
        AssessmentReport
            Validation, diagnostics, and sensitivity in one report.
        """
        from ..assessment import assess_result

        return assess_result(
            self,
            include_refits=include_refits,
            include_retargets=include_retargets,
            arguments=arguments,
            random_state=random_state,
        )

    def refute(self, **kwargs: Any) -> Any:
        """Refused: no adapter refits the node passes (roadmap X105 part (d)).

        Parameters
        ----------
        **kwargs : Any
            Ignored.

        Returns
        -------
        Any
            Never returns.

        Raises
        ------
        CapabilityError
            Always, ``IMPLEMENTATION_GAP``.
        """
        raise CapabilityError.from_refusal(replay_refusal("refute()"))

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


def build_result(
    fit: Any,
    psis: Sequence[FloatArray],
    curves: Sequence[FloatArray],
    record: ContinuousLongitudinalRecord,
    *,
    jump_bounded: bool,
) -> ContinuousLongitudinalResult:
    """Assemble the result of one fit from each regime's targeted values and curves.

    Parameters
    ----------
    fit : longitudinal._Fit
        The fit: its estimator, container, causes and seed.
    psis : sequence of ndarray
        Per regime, ``(J K,)`` the targeted risks, cause-major.
    curves : sequence of ndarray
        Per regime, ``(n, J K)`` the influence curves.
    record : ContinuousLongitudinalRecord
        The record.
    jump_bounded : bool
        Whether the engine bounded a hazard jump.

    Returns
    -------
    ContinuousLongitudinalResult
        The result, with the reference regime first.
    """
    est, data = fit.est, fit.data
    labels = [regime.label for regime in est._regimes]
    order = [labels.index(est.reference)] + [
        r for r, label in enumerate(labels) if label != est.reference
    ]
    arms = tuple(labels[r] for r in order)
    causes = tuple(fit.causes.values())
    J, K = len(causes), len(est.horizons)
    single = J == 1
    n = int(data.n)
    alpha = est.alpha
    last = est.horizons[-1]
    # The label reads the default libraries (plan 3.5.9), not any shipped learner class: a
    # one-member Cox library is the user's choice, which a study of the defaults does not
    # measure. ``_default_nuisances_only`` gates C13 alone.
    default = est._defaults
    settings = {key: getattr(est, key) for key in LONGITUDINAL_DEFAULTS}
    codes = list(fit.causes)
    event_time = np.asarray(data.time, dtype=float)
    event_code = np.asarray(data.event)

    def support(name: str, curve: FloatArray, scale: str, j: int, k: int, *regimes: int) -> Support:
        share = float(np.mean((event_code == codes[j]) & (event_time <= est.horizons[k])))
        return continuous_longitudinal_support(
            scale=scale,
            n=n,
            causes=J,
            cross_fitted=est.cross_fit,
            default_nuisances=default,
            settings=settings,
            event_share=share,
            variance_rel_sd=influence_row(name, curve).variance_rel_sd,
            followers={labels[r]: record.followers_at_last[r] for r in regimes},
            past_cap={labels[r]: (record.caps[r], record.past_cap_share[r][k]) for r in regimes},
            last=last,
            bounded_share=float(record.bounded_share),
            jump_bounded=jump_bounded,
            visit_bounded=record.visit_bound_share > 0,
        )

    def build(
        name: str, value: float, curve: FloatArray, scale: str, sup: Support, **kw: Any
    ) -> ParameterEstimate:
        return make_estimate(
            name,
            float(value),
            curve,
            n=n,
            scale=scale,  # type: ignore[arg-type]
            alpha=alpha,
            rule_widened=False,
            support=sup,
            **kw,
        )

    estimates: dict[str, ParameterEstimate] = {}
    for r in order:
        for j, cause in enumerate(causes):
            for k, time in enumerate(est.horizons):
                p = j * K + k
                name = level_name(labels[r], None if single else cause, time)
                curve = curves[r][:, p]
                estimates[name] = build(
                    name,
                    psis[r][p],
                    curve,
                    "level",
                    support(name, curve, "level", j, k, r),
                    bounds=(0.0, 1.0),
                )
    reference = order[0]
    for r in order[1:]:
        for j, cause in enumerate(causes):
            for k, time in enumerate(est.horizons):
                p = j * K + k
                label = None if single else cause
                psi0, psi1 = psis[reference][p], psis[r][p]
                ic0, ic1 = curves[reference][:, p], curves[r][:, p]
                name = contrast_name("ate_regimen", labels[r], labels[reference], label, time)
                curve = ic1 - ic0
                estimates[name] = build(
                    name,
                    psi1 - psi0,
                    curve,
                    "difference",
                    support(name, curve, "difference", j, k, r, reference),
                    bounds=(-1.0, 1.0),
                )
                if psi0 > 0 and psi1 > 0:
                    name = contrast_name("rr_regimen", labels[r], labels[reference], label, time)
                    curve = ic1 / psi1 - ic0 / psi0
                    estimates[name] = build(
                        name,
                        psi1 / psi0,
                        curve,
                        "ratio",
                        support(name, curve, "ratio", j, k, r, reference),
                        log_psi=float(np.log(psi1 / psi0)),
                    )
    bands = None
    if est.simultaneous:
        from ..inference.multiplier import simultaneous_bands

        banded = {k: v for k, v in estimates.items() if not k.startswith("rr_regimen[")}
        bands = simultaneous_bands(banded, alpha=alpha, random_state=fit.seed)
        bands = replace(bands, support=BAND_SUPPORT)
    return ContinuousLongitudinalResult(
        estimates=estimates,
        record=record,
        simultaneous=bands,
        arms=arms,
        causes=causes,
        horizons=tuple(est.horizons),
        n=n,
        backend=data.backend,
        estimator=est._copy(fit.seed),
        data=data,
    )
