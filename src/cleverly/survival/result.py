"""The result of a continuous-time survival fit."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal

import numpy as np

from .._assessment_cache import AssessmentCache
from .._continuous_survival_family import CONTINUOUS_SURVIVAL_FAMILY
from .._refusal import RefusalReason
from .._support import Support, combine_support
from .._typing import FloatArray
from ..exceptions import CapabilityError
from ..inference.delta import Transform
from ..inference.influence import ParameterEstimate, Scale, make_estimate
from ..inference.results import ratio_contrast, smooth_contrast
from ..inference.sparse import influence_row
from ..inference.switched import PredictableCovariance
from ..utils.frames import emit_frame

#: The registered study that measures :class:`~cleverly.survival.ContinuousSurvivalTMLE`.
CONTINUOUS_SURVIVAL_STUDY: Final[str] = "continuous-time-survival"

#: The share of the observed events (any cause) at or before a target time that the study's
#: gated times reach.  Its gated times sit at the median and the upper quartile of the observed
#: event times (shares 0.5 and 0.75).  Its lower quartile (0.25), where the influence-curve
#: standard error read up to 8% below the sampling spread and coverage fell to 0.88, is published
#: without a verdict; under the switched default it covers 0.932 to 0.961 on this law, but the
#: share floor stays at :data:`SHARE_FLOOR_DEFAULT` because 8 of the 30 first-quartile rows of
#: :data:`VARIANCE_STUDY` fail its calibration rule.  The band leaves at least five sampling
#: standard deviations at n = 1,000 between each gated time and its edge, and so reaches past the
#: measured shares (about 0.44 to 0.80 over the study's draws).  It is calibrated on one law: it
#: reads neither the events of each arm and cause nor n, which :data:`MEASURED_N` bounds.
MEASURED_EVENT_SHARE: Final[tuple[float, float]] = (0.4, 0.85)

#: The smallest sample size at which a risk or risk difference of a fit without
#: ``restricted_horizons=`` reads supported.  The registered study gates n = 1,000, where at the
#: early time the cell with the fewest events (about 16 per fit) covered least; the gated
#: 500-unit cell of :data:`VARIANCE_STUDY` (``v3_half_n``) measured these risks and differences
#: and passed every primary row, which lowers their floor to 500.  The design test checks it
#: against that study's committed ``floors.csv``.  A fit with restricted horizons keeps
#: :data:`FUNCTIONALS_N`.
MEASURED_N: Final[int] = 500

#: The smallest sample size at which a risk or risk difference of an in-sample fit
#: (``cross_fit=False``) reads supported: the n of :data:`VARIANCE_STUDY`'s gated in-sample cell
#: (``v4_in_sample``), whose every primary row passed (Eric, 2026-11-25).  That cell fits two
#: causes and no restricted horizon, so an in-sample fit with one or three causes or a restricted
#: horizon reads experimental.  The design test checks it against that study's record.
IN_SAMPLE_N: Final[int] = 1_000

#: The smallest sample size at which an estimate of a fit with ``restricted_horizons=`` reads
#: supported: the n of :data:`FUNCTIONALS_STUDY`'s one cell.  The variance study's 500-unit cell
#: fits no restricted horizon, and the horizons move every risk of the fit, so its floor does not
#: carry over.  The functionals design test checks it against that study's record.
FUNCTIONALS_N: Final[int] = 1_000

#: The study that measures the default interval's sparse-cell floors (X6 part (a)).
VARIANCE_STUDY: Final[str] = "continuous-survival-variance"

#: The largest relative standard deviation of the influence-curve variance (``variance_rel_sd``)
#: at which a risk or risk difference reads supported: :data:`VARIANCE_STUDY`'s
#: :math:`r^*_{\mathrm{prim}}`, the largest median rung such that every primary row at or below
#: it passes, over its gated cells and its reported heavy-weight cell (``v2_heavy_weights``),
#: whose primary rows set the value; read from that study's committed ``floors.csv`` (the design
#: test checks the value; the run at 0cb80a56 measured it).
MEASURED_REL_SD: Final[float | None] = 0.24582894675266365

#: The same floor for event shares in ``[0.25, 0.40)``, :math:`r^*_{\mathrm{cal}}`, read from the
#: first-quartile rows under the calibration rule.  It applies only if the study lowers the
#: share floor to 0.25 (every first-quartile row passing), and is ``None`` otherwise.
MEASURED_REL_SD_EARLY: Final[float | None] = None

#: The share floor below which no relative-SD floor applies: the lower edge of
#: :data:`MEASURED_EVENT_SHARE` before the study.
SHARE_FLOOR_DEFAULT: Final[float] = 0.4

#: The limits every continuous-time estimate states, supported or not.
CONTINUOUS_LIMITS: Final[tuple[str, ...]] = (
    "One law (van Hage et al. 2025, scenario 1, with an arm-specific term): a three-level factor "
    "and a normal covariate, a binary treatment, two causes with constant hazards and "
    "independent censoring, n = 1,000, the default libraries, cross-fitted. The double "
    "robustness cells add covariate-dependent censoring. At n = 4,000 the same fit covered "
    "0.918 to 0.963 over 400 draws (reported, not gated).",
    "Under near-violations of positivity (treatment probabilities near 0 or 1) the intervals "
    "under-cover badly: a reported cell at n = 4,000 covered 0.36 to 0.96.",
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
    "variance": "switched",
}


#: The registered study that measures a fit with ``restricted_horizons=``: its risks, risk
#: differences, risk ratios, restricted means and their differences.
FUNCTIONALS_STUDY: Final[str] = "continuous-survival-functionals"

#: The share of the observed events (any cause) at or before a restriction time inside which a
#: restricted mean reads supported.  The study restricts at the last target time, by which 75% of
#: the observed events have happened on average (0.69 to 0.80 over 2,000 of its draws, sampling
#: SD 0.016 at n = 1,000).  The lower edge sits five sampling SDs below that mean, as the risks'
#: band does below its gated times; a restricted mean at an earlier restriction time weights the
#: sparse early times more heavily, where the risks' standard error ran low, so it reads
#: experimental.  The upper edge is the risks' (:data:`MEASURED_EVENT_SHARE`).
RESTRICTED_EVENT_SHARE: Final[tuple[float, float]] = (0.67, 0.85)

#: The limits a fit with restricted means states beside :data:`CONTINUOUS_LIMITS`.
FUNCTIONALS_LIMITS: Final[tuple[str, ...]] = (
    "One restriction time, at the last target time (2.32, the upper quartile of the event "
    "times), with three target times. The restricted means read supported at restriction times "
    "by which 67% to 85% of the observed events have happened; a restriction time past the last "
    "target time extends the grid of the walk, which only the exact-law tests measure.",
)

#: The study that measures a fit with one cause and a fit with three (X6 part (b)).
CAUSES_STUDY: Final[str] = "continuous-survival-causes"

#: The numbers of causes at which a risk or risk difference can read supported.
#: :data:`CONTINUOUS_SURVIVAL_STUDY` measures two and :data:`CAUSES_STUDY` one and three.  A
#: number of causes is in this set only when every gated row and every property cell of its law
#: pass in that study's committed record (``performance-tests.csv`` and ``properties.csv``),
#: which the study's design test checks; the declared run at 820bdc39 passed both laws.
MEASURED_CAUSES: Final[frozenset[int]] = frozenset({1, 2, 3})

#: The most causes any study measures.
CAUSES_CAP: Final[int] = 3

#: The sample size :data:`CAUSES_STUDY` measures; a fit with one cause or three below it reads
#: experimental, since :data:`MEASURED_N` was measured with two causes.
CAUSES_N: Final[int] = 1_000

#: The law :data:`CAUSES_STUDY` draws for each number of causes it measures.
_CAUSES_LAWS: Final[Mapping[int, str]] = {
    1: "the registered law with its cause-2 hazard removed",
    3: "the registered law with a third hazard, constant in time, added",
}


def _causes_measurement(causes: int) -> str:
    """What :data:`CAUSES_STUDY` fits for ``causes`` (one or three), as a clause."""
    return (
        f"The {CAUSES_STUDY} study measures {_causes_noun(causes)} on {_CAUSES_LAWS[causes]}, "
        f"at n = {CAUSES_N:,}, cross-fitted with the default libraries, gated at the median and "
        "upper quartile of the law's event times"
    )


def _causes_limits(causes: int) -> tuple[str, ...]:
    """The limit a fit with ``causes`` other than two states beside :data:`CONTINUOUS_LIMITS`."""
    if causes > CAUSES_CAP:
        return (
            f"No study measures {_causes_noun(causes)}: {CONTINUOUS_SURVIVAL_STUDY} measures two "
            f"causes, and {CAUSES_STUDY} one and three.",
        )
    if causes in MEASURED_CAUSES:
        return (f"{_causes_measurement(causes)}.",)
    return (
        f"{_causes_measurement(causes)}, and its committed record does not show every gated row "
        "and property cell of that law passing.",
    )


#: Why a ratio of two continuous-time risks reads experimental in a fit without restricted means.
RATIO_REASON: Final[str] = (
    "The registered continuous-survival-functionals study measures the risk ratio of a fit "
    "with one restricted horizon; no registered study measures the risk ratio of a fit "
    "without one, or an odds ratio."
)


def _experimental(reason: str, limits: tuple[str, ...] = CONTINUOUS_LIMITS) -> Support:
    return Support("experimental", reason, ("X6",), limits=limits)


#: Numbers of causes spelled out in the support texts.
_NUMBER_WORDS: Final[tuple[str, ...]] = ("no", "one", "two", "three", "four", "five", "six")


def _number(count: int) -> str:
    return _NUMBER_WORDS[count] if count < len(_NUMBER_WORDS) else f"{count:,}"


def _causes_noun(causes: int) -> str:
    return f"{_number(causes)} cause{'' if causes == 1 else 's'}"


def _causes_reasons(causes: int, functionals: bool, n: int) -> list[str]:
    """Why a fit's number of causes keeps its estimates experimental, if it does."""
    noun = _causes_noun(causes)
    if causes == 2:
        return []
    if causes > CAUSES_CAP:
        return [
            f"The fit has {noun}, and no study measures more than {_number(CAUSES_CAP)}: "
            f"{CONTINUOUS_SURVIVAL_STUDY} measures two, and {CAUSES_STUDY} measures one and "
            "three. With more causes each has fewer events and the walk targets more risks at "
            "once."
        ]
    if causes not in MEASURED_CAUSES:
        return [
            f"The fit has {noun}, and the committed record of the {CAUSES_STUDY} study does "
            "not show every gated row and property cell of that law passing."
        ]
    if functionals:
        return [
            f"The fit has {noun} and a restricted horizon; the registered {FUNCTIONALS_STUDY} "
            "study measures restricted horizons with two causes only."
        ]
    if n < CAUSES_N:
        return [
            f"The fit has {noun} and {n:,} units; the registered {CAUSES_STUDY} study measures "
            f"{noun} at n = {CAUSES_N:,} only, and the floor of {MEASURED_N:,} units was "
            "measured with two causes."
        ]
    return []


def jump_bound_reason(pairs: int) -> str:
    """Why a fit whose initial hazards the engine bounded reads experimental.

    Parameters
    ----------
    pairs : int
        The (unit, arm) pairs with a cell the bound scaled.

    Returns
    -------
    str
        The reason.
    """
    noun = "pair" if pairs == 1 else "pairs"
    return (
        f"The hazard fit put a combined jump of one or more at {pairs:,} (unit, arm) {noun}, "
        "which the engine scaled below one. With distinct event times the hazard model "
        "extrapolates there, usually at a unit's counterfactual arm for covariates outside "
        "that arm's range, a positivity signal. No registered study gates a fit in which this "
        "bound acted."
    )


def _in_sample_reasons(n: int, causes: int, functionals: bool) -> list[str]:
    """Why an in-sample estimate falls outside :data:`VARIANCE_STUDY`'s in-sample cell.

    Parameters
    ----------
    n : int
        The number of units.
    causes : int
        The number of event types.
    functionals : bool
        Whether the fit targets a restricted horizon.

    Returns
    -------
    list of str
        One reason when the fit differs from the cell, none when it matches it.
    """
    differs = []
    if n < IN_SAMPLE_N:
        differs.append(f"has {n:,} units")
    if causes != 2:
        differs.append(f"has {causes} {'cause' if causes == 1 else 'causes'}")
    if functionals:
        differs.append("targets a restricted horizon")
    if not differs:
        return []
    return [
        f"The fit is in-sample and {' and '.join(differs)}. The registered {VARIANCE_STUDY} "
        f"study's in-sample cell measures in-sample fits with two causes, no restricted horizon "
        f"and n = {IN_SAMPLE_N:,} only, where every primary row passed at coverage 0.937 to "
        "0.951; the other registered studies gate cross-fitted fits."
    ]


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
    variance_rel_sd: float | None = None,
    past_last_time: bool = False,
    jump_bound_pairs: int = 0,
) -> Support:
    """The support of one continuous-time estimate.

    A risk or a risk difference of a fit without ``restricted_horizons=`` reads ``supported``
    when the registered ``continuous-time-survival`` study measures its configuration: two
    causes, at a time by which between 40% and 85% of the observed events have happened
    (:data:`MEASURED_EVENT_SHARE`), from a cross-fitted fit with the default libraries and the
    settings of :data:`MEASURED_DEFAULTS`, with at least :data:`MEASURED_N` units and its
    influence curve's relative SD at most :data:`MEASURED_REL_SD`, the two floors the registered
    ``continuous-survival-variance`` study measures, so its evidence names both studies.  A fit
    with one cause or three (:data:`MEASURED_CAUSES`) reads ``supported`` under the same
    conditions with at least :data:`CAUSES_N` units, the size the registered
    ``continuous-survival-causes`` study measures, and names that study in place of
    ``continuous-time-survival``; more than :data:`CAUSES_CAP` causes always reads experimental.
    A fit with one restricted horizon, at or before its last target time, is measured by the
    registered ``continuous-survival-functionals`` study under the same conditions but with two
    causes and at least :data:`FUNCTIONALS_N` units: its risks, risk differences and risk
    ratios, and its restricted means and their differences at a restriction time by which
    between 67% and 85% of the observed events have happened (:data:`RESTRICTED_EVENT_SHARE`).
    Its risks and differences keep the relative-SD floor and name the variance study too.
    An in-sample fit (``cross_fitted=False``) is measured by the registered
    ``continuous-survival-variance`` study's in-sample cell only: its risks and risk differences
    read ``supported`` under the conditions above with two causes, no restricted horizon and at
    least :data:`IN_SAMPLE_N` units, and name that study alone.
    A fit whose initial hazards the engine bounded (``jump_bound_pairs``) reads experimental:
    no registered study gates a fit in which the bound acted.  Otherwise the estimate reads
    ``experimental`` with every reason that applies.

    Parameters
    ----------
    scale : {"level", "difference", "ratio"}
        The estimate's scale.
    event_share : float
        The share of the observed events, any cause, at or before the estimate's time.
    n : int
        The number of units.
    causes : int
        The number of event types.  More than :data:`CAUSES_CAP` always reads experimental.
    cross_fitted : bool
        Whether the nuisances are fitted out of fold.  An in-sample fit needs two causes, no
        restricted horizon and :data:`IN_SAMPLE_N` units.
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
        :data:`MEASURED_DEFAULTS`: the functionals study measures one.
    variance_rel_sd : float or None
        The relative standard deviation of the estimate's influence-curve variance,
        :math:`\\sqrt{(\\hat\\kappa - 1)/n}`.  A risk or risk difference above
        :data:`MEASURED_REL_SD` reads experimental (above :data:`MEASURED_REL_SD_EARLY` at an
        event share below :data:`SHARE_FLOOR_DEFAULT`).  Ratios and restricted means keep their
        own labels: the floor's study measures risks and risk differences only.
    past_last_time : bool
        Whether a restriction time lies past the fit's last target time, which extends the
        walk's grid beyond it.  The functionals study restricts at the last target time.
    jump_bound_pairs : int
        The (unit, arm) pairs whose initial cause hazard jumps summed to one or more at some
        time and that the engine scaled below one
        (:attr:`~cleverly.survival.ContinuousSurvivalRecord.jump_bound_share`).

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
    >>> continuous_survival_support(
    ...     scale="level", event_share=0.75, n=1_000, causes=2, cross_fitted=True,
    ...     default_nuisances=True, settings=MEASURED_DEFAULTS, restricted=True,
    ...     restricted_horizons=1,
    ... ).evidence
    ('continuous-survival-functionals',)
    >>> continuous_survival_support(
    ...     scale="level", event_share=0.6, n=1_000, causes=2, cross_fitted=False,
    ...     default_nuisances=True, settings=MEASURED_DEFAULTS,
    ... ).evidence
    ('continuous-survival-variance',)
    """
    reasons: list[str] = []
    functionals = restricted_horizons > 0
    if scale == "ratio" and not functionals:
        reasons.append(RATIO_REASON)
    if restricted_horizons > 1:
        reasons.append(
            f"The fit targets {restricted_horizons} restricted horizons; the registered "
            f"{FUNCTIONALS_STUDY} study targets one."
        )
    if functionals and past_last_time:
        reasons.append(
            "A restricted horizon lies past the last target time, which extends the walk's grid "
            f"beyond it; the registered {FUNCTIONALS_STUDY} study restricts at the last target "
            "time, and only the exact-law tests measure the extension."
        )
    low, high = RESTRICTED_EVENT_SHARE if restricted else MEASURED_EVENT_SHARE
    if restricted and event_share < low:
        reasons.append(
            f"Only {event_share:.0%} of the observed events happen by this restriction time. "
            f"The registered {FUNCTIONALS_STUDY} study restricts where about 75% have happened, "
            f"and a restricted mean reads supported from {low:.0%} to {high:.0%}; an earlier "
            "restriction time weights the sparse early times, where the standard error ran "
            "low, more heavily."
        )
    elif event_share < low:
        reasons.append(
            f"Only {event_share:.0%} of the observed events happen by this time. The "
            f"registered study gates times by which {low:.0%} to {high:.0%} have happened, and "
            f"8 of the 30 rows of the {VARIANCE_STUDY} study at its earliest time (25%) fail its "
            "calibration rule, most with a standard error that runs low."
        )
    elif event_share > high:
        reasons.append(
            f"{event_share:.0%} of the observed events happen by this time, beyond the measured "
            f"band ({low:.0%} to {high:.0%}); the registered study's latest gated time sits at "
            "75%."
        )
    floor = MEASURED_REL_SD_EARLY if event_share < SHARE_FLOOR_DEFAULT else MEASURED_REL_SD
    floored = floor is not None and not restricted and scale in ("level", "difference")
    if (
        floored
        and floor is not None
        and variance_rel_sd is not None
        and not variance_rel_sd <= floor
    ):
        reasons.append(
            f"The influence curve of this estimate is sparse: the relative standard deviation of "
            f"its variance is {variance_rel_sd:.3f}, above {floor:.3f}, the largest at which "
            f"every primary row of the {VARIANCE_STUDY} study's cells passes, its reported "
            "heavy-weight cell included."
        )
    if functionals and n < FUNCTIONALS_N:
        reasons.append(
            f"The fit has {n:,} units, and the registered {FUNCTIONALS_STUDY} study measures a "
            f"fit with a restricted horizon at n = {FUNCTIONALS_N:,} only; a smaller sample has "
            "fewer events of each arm and cause."
        )
    elif not functionals and n < MEASURED_N and causes not in MEASURED_CAUSES - {2}:
        # A measured fit with one cause or three states its own floor of CAUSES_N, which
        # names this one, among the causes reasons.
        reasons.append(
            f"The fit has {n:,} units, and the registered {VARIANCE_STUDY} study measures "
            f"risks and risk differences down to n = {MEASURED_N:,}; a smaller sample has fewer "
            "events of each arm and cause."
        )
    reasons.extend(_causes_reasons(causes, functionals, n))
    if jump_bound_pairs:
        reasons.append(jump_bound_reason(jump_bound_pairs))
    if not cross_fitted:
        reasons.extend(_in_sample_reasons(n, causes, functionals))
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
    limits = (*CONTINUOUS_LIMITS, *FUNCTIONALS_LIMITS) if functionals else CONTINUOUS_LIMITS
    if causes != 2:
        limits = (*limits, *_causes_limits(causes))
    if reasons:
        return _experimental(" ".join(reasons), limits)
    if functionals:
        study = FUNCTIONALS_STUDY
    elif causes != 2:
        study = CAUSES_STUDY
    else:
        study = CONTINUOUS_SURVIVAL_STUDY
    if not cross_fitted:
        # Only the variance study's in-sample cell measures an in-sample fit: two causes, no
        # restricted horizon, n = IN_SAMPLE_N, so the estimate here is a risk or a difference.
        return Support(
            "supported",
            f"The registered {VARIANCE_STUDY} study's in-sample cell measures this "
            f"configuration at n = {IN_SAMPLE_N:,} and sets the floor on the sparsity of its "
            "influence curve.",
            (VARIANCE_STUDY,),
            limits=limits,
        )
    # The variance study measures risks and differences at n = 500 and sets their relative-SD
    # floor, so it is evidence for each of them; restricted means and ratios it does not measure.
    evidence = (study, VARIANCE_STUDY) if floored else (study,)
    measured = f"The registered {study} study measures this configuration."
    if floored:
        measured += (
            f" The registered {VARIANCE_STUDY} study sets the floor on the sparsity of its "
            "influence curve"
            + (
                "."
                if functionals or causes != 2
                else f" and measures it down to n = {MEASURED_N:,}."
            )
        )
    return Support("supported", measured, evidence, limits=limits)


#: The support of a simultaneous band: the study measures pointwise intervals only.
BAND_SUPPORT: Final[Support] = _experimental(
    "No registered study measures the simultaneous band of the continuous-time TMLE; the "
    "study measures pointwise intervals only."
)

#: The study that measures the full-refit bootstrap of the continuous-time TMLE (X6 part (d)).
BOOTSTRAP_STUDY: Final[str] = "continuous-survival-bootstrap"

#: The share of failed bootstrap refits above which every bootstrap interval of a fit is withheld
#: (the failure rule): the interval would condition on the replicates that fit.
BOOTSTRAP_FAILURE_SHARE: Final[float] = 0.01

#: Why a bootstrap interval of an in-sample fit reads experimental (X6 plan 8.9): the theory its
#: label would rest on has not been read first-hand, whatever the study reads.
IN_SAMPLE_BOOTSTRAP_REASON: Final[str] = (
    "The bootstrap delta-method conditions are not yet read first-hand. The in-sample bootstrap "
    "of this fit rests on the bootstrap delta method for a Hadamard-differentiable map (van der "
    "Vaart and Wellner 1996, Section 3.9, relayed through Beutner and Zähle), "
    f"and until its conditions are read the {BOOTSTRAP_STUDY} study cannot set this label alone."
)

#: Why a bootstrap interval of a cross-fitted fit reads experimental: no theorem covers it, so no
#: study can set its label.
CROSS_FITTED_BOOTSTRAP_REASON: Final[str] = (
    "No theorem covers a bootstrap of a cross-fitted fit with a selected library; Cai and van der "
    "Laan (2020) warn against it: a cross-validation selector rerun on each bootstrap sample makes "
    "the nonparametric bootstrap generally inconsistent, and can make its intervals too narrow."
)

#: The sentence a studentized bootstrap band adds to its kind's reason.
BOOTSTRAP_BAND_REASON: Final[str] = (
    "No study measures the studentized bootstrap band of the continuous-time TMLE."
)

BootstrapKind = Literal["bootstrap", "bootstrap_bca", "bootstrap_band"]


def continuous_bootstrap_support(
    *, kind: BootstrapKind, cross_fitted: bool, jump_bounded: int = 0, requested: int = 0
) -> Support:
    """The support of a bootstrap interval or band of a continuous-time fit.

    No bootstrap interval of this engine reads ``supported``.  An in-sample fit's percentile
    and BCa intervals wait on a first-hand reading of the bootstrap delta method's conditions
    (:data:`IN_SAMPLE_BOOTSTRAP_REASON`), and a cross-fitted fit's on a theorem that covers them
    (:data:`CROSS_FITTED_BOOTSTRAP_REASON`); the ``continuous-survival-bootstrap`` study measures
    both and sets neither label.  The studentized band adds that no study measures it.

    Parameters
    ----------
    kind : {"bootstrap", "bootstrap_bca", "bootstrap_band"}
        The percentile interval, the BCa interval or the studentized band.
    cross_fitted : bool
        Whether the fit's nuisances are fitted out of fold.
    jump_bounded : int
        The refits in which the engine scaled a combined hazard jump of one or more below one;
        when nonzero the reason says so.
    requested : int
        The refits asked for.

    Returns
    -------
    Support
        An experimental label with its reason, tagged with roadmap X6.

    Examples
    --------
    >>> from cleverly.survival.result import continuous_bootstrap_support
    >>> support = continuous_bootstrap_support(kind="bootstrap", cross_fitted=False)
    >>> support.label, support.reason.split(".")[0]
    ('experimental', 'The bootstrap delta-method conditions are not yet read first-hand')
    """
    if kind not in ("bootstrap", "bootstrap_bca", "bootstrap_band"):
        raise ValueError(f"unknown bootstrap kind {kind!r}")
    reason = CROSS_FITTED_BOOTSTRAP_REASON if cross_fitted else IN_SAMPLE_BOOTSTRAP_REASON
    if kind == "bootstrap_band":
        reason = f"{reason} {BOOTSTRAP_BAND_REASON}"
    if jump_bounded:
        reason = (
            f"{reason} In {jump_bounded} of {requested} refits the engine scaled a combined "
            "hazard jump of one or more below one, an extrapolation that no registered study "
            "gates."
        )
    return Support("experimental", reason, ("X6",))


def bootstrap_withheld_text(failed: int, requested: int, reasons: str, interval: str) -> str:
    """The failure rule's text: too many failed refits, so every bootstrap interval is withheld.

    Parameters
    ----------
    failed : int
        The failed refits.
    requested : int
        The replicates asked for.
    reasons : str
        Each failure's class and first sentence, with its count.
    interval : {"percentile", "bca"}
        The interval the fit would have attached.

    Returns
    -------
    str
        The warning's text.
    """
    name = "percentile" if interval == "percentile" else "BCa"
    return (
        f"{failed} of {requested} bootstrap refits failed ({reasons}); the {name} interval "
        "would condition on the replicates that fit, so it is withheld. The Wald interval "
        "stands."
    )


def _failure_reasons(reasons: Sequence[str]) -> str:
    """Each distinct failure reason with its count, most frequent first."""
    counts: dict[str, int] = {}
    for reason in reasons:
        counts[reason] = counts.get(reason, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return "; ".join(f"{count} x {reason}" for reason, count in ordered)


def attach_bootstrap(
    result: ContinuousSurvivalResult,
    outcomes: Sequence[tuple[Mapping[str, tuple[float, float]] | None, str | None]],
    *,
    interval: Literal["percentile", "bca"],
    simultaneous: bool,
    alpha: float,
    cross_fitted: bool,
    jump_bounded: int = 0,
) -> ContinuousSurvivalResult:
    """``result`` with the bootstrap replicates' intervals and band, or with them withheld.

    Parameters
    ----------
    result : ContinuousSurvivalResult
        The fit.
    outcomes : sequence of tuple
        Per replicate, each estimate's ``(estimate, standard error)``, or ``None`` with the
        reason the refit failed.
    interval : {"percentile", "bca"}
        The interval each estimate carries.
    simultaneous : bool
        Whether to add the studentized band over every risk and risk difference.
    alpha : float
        Significance level.
    cross_fitted : bool
        Whether the fit is cross-fitted, which picks the label of every bootstrap interval.
    jump_bounded : int
        The refits in which the engine's jump bound acted, kept as
        :attr:`~ContinuousSurvivalResult.bootstrap_jump_bounded` and named in every bootstrap
        interval's reason when nonzero.

    Returns
    -------
    ContinuousSurvivalResult
        A copy carrying :attr:`~ContinuousSurvivalResult.bootstrap`, its status and failures,
        and, unless more than :data:`BOOTSTRAP_FAILURE_SHARE` of the refits failed, each
        estimate's :class:`~cleverly.inference.BootstrapSummary` and the band.
    """
    from .._warnings import warn
    from ..exceptions import ConvergenceWarning
    from ..inference.bootstrap import BootstrapResult, bca_acceleration, bootstrap_band

    requested = len(outcomes)
    kept = [values for values, _ in outcomes if values is not None]
    failures = tuple(reason for values, reason in outcomes if values is None and reason)
    failed = requested - len(kept)
    names = list(result.estimates)
    missing = (np.nan, np.nan)
    # The counts are kept even when every refit failed, so the record always says how many
    # ran; the draws are then empty.
    bootstrap = BootstrapResult(
        draws={
            name: np.array([row.get(name, missing)[0] for row in kept], dtype=float)
            for name in names
        },
        n_requested=requested,
        n_failed=failed,
        resampling="iid",
        std_errors={
            name: np.array([row.get(name, missing)[1] for row in kept], dtype=float)
            for name in names
        },
    )
    if not kept or failed > BOOTSTRAP_FAILURE_SHARE * requested:
        warn(
            bootstrap_withheld_text(failed, requested, _failure_reasons(failures), interval),
            ConvergenceWarning,
        )
        return replace(
            result,
            bootstrap=bootstrap,
            bootstrap_status="bootstrap_unstable",
            bootstrap_failures=failures,
            bootstrap_jump_bounded=jump_bounded,
        )
    kind: BootstrapKind = "bootstrap_bca" if interval == "bca" else "bootstrap"
    counts = {"jump_bounded": jump_bounded, "requested": requested}
    support = continuous_bootstrap_support(kind=kind, cross_fitted=cross_fitted, **counts)
    estimates = {}
    for name, estimate in result.estimates.items():
        if interval == "bca":
            drawn = bootstrap.interval(
                name,
                alpha,
                interval="bca",
                estimate=estimate.psi,
                acceleration=bca_acceleration(estimate.influence_curve),
            )
        else:
            drawn = bootstrap.interval(name, alpha)
        estimates[name] = estimate.with_bootstrap(replace(drawn, support=support))
    band = None
    if simultaneous:
        risks = risk_names(result.arms, result.causes, result.horizons)
        band = bootstrap_band(
            bootstrap,
            {name: result.estimates[name].psi for name in risks},
            {name: result.estimates[name].std_error for name in risks},
            alpha=alpha,
            support=continuous_bootstrap_support(
                kind="bootstrap_band", cross_fitted=cross_fitted, **counts
            ),
        )
    return replace(
        result,
        estimates=estimates,
        bootstrap=bootstrap,
        bootstrap_band=band,
        bootstrap_status="ran",
        bootstrap_failures=failures,
        bootstrap_jump_bounded=jump_bounded,
    )


#: The share of random times at which the registered study's score equation held, on average
#: over its fits (``targeting-readings.csv``, ``mean_off_grid_share``): the reason a restricted
#: mean must be targeted rather than read off the targeted risks.
OFF_GRID_SHARE: Final[float] = 0.40

#: The support of a ratio the fit does not report itself: an odds ratio or a survival-view ratio.
RATIO_SUPPORT: Final[Support] = _experimental(RATIO_REASON)

#: The support of a smooth function of reported estimates: no registered study measures one.
CONTRAST_SUPPORT: Final[Support] = _experimental(
    "No registered study measures a smooth function of continuous-time estimates."
)

#: A risk's name, ``risk_regimen[arm @ t=...]`` or ``cif_regimen[arm, cause @ t=...]``, with
#: its time; read only to tell a risk at an untargeted time (refusal C2) from another name.
_LEVEL_PATTERN: Final = re.compile(r"^(?:risk|cif)_regimen\[.+ @ t=(?P<time>[^\]]+)\]$")

if TYPE_CHECKING:
    from ..inference.bootstrap import BootstrapBand, BootstrapResult
    from ._targeting import WalkResult
    from .data import ContinuousSurvivalData
    from .engine import ContinuousSurvivalRecord, ContinuousSurvivalTMLE

__all__ = [
    "CONTINUOUS_SURVIVAL_STUDY",
    "ContinuousSurvivalResult",
    "attach_bootstrap",
    "build_estimates",
    "continuous_bootstrap_support",
    "continuous_survival_support",
    "risk_names",
]


def _influence_view(estimate: ParameterEstimate) -> ParameterEstimate:
    """``estimate`` read under the centred rule, with its influence-curve variance."""
    record = estimate.switched
    if record is None:
        return estimate
    return replace(
        estimate,
        covariance_rule="centered",
        variance=record.influence_variance,
        switched=None,
    )


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


def _json_ready(value: Any) -> Any:
    """``value`` with every non-finite float replaced by ``None``, which strict JSON reads."""
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def build_estimates(
    walk: WalkResult,
    arms: tuple[str, str],
    causes: Mapping[int, str],
    horizons: tuple[float, ...],
    *,
    n: int,
    alpha: float,
    support: Callable[[str, int, float], Support],
    restricted: tuple[float, ...] = (),
    predictable: FloatArray | None = None,
) -> tuple[dict[str, ParameterEstimate], PredictableCovariance | None]:
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
        The support of an estimate, given its scale, the index of its weight row (a target
        time, then a restriction time) and the relative standard deviation of its
        influence-curve variance.
    restricted : tuple of float
        The restriction times, whose targets follow the target times in each arm and cause.
    predictable : ndarray or None
        ``(P, P)`` the predictable-variation second moments of the walk's influence columns
        (:func:`~cleverly.survival._targeting.predictable_covariance`).  Given, every estimate
        reports the switched variance read from it; ``None`` reports the influence-curve
        variance.

    Returns
    -------
    estimates : dict of str to ParameterEstimate
        Every arm's risk, then each difference and ratio, by time; then the restricted means
        time lost of each arm and cause, their differences, and the restricted mean survival
        times of each arm and their differences, by restriction time.
    covariance : PredictableCovariance or None
        The predictable-variation covariance of the levels with each estimate's gradient, or
        ``None`` without ``predictable``.
    """
    labels = list(causes.values())
    J, K = len(labels), len(horizons)
    Q = K + len(restricted)
    P = 2 * J * Q
    single = J == 1
    psi = walk.evaluation.psi
    ic = walk.evaluation.influence
    # The walk targets RMTL / tau; every level here is in reported units (time for an RMTL).
    unit = np.ones(P)
    for a in (0, 1):
        for j in range(J):
            for r, tau in enumerate(restricted):
                unit[(a * J + j) * Q + K + r] = float(tau)
    targets = tuple(
        level_name(arms[a], None if single else labels[j], horizons[q])
        if q < K
        else restricted_name(
            "rmtl_regimen", arms[a], None if single else labels[j], restricted[q - K]
        )
        for a in (0, 1)
        for j in range(J)
        for q in range(Q)
    )
    matrix = None
    if predictable is not None:
        matrix = unit[:, None] * np.asarray(predictable, dtype=float) * unit[None, :]
        # The divisor of the influence-curve variance (ddof 1), so the two compare like with like.
        matrix = matrix / max(n - 1, 1)
    gradients: dict[str, FloatArray] = {}
    out: dict[str, ParameterEstimate] = {}

    def build(
        name: str, value: float, curve: FloatArray, row: FloatArray, scale: str, q: int, **kw: Any
    ) -> ParameterEstimate:
        relative_sd = float(influence_row(name, curve).variance_rel_sd)
        extra: dict[str, Any] = {}
        if matrix is not None:
            gradients[name] = row
            extra = {
                "covariance_rule": "switched",
                "predictable_variance": float(row @ matrix @ row),
            }
        return make_estimate(
            name,
            value,
            curve,
            n=n,
            scale=scale,  # type: ignore[arg-type]
            alpha=alpha,
            rule_widened=False,
            support=support(scale, q, relative_sd),
            **kw,
            **extra,
        )

    def e(p: int) -> FloatArray:
        row = np.zeros(P)
        row[p] = 1.0
        return row

    for a in (0, 1):
        for j, cause in enumerate(labels):
            for k, time in enumerate(horizons):
                p = (a * J + j) * Q + k
                name = level_name(arms[a], None if single else cause, time)
                out[name] = build(name, psi[p], ic[:, p], e(p), "level", k, bounds=(0.0, 1.0))
    for j, cause in enumerate(labels):
        for k, time in enumerate(horizons):
            p0, p1 = j * Q + k, (J + j) * Q + k
            label = None if single else cause
            name = contrast_name("ate_regimen", arms[1], arms[0], label, time)
            out[name] = build(
                name,
                psi[p1] - psi[p0],
                ic[:, p1] - ic[:, p0],
                e(p1) - e(p0),
                "difference",
                k,
                bounds=(-1.0, 1.0),
            )
            name = contrast_name("rr_regimen", arms[1], arms[0], label, time)
            if psi[p0] > 0 and psi[p1] > 0:
                log_ic = ic[:, p1] / psi[p1] - ic[:, p0] / psi[p0]
                out[name] = build(
                    name,
                    psi[p1] / psi[p0],
                    log_ic,
                    e(p1) / psi[p1] - e(p0) / psi[p0],
                    "ratio",
                    k,
                    log_psi=float(np.log(psi[p1] / psi[p0])),
                )
    pair = f"{arms[1]} vs {arms[0]}"
    for r, tau in enumerate(restricted):
        q = K + r
        tau = float(tau)
        span = (-tau, tau)

        def lost(
            a: int, j: int, q: int = q, tau: float = tau
        ) -> tuple[float, FloatArray, FloatArray]:
            # The walk targets RMTL / tau, so the walk does not depend on the unit of time;
            # the report is in time units, and so is the covariance's row.
            p = (a * J + j) * Q + q
            return tau * float(psi[p]), tau * ic[:, p], e(p)

        def level(
            name: str,
            value: float,
            curve: FloatArray,
            row: FloatArray,
            q: int = q,
            tau: float = tau,
        ) -> ParameterEstimate:
            return build(
                name,
                value,
                curve,
                row,
                "level",
                q,
                bounds=(0.0, tau),
                range_interval=True,
            )

        def difference(
            name: str,
            value: float,
            curve: FloatArray,
            row: FloatArray,
            q: int = q,
            span: tuple[float, float] = span,
        ) -> ParameterEstimate:
            return build(name, value, curve, row, "difference", q, bounds=span)

        for a in (0, 1):
            for j, cause in enumerate(labels):
                name = restricted_name("rmtl_regimen", arms[a], None if single else cause, tau)
                out[name] = level(name, *lost(a, j))
        for j, cause in enumerate(labels):
            (v0, c0, g0), (v1, c1, g1) = lost(0, j), lost(1, j)
            name = restricted_name("rmtl_regimen", pair, None if single else cause, tau)
            out[name] = difference(name, v1 - v0, c1 - c0, g1 - g0)
        # RMST(tau) = tau - sum_j RMTL_j(tau); its score is solved by linearity.
        total = []
        for a in (0, 1):
            parts = [lost(a, j) for j in range(J)]
            total.append(
                (
                    sum(v for v, _, _ in parts),
                    np.sum([c for _, c, _ in parts], axis=0),
                    np.sum([g for _, _, g in parts], axis=0),
                )
            )
        for a in (0, 1):
            name = restricted_name("rmst_regimen", arms[a], None, tau)
            out[name] = level(name, tau - total[a][0], -total[a][1], -total[a][2])
        name = restricted_name("rmst_regimen", pair, None, tau)
        out[name] = difference(
            name,
            total[0][0] - total[1][0],
            total[0][1] - total[1][1],
            total[0][2] - total[1][2],
        )
    covariance = None if matrix is None else PredictableCovariance(targets, matrix, gradients)
    return out, covariance


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
    predictable : PredictableCovariance or None
        Under ``variance="switched"``, the predictable-variation covariance of every targeted
        level, with each stored estimate's gradient over them; :meth:`contrast` and
        :meth:`ratio` read it.  ``None`` under ``variance="influence_curve"``.
    estimator : ContinuousSurvivalTMLE or None
        A copy of the estimator that produced the fit, with its seed resolved and the outer
        cross-fitting label of each row recorded, so
        ``estimator.refit(data, random_state=estimator.random_state)`` reproduces the fit.
        ``None`` only on a result built by hand, which the refits refuse.
    data : ContinuousSurvivalData or None
        The rows the fit read. ``None`` only on a result built by hand.
    bootstrap : BootstrapResult or None
        With ``n_bootstrap``, every replicate's estimate and standard error of each estimate,
        and how many refits were requested and failed (``n_requested``, ``n_failed``); the
        draws are empty when every refit failed.  A replicate's standard error is its
        switched-variance standard error with ``simultaneous=True``, whose studentized band
        reads it, and its influence-curve standard error otherwise.  ``None`` without a
        bootstrap.  When at most
        1% of the refits fail, every bootstrap interval and the band read the refits that fit,
        with no warning; ``n_failed`` and :attr:`bootstrap_failures` say how many did not.
    bootstrap_band : BootstrapBand or None
        The studentized bootstrap band over every risk and risk difference, with ``n_bootstrap``
        and ``simultaneous=True`` and unless the intervals are withheld.  It is the estimate
        plus or minus the critical value times its standard error on the identity scale, so a
        small risk's band can cross 0, and it ignores ``band_scale``.
    bootstrap_status : {"ran", "bootstrap_unstable"} or None
        ``"ran"`` when each estimate carries its bootstrap interval, ``"bootstrap_unstable"``
        when more than 1% of the refits failed and every bootstrap interval is withheld
        (the failure rule), and ``None`` without a bootstrap.
    bootstrap_failures : tuple of str
        The class and first sentence of each failed refit's exception, in replicate order.
    bootstrap_jump_bounded : int
        The bootstrap refits that fitted with a nonzero
        :attr:`~cleverly.survival.ContinuousSurvivalRecord.jump_bound_share`: the engine scaled
        some combined hazard jump of one or more below one.  A refit warns of nothing; this
        count, and the reason of every bootstrap interval when it is nonzero, say how often
        the bound acted.

    Notes
    -----
    :attr:`diagnostics`, :attr:`sensitivity`, :meth:`validate` and :meth:`assess` declare a
    row for every assessment operation. The support, nuisance, score and influence
    diagnostics read :attr:`record` and the influence curves, and the E-value reads a risk
    ratio of the fit. ``diagnostics.refute()`` refits :attr:`estimator` on perturbed copies of
    :attr:`data` (the placebo on a contrast, the random common cause and the subset tests, all
    descriptive), and ``diagnostics.truncation_curve()`` reruns the targeting walk at other
    floors from refitted initial hazards. ``sensitivity.simulated_confounding()`` refits
    :attr:`estimator` once per cell on event and censoring times redrawn from the replay's own
    Cox fits under a simulated common cause, reusing the fit's fold plan under its own seed
    (descriptive). Every other sensitivity analysis refuses with its reason and roadmap item.
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
    predictable: PredictableCovariance | None = None
    estimator: ContinuousSurvivalTMLE | None = field(default=None, repr=False)
    data: ContinuousSurvivalData | None = field(default=None, repr=False)
    bootstrap: BootstrapResult | None = field(default=None, repr=False)
    bootstrap_band: BootstrapBand | None = None
    bootstrap_status: Literal["ran", "bootstrap_unstable"] | None = None
    bootstrap_failures: tuple[str, ...] = ()
    bootstrap_jump_bounded: int = 0
    #: Assessment results keyed by operation and normalized arguments. ``init=False``, so a
    #: mapping handed to the constructor by ``dataclasses.replace`` cannot answer for
    #: another fit.
    assessment_cache: AssessmentCache = field(
        default_factory=AssessmentCache, init=False, repr=False
    )

    #: The assessment family: one declaration per assessment operation in
    #: :data:`cleverly.assessment.ASSESSMENT_CAPABILITIES`.
    assessment_family: ClassVar[str] = CONTINUOUS_SURVIVAL_FAMILY
    #: The method identity of the result.
    fitted_method: ClassVar[str] = "tmle"

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
        (:func:`cleverly.inference.results.smooth_contrast`).  Under ``variance="switched"``
        the variance is :math:`h^\\top \\hat\\Sigma_{pv} h` with :math:`h` the function's gradient
        over the targeted levels, raised over the influence-curve variance as for a stored
        estimate, with the switch read on the derived curve.  No registered study measures such
        a function, so the result reads ``experimental``.

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

        def run(estimates: Mapping[str, ParameterEstimate]) -> ParameterEstimate:
            return smooth_contrast(
                estimates,
                function,
                names,
                n=self.n,
                alpha=self._alpha,
                name=name,
                scale=scale,
                gradient=gradient,
                transform=transform,
            )

        derived = self._derive(names, run)
        support = combine_support(
            [CONTRAST_SUPPORT, *(self.estimates[key].support for key in names)]
        )
        return replace(derived, support=support)

    def _derive(
        self,
        names: Sequence[str],
        run: Callable[[Mapping[str, ParameterEstimate]], ParameterEstimate],
    ) -> ParameterEstimate:
        """A derived estimate from ``run``, under the fit's variance.

        Under ``variance="influence_curve"`` this is ``run`` of the stored estimates.  Under
        ``"switched"``, ``run`` builds the derived curve from the inputs' influence-curve
        versions, and the same ``run`` on their gradients over the targeted levels gives the
        derived gradient :math:`h`: the delta method is linear in the curves at fixed values,
        so the derived curve is the levels' curves times :math:`h`.  The derived estimate then
        reports the switched variance of :math:`h^\\top \\hat\\Sigma_{pv} h`, with the switch
        read on its own curve.
        """
        chosen = [self.estimates[key] for key in names]
        covariance = self.predictable
        if covariance is None or not any(e.switched is not None for e in chosen):
            return run(self.estimates)
        view = {key: _influence_view(self.estimates[key]) for key in names}
        derived = run(view)
        rows = {key: replace(view[key], influence_curve=covariance.gradients[key]) for key in names}
        gradient = run(rows).influence_curve
        rebuilt = make_estimate(
            derived.name,
            derived.psi,
            derived.influence_curve,
            n=derived.n,
            rule_widened=derived.rule_widened,
            scale=derived.scale,
            alpha=derived.alpha,
            log_psi=derived.log_psi,
            covariance_rule="switched",
            inference=derived.inference,
            reference_df=derived.reference_df,
            support=derived.support,
            bounds=derived.bounds,
            outcome_family=derived.outcome_family,
            range_interval=derived.range_interval,
            predictable_variance=covariance.quadratic(gradient),
        )
        return replace(rebuilt, transform=derived.transform)

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
        type only.  Under ``variance="switched"`` the log-scale variance is
        :math:`h^\\top \\hat\\Sigma_{pv} h` with :math:`h` the gradient of the log ratio over the
        targeted levels, raised over the influence-curve variance as for a stored estimate, with
        the switch read on the ratio's own curve.  A risk ratio on the risk view carries the
        label of the fit's own ``rr_regimen`` estimate, which the registered
        continuous-survival-functionals study supports only in a fit with one restricted
        horizon; an odds ratio and a survival-view ratio read ``experimental``.

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

        def run(estimates: Mapping[str, ParameterEstimate]) -> ParameterEstimate:
            return ratio_contrast(
                estimates,
                numerator,
                denominator,
                kind=kind,
                complement=view == "survival",
                n=self.n,
                alpha=self._alpha,
                name=label,
            )

        derived = self._derive((numerator, denominator), run)
        if kind == "rr" and view == "risk":
            # The fit's own risk ratio of this pair, in either order, labelled when it was built.
            for first, second in ((top, bottom), (bottom, top)):
                stored = self.estimates.get(
                    contrast_name("rr_regimen", first[0], second[0], first[1], first[2])
                )
                if stored is not None:
                    return replace(derived, support=stored.support)
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
                    f"Refit ContinuousSurvivalTMLE with restricted_horizons=({_t(tau)},), or "
                    f"declare restricted_horizons=({_t(tau)},) on RegimeMean or RegimeContrast."
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
        # The stored difference is on the identity scale, so its negation is the reverse, with
        # the same variances: the switch reads the same relative SD on the negated curve.
        label = restricted_name(head, f"{arm} vs {versus}", cause, tau)
        record = stored.switched
        return make_estimate(
            label,
            -stored.psi,
            -stored.influence_curve,
            n=self.n,
            scale="difference",
            alpha=self._alpha,
            bounds=(-tau, tau),
            rule_widened=False,
            support=stored.support,
            covariance_rule=stored.covariance_rule,
            predictable_variance=None if record is None else record.predictable_variance,
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
            When ``tau`` is not one of the fit's ``restricted_horizons``. The fit declares
            them on the engine, or as ``restricted_horizons=`` on ``RegimeMean`` or
            ``RegimeContrast`` in a study.
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
            When ``tau`` is not one of the fit's ``restricted_horizons``. The fit declares
            them on the engine, or as ``restricted_horizons=`` on ``RegimeMean`` or
            ``RegimeContrast`` in a study.
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
            times, restriction times, units, significance level and variance; and
            ``predictable_covariance``, the targets and matrix of the predictable-variation
            covariance under ``variance="switched"``, else ``None``; and ``bootstrap``, with
            ``n_bootstrap``, the status, the replicates requested and failed, each failure's
            reason and the studentized band, else ``None`` (each estimate's own mapping carries
            its bootstrap interval).  A value that is not a
            finite number, such as the p-value of a range-logit estimate, reads ``None``, so
            the mapping passes ``json.dumps(..., allow_nan=False)``.
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
                    "censoring": record.censoring.chosen_name,
                    "propensity": None
                    if propensity is None
                    else propensity.names[propensity.chosen],
                    "steps": int(record.steps),
                    "halvings": int(record.halvings),
                    "converged": bool(record.converged),
                    "cutoff": float(record.cutoff),
                    "standardized_score": [float(v) for v in record.standardized_score],
                    "bounded_share": float(record.bounded_share),
                    "last_time_floor_share": float(record.last_time_floor_share),
                    "jump_bound_share": float(record.jump_bound_share),
                    "jump_bound_at_risk_share": float(record.jump_bound_at_risk_share),
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
                    "variance": "influence_curve" if self.predictable is None else "switched",
                },
                "predictable_covariance": None
                if self.predictable is None
                else self.predictable.to_dict(),
                "bootstrap": self._bootstrap_dict(),
            }
        )

    def _bootstrap_dict(self) -> dict[str, Any] | None:
        """The bootstrap's status, counts, failures and band, ``None`` without a bootstrap."""
        if self.bootstrap_status is None:
            return None
        band = self.bootstrap_band
        return {
            "status": self.bootstrap_status,
            "n_requested": None if self.bootstrap is None else int(self.bootstrap.n_requested),
            "n_failed": None if self.bootstrap is None else int(self.bootstrap.n_failed),
            "n_kept": None
            if self.bootstrap is None
            else int(self.bootstrap.n_requested - self.bootstrap.n_failed),
            "failures": list(self.bootstrap_failures),
            "jump_bounded": int(self.bootstrap_jump_bounded),
            "band": None
            if band is None
            else {
                "critical_value": float(band.critical_value),
                "n_replicates": int(band.n_replicates),
                "bands": {name: [float(v) for v in limits] for name, limits in band.bands.items()},
                "support": band.support.label,
                "support_reason": band.support.reason,
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

    def plot(self, **kwargs: Any) -> Any:
        """Plot each arm's risk and their difference at the target times.

        Each estimate is drawn at its target time with its interval and, when the fit holds
        one, its simultaneous band: the multiplier band, or with ``band="bootstrap"`` the
        studentized bootstrap band of a fit with ``n_bootstrap=`` and ``simultaneous=True``.
        Nothing is drawn between target times, where the fit estimates no risk.

        Parameters
        ----------
        **kwargs
            Passed to :func:`cleverly.plotting.curve` (``scale=``, ``measure=``, ``band=``,
            ``cause=``, ``regimens=``, ``labels=``, ``units=``, ``time_label=``,
            ``truth=`` and ``title=``).

        Returns
        -------
        Figure
            The levels above and their difference below, one column per cause.

        Raises
        ------
        CapabilityError
            When the fit refuses the requested view, with the fit's own reason.
        ValueError
            When ``band=``, ``measure=``, ``scale=`` or ``cause=`` do not fit together or
            the fit, such as ``band="bootstrap"`` on a fit with no bootstrap band or
            ``cause=`` with ``measure="rmst"``.
        DataError
            When ``cause=`` or ``regimens=`` names a cause or arm the fit does not hold.
        """
        from ..plotting import curve

        return curve(self, **kwargs)

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

    # ------------------------------------------------------------- assessment

    @property
    def diagnostics(self) -> Any:
        """The support, score, nuisance and influence diagnostics of the fit."""
        from ..assessment import DiagnosticsFacade

        return DiagnosticsFacade(self)

    @property
    def sensitivity(self) -> Any:
        """The sensitivity-analysis facade, whose every row refuses on this fit."""
        from ..assessment import SensitivityFacade

        return SensitivityFacade(self)

    def validate(self) -> Any:
        """Run the inexpensive default battery without refitting.

        Returns
        -------
        ValidationReport
            The score-equation, support and nuisance-model checks, read off the record.
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

        With ``include_refits=True`` the refutation and the truncation curve refit the stored
        estimator; the simulated common cause and every sensitivity bound refuse on this fit,
        and the report names each refusal without raising it.

        Parameters
        ----------
        include_refits : bool
            Run requested operations that refit nuisance models.
        include_retargets : bool
            Include moderate retargets.
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

    @property
    def replayability(self) -> Any:
        """Which assessment operations survive from the stored fit."""
        from ..assessment import replayability

        return replayability(self)

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
