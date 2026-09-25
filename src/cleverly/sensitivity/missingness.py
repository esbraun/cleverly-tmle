r"""Sensitivity to a non-ignorable missingness mechanism.

Handling missing outcomes with a ``delta`` model assumes the outcome is missing at
random given ``(A, W)``: among units with the same treatment and covariates, whether
the outcome was recorded carries no information about what it would have been.  That
assumption is not testable from the observed data, and it is often the weakest link
in an analysis of a study with dropout.

This module makes the assumption a *dial* rather than a premise, in the style of
Scharfstein, Rotnitzky & Robins (1999).  On the ``[0, 1]`` outcome scale, the mean
among the *unobserved* units is tilted away from the mean among the observed ones:

.. math::

    \bar Q^{\text{miss}}_\gamma(a, W)
      = \operatorname{expit}\bigl(\operatorname{logit} \bar Q^*(a, W) + \gamma\bigr),

and the full-population regression mixes the two according to the estimated
missingness probability:

.. math::

    \bar Q^{\text{full}}_\gamma(a, W)
      = \pi_a(W)\, \bar Q^*(a, W)
      + \bigl(1 - \pi_a(W)\bigr) \bar Q^{\text{miss}}_\gamma(a, W).

:math:`\gamma = 0` is exactly the MAR analysis, so the curve passes through the
reported estimate by construction -- which is the property that makes it readable.
Positive :math:`\gamma` says the unobserved outcomes were systematically *higher*
than MAR implies; negative, lower.  A useful way to read the output is to find the
:math:`\gamma` at which the conclusion changes and ask whether departures of that
size are plausible given why data went missing.

**One tilt or one per arm.**  The formula above moves every arm's regression by the same
:math:`\gamma`, which is a modelling assumption and not an accident of the two-armed
case: it says the unobserved outcomes are displaced by the same amount whatever
treatment the unit received.  With more than two arms that assumption is easier to doubt
-- dropout after an ineffective arm need not mean what dropout after an effective one
does -- so ``arm_gamma=`` declares a *direction* instead, one multiplier per arm, and the
grid sweeps its magnitude.  It is required to name every arm, because an arm silently
defaulted to 1 would be the modelling choice made quietly that this keyword exists to
make loudly.

Caveat, stated plainly: the confidence intervals on the curve treat :math:`\gamma`
as known and reuse the MAR standard error.  They describe sampling uncertainty at a
fixed :math:`\gamma`, not uncertainty about :math:`\gamma` itself.

**What the tilt reports for a fit that has no interval.**  The curve is a plug-in
sweep of the point estimate, so the whole of it is defined for a fit at a
non-inferential status, such as a selector-path collaborative fit, whose targeted
regression and missingness mechanism this module reads directly.  Only the spread
columns depend on an influence curve the package refuses to claim there.  Such a fit
therefore receives ``plugin_std_err``, ``plugin_interval_lower`` and
``plugin_interval_upper`` in place of ``std_err``, ``ci_lower`` and ``ci_upper``, which
is the swap :func:`~cleverly.sensitivity.positivity.truncation_curve` already makes for
the same fit.  :func:`tipping_gamma` keeps its point-estimate search on such a fit and refuses
``use_ci=True``, because it returns one float and a float carries no column name to
say which of the two an interval crossing came from.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from .._typing import FloatArray
from ..exceptions import CapabilityError, refuse_inference
from ..inference.delta import normal_ci
from ..inference.influence import spread_name
from ..targets.population_intervention import (
    NATURAL_COURSE_TILT_REFUSAL,
    is_natural_course_fit,
)
from ..utils.bounds import expit, logit
from ._parameters import ArmParameter, reported_arm_parameters, stratum_refusal

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..data import CausalData
    from ..estimators._nuisance import RepeatFit
    from ..estimators.base import TMLEResult

__all__ = ["DEFAULT_GAMMA_GRID", "missingness_tilt", "tipping_gamma"]

#: Default tilt values.  On the logit scale, ``gamma = 1`` shifts a mean of 0.5 to
#: about 0.73 -- a substantial departure from MAR, so the grid spans well past the
#: range most analyses would consider plausible.
DEFAULT_GAMMA_GRID: tuple[float, ...] = (
    -2.0,
    -1.5,
    -1.0,
    -0.5,
    -0.25,
    0.0,
    0.25,
    0.5,
    1.0,
    1.5,
    2.0,
)


def _refuse_longitudinal(result: Any) -> str | None:
    """Refuse any assessment family but ``point``.

    First in the table, and not by taste: :class:`~cleverly.longitudinal.LongitudinalData`
    declares no ``has_missing_outcome``, so a later rule would raise ``AttributeError`` on
    a longitudinal result instead of refusing it.  The sentence is byte-identical to the
    one the capability rows published before this table existed, because two example
    notebooks print it.
    """
    if getattr(result, "assessment_family", None) != "point":
        return "no longitudinal missingness-tilt adapter is implemented"
    return None


def _refuse_natural_course(result: Any) -> str | None:
    """Refuse the missing-outcome natural-course mean, which is not indexed by arm."""
    return NATURAL_COURSE_TILT_REFUSAL if is_natural_course_fit(result) else None


def _refuse_complete_outcome(result: Any) -> str | None:
    """Refuse a fit with no missing outcome, which has no observation mechanism to tilt.

    The one rule whose row reads ``not_applicable`` rather than ``unavailable``: the
    question has no subject on this fit, whereas every other rule names a derivation the
    package does not have.  It reads the data flag that the omitted-variable bound's
    response rule reads, so the fit the bound sends to the tilt is a fit this rule admits.
    """
    if result.data.has_missing_outcome:
        return None
    return (
        "the identified functional has no observation mechanism: missingness_tilt requires "
        "a fit with missing outcomes. Pass delta=<column> to fit() so the missingness "
        "mechanism is estimated."
    )


def _refuse_continuous(result: Any) -> str | None:
    """Refuse a continuous dose, whose plug-in reads ``Qbar`` at a policy's dose."""
    if not result.data.is_continuous_treatment:
        return None
    return (
        "missingness_tilt is written for the arm-indexed estimands; this fit declared "
        "a continuous dose with shifts= and reports ey_shift/ate_shift. The tilt "
        "re-mixes the targeted Qbar at each arm under a moved missingness mechanism, "
        "and a shift's plug-in is Qbar at the dose the policy assigns rather than at "
        "an arm -- so the tilt would have to move pi at that dose too, and whether "
        "the tilted parameter is still the shift parameter under a non-ignorable "
        "mechanism has not been derived here. Use "
        "truncation_curve(mechanism=True) for sensitivity to the missingness bound, "
        "or diagnostics.support() for the overlap question."
    )


def _refuse_incremental(result: Any) -> str | None:
    """Refuse an incremental fit, whose targeting also solves a score in ``g``."""
    if result.nuisance.incremental is None:
        return None
    return (
        "missingness_tilt is written for the arm-indexed estimands; this fit declared "
        "incremental interventions and reports ey_ipsi/ate_ipsi. The tilt reweights a "
        "targeted Qbar under a moved missingness mechanism, and on this axis the "
        "targeting is two alternating score equations rather than one -- the second "
        "lives in the tangent space of g, which the tilt does not move but which the "
        "alternation re-solves against a Qbar that has. Whether the tilted parameter "
        "is that alternation's fixed point has not been derived, so reporting a curve "
        "would be guessing. Use diagnostics.support() for the overlap question, or "
        "truncation_curve(mechanism=True) for sensitivity to the missingness bound."
    )


def _refuse_untiltable_parameters(result: Any) -> str | None:
    """Refuse a fit that reports no parameter the tilt can re-mix.

    The tilt re-mixes the arm-indexed means and their linear contrasts, which
    :func:`~cleverly.sensitivity._parameters.reported_arm_parameters` names.  A regime,
    an MSM, or a ratio-only fit reports none of them, and before this rule its row read
    available while every call refused with "no tiltable estimands requested".  A fit
    that reports at least one keeps the rule silent, and the default sweep skips the
    rest as it always has.
    """
    if reported_arm_parameters(result):
        return None
    return (
        "missingness_tilt re-mixes the arm-indexed means and their linear contrasts, "
        f"and this fit reports none of them: {sorted(result.estimates)}. A ratio, a "
        "regime mean, and an MSM coefficient have no derived tilt."
    )


#: Every missingness-tilt refusal that no argument of the call can lift, in the one order
#: the two entry points and the two capability rows use.  The table is ordered, and each
#: rule assumes its predecessors returned ``None``: ``longitudinal`` establishes that the
#: result carries point-treatment ``data``, which every later rule reads, and
#: ``missing_outcome`` establishes the observation mechanism that the arm-indexed rules
#: after it would tilt.  The names are the introspection contract: the capability row
#: reads the name to choose its status, and a test pins the order without respelling a
#: message.
_FIT_WIDE_TILT_RULES: tuple[tuple[str, Callable[[Any], str | None]], ...] = (
    ("longitudinal", _refuse_longitudinal),
    ("natural_course", _refuse_natural_course),
    ("missing_outcome", _refuse_complete_outcome),
    ("continuous", _refuse_continuous),
    ("incremental", _refuse_incremental),
    ("tiltable_parameters", _refuse_untiltable_parameters),
)


def _fit_wide_tilt_rule(result: Any) -> tuple[str, str] | None:
    """The first rule of :data:`_FIT_WIDE_TILT_RULES` that refuses, with its reason."""
    for name, rule in _FIT_WIDE_TILT_RULES:
        reason = rule(result)
        if reason is not None:
            return name, reason
    return None


def fit_wide_tilt_refusal(result: Any) -> str | None:
    """Return the first missingness-tilt refusal that applies to the whole fit.

    Every boundary in :data:`_FIT_WIDE_TILT_RULES` is reachable from capability reporting
    and from execution, and no argument of the call can change its verdict.  One helper
    answers both callers, so a fit the tilt refuses is never advertised as available.
    Request-specific checks, a conditional stratum named in ``estimands=`` and an
    explicit request with no tiltable name, stay in :func:`missingness_tilt`.

    Parameters
    ----------
    result : Any
        Fitted result inspected by this module or by its assessment facade.

    Returns
    -------
    str or None
        Exact refusal reason, or ``None`` when no fit-wide boundary applies.
    """
    rule = _fit_wide_tilt_rule(result)
    return None if rule is None else rule[1]


def missingness_tilt(
    result: TMLEResult,
    gamma: Sequence[float] | None = None,
    *,
    estimands: Sequence[str] | None = None,
    arm_gamma: Mapping[Any, float] | None = None,
) -> Any:
    """Estimate under a range of departures from missingness-at-random.

    Returns a tidy frame with one row per ``(gamma, estimand)``, and one
    ``gamma[<level>]`` column per arm giving the tilt that arm received -- equal to
    ``gamma`` throughout unless ``arm_gamma=`` says otherwise.  Only defined for a fit
    that supplied ``delta``; without missing outcomes there is nothing to tilt.

    Refuses first by :func:`fit_wide_tilt_refusal`, which the capability rows read too.  It
    refuses a longitudinal result, a missing-outcome ``NaturalCourseMean`` fit, a fit with
    no missing outcome, a continuous dose, an incremental fit, and a fit that reports no
    arm-indexed mean or linear contrast, such as a regime, an MSM, or a ratio-only fit.
    The natural-course target is not indexed by arm, so it needs a natural-course
    sensitivity parameter that is not implemented.

    A fit whose inference status supplies no inference, such as a selector-path
    ``CTMLE`` fit, receives ``plugin_std_err``,
    ``plugin_interval_lower`` and ``plugin_interval_upper`` in place of ``std_err``,
    ``ci_lower`` and ``ci_upper``.  The point-estimate curve is unchanged, and no column
    of it claims coverage this package does not supply.

    Parameters
    ----------
    result : TMLEResult
        A fitted result whose outcomes are missing for some rows.
    gamma : sequence of float or None
        Tilt values on the logit scale.  ``None`` uses :data:`DEFAULT_GAMMA_GRID`.
    estimands : sequence of str or None
        Restrict to a subset.  Ratios are excluded automatically: tilting changes the
        counterfactual means, and re-deriving a ratio's log-scale influence curve under
        the tilt would misrepresent the uncertainty.
    arm_gamma : mapping of level to float, or None
        One finite multiplier per arm, keyed by the treatment level as the caller wrote it, so
        that the tilt at arm ``a`` is ``arm_gamma[a] * gamma``.  ``None`` -- the default,
        and what a two-armed fit has always done -- tilts every arm by the same
        ``gamma``.  Every arm must appear; see this module's docstring for why the choice
        is not made silently.  The ``gamma`` column of the returned frame is then the
        magnitude the direction is scaled by rather than the tilt any one arm received,
        which is what the ``gamma[<level>]`` columns beside it report.

    Returns
    -------
    dataframe
        One row per ``(gamma, estimand)``, with one ``gamma[<level>]`` column per
        arm giving the tilt that arm received.
    """
    refusal = fit_wide_tilt_refusal(result)
    if refusal is not None:
        raise CapabilityError(refusal)
    data = result.data
    if result.nuisance.missingness is None:  # pragma: no cover - guarded above
        raise CapabilityError("missingness_tilt requires a fitted missingness mechanism")

    # Which parameters the tilt can re-mix: the arm-indexed linear ones, named for their
    # arms on a fit with more than two. Ratios are excluded by being absent from that
    # map rather than by a second filter here.
    tiltable = reported_arm_parameters(result)
    requested = tuple(estimands if estimands is not None else result.estimates)
    if estimands is not None:
        # Only an explicit request is refused. The default sweep skips whatever it cannot
        # tilt -- a ratio, a stratum -- exactly as it always has, so asking for the whole
        # report still returns the tiltable part of it.
        for name in requested:
            conditional = stratum_refusal(result, name, "the MNAR tilt")
            if conditional is not None:
                raise CapabilityError(conditional)
    parameters = tuple(tiltable[name] for name in requested if name in tiltable)
    if not parameters:
        raise CapabilityError(
            "no tiltable estimands requested; the tilt applies to the arm-indexed means "
            f"and their contrasts, which for this fit are {sorted(tiltable)}"
        )

    direction = _tilt_direction(data, arm_gamma)
    grid = tuple(DEFAULT_GAMMA_GRID if gamma is None else (float(g) for g in gamma))

    rows: list[dict[str, Any]] = []
    for value in grid:
        for parameter in parameters:
            name = parameter.name
            # Combined over the cross-fitting draws, as the fit's own report was. Each
            # draw has its own targeted Qbar and its own missingness mechanism, and the
            # tilt is a function of both, so a tilt read off one draw would sit at a
            # different level from the psi at gamma = 0 that the fit reported -- the curve
            # would step at its own origin. Every estimand here is a level or a
            # difference, so the median is taken on the reported scale.
            psi = float(
                np.median(
                    [
                        _tilted_psi(result, repeat, parameter, value, direction)
                        for repeat in result.repeats
                    ]
                )
            )
            # The names ``truncation_curve`` publishes, for the reason it publishes them: a
            # selector-path collaborative fit supplies no interval, so the sweep reports
            # the retained diagnostic under names that make no coverage claim rather than
            # raising after every tilt has been computed. ``plugin_std_error`` reads the
            # private body ``std_error`` reads, so an ordinary fit's numbers do not move.
            # The interval is centred on the tilted estimate, so only the names come
            # from ``spread_name``.
            status = result[name].inference
            std_error = result[name].plugin_std_error
            low, high = normal_ci(psi, std_error, result.config.alpha_sig)
            rows.append(
                {
                    "gamma": value,
                    "estimand": name,
                    "psi": psi,
                    spread_name("std_err", status): std_error,
                    spread_name("ci_lower", status): low,
                    spread_name("ci_upper", status): high,
                    "is_mar": bool(value == 0.0),
                    # The tilt each arm actually received, appended so the familiar
                    # columns stay where they were. Without these the direction lives
                    # only in the call: a curve read back off disk or handed to a plot
                    # could not say what it swept, and ``gamma`` alone is a magnitude.
                    # Under the default direction they all equal ``gamma``, which is the
                    # two-armed report saying that it tilted both arms alike.
                    **{
                        f"gamma[{data.arm_label(code)}]": value * direction[code]
                        for code in data.arm_codes
                    },
                }
            )

    payload = {key: [row[key] for row in rows] for key in rows[0]}
    return data.frame_like(payload)


def _tilt_direction(data: CausalData, arm_gamma: Mapping[Any, float] | None) -> dict[float, float]:
    """The per-arm multipliers, keyed by arm *code* -- all ones unless declared.

    Keyed by code on the way out and by the user's own level on the way in, which is the
    convention every reported name follows: a caller writes ``{"low": 1.0, "high": 0.5}``
    and never sees ``0.0`` and ``2.0``.
    """
    codes = data.arm_codes
    if arm_gamma is None:
        return dict.fromkeys(codes, 1.0)
    levels = list(data.treatment_levels)
    direction: dict[float, float] = {}
    for label, multiplier in arm_gamma.items():
        matches = [index for index, level in enumerate(levels) if level == label]
        if not matches:
            raise ValueError(
                f"arm_gamma names {label!r}, which is not a level of "
                f"{data.treatment_name}; its levels are {levels}"
            )
        value = float(multiplier)
        if not np.isfinite(value):
            raise ValueError(
                f"arm_gamma multipliers must be finite; got {multiplier!r} for {label!r}"
            )
        direction[float(matches[0])] = value
    missing = [levels[int(code)] for code in codes if code not in direction]
    if missing:
        raise ValueError(
            f"arm_gamma must name every arm, and {missing} are missing. An arm left out "
            "would be tilted by the shared gamma after all, which is the assumption this "
            "keyword exists to state rather than inherit; pass 1.0 to say so."
        )
    return direction


def _tilted_psi(
    result: TMLEResult,
    repeat: RepeatFit,
    parameter: ArmParameter,
    gamma: float,
    direction: Mapping[float, float],
) -> float:
    """One estimand under one cross-fitting draw, at tilt ``gamma``.

    Reads the targeted ``Qbar`` and the missingness mechanism from the *same* draw, which
    is the whole reason :class:`~cleverly.estimators._nuisance.RepeatFit` holds them
    together: mixing one draw's regression with another's mechanism would produce a
    perfectly plausible number for a fit that never happened.

    Every arm is reached by its code -- the column of ``missingness`` included, which is
    keyed by arm exactly as ``targeted.arms`` is -- rather than by position, so the two
    cannot come apart on a fit with more arms than the loop that reads them.
    """
    data = result.data
    scaler = repeat.nuisance.scaler
    missingness = repeat.nuisance.bounded_missingness(result.config.missingness_bound)
    assert missingness is not None
    weights = data.weights
    arms = repeat.nuisance.arms

    targeted = repeat.fluctuations[parameter.group].targeted

    def full(arm: float) -> FloatArray:
        return _tilted(targeted.arms[arm], missingness[:, arms.index(arm)], gamma * direction[arm])

    if parameter.versus is None:
        psi_scaled = float(np.average(full(parameter.arm), weights=weights))
        return scaler.unscale_level(psi_scaled) if not scaler.is_identity else psi_scaled

    contrast = full(parameter.arm) - full(parameter.versus)
    conditioning = parameter.conditions_on
    if conditioning is None:
        psi_scaled = float(np.average(contrast, weights=weights))
    else:
        # The ATT and ATC average the contrast over one arm only, so the arm indicator
        # multiplies the observation weights -- and which arm that is differs per
        # contrast for the ATT, which is why it is read off the parameter.
        indicator = np.asarray(data.treatment == conditioning, dtype=float)
        psi_scaled = float(np.average(contrast, weights=weights * indicator))
    return scaler.unscale_difference(psi_scaled) if not scaler.is_identity else psi_scaled


def _tilted(targeted: FloatArray, observed_probability: FloatArray, gamma: float) -> FloatArray:
    """Mix the observed-data regression with a logit-tilted version for the missing."""
    if gamma == 0.0:
        return np.asarray(targeted, dtype=float)
    missing_mean = expit(logit(targeted) + gamma)
    return np.asarray(
        observed_probability * targeted + (1.0 - observed_probability) * missing_mean,
        dtype=float,
    )


#: The request ``tipping_gamma`` refuses on a fit that supplies no inference, named as the
#: caller writes it.  The raise and the argument-aware capability row both use it.
_TIPPING_INTERVAL_OPERATION = "tipping_gamma(use_ci=True)"


def tipping_gamma(
    result: TMLEResult,
    estimand: str = "ate",
    *,
    null_hypothesis: float = 0.0,
    search: tuple[float, float] = (-8.0, 8.0),
    use_ci: bool = False,
    arm_gamma: Mapping[Any, float] | None = None,
) -> float | None:
    """The tilt at which the conclusion tips.

    Returns the detected crossing with the smallest ``|gamma|`` at which the estimate
    (or, with ``use_ci=True``, the confidence interval) crosses ``null_hypothesis``.
    Returns ``None`` if the search detects no crossing. Reporting this single number is
    usually more informative than the whole curve: it converts "is MAR plausible?"
    into "would the unobserved outcomes have to differ by *this much*?".

    With ``arm_gamma=`` the number is the magnitude at which that *direction* tips the
    conclusion, which is what makes one scalar still meaningful when the arms are tilted
    by different amounts.

    Refuses first by :func:`fit_wide_tilt_refusal`, the table :func:`missingness_tilt`
    reads, because it searches over that same tilt.  The capability row quotes the same
    sentence.

    Refuses ``use_ci=True`` on a fit whose inference status supplies no inference, such
    as a selector-path ``CTMLE`` fit. Such a fit supplies no confidence limit for the
    search to follow.  ``use_ci=False``, the default, searches the point
    estimate and answers for that fit.

    Parameters
    ----------
    result : TMLEResult
        A fitted result with missing outcomes.
    estimand : str
        Alias to search over.
    null_hypothesis : float
        The value the conclusion is said to tip at.
    search : tuple of float
        Lower and upper tilt the search brackets. The pair must be increasing and
        contain zero. The search inspects an interior grid with extra resolution for
        the declared arm-specific tilt magnitudes before refining each crossing.
    use_ci : bool
        Whether to tip when the confidence limit reaches the null rather than the
        point estimate. Refused on a fit whose inference status supplies no
        inference, which reports no confidence limit.
    arm_gamma : mapping of level to float, or None
        One multiplier per arm, as :func:`missingness_tilt` accepts.

    Returns
    -------
    float or None
        The nearest detected tilt at which the conclusion crosses its null, or ``None``
        when the search detects no crossing.
    """
    refusal = fit_wide_tilt_refusal(result)
    if refusal is not None:
        raise CapabilityError(refusal)
    import narwhals as nw
    from scipy import optimize

    lower_search, upper_search = (float(bound) for bound in search)
    if (
        not np.isfinite(lower_search)
        or not np.isfinite(upper_search)
        or not lower_search <= 0.0 <= upper_search
        or lower_search == upper_search
    ):
        raise ValueError(
            "search must be a finite, increasing (lower, upper) pair that contains gamma=0; "
            f"got {search!r}"
        )
    if use_ci and estimand in result.estimates:
        # Before the first tilt.  The curve renames its spread columns on a selector-path
        # collaborative fit, so an interval crossing is still computable there, but this
        # function returns one bare float and no float can carry the name that says the
        # crossing came from a diagnostic.  The point-estimate search, which is the
        # default, answers for that fit unchanged.  A name this fit does not report falls
        # through to ``missingness_tilt``, which lists the tiltable ones.
        refuse_inference(
            result.estimates[estimand].inference, operation=_TIPPING_INTERVAL_OPERATION
        )

    def row_at(value: float) -> Any:
        frame = missingness_tilt(result, [value], estimands=[estimand], arm_gamma=arm_gamma)
        return nw.from_native(frame, eager_only=True)

    baseline_row = row_at(0.0)
    field = "psi"
    if use_ci:
        low = float(baseline_row["ci_lower"][0])
        high = float(baseline_row["ci_upper"][0])
        if low <= null_hypothesis <= high:
            return 0.0
        # Follow the limit between the baseline interval and the null. Its signed
        # distance crosses zero exactly when that limit reaches the null. The old
        # absolute distance never changed sign, so brentq could not find the boundary.
        field = "ci_lower" if null_hypothesis < low else "ci_upper"

    def deviation(value: float) -> float:
        return float(row_at(value)[field][0]) - null_hypothesis

    baseline = float(baseline_row[field][0]) - null_hypothesis
    if baseline == 0.0:
        return 0.0

    # An arm-specific direction can make a contrast nonmonotone even though each arm
    # mean is monotone. Endpoint-only bracketing can therefore miss two interior
    # crossings whose endpoint signs agree. A multiplier also reparameterizes gamma,
    # so add a dense grid over the finite logit transition range for every magnitude.
    # Outside +/-40 on that effective scale expit is at floating-point saturation.
    direction = _tilt_direction(result.data, arm_gamma)
    probe_parts = [np.linspace(lower_search, upper_search, 257), np.asarray([0.0])]
    for magnitude in {abs(value) for value in direction.values() if value != 0.0}:
        transition_lower = max(lower_search, -40.0 / magnitude)
        transition_upper = min(upper_search, 40.0 / magnitude)
        if transition_lower < transition_upper:
            probe_parts.append(np.linspace(transition_lower, transition_upper, 1025))
    probes = np.unique(np.concatenate(probe_parts))
    probe_frame = missingness_tilt(
        result,
        tuple(float(value) for value in probes),
        estimands=[estimand],
        arm_gamma=arm_gamma,
    )
    probe_rows = nw.from_native(probe_frame, eager_only=True)
    probe_deviations = np.asarray(probe_rows[field], dtype=float) - null_hypothesis

    roots = [
        float(value) for value, delta in zip(probes, probe_deviations, strict=False) if delta == 0.0
    ]
    for left, right, left_delta, right_delta in zip(
        probes[:-1], probes[1:], probe_deviations[:-1], probe_deviations[1:], strict=False
    ):
        if not np.isfinite(left_delta) or not np.isfinite(right_delta):
            continue
        if np.sign(left_delta) == np.sign(right_delta):
            continue
        try:
            root = optimize.brentq(deviation, left, right, xtol=1e-4, maxiter=100)
        except (ValueError, RuntimeError):  # pragma: no cover - no sign change
            continue
        roots.append(float(root))
    return min(roots, key=abs) if roots else None
