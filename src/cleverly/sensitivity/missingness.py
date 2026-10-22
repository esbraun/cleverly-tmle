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

**Which parameters.**  Every functional of the tilted regressions that the fit reported
is re-mixed: the arm means and their linear contrasts, the ratios ``rr`` and ``or`` of two
tilted arm means, the natural-course mean ``ey_obs`` at the received arm, ``par`` and
``paf``, a regime mean (the regime density times the tilted arm regressions), and an MSM
coefficient (the working-model projection of the tilted regressions).  Each repeats the
arithmetic of the target that reported it, so every curve passes through the reported
estimate at :math:`\gamma = 0`.

Caveat, stated plainly: the confidence intervals on the curve of an arm mean or a linear
contrast treat :math:`\gamma` as known and reuse the MAR standard error.  They describe
sampling uncertainty at a fixed :math:`\gamma`, not uncertainty about :math:`\gamma`
itself.  Every other parameter claims no interval at :math:`\gamma \neq 0`: its spread
there is the MAR standard error on its inference scale, centred on the tilted point, under
the ``plugin_`` names (roadmap X57 part d).

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
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .._refusal import NO_ALTERNATIVE, Refusal, RefusalReason
from .._typing import FloatArray
from ..data.validate import MISSING_OUTCOME_DECLARATION
from ..estimators._nuisance import tilt_missingness
from ..exceptions import CapabilityError, refuse_inference
from ..inference.delta import log_odds_ratio_influence, log_ratio_influence, wald_ci
from ..inference.influence import _DIAGNOSTIC_NAMES, ParameterEstimate, spread_name
from ..interventions.learned import LEARNED_RULE_TILT_REFUSAL
from ..msm import solve_projection
from ..targets.base import arm_alias, parameter_name
from ..targets.population_intervention import NATURAL_COURSE_TARGET
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


def _refuse_longitudinal(result: Any) -> Refusal | None:
    """Refuse any assessment family but ``point``.

    First in the table, and not by taste: :class:`~cleverly.longitudinal.LongitudinalData`
    declares no ``has_missing_outcome``, so a later rule would raise ``AttributeError`` on
    a longitudinal result instead of refusing it.  Two example notebooks print the
    sentence, so keep it stable.
    """
    if getattr(result, "assessment_family", None) != "point":
        return Refusal(
            RefusalReason.IMPLEMENTATION_GAP,
            "no longitudinal missingness-tilt adapter is implemented.",
            alternative=(
                "Call result.diagnostics.truncation_curve(bounds=[...]) for sensitivity to "
                "the cumulative treatment-and-censoring bound."
            ),
            roadmap="F16",
        )
    return None


def _refuse_learned_rule(result: Any) -> Refusal | None:
    """Refuse a learned-rule fit, for which no tilt was derived.

    Such a fit refuses missing outcomes before any learner, so it has no observation
    mechanism either.  This rule runs first, so the row names the learned-rule reason.
    """
    if result.config.parameter_axis != "learned_rule":
        return None
    return Refusal(
        RefusalReason.IMPLEMENTATION_GAP,
        "missingness_tilt does not cover a learned-rule fit: " + LEARNED_RULE_TILT_REFUSAL + ".",
        alternative=NO_ALTERNATIVE,
        roadmap="F27",
    )


def _refuse_complete_outcome(result: Any) -> Refusal | None:
    """Refuse a fit with no missing outcome, which has no observation mechanism to tilt.

    The one rule whose row reads ``not_applicable`` rather than ``unavailable``: the
    question has no subject on this fit, whereas every other rule names a derivation the
    package does not have.  It reads the data flag that the omitted-variable bound's
    response rule reads, so the fit the bound sends to the tilt is a fit this rule admits.
    """
    if result.data.has_missing_outcome:
        return None
    return Refusal(
        RefusalReason.INVALID_DECLARATION,
        "the identified functional has no observation mechanism: missingness_tilt requires "
        "a fit with missing outcomes. Declare the indicator with "
        f"{MISSING_OUTCOME_DECLARATION}, so the missingness mechanism is estimated.",
    )


def _refuse_continuous(result: Any) -> Refusal | None:
    """Refuse a treatment with a dose, whose plug-in reads ``Qbar`` at a policy's dose."""
    if result.data.has_arms:
        return None
    dose = (
        "a vector treatment with a continuous component"
        if result.data.treatment_kind == "vector"
        else "a continuous dose"
    )
    return Refusal(
        RefusalReason.IMPLEMENTATION_GAP,
        "missingness_tilt is written for the arm-indexed estimands; this fit declared "
        f"{dose} with policies= and reports ey_policy/ate_policy. The tilt "
        "re-mixes the targeted Qbar at each arm under a moved missingness mechanism, "
        "and a shift's plug-in is Qbar at the dose the policy assigns rather than at "
        "an arm. The tilt would have to move pi at that dose too, and whether "
        "the tilted parameter is still the shift parameter under a non-ignorable "
        "mechanism has not been derived here. docs/roadmap.md X57 tracks the tilt's "
        "point curve on a shift fit, with a plug-in spread and no interval.",
        alternative=(
            "Use truncation_curve(mechanism=True) for sensitivity to the missingness bound, "
            "or diagnostics.support() for the overlap question."
        ),
        roadmap="X57",
    )


def _refuse_incremental(result: Any) -> Refusal | None:
    """Refuse an incremental fit, whose targeting also solves a score in ``g``."""
    if result.nuisance.incremental is None:
        return None
    return Refusal(
        RefusalReason.IMPLEMENTATION_GAP,
        "missingness_tilt is written for the arm-indexed estimands; this fit declared "
        "incremental interventions and reports ey_ipsi/ate_ipsi. The tilt reweights a "
        "targeted Qbar under a moved missingness mechanism, and on this axis the "
        "targeting is two alternating score equations rather than one. The second "
        "lives in the tangent space of g, which the tilt does not move but which the "
        "alternation re-solves against a Qbar that has. Whether the tilted parameter "
        "is that alternation's fixed point has not been derived, so reporting a curve "
        "would be guessing. docs/roadmap.md X57 tracks the tilt's point curve on an "
        "incremental fit, with a plug-in spread and no interval.",
        alternative=(
            "Use diagnostics.support() for the overlap question, or "
            "truncation_curve(mechanism=True) for sensitivity to the missingness bound."
        ),
        roadmap="X57",
    )


#: The kinds of reported parameter the tilt re-mixes.  ``arm`` is a mean or a linear
#: contrast of the arm means, which :func:`reported_arm_parameters` names.  The rest are
#: functionals of the same tilted regressions: ``ratio`` is ``rr`` or ``or`` of two tilted
#: arm means, ``natural_course`` is ``ey_obs``, ``par`` and ``paf`` read ``ey_obs`` and the
#: reference arm mean, ``regime`` is a regime mean or contrast, and ``msm`` is one working
#: model coefficient.
_TiltKind = Literal["arm", "ratio", "natural_course", "par", "paf", "regime", "msm"]


@dataclass(frozen=True)
class _TiltTarget:
    """One reported parameter the tilt can re-mix, and how to recompute it.

    Parameters
    ----------
    name : str
        The reported name, which indexes ``result.estimates``.
    kind : str
        One of :data:`_TiltKind`.
    group : str
        The fluctuation that carries the targeted regression the parameter reads.
    code : float or None
        The arm, regime or coefficient code the parameter is about.
    versus : float or None
        The code a contrast is taken against, or ``None`` for a level.
    parameter : ArmParameter or None
        The arm parameter, for the ``arm`` kind only.
    stem : str
        The registered target, ``"rr"`` or ``"or"``, for the ``ratio`` kind.
    """

    name: str
    kind: _TiltKind
    group: str
    code: float | None = None
    versus: float | None = None
    parameter: ArmParameter | None = None
    stem: str = ""

    @property
    def keeps_interval(self) -> bool:
        """Whether a row at ``gamma != 0`` keeps the inferential spread names.

        Only the arm kind does, which is the shipped contract.  Every other kind reports
        its spread at ``gamma != 0`` under the ``plugin_`` names (roadmap X57 part d).
        """
        return self.kind == "arm"


def _tilt_targets(result: Any) -> dict[str, _TiltTarget]:
    """Every reported parameter the tilt can re-mix, by name.

    Composed forward through :func:`~cleverly.targets.parameter_name` and
    :func:`~cleverly.targets.base.arm_alias` from the codes the fit carries, the rule
    :mod:`cleverly.sensitivity._parameters` follows.  A stratum alias is a different name,
    so it is never produced here.  The caller has already checked that the result is a
    point-treatment fit with arms and missing outcomes.
    """
    estimates = result.estimates
    out: dict[str, _TiltTarget] = {
        name: _TiltTarget(name, "arm", parameter.group, parameter=parameter)
        for name, parameter in reported_arm_parameters(result).items()
    }
    data = result.data
    nuisance = result.repeats[0].nuisance
    groups = result.repeats[0].fluctuations

    if nuisance.regimes is not None and "regime" in groups:
        regimes = nuisance.regimes
        label = dict(regimes.labels)
        for code in regimes.codes:
            name = parameter_name("ey_regime", arm=label[code])
            if name in estimates:
                out[name] = _TiltTarget(name, "regime", "regime", code)
            if code == regimes.reference:
                continue
            name = parameter_name("ate_regime", arm=label[code], versus=label[regimes.reference])
            if name in estimates:
                out[name] = _TiltTarget(name, "regime", "regime", code, regimes.reference)
        return out

    if nuisance.msm is not None and "msm" in groups:
        label = dict(nuisance.msm.labels)
        for code in nuisance.msm.codes:
            name = parameter_name("msm", arm=label[code])
            if name in estimates:
                out[name] = _TiltTarget(name, "msm", "msm", code)
        return out

    if "natural_course" in groups and NATURAL_COURSE_TARGET in estimates:
        out[NATURAL_COURSE_TARGET] = _TiltTarget(
            NATURAL_COURSE_TARGET, "natural_course", "natural_course"
        )
    if "mean" not in groups:
        return out
    reference = result.config.reference_arm
    collapse = data.is_binary_treatment
    for stem in ("par", "paf"):
        name = arm_alias(stem, arm=data.arm_label(reference), collapse=collapse)
        if name in estimates and "natural_course" in groups:
            out[name] = _TiltTarget(name, stem, "mean", reference)
    for arm in data.arm_codes:
        if arm == reference:
            continue
        for stem in ("rr", "or"):
            name = arm_alias(
                stem,
                arm=data.arm_label(arm),
                versus=data.arm_label(reference),
                collapse=collapse,
            )
            if name in estimates:
                out[name] = _TiltTarget(name, "ratio", "mean", arm, reference, stem=stem)
    return out


def _tilt_target_names(result: Any) -> tuple[str, ...]:
    """The reported names :func:`missingness_tilt` re-mixes, or ``()`` on a refused fit.

    The eligible set of ``tipping_gamma``, which the sensitivity facade reads to decide
    whether a bare call names one parameter.
    """
    if fit_wide_tilt_refusal(result) is not None:
        return ()
    return tuple(name for name in result.estimates if name in _tilt_targets(result))


def _refuse_untiltable_parameters(result: Any) -> Refusal | None:
    """Refuse a fit that reports no parameter the tilt can re-mix.

    :func:`_tilt_targets` names what the tilt re-mixes.  A shift fit on a discrete
    treatment reports none of them, so every call would refuse with "no tiltable
    estimands requested", and the rule refuses the fit first.  A fit that reports at
    least one keeps the rule silent, and the default sweep skips the rest.
    """
    if _tilt_targets(result):
        return None
    shift = result.nuisance.policies is not None
    return Refusal(
        RefusalReason.IMPLEMENTATION_GAP,
        "missingness_tilt re-mixes the arm means and their contrasts and ratios, the "
        "natural-course mean and the attributable targets, the regime means, and the MSM "
        f"coefficients, and this fit reports none of them: {sorted(result.estimates)}.",
        alternative=(
            "Use truncation_curve(mechanism=True) for sensitivity to the missingness bound."
        ),
        roadmap="X57" if shift else "X82",
    )


#: Every missingness-tilt refusal that no argument of the call can lift, in the one order
#: the two entry points and the two capability rows use.  The table is ordered, and each
#: rule assumes its predecessors returned ``None``: ``longitudinal`` establishes that the
#: result carries point-treatment ``data``, which every later rule reads, and
#: ``missing_outcome`` establishes the observation mechanism that the arm-indexed rules
#: after it would tilt.  The names are the introspection contract: the capability row
#: reads the name to choose its status, and a test pins the order without respelling a
#: message.
_FIT_WIDE_TILT_RULES: tuple[tuple[str, Callable[[Any], Refusal | None]], ...] = (
    ("longitudinal", _refuse_longitudinal),
    ("learned_rule", _refuse_learned_rule),
    ("missing_outcome", _refuse_complete_outcome),
    ("continuous", _refuse_continuous),
    ("incremental", _refuse_incremental),
    ("tiltable_parameters", _refuse_untiltable_parameters),
)


def _fit_wide_tilt_rule(result: Any) -> tuple[str, Refusal] | None:
    """The first rule of :data:`_FIT_WIDE_TILT_RULES` that refuses, with its reason."""
    for name, rule in _FIT_WIDE_TILT_RULES:
        refusal = rule(result)
        if refusal is not None:
            return name, refusal
    return None


def fit_wide_tilt_refusal(result: Any) -> Refusal | None:
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
    Refusal or None
        The first fit-wide refusal, or ``None`` when no fit-wide boundary applies.
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

    Refuses first by ``fit_wide_tilt_refusal``, which the capability rows read too.  It
    refuses a longitudinal result, a learned-rule fit, a fit with no missing outcome, a
    continuous dose, an incremental fit, and a fit that reports no parameter the tilt
    re-mixes, such as a shift fit on a discrete treatment.

    The tilt re-mixes these reported parameters:

    ==================== =====================================================
    parameter            point curve at ``gamma``
    ==================== =====================================================
    arm mean, contrast   the mean or contrast of the tilted arm regressions
    ``rr``, ``or``       the ratio of two tilted arm means
    ``ey_obs``           the mean of the tilted regression at the received arm
    ``par``, ``paf``     ``ey_obs`` and the reference arm mean, both tilted
    regime mean          the regime density times the tilted arm regressions
    MSM coefficient      the working-model projection of the tilted regressions
    ==================== =====================================================

    At ``gamma = 0`` every curve passes through the reported estimate.  An arm mean or
    a linear contrast keeps the reported standard error and its interval names at every
    ``gamma``.  Every other parameter reports its spread at ``gamma != 0`` under
    ``plugin_std_err``, ``plugin_interval_lower`` and ``plugin_interval_upper``: the
    reported standard error on the estimate's inference scale, centred on the tilted
    point.  No interval is claimed there.  A frame that mixes the two kinds carries
    both sets of columns, with NaN where a row has no value.

    A fit whose inference status supplies no inference, such as a selector-path
    ``CTMLE`` fit, receives ``plugin_std_err``,
    ``plugin_interval_lower`` and ``plugin_interval_upper`` in place of ``std_err``,
    ``ci_lower`` and ``ci_upper`` at every ``gamma``.

    Parameters
    ----------
    result : TMLEResult
        A fitted result whose outcomes are missing for some rows.
    gamma : sequence of float or None
        Tilt values on the logit scale.  ``None`` uses :data:`DEFAULT_GAMMA_GRID`.
    estimands : sequence of str or None
        Restrict to a subset.  ``None`` sweeps every reported parameter the tilt
        re-mixes and skips the rest.
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
        raise CapabilityError.from_refusal(refusal)
    data = result.data
    if result.nuisance.missingness is None:  # pragma: no cover - guarded above
        raise CapabilityError(
            "missingness_tilt requires a fitted missingness mechanism",
            reason=RefusalReason.INVALID_DECLARATION,
        )

    tiltable = _tilt_targets(result)
    requested = tuple(estimands if estimands is not None else result.estimates)
    if estimands is not None:
        # Only an explicit request is refused. The default sweep skips whatever it cannot
        # tilt, such as a stratum alias, so asking for the whole report still returns the
        # tiltable part of it.
        for name in requested:
            conditional = stratum_refusal(result, name, "the MNAR tilt")
            if conditional is not None:
                raise CapabilityError.from_refusal(conditional)
    targets = tuple(tiltable[name] for name in requested if name in tiltable)
    if not targets:
        raise CapabilityError(
            "no tiltable estimands requested; the tilt applies to the parameters "
            f"{sorted(tiltable)} of this fit",
            reason=RefusalReason.INVALID_DECLARATION,
        )

    direction = _tilt_direction(data, arm_gamma)
    grid = tuple(DEFAULT_GAMMA_GRID if gamma is None else (float(g) for g in gamma))

    rows: list[dict[str, Any]] = []
    for value in grid:
        for target in targets:
            name = target.name
            estimate = result[name]
            # Combined over the cross-fitting draws, as the fit's own report was. Each
            # draw has its own targeted Qbar and its own missingness mechanism, and the
            # tilt is a function of both, so a tilt read off one draw would sit at a
            # different level from the psi at gamma = 0 that the fit reported -- the curve
            # would step at its own origin. The median is taken on the estimate's
            # inference scale, the log scale for a ratio, as ``median_estimates`` takes it.
            center = float(
                np.median(
                    [
                        _tilted_value(result, repeat, target, value, direction)
                        for repeat in result.repeats
                    ]
                )
            )
            psi = float(np.exp(center)) if target.kind == "ratio" else center
            status = estimate.inference
            if target.keeps_interval:
                # The names ``truncation_curve`` publishes, for the reason it publishes
                # them: a selector-path collaborative fit supplies no interval, so the
                # sweep reports the retained diagnostic under names that make no coverage
                # claim. The interval is centred on the tilted estimate, so only the names
                # come from ``spread_name``.
                std_error = estimate.plugin_std_error
                low, high = wald_ci(
                    psi, std_error, result.config.alpha_sig, df=estimate.reference_df
                )
                spread = {
                    spread_name("std_err", status): std_error,
                    spread_name("ci_lower", status): low,
                    spread_name("ci_upper", status): high,
                }
            else:
                spread = _plugin_spread(estimate, psi, center, mar=value == 0.0)
            rows.append(
                {
                    "gamma": value,
                    "estimand": name,
                    "psi": psi,
                    **spread,
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

    # A frame that mixes the two spread contracts has both sets of columns. The arm rows
    # come first in the column order, so a frame of arm parameters alone is unchanged.
    columns: dict[str, None] = {}
    for row in rows:
        columns.update(dict.fromkeys(row))
    nan = float("nan")
    payload = {key: [row.get(key, nan) for row in rows] for key in columns}
    return data.frame_like(payload)


def _plugin_spread(
    estimate: ParameterEstimate, psi: float, center: float, *, mar: bool
) -> dict[str, float]:
    """The spread columns of a parameter that claims no interval off ``gamma = 0``.

    At ``gamma = 0`` the row is the fit's own report, so it carries the fit's spread under
    the names the fit's inference status gives.  At ``gamma != 0`` it carries the reported
    standard error on the estimate's inference scale, centred on the tilted point, under
    the ``plugin_`` names.  The reported influence curve is the MAR curve, so the spread is
    a diagnostic and not a confidence interval for the tilted parameter.
    """
    status = estimate.inference
    if mar:
        names = {key: spread_name(key, status) for key in _SPREAD_KEYS}
        low, high = estimate.plugin_interval
    else:
        names = {key: _DIAGNOSTIC_NAMES[key] for key in _SPREAD_KEYS}
        moved = replace(
            estimate, psi=psi, log_psi=center if estimate.scale == "ratio" else estimate.log_psi
        )
        low, high = moved.plugin_interval
    return {
        names["std_err"]: estimate.plugin_std_error,
        names["ci_lower"]: low,
        names["ci_upper"]: high,
    }


#: The inferential spread columns, in the order every tilt row publishes them.
_SPREAD_KEYS = ("std_err", "ci_lower", "ci_upper")


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


def _tilted_value(
    result: TMLEResult,
    repeat: RepeatFit,
    target: _TiltTarget,
    gamma: float,
    direction: Mapping[float, float],
) -> float:
    """One parameter under one cross-fitting draw at tilt ``gamma``, on its inference scale.

    Reads the targeted ``Qbar`` and the missingness mechanism from the *same* draw, which
    is the whole reason :class:`~cleverly.estimators._nuisance.RepeatFit` holds them
    together: mixing one draw's regression with another's mechanism would produce a
    perfectly plausible number for a fit that never happened.

    Each branch repeats the arithmetic of the target that reported the parameter, on the
    tilted regressions in place of the targeted ones, so ``gamma = 0`` returns the
    reported value.  A ratio returns its logarithm, which is the scale the repeats are
    combined on.
    """
    data = result.data
    weights = data.weights
    scaler = repeat.nuisance.scaler
    if target.kind == "arm":
        assert target.parameter is not None
        return _tilted_arm_parameter(result, repeat, target.parameter, gamma, direction)
    if target.kind == "msm":
        return _tilted_msm(result, repeat, target, gamma, direction)
    if target.kind == "regime":
        arms = _tilted_arms(result, repeat, "regime", gamma, direction)
        assert repeat.nuisance.regimes is not None
        density = np.asarray(repeat.nuisance.regimes.values, dtype=float)
        levels = repeat.fluctuations["regime"].targeted.levels
        predictions = np.column_stack([arms[level] for level in levels])

        def regime_mean(code: float) -> float:
            mixture = np.einsum("ij,ij->i", density[:, :, int(code)], predictions)
            return float(np.average(mixture, weights=weights))

        assert target.code is not None
        if target.versus is None:
            return _unscale(scaler, regime_mean(target.code), "level")
        return _unscale(scaler, regime_mean(target.code) - regime_mean(target.versus), "difference")

    observed = (
        _tilted_natural_course(result, repeat, gamma, direction)
        if target.kind
        in {
            "natural_course",
            "par",
            "paf",
        }
        else None
    )
    if target.kind == "natural_course":
        assert observed is not None
        return _unscale(scaler, observed, "level")
    arms = _tilted_arms(result, repeat, target.group, gamma, direction)
    assert target.code is not None
    reference = float(np.average(arms[target.code], weights=weights))
    if target.kind == "par":
        assert observed is not None
        return _unscale(scaler, observed - reference, "difference")
    if target.kind == "paf":
        # On the scaled means, as the ``paf`` target computes it, and not unscaled.
        assert observed is not None
        return 1.0 - reference / observed
    # A ratio, on the scaled means as ``_ratio_contrasts`` reads them. The two helpers
    # refuse a mean the log cannot take, as the fit's own target does.
    assert target.versus is not None
    versus = float(np.average(arms[target.versus], weights=weights))
    log_ratio = log_ratio_influence if target.stem == "rr" else log_odds_ratio_influence
    zero = np.zeros(1)
    log_psi, _ = log_ratio(reference, zero, versus, zero)
    return log_psi


def _unscale(scaler: Any, value: float, scale: Literal["level", "difference"]) -> float:
    """Map a scaled mean or difference to the outcome's units, as ``finish`` does."""
    if scaler.is_identity:
        return value
    return scaler.unscale_level(value) if scale == "level" else scaler.unscale_difference(value)


def _tilted_arms(
    result: TMLEResult,
    repeat: RepeatFit,
    group: str,
    gamma: float,
    direction: Mapping[float, float],
) -> dict[float, FloatArray]:
    """The tilted full-population regression at every arm, on the scaled outcome.

    Every arm is reached by its code -- the column of the mechanism included, which is
    keyed by arm exactly as ``targeted.arms`` is -- rather than by position.
    """
    missingness = tilt_missingness(repeat.nuisance, result.config.missingness_bound)
    assert missingness is not None
    arms = repeat.nuisance.arms
    targeted = repeat.fluctuations[group].targeted
    return {
        arm: _tilted(targeted.arms[arm], missingness[:, arms.index(arm)], gamma * direction[arm])
        for arm in targeted.arms
    }


def _tilted_arm_parameter(
    result: TMLEResult,
    repeat: RepeatFit,
    parameter: ArmParameter,
    gamma: float,
    direction: Mapping[float, float],
) -> float:
    """An arm mean or a linear contrast of the arm means, on the reported scale."""
    data = result.data
    weights = data.weights
    scaler = repeat.nuisance.scaler
    arms = _tilted_arms(result, repeat, parameter.group, gamma, direction)
    if parameter.versus is None:
        return _unscale(scaler, float(np.average(arms[parameter.arm], weights=weights)), "level")
    contrast = arms[parameter.arm] - arms[parameter.versus]
    conditioning = parameter.conditions_on
    if conditioning is None:
        psi_scaled = float(np.average(contrast, weights=weights))
    else:
        # The ATT and ATC average the contrast over one arm only, so the arm indicator
        # multiplies the observation weights -- and which arm that is differs per
        # contrast for the ATT, which is why it is read off the parameter.
        indicator = np.asarray(data.treatment == conditioning, dtype=float)
        psi_scaled = float(np.average(contrast, weights=weights * indicator))
    return _unscale(scaler, psi_scaled, "difference")


def _tilted_natural_course(
    result: TMLEResult,
    repeat: RepeatFit,
    gamma: float,
    direction: Mapping[float, float],
) -> float:
    r"""The tilted natural-course mean on the scaled outcome.

    .. math::

        \hat\psi_\gamma = P_n\bigl[\pi(A, W)\,\bar Q^*(A, W)
          + \{1 - \pi(A, W)\}\,\operatorname{expit}\{\operatorname{logit}\bar Q^*(A, W)
          + \gamma_A\}\bigr]

    It reads the natural-course fluctuation's regression at the received arm and the
    mechanism at the received arm, which is the column the natural-course clever
    covariate divides by.  The plug-in replaces the empirical ``Delta Y`` of the observed
    rows by its regression, :math:`\pi \bar Q^*`, so the curve passes through the
    reported estimate at :math:`\gamma = 0`.  The two forms estimate one functional.
    """
    data = result.data
    missingness = tilt_missingness(repeat.nuisance, result.config.missingness_bound)
    assert missingness is not None
    arms = repeat.nuisance.arms
    treatment = np.asarray(data.treatment, dtype=float)
    realised = np.zeros(treatment.shape[0], dtype=float)
    shift = np.zeros(treatment.shape[0], dtype=float)
    for column, arm in enumerate(arms):
        mask = treatment == arm
        realised[mask] = missingness[mask, column]
        shift[mask] = gamma * direction[arm]
    prediction = np.asarray(repeat.fluctuations["natural_course"].targeted.observed, dtype=float)
    tilted = np.where(
        shift == 0.0,
        prediction,
        realised * prediction + (1.0 - realised) * expit(logit(prediction) + shift),
    )
    return float(np.average(tilted, weights=data.weights))


def _tilted_msm(
    result: TMLEResult,
    repeat: RepeatFit,
    target: _TiltTarget,
    gamma: float,
    direction: Mapping[float, float],
) -> float:
    """One working-model coefficient, projected from the tilted arm regressions.

    The projection is :func:`~cleverly.msm.solve_projection` on the outcome's own scale,
    with the fitted design, weights and link, as
    :func:`~cleverly.inference.influence.msm_coefficients` solves it.  The tilt moves the
    counterfactual regression row by row, and the projection is then fixed.
    """
    msm = repeat.nuisance.msm
    assert msm is not None and target.code is not None
    scaler = repeat.nuisance.scaler
    arms = _tilted_arms(result, repeat, "msm", gamma, direction)
    levels = repeat.fluctuations["msm"].targeted.levels
    predictions = scaler.unscale_levels(np.column_stack([arms[level] for level in levels]))
    fit = solve_projection(msm.design, msm.weights, predictions, result.data.weights, str(msm.link))
    return float(fit.beta[int(target.code)])


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


def tipping_interval_refusal(result: Any, estimand: str | None) -> Refusal | None:
    """Why ``tipping_gamma(use_ci=True)`` cannot follow an interval for ``estimand``.

    The tilt of a ratio, a natural-course or attributable target, a regime mean or an
    MSM coefficient claims no interval at ``gamma != 0``, so the interval search has no
    confidence limit to follow.  The point search answers.  ``None`` for an arm mean or a
    linear contrast, and for a name the tilt does not re-mix, which the call refuses
    later with the list of the names it does.

    Parameters
    ----------
    result : Any
        A fitted result that the fit-wide tilt rules admit.
    estimand : str or None
        The requested name.

    Returns
    -------
    Refusal or None
        The refusal, or ``None`` when the interval search can run.
    """
    if estimand is None or fit_wide_tilt_refusal(result) is not None:
        return None
    target = _tilt_targets(result).get(estimand)
    if target is None or target.keeps_interval:
        return None
    return Refusal(
        RefusalReason.IMPLEMENTATION_GAP,
        f"{_TIPPING_INTERVAL_OPERATION} follows a confidence limit, and the tilt of "
        f"{estimand!r} claims no interval at gamma != 0. Its spread there is the MAR "
        "standard error centred on the tilted point, published under the plugin_ names.",
        alternative=f"Call tipping_gamma(result, {estimand!r}) for the point search.",
        roadmap="X57",
    )


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

    Refuses first by ``fit_wide_tilt_refusal``, the table :func:`missingness_tilt`
    reads, because it searches over that same tilt.  The capability row quotes the same
    sentence.

    Refuses ``use_ci=True`` on a fit whose inference status supplies no inference, such
    as a selector-path ``CTMLE`` fit. Such a fit supplies no confidence limit for the
    search to follow.  It refuses ``use_ci=True`` for a ratio, a natural-course or
    attributable target, a regime mean and an MSM coefficient too, whose tilt claims no
    interval at ``gamma != 0``.  ``use_ci=False``, the default, searches the point
    estimate and answers for every one of them.

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
        raise CapabilityError.from_refusal(refusal)
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
    if use_ci:
        interval = tipping_interval_refusal(result, estimand)
        if interval is not None:
            raise CapabilityError.from_refusal(interval)
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
