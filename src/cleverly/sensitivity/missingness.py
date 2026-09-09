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
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from .._typing import FloatArray
from ..exceptions import CapabilityError
from ..inference.delta import normal_ci
from ..utils.bounds import expit, logit
from ._parameters import ArmParameter, arm_parameters, stratum_refusal

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
        One multiplier per arm, keyed by the treatment level as the caller wrote it, so
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
    data = result.data
    if not data.has_missing_outcome:
        raise CapabilityError(
            "missingness_tilt requires a fit with missing outcomes. Pass delta=<column> to "
            "fit() so the missingness mechanism is estimated."
        )
    if data.is_continuous_treatment:
        raise CapabilityError(
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
    if result.nuisance.incremental is not None:
        raise CapabilityError(
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
    if result.nuisance.missingness is None:  # pragma: no cover - guarded above
        raise CapabilityError("missingness_tilt requires a fitted missingness mechanism")

    # Which parameters the tilt can re-mix: the arm-indexed linear ones, named for their
    # arms on a fit with more than two. Ratios are excluded by being absent from that
    # map rather than by a second filter here.
    tiltable = {
        name: parameter
        for name, parameter in arm_parameters(result).items()
        if name in result.estimates
    }
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
            std_error = result[name].std_error
            low, high = normal_ci(psi, std_error, result.config.alpha_sig)
            rows.append(
                {
                    "gamma": value,
                    "estimand": name,
                    "psi": psi,
                    "std_err": std_error,
                    "ci_lower": low,
                    "ci_upper": high,
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
        direction[float(matches[0])] = float(multiplier)
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

    Returns the smallest ``|gamma|`` at which the estimate (or, with ``use_ci=True``,
    the confidence interval) reaches ``null_hypothesis``, or ``None`` if no tilt within
    ``search`` does so.  Reporting this single number is usually more informative than
    the whole curve: it converts "is MAR plausible?" into "would the unobserved
    outcomes have to differ by *this much*?".

    With ``arm_gamma=`` the number is the magnitude at which that *direction* tips the
    conclusion, which is what makes one scalar still meaningful when the arms are tilted
    by different amounts.

    Parameters
    ----------
    result : TMLEResult
        A fitted result with missing outcomes.
    estimand : str
        Alias to search over.
    null_hypothesis : float
        The value the conclusion is said to tip at.
    search : tuple of float
        Lower and upper tilt the search brackets.
    use_ci : bool
        Whether to tip when the confidence limit reaches the null rather than the
        point estimate.
    arm_gamma : mapping of level to float, or None
        One multiplier per arm, as :func:`missingness_tilt` accepts.

    Returns
    -------
    float or None
        The tilt at which the conclusion reaches its null, or ``None`` when no tilt
        inside ``search`` does.
    """
    from scipy import optimize

    def deviation(value: float) -> float:
        frame = missingness_tilt(result, [value], estimands=[estimand], arm_gamma=arm_gamma)
        import narwhals as nw

        row = nw.from_native(frame, eager_only=True)
        if use_ci:
            low = float(row["ci_lower"][0])
            high = float(row["ci_upper"][0])
            if low <= null_hypothesis <= high:
                return 0.0
            return min(abs(low - null_hypothesis), abs(high - null_hypothesis))
        return float(row["psi"][0]) - null_hypothesis

    baseline = deviation(0.0)
    if baseline == 0.0:
        return 0.0
    sign = np.sign(baseline)

    for direction in (1.0, -1.0):
        edge = direction * max(abs(search[0]), abs(search[1]))
        if np.sign(deviation(edge)) == sign and deviation(edge) != 0.0:
            continue
        try:
            root = optimize.brentq(deviation, 0.0, edge, xtol=1e-4, maxiter=100)
        except (ValueError, RuntimeError):  # pragma: no cover - no sign change
            continue
        return float(root)
    return None
