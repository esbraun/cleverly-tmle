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

Caveat, stated plainly: the confidence intervals on the curve treat :math:`\gamma`
as known and reuse the MAR standard error.  They describe sampling uncertainty at a
fixed :math:`\gamma`, not uncertainty about :math:`\gamma` itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from .._typing import FloatArray
from ..inference.delta import normal_ci
from ..utils.bounds import expit, logit

if TYPE_CHECKING:  # pragma: no cover - typing only
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
) -> Any:
    """Estimate under a range of departures from missingness-at-random.

    Returns a tidy frame with one row per ``(gamma, estimand)``.  Only defined for a
    fit that supplied ``delta``; without missing outcomes there is nothing to tilt.

    Parameters
    ----------
    gamma:
        Tilt values on the logit scale.  ``None`` uses :data:`DEFAULT_GAMMA_GRID`.
    estimands:
        Restrict to a subset.  Ratios are excluded automatically: tilting changes the
        counterfactual means, and re-deriving a ratio's log-scale influence curve under
        the tilt would misrepresent the uncertainty.
    """
    data = result.data
    if not data.has_missing_outcome:
        raise ValueError(
            "missingness_tilt requires a fit with missing outcomes. Pass delta=<column> to "
            "fit() so the missingness mechanism is estimated."
        )
    if not data.is_binary_treatment:
        raise ValueError(
            "missingness_tilt is written for a binary treatment; this fit has "
            f"{data.n_arms} arms {list(data.treatment_levels)}. The tilt moves each arm's "
            "missingness mechanism by one shared gamma and reports the estimands that name "
            "two arms (ate/att/atc/ey1/ey0), none of which a multi-arm fit produces. "
            "Extending it means deciding whether gamma is shared across arms or per arm, "
            "which is a modelling choice this function should not make silently."
        )
    if result.nuisance.missingness is None:  # pragma: no cover - guarded above
        raise ValueError("missingness_tilt requires a fitted missingness mechanism")

    names = tuple(
        name
        for name in (estimands if estimands is not None else result.estimates)
        if name in ("ate", "att", "atc", "ey1", "ey0")
    )
    if not names:
        raise ValueError("no tiltable estimands requested; the tilt applies to ate/att/atc/ey1/ey0")

    grid = tuple(DEFAULT_GAMMA_GRID if gamma is None else (float(g) for g in gamma))

    rows: list[dict[str, Any]] = []
    for value in grid:
        for name in names:
            # Averaged over the cross-fitting draws, as the fit's own report was. Each
            # draw has its own targeted Qbar and its own missingness mechanism, and the
            # tilt is a function of both, so a tilt read off one draw would sit at a
            # different level from the psi at gamma = 0 that the fit reported -- the curve
            # would step at its own origin. Every estimand here is a level or a
            # difference, so the plain mean is the right average; there is no ratio to
            # take on the log scale.
            psi = float(
                np.mean([_tilted_psi(result, repeat, name, value) for repeat in result.repeats])
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
                }
            )

    payload = {key: [row[key] for row in rows] for key in rows[0]}
    return data.frame_like(payload)


def _tilted_psi(result: TMLEResult, repeat: RepeatFit, name: str, gamma: float) -> float:
    """One estimand under one cross-fitting draw, at tilt ``gamma``.

    Reads the targeted ``Qbar`` and the missingness mechanism from the *same* draw, which
    is the whole reason :class:`~cleverly.estimators._nuisance.RepeatFit` holds them
    together: mixing one draw's regression with another's mechanism would produce a
    perfectly plausible number for a fit that never happened.
    """
    data = result.data
    scaler = repeat.nuisance.scaler
    missingness = repeat.nuisance.bounded_missingness(result.config.missingness_bound)
    assert missingness is not None
    weights = data.weights

    group = "mean" if name in ("ate", "ey1", "ey0") else name
    targeted = repeat.fluctuations[group].targeted
    full_one = _tilted(targeted.arms[1.0], missingness[:, 1], gamma)
    full_zero = _tilted(targeted.arms[0.0], missingness[:, 0], gamma)

    if name in ("ey1", "ey0"):
        psi_scaled = float(np.average(full_one if name == "ey1" else full_zero, weights=weights))
        return scaler.unscale_level(psi_scaled) if not scaler.is_identity else psi_scaled

    contrast = full_one - full_zero
    if name == "ate":
        psi_scaled = float(np.average(contrast, weights=weights))
    else:
        # The ATT and ATC average the contrast over one arm only, so the arm indicator
        # multiplies the observation weights.
        indicator = data.treatment if name == "att" else 1.0 - data.treatment
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
) -> float | None:
    """The tilt at which the conclusion tips.

    Returns the smallest ``|gamma|`` at which the estimate (or, with ``use_ci=True``,
    the confidence interval) reaches ``null_hypothesis``, or ``None`` if no tilt within
    ``search`` does so.  Reporting this single number is usually more informative than
    the whole curve: it converts "is MAR plausible?" into "would the unobserved
    outcomes have to differ by *this much*?".
    """
    from scipy import optimize

    def deviation(value: float) -> float:
        frame = missingness_tilt(result, [value], estimands=[estimand])
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
