r"""E-values (VanderWeele & Ding, 2017).

The E-value answers the same question as the omitted-variable bound but on the risk
ratio scale, and with a different parameterisation that has become the convention in
epidemiology:

    *the minimum strength of association, on the risk ratio scale, that an
    unmeasured confounder would need with both the treatment and the outcome --
    conditional on the measured covariates -- to fully explain away the observed
    association.*

For an observed risk ratio :math:`RR \ge 1`,

.. math:: E = RR + \sqrt{RR\,(RR - 1)}

and for :math:`RR < 1` the same formula is applied to :math:`1/RR`.  An E-value of 1
means no unmeasured confounding at all is needed -- the estimate is already
compatible with the null.  A large E-value means only an implausibly strong
confounder could account for the finding.

The E-value for the *confidence limit* is usually the more important number: it says
how strong a confounder would need to be to move the interval to include the null,
which is the claim a sceptical reader will make.

Conversions for other effect scales are approximations and are flagged as such:

* **odds ratio**, rare outcome: :math:`RR \approx \sqrt{OR}` (Ding & VanderWeele).
* **continuous outcome**: standardise the difference by the outcome's standard
  deviation, then :math:`RR \approx \exp(0.91 \times d)` (Chinn, 2000; VanderWeele
  2017).  This assumes approximate normality and is only a rough guide -- prefer
  :func:`cleverly.sensitivity.omitted_variable_bounds` for a continuous outcome,
  which needs no such conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..exceptions import CapabilityError
from ..targets import parameter_stem
from ._parameters import arm_parameters

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..estimators.base import TMLEResult

__all__ = ["EValue", "evalue", "evalue_from_rr"]

#: Chinn's factor for converting a standardised mean difference to a risk ratio.
_SMD_TO_LOG_RR = 0.91


def evalue_from_rr(risk_ratio: float) -> float:
    """E-value for a risk ratio.

    >>> round(evalue_from_rr(2.0), 4)
    3.4142
    >>> evalue_from_rr(1.0)
    1.0
    >>> round(evalue_from_rr(0.5), 4)   # symmetric under inversion
    3.4142
    """
    rr = float(risk_ratio)
    if not np.isfinite(rr) or rr <= 0:
        return float("nan")
    if rr < 1.0:
        rr = 1.0 / rr
    return float(rr + np.sqrt(rr * (rr - 1.0)))


def _evalue_for_limit(limit: float, *, above_null: bool) -> float:
    """E-value for a confidence limit, which is 1 when the limit crosses the null."""
    if not np.isfinite(limit) or limit <= 0:
        return float("nan")
    if above_null:
        return 1.0 if limit <= 1.0 else evalue_from_rr(limit)
    return 1.0 if limit >= 1.0 else evalue_from_rr(limit)


@dataclass(frozen=True)
class EValue:
    """E-values for a point estimate and the confidence limit nearest the null.

    Parameters
    ----------
    estimand : str
        Alias of the estimand this row describes.
    scale : str
        Scale the estimate was reported on before conversion.
    risk_ratio : float
        The estimate expressed as a risk ratio.
    risk_ratio_ci : tuple of float
        That ratio's confidence interval.
    point : float
        Risk-ratio association an unmeasured confounder would need with both
        treatment and outcome to explain the point estimate away.
    limit : float
        The same association needed to move the confidence limit across the null.
    approximate : bool
        Whether reaching the risk-ratio scale needed an approximate conversion.
    note : str
        Explanation of that conversion, when one was used.
    """

    estimand: str
    scale: str
    risk_ratio: float
    risk_ratio_ci: tuple[float, float]
    point: float
    limit: float
    approximate: bool
    note: str = ""

    def summary(self) -> str:
        """Return a printable summary.

        Returns
        -------
        str
            A printable report, one line per reported quantity.
        """
        lines = [
            f"E-value for {self.estimand!r} ({self.scale} scale)",
            "-" * 40,
            f"risk ratio {self.risk_ratio:.4g} "
            f"[{self.risk_ratio_ci[0]:.4g}, {self.risk_ratio_ci[1]:.4g}]"
            + (" (approximate conversion)" if self.approximate else ""),
            f"E-value, point estimate     : {self.point:.4f}",
            f"E-value, confidence limit   : {self.limit:.4f}",
            "",
            f"An unmeasured confounder would need risk-ratio associations of at least "
            f"{self.point:.2f} with both treatment and outcome, above and beyond the "
            f"measured covariates, to explain away the point estimate; "
            + (
                f"{self.limit:.2f} to move the interval across the null."
                if self.limit > 1.0
                else "the interval already includes the null."
            ),
        ]
        if self.note:
            lines.extend(["", self.note])
        return "\n".join(lines)

    def to_dict(self) -> dict[str, float | str | bool]:
        """Return a JSON-compatible representation.

        Returns
        -------
        dict
            A JSON-compatible mapping of every reported field.
        """
        return {
            "estimand": self.estimand,
            "scale": self.scale,
            "risk_ratio": self.risk_ratio,
            "rr_ci_lower": self.risk_ratio_ci[0],
            "rr_ci_upper": self.risk_ratio_ci[1],
            "evalue_point": self.point,
            "evalue_limit": self.limit,
            "approximate": self.approximate,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return self.summary()


def _default_estimand(result: TMLEResult) -> str:
    """The parameter ``estimand=None`` reports on: a ratio if one was estimated.

    Chosen by *stem* over the parameters the fit reported, rather than by looking up the
    three bare names, which exist only on a two-armed fit -- there they are the only
    parameters with those stems and this is the rule it always followed.  With more arms
    the first contrast in report order stands in for the report as a whole, the same way
    :meth:`~cleverly.assessment.SensitivityFacade.evalue` picks one.
    """
    for stem in ("rr", "or", "ate"):
        for name in result.estimates:
            if parameter_stem(name) == stem:
                return name
    raise CapabilityError(
        "no estimand suitable for an E-value was estimated; an E-value is a statement "
        f"about a contrast, and this fit reported {sorted(result.estimates)}"
    )


def _baseline_mean(result: TMLEResult, estimand: str) -> str | None:
    """The reported mean under the arm ``estimand`` is contrasted against, if any.

    The risk-difference conversion divides by the baseline risk, and the baseline is the
    contrast's *reference* arm rather than arm ``0``: with ``reference=1`` a two-armed
    fit's ``ate`` is ``E[Y^0] - E[Y^1]`` and dividing by ``EY0`` would convert a
    difference into a ratio of the wrong pair.  Found by arm through
    :func:`~cleverly.sensitivity._parameters.arm_parameters`, so ``ey0`` and ``ey[low]``
    are both recognised -- whichever of them the fit was asked for.
    """
    known = arm_parameters(result)
    parameter = known.get(estimand)
    if parameter is None or parameter.versus is None:  # pragma: no cover - defensive
        return None
    return next(
        (
            name
            for name, candidate in known.items()
            if name in result.estimates
            and candidate.versus is None
            and candidate.arm == parameter.versus
        ),
        None,
    )


def evalue(result: TMLEResult, estimand: str | None = None) -> EValue:
    """E-value for a fitted result.

    ``estimand=None`` picks the most appropriate available: a risk ratio if one was
    estimated, else an odds ratio, else a risk difference via the
    standardised-difference approximation.

    A contrast is named for its arms on a fit with more than two of them, so the
    estimand here is ``"rr[medium vs low]"`` rather than ``"rr"``, and the *stem* is what
    picks the conversion.  One E-value per contrast, on the same terms as one
    omitted-variable bound per contrast: each is a statement about the two arms it names.

    Parameters
    ----------
    result : TMLEResult
        A fitted result.
    estimand : str or None
        Alias to report on. ``None`` picks the ratio-scale estimand the fit
        reported, which is the scale an E-value is defined on.

    Returns
    -------
    EValue
        The association needed to explain the estimate and the interval away.
    """
    if estimand is None:
        estimand = _default_estimand(result)
    if estimand not in result.estimates:
        raise CapabilityError(f"estimand {estimand!r} was not requested in this fit")

    estimate = result[estimand]
    low, high = estimate.ci
    stem = parameter_stem(estimand)

    if stem == "rr":
        rr, ci, approximate, note = estimate.psi, (low, high), False, ""
    elif stem == "or":
        rr = float(np.sqrt(estimate.psi))
        ci = (float(np.sqrt(low)), float(np.sqrt(high)))
        approximate = True
        note = (
            "Converted from the odds ratio with RR ~ sqrt(OR), which assumes a rare "
            "outcome. With a common outcome this understates the risk ratio and so the "
            "E-value is conservative in the wrong direction; prefer estimands=('rr',...)."
        )
    elif stem in ("ate", "att", "atc"):
        if result.config.family == "binomial":
            baseline_name = _baseline_mean(result, estimand)
            baseline = None if baseline_name is None else result.estimates[baseline_name]
            if baseline is None or baseline.psi <= 0:
                raise CapabilityError(
                    "converting a risk difference to a risk ratio needs the mean under the "
                    f"arm {estimand!r} is contrasted against; add 'ey' (or 'ey0'/'ey1') to "
                    "estimands, or request a risk ratio directly"
                )
            rr = float((baseline.psi + estimate.psi) / baseline.psi)
            ci = (
                float((baseline.psi + low) / baseline.psi),
                float((baseline.psi + high) / baseline.psi),
            )
            approximate = True
            note = (
                "Converted from a risk difference using the estimated baseline risk. The "
                "conversion holds the baseline fixed and so ignores its sampling error."
            )
        else:
            observed = result.data.outcome[result.data.observed]
            sd = float(np.std(observed, ddof=1))
            if sd <= 0:
                raise CapabilityError(
                    "cannot standardise the effect: the outcome has zero variance"
                )
            rr = float(np.exp(_SMD_TO_LOG_RR * estimate.psi / sd))
            ci = (
                float(np.exp(_SMD_TO_LOG_RR * low / sd)),
                float(np.exp(_SMD_TO_LOG_RR * high / sd)),
            )
            approximate = True
            note = (
                "Continuous outcome: standardised by sd(Y) = "
                f"{sd:.4g} and converted with RR ~ exp(0.91 d) (Chinn, 2000). This assumes "
                "approximate normality and is a rough guide only -- "
                "sensitivity.omitted_variable_bounds() needs no such conversion and is "
                "preferable here."
            )
    else:
        # Reachable for any reported parameter that is not one of the arm-indexed
        # contrasts: a counterfactual mean, an MSM coefficient, a contrast of two
        # regimes. A level has no association to explain away at all; the other axes have
        # one, and converting it would need that axis's own baseline rather than an arm's.
        raise CapabilityError(
            f"cannot compute an E-value for estimand {estimand!r}: the conversion is "
            "written for a contrast of two arms (ate/att/atc, rr, or), and this is not "
            "one of those. Request a risk ratio, an odds ratio or a risk difference."
        )

    above_null = rr >= 1.0
    limit = ci[0] if above_null else ci[1]
    return EValue(
        estimand=estimand,
        scale="risk ratio",
        risk_ratio=rr,
        risk_ratio_ci=ci,
        point=evalue_from_rr(rr),
        limit=_evalue_for_limit(limit, above_null=above_null),
        approximate=approximate,
        note=note,
    )
