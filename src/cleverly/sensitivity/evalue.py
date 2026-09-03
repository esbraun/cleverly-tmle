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

* **odds ratio**, common outcome: :math:`RR \approx \sqrt{OR}` (VanderWeele, 2017).
  This approximation can lie above or below the risk ratio. For rare outcomes,
  the odds ratio itself approximates the risk ratio.
* **continuous outcome**: standardise the difference by the outcome's standard
  deviation. Chinn's :math:`d \approx \log(OR) / 1.81` gives an odds ratio; the
  additional common-outcome conversion :math:`RR \approx \sqrt{OR}` then gives
  :math:`RR \approx \exp(0.905 \times d)`. This assumes approximate normality and is
  only a rough guide -- prefer
  :func:`cleverly.sensitivity.omitted_variable_bounds` for a continuous outcome,
  which needs no such conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from ..exceptions import CapabilityError
from ._derived import _derived_risk_ratio, _risk_ratio_refusal

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..estimators.base import TMLEResult

__all__ = ["EValue", "evalue", "evalue_from_rr"]

#: Chinn's OR-to-SMD factor followed by the common-outcome square-root conversion.
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


class _EValueRefusal(CapabilityError):
    def __init__(self, status: str, reason: str) -> None:
        super().__init__(reason)
        self.status = status


@dataclass(frozen=True)
class _EValueSelection:
    source: str
    branch: str


def _default_estimand(result: TMLEResult) -> str:
    """Choose a reported contrast from structured parameter identities."""
    for target in ("rr", "or", "ate", "att", "atc"):
        candidates = [
            name
            for name in result.estimates
            if (key := result.parameter_keys.get(name)) is not None
            and key.estimand == target
            and key.axis == "arm"
            and key.reference is not None
            and key.stratum is None
        ]
        if len(candidates) > 1:
            raise _EValueRefusal(
                "unavailable",
                f"an E-value needs one contrast; choose an explicit estimand from {candidates}",
            )
        if candidates:
            return candidates[0]
    raise _EValueRefusal(
        "not_applicable",
        "an E-value needs a supported two-arm contrast; this fit reports no such parameter",
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
    parameter = result.parameter_keys.get(estimand)
    if parameter is None or parameter.reference is None:
        return None
    return next(
        (
            name
            for name, candidate in result.parameter_keys.items()
            if name in result.estimates
            and candidate.axis == "arm"
            and candidate.estimand in {"ey", "ey0", "ey1"}
            and candidate.reference is None
            and candidate.value == parameter.reference
            and candidate.stratum is None
        ),
        None,
    )


def _select_evalue(result: TMLEResult, estimand: str | None) -> _EValueSelection:
    """Select the exact, approximate, or refusal branch for one request."""
    if getattr(result, "assessment_family", None) != "point":
        raise _EValueRefusal(
            "unavailable",
            "no longitudinal sensitivity derivation is registered for an E-value",
        )
    if not result.parameter_keys:
        raise _EValueRefusal(
            "unavailable", "E-values require structured parameter keys retained by the result"
        )
    explicit = estimand is not None
    source = _default_estimand(result) if estimand is None else estimand
    if source not in result.estimates:
        raise _EValueRefusal("unavailable", f"estimand {source!r} was not requested in this fit")
    key = result.parameter_keys.get(source)
    if key is None:
        raise _EValueRefusal("unavailable", f"estimand {source!r} has no structured parameter key")
    if key.axis != "arm" or key.reference is None or key.stratum is not None:
        raise _EValueRefusal(
            "not_applicable",
            f"an E-value needs an unconditioned two-arm contrast, not axis {key.axis!r}",
        )
    if key.estimand == "rr":
        return _EValueSelection(source, "reported_rr")
    if key.estimand == "or" and explicit:
        return _EValueSelection(source, "reported_or")
    if key.estimand in {"att", "atc"}:
        raise _EValueRefusal(
            "unavailable",
            f"{key.estimand.upper()} needs a conditional baseline risk and a supported "
            "conditional ratio target; neither is available",
        )
    if key.estimand not in {"ate", "or"}:
        raise _EValueRefusal(
            "not_applicable",
            f"an E-value has no supported two-arm contrast for target {key.estimand!r}",
        )
    if result.config.family == "gaussian":
        if key.estimand != "ate":
            raise _EValueRefusal("unavailable", "a Gaussian E-value requires a marginal ATE")
        return _EValueSelection(source, "gaussian_ate")
    refusal = _risk_ratio_refusal(result, source)
    if refusal is None:
        return _EValueSelection(source, "derived_rr")
    baseline = _baseline_mean(result, source)
    if (
        key.estimand == "ate"
        and result.assessment_method == "tmle"
        and result.estimator is None
        and baseline is not None
        and result[baseline].psi > 0
    ):
        return _EValueSelection(source, "fixed_baseline_ate")
    if key.estimand == "ate" and result.estimator is None and baseline is None:
        refusal += "; the matching reported reference-arm mean is also absent"
    raise _EValueRefusal("unavailable", refusal)


def _evalue_capability(
    result: TMLEResult, estimand: str | None = None
) -> tuple[bool, str | None, str | None, Literal["summarize", "retarget"]]:
    """Return availability, refusal, and execution class for the actual request."""
    try:
        selected = _select_evalue(result, estimand)
    except _EValueRefusal as error:
        return False, error.status, str(error), "summarize"
    return True, None, None, "retarget" if selected.branch == "derived_rr" else "summarize"


def evalue(result: TMLEResult, estimand: str | None = None) -> EValue:
    """E-value for a fitted result.

    ``estimand=None`` prefers a reported risk ratio. For an eligible typed binomial
    ordinary-TMLE result that reports only a marginal ATE or odds ratio, it retargets the
    retained nuisances to the matching exact risk ratio. An explicit odds-ratio request
    preserves the caller's chosen square-root approximation. Routing uses structured
    parameter keys; display aliases are never parsed for arm identity.

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
    selection = _select_evalue(result, estimand)
    source = selection.source
    estimate = result[source]
    low, high = estimate.ci
    reported_estimand = source

    if selection.branch == "reported_rr":
        rr, ci, approximate, note, scale = (
            estimate.psi,
            (low, high),
            False,
            "",
            "risk ratio",
        )
    elif selection.branch == "reported_or":
        rr = float(np.sqrt(estimate.psi))
        ci = (float(np.sqrt(low)), float(np.sqrt(high)))
        approximate = True
        scale = "odds ratio"
        note = (
            "Converted with RR ~ sqrt(OR), an approximation for common outcomes. "
            "It can lie above or below the risk ratio; prefer a directly estimated risk ratio."
        )
    elif selection.branch == "derived_rr":
        derived = _derived_risk_ratio(result, source)
        rr, ci, approximate, scale = derived.psi, derived.ci, False, "risk ratio"
        reported_estimand = derived.name
        note = (
            f"Derived {derived.name!r} from cached nuisances by retargeting the reported "
            f"source contrast {source!r}; no nuisance model was refit."
        )
    elif selection.branch == "fixed_baseline_ate":
        baseline_name = _baseline_mean(result, source)
        assert baseline_name is not None
        baseline = result[baseline_name]
        rr = float((baseline.psi + estimate.psi) / baseline.psi)
        ci = (
            float((baseline.psi + low) / baseline.psi),
            float((baseline.psi + high) / baseline.psi),
        )
        approximate, scale = True, "risk difference"
        note = (
            "Converted from a risk difference using the reported reference-arm risk. "
            "The conversion holds that risk fixed and ignores its sampling error."
        )
    else:
        observed = result.data.outcome[result.data.observed]
        sd = float(np.std(observed, ddof=1))
        if sd <= 0:
            raise CapabilityError("cannot standardise the effect: the outcome has zero variance")
        rr = float(np.exp(_SMD_TO_LOG_RR * estimate.psi / sd))
        ci = (
            float(np.exp(_SMD_TO_LOG_RR * low / sd)),
            float(np.exp(_SMD_TO_LOG_RR * high / sd)),
        )
        approximate, scale = True, "mean difference"
        note = (
            f"Standardised by sd(Y) = {sd:.4g}. Chinn's log(OR) / 1.81 step maps the "
            "standardised mean difference to an odds ratio. The subsequent common-outcome "
            "square-root conversion and E-value calculation are additional approximations."
        )

    above_null = rr >= 1.0
    limit = ci[0] if above_null else ci[1]
    return EValue(
        estimand=reported_estimand,
        scale=scale,
        risk_ratio=rr,
        risk_ratio_ci=ci,
        point=evalue_from_rr(rr),
        limit=_evalue_for_limit(limit, above_null=above_null),
        approximate=approximate,
        note=note,
    )
