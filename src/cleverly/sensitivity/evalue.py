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
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ..exceptions import CapabilityError
from ._derived import _derived_risk_ratio, _risk_ratio_refusal
from ._parameters import arm_parameter_keys

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..estimators.base import TMLEResult
    from ..inference.influence import ParameterEstimate

__all__ = ["EValue", "evalue", "evalue_from_rr"]

#: Chinn's OR-to-SMD factor followed by the common-outcome square-root conversion.
_SMD_TO_LOG_RR = 1.81 / 2


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
    """E-value for a confidence limit, which is 1 when the limit crosses the null.

    The null test runs before any positivity test. A limit outside the risk-ratio
    parameter space still says which side of the null the interval reaches, and an
    interval that already covers the null needs no confounding to explain it. The
    fixed-baseline conversion is affine, so a risk-difference limit below the negative of
    the baseline risk maps to a nonpositive ratio limit. A point ratio above the null
    reads that lower limit, and a nonpositive value there reports 1, not ``nan``. Only
    ``nan`` itself, which names no side of the null, reports ``nan``.

    A point ratio below the null reads the *upper* limit instead, and the fixed-baseline
    conversion never sends that bound out of the parameter space:
    :func:`~cleverly.inference.delta.normal_ci` gives ``high >= psi``, and
    :func:`_reject_unusable_baseline` has already refused a fit whose ``baseline.psi`` is
    nonpositive or whose ``baseline.psi + psi`` is nonpositive.
    """
    if np.isnan(limit):
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
        That ratio's confidence interval. A converted lower bound below zero is outside
        the risk-ratio parameter space. The report truncates it at zero, records the
        untruncated value in :attr:`truncated_bound`, and says so in :attr:`note`.
    point : float
        Risk-ratio association an unmeasured confounder would need with both
        treatment and outcome to explain the point estimate away.
    limit : float
        The same association needed to move the confidence limit across the null.
    approximate : bool
        Whether reaching the risk-ratio scale needed an approximate conversion.
    note : str
        Explanation of that conversion, when one was used.
    source_estimand : str or None
        Reported source alias, which can differ from a derived output alias.
    truncated_bound : float or None
        The converted lower bound before truncation, when the report truncated one.
        ``None`` when :attr:`risk_ratio_ci` reports the conversion unchanged.
    """

    estimand: str
    scale: str
    risk_ratio: float
    risk_ratio_ci: tuple[float, float]
    point: float
    limit: float
    approximate: bool
    note: str = ""
    source_estimand: str | None = None
    truncated_bound: float | None = None

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

    def to_dict(self) -> dict[str, float | str | bool | None]:
        """Return a JSON-compatible representation.

        ``truncated_bound`` carries the untruncated lower bound, so a caller who reads
        only this mapping still sees that ``rr_ci_lower`` is a boundary value rather than
        a converted confidence limit.

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
            "note": self.note,
            "source_estimand": self.source_estimand,
            "truncated_bound": self.truncated_bound,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return self.summary()


class _EValueRefusal(CapabilityError):
    def __init__(self, status: str, reason: str) -> None:
        super().__init__(reason)
        self.status = status


#: The conversions this module implements. Each one has a branch in
#: :func:`_evalue_from_selection`, and :mod:`cleverly.assessment` reads two of these
#: strings to decide what a combined run costs.
_Branch = Literal[
    "reported_rr",
    "reported_or",
    "derived_rr",
    "fixed_baseline_ate",
    "gaussian_difference",
]


#: The one branch that has to retarget cached nuisances rather than read a stored
#: estimate.  :mod:`cleverly.assessment` prices a combined run by it and decides by it
#: whether an argument-free request replays under its own estimand name, so the string is
#: named here rather than repeated there.
_DERIVED_RR: _Branch = "derived_rr"


@dataclass(frozen=True)
class _EValueSelection:
    source: str
    branch: _Branch


def _default_estimand(result: TMLEResult, keys: dict[str, Any]) -> str:
    """Choose a reported contrast from structured parameter identities."""
    for target in ("rr", "or", "ate", "att", "atc"):
        candidates = [
            name
            for name in result.estimates
            if (key := keys.get(name)) is not None
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


def _baseline_mean(result: TMLEResult, estimand: str, keys: dict[str, Any]) -> str | None:
    """The reported mean under the arm ``estimand`` is contrasted against, if any.

    The risk-difference conversion divides by the baseline risk, and the baseline is the
    contrast's *reference* arm rather than arm ``0``: with ``reference=1`` a two-armed
    fit's ``ate`` is ``E[Y^0] - E[Y^1]`` and dividing by ``EY0`` would convert a
    difference into a ratio of the wrong pair.  Found by arm through
    :func:`~cleverly.sensitivity._parameters.arm_parameter_keys`, so ``ey0`` and
    ``ey[low]`` are both recognised -- whichever of them the fit was asked for.
    """
    parameter = keys.get(estimand)
    if parameter is None or parameter.reference is None:
        return None
    return next(
        (
            name
            for name, candidate in keys.items()
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
    if result.data.is_continuous_treatment:
        raise _EValueRefusal("not_applicable", "an E-value requires a discrete arm contrast")
    keys = arm_parameter_keys(result)
    explicit = estimand is not None
    source = _default_estimand(result, keys) if estimand is None else estimand
    if source not in result.estimates:
        raise _EValueRefusal("unavailable", f"estimand {source!r} was not requested in this fit")
    key = keys.get(source)
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
    if result.config.family == "gaussian" and key.estimand in {"ate", "att", "atc"}:
        return _EValueSelection(source, "gaussian_difference")
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
    refusal = _risk_ratio_refusal(result, source, keys)
    if refusal is None:
        return _EValueSelection(source, "derived_rr")
    if key.estimand == "or":
        return _EValueSelection(source, "reported_or")
    if result.intermediate_value is not None:
        raise _EValueRefusal("unavailable", refusal)
    baseline = _baseline_mean(result, source, keys)
    if baseline is None or not np.isfinite(result[baseline].psi) or result[baseline].psi <= 0:
        refusal += "; a finite positive reported reference-arm mean is also absent"
        raise _EValueRefusal("unavailable", refusal)
    _reject_unusable_baseline(result[baseline], result[source], refusal)
    return _EValueSelection(source, "fixed_baseline_ate")


def _reject_unusable_baseline(
    baseline: ParameterEstimate, estimate: ParameterEstimate, refusal: str
) -> None:
    """Refuse a fixed-baseline conversion whose ratio leaves the parameter space.

    The conversion holds the reference risk fixed and divides by it, so a positive sign
    is not enough. A reference risk its own standard error does not separate from zero
    is an unstable denominator: it sends the ratio to any value the caller cares to name.
    A risk difference at or below the negative of that risk implies a nonpositive risk in
    the contrast arm, which is outside the risk-ratio parameter space. Both are refused
    by name rather than reported as an ordinary approximate E-value.
    """
    error = baseline.std_error
    if not np.isfinite(error) or baseline.psi <= error:
        raise _EValueRefusal(
            "unavailable",
            f"{refusal}; the reported reference-arm mean {baseline.psi:.4g} is not "
            f"separated from zero by its own standard error {error:.4g}, so the "
            "fixed-baseline conversion has no stable denominator",
        )
    ratio = (baseline.psi + estimate.psi) / baseline.psi
    if not np.isfinite(ratio) or ratio <= 0:
        raise _EValueRefusal(
            "unavailable",
            f"{refusal}; the reported difference {estimate.psi:.4g} against the "
            f"reference-arm mean {baseline.psi:.4g} implies a nonpositive risk in the "
            "contrast arm, which no risk ratio describes",
        )


def evalue(result: TMLEResult, estimand: str | None = None) -> EValue:
    """E-value for a fitted result.

    ``estimand=None`` prefers a reported risk ratio. For an eligible binomial
    ordinary-TMLE result that reports only a marginal ATE or odds ratio, it retargets the
    retained nuisances to the matching exact risk ratio. An explicit odds-ratio request
    preserves the caller's chosen square-root approximation. Routing uses structured
    parameter keys or forward-composed fitted arm identities; display aliases are never parsed.
    Unsupported exact retargets can use a reported reference risk for an approximate ATE
    conversion. That conversion needs a reference risk its own standard error separates
    from zero, and it refuses a difference that implies a nonpositive risk in the contrast
    arm. Gaussian ATE, ATT, and ATC use the standardized-difference approximation, which
    divides by the weighted outcome standard deviation on a weighted fit.

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
    return _evalue_from_selection(result, _select_evalue(result, estimand))


def _standardising_sd(result: TMLEResult) -> tuple[float, bool]:
    """The outcome standard deviation the Gaussian conversion divides by.

    A weighted fit estimates a weighted parameter, so the standard deviation that
    standardises it belongs to the population the observation weights describe. The
    unweighted sample deviation would put the numerator and the denominator on two
    scales. The weighted form carries the reliability-weight correction
    ``sum(w) - sum(w^2) / sum(w)``, which is ``n - 1`` when every weight is one. So an
    unweighted fit reports the plain sample deviation.

    Parameters
    ----------
    result : TMLEResult
        A fitted result whose outcome is observed on at least two rows.

    Returns
    -------
    tuple of (float, bool)
        The standard deviation, and whether the observation weights moved it.

    Raises
    ------
    CapabilityError
        If the correction is not positive, which names a row count rather than a
        variance. One observed row gives ``w - w^2 / w = 0``, and no observed row of
        positive weight gives ``0`` as well.
    """
    observed = result.data.outcome[result.data.observed]
    weights = np.asarray(result.data.weights, dtype=float)[result.data.observed]
    total = float(np.sum(weights))
    correction = total - float(np.sum(weights**2)) / total if total > 0 else 0.0
    if correction <= 0:
        raise CapabilityError(
            "cannot standardise the effect: a standard deviation needs at least two "
            f"observed rows of positive weight, and the reliability correction is "
            f"{correction:.4g}"
        )
    mean = float(np.average(observed, weights=weights))
    variance = float(np.sum(weights * (observed - mean) ** 2)) / correction
    return float(np.sqrt(variance)), bool(result.data.is_weighted)


def _evalue_from_selection(result: TMLEResult, selection: _EValueSelection) -> EValue:
    """Execute the same branch used to declare this request available."""
    source = selection.source
    estimate = result[source]
    low, high = estimate.ci
    reported_estimand = source
    truncated_bound: float | None = None

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
        baseline_name = _baseline_mean(result, source, arm_parameter_keys(result))
        assert baseline_name is not None
        baseline = result[baseline_name]
        rr = float((baseline.psi + estimate.psi) / baseline.psi)
        raw = (
            float((baseline.psi + low) / baseline.psi),
            float((baseline.psi + high) / baseline.psi),
        )
        # Only the lower bound can leave the parameter space. ``normal_ci`` gives
        # ``high >= psi`` on a difference scale, and ``_reject_unusable_baseline`` has
        # already refused this fit unless ``baseline.psi > 0`` and
        # ``baseline.psi + estimate.psi > 0``, so ``raw[1] >= rr > 0``.
        truncated_bound = raw[0] if raw[0] < 0.0 else None
        ci = (0.0, raw[1]) if truncated_bound is not None else raw
        approximate, scale = True, "risk difference"
        note = (
            "Converted from a risk difference using the reported reference-arm risk. "
            "The conversion holds that risk fixed and ignores its sampling error."
        )
        if truncated_bound is not None:
            note += (
                " The difference interval reaches below the negative of that risk, so the "
                f"converted lower bound {truncated_bound:.4g} lies outside the risk-ratio "
                "parameter space. The report truncates it at 0, the smallest risk ratio "
                "there is, and keeps the untruncated value in 'truncated_bound'. "
            )
            note += (
                "The point ratio is above the null and the lower bound is below it, so "
                "the interval covers the null and the confidence-limit E-value is 1."
                if rr >= 1.0
                else "The point ratio is below the null, so the confidence-limit E-value "
                "reads the upper bound, which the truncation does not touch. That "
                "interval can still exclude the null."
            )
    elif selection.branch == "gaussian_difference":
        sd, weighted = _standardising_sd(result)
        if not np.isfinite(sd) or sd <= 0:
            raise CapabilityError("cannot standardise the effect: the outcome has no variance")
        rr = float(np.exp(_SMD_TO_LOG_RR * estimate.psi / sd))
        ci = (
            float(np.exp(_SMD_TO_LOG_RR * low / sd)),
            float(np.exp(_SMD_TO_LOG_RR * high / sd)),
        )
        approximate, scale = True, "mean difference"
        note = (
            f"Standardised by {'weighted ' if weighted else ''}sd(Y) = {sd:.4g}. "
            "Chinn's log(OR) / 1.81 step maps the "
            "standardised mean difference to an odds ratio. The subsequent common-outcome "
            "square-root conversion and E-value calculation are additional approximations."
        )
    else:
        raise AssertionError(selection.branch)

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
        source_estimand=source,
        truncated_bound=truncated_bound,
    )
