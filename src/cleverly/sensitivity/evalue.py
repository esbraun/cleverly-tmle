r"""E-values (VanderWeele & Ding, 2017).

The E-value answers the same question as the omitted-variable bound but on the risk
ratio scale, and with a different parameterisation that has become the convention in
epidemiology:

    *the minimum strength, on the risk ratio scale, that each of two confounder
    associations would need if both had equal strength: one with treatment and one
    with outcome, conditional on the measured covariates, to explain away the
    observed association.*

Unequal associations can trade off: one can be below the E-value if the other is
above it. The E-value alone does not establish whether such confounding is plausible.

For an observed risk ratio :math:`RR \ge 1`,

.. math:: E = RR + \sqrt{RR\,(RR - 1)}

and for :math:`RR < 1` the same formula is applied to :math:`1/RR`.  An E-value of 1
means no unmeasured confounding is needed -- the estimate is already
compatible with the null. A larger E-value raises the equal-strength threshold.

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

**Controlled direct effect.**  On a fit with an intermediate variable the target is the
risk ratio :math:`E[Y_{az}] / E[Y_{a'z}]` at the fitted level :math:`z`.  Its bias splits
into one factor per arm, an identity, and Ding and VanderWeele (2016), eAppendix
Proposition A.2, bounds each factor by its own bounding factor.  Arm :math:`a` reads the
cell ratio :math:`\sup_u P(u \mid A = a, Z = z, W) / P(u \mid W)` and arm :math:`a'` the
reciprocal direction, each with its own :math:`RR_{UY}`.  Smith, Mathur and VanderWeele
(2021), Result 1, write the same per-arm product.  With four equal strengths the product
is :math:`BF(E, E)^2`, so the E-value inverts :math:`\sqrt{RR}`: first the branch's own
conversion to the risk-ratio scale, then the square root.  :attr:`EValue.bias_factor`
records ``"per_arm_product"``.

**Readings without coverage evidence.**  A confidence-limit E-value reads an interval, so it
is calibrated only where that interval's coverage is.  The branches this module shipped
before X82 part G read intervals with registered coverage evidence.  Each configuration
part G adds resolves its limit through the registry of
:mod:`cleverly.validation._interpretation`, instrument :data:`EVALUE_LIMIT_INSTRUMENT`.
The ``evalue-limit-coverage`` study registers rows for ``stratum`` and ``time_to_event``
on its measured configurations.  Every other configuration prints with the registry's
descriptive sentence.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .._inference_status import status_record
from .._refusal import NO_ALTERNATIVE, Refusal, RefusalReason
from .._support import UNCLASSIFIED, Support, combine_support, record_refusal_site
from ..assessment import AssessmentStatus
from ..estimators.direct_effect import declares_intermediate
from ..exceptions import CapabilityError
from ..interventions.learned import LEARNED_RULE_EVALUE_REFUSAL
from ..targets.population_intervention import (
    NATURAL_COURSE_TARGET,
    POPULATION_INTERVENTION_TARGETS,
)
from ._derived import _derived_risk_ratio, _risk_ratio_refusal
from ._parameters import arm_parameter_keys

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..estimators.base import TMLEResult
    from ..inference.influence import ParameterEstimate
    from ..validation._interpretation import Resolution

__all__ = ["EValue", "evalue", "evalue_from_rr"]

#: Chinn's OR-to-SMD factor followed by the common-outcome square-root conversion.
_SMD_TO_LOG_RR = 1.81 / 2

#: Why every E-value branch stops on a fit that supplies no inference.  Each branch reads
#: the estimate's interval, or the reference arm's standard error, and such a fit supplies
#: neither.  Raised in :func:`_select_evalue` rather than where the interval is read, so
#: the capability row reports ``unavailable`` instead of the computation raising under a
#: row that advertised ``available=True``.  The status's own reason follows it as a new
#: sentence.
_EVALUE_NEEDS_INFERENCE = "an E-value is built from the reported estimate and its interval. "

#: Why a result that names an intermediate variable by one half only has no E-value.  A
#: controlled direct effect needs both the intermediate column and the fitted level ``z``.
#: A fit always records both, so only an edited result reaches this refusal.
_INCOMPLETE_INTERMEDIATE_REFUSAL = Refusal(
    RefusalReason.INVALID_DECLARATION,
    "this result names an intermediate variable without both its column and its fitted "
    "level, so the controlled direct effect it would report is not defined.",
)

#: The registry instrument of the confidence-limit readings that X82 part G adds.  Each
#: rule is one configuration, the ``+``-joined :data:`_EValueSelection.features`.
EVALUE_LIMIT_INSTRUMENT = "evalue_limit"

#: The conditions of the per-arm bounding factor, printed beside a controlled-direct-effect
#: E-value.  The last one is a condition of the controlled direct effect itself.
_PER_ARM_CONDITIONS = (
    "The per-arm bound assumes Y_az independent of A given (W, U) and of Z given (A, W, U), "
    "with U a baseline latent that A does not affect, positivity given (W, U), and no "
    "confounder of Z and Y that A affects. Each arm's treatment association is the cell "
    "ratio sup_u P(u | A = a, Z = z, W) / P(u | W), which mixes the association of U with A "
    "and with Z. The E-value is the equal strength of the four associations, two per arm."
)

#: The request a ``TimeToEvent`` fit answers, named in each of its refusals.
_TIME_TO_EVENT_FORM = (
    "On a TimeToEvent fit, pass the two reported risks at one horizon, "
    "evalue(result, ('risk_regimen[a1 @ t=2]', 'risk_regimen[a0 @ t=2]')), for the E-value "
    "of their risk ratio (kind='rr'; an odds ratio has no conversion here)."
)

#: The conditions of the ``TimeToEvent`` risk-ratio E-value, printed beside it.
_TIME_TO_EVENT_CONDITIONS = (
    "The risk at a horizon t is the mean of the binary outcome 1(T <= t), here for the event "
    "of the reported cause. The bound needs U to confound the treatment and the event only: "
    "censoring is at random given (A, W), and U does not affect censoring given (A, W)."
)

#: Printed beside the limit of a DR-TMLE fit whose treatment mechanism is estimated.
_LINEAR_REMAINDER_NOTE = (
    "DR-TMLE with an estimated treatment mechanism has a remainder that is linear in the "
    "mechanism error, so its interval is descriptive here."
)


def _evalue_inference_refusal(status: str) -> Refusal:
    """The E-value refusal at a non-inferential status, built on that status's reason."""
    record = status_record(status)
    record_refusal_site("evalue.inference_status", RefusalReason.INFERENCE_GAP, record.reopened_by)
    return Refusal(
        RefusalReason.INFERENCE_GAP,
        _EVALUE_NEEDS_INFERENCE + record.reason,
        alternative=record.alternative,
        roadmap=record.reopened_by,
    )


def _refuse(status: AssessmentStatus, refusal: Refusal) -> _EValueRefusal:
    """Build the E-value refusal that carries one complete record."""
    return _EValueRefusal(
        status,
        refusal.condition,
        reason=refusal.reason,
        alternative=refusal.alternative,
        roadmap=refusal.roadmap,
    )


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
    :func:`~cleverly.inference.delta.wald_ci` gives ``high >= psi``, and
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
        Equal-strength threshold for the confounder's treatment and outcome
        associations on the risk-ratio scale to explain the point estimate away.
        Unequal association strengths can trade off.
    limit : float
        The equal-strength threshold to move the confidence limit to the null.
    approximate : bool
        Whether reaching the risk-ratio scale needed an approximate conversion.
    note : str
        Explanation of that conversion, when one was used.
    source_estimand : str or None
        Reported source alias, which can differ from a derived output alias.
    truncated_bound : float or None
        The converted lower bound before truncation, when the report truncated one.
        ``None`` when :attr:`risk_ratio_ci` reports the conversion unchanged.
    support : Support
        The support of the source estimate, combined with the derived ratio's when the
        row derives one. An E-value built outside a classified fit is experimental.
    bias_factor : {"single", "per_arm_product"}
        ``"single"`` inverts one bounding factor. ``"per_arm_product"`` is the controlled
        direct effect, whose bias is a product of one bounding factor per arm, so the
        E-values invert the square root of the risk ratio and its limit.
    limit_reading : Resolution or None
        What the registry licenses for the confidence-limit E-value of a configuration that
        X82 part G added. ``None`` on the branches whose intervals carry registered
        coverage evidence.
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
    support: Support = UNCLASSIFIED
    bias_factor: Literal["single", "per_arm_product"] = "single"
    limit_reading: Resolution | None = None

    @property
    def limit_descriptive(self) -> bool:
        """Whether the confidence-limit E-value prints without coverage evidence.

        Returns
        -------
        bool
            ``True`` when :attr:`limit_reading` holds a resolution the registry does not
            license.
        """
        return self.limit_reading is not None and not self.limit_reading.licensed

    def summary(self) -> str:
        """Return a printable summary.

        Returns
        -------
        str
            A printable report, one line per reported quantity.
        """
        marker = " [experimental]" if self.support.label == "experimental" else ""
        lines = [
            f"E-value for {self.estimand!r} ({self.scale} scale){marker}",
            "-" * 40,
            f"risk ratio {self.risk_ratio:.4g} "
            f"[{self.risk_ratio_ci[0]:.4g}, {self.risk_ratio_ci[1]:.4g}]"
            + (" (approximate conversion)" if self.approximate else ""),
            f"E-value, point estimate     : {self.point:.4f}",
            f"E-value, confidence limit   : {self.limit:.4f}",
            "",
            (
                f"If its two risk-ratio associations had equal strength, an unmeasured "
                f"confounder would need {self.point:.2f} with both treatment and outcome, "
                f"beyond the measured covariates, to explain away the point estimate; "
                if self.bias_factor == "single"
                else f"If its four risk-ratio associations, two per arm, had equal strength, "
                f"an unmeasured confounder would need {self.point:.2f} with each of them, "
                f"beyond the measured covariates, to explain away the point estimate; "
            )
            + (
                f"{self.limit:.2f} to move the interval to the null."
                if self.limit > 1.0
                else "the interval already includes the null."
            ),
            "Unequal strengths can trade off: one association may be below the E-value "
            "if the other is above it.",
        ]
        if self.limit_reading is not None:
            lines.extend(["", f"confidence limit: {self.limit_reading.sentence}"])
        if self.note:
            lines.extend(["", self.note])
        if self.support.label == "experimental":
            lines.extend(["", f"experimental: {self.support.reason}"])
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
            "support": self.support.label,
            "bias_factor": self.bias_factor,
            "limit_rung": None if self.limit_reading is None else self.limit_reading.rung,
            "limit_licensed": None if self.limit_reading is None else self.limit_reading.licensed,
            "limit_sentence": None if self.limit_reading is None else self.limit_reading.sentence,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return self.summary()


class _EValueRefusal(CapabilityError):
    """One refusal, carrying the status :mod:`cleverly.assessment` publishes for it.

    The status is the enum member rather than its string, so every branch below is checked
    where it is written.  Thirteen call sites spelled the value as a bare literal and only
    the one place that built an :class:`~cleverly.assessment.AssessmentStatus` from it
    validated any of them, which made a misspelling a runtime error on the row it refused.

    The import is module level and introduces no cycle. :mod:`cleverly.assessment` reaches
    this module through function-local imports only, and nothing on its own module-level
    import chain reads :mod:`cleverly.sensitivity`.
    """

    def __init__(
        self,
        status: AssessmentStatus,
        text: str,
        /,
        *,
        reason: RefusalReason,
        alternative: str | None = None,
        roadmap: str | None = None,
    ) -> None:
        super().__init__(text, reason=reason, alternative=alternative, roadmap=roadmap)
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
    "time_to_event_rr",
]


#: The one branch that has to retarget cached nuisances rather than read a stored
#: estimate.  :mod:`cleverly.assessment` prices a combined run by it and decides by it
#: whether an argument-free request replays under its own estimand name, so the string is
#: named here rather than repeated there.
_DERIVED_RR: _Branch = "derived_rr"


@dataclass(frozen=True)
class _EValueSelection:
    """One request's branch, and the part G configurations it reads.

    ``features`` names each configuration that X82 part G added and that this request
    reads, in a fixed order.  An empty tuple is a shipped configuration.  The ``+``-joined
    features are the registry rule of the confidence-limit reading.
    """

    source: str
    branch: _Branch
    features: tuple[str, ...] = ()
    #: The denominator level of a ``time_to_event_rr`` request; ``None`` on every other branch.
    versus: str | None = None


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
                AssessmentStatus.DEFERRED,
                f"an E-value needs one contrast; choose an explicit estimand from {candidates}",
                reason=RefusalReason.INVALID_DECLARATION,
            )
        if candidates:
            return candidates[0]
    raise _EValueRefusal(
        AssessmentStatus.NOT_APPLICABLE,
        "an E-value needs a supported two-arm contrast; this fit reports no such parameter",
        reason=RefusalReason.INVALID_DECLARATION,
    )


def _baseline_mean(result: TMLEResult, estimand: str, keys: dict[str, Any]) -> str | None:
    """The reported mean under the arm ``estimand`` is contrasted against, if any.

    The risk-difference conversion divides by the baseline risk, and the baseline is the
    contrast's *reference* arm, in the contrast's own stratum, rather than arm ``0``: with
    ``reference=1`` a two-armed fit's ``ate`` is ``E[Y^0] - E[Y^1]`` and dividing by ``EY0``
    would convert a difference into a ratio of the wrong pair.  Found by arm through
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
            and candidate.stratum == parameter.stratum
        ),
        None,
    )


def _is_time_to_event(result: Any) -> bool:
    r"""Whether a longitudinal result holds one baseline decision and a binned event time.

    :meth:`~cleverly.longitudinal.LongitudinalData.from_time_to_event` alone records a
    time grid, and it holds the decision over every node and admits no covariate measured
    after baseline.  So the risk at a grid time is a point-treatment risk of the binary
    outcome :math:`1(T \le t)`.
    """
    data = getattr(result, "data", None)
    return (
        getattr(data, "time_grid", None) is not None
        and bool(getattr(data, "treatment_held", False))
        and bool(getattr(data, "is_survival", False))
        and getattr(result, "msm", None) is None
    )


def _time_to_event_refusal(condition: str) -> _EValueRefusal:
    """A refusal of a ``TimeToEvent`` E-value request that names the accepted form."""
    return _EValueRefusal(
        AssessmentStatus.UNAVAILABLE,
        condition,
        reason=RefusalReason.INVALID_DECLARATION,
        alternative=_TIME_TO_EVENT_FORM,
    )


def _select_time_to_event(result: Any, estimand: Any) -> _EValueSelection:
    """Admit the risk ratio of two static regimens at one horizon of a ``TimeToEvent`` fit."""
    if not (isinstance(estimand, tuple) and len(estimand) == 2):
        raise _time_to_event_refusal(
            "an E-value on a TimeToEvent fit reads the risk ratio of two reported risks at "
            f"one horizon, and the request names {estimand!r}."
        )
    numerator, denominator = (str(name) for name in estimand)
    index = getattr(result, "parameter_index", None) or {}
    missing = [name for name in (numerator, denominator) if name not in index]
    if missing:
        raise _time_to_event_refusal(
            f"{missing} are not risks this fit reports; it reports {sorted(index)}."
        )
    (first, cause, horizon), (second, cause_b, horizon_b) = index[numerator], index[denominator]
    if (cause, horizon) != (cause_b, horizon_b) or first == second:
        raise _time_to_event_refusal(
            f"{numerator!r} and {denominator!r} are not the risks of two regimens at one "
            "cause and one horizon."
        )
    from ..longitudinal.regimen import Regimen

    plans = {plan.label: plan for plan in result.config.regimens}
    dynamic = [name for name in (first, second) if type(plans.get(name)) is not Regimen]
    if dynamic:
        raise _EValueRefusal(
            AssessmentStatus.UNAVAILABLE,
            f"regimens {dynamic} are not static arms, and the E-value of a regime contrast "
            "has no derivation here.",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            alternative=NO_ALTERNATIVE,
            roadmap="F34",
        )
    for name in (numerator, denominator):
        if not result.estimates[name].supplies_inference:
            raise _refuse(
                AssessmentStatus.UNAVAILABLE,
                _evalue_inference_refusal(result.estimates[name].inference),
            )
    return _EValueSelection(
        numerator, "time_to_event_rr", features=("time_to_event",), versus=denominator
    )


def _select_evalue(result: TMLEResult, estimand: Any) -> _EValueSelection:
    """Select the exact, approximate, or refusal branch for one request."""
    if getattr(result, "assessment_family", None) != "point":
        time_to_event = _is_time_to_event(result)
        if time_to_event and estimand is not None:
            return _select_time_to_event(result, estimand)
        if time_to_event:
            raise _EValueRefusal(
                AssessmentStatus.DEFERRED,
                "an E-value on a TimeToEvent fit reads the risk ratio of two reported risks "
                "at one horizon; choose them as estimand=(numerator, denominator) from "
                f"{sorted(getattr(result, 'parameter_index', None) or {})}",
                reason=RefusalReason.INVALID_DECLARATION,
            )
        raise _EValueRefusal(
            AssessmentStatus.UNAVAILABLE,
            "no longitudinal sensitivity derivation is registered for an E-value: the "
            "bounding factor needs a per-period latent law for a sequential g-formula "
            "target (docs/roadmap.md F16 tracks this stop).",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            alternative=_TIME_TO_EVENT_FORM if time_to_event else NO_ALTERNATIVE,
            roadmap="F16",
        )
    if not result.data.has_arms:
        raise _EValueRefusal(
            AssessmentStatus.NOT_APPLICABLE,
            "an E-value requires a discrete arm contrast",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    if result.config.parameter_axis == "learned_rule":
        raise _EValueRefusal(
            AssessmentStatus.UNAVAILABLE,
            "an E-value does not cover a learned-rule fit: " + LEARNED_RULE_EVALUE_REFUSAL + ".",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            alternative=NO_ALTERNATIVE,
            roadmap="F27",
        )
    # Fit-wide and ahead of the estimand: an edited result that names one half of an
    # intermediate variable reports no controlled direct effect to bound.
    if declares_intermediate(result) and (
        not result.data.has_intermediate or result.intermediate_value is None
    ):
        raise _refuse(AssessmentStatus.UNAVAILABLE, _INCOMPLETE_INTERMEDIATE_REFUSAL)
    selection = _select_branch(result, estimand)
    return replace(selection, features=_features(result, selection))


def _features(result: TMLEResult, selection: _EValueSelection) -> tuple[str, ...]:
    """The part G configurations a selected request reads, in one fixed order."""
    features: list[str] = []
    if declares_intermediate(result):
        features.append("cde")
    key = arm_parameter_keys(result).get(selection.source)
    if key is not None and key.stratum is not None:
        features.append("stratum")
    if selection.branch == "gaussian_difference" and result.data.has_missing_outcome:
        features.append("response_sd")
    if selection.branch == _DERIVED_RR and result.fitted_method != "tmle":
        features.append(f"derived_{result.fitted_method}")
    return tuple(features)


def _select_branch(result: TMLEResult, estimand: str | None) -> _EValueSelection:
    """Select the branch of one request on a fit that :func:`_select_evalue` admitted."""
    keys = arm_parameter_keys(result)
    explicit = estimand is not None
    source = _default_estimand(result, keys) if estimand is None else estimand
    if source not in result.estimates:
        raise _EValueRefusal(
            AssessmentStatus.UNAVAILABLE,
            f"estimand {source!r} was not requested in this fit",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    key = keys.get(source)
    attributable = source.split("[", 1)[0] if key is None else key.estimand
    if attributable in POPULATION_INTERVENTION_TARGETS:
        raise _EValueRefusal(
            AssessmentStatus.NOT_APPLICABLE,
            "an E-value is defined for a two-arm risk ratio or contrast, and "
            f"{attributable} compares the natural course with one reference arm.",
            reason=RefusalReason.DIFFERENT_ESTIMAND,
            alternative=(
                "Call evalue(result, 'rr') or evalue(result, 'ate') on a fit that also "
                "requests the two-arm contrast, such as estimands=('par', 'ate', 'rr')."
            ),
        )
    if attributable == NATURAL_COURSE_TARGET:
        raise _EValueRefusal(
            AssessmentStatus.NOT_APPLICABLE,
            "an E-value is defined for a two-arm risk ratio or contrast, and ey_obs is the "
            "natural-course mean, not a two-arm contrast.",
            reason=RefusalReason.DIFFERENT_ESTIMAND,
            alternative=(
                "Call evalue(result, 'rr') or evalue(result, 'ate') on a fit that also "
                "requests the two-arm contrast."
            ),
        )
    if key is None:
        raise _EValueRefusal(
            AssessmentStatus.UNAVAILABLE,
            f"estimand {source!r} has no structured parameter key",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    if key.axis != "arm":
        raise _EValueRefusal(
            AssessmentStatus.NOT_APPLICABLE,
            f"an E-value needs an arm contrast, not the {key.axis!r} axis",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    if key.reference is None:
        raise _EValueRefusal(
            AssessmentStatus.NOT_APPLICABLE,
            f"an E-value needs a two-arm contrast; target {key.estimand!r} has no reference arm",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    # Fit-wide rather than branch-scoped, unlike the standardized rule below: the reported
    # and derived ratio branches read ``estimate.ci`` and the Gaussian branch reads the
    # reference arm's ``std_error``, so no branch survives a fit that supplies neither.
    # It runs after the request checks directly above, the requested name, its key and its
    # axis, which say that an E-value is not defined for this request on any fit.  So on a
    # fit without an intermediate variable, a level such as ``ey1`` stays ``not_applicable``
    # here too.
    if not result.estimates[source].supplies_inference:
        raise _refuse(
            AssessmentStatus.UNAVAILABLE,
            _evalue_inference_refusal(result.estimates[source].inference),
        )
    if key.estimand == "rr":
        return _EValueSelection(source, "reported_rr")
    if key.estimand == "or" and explicit:
        return _EValueSelection(source, "reported_or")
    if result.config.family == "gaussian" and key.estimand in {"ate", "att", "atc"}:
        # With a response mechanism the branch standardizes by a missing-at-random
        # plug-in of the population standard deviation (``_features`` names it).
        if result.data.has_missing_outcome and result.data.has_missing_treatment:
            raise _EValueRefusal(
                AssessmentStatus.UNAVAILABLE,
                "the standardized E-value divides by the population sd(Y), and its "
                "missing-at-random plug-in reads E[Y^k | A, W] on every row whose outcome "
                "is missing; a row whose treatment is not recorded has no A to read.",
                reason=RefusalReason.IMPLEMENTATION_GAP,
                alternative=(
                    "Fit a binomial outcome and call evalue(result, 'rr'), or fit the "
                    "rows whose treatment is recorded."
                ),
                roadmap="X82",
            )
        return _EValueSelection(source, "gaussian_difference")
    if key.estimand in {"att", "atc"}:
        raise _EValueRefusal(
            AssessmentStatus.UNAVAILABLE,
            f"{key.estimand.upper()} needs a conditional baseline risk and a supported "
            "conditional ratio target; neither is available.",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            alternative=(
                "Call evalue(result, 'ate') for the marginal ATE of the same arms, which "
                "describes a different population. On a gaussian outcome the standardized "
                "ATT or ATC E-value runs."
            ),
            roadmap="F34",
        )
    if key.estimand not in {"ate", "or"}:
        raise _EValueRefusal(
            AssessmentStatus.NOT_APPLICABLE,
            f"an E-value has no supported two-arm contrast for target {key.estimand!r}",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    refusal = _risk_ratio_refusal(result, source, keys)
    if refusal is None:
        return _EValueSelection(source, "derived_rr")
    if key.estimand == "or":
        return _EValueSelection(source, "reported_or")
    baseline = _baseline_mean(result, source, keys)
    if baseline is None or not np.isfinite(result[baseline].psi) or result[baseline].psi <= 0:
        raise _refuse(
            AssessmentStatus.UNAVAILABLE,
            replace(
                refusal,
                condition=refusal.condition.rstrip(".")
                + "; a finite positive reported reference-arm mean is also absent.",
            ),
        )
    _reject_unusable_baseline(result[baseline], result[source], refusal)
    return _EValueSelection(source, "fixed_baseline_ate")


def _reject_unusable_baseline(
    baseline: ParameterEstimate, estimate: ParameterEstimate, refusal: Refusal
) -> None:
    """Refuse a fixed-baseline conversion whose ratio leaves the parameter space.

    The conversion holds the reference risk fixed and divides by it, so a positive sign
    is not enough. A reference risk its own standard error does not separate from zero
    is an unstable denominator: it sends the ratio to any value the caller cares to name.
    A risk difference at or below the negative of that risk implies a nonpositive risk in
    the contrast arm, which is outside the risk-ratio parameter space. Both are refused
    by name rather than reported as an ordinary approximate E-value.
    """
    # The inner record keeps its code, alternative and roadmap id: the fallback adds a
    # reason why the approximate route fails too, and the exact route stays the remedy.
    error = baseline.std_error
    if not np.isfinite(error) or baseline.psi <= error:
        raise _refuse(
            AssessmentStatus.UNAVAILABLE,
            replace(
                refusal,
                condition=f"{refusal.condition.rstrip('.')}; the reported reference-arm mean "
                f"{baseline.psi:.4g} is not separated from zero by its own standard error "
                f"{error:.4g}, so the fixed-baseline conversion has no stable denominator.",
            ),
        )
    ratio = (baseline.psi + estimate.psi) / baseline.psi
    if not np.isfinite(ratio) or ratio <= 0:
        raise _refuse(
            AssessmentStatus.UNAVAILABLE,
            replace(
                refusal,
                condition=f"{refusal.condition.rstrip('.')}; the reported difference "
                f"{estimate.psi:.4g} against the reference-arm mean {baseline.psi:.4g} "
                "implies a nonpositive risk in the contrast arm, which no risk ratio "
                "describes.",
            ),
        )


def evalue(result: TMLEResult, estimand: str | tuple[str, str] | None = None) -> EValue:
    """E-value for a fitted result.

    ``estimand=None`` prefers a reported risk ratio. For an eligible binomial TMLE,
    C-TMLE, or DR-TMLE result that reports only a marginal ATE or odds ratio, it retargets
    the retained nuisances to the matching exact risk ratio. An explicit odds-ratio request
    preserves the caller's chosen square-root approximation. Routing uses structured
    parameter keys or forward-composed fitted arm identities; display aliases are never parsed.
    Unsupported exact retargets can use a reported reference risk for an approximate ATE
    conversion. That conversion needs a reference risk its own standard error separates
    from zero, and it refuses a difference that implies a nonpositive risk in the contrast
    arm. Gaussian ATE, ATT, and ATC use the standardized-difference approximation, which
    divides by the weighted outcome standard deviation on a weighted fit, and by a
    missing-at-random plug-in of it on a fit with a response mechanism.

    A contrast within a baseline stratum reads that stratum's rows. A controlled direct
    effect inverts a product of one bounding factor per arm. A ``TimeToEvent`` fit takes
    two reported risks at one horizon, and the E-value reads their risk ratio. These
    configurations print their confidence-limit E-value as descriptive until a coverage
    study registers them (``EValue.limit_reading``).

    Parameters
    ----------
    result : TMLEResult
        A fitted result.
    estimand : str, tuple of (str, str), or None
        Alias to report on. ``None`` picks the ratio-scale estimand the fit
        reported, which is the scale an E-value is defined on. On a ``TimeToEvent`` fit,
        the two risks whose ratio the E-value reads, numerator first.

    Returns
    -------
    EValue
        The association needed to explain the estimate and the interval away.

    Raises
    ------
    CapabilityError
        If no branch covers the request. A fit with an intermediate variable reports a
        controlled direct effect, and its E-value inverts a product of one bounding factor
        per arm; a result that names only one half of that variable refuses.
    """
    return _evalue_from_selection(result, _select_evalue(result, estimand))


def _stratum_rows(result: TMLEResult, source: str) -> np.ndarray | None:
    """The rows of the baseline stratum ``source`` conditions on, or ``None`` if marginal.

    The marginal E-value applied inside a stratum is a finite partition: every quantity the
    branch reads is computed on the stratum's rows, as the stratum's own study would.
    """
    key = arm_parameter_keys(result).get(source)
    if key is None or key.stratum is None:
        return None
    data = result.data
    code = list(data.strata_levels).index(tuple(key.stratum))
    return np.asarray(data.strata == code, dtype=bool)


def _standardising_sd(result: TMLEResult, rows: np.ndarray | None = None) -> tuple[float, bool]:
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
    rows : ndarray of bool or None
        The rows of a baseline stratum, for a stratum-conditional contrast. ``None`` reads
        every row.

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
    keep = np.asarray(result.data.observed, dtype=bool)
    if rows is not None:
        keep = keep & rows
    observed = result.data.outcome[keep]
    weights = np.asarray(result.data.weights, dtype=float)[keep]
    total = float(np.sum(weights))
    correction = total - float(np.sum(weights**2)) / total if total > 0 else 0.0
    if correction <= 0:
        raise CapabilityError(
            "cannot standardise the effect: a standard deviation needs at least two "
            f"observed rows of positive weight, and the reliability correction is "
            f"{correction:.4g}",
            reason=RefusalReason.INVALID_DECLARATION,
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
    support = estimate.support

    if selection.branch == "time_to_event_rr":
        assert selection.versus is not None
        ratio = result.ratio(source, selection.versus, kind="rr")
        estimate = ratio
        reported_estimand = ratio.name
        support = ratio.support
        rr, ci, approximate, scale = ratio.psi, ratio.ci, False, "risk ratio"
        note = _TIME_TO_EVENT_CONDITIONS
    elif selection.branch == "reported_rr":
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
        if not derived.supplies_inference:
            # The source supplied inference, so the selection admitted the request. A
            # retargeted ratio whose status supplies none has no interval to read.
            raise _refuse(
                AssessmentStatus.UNAVAILABLE, _evalue_inference_refusal(derived.inference)
            )
        rr, ci, approximate, scale = derived.psi, derived.ci, False, "risk ratio"
        reported_estimand = derived.name
        support = combine_support([estimate.support, derived.support], kind="derived")
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
        # Only the lower bound can leave the parameter space. ``wald_ci`` gives
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
        rows = _stratum_rows(result, source)
        response = result.data.has_missing_outcome
        if response:
            from ._outcome_moments import missing_at_random_sd

            sd, weighted = missing_at_random_sd(result, rows), bool(result.data.is_weighted)
        else:
            sd, weighted = _standardising_sd(result, rows)
        if not np.isfinite(sd) or sd <= 0:
            raise CapabilityError(
                "cannot standardise the effect: the outcome has no variance",
                reason=RefusalReason.INVALID_DECLARATION,
            )
        rr = float(np.exp(_SMD_TO_LOG_RR * estimate.psi / sd))
        ci = (
            float(np.exp(_SMD_TO_LOG_RR * low / sd)),
            float(np.exp(_SMD_TO_LOG_RR * high / sd)),
        )
        approximate, scale = True, "mean difference"
        within = "" if rows is None else " within the stratum"
        plugin = (
            "the missing-at-random plug-in of the population "
            if response
            else ("weighted " if weighted else "")
        )
        note = (
            f"Standardised by {plugin}sd(Y){within} = {sd:.4g}. "
            "Chinn's log(OR) / 1.81 step maps the "
            "standardised mean difference to an odds ratio. The subsequent common-outcome "
            "square-root conversion and E-value calculation are additional approximations."
        )
    else:
        raise AssertionError(selection.branch)
    if selection.branch in ("reported_or", "fixed_baseline_ate", "gaussian_difference"):
        # A converted effect is a derived estimate, so the row must list the derived kind.
        support = combine_support([support], kind="derived")

    above_null = rr >= 1.0
    limit = ci[0] if above_null else ci[1]
    per_arm = "cde" in selection.features
    if per_arm:
        # The bias of a controlled-direct-effect ratio is a product of one bounding factor
        # per arm. With equal strengths the product is BF(E, E)^2, so each E-value inverts
        # the square root, after the branch's own conversion to the risk-ratio scale. The
        # square root keeps a limit on its side of the null.
        note = (note + " " if note else "") + _PER_ARM_CONDITIONS
        point = evalue_from_rr(float(np.sqrt(rr)))
        limit_value = _evalue_for_limit(float(np.sqrt(limit)), above_null=above_null)
    else:
        point = evalue_from_rr(rr)
        limit_value = _evalue_for_limit(limit, above_null=above_null)
    return EValue(
        estimand=reported_estimand,
        scale=scale,
        risk_ratio=rr,
        risk_ratio_ci=ci,
        point=point,
        limit=limit_value,
        approximate=approximate,
        note=note,
        source_estimand=source,
        truncated_bound=truncated_bound,
        support=support,
        bias_factor="per_arm_product" if per_arm else "single",
        limit_reading=_limit_reading(result, selection),
    )


def _limit_reading(result: TMLEResult, selection: _EValueSelection) -> Resolution | None:
    """Ask the registry what a part G confidence-limit E-value may print.

    A shipped configuration reads an interval with registered coverage evidence, and
    returns ``None`` without asking.  Every other configuration resolves its ``+``-joined
    features as one rule, so a registered coverage cell switches on exactly the
    configuration it measured.  The import is local because
    :mod:`cleverly.validation` imports this package.
    """
    if not selection.features:
        return None
    from ..validation._interpretation import fit_configuration, resolve

    try:
        fit = fit_configuration(result, selection.source)
    except (AttributeError, KeyError, TypeError, ValueError):
        # An unreadable configuration licenses only through an unrestricted, exact row.
        fit = None
    reading = resolve(EVALUE_LIMIT_INSTRUMENT, "+".join(selection.features), fit)
    if (
        not reading.licensed
        and result.fitted_method == "drtmle"
        and result.data.known_treatment is None
    ):
        reading = replace(reading, sentence=f"{reading.sentence} {_LINEAR_REMAINDER_NOTE}")
    return reading
