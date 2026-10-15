"""Private derived estimates needed by post-fit sensitivity analyses."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._assessment_cache import _cached
from .._refusal import NO_ALTERNATIVE, Refusal, RefusalReason
from ..estimators.direct_effect import declares_intermediate
from ..exceptions import CapabilityError
from ..inference.influence import ParameterEstimate, median_estimates
from ..targets.base import arm_alias
from ._parameters import arm_parameter_keys

if TYPE_CHECKING:  # pragma: no cover
    from ..estimators.base import TMLEResult


def _derived_risk_ratio(
    result: TMLEResult, source_estimand: str, keys: dict[str, Any] | None = None
) -> ParameterEstimate:
    """Retarget the source contrast's arms to a marginal risk ratio."""
    resolved = arm_parameter_keys(result) if keys is None else keys
    refusal = _risk_ratio_refusal(result, source_estimand, resolved)
    if refusal is not None:
        raise CapabilityError.from_refusal(refusal)
    key = resolved[source_estimand]

    alias = arm_alias(
        "rr", arm=key.value, versus=key.reference, collapse=result.data.is_binary_treatment
    )

    def compute() -> ParameterEstimate:
        reports = []
        for repeat in result.repeats:
            estimates, _ = result.estimator.retarget(
                result.data,
                repeat.nuisance,
                estimands=("rr",),
                g_bounds=result.config.g_bounds,
                g_bounds_conditional=result.config.g_bounds_conditional,
                nuisance_bound=result.config.missingness_bound,
                alpha_sig=result.config.alpha_sig,
            )
            reports.append(estimates)
        combined = median_estimates(reports)
        if alias not in combined:
            raise CapabilityError(
                f"retargeting did not produce the matching risk ratio {alias!r}; "
                f"it produced {list(combined)}. This is a package defect; report it.",
                reason=RefusalReason.INVALID_DECLARATION,
            )
        return combined[alias]

    return _cached(
        result,
        "sensitivity.derived_risk_ratio",
        (key,),
        {},
        compute,
    )


def _invalid(condition: str) -> Refusal:
    """Return an invalid-declaration refusal for a derived risk ratio."""
    return Refusal(RefusalReason.INVALID_DECLARATION, condition)


def _risk_ratio_refusal(
    result: TMLEResult, source_estimand: str, keys: dict[str, Any] | None = None
) -> Refusal | None:
    """Return why cached-nuisance risk-ratio retargeting cannot run."""
    if result.assessment_family != "point":
        return Refusal(
            RefusalReason.IMPLEMENTATION_GAP,
            "derived risk ratios are unavailable for longitudinal results.",
            alternative=NO_ALTERNATIVE,
            roadmap="F16",
        )
    resolved = arm_parameter_keys(result) if keys is None else keys
    if source_estimand not in resolved:
        return _invalid(
            "derived risk ratios require structured parameter keys retained by the fitted result."
        )
    key = resolved[source_estimand]
    if key.estimand not in {"ate", "or"}:
        return _invalid(
            f"derived risk ratios require an ATE or odds-ratio source, not {key.estimand!r}."
        )
    if key.axis != "arm":
        return Refusal(
            RefusalReason.IMPLEMENTATION_GAP,
            f"derived risk ratios require an arm contrast, not the {key.axis!r} axis.",
            alternative=(
                "Fit the risk ratio as an estimand with TMLE(estimands=('ate', 'rr')) "
                "on an arm-indexed fit, then call evalue(result, 'rr')."
            ),
            roadmap="F34",
        )
    if key.reference is None or key.stratum is not None:
        return Refusal(
            RefusalReason.IMPLEMENTATION_GAP,
            "derived risk ratios require an unconditioned marginal arm contrast.",
            alternative=(
                "Call evalue on the marginal alias of the same fit, "
                "or fit the stratum's rows as their own study."
            ),
            roadmap="X65",
        )
    if result.data.is_continuous_treatment:
        return _invalid("derived risk ratios are unavailable for a continuous treatment.")
    if result.config.family != "binomial":
        return _invalid("derived risk ratios require a binomial outcome.")
    method = result.fitted_method
    if method != "tmle":
        return Refusal(
            RefusalReason.IMPLEMENTATION_GAP,
            f"derived risk ratios are unavailable for fitted method {method!r}. "
            "Only ordinary TMLE has a same-seed equality witness.",
            alternative="Fit the risk ratio as an estimand, then call evalue(result, 'rr').",
            roadmap="X65",
        )
    if result.config.cv_evaluation:
        return Refusal(
            RefusalReason.IMPLEMENTATION_GAP,
            "derived risk ratios are unavailable for CV-evaluated fits.",
            alternative="Fit the risk ratio as an estimand, then call evalue(result, 'rr').",
            roadmap="X65",
        )
    if declares_intermediate(result):
        return Refusal(
            RefusalReason.IMPLEMENTATION_GAP,
            "derived risk ratios are unavailable for controlled direct effects because "
            "cached-nuisance retargeting does not pass the fitted intermediate intervention level.",
            alternative=NO_ALTERNATIVE,
            roadmap="X65",
        )
    from ..assessment import replayability

    if not replayability(result).retarget_cached_nuisances or result.estimator is None:
        return _invalid("derived risk ratios require the fitted estimator's retarget path.")
    if not result.repeats:
        return _invalid("derived risk ratios require cached repeat nuisances.")
    return None
