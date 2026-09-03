"""Private derived estimates needed by post-fit sensitivity analyses."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._assessment_cache import _cached
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
        raise CapabilityError(refusal)
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
                f"it produced {list(combined)}"
            )
        return combined[alias]

    return _cached(
        result,
        "sensitivity.derived_risk_ratio",
        (key,),
        {},
        compute,
    )


def _risk_ratio_refusal(
    result: TMLEResult, source_estimand: str, keys: dict[str, Any] | None = None
) -> str | None:
    """Return why cached-nuisance risk-ratio retargeting cannot run."""
    if result.assessment_family != "point":
        return "derived risk ratios are unavailable for longitudinal results"
    resolved = arm_parameter_keys(result) if keys is None else keys
    if source_estimand not in resolved:
        return "derived risk ratios require structured parameter keys retained by the fitted result"
    key = resolved[source_estimand]
    if key.estimand not in {"ate", "or"}:
        return f"derived risk ratios require an ATE or odds-ratio source, not {key.estimand!r}"
    if key.axis != "arm":
        return f"derived risk ratios require an arm contrast, not the {key.axis!r} axis"
    if key.reference is None or key.stratum is not None:
        return "derived risk ratios require an unconditioned marginal arm contrast"
    if result.data.is_continuous_treatment:
        return "derived risk ratios are unavailable for a continuous treatment"
    if result.config.family != "binomial":
        return "derived risk ratios require a binomial outcome"
    method = result.fitted_method
    if method != "tmle":
        return (
            f"derived risk ratios are unavailable for fitted method {method!r}; "
            "only ordinary TMLE has a same-seed equality witness"
        )
    if result.config.cv_evaluation:
        return "derived risk ratios are unavailable for CV-evaluated fits"
    if result.intermediate_value is not None:
        return (
            "derived risk ratios are unavailable for controlled direct effects because no "
            "controlled direct risk-ratio target is registered"
        )
    from ..assessment import replayability

    if not replayability(result).retarget_cached_nuisances or result.estimator is None:
        return "derived risk ratios require the fitted estimator's retarget path"
    if not result.repeats:
        return "derived risk ratios require cached repeat nuisances"
    return None
