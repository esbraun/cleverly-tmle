"""Private derived estimates needed by post-fit sensitivity analyses."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._assessment_cache import _cached
from .._refusal import NO_ALTERNATIVE, Refusal, RefusalReason
from ..estimators.direct_effect import declares_intermediate
from ..exceptions import CapabilityError
from ..inference.influence import ParameterEstimate, median_estimates
from ..targets.base import arm_alias, stratum_alias
from ._parameters import arm_parameter_keys

if TYPE_CHECKING:  # pragma: no cover
    from ..estimators.base import TMLEResult


#: The fitted methods whose cached-nuisance retarget to ``rr`` equals a same-seed fit that
#: requested ``rr``, point and interval.  ``tests/unit/test_evalue_part_g.py`` holds the
#: witness for each one, and for each C-TMLE strategy family.
DERIVED_METHODS: frozenset[str] = frozenset({"tmle", "collaborative_tmle", "drtmle"})


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
    if key.stratum is not None:
        # A stratum-conditional contrast retargets to the ratio of the same stratum, which
        # the joint stratified targeting reports beside the marginal one.
        data = result.data
        code = list(data.strata_levels).index(tuple(key.stratum))
        alias = stratum_alias(alias, data.stratum_label(code))

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
                conditions=result.config.support_conditions,
                # The controlled direct effect fixes Z at the fitted level, and the
                # retarget rebuilds its clever covariate from that level.
                intermediate_value=result.intermediate_value,
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
    if key.reference is None:
        return _invalid("derived risk ratios require a two-arm contrast with a reference arm.")
    if not result.data.has_arms:
        return _invalid(
            "derived risk ratios are unavailable for a treatment with a continuous dose."
        )
    if result.config.family != "binomial":
        return _invalid("derived risk ratios require a binomial outcome.")
    method = result.fitted_method
    if method not in DERIVED_METHODS:
        return Refusal(
            RefusalReason.IMPLEMENTATION_GAP,
            f"derived risk ratios are unavailable for fitted method {method!r}. "
            f"Only {sorted(DERIVED_METHODS)} have a same-seed equality witness.",
            alternative="Fit the risk ratio as an estimand, then call evalue(result, 'rr').",
            roadmap="X82",
        )
    if result.config.cv_evaluation:
        return Refusal(
            RefusalReason.IMPLEMENTATION_GAP,
            "derived risk ratios are unavailable for CV-evaluated fits.",
            alternative="Fit the risk ratio as an estimand, then call evalue(result, 'rr').",
            roadmap="X82",
        )
    if declares_intermediate(result) and (
        not result.data.has_intermediate or result.intermediate_value is None
    ):
        return _invalid(
            "derived risk ratios of a controlled direct effect need both the intermediate "
            "column and the fitted level."
        )
    from ..assessment import replayability

    if not replayability(result).retarget_cached_nuisances or result.estimator is None:
        return _invalid("derived risk ratios require the fitted estimator's retarget path.")
    if not result.repeats:
        return _invalid("derived risk ratios require cached repeat nuisances.")
    return None
