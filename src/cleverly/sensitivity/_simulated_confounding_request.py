"""Request validation and population resolution for the simulated-confounding surface.

:mod:`cleverly.sensitivity.simulated_confounding` perturbs treatment and outcome, refits,
and reports. It runs none of that until the fitted result is checked against the complete
supported boundary. That check reads identification, parameter, estimator, weight, and
shift metadata, and it must finish before the surface draws its latent vector or refits a
cell. It is the larger half of the operation and it shares no state with the perturbation
law, so it lives here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from numbers import Real
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ..estimators.base import MEAN_GROUP_ESTIMANDS, TMLEResult
from ..exceptions import CapabilityError
from ..study import (
    ATC,
    ATE,
    ATT,
    BackdoorMeanContrast,
    ExplicitAdjustmentProvider,
    OddsRatio,
    PopulationAttributableFraction,
    PopulationAttributableRisk,
    RiskRatio,
)
from ._simulated_confounding_common import (
    check_alias,
    check_only_declared_axis,
    check_registered_target,
    check_replay_declaration,
    fixed_axis,
    fixed_key_axis,
    point_study_or_refuse,
)
from ._simulated_confounding_fixed import fixed_replay_refusal, validate_fixed_replay
from ._simulated_confounding_incremental import (
    INCREMENTAL_TARGETS,
    incremental_replay_refusal,
    natural_incremental_means,
    validate_incremental_replay,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .simulated_confounding import ConfounderStrengthGrid


@dataclass(frozen=True)
class _ValidatedRequest:
    estimator: Any
    calibration_names: tuple[str, ...]
    treatment_family: Literal["binary", "continuous"]
    movement_scale: Literal["estimate_difference", "log_ratio"]
    # Excluded from equality for the reason ``AssessmentItem.arguments`` is: the generated
    # ``__eq__`` would return an array here rather than a bool, and ``__hash__`` would
    # raise.  The remaining fields already identify the request.
    # ``test_simulated_confounding_populations.py::
    # test_a_validated_request_compares_and_hashes_without_its_mask`` witnesses both.
    baseline_mask: np.ndarray[Any, Any] = field(compare=False)
    stratum: tuple[Any, ...] | None
    conditioning_code: float | None


#: The declared estimand type each supported contrast alias must carry.  One map, built
#: once, so the contrast branch of :func:`_validate_binary_parameter_state` is written
#: once rather than per scale.
_CONTRAST_TYPES: dict[str, type] = {
    "ate": ATE,
    "att": ATT,
    "atc": ATC,
    "rr": RiskRatio,
    "or": OddsRatio,
}

_ATTRIBUTABLE_TYPES: dict[str, type] = {
    "par": PopulationAttributableRisk,
    "paf": PopulationAttributableFraction,
}

#: The supported aliases a ``CounterfactualMean`` declares, which name one arm rather than
#: a contrast between two.
_MEAN_TARGETS: frozenset[str] = frozenset({"ey", "ey1", "ey0"})

#: The supported contrasts reported on the ratio scale, whose movement is read from the
#: stored log-scale estimate rather than the reported value.
_RATIO_TARGETS: frozenset[str] = frozenset({"rr", "or"})

#: Derived from :data:`_CONTRAST_TYPES`, :data:`_MEAN_TARGETS` and
#: :data:`_ATTRIBUTABLE_TYPES` rather than declared here, so a target added to one of those
#: three cannot fall out of the supported set.  :data:`_RATIO_TARGETS` is not a fourth
#: source.  It names the two contrasts that report on the ratio scale, and both of them
#: already appear in :data:`_CONTRAST_TYPES`.
_BINARY_PARAMETER_TARGETS: frozenset[str] = (
    frozenset(_CONTRAST_TYPES) | _MEAN_TARGETS | frozenset(_ATTRIBUTABLE_TYPES)
)

#: The supported binary aliases whose clever covariate conditions on the observed
#: treatment group instead of averaging over the baseline population.  Derived by
#: subtracting the registry's ``mean`` fluctuation from the supported set, which is the
#: same test :class:`~cleverly.CTMLE` and :class:`~cleverly.DRTMLE` refuse on, so the
#: surface and the estimators cannot disagree about which aliases leave that family.
#: It is ``{"att", "atc"}`` today, and agrees with
#: :data:`~cleverly.utils.bounds.CONDITIONAL_GROUPS` for the same reason.
_CONDITIONAL_TARGETS: frozenset[str] = _BINARY_PARAMETER_TARGETS - MEAN_GROUP_ESTIMANDS

#: The supported binary aliases that only exact ordinary TMLE can replay.
#: :func:`_replay_refusal` keeps a separate message per group, because the two boundaries
#: have different causes.  This names their union once, so the coverage tests that assert
#: which aliases a C-TMLE or DR-TMLE surface exercises need not respell it.
_TMLE_ONLY_TARGETS: frozenset[str] = _CONDITIONAL_TARGETS | frozenset(_ATTRIBUTABLE_TYPES)


#: The six missing-science stops, each naming the artifact the surface would need and the
#: roadmap entry that tracks it.  Named constants rather than inline literals, because the
#: capability row, the execution refusal, and the roadmap must quote one text.
_LONGITUDINAL_REFUSAL = (
    "simulated_confounding has no time-indexed latent law for longitudinal "
    "treatments, censoring, histories, outcomes, and contrasts; "
    "docs/roadmap.md F13 tracks this stop"
)
_MULTI_ARM_REFUSAL = (
    "simulated_confounding has no category-valued perturbation law for a multi-arm "
    "treatment; docs/roadmap.md F8 tracks this stop"
)
_MISSING_OUTCOME_REFUSAL = (
    "simulated_confounding has no joint observation, treatment, and outcome "
    "perturbation law with identified missing-outcome refit semantics; "
    "docs/roadmap.md F12 tracks this stop"
)
_INTERMEDIATE_REFUSAL = (
    "simulated_confounding has no ordered treatment, intermediate, observation, "
    "and outcome law with a controlled-direct-effect contrast contract; "
    "docs/roadmap.md F15 tracks this stop"
)
#: The one stop whose message names the fitted result rather than the surface, because the
#: result is what stores no weight model.  Accuracy over template uniformity: writing this
#: like its five neighbours would attribute the missing artifact to the wrong object.
_ESTIMATED_WEIGHT_REFUSAL = (
    "simulated_confounding cannot replay estimated observation weights; the fitted result "
    "stores no weight model, target-population semantics, or regeneration rule; "
    "docs/roadmap.md F11 tracks this stop"
)
_CLUSTERED_REFUSAL = (
    "simulated_confounding has no source-backed choice among row-level, "
    "cluster-level, and mixed latent causes for clustered fits; "
    "docs/roadmap.md F9 tracks this stop"
)


def _refuse_longitudinal(result: Any) -> str | None:
    """Refuse every declared assessment family except the point-treatment one.

    Runs first, and it is the only rule that runs before the result shape is established,
    so it reads its attribute defensively.  A longitudinal result is not a ``TMLEResult``
    either, and it must hear its own missing-law stop rather than a type report.

    ``_LONGITUDINAL_REFUSAL`` is a claim about *that* family, so only ``"longitudinal"``
    hears it.  Any other declared family refuses with a message that names what the object
    actually declared.  ``evalue._select_evalue``, ``_derived._risk_ratio_refusal`` and
    ``_parameters.arm_parameter_keys`` all refuse the complement of ``"point"`` the same
    way, and none of them asserts a family the object never named.

    An object that declares no family at all is not a fitted result, and the ``result_type``
    rule that runs next owns that message.  Nothing falls through: only an exact
    ``TMLEResult`` passes that rule, and ``TMLEResult.assessment_family`` is the class
    variable ``"point"``, so a result reaching rule three declared its family here.
    """
    family = getattr(result, "assessment_family", None)
    if family == "longitudinal":
        return _LONGITUDINAL_REFUSAL
    if family is not None and family != "point":
        return f"simulated_confounding has no perturbation law for assessment family {family!r}"
    return None


def _refuse_multi_arm(result: Any) -> str | None:
    """Refuse a treatment that is neither binary nor a continuous dose."""
    data = result.data
    if not data.is_continuous_treatment and not data.is_binary_treatment:
        return _MULTI_ARM_REFUSAL
    return None


def _refuse_missing_outcome(result: Any) -> str | None:
    """Refuse a fit whose outcome is unobserved on some rows."""
    if result.data.has_missing_outcome:
        return _MISSING_OUTCOME_REFUSAL
    return None


def _refuse_intermediate(result: Any) -> str | None:
    """Refuse a fit that declares an intermediate variable."""
    if result.data.has_intermediate or result.intermediate_value is not None:
        return _INTERMEDIATE_REFUSAL
    return None


def _refuse_estimated_weights(result: Any) -> str | None:
    """Refuse observation weights the fit estimated rather than received."""
    data = result.data
    if data.weights_name is not None and data.weight_spec.estimated:
        return _ESTIMATED_WEIGHT_REFUSAL
    return None


def _refuse_clustered(result: Any) -> str | None:
    """Refuse a clustered fit, whose latent cause has no source-backed level."""
    if result.data.cluster is not None:
        return _CLUSTERED_REFUSAL
    return None


def _refuse_result_type(result: Any) -> str | None:
    """Refuse any artifact that is not exactly a point-treatment ``TMLEResult``."""
    if type(result) is not TMLEResult:
        return (
            "simulated_confounding supports point-treatment TMLEResult objects only; "
            f"got {type(result).__name__}"
        )
    return None


def _refuse_missing_estimator(result: Any) -> str | None:
    """Refuse a restored or legacy result that stores no replay estimator."""
    if result.estimator is None:
        return (
            "simulated_confounding needs a replayable fitted estimator; this restored or "
            "legacy result has no estimator configuration"
        )
    return None


def _refuse_repeat_provenance(result: Any) -> str | None:
    """Refuse disagreeing repeated-cross-fitting draw counts across stored layers."""
    stored_repeats = result.n_repeats
    configured_repeats = result.config.crossfit.repeats
    replay_repeats = result.estimator.repeats
    if stored_repeats != configured_repeats or stored_repeats != replay_repeats:
        return (
            "simulated_confounding needs consistent repeated-cross-fitting provenance; "
            f"the stored result has {stored_repeats} draw(s), its configuration declares "
            f"{configured_repeats}, and the replay estimator declares {replay_repeats}"
        )
    return None


def _refuse_binary_estimator(result: Any) -> str | None:
    """Refuse a binary fit made by an estimator this surface cannot replay."""
    from ..estimators.ctmle import CTMLE
    from ..estimators.drtmle import DRTMLE
    from ..estimators.tmle import TMLE

    estimator = result.estimator
    if not result.data.is_continuous_treatment and type(estimator) not in {TMLE, CTMLE, DRTMLE}:
        return (
            "simulated_confounding supports ordinary TMLE, collaborative TMLE, and "
            f"complete-outcome DR-TMLE; got {type(estimator).__name__}"
        )
    return None


def _refuse_continuous_estimator(result: Any) -> str | None:
    """Refuse a continuous fit made by anything but exact ordinary TMLE."""
    from ..estimators.tmle import TMLE

    estimator = result.estimator
    if result.data.is_continuous_treatment and type(estimator) is not TMLE:
        return (
            "continuous simulated_confounding supports exact ordinary TMLE only; "
            f"got {type(estimator).__name__}"
        )
    return None


def _refuse_outcome_family(result: Any) -> str | None:
    """Refuse an outcome family with no perturbation law."""
    data = result.data
    if data.family not in {"gaussian", "binomial"}:
        return f"simulated_confounding has no perturbation law for outcome family={data.family!r}"
    return None


def _refuse_weight_kind(result: Any) -> str | None:
    """Refuse a weight kind other than fixed probability weights."""
    data = result.data
    if data.weight_spec.kind != "probability":
        return (
            "simulated_confounding supports fixed probability weights only; "
            f"got weight kind {data.weight_spec.kind!r}"
        )
    return None


def _refuse_weight_provenance(result: Any) -> str | None:
    """Refuse disagreeing weight-column and ``WeightSpec`` names."""
    data = result.data
    if data.weights_name != data.weight_spec.name:
        return (
            "simulated_confounding found inconsistent observation-weight provenance; "
            "the data column and WeightSpec names disagree"
        )
    return None


def _refuse_undeclared_weights(result: Any) -> str | None:
    """Refuse nonconstant weights that no declared column accounts for."""
    data = result.data
    if data.weights_name is None and data.is_weighted:
        return (
            "simulated_confounding found nonconstant observation weights without a declared "
            "weight column"
        )
    return None


def _refuse_identification(result: Any) -> str | None:
    """Refuse a legacy fit that records no identification metadata."""
    if result.identified_effect is None:
        return (
            "simulated_confounding needs identification metadata for a backdoor parameter; "
            "this legacy fit records none"
        )
    return None


def _refuse_functional(result: Any) -> str | None:
    """Refuse an identified functional other than a backdoor mean contrast."""
    functional = result.identified_effect.functional
    if type(functional) is not BackdoorMeanContrast:
        return (
            "simulated_confounding supports a backdoor-identified parameter; "
            f"got {type(functional).__name__}"
        )
    return None


def _refuse_provider(result: Any) -> str | None:
    """Refuse backdoor provenance from anything but an explicit adjustment set."""
    provider = getattr(result.identified_effect, "provider", None)
    if type(provider) is not ExplicitAdjustmentProvider:
        return "simulated_confounding needs registered explicit-adjustment backdoor provenance"
    return None


#: Every simulated-confounding refusal that depends on neither the requested estimand nor
#: the strength grid, in the one order the surface and its capability row both use.  The
#: table is ordered and each rule assumes its predecessors returned ``None``: rule
#: ``result_type`` establishes the artifact shape, and ``missing_estimator`` establishes
#: that a replay estimator exists.  The names are the introspection contract; a test reads
#: them to pin the order without respelling a message.
#:
#: ``result_type`` sits second because every rule after it reads the result's fields
#: directly, and :class:`~cleverly.CausalResult` is a public runtime-checkable protocol that
#: declares ``assessment_family`` and no ``data``.  Ordered any later, the free function
#: :func:`~cleverly.sensitivity.simulated_confounding` raised ``AttributeError`` on a
#: conforming object rather than refusing it.  It costs no user-visible reordering: it
#: returns ``None`` for every real ``TMLEResult``, and ``longitudinal`` still runs first so
#: a longitudinal result hears its own missing-law stop rather than a type report.
_FIT_WIDE_RULES: tuple[tuple[str, Callable[[Any], str | None]], ...] = (
    ("longitudinal", _refuse_longitudinal),
    ("result_type", _refuse_result_type),
    ("multi_arm", _refuse_multi_arm),
    ("missing_outcome", _refuse_missing_outcome),
    ("intermediate", _refuse_intermediate),
    ("estimated_weights", _refuse_estimated_weights),
    ("clustered", _refuse_clustered),
    ("missing_estimator", _refuse_missing_estimator),
    ("repeat_provenance", _refuse_repeat_provenance),
    ("binary_estimator", _refuse_binary_estimator),
    ("continuous_estimator", _refuse_continuous_estimator),
    ("outcome_family", _refuse_outcome_family),
    ("weight_kind", _refuse_weight_kind),
    ("weight_provenance", _refuse_weight_provenance),
    ("undeclared_weights", _refuse_undeclared_weights),
    ("identification", _refuse_identification),
    ("functional", _refuse_functional),
    ("provider", _refuse_provider),
)


def _fit_wide_refusal(result: Any) -> str | None:
    """Return the first simulated-confounding refusal that applies to the whole fit.

    Every boundary in :data:`_FIT_WIDE_RULES` is reachable from capability reporting and
    from execution, and neither the requested parameter nor the strength grid can change
    its verdict. One helper answers both callers, so a fit the surface refuses can never
    be advertised as available. Parameter-specific and grid-specific checks stay in
    :func:`_validate_request`.

    Parameters
    ----------
    result : Any
        Fitted result inspected by the surface or its assessment facade.

    Returns
    -------
    str or None
        Exact refusal reason, or ``None`` when no fit-wide boundary applies.
    """
    for _name, rule in _FIT_WIDE_RULES:
        reason = rule(result)
        if reason is not None:
            return reason
    return None


def _replay_refusal(estimator: Any, estimand: str, stratum: tuple[Any, ...] | None) -> str | None:
    """Say why this estimator cannot replay this request, or ``None`` when it can.

    The estimator-capability boundary of the surface, stated once. The eligibility filter
    advertises the aliases this function accepts, and :func:`_validate_request` refuses
    the rest with the message it returns. Written twice, the filter advertised a marginal
    alias that the guard then refused, and the refusal named strata the request never
    asked for.

    These compositions are refused upstream already, so the branches defend stored
    provenance. Two layers do that refusing, and which one fires first depends on how the
    caller reached the estimator. On the study API,
    :meth:`~cleverly.study.IdentifiedEffect.available_methods` offers
    ``collaborative_tmle`` and ``drtmle`` for one allowlist of targets, and ``att``,
    ``atc``, ``par`` and ``paf`` are all outside it, so the method catalog refuses each of
    them before any estimator is built. On a direct low-level call, ``CTMLE`` and ``DRTMLE``
    reject an estimand outside ``MEAN_GROUP_ESTIMANDS`` when they estimate, which stops
    ``att`` and ``atc`` a second time; ``par`` and ``paf`` are *inside* that set, so this
    filter does not stop them, and their result instead carries no identification metadata,
    which :func:`_validate_request` refuses earlier than this function. Stratified
    reduced-regression targeting is rejected when ``TMLE`` fits. This function is the
    defence in depth of the surface, and it keys on the **requested** stratum rather than
    on whether the data carry strata.

    Parameters
    ----------
    estimator : Any
        Replay estimator stored on the fitted result.
    estimand : str
        Registered target that the requested alias names.
    stratum : tuple or None
        Baseline stratum the requested alias conditions on. ``None`` is marginal.

    Returns
    -------
    str or None
        Refusal message, or ``None`` when the estimator can replay the request.
    """
    from ..estimators.drtmle import DRTMLE
    from ..estimators.tmle import TMLE

    fixed_refusal = fixed_replay_refusal(estimator, estimand)
    if fixed_refusal is not None:
        return fixed_refusal
    if estimand == "msm" and stratum is not None:
        if estimator.msm.doses:
            return (
                "simulated_confounding cannot replay baseline strata for continuous MSMs; "
                "stratified continuous-dose targeting is unsupported"
            )
        if estimator.msm.link != "identity":
            return (
                "simulated_confounding cannot replay baseline strata for nonlinear MSMs; "
                "stratified alternating targeting is unsupported"
            )
    incremental_refusal = incremental_replay_refusal(estimator, estimand, stratum)
    if incremental_refusal is not None:
        return incremental_refusal
    if estimand in _CONDITIONAL_TARGETS and type(estimator) is not TMLE:
        return "simulated_confounding supports ATT and ATC under exact ordinary TMLE only"
    if estimand in _ATTRIBUTABLE_TYPES and type(estimator) is not TMLE:
        return (
            "simulated_confounding supports PAR and PAF under exact ordinary TMLE only; "
            "the identified effect's method catalog evidences no collaborative score and "
            "no reduced-dimension correction for these observed-law contrasts"
        )
    if stratum is not None and type(estimator) is DRTMLE:
        return (
            "simulated_confounding cannot replay a requested baseline stratum under "
            "DR-TMLE; stratified reduced-regression targeting is unsupported"
        )
    return None


def _eligible_binary_parameter_names(result: Any) -> tuple[str, ...]:
    """Return supported binary aliases from structured result metadata."""
    from ..study import ParameterKey

    data = getattr(result, "data", None)
    if data is None or not getattr(data, "is_binary_treatment", False):
        return ()
    estimates = getattr(result, "estimates", {})
    keys = getattr(result, "parameter_keys", {})
    natural_means = natural_incremental_means(result)
    return tuple(
        alias
        for alias, key in keys.items()
        if alias in estimates
        and alias not in natural_means
        and type(key) is ParameterKey
        and (
            (key.estimand in _BINARY_PARAMETER_TARGETS and key.axis == "arm")
            or fixed_key_axis(key) is not None
            or (key.estimand in INCREMENTAL_TARGETS and key.axis == "ipsi")
        )
        and _replay_refusal(result.estimator, key.estimand, key.stratum) is None
    )


def _baseline_population(result: Any, key: Any, identified: Any) -> np.ndarray[Any, Any]:
    """Resolve a fixed baseline population from coherent structured metadata."""
    data = result.data
    if key.stratum is None:
        return np.ones(data.n, dtype=bool)
    error = "simulated_confounding found inconsistent baseline-stratum metadata"
    study = point_study_or_refuse(result, error)
    if (
        type(key.stratum) is not tuple
        or not data.has_strata
        or not data.strata_names
        or len(key.stratum) != len(data.strata_names)
        or key.stratum not in data.strata_levels
    ):
        raise CapabilityError(error)
    if (
        data.strata_levels != study.data.strata_levels
        or not np.array_equal(data.strata, study.data.strata)
        or np.asarray(data.strata).shape != (data.n,)
        or not np.array_equal(np.unique(data.strata), np.arange(data.n_strata))
    ):
        raise CapabilityError(error)
    code = data.strata_levels.index(key.stratum)
    mask = np.asarray(data.strata == code, dtype=bool)
    if not np.any(mask) or float(np.sum(data.weights[mask])) <= 0.0:
        raise CapabilityError(error)
    return mask


def _validate_binary_parameter_state(
    result: Any,
    estimand: str,
    key: Any,
    identified: Any,
    functional: Any,
    estimator: Any,
) -> None:
    """Require one coherent binary parameter across every stored layer."""
    from ..study import CounterfactualMean
    from ..targets.base import parameter_name

    target = key.estimand
    typed_estimand = identified.estimand
    levels = tuple(result.data.treatment_levels)
    expected_alias: str | None = None
    metadata_matches = False

    if target in _CONTRAST_TYPES and type(typed_estimand) is _CONTRAST_TYPES[target]:
        typed_contrast: Any = typed_estimand
        expected_alias = parameter_name(target)
        fitted_reference = result.data.arm_label(result.config.reference_arm)
        fitted_values = tuple(level for level in levels if level != fitted_reference)
        metadata_matches = (
            len(fitted_values) == 1
            and key.value == fitted_values[0]
            and key.reference == fitted_reference
            and typed_contrast.reference in (None, key.reference)
            and functional.reference == typed_contrast.reference
        )
    elif target in _MEAN_TARGETS and type(typed_estimand) is CounterfactualMean:
        expected_value = {
            "ey1": result.data.arm_label(1.0),
            "ey0": result.data.arm_label(0.0),
        }.get(target)
        expected_alias = parameter_name("ey", arm=key.value) if target == "ey" else target
        metadata_matches = (
            key.value in levels
            and key.reference is None
            and functional.reference is None
            and (
                typed_estimand.treatment is None
                if target == "ey"
                else typed_estimand.treatment == key.value == expected_value
            )
        )
    elif target in _ATTRIBUTABLE_TYPES and type(typed_estimand) is _ATTRIBUTABLE_TYPES[target]:
        typed_attributable: Any = typed_estimand
        expected_alias = parameter_name(target)
        fitted_reference = result.data.arm_label(result.config.reference_arm)
        declared_reference = (
            levels[0] if typed_attributable.reference is None else typed_attributable.reference
        )
        # Unlike a two-arm contrast, this key's value names the reference intervention.
        # The observed mean is recomputed by the estimator, not represented by an arm key.
        metadata_matches = (
            key.value == fitted_reference == declared_reference
            and key.reference is None
            and functional.reference == typed_attributable.reference
        )

    error = "simulated_confounding found inconsistent registered binary parameter metadata"
    if expected_alias is None or not metadata_matches:
        raise CapabilityError(error)
    registered = check_registered_target(result, key, "arm", error)
    check_replay_declaration(result, key, error)
    check_alias(
        result,
        estimand,
        key,
        expected_alias,
        error,
        estimate_error=(
            "simulated_confounding found inconsistent ratio-scale estimate metadata"
            if target in _RATIO_TARGETS
            else None
        ),
        finite_error=(
            "simulated_confounding needs a finite positive ratio and stored log-scale estimate"
            if target in _RATIO_TARGETS
            else None
        ),
    )
    if target in _ATTRIBUTABLE_TYPES:
        estimate = result.estimates[estimand]
        expected_scale = "fraction" if target == "paf" else "difference"
        expected_family = "binomial" if target == "paf" else None
        if target == "paf" and result.data.family != "binomial":
            raise CapabilityError(
                "simulated_confounding supports a population attributable fraction for a "
                "binary outcome only"
            )
        if (
            registered.scale != expected_scale
            or registered.requires_family != expected_family
            or registered.parameter_axis != "arm"
            or estimate.name != estimand
            or estimate.scale != expected_scale
            or not np.isfinite(estimate.psi)
        ):
            raise CapabilityError(
                "simulated_confounding found inconsistent identity-scale attributable "
                "parameter metadata"
            )
    if target in _RATIO_TARGETS:
        estimate = result.estimates[estimand]
        if result.data.family != "binomial":
            raise CapabilityError(
                "simulated_confounding supports a risk ratio or odds ratio for a binary "
                "outcome only"
            )
        if estimate.name != estimand or estimate.scale != "ratio":
            raise CapabilityError(
                "simulated_confounding found inconsistent ratio-scale estimate metadata"
            )
        if (
            registered.scale != "ratio"
            or registered.requires_family != "binomial"
            or registered.parameter_axis != "arm"
        ):
            raise CapabilityError(
                "simulated_confounding found inconsistent registered ratio target metadata"
            )
        try:
            inference_value = estimate.inference_value
        except ValueError as error:
            raise CapabilityError(
                "simulated_confounding needs the stored log-scale estimate for a ratio contrast"
            ) from error
        if not np.isfinite(inference_value) or not np.isfinite(estimate.psi) or estimate.psi <= 0.0:
            raise CapabilityError(
                "simulated_confounding needs a finite positive ratio and stored log-scale estimate"
            )
        # Exact equality holds because every site that builds a ratio ``ParameterEstimate``
        # derives ``psi = float(np.exp(log_psi))``: ``targets.builtin._ratio_contrasts``,
        # ``inference.influence.ratio_estimates`` and ``median_estimates``,
        # ``estimators.tmle._average_over_folds``, ``estimators.base`` and
        # ``longitudinal.estimator`` for the exponentiated coefficient view.  This check pins
        # that construction invariant.  An engine that ever computes the two independently
        # must relax it to a tolerance here, rather than leave the caller a refusal they
        # cannot act on.
        if float(np.exp(inference_value)) != float(estimate.psi):
            raise CapabilityError(
                "simulated_confounding found inconsistent reported and log-scale ratio estimates"
            )


def _validate_continuous_policy_state(
    result: Any,
    estimand: str,
    key: Any,
    identified: Any,
    functional: Any,
    estimator: Any,
) -> None:
    """Require one coherent modified-policy request across every stored layer."""
    from ..interventions.shift import Shift
    from ..study import ModifiedTreatmentPolicy, ModifiedTreatmentPolicyEffect
    from ..targets.base import parameter_name

    target = key.estimand
    typed_estimand = identified.estimand
    typed_type = {
        "ey_shift": ModifiedTreatmentPolicy,
        "ate_shift": ModifiedTreatmentPolicyEffect,
    }.get(target)
    if typed_type is None or type(typed_estimand) is not typed_type:
        raise CapabilityError(
            "continuous simulated_confounding found inconsistent registered modified-policy "
            "identification provenance"
        )
    typed_state: Any = typed_estimand
    typed_policies = tuple(typed_state.shifts)
    typed_reference = typed_state.reference

    declared_policies = tuple(functional.interventions)
    replay_policies = tuple(estimator.shifts)
    if any(type(shift) is not Shift for shift in (*declared_policies, *replay_policies)):
        raise CapabilityError(
            "continuous simulated_confounding found inconsistent structured shift metadata"
        )
    declared_names = tuple(shift.name for shift in declared_policies)
    declared_deltas = tuple(float(shift.delta) for shift in declared_policies)
    declared_reference = declared_names[0] if functional.reference is None else functional.reference
    fitted_shifts = result.nuisance.shifts
    fitted_names = () if fitted_shifts is None else tuple(fitted_shifts.names)
    fitted_deltas = () if fitted_shifts is None else tuple(fitted_shifts.deltas)
    fitted_reference = None if fitted_shifts is None else fitted_names[int(fitted_shifts.reference)]
    expected_alias = parameter_name(
        target,
        arm=key.value,
        versus=key.reference if target == "ate_shift" else None,
    )
    expected_shifted = np.column_stack(
        [shift.apply(result.data.treatment)[0] for shift in declared_policies]
    )
    expected_capped = np.column_stack(
        [shift.apply(result.data.treatment)[1] for shift in declared_policies]
    )
    expected_reference = fitted_reference if target == "ate_shift" else None
    if (
        fitted_shifts is None
        or typed_policies != declared_policies
        or typed_reference != functional.reference
        or replay_policies != declared_policies
        or fitted_names != declared_names
        or fitted_deltas != declared_deltas
        or fitted_reference != declared_reference
        or not np.array_equal(fitted_shifts.shifted, expected_shifted)
        or not np.array_equal(fitted_shifts.capped, expected_capped)
        or key.value not in fitted_names
        or key.reference != expected_reference
    ):
        raise CapabilityError(
            "continuous simulated_confounding found inconsistent structured shift metadata"
        )

    error = (
        "continuous simulated_confounding found inconsistent registered modified-policy "
        "identification provenance"
    )
    if getattr(typed_estimand, "name", None) != target:
        raise CapabilityError(error)
    check_registered_target(result, key, "shift", error)
    check_replay_declaration(
        result, key, "continuous simulated_confounding found inconsistent structured shift metadata"
    )
    check_alias(
        result,
        estimand,
        key,
        expected_alias,
        "continuous simulated_confounding found inconsistent structured shift metadata",
    )

    # A zero-delta shift maps every dose to itself, so its policy mean is E[Y] and its
    # counterfactual treatment has no dependence on the dose a common cause would move.
    # The treatment axis of such a surface is identically zero, and the outcome axis
    # reports the level shift ``Y' = Y - k_Y U`` alone.  An ``ate_shift`` contrast that
    # uses the same policy as its reference keeps treatment dependence, so it stays.
    if target == "ey_shift":
        selected = declared_policies[declared_names.index(key.value)]
        if float(selected.delta) == 0.0:
            raise CapabilityError(
                f"continuous simulated_confounding refuses the policy mean {estimand!r}; a "
                "zero-delta policy is the natural course, its mean is E[Y], and it carries no "
                "counterfactual treatment dependence for a simulated common cause to move. "
                "Select a nonzero-delta ey_shift[...] mean, or an ate_shift[...] contrast "
                "that uses the natural course as its reference"
            )


def _zero_delta_policy_means(result: Any) -> frozenset[str]:
    """Name every ``ey_shift`` alias whose policy is the zero-delta natural course.

    Parameters
    ----------
    result : TMLEResult
        Fitted result with identification and structured parameter metadata.

    Returns
    -------
    frozenset of str
        Aliases the selection message must not advertise, because each one names a
        policy mean the surface refuses.
    """
    from ..targets.base import parameter_name

    names = set()
    for policy in result.identified_effect.functional.interventions:
        delta = getattr(policy, "delta", None)
        name = getattr(policy, "name", None)
        if isinstance(name, str) and isinstance(delta, Real) and float(delta) == 0.0:
            names.add(name)
    return frozenset(
        {parameter_name("ey_shift", arm=name) for name in names}
        | {
            alias
            for alias, key in result.parameter_keys.items()
            if key.estimand == "ey_shift" and key.value in names
        }
    )


def _validate_request(
    result: Any,
    estimand: str,
    grid: ConfounderStrengthGrid,
    benchmark_covariates: Any,
) -> _ValidatedRequest:
    """Validate the complete supported boundary before a refit or random draw."""
    from ..study import NaturalCourseMean, ParameterKey

    # Deferred because ``simulated_confounding`` imports this module at its own module
    # scope.  The weighted-statistics block stays beside the perturbation law that shares it.
    from .simulated_confounding import _is_constant_under_weights

    # Every fit-wide boundary lives in one ordered table, which the capability row reads
    # through the same helper.  An unsupported fit is unsupported whatever grid the caller
    # passes, so these refusals precede the two grid-range checks below.
    fit_wide_refusal = _fit_wide_refusal(result)
    if fit_wide_refusal is not None:
        raise CapabilityError(fit_wide_refusal)
    estimator = result.estimator
    data = result.data
    # The ``multi_arm`` rule already refused every treatment that is neither continuous nor
    # binary, so exactly one of the two families holds here.
    treatment_family: Literal["binary", "continuous"] = (
        "continuous" if data.is_continuous_treatment else "binary"
    )
    if treatment_family == "binary" and any(value < 0.0 or value > 0.5 for value in grid.treatment):
        raise ValueError("binary treatment strengths must be between 0 and 0.5")
    if data.family == "binomial" and any(value < 0.0 or value > 0.5 for value in grid.outcome):
        raise ValueError("binomial outcome strengths must be between 0 and 0.5")
    identified = result.identified_effect
    functional = identified.functional
    key = result.parameter_keys.get(estimand)
    if (estimand == "ate" and type(identified.estimand) is NaturalCourseMean) or (
        type(key) is ParameterKey and key.estimand == "ey_obs"
    ):
        raise CapabilityError(
            "simulated_confounding refuses NaturalCourseMean; the natural-course mean is "
            "E[Y] and carries no counterfactual treatment dependence for a simulated "
            "common cause to move. Fit a supported PAR or PAF contrast to compare the "
            "observed mean with a reference intervention"
        )
    if treatment_family == "continuous" and estimand == "ate":
        vacuous = _zero_delta_policy_means(result)
        admissible = [
            name
            for name in result.estimates
            if name.startswith(("ey_shift[", "ate_shift[", "msm[")) and name not in vacuous
        ]
        detail = (
            f"choose one of {admissible}"
            if admissible
            else "this fit reports none that this surface can assess"
        )
        raise ValueError(
            "continuous simulated_confounding requires an explicit ey_shift[...] policy mean "
            f"or ate_shift[...] contrast alias, or an msm[...] coefficient alias; {detail}"
        )
    if estimand not in result.estimates:
        if treatment_family == "binary":
            admissible = list(_eligible_binary_parameter_names(result))
        else:
            # A continuous fit can report the zero-delta natural-course mean, which the
            # next call refuses.  Advertising it here hands the caller a refused alias.
            vacuous = _zero_delta_policy_means(result)
            admissible = [name for name in result.estimates if name not in vacuous]
        detail = (
            f"choose one of {admissible}"
            if admissible
            else "this fit reports none that this surface can assess"
        )
        raise ValueError(f"estimand {estimand!r} is unavailable; {detail}")
    if type(key) is not ParameterKey:
        raise CapabilityError(
            f"simulated_confounding needs a structured parameter key for {estimand!r}"
        )
    baseline_mask = _baseline_population(result, key, identified)
    refusal = _replay_refusal(estimator, key.estimand, key.stratum)
    if refusal is not None:
        raise CapabilityError(refusal)
    if fixed_axis(key.estimand) is not None:
        estimator = validate_fixed_replay(result, estimand, key)
    elif treatment_family == "binary" and key.estimand in INCREMENTAL_TARGETS:
        estimator = validate_incremental_replay(result, estimand, key)
    elif treatment_family == "binary":
        check_only_declared_axis(
            result,
            key,
            "arm",
            "simulated_confounding supports an arm-indexed parameter, "
            "not a regimen, stochastic, incremental, modified-policy, or MSM parameter",
        )
        if key.estimand not in _BINARY_PARAMETER_TARGETS or key.axis != "arm":
            raise CapabilityError(
                "simulated_confounding supports only an ATE, ATT, ATC, counterfactual arm "
                "mean, risk ratio, odds ratio, population attributable risk, or population "
                "attributable fraction; other parameters are outside its source boundary"
            )
        _validate_binary_parameter_state(result, estimand, key, identified, functional, estimator)
    else:
        check_only_declared_axis(
            result,
            key,
            "shift",
            "continuous simulated_confounding supports a modified-treatment-policy "
            "parameter, not an arm, regimen, stochastic, incremental, or MSM parameter",
        )
        if key.estimand not in {"ey_shift", "ate_shift"} or key.axis != "shift":
            raise CapabilityError(
                "continuous simulated_confounding supports only an ey_shift policy mean or "
                "ate_shift contrast; other parameters are outside its source boundary"
            )
        _validate_continuous_policy_state(result, estimand, key, identified, functional, estimator)

    names = tuple(
        [benchmark_covariates] if isinstance(benchmark_covariates, str) else benchmark_covariates
    )
    if len(set(names)) != len(names):
        raise ValueError("benchmark_covariates contains duplicates")
    categorical = {
        name for encoding in data.encodings for name in (encoding.column, *encoding.generated)
    }
    for name in names:
        if not isinstance(name, str):
            raise TypeError("benchmark_covariates must contain only column names")
        if name in categorical:
            raise CapabilityError(
                f"simulated_confounding cannot calibrate categorical covariate {name!r}; "
                "zeroing one encoded column does not define a logical-covariate benchmark"
            )
        if name not in data.covariate_names:
            raise ValueError(
                f"benchmark covariate {name!r} is unavailable; numeric adjustment columns are "
                f"{[name for name in data.covariate_names if name not in categorical]}"
            )
        column = data.covariates[:, data.covariate_names.index(name)]
        if _is_constant_under_weights(column, data.weights):
            raise CapabilityError(
                f"simulated_confounding cannot calibrate constant covariate {name!r}"
            )
    movement_scale: Literal["estimate_difference", "log_ratio"] = (
        "log_ratio" if key.estimand in _RATIO_TARGETS else "estimate_difference"
    )
    conditioning_arm = key.value if key.estimand == "att" else key.reference
    conditioning_code = (
        float(data.treatment_levels.index(conditioning_arm))
        if key.estimand in _CONDITIONAL_TARGETS
        else None
    )
    return _ValidatedRequest(
        estimator,
        names,
        treatment_family,
        movement_scale,
        baseline_mask,
        key.stratum,
        conditioning_code,
    )
