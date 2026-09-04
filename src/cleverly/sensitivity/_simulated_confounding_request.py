"""Request validation and population resolution for the simulated-confounding surface.

:mod:`cleverly.sensitivity.simulated_confounding` perturbs treatment and outcome, refits,
and reports. It runs none of that until the fitted result is checked against the complete
supported boundary. That check reads identification, parameter, estimator, weight, and
shift metadata, and it must finish before the surface draws its latent vector or refits a
cell. It is the larger half of the operation and it shares no state with the perturbation
law, so it lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ..estimators.base import MEAN_GROUP_ESTIMANDS
from ..exceptions import CapabilityError
from ..study import (
    ATC,
    ATE,
    ATT,
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


def _fit_wide_refusal(result: Any) -> str | None:
    """Return the first simulated-confounding refusal that applies to the whole fit.

    These boundaries do not depend on the requested parameter or strength grid. Keep
    their order here so execution and capability reporting name the same first missing
    scientific contract. Parameter-specific replay checks remain in
    :func:`_validate_request` after this helper.

    Parameters
    ----------
    result : Any
        Fitted result inspected by the surface or its assessment facade.

    Returns
    -------
    str or None
        Exact refusal reason, or ``None`` when no fit-wide boundary applies.
    """
    if getattr(result, "assessment_family", None) == "longitudinal":
        return (
            "simulated_confounding has no time-indexed latent law for longitudinal "
            "treatments, censoring, histories, outcomes, and contrasts"
        )
    data = getattr(result, "data", None)
    if data is None:
        return None
    if getattr(data, "has_missing_outcome", False):
        return (
            "simulated_confounding has no joint observation, treatment, and outcome "
            "perturbation law with identified missing-outcome refit semantics"
        )
    if (
        getattr(data, "has_intermediate", False)
        or getattr(result, "intermediate_value", None) is not None
    ):
        return (
            "simulated_confounding has no ordered treatment, intermediate, observation, "
            "and outcome law with a controlled-direct-effect contrast contract"
        )
    weight_spec = getattr(data, "weight_spec", None)
    if getattr(data, "weights_name", None) is not None and getattr(
        weight_spec, "estimated", False
    ):
        return (
            "simulated_confounding cannot replay estimated observation weights; the fitted "
            "result does not store the weight model, target-population semantics, and "
            "regeneration rule needed after perturbation"
        )
    if getattr(data, "cluster", None) is not None:
        return (
            "simulated_confounding has no source-backed choice among row-level, "
            "cluster-level, and mixed latent causes for clustered fits"
        )
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
    from ..estimators.base import TMLEResult
    from ..estimators.ctmle import CTMLE
    from ..estimators.drtmle import DRTMLE
    from ..estimators.tmle import TMLE
    from ..study import (
        BackdoorMeanContrast,
        ExplicitAdjustmentProvider,
        NaturalCourseMean,
        ParameterKey,
    )

    # Deferred because ``simulated_confounding`` imports this module at its own module
    # scope.  The weighted-statistics block stays beside the perturbation law that shares it.
    from .simulated_confounding import _is_constant_under_weights

    fit_wide_refusal = _fit_wide_refusal(result)
    if fit_wide_refusal is not None:
        raise CapabilityError(fit_wide_refusal)
    if type(result) is not TMLEResult:
        raise CapabilityError(
            "simulated_confounding supports point-treatment TMLEResult objects only; "
            "no longitudinal perturbation law is implemented"
        )
    estimator = result.estimator
    if estimator is None:
        raise CapabilityError(
            "simulated_confounding needs a replayable fitted estimator; this restored or "
            "legacy result has no estimator configuration"
        )
    stored_repeats = result.n_repeats
    configured_repeats = result.config.crossfit.repeats
    replay_repeats = estimator.repeats
    if stored_repeats != configured_repeats or stored_repeats != replay_repeats:
        raise CapabilityError(
            "simulated_confounding needs consistent repeated-cross-fitting provenance; "
            f"the stored result has {stored_repeats} draw(s), its configuration declares "
            f"{configured_repeats}, and the replay estimator declares {replay_repeats}"
        )
    data = result.data
    if data.is_continuous_treatment:
        treatment_family: Literal["binary", "continuous"] = "continuous"
    elif data.is_binary_treatment:
        treatment_family = "binary"
    else:
        raise CapabilityError(
            "simulated_confounding has no category-valued perturbation law for a multi-arm "
            "treatment"
        )
    if treatment_family == "binary" and type(estimator) not in {TMLE, CTMLE, DRTMLE}:
        raise CapabilityError(
            "simulated_confounding supports ordinary TMLE, collaborative TMLE, and "
            f"complete-outcome DR-TMLE; got {type(estimator).__name__}"
        )
    if treatment_family == "continuous" and type(estimator) is not TMLE:
        raise CapabilityError(
            "continuous simulated_confounding supports exact ordinary TMLE only; "
            f"got {type(estimator).__name__}"
        )
    if treatment_family == "binary" and any(value < 0.0 or value > 0.5 for value in grid.treatment):
        raise ValueError("binary treatment strengths must be between 0 and 0.5")
    if data.family not in {"gaussian", "binomial"}:
        raise CapabilityError(
            f"simulated_confounding has no perturbation law for outcome family={data.family!r}"
        )
    if data.family == "binomial" and any(value < 0.0 or value > 0.5 for value in grid.outcome):
        raise ValueError("binomial outcome strengths must be between 0 and 0.5")
    if data.weight_spec.kind != "probability":
        raise CapabilityError(
            "simulated_confounding supports fixed probability weights only; "
            f"got weight kind {data.weight_spec.kind!r}"
        )
    if data.weights_name != data.weight_spec.name:
        raise CapabilityError(
            "simulated_confounding found inconsistent observation-weight provenance; "
            "the data column and WeightSpec names disagree"
        )
    if data.weights_name is None and data.is_weighted:
        raise CapabilityError(
            "simulated_confounding found nonconstant observation weights without a declared "
            "weight column"
        )
    identified = result.identified_effect
    if identified is None:
        raise CapabilityError(
            "simulated_confounding needs identification metadata for a backdoor parameter; "
            "this legacy fit records none"
        )
    functional = identified.functional
    if type(functional) is not BackdoorMeanContrast:
        raise CapabilityError(
            "simulated_confounding supports a backdoor-identified parameter; "
            f"got {type(functional).__name__}"
        )
    provider = getattr(identified, "provider", None)
    if type(provider) is not ExplicitAdjustmentProvider:
        raise CapabilityError(
            "simulated_confounding needs registered explicit-adjustment backdoor provenance"
        )
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
