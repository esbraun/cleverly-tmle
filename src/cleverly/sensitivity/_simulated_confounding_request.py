"""Request validation and population resolution for the simulated-confounding surface.

:mod:`cleverly.sensitivity.simulated_confounding` perturbs treatment and outcome, refits,
and reports. It runs none of that until the fitted result is checked against the complete
supported boundary. That check reads identification, parameter, estimator, weight, and
shift metadata, and it must finish before the surface draws its latent vector or refits a
cell. It is the larger half of the operation and it shares no state with the perturbation
law, so it lives here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from numbers import Real
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from .._refusal import NO_ALTERNATIVE, Refusal, RefusalReason
from ..estimators.base import MEAN_GROUP_ESTIMANDS, TMLEResult
from ..estimators.direct_effect import declares_intermediate
from ..exceptions import CapabilityError
from ..interventions.learned import LEARNED_RULE_REPLAY_REFUSAL
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
    treatment_family: Literal["binary", "categorical", "continuous", "vector"]
    movement_scale: Literal["estimate_difference", "log_ratio"]
    # Excluded from equality for the reason ``AssessmentItem.arguments`` is: the generated
    # ``__eq__`` would return an array here rather than a bool, and ``__hash__`` would
    # raise.  The remaining fields already identify the request.
    # ``test_simulated_confounding_populations.py::
    # test_a_validated_request_compares_and_hashes_without_its_mask`` witnesses both.
    baseline_mask: np.ndarray[Any, Any] = field(compare=False)
    stratum: tuple[Any, ...] | None
    conditioning_code: float | None
    #: The structured key of the requested parameter.
    key: Any = field(default=None, compare=False)
    #: The arm codes ``(a, b)`` that the pairwise swap of a multi-arm treatment exchanges,
    #: ``a`` first.  ``None`` on every other treatment.
    arm_codes: tuple[float, float] | None = None
    #: The level codes ``(a, b)`` each categorical component of more than two levels swaps,
    #: as ``(component, (a, b))`` pairs in component order.  Empty on every other treatment.
    component_pairs: tuple[tuple[str, tuple[float, float]], ...] = ()
    #: The declared level of the latent cause: ``"row"``, ``"cluster"`` or ``"mixed"``.
    latent_level: Literal["row", "cluster", "mixed"] = "row"
    #: The intra-cluster correlation of a ``"mixed"`` latent, else ``None``.
    latent_rho: float | None = None
    #: The covariates the declared observation mechanisms read, on a fit with a missing
    #: outcome or a missing treatment.  ``None`` when the caller declares none.
    observation_reads: tuple[str, ...] | None = None
    #: The outcome law on a Gaussian fit whose estimator declares ``q_bounds``.
    bounded_outcome_law: Literal["logit", "normal_score"] | None = None
    #: The declared ``q_bounds`` the bounded law reads.
    outcome_bounds: tuple[float, float] | None = None


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


#: The missing-science stops, each naming the artifact the surface would need and the
#: roadmap rows that track it.  Named constants rather than inline literals, because the
#: capability row, the execution refusal, and the roadmap must quote one text.  The
#: hard-stop review (docs/technical-reference/hard-stop-review.md) moved each declarable
#: law to roadmap X82 part F.  A stop that keeps a part in the research queue names both
#: rows and states the condition that keeps that part there.
_LONGITUDINAL_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "simulated_confounding has no time-indexed latent law for longitudinal "
    "treatments, censoring, histories, outcomes, and contrasts. docs/roadmap.md X82 part F "
    "tracks one time-fixed latent on uncensored terminal-outcome fits, and "
    "docs/roadmap.md F13 tracks censoring, survival outcomes, time-varying covariates "
    "after the first treatment node, and time-varying latents.",
    alternative=NO_ALTERNATIVE,
    roadmap="X82",
)
_MISSING_OUTCOME_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "simulated_confounding moves the treatment of a missing-outcome fit only under a declared "
    "response mechanism that reads the covariates alone: pass observation_reads=(...). "
    "Without it the mechanism may read the treatment, and a treatment perturbation breaks "
    "missing at random. docs/roadmap.md F12 tracks a response mechanism that reads the "
    "treatment.",
    alternative=(
        "Keep every treatment strength at zero for outcome-axis cells, declare "
        "observation_reads=, call evalue(result) for unmeasured confounding on the same "
        "fit, or missingness_tilt(result) for outcome missingness not at random."
    ),
    roadmap="F12",
)
_MISSING_TREATMENT_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "simulated_confounding moves a declared missing treatment only under a declared "
    "treatment-observation mechanism that reads the covariates alone: pass "
    "observation_reads=(...). It refuses an undeclared mechanism for the reason it refuses "
    "one on a missing outcome. docs/roadmap.md F12 tracks a mechanism that reads the "
    "treatment.",
    alternative=(
        "Keep every treatment strength at zero for outcome-axis cells, declare "
        "observation_reads=, or call evalue(result) on the same fit."
    ),
    roadmap="F12",
)
_INTERMEDIATE_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "simulated_confounding holds the treatment of a controlled-direct-effect fit: its cells "
    "flip the intermediate and move the outcome. A treatment perturbation would make the "
    "latent confound the intermediate and the outcome by a path the target does not name. "
    "docs/roadmap.md F15 tracks the treatment axis.",
    alternative=(
        "Keep every treatment strength at zero, and declare intermediate strengths with "
        "ConfounderStrengthGrid(intermediate=(...))."
    ),
    roadmap="F15",
)


def _refuse_longitudinal(result: Any) -> Refusal | None:
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
        return Refusal(
            RefusalReason.INVALID_DECLARATION,
            f"simulated_confounding has no perturbation law for assessment family {family!r}. "
            "This is a package defect; report it.",
        )
    return None


def _refuse_result_type(result: Any) -> Refusal | None:
    """Refuse any artifact that is not exactly a point-treatment ``TMLEResult``."""
    if type(result) is not TMLEResult:
        return Refusal(
            RefusalReason.INVALID_DECLARATION,
            "simulated_confounding supports point-treatment TMLEResult objects only; "
            f"got {type(result).__name__}",
        )
    return None


def _refuse_learned_rule(result: Any) -> Refusal | None:
    """Refuse a learned-rule fit, whose replay relearns the rules and so its target."""
    if result.config.parameter_axis != "learned_rule":
        return None
    return Refusal(
        RefusalReason.IMPLEMENTATION_GAP,
        "simulated_confounding does not cover a learned-rule fit: "
        + LEARNED_RULE_REPLAY_REFUSAL
        + ".",
        alternative=NO_ALTERNATIVE,
        roadmap="X82",
    )


def _refuse_risk_ratio_tilt(result: Any) -> Refusal | None:
    """Refuse a risk-ratio tilt fit, whose replay declares no reading of its policy."""
    if result.config.parameter_axis != "rr_tilt":
        return None
    return Refusal(
        RefusalReason.IMPLEMENTATION_GAP,
        "simulated_confounding does not cover a risk-ratio tilt fit (ey_rr_tilt, "
        "ate_rr_tilt): the tilt is a known randomized rule on the natural treatment, and the "
        "replay declares no reading of that rule on a perturbed draw.",
        alternative=(
            "Fit the static arms (ate) or an incremental odds tilt (ipsi) for this surface."
        ),
        roadmap="X82",
    )


def _refuse_missing_estimator(result: Any) -> Refusal | None:
    """Refuse a result that stores no replay estimator."""
    if result.estimator is None:
        return Refusal(
            RefusalReason.INVALID_DECLARATION,
            "simulated_confounding needs a replayable fitted estimator; this result stores "
            "no estimator configuration",
        )
    return None


def _refuse_repeat_provenance(result: Any) -> Refusal | None:
    """Refuse disagreeing repeated-cross-fitting draw counts across stored layers."""
    stored_repeats = result.n_repeats
    configured_repeats = result.config.crossfit.repeats
    replay_repeats = result.estimator.repeats
    if stored_repeats != configured_repeats or stored_repeats != replay_repeats:
        return Refusal(
            RefusalReason.INVALID_DECLARATION,
            "simulated_confounding needs consistent repeated-cross-fitting provenance; "
            f"the stored result has {stored_repeats} draw(s), its configuration declares "
            f"{configured_repeats}, and the replay estimator declares {replay_repeats}",
        )
    return None


def _refuse_binary_estimator(result: Any) -> Refusal | None:
    """Refuse a binary fit made by an estimator this surface cannot replay.

    An :class:`~cleverly.estimators.AdaptiveIPTW` fit replays as the others do: each cell
    refits the stored estimator on the perturbed data (X82 part I). A declared known
    mechanism travels to the cell as the exact mechanism of the flipped treatment.
    """
    from ..estimators.ctmle import CTMLE
    from ..estimators.drtmle import DRTMLE
    from ..estimators.iptw import AdaptiveIPTW
    from ..estimators.tmle import TMLE

    estimator = result.estimator
    if result.data.has_arms and type(estimator) not in {TMLE, CTMLE, DRTMLE, AdaptiveIPTW}:
        return Refusal(
            RefusalReason.IMPLEMENTATION_GAP,
            "simulated_confounding supports ordinary TMLE, collaborative TMLE, "
            "complete-outcome DR-TMLE, and adaptive-propensity IPTW; got "
            f"{type(estimator).__name__}.",
            alternative="Fit the data with TMLE, CTMLE or DRTMLE and replay that fit.",
            roadmap="X82",
        )
    return None


def _refuse_continuous_estimator(result: Any) -> Refusal | None:
    """Refuse a continuous fit made by anything but exact ordinary TMLE."""
    from ..estimators.tmle import TMLE

    estimator = result.estimator
    if not result.data.has_arms and type(estimator) is not TMLE:
        return Refusal(
            RefusalReason.IMPLEMENTATION_GAP,
            "continuous simulated_confounding supports exact ordinary TMLE only; "
            f"got {type(estimator).__name__}.",
            alternative="Fit the continuous treatment with TMLE and replay that fit.",
            roadmap="X66",
        )
    return None


def _refuse_outcome_family(result: Any) -> Refusal | None:
    """Refuse an outcome family with no perturbation law."""
    data = result.data
    if data.family not in {"gaussian", "binomial"}:
        return Refusal(
            RefusalReason.INVALID_DECLARATION,
            f"simulated_confounding has no perturbation law for outcome family={data.family!r}. "
            "This is a package defect; report it.",
        )
    return None


def _refuse_weight_kind(result: Any) -> Refusal | None:
    """Refuse a weight kind other than fixed probability weights."""
    data = result.data
    if data.weight_spec.kind != "probability":
        return Refusal(
            RefusalReason.INVALID_DECLARATION,
            "simulated_confounding supports fixed probability weights only; "
            f"got weight kind {data.weight_spec.kind!r}",
        )
    return None


def _refuse_weight_provenance(result: Any) -> Refusal | None:
    """Refuse disagreeing weight-column and ``WeightSpec`` names."""
    data = result.data
    if data.weights_name != data.weight_spec.name:
        return Refusal(
            RefusalReason.INVALID_DECLARATION,
            "simulated_confounding found inconsistent observation-weight provenance; "
            "the data column and WeightSpec names disagree",
        )
    return None


def _refuse_undeclared_weights(result: Any) -> Refusal | None:
    """Refuse nonconstant weights that no declared column accounts for."""
    data = result.data
    if data.weights_name is None and data.is_weighted:
        return Refusal(
            RefusalReason.INVALID_DECLARATION,
            "simulated_confounding found nonconstant observation weights without a declared "
            "weight column",
        )
    return None


def _refuse_identification(result: Any) -> Refusal | None:
    """Refuse a fit that records no identification metadata."""
    if result.identified_effect is None:
        return Refusal(
            RefusalReason.INVALID_DECLARATION,
            "simulated_confounding needs identification metadata for a backdoor parameter; "
            "this result records none",
        )
    return None


def _refuse_functional(result: Any) -> Refusal | None:
    """Refuse an identified functional other than a backdoor mean contrast."""
    functional = result.identified_effect.functional
    if type(functional) is not BackdoorMeanContrast:
        return Refusal(
            RefusalReason.INVALID_DECLARATION,
            "simulated_confounding supports a backdoor-identified parameter; "
            f"got {type(functional).__name__}",
        )
    return None


def _refuse_provider(result: Any) -> Refusal | None:
    """Refuse backdoor provenance from anything but an explicit adjustment set."""
    provider = getattr(result.identified_effect, "provider", None)
    if type(provider) is not ExplicitAdjustmentProvider:
        return Refusal(
            RefusalReason.INVALID_DECLARATION,
            "simulated_confounding needs registered explicit-adjustment backdoor provenance",
        )
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
_FIT_WIDE_RULES: tuple[tuple[str, Callable[[Any], Refusal | None]], ...] = (
    ("longitudinal", _refuse_longitudinal),
    ("result_type", _refuse_result_type),
    ("learned_rule", _refuse_learned_rule),
    ("risk_ratio_tilt", _refuse_risk_ratio_tilt),
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


def _fit_wide_refusal(result: Any) -> Refusal | None:
    """Return the first simulated-confounding refusal that applies to the whole fit.

    Every boundary in :data:`_FIT_WIDE_RULES` is reachable from capability reporting and
    from execution, and neither the requested parameter nor the strength grid can change
    its verdict. One helper answers both callers, so a fit the surface refuses can never
    be advertised as available. Parameter-specific and grid-specific checks stay in
    :func:`_validate_request`, and :func:`simulated_confounding_refusal` answers for the
    parameter-specific ones.

    Parameters
    ----------
    result : Any
        Fitted result inspected by the surface or its assessment facade.

    Returns
    -------
    Refusal or None
        The first fit-wide refusal, or ``None`` when no fit-wide boundary applies.
    """
    for _name, rule in _FIT_WIDE_RULES:
        refusal = rule(result)
        if refusal is not None:
            return refusal
    return None


def _replay_refusal(estimator: Any, estimand: str) -> Refusal | None:
    """Say why this estimator cannot replay this request, or ``None`` when it can.

    The estimator-capability boundary of the surface, stated once. The eligibility filter
    advertises the aliases this function accepts, and :func:`_validate_request` refuses
    the rest with the message it returns. Written twice, the filter advertised a marginal
    alias that the guard then refused, and the refusal named strata the request never
    asked for.

    Most of these compositions are refused upstream already, so those branches defend
    stored provenance. Two layers do that refusing, and which one fires first depends on how the
    caller reached the estimator. On the study API,
    :meth:`~cleverly.study.IdentifiedEffect.available_methods` offers
    ``collaborative_tmle`` and ``drtmle`` for one allowlist of targets, and ``att``,
    ``atc``, ``par`` and ``paf`` are all outside it, so the method catalog refuses each of
    them before any estimator is built. On a direct low-level call, ``CTMLE`` and ``DRTMLE``
    reject an estimand outside ``MEAN_GROUP_ESTIMANDS`` when they estimate, which stops
    ``att`` and ``atc`` a second time; ``par`` and ``paf`` are *inside* that set, so this
    filter does not stop them, and their result instead carries no identification metadata,
    which :func:`_validate_request` refuses earlier than this function.

    A requested baseline stratum is no boundary: every replay is a complete refit, and
    every estimator here targets the stratum blocks as the fit did.

    Parameters
    ----------
    estimator : Any
        Replay estimator stored on the fitted result.
    estimand : str
        Registered target that the requested alias names.

    Returns
    -------
    Refusal or None
        The refusal, or ``None`` when the estimator can replay the request.
    """
    from ..estimators.tmle import TMLE

    fixed_refusal = fixed_replay_refusal(estimator, estimand)
    if fixed_refusal is not None:
        return fixed_refusal
    incremental_refusal = incremental_replay_refusal(estimator, estimand)
    if incremental_refusal is not None:
        return incremental_refusal
    if estimand in _CONDITIONAL_TARGETS and type(estimator) is not TMLE:
        return Refusal(
            RefusalReason.IMPLEMENTATION_GAP,
            "simulated_confounding supports ATT and ATC under exact ordinary TMLE only.",
            alternative="Fit the data with TMLE and replay that fit.",
            roadmap="F5",
        )
    if estimand in _ATTRIBUTABLE_TYPES and type(estimator) is not TMLE:
        return Refusal(
            RefusalReason.IMPLEMENTATION_GAP,
            "simulated_confounding supports PAR and PAF under exact ordinary TMLE only; "
            "the identified effect's method catalog evidences no collaborative score and "
            "no reduced-dimension correction for these observed-law contrasts.",
            alternative="Fit the data with TMLE and replay that fit.",
            roadmap="X66",
        )
    return None


def _eligible_binary_parameter_names(result: Any) -> tuple[str, ...]:
    """Return supported binary aliases from structured result metadata."""
    from ..study import ParameterKey

    data = getattr(result, "data", None)
    if (
        data is None
        or not getattr(data, "has_arms", False)
        or data.is_continuous_treatment
        or data.treatment_kind == "vector"
    ):
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
        and _replay_refusal(result.estimator, key.estimand) is None
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
        raise CapabilityError(error, reason=RefusalReason.INVALID_DECLARATION)
    if (
        data.strata_levels != study.data.strata_levels
        or not np.array_equal(data.strata, study.data.strata)
        or np.asarray(data.strata).shape != (data.n,)
        or not np.array_equal(np.unique(data.strata), np.arange(data.n_strata))
    ):
        raise CapabilityError(error, reason=RefusalReason.INVALID_DECLARATION)
    code = data.strata_levels.index(key.stratum)
    mask = np.asarray(data.strata == code, dtype=bool)
    if not np.any(mask) or float(np.sum(data.weights[mask])) <= 0.0:
        raise CapabilityError(error, reason=RefusalReason.INVALID_DECLARATION)
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
    from ..study import ControlledDirectEffect, CounterfactualMean
    from ..targets.base import arm_alias, parameter_name

    target = key.estimand
    typed_estimand = identified.estimand
    if type(typed_estimand) is ControlledDirectEffect:
        # A controlled direct effect wraps one arm contrast at a declared intermediate
        # level.  The level must agree across the declaration, the functional, and the fit.
        if not (
            typed_estimand.intermediate == functional.intermediate == result.intermediate_value
        ):
            raise CapabilityError(
                "simulated_confounding found inconsistent controlled-direct-effect metadata: "
                f"the estimand fixes the intermediate at {typed_estimand.intermediate!r}, the "
                f"functional at {functional.intermediate!r}, and the fit at "
                f"{result.intermediate_value!r}",
                reason=RefusalReason.INVALID_DECLARATION,
            )
        typed_estimand = typed_estimand.contrast
    levels = tuple(result.data.treatment_levels)
    expected_alias: str | None = None
    metadata_matches = False

    if target in _CONTRAST_TYPES and type(typed_estimand) is _CONTRAST_TYPES[target]:
        typed_contrast: Any = typed_estimand
        expected_alias = arm_alias(
            target,
            arm=key.value,
            versus=key.reference,
            collapse=result.data.is_binary_treatment,
        )
        fitted_reference = result.data.arm_label(result.config.reference_arm)
        fitted_values = tuple(level for level in levels if level != fitted_reference)
        # A multi-arm fit reports one contrast per non-reference arm, each against the
        # fitted reference; a two-arm fit reports one.
        metadata_matches = (
            key.value in fitted_values
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
                # A multi-arm arm mean names its arm on the typed estimand, and an
                # all-arms mean names none.
                typed_estimand.treatment in (None, key.value)
                if target == "ey"
                else typed_estimand.treatment == key.value == expected_value
            )
        )
    elif target in _ATTRIBUTABLE_TYPES and type(typed_estimand) is _ATTRIBUTABLE_TYPES[target]:
        typed_attributable: Any = typed_estimand
        expected_alias = arm_alias(target, arm=key.value, collapse=result.data.is_binary_treatment)
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
        raise CapabilityError(error, reason=RefusalReason.INVALID_DECLARATION)
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
                "binary outcome only",
                reason=RefusalReason.INVALID_DECLARATION,
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
                "parameter metadata",
                reason=RefusalReason.INVALID_DECLARATION,
            )
    if target in _RATIO_TARGETS:
        estimate = result.estimates[estimand]
        if result.data.family != "binomial":
            raise CapabilityError(
                "simulated_confounding supports a risk ratio or odds ratio for a binary "
                "outcome only",
                reason=RefusalReason.INVALID_DECLARATION,
            )
        if estimate.name != estimand or estimate.scale != "ratio":
            raise CapabilityError(
                "simulated_confounding found inconsistent ratio-scale estimate metadata",
                reason=RefusalReason.INVALID_DECLARATION,
            )
        if (
            registered.scale != "ratio"
            or registered.requires_family != "binomial"
            or registered.parameter_axis != "arm"
        ):
            raise CapabilityError(
                "simulated_confounding found inconsistent registered ratio target metadata",
                reason=RefusalReason.INVALID_DECLARATION,
            )
        try:
            inference_value = estimate.inference_value
        except ValueError as error:
            raise CapabilityError(
                "simulated_confounding needs the stored log-scale estimate for a ratio contrast",
                reason=RefusalReason.INVALID_DECLARATION,
            ) from error
        if not np.isfinite(inference_value) or not np.isfinite(estimate.psi) or estimate.psi <= 0.0:
            raise CapabilityError(
                "simulated_confounding needs a finite positive ratio and stored log-scale estimate",
                reason=RefusalReason.INVALID_DECLARATION,
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
                "simulated_confounding found inconsistent reported and log-scale ratio estimates",
                reason=RefusalReason.INVALID_DECLARATION,
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
    from ..interventions.policy import (
        POLICY_TYPES,
        RiskRatioTilt,
        _vector_setup,
        describe_policy,
        is_identity_policy,
        lazy_frame,
        policy_branches,
    )
    from ..study import ModifiedTreatmentPolicy, ModifiedTreatmentPolicyEffect
    from ..targets.base import parameter_name

    target = key.estimand
    typed_estimand = identified.estimand
    typed_type = {
        "ey_policy": ModifiedTreatmentPolicy,
        "ate_policy": ModifiedTreatmentPolicyEffect,
    }.get(target)
    if typed_type is None or type(typed_estimand) is not typed_type:
        raise CapabilityError(
            "continuous simulated_confounding found inconsistent registered modified-policy "
            "identification provenance",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    typed_state: Any = typed_estimand
    typed_policies = tuple(typed_state.policies)
    typed_reference = typed_state.reference

    declared_policies = tuple(functional.interventions)
    replay_policies = tuple(estimator.policies)
    if any(
        not isinstance(policy, POLICY_TYPES) or isinstance(policy, RiskRatioTilt)
        for policy in (*declared_policies, *replay_policies)
    ):
        raise CapabilityError(
            "continuous simulated_confounding found inconsistent structured policy metadata",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    declared_names = tuple(policy.name for policy in declared_policies)
    declared_descriptions = tuple(describe_policy(policy) for policy in declared_policies)
    declared_reference = declared_names[0] if functional.reference is None else functional.reference
    fitted_shifts = result.nuisance.policies
    fitted_names = () if fitted_shifts is None else tuple(fitted_shifts.names)
    fitted_descriptions = () if fitted_shifts is None else tuple(fitted_shifts.descriptions)
    fitted_reference = None if fitted_shifts is None else fitted_names[int(fitted_shifts.reference)]
    expected_alias = parameter_name(
        target,
        arm=key.value,
        versus=key.reference if target == "ate_policy" else None,
    )
    # The replay re-applies each declared policy to the replayed dose, so the stored
    # assignment must be the one the declared policies give on the fitted data.
    data = result.data
    if data.treatment_kind == "vector":
        # Each joint policy, branch by branch, as the fit applied it to the components.
        setup = _vector_setup(declared_policies, data)
        expected_shifted = np.stack(setup.assigned, axis=1)
        expected_moved = np.any(expected_shifted != setup.natural[:, None, :], axis=2)
    else:
        observed_dose = np.asarray(data.treatment, dtype=float)
        frame = lazy_frame(
            lambda: data.frame_like(
                {name: data.covariates[:, j] for j, name in enumerate(data.covariate_names)}
            )
        )
        expected_shifted = np.column_stack(
            [
                branch.assign(observed_dose, frame)
                for policy in declared_policies
                for _, branch in policy_branches(policy)
            ]
        )
        expected_moved = expected_shifted != observed_dose[:, None]
    expected_reference = fitted_reference if target == "ate_policy" else None
    if (
        fitted_shifts is None
        or typed_policies != declared_policies
        or typed_reference != functional.reference
        or replay_policies != declared_policies
        or fitted_names != declared_names
        or fitted_descriptions != declared_descriptions
        or fitted_reference != declared_reference
        or not np.array_equal(fitted_shifts.shifted, expected_shifted)
        or not np.array_equal(fitted_shifts.moved, expected_moved)
        or key.value not in fitted_names
        or key.reference != expected_reference
    ):
        raise CapabilityError(
            "continuous simulated_confounding found inconsistent structured policy metadata",
            reason=RefusalReason.INVALID_DECLARATION,
        )

    error = (
        "continuous simulated_confounding found inconsistent registered modified-policy "
        "identification provenance"
    )
    if getattr(typed_estimand, "name", None) != target:
        raise CapabilityError(error, reason=RefusalReason.INVALID_DECLARATION)
    check_registered_target(result, key, "policy", error)
    check_replay_declaration(
        result,
        key,
        "continuous simulated_confounding found inconsistent structured policy metadata",
    )
    check_alias(
        result,
        estimand,
        key,
        expected_alias,
        "continuous simulated_confounding found inconsistent structured policy metadata",
    )

    # A zero-delta shift maps every dose to itself, so its policy mean is E[Y] and its
    # counterfactual treatment has no dependence on the dose a common cause would move.
    # The treatment axis of such a surface is identically zero, and the outcome axis
    # reports the level shift ``Y' = Y - k_Y U`` alone.  An ``ate_policy`` contrast that
    # uses the same policy as its reference keeps treatment dependence, so it stays.
    if target == "ey_policy":
        selected = declared_policies[declared_names.index(key.value)]
        if is_identity_policy(selected):
            raise CapabilityError(
                f"continuous simulated_confounding refuses the policy mean {estimand!r}; a "
                "zero-delta policy is the natural course, its mean is E[Y], and it carries no "
                "counterfactual treatment dependence for a simulated common cause to move.",
                reason=RefusalReason.INVALID_DECLARATION,
                alternative=(
                    "Select a nonzero-delta ey_policy[...] mean, or an ate_policy[...] "
                    "contrast that uses the natural course as its reference."
                ),
            )


def _zero_delta_policy_means(result: Any) -> frozenset[str]:
    """Name every ``ey_policy`` alias whose policy is the zero-delta natural course.

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
    from ..interventions.policy import is_identity_policy
    from ..targets.base import parameter_name

    names = set()
    for policy in result.identified_effect.functional.interventions:
        name = getattr(policy, "name", None)
        if isinstance(name, str) and is_identity_policy(policy):
            names.add(name)
    return frozenset(
        {parameter_name("ey_policy", arm=name) for name in names}
        | {
            alias
            for alias, key in result.parameter_keys.items()
            if key.estimand == "ey_policy" and key.value in names
        }
    )


def _categorical_names(data: Any) -> frozenset[str]:
    """Every encoded categorical column, and every indicator column its encoding generated."""
    return frozenset(
        name for encoding in data.encodings for name in (encoding.column, *encoding.generated)
    )


def _calibration_names(data: Any, benchmark_covariates: Any) -> tuple[str, ...]:
    """Read ``benchmark_covariates`` as column names, and refuse a malformed value.

    A duplicate, a name that is not a string, and a name that is neither a covariate nor a
    categorical column are malformed arguments.  Each one raises ``ValueError`` or
    ``TypeError`` before any refusal.  A categorical column and each of its generated
    indicators are well-formed names; :func:`_calibration_refusal` refuses an indicator.

    Parameters
    ----------
    data : CausalData
        The analysis data of the fit.
    benchmark_covariates : str or sequence of str
        The requested calibration covariates.

    Returns
    -------
    tuple of str
        The requested names, in the order of the request.
    """
    names = tuple(
        [benchmark_covariates] if isinstance(benchmark_covariates, str) else benchmark_covariates
    )
    if len(set(names)) != len(names):
        raise ValueError("benchmark_covariates contains duplicates")
    categorical = _categorical_names(data)
    for name in names:
        if not isinstance(name, str):
            raise TypeError("benchmark_covariates must contain only column names")
        if name not in categorical and name not in data.covariate_names:
            raise ValueError(
                f"benchmark covariate {name!r} is unavailable; adjustment covariates are "
                f"{_benchmark_names(data)}"
            )
    return names


def _benchmark_names(data: Any) -> list[str]:
    """The names a benchmark may take: numeric columns and logical categorical columns."""
    sources = data.indicator_sources
    names: list[str] = []
    for column in data.covariate_names:
        name = sources.get(column, column)
        if name not in names:
            names.append(name)
    return names


def _calibration_refusal(data: Any, names: tuple[str, ...]) -> Refusal | None:
    """Return the refusal of the first requested covariate that cannot calibrate.

    A categorical covariate calibrates as one block over every encoded indicator, so the
    request names its logical column.  A generated indicator is one level of that block,
    and benchmarking it alone does not define a covariate's strength, so it is refused with
    the logical name.  A covariate constant on the positive-weight rows calibrates nothing.

    Parameters
    ----------
    data : CausalData
        The analysis data of the fit.
    names : tuple of str
        Well-formed names from :func:`_calibration_names`.

    Returns
    -------
    Refusal or None
        The refusal for an indicator column or a constant covariate, or ``None`` when
        every named covariate can calibrate.
    """
    # Deferred because ``simulated_confounding`` imports this module at its own module
    # scope.  The weighted-statistics block stays beside the perturbation law that shares it.
    from .simulated_confounding import _block_columns, _is_constant_under_weights

    sources = data.indicator_sources
    for name in names:
        if name in sources:
            return Refusal(
                RefusalReason.INVALID_DECLARATION,
                f"simulated_confounding benchmarks a categorical covariate as one block of its "
                f"indicators; {name!r} is one indicator of {sources[name]!r}.",
                alternative=f"Name the logical covariate {sources[name]!r}.",
            )
        block = _block_columns(data, name)
        if not block:
            return Refusal(
                RefusalReason.INVALID_DECLARATION,
                f"simulated_confounding cannot calibrate {name!r}: none of its encoded "
                "indicators remains in the adjustment set",
            )
        if all(
            _is_constant_under_weights(data.covariates[:, column], data.weights) for column in block
        ):
            return Refusal(
                RefusalReason.INVALID_DECLARATION,
                f"simulated_confounding cannot calibrate constant covariate {name!r}",
            )
    return None


def simulated_confounding_refusal(
    result: Any, estimand: str = "ate", benchmark_covariates: Any = ()
) -> Refusal | None:
    """Return the refusal a ``simulated_confounding`` request meets before any draw.

    :func:`~cleverly.sensitivity.simulated_confounding` raises the sentence this function
    returns, and the ``simulated_confounding`` capability row of
    :class:`~cleverly.assessment.SensitivityFacade` quotes it for the same request.  The
    function reads the fit-wide rules, then :func:`_validated_parameter`, which is the
    check the call runs.  The strength grid cannot change a refusal, so the function does
    not read it.

    A malformed argument is not a refusal.  An unknown estimand, a continuous fit that
    names no policy parameter, and a duplicate, non-string or unknown covariate name each
    raise ``ValueError`` or ``TypeError`` from the call.  This function returns ``None``
    for them, and the call reports them itself.

    Parameters
    ----------
    result : Any
        The fitted result the call would perturb.
    estimand : str
        The parameter alias the call would report.
    benchmark_covariates : str or sequence of str
        The requested calibration covariates.

    Returns
    -------
    Refusal or None
        The refusal, or ``None`` when the call reaches its calibration and its
        draw, or refuses the request as a malformed argument.
    """
    refusal = _fit_wide_refusal(result)
    if refusal is not None:
        return refusal
    try:
        _validated_parameter(result, estimand, benchmark_covariates)
    except CapabilityError as error:
        return error.refusal
    except (TypeError, ValueError):
        return None
    return None


def _validate_request(
    result: Any,
    estimand: str,
    grid: ConfounderStrengthGrid,
    benchmark_covariates: Any,
    *,
    arm_pair: Any = None,
    latent_level: Any = None,
    latent_rho: Any = None,
    observation_reads: Any = None,
    bounded_outcome_law: Any = None,
) -> _ValidatedRequest:
    """Validate the complete supported boundary before a refit or random draw.

    The fit-wide rules come first, then the two grid ranges, then
    :func:`_validated_parameter`.  :func:`simulated_confounding_refusal` reads the first
    and the last of the three.
    """
    # Every fit-wide boundary lives in one ordered table, which the capability row reads
    # through the same helper.  An unsupported fit is unsupported whatever grid the caller
    # passes, so these refusals precede the two grid-range checks below.
    fit_wide_refusal = _fit_wide_refusal(result)
    if fit_wide_refusal is not None:
        raise CapabilityError.from_refusal(fit_wide_refusal)
    data = result.data
    if data.treatment_kind == "vector":
        _check_component_grid(data, grid)
    elif grid.components:
        raise ValueError(
            f"ConfounderStrengthGrid(treatment={{...}}) declares one strength axis per "
            f"component of a vector treatment, and treatment {data.treatment_name!r} is one "
            "column. Pass its strengths as one sequence, treatment=(0.0, ...)"
        )
    # The ``multi_arm`` rule already refused every treatment that is neither continuous nor
    # binary, so a treatment that is not continuous is binary here.
    elif not data.is_continuous_treatment and any(
        value < 0.0 or value > 0.5 for value in cast("tuple[float, ...]", grid.treatment)
    ):
        raise ValueError("binary treatment strengths must be between 0 and 0.5")
    if data.family == "binomial" and any(value < 0.0 or value > 0.5 for value in grid.outcome):
        raise ValueError("binomial outcome strengths must be between 0 and 0.5")
    level, rho = _latent_level(data, grid, latent_level, latent_rho)
    _check_intermediate_axis(result, grid)
    reads = _observation_reads(data, grid, observation_reads)
    law, bounds = _bounded_outcome_law(result, bounded_outcome_law)
    request = _validated_parameter(result, estimand, benchmark_covariates)
    if data.treatment_kind == "vector":
        arm_codes, component_pairs = None, _component_pairs(data, arm_pair)
    else:
        arm_codes, component_pairs = _arm_codes(data, request.key, arm_pair), ()
    return replace(
        request,
        arm_codes=arm_codes,
        component_pairs=component_pairs,
        latent_level=level,
        latent_rho=rho,
        observation_reads=reads,
        bounded_outcome_law=law,
        outcome_bounds=bounds,
    )


_BOUNDED_LAWS = ("logit", "normal_score")


def _bounded_outcome_law(
    result: Any, bounded_outcome_law: Any
) -> tuple[Literal["logit", "normal_score"] | None, tuple[float, float] | None]:
    """Resolve the outcome law of a Gaussian fit whose estimator declares ``q_bounds``.

    Raises
    ------
    ValueError
        If the law is unknown, or is given on a fit without declared bounds.
    """
    bounds = getattr(result.estimator, "q_bounds", None)
    if result.data.family != "gaussian" or bounds is None:
        if bounded_outcome_law is not None:
            raise ValueError(
                "bounded_outcome_law= reads the q_bounds of a Gaussian fit, and this fit "
                "declares none"
            )
        return None, None
    law: Literal["logit", "normal_score"]
    if bounded_outcome_law is None or bounded_outcome_law == "logit":
        law = "logit"
    elif bounded_outcome_law == "normal_score":
        law = "normal_score"
    else:
        raise ValueError(
            f"bounded_outcome_law must be one of {_BOUNDED_LAWS}; got {bounded_outcome_law!r}"
        )
    low, high = (float(value) for value in bounds)
    return law, (low, high)


def _check_intermediate_axis(result: Any, grid: ConfounderStrengthGrid) -> None:
    """Refuse a moved treatment on a controlled direct effect, and an orphan intermediate axis.

    Raises
    ------
    ValueError
        If the grid flips an intermediate that the fit does not carry.
    CapabilityError
        If a cell moves the treatment of a controlled-direct-effect fit.
    """
    if not declares_intermediate(result):
        if any(value != 0.0 for value in grid.intermediate):
            raise ValueError(
                "ConfounderStrengthGrid(intermediate=...) flips the intermediate of a "
                "controlled direct effect, and this fit declares no intermediate"
            )
        return
    if _moved_treatment_columns(result.data, grid):
        raise CapabilityError.from_refusal(_INTERMEDIATE_REFUSAL)


def _observation_reads(
    data: Any, grid: ConfounderStrengthGrid, observation_reads: Any
) -> tuple[str, ...] | None:
    """Resolve the covariates the observation mechanisms read, and refuse a moved treatment.

    On a fit with a missing outcome or a missing treatment, the outcome law acts on the
    recorded outcomes only.  The latent is independent of the data, so outcome-axis cells
    keep missing at random whenever the fit had it.  A treatment perturbation keeps it only
    when every observation mechanism reads the covariates alone, ``Delta`` independent of
    ``(A', Y')`` given ``W``.  The caller declares that by naming the covariates.  The code
    cannot verify the declaration, so the report states it.

    Parameters
    ----------
    data : CausalData
        The fitted data.
    grid : ConfounderStrengthGrid
        The requested strengths.
    observation_reads : sequence of str or None
        The declared covariates.

    Returns
    -------
    tuple of str or None
        The declared covariates, or ``None``.

    Raises
    ------
    ValueError
        If the declaration is malformed, names a role other than an adjustment covariate,
        or is given on a fit with no missing outcome or treatment.
    CapabilityError
        If a cell moves the treatment of such a fit and nothing is declared.
    """
    missing = data.has_missing_outcome or data.has_missing_treatment
    if observation_reads is None:
        if missing and _moved_treatment_columns(data, grid):
            raise CapabilityError.from_refusal(
                _MISSING_OUTCOME_REFUSAL if data.has_missing_outcome else _MISSING_TREATMENT_REFUSAL
            )
        return None
    if not missing:
        raise ValueError(
            "observation_reads= declares the observation mechanisms of a fit with a missing "
            "outcome or treatment; this fit observes every outcome and treatment"
        )
    if isinstance(observation_reads, str) or not all(
        isinstance(name, str) for name in observation_reads
    ):
        raise TypeError("observation_reads must be a sequence of covariate names")
    names = tuple(observation_reads)
    if len(set(names)) != len(names):
        raise ValueError(f"observation_reads names a covariate twice: {names!r}")
    roles = {
        data.treatment_name: "the treatment",
        data.outcome_name: "the outcome",
    }
    for component in data.component_names or ():
        roles[component] = "a treatment component"
    if getattr(data, "intermediate_name", None):
        roles[data.intermediate_name] = "the intermediate"
    covariates = set(_benchmark_names(data))
    for name in names:
        if name in roles:
            raise ValueError(
                f"observation_reads names {name!r}, which is {roles[name]}. The declaration "
                "states that every observation mechanism reads the covariates alone, so it "
                "names adjustment covariates"
            )
        if name not in covariates:
            raise ValueError(
                f"observation_reads names {name!r}, which is not an adjustment covariate of "
                f"this fit; choose from {sorted(covariates)}"
            )
    return names


def _component_pairs(data: Any, arm_pair: Any) -> tuple[tuple[str, tuple[float, float]], ...]:
    """Resolve the pair each categorical component of more than two levels swaps.

    A vector treatment declares ``arm_pair={component: (a, b)}`` for exactly the
    categorical components with more than two levels.  A binary component has one pair
    and a dose has none, so neither may be named.

    Parameters
    ----------
    data : CausalData
        The analysis data of a vector-treatment fit.
    arm_pair : mapping or None
        The declared pairs of component levels.

    Returns
    -------
    tuple of (str, tuple of float)
        The level codes of each swapped pair, in component order.

    Raises
    ------
    ValueError
        If a component that needs a pair has none, or a pair is malformed.
    """
    needs = {
        name: levels
        for name, levels in zip(data.component_names, data.component_levels, strict=True)
        if len(levels) > 2
    }
    if arm_pair is None:
        declared: dict[Any, Any] = {}
    elif isinstance(arm_pair, Mapping):
        declared = dict(arm_pair)
    else:
        raise ValueError(
            "a vector treatment declares arm_pair= as a mapping from each categorical "
            "component of more than two levels to the two levels it swaps"
        )
    if set(declared) != set(needs):
        raise ValueError(
            f"arm_pair= must name exactly the categorical components of more than two levels, "
            f"{sorted(needs)}; got {sorted(map(str, declared))}"
        )
    out = []
    for name in data.component_names:
        if name not in needs:
            continue
        levels = tuple(needs[name])
        pair = tuple(declared[name])
        if len(pair) != 2 or pair[0] == pair[1] or any(level not in levels for level in pair):
            raise ValueError(
                f"arm_pair[{name!r}] names two different levels of {list(levels)}; got "
                f"{declared[name]!r}"
            )
        out.append((name, (float(levels.index(pair[0])), float(levels.index(pair[1])))))
    return tuple(out)


def _arm_codes(data: Any, key: Any, arm_pair: Any) -> tuple[float, float] | None:
    """Resolve the two arms that the pairwise swap of a multi-arm treatment exchanges.

    A two-arm contrast declares its pair: the arm it reports and its reference.  Any other
    parameter of a multi-arm fit needs ``arm_pair=(a, b)``, with ``a`` the arm the
    induced association reads.  A two-arm treatment has one pair, so ``arm_pair`` may only
    restate it there, and a dose or a vector treatment takes no pair.

    Parameters
    ----------
    data : CausalData
        The analysis data of the fit.
    key : ParameterKey
        The structured key of the requested parameter.
    arm_pair : tuple or None
        The declared pair of arm labels.

    Returns
    -------
    tuple of (float, float) or None
        The arm codes ``(a, b)`` of a multi-arm treatment, else ``None``.

    Raises
    ------
    ValueError
        If the pair is missing on a multi-arm fit, names an unknown or repeated arm,
        disagrees with the contrast it reports, or is given where no pair applies.
    """
    levels = tuple(data.treatment_levels)
    multi_arm = (
        data.has_arms
        and data.treatment_kind != "vector"
        and not data.is_continuous_treatment
        and not data.is_binary_treatment
    )
    if arm_pair is not None:
        if not multi_arm and not (data.has_arms and data.is_binary_treatment):
            raise ValueError(
                "arm_pair= names two arms of an arm-coded treatment; "
                "this fit's treatment is not arm-coded"
            )
        pair = tuple(arm_pair)
        if len(pair) != 2 or pair[0] == pair[1] or any(arm not in levels for arm in pair):
            raise ValueError(
                f"arm_pair= names two different arms of {list(levels)}; got {arm_pair!r}"
            )
    else:
        pair = None
    declared = (
        (key.value, key.reference)
        if key is not None
        and key.axis == "arm"
        and key.estimand in _CONTRAST_TYPES
        and key.reference is not None
        else None
    )
    if pair is not None and declared is not None and set(pair) != set(declared):
        raise ValueError(
            f"arm_pair={pair!r} disagrees with the contrast {key.alias!r}, which swaps {declared!r}"
        )
    if not multi_arm:
        if pair is not None and set(pair) != set(levels):
            raise ValueError(f"a two-arm treatment swaps {list(levels)}; got arm_pair={pair!r}")
        return None
    resolved = pair if pair is not None else declared
    if resolved is None:
        raise ValueError(
            f"treatment {data.treatment_name!r} has {len(levels)} arms, and the pairwise swap "
            "needs the two it exchanges: pass arm_pair=(a, b), with a the arm whose "
            "association the cells report"
        )
    return (float(levels.index(resolved[0])), float(levels.index(resolved[1])))


#: The levels a caller may declare for the latent cause.
_LATENT_LEVELS: frozenset[str] = frozenset({"row", "cluster", "mixed"})


def _latent_level(
    data: Any, grid: ConfounderStrengthGrid, latent_level: Any, latent_rho: Any
) -> tuple[Literal["row", "cluster", "mixed"], float | None]:
    """Check the declared level of the latent cause against the fit's clusters.

    A clustered fit declares the level, because no source fixes it (hard-stop review F9).
    An unclustered fit has one level, ``"row"``.  A row-level or mixed latent flips or
    shifts single rows, so on a treatment that is constant within every cluster it breaks
    the cluster assignment the fit modelled.  That refusal applies only when a cell moves
    the treatment: an outcome-only grid changes no treatment.

    Parameters
    ----------
    data : CausalData
        The analysis data of the fit.
    grid : ConfounderStrengthGrid
        The declared strengths.
    latent_level : str or None
        The declared level.
    latent_rho : float or None
        The intra-cluster correlation of a ``"mixed"`` latent.

    Returns
    -------
    tuple of (str, float or None)
        The resolved level and correlation.

    Raises
    ------
    ValueError
        If the level is missing on a clustered fit, unknown, or inconsistent with the
        correlation.
    CapabilityError
        If a row-level or mixed latent would move a cluster-constant treatment.
    """
    clustered = data.cluster is not None
    if latent_level is None:
        if clustered:
            raise ValueError(
                f"the fit is clustered on {data.cluster_name!r}, and no source fixes the level "
                "of a simulated common cause there: declare latent_level='row', 'cluster' "
                "or 'mixed' (with latent_rho=)"
            )
        latent_level = "row"
    if latent_level not in _LATENT_LEVELS:
        raise ValueError(
            f"latent_level must be one of {sorted(_LATENT_LEVELS)}; got {latent_level!r}"
        )
    if not clustered and latent_level != "row":
        raise ValueError(
            f"latent_level={latent_level!r} needs a clustered fit; this fit declares no cluster"
        )
    if latent_level == "mixed":
        if (
            isinstance(latent_rho, bool)
            or not isinstance(latent_rho, Real)
            or not 0.0 < float(latent_rho) < 1.0
        ):
            raise ValueError(
                "latent_level='mixed' needs latent_rho= strictly between 0 and 1, the "
                f"intra-cluster correlation of the latent; got {latent_rho!r}"
            )
        rho: float | None = float(latent_rho)
    else:
        if latent_rho is not None:
            raise ValueError(
                "latent_rho= sets the correlation of latent_level='mixed'; "
                f"got latent_level={latent_level!r}"
            )
        rho = None
    moved_constant = (
        _moved_cluster_constant_columns(data, grid)
        if clustered and latent_level != "cluster"
        else ()
    )
    if moved_constant:
        raise CapabilityError(
            f"simulated_confounding refuses latent_level={latent_level!r} with a nonzero "
            f"treatment strength on {', '.join(map(repr, moved_constant))}: that treatment is "
            f"constant within every cluster of {data.cluster_name!r}, and a latent that varies "
            "within a cluster would move single rows of a cluster-level assignment.",
            reason=RefusalReason.INVALID_DECLARATION,
            alternative=(
                "Declare latent_level='cluster', or keep the strength of every cluster-level "
                "treatment at zero."
            ),
        )
    return latent_level, rho


def _moved_treatment_columns(
    data: Any, grid: ConfounderStrengthGrid
) -> tuple[tuple[str, np.ndarray[Any, Any]], ...]:
    """The treatment columns that some grid point moves, with their values.

    A vector is read per component.  A one-column treatment is one column.
    """
    if (data.treatment_components is not None) != bool(grid.components):
        return ()  # _check_component_grid refuses this grid shape
    if grid.components:
        values = np.asarray(data.treatment_components, dtype=float)
        names = tuple(data.component_names)
        moved = {
            name
            for point in grid.treatment_points
            for name, value in zip(grid.components, point, strict=True)  # type: ignore[arg-type]
            if value != 0.0
        }
    else:
        values = np.asarray(data.treatment, dtype=float).reshape(data.n, 1)
        names = (data.treatment_name,)
        moved = (
            {data.treatment_name} if any(value != 0.0 for value in grid.treatment_points) else set()
        )
    return tuple(
        (name, column) for name, column in zip(names, values.T, strict=True) if name in moved
    )


def _moved_cluster_constant_columns(data: Any, grid: ConfounderStrengthGrid) -> tuple[str, ...]:
    """The treatment columns that hold one value per cluster and that some grid point moves.

    A vector is read per component: a row-varying latent may move a component that varies
    within clusters, but not a component that is constant within every cluster.
    """
    cluster = np.asarray(data.cluster)
    n_clusters = np.unique(cluster).size
    return tuple(
        name
        for name, column in _moved_treatment_columns(data, grid)
        if np.unique(np.column_stack((cluster, column)), axis=0).shape[0] == n_clusters
    )


def _check_component_grid(data: Any, grid: ConfounderStrengthGrid) -> None:
    """Refuse a grid that does not give each component of a vector one strength axis.

    Each component takes its own law, a flip for a binary component and a linear shift for
    a dose, so one strength cannot serve both: a flip probability and a coefficient are
    not on one scale.
    """
    columns = grid.components
    if set(columns) != set(data.component_names):
        raise ValueError(
            f"treatment {data.treatment_name!r} is a vector, and a flip probability and a "
            "linear coefficient are not on one scale, so the grid gives each component its "
            "own strengths: ConfounderStrengthGrid(treatment={"
            + ", ".join(f"{name!r}: (0.0, ...)" for name in data.component_names)
            + f"}}, outcome=...). This grid names {list(columns)}"
        )
    binary = {
        name
        for name, levels in zip(data.component_names, data.component_levels, strict=True)
        if levels
    }
    for column, strengths in grid.treatment:  # type: ignore[misc]
        if column in binary and any(value < 0.0 or value > 0.5 for value in strengths):
            raise ValueError(
                f"binary treatment component {column!r} strengths must be between 0 and 0.5"
            )


def _validated_parameter(
    result: Any, estimand: str, benchmark_covariates: Any
) -> _ValidatedRequest:
    """Check the requested parameter and covariates of a fit that the fit-wide rules admit.

    The malformed covariate names come first, then the parameter, then the calibration
    refusal.  A refusal raises ``CapabilityError``, which
    :func:`simulated_confounding_refusal` returns as its sentence.
    """
    from ..study import NaturalCourseMean, ParameterKey

    estimator = result.estimator
    data = result.data
    names = _calibration_names(data, benchmark_covariates)
    # The ``multi_arm`` rule already refused every treatment that is neither continuous nor
    # binary, so exactly one of the two families holds here.
    treatment_family: Literal["binary", "categorical", "continuous", "vector"] = (
        "vector"
        if data.treatment_kind == "vector"
        else "continuous"
        if data.is_continuous_treatment
        else "binary"
        if data.is_binary_treatment
        else "categorical"
    )
    identified = result.identified_effect
    functional = identified.functional
    key = result.parameter_keys.get(estimand)
    if (estimand == "ate" and type(identified.estimand) is NaturalCourseMean) or (
        type(key) is ParameterKey and key.estimand == "ey_obs"
    ):
        raise CapabilityError(
            "simulated_confounding refuses NaturalCourseMean; the natural-course mean is "
            "E[Y] and carries no counterfactual treatment dependence for a simulated "
            "common cause to move.",
            reason=RefusalReason.INVALID_DECLARATION,
            alternative=(
                "Fit a supported PAR or PAF contrast to compare the observed mean with a "
                "reference intervention."
            ),
        )
    if treatment_family not in {"binary", "categorical"} and estimand == "ate":
        vacuous = _zero_delta_policy_means(result)
        admissible = [
            name
            for name in result.estimates
            if name.startswith(("ey_policy[", "ate_policy[", "msm[")) and name not in vacuous
        ]
        detail = (
            f"choose one of {admissible}"
            if admissible
            else "this fit reports none that this surface can assess"
        )
        raise ValueError(
            "continuous simulated_confounding requires an explicit ey_policy[...] policy mean "
            f"or ate_policy[...] contrast alias, or an msm[...] coefficient alias; {detail}"
        )
    if estimand not in result.estimates:
        if treatment_family in {"binary", "categorical"}:
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
            f"simulated_confounding needs a structured parameter key for {estimand!r}",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    baseline_mask = _baseline_population(result, key, identified)
    refusal = _replay_refusal(estimator, key.estimand)
    if refusal is not None:
        raise CapabilityError.from_refusal(refusal)
    if fixed_axis(key.estimand) is not None:
        estimator = validate_fixed_replay(result, estimand, key)
    elif treatment_family == "binary" and key.estimand in INCREMENTAL_TARGETS:
        estimator = validate_incremental_replay(result, estimand, key)
    elif treatment_family in {"binary", "categorical"}:
        check_only_declared_axis(
            result,
            key,
            "arm",
            Refusal(
                RefusalReason.IMPLEMENTATION_GAP,
                "simulated_confounding supports an arm-indexed parameter, "
                "not a regimen, stochastic, incremental, modified-policy, or MSM parameter.",
                alternative=(
                    "Call simulated_confounding on a binary fit that declares only the arm "
                    "axis, such as TMLE(estimands=('ate',)) with no other counterfactual "
                    "keyword."
                ),
                roadmap="X82",
            ),
            intermediate=True,
        )
        if key.estimand not in _BINARY_PARAMETER_TARGETS or key.axis != "arm":
            raise CapabilityError(
                "simulated_confounding supports only an ATE, ATT, ATC, counterfactual arm "
                "mean, risk ratio, odds ratio, population attributable risk, or population "
                "attributable fraction; other parameters are outside its source boundary.",
                reason=RefusalReason.IMPLEMENTATION_GAP,
                alternative=(
                    "Call simulated_confounding(result, estimand=...) for an ATE, ATT, ATC, "
                    "arm mean, rr, or, par or paf of a fit of the same data."
                ),
                roadmap="X82",
            )
        _validate_binary_parameter_state(result, estimand, key, identified, functional, estimator)
    else:
        check_only_declared_axis(
            result,
            key,
            "policy",
            Refusal(
                RefusalReason.IMPLEMENTATION_GAP,
                "continuous simulated_confounding supports a modified-treatment-policy "
                "parameter, not an arm, regimen, stochastic, incremental, or MSM parameter.",
                alternative=(
                    "Call simulated_confounding on a continuous fit that declares policies= "
                    "alone, such as TMLE(policies=[Shift(delta, cap=...)])."
                ),
                roadmap="X82",
            ),
        )
        if key.estimand not in {"ey_policy", "ate_policy"} or key.axis != "policy":
            raise CapabilityError(
                "continuous simulated_confounding supports only an ey_policy policy mean or "
                "ate_policy contrast; other parameters are outside its source boundary.",
                reason=RefusalReason.IMPLEMENTATION_GAP,
                alternative=(
                    "Call simulated_confounding(result, estimand='ate_policy[...]'), or an "
                    "ey_policy mean other than the natural course."
                ),
                roadmap="X82",
            )
        _validate_continuous_policy_state(result, estimand, key, identified, functional, estimator)

    calibration_refusal = _calibration_refusal(data, names)
    if calibration_refusal is not None:
        raise CapabilityError.from_refusal(calibration_refusal)
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
        key=key,
    )
