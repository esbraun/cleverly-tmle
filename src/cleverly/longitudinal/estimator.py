r"""Longitudinal TMLE: the mean outcome under a treatment regimen.

:class:`LTMLE` estimates :math:`E[Y_{\bar a}]` -- what the mean outcome would have been
had every unit followed the plan :math:`\bar a = (a_1, \ldots, a_T)` and stayed under
observation -- for each regimen the caller declares, plus the contrast of each against a
reference.  The estimator is the sequential-regression TMLE of van der Laan & Gruber
(2012); :mod:`cleverly.longitudinal.sequential` states the recursion, the submodel and
the influence function it solves.

What this buys over the point-treatment estimator is the whole reason the module exists:
a covariate measured between the two treatment decisions is a *confounder of the second
and a consequence of the first*, and no single adjustment set handles both roles.
Conditioning on it blocks the effect of :math:`A_1` that runs through it; leaving it out
confounds :math:`A_2`.  The sequential regression conditions on it at the node where it
is a confounder and integrates it out at the node where it is a mediator, which is what
the recursion is for.

The identification assumptions are the sequential versions of the familiar three, and
they are strictly stronger than at one time point:

* **sequential exchangeability** -- at every node, treatment is independent of the
  counterfactual outcome given the recorded history to that point.  Censoring is
  assumed non-informative on the same terms;
* **sequential positivity** -- every unit has a positive probability of the regimen's
  arm at every node given its history, and of remaining under observation.  The clever
  covariate divides by a *product* of :math:`2T` probabilities, so this is the
  assumption that bites first: :meth:`LongitudinalResult.diagnostics` reports the
  cumulative weight it produced;
* **consistency**, and no interference.

With a **survival** outcome -- one absorbing event indicator per node, declared by passing
``outcome=[...]`` rather than a single name -- the parameter is instead the cumulative
risk curve :math:`F_{\bar a}(t) = P(\text{event by } t)` at every horizon, reported with
the joint influence-curve matrix that makes a simultaneous band over the curve the natural
object.  It is the same recursion and the same clever covariate; what moves is which rows
each node's regression is fitted on.

``weights=`` names a column of observation weights and means what it means on a
point-treatment fit: the parameter is the declared one in the tilted population
:math:`dP_w = w\,dP/E[w]`, every node's nuisance is fitted by weighted loss, every node's
score equation is weighted, and the reported curve is :math:`(w/E[w])\,D^*(P_w)`.  See
:mod:`cleverly.data.weighting` for the statement and its limits, and
:mod:`cleverly.longitudinal.sequential` for what a weight is *not* -- a factor in the
clever covariate.

What is refused rather than approximated is listed in
``docs/technical-reference/longitudinal-tmle.md``, under *Variations*; the short version is
that this estimator answers for a regimen -- static or dynamic -- over a binary or categorical
treatment at every node, for one end-of-study outcome or one absorbing event per cause, with
monotone censoring.  Every point-treatment keyword it does not take is accepted and
rejected with what the derivation would need, rather than arriving as an ``unexpected
keyword argument``; :data:`_REFUSED` is that table.

``msm=`` declares a **working model over the regimens** and makes the report its
coefficients rather than a mean per plan -- the answer to the :math:`2^T` problem, and
what the applied literature reports for a grid of dynamic rules.  It is a projection and
not an assumption; :mod:`cleverly.longitudinal.msm` states it, and says why its
fluctuation is pooled across the regimens where the point-treatment one is not.
"""

from __future__ import annotations

import copy
from abc import ABCMeta, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NamedTuple, NoReturn, TypeAlias

import numpy as np
from sklearn.base import clone

from .._inference_status import (
    ADAPTIVE_BASELINE_STATUS,
    CONSTANT_NODE_STATUS,
    MINIMUM_LONGITUDINAL_INTERVAL_CLUSTERS,
    NO_SIMULTANEOUS_BANDS,
    NO_T_REFERENCE_BANDS,
    PARAMETER_STATUSES,
    T_REFERENCE_NOTE,
    TRIMMED_RATIO_STATUS,
    InferenceStatus,
    is_labelled,
    precedent_status,
    status_record,
    supplies_inference,
)
from .._support import (
    UNCLASSIFIED,
    Classification,
    Support,
    SupportKey,
    band_support,
    classify_longitudinal_fit,
    combine_support,
    record_fit,
    regimen_kinds,
    support_for,
)
from .._typing import (
    BoolArray,
    CumulativeGBounds,
    FloatArray,
    IterateChoice,
    Learner,
    StoppingRule,
)
from .._warnings import FitWarning, records_fit
from .._warnings import warn as _warn
from ..data.weighting import REPORTED_DRAW
from ..estimators._option_support import stopping_conditions
from ..exceptions import (
    CapabilityError,
    CleverlyError,
    DataError,
    DataWarning,
    LongitudinalError,
    MethodConfigurationError,
    MonteCarloWarning,
    ParameterSpaceWarning,
    PositivityWarning,
    Refusal,
    RefusalReason,
)
from ..fluctuation._score import (
    StopContext,
    check_iterate,
    check_step_bound,
    check_stopping_rule,
    residual_line,
    rule_failure_line,
)
from ..inference.bootstrap import (
    BootstrapBand,
    BootstrapInterval,
    BootstrapResult,
    Resampling,
    bootstrap_band,
    run_bootstrap,
    run_index_bootstrap,
)
from ..inference.cluster import (
    cluster_inference_status,
    cluster_reference_df,
    fewest_clusters,
    influence_covariance,
    positive_mass_clause,
)
from ..inference.delta import Transform
from ..inference.influence import (
    ParameterEstimate,
    Scale,
    bootstrap_line,
    has_t_reference,
    make_estimate,
    median_estimates,
    reference_label,
    spread_name,
    support_column,
    support_notes,
    with_precedent_status,
)
from ..inference.multiplier import SimultaneousBands, median_bands, simultaneous_bands
from ..inference.results import (
    RatioKind,
    _named_draw,
    attach_bootstrap,
    check_transform_scale,
    estimate_covariance,
    estimate_curves,
    inference_status,
    linear_function,
    linear_functional,
    median_covariance,
    median_derived,
    ratio_contrast,
    ratio_function,
    refuse_odds_ratio,
    repeat_spread,
    reported_status,
    select_estimates,
    smooth_contrast,
    sole_estimate,
    split_noise_lines,
    with_derived_bootstrap,
)
from ..inference.sparse import influence_report, warn_sparse_influence
from ..interventions import RiskRatioTilt

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..assessment import AssessmentReport
from .._summary import Detail
from ..interventions.policy import Support as DeclaredSupport
from ..learners._fitting import supports_sample_weight
from ..learners.crossfit import (
    CrossFitPlan,
    Folds,
    SplitPlan,
    _fresh_seed,
    check_integrity,
    random_partition,
)
from ..learners.glm import GLM
from ..learners.library import _validate_learner
from ..learners.super_learner import SuperLearner, resolve_learner
from ..methods import (
    BASELINE_HAZARD_POOLING_REFUSAL,
    BASELINE_RATIO_TRIM_REFUSAL,
    IPSI_NO_FLUCTUATION,
    IPSI_POINT_REFUSAL,
    IPSI_RATIO_TRIM_REFUSAL,
    LTMLE_TILT_REFUSAL,
    NODE_BOUNDS_MSM_REFUSAL,
    REPEATED_OUTCOME_REFUSAL,
    SDR_TILT_REFUSAL,
    baseline_no_fluctuation_refusal,
    node_bounds_refusal,
    sdr_no_fluctuation_refusal,
)
from ..msm import MSM, refuse_msm_functions
from ..protocol import protocol_summary_lines
from ..provenance import Provenance, fingerprint_array, fold_fingerprint
from ..provenance import build as provenance_build
from ..targets.base import parameter_name
from ..utils.bounds import (
    DEFAULT_LTMLE_G_BOUNDS,
    OutcomeScaler,
    policy_scale_refusal,
    resolve_cumulative_g_bounds,
)
from ..utils.phases import PhaseProfile, phase, profile_phases
from ..utils.text import format_draw, format_pvalue, format_table
from .baselines import (
    IPW_NORMALIZATIONS,
    fit_gcomp_cell,
    fit_ipw_cell,
    mechanism_reads_parametric,
)
from .data import LongitudinalData, Repeated
from .frequency import FrequencyReplicate, ReportedName, frequency_exact
from .gformula import (
    GFormulaReport,
    GFormulaSettings,
    declared_natural,
    has_set_dose,
    monte_carlo_share,
    normal_learner_refusal,
    plan_node,
    resolve_covariates,
    run_gformula,
)
from .hazard import (
    HazardAlgorithm,
    InitialHazards,
    check_hazard_node_bounds,
    fit_hazards,
    hazard_bounds,
    level_plans,
    project_curves,
    refuse_hazard_design,
    refuse_hazard_settings,
    target_regimens,
)
from .ipsi import IPSICell, fit_cells_ipsi
from .msm import (
    MSMRegimenFit,
    RegimenMSM,
    evaluate_regimen_msm,
    fit_regimens_msm,
    project_sdr_cells,
    refuse_evaluated_msm_functions,
)
from .pooled_hazards import check_hazard_pooling, refuse_pooled_designs
from .ratio_trim import check_ratio_trim, trim_ratios
from .regimen import (
    DynamicRegimen,
    Plan,
    RegimenSpec,
    attach_policy_numerators,
    declares_tuned,
    describe_plan,
    refuse_regimen_rules,
    resolve_plans,
    resolve_regimens,
    resolve_tuned_plans,
    tilt_mix_refusal,
)
from .sdr import FoldMechanism, SDRCell, fit_cells_sdr, fit_fold_mechanisms
from .sequential import (
    Mechanism,
    NodeBound,
    RegimenFit,
    _filled_matrix,
    constant_nodes,
    fit_mechanism,
    fit_regimen,
    known_product_refusal,
    node_ratios,
    preflight_declared_doses,
    preflight_known_mechanisms,
    preflight_mechanism_support,
    preflight_terminal_outcomes,
)

__all__ = [
    "IPSI",
    "LTMLE",
    "SDR",
    "LongitudinalConfig",
    "LongitudinalResult",
    "ParametricGFormula",
    "SequentialGComputation",
    "SequentialIPW",
    "constant_node_parameters",
    "gcomp",
    "gformula",
    "ipsi",
    "ipw",
    "ltmle",
    "sdr",
]


#: Stable replayability omission codes.  These values are persisted and exposed through
#: ``Replayability.unreconstructible``, so callers can branch on them without parsing prose.
LONGITUDINAL_REPLAY_LEARNER_UNCLONABLE = "longitudinal_replay_learner_unclonable"
LONGITUDINAL_REPLAY_RANDOM_STATE_UNSEEDED = "longitudinal_replay_random_state_unseeded"
LONGITUDINAL_REPLAY_RANDOM_STATE_NON_INTEGER = "longitudinal_replay_random_state_non_integer"
LONGITUDINAL_REPLAY_FITTED_BOUND_MISMATCH = "longitudinal_replay_fitted_bound_mismatch"
LONGITUDINAL_REPLAY_REGIMEN_DECLARATION = "longitudinal_replay_regimen_declaration"
LONGITUDINAL_REPLAY_MSM_DECLARATION = "longitudinal_replay_msm_declaration"

#: The line a summary prints when a repeated fit with ``isotonic=True`` builds no band.
NO_ISOTONIC_REPEAT_BANDS = (
    "No simultaneous bands: each draw projects its levels onto a non-decreasing curve, and no "
    "source gives the projection of the limits of a median band, so the fit reports pointwise "
    "intervals only."
)

#: The :class:`LTMLE` settings that :class:`SDR` refuses, through
#: :func:`~cleverly.methods.sdr_no_fluctuation_refusal`.
SDR_REFUSED_SETTINGS: tuple[str, ...] = (
    "alpha",
    "max_iter",
    "tol",
    "stopping",
    "step_bound",
    "step_search",
    "iterate",
)

#: Why a baseline estimator refuses cross-fitting.  ``{estimator}`` is the class name, which
#: :func:`baseline_cross_fit_refusal` fills in.
_BASELINE_CROSS_FIT_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "{estimator} fits in sample. Its interval is the stacked estimating equation of in-sample "
    "parametric fits. With a data-adaptive learner neither baseline has a root-n result, in "
    "sample or cross-fitted, and cross-fitting does not remove the first-order bias of a "
    "plug-in or IPW estimate.",
    alternative="Pass n_folds=1, or for a cross-fitted analysis fit LTMLE or SDR.",
    roadmap="F35",
)
#: Why a baseline estimator refuses repeated cross-fitting.  ``{estimator}`` is the class
#: name, which :func:`baseline_repeats_refusal` fills in.
_BASELINE_REPEATS_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "{estimator} fits in sample, so repeats above 1 has no split to draw again. Its interval "
    "is the stacked estimating equation of one in-sample parametric fit, and no source gives "
    "the split-adjusted median variance of Chernozhukov et al. (2018, Definition 3.3) for it.",
    alternative="Pass repeats=1, or for a repeated cross-fit fit LTMLE or SDR.",
    roadmap="F35",
)
#: Why a baseline estimator refuses a working model over regimens.
BASELINE_MSM_REFUSAL = Refusal(
    RefusalReason.API_REDIRECTION,
    "sequential g-computation, IPW and the parametric g-formula report regimen levels and "
    "contrasts only.",
    alternative="A working model over regimens is fitted by LTMLE(msm=...) or SDR(msm=...).",
)
#: Why a baseline estimator refuses an odds tilt whose mechanism is estimated.
BASELINE_TILT_REFUSAL = Refusal(
    RefusalReason.API_REDIRECTION,
    "an incremental policy depends on the observed treatment law, so its estimator needs a "
    "mechanism term at every tilt node (Kennedy 2019, Theorem 2).",
    alternative=(
        "Fit IPSI(regimens=[...]). With a known treatment mechanism the tilt is a known "
        "stochastic policy, which both baselines fit."
    ),
)
#: Why IPW refuses a modified treatment policy.
IPW_MTP_REFUSAL = Refusal(
    RefusalReason.API_REDIRECTION,
    "IPW of a modified treatment policy needs the density ratio of the shifted law at each "
    "node, which SequentialIPW does not estimate.",
    alternative="Fit SequentialGComputation, LTMLE or SDR.",
)
#: Why IPW refuses a density setting.  ``{name}`` is the setting.
IPW_DENSITY_REFUSAL = (
    "SequentialIPW estimates no density ratio, so {name}= has nothing to set. It applies to "
    "a modified treatment policy, which SequentialGComputation, LTMLE and SDR fit."
)
#: Why IPW refuses an outcome learner.  ``{name}`` is the setting.
IPW_OUTCOME_LEARNER_REFUSAL = (
    "SequentialIPW fits no outcome regression, so {name}= has nothing to fit. The estimate "
    "reads the treatment and censoring mechanisms only."
)
#: Why g-computation refuses a binomial GLM on a continuous outcome with no declared scale.
GCOMP_SCALE_REFUSAL = Refusal(
    RefusalReason.INVALID_DECLARATION,
    "the quasi-binomial GLM fits a continuous outcome on a scale taken from the observed "
    "range, so its estimate moves with the sample extremes and has no fixed limit for an "
    "unbounded outcome.",
    alternative="Declare q_bounds= at the outcome's known support, or pass GLM(family='gaussian').",
)
#: Why a truncation curve refuses a g-computation result.
GCOMP_TRUNCATION_REFUSAL = Refusal(
    RefusalReason.API_REDIRECTION,
    "a g-computation estimate reads no treatment mechanism, so g_bounds does not move it and "
    "no truncation curve exists.",
    alternative="Fit SequentialIPW or LTMLE for a truncation curve.",
)


def baseline_repeats_refusal(estimator: str) -> Refusal:
    """The refusal of repeated cross-fitting by a sequential baseline estimator.

    Parameters
    ----------
    estimator : str
        The class name of the refusing estimator.

    Returns
    -------
    Refusal
        An :attr:`~cleverly.exceptions.RefusalReason.IMPLEMENTATION_GAP` record that F35 in
        ``docs/roadmap.md`` tracks.
    """
    return replace(
        _BASELINE_REPEATS_REFUSAL,
        condition=_BASELINE_REPEATS_REFUSAL.condition.format(estimator=estimator),
    )


#: Why a parametric g-formula fit refuses a pseudo-outcome learner.
#: Why the parametric g-formula refuses a Markov order.
GFORMULA_MARKOV_REFUSAL = Refusal(
    RefusalReason.API_REDIRECTION,
    "the parametric g-formula simulates each covariate and outcome from a model of the full "
    "history, and it does not restrict those models to a Markov window.",
    alternative=(
        "Omit markov_order=, or fit LTMLE, SDR, SequentialGComputation or SequentialIPW, "
        "whose node models read the window."
    ),
)
GFORMULA_PSEUDO_REFUSAL = Refusal(
    RefusalReason.INVALID_DECLARATION,
    "the parametric g-formula fits no sequential pseudo-outcome regression, so pseudo_learner= "
    "has nothing to fit. The estimate reads the covariate, outcome (or hazard) and "
    "natural-treatment models.",
    alternative="Pass covariate_learner=, outcome_learner= or hazard_learner= instead.",
)
#: Why a parametric g-formula fit refuses a risk-ratio tilt.
GFORMULA_RR_TILT_REFUSAL = Refusal(
    RefusalReason.API_REDIRECTION,
    "the parametric g-formula simulates an odds tilt only (Kennedy 2019). It draws the tilted "
    "treatment from the odds-tilted law, and it has no draw for a risk-ratio tilt.",
    alternative=(
        "Declare the tilt as Incremental(delta), whose draw is validated, or fit LTMLE or SDR "
        "for a risk-ratio tilt."
    ),
)
#: Why a parametric g-formula fit refuses an odds tilt at a node with other than two levels.
GFORMULA_CATEGORICAL_TILT_REFUSAL = Refusal(
    RefusalReason.API_REDIRECTION,
    "the parametric g-formula simulates an odds tilt of a node with two levels only. Its "
    "draw and its score term are those of a binary natural-treatment model, and it has no "
    "draw for a categorical odds tilt.",
    alternative="Fit IPSI, which reads an Incremental with one factor per level of a node.",
)
#: Why the parametric g-formula refuses a vector treatment node with a dose component.
GFORMULA_MIXED_NODE_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "the parametric g-formula has no joint model of a vector treatment node with a dose "
    "component. It draws one natural treatment column per node.",
    alternative="Fit LTMLE or SDR, which read a mixed node through its joint policy.",
    roadmap="X73",
)
#: Why a truncation curve refuses a parametric g-formula result.
GFORMULA_TRUNCATION_REFUSAL = Refusal(
    RefusalReason.API_REDIRECTION,
    "a parametric g-formula estimate divides by no treatment mechanism, so g_bounds does not "
    "move it and no truncation curve exists.",
    alternative="Fit SequentialIPW or LTMLE for a truncation curve.",
)


def baseline_cross_fit_refusal(estimator: str) -> Refusal:
    """The refusal of cross-fitting by a sequential baseline estimator.

    Parameters
    ----------
    estimator : str
        The class name of the refusing estimator.

    Returns
    -------
    Refusal
        An :attr:`~cleverly.exceptions.RefusalReason.IMPLEMENTATION_GAP` record that F35 in
        ``docs/roadmap.md`` tracks.
    """
    return replace(
        _BASELINE_CROSS_FIT_REFUSAL,
        condition=_BASELINE_CROSS_FIT_REFUSAL.condition.format(estimator=estimator),
    )


#: The warning when the diagnostic mechanism of a g-computation fit does not fit.
GCOMP_MECHANISM_WARNING = (
    "the diagnostic treatment and censoring mechanism of this g-computation fit did not fit "
    "({error}). The g-computation estimate does not read it, so the fit continues. Where no "
    "unit follows the regimen, the outcome GLMs extrapolate, and the interval holds only if "
    "they are correct there. The positivity diagnostics and the diagnostic curve read the "
    "mechanism, so they are unavailable (nan)"
)


#: Why a one-step level can leave the outcome range, by estimator.
_RANGE_CAUSE: dict[str, str] = {
    "sdr": (
        "An SDR estimate is an average of transformed outcomes and can leave the outcome "
        "range (Díaz et al. 2023, Section 5.3)"
    ),
    "ipsi": (
        "Kennedy's estimator averages influence-function values and can leave the outcome "
        "range (Kennedy 2019, Section 4.1)"
    ),
    "ipw": (
        "A Horvitz-Thompson IPW estimate divides by n rather than by the sum of its weights, "
        "and can leave the outcome range"
    ),
}


def refuse_sdr_scale(
    name: str,
    value: float,
    scale: str,
    *,
    estimator: str = "sdr",
    draw: tuple[int, int] | None = None,
) -> None:
    """Refuse a log or odds functional of a one-step level outside that scale's domain.

    Parameters
    ----------
    name : str
        The level's parameter name.
    value : float
        The level on the scale the functional reads.
    scale : {"log", "odds", "logit"}
        The scale of the functional.
    estimator : {"sdr", "ipsi"}
        The estimator of the level, which the message names.
    draw : tuple of int or None, default=None
        ``(r, R)`` when ``value`` is draw ``r`` of a fit that combines ``R`` draws of the
        cross-fitting split.  The message then names the draw.

    Raises
    ------
    CapabilityError
        When ``value`` is at or below 0 on the log scale, or outside ``(0, 1)`` on the
        odds or logit scale.
    """
    cause = _RANGE_CAUSE[estimator] + "; "
    where = "" if draw is None else f" in draw {draw[0]} of {draw[1]}"
    every = (
        ""
        if draw is None
        else " in every draw, because a repeated fit combines each draw on the "
        f"{'log' if scale == 'log' else scale} scale"
    )
    if scale == "log":
        if value > 0.0:
            return
        raise CapabilityError(
            f"{name} is {value:.4g}{where}, at or below 0. {cause}the log scale needs a "
            f"positive level{every}.",
            reason=RefusalReason.INVALID_DECLARATION,
            alternative="Report the difference scale.",
        )
    if 0.0 < value < 1.0:
        return
    raise CapabilityError(
        f"{name} is {value:.4g}{where}, outside (0, 1). {cause}the {scale} scale needs a "
        f"level in (0, 1){every}.",
        reason=RefusalReason.INVALID_DECLARATION,
        alternative="Report the difference scale.",
    )


def _level_range(
    data: LongitudinalData,
    scaler: OutcomeScaler,
    q_bounds: tuple[float, float] | None,
    horizon: int,
) -> tuple[float, float]:
    """The range of the outcome that a level at ``horizon`` averages.

    A binary or survival outcome lives on ``[0, 1]``.  An end-of-study continuous outcome has
    the range of ``scaler``.  A continuous repeated outcome has the range of its own horizon:
    ``q_bounds`` when declared, else the observed range of ``Y_h`` on the rows uncensored
    through ``h``, padded as the derived scaler pads.  The scaler pools every node (W14), so
    its range would hide a level that leaves the range of ``Y_h`` and stays inside another
    node's.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.
    scaler : OutcomeScaler
        The scaler of the fit.
    q_bounds : tuple of float, optional
        The declared outcome bounds.
    horizon : int
        The horizon of the level.

    Returns
    -------
    tuple of float
        The lower and upper end of the range.
    """
    if data.family == "binomial":
        return 0.0, 1.0
    if not data.is_repeated:
        return scaler.lower, scaler.upper
    observed = data.outcome_at(horizon)[data.uncensored_through(horizon)]
    derived = OutcomeScaler.from_outcome(observed, q_bounds)
    return derived.lower, derived.upper


def _warn_parameter_space(
    estimates: Mapping[str, ParameterEstimate],
    ranges: Mapping[str, tuple[float, float]],
    *,
    estimator: str = "sdr",
) -> None:
    """Warn once when a one-step level leaves the range of the outcome it averages.

    Parameters
    ----------
    estimates : Mapping[str, ParameterEstimate]
        The reported estimates.  Only a level is checked.
    ranges : Mapping[str, tuple of float]
        The range of the outcome behind each level, by parameter name (:func:`_level_range`).
        The message names the range of each level that leaves its own.
    estimator : str
        The estimator kind, for the cause that the message gives.
    """
    outside: dict[tuple[float, float], list[str]] = {}
    for name, estimate in estimates.items():
        if estimate.scale != "level":
            continue
        low, high = ranges[name]
        if not low <= estimate.psi <= high:
            outside.setdefault((low, high), []).append(f"{name} = {estimate.psi:.4g}")
    if not outside:
        return
    described = "; ".join(
        f"{'; '.join(items)}: outside the outcome range [{low:.4g}, {high:.4g}]"
        for (low, high), items in outside.items()
    )
    _warn(
        f"{described}. {_RANGE_CAUSE[estimator]}. It is reported as computed, with no "
        "truncation. A ratio or odds functional of such a level is refused",
        ParameterSpaceWarning,
        stacklevel=3,
    )


#: The summary title of each estimator.
_ESTIMATOR_TITLE: dict[str, str] = {
    "tmle": "TMLE",
    "sdr": "SDR",
    "ipsi": "IPSI",
    "gcomp": "g-computation",
    "ipw": "IPW",
    "parametric_gformula": "parametric g-formula",
}

#: The estimators that fit no targeting step and read parametric nuisances for an interval.
_BASELINES: frozenset[str] = frozenset({"gcomp", "ipw", "parametric_gformula"})

#: Separates the simulation seed of the parametric g-formula from every other stream that
#: ``random_state`` seeds.
_SIMULATION_SEED_TAG = 0x47464D31

#: The share of a parametric g-formula curve's variance above which the fit warns that the
#: simulation error is large.
MONTE_CARLO_WARN_SHARE = 0.10

#: Match the point-treatment estimator's warning threshold.  The exact share remains
#: available in ``diagnostics()`` below this value; the warning is only the interruption.
_TRUNCATION_WARN_FRACTION = 0.05


#: What each point-treatment keyword would need before this estimator could take it.
#: The module docstring and ``docs/technical-reference/longitudinal-tmle.md`` both say these
#: are refused *by name*;
#: without
#: this table they were refused by absence, which is a ``TypeError`` naming no reason.
#: Each entry is a :class:`~cleverly._refusal.Refusal`, raised as a ``CapabilityError``.
_REFUSED: dict[str, Refusal] = {
    # Kept as a key rather than deleted now that observation weights are supported, for the
    # reason ``event`` and ``competing`` are: a weight is a *column of the data*, so it is
    # declared where the columns are read, and falling through to "unexpected keyword
    # argument" here would read as a misspelling rather than as a pointer to the call that
    # takes it.  The refusal this replaced said weights put "a further per-unit factor in
    # the clever covariate's denominator at every node", which was simply wrong: a weight
    # tilts the population, and the denominator is the 2T mechanism factors and nothing
    # else.
    "weights": Refusal(
        RefusalReason.API_REDIRECTION,
        "observation weights are a column of the data rather than a setting on the estimator.",
        alternative=(
            "Declare them where the columns are read: LTMLE(...).fit(frame, weights='w', ...), "
            "or LongitudinalData.from_frame(frame, weights='w', ...)."
        ),
    ),
    "intermediate": Refusal(
        RefusalReason.DIFFERENT_ESTIMAND,
        "a controlled direct effect fixes a mediator at one time point. Over a sequence of "
        "nodes that is a different parameter with a different identification, not a further "
        "column.",
        alternative=(
            "TMLE(..., intermediate=) estimates a controlled direct effect at one decision."
        ),
    ),
    # Kept as a key rather than deleted now that a working model over regimens is
    # supported, for the reason ``weights``, ``event`` and ``competing`` are: it is
    # declared on the *estimator* and not where the columns are read, so falling through
    # to "unexpected keyword argument" at ``.fit()`` would read as a misspelling rather
    # than as a pointer to the call that takes it.
    "msm": Refusal(
        RefusalReason.API_REDIRECTION,
        "a working model is a setting on the estimator rather than a column of the data. Its "
        "design is handed (regimen_label, horizon, baseline_frame).",
        alternative="Declare it where the regimens are: LTMLE(regimens, msm=MSM(...)).",
    ),
    "interventions": Refusal(
        RefusalReason.API_REDIRECTION,
        "a regime is a density over the arms at one node, and a regimen is a plan across "
        "nodes. So the longitudinal analogue of a rule d(W) is a regimen whose nodes are "
        "rules, not a further parameter axis.",
        alternative=(
            "Declare it in regimens=, for example regimens=[DynamicRegimen('treat once L2 "
            "rises', (0, lambda h: h['L2'] > 0), rule_kind='known')], with DynamicRegimen "
            "from cleverly.longitudinal. A node that draws its arm from a known distribution "
            "is a Stochastic node in that plan: DynamicRegimen('mix', (Stochastic(q1, 'q1', "
            "density_kind='known'), 'low'))."
        ),
    ),
    "policies": Refusal(
        RefusalReason.API_REDIRECTION,
        "a modified treatment policy is a node of a regimen rather than a parameter axis.",
        alternative=(
            "Pass regimens={'+0.5': DynamicRegimen('+0.5', (Shift(0.5, cap=4.0),) * 2)}, or a "
            "mapping value Shift(0.5, cap=4.0) for every node. Declare continuous nodes with "
            "LTMLE.fit(..., continuous_treatment=[...]) or "
            "LongitudinalData.from_frame(..., continuous_treatment=[...])."
        ),
    ),
    "incremental": LTMLE_TILT_REFUSAL,
    "delta": Refusal(
        RefusalReason.API_REDIRECTION,
        "an outcome missing for a reason other than censoring is a further node in the "
        "likelihood. Its probability must be estimated and enter the cumulative product, "
        "not be assumed one.",
        alternative="Encode it as a final censoring column.",
    ),
    # Kept as a key rather than deleted now that a survival outcome is supported:
    # dropping it would fall through to "unexpected keyword argument", which reads as a
    # misspelling rather than as a pointer to the keyword that does this.
    "event": Refusal(
        RefusalReason.API_REDIRECTION,
        "a survival outcome is declared by the outcome columns themselves, not beside them.",
        alternative=(
            "Pass one event indicator per time point, outcome=['Y1', 'Y2', ...], in the same "
            "order as treatment=. The fit then reports a cumulative risk at every horizon "
            "rather than one number."
        ),
    ),
    # Kept as a key now that competing risks are supported, for the reason ``event`` is:
    # dropping it would fall through to "unexpected keyword argument", which reads as a
    # misspelling rather than as a pointer to the keyword that does this.
    "competing": Refusal(
        RefusalReason.API_REDIRECTION,
        "competing risks are declared by the outcome columns themselves, not beside them.",
        alternative=(
            "Pass a mapping of cause to its indicator column per time point, "
            "outcome={'relapse': ['R1', 'R2'], 'death': ['D1', 'D2']}. The fit then reports "
            "a cumulative incidence per cause at every horizon."
        ),
    ),
    "eliminate": Refusal(
        RefusalReason.DIFFERENT_ESTIMAND,
        "eliminating the competing events is a different estimand, not a setting on this "
        "one. What is reported here is the cause-specific cumulative incidence with the "
        "competing causes left alone, so a competing event is part of the history and "
        "enters the clever covariate's indicator. Removing it would make it an intervened "
        "node: a further factor per node in the denominator, and its own no-unmeasured-"
        "confounding and positivity assumptions to state.",
        alternative=(
            "Declare outcome={cause: [...]} to estimate the cause-specific cumulative "
            "incidence with the competing causes left alone."
        ),
    ),
    "cross_fit": Refusal(
        RefusalReason.API_REDIRECTION,
        "the fold count is the only control over cross-fitting here, and the config says "
        "which of the two a fit used.",
        alternative="Pass n_folds=1 for in-sample nuisance fits.",
    ),
}


#: What separates a regimen from the horizon it is reported at, inside the brackets of a
#: parameter name and in the key of :attr:`LongitudinalResult.fits`.  Both are *composed*
#: from ``(label, horizon)`` rather than parsed back out of each other, so the two can
#: never drift; this constant exists so that a regimen label containing it can be refused
#: rather than producing a name nobody can read either way.
HORIZON_INFIX = " @ t="

#: What separates a regimen from the cause whose incidence is reported, on a fit that
#: declared competing risks.  Beside :data:`HORIZON_INFIX` and used the same way: composed
#: into a name, never parsed back out of one, and refused inside a label so that two
#: parameters cannot end up sharing a name.
CAUSE_INFIX = ", "


#: The parameter an outcome declaration makes a fit answer; see
#: :attr:`LongitudinalConfig.outcome_kind`.
OutcomeKind: TypeAlias = Literal["end_of_study", "survival", "competing", "repeated"]


def outcome_kind(data: LongitudinalData) -> OutcomeKind:
    """The parameter ``data``'s outcome declaration makes a fit answer.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.

    Returns
    -------
    str
        ``"competing"``, ``"survival"``, ``"repeated"`` or ``"end_of_study"``.
    """
    if data.is_competing:
        return "competing"
    if data.is_survival:
        return "survival"
    if data.is_repeated:
        return "repeated"
    return "end_of_study"


def _index(label: str, cause: str | None, horizon: int, curve: bool) -> str:
    """What goes inside the brackets of a parameter name.

    ``curve`` is :attr:`~cleverly.longitudinal.LongitudinalData.reports_curve`: a survival
    or a repeated outcome reports one parameter per horizon, and the horizon is in the name.
    """
    if not curve:
        return label
    stem = label if cause is None else f"{label}{CAUSE_INFIX}{cause}"
    return f"{stem}{HORIZON_INFIX}{horizon}"


#: Why a fit with two or more causes has no survival view.  One text for
#: :meth:`LongitudinalResult.curve` and :meth:`LongitudinalResult.ratio`.
SURVIVAL_VIEW_COMPETING_REFUSAL = Refusal(
    RefusalReason.DIFFERENT_ESTIMAND,
    "a competing-risk fit reports one cumulative incidence per cause, and 1 - one cause's "
    "incidence is not all-cause survival. Event-free survival is 1 - the sum of the "
    "cause-specific incidences.",
    alternative="Use incidence_total() to inspect that sum.",
)

#: What a repeated-outcome fit reports, in the refusals of the survival-curve functionals.
REPEATED_NOT_A_RISK = (
    "the mean of a repeated outcome at each horizon, not a cumulative risk: a repeated "
    "outcome can return to zero, so neither a survival complement, a restricted mean "
    "survival time nor an incidence total is defined on it."
)
#: What a caller can do instead, in the same refusals.
REPEATED_ALTERNATIVE = "Use contrast() for a linear functional of the means."

#: The head name of a *contrast* between two regimens, whatever the outcome is.  A
#: difference of risks and a difference of survivals are the same number up to a sign, so
#: the view does not rename it.
CONTRAST_HEAD = "ate_regimen"


def _level_head(*, survival: bool, competing: bool, complement: bool) -> str:
    """The head name of one regimen's level parameter.

    The one place the level vocabulary lives.  Two callers need it: :meth:`_estimates`,
    which composes the name a fit reports, and :meth:`LongitudinalResult.curve`, which
    composes the name the survival view reports.  Written twice it is two chances for the
    report and its view to disagree about what a parameter is called, and the second
    caller would have to recover the first's answer by matching a prefix -- which is the
    parsing this module refuses everywhere else.

    Parameters
    ----------
    survival : bool
        Whether the outcome is an event indicator at every node.
    competing : bool
        Whether the fit declared its causes, which makes the level a cause-specific
        cumulative incidence rather than a single-event risk.
    complement : bool
        Whether the reported quantity is the complement :math:`S = 1 - F` rather than the
        risk itself.  A complement is event-free survival and names itself as such.

    Returns
    -------
    str
        The head name, without the bracketed index.

    Raises
    ------
    CapabilityError
        When a complement is asked of a fit that is not a survival fit.  There is no
        cumulative risk to take the complement of, so every name this function could
        return would be the name of a different quantity.  :meth:`LongitudinalResult.curve`
        refuses that request earlier and with the fit's own outcome column in the sentence;
        this is the vocabulary itself declining to answer, so a second caller cannot get a
        risk name back for a survival request.
    """
    if not survival:
        if complement:
            raise CapabilityError(
                "a fit with one end-of-study outcome estimates a mean and not a "
                "cumulative risk, so it has no complement to name. S = 1 - F is a map "
                "on a risk curve.",
                reason=RefusalReason.DIFFERENT_ESTIMAND,
                alternative=(
                    "Pass one outcome column per time point to estimate a cumulative risk "
                    "at every horizon."
                ),
            )
        return "ey_regimen"
    if complement:
        return "survival_regimen"
    return "cif_regimen" if competing else "risk_regimen"


def _fit_key(label: str, cause: str | None, horizon: int, curve: bool) -> str:
    """The key one regimen's fit at one cause and horizon is filed under.

    The same string :func:`_index` builds, and deliberately so: a terminal fit is keyed
    by its regimen label exactly as it always was, and a survival fit by the tuple that
    indexes its parameter.
    """
    return _index(label, cause, horizon, curve)


#: The estimator kinds that fit a repeated outcome.  A kind belongs here once each horizon of
#: its fit is the end-of-study fit of the panel truncated after it, and its evidence exists.
REPEATED_OUTCOME_KINDS: frozenset[str] = frozenset({"tmle", "sdr", "ipsi"})

#: How the refusal names each estimator kind.
_REPEATED_OUTCOME_NAMES: dict[str, str] = {
    "tmle": "LTMLE",
    "sdr": "SDR",
    "ipsi": "IPSI",
    "gcomp": "SequentialGComputation",
    "ipw": "SequentialIPW",
    "parametric_gformula": "ParametricGFormula",
}


def repeated_outcome_refusal(name: str) -> Refusal:
    """The refusal of a repeated outcome by ``name``, naming the kinds that fit one.

    Parameters
    ----------
    name : str
        The class or method name of the refusing estimator.

    Returns
    -------
    Refusal
        ``REPEATED_OUTCOME_REFUSAL`` filled from ``REPEATED_OUTCOME_KINDS``.
    """
    kinds = sorted(_REPEATED_OUTCOME_NAMES.get(kind, kind) for kind in REPEATED_OUTCOME_KINDS)
    listed = kinds[0] if len(kinds) == 1 else f"{', '.join(kinds[:-1])} and {kinds[-1]}"
    verb = "s" if len(kinds) == 1 else ""
    refusal = replace(
        REPEATED_OUTCOME_REFUSAL,
        condition=REPEATED_OUTCOME_REFUSAL.condition.format(name=name),
        alternative=str(REPEATED_OUTCOME_REFUSAL.alternative).format(kinds=listed, verb=verb),
    )
    if name in ("SequentialGComputation", "SequentialIPW", "ParametricGFormula"):
        # The baselines' reduction is its own item.
        return replace(refusal, roadmap="X71")
    return refusal


def refuse_repeated_outcome(estimator: str, name: str) -> None:
    """Refuse a repeated outcome on a longitudinal estimator that has no per-horizon reduction.

    A kind in ``REPEATED_OUTCOME_KINDS`` fits a repeated outcome by one backward pass per
    horizon, each the end-of-study fit of the panel truncated after it.  A kind outside the
    set, such as :class:`SequentialIPW`, fits nothing until its own reduction and evidence exist,
    so it refuses here, before any learner, rather than fitting by inheritance.
    ``tests/unit/test_repeated_outcome_subclasses.py`` fits every concrete subclass of
    ``_SequentialEstimator`` on a repeated outcome and requires either this refusal or,
    for a kind in the set, an estimate and influence curve at each horizon equal to the
    end-of-study fit of the truncated panel.

    Parameters
    ----------
    estimator : str
        The estimator kind, ``"tmle"`` for :class:`LTMLE` itself.
    name : str
        The class name, for the message.

    Raises
    ------
    CapabilityError
        For every kind outside ``REPEATED_OUTCOME_KINDS``.
    """
    if estimator in REPEATED_OUTCOME_KINDS:
        return
    raise CapabilityError.from_refusal(repeated_outcome_refusal(name))


def refuse_unsupported(passed: Mapping[str, Any], *, where: str = "LTMLE") -> None:
    """Refuse a point-treatment keyword by name, saying what it would take to support it."""
    for name in passed:
        refusal = _REFUSED.get(name)
        if refusal is None:
            raise TypeError(
                f"{where} got an unexpected keyword argument {name!r}; see "
                "docs/user-guide/longitudinal.md for what a longitudinal fit supports"
            )
        condition = f"{name}= is not supported by a longitudinal fit: {refusal.condition}"
        raise CapabilityError.from_refusal(replace(refusal, condition=condition))


#: The one sentence a refused replay opens with.  It is a documented contract string --
#: ``docs/technical-reference/scope-and-refusals.md`` quotes it -- so it is written once
#: here and every refusal appends its stable codes to it.
_REPLAY_REFUSAL = "longitudinal truncation replay is unavailable: "


def _caught_refusal(prefix: str, error: Exception) -> Refusal:
    """The refusal of a wrapper that re-raises a caught error under a sentence of its own.

    Parameters
    ----------
    prefix : str
        The wrapper's sentence, which opens the condition.
    error : Exception
        The caught error. A cleverly error passes its reason code, alternative and roadmap
        item on, so the wrapper does not hide a code.

    Returns
    -------
    Refusal
        The inner record under the wrapper's sentence, or an
        ``INVALID_DECLARATION`` record when the caught error is not a cleverly error.
    """
    if isinstance(error, CleverlyError):
        inner = error.refusal
        return replace(inner, condition=prefix + inner.condition)
    text = str(error.args[0]) if error.args else str(error)
    return Refusal(RefusalReason.INVALID_DECLARATION, prefix + text)


def _refuse_replay(*codes: str, reason: RefusalReason) -> NoReturn:
    """Refuse a replay request, naming the stable omission codes that prevent it.

    Parameters
    ----------
    *codes : str
        Stable omission codes, in the order they are reported.
    reason : RefusalReason
        The reason code of the refusal. Every replay omission code is an
        ``INVALID_DECLARATION``.

    Raises
    ------
    CapabilityError
        Always. The prefix is the contract string documented under *Scope and refusals*.
    """
    raise CapabilityError(_REPLAY_REFUSAL + ", ".join(codes), reason=reason)


def _frozen_plans(plans: Sequence[Plan]) -> tuple[Plan, ...]:
    """Copy resolved plans so the recipe owns read-only arms nobody else can write.

    One implementation for the two moments a recipe takes ownership of its plans: when a
    fit builds it, and when a pickle restores it.  Written twice, the stored recipe and the
    fresh one differ in whether the arms a replay divides by can still be mutated in place.
    """
    retained: list[Plan] = []
    for plan in plans:
        values = np.array(plan.values, copy=True)
        values.setflags(write=False)
        policy = tuple(None if entry is None else _frozen(entry) for entry in plan.policy)
        numerators = tuple(
            None if entry is None else _frozen(entry) for entry in plan.mtp_numerators
        )
        retained.append(replace(plan, values=values, policy=policy, mtp_numerators=numerators))
    return tuple(retained)


def _component_fingerprint(data: LongitudinalData) -> tuple[FloatArray, ...]:
    """The arrays a mixed node adds to the data fingerprint, and none without one.

    A mixed node's slot in :attr:`~cleverly.longitudinal.LongitudinalData.treatment` holds its
    presence only, so its component values and labels are digested here.
    """
    if not data.has_mixed_node:
        return ()
    values = tuple(block for block in data.component_values if block is not None)
    labels = np.asarray([repr(levels) for levels in data.component_levels], dtype=str)
    return (*values, labels)


def _frozen(array: FloatArray) -> FloatArray:
    """A read-only copy of ``array``."""
    copied = np.array(array, copy=True)
    copied.setflags(write=False)
    return copied


def _plan_fingerprint(plan: Plan) -> str:
    """Digest the arms a plan assigned this sample, and its policy densities when it has any.

    A plan without a policy node digests ``values`` alone, exactly as before policies
    existed, and so does a plan whose every policy node collapsed to a rule.  A policy
    plan's ``values`` hold the observed codes, so two policies with the same support would
    otherwise share a digest.  Each policy entry is digested in node order, and
    :func:`~cleverly.provenance.fingerprint_array` hashes a ``None`` entry as a marker, so
    moving a policy to another node changes the digest.
    """
    if not plan.has_policy:
        return fingerprint_array(plan.values)
    if not plan.has_mtp:
        return fingerprint_array(plan.values, *plan.policy)
    # A modified treatment policy's carry weights do not tell two policies apart -- every
    # deterministic policy at a continuous node carries a column of ones -- so its assigned
    # doses and level maps are digested too.
    targets = [dose for doses in plan.policy_targets if doses is not None for dose in doses]
    maps = [
        codes.astype(float)
        for branches in plan.mtp_assignments
        if branches is not None
        for _, codes in branches
    ]
    return fingerprint_array(plan.values, *plan.policy, *targets, *maps)


def _truncated_cells(unbounded: FloatArray, bounded: FloatArray) -> BoolArray:
    """Which cumulative cells a bound replaced, over whatever shape the caller hands in.

    The one expression every truncation diagnostic agrees on: a cell is truncated when the
    bounded cumulative probability differs from the raw product that produced it.
    :func:`_clipped_prefix` hands in one node's column, and :func:`_consumed_prefixes` hands
    in a whole ``(n, T)`` matrix.

    Parameters
    ----------
    unbounded : FloatArray
        Raw cumulative probabilities, before ``g_bounds``.
    bounded : FloatArray
        Cumulative probabilities after ``g_bounds``, of the same shape.

    Returns
    -------
    BoolArray
        One flag per cell, true where the bound replaced the raw product.
    """
    return np.asarray(unbounded != bounded)


def _clipped_prefix(
    unbounded: FloatArray, bounded: FloatArray, time: int, rows: BoolArray
) -> BoolArray:
    """Which of ``rows`` had node ``time``'s cumulative prefix replaced by a bound.

    One comparison, in :func:`_truncated_cells`, and three diagnostics that select different
    cells from it.  :meth:`LongitudinalResult._max_truncated` and
    :meth:`LTMLE._warn_on_truncation` pass the pair the fit retained.
    :func:`_score_cell_truncation_counts` rebuilds the same out-of-fold pair at each
    requested bound.  It is the one mechanism a per-regimen recursion divides by, on a
    cross-fitted fit and a single-fold one alike.

    Parameters
    ----------
    unbounded : FloatArray
        ``(n, T)`` raw cumulative probabilities, before ``g_bounds``.
    bounded : FloatArray
        ``(n, T)`` cumulative probabilities after ``g_bounds``.
    time : int
        One-based node index.
    rows : BoolArray
        The rows to report on.

    Returns
    -------
    BoolArray
        One flag per selected row, true where the bound replaced the raw prefix.
    """
    return _truncated_cells(unbounded[:, time - 1], bounded[:, time - 1])[rows]


@dataclass(frozen=True)
class _HazardReplay:
    """The settings a hazard replay re-targets the retained initial hazards with."""

    levels: tuple[Plan, ...]
    algorithm: HazardAlgorithm
    step_size: float
    alpha: float
    max_iter: int
    tol: float


@dataclass(frozen=True)
class _LongitudinalReplayRecipe:
    """Unfitted state required to repeat the bound-dependent recursion."""

    plans: tuple[Plan, ...]
    outcome_learner: Learner | None
    pseudo_learner: Learner | None
    #: The targeting settings of a TMLE fit.  ``None`` on an SDR fit, which fits no
    #: fluctuation.
    alpha: float | None
    max_iter: int | None
    tol: float | None
    n_jobs: int
    omissions: tuple[str, ...] = ()
    #: Whether the fit projected each curve with ``isotonic=True``.
    isotonic: bool = False
    #: The hazard targeting settings, ``None`` on an iterated-mean fit.  A hazard replay
    #: re-targets the retained initial hazards and refits no learner.
    hazard: _HazardReplay | None = None
    #: ``"tmle"``, ``"sdr"``, ``"ipsi"``, ``"gcomp"`` or ``"ipw"``: which recursion a replay
    #: repeats.
    estimator: str = "tmle"
    #: The stopping rule of a TMLE fit's node solves, ``"relative"`` unless the fit declared one.
    stopping: str = "relative"
    #: The step bound of a TMLE fit's node solves, ``None`` unless the fit declared one.
    step_bound: float | None = None
    #: Whether a bounded step searches its length.
    step_search: bool = True
    #: Which iterate of each node's walk a TMLE fit reports: ``"last"``, or ``"best"``.
    iterate: str = "last"
    #: The IPW normalization, ``None`` on every other estimator.
    normalization: str | None = None
    #: Whether every regression a baseline estimator reads is a GLM, which decides its
    #: status.  Always ``True`` on a targeted fit, which does not read it.
    parametric: bool = True
    #: The resolved node bounds of a TMLE fit by cause, ``None`` unless the fit declared any.
    #: A replay refits the bounded nodes with the same bounds.
    node_bounds: dict[str | None, dict[int, NodeBound]] | None = None

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore a persisted recipe, then retake ownership of its plan matrices.

        Pickle restores each plan matrix as a fresh writeable array, so the recipe freezes
        them again, as it does at construction.

        Parameters
        ----------
        state : dict of str to Any
            The instance dictionary the pickle carries.
        """
        self.__dict__.update(state)
        object.__setattr__(self, "plans", _frozen_plans(self.plans))
        if self.hazard is not None:
            object.__setattr__(
                self, "hazard", replace(self.hazard, levels=_frozen_plans(self.hazard.levels))
            )


def _replay_random_state_omissions(learners: Sequence[Learner]) -> tuple[str, ...]:
    """Return stable reasons the recursive learner templates cannot replay exactly."""

    omissions: list[str] = []

    def classify(value: Any) -> None:
        if value is None:
            omissions.append(LONGITUDINAL_REPLAY_RANDOM_STATE_UNSEEDED)
        elif isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            omissions.append(LONGITUDINAL_REPLAY_RANDOM_STATE_NON_INTEGER)

    seen: set[int] = set()

    def visit(value: Any) -> None:
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, Mapping):
            for item in value.values():
                visit(item)
            return
        if isinstance(value, (tuple, list)):
            for item in value:
                # Named Super Learner candidates are ``(name, estimator)`` pairs; the
                # string is harmless and the estimator is reached on the next item.
                visit(item)
            return
        get_params = getattr(value, "get_params", None)
        if not callable(get_params):
            return
        for name, item in get_params(deep=False).items():
            if name == "random_state":
                classify(item)
            else:
                visit(item)

    for learner in learners:
        visit(learner)
    return tuple(dict.fromkeys(omissions))


def _replay_recipe(
    plans: Sequence[Plan],
    outcome_learner: Learner,
    pseudo_learner: Learner,
    *,
    alpha: float | None,
    max_iter: int | None,
    tol: float | None,
    n_jobs: int,
    estimator: str = "tmle",
    stopping: str = "relative",
    step_bound: float | None = None,
    step_search: bool = True,
    iterate: str = "last",
    normalization: str | None = None,
    node_bounds: dict[str | None, dict[int, NodeBound]] | None = None,
) -> _LongitudinalReplayRecipe:
    """Freeze resolved plans and clone the unfitted recursive learner specifications.

    No ``random_state``: the recursion takes none.  What makes a replay deterministic is
    the cloned learner templates, which :func:`_replay_random_state_omissions` audits, and
    the fit's own seed is already recorded on
    :attr:`LongitudinalConfig.random_state` for the reader who wants it.
    """

    learners: list[Learner | None] = []
    omissions: list[str] = []
    for learner in (outcome_learner, pseudo_learner):
        try:
            learners.append(clone(learner))
        except Exception:
            learners.append(None)
            omissions.append(LONGITUDINAL_REPLAY_LEARNER_UNCLONABLE)
    omissions.extend(
        _replay_random_state_omissions([item for item in learners if item is not None])
    )
    return _LongitudinalReplayRecipe(
        plans=_frozen_plans(plans),
        outcome_learner=learners[0],
        pseudo_learner=learners[1],
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        n_jobs=n_jobs,
        omissions=tuple(dict.fromkeys(omissions)),
        estimator=estimator,
        stopping=stopping,
        step_bound=step_bound,
        step_search=step_search,
        iterate=iterate,
        normalization=normalization,
        node_bounds=node_bounds,
    )


def _known_factors(data: LongitudinalData) -> tuple[str, ...]:
    """The treatment and censoring columns whose factor ``data`` declares known."""
    known = data.known_mechanisms
    if known is None:
        return ()
    names: list[str] = []
    for time in range(1, data.n_times + 1):
        if not data.is_held_node(time) and known.treatment_at(time) is not None:
            names.append(data.decision_name(time))
        if data.censoring_names and known.censoring_at(time) is not None:
            names.append(data.censoring_names[time - 1])
    return tuple(names)


@dataclass(frozen=True)
class LongitudinalConfig:
    """A snapshot of the settings a longitudinal fit actually used."""

    family: str
    n_times: int
    #: The outcome column(s): one name for an end-of-study outcome, one per node for a
    #: survival one.  Which it is decides what the fit reports, so it belongs in the
    #: record of what the fit did rather than being recoverable only from the names.
    outcome_names: tuple[str, ...]
    #: The horizons reported, ``(T,)`` on an end-of-study fit.
    horizons: tuple[int, ...]
    #: The absorbing causes reported, empty unless the fit declared competing risks.
    #: Beside :attr:`outcome_names` for the same reason: which of the three parameters a
    #: fit answers is a statement it made, and belongs in the record of what it did.
    causes: tuple[str, ...]
    regimens: tuple[RegimenSpec, ...]
    reference: str
    n_folds: int
    g_bounds: tuple[float, float]
    q_bounds: tuple[float, float] | None
    alpha_sig: float
    random_state: int | None = None
    #: ``(label, digest)`` per regimen, digesting the ``(n, T)`` arms it assigned *this*
    #: sample.  A static plan is already stated in full by its ``1/0``; a rule is not, and
    #: two different rules would otherwise report identically and carry an identical
    #: provenance record -- ``1{L2 > 0}`` and ``1{L2 > 5}`` are different parameters.
    #: Digesting the resolved matrix rather than the callable is what makes this possible
    #: at all: a closure has no stable fingerprint, and the arms are what the fit used.
    plan_fingerprints: tuple[tuple[str, str], ...] = ()
    #: The working model's terms and link, ``None`` on a fit that declared none.
    msm_terms: tuple[str, ...] | None = None
    msm_link: str | None = None
    #: A digest of the *evaluated* design and weights.  A design is a closure and has no
    #: stable fingerprint; the arrays are what the fit was handed, which is the same
    #: reasoning :attr:`plan_fingerprints` rests on.
    msm_fingerprint: str | None = None
    #: ``(label, nodes)`` per regimen with a policy node whose density was one-hot on every
    #: reachable row.  Such a node was resolved as the rule it equals, and the fit ran that
    #: rule's code.  Empty when no policy node collapsed.
    policy_point_mass_nodes: tuple[tuple[str, tuple[int, ...]], ...] = ()
    #: The column of the one baseline decision a held design carries over every node, and
    #: ``None`` on a design with one decision per node.
    held_treatment: str | None = None
    #: The grid times of a container built from a time and an event column, else ``None``.
    time_grid: tuple[float, ...] | None = None
    #: The settings line of a container built from a long panel
    #: (:meth:`~cleverly.longitudinal.PanelLayout.describe`), else ``None``.
    panel: str | None = None
    #: The treatment and censoring columns whose factor the data declares known
    #: (:attr:`~cleverly.longitudinal.LongitudinalData.known_mechanisms`), in node order.
    #: Each fits no learner and is not truncated.  Empty on a fit that declares none.
    known_factors: tuple[str, ...] = ()
    #: How a survival curve is modelled: ``"iterated_mean"`` (one backward recursion per
    #: horizon) or ``"hazard"`` (targeted cause-specific hazards).
    survival_model: str = "iterated_mean"
    #: The hazard targeting algorithm: ``"iterative"`` (per horizon) or ``"one_step"``.
    targeting: str = "iterative"
    #: The first step of a one-step walk.  Recorded, and unused, under ``"iterative"``.
    step_size: float = 1e-3
    #: Whether each curve's reported levels and band limits were projected onto
    #: non-decreasing curves.
    isotonic: bool = False
    #: The declared node bounds on the reported outcome scale, as
    #: ``(cause, grid time, lower, upper)`` in declaration order.  ``cause`` is ``None`` unless
    #: the fit declared competing risks.  Empty on a fit that declared none.
    node_bounds: tuple[tuple[str | None, float, float, float], ...] = ()
    #: ``"tmle"`` for :class:`LTMLE`, ``"sdr"`` for :class:`SDR`, ``"ipsi"`` for
    #: :class:`IPSI`, ``"gcomp"`` for :class:`SequentialGComputation`, ``"ipw"`` for
    #: :class:`SequentialIPW` and ``"parametric_gformula"`` for :class:`ParametricGFormula`.
    estimator: str = "tmle"
    #: How many independent draws of the cross-fitting split the fit combined by their
    #: median.  ``1`` is an ordinary fit.
    repeats: int = 1
    #: The seed of every multiplier draw behind the simultaneous band.  ``random_state``
    #: when that is an integer, a fresh seed recorded here when ``random_state=None`` and
    #: ``repeats > 1``, and ``None`` otherwise.
    multiplier_seed: int | None = None
    #: The IPW normalization, ``"hajek"`` or ``"horvitz_thompson"``, and ``None`` on every
    #: other estimator.
    normalization: str | None = None

    #: Which parameter the outcome declaration made the fit answer: ``"end_of_study"`` (a
    #: mean), ``"survival"`` (a risk curve), ``"competing"`` (an incidence curve per cause)
    #: or ``"repeated"`` (a curve of means).  The report reads this rather than counting
    #: :attr:`outcome_names`, because a survival and a repeated outcome both name a column
    #: per node.
    outcome_kind: OutcomeKind = "end_of_study"

    #: The support key the fit was classified under, read before any learner ran.
    support_key: SupportKey | None = None

    def describe(self, *, contrast: bool) -> list[str]:
        """Return the settings lines of :meth:`LongitudinalResult.summary`.

        Parameters
        ----------
        contrast : bool
            Whether the report holds a contrast, which the reference line describes. A
            report of regime means prints no reference line.

        Returns
        -------
        list of str
            One line per setting.
        """
        plans = ", ".join(
            f"{regimen.label}=({describe_plan(regimen)})" for regimen in self.regimens
        )
        lines = [
            f"time points: {self.n_times}",
            f"outcome family: {self.family}",
            f"regimens: {plans}",
        ]
        if self.estimator == "sdr":
            lines.insert(
                0,
                "estimator: sequentially doubly robust (Diaz, Williams, Hoffman and Schenck "
                "2023, Section 5.3), no fluctuation",
            )
        if self.estimator == "ipsi":
            lines.insert(
                0,
                "estimator: Kennedy (2019) Algorithm 1, the mean of the influence function "
                "at the fitted nuisances (Kim, Kennedy and Naimi 2021, Algorithm 1, under "
                "censoring), no fluctuation",
            )
        if self.estimator == "gcomp":
            lines.insert(
                0,
                "estimator: sequential g-computation, the iterated conditional expectation "
                "plug-in (Bang and Robins 2005, Section 4), no fluctuation; an interval "
                "assumes every outcome GLM is correctly specified",
            )
        if self.estimator == "parametric_gformula":
            lines.insert(
                0,
                "estimator: parametric g-formula, forward simulation of the fitted covariate, "
                "outcome and hazard laws (Robins 1986; McGrath et al. 2020), no fluctuation; "
                "an interval assumes every model is correctly specified",
            )
        if self.estimator == "ipw":
            form = (
                "Hajek (normalized by the weight sum)"
                if self.normalization == "hajek"
                else "Horvitz-Thompson (normalized by n)"
            )
            lines.insert(
                0,
                f"estimator: inverse probability weighting, {form}; an interval assumes "
                "every treatment and censoring GLM is correctly specified",
            )
        if self.held_treatment is not None:
            lines.insert(
                1,
                f"treatment: one baseline decision {self.held_treatment!r}, held over "
                f"{self.n_times} node(s)",
            )
        if self.time_grid is not None:
            lines.insert(1, f"time grid: {', '.join(f'{g:g}' for g in self.time_grid)}")
        if self.panel is not None:
            lines.insert(1, self.panel)
        if self.outcome_kind != "end_of_study":
            lines.insert(
                1,
                (
                    f"outcome: repeated, a non-absorbing outcome at {', '.join(self.outcome_names)}"
                    if self.outcome_kind == "repeated"
                    else f"outcome: survival, event indicator at {', '.join(self.outcome_names)}"
                ),
            )
            reported = ", ".join(str(horizon) for horizon in self.horizons)
            lines.insert(2, f"horizons reported: t = {reported}")
            if self.causes:
                lines[1] = (
                    "outcome: competing risks, absorbing causes "
                    f"{', '.join(self.causes)} at {', '.join(self.outcome_names)}"
                )
                lines.insert(3, f"causes reported: {', '.join(self.causes)}")
        dynamic = {
            regimen.label
            for regimen in self.regimens
            if isinstance(regimen, DynamicRegimen)
            and (regimen.has_rule or regimen.has_policy or not regimen.has_tilt)
        }
        if any(
            isinstance(regimen, DynamicRegimen) and regimen.has_tilt for regimen in self.regimens
        ):
            lines.append(
                "  an 'o' node is an odds tilt: each unit draws its arm from "
                "q_t = delta*pi_t/(delta*pi_t + 1 - pi_t), with pi_t the treatment mechanism "
                "(Kennedy 2019, Equation (1)); every unit at risk is in the plan"
            )
        if dynamic:
            # Worth a line rather than leaving "d" to be guessed at, because it changes
            # how the follower counts below should be read: a static regimen's followers
            # are whoever happened to receive that sequence, and a rule's are whoever the
            # rule would have treated -- a set this sample determines.
            # A repeated outcome's earlier nodes open the later blocks, so they are in H_t.
            history = (
                "L_1, ..., L_t, with Y_{t-1} first in each block L_t, t > 1"
                if self.outcome_kind == "repeated"
                else "L_1, ..., L_t"
            )
            lines.append(
                f"  a 'd' is a rule d_t(H_t) read off [W, {history}]; its followers "
                "are a covariate-dependent set, so the counts below describe this sample"
            )
            # Only for the regimens that have a rule in them, and only when one does:
            # printing a digest of a plan the line above already spells out in full would
            # add noise to every static report to say nothing.
            for label, digest in self.plan_fingerprints:
                if label in dynamic:
                    lines.append(f"  assigned arms, {label}: {digest}")
        policies = [
            regimen.label
            for regimen in self.regimens
            if isinstance(regimen, DynamicRegimen) and regimen.has_policy
        ]
        if policies:
            lines.append(
                "  a 'q' is a known policy q_t(a | H_t) read off [W, L_1, ..., L_t] and the "
                "earlier treatments; each unit draws its arm there, and the plan's units are "
                "those whose observed arm the policy can draw"
            )
        for label, nodes in self.policy_point_mass_nodes:
            listed = ", ".join(str(node) for node in nodes)
            lines.append(
                f"  policy {label}: one-hot at node(s) {listed}, so resolved as the rule it equals"
            )
        if self.msm_terms is not None:
            # A working model has no reference regimen -- what the intercept is against is
            # whatever the design makes it -- so the working model stands where that line
            # would, rather than beside a field that names nothing.
            lines.append(
                f"working model: {len(self.msm_terms)} term(s) "
                f"{', '.join(self.msm_terms)}, link={self.msm_link}"
            )
        elif contrast:
            # A report of means holds no contrast for the reference to describe. The field
            # stays set, because a replay of the result reads it.
            lines.append(f"reference: {self.reference}")
        lines += [
            # A single fold is not cross-fitting, and printing "1 fold(s)" reads as
            # though it were: the nuisances are then fitted on the rows they predict
            # for, which the reported variance does not account for.
            "cross-fitting: none; nuisances fitted in sample"
            if self.n_folds <= 1
            else f"cross-fitting: {self.n_folds} fold(s)"
            + (f", median over {self.repeats} draws" if self.repeats > 1 else ""),
            # Both factors enter the product before its prefix is bounded, as in ltmle's
            # CalcCumG.  Saying "mechanism at every node" used to conceal the order.
            f"g_bounds: fixed [{self.g_bounds[0]:.4g}, {self.g_bounds[1]:.4g}] on each "
            + (
                "partial path treatment-and-censoring probability"
                if self.estimator == "sdr"
                else "cumulative censoring and non-tilt treatment probability; the tilt "
                "factor is unbounded"
                if self.estimator == "ipsi"
                else "cumulative treatment-and-censoring probability"
            )
            + (
                " (package default, an R ltmle-compatible heuristic; "
                "inspect the truncation diagnostics)"
                if self.g_bounds == DEFAULT_LTMLE_G_BOUNDS
                else ""
            ),
        ]
        if len(self.outcome_names) > 1:
            lines.append(f"survival model: {self.survival_model}")
            lines.append(
                f"targeting: {self.targeting}"
                + (f", step size {self.step_size:g}" if self.targeting == "one_step" else "")
            )
            lines.append(
                "isotonic projection: "
                + ("each curve's levels and band limits" if self.isotonic else "none")
            )
        if self.known_factors:
            lines.append(
                f"known factors (declared, no learner): {', '.join(self.known_factors)}; no "
                "truncation bound moved a declared value"
            )
        if self.node_bounds:
            lines.append(
                "node_bounds: "
                + "; ".join(
                    f"{'' if cause is None else f'{cause} '}t={time:g} in [{lower:g}, {upper:g}]"
                    for cause, time, lower, upper in self.node_bounds
                )
                + (
                    " (bounded cause-specific hazards at the named nodes, on the hazard scale)"
                    if self.survival_model == "hazard"
                    else " (bounded logistic fluctuation at the named nodes)"
                )
            )
        if self.q_bounds is not None:
            lines.append(f"q_bounds: [{self.q_bounds[0]:.4g}, {self.q_bounds[1]:.4g}]")
        lines.append(f"confidence level: {(1 - self.alpha_sig) * 100:g}%")
        if self.random_state is not None:
            lines.append(f"random_state: {self.random_state}")
        elif self.multiplier_seed is not None:
            lines.append(f"multiplier seed: {self.multiplier_seed} (drawn, random_state=None)")
        return lines


@dataclass(frozen=True)
class LongitudinalRepeat:
    """One draw of the cross-fitting split of a longitudinal fit, and what it produced.

    A fit with ``repeats=R`` holds ``R`` of these in :attr:`LongitudinalResult.repeats`.
    Each is the complete single-draw fit at its own fold assignment and seed, so draw ``r``
    equals a fit with ``repeats=1`` at ``random_state=seed``.

    Parameters
    ----------
    folds : Folds
        The realized outer-fold assignment of the draw.
    seed : int or None
        The seed of the draw's split.  On a fit with ``repeats > 1`` every learner of the
        draw receives it too.  ``None`` only on an in-sample fit.
    fits : dict of str to RegimenFit
        The draw's sequential fits by regimen, cause and horizon.
    mechanism : Mechanism
        The draw's fitted treatment and observation mechanisms.
    estimates : dict of str to ParameterEstimate
        Every estimate the draw reports, with its influence curve and its own status.
    replay_recipe : object
        The draw's unfitted learner templates and resolved plans, for a truncation replay.
    msm_fits : tuple of MSMRegimenFit
        The draw's projection fits by cause, empty without a working model.
    fold_mechanisms : tuple of FoldMechanism
        The draw's per-fold mechanisms on a cross-fitted SDR or IPSI fit, empty otherwise.
    isotonic_moves : tuple of tuple
        The levels an ``isotonic=True`` projection moved, as
        ``(regimen, cause, horizon, before, after)``.
    initial_hazards : InitialHazards or None
        The initial cause-specific hazards of a ``survival_model="hazard"`` fit.
    """

    folds: Folds
    seed: int | None
    fits: dict[str, RegimenFit]
    mechanism: Mechanism
    estimates: dict[str, ParameterEstimate]
    replay_recipe: _LongitudinalReplayRecipe
    msm_fits: tuple[MSMRegimenFit, ...] = ()
    fold_mechanisms: tuple[FoldMechanism, ...] = ()
    isotonic_moves: tuple[tuple[str, str | None, int, float, float], ...] = ()
    initial_hazards: InitialHazards | None = None


class _DrawOutcome(NamedTuple):
    """What :meth:`_SequentialEstimator._fit_draw` hands back to :meth:`fit`."""

    draw: LongitudinalRepeat
    #: The parameter index of the draw, ``None`` when the fit has no such index.
    parameter_index: dict[str, tuple[str, str | None, int]] | None
    #: The plans with any policy numerator attached.
    plans: tuple[Plan, ...]
    #: The fit status the draw stamped on its estimates.
    status: InferenceStatus
    #: The estimates before an isotonic projection moved a level, ``None`` when none moved.
    unprojected: dict[str, ParameterEstimate] | None
    #: The names, estimates and contributing fits of the draw.
    reported: _Reported
    #: The outcome, pseudo-outcome, treatment and censoring learners the draw resolved.
    learners: tuple[Learner, Learner, Learner, Learner]
    #: Whether the draw fitted the hazard model.
    hazard: bool


@dataclass(frozen=True)
class LongitudinalResult(Mapping[str, ParameterEstimate]):
    """Store estimates and fitted artifacts for longitudinal regimens.

    Behaves as a mapping from parameter name to
    :class:`~cleverly.inference.ParameterEstimate`, so ``result["ey_regimen[always]"]``
    and ``for name in result`` both work.

    Parameters
    ----------
    estimates : dict of str to ParameterEstimate
        Estimates keyed by stable parameter alias. Each declares the fit status, which
        :attr:`inference_status` returns, or a parameter status such as
        ``"constant_node_plugin"``.
    data : LongitudinalData
        Validated time-ordered study data.
    config : LongitudinalConfig
        Normalized sequential-estimator configuration.
    scaler : OutcomeScaler
        Transformation between observed and targeting scales.
    provenance : Provenance
        Runtime, dependency, and data fingerprints.  The fold fingerprint covers every
        draw of the split, in order.
    simultaneous : SimultaneousBands or None
        Joint confidence bands. ``None`` on a fit whose status supplies no inference.
    parameter_index : dict or None
        Structured regimen, cause, and horizon index.
    msm : RegimenMSM or None
        Evaluated working model. Its ``functions_kind`` records whether the source design
        and projection weights passed their declaration checks.
    identified_effect : IdentifiedEffect or None
        Causal question and identifying assumptions.
    method : EstimationMethod or None
        Typed public method configuration.
    parameter_keys : dict of str to ParameterKey
        Structured identities for reported aliases.
    fitted_method : str
        Method identity stamped by the estimator.
    bootstrap : BootstrapResult or None
        The full-refit bootstrap draws, when ``n_bootstrap > 0`` was requested.
    repeats : tuple of LongitudinalRepeat
        One complete fit per draw of the cross-fitting split, in fit order.  One draw on
        an ordinary fit.  With two or more, :attr:`estimates` holds their median, and
        :meth:`contrast`, :meth:`ratio`, :meth:`rmst`, :meth:`rmtl`,
        :meth:`incidence_total` and :meth:`covariance` compute in every draw and combine.
    bootstrap_band : BootstrapBand or None
        The studentized bootstrap sup-t band over the estimates that supply inference.
        ``None`` without a bootstrap, with ``simultaneous=False``, or with fewer than two
        such estimates.
    simulation : GFormulaReport or None
        On a :class:`ParametricGFormula` fit, what the forward simulation drew: the draws, the
        models, the Monte Carlo share of each parameter and the fixed-dose model reliance.
        ``None`` on every other fit.  Keyword-only.
    estimator : object or None
        The estimator settings the fit ran under, with no bootstrap and no simultaneous
        band, which a refutation refits.  ``None`` on a result built by hand.  Keyword-only.
    fit_warnings : tuple of FitWarning
        The package warnings the fit raised, in first-seen order. Set by the fit entry
        point.  Keyword-only.

    Attributes
    ----------
    assessment_cache : dict
        Saved diagnostic and sensitivity outputs. Not a constructor argument, so every
        constructed result owns its own cache. Persistence restores the mapping without
        the constructor.
    inference_status : str
    fits : dict of str to RegimenFit
    mechanism : Mechanism
    folds : Folds
    msm_fits : tuple of MSMRegimenFit
    fold_mechanisms : tuple of FoldMechanism
    replay_recipe : object
    isotonic_moves : tuple of tuple
    initial_hazards : InitialHazards or None
    n_repeats : int
    split_plan : SplitPlan

    See Also
    --------
    cleverly.estimators.TMLEResult : The same contract for a point-treatment fit.
    cleverly.RegimeContrast : The estimand a study declares to get this contrast.
    cleverly.LongitudinalTreatment : The design that names these nodes.
    LongitudinalResult.ratio : The risk ratio or odds ratio of two regimens.
    LongitudinalResult.rmst : The restricted mean survival time of a regimen.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression, LogisticRegression
    >>> from cleverly.datasets import make_longitudinal
    >>> from cleverly.longitudinal import LTMLE
    >>> frame, _ = make_longitudinal(n=200, seed=0)
    >>> result = LTMLE(
    ...     {"always": 1, "never": 0},
    ...     n_folds=2,
    ...     random_state=0,
    ...     outcome_learner=LinearRegression(),
    ...     treatment_learner=LogisticRegression(max_iter=1000),
    ...     censoring_learner=LogisticRegression(max_iter=1000),
    ... ).fit(
    ...     frame,
    ...     outcome="Y",
    ...     treatment=["A1", "A2"],
    ...     baseline=["W1", "W2"],
    ...     time_varying=[[], ["L2"]],
    ...     censoring=["C1", "C2"],
    ... )

    The regimen means and their contrast are keyed by the labels the regimens were
    declared under:

    >>> sorted(result.estimates)
    ['ate_regimen[never vs always]', 'ey_regimen[always]', 'ey_regimen[never]']
    >>> low, high = result["ey_regimen[always]"].ci
    >>> low < high
    True
    """

    estimates: dict[str, ParameterEstimate]
    data: LongitudinalData
    config: LongitudinalConfig
    scaler: OutcomeScaler
    provenance: Provenance
    simultaneous: SimultaneousBands | None = None
    #: ``name -> (regimen, cause, horizon)`` for every reported parameter, composed when
    #: the name was built.  Empty on an end-of-study fit, which has neither index.
    #: ``None`` on a working-model fit, whose parameters are indexed by *term*.
    parameter_index: dict[str, tuple[str, str | None, int]] | None = None
    #: The working model this fit projected onto, ``None`` if it declared none.
    msm: RegimenMSM | None = None
    #: Causal-workflow metadata, present on every result created through CausalStudy.
    identified_effect: Any = None
    method: Any = None
    parameter_keys: dict[str, Any] = field(default_factory=dict)
    fitted_method: str = "tmle"
    #: Assessment results keyed by operation and normalized arguments. Persistence writes
    #: this alongside the sequential artifacts it was computed from.
    #:
    #: ``init=False`` for the reason :attr:`cleverly.estimators.TMLEResult
    #: .assessment_cache` gives: the key names the operation and its arguments and not
    #: the result, so a mapping handed to the constructor by ``dataclasses.replace``
    #: would answer for a fit that never produced it.
    assessment_cache: dict[str, Any] = field(default_factory=dict, init=False)
    #: The full-refit bootstrap draws, ``None`` unless ``n_bootstrap > 0``.  A replayed
    #: fit never reruns the bootstrap.
    bootstrap: BootstrapResult | None = None
    #: On a :class:`ParametricGFormula` fit, what the forward simulation drew: the draws, the
    #: models, the Monte Carlo share of each parameter and the fixed-dose model reliance.
    #: ``None`` on every other fit.
    simulation: GFormulaReport | None = field(default=None, kw_only=True)
    #: The estimator settings the fit ran under, which :meth:`_SequentialEstimator.refit`
    #: runs again on a changed container.  ``None`` on a result built by hand.
    estimator: Any = field(default=None, kw_only=True, compare=False, repr=False)
    #: One complete single-draw fit per draw of the cross-fitting split.
    repeats: tuple[LongitudinalRepeat, ...] = field(kw_only=True)
    #: The studentized bootstrap sup-t band over the estimates that supply inference, when
    #: ``n_bootstrap > 0`` and ``simultaneous=True`` and two or more estimates supply it.
    #: Its ``support`` says whether a registered row lists the ``bootstrap_band`` kind.
    bootstrap_band: BootstrapBand | None = None
    #: The warnings the fit raised (:mod:`cleverly._warnings`), so a summary printed later
    #: still names them.
    fit_warnings: tuple[FitWarning, ...] = field(default=(), kw_only=True)
    #: Which family of assessment declarations applies to this result.  See
    #: :attr:`~cleverly.estimators.TMLEResult.assessment_family`; the method identity is
    #: :attr:`fitted_method`.
    assessment_family: ClassVar[str] = "longitudinal"

    # --------------------------------------------------------------- repeats

    @property
    def fits(self) -> dict[str, RegimenFit]:
        """The first draw's sequential fits by regimen, cause, and horizon.

        On a repeated fit this is draw 1 of ``R`` and nothing else: it describes one fitted
        recursion, and no reported number is computed from it alone.  :attr:`repeats`
        holds every draw.
        """
        return self.repeats[0].fits

    @property
    def mechanism(self) -> Mechanism:
        """The first draw's fitted treatment and observation mechanisms."""
        return self.repeats[0].mechanism

    @property
    def folds(self) -> Folds:
        """The first draw's realized outer-fold assignment."""
        return self.repeats[0].folds

    @property
    def msm_fits(self) -> tuple[MSMRegimenFit, ...]:
        """The first draw's projection fits by cause, empty without a working model."""
        return self.repeats[0].msm_fits

    @property
    def fold_mechanisms(self) -> tuple[FoldMechanism, ...]:
        """The first draw's per-fold mechanisms on a cross-fitted SDR or IPSI fit."""
        return self.repeats[0].fold_mechanisms

    @property
    def replay_recipe(self) -> _LongitudinalReplayRecipe:
        """The first draw's unfitted templates and resolved plans."""
        return self.repeats[0].replay_recipe

    @property
    def isotonic_moves(self) -> tuple[tuple[str, str | None, int, float, float], ...]:
        """The levels the isotonic projection moved, empty when it moved none."""
        return self.repeats[0].isotonic_moves

    @property
    def initial_hazards(self) -> InitialHazards | None:
        """The initial hazards of a ``survival_model="hazard"`` fit, else ``None``."""
        return self.repeats[0].initial_hazards

    @property
    def n_repeats(self) -> int:
        """How many draws of the cross-fitting split this fit combined."""
        return len(self.repeats)

    @property
    def split_plan(self) -> SplitPlan:
        """The fold assignment of every draw, bound to the rows that produced it.

        Its :attr:`~cleverly.SplitPlan.fingerprint` equals
        ``provenance.fold_fingerprint``.  A longitudinal fit records the plan and accepts
        none back.
        """
        return SplitPlan.from_folds(
            (repeat.folds for repeat in self.repeats),
            source_fingerprint=self.provenance.data_fingerprint,
        )

    def repeat_spread(self) -> dict[str, float]:
        """Split spread of each estimate across the cross-fitting draws.

        A diagnostic and not a standard error, read as
        :meth:`cleverly.estimators.TMLEResult.repeat_spread` is read: the standard deviation
        of the draws' estimates on the inference scale, the log scale for a ratio.  A draw
        that is not finite, or a ratio draw that is not positive, gives ``nan``.

        Returns
        -------
        dict of str to float
            The standard deviation across the draws, per reported estimand.

        Raises
        ------
        ValueError
            On a fit with one draw.
        """
        return repeat_spread(
            self.estimates,
            [
                {name: estimate.psi for name, estimate in repeat.estimates.items()}
                for repeat in self.repeats
            ],
        )

    def _draw_estimates(self) -> list[dict[str, ParameterEstimate]]:
        """Each draw's estimates of the names this result reports."""
        return [
            {name: repeat.estimates[name] for name in self.estimates} for repeat in self.repeats
        ]

    # ------------------------------------------------------------- mapping API

    def __getitem__(self, name: str) -> ParameterEstimate:
        try:
            return self.estimates[name]
        except KeyError:
            raise KeyError(
                f"{name!r} was not estimated; this fit reports {list(self.estimates)}"
            ) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self.estimates)

    def __len__(self) -> int:
        return len(self.estimates)

    # -------------------------------------------------------------- inference

    @property
    def estimate(self) -> ParameterEstimate:
        """The sole estimate, refusing ambiguity on a multi-parameter result."""
        return sole_estimate(self.estimates)

    @property
    def inference_status(self) -> InferenceStatus:
        """The inference status of the fit.

        ``LTMLE.fit`` stamps one fit status on every estimate, and a parameter status
        (:data:`~cleverly._inference_status.PARAMETER_STATUSES`) on the parameters it
        applies to.  This property is the fit status, through
        :func:`~cleverly.inference.results.reported_status`, which leaves the parameter
        statuses out unless every estimate carries one.  So it does not say whether one
        estimate supplies inference: read that estimate's ``supplies_inference``.
        ``"influence_curve"`` when the fit reports no estimate, because such a fit refuses
        nothing.  A clustered fit with few clusters takes ``"few_cluster_plugin"``, as a
        point-treatment fit does.
        """
        return reported_status(self.estimates)

    @property
    def n(self) -> int:
        """Return the number of observations."""
        return self.data.n

    def psi(self, name: str | None = None) -> float:
        """Return a point estimate by parameter alias.

        Parameters
        ----------
        name : str or None
            Parameter alias. ``None`` is allowed only when the fit reported one.

        Returns
        -------
        float
            The point estimate for that parameter.
        """
        return self.estimate.psi if name is None else self[name].psi

    @property
    def influence_curves(self) -> dict[str, FloatArray]:
        """Return influence curves by parameter alias.

        The curves of :attr:`estimates`.  On a fit with ``repeats > 1`` each is the curve of
        the draw at the median point, or the mean of the two central draws' curves at an
        even draw count.  It serves marginal diagnostics.  The reported variance is the
        split-adjusted median rule and is not that curve's empirical variance.
        """
        return estimate_curves(self.estimates)

    def covariance(self, names: Sequence[str] | None = None) -> FloatArray:
        """Return joint covariance for selected estimates.

        On a fit with one draw it is the covariance of the reported curves.  On a fit with
        ``repeats > 1`` it is the matrix rule of Chernozhukov et al. (2018, equation 3.14)
        for the selected vector, on the estimates' inference scales: each draw's covariance
        plus the outer product of its displacement from the coordinatewise median, and the
        draw whose matrix has the median operator norm.  At an even draw count it is the
        mean of the two matrices with the central operator norms.  That matrix depends on
        the selection, so an entry can differ between two selections and from the reported
        variance.  A one-name selection is the reported variance exactly.  The reported
        intervals, contrasts and bands read the per-name rule, not this matrix.

        Parameters
        ----------
        names : sequence of str or None
            Aliases in output order. ``None`` selects all estimates.

        Returns
        -------
        ndarray
            Covariance matrix at the independent observation or cluster level. On a fit
            whose status supplies no inference, or a selection with a name that supplies
            none, it is a diagnostic: the plug-in covariance of the reported curves on a
            single-draw fit, and the matrix rule over the per-draw plug-in candidates on a
            repeated fit.
        """
        if self.n_repeats > 1:
            chosen = select_estimates(self.estimates, names)
            return median_covariance(self._draw_estimates(), chosen, cluster=self.data.cluster)
        return estimate_covariance(self.estimates, names, cluster=self.data.cluster)

    def contrast(
        self,
        function: Callable[[FloatArray], float],
        names: Sequence[str],
        *,
        name: str | None = None,
        scale: Scale = "difference",
        gradient: Callable[[FloatArray], FloatArray] | None = None,
        transform: Transform | None = None,
    ) -> ParameterEstimate:
        """Return a smooth contrast with joint delta-method inference.

        Parameters
        ----------
        function : callable
            Smooth scalar function of the selected estimates.
        names : sequence of str
            Aliases passed to ``function`` in order.
        name : str or None
            Alias for the derived estimate.
        scale : str
            Reported scale of the derived estimate. With ``transform=`` it must be
            ``"level"`` or ``"difference"``.
        gradient : callable or None
            Analytic gradient. ``None`` uses central differences.
        transform : Transform or None
            The map the interval and the test are computed on, as for
            :meth:`cleverly.estimators.TMLEResult.contrast`.

        Returns
        -------
        ParameterEstimate
            Derived estimate. It inherits the inference status of the selected
            estimates, so on a fit that supplies no inference it reports the same
            diagnostic and no interval.  On a fit with ``repeats > 1`` it is computed in
            every draw from that draw's estimates and combined by the median rule on its
            inference scale (the transform's scale with ``transform=``), with the status
            the draws' precedence gives.  It is not the function of the median estimates.

        Raises
        ------
        ValueError
            When a draw's value has no finite value or slope under ``transform``, or the
            delta method refuses in a draw.  The message names the draw.
        """
        if self.n_repeats > 1:
            check_transform_scale(scale, transform)
            select_estimates(self.estimates, names)

            def per_draw(estimates: Mapping[str, ParameterEstimate]) -> ParameterEstimate:
                return smooth_contrast(
                    estimates,
                    function,
                    names,
                    n=self.n,
                    cluster=self.data.cluster,
                    alpha=self.config.alpha_sig,
                    name=name,
                    scale=scale,
                    gradient=gradient,
                )

            return median_derived(
                self._draw_estimates(), per_draw, transform=transform, cluster=self.data.cluster
            )
        derived = smooth_contrast(
            self.estimates,
            function,
            names,
            n=self.n,
            cluster=self.data.cluster,
            alpha=self.config.alpha_sig,
            name=name,
            scale=scale,
            gradient=gradient,
            transform=transform,
        )
        return with_derived_bootstrap(
            derived,
            self.estimates,
            self.bootstrap,
            names,
            function,
            cluster=self.data.cluster,
        )

    def _levels(self) -> dict[tuple[str, str | None, int], str]:
        """``(regimen, cause, horizon) -> name`` of every reported level.

        Read from the index composed when each name was built, never parsed.
        """
        index = self.parameter_index or {}
        return {
            key: name
            for name, key in index.items()
            if name in self.estimates and self.estimates[name].scale == "level"
        }

    def _refuse_curve_functional(self, what: str) -> None:
        """Refuse a functional of the survival curve on a fit that reports no curve."""
        if self.msm is not None:
            reported = "the coefficients of a working model"
        elif self.data.is_repeated:
            raise CapabilityError(
                f"{what} sums a survival curve, and this fit reports {REPEATED_NOT_A_RISK}",
                reason=RefusalReason.DIFFERENT_ESTIMAND,
                alternative=REPEATED_ALTERNATIVE,
            )
        elif not self.data.is_survival:
            reported = f"one end-of-study mean of {self.data.outcome_name!r}"
        else:
            return
        raise CapabilityError(
            f"{what} sums a survival curve, and this fit reports {reported}.",
            reason=RefusalReason.DIFFERENT_ESTIMAND,
            alternative="Declare an outcome sequence to estimate a curve.",
        )

    def ratio(
        self,
        numerator: str,
        denominator: str,
        *,
        kind: RatioKind = "rr",
        view: Literal["risk", "survival"] = "risk",
        name: str | None = None,
    ) -> ParameterEstimate:
        """The risk ratio or odds ratio of two regimen levels, with log-scale inference.

        The ratio of two ``ey_regimen`` means, of two ``risk_regimen`` risks at one
        horizon, or of two ``cif_regimen`` incidences at one cause and horizon. The
        interval is computed on the log scale and exponentiated, as
        ``lmtp_contrast(type = "rr")`` and ``type = "or"`` in R ``lmtp`` 1.5.4 compute
        it. ``view="survival"`` takes the ratio of :math:`1 - F` instead of :math:`F`.
        The ratio inherits the inference status of its inputs.

        Parameters
        ----------
        numerator : str
            The name of the level in the numerator.
        denominator : str
            The name of the level in the denominator.
        kind : {"rr", "or"}, default="rr"
            A ratio of the levels, or a ratio of their odds. On an end-of-study fit
            ``"or"`` needs a binary outcome.
        view : {"risk", "survival"}, default="risk"
            Whether to take the ratio of the risks or of the survival probabilities.
        name : str or None, default=None
            The name of the derived estimate. ``None`` composes
            ``rr_regimen[a vs b @ t=h]`` from the parameter index when both inputs are
            levels at one cause and horizon, and ``rr[<numerator> vs <denominator>]``
            otherwise.

        Returns
        -------
        ParameterEstimate
            The ratio, on the ``"ratio"`` scale.

        Raises
        ------
        CapabilityError
            When ``kind="or"`` and an end-of-study outcome is not binary.
        ValueError
            When an input is not a level, when ``view="survival"`` is requested of an
            end-of-study fit or of a fit with two or more causes, or when this fit
            reports the coefficients of a working model.

        See Also
        --------
        cleverly.inference.ParameterEstimate.wald_test : Test the ratio at a null.

        Examples
        --------
        >>> from sklearn.linear_model import LinearRegression, LogisticRegression
        >>> from cleverly.datasets import make_longitudinal
        >>> from cleverly.longitudinal import LTMLE
        >>> frame, _ = make_longitudinal(n=200, seed=0)
        >>> result = LTMLE(
        ...     {"always": 1, "never": 0},
        ...     n_folds=1,
        ...     outcome_learner=LinearRegression(),
        ...     treatment_learner=LogisticRegression(max_iter=1000),
        ...     censoring_learner=LogisticRegression(max_iter=1000),
        ... ).fit(
        ...     frame,
        ...     outcome="Y",
        ...     treatment=["A1", "A2"],
        ...     baseline=["W1", "W2"],
        ...     time_varying=[[], ["L2"]],
        ...     censoring=["C1", "C2"],
        ... )
        >>> rr = result.ratio("ey_regimen[always]", "ey_regimen[never]")
        >>> rr.name, rr.scale
        ('rr_regimen[always vs never]', 'ratio')
        """
        if view not in ("risk", "survival"):
            raise ValueError(f"view must be 'risk' or 'survival'; got {view!r}")
        if self.msm is not None:
            raise CapabilityError(
                "this fit reports the coefficients of a working model, and a ratio of two "
                "coefficients is not a ratio of two regimen means.",
                reason=RefusalReason.DIFFERENT_ESTIMAND,
                alternative="Use result.contrast() for a function of the coefficients.",
            )
        survival = self.data.is_survival
        if view == "survival":
            if self.data.is_repeated:
                raise CapabilityError(
                    f"this fit reports {REPEATED_NOT_A_RISK}",
                    reason=RefusalReason.DIFFERENT_ESTIMAND,
                    alternative=f"Drop view=. {REPEATED_ALTERNATIVE}",
                )
            if not survival:
                raise CapabilityError(
                    "an end-of-study fit reports means, not a survival curve.",
                    reason=RefusalReason.DIFFERENT_ESTIMAND,
                    alternative="Drop view=.",
                )
            if len(self.config.causes) > 1:
                raise CapabilityError.from_refusal(SURVIVAL_VIEW_COMPETING_REFUSAL)
        if kind == "or" and not survival:
            refuse_odds_ratio(self.config.family)
        if self.config.estimator in _RANGE_CAUSE:
            # Each draw's level on a repeated fit, because the ratio is formed per draw:
            # a positive median says nothing about the draw whose level is not.
            for index, repeat in enumerate(self.repeats, start=1):
                for level in (numerator, denominator):
                    value = (self if self.n_repeats == 1 else repeat.estimates)[level].psi
                    refuse_sdr_scale(
                        level,
                        1.0 - value if view == "survival" else value,
                        "odds" if kind == "or" else "log",
                        estimator=self.config.estimator,
                        draw=None if self.n_repeats == 1 else (index, self.n_repeats),
                    )
        label = name or self._ratio_name(numerator, denominator, kind, view)
        if self.n_repeats > 1:
            select_estimates(self.estimates, (numerator, denominator))

            def per_draw(estimates: Mapping[str, ParameterEstimate]) -> ParameterEstimate:
                return ratio_contrast(
                    estimates,
                    numerator,
                    denominator,
                    kind=kind,
                    complement=view == "survival",
                    n=self.n,
                    cluster=self.data.cluster,
                    alpha=self.config.alpha_sig,
                    name=label,
                )

            return median_derived(self._draw_estimates(), per_draw, cluster=self.data.cluster)
        derived = ratio_contrast(
            self.estimates,
            numerator,
            denominator,
            kind=kind,
            complement=view == "survival",
            n=self.n,
            cluster=self.data.cluster,
            alpha=self.config.alpha_sig,
            name=label,
        )
        return with_derived_bootstrap(
            derived,
            self.estimates,
            self.bootstrap,
            (numerator, denominator),
            ratio_function(kind, complement=view == "survival"),
            cluster=self.data.cluster,
        )

    def _ratio_name(self, numerator: str, denominator: str, kind: str, view: str) -> str:
        """The structured name of a regimen ratio, composed forward from the index."""
        prefix = "survival_" if view == "survival" else ""
        index = self.parameter_index or {}
        top, bottom = index.get(numerator), index.get(denominator)
        levels = set(self._levels().values())
        if (
            top is not None
            and bottom is not None
            and numerator in levels
            and denominator in levels
            and top[1:] == bottom[1:]
        ):
            label = f"{top[0]} vs {bottom[0]}"
            arm = _index(label, top[1], top[2], self.data.reports_curve)
            return parameter_name(f"{prefix}{kind}_regimen", arm=arm)
        return f"{prefix}{kind}[{numerator} vs {denominator}]"

    def _restriction(self, horizon: float, what: str) -> tuple[int, tuple[float, ...]]:
        r"""The node index :math:`m` of a restriction time, and the grid spacings below it.

        On the wide path, and on a unit grid, the time is the node index ``m`` from 2 to
        ``n_times + 1`` and every spacing is one.  On any other grid the time is a grid
        value :math:`g_m` with :math:`2 \le m \le K`, and the spacings are
        :math:`g_k - g_{k-1}` for :math:`k = 2, \ldots, m`.
        """
        n_times = self.data.n_times
        grid = self.data.time_grid
        if grid is None or grid == tuple(float(k) for k in range(1, n_times + 1)):
            if (
                isinstance(horizon, bool)
                or not isinstance(
                    horizon, (int, np.integer) if grid is None else (int, float, np.number)
                )
                or float(horizon) != int(horizon)
                or not 2 <= int(horizon) <= n_times + 1
            ):
                raise ValueError(
                    f"horizon must be an integer from 2 to {n_times + 1}; {what} up to t=1 is "
                    "the constant 1"
                )
            return int(horizon), (1.0,) * (int(horizon) - 1)
        values = np.asarray(grid, dtype=float)
        matches = np.flatnonzero(values == float(horizon))
        if isinstance(horizon, bool) or matches.size != 1 or int(matches[0]) == 0:
            shown = ", ".join(f"{g:g}" for g in grid[1:])
            raise ValueError(
                f"horizon {horizon!r} is not a reported grid time; this fit's grid is "
                f"{', '.join(f'{g:g}' for g in grid)}. {what} takes a grid time from the second "
                f"to the last, {shown}: up to the first grid time it is that time times one"
            )
        m = int(matches[0]) + 1
        return m, tuple(float(values[k] - values[k - 1]) for k in range(1, m))

    def _risk_names(
        self, regimen: str, horizon: int, causes: Sequence[str | None], what: str
    ) -> list[str]:
        """The names of the levels at nodes ``1`` to ``horizon - 1``, one per cause.

        ``horizon`` is the node index :math:`m` that :meth:`_restriction` returns.
        """
        levels = self._levels()
        regimens = sorted({key[0] for key in levels})
        if regimen not in regimens:
            raise KeyError(f"unknown regimen {regimen!r}; this fit reports {regimens}")
        reported = sorted({key[2] for key in levels if key[0] == regimen})
        missing = [t for t in range(1, int(horizon)) if t not in reported]
        if missing:
            raise ValueError(
                f"{what} up to t={horizon} needs the risk at every horizon 1 to "
                f"{horizon - 1}, and this fit reports {reported}. Refit with horizons=None "
                f"or include {missing}"
            )
        return [levels[(regimen, cause, t)] for t in range(1, int(horizon)) for cause in causes]

    def rmst(
        self,
        regimen: str,
        horizon: int,
        *,
        versus: str | None = None,
        name: str | None = None,
    ) -> ParameterEstimate:
        r"""The restricted mean survival time of a regimen up to a horizon.

        With :math:`T` the node of the first event and :math:`F(t)` the reported risk,

        .. math::

            \mathrm{RMST}(\tau) = E[\min(T, \tau)]
            = \tau - \sum_{t=1}^{\tau - 1} F(t),

        with the curve :math:`-\sum_{t=1}^{\tau-1} IC_{F(t)}`. The unit is the node
        index. For nodes spaced :math:`\Delta` apart, the RMST in calendar time is
        :math:`\Delta \times \mathrm{RMST}`. On a competing-risk fit, :math:`F` is the
        sum of the cause-specific incidences. The estimate inherits the inference status
        of the risks it sums.

        Parameters
        ----------
        regimen : str
            The regimen label.
        horizon : int
            The restriction time :math:`\tau`, an integer from 2 to ``n_times + 1``.
        versus : str or None, default=None
            A second regimen. The result is then the difference of the two RMSTs.
        name : str or None, default=None
            The name of the derived estimate. ``None`` gives
            ``rmst_regimen[a @ t=tau]`` or ``rmst_regimen[a vs b @ t=tau]``.

        Returns
        -------
        ParameterEstimate
            A ``"level"`` estimate, or a ``"difference"`` estimate with ``versus=``.

        Raises
        ------
        ValueError
            When the fit reports no survival curve, when the horizon is outside the
            range, or when the fit omits a horizon below it.

        Examples
        --------
        >>> from sklearn.linear_model import LinearRegression, LogisticRegression
        >>> from cleverly.datasets import make_longitudinal_survival
        >>> from cleverly.longitudinal import LTMLE
        >>> frame, _ = make_longitudinal_survival(n=300, seed=0)
        >>> result = LTMLE(
        ...     {"always": 1, "never": 0},
        ...     n_folds=1,
        ...     outcome_learner=LinearRegression(),
        ...     treatment_learner=LogisticRegression(max_iter=1000),
        ...     censoring_learner=LogisticRegression(max_iter=1000),
        ... ).fit(
        ...     frame,
        ...     outcome=["Y1", "Y2"],
        ...     treatment=["A1", "A2"],
        ...     baseline=["W1", "W2"],
        ...     time_varying=[[], ["L2"]],
        ...     censoring=["C1", "C2"],
        ... )
        >>> rmst = result.rmst("always", 3, versus="never")
        >>> rmst.name, rmst.scale
        ('rmst_regimen[always vs never @ t=3]', 'difference')
        """
        self._refuse_curve_functional("RMST")
        causes: tuple[str | None, ...] = self.data.cause_labels or (None,)
        node, spacings = self._restriction(horizon, "RMST")
        names = self._risk_names(regimen, node, causes, "RMST")
        per_name = [spacing for spacing in spacings for _ in causes]
        weights = {key: -spacing for key, spacing in zip(names, per_name, strict=True)}
        constant = float(horizon)
        label = regimen
        scale: Scale = "level"
        if versus is not None:
            others = self._risk_names(versus, node, causes, "RMST")
            for other, spacing in zip(others, per_name, strict=True):
                weights[other] = weights.get(other, 0.0) + spacing
            constant = 0.0
            label = f"{regimen} vs {versus}"
            scale = "difference"
        return self._linear(
            weights,
            constant=constant,
            name=name or parameter_name("rmst_regimen", arm=_index(label, None, node, True)),
            scale=scale,
        )

    def _linear(
        self,
        weights: Mapping[str, float],
        *,
        constant: float = 0.0,
        name: str,
        scale: Scale,
    ) -> ParameterEstimate:
        """A constant plus a weighted sum of the estimates, per draw on a repeated fit."""
        if self.n_repeats > 1:

            def per_draw(estimates: Mapping[str, ParameterEstimate]) -> ParameterEstimate:
                return linear_functional(
                    estimates,
                    weights,
                    constant=constant,
                    n=self.n,
                    cluster=self.data.cluster,
                    alpha=self.config.alpha_sig,
                    name=name,
                    scale=scale,
                )

            return median_derived(self._draw_estimates(), per_draw, cluster=self.data.cluster)
        derived = linear_functional(
            self.estimates,
            weights,
            constant=constant,
            n=self.n,
            cluster=self.data.cluster,
            alpha=self.config.alpha_sig,
            name=name,
            scale=scale,
        )
        return with_derived_bootstrap(
            derived,
            self.estimates,
            self.bootstrap,
            tuple(weights),
            linear_function(tuple(weights.values()), constant),
            cluster=self.data.cluster,
        )

    def rmtl(
        self,
        regimen: str,
        horizon: int,
        cause: str | None = None,
        *,
        versus: str | None = None,
        name: str | None = None,
    ) -> ParameterEstimate:
        r"""The restricted mean time lost to one cause, up to a horizon.

        With :math:`J` the cause of the first event and :math:`F_j(t)` its reported
        cumulative incidence,

        .. math::

            \mathrm{RMTL}_j(\tau) = E[(\tau - \min(T, \tau)) 1\{J = j\}]
            = \sum_{t=1}^{\tau - 1} F_j(t),

        with the curve :math:`\sum_{t=1}^{\tau-1} IC_{F_j(t)}`. On a one-event fit it
        is :math:`\tau - \mathrm{RMST}(\tau)`. The unit is the node index.

        Parameters
        ----------
        regimen : str
            The regimen label.
        horizon : int
            The restriction time :math:`\tau`, an integer from 2 to ``n_times + 1``.
        cause : str or None, default=None
            The cause. Required on a fit with two or more causes.
        versus : str or None, default=None
            A second regimen. The result is then the difference of the two RMTLs.
        name : str or None, default=None
            The name of the derived estimate. ``None`` gives
            ``rmtl_regimen[a, cause @ t=tau]``, or ``rmtl_regimen[a @ t=tau]`` on a
            one-event fit.

        Returns
        -------
        ParameterEstimate
            A ``"level"`` estimate, or a ``"difference"`` estimate with ``versus=``.

        Raises
        ------
        ValueError
            When the fit reports no survival curve, when the horizon is outside the
            range, when the fit omits a horizon below it, or when the cause is missing
            or unknown.

        Examples
        --------
        >>> from sklearn.linear_model import LinearRegression, LogisticRegression
        >>> from cleverly.datasets import make_longitudinal_survival
        >>> from cleverly.longitudinal import LTMLE
        >>> frame, _ = make_longitudinal_survival(n=300, seed=0)
        >>> result = LTMLE(
        ...     {"always": 1, "never": 0},
        ...     n_folds=1,
        ...     outcome_learner=LinearRegression(),
        ...     treatment_learner=LogisticRegression(max_iter=1000),
        ...     censoring_learner=LogisticRegression(max_iter=1000),
        ... ).fit(
        ...     frame,
        ...     outcome=["Y1", "Y2"],
        ...     treatment=["A1", "A2"],
        ...     baseline=["W1", "W2"],
        ...     time_varying=[[], ["L2"]],
        ...     censoring=["C1", "C2"],
        ... )
        >>> lost = result.rmtl("always", 3)
        >>> rmst = result.rmst("always", 3)
        >>> round(lost.psi + rmst.psi, 12)
        3.0
        """
        self._refuse_curve_functional("RMTL")
        declared = self.data.cause_labels
        if cause is None:
            if len(declared) > 1:
                raise ValueError(
                    f"this fit declares the causes {list(declared)}; name one with cause="
                )
            chosen: str | None = declared[0] if declared else None
        else:
            if cause not in declared:
                raise ValueError(
                    f"unknown cause {cause!r}; this fit declares {list(declared)}"
                    if declared
                    else "this fit declares no causes; leave cause=None for its one event"
                )
            chosen = cause
        node, spacings = self._restriction(horizon, "RMTL")
        names = self._risk_names(regimen, node, (chosen,), "RMTL")
        weights = dict(zip(names, spacings, strict=True))
        label = regimen
        scale: Scale = "level"
        if versus is not None:
            others = self._risk_names(versus, node, (chosen,), "RMTL")
            for other, spacing in zip(others, spacings, strict=True):
                weights[other] = weights.get(other, 0.0) - spacing
            label = f"{regimen} vs {versus}"
            scale = "difference"
        return self._linear(
            weights,
            name=name or parameter_name("rmtl_regimen", arm=_index(label, chosen, node, True)),
            scale=scale,
        )

    def _names(self, names: Sequence[str] | None) -> tuple[str, ...]:
        return select_estimates(self.estimates, names)

    # ------------------------------------------------------------ diagnostics

    @property
    def converged(self) -> bool:
        """Whether every node's targeting step reached its tolerance, in every draw."""
        return all(fit.converged for repeat in self.repeats for fit in repeat.fits.values())

    @property
    def diagnostics(self) -> Any:
        """Unified support, nuisance, score, and refutation diagnostics."""
        from ..assessment import DiagnosticsFacade

        return DiagnosticsFacade(self)

    def validate(self) -> Any:
        """Run the inexpensive default battery without refitting.

        Returns
        -------
        ValidationReport
            One item per validated check, read off stored artifacts.
        """
        from ..assessment import validate_result

        return validate_result(self)

    def assess(
        self,
        *,
        include_refits: bool = False,
        include_retargets: bool = False,
        arguments: Mapping[str, Mapping[str, Any]] | None = None,
        random_state: int | None = None,
    ) -> AssessmentReport:
        """Run the applicable post-fit assessment battery.

        Parameters
        ----------
        include_refits : bool
            Run requested operations that refit nuisance models.
        include_retargets : bool
            Include moderate retargets; cheap E-value retargets run by default.
        arguments : mapping or None
            Per-operation keyword arguments.
        random_state : int or None
            Common seed for stochastic refit operations.

        Returns
        -------
        cleverly.AssessmentReport
            Validation, diagnostics, and sensitivity in one report.

        See Also
        --------
        LongitudinalResult.validate : Run stored-artifact validation.
        cleverly.assessment.DiagnosticsFacade.run_all : Run diagnostics alone.
        """
        from ..assessment import assess_result

        return assess_result(
            self,
            include_refits=include_refits,
            include_retargets=include_retargets,
            arguments=arguments,
            random_state=random_state,
        )

    @property
    def replayability(self) -> Any:
        """Which assessment operations survive from the stored sequential artifacts."""
        from ..assessment import replayability

        return replayability(self)

    def incidence_total(self) -> Any:
        """Per regimen and horizon, the incidences summed over the causes.

        A unit leaves the risk set through exactly one cause or not at all, so the
        cumulative incidences and the event-free probability exhaust the mass:
        :math:`\\sum_j F_j(k) + S(k) = 1`.  That identity holds of the *parameters*.  It
        does **not** hold of these estimates, and the difference is not a bug to fix.
        Each cause is a separate backward pass -- its own regressions, its own
        fluctuation -- so nothing constrains the sum, and a total above one is possible.

        It is reported rather than removed, on the same reasoning as
        :attr:`~cleverly.sensitivity.positivity.PositivityReport.simplex_deviation`, where
        a multi-arm mechanism's row is deliberately not rescaled back onto the simplex.
        Renormalising would buy a coherent-looking table by moving every cause's estimate
        away from the one its own score equation solved, and would hide the thing worth
        seeing: a total far from one says the causes disagree about how much risk there
        was, which is a statement about the nuisance fits and not about the parameter.

        The ``total`` column carries a standard error because the sum is itself a
        parameter with an influence curve -- the sum of the causes' curves -- so
        ``excess`` can be read against it rather than eyeballed.  On a fit whose status
        supplies no inference the column is ``plugin_std_err``, a diagnostic, as
        :meth:`to_frame` names the same spread.  Each total takes the status of the
        incidences it sums.  On a fit that supplies inference, a total that reads a
        constant-node incidence has NaN under ``std_err``, its spread under
        ``plugin_std_err``, and its status in an ``inference`` column.

        Returns
        -------
        dataframe
            One row per regimen and horizon, with the incidences summed over causes. The
            ``support`` column holds the label of the combined incidences, and a fit with a
            labelled status adds the ``inference`` column.
        """
        if not self.data.is_competing:
            if self.data.is_repeated:
                raise CapabilityError(
                    f"this fit reports {REPEATED_NOT_A_RISK}",
                    reason=RefusalReason.DIFFERENT_ESTIMAND,
                    alternative=REPEATED_ALTERNATIVE,
                )
            raise CapabilityError(
                "this fit reports one parameter per regimen per horizon, so there is "
                "nothing to sum over.",
                reason=RefusalReason.INVALID_DECLARATION,
                alternative=(
                    "Declare competing risks, outcome={cause: [...]}, to estimate a "
                    "cumulative incidence per cause."
                ),
            )
        # Renamed in place, as the point-treatment sweeps rename their spread columns:
        # the number is the same plug-in standard error under either status.  Each total
        # takes the status of the incidences it sums (:func:`inference_status`), so a sum
        # that reads a constant-node incidence is a diagnostic, as ``rmst()`` of one is.
        status = self.inference_status
        column = spread_name("std_err", status)
        totals: list[tuple[str, float, float, float, InferenceStatus, str]] = []
        for regimen in self.config.regimens:
            for horizon in self.config.horizons:
                names = [
                    f"cif_regimen[{_index(regimen.label, cause, horizon, True)}]"
                    for cause in self.config.causes
                ]
                if self.n_repeats > 1:
                    # The sum in every draw, combined by the median rule: the total of the
                    # median incidences is not a median of anything.
                    combined = self._linear(
                        dict.fromkeys(names, 1.0), name="incidence_total", scale="level"
                    )
                    totals.append(
                        (
                            regimen.label,
                            self._time_of(int(horizon)),
                            combined.psi,
                            combined.plugin_std_error,
                            combined.inference,
                            combined.support.label,
                        )
                    )
                    continue
                total = float(sum(self[name].psi for name in names))
                curve = np.sum(
                    np.column_stack([self[name].influence_curve for name in names]), axis=1
                )
                variance = influence_covariance(curve.reshape(-1, 1), cluster=self.data.cluster)
                totals.append(
                    (
                        regimen.label,
                        self._time_of(int(horizon)),
                        total,
                        float(np.sqrt(variance[0, 0])),
                        inference_status(self.estimates, names),
                        combine_support(
                            [self[name].support for name in names], kind="derived"
                        ).label,
                    )
                )
        flagged = [row[4] for row in totals if not supplies_inference(row[4])]
        mixed = supplies_inference(status) and bool(flagged)
        plugin = spread_name("std_err", flagged[0]) if mixed else None
        rows: dict[str, list[Any]] = {
            "regimen": [row[0] for row in totals],
            "time": [row[1] for row in totals],
            "total": [row[2] for row in totals],
            "support": [row[5] for row in totals],
            **(
                {"inference": [row[4] for row in totals]}
                if mixed or any(is_labelled(row[4]) for row in totals)
                else {}
            ),
            column: [np.nan if mixed and row[4] != status else row[3] for row in totals],
            **(
                {plugin: [row[3] if row[4] != status else np.nan for row in totals]}
                if plugin
                else {}
            ),
            "excess": [max(0.0, row[2] - 1.0) for row in totals],
        }
        return self.data.frame_like(rows)

    # ------------------------------------------------- what this fit cannot do

    @property
    def sensitivity(self) -> Any:
        """Return the sensitivity-analysis facade."""
        from ..assessment import SensitivityFacade

        return SensitivityFacade(self)

    def save(self, path: Any) -> Any:
        """Persist the complete fitted result to a trusted joblib artifact.

        The artifact records :data:`cleverly.__version__`. A load by a different version
        warns with :class:`~cleverly.exceptions.VersionMismatchWarning` and does not
        migrate the result.

        Parameters
        ----------
        path : path-like
            Destination file.

        Returns
        -------
        Path
            The destination path, as given.
        """
        from ..estimators.serialize import save as _save

        return _save(self, path)

    # ---------------------------------------------------------------- reports

    def curve(self, scale: str | None = None) -> Any:
        """The report as a curve: a row per parameter per horizon.

        On a repeated outcome (:class:`~cleverly.longitudinal.Repeated`) the only scale is
        ``"mean"``: the mean :math:`E[Y_t^d]` of each regimen and the difference between
        regimens at each horizon.  A repeated outcome can return to zero, so it has no
        survival complement.  The rest of this description is about a survival fit.

        ``scale="risk"`` reports what the fit estimated, the cumulative risk
        :math:`F(t)` and the risk difference between regimens at each horizon.
        ``scale="survival"`` reports :math:`S(t) = 1 - F(t)` instead.  The ``view``
        column names the requested view; the ``scale`` column keeps the vocabulary
        :meth:`to_frame` reports, ``"level"`` or ``"difference"``, which is what
        separates a level row from a contrast row without reading any name.

        A fit that declared **two or more** causes refuses the survival view, because
        event-free survival is the complement of the *sum* of the cause-specific
        incidences and not the complement of one of them.  With a single declared cause
        that sum is that one incidence, so the complement is event-free survival and the
        view is reported.

        The map from one to the other is **not** one rule.  For a level,
        :math:`S = 1 - F`: the estimate is mirrored about a half and so is its interval.
        For a *contrast* it is :math:`S_a - S_b = -(F_a - F_b)`: the estimate is negated
        and the interval negated and swapped.  Applying ``1 - x`` to a risk difference
        would report ``1 - RD``, which is not a quantity, with an interval that would
        look perfectly reasonable -- so this branches on
        :attr:`~cleverly.inference.ParameterEstimate.scale` rather than on the caller
        getting it right.  The standard error is the same either way, both maps being
        linear with slope of modulus one.

        The ``time`` column lives here rather than on :meth:`to_frame`, which keeps the
        column names a point-treatment fit reports.

        On ``scale="survival"`` a level row's ``estimand`` is **not** a key of this
        result: the complement of ``risk_regimen[always @ t=2]`` is reported as
        ``survival_regimen[always @ t=2]``, which the fit never estimated under that
        name.  A contrast row keeps its ``ate_regimen[...]`` name, which is a key, the
        difference being the same parameter up to a sign.  The ``parameter`` column
        carries the key either way, so ``result[row.parameter]`` is defined for every row
        of every view and ``result[row.estimand]`` is not.

        Parameters
        ----------
        scale : {"risk", "survival", "mean"} or None
            Outcome scale for levels and contrasts.  ``None`` reads ``"risk"`` on a survival
            fit and ``"mean"`` on a repeated outcome.

        Returns
        -------
        dataframe
            One row per regimen, cause, and horizon in the input backend, with these
            columns.

            ``estimand``
                The name of the quantity the row reports, under the requested view.
            ``parameter``
                The key of the estimate the row was derived from, always in
                ``result.estimates``.
            ``regimen``
                The regimen of a level, or the ``"a vs b"`` contrast string of a
                difference.
            ``cause``
                The absorbing cause, on a fit that declared its causes only.
            ``time``
                The horizon, as an integer node index, or as the grid time :math:`g_k` on a
                fit with a time grid.
            ``support``
                ``"supported"`` or ``"experimental"``, the label of the estimate the row
                reads. :attr:`~cleverly.inference.ParameterEstimate.support` gives its reason.
            ``inference``
                The inference status of each row, on a fit with a labelled status, shown or
                withheld, or with a parameter that carries one.
            ``psi``, ``std_err``, ``ci_lower``, ``ci_upper``
                The estimate, its standard error, and its confidence interval, under
                the requested view.  On a fit whose status supplies no inference the
                three spread columns are ``plugin_std_err``, ``plugin_interval_lower``
                and ``plugin_interval_upper``, a diagnostic, as :meth:`to_frame` names
                them.  On a fit that supplies inference, a row with a parameter status
                has NaN in the three inferential columns and its diagnostic in the three
                plug-in columns, which the frame then adds.
            ``scale``
                ``"level"`` or ``"difference"``, as :meth:`to_frame` reports it.
            ``view``
                ``"risk"``, ``"survival"`` or ``"mean"``, the view the caller requested.
        """
        repeated = self.data.is_repeated
        if scale is None:
            scale = "mean" if repeated else "risk"
        if scale not in ("risk", "survival", "mean"):
            raise ValueError(f"scale must be 'risk', 'survival' or 'mean'; got {scale!r}")
        if repeated and scale != "mean":
            raise CapabilityError(
                f"this fit reports {REPEATED_NOT_A_RISK}",
                reason=RefusalReason.DIFFERENT_ESTIMAND,
                alternative=f"Use scale='mean'. {REPEATED_ALTERNATIVE}",
            )
        if self.msm is not None:
            raise CapabilityError(
                "this fit reports the coefficients of a working model, and a coefficient "
                "has no horizon to index a curve by: the horizon is inside the design, "
                "which is what lets a coefficient be a trend across horizons rather than "
                "one number per horizon. S = 1 - F is not a map on a coefficient either.",
                reason=RefusalReason.DIFFERENT_ESTIMAND,
                alternative=(
                    "result.to_frame() is the report for this fit, and result.coefficients() "
                    "the view that names what each one is."
                ),
            )
        if not self.data.reports_curve:
            raise CapabilityError(
                f"this fit has one end-of-study outcome ({self.data.outcome_name!r}), so "
                "its report is a number and not a curve.",
                reason=RefusalReason.DIFFERENT_ESTIMAND,
                alternative=(
                    "Pass one outcome column per time point, outcome=[...], to estimate a "
                    "cumulative risk at every horizon. result.to_frame() is the report for "
                    "this fit."
                ),
            )
        if not repeated and scale == "mean":
            raise ValueError(
                "scale='mean' is the view of a repeated outcome; this fit reports a "
                "cumulative risk at every horizon. Use scale='risk' or scale='survival'"
            )
        # Gated on how many causes the fit *declared*, not on whether it declared any.
        # ``outcome={"death": [...]}`` is a competing-risks declaration with one cause,
        # and ``test_one_cause_reproduces_the_single_event_fit`` pins that fit as bit for
        # bit the ``outcome=[...]`` one.  With one cause the causes' sum is that cause's
        # own incidence, so its complement *is* event-free survival: refusing it would
        # refuse a supported composition, and would make two declarations of the same
        # data disagree about what can be reported from an identical fit.
        if scale == "survival" and len(self.config.causes) > 1:
            raise CapabilityError.from_refusal(SURVIVAL_VIEW_COMPETING_REFUSAL)
        competing = self.data.is_competing
        # The names ``to_frame`` publishes for the same numbers, through the same table.
        status = self.inference_status
        # A parameter status, such as a constant-node risk, on a fit that supplies inference
        # adds the ``inference`` column and the plug-in columns, as ``to_frame`` does: that
        # row's inferential columns hold NaN and its plug-in columns hold the diagnostic.
        flagged = [e.inference for e in self.estimates.values() if not e.supplies_inference]
        mixed = supplies_inference(status) and bool(flagged)
        diagnostic = (
            not supplies_inference(status)
            or mixed
            or any(is_labelled(e.inference) for e in self.estimates.values())
        )
        std_name, low_name, high_name = (
            spread_name(name, status) for name in ("std_err", "ci_lower", "ci_upper")
        )
        plugin_names = (
            tuple(spread_name(name, flagged[0]) for name in ("std_err", "ci_lower", "ci_upper"))
            if mixed
            else ()
        )
        rows: dict[str, list[Any]] = {
            "estimand": [],
            "parameter": [],
            "regimen": [],
            **({"cause": []} if competing else {}),
            "time": [],
            "psi": [],
            "support": [],
            **({"inference": []} if diagnostic else {}),
            std_name: [],
            low_name: [],
            high_name: [],
            **({column: [] for column in plugin_names} if mixed else {}),
            "scale": [],
            "view": [],
        }
        # Read from the index composed when the name was built, never split back out of
        # it.  With a cause beside the horizon inside one pair of brackets there is no
        # split that is right in general: a regimen legitimately called ``"a, b"`` would
        # send ``rpartition`` to the wrong comma, and the row would be filed under a
        # regimen that does not exist rather than failing.
        index_of = self.parameter_index or {}
        for name, estimate in self.estimates.items():
            label, cause, horizon = index_of[name]
            # The plug-in bodies, which ``ci`` and ``std_error`` also read, so an ordinary
            # fit's numbers are unchanged and a diagnostic fit's are the same arithmetic.
            low, high = estimate.plugin_interval
            if scale in ("risk", "mean"):
                psi, low, high = estimate.psi, low, high
            elif estimate.scale == "level":
                psi, low, high = 1.0 - estimate.psi, 1.0 - high, 1.0 - low
            else:
                psi, low, high = -estimate.psi, -high, -low
            # Composed forward from the same index, through the same two functions that
            # built ``name`` in the first place.  Recovering the head by matching a
            # prefix would be the parsing the comment above refuses, one step later.
            head = (
                _level_head(
                    survival=not repeated, competing=competing, complement=scale == "survival"
                )
                if estimate.scale == "level"
                else CONTRAST_HEAD
            )
            reported_name = parameter_name(head, arm=_index(label, cause, horizon, True))
            rows["estimand"].append(reported_name)
            rows["parameter"].append(name)
            rows["regimen"].append(label)
            if competing:
                rows["cause"].append(cause)
            rows["time"].append(self._time_of(int(horizon)))
            rows["psi"].append(float(psi))
            rows["support"].append(estimate.support.label)
            if diagnostic:
                rows["inference"].append(estimate.inference)
            withheld = mixed and not estimate.supplies_inference
            spread = (estimate.plugin_std_error, float(low), float(high))
            for column, number in zip((std_name, low_name, high_name), spread, strict=True):
                rows[column].append(np.nan if withheld else number)
            if mixed:
                for column, number in zip(plugin_names, spread, strict=True):
                    rows[column].append(number if withheld else np.nan)
            rows["scale"].append(estimate.scale)
            rows["view"].append(scale)
        return self.data.frame_like(rows)

    def _time_of(self, horizon: int) -> int | float:
        """The reported time of node ``horizon``: its grid time, or the index itself."""
        grid = self.data.time_grid
        return int(horizon) if grid is None else float(grid[horizon - 1])

    def coefficients(self, scale: str = "link") -> Any:
        """A working model's coefficients, on the link scale or exponentiated.

        The same view :meth:`cleverly.estimators.base.TMLEResult.coefficients` is at one
        time point, and it means the same things.  ``scale="link"`` reports
        :math:`\\hat\\beta` with a Wald interval on the scale the model is linear on;
        ``scale="ratio"`` reports :math:`e^{\\hat\\beta}` with the interval exponentiated
        from that scale.  **What the exponential is depends on the link** -- a risk ratio
        under ``"log"`` and an *odds* ratio under ``"logit"`` -- and the intercept is a
        third thing again, a baseline mean or a baseline odds rather than a ratio of
        anything, so the ``scale`` column names which each row is.  An intercept is found
        by its column being constant one, not by its name.

        Refused on an identity-link fit, where :math:`e^\\beta` of a risk difference is
        not a quantity, and on a fit with no working model.  Nothing is re-estimated.

        Parameters
        ----------
        scale : {"link", "ratio"}
            Coefficient scale to report.

        Returns
        -------
        dataframe
            One row per working-model coefficient in the input backend.
        """
        if scale not in ("link", "ratio"):
            raise ValueError(f"scale must be 'link' or 'ratio'; got {scale!r}")
        if self.msm is None:
            raise CapabilityError(
                "this fit has no working model, so it reports a mean per regimen rather "
                "than coefficients.",
                reason=RefusalReason.API_REDIRECTION,
                alternative=(
                    "result.to_frame() is its report. Declare msm= to project those means "
                    "onto a model whose coefficients are the parameters."
                ),
            )
        if scale == "ratio" and self.msm.link == "identity":
            raise CapabilityError(
                "an identity-link working model's coefficients are risk differences, and "
                "exp() of a difference is not a quantity anybody reports.",
                reason=RefusalReason.DIFFERENT_ESTIMAND,
                alternative=(
                    "Declare link='log' for coefficients that are log risk ratios, or "
                    "link='logit' for log odds ratios, and this view exponentiates those."
                ),
            )
        ratio = "risk ratio" if self.msm.link == "log" else "odds ratio"
        rows: list[dict[str, Any]] = []
        for msm_fit in self.msm_fits:
            for column, term in enumerate(self.msm.terms):
                inside = term if msm_fit.cause is None else f"{term}{CAUSE_INFIX}{msm_fit.cause}"
                estimate = self.estimates[f"msm_regimen[{inside}]"]
                if scale == "link":
                    rows.append(estimate.to_dict())
                    continue
                exponentiated = replace(
                    estimate, psi=float(np.exp(estimate.psi)), log_psi=estimate.psi, scale="ratio"
                )
                row = exponentiated.to_dict()
                row["scale"] = (
                    "baseline" if bool(np.all(self.msm.design[:, :, column] == 1.0)) else ratio
                )
                rows.append(row)
        return self.data.frame_like({key: [row[key] for row in rows] for key in rows[0]})

    def to_frame(self) -> Any:
        """One row per reported parameter, in the backend the data came from.

        Built from :meth:`~cleverly.inference.ParameterEstimate.to_dict`, so the columns
        are the ones a point-treatment fit reports under the same names.  Two result
        objects in one library disagreeing on the name of every column is a worse cost
        than the one this saved.

        Returns
        -------
        dataframe
            One row per reported parameter, in the backend the data arrived in.  A fit with
            a time grid adds a ``time`` column, the grid time of each parameter's horizon.
        """
        rows = [estimate.to_dict() for estimate in self.estimates.values()]
        # A parameter status, such as a constant-node risk, gives its row the plug-in column
        # names and an ``inference`` column, so the frame holds the union of the columns and
        # every row names its status.
        keys = list(dict.fromkeys(key for row in rows for key in row))
        for row, estimate in zip(rows, self.estimates.values(), strict=True):
            if "inference" in keys:
                row.setdefault("inference", estimate.inference)
        payload: dict[str, list[Any]] = {
            key: [row.get(key, np.nan) for row in rows] for key in keys
        }
        if self.data.time_grid is not None and self.parameter_index:
            # A fit with a time grid reports each horizon's grid time beside its name, which
            # carries the node index.
            payload["time"] = [
                self._time_of(self.parameter_index[name][2])
                if name in self.parameter_index
                else None
                for name in self.estimates
            ]
        return self.data.frame_like(payload)

    def summary(self, *, detail: Detail = "short") -> str:
        """A printable report: the estimates, then the settings, then the leverage.

        Parameters
        ----------
        detail : {"short", "full"}, default "short"
            ``"short"`` prints the analysis. ``"full"`` also prints the fields of the study
            protocol record, or that the fit had none.

        Returns
        -------
        str
            A printable report: the estimates, the settings, then the leverage.
        """
        level = f"{(1 - self.config.alpha_sig) * 100:g}%"
        # The fit status decides the columns.  On a fit that supplies inference, a parameter
        # with its own status (a constant-node risk) is starred below, row by row.
        status = self.inference_status
        record = None if supplies_inference(status) else status_record(status)
        t_reference = has_t_reference(self.estimates)
        rows = []
        marked: dict[str, Any] = {}
        support_cells = support_column(self.estimates)
        if record is not None:
            # The whole interval column is refused, so it is not printed with a dash under
            # it: a heading would still tell a reader an interval belongs there.
            for name, estimate in self.estimates.items():
                rows.append([name, f"{estimate.psi:.4f}", f"{estimate.plugin_std_error:.4f}"])
            headers = ["parameter", "estimate", record.summary_label]
            if support_cells is not None:
                headers.append("support")
                rows = [[*row, cell] for row, cell in zip(rows, support_cells, strict=True)]
            table = format_table(headers, rows)
        else:
            for name, estimate in self.estimates.items():
                if not estimate.supplies_inference:
                    # A parameter with its own status, such as a constant-node risk: the
                    # plug-in spread is marked, and no interval or p-value is printed.
                    marked[estimate.inference] = status_record(estimate.inference)
                    rows.append(
                        [
                            name,
                            f"{estimate.psi:.4f}",
                            f"{estimate.plugin_std_error:.4f}*",
                            "not reported",
                            "not reported",
                            *([""] if t_reference else []),
                        ]
                    )
                    continue
                low, high = estimate.ci
                rows.append(
                    [
                        name,
                        f"{estimate.psi:.4f}",
                        f"{estimate.std_error:.4f}",
                        f"[{low:.4f}, {high:.4f}]",
                        format_pvalue(estimate.pvalue),
                        *([reference_label(estimate)] if t_reference else []),
                    ]
                )
            headers = [
                "parameter",
                "estimate",
                "std. error",
                f"{level} CI",
                "p-value",
                *(["df"] if t_reference else []),
            ]
            if support_cells is not None:
                headers.append("support")
                rows = [[*row, cell] for row, cell in zip(rows, support_cells, strict=True)]
            table = format_table(headers, rows)
        # ``"level"`` is the scale of every mean, the rule ``to_frame`` reads as well.
        contrast = any(estimate.scale != "level" for estimate in self.estimates.values())
        facts = list(self.config.describe(contrast=contrast))
        if self.simulation is not None:
            facts.extend(self.simulation.describe())
        if self.identified_effect is not None:
            facts.extend(self.identified_effect._result_summary_lines(detail=detail))
        else:
            facts.extend(protocol_summary_lines(None, detail))
        if self.data.cluster is not None:
            # The count the status reads, when it differs from the count of labels: a
            # cluster with zero weight mass contributes nothing to any estimate.
            count = f"{self.data.n_clusters}" + positive_mass_clause(
                self.data.cluster, self.data.weights if self.data.is_weighted else None
            )
            facts.append(f"clusters = {count} ({self.data.cluster_name}, cluster-robust variance)")
        if self.data.is_weighted:
            report = self.data.weight_report()
            facts.append(
                f"observation weights ({report.name or 'weights'}, "
                + ("estimated" if report.estimated else "fixed")
                + f"): effective n = {report.effective_n:.1f}, "
                f"design effect = {report.design_effect:.2f}"
            )
            facts.append(
                "estimand: the parameter in the weight-tilted population dP_w = w dP / E[w]; "
                "see result.data.weight_report()"
            )
        facts.extend(self.provenance.describe())
        lines = [
            f"Longitudinal {_ESTIMATOR_TITLE.get(self.config.estimator, 'TMLE')} "
            f"({self.data.n_times} time points, n = {self.n})",
            "",
            table,
            *(["", *support_notes(self.estimates)] if support_cells is not None else []),
            # The refusal's own reason, as ``TMLEResult.summary`` prints it.
            *(["", record.summary_note()] if record is not None else []),
            *(
                line
                for marked_record in marked.values()
                for line in (
                    "",
                    "* "
                    + marked_record.reason
                    + f" A starred spread above is the {marked_record.summary_label}, a "
                    "diagnostic and not a standard error.",
                )
            ),
            *(
                [
                    "",
                    T_REFERENCE_NOTE.format(
                        clusters=fewest_clusters(
                            self.data.cluster,
                            weights=self.data.weights if self.data.is_weighted else None,
                        ),
                        stratum="",
                    ),
                ]
                if t_reference and self.data.cluster is not None
                else []
            ),
            *(
                split_noise_lines(self.estimates, self.repeat_spread(), self.n_repeats, status)
                if self.n_repeats > 1
                else []
            ),
            "",
            *(f"  {line}" for line in facts),
        ]
        if self.n_repeats > 1:
            # The leverage below reads one fitted recursion, so it says which.
            lines.append(
                f"  leverage of {format_draw(REPORTED_DRAW, self.n_repeats)} "
                "(every draw is in result.repeats):"
            )
        # One line per regimen, from the fit that runs to the last node: "throughout"
        # means through the study, and a horizon-k fit stops at k, where the same
        # sentence would be false.  Per-horizon leverage is what diagnostics() is for.
        # A competing-risks fit holds one fit per regimen per cause.  The followers and the
        # weights come from the regimen's treatment and censoring mechanism, which every
        # cause shares, so the causes give one line; a cause is named only where its
        # line differs from another cause's.
        deepest = max(fit.horizon for fit in self.fits.values())
        followers: dict[str, list[tuple[str | None, str]]] = {}
        for fit in self.fits.values():
            if fit.horizon != deepest:
                continue
            truncated, truncated_time = self._max_truncated(fit)
            # An odds tilt keeps every unit at risk, so the count is of units observed.
            kept = (
                "stayed under observation throughout"
                if isinstance(fit.regimen, DynamicRegimen) and fit.regimen.has_tilt
                else "followed it throughout"
            )
            if fit.simulation is not None:
                kept = "followed it throughout in the data, which the simulation does not need"
            followers.setdefault(fit.regimen.label, []).append(
                (
                    fit.cause,
                    f"{int(fit.scored_nodes[-1].trained_on.sum())} of "
                    f"{self.n} units {kept}; max weight {fit.max_weight:.1f}, "
                    f"effective n {fit.effective_n:.0f}; max truncated share "
                    f"{truncated:.1%} at t={truncated_time}",
                )
            )
        for label, entries in followers.items():
            if len({text for _, text in entries}) == 1:
                lines.append(f"  {label}: {entries[0][1]}")
                continue
            lines.extend(f"  {label} ({cause}): {text}" for cause, text in entries)
        lines.extend(f"  {line}" for line in self._targeting_lines())
        if (
            self.simultaneous is None
            and self.n_repeats > 1
            and self.config.isotonic
            and len(self.estimates) > 1
        ):
            lines.append("")
            lines.append(f"  {NO_ISOTONIC_REPEAT_BANDS}")
        elif self.simultaneous is None and record is not None and len(self.estimates) > 1:
            # Stated rather than left blank, as on a point-treatment fit: the bands are on
            # by default, so a reader would otherwise have to guess why none are here.
            lines.append("")
            lines.append(f"  {NO_SIMULTANEOUS_BANDS}")
        elif self.simultaneous is None and t_reference and len(self.estimates) > 1:
            lines.append("")
            lines.append(f"  {NO_T_REFERENCE_BANDS}")
        if self.simultaneous is not None:
            lines.append("")
            lines.append(
                f"  simultaneous {level} bands (multiplier bootstrap, critical value "
                f"{self.simultaneous.critical_value:.3f} vs "
                f"{self.simultaneous.pointwise_critical_value:.3f} pointwise)"
                f"{' [experimental]' if self.simultaneous.support.label == 'experimental' else ''}:"
            )
            for name, (low, high) in self.simultaneous.bands.items():
                lines.append(f"    {name}  [{low:.4f}, {high:.4f}]")
        if self.bootstrap is not None:
            lines.append("")
            lines.append(
                f"  full-refit bootstrap ({self.bootstrap.resampling} resampling, "
                f"{self.bootstrap.n_requested - self.bootstrap.n_failed} usable replicates, "
                f"{self.bootstrap.n_failed} failed):"
            )
            for name, estimate in self.estimates.items():
                if estimate.bootstrap is not None:
                    lines.append(f"    {name}  {bootstrap_line(estimate)}")
        if self.bootstrap_band is not None:
            band = self.bootstrap_band
            lines.append("")
            marker = " [experimental]" if band.support.label == "experimental" else ""
            lines.append(
                f"  studentized bootstrap sup-t bands ({level}, critical value "
                f"{band.critical_value:.3f}, {band.n_replicates} replicates){marker}:"
            )
            for name, (low, high) in band.bands.items():
                lines.append(f"    {name}  [{low:.4f}, {high:.4f}]")
        rule = self._declared_rule()
        if rule != "relative":
            # Under a declared stopping rule the largest score residual in standard errors,
            # beside the coverage a Wald interval keeps at it.
            residual = residual_line(self.estimates, rule, self.config.alpha_sig)
            if residual:
                lines.append("")
                lines.append(f"  {residual}")
        if not self.converged:
            suffix = " in at least one draw" if self.n_repeats > 1 else ""
            if rule == "relative":
                lines.append(
                    "  warning: at least one node's targeting step did not converge" + suffix
                )
            else:
                missed = self._rule_misses()
                failure = (
                    "; ".join(missed) + ". See res.fits"
                    if missed
                    else "at least one node's targeting step did not converge. See res.fits"
                )
                lines.append(f"  {rule_failure_line(rule, failure)}{suffix}")
        return "\n".join(lines)

    def _declared_rule(self) -> str:
        """The stopping rule the fit declared, ``"relative"`` when it declared none."""
        for fit in self.fits.values():
            if fit.hazard is not None and fit.hazard.stopping != "relative":
                return fit.hazard.stopping
            for step in fit.steps:
                if step.fluctuation is not None and step.fluctuation.stopping != "relative":
                    return step.fluctuation.stopping
        return "relative"

    def _rule_misses(self) -> list[str]:
        """Each node whose largest score over its threshold is above one, once per solve.

        A working model solves one stacked fluctuation per node for every cell it holds, so
        the cells share the solve and it is named once.
        """
        seen: set[int] = set()
        missed: list[str] = []
        for key, fit in self.fits.items():
            record = fit.hazard
            if (
                record is not None
                and id(record) not in seen
                and record.rule_ratio is not None
                and record.rule_ratio > 1.0
            ):
                seen.add(id(record))
                where = "every horizon" if record.algorithm == "one_step" else f"t={fit.horizon}"
                missed.append(
                    f"the hazard targeting of {where} is {record.rule_ratio:.3g} times its "
                    "threshold"
                )
            for step in fit.steps:
                fluctuation = step.fluctuation
                if (
                    fluctuation is None
                    or id(fluctuation) in seen
                    or fluctuation.rule_ratio is None
                    or fluctuation.rule_ratio <= 1.0
                ):
                    continue
                seen.add(id(fluctuation))
                missed.append(
                    f"{key} at t={step.time} is {fluctuation.rule_ratio:.3g} times its threshold"
                )
        return missed

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"LongitudinalResult({', '.join(self.estimates)})"

    def _targeting_lines(self) -> list[str]:
        """What the hazard targeting and the isotonic projection did, one line each."""
        lines: list[str] = []
        records = {id(fit.hazard): fit.hazard for fit in self.fits.values() if fit.hazard}
        if records:
            converged = all(record.converged for record in records.values())
            largest = max(record.relative_score for record in records.values())
            verdict = "converged" if converged else "not converged"
            first = next(iter(records.values()))
            if first.algorithm == "one_step":
                lines.append(
                    f"hazard targeting: one step over {len(first.targets)} parameter(s), "
                    f"{first.n_iter} step(s), largest relative score {largest:.2g}, {verdict}"
                )
            else:
                steps = max(record.n_iter for record in records.values())
                lines.append(
                    f"hazard targeting: per horizon over {len(records)} horizon(s), at most "
                    f"{steps} iteration(s), largest relative score {largest:.2g}, {verdict}"
                )
            if first.share_clipped:
                node, cause, share = max(first.share_clipped, key=lambda item: item[2])
                lines.append(
                    f"hazard node bounds: largest clipped share of initial hazards "
                    f"{100 * share:.1f}% (node {node}, cause {cause + 1}); "
                    f"{len(first.bound_excludes_data)} cell(s) excluded by a bound"
                )
        if self.config.isotonic:
            lines.append(f"isotonic projection moved {len(self.isotonic_moves)} level(s)")
        return lines

    @staticmethod
    def _max_truncated(fit: RegimenFit) -> tuple[float, int]:
        """Largest on-score truncation share and its earliest node.

        Of the out-of-fold mechanism the fit retained, which is the one its targeting
        divided by.
        """
        shares = []
        for node in fit.scored_nodes:
            clipped = _clipped_prefix(
                fit.cumulative_unbounded, fit.cumulative, node.time, node.trained_on
            )
            shares.append((float(np.mean(clipped)) if clipped.size else 0.0, node.time))
        return max(shares, key=lambda item: (item[0], -item[1]))


def _run_recursion(
    data: LongitudinalData,
    plans: Sequence[Plan],
    mechanism: Mechanism,
    *,
    msm: RegimenMSM | None,
    horizons: Sequence[int],
    outcome_learner: Learner,
    pseudo_learner: Learner,
    folds: Folds,
    scaler: OutcomeScaler,
    g_bounds: tuple[float, float],
    alpha: float | None,
    max_iter: int | None,
    tol: float | None,
    n_jobs: int,
    estimator: str = "tmle",
    fold_mechanisms: Sequence[FoldMechanism] = (),
    stopping: str = "relative",
    step_bound: float | None = None,
    step_search: bool = True,
    iterate: str = "last",
    normalization: str | None = None,
    parametric: bool = True,
    gformula: GFormulaSettings | None = None,
    node_bounds: dict[str | None, dict[int, NodeBound]] | None = None,
) -> tuple[dict[str, RegimenFit], tuple[MSMRegimenFit, ...]]:
    """Run the backward recursion over every reported cell, and file the fits by key.

    The one driver :meth:`LTMLE.fit` and :func:`_refit_bound` share.  The statistically
    dangerous arithmetic was never duplicated -- both paths call the same
    :func:`~cleverly.longitudinal.sequential.fit_regimen` and
    :func:`~cleverly.longitudinal.msm.fit_regimens_msm` -- but the *driver* was: the same
    nine solver settings and the same plan, cause and horizon ordering, written twice.  One
    driver is what makes a divergence between the two unrepresentable: a setting reaches the
    recursion once, so neither path can pass it differently from the other.

    Note what one driver does not protect.  Those functions default every solver argument,
    and the settings below reach them as a splatted dict.  So dropping one leaves *both*
    paths on that function's default in silence, and no signature here turns that into a
    ``TypeError``.  That would need the defaults removed in
    :mod:`cleverly.longitudinal.sequential` and :mod:`cleverly.longitudinal.msm`, where the
    nine arguments are declared.

    Parameters
    ----------
    data : LongitudinalData
        Validated time-ordered study data.
    plans : sequence of Plan
        Resolved regimens, each with the arms it assigned this sample.
    mechanism : Mechanism
        The fitted treatment and observation mechanisms.
    msm : RegimenMSM or None
        The *evaluated* working model, or ``None`` for a mean per regimen.
    horizons : sequence of int
        The nodes a parameter is reported at.
    outcome_learner : object
        Learner for the regression at the reported horizon.
    pseudo_learner : object
        Learner for the intermediate pseudo-outcome regressions.
    folds : Folds
        Realized outer-fold assignment.
    scaler : OutcomeScaler
        Transformation between observed and targeting scales.
    g_bounds : tuple of float
        Truncation applied to each cumulative mechanism probability.
    alpha : float or None
        Probability shrink applied before the logit.  ``None`` on an SDR fit.
    max_iter : int or None
        Maximum targeting iterations per node.  ``None`` on an SDR fit.
    tol : float or None
        Targeting tolerance per node.  ``None`` on an SDR fit.
    n_jobs : int
        Worker count for the parallel fan-out.
    estimator : {"tmle", "sdr", "ipsi", "gcomp", "ipw", "parametric_gformula"}
        Which recursion runs.
    fold_mechanisms : sequence of FoldMechanism
        Each outer fold's own mechanism, on a cross-fitted SDR or IPSI fit.
    stopping : {"relative", "scaled_var", "sample_size"}
        The rule that decides each node's score equation is solved.  Only a TMLE fit reads it.
    step_bound : float or None
        The longest step of one iteration of a node's targeting.  ``None`` takes the full Newton
        step.  Only a TMLE fit reads it.
    step_search : bool
        Whether a bounded step searches its length.
    iterate : {"last", "best"}
        Which iterate of each node's walk is reported.  Only a TMLE fit reads it.
    normalization : str or None
        The IPW normalization, on an IPW fit.
    parametric : bool
        On a g-computation or IPW fit, whether the curve is the stacked one.  The status reads
        the same flag, so the two agree.
    gformula : GFormulaSettings or None
        On a parametric g-formula fit, the simulation settings.
    node_bounds : dict or None
        On a TMLE fit, the resolved bound of each bounded node, by cause (``None`` is the
        one cause of a fit without competing risks) and one-based node.  Only a TMLE fit
        reads it.

    Returns
    -------
    tuple
        The fits keyed by :func:`_fit_key`, and the projection fits, empty without a
        working model.
    """
    if estimator == "parametric_gformula":
        assert msm is None and gformula is not None
        return _run_gformula(
            data,
            plans,
            mechanism,
            settings=gformula,
            scaler=scaler,
            g_bounds=g_bounds,
            horizons=horizons,
        ), ()
    if estimator in _BASELINES:
        assert msm is None
        return _run_baseline(
            data,
            plans,
            mechanism,
            horizons=horizons,
            outcome_learner=outcome_learner,
            pseudo_learner=pseudo_learner,
            scaler=scaler,
            g_bounds=g_bounds,
            estimator=estimator,
            normalization=normalization,
            parametric=parametric,
        ), ()
    if estimator == "ipsi":
        return _run_ipsi(
            data,
            plans,
            mechanism,
            msm=msm,
            horizons=horizons,
            outcome_learner=outcome_learner,
            pseudo_learner=pseudo_learner,
            folds=folds,
            scaler=scaler,
            g_bounds=g_bounds,
            fold_mechanisms=fold_mechanisms,
            n_jobs=n_jobs,
        )
    if estimator == "sdr":
        return _run_sdr(
            data,
            plans,
            mechanism,
            msm=msm,
            horizons=horizons,
            outcome_learner=outcome_learner,
            pseudo_learner=pseudo_learner,
            folds=folds,
            scaler=scaler,
            g_bounds=g_bounds,
            fold_mechanisms=fold_mechanisms,
            n_jobs=n_jobs,
        )
    assert alpha is not None and max_iter is not None and tol is not None
    recursion = {
        "outcome_learner": outcome_learner,
        "pseudo_learner": pseudo_learner,
        "folds": folds,
        "scaler": scaler,
        "g_bounds": g_bounds,
        "alpha": alpha,
        "max_iter": max_iter,
        "tol": tol,
        "n_jobs": n_jobs,
        "stopping": stopping,
        "step_bound": step_bound,
        "step_search": step_search,
        "iterate": iterate,
    }
    # ``None`` on a single-event or end-of-study fit, so the comprehension below is one
    # loop deeper for every fit and a second code path for none.
    causes: tuple[str | None, ...] = data.cause_labels or (None,)
    ratios = {plan.label: node_ratios(data, plan, mechanism) for plan in plans}
    if msm is None:
        return {
            _fit_key(plan.label, cause, horizon, data.reports_curve): _with_node_ratio(
                fit_regimen(
                    data,
                    plan,
                    mechanism,
                    horizon=horizon,
                    cause=cause,
                    node_bounds=None if node_bounds is None else node_bounds.get(cause),
                    **recursion,
                ),
                ratios[plan.label],
                mechanism,
            )
            # Regimen-outer, then cause, then horizon, so the report reads down one curve
            # at a time rather than across the regimens at each time.
            for plan in plans
            for cause in causes
            for horizon in horizons
        }, ()
    # The per-cell fits come back so that ``diagnostics()`` can report the leverage and
    # risk sets, which are questions about a regimen and are the same questions whether or
    # not a working model summarises it.
    msm_fits = tuple(
        fit_regimens_msm(data, plans, mechanism, msm, cause=cause, **recursion) for cause in causes
    )
    return {
        _fit_key(fit.regimen.label, fit.cause, fit.horizon, data.reports_curve): _with_node_ratio(
            fit, ratios[fit.regimen.label], mechanism
        )
        for msm_fit in msm_fits
        for fit in msm_fit.fits
    }, msm_fits


def _run_hazard(
    data: LongitudinalData,
    plans: Sequence[Plan],
    levels: Sequence[Plan],
    mechanism: Mechanism,
    initial: InitialHazards,
    *,
    horizons: Sequence[int],
    g_bounds: tuple[float, float],
    alpha: float,
    algorithm: HazardAlgorithm,
    max_iter: int,
    tol: float,
    step_size: float,
    node_bounds: dict[str | None, dict[int, NodeBound]] | None = None,
    stopping: str = "relative",
    step_bound: float | None = None,
    step_search: bool = True,
    iterate: str = "last",
) -> dict[str, RegimenFit]:
    """Target the initial hazards and file one fit per regimen, cause and horizon.

    The one driver :meth:`LTMLE.fit` and :func:`_refit_bound` share on a hazard fit.

    Parameters
    ----------
    data : LongitudinalData
        A survival panel of a held design.
    plans : sequence of Plan
        The regimens' plans.
    levels : sequence of Plan
        One static plan per treatment level, from
        :func:`~cleverly.longitudinal.hazard.level_plans`.
    mechanism : Mechanism
        The mechanism, evaluated at every regimen and every level.
    initial : InitialHazards
        The initial hazards.
    horizons : sequence of int
        The reported horizons.
    g_bounds : tuple of float
        Truncation applied to each cumulative mechanism probability.
    alpha : float
        The probability bound of every hazard.
    algorithm : {"per_horizon", "one_step"}
        The targeting algorithm.
    max_iter : int
        Outer iterations per horizon of a per-horizon fit.
    tol : float
        Relative-score tolerance.
    step_size : float
        The first step of a one-step walk.
    node_bounds : dict or None
        The resolved hazard-scale bounds by cause and one-based node, ``None`` when no node
        is bounded.
    stopping : {"relative", "scaled_var", "sample_size"}
        The stopping rule of the targeting.  A declared rule reads the data's clusters as
        its units, with a span of one, since a hazard lies on the ``[0, 1]`` scale.
    step_bound : float or None
        The longest per-horizon step in ``epsilon``.
    step_search : bool
        Whether a bounded step searches its length.
    iterate : {"last", "best"}
        Which iterate the targeting reports.

    Returns
    -------
    dict of str to RegimenFit
        The fits keyed by :func:`_fit_key`.
    """
    pairs = [mechanism.cumulative_with_unbounded(data, level, g_bounds) for level in levels]
    rows = np.arange(data.n)
    observed = np.asarray(np.nan_to_num(data.treatment[:, 0], nan=0.0), dtype=np.int64)
    unbounded = np.stack([pair[0] for pair in pairs], axis=2)[rows, :, observed]
    bounded = np.stack([pair[1] for pair in pairs], axis=2)[rows, :, observed]
    fits = target_regimens(
        data,
        plans,
        initial,
        [pair[1] for pair in pairs],
        (unbounded, bounded),
        horizons=horizons,
        algorithm=algorithm,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        step_size=step_size,
        node_bounds=hazard_bounds(node_bounds, data.n_times, data.cause_labels or (None,)),
        stopping=stopping,
        stop=None if stopping == "relative" else StopContext(data.cluster, 1.0),
        step_bound=step_bound,
        step_search=step_search,
        iterate=iterate,
    )
    return {
        _fit_key(label, cause, horizon, True): fit for (label, cause, horizon), fit in fits.items()
    }


def _project_fits(
    fits: Mapping[str, RegimenFit],
) -> tuple[dict[str, RegimenFit], tuple[tuple[str, str | None, int, float, float], ...]]:
    """Project each regimen's and cause's levels onto a non-decreasing curve.

    Unit-weight isotonic regression over the reported horizons, in horizon order
    (Westling, van der Laan and Carone 2020, Section 2.1).  Only the level moves: the
    influence curve, and so every standard error, is the fit's own.

    Parameters
    ----------
    fits : mapping of str to RegimenFit
        The fits keyed by :func:`_fit_key`.

    Returns
    -------
    tuple
        The fits with projected levels, and ``(regimen, cause, horizon, before, after)`` of
        every level the projection moved.
    """
    curves: dict[tuple[str, str | None], list[str]] = {}
    for key, fit in fits.items():
        curves.setdefault((fit.regimen.label, fit.cause), []).append(key)
    out = dict(fits)
    moves: list[tuple[str, str | None, int, float, float]] = []
    for (label, cause), keys in curves.items():
        keys = sorted(keys, key=lambda key: fits[key].horizon)
        values = np.array([fits[key].psi_scaled for key in keys])
        projected = project_curves(values[None, :])[0]
        for key, before, after in zip(keys, values, projected, strict=True):
            if after != before:
                out[key] = replace(fits[key], psi_scaled=float(after))
                moves.append((label, cause, fits[key].horizon, float(before), float(after)))
    return out, tuple(moves)


def _project_bands(
    bands: SimultaneousBands | None,
    index: Mapping[str, tuple[str, str | None, int]] | None,
) -> SimultaneousBands | None:
    """Project the band limits of every level curve, and keep every contrast's band.

    Each curve's lower limits and upper limits are projected by the same unit-weight
    isotonic regression as its levels (Westling et al. 2020, Section 2.1 and Theorem 1
    (iv)-(v)): the projected band covers whenever the band it came from covers, and is no
    wider on average or at its maximum.  A contrast keeps its band as built.

    Parameters
    ----------
    bands : SimultaneousBands or None
        The band built at the unprojected levels.
    index : mapping or None
        ``name -> (regimen, cause, horizon)`` of every reported name.

    Returns
    -------
    SimultaneousBands or None
        The band with projected level limits.
    """
    if bands is None or index is None:
        return bands
    curves: dict[tuple[str, str | None], list[tuple[int, str]]] = {}
    for name in bands.bands:
        if name.startswith(CONTRAST_HEAD) or name not in index:
            continue
        label, cause, horizon = index[name]
        curves.setdefault((label, cause), []).append((horizon, name))
    limits = dict(bands.bands)
    for members in curves.values():
        names = [name for _, name in sorted(members)]
        lower = project_curves(np.array([[bands.bands[name][0] for name in names]]))[0]
        upper = project_curves(np.array([[bands.bands[name][1] for name in names]]))[0]
        for name, low, high in zip(names, lower, upper, strict=True):
            limits[name] = (float(low), float(high))
    return replace(bands, bands=limits)


def _run_baseline(
    data: LongitudinalData,
    plans: Sequence[Plan],
    mechanism: Mechanism,
    *,
    horizons: Sequence[int],
    outcome_learner: Learner,
    pseudo_learner: Learner,
    scaler: OutcomeScaler,
    g_bounds: tuple[float, float],
    estimator: str,
    normalization: str | None,
    parametric: bool,
) -> dict[str, RegimenFit]:
    """The g-computation and IPW branch of :func:`_run_recursion`, in the same cell order.

    Each cell is fitted in sample and alone: neither estimator pools a step across cells.
    """
    causes: tuple[str | None, ...] = data.cause_labels or (None,)
    ratios = {plan.label: node_ratios(data, plan, mechanism) for plan in plans}
    fits: dict[str, RegimenFit] = {}
    for plan in plans:
        for cause in causes:
            for horizon in horizons:
                if estimator == "gcomp":
                    fit = fit_gcomp_cell(
                        data,
                        plan,
                        mechanism,
                        scaler=scaler,
                        g_bounds=g_bounds,
                        horizon=horizon,
                        cause=cause,
                        outcome_learner=outcome_learner,
                        pseudo_learner=pseudo_learner,
                        stacked=parametric,
                    )
                else:
                    assert normalization is not None
                    fit = fit_ipw_cell(
                        data,
                        plan,
                        mechanism,
                        scaler=scaler,
                        g_bounds=g_bounds,
                        horizon=horizon,
                        cause=cause,
                        normalization=normalization,
                        stacked=parametric,
                    )
                key = _fit_key(plan.label, cause, horizon, data.reports_curve)
                fits[key] = _with_node_ratio(fit, ratios[plan.label], mechanism)
    return fits


def _run_gformula(
    data: LongitudinalData,
    plans: Sequence[Plan],
    mechanism: Mechanism,
    *,
    settings: GFormulaSettings,
    scaler: OutcomeScaler,
    g_bounds: tuple[float, float],
    horizons: Sequence[int],
) -> dict[str, RegimenFit]:
    """The parametric g-formula branch of :func:`_run_recursion`, in the same cell order.

    The models are fitted once and every plan reads the same uniforms, so a contrast is the
    difference of two cells that share their simulation error.
    """
    fits, _ = run_gformula(
        data,
        plans,
        mechanism,
        settings=settings,
        scaler=scaler,
        g_bounds=g_bounds,
        horizons=horizons,
    )
    ratios = {plan.label: node_ratios(data, plan, mechanism) for plan in plans}
    return {
        _fit_key(label, cause, horizon, data.reports_curve): _with_node_ratio(
            fit, ratios[label], mechanism
        )
        for (label, cause, horizon), fit in fits.items()
    }


def _run_sdr(
    data: LongitudinalData,
    plans: Sequence[Plan],
    mechanism: Mechanism,
    *,
    msm: RegimenMSM | None,
    horizons: Sequence[int],
    outcome_learner: Learner,
    pseudo_learner: Learner,
    folds: Folds,
    scaler: OutcomeScaler,
    g_bounds: tuple[float, float],
    fold_mechanisms: Sequence[FoldMechanism],
    n_jobs: int,
) -> tuple[dict[str, RegimenFit], tuple[MSMRegimenFit, ...]]:
    """The SDR branch of :func:`_run_recursion`, in the same cell order.

    Every cell runs in one call, so a cross-fitted fit has one job per outer fold.  With a
    working model, each cause's cells are projected by
    :func:`~cleverly.longitudinal.msm.project_sdr_cells`.
    """
    causes: tuple[str | None, ...] = data.cause_labels or (None,)
    ratios = {plan.label: node_ratios(data, plan, mechanism) for plan in plans}
    cells = [
        SDRCell(plan.label, cause, horizon)
        for plan in plans
        for cause in causes
        for horizon in horizons
    ]
    fitted = fit_cells_sdr(
        data,
        plans,
        mechanism,
        cells,
        outcome_learner=outcome_learner,
        pseudo_learner=pseudo_learner,
        folds=folds,
        scaler=scaler,
        g_bounds=g_bounds,
        fold_mechanisms=fold_mechanisms,
        n_jobs=n_jobs,
    )
    fits = {
        _fit_key(cell.label, cell.cause, cell.horizon, data.reports_curve): _with_node_ratio(
            fit, ratios[cell.label], mechanism
        )
        for cell, fit in zip(cells, fitted, strict=True)
    }
    if msm is None:
        return fits, ()
    # The projection reads the cells in the working model's own order, which is
    # horizon-outer, not the regimen-outer order of the report.
    by_cell = {
        (cell.label, cell.cause, cell.horizon): fit for cell, fit in zip(cells, fitted, strict=True)
    }
    msm_fits = tuple(
        project_sdr_cells(
            data,
            msm,
            [by_cell[(cell.label, cause, cell.horizon)] for cell in msm.cells],
            scaler,
            cause=cause,
        )
        for cause in causes
    )
    return fits, msm_fits


def _run_ipsi(
    data: LongitudinalData,
    plans: Sequence[Plan],
    mechanism: Mechanism,
    *,
    msm: RegimenMSM | None,
    horizons: Sequence[int],
    outcome_learner: Learner,
    pseudo_learner: Learner,
    folds: Folds,
    scaler: OutcomeScaler,
    g_bounds: tuple[float, float],
    fold_mechanisms: Sequence[FoldMechanism],
    n_jobs: int,
) -> tuple[dict[str, RegimenFit], tuple[MSMRegimenFit, ...]]:
    """The IPSI branch of :func:`_run_recursion`, in the same cell order.

    Every cell runs in one call, so a cross-fitted fit has one job per outer fold.  With a
    working model, each cause's cells are projected by
    :func:`~cleverly.longitudinal.msm.project_sdr_cells`, which reads each cell's
    :math:`\\varphi` as it reads an SDR transform.
    """
    causes: tuple[str | None, ...] = data.cause_labels or (None,)
    cells = [
        IPSICell(plan.label, cause, horizon)
        for plan in plans
        for cause in causes
        for horizon in horizons
    ]
    fitted = fit_cells_ipsi(
        data,
        plans,
        mechanism,
        cells,
        outcome_learner=outcome_learner,
        pseudo_learner=pseudo_learner,
        folds=folds,
        scaler=scaler,
        g_bounds=g_bounds,
        fold_mechanisms=fold_mechanisms,
        n_jobs=n_jobs,
    )
    fits = {
        _fit_key(cell.label, cell.cause, cell.horizon, data.reports_curve): fit
        for cell, fit in zip(cells, fitted, strict=True)
    }
    if msm is None:
        return fits, ()
    by_cell = {
        (cell.label, cell.cause, cell.horizon): fit for cell, fit in zip(cells, fitted, strict=True)
    }
    msm_fits = tuple(
        project_sdr_cells(
            data,
            msm,
            [by_cell[(cell.label, cause, cell.horizon)] for cell in msm.cells],
            scaler,
            cause=cause,
            estimator="ipsi",
        )
        for cause in causes
    )
    return fits, msm_fits


def _with_node_ratio(fit: RegimenFit, ratio: FloatArray | None, mechanism: Mechanism) -> RegimenFit:
    """The fit with its plan's raw per-node ratio and, under ``ratio_trim=``, its trim."""
    trim = mechanism.ratio_trims.get(fit.regimen.label)
    if ratio is None and trim is None:
        return fit
    return replace(fit, node_ratio=ratio, ratio_trim=trim)


@dataclass(frozen=True)
class _Reported:
    """What one recursion reports: the names, their index, and the fits behind each.

    ``contributors`` is the reason this is an object rather than a pair.  The fits behind a
    reported name are known *where the name is composed* and nowhere else: a level is one
    fit, a contrast is its two, and a working model's coefficient is every cell of its
    cause.  Recovering that association afterwards means matching a composed name against
    a second composition of the same vocabulary, which is the parsing this module refuses
    -- and which fails for every estimand at once the day the emitter gains an alias.

    Parameters
    ----------
    estimates : dict of str to ParameterEstimate
        Estimates keyed by the reported name, in report order.
    index : dict or None
        ``name -> (regimen, cause, horizon)``, ``None`` on a working-model fit whose
        parameters are indexed by term.
    contributors : dict of str to tuple of RegimenFit
        The fits that produced each reported name.
    """

    estimates: dict[str, ParameterEstimate]
    index: dict[str, tuple[str, str | None, int]] | None
    contributors: dict[str, tuple[RegimenFit, ...]]


def _inference_status(
    data: LongitudinalData,
    folds: Folds,
    *,
    estimator: str = "tmle",
    parametric: bool = True,
) -> InferenceStatus:
    """The fit status that every estimate of a longitudinal fit starts from.

    :func:`_parameter_status` may replace it on a parameter whose recursion read a constant
    node regression; every other estimate keeps it.

    The point-treatment cluster rule applies to the prepared cluster labels and, on a weighted fit,
    to the unit weights. It reads nothing fitted. ``LongitudinalData`` carries no baseline strata,
    so the count is the number of clusters with positive weight mass in the whole fit, and
    :data:`~cleverly._inference_status.MINIMUM_LONGITUDINAL_INTERVAL_CLUSTERS` is the
    threshold, in sample and cross-fitted: the registered few-cluster studies measure
    ``LTMLE`` from 20 clusters, in sample (``clustered-few-cluster-tmle``) and cross-fitted
    (``few-cluster-cross-fitted-ltmle``).

    The targeting pools every follower, so the curve is centred at the pooled estimate and
    the folds do not enter the status or the degrees of freedom.  This status can be
    ``"few_cluster_plugin"`` only, and :func:`_trimmed_status` adds the trimmed-ratio status
    of a ``ratio_trim=`` fit. ``LTMLE.fit`` and the
    truncation-curve replay ``_refit_bound`` pass the status to :func:`_estimates` or
    :func:`_msm_estimates`, which also set the Student t reference of an inferential fit
    with fewer than :data:`~cleverly._inference_status.FEW_CLUSTER_THRESHOLD` clusters.

    A g-computation or IPW fit adds two statuses.  It takes ``"adaptive_baseline_plugin"``
    when a regression it reads is not a :class:`~cleverly.learners.GLM`, and
    ``"estimated_weight_plugin"`` with weights declared estimated, because its stacked curve
    conditions on the weights.  :func:`~cleverly._inference_status.precedent_status` orders
    them.

    Parameters
    ----------
    data : LongitudinalData
        The prepared data of the fit.
    folds : Folds
        The outer fold assignment of the fit.
    estimator : str
        The recursion the fit ran.
    parametric : bool
        On a g-computation or IPW fit, whether every regression it reads is a GLM.

    Returns
    -------
    str
        One of :data:`~cleverly.inference.influence.InferenceStatus`.
        ``"influence_curve"`` on an unclustered fit.
    """
    cluster = cluster_inference_status(
        data.cluster,
        weights=data.weights if data.is_weighted else None,
        minimum=MINIMUM_LONGITUDINAL_INTERVAL_CLUSTERS,
    )
    if estimator not in _BASELINES:
        return cluster
    return precedent_status(
        (
            cluster,
            "influence_curve" if parametric else ADAPTIVE_BASELINE_STATUS,
            "estimated_weight_plugin" if data.weight_spec.estimated else "influence_curve",
        )
    )


def _trimmed_status(inference: InferenceStatus, mechanism: Mechanism) -> InferenceStatus:
    """The fit status after a ratio trim: the trimmed-ratio status when the trim bound.

    A fit with ``ratio_trim=`` whose trim moved a positive per-node ratio of any plan takes
    :data:`~cleverly._inference_status.TRIMMED_RATIO_STATUS`, unless an earlier status in the
    table applies.  The trimmed share is the one fitted quantity a fit status reads.
    ``LTMLE.fit`` and the truncation-curve replay both apply it, to the same mechanism.

    Parameters
    ----------
    inference : str
        The status of :func:`_inference_status`.
    mechanism : Mechanism
        The fitted mechanism, with its trims attached.

    Returns
    -------
    str
        ``inference``, or the trimmed-ratio status by precedence.
    """
    if any(trim.trimmed_share > 0.0 for trim in mechanism.ratio_trims.values()):
        return precedent_status((inference, TRIMMED_RATIO_STATUS))
    return inference


def _reference_df(data: LongitudinalData, inference: InferenceStatus) -> int | None:
    """The Student t degrees of freedom of every estimate of a longitudinal fit.

    ``LongitudinalData`` carries no baseline strata, so every estimate reads every unit.
    ``None`` on an unclustered fit, on a fit whose status supplies no inference, and at
    :data:`~cleverly._inference_status.FEW_CLUSTER_THRESHOLD` clusters or more.

    Parameters
    ----------
    data : LongitudinalData
        The prepared data of the fit.
    inference : str
        The status the fit stamps on its estimates.

    Returns
    -------
    int or None
        :func:`~cleverly.inference.cluster.cluster_reference_df` of the fit, or ``None``.
    """
    if data.cluster is None or not supplies_inference(inference):
        return None
    return cluster_reference_df(data.cluster, data.weights if data.is_weighted else None)


def _estimates(
    data: LongitudinalData,
    fits: Mapping[str, RegimenFit],
    scaler: OutcomeScaler,
    reference: RegimenSpec,
    *,
    alpha_sig: float,
    inference: InferenceStatus,
    classification: Classification | None = None,
) -> _Reported:
    """Compose every reported name, its estimate, its index and its fits, in one pass.

    Parameters
    ----------
    data : LongitudinalData
        Validated time-ordered study data.
    fits : mapping of str to RegimenFit
        The fits this recursion produced, keyed by :func:`_fit_key`.
    scaler : OutcomeScaler
        Transformation between observed and targeting scales.
    reference : RegimenSpec
        The regimen every contrast is taken against.
    alpha_sig : float
        Significance level of the reported intervals.
    inference : str
        The inference status every estimate declares, from :func:`_inference_status`.
        Stamped here rather than after the call, because the fit and the truncation-curve
        replay both build their estimates here, and the replay must equal the fit field
        for field.

    Returns
    -------
    object
        The estimates, the parameter index, and the fits behind each name.
    """
    survival = data.is_survival
    curve = data.reports_curve
    # ``ey_regimen`` is a mean of the outcome (at each horizon, on a repeated outcome),
    # ``risk_regimen`` a probability of an event by a horizon, and ``cif_regimen`` the
    # probability of leaving through one particular cause by then.  E[Y_k] *is* the second, and
    # with a single cause the
    # second *is* the third -- so one name would not be wrong anywhere.  But the three
    # come from different derivations, and a saved frame or a coverage study's truth
    # dict keyed by name is where that would stop being a distinction without a
    # difference.
    head = _level_head(survival=survival, competing=data.is_competing, complement=False)
    reference_df = _reference_df(data, inference)
    estimates: dict[str, ParameterEstimate] = {}
    # ``name -> (regimen, cause, horizon)``, composed forward here and never parsed
    # back out of the name.  ``curve()`` reads it: with a cause beside the horizon
    # inside one pair of brackets, recovering either by splitting the string would be
    # guessing where a label ends, and a regimen called "a, b" would decide it wrongly.
    index: dict[str, tuple[str, str | None, int]] = {}
    contributors: dict[str, tuple[RegimenFit, ...]] = {}
    for fit in fits.values():
        name = parameter_name(head, arm=_index(fit.regimen.label, fit.cause, fit.horizon, curve))
        index[name] = (fit.regimen.label, fit.cause, fit.horizon)
        contributors[name] = (fit,)
        status = _parameter_status(inference, fit)
        estimates[name] = make_estimate(
            name,
            scaler.unscale_level(fit.psi_scaled),
            scaler.unscale_influence(fit.influence_curve_scaled),
            n=data.n,
            cluster=data.cluster,
            scale="level",
            alpha=alpha_sig,
            inference=status,
            reference_df=reference_df,
            support=_support_of(classification, status, reference_df),
        )
    for fit in fits.values():
        if fit.regimen.label == reference.label:
            continue
        # The contrast is between the same two regimens at the same cause *and* the
        # same horizon; a difference of incidences across either is not a treatment
        # effect, and with two indexes there are now two ways to pair the wrong ones.
        base = fits[_fit_key(reference.label, fit.cause, fit.horizon, curve)]
        contrast = f"{fit.regimen.label} vs {reference.label}"
        name = parameter_name(CONTRAST_HEAD, arm=_index(contrast, fit.cause, fit.horizon, curve))
        index[name] = (contrast, fit.cause, fit.horizon)
        contributors[name] = (fit, base)
        status = _parameter_status(inference, fit, base)
        estimates[name] = make_estimate(
            name,
            scaler.unscale_difference(fit.psi_scaled - base.psi_scaled),
            scaler.unscale_influence(fit.influence_curve_scaled - base.influence_curve_scaled),
            n=data.n,
            cluster=data.cluster,
            scale="difference",
            alpha=alpha_sig,
            inference=status,
            reference_df=reference_df,
            support=_support_of(classification, status, reference_df),
        )
    return _Reported(estimates=estimates, index=index, contributors=contributors)


def _support_of(
    classification: Classification | None, status: InferenceStatus, reference_df: int | None
) -> Support:
    """The support of one estimate: the default without a classification, else the computed one."""
    if classification is None:
        return UNCLASSIFIED
    return support_for(classification, status, reference_df)


def _parameter_status(inference: InferenceStatus, *read: RegimenFit) -> InferenceStatus:
    """The status of a parameter that reads ``read``: the fit's, or the constant-node one.

    A fit with a node regression read as a constant 0 or 1 (:func:`constant_nodes`) has an
    identically zero curve term at that node, so its plug-in standard error leaves that
    node's variance out.  The parameter, and a contrast that reads it, take
    :data:`~cleverly._inference_status.CONSTANT_NODE_STATUS`.  A fit status that withholds
    inference already, such as ``"few_cluster_plugin"``, comes first in the table and so
    stays.  This is the one fitted quantity a status reads.
    """
    if any(constant_nodes(fit) for fit in read):
        return precedent_status((inference, CONSTANT_NODE_STATUS))
    return inference


#: The most parameter names a constant-node warning lists.  A default day-level grid can flag
#: thousands; :func:`constant_node_parameters` returns every one.
_WARNED_NAMES = 10


def _warn_parameter_statuses(estimates: Mapping[str, ParameterEstimate]) -> None:
    """Warn once per parameter status of the fit, with a count and the first names.

    The text after the names is the status record's reason, as ``summary()`` prints it.
    """
    for status in sorted(PARAMETER_STATUSES):
        names = [name for name, estimate in estimates.items() if estimate.inference == status]
        if not names:
            continue
        shown = ", ".join(names[:_WARNED_NAMES])
        more = f", and {len(names) - _WARNED_NAMES} more" if len(names) > _WARNED_NAMES else ""
        _warn(
            f"{len(names)} reported parameter(s) take the {status!r} status: {shown}{more} "
            "(constant_node_parameters(result) lists them). "
            + status_record(status).reason
            + " Report horizons at which the followers of every regimen hold both outcome "
            "values, or declare a coarser grid, for intervals there",
            DataWarning,
            stacklevel=3,
        )


def constant_node_parameters(result: LongitudinalResult) -> tuple[str, ...]:
    """The reported parameters that take the constant-node status.

    The status is :data:`~cleverly._inference_status.CONSTANT_NODE_STATUS`: the parameter's
    recursion read a node regression as a constant 0 or 1, so it reports no interval.

    Parameters
    ----------
    result : LongitudinalResult
        A fitted longitudinal result.

    Returns
    -------
    tuple of str
        The parameter names, in report order.
    """
    return tuple(
        name
        for name, estimate in result.estimates.items()
        if estimate.inference == CONSTANT_NODE_STATUS
    )


def _msm_estimates(
    data: LongitudinalData,
    msm_fits: Sequence[MSMRegimenFit],
    *,
    alpha_sig: float,
    inference: InferenceStatus,
    classification: Classification | None = None,
) -> _Reported:
    """One estimate per working-model term per cause, and no contrasts.

    **Not** unscaled.  ``beta`` and its curve come off the projection already on the
    outcome's own scale, because a coefficient vector has no single
    :class:`~cleverly.inference.Scale` to map back with -- the reason
    :meth:`cleverly.targets.TargetContext.finish_unscaled` exists at one time point.
    Putting them through :meth:`OutcomeScaler.unscale_level` here would apply the map
    twice, and on a binary outcome, where the scaler is the identity, would look
    perfectly fine while doing it.

    No contrasts: a working model reports coefficients, and a difference of two of
    them is ``result.contrast()`` rather than a row nobody asked for.

    Parameters
    ----------
    data : LongitudinalData
        Validated time-ordered study data.
    msm_fits : sequence of MSMRegimenFit
        One projection per cause, in the order the causes are reported.
    alpha_sig : float
        Significance level of the reported intervals.
    inference : str
        The inference status every coefficient declares, for the reason
        :func:`_estimates` gives.
    classification : Classification or None, default None
        The fit's classification, as in :func:`_estimates`.

    Returns
    -------
    object
        The coefficient estimates and the cells behind each, with no parameter index: a
        coefficient is indexed by term rather than by regimen, cause and horizon.
    """
    estimates: dict[str, ParameterEstimate] = {}
    contributors: dict[str, tuple[RegimenFit, ...]] = {}
    reference_df = _reference_df(data, inference)
    for msm_fit in msm_fits:
        for column, term in enumerate(msm_fit.model.terms):
            # Composed forward, as every other name here is: a term may contain a
            # bracket, and a cause may contain the separator.
            inside = term if msm_fit.cause is None else f"{term}{CAUSE_INFIX}{msm_fit.cause}"
            name = f"msm_regimen[{inside}]"
            # Every cell of this cause, because the projection is pooled across them: no
            # coefficient is a statement about one regimen or one horizon alone.
            contributors[name] = tuple(msm_fit.fits)
            estimates[name] = make_estimate(
                name,
                float(msm_fit.beta[column]),
                msm_fit.influence_curves[:, column],
                n=data.n,
                cluster=data.cluster,
                scale="level",
                alpha=alpha_sig,
                inference=inference,
                reference_df=reference_df,
                support=_support_of(classification, inference, reference_df),
            )
    return _Reported(estimates=estimates, index=None, contributors=contributors)


class _SequentialEstimator:
    """The machinery :class:`LTMLE` and :class:`SDR` share.

    Data preparation, plans, the mechanism, the outer split, reporting, the truncation
    replay and the bootstrap.  A subclass names its recursion in :attr:`_estimator` and
    says which targeting settings it holds.  The constructor below takes every shared
    setting.  :class:`LTMLE` adds ``alpha``, ``max_iter`` and ``tol``.
    """

    #: ``"tmle"``, ``"sdr"`` or ``"ipsi"``: the recursion :func:`_run_recursion` runs.
    _estimator: ClassVar[str]
    #: The survival model, the hazard targeting algorithm, its first step and the isotonic
    #: projection.  Only :class:`LTMLE` takes them, so every other estimator keeps the
    #: iterated-mean recursion and reports its curve unprojected.
    survival_model: str = "iterated_mean"
    targeting: str = "iterative"
    step_size: float = 1e-3
    isotonic: bool = False
    #: The declared node bounds.  Only :class:`LTMLE` takes them.
    node_bounds: Mapping[Any, Any] | None = None

    #: Whether a fixed dose, or a known rule that sets a dose, at a continuous node is a
    #: regimen this estimator fits.  Only a model-based estimator accepts one.
    _allow_set_dose: ClassVar[bool] = False
    #: Whether the fit reads a density ratio at a continuous node, which R-b4 of X40 guards.
    _reads_density_ratio: ClassVar[bool] = True

    #: Whether a fit classifies itself for the support record. A bootstrap replicate sets it off.
    _record_support: bool = True
    #: One outer split per draw that a refit uses instead of drawing its own (:meth:`refit`).
    _frozen_folds: tuple[Folds, ...] | None = None
    #: One seed per draw that a refit uses instead of the seeds of ``random_state``.
    _frozen_seeds: tuple[int, ...] | None = None

    def __init__(
        self,
        regimens: Any,
        *,
        reference: str | None = None,
        horizons: Sequence[int] | None = None,
        msm: MSM | None = None,
        outcome_learner: Learner | None = None,
        pseudo_learner: Learner | None = None,
        treatment_learner: Learner | None = None,
        censoring_learner: Learner | None = None,
        n_folds: int = 10,
        learner_folds: int = 5,
        g_bounds: CumulativeGBounds = DEFAULT_LTMLE_G_BOUNDS,
        q_bounds: tuple[float, float] | None = None,
        alpha_sig: float = 0.05,
        simultaneous: bool = True,
        n_multiplier: int = 2000,
        multiplier_kind: str = "rademacher",
        random_state: int | None = None,
        run_id: str | None = None,
        n_jobs: int = 1,
        n_bootstrap: int = 0,
        bootstrap_resampling: Resampling = "auto",
        bootstrap_interval: BootstrapInterval = "percentile",
        density_bins: int | None = None,
        ratio: str = "density",
        ratio_trim: float | None = None,
        hazard_pooling: str = "node",
        repeats: int = 1,
        **refused: Any,
    ) -> None:
        _validate_learner(outcome_learner, "outcome_learner")
        _validate_learner(pseudo_learner, "pseudo_learner")
        _validate_learner(treatment_learner, "treatment_learner")
        _validate_learner(censoring_learner, "censoring_learner")
        self.regimens = regimens
        self.reference = reference
        self.horizons = horizons
        self.msm = msm
        self.outcome_learner = outcome_learner
        self.pseudo_learner = pseudo_learner
        self.treatment_learner = treatment_learner
        self.censoring_learner = censoring_learner
        self.n_folds = n_folds
        self.learner_folds = learner_folds
        self.g_bounds = g_bounds
        self.q_bounds = q_bounds
        self.alpha_sig = alpha_sig
        self.simultaneous = simultaneous
        self.n_multiplier = n_multiplier
        self.multiplier_kind = multiplier_kind
        self.random_state = random_state
        self.run_id = run_id
        self.n_jobs = n_jobs
        self.n_bootstrap = n_bootstrap
        self.bootstrap_resampling = bootstrap_resampling
        self.bootstrap_interval = bootstrap_interval
        self.density_bins = density_bins
        self.ratio = ratio
        self.ratio_trim = ratio_trim
        self.hazard_pooling = hazard_pooling
        self.repeats = repeats
        refuse_unsupported(refused, where=type(self).__name__)
        self._validate_settings()

    def _validate_settings(self) -> None:
        """Refuse settings that would otherwise produce a report meaning something else.

        ``alpha_sig`` is the one worth spelling out: on :class:`LTMLE` it and ``alpha`` are
        a hair apart in name and nowhere near in meaning, so a value that belongs to one and
        is passed as the other has to be refused rather than believed.
        """
        # Stricter than ``TMLE``, which takes the whole of (0, 1), and deliberately: the
        # half this refuses is empty of anything a caller means and is exactly where the
        # shrink lives, so refusing it turns the one swap that has no other symptom into
        # a message.  An interval covering less than half the time is not one either.
        if not 0.0 < self.alpha_sig < 0.5:
            raise ValueError(
                f"alpha_sig must lie in (0, 0.5); got {self.alpha_sig}. It is the "
                "significance level of the reported intervals; alpha= is the "
                "probability shrink applied before the logit, and defaults to 0.9995"
            )
        self._validate_targeting()
        if self.n_folds < 1:
            raise ValueError(f"n_folds must be at least 1; got {self.n_folds}")
        if self.n_bootstrap and self.n_bootstrap < 2:
            raise ValueError(f"n_bootstrap must be 0 or at least 2; got {self.n_bootstrap}")
        if (
            isinstance(self.repeats, bool)
            or not isinstance(self.repeats, (int, np.integer))
            or self.repeats < 1
        ):
            raise ValueError(
                f"repeats must be an integer of at least 1; got {self.repeats!r}. It is how "
                "many independent draws of the cross-fitting split the fit combines by their "
                "median."
            )
        if self.repeats > 1 and self.n_folds <= 1:
            raise MethodConfigurationError(
                "repeats above 1 needs n_folds of 2 or more. One fold fits every nuisance on "
                "every row, so there is no split to draw again. Set n_folds to 2 or more, or "
                "set repeats=1.",
                reason=RefusalReason.INVALID_DECLARATION,
            )
        if self.repeats > 1 and self.n_bootstrap:
            raise MethodConfigurationError(
                "n_bootstrap is not combined with repeats above 1. No registered study "
                "measures a bootstrap of the median over fold draws, and each replicate would "
                "refit R complete draws.",
                reason=RefusalReason.INFERENCE_GAP,
                alternative=(
                    "Fit with repeats=1 for the bootstrap, or with n_bootstrap=0 for the "
                    "split-adjusted median interval."
                ),
            )
        if self.bootstrap_resampling not in ("auto", "iid", "cluster"):
            raise ValueError("resampling must be 'auto', 'iid', or 'cluster'")
        if self.bootstrap_interval not in ("percentile", "bca"):
            raise ValueError(
                f"bootstrap_interval must be 'percentile' or 'bca'; got {self.bootstrap_interval!r}"
            )
        if self.bootstrap_interval != "percentile" and not self.n_bootstrap:
            raise ValueError(
                f"bootstrap_interval={self.bootstrap_interval!r} names an interval of the "
                "full-refit bootstrap, and n_bootstrap=0 runs none. Pass n_bootstrap= too, "
                "such as n_bootstrap=199"
            )
        if self.ratio not in ("density", "classifier"):
            raise ValueError(
                f"ratio must be 'density' or 'classifier'; got {self.ratio!r}. It is how a "
                "continuous node's policy ratio is estimated: a binned conditional density, or "
                "a stacked classification (Diaz et al. 2023, Section 5.4)"
            )
        check_ratio_trim(self.ratio_trim)
        check_hazard_pooling(self.hazard_pooling)
        if self.density_bins is not None and self.density_bins < 3:
            raise ValueError(f"density_bins must be at least 3; got {self.density_bins}")
        if self.msm is not None and not isinstance(self.msm, MSM):
            raise TypeError(
                f"msm= must be a cleverly.msm.MSM; got {type(self.msm).__name__}. A "
                "working model over regimens is declared with the same class a working "
                "model over arms is, and its design is handed "
                "(regimen_label, horizon, baseline_frame)"
            )
        refuse_hazard_settings(
            survival_model=self.survival_model,
            targeting=self.targeting,
            step_size=self.step_size,
            isotonic=self.isotonic,
            msm=self.msm is not None,
            pseudo_learner=self.pseudo_learner is not None,
            step_bound=getattr(self, "step_bound", None),
            ratio_trim=self.ratio_trim,
        )
        if self.msm is not None and self.reference is not None:
            raise ValueError(
                "msm= and reference= cannot be combined. reference= names the regimen "
                "every contrast is taken against, and a working model reports "
                "coefficients rather than contrasts; which regimen is the baseline is "
                "decided by the design you gave msm=, and an intercept is whatever that "
                "design makes it. A difference of two coefficients comes from "
                "result.contrast()."
            )

    def _validate_targeting(self) -> None:
        """Refuse an invalid targeting setting.  A recursion without a fluctuation has none."""

    def _resolve_tilts(self, data: LongitudinalData, plans: tuple[Plan, ...]) -> tuple[Plan, ...]:
        """The plans this estimator fits, given their odds tilt nodes.

        A targeted fit translates a plan whose every tilt node has a declared treatment
        mechanism into a known stochastic policy: :math:`q_t` is then a fixed function of
        the history (Kennedy 2019, Section 3.3), so the plan is an F1 policy plan.  A tilt
        node with an estimated mechanism is refused.

        Parameters
        ----------
        data : LongitudinalData
            The prepared panel.
        plans : tuple of Plan
            The resolved plans.

        Returns
        -------
        tuple of Plan
            The plans, with each tilt node read as a known policy.

        Raises
        ------
        CapabilityError
            When a tilt node's treatment mechanism is estimated.
        """
        known = data.known_mechanisms
        out = []
        for plan in plans:
            if not plan.has_tilt:
                out.append(plan)
                continue
            nodes = [time for time in range(1, data.n_times + 1) if plan.is_tilt_node(time)]
            if known is None or any(known.treatment_at(time) is None for time in nodes):
                raise CapabilityError.from_refusal(LTMLE_TILT_REFUSAL)
            propensity = {time: _filled_matrix(known.treatment_at(time)) for time in nodes}
            out.append(replace(plan.with_tilts(propensity), node_factors=()))
        return tuple(out)

    def _targeting(self) -> dict[str, Any]:
        """The fluctuation settings the recursion and the replay recipe take."""
        return {
            "alpha": None,
            "max_iter": None,
            "tol": None,
            "stopping": "relative",
            "step_bound": None,
            "step_search": True,
            "iterate": "last",
            "node_bounds": None,
        }

    def _normalization(self) -> str | None:
        """The IPW normalization, ``None`` on every other estimator."""
        return None

    def _refuse_design(self, data: LongitudinalData, plans: tuple[Plan, ...]) -> None:
        """Refuse a design this estimator cannot fit, before any learner runs.

        Parameters
        ----------
        data : LongitudinalData
            The prepared panel.
        plans : tuple of Plan
            The resolved plans, with their tilt nodes read.
        """

    def _fit_mechanism(
        self,
        data: LongitudinalData,
        plans: tuple[Plan, ...],
        *,
        treatment_learner: Learner,
        censoring_learner: Learner,
        folds: Folds,
    ) -> Mechanism:
        """Fit the out-of-fold treatment and censoring mechanism.

        Parameters
        ----------
        data : LongitudinalData
            The prepared panel.
        plans : tuple of Plan
            The resolved plans.
        treatment_learner : Learner
            The treatment mechanism learner.
        censoring_learner : Learner
            The censoring mechanism learner.
        folds : Folds
            The realized outer split.

        Returns
        -------
        Mechanism
            The fitted mechanism.
        """
        return fit_mechanism(
            data,
            plans,
            treatment_learner=treatment_learner,
            censoring_learner=censoring_learner,
            folds=folds,
            n_jobs=self.n_jobs,
            density_bins=self.density_bins,
            ratio=self.ratio,
            hazard_pooling=self.hazard_pooling,
        )

    def _parametric(
        self,
        outcome_learner: Learner,
        pseudo_learner: Learner,
        mechanism: Mechanism,
        horizons: Sequence[int],
        plans: Sequence[Plan],
    ) -> bool:
        """Whether every regression the estimator reads is a GLM.

        A targeted fit's status does not read it, so the base answer is ``True``.

        Parameters
        ----------
        outcome_learner : Learner
            The resolved outcome learner.
        pseudo_learner : Learner
            The resolved pseudo-outcome learner.
        mechanism : Mechanism
            The fitted mechanism.
        horizons : sequence of int
            The reported horizons.
        plans : sequence of Plan
            The resolved plans.

        Returns
        -------
        bool
            ``True`` here.
        """
        return True

    def _pseudo_learner(self, seed: int | None) -> Learner:
        """The learner of every node before the horizon, resolved at ``seed``."""
        return resolve_learner(
            self.pseudo_learner,
            task="regression",
            n_folds=self.learner_folds,
            random_state=seed,
            fallback=self.outcome_learner,
        )

    def _preflight_policies(self, prepared: LongitudinalData) -> None:
        """Run the declared-dose and plan checks of :meth:`fit` on ``prepared``, fitting nothing.

        The band-rank placebo of :func:`cleverly.validation.refute` calls this on each placebo
        panel before any refit, so a placebo history that a policy cannot meet stops the call
        before the first learner. The checks are those :meth:`fit` runs in the same order:
        :func:`preflight_declared_doses`, then the resolution of every plan, which runs each
        policy's support, partition, monotonicity, collapse, tie and atom-image checks. A
        tuned cap is fitted on the outer split the refit would draw.

        Parameters
        ----------
        prepared : LongitudinalData
            The prepared panel.

        Raises
        ------
        DataError
            If a declaration contradicts ``prepared``.
        CapabilityError
            If a policy has no density ratio on ``prepared``.
        """
        regimens = resolve_regimens(
            self.regimens,
            prepared.n_decisions,
            held_nodes=prepared.n_times if prepared.treatment_held else None,
        )
        preflight_declared_doses(
            prepared,
            regimens,
            ratio=self.ratio,
            density_bins=self.density_bins,
            undeclared_atoms=self._reads_density_ratio,
        )
        if declares_tuned(regimens):
            resolve_tuned_plans(regimens, prepared, self._draws(prepared)[0][0])
        else:
            resolve_plans(regimens, prepared, allow_set_dose=self._allow_set_dose)

    def _preflight_followers(
        self,
        data: LongitudinalData,
        plans: tuple[Plan, ...],
        horizons: Sequence[int],
        folds: Folds,
    ) -> None:
        """Refuse a regimen, horizon or training fold with no follower, before any learner.

        Parameters
        ----------
        data : LongitudinalData
            The prepared panel.
        plans : tuple of Plan
            The resolved plans.
        horizons : sequence of int
            The reported horizons.
        folds : Folds
            The realized outer split.
        """
        preflight_terminal_outcomes(data, plans, horizons, folds)

    def _gformula_settings(self, data: LongitudinalData) -> GFormulaSettings | None:
        """The simulation settings of a parametric g-formula fit, ``None`` on every other."""
        return None

    def _simulated_parametric(self, parametric: bool, fits: Mapping[str, RegimenFit]) -> bool:
        """Whether the status may report an interval, once the recursion has run.

        Parameters
        ----------
        parametric : bool
            The value :meth:`_parametric` gave before the recursion.
        fits : mapping of str to RegimenFit
            The fits of the recursion.

        Returns
        -------
        bool
            ``parametric`` here.
        """
        return parametric

    def _finish(
        self,
        result: LongitudinalResult,
        reported: _Reported,
        fits: Mapping[str, RegimenFit],
    ) -> LongitudinalResult:
        """Add what only this estimator reports to a built result.

        Parameters
        ----------
        result : LongitudinalResult
            The result before any bootstrap.
        reported : _Reported
            The names, estimates and contributing fits.
        fits : mapping of str to RegimenFit
            The fits of the recursion.

        Returns
        -------
        LongitudinalResult
            The result, unchanged here.
        """
        return result

    @staticmethod
    def profile_phases() -> AbstractContextManager[PhaseProfile]:
        """Collect wall-clock timings of a fit's phases, for a benchmark or a profile.

        .. code-block:: python

            with LTMLE.profile_phases() as profile:
                estimator.fit(data)
            print(profile.summary())

        A diagnostic, not part of a result: the shares depend on the box, the learner
        library and the node count, so recording one on a fitted object would invite it
        being compared across runs that are not comparable.  The phases are
        ``mechanism_fit``, ``mask_construction``, ``pseudo_outcome``,
        ``outcome_learner_fit``, ``clever_covariate``, ``fluctuation``,
        ``influence_curve`` and ``inference``.

        Off unless this is entered, and cheap when off -- see
        :mod:`cleverly.utils.phases`.  It is a static method because it installs a
        thread-global collector rather than instrumenting one estimator.
        """
        return profile_phases()

    @records_fit
    def fit(
        self,
        data: Any,
        *,
        outcome: str | Sequence[str] | Mapping[str, Sequence[str]] | Repeated | None = None,
        treatment: str | Sequence[str] | None = None,
        baseline: Sequence[str] | None = None,
        time_varying: Sequence[Sequence[str]] | None = None,
        censoring: Sequence[str] | None = None,
        id: str | None = None,
        weights: str | None = None,
        weights_type: str = "probability",
        weights_estimated: bool = False,
        family: str = "auto",
        continuous_treatment: Sequence[str] | Mapping[str, DeclaredSupport | None] = (),
        treatment_probabilities: Any = None,
        censoring_probabilities: Any = None,
        dose_atoms: Mapping[str, Sequence[float]] | None = None,
        **refused: Any,
    ) -> LongitudinalResult:
        """Fit on a wide dataframe, or on an already-built :class:`LongitudinalData`.

        ``weights=`` names a column of observation weights, read exactly as
        :meth:`cleverly.TMLE.fit` reads them: the estimand becomes the declared parameter
        in the tilted population ``dP_w = w dP / E[w]``, every node's nuisance is fitted by
        weighted loss, every node's score equation is weighted, and the reported curve is
        ``(w / E[w]) D*(P_w)``.  See :mod:`cleverly.data.weighting`.

        ``id=`` names a cluster column, and the variance is then cluster robust.  A fit
        with fewer than
        :data:`~cleverly._inference_status.MINIMUM_LONGITUDINAL_INTERVAL_CLUSTERS`
        clusters with positive weight mass reports no interval, p-value or standard error:
        every estimate takes the ``"few_cluster_plugin"`` status, as a point-treatment fit
        does.  From that count up to
        :data:`~cleverly._inference_status.FEW_CLUSTER_THRESHOLD` it reports Student t
        intervals with ``J - 2`` degrees of freedom, in sample and cross-fitted.
        The point estimate stands.  The inference reference, section *Clusters*, gives
        the reason.  Under cross-fitting the outer split is drawn whole-cluster and the
        Super Learner folds inside each regression are grouped on the same labels.

        Parameters
        ----------
        data : DataFrame or LongitudinalData
            One wide frame, a row per unit and a column per node, or a container that
            has already read the columns.  A container takes none of the column
            keywords below.
        outcome : str, sequence of str, mapping, Repeated, or None
            One column for an end-of-study outcome, one column per node for a survival
            outcome, a mapping of cause to one column per node for competing risks, or a
            :class:`~cleverly.longitudinal.Repeated` of one non-absorbing outcome per node,
            as for :meth:`LongitudinalData.from_frame`.
        treatment : str, sequence of str, or None
            The treatment column of each node, in time order.  Its length declares ``T``.
            One column name declares one baseline decision held over every node, as for
            :meth:`LongitudinalData.from_frame`.
        baseline : sequence of str or None
            The covariates measured before any treatment.
        time_varying : sequence of sequence of str or None
            One list of covariate columns per node, measured before that node's
            treatment.  ``None`` means that there are none.
        censoring : sequence of str or None
            One column per node, ``1`` where the unit is still under observation after
            that node.  ``None`` means that nobody was censored.
        id : str or None
            A cluster column, read as described above.
        weights : str or None
            A column of observation weights, read as described above.
        weights_type : str
            How to read ``weights``, as for :meth:`LongitudinalData.from_frame`.
        weights_estimated : bool
            Declare that the weights came out of a fitted model.  It changes no number.
        family : {"auto", "binomial", "gaussian"}
            The outcome family.  ``"auto"`` infers it from the outcome.
        continuous_treatment : sequence of str or mapping
            The treatment columns that hold a continuous dose, as for
            :meth:`LongitudinalData.from_frame`.  Each such node takes a modified treatment
            policy.  The mapping form ``{name: Support(lower, upper) or None}`` also
            declares each dose's conditional support, which the fit checks before any
            learner.
        treatment_probabilities : array-like, mapping, or None
            The known treatment mechanism of a sequentially randomized design, as for
            :meth:`LongitudinalData.from_frame`.  A declared node fits no treatment learner,
            and its factor is the declared probability of the arm each row is assigned.
            With every factor known, the remainder is zero and the curve is
            ``D*(Q_inf, g0)`` for any outcome learner (van der Laan and Gruber 2012).  The
            fit refuses, before any learner, a declared cumulative probability that
            ``g_bounds`` would move.  Beside a container the array forms are attached with
            :meth:`LongitudinalData.with_known_mechanisms`.
        censoring_probabilities : array-like, mapping, or None
            The known retention probabilities, as for
            :meth:`LongitudinalData.from_frame`.  A declared node fits no censoring learner.
        dose_atoms : mapping of str to sequence of float, or None
            The declared atoms of each continuous dose, as for
            :meth:`LongitudinalData.from_frame`: the doses with a recorded point mass.
        **refused : Any
            A point-treatment keyword.  Each one raises :class:`CapabilityError` with the
            reason that a longitudinal fit does not support it.

        Returns
        -------
        LongitudinalResult
            The regimen estimates, their contrasts or working-model coefficients, and
            the fits behind them.
        """
        refuse_unsupported(refused, where=f"{type(self).__name__}.fit")
        if self.msm is not None:
            # Checked again here, before any learner, because ``MSM`` checks its design and
            # weight declarations only when it is declared, and a modified model can carry
            # one this version refuses.
            refuse_msm_functions(self.msm)
        # The raw ``regimens=``, before ``_prepare`` and any learner: an inline callable
        # carries no declaration, and a modified regimen can carry one this version
        # refuses.  Its refusal comes before any refusal of the data or the design.
        refuse_regimen_rules(self.regimens)
        prepared = self._prepare(
            data,
            outcome=outcome,
            treatment=treatment,
            baseline=baseline,
            time_varying=time_varying,
            censoring=censoring,
            id=id,
            weights=weights,
            weights_type=weights_type,
            weights_estimated=weights_estimated,
            family=family,
            continuous_treatment=continuous_treatment,
            treatment_probabilities=treatment_probabilities,
            censoring_probabilities=censoring_probabilities,
            dose_atoms=dose_atoms,
        )
        if prepared.is_repeated:
            refuse_repeated_outcome(getattr(self, "_estimator", "tmle"), type(self).__name__)
        regimens = resolve_regimens(
            self.regimens,
            prepared.n_decisions,
            held_nodes=prepared.n_times if prepared.treatment_held else None,
        )
        if prepared.reports_curve:
            infixes = (HORIZON_INFIX,) + ((CAUSE_INFIX,) if prepared.is_competing else ())
            for infix in infixes:
                clashing = sorted(r.label for r in regimens if infix in r.label)
                if clashing:
                    raise ValueError(
                        f"regimen label(s) {clashing} contain {infix!r}, which is what "
                        "separates a regimen from the cause and horizon it is reported at. "
                        "Two parameters would then share a name; rename the regimen"
                    )
                bad_causes = sorted(c for c in prepared.cause_labels if infix in c)
                if bad_causes:
                    raise ValueError(
                        f"cause label(s) {bad_causes} contain {infix!r}, which is what "
                        "separates a cause from the regimen and horizon it is reported at. "
                        "Two parameters would then share a name; rename the cause"
                    )
        # Before any rule is evaluated and before any learner: a hazard fit and a projection
        # read only the design and the regimen declarations to refuse.
        refuse_hazard_design(
            prepared, regimens, survival_model=self.survival_model, isotonic=self.isotonic
        )
        reference = self._reference(regimens)
        # The declared support and the declared atoms of each continuous dose, and the
        # undeclared-atom check, before any plan or learner (X40 parts (a) and (b)).
        preflight_declared_doses(
            prepared,
            regimens,
            ratio=self.ratio,
            density_bins=self.density_bins,
            undeclared_atoms=self._reads_density_ratio,
        )
        # Every rule is called here and nowhere else, so a mask and the design the
        # mechanism was evaluated at cannot disagree about what the regimen assigned.
        tuned_folds: Folds | None = None
        draws: list[tuple[Folds, int | None]] | None = None
        if declares_tuned(regimens):
            if self.repeats > 1:
                raise CapabilityError(
                    "repeats above 1 is not combined with a data-tuned policy cap: the cap "
                    "is fitted on the training rows of one outer split, and the policy of a "
                    "second draw would need its own fit.",
                    reason=RefusalReason.IMPLEMENTATION_GAP,
                    roadmap="X75",
                    alternative="Fit the tuned policy with repeats=1.",
                )
            # A tuned cap is fitted before any learner, on each outer fold's training rows as
            # well as on every row at risk, so the split is drawn first (X40 part (e)).
            draws = self._draws(prepared)
            tuned_folds = draws[0][0]
            plans = resolve_tuned_plans(regimens, prepared, tuned_folds)
            # A fold-average plan reports under its own label; a reference that is one is
            # read under that label too.
            for declared, plan in zip(regimens, plans, strict=True):
                if plan.fold_average and declared.label == reference.label:
                    reference = plan.regimen
            regimens = tuple(
                plan.regimen if plan.fold_average else declared
                for declared, plan in zip(regimens, plans, strict=True)
            )
            self._refuse_fold_average_compositions(plans)
        else:
            plans = resolve_plans(regimens, prepared, allow_set_dose=self._allow_set_dose)
        # An odds tilt node reads the mechanism, so each estimator decides here, before any
        # learner, whether it fits one.
        plans = self._resolve_tilts(prepared, plans)
        self._refuse_design(prepared, plans)
        # Every declared factor is checked against the bounds before any learner.
        preflight_known_mechanisms(prepared, plans, resolve_cumulative_g_bounds(self.g_bounds))

        self._refuse_undeclared_policy_scale(prepared, plans)
        self._refuse_unbounded_cross_fitted_scale(prepared)
        if self.n_bootstrap and self.bootstrap_resampling == "cluster" and prepared.cluster is None:
            # The refusal ``run_bootstrap`` would raise after the fit, raised before any
            # learner.
            raise ValueError("resampling='cluster' requires the data to carry cluster ids")
        if draws is None:
            draws = self._draws(prepared)
        # One seed for every multiplier draw, fixed before any fold draw.
        multiplier_seed = self._multiplier_seed()
        scaler = self._scaler(prepared)
        # Before any learner: a bound the data cannot hold is a declaration error.
        resolved_bounds, declared_bounds = self._resolve_node_bounds(prepared, scaler)
        # A cumulative path probability is not a point-treatment propensity.  There is no
        # automatic rule here: the constructor default is the visible fixed R ``ltmle``
        # heuristic, and every other accepted value is an explicit scalar or pair.
        bounds = resolve_cumulative_g_bounds(self.g_bounds)
        horizons = self._horizons(prepared)
        if self.hazard_pooling == "pooled":
            refuse_pooled_designs(prepared)
        # Every draw is realized and preflighted before the first learner of any draw, so a
        # split that cannot be fitted is refused before any fitting time is spent.
        masks = prepared.regimen_masks(prepared.treatment)
        for index, (folds, _) in enumerate(draws, start=1):
            try:
                preflight_mechanism_support(prepared, masks, folds, ratio=self.ratio)
                self._preflight_followers(prepared, plans, horizons, folds)
            except Exception as error:
                if len(draws) == 1:
                    raise
                raise _named_draw(error, index, len(draws)) from error
        # A bootstrap replicate carries its fit's label and never classifies.  The key reads
        # ``repeats``, so a repeated fit matches no row of a single-draw study.
        classification = (
            classify_longitudinal_fit(
                self,
                prepared,
                source=f"{type(self).__name__}.fit",
                outcome_kind=outcome_kind(prepared),
                cross_fitted=not draws[0][0].is_single,
                known_factors=_known_factors(prepared),
                has_msm=self.msm is not None,
                regimen_kinds=regimen_kinds(plans),
                conditions=stopping_conditions(getattr(self, "stopping", "relative")),
            )
            if self._record_support
            else None
        )
        # One projection per cause -- a cause is a different estimand, not a further column
        # of the design -- over a grid whose cells cross the regimens with the horizons.
        # Evaluated here because the design is a closure and the driver takes arrays.  It
        # reads the plans' labels and the horizons only, so one evaluation serves every draw.
        model = (
            None if self.msm is None else evaluate_regimen_msm(self.msm, prepared, plans, horizons)
        )
        repeated = len(draws) > 1
        outcomes = [
            self._fit_draw(
                prepared,
                plans,
                folds,
                seed,
                # At one draw the learners keep ``random_state``, unchanged.  Each draw of a
                # repeated fit hands its own seed to every learner, so a Super Learner's
                # inner folds redraw with the outer split.
                learner_seed=seed if repeated else self.random_state,
                scaler=scaler,
                bounds=bounds,
                horizons=horizons,
                model=model,
                reference=reference,
                classification=classification,
                node_bounds=resolved_bounds,
            )
            for folds, seed in draws
        ]
        repeats = tuple(outcome.draw for outcome in outcomes)
        parameter_index = outcomes[0].parameter_index
        plans = outcomes[0].plans
        status = outcomes[0].status
        unprojected = outcomes[0].unprojected
        moves = repeats[0].isotonic_moves

        if self._estimator not in ("gcomp", "parametric_gformula"):
            # A g-computation estimate divides by no mechanism, so a bound moves none of it.
            self._warn_on_truncation([repeat.fits for repeat in repeats])

        config = LongitudinalConfig(
            family=prepared.family,
            n_times=prepared.n_times,
            outcome_names=(
                prepared.event_names or prepared.repeated_names or (prepared.outcome_name,)
            ),
            outcome_kind=outcome_kind(prepared),
            horizons=horizons,
            causes=prepared.cause_labels,
            regimens=regimens,
            reference=reference.label,
            n_folds=repeats[0].folds.n_folds,
            g_bounds=bounds,
            q_bounds=self.q_bounds,
            alpha_sig=self.alpha_sig,
            random_state=self.random_state,
            # From the matrix ``resolve_plans`` already built, so this is the assignment
            # the fit ran on rather than a second evaluation of the rules.  The dense
            # codes and not the raw labels: the codes are what every fit actually divides
            # and regresses on, and the levels they index are folded into
            # ``data_fingerprint`` -- so two plans that differ in the arm they assign
            # differ here, and two datasets whose labels differ differ there, without
            # ``repr()``-ing one string per unit per node per regimen to say so.  That
            # also keeps the digest stable across numpy versions, which disagree on
            # ``repr(numpy.str_(...))``.
            plan_fingerprints=tuple((plan.label, _plan_fingerprint(plan)) for plan in plans),
            policy_point_mass_nodes=tuple(
                (plan.label, plan.point_mass_nodes) for plan in plans if plan.point_mass_nodes
            ),
            held_treatment=prepared.treatment_names[0] if prepared.treatment_held else None,
            panel=None if prepared.panel is None else prepared.panel.describe(),
            time_grid=prepared.time_grid,
            known_factors=_known_factors(prepared),
            support_key=None if classification is None else classification.key,
            estimator=self._estimator,
            normalization=self._normalization(),
            msm_terms=None if model is None else model.terms,
            msm_link=None if model is None else str(model.link),
            # The evaluated arrays rather than the design, for the reason the plans are
            # fingerprinted from their resolved matrix: a closure has no stable digest,
            # and what the fit used is what it was handed.
            msm_fingerprint=(
                None if model is None else fingerprint_array(model.design, model.weights)
            ),
            repeats=len(repeats),
            multiplier_seed=multiplier_seed,
            survival_model=self.survival_model,
            targeting=self.targeting,
            step_size=float(self.step_size),
            isotonic=self.isotonic,
            node_bounds=declared_bounds,
        )
        # Each parameter takes the precedent of its draws' statuses before the median, so
        # a parameter that withholds inference in any draw withholds it in the report.  The
        # stored draws keep their own statuses.
        estimates = (
            repeats[0].estimates
            if not repeated
            else median_estimates(with_precedent_status([r.estimates for r in repeats]))
        )
        _warn_parameter_statuses(estimates)
        if self._estimator in _RANGE_CAUSE and parameter_index is not None:
            # Without a working model every draw files each name under the same
            # (regimen, cause, horizon) index, so draw 0's index gives each level's horizon.
            ranges = {
                name: _level_range(prepared, scaler, self.q_bounds, parameter_index[name][2])
                for name in estimates
            }
            _warn_parameter_space(estimates, ranges, estimator=self._estimator)
        with phase("inference"):
            if repeated and self.isotonic:
                # Each draw projects its levels, and the median of non-decreasing curves is
                # non-decreasing.  The limits of a median band have no projection rule: the
                # rule of Westling et al. (2020, Theorem 1 (iv)) is for one estimator, and
                # no source combines it with the median of Chernozhukov et al. (2018).
                bands = None
            elif moves and unprojected is not None:
                # The band is built at the unprojected levels, and each curve's limits are
                # then projected (Westling et al. 2020, Theorem 1 (iv)).  The critical value
                # reads the influence curves only, so it is the same either way.
                bands = _project_bands(
                    self._bands(unprojected, prepared, status, classification=classification),
                    parameter_index,
                )
            else:
                bands = self._bands(
                    estimates, prepared, status, repeats, multiplier_seed, classification
                )
        result = LongitudinalResult(
            estimates=estimates,
            data=prepared,
            config=config,
            scaler=scaler,
            provenance=self._provenance(prepared, [repeat.folds for repeat in repeats]),
            simultaneous=bands,
            parameter_index=parameter_index,
            msm=model,
            fitted_method=self._estimator,
            repeats=repeats,
            estimator=self._refit_template(),
        )
        first = outcomes[0]
        if (
            not repeated
            and self._estimator in ("tmle", "sdr")
            and not (self.n_bootstrap and self.bootstrap_interval == "bca")
            and prepared.cluster is None
            and model is None
        ):
            # A Wald fit of an estimate read from few rows: the variance estimate is noisy
            # and the skew moves the coverage (``cleverly.inference.sparse``). A percentile
            # bootstrap keeps the skew shortfall and its first interval is still the Wald one,
            # so it warns too, with advice for the BCa interval.
            warn_sparse_influence(
                influence_report(
                    {name: e for name, e in estimates.items() if e.supplies_inference},
                    alpha=self.alpha_sig,
                ),
                cross_fitted=not repeats[0].folds.is_single,
                bootstrap_measured=not first.hazard and not self.isotonic,
                percentile_bootstrap=bool(self.n_bootstrap),
            )
        result = self._finish(result, first.reported, repeats[0].fits)
        if self.n_bootstrap:
            result = self._attach_bootstrap(
                result,
                prepared,
                repeats[0].folds,
                plans,
                model=model,
                hazard=first.hazard,
                contributors=first.reported.contributors,
                learners=first.learners,
                g_bounds=bounds,
                classification=classification,
                status=status,
            )
        record_fit(classification, result, source=f"{type(self).__name__}.fit")
        return result

    def _fit_draw(
        self,
        prepared: LongitudinalData,
        plans: tuple[Plan, ...],
        folds: Folds,
        seed: int | None,
        *,
        learner_seed: int | None,
        scaler: OutcomeScaler,
        bounds: tuple[float, float],
        horizons: tuple[int, ...],
        model: RegimenMSM | None,
        reference: RegimenSpec,
        classification: Classification | None = None,
        node_bounds: dict[str | None, dict[int, NodeBound]] | None = None,
    ) -> _DrawOutcome:
        """One complete single-draw fit at one fold assignment.

        Every learner is resolved at ``learner_seed``.  The draw fits its own mechanism, its
        own training-fold mechanisms on a cross-fitted SDR or IPSI fit, its own recursion or
        hazard targeting, and its own estimates, so each in-fit contrast is formed inside the
        draw.

        Returns
        -------
        _DrawOutcome
            The draw, the parameter index, the plans with any policy numerator attached, the
            draw's fit status, and the estimates before an isotonic projection.
        """
        outcome_task = "classification" if prepared.family == "binomial" else "regression"
        outcome_learner = resolve_learner(
            self.outcome_learner,
            task=outcome_task,  # type: ignore[arg-type]
            n_folds=self.learner_folds,
            random_state=learner_seed,
        )
        pseudo_learner = self._pseudo_learner(learner_seed)
        # Build this record before the first backward pass. It holds only cloneable
        # unfitted templates and the already-resolved plan matrices, never a fitted Q.
        targeting = self._targeting()
        targeting["node_bounds"] = node_bounds
        replay_recipe = _replay_recipe(
            plans,
            outcome_learner,
            pseudo_learner,
            **targeting,
            n_jobs=self.n_jobs,
            estimator=self._estimator,
            normalization=self._normalization(),
        )
        hazard = self.survival_model == "hazard"
        # A hazard fit reads the censoring factors at every level of the baseline treatment,
        # so the shared mechanism fit is evaluated at one static plan per level as well.
        levels = level_plans(prepared) if hazard else ()
        treatment_learner = resolve_learner(
            self.treatment_learner,
            task="classification",
            n_folds=self.learner_folds,
            random_state=learner_seed,
        )
        censoring_learner = resolve_learner(
            self.censoring_learner,
            task="classification",
            n_folds=self.learner_folds,
            random_state=learner_seed,
            fallback=self.treatment_learner,
        )
        mechanism = self._fit_mechanism(
            prepared,
            (*plans, *levels),
            treatment_learner=treatment_learner,
            censoring_learner=censoring_learner,
            folds=folds,
        )
        parametric = self._parametric(outcome_learner, pseudo_learner, mechanism, horizons, plans)
        replay_recipe = replace(replay_recipe, parametric=parametric)
        # A modified treatment policy's ratio numerator reads the fitted mechanism, so it is
        # attached to its plan here, once per draw, from that draw's out-of-fold fit.  The
        # replay recipe holds the attached plans, because a truncation replay reuses this
        # mechanism.
        resolved = plans
        plans = attach_policy_numerators(plans, mechanism.policy_numerators)
        if any(plan.has_mtp for plan in plans):
            replay_recipe = replace(replay_recipe, plans=_frozen_plans(plans))
        # A cross-fitted SDR fold reads its own mechanism at its training rows, so each
        # fold fits one on those rows alone (the training-fold recursion).  The out-of-fold
        # mechanism above is what the result reports and what a held-out row divides by.
        fold_mechanisms: tuple[FoldMechanism, ...] = ()
        if self._estimator in ("sdr", "ipsi") and not folds.is_single:
            fold_mechanisms = fit_fold_mechanisms(
                prepared,
                resolved,
                treatment_learner=treatment_learner,
                censoring_learner=censoring_learner,
                folds=folds,
                density_bins=self.density_bins,
                ratio=self.ratio,
                hazard_pooling=self.hazard_pooling,
                n_jobs=self.n_jobs,
            )
        if self.ratio_trim is not None:
            # One trim per plan over the pooled out-of-fold ratios, attached to the mechanism
            # so that every reader of the cumulative product, a truncation replay included,
            # divides by the same trimmed ratio.
            mechanism = replace(
                mechanism, ratio_trims=trim_ratios(prepared, plans, mechanism, self.ratio_trim)
            )
        initial: InitialHazards | None = None
        msm_fits: tuple[MSMRegimenFit, ...]
        if hazard:
            algorithm: HazardAlgorithm = (
                "one_step" if self.targeting == "one_step" else "per_horizon"
            )
            initial = fit_hazards(
                prepared,
                levels,
                learner=outcome_learner,
                folds=folds,
                n_jobs=self.n_jobs,
                hazard_pooling=self.hazard_pooling,
            )
            # A hazard replay refits no learner, so the learner audit does not apply to it.
            replay_recipe = replace(
                replay_recipe,
                omissions=(),
                hazard=_HazardReplay(
                    _frozen_plans(levels),
                    algorithm,
                    float(self.step_size),
                    targeting["alpha"],
                    targeting["max_iter"],
                    targeting["tol"],
                ),
            )
            fits = _run_hazard(
                prepared,
                plans,
                levels,
                mechanism,
                initial,
                horizons=horizons,
                g_bounds=bounds,
                alpha=targeting["alpha"],
                algorithm=algorithm,
                max_iter=targeting["max_iter"],
                tol=targeting["tol"],
                step_size=self.step_size,
                node_bounds=node_bounds,
                stopping=targeting["stopping"],
                step_bound=targeting["step_bound"],
                step_search=targeting["step_search"],
                iterate=targeting["iterate"],
            )
            msm_fits = ()
        else:
            fits, msm_fits = _run_recursion(
                prepared,
                plans,
                mechanism,
                msm=model,
                horizons=horizons,
                outcome_learner=outcome_learner,
                pseudo_learner=pseudo_learner,
                folds=folds,
                scaler=scaler,
                g_bounds=bounds,
                **targeting,
                n_jobs=self.n_jobs,
                estimator=self._estimator,
                fold_mechanisms=fold_mechanisms,
                normalization=self._normalization(),
                parametric=parametric,
                gformula=self._gformula_settings(prepared),
            )
        parametric = self._simulated_parametric(parametric, fits)
        status = _trimmed_status(
            _inference_status(prepared, folds, estimator=self._estimator, parametric=parametric),
            mechanism,
        )
        unprojected_fits = fits
        moves: tuple[tuple[str, str | None, int, float, float], ...] = ()
        if self.isotonic:
            replay_recipe = replace(replay_recipe, isotonic=True)
            fits, moves = _project_fits(fits)
        with phase("influence_curve"):
            reported = (
                _estimates(
                    prepared,
                    fits,
                    scaler,
                    reference,
                    alpha_sig=self.alpha_sig,
                    inference=status,
                    classification=classification,
                )
                if model is None
                else _msm_estimates(
                    prepared,
                    msm_fits,
                    alpha_sig=self.alpha_sig,
                    inference=status,
                    classification=classification,
                )
            )
            # The band is built at the unprojected levels, and each curve's limits are then
            # projected (Westling et al. 2020, Theorem 1 (iv)).
            unprojected = (
                _estimates(
                    prepared,
                    unprojected_fits,
                    scaler,
                    reference,
                    alpha_sig=self.alpha_sig,
                    inference=status,
                    classification=classification,
                ).estimates
                if moves
                else None
            )
        draw = LongitudinalRepeat(
            folds=folds,
            seed=seed,
            fits=fits,
            mechanism=mechanism,
            estimates=reported.estimates,
            replay_recipe=replay_recipe,
            msm_fits=msm_fits,
            fold_mechanisms=fold_mechanisms,
            isotonic_moves=moves,
            initial_hazards=initial,
        )
        return _DrawOutcome(
            draw,
            reported.index,
            plans,
            status,
            unprojected,
            reported,
            (outcome_learner, pseudo_learner, treatment_learner, censoring_learner),
            hazard,
        )

    def _attach_bootstrap(
        self,
        result: LongitudinalResult,
        prepared: LongitudinalData,
        folds: Folds,
        plans: Sequence[Plan],
        *,
        model: Any,
        hazard: bool,
        contributors: Mapping[str, tuple[RegimenFit, ...]],
        learners: tuple[Learner, Learner, Learner, Learner],
        g_bounds: tuple[float, float],
        classification: Classification | None,
        status: InferenceStatus,
    ) -> LongitudinalResult:
        """Run the full-refit bootstrap and attach its intervals and band to ``result``.

        The replicates take the frequency-weighted fast path
        (:class:`~cleverly.longitudinal.frequency.FrequencyReplicate`) when the fit is
        eligible for it, and a materialized refit otherwise.  Both read the same row index
        per replicate, so the choice moves no draw beyond rounding.
        """
        fast = self._frequency_replicate(
            prepared,
            folds,
            plans,
            model=model,
            hazard=hazard,
            contributors=contributors,
            learners=learners,
            g_bounds=g_bounds,
            names=tuple(result.estimates),
        )
        # Spawned from ``random_state`` inside the run, so the stream draws nothing the fit
        # used: its split and its multiplier band are unchanged.
        if fast is not None:
            bootstrap = run_index_bootstrap(
                prepared,
                fast,
                n_replicates=self.n_bootstrap,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
        else:
            bootstrap = run_bootstrap(
                prepared,
                self._bootstrap_estimates,
                n_replicates=self.n_bootstrap,
                resampling=self.bootstrap_resampling,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
        # The support row of the fit decides whether each bootstrap interval is supported:
        # only a row that lists the interval's kind (``bootstrap``, ``bootstrap_bca`` or
        # ``bootstrap_band``) licenses it, and every other fit that publishes it is
        # experimental.
        result = attach_bootstrap(
            result, bootstrap, classification=classification, interval=self.bootstrap_interval
        )
        banded = {
            name: estimate
            for name, estimate in result.estimates.items()
            if estimate.supplies_inference and name in bootstrap.std_errors
        }
        # As for the multiplier band, no source gives a t-calibrated joint band.
        if not self.simultaneous or len(banded) < 2 or has_t_reference(banded):
            return result
        band = bootstrap_band(
            bootstrap,
            {name: estimate.psi for name, estimate in banded.items()},
            {name: estimate.plugin_std_error for name, estimate in banded.items()},
            alpha=self.alpha_sig,
            support=band_support(
                classification,
                status,
                (estimate.support for estimate in banded.values()),
                kind="bootstrap_band",
            ),
        )
        return replace(result, bootstrap_band=band)

    def _frequency_replicate(
        self,
        prepared: LongitudinalData,
        folds: Folds,
        plans: Sequence[Plan],
        *,
        model: Any,
        hazard: bool,
        contributors: Mapping[str, tuple[RegimenFit, ...]],
        learners: tuple[Learner, Learner, Learner, Learner],
        g_bounds: tuple[float, float],
        names: tuple[str, ...],
    ) -> FrequencyReplicate | None:
        """The frequency-weighted replicate of this fit, or ``None`` when it is not exact.

        The fast path covers an unclustered, unweighted, in-sample iterated-mean ``LTMLE``
        or ``SDR`` fit of one end-of-study horizon, with every treatment and censoring
        factor declared known, categorical treatment nodes, no tilt or modified treatment
        policy, and outcome and pseudo-outcome learners declared exact under frequency
        weights (:func:`~cleverly.longitudinal.frequency.frequency_exact`) that accept
        ``sample_weight``.  Every other fit takes the materialized refit.
        """
        outcome_learner, pseudo_learner, treatment_learner, censoring_learner = learners
        eligible = (
            self._estimator in ("tmle", "sdr")
            and not hazard
            and not self.isotonic
            and self.node_bounds is None
            and model is None
            and folds.is_single
            and prepared.cluster is None
            and self.bootstrap_resampling in ("auto", "iid")
            and not prepared.is_weighted
            and not prepared.reports_curve
            and not prepared.is_survival
            and not prepared.is_repeated
            and not prepared.is_competing
            and not prepared.treatment_held
            and all(
                prepared.node_kind(time) == "categorical" for time in range(1, prepared.n_times + 1)
            )
            and _mechanism_declared(prepared)
            and not any(plan.has_tilt or plan.has_mtp for plan in plans)
            and all(
                frequency_exact(learner) and supports_sample_weight(learner)
                for learner in (outcome_learner, pseudo_learner)
            )
        )
        if not eligible:
            return None
        reported = []
        horizons: set[int] = set()
        for name, fits in contributors.items():
            horizons.update(fit.horizon for fit in fits)
            reference = fits[1].regimen.label if len(fits) > 1 else None
            reported.append(ReportedName(name, fits[0].regimen.label, reference))
        produced = tuple(item.name for item in reported)
        # The same names in another order would drop the fast path without a trace, so fail
        # loudly under ``python`` without ``-O``.
        assert produced == names or sorted(produced) != sorted(names), (
            "the contributors and the result list the same estimates in a different order"
        )
        if produced != names or len(horizons) != 1:
            return None
        regimens = tuple(plan.regimen for plan in plans)

        def resolve(data: LongitudinalData) -> tuple[tuple[Plan, ...], Mechanism]:
            resolved = resolve_plans(regimens, data)
            mechanism = fit_mechanism(
                data,
                resolved,
                treatment_learner=treatment_learner,
                censoring_learner=censoring_learner,
                folds=Folds.single(data.n),
                density_bins=self.density_bins,
                ratio=self.ratio,
            )
            return tuple(resolved), mechanism

        return FrequencyReplicate.build(
            prepared,
            resolve=resolve,
            names=reported,
            estimator=self._estimator,
            horizon=horizons.pop(),
            outcome_learner=outcome_learner,
            pseudo_learner=pseudo_learner,
            g_bounds=g_bounds,
            q_bounds=self.q_bounds,
            targeting=self._targeting(),
        )

    def _replicate_estimator(self) -> _SequentialEstimator:
        """This estimator's settings for one bootstrap replicate.

        The same regimens, reference, horizons, working model, bounds and learner
        specifications, with no bootstrap, no simultaneous band and one worker.  Every
        learner is resolved and cloned again inside :meth:`fit`, so a replicate refits
        everything.
        """
        replicate = copy.copy(self)
        replicate._record_support = False
        replicate.n_bootstrap = 0
        replicate.simultaneous = False
        replicate.n_jobs = 1
        return replicate

    def _refit_template(self) -> _SequentialEstimator:
        """The configuration a result keeps, so that a refutation can refit it.

        A copy, so a later change to this instance does not change what the result says it
        was fitted with. It runs no bootstrap and builds no simultaneous band, so a refit
        costs one fit, and it records no support classification. It holds no frozen split.
        """
        template = copy.copy(self)
        template._record_support = False
        template.n_bootstrap = 0
        template.simultaneous = False
        template._frozen_folds = None
        template._frozen_seeds = None
        return template

    def refit(
        self,
        data: LongitudinalData,
        *,
        random_state: int | None = None,
        frozen_folds: Sequence[Folds] | None = None,
        frozen_seeds: Sequence[int | None] | None = None,
    ) -> LongitudinalResult:
        """Run the whole fit again, nuisances included, on a prepared container.

        :func:`cleverly.validation.refute` calls this on a longitudinal result. It changes a
        column, draws units, or appends a covariate, and then refits. The estimator is not
        modified: every setting applies to a copy.

        Parameters
        ----------
        data : LongitudinalData
            The prepared container to fit, usually ``result.data`` with one array changed.
        random_state : int or None
            Seed of the refit. ``None`` keeps this estimator's own.
        frozen_folds : sequence of Folds or None
            One realized outer split per draw, for the rows of ``data``. The refit uses
            these labels instead of drawing a split. ``None`` draws the split as usual.
        frozen_seeds : sequence of int or None
            One seed per draw, replacing the seeds the fit would take from
            ``random_state``. A refit given the original draws' seeds and data with the same
            row count draws the original folds.

        Returns
        -------
        LongitudinalResult
            The result of the whole fit on ``data``.
        """
        estimator = self._refit_template()
        if random_state is not None:
            estimator.random_state = random_state
        if frozen_folds is not None:
            estimator._frozen_folds = tuple(frozen_folds)
        if frozen_seeds is not None:
            estimator._frozen_seeds = tuple(frozen_seeds)  # type: ignore[arg-type]
        return estimator.fit(data)

    def _bootstrap_estimates(self, data: LongitudinalData) -> dict[str, tuple[float, float]]:
        """One bootstrap replicate: a full refit on the resample.

        Runs the same path as :meth:`fit`, not :func:`_refit_bound`, which holds the
        mechanism and the folds fixed.

        Parameters
        ----------
        data : LongitudinalData
            The resampled container.

        Returns
        -------
        dict of str to tuple of float
            The point estimate and the influence-curve standard error of every reported
            parameter.
        """
        refit = self._replicate_estimator().fit(data)
        return {
            name: (estimate.psi, estimate.plugin_std_error)
            for name, estimate in refit.estimates.items()
        }

    # ------------------------------------------------------------- internals

    @staticmethod
    def _prepare(
        data: Any,
        *,
        outcome: str | Sequence[str] | Mapping[str, Sequence[str]] | Repeated | None,
        treatment: str | Sequence[str] | None,
        baseline: Sequence[str] | None,
        time_varying: Sequence[Sequence[str]] | None,
        censoring: Sequence[str] | None,
        id: str | None,
        weights: str | None,
        weights_type: str,
        weights_estimated: bool,
        family: str,
        continuous_treatment: Sequence[str] | Mapping[str, Any] = (),
        treatment_probabilities: Any = None,
        censoring_probabilities: Any = None,
        dose_atoms: Mapping[str, Sequence[float]] | None = None,
    ) -> LongitudinalData:
        if isinstance(data, LongitudinalData):
            declared = {
                "outcome": outcome,
                "treatment": treatment,
                "baseline": baseline,
                "time_varying": time_varying,
                "censoring": censoring,
                "id": id,
                "weights": weights,
            }
            named = sorted(key for key, value in declared.items() if value is not None)
            if family != "auto":
                named.append("family")
            # The two that are not column names, and so cannot be caught by the ``is not
            # None`` sweep above: they say how a weight column is to be *read*, and a
            # container has already read it.
            if weights_type != "probability":
                named.append("weights_type")
            if weights_estimated:
                named.append("weights_estimated")
            if continuous_treatment:
                named.append("continuous_treatment")
            if dose_atoms:
                named.append("dose_atoms")
            if named:
                raise ValueError(
                    f"{named} cannot be combined with a LongitudinalData input; the node "
                    "ordering and the outcome family are already fixed by the container, "
                    "and passing them again cannot change them. Pass them to "
                    "LongitudinalData.from_frame, which is where the columns are read"
                )
            if treatment_probabilities is not None or censoring_probabilities is not None:
                return data.with_known_mechanisms(
                    treatment_probabilities=treatment_probabilities,
                    censoring_probabilities=censoring_probabilities,
                )
            return data
        if outcome is None or treatment is None or baseline is None:
            raise TypeError(
                "fitting from a dataframe needs outcome=, treatment= and baseline=; "
                "pass a LongitudinalData to supply them once"
            )
        return LongitudinalData.from_frame(
            data,
            outcome=outcome,
            treatment=treatment,
            baseline=baseline,
            time_varying=time_varying,
            censoring=censoring,
            id=id,
            weights=weights,
            weights_type=weights_type,
            weights_estimated=weights_estimated,
            family=family,
            continuous_treatment=continuous_treatment,
            treatment_probabilities=treatment_probabilities,
            censoring_probabilities=censoring_probabilities,
            dose_atoms=dose_atoms,
        )

    def _horizons(self, data: LongitudinalData) -> tuple[int, ...]:
        """Which nodes the fit reports a parameter at."""
        if not data.reports_curve:
            if self.horizons is not None:
                raise MethodConfigurationError(
                    "horizons= applies to a survival or a repeated outcome, and this fit has "
                    f"one end-of-study outcome ({data.outcome_name!r}), whose only horizon is "
                    "the end of the study.",
                    reason=RefusalReason.INVALID_DECLARATION,
                    alternative=(
                        "Pass one event column per time point, outcome=[...], or a repeated "
                        "outcome, outcome=Repeated([...]), to report a curve."
                    ),
                )
            return (data.n_times,)
        if self.horizons is None:
            return tuple(range(1, data.n_times + 1))
        if data.time_grid is not None:
            return self._grid_horizons(data.time_grid)
        wanted = tuple(int(horizon) for horizon in self.horizons)
        if not wanted:
            raise ValueError("horizons= is empty; a fit reporting no parameter is not one")
        outside = sorted({h for h in wanted if not 1 <= h <= data.n_times})
        if outside:
            raise ValueError(
                f"horizons= names {outside}, which is outside 1..{data.n_times}; a "
                "horizon is one of the fit's own time points"
            )
        if len(set(wanted)) != len(wanted):
            raise ValueError(f"horizons= repeats a time point: {list(wanted)}")
        return tuple(sorted(wanted))

    def _grid_horizons(self, grid: tuple[float, ...]) -> tuple[int, ...]:
        """The node indices of ``horizons=`` on a fit with a time grid, which names grid times."""
        assert self.horizons is not None
        values = np.asarray(grid, dtype=float)
        wanted: list[int] = []
        for horizon in self.horizons:
            matches = np.flatnonzero(values == float(horizon))
            if isinstance(horizon, bool) or matches.size != 1:
                raise ValueError(
                    f"horizon {horizon!r} is not a reported grid time; this fit's grid is "
                    f"{', '.join(f'{g:g}' for g in grid)}"
                )
            wanted.append(int(matches[0]) + 1)
        if not wanted:
            raise ValueError("horizons= is empty; a fit reporting no parameter is not one")
        if len(set(wanted)) != len(wanted):
            raise ValueError(f"horizons= repeats a time point: {list(self.horizons)}")
        return tuple(sorted(wanted))

    @staticmethod
    def _warn_on_truncation(draws: Sequence[Mapping[str, RegimenFit]]) -> None:
        """Warn once with every regimen/node whose scored rows are materially clipped.

        Of the out-of-fold mechanism, for the reason
        :meth:`LongitudinalResult._max_truncated` gives: this says how much of the mechanism
        the fit divided by the bounds moved.  On a repeated fit the share is the largest
        over the draws, and the warning names the draws that exceeded it.
        """
        found: dict[tuple[str, int], tuple[float, bool]] = {}
        exceeded: dict[tuple[str, int], set[int]] = {}
        for index, fits in enumerate(draws, start=1):
            for fit in fits.values():
                for step in fit.scored_nodes:
                    bounded = fit.cumulative[:, step.time - 1][step.trained_on]
                    clipped = _clipped_prefix(
                        fit.cumulative_unbounded, fit.cumulative, step.time, step.trained_on
                    )
                    if not clipped.size:
                        continue
                    share = float(np.mean(clipped))
                    constant = bool(share == 1.0 and np.unique(bounded).size == 1)
                    key = (fit.regimen.label, step.time)
                    previous = found.get(key, (0.0, False))
                    found[key] = max(previous[0], share), previous[1] or constant
                    if share > _TRUNCATION_WARN_FRACTION:
                        exceeded.setdefault(key, set()).add(index)

        affected = [
            (label, time, share, constant)
            for (label, time), (share, constant) in sorted(found.items())
            if share > _TRUNCATION_WARN_FRACTION
        ]
        if not affected:
            return
        total = len(draws)

        def where(label: str, time: int) -> str:
            if total == 1:
                return ""
            listed = ", ".join(str(index) for index in sorted(exceeded[(label, time)]))
            return f" in draw(s) {listed} of {total}"

        details = "; ".join(
            f"{label} at t={time}: {share:.1%}"
            + where(label, time)
            + (
                " (the bounded cumulative probability, and hence the clever covariate, "
                "is constant on scored rows)"
                if constant
                else ""
            )
            for label, time, share, constant in affected
        )
        _warn(
            "Cumulative mechanism truncation exceeded 5% of scored rows: "
            f"{details}. The fit divides by the truncated weights; "
            "material truncation can trade reduced weight extremes for truncation bias and "
            "can make plug-in "
            "influence-curve inference unreliable. Inspect "
            "res.diagnostics.support().to_frame(), report "
            "the configured bounds, and refit the full backward recursion under "
            "substantively justified alternatives.",
            PositivityWarning,
            stacklevel=3,
        )

    def _bands(
        self,
        estimates: Mapping[str, ParameterEstimate],
        data: LongitudinalData,
        status: InferenceStatus,
        repeats: Sequence[LongitudinalRepeat] = (),
        multiplier_seed: int | None = None,
        classification: Classification | None = None,
    ) -> SimultaneousBands | None:
        """Joint bands over the reported parameters, when there is more than one.

        A fit declaring ``R`` regimens reports ``R`` means and ``R - 1`` contrasts, all
        correlated through the same influence curves -- which is the situation the bands
        exist for, and the reason they are on by default here as they are on a
        point-treatment fit.  One regimen reports one parameter, and a band over one
        estimand is its pointwise interval.

        A band is a joint confidence statement, so a fit that supplies no inference builds
        none, as :meth:`cleverly.TMLE.fit` builds none.  It is skipped rather than raised,
        because ``simultaneous`` defaults to ``True``: a raise would stop every default fit
        with few clusters.  :meth:`LongitudinalResult.summary` states the omission.
        ``status`` is the one the fit stamped on ``estimates``. ``classification`` gives the band
        its support: the row must license a ``band`` kind, and every estimate must be supported.

        On a repeated fit the band is :func:`~cleverly.inference.multiplier.median_bands`:
        the median of the draws' critical values around the split-adjusted estimates, with
        the one ``multiplier_seed`` in every draw.
        """
        if not supplies_inference(status):
            return None
        # A parameter whose status withholds its interval, such as a constant-node risk, stays
        # out of the band, which is then a joint statement over the others. A parameter that
        # shows a labelled interval stays in, and its experimental label reaches the band.
        estimates = {
            name: estimate for name, estimate in estimates.items() if estimate.supplies_inference
        }
        if not self.simultaneous or len(estimates) < 2:
            return None
        # No source gives a t-calibrated joint band, so a fit whose estimates carry a
        # Student t reference builds none either, and the summary says so.
        if has_t_reference(estimates):
            return None
        if len(repeats) > 1:
            bands = median_bands(
                [repeat.estimates for repeat in repeats],
                estimates,
                alpha=self.alpha_sig,
                n_replicates=self.n_multiplier,
                kind=self.multiplier_kind,  # type: ignore[arg-type]
                random_state=multiplier_seed,
                cluster=data.cluster,
            )
        else:
            bands = simultaneous_bands(
                estimates,
                alpha=self.alpha_sig,
                n_replicates=self.n_multiplier,
                kind=self.multiplier_kind,  # type: ignore[arg-type]
                random_state=self.random_state,
                cluster=data.cluster,
            )
        return replace(
            bands,
            support=band_support(
                classification, status, (estimate.support for estimate in estimates.values())
            ),
        )

    def _resolve_node_bounds(
        self, data: LongitudinalData, scaler: OutcomeScaler
    ) -> tuple[
        dict[str | None, dict[int, NodeBound]] | None,
        tuple[tuple[str | None, float, float, float], ...],
    ]:
        """Check the declared node bounds against the fit, before any learner.

        Parameters
        ----------
        data : LongitudinalData
            The prepared panel, whose grid and causes the keys must name.
        scaler : OutcomeScaler
            The outcome transformation, whose range the bounds must lie in.

        Returns
        -------
        resolved : dict or None
            The bound of each bounded node on the ``[0, 1]`` scale, by cause (``None`` for a
            fit without competing risks) and one-based node.  ``None`` when no node is bounded.
            A node whose bounds equal the outcome range is the shipped node and is left out.
            On a fit with ``survival_model="hazard"`` the bounds are on the hazard scale, and
            the range they must lie in is ``[0, 1]``.
        declared : tuple
            ``(cause, grid time, lower, upper)`` on the reported scale, in declaration order.

        Raises
        ------
        ValueError
            When a key is not a grid time or a declared cause, when ``lower >= upper``, or when
            a bound lies outside the outcome range.
        CapabilityError
            On a hazard fit, when a node bounds some causes and not others (R-i6), or when the
            upper bounds of a node sum to more than one (R-i2).
        """
        if self.node_bounds is None:
            return None, ()
        grid = (
            tuple(data.time_grid)
            if data.time_grid is not None
            else tuple(range(1, data.n_times + 1))
        )
        causes = tuple(data.cause_labels)
        if causes:
            entries: list[tuple[str | None, Any, Any]] = []
            for cause, mapping in self.node_bounds.items():
                if cause not in causes:
                    raise ValueError(
                        f"node_bounds names cause {cause!r}; this fit declared {list(causes)}"
                    )
                if not isinstance(mapping, Mapping):
                    raise ValueError(
                        f"node_bounds for cause {cause!r} must map grid times to (lower, upper) "
                        f"pairs; got {type(mapping).__name__}"
                    )
                entries.extend((cause, key, value) for key, value in mapping.items())
        else:
            entries = [(None, key, value) for key, value in self.node_bounds.items()]
        hazard = self.survival_model == "hazard"
        # A hazard is a probability, so a hazard fit reads its bounds on [0, 1] whatever the
        # outcome transformation is.
        low, high = (0.0, 1.0) if hazard else (float(scaler.lower), float(scaler.upper))
        resolved: dict[str | None, dict[int, NodeBound]] = {}
        declared: list[tuple[str | None, float, float, float]] = []
        by_node: dict[int, dict[str | None, tuple[float, float]]] = {}
        for cause, key, value in entries:
            node = next(
                (
                    index
                    for index, point in enumerate(grid, start=1)
                    if isinstance(key, (int, float, np.number)) and float(key) == float(point)
                ),
                None,
            )
            if node is None:
                raise ValueError(
                    f"node_bounds names grid time {key!r}, which is not a node of this fit; "
                    f"its grid is {list(grid)}"
                )
            try:
                lower, upper = (float(bound) for bound in value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"node_bounds at {key!r} must be a (lower, upper) pair of numbers; "
                    f"got {value!r}"
                ) from None
            if not lower < upper:
                raise ValueError(
                    f"node_bounds at {key!r} needs lower < upper; got ({lower!r}, {upper!r})"
                )
            if lower < low or upper > high:
                raise ValueError(
                    f"node_bounds at {key!r} lies outside the "
                    f"{'hazard' if hazard else 'outcome'} range [{low:g}, {high:g}]"
                )
            declared.append((cause, float(grid[node - 1]), lower, upper))
            by_node.setdefault(node, {})[cause] = (lower, upper)
            scaled = (
                (lower, upper)
                if hazard
                else (
                    float((lower - scaler.lower) / scaler.range),
                    float((upper - scaler.lower) / scaler.range),
                )
            )
            if scaled == (0.0, 1.0):
                continue
            resolved.setdefault(cause, {})[node] = NodeBound(
                lower=scaled[0],
                upper=scaled[1],
                time=key,
                reported=(lower, upper),
                outcome=scaler,
            )
        if hazard:
            check_hazard_node_bounds(by_node, grid, causes or (None,))
        return (resolved or None), tuple(declared)

    def _scaler(self, data: LongitudinalData) -> OutcomeScaler:
        """The outcome transformation, refusing a bound that cannot apply.

        The same rule as :meth:`cleverly.TMLE._scaler`: a binary outcome is already on
        the ``[0, 1]`` scale the recursion works in, so ``q_bounds=`` would have nothing
        to do -- and a setting with nothing to do is refused rather than dropped, since
        the caller who passed it believes it took effect.  ``family`` may have been
        inferred, so this can only be checked once the data is built.
        """
        if data.family == "binomial":
            if self.q_bounds is not None:
                raise ValueError("q_bounds does not apply to a binary outcome")
            return OutcomeScaler.identity()
        # Every horizon's observed values on a repeated outcome: one scale for the curve.
        return OutcomeScaler.from_outcome(data.observed_outcomes(), self.q_bounds)

    def _provenance(self, data: LongitudinalData, folds: Sequence[Folds]) -> Provenance:
        """The same record a point-treatment fit carries, over the longitudinal nodes.

        The data fingerprint covers every node rather than three arrays, so two fits
        that differ only in a covariate measured at the second time point have
        different fingerprints -- which is the property the record exists for.  That is
        the whole of the difference from :func:`cleverly.provenance.record`; the
        environment half is built by :func:`cleverly.provenance.build` so that the
        package versions and ``run_id`` this record is *for* cannot go missing here.

        What the *regimens* were is not in here but on
        :attr:`LongitudinalConfig.plan_fingerprints`, deliberately:
        :func:`cleverly.provenance.build` is shared with the point-treatment path, and a
        field only one estimator can fill would make the shared record answer a question
        half its callers have no answer to.
        """
        return provenance_build(
            n=data.n,
            n_covariates=len(data.baseline_names)
            + sum(len(names) for names in data.time_varying_names),
            n_clusters=None if data.cluster is None else data.n_clusters,
            data_fingerprint=fingerprint_array(
                data.outcome,
                data.treatment,
                data.uncensored.astype(float),
                data.baseline,
                data.weights,
                *data.time_varying,
                np.asarray(
                    [
                        repr(level)
                        for time, levels in enumerate(data.treatment_levels, start=1)
                        # A mixed node has no levels; _component_fingerprint digests its
                        # components' labels.
                        if data.node_kind(time) != "mixed"
                        for level in levels
                    ],
                    dtype=str,
                ),
                *_component_fingerprint(data),
                # Which cause fired is in no other array: ``outcome`` and the ``nan``
                # pattern of the later nodes are all-cause.  A repeated outcome's last
                # node is ``outcome``, and its earlier nodes may be screened out of the
                # covariate blocks.  Each is fingerprinted when the fit has it.
                *(
                    ()
                    if data.cause_event is None
                    else (np.asarray(data.cause_event, dtype=float).reshape(data.n, -1),)
                ),
                *(() if data.repeated is None else (data.repeated,)),
            ),
            # Every draw in fit order.  One ``int64`` draw digests as
            # ``fingerprint_array(folds.assignment)`` did.
            fold_fingerprint=fold_fingerprint(draw.assignment for draw in folds),
            random_state=self.random_state,
            run_id=self.run_id,
        )

    def _reference(self, regimens: Sequence[RegimenSpec]) -> RegimenSpec:
        if self.reference is None:
            return regimens[0]
        for regimen in regimens:
            if regimen.label == self.reference:
                return regimen
        raise KeyError(
            f"reference={self.reference!r} is not one of the declared regimens "
            f"{[regimen.label for regimen in regimens]}"
        )

    def _refuse_fold_average_compositions(self, plans: Sequence[Plan]) -> None:
        """Refuse a fold-average target with a working model, the bootstrap, or a targeting option.

        Each outer fold's map is fitted on that fold's training rows, so the target depends
        on the split.  A working model would project
        split-dependent targets, and each bootstrap resample refits every fold's map.
        """
        if not any(plan.fold_average for plan in plans):
            return
        if self.msm is not None:
            raise CapabilityError(
                "target='fold_average' is not combined with msm=: each fold's map is fitted "
                "on that fold's training rows, so a working model would project targets that "
                "depend on the split.",
                reason=RefusalReason.DIFFERENT_ESTIMAND,
                alternative=(
                    "Fit the fold-average target without msm=, or tune the map with a "
                    "finite-dimensional tuner and declare target='population'."
                ),
            )
        if self.n_bootstrap:
            raise CapabilityError(
                "target='fold_average' refuses the full-refit bootstrap (n_bootstrap=0). Each "
                "resample refits every fold's map, so each draw has a different target.",
                reason=RefusalReason.INFERENCE_GAP,
                alternative=(
                    "The fit reports the fold-average variance of Hubbard, Kherad-Pajouh and "
                    "van der Laan (2016), Theorem 1."
                ),
            )
        options = {
            "stopping": getattr(self, "stopping", "relative") != "relative",
            "step_bound": getattr(self, "step_bound", None) is not None,
            "iterate": getattr(self, "iterate", "last") != "last",
        }
        declared = [name for name, set_ in options.items() if set_]
        if declared:
            raise CapabilityError(
                f"target='fold_average' is not combined with {', '.join(declared)}=: each "
                "fold's map is targeted by its own intercept fluctuation, and that solve runs "
                "the shipped rule only.",
                reason=RefusalReason.IMPLEMENTATION_GAP,
                roadmap="X75",
                alternative=(
                    "Fit the fold-average target with the default stopping rule, no step "
                    "bound and the last iterate, or declare target='population'."
                ),
            )

    def _refuse_undeclared_policy_scale(
        self, data: LongitudinalData, plans: Sequence[Plan]
    ) -> None:
        """Refuse a continuous outcome under a modified treatment policy with no ``q_bounds``.

        R-f1 of X40, in sample as well as cross-fitted.  It runs before the cross-fitted
        scale refusal, so a policy fit reads the reason that applies to it.  A static or
        dynamic regimen with no policy node keeps the padded in-sample scale.

        Parameters
        ----------
        data : LongitudinalData
            The prepared panel.
        plans : sequence of Plan
            The resolved plans.

        Raises
        ------
        LongitudinalError
            If some plan holds a modified treatment policy, the outcome is not binary and
            ``q_bounds`` is ``None``.
        """
        if data.family == "binomial" or self.q_bounds is not None:
            return
        if any(plan.has_mtp for plan in plans):
            raise LongitudinalError(policy_scale_refusal(str(data.outcome_name)))

    def _refuse_unbounded_cross_fitted_scale(self, data: LongitudinalData) -> None:
        """Refuse a cross-fitted continuous outcome without a declared ``q_bounds``.

        The family is a fact about the data rather than the declaration, so it cannot be
        asked at construction: ``family`` may be inferred from the outcome. It is asked
        before the split is drawn, so a refused design pays for no fold generation and no
        learner.

        A single-fold fit is left alone. It trains on every row, so there is no held-out
        row whose outcome sets the scale it is scored on.
        """
        if self.n_folds <= 1:
            return
        if data.family != "binomial" and self.q_bounds is None:
            raise LongitudinalError(
                f"a cross-fitted longitudinal fit of a continuous outcome "
                f"(family={data.family!r}) needs a declared q_bounds. With q_bounds=None "
                "the outcome scale is taken from every observed outcome, held-out rows "
                "included, so each fold's recursion is fitted on a scale the rows it "
                "predicts helped set, and no shipped result covers that scale "
                "(docs/technical-reference/cv-tmle.md, fold and outcome-scale "
                "rules). Declare the known outcome support "
                "(Targeting(q_bounds=(lower, upper))), or fit in sample "
                "(CrossFitting(enabled=False), or n_folds=1 on the engine)."
            )

    def _multiplier_seed(self) -> int | None:
        """The seed of every multiplier draw of the fit.

        ``random_state`` when it is an integer.  A repeated fit under
        ``random_state=None`` draws one fresh seed here, before any fold draw, and records
        it on :attr:`LongitudinalConfig.multiplier_seed`, so every draw's critical value and
        a narrowed band read the same multipliers.  ``None`` at one draw, unchanged.
        """
        if self.random_state is not None:
            return int(self.random_state)
        return _fresh_seed() if self.repeats > 1 else None

    def _draws(self, data: LongitudinalData) -> list[tuple[Folds, int | None]]:
        """Every draw of the outer split and its seed, each checked for cluster integrity.

        One draw takes ``random_state``, or a fresh seed under ``None``, as an ordinary fit
        always has.  ``repeats=R`` takes the seeds of
        ``CrossFitPlan(random_state=..., repeats=R).seeds()``, spawned from
        ``random_state``, and replaces each ``None`` with a fresh seed.  The integrity
        check is made here rather than inside :meth:`_folds`, so that a subclass that draws
        its own split is held to the same rule: a cluster split across folds puts a unit's
        own cluster in the training fold of the rows it predicts.  A no-op without
        clusters.
        """
        if self._frozen_folds is not None:
            frozen = self._frozen_folds
            seeds_given = self._frozen_seeds
            if seeds_given is not None and len(seeds_given) != len(frozen):
                raise ValueError(
                    "frozen_seeds must hold one seed per frozen split; "
                    f"got {len(seeds_given)} for {len(frozen)} split(s)"
                )
            given: list[tuple[Folds, int | None]] = []
            for index, folds in enumerate(frozen):
                if folds.n != data.n:
                    raise ValueError(
                        f"a frozen split labels {folds.n} rows and the refit has {data.n}"
                    )
                check_integrity(folds, cluster=data.cluster)
                chosen = None if seeds_given is None else seeds_given[index]
                seed = None if chosen is None else int(chosen)
                if seed is None and not folds.is_single:
                    seed = _fresh_seed() if self.random_state is None else int(self.random_state)
                given.append((folds, seed))
            return given
        if self.n_folds <= 1:
            single = Folds.single(data.n)
            check_integrity(single, cluster=data.cluster)
            return [(single, None)]
        if self._frozen_seeds is not None:
            draws_by_seed: list[tuple[Folds, int | None]] = []
            for seed in self._frozen_seeds:
                if seed is None:
                    raise ValueError("frozen_seeds holds no seed for a cross-fitted draw")
                folds = self._folds(data, int(seed))
                check_integrity(folds, cluster=data.cluster)
                draws_by_seed.append((folds, int(seed)))
            return draws_by_seed
        if self.repeats == 1:
            seeds: list[int] = [
                _fresh_seed() if self.random_state is None else int(self.random_state)
            ]
        else:
            planned = CrossFitPlan(
                n_folds=self.n_folds,
                learner_folds=self.learner_folds,
                random_state=self.random_state,
                repeats=self.repeats,
            ).seeds()
            seeds = [_fresh_seed() if seed is None else int(seed) for seed in planned]
        draws: list[tuple[Folds, int | None]] = []
        for seed in seeds:
            folds = self._folds(data, seed)
            check_integrity(folds, cluster=data.cluster)
            draws.append((folds, seed))
        return draws

    def _folds(self, data: LongitudinalData, seed: int | None = None) -> Folds:
        """One unstratified draw of the outer split at ``seed``, or the in-sample single fold.

        The split reads ``n``, the seed and, with ``id=``, the cluster labels, and nothing
        else.  With ``id=`` the draw is grouped: every cluster lands whole in one fold, so a
        validation fold shares no cluster with its training complement.  It used to be stratified on
        the first treatment node, which made the assignment a function of the treatment
        the fit then conditions on; what that stratification bought -- every training
        complement carrying every first-node arm -- is now *checked* on the realized draw
        by :func:`~cleverly.longitudinal.sequential.preflight_mechanism_support`, before
        the first learner runs.
        """
        if self.n_folds <= 1:
            return Folds.single(data.n)
        return random_partition(
            data.n,
            self.n_folds,
            cluster=data.cluster,
            seed=_fresh_seed() if seed is None else seed,
        )


class LTMLE(_SequentialEstimator):
    """Sequential-regression TMLE for a treatment regimen over time.

    Parameters
    ----------
    regimens : mapping, Regimen, DynamicRegimen, or sequence of regimens
        Mapping from label to plan: a sequence of ``T`` arms, or a single arm meaning
        that arm at every node.  ``{"always": 1, "never": 0}`` is the usual pair.  A plan
        with a rule node is a :class:`~cleverly.longitudinal.DynamicRegimen` declared
        ``rule_kind="known"``, as a mapping value or an item of a sequence.  A callable
        written inline in a mapping carries no declaration, and :meth:`fit` refuses it
        before any learner.  A node that draws its arm from a known policy is a
        :class:`~cleverly.interventions.Stochastic` node declared ``density_kind="known"``,
        inside a :class:`~cleverly.longitudinal.DynamicRegimen` or as a mapping value.
    reference : str or None
        Which regimen contrasts are taken against; the first declared by default.
        Part of the estimand rather than a display setting -- ``ate_regimen[a vs b]``
        and ``ate_regimen[a vs c]`` are different parameters.  Refused with ``msm=``.
    horizons : sequence of int or None
        Which time points a **survival** fit reports the cumulative risk at.  ``None``
        reports all of them, which is the curve.  Each horizon is its own backward pass
        -- the pseudo-outcome carried back differs at every node, so nothing is shared
        between them but the mechanism -- and the cost is therefore ``T(T+1)/2``
        regressions per regimen rather than ``T``.  At two or three nodes that is not
        worth a keyword; over a monthly panel it is the difference between a fit and an
        afternoon, so name the horizons you will report.  Refused on a fit with one
        end-of-study outcome, where the only horizon is the end of the study.
    msm : MSM or None
        A working model over the regimen and horizon cells, which reports coefficients
        and no contrasts.  Above one fold, each fold runs the untargeted recursion of
        every cell, and one stacked fluctuation per node pools the update over every
        follower of every cell.
    outcome_learner : object or None
        Learner specification for the regression at the reported horizon, as for
        :class:`~cleverly.TMLE`.
    pseudo_learner : object or None
        Learner specification for the intermediate regressions, whose outcome is a
        ``[0, 1]``-valued prediction rather than the outcome itself.  It defaults to
        ``outcome_learner``'s library read as a regression.
    treatment_learner : object or None
        Learner specification for the treatment mechanism at each node, as for
        :class:`~cleverly.TMLE`.
    censoring_learner : object or None
        Learner specification for the censoring mechanism at each node, as for
        :class:`~cleverly.TMLE`.
    n_folds : int
        Outer cross-fitting folds; one split serves every node and every regimen, so a unit
        is out of fold in all of them at once.  Above one fold the fit follows the
        cross-fitted construction of Díaz, Williams, Hoffman and Schenck (2023, *JASA*
        118(542), Section 5.2, Steps 1-4).  Each fold fits the mechanism and runs an
        *untargeted* backward regression sequence on its training rows, and the held-out
        predictions are stitched into one out-of-fold initial estimate per node.  One pooled
        fluctuation per node then targets those predictions over every follower, with the
        out-of-fold mechanism in its loss weight, so every node's score equation is solved
        as on a single-fold fit.  The fit divides by the out-of-fold mechanism alone.
        With ``msm=``, the fold recursions run for every regimen and horizon cell, and the
        pooled fluctuation at each node is one stacked solve over every live cell.
    learner_folds : int
        The inner folds of the default :class:`~cleverly.learners.SuperLearner` that a
        learner slot left at ``None`` builds.
    g_bounds : float or tuple of float
        Fixed truncation applied to each cumulative treatment-and-censoring probability,
        after multiplying the raw node factors.  The default is the explicit pair
        ``(0.01, 1.0)``, R ``ltmle``'s heuristic convention.  It is not an automatic,
        sample-size-dependent, or follow-up-depth-dependent selection procedure.
    q_bounds : tuple of float or None
        The bounds that map a continuous outcome onto the ``[0, 1]`` scale of the
        recursion.  ``None`` widens the observed range by 10 % on each side.  Above one fold, a
        continuous outcome must declare them, and a binary outcome refuses them.
    alpha : float
        Predicted probabilities are bounded into ``[1 - alpha, alpha]`` before the logit
        is taken, as for :class:`~cleverly.TMLE`.
    alpha_sig : float
        Significance level for confidence intervals, as for :class:`~cleverly.TMLE`.
        The two ``alpha``\\ s mean what they mean there and not the other way round: this
        pair used to be spelled ``alpha`` / ``alpha_shrink`` here, which made
        ``LTMLE(alpha=0.9995)`` a silent 0.05 %-level interval.
    simultaneous : bool
        Whether to report simultaneous confidence bands across the reported parameters,
        via the multiplier bootstrap.  A fit with several regimens reports several
        correlated parameters, which is what the bands are for; see
        :mod:`cleverly.inference.multiplier`.  A fit whose inference status supplies no
        inference builds no band, and :meth:`LongitudinalResult.summary` says so.
    n_multiplier : int
        The number of multiplier draws behind each band.
    multiplier_kind : {"rademacher", "mammen", "normal"}
        The distribution of the multiplier draws.
    max_iter : int
        Maximum targeting iterations per node.
    tol : float
        Targeting tolerance per node.  It is the bar of ``stopping="relative"``.
    stopping : {"relative", "scaled_var", "sample_size"}
        The rule that decides each node's score equation is solved.  ``"relative"`` is the
        shipped test, ``tol`` on the score relative to its largest possible magnitude.
        ``"sample_size"`` stops a node when its score, in the outcome's units, is at most
        ``1 / n_u``, and ``"scaled_var"`` when it is at most ``sd / (sqrt(n_u) * min(log n_u,
        10))``, as ``tmle3`` does.  Here ``n_u`` is the number of distinct units, clusters
        when the data declare ``id=``, among the rows of the panel, and ``sd`` is the sample
        standard deviation of the node's per-unit score contribution.  A node stops at the
        first Newton iterate that meets the rule, so a declared rule changes ``converged``,
        ``n_iter`` and the warning, and it moves the estimate by the part of the last Newton
        step it skips.  Under ``survival_model="hazard"`` the rule judges the score vector of
        each horizon (``targeting="iterative"``) or of every target (``"one_step"``), with
        each target's influence curve as its contribution.
    step_bound : float or None
        The longest step of one iteration of a node's targeting, in the units of the
        fluctuation coefficient.  ``None`` takes the full Newton step.  A positive number takes
        the ``tmle3`` constrained step: the node's single coefficient moves by ``sign(s)`` times
        a length in ``(0, step_bound]``, where ``s`` is the node's score.  A bounded node may
        stop short of ``tol``, and then it warns and reports ``converged=False``.  Under
        ``survival_model="hazard"`` and ``targeting="iterative"`` each horizon steps along
        ``s / ||s||``, with ``s`` its score vector.  The one-step hazard walk takes its steps
        in ``step_size``, so a step bound beside it raises
        :class:`~cleverly.exceptions.MethodConfigurationError`.
    step_search : bool
        With ``step_bound``, ``True`` takes the length that minimises the node's weighted mean
        quasi-loss over ``(0, step_bound]``, and ``False`` takes the whole bound.  It needs
        ``step_bound``.
    iterate : {"last", "best"}
        Which iterate of a node's targeting walk is reported.  ``"last"`` reports the iterate at
        which the walk stopped.  ``"best"`` reports the iterate ``k >= 1`` whose score has the
        smallest 2-norm, the first of any tie, and judges the node's ``converged`` there.  A
        bounded step without a search can oscillate around the root, which is where the two
        differ.  The one-step hazard walk ranks its steps by the norm it walks on and takes the
        last of any tie, so ``"best"`` there reports the last accepted step.
    random_state : int or None
        The seed of the outer split.  A default learner, and a
        :class:`~cleverly.learners.SuperLearner` with no seed of its own, inherit it.
        ``None`` draws a fresh seed for the split.
    run_id : str or None
        An identifier of your own, recorded on :attr:`LongitudinalResult.provenance`.
    n_jobs : int
        Worker count for the parallel fan-out of the recursion.  With ``n_bootstrap > 0``
        the replicates run in parallel and each replicate fit runs with one worker, so
        there is one parallel layer.
    n_bootstrap : int
        Full-refit bootstrap replicates, ``0`` for none.  Otherwise at least 2.  Each
        replicate resamples whole rows of the wide table, or whole clusters, and refits
        everything: the outer split, the treatment and censoring mechanisms, every
        backward regression and fluctuation, and the working-model projection.  A failed
        replicate is dropped and counted, never redrawn.  The bootstrap stream is spawned
        from ``random_state`` and draws nothing the fit itself uses, so the fit's own
        numbers do not move.  A truncation-curve replay does not rerun it.  The percentile
        interval is published as inference with the support label of the fit: ``supported``
        when a registered row lists ``bootstrap`` for the fit, and ``experimental``
        otherwise.  A fit that withholds its interval, or that has fewer clusters than the
        floor, prints ``bootstrap sd`` and a percentile range.  The registered study measures
        it with correctly specified cell-mean nuisances on finite binary laws, at
        ``n = 1000``.  Each replicate also records its own influence-curve standard error,
        which the studentized sup-t band (:attr:`LongitudinalResult.bootstrap_band`) divides
        by.  An in-sample fit with declared mechanisms and frequency-exact learners computes
        each replicate as a frequency-weighted fit on the distinct rows
        (:class:`~cleverly.longitudinal.frequency.FrequencyReplicate`), which equals the
        materialized refit to rounding.
    bootstrap_resampling : {"auto", "iid", "cluster"}
        ``"auto"`` resamples clusters when ``id=`` is declared, and rows otherwise.
    repeats : int
        How many independent draws of the outer split the fit combines, at least 1.  Each
        draw is a complete fit at its own fold assignment and seed, spawned from
        ``random_state`` as :meth:`cleverly.learners.CrossFitPlan.seeds` spawns them, and
        every learner of that draw takes that seed.  The report is the median of the draws
        on each estimate's inference scale, with the split-adjusted variance of
        Chernozhukov et al. (2018, Definition 3.3 and equation 3.14), and
        :attr:`LongitudinalResult.repeats` holds every draw.  Above 1 it needs
        ``n_folds`` of 2 or more and ``n_bootstrap=0``.  At 1 the fit is unchanged, and
        the learners keep ``random_state``.
    bootstrap_interval : {"percentile", "bca"}, default="percentile"
        The pointwise bootstrap interval.  ``"bca"`` is the bias-corrected and accelerated
        interval, with the acceleration from each estimate's own influence curve (summed within
        clusters when clusters are resampled).  It needs ``n_bootstrap``; 199 replicates is
        the X39 study setting.  It carries the label ``experimental`` until a registered row
        lists the ``bootstrap_bca`` kind for the fit.
    survival_model : {"iterated_mean", "hazard"}
        How a survival curve is modelled.  ``"iterated_mean"`` runs one backward
        recursion per horizon, and its curve can decrease.  ``"hazard"`` fits the
        cause-specific hazard at every node and targets it (Moore and van der Laan 2009;
        Cai and van der Laan 2020).  It needs one baseline treatment held over every node,
        categorical, and no covariate after baseline; other designs are refused before any
        learner.  The outcome learner is then the hazard learner.
    targeting : {"iterative", "one_step"}
        The hazard targeting algorithm.  ``"iterative"`` targets each horizon on its own,
        restarting from the initial hazards.  ``"one_step"`` walks one set of hazards until
        every horizon's score is solved, so every curve is non-decreasing.  It requires
        ``survival_model="hazard"``.
    step_size : float
        The first step of the one-step walk, halved after any step that raises the score
        norm.  Accepted and unused under ``targeting="iterative"``.
    isotonic : bool
        Whether to project each regimen's and cause's reported levels, and the limits of
        its simultaneous band, onto a non-decreasing curve over the reported horizons by
        unit-weight isotonic regression (Westling, van der Laan and Carone 2020).  Every
        influence curve, standard error and critical value is unchanged.  A curve with no
        violation is returned bit for bit.
    density_bins : int or None
        The bin count of a continuous node's conditional density.
    ratio : {"density", "classifier"}
        How a continuous node's policy ratio is estimated.
    ratio_trim : float or None
        A quantile in ``(0, 1)`` at which every per-node density ratio is trimmed, as
        ``lmtp``'s ``.trim`` does (:mod:`cleverly.longitudinal.ratio_trim`).  A fit whose
        trim moves a ratio reports no interval.  ``None`` trims nothing.
        ``survival_model="hazard"`` refuses it with
        :class:`~cleverly.exceptions.CapabilityError`, because the hazard targeting has no
        density-ratio matrix to trim.
    hazard_pooling : {"node", "pooled"}
        ``"pooled"`` fits one censoring model over every node's at-risk rows, with the node
        time as a column (:mod:`cleverly.longitudinal.pooled_hazards`).  Under
        ``survival_model="hazard"`` it also fits each cause's hazard once over the eligible
        rows of every node.
    node_bounds : mapping or None
        A declared outcome range for chosen nodes of an iterated-mean fit.  The keys are grid
        times, the values are ``(lower, upper)`` pairs on the reported outcome scale, and a
        competing-risk fit takes one such mapping for each cause name.  A named node regresses
        ``(Y - lower) / (upper - lower)`` with the pseudo-outcome learner, clips the prediction
        to ``[0, 1]`` and maps it back into the bounds.  It fluctuates along the logit of the
        rescaled prediction (Gruber and van der Laan 2010), so its targeted prediction stays
        inside the bounds.  A node with no entry, and ``None``, give the shipped fit bit for
        bit, and so does a node whose bounds equal the outcome range.  The score equation of
        a bounded node has a root only when the weighted mean outcome of the node's followers
        lies strictly inside the bounds.  Otherwise the node runs no Newton step, the fit warns
        with :class:`~cleverly.exceptions.ConvergenceWarning`, and the node reports
        ``failure="bound_excludes_data"`` and ``converged=False``.  A grid time that is not a
        node, a cause the fit did not declare, ``lower >= upper`` and bounds outside the outcome
        range raise ``ValueError`` before any learner.  ``msm=`` raises
        :class:`~cleverly.exceptions.CapabilityError`.  Under ``survival_model="hazard"`` the
        bounds are on the cause-specific hazard, inside ``[0, 1]``, and a named node bounds
        every cause.  Cause :math:`j` there fluctuates along
        :math:`\\operatorname{logit}((\\lambda_j - l_j) / (u_j - l_j))`, the initial hazards
        are clipped into the bounds, and the record holds the clipped share.  A node that
        bounds some causes and not others, or whose upper bounds sum to more than one, raises
        :class:`~cleverly.exceptions.CapabilityError` before any learner.
    **refused : Any
        A point-treatment keyword.  Each one raises :class:`CapabilityError` with the
        reason that a longitudinal fit does not support it.
    """

    _estimator: ClassVar[str] = "tmle"

    def __init__(
        self,
        regimens: Any,
        *,
        reference: str | None = None,
        horizons: Sequence[int] | None = None,
        msm: MSM | None = None,
        outcome_learner: Learner | None = None,
        pseudo_learner: Learner | None = None,
        treatment_learner: Learner | None = None,
        censoring_learner: Learner | None = None,
        n_folds: int = 10,
        learner_folds: int = 5,
        g_bounds: CumulativeGBounds = DEFAULT_LTMLE_G_BOUNDS,
        q_bounds: tuple[float, float] | None = None,
        alpha: float = 0.9995,
        alpha_sig: float = 0.05,
        simultaneous: bool = True,
        n_multiplier: int = 2000,
        multiplier_kind: str = "rademacher",
        max_iter: int = 20,
        tol: float = 1e-10,
        random_state: int | None = None,
        run_id: str | None = None,
        n_jobs: int = 1,
        n_bootstrap: int = 0,
        bootstrap_resampling: Resampling = "auto",
        bootstrap_interval: BootstrapInterval = "percentile",
        density_bins: int | None = None,
        ratio: str = "density",
        repeats: int = 1,
        stopping: StoppingRule = "relative",
        step_bound: float | None = None,
        step_search: bool = True,
        iterate: IterateChoice = "last",
        survival_model: str = "iterated_mean",
        targeting: str = "iterative",
        step_size: float = 1e-3,
        isotonic: bool = False,
        ratio_trim: float | None = None,
        hazard_pooling: str = "node",
        node_bounds: Mapping[Any, Any] | None = None,
        **refused: Any,
    ) -> None:
        self.node_bounds = node_bounds
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.stopping = stopping
        self.step_bound = step_bound
        self.step_search = step_search
        self.iterate = iterate
        self.survival_model = survival_model
        self.targeting = targeting
        self.step_size = step_size
        self.isotonic = isotonic
        super().__init__(
            regimens,
            reference=reference,
            horizons=horizons,
            msm=msm,
            outcome_learner=outcome_learner,
            pseudo_learner=pseudo_learner,
            treatment_learner=treatment_learner,
            censoring_learner=censoring_learner,
            n_folds=n_folds,
            learner_folds=learner_folds,
            g_bounds=g_bounds,
            q_bounds=q_bounds,
            alpha_sig=alpha_sig,
            simultaneous=simultaneous,
            n_multiplier=n_multiplier,
            multiplier_kind=multiplier_kind,
            random_state=random_state,
            run_id=run_id,
            n_jobs=n_jobs,
            n_bootstrap=n_bootstrap,
            bootstrap_resampling=bootstrap_resampling,
            bootstrap_interval=bootstrap_interval,
            density_bins=density_bins,
            ratio=ratio,
            ratio_trim=ratio_trim,
            hazard_pooling=hazard_pooling,
            repeats=repeats,
            **refused,
        )

    def _validate_targeting(self) -> None:
        """Refuse a shrink outside ``(0.5, 1)`` and an iteration limit below one."""
        if not 0.5 < self.alpha < 1.0:
            raise ValueError(
                f"alpha must lie in (0.5, 1); got {self.alpha}. Predicted probabilities "
                "are bounded into [1 - alpha, alpha]; alpha_sig= is the significance "
                "level, and defaults to 0.05"
            )
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be at least 1; got {self.max_iter}")
        check_iterate(self.iterate, "logistic")
        check_stopping_rule(self.stopping, "logistic")
        check_step_bound(self.step_bound, self.step_search, "logistic")
        if self.node_bounds is not None:
            if not isinstance(self.node_bounds, Mapping):
                raise ValueError(
                    "node_bounds must map grid times to (lower, upper) pairs, or one such "
                    f"mapping to each cause; got {type(self.node_bounds).__name__}"
                )
            if self.msm is not None:
                raise CapabilityError(refusal=NODE_BOUNDS_MSM_REFUSAL)

    def _targeting(self) -> dict[str, Any]:
        """The logistic shrink, the iteration limit, the score tolerance and the step rules."""
        return {
            "alpha": self.alpha,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "stopping": self.stopping,
            "step_bound": self.step_bound,
            "step_search": self.step_search,
            "iterate": self.iterate,
            "node_bounds": None,
        }


class SDR(_SequentialEstimator):
    r"""Sequentially doubly robust estimation of a treatment regimen over time.

    The estimator of Díaz, Williams, Hoffman and Schenck (2023, *JASA* 118(542), Section
    5.3).  It runs the backward sequential regression of :class:`LTMLE` and fits no
    fluctuation.  Instead, the regression at each node is fitted to the transformed outcome
    of the paper's Equation (4), which adds every later node's weighted residual to the
    carried prediction.  The estimate is the weighted mean of the first node's transform,
    and the influence curve is that transform minus the estimate.  The estimate stays
    consistent when, at each node, either the outcome regression or the treatment and
    censoring mechanism is correct (the paper's Lemma 4).  An interval needs the rate
    conditions of the paper's Theorem 4.  See
    :mod:`cleverly.longitudinal.sdr` for the recursion.

    **A repeated outcome.**  With ``outcome=Repeated([...])`` the fit reports one mean per
    plan and horizon.  Each horizon is the end-of-study fit of the panel truncated after it,
    with :math:`Y_h` as the outcome and each earlier :math:`Y_s` as a covariate.  Equation
    (4), Lemmas 3 and 4 and Theorem 4 of the paper apply to each horizon on its prefix.  So
    the estimate of horizon :math:`h` is consistent when, at each node :math:`t \le h`, the
    ratio or the sequential regression is consistent.  The band of ``simultaneous=True``
    covers every plan and horizon jointly, from the stacked influence curves.

    A transform can leave ``[0, 1]``, so the default pseudo learner is a regression
    :class:`~cleverly.learners.SuperLearner` with no clip, as ``lmtp`` fits it.  A pseudo
    learner that clips is accepted, and it moves the estimate away from ``lmtp_sdr``.  A
    learner that reads its target as a probability raises its own error on a target outside
    ``[0, 1]``.

    Parameters
    ----------
    regimens : mapping, Regimen, DynamicRegimen, or sequence of regimens
        The regimens, as for :class:`LTMLE`.
    reference : str or None
        The regimen every contrast is taken against, as for :class:`LTMLE`.
    horizons : sequence of int or None
        The horizons of a survival or repeated outcome, as for :class:`LTMLE`.  ``None``
        reports every node.  A repeated outcome reports one mean per plan and horizon, each
        from the truncated end-of-study fit.  Refused on a fit with one end-of-study outcome.
    msm : MSM or None
        A working model over the regimen and horizon cells.  Each cell's first-node
        transform is projected onto it.  The identity link is supported.  A logit or log
        projection that does not converge is refused.
    outcome_learner : object or None
        The regression at the reported horizon, as for :class:`LTMLE`.
    pseudo_learner : object or None
        The regression of every earlier node, on a real-valued target.  It defaults to
        ``outcome_learner``, and otherwise to a regression
        :class:`~cleverly.learners.SuperLearner` with ``clip=None``.
    treatment_learner : object or None
        The treatment mechanism at each node, as for :class:`LTMLE`.
    censoring_learner : object or None
        The censoring mechanism at each node, as for :class:`LTMLE`.
    n_folds : int
        Outer cross-fitting folds.  Above one fold, each fold fits the mechanism and the
        backward recursion on its training rows alone, with its own mechanism in every
        training row's transform.  Its held-out rows take the first-node transform from
        that fold's models, and the estimate pools every row.
    learner_folds : int
        The inner folds of a default :class:`~cleverly.learners.SuperLearner`.
    g_bounds : float or tuple of float
        The bound pair applied to each partial path probability
        :math:`\prod_{k=t}^{s} p_k`.  At :math:`t = 1` it is the cumulative bound of
        :class:`LTMLE`.  The default is ``(0.01, 1.0)``.
    q_bounds : tuple of float or None
        The bounds of a continuous outcome, as for :class:`LTMLE`.
    alpha_sig : float
        Significance level for confidence intervals.
    simultaneous : bool
        Whether to report simultaneous confidence bands, as for :class:`LTMLE`.
    n_multiplier : int
        The number of multiplier draws behind each band.
    multiplier_kind : {"rademacher", "mammen", "normal"}
        The distribution of the multiplier draws.
    random_state : int or None
        The seed of the outer split, inherited by a default learner.
    run_id : str or None
        An identifier of your own, recorded on :attr:`LongitudinalResult.provenance`.
    n_jobs : int
        Worker count for the parallel fan-out.
    n_bootstrap : int
        Full-refit bootstrap replicates, ``0`` for none.  No registered row lists a
        bootstrap kind for an ``SDR`` fit, so its intervals carry the label
        ``experimental``.  The replicates are computed as for :class:`LTMLE`.
    bootstrap_resampling : {"auto", "iid", "cluster"}
        ``"auto"`` resamples clusters when ``id=`` is declared, and rows otherwise.
    bootstrap_interval : {"percentile", "bca"}, default="percentile"
        The pointwise bootstrap interval, as for :class:`LTMLE`.  It carries the label
        ``experimental``.
    density_bins : int or None
        The bin count of a continuous node's conditional density, as for :class:`LTMLE`.
    ratio : {"density", "classifier"}
        How a continuous node's policy ratio is estimated, as for :class:`LTMLE`.
    ratio_trim : float or None
        The per-node ratio trim, as for :class:`LTMLE`.  Each partial weight
        :math:`W_{t,s}` is then the product of the trimmed ratios, as in ``lmtp``
        (:func:`~cleverly.longitudinal.sdr.trimmed_factors`).
    hazard_pooling : {"node", "pooled"}
        The pooling, as for :class:`LTMLE`.  Each cross-fitted fold pools its own censoring
        model.
    repeats : int
        How many independent draws of the outer split the fit combines by their median,
        as for :class:`LTMLE`.
    **refused : Any
        ``alpha``, ``max_iter``, ``tol``, ``stopping``, ``step_bound``, ``step_search`` and
        ``iterate``, which raise :class:`MethodConfigurationError` because the estimator fits no
        fluctuation, and each point-treatment keyword that :class:`LTMLE` refuses.

    See Also
    --------
    LTMLE : The targeted estimator of the same parameters.
    sdr : The one-call form of this estimator.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression, LogisticRegression
    >>> from cleverly.datasets import make_longitudinal
    >>> from cleverly.longitudinal import SDR
    >>> frame, _ = make_longitudinal(n=200, seed=0)
    >>> result = SDR(
    ...     {"always": 1, "never": 0},
    ...     n_folds=1,
    ...     outcome_learner=LinearRegression(),
    ...     treatment_learner=LogisticRegression(max_iter=1000),
    ...     censoring_learner=LogisticRegression(max_iter=1000),
    ... ).fit(
    ...     frame,
    ...     outcome="Y",
    ...     treatment=["A1", "A2"],
    ...     baseline=["W1", "W2"],
    ...     time_varying=[[], ["L2"]],
    ...     censoring=["C1", "C2"],
    ... )
    >>> sorted(result.estimates)
    ['ate_regimen[never vs always]', 'ey_regimen[always]', 'ey_regimen[never]']
    >>> result.config.estimator
    'sdr'
    """

    _estimator: ClassVar[str] = "sdr"

    def __init__(
        self,
        regimens: Any,
        *,
        reference: str | None = None,
        horizons: Sequence[int] | None = None,
        msm: MSM | None = None,
        outcome_learner: Learner | None = None,
        pseudo_learner: Learner | None = None,
        treatment_learner: Learner | None = None,
        censoring_learner: Learner | None = None,
        n_folds: int = 10,
        learner_folds: int = 5,
        g_bounds: CumulativeGBounds = DEFAULT_LTMLE_G_BOUNDS,
        q_bounds: tuple[float, float] | None = None,
        alpha_sig: float = 0.05,
        simultaneous: bool = True,
        n_multiplier: int = 2000,
        multiplier_kind: str = "rademacher",
        random_state: int | None = None,
        run_id: str | None = None,
        n_jobs: int = 1,
        n_bootstrap: int = 0,
        bootstrap_resampling: Resampling = "auto",
        bootstrap_interval: BootstrapInterval = "percentile",
        density_bins: int | None = None,
        ratio: str = "density",
        ratio_trim: float | None = None,
        hazard_pooling: str = "node",
        repeats: int = 1,
        **refused: Any,
    ) -> None:
        if "node_bounds" in refused:
            raise MethodConfigurationError.from_refusal(node_bounds_refusal("the SDR estimator"))
        named = [name for name in SDR_REFUSED_SETTINGS if name in refused]
        if named:
            raise MethodConfigurationError.from_refusal(sdr_no_fluctuation_refusal(named[0]))
        super().__init__(
            regimens,
            reference=reference,
            horizons=horizons,
            msm=msm,
            outcome_learner=outcome_learner,
            pseudo_learner=pseudo_learner,
            treatment_learner=treatment_learner,
            censoring_learner=censoring_learner,
            n_folds=n_folds,
            learner_folds=learner_folds,
            g_bounds=g_bounds,
            q_bounds=q_bounds,
            alpha_sig=alpha_sig,
            simultaneous=simultaneous,
            n_multiplier=n_multiplier,
            multiplier_kind=multiplier_kind,
            random_state=random_state,
            run_id=run_id,
            n_jobs=n_jobs,
            n_bootstrap=n_bootstrap,
            bootstrap_resampling=bootstrap_resampling,
            bootstrap_interval=bootstrap_interval,
            density_bins=density_bins,
            ratio=ratio,
            ratio_trim=ratio_trim,
            hazard_pooling=hazard_pooling,
            repeats=repeats,
            **refused,
        )

    def _resolve_tilts(self, data: LongitudinalData, plans: tuple[Plan, ...]) -> tuple[Plan, ...]:
        """Refuse an odds tilt node: an SDR fit needs a policy that does not read the law.

        Parameters
        ----------
        data : LongitudinalData
            The prepared panel.
        plans : tuple of Plan
            The resolved plans.

        Returns
        -------
        tuple of Plan
            The plans, unchanged.

        Raises
        ------
        CapabilityError
            When a plan holds a tilt node.
        """
        if any(plan.has_tilt for plan in plans):
            raise CapabilityError.from_refusal(SDR_TILT_REFUSAL)
        return plans

    def _pseudo_learner(self, seed: int | None) -> Learner:
        """The learner of every node before the horizon, unclipped by default.

        A transform leaves ``[0, 1]``, and ``lmtp`` regresses it as it is.  So the default
        is the regression :class:`~cleverly.learners.SuperLearner` without the clip that
        :func:`~cleverly.learners.super_learner.resolve_learner` puts on its default.
        """
        if self.pseudo_learner is None and self.outcome_learner is None:
            return SuperLearner(
                task="regression",
                n_folds=self.learner_folds,
                clip=None,
                random_state=seed,
                n_jobs=1,
            )
        return super()._pseudo_learner(seed)


#: The settings :class:`IPSI` refuses with :data:`~cleverly.methods.IPSI_NO_FLUCTUATION`.
IPSI_REFUSED_SETTINGS: tuple[str, ...] = (
    "alpha",
    "max_iter",
    "tol",
    "stopping",
    "step_bound",
    "step_search",
    "iterate",
    "targeting",
    "fluctuation",
)


class IPSI(_SequentialEstimator):
    r"""Incremental propensity-score interventions over time, by Kennedy's estimator.

    An odds tilt multiplies each unit's odds of treatment at a node by :math:`\delta_t`
    (Kennedy 2019, Equation (1)).  A node with three or more levels takes one factor per
    level, :math:`q_t(a \mid H_t) \propto \delta_{t,a}\pi_t(a \mid H_t)`, declared as
    ``Incremental({level: factor})``.  The plan is identified without treatment positivity
    at its tilt nodes (Theorem 1).  The estimator is Algorithm 1 of Kennedy (2019), with the
    dropout weights of Kim, Kennedy and Naimi (2021, Algorithm 1): the mean of the efficient
    influence function at nuisances fitted by the training-fold recursion.  It fits no
    fluctuation.  The influence function has a mechanism term at each tilt node, so the
    estimator is not doubly robust: the treatment mechanism must be consistent at every
    tilt node (Kennedy 2019, Section 3.3).  See :mod:`cleverly.longitudinal.ipsi` for the
    recursion.

    ``IPSI`` is Kennedy's odds tilt.  ``lmtp::ipsi`` is the risk-ratio tilt of Díaz et al.
    (2023), which this package fits as :class:`~cleverly.interventions.RiskRatioTilt` with
    :class:`LTMLE` or :class:`SDR`.

    **A repeated outcome.**  With ``outcome=Repeated([...])`` the fit reports one mean per
    horizon.  Each horizon is the end-of-study fit of the panel truncated after it, with
    :math:`Y_h` as the outcome and each earlier :math:`Y_s` as a covariate.  The functional at
    horizon :math:`h` is Theorem 3.1 of Kim, Kennedy and Naimi (2021) at :math:`t = h`.  The
    influence function is their Theorem 4.1 on the prefix.  The band of ``simultaneous=True``
    covers the declared tilts and horizons jointly, which is their Theorem 5.1 restricted to
    the declared finite grid of :math:`\delta`.  It does not cover the values of :math:`\delta`
    between the declared ones.  The estimator is singly robust: a wrong treatment mechanism
    leaves a bias that a correct outcome regression does not remove.

    Parameters
    ----------
    regimens : mapping, sequence of Incremental, or sequence of regimens
        The plans.  A sequence of :class:`~cleverly.interventions.Incremental` objects is
        shorthand: each one tilts every node, and its ``name`` is the label.  One
        ``Incremental`` applies to every node, so a scalar needs two levels at each node and
        a mapping key must be a level of each node.  A mapping
        value ``Incremental(2.0)`` tilts every node.  A
        :class:`~cleverly.longitudinal.DynamicRegimen` can hold a different tilt per node,
        and label, rule and known-policy nodes beside the tilts.  A plan with no tilt node
        is accepted, so a tilt can be contrasted with a static plan on one fit.
    reference : str or None
        The regimen every contrast is taken against, the first declared by default.  It
        must be one of ``regimens``: list ``Incremental(1.0)`` to contrast with the natural
        course.  Under dropout, :math:`\psi(1)` is the mean with dropout prevented, not the
        observed mean.
    horizons : sequence of int or None
        The horizons of a survival or repeated outcome, as for :class:`LTMLE`.  ``None``
        reports every node.  A repeated outcome reports one mean per horizon, each from the
        truncated end-of-study fit.  Refused on a fit with one end-of-study outcome.
    msm : MSM or None
        A working model over the regimen and horizon cells.  Each cell's influence-function
        values are projected onto it.  The identity link is supported.  A logit or log
        projection that does not converge is refused.
    outcome_learner : object or None
        The regression at the reported horizon, as for :class:`LTMLE`.
    pseudo_learner : object or None
        The regression of every earlier node.  It defaults to ``outcome_learner``, and
        otherwise to a regression :class:`~cleverly.learners.SuperLearner` with no clip,
        as ``npcausal::ipsi`` fits it.
    treatment_learner : object or None
        The treatment mechanism at each node.  At a tilt node it is part of the estimand.
    censoring_learner : object or None
        The censoring mechanism at each node.
    n_folds : int
        Outer cross-fitting folds.  Above one fold, each fold fits every nuisance on its
        training rows alone, and its held-out rows take the influence function from that
        fold's models (Kennedy 2019, Algorithm 1).  One fold is Kennedy's Algorithm 2, which
        needs Donsker-type conditions.
    learner_folds : int
        The inner folds of a default :class:`~cleverly.learners.SuperLearner`.
    g_bounds : float or tuple of float
        The bound pair of the cumulative censoring, label, rule and policy factors.  The
        tilt factor is never bounded.  The default is ``(0.01, 1.0)``.
    q_bounds : tuple of float or None
        The bounds of a continuous outcome, as for :class:`LTMLE`.
    alpha_sig : float
        Significance level for confidence intervals.
    simultaneous : bool
        Whether to report simultaneous confidence bands over the plans and horizons.  The
        band covers the declared plans only, not a continuum of :math:`\delta`.
    n_multiplier : int
        The number of multiplier draws behind each band.
    multiplier_kind : {"rademacher", "mammen", "normal"}
        The distribution of the multiplier draws.
    random_state : int or None
        The seed of the outer split, inherited by a default learner.
    run_id : str or None
        An identifier of your own, recorded on :attr:`LongitudinalResult.provenance`.
    n_jobs : int
        Worker count for the parallel fan-out.
    n_bootstrap : int
        Full-refit bootstrap replicates, ``0`` for none.  No registered study measures the
        IPSI bootstrap, so the draws report a bootstrap sd and no percentile inference.
    bootstrap_resampling : {"auto", "iid", "cluster"}
        ``"auto"`` resamples clusters when ``id=`` is declared, and rows otherwise.
    hazard_pooling : {"node", "pooled"}
        The pooling, as for :class:`LTMLE`.
    repeats : int
        How many independent draws of the outer split the fit combines by their median,
        as for :class:`LTMLE`.
    **refused : Any
        ``alpha``, ``max_iter``, ``tol``, ``stopping``, ``step_bound``, ``step_search``,
        ``iterate``, ``targeting`` and ``fluctuation``, which raise
        :class:`MethodConfigurationError` because the estimator fits no fluctuation, and each
        point-treatment keyword that :class:`LTMLE` refuses.  ``ratio_trim`` other than
        ``None`` raises :class:`~cleverly.exceptions.CapabilityError`
        (:data:`~cleverly.methods.IPSI_RATIO_TRIM_REFUSAL`).

    See Also
    --------
    LTMLE : The targeted estimator of static, dynamic and known-policy plans.
    ipsi : The one-call form of this estimator.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression, LogisticRegression
    >>> from cleverly.datasets import make_longitudinal
    >>> from cleverly.interventions import Incremental
    >>> from cleverly.longitudinal import IPSI
    >>> frame, _ = make_longitudinal(n=200, seed=0)
    >>> result = IPSI(
    ...     [Incremental(1.0), Incremental(2.0)],
    ...     n_folds=1,
    ...     outcome_learner=LinearRegression(),
    ...     treatment_learner=LogisticRegression(max_iter=1000),
    ...     censoring_learner=LogisticRegression(max_iter=1000),
    ... ).fit(
    ...     frame,
    ...     outcome="Y",
    ...     treatment=["A1", "A2"],
    ...     baseline=["W1", "W2"],
    ...     time_varying=[[], ["L2"]],
    ...     censoring=["C1", "C2"],
    ... )
    >>> sorted(result.estimates)
    ['ate_regimen[odds x2 vs natural course]', 'ey_regimen[natural course]', 'ey_regimen[odds x2]']
    >>> result.config.estimator
    'ipsi'
    """

    _estimator: ClassVar[str] = "ipsi"

    def __init__(
        self,
        regimens: Any,
        *,
        reference: str | None = None,
        horizons: Sequence[int] | None = None,
        msm: MSM | None = None,
        outcome_learner: Learner | None = None,
        pseudo_learner: Learner | None = None,
        treatment_learner: Learner | None = None,
        censoring_learner: Learner | None = None,
        n_folds: int = 10,
        learner_folds: int = 5,
        g_bounds: CumulativeGBounds = DEFAULT_LTMLE_G_BOUNDS,
        q_bounds: tuple[float, float] | None = None,
        alpha_sig: float = 0.05,
        simultaneous: bool = True,
        n_multiplier: int = 2000,
        multiplier_kind: str = "rademacher",
        random_state: int | None = None,
        run_id: str | None = None,
        n_jobs: int = 1,
        n_bootstrap: int = 0,
        bootstrap_resampling: Resampling = "auto",
        hazard_pooling: str = "node",
        repeats: int = 1,
        **refused: Any,
    ) -> None:
        if "node_bounds" in refused:
            raise MethodConfigurationError.from_refusal(node_bounds_refusal("IPSI"))
        if any(name in refused for name in IPSI_REFUSED_SETTINGS):
            raise MethodConfigurationError.from_refusal(IPSI_NO_FLUCTUATION)
        if refused.pop("ratio_trim", None) is not None:
            raise CapabilityError.from_refusal(IPSI_RATIO_TRIM_REFUSAL)
        super().__init__(
            _tilt_shorthand(regimens),
            reference=reference,
            horizons=horizons,
            msm=msm,
            outcome_learner=outcome_learner,
            pseudo_learner=pseudo_learner,
            treatment_learner=treatment_learner,
            censoring_learner=censoring_learner,
            n_folds=n_folds,
            learner_folds=learner_folds,
            g_bounds=g_bounds,
            q_bounds=q_bounds,
            alpha_sig=alpha_sig,
            simultaneous=simultaneous,
            n_multiplier=n_multiplier,
            multiplier_kind=multiplier_kind,
            random_state=random_state,
            run_id=run_id,
            n_jobs=n_jobs,
            n_bootstrap=n_bootstrap,
            bootstrap_resampling=bootstrap_resampling,
            hazard_pooling=hazard_pooling,
            repeats=repeats,
            **refused,
        )

    @records_fit
    def fit(self, data: Any, **kwargs: Any) -> LongitudinalResult:
        """Fit on a wide dataframe, or on an already-built :class:`LongitudinalData`.

        The keywords are those of :meth:`LTMLE.fit`.  A declared treatment mechanism at a
        tilt node removes that node's mechanism term from the influence function, because
        the tilt is then a known policy (Kennedy 2019, Section 3.3).

        Parameters
        ----------
        data : DataFrame or LongitudinalData
            One wide frame, a row per unit and a column per node, or a container.
        **kwargs : Any
            The column and declaration keywords of :meth:`LTMLE.fit`.

        Returns
        -------
        LongitudinalResult
            The plan estimates, their contrasts or working-model coefficients, and the fits
            behind them.

        Raises
        ------
        MethodConfigurationError
            When ``data`` is a point-treatment :class:`~cleverly.CausalData`.
        """
        from ..data.causal_data import CausalData

        if isinstance(data, CausalData):
            raise MethodConfigurationError.from_refusal(IPSI_POINT_REFUSAL)
        return super().fit(data, **kwargs)

    def _resolve_tilts(self, data: LongitudinalData, plans: tuple[Plan, ...]) -> tuple[Plan, ...]:
        """Refuse a plan node this estimator has no ratio for, before any learner.

        Parameters
        ----------
        data : LongitudinalData
            The prepared panel.
        plans : tuple of Plan
            The resolved plans.

        Returns
        -------
        tuple of Plan
            The plans, unchanged.

        Raises
        ------
        CapabilityError
            When a plan sets a node by a modified treatment policy or a risk-ratio tilt, or
            when the design holds one baseline decision over every node.
        """
        if data.treatment_held:
            raise CapabilityError(
                "IPSI fits one treatment decision per node, and this design holds one "
                "baseline decision over every node.",
                reason=RefusalReason.API_REDIRECTION,
                alternative="For one decision fit TMLE(incremental=[...]) on the baseline row.",
            )
        for plan in plans:
            for time in plan.mtp_nodes:
                node = plan.regimen.plan[time - 1]  # type: ignore[union-attr]
                kind = type(node).__name__
                raise CapabilityError.from_refusal(
                    tilt_mix_refusal(
                        plan.label,
                        data.treatment_names[time - 1],
                        "a risk-ratio tilt"
                        if kind == "RiskRatioTilt"
                        else f"a modified treatment policy ({kind})",
                    )
                )
        return plans

    def _pseudo_learner(self, seed: int | None) -> Learner:
        """The learner of every node before the horizon, unclipped by default."""
        if self.pseudo_learner is None and self.outcome_learner is None:
            return SuperLearner(
                task="regression",
                n_folds=self.learner_folds,
                clip=None,
                random_state=seed,
                n_jobs=1,
            )
        return super()._pseudo_learner(seed)


class _Baseline(_SequentialEstimator, metaclass=ABCMeta):
    """The settings the two baseline estimators share and refuse.

    Both fit in sample with no fluctuation and no working model, and every learner defaults to
    a main-terms :class:`~cleverly.learners.GLM`, as R ``ltmle``'s ``SL.library="glm"``.  The
    class is abstract: each baseline names the regressions its status reads.
    """

    def _fit_diagnostic_mechanism(
        self,
        data: LongitudinalData,
        plans: tuple[Plan, ...],
        *,
        treatment_learner: Learner,
        censoring_learner: Learner,
        folds: Folds,
    ) -> Mechanism:
        """Fit the diagnostic mechanism, and degrade the diagnostics when it cannot fit.

        Parameters
        ----------
        data : LongitudinalData
            The prepared panel.
        plans : tuple of Plan
            The resolved plans.
        treatment_learner : Learner
            The treatment mechanism learner.
        censoring_learner : Learner
            The censoring mechanism learner.
        folds : Folds
            The single in-sample fold.

        Returns
        -------
        Mechanism
            The fitted mechanism, or one that holds ``nan`` everywhere.

        Warns
        -----
        PositivityWarning
            When the mechanism cannot fit.
        """
        try:
            return fit_mechanism(
                data,
                plans,
                treatment_learner=treatment_learner,
                censoring_learner=censoring_learner,
                folds=folds,
                n_jobs=self.n_jobs,
                density_bins=self.density_bins,
                ratio=self.ratio,
            )
        except DataError as error:
            _warn(GCOMP_MECHANISM_WARNING.format(error=error), PositivityWarning, stacklevel=5)
            return _unavailable_mechanism(data, plans)

    @abstractmethod
    def _parametric(
        self,
        outcome_learner: Learner,
        pseudo_learner: Learner,
        mechanism: Mechanism,
        horizons: Sequence[int],
        plans: Sequence[Plan],
    ) -> bool:
        """Whether every regression the estimator reads is a GLM.

        Parameters
        ----------
        outcome_learner : Learner
            The resolved outcome learner.
        pseudo_learner : Learner
            The resolved pseudo-outcome learner.
        mechanism : Mechanism
            The fitted mechanism.
        horizons : sequence of int
            The reported horizons.
        plans : sequence of Plan
            The resolved plans.

        Returns
        -------
        bool
            Whether the status may report an interval.
        """

    def __init__(
        self,
        regimens: Any,
        *,
        reference: str | None = None,
        horizons: Sequence[int] | None = None,
        outcome_learner: Learner | None = None,
        pseudo_learner: Learner | None = None,
        treatment_learner: Learner | None = None,
        censoring_learner: Learner | None = None,
        n_folds: int = 1,
        g_bounds: CumulativeGBounds = DEFAULT_LTMLE_G_BOUNDS,
        q_bounds: tuple[float, float] | None = None,
        alpha_sig: float = 0.05,
        simultaneous: bool = True,
        n_multiplier: int = 2000,
        multiplier_kind: str = "rademacher",
        random_state: int | None = None,
        run_id: str | None = None,
        n_jobs: int = 1,
        n_bootstrap: int = 0,
        bootstrap_resampling: Resampling = "auto",
        density_bins: int | None = None,
        ratio: str = "density",
        msm: MSM | None = None,
        **refused: Any,
    ) -> None:
        if "node_bounds" in refused:
            raise MethodConfigurationError.from_refusal(
                node_bounds_refusal("the sequential g-computation and IPW estimators")
            )
        named = [name for name in SDR_REFUSED_SETTINGS if name in refused]
        if named:
            raise MethodConfigurationError.from_refusal(baseline_no_fluctuation_refusal(named[0]))
        if msm is not None:
            raise CapabilityError(refusal=BASELINE_MSM_REFUSAL)
        if refused.get("ratio_trim") is not None:
            raise CapabilityError(refusal=BASELINE_RATIO_TRIM_REFUSAL)
        if refused.get("hazard_pooling", "node") != "node":
            raise CapabilityError(refusal=BASELINE_HAZARD_POOLING_REFUSAL)
        super().__init__(
            regimens,
            reference=reference,
            horizons=horizons,
            outcome_learner=outcome_learner,
            pseudo_learner=pseudo_learner,
            treatment_learner=GLM() if treatment_learner is None else treatment_learner,
            censoring_learner=GLM() if censoring_learner is None else censoring_learner,
            n_folds=n_folds,
            g_bounds=g_bounds,
            q_bounds=q_bounds,
            alpha_sig=alpha_sig,
            simultaneous=simultaneous,
            n_multiplier=n_multiplier,
            multiplier_kind=multiplier_kind,
            random_state=random_state,
            run_id=run_id,
            n_jobs=n_jobs,
            n_bootstrap=n_bootstrap,
            bootstrap_resampling=bootstrap_resampling,
            density_bins=density_bins,
            ratio=ratio,
            **refused,
        )

    def _validate_settings(self) -> None:
        """Refuse cross-fitting before any other setting is read."""
        if isinstance(self.repeats, (int, np.integer)) and self.repeats > 1:
            raise CapabilityError(refusal=baseline_repeats_refusal(type(self).__name__))
        if self.n_folds > 1:
            raise CapabilityError(refusal=baseline_cross_fit_refusal(type(self).__name__))
        super()._validate_settings()

    def _resolve_tilts(self, data: LongitudinalData, plans: tuple[Plan, ...]) -> tuple[Plan, ...]:
        """Read a tilt with a known mechanism as a known policy, and refuse the rest.

        Parameters
        ----------
        data : LongitudinalData
            The prepared panel.
        plans : tuple of Plan
            The resolved plans.

        Returns
        -------
        tuple of Plan
            The plans, with each tilt node read as a known policy.

        Raises
        ------
        CapabilityError
            When a tilt node's treatment mechanism is estimated.
        """
        try:
            return super()._resolve_tilts(data, plans)
        except CapabilityError as error:
            raise CapabilityError(refusal=BASELINE_TILT_REFUSAL) from error


class SequentialGComputation(_Baseline):
    r"""Sequential g-computation of a treatment regimen over time.

    The iterated conditional expectation form of the longitudinal g-formula (Bang and Robins
    2005, Section 4).  It runs the backward sequential regression of :class:`LTMLE` with no
    fluctuation, and the estimate is the weighted mean of the first node's prediction.  See
    :mod:`cleverly.longitudinal.baselines`.

    The estimate is consistent only when every outcome regression is correctly specified.
    With every regression a :class:`~cleverly.learners.GLM`, the interval is the stacked
    estimating equation of the GLM fits and the estimate (Stefanski and Boos 2002, Section 2).
    That interval is valid for the target only when every GLM is correctly specified.  Under
    misspecification it covers the limit of the misspecified fit.  With another learner the
    fit takes the ``"adaptive_baseline_plugin"`` status: the point estimate stands, and the
    diagnostic curve is the efficient influence curve at the untargeted fits.

    The treatment and censoring mechanisms are fitted for the positivity diagnostics only.
    The estimate does not read them, and ``g_bounds`` moves only those diagnostics.

    Parameters
    ----------
    regimens : mapping, Regimen, DynamicRegimen, or sequence of regimens
        The regimens, as for :class:`LTMLE`.  Modified treatment policies are accepted: the
        recursion predicts each node regression at the policy's dose.
    reference : str or None
        The regimen every contrast is taken against, as for :class:`LTMLE`.
    horizons : sequence of int or None
        The survival horizons, as for :class:`LTMLE`.
    outcome_learner : object or None
        The regression at the reported horizon.  The default is a main-terms
        :class:`~cleverly.learners.GLM`.
    pseudo_learner : object or None
        The regression of every earlier node.  It defaults to ``outcome_learner``.
    treatment_learner : object or None
        The treatment mechanism of the diagnostics.  The default is a
        :class:`~cleverly.learners.GLM`.
    censoring_learner : object or None
        The censoring mechanism of the diagnostics.  The default is a
        :class:`~cleverly.learners.GLM`.
    n_folds : int
        Must be ``1``.  The estimator fits in sample.
    g_bounds : float or tuple of float
        The cumulative bound of the diagnostic clever covariate.
    q_bounds : tuple of float or None
        The known support of a continuous outcome.  A binomial-family GLM on a continuous
        outcome needs it.
    alpha_sig : float
        Significance level for confidence intervals.
    simultaneous : bool
        Whether to report simultaneous confidence bands, as for :class:`LTMLE`.
    n_multiplier : int
        The number of multiplier draws behind each band.
    multiplier_kind : {"rademacher", "mammen", "normal"}
        The distribution of the multiplier draws.
    random_state : int or None
        The seed inherited by a default learner.
    run_id : str or None
        An identifier of your own, recorded on :attr:`LongitudinalResult.provenance`.
    n_jobs : int
        Worker count for the parallel fan-out.
    n_bootstrap : int
        Full-refit bootstrap replicates, ``0`` for none.  The draws report a bootstrap sd and
        no percentile inference.
    bootstrap_resampling : {"auto", "iid", "cluster"}
        ``"auto"`` resamples clusters when ``id=`` is declared, and rows otherwise.
    density_bins : int or None
        The bin count of a continuous node's diagnostic density.
    ratio : {"density", "classifier"}
        How a continuous node's diagnostic policy ratio is estimated.
    msm : None
        Refused: a working model over regimens is fitted by :class:`LTMLE` or :class:`SDR`.
    **refused : Any
        ``alpha``, ``max_iter`` and ``tol``, which raise :class:`TypeError` because the
        estimator fits no fluctuation, and each point-treatment keyword that :class:`LTMLE`
        refuses.  ``ratio_trim`` other than ``None`` and ``hazard_pooling`` other than
        ``"node"`` raise :class:`CapabilityError`.

    See Also
    --------
    SequentialIPW : The inverse probability weighted estimator of the same parameters.
    LTMLE : The targeted estimator of the same parameters.
    gcomp : The one-call form of this estimator.

    Examples
    --------
    >>> from cleverly.datasets import make_longitudinal
    >>> from cleverly.longitudinal import SequentialGComputation
    >>> frame, _ = make_longitudinal(n=300, seed=0)
    >>> result = SequentialGComputation({"always": 1, "never": 0}).fit(
    ...     frame,
    ...     outcome="Y",
    ...     treatment=["A1", "A2"],
    ...     baseline=["W1", "W2"],
    ...     time_varying=[[], ["L2"]],
    ...     censoring=["C1", "C2"],
    ... )
    >>> sorted(result.estimates)
    ['ate_regimen[never vs always]', 'ey_regimen[always]', 'ey_regimen[never]']
    >>> result.inference_status
    'influence_curve'
    """

    _estimator: ClassVar[str] = "gcomp"

    def __init__(
        self,
        regimens: Any,
        *,
        outcome_learner: Learner | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            regimens,
            outcome_learner=GLM() if outcome_learner is None else outcome_learner,
            **kwargs,
        )

    def _refuse_design(self, data: LongitudinalData, plans: tuple[Plan, ...]) -> None:
        """Refuse a binomial GLM on a continuous outcome with no declared scale.

        Parameters
        ----------
        data : LongitudinalData
            The prepared panel.
        plans : tuple of Plan
            The resolved plans.

        Raises
        ------
        CapabilityError
            When the outcome is continuous, ``q_bounds`` is ``None`` and a read GLM is not
            Gaussian.
        """
        if data.family == "binomial" or self.q_bounds is not None:
            return
        pseudo = self.outcome_learner if self.pseudo_learner is None else self.pseudo_learner
        if any(
            isinstance(learner, GLM) and learner.family != "gaussian"
            for learner in (self.outcome_learner, pseudo)
        ):
            raise CapabilityError(refusal=GCOMP_SCALE_REFUSAL)

    def _fit_mechanism(
        self,
        data: LongitudinalData,
        plans: tuple[Plan, ...],
        *,
        treatment_learner: Learner,
        censoring_learner: Learner,
        folds: Folds,
    ) -> Mechanism:
        """Fit the diagnostic mechanism, and degrade the diagnostics when it cannot fit.

        Parameters
        ----------
        data : LongitudinalData
            The prepared panel.
        plans : tuple of Plan
            The resolved plans.
        treatment_learner : Learner
            The treatment mechanism learner.
        censoring_learner : Learner
            The censoring mechanism learner.
        folds : Folds
            The single in-sample fold.

        Returns
        -------
        Mechanism
            The fitted mechanism, or one that holds ``nan`` everywhere.
        """
        return self._fit_diagnostic_mechanism(
            data,
            plans,
            treatment_learner=treatment_learner,
            censoring_learner=censoring_learner,
            folds=folds,
        )

    def _parametric(
        self,
        outcome_learner: Learner,
        pseudo_learner: Learner,
        mechanism: Mechanism,
        horizons: Sequence[int],
        plans: Sequence[Plan],
    ) -> bool:
        """Whether every outcome regression is a GLM.

        Parameters
        ----------
        outcome_learner : Learner
            The resolved outcome learner.
        pseudo_learner : Learner
            The resolved pseudo-outcome learner.
        mechanism : Mechanism
            The diagnostic mechanism, which the estimate does not read.
        horizons : sequence of int
            The reported horizons.
        plans : sequence of Plan
            The resolved plans.

        Returns
        -------
        bool
            ``True`` when the outcome learner, and the pseudo-outcome learner of a fit with a
            node before its last horizon, are GLMs.
        """
        if not isinstance(outcome_learner, GLM):
            return False
        return max(horizons) == 1 or isinstance(pseudo_learner, GLM)


class SequentialIPW(_Baseline):
    r"""Inverse probability weighting of a treatment regimen over time.

    The estimate reweights the units that follow a regimen by the cumulative ratio of the
    intervention density to the treatment and censoring mechanism, the clever covariate of
    :class:`LTMLE`.  ``normalization="hajek"`` divides the weighted outcome sum by the weight
    sum, as R ``ltmle``'s ``CalcIPTW`` does.  ``"horvitz_thompson"`` divides by ``n``.  A
    survival risk under Hajek is the product of the IPW discrete hazards, and a competing
    cause's incidence is their Aalen-Johansen sum.  See :mod:`cleverly.longitudinal.baselines`.

    The estimate is consistent only when every treatment and censoring model is correctly
    specified.  With every model a :class:`~cleverly.learners.GLM`, the interval is the
    stacked estimating equation of the mechanism fits and the estimate, which adds the
    projection term of each mechanism score (Lunceford and Davidian 2004, Section 3.1).  That
    interval is valid for the target only when every GLM is correctly specified.  With
    another learner the fit takes the ``"adaptive_baseline_plugin"`` status, and the
    diagnostic curve is the fixed-mechanism curve.

    Parameters
    ----------
    regimens : mapping, Regimen, DynamicRegimen, or sequence of regimens
        The regimens, as for :class:`LTMLE`.  A modified treatment policy is refused.
    normalization : {"hajek", "horvitz_thompson"}
        How the weighted outcome sum is normalized.
    reference : str or None
        The regimen every contrast is taken against, as for :class:`LTMLE`.
    horizons : sequence of int or None
        The survival horizons, as for :class:`LTMLE`.
    treatment_learner : object or None
        The treatment mechanism at each node.  The default is a
        :class:`~cleverly.learners.GLM`.
    censoring_learner : object or None
        The censoring mechanism at each node.  The default is a
        :class:`~cleverly.learners.GLM`.
    n_folds : int
        Must be ``1``.  The estimator fits in sample.
    g_bounds : float or tuple of float
        The bound pair applied to each cumulative mechanism probability.  The default is
        ``(0.01, 1.0)``.  A bound that binds changes the estimator, and
        :func:`~cleverly.longitudinal.estimator.longitudinal_truncation_curve` replays it.
    q_bounds : tuple of float or None
        The bounds of a continuous outcome, as for :class:`LTMLE`.
    alpha_sig : float
        Significance level for confidence intervals.
    simultaneous : bool
        Whether to report simultaneous confidence bands, as for :class:`LTMLE`.
    n_multiplier : int
        The number of multiplier draws behind each band.
    multiplier_kind : {"rademacher", "mammen", "normal"}
        The distribution of the multiplier draws.
    random_state : int or None
        The seed inherited by a default learner.
    run_id : str or None
        An identifier of your own, recorded on :attr:`LongitudinalResult.provenance`.
    n_jobs : int
        Worker count for the parallel fan-out.
    n_bootstrap : int
        Full-refit bootstrap replicates, ``0`` for none.  The draws report a bootstrap sd and
        no percentile inference.
    bootstrap_resampling : {"auto", "iid", "cluster"}
        ``"auto"`` resamples clusters when ``id=`` is declared, and rows otherwise.
    **refused : Any
        ``outcome_learner``, ``pseudo_learner``, ``density_bins``, ``ratio`` and ``msm``,
        which raise because the estimator fits no outcome regression and no density ratio;
        ``alpha``, ``max_iter`` and ``tol``, which raise :class:`TypeError`; and each
        point-treatment keyword that :class:`LTMLE` refuses.  ``ratio_trim`` other than
        ``None`` and ``hazard_pooling`` other than ``"node"`` raise
        :class:`CapabilityError`.

    See Also
    --------
    SequentialGComputation : The sequential regression estimator of the same parameters.
    LTMLE : The targeted estimator of the same parameters.
    ipw : The one-call form of this estimator.

    Examples
    --------
    >>> from cleverly.datasets import make_longitudinal
    >>> from cleverly.longitudinal import SequentialIPW
    >>> frame, _ = make_longitudinal(n=300, seed=0)
    >>> result = SequentialIPW({"always": 1, "never": 0}).fit(
    ...     frame,
    ...     outcome="Y",
    ...     treatment=["A1", "A2"],
    ...     baseline=["W1", "W2"],
    ...     time_varying=[[], ["L2"]],
    ...     censoring=["C1", "C2"],
    ... )
    >>> sorted(result.estimates)
    ['ate_regimen[never vs always]', 'ey_regimen[always]', 'ey_regimen[never]']
    >>> result.config.normalization
    'hajek'
    """

    _estimator: ClassVar[str] = "ipw"

    def __init__(
        self,
        regimens: Any,
        *,
        normalization: str = "hajek",
        **kwargs: Any,
    ) -> None:
        for name in ("outcome_learner", "pseudo_learner"):
            if kwargs.pop(name, None) is not None:
                raise CapabilityError(
                    IPW_OUTCOME_LEARNER_REFUSAL.format(name=name),
                    reason=RefusalReason.INVALID_DECLARATION,
                )
        if kwargs.get("density_bins") is not None:
            raise CapabilityError(
                IPW_DENSITY_REFUSAL.format(name="density_bins"),
                reason=RefusalReason.INVALID_DECLARATION,
            )
        if kwargs.get("ratio", "density") != "density":
            raise CapabilityError(
                IPW_DENSITY_REFUSAL.format(name="ratio"),
                reason=RefusalReason.INVALID_DECLARATION,
            )
        if normalization not in IPW_NORMALIZATIONS:
            raise ValueError(
                f"normalization must be one of {list(IPW_NORMALIZATIONS)}; got "
                f"{normalization!r}. 'hajek' divides the weighted outcome sum by the weight "
                "sum, and 'horvitz_thompson' divides it by n"
            )
        self.normalization = normalization
        # IPW fits no outcome regression.  The template is a GLM rather than ``None`` so the
        # replay recipe clones a learner with no seed to audit, as it does for the mechanism.
        super().__init__(regimens, outcome_learner=GLM(), **kwargs)

    def _normalization(self) -> str | None:
        """The declared normalization."""
        return self.normalization

    def _refuse_design(self, data: LongitudinalData, plans: tuple[Plan, ...]) -> None:
        """Refuse a modified treatment policy.

        Parameters
        ----------
        data : LongitudinalData
            The prepared panel.
        plans : tuple of Plan
            The resolved plans.

        Raises
        ------
        CapabilityError
            When a plan holds a modified treatment policy node.
        """
        if any(plan.has_mtp for plan in plans):
            raise CapabilityError(refusal=IPW_MTP_REFUSAL)

    def _fit_mechanism(
        self,
        data: LongitudinalData,
        plans: tuple[Plan, ...],
        *,
        treatment_learner: Learner,
        censoring_learner: Learner,
        folds: Folds,
    ) -> Mechanism:
        """Fit the mechanism and keep each node's model for the stacked curve.

        Parameters
        ----------
        data : LongitudinalData
            The prepared panel.
        plans : tuple of Plan
            The resolved plans.
        treatment_learner : Learner
            The treatment mechanism learner.
        censoring_learner : Learner
            The censoring mechanism learner.
        folds : Folds
            The single in-sample fold.

        Returns
        -------
        Mechanism
            The fitted mechanism with its models.
        """
        return fit_mechanism(
            data,
            plans,
            treatment_learner=treatment_learner,
            censoring_learner=censoring_learner,
            folds=folds,
            n_jobs=self.n_jobs,
            keep_models=True,
        )

    def _parametric(
        self,
        outcome_learner: Learner,
        pseudo_learner: Learner,
        mechanism: Mechanism,
        horizons: Sequence[int],
        plans: Sequence[Plan],
    ) -> bool:
        """Whether every fitted treatment and censoring model is a GLM.

        Parameters
        ----------
        outcome_learner : Learner
            Unused: IPW fits no outcome regression.
        pseudo_learner : Learner
            Unused.
        mechanism : Mechanism
            The fitted mechanism with its models.
        horizons : sequence of int
            The reported horizons.
        plans : sequence of Plan
            Unused.

        Returns
        -------
        bool
            ``True`` when every kept model is a GLM.
        """
        return mechanism_reads_parametric(mechanism)


def _unavailable_mechanism(data: LongitudinalData, plans: Sequence[Plan]) -> Mechanism:
    """A mechanism that holds ``nan`` at every factor, for a diagnostic fit that failed."""
    missing = np.full(data.n, np.nan)
    by_plan = {plan.label: missing for plan in plans}
    levels = [len(level) for level in data.treatment_levels]
    return Mechanism(
        treatment=tuple(dict(by_plan) for _ in range(data.n_times)),
        censoring=tuple(dict(by_plan) for _ in range(data.n_times)),
        treatment_observed=tuple(np.full((data.n, k), np.nan) for k in levels),
        censoring_observed=tuple(missing for _ in data.censoring_names),
        policy_numerators={
            plan.label: {
                time: missing for time in range(1, data.n_times + 1) if plan.is_mtp_node(time)
            }
            for plan in plans
            if plan.has_mtp
        },
    )


def _tilt_shorthand(regimens: Any) -> Any:
    """Read a sequence of :class:`~cleverly.interventions.Incremental` as one plan each.

    Each tilt broadcasts to every node, under its own ``name``.  Any other argument passes
    through, and :func:`~cleverly.longitudinal.regimen.resolve_regimens` reads it.
    """
    from ..interventions.incremental import Incremental

    if (
        isinstance(regimens, Sequence)
        and not isinstance(regimens, (str, bytes))
        and regimens
        and all(isinstance(entry, Incremental) for entry in regimens)
    ):
        names = [entry.name for entry in regimens]
        if len(set(names)) != len(names):
            raise DataError(
                f"two tilts share the name {sorted(n for n in names if names.count(n) > 1)[0]!r}; "
                "a name labels a parameter, so give each Incremental its own name="
            )
        return {entry.name: entry for entry in regimens}
    return regimens


class ParametricGFormula(_Baseline):
    r"""The parametric g-formula of a treatment regimen over time, by forward simulation.

    The estimator fits a conditional model for every time-varying covariate at nodes ``2..T``,
    for the outcome (or the discrete hazards of a survival outcome), and for the natural
    treatment where a plan reads it.  It then simulates each unit's history forward under each
    regimen and averages the simulated outcomes (Robins 1986; Taubman et al. 2009; McGrath
    et al. 2020).  The estimate is consistent only when every fitted law is correctly
    specified, and the draw order within a node defines non-nested models.  See
    :mod:`cleverly.longitudinal.gformula`.

    With every model a :class:`~cleverly.learners.GLM`, the interval stacks the estimating
    equations of the models and the estimate (Stefanski and Boos 2002, Section 2).  The
    simulation error is part of that interval, and ``result.simulation`` reports its share of
    the variance.  Any other learner gives the ``"adaptive_baseline_plugin"`` status: the point
    estimate stands, the interval is withheld, and a diagnostic curve stays under
    ``plugin_std_error``.

    Parameters
    ----------
    regimens : mapping, Regimen, DynamicRegimen, or sequence of regimens
        The regimens, as for :class:`LTMLE`.  Static, dynamic, known stochastic and modified
        treatment policies are accepted, and so is an odds tilt.  A tilt, and a modified
        treatment policy at a node ``t >= 2``, draw the natural treatment from a fitted model,
        or from the declared mechanism (``treatment_probabilities=``) of a node that has one.
        A fixed number at a continuous treatment node is a fixed dose, and a known rule
        (``rule_kind="known"``) there sets the dose from the simulated history.  Either reads
        the model of the dose and reports a support diagnostic.
    n_draws : int
        The simulated histories per unit behind the estimate.
    n_derivative_draws : int or None
        The simulated histories per unit behind the parameter derivatives, at least two.
        ``None`` takes the larger of ``n_draws`` and ten.
    covariate_types : mapping or None
        The type of each time-varying column at nodes ``2..T``, as ``"binary"``,
        ``"categorical"`` or ``"normal"``.  A tuple of one-hot columns is one categorical
        variable.  A column with values in ``{0, 1}`` is binary without a declaration, and
        any other undeclared column is refused.
    covariate_learner : object or None
        The template of every covariate model.  The default is a main-terms
        :class:`~cleverly.learners.GLM`.  A Gaussian covariate needs a GLM.
    outcome_learner : object or None
        The outcome model of an end-of-study outcome.  The default is a main-terms
        :class:`~cleverly.learners.GLM` of the outcome's family.
    hazard_learner : object or None
        The template of every hazard model of a survival outcome.  The default is a GLM.
    pseudo_learner : object or None
        Refused when given: the estimator fits no sequential pseudo-outcome regression.
    **kwargs : Any
        The settings of :class:`SequentialGComputation` that apply: ``reference``,
        ``horizons``, ``treatment_learner`` (the natural-treatment models and the diagnostic
        mechanism), ``censoring_learner``, ``g_bounds``, ``q_bounds``, ``alpha_sig``,
        ``simultaneous``, ``n_multiplier``, ``multiplier_kind``, ``random_state``,
        ``run_id``, ``n_jobs``, ``n_bootstrap``, ``bootstrap_resampling``, ``density_bins``
        and ``ratio``.  ``n_folds`` must be ``1``.  ``msm``, ``alpha``, ``max_iter`` and
        ``tol`` are refused.

    See Also
    --------
    SequentialGComputation : The iterated conditional expectation form of the same parameter.
    LTMLE : The targeted estimator of the same parameter.
    gformula : The one-call form of this estimator.

    Notes
    -----
    A repeated outcome is refused: roadmap row X71 tracks it.  The simulation draws come from a
    counter-based Philox generator keyed by ``random_state``, so a fit repeats exactly.

    A declared treatment mechanism at a tilt node, or at a modified-treatment-policy node
    ``t >= 2``, replaces the natural-treatment model.  The simulation reads the declared
    probabilities of each unit as that unit's natural law at every simulated history, so the
    draw is the declared law only when the declaration does not depend on a covariate that the
    simulation redraws.  That node has no learner and no score term.  A unit with no declared
    value at such a node is refused.

    A fit with a fixed dose or a rule-set dose reports no mechanism diagnostics for any regimen
    of the fit, because a dose has no density ratio.  Fit the other regimens in a separate call to
    read their diagnostics.

    Examples
    --------
    >>> from cleverly.datasets import make_longitudinal
    >>> from cleverly.longitudinal import ParametricGFormula
    >>> frame, _ = make_longitudinal(n=300, seed=0)
    >>> result = ParametricGFormula(
    ...     {"always": 1, "never": 0}, covariate_types={"L2": "normal"}, random_state=0
    ... ).fit(
    ...     frame,
    ...     outcome="Y",
    ...     treatment=["A1", "A2"],
    ...     baseline=["W1", "W2"],
    ...     time_varying=[[], ["L2"]],
    ...     censoring=["C1", "C2"],
    ... )
    >>> sorted(result.estimates)
    ['ate_regimen[never vs always]', 'ey_regimen[always]', 'ey_regimen[never]']
    >>> result.inference_status
    'influence_curve'
    """

    _estimator: ClassVar[str] = "parametric_gformula"
    _allow_set_dose: ClassVar[bool] = True
    #: The simulation draws the natural dose from its fitted law and reads no ratio.
    _reads_density_ratio: ClassVar[bool] = False

    def __init__(
        self,
        regimens: Any,
        *,
        n_draws: int = 10,
        n_derivative_draws: int | None = None,
        covariate_types: Mapping[Any, str] | None = None,
        covariate_learner: Learner | None = None,
        outcome_learner: Learner | None = None,
        hazard_learner: Learner | None = None,
        pseudo_learner: Learner | None = None,
        **kwargs: Any,
    ) -> None:
        if pseudo_learner is not None:
            raise CapabilityError(refusal=GFORMULA_PSEUDO_REFUSAL)
        _validate_learner(covariate_learner, "covariate_learner")
        _validate_learner(hazard_learner, "hazard_learner")
        if isinstance(n_draws, bool) or not isinstance(n_draws, (int, np.integer)) or n_draws < 1:
            raise ValueError(f"n_draws must be a positive integer; got {n_draws!r}")
        if n_derivative_draws is not None and (
            isinstance(n_derivative_draws, bool)
            or not isinstance(n_derivative_draws, (int, np.integer))
            or n_derivative_draws < 2
        ):
            raise ValueError(
                "n_derivative_draws must be an integer of at least 2, or None; got "
                f"{n_derivative_draws!r}. The leave-one-copy-out baseline of the score-function "
                "derivative needs a second copy"
            )
        self.n_draws = int(n_draws)
        self.n_derivative_draws = (
            max(self.n_draws, 10) if n_derivative_draws is None else int(n_derivative_draws)
        )
        self.covariate_types = covariate_types
        self.covariate_learner = covariate_learner
        self.hazard_learner = hazard_learner
        #: The outcome learner as the caller declared it: ``None`` is a GLM of the data's family.
        self._declared_outcome = outcome_learner
        #: A test hook, not API, that replaces the simulation with exact terms.
        self._exact: Callable[..., Any] | None = None
        super().__init__(
            _tilt_shorthand(regimens),
            outcome_learner=GLM() if outcome_learner is None else outcome_learner,
            **kwargs,
        )

    def _resolve_tilts(self, data: LongitudinalData, plans: tuple[Plan, ...]) -> tuple[Plan, ...]:
        """Keep an odds tilt as it is: the simulation draws the tilted treatment itself.

        Parameters
        ----------
        data : LongitudinalData
            The prepared panel.
        plans : tuple of Plan
            The resolved plans.

        Returns
        -------
        tuple of Plan
            The plans, unchanged.
        """
        return plans

    def _refuse_design(self, data: LongitudinalData, plans: tuple[Plan, ...]) -> None:
        """Refuse a setting or a covariate that the simulation cannot fit, before any learner.

        Parameters
        ----------
        data : LongitudinalData
            The prepared panel.
        plans : tuple of Plan
            The resolved plans.

        Raises
        ------
        CapabilityError
            When the data declare a Markov order, a plan holds a risk-ratio tilt or an odds
            tilt at a node with other than two levels, a covariate is untyped, a learner is declared
            that nothing reads, a learner carries a family the fit overrides, a Gaussian
            covariate has a learner that is not a GLM, a binomial GLM meets a continuous
            outcome with no declared scale, or a declared treatment mechanism is missing for a
            unit at a node whose natural treatment the simulation draws.
        """
        if data.markov_order is not None:
            raise CapabilityError(refusal=GFORMULA_MARKOV_REFUSAL)
        declared_natural(data, plans)
        for plan in plans:
            for node in range(1, data.n_times + 1):
                kind, spec = plan_node(plan.regimen, node)
                if kind == "mtp" and isinstance(spec, RiskRatioTilt):
                    raise CapabilityError(refusal=GFORMULA_RR_TILT_REFUSAL)
                if (
                    kind == "tilt"
                    and not data.is_held_node(node)
                    and len(data.treatment_levels[node - 1]) != 2
                ):
                    raise CapabilityError(refusal=GFORMULA_CATEGORICAL_TILT_REFUSAL)
        variables = resolve_covariates(data, self.covariate_types)
        for argument, learner in (
            ("covariate_learner", self.covariate_learner),
            ("hazard_learner", self.hazard_learner),
            ("treatment_learner", self.treatment_learner),
        ):
            if isinstance(learner, GLM) and learner.family != "auto":
                raise CapabilityError(
                    f"{argument}=GLM(family={learner.family!r}) names a family, and the fit "
                    "chooses the family of each model from its response: binomial for a "
                    "binary one, multinomial for a categorical one, Gaussian for a normal one.",
                    reason=RefusalReason.INVALID_DECLARATION,
                    alternative=f"Pass {argument}=GLM() with the default family.",
                )
        if data.is_survival and self._declared_outcome is not None:
            raise CapabilityError(
                "outcome_learner= has nothing to fit on a survival outcome: the outcome law is "
                "the discrete hazard at each node.",
                reason=RefusalReason.INVALID_DECLARATION,
                alternative="Pass hazard_learner= instead.",
            )
        if not data.is_survival and self.hazard_learner is not None:
            raise CapabilityError(
                "hazard_learner= has nothing to fit on an end-of-study outcome.",
                reason=RefusalReason.INVALID_DECLARATION,
                alternative="Pass outcome_learner= instead.",
            )
        normal = [
            variable
            for node in variables
            for variable in node
            if variable.modelled and variable.kind == "normal"
        ]
        if normal and not (
            self.covariate_learner is None or isinstance(self.covariate_learner, GLM)
        ):
            raise normal_learner_refusal("covariate_learner", normal[0].name)
        if (
            data.family != "binomial"
            and self.q_bounds is None
            and isinstance(self._declared_outcome, GLM)
            and self._declared_outcome.family != "gaussian"
        ):
            raise CapabilityError(refusal=GCOMP_SCALE_REFUSAL)

    def _refuse_undeclared_policy_scale(
        self, data: LongitudinalData, plans: Sequence[Plan]
    ) -> None:
        """Keep a continuous outcome under a modified treatment policy without ``q_bounds``.

        R-f1 of X40 protects the [0, 1] outcome map of the sequential regressions.  The
        simulation fits the outcome law on its own scale and maps nothing, so the refusal
        does not apply here; :meth:`_refuse_design` keeps the scale refusal that does.

        Parameters
        ----------
        data : LongitudinalData
            The prepared panel.
        plans : sequence of Plan
            The resolved plans.
        """

    def _fit_mechanism(
        self,
        data: LongitudinalData,
        plans: tuple[Plan, ...],
        *,
        treatment_learner: Learner,
        censoring_learner: Learner,
        folds: Folds,
    ) -> Mechanism:
        """Fit the diagnostic mechanism, and degrade the diagnostics when it cannot fit.

        A fixed dose or a rule-set dose has no density ratio, so a fit with one reports no
        mechanism diagnostics and reads the support diagnostic of ``result.simulation``
        instead.

        Parameters
        ----------
        data : LongitudinalData
            The prepared panel.
        plans : tuple of Plan
            The resolved plans.
        treatment_learner : Learner
            The treatment mechanism learner.
        censoring_learner : Learner
            The censoring mechanism learner.
        folds : Folds
            The single in-sample fold.

        Returns
        -------
        Mechanism
            The fitted mechanism, or one that holds ``nan`` everywhere.
        """
        if has_set_dose(data, plans):
            return _unavailable_mechanism(data, plans)
        return self._fit_diagnostic_mechanism(
            data,
            plans,
            treatment_learner=treatment_learner,
            censoring_learner=censoring_learner,
            folds=folds,
        )

    def _parametric(
        self,
        outcome_learner: Learner,
        pseudo_learner: Learner,
        mechanism: Mechanism,
        horizons: Sequence[int],
        plans: Sequence[Plan],
    ) -> bool:
        """The placeholder: the fitted models decide, in :meth:`_simulated_parametric`.

        Parameters
        ----------
        outcome_learner : Learner
            The resolved outcome learner.
        pseudo_learner : Learner
            The resolved pseudo-outcome learner.
        mechanism : Mechanism
            The diagnostic mechanism.
        horizons : sequence of int
            The reported horizons.
        plans : sequence of Plan
            The resolved plans.

        Returns
        -------
        bool
            ``True`` here.
        """
        return True

    def _simulated_parametric(self, parametric: bool, fits: Mapping[str, RegimenFit]) -> bool:
        """Whether every model the estimate read is a GLM, from the fit's own report.

        Parameters
        ----------
        parametric : bool
            The value before the fit.
        fits : mapping of str to RegimenFit
            The fits of the recursion.

        Returns
        -------
        bool
            The report's flag.
        """
        for fit in fits.values():
            if fit.simulation is not None and fit.simulation.report is not None:
                return bool(fit.simulation.report.parametric)
        return parametric

    def _preflight_followers(
        self,
        data: LongitudinalData,
        plans: tuple[Plan, ...],
        horizons: Sequence[int],
        folds: Folds,
    ) -> None:
        """Skip the follower check: a regimen with no follower is extrapolation by the model.

        The aliased-design refusal of the simulation covers the cases the model cannot
        extrapolate.

        Parameters
        ----------
        data : LongitudinalData
            The prepared panel.
        plans : tuple of Plan
            The resolved plans.
        horizons : sequence of int
            The reported horizons.
        folds : Folds
            The realized outer split.

        Raises
        ------
        CapabilityError
            When a treatment node is a vector with a dose component.
        """
        for time in range(1, data.n_times + 1):
            if data.node_kind(time) == "mixed":
                raise CapabilityError.from_refusal(GFORMULA_MIXED_NODE_REFUSAL)

    def _gformula_settings(self, data: LongitudinalData) -> GFormulaSettings | None:
        """The simulation settings: draws, learners and a seed drawn from ``random_state``.

        Parameters
        ----------
        data : LongitudinalData
            The prepared panel, which the settings do not read.

        Returns
        -------
        GFormulaSettings
            The settings the simulation reads.
        """
        # A sequence of its own, so the seed shares nothing with the bootstrap's spawned children
        # or with any other consumer of ``random_state``.
        entropy = np.random.SeedSequence(self.random_state).entropy
        assert isinstance(entropy, int)
        child = np.random.SeedSequence([entropy, _SIMULATION_SEED_TAG])
        seed = int(child.generate_state(1, dtype=np.uint64)[0]) >> 2
        return GFormulaSettings(
            n_draws=self.n_draws,
            n_derivative_draws=self.n_derivative_draws,
            covariate_types=self.covariate_types,
            covariate_learner=self.covariate_learner,
            outcome_learner=self._declared_outcome,
            hazard_learner=self.hazard_learner,
            treatment_learner=self.treatment_learner,
            seed=seed,
            exact=self._exact,
            reference=self.reference,
        )

    def _finish(
        self,
        result: LongitudinalResult,
        reported: _Reported,
        fits: Mapping[str, RegimenFit],
    ) -> LongitudinalResult:
        """Attach the simulation report, with the Monte Carlo share of every parameter.

        Parameters
        ----------
        result : LongitudinalResult
            The result before any bootstrap.
        reported : _Reported
            The names, estimates and contributing fits.
        fits : mapping of str to RegimenFit
            The fits of the recursion.

        Returns
        -------
        LongitudinalResult
            The result, with ``simulation`` set.

        Warns
        -----
        MonteCarloWarning
            When the simulation is more than 10% of a parameter's variance.
        """
        shares: dict[str, float] = {}
        for name, contributors in reported.contributors.items():
            curve = contributors[0].influence_curve_scaled
            if len(contributors) == 2:
                curve = curve - contributors[1].influence_curve_scaled
            shares[name] = monte_carlo_share(result.data, contributors, curve)
        first = next(iter(fits.values()))
        assert first.simulation is not None
        report = replace(first.simulation.report, monte_carlo_share=shares)
        over = {
            name: share
            for name, share in shares.items()
            if np.isfinite(share) and share > MONTE_CARLO_WARN_SHARE
        }
        if over:
            worst = max(over, key=lambda name: over[name])
            _warn(
                f"the simulation is {over[worst]:.0%} of the variance of {worst!r}"
                + (f" and of {len(over) - 1} other parameter(s)" if len(over) > 1 else "")
                + f", above {MONTE_CARLO_WARN_SHARE:.0%}. The interval includes it. "
                f"Raise n_draws= (now {self.n_draws}) to lower it.",
                MonteCarloWarning,
                stacklevel=4,
            )
        return replace(result, simulation=report)


@dataclass(frozen=True)
class _BoundReplay:
    """One bound's complete replay of the bound-dependent recursion.

    Parameters
    ----------
    estimates : dict of str to ParameterEstimate
        The replayed estimates, in the fit's own report order.
    fits : dict of str to RegimenFit
        The replayed fits, keyed by :func:`_fit_key`.
    msm_fits : tuple of MSMRegimenFit
        The replayed projections, empty without a working model.
    contributors : dict of str to tuple of RegimenFit
        The replayed fits behind each reported name, composed where the name was.
    """

    estimates: dict[str, ParameterEstimate]
    fits: dict[str, RegimenFit]
    msm_fits: tuple[MSMRegimenFit, ...]
    contributors: dict[str, tuple[RegimenFit, ...]]


def _mechanism_declared(data: LongitudinalData) -> bool:
    """Whether every treatment and censoring factor of ``data`` is declared known."""
    known = data.known_mechanisms
    if known is None:
        return False
    nodes = range(1, data.n_times + 1)
    if not all(known.treatment_at(time) is not None for time in nodes):
        return False
    return not data.censoring_names or all(known.censoring_at(time) is not None for time in nodes)


def _refit_bound(
    result: LongitudinalResult,
    draw: LongitudinalRepeat,
    bounds: tuple[float, float],
) -> _BoundReplay:
    """Repeat every bound-dependent regression and targeting update of one draw.

    A hazard fit re-targets its retained initial hazards at ``bounds`` and refits no
    learner, because the bounds enter the targeting and not the hazard regressions.

    Parameters
    ----------
    result : LongitudinalResult
        The fit being replayed. Its prepared data and scaler are reused.
    draw : LongitudinalRepeat
        The draw being replayed. Its mechanism, folds, fold mechanisms and replay recipe
        are reused.
    bounds : tuple of float
        The cumulative bound pair this replay runs under.

    Returns
    -------
    object
        The replayed estimates in the fit's own report order, the per-cell fits, the
        projection fits, and the fits behind each reported name.

    Raises
    ------
    CapabilityError
        When the retained learner templates could not be cloned.
    RuntimeError
        When the replay did not report a parameter the fit reports.
    """

    refuse_regimen_rules(result.config.regimens)
    refuse_evaluated_msm_functions(result.msm)
    recipe = draw.replay_recipe
    fits: dict[str, RegimenFit]
    msm_fits: tuple[MSMRegimenFit, ...]
    if recipe.hazard is not None and draw.initial_hazards is not None:
        fits = _run_hazard(
            result.data,
            recipe.plans,
            recipe.hazard.levels,
            draw.mechanism,
            draw.initial_hazards,
            horizons=result.config.horizons,
            g_bounds=bounds,
            alpha=recipe.hazard.alpha,
            algorithm=recipe.hazard.algorithm,
            max_iter=recipe.hazard.max_iter,
            tol=recipe.hazard.tol,
            step_size=recipe.hazard.step_size,
            node_bounds=recipe.node_bounds,
            stopping=recipe.stopping,
            step_bound=recipe.step_bound,
            step_search=recipe.step_search,
            iterate=recipe.iterate,
        )
        msm_fits = ()
    else:
        if recipe.outcome_learner is None or recipe.pseudo_learner is None:
            _refuse_replay(
                LONGITUDINAL_REPLAY_LEARNER_UNCLONABLE, reason=RefusalReason.INVALID_DECLARATION
            )
        fits, msm_fits = _run_recursion(
            result.data,
            recipe.plans,
            draw.mechanism,
            msm=result.msm,
            horizons=result.config.horizons,
            outcome_learner=clone(recipe.outcome_learner),
            pseudo_learner=clone(recipe.pseudo_learner),
            folds=draw.folds,
            scaler=result.scaler,
            g_bounds=bounds,
            alpha=recipe.alpha,
            max_iter=recipe.max_iter,
            tol=recipe.tol,
            n_jobs=recipe.n_jobs,
            estimator=recipe.estimator,
            fold_mechanisms=draw.fold_mechanisms,
            stopping=recipe.stopping,
            step_bound=recipe.step_bound,
            step_search=recipe.step_search,
            iterate=recipe.iterate,
            normalization=recipe.normalization,
            parametric=recipe.parametric,
            node_bounds=recipe.node_bounds,
        )
    # The fit reports the projected levels, so the replay projects them too.
    if recipe.isotonic:
        fits, _ = _project_fits(fits)
    # The status the fit stamped, recomputed from the same data and folds, so the replay at
    # the fitted bound equals the fit in every field ``_fitted_replay_matches`` compares.
    status = _trimmed_status(
        _inference_status(
            result.data, draw.folds, estimator=recipe.estimator, parametric=recipe.parametric
        ),
        draw.mechanism,
    )
    if result.msm is None:
        reference = next(
            plan.regimen for plan in recipe.plans if plan.label == result.config.reference
        )
        reported = _estimates(
            result.data,
            fits,
            result.scaler,
            reference,
            alpha_sig=result.config.alpha_sig,
            inference=status,
        )
    else:
        reported = _msm_estimates(
            result.data, msm_fits, alpha_sig=result.config.alpha_sig, inference=status
        )
    missing = [name for name in result.estimates if name not in reported.estimates]
    if missing:
        raise RuntimeError(
            f"longitudinal replay at bounds {bounds} omitted reported parameters {missing}"
        )
    return _BoundReplay(
        estimates={
            name: replace(reported.estimates[name], support=result.estimates[name].support)
            for name in result.estimates
        },
        fits=fits,
        msm_fits=msm_fits,
        contributors={name: reported.contributors[name] for name in result.estimates},
    )


def _exact_replay_equal(left: Any, right: Any) -> bool:
    """Recursively compare retained numerical fit artifacts without a tolerance."""

    if type(left) is not type(right):
        return False
    if isinstance(left, np.ndarray):
        return bool(np.array_equal(left, right, equal_nan=True))
    if is_dataclass(left) and not isinstance(left, type):
        return all(
            _exact_replay_equal(getattr(left, item.name), getattr(right, item.name))
            for item in fields(left)
        )
    if isinstance(left, Mapping):
        return tuple(left) == tuple(right) and all(
            _exact_replay_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(
            _exact_replay_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, float) and np.isnan(left):
        return bool(np.isnan(right))
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return left is right


def _fitted_replay_matches(
    result: LongitudinalResult, draw: LongitudinalRepeat, replay: _BoundReplay
) -> bool:
    """Whether replay reproduced every retained bound-dependent fitted artifact of a draw.

    Parameters
    ----------
    result : LongitudinalResult
        The fit being replayed.
    draw : LongitudinalRepeat
        The draw the replay ran.
    replay : object
        The replay run at the fit's own bounds.

    Returns
    -------
    bool
        Whether the estimates, the per-cell fits and the projections all match exactly.
    """

    # A replay never reruns the bootstrap, so the summaries are stripped on both sides.
    # ``_exact_replay_equal`` recurses into every dataclass field and would otherwise
    # refuse every bootstrapped fit.  The draw's own estimates of the reported names: on a
    # one-draw fit they are the reported estimates.
    reported = result.estimates if result.n_repeats == 1 else draw.estimates
    fitted = {name: replace(reported[name], bootstrap=None) for name in result.estimates}
    replayed = {name: replace(e, bootstrap=None) for name, e in replay.estimates.items()}
    return (
        _exact_replay_equal(fitted, replayed)
        and _exact_replay_equal(draw.fits, replay.fits)
        and _exact_replay_equal(draw.msm_fits, replay.msm_fits)
    )


def _consumed_prefixes(
    data: LongitudinalData,
    mechanism: Mechanism,
    plans: Sequence[Plan],
    bounds: tuple[float, float],
) -> dict[str, BoolArray]:
    """Per regimen, which cumulative cells a bound replaced in the mechanism the fit read.

    A recursion divides by one mechanism at any fold count, per regimen or over a working
    model's cells: the out-of-fold pair from
    :meth:`~cleverly.longitudinal.sequential.Mechanism.cumulative_with_unbounded`.  A
    single-fold fit reads it in its one pass.  A cross-fitted fit runs untargeted fold
    recursions, which read no mechanism, and then one pooled fluctuation per node, which
    reads this pair.

    Parameters
    ----------
    data : LongitudinalData
        The prepared data the fit ran on.
    mechanism : Mechanism
        The fitted mechanism.
    plans : sequence of Plan
        The resolved regimens the replay ran.
    bounds : tuple of float
        The cumulative bound pair this replay ran under.

    Returns
    -------
    dict
        Regimen label to one ``(n, T)`` truncation mask.
    """
    return {
        plan.label: _truncated_cells(*mechanism.cumulative_with_unbounded(data, plan, bounds))
        for plan in plans
    }


def _score_cell_truncation_counts(
    fits: Sequence[RegimenFit],
    consumed: Mapping[str, BoolArray],
) -> tuple[int, int]:
    """Count the truncated mechanism cells the reported recursion consumed.

    A *cell* is one ``(node, row)`` cumulative probability the targeting read, and a node's
    cells are its ``trained_on`` rows: the rows whose clever covariate is nonzero, and so
    the rows that node's score is summed over.  That holds at every fold count, because a
    cross-fitted fit solves one pooled fluctuation per node over every follower with the
    out-of-fold mechanism.  The published count therefore has one denominator for a
    single-fold fit and a cross-fitted one.

    Parameters
    ----------
    fits : sequence of RegimenFit
        The fits behind one reported name.
    consumed : mapping
        Regimen label to its truncation mask, from :func:`_consumed_prefixes`.

    Returns
    -------
    tuple of int
        The truncated and the evaluated cell counts.
    """

    truncated = 0
    evaluated = 0
    for fit in fits:
        # A hazard fit divides by the mechanism at each unit's observed level, which its
        # own cumulative arrays hold, rather than by the regimen's.
        mask = (
            consumed[fit.regimen.label]
            if fit.hazard is None
            else _truncated_cells(fit.cumulative_unbounded, fit.cumulative)
        )
        for step in fit.scored_nodes:
            clipped = mask[:, step.time - 1][step.trained_on]
            truncated += int(np.count_nonzero(clipped))
            evaluated += int(clipped.size)
    return truncated, evaluated


def longitudinal_truncation_curve(
    result: LongitudinalResult,
    bounds: Sequence[CumulativeGBounds],
    *,
    estimands: Sequence[str] | None = None,
) -> Any:
    """Refit the complete longitudinal recursion at explicit cumulative bounds.

    Parameters
    ----------
    result : LongitudinalResult
        The fit to replay.
    bounds : sequence
        The cumulative bound grid. A scalar ``b`` means ``(b, 1)``, and a pair is kept
        exactly. Repeated entries are replayed once and reported once.
    estimands : sequence of str or None
        Reported estimands to include. All of them are the default.

    Returns
    -------
    dataframe
        One row per distinct bound pair per requested estimand.

    Raises
    ------
    CapabilityError
        When a regimen of the result has a rule node that is not declared
        ``rule_kind="known"``, when the grid is empty or holds a bound this package cannot
        resolve, when it names an estimand this fit does not report, when a pair truncates
        the declared factors' cumulative product that the fit refuses to truncate, or when
        the stored artifacts cannot replay the recursion.
    DataError
        When a regimen of the result carries a ``rule_kind`` outside the three states.

    Notes
    -----
    A replay does not rerun the full-refit bootstrap.  The curve reports point estimates
    only, and the replay comparison at the fitted bound ignores the bootstrap summaries.

    It runs :func:`~cleverly.longitudinal.regimen.refuse_regimen_rules` on
    ``result.config.regimens`` first.  A regimen changed with ``object.__setattr__`` can
    carry a declaration this version refuses, and the replay would report a
    recomputation for it.
    """
    refuse_regimen_rules(result.config.regimens)
    refuse_evaluated_msm_functions(result.msm)
    if result.config.estimator == "gcomp":
        raise CapabilityError(refusal=GCOMP_TRUNCATION_REFUSAL)
    if result.config.estimator == "parametric_gformula":
        raise CapabilityError(refusal=GFORMULA_TRUNCATION_REFUSAL)
    omissions = tuple(
        dict.fromkeys(item for draw in result.repeats for item in draw.replay_recipe.omissions)
    )
    if omissions:
        _refuse_replay(*omissions, reason=RefusalReason.INVALID_DECLARATION)
    # Resolved first and deduplicated after, so that ``0.2`` and ``(0.2, 1.0)`` are the one
    # bound pair they denote rather than two replays of it.  A replay is a complete backward
    # recursion per regimen, cause and horizon, so a duplicate entry is minutes of work for
    # a row the frame already carries.
    #
    # Every shape of a bad grid is a caller-input refusal rather than a ``ValueError``,
    # because this function is reached through ``run_all(arguments=...)``, which catches a
    # capability refusal per row and lets every other operation report.  A bare exception
    # there aborts the whole battery over one grid typo.  The resolver reports a bad pair, a
    # bad scalar and the absent ``"auto"`` case with a bare ``ValueError``, and a bare
    # ``0.2`` written for ``[0.2]`` raises ``TypeError`` from the iteration itself.  The
    # ``try`` covers the resolution alone, so a defect in the replay below still surfaces as
    # the defect it is.
    try:
        grid = tuple(dict.fromkeys(resolve_cumulative_g_bounds(bound) for bound in bounds))
    except (TypeError, ValueError) as error:
        # The resolver's own sentence is reused, because it names the offending value and the
        # condition it failed.
        raise CapabilityError.from_refusal(
            _caught_refusal(
                "a longitudinal truncation curve cannot resolve the requested cumulative "
                "bound grid, which is a sequence of scalars or of (lower, upper) pairs: ",
                error,
            )
        ) from error
    if not grid:
        raise CapabilityError(
            "a longitudinal truncation curve needs at least one cumulative bound pair; "
            "the requested grid is empty.",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    # A pair the fit refuses on its declared factors is refused here too, before any replay.
    # Such a pair truncates a known product, so its row would report a movement that no
    # admitted fit can show and that reflects no variance trade.
    for pair in grid:
        refusal = known_product_refusal(result.data, result.repeats[0].replay_recipe.plans, pair)
        if refusal is not None:
            raise CapabilityError.from_refusal(
                replace(
                    refusal,
                    condition=(
                        "a longitudinal truncation curve replays only the bound pairs that a "
                        "fit admits, and a fit refuses this one: " + refusal.condition
                    ),
                    alternative=(
                        "Remove the pair from the bounds grid. A pair (lower, 1) with lower "
                        "below the smallest declared cumulative probability is admitted."
                    ),
                )
            )
    try:
        reported = select_estimates(result.estimates, estimands)
    except (KeyError, ValueError) as error:
        # Same reasoning as the empty grid: an estimand this fit does not report is a
        # caller-input refusal, not a defect in the battery that asked for it.  The
        # selector's own sentence is reused, and ``args[0]`` rather than ``str`` because a
        # ``KeyError`` renders its message in quotes.
        raise CapabilityError.from_refusal(
            _caught_refusal(
                "a longitudinal truncation curve cannot report the requested estimands: ",
                error,
            )
        ) from error
    fitted_bounds = result.config.g_bounds
    # A point estimate can agree while a stochastic Q recursion differs underneath it.
    # Prove the retained recipe reconstructs the fit before reporting *any* requested
    # bound, including grids that deliberately omit the fitted pair.  On a repeated fit
    # every draw is replayed and checked, and each bound reports the median of the draws.
    fitted_replays = []
    for draw in result.repeats:
        replay = _refit_bound(result, draw, fitted_bounds)
        if not _fitted_replay_matches(result, draw, replay):
            _refuse_replay(
                LONGITUDINAL_REPLAY_FITTED_BOUND_MISMATCH,
                reason=RefusalReason.INVALID_DECLARATION,
            )
        fitted_replays.append(replay)
    rows: list[dict[str, Any]] = []
    for lower, upper in grid:
        replays = (
            fitted_replays
            if (lower, upper) == fitted_bounds
            else [_refit_bound(result, draw, (lower, upper)) for draw in result.repeats]
        )
        if len(replays) == 1:
            estimates = replays[0].estimates
        else:
            estimates = median_estimates(
                with_precedent_status([replay.estimates for replay in replays])
            )
        counts: dict[str, tuple[int, int]] = dict.fromkeys(reported, (0, 0))
        for draw, replay in zip(result.repeats, replays, strict=True):
            # Once per bound and draw rather than once per estimand: the masks are a
            # property of the draw's mechanism and the bound, and every estimand at this
            # bound reads the same ones.
            consumed = _consumed_prefixes(
                result.data, draw.mechanism, draw.replay_recipe.plans, (lower, upper)
            )
            for name in reported:
                truncated, evaluated = _score_cell_truncation_counts(
                    replay.contributors[name], consumed
                )
                previous = counts[name]
                counts[name] = (previous[0] + truncated, previous[1] + evaluated)
        for name in reported:
            truncated, evaluated = counts[name]
            fitted = float(result.estimates[name].psi)
            estimate = float(estimates[name].psi)
            rows.append(
                {
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "estimand": name,
                    "psi": estimate,
                    "is_fitted_bound": (lower, upper) == fitted_bounds,
                    "fitted_lower_bound": fitted_bounds[0],
                    "fitted_upper_bound": fitted_bounds[1],
                    "fitted_psi": fitted,
                    "delta_from_fitted": estimate - fitted,
                    "truncated_score_cells": truncated,
                    "evaluated_score_cells": evaluated,
                }
            )
    return result.data.frame_like({name: [row[name] for row in rows] for name in rows[0]})


def ltmle(
    data: Any,
    *,
    regimens: Any,
    outcome: str | Sequence[str] | Mapping[str, Sequence[str]] | Repeated,
    treatment: Sequence[str],
    baseline: Sequence[str],
    time_varying: Sequence[Sequence[str]] | None = None,
    censoring: Sequence[str] | None = None,
    id: str | None = None,
    weights: str | None = None,
    weights_type: str = "probability",
    weights_estimated: bool = False,
    family: str = "auto",
    **kwargs: Any,
) -> LongitudinalResult:
    """One-call longitudinal TMLE, mirroring :func:`cleverly.tmle`."""
    return LTMLE(regimens, **kwargs).fit(
        data,
        outcome=outcome,
        treatment=treatment,
        baseline=baseline,
        time_varying=time_varying,
        censoring=censoring,
        id=id,
        weights=weights,
        weights_type=weights_type,
        weights_estimated=weights_estimated,
        family=family,
    )


def sdr(
    data: Any,
    *,
    regimens: Any,
    outcome: str | Sequence[str] | Mapping[str, Sequence[str]],
    treatment: Sequence[str],
    baseline: Sequence[str],
    time_varying: Sequence[Sequence[str]] | None = None,
    censoring: Sequence[str] | None = None,
    id: str | None = None,
    weights: str | None = None,
    weights_type: str = "probability",
    weights_estimated: bool = False,
    family: str = "auto",
    **kwargs: Any,
) -> LongitudinalResult:
    """One-call sequentially doubly robust estimation, mirroring :func:`ltmle`."""
    return SDR(regimens, **kwargs).fit(
        data,
        outcome=outcome,
        treatment=treatment,
        baseline=baseline,
        time_varying=time_varying,
        censoring=censoring,
        id=id,
        weights=weights,
        weights_type=weights_type,
        weights_estimated=weights_estimated,
        family=family,
    )


def ipsi(
    data: Any,
    *,
    regimens: Any,
    outcome: str | Sequence[str] | Mapping[str, Sequence[str]],
    treatment: Sequence[str],
    baseline: Sequence[str],
    time_varying: Sequence[Sequence[str]] | None = None,
    censoring: Sequence[str] | None = None,
    id: str | None = None,
    weights: str | None = None,
    weights_type: str = "probability",
    weights_estimated: bool = False,
    family: str = "auto",
    **kwargs: Any,
) -> LongitudinalResult:
    """One-call incremental propensity-score estimation over time, mirroring :func:`ltmle`."""
    return IPSI(regimens, **kwargs).fit(
        data,
        outcome=outcome,
        treatment=treatment,
        baseline=baseline,
        time_varying=time_varying,
        censoring=censoring,
        id=id,
        weights=weights,
        weights_type=weights_type,
        weights_estimated=weights_estimated,
        family=family,
    )


def gcomp(
    data: Any,
    *,
    regimens: Any,
    outcome: str | Sequence[str] | Mapping[str, Sequence[str]],
    treatment: Sequence[str],
    baseline: Sequence[str],
    time_varying: Sequence[Sequence[str]] | None = None,
    censoring: Sequence[str] | None = None,
    id: str | None = None,
    weights: str | None = None,
    weights_type: str = "probability",
    weights_estimated: bool = False,
    family: str = "auto",
    **kwargs: Any,
) -> LongitudinalResult:
    """One-call sequential g-computation, mirroring :func:`ltmle`."""
    return SequentialGComputation(regimens, **kwargs).fit(
        data,
        outcome=outcome,
        treatment=treatment,
        baseline=baseline,
        time_varying=time_varying,
        censoring=censoring,
        id=id,
        weights=weights,
        weights_type=weights_type,
        weights_estimated=weights_estimated,
        family=family,
    )


def ipw(
    data: Any,
    *,
    regimens: Any,
    outcome: str | Sequence[str] | Mapping[str, Sequence[str]],
    treatment: Sequence[str],
    baseline: Sequence[str],
    time_varying: Sequence[Sequence[str]] | None = None,
    censoring: Sequence[str] | None = None,
    id: str | None = None,
    weights: str | None = None,
    weights_type: str = "probability",
    weights_estimated: bool = False,
    family: str = "auto",
    **kwargs: Any,
) -> LongitudinalResult:
    """One-call inverse probability weighting over time, mirroring :func:`ltmle`."""
    return SequentialIPW(regimens, **kwargs).fit(
        data,
        outcome=outcome,
        treatment=treatment,
        baseline=baseline,
        time_varying=time_varying,
        censoring=censoring,
        id=id,
        weights=weights,
        weights_type=weights_type,
        weights_estimated=weights_estimated,
        family=family,
    )


def gformula(
    data: Any,
    *,
    regimens: Any,
    outcome: str | Sequence[str] | Mapping[str, Sequence[str]],
    treatment: Sequence[str],
    baseline: Sequence[str],
    time_varying: Sequence[Sequence[str]] | None = None,
    censoring: Sequence[str] | None = None,
    id: str | None = None,
    weights: str | None = None,
    weights_type: str = "probability",
    weights_estimated: bool = False,
    family: str = "auto",
    continuous_treatment: Sequence[str] = (),
    **kwargs: Any,
) -> LongitudinalResult:
    """One-call parametric g-formula, mirroring :func:`ltmle`.

    Parameters
    ----------
    data : DataFrame or LongitudinalData
        One wide frame, or a container that has already read the columns.
    regimens : mapping, Regimen, DynamicRegimen, or sequence of regimens
        The regimens, as for :class:`ParametricGFormula`.
    outcome : str, sequence of str, or mapping
        The outcome columns, as for :meth:`ParametricGFormula.fit`.
    treatment : sequence of str
        The treatment column of each node.
    baseline : sequence of str
        The covariates measured before any treatment.
    time_varying : sequence of sequence of str or None
        One list of covariate columns per node.
    censoring : sequence of str or None
        One column per node, ``1`` where the unit is still observed.
    id : str or None
        A cluster column.
    weights : str or None
        A column of observation weights.
    weights_type : str
        How to read ``weights``.
    weights_estimated : bool
        Declare that the weights came out of a fitted model.
    family : {"auto", "binomial", "gaussian"}
        The outcome family.
    continuous_treatment : sequence of str
        The treatment columns that hold a continuous dose.
    **kwargs : Any
        The settings of :class:`ParametricGFormula`.

    Returns
    -------
    LongitudinalResult
        The regimen estimates and contrasts.

    See Also
    --------
    ParametricGFormula : The estimator this function builds.
    gcomp : The iterated conditional expectation form of the same parameter.

    Examples
    --------
    >>> from cleverly.datasets import make_longitudinal
    >>> from cleverly.longitudinal import gformula
    >>> frame, _ = make_longitudinal(n=300, seed=0)
    >>> result = gformula(
    ...     frame,
    ...     regimens={"always": 1, "never": 0},
    ...     outcome="Y",
    ...     treatment=["A1", "A2"],
    ...     baseline=["W1", "W2"],
    ...     time_varying=[[], ["L2"]],
    ...     censoring=["C1", "C2"],
    ...     covariate_types={"L2": "normal"},
    ...     random_state=0,
    ... )
    >>> sorted(result.estimates)
    ['ate_regimen[never vs always]', 'ey_regimen[always]', 'ey_regimen[never]']
    """
    return ParametricGFormula(regimens, **kwargs).fit(
        data,
        outcome=outcome,
        treatment=treatment,
        baseline=baseline,
        time_varying=time_varying,
        censoring=censoring,
        id=id,
        weights=weights,
        weights_type=weights_type,
        weights_estimated=weights_estimated,
        family=family,
        continuous_treatment=continuous_treatment,
    )
