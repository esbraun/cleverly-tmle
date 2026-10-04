r"""Classic point-treatment TMLE.

Estimates the effect of a binary treatment ``A`` on an outcome ``Y``, adjusting for
baseline covariates ``W``, under the usual identification assumptions
(consistency, no interference, no unmeasured confounding given ``W``, positivity).

The procedure:

1. **Initial fits.** Estimate ``g(W) = P(A = 1 | W)`` and
   ``Qbar(A, W) = E[Y | A, W]`` -- by default with a
   :class:`~cleverly.learners.SuperLearner`, cross-fitted so no observation is
   predicted by a model that saw it.
2. **Targeting.** Fluctuate ``Qbar`` along a submodel whose score is the efficient
   influence function of the target parameter, and solve for the fluctuation
   coefficient.  This makes the estimator solve the estimated efficient score
   equation ``P_n D*(hat P) = 0``.
3. **Plug in.** Average the targeted predictions to get the estimate, and use the
   influence curve for inference.

Solving that equation is what the guarantees are *built on*, but it does not by itself
supply them, and it is worth separating the two conditions that get conflated:

* **Double robustness** -- consistency if *either* ``g`` or ``Qbar`` is consistent --
  comes from the second-order remainder of the von Mises expansion carrying *both*
  nuisance errors, so that either one being zero kills it.  It needs identification and
  positivity, and it needs no rate on either nuisance.
  ``tests/unit/test_remainder.py`` checks that remainder against its closed form.
* **Asymptotic linearity, valid Wald intervals and efficiency** need more: both
  nuisances consistent at rates whose *product* is ``o(n^{-1/2})``, the score solved to
  ``o_P(n^{-1/2})``, the estimated influence curve converging in ``L_2(P_0)``, and
  control of the empirical-process term (which cross-fitting supplies).  See
  ``targeting_scheme`` below for the full statement.

The practical asymmetry that follows is worth knowing: in the doubly-robust-but-not-
efficient case, where one nuisance is inconsistent, the point estimate is still
consistent but the influence-curve standard error generally is *not*.

Because the targeting step solves an estimating equation rather than optimising a
prediction loss, the resulting estimate is not shrunk toward the null by
regularisation in the nuisance models -- which is what separates TMLE from
plugging machine-learning predictions into a G-computation formula.

Two arguments to :meth:`TMLE.fit` change the *estimand* rather than the estimator, and
both add a factor to the mechanism half of the double-robustness statement above.
``delta=`` puts ``P(Delta = 1 | A, W)`` in the clever covariate's denominator, so the
guarantee becomes "``Qbar`` right, or the product ``g * pi`` right"
(:mod:`cleverly.fluctuation.submodel`).  ``intermediate=`` targets a *controlled direct
effect* -- the effect of ``A`` holding a post-treatment variable ``Z`` fixed at a level
``z`` -- which is a different parameter for each ``z``, so the
:class:`~cleverly.estimators.base.TMLEResultSet` that ``fit`` returns holds one result per
level of ``Z`` instead of the single one an ordinary fit produces.  That path rests on an
identification assumption the average treatment effect does not need, and it is not a
general longitudinal estimator; :mod:`cleverly.estimators.direct_effect` writes the
parameter down, derives its influence function, and says where the boundary is.

``fit`` returns that set in *both* cases.  An ordinary fit is the single-entry one, keyed
``None``, and :meth:`~cleverly.estimators.base.TMLEResultSet.single` is how to reach it.
The alternative -- returning a bare result when there is one and a set when there are two
-- made the return type depend on an argument, which every caller then had to branch on.

Example
-------
>>> from cleverly.estimators import TMLE
>>> from cleverly.datasets import make_nonlinear_bounded
>>> from sklearn.linear_model import LinearRegression, LogisticRegression
>>> frame, _ = make_nonlinear_bounded(n=200, seed=0)
>>> res = TMLE(
...     outcome_learner=LinearRegression(),
...     treatment_learner=LogisticRegression(max_iter=1000),
...     q_bounds=(0.0, 1.0),
...     n_folds=2,
...     random_state=0,
... ).fit(frame, outcome="Y", treatment="A").single()
>>> sorted(res.estimates)
['atc', 'ate', 'att', 'ey0', 'ey1']

``Y`` is a proportion here, so ``q_bounds=(0.0, 1.0)`` states its support rather than
assuming one.  A cross-fitted fit needs that: with ``q_bounds=None`` the outcome scale
comes from every observed outcome, held-out rows included, and each fold's nuisance would
be fitted on a scale the rows it predicts helped set.  A continuous outcome with no known
support is fitted in sample instead, with ``cross_fit=False``.
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from functools import partial
from typing import Any, Literal, cast, get_args

import numpy as np

from .._inference_status import (
    HELD_OUT_SCALE,
    InferenceStatus,
    supplies_inference,
)
from .._typing import (
    BoolArray,
    EstimandName,
    Family,
    FloatArray,
    FluctuationKind,
    FoldStrata,
    GBounds,
    IntArray,
    Learner,
    ParameterAxis,
    TargetingMethod,
    TargetingScheme,
)
from ..data.causal_data import CausalData, TreatmentKind, arm_share
from ..data.validate import MISSING_OUTCOME_DECLARATION
from ..exceptions import (
    CapabilityError,
    ConvergenceWarning,
    DataError,
    PositivityWarning,
    WeightingWarning,
    refuse_after_repeats,
)
from ..fluctuation._score import relative_score, score_columns, score_scale
from ..fluctuation.iterative import (
    Fluctuation,
    FoldFluctuation,
    InitialFit,
    dominant_failure,
)
from ..fluctuation.mechanism import mechanism_covariate, needs_mechanism
from ..fluctuation.submodel import (
    Submodel,
    TargetGroup,
    restrict,
    stitch,
    stratify,
    stratify_columns,
)
from ..inference.bootstrap import Resampling, run_bootstrap
from ..inference.cluster import cluster_inference_status, cross_validated_variance
from ..inference.influence import (
    ArmMean,
    ClusterReference,
    CorrectionParts,
    ParameterEstimate,
    has_t_reference,
    make_estimate,
    median_estimates,
    missing_outcome_correction_parts,
    natural_course_mean,
    reduced_correction_parts,
    stamp_inference,
)
from ..inference.multiplier import MultiplierKind, simultaneous_bands
from ..inference.results import attach_bootstrap
from ..interventions import (
    Incremental,
    IPSISet,
    Policy,
    PolicySet,
    RegimeSet,
    RiskRatioTilt,
    as_interventions,
)
from ..interventions.base import refuse_mixed_interventions, refuse_regime_densities
from ..interventions.incremental import refuse_multi_arm_tilt
from ..interventions.learned import (
    LEARNED_RULE_REMEDY,
    LearnedRule,
    _learned_rule_regimes,
    learned_rule_configuration_refusal,
    learned_rule_record,
    learned_rule_scheme_refusal,
    refuse_learned_rule_composition,
)
from ..interventions.policy import refuse_policy_declarations
from ..learners._fitting import Task, infer_task
from ..learners.crossfit import (
    _POST_DRAW_REMEDY,
    CrossFitPlan,
    Folds,
    SplitPlan,
    _cross_fit_policy_refusal,
    _fresh_seed,
    make_folds,
    missing_training_support,
)
from ..learners.library import _validate_learner
from ..learners.super_learner import SuperLearner, resolve_learner
from ..msm import MSM, MSMSet, refuse_continuous_msm_mechanisms, refuse_msm_functions
from ..provenance import data_fingerprint
from ..provenance import record as provenance_record
from ..targets import (
    TARGETS,
    TargetContext,
    _off_axis_reason,
    groups_for,
    parameter_stem,
    targets_for,
)
from ..targets.base import stratum_alias
from ..targets.population_intervention import (
    NATURAL_COURSE_TARGET,
    POPULATION_INTERVENTION_TARGETS,
)
from ..utils.bounds import OutcomeScaler, g_bounds_for, resolve_g_bounds
from ..utils.frames import is_dataframe
from ._nuisance import NuisanceEstimates, RepeatFit, fit_nuisances
from ._strata import (
    absent_arm_error,
    check_stratified_targets,
    embed_stratum_curve,
    stratum_probabilities,
    stratum_source,
    unscale_block,
)
from .base import (
    CVTargeting,
    TMLEConfig,
    TMLEResult,
    TMLEResultSet,
    resolve_estimands,
)
from .composite import (
    composite_state,
    missing_data_route,
    missing_treatment_refusal,
    unidentified_targets,
)
from .targeting import (
    DEFAULT_MAX_OUTER,
    ProjectionFluctuation,
    ReductionSpec,
    TargetingSpec,
    build_submodel,
    needs_projection,
    needs_reduction,
    reported_beta,
    solve_submodel,
    solve_with_mechanism,
    solve_with_projection,
    solve_with_reduction,
    solve_with_stratified_projection,
    stratum_blocks,
)

__all__ = ["TMLE", "tmle"]

#: Lower bound applied to the missingness and intermediate mechanisms.  These enter
#: the clever covariate as a denominator just as ``g`` does, so they need the same
#: protection against near-zero values.
DEFAULT_NUISANCE_BOUND = 0.01

#: Warn when this fraction of the sample has a propensity outside the truncation
#: bounds -- at that point the estimate rests on extrapolation, not on data.
_TRUNCATION_WARN_FRACTION = 0.05

#: The remedy a stacked natural-course refusal names when the in-sample estimator can run.
#: The in-sample fit reads one fold, and one fold balances nothing, so the fold policy is
#: not part of the remedy: disabling cross-fitting is the whole of it.
_IN_SAMPLE_NATURAL_COURSE_REMEDY = (
    "disable cross-fitting (CrossFitting(enabled=False), or cross_fit=False on the engine)"
)


def _is_natural_course(data: CausalData, estimands: tuple[str, ...]) -> bool:
    """Whether this fit is the missing-outcome natural-course mean.

    The one place the composition is spelled.  Three callers need it before the
    nuisances exist -- the config that records whether ``g`` was fitted, the fit that
    passes ``fit_treatment=``, and the group selection that picks the fluctuation -- so
    none of them can ask :attr:`NuisanceEstimates.fits_treatment`, which is the same
    question answered after the fact.

    Parameters
    ----------
    data : CausalData
        The prepared data, read for its observation mask.
    estimands : tuple of str
        The resolved estimand names.

    Returns
    -------
    bool
        True when the outcome is missing for some rows and the only target is the
        natural-course mean.
    """
    return data.has_missing_outcome and tuple(estimands) == (NATURAL_COURSE_TARGET,)


def _is_joint_natural_course(data: CausalData, estimands: tuple[str, ...]) -> bool:
    """Whether this fit stacks the natural-course mean with arm-indexed targets.

    The one place the joint composition is spelled. With missing outcomes, ``par`` and
    ``paf`` read the natural-course mean beside a reference arm mean, and ``ey_obs``
    beside any arm target is the same stack. The fit solves the shipped natural-course
    fluctuation and the shipped ``mean`` fluctuation separately, from the same initial
    fit, and reports every estimate from the ``mean`` context.

    Parameters
    ----------
    data : CausalData
        The prepared data, read for its observation mask.
    estimands : tuple of str
        The resolved estimand names.

    Returns
    -------
    bool
        True when the outcome is missing for some rows, the request reads the
        natural-course mean, and it is not the scalar natural-course mean alone.
    """
    return (
        data.has_missing_outcome
        and bool({NATURAL_COURSE_TARGET, *POPULATION_INTERVENTION_TARGETS} & set(estimands))
        and tuple(estimands) != (NATURAL_COURSE_TARGET,)
    )


def _reads_natural_course(data: CausalData, estimands: tuple[str, ...]) -> bool:
    """Whether a missing-outcome fit reads the natural-course mean, alone or jointly."""
    return _is_natural_course(data, estimands) or _is_joint_natural_course(data, estimands)


def _is_arm_indexed_missing_crossfit(
    data: CausalData,
    estimands: tuple[str, ...],
    *,
    cross_fit: bool,
    axis: ParameterAxis,
) -> bool:
    """Whether this fit is a cross-fitted arm-indexed fit with missing outcomes.

    The surface of the arm-indexed stacked contract in ``point-treatment-tmle.md``. The
    shift, incremental, regime, MSM, and controlled-direct-effect fits are outside it,
    and so is the scalar natural-course mean, which has its own contract. A joint fit
    that stacks the natural-course mean with arm targets is on both surfaces.

    Parameters
    ----------
    data : CausalData
        The prepared data, read for its observation mask, intermediate, and treatment kind.
    estimands : tuple of str
        The resolved estimand names.
    cross_fit : bool
        Whether the nuisances are cross-fitted.
    axis : {"arm", "regime", "learned_rule", "policy", "rr_tilt", "ipsi", "msm"}
        What the fit's parameters are indexed by.

    Returns
    -------
    bool
        True when the outcome is missing for some rows, the nuisances are cross-fitted,
        the parameters are indexed by a discrete treatment's arms, no intermediate is
        declared, and the request is not the scalar natural-course mean.
    """
    return (
        data.has_missing_outcome
        and cross_fit
        and axis == "arm"
        and not data.has_intermediate
        and not data.is_continuous_treatment
        and not _is_natural_course(data, estimands)
    )


#: The arm-indexed estimands the stacked MAR contract admits, before the outcome family
#: and arm count narrow them.
_ARM_INDEXED_ADMITTED = frozenset({"ey_obs", "par", "paf", "ey", "ey0", "ey1", "ate", "rr", "or"})

#: The names that read the natural-course mean, which the stacked contract admits for a
#: binary outcome only.
_NATURAL_COURSE_NAMES = frozenset({NATURAL_COURSE_TARGET, *POPULATION_INTERVENTION_TARGETS})

#: The first clause of every arm-indexed stacked contract refusal.
_ARM_INDEXED_CONTRACT = (
    "Cross-fitted TMLE of arm-indexed means and contrasts with missing outcomes supports "
    "one audited stacked CV-TMLE contract; "
)

#: The remedy an arm-indexed missing-outcome refusal names when the in-sample estimator
#: can run. One fold balances nothing, so the fold policy is not part of the remedy.
_IN_SAMPLE_ARM_INDEXED_REMEDY = (
    "fit in sample with cross_fit=False on the engine (CrossFitting(enabled=False))"
)

#: The audit and the remedy that both outcome-scale refusals end with.
_SCALE_AUDIT = "(docs/technical-reference/cv-tmle.md, fold and outcome-scale rules)"
#: The learned-rule value has no in-sample fit (X11 (a)), so its outcome-scale refusal
#: names the declared support alone.
_SCALE_SUPPORT = "(Targeting(q_bounds=(lower, upper)))"
_SCALE_REMEDY = (
    f"{_SCALE_SUPPORT}. Without a known finite support, " + _IN_SAMPLE_ARM_INDEXED_REMEDY
)

#: The first clause of the refusal of every other cross-fitted missing-outcome target.
#: Both contracts need arms, so the clause says so: a continuous treatment has neither,
#: and its ``Shift(0.0, cap=None)`` natural course is a shift target this refusal meets.
_CROSS_FITTED_MISSING_CONTRACTS = (
    "Cross-fitted TMLE with missing outcomes (delta=) has two audited stacked CV-TMLE "
    "contracts, the natural-course mean and the arm-indexed means and contrasts, and both "
    "apply to a discrete treatment only; "
)

#: What each parameter axis outside the two contracts is called in that refusal.
_OFF_CONTRACT_AXIS_NAMES: dict[ParameterAxis, str] = {
    "policy": "policy",
    "rr_tilt": "risk-ratio tilt",
    "ipsi": "incremental",
    "regime": "regime",
    "msm": "MSM",
}


def _independent_units(data: CausalData) -> tuple[IntArray, str]:
    """One label per row naming the unit a split moves as a whole, and what to call it.

    A row is the unit of an iid sample, and a *cluster* is the unit once ``id=`` declares
    one: a grouped draw moves every row of a cluster together, so an arm held by one
    cluster is an arm some training complement must lack however the split falls. Reading
    the right unit is what separates a sample that cannot be cross-fitted from one that
    can, so it is written once and named.

    Parameters
    ----------
    data : CausalData
        The prepared data.

    Returns
    -------
    tuple
        The unit label of each row, and the singular noun a refusal calls it.
    """
    if data.cluster is None:
        return np.arange(int(np.asarray(data.treatment).size), dtype=np.int64), "row"
    return np.asarray(data.cluster).reshape(-1), "cluster"


#: Why a package classification Super Learner needs two rows of every class in a training
#: complement. Written once because three refusals state it: the arm-indexed contract's
#: sample and complement checks, and the generic cross-fitting preflight.
_SUPER_LEARNER_INNER_SPLIT_RULE = (
    "The {role} learner is a package SuperLearner with a classification task, and its "
    "inner stratified split needs at least two rows in each class of its target in each "
    "training complement."
)

#: Why a repeated fit cannot carry a simultaneous band.  Written once because the refusal
#: is raised twice: cheaply from the requested estimands before the fit, and again at the
#: construction site, which is the only place the eventual number of estimates is known.
_REPEATED_BANDS_REASON = (
    "The multiplier construction would use a central draw's curve rather than the "
    "split-adjusted median estimator. Set simultaneous=False, report one estimate, "
    "or fit one split."
)


#: Whether the point-treatment full-refit bootstrap's percentile interval is published as
#: inference.  It shipped as inference before the registered study
#: ``full-refit-bootstrap-and-derived-contrasts`` measured it.  A red ``boot_ate_point_tmle``
#: cell sets this to ``False``, so the interval then publishes as a diagnostic.
POINT_BOOTSTRAP_INFERENTIAL = True


def _all_tilts(policies: Sequence[object]) -> bool:
    """Whether every declared policy is a risk-ratio tilt, which has its own estimands."""
    return bool(policies) and all(isinstance(item, RiskRatioTilt) for item in policies)


def _refuse_mixed_tilts(policies: Sequence[object]) -> None:
    """Refuse a fit that declares risk-ratio tilts beside other policies.

    A tilt reports ``ey_rr_tilt`` and ``ate_rr_tilt``, its own estimand names, so it is not
    reported under another policy's name.  One fit reports one parameter axis, so the two
    kinds are fitted separately.
    """
    tilts = [isinstance(item, RiskRatioTilt) for item in policies]
    if any(tilts) and not all(tilts):
        raise CapabilityError(
            "policies= holds RiskRatioTilt beside other policies. A risk-ratio tilt reports "
            "its own estimands, ey_rr_tilt and ate_rr_tilt, and one fit reports one parameter "
            "axis. Fit the tilts and the other policies separately; RiskRatioTilt(1.0) is the "
            "natural course a tilt fit contrasts against"
        )


def _default_treatment_kind(policies: Sequence[object]) -> Literal["discrete", "continuous"]:
    """The treatment kind ``policies=`` declares when ``treatment_kind=`` is not passed.

    A risk-ratio tilt and a label map read labels, so either one declares a categorical
    treatment.  Every other policy moves a dose and so declares a continuous one.  No
    declared policy is a discrete fit.
    """
    from ..interventions import ModifiedPolicy

    if not policies:
        return "discrete"
    categorical = any(
        isinstance(item, RiskRatioTilt)
        or (isinstance(item, ModifiedPolicy) and item.apply is not None)
        for item in policies
    )
    return "discrete" if categorical else "continuous"


class TMLE:
    """Targeted maximum likelihood estimator for a binary point treatment.

    Parameters
    ----------
    outcome_learner, treatment_learner:
        Scikit-learn-compatible nuisance estimators for ``Qbar(A, W)`` and ``g(W)``.
        When omitted, each task receives the concrete default
        :class:`~cleverly.learners.SuperLearner`.
    missingness_learner, intermediate_learner:
        Estimators for ``P(Delta = 1 | A, W)`` and ``P(Z = 1 | A, W)``.  Default to the
        same specification as ``treatment_learner``; only used when the data supplies
        ``delta`` / ``intermediate``.
    family:
        ``"binomial"``, ``"gaussian"``, or ``"auto"`` to infer from the outcome.
    fluctuation:
        ``"logistic"`` (default) keeps targeted predictions inside the outcome's
        range; ``"linear"`` matches R's ``fluctuation="linear"``.
    targeting:
        ``"iterative"`` solves the fluctuation by Newton--Raphson; ``"one_step"`` walks
        the universal least-favorable submodel, which is more robust when the
        fluctuation has to travel far.
    cross_fit:
        Fit nuisances out of fold (default).  ``False`` reproduces R's
        ``cvQinit = FALSE`` and is only appropriate for simple parametric nuisance
        models.

        With missing outcomes (``delta=``), a cross-fitted fit of the arm-indexed means
        and contrasts (``ey``, ``ey0``, ``ey1``, ``ate``, ``rr``, ``or``) follows one
        audited stacked CV-TMLE contract: ``stratify_folds="none"``, ``n_folds`` of at
        least 2, ``repeats=1``, ``targeting_scheme="pooled"``,
        ``cv_evaluation=False``, the logistic iterative fluctuation without
        ``target_weights``, a binary outcome or a continuous one with a fixed
        ``q_bounds``, unweighted iid rows without baseline strata, and no bootstrap.
        The fit refuses ``att``, ``atc``, and every other setting before any learner is
        fitted, and it refuses a sample or a fold whose training complement cannot fit
        the response, treatment, and outcome learners. The point-treatment TMLE reference
        gives the contract and its evidence.

        With missing outcomes, a cross-fitted shift, incremental, regime, MSM, or
        controlled-direct-effect fit is refused before any learner is fitted, because no
        audited result covers it (F21 in ``docs/roadmap.md``). Fit it in sample.
        A continuous-dose MSM with missing outcomes or ``intermediate=`` is refused at
        every setting (X10 in ``docs/roadmap.md``).
    targeting_scheme:
        Where the fluctuation is fit, given cross-fitted nuisances.  ``"pooled"``
        (default) fits one common ``epsilon`` vector on the stacked out-of-fold rows.
        This is Levy's easy CV-TMLE and is the package default. The pinned R ``tmle3``
        source snapshot cited in the references implements the same update when
        ``tmle3_Update(cvtmle=TRUE)`` requests its ``"validation"`` likelihood; Levy's
        paper, not a moving package version, defines the behavior here.

        ``"fold"`` is an additional finite-sample variant that fits a different
        coefficient inside every validation fold.  It is retained for diagnostics and
        comparison, but is not the algorithm those sources define.  In either scheme the
        outcomes in a validation fold are used to fit the coefficient that fluctuates
        that fold; sample splitting applies to the initial nuisance fits, not to epsilon.

        =========================================== ==============================
        setting                                      estimator
        =========================================== ==============================
        ``cross_fit=True, targeting_scheme="pooled"`` stacked CV-TMLE (Levy; default)
        ``targeting_scheme="pooled", cv_evaluation``  fold-evaluated CV-TMLE
        ``targeting_scheme="fold"``                   fold-specific targeted TMLE
        =========================================== ==============================

        The stacked implementation weights rows in the usual empirical distribution,
        exactly like an ordinary TMLE after the out-of-fold predictions have been
        assembled.  ``cv_evaluation=True`` instead normalises observation weights inside
        each fold before taking the original construction's equal ``1/V`` average.
        Cross-fitting removes the entropy condition on the initial nuisance estimators;
        it does not remove the product-rate, positivity, score-convergence, or ``L_2``
        influence-curve conditions.
    cv_evaluation:
        Use the original fold-evaluated construction: evaluate the updated distribution
        in each validation fold, average with weight ``1/V``, and use the matching
        cross-validated variance.  Requires ``cross_fit=True``.  Linear levels and
        contrasts plus ATT/ATC are supported; ``rr``, ``or`` and MSM coefficients are
        refused because their nonlinear fold aggregation has a fold-varying gradient
        that the ordinary fluctuation does not target.  Their stacked-validation reports
        remain available with the default ``cv_evaluation=False``.

        When this setting is true, ``result.cv_targeting`` holds the fold-evaluated
        report and the whole-sample plug-in at the same fold-reweighted fluctuation.
        That plug-in equals the default stacked report only when every fold has equal
        weight mass, which for unweighted rows means equal fold sizes.  A default
        stacked fit builds no such object: its ``result.cv_targeting`` is ``None``.
        For unequal folds the stored
        influence-curve rows are scaled by ``n/(V*n_v)`` so they represent the reported
        equal-fold estimator under the full empirical mean.

        Combines with ``repeats=R``; see there for the variance rule.  :meth:`retarget`
        and the sensitivity analyses follow the setting, so a truncation or missingness
        sweep perturbs the same fold-evaluated estimator the headline reports.
    learned_rule:
        A :class:`~cleverly.interventions.LearnedRule` to estimate the fold-average value of the
        rule learned inside each outer training fold (``ey_learned_rule``).  The learned-rule
        contract of ``docs/technical-reference/point-treatment-tmle.md`` states it.  The fit needs
        ``cross_fit=True``, ``cv_evaluation=True``, ``targeting_scheme="pooled"``, ``repeats=1`` and
        ``n_bootstrap=0``, and refuses every other scheme before any learner, with that remedy.
        It also refuses ``interventions=``, ``policies=``, ``incremental=``, ``msm=``,
        ``reference=`` and an arm estimand beside it, and a continuous or multi-arm treatment,
        missing outcomes, ``intermediate=``, ``weights=``, ``id=`` and ``strata=``.
        :func:`~cleverly.interventions.learned.refuse_learned_rule_composition` states the order.
        The result records the fold summaries under ``result.extra["learned_rule"]``.
    n_folds, learner_folds:
        Outer cross-fitting folds, and the inner folds a Super Learner uses to score
        its candidates.
    split_plan:
        Reusable outer-fold assignments in repeat-major order. This controls only the
        outer folds, so ``random_state`` still controls learner and C-TMLE selection folds.
    repeats:
        How many independent draws of the whole cross-fitting split to combine. ``1``
        (default) is an ordinary fit, and is bit-for-bit an
        ordinary fit rather than an equivalent one.

        A single split is one draw from a randomised procedure, and on a moderate sample
        two seeds can move ``psi`` by an appreciable fraction of its standard error.
        Repeating the split and taking the median is the reporting rule the source below
        gives for that situation. The registered ``repeat_stability`` property compares
        one and three draws across 400 paired fold seeds on a fixed binary sample. Its
        committed result measures the reduction in across-seed spread for that design.
        Read :meth:`~cleverly.estimators.TMLEResult.repeat_spread` on your own fit.
        The point estimate is

        .. math:: \\widetilde\\psi = \\operatorname{median}_r(\\psi_r).

        Ratios use the log scale. The variance is the median of each draw's variance plus
        its squared displacement from the median point. This is the repeated-splitting
        rule in Chernozhukov et al. (2018, equation 3.14) and zEpid's cross-fit TMLE.
        It costs ``R`` times a fit.

        A draw redraws *every* split, not only the outer one: the inner cross-validation
        that scores the Super Learner's candidates, and C-TMLE's selection folds, are
        drawn from the draw's own seed.  Holding those fixed would average over one stage
        of a randomised procedure while pinning the rest.

        The aggregation is the **median**, and only the median. There is no aggregation
        setting and no arithmetic-mean compatibility path. The marginal interval follows
        the established within-plus-between rule. Joint covariance, post-fit contrasts,
        and simultaneous bands are refused. The retained central-draw curve does not
        represent the split-adjusted median estimator that a multiplier band would need.

        ``result.repeats`` holds the per-draw nuisance fits, fluctuations and point
        estimates, and ``result.repeat_spread()`` reports how far the draws moved --
        a diagnostic of the fold noise, never a standard error.  Every sensitivity
        analysis that produces a number follows all ``R``; the diagnostics that describe
        a fitted *mechanism* report the first draw and say so.

        With ``cv_evaluation=True``, each draw contributes its fold-evaluated point and
        cross-validated variance to the same median combination rule.
    stratify_folds:
        What the outer folds are balanced on.  ``"none"`` is the default and the only
        value a cross-fitted fit accepts.  It draws ordinary unstratified V-folds from
        the seed alone, so the assignment reads neither the treatment nor the outcome
        and the cross-fitting argument conditions on a split the data did not choose.

        ``"treatment"`` and ``"treatment+outcome"`` balance the arms, and the arms
        crossed with the outcome, across folds.  Both are refused under cross-fitting.
        No shipped result covers a partition read off the data the fit then conditions
        on (``docs/technical-reference/cv-tmle.md``, fold and outcome-scale rules), and a
        rare level is a reason to fit in sample or
        to collect more of it rather than to let the outcome choose the folds.  A fit
        with ``cross_fit=False`` draws no split, so it accepts any value and uses none
        of them. Selector-based :class:`~cleverly.CTMLE` draws selection folds at every
        setting and refuses the two everywhere. Outcome-adaptive C-TMLE draws no
        selection folds, so its in-sample fit accepts an unused policy.
    g_bounds:
        Propensity truncation.  ``"auto"`` uses ``5 / (sqrt(n) log n)`` for the
        ATE family and ``0.025`` for the ATT/ATC, matching R's ``tmle``.
    q_bounds:
        Assumed support of a continuous outcome (R's ``Qbounds``).  ``None`` widens the
        observed range by 10%, which reads the held-out rows, so a cross-fitted fit of a
        continuous outcome requires the support to be declared here.
    alpha:
        Predicted probabilities are bounded into ``[1 - alpha, alpha]`` before the
        logit is taken.
    target_weights:
        Use the weighted form of the fluctuation (R's ``target.gwt``).
    screen_treatment, screen_threshold, min_retain:
        Pre-screen covariates for the treatment model (R's ``prescreenW.g``).
    estimands:
        Which estimands to report; ``"all"`` requests everything the outcome type
        supports.
    alpha_sig:
        Significance level for confidence intervals.
    n_bootstrap, bootstrap_resampling:
        Run a full-refit bootstrap with this many replicates (R's ``B``): each replicate
        refits the nuisances and the targeting step on a resample.
    simultaneous, n_multiplier, multiplier_kind:
        Simultaneous confidence bands across estimands via the multiplier bootstrap.
        ``multiplier_kind="rademacher"`` (default) and ``"mammen"`` resample and so
        stay accurate when the influence curve has leverage; ``"normal"`` is sampled
        in closed form from the max-t distribution and is far cheaper, but depends on
        the influence curves only through their covariance and is biased conservative
        under weak overlap.  See :mod:`cleverly.inference.multiplier`.
    step_size, max_iter, tol:
        Targeting-step controls.
    run_id:
        An identifier of your own -- an experiment id, a ticket number -- recorded on
        :attr:`TMLEResult.provenance`.  The library records no git commit of its own:
        it must not assume it is being run from inside a repository.
    random_state, n_jobs:
        Reproducibility and parallelism.

    Notes
    -----
    An unfitted instance is reusable: :meth:`fit` returns a new result object and
    does not mutate the estimator's configuration.
    """

    _assessment_method = "tmle"

    def _inference_status(self, data: CausalData) -> InferenceStatus:
        """Whether this estimator's estimates on ``data`` carry inference or a diagnostic.

        A hook rather than a check at the assembly point, because only the estimator
        knows what it did. It reads the estimator configuration and the prepared data
        and nothing fitted, so the status can be determined without learner results. Two
        callers ask it: ``_retarget_detailed`` stamps the estimates, and
        ``variable_importance`` refuses before its first fit. An override that finds more
        than one status resolves them with
        :func:`~cleverly._inference_status.precedent_status`.

        Parameters
        ----------
        data : CausalData
            The prepared data the estimates are fitted on.

        Returns
        -------
        str
            One of :data:`~cleverly.inference.influence.InferenceStatus`.
            :func:`~cleverly.inference.cluster.cluster_inference_status` reads the cluster
            labels. It gives ``"influence_curve"`` on an unclustered fit, and
            ``"few_cluster_plugin"`` on a fit with fewer than ten clusters with positive
            weight mass in total or in one baseline stratum. The cluster sizes do not
            enter, in sample or cross-fitted.
        """
        return cluster_inference_status(
            data.cluster,
            strata=data.strata,
            weights=data.weights if data.is_weighted else None,
        )

    def _refuse_undeclared_functions(self) -> None:
        """Raise unless every function of the configuration is declared known.

        :func:`~cleverly.msm.refuse_msm_functions` on the working model, then
        :func:`~cleverly.interventions.base.refuse_regime_densities` on the regimes. Each
        call site states why it runs this: ``_resolve_estimands_for_data`` before any
        learner, ``_retarget_detailed`` before every recomputation, and
        :func:`cleverly.variable_importance` before its first fit.

        Raises
        ------
        DataError
            If a function is not callable, or a declaration is not one of the three states.
        CapabilityError
            If a declaration is ``None`` or ``"estimated"``.
        """
        if self.msm is not None:
            refuse_msm_functions(self.msm)
        refuse_regime_densities(self.interventions)

    def __init__(
        self,
        *,
        outcome_learner: Learner | None = None,
        treatment_learner: Learner | None = None,
        missingness_learner: Learner | None = None,
        intermediate_learner: Learner | None = None,
        family: Family = "auto",
        fluctuation: FluctuationKind = "logistic",
        targeting: TargetingMethod = "iterative",
        cross_fit: bool = True,
        targeting_scheme: TargetingScheme = "pooled",
        cv_evaluation: bool = False,
        n_folds: int = 10,
        learner_folds: int = 5,
        repeats: int = 1,
        stratify_folds: FoldStrata = "none",
        split_plan: SplitPlan | None = None,
        g_bounds: GBounds = "auto",
        q_bounds: tuple[float, float] | None = None,
        alpha: float = 0.9995,
        nuisance_bound: float = DEFAULT_NUISANCE_BOUND,
        target_weights: bool = False,
        screen_treatment: bool = False,
        screen_threshold: float = 0.1,
        min_retain: int | None = None,
        estimands: Sequence[EstimandName] | str | None = None,
        interventions: Sequence[Any] | None = None,
        policies: Sequence[Policy] | None = None,
        incremental: Sequence[Incremental] | None = None,
        msm: MSM | None = None,
        learned_rule: LearnedRule | None = None,
        density_bins: int = 20,
        ratio: Literal["density", "classifier"] = "density",
        reference: Any = None,
        alpha_sig: float = 0.05,
        n_bootstrap: int = 0,
        bootstrap_resampling: Resampling = "auto",
        simultaneous: bool = True,
        n_multiplier: int = 1000,
        multiplier_kind: MultiplierKind = "rademacher",
        step_size: float = 1e-3,
        max_iter: int = 20,
        tol: float = 1e-10,
        random_state: int | None = None,
        run_id: str | None = None,
        n_jobs: int = 1,
    ) -> None:
        _validate_learner(outcome_learner, "outcome_learner")
        _validate_learner(treatment_learner, "treatment_learner")
        _validate_learner(missingness_learner, "missingness_learner")
        _validate_learner(intermediate_learner, "intermediate_learner")
        self.run_id = run_id
        self.outcome_learner = outcome_learner
        self.treatment_learner = treatment_learner
        self.missingness_learner = missingness_learner
        self.intermediate_learner = intermediate_learner
        self.family = family
        self.fluctuation = fluctuation
        self.targeting = targeting
        self.cross_fit = cross_fit
        self.targeting_scheme = targeting_scheme
        self.cv_evaluation = cv_evaluation
        self.n_folds = n_folds
        self.learner_folds = learner_folds
        self.repeats = repeats
        self.stratify_folds = stratify_folds
        self.split_plan = split_plan
        self.g_bounds = g_bounds
        self.q_bounds = q_bounds
        self.alpha = alpha
        self.nuisance_bound = nuisance_bound
        self.target_weights = target_weights
        self.screen_treatment = screen_treatment
        self.screen_threshold = screen_threshold
        self.min_retain = min_retain
        self.estimands = estimands
        self.interventions = as_interventions(interventions)
        self.policies = tuple(policies or ())
        self.incremental = tuple(incremental or ())
        refuse_mixed_interventions(self.policies, kind="policy", holder="policies=")
        refuse_policy_declarations(self.policies)
        _refuse_mixed_tilts(self.policies)
        refuse_mixed_interventions(self.incremental, kind="incremental", holder="incremental=")
        self.msm = msm
        self.learned_rule = learned_rule
        self.density_bins = density_bins
        self.ratio = ratio
        self.reference = reference
        self.alpha_sig = alpha_sig
        self.n_bootstrap = n_bootstrap
        self.bootstrap_resampling = bootstrap_resampling
        self.simultaneous = simultaneous
        self.n_multiplier = n_multiplier
        self.multiplier_kind = multiplier_kind
        self.step_size = step_size
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._validate_settings()

    def _uses_corrections(self) -> bool:
        return False

    def _validate_settings(self) -> None:
        if self.learned_rule is not None:
            if not isinstance(self.learned_rule, LearnedRule):
                raise DataError(
                    "learned_rule= takes a cleverly.interventions.LearnedRule; got "
                    f"{type(self.learned_rule).__name__}"
                )
            # Rows 1 and 2 of the learned-rule refusal table read the configuration alone, and
            # they come first in its order, so they refuse where they were written.  The
            # preflight asks them again for a copied or modified estimator.
            refusal = learned_rule_configuration_refusal(self)
            if refusal is not None:
                raise CapabilityError(refusal)
        if self.fluctuation not in ("logistic", "linear"):
            raise ValueError(
                f"fluctuation must be 'logistic' or 'linear'; got {self.fluctuation!r}"
            )
        if self.targeting not in ("iterative", "one_step"):
            raise ValueError(f"targeting must be 'iterative' or 'one_step'; got {self.targeting!r}")
        if self.targeting_scheme not in ("pooled", "fold"):
            raise ValueError(
                f"targeting_scheme must be 'pooled' or 'fold'; got {self.targeting_scheme!r}"
            )
        if self.cv_evaluation and not self.cross_fit:
            raise ValueError(
                "cv_evaluation=True reports the estimand over validation folds and "
                "needs cross-fitted nuisance predictions; pass cross_fit=True"
            )
        if self.targeting_scheme == "fold" and not self.cross_fit:
            warnings.warn(
                "targeting_scheme='fold' needs validation folds to target within, and "
                "cross_fit=False leaves none; falling back to pooled targeting. Set "
                "cross_fit=True for the fold-specific targeting extension.",
                UserWarning,
                stacklevel=3,
            )
        if self.targeting_scheme == "fold" and self.incremental:
            raise CapabilityError(
                "targeting_scheme='fold' is not implemented for incremental interventions: "
                "their targeting alternates the outcome and treatment mechanisms, and a "
                "fold-specific version needs both equations re-solved inside every fold. "
                "Use targeting_scheme='pooled', which is the literature-backed common-validation "
                "CV-TMLE update, or fit without incremental=."
            )
        if self.targeting == "one_step" and self.fluctuation == "linear":
            raise ValueError(
                "targeting='one_step' walks a logistic submodel and cannot be combined with "
                "fluctuation='linear'"
            )
        if not 0.0 < self.alpha_sig < 1.0:
            raise ValueError(f"alpha_sig must lie in (0, 1); got {self.alpha_sig}")
        if not 0.0 < self.nuisance_bound < 0.5:
            raise ValueError(f"nuisance_bound must lie in (0, 0.5); got {self.nuisance_bound}")
        if self.n_bootstrap and self.n_bootstrap < 2:
            raise ValueError(f"n_bootstrap must be 0 or at least 2; got {self.n_bootstrap}")
        declared = [
            name
            for name, value in (
                ("interventions=", self.interventions),
                ("policies=", self.policies),
                ("incremental=", self.incremental),
            )
            if value
        ]
        if len(declared) > 1:
            raise CapabilityError(
                f"{' and '.join(declared)} each declare what this fit's counterfactuals "
                "are -- a regime assigns a distribution over the arms from W alone, a "
                "shift moves the dose the unit actually received, and an incremental "
                "intervention tilts the odds of the mechanism that was already there -- "
                "and one fluctuation cannot "
                "solve their score equations at once. Fit them separately. "
                "docs/roadmap.md F17 tracks this stop."
            )
        if self.incremental and self.g_bounds != "auto":
            raise ValueError(
                "g_bounds= truncates the treatment mechanism, and on an incremental fit "
                "the mechanism is part of the *estimand*: q_delta = delta*g / "
                "(delta*g + 1 - g), so truncating g moves Psi(delta) itself rather than "
                "regularising a denominator. It is also unnecessary -- the clever "
                "covariate is delta/D at A=1 and 1/D at A=0, both between "
                "min(delta, 1/delta) and max(delta, 1/delta) whatever g is, which is the "
                "point of an incremental intervention. Leave g_bounds at its default. "
                "(nuisance_bound= is a different matter and is accepted: with delta= the "
                "missingness mechanism divides the covariate and is not in the estimand, "
                "so bounding it regularises rather than retargets.)"
            )
        if self.msm is not None and (self.interventions or self.policies or self.incremental):
            other = (
                "interventions="
                if self.interventions
                else "policies="
                if self.policies
                else "incremental="
            )
            raise CapabilityError(
                f"msm= and {other} cannot be combined. A working model summarises the "
                "counterfactual means with p score equations, one per term, and "
                f"{other} replaces what those means are; one fluctuation cannot solve "
                "both. A working model over declared regimes is a coherent estimand and "
                "is not implemented -- its design would have to be indexed by regime "
                "rather than by arm. docs/roadmap.md F17 tracks this stop."
                + (
                    " A working model over modified treatment policies is fitted by LTMLE "
                    "with one node: LTMLE(regimens={...}, msm=MSM(...))."
                    if self.policies
                    else ""
                )
            )
        if self.msm is not None and self.reference is not None:
            raise ValueError(
                "reference= names the arm, regime or shift every contrast is taken "
                "against, and a working model reports coefficients rather than contrasts "
                "-- there is nothing for it to be a reference for. Which arm is the "
                "baseline is decided by the design you gave msm=, usually by an intercept "
                "column. A difference of two coefficients comes from result.contrast()."
            )
        if self.stratify_folds not in get_args(FoldStrata):
            allowed = ", ".join(repr(value) for value in get_args(FoldStrata))
            raise ValueError(
                f"stratify_folds must be one of {allowed}; got {self.stratify_folds!r}"
            )
        if self.ratio not in ("density", "classifier"):
            raise ValueError(
                f"ratio must be 'density' or 'classifier'; got {self.ratio!r}. 'density' "
                "estimates a policy's ratio from a binned conditional density of the dose, and "
                "'classifier' from a stacked classification (Diaz et al. 2023, Section 5.4)"
            )
        if self.density_bins < 3:
            raise ValueError(
                f"density_bins must be at least 3; got {self.density_bins}. Two bins make "
                "the density a single hazard, which cannot describe a dose-response."
            )
        # One ordered message source for the declaration and the engine, under two
        # exception contracts: the declaration raises MethodConfigurationError and the
        # engine raises ValueError.  See ``_cross_fit_policy_refusal``.
        reason = self._cross_fit_policy_reason()
        if reason is not None:
            # A learned-rule fit hears rows 10 to 14 of its refusal table before a shared policy
            # sentence.  Their remedy is the configuration that fits, and the shared remedies
            # (enable cross-fitting, or set repeats=1) meet one of those rows next.  The rows raise
            # ``CapabilityError``, as every row of that table does.  A policy that passes the shared
            # checks meets these rows in the preflight instead, after the data rows, because a
            # refusal that no setting repairs comes first.
            learned = None if self.learned_rule is None else learned_rule_scheme_refusal(self)
            if learned is not None:
                raise CapabilityError(learned)
            raise ValueError(reason)

    def _cross_fit_policy_reason(self) -> str | None:
        """Why this estimator's declared fold policy cannot run, or ``None``.

        Asked twice on purpose.  :meth:`_validate_settings` asks it at construction, so a
        declaration that cannot run is refused where it was written.  A fit asks it again
        in :meth:`_resolve_estimands_for_data`, because two supported operations reach a
        fit without running ``__init__``: :meth:`refit` copies an estimator, and a caller
        can reassign an attribute of a constructed estimator.  Neither may fit under a
        policy this one refuses.

        A learned-rule fit has no in-sample fit, so its sentence offers none.
        """
        return _cross_fit_policy_refusal(
            cross_fit=self.cross_fit,
            n_folds=self.n_folds,
            repeats=self.repeats,
            split_plan=self.split_plan,
            n_bootstrap=self.n_bootstrap,
            stratify_folds=self.stratify_folds,
            collaborative=(
                self._assessment_method == "collaborative_tmle"
                # Only a ``CTMLE`` reports this method, and its constructor sets
                # ``strategy`` before the base constructor asks this policy. The
                # outcome-adaptive strategy draws no selector folds.
                and cast("Any", self).strategy != "oat"
            ),
            option_name="cross_fit",
            learned_rule=self.learned_rule is not None,
        )

    # ------------------------------------------------------------------- fit

    def fit(
        self,
        data: Any,
        *,
        outcome: str | None = None,
        treatment: str | None = None,
        covariates: Sequence[str] | None = None,
        delta: str | None = None,
        weights: str | None = None,
        weights_type: str = "probability",
        weights_estimated: bool = False,
        id: str | None = None,
        intermediate: str | None = None,
        strata: Sequence[str] | None = None,
        treatment_kind: TreatmentKind | None = None,
        treatment_delta: str | None = None,
    ) -> TMLEResultSet:
        """Fit the estimator.

        Parameters
        ----------
        data:
            A pandas or polars dataframe, or a prepared
            :class:`~cleverly.data.CausalData`.
        outcome, treatment, covariates, delta, weights, id, intermediate:
            Column names, when ``data`` is a dataframe.  ``covariates=None`` uses every
            column not claimed by another role.
        treatment_kind:
            ``"discrete"`` to code the treatment column into arms, ``"continuous"`` to
            keep its own values and model it with a conditional density.  ``None``
            follows ``policies=``: a modified treatment policy moves a dose and names no
            arm, so declaring one declares the treatment continuous.

            That is a default read off another *declaration*, not off the data -- a
            column at fifteen distinct values could reasonably be read either way, and
            :class:`~cleverly.data.CausalData` refuses to guess from the level count for
            exactly that reason.  Pass this explicitly to override, including to get the
            arm-coded refusal for a ``policies=`` fit on a treatment that really has arms.
        weights_type, weights_estimated:
            How to read ``weights``.  Supplying weights changes the estimand to the
            causal parameter in the weight-tilted population -- see
            :mod:`cleverly.data.weighting` and
            :meth:`~cleverly.data.CausalData.from_frame`.
        treatment_delta:
            The 0/1 column that is 1 where the treatment is recorded, which declares a
            treatment missing at random.  The fit then runs the composite-indicator
            construction of :mod:`cleverly.estimators.composite` for the arm means and
            their contrasts, in sample.  See :meth:`~cleverly.data.CausalData.from_frame`.

        Returns
        -------
        A :class:`~cleverly.estimators.base.TMLEResultSet`: one
        :class:`~cleverly.estimators.base.TMLEResult` per parameter estimated.  An
        ordinary fit holds a single result, keyed ``None`` -- reach it with
        ``.single()``.  Passing ``intermediate=`` holds one per level of the intermediate,
        keyed by the level, because a controlled direct effect is a different parameter at
        each.
        """
        prepared = self._prepare(
            data,
            outcome=outcome,
            treatment=treatment,
            covariates=covariates,
            delta=delta,
            weights=weights,
            weights_type=weights_type,
            weights_estimated=weights_estimated,
            id=id,
            intermediate=intermediate,
            strata=strata,
            treatment_kind=treatment_kind,
            treatment_delta=treatment_delta,
        )
        if not prepared.has_intermediate:
            return TMLEResultSet({None: self._fit_single(prepared, intermediate_value=None)})

        # The intermediate path otherwise fits shared nuisances before `_fit_single`.
        # Run the full preflight here as well, so every unsupported intermediate
        # composition fails before the shared nuisance learners are fitted.
        estimands = self._preflight_fit_configuration(prepared)

        # The controlled direct effect at z = 0 and at z = 1 are different parameters,
        # so they get one result each -- but they are estimated from *identical*
        # nuisance models. Every model here (propensity, missingness, the intermediate
        # mechanism, and the outcome regression, whose design uses the observed Z)
        # is level-independent; only the counterfactual designs the outcome regression
        # is predicted onto differ. Fitting per level refits all four to obtain two
        # extra prediction vectors, so the levels are estimated in one pass and the
        # targeting step is run twice against the shared fits.
        levels = (0.0, 1.0)
        if self._shares_nuisances_across_levels():
            shared = self._prepare_shared(prepared, levels, estimands)
            results: dict[float | None, TMLEResult] = {
                value: self._fit_single(prepared, intermediate_value=value, shared=shared)
                for value in levels
            }
        else:
            results = {
                value: self._fit_single(prepared, intermediate_value=value) for value in levels
            }
        return TMLEResultSet(results, prepared.intermediate_name or "Z")

    def _shares_nuisances_across_levels(self) -> bool:
        """Whether the two controlled direct effects can share one set of nuisance fits.

        True for the base estimator, where the nuisances are level-independent by
        construction.  A variant that chooses *which* nuisance to hand to the targeting
        step -- :class:`~cleverly.CTMLE`, whose propensity selection is scored against a
        level-specific targeted loss -- must opt out and refit per level.
        """
        return type(self)._nuisances is TMLE._nuisances

    def _prepare_shared(
        self, data: CausalData, levels: Sequence[float], estimands: tuple[str, ...]
    ) -> tuple[OutcomeScaler, tuple[tuple[Folds, NuisanceEstimates, int], ...]]:
        """Fit the level-independent nuisances once, for every requested level.

        One ``(folds, nuisance, seed)`` triple per repeat: the levels share nuisances *within* a
        draw, which is what this method exists for, and the draws remain separate, which
        is what makes them repeats.  The scaler is a function of the outcome alone and so
        is shared across both.
        """
        scaler = self._scaler(data)
        draws = []
        for folds, seed in self._repeat_draws(data, estimands):
            draws.append(
                (
                    folds,
                    self._fit_nuisances(
                        data, folds, scaler, levels[0], tuple(levels[1:]), seed=seed
                    ),
                    seed,
                )
            )
        return scaler, tuple(draws)

    def refit(
        self,
        data: CausalData,
        *,
        intermediate_value: float | None = None,
        random_state: int | None = None,
    ) -> TMLEResult:
        """Run the whole fit again -- nuisances included -- on already-prepared data.

        This is the expensive counterpart to :meth:`retarget`, and the distinction
        matters.  ``retarget`` re-solves the fluctuation against cached nuisance
        estimates, so it needs nothing but arrays and is what every truncation sweep
        and every bootstrap replicate uses.  ``refit`` re-learns the nuisances, which
        is unavoidable when the *data* changed: the negative-control refutations
        (:mod:`cleverly.validation.refute`) replace the treatment or the outcome, and
        the omitted-variable analysis drops a covariate from the adjustment set.

        Pass ``intermediate_value`` when the data carries an intermediate variable, so
        the refit targets the same controlled direct effect as the original.

        ``random_state`` runs the refit under a seed of the caller's choosing, on the
        package convention that ``None`` means this estimator's own.  A refit re-learns
        the nuisances, so an estimator carrying no seed redraws its folds every time and
        gives a different answer to the same question.  A caller that has to repeat a
        refit -- :mod:`cleverly.validation.refute` reports the seed it used, so that a
        reader can -- supplies one here.  The estimator is not modified: the seed applies
        to a copy, and this instance keeps the ``random_state`` it was built with.

        A supplied :class:`~cleverly.SplitPlan` reaches the refit *unbound* from the rows
        it was realised on, and the copy is what makes that local too.  The binding exists
        to catch a plan reused on other rows, where a label silently points at another
        unit.  Every refit in this package is the same rows with a column replaced,
        perturbed, dropped or added -- which is the exact condition
        :meth:`~cleverly.SplitPlan.validate` names as safe to reuse labels under -- so
        holding the fingerprint here would refuse the placebo and negative-control
        refutations for a change that moves no row.  What a refit cannot do is change the
        row *set*: the count check inside ``validate`` still refuses that, and
        :func:`~cleverly.validation.refute` refuses the subsampling test up front.
        :meth:`~cleverly.SplitPlan.unbound` keeps the plan's generator record, so the
        refit accepts the plan and draws its labels again to check them.

        The refit first asks :meth:`_configured_for_refit` which estimator fits ``data``.
        This estimator answers itself.  A :class:`~cleverly.CTMLE` with an explicit
        ``ordering=`` places an added covariate after the declared ordering, a
        ``"discrete"`` :class:`~cleverly.CTMLE` appends it to every candidate, and a
        :class:`~cleverly.DRTMLE` drops an ``evaluation=`` companion that lacks a covariate
        of ``data``.  Either answer is a copy, so this instance is not modified.

        Parameters
        ----------
        data : CausalData
            The prepared data to fit, usually ``result.data`` with one column changed.
        intermediate_value : float or None
            The value of the intermediate variable the controlled direct effect fixes.
        random_state : int or None
            Seed of the refit.  ``None`` keeps this estimator's own.

        Returns
        -------
        TMLEResult
            The result of the whole fit on ``data``.
        """
        estimator = self._configured_for_refit(data)
        plan = estimator.split_plan
        if plan is not None and plan.source_fingerprint is not None:
            if estimator is self:
                estimator = copy.copy(self)
            estimator.split_plan = plan.unbound()
        if random_state is not None and random_state != estimator.random_state:
            if estimator is self:
                estimator = copy.copy(self)
            estimator.random_state = random_state
        return estimator._fit_single(data, intermediate_value=intermediate_value)

    def _configured_for_refit(self, data: CausalData) -> TMLE:
        """The estimator a refit on ``data`` runs, which is this one.

        A subclass whose configuration names the covariates of the data it was fitted on
        overrides this, so that a refit that adds a covariate still has a configuration
        that covers it.  An override returns a copy and never modifies this instance.

        Parameters
        ----------
        data : CausalData
            The prepared data the refit fits.

        Returns
        -------
        TMLE
            This estimator.
        """
        return self

    def _prepare(
        self,
        data: Any,
        *,
        outcome: str | None,
        treatment: str | None,
        covariates: Sequence[str] | None,
        delta: str | None,
        weights: str | None,
        id: str | None,
        intermediate: str | None,
        strata: Sequence[str] | None = None,
        weights_type: str = "probability",
        weights_estimated: bool = False,
        treatment_kind: TreatmentKind | None = None,
        treatment_delta: str | None = None,
    ) -> CausalData:
        """Coerce whatever the caller passed into a validated :class:`CausalData`."""
        if isinstance(data, CausalData):
            if weights_type != "probability" or weights_estimated:
                raise ValueError(
                    "weights_type/weights_estimated cannot be combined with a CausalData "
                    "input; pass them to CausalData.from_frame or from_arrays, which is "
                    "where the weight column is read"
                )
            if any(
                value is not None
                for value in (
                    outcome,
                    treatment,
                    covariates,
                    delta,
                    weights,
                    id,
                    intermediate,
                    strata,
                    treatment_kind,
                    treatment_delta,
                )
            ):
                raise ValueError(
                    "column names cannot be combined with a CausalData input; the roles are "
                    "already assigned"
                )
            return data
        if not is_dataframe(data):
            raise TypeError(
                "fit expects a pandas or polars DataFrame, or a CausalData. For numpy arrays "
                "use cleverly.tmle(Y, A, W, ...) or CausalData.from_arrays."
            )
        if outcome is None or treatment is None:
            raise ValueError("outcome= and treatment= are required when fitting from a dataframe")
        return CausalData.from_frame(
            data,
            outcome=outcome,
            treatment=treatment,
            covariates=covariates,
            delta=delta,
            weights=weights,
            weights_type=weights_type,
            weights_estimated=weights_estimated,
            id=id,
            intermediate=intermediate,
            strata=strata,
            family=self.family,
            treatment_kind=(
                _default_treatment_kind(self.policies) if treatment_kind is None else treatment_kind
            ),
            treatment_delta=treatment_delta,
        )

    def _preflight_fit_configuration(self, data: CausalData) -> tuple[str, ...]:
        """Resolve targets and run every configuration guard before fitting learners.

        Both fit and replayability use this path, including subclass estimand checks.
        The returned targets include any exclusions for ``estimands="all"``.
        Scaler, propensity-bound, and reference checks need no generated folds and run
        here too. The validation of realised folds stays in :meth:`_repeat_draws`.

        A learned-rule fit runs
        :func:`~cleverly.interventions.learned.refuse_learned_rule_composition` first, before
        :meth:`_check_policies`.  That check would answer a continuous treatment by suggesting
        ``policies=``, which a learned-rule fit refuses.

        A declared missing treatment meets
        :func:`~cleverly.estimators.composite.missing_treatment_refusal` before both, which
        is the one gate for it.  The default and ``"all"`` target lists then drop the
        targets that the composite construction cannot identify, as they drop
        ``par`` and ``paf`` beside ``delta=``.
        """
        if data.has_missing_treatment:
            named = None if self.estimands is None or self.estimands == "all" else self.estimands
            refusal = missing_treatment_refusal(
                self,
                data,
                None
                if named is None
                else resolve_estimands(named, data.family, data.n_arms, axis=self._axis),
            )
            if refusal is not None:
                raise refusal
        if self.learned_rule is not None:
            refuse_learned_rule_composition(data, self)
        self._check_policies(data)
        self._check_incremental(data)
        estimands = self._resolve_estimands_for_data(data)
        if data.has_missing_treatment:
            estimands = tuple(name for name in estimands if name not in unidentified_targets())
        if data.has_strata and self.targeting_scheme == "fold":
            raise CapabilityError(
                "baseline strata are not combined with targeting_scheme='fold': each stratum "
                "would need a fold-local update in place of the pooled one, and no published "
                "result covers a fold-local update. Use the default pooled targeting scheme"
            )
        if data.has_strata and self.cv_evaluation:
            raise CapabilityError(
                "baseline strata are not implemented with cv_evaluation=True: the "
                "fold-evaluated estimate needs stratum shares and a stratum-indexed fold "
                "average inside every validation fold. Use cv_evaluation=False. "
                "docs/roadmap.md X28 tracks it"
            )
        if self.simultaneous and len(estimands) > 1:
            # Guarded by the band's own construction condition, not by ``simultaneous``
            # alone.  ``simultaneous`` defaults to True and a band needs two estimates, so
            # refusing on the flag would refuse every single-estimand repeated fit over a
            # band the fit would never build.  One estimand can still expand into several
            # estimates, so the construction site refuses again once the count is known.
            refuse_after_repeats(
                self.repeats, operation="simultaneous=True", reason=_REPEATED_BANDS_REASON
            )
        population_intervention = POPULATION_INTERVENTION_TARGETS.intersection(estimands)
        # ``estimands="all"`` drops ``ey_obs``, ``par`` and ``paf`` beside either
        # declaration. With missing outcomes a joint natural-course fit is a named
        # request: it adds a second fluctuation, so an ``all`` fit keeps the arm-only
        # fluctuation set and its estimates bit for bit. Beside ``intermediate=`` the
        # composition has no identified parameter, and an explicit list refuses below.
        droppable = population_intervention | ({NATURAL_COURSE_TARGET} & set(estimands))
        if (
            droppable
            and self.estimands == "all"
            and (data.has_missing_outcome or data.has_intermediate)
        ):
            estimands = tuple(name for name in estimands if name not in droppable)
            population_intervention = frozenset()
        # The natural-course contract refuses ``intermediate=`` with missing outcomes, but
        # it returns early on a complete outcome. Without this check a complete-outcome
        # fit runs, publishes the plain empirical mean of Y once per level of Z, and
        # labels each copy a controlled direct effect -- a different estimand reported
        # under a name it did not earn.
        beside_intermediate = population_intervention | ({NATURAL_COURSE_TARGET} & set(estimands))
        if beside_intermediate and data.has_intermediate:
            raise CapabilityError(
                f"{sorted(beside_intermediate)} do not yet support intermediate=: "
                "combining the natural course with a controlled mediator intervention "
                "needs a separately identified population-intervention parameter"
            )
        if self.cv_evaluation:
            unsupported = [
                name for name in estimands if parameter_stem(name) in {"rr", "or", "msm"}
            ]
            if unsupported:
                raise CapabilityError(
                    "cv_evaluation=True does not yet support "
                    f"{unsupported}: averaging a nonlinear parameter over folds changes "
                    "its gradient fold by fold, so the ordinary mean/MSM fluctuation no "
                    "longer solves that cross-validated score. The stacked-validation "
                    "report (cv_evaluation=False) remains supported; request linear "
                    "levels/contrasts or ATT/ATC for fold-wise evaluation."
                )

        # These checks also run in `_scaler` and `_config` before any nuisance learner.
        # Read the same methods here so copied settings reach the replay slot.
        self._scaler(data)
        resolve_g_bounds(self.g_bounds, self._bounds_n(data), for_att=False)
        resolve_g_bounds(self.g_bounds, self._bounds_n(data), for_att=True)
        self._reference_arm(data)
        return estimands

    def _fit_single(
        self,
        data: CausalData,
        *,
        intermediate_value: float | None,
        shared: (
            tuple[OutcomeScaler, tuple[tuple[Folds, NuisanceEstimates, int], ...]] | None
        ) = None,
    ) -> TMLEResult:
        """Fit for one value of the intermediate (or for no intermediate at all).

        ``shared`` supplies nuisance fits already computed for every level of the
        intermediate; only the targeting step then runs per level.

        With ``repeats=R`` the whole construction below -- split, nuisances, targeting --
        runs ``R`` times and the reports are combined by their median. The loop sits here, around
        :meth:`_nuisances` rather than inside it, which is what makes it free for the
        variants: :class:`~cleverly.CTMLE` overrides that method alone, so its propensity
        selection is repeated per draw without ``estimators/ctmle.py`` knowing repeats
        exist. Bootstrap inference repeats the same complete procedure. A simultaneous band
        is refused, because its multiplier draws would use the retained central-draw curve
        rather than the split-adjusted median estimator. The refusal needs two or more
        estimates, which is what a band needs. A repeated fit that reports one estimate
        builds no band, so it is allowed.
        """
        estimands = self._preflight_fit_configuration(data)

        extra: dict[str, Any] = {}
        if shared is not None:
            scaler, pooled = shared
            fold_draws = [folds for folds, _, _ in pooled]
            draw_seeds = [seed for _, _, seed in pooled]
            nuisances = [
                nuisance.at_level(cast("float", intermediate_value)) for _, nuisance, _ in pooled
            ]
            config = self._config(data, estimands, scaler, fold_draws[0])
        else:
            scaler = self._scaler(data)
            draws = self._repeat_draws(data, estimands)
            fold_draws = [folds for folds, _ in draws]
            draw_seeds = [seed for _, seed in draws]
            # The realised fold count can differ between draws when a cap fires on one and
            # not another, so the config -- like every read-through attribute on the result
            # -- describes the first draw.  It is then the *same* config for every draw,
            # which matters: the truncation bounds a draw is fitted under must not depend
            # on which draw it is, or the R estimates would not be estimating one thing.
            config = self._config(data, estimands, scaler, fold_draws[0])
            nuisances = []
            for index, (folds, seed) in enumerate(draws):
                nuisance, draw_extra = self._nuisances(
                    data, folds, scaler, config, intermediate_value, seed=seed
                )
                nuisances.append(nuisance)
                if index == 0:
                    extra = draw_extra

        # The construction this fit ran, which every reader of the result branches on.
        extra = {**extra, "missing_data": missing_data_route(self, data)}
        self._warn_on_positivity(data, nuisances[0], config, intermediate_value)
        self._warn_on_estimated_weights(data)

        per_repeat: list[dict[str, ParameterEstimate]] = []
        repeats: list[RepeatFit] = []
        details: list[CVTargeting | None] = []
        for nuisance, seed in zip(nuisances, draw_seeds, strict=True):
            # ``_retarget_detailed`` applies ``_inference_status`` to what it returns, and to
            # both fold-level reports, so ``per_repeat``, ``median_estimates``,
            # ``result.repeats`` and ``result.cv_targeting`` all carry it.
            estimates, fluctuations, detail = self._retarget_detailed(
                data,
                nuisance,
                estimands=estimands,
                intermediate_value=intermediate_value,
                g_bounds=config.g_bounds,
                g_bounds_conditional=config.g_bounds_conditional,
            )
            per_repeat.append(estimates)
            repeats.append(
                RepeatFit(
                    nuisance=nuisance,
                    fluctuations=fluctuations,
                    psi={name: value.psi for name, value in estimates.items()},
                    seed=seed,
                )
            )
            details.append(detail)

        estimates = median_estimates(per_repeat)
        cv_detail = self._cv_detail(details, cluster=data.cluster)
        if self.cv_evaluation and cv_detail is None:
            raise RuntimeError(
                "cv_evaluation=True needs at least two realised validation folds, but "
                "the requested split collapsed to one. Use fewer-stratified data, or "
                "fit without fold-evaluated CV-TMLE."
            )
        # A learned-rule fit is fold-evaluated, so the check above leaves ``cv_detail`` set.
        # The second test is for the type checker.
        if self.learned_rule is not None and cv_detail is not None:
            # One draw: ``refuse_learned_rule_composition`` refuses repeats.
            (name,) = cv_detail.fold_estimates
            record = learned_rule_record(
                self.learned_rule, nuisances[0], cv_detail.fold_estimates[name]
            )
            extra = {**extra, "learned_rule": record}
        result = TMLEResult(
            estimates=estimates,
            repeats=tuple(repeats),
            data=data,
            config=config,
            estimator=self,
            fitted_method=self._assessment_method,
            solved_corrections=self._uses_corrections(),
            provenance=provenance_record(
                data, fold_draws, random_state=self.random_state, run_id=self.run_id
            ),
            intermediate_value=intermediate_value,
            extra=extra if cv_detail is None else {**extra, "cv_tmle": cv_detail},
        )

        # A simultaneous band is a joint confidence statement, so a fit that supplies no inference
        # builds none.  Skipped rather than raised, because ``simultaneous`` defaults to ``True``:
        # raising would stop every default-configured selector-path collaborative fit over an output
        # that the selector status refuses to report anyway.  The omission is not silent --
        # ``summary()`` prints the reason, and ``simultaneous_bands()`` called directly still
        # refuses, because that is an explicit request.
        # A band over an estimate with a Student t reference is skipped for the same
        # reason: no source gives a t-calibrated joint band, and ``summary()`` says so.
        if (
            self.simultaneous
            and len(estimates) > 1
            and supplies_inference(result.inference_status)
            and not has_t_reference(estimates)
        ):
            refuse_after_repeats(
                self.repeats, operation="simultaneous=True", reason=_REPEATED_BANDS_REASON
            )
            bands = simultaneous_bands(
                estimates,
                alpha=self.alpha_sig,
                n_replicates=self.n_multiplier,
                kind=self.multiplier_kind,
                random_state=self.random_state,
                cluster=data.cluster,
            )
            result = replace(result, simultaneous=bands)

        if self.n_bootstrap:
            bootstrap = run_bootstrap(
                data,
                lambda replicate: self._bootstrap_point_estimates(
                    replicate, intermediate_value, estimands
                ),
                n_replicates=self.n_bootstrap,
                resampling=self.bootstrap_resampling,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
            result = attach_bootstrap(result, bootstrap, inferential=POINT_BOOTSTRAP_INFERENTIAL)

        return result

    def _cv_detail(
        self, details: Sequence[CVTargeting | None], *, cluster: IntArray | None
    ) -> CVTargeting | None:
        """One fold-level report for the whole fit, however many draws it combines.

        The fields split by what they *are*.  ``pooled``, ``fold_evaluated`` and ``variance``
        are estimates, so they follow every draw exactly as the headline report does.
        ``n_folds``, ``fold_sizes``, ``fold_estimates`` and ``fold_epsilon`` are indexed
        by fold, and fold 3 of one draw is not fold 3 of another -- there is no
        correspondence to combine along -- so they describe the first draw and
        :class:`~cleverly.CVTargeting` says which.

        A draw that produced no fold detail at all while others did would mean the draws
        were not reporting the same estimator, so under ``cv_evaluation`` it is refused
        rather than averaged around.
        """
        present = [detail for detail in details if detail is not None]
        if not present:
            return None
        if self.cv_evaluation and len(present) != len(details):
            raise RuntimeError(
                f"{len(details) - len(present)} of {len(details)} cross-fitting draws "
                "produced no validation folds to evaluate within while the others did, so "
                "cv_evaluation=True would combine fold-wise estimates from some draws "
                "with pooled ones from the rest under a single name. Re-run with fewer "
                "n_folds, or with repeats=1."
            )
        first = present[0]
        if len(present) == 1:
            return first
        fold_evaluated = median_estimates([detail.fold_evaluated for detail in present])
        return replace(
            first,
            repeats=len(present),
            pooled=median_estimates([detail.pooled for detail in present]),
            fold_evaluated=fold_evaluated,
            variance={name: value.variance for name, value in fold_evaluated.items()},
        )

    # ------------------------------------------------------------- internals

    def _scaler(self, data: CausalData) -> OutcomeScaler:
        """The outcome transformation: identity for a binary outcome, scaling else."""
        if data.family == "binomial":
            if self.q_bounds is not None:
                raise ValueError("q_bounds does not apply to a binary outcome")
            return OutcomeScaler.identity()
        observed = data.outcome[data.observed]
        return OutcomeScaler.from_outcome(observed, self.q_bounds)

    def _resolve_estimands_for_data(self, data: CausalData) -> tuple[str, ...]:
        """Resolve targets and enforce the initial data and declaration refusals.

        This method first checks the declared fold policy without reading the data, because a
        fit can arrive here without having run ``__init__``: :meth:`refit` copies an
        estimator, and a caller can reassign an attribute of a constructed estimator.  It
        then runs :func:`~cleverly.msm.refuse_msm_functions` and
        :func:`~cleverly.interventions.base.refuse_regime_densities`, through
        :meth:`_refuse_undeclared_functions`, for the same reason: a modified model or
        regime can carry a declaration this version refuses.
        The regime check covers a ``Stochastic`` density, a ``Rule``, and a user-written
        ``Intervention``, which ``TMLE.__init__`` admits without a check.
        Those functions state what each one refuses.  The declaration check runs before
        the data contracts in this method. :meth:`MSMSet.evaluate
        <cleverly.msm.MSMSet.evaluate>` checks the model again, but a fit reaches it only
        after those refusals.

        The refusal of a continuous-dose MSM with a missing outcome or an intermediate
        variable, :func:`~cleverly.msm.refuse_continuous_msm_mechanisms`, runs next. It
        holds at every ``cross_fit`` setting, so it runs before the cross-fitted refusal,
        whose remedy is the in-sample fit.

        The data checks of baseline strata,
        :func:`~cleverly.estimators._strata.check_stratified_targets`, run next: a stratum
        that lacks a treatment arm beside an incremental target, and a working model that
        is singular inside a stratum.

        The natural-course contract runs next, because it resolves the target list the
        arm-indexed missing-outcome contract then reads.  The refusal of every other
        cross-fitted missing-outcome target follows them.  All three name a narrower
        surface than the outcome-scale rule below, so each keeps its own sentence and the
        general rule catches what is left.

        A subclass extends this method with the refusals of its own design, such as the
        collaborative refusal of clustered data, and calls it last.  Every refusal here and
        in an override reads only the configuration and ``data``, and none fits a learner.
        :meth:`_refit_configuration_refusal` runs the enclosing
        :meth:`_preflight_fit_configuration`, including this override chain and its later
        guards, to answer whether a refit runs.
        """
        self._refuse_fold_policy()
        self._refuse_undeclared_functions()
        if self.msm is not None:
            refuse_continuous_msm_mechanisms(data, subject="An MSM (msm=)", missingness="delta=")
        check_stratified_targets(data, incremental=bool(self.incremental), msm=self.msm)
        estimands = self._resolve_natural_course_contract(data)
        self._resolve_arm_indexed_missing_contract(data, estimands)
        self._refuse_cross_fitted_missing_off_contract(data, estimands)
        self._refuse_unbounded_cross_fitted_scale(data)
        return estimands

    def _fold_policy_refusal(self) -> str | None:
        """The sentence a fit under a fold policy this version refuses raises, or ``None``.

        Returns
        -------
        str or None
            What :meth:`_cross_fit_policy_reason` returns, then the sentence that names a
            copied or modified estimator, or ``None`` when the policy runs.
        """
        reason = self._cross_fit_policy_reason()
        if reason is None:
            return None
        return (
            f"{reason}. This fit was configured under a fold policy this version "
            "refuses, which a copied or modified estimator can still carry"
        )

    def _refuse_fold_policy(self) -> None:
        """Raise the sentence of :meth:`_fold_policy_refusal`, when the policy is refused.

        Two call sites run it first: ``_resolve_estimands_for_data``, before any learner of
        a fit or a refit, and :func:`cleverly.variable_importance`, before it asks the
        status of a copied or modified estimator.

        Raises
        ------
        CapabilityError
            If this version refuses the declared fold policy.
        """
        reason = self._fold_policy_refusal()
        if reason is not None:
            raise CapabilityError(reason)

    def _refit_configuration_refusal(self, data: CausalData) -> str | None:
        """Why a refit of this configuration on ``data`` is refused before any learner.

        The ``refit_nuisances`` slot of :func:`~cleverly.assessment.replayability` reads
        this, so the slot agrees with :meth:`refit`.  It runs the chain that :meth:`refit`
        runs before any learner: :meth:`_configured_for_refit`, and then
        :meth:`_preflight_fit_configuration`, which also calls
        :meth:`_resolve_estimands_for_data` with every subclass override.  One chain
        serves both, so a refusal that a subclass adds reaches the slot without a second
        list.  The declaration check in that chain has its own replay code, and
        :func:`~cleverly.assessment.replayability` asks it first.

        The chain reads the configuration and ``data`` and fits nothing, so each
        ``ValueError`` it raises is a refusal.  That includes
        :class:`~cleverly.exceptions.CapabilityError` and
        :class:`~cleverly.exceptions.DataError`, which are both ``ValueError``, and a
        malformed setting that a copied estimator can carry, such as ``q_bounds`` on a
        binary outcome or a malformed ``g_bounds`` pair.  Any other exception, a
        ``NotImplementedError`` included, is a defect and propagates.

        Parameters
        ----------
        data : CausalData
            The prepared data the refit would fit, which is the result's own data.

        Returns
        -------
        str or None
            The sentence the refit raises, or ``None`` when no check refuses.
        """
        try:
            self._configured_for_refit(data)._preflight_fit_configuration(data)
        except ValueError as error:
            return str(error)
        return None

    def _refuse_cross_fitted_missing_off_contract(
        self, data: CausalData, estimands: tuple[str, ...]
    ) -> None:
        """Refuse a cross-fitted missing-outcome fit that neither audited contract covers.

        The two contracts are the natural-course mean and the arm-indexed means and
        contrasts. A shift, incremental, regime, or MSM axis, or a declared intermediate,
        puts the fit outside both. Without this refusal such a fit ran and reported an
        interval that no audit read a source for, and a ``Static`` regime or a saturated
        MSM reproduced the arm-indexed fit while it escaped that contract's refusals of
        ``repeats``, fold targeting, ``cv_evaluation``, the linear fluctuation, and
        ``id=``. F21 in ``docs/roadmap.md`` holds the missing results.

        Ordinary TMLE alone reaches this surface. :class:`~cleverly.DRTMLE` refuses the
        four axes at construction and ``intermediate=`` before its nuisances, and
        :class:`~cleverly.CTMLE` refuses every axis-indexed estimand and ``intermediate=``
        before this method runs. The natural-course mean, alone or beside arm targets, is
        arm-axis only, and its contract refuses ``intermediate=``, so it cannot reach the
        check below.
        """
        if not (self._assessment_method == "tmle" and self.cross_fit and data.has_missing_outcome):
            return
        if self._axis == "arm" and not data.has_intermediate:
            return
        parts = []
        if self._axis != "arm":
            parts.append(_OFF_CONTRACT_AXIS_NAMES[self._axis])
        if data.has_intermediate:
            parts.append("controlled-direct-effect")
        raise CapabilityError(
            _CROSS_FITTED_MISSING_CONTRACTS
            + f"no audited result covers {' and '.join(parts)} targets under cross-fitting "
            "with missing outcomes (F21 in docs/roadmap.md). To estimate them, "
            + _IN_SAMPLE_ARM_INDEXED_REMEDY
        )

    def _outcome_scale_refusal(self, data: CausalData) -> str | None:
        """The sentence a cross-fitted fit on an undeclared outcome scale raises, or ``None``.

        :meth:`_scaler` maps a continuous outcome onto ``[0, 1]`` before ``Qbar`` is
        fitted, and with ``q_bounds=None`` it takes the endpoints from the observed
        outcomes of the whole sample.  Every fold's nuisance is then fitted on a scale the
        rows it predicts helped choose, so the split no longer separates what a fold saw
        from what it is scored on.  Declaring the support is the remedy that keeps the
        separation, and fitting in sample is the one that needs no support.
        :meth:`_refuse_unbounded_cross_fitted_scale` raises the sentence.

        Parameters
        ----------
        data : CausalData
            The prepared data. Its outcome family and outcome name are read.

        Returns
        -------
        str or None
            The refusal, or ``None`` when the fit is in sample, the outcome is binary, or
            ``q_bounds`` is declared.
        """
        if not self.cross_fit or data.family == "binomial" or self.q_bounds is not None:
            return None
        return (
            f"a cross-fitted fit of a continuous outcome ({data.outcome_name}, "
            f"family={data.family!r}) needs a declared q_bounds. With q_bounds=None the "
            f"outcome scale is taken {HELD_OUT_SCALE}, "
            "so each fold's nuisance is fitted on a scale the rows it predicts helped set, "
            f"and no shipped result covers that scale {_SCALE_AUDIT}. Declare the "
            "known outcome support "
            + (f"{_SCALE_SUPPORT}." if self.learned_rule is not None else _SCALE_REMEDY)
        )

    def _refuse_unbounded_cross_fitted_scale(self, data: CausalData) -> None:
        """Raise the sentence of :meth:`_outcome_scale_refusal`, when the scale is refused.

        Two call sites run it: ``_resolve_estimands_for_data``, before any learner of a fit
        or a refit, and :func:`cleverly.variable_importance`, on each candidate's prepared
        data before it asks the status.

        Parameters
        ----------
        data : CausalData
            The prepared data.

        Raises
        ------
        CapabilityError
            If this version refuses the outcome scale of a cross-fitted fit on ``data``.
        """
        reason = self._outcome_scale_refusal(data)
        if reason is not None:
            raise CapabilityError(reason)

    def _on_arm_indexed_stacked_surface(self, data: CausalData, estimands: tuple[str, ...]) -> bool:
        """Whether the arm-indexed stacked MAR contract governs this ordinary TMLE fit.

        A joint natural-course fit is on this surface and on the natural-course contract,
        so it meets both contracts' refusals and preflights. :class:`~cleverly.DRTMLE`
        raises its own cross-fitted missing-outcome refusal.
        """
        return (
            _is_arm_indexed_missing_crossfit(
                data, estimands, cross_fit=self.cross_fit, axis=self._axis
            )
            and self._assessment_method != "drtmle"
        )

    def _resolve_arm_indexed_missing_contract(
        self, data: CausalData, estimands: tuple[str, ...]
    ) -> None:
        """Refuse every cross-fitted arm-indexed missing-outcome fit outside the contract.

        The contract is the arm-indexed stacked contract in ``point-treatment-tmle.md``. Each
        refusal runs before fold generation and names the missing result, the engine
        keyword, and the public spelling. The checks run in a fixed order, so a fit that
        breaks several rules receives the first one.
        """
        if not self._on_arm_indexed_stacked_surface(data, estimands):
            return

        def refuse(reason: str) -> None:
            raise CapabilityError(_ARM_INDEXED_CONTRACT + reason)

        if self._assessment_method == "collaborative_tmle":
            refuse(
                "C-TMLE (CTMLE, or CollaborativeTMLEMethod) has no audited selection and "
                "inference result with cross-fitted arm-indexed missing outcomes. Use the "
                "ordinary TMLE (TMLE, or TMLEMethod)"
            )
        if self.fluctuation != "logistic":
            refuse(
                "no audited result covers the linear fluctuation. Set fluctuation='logistic' "
                "(Targeting(fluctuation='logistic'))"
            )
        if self.targeting != "iterative":
            refuse(
                "no audited result covers the one-step update. Set targeting='iterative' "
                "(Targeting(algorithm='iterative'))"
            )
        if self.target_weights:
            refuse(
                "no audited result covers the weighted fluctuation. Set target_weights=False "
                "(Targeting(target_weights=False))"
            )
        if self.cv_evaluation:
            refuse(
                "the fold-evaluated construction has no review of its fold plug-in and "
                "variance law. Set cv_evaluation=False (CrossFitting(fold_evaluation=False))"
            )
        conditional = [name for name in estimands if name in ("att", "atc")]
        if conditional:
            # The stacked natural course needs a binary outcome, so its names are offered
            # only where its contract can run.
            admitted = [
                name
                for name in resolve_estimands("all", data.family, data.n_arms)
                if name in _ARM_INDEXED_ADMITTED
                and (data.family == "binomial" or name not in _NATURAL_COURSE_NAMES)
            ]
            refuse(
                f"no audited result covers {conditional}. The default estimand list and "
                "estimands='all' include att and atc, and the fit drops no requested "
                f"estimand silently. Request estimands from {admitted}"
            )
        if data.family != "binomial" and self.q_bounds is None:
            refuse(
                f"a continuous outcome with q_bounds=None takes its scale {HELD_OUT_SCALE}, "
                f"and no reviewed result covers that scale {_SCALE_AUDIT}. Declare q_bounds "
                f"equal to the known outcome support {_SCALE_REMEDY}"
            )
        if self.split_plan is not None:
            refuse(
                "no audit covers the balance and weighting of a supplied split plan. Leave "
                "split_plan=None (CrossFitting(split_plan=None)) for package-generated folds"
            )
        if self.repeats != 1:
            refuse(
                "no direct interval result covers the repeated-split report after CV-TMLE "
                "targeting (F21). Set repeats=1 (CrossFitting(repeats=1))"
            )
        if self.targeting_scheme != "pooled":
            refuse(
                "no direct interval result covers fold-specific targeting (F21). Set "
                "targeting_scheme='pooled' (CrossFitting(targeting_scheme='pooled'))"
            )
        if data.weights_name is not None or data.is_weighted:
            refuse(
                "the contract covers unweighted iid rows. Drop weights= from fit "
                "(PointTreatment(weights=None))"
            )
        if data.cluster is not None:
            refuse(
                "the contract covers unweighted iid rows. Drop id= from fit "
                "(PointTreatment(cluster=None))"
            )
        if data.has_strata:
            refuse(
                "no audited result covers baseline strata. Drop strata= from fit "
                "(PointTreatment(strata=()))"
            )
        if self.n_bootstrap:
            refuse(
                "no audited result covers bootstrap inference. Set n_bootstrap=0 "
                "(Inference(n_bootstrap=0))"
            )

    def _resolve_natural_course_contract(self, data: CausalData) -> tuple[str, ...]:
        """Resolve targets and enforce the supported natural-course compositions.

        One resolver covers the scalar natural-course mean and the joint fit that stacks it
        with arm targets (``par``, ``paf``, or ``ey_obs`` beside arm means and contrasts).
        Both use the shipped natural-course fluctuation, so both meet the same refusals.
        """
        estimands = resolve_estimands(self.estimands, data.family, data.n_arms, axis=self._axis)
        if not data.has_missing_outcome:
            return estimands
        if self.estimands == "all":
            # A joint natural-course fit is a named request, so an ``all`` fit keeps the
            # shipped arm-only fluctuation set and its bit-identical estimates.
            return tuple(
                name
                for name in estimands
                if name != NATURAL_COURSE_TARGET and name not in POPULATION_INTERVENTION_TARGETS
            )
        if not _reads_natural_course(data, estimands):
            return estimands

        def refuse(reason: str) -> None:
            raise CapabilityError(
                "NaturalCourseMean, PAR and PAF with missing outcomes currently support "
                f"ordinary TMLE under their audited implementation contracts; {reason}"
            )

        if self._assessment_method != "tmle":
            reason = (
                "use ordinary TMLE; collaborative and doubly robust estimator variants "
                "need separate targeting and inference results"
            )
            if self._assessment_method == "drtmle" and not getattr(self, "guard", True):
                reason += ". guard=() is the ordinary TMLE; use TMLE"
            refuse(reason)
        if self.repeats != 1:
            refuse("set repeats=1")
        if self.cross_fit and self.targeting_scheme != "pooled":
            refuse("the cross-fitted estimator requires targeting_scheme='pooled'")
        if self.cross_fit and self.cv_evaluation:
            refuse("the cross-fitted estimator requires cv_evaluation=False")
        if self.cross_fit and self.split_plan is not None:
            refuse(
                "the cross-fitted estimator requires package-generated folds; leave split_plan=None"
            )
        if self.fluctuation != "logistic":
            refuse("set fluctuation='logistic'")
        if self.targeting != "iterative":
            refuse("set targeting='iterative'")
        if self.target_weights:
            refuse("set target_weights=False")
        if self.n_bootstrap:
            refuse(
                "set n_bootstrap=0; no audited bootstrap result covers this fit "
                "(F2 in docs/roadmap.md)"
            )
        if self.cross_fit and (data.weights_name is not None or data.is_weighted):
            refuse(
                "the stacked contract covers unweighted iid rows. Drop weights= from fit, "
                "or fit in sample"
            )
        if self.cross_fit and data.cluster is not None:
            refuse(
                "the stacked contract covers unweighted iid rows. Drop id= from fit, "
                "or fit in sample"
            )
        if data.has_intermediate:
            refuse("intermediate= is not implemented")
        if data.family != "binomial":
            if self.cross_fit:
                refuse("the cross-fitted estimator requires a binary outcome")
            if self.q_bounds is None:
                refuse("continuous outcomes require fixed, analyst-declared q_bounds")
        return estimands

    def _preflight_missing_outcome_folds(
        self,
        data: CausalData,
        estimands: tuple[str, ...],
        folds: Sequence[Folds],
    ) -> None:
        """Check every realized draw of a cross-fitted missing-outcome fit before learners.

        The natural-course mean and the arm-indexed stacked contract each have their own
        minimum content. A single-fold draw fits in sample, so neither check applies.
        """
        self._preflight_natural_course_folds(data, estimands, folds)
        if (
            self._assessment_method == "tmle"
            and self._on_arm_indexed_stacked_surface(data, estimands)
            and not any(draw.is_single for draw in folds)
        ):
            self._preflight_arm_indexed_folds(data, folds)

    def _preflight_arm_indexed_folds(self, data: CausalData, folds: Sequence[Folds]) -> None:
        """Check the arm-indexed stacked contract's minimum content before any learner fit.

        The sample needs two respondents, two nonrespondents, two respondents in each
        arm, and, for a binary outcome, two respondents with each outcome. Each training
        complement needs one respondent, one nonrespondent, one row in each arm, one
        respondent in each arm, and, for a binary outcome, both outcome classes among its
        respondents. A role whose resolved learner is a package
        :class:`~cleverly.learners.SuperLearner` with a classification task needs three
        rows in each class of its target in the sample, and two in each training
        complement, because that learner's inner split stratifies on the target. A
        sample below its minimum is refused with a remedy that does not repartition,
        because no partition can succeed. This check reads the resolved learner of each
        role and fits nothing. It cannot see a Super Learner nested inside a user
        pipeline.
        """
        subject = "cross-fitted TMLE of arm-indexed means and contrasts with missing outcomes"
        observed = np.asarray(data.observed, dtype=bool)
        treatment = np.asarray(data.treatment, dtype=float)
        arms = np.asarray(data.arm_codes, dtype=float)
        n_response = int(np.count_nonzero(observed))
        shortfalls = [
            (n_response, "respondent(s)"),
            (observed.size - n_response, "nonrespondent(s)"),
            *(
                (
                    int(np.count_nonzero(observed & (treatment == arm))),
                    f"respondent(s) in arm {data.arm_label(arm)}",
                )
                for arm in arms
            ),
        ]
        for count, kind in shortfalls:
            if count < 2:
                raise DataError(
                    f"{subject} needs at least two respondents, two nonrespondents, and two "
                    f"respondents in each arm, and the sample has {count} {kind}. Every "
                    "partition leaves some training complement short, so no fold count or "
                    f"random_state can fit the nuisances; {_IN_SAMPLE_ARM_INDEXED_REMEDY}"
                )

        # A sample minimum below which no partition can succeed. A class with c rows puts
        # c_v of them in validation fold v, so fold v's complement holds c - c_v of them,
        # and some fold holds c_v >= 1: every partition leaves a complement with at most
        # c - 1. A complement minimum of k therefore needs c >= k + 1 in the sample. That
        # is also the whole of the sample condition: the minimum is k + 1, two for the
        # binary outcome classes (k = 1) and three for a Super Learner classification role
        # (k = 2). Below it no partition succeeds, which is what lets this refusal state a
        # minimum; at or above it whether a *particular* drawn split succeeds is the
        # question the complement checks below ask, and they name no redraw.
        binary = data.family == "binomial"
        if binary:
            responses = np.asarray(data.outcome, dtype=float)[observed]
            for value in (0.0, 1.0):
                count = int(np.count_nonzero(responses == value))
                if count == 0:
                    raise DataError(
                        f"{subject} needs both outcome classes among the respondents, and "
                        f"no respondent has outcome {value:g}. No partition can supply it: "
                        "no fold count or random_state can fit the outcome regression by "
                        "cross-fitting, and an in-sample fit trains the outcome learner on "
                        "the same single class."
                    )
                if count < 2:
                    raise DataError(
                        f"{subject} needs at least two respondents with each outcome, and "
                        f"the sample has {count} respondent(s) with outcome {value:g}. Every "
                        "partition leaves some training complement without that outcome, "
                        "so no fold count or random_state can fit the outcome regression; "
                        f"{_IN_SAMPLE_ARM_INDEXED_REMEDY}"
                    )
        sample_gap = self._super_learner_sample_shortfall(data)
        if sample_gap is not None:
            role, described, count = sample_gap
            # An in-sample Super Learner splits all c rows itself, and its own
            # stratified split needs two, so that remedy holds only at c = 2.
            in_sample = f", or {_IN_SAMPLE_ARM_INDEXED_REMEDY}" if count == 2 else ""
            raise DataError(
                f"{subject} cannot fit the {role} learner: the sample holds {count} "
                f"{described}. {_SUPER_LEARNER_INNER_SPLIT_RULE.format(role=role)} Every "
                f"partition leaves some complement with at most {count - 1}, so no fold "
                f"count or random_state can fit it. Replace the {role} learner with one "
                f"that is not a package classification SuperLearner{in_sample}."
            )

        remedy = _POST_DRAW_REMEDY.format(remedy=_IN_SAMPLE_ARM_INDEXED_REMEDY)
        responding_arm = np.where(observed, treatment, -1.0)
        support: list[tuple[str, FloatArray | BoolArray, FloatArray | BoolArray]] = [
            ("response", observed, np.array([False, True])),
            ("arm", treatment, arms),
            ("responding arm", responding_arm, arms),
        ]
        if binary:
            outcome = np.where(observed, np.asarray(data.outcome, dtype=float), -1.0)
            support.append(("outcome", outcome, np.array([0.0, 1.0])))

        def describe(name: str, value: float) -> str:
            if name == "response":
                return "respondent" if value else "nonrespondent"
            if name == "arm":
                return f"row in arm {data.arm_label(value)}"
            if name == "responding arm":
                return f"respondent in arm {data.arm_label(value)}"
            return f"respondent with outcome {value:g}"

        for repeat, draw in enumerate(folds):
            gap = missing_training_support(draw, support)
            if gap is not None:
                fold, name, missing = gap
                raise DataError(
                    f"{subject} cannot fit its nuisances because repeat {repeat}, fold "
                    f"{fold}'s training complement contains no "
                    f"{describe(name, float(missing[0]))}. {remedy}"
                )

        complement_gap = self._super_learner_complement_shortfall(data, folds)
        if complement_gap is not None:
            role, repeat, fold, described, count = complement_gap
            raise DataError(
                f"{subject} cannot fit the {role} learner because repeat {repeat}, fold "
                f"{fold}'s training complement holds {count} {described}. "
                f"{_SUPER_LEARNER_INNER_SPLIT_RULE.format(role=role)} {remedy}"
            )

    def _super_learner_inner_split_roles(
        self, data: CausalData, *, fits_treatment: bool = True
    ) -> tuple[tuple[str, Task | None, BoolArray, FloatArray, Callable[[float], str]], ...]:
        """The package Super Learner roles and the targets each outer fit trains on.

        A package :class:`~cleverly.learners.SuperLearner` with a classification task
        stratifies its inner folds on the target it is fitted to. Each role is resolved
        exactly as the fit resolves it, and nothing is fitted. The outcome target is the
        scaled outcome the fit trains on. A Super Learner without a task infers it from
        each training complement, whose task may differ from the full sample's task.

        Parameters
        ----------
        data : CausalData
            The prepared data.

        Returns
        -------
        tuple of tuple
            ``(role, task, rows, target, label)`` for each package Super Learner role:
            its declared task, fitting rows, target on every row, and class description.
        """
        observed = np.asarray(data.observed, dtype=bool)
        everyone = np.ones(observed.size, dtype=bool)
        outcome_task: Task = "classification" if data.family == "binomial" else "regression"
        scaler = self._scaler(data)
        roles: list[tuple[str, Learner, BoolArray, FloatArray, Callable[[float], str]]] = [
            (
                "outcome",
                self._resolve_learner(self.outcome_learner, task=outcome_task),
                observed,
                scaler.scale(data.outcome),
                lambda value: f"respondent(s) with outcome {scaler.unscale_level(value):g}",
            ),
        ]
        if fits_treatment and not data.is_continuous_treatment:
            roles.append(
                (
                    "treatment",
                    self._resolve_learner(self.treatment_learner, task="classification"),
                    everyone,
                    np.asarray(data.treatment, dtype=float),
                    lambda value: f"row(s) in arm {data.arm_label(value)}",
                )
            )
        roles.append(
            (
                "response",
                self._resolve_learner(
                    self.missingness_learner,
                    task="classification",
                    fallback=self.treatment_learner,
                ),
                everyone,
                observed.astype(float),
                lambda value: "respondent(s)" if value else "nonrespondent(s)",
            )
        )
        return tuple(
            (role, learner.task, rows, target, label)
            for role, learner, rows, target, label in roles
            if isinstance(learner, SuperLearner)
        )

    def _super_learner_sample_shortfall(
        self, data: CausalData, *, fits_treatment: bool = True
    ) -> tuple[str, str, int] | None:
        """The first Super Learner class the whole sample holds too few rows of.

        A class with ``c`` rows leaves at most ``c - 1`` in some training complement,
        whatever the partition, and the inner stratified split needs two. So ``c <= 2`` is
        a fact about the sample rather than about the split, which is why it is asked
        before any split is drawn and why its refusal may state a minimum.

        Parameters
        ----------
        data : CausalData
            The prepared data.

        Returns
        -------
        tuple or None
            ``(role, described class, count)`` for the first shortfall in role order, or
            ``None``.
        """
        for role, task, rows, target, label in self._super_learner_inner_split_roles(
            data, fits_treatment=fits_treatment
        ):
            if (task or infer_task(target[rows])) != "classification":
                continue
            for value in np.unique(target[rows]):
                count = int(np.count_nonzero(target[rows] == value))
                if count <= 2:
                    return role, label(float(value)), count
        return None

    def _super_learner_complement_shortfall(
        self, data: CausalData, folds: Sequence[Folds], *, fits_treatment: bool = True
    ) -> tuple[str, int, int, str, int] | None:
        """The first training complement too thin for a Super Learner's inner split.

        Parameters
        ----------
        data : CausalData
            The prepared data.
        folds : sequence of Folds
            One realized draw per repeat.

        Returns
        -------
        tuple or None
            ``(role, repeat, fold, described class, count)`` for the first shortfall, or
            ``None`` when every complement carries two rows of every class.
        """
        size = int(np.asarray(data.observed, dtype=bool).size)
        for role, task, rows, target, label in self._super_learner_inner_split_roles(
            data, fits_treatment=fits_treatment
        ):
            sample_classes = np.unique(target[rows])
            sample_classifies = (task or infer_task(target[rows])) == "classification"
            for repeat, draw in enumerate(folds):
                for fold, (train, _) in enumerate(draw):
                    kept = np.zeros(size, dtype=bool)
                    kept[train] = True
                    values = target[kept & rows]
                    if (task or infer_task(values)) != "classification":
                        continue
                    # A task-free learner can switch from regression on the sample to
                    # classification on a complement that loses an intermediate level.
                    # In that case only the complement's classes enter its inner split.
                    classes = sample_classes if sample_classifies else np.unique(values)
                    for value in classes:
                        count = int(np.count_nonzero(values == value))
                        if count < 2:
                            return role, repeat, fold, label(float(value)), count
        return None

    def _preflight_training_support(
        self,
        data: CausalData,
        estimands: tuple[str, ...],
        folds: Sequence[Folds],
    ) -> None:
        """Check what every generated split owes a discrete-treatment fit, before learners.

        Unstratified folds are drawn from the seed alone, so nothing in the draw promises
        that each training complement carries every arm, both outcome classes of a binary
        outcome, or two rows of every class a package Super Learner stratifies its inner
        split on. Stratification used to buy the first of those and ``resolve_n_folds``
        capped the fold count to keep it. Neither is available to a split that must not
        read the treatment, so the property is *checked* on the realized draw instead, and
        checked before the first learner rather than reported from inside one.

        Three questions in one pass, in the order a reader can act on:

        1. the sample minimum, which no partition can repair: each arm, and each class of
           a binary outcome, has to appear in two independent units, rows when the data
           are iid and *clusters* when ``id=`` declared them, because a cluster is atomic
           and a split moves whole clusters;
        2. the support of each training complement, for the arms, for the rows whose
           outcome was observed, and for the binary outcome classes among them;
        3. the two-rows-per-class rule of a package classification Super Learner.

        The first is a statement about the sample, so its refusal states the minimum. The
        other two are statements about one drawn split, so they name no redraw.

        A continuous dose has no arms, so its check still covers outcome and response
        support but omits treatment-arm support. What a density fit needs from a fold is
        chosen inside that fold. A single-fold draw is skipped because it trains on every
        row.
        """
        if any(draw.is_single for draw in folds):
            return
        # The arm-indexed stacked contract and the natural-course mean each run their own
        # preflight and name their own contract, so this general one does not follow them
        # with a wider sentence about the same draw. What the two cover differs, and
        # neither is this check with a different name. The stacked contract asks the arm,
        # responding-arm, response and outcome-class questions itself, and states its own
        # sample minimums. The natural-course preflight asks about response and outcome
        # support without treatment-arm questions: that mean fits no treatment mechanism.
        if _is_natural_course(data, estimands) or self._on_arm_indexed_stacked_surface(
            data, estimands
        ):
            return
        names = {"collaborative_tmle": "C-TMLE", "drtmle": "DR-TMLE"}
        self._check_training_support(
            data,
            folds,
            subject=f"cross-fitted {names.get(self._assessment_method, 'TMLE')}",
            # The default remedy fits in sample, and the learned-rule value has none.
            **({"remedy": LEARNED_RULE_REMEDY} if self.learned_rule is not None else {}),
        )

    def _check_training_support(
        self,
        data: CausalData,
        folds: Sequence[Folds],
        *,
        subject: str,
        where: str = "repeat {repeat}, fold {fold}",
        remedy: str = _IN_SAMPLE_ARM_INDEXED_REMEDY,
    ) -> None:
        """Ask one realized partition the three questions above, under a caller's name.

        Separate from :meth:`_preflight_training_support` because a collaborative fit has a
        second partition to ask them of. Its selection folds are drawn from the same seed
        and read the data no more than the outer folds do, and a selection fold whose
        training rows lack an arm makes the candidate's propensity unfittable inside the
        search rather than in the outer loop.

        Parameters
        ----------
        data : CausalData
            The rows the partition labels.
        folds : sequence of Folds
            One realized partition per repeat.
        subject : str
            What the refusal calls the fit, as the reader would name it.
        where : str
            What the refusal calls the fold that failed, formatted with ``repeat`` and
            ``fold``.
        remedy : str
            The way out that the refusal names. The default fits in sample, which is true
            of an outer split and not of a split that an in-sample fit draws itself.
        """
        treatment = np.asarray(data.treatment, dtype=float)
        arms = np.asarray(data.arm_codes, dtype=float)
        observed = np.asarray(data.observed, dtype=bool)
        unit_of, unit = _independent_units(data)
        for arm in arms:
            count = int(np.unique(unit_of[treatment == arm]).size)
            if count < 2:
                raise DataError(
                    f"{subject} needs each treatment arm in at least two independent "
                    f"units, and arm {data.arm_label(arm)} appears in {count} {unit}(s). "
                    f"A split moves whole {unit}s, so every partition leaves some training "
                    "complement without that arm and no fold count or seed can fit the "
                    f"treatment mechanism; {remedy}"
                )
        if data.family == "binomial":
            # The sample minimum the outcome classes owe, stated for the same reason the
            # arm minimum above is: a split moves whole units, so one unit holding a class
            # leaves the complement of its own fold without it, whatever the fold count
            # and whatever the seed. The remedy is the only one that can work. Fitting in
            # sample is not offered here, because a package classification SuperLearner
            # splits the same rows again and its inner split needs two of each class too.
            outcome_values = np.asarray(data.outcome, dtype=float)
            for value in (0.0, 1.0):
                count = int(np.unique(unit_of[observed & (outcome_values == value)]).size)
                if count < 2:
                    raise DataError(
                        f"{subject} needs each outcome class in at least two independent "
                        f"units with an observed outcome, and outcome {value:g} appears in "
                        f"{count} {unit}(s). A split moves whole {unit}s, so every partition "
                        "leaves some training complement without that class and no fold "
                        "count or seed can fit the outcome regression; collect more "
                        f"observations with outcome {value:g}."
                    )
        support: list[tuple[str, FloatArray | BoolArray, FloatArray | BoolArray]] = [
            # Every family, not the binomial one alone. The outcome regression trains on
            # the rows whose outcome was observed, on whatever scale they are on, so a
            # complement holding none of them cannot fit it. Under no missingness this
            # asks nothing, because every row is then a respondent.
            ("response", observed, np.array([True])),
        ]
        if not data.is_continuous_treatment:
            support.insert(0, ("arm", treatment, arms))
        if data.family == "binomial":
            outcome = np.where(observed, np.asarray(data.outcome, dtype=float), -1.0)
            support.append(("outcome", outcome, np.array([0.0, 1.0])))
        post_draw = _POST_DRAW_REMEDY.format(remedy=remedy)
        for repeat, draw in enumerate(folds):
            gap = missing_training_support(draw, support)
            if gap is not None:
                fold, name, missing = gap
                value = float(missing[0])
                if name == "arm":
                    described = f"row in arm {data.arm_label(value)}"
                elif name == "response":
                    described = "row with an observed outcome"
                else:
                    described = f"observed outcome {value:g}"
                raise DataError(
                    f"{subject} cannot fit its nuisances because "
                    f"{where.format(repeat=repeat, fold=fold)}'s training complement "
                    f"contains no {described}. {post_draw}"
                )
        complement_gap = self._super_learner_complement_shortfall(data, folds)
        if complement_gap is not None:
            role, repeat, fold, described, count = complement_gap
            raise DataError(
                f"{subject} cannot fit the {role} learner because "
                f"{where.format(repeat=repeat, fold=fold)}'s training complement holds "
                f"{count} {described}. {_SUPER_LEARNER_INNER_SPLIT_RULE.format(role=role)} "
                f"{post_draw}"
            )

    def _preflight_natural_course_folds(
        self,
        data: CausalData,
        estimands: tuple[str, ...],
        folds: Sequence[Folds],
    ) -> None:
        """Check natural-course response and outcome support before nuisance fitting.

        The scalar mean and the joint fit both run it. A joint fit also runs the
        arm-indexed preflight, which asks the arm questions. A scalar fit with three or
        more arms asks the per-arm respondent question here: the outcome regression reads
        the realised arm, and a complement without respondents in one arm extrapolates
        that arm's held-out predictions. A two-arm scalar fit keeps the shipped checks.
        """
        if not _reads_natural_course(data, estimands) or any(draw.is_single for draw in folds):
            return
        observed = np.asarray(data.observed, dtype=bool)
        n_response = int(np.count_nonzero(observed))
        if n_response < 2 or observed.size - n_response < 2:
            kind = "respondent" if n_response < 2 else "nonrespondent"
            count = n_response if n_response < 2 else observed.size - n_response
            raise DataError(
                "cross-fitted NaturalCourseMean, PAR or PAF needs at least two of each "
                "response kind, "
                f"and the sample has {count} {kind}(s). Every partition leaves some training "
                "complement without one, so no fold count or random_state can fit the "
                f"response and outcome nuisances; {_IN_SAMPLE_NATURAL_COURSE_REMEDY}"
            )
        outcome = np.asarray(data.outcome, dtype=float)
        if data.family == "binomial":
            for value in (0.0, 1.0):
                count = int(np.count_nonzero(observed & (outcome == value)))
                if count < 2:
                    raise DataError(
                        "cross-fitted NaturalCourseMean, PAR or PAF needs at least two respondents "
                        f"with each outcome, and the sample has {count} respondent(s) "
                        f"with outcome {value:g}. Every partition leaves some training "
                        "complement without that outcome, so no fold count or random_state "
                        f"can fit the outcome regression; {_IN_SAMPLE_NATURAL_COURSE_REMEDY}"
                    )
        per_arm = _is_natural_course(data, estimands) and data.n_arms >= 3
        treatment = np.asarray(data.treatment, dtype=float)
        arms = np.asarray(data.arm_codes, dtype=float)
        if per_arm:
            for arm in arms:
                count = int(np.count_nonzero(observed & (treatment == arm)))
                if count < 2:
                    raise DataError(
                        "cross-fitted NaturalCourseMean with three or more arms needs at "
                        "least two respondents in each arm, and the sample has "
                        f"{count} respondent(s) in arm {data.arm_label(arm)}. Every "
                        "partition leaves some training complement without one, so no fold "
                        "count or random_state can fit the outcome regression; "
                        f"{_IN_SAMPLE_NATURAL_COURSE_REMEDY}"
                    )
        sample_gap = self._super_learner_sample_shortfall(data, fits_treatment=False)
        if sample_gap is not None:
            role, described, count = sample_gap
            raise DataError(
                "cross-fitted NaturalCourseMean, PAR or PAF cannot fit the "
                f"{role} learner: the sample "
                f"holds {count} {described}. "
                f"{_SUPER_LEARNER_INNER_SPLIT_RULE.format(role=role)} Collect more "
                "observations or use a learner without that inner split."
            )
        support: list[tuple[str, FloatArray | BoolArray, FloatArray | BoolArray]] = [
            ("response", observed, np.array([False, True])),
        ]
        if data.family == "binomial":
            support.append(("outcome", np.where(observed, outcome, -1.0), np.array([0.0, 1.0])))
        if per_arm:
            support.append(("responding arm", np.where(observed, treatment, -1.0), arms))
        for repeat, draw in enumerate(folds):
            gap = missing_training_support(draw, support)
            if gap is not None:
                fold, name, missing = gap
                if name == "response":
                    described = "respondent" if bool(missing[0]) else "nonrespondent"
                elif name == "responding arm":
                    described = f"respondent in arm {data.arm_label(float(missing[0]))}"
                else:
                    described = f"respondent with outcome {float(missing[0]):g}"
                raise DataError(
                    "cross-fitted NaturalCourseMean, PAR or PAF cannot fit its response "
                    "and outcome "
                    f"nuisances because repeat {repeat}, fold {fold}'s training complement "
                    f"contains no {described}. "
                    + _POST_DRAW_REMEDY.format(remedy=_IN_SAMPLE_NATURAL_COURSE_REMEDY)
                )
        complement_gap = self._super_learner_complement_shortfall(data, folds, fits_treatment=False)
        if complement_gap is not None:
            role, repeat, fold, described, count = complement_gap
            raise DataError(
                "cross-fitted NaturalCourseMean, PAR or PAF cannot fit the "
                f"{role} learner because repeat {repeat}, fold {fold}'s training "
                f"complement holds {count} {described}. "
                f"{_SUPER_LEARNER_INNER_SPLIT_RULE.format(role=role)} "
                + _POST_DRAW_REMEDY.format(remedy=_IN_SAMPLE_NATURAL_COURSE_REMEDY)
            )

    @property
    def _axis(self) -> ParameterAxis:
        """What this fit's parameters are indexed by, from which keyword was passed.

        Read off the declaration rather than off the data, so that asking a continuous
        fit for ``ate`` is refused by name instead of resolving to a report the treatment
        cannot support.  ``_validate_settings`` has already refused the keywords in
        combination, so at most one branch can be taken.
        """
        if self.learned_rule is not None:
            return "learned_rule"
        if self.msm is not None:
            return "msm"
        if self.policies:
            return "rr_tilt" if _all_tilts(self.policies) else "policy"
        if self.incremental:
            return "ipsi"
        if self.interventions:
            return "regime"
        return "arm"

    def _check_incremental(self, data: CausalData) -> None:
        """Refuse what an incremental fit has no derivation for.

        An unsupported composition is refused by name; the arm-indexed estimands support
        both of these and are the thing to reach for.

        ``delta=`` used to be refused here too, on the grounds that a further mechanism in
        the outcome half of the covariate would be a different derivation and no oracle law
        covered it.  Both halves of that were wrong.  ``tests/discrete_law_mar.py`` is such
        a law, and taken to it the derivation *is* the same one with an extra factor:
        :math:`\\pi(A, W)` divides the outcome-side covariate and Kennedy's mechanism term
        is untouched, because :math:`q_\\delta` is a functional of :math:`P(A \\mid W)` and
        both :math:`A` and :math:`W` are recorded whatever happens to :math:`Y`.  What does
        change is the *guarantee*: see ``tests/unit/test_remainder_ipsi_mar.py``.
        """
        if not self.incremental:
            return
        refuse_multi_arm_tilt(data)
        if data.has_intermediate:
            raise CapabilityError(
                "incremental= and intermediate= are not combined. A controlled direct "
                "effect under a tilt of the treatment mechanism is a parameter this "
                "package has not written down, and reporting one would mean guessing at "
                "its influence function. "
                "docs/roadmap.md F6 tracks it."
            )

    def _check_policies(self, data: CausalData) -> None:
        """Refuse a policy the treatment cannot carry, and a dose with no policy declared.

        Both directions matter.  A policy reads the treatment that a unit received,
        :math:`d(A, W)`.  On a continuous treatment its density ratio is Equation (3) of
        Díaz, Williams, Hoffman and Schenck (2023); on a categorical one it is their discrete
        formula, and a policy must map levels to levels.  A risk-ratio tilt sets a binary
        treatment coded 0 and 1.  A ``Rule`` is :math:`d(W)`, a different policy.  A
        continuous treatment with no ``policies=`` has no estimand at all, since every
        registered arm-indexed target names a level it has none of.
        """
        if self.policies and any(isinstance(item, RiskRatioTilt) for item in self.policies):
            levels = [] if data.is_continuous_treatment else list(data.treatment_levels)
            if data.is_continuous_treatment or sorted(float(level) for level in levels) != [
                0.0,
                1.0,
            ]:
                raise DataError(
                    f"RiskRatioTilt sets a binary treatment coded 0 and 1, but "
                    f"{data.treatment_name} "
                    + ("is continuous" if data.is_continuous_treatment else f"has levels {levels}")
                    + ". It is lmtp's ipsi(), which keeps a treated unit treated with "
                    "probability delta and otherwise sets it to 0"
                )
        if data.is_continuous_treatment and not self.policies and self.msm is None:
            # The suggested shift meets the F21 refusal of a cross-fitted fit with missing
            # outcomes, so that fit is told which fit estimates the natural course.  The
            # condition is the one ``_refuse_cross_fitted_missing_off_contract`` refuses on.
            in_sample = (
                " A cross-fitted fit with missing outcomes, declared with "
                f"{MISSING_OUTCOME_DECLARATION}, refuses a shift target (F21 in "
                "docs/roadmap.md). To estimate the natural course, "
                f"{_IN_SAMPLE_ARM_INDEXED_REMEDY}."
                if self._assessment_method == "tmle" and self.cross_fit and data.has_missing_outcome
                else ""
            )
            raise DataError(
                f"{data.treatment_name} was declared continuous, so it has no arms and "
                "none of the arm-indexed estimands name a parameter it has. Say which "
                "doses to compare with policies=[Shift(delta, cap=...), ...]; "
                "Shift(0.0, cap=None) is the natural course, whose mean is E[Y]. Or, "
                "with every outcome observed and no intermediate=, declare an MSM with a "
                f"dose integration grid.{in_sample}"
            )

    def _reference_arm(
        self,
        data: CausalData,
        regimes: RegimeSet | None = None,
        policies: PolicySet | None = None,
    ) -> float:
        """The arm -- or regime, or shift -- code every contrast is taken against.

        On a regime or shift fit the contrasts are between *regimes* (or *shifts*), so
        ``reference=`` names one of them and the code returned indexes
        :class:`~cleverly.interventions.RegimeSet` or
        :class:`~cleverly.interventions.PolicySet`.  :meth:`_regimes` and
        :func:`~cleverly.estimators._nuisance.fit_nuisances` have already validated the
        name against what was declared, which is why this simply reads the code back.

        ``reference=None`` uses the lowest arm, which for a binary treatment is the
        control and so leaves ``ate`` meaning exactly what it always did.  Otherwise the
        value is matched against the treatment's *own* levels -- pass ``"low"``, not
        ``1.0`` -- because the codes are an encoding detail and the labels are what the
        caller wrote down.  Levels sort in their natural order, which for strings is
        alphabetical, so the default reference on ``{"high", "low", "medium"}`` is
        ``"high"``; this argument is how to say otherwise.
        """
        if self.msm is not None:
            # Coefficients have no contrast reference; the config field is retained for
            # the shared result schema and is ignored on the MSM parameter axis.
            return 0.0
        if policies is not None:
            return policies.reference
        if regimes is not None:
            return regimes.reference
        if self.policies:
            # Resolved from the shift *names*, for the reason the regime branch gives.
            return self._reference_shift()
        if self.incremental:
            # Resolved from the tilt *names*, for the reason the regime branch gives.
            return self._reference_incremental()
        if self.interventions:
            # Resolved from the regime *names* rather than from an evaluated RegimeSet,
            # so the config -- built before any nuisance is fitted -- can record it, and
            # so a mistyped reference fails before the fitting rather than after it.
            return self._reference_regime()
        if self.reference is None:
            return data.arm_codes[0]
        labels = list(data.treatment_levels)
        for code, label in zip(data.arm_codes, labels, strict=True):
            if label == self.reference or code == self.reference:
                return code
        raise DataError(
            f"reference={self.reference!r} is not a level of {data.treatment_name}; its "
            f"levels are {labels}"
        )

    def _folds(self, data: CausalData, seed: int | None = None) -> Folds:
        """One draw of the split, from ``seed`` or from the plan's own.

        ``seed=None`` means "the plan's".  :meth:`_repeat_draws` always passes a concrete
        seed, because it resolves an undeclared seed before it draws.  An unstratified
        draw goes through :func:`~cleverly.learners.random_partition` inside
        :func:`make_folds`, so its :attr:`Folds.origin` records the seed.
        """
        plan = self.crossfit_plan(data)
        if not plan.cross_fit:
            return Folds.single(data.n)
        return make_folds(
            data.n,
            plan.n_folds,
            stratify=self._fold_strata(data),
            cluster=data.cluster,
            random_state=plan.random_state if seed is None else seed,
        )

    def _repeat_draws(
        self, data: CausalData, estimands: tuple[str, ...]
    ) -> tuple[tuple[Folds, int], ...]:
        """Realize and validate all outer draws before fitting any nuisance model.

        Every realized draw also passes :meth:`_preflight_missing_outcome_folds` and
        :meth:`_preflight_training_support`, so each fit path that draws folds checks what
        its nuisances need from every training complement before its first learner fit.

        The supplied path first asks :meth:`SplitPlan.verify` whether the plan's generator
        record draws exactly these labels on these rows.  It then asks
        :meth:`SplitPlan.validate` whether the labels can serve these rows, and asks
        nothing else.  It does *not* compare the plan's fold count
        against :func:`resolve_n_folds`, which answers "how many folds could a generated
        stratified split make here" -- a question about a split nobody is generating.  A
        usable plan may hold more folds than that: the rarest stratum has to reach every
        training complement, not to appear once in every fold, and ``validate`` checks
        that property directly.  Declaration time rules out the one direction no cap can
        produce, more folds than declared, which ``SplitPlan._policy_refusal`` does.

        Every draw gets a concrete seed, the single-fold draw of ``cross_fit=False``
        included.  Under ``random_state=None``, :meth:`CrossFitPlan.seeds` returns ``None``
        per repeat, and this method replaces each one with a fresh seed from
        operating-system entropy.  The same seed then reaches the folds, the learners and
        the C-TMLE selection folds, and :attr:`RepeatFit.seed` records it on the result.
        A caller-declared seed passes through unchanged.
        """
        seeds = tuple(
            _fresh_seed() if seed is None else seed for seed in self.crossfit_plan(data).seeds()
        )
        supplied = self.split_plan
        if supplied is None:
            folds = tuple(self._folds(data, seed) for seed in seeds)
        else:
            # The record check comes first: labels the recorded generator does not draw
            # are refused whatever support they would give.
            supplied.verify(n=data.n, cluster=data.cluster)
            stratify = self._fold_strata(data)
            folds = supplied.validate(
                n=data.n,
                cluster=data.cluster,
                treatment=None if data.is_continuous_treatment else data.treatment,
                stratify=stratify,
                source_fingerprint=data_fingerprint(data),
            )
        self._preflight_cluster_validation_folds(data, folds)
        self._preflight_missing_outcome_folds(data, estimands, folds)
        self._preflight_training_support(data, estimands, folds)
        return tuple(zip(folds, seeds, strict=True))

    def _preflight_cluster_validation_folds(self, data: CausalData, folds: Sequence[Folds]) -> None:
        """Refuse a fold-wise clustered split with a validation fold of one cluster.

        Two settings evaluate each validation fold on its own. ``cv_evaluation=True`` takes
        the centred variance of the cluster totals inside each fold
        (:func:`~cleverly.inference.cluster.cross_validated_variance`), and
        ``targeting_scheme="fold"`` builds each fold's estimate with a cluster-robust
        variance of its own rows. Both need at least two clusters in every validation fold.
        The check reads every realized or supplied draw before the first learner.
        :func:`~cleverly.learners.crossfit.random_partition` deals the clusters into
        near-equal counts, so a generated split passes when the fold count is at most half
        the cluster count.
        """
        fold_targeting = self.targeting_scheme == "fold"
        if not (self.cv_evaluation or fold_targeting) or data.cluster is None:
            return
        setting = "cv_evaluation=True" if self.cv_evaluation else 'targeting_scheme="fold"'
        stacked = (
            "the stacked report (cv_evaluation=False, CrossFitting(fold_evaluation=False))"
            if self.cv_evaluation
            else 'one pooled fluctuation (targeting_scheme="pooled")'
        )
        for draw in folds:
            if draw.is_single:
                continue
            for fold, (_, test) in enumerate(draw):
                held = int(np.unique(data.cluster[test]).size)
                if held < 2:
                    n_clusters = int(np.unique(data.cluster).size)
                    raise CapabilityError(
                        f"{setting} needs at least 2 clusters in every validation fold, "
                        "because each fold's variance compares cluster totals inside the "
                        f"fold. This split puts {n_clusters} clusters into {draw.n_folds} "
                        f"folds, and fold {fold} holds {held}. Request at most "
                        f"{n_clusters // 2} folds (CrossFitting(n_folds=...)), or use "
                        f"{stacked}."
                    )

    def _resolve_learner(
        self,
        spec: Learner | None,
        *,
        task: Task,
        fallback: Learner | None = None,
        seed: int | None = None,
    ) -> Learner:
        """Turn a learner specification into a fitted-per-fold estimator.

        ``seed`` is the draw's, under the same convention :meth:`_folds` uses: ``None``
        means "the estimator's own ``random_state``".  It reaches the Super Learner's
        *inner* split, so a repeat redraws the whole nested cross-validation rather than
        only the outer one -- which is what makes ``repeats=R`` a median over the
        randomised procedure instead of over one stage of it.
        """
        return resolve_learner(
            spec,
            task=task,
            n_folds=self.learner_folds,
            random_state=self.random_state if seed is None else seed,
            fallback=fallback,
        )

    def _fit_nuisances(
        self,
        data: CausalData,
        folds: Folds,
        scaler: OutcomeScaler,
        intermediate_value: float | None,
        extra_levels: Sequence[float] = (),
        seed: int | None = None,
        companion: CausalData | None = None,
        fit_treatment: bool = True,
    ) -> NuisanceEstimates:
        outcome_task: Task = "classification" if data.family == "binomial" else "regression"
        msm = self._msm(data)
        estimates = fit_nuisances(
            data,
            outcome_learner=self._resolve_learner(
                self.outcome_learner, task=outcome_task, seed=seed
            ),
            treatment_learner=self._resolve_learner(
                self.treatment_learner, task="classification", seed=seed
            ),
            missingness_learner=(
                self._resolve_learner(
                    self.missingness_learner,
                    task="classification",
                    fallback=self.treatment_learner,
                    seed=seed,
                )
                if data.has_missing_outcome or data.has_missing_treatment
                else None
            ),
            intermediate_learner=(
                self._resolve_learner(
                    self.intermediate_learner,
                    task="classification",
                    fallback=self.treatment_learner,
                    seed=seed,
                )
                if data.has_intermediate
                else None
            ),
            folds=folds,
            scaler=scaler,
            intermediate_value=intermediate_value,
            extra_levels=extra_levels,
            screen_treatment=self.screen_treatment,
            screen_threshold=self.screen_threshold,
            min_retain=self.min_retain,
            policies=self.policies,
            policy_reference=None if self.reference is None else str(self.reference),
            incremental=self.incremental,
            incremental_reference=None if self.reference is None else str(self.reference),
            density_bins=self.density_bins,
            policy_ratio=self.ratio,
            msm=msm,
            companion=companion,
            n_jobs=self.n_jobs,
            fit_treatment=fit_treatment,
        )
        # Evaluated once and carried with the fits, so that every reuse -- retarget, and
        # so the truncation curve, the MNAR tilt, the omitted-variable bound -- targets
        # the regimes and the working model this fit declared, without re-running the
        # caller's rules or its design.
        return replace(estimates, regimes=self._regimes(data, estimates), msm=msm)

    def _msm(self, data: CausalData) -> MSMSet | None:
        """The declared working model evaluated on ``data``, or ``None`` if none was."""
        return None if self.msm is None else MSMSet.evaluate(self.msm, data)

    def _regimes(self, data: CausalData, estimates: NuisanceEstimates) -> RegimeSet | None:
        """The regimes of this fit on ``data``, or ``None`` for an arm-indexed fit.

        A learned-rule fit reads its fold-local plug-in rule off the cross-fitted outcome
        regression in ``estimates``: row ``i`` of fold ``v`` carries the prediction of the
        model fitted on the complement of ``v``.  Every other fit evaluates its declared
        regimes on ``data`` and ignores ``estimates``.
        """
        if self.learned_rule is not None:
            return _learned_rule_regimes(estimates, self.learned_rule)
        if not self.interventions:
            return None
        reference = None if self.reference is None else str(self.reference)
        return RegimeSet.evaluate(self.interventions, data, reference=reference)

    def _reference_regime(self) -> float:
        """The regime code contrasts are taken against, from ``reference=`` and the names."""
        names = [intervention.name for intervention in self.interventions]
        if self.reference is None:
            return 0.0
        if str(self.reference) not in names:
            raise DataError(f"reference={self.reference!r} is not one of the regimes {names}")
        return float(names.index(str(self.reference)))

    def _reference_incremental(self) -> float:
        """The tilt code contrasts are taken against, from ``reference=`` and the names.

        Defaults to the first declared, which is the rule the arms, regimes and shifts
        follow.  Declaring ``Incremental(1.0)`` first is the usual way to make
        ``ate_ipsi`` read as *the effect of tilting*, since q_1 is the mechanism itself.
        """
        names = [item.name for item in self.incremental]
        if self.reference is None:
            return 0.0
        if str(self.reference) not in names:
            raise DataError(
                f"reference={self.reference!r} is not one of the incremental interventions {names}"
            )
        return float(names.index(str(self.reference)))

    def _reference_shift(self) -> float:
        """The policy code contrasts are taken against, from ``reference=`` and the names.

        Defaults to the first declared policy rather than to the natural course, which is
        the same rule the arms and regimes follow -- ``reference=`` is how to say
        otherwise, and declaring ``Shift(0.0, cap=None)`` first is the usual way to make
        ``ate_policy`` read as *the effect of shifting*.
        """
        names = [policy.name for policy in self.policies]
        if self.reference is None:
            return 0.0
        if str(self.reference) not in names:
            raise DataError(f"reference={self.reference!r} is not one of the policies {names}")
        return float(names.index(str(self.reference)))

    def _nuisances(
        self,
        data: CausalData,
        folds: Folds,
        scaler: OutcomeScaler,
        config: TMLEConfig,
        intermediate_value: float | None,
        seed: int | None = None,
    ) -> tuple[NuisanceEstimates, dict[str, Any]]:
        """The nuisance fits to target against, plus any variant-specific diagnostics.

        The extension point for TMLE variants that differ only in *which* nuisance
        estimate they hand to the targeting step -- :class:`~cleverly.CTMLE` selects a
        propensity model here and reports the selection path in the extras.

        ``seed`` is the draw's, and an override that randomises anything of its own must
        thread it through rather than reach for ``self.random_state``: under ``repeats=R``
        every stage of the split is redrawn per draw, and a stage that is not would be
        held fixed across draws that were supposed to be independent.
        """
        natural_course = _is_natural_course(data, config.estimands)
        return (
            self._fit_nuisances(
                data,
                folds,
                scaler,
                intermediate_value,
                seed=seed,
                fit_treatment=not natural_course,
            ),
            {},
        )

    @staticmethod
    def _bounds_n(data: CausalData) -> float:
        """The sample size ``g_bounds="auto"`` is resolved at.

        The Kish effective sample size, which equals ``data.n`` exactly when the weights
        are constant.  Truncating a weighted fit at the bound implied by its row count
        would leave the clever covariate freer than the information in the sample
        supports -- see :func:`~cleverly.utils.bounds.resolve_g_bounds`.
        """
        return data.effective_n

    def _config(
        self,
        data: CausalData,
        estimands: tuple[str, ...],
        scaler: OutcomeScaler,
        folds: Folds,
    ) -> TMLEConfig:
        return TMLEConfig(
            family=data.family,
            targeting_spec=self.targeting_spec(),
            targeting_scheme=(
                self.targeting_scheme if self.cross_fit and not folds.is_single else "pooled"
            ),
            cross_fit=self.cross_fit,
            cv_evaluation=self.cv_evaluation and self.cross_fit and not folds.is_single,
            n_folds=folds.n_folds,
            g_bounds=resolve_g_bounds(self.g_bounds, self._bounds_n(data), for_att=False),
            g_bounds_conditional=resolve_g_bounds(
                self.g_bounds, self._bounds_n(data), for_att=True
            ),
            auto_bounds_n=(
                data.effective_n if self.g_bounds == "auto" and data.is_weighted else None
            ),
            missingness_bound=self.nuisance_bound,
            bounded_mechanisms=tuple(
                name
                for name, present in (
                    ("P(Delta=1|A,W)", data.has_missing_outcome),
                    ("P(Delta_A=1|W)", data.has_missing_treatment),
                    ("P(Z=z|A,W)", data.has_intermediate),
                )
                if present
            ),
            fits_treatment=not _is_natural_course(data, estimands),
            continuous_treatment=data.is_continuous_treatment,
            q_bounds=None if scaler.is_identity else (scaler.lower, scaler.upper),
            screen_treatment=self.screen_treatment,
            estimands=estimands,
            alpha_sig=self.alpha_sig,
            random_state=self.random_state,
            n_bootstrap=self.n_bootstrap,
            reference_arm=self._reference_arm(data),
            parameter_axis=self._axis,
            crossfit=self.crossfit_plan(data),
        )

    def _warn_on_estimated_weights(self, data: CausalData) -> None:
        """Warn that a bootstrap does not rescue inference for estimated weights.

        A user told that estimated weights need "a bootstrap that re-derives them" will
        reach for ``n_bootstrap=``, and it is the wrong tool: every replicate inherits the
        weights it was handed and merely renormalises them, so the bootstrap interval
        conditions on the fitted weights. Saying so is cheap; letting the mistake pass
        silently is not.  The warning does not compare the bootstrap with an
        influence-curve interval, because a :class:`~cleverly.DRTMLE` fit with a guard and
        estimated weights reports none (the ``"estimated_weight_plugin"`` status).
        """
        if not (data.declares_estimated_weights and self.n_bootstrap):
            return
        warnings.warn(
            "weights_estimated=True with n_bootstrap: the bootstrap resamples rows and "
            "renormalises the weights it was given, never re-deriving them, so its "
            "intervals condition on the fitted weights. Re-deriving the weights inside "
            "each replicate needs the model that produced them, which this package never "
            "sees. See cleverly.data.weighting.",
            WeightingWarning,
            stacklevel=3,
        )

    def _warn_on_positivity(
        self,
        data: CausalData,
        nuisance: NuisanceEstimates,
        config: TMLEConfig,
        intermediate_value: float | None = None,
    ) -> None:
        lower, upper = config.g_bounds
        # Counted per *unit*: a row is extrapolated if any arm's probability is outside the
        # bounds, since one binding denominator is enough to give that row unbounded
        # leverage.  Which cells are outside is `Propensity.truncate`'s rule rather than a
        # predicate written here, so the warning counts the rows the targeting step really
        # truncates even when the bound pair is asymmetric.
        if nuisance.fits_treatment:
            outside = nuisance.propensity.truncate(config.g_bounds).fraction
            if outside > _TRUNCATION_WARN_FRACTION:
                warnings.warn(
                    f"{outside:.1%} of units have an estimated treatment probability outside the "
                    f"truncation bounds [{lower:.4g}, {upper:.4g}] for at least one arm. Those "
                    "units still contribute, but their contributions use bounded rather "
                    "than fitted "
                    "mechanism values and are sensitive to this regularization. Inspect "
                    "res.diagnostics.support() and res.diagnostics.truncation_curve() before "
                    "trusting the estimate.",
                    PositivityWarning,
                    stacklevel=3,
                )

        # The propensity is not the only denominator in the clever covariate. A
        # missingness or intermediate probability near zero gives a row exactly the same
        # unbounded leverage, and it is the one a reader is least likely to be watching
        # for -- overlap in g can look immaculate while the estimate rests on a handful
        # of rows that were very unlikely to be observed at all.
        #
        # The intermediate entry has to be the density for the level *being targeted*.
        # ``nuisance.intermediate`` holds P(Z = 1 | A, W), but the covariate divides by
        # its complement when z = 0, so reading the raw array checks the wrong tail: a
        # sample with P(Z = 1 | A, W) = 0.999 is a severe positivity violation for the
        # z = 0 effect and none at all for the z = 1 one.
        natural_course = not nuisance.fits_treatment
        missingness_values = (
            nuisance.missingness_at_realised_arm(data.treatment)
            if natural_course
            else nuisance.missingness
        )
        candidates: list[tuple[str, FloatArray | None]] = [
            ("P(Delta = 1 | A, W)", missingness_values),
            # The treatment observation factor of a declared missing treatment divides the
            # composite covariate as the response mechanism does.
            ("P(Delta_A = 1 | W)", nuisance.treatment_observation),
        ]
        if nuisance.intermediate is not None and intermediate_value is not None:
            candidates.append(
                (
                    f"P(Z = {intermediate_value:.0f} | A, W)",
                    nuisance.intermediate_density(intermediate_value, 0.0),
                )
            )

        for label, values in candidates:
            if values is None:
                continue
            below = float(np.mean(np.asarray(values, dtype=float) < config.missingness_bound))
            if below > _TRUNCATION_WARN_FRACTION:
                guidance = (
                    "That probability is the sole denominator in the natural-course "
                    "response-residual covariate, so those rows carry outsized leverage and "
                    "the bound is trading bias for variance. Inspect "
                    "res.diagnostics.nuisance_models() and re-run with a different "
                    "nuisance_bound."
                    if natural_course
                    else "That probability divides the clever covariate just as g(W) does, "
                    "so those rows carry outsized leverage and the bound is trading bias for "
                    "variance. Inspect res.diagnostics.support() and re-run with a different "
                    "nuisance_bound."
                )
                warnings.warn(
                    f"{below:.1%} of estimated {label} values fall below the nuisance bound "
                    f"{config.missingness_bound:.4g}. {guidance}",
                    PositivityWarning,
                    stacklevel=3,
                )

    # ------------------------------------------------------- targeting layer

    def retarget(
        self,
        data: CausalData,
        nuisance: NuisanceEstimates,
        *,
        estimands: Sequence[str],
        intermediate_value: float | None = None,
        g_bounds: tuple[float, float] | None = None,
        g_bounds_conditional: tuple[float, float] | None = None,
        missingness: FloatArray | None = None,
        nuisance_bound: float | None = None,
        alpha_sig: float | None = None,
    ) -> tuple[dict[str, ParameterEstimate], dict[str, Fluctuation]]:
        """Run the targeting step and build estimates from cached nuisance fits.

        Separated from :meth:`fit` because every sensitivity analysis is exactly this
        operation with one input perturbed -- a different truncation bound, a tilted
        missingness mechanism -- and re-running it costs a fraction of a full refit.

        ``nuisance_bound`` overrides the lower bound on the missingness and intermediate
        mechanisms, which is the other denominator in the clever covariate and so the
        other bound whose influence on the answer is worth sweeping.
        """
        estimates, fluctuations, _ = self._retarget_detailed(
            data,
            nuisance,
            estimands=estimands,
            intermediate_value=intermediate_value,
            g_bounds=g_bounds,
            g_bounds_conditional=g_bounds_conditional,
            missingness=missingness,
            nuisance_bound=nuisance_bound,
            alpha_sig=alpha_sig,
        )
        return estimates, fluctuations

    def _retarget_detailed(
        self,
        data: CausalData,
        nuisance: NuisanceEstimates,
        *,
        estimands: Sequence[str],
        intermediate_value: float | None = None,
        g_bounds: tuple[float, float] | None = None,
        g_bounds_conditional: tuple[float, float] | None = None,
        missingness: FloatArray | None = None,
        nuisance_bound: float | None = None,
        alpha_sig: float | None = None,
    ) -> tuple[dict[str, ParameterEstimate], dict[str, Fluctuation], CVTargeting | None]:
        """:meth:`retarget`, plus the fold-level report when targeting went fold by fold.

        The extra return value is what :meth:`fit` puts on ``result.cv_targeting``.  It
        is kept out of :meth:`retarget` so that the sensitivity analyses, which call
        that method on every perturbed input, keep their two-value signature.

        It runs :func:`~cleverly.msm.refuse_msm_functions` and
        :func:`~cleverly.interventions.base.refuse_regime_densities` first, through
        :meth:`_refuse_undeclared_functions`, as :meth:`fit` does.  Every sweep that
        recomputes an estimate comes through here, so a recomputation refuses when those
        functions refuse a modified model or regime: a ``Stochastic`` density, a ``Rule``,
        or a user-written ``Intervention``.  They state which declarations they refuse.

        A requested target of another parameter axis is refused next.  :meth:`fit`
        resolves its targets on the fit's own axis, but a direct call can name any
        target, and the nuisances of this fit would then be read as that target's.  A
        learned rule would be reported as a known regime.
        """
        self._refuse_undeclared_functions()
        requested = tuple(estimands)
        off_axis = [
            name
            for name in requested
            if name in TARGETS and not TARGETS[name].matches_axis(self._axis)
        ]
        if off_axis:
            raise CapabilityError(
                f"retarget cannot report {off_axis} from this fit's nuisances. "
                f"{_off_axis_reason(off_axis, self._axis)}."
            )
        level = self.alpha_sig if alpha_sig is None else alpha_sig
        regimes = nuisance.regimes
        reference = self._reference_arm(data, regimes, nuisance.policies)
        mean_bounds = g_bounds or resolve_g_bounds(
            self.g_bounds, self._bounds_n(data), for_att=False
        )
        conditional_bounds = g_bounds_conditional or resolve_g_bounds(
            self.g_bounds, self._bounds_n(data), for_att=True
        )
        # The inference status reads the prepared data, which the composite route replaces.
        prepared = data
        if missing_data_route(self, data) == "composite":
            # Targeting, the curve and the corrections read the composite indicator, and
            # the composite mechanism is formed here from the fitted factors at this call's
            # bounds.  So a retarget at new bounds recomputes it and cannot read a stale one.
            # The gate admits the arm means, their contrasts, regimes and arm MSMs, which
            # read the arm covariates only, so the conditional bounds are never read here.
            state = composite_state(
                data,
                nuisance,
                g_bounds=mean_bounds,
                nuisance_bound=self.nuisance_bound if nuisance_bound is None else nuisance_bound,
                missingness=missingness,
            )
            data, nuisance, mean_bounds = state.data, state.nuisance, state.bounds
            conditional_bounds = mean_bounds
            missingness = None

        estimates: dict[str, ParameterEstimate] = {}
        fluctuations: dict[str, Fluctuation] = {}
        pooled_report: dict[str, ParameterEstimate] = {}
        fold_evaluated_report: dict[str, ParameterEstimate] = {}
        fold_estimates: dict[str, tuple[float, ...]] = {}
        epsilon: dict[str, tuple[float, ...]] = {}
        fold_epsilon: dict[str, tuple[tuple[float, ...], ...]] = {}
        # The stratum code of every stratum estimate, which the stamp reads for the
        # Student t reference of a clustered fit. Every other estimate reads every row.
        stratum_of: dict[str, int] = {}
        indices: list[IntArray] = []
        validation_indices = (
            [] if nuisance.folds.is_single else [test for _, test in nuisance.folds]
        )

        joint = _is_joint_natural_course(data, requested)
        groups: list[TargetGroup] = (
            ["natural_course"]
            if _is_natural_course(data, requested)
            else ["natural_course", *self._groups(requested)]
            if joint
            else self._groups(requested)
        )
        # The joint route solves the natural-course fluctuation first and holds its mean
        # for the ``mean`` context, which reports every estimate of the fit.
        natural_course: ArmMean | None = None
        natural_course_strata: dict[int, ArmMean] = {}
        for group in groups:
            bounds = g_bounds_for(group, mean_bounds, conditional_bounds)
            # A group whose parameter is defined *through* the mechanism has a second
            # score equation, so its targeting alternates and returns the nuisances
            # re-tilted at the targeted g. `targeted` is what the estimates are read
            # from; `nuisance` stays the initial fit and is what the result reports.
            targeted = nuisance
            targeting_submodel: Submodel | None = None
            if needs_mechanism(group):
                submodel, fluctuation, targeted, targeting_submodel = self._solve_mechanism(
                    data, nuisance, group, bounds, nuisance_bound
                )
            elif needs_reduction(nuisance, group):
                # A fit carrying reduced-dimension regressions solves two further score
                # equations, one of which fluctuates g. Nothing about the *reported*
                # nuisances moves -- the estimand is still the plug-in mean of the targeted
                # regression -- so this returns two values, as the projection does and
                # unlike the mechanism alternation.
                submodel, fluctuation, targeting_submodel = self._solve_reduction(
                    data, nuisance, group, bounds, nuisance_bound
                )
            elif needs_projection(nuisance, group):
                # A working model with a non-identity link has a clever covariate that
                # reads its own coefficients, so the covariate and the projection are
                # solved for together. Nothing about the nuisances moves, which is why
                # this returns two values where the mechanism alternation returns three.
                submodel, fluctuation = self._solve_projection(
                    data, nuisance, group, bounds, nuisance_bound
                )
                if data.has_strata:
                    # The marginal coefficients above are the unstratified solve.  The
                    # stratum coefficients need their own fluctuation, because each
                    # stratum block reads its own coefficients.
                    targeting_submodel, nested = self._solve_stratified_projection(
                        data, nuisance, group, bounds, nuisance_bound
                    )
                    fluctuation = replace(fluctuation, stratified=nested)
            else:
                submodel = self._submodel(
                    data,
                    nuisance,
                    group,
                    bounds,
                    intermediate_value,
                    missingness,
                    nuisance_bound,
                    # The conditional-effect fluctuations contrast against this arm, and
                    # so must contrast against the *same* one the estimand layer reports
                    # against -- which is why it is read once, here, rather than resolved
                    # again inside the builder.
                    reference,
                )
                targeting_submodel = (
                    self._stratified_submodel(
                        data,
                        nuisance,
                        group,
                        bounds,
                        intermediate_value,
                        missingness,
                        nuisance_bound,
                        reference,
                    )
                    if data.has_strata
                    else submodel
                )
                _, fluctuation = self._solve(data, nuisance, targeting_submodel)
            fluctuations[group] = fluctuation

            if joint and group == "natural_course":
                # No target is built from this context: ``_estimates_for`` maps the group
                # to ``mean``, so building here would emit a second copy of every
                # mean-group target, and dict order would decide which one survives.
                natural_course = natural_course_mean(
                    nuisance.scaler.scale(data.outcome),
                    fluctuation.targeted,
                    submodel,
                    data.weights,
                    data.observed,
                )
                if data.has_strata:
                    assert targeting_submodel is not None
                    natural_course_strata = self._stratum_natural_course(
                        data, nuisance, submodel, targeting_submodel, fluctuation
                    )
                continue
            pooled = self._estimates_for(
                data,
                targeted,
                group,
                submodel,
                fluctuation,
                requested,
                level,
                reference,
                natural_course=natural_course if group == "mean" else None,
            )
            pooled_report.update(pooled)
            if data.has_strata:
                assert targeting_submodel is not None
                stratified, codes = self._stratum_estimates(
                    data,
                    targeted,
                    group,
                    submodel,
                    targeting_submodel,
                    fluctuation,
                    requested,
                    level,
                    reference,
                    natural_course=natural_course_strata if group == "mean" else None,
                )
                stratum_of.update(codes)
                pooled.update(stratified)
                pooled_report.update(stratified)
            group_indices = (
                [record.index for record in fluctuation.folds]
                if fluctuation.folds
                else (validation_indices if self.cv_evaluation else [])
            )
            if not group_indices:
                estimates.update(pooled)
                continue

            indices = group_indices
            epsilon[group] = tuple(fluctuation.epsilon.tolist())
            if fluctuation.folds:
                fold_epsilon[group] = tuple(
                    tuple(record.epsilon.tolist()) for record in fluctuation.folds
                )
            # The *targets* to rebuild per fold, not the parameter names the pooled fit
            # produced: a target reports one parameter per arm, and `targets_for` selects
            # by target name. A target that a fold cannot evaluate is dropped there by
            # `drop_undefined`, which is what `_average_over_folds` then reconciles.
            per_fold = [
                self._fold_estimates(
                    data, targeted, group, submodel, fluctuation, requested, level, index
                )
                for index in indices
            ]  # regimes ride along on `nuisance`, and are sliced per fold below
            fold_evaluated = _average_over_folds(
                per_fold, tuple(pooled), indices, n=data.n, cluster=data.cluster, alpha=level
            )
            fold_evaluated_report.update(fold_evaluated)
            fold_estimates.update(
                {
                    name: tuple(values[name].psi for values in per_fold)
                    for name in fold_evaluated  # only the estimands every fold could compute
                }
            )
            estimates.update(fold_evaluated if self.cv_evaluation else pooled)

        # Stamped at the one place every estimate this estimator produces comes from, so
        # ``fit``, ``retarget`` and every sensitivity sweep that retargets a perturbed
        # input all report the same status.  Stamping in ``fit`` alone would leave the
        # truncation curve and the refutations building intervals the fit itself refuses.
        # The two fold-level reports are stamped here too, because ``CVTargeting``
        # publishes their standard errors and reads its status off them. A clustered fit
        # that supplies inference also takes its Student t reference here, from the rows
        # each estimate reads, so no builder sets degrees of freedom of its own.
        status = self._inference_status(prepared)
        cluster_reference = (
            None
            if prepared.cluster is None
            else ClusterReference(
                prepared.cluster,
                weights=prepared.weights if prepared.is_weighted else None,
                strata=prepared.strata,
                stratum_of=stratum_of,
            )
        )
        headline_reference: ClusterReference | None = cluster_reference
        if cluster_reference is not None and self.cv_evaluation and indices:
            # The headline is the fold-evaluated report, whose variance has J - V degrees of
            # freedom (ClusterReference). The stacked report keeps J - 2.
            headline_reference = cluster_reference.with_fold_evaluated(
                fold_evaluated_report, len(indices)
            )
        ordered = stamp_inference(
            _in_report_order(estimates, requested), status, headline_reference
        )
        detail = (
            CVTargeting(
                n_folds=len(indices),
                fold_sizes=tuple(int(index.size) for index in indices),
                variance={name: value.variance for name, value in fold_evaluated_report.items()},
                fold_estimates=fold_estimates,
                epsilon=epsilon,
                fold_epsilon=fold_epsilon,
                pooled=_in_report_order(pooled_report, requested),
                fold_evaluated=_in_report_order(fold_evaluated_report, requested),
                backend=data.backend,
            ).stamped(status, cluster_reference)
            if indices
            else None
        )
        return ordered, fluctuations, detail

    def _fold_estimates(
        self,
        data: CausalData,
        nuisance: NuisanceEstimates,
        group: TargetGroup,
        submodel: Submodel,
        fluctuation: Fluctuation,
        supported: Sequence[str],
        alpha_sig: float,
        index: IntArray,
    ) -> dict[str, ParameterEstimate]:
        """Every estimand this group supports, computed inside one validation fold.

        A fold can be degenerate where the whole sample is not: too few units in the
        conditioning arm for an ATT, or a counterfactual mean at the boundary that leaves
        a ratio undefined.  Those estimands are dropped from this fold's report rather
        than allowed to abort the others; :func:`_average_over_folds` then drops them
        from the fold-evaluated estimate altogether and says so.

        Only a target that declares ``undefined_when`` may be dropped, and only its own
        entry is lost.  This replaces a bare ``except ValueError`` that retried without
        ``{"rr", "or"}`` and then returned an empty dict -- which turned any exception
        anywhere in the estimate path into a fold that silently reported nothing.
        """
        return self._estimates_for(
            data,
            nuisance,
            group,
            submodel,
            fluctuation,
            supported,
            alpha_sig,
            self._reference_arm(data, nuisance.regimes, nuisance.policies),
            index=index,
            drop_undefined=True,
        )

    @staticmethod
    def _groups(estimands: Sequence[str]) -> list[TargetGroup]:
        """Which fluctuations must be fit to cover the requested estimands.

        Each estimand family gets its own targeting step, because each has its own
        efficient influence function and therefore its own score equation to solve.
        """
        return groups_for(estimands)

    def targeting_spec(self) -> TargetingSpec:
        """The targeting settings this estimator would use, as one object.

        Recorded on every result via :attr:`TMLEConfig.targeting_spec`, so re-solving
        a fluctuation never needs the estimator itself.
        """
        return TargetingSpec(
            targeting=self.targeting,
            fluctuation=self.fluctuation,
            target_weights=self.target_weights,
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=self.tol,
            step_size=self.step_size,
        )

    def _fold_strata(self, data: CausalData) -> FloatArray | None:
        """What the outer folds are balanced on, as one code per row.

        ``None`` when there is nothing to balance.  A dose is the case that matters:
        stratifying on it would ask for folds balanced on a variable whose every value is
        its own stratum, which caps the fold count at the rarest "class" -- one row -- and
        refuses to split at all.  The density's bins are what a continuous treatment
        stratifies on in spirit, and they are chosen inside ``fit_conditional_density``
        from the training rows of each fold, so they cannot be known here without leaking
        the split into itself.

        Every fit reaches this method with ``stratify_folds="none"`` and leaves at the
        first branch below.  :meth:`_cross_fit_policy_reason` refuses the other two
        policies at construction, and again in :meth:`_resolve_estimands_for_data` for a
        copied or modified estimator, and it refuses them under cross-fitting and
        at every selector-based collaborative setting. The remaining branches are therefore
        reachable only by replacing this method, which is what the deliberate-mutation controls in
        ``tests/unit/test_fold_policy_rules.py`` do to put a stratified split into
        production.  The two ``"treatment+outcome"`` refusals that used to stand here
        were deleted for the same reason: no fit could meet them.
        """
        if self.stratify_folds == "none":
            return None
        if data.is_continuous_treatment:
            return None
        if self.stratify_folds == "treatment":
            return data.treatment
        outcome = np.where(data.observed, data.outcome, -1.0)
        codes: FloatArray = np.unique(
            np.column_stack([data.treatment, outcome]), axis=0, return_inverse=True
        )[1].astype(float)
        return codes

    def crossfit_plan(self, data: CausalData) -> CrossFitPlan:
        """The fold policy this estimator declared, as one object.

        Recorded on every result via :attr:`TMLEConfig.crossfit`, beside the fold count
        the fit actually ran.  The two can differ -- ``resolve_n_folds`` caps at the
        rarest stratum and again at the cluster count -- and the warnings that say so are
        gone by the time anyone reads the result.

        Takes ``data`` because two of the fields are answers about it rather than
        settings: whether clusters were declared, and whether the treatment has strata to
        balance at all.  Both decisions are made here and in :meth:`_folds`, which is one
        place too many, so :meth:`_folds` reads them off the plan.

        ``stratify_by`` records what the folds were held to.  It is empty on every fit
        this version runs, because :meth:`_cross_fit_policy_reason` refuses the two
        balancing policies wherever a split is drawn.  The field stays because a copied or
        modified estimator can declare such a policy, and the plan records it before the
        fit refuses it.  ``scheme`` names
        only the splits this version draws for the same reason: ``"stratified"`` and
        ``"stratified-grouped"`` needed a nonempty ``stratify_by``, so no fit could record
        them, and they were deleted.
        """
        cross_fit = self.cross_fit
        supplied = self.split_plan
        stratify_by: tuple[str, ...]
        if not cross_fit or data.is_continuous_treatment or self.stratify_folds == "none":
            stratify_by = ()
        elif self.stratify_folds == "treatment":
            stratify_by = (data.treatment_name,)
        else:
            stratify_by = (data.treatment_name, data.outcome_name)
        clustered = cross_fit and data.cluster is not None
        if supplied is not None:
            scheme = "supplied"
        elif not cross_fit:
            scheme = "none"
        elif clustered:
            scheme = "grouped"
        else:
            scheme = "vfold"
        return CrossFitPlan(
            n_folds=self.n_folds if cross_fit else 1,
            learner_folds=self.learner_folds,
            scheme=scheme,
            stratify_by=stratify_by,
            random_state=self.random_state,
            repeats=self.repeats,
        )

    def _submodel(
        self,
        data: CausalData,
        nuisance: NuisanceEstimates,
        group: TargetGroup,
        bounds: tuple[float, float],
        intermediate_value: float | None,
        missingness_override: FloatArray | None,
        nuisance_bound: float | None = None,
        reference: float | None = None,
    ) -> Submodel:
        lower = self.nuisance_bound if nuisance_bound is None else float(nuisance_bound)
        submodel = build_submodel(
            data,
            nuisance,
            group,
            bounds=bounds,
            nuisance_bound=lower,
            intermediate_value=intermediate_value,
            missingness_override=missingness_override,
            reference=reference,
        )
        if (
            group not in ("att", "atc")
            or not self.cross_fit
            or nuisance.folds.is_single
            or (self.targeting_scheme == "pooled" and not self.cv_evaluation)
        ):
            return submodel

        # A fold-evaluated CV-TMLE updates fold-specific distributions. ATT/ATC's
        # gradient contains the empirical arm probability of that distribution, so its
        # clever covariate must be rebuilt with each validation fold's probability before
        # the pieces are stacked (common epsilon) or solved separately (the extension).
        # Restricting a covariate built with the full-sample share would target a different
        # score. The Levy/tmle3 stacked report uses the full empirical distribution and
        # therefore correctly keeps the full-sample share above.
        pieces = []
        for _, test in nuisance.folds:
            fractions = np.array(
                [arm_share(data.treatment, data.weights, arm, mask=test) for arm in nuisance.arms],
                dtype=float,
            )
            fold_submodel = build_submodel(
                data,
                nuisance,
                group,
                bounds=bounds,
                nuisance_bound=lower,
                intermediate_value=intermediate_value,
                missingness_override=missingness_override,
                reference=reference,
                arm_fractions=fractions,
            )
            pieces.append((test, restrict(fold_submodel, test)))
        return stitch(pieces, data.n)

    def _stratified_submodel(
        self,
        data: CausalData,
        nuisance: NuisanceEstimates,
        group: TargetGroup,
        bounds: tuple[float, float],
        intermediate_value: float | None,
        missingness_override: FloatArray | None,
        nuisance_bound: float | None,
        reference: float,
        msm_beta: Sequence[FloatArray] | None = None,
    ) -> Submodel:
        r"""One disjoint score block per baseline stratum.

        For stratum ``s`` the block is ``I(S=s) H_s / P_n(S=s)``.  ``H_s`` is
        rebuilt with the *conditional* arm shares for ATT/ATC; multiplying a globally
        normalised conditional-effect covariate, as a generic wrapper would do, targets
        the wrong denominator.  The blocks have disjoint support, so no redundant
        marginal column is added: the marginal score is their empirical weighted sum.

        Arm shares are read only on an arm-coded treatment.  A continuous dose has no arm
        shares, and the natural-course group fits no treatment model.  ``msm_beta`` holds
        one coefficient vector per stratum, for a working model whose covariate reads its
        coefficients.
        """
        assert data.strata is not None
        lower = self.nuisance_bound if nuisance_bound is None else float(nuisance_bound)
        probabilities = stratum_probabilities(data)
        reads_shares = not data.is_continuous_treatment and group != "natural_course"
        pieces: list[Submodel] = []
        for code in range(data.n_strata):
            mask = data.strata == code
            fractions = None
            if reads_shares:
                fractions = np.array(
                    [
                        arm_share(data.treatment, data.weights, arm, mask=mask)
                        for arm in nuisance.arms
                    ],
                    dtype=float,
                )
                if fractions.size and np.any(fractions <= 0.0):
                    raise absent_arm_error(data, nuisance.arms, fractions, code, group)
            pieces.append(
                build_submodel(
                    data,
                    nuisance,
                    group,
                    bounds=bounds,
                    nuisance_bound=lower,
                    intermediate_value=intermediate_value,
                    missingness_override=missingness_override,
                    reference=reference,
                    arm_fractions=fractions,
                    msm_beta=None if msm_beta is None else msm_beta[code],
                )
            )
        labels = [data.stratum_label(code) for code in range(data.n_strata)]
        return stratify(pieces, data.strata, probabilities, labels)

    def _solve(
        self, data: CausalData, nuisance: NuisanceEstimates, submodel: Submodel
    ) -> tuple[Submodel, Fluctuation]:
        """Solve the fluctuation, pooled over folds or one fluctuation per fold.

        Returns the submodel beside the fluctuation because under fold-wise targeting the
        two are no longer independent: a covariate that reads a fold-specific quantity --
        a linked working model's ``beta`` -- differs between folds, and the score has to be
        taken against the covariate each row was actually fluctuated by.  For every other
        group the returned submodel is the one that went in, value for value.
        """
        scaled = nuisance.scaler.scale(data.outcome)
        if self.targeting_scheme == "fold" and self.cross_fit:
            if not nuisance.folds.is_single:
                return self._solve_by_fold(
                    data,
                    nuisance,
                    lambda test: (
                        restrict(submodel, test),
                        self._solve_rows(
                            scaled[test],
                            _slice_fit(nuisance.outcome, test),
                            restrict(submodel, test),
                            data.weights[test],
                            data.observed[test],
                            warn=False,
                        ),
                    ),
                    submodel.group,
                )
            # Only reachable when resolve_n_folds collapsed the split -- too few units
            # in the rarer treatment arm to stratify. The constructor already warned
            # about the cross_fit=False route, so this is the remaining silent one.
            warnings.warn(
                "targeting_scheme='fold' was requested but the data supports only a "
                "single fold, so there is no validation split to target within; "
                "falling back to pooled targeting.",
                UserWarning,
                stacklevel=3,
            )
        return submodel, self._solve_rows(
            scaled,
            nuisance.outcome,
            submodel,
            self._validation_weights(data, nuisance),
            data.observed,
        )

    def _validation_weights(self, data: CausalData, nuisance: NuisanceEstimates) -> FloatArray:
        """Weights for one common update over the validation losses.

        Levy's easy implementation stacks the out-of-fold predictions and runs the
        ordinary empirical-risk update. The pinned ``tmle3`` snapshot corroborates that
        path. It therefore keeps ``data.weights`` unchanged. The original fold-evaluated
        construction selected by
        ``cv_evaluation=True`` instead defines its risk as the equal average of the
        validation-fold empirical risks.  Those risks normalise observation weights
        *inside* each fold.  Multiplying a fold by ``n / (V * sum(w_fold))`` expresses
        that objective as one stacked regression, including when folds have unequal row
        counts or unequal sampling-weight mass.

        A fold-specific update is invariant to multiplying all of its weights by a
        constant and never calls this helper.  A non-cross-fitted fit has no validation
        risks to average and likewise keeps the original empirical measure.
        """
        if not self.cv_evaluation or not self.cross_fit or nuisance.folds.is_single:
            return data.weights
        weights = np.array(data.weights, dtype=float, copy=True)
        n_folds = len(nuisance.folds)
        for _, test in nuisance.folds:
            mass = float(np.sum(weights[test]))
            if mass <= 0.0:
                raise ValueError("each validation fold must have positive observation-weight mass")
            weights[test] *= data.n / (n_folds * mass)
        return weights

    def _reduction(self, data: CausalData, nuisance: NuisanceEstimates) -> ReductionSpec | None:
        """How to refit the reduced-dimension regressions, or ``None`` for a plain fit.

        The extension point for the doubly-robust variant, and the one place a targeting
        step here needs a learner.  :class:`~cleverly.DRTMLE` returns a closure over the
        learners it resolved; every other estimator returns ``None``, which is what makes
        a plain ``TMLE`` handed somebody else's nuisances refuse rather than re-solve the
        extra equations against arrays it cannot refresh.
        """
        del data, nuisance
        return None

    def _solve_mechanism(
        self,
        data: CausalData,
        nuisance: NuisanceEstimates,
        group: TargetGroup,
        bounds: tuple[float, float],
        nuisance_bound: float | None,
    ) -> tuple[Submodel, Fluctuation, NuisanceEstimates, Submodel]:
        """Alternate the outcome and the treatment mechanism, with one block per stratum.

        Returns the marginal submodel, the fluctuation, the re-tilted nuisances, and the
        submodel the fluctuation solved.  The last two submodels are one object without
        baseline strata.  With strata, both equations take one block per stratum
        (:func:`~cleverly.fluctuation.submodel.stratify_columns`), and the marginal submodel
        is rebuilt at the targeted tilt for the marginal estimates.  Its score is the
        ``P_n(S = s)``-weighted sum of the solved blocks.
        """
        lower = self.nuisance_bound if nuisance_bound is None else float(nuisance_bound)
        outcome_submodel = None
        mechanism_columns = None
        if data.has_strata:
            assert data.strata is not None
            strata = data.strata
            probabilities = stratum_probabilities(data)
            reference = self._reference_arm(data, nuisance.regimes, nuisance.policies)

            def outcome_submodel(current: NuisanceEstimates) -> Submodel:
                return self._stratified_submodel(
                    data, current, group, bounds, None, None, lower, reference
                )

            def mechanism_columns(targeted: InitialFit, tilt: IPSISet) -> FloatArray:
                return stratify_columns(
                    mechanism_covariate(group, targeted, tilt), strata, probabilities
                )

        solved, fluctuation, targeted = solve_with_mechanism(
            data,
            nuisance,
            group,
            self.targeting_spec(),
            bounds=bounds,
            nuisance_bound=lower,
            scaled=nuisance.scaler.scale(data.outcome),
            weights=self._validation_weights(data, nuisance),
            observed=data.observed,
            outcome_submodel=outcome_submodel,
            mechanism_columns=mechanism_columns,
        )
        if not data.has_strata:
            return solved, fluctuation, targeted, solved
        marginal = build_submodel(data, targeted, group, bounds=bounds, nuisance_bound=lower)
        return marginal, fluctuation, targeted, solved

    def _solve_reduction(
        self,
        data: CausalData,
        nuisance: NuisanceEstimates,
        group: TargetGroup,
        bounds: tuple[float, float],
        nuisance_bound: float | None,
    ) -> tuple[Submodel, Fluctuation, Submodel]:
        """Alternate the outcome, the mechanism and the reduced regressions.

        Returns the marginal submodel, the fluctuation, and the submodel the outcome
        equation was solved along.  With baseline strata every equation takes one block per
        stratum (:class:`~cleverly.estimators.targeting.StratumBlocks`), and the last is the
        marginal submodel's blocks.  Without strata the two submodels are one object.

        Pooled only.  Fold-wise targeting would need each fold's reduced regressions fitted
        out of that fold and its own alternation run inside it, which is a derivation rather
        than a loop -- :class:`~cleverly.DRTMLE` refuses ``targeting_scheme="fold"`` by name
        rather than quietly targeting pooled, which is what the mechanism alternation does.
        """
        reduction = self._reduction(data, nuisance)
        if reduction is None:
            raise ValueError(
                "these nuisances carry reduced-dimension regressions, so the targeting step "
                "has two further score equations to solve -- and solving them refits those "
                f"regressions against the targeted pair, which a {type(self).__name__} has "
                "no learners for. Retarget with the DRTMLE that fitted them, or drop "
                "`reduced` to report a plain TMLE under a plain TMLE's name."
            )
        blocks = stratum_blocks(data, nuisance)
        submodel, fluctuation = solve_with_reduction(
            data,
            nuisance,
            group,
            self.targeting_spec(),
            reduction=reduction,
            bounds=bounds,
            nuisance_bound=self.nuisance_bound if nuisance_bound is None else nuisance_bound,
            scaled=nuisance.scaler.scale(data.outcome),
            weights=data.weights,
            observed=data.observed,
            # Read off the estimator rather than left to the function's default. Only a
            # `DRTMLE` reaches here -- the branch above refuses any other estimator carrying
            # reduced regressions -- but the method is defined on `TMLE`, so the attribute is
            # fetched defensively rather than assumed onto a class that does not declare it.
            max_outer=getattr(self, "max_outer", DEFAULT_MAX_OUTER),
            blocks=blocks,
        )
        return submodel, fluctuation, submodel if blocks is None else blocks.submodel(submodel)

    def _solve_projection(
        self,
        data: CausalData,
        nuisance: NuisanceEstimates,
        group: TargetGroup,
        bounds: tuple[float, float],
        nuisance_bound: float | None,
    ) -> tuple[Submodel, Fluctuation]:
        """Alternate the projection and the fluctuation, pooled or fold by fold.

        Under fold-wise targeting each fold runs its own alternation and so gets its own
        ``beta``.  This removes cross-fold coupling through a pooled projection, but the
        rows in a fold still fit both the ``beta`` and epsilon used for that fold.  Each
        fold's score is zero at the beta its own rows were fluctuated at, so the stitched
        score is zero as well.
        """
        spec = self.targeting_spec()
        lower = self.nuisance_bound if nuisance_bound is None else float(nuisance_bound)
        scaled = nuisance.scaler.scale(data.outcome)
        alternate = partial(
            solve_with_projection,
            data,
            nuisance,
            group,
            spec,
            bounds=bounds,
            nuisance_bound=lower,
            scaled=scaled,
            weights=self._validation_weights(data, nuisance),
            observed=data.observed,
        )
        if self.targeting_scheme == "fold" and self.cross_fit and not nuisance.folds.is_single:
            per_fold: list[ProjectionFluctuation] = []

            def one_fold(test: IntArray) -> tuple[Submodel, Fluctuation]:
                fold_submodel, fold_fluctuation = alternate(rows=test, warn=False)
                record = fold_fluctuation.projection
                assert isinstance(record, ProjectionFluctuation)
                per_fold.append(record)
                return fold_submodel, fold_fluctuation

            submodel, fluctuation = self._solve_by_fold(data, nuisance, one_fold, group)
            # There is no single beta the covariate was built at here -- each fold had its
            # own, which is the point -- but there is a single beta the coefficients are
            # *reported* at: the projection of the stitched targeted fit, which is the
            # solve `msm_coefficients` runs. That is what a diagnostic rebuilding the
            # covariate wants, so it is what the record carries, with the folds beside it.
            beta = reported_beta(nuisance, fluctuation.targeted, data.weights)
            assert beta is not None
            return submodel, replace(
                fluctuation,
                projection=ProjectionFluctuation(
                    beta=beta,
                    trace=tuple(
                        (i, *record.trace[-1][1:]) for i, record in enumerate(per_fold) if record
                    ),
                    converged=all(record.converged for record in per_fold),
                    failure=next(
                        (record.failure for record in per_fold if record.failure is not None), None
                    ),
                    folds=tuple(per_fold),
                ),
            )
        return alternate()

    def _solve_stratified_projection(
        self,
        data: CausalData,
        nuisance: NuisanceEstimates,
        group: TargetGroup,
        bounds: tuple[float, float],
        nuisance_bound: float | None,
    ) -> tuple[Submodel, Fluctuation]:
        """The stratum coefficients of a linked working model, pooled over the sample.

        Fold-wise targeting and fold evaluation refuse baseline strata before any learner,
        so this has no fold form.
        """
        lower = self.nuisance_bound if nuisance_bound is None else float(nuisance_bound)
        reference = self._reference_arm(data, nuisance.regimes, nuisance.policies)
        return solve_with_stratified_projection(
            data,
            nuisance,
            group,
            self.targeting_spec(),
            bounds=bounds,
            nuisance_bound=lower,
            scaled=nuisance.scaler.scale(data.outcome),
            weights=self._validation_weights(data, nuisance),
            observed=data.observed,
            blocks_at=lambda betas: self._stratified_submodel(
                data, nuisance, group, bounds, None, None, lower, reference, msm_beta=betas
            ),
        )

    def _solve_rows(
        self,
        scaled: FloatArray,
        initial: InitialFit,
        submodel: Submodel,
        weights: FloatArray,
        observed: BoolArray,
        *,
        warn: bool = True,
    ) -> Fluctuation:
        return solve_submodel(
            scaled, initial, submodel, weights, observed, self.targeting_spec(), warn=warn
        )

    def _solve_by_fold(
        self,
        data: CausalData,
        nuisance: NuisanceEstimates,
        per_fold: Callable[[IntArray], tuple[Submodel, Fluctuation]],
        group: TargetGroup,
    ) -> tuple[Submodel, Fluctuation]:
        """The optional fold-specific targeting extension.

        Each fold's ``epsilon`` is fit only against rows whose nuisance predictions came
        from a model trained on the other folds.  Unlike common-update CV-TMLE, which fits
        one common coefficient by pooling the validation losses, this extension fits a
        coefficient separately on each validation fold.  The fold's outcomes therefore
        *do* contribute to the coefficient that fluctuates that fold; cross-fitting
        applies to the initial nuisance predictions, not to epsilon.

        The fold-specific targeted predictions are stitched back into a full-length fit.
        Because each fold's score is zero on its own rows, the pooled score -- a sum over
        folds -- is zero too, so the estimating equation is still solved exactly on the
        full sample.  The reported ``epsilon`` is the mass-weighted average across folds
        and is a summary only; the per-fold values are kept in
        :attr:`~cleverly.fluctuation.Fluctuation.folds`.

        Stitching gives the pooled report for this separate-epsilon extension. With
        ``cv_evaluation=True`` the same fold-specific updates can also be evaluated fold
        by fold, but the result remains the fold-specific extension rather than Zheng &
        van der Laan's common-update estimator.

        ``per_fold`` returns that fold's *covariate* as well as its fluctuation, because
        the two come apart when the covariate reads something fold-specific -- a linked
        working model's ``beta``, which is solved for on the fold's own rows so that no row
        contributes to any coefficient that fluctuates it.  The pieces are stitched back by
        index, so the pooled score is taken against the covariate each row was actually
        fluctuated by and stays exactly zero.  Where the covariate is the same on every
        fold, restricting and stitching returns the array that went in, value for value.
        """
        n = data.n
        observed = np.empty(n)
        # Reassembled arm by arm from whatever arms the nuisance fit carries, rather than
        # from a hardcoded pair, so a fold-targeted fit needs no change per arm count.
        arms = {level: np.empty(n) for level in nuisance.outcome.arms}
        fold_records: list[FoldFluctuation] = []
        pieces: list[tuple[IntArray, Submodel]] = []
        absolute_score_weight_parts: list[FloatArray] = []
        masses = []
        reasons: list[str] = []
        iterations = 0

        for _, test in nuisance.folds:
            fold_submodel, fold_fluctuation = per_fold(test)
            pieces.append((test, fold_submodel))
            if fold_fluctuation.absolute_score_weights is None:  # pragma: no cover - invariant
                raise RuntimeError("a newly solved fold did not retain its absolute score weights")
            absolute_score_weight_parts.append(fold_fluctuation.absolute_score_weights)
            observed[test] = fold_fluctuation.targeted.observed
            for level, values in fold_fluctuation.targeted.arms.items():
                arms[level][test] = values
            fold_records.append(
                FoldFluctuation(
                    index=test,
                    epsilon=fold_fluctuation.epsilon,
                    score=fold_fluctuation.score,
                    converged=fold_fluctuation.converged,
                    n_iter=fold_fluctuation.n_iter,
                    trace=fold_fluctuation.trace,
                    score_scale=fold_fluctuation.score_scale,
                )
            )
            masses.append(float(data.weights[test].sum()))
            reasons.append(fold_fluctuation.failure or "unknown")
            iterations += fold_fluctuation.n_iter

        targeted = InitialFit(observed, arms)
        weights_array = np.asarray(masses)
        epsilon = np.average(
            np.vstack([record.epsilon for record in fold_records]), axis=0, weights=weights_array
        )
        scaled = nuisance.scaler.scale(data.outcome)
        submodel = stitch(pieces, n)
        score = score_columns(
            scaled, targeted.observed, submodel.observed, data.weights, data.observed
        )
        scale = score_scale(submodel.observed, data.weights, data.observed)
        score_before = score_columns(
            scaled, nuisance.outcome.observed, submodel.observed, data.weights, data.observed
        )

        # Per-fold solves run with warn=False so ten folds cannot emit ten warnings.
        # That left a fold-targeted fit able to fail in three folds of ten and say
        # nothing at all, since the pooled score can still look solved: each fold's
        # score is near zero on its own rows and the failures average out. Report the
        # count once, naming the modes.
        failed = [i for i, record in enumerate(fold_records) if not record.converged]
        modes = sorted({reasons[i] for i in failed})
        if failed:
            warnings.warn(
                f"{len(failed)} of {len(fold_records)} fold(s) did not converge in the "
                f"{group!r} targeting step ({', '.join(modes)}). The pooled score "
                "can still look solved because each fold's score is near zero on its own "
                "rows; inspect res.fluctuations[group].folds for the per-fold detail.",
                ConvergenceWarning,
                stacklevel=3,
            )

        return submodel, Fluctuation(
            epsilon=epsilon,
            targeted=targeted,
            score=score,
            converged=bool(relative_score(score, scale) <= self.tol),
            n_iter=iterations,
            # These are independent fold solves, not one iteration trajectory.
            trace=(),
            method="iterative" if self.targeting == "iterative" else "one_step",
            names=submodel.names,
            score_scale=scale,
            folds=tuple(fold_records),
            score_initial=score_before,
            n_solver_calls=len(fold_records),
            failure=dominant_failure(reasons, failed),
            # Each fold solved its own projection, so there is no single beta the pooled
            # covariate was built at; the per-fold ones live on the pieces that were
            # stitched, and the *reported* coefficients come from the stitched fit.
            projection=None,
            # Keep each fold solver's own scoring weights.  In particular,
            # ``cv_evaluation=True`` normalises observation-weight mass inside a fold;
            # recomputing from ``data.weights`` here would restore the unequal masses the
            # fitted validation-risk objective deliberately removed.  Row order is not
            # needed by the concentration diagnostic, while column order is shared by the
            # stitched submodels and checked above.
            absolute_score_weights=np.vstack(absolute_score_weight_parts),
        )

    @staticmethod
    def _parameter_axis(
        data: CausalData,
        regimes: RegimeSet | None,
        policies: PolicySet | None,
        incremental: IPSISet | None,
        msm: MSMSet | None,
    ) -> tuple[tuple[float, ...], dict[float, Any]]:
        """The codes this fit's parameters are keyed by, and what to report them as.

        Exactly one of the four sources is live, which :meth:`_validate_settings` and
        :meth:`_check_policies` have already established: a fit cannot declare two of the
        keywords, and a continuous treatment must declare ``policies=``.

        The working model's codes index its *terms*, not its arms -- which is the whole of
        what makes ``msm`` a fourth axis rather than a target on the arm axis.
        """
        if msm is not None:
            return msm.codes, dict(msm.labels)
        if policies is not None:
            return policies.codes, dict(policies.labels)
        if incremental is not None:
            return incremental.codes, dict(incremental.labels)
        if regimes is not None:
            return regimes.codes, dict(regimes.labels)
        return data.arm_codes, {arm: data.arm_label(arm) for arm in data.arm_codes}

    def _corrections(
        self,
        data: CausalData,
        nuisance: NuisanceEstimates,
        fluctuation: Fluctuation,
        targeted: InitialFit,
        scaled: FloatArray,
    ) -> dict[float, FloatArray] | None:
        """``D*_Q + D*_g`` per arm for a doubly-robust fit, ``None`` for every other."""
        parts = correction_parts(data, nuisance, fluctuation, targeted, scaled)
        return None if parts is None else parts.total()

    def _estimates_for(
        self,
        data: CausalData,
        nuisance: NuisanceEstimates,
        group: TargetGroup,
        submodel: Submodel,
        fluctuation: Fluctuation,
        requested: Sequence[str],
        alpha_sig: float,
        reference: float,
        index: IntArray | None = None,
        drop_undefined: bool = False,
        natural_course: ArmMean | None = None,
    ) -> dict[str, ParameterEstimate]:
        """Build every estimand that this fluctuation supports.

        ``index`` restricts every input to one validation fold, which is what the
        fold-evaluated CV-TMLE needs; ``None`` uses the whole sample.  Weights are
        renormalised within the fold so that the fold's estimate and influence curve are
        exactly what a standalone fit on those rows would produce -- the package's
        convention is mean-one weights, and a fold's slice of a globally normalised
        vector does not satisfy it.

        ``natural_course`` is the joint route's natural-course mean on the rows ``index``
        selects, solved by its own fluctuation.  The joint route admits no fold evaluation,
        so ``index`` selects a baseline stratum there, and the mean is that stratum's.
        """
        scaler = nuisance.scaler
        scaled = scaler.scale(data.outcome)
        targeted = fluctuation.targeted
        weights, observed = data.weights, data.observed
        treatment, cluster, n = data.treatment, data.cluster, data.n
        regimes = nuisance.regimes
        policies = nuisance.policies
        # Already the *targeted* tilt: `solve_with_mechanism` returns a NuisanceEstimates
        # carrying the fluctuated mechanism, and it is that one which reaches here.
        incremental = nuisance.incremental
        msm = nuisance.msm
        # The two terms doubly-robust inference subtracts, built from the arrays the
        # alternation exited at: the refitted reductions and the *targeted* mechanism, both
        # of which live on the fluctuation rather than on the nuisances. `None` for every
        # other fit, and then `counterfactual_means` is untouched character for character.
        corrections = self._corrections(data, nuisance, fluctuation, targeted, scaled)
        if index is not None:
            scaled = scaled[index]
            targeted = _slice_fit(targeted, index)
            submodel = restrict(submodel, index)
            weights = weights[index]
            weights = weights / weights.mean()
            observed = observed[index]
            treatment = treatment[index]
            cluster = None if cluster is None else cluster[index]
            regimes = None if regimes is None else regimes.subset(index)
            policies = None if policies is None else policies.subset(index)
            incremental = None if incremental is None else incremental.subset(index)
            msm = None if msm is None else msm.subset(index)
            corrections = (
                None
                if corrections is None
                else {arm: values[index] for arm, values in corrections.items()}
            )
            n = int(index.size)

        # On a regime, shift, tilt or working-model fit the parameter axis is that rather
        # than the arm, so the context is keyed by that code and labelled with those names.
        # The five cases are the same shape on purpose --
        # see TargetContext.arms. `data.arm_label` is not reached on a continuous fit,
        # where it would raise.
        codes, labels = self._parameter_axis(data, regimes, policies, incremental, msm)
        context = TargetContext(
            scaled=scaled,
            targeted=targeted,
            submodel=submodel,
            treatment=treatment,
            weights=weights,
            observed=observed,
            scaler=scaler,
            n=n,
            cluster=cluster,
            alpha_sig=alpha_sig,
            arms=codes,
            arm_labels=labels,
            reference=reference,
            regimes=None if regimes is None else regimes.values,
            corrections=corrections,
            policies=None if policies is None else policies.design,
            policy_mixing=None if policies is None else policies.mixing,
            incremental=incremental,
            msm_design=None if msm is None else msm.design,
            msm_weights=None if msm is None else msm.weights,
            msm_link="identity" if msm is None else str(msm.link),
            always_label=(
                regimes is not None
                or policies is not None
                or incremental is not None
                or msm is not None
            ),
            # The stacked natural-course estimator declares the raw second moment, so its variance,
            # covariance and contrasts all read one rule. ``make_estimate`` owns the map
            # from that rule to the stored variance.
            covariance_rule=(
                "second_moment" if group == "natural_course" and self.cross_fit else "centered"
            ),
            natural_course=natural_course,
        )
        # One context per fluctuation, shared by every target in the group: the
        # mean-group estimands are different functionals of the same targeted
        # distribution, and `context.means` computes the counterfactual means once.
        out: dict[str, ParameterEstimate] = {}
        target_group = "mean" if group == "natural_course" else group
        for target in targets_for(target_group, requested):
            try:
                # One target is one functional, not one number: with K arms `ey` is a mean
                # per arm and `ate` a contrast per non-reference arm, and each comes back
                # under its own name.
                for estimate in target.build(context):
                    out[estimate.name] = estimate
            except ValueError:
                # A target that declares `undefined_when` may legitimately fail on a
                # subsample; anything else failing is a bug and must not be swallowed.
                if not (drop_undefined and target.undefined_when):
                    raise
        return out

    def _stratum_estimates(
        self,
        data: CausalData,
        nuisance: NuisanceEstimates,
        group: TargetGroup,
        marginal_submodel: Submodel,
        targeting_submodel: Submodel,
        fluctuation: Fluctuation,
        requested: Sequence[str],
        alpha_sig: float,
        reference: float,
        natural_course: Mapping[int, ArmMean] | None = None,
    ) -> tuple[dict[str, ParameterEstimate], dict[str, int]]:
        """Conditional plug-ins and full-sample influence curves for every stratum.

        ``natural_course`` holds the joint route's natural-course mean of each stratum,
        keyed by stratum code (:meth:`_stratum_natural_course`).

        The second mapping gives the stratum code of each estimate, which is the row set
        it reads. :meth:`_retarget_detailed` passes it to the stamp, which sets the Student
        t reference of a clustered stratum estimate from the stratum's own cluster count.

        A linked working model reads the stratum fluctuation nested on the marginal one
        (:attr:`~cleverly.fluctuation.Fluctuation.stratified`).  Every other group solved
        its blocks in the one fluctuation.
        """
        assert data.strata is not None
        stratum_fluctuation = stratum_source(fluctuation)
        width = marginal_submodel.dim
        if targeting_submodel.dim != width * data.n_strata:
            raise RuntimeError(
                "the stratified targeting submodel does not contain one base block per stratum"
            )
        out: dict[str, ParameterEstimate] = {}
        codes: dict[str, int] = {}
        for code, probability in enumerate(stratum_probabilities(data)):
            index = np.flatnonzero(data.strata == code).astype(np.int64)
            block = slice(code * width, (code + 1) * width)
            # Undo I_s / p_s before the ordinary target builder renormalises weights in
            # the subset.  Its resulting curve is on the n_s-row empirical scale; the
            # n/n_s embedding below restores I_s D_s / P_n(S=s), the full-law gradient.
            conditional_submodel = Submodel(
                unscale_block(targeting_submodel.observed[:, block], probability),
                {
                    arm: unscale_block(values[:, block], probability)
                    for arm, values in targeting_submodel.arms.items()
                },
                marginal_submodel.names,
                group,
                dict(marginal_submodel.arm_columns),
                dict(marginal_submodel.contrast_columns),
            )
            estimates = self._estimates_for(
                data,
                nuisance,
                group,
                conditional_submodel,
                stratum_fluctuation,
                requested,
                alpha_sig,
                reference,
                index=index,
                natural_course=None if not natural_course else natural_course[code],
            )
            label = data.stratum_label(code)
            for estimate in estimates.values():
                name = stratum_alias(estimate.name, label)
                curve = embed_stratum_curve(estimate.influence_curve, index, data.n)
                out[name] = make_estimate(
                    name,
                    estimate.psi,
                    curve,
                    n=data.n,
                    cluster=data.cluster,
                    scale=estimate.scale,
                    alpha=estimate.alpha,
                    log_psi=estimate.log_psi,
                    covariance_rule=estimate.covariance_rule,
                )
                codes[name] = code
        return out, codes

    def _stratum_natural_course(
        self,
        data: CausalData,
        nuisance: NuisanceEstimates,
        marginal_submodel: Submodel,
        targeting_submodel: Submodel,
        fluctuation: Fluctuation,
    ) -> dict[int, ArmMean]:
        """The joint route's natural-course mean inside each stratum, on the stratum's rows.

        The natural-course fluctuation solved one block per stratum, so each stratum's
        block, un-scaled, is that stratum's natural-course covariate.  The mean and its curve
        are on the stratum's own rows and weights, as :meth:`_estimates_for` reads every
        stratum target.
        """
        assert data.strata is not None
        width = marginal_submodel.dim
        scaled = nuisance.scaler.scale(data.outcome)
        out: dict[int, ArmMean] = {}
        for code, probability in enumerate(stratum_probabilities(data)):
            index = np.flatnonzero(data.strata == code).astype(np.int64)
            block = slice(code * width, (code + 1) * width)
            covariate = Submodel(
                unscale_block(targeting_submodel.observed[:, block], probability),
                {
                    arm: unscale_block(values[:, block], probability)
                    for arm, values in targeting_submodel.arms.items()
                },
                marginal_submodel.names,
                "natural_course",
            )
            weights = data.weights[index]
            out[code] = natural_course_mean(
                scaled[index],
                _slice_fit(fluctuation.targeted, index),
                restrict(covariate, index),
                weights / weights.mean(),
                data.observed[index],
            )
        return out

    def _bootstrap_point_estimates(
        self,
        data: CausalData,
        intermediate_value: float | None,
        estimands: tuple[str, ...],
    ) -> Mapping[str, float]:
        """One bootstrap replicate: a full refit, point estimates only.

        Goes through :meth:`_nuisances` rather than :meth:`_fit_nuisances` so that a
        variant which *selects* a nuisance model repeats that selection in every
        replicate -- otherwise the bootstrap would understate the variability the
        selection itself contributes.

        A replicate repeats the cross-fitting draws for the same reason, which is why the
        loop is here rather than around the caller: the bootstrap has to resample the
        estimator that was reported, and under ``repeats=R`` that estimator is the median
        of ``R`` draws, whose fold noise is already reduced. Bootstrapping a single
        draw instead would attribute variability to the report that it does not
        have.  It costs ``B * R`` fits, which is the honest price of the two settings
        together.

        ``estimands`` is the point fit's resolved tuple.  Resolving ``self.estimands``
        again on a replicate would undo what the point fit's preflight dropped: an
        ``estimands="all"`` fit with ``delta=`` or ``intermediate=`` drops ``ey_obs``,
        ``par`` and ``paf``, and a replicate that asked for them again failed, or
        published draws for names the fit never reported.
        """
        scaler = self._scaler(data)
        draws = self._repeat_draws(data, estimands)
        fold_draws = [folds for folds, _ in draws]
        config = self._config(data, estimands, scaler, fold_draws[0])
        per_repeat = []
        for folds, seed in draws:
            nuisance, _ = self._nuisances(
                data, folds, scaler, config, intermediate_value, seed=seed
            )
            estimates, _ = self.retarget(
                data,
                nuisance,
                estimands=estimands,
                intermediate_value=intermediate_value,
            )
            per_repeat.append(estimates)
        combined = median_estimates(per_repeat)
        return {name: estimate.psi for name, estimate in combined.items()}


def reported_mechanism(
    nuisance: NuisanceEstimates,
    fluctuation: Fluctuation,
    arms: tuple[float, ...],
) -> FloatArray:
    """The mechanism the reduced corrections were formed at, in the shape they read it in.

    A tilted mechanism is whatever the alternation solved for. Missing-outcome targeting
    keeps treatment and observation mechanisms separate; this function returns the former,
    while the latter is stored on the reduction fluctuation.

    Shared by :func:`correction_parts` and
    :func:`~cleverly.validation.drtmle.correction_check` for the same reason
    ``correction_parts`` is module level: the reported curve and the diagnostic that
    checks it must not be able to describe different mechanisms.
    """
    if fluctuation.mechanism is not None:
        return np.asarray(fluctuation.mechanism.propensity, dtype=float)
    # The carrier rule: one column exactly when the two-arm complement form applies.  A
    # composite mechanism is off the simplex and is returned whole at every arm count.
    if len(arms) == 2 and nuisance.propensity.simplex:
        return nuisance.propensity.arm(arms[1])
    return np.asarray(nuisance.propensity.values, dtype=float)


def correction_parts(
    data: CausalData,
    nuisance: NuisanceEstimates,
    fluctuation: Fluctuation,
    targeted: InitialFit,
    scaled: FloatArray,
) -> CorrectionParts | None:
    """The doubly-robust corrections at the state a fit returned; ``None`` for other fits.

    Read entirely off the fluctuation, which is where the alternation left the pieces: the
    refitted reduced regressions, the targeted mechanism and the truncation the two extra
    covariates divided by.  A curve built from ``result.nuisance`` instead would be the
    curve of a fit nobody ran -- those arrays are deliberately the *initial* ones.  Without
    the ``"Q"`` guard no mechanism was tilted and the initial one is what equation (10) was
    solved beside, so that is what the curve reads.  :func:`reported_mechanism` makes that
    choice, and it returns the **treatment** mechanism throughout: missing-outcome targeting
    keeps the observation mechanism separate, and it arrives here on
    :attr:`~cleverly.estimators.targeting.ReductionFluctuation.observation` instead.

    Module level rather than a method of :class:`TMLE`, and that is the whole point: the
    reported curve and :func:`~cleverly.validation.drtmle.correction_check`'s identity both
    come through here, so neither can end up describing a different state from the other.
    A result read back from disk has no estimator to ask, which is the other reason.

    ``guard`` comes off the record too, and is what says which corrections belong in the
    curve at all: a fit guarding one nuisance solves one of the two extra equations and
    subtracts one term.  It was read here by
    :func:`~cleverly.validation.drtmle.correction_check` and not by the curve, which is
    the partial-guard correction invariant.
    """
    reduction = fluctuation.reduction
    if reduction is None:
        return None
    mechanism = reported_mechanism(nuisance, fluctuation, reduction.reduced.arms)
    if reduction.observation is not None:
        if reduction.missingness_bound is None:
            raise ValueError("a missing-outcome reduction record has no observation bound")
        return missing_outcome_correction_parts(
            scaled,
            targeted,
            data.treatment,
            data.observed,
            reduction.reduced,
            mechanism,
            np.asarray(reduction.observation.propensity, dtype=float),
            g_bounds=reduction.bounds,
            missingness_bound=reduction.missingness_bound,
            guard=tuple(reduction.guard),
        )
    return reduced_correction_parts(
        scaled,
        targeted,
        data.treatment,
        reduction.reduced,
        mechanism,
        bounds=reduction.bounds,
        guard=tuple(reduction.guard),
    )


def _slice_fit(fit: InitialFit, index: IntArray) -> InitialFit:
    """The targeted (or initial) predictions for one subset of rows."""
    return fit.map_arms(lambda values: values[index])


def _in_report_order(
    estimates: Mapping[str, ParameterEstimate], requested: Sequence[str]
) -> dict[str, ParameterEstimate]:
    """Order reported parameters by the *target* that produced them.

    ``requested`` holds target names in registry order; ``estimates`` is keyed by
    parameter name, which is the target's name for a two-armed fit and
    ``"ate[medium vs low]"`` for a wider one.  Grouping by
    :func:`~cleverly.targets.parameter_stem` restores the registry's report order across
    groups -- the targeting steps run group by group, so the raw insertion order
    interleaves as ``ate, ey1, ey0, att, atc`` rather than the registry's
    ``ate, att, atc, ey1, ey0``.

    Within one target the parameters keep the order it emitted them in, which is arm
    order.
    """
    order = {name: position for position, name in enumerate(requested)}
    fallback = len(order)
    return {
        name: estimates[name]
        for name in sorted(estimates, key=lambda n: order.get(parameter_stem(n), fallback))
    }


def _average_over_folds(
    per_fold: Sequence[Mapping[str, ParameterEstimate]],
    supported: Sequence[str],
    indices: Sequence[IntArray],
    *,
    n: int,
    cluster: IntArray | None,
    alpha: float,
) -> dict[str, ParameterEstimate]:
    """Assemble the original fold-evaluated CV-TMLE from its fold-wise pieces.

    The point estimate is the unweighted ``1/V`` average of the fold plug-ins, matching
    Zheng & van der Laan and matching the fold weighting
    :func:`~cleverly.inference.cross_validated_variance` already uses -- so the estimate
    and its variance are weighted the same way, with no extra knob.  Observation weights
    still apply *within* a fold.  Ratios are averaged on the log scale, which is where
    their influence curve and Wald interval live, so that ``psi == exp(log_psi)`` holds
    and :attr:`~cleverly.inference.ParameterEstimate.ci` stays on the boundary-respecting
    scale.

    The variance reads the raw fold-specific curves.  The curve stored on the aggregate
    report is additionally scaled by ``n / (V n_v)`` inside fold ``v`` so its ordinary
    full-sample mean represents the equal ``1/V`` fold average even when fold sizes differ.
    A common validation update need not make any one fold's score zero, which is why
    :func:`cross_validated_variance` uses the uncentred fold second moments for rows; the
    cluster branch centres within each fold.
    """
    out: dict[str, ParameterEstimate] = {}
    dropped: list[str] = []
    n_clusters = n if cluster is None else int(np.unique(cluster).size)

    for name in supported:
        if not all(name in values for values in per_fold):
            dropped.append(name)
            continue
        parts = [values[name] for values in per_fold]
        fold_curve = np.empty(n, dtype=float)
        influence_curve = np.empty(n, dtype=float)
        for index, part in zip(indices, parts, strict=True):
            fold_curve[index] = part.influence_curve
            # ``ParameterEstimate.influence_curve`` is represented under the full
            # empirical mean.  An equal ``1/V`` average of fold estimators whose own
            # curves are averaged over ``n_v`` rows therefore needs the factor
            # ``n / (V n_v)`` on rows from fold v.  It is one only for equal folds.
            influence_curve[index] = n / (len(indices) * index.size) * part.influence_curve

        scale = parts[0].scale
        log_psi: float | None = None
        if scale == "ratio":
            mean_log = float(np.mean([part.log_psi for part in parts]))
            log_psi = mean_log
            psi = float(np.exp(mean_log))
        else:
            psi = float(np.mean([part.psi for part in parts]))

        out[name] = ParameterEstimate(
            name=name,
            psi=psi,
            influence_curve=influence_curve,
            variance=cross_validated_variance(fold_curve, indices, cluster),
            n=n,
            n_clusters=n_clusters,
            scale=scale,
            alpha=alpha,
            log_psi=log_psi,
            # Declared, not inherited from the fold pieces. The stored variance is the
            # cross-validated one above, and covariance reads the curve under the centered
            # rule. docs/technical-reference/inference.md states this contract.
            covariance_rule="centered",
        )

    if dropped:
        warnings.warn(
            f"{', '.join(dropped)} could not be evaluated inside every validation fold "
            "-- a fold with no units in the conditioning arm, or a counterfactual mean "
            "at the boundary -- so it is omitted from the cross-validated report. Fewer "
            "folds (n_folds=) would give each one more units to work with.",
            UserWarning,
            stacklevel=3,
        )
    return out


def tmle(
    Y: Any,
    A: Any,
    W: Any,
    *,
    Delta: Any = None,
    DeltaA: Any = None,
    Z: Any = None,
    obsWeights: Any = None,
    weights_type: str = "probability",
    weights_estimated: bool = False,
    id: Any = None,
    covariate_names: Sequence[str] | None = None,
    **kwargs: Any,
) -> TMLEResultSet:
    """Array-oriented entry point, mirroring ``tmle(Y, A, W, ...)`` in R.

    A thin wrapper over :class:`TMLE`; the argument names follow R's ``tmle`` package
    so existing analysis scripts translate directly.  Every keyword accepted by
    :class:`TMLE` may be passed through.  ``DeltaA`` is the treatment observation
    indicator in R ``drtmle``'s spelling, beside ``Delta`` for the outcome.

    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression, LogisticRegression
    >>> from cleverly.estimators.tmle import tmle
    >>> rng = np.random.default_rng(0)
    >>> W = rng.normal(size=(500, 3))
    >>> A = rng.binomial(1, 0.5, 500).astype(float)
    >>> Y = A + W[:, 0] + rng.normal(size=500)
    >>> res = tmle(
    ...     Y,
    ...     A,
    ...     W,
    ...     outcome_learner=LinearRegression(),
    ...     treatment_learner=LogisticRegression(max_iter=1000),
    ...     cross_fit=False,
    ... ).single()
    >>> bool(np.isfinite(res.psi("ate")))
    True
    """
    data = CausalData.from_arrays(
        Y,
        A,
        W,
        covariate_names=covariate_names,
        delta=Delta,
        weights=obsWeights,
        weights_type=weights_type,
        weights_estimated=weights_estimated,
        id=id,
        intermediate=Z,
        family=kwargs.get("family", "auto"),
        treatment_delta=DeltaA,
    )
    estimator = TMLE(**kwargs)
    return estimator.fit(data)
