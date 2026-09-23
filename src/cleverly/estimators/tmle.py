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
from typing import Any, cast, get_args

import numpy as np

from .._inference_status import InferenceStatus, supplies_inference
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
from ..fluctuation.mechanism import needs_mechanism
from ..fluctuation.submodel import Submodel, TargetGroup, restrict, stitch
from ..inference.bootstrap import Resampling, run_bootstrap
from ..inference.cluster import cluster_inference_status, cross_validated_variance
from ..inference.influence import (
    CorrectionParts,
    ParameterEstimate,
    make_estimate,
    median_estimates,
    missing_outcome_correction_parts,
    reduced_correction_parts,
    stamp_inference,
)
from ..inference.multiplier import MultiplierKind, simultaneous_bands
from ..interventions import Incremental, IPSISet, RegimeSet, Shift, ShiftSet, as_interventions
from ..interventions.base import refuse_regime_densities
from ..interventions.incremental import refuse_multi_arm_tilt
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
from ..msm import MSM, MSMSet, refuse_projection_weights
from ..provenance import data_fingerprint
from ..provenance import record as provenance_record
from ..targets import TargetContext, groups_for, parameter_stem, targets_for
from ..targets.base import stratum_alias
from ..targets.population_intervention import (
    NATURAL_COURSE_TARGET,
    POPULATION_INTERVENTION_TARGETS,
    population_intervention_refusal,
)
from ..utils.bounds import OutcomeScaler, g_bounds_for, resolve_g_bounds
from ..utils.frames import is_dataframe
from ._nuisance import NuisanceEstimates, RepeatFit, fit_nuisances
from .base import (
    CVTargeting,
    TMLEConfig,
    TMLEResult,
    TMLEResultSet,
    attach_bootstrap,
    resolve_estimands,
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
    and so is the natural-course mean, which has its own contract.

    Parameters
    ----------
    data : CausalData
        The prepared data, read for its observation mask, intermediate, and treatment kind.
    estimands : tuple of str
        The resolved estimand names.
    cross_fit : bool
        Whether the nuisances are cross-fitted.
    axis : {"arm", "regime", "shift", "ipsi", "msm"}
        What the fit's parameters are indexed by.

    Returns
    -------
    bool
        True when the outcome is missing for some rows, the nuisances are cross-fitted,
        the parameters are indexed by a discrete treatment's arms, no intermediate is
        declared, and the natural-course mean is not requested.
    """
    return (
        data.has_missing_outcome
        and cross_fit
        and axis == "arm"
        and not data.has_intermediate
        and not data.is_continuous_treatment
        and NATURAL_COURSE_TARGET not in estimands
    )


#: The arm-indexed estimands the stacked MAR contract admits, before the outcome family
#: and arm count narrow them.
_ARM_INDEXED_ADMITTED = frozenset({"ey", "ey0", "ey1", "ate", "rr", "or"})

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
    "shift": "shift",
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
        Run a targeted bootstrap with this many replicates (R's ``B``).
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
        and nothing fitted, so the status can be determined without learner results. Three callers
        ask it: ``_retarget_detailed`` stamps the estimates, ``TMLEResult.__setstate__``
        re-stamps a restored artifact, and ``variable_importance`` refuses before its
        first fit. An override that finds more than one status resolves them with
        :func:`~cleverly._inference_status.precedent_status`.

        Parameters
        ----------
        data : CausalData
            The prepared data the estimates are fitted on.

        Returns
        -------
        str
            One of :data:`~cleverly.inference.influence.InferenceStatus`. The ordinary
            estimator returns what
            :func:`~cleverly.inference.cluster.cluster_inference_status` gives the cluster
            labels: ``"influence_curve"`` on an unclustered fit, and a clustered status
            on a cross-fitted fit at unequal cluster sizes, in rows or in weight mass,
            or on a fit with few clusters in total or in one baseline stratum.
        """
        return cluster_inference_status(
            data.cluster,
            cross_fit=self.cross_fit,
            strata=data.strata,
            weights=data.weights if data.is_weighted else None,
        )

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
        shifts: Sequence[Shift] | None = None,
        incremental: Sequence[Incremental] | None = None,
        msm: MSM | None = None,
        density_bins: int = 20,
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
        self.shifts = tuple(shifts or ())
        self.incremental = tuple(incremental or ())
        self.msm = msm
        self.density_bins = density_bins
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
            raise ValueError(
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
                ("shifts=", self.shifts),
                ("incremental=", self.incremental),
            )
            if value
        ]
        if len(declared) > 1:
            raise ValueError(
                f"{' and '.join(declared)} each declare what this fit's counterfactuals "
                "are -- a regime assigns an arm from W alone, a shift moves the dose the "
                "unit actually received, and an incremental intervention tilts the odds "
                "of the mechanism that was already there -- and one fluctuation cannot "
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
        if self.msm is not None and (self.interventions or self.shifts or self.incremental):
            other = (
                "interventions="
                if self.interventions
                else "shifts="
                if self.shifts
                else "incremental="
            )
            raise ValueError(
                f"msm= and {other} cannot be combined. A working model summarises the "
                "counterfactual means with p score equations, one per term, and "
                f"{other} replaces what those means are; one fluctuation cannot solve "
                "both. A working model over declared regimes is a coherent estimand and "
                "is not implemented -- its design would have to be indexed by regime "
                "rather than by arm. docs/roadmap.md F17 tracks this stop."
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
            raise ValueError(reason)

    def _cross_fit_policy_reason(self) -> str | None:
        """Why this estimator's declared fold policy cannot run, or ``None``.

        Asked twice on purpose.  :meth:`_validate_settings` asks it at construction, so a
        declaration that cannot run is refused where it was written.  A fit asks it again
        in :meth:`_resolve_estimands_for_data`, because two supported operations reach a
        fit without running ``__init__``: :meth:`refit` copies an estimator, and a result
        restored from a pickle written by an earlier version arrives with whatever policy
        that version allowed.  Neither may fit under a policy this one refuses.
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
                # Deliberately ``!= "oat"`` and not
                # :func:`~cleverly.estimators.ctmle.is_selector_strategy`. The only way to
                # reach this line without a ``strategy`` attribute is a pickle from a
                # version that predates the field, and ``None != "oat"`` holds the fold
                # policy for it while ``is_selector_strategy(None)`` would release it.
                # Swapping the spelling would quietly relax this refusal for restored
                # artifacts.
                and getattr(self, "strategy", None) != "oat"
            ),
            option_name="cross_fit",
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
            follows ``shifts=``: a modified treatment policy moves a dose and names no
            arm, so declaring one declares the treatment continuous.

            That is a default read off another *declaration*, not off the data -- a
            column at fifteen distinct values could reasonably be read either way, and
            :class:`~cleverly.data.CausalData` refuses to guess from the level count for
            exactly that reason.  Pass this explicitly to override, including to get the
            arm-coded refusal for a ``shifts=`` fit on a treatment that really has arms.
        weights_type, weights_estimated:
            How to read ``weights``.  Supplying weights changes the estimand to the
            causal parameter in the weight-tilted population -- see
            :mod:`cleverly.data.weighting` and
            :meth:`~cleverly.data.CausalData.from_frame`.

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
        )
        if not prepared.has_intermediate:
            return TMLEResultSet({None: self._fit_single(prepared, intermediate_value=None)})

        # The intermediate path otherwise fits shared nuisances before `_fit_single`.
        # Resolve this boundary here as well, so every unsupported intermediate
        # composition fails before any learner is fitted. The ordinary path resolves in
        # `_fit_single`, after its axis-specific structural checks have named their own
        # refusals.
        estimands = self._resolve_estimands_for_data(prepared)

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
        perturbed or dropped -- which is the exact condition
        :meth:`~cleverly.SplitPlan.validate` names as safe to reuse labels under -- so
        holding the fingerprint here would refuse the placebo and negative-control
        refutations for a change that moves no row.  What a refit cannot do is change the
        row *set*: the count check inside ``validate`` still refuses that, and
        :func:`~cleverly.validation.refute` refuses the subsampling test up front.
        :meth:`~cleverly.SplitPlan.unbound` keeps the plan's generator record, so the
        refit accepts the plan and draws its labels again to check them.
        """
        estimator = self
        plan = self.split_plan
        if plan is not None and plan.source_fingerprint is not None:
            estimator = copy.copy(self)
            estimator.split_plan = plan.unbound()
        if random_state is not None and random_state != self.random_state:
            if estimator is self:
                estimator = copy.copy(self)
            estimator.random_state = random_state
        return estimator._fit_single(data, intermediate_value=intermediate_value)

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
                ("continuous" if self.shifts else "discrete")
                if treatment_kind is None
                else treatment_kind
            ),
        )

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
        self._check_shifts(data)
        self._check_incremental(data)
        estimands = self._resolve_estimands_for_data(data)
        if data.has_strata and data.is_continuous_treatment and self.msm is not None:
            raise NotImplementedError(
                "continuous MSMs do not yet support baseline strata; conditional dose "
                "projections need a stratum-specific density-ratio targeting construction. "
                "Fit the marginal MSM projection."
            )
        if data.has_strata and (self.cv_evaluation or self.targeting_scheme == "fold"):
            raise NotImplementedError(
                "baseline strata currently use one joint pooled fluctuation. "
                "cv_evaluation=True or targeting_scheme='fold' would require the "
                "stratum probabilities and conditional treatment shares to be rebuilt "
                "inside every validation fold; use the default pooled targeting scheme"
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
        # ``ey_obs`` shares ``par`` and ``paf``'s fate beside either declaration, but no
        # longer their set: its missing-outcome score equation is implemented and theirs
        # are not.  It has to rejoin them *here*, because ``estimands="all"`` asks for
        # whatever the data supports rather than for a named list, and dropping a target
        # the composition cannot express is what that request means.  Refusing instead
        # would make ``estimands="all"`` fail on a design that worked before ``ey_obs``
        # left the set.  An explicit list still refuses, below.
        droppable = population_intervention | ({NATURAL_COURSE_TARGET} & set(estimands))
        if (
            droppable
            and self.estimands == "all"
            and (data.has_missing_outcome or data.has_intermediate)
        ):
            estimands = tuple(name for name in estimands if name not in droppable)
            population_intervention = frozenset()
        if population_intervention and data.has_missing_outcome:
            raise population_intervention_refusal(population_intervention, declaration="delta=")
        # ``ey_obs`` left ``POPULATION_INTERVENTION_TARGETS`` when its missing-outcome
        # score equation landed, and that set was the only thing refusing it beside a
        # controlled mediator intervention.  The natural-course boundary below cannot
        # cover this: it returns early on a complete outcome, which is exactly the case
        # that slipped.  Without this line the fit runs, publishes the plain empirical
        # mean of Y once per level of Z, and labels each copy a controlled direct
        # effect -- a different estimand reported under a name it did not earn.
        beside_intermediate = population_intervention | ({NATURAL_COURSE_TARGET} & set(estimands))
        if beside_intermediate and data.has_intermediate:
            # A ``CleverlyError`` like the ``delta=`` refusal three lines above, and for
            # the reason docs/architecture-invariants.md gives: a caller must not have to
            # catch an implementation-language exception beside a library one for two
            # refusals of the same shape. The reason differs, so the sentence is written
            # here rather than built by ``population_intervention_refusal``.
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
                raise ValueError(
                    "cv_evaluation=True does not yet support "
                    f"{unsupported}: averaging a nonlinear parameter over folds changes "
                    "its gradient fold by fold, so the ordinary mean/MSM fluctuation no "
                    "longer solves that cross-validated score. The stacked-validation "
                    "report (cv_evaluation=False) remains supported; request linear "
                    "levels/contrasts or ATT/ATC for fold-wise evaluation."
                )

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

        # A simultaneous band is a joint confidence statement, so a fit that supplies no
        # inference builds none.  Skipped rather than raised, because ``simultaneous``
        # defaults to ``True``: raising would stop every default-configured selector-path
        # collaborative fit over an output RM12 refuses to report anyway.  The omission is
        # not silent -- ``summary()`` prints the reason, and ``simultaneous_bands()``
        # called directly still refuses, because that is an explicit request.
        if self.simultaneous and len(estimates) > 1 and supplies_inference(result.inference_status):
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
                lambda replicate: self._bootstrap_point_estimates(replicate, intermediate_value),
                n_replicates=self.n_bootstrap,
                resampling=self.bootstrap_resampling,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
            result = attach_bootstrap(result, bootstrap)

        return result

    def _cv_detail(
        self, details: Sequence[CVTargeting | None], *, cluster: IntArray | None
    ) -> CVTargeting | None:
        """One fold-level report for the whole fit, however many draws it combines.

        The fields split by what they *are*.  ``pooled``, ``canonical`` and ``variance``
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
        canonical = median_estimates([detail.canonical for detail in present])
        return replace(
            first,
            repeats=len(present),
            pooled=median_estimates([detail.pooled for detail in present]),
            canonical=canonical,
            variance={name: value.variance for name, value in canonical.items()},
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
        """Resolve targets and enforce every refusal that precedes fitting.

        The declared fold policy is checked first and without reading the data, because a
        fit can arrive here without having run ``__init__``: :meth:`refit` copies an
        estimator, and an estimator restored from a pickle written by an earlier version
        carries whatever policy that version allowed.  The working model's projection-weight
        declaration is checked next for the same reason: ``MSM`` checks it when it is
        declared, and a restored or modified model can carry one this version refuses.
        Each ``Stochastic`` regime's density declaration follows, for the same reason and
        before any density is evaluated (roadmap row RM25).

        The natural-course contract runs next, because it resolves the target list the
        arm-indexed missing-outcome contract then reads.  The refusal of every other
        cross-fitted missing-outcome target follows them.  All three name a narrower
        surface than the outcome-scale rule below, so each keeps its own sentence and the
        general rule catches what is left.
        """
        reason = self._cross_fit_policy_reason()
        if reason is not None:
            raise ValueError(
                f"{reason}. This fit was configured under a fold policy this version "
                "refuses, which a restored result or a copied estimator can still carry"
            )
        if self.msm is not None:
            refuse_projection_weights(self.msm)
        refuse_regime_densities(self.interventions)
        estimands = self._resolve_natural_course_contract(data)
        self._resolve_arm_indexed_missing_contract(data, estimands)
        self._refuse_cross_fitted_missing_off_contract(data, estimands)
        self._refuse_unbounded_cross_fitted_scale(data)
        return estimands

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
        before this method runs. A requested population-intervention target keeps its
        F20 refusal. The natural-course mean is arm-axis only, and its contract refuses
        ``intermediate=``, so it cannot reach the check below.
        """
        if not (self._assessment_method == "tmle" and self.cross_fit and data.has_missing_outcome):
            return
        if self._axis == "arm" and not data.has_intermediate:
            return
        if POPULATION_INTERVENTION_TARGETS.intersection(estimands):
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

    def _refuse_unbounded_cross_fitted_scale(self, data: CausalData) -> None:
        """Refuse a cross-fitted continuous outcome whose scale the held-out rows set.

        :meth:`_scaler` maps a continuous outcome onto ``[0, 1]`` before ``Qbar`` is
        fitted, and with ``q_bounds=None`` it takes the endpoints from the observed
        outcomes of the whole sample.  Every fold's nuisance is then fitted on a scale the
        rows it predicts helped choose, so the split no longer separates what a fold saw
        from what it is scored on.  Declaring the support is the remedy that keeps the
        separation, and fitting in sample is the one that needs no support.
        """
        if not self.cross_fit or data.family == "binomial" or self.q_bounds is not None:
            return
        raise CapabilityError(
            f"a cross-fitted fit of a continuous outcome ({data.outcome_name}, "
            f"family={data.family!r}) needs a declared q_bounds. With q_bounds=None the "
            "outcome scale is taken from every observed outcome, held-out rows included, "
            "so each fold's nuisance is fitted on a scale the rows it predicts helped set, "
            "and no shipped result covers that scale "
            "(docs/technical-reference/cv-tmle.md, fold and outcome-scale rules). Declare the "
            "known outcome support (Targeting(q_bounds=(lower, upper))). Without a known "
            f"finite support, {_IN_SAMPLE_ARM_INDEXED_REMEDY}"
        )

    def _on_arm_indexed_stacked_surface(self, data: CausalData, estimands: tuple[str, ...]) -> bool:
        """Whether the arm-indexed stacked MAR contract governs this ordinary TMLE fit.

        A requested population-intervention target keeps its own F20 refusal, which the
        fit raises later with its own sentence. :class:`~cleverly.DRTMLE` raises its own
        cross-fitted missing-outcome refusal.
        """
        return (
            _is_arm_indexed_missing_crossfit(
                data, estimands, cross_fit=self.cross_fit, axis=self._axis
            )
            and not POPULATION_INTERVENTION_TARGETS.intersection(estimands)
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
            admitted = [
                name
                for name in resolve_estimands("all", data.family, data.n_arms)
                if name in _ARM_INDEXED_ADMITTED
            ]
            refuse(
                f"no audited result covers {conditional}. The default estimand list and "
                "estimands='all' include att and atc, and the fit drops no requested "
                f"estimand silently. Request estimands from {admitted}"
            )
        if data.family != "binomial" and self.q_bounds is None:
            refuse(
                "a continuous outcome with q_bounds=None takes its scale from every observed "
                "outcome, held-out rows included, and no reviewed result covers that scale "
                "(docs/technical-reference/cv-tmle.md, fold and outcome-scale "
                "rules). Declare q_bounds equal to the known outcome support "
                "(Targeting(q_bounds=(lower, upper))). Without a known finite support, "
                f"{_IN_SAMPLE_ARM_INDEXED_REMEDY}"
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
        """Resolve targets and enforce the supported natural-course compositions."""
        estimands = resolve_estimands(self.estimands, data.family, data.n_arms, axis=self._axis)
        if not data.has_missing_outcome:
            return estimands
        if self.estimands == "all":
            # The default remains the set of jointly targetable parameters.  This is a
            # deliberately scalar construction, while PAR/PAF remain tracked by F20.
            return tuple(
                name
                for name in estimands
                if name != NATURAL_COURSE_TARGET and name not in POPULATION_INTERVENTION_TARGETS
            )
        if NATURAL_COURSE_TARGET not in estimands:
            return estimands

        def refuse(reason: str) -> None:
            raise CapabilityError(
                "NaturalCourseMean with missing outcomes currently supports one scalar "
                f"TMLE under its audited implementation contracts; {reason}"
            )

        if len(estimands) != 1:
            refuse("request ey_obs by itself; joint targeting is not implemented")
        if self._assessment_method != "tmle":
            refuse(
                "use ordinary TMLE; collaborative and doubly robust estimator variants "
                "need separate targeting and inference results"
            )
        if not data.is_binary_treatment:
            refuse("the treatment must have exactly two arms")
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
            refuse("set n_bootstrap=0; bootstrap inference is not implemented")
        if data.is_weighted:
            refuse("observation weights are not implemented")
        if data.cluster is not None:
            refuse("clustered inference is not implemented")
        if data.has_strata:
            refuse("baseline strata are not implemented")
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
            data, folds, subject=f"cross-fitted {names.get(self._assessment_method, 'TMLE')}"
        )

    def _check_training_support(
        self, data: CausalData, folds: Sequence[Folds], *, subject: str
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
                    f"treatment mechanism; {_IN_SAMPLE_ARM_INDEXED_REMEDY}"
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
        remedy = _POST_DRAW_REMEDY.format(remedy=_IN_SAMPLE_ARM_INDEXED_REMEDY)
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
                    f"{subject} cannot fit its nuisances because repeat {repeat}, fold "
                    f"{fold}'s training complement contains no {described}. {remedy}"
                )
        complement_gap = self._super_learner_complement_shortfall(data, folds)
        if complement_gap is not None:
            role, repeat, fold, described, count = complement_gap
            raise DataError(
                f"{subject} cannot fit the {role} learner because repeat {repeat}, fold "
                f"{fold}'s training complement holds {count} {described}. "
                f"{_SUPER_LEARNER_INNER_SPLIT_RULE.format(role=role)} {remedy}"
            )

    def _preflight_natural_course_folds(
        self,
        data: CausalData,
        estimands: tuple[str, ...],
        folds: Sequence[Folds],
    ) -> None:
        """Check natural-course response and outcome support before nuisance fitting."""
        if not _is_natural_course(data, estimands) or any(draw.is_single for draw in folds):
            return
        observed = np.asarray(data.observed, dtype=bool)
        n_response = int(np.count_nonzero(observed))
        if n_response < 2 or observed.size - n_response < 2:
            kind = "respondent" if n_response < 2 else "nonrespondent"
            count = n_response if n_response < 2 else observed.size - n_response
            raise DataError(
                "cross-fitted NaturalCourseMean needs at least two of each response kind, "
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
                        "cross-fitted NaturalCourseMean needs at least two respondents "
                        f"with each outcome, and the sample has {count} respondent(s) "
                        f"with outcome {value:g}. Every partition leaves some training "
                        "complement without that outcome, so no fold count or random_state "
                        f"can fit the outcome regression; {_IN_SAMPLE_NATURAL_COURSE_REMEDY}"
                    )
        sample_gap = self._super_learner_sample_shortfall(data, fits_treatment=False)
        if sample_gap is not None:
            role, described, count = sample_gap
            raise DataError(
                f"cross-fitted NaturalCourseMean cannot fit the {role} learner: the sample "
                f"holds {count} {described}. "
                f"{_SUPER_LEARNER_INNER_SPLIT_RULE.format(role=role)} Collect more "
                "observations or use a learner without that inner split."
            )
        support: list[tuple[str, FloatArray | BoolArray, FloatArray | BoolArray]] = [
            ("response", observed, np.array([False, True])),
        ]
        if data.family == "binomial":
            support.append(("outcome", np.where(observed, outcome, -1.0), np.array([0.0, 1.0])))
        for repeat, draw in enumerate(folds):
            gap = missing_training_support(draw, support)
            if gap is not None:
                fold, name, missing = gap
                if name == "response":
                    described = "respondent" if bool(missing[0]) else "nonrespondent"
                else:
                    described = f"respondent with outcome {float(missing[0]):g}"
                raise DataError(
                    "cross-fitted NaturalCourseMean cannot fit its response and outcome "
                    f"nuisances because repeat {repeat}, fold {fold}'s training complement "
                    f"contains no {described}. "
                    + _POST_DRAW_REMEDY.format(remedy=_IN_SAMPLE_NATURAL_COURSE_REMEDY)
                )
        complement_gap = self._super_learner_complement_shortfall(data, folds, fits_treatment=False)
        if complement_gap is not None:
            role, repeat, fold, described, count = complement_gap
            raise DataError(
                "cross-fitted NaturalCourseMean cannot fit the "
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
        if self.msm is not None:
            return "msm"
        if self.shifts:
            return "shift"
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
            raise ValueError(
                "incremental= and intermediate= are not combined. A controlled direct "
                "effect under a tilt of the treatment mechanism is a parameter this "
                "package has not written down, and reporting one would mean guessing at "
                "its influence function."
            )

    def _check_shifts(self, data: CausalData) -> None:
        """Refuse a shift the treatment cannot carry, and a dose with no policy declared.

        Both directions matter.  A shift of an arm-coded treatment is a ``Rule`` written
        the wrong way round -- ``d(a, w) = a + 1`` on arms ``{0, 1}`` assigns an arm that
        does not exist -- and a continuous treatment with no ``shifts=`` has no estimand
        at all, since every registered arm-indexed target names a level it has none of.
        """
        if self.shifts and not data.is_continuous_treatment:
            raise DataError(
                f"shifts= declares a modified treatment policy, which needs a continuous "
                f"treatment, but {data.treatment_name} has arms "
                f"{list(data.treatment_levels)}. A shift of a discrete treatment assigns "
                "an arm as a function of (A, W), which is a Rule -- pass it to "
                "interventions=. To treat this column as a dose, build the CausalData "
                "with treatment_kind='continuous'."
            )
        if data.is_continuous_treatment and not self.shifts and self.msm is None:
            raise DataError(
                f"{data.treatment_name} was declared continuous, so it has no arms and "
                "none of the arm-indexed estimands name a parameter it has. Say which "
                "doses to compare with shifts=[Shift(delta, cap=...), ...]; "
                "Shift(0.0, cap=None) is the natural course, whose mean is E[Y], or "
                "declare an MSM with a dose integration grid."
            )

    def _reference_arm(
        self,
        data: CausalData,
        regimes: RegimeSet | None = None,
        shifts: ShiftSet | None = None,
    ) -> float:
        """The arm -- or regime, or shift -- code every contrast is taken against.

        On a regime or shift fit the contrasts are between *regimes* (or *shifts*), so
        ``reference=`` names one of them and the code returned indexes
        :class:`~cleverly.interventions.RegimeSet` or
        :class:`~cleverly.interventions.ShiftSet`.  :meth:`_regimes` and
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
        if shifts is not None:
            return shifts.reference
        if regimes is not None:
            return regimes.reference
        if self.shifts:
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
        self._preflight_missing_outcome_folds(data, estimands, folds)
        self._preflight_training_support(data, estimands, folds)
        return tuple(zip(folds, seeds, strict=True))

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
                if data.has_missing_outcome
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
            shifts=self.shifts,
            shift_reference=None if self.reference is None else str(self.reference),
            incremental=self.incremental,
            incremental_reference=None if self.reference is None else str(self.reference),
            density_bins=self.density_bins,
            msm=msm,
            companion=companion,
            n_jobs=self.n_jobs,
            fit_treatment=fit_treatment,
        )
        # Evaluated once and carried with the fits, so that every reuse -- retarget, and
        # so the truncation curve, the MNAR tilt, the omitted-variable bound -- targets
        # the regimes and the working model this fit declared, without re-running the
        # caller's rules or its design.
        return replace(estimates, regimes=self._regimes(data), msm=msm)

    def _msm(self, data: CausalData) -> MSMSet | None:
        """The declared working model evaluated on ``data``, or ``None`` if none was."""
        return None if self.msm is None else MSMSet.evaluate(self.msm, data)

    def _regimes(self, data: CausalData) -> RegimeSet | None:
        """The declared regimes evaluated on ``data``, or ``None`` for an arm-indexed fit."""
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
        """The shift code contrasts are taken against, from ``reference=`` and the names.

        Defaults to the first declared shift rather than to the natural course, which is
        the same rule the arms and regimes follow -- ``reference=`` is how to say
        otherwise, and declaring ``Shift(0.0, cap=None)`` first is the usual way to make
        ``ate_shift`` read as *the effect of shifting*.
        """
        names = [shift.name for shift in self.shifts]
        if self.reference is None:
            return 0.0
        if str(self.reference) not in names:
            raise DataError(f"reference={self.reference!r} is not one of the shifts {names}")
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
                    ("P(Z=z|A,W)", data.has_intermediate),
                )
                if present
            ),
            fits_treatment=not _is_natural_course(data, estimands),
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
            ("P(Delta = 1 | A, W)", missingness_values)
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

        It checks the MSM projection-weight declaration and then each ``Stochastic``
        regime's density declaration first, as :meth:`fit` does. Every sweep that
        recomputes an estimate comes through here, so a result restored from an artifact
        written before ``MSM.weights_kind`` or ``Stochastic.density_kind`` existed refuses
        each recomputation (roadmap rows RM13 and RM25). Loading re-checks nothing: that
        result keeps the estimates it stored, and they answer as they were saved.
        """
        if self.msm is not None:
            refuse_projection_weights(self.msm)
        refuse_regime_densities(self.interventions)
        requested = tuple(estimands)
        level = self.alpha_sig if alpha_sig is None else alpha_sig
        regimes = nuisance.regimes
        reference = self._reference_arm(data, regimes, nuisance.shifts)
        mean_bounds = g_bounds or resolve_g_bounds(
            self.g_bounds, self._bounds_n(data), for_att=False
        )
        conditional_bounds = g_bounds_conditional or resolve_g_bounds(
            self.g_bounds, self._bounds_n(data), for_att=True
        )

        estimates: dict[str, ParameterEstimate] = {}
        fluctuations: dict[str, Fluctuation] = {}
        pooled_report: dict[str, ParameterEstimate] = {}
        canonical_report: dict[str, ParameterEstimate] = {}
        fold_estimates: dict[str, tuple[float, ...]] = {}
        epsilon: dict[str, tuple[float, ...]] = {}
        fold_epsilon: dict[str, tuple[tuple[float, ...], ...]] = {}
        indices: list[IntArray] = []
        validation_indices = (
            [] if nuisance.folds.is_single else [test for _, test in nuisance.folds]
        )

        groups = (
            ["natural_course"] if _is_natural_course(data, requested) else self._groups(requested)
        )
        for group in groups:
            bounds = g_bounds_for(group, mean_bounds, conditional_bounds)
            # A group whose parameter is defined *through* the mechanism has a second
            # score equation, so its targeting alternates and returns the nuisances
            # re-tilted at the targeted g. `targeted` is what the estimates are read
            # from; `nuisance` stays the initial fit and is what the result reports.
            targeted = nuisance
            targeting_submodel: Submodel | None = None
            if data.has_strata and (
                needs_mechanism(group)
                or needs_reduction(nuisance, group)
                or needs_projection(nuisance, group)
            ):
                raise NotImplementedError(
                    f"baseline strata are not yet combined with the {group!r} group's "
                    "alternating targeting equations. Fit the marginal parameter, or "
                    "use an arm/regime/shift target whose outcome fluctuation is fixed."
                )
            if needs_mechanism(group):
                submodel, fluctuation, targeted = solve_with_mechanism(
                    data,
                    nuisance,
                    group,
                    self.targeting_spec(),
                    bounds=bounds,
                    nuisance_bound=self.nuisance_bound
                    if nuisance_bound is None
                    else nuisance_bound,
                    scaled=nuisance.scaler.scale(data.outcome),
                    weights=self._validation_weights(data, nuisance),
                    observed=data.observed,
                )
            elif needs_reduction(nuisance, group):
                # A fit carrying reduced-dimension regressions solves two further score
                # equations, one of which fluctuates g. Nothing about the *reported*
                # nuisances moves -- the estimand is still the plug-in mean of the targeted
                # regression -- so this returns two values, as the projection does and
                # unlike the mechanism alternation.
                submodel, fluctuation = self._solve_reduction(
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

            pooled = self._estimates_for(
                data, targeted, group, submodel, fluctuation, requested, level, reference
            )
            pooled_report.update(pooled)
            if data.has_strata:
                assert targeting_submodel is not None
                stratified = self._stratum_estimates(
                    data,
                    targeted,
                    group,
                    submodel,
                    targeting_submodel,
                    fluctuation,
                    requested,
                    level,
                    reference,
                )
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
            canonical = _average_over_folds(
                per_fold, tuple(pooled), indices, n=data.n, cluster=data.cluster, alpha=level
            )
            canonical_report.update(canonical)
            fold_estimates.update(
                {
                    name: tuple(values[name].psi for values in per_fold)
                    for name in canonical  # only the estimands every fold could compute
                }
            )
            estimates.update(canonical if self.cv_evaluation else pooled)

        # Stamped at the one place every estimate this estimator produces comes from, so
        # ``fit``, ``retarget`` and every sensitivity sweep that retargets a perturbed
        # input all report the same status.  Stamping in ``fit`` alone would leave the
        # truncation curve and the refutations building intervals the fit itself refuses.
        # The two fold-level reports are stamped here too, because ``CVTargeting``
        # publishes their standard errors and reads its status off them.
        status = self._inference_status(data)
        ordered = stamp_inference(_in_report_order(estimates, requested), status)
        detail = (
            CVTargeting(
                n_folds=len(indices),
                fold_sizes=tuple(int(index.size) for index in indices),
                variance={name: value.variance for name, value in canonical_report.items()},
                fold_estimates=fold_estimates,
                epsilon=epsilon,
                fold_epsilon=fold_epsilon,
                pooled=_in_report_order(pooled_report, requested),
                canonical=_in_report_order(canonical_report, requested),
                backend=data.backend,
            ).stamped(status)
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
            self._reference_arm(data, nuisance.regimes, nuisance.shifts),
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
        restored result or a copied estimator, and it refuses them under cross-fitting and
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
        balancing policies wherever a split is drawn.  The field stays because a result
        restored from an earlier version carries the policy that version allowed, and
        :class:`~cleverly.learners.crossfit.CrossFitPlan` reads it back.  ``scheme`` names
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
    ) -> Submodel:
        r"""One disjoint score block per baseline stratum.

        For stratum ``s`` the block is ``I(S=s) H_s / P_n(S=s)``.  ``H_s`` is
        rebuilt with the *conditional* arm shares for ATT/ATC; multiplying a globally
        normalised conditional-effect covariate, as a generic wrapper would do, targets
        the wrong denominator.  The blocks have disjoint support, so no redundant
        marginal column is added: the marginal score is their empirical weighted sum.
        """
        assert data.strata is not None
        lower = self.nuisance_bound if nuisance_bound is None else float(nuisance_bound)
        pieces: list[Submodel] = []
        for code in range(data.n_strata):
            mask = data.strata == code
            probability = float(np.average(mask, weights=data.weights))
            fractions = np.array(
                [arm_share(data.treatment, data.weights, arm, mask=mask) for arm in nuisance.arms],
                dtype=float,
            )
            if fractions.size and np.any(fractions <= 0.0):
                absent = [
                    data.arm_label(arm)
                    for arm, fraction in zip(nuisance.arms, fractions, strict=True)
                    if fraction <= 0.0
                ]
                raise DataError(
                    f"baseline stratum {data.stratum_label(code)} contains no positive-"
                    f"weight observations from treatment arm(s) {absent}; its empirical "
                    "targeting score is unidentified"
                )
            base = build_submodel(
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
            multiplier = mask.astype(float) / probability
            label = data.stratum_label(code)
            pieces.append(
                Submodel(
                    base.observed * multiplier[:, None],
                    {arm: values * multiplier[:, None] for arm, values in base.arms.items()},
                    tuple(f"{name} | {label}" for name in base.names),
                    base.group,
                )
            )
        return Submodel(
            np.hstack([piece.observed for piece in pieces]),
            {arm: np.hstack([piece.arms[arm] for piece in pieces]) for arm in pieces[0].arms},
            tuple(name for piece in pieces for name in piece.names),
            group,
        )

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

    def _solve_reduction(
        self,
        data: CausalData,
        nuisance: NuisanceEstimates,
        group: TargetGroup,
        bounds: tuple[float, float],
        nuisance_bound: float | None,
    ) -> tuple[Submodel, Fluctuation]:
        """Alternate the outcome, the mechanism and the reduced regressions.

        Pooled only.  Fold-wise targeting would need each fold's reduced regressions fitted
        out of that fold and its own alternation run inside it, which is a derivation rather
        than a loop -- :class:`~cleverly.DRTMLE` refuses ``targeting_scheme="fold"`` by name
        rather than quietly targeting pooled, which is what the mechanism alternation does.
        """
        reduction = self._reduction(data, nuisance)
        if reduction is None:
            raise NotImplementedError(
                "these nuisances carry reduced-dimension regressions, so the targeting step "
                "has two further score equations to solve -- and solving them refits those "
                f"regressions against the targeted pair, which a {type(self).__name__} has "
                "no learners for. Retarget with the DRTMLE that fitted them, or drop "
                "`reduced` to report a plain TMLE under a plain TMLE's name."
            )
        return solve_with_reduction(
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
        )

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
        shifts: ShiftSet | None,
        incremental: IPSISet | None,
        msm: MSMSet | None,
    ) -> tuple[tuple[float, ...], dict[float, Any]]:
        """The codes this fit's parameters are keyed by, and what to report them as.

        Exactly one of the four sources is live, which :meth:`_validate_settings` and
        :meth:`_check_shifts` have already established: a fit cannot declare two of the
        keywords, and a continuous treatment must declare ``shifts=``.

        The working model's codes index its *terms*, not its arms -- which is the whole of
        what makes ``msm`` a fourth axis rather than a target on the arm axis.
        """
        if msm is not None:
            return msm.codes, dict(msm.labels)
        if shifts is not None:
            return shifts.codes, dict(shifts.labels)
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
    ) -> dict[str, ParameterEstimate]:
        """Build every estimand that this fluctuation supports.

        ``index`` restricts every input to one validation fold, which is what the
        fold-evaluated CV-TMLE needs; ``None`` uses the whole sample.  Weights are
        renormalised within the fold so that the fold's estimate and influence curve are
        exactly what a standalone fit on those rows would produce -- the package's
        convention is mean-one weights, and a fold's slice of a globally normalised
        vector does not satisfy it.
        """
        scaler = nuisance.scaler
        scaled = scaler.scale(data.outcome)
        targeted = fluctuation.targeted
        weights, observed = data.weights, data.observed
        treatment, cluster, n = data.treatment, data.cluster, data.n
        regimes = nuisance.regimes
        shifts = nuisance.shifts
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
            shifts = None if shifts is None else shifts.subset(index)
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
        codes, labels = self._parameter_axis(data, regimes, shifts, incremental, msm)
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
            shifts=None if shifts is None else shifts.design,
            incremental=incremental,
            msm_design=None if msm is None else msm.design,
            msm_weights=None if msm is None else msm.weights,
            msm_link="identity" if msm is None else str(msm.link),
            always_label=(
                regimes is not None
                or shifts is not None
                or incremental is not None
                or msm is not None
            ),
            # The stacked natural-course estimator declares the raw second moment, so its variance,
            # covariance and contrasts all read one rule. ``make_estimate`` owns the map
            # from that rule to the stored variance.
            covariance_rule=(
                "second_moment" if group == "natural_course" and self.cross_fit else "centered"
            ),
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
    ) -> dict[str, ParameterEstimate]:
        """Conditional plug-ins and full-sample influence curves for every stratum."""
        assert data.strata is not None
        width = marginal_submodel.dim
        if targeting_submodel.dim != width * data.n_strata:
            raise RuntimeError(
                "the stratified targeting submodel does not contain one base block per stratum"
            )
        out: dict[str, ParameterEstimate] = {}
        for code in range(data.n_strata):
            index = np.flatnonzero(data.strata == code).astype(np.int64)
            probability = float(np.average(data.strata == code, weights=data.weights))
            block = slice(code * width, (code + 1) * width)
            # Undo I_s / p_s before the ordinary target builder renormalises weights in
            # the subset.  Its resulting curve is on the n_s-row empirical scale; the
            # n/n_s embedding below restores I_s D_s / P_n(S=s), the full-law gradient.
            conditional_submodel = Submodel(
                targeting_submodel.observed[:, block] * probability,
                {
                    arm: values[:, block] * probability
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
                fluctuation,
                requested,
                alpha_sig,
                reference,
                index=index,
            )
            label = data.stratum_label(code)
            for estimate in estimates.values():
                name = stratum_alias(estimate.name, label)
                curve = np.zeros(data.n, dtype=float)
                curve[index] = estimate.influence_curve * (data.n / index.size)
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
        return out

    def _bootstrap_point_estimates(
        self, data: CausalData, intermediate_value: float | None
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
        """
        estimands = resolve_estimands(self.estimands, data.family, data.n_arms, axis=self._axis)
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
    if len(arms) == 2:
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
    :func:`cross_validated_variance` uses the uncentred fold second moments.
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
    :class:`TMLE` may be passed through.

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
    )
    estimator = TMLE(**kwargs)
    return estimator.fit(data)
