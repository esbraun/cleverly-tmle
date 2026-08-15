r"""Classic point-treatment TMLE.

Estimates the effect of a binary treatment ``A`` on an outcome ``Y``, adjusting for
baseline covariates ``W``, under the usual identification assumptions
(consistency, no unmeasured confounding given ``W``, positivity).

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
  comes from the second-order remainder of the von Mises expansion being a *product* of
  the two nuisance errors, so that either factor being zero kills it.  It needs
  identification and positivity, and it needs no rate on either nuisance.
  ``tests/unit/test_remainder.py`` checks the product form exactly.
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
>>> from cleverly import TMLE
>>> from cleverly.datasets import make_nonlinear_ate
>>> frame, truth = make_nonlinear_ate(n=1000, seed=0)
>>> res = TMLE(random_state=0).fit(frame, outcome="Y", treatment="A").single()
>>> print(res.summary())                      # doctest: +SKIP
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from functools import partial
from typing import Any, cast

import numpy as np

from .._typing import (
    BoolArray,
    Estimand,
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
from ..data.causal_data import CausalData, TreatmentKind
from ..exceptions import (
    ConvergenceWarning,
    DataError,
    PositivityWarning,
    WeightingWarning,
)
from ..fluctuation._score import relative_score, score_columns, score_scale
from ..fluctuation.iterative import (
    Fluctuation,
    FoldFluctuation,
    InitialFit,
    TargetingFailure,
)
from ..fluctuation.mechanism import needs_mechanism
from ..fluctuation.submodel import Submodel, TargetGroup, restrict, stitch
from ..inference.bootstrap import Resampling, run_bootstrap
from ..inference.cluster import cross_validated_variance
from ..inference.influence import (
    CorrectionParts,
    ParameterEstimate,
    average_estimates,
    make_estimate,
    reduced_correction_parts,
)
from ..inference.multiplier import MultiplierKind, simultaneous_bands
from ..interventions import Incremental, IPSISet, RegimeSet, Shift, ShiftSet, as_interventions
from ..learners._fitting import Task
from ..learners.crossfit import CrossFitPlan, Folds, make_folds
from ..learners.super_learner import resolve_learner
from ..msm import MSM, MSMSet
from ..provenance import record as provenance_record
from ..targets import TargetContext, groups_for, parameter_stem, targets_for
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


class TMLE:
    """Targeted maximum likelihood estimator for a binary point treatment.

    Parameters
    ----------
    outcome_learner, treatment_learner:
        Nuisance estimators for ``Qbar(A, W)`` and ``g(W)``.  A string or list is
        treated as a :class:`~cleverly.learners.SuperLearner` library specification
        (see :data:`cleverly.learners.LIBRARY_PRESETS`); any scikit-learn compatible
        estimator is used directly.
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

        The fold-evaluated and stacked reports are both available on
        ``result.cv_targeting`` when this setting is true.  For unequal folds the stored
        influence-curve rows are scaled by ``n/(V*n_v)`` so they represent the reported
        equal-fold estimator under the full empirical mean.

        Combines with ``repeats=R``; see there for the variance rule.  :meth:`retarget`
        and the sensitivity analyses follow the setting, so a truncation or missingness
        sweep perturbs the same fold-evaluated estimator the headline reports.
    n_folds, learner_folds:
        Outer cross-fitting folds, and the inner folds a Super Learner uses to score
        its candidates.
    repeats:
        How many independent draws of the whole cross-fitting split to average the
        estimate over.  ``1`` (default) is an ordinary fit, and is bit-for-bit an
        ordinary fit rather than an equivalent one.

        A single split is one draw from a randomised procedure, and on a moderate sample
        two seeds can move ``psi`` by an appreciable fraction of its standard error.
        Repeating the split and averaging removes that component of the variability
        without touching the estimand.  Every row is out of fold in every draw, so

        .. math:: \\bar\\psi = \\tfrac{1}{R}\\sum_r \\psi_r

        is the same functional of the same data, with influence curve
        :math:`\\tfrac{1}{R}\\sum_r \\mathrm{IC}_r` -- which keeps the variance, the delta
        method, the cluster-robust standard error and the simultaneous bands coherent,
        because all of them are computed from the curve.  Costs ``R`` times a fit.

        A draw redraws *every* split, not only the outer one: the inner cross-validation
        that scores the Super Learner's candidates, and C-TMLE's selection folds, are
        drawn from the draw's own seed.  Holding those fixed would average over one stage
        of a randomised procedure while pinning the rest.

        The aggregation is the **mean**, and only the mean.  The median-of-estimates
        aggregation common in the double-machine-learning literature (Chernozhukov et al.
        2018) is deliberately not offered: the median of the ``psi_r`` is not the
        estimator whose influence curve is the median of the ``IC_r``, so its variance,
        its delta method and its bands would every one of them be describing a different
        quantity than the point estimate they were attached to.

        ``result.repeats`` holds the per-draw nuisance fits, fluctuations and point
        estimates, and ``result.repeat_spread()`` reports how far the draws moved --
        a diagnostic of the fold noise, never a standard error.  Every sensitivity
        analysis that produces a number follows all ``R``; the diagnostics that describe
        a fitted *mechanism* report the first draw and say so.

        With ``cv_evaluation=True`` the point estimate is the mean of the ``R``
        fold-evaluated
        CV-TMLE estimates, but the standard error cannot come from the averaged curve: the
        cross-validated variance is defined by a fold partition and the average belongs to
        none of the ``R``.  Reported instead is the mean of the ``R`` cross-validated
        variances, each computed on its own draw's partition.  That is consistent for the
        same limit and errs conservative in finite samples; the derivation, and why a
        cross-validated variance *of the averaged curve* would be vacuous rather than
        merely arbitrary, are in :func:`_with_cross_validated_variance`.
    stratify_folds:
        What the outer folds are balanced on.  ``"treatment"``, the default, balances the
        arms so no fold is left without one and the propensity model is fittable
        everywhere.  ``"treatment+outcome"`` crosses the outcome in, which matters when
        events are rare: an arm-balanced fold can still contain none of them, and an
        outcome regression fitted on a fold with no events is degenerate.  The fold count
        is then capped at the rarest *cell* rather than the rarer arm, so asking for ten
        folds on a 2% event rate will reduce them, with a warning saying so.

        Binary outcomes only, and refused on a continuous dose -- both refusals name what
        they would need.  An unobserved outcome (``delta=``) is its own stratum rather
        than being pooled with ``Y = 0``, since a fold with no *observed* outcomes in an
        arm cannot fit the regression either.

        Worth stating plainly: this makes the fold assignment a function of the outcome,
        and the cross-fitting argument conditions on the split.  That is a statement about
        which splits are conditioned on, not a bias -- and the alternative it is weighed
        against is a fold that cannot fit the regression at all.
    g_bounds:
        Propensity truncation.  ``"auto"`` uses ``5 / (sqrt(n) log n)`` for the
        ATE family and ``0.025`` for the ATT/ATC, matching R's ``tmle``.
    q_bounds:
        Assumed support of a continuous outcome (R's ``Qbounds``).  ``None`` widens the
        observed range by 10%.
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

    def __init__(
        self,
        *,
        outcome_learner: Learner | str | Sequence[Any] | None = "default",
        treatment_learner: Learner | str | Sequence[Any] | None = "default",
        missingness_learner: Learner | str | Sequence[Any] | None = None,
        intermediate_learner: Learner | str | Sequence[Any] | None = None,
        family: Family = "auto",
        fluctuation: FluctuationKind = "logistic",
        targeting: TargetingMethod = "iterative",
        cross_fit: bool = True,
        targeting_scheme: TargetingScheme = "pooled",
        cv_evaluation: bool = False,
        n_folds: int = 10,
        learner_folds: int = 5,
        repeats: int = 1,
        stratify_folds: FoldStrata = "treatment",
        g_bounds: GBounds = "auto",
        q_bounds: tuple[float, float] | None = None,
        alpha: float = 0.9995,
        nuisance_bound: float = DEFAULT_NUISANCE_BOUND,
        target_weights: bool = False,
        screen_treatment: bool = False,
        screen_threshold: float = 0.1,
        min_retain: int | None = None,
        estimands: Sequence[Estimand] | str | None = None,
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
                "solve their score equations at once. Fit them separately."
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
                "rather than by arm."
            )
        if self.msm is not None and self.reference is not None:
            raise ValueError(
                "reference= names the arm, regime or shift every contrast is taken "
                "against, and a working model reports coefficients rather than contrasts "
                "-- there is nothing for it to be a reference for. Which arm is the "
                "baseline is decided by the design you gave msm=, usually by an intercept "
                "column. A difference of two coefficients comes from result.contrast()."
            )
        if self.stratify_folds not in ("treatment", "treatment+outcome"):
            raise ValueError(
                "stratify_folds must be 'treatment' or 'treatment+outcome'; got "
                f"{self.stratify_folds!r}"
            )
        if self.density_bins < 3:
            raise ValueError(
                f"density_bins must be at least 3; got {self.density_bins}. Two bins make "
                "the density a single hazard, which cannot describe a dose-response."
            )
        if self.repeats < 1:
            raise ValueError(f"repeats must be at least 1; got {self.repeats}")
        if self.repeats > 1 and not self.cross_fit:
            raise ValueError(
                "repeats= averages the estimate over independent draws of the "
                "cross-fitting split, and cross_fit=False makes no split to draw. There "
                "is no fold noise to average away when every nuisance is fitted in "
                "sample. Set cross_fit=True, or leave repeats at 1."
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
            shared = self._prepare_shared(prepared, levels)
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
        self, data: CausalData, levels: Sequence[float]
    ) -> tuple[OutcomeScaler, tuple[tuple[Folds, NuisanceEstimates], ...]]:
        """Fit the level-independent nuisances once, for every requested level.

        One ``(folds, nuisance)`` pair per repeat: the levels share nuisances *within* a
        draw, which is what this method exists for, and the draws remain separate, which
        is what makes them repeats.  The scaler is a function of the outcome alone and so
        is shared across both.
        """
        scaler = self._scaler(data)
        draws = []
        for seed in self.crossfit_plan(data).seeds():
            folds = self._folds(data, seed)
            draws.append(
                (
                    folds,
                    self._fit_nuisances(
                        data, folds, scaler, levels[0], tuple(levels[1:]), seed=seed
                    ),
                )
            )
        return scaler, tuple(draws)

    def refit(self, data: CausalData, *, intermediate_value: float | None = None) -> TMLEResult:
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
        """
        return self._fit_single(data, intermediate_value=intermediate_value)

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
        shared: tuple[OutcomeScaler, tuple[tuple[Folds, NuisanceEstimates], ...]] | None = None,
    ) -> TMLEResult:
        """Fit for one value of the intermediate (or for no intermediate at all).

        ``shared`` supplies nuisance fits already computed for every level of the
        intermediate; only the targeting step then runs per level.

        With ``repeats=R`` the whole construction below -- split, nuisances, targeting --
        runs ``R`` times and the reports are averaged.  The loop sits here, around
        :meth:`_nuisances` rather than inside it, which is what makes it free for the
        variants: :class:`~cleverly.CTMLE` overrides that method alone, so its propensity
        selection is repeated per draw without ``estimators/ctmle.py`` knowing repeats
        exist.  The bootstrap and the simultaneous bands sit *after* the loop and read the
        averaged estimates, so they need no change either.

        The one report that cannot be assembled by averaging alone is the fold-evaluated
        CV-TMLE variance, which belongs to a fold partition rather than to a curve; see
        :func:`_with_cross_validated_variance` and :meth:`_cv_detail`.
        """
        self._check_shifts(data)
        self._check_incremental(data)
        if data.has_strata and (self.cv_evaluation or self.targeting_scheme == "fold"):
            raise NotImplementedError(
                "baseline strata currently use one joint pooled fluctuation. "
                "cv_evaluation=True or targeting_scheme='fold' would require the "
                "stratum probabilities and conditional treatment shares to be rebuilt "
                "inside every validation fold; use the default pooled targeting scheme"
            )
        estimands = resolve_estimands(self.estimands, data.family, data.n_arms, axis=self._axis)
        population_intervention = {"ey_obs", "par", "paf"}.intersection(estimands)
        if (
            population_intervention
            and self.estimands == "all"
            and (data.has_missing_outcome or data.has_intermediate)
        ):
            estimands = tuple(name for name in estimands if name not in population_intervention)
            population_intervention = set()
        if population_intervention and data.has_missing_outcome:
            raise NotImplementedError(
                f"{sorted(population_intervention)} do not yet support delta=: under MAR "
                "the natural-course mean E[Y] needs an additional outcome/missingness "
                "score equation; using complete cases would estimate a different parameter"
            )
        if population_intervention and data.has_intermediate:
            raise NotImplementedError(
                f"{sorted(population_intervention)} do not yet support intermediate=: "
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
            fold_draws = [folds for folds, _ in pooled]
            nuisances = [
                nuisance.at_level(cast("float", intermediate_value)) for _, nuisance in pooled
            ]
            config = self._config(data, estimands, scaler, fold_draws[0])
        else:
            scaler = self._scaler(data)
            seeds = self.crossfit_plan(data).seeds()
            fold_draws = [self._folds(data, seed) for seed in seeds]
            # The realised fold count can differ between draws when a cap fires on one and
            # not another, so the config -- like every read-through attribute on the result
            # -- describes the first draw.  It is then the *same* config for every draw,
            # which matters: the truncation bounds a draw is fitted under must not depend
            # on which draw it is, or the R estimates would not be estimating one thing.
            config = self._config(data, estimands, scaler, fold_draws[0])
            nuisances = []
            for index, (folds, seed) in enumerate(zip(fold_draws, seeds, strict=True)):
                nuisance, draw_extra = self._nuisances(
                    data, folds, scaler, config, intermediate_value, seed=seed
                )
                nuisances.append(nuisance)
                if index == 0:
                    extra = draw_extra

        self._warn_on_positivity(nuisances[0], config, intermediate_value)
        self._warn_on_estimated_weights(data)

        per_repeat: list[dict[str, ParameterEstimate]] = []
        repeats: list[RepeatFit] = []
        details: list[CVTargeting | None] = []
        for nuisance in nuisances:
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
                )
            )
            details.append(detail)

        estimates = average_estimates(per_repeat, cluster=data.cluster)
        cv_detail = self._cv_detail(details, cluster=data.cluster)
        if self.cv_evaluation and cv_detail is None:
            raise RuntimeError(
                "cv_evaluation=True needs at least two realised validation folds, but "
                "the requested split collapsed to one. Use fewer-stratified data, or "
                "fit without fold-evaluated CV-TMLE."
            )
        if self.cv_evaluation and cv_detail is not None:
            estimates = _with_cross_validated_variance(
                estimates, [detail.variance for detail in cast("list[CVTargeting]", details)]
            )

        result = TMLEResult(
            estimates=estimates,
            repeats=tuple(repeats),
            data=data,
            config=config,
            estimator=self,
            provenance=provenance_record(
                data, fold_draws, random_state=self.random_state, run_id=self.run_id
            ),
            intermediate_value=intermediate_value,
            extra=extra if cv_detail is None else {**extra, "cv_tmle": cv_detail},
        )

        if self.simultaneous and len(estimates) > 1:
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
        """One fold-level report for the whole fit, however many draws it averaged.

        The fields split by what they *are*.  ``pooled``, ``canonical`` and ``variance``
        are estimates, so they follow every draw exactly as the headline report does.
        ``n_folds``, ``fold_sizes``, ``fold_estimates`` and ``fold_epsilon`` are indexed
        by fold, and fold 3 of one draw is not fold 3 of another -- there is no
        correspondence to average along -- so they describe the first draw and
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
                "cv_evaluation=True would be averaging fold-wise estimates from some draws "
                "with pooled ones from the rest under a single name. Re-run with fewer "
                "n_folds, or with repeats=1."
            )
        first = present[0]
        if len(present) == 1:
            return first
        canonical = _with_cross_validated_variance(
            average_estimates([detail.canonical for detail in present], cluster=cluster),
            [detail.variance for detail in present],
        )
        return replace(
            first,
            repeats=len(present),
            pooled=average_estimates([detail.pooled for detail in present], cluster=cluster),
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
        if data.n_arms != 2:
            raise DataError(
                f"an incremental propensity-score intervention tilts the *odds* of "
                f"treatment, which names two arms; {data.treatment_name} has "
                f"{data.n_arms} ({list(data.treatment_levels)}). Kennedy's tilt has no "
                "single-parameter generalisation to a multinomial mechanism -- one odds "
                "per contrast would be a different intervention with a different "
                "influence function."
            )
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

        ``seed=None`` means "the plan's", which is unambiguous rather than merely
        convenient: :meth:`CrossFitPlan.seeds` hands a repeat ``None`` in exactly the case
        where ``random_state`` is ``None`` too, so the two readings never disagree.
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

    def _resolve_learner(
        self,
        spec: Learner | str | Sequence[Any] | None,
        *,
        task: Task,
        fallback: Learner | str | Sequence[Any] | None = None,
        seed: int | None = None,
    ) -> Learner:
        """Turn a learner specification into a fitted-per-fold estimator.

        ``seed`` is the draw's, under the same convention :meth:`_folds` uses: ``None``
        means "the estimator's own ``random_state``".  It reaches the Super Learner's
        *inner* split, so a repeat redraws the whole nested cross-validation rather than
        only the outer one -- which is what makes ``repeats=R`` an average over the
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
        return self._fit_nuisances(data, folds, scaler, intermediate_value, seed=seed), {}

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
        conditions on the fitted weights exactly as the influence-curve one does. Saying
        so is cheap; letting the mistake pass silently is not.
        """
        if not (data.is_weighted and data.weight_spec.estimated and self.n_bootstrap):
            return
        warnings.warn(
            "weights_estimated=True with n_bootstrap: the bootstrap resamples rows and "
            "renormalises the weights it was given, never re-deriving them, so its "
            "intervals condition on the fitted weights just as the influence-curve ones "
            "do. Re-deriving the weights inside each replicate needs the model that "
            "produced them, which this package never sees. See cleverly.data.weighting.",
            WeightingWarning,
            stacklevel=3,
        )

    def _warn_on_positivity(
        self,
        nuisance: NuisanceEstimates,
        config: TMLEConfig,
        intermediate_value: float | None = None,
    ) -> None:
        lower, upper = config.g_bounds
        # Counted per *unit*: a row is extrapolated if any arm's probability is outside the
        # bounds, since one binding denominator is enough to give that row unbounded
        # leverage.  With two arms and the symmetric bounds ``"auto"`` and a scalar both
        # produce, ``g0 < lower`` exactly when ``g1 > upper``, so this is the same count the
        # single-vector form reported.
        mechanism = np.asarray(nuisance.propensity.values, dtype=float)
        outside = float(np.mean(np.any((mechanism < lower) | (mechanism > upper), axis=1)))
        if outside > _TRUNCATION_WARN_FRACTION:
            warnings.warn(
                f"{outside:.1%} of units have an estimated treatment probability outside the "
                f"truncation bounds [{lower:.4g}, {upper:.4g}] for at least one arm, so those "
                "units' contributions rest on extrapolation rather than data. Inspect "
                "res.sensitivity.positivity() and res.sensitivity.truncation_curve() before "
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
        candidates: list[tuple[str, FloatArray | None]] = [
            ("P(Delta = 1 | A, W)", nuisance.missingness)
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
                warnings.warn(
                    f"{below:.1%} of estimated {label} values fall below the nuisance bound "
                    f"{config.missingness_bound:.4g}. That probability divides the clever "
                    "covariate just as g(W) does, so those rows carry outsized leverage and "
                    "the bound is trading bias for variance. Inspect "
                    "res.sensitivity.positivity() and re-run with a different nuisance_bound.",
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
        """
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

        for group in self._groups(requested):
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

        ordered = _in_report_order(estimates, requested)
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
            )
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

        With ``stratify_folds="treatment+outcome"`` the code is the treatment crossed with
        the outcome, and an unobserved outcome is its own level rather than being folded
        in with ``Y = 0``: a fold with no *observed* outcomes in an arm cannot fit the
        outcome regression either, which is the failure the option exists to prevent.
        ``resolve_n_folds`` then caps the fold count at the rarest cell rather than the
        rarer arm, which is the whole mechanism -- no other code changes.
        """
        if self.stratify_folds == "treatment+outcome":
            # Refused rather than quietly ignored: a caller who asked for this asked
            # because they have a rare level to protect, and silently not protecting it
            # is the worst of the three outcomes.
            if data.is_continuous_treatment:
                raise DataError(
                    "stratify_folds='treatment+outcome' needs arms to cross the outcome "
                    "with, and this fit declared a continuous dose with shifts=. A dose "
                    "has no strata: every value is its own, which caps the fold count at "
                    "one row. What a continuous treatment balances on in spirit is the "
                    "density's bins, and those are chosen inside each training fold."
                )
            if data.family != "binomial":
                raise DataError(
                    "stratify_folds='treatment+outcome' needs a binary outcome to have a "
                    f"rare level worth balancing, and this fit's family is "
                    f"{data.family!r}. Crossing a continuous outcome in would make every "
                    "distinct value its own stratum and refuse to split at all; leave "
                    "stratify_folds='treatment'."
                )
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
        rarest stratum and ``make_folds`` at the cluster count -- and the warnings that
        say so are gone by the time anyone reads the result.

        Takes ``data`` because two of the fields are answers about it rather than
        settings: whether clusters were declared, and whether the treatment has strata to
        balance at all.  Both decisions are made here and in :meth:`_folds`, which is one
        place too many, so :meth:`_folds` reads them off the plan.
        """
        cross_fit = self.cross_fit
        if not cross_fit or data.is_continuous_treatment:
            stratify_by: tuple[str, ...] = ()
        elif self.stratify_folds == "treatment":
            stratify_by = (data.treatment_name,)
        else:
            stratify_by = (data.treatment_name, data.outcome_name)
        clustered = cross_fit and data.cluster is not None
        if not cross_fit:
            scheme = "none"
        elif clustered and stratify_by:
            scheme = "stratified-grouped"
        elif clustered:
            scheme = "grouped"
        elif stratify_by:
            scheme = "stratified"
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
            fold_weights = data.weights[test]
            fractions = np.array(
                [
                    np.average(data.treatment[test] == arm, weights=fold_weights)
                    for arm in nuisance.arms
                ],
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
                [
                    np.average(data.treatment[mask] == arm, weights=data.weights[mask])
                    for arm in nuisance.arms
                ],
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
        masses = []
        traces = []
        reasons: list[str] = []
        iterations = 0

        for _, test in nuisance.folds:
            fold_submodel, fold_fluctuation = per_fold(test)
            pieces.append((test, fold_submodel))
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
                )
            )
            masses.append(float(data.weights[test].sum()))
            traces.append(fold_fluctuation.trace[-1] if fold_fluctuation.trace else float("nan"))
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
            trace=tuple(traces),
            method="iterative" if self.targeting == "iterative" else "one_step",
            names=submodel.names,
            score_scale=scale,
            folds=tuple(fold_records),
            score_initial=score_before,
            n_solver_calls=len(fold_records),
            failure=_dominant_failure(reasons, failed),
            # Each fold solved its own projection, so there is no single beta the pooled
            # covariate was built at; the per-fold ones live on the pieces that were
            # stitched, and the *reported* coefficients come from the stitched fit.
            projection=None,
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
        )
        # One context per fluctuation, shared by every target in the group: the
        # mean-group estimands are different functionals of the same targeted
        # distribution, and `context.means` computes the counterfactual means once.
        out: dict[str, ParameterEstimate] = {}
        for target in targets_for(group, requested):
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
                name = f"{estimate.name}[{label}]"
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
        estimator that was reported, and under ``repeats=R`` that estimator is the average
        of ``R`` draws, whose fold noise is already averaged down.  Bootstrapping a single
        draw instead would attribute variability to ``psi_bar`` that ``psi_bar`` does not
        have.  It costs ``B * R`` fits, which is the honest price of the two settings
        together.
        """
        estimands = resolve_estimands(self.estimands, data.family, data.n_arms, axis=self._axis)
        scaler = self._scaler(data)
        seeds = self.crossfit_plan(data).seeds()
        fold_draws = [self._folds(data, seed) for seed in seeds]
        config = self._config(data, estimands, scaler, fold_draws[0])
        per_repeat = []
        for folds, seed in zip(fold_draws, seeds, strict=True):
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
        averaged = average_estimates(per_repeat, cluster=data.cluster)
        return {name: estimate.psi for name, estimate in averaged.items()}


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
    solved beside, so that is what the curve reads.

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
    mechanism = (
        fluctuation.mechanism.propensity
        if fluctuation.mechanism is not None
        else (
            nuisance.propensity.arm(reduction.reduced.arms[1])
            if len(reduction.reduced.arms) == 2
            else nuisance.propensity.values
        )
    )
    return reduced_correction_parts(
        scaled,
        targeted,
        data.treatment,
        reduction.reduced,
        mechanism,
        bounds=reduction.bounds,
        observed=data.observed,
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


def _with_cross_validated_variance(
    averaged: Mapping[str, ParameterEstimate],
    per_repeat_variance: Sequence[Mapping[str, float]],
) -> dict[str, ParameterEstimate]:
    r"""Give an averaged fold-evaluated report its draws' mean CV variance.

    Repeated cross-fitting reports :math:`\bar\psi = \frac1R\sum_r \psi_r` with influence
    curve :math:`\frac1R\sum_r \mathrm{IC}_r`, and everywhere else in this library the
    variance is then taken *from that curve*.  Under ``cv_evaluation`` it cannot be: the
    cross-validated variance of Zheng & van der Laan is defined by a fold partition, and
    the averaged curve belongs to none of the ``R`` partitions that made it.  What is
    reported instead is the mean of the ``R`` cross-validated variances, each computed on
    its own draw's partition from that draw's own fold-specific curve:

    .. math:: \bar\sigma^2 = \frac1R \sum_r \hat\sigma^2_{CV,r}.

    Two things make that the right quantity rather than merely an available one.  Each
    :math:`\hat\sigma^2_{CV,r}` is consistent for :math:`\mathrm{Var}(D^*)/n`, which is
    also what :math:`\mathrm{Var}(\bar\psi)` converges to, so nothing is given up
    asymptotically.  And in finite samples it errs *conservative*, never the other way:
    :math:`\mathrm{Var}(\bar\psi) = R^{-2}\sum_r\sum_s \mathrm{Cov}(\psi_r, \psi_s) \le
    \big(\frac1R\sum_r \mathrm{sd}(\psi_r)\big)^2 \le \frac1R\sum_r
    \mathrm{Var}(\psi_r)`, by Cauchy-Schwarz and then Jensen.  Erring that way is the
    whole reason to have asked for the cross-validated variance in the first place.  At
    ``R = 1`` the mean of one number is that number, so the construction is unchanged.

    The alternative that looks more natural -- hand the *averaged* curve to
    :func:`~cleverly.inference.cross_validated_variance` under one draw's partition -- is
    not merely arbitrary in its choice of partition, it is vacuous.  At equal fold sizes
    :math:`\frac1V\sum_v \frac{1}{n_v}\sum_{i \in \mathcal V_v} \mathrm{IC}_i^2 =
    \frac1n\sum_i \mathrm{IC}_i^2` for *every* partition, so the fold structure
    contributes nothing at all and the result is the pooled uncentred second moment
    wearing a cross-validated name.  ``tests/unit/test_repeated_crossfit.py`` keeps that
    identity as a negative control.

    Only names present in every draw's variance mapping are touched.  A targeting group
    that produced no folds is reported pooled and keeps its from-curve variance; a name
    :func:`~cleverly.inference.average_estimates` already dropped never arrives here.
    """
    if not per_repeat_variance:
        return dict(averaged)
    return {
        name: (
            replace(
                value,
                variance=float(np.mean([draw[name] for draw in per_repeat_variance])),
            )
            if all(name in draw for draw in per_repeat_variance)
            else value
        )
        for name, value in averaged.items()
    }


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
    >>> from cleverly import tmle
    >>> rng = np.random.default_rng(0)
    >>> W = rng.normal(size=(500, 3))
    >>> A = rng.binomial(1, 0.5, 500).astype(float)
    >>> Y = A + W[:, 0] + rng.normal(size=500)
    >>> res = tmle(Y, A, W, outcome_learner="glm", treatment_learner="glm").single()
    >>> round(res.psi("ate"), 1)                    # doctest: +SKIP
    1.0
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


def _dominant_failure(reasons: Sequence[str], failed: Sequence[int]) -> TargetingFailure | None:
    """The most common failure mode across folds, for the summary line.

    A single label cannot describe ten folds, so the per-fold detail stays on
    ``Fluctuation.folds``; this is only what to print when there is room for one word.
    """
    if not failed:
        return None
    modes = [reasons[i] for i in failed if reasons[i] != "unknown"]
    if not modes:
        return "max_iter_reached"
    return cast("TargetingFailure", max(set(modes), key=modes.count))
