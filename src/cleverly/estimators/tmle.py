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
``z`` -- which is a different parameter for each ``z``, so ``fit`` returns a
:class:`~cleverly.estimators.base.TMLEResultSet` with one result per level rather than a
single :class:`~cleverly.estimators.base.TMLEResult`.  That path rests on an
identification assumption the average treatment effect does not need, and it is not a
general longitudinal estimator; :mod:`cleverly.estimators.direct_effect` writes the
parameter down, derives its influence function, and says where the boundary is.

Example
-------
>>> from cleverly import TMLE
>>> from cleverly.datasets import make_nonlinear_ate
>>> frame, truth = make_nonlinear_ate(n=1000, seed=0)
>>> res = TMLE(random_state=0).fit(frame, outcome="Y", treatment="A")
>>> print(res.summary())                      # doctest: +SKIP
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from .._typing import (
    BoolArray,
    Estimand,
    Family,
    FloatArray,
    FluctuationKind,
    GBounds,
    IntArray,
    Learner,
    TargetingMethod,
    TargetingScheme,
)
from ..data.causal_data import CausalData
from ..exceptions import PositivityWarning, WeightingWarning
from ..fluctuation._score import relative_score, score_columns, score_scale
from ..fluctuation.iterative import (
    Fluctuation,
    FoldFluctuation,
    InitialFit,
)
from ..fluctuation.submodel import Submodel, TargetGroup, restrict
from ..inference.bootstrap import Resampling, run_bootstrap
from ..inference.cluster import cross_validated_variance
from ..inference.influence import (
    ParameterEstimate,
)
from ..inference.multiplier import MultiplierKind, simultaneous_bands
from ..learners._fitting import Task
from ..learners.crossfit import Folds, make_folds
from ..learners.super_learner import resolve_learner
from ..provenance import record as provenance_record
from ..targets import TargetContext, groups_for, targets_for
from ..utils.bounds import OutcomeScaler, resolve_g_bounds
from ..utils.frames import is_dataframe
from ._nuisance import NuisanceEstimates, fit_nuisances
from .base import (
    CVTargeting,
    TMLEConfig,
    TMLEResult,
    TMLEResultSet,
    attach_bootstrap,
    resolve_estimands,
)
from .targeting import TargetingSpec, build_submodel, solve_submodel

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
        (default) fits a single ``epsilon`` vector on all rows at once; ``"fold"`` fits
        a separate one inside each validation fold, which is the targeting step of Zheng
        & van der Laan's (2011) CV-TMLE.  Both solve the same pooled score equation
        exactly.  Together with ``cv_evaluation`` these name three distinct estimators,
        and it is worth being precise about which one is running:

        =========================================== ==============================
        setting                                      estimator
        =========================================== ==============================
        ``cross_fit=True, targeting_scheme="pooled"`` cross-fitted TMLE
        ``targeting_scheme="fold"``                   fold-targeted CV-TMLE
        ``targeting_scheme="fold", cv_evaluation``    canonical CV-TMLE
        =========================================== ==============================

        Cross-fitting the nuisances is what removes the Donsker condition on the
        nuisance *estimators*: fit out of fold, no model predicts a row it was trained
        on, and the empirical-process term involving them vanishes.  Pooled targeting
        on top of cross-fitted nuisances then adds an empirical-process term of its
        own, because ``epsilon`` is fit on the same rows it fluctuates.  That term is
        controlled -- but by a separate argument, not by cross-fitting.  Sample
        splitting is what licenses the argument: *conditional on the training-fold
        fits*, ``Qbar`` is a fixed function, so how it was learned stops mattering.
        Conditionally, the fluctuated family ``{Qbar(epsilon)}`` is indexed by a fixed
        finite-dimensional coefficient ranging over a compact neighbourhood, and is
        Lipschitz in it, so it has manageable entropy and is Donsker however complex
        ``Qbar`` is.  Finite-dimensional, not scalar: ``epsilon`` has one entry per
        clever-covariate column, which is two for the ``mean`` group -- one per arm,
        the submodel behind ``ey1``, ``ey0``, ``ate``, ``rr`` and ``or`` -- and one
        for ``att`` and ``atc``.

        That argument buys the empirical-process term and nothing else.  It is not a
        substitute for nuisance convergence, and the pooled estimator is asymptotically
        linear and efficient only with the rest of the usual conditions in place: the
        estimated influence curve converging in ``L_2(P_0)``; positivity, which bounds
        the clever covariate and which the ``g_bounds`` truncation supplies; the score
        equation solved to ``o_P(n^{-1/2})``; ``epsilon_hat`` converging in probability
        within that compact neighbourhood; and a second-order remainder that is
        ``o_P(n^{-1/2})`` *by a product rate on* ``ghat`` *and* ``Qbarhat`` -- a
        condition on the learners themselves, which finite-dimensionality of the
        fluctuation does nothing to supply.

        One further asymmetry is worth stating plainly.  With ``cross_fit=True`` every
        row's *nuisance* prediction is out of fold, but under pooled targeting its
        *targeted* prediction is not: a single ``epsilon_hat`` fit across all validation
        rows carries other folds' outcomes into every row's fluctuation.  The pooled
        coefficient stays low-dimensional, so this does not break the argument above,
        but it does mean the argument has to handle a random pooled coefficient rather
        than claim each targeted prediction is purely out-of-sample.  Fold-wise
        targeting is what makes that claim true.

        So the two schemes share a first-order limit under those conditions.  They are
        not the same estimator, and outside them -- and in finite samples generally --
        their behaviour and their remainder arguments differ.
        Zheng & van der Laan prove their result for the fold-targeted construction
        specifically; the pooled scheme is the cross-fitted TMLE of the
        double/debiased-machine-learning literature.  The statistical validation tier
        measures the two as equivalent on the processes it covers, which is evidence
        about those processes, not a general equivalence.

        Choose ``"fold"`` for the construction Zheng & van der Laan analyse and for the
        per-fold diagnostics it records on ``result.cv_targeting``.  Falls back to
        pooled targeting with a warning when there are no validation folds to target
        within (``cross_fit=False``, or a single fold).
    cv_evaluation:
        Report the *canonical* CV-TMLE estimate rather than the pooled one.  Requires
        ``targeting_scheme="fold"``.

        Fold-specific targeting is only the first of canonical CV-TMLE's three parts;
        the other two are fold-wise evaluation of the parameter and the cross-validated
        variance.  With ``cv_evaluation=False`` (default) this estimator does neither:
        the fold-targeted predictions are stitched back together and the *pooled*
        plug-in and pooled influence-curve standard error are reported.  For ``ate``,
        ``ey1`` and ``ey0`` that is the same number -- those are linear in the targeted
        predictions, so the pooled mean is the fold average -- but for ``rr``, ``or``,
        ``att`` and ``atc`` it is not: a ratio of means is not a mean of ratios, and the
        pooled ATT weights by the whole sample's treated share rather than each fold's.

        With ``cv_evaluation=True`` each estimand is computed inside each validation
        fold and averaged over folds with weight ``1/V``, and the standard error is the
        cross-validated one (:func:`~cleverly.inference.cross_validated_variance`).
        Ratios are averaged on the log scale, which is where their influence curve and
        confidence interval already live.  Both reports are always available on
        ``result.cv_targeting`` regardless of this setting.
    n_folds, learner_folds:
        Outer cross-fitting folds, and the inner folds a Super Learner uses to score
        its candidates.
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
        g_bounds: GBounds = "auto",
        q_bounds: tuple[float, float] | None = None,
        alpha: float = 0.9995,
        nuisance_bound: float = DEFAULT_NUISANCE_BOUND,
        target_weights: bool = False,
        screen_treatment: bool = False,
        screen_threshold: float = 0.1,
        min_retain: int | None = None,
        estimands: Sequence[Estimand] | str | None = None,
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
        self.g_bounds = g_bounds
        self.q_bounds = q_bounds
        self.alpha = alpha
        self.nuisance_bound = nuisance_bound
        self.target_weights = target_weights
        self.screen_treatment = screen_treatment
        self.screen_threshold = screen_threshold
        self.min_retain = min_retain
        self.estimands = estimands
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
        if self.cv_evaluation and self.targeting_scheme != "fold":
            raise ValueError(
                "cv_evaluation=True reports the estimand fold by fold and needs a "
                "fold-specific fit to evaluate; pass targeting_scheme='fold'"
            )
        if self.targeting_scheme == "fold" and not self.cross_fit:
            warnings.warn(
                "targeting_scheme='fold' needs validation folds to target within, and "
                "cross_fit=False leaves none; falling back to pooled targeting. Set "
                "cross_fit=True for a fold-targeted CV-TMLE.",
                UserWarning,
                stacklevel=3,
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
    ) -> TMLEResult | TMLEResultSet:
        """Fit the estimator.

        Parameters
        ----------
        data:
            A pandas or polars dataframe, or a prepared
            :class:`~cleverly.data.CausalData`.
        outcome, treatment, covariates, delta, weights, id, intermediate:
            Column names, when ``data`` is a dataframe.  ``covariates=None`` uses every
            column not claimed by another role.
        weights_type, weights_estimated:
            How to read ``weights``.  Supplying weights changes the estimand to the
            causal parameter in the weight-tilted population -- see
            :mod:`cleverly.data.weighting` and
            :meth:`~cleverly.data.CausalData.from_frame`.

        Returns
        -------
        A :class:`~cleverly.estimators.base.TMLEResult`, or a
        :class:`~cleverly.estimators.base.TMLEResultSet` with one result per level of
        the intermediate variable when ``intermediate`` is supplied.
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
        )

        if not prepared.has_intermediate:
            return self._fit_single(prepared, intermediate_value=None)

        results = {
            value: self._fit_single(prepared, intermediate_value=value) for value in (0.0, 1.0)
        }
        return TMLEResultSet(results, prepared.intermediate_name or "Z")

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
        weights_type: str = "probability",
        weights_estimated: bool = False,
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
                for value in (outcome, treatment, covariates, delta, weights, id, intermediate)
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
            family=self.family,
        )

    def _fit_single(self, data: CausalData, *, intermediate_value: float | None) -> TMLEResult:
        """Fit for one value of the intermediate (or for no intermediate at all)."""
        estimands = resolve_estimands(self.estimands, data.family)
        scaler = self._scaler(data)
        folds = self._folds(data)
        config = self._config(data, estimands, scaler, folds)
        nuisance, extra = self._nuisances(data, folds, scaler, config, intermediate_value)
        self._warn_on_positivity(nuisance, config, intermediate_value)
        self._warn_on_estimated_weights(data)

        estimates, fluctuations, cv_detail = self._retarget_detailed(
            data,
            nuisance,
            estimands=estimands,
            intermediate_value=intermediate_value,
            g_bounds=config.g_bounds,
            g_bounds_conditional=config.g_bounds_conditional,
        )

        result = TMLEResult(
            estimates=estimates,
            fluctuations=fluctuations,
            nuisance=nuisance,
            data=data,
            config=config,
            estimator=self,
            provenance=provenance_record(
                data, folds, random_state=self.random_state, run_id=self.run_id
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

    # ------------------------------------------------------------- internals

    def _scaler(self, data: CausalData) -> OutcomeScaler:
        """The outcome transformation: identity for a binary outcome, scaling else."""
        if data.family == "binomial":
            if self.q_bounds is not None:
                raise ValueError("q_bounds does not apply to a binary outcome")
            return OutcomeScaler.identity()
        observed = data.outcome[data.observed]
        return OutcomeScaler.from_outcome(observed, self.q_bounds)

    def _folds(self, data: CausalData) -> Folds:
        if not self.cross_fit:
            return Folds.single(data.n)
        return make_folds(
            data.n,
            self.n_folds,
            stratify=data.treatment,
            cluster=data.cluster,
            random_state=self.random_state,
        )

    def _resolve_learner(
        self,
        spec: Learner | str | Sequence[Any] | None,
        *,
        task: Task,
        fallback: Learner | str | Sequence[Any] | None = None,
    ) -> Learner:
        """Turn a learner specification into a fitted-per-fold estimator."""
        return resolve_learner(
            spec,
            task=task,
            n_folds=self.learner_folds,
            random_state=self.random_state,
            fallback=fallback,
        )

    def _fit_nuisances(
        self,
        data: CausalData,
        folds: Folds,
        scaler: OutcomeScaler,
        intermediate_value: float | None,
    ) -> NuisanceEstimates:
        outcome_task: Task = "classification" if data.family == "binomial" else "regression"
        return fit_nuisances(
            data,
            outcome_learner=self._resolve_learner(self.outcome_learner, task=outcome_task),
            treatment_learner=self._resolve_learner(self.treatment_learner, task="classification"),
            missingness_learner=(
                self._resolve_learner(
                    self.missingness_learner,
                    task="classification",
                    fallback=self.treatment_learner,
                )
                if data.has_missing_outcome
                else None
            ),
            intermediate_learner=(
                self._resolve_learner(
                    self.intermediate_learner,
                    task="classification",
                    fallback=self.treatment_learner,
                )
                if data.has_intermediate
                else None
            ),
            folds=folds,
            scaler=scaler,
            intermediate_value=intermediate_value,
            screen_treatment=self.screen_treatment,
            screen_threshold=self.screen_threshold,
            min_retain=self.min_retain,
            n_jobs=self.n_jobs,
        )

    def _nuisances(
        self,
        data: CausalData,
        folds: Folds,
        scaler: OutcomeScaler,
        config: TMLEConfig,
        intermediate_value: float | None,
    ) -> tuple[NuisanceEstimates, dict[str, Any]]:
        """The nuisance fits to target against, plus any variant-specific diagnostics.

        The extension point for TMLE variants that differ only in *which* nuisance
        estimate they hand to the targeting step -- :class:`~cleverly.CTMLE` selects a
        propensity model here and reports the selection path in the extras.
        """
        return self._fit_nuisances(data, folds, scaler, intermediate_value), {}

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
        propensity = nuisance.propensity
        outside = float(np.mean((propensity < lower) | (propensity > upper)))
        if outside > _TRUNCATION_WARN_FRACTION:
            warnings.warn(
                f"{outside:.1%} of estimated propensity scores fall outside the truncation "
                f"bounds [{lower:.4g}, {upper:.4g}], so those units' contributions rest on "
                "extrapolation rather than data. Inspect res.sensitivity.positivity() and "
                "res.sensitivity.truncation_curve() before trusting the estimate.",
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
        fold_epsilon: dict[str, tuple[tuple[float, ...], ...]] = {}
        indices: list[IntArray] = []

        for group in self._groups(requested):
            bounds = mean_bounds if group == "mean" else conditional_bounds
            submodel = self._submodel(
                data, nuisance, group, bounds, intermediate_value, missingness, nuisance_bound
            )
            fluctuation = self._solve(data, nuisance, submodel)
            fluctuations[group] = fluctuation

            pooled = self._estimates_for(
                data, nuisance, group, submodel, fluctuation, requested, level
            )
            pooled_report.update(pooled)
            if not fluctuation.folds:
                estimates.update(pooled)
                continue

            indices = [record.index for record in fluctuation.folds]
            fold_epsilon[group] = tuple(
                tuple(record.epsilon.tolist()) for record in fluctuation.folds
            )
            per_fold = [
                self._fold_estimates(
                    data, nuisance, group, submodel, fluctuation, tuple(pooled), level, index
                )
                for index in indices
            ]
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

        ordered = {name: estimates[name] for name in requested if name in estimates}
        detail = (
            CVTargeting(
                n_folds=len(indices),
                fold_sizes=tuple(int(index.size) for index in indices),
                variance={name: value.variance for name, value in canonical_report.items()},
                fold_estimates=fold_estimates,
                fold_epsilon=fold_epsilon,
                pooled={name: pooled_report[name] for name in requested if name in pooled_report},
                canonical={
                    name: canonical_report[name] for name in requested if name in canonical_report
                },
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
        from the canonical estimate altogether and says so.

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

    def _submodel(
        self,
        data: CausalData,
        nuisance: NuisanceEstimates,
        group: TargetGroup,
        bounds: tuple[float, float],
        intermediate_value: float | None,
        missingness_override: FloatArray | None,
        nuisance_bound: float | None = None,
    ) -> Submodel:
        return build_submodel(
            data,
            nuisance,
            group,
            bounds=bounds,
            nuisance_bound=(
                self.nuisance_bound if nuisance_bound is None else float(nuisance_bound)
            ),
            intermediate_value=intermediate_value,
            missingness_override=missingness_override,
        )

    def _solve(
        self, data: CausalData, nuisance: NuisanceEstimates, submodel: Submodel
    ) -> Fluctuation:
        """Solve the fluctuation, pooled over folds or one fluctuation per fold."""
        scaled = nuisance.scaler.scale(data.outcome)
        if self.targeting_scheme == "fold" and self.cross_fit:
            if not nuisance.folds.is_single:
                return self._solve_by_fold(data, nuisance, submodel, scaled)
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
        return self._solve_rows(scaled, nuisance.outcome, submodel, data.weights, data.observed)

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
        submodel: Submodel,
        scaled: FloatArray,
    ) -> Fluctuation:
        """The targeting step of CV-TMLE: a separate fluctuation on each fold.

        Each fold's ``epsilon`` is fit only against rows whose nuisance predictions came
        from a model trained on the other folds -- the targeting step of Zheng & van der
        Laan's (2011) CV-TMLE.  What changes relative to pooled targeting is the
        *coefficient*.  With ``cross_fit=True`` the pooled scheme already draws every
        row's nuisance prediction from a model that never saw it, but its single
        ``epsilon_hat`` is fit across all validation rows at once, so every row's
        fluctuation depends on every other row's outcome and the targeted prediction is
        out of sample only in its nuisance component.  Fitting within a fold removes
        that last coupling: no row contributes to the coefficient that fluctuates it,
        and the targeted prediction is out of sample throughout.

        The fold-specific targeted predictions are stitched back into a full-length fit.
        Because each fold's score is zero on its own rows, the pooled score -- a sum over
        folds -- is zero too, so the estimating equation is still solved exactly on the
        full sample.  The reported ``epsilon`` is the mass-weighted average across folds
        and is a summary only; the per-fold values are kept in
        :attr:`~cleverly.fluctuation.Fluctuation.folds`.

        Stitching is where this stops short of canonical CV-TMLE, which evaluates the
        parameter fold by fold rather than once over the reassembled fit.  The two agree
        for estimands linear in the targeted predictions and diverge for the rest; see
        ``cv_evaluation``, which reports the canonical construction instead.
        """
        n = data.n
        observed = np.empty(n)
        at_one = np.empty(n)
        at_zero = np.empty(n)
        fold_records: list[FoldFluctuation] = []
        masses = []
        traces = []
        iterations = 0

        for _, test in nuisance.folds:
            fold_fluctuation = self._solve_rows(
                scaled[test],
                InitialFit(
                    nuisance.outcome.observed[test],
                    nuisance.outcome.at_one[test],
                    nuisance.outcome.at_zero[test],
                ),
                restrict(submodel, test),
                data.weights[test],
                data.observed[test],
                warn=False,
            )
            observed[test] = fold_fluctuation.targeted.observed
            at_one[test] = fold_fluctuation.targeted.at_one
            at_zero[test] = fold_fluctuation.targeted.at_zero
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
            iterations += fold_fluctuation.n_iter

        targeted = InitialFit(observed, at_one, at_zero)
        weights_array = np.asarray(masses)
        epsilon = np.average(
            np.vstack([record.epsilon for record in fold_records]), axis=0, weights=weights_array
        )
        score = score_columns(
            scaled, targeted.observed, submodel.observed, data.weights, data.observed
        )
        scale = score_scale(submodel.observed, data.weights, data.observed)
        return Fluctuation(
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
        )

    def _estimates_for(
        self,
        data: CausalData,
        nuisance: NuisanceEstimates,
        group: TargetGroup,
        submodel: Submodel,
        fluctuation: Fluctuation,
        requested: Sequence[str],
        alpha_sig: float,
        index: IntArray | None = None,
        drop_undefined: bool = False,
    ) -> dict[str, ParameterEstimate]:
        """Build every estimand that this fluctuation supports.

        ``index`` restricts every input to one validation fold, which is what the
        canonical CV-TMLE evaluation needs; ``None`` uses the whole sample.  Weights are
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
        if index is not None:
            scaled = scaled[index]
            targeted = _slice_fit(targeted, index)
            submodel = restrict(submodel, index)
            weights = weights[index]
            weights = weights / weights.mean()
            observed = observed[index]
            treatment = treatment[index]
            cluster = None if cluster is None else cluster[index]
            n = int(index.size)

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
        )
        # One context per fluctuation, shared by every target in the group: the five
        # mean-group estimands are different functionals of the same targeted
        # distribution, and `context.means` computes the counterfactual means once.
        out: dict[str, ParameterEstimate] = {}
        for target in targets_for(group, requested):
            try:
                out[target.name] = target.build(context)
            except ValueError:
                # A target that declares `undefined_when` may legitimately fail on a
                # subsample; anything else failing is a bug and must not be swallowed.
                if not (drop_undefined and target.undefined_when):
                    raise
        return out

    def _bootstrap_point_estimates(
        self, data: CausalData, intermediate_value: float | None
    ) -> Mapping[str, float]:
        """One bootstrap replicate: a full refit, point estimates only.

        Goes through :meth:`_nuisances` rather than :meth:`_fit_nuisances` so that a
        variant which *selects* a nuisance model repeats that selection in every
        replicate -- otherwise the bootstrap would understate the variability the
        selection itself contributes.
        """
        estimands = resolve_estimands(self.estimands, data.family)
        scaler = self._scaler(data)
        folds = self._folds(data)
        config = self._config(data, estimands, scaler, folds)
        nuisance, _ = self._nuisances(data, folds, scaler, config, intermediate_value)
        estimates, _ = self.retarget(
            data,
            nuisance,
            estimands=estimands,
            intermediate_value=intermediate_value,
        )
        return {name: estimate.psi for name, estimate in estimates.items()}


def _slice_fit(fit: InitialFit, index: IntArray) -> InitialFit:
    """The targeted (or initial) predictions for one subset of rows."""
    return InitialFit(fit.observed[index], fit.at_one[index], fit.at_zero[index])


def _average_over_folds(
    per_fold: Sequence[Mapping[str, ParameterEstimate]],
    supported: Sequence[str],
    indices: Sequence[IntArray],
    *,
    n: int,
    cluster: IntArray | None,
    alpha: float,
) -> dict[str, ParameterEstimate]:
    """Assemble the canonical CV-TMLE estimate from its fold-wise pieces.

    The point estimate is the unweighted ``1/V`` average of the fold plug-ins, matching
    Zheng & van der Laan and matching the fold weighting
    :func:`~cleverly.inference.cross_validated_variance` already uses -- so the estimate
    and its variance are weighted the same way, with no extra knob.  Observation weights
    still apply *within* a fold.  Ratios are averaged on the log scale, which is where
    their influence curve and Wald interval live, so that ``psi == exp(log_psi)`` holds
    and :attr:`~cleverly.inference.ParameterEstimate.ci` stays on the boundary-respecting
    scale.

    The influence curve is the fold-specific curves stitched back together by index.
    Unlike the pooled curve it is centred at each fold's own estimate, so its mean is
    zero *within* every fold -- which is exactly the property the uncentred second moment
    in ``cross_validated_variance`` assumes.
    """
    out: dict[str, ParameterEstimate] = {}
    dropped: list[str] = []
    n_clusters = n if cluster is None else int(np.unique(cluster).size)

    for name in supported:
        if not all(name in values for values in per_fold):
            dropped.append(name)
            continue
        parts = [values[name] for values in per_fold]
        influence_curve = np.empty(n, dtype=float)
        for index, part in zip(indices, parts, strict=True):
            influence_curve[index] = part.influence_curve

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
            variance=cross_validated_variance(influence_curve, indices, cluster),
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
) -> TMLEResult | TMLEResultSet:
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
    >>> res = tmle(Y, A, W, outcome_learner="glm", treatment_learner="glm")
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
