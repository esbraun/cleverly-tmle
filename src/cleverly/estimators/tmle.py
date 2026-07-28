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
   coefficient.  This is the step that makes the estimator doubly robust: it is
   consistent if *either* ``g`` or ``Qbar`` is consistent, and efficient if both are.
3. **Plug in.** Average the targeted predictions to get the estimate, and use the
   influence curve for inference.

Because the targeting step solves an estimating equation rather than optimising a
prediction loss, the resulting estimate is not shrunk toward the null by
regularisation in the nuisance models -- which is what separates TMLE from
plugging machine-learning predictions into a G-computation formula.

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
    Learner,
    TargetingMethod,
    TargetingScheme,
)
from ..data.causal_data import CausalData
from ..exceptions import PositivityWarning
from ..fluctuation.iterative import (
    Fluctuation,
    InitialFit,
    _relative,
    _score_scale,
    solve_fluctuation,
)
from ..fluctuation.one_step import solve_one_step
from ..fluctuation.submodel import Submodel, TargetGroup, submodel_for
from ..inference.bootstrap import Resampling, run_bootstrap
from ..inference.influence import (
    ParameterEstimate,
    atc_estimate,
    att_estimate,
    counterfactual_means,
    make_estimate,
    ratio_estimates,
    unscale,
)
from ..inference.multiplier import MultiplierKind, simultaneous_bands
from ..learners._fitting import Task
from ..learners.crossfit import Folds, make_folds
from ..learners.super_learner import SuperLearner
from ..utils.bounds import OutcomeScaler, resolve_g_bounds
from ..utils.frames import is_dataframe
from ._nuisance import NuisanceEstimates, fit_nuisances
from .base import (
    MEAN_GROUP_ESTIMANDS,
    TMLEConfig,
    TMLEResult,
    TMLEResultSet,
    attach_bootstrap,
    resolve_estimands,
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
        ``"pooled"`` fits one fluctuation on the cross-fitted predictions;
        ``"fold"`` fits a fluctuation per validation fold -- the CV-TMLE of Zheng &
        van der Laan (2011).  Both solve the pooled score equation exactly.
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
    step_size, max_iter, tol:
        Targeting-step controls.
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
        n_jobs: int = 1,
    ) -> None:
        self.outcome_learner = outcome_learner
        self.treatment_learner = treatment_learner
        self.missingness_learner = missingness_learner
        self.intermediate_learner = intermediate_learner
        self.family = family
        self.fluctuation = fluctuation
        self.targeting = targeting
        self.cross_fit = cross_fit
        self.targeting_scheme = targeting_scheme
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
            id=id,
            intermediate=intermediate,
        )

        if not prepared.has_intermediate:
            return self._fit_single(prepared, intermediate_value=None)

        results = {
            value: self._fit_single(prepared, intermediate_value=value) for value in (0.0, 1.0)
        }
        return TMLEResultSet(results, prepared.intermediate_name or "Z")

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
    ) -> CausalData:
        """Coerce whatever the caller passed into a validated :class:`CausalData`."""
        if isinstance(data, CausalData):
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
            id=id,
            intermediate=intermediate,
            family=self.family,
        )

    def _fit_single(self, data: CausalData, *, intermediate_value: float | None) -> TMLEResult:
        """Fit for one value of the intermediate (or for no intermediate at all)."""
        estimands = resolve_estimands(self.estimands, data.family)
        scaler = self._scaler(data)
        folds = self._folds(data)
        nuisance = self._fit_nuisances(data, folds, scaler, intermediate_value)

        config = self._config(data, estimands, scaler, folds)
        self._warn_on_positivity(nuisance, config)

        estimates, fluctuations = self.retarget(
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
            intermediate_value=intermediate_value,
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
        if spec is None:
            spec = fallback
        if spec is None or isinstance(spec, (str, list, tuple)):
            return SuperLearner(
                library="default" if spec is None else spec,
                task=task,
                n_folds=self.learner_folds,
                clip=(0.0, 1.0),
                random_state=self.random_state,
                n_jobs=1,
            )
        return spec

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

    def _config(
        self,
        data: CausalData,
        estimands: tuple[str, ...],
        scaler: OutcomeScaler,
        folds: Folds,
    ) -> TMLEConfig:
        return TMLEConfig(
            family=data.family,
            fluctuation=self.fluctuation,
            targeting=self.targeting,
            targeting_scheme=self.targeting_scheme if self.cross_fit else "pooled",
            cross_fit=self.cross_fit,
            n_folds=folds.n_folds,
            g_bounds=resolve_g_bounds(self.g_bounds, data.n, for_att=False),
            g_bounds_conditional=resolve_g_bounds(self.g_bounds, data.n, for_att=True),
            missingness_bound=self.nuisance_bound,
            q_bounds=None if scaler.is_identity else (scaler.lower, scaler.upper),
            alpha=self.alpha,
            target_weights=self.target_weights,
            screen_treatment=self.screen_treatment,
            estimands=estimands,
            alpha_sig=self.alpha_sig,
            random_state=self.random_state,
            n_bootstrap=self.n_bootstrap,
        )

    def _warn_on_positivity(self, nuisance: NuisanceEstimates, config: TMLEConfig) -> None:
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
        alpha_sig: float | None = None,
    ) -> tuple[dict[str, ParameterEstimate], dict[str, Fluctuation]]:
        """Run the targeting step and build estimates from cached nuisance fits.

        Separated from :meth:`fit` because every sensitivity analysis is exactly this
        operation with one input perturbed -- a different truncation bound, a tilted
        missingness mechanism -- and re-running it costs a fraction of a full refit.
        """
        requested = tuple(estimands)
        level = self.alpha_sig if alpha_sig is None else alpha_sig
        mean_bounds = g_bounds or resolve_g_bounds(self.g_bounds, data.n, for_att=False)
        conditional_bounds = g_bounds_conditional or resolve_g_bounds(
            self.g_bounds, data.n, for_att=True
        )

        estimates: dict[str, ParameterEstimate] = {}
        fluctuations: dict[str, Fluctuation] = {}

        for group in self._groups(requested):
            bounds = mean_bounds if group == "mean" else conditional_bounds
            submodel = self._submodel(
                data, nuisance, group, bounds, intermediate_value, missingness
            )
            fluctuation = self._solve(data, nuisance, submodel)
            fluctuations[group] = fluctuation
            estimates.update(
                self._estimates_for(data, nuisance, group, submodel, fluctuation, requested, level)
            )

        ordered = {name: estimates[name] for name in requested if name in estimates}
        return ordered, fluctuations

    @staticmethod
    def _groups(estimands: Sequence[str]) -> list[TargetGroup]:
        """Which fluctuations must be fit to cover the requested estimands.

        Each estimand family gets its own targeting step, because each has its own
        efficient influence function and therefore its own score equation to solve.
        """
        groups: list[TargetGroup] = []
        if any(name in MEAN_GROUP_ESTIMANDS for name in estimands):
            groups.append("mean")
        if "att" in estimands:
            groups.append("att")
        if "atc" in estimands:
            groups.append("atc")
        return groups

    def _submodel(
        self,
        data: CausalData,
        nuisance: NuisanceEstimates,
        group: TargetGroup,
        bounds: tuple[float, float],
        intermediate_value: float | None,
        missingness_override: FloatArray | None,
    ) -> Submodel:
        propensity = nuisance.bounded_propensity(bounds)
        missingness = (
            nuisance.bounded_missingness(self.nuisance_bound)
            if missingness_override is None
            else np.clip(np.asarray(missingness_override, dtype=float), self.nuisance_bound, 1.0)
        )
        intermediate_density = None
        selection = None
        if data.has_intermediate:
            assert intermediate_value is not None and data.intermediate is not None
            intermediate_density = nuisance.intermediate_density(
                intermediate_value, self.nuisance_bound
            )
            selection = (data.intermediate == intermediate_value).astype(float)
        return submodel_for(
            group,
            data.treatment,
            propensity,
            treated_fraction=data.treated_fraction,
            missingness=missingness,
            intermediate_density=intermediate_density,
            selection=selection,
        )

    def _solve(
        self, data: CausalData, nuisance: NuisanceEstimates, submodel: Submodel
    ) -> Fluctuation:
        """Solve the fluctuation, pooled over folds or one fluctuation per fold."""
        scaled = nuisance.scaler.scale(data.outcome)
        if self.targeting_scheme == "fold" and self.cross_fit and not nuisance.folds.is_single:
            return self._solve_by_fold(data, nuisance, submodel, scaled)
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
        if self.targeting == "one_step":
            return solve_one_step(
                scaled,
                initial,
                submodel,
                weights,
                observed,
                target_weights=self.target_weights,
                alpha=self.alpha,
                step_size=self.step_size,
                tol=self.tol,
                warn=warn,
            )
        return solve_fluctuation(
            scaled,
            initial,
            submodel,
            weights,
            observed,
            kind=self.fluctuation,
            target_weights=self.target_weights,
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=self.tol,
            warn=warn,
        )

    def _solve_by_fold(
        self,
        data: CausalData,
        nuisance: NuisanceEstimates,
        submodel: Submodel,
        scaled: FloatArray,
    ) -> Fluctuation:
        """CV-TMLE: a separate fluctuation on each validation fold.

        The fold-specific targeted predictions are stitched back into a full-length
        fit.  Because each fold's score is zero on its own rows, the pooled score --
        a sum over folds -- is zero too, so the estimating equation is still solved
        exactly on the full sample.
        """
        n = data.n
        observed = np.empty(n)
        at_one = np.empty(n)
        at_zero = np.empty(n)
        epsilons = []
        masses = []
        traces = []
        iterations = 0
        converged = True

        for _, test in nuisance.folds:
            fold_fluctuation = self._solve_rows(
                scaled[test],
                InitialFit(
                    nuisance.outcome.observed[test],
                    nuisance.outcome.at_one[test],
                    nuisance.outcome.at_zero[test],
                ),
                Submodel(
                    submodel.observed[test],
                    submodel.at_one[test],
                    submodel.at_zero[test],
                    submodel.names,
                    submodel.group,
                ),
                data.weights[test],
                data.observed[test],
                warn=False,
            )
            observed[test] = fold_fluctuation.targeted.observed
            at_one[test] = fold_fluctuation.targeted.at_one
            at_zero[test] = fold_fluctuation.targeted.at_zero
            epsilons.append(fold_fluctuation.epsilon)
            masses.append(float(data.weights[test].sum()))
            traces.append(fold_fluctuation.trace[-1] if fold_fluctuation.trace else float("nan"))
            iterations += fold_fluctuation.n_iter
            converged = converged and fold_fluctuation.converged

        targeted = InitialFit(observed, at_one, at_zero)
        weights_array = np.asarray(masses)
        epsilon = np.average(np.vstack(epsilons), axis=0, weights=weights_array)
        score = _score_of(scaled, targeted, submodel, data.weights, data.observed)
        scale = _score_scale(submodel.observed, data.weights, data.observed)
        return Fluctuation(
            epsilon=epsilon,
            targeted=targeted,
            score=score,
            converged=bool(_relative(score, scale) <= self.tol),
            n_iter=iterations,
            trace=tuple(traces),
            method="iterative" if self.targeting == "iterative" else "one_step",
            names=submodel.names,
            score_scale=scale,
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
    ) -> dict[str, ParameterEstimate]:
        """Build every estimand that this fluctuation supports."""
        scaled = nuisance.scaler.scale(data.outcome)
        scaler = nuisance.scaler
        out: dict[str, ParameterEstimate] = {}
        common: dict[str, Any] = {
            "n": data.n,
            "cluster": data.cluster,
            "alpha": alpha_sig,
        }

        if group == "mean":
            psi_one, ic_one, psi_zero, ic_zero = counterfactual_means(
                scaled, fluctuation.targeted, submodel, data.weights, data.observed
            )
            if "ey1" in requested:
                value, ic = unscale(psi_one, ic_one, scaler, "level")
                out["ey1"] = make_estimate("ey1", value, ic, scale="level", **common)
            if "ey0" in requested:
                value, ic = unscale(psi_zero, ic_zero, scaler, "level")
                out["ey0"] = make_estimate("ey0", value, ic, scale="level", **common)
            if "ate" in requested:
                value, ic = unscale(psi_one - psi_zero, ic_one - ic_zero, scaler, "difference")
                out["ate"] = make_estimate("ate", value, ic, scale="difference", **common)
            ratios = tuple(name for name in ("rr", "or") if name in requested)
            if ratios:
                out.update(
                    ratio_estimates(
                        psi_one,
                        ic_one,
                        psi_zero,
                        ic_zero,
                        n=data.n,
                        cluster=data.cluster,
                        alpha=alpha_sig,
                        which=ratios,
                    )
                )
            return out

        estimator_fn = att_estimate if group == "att" else atc_estimate
        psi, ic = estimator_fn(
            scaled,
            fluctuation.targeted,
            submodel,
            data.treatment,
            data.weights,
            data.observed,
        )
        value, unscaled_ic = unscale(psi, ic, scaler, "difference")
        out[group] = make_estimate(group, value, unscaled_ic, scale="difference", **common)
        return out

    def _bootstrap_point_estimates(
        self, data: CausalData, intermediate_value: float | None
    ) -> Mapping[str, float]:
        """One bootstrap replicate: a full refit, point estimates only."""
        estimands = resolve_estimands(self.estimands, data.family)
        scaler = self._scaler(data)
        folds = self._folds(data)
        nuisance = self._fit_nuisances(data, folds, scaler, intermediate_value)
        estimates, _ = self.retarget(
            data,
            nuisance,
            estimands=estimands,
            intermediate_value=intermediate_value,
        )
        return {name: estimate.psi for name, estimate in estimates.items()}


def _score_of(
    scaled: FloatArray,
    targeted: InitialFit,
    submodel: Submodel,
    weights: FloatArray,
    observed: BoolArray,
) -> FloatArray:
    """``mean(w * h * (Y - Q*))`` per clever-covariate column."""
    residual = np.where(observed, scaled - targeted.observed, 0.0)
    return np.asarray(((weights * residual)[:, None] * submodel.observed).mean(axis=0), dtype=float)


def tmle(
    Y: Any,
    A: Any,
    W: Any,
    *,
    Delta: Any = None,
    Z: Any = None,
    obsWeights: Any = None,
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
        id=id,
        intermediate=Z,
        family=kwargs.get("family", "auto"),
    )
    estimator = TMLE(**kwargs)
    return estimator.fit(data)
