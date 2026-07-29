"""Cross-fitted estimation of the nuisance parameters.

TMLE needs three or four regressions before any targeting happens:

``g(W) = P(A = 1 | W)``
    the treatment mechanism, which enters the clever covariate;
``Qbar(A, W) = E[Y | A, W, Delta = 1]``
    the outcome regression, which is what gets fluctuated;
``P(Delta = 1 | A, W)``
    the missingness mechanism, when outcomes can be unobserved;
``P(Z = 1 | A, W)``
    the intermediate mechanism, for controlled direct effects.

All of them are fit **out of fold**: each observation's prediction comes from a
model that never saw it.  With machine-learning nuisance estimators this is not
optional -- in-sample predictions are overfit, the residuals they produce are too
small, and the targeting step cannot repair the resulting bias.  Cross-fitting is
also what lets the influence-curve variance stay valid without a Donsker
condition on the nuisance estimators.

The *raw*, un-truncated propensity is retained alongside the bounded one so that
sensitivity analyses can re-truncate and re-target without refitting anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .._typing import BoolArray, FloatArray, IntArray, Learner
from ..data.causal_data import CausalData
from ..fluctuation.iterative import InitialFit
from ..learners._fitting import Task, as_target, fit_learner, predict_mean
from ..learners.crossfit import Folds
from ..learners.screeners import CorrelationScreener
from ..learners.super_learner import SuperLearner, SuperLearnerDiagnostics
from ..utils.bounds import OutcomeScaler, bound
from ..utils.parallel import map_parallel
from .direct_effect import check_level

__all__ = ["NuisanceEstimates", "cross_fit_predictions", "fit_nuisances"]


@dataclass(frozen=True)
class NuisanceEstimates:
    """Cross-fitted nuisance predictions and the diagnostics that came with them.

    Attributes
    ----------
    propensity:
        Out-of-fold ``g(W)``, *not* truncated.  Truncation is applied per estimand
        family at targeting time, because the ATT tolerates far less extrapolation
        than the ATE and so uses a tighter bound.
    outcome:
        Initial outcome regression on the ``[0, 1]`` scale, at the observed
        treatment and at both counterfactual arms.
    missingness, intermediate:
        ``(n, 2)`` arrays indexed by treatment arm, or ``None`` when not applicable.
    scaler:
        The transformation used to put the outcome on ``[0, 1]``.
    diagnostics:
        Per-nuisance Super Learner weights and cross-validated risks, when a
        Super Learner was used.
    """

    propensity: FloatArray
    outcome: InitialFit
    scaler: OutcomeScaler
    folds: Folds
    missingness: FloatArray | None = None
    intermediate: FloatArray | None = None
    treatment_covariates: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)
    outcome_task: Task = "regression"

    @property
    def n(self) -> int:
        return int(self.propensity.shape[0])

    def bounded_propensity(self, bounds: tuple[float, float]) -> FloatArray:
        """``g(W)`` truncated into ``bounds``."""
        return bound(self.propensity, bounds[0], bounds[1])

    def bounded_missingness(self, lower: float) -> FloatArray | None:
        """``P(Delta = 1 | A, W)`` truncated away from zero."""
        if self.missingness is None:
            return None
        return bound(self.missingness, lower, 1.0)

    def intermediate_density(self, value: float, lower: float) -> FloatArray | None:
        """``P(Z = z | A = a, W)`` for the targeted ``z``, truncated away from zero.

        Validates ``value`` because the branch below is silent about a level it does not
        recognise: anything other than ``1`` would fall through to the complement and
        return a perfectly plausible ``P(Z = 0 | A, W)`` for a parameter nobody asked
        for.  This accessor is the one place the ``z`` / ``1 - z`` convention lives, and
        every diagnostic that reports a denominator has to come through it.
        """
        if self.intermediate is None:
            return None
        level = check_level(value)
        probs = self.intermediate if level == 1.0 else 1.0 - self.intermediate
        return bound(probs, lower, 1.0)


def cross_fit_predictions(
    learner: Learner,
    design: FloatArray,
    target: FloatArray,
    weights: FloatArray,
    folds: Folds,
    *,
    task: Task,
    predict_designs: dict[str, FloatArray],
    fit_mask: BoolArray | None = None,
    groups: IntArray | None = None,
    clip: tuple[float, float] | None = None,
    n_jobs: int = 1,
) -> tuple[dict[str, FloatArray], list[SuperLearnerDiagnostics]]:
    """Out-of-fold predictions of one nuisance regression.

    Parameters
    ----------
    design, target, weights:
        Training data for the regression.
    predict_designs:
        Named design matrices to predict on -- for the outcome regression these are
        the observed treatment and the two counterfactual arms, so a single pass
        over the folds produces everything the fluctuation needs.
    fit_mask:
        Rows eligible for *training*.  The outcome regression is fit only where the
        outcome is observed, but must still predict everywhere.
    groups:
        Cluster codes, forwarded to a Super Learner so its inner folds keep clusters
        intact too.

    Returns
    -------
    Predictions per named design, and the Super Learner diagnostics per fold (empty
    when the learner is not a Super Learner).
    """
    n = design.shape[0]
    mask = np.ones(n, dtype=bool) if fit_mask is None else np.asarray(fit_mask, dtype=bool)
    if not mask.any():
        raise ValueError("no rows are eligible for fitting this nuisance model")

    if folds.is_single:
        rows = np.flatnonzero(mask)
        model = _fit_with_groups(learner, design, target, weights, rows, task, groups)
        predictions = {
            name: _clip(predict_mean(model, matrix, task), clip)
            for name, matrix in predict_designs.items()
        }
        diagnostics = getattr(model, "diagnostics_", None)
        return predictions, [diagnostics] if diagnostics is not None else []

    jobs = [(train, test) for train, test in folds]

    def run_fold(train: IntArray, test: IntArray) -> tuple[IntArray, dict[str, FloatArray], Any]:
        rows = train[mask[train]]
        if rows.size == 0:
            raise ValueError(
                "a cross-fitting fold has no trainable rows for a nuisance model; "
                "reduce n_folds or supply cluster-aware folds"
            )
        model = _fit_with_groups(learner, design, target, weights, rows, task, groups)
        predictions = {
            name: _clip(predict_mean(model, matrix[test], task), clip)
            for name, matrix in predict_designs.items()
        }
        return test, predictions, getattr(model, "diagnostics_", None)

    results = map_parallel(run_fold, jobs, n_jobs=n_jobs)
    out = {name: np.empty(n, dtype=float) for name in predict_designs}
    diagnostics_list: list[SuperLearnerDiagnostics] = []
    for test, predictions, diagnostics in results:
        for name, values in predictions.items():
            out[name][test] = values
        if diagnostics is not None:
            diagnostics_list.append(diagnostics)
    return out, diagnostics_list


def _fit_with_groups(
    learner: Learner,
    design: FloatArray,
    target: FloatArray,
    weights: FloatArray,
    rows: IntArray,
    task: Task,
    groups: IntArray | None,
) -> Learner:
    """Fit a learner, passing cluster codes on to a Super Learner's inner folds."""
    if isinstance(learner, SuperLearner) and groups is not None:
        from sklearn.base import clone

        model = clone(learner)
        model.fit(
            design[rows],
            as_target(target[rows], task),
            sample_weight=weights[rows],
            groups=np.asarray(groups)[rows],
        )
        return model
    return fit_learner(
        learner,
        design[rows],
        as_target(target[rows], task),
        weights[rows],
        warn_unweighted=False,
    )


def _clip(values: FloatArray, clip: tuple[float, float] | None) -> FloatArray:
    if clip is None:
        return np.asarray(values, dtype=float)
    return np.clip(np.asarray(values, dtype=float), clip[0], clip[1])


def _screened(learner: Learner, threshold: float, min_retain: int | None) -> Learner:
    """Wrap a learner so covariates are screened inside each training fold.

    Screening must happen *inside* the fold: choosing covariates on the full sample
    and then cross-fitting leaks the held-out rows into the model-selection step.
    """
    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            ("screen", CorrelationScreener(threshold=threshold, min_retain=min_retain)),
            ("model", learner),
        ]
    )


def fit_nuisances(
    data: CausalData,
    *,
    outcome_learner: Learner,
    treatment_learner: Learner,
    missingness_learner: Learner | None,
    intermediate_learner: Learner | None,
    folds: Folds,
    scaler: OutcomeScaler,
    intermediate_value: float | None = None,
    screen_treatment: bool = False,
    screen_threshold: float = 0.1,
    min_retain: int | None = None,
    n_jobs: int = 1,
) -> NuisanceEstimates:
    """Fit every nuisance model this estimator needs.

    ``intermediate_value`` selects which controlled direct effect the outcome
    regression is evaluated at; it is required when the data carries an
    intermediate variable.
    """
    diagnostics: dict[str, Any] = {}
    groups = data.cluster

    # --- treatment mechanism -------------------------------------------------
    treatment_model = (
        _screened(treatment_learner, screen_threshold, min_retain)
        if screen_treatment
        else treatment_learner
    )
    propensity_out, propensity_diagnostics = cross_fit_predictions(
        treatment_model,
        data.covariates,
        data.treatment,
        data.weights,
        folds,
        task="classification",
        predict_designs={"g1": data.covariates},
        groups=groups,
        clip=(0.0, 1.0),
        n_jobs=n_jobs,
    )
    propensity = propensity_out["g1"]
    if propensity_diagnostics:
        diagnostics["propensity"] = propensity_diagnostics

    retained = data.covariate_names
    if screen_treatment:
        retained = _retained_covariates(data, screen_threshold, min_retain)

    # --- missingness mechanism ----------------------------------------------
    missingness = None
    if data.has_missing_outcome:
        if missingness_learner is None:
            raise ValueError(
                "the data has missing outcomes but no missingness_learner was supplied"
            )
        missing_out, missing_diagnostics = cross_fit_predictions(
            missingness_learner,
            data.missingness_design(),
            data.observed.astype(float),
            data.weights,
            folds,
            task="classification",
            predict_designs={
                "pi0": data.counterfactual_design(0.0),
                "pi1": data.counterfactual_design(1.0),
            },
            groups=groups,
            clip=(0.0, 1.0),
            n_jobs=n_jobs,
        )
        missingness = np.column_stack([missing_out["pi0"], missing_out["pi1"]])
        if missing_diagnostics:
            diagnostics["missingness"] = missing_diagnostics

    # --- intermediate mechanism ---------------------------------------------
    intermediate = None
    if data.has_intermediate:
        if intermediate_learner is None:
            raise ValueError(
                "the data has an intermediate variable but no intermediate_learner was supplied"
            )
        assert data.intermediate is not None
        # ``[A, W]``, and deliberately not ``missingness_design()`` even though the two
        # are the same array today.  ``q(Z | A, W)`` conditions on the propensity's set
        # extended by the treatment, which is the time ordering; it does not inherit the
        # missingness model's assumptions, and a longitudinal extension that added ``Z``
        # to the missingness design would silently make this ``P(Z | A, W, Z)``.
        intermediate_out, intermediate_diagnostics = cross_fit_predictions(
            intermediate_learner,
            data.treatment_design(),
            data.intermediate,
            data.weights,
            folds,
            task="classification",
            predict_designs={
                "pz0": data.counterfactual_design(0.0),
                "pz1": data.counterfactual_design(1.0),
            },
            groups=groups,
            clip=(0.0, 1.0),
            n_jobs=n_jobs,
        )
        intermediate = np.column_stack([intermediate_out["pz0"], intermediate_out["pz1"]])
        if intermediate_diagnostics:
            diagnostics["intermediate"] = intermediate_diagnostics

    # --- outcome regression --------------------------------------------------
    if data.has_intermediate and intermediate_value is None:
        raise ValueError("intermediate_value is required when the data carries an intermediate")
    include_z = data.has_intermediate
    outcome_design = data.treatment_design(include_intermediate=include_z)
    scaled = scaler.scale(data.outcome)
    outcome_task: Task = "classification" if data.family == "binomial" else "regression"

    outcome_out, outcome_diagnostics = cross_fit_predictions(
        outcome_learner,
        outcome_design,
        scaled,
        data.weights,
        folds,
        task=outcome_task,
        predict_designs={
            "observed": outcome_design,
            "at_one": data.counterfactual_design(1.0, intermediate_value=intermediate_value),
            "at_zero": data.counterfactual_design(0.0, intermediate_value=intermediate_value),
        },
        fit_mask=data.observed,
        groups=groups,
        clip=(0.0, 1.0),
        n_jobs=n_jobs,
    )
    if outcome_diagnostics:
        diagnostics["outcome"] = outcome_diagnostics

    return NuisanceEstimates(
        propensity=propensity,
        outcome=InitialFit(outcome_out["observed"], outcome_out["at_one"], outcome_out["at_zero"]),
        scaler=scaler,
        folds=folds,
        missingness=missingness,
        intermediate=intermediate,
        treatment_covariates=tuple(retained),
        diagnostics=diagnostics,
        outcome_task=outcome_task,
    )


def _retained_covariates(
    data: CausalData, threshold: float, min_retain: int | None
) -> tuple[str, ...]:
    """Which covariates a full-sample screen would keep, for reporting only.

    The fits themselves screen inside each fold; this is the whole-sample answer,
    reported so the user can see what the screen is doing.
    """
    from ..learners.screeners import screen_by_correlation

    keep = screen_by_correlation(
        data.covariates,
        data.treatment,
        threshold=threshold,
        min_retain=min_retain,
        sample_weight=data.weights,
    )
    return tuple(name for name, flag in zip(data.covariate_names, keep, strict=True) if flag)
