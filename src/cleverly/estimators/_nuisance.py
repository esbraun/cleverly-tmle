"""Cross-fitted estimation of the nuisance parameters.

TMLE needs three or four regressions before any targeting happens:

``g(a | W) = P(A = a | W)``
    the treatment mechanism, one column per arm, which enters the clever covariate;
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

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .._typing import BoolArray, FloatArray, IntArray, Learner
from ..data.causal_data import CausalData
from ..fluctuation.iterative import InitialFit
from ..interventions import RegimeSet, Shift, ShiftSet
from ..learners._fitting import (
    Task,
    as_target,
    fit_learner,
    predict_mean,
    predict_probabilities,
)
from ..learners.crossfit import Folds
from ..learners.density import ConditionalDensity, fit_conditional_density
from ..learners.screeners import CorrelationScreener
from ..learners.super_learner import SuperLearnerDiagnostics
from ..msm import MSMSet
from ..utils.bounds import OutcomeScaler, bound
from ..utils.parallel import map_parallel
from .direct_effect import check_level

__all__ = ["NuisanceEstimates", "Propensity", "cross_fit_predictions", "fit_nuisances"]


@dataclass(frozen=True)
class Propensity:
    r"""The treatment mechanism :math:`g(a \mid W)`, one column per arm.

    ``values`` is ``(n, K)`` with column ``j`` holding :math:`P(A = \text{arms}[j] \mid W)`
    out of fold and **untruncated**; truncation happens at targeting time via
    :meth:`bounded`, because the ATT tolerates far less extrapolation than the ATE and
    so uses a tighter bound, and because a sensitivity sweep must be able to re-truncate
    without refitting.

    A matrix rather than the single :math:`g_1(W)` vector this used to be, even for two
    arms -- where column 0 is exactly ``1 - g1`` and the arithmetic is unchanged.  With
    more than two arms there is no margin to be the propensity: the mechanism is a
    distribution over the arms, and every arm needs its own denominator.
    """

    values: FloatArray
    arms: tuple[float, ...]

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.arms):
            raise ValueError(
                f"propensity must be (n, {len(self.arms)}) for arms {list(self.arms)}; "
                f"got shape {values.shape}"
            )

    @property
    def n(self) -> int:
        return int(np.asarray(self.values).shape[0])

    @property
    def n_arms(self) -> int:
        return len(self.arms)

    def column_for(self, arm: float) -> int:
        """Index of the column holding ``P(A = arm | W)``."""
        match = [j for j, level in enumerate(self.arms) if level == float(arm)]
        if not match:
            raise KeyError(f"arm {float(arm)!r} is not one of {list(self.arms)}")
        return match[0]

    def arm(self, arm: float) -> FloatArray:
        """``P(A = arm | W)``, untruncated."""
        return np.asarray(self.values, dtype=float)[:, self.column_for(arm)]

    def bounded(self, bounds: tuple[float, float]) -> FloatArray:
        r"""The ``(n, K)`` mechanism truncated into ``bounds``.

        **Two arms keep the complement form.**  ``g1`` is clipped and arm 0 is taken as
        ``1 - g1``, which is exactly what the estimator has always done -- and it is not
        the same as clipping both columns when ``bounds`` is asymmetric, so this is what
        keeps every binary regression fixture valid.

        **More than two arms are clipped column by column, and are not renormalised.**
        Flooring a row of a multinomial breaks :math:`\sum_a g_a = 1`, and the obvious
        repair -- rescale the row -- would undo the only thing truncation is for, since
        rescaling can push a column back below the floor.  Nothing downstream needs the
        simplex: arm ``a``'s clever covariate reads only :math:`\tilde g_a`, and the
        plug-in is an average of targeted predictions containing no mechanism at all, so
        no bound can move :math:`\Psi`.  What a binding bound moves is the second-order
        remainder :math:`R_2`, exactly as :mod:`cleverly.fluctuation.submodel` sets out.
        :meth:`~cleverly.sensitivity.PositivityReport` reports how far the truncated rows
        depart from summing to one.
        """
        lower, upper = float(bounds[0]), float(bounds[1])
        values = np.asarray(self.values, dtype=float)
        if self.n_arms == 2:
            one = bound(values[:, self.column_for(1.0)], lower, upper)
            columns = {self.column_for(1.0): one, self.column_for(0.0): 1.0 - one}
            return np.column_stack([columns[j] for j in range(2)])
        return bound(values, lower, upper)


@dataclass(frozen=True)
class NuisanceEstimates:
    """Cross-fitted nuisance predictions and the diagnostics that came with them.

    Attributes
    ----------
    propensity:
        Out-of-fold ``g(a | W)`` for every arm, *not* truncated -- see
        :class:`Propensity`.  On a **continuous** treatment there are no arms and no
        propensity: the mechanism is ``density`` instead, and this holds an ``(n, 0)``
        placeholder so that ``n`` and ``arms`` keep answering.  Every reader that would
        misread that placeholder as a real mechanism refuses a continuous fit by name --
        :func:`~cleverly.sensitivity.positivity_report` is the one that would otherwise
        report a spurious simplex deviation.
    outcome:
        Initial outcome regression on the ``[0, 1]`` scale, at the observed
        treatment and at every counterfactual arm.
    missingness, intermediate:
        ``(n, K)`` arrays indexed by treatment arm, or ``None`` when not applicable.
    scaler:
        The transformation used to put the outcome on ``[0, 1]``.
    diagnostics:
        Per-nuisance Super Learner weights and cross-validated risks, when a
        Super Learner was used.
    """

    propensity: Propensity
    outcome: InitialFit
    scaler: OutcomeScaler
    folds: Folds
    missingness: FloatArray | None = None
    intermediate: FloatArray | None = None
    treatment_covariates: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)
    outcome_task: Task = "regression"
    #: Outcome regression evaluated at every requested level of the intermediate
    #: variable, when more than one was asked for.  ``outcome`` is the entry for the
    #: level this object currently targets; :meth:`at_level` swaps it.
    #:
    #: Populated because the controlled direct effect at ``z = 0`` and at ``z = 1``
    #: share *every* nuisance model -- the only difference is which counterfactual
    #: design the outcome regression is predicted onto -- so fitting them separately
    #: refits all four models to get two extra prediction vectors.
    outcome_by_level: dict[float, InitialFit] = field(default_factory=dict)
    #: The regimes this fit targets, evaluated at every row, or ``None`` for an
    #: arm-indexed fit.  Carried with the nuisances rather than recomputed because
    #: everything that reuses a fit reuses them: ``retarget``, and so the truncation
    #: curve, the MNAR tilt and the omitted-variable bound, all take a
    #: :class:`NuisanceEstimates` and must target the same regimes the fit declared --
    #: including on a result read back from disk, where the rules that built the
    #: densities are no longer callable.
    regimes: RegimeSet | None = None
    #: The conditional treatment density ``g(a | W)`` on a continuous fit, evaluated at
    #: every row and every bin, or ``None`` when the treatment has arms.  This is the
    #: mechanism half of double robustness there, standing where ``propensity`` stands
    #: for a discrete treatment.
    density: ConditionalDensity | None = None
    #: The shifts this fit targets, evaluated against ``density``, or ``None`` for a fit
    #: that declared none.  Carried for the reason ``regimes`` is, and built *inside*
    #: :func:`fit_nuisances` rather than beside it -- unlike a regime, a shift's clever
    #: covariate is a ratio of densities, so it cannot be evaluated until the density
    #: exists, and evaluating it here is what makes "g(A | W) and g(A - delta | W) come
    #: from the same out-of-fold model" structural rather than an invariant to maintain.
    shifts: ShiftSet | None = None
    #: The working model this fit projects the counterfactual means onto, evaluated at
    #: every row and every arm, or ``None`` for a fit that declared none.  Carried for the
    #: reason ``regimes`` is, and built *beside* :func:`fit_nuisances` rather than inside
    #: it -- unlike a shift, a working model's design is a function of the covariates
    #: alone and needs no mechanism to evaluate.
    msm: MSMSet | None = None

    @property
    def n(self) -> int:
        return self.propensity.n

    @property
    def arms(self) -> tuple[float, ...]:
        """The arm codes every per-arm array here is keyed by."""
        return self.propensity.arms

    def at_level(self, value: float) -> NuisanceEstimates:
        """The same nuisances, with the outcome regression evaluated at ``Z = value``."""
        fit = self.outcome_by_level.get(check_level(value))
        if fit is None:
            raise KeyError(
                f"no outcome regression was computed at intermediate level {value!r}; "
                f"have {sorted(self.outcome_by_level)}"
            )
        return replace(self, outcome=fit)

    def bounded_propensity(self, bounds: tuple[float, float]) -> FloatArray:
        """``g(a | W)`` truncated into ``bounds``, ``(n, K)`` -- see :meth:`Propensity.bounded`."""
        return self.propensity.bounded(bounds)

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
    classes: Sequence[float] | None = None,
    n_jobs: int = 1,
) -> tuple[dict[str, FloatArray], list[SuperLearnerDiagnostics]]:
    """Out-of-fold predictions of one nuisance regression.

    Parameters
    ----------
    design, target, weights:
        Training data for the regression.
    predict_designs:
        Named design matrices to predict on -- for the outcome regression these are
        the observed treatment and every counterfactual arm, so a single pass over the
        folds produces everything the fluctuation needs.
    fit_mask:
        Rows eligible for *training*.  The outcome regression is fit only where the
        outcome is observed, but must still predict everywhere.
    groups:
        Cluster codes, forwarded to any learner that cross-validates internally so its
        inner folds keep clusters intact too -- see :func:`_fit_with_groups`.
    classes:
        Set for a nuisance that is a conditional *distribution* over these classes
        rather than a single conditional mean -- the treatment mechanism of a ``K``-armed
        treatment.  Each named prediction then comes back ``(n, K)`` instead of ``(n,)``,
        with columns in ``classes`` order.

    Returns
    -------
    Predictions per named design, and the Super Learner diagnostics per fold (empty
    when the learner is not a Super Learner).
    """
    n = design.shape[0]
    mask = np.ones(n, dtype=bool) if fit_mask is None else np.asarray(fit_mask, dtype=bool)
    if not mask.any():
        raise ValueError("no rows are eligible for fitting this nuisance model")

    def predict(model: Learner, matrix: FloatArray) -> FloatArray:
        if classes is None:
            return _clip(predict_mean(model, matrix, task), clip)
        return _clip(predict_probabilities(model, matrix, classes), clip)

    if folds.is_single:
        rows = np.flatnonzero(mask)
        model = _fit_with_groups(learner, design, target, weights, rows, task, groups)
        predictions = {name: predict(model, matrix) for name, matrix in predict_designs.items()}
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
            name: predict(model, matrix[test]) for name, matrix in predict_designs.items()
        }
        return test, predictions, getattr(model, "diagnostics_", None)

    results = map_parallel(run_fold, jobs, n_jobs=n_jobs)
    shape: tuple[int, ...] = (n,) if classes is None else (n, len(tuple(classes)))
    out = {name: np.empty(shape, dtype=float) for name in predict_designs}
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
    """Fit a learner on ``rows``, passing cluster codes on to its inner folds.

    ``rows`` is a subset of one outer training fold, so a learner that cross-validates
    internally -- a :class:`~cleverly.learners.SuperLearner` scoring its candidates --
    only ever splits rows this fold was already allowed to train on.  What it needs told
    is the *cluster* structure, since an inner fold that splits a cluster scores a
    candidate on rows correlated with its own training set.

    :func:`~cleverly.learners._fitting.fit_learner` does the routing, which matters
    because ``screen_treatment=True`` wraps the learner in a pipeline: this used to test
    ``isinstance(learner, SuperLearner)`` and so dropped the cluster codes for exactly
    the configuration that asked for both.  The codes are subset to ``rows`` first, to
    stay aligned with the design handed alongside them.
    """
    return fit_learner(
        learner,
        design[rows],
        as_target(target[rows], task),
        weights[rows],
        groups=None if groups is None else np.asarray(groups)[rows],
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


def _arm_key(arm: float) -> str:
    """Name of the counterfactual design for one arm, where no intermediate is involved."""
    return f"arm@{arm}"


def _design_key(arm: float, level: float | None) -> str:
    """Name of the counterfactual design for one arm at one intermediate level.

    A single flat namespace, because ``cross_fit_predictions`` takes one dict of designs
    and returns one dict of predictions; the key has to carry both coordinates.
    """
    return f"arm@{arm}|z@{level}"


def _requested_levels(
    intermediate_value: float | None, extra_levels: Sequence[float]
) -> tuple[float, ...]:
    """The levels to evaluate the outcome regression at, primary one first."""
    if intermediate_value is None:
        return (None,)  # type: ignore[return-value]
    primary = check_level(intermediate_value)
    ordered = [primary]
    for level in extra_levels:
        checked = check_level(level)
        if checked not in ordered:
            ordered.append(checked)
    return tuple(ordered)


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
    extra_levels: Sequence[float] = (),
    screen_treatment: bool = False,
    screen_threshold: float = 0.1,
    min_retain: int | None = None,
    shifts: Sequence[Shift] = (),
    shift_reference: str | None = None,
    density_bins: int = 20,
    n_jobs: int = 1,
) -> NuisanceEstimates:
    """Fit every nuisance model this estimator needs.

    ``intermediate_value`` selects which controlled direct effect the outcome
    regression is evaluated at; it is required when the data carries an
    intermediate variable.  ``extra_levels`` asks for the outcome regression to be
    evaluated at further levels in the *same* pass over the folds, which is what lets
    both controlled direct effects be estimated from one set of nuisance fits.

    ``shifts`` declares modified treatment policies, which a continuous treatment
    requires and an arm-coded one refuses.  They are evaluated *here*, against the
    density fitted a few lines above, rather than by the caller: the clever covariate is
    the ratio :math:`g(a - \\delta \\mid W) / g(a \\mid W)`, so numerator and denominator
    come from one out-of-fold model by construction and there is no second model for a
    later step to get wrong.
    """
    diagnostics: dict[str, Any] = {}
    groups = data.cluster

    # --- treatment mechanism -------------------------------------------------
    treatment_model = (
        _screened(treatment_learner, screen_threshold, min_retain)
        if screen_treatment
        else treatment_learner
    )
    arms = data.arm_codes
    density: ConditionalDensity | None = None
    shift_set: ShiftSet | None = None
    if data.is_continuous_treatment:
        # A dose has no arms, so there is no P(A = a | W) to classify: the mechanism is a
        # density. The learner is the same one either way -- fit_conditional_density
        # factorises the density into bin hazards, each a conditional probability of a
        # binary event, so the classifier in treatment_learner= estimates all of them at
        # once. The propensity below is an (n, 0) placeholder; see NuisanceEstimates.
        density, density_diagnostics = fit_conditional_density(
            treatment_model,
            data.covariates,
            data.treatment,
            data.weights,
            folds,
            n_bins=density_bins,
            groups=groups,
            n_jobs=n_jobs,
        )
        diagnostics["density"] = density_diagnostics
        propensity = Propensity(np.zeros((data.n, 0)), ())
        shift_set = ShiftSet.evaluate(tuple(shifts), data, density, reference=shift_reference)
    else:
        propensity_out, propensity_diagnostics = cross_fit_predictions(
            treatment_model,
            data.covariates,
            data.treatment,
            data.weights,
            folds,
            task="classification",
            predict_designs={"g": data.covariates},
            groups=groups,
            clip=(0.0, 1.0),
            classes=arms,
            n_jobs=n_jobs,
        )
        propensity = Propensity(propensity_out["g"], arms)
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
            predict_designs={_arm_key(arm): data.counterfactual_design(arm) for arm in arms},
            groups=groups,
            clip=(0.0, 1.0),
            n_jobs=n_jobs,
        )
        missingness = np.column_stack([missing_out[_arm_key(arm)] for arm in arms])
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
            predict_designs={_arm_key(arm): data.counterfactual_design(arm) for arm in arms},
            groups=groups,
            clip=(0.0, 1.0),
            n_jobs=n_jobs,
        )
        intermediate = np.column_stack([intermediate_out[_arm_key(arm)] for arm in arms])
        if intermediate_diagnostics:
            diagnostics["intermediate"] = intermediate_diagnostics

    # --- outcome regression --------------------------------------------------
    if data.has_intermediate and intermediate_value is None:
        raise ValueError("intermediate_value is required when the data carries an intermediate")
    include_z = data.has_intermediate
    outcome_design = data.treatment_design(include_intermediate=include_z)
    scaled = scaler.scale(data.outcome)
    outcome_task: Task = "classification" if data.family == "binomial" else "regression"

    # Every requested level of the intermediate in one pass over the folds. The
    # outcome *model* does not depend on the level -- its design uses the observed Z --
    # so the levels differ only in which counterfactual design it is predicted onto,
    # which is exactly what predict_designs is for. Fitting them separately refits all
    # four nuisance models to obtain two extra prediction vectors.
    levels = _requested_levels(intermediate_value, extra_levels)

    # What the counterfactual predictions are keyed by, and what treatment value each
    # one sets. An arm fit sets one level for everybody; a shift fit sets a *different*
    # dose per row, which is what makes a modified treatment policy modified -- so the
    # value here is an (n,) array rather than a scalar, and the keys are shift codes.
    if shift_set is None:
        counterfactual: dict[float, float | FloatArray] = {arm: arm for arm in arms}
    else:
        counterfactual = {
            code: shift_set.shifted[:, index] for index, code in enumerate(shift_set.codes)
        }

    designs: dict[str, FloatArray] = {"observed": outcome_design}
    for level in levels:
        for code, value in counterfactual.items():
            designs[_design_key(code, level)] = data.counterfactual_design(
                value, intermediate_value=level
            )

    outcome_out, outcome_diagnostics = cross_fit_predictions(
        outcome_learner,
        outcome_design,
        scaled,
        data.weights,
        folds,
        task=outcome_task,
        predict_designs=designs,
        fit_mask=data.observed,
        groups=groups,
        clip=(0.0, 1.0),
        n_jobs=n_jobs,
    )
    if outcome_diagnostics:
        diagnostics["outcome"] = outcome_diagnostics

    by_level = {
        level: InitialFit(
            outcome_out["observed"],
            {code: outcome_out[_design_key(code, level)] for code in counterfactual},
        )
        for level in levels
    }
    primary = by_level[levels[0]]

    return NuisanceEstimates(
        propensity=propensity,
        outcome=primary,
        outcome_by_level=by_level if data.has_intermediate else {},
        scaler=scaler,
        folds=folds,
        missingness=missingness,
        intermediate=intermediate,
        treatment_covariates=tuple(retained),
        diagnostics=diagnostics,
        outcome_task=outcome_task,
        density=density,
        shifts=shift_set,
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
