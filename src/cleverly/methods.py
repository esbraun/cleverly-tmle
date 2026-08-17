"""Typed estimation-method configuration for the causal-workflow API."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from ._typing import FluctuationKind, FoldStrata, GBounds, TargetingMethod, TargetingScheme
from .inference.bootstrap import Resampling
from .inference.multiplier import MultiplierKind

__all__ = [
    "SHORTCUTS",
    "CrossFitting",
    "Inference",
    "MethodAvailability",
    "ModelSpec",
    "Runtime",
    "TMLEMethod",
    "Targeting",
]


@dataclass(frozen=True)
class MethodAvailability:
    """Whether an estimation method can estimate an identified effect."""

    name: str
    available: bool
    reason: str | None = None


@dataclass(frozen=True)
class ModelSpec:
    """Nuisance-learning choices for analytic point-treatment TMLE."""

    outcome_learner: Any = "default"
    treatment_learner: Any = "default"
    missingness_learner: Any = None
    screen_treatment: bool = False
    screen_threshold: float = 0.1
    min_retain: int | None = None


@dataclass(frozen=True)
class CrossFitting:
    """Outer and learner-level sample-splitting choices."""

    enabled: bool = True
    n_folds: int = 10
    learner_folds: int = 5
    repeats: int = 1
    stratify_by: FoldStrata = "treatment"
    targeting_scheme: TargetingScheme = "pooled"
    fold_evaluation: bool = False


@dataclass(frozen=True)
class Targeting:
    """Fluctuation, bounds, and solver choices."""

    fluctuation: FluctuationKind = "logistic"
    algorithm: TargetingMethod = "iterative"
    g_bounds: GBounds = "auto"
    q_bounds: tuple[float, float] | None = None
    nuisance_bound: float = 0.01
    submodel_alpha: float = 0.9995
    target_weights: bool = False
    step_size: float = 1e-3
    max_iter: int = 20
    tol: float = 1e-10


@dataclass(frozen=True)
class Inference:
    """Confidence-interval and resampling choices."""

    #: Significance level of the reported intervals, and what the ``alpha=`` shortcut sets.
    #: Not to be confused with :attr:`Targeting.submodel_alpha`, which bounds the logistic
    #: submodel and is reached as ``submodel_alpha=``.
    alpha: float = 0.05
    n_bootstrap: int = 0
    bootstrap_resampling: Resampling = "auto"
    simultaneous: bool = True
    n_multiplier: int = 1000
    multiplier_kind: MultiplierKind = "rademacher"


@dataclass(frozen=True)
class Runtime:
    """Reproducibility, provenance, and resource choices."""

    random_state: int | None = None
    run_id: str | None = None
    n_jobs: int = 1


#: Flat keyword shortcut to ``(configuration group, field)``, read by
#: :meth:`TMLEMethod.with_overrides`.  Module level so the invariant that keeps it honest is
#: testable: a shortcut spelled the same as one of the groups' fields must set *that* field.
#: ``tests.unit.test_causal_study`` checks it, after ``alpha=`` was found routing to
#: :attr:`Targeting.submodel_alpha` while :attr:`Inference.alpha` was the interval level.
SHORTCUTS: dict[str, dict[str, str]] = {
    "models": {
        "outcome_learner": "outcome_learner",
        "treatment_learner": "treatment_learner",
        "missingness_learner": "missingness_learner",
        "screen_treatment": "screen_treatment",
        "screen_threshold": "screen_threshold",
        "min_retain": "min_retain",
    },
    "cross_fitting": {
        "cross_fit": "enabled",
        "n_folds": "n_folds",
        "learner_folds": "learner_folds",
        "repeats": "repeats",
        "stratify_folds": "stratify_by",
        "targeting_scheme": "targeting_scheme",
        "cv_evaluation": "fold_evaluation",
    },
    "targeting": {
        "fluctuation": "fluctuation",
        "targeting": "algorithm",
        "g_bounds": "g_bounds",
        "q_bounds": "q_bounds",
        "nuisance_bound": "nuisance_bound",
        "submodel_alpha": "submodel_alpha",
        "target_weights": "target_weights",
        "step_size": "step_size",
        "max_iter": "max_iter",
        "tol": "tol",
    },
    "inference": {
        "alpha": "alpha",
        "n_bootstrap": "n_bootstrap",
        "bootstrap_resampling": "bootstrap_resampling",
        "simultaneous": "simultaneous",
        "n_multiplier": "n_multiplier",
        "multiplier_kind": "multiplier_kind",
    },
    "runtime": {
        "random_state": "random_state",
        "run_id": "run_id",
        "n_jobs": "n_jobs",
    },
}


@dataclass(frozen=True)
class TMLEMethod:
    """Normalized configuration for the built-in analytic TMLE method.

    ``IdentifiedEffect.estimate(method="tmle", ...)`` normalizes keyword shortcuts into
    this object before fitting. Passing one directly gives the same computational path.
    """

    models: ModelSpec = ModelSpec()
    cross_fitting: CrossFitting = CrossFitting()
    targeting: Targeting = Targeting()
    inference: Inference = Inference()
    runtime: Runtime = Runtime()

    def with_overrides(self, **overrides: Any) -> TMLEMethod:
        """Return a copy with supported convenience keywords normalized by concern.

        **A shortcut that is also a field name must set that field.**  ``alpha=`` used to
        route to :attr:`Targeting.submodel_alpha` -- the logistic-submodel bound, following
        the legacy ``TMLE(alpha=..., alpha_sig=...)`` spelling -- while the field actually
        *named* ``alpha`` was :attr:`Inference.alpha`, the significance level.  So
        ``estimate(alpha=0.10)`` was accepted, moved the shrink bound, and left the interval
        at 95%: a silently wrong knob rather than an error.  ``alpha=`` is now the
        significance level, the bound is reached as ``submodel_alpha=``, and the legacy
        ``alpha_sig=`` spelling is gone.  ``tests.unit.test_causal_study`` pins the rule.
        """
        groups = SHORTCUTS
        known = {name for fields in groups.values() for name in fields}
        unknown = sorted(set(overrides) - known)
        if unknown:
            raise TypeError(
                f"unsupported tmle option(s): {unknown}. Study-design roles belong on "
                "PointTreatment; advanced method options require a typed method object."
            )
        method = self
        for group_name, fields in groups.items():
            changes = {
                attribute: overrides[shortcut]
                for shortcut, attribute in fields.items()
                if shortcut in overrides
            }
            if changes:
                method = replace(
                    method,
                    **{group_name: replace(getattr(method, group_name), **changes)},
                )
        return method

    def estimator_kwargs(self) -> dict[str, Any]:
        """Translate the normalized groups into the existing TMLE engine's API."""
        models = self.models
        cross = self.cross_fitting
        targeting = self.targeting
        inference = self.inference
        runtime = self.runtime
        return {
            "outcome_learner": models.outcome_learner,
            "treatment_learner": models.treatment_learner,
            "missingness_learner": models.missingness_learner,
            "screen_treatment": models.screen_treatment,
            "screen_threshold": models.screen_threshold,
            "min_retain": models.min_retain,
            "cross_fit": cross.enabled,
            "n_folds": cross.n_folds,
            "learner_folds": cross.learner_folds,
            "repeats": cross.repeats,
            "stratify_folds": cross.stratify_by,
            "targeting_scheme": cross.targeting_scheme,
            "cv_evaluation": cross.fold_evaluation,
            "fluctuation": targeting.fluctuation,
            "targeting": targeting.algorithm,
            "g_bounds": targeting.g_bounds,
            "q_bounds": targeting.q_bounds,
            "nuisance_bound": targeting.nuisance_bound,
            "alpha": targeting.submodel_alpha,
            "target_weights": targeting.target_weights,
            "step_size": targeting.step_size,
            "max_iter": targeting.max_iter,
            "tol": targeting.tol,
            "alpha_sig": inference.alpha,
            "n_bootstrap": inference.n_bootstrap,
            "bootstrap_resampling": inference.bootstrap_resampling,
            "simultaneous": inference.simultaneous,
            "n_multiplier": inference.n_multiplier,
            "multiplier_kind": inference.multiplier_kind,
            "random_state": runtime.random_state,
            "run_id": runtime.run_id,
            "n_jobs": runtime.n_jobs,
        }
