"""Typed estimation-method configuration for the causal-workflow API."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol, runtime_checkable

from ._typing import FluctuationKind, FoldStrata, GBounds, TargetingMethod, TargetingScheme
from .inference.bootstrap import Resampling
from .inference.multiplier import MultiplierKind

__all__ = [
    "DEFAULT_LONGITUDINAL_G_BOUNDS",
    "DEFAULT_LONGITUDINAL_MULTIPLIER",
    "DEFAULT_POINT_MULTIPLIER",
    "SHORTCUTS",
    "CollaborativeTMLEMethod",
    "CrossFitting",
    "DRTMLEMethod",
    "EstimationMethod",
    "Inference",
    "MethodAvailability",
    "ModelSpec",
    "Runtime",
    "TMLEMethod",
    "Targeting",
]


#: What ``"auto"`` resolves to, per engine.  These restate the engines' own signature defaults
#: rather than importing them, because importing ``cleverly.estimators`` here would invert this
#: module's place in the dependency order.  A restated default is a default that can drift, so
#: ``tests.unit.test_causal_study`` reads both signatures and fails when one of these stops
#: matching -- which is what makes the duplication safe rather than merely tidy.
DEFAULT_POINT_MULTIPLIER = 1000
DEFAULT_LONGITUDINAL_MULTIPLIER = 2000
DEFAULT_LONGITUDINAL_G_BOUNDS = (0.01, 1.0)


@dataclass(frozen=True)
class MethodAvailability:
    """Whether an estimation method can estimate an identified effect."""

    name: str
    available: bool
    reason: str | None = None


#: Marks a field that may legitimately hold an object persistence cannot describe -- a fitted
#: scikit-learn estimator, a ``SuperLearner``, a custom evaluation callable.  Such a field is
#: written as a non-reconstructible placeholder rather than refusing the whole file, which is
#: the rule ``docs/public-api-redesign.md`` states for custom configurations.  It is declared
#: per field rather than inferred, so that an unrepresentable object appearing anywhere *else*
#: still fails loudly instead of being quietly dropped.
OPAQUE: dict[str, str] = {"persist": "opaque"}


@dataclass(frozen=True)
class ModelSpec:
    """Nuisance-learning choices for analytic point-treatment TMLE."""

    outcome_learner: Any = field(default="default", metadata=OPAQUE)
    treatment_learner: Any = field(default="default", metadata=OPAQUE)
    missingness_learner: Any = field(default=None, metadata=OPAQUE)
    intermediate_learner: Any = field(default=None, metadata=OPAQUE)
    pseudo_learner: Any = field(default=None, metadata=OPAQUE)
    censoring_learner: Any = field(default=None, metadata=OPAQUE)
    density_bins: int = 20
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
    #: Multiplier-bootstrap draws behind the simultaneous bands.  ``"auto"`` is each engine's
    #: own default, which is not one number: the point path draws 1000 and the sequential path
    #: draws 2000.  A single literal here would silently pick one of them for the other engine
    #: -- and it did, halving every study-driven longitudinal fit's draws with nothing to read
    #: that said so.  Resolved in :meth:`TMLEMethod.estimator_kwargs`, exactly as ``g_bounds``
    #: is; the stored configuration keeps ``"auto"`` so it still says "the engine's default".
    n_multiplier: int | Literal["auto"] = "auto"
    multiplier_kind: MultiplierKind = "rademacher"


@dataclass(frozen=True)
class Runtime:
    """Reproducibility, provenance, and resource choices."""

    random_state: int | None = None
    run_id: str | None = None
    n_jobs: int = 1


@runtime_checkable
class EstimationMethod(Protocol):
    """A typed method that can build one of the package's evidenced engines."""

    @property
    def name(self) -> str: ...

    def with_overrides(self, **overrides: Any) -> EstimationMethod:
        """Return a normalized copy after applying convenience options."""

    def estimator_kwargs(self, *, longitudinal: bool = False) -> dict[str, Any]:
        """Translate the public configuration into an implementation-engine request."""


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
        "intermediate_learner": "intermediate_learner",
        "pseudo_learner": "pseudo_learner",
        "censoring_learner": "censoring_learner",
        "density_bins": "density_bins",
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
    name: str = "tmle"

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

    def estimator_kwargs(self, *, longitudinal: bool = False) -> dict[str, Any]:
        """Translate the normalized groups into the corresponding engine's API."""
        models = self.models
        cross = self.cross_fitting
        targeting = self.targeting
        inference = self.inference
        runtime = self.runtime
        common = {
            "outcome_learner": models.outcome_learner,
            "n_folds": cross.n_folds,
            "learner_folds": cross.learner_folds,
            "g_bounds": targeting.g_bounds,
            "q_bounds": targeting.q_bounds,
            "alpha": targeting.submodel_alpha,
            "max_iter": targeting.max_iter,
            "tol": targeting.tol,
            "alpha_sig": inference.alpha,
            "simultaneous": inference.simultaneous,
            "n_multiplier": inference.n_multiplier,
            "multiplier_kind": inference.multiplier_kind,
            "random_state": runtime.random_state,
            "run_id": runtime.run_id,
            "n_jobs": runtime.n_jobs,
        }
        if longitudinal:
            if cross.repeats != 1:
                raise ValueError("longitudinal TMLE does not yet support repeated cross-fitting")
            if not cross.enabled:
                common["n_folds"] = 1
            common.update(
                {
                    "pseudo_learner": models.pseudo_learner,
                    "treatment_learner": models.treatment_learner,
                    "censoring_learner": models.censoring_learner,
                }
            )
            # Longitudinal cumulative bounds have a fixed-pair contract. ``auto`` is a
            # point-treatment policy and has no sequential meaning.
            if common["g_bounds"] == "auto":
                common["g_bounds"] = DEFAULT_LONGITUDINAL_G_BOUNDS
            if common["n_multiplier"] == "auto":
                common["n_multiplier"] = DEFAULT_LONGITUDINAL_MULTIPLIER
            return common
        if common["n_multiplier"] == "auto":
            common["n_multiplier"] = DEFAULT_POINT_MULTIPLIER
        common.update(
            {
                "treatment_learner": models.treatment_learner,
                "missingness_learner": models.missingness_learner,
                "intermediate_learner": models.intermediate_learner,
                "density_bins": models.density_bins,
                "screen_treatment": models.screen_treatment,
                "screen_threshold": models.screen_threshold,
                "min_retain": models.min_retain,
                "cross_fit": cross.enabled,
                "repeats": cross.repeats,
                "stratify_folds": cross.stratify_by,
                "targeting_scheme": cross.targeting_scheme,
                "cv_evaluation": cross.fold_evaluation,
                "fluctuation": targeting.fluctuation,
                "targeting": targeting.algorithm,
                "nuisance_bound": targeting.nuisance_bound,
                "target_weights": targeting.target_weights,
                "step_size": targeting.step_size,
                "n_bootstrap": inference.n_bootstrap,
                "bootstrap_resampling": inference.bootstrap_resampling,
            }
        )
        return common


@dataclass(frozen=True)
class CollaborativeTMLEMethod(TMLEMethod):
    """Typed configuration for the existing collaborative TMLE strategy."""

    strategy: str = "greedy"
    preorder: str | None = None
    ordering: tuple[str, ...] | None = None
    candidates: tuple[tuple[str, ...], ...] | None = None
    selection_folds: int = 5
    selection_inner_folds: int = 2
    loss: str = "auto"
    penalty: bool = True
    selection_estimand: str = "ate"
    name: str = "collaborative_tmle"

    def estimator_kwargs(self, *, longitudinal: bool = False) -> dict[str, Any]:
        if longitudinal:
            raise ValueError("collaborative TMLE has no longitudinal derivation")
        return {
            **super().estimator_kwargs(),
            "strategy": self.strategy,
            "preorder": self.preorder,
            "ordering": self.ordering,
            "candidates": self.candidates,
            "selection_folds": self.selection_folds,
            "selection_inner_folds": self.selection_inner_folds,
            "loss": self.loss,
            "penalty": self.penalty,
            "ctmle_estimand": self.selection_estimand,
        }


@dataclass(frozen=True)
class DRTMLEMethod(TMLEMethod):
    """Typed configuration for the doubly-robust-inference TMLE variant."""

    guard: tuple[str, ...] = ("Q", "g")
    reduction: str = "univariate"
    reduced_outcome_learner: Any = field(default=None, metadata=OPAQUE)
    reduced_treatment_learner: Any = field(default=None, metadata=OPAQUE)
    reduced_crossfit: str = "pooled"
    update_order: str = "cleverly"
    evaluation: Any = field(default=None, metadata=OPAQUE)
    randomized: bool = False
    treatment_probabilities: Any = None
    name: str = "drtmle"

    def estimator_kwargs(self, *, longitudinal: bool = False) -> dict[str, Any]:
        if longitudinal:
            raise ValueError("DR-TMLE has no longitudinal derivation")
        return {
            **super().estimator_kwargs(),
            "guard": self.guard,
            "reduction": self.reduction,
            "reduced_outcome_learner": self.reduced_outcome_learner,
            "reduced_treatment_learner": self.reduced_treatment_learner,
            "reduced_crossfit": self.reduced_crossfit,
            "update_order": self.update_order,
            "evaluation": self.evaluation,
            "randomized": self.randomized,
        }
