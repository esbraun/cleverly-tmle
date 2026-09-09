"""Typed estimation-method configuration for the causal-workflow API."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol, runtime_checkable

from ._typing import FluctuationKind, FoldStrata, GBounds, TargetingMethod, TargetingScheme
from .exceptions import MethodConfigurationError
from .inference.bootstrap import Resampling
from .inference.multiplier import MultiplierKind
from .learners.library import _validate_learner

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
    """Report whether a method can estimate an identified effect.

    Parameters
    ----------
    name : str
        Method name accepted by ``estimate(method=...)``.
    available : bool
        Whether the method supports the identified functional.
    reason : str or None
        Refusal reason when ``available`` is false.
    """

    name: str
    available: bool
    reason: str | None = None


@dataclass(frozen=True)
class ModelSpec:
    """Configure nuisance learning for TMLE methods.

    Parameters
    ----------
    outcome_learner : estimator or None
        Outcome-regression learner. ``None`` uses the default library.
    treatment_learner : estimator or None
        Treatment-mechanism learner.
    missingness_learner : estimator or None
        Outcome-observation learner.
    intermediate_learner : estimator or None
        Intermediate-variable learner for controlled direct effects.
    pseudo_learner : estimator or None
        Sequential pseudo-outcome learner.
    censoring_learner : estimator or None
        Longitudinal observation-mechanism learner.
    density_bins : int
        Number of bins for a continuous-treatment conditional density.
    screen_treatment : bool
        Whether to screen treatment-mechanism covariates by correlation.
    screen_threshold : float
        Absolute correlation threshold used by screening.
    min_retain : int or None
        Minimum number of covariates retained by screening.

    See Also
    --------
    TMLEMethod : Method configuration this object is the model half of.
    cleverly.SuperLearner : Ensemble to pass when one learner is not enough.
    CrossFitting : How the learners set here are fitted out of fold.

    Examples
    --------
    >>> from cleverly import ModelSpec
    >>> from sklearn.linear_model import LinearRegression
    >>> models = ModelSpec(outcome_learner=LinearRegression())
    >>> type(models.outcome_learner).__name__
    'LinearRegression'
    """

    outcome_learner: Any = None
    treatment_learner: Any = None
    missingness_learner: Any = None
    intermediate_learner: Any = None
    pseudo_learner: Any = None
    censoring_learner: Any = None
    density_bins: int = 20
    screen_treatment: bool = False
    screen_threshold: float = 0.1
    min_retain: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "outcome_learner",
            "treatment_learner",
            "missingness_learner",
            "intermediate_learner",
            "pseudo_learner",
            "censoring_learner",
        ):
            _validate_learner(getattr(self, name), name)


@dataclass(frozen=True)
class CrossFitting:
    """Configure outer and learner-level sample splitting.

    Parameters
    ----------
    enabled : bool, default=True
        Whether to estimate nuisances out of fold.
    n_folds : int, default=10
        Number of outer folds.
    learner_folds : int, default=5
        Inner folds used by ensemble learners.
    repeats : int, default=1
        Independent outer-fold assignments to combine by their median.
    stratify_by : {"treatment", "treatment+outcome"}, default="treatment"
        Fold-stratification policy.
    targeting_scheme : {"pooled", "fold"}, default="pooled"
        Pooled or fold-specific targeting scheme.
    fold_evaluation : bool, default=False
        Whether to retain fold-evaluated CV-TMLE estimates.

    See Also
    --------
    TMLEMethod : Method configuration this object is the splitting half of.
    Targeting : Whether the fluctuation is pooled or fitted per fold.
    cleverly.learners.make_folds : The partition this configuration asks for.

    Examples
    --------
    Fewer folds, combined over three independent splits:

    >>> from cleverly import CrossFitting
    >>> folds = CrossFitting(n_folds=5, repeats=3)
    >>> folds.n_folds, folds.repeats
    (5, 3)

    Turn cross-fitting off to fit every nuisance on all the rows:

    >>> CrossFitting(enabled=False).enabled
    False
    """

    enabled: bool = True
    n_folds: int = 10
    learner_folds: int = 5
    repeats: int = 1
    stratify_by: FoldStrata = "treatment"
    targeting_scheme: TargetingScheme = "pooled"
    fold_evaluation: bool = False


@dataclass(frozen=True)
class Targeting:
    """Configure fluctuation, nuisance bounds, and solver behavior.

    Parameters
    ----------
    fluctuation : {"logistic", "linear"}, default="logistic"
        Fluctuation submodel family.
    algorithm : {"iterative", "one_step"}, default="iterative"
        Targeting algorithm.
    g_bounds : {"auto"}, float, or tuple of float, default="auto"
        Treatment or cumulative-mechanism bounds.
    q_bounds : tuple of float or None, default=None
        Optional outcome-regression bounds.
    nuisance_bound : float, default=0.01
        Lower bound used by auxiliary nuisance regressions.
    submodel_alpha : float, default=0.9995
        Logistic-submodel shrink bound.
    target_weights : bool, default=False
        Whether probability weights enter the targeting score.
    step_size : float, default=1e-3
        Step size for compatible targeting.
    max_iter : int, default=20
        Maximum targeting iterations.
    tol : float, default=1e-10
        Score-equation convergence tolerance.

    See Also
    --------
    TMLEMethod : Method configuration this object is the targeting half of.
    Inference : Interval settings, which own ``alpha`` rather than ``submodel_alpha``.
    cleverly.sensitivity.truncation_curve : What the estimate does across ``g_bounds``.

    Examples
    --------
    Truncate the treatment mechanism at a fixed bound rather than the data-driven one:

    >>> from cleverly import Targeting
    >>> Targeting(g_bounds=0.025).g_bounds
    0.025

    >>> Targeting().g_bounds
    'auto'
    """

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
    """Configure confidence intervals and resampling.

    Parameters
    ----------
    alpha : float, default=0.05
        Significance level for reported intervals.
    n_bootstrap : int, default=0
        Bootstrap fits. Zero disables refit bootstrap inference.
    bootstrap_resampling : {"auto", "iid", "cluster"}, default="auto"
        Unit or cluster resampling policy.
    simultaneous : bool, default=True
        Whether multi-parameter results include simultaneous bands.
    n_multiplier : int or {"auto"}, default="auto"
        Multiplier draws for simultaneous bands.
    multiplier_kind : {"rademacher", "mammen", "normal"}, default="rademacher"
        Distribution of multiplier weights.

    See Also
    --------
    TMLEMethod : Method configuration this object is the inference half of.
    Targeting : Owns ``submodel_alpha``, which is not a significance level.
    cleverly.inference.run_bootstrap : The refit bootstrap ``n_bootstrap`` requests.

    Examples
    --------
    Report 99 percent intervals and add a cluster bootstrap:

    >>> from cleverly import Inference
    >>> inference = Inference(alpha=0.01, n_bootstrap=200, bootstrap_resampling="cluster")
    >>> inference.alpha, inference.n_bootstrap, inference.bootstrap_resampling
    (0.01, 200, 'cluster')

    ``alpha`` is the significance level. The confidence level is ``1 - alpha``. The
    logistic-submodel bound lives on
    :class:`Targeting` and is spelled differently for that reason:

    >>> from cleverly import Targeting
    >>> Targeting().submodel_alpha
    0.9995
    """

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
    """Configure reproducibility, provenance, and resources.

    Parameters
    ----------
    random_state : int or None, default=None
        Seed for folds, learners, and resampling.
    run_id : str or None, default=None
        User-defined identifier stored in provenance.
    n_jobs : int, default=1
        Maximum parallel jobs requested by the fit.

    See Also
    --------
    TMLEMethod : Method configuration this object is the runtime half of.
    cleverly.Provenance : Where ``run_id`` and the fit fingerprints are recorded.
    cleverly.learners.set_thread_limit : Native threads per fit, which ``n_jobs`` does not set.

    Examples
    --------
    >>> from cleverly import Runtime
    >>> runtime = Runtime(random_state=0, run_id="pilot-2026-03", n_jobs=4)
    >>> runtime.random_state, runtime.run_id, runtime.n_jobs
    (0, 'pilot-2026-03', 4)
    """

    random_state: int | None = None
    run_id: str | None = None
    n_jobs: int = 1


@runtime_checkable
class EstimationMethod(Protocol):
    """Define the contract for an evidenced estimation method.

    Parameters
    ----------
    *args, **kwargs
        Present because :func:`typing.runtime_checkable` gives a protocol a synthetic
        constructor. A protocol is implemented, not instantiated.

    Attributes
    ----------
    name : str

    See Also
    --------
    TMLEMethod : The analytic point-treatment implementation.
    CollaborativeTMLEMethod : Adds collaborative treatment-mechanism selection.
    DRTMLEMethod : Adds the reduced-dimension doubly robust correction.
    cleverly.IdentifiedEffect : What a method is handed to in order to fit.

    Examples
    --------
    Every shipped method satisfies the protocol, so application code can accept any of
    them and read the same members:

    >>> from cleverly import DRTMLEMethod, EstimationMethod, TMLEMethod
    >>> methods = [TMLEMethod(), DRTMLEMethod()]
    >>> [method.name for method in methods]
    ['tmle', 'drtmle']
    >>> all(isinstance(method, EstimationMethod) for method in methods)
    True
    """

    @property
    def name(self) -> str:
        """Return the stable method name."""
        ...

    def with_overrides(self, **overrides: Any) -> EstimationMethod:
        """Return a normalized copy after applying convenience options.

        Parameters
        ----------
        **overrides : Any
            Supported flat configuration shortcuts.

        Returns
        -------
        EstimationMethod
            New immutable method configuration.
        """

    def estimator_kwargs(self, *, longitudinal: bool = False) -> dict[str, Any]:
        """Translate public configuration into an engine request.

        Parameters
        ----------
        longitudinal : bool
            Whether to translate for the sequential estimator.

        Returns
        -------
        dict of str to Any
            Validated keyword arguments for the implementation engine.
        """


#: Flat keyword shortcut to ``(configuration group, field)``, read by
#: :meth:`TMLEMethod.with_overrides`.  Module level so the invariant that keeps it honest is
#: testable: a shortcut spelled the same as one of the groups' fields must set *that* field.
#: ``tests.unit.test_causal_study`` checks it, after ``alpha=`` was found routing to
#: :attr:`Targeting.submodel_alpha` while :attr:`Inference.alpha` was the significance level.
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


#: Point-engine settings with no longitudinal implementation. The public names are kept beside
#: their normalized fields so an error names the declaration the caller wrote. ``cross_fit`` is
#: deliberately absent: the longitudinal translation supports it by resolving ``False`` to
#: ``n_folds=1``. All 17 settings below must either acquire a longitudinal derivation or remain
#: explicit refusals; dropping one from the translation is never a supported interpretation.
_LONGITUDINAL_POINT_ONLY: tuple[tuple[str, str, str], ...] = (
    ("models", "missingness_learner", "missingness_learner"),
    ("models", "intermediate_learner", "intermediate_learner"),
    ("models", "density_bins", "density_bins"),
    ("models", "screen_treatment", "screen_treatment"),
    ("models", "screen_threshold", "screen_threshold"),
    ("models", "min_retain", "min_retain"),
    ("cross_fitting", "repeats", "repeats"),
    ("cross_fitting", "stratify_by", "stratify_folds"),
    ("cross_fitting", "targeting_scheme", "targeting_scheme"),
    ("cross_fitting", "fold_evaluation", "cv_evaluation"),
    ("targeting", "fluctuation", "fluctuation"),
    ("targeting", "algorithm", "targeting"),
    ("targeting", "nuisance_bound", "nuisance_bound"),
    ("targeting", "target_weights", "target_weights"),
    ("targeting", "step_size", "step_size"),
    ("inference", "n_bootstrap", "n_bootstrap"),
    ("inference", "bootstrap_resampling", "bootstrap_resampling"),
)


def _differs_from_default(value: Any, default: Any) -> bool:
    """Compare normalized settings without asking learner objects to compare to ``None``."""
    if default is None:
        return value is not None
    return bool(value != default)


@dataclass(frozen=True)
class TMLEMethod:
    """Normalized configuration for the built-in analytic TMLE method.

    ``IdentifiedEffect.estimate(method="tmle", ...)`` normalizes keyword shortcuts into
    this object before fitting. Passing one directly gives the same computational path.

    Parameters
    ----------
    models : ModelSpec
        Nuisance learners and related model settings.
    cross_fitting : CrossFitting
        Sample-splitting settings.
    targeting : Targeting
        Fluctuation and solver settings.
    inference : Inference
        Interval and resampling settings.
    runtime : Runtime
        Seed, run identifier, and parallelism.
    name : str
        Stable method name.

    See Also
    --------
    CollaborativeTMLEMethod : Selects the treatment mechanism against the targeted loss.
    DRTMLEMethod : Corrects both nuisances with reduced-dimension regressions.
    ModelSpec : The learners this configuration fits with.
    cleverly.IdentifiedEffect : What this configuration is passed to.

    Examples
    --------
    >>> from cleverly import CrossFitting, TMLEMethod
    >>> method = TMLEMethod(cross_fitting=CrossFitting(n_folds=5), name="tmle")
    >>> method.cross_fitting.n_folds
    5

    The flat shortcuts route to the same fields, so the two spellings agree:

    >>> TMLEMethod().with_overrides(n_folds=5) == method
    True
    """

    models: ModelSpec = ModelSpec()
    cross_fitting: CrossFitting = CrossFitting()
    targeting: Targeting = Targeting()
    inference: Inference = Inference()
    runtime: Runtime = Runtime()
    name: str = "tmle"

    def with_overrides(self, **overrides: Any) -> TMLEMethod:
        """Return a copy with flat shortcuts normalized by concern.

        Parameters
        ----------
        **overrides : Any
            Fields listed in the public shortcut table, such as ``n_folds`` or ``alpha``.

        Returns
        -------
        TMLEMethod
            New immutable configuration.

        Raises
        ------
        MethodConfigurationError
            If an option is not supported by the method.

        Notes
        -----
        ``alpha=`` sets :attr:`Inference.alpha`, the reported interval's significance
        level. Use ``submodel_alpha=`` for :attr:`Targeting.submodel_alpha`, the separate
        logistic-submodel bound.
        """
        groups = SHORTCUTS
        known = {name for fields in groups.values() for name in fields}
        unknown = sorted(set(overrides) - known)
        if unknown:
            raise MethodConfigurationError(
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
        """Translate normalized groups into the selected engine API.

        Parameters
        ----------
        longitudinal : bool
            Whether to target the longitudinal engine.

        Returns
        -------
        dict of str to Any
            Engine keyword arguments.
        """
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
            defaults = TMLEMethod()
            refused = [
                public_name
                for group_name, field_name, public_name in _LONGITUDINAL_POINT_ONLY
                if _differs_from_default(
                    getattr(getattr(self, group_name), field_name),
                    getattr(getattr(defaults, group_name), field_name),
                )
            ]
            if refused:
                raise MethodConfigurationError(
                    "longitudinal TMLE cannot apply point-treatment option(s): "
                    f"{refused}. Remove them or use a PointTreatment design."
                )
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
    """Configure collaborative treatment-mechanism selection.

    Parameters
    ----------
    models : ModelSpec
        Nuisance learners and related model settings.
    cross_fitting : CrossFitting
        Sample-splitting settings.
    targeting : Targeting
        Fluctuation and solver settings.
    inference : Inference
        Interval and resampling settings.
    runtime : Runtime
        Seed, run identifier, and parallelism.
    name : str
        Stable method name.
    strategy : {"greedy", "ordered", "discrete", "oat"}, default="greedy"
        Collaborative search strategy.
    preorder : {"logistic", "partial_correlation"} or None, default=None
        Preordering rule for candidate covariates.
    ordering : tuple of str or None, default=None
        Explicit covariate order.
    candidates : tuple of tuple of str or None, default=None
        Explicit nested adjustment candidates.
    selection_folds : int, default=5
        Outer folds used to select a candidate.
    selection_inner_folds : int, default=2
        Inner folds used to evaluate learners during selection.
    loss : {"auto", "loglik", "squared"}, default="auto"
        Candidate-selection loss.
    penalty : bool, default=True
        Whether to apply the collaborative complexity penalty.
    selection_estimand : str, default="ate"
        Estimand used by the selector.

    See Also
    --------
    TMLEMethod : The base configuration, without covariate selection.
    DRTMLEMethod : The other correction for a mechanism that is hard to fit.
    cleverly.estimators.ctmle : Why selection is made against the targeted loss.

    Examples
    --------
    >>> from cleverly import CollaborativeTMLEMethod
    >>> method = CollaborativeTMLEMethod(strategy="greedy", selection_folds=3)
    >>> method.name, method.selection_folds
    ('collaborative_tmle', 3)

    The collaborative fields are additions.  Everything :class:`TMLEMethod` configures is
    still configured the same way:

    >>> from cleverly import CrossFitting
    >>> CollaborativeTMLEMethod(cross_fitting=CrossFitting(n_folds=5)).cross_fitting.n_folds
    5
    """

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
        """Translate this configuration for the point-treatment CTMLE engine.

        Parameters
        ----------
        longitudinal : bool
            Must be false because no longitudinal collaborative score is supported.

        Returns
        -------
        dict of str to Any
            CTMLE engine keyword arguments.
        """
        if longitudinal:
            raise MethodConfigurationError("collaborative TMLE has no longitudinal derivation")
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
    """Configure the reduced-dimension DR-TMLE correction.

    Parameters
    ----------
    models : ModelSpec
        Nuisance learners and related model settings.
    cross_fitting : CrossFitting
        Sample-splitting settings.
    targeting : Targeting
        Fluctuation and solver settings.
    inference : Inference
        Interval and resampling settings.
    runtime : Runtime
        Seed, run identifier, and parallelism.
    name : str
        Stable method name.
    guard : tuple of {"Q", "g"}, default=("Q", "g")
        Nuisance components protected by the correction.
    reduction : {"univariate", "bivariate"}, default="univariate"
        Reduced-regression family.
    reduced_outcome_learner : estimator or None, default=None
        Learner for the reduced outcome regression.
    reduced_treatment_learner : estimator or None, default=None
        Learner for the reduced treatment regression.
    reduced_crossfit : {"pooled", "nested"}, default="pooled"
        Cross-fitting policy for reduced regressions.
    update_order : {"drtmle", "benkeser"}, default="drtmle"
        Order of reduced targeting updates.
    evaluation : dataframe or None, default=None
        Optional evaluation data for a companion fit.
    randomized : bool, default=False
        Whether treatment probabilities are known by design.
    treatment_probabilities : array-like or mapping, default=None
        Known treatment probabilities for randomized treatment.

    See Also
    --------
    TMLEMethod : The base configuration, without the reduced-dimension correction.
    CollaborativeTMLEMethod : The other correction for a mechanism that is hard to fit.
    ModelSpec : Owns the first-stage learners the reduced regressions correct.

    Examples
    --------
    >>> from cleverly import DRTMLEMethod
    >>> method = DRTMLEMethod(guard=("Q", "g"))
    >>> method.name, method.guard
    ('drtmle', ('Q', 'g'))

    A randomized trial knows its own treatment mechanism, so only the outcome regression
    needs guarding:

    >>> DRTMLEMethod(guard=("Q",), randomized=True).guard
    ('Q',)
    """

    guard: tuple[str, ...] = ("Q", "g")
    reduction: str = "univariate"
    reduced_outcome_learner: Any = None
    reduced_treatment_learner: Any = None
    reduced_crossfit: str = "pooled"
    update_order: str = "drtmle"
    evaluation: Any = None
    randomized: bool = False
    treatment_probabilities: Any = None
    name: str = "drtmle"

    def __post_init__(self) -> None:
        _validate_learner(self.reduced_outcome_learner, "reduced_outcome_learner")
        _validate_learner(self.reduced_treatment_learner, "reduced_treatment_learner")

    def estimator_kwargs(self, *, longitudinal: bool = False) -> dict[str, Any]:
        """Translate this configuration for the point-treatment DR-TMLE engine.

        Parameters
        ----------
        longitudinal : bool
            Must be false because no longitudinal correction is supported.

        Returns
        -------
        dict of str to Any
            DR-TMLE engine keyword arguments.
        """
        if longitudinal:
            raise MethodConfigurationError("DR-TMLE has no longitudinal derivation")
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
