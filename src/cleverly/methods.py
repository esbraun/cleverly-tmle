"""Typed estimation-method configuration for the causal-workflow API."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

from ._typing import (
    FluctuationKind,
    FoldStrata,
    GBounds,
    IterateChoice,
    StoppingRule,
    SurvivalModel,
    TargetingMethod,
    TargetingScheme,
)
from .exceptions import MethodConfigurationError, Refusal, RefusalReason
from .fluctuation._score import check_iterate, check_step_bound, check_stopping_rule
from .inference.bootstrap import Resampling
from .inference.multiplier import MultiplierKind
from .learners.crossfit import SplitPlan, _cross_fit_policy_refusal
from .learners.library import _validate_learner
from .longitudinal.pooled_hazards import check_hazard_pooling
from .longitudinal.ratio_trim import check_ratio_trim

__all__ = [
    "ADAPTIVE_IPTW_LONGITUDINAL_REFUSAL",
    "BASELINE_HAZARD_POOLING_REFUSAL",
    "BASELINE_POINT_REFUSAL",
    "BASELINE_RATIO_TRIM_REFUSAL",
    "DEFAULT_LONGITUDINAL_G_BOUNDS",
    "DEFAULT_LONGITUDINAL_MULTIPLIER",
    "DEFAULT_POINT_MULTIPLIER",
    "IPSI_NO_FLUCTUATION",
    "IPSI_POINT_REFUSAL",
    "IPSI_RATIO_TRIM_REFUSAL",
    "IPW_POINT_REFUSAL",
    "LTMLE_TILT_REFUSAL",
    "NODE_BOUNDS_MSM_REFUSAL",
    "REPEATED_OUTCOME_REFUSAL",
    "SDR_POINT_REFUSAL",
    "SDR_TILT_REFUSAL",
    "SHORTCUTS",
    "AdaptiveIPTWMethod",
    "CollaborativeTMLEMethod",
    "CrossFitting",
    "DRTMLEMethod",
    "EstimationMethod",
    "GComputationMethod",
    "IPSIMethod",
    "IPWMethod",
    "Inference",
    "MethodAvailability",
    "ModelSpec",
    "ParametricGFormulaMethod",
    "Runtime",
    "SDRMethod",
    "TMLEMethod",
    "Targeting",
    "baseline_no_fluctuation_refusal",
    "node_bounds_refusal",
    "sdr_no_fluctuation_refusal",
]


#: What ``"auto"`` resolves to, per engine.  These restate the engines' own signature defaults
#: rather than importing them, because importing ``cleverly.estimators`` here would invert this
#: module's place in the dependency order.  A restated default is a default that can drift, so
#: ``tests.unit.test_causal_study`` reads both signatures and fails when one of these stops
#: matching -- which is what makes the duplication safe rather than merely tidy.
DEFAULT_POINT_MULTIPLIER = 1000
DEFAULT_LONGITUDINAL_MULTIPLIER = 2000
DEFAULT_LONGITUDINAL_G_BOUNDS = (0.01, 1.0)

#: Why the SDR estimator refuses a fluctuation setting. The setting has no meaning on an
#: estimator with no fluctuation, so the code is the one :data:`IPSI_NO_FLUCTUATION` takes.
#: The alternative holds ``{name}``, the engine's own name for the setting, which
#: :func:`sdr_no_fluctuation_refusal` fills in.
_SDR_NO_FLUCTUATION = Refusal(
    RefusalReason.INVALID_DECLARATION,
    "the SDR estimator fits no fluctuation, so it has no logistic shrink, iteration limit or "
    "score tolerance.",
    alternative="LTMLE takes {name}=, so pass the setting there.",
)
#: Why a baseline estimator refuses a fluctuation setting.  The alternative holds ``{name}``,
#: which :func:`baseline_no_fluctuation_refusal` fills in.
_BASELINE_NO_FLUCTUATION = Refusal(
    RefusalReason.INVALID_DECLARATION,
    "the sequential g-computation and IPW estimators fit no fluctuation, so they have no "
    "logistic shrink, iteration limit or score tolerance.",
    alternative="LTMLE takes {name}=, so pass the setting there.",
)
#: Why a baseline estimator refuses ``ratio_trim=``.
BASELINE_RATIO_TRIM_REFUSAL = Refusal(
    RefusalReason.API_REDIRECTION,
    "the baseline estimators (sequential g-computation, sequential IPW and the parametric "
    "g-formula) do not apply ratio_trim=. The trim follows lmtp's rule for the per-node "
    "density ratios of a targeted fit.",
    alternative="Omit ratio_trim=, or trim the ratios of an LTMLE or SDR fit.",
)
#: Why a baseline estimator refuses ``hazard_pooling="pooled"``.
BASELINE_HAZARD_POOLING_REFUSAL = Refusal(
    RefusalReason.API_REDIRECTION,
    "the baseline estimators (sequential g-computation, sequential IPW and the parametric "
    "g-formula) fit their own per-node parametric models and do not read a pooled "
    "censoring hazard.",
    alternative="Pass hazard_pooling='node', or fit LTMLE or SDR for a pooled censoring hazard.",
)
#: Why a point-treatment functional refuses ``method="gcomp"``, ``method="ipw"`` and
#: ``method="parametric_gformula"``.
BASELINE_POINT_REFUSAL = Refusal(
    RefusalReason.API_REDIRECTION,
    "sequential g-computation, sequential IPW and the parametric g-formula are the baselines "
    "of a longitudinal fit.",
    alternative="A point-treatment functional fits tmle, drtmle or collaborative_tmle.",
)
#: Why a point-treatment functional refuses ``method="ipw"`` (Q30). It names the point-treatment
#: IPW estimate beside the targeted estimators.
IPW_POINT_REFUSAL = Refusal(
    RefusalReason.API_REDIRECTION,
    "sequential IPW is the baseline of a longitudinal fit.",
    alternative=(
        "A point-treatment functional fits tmle, drtmle or collaborative_tmle. "
        "adaptive_iptw is the point-treatment IPW estimate."
    ),
)
#: Why a longitudinal design refuses ``method="adaptive_iptw"`` (R12).
ADAPTIVE_IPTW_LONGITUDINAL_REFUSAL = Refusal(
    RefusalReason.API_REDIRECTION,
    "Adaptive IPTW has no longitudinal derivation.",
    alternative="Use SequentialIPW (method='ipw').",
)
#: Why a point-treatment functional refuses ``method="sdr"`` (R4).
SDR_POINT_REFUSAL = Refusal(
    RefusalReason.API_REDIRECTION,
    "the SDR estimator is the sequential regression of a longitudinal fit.",
    alternative="A point-treatment functional fits tmle, drtmle or collaborative_tmle.",
)
#: Why ``LTMLE`` refuses an odds tilt with an estimated mechanism (R1 in ``test_ipsi_e2e.py``).
LTMLE_TILT_REFUSAL = Refusal(
    RefusalReason.API_REDIRECTION,
    "an odds tilt reads the estimated treatment mechanism, so its influence function has a "
    "mechanism term at every node (Kennedy 2019, Theorem 2), and LTMLE fits no mechanism "
    "fluctuation over time.",
    alternative=(
        "Fit IPSI(regimens=[Incremental(2.0), ...]), which runs Kennedy's cross-fitted "
        "estimator. With a known treatment mechanism, LTMLE fits the tilt as a known "
        "stochastic policy."
    ),
)
#: Why ``SDR`` refuses an odds tilt (R2 in ``test_ipsi_e2e.py``).
SDR_TILT_REFUSAL = Refusal(
    RefusalReason.API_REDIRECTION,
    "the SDR estimator needs a policy that does not depend on the observed-data law (Díaz, "
    "Williams, Hoffman and Schenck 2023, Section 4). An odds tilt reads the treatment "
    "mechanism, so its estimator is not doubly robust (Kennedy 2019, Section 3.3).",
    alternative="Fit IPSI(regimens=[...]).",
)
#: Why an estimator outside ``REPEATED_OUTCOME_KINDS`` refuses a repeated outcome (R7,
#: ``refuse_repeated_outcome``).  ``{name}`` is the refusing class and ``{kinds}`` the
#: estimators that fit one, which ``repeated_outcome_refusal`` fills in.
REPEATED_OUTCOME_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "{name} has no per-horizon reduction for a repeated outcome (outcome=Repeated([...])).",
    alternative=(
        "{kinds} fit{verb} one. Each horizon is the end-of-study fit of the panel truncated "
        "after it."
    ),
    roadmap="X32b",
)
#: Why an estimator with no logistic node fluctuation refuses ``node_bounds=``. The condition
#: holds ``{estimator}``, which :func:`node_bounds_refusal` fills in.
_NODE_BOUNDS_NO_FLUCTUATION = Refusal(
    RefusalReason.INVALID_DECLARATION,
    "{estimator} fits no logistic node fluctuation, so a node bound has nothing to bound.",
    alternative="LTMLE(node_bounds=...) bounds the logistic fluctuation of each node.",
)
#: Why ``node_bounds=`` meets a working model over regimens (``msm=``) today.
NODE_BOUNDS_MSM_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "node_bounds= bounds the intercept fluctuation of a per-regimen mean. A working model over "
    "regimens pools one fluctuation across the regimens, and its bounded form is not built.",
    alternative="Fit node_bounds= without msm=, and report the regimen levels.",
    roadmap="X78",
)
#: Why the IPSI estimator refuses a fluctuation setting (R5 in ``test_ipsi_e2e.py``).
IPSI_NO_FLUCTUATION = Refusal(
    RefusalReason.INVALID_DECLARATION,
    "IPSI fits no fluctuation: Kennedy's estimator averages the influence function at the "
    "fitted nuisances, so it has no logistic shrink, iteration limit, score tolerance or "
    "submodel.",
)
#: Why the IPSI estimator refuses ``ratio_trim=``.
IPSI_RATIO_TRIM_REFUSAL = Refusal(
    RefusalReason.INVALID_DECLARATION,
    "IPSI does not apply ratio_trim=. The trim follows lmtp's rule for a per-node density "
    "ratio, and an IPSI tilt node has no such ratio in its weight.",
    alternative="Omit ratio_trim=, or trim a static, dynamic or policy regimen with LTMLE or SDR.",
)
#: Why ``method="sdr"`` and ``method="ipsi"`` refuse the hazard and isotonic settings.
SURVIVAL_CURVE_TMLE_ONLY = (
    "survival_model=, a hazard targeting= algorithm, step_size= and isotonic= are settings of "
    "the targeted fit, method='tmle'. The {method} estimator fits no hazard, and no "
    "registered study measures its projected curve. Fit method='tmle', or remove them"
)
#: Why a point treatment refuses the IPSI estimator (R8 in ``test_ipsi_e2e.py``).
IPSI_POINT_REFUSAL = Refusal(
    RefusalReason.API_REDIRECTION,
    "IPSI fits a LongitudinalData panel.",
    alternative="For a point treatment fit TMLE(incremental=[...]), or pass a one-node panel.",
)


def baseline_no_fluctuation_refusal(name: str) -> Refusal:
    """The refusal of a fluctuation setting on a sequential g-computation or IPW fit.

    Parameters
    ----------
    name : str
        The engine's name for the refused setting, such as ``"alpha"``.

    Returns
    -------
    Refusal
        An :attr:`~cleverly.exceptions.RefusalReason.INVALID_DECLARATION` record whose
        alternative names ``name`` as the setting ``LTMLE`` takes.
    """
    return replace(
        _BASELINE_NO_FLUCTUATION,
        alternative=str(_BASELINE_NO_FLUCTUATION.alternative).format(name=name),
    )


def node_bounds_refusal(estimator: str) -> Refusal:
    """The refusal of ``node_bounds=`` on an estimator with no logistic node fluctuation.

    Parameters
    ----------
    estimator : str
        The refusing estimator, such as ``"the SDR estimator"``.

    Returns
    -------
    Refusal
        An :attr:`~cleverly.exceptions.RefusalReason.INVALID_DECLARATION` record whose
        alternative names ``LTMLE(node_bounds=...)``.
    """
    return replace(
        _NODE_BOUNDS_NO_FLUCTUATION,
        condition=str(_NODE_BOUNDS_NO_FLUCTUATION.condition).format(estimator=estimator),
    )


def sdr_no_fluctuation_refusal(name: str) -> Refusal:
    """The refusal of a fluctuation setting on an SDR fit.

    Parameters
    ----------
    name : str
        The engine's name for the refused setting, such as ``"alpha"``.

    Returns
    -------
    Refusal
        An :attr:`~cleverly.exceptions.RefusalReason.INVALID_DECLARATION` record, as
        :data:`IPSI_NO_FLUCTUATION` is, whose alternative names ``name`` as the setting
        ``LTMLE`` takes.
    """
    return replace(
        _SDR_NO_FLUCTUATION,
        alternative=str(_SDR_NO_FLUCTUATION.alternative).format(name=name),
    )


@dataclass(frozen=True)
class MethodAvailability:
    """Report whether a method can estimate an identified effect.

    Parameters
    ----------
    name : str
        Method name accepted by ``estimate(method=...)``.
    available : bool
        Whether the method supports the identified functional. The record is per method
        name. A configuration of an available method can still refuse before any learner.
    refusal : Refusal or None
        The record of the refusal when ``available`` is false: its reason code, the failed
        condition and the alternative. :attr:`reason` is its text, and a selection of the
        method raises the same record.
    """

    name: str
    available: bool
    refusal: Refusal | None = None

    @property
    def reason(self) -> str | None:
        """The text of :attr:`refusal`, or ``None`` when the method is available."""
        return None if self.refusal is None else self.refusal.message()


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
        Observation learner.  It fits the outcome observation mechanism and, for a declared
        missing treatment, the treatment observation mechanism ``P(Delta_A = 1 | W)``.
    intermediate_learner : estimator or None
        Intermediate-variable learner for controlled direct effects.
    pseudo_learner : estimator or None
        Sequential pseudo-outcome learner.
    censoring_learner : estimator or None
        Longitudinal observation-mechanism learner.
    density_bins : int or None
        Number of bins for a continuous-treatment conditional density.  ``None`` grows the
        count with the sample, ``max(20, ceil(2 n^(1/3)))``.  A growing count is necessary for
        a consistent density, and the hazard learner must also be consistent.
    ratio : {"density", "classifier"}
        How a modified treatment policy's density ratio is estimated.  ``"density"`` divides
        two fitted conditional densities.  ``"classifier"`` reads the odds of a stacked
        classifier (Díaz et al. 2023, Section 5.4).  It applies to a point fit with
        ``ModifiedTreatmentPolicy`` and to a longitudinal fit with a continuous or vector
        node.  A study refuses ``"classifier"`` where no policy ratio is estimated.
    screen_treatment : bool
        Whether to screen treatment-mechanism covariates by correlation.
    screen_threshold : float
        Absolute correlation threshold used by screening.
    min_retain : int or None
        Minimum number of covariates retained by screening.
    hazard_pooling : {"node", "pooled"}
        How a longitudinal fit estimates its censoring mechanism.  ``"node"`` fits one model
        per node.  ``"pooled"`` fits one model over every at-risk unit and node, with the
        node's grid time as a column, as ``survtmle`` does.  Under
        ``survival_model="hazard"`` it also fits each cause's hazard once over the eligible
        rows of every node.  It needs one baseline treatment and no time-varying covariate.
    outcome_by_arm : bool
        Whether to fit one outcome regression per treatment arm, each on the arm's rows and
        the covariates alone (R's ``drtmle(stratify=TRUE)``).  The default fits one regression
        on the covariates and the treatment.  A continuous dose has no arms and is refused,
        and so is a longitudinal design.
    survival_model : {"iterated_mean", "hazard"}
        How a longitudinal fit models a survival curve.  ``"hazard"`` targets the
        cause-specific hazards of a held baseline treatment; see
        :class:`~cleverly.longitudinal.LTMLE`.  A point-treatment fit refuses
        ``"hazard"``.

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

    Fit one outcome regression for each treatment arm, as ``drtmle(stratify=TRUE)`` does:

    >>> ModelSpec(outcome_learner=LinearRegression(), outcome_by_arm=True).outcome_by_arm
    True
    """

    outcome_learner: Any = None
    treatment_learner: Any = None
    missingness_learner: Any = None
    intermediate_learner: Any = None
    pseudo_learner: Any = None
    censoring_learner: Any = None
    density_bins: int | None = None
    ratio: Literal["density", "classifier"] = "density"
    screen_treatment: bool = False
    screen_threshold: float = 0.1
    min_retain: int | None = None
    hazard_pooling: str = "node"
    outcome_by_arm: bool = False
    survival_model: SurvivalModel = "iterated_mean"

    def __post_init__(self) -> None:
        check_hazard_pooling(self.hazard_pooling)
        for name in (
            "outcome_learner",
            "treatment_learner",
            "missingness_learner",
            "intermediate_learner",
            "pseudo_learner",
            "censoring_learner",
        ):
            _validate_learner(getattr(self, name), name)
        if self.ratio not in ("density", "classifier"):
            raise MethodConfigurationError(
                f"ratio must be 'density' or 'classifier'; got {self.ratio!r}"
            )


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
        Independent outer-fold assignments to combine by their median, on a point-treatment
        or a longitudinal design. A value above one requires ``enabled=True``.
    stratify_by : {"none", "treatment", "treatment+outcome"}, default="none"
        Fold-stratification policy. ``"none"`` draws folds that read neither the
        treatment nor the outcome, and it is the only value a fit that draws a split
        accepts. ``"treatment"`` and ``"treatment+outcome"`` are refused with
        ``enabled=True``, and refused at every setting by selector-based
        :class:`CollaborativeTMLEMethod` fits, whose search draws selection folds without
        cross-fitting. In-sample outcome-adaptive fits draw no such split and accept the
        unused declaration. No shipped result covers a partition read off the data the
        fit then conditions on.
    targeting_scheme : {"pooled", "fold"}, default="pooled"
        Pooled or fold-specific targeting scheme.
    fold_evaluation : bool, default=False
        Whether to retain fold-evaluated CV-TMLE estimates.
    split_plan : SplitPlan, optional
        Reusable outer-fold assignments in repeat-major order. The plan must record
        how :func:`~cleverly.learners.random_partition` drew each repeat, as
        ``result.split_plan`` from an unstratified fit does. A plan without that record
        is refused here.

    See Also
    --------
    TMLEMethod : Method configuration this object is the splitting half of.
    Targeting : Whether the fluctuation is pooled or fitted per fold.
    SplitPlan : Reusable outer-fold assignments accepted by this configuration.
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
    stratify_by: FoldStrata = "none"
    targeting_scheme: TargetingScheme = "pooled"
    fold_evaluation: bool = False
    split_plan: SplitPlan | None = None

    def __post_init__(self) -> None:
        # One ordered message source, two exception contracts: the engine raises
        # ValueError for the same inputs. ``n_bootstrap`` lives on ``Inference``, so
        # ``TMLEMethod.__post_init__`` asks the bootstrap question, and whether the
        # method is a collaborative one is a fact about the method rather than about
        # this group, so ``CollaborativeTMLEMethod`` asks the selection-fold question.
        # See ``_cross_fit_policy_refusal``.
        refusal = _cross_fit_policy_refusal(
            cross_fit=self.enabled,
            n_folds=self.n_folds,
            repeats=self.repeats,
            split_plan=self.split_plan,
            stratify_folds=self.stratify_by,
            option_name="enabled",
        )
        if refusal is not None:
            raise MethodConfigurationError.from_refusal(refusal)


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
        The known support of a continuous outcome. ``None`` widens the observed range by
        ten percent, which reads every observed outcome, so a cross-fitted fit of a
        continuous outcome requires the support here. Refused on a binary outcome, which
        the estimator does not rescale. ``cleverly.datasets.make_nonlinear_bounded`` draws
        a proportion, for which ``q_bounds=(0.0, 1.0)`` states a fact about the law.
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
        Score-equation convergence tolerance. It is the bar of ``stopping="relative"``,
        and the tolerance of the inner Newton solve under every rule.
    stopping : {"relative", "scaled_var", "sample_size"}, default="relative"
        The rule that decides the score equation is solved. ``"relative"`` is the
        shipped test: the largest score over the mean absolute clever covariate is at
        most ``tol``. ``"scaled_var"`` and ``"sample_size"`` are the ``tmle3``
        ``convergence_type`` rules. Each compares the score, in the outcome's units, with
        ``sd / (sqrt(n_u) * min(log(n_u), 10))`` or ``1 / n_u``, where ``n_u`` counts the
        distinct units among the rows the solver receives and ``sd`` is the sample SD of
        the column's per-unit influence curve. A declared rule is refused beside
        ``fluctuation="linear"``, which has no iteration to stop.
    step_bound : float or None, default=None
        The longest step of one targeting iteration, in the units of the fluctuation
        coefficient. ``None`` takes the full Newton step. A positive number replaces the
        Newton step by the ``tmle3`` constrained step: the iteration moves along the
        normalised score vector ``s / ||s||`` and goes no further than ``step_bound``.
        It needs ``algorithm="iterative"`` and ``fluctuation="logistic"``. A bounded walk
        can stop short of ``tol``. Then the fit warns, reports ``converged=False`` and
        ``failure="max_iter_reached"``, and ``score_check`` fails it. The ``tmle3`` pairing
        is ``stopping="sample_size"`` with a larger ``max_iter``. Under
        ``stopping="relative"`` the walk can need more than the default ``max_iter``.
        Floating-point rounding and the platform affect the iteration count near ``tol``.
        The six arm-mean fits of ``tests/unit/test_step_bound.py::TestRelativeReachability``
        check the score criterion and recovery with a larger iteration budget.
        A fit whose initial regression already meets the rule takes no step.
    step_search : bool, default=True
        Under ``step_bound``, ``True`` takes the length in ``(0, step_bound]`` that
        minimises the weighted mean quasi-loss along the step, as ``tmle3`` does with
        ``optim_delta_epsilon``. ``False`` takes the whole bound, as ``delta_epsilon`` does.
        It needs ``step_bound``.
    iterate : {"last", "best"}, default="last"
        Which iterate of the targeting walk the fit reports. ``"last"`` reports the iterate
        at which the walk stopped. ``"best"`` is ``use_best`` of ``tmle3``: it reports the
        iterate ``k >= 1`` whose raw score vector has the smallest 2-norm, the first of any
        tie. The initial fit is never reported while a step exists. The walk still stops by
        the rule, ``n_iter`` and ``trace`` describe the whole walk, and ``converged`` is
        judged at the reported iterate. ``Fluctuation.best_iterate`` records the index. The
        best iterate differs from the last on a walk that oscillates, such as
        ``step_search=False``. On ``algorithm="one_step"`` the norm never rises, so the best
        iterate is the last. It is refused beside ``fluctuation="linear"``, which has no
        iterates.
    isotonic : bool, default=False
        Whether a longitudinal survival fit projects each curve's levels and band limits
        onto a non-decreasing curve; see :class:`~cleverly.longitudinal.LTMLE`.  A
        point-treatment fit refuses ``True``.
    ratio_trim : float or None, default=None
        A longitudinal fit's per-node density ratios are trimmed at this quantile of all of
        them, as ``lmtp_control(.trim=)`` does. A fit whose trim moves a ratio reports its
        point estimate with the ``"trimmed_ratio_plugin"`` status and no interval. ``None``
        trims nothing. A point-treatment fit refuses it.
    node_bounds : mapping or None, default=None
        A declared outcome range for chosen nodes of a longitudinal fit.
        The keys are grid times and the values are ``(lower, upper)`` pairs on the reported
        outcome scale. A competing-risk fit takes one such mapping for each cause name.
        A named node regresses the outcome rescaled to ``[0, 1]`` by its bounds. It fluctuates
        along the logit of the rescaled prediction, as Gruber and van der Laan (2010) do for
        a bounded continuous outcome. Its targeted prediction stays inside the bounds. A node
        with no entry takes the shipped fit, bit for bit. The fit warns and reports the node
        as not converged, with ``failure="bound_excludes_data"``, when the weighted mean
        outcome of the node's followers lies outside the bounds, because no fluctuation
        inside the bounds solves the score equation. ``node_bounds`` needs no ``msm=``. Under
        ``survival_model="hazard"`` the bounds are on each cause-specific hazard, and a named
        node bounds every cause. A point-treatment fit, ``SDR``, ``IPSI`` and the baselines
        refuse it. See :class:`~cleverly.longitudinal.LTMLE`.

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

    Stop the targeting step by the ``tmle3`` ``scaled_var`` rule instead of the shipped
    relative score:

    >>> Targeting(stopping="scaled_var").stopping
    'scaled_var'
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
    stopping: StoppingRule = "relative"
    step_bound: float | None = None
    step_search: bool = True
    iterate: IterateChoice = "last"
    isotonic: bool = False
    ratio_trim: float | None = None
    node_bounds: Mapping[Any, Any] | None = None

    def __post_init__(self) -> None:
        check_iterate(self.iterate, self.fluctuation)
        check_stopping_rule(self.stopping, self.fluctuation)
        check_step_bound(self.step_bound, self.step_search, self.fluctuation, self.algorithm)
        check_ratio_trim(self.ratio_trim)


@dataclass(frozen=True)
class Inference:
    """Configure confidence intervals and resampling.

    Parameters
    ----------
    alpha : float, default=0.05
        Significance level for reported intervals.
    n_bootstrap : int, default=0
        Bootstrap fits. Zero disables full-refit bootstrap inference.
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
    cleverly.inference.run_bootstrap : The full-refit bootstrap ``n_bootstrap`` requests.

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
        "ratio": "ratio",
        "screen_treatment": "screen_treatment",
        "screen_threshold": "screen_threshold",
        "min_retain": "min_retain",
        "hazard_pooling": "hazard_pooling",
        "outcome_by_arm": "outcome_by_arm",
        "survival_model": "survival_model",
    },
    "cross_fitting": {
        "cross_fit": "enabled",
        "n_folds": "n_folds",
        "learner_folds": "learner_folds",
        "repeats": "repeats",
        "stratify_folds": "stratify_by",
        "targeting_scheme": "targeting_scheme",
        "cv_evaluation": "fold_evaluation",
        "split_plan": "split_plan",
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
        "ratio_trim": "ratio_trim",
        "stopping": "stopping",
        "step_bound": "step_bound",
        "step_search": "step_search",
        "iterate": "iterate",
        "isotonic": "isotonic",
        "node_bounds": "node_bounds",
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
#: ``n_folds=1``. Each setting below must either acquire a longitudinal derivation or remain
#: an explicit refusal; dropping one from the translation is never a supported interpretation.
_LONGITUDINAL_POINT_ONLY: tuple[tuple[str, str, str], ...] = (
    ("models", "missingness_learner", "missingness_learner"),
    ("models", "intermediate_learner", "intermediate_learner"),
    ("models", "screen_treatment", "screen_treatment"),
    ("models", "screen_threshold", "screen_threshold"),
    ("models", "min_retain", "min_retain"),
    ("models", "outcome_by_arm", "outcome_by_arm"),
    ("cross_fitting", "stratify_by", "stratify_folds"),
    ("cross_fitting", "targeting_scheme", "targeting_scheme"),
    ("cross_fitting", "fold_evaluation", "cv_evaluation"),
    ("cross_fitting", "split_plan", "split_plan"),
    ("targeting", "fluctuation", "fluctuation"),
    ("targeting", "nuisance_bound", "nuisance_bound"),
    ("targeting", "target_weights", "target_weights"),
)


#: The refusal of ``ratio_trim=`` on a point-treatment fit, which names the bound it has.
_POINT_RATIO_TRIM = Refusal(
    RefusalReason.API_REDIRECTION,
    "ratio_trim= trims the per-node density ratios of a longitudinal fit, as lmtp's .trim "
    "does. A point-treatment fit bounds its mechanism with g_bounds=.",
    alternative=(
        "Declare the point treatment as a one-node LongitudinalTreatment to trim its ratio."
    ),
)

#: The refusal of ``hazard_pooling=`` other than ``"node"`` on a point-treatment fit.
_POINT_HAZARD_POOLING = Refusal(
    RefusalReason.INVALID_DECLARATION,
    "point-treatment TMLE cannot apply longitudinal option(s): ['hazard_pooling']. "
    "hazard_pooling= pools the censoring model of a longitudinal fit over its nodes, and a "
    "point-treatment fit has one censoring indicator.",
    alternative="Remove it, or use a LongitudinalTreatment design.",
)

#: The refusal of ``node_bounds=`` on a point-treatment functional (R-i5).
_POINT_NODE_BOUNDS = Refusal(
    RefusalReason.INVALID_DECLARATION,
    "node_bounds= bounds the logistic fluctuation of each node of a longitudinal "
    "iterated-mean fit, and a point-treatment functional has no such nodes.",
    alternative="Remove it, or declare a longitudinal or TimeToEvent design.",
)


def _differs_from_default(value: Any, default: Any) -> bool:
    """Compare normalized settings without asking learner objects to compare to ``None``."""
    if default is None:
        return value is not None
    return bool(value != default)


def _refuse_node_bounds(method: TMLEMethod, estimator: str) -> None:
    """Refuse ``node_bounds=`` on an estimator that fits no logistic node fluctuation."""
    if method.targeting.node_bounds is not None:
        raise MethodConfigurationError.from_refusal(node_bounds_refusal(estimator))


def _refuse_curve_settings(method: TMLEMethod, name: str) -> None:
    """Refuse a hazard or isotonic setting on an estimator that is not the targeted fit."""
    if (
        method.models.survival_model != "iterated_mean"
        or method.targeting.algorithm != "iterative"
        or method.targeting.step_size != Targeting().step_size
        or method.targeting.isotonic
    ):
        raise MethodConfigurationError(SURVIVAL_CURVE_TMLE_ONLY.format(method=name))


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

    #: Whether this method can draw collaborative selection folds. The fold-policy
    #: refusal also checks the strategy: OAT selects nothing and draws none of them.
    _collaborative: ClassVar[bool] = False
    #: Why this method refuses ``ratio="classifier"``: ``None`` where it estimates a policy
    #: ratio.  A method that fits no policy ratio refuses the setting at construction rather
    #: than ignore it.
    _ratio_refusal: ClassVar[Refusal | None] = None
    #: Why this method refuses ``density_bins=``: ``None`` where it fits a density.
    _density_bins_refusal: ClassVar[Refusal | None] = None

    models: ModelSpec = ModelSpec()
    cross_fitting: CrossFitting = CrossFitting()
    targeting: Targeting = Targeting()
    inference: Inference = Inference()
    runtime: Runtime = Runtime()
    name: str = "tmle"

    def __post_init__(self) -> None:
        """Run the ordered cross-fitting refusal with the bootstrap setting included.

        The check repeats every rule :class:`CrossFitting` already applied, and adds the one
        rule that group cannot apply alone: a supplied ``split_plan`` lives on
        :class:`CrossFitting` and ``n_bootstrap`` on :class:`Inference`. A cross-group rule
        belongs to the object that holds both groups, and it belongs at construction:
        :meth:`estimator_kwargs` is a translation, and a refusal buried in a translation fires
        later than the declaration that earned it. ``DRTMLEMethod.__post_init__`` is the
        precedent. The reason comes from the same ordered refusal the engine reads, so both
        layers name one reason for one input.
        """
        cross = self.cross_fitting
        refusal = _cross_fit_policy_refusal(
            cross_fit=cross.enabled,
            n_folds=cross.n_folds,
            repeats=cross.repeats,
            split_plan=cross.split_plan,
            n_bootstrap=self.inference.n_bootstrap,
            stratify_folds=cross.stratify_by,
            collaborative=self._collaborative
            and getattr(self, "strategy", None) not in ("oat", "derivative"),
            option_name="enabled",
        )
        if refusal is not None:
            raise MethodConfigurationError.from_refusal(refusal)
        if self.models.ratio != "density" and self._ratio_refusal is not None:
            raise MethodConfigurationError.from_refusal(self._ratio_refusal)
        if self.models.density_bins is not None and self._density_bins_refusal is not None:
            raise MethodConfigurationError.from_refusal(self._density_bins_refusal)

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
                f"unsupported tmle option(s): {unknown}.",
                reason=RefusalReason.INVALID_DECLARATION,
                alternative=(
                    "Study-design roles belong on PointTreatment; advanced method options "
                    "require a typed method object."
                ),
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
        split_plan = cross.split_plan
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
                    "n_bootstrap": inference.n_bootstrap,
                    "bootstrap_resampling": inference.bootstrap_resampling,
                    "repeats": cross.repeats,
                    "stopping": targeting.stopping,
                    "step_bound": targeting.step_bound,
                    "step_search": targeting.step_search,
                    "iterate": targeting.iterate,
                    "density_bins": models.density_bins,
                    "ratio": models.ratio,
                    "survival_model": models.survival_model,
                    "targeting": targeting.algorithm,
                    "step_size": targeting.step_size,
                    "isotonic": targeting.isotonic,
                    "node_bounds": targeting.node_bounds,
                }
            )
            # Longitudinal cumulative bounds have a fixed-pair contract. ``auto`` is a
            # point-treatment policy and has no sequential meaning.
            if common["g_bounds"] == "auto":
                common["g_bounds"] = DEFAULT_LONGITUDINAL_G_BOUNDS
            if common["n_multiplier"] == "auto":
                common["n_multiplier"] = DEFAULT_LONGITUDINAL_MULTIPLIER
            common["ratio_trim"] = targeting.ratio_trim
            common["hazard_pooling"] = models.hazard_pooling
            return common
        if targeting.ratio_trim is not None:
            raise MethodConfigurationError.from_refusal(_POINT_RATIO_TRIM)
        if models.hazard_pooling != TMLEMethod().models.hazard_pooling:
            raise MethodConfigurationError.from_refusal(_POINT_HAZARD_POOLING)
        if targeting.node_bounds is not None:
            raise MethodConfigurationError.from_refusal(_POINT_NODE_BOUNDS)
        if models.survival_model != "iterated_mean" or targeting.isotonic:
            raise MethodConfigurationError(
                "survival_model= and isotonic= apply to a survival curve over time nodes, and "
                "a point-treatment functional has no curve. Remove them, or declare a "
                "longitudinal or TimeToEvent design"
            )
        if common["n_multiplier"] == "auto":
            common["n_multiplier"] = DEFAULT_POINT_MULTIPLIER
        common.update(
            {
                "treatment_learner": models.treatment_learner,
                "missingness_learner": models.missingness_learner,
                "intermediate_learner": models.intermediate_learner,
                "density_bins": models.density_bins,
                "ratio": models.ratio,
                "screen_treatment": models.screen_treatment,
                "screen_threshold": models.screen_threshold,
                "min_retain": models.min_retain,
                "outcome_by_arm": models.outcome_by_arm,
                "cross_fit": cross.enabled,
                "repeats": cross.repeats,
                "split_plan": split_plan,
                "stratify_folds": cross.stratify_by,
                "targeting_scheme": cross.targeting_scheme,
                "cv_evaluation": cross.fold_evaluation,
                "fluctuation": targeting.fluctuation,
                "targeting": targeting.algorithm,
                "nuisance_bound": targeting.nuisance_bound,
                "target_weights": targeting.target_weights,
                "step_size": targeting.step_size,
                "stopping": targeting.stopping,
                "step_bound": targeting.step_bound,
                "step_search": targeting.step_search,
                "iterate": targeting.iterate,
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
    strategy : {"greedy", "ordered", "discrete", "lasso", "learners", "derivative", "oat"}, \
default="greedy"
        Collaborative search strategy. ``"greedy"``, ``"ordered"``, ``"discrete"``,
        ``"lasso"`` and ``"learners"`` report no confidence interval, no p-value and no
        standard error, with two exceptions that the Notes state. F18 in the roadmap holds
        the reason. ``"oat"`` reports no interval, for the reason F19 holds, except on a
        per-arm fit with ``ParametricOutcome`` and ``ParametricMechanism`` that
        ``cleverly.estimators.ctmle.per_arm_parametric_support`` admits (X58), in sample or
        cross-fitted, with any number of arms and with observation weights. That fit reports
        pointwise intervals and a simultaneous band. ``"derivative"`` is C-TMLE0. It
        selects nothing and reports the ordinary TMLE interval.
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
    oat_design : {"per_arm", "shared"} or None, default=None
        The treatment design of ``strategy="oat"``. ``None`` resolves to ``"per_arm"``.
        ``"shared"`` fits the one categorical mechanism of ``ctmle3`` and reports no
        interval, for the reason F19 in the roadmap holds. ``"per_arm"`` reports an interval
        only on the fits that the ``strategy`` entry names.
    lambdas : tuple of float or None, default=None
        The penalty grid of ``strategy="lasso"``: two or more penalties, strictly
        decreasing, with 0 allowed only as the last value. ``None`` uses ``glmnet``'s default
        grid with 0 appended. Under ``strategy="derivative"``, the pair ``(mechanism,
        partner)`` of two different penalties, or ``None`` to choose the mechanism penalty by
        cross-validation.
    treatment_candidates : tuple or None, default=None
        The declared candidates of ``strategy="learners"``, from the least to the most
        adaptive: unfitted learners, or :class:`~cleverly.estimators.PropensityArrays`
        pairs. ``treatment_learner`` on every covariate is appended as the last candidate.
        Under ``strategy="derivative"``, the pair ``(mechanism, partner)`` of two unfitted
        learners or two arrays of treatment probabilities.
    selection_split : tuple of int or None, default=None
        Fold labels ``0`` to ``V - 1``, one for each row, used in place of the random
        selection split. ``selection_folds`` must equal ``V``.
    patience : int or None, default=None
        Stop the cross-validated scoring of the candidate path after this many candidates
        score worse than the best so far, as ``patience`` does in R ``ctmle``. ``None`` reads
        the whole path. It applies to ``"greedy"``, ``"ordered"``, ``"discrete"``,
        ``"lasso"`` and ``"learners"``. The fit keeps its ``working_mechanism_plugin`` status.

    See Also
    --------
    TMLEMethod : The base configuration, without covariate selection.
    DRTMLEMethod : Extra score equations against one inconsistent nuisance, valid in the cases its
        contract lists.
    cleverly.estimators.ctmle : Why selection is made against the targeted loss.

    Notes
    -----
    The default ``"greedy"`` strategy selects a working treatment mechanism, and so do
    ``"ordered"``, ``"discrete"``, ``"lasso"`` and ``"learners"``. No result shows that the
    reported curve is the estimator's influence curve at that mechanism. Their estimates
    therefore raise :class:`~cleverly.exceptions.CapabilityError` from ``std_error``, ``ci``
    and ``pvalue``. Each estimate keeps ``plugin_std_error`` and ``plugin_interval`` as a
    diagnostic, which is not a confidence statement. For an interval, use
    :class:`TMLEMethod`. There are two exceptions. A ``"discrete"`` fit whose one declared
    candidate is the full adjustment set, and a ``"learners"`` fit with an empty
    ``treatment_candidates`` list, select nothing, equal the ordinary TMLE, and report the
    ordinary interval.

    ``strategy="oat"`` selects nothing, and it keeps the diagnostic only on either design:
    condition (v) of Benkeser, Cai and van der Laan (2020), Theorem 1, fails for an
    estimated outcome regression.

    ``strategy="derivative"`` adds a derivative column for each arm to one TMLE update.
    The columns contain the clever covariates, so the update solves the efficient influence
    curve equation, and the fit reports the ordinary TMLE interval.

    Examples
    --------
    >>> from cleverly import CollaborativeTMLEMethod
    >>> method = CollaborativeTMLEMethod(strategy="greedy", selection_folds=3)
    >>> method.name, method.selection_folds
    ('collaborative_tmle', 3)

    Stop the scoring of the candidate path once two candidates score worse than the best so far:

    >>> CollaborativeTMLEMethod(strategy="greedy", patience=2).patience
    2

    The collaborative fields are additions.  Everything :class:`TMLEMethod` configures is
    still configured the same way:

    >>> from cleverly import CrossFitting
    >>> CollaborativeTMLEMethod(cross_fitting=CrossFitting(n_folds=5)).cross_fitting.n_folds
    5
    """

    _collaborative: ClassVar[bool] = True
    _ratio_refusal: ClassVar[Refusal | None] = Refusal(
        RefusalReason.INVALID_DECLARATION,
        "ratio='classifier' sets how a modified treatment policy's density ratio is "
        "estimated, and collaborative TMLE fits no policies=: its collaborative score is "
        "derived for arm means.",
        alternative="Drop ratio=, or fit ModifiedTreatmentPolicy with TMLEMethod.",
    )

    strategy: str = "greedy"
    preorder: str | None = None
    ordering: tuple[str, ...] | None = None
    candidates: tuple[tuple[str, ...], ...] | None = None
    selection_folds: int = 5
    selection_inner_folds: int = 2
    loss: str = "auto"
    penalty: bool = True
    selection_estimand: str = "ate"
    oat_design: str | None = None
    lambdas: tuple[float, ...] | None = None
    treatment_candidates: tuple[Any, ...] | None = None
    selection_split: tuple[int, ...] | None = None
    patience: int | None = None
    name: str = "collaborative_tmle"

    def __post_init__(self) -> None:
        super().__post_init__()
        from .estimators.ctmle import PropensityArrays, _is_array_like

        for name in ("lambdas", "treatment_candidates", "selection_split"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, tuple):
                object.__setattr__(self, name, tuple(value))
        # A lone array entry passes through, so the engine names the ctmleGeneral pair.
        for index, entry in enumerate(self.treatment_candidates or ()):
            if not isinstance(entry, PropensityArrays) and not _is_array_like(entry):
                _validate_learner(entry, f"treatment_candidates[{index}]")

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
            raise MethodConfigurationError(
                "collaborative TMLE has no longitudinal derivation.",
                reason=RefusalReason.IMPLEMENTATION_GAP,
                alternative="Fit a longitudinal design with method='tmle' (LTMLE) or 'sdr'.",
                roadmap="F5",
            )
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
            "oat_design": self.oat_design,
            "lambdas": self.lambdas,
            "treatment_candidates": self.treatment_candidates,
            "selection_split": self.selection_split,
            "patience": self.patience,
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
        The extra score equations to solve, named by the nuisance whose misspecification each
        one corrects: ``"Q"`` adds equation (9) and ``"g"`` adds equation (10).
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
        Whether treatment was randomized, for a fit with missing outcomes.  ``False`` fits an
        observational missing outcome with the composite indicator.  Known treatment
        probabilities belong to the design:
        ``PointTreatment(treatment_probabilities=...)``.

    See Also
    --------
    TMLEMethod : The base configuration, without the reduced-dimension correction.
    CollaborativeTMLEMethod : A treatment mechanism selected for the targeted estimate.
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

    _ratio_refusal: ClassVar[Refusal | None] = Refusal(
        RefusalReason.INVALID_DECLARATION,
        "ratio='classifier' sets how a modified treatment policy's density ratio is "
        "estimated, and DR-TMLE fits no policies=: its reduced-dimension regressions are "
        "derived for counterfactual means under static treatment.",
        alternative="Drop ratio=, or fit ModifiedTreatmentPolicy with TMLEMethod.",
    )

    guard: tuple[str, ...] = ("Q", "g")
    reduction: str = "univariate"
    reduced_outcome_learner: Any = None
    reduced_treatment_learner: Any = None
    reduced_crossfit: str = "pooled"
    update_order: str = "drtmle"
    evaluation: Any = None
    randomized: bool = False
    name: str = "drtmle"

    def __post_init__(self) -> None:
        super().__post_init__()
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
            raise MethodConfigurationError(
                "DR-TMLE has no longitudinal derivation.",
                reason=RefusalReason.IMPLEMENTATION_GAP,
                alternative=(
                    "Fit a longitudinal design with method='sdr' (sequentially doubly robust)."
                ),
                roadmap="F5",
            )
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


@dataclass(frozen=True)
class AdaptiveIPTWMethod(TMLEMethod):
    """Configure the adaptive-propensity IPTW estimator.

    The estimator reports the Horvitz-Thompson mean of each arm with a propensity that is
    targeted along the reduced regression of the outcome on the fitted propensity (van der
    Laan 2012, WP302, Theorem 1). It fits no outcome regression, so the fluctuation settings
    of :class:`Targeting` and the ``ratio`` and ``density_bins`` settings of :class:`ModelSpec`
    stay at their defaults. An interval needs a known treatment mechanism. Otherwise the
    fit reports point estimates under the status ``estimated_mechanism_plugin``.

    Parameters
    ----------
    models : ModelSpec
        Nuisance learners and related model settings.
    cross_fitting : CrossFitting
        Sample-splitting settings.
    targeting : Targeting
        Mechanism-tilt and solver settings.
    inference : Inference
        Interval and resampling settings.
    runtime : Runtime
        Seed, run identifier, and parallelism.
    name : str
        Stable method name.
    form : {"targeted", "one_step"}, default="targeted"
        The reported form. ``"targeted"`` is the Horvitz-Thompson mean at the tilted
        propensity. ``"one_step"`` is the estimating-equation correction at the initial
        nuisances. A fit computes both and records the other.
    reduced_outcome_learner : estimator or None, default=None
        Learner for the reduced regression of the outcome on the fitted propensity. ``None``
        takes ``models.outcome_learner``, and then the default Super Learner.

    See Also
    --------
    TMLEMethod : Fits an outcome regression and a fluctuation of it.
    DRTMLEMethod : Corrects both nuisances with reduced-dimension regressions.
    ModelSpec : Owns the treatment learner the propensity is fitted with.

    Examples
    --------
    >>> from cleverly import AdaptiveIPTWMethod
    >>> method = AdaptiveIPTWMethod(form="one_step")
    >>> method.name, method.form
    ('adaptive_iptw', 'one_step')
    >>> sorted(method.estimator_kwargs())[:3]
    ['alpha', 'alpha_sig', 'bootstrap_resampling']
    """

    _ratio_refusal: ClassVar[Refusal | None] = Refusal(
        RefusalReason.INVALID_DECLARATION,
        "AdaptiveIPTW fits no outcome regression, so ratio= does not apply. Remove it.",
    )
    _density_bins_refusal: ClassVar[Refusal | None] = Refusal(
        RefusalReason.INVALID_DECLARATION,
        "AdaptiveIPTW fits no outcome regression, so density_bins= does not apply. Remove it.",
    )

    form: str = "targeted"
    reduced_outcome_learner: Any = None
    name: str = "adaptive_iptw"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.form not in ("targeted", "one_step"):
            raise ValueError(f"form must be 'targeted' or 'one_step'; got {self.form!r}")
        _validate_learner(self.reduced_outcome_learner, "reduced_outcome_learner")

    def estimator_kwargs(self, *, longitudinal: bool = False) -> dict[str, Any]:
        """Translate this configuration for the point-treatment adaptive IPTW engine.

        Parameters
        ----------
        longitudinal : bool
            Must be false because adaptive IPTW has no longitudinal derivation.

        Returns
        -------
        dict of str to Any
            Engine keyword arguments.

        Raises
        ------
        MethodConfigurationError
            When ``longitudinal`` is true.
        """
        if longitudinal:
            raise MethodConfigurationError.from_refusal(ADAPTIVE_IPTW_LONGITUDINAL_REFUSAL)
        return {
            **super().estimator_kwargs(),
            "form": self.form,
            "reduced_outcome_learner": self.reduced_outcome_learner,
        }


@dataclass(frozen=True)
class SDRMethod(TMLEMethod):
    """Configure the sequentially doubly robust longitudinal estimator.

    The estimator of Díaz, Williams, Hoffman and Schenck (2023, Section 5.3), fitted by
    :class:`cleverly.longitudinal.SDR`.  It runs the sequential regression of a longitudinal
    fit and fits no fluctuation, so the fluctuation settings of :class:`Targeting` stay at
    their defaults.  ``g_bounds`` bounds each partial path probability.  A point-treatment
    functional refuses the method.

    Parameters
    ----------
    models : ModelSpec
        Nuisance learners.  ``pseudo_learner`` fits a real-valued target, and its default is
        a regression Super Learner with no clip.
    cross_fitting : CrossFitting
        Sample-splitting settings.  Above one fold, each fold fits every nuisance on its
        training rows.
    targeting : Targeting
        ``g_bounds`` and ``q_bounds``.  ``submodel_alpha``, ``max_iter``, ``tol``,
        ``stopping``, ``step_bound``, ``step_search`` and ``iterate`` are refused, because the
        estimator fits no fluctuation.
    inference : Inference
        Interval and resampling settings.  The bootstrap runs and reports a bootstrap sd,
        with no percentile inference.
    runtime : Runtime
        Seed, run identifier, and parallelism.
    name : str
        Stable method name.

    See Also
    --------
    TMLEMethod : The targeted estimator of the same longitudinal parameters.
    cleverly.longitudinal.SDR : The engine this configuration is translated for.
    cleverly.LongitudinalTreatment : The design whose functionals take this method.

    Examples
    --------
    >>> from cleverly import CrossFitting, SDRMethod
    >>> method = SDRMethod(cross_fitting=CrossFitting(n_folds=5))
    >>> method.name, method.cross_fitting.n_folds
    ('sdr', 5)
    >>> "alpha" in method.estimator_kwargs(longitudinal=True)
    False
    """

    name: str = "sdr"

    def estimator_kwargs(self, *, longitudinal: bool = False) -> dict[str, Any]:
        """Translate this configuration for the longitudinal SDR engine.

        Parameters
        ----------
        longitudinal : bool
            Must be true, because the estimator is a longitudinal sequential regression.

        Returns
        -------
        dict of str to Any
            SDR engine keyword arguments.

        Raises
        ------
        MethodConfigurationError
            When ``longitudinal`` is false, or when a fluctuation setting differs from its
            default.
        """
        if not longitudinal:
            raise MethodConfigurationError.from_refusal(SDR_POINT_REFUSAL)
        defaults = Targeting()
        for field_name, engine_name in (
            ("submodel_alpha", "alpha"),
            ("max_iter", "max_iter"),
            ("tol", "tol"),
            ("stopping", "stopping"),
            ("step_bound", "step_bound"),
            ("step_search", "step_search"),
            ("iterate", "iterate"),
        ):
            if _differs_from_default(
                getattr(self.targeting, field_name), getattr(defaults, field_name)
            ):
                raise MethodConfigurationError.from_refusal(sdr_no_fluctuation_refusal(engine_name))
        _refuse_node_bounds(self, "the SDR estimator")
        _refuse_curve_settings(self, "SDR")
        kwargs = super().estimator_kwargs(longitudinal=True)
        for engine_name in (
            "alpha",
            "max_iter",
            "tol",
            "stopping",
            "step_bound",
            "step_search",
            "iterate",
            "survival_model",
            "targeting",
            "step_size",
            "isotonic",
            "node_bounds",
        ):
            del kwargs[engine_name]
        return kwargs


#: The engine each baseline method configures, as the curve-setting refusal names it.
_BASELINE_ENGINES = {
    "gcomp": "SequentialGComputation",
    "ipw": "SequentialIPW",
    "parametric_gformula": "ParametricGFormula",
}


@dataclass(frozen=True)
class _BaselineMethod(TMLEMethod):
    """The settings the two sequential baseline methods share.

    The baselines fit in sample, so ``cross_fitting`` defaults to ``enabled=False``.  They fit
    no fluctuation, so the fluctuation settings of :class:`Targeting` stay at their defaults.
    """

    #: Why this baseline refuses a point-treatment functional.
    _point_refusal: ClassVar[Refusal] = BASELINE_POINT_REFUSAL

    cross_fitting: CrossFitting = CrossFitting(enabled=False)

    def estimator_kwargs(self, *, longitudinal: bool = False) -> dict[str, Any]:
        """Translate this configuration for a longitudinal baseline engine.

        Parameters
        ----------
        longitudinal : bool
            Must be true, because the baselines are longitudinal estimators.

        Returns
        -------
        dict of str to Any
            Engine keyword arguments.

        Raises
        ------
        MethodConfigurationError
            When ``longitudinal`` is false, when ``ratio_trim`` or ``hazard_pooling`` is set,
            when a fluctuation setting differs from its default, or when a survival-curve
            setting of the targeted fit is set.
        """
        if not longitudinal:
            raise MethodConfigurationError.from_refusal(self._point_refusal)
        if self.targeting.ratio_trim is not None:
            raise MethodConfigurationError.from_refusal(BASELINE_RATIO_TRIM_REFUSAL)
        if self.models.hazard_pooling != "node":
            raise MethodConfigurationError.from_refusal(BASELINE_HAZARD_POOLING_REFUSAL)
        defaults = Targeting()
        for field_name, engine_name in (
            ("submodel_alpha", "alpha"),
            ("max_iter", "max_iter"),
            ("tol", "tol"),
            ("stopping", "stopping"),
            ("step_bound", "step_bound"),
            ("step_search", "step_search"),
            ("iterate", "iterate"),
        ):
            if _differs_from_default(
                getattr(self.targeting, field_name), getattr(defaults, field_name)
            ):
                raise MethodConfigurationError.from_refusal(
                    baseline_no_fluctuation_refusal(engine_name)
                )
        _refuse_node_bounds(self, "the sequential g-computation and IPW estimators")
        _refuse_curve_settings(self, _BASELINE_ENGINES[self.name])
        kwargs = super().estimator_kwargs(longitudinal=True)
        for engine_name in (
            "alpha",
            "max_iter",
            "tol",
            "stopping",
            "step_bound",
            "step_search",
            "iterate",
            "survival_model",
            "targeting",
            "step_size",
            "isotonic",
            "node_bounds",
        ):
            del kwargs[engine_name]
        return kwargs


@dataclass(frozen=True)
class GComputationMethod(_BaselineMethod):
    """Configure sequential g-computation of a longitudinal regimen.

    The iterated conditional expectation of Bang and Robins (2005, Section 4), fitted by
    :class:`cleverly.longitudinal.SequentialGComputation`.  The method fits in sample, so
    ``cross_fitting`` defaults to ``CrossFitting(enabled=False)``, and above one fold the
    engine refuses.  It fits no fluctuation, so the fluctuation settings of
    :class:`Targeting` stay at their defaults.  Every learner left at ``None`` is a
    main-terms GLM.  A point-treatment functional, a working model over regimens, and an
    odds tilt with an estimated mechanism refuse the method.

    Parameters
    ----------
    models : ModelSpec
        Nuisance learners.  ``outcome_learner`` fits each backward regression.
    cross_fitting : CrossFitting
        Sample-splitting settings.  The default is in sample, and the engine refuses more
        than one fold.
    targeting : Targeting
        ``g_bounds`` and ``q_bounds``.  ``g_bounds`` moves the positivity diagnostics only.
        ``submodel_alpha``, ``max_iter``, ``tol``, ``stopping``, ``step_bound``,
        ``step_search`` and ``iterate`` are refused, because the estimator fits no
        fluctuation.
    inference : Inference
        Interval and resampling settings.
    runtime : Runtime
        Seed, run identifier, and parallelism.
    name : str
        Stable method name.

    See Also
    --------
    IPWMethod : The inverse-probability-weighted baseline.
    SDRMethod : The sequentially doubly robust estimator of the same parameters.
    cleverly.longitudinal.SequentialGComputation : The engine this configuration is translated for.
    cleverly.LongitudinalTreatment : The design whose functionals take this method.

    Examples
    --------
    >>> from cleverly import GComputationMethod
    >>> method = GComputationMethod()
    >>> method.name, method.cross_fitting.enabled
    ('gcomp', False)
    >>> kwargs = method.estimator_kwargs(longitudinal=True)
    >>> kwargs["n_folds"], "alpha" in kwargs
    (1, False)
    """

    name: str = "gcomp"


@dataclass(frozen=True)
class IPWMethod(_BaselineMethod):
    """Configure sequential inverse probability weighting of a longitudinal regimen.

    The estimator fitted by :class:`cleverly.longitudinal.SequentialIPW`.  It reads the
    treatment and censoring mechanisms only, and fits no outcome regression.  The method
    fits in sample, so ``cross_fitting`` defaults to ``CrossFitting(enabled=False)``.  It
    fits no fluctuation, so the fluctuation settings of :class:`Targeting` stay at their
    defaults.  A point-treatment functional, a working model over regimens, a modified
    treatment policy, and an odds tilt with an estimated mechanism refuse the method.

    Parameters
    ----------
    models : ModelSpec
        Nuisance learners.  ``treatment_learner`` and ``censoring_learner`` fit the
        mechanisms.  ``outcome_learner`` and ``pseudo_learner`` are refused.
    cross_fitting : CrossFitting
        Sample-splitting settings.  The default is in sample, and the engine refuses more
        than one fold.
    targeting : Targeting
        ``g_bounds`` and ``q_bounds``.  ``submodel_alpha``, ``max_iter``, ``tol``,
        ``stopping``, ``step_bound``, ``step_search`` and ``iterate`` are refused, because
        the estimator fits no fluctuation.
    inference : Inference
        Interval and resampling settings.
    runtime : Runtime
        Seed, run identifier, and parallelism.
    name : str
        Stable method name.
    normalization : {"hajek", "horvitz_thompson"}, default="hajek"
        ``"hajek"`` divides the weighted outcome sum by the weight sum.
        ``"horvitz_thompson"`` divides by ``n``.

    See Also
    --------
    GComputationMethod : The outcome-regression baseline.
    SDRMethod : The sequentially doubly robust estimator of the same parameters.
    cleverly.longitudinal.SequentialIPW : The engine this configuration is translated for.
    cleverly.LongitudinalTreatment : The design whose functionals take this method.

    Examples
    --------
    >>> from cleverly import IPWMethod
    >>> method = IPWMethod(normalization="horvitz_thompson")
    >>> method.name, method.cross_fitting.enabled
    ('ipw', False)
    >>> method.estimator_kwargs(longitudinal=True)["normalization"]
    'horvitz_thompson'
    """

    _point_refusal: ClassVar[Refusal] = IPW_POINT_REFUSAL

    name: str = "ipw"
    normalization: Literal["hajek", "horvitz_thompson"] = "hajek"

    def estimator_kwargs(self, *, longitudinal: bool = False) -> dict[str, Any]:
        """Translate this configuration for the longitudinal IPW engine.

        Parameters
        ----------
        longitudinal : bool
            Must be true, because the estimator is longitudinal.

        Returns
        -------
        dict of str to Any
            IPW engine keyword arguments, with ``normalization``.

        Raises
        ------
        MethodConfigurationError
            When ``longitudinal`` is false, or when a fluctuation setting differs from its
            default.
        """
        kwargs = super().estimator_kwargs(longitudinal=longitudinal)
        kwargs["normalization"] = self.normalization
        return kwargs


@dataclass(frozen=True)
class ParametricGFormulaMethod(_BaselineMethod):
    """Configure the parametric g-formula of a longitudinal regimen.

    Forward simulation of fitted covariate, outcome and hazard laws (Robins 1986; McGrath et
    al. 2020), fitted by :class:`cleverly.longitudinal.ParametricGFormula`.  The method fits in
    sample, so ``cross_fitting`` defaults to ``CrossFitting(enabled=False)``, and above one fold
    the engine refuses.  It fits no fluctuation, so the fluctuation settings of
    :class:`Targeting` stay at their defaults.  A point-treatment functional and a working model
    over regimens refuse the method.

    Parameters
    ----------
    models : ModelSpec
        Nuisance learners.  ``outcome_learner`` fits the outcome model, and
        ``treatment_learner`` fits the natural-treatment models.  ``pseudo_learner`` is refused.
    cross_fitting : CrossFitting
        Sample-splitting settings.  The default is in sample, and the engine refuses more
        than one fold.
    targeting : Targeting
        ``g_bounds`` and ``q_bounds``.  ``g_bounds`` moves the positivity diagnostics only.
    inference : Inference
        Interval and resampling settings.
    runtime : Runtime
        Seed, run identifier, and parallelism.
    name : str
        Stable method name.
    n_draws : int
        The simulated histories per unit behind the estimate.
    n_derivative_draws : int or None
        The simulated histories per unit behind the derivatives.  ``None`` takes the larger of
        ``n_draws`` and ten.
    covariate_types : mapping or None
        The type of each time-varying column at nodes ``2..T``.
    covariate_learner : estimator or None
        The template of every covariate model.  ``None`` is a main-terms GLM.
    hazard_learner : estimator or None
        The template of every hazard model of a survival outcome.  ``None`` is a GLM.

    See Also
    --------
    GComputationMethod : The iterated conditional expectation form of the same parameter.
    cleverly.longitudinal.ParametricGFormula : The engine this configuration is translated for.
    cleverly.LongitudinalTreatment : The design whose functionals take this method.

    Examples
    --------
    >>> from cleverly import ParametricGFormulaMethod
    >>> method = ParametricGFormulaMethod(n_draws=20)
    >>> method.name, method.cross_fitting.enabled
    ('parametric_gformula', False)
    >>> kwargs = method.estimator_kwargs(longitudinal=True)
    >>> kwargs["n_draws"], kwargs["n_folds"], "alpha" in kwargs
    (20, 1, False)
    """

    name: str = "parametric_gformula"
    n_draws: int = 10
    n_derivative_draws: int | None = None
    covariate_types: Any = None
    covariate_learner: Any = None
    hazard_learner: Any = None

    def __post_init__(self) -> None:
        _validate_learner(self.covariate_learner, "covariate_learner")
        _validate_learner(self.hazard_learner, "hazard_learner")

    def estimator_kwargs(self, *, longitudinal: bool = False) -> dict[str, Any]:
        """Translate this configuration for the longitudinal g-formula engine.

        Parameters
        ----------
        longitudinal : bool
            Must be true, because the estimator is longitudinal.

        Returns
        -------
        dict of str to Any
            Engine keyword arguments, with the simulation settings.

        Raises
        ------
        MethodConfigurationError
            When ``longitudinal`` is false, or when a fluctuation setting differs from its
            default.
        """
        kwargs = super().estimator_kwargs(longitudinal=longitudinal)
        kwargs.update(
            n_draws=self.n_draws,
            n_derivative_draws=self.n_derivative_draws,
            covariate_types=self.covariate_types,
            covariate_learner=self.covariate_learner,
            hazard_learner=self.hazard_learner,
        )
        return kwargs


@dataclass(frozen=True)
class IPSIMethod(TMLEMethod):
    """Configure Kennedy's estimator of incremental propensity-score interventions.

    The estimator of Kennedy (2019, Algorithm 1), with the dropout weights of Kim, Kennedy
    and Naimi (2021), fitted by :class:`cleverly.longitudinal.IPSI`.  It averages the
    influence function at the fitted nuisances and fits no fluctuation, so the fluctuation
    settings of :class:`Targeting` stay at their defaults.  ``g_bounds`` bounds the
    cumulative censoring factor.  The tilt factor is never bounded.  A point-treatment
    functional refuses the method.

    Parameters
    ----------
    models : ModelSpec
        Nuisance learners.  ``pseudo_learner`` fits a real-valued target, and its default is
        a regression Super Learner with no clip.
    cross_fitting : CrossFitting
        Sample-splitting settings.  Above one fold, each fold fits every nuisance on its
        training rows.
    targeting : Targeting
        ``g_bounds`` and ``q_bounds``.  ``submodel_alpha``, ``max_iter``, ``tol``,
        ``stopping``, ``step_bound``, ``step_search`` and ``iterate`` are refused, because the
        estimator fits no fluctuation.
    inference : Inference
        Interval and resampling settings.  The bootstrap runs and reports a bootstrap sd,
        with no percentile inference.
    runtime : Runtime
        Seed, run identifier, and parallelism.
    name : str
        Stable method name.

    See Also
    --------
    SDRMethod : The sequential regression of static, dynamic and known-policy plans.
    cleverly.longitudinal.IPSI : The engine this configuration is translated for.
    cleverly.IncrementalEffect : The functional that takes this method.

    Examples
    --------
    >>> from cleverly import CrossFitting, IPSIMethod
    >>> method = IPSIMethod(cross_fitting=CrossFitting(n_folds=5))
    >>> method.name, method.cross_fitting.n_folds
    ('ipsi', 5)
    >>> "alpha" in method.estimator_kwargs(longitudinal=True)
    False
    """

    _ratio_refusal: ClassVar[Refusal | None] = Refusal(
        RefusalReason.INVALID_DECLARATION,
        "ratio='classifier' sets how a modified treatment policy's density ratio is "
        "estimated, and IPSI estimates none: an odds tilt's ratio is a closed form in the "
        "fitted propensity of a binary node.",
        alternative="Drop ratio=, or fit the policy with TMLEMethod or SDRMethod.",
    )
    _density_bins_refusal: ClassVar[Refusal | None] = Refusal(
        RefusalReason.INVALID_DECLARATION,
        "density_bins= sets the bins of a continuous node's conditional density, and IPSI "
        "fits no density: it tilts the odds of a binary node.",
        alternative="Drop density_bins=, or fit the continuous node with TMLEMethod or SDRMethod.",
    )

    name: str = "ipsi"

    def estimator_kwargs(self, *, longitudinal: bool = False) -> dict[str, Any]:
        """Translate this configuration for the longitudinal IPSI engine.

        Parameters
        ----------
        longitudinal : bool
            Must be true, because the estimator fits a longitudinal panel.

        Returns
        -------
        dict of str to Any
            IPSI engine keyword arguments.

        Raises
        ------
        MethodConfigurationError
            When ``longitudinal`` is false, when a fluctuation setting differs from its
            default, or when ``ratio_trim`` is set.
        """
        if not longitudinal:
            raise MethodConfigurationError.from_refusal(IPSI_POINT_REFUSAL)
        defaults = Targeting()
        for field_name in (
            "submodel_alpha",
            "max_iter",
            "tol",
            "stopping",
            "step_bound",
            "step_search",
            "iterate",
        ):
            if _differs_from_default(
                getattr(self.targeting, field_name), getattr(defaults, field_name)
            ):
                raise MethodConfigurationError.from_refusal(IPSI_NO_FLUCTUATION)
        if self.targeting.ratio_trim is not None:
            raise MethodConfigurationError.from_refusal(IPSI_RATIO_TRIM_REFUSAL)
        _refuse_node_bounds(self, "IPSI")
        _refuse_curve_settings(self, "IPSI")
        kwargs = super().estimator_kwargs(longitudinal=True)
        for engine_name in (
            "alpha",
            "max_iter",
            "tol",
            "stopping",
            "step_bound",
            "step_search",
            "iterate",
            "survival_model",
            "targeting",
            "step_size",
            "isotonic",
            "node_bounds",
        ):
            del kwargs[engine_name]
        return kwargs
