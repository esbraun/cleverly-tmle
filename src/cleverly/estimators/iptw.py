r"""Adaptive-propensity IPTW: a Horvitz-Thompson mean with a propensity targeted for it.

For each treatment arm :math:`a`, the estimator reports the weighted Horvitz-Thompson mean

.. math::

    \hat\psi = P_n[w\, I\, T / g^*] \,/\, P_n[w],

where :math:`I` is an indicator, :math:`g^*` is the targeted mechanism and :math:`T` is the
outcome.  For a plain arm mean :math:`I = 1\{A = a\}` and :math:`T = Y`.  The mechanism
:math:`g` is tilted along the covariate :math:`H = Q_r / g`, with
:math:`Q_r = E(T \mid I = 1, g(W))` the reduced regression of the outcome on the fitted
mechanism, until :math:`P_n[w\, H (I - g^*)] = 0`.  The base result is van der Laan (2012),
*Statistical Inference when using Data Adaptive Estimators of Nuisance Parameters* (U.C.
Berkeley Division of Biostatistics Working Paper 302), Theorem 1, and the reference
implementation is ``drtmle::adaptive_iptw``.  The contract page is
``docs/technical-reference/adaptive-iptw.md``.

.. warning::

   **An interval needs a known mechanism.**  Theorem 1 needs
   :math:`P_0[(Q_0 - Q_{r,0})(g_0 - g_n)/g_0] = o_P(n^{-1/2})`.  That term is linear in
   :math:`g_n - g_0`.  It vanishes when the outcome regression is a function of the true
   :math:`g_0`, or when the fitted :math:`g_n` is a function of the true one.  No fit can
   check either.  So a fit reports a standard error, an interval and a p-value only when the
   whole mechanism of every reported key is known (complete data and a declared
   ``treatment_probabilities=``).  Every other fit reports point estimates and takes the
   status ``estimated_mechanism_plugin``.  The plug-in spread of the Theorem 1 curve remains
   under ``plugin_std_error`` as a diagnostic.

**Key kinds.**  A *key* is one targeting problem :math:`(I, g, T)`.

=================  ===========================================  ============================
kind               indicator :math:`I`                          outcome :math:`T`
=================  ===========================================  ============================
arm                :math:`1\{A = a\}`                           :math:`Y`
composite arm      :math:`\Delta_A \Delta 1\{A = a\}`           :math:`Y`
deterministic      :math:`1\{A = d(W)\}`                        :math:`Y`
regime
stochastic regime  :math:`1\{A = a\}`, one key per regime and   :math:`g^*(a \mid W) Y`
                   arm
natural course     :math:`\Delta`                               :math:`Y`
=================  ===========================================  ============================

Every key has its own tilt, reduced regression and score equation, and baseline strata add one
of each per stratum.  A stochastic regime sums its arm keys.

**Forms.**  ``form="targeted"`` (the default) reports :math:`P_n[w I T / g^*]/P_n[w]` at the
tilted mechanism.  ``form="one_step"`` reports the estimating-equation correction
:math:`P_n[w I T/g]/P_n[w] - P_n[w D_g]/P_n[w]` at the initial nuisances, with
:math:`D_g = (Q_r/g)(I - g)`.  A fit computes both and records the other under
``result.extra["adaptive_iptw"]``.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator, is_classifier

from .._inference_status import InferenceStatus, precedent_status
from .._support import Classification, Preflight
from .._typing import BoolArray, FloatArray, IntArray, Learner
from ..data.causal_data import CausalData
from ..exceptions import (
    CapabilityError,
    ConvergenceWarning,
    DataError,
    DataWarning,
    MethodConfigurationError,
    Refusal,
    RefusalReason,
)
from ..fluctuation._score import relative_score, score_columns, score_scale
from ..fluctuation.iterative import Fluctuation
from ..fluctuation.mechanism import apply_mechanism_tilt, solve_bounded_mechanism
from ..fluctuation.submodel import Submodel
from ..inference.influence import (
    ArmMean,
    ClusterReference,
    ParameterEstimate,
    make_estimate,
    stamp_inference,
)
from ..learners._fitting import Task, fit_learner
from ..learners.crossfit import Folds, missing_training_support
from ..learners.library import _validate_learner
from ..learners.super_learner import SuperLearnerDiagnostics
from ..targets import TARGETS, TargetContext, _off_axis_reason, parameter_stem, targets_for
from ..targets.base import READS_SINK, stratum_alias
from ..utils.bounds import OutcomeScaler, resolve_g_bounds
from ._nuisance import NuisanceEstimates, UnfittedOutcome, fit_fold_mechanisms
from ._option_support import point_option_conditions
from ._strata import (
    check_stratified_targets,
    embed_stratum_curve,
    stratum_probabilities,
)
from .base import MEAN_GROUP_ESTIMANDS, TMLEConfig, resolve_estimands
from .composite import composite_state, composite_view
from .reduced import _reduced_column
from .tmle import TMLE, _in_report_order, _is_natural_course

__all__ = [
    "FORMS",
    "OPERATION_REFUSALS",
    "AdaptiveIPTW",
    "AdaptiveIPTWFit",
    "iptw_missing_data_route",
]

#: The two reported forms, shipped form first.
FORMS = ("targeted", "one_step")

#: The construction a fit records under ``extra["missing_data"]``.
IptwRoute = Literal["complete", "composite", "missing_outcome"]

#: The key of the natural-course mean in the owner maps of :class:`AdaptiveIPTWFit`.
NATURAL_COURSE_KEY = "ey_obs"

#: Estimand stems that read the natural-course mean.
_NATURAL_STEMS = frozenset({"ey_obs", "par", "paf"})

#: The stems the range rule guards, with the domain each needs.
_RANGE_STEMS = {"rr": "positive", "or": "unit", "paf": "unit"}


def iptw_missing_data_route(data: CausalData, estimands: Sequence[str]) -> IptwRoute:
    """The construction an AdaptiveIPTW fit runs on ``data``.

    Any missing treatment or missing outcome takes the composite indicator.  The scalar
    natural-course request records ``"missing_outcome"``, as TMLE records the same request,
    because it uses no composite mechanism.  This function does not call
    :func:`~cleverly.estimators.composite.missing_data_route`, which names a
    guard-dependent route.

    Parameters
    ----------
    data : CausalData
        The prepared data.
    estimands : sequence of str
        The resolved targets.

    Returns
    -------
    str
        ``"complete"``, ``"composite"`` or ``"missing_outcome"``.
    """
    if data.has_missing_treatment:
        return "composite"
    if not data.has_missing_outcome:
        return "complete"
    if tuple(estimands) == (NATURAL_COURSE_KEY,):
        return "missing_outcome"
    return "composite"


# ---------------------------------------------------------------------------- refusals

#: Defaults of the inherited keywords, read from :class:`~cleverly.TMLE`'s signature so
#: that a default cannot drift from the refusal that compares with it.
_DEFAULTS: dict[str, Any] = {
    name: parameter.default
    for name, parameter in inspect.signature(TMLE.__init__).parameters.items()
    if parameter.default is not inspect.Parameter.empty
}

#: Keywords that act on an outcome fluctuation, which this estimator does not run.
_OUTCOME_KEYWORDS = (
    "fluctuation",
    "q_bounds",
    "alpha",
    "target_weights",
    "step_size",
    "targeting",
    "outcome_by_arm",
    "iterate",
    "stopping",
    "step_bound",
    "step_search",
    "ratio",
    "density_bins",
)

_TMLE_ALTERNATIVE = "Use TMLE, which estimates it."

R1 = Refusal(
    RefusalReason.API_REDIRECTION,
    "AdaptiveIPTW weights each row by the probability of its own treatment arm, so it needs an "
    "arm-coded treatment.",
    alternative="For a continuous dose, use TMLE with policies=[Shift(...)].",
)

R4 = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "AdaptiveIPTW and intermediate= are not combined. A controlled direct effect fixes an "
    "intermediate variable, and no weighting of AdaptiveIPTW is derived for that functional.",
    alternative="Fit method='tmle'.",
    roadmap="F5",
)

R8 = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "AdaptiveIPTW targets the propensity once on all rows, as drtmle does. "
    "targeting_scheme='fold' replaces that pooled update with one update per fold, and no "
    "reviewed result covers it.",
    alternative="Use targeting_scheme='pooled'.",
    roadmap="X61",
)

R9 = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "AdaptiveIPTW reports the pooled estimate. cv_evaluation=True evaluates each fold on its "
    "own, and no reviewed result covers that for the targeted propensity.",
    alternative="Use cv_evaluation=False.",
    roadmap="X61",
)

R10 = Refusal(
    RefusalReason.INFERENCE_GAP,
    "AdaptiveIPTW reports a Wald interval from its influence curve only when the treatment "
    "mechanism is known. No registered study measures a bootstrap of the targeted propensity, "
    "so n_bootstrap= is refused.",
    alternative="Declare the treatment probabilities (treatment_probabilities=) for a Wald "
    "interval, or fit tmle or drtmle for a bootstrap.",
    roadmap="X55",
)

R16 = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "AdaptiveIPTW fits one split. No registered study measures a repeat aggregate of the "
    "targeted propensity, so repeats= must be 1.",
    alternative="Use repeats=1.",
    roadmap="X81",
)

R17 = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "AdaptiveIPTW does not cross-fit a regime with missing data. No registered study cell "
    "measures that composition.",
    alternative="Pass cross_fit=False (CrossFitting(enabled=False) on a method). The in-sample "
    "fit reports point estimates.",
    roadmap="X81",
)

R21 = Refusal(
    RefusalReason.INVALID_DECLARATION,
    "A cross-fitted AdaptiveIPTW trains fold k's reduced regression on designs from models that "
    "left out fold k and the training row's own fold, so it needs at least three folds.",
    alternative="Pass n_folds=3 or more, or cross_fit=False.",
)

R19 = Refusal(
    RefusalReason.INVALID_DECLARATION,
    "A stochastic regime regresses the transformed outcome g*(a | W) Y, which is not binary, "
    "so the reduced regression needs a regressor. Pass a regressor as reduced_outcome_learner=.",
)

#: Why each operation that needs an outcome regression refuses an AdaptiveIPTW fit, by
#: operation. The operation functions and the assessment rows read the same record.
OPERATION_REFUSALS: dict[str, Refusal] = {
    "omitted_variable": Refusal(
        RefusalReason.API_REDIRECTION,
        "the omitted-variable bound needs the outcome regression's residual, and AdaptiveIPTW "
        "fits no outcome regression.",
        alternative=_TMLE_ALTERNATIVE,
    ),
    "validate": Refusal(
        RefusalReason.API_REDIRECTION,
        "validate() checks the outcome regression and the targeting of a TMLE fit, and "
        "AdaptiveIPTW fits no outcome regression.",
        alternative="Use result.diagnostics.nuisance_models() for the treatment mechanism.",
    ),
    "truncation_curve": Refusal(
        RefusalReason.IMPLEMENTATION_GAP,
        "truncation_curve() refits the targeting of a TMLE fit at each bound, and no "
        "retargeting of the IPTW propensity is implemented.",
        alternative="Refit AdaptiveIPTW with another g_bounds=, or fit method='tmle'.",
        roadmap="X81",
    ),
    "refute": Refusal(
        RefusalReason.IMPLEMENTATION_GAP,
        "refute() refits a TMLE under placebo and resampled data, and no refutation of "
        "AdaptiveIPTW is registered.",
        alternative=_TMLE_ALTERNATIVE,
        roadmap="X82",
    ),
    "simulated_confounding": Refusal(
        RefusalReason.IMPLEMENTATION_GAP,
        "simulated_confounding() refits or tilts the outcome regression, and AdaptiveIPTW "
        "fits no outcome regression.",
        alternative=_TMLE_ALTERNATIVE,
        roadmap="X82",
    ),
    "missingness_tilt": Refusal(
        RefusalReason.API_REDIRECTION,
        "missingness() and tipping_gamma() refit or tilt the outcome regression, and "
        "AdaptiveIPTW fits no outcome regression.",
        alternative=_TMLE_ALTERNATIVE,
    ),
}

ESTIMATED_MECHANISM_NOTE = (
    "AdaptiveIPTW reports a point estimate and no interval when a mechanism is estimated."
)


def _redirect(condition: str) -> Refusal:
    """An API redirection to TMLE that opens with ``condition``."""
    return Refusal(RefusalReason.API_REDIRECTION, condition, alternative=_TMLE_ALTERNATIVE)


def axis_refusal(axis: str) -> Refusal:
    """The R2b record: the keyword ``axis`` belongs to TMLE.

    Parameters
    ----------
    axis : str
        The keyword, such as ``"policies"``.

    Returns
    -------
    Refusal
        An :attr:`~cleverly._refusal.RefusalReason.API_REDIRECTION` record.
    """
    return Refusal(
        RefusalReason.API_REDIRECTION,
        "AdaptiveIPTW estimates arm means, the means of known regimes, the natural-course "
        f"mean, and the targets built from them. For {axis}, use TMLE, which estimates it.",
        alternative=_TMLE_ALTERNATIVE,
    )


def target_refusal(target: str) -> Refusal:
    """The R3 record: ``target`` belongs to TMLE.

    Parameters
    ----------
    target : str
        The estimand, such as ``"att"``.

    Returns
    -------
    Refusal
        An :attr:`~cleverly._refusal.RefusalReason.API_REDIRECTION` record.
    """
    return Refusal(
        RefusalReason.API_REDIRECTION,
        "AdaptiveIPTW reports arm means, the means of known regimes, the natural-course mean, "
        f"and the targets built from them. For {target}, use TMLE, which estimates it.",
        alternative=_TMLE_ALTERNATIVE,
    )


def keyword_refusal(keyword: str) -> Refusal:
    """The R13 record: ``keyword`` acts on an outcome fluctuation.

    Parameters
    ----------
    keyword : str
        The keyword, such as ``"q_bounds"``.

    Returns
    -------
    Refusal
        An :attr:`~cleverly._refusal.RefusalReason.INVALID_DECLARATION` record.
    """
    return Refusal(
        RefusalReason.INVALID_DECLARATION,
        f"AdaptiveIPTW fits no outcome regression, so {keyword}= does not apply. Remove it.",
    )


LEARNER_PAIR_REFUSAL = Refusal(
    RefusalReason.INVALID_DECLARATION,
    "outcome_learner= and reduced_outcome_learner= both name the one reduced regression. "
    "Pass one of them.",
)


def settings_refusal(setting: Callable[[str], Any]) -> tuple[type[Exception], Refusal] | None:
    """The first data-free refusal of a configuration, with the error class that raises it.

    The constructor asks it of its keywords and the preflight asks it of the attributes, so a
    modified estimator meets the same refusal before any learner.

    Parameters
    ----------
    setting : callable
        Maps a keyword name to its value.

    Returns
    -------
    tuple or None
        The error class and the refusal of the first rule that applies, or ``None``.
    """
    if setting("learned_rule") is not None:
        return CapabilityError, axis_refusal("learned_rule=")
    if setting("policies"):
        return CapabilityError, axis_refusal("policies=")
    if setting("incremental"):
        return CapabilityError, axis_refusal("incremental=")
    if setting("msm") is not None:
        return CapabilityError, target_refusal("an MSM")
    if setting("intermediate_learner") is not None:
        return CapabilityError, R4
    for keyword in _OUTCOME_KEYWORDS:
        if setting(keyword) != _DEFAULTS[keyword]:
            return MethodConfigurationError, keyword_refusal(keyword)
    if setting("reduced_outcome_learner") is not None and setting("outcome_learner") is not None:
        return MethodConfigurationError, LEARNER_PAIR_REFUSAL
    if setting("targeting_scheme") != _DEFAULTS["targeting_scheme"]:
        return MethodConfigurationError, R8
    if setting("cv_evaluation"):
        return MethodConfigurationError, R9
    if (
        setting("n_bootstrap")
        or setting("bootstrap_resampling") != _DEFAULTS["bootstrap_resampling"]
    ):
        return MethodConfigurationError, R10
    if setting("repeats") != 1:
        return MethodConfigurationError, R16
    return None


# ------------------------------------------------------------------- learner guard


class ConstantDesignGuard(BaseEstimator):  # numpydoc ignore=PR01,PR02
    """A learner that returns the weighted mean when its design takes one value.

    D7 of the plan: when a training design has one distinct value (a known constant mechanism,
    in sample or in a training fold), the reduced regression is the weighted mean of the
    outcome over the key's training rows, and the wrapped learner does not run.  Any other
    design goes to the wrapped learner unchanged.

    Parameters
    ----------
    learner : Learner
        The learner for a design with more than one value.
    task : {"classification", "regression"}
        What the learner predicts.
    """

    def __init__(self, learner: Learner, task: str = "regression") -> None:
        self.learner = learner
        self.task = task

    def fit(
        self,
        x: Any,
        y: Any,
        sample_weight: Any = None,
        groups: Any = None,
    ) -> ConstantDesignGuard:
        """Fit the weighted mean or the wrapped learner.

        Parameters
        ----------
        x : ndarray
            The design.
        y : ndarray
            The target.
        sample_weight : ndarray or None
            Observation weights.
        groups : ndarray or None
            Cluster codes for an inner split of the wrapped learner.

        Returns
        -------
        ConstantDesignGuard
            The fitted guard.
        """
        matrix = np.asarray(x, dtype=float)
        target = np.asarray(y, dtype=float).reshape(-1)
        self.constant_ = bool(matrix.size and np.all(matrix == matrix[:1]))
        if self.constant_:
            weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
            self.value_ = float(np.average(target, weights=weights))
            self.classes_ = np.array([0, 1])
            self.diagnostics_: SuperLearnerDiagnostics | None = None
            return self
        self.learner_ = fit_learner(
            self.learner,
            matrix,
            y,
            None if sample_weight is None else np.asarray(sample_weight, dtype=float),
            groups=groups,
            warn_unweighted=False,
        )
        self.classes_ = np.asarray(getattr(self.learner_, "classes_", [0, 1]))
        self.diagnostics_ = getattr(self.learner_, "diagnostics_", None)
        return self

    def predict(self, x: Any) -> FloatArray:
        """Predict the conditional mean.

        Parameters
        ----------
        x : ndarray
            The design.

        Returns
        -------
        ndarray
            The mean, ``(n,)``.
        """
        matrix = np.asarray(x, dtype=float)
        if self.constant_:
            return np.full(matrix.shape[0], self.value_, dtype=float)
        return np.asarray(self.learner_.predict(matrix), dtype=float).reshape(-1)

    def predict_proba(self, x: Any) -> FloatArray:
        """Predict class probabilities for a binary target.

        Parameters
        ----------
        x : ndarray
            The design.

        Returns
        -------
        ndarray
            ``(n, 2)`` probabilities of the classes 0 and 1.
        """
        matrix = np.asarray(x, dtype=float)
        if self.constant_:
            positive = np.full(matrix.shape[0], self.value_, dtype=float)
            return np.column_stack([1.0 - positive, positive])
        if hasattr(self.learner_, "predict_proba"):
            return np.asarray(self.learner_.predict_proba(matrix), dtype=float)
        positive = np.asarray(self.learner_.predict(matrix), dtype=float).reshape(-1)
        return np.column_stack([1.0 - positive, positive])


# ------------------------------------------------------------------------ key records


@dataclass(frozen=True)
class _Spec:
    """One targeting problem before its mechanism is read.

    Parameters
    ----------
    name : str
        The key's name.
    kind : str
        ``"arm"``, ``"regime"``, ``"regime_arm"`` or ``"natural_course"``.
    owner : float or None
        The code of the mean this key adds to, or ``None`` for the natural-course mean.
    owner_name : str
        The label of that mean.
    indicator : ndarray
        The indicator :math:`I`, ``(n,)``.
    target : ndarray
        The outcome :math:`T`, ``(n,)``. It is zero where the indicator is zero.
    arm_weights : ndarray or None
        ``(n, K)`` weights that turn the ``(n, K)`` mechanism into this key's mechanism, or
        ``None`` for the natural-course key.
    task : {"classification", "regression"}
        What the reduced regression predicts.
    """

    name: str
    kind: str
    owner: float | None
    owner_name: str
    indicator: FloatArray
    target: FloatArray
    arm_weights: FloatArray | None
    task: Task


@dataclass
class _Solved:
    """One key's targeting, in full."""

    spec: _Spec
    mechanism_initial: FloatArray
    mechanism: FloatArray
    regression_initial: FloatArray
    regression: FloatArray
    steps: list[FloatArray]
    score_initial: FloatArray
    score: FloatArray
    scale: FloatArray
    trace: list[float]
    covariate: FloatArray
    constant_design: bool
    diagnostics: list[SuperLearnerDiagnostics]
    failure: str | None
    bounds: tuple[float, float]


@dataclass(frozen=True)
class AdaptiveIPTWFit:
    """What an AdaptiveIPTW fit did, reported under ``result.extra["adaptive_iptw"]``.

    Parameters
    ----------
    form : str
        The reported form, ``"targeted"`` or ``"one_step"``.
    axis : str
        ``"arm"`` or ``"regime"``.
    arms : tuple of str
        The labels of the arms, or of the regimes on a regime fit.
    strata : tuple of str or None
        The stratum labels, or ``None`` without baseline strata.
    regimes : tuple of str or None
        The regime labels, or ``None`` on an arm fit.
    regime_route : dict of str to str
        The route of each regime: ``"S9"`` for a 0 or 1 density and ``"S11"`` for a
        stochastic one.
    keys : tuple of str
        The targeting problems, one per arm, regime, regime and arm, or natural course, and
        per stratum.
    untargeted : dict of str to float
        The Horvitz-Thompson mean at the initial mechanism, for each reported mean.
    one_step : dict of str to tuple of float
        The one-step estimate and the plug-in standard error of its curve.
    targeted : dict of str to tuple of float
        The targeted estimate and the plug-in standard error of its curve. On a fit whose
        mechanism is known, this equals the reported standard error.
    epsilon : dict of str to tuple of float
        The tilt step of each round, for each key.
    rounds : int
        The most rounds any key took.
    stop : str
        ``"tolerance"`` when every key met the relative score rule, and ``"cap"`` otherwise.
    score : dict of str to float
        :math:`P_n[w D_g]` at the final state, for each key.
    reduced_learner : str
        The representation of the reduced-regression learner.
    constant_design : tuple of str
        The keys whose design took one value in a training set, so the reduced regression
        was the weighted mean and no learner ran.
    bound_active : dict of str to int
        The rows of each key whose final mechanism sits at a bound of ``g_bounds``.
    missing_data : str
        The route of the fit.
    diagnostics : dict of str to list
        The Super Learner diagnostics of the last reduced-regression fit of each key.
    """

    form: str
    axis: str
    arms: tuple[str, ...]
    strata: tuple[str, ...] | None
    regimes: tuple[str, ...] | None
    regime_route: dict[str, str]
    keys: tuple[str, ...]
    untargeted: dict[str, float]
    one_step: dict[str, tuple[float, float]]
    targeted: dict[str, tuple[float, float]]
    epsilon: dict[str, tuple[float, ...]]
    rounds: int
    stop: str
    score: dict[str, float]
    reduced_learner: str
    constant_design: tuple[str, ...]
    bound_active: dict[str, int]
    missing_data: str
    diagnostics: dict[str, list[SuperLearnerDiagnostics]] = field(default_factory=dict)

    def describe(self) -> str:
        """Return the line that :meth:`~cleverly.TMLEResult.summary` prints for this fit.

        Returns
        -------
        str
            The form, the rounds, the stop reason and the route.
        """
        line = (
            f"AdaptiveIPTW: {self.form} form; {self.rounds} round(s) of targeting, stopped at "
            f"{self.stop}; {len(self.keys)} key(s)"
        )
        if self.missing_data != "complete":
            line += f"; missing data route {self.missing_data}"
        return line


@dataclass(frozen=True)
class IptwFluctuation(Fluctuation):
    """A :class:`~cleverly.fluctuation.Fluctuation` that also carries the AdaptiveIPTW record.

    Parameters
    ----------
    record : AdaptiveIPTWFit or None
        What the fit did.
    """

    record: AdaptiveIPTWFit | None = None


@dataclass
class _IptwContext(TargetContext):
    """A :class:`~cleverly.targets.TargetContext` whose means are already computed."""

    owner_means: dict[float, ArmMean] = field(default_factory=dict)

    @property
    def means(self) -> dict[float, ArmMean]:
        return self.owner_means

    @property
    def observed_mean(self) -> ArmMean:
        if self.natural_course is not None:
            return self.natural_course
        y = np.asarray(self.scaled, dtype=float)
        w = np.asarray(self.weights, dtype=float)
        psi = float(np.average(y, weights=w))
        return ArmMean(psi, w * (y - psi))


# ---------------------------------------------------------------- numerical helpers


def _everywhere(n: int) -> BoolArray:
    return np.ones(n, dtype=bool)


def _blocks(strata: IntArray | None, n: int, n_strata: int) -> FloatArray:
    """The ``(n, S)`` stratum indicators, or one column of ones without strata."""
    if strata is None:
        return np.ones((n, 1), dtype=float)
    return (np.asarray(strata)[:, None] == np.arange(n_strata)[None, :]).astype(float)


def _ht_terms(solved: _Solved, form: str) -> tuple[FloatArray, FloatArray, FloatArray]:
    """``(x, d, g)`` of one form: the Horvitz-Thompson term, its correction and the mechanism.

    The estimate is ``P[w x]/P[w] - P[w d]/P[w]`` for the one-step form and
    ``P[w x]/P[w]`` with the curve ``w (x - d - psi)/P[w]`` for the targeted form.
    """
    spec = solved.spec
    if form == "one_step":
        g, qr = solved.mechanism_initial, solved.regression_initial
    else:
        g, qr = solved.mechanism, solved.regression
    x = spec.indicator * spec.target / g
    d = (qr / g) * (spec.indicator - g)
    return x, d, g


def _owner_values(
    solved: Sequence[_Solved],
    form: str,
    weights: FloatArray,
    rows: IntArray | None,
) -> dict[float | None, tuple[float, FloatArray]]:
    """Each owner's mean and curve on ``rows``, summed over the owner's keys.

    ``form`` is ``"targeted"``, ``"one_step"`` or ``"untargeted"``. The curve is on the
    ``len(rows)`` scale, or on ``n`` without ``rows``.
    """
    out: dict[float | None, tuple[float, FloatArray]] = {}
    index = slice(None) if rows is None else rows
    w = np.asarray(weights, dtype=float)[index]
    total = float(np.sum(w))
    mean_w = total / w.size
    for item in solved:
        if form == "untargeted":
            x, d, _ = _ht_terms(item, "one_step")
            psi = float(np.sum(w * x[index]) / total)
            curve = np.zeros(w.size)
        else:
            x, d, _ = _ht_terms(item, form)
            x_rows, d_rows = x[index], d[index]
            psi = float(np.sum(w * x_rows) / total)
            if form == "one_step":
                psi -= float(np.sum(w * d_rows) / total)
            curve = w * (x_rows - d_rows - psi) / mean_w
        owner = item.spec.owner
        if owner in out:
            previous_psi, previous_curve = out[owner]
            out[owner] = (previous_psi + psi, previous_curve + curve)
        else:
            out[owner] = (psi, curve)
    return out


def _tilt_fold_slab(
    slab: FloatArray,
    covariates: FloatArray,
    epsilon: FloatArray,
    *,
    bounds: tuple[float, float],
    assignment: IntArray,
    production: FloatArray,
) -> FloatArray:
    r"""Move every fold's mechanism slab by one pooled fluctuation, each along its own covariate.

    ``slab`` is ``(K, n)``: entry ``k`` is the design fold ``k``'s reduced regression reads,
    fold ``k``'s own mechanism at its rows and a leave-two-folds-out mechanism elsewhere
    (:class:`~cleverly.estimators._nuisance.FoldMechanisms`).  ``covariates`` is
    ``(K, n, p)``: entry ``k`` is fold ``k``'s clever covariate :math:`H_k` at entry ``k``
    of the slab.  The one ``epsilon`` the pooled score solved moves entry ``k`` along
    :math:`H_k`, by :func:`~cleverly.fluctuation.mechanism.apply_mechanism_tilt` with
    ``bounds``.  Tilting entry ``k`` along the production covariate instead would carry the
    other folds' regressions, and so fold ``k``'s own rows, into fold ``k``'s next design.
    The first step already reads :math:`H_k`, from the round-0 regressions.

    Each row's own fold entry is then set to ``production``, the array the solver returned,
    so the score, the estimate and the slab read one mechanism at every own-fold entry.

    Parameters
    ----------
    slab : ndarray
        ``(K, n)``, each fold's design before the step.
    covariates : ndarray
        ``(K, n, p)``, each fold's covariate at its slab entry.
    epsilon : ndarray
        ``(p,)``, the pooled fluctuation.
    bounds : tuple of float
        The truncation of the key's mechanism.
    assignment : ndarray
        ``(n,)``, each row's outer fold.
    production : ndarray
        ``(n,)``, the tilted production mechanism.

    Returns
    -------
    ndarray
        ``(K, n)``, the slab after the step.
    """
    moved = np.stack(
        [
            apply_mechanism_tilt(entry, covariate, epsilon, bounds=bounds)
            for entry, covariate in zip(slab, covariates, strict=True)
        ]
    )
    rows = np.arange(moved.shape[1])
    moved[np.asarray(assignment), rows] = np.asarray(production, dtype=float)
    return moved


def refuse_untrainable_mechanism_pairs(
    data: CausalData, folds: Sequence[Folds], *, treatment: bool
) -> None:
    r"""Refuse a split in which two folds hold every trainable row of a class a mechanism needs.

    The fold-wise reduced regression refits each estimated mechanism on the rows outside
    folds ``k`` and ``m``, for every pair of outer folds. A refit that sees one class cannot
    be fitted by a classifier, or silently returns a constant design. The check runs after
    the folds are drawn and before any learner is fitted. A cluster is one unit, because a
    fold holds whole clusters.

    ============================ ==============================================
    estimated factor             rows outside the pair must hold
    ============================ ==============================================
    treatment                    every arm among the treatment-recorded rows
    outcome observation          both values of ``Delta`` among those rows
    treatment observation        both values of ``DeltaA``
    ============================ ==============================================

    This check is distinct from the DR-TMLE leave-two-out check, which reads arm, observed
    outcome and baseline stratum; the two may merge later.

    Parameters
    ----------
    data : CausalData
        The prepared data.
    folds : sequence of Folds
        The drawn splits, one per repeat.
    treatment : bool
        Whether the treatment mechanism is estimated (false for a known mechanism or the
        natural-course mean).

    Raises
    ------
    DataError
        If the rows outside some pair of folds lack a class of an estimated factor.
    """
    recorded = (
        np.asarray(data.treatment_recorded, dtype=bool)
        if data.has_missing_treatment
        else np.ones(data.n, dtype=bool)
    )
    observed = np.asarray(data.observed, dtype=bool)
    treated = np.asarray(data.treatment)
    # (factor, class label, rows the class is counted over, per-row class value)
    needs: list[tuple[str, str, BoolArray, np.ndarray]] = []
    if treatment and data.known_treatment is None:
        for arm in np.unique(treated[recorded]):
            needs.append(("treatment", f"arm {arm:g}", recorded, treated == arm))
    if data.has_missing_outcome:
        for value in (True, False):
            needs.append(
                ("outcome observation", f"Delta = {int(value)}", recorded, observed == value)
            )
    if data.has_missing_treatment:
        everywhere = np.ones(data.n, dtype=bool)
        for value in (True, False):
            needs.append(
                ("treatment observation", f"DeltaA = {int(value)}", everywhere, recorded == value)
            )
    if not needs:
        return
    for repeat, draw in enumerate(folds):
        if draw.is_single or draw.n_folds < 3:
            continue
        assignment = np.asarray(draw.assignment)
        for first in range(draw.n_folds):
            for second in range(first + 1, draw.n_folds):
                outside = (assignment != first) & (assignment != second)
                for factor, label, region, indicator in needs:
                    if not np.any(outside & region & indicator):
                        raise DataError(
                            f"cross-fitted AdaptiveIPTW refits the {factor} mechanism on the rows "
                            f"outside each pair of folds, and the rows outside folds {first} and "
                            f"{second} (repeat {repeat}) hold no row with {label}, so the "
                            "mechanism cannot be fitted there. Fewer folds do not help, because "
                            "the fit needs at least three. Supply more data, or pass "
                            "cross_fit=False to fit in sample."
                        )


class AdaptiveIPTW(TMLE):
    r"""Adaptive-propensity IPTW for a discrete treatment.

    Reports the Horvitz-Thompson mean of each treatment arm, with a propensity that is
    targeted along the reduced regression of the outcome on the fitted propensity (van der
    Laan 2012, WP302, Theorem 1; the reference implementation is ``drtmle::adaptive_iptw``),
    and the targets built from those means.  It fits no outcome regression.

    **A standard error, an interval and a p-value need a known mechanism.**  A fit whose
    mechanism is estimated reports point estimates and takes the status
    ``estimated_mechanism_plugin``, because Theorem 1 needs a term that is linear in the
    mechanism error to vanish, and no fit can check that.  The module docstring states the
    condition.

    Parameters
    ----------
    form : {"targeted", "one_step"}
        The reported form. ``"targeted"`` is the Horvitz-Thompson mean at the tilted
        propensity. ``"one_step"`` is the estimating-equation correction at the initial
        nuisances. A fit computes both and records the other.
    reduced_outcome_learner : Learner or None
        The learner of the reduced regression :math:`E(T \mid I = 1, g(W))`. ``None`` takes
        ``outcome_learner``, and then the default Super Learner. Pass a fixed-basis,
        one-dimensional learner under ``cross_fit=True``: a selecting learner is outside the
        cross-fitting conditions of the contract page.
    **kwargs : Any
        The keywords of :class:`~cleverly.TMLE`. The outcome-fluctuation keywords
        (``fluctuation``, ``q_bounds``, ``alpha``, ``target_weights``, ``step_size``,
        ``targeting``, ``outcome_by_arm``, ``iterate``, ``stopping``, ``step_bound``,
        ``step_search``, ``ratio`` and ``density_bins``) raise
        :class:`~cleverly.exceptions.MethodConfigurationError` at a non-default value.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.linear_model import LinearRegression, LogisticRegression
    >>> from cleverly.estimators import AdaptiveIPTW
    >>> rng = np.random.default_rng(0)
    >>> w = rng.normal(size=300)
    >>> a = rng.binomial(1, 1 / (1 + np.exp(-0.5 * w)))
    >>> y = 1.0 + 0.5 * a + 0.3 * w + rng.normal(size=300)
    >>> frame = pd.DataFrame({"y": y, "a": a, "w": w})
    >>> result = AdaptiveIPTW(
    ...     treatment_learner=LogisticRegression(),
    ...     reduced_outcome_learner=LinearRegression(),
    ...     estimands=["ey1", "ey0", "ate"],
    ...     n_folds=3,
    ...     random_state=1,
    ... ).fit(frame, outcome="y", treatment="a", covariates=["w"]).single()
    >>> result.inference_status
    'estimated_mechanism_plugin'
    >>> sorted(result.estimates)
    ['ate', 'ey0', 'ey1']
    """

    _assessment_method = "adaptive_iptw"
    #: The fit has no outcome regression for a simulated common cause to perturb.
    simulated_confounding_replay = False

    def __init__(
        self,
        *,
        form: str = "targeted",
        reduced_outcome_learner: Learner | None = None,
        **kwargs: Any,
    ) -> None:
        if form not in FORMS:
            raise ValueError(f"form must be 'targeted' or 'one_step'; got {form!r}")
        _validate_learner(reduced_outcome_learner, "reduced_outcome_learner")
        refused = settings_refusal(
            lambda name: (
                reduced_outcome_learner
                if name == "reduced_outcome_learner"
                else kwargs.get(name, _DEFAULTS.get(name))
            )
        )
        if refused is not None:
            raise refused[0].from_refusal(refused[1])  # type: ignore[attr-defined]
        super().__init__(**kwargs)
        self.form = form
        self.reduced_outcome_learner = reduced_outcome_learner

    # ------------------------------------------------------------------ hooks

    def _uses_corrections(self) -> bool:
        return False

    def _scaler(self, data: CausalData) -> OutcomeScaler:
        """The identity: a Horvitz-Thompson mean divides the outcome by nothing it scales."""
        del data
        return OutcomeScaler.identity()

    def _outcome_scale_refusal(self, data: CausalData) -> Refusal | None:
        del data
        return None

    def _missing_data_route(self, data: CausalData, estimands: tuple[str, ...]) -> str:
        return iptw_missing_data_route(data, estimands)

    @staticmethod
    def _closed_form_only(data: CausalData, names: Any) -> bool:
        """Whether the reported names are the complete-data ``ey_obs`` alone, which reads no key."""
        return (
            not data.has_missing_outcome
            and isinstance(names, (list, tuple))
            and len(names) > 0
            and set(names) == {NATURAL_COURSE_KEY}
        )

    def _has_known_keys(self, data: CausalData) -> bool:
        """Whether every key's whole mechanism is known (H2)."""
        return (
            data.known_treatment is not None
            and not data.has_missing_outcome
            and not data.has_missing_treatment
        )

    def _inference_status(
        self, data: CausalData, estimands: Sequence[str] | None = None
    ) -> InferenceStatus:
        """Withhold inference when any reported key has an estimated mechanism.

        A key has a known mechanism only on complete data that declares the treatment
        probabilities. A complete-data ``ey_obs`` alone reads no key. The names being
        reported and the prepared data decide it, and no fitted quantity. A fit and a
        ``retarget`` call pass the names they report, so a ``retarget`` to other names
        does not inherit the status of the constructor's list.

        Parameters
        ----------
        data : CausalData
            The prepared data.
        estimands : sequence of str, optional
            The names the call reports. The constructor's ``estimands`` when omitted.

        Returns
        -------
        str
            One of :data:`~cleverly.inference.influence.InferenceStatus`.
        """
        names: Any = self.estimands
        if estimands is not None:
            names = tuple(str(name) for name in estimands)
        if self._closed_form_only(data, names):
            return super()._inference_status(data)
        estimated = not self._has_known_keys(data)
        return precedent_status(
            [
                "estimated_mechanism_plugin" if estimated else "influence_curve",
                "estimated_weight_plugin" if data.declares_estimated_weights else "influence_curve",
                super()._inference_status(data),
            ]
        )

    # -------------------------------------------------------------- preflight

    def _check_iptw(self, data: CausalData) -> None:
        """The refusals that run before any learner, each by name."""
        refused = settings_refusal(lambda name: getattr(self, name, _DEFAULTS.get(name)))
        if refused is not None:
            raise refused[0].from_refusal(refused[1])  # type: ignore[attr-defined]
        if data.treatment_kind != "discrete":
            raise CapabilityError.from_refusal(R1)
        if data.has_intermediate:
            raise CapabilityError.from_refusal(R4)
        if (
            self.interventions
            and self.cross_fit
            and (data.has_missing_outcome or data.has_missing_treatment)
        ):
            raise CapabilityError.from_refusal(R17)
        if self.interventions:
            self._check_regime_learner(data)
        if self.cross_fit and int(self.n_folds) < 3:
            raise MethodConfigurationError.from_refusal(R21)

    def _check_regime_learner(self, data: CausalData) -> None:
        """R19: a classifier cannot regress the transformed outcome of a stochastic regime."""
        learners = [self.reduced_outcome_learner, self.outcome_learner]
        if not any(_is_classifier(learner) for learner in learners):
            return
        regimes = self._regimes(data, None)  # type: ignore[arg-type]
        assert regimes is not None
        for code in regimes.codes:
            if not np.all(np.isin(regimes.column(code), (0.0, 1.0))):
                raise MethodConfigurationError.from_refusal(R19)

    def _preflight_fit_configuration(self, data: CausalData) -> Preflight:
        """Run :meth:`_check_iptw` first, so a refusal precedes every shared lookup."""
        self._check_iptw(data)
        return super()._preflight_fit_configuration(data)

    def _resolved_names(self, data: CausalData, *, refuse_outside: bool) -> tuple[str, ...]:
        """The names the constructor's ``estimands`` resolves to on this data.

        The default list and ``"all"`` keep the mean-group names, and ``"all"`` on a missing
        outcome drops the natural-course stems. An explicit list that names a target outside
        the group is refused when ``refuse_outside`` holds.
        """
        estimands = resolve_estimands(self.estimands, data.family, data.n_arms, axis=self._axis)
        if self._axis == "arm":
            outside = [name for name in estimands if name not in MEAN_GROUP_ESTIMANDS]
            if self.estimands is None or self.estimands == "all":
                estimands = tuple(name for name in estimands if name in MEAN_GROUP_ESTIMANDS)
            elif outside and refuse_outside:
                raise CapabilityError.from_refusal(target_refusal(", ".join(outside)))
            if data.has_missing_outcome and self.estimands == "all":
                estimands = tuple(name for name in estimands if name not in _NATURAL_STEMS)
        return tuple(estimands)

    def _resolve_estimands_for_data(self, data: CausalData) -> Preflight:
        """Resolve the mean-group targets, or the regime targets, and refuse the rest.

        It keeps the fold-policy, undeclared-function and stratified-target checks of the
        inherited resolver and the group-A conditions of the point options. It does not
        inherit the natural-course and arm-indexed contracts, whose texts are TMLE's.
        """
        self._refuse_fold_policy()
        self._refuse_undeclared_functions()
        check_stratified_targets(data, incremental=False, msm=None)
        estimands = self._resolved_names(data, refuse_outside=True)
        conditions: list[Refusal] = list(point_option_conditions(self, data))
        return Preflight(estimands, tuple(conditions))

    def _preflight_training_support(
        self,
        data: CausalData,
        estimands: tuple[str, ...],
        folds: Sequence[Folds],
    ) -> None:
        """Check the shipped training support, then the rows each key trains on."""
        super()._preflight_training_support(data, estimands, folds)
        if any(draw.is_single for draw in folds):
            return
        refuse_untrainable_mechanism_pairs(
            data, folds, treatment=not _is_natural_course(data, estimands)
        )
        regimes = self._regimes(data, None)  # type: ignore[arg-type]
        for spec in self._specs(data, estimands, regimes):
            support: list[tuple[str, FloatArray | BoolArray, FloatArray | BoolArray]] = [
                (spec.name, spec.indicator > 0.0, np.array([True]))
            ]
            for repeat, draw in enumerate(folds):
                gap = missing_training_support(draw, support)
                if gap is not None:
                    fold, name, _ = gap
                    raise DataError(
                        f"cross-fitted AdaptiveIPTW needs rows of the key {name!r} in every "
                        f"training complement, and repeat {repeat}, fold {fold} has none, so "
                        "the reduced regression cannot be fitted there. Use fewer folds, or "
                        "pass cross_fit=False."
                    )

    # --------------------------------------------------------------- nuisances

    def _nuisances(
        self,
        data: CausalData,
        folds: Folds,
        scaler: OutcomeScaler,
        config: TMLEConfig,
        intermediate_value: float | None,
        seed: int | None = None,
    ) -> tuple[NuisanceEstimates, dict[str, Any]]:
        """Fit the treatment and missingness mechanisms, and no outcome regression."""
        # A complete-data ``ey_obs`` alone reads no mechanism, yet its treatment learner still
        # runs: ``retarget`` and the sensitivity sweeps read the cached nuisances for other
        # targets, and an unfitted mechanism would make them fail.
        natural_course = _is_natural_course(data, config.estimands)
        nuisance = self._fit_nuisances(
            data,
            folds,
            scaler,
            intermediate_value,
            seed=seed,
            fit_treatment=not natural_course,
            fit_outcome=False,
        )
        if folds.is_single:
            return nuisance, {}
        # The fold-wise reduced regression trains fold k on leave-two-folds-out designs
        # (F1 of the X21 run-1 diagnosis), fitted once here from the same resolved learners.
        observing = data.has_missing_outcome or data.has_missing_treatment
        slabs = fit_fold_mechanisms(
            data,
            nuisance,
            treatment_learner=(
                None
                if natural_course
                else self._resolve_learner(self.treatment_learner, task="classification", seed=seed)
            ),
            missingness_learner=(
                self._resolve_learner(
                    self.missingness_learner,
                    task="classification",
                    fallback=self.treatment_learner,
                    seed=seed,
                )
                if observing
                else None
            ),
            n_jobs=self.n_jobs,
        )
        return replace(nuisance, fold_mechanisms=slabs), {}

    # ------------------------------------------------------------------- specs

    def _specs(
        self,
        data: CausalData,
        requested: Sequence[str],
        regimes: Any,
    ) -> list[_Spec]:
        """The targeting problems of a fit, from the data alone."""
        stems = set(requested)
        route = iptw_missing_data_route(data, tuple(requested))
        view = composite_view(data) if route == "composite" else data
        treatment = np.asarray(view.treatment, dtype=float)
        n = treatment.shape[0]
        recorded = np.isfinite(treatment) & np.asarray(view.observed, dtype=bool)
        outcome = np.where(recorded, np.asarray(view.outcome, dtype=float), 0.0)
        task: Task = "classification" if data.family == "binomial" else "regression"
        specs: list[_Spec] = []
        if data.has_arms and stems - {NATURAL_COURSE_KEY} and route != "missing_outcome":
            arms = tuple(float(code) for code in data.arm_codes)
            arm_labels = {code: str(data.arm_label(code)) for code in arms}
            onehot = {code: (treatment == code).astype(float) for code in arms}
            if regimes is None:
                for column, code in enumerate(arms):
                    weights = np.zeros((n, len(arms)))
                    weights[:, column] = 1.0
                    specs.append(
                        _Spec(
                            arm_labels[code],
                            "arm",
                            code,
                            arm_labels[code],
                            onehot[code],
                            np.where(onehot[code] > 0, outcome, 0.0),
                            weights,
                            task,
                        )
                    )
            else:
                arm_index = np.full(n, -1, dtype=int)
                for column, code in enumerate(arms):
                    arm_index[treatment == code] = column
                rows = np.arange(n)
                for code in regimes.codes:
                    density = np.asarray(regimes.column(code), dtype=float)
                    label = str(regimes.label(code))
                    if np.all(np.isin(density, (0.0, 1.0))):
                        chosen = np.where(
                            arm_index >= 0, density[rows, np.clip(arm_index, 0, None)], 0.0
                        )
                        specs.append(
                            _Spec(
                                label,
                                "regime",
                                float(code),
                                label,
                                np.where(recorded, chosen, 0.0),
                                outcome,
                                density,
                                task,
                            )
                        )
                        continue
                    for column, arm in enumerate(arms):
                        if not np.any(density[:, column] != 0.0):
                            continue
                        weights = np.zeros((n, len(arms)))
                        weights[:, column] = 1.0
                        specs.append(
                            _Spec(
                                f"{label}/{arm_labels[arm]}",
                                "regime_arm",
                                float(code),
                                label,
                                onehot[arm],
                                np.where(onehot[arm] > 0, density[:, column] * outcome, 0.0),
                                weights,
                                "regression",
                            )
                        )
        if data.has_missing_outcome and stems & _NATURAL_STEMS:
            observed = np.asarray(data.observed, dtype=bool)
            specs.append(
                _Spec(
                    NATURAL_COURSE_KEY,
                    "natural_course",
                    None,
                    NATURAL_COURSE_KEY,
                    observed.astype(float),
                    np.where(observed, np.asarray(data.outcome, dtype=float), 0.0),
                    None,
                    task,
                )
            )
        return specs

    # ------------------------------------------------------------------ solver

    def _reduced_learner(self, task: Task) -> Learner:
        """The learner of the reduced regression, guarded for a constant design (D7)."""
        learner = self._resolve_learner(
            self.reduced_outcome_learner, task=task, fallback=self.outcome_learner
        )
        return ConstantDesignGuard(learner, task)

    def _solve_key(
        self,
        data: CausalData,
        nuisance: NuisanceEstimates,
        spec: _Spec,
        mechanism: FloatArray,
        bounds: tuple[float, float],
        learner: Learner,
        fold_mechanism: FloatArray | None = None,
    ) -> _Solved:
        """Target one key: tilt, refit the reduced regression on the tilt, repeat.

        ``fold_mechanism`` is the ``(K, n)`` slab of a cross-fitted fit: entry ``k`` is fold
        ``k``'s mechanism at its own rows and a leave-two-folds-out mechanism at every other
        row (:class:`~cleverly.estimators._nuisance.FoldMechanisms`).  The key then takes the
        fold-wise construction: fold ``k``'s reduced regression trains on its training rows
        at entry ``k`` and predicts at fold ``k``'s rows, the one ``epsilon`` is solved at
        each row's own fold, and :func:`_tilt_fold_slab` moves
        entry ``k`` along fold ``k``'s own covariate.  No row of fold ``k`` then reaches fold
        ``k``'s regression.  ``None`` is the in-sample loop.
        """
        n = data.n
        weights = np.asarray(data.weights, dtype=float)
        lower, upper = float(bounds[0]), float(bounds[1])
        indicator, target = spec.indicator, spec.target
        n_strata = data.n_strata if data.strata is not None else 1
        blocks = _blocks(data.strata, n, n_strata)
        fit_mask = indicator > 0.0
        diagnostics: list[SuperLearnerDiagnostics] = []
        constant = False
        foldwise = fold_mechanism is not None and not nuisance.folds.is_single
        own = np.asarray(nuisance.folds.assignment) if foldwise else None
        rows = np.arange(n)
        # The fold-wise state: each fold's mechanism and regression at every row.  The own-fold
        # entries are the production arrays, so the score and the estimate read exactly them.
        slab: FloatArray | None = None
        slab_regression: FloatArray | None = None

        def covariate(regression: FloatArray, g: FloatArray) -> FloatArray:
            return (regression / g)[:, None] * blocks

        def fit(design: FloatArray) -> FloatArray:
            nonlocal constant, slab_regression
            task = spec.task
            constant_folds: list[int] | None = [] if task == "classification" else None
            values, at_folds = _reduced_column(
                learner,
                design=design,
                target=target,
                training=None,
                companion=None,
                fit_mask=fit_mask,
                data=data,
                nuisance=nuisance,
                task=task,
                clip=(0.0, 1.0) if task == "classification" else None,
                n_jobs=self.n_jobs,
                diagnostics=diagnostics,
                constant_folds=constant_folds,
                fold_designs=None if slab is None else list(slab),
            )
            slab_regression = at_folds
            constant = constant or _constant_design(design, fit_mask, data, nuisance.folds, slab)
            return values

        everywhere = _everywhere(n)
        g = np.clip(np.asarray(mechanism, dtype=float), lower, upper)
        if foldwise:
            assert fold_mechanism is not None and own is not None
            slab = np.clip(np.asarray(fold_mechanism, dtype=float), lower, upper)
            slab[own, rows] = g
        g_initial = g.copy()
        regression = fit(g)
        regression_initial = regression
        h = covariate(regression, g)
        score = score_columns(indicator, g, h, weights, everywhere)
        scale = score_scale(h, weights, everywhere)
        score_initial = score
        trace = [relative_score(score, scale)]
        steps: list[FloatArray] = []
        failure: str | None = None
        rounds = 0
        while relative_score(score, scale) > self.tol and rounds < self.max_iter:
            fluctuation = solve_bounded_mechanism(indicator, g, h, weights, bounds=(lower, upper))
            g = np.asarray(fluctuation.propensity, dtype=float)
            steps.append(np.asarray(fluctuation.epsilon, dtype=float))
            if slab is not None:
                assert slab_regression is not None and own is not None
                slab = _tilt_fold_slab(
                    slab,
                    np.stack(
                        [
                            covariate(fold_q, fold_g)
                            for fold_g, fold_q in zip(slab, slab_regression, strict=True)
                        ]
                    ),
                    np.asarray(fluctuation.epsilon, dtype=float),
                    bounds=(lower, upper),
                    assignment=own,
                    production=g,
                )
            failure = fluctuation.failure
            regression = fit(g)
            h = covariate(regression, g)
            score = score_columns(indicator, g, h, weights, everywhere)
            scale = score_scale(h, weights, everywhere)
            trace.append(relative_score(score, scale))
            rounds += 1
        return _Solved(
            spec=spec,
            mechanism_initial=g_initial,
            mechanism=g,
            regression_initial=regression_initial,
            regression=regression,
            steps=steps,
            score_initial=score_initial,
            score=score,
            scale=scale,
            trace=trace,
            covariate=h,
            constant_design=constant,
            diagnostics=diagnostics,
            failure=failure,
            bounds=(lower, upper),
        )

    # ------------------------------------------------------------- retargeting

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
        classification: Classification | None = None,
    ) -> tuple[dict[str, ParameterEstimate], dict[str, Fluctuation], None]:
        """Target every key from the cached nuisances and build the estimates.

        The reduced regressions are fitted here, so unlike a TMLE retarget this runs
        learners.  The signature is the inherited one; ``g_bounds_conditional`` and
        ``intermediate_value`` are not read.
        """
        del g_bounds_conditional, intermediate_value
        self._refuse_undeclared_functions()
        requested = tuple(estimands)
        off_axis = [
            name
            for name in requested
            if name in TARGETS and not TARGETS[name].matches_axis(self._axis)
        ]
        if off_axis:
            explanation, route = _off_axis_reason(off_axis, self._axis)
            raise CapabilityError(
                f"retarget cannot report {off_axis} from this fit's nuisances. {explanation}.",
                reason=RefusalReason.API_REDIRECTION,
                alternative=f"{route}.",
            )
        if missingness is not None:
            raise CapabilityError(
                "AdaptiveIPTW fits no outcome regression, and a missingness tilt reweights one.",
                reason=RefusalReason.API_REDIRECTION,
                alternative=_TMLE_ALTERNATIVE,
            )
        level = self.alpha_sig if alpha_sig is None else alpha_sig
        regimes = nuisance.regimes
        reference = self._reference_arm(data, regimes, nuisance.policies)
        mean_bounds = g_bounds or resolve_g_bounds(
            self.g_bounds, self._bounds_n(data), for_att=False
        )
        floor = self.nuisance_bound if nuisance_bound is None else nuisance_bound
        route = iptw_missing_data_route(data, requested)
        specs = self._specs(data, requested, regimes)

        # The mechanism each key divides by. A composite fit reads the composite product at
        # this call's bounds, so a retarget at new bounds cannot read a stale one.
        propensity: FloatArray | None = None
        arm_bounds = mean_bounds
        # The fold-wise construction reads each fold's mechanism at every row (F1 of the X21
        # run-1 diagnosis).  A fit in sample, or one that kept no fold mechanisms, has none.
        by_fold = (
            [nuisance.at_fold(fold) for fold in range(nuisance.folds.n_folds)]
            if nuisance.fold_mechanisms is not None and not nuisance.folds.is_single
            else None
        )
        fold_propensity: FloatArray | None = None
        if any(spec.arm_weights is not None for spec in specs):
            if route == "composite":
                state = composite_state(data, nuisance, g_bounds=mean_bounds, nuisance_bound=floor)
                fit_data = state.data
                arm_bounds = state.bounds
                propensity = np.asarray(state.nuisance.propensity.bounded(arm_bounds), dtype=float)
                if by_fold is not None:
                    fold_propensity = np.stack(
                        [
                            composite_state(
                                data, each, g_bounds=mean_bounds, nuisance_bound=floor
                            ).nuisance.propensity.bounded(arm_bounds)
                            for each in by_fold
                        ]
                    )
            else:
                fit_data = data
                propensity = np.asarray(nuisance.propensity.bounded(mean_bounds), dtype=float)
                if by_fold is not None:
                    fold_propensity = np.stack(
                        [each.propensity.bounded(mean_bounds) for each in by_fold]
                    )
        else:
            fit_data = data
        learners = {task: self._reduced_learner(task) for task in {spec.task for spec in specs}}

        solved: list[_Solved] = []
        top = float(np.nextafter(1.0, 0.0))
        realised = np.asarray(data.treatment, float)
        for spec in specs:
            fold_mechanism: FloatArray | None = None
            if spec.arm_weights is not None:
                assert propensity is not None
                mechanism = np.sum(propensity * spec.arm_weights, axis=1)
                if fold_propensity is not None:
                    fold_mechanism = np.sum(fold_propensity * spec.arm_weights[None], axis=2)
                bounds = arm_bounds
                source = fit_data
            else:
                pi = nuisance.missingness_at_realised_arm(realised)
                if pi is None:
                    raise RuntimeError("the natural-course key needs the fitted response model")
                mechanism = np.clip(pi, floor, top)
                if by_fold is not None:
                    fold_pi = [each.missingness_at_realised_arm(realised) for each in by_fold]
                    if any(each is None for each in fold_pi):
                        raise RuntimeError(
                            "the natural-course key needs every fold's response model"
                        )
                    fold_mechanism = np.clip(np.stack(fold_pi), floor, top)  # type: ignore[arg-type]
                bounds = (float(floor), top)
                source = data
            solved.append(
                self._solve_key(
                    source,
                    nuisance,
                    spec,
                    mechanism,
                    bounds,
                    learners[spec.task],
                    fold_mechanism=fold_mechanism,
                )
            )

        self._warn_on_cap(solved)

        requested = self._apply_range_rule(data, requested, solved, regimes)
        estimates, stratum_of = self._assemble(data, nuisance, solved, requested, level, reference)
        record = self._record(data, nuisance, solved, route)
        group = "regime" if regimes is not None else "mean"
        status = self._inference_status(data, requested)
        cluster_reference = (
            None
            if data.cluster is None
            else ClusterReference(
                data.cluster,
                weights=data.weights if data.is_weighted else None,
                strata=data.strata,
                stratum_of=stratum_of,
            )
        )
        ordered = stamp_inference(
            _in_report_order(estimates, requested), status, cluster_reference, classification
        )
        return ordered, {group: self._fluctuation(data, solved, record)}, None

    def _warn_on_cap(self, solved: Sequence[_Solved]) -> None:
        """Warn once when a key stopped at ``max_iter`` with an unmet score rule."""
        unmet = [
            item.spec.name for item in solved if relative_score(item.score, item.scale) > self.tol
        ]
        if unmet:
            warnings.warn(
                f"AdaptiveIPTW stopped at max_iter={self.max_iter} before the relative score "
                f"reached tol={self.tol:g} for {sorted(set(unmet))}. A refit of the reduced "
                "regression on each tilted propensity moves the score, and a flexible "
                "learner can keep it above the tolerance. The score check reports it.",
                ConvergenceWarning,
                stacklevel=3,
            )

    def _apply_range_rule(
        self,
        data: CausalData,
        requested: tuple[str, ...],
        solved: Sequence[_Solved],
        regimes: Any,
    ) -> tuple[str, ...]:
        """Refuse a ratio target whose Horvitz-Thompson mean left its range.

        A Horvitz-Thompson mean is not a substitution estimator, so a binary-outcome mean can
        leave ``(0, 1)``. An explicit request for ``rr``, ``or`` or ``paf`` raises
        :class:`~cleverly.exceptions.DataError`. A default or ``"all"`` list drops the target
        and warns, so a default fit on a small sample still returns its arm means.

        The boundary test has a margin of ``tol`` times the summed score scale of the owner's
        keys. For an arm whose observed outcomes are all 1, the reduced regression is 1 and the
        targeted score is ``P_n[I / g*] - 1``, which is the arm mean minus 1. The loop stops
        when that score is at most ``tol`` times its scale, so the fit does not resolve the mean
        closer than the margin to the boundary. Without the margin, rounding decides the rule.
        """
        guarded = [name for name in requested if parameter_stem(name) in _RANGE_STEMS]
        if not guarded or data.family != "binomial" or regimes is not None:
            return requested
        weights = np.asarray(data.weights, dtype=float)
        means = _owner_values(solved, self.form, weights, None)
        labels = {item.spec.owner: item.spec.owner_name for item in solved}
        margins: dict[float | None, float] = {}
        for item in solved:
            owner = item.spec.owner
            margins[owner] = margins.get(owner, 0.0) + self.tol * float(np.sum(item.scale))
        failures: dict[str, str] = {}
        for name in guarded:
            stem = parameter_stem(name)
            domain = _RANGE_STEMS[stem]
            for owner, (psi, _) in means.items():
                if owner is None and stem != "paf":
                    continue
                if stem == "paf" and owner is not None and owner != self._reference_arm(data):
                    continue
                margin = margins[owner]
                inside = psi > margin if domain == "positive" else margin < psi < 1.0 - margin
                if not inside:
                    owner_label = NATURAL_COURSE_KEY if owner is None else labels[owner]
                    where = "which is not positive" if domain == "positive" else "outside (0, 1)"
                    failures[name] = (
                        f"AdaptiveIPTW estimated E[Y({owner_label})] = {psi:.12g}, {where} "
                        f"at the fit's resolution of {margin:.2g} (tol times the score scale). "
                        "A Horvitz-Thompson mean is not confined to the outcome's range, so "
                        f"{name} is undefined on this sample. Request ate and the arm means, "
                        "or use TMLE, whose substitution means stay in (0, 1)"
                    )
                    break
        if not failures:
            return requested
        # Only the default report and ``"all"`` drop a target that is undefined on the
        # sample. A call whose names differ from the default list raises, so a report never
        # silently omits a name the caller asked for. A ``retarget`` that names exactly the
        # default list is treated as the default and drops with the warning.
        defaulted = (self.estimands is None or self.estimands == "all") and tuple(
            requested
        ) == self._resolved_names(data, refuse_outside=False)
        if defaulted:
            warnings.warn(
                f"AdaptiveIPTW dropped {sorted(failures)} from the default report: "
                + next(iter(failures.values())),
                DataWarning,
                stacklevel=3,
            )
            return tuple(name for name in requested if name not in failures)
        raise DataError(next(iter(failures.values())))

    # ---------------------------------------------------------------- assembly

    def _context(
        self,
        data: CausalData,
        nuisance: NuisanceEstimates,
        owner_means: dict[float, ArmMean],
        natural: ArmMean | None,
        level: float,
        reference: float,
        rows: IntArray | None,
    ) -> _IptwContext:
        """The context the targets build from, on ``rows`` or on every row."""
        regimes = nuisance.regimes
        scaled = np.asarray(data.outcome, dtype=float)
        weights = np.asarray(data.weights, dtype=float)
        observed = np.asarray(data.observed, dtype=bool)
        treatment = np.asarray(data.treatment, dtype=float)
        cluster = data.cluster
        n = data.n
        if rows is not None:
            scaled, observed, treatment = scaled[rows], observed[rows], treatment[rows]
            weights = weights[rows]
            weights = weights / weights.mean()
            cluster = None if cluster is None else cluster[rows]
            n = int(rows.size)
        codes, labels = self._parameter_axis(data, regimes, None, None, None)
        return _IptwContext(
            scaled=scaled,
            targeted=UnfittedOutcome(),
            submodel=Submodel(
                np.zeros((n, 0)), {code: np.zeros((n, 0)) for code in codes}, (), "mean"
            ),
            treatment=treatment,
            weights=weights,
            observed=observed,
            scaler=nuisance.scaler,
            n=n,
            cluster=cluster,
            alpha_sig=level,
            arms=codes,
            arm_labels=labels,
            reference=reference,
            regimes=None if regimes is None else regimes.values,
            always_label=regimes is not None,
            natural_course=natural,
            owner_means=owner_means,
        )

    def _targets(
        self, context: _IptwContext, group: str, requested: Sequence[str]
    ) -> dict[str, ParameterEstimate]:
        """Build every requested target of ``group`` from a context."""
        out: dict[str, ParameterEstimate] = {}
        for target in targets_for(group, requested):
            for estimate in target.build(context):
                out[estimate.name] = estimate
        sink = READS_SINK.get()
        if sink is not None:
            sink.update(context.reads)
        return out

    def _assemble(
        self,
        data: CausalData,
        nuisance: NuisanceEstimates,
        solved: Sequence[_Solved],
        requested: tuple[str, ...],
        level: float,
        reference: float,
    ) -> tuple[dict[str, ParameterEstimate], dict[str, int]]:
        """The estimates, and the stratum code of each stratum estimate."""
        regimes = nuisance.regimes
        group = "regime" if regimes is not None else "mean"
        weights = np.asarray(data.weights, dtype=float)

        def means_on(
            rows: IntArray | None, form: str
        ) -> tuple[dict[float, ArmMean], ArmMean | None]:
            values = _owner_values(solved, form, weights, rows)
            natural = values.pop(None, None)
            return (
                {
                    float(owner): ArmMean(psi, curve)
                    for owner, (psi, curve) in values.items()
                    if owner is not None
                },
                None if natural is None else ArmMean(*natural),
            )

        owner_means, natural = means_on(None, self.form)
        context = self._context(data, nuisance, owner_means, natural, level, reference, None)
        estimates = self._targets(context, group, requested)
        stratum_of: dict[str, int] = {}
        if data.strata is not None:
            for code, _ in enumerate(stratum_probabilities(data)):
                rows = np.flatnonzero(data.strata == code).astype(np.int64)
                owner_s, natural_s = means_on(rows, self.form)
                stratum_context = self._context(
                    data, nuisance, owner_s, natural_s, level, reference, rows
                )
                label = data.stratum_label(code)
                for estimate in self._targets(stratum_context, group, requested).values():
                    name = stratum_alias(estimate.name, label)
                    curve = embed_stratum_curve(estimate.influence_curve, rows, data.n)
                    estimates[name] = make_estimate(
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
                    stratum_of[name] = code
        return estimates, stratum_of

    def _record(
        self,
        data: CausalData,
        nuisance: NuisanceEstimates,
        solved: Sequence[_Solved],
        route: str,
    ) -> AdaptiveIPTWFit:
        """The :class:`AdaptiveIPTWFit` of the targeted solutions."""
        weights = np.asarray(data.weights, dtype=float)
        labels = {item.spec.owner: item.spec.owner_name for item in solved}
        untargeted = {
            NATURAL_COURSE_KEY if owner is None else labels[owner]: psi
            for owner, (psi, _) in _owner_values(solved, "untargeted", weights, None).items()
        }

        def with_error(form: str) -> dict[str, tuple[float, float]]:
            out: dict[str, tuple[float, float]] = {}
            for owner, (psi, curve) in _owner_values(solved, form, weights, None).items():
                name = NATURAL_COURSE_KEY if owner is None else labels[owner]
                estimate = make_estimate(name, psi, curve, n=data.n, cluster=data.cluster)
                out[name] = (psi, float(estimate.plugin_std_error))
            return out

        strata = (
            None
            if data.strata is None
            else tuple(str(data.stratum_label(code)) for code in range(data.n_strata))
        )
        suffix = [""] if strata is None else [f" | {label}" for label in strata]
        epsilon: dict[str, tuple[float, ...]] = {}
        score: dict[str, float] = {}
        bound_active: dict[str, int] = {}
        keys: list[str] = []
        constant: list[str] = []
        diagnostics: dict[str, list[SuperLearnerDiagnostics]] = {}
        for item in solved:
            lower, upper = item.bounds
            active = int(np.count_nonzero((item.mechanism <= lower) | (item.mechanism >= upper)))
            for column, tail in enumerate(suffix):
                name = f"{item.spec.name}{tail}"
                keys.append(name)
                epsilon[name] = tuple(float(step[column]) for step in item.steps)
                score[name] = float(item.score[column])
                bound_active[name] = active
                if item.constant_design:
                    constant.append(name)
            diagnostics[item.spec.name] = item.diagnostics
        regimes = nuisance.regimes
        rounds = max((len(item.steps) for item in solved), default=0)
        met = all(relative_score(item.score, item.scale) <= self.tol for item in solved)
        route_of: dict[str, str] = {}
        if regimes is not None:
            for item in solved:
                route_of[item.spec.owner_name] = "S11" if item.spec.kind == "regime_arm" else "S9"
        arm_labels = (
            tuple(str(regimes.label(code)) for code in regimes.codes)
            if regimes is not None
            else tuple(str(data.arm_label(code)) for code in data.arm_codes)
            if data.has_arms
            else ()
        )
        learner = self.reduced_outcome_learner
        return AdaptiveIPTWFit(
            form=self.form,
            axis="regime" if regimes is not None else "arm",
            arms=arm_labels,
            strata=strata,
            regimes=None if regimes is None else tuple(str(name) for name in regimes.names),
            regime_route=route_of,
            keys=tuple(keys),
            untargeted=untargeted,
            one_step=with_error("one_step"),
            targeted=with_error("targeted"),
            epsilon=epsilon,
            rounds=rounds,
            stop="tolerance" if met else "cap",
            score=score,
            reduced_learner=repr(learner if learner is not None else self.outcome_learner),
            constant_design=tuple(constant),
            bound_active=bound_active,
            missing_data=route,
            diagnostics=diagnostics,
        )

    def _fluctuation(
        self, data: CausalData, solved: Sequence[_Solved], record: AdaptiveIPTWFit
    ) -> IptwFluctuation:
        """The fluctuation record: one score entry per key and stratum."""
        names: list[str] = []
        score: list[FloatArray] = []
        scale: list[FloatArray] = []
        initial: list[FloatArray] = []
        epsilon: list[float] = []
        weights = np.asarray(data.weights, dtype=float)
        absolute: list[FloatArray] = []
        suffix = (
            [""]
            if data.strata is None
            else [f" | {data.stratum_label(code)}" for code in range(data.n_strata)]
        )
        for item in solved:
            for column, tail in enumerate(suffix):
                names.append(f"{item.spec.name}{tail}")
                epsilon.append(float(sum(float(step[column]) for step in item.steps)))
                absolute.append(np.abs(weights * item.covariate[:, column]))
            score.append(item.score)
            scale.append(item.scale)
            initial.append(item.score_initial)
        trace = tuple(
            max(
                (item.trace[min(step, len(item.trace) - 1)] for item in solved),
                default=0.0,
            )
            for step in range(max((len(item.trace) for item in solved), default=1))
        )
        failures = [item.failure for item in solved if item.failure is not None]
        converged = all(relative_score(item.score, item.scale) <= self.tol for item in solved)
        return IptwFluctuation(
            epsilon=np.asarray(epsilon, dtype=float),
            targeted=UnfittedOutcome(),
            score=np.concatenate(score) if score else np.zeros(0),
            converged=converged,
            n_iter=max((len(item.steps) for item in solved), default=0),
            trace=trace,
            method="iterative",
            names=tuple(names),
            score_scale=np.concatenate(scale) if scale else np.zeros(0),
            score_initial=np.concatenate(initial) if initial else np.zeros(0),
            failure=None if converged else (failures[0] if failures else "max_iter_reached"),  # type: ignore[arg-type]
            absolute_score_weights=np.column_stack(absolute) if absolute else None,
            record=record,
        )

    # -------------------------------------------------------------------- fit

    def _fit_single(
        self,
        data: CausalData,
        *,
        intermediate_value: float | None,
        shared: Any = None,
    ) -> Any:
        """Fit, then move the record of the targeting to ``result.extra["adaptive_iptw"]``."""
        result = super()._fit_single(data, intermediate_value=intermediate_value, shared=shared)
        fluctuation = next(iter(result.repeats[0].fluctuations.values()))
        record = getattr(fluctuation, "record", None)
        reported = tuple(result.estimates)
        config = result.config
        if tuple(config.estimands) != reported and config.estimands:
            kept = tuple(name for name in config.estimands if name in _stems(reported))
            config = replace(config, estimands=kept)
        return replace(
            result,
            config=config,
            extra={**result.extra, "adaptive_iptw": record},
        )


def _stems(names: Sequence[str]) -> set[str]:
    return {parameter_stem(name) for name in names}


def _is_classifier(learner: Any) -> bool:
    """Whether an explicitly passed learner predicts classes."""
    if learner is None:
        return False
    if getattr(learner, "task", None) == "classification":
        return True
    try:
        return bool(is_classifier(learner))
    except Exception:  # pragma: no cover - a learner that sklearn cannot inspect
        return False


def _constant_design(
    design: FloatArray,
    mask: BoolArray,
    data: CausalData,
    folds: Folds,
    fold_designs: FloatArray | None = None,
) -> bool:
    """Whether the design is one value on the fit rows of the sample or of a training fold.

    With ``fold_designs``, the ``(K, n)`` fold-wise slab, training fold ``k`` reads entry ``k``.
    """
    values = np.asarray(design, dtype=float)
    if folds.is_single:
        rows = np.flatnonzero(mask)
        return bool(rows.size and np.all(values[rows] == values[rows][0]))
    for index, (train, _) in enumerate(folds):
        rows = train[mask[train]]
        here = values if fold_designs is None else np.asarray(fold_designs[index], dtype=float)
        if rows.size and np.all(here[rows] == here[rows][0]):
            return True
    return False
