r"""Collaborative TMLE for a time-to-event outcome under a baseline treatment.

Stitelman and van der Laan (2010), *The International Journal of Biostatistics* 6(1),
Article 21, Sections 10 and 11.  The parameter is the treatment-specific risk
:math:`F_a(t) = 1 - \psi_a(t)`, with :math:`\psi_a(t) = E_W S(t \mid a, W)` the survival of
their (4), for one binary baseline treatment held over every node, baseline covariates,
one event type and right censoring.  The targeting step is the hazard fluctuation of
:func:`~cleverly.longitudinal.hazard.solve_per_horizon`, their (16): the treatment
mechanism :math:`g_1` and the censoring hazard :math:`g_2` are held fixed while the event
hazard moves.

What the collaborative step selects is the pair :math:`(g_1, g_2)`.  For each horizon
:math:`t` and each static arm :math:`a` the selector builds a greedy sequence of moves.
:math:`M_0` is an intercept-only main-term logistic :math:`g_1` and a :math:`g_2` pooled over
the person-nodes at risk of censoring with one indicator per node.  Each move adds one
baseline covariate to :math:`g_1`, or one term of :math:`\{A\} \cup W` to :math:`g_2`,
whichever gives the targeted fit with the smallest penalized risk.  When no move lowers the
risk strictly, the current targeted hazards become the new initial fit and the search runs
again from them.  If that pass finds no strict improvement either, its best move is taken
(van der Laan and Gruber 2010; the source states no rule for this case).  The number of moves
is chosen by ``selection_folds``-fold cross-validation of the penalized risk, summed over the
folds, and the full-data sequence is cut there.

The risk of a targeted fit on a set of rows :math:`V` is the negative weighted binomial
log-likelihood of its event hazard over the person-nodes at risk at :math:`k \le t`, plus the
printed penalty (24) of the source,

.. math::

    \operatorname{pen}_a(V) = \frac{1}{|V|} \sum_{i \in V} \sum_{k \le t}
        \frac{1}{\pi_k(a, W_i)} \frac{\hat S(t \mid a, W_i)}{\hat S(k \mid a, W_i)}
        \hat\lambda_k(a, W_i)\bigl(1 - \hat\lambda_k(a, W_i)\bigr),

where :math:`\pi_k` is the candidate's bounded cumulative :math:`g_1 \prod_{s \le k} c_s`
that the targeting divides by, and :math:`\hat\lambda`, :math:`\hat S` are the initial
hazards.  The penalty is not the influence-curve variance: that variance squares the survival
ratio and carries an at-risk factor.  The printed form is the one the source's simulations
used.

The source orders the event before censoring within an interval.  The container orders
censoring at node :math:`k` before the event there, so the container's :math:`c_k` is the
source's :math:`1 - g_2(k - 1)`, and :math:`G_k` is the source's product when
:math:`c_1 \equiv 1`, which every :meth:`~cleverly.longitudinal.LongitudinalData.from_time_to_event`
container holds.

The source states no theorem, so a fit reports points: every estimate takes the
``"working_mechanism_plugin"`` status, with the hazard influence curve at the selected state
kept as ``plugin_std_error`` and ``plugin_interval``.  F18 in ``docs/roadmap.md`` reopens
inference.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar, Final

import numpy as np
from scipy.special import expit, xlogy

from .._inference_status import InferenceStatus, precedent_status
from .._refusal import NO_ALTERNATIVE, Refusal, RefusalReason
from .._typing import BoolArray, FloatArray, Learner
from .._warnings import warn as _warn
from ..exceptions import CapabilityError, ConvergenceWarning, LongitudinalError
from ..inference.multiplier import (
    DEFAULT_BAND_REFERENCE,
    DEFAULT_BAND_SCALE,
    DEFAULT_KIND,
    DEFAULT_N_REPLICATES,
)
from ..learners.crossfit import Folds, make_folds
from ..learners.glm import _active_columns
from ..utils.bounds import bound
from .data import LongitudinalData
from .estimator import (
    DEFAULT_LTMLE_G_BOUNDS,
    LTMLE,
    LongitudinalResult,
    _fit_key,
    _frozen_plans,
    _HazardBlock,
    _HazardReplay,
    _LongitudinalReplayRecipe,
    _Reported,
)
from .hazard import (
    HazardFit,
    HazardInputs,
    HazardState,
    InitialHazards,
    fit_hazards,
    hazard_inputs,
    influence,
    level_label,
    solve_per_horizon,
    walk_loss,
)
from .regimen import DynamicRegimen, Plan, resolve_regimens
from .separable import split_separable
from .sequential import Mechanism, NodeBound, RegimenFit

if TYPE_CHECKING:
    from .._summary import Detail, Summary

__all__ = [
    "PENALTY_FORM",
    "SurvivalCTMLE",
    "SurvivalSelection",
    "source_penalty",
    "survival_design_refusal",
]

#: The penalty the selector adds: the printed (24) of Stitelman and van der Laan (2010).
PENALTY_FORM: Final = "printed_24"
#: How the penalty enters the risk: the loss summed over the scored rows, plus the penalty.
COMBINATION: Final = "sum_loss_plus_penalty"
#: The clip of each hazard inside the selection loss, as point ``CTMLE.loss`` clips.
_LOSS_EPS: Final = 1e-12
#: The retention a fold node holds when every eligible training row is censored there.
_HELD_RETENTION: Final = 1e-12
#: The kind of a node a mechanism holds instead of fitting.
NO_CENSORING: Final = "no_censoring"
ALL_CENSORED: Final = "all_censored"
#: The Newton settings of the main-term logistic fits, those of :class:`~cleverly.learners.GLM`.
_GLM_MAX_ITER: Final = 100
_GLM_TOL: Final = 1e-10
_SEPARATION_ETA: Final = 30.0
_GLM_HALVINGS: Final = 30
#: The treatment column's name in the censoring pool.
TREATMENT_TERM: Final = "A"

_ESTIMATOR_ALTERNATIVE = (
    "Fit LTMLE(survival_model='hazard') for the hazard TMLE with a fitted treatment and "
    "censoring mechanism."
)

# ----------------------------------------------------------------------------- refusals

CAUSES_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "SurvivalCTMLE selects the mechanisms of a time-to-event outcome with one event type, "
    "and this fit declares two or more causes. Stitelman and van der Laan (2010) define one "
    "event type, and no source states a cause-specific penalty.",
    alternative=_ESTIMATOR_ALTERNATIVE,
    roadmap="X62",
)
ELIMINATE_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "SurvivalCTMLE does not take eliminate=. An eliminated cause adds a third mechanism, the "
    "competing factor, which the moves of the source do not cover.",
    alternative=_ESTIMATOR_ALTERNATIVE,
    roadmap="X62",
)
SEPARABLE_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "SurvivalCTMLE does not fit a Separable regimen. A separable effect needs the factor-wise "
    "hazard targeting of LTMLE, and no source selects its mechanism collaboratively.",
    alternative=(
        "Fit the Separable regimen with LTMLE(survival_model='hazard', targeting='one_step')."
    ),
    roadmap="X62",
)
ARMS_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "SurvivalCTMLE fits static arms of a binary baseline treatment. Stitelman and van der "
    "Laan (2010) define the selection for a static arm of a binary treatment, so a rule, a "
    "known stochastic policy and a treatment with more than two levels are not built.",
    alternative=_ESTIMATOR_ALTERNATIVE,
    roadmap="X62",
)
POLICY_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "SurvivalCTMLE does not fit a modified treatment policy or an odds tilt. Each candidate "
    "treatment mechanism of the collaborative search would define a different parameter.",
    alternative=_ESTIMATOR_ALTERNATIVE,
    roadmap="F5",
)
SEQUENTIAL_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "SurvivalCTMLE fits one baseline treatment held over every node with baseline covariates "
    "only. A treatment at several nodes, or a covariate measured after baseline, needs a "
    "sequential collaborative score that no read source derives.",
    alternative="Fit LTMLE, which supports a treatment at several nodes and later covariates.",
    roadmap="F5",
)
DOSE_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "SurvivalCTMLE needs a categorical baseline treatment. A continuous dose needs a "
    "collaborative sequence of conditional density estimators that no read source derives.",
    alternative="Fit LTMLE with a modified treatment policy for a dose.",
    roadmap="F5",
)
OUTCOME_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "SurvivalCTMLE selects the mechanisms of a time-to-event outcome, and this fit declares an "
    "end-of-study or repeated outcome. The source scores each candidate by the likelihood of "
    "the event hazard, which such an outcome does not have.",
    alternative="Fit LTMLE for an end-of-study or repeated outcome.",
    roadmap="X62",
)
KNOWN_REFUSAL = Refusal(
    RefusalReason.API_REDIRECTION,
    "SurvivalCTMLE selects the treatment and censoring mechanisms, and this fit declares "
    "them known, so there is nothing to select.",
    alternative=(
        "Fit LTMLE(survival_model='hazard'), which reads the declared treatment and "
        "censoring probabilities."
    ),
)
CLUSTER_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "SurvivalCTMLE does not take id=. The collaborative search has no clustered result, as "
    "point CTMLE has none for its selector paths.",
    alternative="Fit LTMLE(survival_model='hazard') for a clustered panel.",
    roadmap="X51",
)
SURVIVAL_MODEL_REFUSAL = Refusal(
    RefusalReason.API_REDIRECTION,
    "SurvivalCTMLE targets the event hazard, the per-horizon fluctuation (16) of the source, "
    "so survival_model is 'hazard'.",
    alternative="Fit LTMLE(survival_model='iterated_mean') for the iterated-mean recursion.",
)
ONE_STEP_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "SurvivalCTMLE runs the per-horizon fluctuation of the source. targeting='one_step' walks "
    "every horizon together, which would make the selection joint across horizons, and no "
    "source gives that selection.",
    alternative=(
        "Keep targeting='iterative', or fit LTMLE(targeting='one_step') for a monotone curve."
    ),
    roadmap="X62",
)
STRATEGY_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "SurvivalCTMLE builds the greedy sequence of Stitelman and van der Laan (2010). The "
    "ordered, discrete, lasso, learners, derivative and oat strategies have no survival "
    "source.",
    alternative="Keep strategy='greedy'.",
    roadmap="X62",
)
BAND_REFUSAL = Refusal(
    RefusalReason.INFERENCE_GAP,
    "SurvivalCTMLE builds no simultaneous band. A band is a joint confidence statement, and no "
    "result supports a confidence statement after the collaborative selection.",
    alternative="Leave simultaneous=False; the plug-in curve stays as a diagnostic.",
    roadmap="F18",
)
BOOTSTRAP_REFUSAL = Refusal(
    RefusalReason.INFERENCE_GAP,
    "SurvivalCTMLE draws no bootstrap and no repeated split. The source offers a bootstrap "
    "but gives no evidence for it, and no result supports an interval after the "
    "collaborative selection.",
    alternative="Fit with repeats=1 and n_bootstrap=0; the estimate is a point.",
    roadmap="F18",
)
RULES_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "SurvivalCTMLE measures the targeting of each candidate at the default stopping rule, "
    "step and iterate only, so stopping=, step_bound= and iterate= keep their defaults.",
    alternative="Drop the targeting rule, or fit LTMLE(survival_model='hazard') with it.",
    roadmap="X62",
)
PROJECTION_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "SurvivalCTMLE does not take msm=, node_bounds=, ratio_trim= or isotonic=True. None of "
    "them is in the source, and a projection of separately selected points has no source.",
    alternative="Fit LTMLE(survival_model='hazard') with the setting.",
    roadmap="X62",
)
LEARNERS_REFUSAL = Refusal(
    RefusalReason.API_REDIRECTION,
    "SurvivalCTMLE fits its candidate mechanisms as the main-term logistic models of the "
    "source, so it takes no treatment_learner=, censoring_learner=, competing_learner= or "
    "pseudo_learner=. outcome_learner= fits the initial event hazard.",
    alternative="Pass the hazard learner as outcome_learner=.",
)
Q_BOUNDS_REFUSAL = Refusal(
    RefusalReason.INVALID_DECLARATION,
    "SurvivalCTMLE does not take q_bounds=. The outcome is an event indicator on [0, 1], so "
    "there is no outcome range to declare.",
)
DOSE_SETTINGS_REFUSAL = Refusal(
    RefusalReason.INVALID_DECLARATION,
    "SurvivalCTMLE does not take density_bins= or ratio='classifier'. Both set how a "
    "continuous dose or a policy ratio is estimated, and the fit reads static arms of a "
    "binary treatment.",
)
LOSS_REFUSAL = Refusal(
    RefusalReason.INVALID_DECLARATION,
    "SurvivalCTMLE scores each candidate by the hazard log-likelihood penalized by (24) of the "
    "source, so loss='squared' is not taken; 'auto' and 'loglik' both mean that likelihood. "
    "The source reports that a squared-error loss did worse.",
)
INNER_FOLDS_REFUSAL = Refusal(
    RefusalReason.INVALID_DECLARATION,
    "SurvivalCTMLE does not take selection_inner_folds=. The source fits the initial hazard "
    "once on the full data and holds it fixed in every selection fold, so there is no inner "
    "split.",
)
ESTIMAND_REFUSAL = Refusal(
    RefusalReason.INVALID_DECLARATION,
    "SurvivalCTMLE does not take selection_estimand=. It selects once per arm and horizon, as "
    "the source does, and each contrast combines the two arms' selections.",
)
TRUNCATION_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "A SurvivalCTMLE result has no truncation curve. Each bound would rerun the collaborative "
    "selection, and no study measures that replay.",
    alternative="Fit LTMLE(survival_model='hazard') at each bound for a curve.",
    roadmap="X62",
)
SIMULATED_CONFOUNDING_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "simulated_confounding has no replay for a SurvivalCTMLE result. Each replayed panel would "
    "rerun the collaborative selection, and no study measures that replay.",
    alternative=NO_ALTERNATIVE,
    roadmap="X62",
)


def settings_refusal(name: str) -> Refusal:
    """The refusal of a setting that belongs to a point C-TMLE strategy SurvivalCTMLE refuses.

    Parameters
    ----------
    name : str
        The keyword, such as ``"patience"``.

    Returns
    -------
    Refusal
        An ``invalid_declaration`` refusal that names the keyword.
    """
    return Refusal(
        RefusalReason.INVALID_DECLARATION,
        f"SurvivalCTMLE does not take {name}=. It is a setting of a point CTMLE strategy other "
        "than the greedy sequence, which SurvivalCTMLE builds.",
    )


def all_censored_refusal(node: int, time: float) -> Refusal:
    """The refusal of a node at which every eligible row is censored, at or before a horizon.

    Parameters
    ----------
    node : int
        The node, counted from one.
    time : float
        Its grid time, or the node itself on a wide panel.

    Returns
    -------
    Refusal
        An ``invalid_declaration`` refusal.
    """
    return Refusal(
        RefusalReason.INVALID_DECLARATION,
        f"every row at risk of censoring at node {node} (time {time:g}) is censored there, so "
        "no unit is at risk of the event at that node or after it, and the censoring model has "
        "no retained row to fit.",
        alternative="Set horizons= to times before that node.",
    )


def survival_design_refusal(data: LongitudinalData, regimens: Any) -> Refusal | None:
    """The refusal a prepared panel and its regimens meet, or ``None`` when SurvivalCTMLE fits them.

    One function for the estimator and for ``CausalStudy.available_methods()``, so the listing
    and the fit name one reason.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel, with any eliminated cause applied.
    regimens : Any
        The ``regimens=`` argument, or the resolved regimens.

    Returns
    -------
    Refusal or None
        The first refusal of the design, in the order of the section 1 table of the design.
    """
    _, separable = split_separable(regimens)
    if separable:
        return SEPARABLE_REFUSAL
    if not data.treatment_held and data.n_decisions > 1:
        return SEQUENTIAL_REFUSAL
    if any(names for names in data.time_varying_names):
        return SEQUENTIAL_REFUSAL
    if data.node_kind(1) != "categorical":
        return DOSE_REFUSAL
    if not data.is_survival:
        return OUTCOME_REFUSAL
    if data.eliminated:
        return ELIMINATE_REFUSAL
    if len(data.cause_labels) > 1:
        return CAUSES_REFUSAL
    if len(data.treatment_levels[0]) != 2:
        return ARMS_REFUSAL
    resolved = resolve_regimens(
        regimens,
        data.n_decisions,
        held_nodes=data.n_times if data.treatment_held else None,
    )
    for regimen in resolved:
        if isinstance(regimen, DynamicRegimen):
            if regimen.has_mtp or regimen.has_tilt:
                return POLICY_REFUSAL
            return ARMS_REFUSAL
    if data.known_mechanisms is not None:
        return KNOWN_REFUSAL
    if data.cluster is not None:
        return CLUSTER_REFUSAL
    return None


def eligible_censoring(data: LongitudinalData) -> tuple[BoolArray, BoolArray]:
    """``(n, T)`` the rows at risk of censoring at each node, and whether each was censored.

    A unit is at risk of censoring at node :math:`k` when it is uncensored and event-free
    through :math:`k - 1`, the rows :func:`~cleverly.longitudinal.sequential.fit_mechanism`
    fits a censoring factor on.

    Parameters
    ----------
    data : LongitudinalData
        A survival panel.

    Returns
    -------
    tuple of BoolArray
        The eligible rows, and the censoring indicator :math:`1 - C_k` on them.
    """
    masks = data.regimen_masks(data.treatment)
    eligible = masks.uncensored[:, :-1] & masks.event_free[:, :-1]
    censored = eligible & ~np.asarray(data.censoring_target, dtype=bool)
    return eligible, censored


def all_censored_nodes(data: LongitudinalData, through: int) -> tuple[int, ...]:
    """The nodes up to ``through`` at which every eligible row of positive weight is censored.

    Parameters
    ----------
    data : LongitudinalData
        A survival panel.
    through : int
        The last node to check, counted from one.

    Returns
    -------
    tuple of int
        The nodes, counted from one.
    """
    if not data.censoring_names:
        return ()
    eligible, censored = eligible_censoring(data)
    positive = (data.weights > 0.0)[:, None]
    out = []
    for k in range(through):
        rows = eligible[:, k] & positive[:, 0]
        if rows.any() and bool(np.all(censored[rows, k])):
            out.append(k + 1)
    return tuple(out)


# ------------------------------------------------------------------------- the penalty


def source_penalty(
    inverse: FloatArray,
    survival: FloatArray,
    hazards: FloatArray,
    weights: FloatArray,
    horizon: int,
) -> float:
    r"""The printed penalty (24) of Stitelman and van der Laan (2010) for one arm.

    .. math::

        \frac{\sum_i w_i \sum_{k \le t} \pi_k^{-1}(a, W_i)
            \frac{\hat S(t \mid a, W_i)}{\hat S(k \mid a, W_i)}
            \hat\lambda_k(a, W_i)(1 - \hat\lambda_k(a, W_i))}{\sum_i w_i},

    averaged over every row of positive weight, with no at-risk or :math:`A = a` indicator, as
    printed.  :math:`\pi_k` is the candidate's jointly bounded cumulative mechanism, and
    :math:`\hat S`, :math:`\hat\lambda` are the initial fit.  A ratio whose denominator is zero
    counts as zero, because its numerator is zero too.

    Parameters
    ----------
    inverse : FloatArray
        ``(n, T)`` :math:`1 / \pi_k(a, W_i)`, ``HazardInputs.inverse[:, :, a]``.
    survival : FloatArray
        ``(n, T)`` initial :math:`\hat S(k \mid a, W_i)`.
    hazards : FloatArray
        ``(n, T)`` initial :math:`\hat\lambda_k(a, W_i)`.
    weights : FloatArray
        ``(n,)`` the weights of the scored rows, zero elsewhere.
    horizon : int
        The horizon :math:`t`, counted from one.

    Returns
    -------
    float
        The penalty.  Zero when no row has positive weight.
    """
    w = np.asarray(weights, dtype=float)
    total = float(np.sum(w))
    if total <= 0.0:
        return 0.0
    s = np.asarray(survival, dtype=float)[:, :horizon]
    ratio = np.divide(s[:, -1:], s, out=np.zeros_like(s), where=s > 0.0)
    lam = np.asarray(hazards, dtype=float)[:, :horizon]
    terms = np.asarray(inverse, dtype=float)[:, :horizon] * ratio * lam * (1.0 - lam)
    return float(np.sum(w * terms.sum(axis=1)) / total)


def selection_loss(
    state: HazardState, inputs: HazardInputs, horizon: int, *, eps: float = _LOSS_EPS
) -> tuple[float, int]:
    """The selection loss of a targeted state on the rows of positive weight, and its clip count.

    The negative weighted binomial log-likelihood of the event hazard over the person-nodes at
    risk at :math:`k \\le t`, :func:`~cleverly.longitudinal.hazard.walk_loss` with
    ``last_node=horizon``.  Each hazard at an observed level is clipped into
    ``[eps, 1 - eps]`` first, as point ``CTMLE.loss`` clips.  No hazard reaches the clip in an
    ordinary fit: a held cell is held from the full data, and every other logit is bounded by
    ``alpha``.  The clip is a backstop, and the count says how often it acted.

    Parameters
    ----------
    state : HazardState
        The targeted hazards.
    inputs : HazardInputs
        The targeting inputs, with the weights of the scored rows and zero elsewhere.
    horizon : int
        The horizon, counted from one.
    eps : float, default=1e-12
        The clip.

    Returns
    -------
    tuple of float and int
        The loss, and the count of person-nodes whose hazard the clip moved.
    """
    rows, levels = inputs.rows, inputs.observed
    hazards = state.hazards[rows, :, levels][:, :, 0]
    counted = (
        inputs.at_risk
        & (inputs.weights != 0.0)[:, None]
        & (np.arange(inputs.n_times) < horizon)[None, :]
    )
    clipped = counted & ((hazards < eps) | (hazards > 1.0 - eps))
    count = int(np.count_nonzero(clipped))
    if count == 0:
        return walk_loss(state, inputs, last_node=horizon), 0
    lam = np.clip(hazards, eps, 1.0 - eps)
    events = inputs.events[:, :, 0]
    terms = xlogy(events, lam) + xlogy(1.0 - events, 1.0 - lam)
    return -float(np.sum(np.where(counted, inputs.weights[:, None] * terms, 0.0))), count


# --------------------------------------------------------------------- the mechanisms


@dataclass(frozen=True)
class _Logistic:
    """One main-term logistic fit: its coefficients, its active columns and its convergence."""

    coef: FloatArray
    active: BoolArray
    converged: bool


def _logistic(design: FloatArray, target: FloatArray, weights: FloatArray) -> _Logistic:
    """Fit an unpenalized weighted logistic regression with an intercept, by Newton's method.

    The iteration of :class:`~cleverly.learners.GLM` (binomial): a column that is a linear
    combination of the earlier columns on the rows of positive weight is aliased and enters
    with coefficient zero, Newton starts from zero, and it stops when the largest step is at
    most ``1e-10`` times one plus the largest coefficient.  Where ``GLM`` raises, at a
    non-converged or separated fit, this keeps the last iterate and reports
    ``converged=False``, so the bounded cumulative product of the candidate is used.  A
    step that is not finite ends the iteration, and a step whose deviance is not finite is
    halved, so the iterate kept is always finite.
    """
    # Overflow is handled by the finiteness checks below, so numpy need not warn of it.
    with np.errstate(over="ignore", invalid="ignore"):
        return _newton(design, target, weights)


def _newton(design: FloatArray, target: FloatArray, weights: FloatArray) -> _Logistic:
    full = np.column_stack([np.ones(design.shape[0]), design])
    active = _active_columns(full, weights)
    x = full[:, active]
    coef = np.zeros(x.shape[1])
    converged = False
    deviance = _deviance(x, coef, target, weights)
    for _ in range(_GLM_MAX_ITER):
        mu = expit(x @ coef)
        score = x.T @ (weights * (target - mu))
        information = (x * (weights * mu * (1.0 - mu))[:, None]).T @ x
        try:
            move = np.linalg.solve(information, score)
        except np.linalg.LinAlgError:
            move = np.linalg.lstsq(information, score, rcond=None)[0]
        if not np.all(np.isfinite(move)):
            break
        # Halve a step whose deviance is not finite; a finite full step is taken as Newton
        # takes it, so a well-posed fit follows the GLM iteration exactly.
        for _ in range(_GLM_HALVINGS):
            trial = _deviance(x, coef + move, target, weights)
            if np.isfinite(trial):
                break
            move = move / 2.0
        else:
            break
        coef = coef + move
        deviance = trial
        if np.max(np.abs(move)) <= _GLM_TOL * (1.0 + np.max(np.abs(coef))):
            converged = True
            break
    if not np.isfinite(deviance):
        converged = False
    eta = x @ coef
    if np.any(np.abs(eta[weights > 0]) > _SEPARATION_ETA):
        converged = False
    return _Logistic(np.asarray(coef, dtype=float), np.asarray(active, dtype=bool), converged)


def _deviance(x: FloatArray, coef: FloatArray, target: FloatArray, weights: FloatArray) -> float:
    """The weighted binomial deviance at ``coef``, ``inf`` where any term is not finite."""
    with np.errstate(all="ignore"):
        eta = x @ coef
        terms = weights * (np.logaddexp(0.0, eta) - target * eta)
        value = float(2.0 * np.sum(terms))
    return value if np.isfinite(value) else np.inf


def _finite_argmin(values: Sequence[float] | FloatArray) -> int:
    """The first index of the least finite value; a non-finite value is never the least.

    Returns ``-1`` when no value is finite.
    """
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    if not finite.any():
        return -1
    return int(np.argmin(np.where(finite, array, np.inf)))


def _predict(fit: _Logistic, design: FloatArray) -> FloatArray:
    full = np.column_stack([np.ones(design.shape[0]), design])
    return np.asarray(expit(full[:, fit.active] @ fit.coef), dtype=float)


@dataclass(frozen=True)
class _Treatment:
    """``(n, L)`` :math:`\\hat g_1(l \\mid W)` of one branch and term set."""

    probabilities: FloatArray
    converged: bool
    dropped: int


@dataclass(frozen=True)
class _Retention:
    """``(n, T, L)`` :math:`\\hat c_k(l, W) = 1 - \\hat g_2(k \\mid l, W)` of one branch."""

    retention: FloatArray
    converged: bool
    dropped: int
    #: ``(fold, node, kind)`` of each node held rather than fitted, fold 0 for one fit.
    held: tuple[tuple[int, int, str], ...]


@dataclass(frozen=True)
class Candidate:
    """One candidate pair of mechanisms, evaluated at every treatment level.

    Parameters
    ----------
    g1_terms : tuple of str
        The baseline covariates of the treatment model, in data order.
    g2_terms : tuple of str
        The terms of the censoring model, ``"A"`` first, then covariates in data order.
    treatment : FloatArray
        ``(n, L)`` :math:`\\hat g_1(l \\mid W_i)`.
    retention : FloatArray
        ``(n, T, L)`` :math:`\\hat c_k(l, W_i)`, the censoring model evaluated at
        :math:`A = l` for every unit.
    unbounded : tuple of FloatArray
        One ``(n, T)`` raw cumulative :math:`g_1 \\prod_{s \\le k} c_s` per level.
    cumulative : tuple of FloatArray
        The same, bounded into ``g_bounds`` on each prefix.
    converged : bool
        Whether both logistic fits converged.
    dropped : int
        The terms the two fits aliased with earlier columns.
    held : tuple of tuple
        ``(fold, node, kind)`` of each censoring node held rather than fitted.
    """

    g1_terms: tuple[str, ...]
    g2_terms: tuple[str, ...]
    treatment: FloatArray
    retention: FloatArray
    unbounded: tuple[FloatArray, ...]
    cumulative: tuple[FloatArray, ...]
    converged: bool
    dropped: int
    held: tuple[tuple[int, int, str], ...]

    @property
    def inverse(self) -> FloatArray:
        """``(n, T, L)`` one over the bounded cumulative, as :func:`hazard_inputs` builds it."""
        return np.stack([1.0 / np.asarray(c, dtype=float) for c in self.cumulative], axis=2)


class _Mechanisms:
    """The candidate mechanisms of one fit, cached by branch, mechanism and term set."""

    def __init__(
        self,
        data: LongitudinalData,
        outer: Folds,
        bounds: tuple[float, float],
        *,
        evaluate_at_observed: bool = False,
    ) -> None:
        self.data = data
        self.outer = outer
        self.bounds = bounds
        self.covariates = tuple(data.baseline_names)
        self.observed = np.asarray(np.nan_to_num(data.treatment[:, 0], nan=0.0), dtype=np.int64)
        self.n_levels = len(data.treatment_levels[0])
        self.eligible, self.censored = eligible_censoring(data)
        self.positive = np.asarray(data.weights > 0.0)
        #: A mutation hook for the full-set check: evaluate the censoring model at each unit's
        #: observed treatment rather than at every level.  Never set by a fit.
        self.evaluate_at_observed = evaluate_at_observed
        self._g1: dict[tuple[str, tuple[str, ...]], _Treatment] = {}
        self._g2: dict[tuple[str, tuple[str, ...]], _Retention] = {}
        self._pairs: dict[tuple[str, tuple[str, ...], tuple[str, ...]], Candidate] = {}

    # -- branches

    def _splits(
        self, branch: str, train: BoolArray | None
    ) -> list[tuple[int, BoolArray, BoolArray]]:
        """``(fold, fit rows, predicted rows)`` of a branch.

        The full branch fits on every row in sample, or out of fold on the outer split.  A
        selection-fold branch fits on its training rows and predicts every row.
        """
        n = self.data.n
        everyone = np.ones(n, dtype=bool)
        if train is not None:
            return [(0, train & self.positive, everyone)]
        if self.outer.is_single:
            return [(0, self.positive.copy(), everyone)]
        out = []
        for fold, (fit_rows, test) in enumerate(self.outer, start=1):
            mask = np.zeros(n, dtype=bool)
            mask[fit_rows] = True
            predict = np.zeros(n, dtype=bool)
            predict[test] = True
            out.append((fold, mask & self.positive, predict))
        return out

    # -- g1

    def treatment(self, branch: str, train: BoolArray | None, terms: tuple[str, ...]) -> _Treatment:
        key = (branch, terms)
        cached = self._g1.get(key)
        if cached is not None:
            return cached
        data = self.data
        target = (self.observed == 1).astype(float)
        columns = [self.covariates.index(name) for name in terms]
        design = np.asarray(data.baseline[:, columns], dtype=float).reshape(data.n, len(columns))
        p = np.empty(data.n)
        converged, dropped = True, 0
        for _, fit_rows, predict in self._splits(branch, train):
            w = data.weights[fit_rows]
            if not terms:
                # M0: the weighted share treated, by hand, as point _intercept_propensity does.
                p[predict] = float(np.sum(w * target[fit_rows]) / np.sum(w))
                continue
            fit = _logistic(design[fit_rows], target[fit_rows], w)
            converged = converged and fit.converged
            dropped = max(dropped, int(np.count_nonzero(~fit.active[1:])))
            p[predict] = _predict(fit, design[predict])
        probabilities = np.column_stack([1.0 - p, p])
        out = _Treatment(probabilities, converged, dropped)
        self._g1[key] = out
        return out

    # -- g2

    def retention(self, branch: str, train: BoolArray | None, terms: tuple[str, ...]) -> _Retention:
        key = (branch, terms)
        cached = self._g2.get(key)
        if cached is not None:
            return cached
        data = self.data
        n, n_times, n_levels = data.n, data.n_times, self.n_levels
        retention = np.ones((n, n_times, n_levels))
        converged, dropped = True, 0
        held: list[tuple[int, int, str]] = []
        if data.censoring_names:
            for fold, fit_rows, predict in self._splits(branch, train):
                fitted: list[int] = []
                for k in range(n_times):
                    rows = self.eligible[:, k] & fit_rows
                    if not rows.any() or not self.censored[rows, k].any():
                        held.append((fold, k + 1, NO_CENSORING))
                    elif bool(np.all(self.censored[rows, k])):
                        held.append((fold, k + 1, ALL_CENSORED))
                        retention[predict, k, :] = _HELD_RETENTION
                    else:
                        fitted.append(k)
                if not fitted:
                    continue
                design_rows, target, weights = [], [], []
                for k in fitted:
                    members = np.flatnonzero(self.eligible[:, k] & fit_rows)
                    design_rows.append(self._g2_design(members, k, fitted, terms, None))
                    target.append(self.censored[members, k].astype(float))
                    weights.append(data.weights[members])
                fit = _logistic(
                    np.vstack(design_rows), np.concatenate(target), np.concatenate(weights)
                )
                converged = converged and fit.converged
                indicators = len(fitted) - 1
                dropped = max(dropped, int(np.count_nonzero(~fit.active[1 + indicators :])))
                index = np.flatnonzero(predict)
                for k in fitted:
                    for level in range(n_levels):
                        at = None if self.evaluate_at_observed else level
                        hazard = _predict(fit, self._g2_design(index, k, fitted, terms, at))
                        retention[index, k, level] = 1.0 - hazard
        out = _Retention(retention, converged, dropped, tuple(held))
        self._g2[key] = out
        return out

    def _g2_design(
        self,
        rows: Any,
        node: int,
        fitted: Sequence[int],
        terms: tuple[str, ...],
        level: int | None,
    ) -> FloatArray:
        """The pooled censoring design of ``rows`` at ``node``.

        One indicator per fitted node after the first, then the terms.  ``level=None`` reads
        each row's observed treatment, which is what the fit trains on.
        """
        size = len(rows) if not isinstance(rows, slice) else self.data.n
        columns = [np.full(size, float(node == other)) for other in fitted[1:]]
        for term in terms:
            if term == TREATMENT_TERM:
                values = self.observed[rows] if level is None else np.full(size, level)
                columns.append(np.asarray(values, dtype=float))
            else:
                columns.append(
                    np.asarray(self.data.baseline[rows, self.covariates.index(term)], dtype=float)
                )
        if not columns:
            return np.zeros((size, 0))
        return np.column_stack(columns)

    # -- pairs

    def candidate(
        self,
        branch: str,
        train: BoolArray | None,
        g1_terms: tuple[str, ...],
        g2_terms: tuple[str, ...],
    ) -> Candidate:
        """The candidate pair, with its cumulative mechanism at every level."""
        key = (branch, g1_terms, g2_terms)
        cached = self._pairs.get(key)
        if cached is not None:
            return cached
        g1 = self.treatment(branch, train, g1_terms)
        g2 = self.retention(branch, train, g2_terms)
        data = self.data
        lower, upper = self.bounds
        unbounded, cumulative = [], []
        for level in range(self.n_levels):
            # The product first and the bound on each prefix, in the order of
            # Mechanism._cumulative_with_unbounded, so the full-set candidate reproduces the
            # hazard LTMLE fit at rtol=0.
            running = np.ones(data.n)
            raw_columns, bounded_columns = [], []
            for k in range(data.n_times):
                if k == 0:
                    running = running * g1.probabilities[:, level]
                if data.censoring_names:
                    running = running * g2.retention[:, k, level]
                raw_columns.append(running)
                bounded_columns.append(bound(running, lower, upper))
            unbounded.append(np.column_stack(raw_columns))
            cumulative.append(np.column_stack(bounded_columns))
        out = Candidate(
            g1_terms=g1_terms,
            g2_terms=g2_terms,
            treatment=g1.probabilities,
            retention=g2.retention,
            unbounded=tuple(unbounded),
            cumulative=tuple(cumulative),
            converged=g1.converged and g2.converged,
            dropped=g1.dropped + g2.dropped,
            held=g2.held,
        )
        self._pairs[key] = out
        return out


# ----------------------------------------------------------------------- the selector


@dataclass(frozen=True)
class _Position:
    """One position of a selection path: a candidate and its targeted fit."""

    move: tuple[str, str] | None
    candidate: Candidate
    state: HazardState
    record: HazardFit
    inputs: HazardInputs
    n_steps: int
    loss: float
    penalty: float
    clipped: int

    @property
    def risk(self) -> float:
        return self.loss + self.penalty


_Move = tuple[str, str]


def _finite(value: float) -> float:
    """``value``, or ``inf`` where it is not finite."""
    return float(value) if np.isfinite(value) else np.inf


def greedy_rule(
    pools: Sequence[Sequence[tuple[str, str]]],
    score: Callable[[tuple[tuple[str, str], ...], tuple[str, str] | None, int], float],
) -> list[tuple[tuple[str, str] | None, int]]:
    """The greedy move sequence of the source with the forced addition of van der Laan and Gruber.

    A scripted form of the rule :class:`_HazardSelector` follows, which a test drives with a
    risk table.  ``pools`` lists the moves in try order.  ``score(accepted, move, n_steps)``
    returns the risk of the candidate built from the ``accepted`` moves plus ``move``,
    targeted after ``n_steps`` restarts (``move=None`` scores the accepted candidate itself).
    A move is accepted when its risk is strictly below the current candidate's.  Otherwise the
    search restarts once from the current targeted fit and accepts that pass's best move,
    whatever its risk.  Ties go to the first move in try order, where point ``CTMLE``'s
    ``_forward_path`` reads its own tie rule.  A non-finite risk counts as ``inf``: it never
    beats a finite one, and it is forced in only when every remaining move is non-finite.
    :meth:`_HazardSelector.path` runs this function with a ``score`` that builds and targets
    each candidate, so the test of the rule is a test of the fit's path.

    Parameters
    ----------
    pools : sequence of sequence of tuple
        One pool per mechanism, each a sequence of ``(mechanism, term)`` moves in try order.
    score : callable
        The risk of a move after the accepted moves, at a restart count.

    Returns
    -------
    list of tuple
        ``(move, n_steps)`` per position, starting with ``(None, 0)``.
    """
    remaining = [move for pool in pools for move in pool]
    accepted: list[tuple[str, str]] = []
    path: list[tuple[tuple[str, str] | None, int]] = [(None, 0)]
    current = _finite(score((), None, 0))
    steps = 0
    stepped = False
    while remaining:
        risks = [_finite(score(tuple(accepted), move, steps)) for move in remaining]
        best = int(np.argmin(risks))
        if not risks[best] < current and not stepped:
            steps += 1
            stepped = True
            continue
        stepped = False
        move = remaining.pop(best)
        accepted.append(move)
        path.append((move, steps))
        current = risks[best]
    return path


class _HazardSelector:
    """The collaborative selection of one fit: every arm and horizon, with their folds."""

    def __init__(
        self,
        data: LongitudinalData,
        plans: Sequence[Plan],
        initial: InitialHazards,
        *,
        outer: Folds,
        selection: Folds,
        bounds: tuple[float, float],
        alpha: float,
        max_iter: int,
        tol: float,
        penalty: bool,
        mechanisms: _Mechanisms | None = None,
    ) -> None:
        self.data = data
        self.plans = tuple(plans)
        self.initial = initial
        self.outer = outer
        self.selection = selection
        self.bounds = bounds
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.penalty = penalty
        self.mechanisms = mechanisms if mechanisms is not None else _Mechanisms(data, outer, bounds)
        covariates = self.mechanisms.covariates
        self.pools: tuple[tuple[tuple[str, str], ...], ...] = (
            tuple(("g1", name) for name in covariates),
            (("g2", TREATMENT_TERM), *(("g2", name) for name in covariates)),
        )
        ones = [np.ones((data.n, data.n_times)) for _ in range(self.mechanisms.n_levels)]
        self.base: dict[str, HazardInputs] = {
            plan.label: hazard_inputs(data, [plan], ones, alpha=alpha) for plan in self.plans
        }
        first = next(iter(self.base.values()))
        self.start = HazardState.from_hazards(
            initial.hazards, first.bounds, first.empty, first.full
        )

    # -- one candidate

    def arm(self, plan: Plan) -> int:
        """The level code a static plan assigns."""
        return int(np.asarray(plan.values)[0, 0])

    def inputs(
        self, plan: Plan, candidate: Candidate, weights: FloatArray | None = None
    ) -> HazardInputs:
        """The targeting inputs of ``plan`` at a candidate, with ``weights`` when given."""
        inputs = replace(self.base[plan.label], inverse=candidate.inverse)
        if weights is not None:
            inputs = replace(inputs, weights=weights)
        return inputs

    def target(
        self,
        plan: Plan,
        candidate: Candidate,
        start: HazardState,
        horizon: int,
        weights: FloatArray,
    ) -> tuple[HazardState, HazardFit, HazardInputs]:
        """Target arm ``plan`` alone at ``horizon`` from ``start``, fitting on ``weights``."""
        inputs = self.inputs(plan, candidate, weights)
        state, record = solve_per_horizon(
            start,
            inputs,
            horizon,
            ((plan.label, None, horizon),),
            max_iter=self.max_iter,
            tol=self.tol,
            warn=False,
        )
        return state, record, inputs

    def score(
        self,
        plan: Plan,
        candidate: Candidate,
        state: HazardState,
        horizon: int,
        weights: FloatArray,
    ) -> tuple[float, float, int]:
        """``(loss, penalty, clipped)`` of a targeted state on the rows of positive weight."""
        inputs = self.inputs(plan, candidate, weights)
        loss, clipped = selection_loss(state, inputs, horizon)
        if not self.penalty:
            return loss, 0.0, clipped
        level = self.arm(plan)
        penalty = source_penalty(
            inputs.inverse[:, :, level],
            self.start.survival[:, :, level],
            self.start.hazards[:, :, level, 0],
            weights,
            horizon,
        )
        return loss, penalty, clipped

    def position(
        self,
        plan: Plan,
        branch: str,
        train: BoolArray | None,
        terms: tuple[tuple[str, ...], tuple[str, ...]],
        move: tuple[str, str] | None,
        start: HazardState,
        horizon: int,
        weights: FloatArray,
        n_steps: int,
    ) -> _Position:
        candidate = self.mechanisms.candidate(branch, train, terms[0], terms[1])
        state, record, inputs = self.target(plan, candidate, start, horizon, weights)
        loss, penalty, clipped = self.score(plan, candidate, state, horizon, weights)
        return _Position(move, candidate, state, record, inputs, n_steps, loss, penalty, clipped)

    # -- one path

    def _ordered(self, mechanism: int, terms: set[str]) -> tuple[str, ...]:
        """A term set in try order, so one model has one cache key whatever the move order."""
        return tuple(name for _, name in self.pools[mechanism] if name in terms)

    def path(
        self, plan: Plan, horizon: int, branch: str, train: BoolArray | None
    ) -> list[_Position]:
        """The greedy sequence of ``plan`` at ``horizon`` on one branch's training rows.

        :func:`greedy_rule` decides every move; this method supplies its ``score``, which
        builds the candidate of the accepted moves plus one, targets it from the start of
        the current pass, and caches the position.  A restart count the rule has not used
        yet starts from the targeted fit of the candidate it has accepted so far.
        """
        weights = self.data.weights * (1.0 if train is None else train.astype(float))
        cache: dict[tuple[tuple[_Move, ...], _Move | None, int], _Position] = {}
        accepted_at: dict[tuple[_Move, ...], _Position] = {}
        starts: dict[int, HazardState] = {0: self.start}

        def score(accepted: tuple[_Move, ...], move: _Move | None, n_steps: int) -> float:
            if accepted and accepted not in accepted_at:
                # The rule accepted its last move in the pass that scored it.
                accepted_at[accepted] = cache[(accepted[:-1], accepted[-1], n_steps)]
            if n_steps not in starts:
                # No move lowered the risk: the current targeted fit becomes the initial
                # fit, and the search runs again from it.
                current = accepted_at[accepted] if accepted else cache[((), None, 0)]
                starts[n_steps] = current.state
            key = (accepted, move, n_steps)
            if key not in cache:
                chosen: tuple[set[str], set[str]] = (set(), set())
                for mechanism, name in (*accepted, *(() if move is None else (move,))):
                    chosen[0 if mechanism == "g1" else 1].add(name)
                terms = (self._ordered(0, chosen[0]), self._ordered(1, chosen[1]))
                cache[key] = self.position(
                    plan, branch, train, terms, move, starts[n_steps], horizon, weights, n_steps
                )
            return cache[key].risk

        rule = greedy_rule(self.pools, score)
        moves = [move for move, _ in rule[1:] if move is not None]
        return [cache[((), None, 0)]] + [
            cache[(tuple(moves[:index]), move, n_steps)]
            for index, (move, n_steps) in enumerate(rule[1:])
        ]

    def cross_validate(
        self, plan: Plan, horizon: int
    ) -> tuple[FloatArray, tuple[tuple[int, int, str], ...]]:
        """The cross-validated penalized risk of every move count, summed over the folds."""
        n_positions = sum(len(pool) for pool in self.pools) + 1
        risk = np.zeros(n_positions)
        held: list[tuple[int, int, str]] = []
        for fold, (fit_rows, _test) in enumerate(self.selection, start=1):
            train = np.zeros(self.data.n, dtype=bool)
            train[fit_rows] = True
            branch = f"cv{fold}"
            path = self.path(plan, horizon, branch, train)
            weights = self.data.weights * (~train).astype(float)
            for index, position in enumerate(path):
                loss, penalty, _ = self.score(
                    plan, position.candidate, position.state, horizon, weights
                )
                risk[index] += loss + penalty
            held.extend((fold, node, kind) for _, node, kind in path[0].candidate.held)
        return risk, tuple(sorted(set(held)))


# ------------------------------------------------------------------------ the record


@dataclass(frozen=True)
class SurvivalSelection:
    """The collaborative selection of one arm at one horizon.

    Parameters
    ----------
    arm : str
        The regimen label.
    horizon : int
        The horizon, counted from one.
    moves : tuple
        The move of each path position: ``None`` at :math:`M_0`, then ``("g1", term)`` or
        ``("g2", term)``.
    g1_terms : tuple of tuple of str
        The treatment-model covariates at each position.
    g2_terms : tuple of tuple of str
        The censoring-model terms at each position, beside the node indicators.
    n_steps : tuple of int
        The restarts the search had made when it reached each position.
    train_loss : FloatArray
        The selection loss of each position on the full data.
    train_penalty : FloatArray
        The penalty (24) of each position on the full data, zero without a penalty.
    train_risk : FloatArray
        Their sum.
    cv_risk : FloatArray
        The cross-validated penalized risk of each move count, summed over the folds.  It
        chooses :attr:`selected`.
    converged : tuple of bool
        Whether each position's targeting solve converged on the full data.
    clipped : tuple of int
        The person-nodes whose hazard the loss clip moved at each position.
    dropped_terms : tuple of int
        The terms each position's two fits aliased with earlier columns.
    selected : int
        The position the cross-validated risk chose, the first of ties.
    penalized : bool
        Whether the risk adds the penalty.
    penalty_form : str
        ``"printed_24"``: the printed (24) of the source, not the influence-curve variance.
    combination : str
        ``"sum_loss_plus_penalty"``: the loss summed over the scored rows plus the penalty.
    folds : Folds
        The realized selection split, with its origin.
    fold_held_nodes : tuple of tuple
        ``(fold, node, kind)`` of each censoring node a selection fold held rather than fitted:
        ``"no_censoring"`` at retention one, ``"all_censored"`` at retention ``1e-12``.
    covariates : tuple of str
        The baseline covariates the search drew from, in data order.
    diagnostic_horizon : int
        The horizon whose selections ``result.mechanism`` reports.
    """

    arm: str
    horizon: int
    moves: tuple[tuple[str, str] | None, ...]
    g1_terms: tuple[tuple[str, ...], ...]
    g2_terms: tuple[tuple[str, ...], ...]
    n_steps: tuple[int, ...]
    train_loss: FloatArray
    train_penalty: FloatArray
    train_risk: FloatArray
    cv_risk: FloatArray
    converged: tuple[bool, ...]
    clipped: tuple[int, ...]
    dropped_terms: tuple[int, ...]
    selected: int
    penalized: bool
    folds: Folds
    covariates: tuple[str, ...]
    diagnostic_horizon: int
    fold_held_nodes: tuple[tuple[int, int, str], ...] = ()
    penalty_form: str = PENALTY_FORM
    combination: str = COMBINATION

    @property
    def selected_terms(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """The treatment-model covariates and the censoring-model terms of the selected position."""
        return self.g1_terms[self.selected], self.g2_terms[self.selected]

    def _label(self, index: int) -> str:
        move = self.moves[index]
        return "M0" if move is None else f"+{move[1]} in {move[0]}"

    def to_frame(self, data: LongitudinalData | None = None) -> Any:
        """One row per path position.

        Parameters
        ----------
        data : LongitudinalData or None
            When given, the rows come back in that data's frame type.

        Returns
        -------
        dict or frame
            The columns ``position``, ``move``, ``g1_terms``, ``g2_terms``, ``n_steps``,
            ``train_loss``, ``train_penalty``, ``train_risk``, ``cv_risk``, ``converged``,
            ``clipped``, ``dropped_terms`` and ``selected``.
        """
        size = len(self.moves)
        payload: dict[str, Any] = {
            "position": list(range(size)),
            "move": [self._label(index) for index in range(size)],
            "g1_terms": [", ".join(terms) or "intercept" for terms in self.g1_terms],
            "g2_terms": [", ".join(terms) or "nodes only" for terms in self.g2_terms],
            "n_steps": list(self.n_steps),
            "train_loss": self.train_loss.tolist(),
            "train_penalty": self.train_penalty.tolist(),
            "train_risk": self.train_risk.tolist(),
            "cv_risk": self.cv_risk.tolist(),
            "converged": list(self.converged),
            "clipped": list(self.clipped),
            "dropped_terms": list(self.dropped_terms),
            "selected": [index == self.selected for index in range(size)],
        }
        if data is None:
            return payload
        return data.frame_like(payload)

    def summary(self, *, detail: Detail = "short") -> Summary:
        """A printable report of the selection path.

        Parameters
        ----------
        detail : {"short", "full"}, default "short"
            Accepted for symmetry with the other records; both print the same.

        Returns
        -------
        Summary
            The path, one row per position, and the chosen position.
        """
        from .._summary import Cell, Summary, SummaryTable, check_detail

        check_detail(detail)
        rows = []
        for index in range(len(self.moves)):
            g1 = ", ".join(self.g1_terms[index]) or "intercept"
            g2 = ", ".join(self.g2_terms[index]) or "nodes only"
            mark = "chosen" if index == self.selected else ""
            rows.append(
                (
                    Cell(index, str(index)),
                    Cell(g1, g1),
                    Cell(g2, g2),
                    Cell(float(self.train_risk[index]), f"{self.train_risk[index]:.3f}"),
                    Cell(float(self.cv_risk[index]), f"{self.cv_risk[index]:.3f}"),
                    Cell(mark, mark),
                )
            )
        table = SummaryTable(
            ("moves", "treatment model", "censoring model", "risk", "cv risk", ""),
            tuple(rows),
            kind="other",
        )
        chosen_g1, chosen_g2 = self.selected_terms
        header = (
            ("Arm", self.arm),
            ("Horizon", str(self.horizon)),
            ("Penalty", "printed (24)" if self.penalized else "none"),
            ("Folds", str(self.folds.n_folds)),
            ("Chosen", f"{self.selected} moves"),
            ("Treatment model", ", ".join(chosen_g1) or "intercept"),
            ("Censoring model", ", ".join(chosen_g2) or "nodes only"),
        )
        return Summary.build(
            title="Collaborative survival selection", header=header, tables=(table,)
        )

    def _repr_html_(self) -> str:
        return self.summary().as_html()

    def __repr__(self) -> str:
        g1, g2 = self.selected_terms
        return (
            f"SurvivalSelection(arm={self.arm!r}, horizon={self.horizon}, "
            f"selected={self.selected}, g1={list(g1)}, g2={list(g2)})"
        )


# ------------------------------------------------------------------------ the estimator


class SurvivalCTMLE(LTMLE):  # numpydoc ignore=PR01,PR02
    r"""Collaborative TMLE of a treatment-specific risk curve (Stitelman and van der Laan 2010).

    For each horizon and each static arm of a binary baseline treatment, the fit selects a
    main-term logistic treatment model and a main-term pooled logistic censoring model by the
    greedy collaborative search of the source, and reports the risk of the arm from that
    selection's targeted event hazard.  The module docstring of
    :mod:`cleverly.longitudinal.collaborative` states the search, the loss and the penalty.
    Every estimate is a point: ``std_error``, ``ci`` and ``pvalue`` raise, and
    ``plugin_std_error`` and ``plugin_interval`` keep the hazard influence curve at the
    selected state as a diagnostic.

    Parameters
    ----------
    regimens : mapping or sequence
        Static arms of the binary baseline treatment, such as ``{"a1": 1, "a0": 0}``.
    reference : str or None, default=None
        The regimen every contrast is taken against; the first declared by default.
    horizons : sequence of int or None, default=None
        The nodes the curve is reported at.  Each horizon runs its own selections, so naming
        the horizons you need saves time.
    outcome_learner : estimator or None, default=None
        The learner of the initial event hazard :math:`\hat Q_2`, fitted once on the full data
        and held fixed through the selection.  ``None`` uses the default Super Learner.
    n_folds : int, default=1
        The outer split.  ``1`` fits :math:`\hat Q_2` and every candidate in sample, the
        source's setting.  At ``2`` or more, :math:`\hat Q_2` and each full-data candidate are
        fitted out of fold.
    learner_folds : int, default=5
        The internal folds of a Super Learner outcome learner.
    g_bounds : float or tuple of float, default=(0.01, 1.0)
        The bound of each cumulative mechanism product, as on hazard ``LTMLE``.
    alpha : float, default=0.9995
        The probability bound of every hazard.
    alpha_sig : float, default=0.05
        The significance level of the plug-in diagnostic interval.
    max_iter : int, default=20
        Outer iterations of each targeting solve.
    tol : float, default=1e-10
        The relative-score tolerance of each targeting solve.
    selection_folds : int, default=5
        The folds of the cross-validated move count.  The source does not state its count.
    selection_split : sequence of int or None, default=None
        Fold labels ``0`` to ``V - 1`` per row, in place of the random selection split.
    penalty : bool, default=True
        Whether the risk adds the printed penalty (24) of the source.
    hazard_pooling : {"node", "pooled"}, default="node"
        How :math:`\hat Q_2` pools the nodes.  The censoring candidates always pool the nodes
        with one indicator per node.
    random_state : int or None, default=None
        Seeds the outer split, the selection split and the outcome learner.
    run_id : str or None, default=None
        A label for the run record.
    n_jobs : int, default=1
        Workers of the outcome learner.

    See Also
    --------
    cleverly.longitudinal.LTMLE : Hazard targeting with a fitted mechanism, with intervals.
    cleverly.estimators.CTMLE : Collaborative TMLE of a point-treatment mean.

    Notes
    -----
    The other keywords of :class:`~cleverly.longitudinal.LTMLE` and of point
    :class:`~cleverly.estimators.CTMLE` are accepted only at their defaults, and every other
    value raises before any learner with the reason the design records: a band, a bootstrap,
    a repeated split, a non-greedy strategy, a learner for the mechanisms, a projection and a
    targeting rule each have no source here.  ``loss="loglik"`` equals the default ``"auto"``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.linear_model import LogisticRegression
    >>> from cleverly.longitudinal import LongitudinalData, SurvivalCTMLE
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> w1, w2 = rng.normal(size=n), rng.normal(size=n)
    >>> a = rng.binomial(1, 1 / (1 + np.exp(-0.5 * w1)))
    >>> event = rng.binomial(1, 1 / (1 + np.exp(0.5 - 0.6 * w1 + 0.5 * a)))
    >>> time = np.where(event == 1, rng.integers(1, 3, size=n), 2).astype(float)
    >>> frame = pd.DataFrame({"time": time, "event": event, "A": a, "W1": w1, "W2": w2})
    >>> data = LongitudinalData.from_time_to_event(
    ...     frame, time="time", event="event", treatment="A", baseline=["W1", "W2"]
    ... )
    >>> result = SurvivalCTMLE(
    ...     {"a1": 1, "a0": 0},
    ...     reference="a0",
    ...     outcome_learner=LogisticRegression(),
    ...     selection_folds=3,
    ...     random_state=0,
    ... ).fit(data)
    >>> sorted(result.selection)
    [('a0', 1), ('a0', 2), ('a1', 1), ('a1', 2)]
    >>> result.inference_status
    'working_mechanism_plugin'
    """

    #: The support-record engine, read in place of :attr:`_estimator`, which keeps ``"tmle"``
    #: for the status and the hazard machinery.
    _support_engine: ClassVar[tuple[str, ...]] = ("collaborative_tmle", "greedy")

    def __init__(
        self,
        regimens: Any,
        *,
        reference: str | None = None,
        horizons: Sequence[int] | None = None,
        outcome_learner: Learner | None = None,
        n_folds: int = 1,
        learner_folds: int = 5,
        g_bounds: Any = DEFAULT_LTMLE_G_BOUNDS,
        alpha: float = 0.9995,
        alpha_sig: float = 0.05,
        max_iter: int = 20,
        tol: float = 1e-10,
        selection_folds: int = 5,
        selection_split: Sequence[int] | None = None,
        penalty: bool = True,
        hazard_pooling: str = "node",
        random_state: int | None = None,
        run_id: str | None = None,
        n_jobs: int = 1,
        survival_model: str = "hazard",
        targeting: str = "iterative",
        simultaneous: bool = False,
        n_multiplier: int = DEFAULT_N_REPLICATES,
        multiplier_kind: str = DEFAULT_KIND,
        band_scale: str = DEFAULT_BAND_SCALE,
        band_reference: str = DEFAULT_BAND_REFERENCE,
        repeats: int = 1,
        n_bootstrap: int = 0,
        bootstrap_resampling: Any = "auto",
        bootstrap_interval: Any = "percentile",
        stopping: Any = "relative",
        step_bound: float | None = None,
        step_search: bool = True,
        iterate: Any = "last",
        step_size: float = 1e-3,
        q_bounds: tuple[float, float] | None = None,
        msm: Any = None,
        node_bounds: Mapping[Any, Any] | None = None,
        ratio_trim: float | None = None,
        isotonic: bool = False,
        pseudo_learner: Learner | None = None,
        treatment_learner: Learner | None = None,
        censoring_learner: Learner | None = None,
        competing_learner: Learner | None = None,
        density_bins: int | None = None,
        ratio: str = "density",
        strategy: str = "greedy",
        patience: int | None = None,
        candidates: Any = None,
        ordering: Any = None,
        preorder: str | None = None,
        lambdas: Any = None,
        treatment_candidates: Any = None,
        oat_design: str | None = None,
        selection_inner_folds: int = 2,
        selection_estimand: str | None = None,
        loss: str = "auto",
        **refused: Any,
    ) -> None:
        self.selection_folds = selection_folds
        self.selection_split = None if selection_split is None else tuple(selection_split)
        self.penalty = penalty
        self.strategy = strategy
        self.patience = patience
        self.candidates = candidates
        self.ordering = ordering
        self.preorder = preorder
        self.lambdas = lambdas
        self.treatment_candidates = treatment_candidates
        self.oat_design = oat_design
        self.selection_inner_folds = selection_inner_folds
        self.selection_estimand = selection_estimand
        self.loss = loss
        self._pending: dict[tuple[str, int], SurvivalSelection] | None = None
        super().__init__(
            regimens,
            reference=reference,
            horizons=horizons,
            msm=msm,
            outcome_learner=outcome_learner,
            pseudo_learner=pseudo_learner,
            treatment_learner=treatment_learner,
            censoring_learner=censoring_learner,
            competing_learner=competing_learner,
            n_folds=n_folds,
            learner_folds=learner_folds,
            g_bounds=g_bounds,
            q_bounds=q_bounds,
            alpha=alpha,
            alpha_sig=alpha_sig,
            simultaneous=simultaneous,
            n_multiplier=n_multiplier,
            multiplier_kind=multiplier_kind,
            band_scale=band_scale,
            band_reference=band_reference,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            run_id=run_id,
            n_jobs=n_jobs,
            n_bootstrap=n_bootstrap,
            bootstrap_resampling=bootstrap_resampling,
            bootstrap_interval=bootstrap_interval,
            density_bins=density_bins,
            ratio=ratio,
            repeats=repeats,
            stopping=stopping,
            step_bound=step_bound,
            step_search=step_search,
            iterate=iterate,
            survival_model=survival_model,
            targeting=targeting,
            step_size=step_size,
            isotonic=isotonic,
            ratio_trim=ratio_trim,
            hazard_pooling=hazard_pooling,
            node_bounds=node_bounds,
            **refused,
        )

    # -- refusals

    def _validate_settings(self) -> None:
        """Refuse every setting section 1 of the design refuses, then run the shared checks."""
        refusal = self._settings_refusal()
        if refusal is not None:
            # One convention with point CTMLE: every refusal is a CapabilityError, whether it
            # names a missing implementation or redirects to another estimator.
            raise CapabilityError.from_refusal(refusal)
        if isinstance(self.selection_folds, bool) or not (
            isinstance(self.selection_folds, (int, np.integer)) and self.selection_folds >= 2
        ):
            raise ValueError(
                f"selection_folds must be an integer of at least 2; got {self.selection_folds!r}"
            )
        if not isinstance(self.penalty, bool):
            raise ValueError(f"penalty must be True or False; got {self.penalty!r}")
        if self.selection_split is not None:
            from ..estimators.ctmle import _declared_fold_count

            declared = _declared_fold_count(self.selection_split)
            if declared is not None and declared != self.selection_folds:
                raise ValueError(
                    f"selection_split= declares {declared} folds, and selection_folds= is "
                    f"{self.selection_folds}. The two must agree. Pass "
                    f"selection_folds={declared} with this split"
                )
        super()._validate_settings()

    def _settings_refusal(self) -> Refusal | None:
        """The first refused setting, in the order of the design's table."""
        if self.survival_model != "hazard":
            return SURVIVAL_MODEL_REFUSAL
        if self.targeting != "iterative":
            return ONE_STEP_REFUSAL
        if self.strategy != "greedy":
            return STRATEGY_REFUSAL
        for name in (
            "patience",
            "candidates",
            "ordering",
            "preorder",
            "lambdas",
            "treatment_candidates",
            "oat_design",
        ):
            if getattr(self, name) is not None:
                return settings_refusal(name)
        if self.selection_inner_folds != 2:
            return INNER_FOLDS_REFUSAL
        if self.selection_estimand is not None:
            return ESTIMAND_REFUSAL
        if self.loss not in ("auto", "loglik"):
            return LOSS_REFUSAL
        if (
            self.simultaneous
            or self.n_multiplier != DEFAULT_N_REPLICATES
            or self.multiplier_kind != DEFAULT_KIND
            or self.band_scale != DEFAULT_BAND_SCALE
            or self.band_reference != DEFAULT_BAND_REFERENCE
        ):
            return BAND_REFUSAL
        if (
            self.repeats != 1
            or self.n_bootstrap
            or self.bootstrap_resampling != "auto"
            or self.bootstrap_interval != "percentile"
        ):
            return BOOTSTRAP_REFUSAL
        if self.stopping != "relative" or self.step_bound is not None or self.iterate != "last":
            return RULES_REFUSAL
        if self.q_bounds is not None:
            return Q_BOUNDS_REFUSAL
        if (
            self.msm is not None
            or self.node_bounds is not None
            or self.ratio_trim is not None
            or self.isotonic
        ):
            return PROJECTION_REFUSAL
        if any(
            learner is not None
            for learner in (
                self.treatment_learner,
                self.censoring_learner,
                self.competing_learner,
                self.pseudo_learner,
            )
        ):
            return LEARNERS_REFUSAL
        if self.density_bins is not None or self.ratio != "density":
            return DOSE_SETTINGS_REFUSAL
        return None

    def _refuse_prepared(self, data: LongitudinalData) -> None:
        """Refuse a design section 1 of the design refuses, and a node every row leaves at.

        Parameters
        ----------
        data : LongitudinalData
            The prepared panel.
        """
        refusal = survival_design_refusal(data, self.regimens)
        if refusal is not None:
            raise CapabilityError.from_refusal(refusal)
        horizons = self._horizons(data)
        nodes = all_censored_nodes(data, max(horizons))
        if nodes:
            grid = data.time_grid
            time = float(grid[nodes[0] - 1]) if grid is not None else float(nodes[0])
            raise CapabilityError.from_refusal(all_censored_refusal(nodes[0], time))

    # -- the selection

    def _selection_partition(self, data: LongitudinalData) -> Folds:
        """The selection split: the declared labels, or one draw from ``random_state``."""
        if self.selection_split is not None:
            from ..estimators.ctmle import _declared_split

            return _declared_split(self.selection_split, data.n)
        return make_folds(data.n, self.selection_folds, random_state=self.random_state)

    def _hazard_block(
        self,
        prepared: LongitudinalData,
        plans: tuple[Plan, ...],
        levels: tuple[Plan, ...],
        folds: Folds,
        *,
        learners: tuple[Learner, Learner, Learner, Learner, Learner | None],
        horizons: tuple[int, ...],
        bounds: tuple[float, float],
        targeting: dict[str, Any],
        replay_recipe: _LongitudinalReplayRecipe,
        node_bounds: dict[str | None, dict[int, NodeBound]] | None,
        separable: tuple[Any, ...],
    ) -> _HazardBlock:
        """Fit :math:`\\hat Q_2` once, select each arm's mechanisms at each horizon, and report.

        No Super Learner mechanism is fitted.  The reported mechanism is assembled from the
        selections at the last horizon (:func:`assemble_mechanism`).

        Parameters
        ----------
        prepared : LongitudinalData
            The prepared panel.
        plans : tuple of Plan
            The static plans.
        levels : tuple of Plan
            One static plan per treatment level.
        folds : Folds
            The realized outer split.
        learners : tuple
            The resolved learners; only the outcome learner is read.
        horizons : tuple of int
            The reported horizons.
        bounds : tuple of float
            The cumulative mechanism bounds.
        targeting : dict
            The fluctuation settings.
        replay_recipe : _LongitudinalReplayRecipe
            The draw's replay recipe.
        node_bounds : dict or None
            Always ``None``: the constructor refuses node bounds.
        separable : tuple
            Always empty: the design refusal refuses a separable regimen.

        Returns
        -------
        _HazardBlock
            The assembled mechanism, the fits, the initial hazards and the recipe.
        """
        initial = fit_hazards(
            prepared,
            levels,
            learner=learners[0],
            folds=folds,
            n_jobs=self.n_jobs,
            hazard_pooling=self.hazard_pooling,
        )
        selector = _HazardSelector(
            prepared,
            plans,
            initial,
            outer=folds,
            selection=self._selection_partition(prepared),
            bounds=bounds,
            alpha=targeting["alpha"],
            max_iter=targeting["max_iter"],
            tol=targeting["tol"],
            penalty=self.penalty,
        )
        fits, selections, positions = run_selection(selector, horizons)
        self._pending = selections
        mechanism = assemble_mechanism(prepared, plans, positions, horizons[-1], selector)
        replay_recipe = replace(
            replay_recipe,
            omissions=(),
            hazard=_HazardReplay(
                _frozen_plans(levels),
                "per_horizon",
                float(self.step_size),
                targeting["alpha"],
                targeting["max_iter"],
                targeting["tol"],
                (),
            ),
        )
        return _HazardBlock(mechanism, fits, initial, replay_recipe, plans, True)

    def _draw_status(
        self,
        prepared: LongitudinalData,
        folds: Folds,
        parametric: bool,
        mechanism: Mechanism,
    ) -> InferenceStatus:
        """``"working_mechanism_plugin"``: the source states no theorem (F18).

        Parameters
        ----------
        prepared : LongitudinalData
            The prepared panel.
        folds : Folds
            The realized outer split.
        parametric : bool
            Not read.
        mechanism : Mechanism
            Not read: no trim is accepted.

        Returns
        -------
        str
            The working-mechanism status, unless an earlier withheld status applies.
        """
        return precedent_status(
            (
                super()._draw_status(prepared, folds, parametric, mechanism),
                "working_mechanism_plugin",
            )
        )

    def _finish(
        self,
        result: LongitudinalResult,
        reported: _Reported,
        fits: Mapping[str, RegimenFit],
    ) -> LongitudinalResult:
        """Attach the selection records.

        Parameters
        ----------
        result : LongitudinalResult
            The result before any bootstrap.
        reported : _Reported
            The names, estimates and contributing fits.
        fits : mapping of str to RegimenFit
            The fits of the selection.

        Returns
        -------
        LongitudinalResult
            The result with :attr:`~cleverly.longitudinal.LongitudinalResult.selection` set.
        """
        selections, self._pending = self._pending, None
        return replace(result, selection=selections)

    def _refit_template(self) -> SurvivalCTMLE:
        """The configuration a result keeps, without the pending selection of this fit.

        Returns
        -------
        SurvivalCTMLE
            The template.
        """
        template = super()._refit_template()
        assert isinstance(template, SurvivalCTMLE)
        template._pending = None
        return template


def run_selection(
    selector: _HazardSelector, horizons: Sequence[int]
) -> tuple[
    dict[str, RegimenFit],
    dict[tuple[str, int], SurvivalSelection],
    dict[tuple[str, int], _Position],
]:
    """Run every arm's selection at every horizon, and report each selected targeted fit.

    Parameters
    ----------
    selector : _HazardSelector
        The selector of the fit.
    horizons : sequence of int
        The reported horizons.

    Returns
    -------
    tuple
        The fits keyed by ``_fit_key``, the selection records keyed by ``(label, horizon)``,
        and the selected full-data position of each.
    """
    data = selector.data
    fits: dict[str, RegimenFit] = {}
    records: dict[tuple[str, int], SurvivalSelection] = {}
    positions: dict[tuple[str, int], _Position] = {}
    diagnostic = int(horizons[-1])
    unconverged: list[str] = []
    glm_unconverged = 0
    by_arm: dict[
        tuple[int, int], tuple[list[_Position], FloatArray, tuple[tuple[int, int, str], ...]]
    ] = {}
    for plan in selector.plans:
        arm = selector.arm(plan)
        for t in horizons:
            key = (arm, int(t))
            if key not in by_arm:
                path = selector.path(plan, t, "full", None)
                cv_risk, held = selector.cross_validate(plan, t)
                by_arm[key] = (path, cv_risk, held)
            path, cv_risk, held = by_arm[key]
            chosen = _finite_argmin(cv_risk)
            if chosen < 0:
                raise LongitudinalError(
                    f"every cross-validated risk of {plan.label} at horizon {t} is not finite, "
                    "so no move count can be selected. Check the overlap of the arms and of "
                    "censoring, or bind g_bounds further from zero"
                )
            position = path[chosen]
            positions[(plan.label, int(t))] = position
            glm_unconverged += sum(not p.candidate.converged for p in path)
            if not position.record.converged:
                unconverged.append(f"{plan.label} at horizon {t}")
            records[(plan.label, int(t))] = SurvivalSelection(
                arm=plan.label,
                horizon=int(t),
                moves=tuple(p.move for p in path),
                g1_terms=tuple(p.candidate.g1_terms for p in path),
                g2_terms=tuple(p.candidate.g2_terms for p in path),
                n_steps=tuple(p.n_steps for p in path),
                train_loss=np.array([p.loss for p in path]),
                train_penalty=np.array([p.penalty for p in path]),
                train_risk=np.array([p.risk for p in path]),
                cv_risk=np.asarray(cv_risk, dtype=float),
                converged=tuple(bool(p.record.converged) for p in path),
                clipped=tuple(p.clipped for p in path),
                dropped_terms=tuple(p.candidate.dropped for p in path),
                selected=chosen,
                penalized=selector.penalty,
                folds=selector.selection,
                covariates=selector.mechanisms.covariates,
                diagnostic_horizon=diagnostic,
                fold_held_nodes=held,
            )
            fits[_fit_key(plan.label, None, int(t), True)] = _regimen_fit(
                data, plan, position, int(t)
            )
    if glm_unconverged:
        _warn(
            f"{glm_unconverged} candidate mechanism fit(s) of the collaborative search did not "
            "converge or reached separation; each used its last Newton iterate, bounded by "
            "g_bounds. Fewer or binarized covariates usually help",
            ConvergenceWarning,
            stacklevel=4,
        )
    if unconverged:
        _warn(
            "the hazard targeting of the selected candidate did not converge for "
            f"{', '.join(unconverged)}. Raise max_iter, or check result.selection for a "
            "positivity problem",
            ConvergenceWarning,
            stacklevel=4,
        )
    return fits, records, positions


def _regimen_fit(
    data: LongitudinalData, plan: Plan, position: _Position, horizon: int
) -> RegimenFit:
    """The fit record of one arm at one horizon, from its selected targeted state."""
    inputs = position.inputs
    # The scores and the curve are read on every row, whatever rows the solve was fitted on.
    inputs = replace(inputs, weights=np.asarray(data.weights, dtype=float))
    levels, curves = influence(position.state, inputs, (horizon,))
    rows = np.arange(data.n)
    observed = inputs.observed
    candidate = position.candidate
    unbounded = np.stack(candidate.unbounded, axis=2)[rows, :, observed]
    bounded = np.stack(candidate.cumulative, axis=2)[rows, :, observed]
    q = inputs.q_observed()
    clever = np.where(inputs.at_risk, q[0][:, None] * inputs.inverse_observed(), 0.0)
    clever[:, horizon:] = 0.0
    return RegimenFit(
        regimen=plan.regimen,
        psi_scaled=float(levels[0, 0, 0]),
        influence_curve_scaled=np.asarray(curves[0, 0, 0], dtype=float),
        horizon=horizon,
        cause=None,
        steps=(),
        cumulative_unbounded=unbounded,
        cumulative=bounded,
        assignment=np.asarray(plan.values),
        obs_weights=np.asarray(data.weights, dtype=float),
        policy=plan.policy,
        hazard=position.record,
        hazard_clever=clever,
        hazard_constant_nodes=inputs.constant_nodes(0, horizon),
    )


def assemble_mechanism(
    data: LongitudinalData,
    plans: Sequence[Plan],
    positions: Mapping[tuple[str, int], _Position],
    horizon: int,
    selector: _HazardSelector,
) -> Mechanism:
    """The mechanism of the selections at the diagnostic horizon, for the diagnostics.

    ``treatment[0][label]`` and ``censoring[k][label]`` of a regimen come from its arm's
    selected candidate at its level, and the later treatment nodes hold ones, as on every held
    design.  Each level's internal label reads the selection of a regimen at that level, or the
    first regimen's candidate evaluated at that level when no regimen assigns it.  The observed
    arrays read, row by row, the selection of the level the row received.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.
    plans : sequence of Plan
        The static plans.
    positions : mapping
        The selected full-data position of each ``(label, horizon)``.
    horizon : int
        The diagnostic horizon, the last reported one.
    selector : _HazardSelector
        The selector, for the held nodes of the full data.

    Returns
    -------
    Mechanism
        The assembled mechanism.
    """
    n, n_times = data.n, data.n_times
    n_levels = selector.mechanisms.n_levels
    chosen: dict[str, Candidate] = {
        plan.label: positions[(plan.label, horizon)].candidate for plan in plans
    }
    by_level: dict[int, Candidate] = {}
    for plan in plans:
        by_level.setdefault(selector.arm(plan), chosen[plan.label])
    first = chosen[plans[0].label]
    for level in range(n_levels):
        by_level.setdefault(level, first)
    labels: list[tuple[str, int, Candidate]] = [
        (plan.label, selector.arm(plan), chosen[plan.label]) for plan in plans
    ]
    labels += [(level_label(level), level, by_level[level]) for level in range(n_levels)]
    treatment: list[dict[str, FloatArray]] = []
    censoring: list[dict[str, FloatArray]] = []
    observed = selector.mechanisms.observed
    rows = np.arange(n)
    treatment_observed: list[FloatArray] = []
    censoring_observed: list[FloatArray] = []
    for k in range(n_times):
        if k == 0:
            treatment.append({label: c.treatment[:, level] for label, level, c in labels})
            matrix = np.empty((n, n_levels))
            for level in range(n_levels):
                received = observed == level
                matrix[received] = by_level[level].treatment[received]
            treatment_observed.append(matrix)
        else:
            treatment.append({label: np.ones(n) for label, _, _ in labels})
            treatment_observed.append(np.zeros((n, 0)))
        censoring.append({label: c.retention[:, k, level] for label, level, c in labels})
        if data.censoring_names:
            retained = np.empty(n)
            for level in range(n_levels):
                received = observed == level
                retained[received] = by_level[level].retention[rows[received], k, level]
            censoring_observed.append(retained)
    held = first.held
    no_censoring_nodes = tuple(
        sorted({node for fold, node, kind in held if fold == 0 and kind == NO_CENSORING})
    )
    no_censoring_folds: dict[int, tuple[int, ...]] = {}
    for fold, node, kind in held:
        if fold > 0 and kind == NO_CENSORING:
            no_censoring_folds[node] = tuple(sorted({*no_censoring_folds.get(node, ()), fold}))
    return Mechanism(
        treatment=tuple(treatment),
        censoring=tuple(censoring),
        treatment_observed=tuple(treatment_observed),
        censoring_observed=tuple(censoring_observed),
        treatment_diagnostics=tuple(() for _ in range(n_times)),
        censoring_diagnostics=tuple(() for _ in range(len(censoring_observed))),
        no_censoring_nodes=no_censoring_nodes,
        no_censoring_folds=no_censoring_folds,
    )


#: Exposed for the full-set check: the candidate builder and the selector.
CandidateBuilder = _Mechanisms
Selector = _HazardSelector
