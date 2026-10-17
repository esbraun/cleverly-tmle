r"""Hazard-based targeting of a survival curve under a held baseline treatment.

The iterated-mean recursion of :mod:`cleverly.longitudinal.sequential` reports each
horizon from its own backward pass, so its curve can decrease.  This module models the
curve through its discrete cause-specific hazards instead,

.. math::

    \lambda_l(k \mid a, W) = P(\text{event of cause } l \text{ at } k \mid
        \text{uncensored through } k,\ \text{event-free before } k,\ A = a,\ W),

with :math:`S(k \mid a, W) = \prod_{s \le k}(1 - \sum_l \lambda_l(s \mid a, W))` and
:math:`F_j(t \mid a, W) = \sum_{k \le t} S(k - 1 \mid a, W)\,\lambda_j(k \mid a, W)`.  The
parameter of regimen :math:`r` and cause :math:`j` at horizon :math:`t` is
:math:`\psi_{r,j}(t) = E \sum_a q_r(a \mid W) F_j(t \mid a, W)`, where :math:`q_r` is a
static arm, a known rule or a known policy at the one baseline decision.

The efficient influence function (Moore and van der Laan 2009, Section 4.1, (11)-(16);
Rytgaard and van der Laan 2024, Section 3.1) is

.. math::

    D_{r,j,t} = \sum_{k \le t} \sum_l h^{r,j,t}_l(k, A, W)\, R_k\,
        \bigl(dN_l(k) - \lambda_l(k \mid A, W)\bigr)
        + \sum_a q_r(a \mid W) F_j(t \mid a, W) - \psi_{r,j}(t),

    h^{r,j,t}_l(k, a, W) = \frac{q_r(a \mid W)}{g(a \mid W)\, G_k(a, W)}
        \Bigl[1\{l = j\} - \frac{F_j(t \mid a, W) - F_j(k \mid a, W)}{S(k \mid a, W)}\Bigr],

with :math:`R_k` the indicator of being uncensored through :math:`k` and event-free before
:math:`k`, and :math:`G_k = \prod_{s \le k} c_s` the retention through node :math:`k`.  A
unit is censored at node :math:`k` before its event there can be seen, which is the
convention of :class:`~cleverly.longitudinal.data.LongitudinalData`.

The hazards are held as softmax logits :math:`\eta_l = \log(\lambda_l / \lambda_0)`, with
:math:`\lambda_0` the probability of no event.  The fluctuation is the multinomial
logistic submodel :math:`\eta_l(\epsilon) = \eta_l + \sum_m \epsilon_m h^m_l`, which at one
cause is Moore's (10).  Each logit is clipped into
:math:`[\operatorname{logit}(1 - \alpha), \operatorname{logit}(\alpha)]` after every
update, so :math:`\sum_l \lambda_l < 1` always holds and nothing is renormalised.

Two algorithms target the hazards.

``per_horizon`` (Moore and van der Laan 2009, Section 4.1; ``survtmle``
``method = "hazard"``) targets every regimen's and cause's parameter at one horizon by
Newton steps on the fluctuation, rebuilding :math:`h` after each, and restarts from the
initial hazards at the next horizon.

``one_step`` (Cai and van der Laan 2020, Section 5, Algorithm 2; Rytgaard and van der Laan
2024, Section 3.4) walks one set of hazards along the universal least favorable submodel
until every target's score is solved.  The direction is
:math:`\Sigma^{-1} s / \lVert s \rVert_\Sigma` with :math:`s = P_n D^{(1)}` and the fixed
:math:`\Sigma = \operatorname{diag}(P_n D_m^2)` of the initial fit (Rytgaard and van der Laan
2024, Remark 2).  Every horizon then reads one set of hazards, so each curve is monotone.

Every plug-in mean over the horizons reduces its column in one value-independent order shared
by every horizon.  The incidence :math:`F_j(t)` of each unit is a running sum of non-negative
terms, a policy mixture of such sums is a sum of non-negative products, and a weighted mean
of non-decreasing columns reduced in one order is non-decreasing, because rounding to
nearest is monotone.  So a one-step curve is non-decreasing in floating point with no
tolerance, and ``tests/unit/test_survival_monotone.py`` pins it.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.special import xlogy

from .._typing import BoolArray, FloatArray, IntArray, Learner
from ..estimators._nuisance import cross_fit_predictions
from ..exceptions import (
    CapabilityError,
    ConvergenceWarning,
    Refusal,
    RefusalReason,
)
from ..fluctuation._score import relative_score
from ..fluctuation.iterative import TargetingFailure
from ..learners.crossfit import Folds
from .data import LongitudinalData
from .pooled_hazards import refuse_pooled_designs, stacked_design
from .regimen import Plan
from .sequential import RegimenFit

__all__ = [
    "SIGMA_FLOOR",
    "HazardAlgorithm",
    "HazardFit",
    "HazardInputs",
    "HazardState",
    "InitialHazards",
    "fit_hazards",
    "hazard_inputs",
    "hazard_rules_refusal",
    "project_curves",
    "refuse_hazard_design",
    "refuse_hazard_settings",
    "solve_per_horizon",
    "target_regimens",
    "walk_curve",
]

#: The two hazard targeting algorithms.
HazardAlgorithm = Literal["per_horizon", "one_step"]

#: The roadmap item that tracks hazard targeting outside its held baseline design.
HAZARD_ROADMAP = "X70"

#: The roadmap item that tracks the targeting rules (``iterate=``, ``stopping=`` and
#: ``step_bound=``) on the hazard walk.
HAZARD_RULES_ROADMAP = "X33"

#: The variance floor of a one-step target whose initial influence curve is zero, so that
#: the fixed metric :math:`\Sigma` stays invertible.
SIGMA_FLOOR = 1e-12

#: Inner Newton iterations of one per-horizon fluctuation, with :math:`h` held fixed.
_NEWTON_ITER = 50

#: The logit increment of the forward differences behind a per-horizon Newton step.
_DIFFERENCE = 1e-6

#: The shortest one-step step before the walk gives up, as in ``solve_one_step``.
_MIN_STEP = 1e-14

#: The log-likelihood tolerance of the walk.  A finite Euler step carries an
#: :math:`O(dx^2)` term, so the accepted log-likelihood is non-decreasing only to within it.
LOGLIK_TOLERANCE = 1e-8


@dataclass(frozen=True)
class InitialHazards:
    """The initial cause-specific hazards at every level of the baseline treatment.

    Parameters
    ----------
    hazards : FloatArray
        ``(n, T, L, J)`` initial hazards :math:`\\lambda_l(k \\mid a, W)`, by unit, node,
        level and cause.
    no_event_nodes : tuple of tuple of int
        ``(node, cause index)`` pairs whose training rows held no event, so the fit
        predicts zero there and fits no learner.
    diagnostics : tuple
        Super Learner diagnostics per node and cause, in fit order.
    """

    hazards: FloatArray
    no_event_nodes: tuple[tuple[int, int], ...] = ()
    diagnostics: tuple[tuple[object, ...], ...] = ()


@dataclass(frozen=True)
class HazardInputs:
    """Everything the targeting reads besides the hazards themselves.

    Parameters
    ----------
    at_risk : BoolArray
        ``(n, T)`` :math:`R_k`: uncensored through ``k`` and event-free before ``k``.
    events : FloatArray
        ``(n, T, J)`` :math:`dN_l(k)`, the event of each cause at each node, on the rows at
        risk and zero elsewhere.
    observed : IntArray
        ``(n,)`` dense code of the baseline treatment each unit received.
    inverse : FloatArray
        ``(n, T, L)`` :math:`1 / (g(a \\mid W) G_k(a, W))` at every level, after
        ``g_bounds``.
    q : FloatArray
        ``(R, n, L)`` intervention probability :math:`q_r(a \\mid W)` of each regimen.
    weights : FloatArray
        ``(n,)`` observation weights, mean one.
    alpha : float
        The probability bound; each softmax logit is clipped into
        ``[logit(1 - alpha), logit(alpha)]``.
    empty : BoolArray
        ``(T, L, J)`` cells whose rows at risk with that observed level, of positive weight,
        exist and hold no event of that cause.  The hazard there is held at zero.
    full : BoolArray
        ``(T, L)`` cells whose rows at risk with that observed level, of positive weight,
        exist and all hold an event.  The all-cause hazard there is held at one.
    """

    at_risk: BoolArray
    events: FloatArray
    observed: IntArray
    inverse: FloatArray
    q: FloatArray
    weights: FloatArray
    alpha: float
    empty: BoolArray
    full: BoolArray

    @property
    def n(self) -> int:
        return int(self.at_risk.shape[0])

    @property
    def n_times(self) -> int:
        return int(self.at_risk.shape[1])

    @property
    def n_levels(self) -> int:
        return int(self.inverse.shape[2])

    @property
    def n_causes(self) -> int:
        return int(self.events.shape[2])

    @property
    def n_regimens(self) -> int:
        return int(self.q.shape[0])

    @property
    def bounds(self) -> tuple[float, float]:
        """The clip interval of every softmax logit."""
        upper = float(np.log(self.alpha / (1.0 - self.alpha)))
        return -upper, upper

    @property
    def rows(self) -> IntArray:
        return np.arange(self.n)

    def q_observed(self) -> FloatArray:
        """``(R, n)`` :math:`q_r(A \\mid W)` at the observed level."""
        return np.asarray(self.q[:, self.rows, self.observed], dtype=float)

    def inverse_observed(self) -> FloatArray:
        """``(n, T)`` the inverse mechanism at the observed level."""
        return np.asarray(self.inverse[self.rows, :, self.observed], dtype=float)

    def constant_nodes(self, regimen: int, horizon: int) -> tuple[int, ...]:
        """The nodes up to ``horizon`` with an empty or full cell at a level ``regimen`` reads.

        A regimen reads a level when it puts mass on that level for some unit.  The
        influence curve term of such a cell is identically zero, which is the case
        :func:`~cleverly.longitudinal.sequential.constant_nodes` reports.
        """
        read = np.any(self.q[regimen] > 0.0, axis=0)
        flagged = (self.empty.any(axis=2) | self.full)[:horizon][:, read].any(axis=1)
        return tuple(int(k) + 1 for k in np.flatnonzero(flagged))


@dataclass(frozen=True)
class HazardState:
    """One set of hazards, held as clipped softmax logits.

    Parameters
    ----------
    logits : FloatArray
        ``(n, T, L, J)`` :math:`\\eta_l = \\log(\\lambda_l / \\lambda_0)`.
    full : BoolArray or None
        ``(T, L)`` cells whose no-event probability :math:`\\lambda_0` is held at exactly
        zero.  The logits there split the event among the causes.
    """

    logits: FloatArray
    full: BoolArray | None = None
    hazards: FloatArray = field(init=False, repr=False)
    survival: FloatArray = field(init=False, repr=False)
    incidence: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        eta = np.asarray(self.logits, dtype=float)
        expo = np.exp(eta)
        events = expo.sum(axis=-1)
        if self.full is None:
            denominator = 1.0 + events
            none = 1.0 / denominator
        else:
            held = np.asarray(self.full, dtype=bool)[None]
            denominator = np.where(held, events, 1.0 + events)
            none = np.where(held, 0.0, 1.0 / denominator)
        hazards = expo / denominator[..., None]
        survival = np.cumprod(none, axis=1)
        before = np.concatenate([np.ones_like(survival[:, :1]), survival[:, :-1]], axis=1)
        # A running sum of non-negative terms, so every unit's incidence is non-decreasing
        # over the nodes in floating point.
        incidence = np.cumsum(before[..., None] * hazards, axis=1)
        object.__setattr__(self, "hazards", hazards)
        object.__setattr__(self, "survival", survival)
        object.__setattr__(self, "incidence", incidence)

    @classmethod
    def from_hazards(
        cls,
        hazards: FloatArray,
        bounds: tuple[float, float],
        empty: BoolArray | None = None,
        full: BoolArray | None = None,
    ) -> HazardState:
        """The state of ``(n, T, L, J)`` hazards, with each logit clipped into ``bounds``.

        A cell of ``empty``, ``(T, L, J)``, holds the hazard at exactly zero for every unit,
        its logit at :math:`-\\infty`.  A cell of ``full``, ``(T, L)``, holds the all-cause
        hazard at exactly one for every unit.  No fluctuation moves either.
        """
        lam = np.asarray(hazards, dtype=float)
        none = 1.0 - lam.sum(axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            eta = np.log(lam) - np.log(none)[..., None]
        eta = np.clip(
            np.nan_to_num(eta, nan=bounds[0], posinf=bounds[1], neginf=bounds[0]), *bounds
        )
        if empty is not None:
            eta = np.where(np.asarray(empty, dtype=bool)[None], -np.inf, eta)
        return cls(eta, full)

    def moved(self, delta: FloatArray, bounds: tuple[float, float]) -> HazardState:
        """The state after adding ``delta`` to every logit, then clipping.

        A logit at :math:`-\\infty`, a hazard held at zero, stays there, and so does a held
        all-cause hazard of one.
        """
        return HazardState(
            np.where(np.isneginf(self.logits), -np.inf, np.clip(self.logits + delta, *bounds)),
            self.full,
        )

    def levels(self, q: FloatArray, weights: FloatArray) -> FloatArray:
        """``(R, J, T)`` plug-in incidence of every regimen and cause at every node.

        One reduction per regimen and cause, over an ``(n, T)`` array in one order shared by
        every horizon, which keeps a curve non-decreasing in floating point.
        """
        out = np.empty((q.shape[0], self.incidence.shape[3], self.incidence.shape[1]))
        for r in range(q.shape[0]):
            for j in range(self.incidence.shape[3]):
                out[r, j] = np.average(self.mixture(q[r], j), axis=0, weights=weights)
        return out

    def mixture(self, q: FloatArray, cause: int) -> FloatArray:
        """``(n, T)`` :math:`\\sum_a q(a \\mid W) F_j(\\cdot \\mid a, W)`, summed in level order."""
        total = q[:, 0, None] * self.incidence[:, :, 0, cause]
        for level in range(1, q.shape[1]):
            total = total + q[:, level, None] * self.incidence[:, :, level, cause]
        return np.asarray(total, dtype=float)


@dataclass(frozen=True)
class HazardFit:
    """The targeting record of a hazard fit.

    A one-step fit has one record shared by every reported parameter.  A per-horizon fit
    has one record per horizon.

    Parameters
    ----------
    algorithm : {"per_horizon", "one_step"}
        The targeting algorithm.
    targets : tuple of tuple
        ``(regimen, cause, horizon)`` of every target, in score order.
    converged : bool
        Whether every target's relative score reached ``tol``.
    n_iter : int
        Outer iterations of a per-horizon fit, or accepted steps of a one-step walk.
    score_initial : FloatArray
        The score of every target at the initial hazards.
    score : FloatArray
        The score of every target at the reported hazards.
    score_scale : FloatArray
        The largest each score could be, ``mean(|w| sum_l |h_l|)`` over the rows at risk.
    trace : tuple of float
        The relative score after each iteration or accepted step, starting at the initial
        fit.
    initial_levels : FloatArray
        The plug-in value of every target at the initial hazards.
    sigma : FloatArray or None
        The fixed diagonal metric of a one-step walk.
    norm_trace : tuple of float
        The accepted :math:`\\Sigma`-norm of the score of a one-step walk.
    loglik_trace : tuple of float
        The accepted log-likelihood of a one-step walk.
    failure : str or None
        Why the targeting stopped short, ``None`` when it converged.
    tol : float
        The relative-score tolerance.
    """

    algorithm: HazardAlgorithm
    targets: tuple[tuple[str, str | None, int], ...]
    converged: bool
    n_iter: int
    score_initial: FloatArray
    score: FloatArray
    score_scale: FloatArray
    trace: tuple[float, ...]
    initial_levels: FloatArray
    sigma: FloatArray | None = None
    norm_trace: tuple[float, ...] = ()
    loglik_trace: tuple[float, ...] = ()
    failure: TargetingFailure | None = None
    tol: float = 0.0

    @property
    def relative_score(self) -> float:
        """The largest target score relative to its scale, at the reported hazards."""
        return relative_score(self.score, self.score_scale)


# ----------------------------------------------------------------------- nuisances


def level_plans(data: LongitudinalData) -> tuple[Plan, ...]:
    """One static plan per level of the baseline treatment, for the mechanism at each level.

    Each plan holds its level's code at every node, as the static plan of that arm does on
    a held design.  Its label cannot clash with a regimen's, since a regimen label never
    starts with a NUL character.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel of a held design.

    Returns
    -------
    tuple of Plan
        One plan per level, in code order.
    """
    from .regimen import Regimen

    out = []
    for code, level in enumerate(data.treatment_levels[0]):
        regimen = Regimen(level_label(code), (level,) * data.n_decisions)
        out.append(Plan(regimen, np.full((data.n, data.n_times), float(code))))
    return tuple(out)


def level_label(code: int) -> str:
    """The internal label of the mechanism evaluated at level ``code``."""
    return f"\x00hazard-level-{code}"


def regimen_weights(data: LongitudinalData, plans: Sequence[Plan]) -> FloatArray:
    """``(R, n, L)`` :math:`q_r(a \\mid W)` of every regimen at the baseline decision.

    A static arm and a rule are one-hot at the arm they assign; a policy is its evaluated
    density.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel of a held design.
    plans : sequence of Plan
        The regimens' plans.

    Returns
    -------
    FloatArray
        One ``(n, L)`` matrix per regimen.
    """
    n_levels = len(data.treatment_levels[0])
    out = np.zeros((len(plans), data.n, n_levels))
    for r, plan in enumerate(plans):
        if plan.is_policy_node(1):
            out[r] = np.asarray(plan.policy_at(1), dtype=float)
        else:
            codes = np.asarray(plan.values[:, 0], dtype=np.int64)
            out[r, np.arange(data.n), codes] = 1.0
    return out


def event_indicators(data: LongitudinalData) -> tuple[BoolArray, FloatArray]:
    """``R_k`` and ``dN_l(k)`` of every unit, node and cause.

    Parameters
    ----------
    data : LongitudinalData
        A survival panel.

    Returns
    -------
    tuple
        ``(n, T)`` at-risk indicator and ``(n, T, J)`` event indicator, zero off the risk
        set.
    """
    assert data.event is not None
    n, n_times = data.n, data.n_times
    free = np.ones((n, n_times + 1), dtype=bool)
    free[:, 1:] = ~data.event
    uncensored = np.cumprod(data.uncensored, axis=1).astype(bool)
    at_risk = uncensored & free[:, :-1]
    cumulative = data.event[:, :, None] if data.cause_event is None else data.cause_event
    previous = np.concatenate(
        [np.zeros((n, 1, cumulative.shape[2]), dtype=bool), cumulative[:, :-1]], axis=1
    )
    events = (cumulative & ~previous & at_risk[:, :, None]).astype(float)
    return at_risk, events


def fit_hazards(
    data: LongitudinalData,
    levels: Sequence[Plan],
    *,
    learner: Learner,
    folds: Folds,
    n_jobs: int = 1,
    hazard_pooling: str = "node",
) -> InitialHazards:
    """Fit the sequential pseudo-hazards at every node and cause, out of fold.

    Cause :math:`j` at node :math:`k` regresses its event on the rows at risk that had no
    event of an earlier cause at :math:`k`, which is ``survtmle``'s
    :math:`\\rho_j`; then :math:`\\lambda_j = \\rho_j (1 - \\sum_{i < j} \\lambda_i)`
    (``hazards_estimate.R``).  The design is the node's mechanism history, the baseline
    covariates and the baseline treatment, and every level is predicted for every unit.

    Parameters
    ----------
    data : LongitudinalData
        A survival panel of a held design with no covariate after baseline.
    levels : sequence of Plan
        One plan per treatment level, from :func:`level_plans`.
    learner : object
        The hazard learner, a classifier.
    folds : Folds
        The outer split.
    n_jobs : int
        Worker count.
    hazard_pooling : {"node", "pooled"}
        ``"pooled"`` fits each cause's :math:`\\rho_j` once over the eligible rows of every
        node, with the node's grid time as a numeric column, as ``survtmle``'s ``glm.ftime``
        reads ``t``.  A unit keeps its fold at every node.  A cause with no event at any node
        pools nothing and has zero hazard.

    Returns
    -------
    InitialHazards
        The hazards and the cells that held no event.  Under ``"pooled"`` a cause with no event
        lists every node, and a node without an event of a cause is a row of the pool.
    """
    at_risk, events = event_indicators(data)
    n_causes = events.shape[2]
    if hazard_pooling == "pooled":
        return _fit_pooled_hazards(
            data, levels, at_risk, events, learner=learner, folds=folds, n_jobs=n_jobs
        )
    hazards = np.zeros((data.n, data.n_times, len(levels), n_causes))
    no_events: list[tuple[int, int]] = []
    diagnostics: list[tuple[object, ...]] = []
    for time in range(1, data.n_times + 1):
        design = data.history_design(time, include_current=True)
        predict = {
            plan.label: data.history_design(time, treatment=plan.values, include_current=True)
            for plan in levels
        }
        earlier = np.zeros(data.n, dtype=bool)
        taken = np.zeros((data.n, len(levels)))
        for cause in range(n_causes):
            rows = at_risk[:, time - 1] & ~earlier
            target = events[:, time - 1, cause]
            if not np.any(target[rows] * data.weights[rows] > 0):
                no_events.append((time, cause))
                diagnostics.append(())
                rho = np.zeros((data.n, len(levels)))
            else:
                constant: list[int] = []
                predictions, fold_diagnostics = cross_fit_predictions(
                    learner,
                    design,
                    target,
                    data.weights,
                    folds,
                    task="classification",
                    predict_designs=predict,
                    fit_mask=rows,
                    groups=data.cluster,
                    clip=(0.0, 1.0),
                    n_jobs=n_jobs,
                    constant_folds=constant,
                )
                diagnostics.append(tuple(fold_diagnostics))
                rho = np.column_stack([predictions[plan.label] for plan in levels])
            hazards[:, time - 1, :, cause] = rho * (1.0 - taken)
            taken = taken + hazards[:, time - 1, :, cause]
            earlier = earlier | (target > 0)
    return InitialHazards(hazards, tuple(no_events), tuple(diagnostics))


def _fit_pooled_hazards(
    data: LongitudinalData,
    levels: Sequence[Plan],
    at_risk: BoolArray,
    events: FloatArray,
    *,
    learner: Learner,
    folds: Folds,
    n_jobs: int,
) -> InitialHazards:
    """One regression per cause over the eligible rows of every node."""
    n, n_times, n_causes = data.n, data.n_times, events.shape[2]
    nodes = tuple(range(1, n_times + 1))
    refuse_pooled_designs(data)
    stacked_folds = Folds(np.tile(folds.assignment, n_times), folds.n_folds)
    groups = None if data.cluster is None else np.tile(data.cluster, n_times)
    weights = np.tile(data.weights, n_times)
    designs = {plan.label: stacked_design(data, nodes, plan) for plan in levels}
    observed_key = "\0observed_hazard"
    designs[observed_key] = stacked_design(data, nodes)
    hazards = np.zeros((n, n_times, len(levels), n_causes))
    no_events: list[tuple[int, int]] = []
    diagnostics: list[tuple[object, ...]] = []
    earlier = np.zeros((n, n_times), dtype=bool)
    taken = np.zeros((n, n_times, len(levels)))
    for cause in range(n_causes):
        rows = at_risk & ~earlier
        target = events[:, :, cause]
        if not np.any((target * data.weights[:, None])[rows] > 0):
            no_events.extend((time, cause) for time in nodes)
            diagnostics.append(())
            rho = np.zeros((n, n_times, len(levels)))
        else:
            # Node-major stacking, as stacked_design orders its blocks.
            predictions, fold_diagnostics = cross_fit_predictions(
                learner,
                designs[observed_key],
                target.T.reshape(-1),
                weights,
                stacked_folds,
                task="classification",
                predict_designs=designs,
                fit_mask=rows.T.reshape(-1),
                groups=groups,
                clip=(0.0, 1.0),
                n_jobs=n_jobs,
            )
            diagnostics.append(tuple(fold_diagnostics))
            rho = np.stack(
                [
                    np.asarray(predictions[plan.label], dtype=float).reshape(n_times, n).T
                    for plan in levels
                ],
                axis=2,
            )
        hazards[:, :, :, cause] = rho * (1.0 - taken)
        taken = taken + hazards[:, :, :, cause]
        earlier = earlier | (target > 0)
    return InitialHazards(hazards, tuple(no_events), tuple(diagnostics))


def hazard_inputs(
    data: LongitudinalData,
    plans: Sequence[Plan],
    cumulative: Sequence[FloatArray],
    *,
    alpha: float,
) -> HazardInputs:
    """Assemble the targeting inputs from the panel, the plans and the bounded mechanism.

    Parameters
    ----------
    data : LongitudinalData
        A survival panel of a held design.
    plans : sequence of Plan
        The regimens' plans, in report order.
    cumulative : sequence of FloatArray
        One ``(n, T)`` bounded cumulative :math:`g(a \\mid W) G_k(a, W)` per level.
    alpha : float
        The probability bound.

    Returns
    -------
    HazardInputs
        The targeting inputs.
    """
    at_risk, events = event_indicators(data)
    inverse = np.stack([1.0 / np.asarray(c, dtype=float) for c in cumulative], axis=2)
    observed = np.asarray(np.nan_to_num(data.treatment[:, 0], nan=0.0), dtype=np.int64)
    weights = np.asarray(data.weights, dtype=float)
    n_levels = inverse.shape[2]
    # (n, T, L): the rows at risk of positive weight, by observed level.
    followers = (at_risk & (weights > 0.0)[:, None])[:, :, None] & (
        observed[:, None, None] == np.arange(n_levels)[None, None, :]
    )
    present = followers.any(axis=0)
    happened = events > 0.0
    with_event = np.einsum("nkl,nkj->klj", followers, happened)
    without_event = np.einsum("nkl,nk->kl", followers, ~happened.any(axis=-1))
    return HazardInputs(
        at_risk=at_risk,
        events=events,
        observed=observed,
        inverse=inverse,
        q=regimen_weights(data, plans),
        weights=weights,
        alpha=alpha,
        empty=present[:, :, None] & (with_event == 0),
        full=present & (without_event == 0),
    )


# ------------------------------------------------------------------- the EIF algebra


def _per_survivor(value: FloatArray, survival: FloatArray) -> FloatArray:
    """``value / survival``, and zero where a held all-cause hazard of one left no survivor.

    Each numerator is zero where the survival is zero, so the ratio is zero there as well.
    """
    alive = survival > 0.0
    return np.asarray(
        np.divide(
            value,
            survival,
            out=np.zeros(np.broadcast_shapes(value.shape, survival.shape)),
            where=alive,
        ),
        dtype=float,
    )


@dataclass(frozen=True)
class _Observed:
    """The state read at each unit's observed level."""

    incidence: FloatArray  # (n, T, J)
    survival: FloatArray  # (n, T)
    residual: FloatArray  # (n, T, J), zero off the risk set
    total: FloatArray  # (n, T)


def _observed(state: HazardState, inputs: HazardInputs) -> _Observed:
    rows, levels = inputs.rows, inputs.observed
    hazards = state.hazards[rows, :, levels]
    residual = np.where(inputs.at_risk[:, :, None], inputs.events - hazards, 0.0)
    return _Observed(
        incidence=state.incidence[rows, :, levels],
        survival=state.survival[rows, :, levels],
        residual=residual,
        total=residual.sum(axis=-1),
    )


def _score_terms(
    state: HazardState, inputs: HazardInputs, horizons: Sequence[int]
) -> tuple[FloatArray, FloatArray]:
    """``(R, J, H, n)`` score term of every unit, and ``(R, J, H)`` score scale.

    The score term of target ``(r, j, t)`` is
    :math:`q_r(A) \\sum_{k \\le t} \\frac{R_k}{g G_k}[\\mathrm{res}_j(k) -
    \\frac{F_j(t) - F_j(k)}{S(k)} \\sum_l \\mathrm{res}_l(k)]`, read through two running
    sums over the nodes.
    """
    obs = _observed(state, inputs)
    inverse = inputs.inverse_observed()
    q = inputs.q_observed()
    ratio = _per_survivor(obs.total, obs.survival)
    running_j = np.cumsum(
        inverse[:, :, None] * (obs.residual + obs.incidence * ratio[:, :, None]), axis=1
    )
    running = np.cumsum(inverse * ratio, axis=1)
    index = [t - 1 for t in horizons]
    # (n, J, H)
    unweighted = (
        running_j[:, index, :].transpose(0, 2, 1)
        - obs.incidence[:, index, :].transpose(0, 2, 1) * running[:, index][:, None, :]
    )
    terms = q[:, None, None, :] * unweighted.transpose(1, 2, 0)[None]
    # The scale: sum_l |h_l| = q / (g G) (|1 - c| + (J - 1)|c|) on the rows at risk.
    n_causes = inputs.n_causes
    weight = np.abs(inputs.weights)
    scale = np.empty((inputs.n_regimens, n_causes, len(horizons)))
    mask = inputs.at_risk
    for h, t in enumerate(horizons):
        nodes = np.arange(inputs.n_times) < t
        for j in range(n_causes):
            c = _per_survivor(
                obs.incidence[:, t - 1, j][:, None] - obs.incidence[:, :, j], obs.survival
            )
            size = np.abs(1.0 - c) + (n_causes - 1) * np.abs(c)
            per_row = np.where(mask & nodes[None, :], inverse * size, 0.0).sum(axis=1)
            scale[:, j, h] = (weight[None, :] * q * per_row[None, :]).mean(axis=1)
    return terms, scale


def scores(
    state: HazardState, inputs: HazardInputs, horizons: Sequence[int]
) -> tuple[FloatArray, FloatArray]:
    """``(R, J, H)`` weighted score :math:`P_n D^{(1)}` of every target, and its scale.

    Parameters
    ----------
    state : HazardState
        The current hazards.
    inputs : HazardInputs
        The targeting inputs.
    horizons : sequence of int
        The target horizons, counted from one.

    Returns
    -------
    tuple of FloatArray
        The scores and the per-target scale :func:`~cleverly.fluctuation._score.score_scale`
        would give.
    """
    terms, scale = _score_terms(state, inputs, horizons)
    return np.asarray((terms * inputs.weights).mean(axis=-1), dtype=float), scale


def influence(
    state: HazardState, inputs: HazardInputs, horizons: Sequence[int]
) -> tuple[FloatArray, FloatArray]:
    """The plug-in levels and the weighted influence curve of every target.

    Parameters
    ----------
    state : HazardState
        The reported hazards.
    inputs : HazardInputs
        The targeting inputs.
    horizons : sequence of int
        The target horizons, counted from one.

    Returns
    -------
    tuple of FloatArray
        ``(R, J, H)`` levels and ``(R, J, H, n)`` influence curves
        :math:`(w / E[w]) D^*(P_w)`.
    """
    terms, _ = _score_terms(state, inputs, horizons)
    index = [t - 1 for t in horizons]
    levels = state.levels(inputs.q, inputs.weights)[:, :, index]
    curves = np.empty_like(terms)
    for r in range(inputs.n_regimens):
        for j in range(inputs.n_causes):
            plug = state.mixture(inputs.q[r], j)[:, index].T  # (H, n)
            curves[r, j] = terms[r, j] + plug - levels[r, j][:, None]
    return levels, curves * inputs.weights


def direction(
    state: HazardState, inputs: HazardInputs, horizons: Sequence[int], coefficients: FloatArray
) -> FloatArray:
    """``(n, T, L, J)`` :math:`\\sum_m c_m h^m_l(k, a, W)` at every unit, node, level and cause.

    With :math:`D_{rj}(k) = \\sum_{t \\ge k} c_{rjt}` and
    :math:`E_{rj}(k, a) = \\sum_{t \\ge k} c_{rjt} F_j(t \\mid a)`, the sum is
    :math:`\\frac{1}{g G_k} \\sum_r q_r(a) \\sum_j [1\\{l = j\\} D_{rj}(k) -
    \\frac{E_{rj}(k, a) - F_j(k \\mid a) D_{rj}(k)}{S(k \\mid a)}]`, so no per-target
    covariate is materialised.

    Parameters
    ----------
    state : HazardState
        The current hazards.
    inputs : HazardInputs
        The targeting inputs.
    horizons : sequence of int
        The target horizons, counted from one.
    coefficients : FloatArray
        ``(R, J, H)`` coefficient of every target.

    Returns
    -------
    FloatArray
        The logit increment per unit coefficient.
    """
    n_regimens, n_causes, _ = coefficients.shape
    by_node = np.zeros((n_regimens, n_causes, inputs.n_times))
    for h, t in enumerate(horizons):
        by_node[:, :, t - 1] += coefficients[:, :, h]
    # Sums over t >= k: a reversed running sum over the nodes.
    tail = np.flip(np.cumsum(np.flip(by_node, axis=2), axis=2), axis=2)
    incidence, survival = state.incidence, state.survival
    out = np.zeros_like(state.logits)
    for r in range(n_regimens):
        inner = np.zeros_like(state.logits)
        for j in range(n_causes):
            weighted = by_node[r, j][None, :, None] * incidence[..., j]
            expected = np.flip(np.cumsum(np.flip(weighted, axis=1), axis=1), axis=1)
            spill = _per_survivor(
                expected - incidence[..., j] * tail[r, j][None, :, None], survival
            )
            inner -= spill[..., None]
            inner[..., j] += tail[r, j][None, :, None]
        out += inputs.q[r][:, None, :, None] * inner
    return np.asarray(inputs.inverse[..., None] * out, dtype=float)


def loglik(state: HazardState, inputs: HazardInputs) -> float:
    """The weighted multinomial log-likelihood of the observed events on the risk set."""
    rows, levels = inputs.rows, inputs.observed
    hazards = state.hazards[rows, :, levels]
    none = 1.0 - hazards.sum(axis=-1)
    terms = xlogy(inputs.events, hazards).sum(axis=-1) + xlogy(
        1.0 - inputs.events.sum(axis=-1), none
    )
    counted = inputs.at_risk & (inputs.weights != 0.0)[:, None]
    return float(np.sum(np.where(counted, inputs.weights[:, None] * terms, 0.0)))


def _clever_observed(state: HazardState, inputs: HazardInputs, horizon: int) -> FloatArray:
    """``(R * J, n, T, J)`` :math:`h^m_l(k, A, W)` at the observed level, for one horizon."""
    obs = _observed(state, inputs)
    inverse = inputs.inverse_observed()
    q = inputs.q_observed()
    nodes = (np.arange(inputs.n_times) < horizon)[None, :]
    out = []
    for r in range(inputs.n_regimens):
        for j in range(inputs.n_causes):
            c = _per_survivor(
                obs.incidence[:, horizon - 1, j][:, None] - obs.incidence[:, :, j], obs.survival
            )
            h = -np.repeat(c[:, :, None], inputs.n_causes, axis=2)
            h[:, :, j] += 1.0
            out.append(np.where(nodes[:, :, None], (q[r][:, None] * inverse)[:, :, None] * h, 0.0))
    return np.asarray(out, dtype=float)


def _classify(
    state: HazardState, inputs: HazardInputs, steps: int, max_steps: int
) -> TargetingFailure:
    if steps >= max_steps:
        return "max_iter_reached"
    low, high = inputs.bounds
    eta = state.logits[inputs.rows, :, inputs.observed][inputs.at_risk]
    eta = eta[np.isfinite(eta)]
    if eta.size and np.mean((eta <= low) | (eta >= high)) > 0.01:
        return "bounds_pinned"
    return "line_search_exhausted"


# --------------------------------------------------------------------- algorithms


def solve_per_horizon(
    initial: HazardState,
    inputs: HazardInputs,
    horizon: int,
    targets: tuple[tuple[str, str | None, int], ...],
    *,
    max_iter: int,
    tol: float,
    warn: bool = True,
) -> tuple[HazardState, HazardFit]:
    """Target every regimen's and cause's parameter at one horizon (part (a)).

    Each outer iteration moves every level's hazards along the current clever covariates
    by :math:`\\epsilon \\in \\mathbb R^{RJ}`, then rebuilds :math:`h`.  The step is a
    Newton step on the scores themselves (:func:`_secant_step`).  When that step does not
    reduce the scores, the iteration takes the maximum-likelihood fluctuation with
    :math:`h` held fixed instead (Moore and van der Laan 2009, (13)).  It stops when every
    target's relative score is at most ``tol``, or after ``max_iter`` iterations.

    Parameters
    ----------
    initial : HazardState
        The initial hazards.
    inputs : HazardInputs
        The targeting inputs.
    horizon : int
        The horizon, counted from one.
    targets : tuple of tuple
        ``(regimen, cause, horizon)`` of each target, in ``(R, J)`` order.
    max_iter : int
        Outer iterations.
    tol : float
        Relative-score tolerance.
    warn : bool
        Whether a fit that stops short raises a ``ConvergenceWarning``.

    Returns
    -------
    tuple
        The targeted state and its record.
    """
    horizons = (horizon,)
    bounds = inputs.bounds
    state = initial
    score, scale = scores(state, inputs, horizons)
    first = score.reshape(-1).copy()
    trace = [relative_score(score.reshape(-1), scale.reshape(-1))]
    initial_levels = state.levels(inputs.q, inputs.weights)[:, :, horizon - 1].reshape(-1)
    iterations = 0
    shape = (inputs.n_regimens, inputs.n_causes, 1)
    while trace[-1] > tol and iterations < max_iter:
        size = inputs.n_regimens * inputs.n_causes
        basis = [
            direction(state, inputs, horizons, np.eye(size)[m].reshape(shape)) for m in range(size)
        ]
        moved = _secant_step(state, inputs, horizons, basis, score, scale)
        if moved is None:
            # The score equations did not move toward zero along the Newton step, so fall
            # back to the maximum-likelihood fluctuation with h held fixed (Moore (13)).
            epsilon = _newton(state, inputs, _clever_observed(state, inputs, horizon), tol)
            if not np.all(np.isfinite(epsilon)):
                break
            moved = state.moved(np.tensordot(epsilon, np.asarray(basis), axes=1), bounds)
        state = moved
        iterations += 1
        score, scale = scores(state, inputs, horizons)
        trace.append(relative_score(score.reshape(-1), scale.reshape(-1)))
    converged = bool(trace[-1] <= tol)
    if not converged and warn:
        warnings.warn(
            f"hazard targeting at horizon {horizon} stopped after {iterations} iteration(s) "
            f"with a relative score of {trace[-1]:.3g} > {tol:g}. Raise max_iter, or check "
            "result.diagnostics for a positivity problem",
            ConvergenceWarning,
            stacklevel=3,
        )
    return state, HazardFit(
        algorithm="per_horizon",
        targets=targets,
        converged=converged,
        n_iter=iterations,
        score_initial=first,
        score=score.reshape(-1),
        score_scale=scale.reshape(-1),
        trace=tuple(trace),
        initial_levels=initial_levels,
        failure=None if converged else _classify(state, inputs, iterations, max_iter),
        tol=tol,
    )


def _secant_step(
    state: HazardState,
    inputs: HazardInputs,
    horizons: tuple[int, ...],
    basis: Sequence[FloatArray],
    score: FloatArray,
    scale: FloatArray,
) -> HazardState | None:
    r"""One Newton step on every target's score along the current clever covariates.

    The fluctuation moves the logits by :math:`\sum_m \epsilon_m h^m`, with :math:`h`
    taken at the current hazards.  The Jacobian of the scores in :math:`\epsilon` is read
    by forward differences, so it carries the change of :math:`h` with the hazards that a
    fluctuation with :math:`h` held fixed leaves out.  This departs from Moore and van der
    Laan (2009, eq. 13), which fits :math:`\\epsilon` by maximum likelihood with :math:`h`
    held fixed.  That fit is the fallback, :func:`_newton`, and both endpoints solve the
    same score equations.  The step is halved until the scaled score norm
    falls, and ``None`` comes back when no halving makes it fall.
    """
    bounds = inputs.bounds
    current = (score / scale).reshape(-1)
    norm = float(np.linalg.norm(current))
    size = len(basis)
    jacobian = np.empty((size, size))
    for m, delta in enumerate(basis):
        reach = _DIFFERENCE / max(float(np.max(np.abs(delta))), 1e-300)
        probe, _ = scores(state.moved(reach * delta, bounds), inputs, horizons)
        jacobian[:, m] = ((probe / scale).reshape(-1) - current) / reach
    step = -np.linalg.lstsq(jacobian, current, rcond=None)[0]
    if not np.all(np.isfinite(step)):
        return None
    length = 1.0
    while length > _MIN_STEP:
        candidate = state.moved(np.tensordot(length * step, np.asarray(basis), axes=1), bounds)
        moved, _ = scores(candidate, inputs, horizons)
        if float(np.linalg.norm((moved / scale).reshape(-1))) < norm:
            return candidate
        length *= 0.5
    return None


def _newton(state: HazardState, inputs: HazardInputs, clever: FloatArray, tol: float) -> FloatArray:
    """Fit the fluctuation coefficients with :math:`h` held fixed, from zero."""
    rows, levels = inputs.rows, inputs.observed
    base = state.logits[rows, :, levels]
    held = np.isneginf(base)
    low, high = inputs.bounds
    w = inputs.weights[:, None]
    mask = inputs.at_risk
    n = inputs.n
    epsilon = np.zeros(clever.shape[0])

    def evaluate(eps: FloatArray) -> tuple[float, FloatArray, FloatArray]:
        eta = np.where(held, -np.inf, np.clip(base + np.tensordot(eps, clever, axes=1), low, high))
        expo = np.exp(eta)
        denom = 1.0 + expo.sum(axis=-1)
        lam = expo / denom[..., None]
        none = 1.0 / denom
        terms = xlogy(inputs.events, lam).sum(-1) + xlogy(1.0 - inputs.events.sum(-1), none)
        value = float(np.sum(np.where(mask & (w != 0.0), w * terms, 0.0)))
        residual = np.where(mask[..., None], inputs.events - lam, 0.0)
        gradient = np.einsum("mnkl,nkl->m", clever, w[..., None] * residual) / n
        mean = np.einsum("mnkl,nkl->mnk", clever, lam)
        weighted = np.where(mask, w, 0.0)[None, ..., None]
        info = (
            np.einsum("mnkl,pnkl->mp", clever * weighted, clever * lam[None])
            - np.einsum("mnk,pnk->mp", mean * weighted[..., 0], mean)
        ) / n
        return value, gradient, info

    value, gradient, info = evaluate(epsilon)
    scale = np.maximum(np.abs(clever).sum(-1)[:, mask].sum(-1) / n, 1e-300)
    for _ in range(_NEWTON_ITER):
        if np.max(np.abs(gradient) / scale) <= tol * 1e-2:
            break
        step = np.linalg.lstsq(info, gradient, rcond=None)[0]
        accepted = False
        length = 1.0
        while length > _MIN_STEP:
            candidate = epsilon + length * step
            new_value, new_gradient, new_info = evaluate(candidate)
            if new_value >= value - 1e-12 * abs(value):
                epsilon, value, gradient, info = candidate, new_value, new_gradient, new_info
                accepted = True
                break
            length *= 0.5
        if not accepted:
            break
    return epsilon


def walk_curve(
    initial: HazardState,
    inputs: HazardInputs,
    horizons: Sequence[int],
    targets: tuple[tuple[str, str | None, int], ...],
    *,
    step_size: float,
    tol: float,
    max_steps: int = 20_000,
    warn: bool = True,
) -> tuple[HazardState, HazardFit]:
    """Walk the universal least favorable submodel over every target (part (b)).

    Parameters
    ----------
    initial : HazardState
        The initial hazards.
    inputs : HazardInputs
        The targeting inputs.
    horizons : sequence of int
        Every reported horizon, counted from one.
    targets : tuple of tuple
        ``(regimen, cause, horizon)`` of each target, in ``(R, J, H)`` order.
    step_size : float
        The first step length.
    tol : float
        Relative-score tolerance.
    max_steps : int
        The step cap.
    warn : bool
        Whether a walk that stops short raises a ``ConvergenceWarning``.

    Returns
    -------
    tuple
        The targeted state and its record.
    """
    bounds = inputs.bounds
    state = initial
    levels, curves = influence(state, inputs, horizons)
    sigma = np.maximum(np.mean(curves**2, axis=-1), SIGMA_FLOOR)
    score, scale = scores(state, inputs, horizons)
    first = score.reshape(-1).copy()
    norm = float(np.sqrt(np.sum(score**2 / sigma)))
    trace = [relative_score(score.reshape(-1), scale.reshape(-1))]
    norms = [norm]
    logliks = [loglik(state, inputs)]
    dx = float(step_size)
    steps = 0
    while trace[-1] > tol and steps < max_steps and norm > 0.0:
        coefficients = score / sigma / norm
        candidate = state.moved(dx * direction(state, inputs, horizons, coefficients), bounds)
        candidate_score, candidate_scale = scores(candidate, inputs, horizons)
        candidate_norm = float(np.sqrt(np.sum(candidate_score**2 / sigma)))
        if candidate_norm > norm:
            dx *= 0.5
            if dx < _MIN_STEP:
                break
            continue
        state, score, scale, norm = candidate, candidate_score, candidate_scale, candidate_norm
        steps += 1
        trace.append(relative_score(score.reshape(-1), scale.reshape(-1)))
        norms.append(norm)
        logliks.append(loglik(state, inputs))
    converged = bool(trace[-1] <= tol)
    if not converged and warn:
        warnings.warn(
            f"one-step hazard targeting stopped after {steps} step(s) with a relative score "
            f"of {trace[-1]:.3g} > {tol:g}. Try a smaller step_size, or check "
            "result.diagnostics for a positivity problem",
            ConvergenceWarning,
            stacklevel=3,
        )
    return state, HazardFit(
        algorithm="one_step",
        targets=targets,
        converged=converged,
        n_iter=steps,
        score_initial=first,
        score=score.reshape(-1),
        score_scale=scale.reshape(-1),
        trace=tuple(trace),
        initial_levels=levels.reshape(-1),
        sigma=sigma.reshape(-1),
        norm_trace=tuple(norms),
        loglik_trace=tuple(logliks),
        failure=None if converged else _classify(state, inputs, steps, max_steps),
        tol=tol,
    )


# ------------------------------------------------------------------ isotonic projection


def project_curves(values: FloatArray) -> FloatArray:
    """Unit-weight isotonic (non-decreasing) regression of each row, in column order.

    Westling, van der Laan and Carone (2020), Section 2.1: on a grid in one dimension the
    projection is the pool-adjacent-violators algorithm.  A row that is already
    non-decreasing comes back bit for bit.

    Parameters
    ----------
    values : FloatArray
        ``(m, H)`` curves over ``H`` ordered horizons.

    Returns
    -------
    FloatArray
        The projected curves.
    """
    from scipy.optimize import isotonic_regression

    array = np.atleast_2d(np.asarray(values, dtype=float))
    out = array.copy()
    for row in range(array.shape[0]):
        if np.all(np.diff(array[row]) >= 0.0):
            continue
        out[row] = isotonic_regression(array[row], increasing=True).x
    return out


# ------------------------------------------------------------------------- the driver


def target_regimens(
    data: LongitudinalData,
    plans: Sequence[Plan],
    initial: InitialHazards,
    cumulative: Sequence[FloatArray],
    observed_cumulative: tuple[FloatArray, FloatArray],
    *,
    horizons: Sequence[int],
    algorithm: HazardAlgorithm,
    alpha: float,
    max_iter: int,
    tol: float,
    step_size: float,
) -> dict[tuple[str, str | None, int], RegimenFit]:
    """Target the hazards and report one fit per regimen, cause and horizon.

    Parameters
    ----------
    data : LongitudinalData
        A survival panel of a held design.
    plans : sequence of Plan
        The regimens' plans, in report order.
    initial : InitialHazards
        The initial hazards from :func:`fit_hazards`.
    cumulative : sequence of FloatArray
        One ``(n, T)`` bounded :math:`g G_k` per treatment level.
    observed_cumulative : tuple of FloatArray
        The unbounded and bounded ``(n, T)`` :math:`g G_k` at each unit's observed level,
        for the diagnostics.
    horizons : sequence of int
        The reported horizons, counted from one.
    algorithm : {"per_horizon", "one_step"}
        The targeting algorithm.
    alpha : float
        The probability bound.
    max_iter : int
        Outer iterations per horizon of a per-horizon fit.
    tol : float
        Relative-score tolerance.
    step_size : float
        The first step of a one-step walk.

    Returns
    -------
    dict
        One :class:`~cleverly.longitudinal.sequential.RegimenFit` per
        ``(regimen, cause, horizon)``, regimen-outer, then cause, then horizon.
    """
    inputs = hazard_inputs(data, plans, cumulative, alpha=alpha)
    start = HazardState.from_hazards(initial.hazards, inputs.bounds, inputs.empty, inputs.full)
    causes: tuple[str | None, ...] = data.cause_labels or (None,)
    horizons = tuple(horizons)
    reports: dict[int, tuple[HazardState, HazardFit]] = {}
    if algorithm == "one_step":
        targets = tuple(
            (plan.label, cause, t) for plan in plans for cause in causes for t in horizons
        )
        shared = walk_curve(start, inputs, horizons, targets, step_size=step_size, tol=tol)
        reports = dict.fromkeys(horizons, shared)
    else:
        for t in horizons:
            targets = tuple((plan.label, cause, t) for plan in plans for cause in causes)
            reports[t] = solve_per_horizon(start, inputs, t, targets, max_iter=max_iter, tol=tol)
    inverse = inputs.inverse_observed()
    q = inputs.q_observed()
    out: dict[tuple[str, str | None, int], RegimenFit] = {}
    computed: dict[int, tuple[FloatArray, FloatArray]] = {}
    for r, plan in enumerate(plans):
        for j, cause in enumerate(causes):
            for t in horizons:
                state, record = reports[t]
                if id(state) not in computed:
                    computed[id(state)] = influence(state, inputs, horizons)
                levels, curves = computed[id(state)]
                h = horizons.index(t)
                clever = np.where(inputs.at_risk, q[r][:, None] * inverse, 0.0)
                clever[:, t:] = 0.0
                out[(plan.label, cause, t)] = RegimenFit(
                    regimen=plan.regimen,
                    psi_scaled=float(levels[r, j, h]),
                    influence_curve_scaled=np.asarray(curves[r, j, h], dtype=float),
                    horizon=t,
                    cause=cause,
                    steps=(),
                    cumulative_unbounded=observed_cumulative[0],
                    cumulative=observed_cumulative[1],
                    assignment=np.asarray(plan.values),
                    obs_weights=np.asarray(data.weights, dtype=float),
                    policy=plan.policy,
                    hazard=record,
                    hazard_clever=clever,
                    hazard_constant_nodes=inputs.constant_nodes(r, t),
                )
    return out


# ------------------------------------------------------------------------ refusals


def hazard_rules_refusal(names: Sequence[str]) -> Refusal:
    """The refusal of a targeting rule the hazard walk does not read (R-abc4).

    Parameters
    ----------
    names : sequence of str
        The declared rules, each as the keyword the caller wrote, such as ``"stopping="``.

    Returns
    -------
    Refusal
        An implementation gap tracked by :data:`HAZARD_RULES_ROADMAP`.  The alternative is
        the iterated-mean survival fit, which applies each of these settings at every node.
    """
    listed = ", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else names[0]
    return Refusal(
        RefusalReason.IMPLEMENTATION_GAP,
        f"survival_model='hazard' walks its own path in steps of step_size, so {listed} do not "
        "apply to it.",
        alternative=(
            f"survival_model='iterated_mean' with targeting='iterative' applies {listed} at "
            "each node."
        ),
        roadmap=HAZARD_RULES_ROADMAP,
    )


def refuse_hazard_settings(
    *,
    survival_model: str,
    targeting: str,
    step_size: float,
    isotonic: bool,
    msm: bool,
    pseudo_learner: bool,
    iterate: str = "last",
    stopping: str = "relative",
    step_bound: float | None = None,
) -> None:
    """Refuse a hazard or isotonic setting that needs no data (H4, H5, H6, H8, R-abc4).

    Parameters
    ----------
    survival_model : str
        The ``survival_model=`` value.
    targeting : str
        The ``targeting=`` value.
    step_size : float
        The ``step_size=`` value.
    isotonic : bool
        The ``isotonic=`` value.
    msm : bool
        Whether ``msm=`` was passed.
    pseudo_learner : bool
        Whether ``pseudo_learner=`` was passed.
    iterate : str, default="last"
        The ``iterate=`` value. The caller has validated it.
    stopping : str, default="relative"
        The ``stopping=`` value. The caller has validated it.
    step_bound : float or None, default=None
        The ``step_bound=`` value. The caller has validated it.

    Raises
    ------
    ValueError
        On a malformed setting.
    CapabilityError
        On a combination that is not built.
    """
    if survival_model not in ("iterated_mean", "hazard"):
        raise ValueError(
            f"survival_model must be 'iterated_mean' or 'hazard'; got {survival_model!r}"
        )
    if targeting not in ("iterative", "one_step"):
        raise ValueError(f"targeting must be 'iterative' or 'one_step'; got {targeting!r}")
    if isinstance(step_size, bool) or not (
        isinstance(step_size, (int, float)) and np.isfinite(step_size) and step_size > 0
    ):
        raise ValueError(f"step_size must be a positive finite number; got {step_size!r}")
    if not isinstance(isotonic, bool):
        raise ValueError(f"isotonic must be True or False; got {isotonic!r}")
    if targeting == "one_step" and survival_model != "hazard":
        raise ValueError(
            "targeting='one_step' requires survival_model='hazard'. The one-step walk moves "
            "one set of hazards toward the score equations of every horizon together (Cai "
            "and van der Laan 2020). A one-step update of the iterated-mean recursion is not "
            "built. Set survival_model='hazard' on a survival design with one baseline "
            "treatment, or keep targeting='iterative' and set isotonic=True for a monotone "
            "curve"
        )
    if msm and (survival_model == "hazard" or isotonic):
        option = "survival_model='hazard'" if survival_model == "hazard" else "isotonic=True"
        raise CapabilityError(
            f"msm= with {option} is not built. A working model reports coefficients over "
            "regimen and horizon cells.",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            roadmap=HAZARD_ROADMAP,
            alternative=(
                "Fit msm= with survival_model='iterated_mean' and isotonic=False, or report "
                "the curve and take contrasts."
            ),
        )
    if survival_model == "hazard":
        declared = [
            name
            for name, differs in (
                ("iterate=", iterate != "last"),
                ("stopping=", stopping != "relative"),
                ("step_bound=", step_bound is not None),
            )
            if differs
        ]
        if declared:
            raise CapabilityError(refusal=hazard_rules_refusal(declared))
    if pseudo_learner and survival_model == "hazard":
        raise ValueError(
            "pseudo_learner= is not used with survival_model='hazard'. A hazard fit "
            "regresses the event indicator at each node and fits no pseudo-outcome. Pass "
            "the hazard learner as outcome_learner="
        )


def refuse_hazard_design(
    data: LongitudinalData, regimens: Sequence[object], *, survival_model: str, isotonic: bool
) -> None:
    """Refuse a design the hazard model or the projection does not support (H1-H3, H7, H11).

    Called after the data and the regimens are read, before any learner.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.
    regimens : sequence of Regimen or DynamicRegimen
        The resolved regimens, before any rule is evaluated.
    survival_model : str
        The ``survival_model=`` value.
    isotonic : bool
        The ``isotonic=`` value.

    Raises
    ------
    CapabilityError
        On any refused design.
    """
    if survival_model == "hazard" and not data.is_survival:
        raise CapabilityError(
            "survival_model='hazard' needs an event indicator at every node, and this fit "
            "declares one end-of-study outcome. A hazard is the chance of the event at a node "
            "among the units still at risk, so an outcome measured once has none.",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    if isotonic and not data.is_survival:
        raise CapabilityError(
            "isotonic=True makes a curve over the horizons monotone, and an end-of-study fit "
            "reports one horizon.",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    if survival_model != "hazard":
        return
    if not data.treatment_held and data.n_decisions > 1:
        raise CapabilityError(
            "survival_model='hazard' is built for one baseline treatment held over every "
            f"node, and this design declares a treatment at {data.n_decisions} nodes. Hazard "
            "targeting for a treatment at several nodes is not built.",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            roadmap=HAZARD_ROADMAP,
            alternative=(
                "LTMLE with survival_model='iterated_mean' estimates this curve, and "
                "isotonic=True makes it monotone. If one baseline treatment is held over every "
                "node, declare treatment='A'."
            ),
        )
    later = [name for block in data.time_varying_names for name in block]
    if later:
        raise CapabilityError(
            "survival_model='hazard' models the event and censoring hazards given the "
            "baseline treatment and the baseline covariates. This fit declares covariates "
            f"measured after baseline ({', '.join(later)}), and hazard targeting with such "
            "covariates is not built: it needs a model and a fluctuation of each later "
            "covariate.",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            roadmap=HAZARD_ROADMAP,
            alternative=(
                "LTMLE with survival_model='iterated_mean' adjusts for them, and "
                "isotonic=True makes its curve monotone."
            ),
        )
    # Read the kind, not ``is_continuous_node``: a vector with a dose has no one-column
    # answer, and it needs the same density this refusal names.
    kind = data.node_kind(1)
    if kind != "categorical":
        what = "is continuous" if kind == "continuous" else "is a vector with a dose"
        raise CapabilityError(
            "survival_model='hazard' needs a categorical baseline treatment, and treatment "
            f"{data.decision_name(1)!r} {what}. The hazard plug-in would need the "
            "treatment density at every dose, and that construction is not built.",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            roadmap=HAZARD_ROADMAP,
            alternative="Use survival_model='iterated_mean'.",
        )
    for regimen in regimens:
        if bool(getattr(regimen, "has_mtp", False)):
            label = getattr(regimen, "label", "")
            raise CapabilityError(
                f"regimen {label!r} is a modified treatment policy, and "
                "survival_model='hazard' does not support one. Its plug-in evaluates the "
                "event and censoring hazards at the shifted dose, and that construction is "
                "not built.",
                reason=RefusalReason.IMPLEMENTATION_GAP,
                roadmap=HAZARD_ROADMAP,
                alternative=(
                    "Use survival_model='iterated_mean', which supports a modified treatment "
                    "policy at baseline."
                ),
            )
