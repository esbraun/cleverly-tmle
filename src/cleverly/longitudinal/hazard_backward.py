r"""Hazard targeting with covariates measured after baseline, on a held design (X70 part (b1)).

The shipped hazard engine (:mod:`cleverly.longitudinal.hazard`) models the event hazards given
the baseline treatment and the baseline covariates.  A covariate :math:`L_k` measured at a node
:math:`k \ge 2` changes that: the hazard at node :math:`k` reads :math:`H_k = (W, \bar L_k, A_1)`,
and the plug-in needs the law of :math:`L_k` given the past, which this module never models.
Following Rytgaard, Gerds and van der Laan (2022), each node's event hazards and a regression
that integrates the later covariates out are targeted with the clever weight of the regimen.

**The parameter.**  With :math:`q_{r,k}` the regimen's density of :math:`A_k` (one-hot at a
static arm or a rule, the policy at a policy node, the identity :math:`1\{a = A_1\}` at every
node after the first of a held design), the parameter of cause :math:`j` at horizon :math:`t` is
:math:`\psi_{r,j}(t) = E\,Q^r_{1,jt}(H_1)` with

.. math::

    Q^r_{k,jt}(H_k) = \sum_a q_{r,k}(a \mid H_k)\Bigl[\lambda_j(k \mid H_k, a)
        + \Bigl(1 - \sum_l \lambda_l(k \mid H_k, a)\Bigr) M^r_{k,jt}(H_k, a)\Bigr],

:math:`M^r_{k,jt}(H_k, a) = E[Q^r_{k+1,jt}(H_{k+1}) \mid H_k, A_k = a, C_k = 1, Y_k = 0]` for
:math:`k < t` and :math:`M^r_{t,jt} = 0`.  It is the iterated-mean recursion of
:mod:`cleverly.longitudinal.sequential` with its node regression split by the value of
:math:`Y_k`, so the identification, the efficient influence function and the efficiency are
the iterated-mean fit's.

**The EIF, regrouped.**  With :math:`h^r_k` the iterated-mean clever covariate of the plan at
node :math:`k` and :math:`\kappa^l(k) = 1\{l = j\} - M^r_{k,jt}`, the identity
:math:`Z_k - \bar Q_k = \sum_l (dN_l(k) - \lambda_l(k)) \kappa^l(k) + (1 - \sum_l dN_l(k))
(Q_{k+1} - M_k)` gives

.. math::

    D_{r,j,t} = \sum_{k \le t} h^r_k R_k \Bigl[\sum_l \kappa^l(k)\bigl(dN_l(k) - \lambda_l(k)\bigr)
        + \Bigl(1 - \sum_l dN_l(k)\Bigr)\bigl(Q^r_{k+1,jt}(H_{k+1}) - M^r_{k,jt}\bigr)\Bigr]
        + Q^r_{1,jt}(H_1) - \psi_{r,j}(t),

a hazard term and a covariate term (Rytgaard et al. 2022, Theorem 1, display (19), on a grid).

**The continuation as a hazard curve.**  :math:`M^r_{k,jt}` for every horizon and cause at once
is the incidence of one sub-distribution of the event time after node :math:`k`, held by its
discrete hazards :math:`\nu^r_k(s, l)`, :math:`s = k + 1, \ldots, t_{\max}`.  Its outcome is the
sub-distribution :math:`p_i(s, l) = \sum_a q_{r,k+1}(a \mid H^i_{k+1}) S_a(s - 1)
\lambda_{a,l}(s)` that the targeted curve of node :math:`k + 1` implies, and it is fitted by the
soft-label hazard loss, which is linear in :math:`p` (:func:`fit_continuation`).  For any
hazards the counting-process identity
:math:`\sum_s R_s \sum_l \kappa^{l,\nu}(s)(dN_l(s) - \nu_l(s)) = 1\{T \le t, j\} - F^\nu_j(t)`
is linear in the law of the event time, so with :math:`R_s \to \rho(s)` and
:math:`dN_l(s) \to p(s, l)` the covariate term is the shipped hazard score of the
:math:`\nu` curve on a soft panel.

**The node panel.**  At node :math:`k` the node hazard (sub-node 0) and the continuation
(sub-nodes :math:`1, \ldots, t_{\max} - k`) form one curve per slot, a
:class:`~cleverly.longitudinal.hazard.HazardInputs` in the weighted form of a modified treatment
policy walk (:func:`node_panel`).  Its shipped score is the node-:math:`k` bracket above times
:math:`h^r_k`, for every horizon and cause together, and the shipped
:func:`~cleverly.longitudinal.hazard.walk_curve` targets it unchanged.

**One backward pass.**  Node :math:`k`'s scores read the node-:math:`k` hazards, the later nodes
and the mechanism only.  So the pass targets node :math:`t_{\max}` first, builds node
:math:`t_{\max} - 1`'s continuation from that targeted curve, and so on down to node 1, each node
walked once after every later node is final (:func:`backward_pass`).  After node 1 every term
of :math:`P_n D^*` is solved.  The plug-in is node 1's panel mixed over its slots, a running
sum of non-negative products per unit reduced in one order, so each reported curve is
non-decreasing in floating point, and each unit's causes sum to at most one.

**Cross-fitting** follows the shipped iterated-mean construction (Díaz, Williams, Hoffman and
Schenck 2023, Section 5.2): each outer fold runs an untargeted backward pass on its training
rows, with node hazards and continuations fitted there, the held-out predictions are stitched,
and one pooled pass walks every node with the stitched fits as offsets and the pooled targeted
pseudo-outcome as the soft events.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .._typing import BoolArray, FloatArray, IntArray, Learner
from ..estimators._nuisance import cross_fit_predictions
from ..exceptions import CapabilityError, Refusal, RefusalReason
from ..fluctuation._score import relative_score
from ..learners.crossfit import Folds
from .data import LongitudinalData, RegimenMasks
from .hazard import (
    HazardBounds,
    HazardFit,
    HazardInputs,
    HazardState,
    InitialHazards,
    MTPSlots,
    _score_terms,
    event_indicators,
    fit_hazards,
    influence,
    scores,
    walk_curve,
)
from .regimen import Plan
from .sequential import Mechanism, RegimenFit, _clever_covariate, node_ratios, outcome_design

__all__ = [
    "BackwardFit",
    "NodePanel",
    "NodeSlots",
    "backward_pass",
    "fit_continuation",
    "fold_hazards",
    "hazard_iterative_refusal",
    "hazard_msm_refusal",
    "hazard_replay_refusal",
    "hazard_markov_refusal",
    "refuse_hazard_markov",
    "hazard_stopping_refusal",
    "later_covariates",
    "node_panel",
    "node_slots",
    "pseudo_distribution",
    "soft_held_cells",
    "target_time_varying",
]

#: Whether the backward pass walks each node.  ``False`` runs the untargeted backward pass:
#: no walk, and each continuation fitted on the untargeted pseudo-outcome.  A private seam for
#: the targeting-necessity family of the ``hazard-time-varying`` study and one unit test; no
#: public argument reaches it.
_TARGETED = True


def later_covariates(data: LongitudinalData) -> tuple[str, ...]:
    """The covariates measured at a node after the first, in node order.

    A block at node 1 is measured before the first treatment decision, so it is a baseline
    confounder: a design whose only block is ``L_1`` runs the shipped hazard engine.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.

    Returns
    -------
    tuple of str
        The names of every covariate of a block at a node ``k >= 2``.
    """
    return tuple(name for block in data.time_varying_names[1:] for name in block)


# ------------------------------------------------------------------------ refusals


def hazard_iterative_refusal(names: Sequence[str]) -> Refusal:
    """The refusal of the per-horizon hazard algorithm with covariates after baseline (H28).

    Parameters
    ----------
    names : sequence of str
        The covariates measured after baseline.

    Returns
    -------
    Refusal
        An ``api_redirection`` refusal.
    """
    return Refusal(
        RefusalReason.API_REDIRECTION,
        "survival_model='hazard' with covariates measured after baseline "
        f"({', '.join(names)}) needs targeting='one_step'. The fit targets each node's hazards "
        "and the regression of its future once, after the later nodes are final, over every "
        "horizon and cause together. The per-horizon algorithm restarts from the initial fits "
        "at each horizon, and each restart would refit every regression of the future, whose "
        "outcome is built from the targeted later hazards.",
        alternative="Set targeting='one_step', or use survival_model='iterated_mean'.",
    )


#: The refusal of ``msm=`` with hazards and covariates after baseline (H29).
HAZARD_TIME_VARYING_MSM_REFUSAL = Refusal(
    RefusalReason.API_REDIRECTION,
    "msm= with survival_model='hazard' on a design with covariates measured after baseline is "
    "not built. Each regimen is targeted by its own backward pass, and a coefficient score sums "
    "cells of every regimen, so it needs one joint walk per node over every regimen's panel, "
    "which this fit does not run.",
    alternative="Fit msm= with survival_model='iterated_mean'.",
)


def hazard_msm_refusal() -> Refusal:
    """The refusal of ``msm=`` with hazards and covariates after baseline (H29).

    Returns
    -------
    Refusal
        :data:`HAZARD_TIME_VARYING_MSM_REFUSAL`.
    """
    return HAZARD_TIME_VARYING_MSM_REFUSAL


def hazard_stopping_refusal(rule: str) -> Refusal:
    """The refusal of a declared stopping rule with hazards and later covariates (H30).

    Parameters
    ----------
    rule : str
        The declared ``stopping=`` value.

    Returns
    -------
    Refusal
        An ``api_redirection`` refusal.
    """
    return Refusal(
        RefusalReason.API_REDIRECTION,
        f"stopping={rule!r} with survival_model='hazard' on a design with covariates measured "
        "after baseline is not built. The rule sets each target's threshold from the spread of "
        "its whole influence curve, which this fit has only after its backward pass, and each "
        "node walk stops on its own relative score. No split of the threshold over the node "
        "walks is implemented.",
        alternative="Use stopping='relative' (the default).",
    )


def hazard_markov_refusal(order: int) -> Refusal:
    """The refusal of ``markov_order=`` beside ``survival_model="hazard"`` (H32).

    The order-``k`` window of :meth:`LongitudinalData.history_design` starts its treatment
    blocks at node ``t - k - 1`` and skips identity nodes, so on a held design every node
    hazard design after node ``k + 2`` loses the held treatment (and a block at node 1).  The
    hazards are then pooled over the arms the regimens compare, which only a correct
    mechanism repairs.

    Parameters
    ----------
    order : int
        The declared Markov order.

    Returns
    -------
    Refusal
        An ``api_redirection`` refusal.
    """
    return Refusal(
        RefusalReason.API_REDIRECTION,
        f"markov_order={order} with survival_model='hazard' is not built. On a design with one "
        "baseline treatment held over the nodes, the window starts its treatment blocks after "
        "node 1, so the node hazard designs past the window leave out the held treatment, and "
        "a covariate declared at node 1 with it. Those hazards would be pooled over the arms "
        "the regimens compare, and the estimate would rest on the treatment and censoring "
        "models alone.",
        alternative="Use survival_model='iterated_mean' with markov_order=, or omit markov_order=.",
    )


def refuse_hazard_markov(data: LongitudinalData) -> None:
    """Raise H32 when ``data`` declares a Markov order.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel of a hazard fit.

    Raises
    ------
    CapabilityError
        When ``data.markov_order`` is set.
    """
    if data.markov_order is not None:
        raise CapabilityError.from_refusal(hazard_markov_refusal(data.markov_order))


#: The refusal of a hazard replay on a fit with covariates after baseline (H31).
HAZARD_TIME_VARYING_REPLAY_REFUSAL = Refusal(
    RefusalReason.API_REDIRECTION,
    "the replay re-targets retained initial hazards, and this fit's regressions of the future "
    "were fitted on outcomes built from targeted later hazards, which change with g_bounds, so "
    "a replay would refit learners.",
    alternative="Refit with each g_bounds= value.",
)


def hazard_replay_refusal() -> Refusal:
    """The refusal of a hazard replay on a fit with covariates after baseline (H31).

    Returns
    -------
    Refusal
        :data:`HAZARD_TIME_VARYING_REPLAY_REFUSAL`.
    """
    return HAZARD_TIME_VARYING_REPLAY_REFUSAL


#: The condition that keeps a hazard fit with covariates after baseline experimental: its
#: registered study (``hazard-time-varying``) has not run.  A support classification reads it.
HAZARD_TIME_VARYING_CONDITION = Refusal(
    RefusalReason.INFERENCE_GAP,
    "survival_model='hazard' with covariates measured after baseline has no registered "
    "simulation study yet, so the coverage of its intervals is not measured.",
    alternative=(
        "survival_model='iterated_mean' with isotonic=True has registered studies on designs "
        "with covariates after baseline."
    ),
    roadmap="X70",
)


# ------------------------------------------------------------------------ the slots


@dataclass(frozen=True)
class NodeSlots:
    """The regimen's values of the treatment at one node, for every unit.

    Parameters
    ----------
    node : int
        The node, counted from one.
    hazards : FloatArray
        ``(n, U, J)`` initial node hazards at each slot.
    q : FloatArray
        ``(n, U)`` the regimen's mixture over the slots: one at the one slot of a static arm,
        a rule or an identity node, the policy density at a policy node, and the branch
        probabilities (zero on the observed slot) at a continuous modified treatment policy.
    keys : IntArray
        ``(n, U)`` the held-cell key of each slot: the level of the treatment, or the key of
        :class:`~cleverly.longitudinal.hazard.MTPSlots` at a continuous node.
    n_keys : int
        The number of keys.
    observed : IntArray
        ``(n,)`` the slot of each unit's observed treatment, zero where the regimen's slots do
        not hold it (a non-follower of a static arm or a rule, whose loss weight is zero).
    design : FloatArray
        The continuation's training design at the node, at the observed treatment.
    designs : tuple of FloatArray
        The continuation's prediction design at each slot.
    """

    node: int
    hazards: FloatArray
    q: FloatArray
    keys: IntArray
    n_keys: int
    observed: IntArray
    design: FloatArray
    designs: tuple[FloatArray, ...]

    @property
    def n_slots(self) -> int:
        """The slot count ``U``."""
        return int(self.q.shape[1])


def _mtp_slots(data: LongitudinalData, plan: Plan) -> MTPSlots | None:
    from .hazard import mtp_slots

    return mtp_slots(data, plan)


def node_slots(
    data: LongitudinalData,
    plan: Plan,
    node: int,
    initial: InitialHazards,
    slots: MTPSlots | None = None,
) -> NodeSlots:
    """The slots of ``plan`` at ``node`` of a held design.

    At a node after the first the treatment is held, so the one slot is the observed
    treatment.  At node 1 the slots are the assigned arm of a static arm or a rule, every level
    at a categorical policy node (a known policy, a resolved odds tilt, or a modified treatment
    policy's level map), and the observed dose with each branch's dose at a continuous
    modified treatment policy.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel of a held design.
    plan : Plan
        The resolved plan, with any policy numerator attached.
    node : int
        The node, counted from one.
    initial : InitialHazards
        The initial node hazards.
    slots : MTPSlots or None
        The plan's slots at a continuous baseline node, ``None`` at a categorical one.

    Returns
    -------
    NodeSlots
        The slots.
    """
    n = data.n
    rows = np.arange(n)
    n_levels = 0 if slots is not None else len(data.treatment_levels[0])
    observed_code = np.asarray(np.nan_to_num(data.treatment[:, 0], nan=0.0), dtype=np.int64)
    design = outcome_design(data, plan, node)
    if node > 1:
        if slots is None:
            hazards = initial.hazards[rows, node - 1, observed_code][:, None, :]
            keys, n_keys = observed_code[:, None], n_levels
        else:
            assert initial.observed is not None
            hazards = initial.observed[:, node - 1][:, None, :]
            keys, n_keys = slots.keys[:, :1], slots.n_keys
        return NodeSlots(
            node,
            np.asarray(hazards, dtype=float),
            np.ones((n, 1)),
            np.asarray(keys, dtype=np.int64),
            n_keys,
            np.zeros(n, dtype=np.int64),
            design,
            (design,),
        )
    if slots is not None:
        # A continuous modified treatment policy: slot 0 the observed dose, slot b the dose of
        # branch b, each predicted from the continuation at its own design.
        branches = slots.n_slots - 1
        return NodeSlots(
            node,
            np.asarray(initial.slots[plan.label][:, 0], dtype=float),
            np.concatenate([np.zeros((n, 1)), slots.probabilities], axis=1),
            np.asarray(slots.keys, dtype=np.int64),
            slots.n_keys,
            np.zeros(n, dtype=np.int64),
            design,
            (design, *(outcome_design(data, plan, 1, arm=b) for b in range(branches))),
        )
    if plan.is_policy_node(1):
        return NodeSlots(
            node,
            np.asarray(initial.hazards[:, 0], dtype=float),
            np.asarray(plan.policy_at(1), dtype=float),
            np.repeat(np.arange(n_levels, dtype=np.int64)[None, :], n, axis=0),
            n_levels,
            observed_code,
            design,
            tuple(outcome_design(data, plan, 1, arm=code) for code in range(n_levels)),
        )
    assigned = np.asarray(np.nan_to_num(plan.values[:, 0], nan=0.0), dtype=np.int64)
    return NodeSlots(
        node,
        np.asarray(initial.hazards[rows, 0, assigned], dtype=float)[:, None, :],
        np.ones((n, 1)),
        assigned[:, None],
        n_levels,
        np.zeros(n, dtype=np.int64),
        design,
        (design,),
    )


# ------------------------------------------------------------------------ the panel


def soft_held_cells(
    at_risk: BoolArray,
    events: FloatArray,
    loss_weight: FloatArray,
    observed: IntArray,
    n_cells: int,
    node_bounds: HazardBounds | None,
) -> tuple[BoolArray, BoolArray, BoolArray | None]:
    """The held cells of a panel whose events are fractions in ``[0, 1]``.

    The followers of cell ``c`` at a sub-node are the rows at risk there with a positive
    ``(n, S)`` loss weight whose observed key is ``c``.  A cell is empty for a cause when no
    follower carries a positive fraction of it, and full when every follower's all-cause
    fraction is one.  On binary events and a loss weight equal to the observation weight these
    are the cells of :func:`~cleverly.longitudinal.hazard._held_cells`, which reads the
    ``(n,)`` observation weight and marks a cell full when no follower is event-free, so a
    fractional event would count as a full one there.  A bounded sub-node holds its cells as
    the shipped function does.

    Parameters
    ----------
    at_risk : BoolArray
        ``(n, S)`` the rows at risk at each sub-node.
    events : FloatArray
        ``(n, S, J)`` the event fractions.
    loss_weight : FloatArray
        ``(n, S)`` the loss weight.
    observed : IntArray
        ``(n,)`` the key of each row's observed slot.
    n_cells : int
        The number of keys.
    node_bounds : HazardBounds or None
        The bounded sub-nodes.

    Returns
    -------
    tuple
        The ``(S, K, J)`` empty, ``(S, K)`` full and the excluded cells, ``None`` without a
        bound.
    """
    followers = (at_risk & (loss_weight > 0.0))[:, :, None] & (
        observed[:, None, None] == np.arange(n_cells)[None, None, :]
    )
    present = followers.any(axis=0)
    with_event = np.einsum("nkl,nkj->klj", followers, events > 0.0)
    partial = np.einsum("nkl,nk->kl", followers, events.sum(axis=-1) < 1.0)
    empty = present[:, :, None] & (with_event == 0)
    full = present & (partial == 0)
    excluded = None
    if node_bounds is not None and node_bounds.active:
        at = node_bounds.bounded[:, None, None]
        floor = (node_bounds.lower > 0.0)[:, None, :]
        held = empty & ~(at & floor)
        excluded = at & ((empty & floor) | (full[:, :, None] & ~held))
        empty = held
        full = full & ~node_bounds.bounded[:, None]
    return empty, full, excluded


@dataclass(frozen=True)
class NodePanel:
    """One node's panel: its node hazard and its continuation, as one curve per slot.

    Parameters
    ----------
    node : int
        The node, counted from one.
    rows : IntArray
        ``(n_k,)`` the panel's rows of the data: the regimen's rows at risk entering the node.
    inputs : HazardInputs
        The weighted targeting inputs, sub-node axis of length ``t_max - k + 1``.
    initial : HazardState
        The initial curve: the node hazards at sub-node 0 and the continuation after it.
    horizons : tuple of int
        The reported horizons ``t >= k``, as sub-node counts ``t - k + 1``.
    reported : tuple of int
        The same horizons as nodes.
    """

    node: int
    rows: IntArray
    inputs: HazardInputs
    initial: HazardState
    horizons: tuple[int, ...]
    reported: tuple[int, ...]


def _soft_at_risk(pseudo: FloatArray) -> FloatArray:
    """``(n, S)`` :math:`\\rho(s) = 1 - \\sum_{u < s, l} p(u, l)`, one at the first sub-node."""
    spent = np.cumsum(pseudo.sum(axis=-1), axis=1)
    return np.asarray(
        1.0 - np.concatenate([np.zeros((pseudo.shape[0], 1)), spent[:, :-1]], axis=1),
        dtype=float,
    )


def node_panel(
    data: LongitudinalData,
    slots: NodeSlots,
    continuation: FloatArray | None,
    pseudo: FloatArray | None,
    weight: FloatArray,
    *,
    rows: BoolArray,
    following: BoolArray | None,
    horizons: Sequence[int],
    alpha: float,
    node_bounds: HazardBounds | None,
) -> NodePanel:
    """The node panel of 2.5.4 on the rows entering the node.

    ==============  =====================  ===================================================
    field           sub-node 0             sub-node ``s > k``
    ==============  =====================  ===================================================
    ``at_risk``     :math:`R_k`            the rows entering ``k + 1``, with :math:`\\rho > 0`
    ``events``      :math:`dN_l(k)`        :math:`p(s, l) / \\rho(s)`
    ``inverse``     :math:`h_k`            :math:`h_k \\rho(s)`
    ``loss_weight`` :math:`w h_k`          :math:`w h_k \\rho(s)`
    ==============  =====================  ===================================================

    ``q`` is the slot mixture, read by the plug-in only, and ``observed`` the slot of the
    observed treatment, slot 0 on a row whose loss weight is zero.  A node bound applies to
    sub-node 0 only: it bounds the data's hazard, and a continuation hazard is not one.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.
    slots : NodeSlots
        The regimen's slots at the node.
    continuation : FloatArray or None
        ``(n, S - 1, U, J)`` the continuation hazards at each slot, ``None`` at the last node.
    pseudo : FloatArray or None
        ``(n, S - 1, J)`` the pseudo-outcome of the next node, zero off its rows, ``None`` at
        the last node.
    weight : FloatArray
        ``(n,)`` :math:`h^r_k`, the regimen's clever covariate at the node, on the followers.
    rows : BoolArray
        ``(n,)`` the regimen's rows at risk entering the node.
    following : BoolArray or None
        ``(n,)`` the regimen's rows entering the next node, ``None`` at the last node.
    horizons : sequence of int
        The reported horizons.
    alpha : float
        The probability bound.
    node_bounds : HazardBounds or None
        The data's bounded nodes.

    Returns
    -------
    NodePanel
        The panel and its initial curve.
    """
    k = slots.node
    index = np.flatnonzero(rows)
    at_risk_data, events_data = event_indicators(data)
    n_sub = 1 if continuation is None else 1 + continuation.shape[1]
    n_causes = events_data.shape[2]
    h = np.asarray(weight, dtype=float)[index]
    hazards = np.empty((index.size, n_sub, slots.n_slots, n_causes))
    hazards[:, 0] = slots.hazards[index]
    at_risk = np.zeros((index.size, n_sub), dtype=bool)
    at_risk[:, 0] = at_risk_data[index, k - 1]
    events = np.zeros((index.size, n_sub, n_causes))
    events[:, 0] = events_data[index, k - 1]
    inverse = np.zeros((index.size, n_sub))
    inverse[:, 0] = h
    if continuation is not None:
        assert pseudo is not None and following is not None
        hazards[:, 1:] = continuation[index]
        p = pseudo[index]
        rho = _soft_at_risk(p)
        alive = following[index][:, None] & (rho > 0.0)
        at_risk[:, 1:] = alive
        events[:, 1:] = np.where(alive[..., None], p / np.where(alive, rho, 1.0)[..., None], 0.0)
        inverse[:, 1:] = np.where(alive, h[:, None] * rho, 0.0)
    weights = np.asarray(data.weights, dtype=float)[index]
    loss_weight = weights[:, None] * inverse
    observed = np.where(h > 0.0, slots.observed[index], 0)
    cells = slots.keys[index]
    panel_bounds = None
    if node_bounds is not None and node_bounds.active and node_bounds.bounded[k - 1]:
        bounded = np.zeros(n_sub, dtype=bool)
        bounded[0] = True
        lower = np.zeros((n_sub, n_causes))
        scale = np.ones((n_sub, n_causes))
        lower[0] = node_bounds.lower[k - 1]
        scale[0] = node_bounds.scale[k - 1]
        panel_bounds = HazardBounds(bounded, lower, scale)
    empty, full, excluded = soft_held_cells(
        at_risk,
        events,
        loss_weight,
        cells[np.arange(index.size), observed],
        slots.n_keys,
        panel_bounds,
    )
    inputs = HazardInputs(
        at_risk=at_risk,
        events=events,
        observed=np.asarray(observed, dtype=np.int64),
        inverse=np.repeat(inverse[:, :, None], slots.n_slots, axis=2),
        q=slots.q[index][None],
        weights=weights,
        alpha=alpha,
        empty=empty,
        full=full,
        node_bounds=panel_bounds,
        excluded=excluded,
        loss_weight=loss_weight,
        cells=cells,
    )
    held_empty, held_full = inputs.held_cells()
    initial = HazardState.from_hazards(hazards, inputs.bounds, held_empty, held_full, panel_bounds)
    reported = tuple(t for t in horizons if t >= k)
    return NodePanel(k, index, inputs, initial, tuple(t - k + 1 for t in reported), reported)


def pseudo_distribution(state: HazardState, q: FloatArray, rows: IntArray, n: int) -> FloatArray:
    """``(n, S, J)`` the sub-distribution of the event time that a targeted panel implies.

    :math:`p_i(s, l) = \\sum_a q(a)\\, S_a(s - 1)\\, \\lambda_{a,l}(s)`, each term a product of
    non-negative numbers, summed in slot order, on the panel's rows and zero elsewhere.

    Parameters
    ----------
    state : HazardState
        The panel's curve, ``(n_k, S, U, J)``.
    q : FloatArray
        ``(n_k, U)`` the slot mixture.
    rows : IntArray
        ``(n_k,)`` the panel's rows of the data.
    n : int
        The row count of the data.

    Returns
    -------
    FloatArray
        The pseudo-outcome, scattered back to every row.
    """
    survival = state.survival
    before = np.concatenate([np.ones_like(survival[:, :1]), survival[:, :-1]], axis=1)
    terms = before[..., None] * state.hazards  # (n_k, S, U, J)
    total = q[:, 0, None, None] * terms[:, :, 0]
    for slot in range(1, q.shape[1]):
        total = total + q[:, slot, None, None] * terms[:, :, slot]
    out = np.zeros((n, *total.shape[1:]))
    out[rows] = total
    return out


# ------------------------------------------------------------------ the continuation


def _stack(design: FloatArray, n_later: int) -> FloatArray:
    """The design stacked over the later nodes, node-major, with one indicator per later node."""
    n = design.shape[0]
    blocks = []
    for later in range(n_later):
        indicator = np.zeros((n, n_later))
        indicator[:, later] = 1.0
        blocks.append(np.hstack([design, indicator]))
    return np.vstack(blocks)


def fit_continuation(
    data: LongitudinalData,
    slots: NodeSlots,
    pseudo: FloatArray,
    train: BoolArray,
    *,
    learner: Learner,
    folds: Folds,
    n_jobs: int = 1,
    cause_order: Sequence[int] | None = None,
) -> FloatArray:
    """Fit the continuation of one node: the hazards of the future, the later covariates out.

    Cause :math:`j` regresses the fraction :math:`p(s, j) / (\\rho(s) - \\sum_{l < j} p(s, l))`
    with weight :math:`w(\\rho(s) - \\sum_{l < j} p(s, l))` on the node's design stacked over the
    later nodes, one indicator column per later node :math:`s`, and
    :math:`\\nu_j = \\rho_j (1 - \\sum_{i < j} \\nu_i)`, as
    :func:`~cleverly.longitudinal.hazard.fit_hazards` builds the node hazards.  The soft-label
    hazard loss is linear in :math:`p`, so its minimiser at each history is the hazard of the
    mean sub-distribution (Rytgaard et al. 2022 pool their regressions over the later times
    likewise).  A cause whose weighted fractions are all zero, or all one, fits no learner and
    predicts that value, as :func:`~cleverly.longitudinal.sequential.constant_target` does.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.
    slots : NodeSlots
        The node's slots, whose designs the fit is trained and predicted at.
    pseudo : FloatArray
        ``(n, S, J)`` the next node's pseudo-outcome, zero off its rows.
    train : BoolArray
        ``(n,)`` the rows the fit may train on: the regimen's rows entering the next node,
        narrowed to an outer fold's training rows in a fold's untargeted pass.
    learner : Learner
        The ``pseudo_learner``, a regression.
    folds : Folds
        The split the fit is out of fold on.
    n_jobs : int
        Worker count.
    cause_order : sequence of int or None
        The cause indices in fit order, ``None`` for code order.

    Returns
    -------
    FloatArray
        ``(n, S, U, J)`` the continuation hazards at each slot.
    """
    n, n_later, n_causes = pseudo.shape
    rho = _soft_at_risk(pseudo)
    design = _stack(slots.design, n_later)
    predict = {f"slot_{u}": _stack(slots.designs[u], n_later) for u in range(slots.n_slots)}
    stacked_folds = Folds(np.tile(folds.assignment, n_later), folds.n_folds)
    groups = None if data.cluster is None else np.tile(data.cluster, n_later)
    weights = np.asarray(data.weights, dtype=float)
    earlier = np.zeros((n, n_later))
    taken = np.zeros((n, n_later, slots.n_slots))
    out = np.zeros((n, n_later, slots.n_slots, n_causes))
    for cause in range(n_causes) if cause_order is None else cause_order:
        remaining = rho - earlier
        positive = train[:, None] & (remaining > 0.0)
        target = np.where(positive, pseudo[:, :, cause] / np.where(positive, remaining, 1.0), 0.0)
        weight = np.where(positive, weights[:, None] * remaining, 0.0)
        fitted = positive & (weight > 0.0)
        if not np.any(target[fitted] > 0.0):
            fraction = np.zeros((n, n_later, slots.n_slots))
        elif np.all(target[fitted] >= 1.0):
            fraction = np.ones((n, n_later, slots.n_slots))
        else:
            predictions, _ = cross_fit_predictions(
                learner,
                design,
                target.T.reshape(-1),
                weight.T.reshape(-1),
                stacked_folds,
                task="regression",
                predict_designs=predict,
                fit_mask=fitted.T.reshape(-1),
                groups=groups,
                clip=(0.0, 1.0),
                n_jobs=n_jobs,
            )
            fraction = np.stack(
                [
                    np.asarray(predictions[f"slot_{u}"], dtype=float).reshape(n_later, n).T
                    for u in range(slots.n_slots)
                ],
                axis=2,
            )
        out[..., cause] = np.clip(fraction, 0.0, 1.0) * (1.0 - taken)
        taken = taken + out[..., cause]
        earlier = earlier + pseudo[:, :, cause]
    return out


# ------------------------------------------------------------------- the pass


def node_curve_terms(
    state: HazardState, inputs: HazardInputs, horizons: Sequence[int]
) -> FloatArray:
    """``(J, H, n_k)`` the score term of each panel row, the node's part of the curve.

    Parameters
    ----------
    state : HazardState
        The panel's targeted curve.
    inputs : HazardInputs
        The panel's inputs.
    horizons : sequence of int
        The panel's horizons as sub-node counts.

    Returns
    -------
    FloatArray
        The node-:math:`k` bracket of the regrouped EIF times :math:`h^r_k`, unweighted.
    """
    terms, _ = _score_terms(state, inputs, horizons)
    return np.asarray(terms[0], dtype=float)


@dataclass(frozen=True)
class BackwardFit:
    """One regimen's backward pass.

    Parameters
    ----------
    levels : FloatArray
        ``(J, H)`` the plug-in of every cause and reported horizon.
    curves : FloatArray
        ``(J, H, n)`` the weighted influence curve: the sum over the nodes of each final
        panel's score terms, plus node 1's plug-in minus its level, times the observation
        weight.
    record : HazardFit
        The regimen's record: scores and scales summed over the nodes, each node's multiplied
        by :math:`n_k / n`.
    panels : tuple of NodePanel
        The final panel of each node, node 1 first, with its targeted state in ``states``.
    states : tuple of HazardState
        The targeted curve of each node, node 1 first.
    """

    levels: FloatArray
    curves: FloatArray
    record: HazardFit
    panels: tuple[NodePanel, ...] = ()
    states: tuple[HazardState, ...] = ()


#: How a pass obtains node ``k``'s continuation from the next node's pseudo-outcome.
ContinuationSource = Callable[[int, NodeSlots, FloatArray], FloatArray]


def _walked(
    panel: NodePanel,
    targets: tuple[tuple[str, str | None, int], ...],
    *,
    walk: bool,
    step_size: float,
    tol: float,
    max_steps: int,
    iterate: str,
    warn: bool,
) -> tuple[HazardState, HazardFit]:
    """Walk one node's panel, or record its untargeted state."""
    if walk:
        return walk_curve(
            panel.initial,
            panel.inputs,
            panel.horizons,
            targets,
            step_size=step_size,
            tol=tol,
            max_steps=max_steps,
            warn=warn,
            iterate=iterate,
        )
    levels, _ = influence(panel.initial, panel.inputs, panel.horizons)
    score, scale = scores(panel.initial, panel.inputs, panel.horizons)
    relative = relative_score(score.reshape(-1), scale.reshape(-1))
    return panel.initial, HazardFit(
        algorithm="one_step",
        targets=targets,
        converged=bool(relative <= tol),
        n_iter=0,
        score_initial=score.reshape(-1),
        score=score.reshape(-1),
        score_scale=scale.reshape(-1),
        trace=(relative,),
        initial_levels=levels.reshape(-1),
        tol=tol,
    )


def backward_pass(
    data: LongitudinalData,
    plan: Plan,
    initial: InitialHazards,
    weight: FloatArray,
    continuation: ContinuationSource,
    *,
    horizons: Sequence[int],
    alpha: float,
    tol: float,
    step_size: float,
    max_steps: int = 20_000,
    iterate: str = "last",
    walk: bool = True,
    warn: bool = True,
    node_bounds: HazardBounds | None = None,
    slots: MTPSlots | None = None,
) -> BackwardFit:
    """Run one regimen's backward pass: one walk per node, each after the later nodes are final.

    At node :math:`k`, from :math:`t_{\\max}` down to 1, the pass obtains the continuation from
    the next node's pseudo-outcome (``continuation``), builds the node panel, walks it with the
    shipped :func:`~cleverly.longitudinal.hazard.walk_curve` over every reported horizon
    :math:`t \\ge k` and every cause, and reads the pseudo-outcome of node :math:`k - 1` from the
    targeted panel.  Each node's score and scale are multiplied by :math:`n_k / n`, so the
    record's sum is the shipped score and scale of the whole target.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel of a held design.
    plan : Plan
        The resolved plan.
    initial : InitialHazards
        The node hazards: in sample, or the stitched out-of-fold fits.
    weight : FloatArray
        ``(n, T)`` :math:`h^r_k`, the plan's iterated-mean clever covariate at each node.
    continuation : callable
        ``(k, slots, pseudo) -> (n, S, U, J)``: node ``k``'s continuation from the next node's
        pseudo-outcome.  In sample it fits :func:`fit_continuation`; the pooled cross-fitted
        pass reads the stitched untargeted fits.
    horizons : sequence of int
        The reported horizons.
    alpha : float
        The probability bound.
    tol : float
        The relative-score tolerance of each node walk.
    step_size : float
        The first step of each node walk.
    max_steps : int
        The step cap of each node walk.
    iterate : {"last", "best"}
        Which step each node walk reports.
    walk : bool
        Whether each node is walked.  ``False`` runs the untargeted pass.
    warn : bool
        Whether a node walk that stops short warns.
    node_bounds : HazardBounds or None
        The data's bounded nodes.
    slots : MTPSlots or None
        The plan's slots at a continuous baseline node.

    Returns
    -------
    BackwardFit
        The levels, the influence curves and the record.
    """
    horizons = tuple(horizons)
    t_max = max(horizons)
    causes: tuple[str | None, ...] = data.cause_labels or (None,)
    masks = plan.masks(data)
    n = data.n
    panels: list[NodePanel] = []
    states: list[HazardState] = []
    fits: list[HazardFit] = []
    pseudo: FloatArray | None = None
    for k in range(t_max, 0, -1):
        node = node_slots(data, plan, k, initial, slots)
        nu = None if pseudo is None else continuation(k, node, pseudo)
        panel = node_panel(
            data,
            node,
            nu,
            pseudo,
            weight[:, k - 1],
            rows=masks.at_risk(k),
            following=None if pseudo is None else masks.at_risk(k + 1),
            horizons=horizons,
            alpha=alpha,
            node_bounds=node_bounds,
        )
        targets = tuple((plan.label, cause, t) for cause in causes for t in panel.reported)
        state, fit = _walked(
            panel,
            targets,
            walk=walk,
            step_size=step_size,
            tol=tol,
            max_steps=max_steps,
            iterate=iterate,
            warn=warn,
        )
        panels.append(panel)
        states.append(state)
        fits.append(fit)
        pseudo = pseudo_distribution(state, panel.inputs.q[0], panel.rows, n)
    panels.reverse()
    states.reverse()
    fits.reverse()
    return _assemble(data, plan, panels, states, fits, horizons, causes)


def _assemble(
    data: LongitudinalData,
    plan: Plan,
    panels: Sequence[NodePanel],
    states: Sequence[HazardState],
    fits: Sequence[HazardFit],
    horizons: tuple[int, ...],
    causes: tuple[str | None, ...],
) -> BackwardFit:
    """The levels, curves and record of a pass from its final node panels."""
    n, n_causes, n_horizons = data.n, len(causes), len(horizons)
    weights = np.asarray(data.weights, dtype=float)
    curves = np.zeros((n_causes, n_horizons, n))
    score = np.zeros((n_causes, n_horizons))
    first = np.zeros((n_causes, n_horizons))
    scale = np.zeros((n_causes, n_horizons))
    for panel, state, fit in zip(panels, states, fits, strict=True):
        share = panel.rows.size / n
        columns = [horizons.index(t) for t in panel.reported]
        terms = node_curve_terms(state, panel.inputs, panel.horizons)  # (J, H_k, n_k)
        for position, h in enumerate(columns):
            curves[:, h, panel.rows] += terms[:, position]
        shape = (n_causes, len(columns))
        score[:, columns] += share * fit.score.reshape(shape)
        first[:, columns] += share * fit.score_initial.reshape(shape)
        scale[:, columns] += share * fit.score_scale.reshape(shape)
    head, state = panels[0], states[0]
    levels = state.levels(head.inputs.q, head.inputs.weights)[0][:, [t - 1 for t in horizons]]
    plug = np.stack(
        [
            state.mixture(head.inputs.q[0], j)[:, [t - 1 for t in horizons]].T
            for j in range(n_causes)
        ]
    )  # (J, H, n_1)
    curves[:, :, head.rows] += plug - levels[:, :, None]
    curves = curves * weights
    converged = all(fit.converged for fit in fits)
    failure = next((fit.failure for fit in fits if not fit.converged), None)
    record = HazardFit(
        algorithm="one_step",
        targets=tuple((plan.label, cause, t) for cause in causes for t in horizons),
        converged=converged,
        n_iter=sum(fit.n_iter for fit in fits),
        score_initial=first.reshape(-1),
        score=score.reshape(-1),
        score_scale=scale.reshape(-1),
        trace=tuple(fit.trace[-1] for fit in reversed(fits)),
        initial_levels=np.asarray(fits[0].initial_levels, dtype=float),
        failure=failure if not converged else None,
        tol=fits[0].tol,
        nodes=tuple(fits),
    )
    return BackwardFit(levels, curves, record, tuple(panels), tuple(states))


# ---------------------------------------------------------------- cross-fitting


def fold_hazards(
    data: LongitudinalData,
    levels: Sequence[Plan],
    *,
    learner: Learner,
    folds: Folds,
    n_jobs: int = 1,
    slots: Sequence[MTPSlots] = (),
) -> tuple[InitialHazards, tuple[InitialHazards, ...]]:
    """Each outer fold's node hazards on its training rows, and their held-out stitch.

    Fold :math:`f`'s hazards are fitted on its training rows and predicted on every row, so its
    untargeted backward pass reads no held-out row (Díaz et al. 2023, Section 5.2, Steps 1-4).
    The stitch takes each row from the fold that held it out, which is the out-of-fold fit of
    :func:`~cleverly.longitudinal.hazard.fit_hazards`.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.
    levels : sequence of Plan
        One plan per treatment level.
    learner : Learner
        The hazard learner.
    folds : Folds
        The outer split.
    n_jobs : int
        Worker count.
    slots : sequence of MTPSlots
        The slots of each modified treatment policy at a continuous baseline node.

    Returns
    -------
    tuple
        The stitched hazards and each fold's.
    """
    single = Folds.single(data.n)
    per_fold = []
    for _, test in folds:
        train = np.ones(data.n, dtype=bool)
        train[test] = False
        per_fold.append(
            fit_hazards(
                data,
                levels,
                learner=learner,
                folds=single,
                n_jobs=n_jobs,
                slots=slots,
                fit_rows=train,
            )
        )
    hazards = np.array(per_fold[0].hazards, copy=True)
    stitched_slots = {label: np.array(v, copy=True) for label, v in per_fold[0].slots.items()}
    observed = None if per_fold[0].observed is None else np.array(per_fold[0].observed, copy=True)
    for fold, (_, test) in enumerate(folds):
        source = per_fold[fold]
        hazards[test] = source.hazards[test]
        for label in stitched_slots:
            stitched_slots[label][test] = source.slots[label][test]
        if observed is not None:
            assert source.observed is not None
            observed[test] = source.observed[test]
    no_event = set(per_fold[0].no_event_nodes)
    for source in per_fold[1:]:
        no_event &= set(source.no_event_nodes)
    diagnostics = tuple(
        tuple(entry for source in per_fold for entry in source.diagnostics[index])
        for index in range(len(per_fold[0].diagnostics))
    )
    stitched = InitialHazards(
        hazards, tuple(sorted(no_event)), diagnostics, stitched_slots, observed
    )
    return stitched, tuple(per_fold)


# ------------------------------------------------------------------- the driver


def clever_weights(
    data: LongitudinalData, plan: Plan, mechanism: Mechanism, g_bounds: tuple[float, float]
) -> FloatArray:
    """``(n, T)`` :math:`h^r_k`, the iterated-mean clever covariate of ``plan`` at each node.

    The plan's cumulative intervention density over its bounded cumulative mechanism on the
    rows that follow the plan through the node, and zero elsewhere: the expression of
    :func:`~cleverly.longitudinal.sequential._clever_covariate`, called with the same inputs.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.
    plan : Plan
        The resolved plan, with any policy numerator attached.
    mechanism : Mechanism
        The fitted mechanism, evaluated at ``plan``.
    g_bounds : tuple of float
        The cumulative bounds.

    Returns
    -------
    FloatArray
        One column per node.
    """
    masks = plan.masks(data)
    cumulative = mechanism.cumulative(data, plan, g_bounds)
    numerator = plan.cumulative_numerator(data)
    columns = []
    for time in range(1, data.n_times + 1):
        _, clever = _clever_covariate(
            masks.at_risk(time),
            masks.following(time),
            cumulative,
            time,
            numerator=None if numerator is None else numerator[:, time - 1],
        )
        columns.append(clever)
    return np.column_stack(columns)


def target_time_varying(
    data: LongitudinalData,
    plans: Sequence[Plan],
    levels: Sequence[Plan],
    mechanism: Mechanism,
    *,
    outcome_learner: Learner,
    pseudo_learner: Learner,
    folds: Folds,
    horizons: Sequence[int],
    g_bounds: tuple[float, float],
    alpha: float,
    tol: float,
    step_size: float,
    iterate: str = "last",
    node_bounds: HazardBounds | None = None,
    n_jobs: int = 1,
    max_steps: int = 20_000,
) -> tuple[InitialHazards, dict[tuple[str, str | None, int], RegimenFit]]:
    """Fit and target a hazard fit with covariates after baseline, one pass per regimen.

    The node hazards are fitted once, on every row at risk with the current treatment in the
    design, and every regimen walks its own copy.  In sample each continuation is fitted on the
    targeted pseudo-outcome of the next node.  Cross-fitted, each outer fold runs an untargeted
    pass on its training rows, the held-out node hazards and continuations are stitched, and
    one pooled pass walks every node.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel of a held design with a covariate after baseline.
    plans : sequence of Plan
        The resolved plans, with any policy numerator attached.
    levels : sequence of Plan
        One plan per treatment level, empty at a continuous baseline node.
    mechanism : Mechanism
        The out-of-fold mechanism, evaluated at every plan.
    outcome_learner : Learner
        The node hazard learner, a classifier.
    pseudo_learner : Learner
        The continuation learner, a regression.
    folds : Folds
        The outer split.
    horizons : sequence of int
        The reported horizons.
    g_bounds : tuple of float
        The cumulative bounds.
    alpha : float
        The probability bound of every hazard.
    tol : float
        The relative-score tolerance of each node walk.
    step_size : float
        The first step of each node walk.
    iterate : {"last", "best"}
        Which step each node walk reports.
    node_bounds : HazardBounds or None
        The data's bounded nodes.
    n_jobs : int
        Worker count.
    max_steps : int
        The step cap of each node walk.

    Returns
    -------
    tuple
        The node hazards (stitched when cross-fitted), and one
        :class:`~cleverly.longitudinal.sequential.RegimenFit` per ``(regimen, cause, horizon)``,
        regimen-outer, then cause, then horizon.
    """
    horizons = tuple(horizons)
    t_max = max(horizons)
    slots = {plan.label: _mtp_slots(data, plan) for plan in plans}
    mtp = tuple(entry for entry in slots.values() if entry is not None)
    single = folds.is_single
    walk = _TARGETED
    if single:
        initial = fit_hazards(
            data, levels, learner=outcome_learner, folds=folds, n_jobs=n_jobs, slots=mtp
        )
        per_fold: tuple[InitialHazards, ...] = ()
    else:
        initial, per_fold = fold_hazards(
            data, levels, learner=outcome_learner, folds=folds, n_jobs=n_jobs, slots=mtp
        )
    causes: tuple[str | None, ...] = data.cause_labels or (None,)
    out: dict[tuple[str, str | None, int], RegimenFit] = {}
    for plan in plans:
        weight = clever_weights(data, plan, mechanism, g_bounds)
        entry = slots[plan.label]
        masks = plan.masks(data)

        def fitted(
            k: int,
            node: NodeSlots,
            pseudo: FloatArray,
            train: BoolArray | None = None,
            split: Folds = folds,
            masks: RegimenMasks = masks,
        ) -> FloatArray:
            rows = masks.at_risk(k + 1) if train is None else masks.at_risk(k + 1) & train
            return fit_continuation(
                data, node, pseudo, rows, learner=pseudo_learner, folds=split, n_jobs=n_jobs
            )

        if single:
            source: ContinuationSource = fitted
        else:
            stitched = _stitched_continuations(
                data,
                plan,
                weight,
                per_fold,
                folds,
                fitted,
                horizons=horizons,
                alpha=alpha,
                node_bounds=node_bounds,
                slots=entry,
            )

            def source(
                k: int,
                node: NodeSlots,
                pseudo: FloatArray,
                stitched: dict[int, FloatArray] = stitched,
            ) -> FloatArray:
                return stitched[k]

        fit = backward_pass(
            data,
            plan,
            initial,
            weight,
            source,
            horizons=horizons,
            alpha=alpha,
            tol=tol,
            step_size=step_size,
            max_steps=max_steps,
            iterate=iterate,
            walk=walk,
            warn=walk,
            node_bounds=node_bounds,
            slots=entry,
        )
        out.update(
            _regimen_fits(data, plan, mechanism, fit, weight, horizons, causes, g_bounds, t_max)
        )
    return initial, out


def _stitched_continuations(
    data: LongitudinalData,
    plan: Plan,
    weight: FloatArray,
    per_fold: Sequence[InitialHazards],
    folds: Folds,
    fitted: Callable[..., FloatArray],
    *,
    horizons: tuple[int, ...],
    alpha: float,
    node_bounds: HazardBounds | None,
    slots: MTPSlots | None,
) -> dict[int, FloatArray]:
    """Each node's continuation, stitched from the folds' untargeted passes on training rows."""
    single = Folds.single(data.n)
    stitched: dict[int, FloatArray] = {}
    for fold, (_, test) in enumerate(folds):
        train = np.ones(data.n, dtype=bool)
        train[test] = False

        def in_fold(
            k: int,
            node: NodeSlots,
            pseudo: FloatArray,
            train: BoolArray = train,
            test: IntArray = test,
        ) -> FloatArray:
            nu = fitted(k, node, pseudo, train, single)
            if k not in stitched:
                stitched[k] = np.zeros_like(nu)
            stitched[k][test] = nu[test]
            return nu

        backward_pass(
            data,
            plan,
            per_fold[fold],
            weight,
            in_fold,
            horizons=horizons,
            alpha=alpha,
            tol=1.0,
            step_size=1.0,
            walk=False,
            warn=False,
            node_bounds=node_bounds,
            slots=slots,
        )
    return stitched


def _regimen_fits(
    data: LongitudinalData,
    plan: Plan,
    mechanism: Mechanism,
    fit: BackwardFit,
    weight: FloatArray,
    horizons: tuple[int, ...],
    causes: tuple[str | None, ...],
    g_bounds: tuple[float, float],
    t_max: int,
) -> dict[tuple[str, str | None, int], RegimenFit]:
    """One :class:`RegimenFit` per cause and horizon of one regimen's pass."""
    cumulative = mechanism.cumulative_with_unbounded(data, plan, g_bounds)
    at_risk, _ = event_indicators(data)
    numerator = plan.cumulative_numerator(data)
    ratio = node_ratios(data, plan, mechanism)
    out: dict[tuple[str, str | None, int], RegimenFit] = {}
    for j, cause in enumerate(causes):
        for h, t in enumerate(horizons):
            clever = np.where(at_risk, weight, 0.0)
            clever[:, t:] = 0.0
            constant = tuple(
                panel.node
                for panel in fit.panels
                if panel.node <= t and panel.inputs.constant_nodes(0, t - panel.node + 1)
            )
            out[(plan.label, cause, t)] = RegimenFit(
                regimen=plan.regimen,
                psi_scaled=float(fit.levels[j, h]),
                influence_curve_scaled=np.asarray(fit.curves[j, h], dtype=float),
                horizon=t,
                cause=cause,
                steps=(),
                cumulative_unbounded=cumulative[0],
                cumulative=cumulative[1],
                assignment=np.asarray(plan.values),
                obs_weights=np.asarray(data.weights, dtype=float),
                policy=plan.policy,
                hazard=fit.record,
                hazard_clever=clever,
                hazard_constant_nodes=constant,
                cumulative_numerator=numerator,
                node_ratio=ratio,
            )
    return out
