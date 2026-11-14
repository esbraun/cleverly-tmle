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
static arm, a known rule or a known policy at the one baseline decision.  A modified
treatment policy :math:`d` has the parameter
:math:`\psi_{d,j}(t) = E \sum_b p_b F_j(t \mid d_b(A, W), W)` and its own weighted walk
(:func:`target_mtp`).

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

``LTMLE(node_bounds=)`` bounds the cause-specific hazards of the named nodes (Benkeser,
Carone and Gilbert 2018, Section 3, eq. (3); ``survtmle`` ``bounds=``).  At a bounded node
cause :math:`j` holds :math:`\theta_j = \operatorname{logit}((\lambda_j - l_j) / s_j)` with
the fixed scale :math:`s_j = u_j - l_j`, clipped as a softmax logit is.  The fluctuation adds
:math:`s_j \sum_m \epsilon_m h^m_j` to :math:`\theta_j`, and the loss of the node is the
quasi-binomial loss of :math:`(dN_j - l_j) / s_j`.  Its derivative in :math:`\epsilon_m` is
:math:`h^m_j (dN_j - \lambda_j)`, the efficient-influence-function hazard score, so both
algorithms solve the same score equations.  The upper bounds of a node sum to at most one,
so the clip keeps :math:`\sum_j \lambda_j < 1`.  Every other node keeps the softmax logits.

Every plug-in mean over the horizons reduces its column in one value-independent order shared
by every horizon.  The incidence :math:`F_j(t)` of each unit is a running sum of non-negative
terms, a policy mixture of such sums is a sum of non-negative products, and a weighted mean
of non-decreasing columns reduced in one order is non-decreasing, because rounding to
nearest is monotone.  So a one-step curve is non-decreasing in floating point with no
tolerance, and ``tests/unit/test_survival_monotone.py`` pins it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
from scipy.special import expit, xlogy

from .._typing import BoolArray, FloatArray, IntArray, Learner
from .._warnings import warn as _warn
from ..estimators._nuisance import cross_fit_predictions
from ..exceptions import (
    CapabilityError,
    ConvergenceWarning,
    MethodConfigurationError,
    Refusal,
    RefusalReason,
)
from ..fluctuation._score import (
    STOP_CONTEXT_MISSING,
    BestIterate,
    StopContext,
    iterate_norm,
    relative_score,
    rule_ratio_at,
)
from ..fluctuation.iterative import TargetingFailure, _line_step, _Visited
from ..learners.crossfit import Folds
from ..msm import ProjectionFit, solve_projection
from .data import LongitudinalData
from .msm import MSMRegimenFit, RegimenMSM
from .pooled_hazards import refuse_pooled_designs, stacked_design
from .regimen import Plan, RegimenSpec, declares_tuned
from .sequential import NodeBound, RegimenFit

__all__ = [
    "HAZARD_BOUND_EXCLUDES_DATA",
    "HAZARD_STEP_BOUND_ONE_STEP_REFUSAL",
    "SIGMA_FLOOR",
    "WALK_LOSS_TOLERANCE",
    "HazardAlgorithm",
    "HazardBounds",
    "HazardFit",
    "HazardInputs",
    "HazardState",
    "InitialHazards",
    "MTPSlots",
    "check_hazard_node_bounds",
    "clip_logits",
    "clipped_shares",
    "fit_hazards",
    "hazard_bounds",
    "hazard_inputs",
    "hazard_msm_mtp_refusal",
    "hazard_mtp_replay_refusal",
    "hazard_separable_mtp_refusal",
    "hazard_tuned_cap_refusal",
    "mtp_inputs",
    "mtp_slots",
    "project_curves",
    "refuse_hazard_design",
    "refuse_hazard_settings",
    "refuse_separable_mtp",
    "slot_design",
    "softmax_logits",
    "solve_per_horizon",
    "target_mtp",
    "target_regimens",
    "walk_curve",
    "walk_loss",
]

#: The two hazard targeting algorithms.
HazardAlgorithm = Literal["per_horizon", "one_step", "untargeted"]

#: The roadmap item that tracks hazard targeting outside its held baseline design.
HAZARD_ROADMAP = "X70"

#: The refusal of ``step_bound=`` beside the one-step hazard walk, which already walks the
#: universal least favorable submodel in steps of ``step_size``.  The same code and shape as
#: :data:`~cleverly.fluctuation._score.STEP_BOUND_ONE_STEP_REFUSAL`, in the hazard keywords.
HAZARD_STEP_BOUND_ONE_STEP_REFUSAL = Refusal(
    RefusalReason.API_REDIRECTION,
    "targeting='one_step' with survival_model='hazard' already walks the universal "
    "least-favorable submodel in steps of step_size.",
    "step_bound= bounds the steps of targeting='iterative'.",
)

#: The refusal of ``msm=`` beside ``isotonic=True`` (H5).
HAZARD_MSM_ISOTONIC_REFUSAL = Refusal(
    RefusalReason.API_REDIRECTION,
    "msm= with isotonic=True is not built. The isotonic projection moves the levels of one "
    "curve and leaves their influence curves as they were, and a working model fitted to "
    "projected levels has no influence curve this package derives.",
    "Set isotonic=False. survival_model='iterated_mean' takes msm= on every design, and "
    "survival_model='hazard' with targeting='one_step' takes it where the design admits a "
    "hazard fit and projects monotone curves.",
)

#: The refusal of ``msm=`` beside the per-horizon hazard algorithm (H14).
HAZARD_MSM_PER_HORIZON_REFUSAL = Refusal(
    RefusalReason.API_REDIRECTION,
    "msm= with survival_model='hazard' needs targeting='one_step'. A working model's cells "
    "span the horizons, and the per-horizon algorithm restarts from the initial hazards at "
    "each horizon, so no one set of hazards solves the coefficient scores.",
    "Set targeting='one_step'.",
)


def hazard_tuned_cap_refusal(label: str) -> Refusal:
    """The refusal of a tuned cap on a hazard fit (H16).

    Parameters
    ----------
    label : str
        The regimen's label.

    Returns
    -------
    Refusal
        An ``api_redirection``.
    """
    return Refusal(
        RefusalReason.API_REDIRECTION,
        f"regimen {label!r} tunes its cap from the treatment law, and survival_model='hazard' "
        "does not build the tuner's term in the influence curve.",
        "Use survival_model='iterated_mean', which builds it.",
    )


def hazard_msm_mtp_refusal(label: str) -> Refusal:
    """The refusal of ``msm=`` beside a modified treatment policy on a hazard fit (H19).

    Parameters
    ----------
    label : str
        The policy regimen's label.

    Returns
    -------
    Refusal
        An ``api_redirection``.
    """
    return Refusal(
        RefusalReason.API_REDIRECTION,
        f"msm= with survival_model='hazard' cannot take the modified treatment policy "
        f"{label!r}. Each such policy is targeted by its own weighted walk, and a coefficient "
        "score sums cells of every regimen, so no one walk solves it.",
        "Fit msm= with survival_model='iterated_mean', which takes a modified treatment "
        "policy cell.",
    )


def hazard_separable_mtp_refusal(label: str) -> Refusal:
    """The refusal of a separable regimen beside a modified treatment policy (H20).

    Parameters
    ----------
    label : str
        The policy regimen's label.

    Returns
    -------
    Refusal
        An ``api_redirection``.
    """
    return Refusal(
        RefusalReason.API_REDIRECTION,
        f"a Separable regimen and the modified treatment policy {label!r} cannot share one "
        "fit. A separable fit targets every regimen of the fit on the two hazard factors, and "
        "a modified treatment policy needs its own weighted walk.",
        "Fit the separable regimens and the modified treatment policy in two LTMLE calls.",
    )


def hazard_mtp_replay_refusal(label: str) -> Refusal:
    """The refusal of a truncation replay of a fit with a modified treatment policy walk (H25).

    Parameters
    ----------
    label : str
        The policy regimen's label.

    Returns
    -------
    Refusal
        An ``api_redirection``.
    """
    return Refusal(
        RefusalReason.API_REDIRECTION,
        "the truncation replay re-targets the retained initial hazards of one shared walk, and "
        f"regimen {label!r} was targeted by its own weighted walk, whose slot hazards and loss "
        "weights the replay does not rebuild.",
        "Refit with each g_bounds= value, or replay a fit without a modified treatment policy.",
    )


def refuse_separable_mtp(regimens: Sequence[object]) -> None:
    """Refuse a modified treatment policy beside a separable regimen (H20), before any learner.

    Called before the companion check of
    :func:`~cleverly.longitudinal.separable.refuse_separable_companions`, so a policy meets
    the hazard refusal that names its own walk.

    Parameters
    ----------
    regimens : sequence of Regimen or DynamicRegimen
        The resolved regimens declared beside the separable entries.

    Raises
    ------
    CapabilityError
        When a regimen holds a modified treatment policy.
    """
    for regimen in regimens:
        if bool(getattr(regimen, "has_mtp", False)):
            raise CapabilityError.from_refusal(
                hazard_separable_mtp_refusal(getattr(regimen, "label", ""))
            )


#: The roadmap item that tracks a ratio trim on the hazard path.
HAZARD_RATIO_TRIM_ROADMAP = "X80"

#: The refusal of ``ratio_trim=`` beside ``survival_model="hazard"`` (R-d1).  ``lmtp``'s
#: ``.trim`` trims the density ratios of its sequential regression, and the hazard targeting
#: has no such ratio matrix: its clever covariates read the cumulative inverse treatment and
#: censoring probabilities of each node, and no source defines their trim.
HAZARD_RATIO_TRIM_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "ratio_trim= with survival_model='hazard' is not built. lmtp's trim acts on the density "
    "ratios of the iterated-mean recursion, and the hazard targeting has no trim of its "
    "clever covariates.",
    "Fit ratio_trim= with survival_model='iterated_mean', or fit survival_model='hazard' "
    "without ratio_trim= and bound the treatment mechanism with g_bounds=.",
    roadmap=HAZARD_RATIO_TRIM_ROADMAP,
)

#: The width :math:`u_j - l_j` below which a bounded hazard node warns: its fluctuation moves
#: the hazard by at most that much, so the score there barely responds to a step.
SMALL_SCALE = 1e-4

#: The variance floor of a one-step target whose initial influence curve is zero, so that
#: the fixed metric :math:`\Sigma` stays invertible.
SIGMA_FLOOR = 1e-12

#: Inner Newton iterations of one per-horizon fluctuation, with :math:`h` held fixed.
_NEWTON_ITER = 50

#: The logit increment of the forward differences behind a per-horizon Newton step.
_DIFFERENCE = 1e-6

#: The shortest one-step step before the walk gives up, as in ``solve_one_step``.
_MIN_STEP = 1e-14

#: The walk-loss tolerance of the walk.  A finite Euler step carries an :math:`O(dx^2)` term,
#: so the accepted walk loss is non-increasing only to within it.
WALK_LOSS_TOLERANCE = 1e-8


@dataclass(frozen=True)
class HazardBounds:
    """The declared bounds of the bounded hazard nodes, on the hazard scale.

    Parameters
    ----------
    bounded : BoolArray
        ``(T,)`` whether each node is bounded.  A bounded node bounds every cause.
    lower : FloatArray
        ``(T, J)`` lower bound :math:`l_j(k)`, zero at an unbounded node.
    scale : FloatArray
        ``(T, J)`` fixed scale :math:`s_j(k) = u_j(k) - l_j(k)`, one at an unbounded node.
        It multiplies the logit step of each cause, so an unbounded node takes the shipped
        step.
    """

    bounded: BoolArray
    lower: FloatArray
    scale: FloatArray

    @property
    def active(self) -> bool:
        """Whether some node is bounded."""
        return bool(np.any(self.bounded))

    def hazards(self, logits: FloatArray) -> FloatArray:
        """``l + s expit(theta)`` at the bounded nodes of ``(n, T, L, J)`` logits.

        Returns
        -------
        FloatArray
            ``(n, K, L, J)`` hazards of the ``K`` bounded nodes.
        """
        k = self.bounded
        return np.asarray(
            self.lower[k][None, :, None, :] + self.scale[k][None, :, None, :] * expit(logits[:, k]),
            dtype=float,
        )


def hazard_bounds(
    node_bounds: dict[str | None, dict[int, NodeBound]] | None,
    n_times: int,
    causes: Sequence[str | None],
) -> HazardBounds | None:
    """The hazard-scale bounds of a fit, from the resolved ``node_bounds=``.

    Parameters
    ----------
    node_bounds : dict or None
        The resolved bounds by cause (``None`` for one cause) and one-based node.  The
        estimator has checked that a node bounds every cause or none.
    n_times : int
        The node count.
    causes : sequence of str or None
        The cause labels in event-code order, ``(None,)`` for one cause.

    Returns
    -------
    HazardBounds or None
        ``None`` when no node is bounded.
    """
    if not node_bounds:
        return None
    bounded = np.zeros(n_times, dtype=bool)
    lower = np.zeros((n_times, len(causes)))
    scale = np.ones((n_times, len(causes)))
    for j, cause in enumerate(causes):
        for node, bound in node_bounds.get(cause, {}).items():
            bounded[node - 1] = True
            lower[node - 1, j] = bound.lower
            scale[node - 1, j] = bound.upper - bound.lower
    return HazardBounds(bounded, lower, scale)


def check_hazard_node_bounds(
    by_node: dict[int, dict[str | None, tuple[float, float]]],
    grid: Sequence[float],
    causes: Sequence[str | None],
) -> None:
    """Refuse hazard-scale node bounds that the bounded fluctuation cannot hold (R-i2, R-i6).

    A bounded node holds every cause on the fixed scale :math:`s_j = u_j - l_j`.  A cause left
    unbounded would share the mass the bounded causes leave, so its scale would move with
    theirs (R-i6).  The clip keeps each hazard below its upper bound, so upper bounds that sum
    to at most one keep :math:`\\sum_j \\lambda_j < 1` (R-i2).  A width below
    :data:`SMALL_SCALE` warns.

    Parameters
    ----------
    by_node : dict
        The declared ``(lower, upper)`` of each cause, by one-based node.
    grid : sequence of float
        The grid time of each node.
    causes : sequence of str or None
        The cause labels, ``(None,)`` for one cause.

    Raises
    ------
    CapabilityError
        When a node bounds some causes and not others, or when its upper bounds sum to more
        than one.
    """
    for node in sorted(by_node):
        bounds = by_node[node]
        time = grid[node - 1]
        missing = [cause for cause in causes if cause not in bounds]
        if missing:
            named = [cause for cause in causes if cause in bounds]
            raise CapabilityError(
                refusal=Refusal(
                    RefusalReason.INVALID_DECLARATION,
                    f"node_bounds bounds cause(s) {named} at {time!r} and not {missing}; a "
                    "hazard fit bounds every cause of a node or none. A bounded node holds each "
                    "cause on a fixed scale, and an unbounded cause there would share the mass "
                    "the bounded causes leave.",
                    alternative=f"Bound every cause at {time!r}, or none of them.",
                )
            )
        total = float(sum(upper for _, upper in bounds.values()))
        if total > 1.0:
            raise CapabilityError(
                refusal=Refusal(
                    RefusalReason.INVALID_DECLARATION,
                    f"node_bounds at {time!r} has upper bounds summing to {total:g}; a hazard "
                    "fit needs them to sum to at most 1, so that the bounded hazards leave room "
                    "for no event.",
                    alternative=(
                        f"Lower the upper bounds at {time!r} so that they sum to at most 1."
                    ),
                )
            )
        for cause, (lower, upper) in bounds.items():
            if upper - lower < SMALL_SCALE:
                which = "" if cause is None else f" for cause {cause!r}"
                _warn(
                    f"node_bounds at {time!r}{which} has a width of {upper - lower:.3g} on the "
                    f"hazard scale, below {SMALL_SCALE:g}. The fluctuation moves that hazard by "
                    "at most the width, so its score barely responds to a step.",
                    ConvergenceWarning,
                    stacklevel=4,
                )


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
        Super Learner diagnostics per node and cause, in cause order.
    slots : dict of str to FloatArray
        By regimen label, the ``(n, T, U, J)`` hazards at each slot of a modified treatment
        policy at a continuous or mixed baseline node (:class:`MTPSlots`).  Empty on every
        other fit.
    observed : FloatArray or None
        ``(n, T, J)`` the hazards at each unit's observed treatment on a fit whose baseline
        node is continuous or mixed, which has no levels.  ``None`` on a categorical node,
        where the observed level of ``hazards`` holds them.
    """

    hazards: FloatArray
    no_event_nodes: tuple[tuple[int, int], ...] = ()
    diagnostics: tuple[tuple[object, ...], ...] = ()
    slots: dict[str, FloatArray] = field(default_factory=dict)
    observed: FloatArray | None = None


@dataclass(frozen=True)
class HazardInputs:
    """Everything the targeting reads besides the hazards themselves.

    Parameters
    ----------
    at_risk : BoolArray
        ``(n, T)`` :math:`R_k`: uncensored through ``k`` and event-free before ``k``.
    events : FloatArray
        ``(n, T, J)`` :math:`dN_l(k)`, the event of each cause at each node, on the rows at
        risk and zero elsewhere.  A fraction in ``[0, 1]`` on the node panel of a fit with
        covariates after baseline: the walk loss and the scores are linear in it.
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
    node_bounds : HazardBounds or None
        The bounded nodes, ``None`` when the fit declared none.
    excluded : BoolArray or None
        ``(T, L, J)`` cells of a bounded node whose maximum-likelihood hazard lies outside
        the bound: an empty cell under a positive lower bound, and every cause of a full
        cell.  No such cell is held, so the fluctuation pins it at the clip.  ``None`` when
        no node is bounded.
    loss_weight : FloatArray or None
        ``(n, T)`` loss weight :math:`\\omega_k = w\\, r(A, W) / G_k(A, W)` of a modified
        treatment policy walk, ``None`` on every other fit.  When it is set the clever
        covariate is :math:`\\kappa_{j,t}` alone, with no :math:`q / (g G)` factor, the loss
        and the score of node :math:`k` are weighted by :math:`\\omega_k`, ``inverse`` holds
        :math:`r / G_k` at every slot (only the observed slot is read), and ``q`` carries the
        branch probabilities of the plug-in alone.
    cells : IntArray or None
        ``(n, L)`` the held-cell key of each unit's slot, ``None`` when the slots are the
        treatment levels.  On a slot state the cell arrays ``empty``, ``full`` and
        ``excluded`` are indexed by key in place of level, and every slot whose key is ``c``
        takes the hold of cell ``c``.
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
    node_bounds: HazardBounds | None = None
    excluded: BoolArray | None = None
    loss_weight: FloatArray | None = None
    cells: IntArray | None = None

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

    @property
    def weighted(self) -> bool:
        """Whether the walk weights its loss (a modified treatment policy walk)."""
        return self.loss_weight is not None

    def node_weights(self) -> FloatArray:
        """``(n, T)`` the weight of each unit's loss term at each node.

        The observation weight, or the loss weight :math:`\\omega_k` of a weighted walk.
        """
        if self.loss_weight is not None:
            return np.asarray(self.loss_weight, dtype=float)
        return np.broadcast_to(self.weights[:, None], self.at_risk.shape)

    def q_observed(self) -> FloatArray:
        """``(R, n)`` :math:`q_r(A \\mid W)` at the observed level.

        A weighted walk carries its ratio in ``inverse`` and in its loss weight, so its
        factor here is one.
        """
        if self.weighted:
            return np.ones((self.n_regimens, self.n))
        return np.asarray(self.q[:, self.rows, self.observed], dtype=float)

    def held_cells(self) -> tuple[BoolArray, BoolArray]:
        """``empty`` and ``full`` in the shape the state reads.

        On a level state these are the ``(T, L, J)`` and ``(T, L)`` cell arrays.  On a slot
        state each unit's slot reads the cell of its key, so they are ``(n, T, L, J)`` and
        ``(n, T, L)``.

        Returns
        -------
        tuple of BoolArray
            The held-at-zero and held-at-one arrays.
        """
        if self.cells is None:
            return self.empty, self.full
        empty = np.moveaxis(self.empty[:, self.cells], 0, 1)
        full = np.moveaxis(self.full[:, self.cells], 0, 1)
        return np.asarray(empty, dtype=bool), np.asarray(full, dtype=bool)

    def inverse_observed(self) -> FloatArray:
        """``(n, T)`` the inverse mechanism at the observed level."""
        return np.asarray(self.inverse[self.rows, :, self.observed], dtype=float)

    def constant_nodes(self, regimen: int, horizon: int) -> tuple[int, ...]:
        """The nodes up to ``horizon`` with an empty or full cell at a level ``regimen`` reads.

        A regimen reads a level when it puts mass on that level for some unit.  The
        influence curve term of such a cell is identically zero, which is the case
        :func:`~cleverly.longitudinal.sequential.constant_nodes` reports.  A cell of a
        bounded node that the bound excludes is not held, so it is left out.
        """
        if self.cells is None:
            read = np.any(self.q[regimen] > 0.0, axis=0)
        else:
            read = np.zeros(self.full.shape[1], dtype=bool)
            read[np.unique(self.cells[self.q[regimen] > 0.0])] = True
        flagged = (self.empty.any(axis=2) | self.full)[:horizon][:, read].any(axis=1)
        return tuple(int(k) + 1 for k in np.flatnonzero(flagged))


def clip_logits(
    logits: FloatArray, bounds: tuple[float, float], bounded: BoolArray | None = None
) -> FloatArray:
    """Clip softmax cause logits into ``bounds`` without losing the ratios between causes.

    Where a unit's largest cause logit at a node exceeds the upper bound, every cause logit of
    that unit and node moves down by the same constant, so the largest sits at the bound and
    :math:`\\lambda_j / \\lambda_l` is kept.  Each logit is then clipped into ``bounds``, so a
    cause that the shift pushes under the lower bound comes back up to it.  With one cause the
    shift is the clip.  A node of ``bounded``, ``(T,)``, holds independent per-cause logits and
    takes the per-cause clip alone.

    Parameters
    ----------
    logits : FloatArray
        ``(n, T, ..., J)`` cause logits, the node on the second axis.
    bounds : tuple of float
        The clip interval.
    bounded : BoolArray or None, default=None
        ``(T,)`` the bounded nodes.

    Returns
    -------
    FloatArray
        The clipped logits.  A logit at :math:`-\\infty` comes back at the lower bound.
    """
    eta = np.asarray(logits, dtype=float)
    low, high = bounds
    top = np.max(eta, axis=-1)
    over = np.isfinite(top) & (top > high)
    if bounded is not None:
        node = np.asarray(bounded, dtype=bool).reshape((1, -1) + (1,) * (eta.ndim - 3))
        over = over & ~node
    with np.errstate(invalid="ignore"):
        # (eta - top) + high puts the largest logit at exactly ``high``, as the clip would.
        shifted = np.where(over[..., None], (eta - top[..., None]) + high, eta)
    return np.asarray(np.clip(shifted, low, high), dtype=float)


def _per_unit(cells: BoolArray, ndim: int) -> BoolArray:
    """A cell array with a unit axis: shared cells gain a leading axis of one.

    A level state holds one ``(T, L, ...)`` cell array for every unit, and a slot state one
    per unit, ``(n, T, L, ...)``, already of ``ndim`` axes.
    """
    return cells if cells.ndim == ndim else cells[None]


def softmax_logits(hazards: FloatArray, high: float, full: BoolArray | None = None) -> FloatArray:
    """``(n, T, L, J)`` softmax logits of hazards, before the clip into the bounds.

    A unit with some mass left on no event takes :math:`\\log(\\lambda_j / \\lambda_0)`.  A
    unit with none left, which is every unit of a ``full`` cell, ``(T, L)``, and any other unit
    whose fitted hazards sum to one or more, has no finite reference.  It takes
    :math:`\\log(\\lambda_j / \\max_l \\lambda_l) + \\text{high}`, the ratio-keeping upper
    clip of :func:`clip_logits`: the largest cause sits at ``high`` and every ratio
    :math:`\\lambda_j / \\lambda_l` is kept down to the lower bound.  A full cell's softmax has
    no no-event category, so the common constant does not move its split.

    Parameters
    ----------
    hazards : FloatArray
        ``(n, T, L, J)`` hazards.
    high : float
        The upper clip of a logit.
    full : BoolArray or None, default=None
        ``(T, L)`` cells whose all-cause hazard is held at one.

    Returns
    -------
    FloatArray
        The logits, unclipped.  A zero hazard has a logit of :math:`-\\infty`.
    """
    lam = np.asarray(hazards, dtype=float)
    reference = 1.0 - lam.sum(axis=-1)
    largest = lam.max(axis=-1)
    spent = reference <= 0.0
    if full is not None:
        spent = spent | _per_unit(np.asarray(full, dtype=bool), 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.log(lam) - np.log(reference)[..., None]
        topped = np.log(lam) - np.log(largest)[..., None] + high
    return np.asarray(np.where((spent & (largest > 0.0))[..., None], topped, eta), dtype=float)


@dataclass(frozen=True)
class HazardState:
    """One set of hazards, held as clipped softmax logits.

    Parameters
    ----------
    logits : FloatArray
        ``(n, T, L, J)`` :math:`\\eta_l = \\log(\\lambda_l / \\lambda_0)`, clipped by
        :func:`clip_logits`.
    full : BoolArray or None
        ``(T, L)`` cells whose no-event probability :math:`\\lambda_0` is held at exactly
        zero.  The logits there split the event among the causes,
        :math:`\\lambda_l = e^{\\eta_l} / \\sum_m e^{\\eta_m}`, so a common constant added
        to them leaves the split unchanged.
    node_bounds : HazardBounds or None
        The bounded nodes.  There ``logits`` holds
        :math:`\\theta_j = \\operatorname{logit}((\\lambda_j - l_j) / s_j)` for every cause.
    """

    logits: FloatArray
    full: BoolArray | None = None
    node_bounds: HazardBounds | None = None
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
            held = _per_unit(np.asarray(self.full, dtype=bool), 3)
            denominator = np.where(held, events, 1.0 + events)
            none = np.where(held, 0.0, 1.0 / denominator)
        hazards = expo / denominator[..., None]
        bounds = self.node_bounds
        if bounds is not None and bounds.active:
            bounded = bounds.hazards(eta)
            hazards[:, bounds.bounded] = bounded
            none = np.array(none, dtype=float)
            none[:, bounds.bounded] = 1.0 - bounded.sum(axis=-1)
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
        node_bounds: HazardBounds | None = None,
    ) -> HazardState:
        """The state of ``(n, T, L, J)`` hazards, with the logits clipped by :func:`clip_logits`.

        A cell of ``empty``, ``(T, L, J)``, holds the hazard at exactly zero for every unit,
        its logit at :math:`-\\infty`.  A cell of ``full``, ``(T, L)``, holds the all-cause
        hazard at exactly one for every unit, and its logits are those of
        :func:`softmax_logits`, which keep the fitted ratios :math:`\\lambda_j / \\lambda_l`
        among the causes down to the lower bound.  No fluctuation moves either hold.
        At a node of ``node_bounds`` the logit is
        :math:`\\operatorname{logit}((\\lambda_j - l_j) / s_j)`, clipped into ``bounds``, so the
        hazard lies in :math:`[l_j + (1 - \\alpha) s_j, u_j - (1 - \\alpha) s_j]`.
        """
        lam = np.asarray(hazards, dtype=float)
        low, high = bounds
        eta = softmax_logits(lam, high, full)
        bounded = None
        if node_bounds is not None and node_bounds.active:
            k = bounded = node_bounds.bounded
            share = np.clip(
                (lam[:, k] - node_bounds.lower[k][None, :, None, :])
                / node_bounds.scale[k][None, :, None, :],
                0.0,
                1.0,
            )
            with np.errstate(divide="ignore"):
                eta[:, k] = np.log(share) - np.log(1.0 - share)
        eta = clip_logits(np.nan_to_num(eta, nan=low, posinf=high, neginf=low), bounds, bounded)
        if empty is not None:
            eta = np.where(_per_unit(np.asarray(empty, dtype=bool), 4), -np.inf, eta)
        return cls(eta, full, node_bounds)

    def moved(self, delta: FloatArray, bounds: tuple[float, float]) -> HazardState:
        """The state after adding ``delta`` to every logit, then clipping by :func:`clip_logits`.

        A logit at :math:`-\\infty`, a hazard held at zero, stays there, and so does a held
        all-cause hazard of one.  The bounded nodes stay bounded: ``delta`` there is the
        step on :math:`\\theta_j`, which :func:`direction` has scaled by :math:`s_j`.
        """
        node_bounds = self.node_bounds
        bounded = node_bounds.bounded if node_bounds is not None and node_bounds.active else None
        return HazardState(
            np.where(
                np.isneginf(self.logits),
                -np.inf,
                clip_logits(self.logits + delta, bounds, bounded),
            ),
            self.full,
            self.node_bounds,
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
        ``(regimen, cause, horizon)`` of every target, in score order.  A separable-effects
        fit targets each hazard factor apart and appends the factor,
        ``(regimen, cause, horizon, factor)``.  A working-model fit targets the score of
        every coefficient, ``(term, cause)``.
    converged : bool
        Whether every target's relative score reached ``tol`` at the reported hazards.  Under
        a declared stopping rule, whether the rule ratio is at most one there.
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
        The plug-in value of every target at the initial hazards.  For a working-model fit,
        the coefficients projected from the initial hazards.
    sigma : FloatArray or None
        The fixed diagonal metric of a one-step walk.  A working-model walk uses the full
        covariance of its targets' curves, and this is that covariance's floored diagonal.
    norm_trace : tuple of float
        The accepted :math:`\\Sigma`-norm of the score of a one-step walk.
    walk_loss_trace : tuple of float
        The accepted walk loss of a one-step walk: the negative weighted log-likelihood at
        the unbounded nodes plus the quasi-binomial loss at the bounded nodes.
    failure : str or None
        Why the targeting stopped short, ``None`` when it converged.
    tol : float
        The relative-score tolerance.
    stopping : {"relative", "scaled_var", "sample_size"}
        The stopping rule the targeting stopped by.
    thresholds : FloatArray or None
        The threshold of every target under a declared rule, at the reported hazards.
    rule_ratio : float or None
        The largest score over its threshold under a declared rule, at the reported hazards.
    best_iterate : int or None
        The reported iterate under ``iterate="best"``, ``None`` under ``"last"``.
    bound_excludes_data : tuple of tuple of int
        ``(node, level code, cause index)`` of every cell that a node bound excludes, with
        the node counted from one.  The fluctuation pins such a cell at the clip.
    share_clipped : tuple of tuple
        ``(node, cause index, share)`` at every bounded node and cause: the share of the
        ``(unit, level)`` initial hazards that the bound clipped.
    phase : {"walk", "iterative", "jacobian"} or None
        For a separable fit, the targeting phase that reached the reported state: the
        one-step walk, the iterative TMLE, or Newton steps on the full score Jacobian.
        ``None`` for every other fit.
    nodes : tuple of HazardFit
        On a fit with covariates after baseline, the record of each node walk of the regimen's
        backward pass, node 1 first (:mod:`cleverly.longitudinal.hazard_backward`).  The
        regimen's record then sums the node scores and scales, each multiplied by
        :math:`n_k / n`, ``n_iter`` is the total of accepted steps, ``trace`` holds the final
        relative score of each node walk in walk order (the last node first), and
        ``initial_levels`` is node 1's initial plug-in.  That plug-in's continuation was
        fitted on the *targeted* later nodes, so it is not the untargeted estimator's
        plug-in; :func:`~cleverly.longitudinal.hazard_backward.untargeted` gives that one.
        The summed score of a stalled node is divided by the scale summed over every node and
        can cancel against another node's, so :attr:`relative_score` reads the largest node
        relative score instead.  Empty on every other fit.
    """

    algorithm: HazardAlgorithm
    targets: tuple[
        tuple[str, str | None, int] | tuple[str, str | None, int, str] | tuple[str, str | None],
        ...,
    ]
    converged: bool
    n_iter: int
    score_initial: FloatArray
    score: FloatArray
    score_scale: FloatArray
    trace: tuple[float, ...]
    initial_levels: FloatArray
    sigma: FloatArray | None = None
    norm_trace: tuple[float, ...] = ()
    walk_loss_trace: tuple[float, ...] = ()
    failure: TargetingFailure | None = None
    tol: float = 0.0
    stopping: str = "relative"
    thresholds: FloatArray | None = None
    rule_ratio: float | None = None
    best_iterate: int | None = None
    bound_excludes_data: tuple[tuple[int, int, int], ...] = ()
    share_clipped: tuple[tuple[int, int, float], ...] = ()
    phase: Literal["walk", "iterative", "jacobian"] | None = None
    nodes: tuple[HazardFit, ...] = ()

    @property
    def relative_score(self) -> float:
        """The largest target score relative to its scale, at the reported hazards.

        On a backward pass the largest over the node walks, each read on its own score and
        scale, so a stalled node walk is not diluted by the others.
        """
        if self.nodes:
            return max(node.relative_score for node in self.nodes)
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
    cause_order: Sequence[int] | None = None,
    slots: Sequence[MTPSlots] = (),
    fit_rows: BoolArray | None = None,
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
        A survival panel of a held design.  A covariate block at a node up to ``k`` enters
        the node-``k`` design through the mechanism history.
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
    cause_order : sequence of int or None
        The cause indices in fit order, ``None`` for code order.  A separable fit fits its
        declared first cause first, so that :math:`\\rho` of the second cause is its hazard
        among those free of the first.
    slots : sequence of MTPSlots
        The slots of each modified treatment policy at a continuous or mixed baseline node.
        Every slot is predicted from the same fit, at :func:`slot_design`, and the observed
        treatment is predicted at the fit's own design.  ``"pooled"`` is refused beside one
        (H24).
    fit_rows : BoolArray or None
        ``(n,)`` the rows every regression may train on, ``None`` for every row.  An outer
        fold's untargeted backward pass of a fit with covariates after baseline trains on its
        own training rows with the single fold, and predicts every row.

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
            data,
            levels,
            at_risk,
            events,
            learner=learner,
            folds=folds,
            n_jobs=n_jobs,
            cause_order=cause_order,
        )
    hazards = np.zeros((data.n, data.n_times, len(levels), n_causes))
    slot_hazards = {
        entry.label: np.zeros((data.n, data.n_times, entry.n_slots, n_causes)) for entry in slots
    }
    observed = np.zeros((data.n, data.n_times, n_causes)) if slots else None
    no_events: list[tuple[int, int]] = []
    diagnostics: list[tuple[object, ...]] = []
    for time in range(1, data.n_times + 1):
        design = data.history_design(time, include_current=True)
        predict = {
            plan.label: data.history_design(time, treatment=plan.values, include_current=True)
            for plan in levels
        }
        slot_keys = _slot_predictions(data, time, slots, design, predict)
        earlier = np.zeros(data.n, dtype=bool)
        taken = np.zeros((data.n, len(levels)))
        slot_taken = {label: np.zeros((data.n, len(keys))) for label, keys in slot_keys.items()}
        # Filed by cause index, so the nuisance report reads cause order whatever the fit
        # order.
        node_diagnostics: dict[int, tuple[object, ...]] = {}
        for cause in range(n_causes) if cause_order is None else cause_order:
            rows = at_risk[:, time - 1] & ~earlier
            if fit_rows is not None:
                rows = rows & fit_rows
            target = events[:, time - 1, cause]
            if not np.any(target[rows] * data.weights[rows] > 0):
                no_events.append((time, cause))
                node_diagnostics[cause] = ()
                rho = np.zeros((data.n, len(levels)))
                predictions = None
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
                node_diagnostics[cause] = tuple(fold_diagnostics)
                rho = (
                    np.column_stack([predictions[plan.label] for plan in levels])
                    if levels
                    else np.zeros((data.n, 0))
                )
            hazards[:, time - 1, :, cause] = rho * (1.0 - taken)
            taken = taken + hazards[:, time - 1, :, cause]
            for label, keys in slot_keys.items():
                slot_rho = (
                    np.zeros((data.n, len(keys)))
                    if predictions is None
                    else np.column_stack([predictions[key] for key in keys])
                )
                slot_hazards[label][:, time - 1, :, cause] = slot_rho * (1.0 - slot_taken[label])
                slot_taken[label] = slot_taken[label] + slot_hazards[label][:, time - 1, :, cause]
            if observed is not None:
                # Slot 0 of every policy is the observed treatment.
                observed[:, time - 1, cause] = slot_hazards[slots[0].label][:, time - 1, 0, cause]
            earlier = earlier | (target > 0)
        diagnostics.extend(node_diagnostics[cause] for cause in range(n_causes))
    return InitialHazards(
        hazards, tuple(sorted(no_events)), tuple(diagnostics), slot_hazards, observed
    )


#: The prediction key of the observed treatment's design, shared by every policy's slot 0.
_OBSERVED_SLOT = "\x00hazard-observed"


def _slot_predictions(
    data: LongitudinalData,
    time: int,
    slots: Sequence[MTPSlots],
    design: FloatArray,
    predict: dict[str, FloatArray],
) -> dict[str, tuple[str, ...]]:
    """Add each policy slot's design at node ``time`` to ``predict``; the keys by policy.

    The observed slot reads the fit's own design, so its prediction is the out-of-fold fit
    at the observed treatment.  A key never clashes with a level plan's label.
    """
    out: dict[str, tuple[str, ...]] = {}
    if slots:
        predict[_OBSERVED_SLOT] = design
    for entry in slots:
        keys = [_OBSERVED_SLOT]
        for slot in range(1, entry.n_slots):
            key = f"\x00hazard-slot-{entry.label}-{slot}"
            predict[key] = slot_design(data, time, entry, slot)
            keys.append(key)
        out[entry.label] = tuple(keys)
    return out


def _fit_pooled_hazards(
    data: LongitudinalData,
    levels: Sequence[Plan],
    at_risk: BoolArray,
    events: FloatArray,
    *,
    learner: Learner,
    folds: Folds,
    n_jobs: int,
    cause_order: Sequence[int] | None = None,
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
    by_cause: dict[int, tuple[object, ...]] = {}
    earlier = np.zeros((n, n_times), dtype=bool)
    taken = np.zeros((n, n_times, len(levels)))
    for cause in range(n_causes) if cause_order is None else cause_order:
        rows = at_risk & ~earlier
        target = events[:, :, cause]
        if not np.any((target * data.weights[:, None])[rows] > 0):
            no_events.extend((time, cause) for time in nodes)
            by_cause[cause] = ()
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
            by_cause[cause] = tuple(fold_diagnostics)
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
    # The nuisance report reads one entry per node and cause, node-major, as the per-node
    # path writes them. Each node of a cause repeats that cause's one pooled fit.
    diagnostics = tuple(by_cause[cause] for _ in nodes for cause in range(n_causes))
    return InitialHazards(hazards, tuple(sorted(no_events)), diagnostics)


def hazard_inputs(
    data: LongitudinalData,
    plans: Sequence[Plan],
    cumulative: Sequence[FloatArray],
    *,
    alpha: float,
    node_bounds: HazardBounds | None = None,
) -> HazardInputs:
    """Assemble the targeting inputs from the panel, the plans and the bounded mechanism.

    At a bounded node an empty cell is held at zero only under a lower bound of zero, and
    no full cell is held (finding M3 of the bounded-hazard design).  Every other such cell
    is an ordinary cell that the bound excludes.

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
    node_bounds : HazardBounds or None
        The bounded nodes.

    Returns
    -------
    HazardInputs
        The targeting inputs.
    """
    at_risk, events = event_indicators(data)
    inverse = np.stack([1.0 / np.asarray(c, dtype=float) for c in cumulative], axis=2)
    observed = np.asarray(np.nan_to_num(data.treatment[:, 0], nan=0.0), dtype=np.int64)
    weights = np.asarray(data.weights, dtype=float)
    empty, full, excluded = _held_cells(
        at_risk, events, weights, observed, inverse.shape[2], node_bounds
    )
    return HazardInputs(
        at_risk=at_risk,
        events=events,
        observed=observed,
        inverse=inverse,
        q=regimen_weights(data, plans),
        weights=weights,
        alpha=alpha,
        empty=empty,
        full=full,
        node_bounds=node_bounds,
        excluded=excluded,
    )


@dataclass(frozen=True)
class MTPSlots:
    """The slots of a modified treatment policy at a continuous or mixed baseline node.

    Slot 0 is each unit's observed treatment and slot ``b`` the treatment branch ``b`` of
    the policy assigns it, all of the unit's own values.  Every unit has its own values, so
    the hazards are predicted at each slot for each unit.

    Parameters
    ----------
    label : str
        The regimen's label.
    kind : {"continuous", "mixed"}
        The kind of the baseline node.
    values : tuple of FloatArray
        One entry per slot: the ``(n,)`` dose at a continuous node, the ``(n, d)`` component
        values at a mixed one.
    probabilities : FloatArray
        ``(n, B)`` the probability of each branch, the plug-in's mixture weights.
    keys : IntArray
        ``(n, 1 + B)`` the held-cell key of each slot: zero at a continuous node, whose
        cell is (node, cause), and the code of the slot's categorical component values at a
        mixed node, whose cell is (node, categorical values, cause).
    n_keys : int
        The number of keys.
    """

    label: str
    kind: str
    values: tuple[FloatArray, ...]
    probabilities: FloatArray
    keys: IntArray
    n_keys: int

    @property
    def n_slots(self) -> int:
        """The slots: the observed treatment and one per branch."""
        return len(self.values)


def mtp_slots(data: LongitudinalData, plan: Plan) -> MTPSlots | None:
    """The slots of ``plan``'s modified treatment policy at a continuous or mixed node.

    At a categorical node every slot value is a treatment level, so the walk keeps the
    level state and reads the policy's level map through its carry weights; this returns
    ``None`` there, and on a plan with no policy at the baseline decision.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel of a held design.
    plan : Plan
        A resolved plan.

    Returns
    -------
    MTPSlots or None
        The slots, built from the plan's branch targets (``Plan.carry_targets(1)``).
    """
    kind = data.node_kind(1)
    if kind == "categorical" or not plan.is_mtp_node(1):
        return None
    targets = plan.carry_targets(1)
    assert targets is not None
    observed = np.nan_to_num(np.asarray(data.node_values(1), dtype=float), nan=0.0)
    if kind == "continuous":
        values: tuple[FloatArray, ...] = (
            observed[:, 0],
            *(np.asarray(dose, dtype=float).reshape(-1) for dose in targets),
        )
        keys = np.zeros((data.n, len(values)), dtype=np.int64)
        n_keys = 1
    else:
        values = (observed, *(np.asarray(block, dtype=float) for block in targets))
        # The categorical components of each slot, which key its held cell.
        labelled = [index for index, levels in enumerate(data.component_levels[0]) if levels]
        if labelled:
            stacked = np.stack([block[:, labelled] for block in values], axis=1)  # (n, U, c)
            codes, inverse = np.unique(
                stacked.reshape(-1, len(labelled)), axis=0, return_inverse=True
            )
            keys = np.asarray(inverse, dtype=np.int64).reshape(data.n, len(values))
            n_keys = int(codes.shape[0])
        else:
            # A vector of doses alone: its cell is (node, cause), as at a continuous node.
            keys = np.zeros((data.n, len(values)), dtype=np.int64)
            n_keys = 1
    probabilities = np.asarray(plan.policy_at(1), dtype=float)
    return MTPSlots(plan.label, kind, values, probabilities, keys, n_keys)


def slot_design(data: LongitudinalData, time: int, slots: MTPSlots, slot: int) -> FloatArray:
    """The hazard design of node ``time`` at slot ``slot`` of a modified treatment policy.

    At a continuous node it is :meth:`~LongitudinalData.history_design` at the slot's dose.
    At a mixed node :meth:`~LongitudinalData.history_design` ignores an assignment and
    reads the observed components, so the node-1 component block of the observed design is
    replaced by :meth:`~LongitudinalData.component_design` at the slot's values.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel of a held design.
    time : int
        The node, counted from one.
    slots : MTPSlots
        The policy's slots.
    slot : int
        The slot, zero for the observed treatment.

    Returns
    -------
    FloatArray
        The design, one row per unit.
    """
    values = slots.values[slot]
    if slots.kind == "continuous":
        assigned = np.repeat(values[:, None], data.n_times, axis=1)
        return data.history_design(time, treatment=assigned, include_current=True)
    observed = data.history_design(time, include_current=True)
    block = data.component_design(1)
    width = block.shape[1]
    # On a held design the node-1 block is the last one: every later node is held.
    if not np.array_equal(observed[:, -width:], block):  # pragma: no cover - layout guard
        raise AssertionError("the node-1 component block is not the last block of the design")
    return np.hstack([observed[:, :-width], data.component_design(1, values=values)])


def mtp_inputs(
    data: LongitudinalData,
    plan: Plan,
    slots: MTPSlots | None,
    cumulative: FloatArray,
    level_cumulative: Sequence[FloatArray],
    *,
    alpha: float,
    node_bounds: HazardBounds | None = None,
) -> HazardInputs:
    r"""The weighted targeting inputs of one modified treatment policy regimen.

    The loss weight of node :math:`k` is
    :math:`\omega_k = w \cdot` ``plan.cumulative_numerator(data)[:, k]`` /
    ``cumulative[:, k]``, the convention of the iterated-mean fit of the same plan: at a
    categorical node the numerator is :math:`g^d(A \mid W)` and the bounded cumulative holds
    :math:`g(A \mid W) G_k`, and at a continuous or mixed node the numerator is the ratio
    :math:`r` and the treatment factor of the cumulative is one.

    At a categorical node the slots are the treatment levels and the cells are the shipped
    level cells.  At a continuous node every slot of a unit shares the (node, cause) cell, and
    at a mixed node the (node, categorical values, cause) cell of its categorical values.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel of a held design.
    plan : Plan
        The policy's plan, with its ratio numerator attached.
    slots : MTPSlots or None
        The slots at a continuous or mixed node, ``None`` at a categorical one.
    cumulative : FloatArray
        ``(n, T)`` the bounded cumulative mechanism of the plan, evaluated at the plan
        alone.
    level_cumulative : sequence of FloatArray
        One ``(n, T)`` bounded :math:`g G_k` per level at a categorical node, empty
        elsewhere.  Only the cells are read from it.
    alpha : float
        The probability bound.
    node_bounds : HazardBounds or None
        The bounded nodes.

    Returns
    -------
    HazardInputs
        The inputs of the policy's own walk.
    """
    numerator = plan.cumulative_numerator(data)
    assert numerator is not None
    denominator = np.asarray(cumulative, dtype=float)
    ratio = np.divide(
        numerator, denominator, out=np.zeros(denominator.shape), where=denominator > 0.0
    )
    weights = np.asarray(data.weights, dtype=float)
    loss_weight = weights[:, None] * ratio
    at_risk, events = event_indicators(data)
    if slots is None:
        # The slots are the levels: the observed level keys each unit's cell, and the
        # policy's carry over the levels is ``q``.
        n_slots = len(level_cumulative)
        observed = np.asarray(
            np.nan_to_num(np.asarray(data.node_values(1), dtype=float)[:, 0], nan=0.0),
            dtype=np.int64,
        )
        cell_keys, n_cells, cells = observed, n_slots, None
        q = regimen_weights(data, [plan])
    else:
        n_slots = slots.n_slots
        observed = np.zeros(data.n, dtype=np.int64)
        cell_keys, n_cells, cells = slots.keys[:, 0], slots.n_keys, slots.keys
        q = np.concatenate([np.zeros((data.n, 1)), slots.probabilities], axis=1)[None]
    empty, full, excluded = _held_cells(at_risk, events, weights, cell_keys, n_cells, node_bounds)
    return HazardInputs(
        at_risk=at_risk,
        events=events,
        observed=observed,
        inverse=np.repeat(ratio[:, :, None], n_slots, axis=2),
        q=q,
        weights=weights,
        alpha=alpha,
        empty=empty,
        full=full,
        node_bounds=node_bounds,
        excluded=excluded,
        loss_weight=loss_weight,
        cells=cells,
    )


def _held_cells(
    at_risk: BoolArray,
    events: FloatArray,
    weights: FloatArray,
    observed: IntArray,
    n_cells: int,
    node_bounds: HazardBounds | None,
) -> tuple[BoolArray, BoolArray, BoolArray | None]:
    """The ``(T, K, J)`` empty, ``(T, K)`` full and excluded cells over ``n_cells`` keys.

    A cell is the rows at risk of positive weight whose observed key is ``c``: the
    treatment level on a level state, and the key of the observed slot on a slot state.
    At a bounded node an empty cell is held at zero only under a lower bound of zero, and
    no full cell is held (finding M3 of the bounded-hazard design).  Every other such cell
    is an ordinary cell that the bound excludes.
    """
    # (n, T, K): the rows at risk of positive weight, by observed key.
    followers = (at_risk & (weights > 0.0)[:, None])[:, :, None] & (
        observed[:, None, None] == np.arange(n_cells)[None, None, :]
    )
    present = followers.any(axis=0)
    happened = events > 0.0
    with_event = np.einsum("nkl,nkj->klj", followers, happened)
    without_event = np.einsum("nkl,nk->kl", followers, ~happened.any(axis=-1))
    empty = present[:, :, None] & (with_event == 0)
    full = present & (without_event == 0)
    excluded = None
    if node_bounds is not None and node_bounds.active:
        at = node_bounds.bounded[:, None, None]
        floor = (node_bounds.lower > 0.0)[:, None, :]
        held = empty & ~(at & floor)
        excluded = at & ((empty & floor) | (full[:, :, None] & ~held))
        empty = held
        full = full & ~node_bounds.bounded[:, None]
    return empty, full, excluded


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
    """``(R, J, H, n)`` score term of every unit, and ``(R, J, H)`` score scale."""
    terms, scale, _ = _score_parts(state, inputs, horizons)
    return terms, scale


def _score_parts(
    state: HazardState, inputs: HazardInputs, horizons: Sequence[int]
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """The score terms and scale of :func:`_score_terms`, and each unit's scale row.

    The third array is ``(R, J, H, n)``: the term of each unit in the mean that gives the
    scale, which a working-model fit sums over the cells with weights of its own.

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
    rows = np.empty((inputs.n_regimens, n_causes, len(horizons), inputs.n))
    mask = inputs.at_risk
    for h, t in enumerate(horizons):
        nodes = np.arange(inputs.n_times) < t
        for j in range(n_causes):
            c = _per_survivor(
                obs.incidence[:, t - 1, j][:, None] - obs.incidence[:, :, j], obs.survival
            )
            size = np.abs(1.0 - c) + (n_causes - 1) * np.abs(c)
            per_row = np.where(mask & nodes[None, :], inverse * size, 0.0).sum(axis=1)
            row = weight[None, :] * q * per_row[None, :]
            rows[:, j, h] = row
            scale[:, j, h] = row.mean(axis=1)
    return terms, scale, rows


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
    covariate is materialised.  At a bounded node the increment of :math:`\\theta_j` is that
    sum times the fixed scale :math:`s_j`.

    Parameters
    ----------
    state : HazardState
        The current hazards.
    inputs : HazardInputs
        The targeting inputs.
    horizons : sequence of int
        The target horizons, counted from one.
    coefficients : FloatArray
        ``(R, J, H)`` coefficient of every target, or ``(n, R, J, H)`` coefficients that
        differ by unit, as the cell coefficients of a working-model target do
        (:class:`MSMTargets`).

    Returns
    -------
    FloatArray
        The logit increment per unit coefficient.
    """
    # A shared coefficient is a unit axis of length one, which broadcasts over the units
    # with the same arithmetic as the per-unit array.
    per_unit = coefficients if coefficients.ndim == 4 else coefficients[None]
    _, n_regimens, n_causes, _ = per_unit.shape
    by_node = np.zeros((per_unit.shape[0], n_regimens, n_causes, inputs.n_times))
    for h, t in enumerate(horizons):
        by_node[..., t - 1] += per_unit[..., h]
    # Sums over t >= k: a reversed running sum over the nodes.
    tail = np.flip(np.cumsum(np.flip(by_node, axis=3), axis=3), axis=3)
    incidence, survival = state.incidence, state.survival
    out = np.zeros_like(state.logits)
    for r in range(n_regimens):
        inner = np.zeros_like(state.logits)
        for j in range(n_causes):
            weighted = by_node[:, r, j][:, :, None] * incidence[..., j]
            expected = np.flip(np.cumsum(np.flip(weighted, axis=1), axis=1), axis=1)
            spill = _per_survivor(
                expected - incidence[..., j] * tail[:, r, j][:, :, None], survival
            )
            inner -= spill[..., None]
            inner[..., j] += tail[:, r, j][:, :, None]
        if inputs.weighted:
            # The weight sits in the loss, so the covariate at every slot is kappa alone.
            out += inner
        else:
            out += inputs.q[r][:, None, :, None] * inner
    step = out if inputs.weighted else np.asarray(inputs.inverse[..., None] * out, dtype=float)
    bounds = state.node_bounds
    if bounds is not None and bounds.active:
        step = step * bounds.scale[None, :, None, :]
    return step


def _quasi_terms(events: FloatArray, logits: FloatArray, bounds: HazardBounds) -> FloatArray:
    """``(n, K)`` quasi-binomial log-likelihood of the ``K`` bounded nodes, summed over causes.

    The outcome of cause :math:`j` is :math:`(dN_j - l_j) / s_j` and its mean is
    :math:`\\operatorname{expit}(\\theta_j)`.  The outcome lies outside ``[0, 1]`` when
    :math:`l_j > 0` and no event occurs, which the quasi-likelihood allows.
    """
    k = bounds.bounded
    p = expit(logits[:, k])
    y = (events[:, k] - bounds.lower[k][None]) / bounds.scale[k][None]
    return np.asarray((xlogy(y, p) + xlogy(1.0 - y, 1.0 - p)).sum(axis=-1), dtype=float)


def walk_loss(state: HazardState, inputs: HazardInputs, *, last_node: int | None = None) -> float:
    """The walk loss of the observed events on the risk set.

    It is the negative weighted multinomial log-likelihood at every unbounded node, plus the
    negative weighted quasi-binomial log-likelihood of :func:`_quasi_terms` at every bounded
    node.  A one-step walk records it after each accepted step.

    Parameters
    ----------
    state : HazardState
        The current hazards.
    inputs : HazardInputs
        The targeting inputs.
    last_node : int or None, default=None
        Count only the person-nodes ``k <= last_node``, counted from one.  The collaborative
        survival selector scores a fluctuation at one horizon, which leaves the later nodes
        unchanged.  ``None`` counts every node.

    Returns
    -------
    float
        The loss.
    """
    counted, base = _walk_weights(inputs, last_node=last_node)
    return -float(np.sum(np.where(counted, base * _walk_terms(state, inputs), 0.0)))


def _walk_weights(
    inputs: HazardInputs, *, last_node: int | None = None
) -> tuple[BoolArray, FloatArray]:
    """``(n, T)`` the person-nodes the walk loss counts, and the weight of each.

    A person-node counts when the unit is at risk there with a nonzero weight, and, under
    ``last_node``, when the node is at most ``last_node``.  The weight is
    :meth:`HazardInputs.node_weights`: the observation weight, or the loss weight
    :math:`\\omega_k` of a modified treatment policy walk.  The weight is not zeroed outside
    the mask: a term there can be infinite (a unit not at risk at a full cell), and
    ``0 * inf`` warns.  Every reader takes ``np.where(counted, base * x, 0.0)``.

    Parameters
    ----------
    inputs : HazardInputs
        The targeting inputs.
    last_node : int or None, default=None
        The last node counted, counted from one.  ``None`` counts every node.

    Returns
    -------
    counted : BoolArray
        The counted person-nodes.
    base : FloatArray
        The weight of each person-node.
    """
    base = inputs.node_weights()
    counted = inputs.at_risk & (base != 0.0)
    if last_node is not None:
        counted = counted & (np.arange(inputs.n_times) < last_node)[None, :]
    return counted, base


def _walk_terms(state: HazardState, inputs: HazardInputs) -> FloatArray:
    """``(n, T)`` the log-likelihood term of each person-node at its observed level.

    The multinomial term at an unbounded node and the quasi-binomial term of
    :func:`_quasi_terms` at a bounded one, before the weights and the risk set.
    """
    rows, levels = inputs.rows, inputs.observed
    hazards = state.hazards[rows, :, levels]
    none = 1.0 - hazards.sum(axis=-1)
    terms = xlogy(inputs.events, hazards).sum(axis=-1) + xlogy(
        1.0 - inputs.events.sum(axis=-1), none
    )
    bounds = state.node_bounds
    if bounds is not None and bounds.active:
        terms[:, bounds.bounded] = _quasi_terms(
            inputs.events, state.logits[rows, :, levels], bounds
        )
    return np.asarray(terms, dtype=float)


def _clip_tangent(
    logits: FloatArray,
    move: FloatArray,
    bounds: tuple[float, float],
    bounded: BoolArray | None = None,
) -> FloatArray:
    """The right derivative in ``t`` of ``clip_logits(logits + t * move)`` at ``t = 0``.

    ``logits`` are the logits before the clip, finite or :math:`-\\infty`.  Per unit and node:
    the top cause is the largest logit, ties broken toward the largest ``move``, which stays
    largest for ``t`` just above 0.  The ratio-keeping shift is active when the top lies above
    ``high``, or at ``high`` and moving up, except at a node of ``bounded``.  Under the shift
    each cause moves at its own rate less the top's, so the top stays at ``high``.  A cause at
    or past a bound and moving outward is held (0), as is a logit at :math:`-\\infty`.

    The derivative is one-sided from the right because the step moves to ``t > 0``.  Clipped
    logits start exactly on a bound, and the left derivative would count a logit at ``high``
    moving up as moving, although the clip holds it.

    Parameters
    ----------
    logits : FloatArray
        ``(n, T, ..., J)`` logits, the node on the second axis.
    move : FloatArray
        The move, the same shape.
    bounds : tuple of float
        The clip interval.
    bounded : BoolArray or None, default=None
        ``(T,)`` the bounded nodes.

    Returns
    -------
    FloatArray
        The tangent, the same shape.
    """
    z = np.asarray(logits, dtype=float)
    m = np.asarray(move, dtype=float)
    low, high = bounds
    finite = np.isfinite(z)
    top = np.max(z, axis=-1)
    ties = np.where(finite & (z == top[..., None]), m, -np.inf)
    leader = np.take_along_axis(m, np.argmax(ties, axis=-1)[..., None], axis=-1)[..., 0]
    over = np.isfinite(top) & ((top > high) | ((top == high) & (leader > 0.0)))
    if bounded is not None:
        node = np.asarray(bounded, dtype=bool).reshape((1, -1) + (1,) * (z.ndim - 3))
        over = over & ~node
    with np.errstate(invalid="ignore"):
        shifted = np.where(over[..., None], (z - top[..., None]) + high, z)
    rate = np.where(over[..., None], m - leader[..., None], m)
    free = (shifted > low) & (shifted < high)
    rising = (shifted == low) & (rate > 0.0)
    falling = (shifted == high) & (rate < 0.0)
    tangent = np.where(free | rising | falling, rate, 0.0)
    return np.asarray(np.where(finite, tangent, 0.0), dtype=float)


@dataclass(frozen=True)
class _WalkLine:
    """The walk loss along one move, and its exact slope in the length.

    Parameters
    ----------
    state : HazardState
        The hazards at length 0.
    inputs : HazardInputs
        The targeting inputs.
    move : FloatArray
        ``(n, T, L, J)`` the logit move of unit length.
    logits : FloatArray
        ``(n, T, J)`` the state's logits at the observed level.
    step : FloatArray
        ``(n, T, J)`` the move at the observed level.
    counted : BoolArray
        ``(n, T)`` the counted person-nodes of :func:`_walk_weights`.
    base : FloatArray
        ``(n, T)`` their weights.
    total : float
        The sum of the observation weights, which divides the loss.
    bounded : BoolArray or None
        ``(T,)`` the bounded nodes, ``None`` when no node is bounded.
    """

    state: HazardState
    inputs: HazardInputs
    move: FloatArray
    logits: FloatArray
    step: FloatArray
    counted: BoolArray
    base: FloatArray
    total: float
    bounded: BoolArray | None

    @classmethod
    def along(cls, state: HazardState, inputs: HazardInputs, move: FloatArray) -> _WalkLine:
        """The line from ``state`` along ``move``."""
        rows, levels = inputs.rows, inputs.observed
        counted, base = _walk_weights(inputs)
        node_bounds = state.node_bounds
        bounded = node_bounds.bounded if node_bounds is not None and node_bounds.active else None
        return cls(
            state,
            inputs,
            move,
            np.asarray(state.logits[rows, :, levels], dtype=float),
            np.asarray(move[rows, :, levels], dtype=float),
            counted,
            base,
            float(np.sum(inputs.weights)),
            bounded,
        )

    def loss(self, t: float) -> float:
        """The walk loss at length ``t``, divided by the total weight."""
        return walk_loss(self.state.moved(t * self.move, self.inputs.bounds), self.inputs) / (
            self.total
        )

    def slope(self, t: float) -> float:
        """The right derivative of :meth:`loss` at length ``t``.

        It is :math:`-\\sum W G \\tau / \\sum w`.  :math:`G` is the derivative of a
        person-node's term in its clipped logit: :math:`dN_j - \\lambda_j` at an unbounded
        node, full cells included, and :math:`(dN_j - l_j) / s_j - \\operatorname{expit}
        (\\theta_j)` at a bounded one.  :math:`\\tau` is :func:`_clip_tangent` of the
        moved logits.
        """
        inputs = self.inputs
        rows, levels = inputs.rows, inputs.observed
        moved = self.state.moved(t * self.move, inputs.bounds)
        logits = np.where(np.isneginf(self.logits), -np.inf, self.logits + t * self.step)
        tangent = _clip_tangent(logits, self.step, inputs.bounds, self.bounded)
        grad = inputs.events - moved.hazards[rows, :, levels]
        node_bounds = self.state.node_bounds
        if self.bounded is not None and node_bounds is not None:
            k = self.bounded
            outcome = (inputs.events[:, k] - node_bounds.lower[k][None]) / node_bounds.scale[k][
                None
            ]
            grad[:, k] = outcome - expit(moved.logits[rows, :, levels][:, k])
        summand = self.base[..., None] * grad * tangent
        return -float(np.sum(np.where(self.counted[..., None], summand, 0.0))) / self.total

    def mass(self) -> float:
        """The absolute mass of the loss at length 0, which scales the descent check."""
        terms = _walk_terms(self.state, self.inputs)
        return float(np.sum(np.abs(np.where(self.counted, self.base * terms, 0.0)))) / self.total


def _clever_observed(state: HazardState, inputs: HazardInputs, horizon: int) -> FloatArray:
    """``(R * J, n, T, J)`` :math:`h^m_l(k, A, W)` at the observed level, for one horizon."""
    obs = _observed(state, inputs)
    # A weighted walk's covariate is kappa alone: its ratio sits in the loss weight.
    inverse = np.ones(inputs.at_risk.shape) if inputs.weighted else inputs.inverse_observed()
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
    state: HazardState, inputs: HazardInputs, steps: int, max_steps: int, horizon: int
) -> TargetingFailure:
    """The cause of an unconverged hazard fit through node ``horizon``.

    An excluded cell enters the scores only at a node at or before the horizon, so a cell
    after it does not explain the failure and the step count is read instead.
    """
    if inputs.excluded is not None and inputs.excluded[:horizon].any():
        return "bound_excludes_data"
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
    stopping: str = "relative",
    stop: StopContext | None = None,
    step_bound: float | None = None,
    step_search: bool = True,
    iterate: str = "last",
) -> tuple[HazardState, HazardFit]:
    """Target every regimen's and cause's parameter at one horizon (part (a)).

    Each outer iteration moves every level's hazards along the current clever covariates
    by :math:`\\epsilon \\in \\mathbb R^{RJ}`, then rebuilds :math:`h`.  The step is a
    Newton step on the scores themselves (:func:`_secant_step`).  When that step does not
    reduce the scores, the iteration takes the maximum-likelihood fluctuation with
    :math:`h` held fixed instead (Moore and van der Laan 2009, (13)).  It stops when every
    target's relative score is at most ``tol``, or after ``max_iter`` iterations.

    Under a declared ``stopping`` rule the loop stops when the largest score over its
    threshold is at most one (:func:`~cleverly.fluctuation._score.rule_ratio_at`), and the
    inner fallback fit tightens its tolerance to ``min(tol, 1e-12)``.  Under ``step_bound``
    every step is the constrained step of :func:`_bounded_hazard_step`, and under its search the
    loop also stops when the logits return to a state it has visited, since the step is a
    function of the state and the loop would repeat the cycle.  Under
    ``iterate="best"`` the fit reports the iterate ``k >= 1`` with the smallest raw score
    2-norm, the first of tied iterates, and judges ``converged``, ``failure`` and the rule
    ratio there.

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
    stopping : {"relative", "scaled_var", "sample_size"}
        The stopping rule.
    stop : StopContext or None
        The units and the span a declared rule reads.
    step_bound : float or None
        The longest step in :math:`\\epsilon`, ``None`` for the shipped step.
    step_search : bool
        Whether a bounded step searches its length.
    iterate : {"last", "best"}
        Which iterate the fit reports.

    Returns
    -------
    tuple
        The targeted state and its record.
    """
    declared = stopping != "relative"
    if declared and stop is None:
        raise ValueError(STOP_CONTEXT_MISSING)
    horizons = (horizon,)
    bounds = inputs.bounds
    state = initial
    score, scale = scores(state, inputs, horizons)
    first = score.reshape(-1).copy()
    trace = [relative_score(score.reshape(-1), scale.reshape(-1))]
    initial_levels = state.levels(inputs.q, inputs.weights)[:, :, horizon - 1].reshape(-1)
    iterations = 0
    shape = (inputs.n_regimens, inputs.n_causes, 1)
    thresholds: FloatArray | None = None
    ratio: float | None = None
    if declared:
        assert stop is not None
        thresholds, ratio = _rule_state(state, inputs, horizons, score, stop, stopping)
    best = BestIterate() if iterate == "best" else None
    inner_tol = min(tol, 1e-12) if declared else tol * 1e-2
    visited = _Visited(state.logits) if step_bound is not None and step_search else None

    def met() -> bool:
        # ``not > tol`` rather than ``<= tol``: a NaN relative score stops the default rule,
        # as the shipped ``while trace[-1] > tol`` did.
        return ratio <= 1.0 if ratio is not None else not trace[-1] > tol

    while not met() and iterations < max_iter:
        size = inputs.n_regimens * inputs.n_causes
        basis = [
            direction(state, inputs, horizons, np.eye(size)[m].reshape(shape)) for m in range(size)
        ]
        if step_bound is not None:
            moved = _bounded_hazard_step(state, inputs, basis, score, step_bound, step_search)
            if moved is None:
                break
        else:
            moved = _secant_step(state, inputs, horizons, basis, score, scale)
        if moved is None:
            # The score equations did not move toward zero along the Newton step, so fall
            # back to the maximum-likelihood fluctuation with h held fixed (Moore (13)).
            epsilon = _newton(state, inputs, _clever_observed(state, inputs, horizon), inner_tol)
            if not np.all(np.isfinite(epsilon)):
                break
            moved = state.moved(np.tensordot(epsilon, np.asarray(basis), axes=1), bounds)
        state = moved
        iterations += 1
        score, scale = scores(state, inputs, horizons)
        trace.append(relative_score(score.reshape(-1), scale.reshape(-1)))
        if declared:
            assert stop is not None
            thresholds, ratio = _rule_state(state, inputs, horizons, score, stop, stopping)
        if best is not None:
            best.offer(iterations, iterate_norm(score), (state, score, scale, thresholds, ratio))
        if visited is not None and not met() and visited.seen(state.logits):
            # The searched step is a function of the state: a revisit would repeat its cycle.
            break
    reported: int | None = None
    if best is not None:
        reported = iterations
        if best.state is not None and best.k is not None:
            reported = best.k
            state, score, scale, thresholds, ratio = best.state
    relative = relative_score(score.reshape(-1), scale.reshape(-1))
    converged = bool(ratio <= 1.0) if ratio is not None else bool(relative <= tol)
    if not converged and warn:
        detail = (
            f"stopped after {iterations} iteration(s) with a relative score of "
            f"{relative:.3g} > {tol:g}"
            if ratio is None
            else f"did not meet the {stopping} stopping rule after {iterations} iteration(s) "
            f"(largest score over its threshold = {ratio:.3g})"
        )
        _warn(
            f"hazard targeting at horizon {horizon} {detail}. Raise max_iter, or check "
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
        failure=None if converged else _classify(state, inputs, iterations, max_iter, horizon),
        tol=tol,
        stopping=stopping,
        thresholds=thresholds,
        rule_ratio=ratio,
        best_iterate=reported,
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


def _newton(
    state: HazardState, inputs: HazardInputs, clever: FloatArray, inner_tol: float
) -> FloatArray:
    """Fit the fluctuation coefficients with :math:`h` held fixed, from zero.

    At an unbounded node the loss is the multinomial log-likelihood, over the causes alone in a
    full cell.  At a bounded node it is
    the quasi-binomial loss of each cause, the logit moves by :math:`s_j \\epsilon \\cdot h_j`,
    and the information is :math:`\\sum_j (s_j h_j)^2 p_j (1 - p_j)` with no cross-cause
    term.  The gradient is the score :math:`h_j (dN_j - \\lambda_j)` at both kinds of node.
    The fit stops when every relative gradient is at most ``inner_tol``.
    """
    rows, levels = inputs.rows, inputs.observed
    base = state.logits[rows, :, levels]
    held = np.isneginf(base)
    cut = inputs.bounds
    # (n, 1), or the (n, T) loss weight of a weighted walk: every use broadcasts over nodes.
    w = inputs.node_weights() if inputs.weighted else inputs.weights[:, None]
    mask = inputs.at_risk
    n = inputs.n
    epsilon = np.zeros(clever.shape[0])
    bounds = state.node_bounds
    # (n, T) the observed cells whose all-cause hazard is held at one: their softmax has no
    # no-event category, as in HazardState.  hazard_inputs holds no full cell at a bounded node.
    if state.full is None:
        full = np.zeros(mask.shape, dtype=bool)
    elif np.ndim(state.full) == 3:
        full = np.asarray(state.full, dtype=bool)[rows, :, levels]
    else:
        full = np.asarray(state.full, dtype=bool)[:, levels].T

    def softmax(eta: FloatArray) -> tuple[FloatArray, FloatArray]:
        expo = np.exp(eta)
        events = expo.sum(axis=-1)
        denom = np.where(full, events, 1.0 + events)
        return expo / denom[..., None], np.where(full, 0.0, 1.0 / denom)

    def evaluate(eps: FloatArray) -> tuple[float, FloatArray, FloatArray]:
        eta = np.where(held, -np.inf, clip_logits(base + np.tensordot(eps, clever, axes=1), cut))
        lam, none = softmax(eta)
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

    def evaluate_bounded(eps: FloatArray) -> tuple[float, FloatArray, FloatArray]:
        assert bounds is not None
        k = bounds.bounded
        step = bounds.scale[None] * np.tensordot(eps, clever, axes=1)
        eta = np.where(held, -np.inf, clip_logits(base + step, cut, k))
        lam, none = softmax(eta)
        p = expit(eta)
        lam[:, k] = bounds.lower[k][None] + bounds.scale[k][None] * p[:, k]
        none[:, k] = 1.0 - lam[:, k].sum(axis=-1)
        terms = xlogy(inputs.events, lam).sum(-1) + xlogy(1.0 - inputs.events.sum(-1), none)
        y = (inputs.events[:, k] - bounds.lower[k][None]) / bounds.scale[k][None]
        terms[:, k] = (xlogy(y, p[:, k]) + xlogy(1.0 - y, 1.0 - p[:, k])).sum(-1)
        value = float(np.sum(np.where(mask & (w != 0.0), w * terms, 0.0)))
        residual = np.where(mask[..., None], inputs.events - lam, 0.0)
        gradient = np.einsum("mnkl,nkl->m", clever, w[..., None] * residual) / n
        free = np.where(mask & ~k[None, :], w, 0.0)[None, ..., None]
        mean = np.einsum("mnkl,nkl->mnk", clever, lam)
        info_free = np.einsum("mnkl,pnkl->mp", clever * free, clever * lam[None]) - np.einsum(
            "mnk,pnk->mp", mean * free[..., 0], mean
        )
        tied = np.where(mask & k[None, :], w, 0.0)[None, ..., None]
        scaled = clever * bounds.scale[None, None]
        info_bounded = np.einsum("mnkl,pnkl->mp", scaled * tied, scaled * (p * (1.0 - p))[None])
        return value, gradient, (info_free + info_bounded) / n

    solve = evaluate if bounds is None or not bounds.active else evaluate_bounded
    value, gradient, info = solve(epsilon)
    scale = np.maximum(np.abs(clever).sum(-1)[:, mask].sum(-1) / n, 1e-300)
    for _ in range(_NEWTON_ITER):
        if np.max(np.abs(gradient) / scale) <= inner_tol:
            break
        step = np.linalg.lstsq(info, gradient, rcond=None)[0]
        accepted = False
        length = 1.0
        while length > _MIN_STEP:
            candidate = epsilon + length * step
            new_value, new_gradient, new_info = solve(candidate)
            if new_value >= value - 1e-12 * abs(value):
                epsilon, value, gradient, info = candidate, new_value, new_gradient, new_info
                accepted = True
                break
            length *= 0.5
        if not accepted:
            break
    return epsilon


def _rule_state(
    state: HazardState,
    inputs: HazardInputs,
    horizons: Sequence[int],
    score: FloatArray,
    stop: StopContext,
    stopping: str,
    combination: MSMTargets | None = None,
) -> tuple[FloatArray, float]:
    """Thresholds and the largest score over its threshold, at ``state``.

    The contribution of a target is its influence curve under unit weights, and its score
    part under other weights, as declared difference D4 of the targeting options reads a
    point fit (finding M4 of the bounded-hazard design).  A working-model target sums the
    contributions of its cells through ``combination``.
    """
    if np.allclose(inputs.weights, 1.0, rtol=0.0, atol=1e-12):
        _, curves = influence(state, inputs, horizons)
    else:
        terms, _ = _score_terms(state, inputs, horizons)
        curves = terms * inputs.weights
    if combination is not None:
        curves = combination.combine(curves)
    contributions = curves.reshape(-1, inputs.n).T
    return rule_ratio_at(score.reshape(-1), contributions, stop, stopping)


def _bounded_hazard_step(
    state: HazardState,
    inputs: HazardInputs,
    basis: Sequence[FloatArray],
    score: FloatArray,
    step_bound: float,
    step_search: bool,
) -> HazardState | None:
    r"""The constrained step of ``tmle3`` on the per-horizon fluctuation.

    With :math:`d = s / \lVert s \rVert_2` and :math:`s` the raw score of every target at
    the horizon, the step is :math:`t^* d` in :math:`\epsilon`.  Under ``step_search`` the
    length minimises :func:`walk_loss` along :math:`d` with :math:`h` held fixed, over
    :math:`t \in [0, \delta]`.  :func:`~cleverly.fluctuation.iterative._line_step` finds it
    as the root of the exact slope of :class:`_WalkLine`, to machine precision.  The slope is
    the right derivative at the kinks of the clip, because the step moves forward.

    With no logit held at a bound against the move, the slope at zero is
    :math:`-\lVert s \rVert` (weights have mean one).  A logit held at a bound drops out of
    the slope but not out of the score, so the slope at zero can be larger, and even
    non-negative: the length is then 0.  The clip can also make the line loss non-unimodal,
    so the step is checked for descent, :math:`L(t^*) \le L(0) + 64\,\epsilon\,A` with
    :math:`A` the absolute mass of the loss, and halves until it passes.  Without the search
    the length is :math:`\delta`.  ``None`` comes back when the score is zero or the length
    is zero, so nothing moves.
    """
    current = score.reshape(-1)
    norm = float(np.linalg.norm(current))
    if norm == 0.0:
        return None
    move = np.tensordot(current / norm, np.asarray(basis), axes=1)
    bounds = inputs.bounds
    length = float(step_bound)
    if step_search:
        line = _WalkLine.along(state, inputs, move)
        length = _line_step(line.slope, length, loss=line.loss, mass=line.mass())
    if length <= 0.0:
        return None
    return state.moved(length * move, bounds)


@dataclass(frozen=True)
class MSMTargets:
    r"""The coefficient scores of a working model, as per-unit sums of the cell scores.

    Cell :math:`c = (r, t)` of the working model reads target ``(r, j, t)`` of the walk for
    each cause :math:`j`.  With :math:`C_{icm} = h(c, V_i)\varphi_m(c, V_i)`, the model's
    :attr:`~cleverly.longitudinal.msm.RegimenMSM.weighted_design`, the score of coefficient
    :math:`m` of cause :math:`j` is :math:`P_n\, w \sum_c C_{icm}\, \mathrm{term}_{cj}(O_i)`,
    and the walk's direction along a coefficient vector :math:`\gamma_j` is :func:`direction`
    with the per-unit cell coefficients :math:`\sum_m \gamma_{jm} C_{icm}`.  Targeting the
    cells instead solves these scores only when :math:`h\varphi` does not vary with
    :math:`V`.

    Parameters
    ----------
    model : RegimenMSM
        The evaluated working model.
    regimen : IntArray
        ``(C,)`` regimen index of each cell, in the walk's regimen order.
    horizon : IntArray
        ``(C,)`` index of each cell's horizon among the walk's horizons.
    n_regimens : int
        The regimens the walk targets.
    n_horizons : int
        The horizons the walk targets.
    """

    model: RegimenMSM
    regimen: IntArray
    horizon: IntArray
    n_regimens: int
    n_horizons: int

    @classmethod
    def of(cls, model: RegimenMSM, labels: Sequence[str], horizons: Sequence[int]) -> MSMTargets:
        """The targets of ``model`` over regimens ``labels`` and ``horizons``.

        Parameters
        ----------
        model : RegimenMSM
            The evaluated working model.
        labels : sequence of str
            The regimen labels, in the walk's order.
        horizons : sequence of int
            The walk's horizons, counted from one.

        Returns
        -------
        MSMTargets
            The cell maps.
        """
        labels, horizons = list(labels), list(horizons)
        return cls(
            model,
            np.array([labels.index(cell.label) for cell in model.cells], dtype=np.int64),
            np.array([horizons.index(cell.horizon) for cell in model.cells], dtype=np.int64),
            len(labels),
            len(horizons),
        )

    def combine(self, primitive: FloatArray, *, absolute: bool = False) -> FloatArray:
        r"""``(J, p, n)`` per-unit coefficient sums of ``(R, J, H, n)`` per-unit cell arrays.

        Parameters
        ----------
        primitive : FloatArray
            One value per regimen, cause, horizon and unit.
        absolute : bool, default False
            Whether to sum with :math:`|C|`, as a score scale does.

        Returns
        -------
        FloatArray
            :math:`\sum_c C_{icm}\, x_{r_c j h_c i}` for every cause, coefficient and unit.
        """
        design = self.model.weighted_design
        if absolute:
            design = np.abs(design)
        gathered = primitive[self.regimen, :, self.horizon]  # (C, J, n)
        return np.asarray(np.einsum("cjn,ncm->jmn", gathered, design), dtype=float)

    def coefficients(self, gamma: FloatArray) -> FloatArray:
        r"""``(n, R, J, H)`` per-unit cell coefficients of the coefficient direction ``gamma``.

        Parameters
        ----------
        gamma : FloatArray
            ``(J, p)`` weight of every coefficient score.

        Returns
        -------
        FloatArray
            :math:`\sum_m \gamma_{jm} C_{icm}` filed at each cell's regimen and horizon.
        """
        per_cell = np.einsum("ncm,jm->ncj", self.model.weighted_design, gamma)
        out = np.zeros((per_cell.shape[0], self.n_regimens, gamma.shape[0], self.n_horizons))
        for c, (r, h) in enumerate(zip(self.regimen, self.horizon, strict=True)):
            out[:, r, :, h] += per_cell[:, c]
        return out

    def plug_ins(
        self, state: HazardState, inputs: HazardInputs, horizons: Sequence[int], cause: int
    ) -> FloatArray:
        r"""``(n, C)`` plug-in :math:`\Psi_{c,j}(W_i)` of every cell for cause ``cause``.

        Parameters
        ----------
        state : HazardState
            The hazards.
        inputs : HazardInputs
            The targeting inputs.
        horizons : sequence of int
            The walk's horizons, counted from one.
        cause : int
            The cause index.

        Returns
        -------
        FloatArray
            :math:`\sum_a q_r(a \mid W_i) F_j(t \mid a, W_i)` at each cell :math:`(r, t)`.
        """
        mixtures = {int(r): state.mixture(inputs.q[r], cause) for r in set(self.regimen)}
        return np.column_stack(
            [
                mixtures[int(r)][:, horizons[h] - 1]
                for r, h in zip(self.regimen, self.horizon, strict=True)
            ]
        )

    def project(
        self, state: HazardState, inputs: HazardInputs, horizons: Sequence[int], cause: int
    ) -> tuple[FloatArray, ProjectionFit]:
        """The plug-ins of :meth:`plug_ins` and their projection onto the working model.

        Parameters
        ----------
        state : HazardState
            The hazards.
        inputs : HazardInputs
            The targeting inputs.
        horizons : sequence of int
            The walk's horizons, counted from one.
        cause : int
            The cause index.

        Returns
        -------
        tuple
            The ``(n, C)`` plug-ins and the :func:`cleverly.msm.solve_projection` fit.
        """
        plug = self.plug_ins(state, inputs, horizons, cause)
        model = self.model
        return plug, solve_projection(
            model.design, model.weights, plug, inputs.weights, str(model.link)
        )


def _full_metric(curves: FloatArray, sigma: FloatArray) -> FloatArray:
    """The ``(m, m)`` empirical covariance of the walked targets' curves, floored on its diagonal.

    A working model's coefficient scores are strongly correlated (an intercept, a dose and a
    trend share most of their cells), and a diagonal metric then walks a narrow valley in
    thousands of steps.  The full covariance whitens the scores, so every direction of the
    score decays at about the same rate along the submodel.  The metric is fixed at the
    initial fit, as Rytgaard and van der Laan (2024), Remark 2, fix their diagonal one.  It
    changes the direction of each step and not where the walk may stop: a step is accepted
    only when the metric norm of the score does not rise, the walk stops on the relative
    score, which does not read the metric, and the norm of a positive definite metric is
    zero only at a zero score.
    """
    flat = curves.reshape(-1, curves.shape[-1])
    metric = flat @ flat.T / flat.shape[1]
    index = np.arange(metric.shape[0])
    metric[index, index] = sigma.reshape(-1)
    # A ridge of the floor keeps a singular covariance (a coefficient whose curve is a
    # combination of the others') invertible.
    return np.asarray(metric + SIGMA_FLOOR * np.eye(metric.shape[0]), dtype=float)


def _metric_norm(score: FloatArray, sigma: FloatArray, metric: FloatArray | None) -> float:
    """The walk's score norm: diagonal ``sigma``, or the full ``metric`` when one is set."""
    if metric is None:
        return float(np.sqrt(np.sum(score**2 / sigma)))
    flat = score.reshape(-1)
    return float(np.sqrt(max(float(flat @ np.linalg.solve(metric, flat)), 0.0)))


def _walk_scores(
    state: HazardState,
    inputs: HazardInputs,
    horizons: Sequence[int],
    combination: MSMTargets | None,
) -> tuple[FloatArray, FloatArray]:
    """The scores and scales the walk targets: of the cells, or of the coefficients."""
    if combination is None:
        return scores(state, inputs, horizons)
    terms, _, rows = _score_parts(state, inputs, horizons)
    score = combination.combine(terms * inputs.weights).mean(axis=-1)
    scale = combination.combine(rows, absolute=True).mean(axis=-1)
    return np.asarray(score, dtype=float), np.asarray(scale, dtype=float)


def walk_curve(
    initial: HazardState,
    inputs: HazardInputs,
    horizons: Sequence[int],
    targets: tuple[tuple[str, str | None, int], ...] | tuple[tuple[str, str | None], ...],
    *,
    step_size: float,
    tol: float,
    max_steps: int = 20_000,
    warn: bool = True,
    stopping: str = "relative",
    stop: StopContext | None = None,
    iterate: str = "last",
    combination: MSMTargets | None = None,
) -> tuple[HazardState, HazardFit]:
    """Walk the universal least favorable submodel over every target (part (b)).

    The walk accepts a step only when the :math:`\\Sigma`-norm of the score does not rise.
    Under a declared ``stopping`` rule it stops when the largest score over its threshold,
    over every target, is at most one.  Under ``iterate="best"`` it reports the step
    ``k >= 1`` with the smallest :math:`\\Sigma`-norm, the last of tied steps, which is the
    last accepted step (finding M5 of the bounded-hazard design).  ``converged``, ``failure``
    and the rule ratio are judged at the reported step.

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
    stopping : {"relative", "scaled_var", "sample_size"}
        The stopping rule.
    stop : StopContext or None
        The units and the span a declared rule reads.
    iterate : {"last", "best"}
        Which step the walk reports.
    combination : MSMTargets or None, default None
        A working model whose coefficient scores the walk targets in place of the cell
        scores.  ``targets`` then names ``(term, cause)`` of every coefficient.

    Returns
    -------
    tuple
        The targeted state and its record.
    """
    declared = stopping != "relative"
    if declared and stop is None:
        raise ValueError(STOP_CONTEXT_MISSING)
    bounds = inputs.bounds
    state = initial
    levels, curves = influence(state, inputs, horizons)
    if combination is not None:
        # The metric of a coefficient target is its cell-centred curve, and its initial
        # level the coefficients projected from the initial hazards.
        curves = combination.combine(curves)
        levels = np.stack(
            [
                combination.project(state, inputs, horizons, j)[1].beta
                for j in range(inputs.n_causes)
            ]
        )
    sigma = np.maximum(np.mean(curves**2, axis=-1), SIGMA_FLOOR)
    metric = None if combination is None else _full_metric(curves, sigma)
    score, scale = _walk_scores(state, inputs, horizons, combination)
    first = score.reshape(-1).copy()
    norm = _metric_norm(score, sigma, metric)
    trace = [relative_score(score.reshape(-1), scale.reshape(-1))]
    norms = [norm]
    losses = [walk_loss(state, inputs)]
    thresholds: FloatArray | None = None
    ratio: float | None = None
    if declared:
        assert stop is not None
        thresholds, ratio = _rule_state(state, inputs, horizons, score, stop, stopping, combination)
    best = BestIterate(last_on_ties=True) if iterate == "best" else None

    def met() -> bool:
        # ``not > tol`` rather than ``<= tol``: a NaN relative score stops the default rule,
        # as the shipped ``while trace[-1] > tol`` did.
        return ratio <= 1.0 if ratio is not None else not trace[-1] > tol

    dx = float(step_size)
    steps = 0
    while not met() and steps < max_steps and norm > 0.0:
        if combination is None:
            coefficients = score / sigma / norm
        else:
            assert metric is not None
            gamma = np.linalg.solve(metric, score.reshape(-1)).reshape(score.shape) / norm
            coefficients = combination.coefficients(gamma)
        candidate = state.moved(dx * direction(state, inputs, horizons, coefficients), bounds)
        candidate_score, candidate_scale = _walk_scores(candidate, inputs, horizons, combination)
        candidate_norm = _metric_norm(candidate_score, sigma, metric)
        if candidate_norm > norm:
            dx *= 0.5
            if dx < _MIN_STEP:
                break
            continue
        state, score, scale, norm = candidate, candidate_score, candidate_scale, candidate_norm
        steps += 1
        trace.append(relative_score(score.reshape(-1), scale.reshape(-1)))
        norms.append(norm)
        losses.append(walk_loss(state, inputs))
        if declared:
            assert stop is not None
            thresholds, ratio = _rule_state(
                state, inputs, horizons, score, stop, stopping, combination
            )
        if best is not None:
            best.offer(steps, norm, (state, score, scale, thresholds, ratio))
    reported: int | None = None
    if best is not None:
        reported = steps
        if best.state is not None and best.k is not None:
            reported = best.k
            state, score, scale, thresholds, ratio = best.state
    relative = relative_score(score.reshape(-1), scale.reshape(-1))
    converged = bool(ratio <= 1.0) if ratio is not None else bool(relative <= tol)
    if not converged and warn:
        detail = (
            f"with a relative score of {relative:.3g} > {tol:g}"
            if ratio is None
            else f"without meeting the {stopping} stopping rule (largest score over its "
            f"threshold = {ratio:.3g})"
        )
        _warn(
            f"one-step hazard targeting stopped after {steps} step(s) {detail}. Try a smaller "
            "step_size, or check result.diagnostics for a positivity problem",
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
        walk_loss_trace=tuple(losses),
        failure=None if converged else _classify(state, inputs, steps, max_steps, max(horizons)),
        tol=tol,
        stopping=stopping,
        thresholds=thresholds,
        rule_ratio=ratio,
        best_iterate=reported,
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


#: The warning of a bounded hazard fit with a cell that its bound excludes.
HAZARD_BOUND_EXCLUDES_DATA = (
    "the declared node bound at node(s) {nodes} excludes the data: {count} cell(s) of a "
    "treatment level and a cause hold no event under a positive lower bound, or an event in "
    "every row at risk. The hazard of such a cell is pinned at the clip of its bound, and "
    "HazardFit.bound_excludes_data lists the cells. Widen node_bounds at those nodes."
)


def clipped_shares(hazards: FloatArray, inputs: HazardInputs) -> tuple[tuple[int, int, float], ...]:
    """The share of initial hazards that the bound clips, at every bounded node and cause.

    At a bounded node the clipped logit keeps the hazard of cause :math:`j` in
    :math:`[l_j + (1 - \\alpha) s_j, u_j - (1 - \\alpha) s_j]`.  The share counts the
    ``(unit, level)`` initial hazards outside that interval.  A cell held at zero is not
    clipped and is left out of both counts.

    Parameters
    ----------
    hazards : FloatArray
        ``(n, T, L, J)`` initial hazards.
    inputs : HazardInputs
        The targeting inputs of a bounded fit.

    Returns
    -------
    tuple of tuple
        ``(node, cause index, share)`` with the node counted from one, in node then cause
        order.  Empty when no node is bounded.
    """
    bounds = inputs.node_bounds
    if bounds is None or not bounds.active:
        return ()
    out: list[tuple[int, int, float]] = []
    alpha = inputs.alpha
    for k in np.flatnonzero(bounds.bounded):
        for j in range(inputs.n_causes):
            lower = bounds.lower[k, j] + (1.0 - alpha) * bounds.scale[k, j]
            upper = bounds.lower[k, j] + alpha * bounds.scale[k, j]
            values = np.asarray(hazards[:, k, :, j], dtype=float)
            held = inputs.empty[k, :, j]
            # A slot state reads the hold of each slot's cell key.
            counted = ~(
                np.broadcast_to(held[None, :], values.shape)
                if inputs.cells is None
                else held[inputs.cells]
            )
            outside = counted & ((values < lower) | (values > upper))
            total = int(np.count_nonzero(counted))
            share = float(np.count_nonzero(outside)) / total if total else 0.0
            out.append((int(k) + 1, j, share))
    return tuple(out)


def _bounded_records(
    reports: dict[int, tuple[HazardState, HazardFit]],
    hazards: FloatArray,
    inputs: HazardInputs,
    data: LongitudinalData,
) -> dict[int, tuple[HazardState, HazardFit]]:
    """Stamp the clipped shares and the excluded cells on every record, and warn once."""
    shares = clipped_shares(hazards, inputs)
    excluded: tuple[tuple[int, int, int], ...] = ()
    if inputs.excluded is not None:
        excluded = tuple(
            (int(k) + 1, int(level), int(j)) for k, level, j in np.argwhere(inputs.excluded)
        )
    if excluded:
        grid = data.time_grid
        nodes = sorted({cell[0] for cell in excluded})
        named = ", ".join(f"{grid[k - 1] if grid is not None else k:g}" for k in nodes)
        _warn(
            HAZARD_BOUND_EXCLUDES_DATA.format(nodes=named, count=len(excluded)),
            ConvergenceWarning,
            stacklevel=3,
        )
    stamped: dict[int, HazardFit] = {}
    out: dict[int, tuple[HazardState, HazardFit]] = {}
    for t, (state, record) in reports.items():
        if id(record) not in stamped:
            stamped[id(record)] = replace(
                record, bound_excludes_data=excluded, share_clipped=shares
            )
        out[t] = (state, stamped[id(record)])
    return out


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
    node_bounds: HazardBounds | None = None,
    stopping: str = "relative",
    stop: StopContext | None = None,
    step_bound: float | None = None,
    step_search: bool = True,
    iterate: str = "last",
) -> dict[tuple[str, str | None, int], RegimenFit]:
    """Target the hazards and report one fit per regimen, cause and horizon.

    At a bounded node the initial hazards are clipped into the bound, and every record holds
    the clipped share and the cells the bound excludes.  A fit with such a cell warns once.

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
    node_bounds : HazardBounds or None
        The bounded nodes, from :func:`hazard_bounds`.
    stopping : {"relative", "scaled_var", "sample_size"}
        The stopping rule of both algorithms.
    stop : StopContext or None
        The units and the span a declared rule reads.
    step_bound : float or None
        The longest per-horizon step in :math:`\\epsilon`.  The caller refuses it beside
        the one-step walk.
    step_search : bool
        Whether a bounded step searches its length.
    iterate : {"last", "best"}
        Which iterate each algorithm reports.

    Returns
    -------
    dict
        One :class:`~cleverly.longitudinal.sequential.RegimenFit` per
        ``(regimen, cause, horizon)``, regimen-outer, then cause, then horizon.
    """
    inputs = hazard_inputs(data, plans, cumulative, alpha=alpha, node_bounds=node_bounds)
    start = HazardState.from_hazards(
        initial.hazards, inputs.bounds, inputs.empty, inputs.full, node_bounds
    )
    horizons = tuple(horizons)
    reports = _target(
        data,
        [plan.label for plan in plans],
        start,
        inputs,
        horizons,
        algorithm=algorithm,
        max_iter=max_iter,
        tol=tol,
        step_size=step_size,
        stopping=stopping,
        stop=stop,
        step_bound=step_bound,
        step_search=step_search,
        iterate=iterate,
    )
    if inputs.node_bounds is not None:
        reports = _bounded_records(reports, initial.hazards, inputs, data)
    return _cell_fits(data, plans, inputs, reports, horizons, observed_cumulative)


def _target(
    data: LongitudinalData,
    labels: Sequence[str],
    start: HazardState,
    inputs: HazardInputs,
    horizons: tuple[int, ...],
    *,
    algorithm: HazardAlgorithm,
    max_iter: int,
    tol: float,
    step_size: float,
    stopping: str,
    stop: StopContext | None,
    step_bound: float | None,
    step_search: bool,
    iterate: str,
) -> dict[int, tuple[HazardState, HazardFit]]:
    """One walk over every horizon, or one per-horizon fit per horizon, by horizon."""
    causes: tuple[str | None, ...] = data.cause_labels or (None,)
    reports: dict[int, tuple[HazardState, HazardFit]] = {}
    if algorithm == "one_step":
        targets = tuple((label, cause, t) for label in labels for cause in causes for t in horizons)
        shared = walk_curve(
            start,
            inputs,
            horizons,
            targets,
            step_size=step_size,
            tol=tol,
            stopping=stopping,
            stop=stop,
            iterate=iterate,
        )
        return dict.fromkeys(horizons, shared)
    for t in horizons:
        targets = tuple((label, cause, t) for label in labels for cause in causes)
        reports[t] = solve_per_horizon(
            start,
            inputs,
            t,
            targets,
            max_iter=max_iter,
            tol=tol,
            stopping=stopping,
            stop=stop,
            step_bound=step_bound,
            step_search=step_search,
            iterate=iterate,
        )
    return reports


def target_mtp(
    data: LongitudinalData,
    plan: Plan,
    initial: InitialHazards,
    inputs: HazardInputs,
    observed_cumulative: tuple[FloatArray, FloatArray],
    *,
    node_ratio: FloatArray | None,
    horizons: Sequence[int],
    algorithm: HazardAlgorithm,
    max_iter: int,
    tol: float,
    step_size: float,
    stopping: str = "relative",
    stop: StopContext | None = None,
    step_bound: float | None = None,
    step_search: bool = True,
    iterate: str = "last",
) -> dict[tuple[str, str | None, int], RegimenFit]:
    r"""Target one modified treatment policy regimen by its own weighted walk.

    The clever covariate is :math:`\kappa_{j,t}(k, a, W)` at every slot and the loss of node
    :math:`k` is weighted by :math:`\omega_k = w\, r(A, W) / G_k(A, W)` at the observed
    treatment (:func:`mtp_inputs`), so a walk that solves the weighted scores solves the
    efficient influence function of
    :math:`\psi_{d,j}(t) = E \sum_b p_b F_j(t \mid d_b(A, W), W)` (Díaz Muñoz and van der
    Laan 2012, Result 1, with the chain rule), and the plug-in mixes the branch slots.  Every
    horizon and cause of the regimen share the walk, so its curve is monotone.

    Parameters
    ----------
    data : LongitudinalData
        A survival panel of a held design.
    plan : Plan
        The policy's plan, with its ratio numerator attached.
    initial : InitialHazards
        The initial hazards from :func:`fit_hazards`.
    inputs : HazardInputs
        The policy's weighted inputs from :func:`mtp_inputs`.
    observed_cumulative : tuple of FloatArray
        The unbounded and bounded ``(n, T)`` cumulative mechanism of the plan, for the
        diagnostics.
    node_ratio : FloatArray or None
        The plan's raw per-node ratio, for the diagnostics.
    horizons : sequence of int
        The reported horizons, counted from one.
    algorithm : {"per_horizon", "one_step"}
        The targeting algorithm.
    max_iter : int
        Outer iterations per horizon of a per-horizon fit.
    tol : float
        Relative-score tolerance.
    step_size : float
        The first step of a one-step walk.
    stopping : {"relative", "scaled_var", "sample_size"}
        The stopping rule.
    stop : StopContext or None
        The units and the span a declared rule reads.
    step_bound : float or None
        The longest per-horizon step in :math:`\epsilon`.
    step_search : bool
        Whether a bounded step searches its length.
    iterate : {"last", "best"}
        Which iterate each algorithm reports.

    Returns
    -------
    dict
        One :class:`~cleverly.longitudinal.sequential.RegimenFit` per ``(regimen, cause,
        horizon)``, cause-outer, then horizon.
    """
    hazards = initial.hazards if inputs.cells is None else initial.slots[plan.label]
    empty, full = inputs.held_cells()
    start = HazardState.from_hazards(hazards, inputs.bounds, empty, full, inputs.node_bounds)
    horizons = tuple(horizons)
    reports = _target(
        data,
        [plan.label],
        start,
        inputs,
        horizons,
        algorithm=algorithm,
        max_iter=max_iter,
        tol=tol,
        step_size=step_size,
        stopping=stopping,
        stop=stop,
        step_bound=step_bound,
        step_search=step_search,
        iterate=iterate,
    )
    if inputs.node_bounds is not None:
        reports = _bounded_records(reports, hazards, inputs, data)
    return _cell_fits(
        data,
        [plan],
        inputs,
        reports,
        horizons,
        observed_cumulative,
        numerator=plan.cumulative_numerator(data),
        node_ratio=node_ratio,
    )


def _cell_fits(
    data: LongitudinalData,
    plans: Sequence[Plan],
    inputs: HazardInputs,
    reports: dict[int, tuple[HazardState, HazardFit]],
    horizons: tuple[int, ...],
    observed_cumulative: tuple[FloatArray, FloatArray],
    *,
    numerator: FloatArray | None = None,
    node_ratio: FloatArray | None = None,
) -> dict[tuple[str, str | None, int], RegimenFit]:
    """One :class:`RegimenFit` per regimen, cause and horizon, at each horizon's report.

    A modified treatment policy's fit also files the plan's cumulative ratio numerator and
    its raw per-node ratio, as the iterated-mean fit of the plan does.
    """
    causes: tuple[str | None, ...] = data.cause_labels or (None,)
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
                    cumulative_numerator=numerator,
                    node_ratio=node_ratio,
                )
    return out


def target_hazard_msm(
    data: LongitudinalData,
    plans: Sequence[Plan],
    initial: InitialHazards,
    cumulative: Sequence[FloatArray],
    observed_cumulative: tuple[FloatArray, FloatArray],
    model: RegimenMSM,
    *,
    horizons: Sequence[int],
    alpha: float,
    tol: float,
    step_size: float,
    stopping: str = "relative",
    stop: StopContext | None = None,
    iterate: str = "last",
    max_steps: int = 20_000,
) -> tuple[dict[tuple[str, str | None, int], RegimenFit], tuple[MSMRegimenFit, ...]]:
    r"""Target the coefficient scores of a working model, and project each cause's curve.

    One walk of :func:`walk_curve` moves the hazards along the coefficient scores
    :math:`P_n\, w \sum_c h\varphi\, D^{(1)}_{c,j}` of every cause together
    (:class:`MSMTargets`).  At the targeted hazards each cause's plug-ins
    :math:`\Psi^*_{c,j}(W_i)` are projected by :func:`cleverly.msm.solve_projection`, and
    the coefficients' influence curve is

    .. math::

        D^*_\beta = M^{-1} \sum_c h\varphi\, w\,
            \big[D^{(1)}_{c,j} + \Psi^*_{c,j}(W) - m(c, V; \hat\beta)\big],

    the curve of :func:`~cleverly.longitudinal.msm.fit_regimens_msm` with the cell
    residual replaced by the hazard part of the cell's curve (Rosenblum and van der Laan
    2010, (17)).  A hazard fit has no outcome scaler, so the plug-ins enter raw.

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
        The unbounded and bounded ``(n, T)`` :math:`g G_k` at each unit's observed level.
    model : RegimenMSM
        The working model, evaluated at every ``(regimen, horizon)`` cell.
    horizons : sequence of int
        The reported horizons, counted from one.
    alpha : float
        The probability bound.
    tol : float
        Relative-score tolerance.
    step_size : float
        The first step of the walk.
    stopping : {"relative", "scaled_var", "sample_size"}
        The stopping rule.
    stop : StopContext or None
        The units and the span a declared rule reads.
    iterate : {"last", "best"}
        Which step the walk reports.
    max_steps : int
        The step cap of the walk.

    Returns
    -------
    tuple
        One :class:`~cleverly.longitudinal.sequential.RegimenFit` per regimen, cause and
        horizon, for the diagnostics, and one
        :class:`~cleverly.longitudinal.msm.MSMRegimenFit` per cause.
    """
    # No node is bounded: ``node_bounds=`` beside ``msm=`` is refused at construction (R-i3).
    inputs = hazard_inputs(data, plans, cumulative, alpha=alpha)
    start = HazardState.from_hazards(initial.hazards, inputs.bounds, inputs.empty, inputs.full)
    causes: tuple[str | None, ...] = data.cause_labels or (None,)
    horizons = tuple(horizons)
    combination = MSMTargets.of(model, [plan.label for plan in plans], horizons)
    targets = tuple((term, cause) for cause in causes for term in model.terms)
    shared = walk_curve(
        start,
        inputs,
        horizons,
        targets,
        step_size=step_size,
        tol=tol,
        max_steps=max_steps,
        stopping=stopping,
        stop=stop,
        iterate=iterate,
        combination=combination,
    )
    reports = dict.fromkeys(horizons, shared)
    state, record = shared
    fits = _cell_fits(data, plans, inputs, reports, horizons, observed_cumulative)
    terms, _ = _score_terms(state, inputs, horizons)
    weights = np.asarray(inputs.weights, dtype=float)
    out: list[MSMRegimenFit] = []
    for j, cause in enumerate(causes):
        plug, projection = combination.project(state, inputs, horizons, j)
        residual = terms[combination.regimen, j, combination.horizon].T  # (n, C)
        bracket = residual + plug - model.fitted(projection.beta)
        contribution = np.einsum("ikp,ik->ip", model.weighted_design, weights[:, None] * bracket)
        curves = np.linalg.solve(projection.jacobian, contribution.T).T
        out.append(
            MSMRegimenFit(
                model=model,
                cause=cause,
                beta=projection.beta,
                influence_curves=np.asarray(curves, dtype=float),
                projection=projection,
                fits=tuple(fits[(cell.label, cause, cell.horizon)] for cell in model.cells),
                nodes=(),
                fluctuation_design=model.fluctuation_design,
                hazard=record,
            )
        )
    return fits, tuple(out)


# ------------------------------------------------------------------------ refusals


def refuse_hazard_settings(
    *,
    survival_model: str,
    targeting: str,
    step_size: float,
    isotonic: bool,
    msm: bool,
    step_bound: float | None = None,
    ratio_trim: float | None = None,
) -> None:
    """Refuse a hazard or isotonic setting that needs no data (H4, H5, H8, H13, H14, R-d1).

    ``pseudo_learner=`` (H6) reads the design, so :func:`refuse_hazard_design` refuses it.

    ``iterate=`` and ``stopping=`` apply to both hazard algorithms, and ``step_bound=`` to
    ``targeting="iterative"``.  The one-step walk takes its steps in ``step_size``, so a step
    bound beside it is refused as a redirection.

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
    step_bound : float or None, default=None
        The ``step_bound=`` value. The caller has validated it.
    ratio_trim : float or None, default=None
        The ``ratio_trim=`` value. The caller has validated it.

    Raises
    ------
    ValueError
        On a malformed setting.
    CapabilityError
        On a combination that is not built.
    MethodConfigurationError
        On ``step_bound=`` beside the one-step hazard walk, or ``msm=`` beside the
        per-horizon hazard algorithm.
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
    if msm and isotonic:
        raise CapabilityError.from_refusal(HAZARD_MSM_ISOTONIC_REFUSAL)
    if msm and survival_model == "hazard" and targeting != "one_step":
        raise MethodConfigurationError.from_refusal(HAZARD_MSM_PER_HORIZON_REFUSAL)
    if survival_model == "hazard" and targeting == "one_step" and step_bound is not None:
        raise MethodConfigurationError.from_refusal(HAZARD_STEP_BOUND_ONE_STEP_REFUSAL)
    if survival_model == "hazard" and ratio_trim is not None:
        raise CapabilityError.from_refusal(HAZARD_RATIO_TRIM_REFUSAL)


def refuse_hazard_design(
    data: LongitudinalData,
    regimens: Sequence[RegimenSpec],
    *,
    survival_model: str,
    isotonic: bool,
    msm: bool = False,
    targeting: str = "one_step",
    stopping: str = "relative",
    pseudo_learner: bool = False,
) -> None:
    """Refuse a design the hazard model or the projection does not support.

    Called after the data and the regimens are read, before any learner: H1, H2, H6, H16,
    H19 and, on a held design with covariates after baseline, H28 to H30.  A modified
    treatment policy at the baseline decision is fitted on a categorical, continuous or
    mixed node (X70 part (c)), except with a tuned cap (H16) or beside ``msm=`` (H19).  A
    held design with covariates after baseline is fitted by the backward pass of
    :mod:`cleverly.longitudinal.hazard_backward` (X70 part (b1)).

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
    msm : bool, default False
        Whether ``msm=`` was passed.
    targeting : str, default "one_step"
        The ``targeting=`` value.
    stopping : str, default "relative"
        The ``stopping=`` value.
    pseudo_learner : bool, default False
        Whether ``pseudo_learner=`` was passed.

    Raises
    ------
    CapabilityError
        On any refused design.
    MethodConfigurationError
        On the per-horizon algorithm (H28) or a declared stopping rule (H30) beside covariates
        after baseline.
    ValueError
        On ``pseudo_learner=`` beside a design without covariates after baseline (H6).
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
    from .hazard_backward import (
        hazard_iterative_refusal,
        hazard_msm_refusal,
        hazard_stopping_refusal,
        later_covariates,
    )

    blocks = [name for block in data.time_varying_names for name in block]
    multinode = not data.treatment_held and data.n_decisions > 1
    if multinode and blocks:
        raise CapabilityError(
            "survival_model='hazard' on a design with a treatment at several nodes models the "
            "event and censoring hazards given the treatment history and the baseline "
            "covariates. This fit declares covariates measured after baseline "
            f"({', '.join(blocks)}), and hazard targeting with such covariates on a design with "
            "a treatment at several nodes is not built.",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            roadmap=HAZARD_ROADMAP,
            alternative=(
                "LTMLE with survival_model='iterated_mean' adjusts for them, and "
                "isotonic=True makes its curve monotone. If one baseline treatment is held over "
                "every node, declare treatment='A'."
            ),
        )
    if multinode:
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
    later = later_covariates(data)
    if later:
        if targeting != "one_step":
            raise MethodConfigurationError.from_refusal(hazard_iterative_refusal(later))
        if msm:
            raise CapabilityError.from_refusal(hazard_msm_refusal())
        if stopping != "relative":
            raise MethodConfigurationError.from_refusal(hazard_stopping_refusal(stopping))
    elif pseudo_learner:
        raise ValueError(
            "pseudo_learner= is not used with survival_model='hazard' on a design with "
            "baseline covariates only. Such a fit regresses the event indicator at each node "
            "and fits no pseudo-outcome. Pass the hazard learner as outcome_learner="
        )
    for regimen in regimens:
        label = getattr(regimen, "label", "")
        if declares_tuned([regimen]):
            raise CapabilityError.from_refusal(hazard_tuned_cap_refusal(label))
        if msm and bool(getattr(regimen, "has_mtp", False)):
            raise CapabilityError.from_refusal(hazard_msm_mtp_refusal(label))
