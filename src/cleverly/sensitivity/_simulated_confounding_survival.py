"""The simulated common-cause replay of a held survival or competing-risk fit (X91 part C).

The shipped latent :math:`U_i` moves the held baseline treatment exactly as on an
end-of-study fit.  The outcome strength :math:`\\gamma` shifts the per-node event log-odds of
each unit by :math:`\\gamma U_i`.  The follow-up is then replayed node by node, censoring
first and the event second, as the panel orders them.

Each replayed indicator couples to the observed one through a uniform drawn conditional
on it: with fitted probability :math:`p` and observed value :math:`b`, the uniform is
:math:`pR` when :math:`b = 1` and :math:`p + (1 - p)R` when :math:`b = 0`, and the replayed
indicator is :math:`\\mathbf 1\\{V < p'\\}`.  At zero strength every observed indicator is
kept as it is, so the zero cell is the observed panel.  When :math:`p` is the true
conditional, :math:`V` is uniform and independent of :math:`U`, so the replayed indicator has
probability :math:`p'` exactly.

The event model is read at the observed treatment, which is the shipped point semantics: a
flipped unit keeps the outcome process it had, and the latent reaches it only through
:math:`\\gamma`.  The censoring model is read at the perturbed treatment, which keeps
censoring at random given the replayed past when the censoring model is right.  The
observed censoring was generated under the observed treatment, which depends on the latent
given the perturbed one.

The replay fits its own pooled logistic models once per surface, on the observed panel and
with the observation weights.  A node at which the fitted indicator takes one value only is
held at the clip bound rather than fitted, so a node with no event does not separate the
regression.  The design note is ``X91-C-design.md`` in the planning repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

#: The tag of the coupling stream, apart from the latent tag 3, the refutation tags one to
#: four, ``_COLLISION_TAG`` and ``_SIMULATION_SEED_TAG``.  "CPL1" in ASCII.
COUPLING_SEED_TAG = 0x43504C31

#: The fitted probabilities stay in ``[CLIP, 1 - CLIP]``, so every shifted log-odds is finite.
CLIP = 1e-6

#: The L2 penalty of every replay model, large enough to leave a fit in effect
#: unpenalized and small enough to keep it finite under separation.
PENALTY_C = 1e4

#: The iteration limit of every replay model.
MAX_ITER = 1000

#: The solver tolerance of every replay model, tight enough that a coefficient read as a
#: calibration strength is the penalized maximum to about six decimals.
TOLERANCE = 1e-8

#: The number of uniform arrays drawn from the coupling stream: censoring, event, cause, and
#: the other-cause event of a per-cause law.
_N_STREAMS = 4


def coupling_seed(root_seed: int) -> int:
    """Derive the coupling-uniform seed from a tagged child of the root seed.

    Parameters
    ----------
    root_seed : int
        Seed resolved for the complete operation.

    Returns
    -------
    int
        Seed of the coupling uniforms.
    """
    sequence = np.random.SeedSequence([root_seed, COUPLING_SEED_TAG])
    return int(sequence.generate_state(1)[0])


def coupling_uniforms(n: int, n_times: int, seed: int) -> np.ndarray[Any, Any]:
    """``(4, n, T)`` uniforms shared by every cell of one surface.

    Parameters
    ----------
    n : int
        Units.
    n_times : int
        Nodes.
    seed : int
        The coupling seed.

    Returns
    -------
    ndarray
        Censoring, event, cause, and other-cause uniforms, in that order.
    """
    return np.random.default_rng(seed).uniform(size=(_N_STREAMS, n, n_times))


def conditional_uniform(
    p: np.ndarray[Any, Any],
    observed_value: np.ndarray[Any, Any],
    raw: np.ndarray[Any, Any],
    observed: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """The coupling uniform: conditioned on the observed indicator where there is one.

    Parameters
    ----------
    p : ndarray
        The fitted probability of the indicator.
    observed_value : ndarray
        The observed indicator, read only where ``observed``.
    raw : ndarray
        Plain uniforms.
    observed : ndarray
        Where the indicator was observed.

    Returns
    -------
    ndarray
        ``pR`` where the observed value is one, ``p + (1 - p)R`` where it is zero, and
        ``R`` where nothing was observed.
    """
    one = observed & observed_value.astype(bool)
    zero = observed & ~observed_value.astype(bool)
    return np.where(one, p * raw, np.where(zero, p + (1.0 - p) * raw, raw))


def shifted(p: np.ndarray[Any, Any], shift: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """``expit(logit(p) + shift)``."""
    logit = np.log(p) - np.log1p(-p)
    return 1.0 / (1.0 + np.exp(-(logit + shift)))


def draw_cause(
    probabilities: np.ndarray[Any, Any], raw: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    """The cause index of each row by the inverse CDF of its shares."""
    cumulative = np.cumsum(probabilities, axis=1)
    index = (cumulative < raw[:, None] * cumulative[:, -1:]).sum(axis=1)
    return np.minimum(index, probabilities.shape[1] - 1)


@dataclass(frozen=True)
class PooledModel:
    """A pooled logistic or multinomial model over person-nodes.

    Parameters
    ----------
    name : str
        What the model fits.
    model : object or None
        The fitted scikit-learn model, or ``None`` when no node carries both values.
    held : dict
        Per degenerate node, the probability vector it is held at.
    n_classes : int
        Classes of the target.
    n_times : int
        Nodes.
    converged : bool
        Whether the fit stopped before its iteration limit.
    design : ndarray or None
        The fitted rows' design, kept for the calibration.
    target : ndarray or None
        The fitted rows' target.
    row_weights : ndarray or None
        The fitted rows' weights.
    """

    name: str
    model: Any
    held: dict[int, np.ndarray[Any, Any]]
    n_classes: int
    n_times: int
    converged: bool
    design: np.ndarray[Any, Any] | None = None
    target: np.ndarray[Any, Any] | None = None
    row_weights: np.ndarray[Any, Any] | None = None

    def probabilities(
        self, node: int, baseline: np.ndarray[Any, Any], treatment: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        """``(n, classes)`` fitted probabilities at node ``node`` (from zero)."""
        n = baseline.shape[0]
        if node in self.held:
            return np.broadcast_to(self.held[node], (n, self.n_classes)).copy()
        if self.model is None:
            return np.broadcast_to(self._nearest_held(node), (n, self.n_classes)).copy()
        design = person_node_design(baseline, treatment, np.full(n, node), self.n_times)
        fitted = np.zeros((n, self.n_classes))
        fitted[:, np.asarray(self.model.classes_, dtype=int)] = self.model.predict_proba(design)
        return fitted

    def _nearest_held(self, node: int) -> np.ndarray[Any, Any]:
        """The held vector of the nearest observed node, for a node nobody was observed at.

        A model with no fitted node has every observed node held at one class, so a node
        with no observed rows reads its nearest neighbour.  A model with no observed node
        at all holds class zero (no event, no censoring) at the clip.
        """
        if self.held:
            nearest = min(self.held, key=lambda held: (abs(held - node), held))
            return self.held[nearest]
        vector = np.zeros(self.n_classes)
        vector[0] = 1.0
        return np.clip(vector, CLIP, 1.0 - CLIP) if self.n_classes == 2 else vector

    def probability(
        self, node: int, baseline: np.ndarray[Any, Any], treatment: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        """The clipped probability of class one of a binary model."""
        return np.clip(self.probabilities(node, baseline, treatment)[:, 1], CLIP, 1.0 - CLIP)

    def describe(self) -> str:
        """One line naming the model, its settings, and whether it converged."""
        held = sorted(node + 1 for node in self.held)
        return (
            f"{self.name}: pooled {'multinomial' if self.n_classes > 2 else 'logistic'} "
            f"regression on the standardized baseline block, the node-1 treatment and node "
            f"indicators, L2 penalty C={PENALTY_C:g}, lbfgs, max_iter={MAX_ITER}, "
            f"tol={TOLERANCE:g}, probabilities clipped to [{CLIP:g}, 1 - {CLIP:g}]"
            + (f"; nodes {held} held at one value" if held else "")
            + ("" if self.converged else "; did not converge")
        )


def person_node_design(
    baseline: np.ndarray[Any, Any],
    treatment: np.ndarray[Any, Any],
    nodes: np.ndarray[Any, Any],
    n_times: int,
) -> np.ndarray[Any, Any]:
    """Rows ``[W, A, 1{node = 1}, ..., 1{node = T}]`` of a person-node table."""
    indicators = np.zeros((nodes.shape[0], n_times))
    indicators[np.arange(nodes.shape[0]), nodes] = 1.0
    return np.column_stack((baseline, treatment, indicators))


def fit_pooled(
    name: str,
    baseline: np.ndarray[Any, Any],
    treatment: np.ndarray[Any, Any],
    rows: np.ndarray[Any, Any],
    target: np.ndarray[Any, Any],
    weights: np.ndarray[Any, Any],
    n_classes: int,
) -> PooledModel:
    """Fit one pooled model over the person-nodes ``rows`` marks.

    Parameters
    ----------
    name : str
        What the model fits.
    baseline : ndarray
        ``(n, p)`` standardized baseline block.
    treatment : ndarray
        ``(n,)`` node-1 treatment.
    rows : ndarray
        ``(n, T)`` person-nodes in the fit.
    target : ndarray
        ``(n, T)`` integer class of each person-node, read where ``rows``.
    weights : ndarray
        ``(n,)`` observation weights.
    n_classes : int
        Classes of the target.

    Returns
    -------
    PooledModel
        The fit.  A node whose rows carry one class is held at it (clipped for a binary
        target), and a node with no rows is predicted by the pooled fit.
    """
    from sklearn.linear_model import LogisticRegression

    n_times = rows.shape[1]
    held: dict[int, np.ndarray[Any, Any]] = {}
    fitted_nodes = []
    for node in range(n_times):
        here = rows[:, node] & (weights > 0.0)
        if not np.any(here):
            continue
        seen = np.unique(target[here, node])
        if seen.size == 1:
            vector = np.zeros(n_classes)
            vector[int(seen[0])] = 1.0
            if n_classes == 2:
                vector = np.clip(vector, CLIP, 1.0 - CLIP)
            held[node] = vector
        else:
            fitted_nodes.append(node)
    keep = rows & (weights[:, None] > 0.0)
    keep[:, [node for node in range(n_times) if node not in fitted_nodes]] = False
    unit, node_index = np.nonzero(keep)
    classes = np.unique(target[unit, node_index]) if unit.size else np.array([])
    if classes.size < 2:
        return PooledModel(name, None, held, n_classes, n_times, True)
    design = person_node_design(baseline[unit], treatment[unit], node_index, n_times)
    y = target[unit, node_index].astype(int)
    row_weights = weights[unit]
    model = LogisticRegression(C=PENALTY_C, max_iter=MAX_ITER, tol=TOLERANCE)
    model.fit(design, y, sample_weight=row_weights)
    converged = int(np.max(model.n_iter_)) < MAX_ITER
    return PooledModel(name, model, held, n_classes, n_times, converged, design, y, row_weights)


@dataclass(frozen=True)
class ReplayModels:
    """The fitted conditionals the replay reads.

    Parameters
    ----------
    baseline : ndarray
        ``(n, p)`` weighted-standardized baseline block.
    event : PooledModel
        The all-cause event, or with a named cause the named cause among those free of
        every other cause.
    censoring : PooledModel or None
        Censoring at a node, or ``None`` when no unit was censored.
    shares : PooledModel or None
        The cause given an event, on a competing panel.
    other : PooledModel or None
        The event of any cause other than the named one, with a named cause.
    cause_index : int or None
        The named cause, or ``None`` for the all-cause law.
    """

    baseline: np.ndarray[Any, Any]
    event: PooledModel
    censoring: PooledModel | None
    shares: PooledModel | None
    other: PooledModel | None
    cause_index: int | None

    def describe(self) -> tuple[str, ...]:
        """One line per fitted model."""
        models = (self.event, self.censoring, self.shares, self.other)
        return tuple(model.describe() for model in models if model is not None)


@dataclass(frozen=True)
class Observed:
    """The observed follow-up, person-node by person-node.

    Parameters
    ----------
    present : ndarray
        ``(n, T)`` in the study before the censoring node: a censoring indicator exists.
    censored : ndarray
        ``(n, T)`` censored at the node.
    at_risk : ndarray
        ``(n, T)`` at risk for the event: an event indicator exists.
    event : ndarray
        ``(n, T)`` had an event (any cause) at the node.
    cause : ndarray
        ``(n, T)`` the cause index of that event, zero elsewhere.
    treatment : ndarray
        ``(n,)`` observed node-1 treatment.
    """

    present: np.ndarray[Any, Any]
    censored: np.ndarray[Any, Any]
    at_risk: np.ndarray[Any, Any]
    event: np.ndarray[Any, Any]
    cause: np.ndarray[Any, Any]
    treatment: np.ndarray[Any, Any]


def observed_followup(data: Any) -> Observed:
    """Read the observed panel as person-node indicators.

    Parameters
    ----------
    data : LongitudinalData
        A held survival panel.

    Returns
    -------
    Observed
        The indicators.
    """
    from ..longitudinal.hazard import event_indicators

    at_risk, events = event_indicators(data)
    n, n_times = data.n, data.n_times
    present = np.ones((n, n_times), dtype=bool)
    for node in range(1, n_times):
        present[:, node] = data.uncensored_through(node) & data.event_free_through(node)
    censored = present & ~np.asarray(data.uncensored, dtype=bool)
    fired = events.sum(axis=2) > 0.0
    cause = np.where(fired, events.argmax(axis=2), 0)
    return Observed(
        present=present,
        censored=censored,
        at_risk=at_risk,
        event=fired,
        cause=cause,
        treatment=np.asarray(data.node_codes(1), dtype=float),
    )


def standardized_baseline(data: Any) -> np.ndarray[Any, Any]:
    """The baseline block standardized at the unit level under the observation weights."""
    from sklearn.preprocessing import StandardScaler

    from .simulated_confounding import _weights_are_constant

    baseline = np.asarray(data.baseline, dtype=float)
    weights = np.asarray(data.weights, dtype=float)
    if _weights_are_constant(weights):
        return np.asarray(StandardScaler().fit_transform(baseline))
    return np.asarray(StandardScaler().fit(baseline, sample_weight=weights).transform(baseline))


def fit_replay_models(data: Any, observed: Observed, cause: str | None) -> ReplayModels:
    """Fit the replay's conditionals once, on the observed panel.

    Parameters
    ----------
    data : LongitudinalData
        A held survival panel.
    observed : Observed
        Its observed follow-up.
    cause : str or None
        The named cause of a per-cause law, or ``None``.

    Returns
    -------
    ReplayModels
        The fits.
    """
    baseline = standardized_baseline(data)
    weights = np.asarray(data.weights, dtype=float)
    a = observed.treatment
    n_causes = len(data.cause_labels) if data.cause_event is not None else 1
    cause_index = None if cause is None else list(data.cause_labels).index(cause)
    censoring = None
    if np.any(observed.censored):
        censoring = fit_pooled(
            "censoring at a node",
            baseline,
            a,
            observed.present,
            observed.censored.astype(int),
            weights,
            2,
        )
    shares = None
    if n_causes > 1:
        shares = fit_pooled(
            "cause given an event",
            baseline,
            a,
            observed.event,
            observed.cause,
            weights,
            n_causes,
        )
    if cause_index is None:
        event = fit_pooled(
            "event of any cause among those at risk",
            baseline,
            a,
            observed.at_risk,
            observed.event.astype(int),
            weights,
            2,
        )
        return ReplayModels(baseline, event, censoring, shares, None, None)
    other_event = observed.event & (observed.cause != cause_index)
    named = observed.event & (observed.cause == cause_index)
    other = fit_pooled(
        f"event of a cause other than {cause!r} among those at risk",
        baseline,
        a,
        observed.at_risk,
        other_event.astype(int),
        weights,
        2,
    )
    event = fit_pooled(
        f"event of {cause!r} among those at risk and free of every other cause",
        baseline,
        a,
        observed.at_risk & ~other_event,
        named.astype(int),
        weights,
        2,
    )
    return ReplayModels(baseline, event, censoring, shares, other, cause_index)


@dataclass(frozen=True)
class Replay:
    """One cell's replayed follow-up and the shares of its rows the fitted models drew.

    Parameters
    ----------
    censoring : ndarray or None
        ``(n, T)`` raw censoring columns, or ``None`` on a panel without censoring nodes.
    event : ndarray
        ``(n, T, J)`` raw event columns.
    censoring_redrawn_share : float or None
        Weighted share of replayed person-nodes whose observed censoring was redrawn at a
        changed treatment.
    extended_share : float or None
        Weighted share of replayed person-nodes beyond the unit's observed follow-up.
    new_event_share : float or None
        Weighted share of replayed events at a node where the unit had no observed event.
    """

    censoring: np.ndarray[Any, Any] | None
    event: np.ndarray[Any, Any]
    censoring_redrawn_share: float | None
    extended_share: float | None
    new_event_share: float | None


def _share(
    numerator: np.ndarray[Any, Any], denominator: np.ndarray[Any, Any], weights: Any
) -> float | None:
    total = float(np.sum(denominator * weights[:, None]))
    if total <= 0.0:
        return None
    return float(np.sum(numerator * weights[:, None]) / total)


def replay_followup(
    data: Any,
    observed: Observed,
    models: ReplayModels,
    uniforms: np.ndarray[Any, Any],
    latent: np.ndarray[Any, Any],
    treatment: np.ndarray[Any, Any],
    strength: float,
) -> Replay:
    """Replay one cell's follow-up, node by node.

    Parameters
    ----------
    data : LongitudinalData
        The observed held survival panel.
    observed : Observed
        Its observed follow-up.
    models : ReplayModels
        The fitted conditionals.
    uniforms : ndarray
        The ``(4, n, T)`` coupling uniforms.
    latent : ndarray
        The shared latent.
    treatment : ndarray
        ``(n,)`` the cell's perturbed node-1 treatment.
    strength : float
        The outcome strength: the shift of the event log-odds per unit of latent.

    Returns
    -------
    Replay
        The raw columns and the shares.
    """
    n, n_times = data.n, data.n_times
    n_causes = len(data.cause_labels) if data.cause_event is not None else 1
    weights = np.asarray(data.weights, dtype=float)
    baseline = models.baseline
    a = observed.treatment
    changed = treatment != a
    shift = strength * latent
    zero = strength == 0.0
    r_censoring, r_event, r_cause, r_other = uniforms
    censoring_raw = np.full((n, n_times), np.nan) if data.censoring_names else None
    event_raw = np.full((n, n_times, n_causes), np.nan)
    inside = np.ones(n, dtype=bool)
    replayed = np.zeros((n, n_times), dtype=bool)
    redrawn = np.zeros((n, n_times), dtype=bool)
    fired_all = np.zeros((n, n_times), dtype=bool)
    new_events = np.zeros((n, n_times), dtype=bool)
    for node in range(n_times):
        replayed[:, node] = inside
        present = observed.present[:, node]
        censored = np.zeros(n, dtype=bool)
        if models.censoring is not None:
            p_observed = models.censoring.probability(node, baseline, a)
            p_replay = models.censoring.probability(node, baseline, treatment)
            v = conditional_uniform(
                p_observed, observed.censored[:, node], r_censoring[:, node], present
            )
            kept = present & ~changed
            censored = inside & np.where(kept, observed.censored[:, node], v < p_replay)
            redrawn[:, node] = inside & present & changed
        if censoring_raw is not None:
            censoring_raw[inside, node] = np.where(censored[inside], 0.0, 1.0)
        stays = inside & ~censored
        at_risk = observed.at_risk[:, node]
        observed_event = observed.event[:, node]
        observed_cause = observed.cause[:, node]
        shares = None if models.shares is None else models.shares.probabilities(node, baseline, a)
        if models.cause_index is None:
            h = models.event.probability(node, baseline, a)
            v = conditional_uniform(h, observed_event, r_event[:, node], at_risk)
            draw = v < shifted(h, shift)
            fired = stays & np.where(at_risk & zero, observed_event, draw)
            if shares is None:
                cause = np.zeros(n, dtype=int)
            else:
                cause = np.where(
                    at_risk & observed_event, observed_cause, draw_cause(shares, r_cause[:, node])
                )
        else:
            j = models.cause_index
            assert models.other is not None and shares is not None
            observed_other = observed_event & (observed_cause != j)
            o = models.other.probability(node, baseline, a)
            v_other = conditional_uniform(o, observed_other, r_other[:, node], at_risk)
            other = stays & np.where(at_risk, observed_other, v_other < o)
            free = at_risk & ~observed_other
            rho = models.event.probability(node, baseline, a)
            v = conditional_uniform(rho, observed_event & ~observed_other, r_event[:, node], free)
            draw = v < shifted(rho, shift)
            named = stays & ~other & np.where(free & zero, observed_event & ~observed_other, draw)
            others = shares.copy()
            others[:, j] = 0.0
            other_cause = np.where(
                at_risk & observed_other, observed_cause, draw_cause(others, r_cause[:, node])
            )
            fired = other | named
            cause = np.where(named, j, other_cause)
        event_raw[stays, node, :] = 0.0
        event_raw[fired, node, cause[fired]] = 1.0
        fired_all[:, node] = fired
        new_events[:, node] = fired & ~(at_risk & observed_event)
        inside = stays & ~fired
    return Replay(
        censoring=censoring_raw,
        event=event_raw,
        censoring_redrawn_share=(
            None if models.censoring is None else _share(redrawn, replayed, weights)
        ),
        extended_share=_share(replayed & ~observed.present, replayed, weights),
        new_event_share=_share(new_events, fired_all, weights),
    )


def calibration_strength(model: PooledModel, column: int) -> tuple[float, str]:
    """The fitted coefficient of one standardized baseline column, or ``nan`` and why.

    Parameters
    ----------
    model : PooledModel
        The replay's event model.
    column : int
        The covariate's position in the baseline block.

    Returns
    -------
    tuple of (float, str)
        The coefficient on the event log-odds scale and the method text.
    """
    from .simulated_confounding import (
        NONCONVERGENCE_METHOD,
        SEPARATION_METHOD,
        _classes_separate,
    )

    if model.model is None or model.design is None or model.target is None:
        return float("nan"), SEPARATION_METHOD
    if _classes_separate(model.design, model.target):
        return float("nan"), SEPARATION_METHOD
    if not model.converged:
        return float("nan"), NONCONVERGENCE_METHOD
    return float(model.model.coef_[0, column]), REPLAY_HAZARD_METHOD


#: The calibration rule of a survival outcome row.
REPLAY_HAZARD_METHOD = (
    "coefficient of the standardized covariate in the replay's pooled logistic event model "
    "(log-odds per standard deviation)"
)
