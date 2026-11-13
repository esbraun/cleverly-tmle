"""Hazard targeting with a covariate after baseline on a held design (X70 part (b1)).

Law (H) of :mod:`tests.discrete_law_hazard_time_varying` is passed as data, one row per
support point weighted by its probability, so with the correct mechanism a targeted fit must
return the enumerated truth to floating tolerance whatever the hazard and continuation
learners.  The detection case of every mutation is "wrong hazard and wrong continuation (both
without ``L_k``), correct mechanism" on a law whose dropout reads ``L_k``; each mutation is the
defect a coding error would make, applied to the score and the direction alike, and keeps a
control law on which its term vanishes.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.exceptions import (
    CapabilityError,
    ConvergenceWarning,
    MethodConfigurationError,
    RefusalReason,
)
from cleverly.interventions import Incremental, ModifiedPolicy, Shift, Stochastic
from cleverly.longitudinal import (
    LTMLE,
    DynamicRegimen,
    LongitudinalData,
    Separable,
    SurvivalCTMLE,
    SurvivalDRTMLE,
)
from cleverly.longitudinal import hazard as hazard_module
from cleverly.longitudinal import hazard_backward as backward
from cleverly.longitudinal.hazard import HazardInputs, HazardState, _held_cells, scores, walk_loss
from cleverly.longitudinal.hazard_backward import node_curve_terms, soft_held_cells
from cleverly.msm import MSM
from tests.discrete_law_hazard_time_varying import CellMeans, CellProbabilities, HeldLaw, NoL

TOL = 1e-10
EXACT = 1e-8
MISSED = 1e-3
STEP = 2e-2
NEVER, ALWAYS = "never", "always"
STATIC = {NEVER: 0, ALWAYS: 1}


class Logistic(LogisticRegression):  # type: ignore[misc]
    """An unpenalised logistic regression, additive in its design."""

    def __init__(self) -> None:
        super().__init__(C=np.inf, tol=1e-12, max_iter=10_000)


def wrong_hazard() -> NoL:
    """Logistic in ``(k, A, W)``: the node fit without ``L_k``."""
    return NoL(Logistic())


def wrong_continuation() -> NoL:
    """Cell means over ``(W, A, s)``: the continuation without ``L_k``."""
    return NoL(CellMeans())


def covariate_case() -> dict[str, Any]:
    """The detection case of a covariate-term mutation.

    The hazard reads ``L_k`` but is misspecified (additive logistic on the full node design),
    and the continuation omits ``L_k`` (a constant).  With a hazard without ``L_k`` as well,
    every fitted ``Q_{k+1}`` is a function of ``(W, A)``, which a continuation without ``L_k``
    reproduces, so the covariate term is exactly zero at the initial fit and a defect in it
    shows only through second-order movement.
    """
    return {"hazard": Logistic(), "continuation": DummyRegressor()}


def fit(
    law: HeldLaw,
    regimens: dict[str, Any],
    *,
    hazard: Any = None,
    continuation: Any = None,
    frame: pd.DataFrame | None = None,
    columns: dict[str, Any] | None = None,
    **settings: Any,
) -> Any:
    """A one-step hazard fit of law-as-data, wrong hazard and continuation by default.

    ``continuation=False`` passes no ``pseudo_learner``, as a baseline-only fit needs.
    """
    if continuation is not False:
        settings["pseudo_learner"] = wrong_continuation() if continuation is None else continuation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LTMLE(
            regimens,
            n_folds=settings.pop("n_folds", 1),
            survival_model=settings.pop("survival_model", "hazard"),
            targeting=settings.pop("targeting", "one_step"),
            step_size=STEP,
            tol=settings.pop("tol", TOL),
            g_bounds=settings.pop("g_bounds", (1e-8, 1.0)),
            outcome_learner=wrong_hazard() if hazard is None else hazard,
            treatment_learner=settings.pop("treatment_learner", CellProbabilities()),
            censoring_learner=CellMeans(),
            **settings,
        ).fit(
            law.frame() if frame is None else frame,
            **(law.columns() if columns is None else columns),
        )


def estimate(result: Any, label: str, t: int, cause: str | None = None) -> float:
    if cause is None:
        return float(result.estimates[f"risk_regimen[{label} @ t={t}]"].psi)
    return float(result.estimates[f"cif_regimen[{label}, {cause} @ t={t}]"].psi)


def one_hot(level: int, arms: int = 2) -> Callable[[int, int], np.ndarray]:
    return lambda w1, w2: np.eye(arms)[level]


def worst_miss(
    law: HeldLaw,
    result: Any,
    densities: dict[str, Callable[..., np.ndarray]],
    horizons: Any = None,
) -> float:
    """The largest distance of an estimate from the enumerated truth."""
    causes = {1: None} if law.causes == 1 else {1: "relapse", 2: "death"}
    miss = 0.0
    for label, density in densities.items():
        for j, cause in causes.items():
            for t in range(1, law.n_times + 1) if horizons is None else horizons:
                miss = max(miss, abs(estimate(result, label, t, cause) - law.truth(density, t, j)))
    return miss


STATIC_DENSITIES = {NEVER: one_hot(0), ALWAYS: one_hot(1)}


# --------------------------------------------------------------- identities


def test_the_regrouping_identity_holds_row_by_row() -> None:
    """``Z - Qbar`` equals the hazard term plus the covariate term (2.5.2), for any inputs."""
    rng = np.random.default_rng(0)
    n, n_causes = 200, 2
    lam = rng.dirichlet(np.ones(n_causes + 1), size=n)[:, :n_causes]
    m, later = rng.random(n), rng.random(n)
    code = rng.integers(0, n_causes + 1, n)
    dn = np.stack([(code == l + 1).astype(float) for l in range(n_causes)], axis=1)
    for j in range(n_causes):
        z = dn[:, j] + (1.0 - dn.sum(axis=1)) * later
        q_bar = lam[:, j] + (1.0 - lam.sum(axis=1)) * m
        kappa = np.stack([(1.0 if l == j else 0.0) - m for l in range(n_causes)], axis=1)
        regrouped = (kappa * (dn - lam)).sum(axis=1) + (1.0 - dn.sum(axis=1)) * (later - m)
        np.testing.assert_allclose(z - q_bar, regrouped, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("n_causes", [1, 2])
def test_the_soft_hazard_identity_holds_on_the_shipped_score(n_causes: int) -> None:
    """The shipped score of a soft panel is ``sum_{s <= t} p(s, j) - F_j^nu(t)`` (2.5.3).

    Each row is one random case: a sub-distribution ``p`` over five nodes and random hazards
    ``nu``.  The panel holds ``rho(s)`` as the at-risk weight and ``p / rho`` as the events.
    """
    rng = np.random.default_rng(1)
    n, nodes = 200, 5
    nu = rng.dirichlet(np.ones(n_causes + 1), size=(n, nodes))[..., :n_causes][:, :, None, :]
    p = rng.dirichlet(np.ones(nodes * n_causes + 1), size=n)[:, :-1].reshape(n, nodes, n_causes)
    spent = np.cumsum(p.sum(axis=-1), axis=1)
    rho = 1.0 - np.concatenate([np.zeros((n, 1)), spent[:, :-1]], axis=1)
    alive = rho > 0.0
    inputs = HazardInputs(
        at_risk=alive,
        events=np.where(alive[..., None], p / np.where(alive, rho, 1.0)[..., None], 0.0),
        observed=np.zeros(n, dtype=np.int64),
        inverse=rho[:, :, None],
        q=np.ones((1, n, 1)),
        weights=np.ones(n),
        alpha=1.0 - 1e-12,
        empty=np.zeros((nodes, 1, n_causes), dtype=bool),
        full=np.zeros((nodes, 1), dtype=bool),
        loss_weight=rho,
    )
    state = HazardState.from_hazards(nu, inputs.bounds)
    horizons = list(range(1, nodes + 1))
    terms = node_curve_terms(state, inputs, horizons)
    incidence = state.incidence[:, :, 0, :]
    for j in range(n_causes):
        for h, t in enumerate(horizons):
            expected = p[:, :t, j].sum(axis=1) - incidence[:, t - 1, j]
            np.testing.assert_allclose(terms[j, h], expected, rtol=0.0, atol=1e-10)


def _review_panel() -> tuple[HazardInputs, np.ndarray, dict[str, np.ndarray]]:
    """Plan review 3's soft panel (``x70b-review/panel.py``): 300 rows, 4 sub-nodes, 2 slots."""
    rng = np.random.default_rng(3)
    n, n_sub, n_slots, n_causes = 300, 4, 2, 2
    lam0 = rng.dirichlet(np.ones(n_causes + 1), size=(n, n_slots))[..., :n_causes] * 0.6
    nu = rng.dirichlet(np.ones(n_causes + 1), size=(n, n_sub - 1, n_slots))[..., :n_causes] * 0.6
    hazards = np.concatenate([lam0[:, None], nu], axis=1)
    observed = rng.integers(0, n_slots, n)
    at_k = rng.random(n) < 0.9
    code = rng.integers(0, n_causes + 1, n)
    dn = np.zeros((n, n_causes))
    dn[np.arange(n), np.minimum(code, n_causes - 1)] = code < n_causes
    p = rng.dirichlet(np.ones((n_sub - 1) * n_causes + 1), size=n)[:, :-1]
    p = p.reshape(n, n_sub - 1, n_causes) * 0.9
    rho = 1.0 - np.concatenate([np.zeros((n, 1)), np.cumsum(p.sum(-1), 1)[:, :-1]], 1)
    h = rng.uniform(0.5, 3.0, n) * at_k
    w = rng.uniform(0.5, 1.5, n)
    w /= w.mean()
    survived = at_k & (code == n_causes)
    at_risk = np.zeros((n, n_sub), dtype=bool)
    at_risk[:, 0] = at_k
    at_risk[:, 1:] = survived[:, None] & (rho > 0)
    events = np.zeros((n, n_sub, n_causes))
    events[:, 0] = dn * at_k[:, None]
    events[:, 1:] = np.where(at_risk[:, 1:, None], p / np.where(rho > 0, rho, 1)[..., None], 0)
    inverse = np.ones((n, n_sub)) * h[:, None]
    inverse[:, 1:] *= rho
    q = rng.dirichlet(np.ones(n_slots), size=n)[None]
    inputs = HazardInputs(
        at_risk=at_risk,
        events=events,
        observed=observed,
        inverse=np.repeat(inverse[:, :, None], n_slots, axis=2),
        q=q,
        weights=w,
        alpha=0.999999,
        empty=np.zeros((n_sub, n_slots, n_causes), dtype=bool),
        full=np.zeros((n_sub, n_slots), dtype=bool),
        loss_weight=w[:, None] * inverse,
    )
    parts = {"dn": dn, "p": p, "h": h, "w": w, "at_k": at_k, "survived": survived}
    return inputs, hazards, parts


def test_the_soft_held_cells_mark_no_fractional_cell_full() -> None:
    """The shipped held cells mark fractional cells full; the variant marks none (check 5).

    With the variant's cells the shipped scores equal the hand EIF bracket and the walk loss
    is finite.
    """
    inputs, hazards, parts = _review_panel()
    _, shipped_full, _ = _held_cells(
        inputs.at_risk, inputs.events, inputs.weights, inputs.observed, 2, None
    )
    assert int(shipped_full.sum()) == 6
    empty, full, _ = soft_held_cells(
        inputs.at_risk, inputs.events, inputs.loss_weight, inputs.observed, 2, None
    )
    assert not full.any()
    assert not empty.any()
    inputs = replace(inputs, empty=empty, full=full)
    state = HazardState.from_hazards(hazards, inputs.bounds, empty, full)
    horizons = [1, 2, 3, 4]
    score, _ = scores(state, inputs, horizons)
    n = inputs.n
    rows = np.arange(n)
    lam = state.hazards[rows, :, inputs.observed]
    incidence = state.incidence[rows, :, inputs.observed]
    survival = state.survival[rows, :, inputs.observed]
    for j in range(2):
        for h, t in enumerate(horizons):
            m = (incidence[:, t - 1, j] - incidence[:, 0, j]) / survival[:, 0]
            kappa = np.stack([(1.0 if l == j else 0.0) - m for l in range(2)], axis=1)
            hazard_term = (kappa * (parts["dn"] - lam[:, 0])).sum(axis=1) * parts["at_k"]
            covariate = np.where(parts["survived"], parts["p"][:, : t - 1, j].sum(axis=1) - m, 0.0)
            hand = np.mean(parts["w"] * parts["h"] * (hazard_term + covariate))
            assert score[0, j, h] == pytest.approx(hand, abs=1e-12)
    assert np.isfinite(walk_loss(state, inputs))


def test_the_soft_held_cells_are_the_shipped_cells_on_binary_events() -> None:
    """On binary events and the observation weight the variant returns the shipped cells."""
    rng = np.random.default_rng(5)
    n, nodes, n_causes, levels = 400, 3, 2, 3
    at_risk = rng.random((n, nodes)) < 0.8
    code = rng.integers(0, 4, (n, nodes))
    events = np.stack([(code == l + 1) for l in range(n_causes)], axis=-1).astype(float)
    events[(code == 3) & (np.arange(nodes) == 2)[None, :]] = 0.0
    events *= at_risk[..., None]
    # A cell with every follower holding an event, and one with no event of cause 2.
    observed = rng.integers(0, levels, n)
    events[observed == 2, 1] = [1.0, 0.0]
    events[observed == 1, 2, 1] = 0.0
    weights = rng.uniform(0.0, 2.0, n) * (rng.random(n) < 0.9)
    shipped = _held_cells(at_risk, events, weights, observed, levels, None)
    variant = soft_held_cells(
        at_risk, events, np.broadcast_to(weights[:, None], at_risk.shape), observed, levels, None
    )
    assert shipped[0].any() and shipped[1].any()
    np.testing.assert_array_equal(shipped[0], variant[0])
    np.testing.assert_array_equal(shipped[1], variant[1])


# ------------------------------------------------------------------ spies


@contextmanager
def spied(name: str) -> Iterator[list[Any]]:
    """Record every return value of ``hazard_backward.<name>`` while the block runs."""
    original = getattr(backward, name)
    calls: list[Any] = []

    def spy(*args: Any, **kwargs: Any) -> Any:
        out = original(*args, **kwargs)
        calls.append(out)
        return out

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(backward, name, spy)
        yield calls


@contextmanager
def mutated_panel(change: Callable[[Any], HazardInputs]) -> Iterator[None]:
    """Every node panel's inputs replaced by ``change(panel)``, its initial state rebuilt."""
    original = backward.node_panel

    def mutant(*args: Any, **kwargs: Any) -> Any:
        panel = original(*args, **kwargs)
        inputs = change(panel)
        empty, full = inputs.held_cells()
        hazards = panel.initial.hazards
        initial = HazardState.from_hazards(hazards, inputs.bounds, empty, full, inputs.node_bounds)
        return replace(panel, inputs=inputs, initial=initial)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(backward, "node_panel", mutant)
        yield


@contextmanager
def frozen_direction(sub_nodes: slice) -> Iterator[None]:
    """The walk's direction zeroed at ``sub_nodes``, so the walk does not move those hazards."""
    original = hazard_module.direction

    def mutant(*args: Any, **kwargs: Any) -> Any:
        step = np.array(original(*args, **kwargs), copy=True)
        step[:, sub_nodes] = 0.0
        return step

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(hazard_module, "direction", mutant)
        yield


def _drop(panel: Any, sub_nodes: slice) -> HazardInputs:
    at_risk = np.array(panel.inputs.at_risk, copy=True)
    at_risk[:, sub_nodes] = False
    return replace(panel.inputs, at_risk=at_risk)


def test_a_panel_holds_its_node_rows_and_its_scale_sums_to_the_record() -> None:
    """Panel rows, the ``n_k / n`` rescaling of the scale, and a zero-weight row's slot (4.4)."""
    law = HeldLaw()
    frame = law.frame()
    with spied("backward_pass") as passes:
        result = fit(law, STATIC, frame=frame)
    n = len(frame)
    for (label, level), entry in zip(STATIC.items(), passes, strict=True):
        node2 = next(panel for panel in entry.panels if panel.node == 2)
        entering = (frame["C1"] == 1) & (frame["Y1"] == 0) & (frame["A"] == level)
        np.testing.assert_array_equal(node2.rows, np.flatnonzero(entering.to_numpy()))
        scale = np.zeros((1, law.n_times))
        for panel, state in zip(entry.panels, entry.states, strict=True):
            rows = hazard_module._score_parts(state, panel.inputs, panel.horizons)[2]
            scale[:, [t - 1 for t in panel.reported]] += rows[0].sum(axis=-1) / n
        record = result.fits[f"{label} @ t=4"].hazard
        np.testing.assert_allclose(record.score_scale, scale.reshape(-1), rtol=0.0, atol=1e-12)
    # A policy that puts no mass on A = 0 when W1 = 1: those rows carry no weight at node 1,
    # whose two slots let the observed slot change.
    with spied("backward_pass") as passes:
        fit(law, {"half": Stochastic(_half_policy, "q", density_kind="known")}, frame=frame)
    node1 = passes[0].panels[0]
    state = passes[0].states[0]
    inputs = node1.inputs
    assert node1.node == 1 and inputs.q.shape[-1] == 2
    idle = inputs.loss_weight[:, 0] == 0.0
    assert idle.any()
    moved = replace(inputs, observed=np.where(idle, 1 - inputs.observed, inputs.observed))
    np.testing.assert_array_equal(
        scores(state, moved, node1.horizons)[0], scores(state, inputs, node1.horizons)[0]
    )


def _half_policy(frame: Any) -> np.ndarray:
    p = np.where(np.asarray(frame["W1"], dtype=float) == 1.0, 1.0, 0.5)
    return np.column_stack([1.0 - p, p])


# --------------------------------------------------------- law-as-data witnesses


def _policy(frame: Any) -> np.ndarray:
    p = 0.3 + 0.4 * np.asarray(frame["W1"], dtype=float)
    return np.column_stack([1.0 - p, p])


POLICY = Stochastic(_policy, "q", density_kind="known")
RULE = DynamicRegimen("rule", (lambda h: np.asarray(h["W1"], dtype=int),), rule_kind="known")


def policy_density(w1: int, w2: int) -> np.ndarray:
    p = 0.3 + 0.4 * w1
    return np.array([1.0 - p, p])


def rule_density(w1: int, w2: int) -> np.ndarray:
    return np.eye(2)[w1]


DENSITIES = {**STATIC_DENSITIES, "policy": policy_density, "rule": rule_density}
REGIMENS = {**STATIC, "policy": POLICY, "rule": RULE}


@pytest.mark.parametrize("causes", [1, 2])
def test_a_wrong_hazard_and_continuation_with_the_right_mechanism_is_exact(causes: int) -> None:
    """Static arms, a rule and a known policy return the truth at every horizon and cause."""
    law = HeldLaw(causes=causes)
    result = fit(law, REGIMENS)
    assert worst_miss(law, result, DENSITIES) < EXACT
    for label in REGIMENS:
        record = result.fits[f"{label}{', relapse' if causes == 2 else ''} @ t=4"].hazard
        assert record.converged
        assert len(record.nodes) == law.n_times
        assert record.n_iter > 0
    # The untargeted plug-in misses, so the exact estimate is not an accident.
    with backward.untargeted():
        untargeted = fit(law, REGIMENS)
    assert worst_miss(law, untargeted, DENSITIES) > MISSED
    record = untargeted.fits[f"{NEVER}{', relapse' if causes == 2 else ''} @ t=4"].hazard
    assert record.n_iter == 0
    assert record.algorithm == "untargeted"
    assert all(node.algorithm == "untargeted" for node in record.nodes)
    assert "was not targeted" in str(untargeted.summary(detail="full"))
    # The context is left: the next fit is targeted.
    assert (
        fit(law, REGIMENS)
        .fits[f"{NEVER}{', relapse' if causes == 2 else ''} @ t=4"]
        .hazard.algorithm
        == "one_step"
    )


def _up(a: Any, h: Any) -> Any:
    return np.minimum(np.asarray(a, dtype=float) + 1.0, 2.0)


def up_density(law: HeldLaw) -> Callable[[int, int], np.ndarray]:
    """The law of ``min(A + 1, 2)`` given ``W``: the 'one level up' policy as a density."""

    def density(w1: int, w2: int) -> np.ndarray:
        g = np.array([law.g(a, w1, w2) for a in range(3)])
        return np.array([0.0, g[0], g[1] + g[2]])

    return density


@pytest.mark.parametrize("causes", [1, 2])
def test_a_misspecified_hazard_reading_the_covariate_is_exact(causes: int) -> None:
    """The covariate-term detection case unmutated: the one witness whose covariate term is first
    order at the initial fit."""
    law = HeldLaw(causes=causes)
    result = fit(law, REGIMENS, **covariate_case())
    assert worst_miss(law, result, DENSITIES) < EXACT


@pytest.mark.parametrize("ratio", ["density", "classifier"])
def test_a_one_level_up_policy_on_three_arms_is_exact(ratio: str) -> None:
    """A modified treatment policy at node 1 of a three-arm held design returns the truth."""
    law = HeldLaw(arms=3)
    up = ModifiedPolicy("up", apply=_up, policy_kind="known")
    result = fit(law, {"up": up, "low": 0}, ratio=ratio)
    assert worst_miss(law, result, {"up": up_density(law), "low": one_hot(0, 3)}) < EXACT


@pytest.mark.parametrize("case", ["plan", "covariate"])
def test_a_continuous_dose_shift_at_node_one_is_exact(case: str) -> None:
    """``Shift(1.0, cap=2.0)`` on the three-arm law read as a continuous dose: the slot branches
    of a continuous node 1 (observed dose and branch dose)."""
    import cleverly.interventions.policy as policy_module
    from tests.longitudinal_mtp import exact_bins

    law = HeldLaw(arms=3)
    columns = law.columns()
    columns["continuous_treatment"] = ["A"]
    kw = covariate_case() if case == "covariate" else {}
    with pytest.MonkeyPatch.context() as patch, exact_bins():
        # The law's doses are the three atoms 0, 1, 2; the atom check would read them as a
        # categorical node.
        patch.setattr(policy_module, "_ATOM_CHECK", False)
        with spied("node_slots") as calls:
            result = fit(law, {"up": Shift(1.0, cap=2.0)}, columns=columns, **kw)
    head = next(entry for entry in calls if entry.node == 1)
    assert head.n_slots == 2
    assert worst_miss(law, result, {"up": up_density(law)}) < EXACT


def _tilt_columns(law: HeldLaw, frame: pd.DataFrame) -> dict[str, Any]:
    columns = law.columns()
    frame["p1"] = [
        law.g(1, int(w1), int(w2)) for w1, w2 in zip(frame["W1"], frame["W2"], strict=True)
    ]
    frame["p0"] = 1.0 - frame["p1"]
    columns["treatment_probabilities"] = {"A": {0: "p0", 1: "p1"}}
    return columns


def tilt_density(law: HeldLaw, delta: float) -> Callable[[int, int], np.ndarray]:
    def density(w1: int, w2: int) -> np.ndarray:
        p = law.g(1, w1, w2)
        tilted = delta * p / (delta * p + 1.0 - p)
        return np.array([1.0 - tilted, tilted])

    return density


def test_an_odds_tilt_of_a_declared_mechanism_is_exact() -> None:
    """A node-1 tilt of a declared mechanism arrives as a known policy and returns the truth."""
    law = HeldLaw()
    frame = law.frame()
    result = fit(law, {"tilt": Incremental(2.0)}, frame=frame, columns=_tilt_columns(law, frame))
    assert worst_miss(law, result, {"tilt": tilt_density(law, 2.0)}) < EXACT


# ---------------------------------------------------------------- influence curve


def _saturated(law: HeldLaw, regimens: dict[str, Any], **settings: Any) -> Any:
    return fit(law, regimens, hazard=CellMeans(), continuation=CellMeans(), **settings)


def test_the_influence_curve_is_the_efficient_influence_function() -> None:
    """Saturated learners: no walk moves, and the curve is the iterated-mean fit's and the EIF."""
    law = HeldLaw(causes=1)
    regimens = {**STATIC, "policy": POLICY}
    result = _saturated(law, regimens)
    reference = _saturated(law, regimens, survival_model="iterated_mean", targeting="iterative")
    frame = law.frame()
    weights = frame["wt"].to_numpy() / frame["wt"].mean()
    densities = {**STATIC_DENSITIES, "policy": policy_density}
    for label in regimens:
        for t in range(1, law.n_times + 1):
            key = f"{label} @ t={t}"
            curve = result.fits[key].influence_curve_scaled
            assert result.fits[key].hazard.n_iter == 0
            np.testing.assert_allclose(
                curve, reference.fits[key].influence_curve_scaled, rtol=0.0, atol=1e-10
            )
            exact = law.eif(densities[label], t) * weights
            np.testing.assert_allclose(curve, exact, rtol=0.0, atol=1e-10)


def _curve_without_covariate_term() -> Any:
    original = backward.node_curve_terms

    def mutant(state: Any, inputs: HazardInputs, horizons: Any) -> Any:
        at_risk = np.array(inputs.at_risk, copy=True)
        at_risk[:, 1:] = False
        return original(state, replace(inputs, at_risk=at_risk), horizons)

    return mutant


@pytest.mark.parametrize("moves", [True, False])
def test_a_curve_without_its_covariate_term_misses_the_efficient_spread(moves: bool) -> None:
    """Mutation: the curve read from sub-node 0 only.  Control: ``L`` moves no later hazard."""
    law = HeldLaw(l_moves_hazard=moves)
    full = _saturated(law, STATIC)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(backward, "node_curve_terms", _curve_without_covariate_term())
        mutated = _saturated(law, STATIC)
    frame = law.frame()
    weights = frame["wt"].to_numpy() / frame["wt"].mean()
    probs = law.probs / law.probs.sum()
    for label, level in STATIC.items():
        key = f"{label} @ t=4"
        exact = law.eif(one_hot(level), 4)
        sd = np.sqrt(np.sum(probs * exact**2))
        curve = mutated.fits[key].influence_curve_scaled / weights
        mutated_sd = np.sqrt(np.sum(probs * curve**2))
        if moves:
            assert abs(mutated_sd / sd - 1.0) > 0.01
        else:
            np.testing.assert_allclose(
                mutated.fits[key].influence_curve_scaled,
                full.fits[key].influence_curve_scaled,
                rtol=0.0,
                atol=1e-10,
            )


# ------------------------------------------------------------------ mutations


def _detect(law: HeldLaw, regimens: dict[str, Any], densities: dict[str, Any], **kw: Any) -> float:
    return worst_miss(law, fit(law, regimens, **kw), densities)


def test_mutation_no_continuation_term() -> None:
    """Sub-nodes after 0 leave the score and the direction; a saturated continuation controls it.

    Detection is :func:`covariate_case`.  ``T = 3``: at ``T = 4`` the control's frozen walk at
    node 2 needs about 1e5 steps to reach ``tol``, a property of a walk with frozen directions.
    """
    law = HeldLaw(n_times=3)
    with (
        mutated_panel(lambda panel: _drop(panel, slice(1, None))),
        frozen_direction(slice(1, None)),
    ):
        assert _detect(law, STATIC, STATIC_DENSITIES, **covariate_case()) > MISSED
        control = _detect(
            law, STATIC, STATIC_DENSITIES, hazard=Logistic(), continuation=CellMeans()
        )
        assert control < EXACT


def test_mutation_no_hazard_term() -> None:
    """Sub-node 0 leaves the panel; a saturated hazard learner is the control."""
    law = HeldLaw()
    with mutated_panel(lambda panel: _drop(panel, slice(0, 1))), frozen_direction(slice(0, 1)):
        assert _detect(law, STATIC, STATIC_DENSITIES) > MISSED
        assert _detect(law, STATIC, STATIC_DENSITIES, hazard=CellMeans()) < EXACT


def _with_retention(law: HeldLaw, frame: pd.DataFrame) -> Callable[[Any], HazardInputs]:
    def change(panel: Any) -> HazardInputs:
        c = law.true_retention(frame, panel.node)[panel.rows]
        inverse = np.array(panel.inputs.inverse, copy=True)
        loss = np.array(panel.inputs.loss_weight, copy=True)
        inverse[:, 1:] *= c[:, None, None]
        loss[:, 1:] *= c[:, None]
        return replace(panel.inputs, inverse=inverse, loss_weight=loss)

    return change


@pytest.mark.parametrize("censored", [True, False])
def test_mutation_weight_without_its_own_node_retention(censored: bool) -> None:
    """The covariate term weighted by ``h_k c_k``; the control law has no dropout."""
    law = HeldLaw(censored=censored)
    with mutated_panel(_with_retention(law, law.frame())):
        miss = _detect(law, STATIC, STATIC_DENSITIES, **covariate_case())
    assert miss > MISSED if censored else miss < EXACT


@pytest.mark.parametrize("saturated", [False, True])
def test_mutation_stale_pseudo_outcome(saturated: bool) -> None:
    """Node ``k``'s soft events read node ``k + 1``'s initial curve; saturated fits are the control."""
    law = HeldLaw()
    last: list[Any] = []
    walk = backward.walk_curve
    pseudo = backward.pseudo_distribution

    def walk_spy(initial: Any, *args: Any, **kwargs: Any) -> Any:
        last[:] = [initial]
        return walk(initial, *args, **kwargs)

    def stale(state: Any, *args: Any) -> Any:
        return pseudo(last[0] if last else state, *args)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(backward, "walk_curve", walk_spy)
        patch.setattr(backward, "pseudo_distribution", stale)
        kw = {"hazard": CellMeans(), "continuation": CellMeans()} if saturated else covariate_case()
        miss = _detect(law, STATIC, STATIC_DENSITIES, **kw)
    assert miss < EXACT if saturated else miss > MISSED


def _hard_at_risk(panel: Any) -> HazardInputs:
    inputs = panel.inputs
    h = inputs.inverse[:, 0, 0]
    rho = np.divide(
        inputs.inverse[:, 1:, 0],
        h[:, None],
        out=np.zeros_like(inputs.inverse[:, 1:, 0]),
        where=h[:, None] > 0,
    )
    alive = np.array(inputs.at_risk, copy=True)
    entered = alive[:, 1:].any(axis=1) | (rho > 0).any(axis=1)
    alive[:, 1:] = entered[:, None] & (h[:, None] > 0)
    events = np.array(inputs.events, copy=True)
    events[:, 1:] = events[:, 1:] * rho[..., None]
    inverse = np.array(inputs.inverse, copy=True)
    inverse[:, 1:] = h[:, None, None]
    loss = np.array(inputs.loss_weight, copy=True)
    loss[:, 1:] = inputs.weights[:, None] * h[:, None]
    return replace(inputs, at_risk=alive, events=events, inverse=inverse, loss_weight=loss)


@pytest.mark.parametrize("last_only", [False, True])
def test_mutation_hard_at_risk(last_only: bool) -> None:
    """``rho`` replaced by one at every later sub-node; events at the last node only control it."""
    law = HeldLaw(last_only=last_only)
    with mutated_panel(_hard_at_risk):
        miss = _detect(law, STATIC, STATIC_DENSITIES, **covariate_case())
    assert miss < EXACT if last_only else miss > MISSED


@contextmanager
def tail_dropped() -> Iterator[None]:
    """``kappa = 1{l = j}`` alone at sub-nodes after 0: every tail ratio zeroed there."""
    original = hazard_module._per_survivor

    def mutant(value: Any, survival: Any) -> Any:
        out = np.array(original(value, survival), copy=True)
        if out.ndim >= 2:
            out[:, 1:] = 0.0
        return out

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(hazard_module, "_per_survivor", mutant)
        yield


@pytest.mark.parametrize("n_times", [4, 2])
def test_mutation_continuation_kappa_without_its_tail(n_times: int) -> None:
    """The fit reports ``t = T`` only.  At ``T = 2`` the tail at node 1's one continuation
    sub-node, which is the horizon, is zero: the control.

    Do not widen this to every horizon.  A node walk that targets every ``t >= k`` solves the
    per-horizon prefix sums of the continuation residuals, and when the tail ratio is the same
    for every unit the mutated score family spans the correct one, so the mutation changes
    nothing (1.3e-10 with a constant continuation).  With one target per node the tail enters
    at first order.
    """
    law = HeldLaw(n_times=n_times)
    with tail_dropped():
        result = fit(law, STATIC, horizons=[n_times], **covariate_case())
    miss = worst_miss(law, result, STATIC_DENSITIES, horizons=[n_times])
    assert miss > MISSED if n_times == 4 else miss < EXACT


def _follows(w1: int, w2: int) -> int:
    return w1


def _observed_slot(panel: Any) -> HazardInputs:
    if panel.node != 1:
        return panel.inputs
    q = np.zeros_like(panel.inputs.q)
    q[0, np.arange(panel.inputs.n), panel.inputs.observed] = 1.0
    return replace(panel.inputs, q=q)


def _one_hot_policy(frame: Any) -> np.ndarray:
    return np.eye(2)[np.asarray(frame["W1"], dtype=int)]


@pytest.mark.parametrize("control", [False, True])
def test_mutation_observed_slot_at_a_policy_node(control: bool) -> None:
    """The node-1 plug-in reads the observed level, not the policy mixture.

    The control is a one-hot policy at ``d(W) = W1`` on a law where every unit takes
    ``A = W1``, so the observed level is the policy's slot for every unit.
    """
    law = HeldLaw(follow=_follows) if control else HeldLaw()
    policy = Stochastic(_one_hot_policy if control else _policy, "q", density_kind="known")
    density = rule_density if control else policy_density
    with spied("node_slots") as calls, mutated_panel(_observed_slot):
        result = fit(law, {"policy": policy})
    head = [entry for entry in calls if entry.node == 1]
    miss = worst_miss(law, result, {"policy": density})
    if control:
        # The resolver turns a one-hot density into the rule it equals, so the node holds one
        # slot and the mutation touches nothing: a no-touch check of the observed slot.
        assert head[0].n_slots == 1
        assert miss < EXACT
    else:
        assert head[0].n_slots == 2
        assert miss > MISSED


# ---------------------------------------------------------------- coherence


def _draw(law: HeldLaw, n: int, seed: int) -> pd.DataFrame:
    frame = law.frame()
    rng = np.random.default_rng(seed)
    probs = law.probs / law.probs.sum()
    rows = rng.choice(len(frame), size=n, p=probs)
    out = frame.iloc[rows].reset_index(drop=True)
    out["wt"] = 1.0
    return out


def test_the_curve_is_coherent() -> None:
    """Each one-step curve is non-decreasing; causes and survival sum to one per unit.

    The draw (300 rows, seed 2) is fixed: on it the iterated-mean curve with the same
    learners decreases by about 1e-3, so the property is not vacuous.  A search of seeds 0-7
    at 300 and 1000 rows found this one draw where the iterated-mean curve decreases by more
    than 1e-6.
    """
    law = HeldLaw(causes=2, flat=3)
    frame = _draw(law, 300, 2)
    learners = {"hazard": Logistic(), "continuation": LinearRegression()}
    with spied("backward_pass") as passes:
        result = fit(law, STATIC, frame=frame, **learners)
    for label in STATIC:
        for cause in ("relapse", "death"):
            curve = [estimate(result, label, t, cause) for t in range(1, 5)]
            assert np.all(np.diff(curve) >= 0.0)
    assert passes
    for entry in passes:
        for state in entry.states:
            total = state.incidence.sum(axis=-1) + state.survival
            np.testing.assert_allclose(total, 1.0, rtol=0.0, atol=1e-12)
    reference = fit(
        law,
        STATIC,
        frame=frame,
        survival_model="iterated_mean",
        targeting="iterative",
        **learners,
    )
    drops = [
        estimate(reference, label, t, cause) - estimate(reference, label, t + 1, cause)
        for label in STATIC
        for cause in ("relapse", "death")
        for t in range(1, 4)
    ]
    assert max(drops) > 1e-6


# --------------------------------------------------------------- weights


def test_the_weight_is_the_iterated_mean_clever_covariate() -> None:
    """``h_k`` equals the iterated-mean fit's clever covariate of the same plan at every node."""
    law = HeldLaw()
    frame = law.frame()
    columns = _tilt_columns(law, frame)
    regimens = {**REGIMENS, "tilt": Incremental(2.0)}
    hazard = fit(law, regimens, frame=frame, columns=columns)
    reference = fit(
        law,
        regimens,
        frame=frame,
        columns=columns,
        survival_model="iterated_mean",
        targeting="iterative",
    )
    for label in regimens:
        for t in (1, 4):
            ours = hazard.fits[f"{label} @ t={t}"].hazard_clever
            steps = reference.fits[f"{label} @ t={t}"].steps
            for step in steps:
                np.testing.assert_allclose(
                    ours[:, step.time - 1], step.clever, rtol=1e-12, atol=1e-12
                )


@pytest.mark.parametrize("ratio", ["density", "classifier"])
def test_the_weight_of_a_policy_on_three_arms_is_the_iterated_mean_one(ratio: str) -> None:
    law = HeldLaw(arms=3)
    up = {"up": ModifiedPolicy("up", apply=_up, policy_kind="known")}
    hazard = fit(law, up, ratio=ratio)
    reference = fit(law, up, ratio=ratio, survival_model="iterated_mean", targeting="iterative")
    ours = hazard.fits["up @ t=4"].hazard_clever
    for step in reference.fits["up @ t=4"].steps:
        np.testing.assert_allclose(ours[:, step.time - 1], step.clever, rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------- held cells


def test_a_continuation_cell_with_no_event_of_a_cause_is_held_at_zero() -> None:
    """Cause 2 cannot occur after node 2: its continuation cells are held, and the fit is exact."""
    law = HeldLaw(causes=2, no_late_second=True)
    with spied("node_panel") as panels:
        result = fit(law, STATIC)
    assert worst_miss(law, result, STATIC_DENSITIES) < EXACT
    node1 = [panel for panel in panels if panel.node == 1]
    # Sub-nodes 2 and 3 are nodes 3 and 4, where cause 2 never happens.
    assert all(panel.inputs.empty[2:, :, 1].any() for panel in node1)
    assert all(result.fits[f"{label}, death @ t=4"].hazard.converged for label in STATIC)


# --------------------------------------------------------------- L1 alone


@pytest.mark.parametrize(
    ("model", "pooling"),
    [("hazard", "node"), ("hazard", "pooled"), ("iterated_mean", "pooled")],
)
def test_a_covariate_at_node_one_alone_is_the_baseline_fit(model: str, pooling: str) -> None:
    """``W2`` as a node-1 block gives the fit with ``W2`` declared baseline.

    The iterated-mean case is pooled censoring, newly reachable since the pooled trigger reads
    blocks after the first node only.
    """
    law = HeldLaw(n_times=3)
    frame = law.frame()
    for name in ("L2", "L3"):
        frame = frame.drop(columns=name)
    base = law.columns()
    base["time_varying"] = [[], [], []]
    first = law.columns(l1=True)
    first["time_varying"] = [["W2"], [], []]
    kw: dict[str, Any] = {"hazard": CellMeans(), "hazard_pooling": pooling}
    if model == "hazard":
        kw["continuation"] = False
    else:
        kw.update(continuation=CellMeans(), survival_model=model, targeting="iterative")
    a = fit(law, STATIC, frame=frame, columns=base, **kw)
    b = fit(law, STATIC, frame=frame, columns=first, **kw)
    for name, value in a.estimates.items():
        assert b.estimates[name].psi == pytest.approx(value.psi, abs=1e-12)


def _fit_settings() -> dict[str, Any]:
    return {
        "outcome_learner": CellMeans(),
        "treatment_learner": CellProbabilities(),
        "censoring_learner": CellMeans(),
        "n_folds": 1,
    }


def _l1_or_l2(law: HeldLaw, l1_only: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = law.frame()
    columns = law.columns(l1=l1_only)
    if l1_only:
        frame = frame.drop(columns=[f"L{k}" for k in range(2, law.n_times + 1)])
        columns["time_varying"] = [["W2"]] + [[] for _ in range(law.n_times - 1)]
    return frame, columns


@pytest.mark.parametrize("l1_only", [True, False])
def test_the_companions_still_refuse_covariates_after_baseline(l1_only: bool) -> None:
    """``SurvivalCTMLE``, ``SurvivalDRTMLE`` and ``Separable`` keep their trigger, ``L_1`` included."""
    frame, columns = _l1_or_l2(HeldLaw(n_times=2), l1_only)
    baseline_only = "with baseline covariates only"
    with pytest.raises(CapabilityError, match=baseline_only):
        SurvivalCTMLE(STATIC, outcome_learner=CellMeans()).fit(frame, **columns)
    with pytest.raises(CapabilityError, match=baseline_only):
        SurvivalDRTMLE(
            STATIC,
            reference=NEVER,
            outcome_learner=CellMeans(),
            treatment_learner=CellMeans(),
            censoring_learner=CellMeans(),
        ).fit(frame, **columns)
    frame, columns = _l1_or_l2(HeldLaw(n_times=2, causes=2), l1_only)
    separable = Separable(arms={"relapse": 1, "death": 0}, first="relapse")
    with pytest.raises(CapabilityError, match="after baseline"):
        LTMLE(
            {NEVER: 0, "sep": separable},
            survival_model="hazard",
            targeting="one_step",
            **_fit_settings(),
        ).fit(frame, **columns)


# -------------------------------------------------------------- carried settings


def _markov_data(law: HeldLaw) -> LongitudinalData:
    return replace(LongitudinalData.from_frame(law.frame(), **law.columns()), markov_order=0)


@pytest.mark.parametrize("l1_only", [False, True])
def test_h32_a_markov_order_is_refused_before_any_learner(l1_only: bool) -> None:
    law = HeldLaw(n_times=4, censored=False)
    frame, columns = _l1_or_l2(law, l1_only)
    data = replace(LongitudinalData.from_frame(frame, **columns), markov_order=0)
    message = (
        "markov_order=0 with survival_model='hazard' is not built. On a design with one "
        "baseline treatment held over the nodes, the window starts its treatment blocks after "
        "node 1"
    )
    with pytest.raises(CapabilityError, match=_escape(message)) as raised:
        LTMLE(STATIC, survival_model="hazard", targeting="one_step", **_refuse()).fit(data)
    assert raised.value.refusal.reason is RefusalReason.API_REDIRECTION


def test_the_window_would_pool_the_node_hazards_over_the_arms() -> None:
    """Why H32 exists: saturated outcome learners and a wrong mechanism under ``markov_order=0``.

    Outcome-side double robustness should make the fit exact.  With the refusal lifted the
    hazard fit misses, because the order-0 window leaves the held treatment out of the node
    hazard designs from node 3 on; the iterated-mean fit on the same data, whose static arms
    regress on followers only, is exact.
    """
    from sklearn.dummy import DummyClassifier

    law = HeldLaw(censored=False)
    data = _markov_data(law)
    assert data.history_design(3, include_current=True).shape[1] == 3  # W1, W2, L3: no A
    settings = {
        "frame": data,
        "columns": {},
        "hazard": CellMeans(),
        "continuation": CellMeans(),
        "treatment_learner": DummyClassifier(strategy="prior"),
    }
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(backward, "refuse_hazard_markov", lambda data: None)
        lifted = fit(law, STATIC, **settings)
    assert worst_miss(law, lifted, STATIC_DENSITIES) > MISSED
    reference = fit(law, STATIC, survival_model="iterated_mean", targeting="iterative", **settings)
    assert worst_miss(law, reference, STATIC_DENSITIES) < EXACT


def test_node_bounds_bound_the_node_hazard_only() -> None:
    """Bounds that contain the true node hazards: the truth, and the continuation unbounded."""
    law = HeldLaw()
    with spied("node_panel") as panels:
        result = fit(law, STATIC, node_bounds={1.0: (0.0, 0.6), 2.0: (0.0, 0.6)})
    assert worst_miss(law, result, STATIC_DENSITIES) < EXACT
    node1 = next(panel for panel in panels if panel.node == 1)
    bounds = node1.inputs.node_bounds
    assert bounds is not None
    assert bounds.bounded[0] and not bounds.bounded[1:].any()


def test_a_bound_that_excludes_a_cell_warns_and_is_stamped_on_the_record() -> None:
    """Events at the last node only, a positive lower bound at node 2: that node cell is out."""
    law = HeldLaw(n_times=3, last_only=True)
    with pytest.warns(ConvergenceWarning, match="excludes the data") as caught:
        result = LTMLE(
            STATIC,
            survival_model="hazard",
            targeting="one_step",
            step_size=STEP,
            tol=TOL,
            node_bounds={2.0: (0.01, 0.6)},
            **{**_fit_settings(), "pseudo_learner": wrong_continuation()},
        ).fit(law.frame(), **law.columns())
    texts = [str(w.message) for w in caught if "excludes the data" in str(w.message)]
    assert len(texts) == 1
    assert texts[0].startswith("the declared node bound at node(s) 2 excludes the data")
    for fit_ in result.fits.values():
        record = fit_.hazard
        assert record.bound_excludes_data
        assert {cell[0] for cell in record.bound_excludes_data} == {2}
        assert {entry[0] for entry in record.share_clipped} == {2}
    assert "node bounds clipped" in str(result.summary(detail="full"))


def test_clusters_iterate_best_and_the_bootstrap_run_on_this_path() -> None:
    """Smoke checks of three carried settings: each fit runs and stays exact where it can."""
    law = HeldLaw(n_times=3)
    frame = law.frame()
    frame["cluster"] = np.arange(len(frame)) // 4
    clustered = fit(law, STATIC, frame=frame, columns={**law.columns(), "id": "cluster"})
    assert worst_miss(law, clustered, STATIC_DENSITIES) < EXACT
    assert np.isfinite(clustered.estimates["risk_regimen[never @ t=3]"].std_error)
    best = fit(law, STATIC, iterate="best")
    assert worst_miss(law, best, STATIC_DENSITIES) < EXACT
    assert all(node.best_iterate is not None for node in best.fits["never @ t=3"].hazard.nodes)
    booted = fit(law, STATIC, n_bootstrap=3, random_state=0)
    assert booted.bootstrap is not None
    assert booted.bootstrap.n_failed == 0


def test_the_nuisance_report_names_each_unscored_continuation() -> None:
    from cleverly.validation.longitudinal import LONGITUDINAL_CONTINUATION_NOT_SCORED

    law = HeldLaw(n_times=3, causes=2)
    report = fit(law, STATIC).diagnostics.nuisance_models()
    rows = {
        (row.regimen, row.time, row.cause)
        for row in report.omissions
        if row.reason == LONGITUDINAL_CONTINUATION_NOT_SCORED
    }
    expected = {
        (label, time, cause)
        for label in STATIC
        for time in (1, 2)
        for cause in ("relapse", "death")
    }
    assert rows == expected
    assert any(row.role == "hazard" for row in report.rows)


def test_cross_fitting_covers_the_truth_and_solves_every_node() -> None:
    """Five folds on draws: within four standard errors, and every pooled node walk solved.

    ``CellMeans`` predicts zero on a held-out row whose cell its training folds never saw, so
    the cumulative mechanism is bounded at 0.01 (``g_bounds=(0.01, 1.0)`` is the default
    ``DEFAULT_LTMLE_G_BOUNDS``, passed here to name it), below the law's smallest true cumulative
    (about 0.026, ``0.25 * expit(0.3) ** 4``): the bound moves no correct prediction.
    """
    law = HeldLaw(causes=1)
    frame = _draw(law, 20_000, 3)
    result = fit(
        law,
        STATIC,
        frame=frame,
        n_folds=5,
        hazard=wrong_hazard(),
        continuation=CellMeans(),
        g_bounds=(0.01, 1.0),
        random_state=0,
    )
    for label, level in STATIC.items():
        for t in range(1, law.n_times + 1):
            value = result.estimates[f"risk_regimen[{label} @ t={t}]"]
            truth = law.truth(one_hot(level), t)
            assert abs(value.psi - truth) < 4.0 * value.std_error
        record = result.fits[f"{label} @ t=4"].hazard
        assert record.converged
        assert all(node.relative_score <= TOL for node in record.nodes)


def test_a_node_walk_stopped_short_is_not_converged_and_warns() -> None:
    law = HeldLaw()
    walk = backward.walk_curve

    def one_step(*args: Any, **kwargs: Any) -> Any:
        kwargs["max_steps"] = 1
        return walk(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(backward, "walk_curve", one_step)
        with pytest.warns(ConvergenceWarning, match="one-step hazard targeting stopped"):
            result = LTMLE(
                STATIC,
                n_folds=1,
                survival_model="hazard",
                targeting="one_step",
                step_size=STEP,
                tol=TOL,
                g_bounds=(1e-8, 1.0),
                outcome_learner=wrong_hazard(),
                pseudo_learner=wrong_continuation(),
                treatment_learner=CellProbabilities(),
                censoring_learner=CellMeans(),
            ).fit(law.frame(), **law.columns())
    assert not result.fits["never @ t=4"].hazard.converged


def test_one_stalled_node_walk_is_read_on_its_own_scale() -> None:
    """One node walk stopped short: the record is not converged and its relative score is the
    node's own, not the share-weighted sum over every node, which divides the stalled node's
    score by the other nodes' scales.  The score-equation row reads the same and is unsolved."""
    from cleverly.fluctuation._score import relative_score
    from cleverly.validation.longitudinal import _hazard_score_row

    law = HeldLaw()
    walk = backward.walk_curve
    calls: list[int] = []

    def last_node_stalls(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        if len(calls) % law.n_times == 1:  # each pass walks the last node first
            kwargs["max_steps"] = 1
        return walk(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(backward, "walk_curve", last_node_stalls)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = LTMLE(
                STATIC,
                n_folds=1,
                survival_model="hazard",
                targeting="one_step",
                step_size=STEP,
                tol=TOL,
                g_bounds=(1e-8, 1.0),
                outcome_learner=wrong_hazard(),
                pseudo_learner=wrong_continuation(),
                treatment_learner=CellProbabilities(),
                censoring_learner=CellMeans(),
            ).fit(law.frame(), **law.columns())
    fit_record = result.fits[f"never @ t={law.n_times}"]
    record = fit_record.hazard
    stalled = record.nodes[-1]
    assert not stalled.converged and all(node.converged for node in record.nodes[:-1])
    assert not record.converged
    diluted = relative_score(record.score, record.score_scale)
    assert record.relative_score == max(node.relative_score for node in record.nodes)
    assert record.relative_score == stalled.relative_score > TOL
    assert record.relative_score > diluted
    row = _hazard_score_row(result, fit_record, tolerance=TOL)
    assert row.relative_score > TOL
    assert row.relative_score > diluted
    assert not row.converged and not row.passed


# ---------------------------------------------------------------- continuous L


def _gaussian_frame(n: int, seed: int) -> pd.DataFrame:
    """Law (H) with ``L_k`` Gaussian AR(1) given ``(W1, A, L_{k-1})``, ``T = 3``."""
    rng = np.random.default_rng(seed)
    w1 = (rng.random(n) < 0.4).astype(float)
    w2 = (rng.random(n) < np.where(w1 == 1, 0.55, 0.3)).astype(float)
    a = (rng.random(n) < 0.3 + 0.2 * w1 + 0.25 * w2).astype(float)
    out: dict[str, np.ndarray] = {"W1": w1, "W2": w2, "A": a}
    alive = np.ones(n, dtype=bool)
    l = np.zeros(n)
    for k in range(1, 4):
        if k > 1:
            l = np.where(alive, 0.5 * l + 0.6 * w1 - 0.5 * a + rng.normal(size=n), np.nan)
            out[f"L{k}"] = l
            l = np.nan_to_num(l)
        c = rng.random(n) < expit(1.8 - 0.4 * w2 + 0.3 * a - 0.8 * l)
        out[f"C{k}"] = np.where(alive, c, np.nan)
        alive = alive & c
        y = rng.random(n) < expit(-1.6 + 0.2 * k - 0.6 * a + 0.5 * w1 + 0.9 * l)
        out[f"Y{k}"] = np.where(alive, y, np.nan)
        alive = alive & ~y
    return pd.DataFrame(out)


def _gaussian_truth(a: int, t: int, nodes: int = 40) -> float:
    """Nested Gauss-Hermite quadrature of the risk at horizon ``t`` under arm ``a``."""
    x, wq = np.polynomial.hermite_e.hermegauss(nodes)
    wq = wq / wq.sum()

    def q(k: int, w1: float, l: float) -> float:
        lam = expit(-1.6 + 0.2 * k - 0.6 * a + 0.5 * w1 + 0.9 * l)
        if k == t:
            return float(lam)
        mean = 0.5 * l + 0.6 * w1 - 0.5 * a
        later = sum(weight * q(k + 1, w1, mean + node) for node, weight in zip(x, wq, strict=True))
        return float(lam + (1.0 - lam) * later)

    psi = 0.0
    for w1, p1 in ((0.0, 0.6), (1.0, 0.4)):
        psi += p1 * q(1, w1, 0.0)
    return psi


class ClippedLinear(LinearRegression):  # type: ignore[misc]
    """``LinearRegression`` with its predictions clipped into ``[0, 1]``."""

    def predict(self, X: Any) -> np.ndarray:
        return np.clip(super().predict(X), 0.0, 1.0)


def test_a_continuous_covariate_after_baseline_covers_the_truth() -> None:
    """Gaussian ``L_k``: logistic mechanism (correct), wrong hazard and continuation, ``T = 3``."""
    frame = _gaussian_frame(20_000, 7)
    for t in (2, 3):
        assert abs(_gaussian_truth(1, t, 40) - _gaussian_truth(1, t, 60)) < 1e-10
    columns = {
        "treatment": "A",
        "baseline": ["W1", "W2"],
        "censoring": ["C1", "C2", "C3"],
        "time_varying": [[], ["L2"], ["L3"]],
        "outcome": ["Y1", "Y2", "Y3"],
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = LTMLE(
            STATIC,
            n_folds=1,
            survival_model="hazard",
            targeting="one_step",
            step_size=STEP,
            tol=TOL,
            outcome_learner=wrong_hazard(),
            pseudo_learner=ClippedLinear(),
            treatment_learner=Logistic(),
            censoring_learner=Logistic(),
        ).fit(frame, **columns)
    for label, level in STATIC.items():
        for t in (1, 2, 3):
            value = result.estimates[f"risk_regimen[{label} @ t={t}]"]
            assert abs(value.psi - _gaussian_truth(level, t)) < 4.0 * value.std_error
        record = result.fits[f"{label} @ t=3"].hazard
        assert all(node.relative_score <= TOL for node in record.nodes)


# ------------------------------------------------------------------ refusals


def _refuse() -> dict[str, Any]:
    class Refuse(CellMeans):
        def fit(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("a learner ran")

    return {
        "outcome_learner": Refuse(),
        "treatment_learner": Refuse(),
        "censoring_learner": Refuse(),
        "n_folds": 1,
    }


def test_h28_the_per_horizon_algorithm_is_refused() -> None:
    law = HeldLaw()
    message = (
        "survival_model='hazard' with covariates measured after baseline (L2, L3, L4) needs "
        "targeting='one_step'. The fit targets each node's hazards and the regression of its "
        "future once, after the later nodes are final, over every horizon and cause together. "
        "The per-horizon algorithm restarts from the initial fits at each horizon, and each "
        "restart would refit every regression of the future, whose outcome is built from the "
        "targeted later hazards. Set targeting='one_step', or use "
        "survival_model='iterated_mean'."
    )
    with pytest.raises(MethodConfigurationError, match=_escape(message)) as raised:
        LTMLE(STATIC, survival_model="hazard", **_refuse()).fit(law.frame(), **law.columns())
    assert raised.value.refusal.reason is RefusalReason.API_REDIRECTION


def _escape(text: str) -> str:
    import re

    return re.escape(text)


def _msm() -> MSM:
    def design(label: object, horizon: int, frame: object) -> np.ndarray:
        return np.ones((len(frame), 1))  # type: ignore[arg-type]

    return MSM(design=design, terms=("intercept",), design_kind="known")


def test_h29_a_working_model_is_refused() -> None:
    law = HeldLaw()
    message = (
        "msm= with survival_model='hazard' on a design with covariates measured after baseline "
        "is not built. Each regimen is targeted by its own backward pass, and a coefficient "
        "score sums cells of every regimen, so it needs one joint walk per node over every "
        "regimen's panel, which this fit does not run. Fit msm= with "
        "survival_model='iterated_mean'."
    )
    with pytest.raises(CapabilityError, match=_escape(message)) as raised:
        LTMLE(STATIC, survival_model="hazard", targeting="one_step", msm=_msm(), **_refuse()).fit(
            law.frame(), **law.columns()
        )
    assert raised.value.refusal.reason is RefusalReason.API_REDIRECTION
    assert raised.value.refusal.roadmap is None


@pytest.mark.parametrize("rule", ["scaled_var", "sample_size"])
def test_h30_a_declared_stopping_rule_is_refused(rule: str) -> None:
    law = HeldLaw()
    message = (
        f"stopping={rule!r} with survival_model='hazard' on a design with covariates measured "
        "after baseline is not built. The rule sets each target's threshold from the spread of "
        "its whole influence curve, which this fit has only after its backward pass, and each "
        "node walk stops on its own relative score. No split of the threshold over the node "
        "walks is implemented. Use stopping='relative' (the default)."
    )
    with pytest.raises(MethodConfigurationError, match=_escape(message)) as raised:
        LTMLE(
            STATIC, survival_model="hazard", targeting="one_step", stopping=rule, **_refuse()
        ).fit(law.frame(), **law.columns())
    assert raised.value.refusal.reason is RefusalReason.API_REDIRECTION


def test_h31_a_truncation_replay_is_refused() -> None:
    law = HeldLaw()
    result = fit(law, STATIC)
    message = (
        "the replay re-targets retained initial hazards, and this fit's regressions of the "
        "future were fitted on outcomes built from targeted later hazards, which change with "
        "g_bounds, so a replay would refit learners. Refit with each g_bounds= value."
    )
    with pytest.raises(CapabilityError, match=_escape(message)) as raised:
        result.diagnostics.truncation_curve(bounds=[(1e-3, 1.0), (1e-2, 1.0)])
    assert raised.value.refusal.reason is RefusalReason.API_REDIRECTION


def test_h6_a_pseudo_learner_is_used_with_covariates_after_baseline() -> None:
    """With covariates after baseline ``pseudo_learner`` fits the continuation, and is read."""
    law = HeldLaw(n_times=2)
    calls: list[int] = []

    class Counting(CellMeans):
        def fit(self, X: Any, y: Any, sample_weight: Any = None) -> Any:
            calls.append(1)
            return super().fit(X, y, sample_weight)

    fit(law, STATIC, continuation=Counting())
    assert calls


def test_the_pooled_trigger_reads_blocks_after_the_first_node() -> None:
    law = HeldLaw()
    with pytest.raises(CapabilityError, match="hazard_pooling='pooled' fits one regression"):
        LTMLE(
            STATIC,
            survival_model="hazard",
            targeting="one_step",
            hazard_pooling="pooled",
            **_refuse(),
        ).fit(law.frame(), **law.columns())


def test_a_hazard_fit_with_later_covariates_stays_experimental() -> None:
    law = HeldLaw(n_times=2)
    result = fit(law, STATIC)
    statuses = {estimate.support.label for estimate in result.estimates.values()}
    assert statuses == {"experimental"}
