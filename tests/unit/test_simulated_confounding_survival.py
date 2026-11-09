"""The simulated common-cause replay of held survival and competing-risk fits (X91 part C).

The replay couples each replayed censoring and event indicator to the observed one through a
uniform drawn conditional on it.  The checks here are:

- the zero-strength identity: the replayed panel is the observed panel, and its refit is the
  original estimate;
- the support: every replayed panel passes the panel's own sweep;
- the coupling law, exactly: a closed-form chain over observed and replayed states shows the
  replayed censoring follows the censoring model at the perturbed treatment and the replayed
  event follows the shifted event model, whatever the observed path;
- the code's coupling on a stratified uniform grid, against the closed form;
- the wiring, by simulation with the true conditionals supplied as the replay models, against
  deliberate mutations that each break one claim;
- the calibration scale, against an in-test penalized IRLS fit of the same table.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.longitudinal import LTMLE, LongitudinalData
from cleverly.sensitivity import ConfounderStrengthGrid, simulated_confounding
from cleverly.sensitivity import _simulated_confounding_survival as survival
from cleverly.sensitivity._simulated_confounding_longitudinal import (
    _CENSORING_REFUSAL,
    _KNOWN_CENSORING_REFUSAL,
    _SEPARABLE_REFUSAL,
    _SURVIVAL_NOT_HELD_REFUSAL,
    longitudinal_fit_refusal,
)
from cleverly.sensitivity.simulated_confounding import _latent_child_seed

N_TIMES = 3


def _expit(x: Any) -> Any:
    return 1.0 / (1.0 + np.exp(-x))


# ------------------------------------------------------------------ a held panel law


def _censoring(node: int, w1: Any, a: Any) -> Any:
    return _expit(-2.0 + 0.9 * a + 0.4 * w1 + 0.1 * node)


def _hazard(node: int, w2: Any, a: Any) -> Any:
    return _expit(-1.6 + 1.2 * a - 0.5 * w2 + 0.2 * node)


def _first_cause(w1: Any, a: Any) -> Any:
    """P(cause 0 | an event)."""
    return _expit(0.3 + 0.5 * w1 - 0.6 * a)


def _panel(
    n: int,
    seed: int,
    *,
    causes: int = 1,
    censored: bool = True,
    weighted: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """A held-treatment panel: baseline W1, W2, treatment A, then C_k and the event at k."""
    rng = np.random.default_rng(seed)
    w1, w2 = rng.normal(size=n), rng.normal(size=n)
    a = (rng.uniform(size=n) < _expit(0.4 * w1)).astype(float)
    frame: dict[str, Any] = {"W1": w1, "W2": w2, "A": a}
    inside = np.ones(n, dtype=bool)
    labels = ("relapse", "death")[:causes] if causes > 1 else ("Y",)
    for node in range(1, N_TIMES + 1):
        if censored:
            left = inside & (rng.uniform(size=n) < _censoring(node, w1, a))
            frame[f"C{node}"] = np.where(inside, np.where(left, 0.0, 1.0), np.nan)
            inside = inside & ~left
        fired = inside & (rng.uniform(size=n) < _hazard(node, w2, a))
        first = rng.uniform(size=n) < _first_cause(w1, a)
        for index, label in enumerate(labels):
            hit = fired & (first if index == 0 else ~first) if causes > 1 else fired
            frame[f"{label}{node}"] = np.where(inside, hit.astype(float), np.nan)
        inside = inside & ~fired
    if weighted:
        frame["wt"] = rng.uniform(0.5, 1.5, size=n)
    columns: dict[str, Any] = {"treatment": "A", "baseline": ["W1", "W2"]}
    if causes > 1:
        columns["outcome"] = {
            label: [f"{label}{node}" for node in range(1, N_TIMES + 1)] for label in labels
        }
    else:
        columns["outcome"] = [f"Y{node}" for node in range(1, N_TIMES + 1)]
    if censored:
        columns["censoring"] = [f"C{node}" for node in range(1, N_TIMES + 1)]
    if weighted:
        columns["weights"] = "wt"
    return pd.DataFrame(frame), columns


def _data(n: int = 600, seed: int = 1, **options: Any) -> LongitudinalData:
    frame, columns = _panel(n, seed, **options)
    return LongitudinalData.from_frame(frame, **columns)


def _same(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return bool(np.array_equal(np.asarray(left), np.asarray(right), equal_nan=True))


PANELS: dict[str, dict[str, Any]] = {
    "survival": {},
    "competing": {"causes": 2},
    "uncensored": {"censored": False},
    "weighted": {"weighted": True},
}


# ------------------------------------------------------------------ the identity


def _replayed(
    data: LongitudinalData,
    *,
    treatment: Any = None,
    strength: float = 0.0,
    cause: str | None = None,
    seed: int = 5,
    models: Any = None,
) -> tuple[LongitudinalData, survival.Replay]:
    observed = survival.observed_followup(data)
    fitted = models if models is not None else survival.fit_replay_models(data, observed, cause)
    uniforms = survival.coupling_uniforms(data.n, data.n_times, seed)
    latent = np.random.default_rng(seed + 1).normal(size=data.n)
    a = observed.treatment if treatment is None else treatment
    replay = survival.replay_followup(data, observed, fitted, uniforms, latent, a, strength)
    return data.with_replayed_followup(a, replay.censoring, replay.event), replay


@pytest.mark.parametrize("panel", list(PANELS))
def test_a_zero_strength_replay_is_the_observed_panel(panel: str) -> None:
    data = _data(**PANELS[panel])
    replayed, replay = _replayed(data)
    for field in ("outcome", "event", "cause_event", "uncensored", "treatment", "weights"):
        assert _same(getattr(replayed, field), getattr(data, field)), field
    assert replay.extended_share == 0.0
    assert replay.new_event_share == 0.0


def test_a_zero_strength_named_cause_replay_is_the_observed_panel() -> None:
    data = _data(causes=2)
    replayed, _ = _replayed(data, cause="death")
    for field in ("event", "cause_event", "uncensored", "treatment"):
        assert _same(getattr(replayed, field), getattr(data, field)), field


def test_a_panel_with_censoring_columns_and_no_censoring_keeps_its_encoding() -> None:
    frame, columns = _panel(400, 3, censored=False)
    for node in range(1, N_TIMES + 1):
        frame[f"C{node}"] = np.where(frame[f"Y{node}"].notna(), 1.0, np.nan)
    data = LongitudinalData.from_frame(
        frame, **columns, censoring=[f"C{node}" for node in range(1, N_TIMES + 1)]
    )
    assert data.censoring_names
    assert not survival.observed_followup(data).censored.any()
    replayed, replay = _replayed(data, strength=0.8)
    assert replay.censoring_redrawn_share is None
    # The outcome strength moves the events, and the encoding of ``uncensored`` follows the
    # panel's own: false after an event, as the sweep writes it.
    assert replayed.event is not None and replayed.uncensored is not None
    exited = replayed.event[:, :-1]
    assert not np.any(replayed.uncensored[:, 1:][exited])


def test_a_zero_strength_refit_reproduces_the_estimate() -> None:
    frame, columns = _panel(800, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result: Any = _hazard_ltmle().fit(frame, **columns)
        replayed, _ = _replayed(result.data)
        refitted = result.estimator.refit(replayed, random_state=3)
    for name, parameter in result.estimates.items():
        assert refitted[name].psi == parameter.psi, name


def _hazard_ltmle() -> LTMLE:
    learner = LogisticRegression(max_iter=1000)
    return LTMLE(
        {"a1": 1, "a0": 0},
        n_folds=1,
        random_state=3,
        survival_model="hazard",
        outcome_learner=learner,
        treatment_learner=learner,
        censoring_learner=learner,
    )


# ------------------------------------------------------------------ the support


@pytest.mark.parametrize("panel", list(PANELS))
def test_every_replayed_panel_passes_the_panel_sweep(panel: str) -> None:
    data = _data(**PANELS[panel])
    rng = np.random.default_rng(11)
    observed = survival.observed_followup(data)
    for _ in range(12):
        flip = rng.uniform(size=data.n) < rng.uniform(0.0, 0.5)
        treatment = np.where(flip, 1.0 - observed.treatment, observed.treatment)
        strength = float(rng.normal(scale=2.0))
        replayed, _ = _replayed(data, treatment=treatment, strength=strength)
        event = replayed.event
        assert event is not None and replayed.uncensored is not None
        assert np.all(event[:, 1:] >= event[:, :-1])
        assert np.all(replayed.uncensored[:, 1:] <= replayed.uncensored[:, :-1]) or not (
            data.censoring_names
        )
        if replayed.cause_event is not None:
            assert np.array_equal(replayed.cause_event.any(axis=2), event)
            assert np.all(replayed.cause_event.sum(axis=2) <= 1)
        present = np.isfinite(replayed.treatment)
        assert np.all(replayed.treatment[present] == np.repeat(treatment[:, None], 3, 1)[present])


# ------------------------------------------------------------------ the coupling law, exactly


def _transition(b: int, p: float, q: float) -> float:
    """P(B' = 1 | B = b) under the coupling with fitted p and replay probability q."""
    if b == 1:
        return min(p, q) / p
    return max(0.0, q - p) / (1.0 - p)


def _chain(
    c_observed: list[float],
    c_replay: list[float],
    h_observed: list[float],
    h_replay: list[float],
    changed: bool,
) -> tuple[list[float], list[float]]:
    """The replayed censoring and event law, by an exact chain over joint states.

    The observed process draws censoring with ``c_observed`` and events with
    ``h_observed``.  The replay couples to it while both are in the study and draws
    afresh once the observed unit has left.  Returns, per node, P(C' = 1 | replay present)
    and P(E' = 1 | replay at risk).
    """
    joint = {("in", "in"): 1.0}
    censoring_law, event_law = [], []
    for node in range(len(c_observed)):
        c, cr = c_observed[node], c_replay[node]
        after: dict[tuple[str, str], float] = {}
        censored_mass, present_mass = 0.0, 0.0
        for (obs, rep), mass in joint.items():
            if rep == "out":
                after[(obs, rep)] = after.get((obs, rep), 0.0) + mass
                continue
            present_mass += mass
            if obs == "in":
                for b, pb in ((1, c), (0, 1.0 - c)):
                    moved = (1.0 if b else 0.0) if not changed else _transition(b, c, cr)
                    for b_rep, pr in ((1, moved), (0, 1.0 - moved)):
                        state = ("out" if b else "in", "censored" if b_rep else "in")
                        after[state] = after.get(state, 0.0) + mass * pb * pr
                        censored_mass += mass * pb * pr * b_rep
            else:
                for b_rep, pr in ((1, cr), (0, 1.0 - cr)):
                    state = (obs, "censored" if b_rep else "in")
                    after[state] = after.get(state, 0.0) + mass * pr
                    censored_mass += mass * pr * b_rep
        censoring_law.append(censored_mass / present_mass)
        joint = {}
        h, hr = h_observed[node], h_replay[node]
        fired_mass, risk_mass = 0.0, 0.0
        for (obs, rep), mass in after.items():
            if rep in ("out", "censored"):
                joint[(obs, "out")] = joint.get((obs, "out"), 0.0) + mass
                continue
            risk_mass += mass
            if obs == "in":
                for b, pb in ((1, h), (0, 1.0 - h)):
                    moved = _transition(b, h, hr)
                    for b_rep, pr in ((1, moved), (0, 1.0 - moved)):
                        state = ("out" if b else "in", "out" if b_rep else "in")
                        joint[state] = joint.get(state, 0.0) + mass * pb * pr
                        fired_mass += mass * pb * pr * b_rep
            else:
                for b_rep, pr in ((1, hr), (0, 1.0 - hr)):
                    state = (obs, "out" if b_rep else "in")
                    joint[state] = joint.get(state, 0.0) + mass * pr
                    fired_mass += mass * pr * b_rep
        event_law.append(fired_mass / risk_mass)
    return censoring_law, event_law


@pytest.mark.parametrize("changed", [False, True])
@pytest.mark.parametrize("shift", [-1.5, 0.0, 0.7, 2.5])
def test_the_replayed_follow_up_has_the_declared_law_exactly(changed: bool, shift: float) -> None:
    w1, w2 = 0.3, -0.8
    a, a_replay = 1.0, (0.0 if changed else 1.0)
    nodes = range(1, 4)
    c_observed = [float(_censoring(k, w1, a)) for k in nodes]
    c_replay = [float(_censoring(k, w1, a_replay)) for k in nodes]
    h = [float(_hazard(k, w2, a)) for k in nodes]
    h_replay = [float(_expit(np.log(p / (1 - p)) + shift)) for p in h]
    censoring_law, event_law = _chain(c_observed, c_replay, h, h_replay, changed)
    np.testing.assert_allclose(censoring_law, c_replay, rtol=0, atol=1e-12)
    np.testing.assert_allclose(event_law, h_replay, rtol=0, atol=1e-12)


def test_keeping_the_observed_censoring_breaks_the_law() -> None:
    """The deliberate mutation: a flipped unit keeps its observed censoring."""
    w1, w2 = 0.3, -0.8
    nodes = range(1, 4)
    c_observed = [float(_censoring(k, w1, 1.0)) for k in nodes]
    c_replay = [float(_censoring(k, w1, 0.0)) for k in nodes]
    h = [float(_hazard(k, w2, 1.0)) for k in nodes]
    censoring_law, _ = _chain(c_observed, c_replay, h, h, changed=False)
    assert np.max(np.abs(np.asarray(censoring_law) - c_replay)) > 0.05


@pytest.mark.parametrize(
    ("b", "p", "q"), [(1, 0.3, 0.1), (1, 0.3, 0.6), (0, 0.3, 0.1), (0, 0.3, 0.6), (0, 0.05, 0.95)]
)
def test_the_coded_coupling_matches_the_transition(b: int, p: float, q: float) -> None:
    """On stratified uniforms, the code's indicator frequency is the closed form."""
    m = 200_000
    raw = (np.arange(m) + 0.5) / m
    v = survival.conditional_uniform(np.full(m, p), np.full(m, b), raw, np.ones(m, dtype=bool))
    assert abs(float(np.mean(v < q)) - _transition(b, p, q)) <= 2.0 / m
    # Where nothing was observed the uniform is plain.
    plain = survival.conditional_uniform(np.full(m, p), np.full(m, b), raw, np.zeros(m, bool))
    assert abs(float(np.mean(plain < q)) - q) <= 2.0 / m


# ------------------------------------------------------------------ the wiring, by simulation


class _True:
    """The true conditional of the panel law, standing in for a fitted replay model."""

    def __init__(self, kind: str, w1: Any, w2: Any) -> None:
        self.kind, self.w1, self.w2 = kind, w1, w2

    def probability(self, node: int, baseline: Any, treatment: Any) -> Any:
        if self.kind == "censoring":
            return _censoring(node + 1, self.w1, treatment)
        return _hazard(node + 1, self.w2, treatment)

    def probabilities(self, node: int, baseline: Any, treatment: Any) -> Any:
        first = _first_cause(self.w1, treatment)
        return np.column_stack((first, 1.0 - first))

    def describe(self) -> str:
        return self.kind


def _true_models(data: LongitudinalData, causes: int) -> survival.ReplayModels:
    w1, w2 = np.asarray(data.baseline, dtype=float).T
    return survival.ReplayModels(
        baseline=np.asarray(data.baseline, dtype=float),
        event=_True("event", w1, w2),  # type: ignore[arg-type]
        censoring=_True("censoring", w1, w2),  # type: ignore[arg-type]
        shares=None if causes == 1 else _True("shares", w1, w2),  # type: ignore[arg-type]
        other=None,
        cause_index=None,
    )


def _simulated(
    models: Any, *, causes: int = 1, strength: float = 1.2, n: int = 60_000
) -> dict[str, Any]:
    data = _data(n=n, seed=21, causes=causes)
    observed = survival.observed_followup(data)
    latent = np.random.default_rng(22).normal(size=data.n)
    flipped = latent >= 0.5244  # the upper 30% tail
    treatment = np.where(flipped, 1.0 - observed.treatment, observed.treatment)
    uniforms = survival.coupling_uniforms(data.n, data.n_times, 23)
    replay = survival.replay_followup(
        data, observed, models(data), uniforms, latent, treatment, strength
    )
    replayed = data.with_replayed_followup(treatment, replay.censoring, replay.event)
    return {
        "data": data,
        "observed": observed,
        "latent": latent,
        "treatment": treatment,
        "replay": replay,
        "replayed": replayed,
    }


def _censoring_residual_on_latent(run: dict[str, Any]) -> float:
    """The largest |t| of the replayed censoring residual among flipped units, by arm.

    A flipped unit is one whose latent passed the flip threshold, so a residual that
    differs from zero among them is a dependence of the replayed censoring on the latent
    given the replayed past.  The two perturbed arms are read apart, because a flip moves
    the censoring probability in opposite directions in the two.
    """
    w1 = np.asarray(run["data"].baseline, dtype=float)[:, 0]
    raw = run["replay"].censoring
    flipped = run["treatment"] != run["observed"].treatment
    worst = 0.0
    for arm in (0.0, 1.0):
        residuals = []
        for node in range(N_TIMES):
            present = np.isfinite(raw[:, node]) & flipped & (run["treatment"] == arm)
            censored = raw[present, node] == 0.0
            expected = _censoring(node + 1, w1[present], run["treatment"][present])
            residuals.append(censored - expected)
        r = np.concatenate(residuals)
        worst = max(worst, abs(float(np.mean(r)) / float(np.std(r) / np.sqrt(len(r)))))
    return worst


def _event_residual(run: dict[str, Any], strength: float) -> float:
    """The largest |t| of the replayed event residual among flipped units at risk, by arm."""
    w2 = np.asarray(run["data"].baseline, dtype=float)[:, 1]
    raw = run["replay"].event.sum(axis=2)
    flipped = run["treatment"] != run["observed"].treatment
    worst = 0.0
    for arm in (0.0, 1.0):
        residuals = []
        for node in range(N_TIMES):
            at_risk = np.isfinite(raw[:, node]) & flipped & (run["treatment"] == arm)
            h = _hazard(node + 1, w2[at_risk], run["observed"].treatment[at_risk])
            expected = _expit(np.log(h / (1 - h)) + strength * run["latent"][at_risk])
            residuals.append(raw[at_risk, node] - expected)
        r = np.concatenate(residuals)
        worst = max(worst, abs(float(np.mean(r)) / float(np.std(r) / np.sqrt(len(r)))))
    return worst


def test_the_replayed_censoring_is_at_random_given_the_replayed_past() -> None:
    run = _simulated(lambda data: _true_models(data, 1))
    assert abs(_censoring_residual_on_latent(run)) < 4.0


def test_keeping_the_observed_censoring_shows_in_simulation() -> None:
    """Mutation: the censoring model read at the observed treatment keeps the observed C."""

    def mutated(data: LongitudinalData) -> survival.ReplayModels:
        models = _true_models(data, 1)
        observed_a = survival.observed_followup(data).treatment

        class _Observed(_True):
            def probability(self, node: int, baseline: Any, treatment: Any) -> Any:
                return _censoring(node + 1, self.w1, observed_a)

        w1, w2 = np.asarray(data.baseline, dtype=float).T
        return replace(models, censoring=_Observed("censoring", w1, w2))  # type: ignore[arg-type]

    run = _simulated(mutated)
    assert abs(_censoring_residual_on_latent(run)) > 8.0


def test_the_replayed_event_follows_the_shifted_model_at_the_observed_treatment() -> None:
    run = _simulated(lambda data: _true_models(data, 1), strength=1.2)
    assert _event_residual(run, 1.2) < 4.0


def test_reading_the_event_model_at_the_perturbed_treatment_shows_in_simulation() -> None:
    """Mutation: the event model read at the perturbed treatment."""

    def mutated(data: LongitudinalData) -> survival.ReplayModels:
        models = _true_models(data, 1)
        observed = survival.observed_followup(data)
        latent = np.random.default_rng(22).normal(size=data.n)
        perturbed = np.where(latent >= 0.5244, 1.0 - observed.treatment, observed.treatment)

        class _Perturbed(_True):
            def probability(self, node: int, baseline: Any, treatment: Any) -> Any:
                return _hazard(node + 1, self.w2, perturbed)

        w1, w2 = np.asarray(data.baseline, dtype=float).T
        return replace(models, event=_Perturbed("event", w1, w2))  # type: ignore[arg-type]

    run = _simulated(mutated, strength=1.2)
    assert _event_residual(run, 1.2) > 8.0


def test_a_new_event_draws_its_cause_from_the_shares() -> None:
    run = _simulated(lambda data: _true_models(data, 2), causes=2, strength=1.5)
    w1 = np.asarray(run["data"].baseline, dtype=float)[:, 0]
    raw = run["replay"].event
    observed = run["observed"]
    hits, expected = [], []
    for node in range(N_TIMES):
        fired = np.nansum(raw[:, node, :], axis=1) == 1.0
        new = fired & ~(observed.at_risk[:, node] & observed.event[:, node])
        hits.append(raw[new, node, 0])
        expected.append(_first_cause(w1[new], observed.treatment[new]))
    h, e = np.concatenate(hits), np.concatenate(expected)
    assert len(h) > 2000
    assert abs(float(np.mean(h - e))) < 4.0 * float(np.std(h - e) / np.sqrt(len(h)))


# ------------------------------------------------------------------ a named cause


def test_a_named_cause_moves_only_its_own_hazard() -> None:
    data = _data(n=20_000, seed=31, causes=2)
    observed = survival.observed_followup(data)
    j = list(data.cause_labels).index("death")
    rates = {}
    for strength in (-2.0, 0.0, 2.0):
        replayed, _ = _replayed(data, strength=strength, cause="death")
        events = replayed.cause_event
        assert events is not None
        first = events.copy()
        first[:, 1:] &= ~events[:, :-1]
        other, named = 0.0, 0.0
        for node in range(N_TIMES):
            at_risk = np.isfinite(replayed.treatment[:, node]) & replayed.uncensored[:, node]
            other += float(np.sum(first[at_risk, node, 1 - j]))
            named += float(np.sum(first[at_risk, node, j]))
        rates[strength] = (other, named)
    assert observed.event.any()
    # The named cause moves with the strength; the other cause's count moves only through
    # the risk set, which the named cause shrinks or grows, and far less.
    assert rates[2.0][1] != rates[0.0][1] and rates[-2.0][1] != rates[0.0][1]
    named_move = abs(rates[2.0][1] - rates[0.0][1])
    other_move = abs(rates[2.0][0] - rates[0.0][0])
    assert other_move < 0.25 * named_move


# ------------------------------------------------------------------ calibration


def _penalized_irls(x: Any, y: Any, w: Any, c: float) -> Any:
    """Weighted logistic regression with an L2 penalty 1/(2c) on the slopes, by Newton."""
    design = np.column_stack((np.ones(len(y)), x))
    beta = np.zeros(design.shape[1])
    penalty = np.full(design.shape[1], 1.0 / c)
    penalty[0] = 0.0
    for _ in range(100):
        p = _expit(design @ beta)
        gradient = design.T @ (w * (y - p)) - penalty * beta
        hessian = (design * (w * p * (1 - p))[:, None]).T @ design + np.diag(penalty)
        step = np.linalg.solve(hessian, gradient)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-12:
            break
    return beta


def test_the_benchmark_is_the_event_model_coefficient() -> None:
    data = _data(n=3000, seed=41, weighted=True)
    observed = survival.observed_followup(data)
    models = survival.fit_replay_models(data, observed, None)
    strength, method = survival.calibration_strength(models.event, 1)
    assert method == survival.REPLAY_HAZARD_METHOD
    baseline = survival.standardized_baseline(data)
    unit, node = np.nonzero(observed.at_risk)
    x = survival.person_node_design(baseline[unit], observed.treatment[unit], node, data.n_times)
    y = observed.event[unit, node].astype(float)
    w = np.asarray(data.weights, dtype=float)[unit]
    beta = _penalized_irls(x, y, w, survival.PENALTY_C)
    assert strength == pytest.approx(beta[1 + 1], abs=1e-6)
    assert strength < 0.0  # W2 lowers the hazard in the law


def test_a_surface_reads_the_benchmark_on_the_outcome_strength_scale() -> None:
    frame, columns = _panel(800, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result: Any = _hazard_ltmle().fit(frame, **columns)
        surface = simulated_confounding(
            result,
            "risk_regimen[a1 @ t=3]",
            grid=ConfounderStrengthGrid((0.0,), (0.0,)),
            benchmark_covariates=("W2",),
        )
    outcome = [row for row in surface.calibrations if row.role == "outcome"]
    assert len(outcome) == 1 and outcome[0].method == survival.REPLAY_HAZARD_METHOD


# ------------------------------------------------------------------ the surface


def test_a_survival_surface_runs_and_reports_its_replay() -> None:
    frame, columns = _panel(800, 4, causes=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result: Any = _hazard_ltmle().fit(frame, **columns)
        name = "cif_regimen[a1, death @ t=3]"
        grid = ConfounderStrengthGrid((0.0, 0.2), (0.0, 1.0))
        surface = simulated_confounding(result, name, grid=grid, random_state=3)
        again = simulated_confounding(result, name, grid=grid, random_state=3)
    assert surface.complete
    assert surface.cells[0].estimate == result[name].psi
    assert [cell.estimate for cell in surface.cells] == [cell.estimate for cell in again.cells]
    assert surface.coupling_seed == survival.coupling_seed(3)
    assert surface.replay_models and surface.outcome_cause is None
    moved = surface.cells[3]
    assert moved.extended_share is not None and moved.extended_share > 0.0
    assert moved.censoring_redrawn_share is not None and moved.censoring_redrawn_share > 0.0
    assert "fitted shares" in surface.outcome_law
    frame_out = surface.to_frame()
    assert {"censoring_redrawn_share", "extended_share", "new_event_share"} <= set(
        frame_out.columns
    )


def test_outcome_cause_is_checked_before_work() -> None:
    frame, columns = _panel(400, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result: Any = _hazard_ltmle().fit(frame, **columns)
    grid = ConfounderStrengthGrid((0.0,), (0.0, 1.0))
    with pytest.raises(ValueError, match="this fit has one outcome"):
        simulated_confounding(result, "risk_regimen[a1 @ t=3]", grid=grid, outcome_cause="Y")
    frame, columns = _panel(400, 4, causes=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        competing: Any = _hazard_ltmle().fit(frame, **columns)
    with pytest.raises(ValueError, match="is not a cause of this fit"):
        simulated_confounding(
            competing, "cif_regimen[a1, death @ t=3]", grid=grid, outcome_cause="stroke"
        )


def test_a_signed_outcome_strength_is_accepted_on_a_survival_fit() -> None:
    frame, columns = _panel(400, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result: Any = _hazard_ltmle().fit(frame, **columns)
        surface = simulated_confounding(
            result,
            "risk_regimen[a1 @ t=3]",
            grid=ConfounderStrengthGrid((0.0,), (0.0, -2.0, 3.0)),
        )
    assert surface.complete


# ------------------------------------------------------------------ refusals and seeds


def test_a_survival_fit_that_is_not_held_is_refused() -> None:
    frame, columns = _panel(400, 4)
    frame = frame.assign(A1=frame["A"], A2=frame["A"], A3=frame["A"])
    for node in (2, 3):
        frame.loc[frame[f"C{node}"].isna(), f"A{node}"] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result: Any = LTMLE(
            {"a1": [1, 1, 1], "a0": [0, 0, 0]},
            n_folds=1,
            random_state=3,
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            censoring_learner=LogisticRegression(max_iter=1000),
        ).fit(frame, **{**columns, "treatment": ["A1", "A2", "A3"]})
    assert longitudinal_fit_refusal(result) == _SURVIVAL_NOT_HELD_REFUSAL


def test_the_refusal_texts_name_their_own_argument() -> None:
    assert "held over every node" in _SURVIVAL_NOT_HELD_REFUSAL.condition
    assert "censored end-of-study fit" in _CENSORING_REFUSAL.condition
    assert "dismissible-component" in _SEPARABLE_REFUSAL.condition
    assert _SEPARABLE_REFUSAL.roadmap == "X91"
    assert "declared censoring mechanism" in _KNOWN_CENSORING_REFUSAL.condition


def test_the_coupling_stream_aliases_no_other_child_stream() -> None:
    from cleverly.validation.refute import _CHILD_SEED_TAGS, _generated_child_seeds

    for root in (0, 7, 19, 41, 2**31 - 1):
        coupling = survival.coupling_seed(root)
        assert coupling != _latent_child_seed(root)
        assert coupling != root
        for name in _CHILD_SEED_TAGS:
            assert coupling not in _generated_child_seeds(root, name, 4)
    assert survival.COUPLING_SEED_TAG not in (1, 2, 3, 4)
