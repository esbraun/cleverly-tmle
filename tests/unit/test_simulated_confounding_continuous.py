"""The simulated common cause on a continuous-time survival fit (X101 PR 3, checks T14a to T14i).

The replay redraws each unit's event and censoring times from the replay's grouped
proportional-hazards laws, through latent levels drawn given the observed outcome. These checks
pin the zero-cell identity, the law of the replayed data under known models (continuous and
discrete, with a cause atom shared by both causes), the sign, a correct-model control, the
calibration, the refusals, the seeds and shares, and the plot and summary text. Each component
that vanishes at the truth carries a deliberate mutation that must fail.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import math
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly.exceptions import CapabilityError
from cleverly.sensitivity import ConfounderStrengthGrid, simulated_confounding
from cleverly.sensitivity import _simulated_confounding_continuous as cc
from cleverly.sensitivity._simulated_confounding_survival import coupling_seed
from cleverly.sensitivity.simulated_confounding import (
    _draw_latent,
    _flip_mask,
    _latent_child_seed,
)
from cleverly.survival import ContinuousSurvivalTMLE, CoxHazard
from tests.unit._capability_sweep_support import fit_continuous_survival_restricted
from tests.unit.test_continuous_survival_refute import (
    _cross_fitted,
    _fit,
    _frame,
    _fresh,
    _in_sample,
    _same,
)

RD = "ate_regimen[1 vs 0, 1 @ t=1]"
SWEEP_GRID = ConfounderStrengthGrid((0.0, 0.22), (0.0, 0.17))


# ------------------------------------------------------------------------------ support


@dataclasses.dataclass(frozen=True)
class _Observed:
    """A fit's coded arrays and the replay state its surface would build."""

    result: Any
    time: np.ndarray[Any, Any]
    coded: np.ndarray[Any, Any]
    codes: list[int]
    labels: list[str]
    cause_index: np.ndarray[Any, Any]
    state: Any
    seed: int

    @property
    def n(self) -> int:
        return int(self.time.size)

    def uniforms(self) -> np.ndarray[Any, Any]:
        return np.random.default_rng(self.seed).uniform(size=(len(self.codes) + 3, self.n))

    def latent(self, root: int = 0) -> np.ndarray[Any, Any]:
        return np.asarray(_draw_latent(self.n, _latent_child_seed(root), None, "row", None))

    def replay(
        self,
        *,
        perturbed: Any = None,
        latent: Any = None,
        shifts: Any = None,
        levels: Any = None,
    ) -> cc.Replay:
        J = len(self.codes)
        return cc.replay_cell(
            self.state.models,
            self.state.levels if levels is None else levels,
            self.time,
            self.cause_index,
            self.coded,
            self.coded if perturbed is None else perturbed,
            np.zeros(self.n) if latent is None else latent,
            np.zeros(J) if shifts is None else np.asarray(shifts, dtype=float),
        )


def _observed(result: Any, root: int = 0) -> _Observed:
    coded, codes, labels, _, cause_index = cc._observed_arrays(result)
    seed = coupling_seed(root)
    state = cc._state(result, coded, codes, labels, cause_index, seed)
    time = np.asarray(result.data.time, dtype=float)
    return _Observed(result, time, coded, codes, labels, cause_index, state, seed)


def _reproduces(observed: _Observed, replay: cc.Replay) -> bool:
    return bool(
        np.array_equal(replay.time, observed.time)
        and np.array_equal(replay.cause_index, observed.cause_index)
    )


def _quiet(call: Callable[[], Any]) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return call()


def _surface(result: Any, estimand: str = RD, grid: Any = SWEEP_GRID, **kwargs: Any) -> Any:
    kwargs.setdefault("random_state", 0)
    return _quiet(lambda: simulated_confounding(result, estimand, grid=grid, **kwargs))


def _one_cause_frame() -> pd.DataFrame:
    frame = _frame()
    return frame.assign(event=(frame.event > 0).astype(int))


def _uncensored_frame() -> pd.DataFrame:
    frame = _frame()
    return frame.assign(event=np.where(frame.event == 0, 1, frame.event))


def _rounded_frame() -> pd.DataFrame:
    """Times rounded to 0.1: ties between events, and between events and censoring."""
    frame = _frame()
    return frame.assign(time=np.maximum(np.round(frame.time, 1), 0.1))


@functools.cache
def _fits() -> dict[str, Any]:
    return {
        "sweep": _in_sample(),
        "restricted": _quiet(fit_continuous_survival_restricted),
        "one_cause": _fit(_one_cause_frame()),
        "uncensored": _fit(_uncensored_frame()),
        "rounded": _fit(_rounded_frame()),
    }


@functools.cache
def _sweep() -> _Observed:
    return _observed(_in_sample())


# ------------------------------------------------------------------ T14a zero-cell identity


@pytest.mark.parametrize("name", ["sweep", "restricted", "one_cause", "uncensored", "rounded"])
def test_t14a_the_zero_replay_is_the_observed_data(name: str) -> None:
    """The general path at zero strength returns ``time``, ``event`` and ``treatment``."""
    observed = _observed(_fits()[name])
    assert observed.state.error is None, observed.state.error
    zero = observed.replay()
    assert _reproduces(observed, zero)
    if name == "uncensored":
        assert observed.state.models.censoring is None
    if name == "one_cause":
        assert len(observed.codes) == 1
    if name == "rounded":
        # The rounded fit holds both kinds of tie the identity must survive.
        frame = _rounded_frame()
        event_times = frame.time[frame.event > 0]
        assert event_times.duplicated().any()
        assert np.isin(frame.time[frame.event == 0], event_times).any()


def test_t14a_a_named_cause_zero_replay_is_the_observed_data() -> None:
    observed = _sweep()
    shifts = np.array([0.0, 0.0])
    assert _reproduces(observed, observed.replay(shifts=shifts, latent=observed.latent()))


@pytest.mark.parametrize("outcome_cause", [None, "2"])
def test_t14a_the_anchor_cell_is_the_original_estimate(outcome_cause: Any) -> None:
    result = _fresh(_in_sample())
    surface = _surface(result, outcome_cause=outcome_cause)
    anchor = surface.cells[0]
    assert (anchor.treatment_strength, anchor.outcome_strength) == (0.0, 0.0)
    assert anchor.estimate == result[RD].psi and anchor.displacement == 0.0
    assert surface.complete and surface.outcome_cause == outcome_cause


@pytest.mark.parametrize("fit", [_in_sample, _cross_fitted], ids=["in_sample", "cross_fitted"])
def test_t14a_a_refit_under_the_fold_plan_is_the_fit(fit: Callable[[], Any]) -> None:
    result = fit()
    estimator = result.estimator
    again = _quiet(
        lambda: estimator.refit(
            result.data, random_state=estimator.random_state, _frozen_plan=estimator._fold_plan
        )
    )
    assert _same(again, result)


def test_t14a_a_plan_without_its_generator_state_fails_the_refit() -> None:
    """Mutation: the plan's folds alone do not fix the off-grid draw."""
    result = _in_sample()
    estimator = result.estimator
    other = dataclasses.replace(
        estimator._fold_plan, rng_state=np.random.default_rng(1).bit_generator.state
    )
    again = _quiet(
        lambda: estimator.refit(
            result.data, random_state=estimator.random_state, _frozen_plan=other
        )
    )
    assert not _same(again, result)
    assert again.record.off_grid_share != result.record.off_grid_share


def _treatment_only(observed: _Observed, levels: Any = None) -> tuple[Any, ...]:
    latent = observed.latent()
    mask = _flip_mask(latent, 0.22)
    perturbed = np.where(mask, 1 - observed.coded, observed.coded)
    replay = observed.replay(perturbed=perturbed, latent=latent, levels=levels)
    return mask, replay


def test_t14a_a_treatment_only_cell_moves_only_flipped_units() -> None:
    observed = _sweep()
    mask, replay = _treatment_only(observed)
    keep = ~mask
    assert np.array_equal(replay.time[keep], observed.time[keep])
    assert np.array_equal(replay.cause_index[keep], observed.cause_index[keep])
    # A flipped unit's event stays unless its redrawn censoring comes first.
    events = mask & (observed.cause_index >= 0)
    same = (replay.time == observed.time) & (replay.cause_index == observed.cause_index)
    earlier = (replay.cause_index == -1) & (replay.time < observed.time)
    assert np.all(same[events] | earlier[events])
    # Nonzero witness: the censoring redraw moves some flipped unit.
    assert not np.all(same[mask])


def test_t14a_a_censoring_redraw_of_every_unit_fails_the_witness() -> None:
    """Mutation: censoring drawn afresh for every unit moves units the cell did not flip."""
    observed = _sweep()
    levels = observed.state.levels
    model = observed.state.models.censoring
    raw = observed.uniforms()[len(observed.codes)]
    fresh = -np.log1p(-raw) / np.exp(levels.eta_censoring)
    mask, replay = _treatment_only(observed, dataclasses.replace(levels, censoring=fresh))
    keep = ~mask
    assert model is not None
    assert not (
        np.array_equal(replay.time[keep], observed.time[keep])
        and np.array_equal(replay.cause_index[keep], observed.cause_index[keep])
    )


def test_t14a_a_plain_cause_level_for_a_censored_unit_breaks_the_identity() -> None:
    """Mutation (the plan's control): an event drawn before ``C`` at zero strength."""
    observed = _sweep()
    levels = observed.state.levels
    raw = observed.uniforms()[: len(observed.codes)]
    censored = observed.cause_index < 0
    cause = levels.cause.copy()
    cause[:, censored] = (-np.log1p(-raw) / np.exp(levels.eta))[:, censored]
    zero = observed.replay(levels=dataclasses.replace(levels, cause=cause))
    assert not _reproduces(observed, zero)
    assert np.any((zero.cause_index >= 0) & censored & (zero.time < observed.time))


def test_t14a_a_plain_censoring_level_for_an_event_unit_breaks_the_identity() -> None:
    """Mutation: a plain ``V_c`` for an event unit censors it before ``T``."""
    observed = _sweep()
    levels = observed.state.levels
    raw = observed.uniforms()[len(observed.codes)]
    event = observed.cause_index >= 0
    censoring = levels.censoring.copy()
    censoring[event] = (-np.log1p(-raw) / np.exp(levels.eta_censoring))[event]
    zero = observed.replay(levels=dataclasses.replace(levels, censoring=censoring))
    assert not _reproduces(observed, zero)
    assert np.any((zero.cause_index == -1) & event & (zero.time < observed.time))


# ------------------------------------------------------------------ T14b known hazards


class _Constant:
    """A constant baseline hazard ``rate`` with log-linear covariates: a continuous law."""

    atomic = False

    def __init__(self, rate: float, beta: Any) -> None:
        self.rate = float(rate)
        self.beta = np.asarray(beta, dtype=float)

    def eta(self, X: Any) -> np.ndarray[Any, Any]:
        return np.asarray(X, dtype=float) @ self.beta

    def cumulative(self, t: Any) -> np.ndarray[Any, Any]:
        return self.rate * np.asarray(t, dtype=float)

    left = cumulative

    def jump(self, t: Any) -> np.ndarray[Any, Any]:
        return np.zeros_like(np.asarray(t, dtype=float))

    def inverse(self, level: Any) -> np.ndarray[Any, Any]:
        return np.asarray(level, dtype=float) / self.rate


def _constant_law(
    n: int, seed: int, *, censored: bool = False, arm: float = 1.0, rate: float = 1.0
) -> dict[str, Any]:
    """Two causes with constant proportional hazards, and optionally censoring on ``A``.

    ``arm`` scales every treatment coefficient and ``rate`` both cause rates.
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    a = rng.binomial(1, 1 / (1 + np.exp(-0.3 * w))).astype(float)
    X = np.column_stack([a, w])
    causes = (
        _Constant(0.3 * rate, [-1.0 * arm, 0.5]),
        _Constant(0.15 * rate, [0.5 * arm, -0.5]),
    )
    rates = np.stack([model.rate * np.exp(model.eta(X)) for model in causes])
    t = rng.exponential(1 / rates.sum(axis=0))
    cause = np.where(rng.uniform(size=n) < rates[0] / rates.sum(axis=0), 0, 1)
    censoring = _Constant(0.2, [1.0 * arm, 0.2]) if censored else None
    time, cause_index = t, cause
    if censoring is not None:
        c = rng.exponential(1 / (censoring.rate * np.exp(censoring.eta(X))))
        time = np.minimum(t, c)
        cause_index = np.where(t <= c, cause, -1)
    models = cc.ReplayModels(causes, censoring, X, X[:, 1:])
    return {"time": time, "cause_index": cause_index, "X": X, "a": a, "w": w, "models": models}


def _levels(law: dict[str, Any], seed: int = 11) -> cc.Levels:
    n = law["time"].size
    uniforms = np.random.default_rng(seed).uniform(size=(len(law["models"].causes) + 3, n))
    return cc.observed_levels(law["models"], law["time"], law["cause_index"], uniforms)


@pytest.mark.parametrize("gamma", [-0.8, -0.3, 0.4, 1.1])
def test_t14b_known_hazards_scale_every_event_time(gamma: float) -> None:
    law = _constant_law(400, 2)
    levels = _levels(law)
    time, cause_index, a = law["time"], law["cause_index"], law["a"]
    latent = np.random.default_rng(3).normal(size=time.size)
    shifts = np.array([gamma, gamma])
    replay = cc.replay_cell(law["models"], levels, time, cause_index, a, a, latent, shifts)
    tau = float(time.max())
    scaled = time * np.exp(-gamma * latent)
    np.testing.assert_allclose(replay.time, np.minimum(scaled, tau), rtol=1e-12, atol=0.0)
    inside = scaled <= tau
    assert np.array_equal(replay.cause_index[inside], cause_index[inside])
    assert np.all(replay.cause_index[~inside] == -1)
    # Mutation: the sign flip of the shift.
    flipped = cc.replay_cell(law["models"], levels, time, cause_index, a, a, latent, -shifts)
    assert not np.allclose(flipped.time, np.minimum(scaled, tau), rtol=1e-6, atol=0.0)


def test_t14b_a_named_cause_moves_only_its_own_latent_time() -> None:
    law = _constant_law(400, 2)
    levels = _levels(law)
    a = law["a"]
    latent = np.random.default_rng(3).normal(size=a.size)
    zero, _ = cc.latent_draws(law["models"], levels, a, a, latent, np.zeros(2))
    named, _ = cc.latent_draws(law["models"], levels, a, a, latent, np.array([0.7, 0.0]))
    np.testing.assert_allclose(named[0], zero[0] * np.exp(-0.7 * latent), rtol=1e-12, atol=0.0)
    assert np.array_equal(named[1], zero[1])
    # Mutation: a named-cause law that shifts every cause moves the other cause.
    every, _ = cc.latent_draws(law["models"], levels, a, a, latent, np.array([0.7, 0.7]))
    assert not np.array_equal(every[1], zero[1])


def test_t14b_the_model_seam_runs_a_surface_on_known_hazards() -> None:
    """The private ``_replay_models`` seam replays a fit through supplied models."""
    law = _constant_law(300, 4)
    frame = pd.DataFrame(
        {
            "time": law["time"],
            "event": law["cause_index"] + 1,
            "A": law["a"].astype(int),
            "W": law["w"],
        }
    )
    result = _fit(frame)
    grid = ConfounderStrengthGrid((0.0, 0.2), (0.0, 0.5))
    request = cc.validate_continuous_request(result, RD, grid, ())
    surface = _quiet(
        lambda: cc.continuous_surface(result, RD, grid, request, 0, _replay_models=law["models"])
    )
    assert surface.complete
    assert surface.cells[0].estimate == result[RD].psi
    assert surface.cells[-1].estimate != result[RD].psi


# ------------------------------------------------------------------------ T14c sign


def _event_correlation(gamma: float) -> float:
    observed = _sweep()
    latent = observed.latent()
    replay = observed.replay(latent=latent, shifts=[gamma, gamma])
    by_last = (replay.cause_index >= 0) & (replay.time <= 2.0)
    return float(np.corrcoef(latent, by_last.astype(float))[0, 1])


def test_t14c_the_latent_raises_the_event_risk_with_the_sign_of_the_strength() -> None:
    """At treatment strength 0, the correlation of ``U`` with an event by ``t = 2`` has the
    sign of ``gamma``. The margin 0.1 sits well inside the values at the fixed seed (about
    +0.14 and -0.31); the sign-flip mutation is the ``-gamma`` reading, which reverses it."""
    up, down = _event_correlation(0.5), _event_correlation(-0.5)
    assert up > 0.1
    assert down < -0.1


# ------------------------------------------------------------------ T14d exact discrete law


class _Step:
    """A grouped proportional-hazards law: baseline jumps at ``atoms``, log-linear covariates."""

    atomic = True

    def __init__(self, atoms: Any, jumps: Any, beta: Any) -> None:
        self.atoms = np.asarray(atoms, dtype=float)
        self.jumps = np.asarray(jumps, dtype=float)
        self.beta = np.asarray(beta, dtype=float)
        self._cum = np.concatenate([[0.0], np.cumsum(self.jumps)])

    def eta(self, X: Any) -> np.ndarray[Any, Any]:
        return np.asarray(X, dtype=float) @ self.beta

    def cumulative(self, t: Any) -> np.ndarray[Any, Any]:
        return self._cum[np.searchsorted(self.atoms, t, side="right")]

    def left(self, t: Any) -> np.ndarray[Any, Any]:
        return self._cum[np.searchsorted(self.atoms, t, side="left")]

    def jump(self, t: Any) -> np.ndarray[Any, Any]:
        t = np.asarray(t, dtype=float)
        index = np.minimum(np.searchsorted(self.atoms, t), self.atoms.size - 1)
        return np.where(self.atoms[index] == t, self.jumps[index], 0.0)

    def inverse(self, level: Any) -> np.ndarray[Any, Any]:
        index = np.searchsorted(self._cum[1:], level, side="right")
        return np.where(
            index < self.atoms.size, self.atoms[np.minimum(index, self.atoms.size - 1)], np.inf
        )


# Two causes with three atoms each, sharing the atom 2; censoring with atoms 1.5 and 3, the
# second shared with cause 1, and a hazard that depends on A. The design is [A, L].
STEP_CAUSES = (
    _Step([1.0, 2.0, 3.0], [0.15, 0.25, 0.2], [0.4, -0.3]),
    _Step([2.0, 4.0, 5.0], [0.3, 0.1, 0.2], [-0.5, 0.6]),
)
STEP_CENSORING = _Step([1.5, 3.0], [0.2, 0.3], [0.8, 0.2])
TAU = 10.0


def _event_law(x: Any, shifts: Any) -> tuple[dict[tuple[float, int], float], float]:
    """``P(T^e = u, D = k)`` of the grouped law with jumps ``e^{eta_k + shift_k} dLambda_k``,
    a tie going to each tied cause in proportion to its jump, and ``P(no event)``."""
    J = len(STEP_CAUSES)
    atoms = sorted(set(itertools.chain.from_iterable(m.atoms for m in STEP_CAUSES)))
    law: dict[tuple[float, int], float] = {}
    alive = 1.0
    for u in atoms:
        h = [
            float(math.exp(float(m.eta(x)) + shifts[i]) * m.jump(np.array([u]))[0])
            for i, m in enumerate(STEP_CAUSES)
        ]
        p = [-math.expm1(-value) for value in h]
        for members in itertools.product([0, 1], repeat=J):
            chosen = [i for i in range(J) if members[i]]
            if not chosen or any(h[i] == 0.0 for i in chosen):
                continue
            weight = math.prod(p[i] if members[i] else 1.0 - p[i] for i in range(J))
            total = sum(h[i] for i in chosen)
            for k in chosen:
                law[u, k] = law.get((u, k), 0.0) + alive * weight * h[k] / total
        alive *= math.prod(1.0 - value for value in p)
    return law, alive


def _censoring_law(x: Any) -> tuple[dict[float, float], float]:
    law: dict[float, float] = {}
    alive = 1.0
    scale = math.exp(float(STEP_CENSORING.eta(x)))
    for atom, jump in zip(STEP_CENSORING.atoms, STEP_CENSORING.jumps, strict=True):
        p = -math.expm1(-scale * jump)
        law[float(atom)] = alive * p
        alive *= 1.0 - p
    return law, alive


def _combined(event: Any, censoring: Any) -> dict[tuple[float, int], float]:
    """``P(T = t, D = k)`` with the event first at a shared time and the cap at ``TAU``."""
    (ev, ev_none), (ce, ce_none) = event, censoring
    law: dict[tuple[float, int], float] = {}
    for (u, k), p in ev.items():
        law[u, k] = p * (ce_none + sum(q for c, q in ce.items() if c >= u))
    for c, q in ce.items():
        law[c, -1] = q * (ev_none + sum(p for (u, _), p in ev.items() if u > c))
    law[TAU, -1] = ev_none * ce_none
    return law


def _levels_at(model: _Step, u: float, eta: float, thresholds: list[float]) -> list[float]:
    """Every uniform at which a level drawn at ``u`` (on the jump, past ``u`` or past ``u-``)
    crosses one of ``thresholds``: the breakpoints of the coded inversion."""
    scale = math.exp(eta)
    left = float(model.left(np.array([u]))[0])
    top = float(model.cumulative(np.array([u]))[0])
    h = scale * float(model.jump(np.array([u]))[0])
    points = []
    for level in thresholds:
        if h > 0.0 and left < level < top:
            points.append(math.expm1(-(level - left) * scale) / math.expm1(-h))
        for start in (left, top):
            if level > start:
                points.append(-math.expm1(-(level - start) * scale))
    return points


def _cells(breakpoints: list[list[float]]) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Midpoints and volumes of the product grid cut at ``breakpoints`` in each dimension."""
    mids, widths = [], []
    for points in breakpoints:
        edges = np.unique(np.clip([0.0, 1.0, *points], 0.0, 1.0))
        mids.append(0.5 * (edges[1:] + edges[:-1]))
        widths.append(np.diff(edges))
    grids = np.meshgrid(*mids, indexing="ij")
    volumes = np.meshgrid(*widths, indexing="ij")
    uniforms = np.stack([g.ravel() for g in grids])
    return uniforms, np.prod(np.stack([v.ravel() for v in volumes]), axis=0)


def _replayed_law(
    level: float, a: int, a_new: int, u_latent: float, gamma: Any
) -> tuple[dict[Any, float], dict[Any, float]]:
    """The coded replay's exact law, mixed over the observed outcomes at ``(L, A)``.

    Returns the law of ``(T', D')`` and the joint law of the latent event ``(T^e', D^e')``
    and the replayed censoring time ``C'``.
    """
    x, x_new = np.array([a, level]), np.array([a_new, level])
    observed = _combined(_event_law(x, [0.0, 0.0]), _censoring_law(x))
    shifts = [g * u_latent for g in gamma]
    eta = [float(m.eta(x)) for m in STEP_CAUSES]
    eta_c, eta_c_new = float(STEP_CENSORING.eta(x)), float(STEP_CENSORING.eta(x_new))
    shared = [
        u
        for u in sorted(set(itertools.chain.from_iterable(m.atoms for m in STEP_CAUSES)))
        if all(m.jump(np.array([u]))[0] > 0 for m in STEP_CAUSES)
    ]
    outcome: dict[Any, float] = {}
    joint: dict[Any, float] = {}
    for (u, k), probability in observed.items():
        if probability == 0.0:
            continue
        cuts: list[list[float]] = []
        for i, model in enumerate(STEP_CAUSES):
            thresholds = [float(c) * math.exp(shifts[i]) for c in model._cum[1:]]
            cuts.append(_levels_at(model, u, eta[i], thresholds))
        ratio = math.exp(eta_c_new - eta_c) if a_new != a else 1.0
        thresholds = [float(c) * ratio for c in STEP_CENSORING._cum[1:]]
        cuts.append(_levels_at(STEP_CENSORING, u, eta_c, thresholds))
        h = np.array([[math.exp(eta[i]) * float(m.jump(np.array([u]))[0])] for i, m in
                      enumerate(STEP_CAUSES)])  # fmt: skip
        set_cuts: list[float] = []
        tie_cuts: list[float] = []
        if k >= 0:
            # The tied-set stream: the posterior weights of each subset of the other causes.
            others = [i for i in range(len(STEP_CAUSES)) if i != k]
            p = {i: -math.expm1(-float(h[i, 0])) for i in others}
            weights = []
            for mask in range(1 << len(others)):  # the code's order: bit b is others[b]
                members = [others[b] for b in range(len(others)) if mask >> b & 1]
                prior = math.prod(p[i] if i in members else 1.0 - p[i] for i in others)
                total = float(h[k, 0]) + sum(float(h[i, 0]) for i in members)
                weights.append(prior * (float(h[k, 0]) / total if total > 0 else 1.0))
            cumulative = np.cumsum(weights)
            set_cuts = list(cumulative[:-1] / cumulative[-1])
        for v in shared:
            w = [math.exp(eta[i] + shifts[i]) * float(m.jump(np.array([v]))[0]) for i, m in
                 enumerate(STEP_CAUSES)]  # fmt: skip
            position = w[0] / sum(w)
            tie_cuts.append(position)
            if k >= 0 and v == u:
                h0, h1 = float(h[0, 0]), float(h[1, 0])
                low, high = (0.0, h0 / (h0 + h1)) if k == 0 else (h0 / (h0 + h1), 1.0)
                tie_cuts.append((position - low) / (high - low))
        cuts.extend([set_cuts, tie_cuts])
        uniforms, volumes = _cells(cuts)
        m = volumes.size
        design = np.tile(x, (m, 1))
        models = cc.ReplayModels(STEP_CAUSES, STEP_CENSORING, design, design[:, 1:])
        time = np.full(m, u)
        cause_index = np.full(m, k)
        levels = cc.observed_levels(models, time, cause_index, uniforms)
        treatment, perturbed = np.full(m, a), np.full(m, a_new)
        latent = np.full(m, u_latent)
        gamma_array = np.asarray(gamma, dtype=float)
        replay = cc.replay_cell(
            models, levels, time, cause_index, treatment, perturbed, latent, gamma_array, tau=TAU
        )
        bare = dataclasses.replace(models, censoring=None)
        event = cc.replay_cell(
            bare, levels, time, cause_index, treatment, perturbed, latent, gamma_array, tau=TAU
        )
        _, censored_at = cc.latent_draws(models, levels, treatment, perturbed, latent, gamma_array)
        weight = probability * volumes
        for key, mass in _tally(zip(replay.time, replay.cause_index, strict=True), weight):
            outcome[key] = outcome.get(key, 0.0) + mass
        rows = zip(event.time, event.cause_index, censored_at, strict=True)
        for key, mass in _tally(rows, weight):
            joint[key] = joint.get(key, 0.0) + mass
    return outcome, joint


def _tally(rows: Any, weight: np.ndarray[Any, Any]) -> list[tuple[Any, float]]:
    totals: dict[Any, float] = {}
    for row, w in zip(rows, weight, strict=True):
        key = tuple(float(v) if isinstance(v, float | np.floating) else int(v) for v in row)
        totals[key] = totals.get(key, 0.0) + float(w)
    return list(totals.items())


def _law_error(level: float, a: int, a_new: int, u_latent: float, gamma: Any) -> float:
    """The largest gap between the coded replay's law and the declared law."""
    outcome, joint = _replayed_law(level, a, a_new, u_latent, gamma)
    x, x_new = np.array([a, level]), np.array([a_new, level])
    event = _event_law(x, [g * u_latent for g in gamma])
    censoring = _censoring_law(x_new)
    target = _combined(event, censoring)
    (ev, ev_none), (ce, ce_none) = event, censoring
    events = {**{(u, k): p for (u, k), p in ev.items()}, (TAU, -1): ev_none}
    censorings = {**ce, math.inf: ce_none}
    product = {(u, k, c): p * q for (u, k), p in events.items() for c, q in censorings.items()}
    gaps = [abs(outcome.get(key, 0.0) - target.get(key, 0.0)) for key in {*outcome, *target}]
    gaps += [abs(joint.get(key, 0.0) - product.get(key, 0.0)) for key in {*joint, *product}]
    return max(gaps)


STRATA = list(itertools.product([0.0, 1.0], [0, 1], [0, 1]))
LATENTS = [(-1.3, (0.7, 0.7)), (0.8, (0.9, 0.0)), (0.0, (0.0, 0.0)), (1.1, (0.0, -0.6))]


@pytest.mark.parametrize(("u_latent", "gamma"), LATENTS)
def test_t14d_the_replay_has_the_declared_discrete_law(u_latent: float, gamma: Any) -> None:
    """Exact: the replayed outcome is the grouped law with shifted jumps and the tie
    convention at the observed treatment, censored at the perturbed treatment, and the
    replayed censoring is independent of the latent event and of ``U`` and ``A``."""
    for level, a, a_new in STRATA:
        assert _law_error(level, a, a_new, u_latent, gamma) < 1e-12, (level, a, a_new)


def _worst(cases: Any = None) -> float:
    cases = cases or [(level, a, a_new, u, g) for level, a, a_new in STRATA for u, g in LATENTS]
    return max(_law_error(*case) for case in cases)


def test_t14d_rev1_conditioning_at_a_shared_atom_fails_the_law(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation: every other cause after the shared atom (rev 1) misses the tied set."""

    def alone(h: Any, k: int, position: Any) -> np.ndarray[Any, Any]:
        tied = np.zeros(h.shape, dtype=bool)
        tied[k] = True
        return tied

    monkeypatch.setattr(cc, "_tied_sets", alone)
    assert _worst([(0.0, 0, 0, 0.8, (0.9, 0.0)), (1.0, 1, 1, -1.3, (0.7, 0.7))]) > 1e-3


def test_t14d_reading_the_event_model_at_the_perturbed_treatment_fails_the_law(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation: the event read at ``A'`` fails for flipped units, even at ``gamma = 0``."""
    original = cc.latent_draws

    def at_new(models: Any, levels: Any, treatment: Any, perturbed: Any, latent: Any,
               shifts: Any) -> Any:  # fmt: skip
        times, censored_at = original(models, levels, treatment, perturbed, latent, shifts)
        changed = perturbed != treatment
        design = models.design.copy()
        design[:, 0] = perturbed
        for k, model in enumerate(models.causes):
            moved = levels.cause[k] * np.exp(
                -shifts[k] * latent + levels.eta[k] - model.eta(design)
            )
            times[k, changed] = model.inverse(moved)[changed]
        return times, censored_at

    monkeypatch.setattr(cc, "latent_draws", at_new)
    assert _worst([(0.0, 0, 1, 0.0, (0.0, 0.0))]) > 1e-3


def test_t14d_keeping_the_observed_censoring_fails_the_law(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation: ``V_c`` conditioned at ``A'`` keeps the observed censoring when the arm flips."""
    original = cc.latent_draws

    def kept(models: Any, levels: Any, treatment: Any, perturbed: Any, latent: Any,
             shifts: Any) -> Any:  # fmt: skip
        times, censored_at = original(models, levels, treatment, treatment, latent, shifts)
        if models.censoring is None:
            return times, censored_at
        return times, models.censoring.inverse(levels.censoring)

    monkeypatch.setattr(cc, "latent_draws", kept)
    assert _worst([(0.0, 0, 1, 0.0, (0.0, 0.0))]) > 1e-3


def test_t14d_censoring_conditioned_past_the_event_time_fails_the_law(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation: ``S_c(T)`` for ``S_c(T^-)`` loses censoring at an event's own atom.

    The zero-cell identity cannot see this (the event wins a tie with censoring), so the
    control sits here, where the censoring law at the shared atom 3 reads it.
    """
    original = cc._after

    def past(model: Any, t: Any, eta: Any, raw: Any, *, left: bool = False) -> Any:
        return original(model, t, eta, raw)

    monkeypatch.setattr(cc, "_after", past)
    assert _worst([(0.0, 0, 0, 0.0, (0.0, 0.0))]) > 1e-3


# ------------------------------------------------------------- T14e correct-model control


def _newton_cox(
    Z: np.ndarray[Any, Any], time: np.ndarray[Any, Any], event: np.ndarray[Any, Any]
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """The Breslow partial-likelihood maximizer by Newton's method, and its standard errors."""
    order = np.argsort(-time, kind="stable")
    Z, time, event = Z[order], time[order], event[order]
    beta = np.zeros(Z.shape[1])
    times = np.unique(time[event > 0])
    # Rows at risk at each event time: the first rows in descending time order.
    at_risk = np.searchsorted(-time, -times, side="right")
    deaths = np.array([np.sum((time == t) & (event > 0)) for t in times], dtype=float)
    z_events = Z[event > 0].sum(axis=0)
    info = np.eye(Z.shape[1])
    for _ in range(50):
        risk = np.exp(Z @ beta)
        s0 = np.cumsum(risk)[at_risk - 1]
        s1 = np.cumsum(risk[:, None] * Z, axis=0)[at_risk - 1]
        s2 = np.cumsum(risk[:, None, None] * Z[:, :, None] * Z[:, None, :], axis=0)[at_risk - 1]
        mean = s1 / s0[:, None]
        grad = z_events - (deaths[:, None] * mean).sum(axis=0)
        info = (
            deaths[:, None, None] * (s2 / s0[:, None, None] - mean[:, :, None] * mean[:, None, :])
        ).sum(axis=0)
        step = np.linalg.solve(info, grad)
        if np.max(np.abs(step)) > 1.0:  # a damped first step keeps Newton inside the basin
            step = step / np.max(np.abs(step))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-12:
            break
    return beta, np.sqrt(np.diag(np.linalg.inv(info)))


def _control_t(
    event_mutation: bool = False, censoring_mutation: bool = False
) -> dict[str, list[float]]:
    """The |t| of each replayed hazard's latent coefficient against its declared value."""
    # Strong arms and frequent events let each mutation show at n = 2,000 (|t| above 9 at
    # this seed, the correct models near 2).
    law = _constant_law(2000, 21, censored=True, arm=2.5, rate=3.0)
    models, time, cause_index, a = law["models"], law["time"], law["cause_index"], law["a"]
    levels = _levels(law, seed=22)
    latent = np.random.default_rng(23).normal(size=time.size)
    perturbed = np.where(_flip_mask(latent, 0.3), 1 - a, a)
    gamma = 0.5
    with pytest.MonkeyPatch.context() as patch:
        original = cc.latent_draws
        if event_mutation or censoring_mutation:

            def mutated(models: Any, levels: Any, treatment: Any, perturbed: Any, latent: Any,
                        shifts: Any) -> Any:  # fmt: skip
                times, censored_at = original(models, levels, treatment, perturbed, latent, shifts)
                design = models.design.copy()
                design[:, 0] = perturbed
                if event_mutation:
                    for k, model in enumerate(models.causes):
                        moved = levels.cause[k] * np.exp(
                            -shifts[k] * latent + levels.eta[k] - model.eta(design)
                        )
                        times[k] = model.inverse(moved)
                if censoring_mutation:
                    censored_at = models.censoring.inverse(levels.censoring)
                return times, censored_at

            patch.setattr(cc, "latent_draws", mutated)
        replay = cc.replay_cell(
            models, levels, time, cause_index, a, perturbed, latent, np.array([gamma, gamma])
        )
    w = law["w"]
    tau = float(time.max())
    stats: dict[str, list[float]] = {"causes": [], "censoring": []}
    # Split by arm: within an observed arm a flip of A moves only A', so an event read at A'
    # shows in the latent coefficient there; within a perturbed arm censoring read at the
    # observed A shows the same way. Pooled over arms the two biases cancel.
    for arm in (0.0, 1.0):
        rows = a == arm
        Z = np.column_stack([w, latent])[rows]
        for k in range(2):
            event = (replay.cause_index[rows] == k).astype(float)
            beta, se = _newton_cox(Z, replay.time[rows], event)
            stats["causes"].append(abs(beta[1] - gamma) / se[1])
        rows = perturbed == arm
        left = (replay.cause_index[rows] == -1) & (replay.time[rows] < tau)
        beta, se = _newton_cox(
            np.column_stack([w, latent])[rows], replay.time[rows], left.astype(float)
        )
        stats["censoring"].append(abs(beta[1]) / se[1])
    return stats


def test_t14e_the_replay_under_the_true_models_has_the_declared_hazards() -> None:
    """Each cause's hazard on the observed ``A`` reads the latent at ``gamma``; censoring on
    ``A'`` does not read it. Both models are conditionally correct, because ``A`` is a
    function of ``(A', U)``."""
    stats = _control_t()
    assert max(stats["causes"] + stats["censoring"]) < 4


def test_t14e_reading_the_event_at_the_perturbed_treatment_shows() -> None:
    """Mutation: the event read at ``A'`` biases each cause's latent coefficient."""
    assert max(_control_t(event_mutation=True)["causes"]) > 8


def test_t14e_keeping_the_observed_censoring_shows() -> None:
    """Mutation: censoring kept at the observed arm makes it read the latent."""
    assert max(_control_t(censoring_mutation=True)["censoring"]) > 8


# ---------------------------------------------------------------------- T14f calibration


def _wide_frame() -> pd.DataFrame:
    """The sweep law with its covariate on a scale whose SD is about 5."""
    frame = _frame()
    return frame.assign(W=5.0 * frame.W + 2.0)


def test_t14f_each_outcome_row_is_the_standardized_hazard_coefficient() -> None:
    result = _fit(_wide_frame())
    surface = _surface(result, benchmark_covariates=("W",))
    rows = {(row.role, row.component): row for row in surface.calibrations}
    assert rows["treatment", None].family == "binomial"
    data = result.data
    w = np.asarray(data.baseline, dtype=float)[:, 0]
    assert abs(np.std(w) - 5.0) < 0.6
    a = np.asarray(data.treatment, dtype=float)
    time = np.asarray(data.time, dtype=float)
    event = np.asarray(data.event, dtype=int)
    standardized = (w - w.mean()) / w.std()
    for code, label in [(1, "1"), (2, "2")]:
        row = rows["outcome", label]
        assert row.family == "hazard" and row.method == cc.REPLAY_COX_METHOD
        beta, _ = _newton_cox(np.column_stack([a, standardized]), time, (event == code) * 1.0)
        assert row.strength == pytest.approx(beta[1], abs=1e-8)
        # Mutation: the raw covariate's coefficient is per unit, not per SD.
        raw, _ = _newton_cox(np.column_stack([a, w]), time, (event == code) * 1.0)
        assert abs(row.strength - raw[1]) > 1e-3


def test_t14f_a_covariate_outside_the_hazard_reads_near_zero() -> None:
    law = _constant_law(2000, 21, censored=True)
    noise = np.random.default_rng(31).normal(size=law["time"].size)
    frame = pd.DataFrame(
        {
            "time": law["time"],
            "event": law["cause_index"] + 1,
            "A": law["a"],
            "W": law["w"],
            "N": noise,
        }
    )
    result = _quiet(
        lambda: ContinuousSurvivalTMLE(
            [1.0],
            cross_fit=False,
            hazard_library=[CoxHazard("main")],
            censoring_library=[CoxHazard("main")],
            random_state=0,
        ).fit(frame, time="time", event="event", treatment="A", baseline=["W", "N"])
    )
    name = next(iter(result.estimates))
    grid = ConfounderStrengthGrid((0.0,), (0.0,))
    surface = _surface(result, name, grid, benchmark_covariates=("N",))
    standardized = np.column_stack([law["a"], law["w"] / law["w"].std(), noise / noise.std()])
    for k, label in enumerate(["1", "2"]):
        (row,) = [r for r in surface.calibrations if r.component == label]
        _, se = _newton_cox(standardized, law["time"], (law["cause_index"] == k) * 1.0)
        assert abs(row.strength) < 3 * se[2]


# ------------------------------------------------------------- T14g refusals and arguments


@pytest.mark.parametrize("missing", ["estimator", "data"])
def test_t14g_a_result_without_its_estimator_or_data_is_refused(missing: str) -> None:
    bare = dataclasses.replace(_in_sample(), **{missing: None})
    row = bare.sensitivity.capability("simulated_confounding")
    assert not row.available and row.refusal is not None
    with pytest.raises(CapabilityError) as free:
        simulated_confounding(bare, RD, grid=SWEEP_GRID, random_state=0)
    with pytest.raises(CapabilityError) as facade:
        bare.sensitivity.simulated_confounding(estimand=RD, grid=SWEEP_GRID, random_state=0)
    assert free.value.refusal == row.refusal
    assert free.value.refusal == cc.MISSING_REFUSAL
    assert facade.value.refusal.reason is row.refusal.reason
    assert facade.value.refusal.condition.endswith(row.refusal.condition)


MALFORMED = [
    ({"arm_pair": (0, 1)}, "arm_pair"),
    ({"latent_level": "cluster"}, "latent_level"),
    ({"latent_rho": 0.3}, "latent_level and latent_rho"),
    ({"observation_reads": ("treatment",)}, "observation_reads"),
    ({"bounded_outcome_law": "logit"}, "bounded_outcome_law"),
    ({"grid": ConfounderStrengthGrid((0.0,), (0.0,), intermediate=(0.0, 0.5))}, "intermediate"),
    ({"grid": ConfounderStrengthGrid({"A": (0.0, 0.2)}, (0.0,))}, "treatment component"),
    ({"grid": ConfounderStrengthGrid((0.0, 0.6), (0.0,))}, "between 0 and 0.5"),
    ({"outcome_cause": "3"}, "not a cause of this fit"),
    ({"benchmark_covariates": ("W", "W")}, "duplicates"),
    ({"benchmark_covariates": ("Z",)}, "unavailable"),
    ({"estimand": "ate"}, "was not reported"),
]


@pytest.mark.parametrize(("arguments", "words"), MALFORMED)
def test_t14g_each_malformed_argument_raises(
    arguments: dict[str, Any], words: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every argument is checked before any fit, draw or refit."""

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("work started before the request was checked")

    monkeypatch.setattr(cc, "fit_replay_models", forbidden)
    options: dict[str, Any] = {"estimand": RD, "grid": SWEEP_GRID, "random_state": 0}
    options.update(arguments)
    estimand = options.pop("estimand")
    with pytest.raises(ValueError, match=words):
        simulated_confounding(_in_sample(), estimand, **options)


def test_t14g_a_named_cause_on_a_one_cause_fit_raises() -> None:
    result = _fits()["one_cause"]
    with pytest.raises(ValueError, match="declares one cause"):
        simulated_confounding(
            result, "ate_regimen[1 vs 0 @ t=1]", grid=SWEEP_GRID, outcome_cause="1"
        )


def test_t14g_a_non_finite_outcome_strength_raises() -> None:
    """The grid itself refuses a non-finite strength, before the continuous checks run."""
    with pytest.raises(ValueError, match="must be finite"):
        ConfounderStrengthGrid((0.0,), (0.0, math.inf))


def test_t14g_a_negative_outcome_strength_runs() -> None:
    surface = _surface(_fresh(_in_sample()), grid=ConfounderStrengthGrid((0.0,), (-0.5, 0.0)))
    assert surface.complete


def test_t14g_an_overflowing_replay_model_fails_every_moved_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = cc.fit_replay_models

    def overflowing(*args: Any, **kwargs: Any) -> Any:
        models = original(*args, **kwargs)
        first = models.causes[0]
        broken = dataclasses.replace(first, model=_Overflow(first.model))
        models = dataclasses.replace(models, causes=(broken, *models.causes[1:]))
        time, code, _, _, codes, _ = args
        return dataclasses.replace(models, failure=cc.check_models(models, time, code, codes))

    monkeypatch.setattr(cc, "fit_replay_models", overflowing)
    result = _fresh(_in_sample())
    surface = _surface(result)
    anchor, *moved = surface.cells
    assert anchor.estimate == result[RD].psi
    assert moved and all(cell.failure is not None for cell in moved)
    assert all("not finite" in cell.failure.message for cell in moved)
    assert any("cannot be read" in line for line in surface.replay_models)


class _Overflow:
    """A fitted Cox model whose linear predictor overflows."""

    def __init__(self, model: Any) -> None:
        self._model = model
        self.coef_ = model.coef_

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def linear_predictor(self, X: Any) -> np.ndarray[Any, Any]:
        values = np.asarray(self._model.linear_predictor(X), dtype=float).copy()
        values[0] = np.inf
        return values


# ------------------------------------------------------------- T14h seeds, shares, failures


@functools.cache
def _sweep_surface(random_state: int = 0) -> Any:
    return _surface(_fresh(_in_sample()), random_state=random_state)


def test_t14h_the_same_seed_gives_the_same_cells() -> None:
    again = _surface(_fresh(_in_sample()))
    assert [cell.estimate for cell in again.cells] == [
        cell.estimate for cell in _sweep_surface().cells
    ]
    assert again.coupling_seed == coupling_seed(0)


def test_t14h_the_fold_plan_is_frozen_only_under_the_fit_seed() -> None:
    assert _sweep_surface().refit_folds == "original"
    assert _sweep_surface(5).refit_folds == "redrawn"


def test_t14h_the_shares_vanish_at_zero_and_move_off_it() -> None:
    surface = _sweep_surface()
    anchor = surface.cells[0]
    # The anchor reuses the fit, so it carries no shares; the zero replay's are zero.
    assert anchor.extended_share is None and anchor.new_event_share is None
    zero = _sweep().replay()
    assert (zero.censoring_redrawn_share, zero.extended_share, zero.new_event_share) == (
        0.0,
        0.0,
        0.0,
    )
    (moved,) = [
        c for c in surface.cells if (c.treatment_strength, c.outcome_strength) == (0.22, 0.17)
    ]
    assert moved.censoring_redrawn_share is not None and moved.censoring_redrawn_share > 0.0
    assert moved.extended_share is not None and moved.extended_share > 0.0
    assert moved.new_event_share is not None and moved.new_event_share > 0.0


def test_t14h_no_cell_of_the_sweep_grid_fails() -> None:
    surface = _sweep_surface()
    assert surface.complete
    assert all(cell.failure is None for cell in surface.cells)
    # Nonzero witness: some refit moves the estimate.
    assert any(cell.estimate != surface.original_estimate for cell in surface.cells[1:])


# ------------------------------------------------------------- T14i plot, summary and sweep


def _plot_text(surface: Any) -> tuple[str, list[str]]:
    import matplotlib.pyplot as plt

    from cleverly.plotting import confounding

    figure = confounding(surface)
    try:
        ax = figure.axes[0]
        lines = ax.get_ylabel().split("\n")
        label = "".join(
            line if line.endswith("-") or i == len(lines) - 1 else line + " "
            for i, line in enumerate(lines)
        )
        marks = [
            text.get_text() for text in ax.texts if text.get_gid() == "cleverly-calibration-label"
        ]
    finally:
        plt.close(figure)
    return label, marks


def _summary_text(surface: Any) -> str:
    return " ".join(str(surface.summary()).split())


BINOMIAL_LABEL = "Share of units whose outcome is flipped"


def test_t14i_the_continuous_surface_reads_on_the_hazard_scale() -> None:
    from cleverly._summary.vocabulary import NOTE_TEXTS

    # The grid spans the calibration rows (about 0.43, 0.53 and -0.49), so each is drawn.
    grid = ConfounderStrengthGrid((0.0, 0.25, 0.5), (-0.6, 0.0, 0.6))
    surface = _surface(_fresh(_in_sample()), grid=grid, benchmark_covariates=("W",))
    label, marks = _plot_text(surface)
    assert label == "Shift of each cause-specific log-hazard per SD of the simulated cause"
    assert sorted(marks) == ["W", "W (1)", "W (2)"]
    assert _unit_words(surface) == {"log-hazard coefficient"}
    text = _summary_text(surface)
    law = NOTE_TEXTS["law_outcome_continuous_hazard"].format(causes="every cause")
    assert " ".join(law.split()) in text
    # Mutation: the binomial words on this surface.
    assert BINOMIAL_LABEL not in label
    assert " ".join(NOTE_TEXTS["law_outcome_binomial"].split()) not in text


def test_t14i_the_survival_panel_surface_reads_on_the_log_odds_scale() -> None:
    from cleverly._summary.vocabulary import NOTE_TEXTS
    from tests.unit.test_simulated_confounding_survival import _hazard_ltmle, _panel

    frame, columns = _panel(600, 4)
    result = _quiet(lambda: _hazard_ltmle().fit(frame, **columns))
    name = "risk_regimen[a1 @ t=3]"
    surface = _surface(
        result,
        name,
        ConfounderStrengthGrid((0.0, 0.2), (-1.0, 0.0, 1.0)),
        benchmark_covariates=("W2",),
        random_state=3,
    )
    assert surface.outcome_family == "survival"
    label, marks = _plot_text(surface)
    assert label == "Shift of the per-node event log-odds per SD of the simulated cause"
    assert marks.count("W2") >= 1
    assert _unit_words(surface) == {"log-odds coefficient"}
    text = _summary_text(surface)
    assert " ".join(NOTE_TEXTS["law_outcome_panel_hazard"].split()) in text
    # Mutation: the binomial words on this surface.
    assert BINOMIAL_LABEL not in label
    assert " ".join(NOTE_TEXTS["law_outcome_binomial"].split()) not in text


def _unit_words(surface: Any) -> set[str]:
    """The unit word of each outcome calibration row, read where the plot lists it in words.

    On a time-to-event surface an outcome row is drawn as a mark on the outcome axis, so the
    surface is relabelled Gaussian to send every row to the plot's list of rows in words.
    """
    from cleverly.plotting._confounding import _calibration_marks, _Dim

    relabelled = dataclasses.replace(surface, outcome_family="gaussian")
    axis = _Dim("treatment", [0.0], float, "", "", "binary")
    _, _, words = _calibration_marks(relabelled, axis)
    outcome = [item for item in words if "(outcome" in item]
    assert outcome
    return {item.split(", ")[1] for item in outcome}


def test_t14i_the_sweep_runs_the_row_on_both_continuous_kinds() -> None:
    from tests.unit._capability_sweep_support import KINDS

    for kind in ("continuous_survival", "continuous_survival+restricted"):
        assert "simulated_confounding" in KINDS[kind].must_run
