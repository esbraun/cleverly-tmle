"""Mutation controls for hazard-based survival targeting (X14 M1-M9, M11, M12, M14, M15).

An exact-law identity is blind to a term that vanishes at the truth.  Each test applies one
deliberate mutation by monkeypatch and checks that the identity it should break does break.
The wrong hazard learner of ``tests/unit/test_hazard_survival_law.py`` is the witness: with
it, every targeting term is nonzero.  No source file is edited.

Two controls are positive.  A sign flip of the clever covariate leaves the per-horizon Newton
step exact, because the step solves the scores along whatever basis it is given (M4).  A
swapped cause order changes the initial hazards and leaves the targeted estimate exact (M12).
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from cleverly.exceptions import ConvergenceWarning, WeightingWarning
from cleverly.longitudinal import LTMLE
from cleverly.longitudinal import hazard as hazard_module
from cleverly.longitudinal.hazard import HazardInputs, HazardState
from tests.discrete_law_point_survival import (
    CellMeans,
    CellProbabilities,
    PointLaw,
    known,
    static,
)
from tests.unit.test_hazard_survival_law import (
    POLICY,
    STEP,
    STOCHASTIC,
    ArmOnly,
    _weighted,
)

ONE = PointLaw("survival", 2)
TWO = PointLaw("survival", 2, causes=2)
REGIMENS = {"a0": 0, "a1": 1, "a2": 2}
#: How far a mutated estimate must sit from the truth for the identity to count as broken.
BROKEN = 1e-6


def _fit(law: PointLaw, targeting: str, **settings: Any) -> Any:
    frame = settings.pop("frame", None)
    columns = settings.pop("columns", {})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", WeightingWarning)
        return LTMLE(
            settings.pop("regimens", REGIMENS),
            n_folds=1,
            survival_model="hazard",
            targeting=targeting,
            step_size=STEP,
            outcome_learner=settings.pop("learner", ArmOnly()),
            treatment_learner=CellProbabilities(),
            censoring_learner=CellMeans(),
        ).fit(law.frame() if frame is None else frame, **{**law.fit_columns(), **columns})


def _miss(law: PointLaw, result: Any) -> float:
    """The largest distance from a static-arm estimate to the truth."""
    worst = 0.0
    for fit in result.fits.values():
        arm = REGIMENS[fit.regimen.label]
        code = 1 if fit.cause in (None, "relapse") else 2
        truth = law.functional(law.probs, static(arm), fit.horizon, code)
        worst = max(worst, abs(fit.psi_scaled - truth))
    return worst


def _unsolved(result: Any) -> bool:
    return not all(row.passed for row in result.diagnostics.score_equations().rows)


def _gateaux_gap(law: PointLaw, result: Any) -> float:
    rows = law.first_row_of()
    worst = 0.0
    for fit in result.fits.values():
        arm = REGIMENS[fit.regimen.label]
        eif = law.eif(static(arm), fit.horizon)
        name = f"risk_regimen[{fit.regimen.label} @ t={fit.horizon}]"
        worst = max(worst, float(np.max(np.abs(result[name].influence_curve[rows] - eif))))
    return worst


@pytest.mark.parametrize("targeting", ["iterative", "one_step"])
def test_the_unmutated_witness_is_exact(targeting: str) -> None:
    """The baseline every control below departs from."""
    for law in (ONE, TWO):
        result = _fit(law, targeting)
        assert _miss(law, result) < 1e-8
        assert not _unsolved(result)


# ------------------------------------------------------------------- M1 censoring index


@pytest.mark.parametrize("targeting", ["iterative", "one_step"])
def test_m1_the_censoring_factor_of_the_previous_node_breaks_double_robustness(
    monkeypatch: pytest.MonkeyPatch, targeting: str
) -> None:
    original = hazard_module.hazard_inputs

    def lagged(*args: Any, **kwargs: Any) -> HazardInputs:
        inputs = original(*args, **kwargs)
        inverse = inputs.inverse.copy()
        inverse[:, 1:] = inputs.inverse[:, :-1]
        return HazardInputs(**{**inputs.__dict__, "inverse": inverse})

    monkeypatch.setattr(hazard_module, "hazard_inputs", lagged)
    assert _miss(ONE, _fit(ONE, targeting)) > BROKEN


# -------------------------------------------------------------------- M2 survival ratio


@pytest.mark.parametrize("targeting", ["iterative", "one_step"])
def test_m2_the_survival_of_the_previous_node_breaks_the_influence_curve(
    monkeypatch: pytest.MonkeyPatch, targeting: str
) -> None:
    """The clever covariate divides by S(k-1) in place of S(k)."""
    original = HazardState.__post_init__

    def lagged(self: HazardState) -> None:
        original(self)
        survival = self.survival
        before = np.concatenate([np.ones_like(survival[:, :1]), survival[:, :-1]], axis=1)
        object.__setattr__(self, "survival", before)

    monkeypatch.setattr(HazardState, "__post_init__", lagged)
    result = _fit(ONE, targeting)
    assert _miss(ONE, result) > BROKEN
    saturated = _fit(ONE, targeting, learner=CellMeans())
    assert _gateaux_gap(ONE, saturated) > BROKEN


# ------------------------------------------------------------------- M3 other-cause term


def _own_cause_only_terms(state: HazardState, inputs: HazardInputs, horizons: Any) -> Any:
    """``_score_terms`` with the residual of every cause other than ``j`` dropped."""
    obs = hazard_module._observed(state, inputs)
    inverse = inputs.inverse_observed()
    q = inputs.q_observed()
    own = obs.residual / obs.survival[:, :, None]
    running_j = np.cumsum(inverse[:, :, None] * (obs.residual + obs.incidence * own), axis=1)
    running = np.cumsum(inverse[:, :, None] * own, axis=1)
    index = [t - 1 for t in horizons]
    unweighted = running_j[:, index, :] - obs.incidence[:, index, :] * running[:, index, :]
    terms = q[:, None, None, :] * unweighted.transpose(2, 1, 0)[None]
    _, scale = _ORIGINAL_TERMS(state, inputs, horizons)
    return terms, scale


_ORIGINAL_TERMS = hazard_module._score_terms


@pytest.mark.parametrize("targeting", ["iterative", "one_step"])
def test_m3_dropping_the_other_cause_term_breaks_double_robustness(
    monkeypatch: pytest.MonkeyPatch, targeting: str
) -> None:
    """The clever covariate of cause ``j`` loses its bracket on the other cause's residual."""
    monkeypatch.setattr(hazard_module, "_score_terms", _own_cause_only_terms)
    assert _miss(TWO, _fit(TWO, targeting)) > BROKEN


# -------------------------------------------------------------------------------- M4 sign


def _flipped(original: Any) -> Any:
    def direction(*args: Any, **kwargs: Any) -> Any:
        return -original(*args, **kwargs)

    return direction


def test_m4_a_flipped_clever_covariate_stalls_the_walk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hazard_module, "direction", _flipped(hazard_module.direction))
    result = _fit(ONE, "one_step")
    assert _unsolved(result)
    assert _miss(ONE, result) > BROKEN


def test_m4_a_newton_step_absorbs_the_sign(monkeypatch: pytest.MonkeyPatch) -> None:
    """Positive control: the per-horizon step solves the scores along any basis."""
    monkeypatch.setattr(hazard_module, "direction", _flipped(hazard_module.direction))
    result = _fit(ONE, "iterative")
    assert not _unsolved(result)
    assert _miss(ONE, result) < 1e-8


# -------------------------------------------------------------------- M5 stale direction


@pytest.mark.parametrize("targeting", ["iterative", "one_step"])
def test_m5_a_clever_covariate_frozen_at_the_initial_fit_breaks_double_robustness(
    monkeypatch: pytest.MonkeyPatch, targeting: str
) -> None:
    """The score reads the incidence and survival of the initial hazards at every step.

    A stale direction alone cannot do this: both algorithms stop on the scores at the
    current hazards, so a stale step only slows them.  A stale score solves the wrong
    equation, and the estimate leaves the truth.
    """
    original = hazard_module._observed
    frozen: dict[str, Any] = {}

    def stale(state: HazardState, inputs: Any) -> Any:
        current = original(state, inputs)
        first = frozen.setdefault("first", current)
        return type(current)(
            incidence=first.incidence,
            survival=first.survival,
            residual=current.residual,
            total=current.total,
        )

    monkeypatch.setattr(hazard_module, "_observed", stale)
    assert _miss(ONE, _fit(ONE, targeting)) > BROKEN


# ------------------------------------------------------------------ M6 last horizon only


def test_m6_targeting_the_last_horizon_only_leaves_earlier_scores_unsolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The walk sees the last horizon's scores only.

    The record then reads solved, so E4 reads the score off the influence curve, whose mean
    is the score of its target: the plug-in term is centred by construction.
    """
    original = hazard_module.scores

    def last_only(state: HazardState, inputs: Any, horizons: Any) -> tuple[Any, Any]:
        score, scale = original(state, inputs, horizons)
        kept = np.zeros_like(score)
        kept[:, :, -1] = score[:, :, -1]
        return kept, scale

    monkeypatch.setattr(hazard_module, "scores", last_only)
    result = _fit(ONE, "one_step")
    first = abs(float(np.mean(result["risk_regimen[a1 @ t=1]"].influence_curve)))
    last = abs(float(np.mean(result["risk_regimen[a1 @ t=2]"].influence_curve)))
    assert first > 1e-6
    assert last < 1e-8


# ----------------------------------------------------------------------------- M7 weights


@pytest.mark.parametrize("targeting", ["iterative", "one_step"])
def test_m7_an_unweighted_fluctuation_misses_the_tilted_law(
    monkeypatch: pytest.MonkeyPatch, targeting: str
) -> None:
    weighted, repeated, _ = _weighted(ONE)
    truth = _fit(ONE, targeting, learner=CellMeans(), frame=repeated)

    def unweighted(state: Any, inputs: Any, horizons: Any) -> tuple[Any, Any]:
        terms, scale = hazard_module._score_terms(state, inputs, horizons)
        return np.asarray(terms.mean(axis=-1), dtype=float), scale

    monkeypatch.setattr(hazard_module, "scores", unweighted)
    wrong = _fit(ONE, targeting, frame=weighted, columns={"weights": "wt"})
    gap = max(abs(wrong[name].psi - truth[name].psi) for name in truth.estimates)
    assert gap > BROKEN


# ----------------------------------------------------------------------- M8 policy weight


@pytest.mark.parametrize("targeting", ["iterative", "one_step"])
def test_m8_an_indicator_in_place_of_the_policy_weight_breaks_the_policy(
    monkeypatch: pytest.MonkeyPatch, targeting: str
) -> None:
    def indicator(self: HazardInputs) -> Any:
        q = self.q[:, self.rows, self.observed]
        return (q > 0).astype(float)

    monkeypatch.setattr(HazardInputs, "q_observed", indicator)
    result = _fit(ONE, targeting, regimens={"q": STOCHASTIC})
    gaps = [
        abs(result[f"risk_regimen[q @ t={t}]"].psi - ONE.functional(ONE.probs, known(POLICY), t))
        for t in (1, 2)
    ]
    assert max(gaps) > BROKEN


# -------------------------------------------------------------------------- M9 arm levels


@pytest.mark.parametrize("targeting", ["iterative", "one_step"])
def test_m9_updating_the_observed_level_only_breaks_double_robustness(
    monkeypatch: pytest.MonkeyPatch, targeting: str
) -> None:
    original = hazard_module.direction

    def observed_only(state: HazardState, inputs: Any, *args: Any) -> Any:
        out = original(state, inputs, *args)
        observed = np.arange(out.shape[2])[None, :] == inputs.observed[:, None]
        return out * observed[:, None, :, None]

    monkeypatch.setattr(hazard_module, "direction", observed_only)
    result = _fit(ONE, targeting)
    assert _miss(ONE, result) > BROKEN or _unsolved(result)


# ------------------------------------------------------------------------- M11 third arm


@pytest.mark.parametrize("targeting", ["iterative", "one_step"])
def test_m11_a_wrong_level_map_breaks_the_saturated_estimate(
    monkeypatch: pytest.MonkeyPatch, targeting: str
) -> None:
    original = hazard_module.regimen_weights

    def swapped(*args: Any) -> Any:
        return original(*args)[..., [0, 2, 1]]

    monkeypatch.setattr(hazard_module, "regimen_weights", swapped)
    assert _miss(ONE, _fit(ONE, targeting, learner=CellMeans())) > BROKEN


# -------------------------------------------------------------------- M12 competing order


@pytest.mark.parametrize("targeting", ["iterative", "one_step"])
def test_m12_the_cause_order_moves_the_initial_fit_and_not_the_target(targeting: str) -> None:
    """Positive control: the sequential pseudo-hazards depend on the order, the target not."""
    columns = TWO.fit_columns()
    outcome = columns["outcome"]
    reordered = {"death": outcome["death"], "relapse": outcome["relapse"]}
    # A saturated learner gives every order the same hazards, so the witness is parametric.
    learner = LogisticRegression(max_iter=1000)
    first = _fit(TWO, targeting, learner=learner)
    second = _fit(TWO, targeting, learner=clone(learner), columns={"outcome": reordered})
    assert _miss(TWO, first) < 1e-8
    assert _miss(TWO, second) < 1e-8
    initial = {
        (t[0], t[1], t[2]): float(v)
        for fit in first.fits.values()
        for t, v in zip(fit.hazard.targets, fit.hazard.initial_levels, strict=True)
    }
    other = {
        (t[0], t[1], t[2]): float(v)
        for fit in second.fits.values()
        for t, v in zip(fit.hazard.targets, fit.hazard.initial_levels, strict=True)
    }
    assert max(abs(initial[key] - other[key]) for key in initial) > 1e-3


# -------------------------------------------------------------------- M14 moving metric


def _moving_metric_walk(
    initial: HazardState,
    inputs: HazardInputs,
    horizons: Any,
    targets: Any,
    *,
    step_size: float,
    tol: float,
    max_steps: int = 20_000,
    warn: bool = True,
) -> Any:
    """``walk_curve`` with the metric re-estimated after every accepted step."""
    from dataclasses import replace

    from cleverly.fluctuation._score import relative_score

    state = initial
    _, curves = hazard_module.influence(state, inputs, horizons)
    sigma = np.maximum(np.mean(curves**2, axis=-1), hazard_module.SIGMA_FLOOR)
    score, scale = hazard_module.scores(state, inputs, horizons)
    norm = float(np.sqrt(np.sum(score**2 / sigma)))
    norms, trace = [norm], [relative_score(score.reshape(-1), scale.reshape(-1))]
    dx, steps = float(step_size), 0
    while trace[-1] > tol and steps < max_steps and norm > 0.0:
        step = hazard_module.direction(state, inputs, horizons, score / sigma / norm)
        candidate = state.moved(dx * step, inputs.bounds)
        candidate_score, candidate_scale = hazard_module.scores(candidate, inputs, horizons)
        candidate_norm = float(np.sqrt(np.sum(candidate_score**2 / sigma)))
        if candidate_norm > norm:
            dx *= 0.5
            if dx < hazard_module._MIN_STEP:
                break
            continue
        state, score, scale, steps = candidate, candidate_score, candidate_scale, steps + 1
        _, curves = hazard_module.influence(state, inputs, horizons)
        sigma = np.maximum(np.mean(curves**2, axis=-1), hazard_module.SIGMA_FLOOR)
        norm = float(np.sqrt(np.sum(score**2 / sigma)))
        norms.append(norm)
        trace.append(relative_score(score.reshape(-1), scale.reshape(-1)))
    _, record = _ORIGINAL_WALK(
        initial, inputs, horizons, targets, step_size=step_size, tol=tol, warn=False
    )
    return state, replace(
        record, sigma=sigma.reshape(-1), norm_trace=tuple(norms), n_iter=steps, trace=tuple(trace)
    )


_ORIGINAL_WALK = hazard_module.walk_curve


def _metric_ratios(monkeypatch: pytest.MonkeyPatch) -> np.ndarray:
    """``score / (coefficients * sigma_initial)`` at every direction the walk asks for.

    A spy on :func:`hazard.direction` records the coefficients each step is built from, and
    the scores of the state it is built at.  A walk with the initial metric fixed gives one
    ratio, the inverse of each step's norm, across every component of a step.
    """
    calls: list[tuple[Any, Any, Any, np.ndarray]] = []
    original = hazard_module.direction

    def spy(state: Any, inputs: Any, horizons: Any, coefficients: Any) -> Any:
        calls.append((state, inputs, horizons, np.array(coefficients, dtype=float)))
        return original(state, inputs, horizons, coefficients)

    monkeypatch.setattr(hazard_module, "direction", spy)
    result = _fit(ONE, "one_step")
    inputs = _inputs(result)
    start = HazardState.from_hazards(result.initial_hazards.hazards, inputs.bounds, inputs.empty)
    _, curves = hazard_module.influence(start, inputs, (1, 2))
    sigma = np.maximum(np.mean(curves**2, axis=-1), hazard_module.SIGMA_FLOOR)
    assert len(calls) > 1
    ratios = []
    for state, step_inputs, horizons, coefficients in calls:
        score, _ = hazard_module.scores(state, step_inputs, horizons)
        moved = coefficients != 0.0
        ratio = score[moved] / (coefficients * sigma)[moved]
        ratios.append(np.ptp(ratio) / np.max(np.abs(ratio)))
    return np.asarray(ratios)


def test_m14_the_shipped_walk_steps_in_the_initial_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive control: every step's coefficients are the score in the initial metric."""
    assert np.max(_metric_ratios(monkeypatch)) < 1e-10


def test_m14_a_metric_re_estimated_at_every_step_fails_the_metric_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mutated walk steps in a moving metric, which the spy reads off the coefficients."""
    monkeypatch.setattr(hazard_module, "walk_curve", _moving_metric_walk)
    assert np.max(_metric_ratios(monkeypatch)) > 1e-6


def _inputs(result: Any) -> HazardInputs:
    recipe = result.replay_recipe
    cumulative = [
        result.mechanism.cumulative(result.data, level, result.config.g_bounds)
        for level in recipe.hazard.levels
    ]
    return hazard_module.hazard_inputs(result.data, recipe.plans, cumulative, alpha=recipe.alpha)


# ------------------------------------------------------------- M15 clip, then renormalise


def _extreme() -> np.ndarray:
    """Hazards at and beyond the bound: one cause near one, both causes summing to one."""
    return np.array([[[[0.9999, 0.0001]], [[0.5, 0.5]], [[1.0, 0.0]], [[1e-9, 1e-9]]]])


def test_m15_the_softmax_clip_keeps_some_mass_on_no_event() -> None:
    bounds = (-np.log(0.9995 / 0.0005), np.log(0.9995 / 0.0005))
    state = HazardState.from_hazards(_extreme(), bounds)
    assert np.all(state.hazards.sum(axis=-1) < 1.0)
    assert np.all(state.hazards > 0.0)
