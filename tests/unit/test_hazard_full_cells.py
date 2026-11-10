"""A hazard cell whose followers all hold an event is held at one (X14).

The witness sample draws 400 units over three nodes.  Every unit of arm 1 still at risk at
node 2 has the event there, and every other hazard is about 0.2.  The maximum-likelihood
all-cause hazard of that cell is one, which no clipped logit reaches, so a walk that clips it
spends its whole step budget.  The fit holds the all-cause hazard at exactly one, the mirror
of the empty-cell hold in ``tests/unit/test_hazard_empty_cells.py``, and every parameter that
reads the cell takes the ``"constant_node_plugin"`` status.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from cleverly.longitudinal import LTMLE, LongitudinalData
from cleverly.longitudinal.hazard import (
    HazardState,
    clip_logits,
    hazard_inputs,
    walk_curve,
    walk_loss,
)
from cleverly.longitudinal.sequential import constant_nodes

REGIMENS = {"a1": 1, "a0": 0}
HORIZONS = (1, 2)


def _data() -> LongitudinalData:
    rng = np.random.default_rng(1)
    n = 400
    w1 = rng.binomial(1, 0.5, n)
    w2 = rng.integers(0, 4, n)
    a = rng.binomial(1, 0.5, n)
    time = np.full(n, 3)
    event = np.zeros(n, dtype=int)
    for i in range(n):
        for k in (1, 2, 3):
            if a[i] == 1 and k == 2:
                time[i], event[i] = 2, 1
                break
            if rng.uniform() < 0.2:
                time[i], event[i] = k, 1
                break
    frame = pd.DataFrame({"W1": w1, "W2": w2, "A": a, "time": time, "event": event})
    return LongitudinalData.from_time_to_event(
        frame, time="time", event="event", treatment="A", baseline=["W1", "W2"], grid=[1, 2, 3]
    )


def _fit(targeting: str) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LTMLE(
            REGIMENS,
            n_folds=1,
            horizons=list(HORIZONS),
            random_state=0,
            survival_model="hazard",
            targeting=targeting,
            outcome_learner=LogisticRegression(max_iter=1000),
            treatment_learner=LogisticRegression(max_iter=1000),
            censoring_learner=LogisticRegression(max_iter=1000),
        ).fit(_data())


@pytest.fixture(scope="module")
def one_step() -> Any:
    return _fit("one_step")


def _inputs(result: Any) -> Any:
    recipe = result.replay_recipe
    cumulative = [
        result.mechanism.cumulative(result.data, level, result.config.g_bounds)
        for level in recipe.hazard.levels
    ]
    return hazard_inputs(result.data, recipe.plans, cumulative, alpha=recipe.alpha)


def test_the_witness_cell_is_the_one_full_cell(one_step: Any) -> None:
    inputs = _inputs(one_step)
    arm = int(np.flatnonzero(np.asarray(one_step.data.treatment_levels[0]) == 1)[0])
    expected = np.zeros_like(inputs.full)
    expected[1, arm] = True
    assert np.array_equal(inputs.full, expected)
    assert not inputs.empty.any()


def test_the_walk_converges_with_the_cell_held_at_one(one_step: Any) -> None:
    record = next(iter(one_step.fits.values())).hazard
    assert record.converged
    inputs = _inputs(one_step)
    start = HazardState.from_hazards(
        one_step.initial_hazards.hazards, inputs.bounds, inputs.empty, inputs.full
    )
    held = inputs.full
    assert np.all(start.hazards[:, held].sum(axis=-1) == 1.0)
    assert np.all(start.survival[:, held] == 0.0)
    assert np.isfinite(walk_loss(start, inputs))
    moved = start.moved(np.ones_like(start.logits), inputs.bounds)
    assert np.all(moved.hazards[:, held].sum(axis=-1) == 1.0)


def test_m_a_clipped_cell_stalls_the_walk(one_step: Any) -> None:
    """Deliberate mutation: clip the full cell at the bound instead of holding it."""
    inputs = _inputs(one_step)
    record = next(iter(one_step.fits.values())).hazard
    clipped = HazardState.from_hazards(
        one_step.initial_hazards.hazards, inputs.bounds, inputs.empty
    )
    budget = 4 * record.n_iter
    _, mutated = walk_curve(
        clipped,
        inputs,
        HORIZONS,
        record.targets,
        step_size=one_step.replay_recipe.hazard.step_size,
        tol=record.tol,
        max_steps=budget,
        warn=False,
    )
    assert record.n_iter < budget
    assert not mutated.converged


def test_the_parameters_reading_the_cell_take_the_constant_node_status(one_step: Any) -> None:
    for label in REGIMENS:
        for horizon in HORIZONS:
            expected = (2,) if label == "a1" and horizon == 2 else ()
            assert constant_nodes(one_step.fits[f"{label} @ t={horizon}"]) == expected
            estimate = one_step[f"risk_regimen[{label} @ t={horizon}]"]
            assert (estimate.inference == "constant_node_plugin") == bool(expected)
    assert one_step["risk_regimen[a1 @ t=2]"].psi == 1.0


def test_a_per_horizon_fit_holds_the_cell_and_converges() -> None:
    result = _fit("iterative")
    for fit in result.fits.values():
        assert fit.hazard.converged
    assert constant_nodes(result.fits["a1 @ t=2"]) == (2,)
    assert constant_nodes(result.fits["a0 @ t=2"]) == ()


# ------------------------------------------------- a full cell with several causes (CDR-003)
#
# A full cell holds the all-cause hazard at one, which fixes the sum of the cause hazards and
# nothing else.  The logits must keep the fitted ratios between the causes.  The witness above
# has one cause, which takes the whole event whatever its logit, so it cannot see the split.

BOUNDS = (-7.6, 7.6)


def _full_cell(shares: list[float]) -> np.ndarray:
    """``(4, 1, 1, J)`` hazards summing to one, the shares rotated across the units."""
    base = np.asarray(shares, dtype=float)
    rows = [np.roll(base, shift) for shift in range(4)]
    return np.asarray(rows, dtype=float)[:, None, None, :]


@pytest.mark.parametrize("shares", [[0.2, 0.8], [0.1, 0.3, 0.6]])
def test_a_full_cell_keeps_the_cause_shares(shares: list[float]) -> None:
    hazards = _full_cell(shares)
    full = np.ones((1, 1), dtype=bool)
    state = HazardState.from_hazards(hazards, BOUNDS, full=full)
    np.testing.assert_allclose(state.hazards, hazards, rtol=0.0, atol=1e-12)
    assert np.all(state.survival == 0.0)
    # A common shift of the logits is the identity on a full cell.
    shifted = state.moved(np.full_like(state.logits, 3.0), BOUNDS)
    np.testing.assert_allclose(shifted.hazards, hazards, rtol=0.0, atol=1e-12)


def test_a_full_cell_keeps_an_empty_cause_at_zero() -> None:
    hazards = _full_cell([0.25, 0.0, 0.75])[:1]
    empty = np.array([[[False, True, False]]])
    state = HazardState.from_hazards(hazards, BOUNDS, empty=empty, full=np.ones((1, 1), bool))
    assert np.isneginf(state.logits[0, 0, 0, 1])
    np.testing.assert_allclose(state.hazards[0, 0, 0], [0.25, 0.0, 0.75], atol=1e-12)


def test_a_full_cell_keeps_a_small_share() -> None:
    """The largest cause sits at the upper bound, so a share far below the others survives.

    The fitted hazards sum to 0.9, as a learner's can in a full cell; the hold rescales them
    to one and keeps their ratio.
    """
    hazards = np.array([[[[1e-5, 0.9 - 1e-5]]]])
    state = HazardState.from_hazards(hazards, BOUNDS, full=np.ones((1, 1), dtype=bool))
    np.testing.assert_allclose(state.hazards, hazards / 0.9, rtol=1e-9, atol=0.0)


def _no_event_reference(
    hazards: np.ndarray, high: float, full: np.ndarray | None = None
) -> np.ndarray:
    """The logits before CDR-003: log(lambda_j / (1 - sum)) in every cell."""
    lam = np.asarray(hazards, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(np.log(lam) - np.log(1.0 - lam.sum(axis=-1))[..., None])


def test_m_a_no_event_reference_loses_the_shares(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deliberate mutation: the old no-event reference in from_hazards."""
    from cleverly.longitudinal import hazard as hazard_module

    monkeypatch.setattr(hazard_module, "softmax_logits", _no_event_reference)
    hazards = _full_cell([0.2, 0.8])
    mutated = HazardState.from_hazards(hazards, BOUNDS, full=np.ones((1, 1), dtype=bool))
    np.testing.assert_allclose(mutated.hazards, 0.5)
    result = _competing_fit("one_step")
    assert abs(result["cif_regimen[a0, one @ t=1]"].psi - 0.5) > 0.1


def _one_node_law() -> LongitudinalData:
    """Every unit has one of two causes at the one node; P(cause 1 | A, W) = 0.2 + 0.6 W.

    W and A are dependent (P(A = 1 | W) = 0.2 + 0.6 W), so a constant treatment mechanism is
    wrong.  E[lambda_1(1 | a, W)] = 0.5 in either arm, so every risk is 0.5 and every
    contrast zero.
    """
    rows: list[tuple[int, int, int, int]] = []
    for w, a, n in [(0, 0, 400), (0, 1, 100), (1, 0, 100), (1, 1, 400)]:
        first = int(n * (0.2 if w == 0 else 0.8))
        rows.extend([(w, a, 1, 1)] * first + [(w, a, 1, 2)] * (n - first))
    frame = pd.DataFrame(rows, columns=["W", "A", "time", "event"])
    return LongitudinalData.from_time_to_event(
        frame,
        time="time",
        event="event",
        treatment="A",
        baseline=["W"],
        grid=[1],
        causes={1: "one", 2: "two"},
    )


def _competing_fit(targeting: str) -> Any:
    from sklearn.dummy import DummyClassifier
    from sklearn.tree import DecisionTreeClassifier

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LTMLE(
            {"a0": 0, "a1": 1},
            n_folds=1,
            survival_model="hazard",
            targeting=targeting,
            outcome_learner=DecisionTreeClassifier(random_state=0),
            treatment_learner=DummyClassifier(strategy="prior"),
            censoring_learner=DummyClassifier(strategy="prior"),
            step_size=0.05,
        ).fit(_one_node_law())


@pytest.mark.parametrize("targeting", ["one_step", "iterative"])
def test_a_correct_cause_split_survives_a_wrong_treatment_mechanism(targeting: str) -> None:
    result = _competing_fit(targeting)
    inputs = _inputs(result)
    assert inputs.full.all()
    start = HazardState.from_hazards(
        result.initial_hazards.hazards, inputs.bounds, inputs.empty, inputs.full
    )
    np.testing.assert_allclose(start.hazards, result.initial_hazards.hazards, atol=1e-12)
    for fit in result.fits.values():
        assert fit.hazard.converged
    for arm in ("a0", "a1"):
        for cause in ("one", "two"):
            assert result[f"cif_regimen[{arm}, {cause} @ t=1]"].psi == pytest.approx(0.5, abs=1e-9)
    for cause in ("one", "two"):
        assert result[f"ate_regimen[a1 vs a0, {cause} @ t=1]"].psi == pytest.approx(0.0, abs=1e-9)


def test_the_likelihood_fallback_reads_a_full_cell_over_the_causes() -> None:
    """At the correct split every within-cell residual mean is zero, so the fallback's
    maximum-likelihood fluctuation is zero.  Deliberate mutation: the same logits read with a
    no-event category (``full=None``) give a nonzero fluctuation."""
    from dataclasses import replace

    from cleverly.longitudinal.hazard import _clever_observed, _newton

    result = _competing_fit("one_step")
    inputs = _inputs(result)
    start = HazardState.from_hazards(
        result.initial_hazards.hazards, inputs.bounds, inputs.empty, inputs.full
    )
    clever = _clever_observed(start, inputs, 1)
    epsilon = _newton(start, inputs, clever, 1e-12)
    np.testing.assert_allclose(epsilon, 0.0, atol=1e-10)
    mutated = _newton(replace(start, full=None), inputs, clever, 1e-12)
    assert np.max(np.abs(mutated)) > 1e-3


# ------------------------------------- the upper clip keeps the cause ratios in any cell
#
# Outside a full cell a unit can still have a fitted all-cause hazard near one.  When its
# largest cause logit passes the upper bound, every cause logit of that unit moves down by one
# constant, so the ratios survive and the all-cause hazard stays bounded below one.


def _near_full_unit() -> np.ndarray:
    """``(3, 1, 1, 2)`` hazards: no event mass left, almost none left, an ordinary unit."""
    return np.array([[[[0.2, 0.8]]], [[[0.2, 0.8 - 1e-6]]], [[[0.1, 0.3]]]])


def test_a_unit_near_one_keeps_its_cause_ratio() -> None:
    hazards = _near_full_unit()
    state = HazardState.from_hazards(hazards, BOUNDS)
    near, ordinary = state.hazards[0, 0, 0], state.hazards[2, 0, 0]
    assert near[1] / near[0] == pytest.approx(4.0, rel=1e-12)
    almost = state.hazards[1, 0, 0]
    assert almost[1] / almost[0] == pytest.approx(4.0 - 5e-6, rel=1e-9)
    assert state.logits[0, 0, 0].max() == pytest.approx(BOUNDS[1], abs=1e-12)
    assert near.sum() < 1.0
    assert 1.0 - near.sum() == pytest.approx(1.0 / (1.0 + 1.25 * np.exp(BOUNDS[1])), rel=1e-9)
    np.testing.assert_allclose(ordinary, [0.1, 0.3], atol=1e-12)
    # A step that pushes the logits up keeps the ratio at the bound.
    moved = state.moved(np.full_like(state.logits, 1.5), BOUNDS)
    stepped = moved.hazards[0, 0, 0]
    assert stepped[1] / stepped[0] == pytest.approx(4.0, rel=1e-12)
    assert moved.logits[0, 0, 0].max() == pytest.approx(BOUNDS[1], abs=1e-12)
    # A step along the cause shares moves the ratio by exactly that step.
    tilted = state.moved(np.array([0.0, 0.5])[None, None, None, :], BOUNDS).hazards[0, 0, 0]
    assert tilted[1] / tilted[0] == pytest.approx(4.0 * np.exp(0.5), rel=1e-12)


def test_the_shift_stops_at_the_lower_bound() -> None:
    hazards = np.array([[[[1e-8, 1.0 - 1e-8]]]])
    state = HazardState.from_hazards(hazards, BOUNDS)
    logits = state.logits[0, 0, 0]
    assert logits[1] == pytest.approx(BOUNDS[1], abs=1e-12)
    assert logits[0] == BOUNDS[0]


def test_one_cause_takes_the_plain_clip() -> None:
    """With one cause the shift is the clip, bit for bit, so single-cause fits are unchanged."""
    hazards = np.array([[[[0.99999]]], [[[1.0]]], [[[0.3]]], [[[1e-9]]]])
    state = HazardState.from_hazards(hazards, BOUNDS)
    with np.errstate(divide="ignore"):
        old = np.log(hazards) - np.log(1.0 - hazards.sum(axis=-1))[..., None]
    plain = np.clip(np.nan_to_num(old, posinf=BOUNDS[1], neginf=BOUNDS[0]), *BOUNDS)
    np.testing.assert_array_equal(state.logits, plain)
    logits = np.random.default_rng(0).normal(0.0, 6.0, (50, 3, 2, 1))
    np.testing.assert_array_equal(clip_logits(logits, BOUNDS), np.clip(logits, *BOUNDS))


def test_m_a_per_cause_upper_clip_loses_the_ratio(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deliberate mutation: clip each cause logit on its own in clip_logits, as before CDR-003."""
    from cleverly.longitudinal import hazard as hazard_module

    monkeypatch.setattr(
        hazard_module, "clip_logits", lambda logits, bounds, bounded=None: np.clip(logits, *bounds)
    )
    state = HazardState.from_hazards(_near_full_unit(), BOUNDS)
    almost = state.hazards[1, 0, 0]
    assert almost[1] / almost[0] == pytest.approx(1.0)
    moved = HazardState.from_hazards(_near_full_unit()[2:], BOUNDS).moved(
        np.full((1, 1, 1, 2), 10.0), BOUNDS
    )
    stepped = moved.hazards[0, 0, 0]
    assert stepped[1] / stepped[0] == pytest.approx(1.0)
