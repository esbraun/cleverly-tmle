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
from cleverly.longitudinal.hazard import HazardState, hazard_inputs, walk_curve, walk_loss
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
