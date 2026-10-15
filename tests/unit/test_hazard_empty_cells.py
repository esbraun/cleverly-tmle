"""A hazard cell whose followers hold no event is held at zero (X14).

The witness sample is ``make_point_survival(n=400, seed=36, hazard_scale={3: 0.05})``: arm 1
has followers at risk at node 3 and none of them has the event.  The joint score equations
then need that hazard below the clip floor, so a walk that clips it never converges.  The
fit holds the hazard at exactly zero, its maximum-likelihood value, and every parameter that
reads the cell takes the ``"constant_node_plugin"`` status, as an iterated-mean node with a
constant target does.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from cleverly.datasets import make_point_survival
from cleverly.longitudinal import LTMLE
from cleverly.longitudinal.hazard import HazardState, hazard_inputs, loglik, walk_curve
from cleverly.longitudinal.sequential import constant_nodes

COLUMNS = {
    "outcome": [f"Y{k}" for k in range(1, 6)],
    "treatment": "A",
    "baseline": ["W1", "W2"],
    "censoring": [f"C{k}" for k in range(1, 6)],
}
REGIMENS = {"a1": 1, "a0": 0}
HORIZONS = (1, 2, 3, 4, 5)


def _fit(targeting: str) -> Any:
    frame, _ = make_point_survival(n=400, seed=36, layout="wide", hazard_scale={3: 0.05})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LTMLE(
            REGIMENS,
            n_folds=1,
            random_state=0,
            survival_model="hazard",
            targeting=targeting,
            outcome_learner=LogisticRegression(max_iter=1000),
            treatment_learner=LogisticRegression(max_iter=1000),
            censoring_learner=LogisticRegression(max_iter=1000),
        ).fit(frame, **COLUMNS)


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


def test_the_witness_cell_is_the_one_empty_cell(one_step: Any) -> None:
    inputs = _inputs(one_step)
    arm = int(np.flatnonzero(np.asarray(one_step.data.treatment_levels[0]) == 1)[0])
    expected = np.zeros_like(inputs.empty)
    expected[2, arm, 0] = True
    assert np.array_equal(inputs.empty, expected)
    assert not inputs.full.any()


def test_the_walk_converges_with_the_cell_held_at_zero(one_step: Any) -> None:
    record = next(iter(one_step.fits.values())).hazard
    assert record.converged
    inputs = _inputs(one_step)
    start = HazardState.from_hazards(one_step.initial_hazards.hazards, inputs.bounds, inputs.empty)
    assert np.all(start.hazards[:, inputs.empty] == 0.0)
    assert np.isfinite(loglik(start, inputs))
    moved = start.moved(np.ones_like(start.logits), inputs.bounds)
    assert np.all(moved.hazards[:, inputs.empty] == 0.0)


def test_m_a_clipped_cell_stalls_the_walk(one_step: Any) -> None:
    """Deliberate mutation: clip the empty cell at the floor instead of holding it."""
    inputs = _inputs(one_step)
    record = next(iter(one_step.fits.values())).hazard
    clipped = HazardState.from_hazards(one_step.initial_hazards.hazards, inputs.bounds)
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
            expected = (3,) if label == "a1" and horizon >= 3 else ()
            assert constant_nodes(one_step.fits[f"{label} @ t={horizon}"]) == expected
            estimate = one_step[f"risk_regimen[{label} @ t={horizon}]"]
            assert (estimate.inference == "constant_node_plugin") == bool(expected)
            assert np.isfinite(estimate.plugin_std_error)


def test_a_per_horizon_fit_holds_the_cell_and_converges() -> None:
    result = _fit("iterative")
    for fit in result.fits.values():
        assert fit.hazard.converged
    assert constant_nodes(result.fits["a1 @ t=3"]) == (3,)
    assert constant_nodes(result.fits["a1 @ t=2"]) == ()
    assert constant_nodes(result.fits["a0 @ t=5"]) == ()
