"""Nonzero witnesses for composite clipping and stratum correction identities."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import cleverly.validation.drtmle as diagnostic_module
from cleverly.estimators._nuisance import Propensity
from cleverly.estimators.composite import (
    composite_clipping_mask,
    composite_state,
    targeting_inputs,
)
from cleverly.validation.drtmle import MARGIN_ACTIVE, _marginal_scores, correction_check
from tests.unit.test_composite_missing_data import _oracle_fit
from tests.unit.test_stratified_drtmle_exact import CASES, fit, rows


def test_composite_mask_uses_the_binary_complement_clipping_rule() -> None:
    nuisance = SimpleNamespace(
        propensity=Propensity(np.array([[0.4 + 1e-12, 0.6]]), (0.0, 1.0)),
        missingness=np.array([[0.7, 0.8]]),
        treatment_observation=np.array([0.9]),
    )
    assert not composite_clipping_mask(nuisance, (0.1, 0.8), 0.01).any()
    # At asymmetric bounds only the upper-arm probability decides binary clipping.
    nuisance.propensity = Propensity(np.array([[0.9, 0.1]]), (0.0, 1.0))
    assert not composite_clipping_mask(nuisance, (0.05, 0.8), 0.01).any()


def test_composite_factor_clipping_counts_overlapping_cells_once() -> None:
    nuisance = SimpleNamespace(
        propensity=Propensity(np.array([[0.005, 0.995], [0.4, 0.6]]), (0.0, 1.0)),
        missingness=np.array([[0.005, 0.8], [0.005, 0.8]]),
        treatment_observation=np.array([0.005, 0.9]),
    )
    np.testing.assert_array_equal(
        composite_clipping_mask(nuisance, (0.01, 0.99), 0.01),
        [[True, True], [True, False]],
    )


@pytest.mark.parametrize("factor", ("propensity", "missingness", "treatment_observation"))
def test_composite_initial_clipping_survives_the_product(factor: str) -> None:
    result = _oracle_fit("two")
    original = correction_check(result, tolerance=1e-6)
    assert original.contract == "theorem"
    assert original.margin > MARGIN_ACTIVE and original.gr1_margin > MARGIN_ACTIVE
    nuisance = result.nuisance
    if factor == "propensity":
        values = np.asarray(nuisance.propensity.values).copy()
        values[0] = [0.005, 0.995]
        nuisance = replace(nuisance, propensity=Propensity(values, nuisance.arms))
    else:
        values = np.asarray(getattr(nuisance, factor)).copy()
        values[0] = 0.005
        nuisance = replace(nuisance, **{factor: values})
    result = replace(
        result,
        repeats=(replace(result.repeats[0], nuisance=nuisance),),
    )
    state = composite_state(
        result.data,
        nuisance,
        g_bounds=result.config.g_bounds,
        nuisance_bound=result.config.missingness_bound,
    )
    # The old diagnostic sees an interior, already-bounded product in every cell.
    assert np.all(state.nuisance.propensity.values >= state.bounds[0])
    assert np.all(state.nuisance.propensity.values <= state.bounds[1])
    check = correction_check(result, tolerance=1e-6)
    # Exactly one row changes both arm denominators; the known count is independent
    # of the production helper used by correction_check.
    assert {row.initial_clipped for row in check.rows} == {2}
    assert check.margin > MARGIN_ACTIVE and check.gr1_margin > MARGIN_ACTIVE
    assert "g-hat at the initial fit" in check.truncations_active
    assert check.contract == "bound-active"


@pytest.mark.parametrize("arm_major", (False, True))
def test_stratum_score_reconstruction_uses_weighted_masses(arm_major: bool) -> None:
    data = SimpleNamespace(
        has_strata=True,
        strata=np.array([0, 0, 1]),
        n_strata=2,
        weights=np.array([1.0, 2.0, 3.0]),
    )
    by_arm = np.array([[1.0, 3.0], [2.0, 8.0], [4.0, 12.0]])
    scores = (by_arm if arm_major else by_arm.T).reshape(-1)
    np.testing.assert_array_equal(
        _marginal_scores(scores, data, 3, arm_major=arm_major), [2.0, 5.0, 8.0]
    )


@pytest.fixture(scope="module")
def stratum_frame() -> Any:
    return rows()


def _nonzero_returned_state(result: Any) -> Any:
    """A hypothetical nonzero QR state with independently recorded block scores.

    Saturated Q/g solves can close all three equations in one round. To exercise
    the reader there, perturb QR alone on the original fit's treatment, mechanism,
    predictions and strata. QR enters equation (9), not equation (10).
    """
    repeat = result.repeats[0]
    fluctuation = repeat.fluctuations["mean"]
    reduction = fluctuation.reduction
    mechanism = fluctuation.mechanism
    data, _ = targeting_inputs(result, repeat.nuisance)
    raw_g = np.asarray(mechanism.propensity)
    arm_major = raw_g.ndim == 2
    if arm_major:
        g = np.clip(raw_g, *reduction.bounds)
    else:
        one = np.clip(raw_g, *reduction.bounds)
        g = np.column_stack([1.0 - one, one])
    indicators = np.column_stack([data.treatment == arm for arm in reduction.reduced.arms])
    residual = indicators - g
    # Distinct perturbations make every arm and stratum live, within residual bounds.
    arm_scale = 0.05 * (1.0 + np.arange(g.shape[1]))
    qr = (1.0 + data.strata[:, None]) * arm_scale * residual
    reduced = replace(reduction.reduced, qr=qr)
    # Equation (9): D_g,a = (QR_a / g_a) (C_a - g_a).
    contributions = qr / g * residual
    block_scores = []
    for code in range(data.n_strata):
        inside = data.strata == code
        probability = np.average(inside, weights=data.weights)
        block_scores.append(
            np.mean(data.weights[:, None] * contributions * inside[:, None] / probability, axis=0)
        )
    matrix = np.asarray(block_scores)
    recorded = (matrix.T if arm_major else matrix).reshape(-1)
    fluctuation = replace(
        fluctuation,
        reduction=replace(reduction, reduced=reduced),
        mechanism=replace(mechanism, score=recorded),
    )
    return replace(result, repeats=(replace(repeat, fluctuations={"mean": fluctuation}),))


@pytest.mark.parametrize("route", ("complete", "missing_randomized"))
@pytest.mark.parametrize("upper_arm, clipped", ((0.1, False), (0.01, True)))
def test_binary_initial_clipping_uses_the_complement_rule(
    stratum_frame: Any, route: str, upper_arm: float, clipped: bool
) -> None:
    result = fit(stratum_frame, route)
    repeat = result.repeats[0]
    values = np.tile([1.0 - upper_arm + 1e-12, upper_arm], (result.data.n, 1))
    nuisance = replace(repeat.nuisance, propensity=Propensity(values, (0.0, 1.0)))
    fluctuation = repeat.fluctuations["mean"]
    fluctuation = replace(fluctuation, reduction=replace(fluctuation.reduction, bounds=(0.05, 0.8)))
    result = replace(
        result, repeats=(replace(repeat, nuisance=nuisance, fluctuations={"mean": fluctuation}),)
    )
    check = correction_check(result, tolerance=1e-6)
    assert {row.initial_clipped for row in check.rows} == {2 * result.data.n if clipped else 0}


@pytest.mark.parametrize(("route", "guard", "reduction"), CASES)
def test_nonzero_stratum_correction_scores_match_the_reported_means(
    monkeypatch: pytest.MonkeyPatch,
    stratum_frame: Any,
    route: str,
    guard: tuple[str, ...],
    reduction: str,
) -> None:
    result = fit(
        stratum_frame,
        route,
        guard=guard,
        reduction=reduction,
        max_outer=1,
    )
    if guard == ("Q", "g") and route not in ("missing_randomized", "missing_known_three_arm"):
        result = _nonzero_returned_state(result)
    check = correction_check(result, tolerance=1e-6)
    assert check.rows
    # Unsaturated capped fits and explicitly perturbed states both leave live scores.
    assert max(abs(row.reported) for row in check.rows) > 1e-8
    assert check.identity_failures() == ()
    if guard == ("g",):
        # Only the unsolved D_g term is live in this fixture. Identity failures
        # intentionally judge solved terms, so it cannot observe the old indexing.
        return
    monkeypatch.setattr(
        diagnostic_module,
        "_marginal_scores",
        lambda scores, data, arms, **kwargs: np.asarray(scores)[:arms],
    )
    assert correction_check(result, tolerance=1e-6).identity_failures()
