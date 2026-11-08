"""Exact-law checks of separable effects on a two-cause survival law (X91 part D).

The law of :mod:`tests.discrete_law_point_survival` with two causes is realised exactly by
its sample, so a fit with saturated cell learners starts at the truth.  The separable risk is
Stensrud et al. (2022) identification (8), computed by the law independently of the engine,
and its efficient influence curve is the complex-step Gateaux derivative of that formula.

Robustness is checked from a non-oracle start, with one target so that the targets cannot
saturate a factor's cells (design rev 2.1, R2): with the exact mechanism, the factor-wise
targeting recovers the truth when either the competing hazard or the event hazard among the
competing-free is wrong, and misses it when both are.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from cleverly.exceptions import CapabilityError, DataError
from cleverly.longitudinal import LTMLE, SDR, LongitudinalData, Separable
from cleverly.longitudinal.separable import (
    SeparableInputs,
    SeparableState,
    SeparableTarget,
    influence,
    scores,
    separable_direction,
    walk_loss,
    walk_separable,
)
from tests.discrete_law_point_survival import (
    CellMeans,
    CellProbabilities,
    G,
    _retained,
    static,
    survival_law,
)

RELAPSE, DEATH = 1, 2
HORIZONS = (1, 2, 3)
NO_TRUNCATION = (1e-9, 1.0)
#: How far a mutated or both-wrong result must sit from the truth.
FLOOR = 1e-3
#: Separable arms by event code: death's component at level 0, relapse's at level 1.
ARMS = {DEATH: 0, RELAPSE: 1}
SEPARABLE = Separable(arms={"death": 0, "relapse": 1}, first="death")


def _law() -> Any:
    return survival_law(causes=2)


def _data() -> LongitudinalData:
    law = _law()
    return LongitudinalData.from_frame(law.frame(), **law.fit_columns())


def _ltmle(**kwargs: Any) -> LTMLE:
    settings: dict[str, Any] = {
        "survival_model": "hazard",
        "targeting": "one_step",
        "n_folds": 1,
        "g_bounds": NO_TRUNCATION,
        "outcome_learner": CellMeans(),
        "treatment_learner": CellProbabilities(),
        "censoring_learner": CellMeans(),
        **kwargs,
    }
    regimens = settings.pop("regimens", {"a0": 0, "a1": 1, "sep": SEPARABLE})
    return LTMLE(regimens, **settings)


def _fit(**kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _ltmle(**kwargs).fit(_data())


@pytest.fixture(scope="module")
def fit() -> Any:
    return _fit()


def _name(label: str, cause: str, horizon: int) -> str:
    return f"cif_regimen[{label}, {cause} @ t={horizon}]"


CAUSES = (("relapse", RELAPSE), ("death", DEATH))


# --------------------------------------------------------------- identities


@pytest.mark.parametrize("horizon", HORIZONS)
@pytest.mark.parametrize(("cause", "code"), CAUSES)
def test_the_estimate_is_the_separable_g_formula(
    fit: Any, cause: str, code: int, horizon: int
) -> None:
    law = _law()
    truth = law.separable(law.probs, ARMS, DEATH, horizon, code)
    assert fit.estimates[_name("sep", cause, horizon)].psi == pytest.approx(truth, abs=1e-12)


@pytest.mark.parametrize("horizon", HORIZONS)
@pytest.mark.parametrize(("cause", "code"), CAUSES)
def test_the_influence_curve_is_the_gateaux_derivative(
    fit: Any, cause: str, code: int, horizon: int
) -> None:
    law = _law()
    eif = law.separable_eif(ARMS, DEATH, horizon, code)
    curve = fit.estimates[_name("sep", cause, horizon)].influence_curve[law.first_row_of()]
    np.testing.assert_allclose(curve, eif, atol=1e-12)


@pytest.mark.parametrize("arm", (0, 1))
@pytest.mark.parametrize(("cause", "code"), CAUSES)
def test_the_static_arms_of_a_separable_fit_are_the_cause_incidences(
    fit: Any, arm: int, cause: str, code: int
) -> None:
    """The static arms go through the factor engine as separable plans with equal levels."""
    law = _law()
    for horizon in HORIZONS:
        truth = law.functional(law.probs, static(arm), horizon, code)
        estimate = fit.estimates[_name(f"a{arm}", cause, horizon)]
        assert estimate.psi == pytest.approx(truth, abs=1e-12)
        eif = law.eif(static(arm), horizon, code)
        np.testing.assert_allclose(estimate.influence_curve[law.first_row_of()], eif, atol=1e-12)


def test_equal_levels_reduce_to_the_static_incidence() -> None:
    """A Separable whose two arms agree is the static arm, through the separable engine."""
    same = Separable(arms={"death": 1, "relapse": 1}, first="death")
    result = _fit(regimens={"a0": 0, "same": same})
    law = _law()
    for cause, code in CAUSES:
        for horizon in HORIZONS:
            truth = law.functional(law.probs, static(1), horizon, code)
            assert result.estimates[_name("same", cause, horizon)].psi == pytest.approx(
                truth, abs=1e-12
            )


def test_the_declared_order_is_part_of_the_estimand() -> None:
    """Counting relapse first gives the other functional, which differs at different arms."""
    other = Separable(arms={"death": 0, "relapse": 1}, first="relapse")
    result = _fit(regimens={"a0": 0, "sep": other})
    law = _law()
    for horizon in (2, 3):
        declared = law.separable(law.probs, ARMS, RELAPSE, horizon, RELAPSE)
        assert result.estimates[_name("sep", "relapse", horizon)].psi == pytest.approx(
            declared, abs=1e-12
        )
        assert abs(declared - law.separable(law.probs, ARMS, DEATH, horizon, RELAPSE)) > FLOOR


def test_the_separable_risk_differs_from_both_static_risks(fit: Any) -> None:
    """The nonzero witness: the cross-arm composition moves the risk off both static arms."""
    for horizon in (2, 3):
        separable = fit.estimates[_name("sep", "relapse", horizon)].psi
        for arm in ("a0", "a1"):
            assert abs(separable - fit.estimates[_name(arm, "relapse", horizon)].psi) > FLOOR


def test_a_one_step_separable_curve_is_monotone(fit: Any) -> None:
    for cause, _ in CAUSES:
        values = [fit.estimates[_name("sep", cause, t)].psi for t in HORIZONS]
        assert values == sorted(values)


# --------------------------------------------------- the engine from a wrong start


def _engine() -> tuple[Any, SeparableInputs, SeparableState]:
    """The engine's inputs at the exact mechanism, and the exact factors."""
    law = _law()
    frame = law.frame()
    n, n_times, n_levels = len(frame), 3, 3
    w = frame["W"].to_numpy(dtype=int)
    a = frame["A"].to_numpy(dtype=int)
    hazards = np.zeros((n, n_times, n_levels, 2))
    for ww in (0, 1):
        for level in range(n_levels):
            for node in range(1, n_times + 1):
                for code in (RELAPSE, DEATH):
                    value = law._node_hazard(law.probs, ww, level, node, code)
                    hazards[w == ww, node - 1, level, code - 1] = value
    inverse = np.zeros((n, n_times, n_levels))
    for level in range(n_levels):
        cumulative = G[w, level].copy()
        for node in range(1, n_times + 1):
            cumulative = cumulative * np.array([_retained(node, x, level, 0) for x in w])
            inverse[:, node - 1, level] = 1.0 / cumulative
    at_risk = np.zeros((n, n_times), dtype=bool)
    alive = np.ones(n, dtype=bool)
    event_d = np.zeros((n, n_times))
    event_y = np.zeros((n, n_times))
    for k in range(n_times):
        at_risk[:, k] = alive & (frame[f"C{k + 1}"].to_numpy() == 1)
        died = np.nan_to_num(frame[f"D{k + 1}"].to_numpy()) == 1
        relapsed = np.nan_to_num(frame[f"R{k + 1}"].to_numpy()) == 1
        event_d[:, k] = np.where(at_risk[:, k], died, 0.0)
        event_y[:, k] = np.where(at_risk[:, k], relapsed, 0.0)
        alive = at_risk[:, k] & ~died & ~relapsed
    inputs = SeparableInputs(at_risk, event_d, event_y, a, inverse, np.ones(n), 1.0 - 1e-6)
    none = np.zeros((n_times, n_levels), dtype=bool)
    state = SeparableState.from_hazards(hazards, DEATH - 1, inputs.bounds, (none,) * 4)
    return law, inputs, state


#: One target: relapse at the last horizon, so no factor's cells saturate (R2).
TARGET = [SeparableTarget("sep", (0, 1), "event", "relapse", 3)]


def _shift(frame_w: np.ndarray, n_times: int, by_level: tuple[float, float, float]) -> np.ndarray:
    """A logit shift that depends on ``W`` and the level, the same at every node."""
    per_unit = np.where(frame_w == 0, 0.6, -0.5)
    return np.stack([per_unit] * n_times, axis=1)[:, :, None] * np.array(by_level)[None, None]


def _targeted(shift_d: Any, shift_y: Any) -> tuple[float, float, int]:
    law, inputs, exact = _engine()
    start = SeparableState(exact.beta_d + shift_d, exact.beta_y + shift_y)
    targeted, record = walk_separable(start, inputs, TARGET, step_size=0.05, tol=1e-12)
    assert record.converged
    # Fewer targets than a factor's (W, node) cells at a level, so no check passes because
    # the scores pin every cell (R2).
    assert len(TARGET) < 2 * 3
    truth = law.separable(law.probs, ARMS, DEATH, 3, RELAPSE)
    before, _ = influence(start, inputs, TARGET)
    after, _ = influence(targeted, inputs, TARGET)
    return float(before[0] - truth), float(after[0] - truth), record.n_iter


def _shifts() -> tuple[np.ndarray, np.ndarray]:
    law = _law()
    w = law.frame()["W"].to_numpy(dtype=int)
    return _shift(w, 3, (1.0, -1.0, 0.5)), _shift(w, 3, (0.8, 1.0, -1.0))


def test_a_wrong_event_hazard_is_corrected_by_the_targeting() -> None:
    """lambda_D right, h_Y wrong at every level, exact mechanism: the targeted risk is true."""
    _, shift_y = _shifts()
    start_bias, bias, _ = _targeted(0.0, shift_y)
    assert abs(start_bias) > 5 * FLOOR
    assert abs(bias) < 1e-10


def test_a_wrong_competing_hazard_is_corrected_by_the_targeting() -> None:
    """h_Y right, lambda_D wrong at every level, exact mechanism: the targeted risk is true."""
    shift_d, _ = _shifts()
    start_bias, bias, _ = _targeted(shift_d, 0.0)
    assert abs(start_bias) > 5 * FLOOR
    assert abs(bias) < 1e-10


def test_both_hazards_wrong_miss_the_truth() -> None:
    """The control: with both factors wrong the remainder is not zero (design section 2a)."""
    shift_d, shift_y = _shifts()
    _, bias, _ = _targeted(shift_d, shift_y)
    assert abs(bias) > FLOOR


def test_one_coefficient_shared_by_both_factors_loses_the_robustness() -> None:
    """The mutation: one coefficient per target across the two factors, as a softmax
    fluctuation has, solves only the summed score, and misses with h_Y wrong.

    The miss is second order on this law (about 6e-5), five orders above the factor-wise
    targeting's miss from the same start.
    """
    law, inputs, exact = _engine()
    _, shift_y = _shifts()
    state = SeparableState(exact.beta_d, exact.beta_y + shift_y)
    for _ in range(200):
        score, _ = scores(state, inputs, TARGET)
        total = float(score.sum())
        if abs(total) < 1e-14:
            break
        delta_d, delta_y = separable_direction(state, inputs, TARGET, np.ones(2))
        probe = state.moved(1e-6 * delta_d, 1e-6 * delta_y, inputs.bounds)
        slope = (float(scores(probe, inputs, TARGET)[0].sum()) - total) / 1e-6
        state = state.moved(-total / slope * delta_d, -total / slope * delta_y, inputs.bounds)
    assert abs(float(scores(state, inputs, TARGET)[0].sum())) < 1e-12
    truth = law.separable(law.probs, ARMS, DEATH, 3, RELAPSE)
    assert abs(influence(state, inputs, TARGET)[0][0] - truth) > 1e-5


def test_the_walk_loss_falls_at_the_rate_of_the_score() -> None:
    """At an unsolved state the loss derivative along the direction of c is -n c . s."""
    _, inputs, exact = _engine()
    shift_d, shift_y = _shifts()
    state = SeparableState(exact.beta_d + shift_d, exact.beta_y + shift_y)
    score, _ = scores(state, inputs, TARGET)
    for coefficients in (np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.3, -0.7])):
        delta_d, delta_y = separable_direction(state, inputs, TARGET, coefficients)
        step = 1e-6
        up = walk_loss(state.moved(step * delta_d, step * delta_y, inputs.bounds), inputs)
        down = walk_loss(state.moved(-step * delta_d, -step * delta_y, inputs.bounds), inputs)
        slope = (up - down) / (2 * step)
        assert slope == pytest.approx(-inputs.n * float(coefficients @ score), rel=1e-5)


# ------------------------------------------------------------------- refusals


def test_other_estimators_refuse_a_separable_regimen() -> None:
    with pytest.raises(CapabilityError, match="only LTMLE"):
        SDR({"a0": 0, "sep": SEPARABLE}, n_folds=1).fit(_data())


def test_the_iterated_mean_survival_model_refuses_a_separable_regimen() -> None:
    with pytest.raises(CapabilityError, match="only LTMLE\\(survival_model='hazard'\\)"):
        _fit(survival_model="iterated_mean", targeting="iterative")


def test_the_per_horizon_algorithm_redirects_to_one_step() -> None:
    with pytest.raises(CapabilityError, match="targeting='one_step'"):
        _fit(targeting="iterative")


def test_a_separable_fit_needs_a_static_reference() -> None:
    with pytest.raises(DataError, match="at least one static arm"):
        _fit(regimens={"sep": SEPARABLE})


def test_one_cause_is_refused() -> None:
    law = survival_law()
    data = LongitudinalData.from_frame(law.frame(), **law.fit_columns())
    with (
        warnings.catch_warnings(),
        pytest.raises(DataError, match="exactly two causes"),
    ):
        warnings.simplefilter("ignore")
        _ltmle(regimens={"a0": 0, "sep": Separable(arms={"x": 0, "y": 1}, first="x")}).fit(data)


@pytest.mark.parametrize(
    ("declared", "match"),
    [
        (Separable(arms={"death": 0, "progression": 1}, first="death"), "panel's causes"),
        (Separable(arms={"death": 0, "relapse": 7}, first="death"), "does not take"),
    ],
)
def test_an_arm_that_does_not_match_the_panel_is_refused(declared: Separable, match: str) -> None:
    with pytest.raises(DataError, match=match):
        _fit(regimens={"a0": 0, "sep": declared})


def test_a_malformed_declaration_is_refused() -> None:
    with pytest.raises(DataError, match="two causes"):
        Separable(arms={"death": 0}, first="death")
    with pytest.raises(DataError, match="must name one of the causes"):
        Separable(arms={"death": 0, "relapse": 1}, first="progression")


def test_two_orders_in_one_fit_are_refused() -> None:
    other = Separable(arms={"death": 1, "relapse": 0}, first="relapse")
    with pytest.raises(CapabilityError, match="same first cause"):
        _fit(regimens={"a0": 0, "sep": SEPARABLE, "other": other})


def test_eliminate_is_refused() -> None:
    law = _law()
    with warnings.catch_warnings(), pytest.raises(CapabilityError, match="eliminate="):
        warnings.simplefilter("ignore")
        _ltmle().fit(law.frame(), eliminate=["death"], **law.fit_columns())


@pytest.mark.parametrize(
    "setting",
    [
        {"stopping": "scaled_var"},
        {"iterate": "best"},
        {"n_bootstrap": 20},
        {"node_bounds": {"death": {1: (0.0, 0.9)}}},
    ],
    ids=["stopping", "iterate", "bootstrap", "node_bounds"],
)
def test_options_without_a_factor_wise_form_are_refused(setting: dict[str, Any]) -> None:
    with pytest.raises(CapabilityError, match="not built for a fit with a Separable"):
        _fit(**setting)
