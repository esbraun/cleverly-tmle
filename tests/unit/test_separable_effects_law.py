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
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest

import cleverly.longitudinal.separable as separable_module
from cleverly._support import classify_key
from cleverly._support_rows import LONGITUDINAL_ROWS
from cleverly.exceptions import CapabilityError, DataError
from cleverly.fluctuation._score import relative_score
from cleverly.longitudinal import LTMLE, SDR, DynamicRegimen, LongitudinalData, Separable
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


def _fit(data: LongitudinalData | None = None, **kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _ltmle(**kwargs).fit(_data() if data is None else data)


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


#: The three phases of the targeting: the walk alone, the iterative TMLE from the start (the
#: handover forced at once), and Newton steps on the full score Jacobian from the start (the
#: iterative TMLE skipped too), so each robustness witness checks every phase.
FALLBACKS = ["iterative", "jacobian"]
PHASES = pytest.mark.parametrize("phase", ["walk", *FALLBACKS])


def _phase(monkeypatch: pytest.MonkeyPatch, phase: str) -> None:
    if phase in FALLBACKS:
        monkeypatch.setattr(separable_module, "_STALL", 0)
    if phase == "jacobian":
        monkeypatch.setattr(separable_module, "_ITERATIVE_STEPS", 0)


@PHASES
def test_a_wrong_event_hazard_is_corrected_by_the_targeting(
    monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    """lambda_D right, h_Y wrong at every level, exact mechanism: the targeted risk is true."""
    _phase(monkeypatch, phase)
    _, shift_y = _shifts()
    start_bias, bias, _ = _targeted(0.0, shift_y)
    assert abs(start_bias) > 5 * FLOOR
    assert abs(bias) < 1e-10


@PHASES
def test_a_wrong_competing_hazard_is_corrected_by_the_targeting(
    monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    """h_Y right, lambda_D wrong at every level, exact mechanism: the targeted risk is true."""
    _phase(monkeypatch, phase)
    shift_d, _ = _shifts()
    start_bias, bias, _ = _targeted(shift_d, 0.0)
    assert abs(start_bias) > 5 * FLOOR
    assert abs(bias) < 1e-10


@PHASES
def test_both_hazards_wrong_miss_the_truth(monkeypatch: pytest.MonkeyPatch, phase: str) -> None:
    """The control: with both factors wrong the remainder is not zero (design section 2a)."""
    _phase(monkeypatch, phase)
    shift_d, shift_y = _shifts()
    _, bias, _ = _targeted(shift_d, shift_y)
    assert abs(bias) > FLOOR


@pytest.mark.parametrize("phase", FALLBACKS)
def test_a_fallback_record_describes_the_reported_state(
    monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    """After a fallback phase the trace ends at the reported state and counts its steps."""
    _phase(monkeypatch, phase)
    _, inputs, exact = _engine()
    shift_d, _ = _shifts()
    start = SeparableState(exact.beta_d + shift_d, exact.beta_y)
    targeted, record = walk_separable(start, inputs, TARGET, step_size=0.05, tol=1e-12)
    score, scale = scores(targeted, inputs, TARGET)
    assert record.trace[-1] == pytest.approx(relative_score(score, scale), rel=1e-12)
    assert len(record.trace) == record.n_iter + 1
    assert record.converged and record.failure is None


@PHASES
def test_every_phase_moves_each_factor_only_at_its_own_level(
    monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    """The target reads lambda_D at a_D = 0 and h_Y at a_Y = 1: every phase moves those two
    factors (the nonzero witness) and leaves lambda_D at levels 1 and 2 and h_Y at levels 0 and
    2 exactly where they started, so the reported state lies in the factor-wise submodel."""
    _phase(monkeypatch, phase)
    _, inputs, exact = _engine()
    shift_d, shift_y = _shifts()
    start = SeparableState(exact.beta_d + shift_d, exact.beta_y + shift_y)
    targeted, record = walk_separable(start, inputs, TARGET, step_size=0.05, tol=1e-12)
    assert record.converged
    assert np.max(np.abs(targeted.beta_d[:, :, 0] - start.beta_d[:, :, 0])) > 1e-3
    assert np.max(np.abs(targeted.beta_y[:, :, 1] - start.beta_y[:, :, 1])) > 1e-3
    np.testing.assert_array_equal(targeted.beta_d[:, :, 1:], start.beta_d[:, :, 1:])
    np.testing.assert_array_equal(targeted.beta_y[:, :, [0, 2]], start.beta_y[:, :, [0, 2]])


@pytest.mark.parametrize("phase", FALLBACKS)
def test_the_fallbacks_read_observation_weights_as_duplicated_rows(
    monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    """A weight of two on the units at level 1 targets as those units listed twice, through
    each fallback: the iterative TMLE is the first code to fit a likelihood with the weights.

    The walk is held in place, so both fallbacks start from the same state: its fixed
    :math:`\\Sigma` reads the weights as sampling weights, which the duplicated panel does not.
    """
    _phase(monkeypatch, phase)

    def still(state: SeparableState, *_: object) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros_like(state.beta_d), np.zeros_like(state.beta_y)

    monkeypatch.setattr(separable_module, "separable_direction", still)
    _, inputs, exact = _engine()
    shift_d, shift_y = _shifts()
    start = SeparableState(exact.beta_d + shift_d, exact.beta_y + shift_y)
    twice = np.flatnonzero(inputs.observed == 1)
    weights = np.where(inputs.observed == 1, 2.0, 1.0)
    weighted = replace(inputs, weights=weights / weights.mean())
    order = np.concatenate([np.arange(inputs.n), twice])
    duplicated = replace(
        inputs,
        at_risk=inputs.at_risk[order],
        event_d=inputs.event_d[order],
        event_y=inputs.event_y[order],
        observed=inputs.observed[order],
        inverse=inputs.inverse[order],
        weights=np.ones(order.size),
    )
    start_twice = SeparableState(start.beta_d[order], start.beta_y[order])
    one, record_one = walk_separable(start, weighted, TARGET, step_size=0.05, tol=1e-10)
    two, record_two = walk_separable(start_twice, duplicated, TARGET, step_size=0.05, tol=1e-10)
    assert record_one.converged and record_two.converged
    assert record_one.phase == record_two.phase == phase
    assert influence(one, weighted, TARGET)[0][0] == pytest.approx(
        influence(two, duplicated, TARGET)[0][0], abs=1e-10
    )


def test_a_cell_that_starts_at_the_clip_moves_in_when_its_rows_hold_events() -> None:
    """A start at the lower clip, as an out-of-fold start gives when the training fold held no
    death in a cell, leaves the clip in one iterative-TMLE fit when the cell's own rows hold
    deaths: the targets span lambda_D's cells at level 0, so the cell reaches its empirical
    hazard.  A fit that read the clipped likelihood gives those rows no gradient and leaves the
    cell at the clip, as on the pilot draw that stopped at a relative score of 5e-3."""
    law, inputs, exact = _engine()
    w = law.frame()["W"].to_numpy(dtype=int)
    rows = (w == 0) & (inputs.observed == 0) & inputs.at_risk[:, 2]
    hazard = inputs.event_d[rows, 2].mean()
    assert 0.0 < hazard < 1.0
    beta_d = exact.beta_d.copy()
    beta_d[w == 0, 2, 0] = inputs.bounds[0]
    state = SeparableState(beta_d, exact.beta_y)
    targets = _saturating_targets()
    covariates = [separable_module._covariates(state, inputs, target) for target in targets]
    delta = separable_module._factor_fit(state, inputs, targets, 0, covariates)
    moved = state.moved(delta, np.zeros_like(delta), inputs.bounds)
    np.testing.assert_allclose(moved.lam_d[w == 0, 2, 0], hazard, rtol=1e-6)


@pytest.mark.parametrize("phase", FALLBACKS)
def test_a_held_cell_stays_held_through_the_fallbacks(
    monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    """With no death at the last node at level 0, that cell is held at a hazard of zero: each
    fallback leaves it there and still converges."""
    _phase(monkeypatch, phase)
    _, inputs, exact = _engine()
    event_d = inputs.event_d.copy()
    event_d[inputs.observed == 0, 2] = 0.0
    emptied = replace(inputs, event_d=event_d)
    none = np.zeros((3, 3), dtype=bool)
    empty_d = none.copy()
    empty_d[2, 0] = True
    hazards = np.stack([exact.h_y * (1.0 - exact.lam_d), exact.lam_d], axis=3)
    held = SeparableState.from_hazards(
        hazards, DEATH - 1, emptied.bounds, (empty_d, none, none, none)
    )
    shift_d, shift_y = _shifts()
    start = SeparableState(held.beta_d + shift_d, held.beta_y + shift_y)
    targeted, record = walk_separable(start, emptied, TARGET, step_size=0.05, tol=1e-12)
    assert record.converged
    assert record.phase == phase
    assert np.all(np.isneginf(targeted.beta_d[:, 2, 0]))


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


def test_a_rule_beside_a_separable_regimen_is_refused() -> None:
    rule = DynamicRegimen("rule", (lambda h: (h["W"] > 0).astype(int),), rule_kind="known")
    with pytest.raises(CapabilityError, match="'rule' is not a static arm"):
        _fit(regimens={"a0": 0, "rule": rule, "sep": SEPARABLE})


def test_covariates_after_baseline_are_refused_for_want_of_an_influence_curve() -> None:
    law = survival_law(causes=2, with_l2=True)
    with (
        warnings.catch_warnings(),
        pytest.raises(CapabilityError, match="Equation \\(19\\)") as caught,
    ):
        warnings.simplefilter("ignore")
        _ltmle().fit(law.frame(), **law.fit_columns())
    assert caught.value.refusal.reason == "inference_gap"


# --------------------------------------------------------------- positivity


def _blocked(frame: Any, level: int, node: int, code: str) -> Any:
    """Give every unit at risk at ``node`` on ``level`` an event: ``code`` names it."""
    out = frame.copy()
    risk = (out["A"] == level) & (out[f"C{node}"] == 1) & out[f"R{node}"].notna()
    if code == "death":
        out.loc[risk, [f"R{node}", f"D{node}"]] = (0.0, 1.0)
    else:
        free = risk & (out[f"R{node}"] == 0) & (out[f"D{node}"] == 0)
        out.loc[free, f"D{node}"] = 1.0
    for later in range(node + 1, 4):
        out.loc[risk, [f"A{later}", f"C{later}", f"R{later}", f"D{later}"]] = np.nan
    return out


def test_a_node_where_every_event_is_death_at_the_relapse_level_is_refused() -> None:
    """``h_Y`` at :math:`a_Y` is 0/0 there, so the separable risk is not identified."""
    law = _law()
    frame = _blocked(law.frame(), level=1, node=2, code="death")
    with (
        warnings.catch_warnings(),
        pytest.raises(CapabilityError, match="positivity \\(7\\)"),
    ):
        warnings.simplefilter("ignore")
        _ltmle(regimens={"a2": 2, "sep": SEPARABLE}).fit(frame, **law.fit_columns())


def test_horizons_past_a_node_nobody_survives_at_the_death_level_are_dropped() -> None:
    law = _law()
    frame = _blocked(law.frame(), level=0, node=2, code="any")
    with pytest.warns(UserWarning, match="horizons after it are not reported") as caught:
        result = _ltmle(regimens={"a2": 2, "sep": SEPARABLE}).fit(frame, **law.fit_columns())
    assert "node 2" in str(caught[0].message)
    names = set(result.estimates)
    assert _name("sep", "relapse", 2) in names
    assert _name("sep", "relapse", 3) not in names
    assert _name("a2", "relapse", 3) in names
    assert result.estimates[_name("sep", "relapse", 2)].std_error is not None


# ------------------------------------------------------------- compositions


def test_clusters_widen_the_interval_and_keep_the_estimate(fit: Any) -> None:
    law = _law()
    frame = law.frame()
    frame = frame.assign(cl=np.arange(len(frame)) // 3)
    data = LongitudinalData.from_frame(frame, id="cl", **law.fit_columns())
    clustered = _fit(data=data)
    for name in (_name("sep", "relapse", 3), _name("a1", "death", 2)):
        assert clustered.estimates[name].psi == pytest.approx(fit.estimates[name].psi, abs=1e-12)
        assert clustered.estimates[name].std_error != pytest.approx(
            fit.estimates[name].std_error, rel=1e-3
        )


def test_pooled_hazards_and_isotonic_compose(fit: Any) -> None:
    for setting in ({"hazard_pooling": "pooled"}, {"isotonic": True}):
        other = _fit(**setting)
        for name, estimate in fit.estimates.items():
            assert other.estimates[name].psi == pytest.approx(estimate.psi, abs=1e-10), setting


def test_the_restricted_mean_reads_the_separable_curves(fit: Any) -> None:
    """RMST is ``tau`` minus the summed incidence of both causes before ``tau``."""
    rmst = fit.rmst("sep", 3)
    total = sum(
        fit.estimates[_name("sep", cause, t)].psi for cause in ("relapse", "death") for t in (1, 2)
    )
    assert rmst.psi == pytest.approx(3 - total, abs=1e-12)


# ------------------------------------------------------------- reporting


def test_the_support_key_marks_a_separable_fit(fit: Any) -> None:
    """Its static arms come from the factor-wise engine, so no static-arm row may cover them."""
    assert "separable" in fit.config.support_key.family
    assert "separable" not in _fit(regimens={"a0": 0, "a1": 1}).config.support_key.family


def test_the_support_key_holds_the_position_of_the_first_cause(fit: Any) -> None:
    """The study counted the second-listed cause first, so the other order reads no ``G49``."""
    causes = tuple(_data().cause_labels)
    qualifiers = fit.config.support_key.qualifiers
    assert f"separable_first_cause_{causes.index('death') + 1}" in qualifiers
    swapped = Separable(arms={"death": 0, "relapse": 1}, first="relapse")
    other = _fit(regimens={"a0": 0, "a1": 1, "sep": swapped}).config.support_key
    assert f"separable_first_cause_{causes.index('relapse') + 1}" in other.qualifiers
    assert other.qualifiers != qualifiers
    for row in (row for row in LONGITUDINAL_ROWS if row.id.startswith("G49")):
        assert (
            classify_key(
                replace(row.key, qualifiers=frozenset({"separable_first_cause_1"})), {}
            ).row
            is not row
        )
        assert classify_key(row.key, {}).row is row


def test_the_summary_names_both_components_and_the_order(fit: Any) -> None:
    text = " ".join(str(fit.summary()).split())
    assert "the part acting on death at 0, the part acting on relapse at 1" in text
    assert "death counted first within a time step" in text


def _captured(fit: Any) -> tuple[Any, dict[str, Any]]:
    """Run ``fit()`` and keep the inputs of its one call to :func:`walk_separable`."""
    captured: dict[str, Any] = {}
    original = separable_module.walk_separable

    def recording(start: Any, inputs: Any, targets: Any, **settings: Any) -> Any:
        captured.update(start=start, inputs=inputs, targets=targets, settings=settings)
        return original(start, inputs, targets, **settings)

    separable_module.walk_separable = recording  # type: ignore[assignment]
    try:
        result = fit()
    finally:
        separable_module.walk_separable = original  # type: ignore[assignment]
    return result, captured


def _rewalk(captured: dict[str, Any]) -> Any:
    """The record of the captured walk, rerun under the module's current constants."""
    _, record = walk_separable(
        captured["start"],
        captured["inputs"],
        captured["targets"],
        **{**captured["settings"], "warn": False},
    )
    return record


def _solved(result: Any) -> Any:
    record = result.fits[next(iter(result.fits))].hazard
    assert record.converged
    assert record.tol == pytest.approx(1e-10)
    assert relative_score(record.score, record.score_scale) <= record.tol
    return record


def test_a_cross_fitted_separable_fit_solves_every_factor_score() -> None:
    """From a cross-fitted start the walk hands over to the fallbacks and converges.

    The fold split is seeded, so the start is the same on every run.  Split 0 converges in
    the iterative TMLE; the two tests below cover the splits that need each fallback.
    """
    _solved(_fit(n_folds=2, random_state=0))


#: A two-fold split of the exact law whose nearly collinear targets leave the iterative TMLE
#: short of ``tol`` (near 7e-5); Newton steps on the full score Jacobian finish it.
COLLINEAR_SPLIT = 163


def test_nearly_collinear_targets_converge_through_the_jacobian_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Split 163 converges, and the mutation control without the Jacobian steps stalls."""
    result, captured = _captured(lambda: _fit(n_folds=2, random_state=COLLINEAR_SPLIT))
    _solved(result)
    monkeypatch.setattr(separable_module, "_JACOBIAN_STEPS", 0)
    record = _rewalk(captured)
    assert not record.converged
    assert record.failure == "newton_stalled"


#: A primary draw of the ``separable-effects`` study whose 12 targets per factor and level
#: saturate that factor's 12 cells across the two folds: solving the scores swaps the folds'
#: cell hazards, which the walk and the Jacobian steps cannot reach (both stall near 1e-3).
SATURATED_DRAW = 29


def test_saturating_targets_converge_through_the_iterative_tmle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Draw 29 converges, and the mutation control without the iterative TMLE stalls."""
    from tests.studies import separable_effects as study

    frame, _ = study.draw_scenario(study.CROSS_FITTED, study.PRIMARY_N, SATURATED_DRAW)

    def fit() -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return study.fit(frame, study.CROSS_FITTED)

    result, captured = _captured(fit)
    _solved(result)
    monkeypatch.setattr(separable_module, "_ITERATIVE_STEPS", 0)
    record = _rewalk(captured)
    assert not record.converged
    assert record.failure == "newton_stalled"
    assert relative_score(record.score, record.score_scale) > 1e-4


def _saturating_targets() -> list[SeparableTarget]:
    """The 12 targets of ``a0`` and ``sep`` that move lambda_D at level 0: as many as that
    factor's (W, node) cells, so solving their scores sets each cell to its own hazard."""
    return [
        SeparableTarget(label, levels, role, cause, horizon)
        for label, levels in (("a0", (0, 0)), ("sep", (0, 1)))
        for role, cause in (("competing", "death"), ("event", "relapse"))
        for horizon in HORIZONS
    ]


def test_a_solution_outside_the_bound_reads_cell_separated() -> None:
    """With no death at the last node among W = 0 units at level 0, the 12 targets that move
    lambda_D at level 0 saturate its cells and solve only at a hazard of zero.  The cell's
    logit stops at the clip and the fit says why."""
    law, inputs, exact = _engine()
    w = law.frame()["W"].to_numpy(dtype=int)
    cell = (w == 0) & (inputs.observed == 0)
    assert inputs.event_d[cell, 2].sum() > 0
    event_d = inputs.event_d.copy()
    event_d[cell, 2] = 0.0
    emptied = replace(inputs, event_d=event_d)
    targets = _saturating_targets()
    targeted, record = walk_separable(
        exact, emptied, targets, step_size=0.05, tol=1e-12, warn=False
    )
    assert not record.converged
    assert record.failure == "cell_separated"
    np.testing.assert_allclose(targeted.beta_d[cell, 2, 0], emptied.bounds[0], rtol=1e-12)


@pytest.mark.parametrize(("offset", "separated"), [(0.0, True), (0.5, False)])
def test_rows_read_as_separated_only_on_the_clip(offset: float, separated: bool) -> None:
    """Rows with no death whose competing logit sits on the lower clip carry the score and read
    as separated; the same rows half a logit above the clip do not."""
    law, inputs, exact = _engine()
    w = law.frame()["W"].to_numpy(dtype=int)
    cell = (w == 0) & (inputs.observed == 0)
    event_d = inputs.event_d.copy()
    event_d[cell, 2] = 0.0
    emptied = replace(inputs, event_d=event_d)
    beta_d = exact.beta_d.copy()
    beta_d[w == 0, 2, 0] = emptied.bounds[0] + offset
    state = SeparableState(beta_d, exact.beta_y)
    targets = [SeparableTarget("a0", (0, 0), "competing", "death", 3)]
    assert separable_module._separated(state, emptied, targets, np.ones(2)) is separated


def test_a_rare_cell_with_an_event_near_the_clip_is_not_separated() -> None:
    """One death in a cell of W = 0 units at level 0 puts the cell's root at a hazard of one in
    its size less one, and alpha sets the clip half a logit below it.  The fit reaches the root
    inside the clip.  The iterative TMLE leaves a residual of rounding size there, and the Newton
    steps solve it.  A rule that read the non-events near the clip alone would call the state
    separated and skip them."""
    law, inputs, exact = _engine()
    w = law.frame()["W"].to_numpy(dtype=int)
    cell = np.flatnonzero((w == 0) & (inputs.observed == 0) & inputs.at_risk[:, 2])
    event_d = inputs.event_d.copy()
    event_y = inputs.event_y.copy()
    event_d[cell, 2] = 0.0
    event_d[cell[0], 2] = 1.0
    event_y[cell[0], 2] = 0.0
    root = np.log(1.0 / (len(cell) - 1.0))
    rare = replace(
        inputs, event_d=event_d, event_y=event_y, alpha=float(1.0 / (1.0 + np.exp(root - 0.5)))
    )
    targets = _saturating_targets()
    solved, record = walk_separable(exact, rare, targets, step_size=0.05, tol=1e-13, warn=False)
    assert record.converged
    np.testing.assert_allclose(solved.beta_d[cell, 2, 0], root, rtol=1e-8)
    assert not separable_module._separated(solved, rare, targets, record.sigma)


def test_a_cell_whose_rows_all_die_reads_cell_separated_at_the_upper_clip() -> None:
    """When every W = 0 unit at level 0 at risk at the last node dies there, the cell solves only
    at a hazard of one.  Its logit stops at the upper clip and the fit says why."""
    law, inputs, exact = _engine()
    w = law.frame()["W"].to_numpy(dtype=int)
    cell = (w == 0) & (inputs.observed == 0) & inputs.at_risk[:, 2]
    event_d = inputs.event_d.copy()
    event_y = inputs.event_y.copy()
    event_d[cell, 2] = 1.0
    event_y[cell, 2] = 0.0
    filled = replace(inputs, event_d=event_d, event_y=event_y)
    targeted, record = walk_separable(
        exact, filled, _saturating_targets(), step_size=0.05, tol=1e-12, warn=False
    )
    assert record.failure == "cell_separated"
    np.testing.assert_allclose(targeted.beta_d[cell, 2, 0], filled.bounds[1], rtol=1e-12)


def test_a_separated_fit_skips_the_jacobian_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    """Once the iterative TMLE stops on a separated state, the Newton steps cannot reach a root
    the clip excludes, so the walk does not take them."""
    calls: list[int] = []

    def counted(*args: Any) -> Any:
        calls.append(1)
        raise AssertionError("the Newton steps ran on a separated state")

    monkeypatch.setattr(separable_module, "_jacobian_direction", counted)
    law, inputs, exact = _engine()
    w = law.frame()["W"].to_numpy(dtype=int)
    cell = (w == 0) & (inputs.observed == 0)
    event_d = inputs.event_d.copy()
    event_d[cell, 2] = 0.0
    emptied = replace(inputs, event_d=event_d)
    _, record = walk_separable(
        exact, emptied, _saturating_targets(), step_size=0.05, tol=1e-12, warn=False
    )
    assert record.failure == "cell_separated"
    assert record.phase == "iterative"
    assert not calls


def test_the_jacobian_phase_stops_within_its_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """On the separated fixture with the iterative TMLE skipped, the Newton steps on the full
    score Jacobian stop at their cap: one walk step, then at most three."""
    monkeypatch.setattr(separable_module, "_STALL", 0)
    monkeypatch.setattr(separable_module, "_ITERATIVE_STEPS", 0)
    monkeypatch.setattr(separable_module, "_JACOBIAN_STEPS", 3)
    law, inputs, exact = _engine()
    w = law.frame()["W"].to_numpy(dtype=int)
    cell = (w == 0) & (inputs.observed == 0)
    event_d = inputs.event_d.copy()
    event_d[cell, 2] = 0.0
    emptied = replace(inputs, event_d=event_d)
    _, record = walk_separable(
        exact, emptied, _saturating_targets(), step_size=0.05, tol=1e-12, warn=False
    )
    assert not record.converged
    assert record.n_iter <= 1 + 3


def test_a_study_reports_the_separable_parameters() -> None:
    from sklearn.linear_model import LogisticRegression

    from cleverly import CausalStudy, RegimeMean, TimeToEvent
    from cleverly.datasets import make_point_survival

    frame, _ = make_point_survival(n=1_000, seed=4, n_times=4, causes=2)
    design = TimeToEvent(
        time="time",
        event="event",
        treatment="A",
        baseline=("W1", "W2"),
        grid=(1, 2, 3, 4),
        causes={1: "relapse", 2: "death"},
    )
    learner = LogisticRegression(max_iter=1000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = CausalStudy(frame, design=design).estimate(
            RegimeMean(
                {"arm0": 0, "split": Separable(arms={"relapse": 1, "death": 0}, first="death")}
            ),
            survival_model="hazard",
            targeting="one_step",
            outcome_learner=learner,
            treatment_learner=learner,
            censoring_learner=learner,
            cross_fit=False,
        )
    assert "cif_regimen[split, relapse @ t=4]" in result.estimates


def test_the_score_check_reads_both_factor_scores(fit: Any) -> None:
    """The solver row of a separable target reports the larger of its two factor scores."""
    frame = fit.diagnostics.score_equations().to_frame()
    rows = frame[frame["regimen"] == "sep"]
    assert len(rows) == 2 * len(HORIZONS)
    assert rows["passed"].all()
    record = fit.fits[next(iter(fit.fits))].hazard
    relapse = [i for i, entry in enumerate(record.targets) if entry[:3] == ("sep", "relapse", 3)]
    assert len(relapse) == 2
    largest = max(abs(record.score[i]) / record.score_scale[i] for i in relapse)
    row = rows[(rows["cause"] == "relapse") & (rows["horizon"] == 3)]
    assert float(row["relative_score"].iloc[0]) == pytest.approx(largest, rel=1e-12)
    fit.validate()


def test_a_stochastic_policy_beside_a_separable_regimen_is_refused() -> None:
    from cleverly.interventions import Stochastic

    def half(frame: Any) -> np.ndarray:
        return np.full((len(frame), 3), 1.0 / 3.0)

    mix = DynamicRegimen("mix", (Stochastic(half, "third", density_kind="known"),))
    with pytest.raises(CapabilityError, match="'mix' is not a static arm"):
        _fit(regimens={"a0": 0, "mix": mix, "sep": SEPARABLE})


def test_observation_weights_equal_duplicated_rows() -> None:
    """A weight of two on a unit gives the fit on the panel with that unit twice."""
    law = _law()
    frame = law.frame()
    doubled = (frame["W"] == 1) & (frame["A"] == 2)
    weighted = LongitudinalData.from_frame(
        frame.assign(wt=np.where(doubled, 2.0, 1.0)), weights="wt", **law.fit_columns()
    )
    duplicated = LongitudinalData.from_frame(
        pd.concat([frame, frame[doubled]], ignore_index=True), **law.fit_columns()
    )
    one, two = _fit(data=weighted), _fit(data=duplicated)
    for name in (_name("sep", "relapse", 3), _name("sep", "death", 2), _name("a1", "relapse", 3)):
        assert one.estimates[name].psi == pytest.approx(two.estimates[name].psi, abs=1e-10)
