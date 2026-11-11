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


#: The two phases of the targeting: the walk alone, and Newton steps from the start (the
#: handover forced at once), so each robustness witness checks both.
PHASES = pytest.mark.parametrize("phase", ["walk", "newton"])


def _phase(monkeypatch: pytest.MonkeyPatch, phase: str) -> None:
    if phase == "newton":
        monkeypatch.setattr(separable_module, "_STALL", 0)


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


def test_the_newton_record_describes_the_reported_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """After Newton steps the trace ends at the reported state and counts its steps."""
    monkeypatch.setattr(separable_module, "_STALL", 0)
    _, inputs, exact = _engine()
    shift_d, _ = _shifts()
    start = SeparableState(exact.beta_d + shift_d, exact.beta_y)
    targeted, record = walk_separable(start, inputs, TARGET, step_size=0.05, tol=1e-12)
    score, scale = scores(targeted, inputs, TARGET)
    assert record.trace[-1] == pytest.approx(relative_score(score, scale), rel=1e-12)
    assert len(record.trace) == record.n_iter + 1
    assert record.converged and record.failure is None


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


def test_a_cross_fitted_separable_fit_solves_every_factor_score() -> None:
    """From a cross-fitted start the walk hands over to Newton steps and still converges.

    The fold split is seeded: an unseeded split drew a new start on every run, and an unlucky
    one stopped short of the tolerance on CI.  Seeds 0 to 39 all converge.
    """
    result = _fit(n_folds=2, random_state=0)
    record = result.fits[next(iter(result.fits))].hazard
    assert record.converged
    assert record.tol == pytest.approx(1e-10)
    assert relative_score(record.score, record.score_scale) <= record.tol


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
