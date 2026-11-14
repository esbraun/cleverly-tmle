"""The continuous-time survival TMLE (X91 part G) on an exact law, plus refusals.

The law is the two-cause ``PointLaw`` of ``tests/discrete_law_point_survival.py`` restricted to
arms 0 and 1.  Its sample realises the law exactly, so the stratified Nelson-Aalen fit is the
law's own hazard, the g-formula is a quotient of cell masses, and the efficient influence
function is the complex-step Gateaux derivative of that quotient.  Censoring at node ``k`` is
time ``k - 1``, so censoring ties with the events of node ``k - 1``.
"""

from __future__ import annotations

import warnings
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly import CausalStudy, RegimeContrast, RegimeMean, TimeToEvent
from cleverly.exceptions import (
    CapabilityError,
    ConvergenceWarning,
    DataError,
    MethodConfigurationError,
)
from cleverly.survival import ContinuousSurvivalTMLE, CoxHazard, StratifiedNelsonAalen, _targeting
from cleverly.survival._targeting import Problem, evaluate, point_weights, walk
from cleverly.survival.cox import (
    _design,
    _loglik_parts,
    _risk_sets,
    partial_log_likelihood,
    select_hazard,
)
from tests.discrete_law_point_survival import PointLaw

HORIZONS = (1, 2, 3)
TARGETS = [(a, j, h) for a in (0, 1) for j in (1, 2) for h in HORIZONS]


@cache
def _law() -> dict[str, Any]:
    law = PointLaw("survival", 3, 2, False, False)
    frame = law.long_frame([1.0, 2.0, 3.0])
    keep = frame["A"].to_numpy() != 2
    arm_of = np.array([point[law.columns.index("A")] for point in law.support])
    points = np.flatnonzero(arm_of != 2)
    first = (np.cumsum(keep) - 1)[law.first_row_of()[points]]
    base = law.probs * (arm_of != 2)
    return {
        "law": law,
        "frame": frame[keep].reset_index(drop=True),
        "points": points,
        "first": first,
        "base": base / base.sum(),
    }


def _functional(probs: Any, arm: int, horizon: int, cause: int) -> Any:
    law = _law()["law"]
    total = sum(law._mass(probs, {"A": b}) for b in (0, 1))
    return sum(
        (law._mass(probs, {"W": w, "A": 0}) + law._mass(probs, {"W": w, "A": 1}))
        / total
        * law._regression(probs, {"W": w, "A": arm}, 1, horizon, cause, None)
        for w in (0, 1)
    )


def _truth(arm: int, horizon: int, cause: int) -> float:
    return float(np.real(_functional(_law()["base"].astype(float), arm, horizon, cause)))


@cache
def _gateaux(arm: int, horizon: int, cause: int) -> np.ndarray:
    data = _law()
    out = []
    for point in data["points"]:
        mass = np.zeros(data["base"].size, dtype=complex)
        mass[point] = 1.0
        perturbed = (1 - 1e-30j) * data["base"] + 1e-30j * mass
        out.append(np.imag(_functional(perturbed, arm, horizon, cause)) / 1e-30)
    return np.array(out)


def _arrays() -> tuple[np.ndarray, ...]:
    frame = _law()["frame"]
    W = frame["W"].to_numpy().astype(float)
    A = frame["A"].to_numpy().astype(int)
    T = frame["time"].to_numpy().astype(float)
    D = frame["event"].to_numpy().astype(int)
    return W, A, T, D


def _jumps(model: Any, X: np.ndarray, grid: np.ndarray) -> np.ndarray:
    out = np.zeros((2, X.shape[0], grid.size))
    for a in (0, 1):
        Xa = X.copy()
        Xa[:, 0] = a
        cumulative = model.cumulative_hazard(Xa, grid)
        out[a] = np.diff(np.concatenate([np.zeros((X.shape[0], 1)), cumulative], axis=1), axis=1)
    return out


def _problem(
    *, hazard_by: tuple[int, ...] = (0, 1), tie_rule: bool = True, before: bool = True
) -> tuple[Problem, np.ndarray]:
    """The exact-law problem: saturated or arm-only hazards, the law's own G."""
    W, A, T, D = _arrays()
    X = np.column_stack([A, W]).astype(float)
    grid = np.unique(T)
    dlam = np.zeros((2, 2, T.size, grid.size))
    for j in range(2):
        model = StratifiedNelsonAalen(by=hazard_by).fit(X, T, (j + 1 == D).astype(float))
        dlam[:, j] = _jumps(model, X, grid)
    censor_time = np.where(D > 0, T - 0.25, T) if tie_rule else T
    censoring = StratifiedNelsonAalen(by=(0, 1)).fit(X, censor_time, (D == 0).astype(float))
    retention = np.cumprod(1 - _jumps(censoring, X, grid), axis=-1)
    if before:
        retention = np.concatenate([np.ones((2, T.size, 1)), retention[..., :-1]], axis=-1)
    p1 = np.array([np.mean(A[w == W] == 1) for w in W])
    pi = np.stack([1 - p1, p1])
    at_risk = (T[:, None] >= grid[None, :]).astype(float)
    events = np.stack(
        [((T[:, None] == grid[None, :]) & (D[:, None] == j + 1)).astype(float) for j in range(2)]
    )
    index = np.searchsorted(grid, HORIZONS, side="right") - 1
    weights = point_weights(index, grid.size)
    return Problem(A, at_risk, events, 1.0 / (pi[:, :, None] * retention), weights), dlam


def _ic_error(problem: Problem, dlam: np.ndarray) -> float:
    first = _law()["first"]
    evaluation = evaluate(problem, dlam)
    return max(
        float(np.max(np.abs(evaluation.influence[first, p] - _gateaux(a, h, j))))
        for p, (a, j, h) in enumerate(TARGETS)
    )


def _bias(psi: np.ndarray) -> np.ndarray:
    return np.array([psi[p] - _truth(a, h, j) for p, (a, j, h) in enumerate(TARGETS)])


# ------------------------------------------------------------------ check 1 and 2


def test_the_saturated_fit_is_the_g_formula_and_needs_no_step() -> None:
    problem, dlam = _problem()
    result = walk(problem, dlam, cutoff=1e-12)
    assert result.steps == 0
    np.testing.assert_allclose(_bias(result.evaluation.psi), 0.0, atol=1e-14)
    np.testing.assert_allclose(result.evaluation.score, 0.0, atol=1e-14)


def test_the_influence_curve_is_the_gateaux_derivative() -> None:
    assert _ic_error(*_problem()) < 1e-12


@pytest.mark.parametrize(
    ("mutation", "kwargs"),
    [
        ("censoring risk set holds tied events", {"tie_rule": False}),
        ("S_c(u) in place of S_c(u-)", {"before": False}),
    ],
)
def test_each_censoring_mutation_breaks_the_influence_curve(
    mutation: str, kwargs: dict[str, bool]
) -> None:
    # The estimate never reads S_c at the saturated fit, so only the curve can see these.
    with np.errstate(divide="ignore", invalid="ignore"):
        error = _ic_error(*_problem(**kwargs))
    assert not error <= 0.1, mutation


def test_the_exponential_survival_mutation_breaks_the_curve_and_the_sum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem, dlam = _problem()

    def exponential(jumps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        survival = np.exp(-np.cumsum(jumps.sum(axis=1), axis=-1))
        before = np.concatenate([np.ones((*survival.shape[:-1], 1)), survival[..., :-1]], -1)
        return survival, np.cumsum(before[:, None] * jumps, axis=-1)

    monkeypatch.setattr(_targeting, "curves", exponential)
    assert _ic_error(problem, dlam) > 1e-3
    survival, incidence = exponential(dlam)
    assert np.max(np.abs(survival + incidence.sum(axis=1) - 1.0)) > 1e-3


# ------------------------------------------------------------------- check 3


@pytest.fixture(scope="module")
def wrong_hazard() -> tuple[Problem, np.ndarray]:
    return _problem(hazard_by=(0,))


@pytest.mark.parametrize("norm", ["covariance", "variance", "euclidean"])
def test_targeting_a_wrong_hazard_reaches_the_truth(
    wrong_hazard: tuple[Problem, np.ndarray], norm: str
) -> None:
    problem, dlam = wrong_hazard
    result = walk(problem, dlam, cutoff=1e-7, norm=norm, max_steps=20_000)  # type: ignore[arg-type]
    initial = np.abs(_bias(result.initial.psi))
    witnesses = initial > 1e-3
    assert witnesses.sum() >= 6
    assert result.converged
    assert np.max(np.abs(_bias(result.evaluation.psi))) < 1e-6 * np.max(initial) * 1e3


def test_tilting_only_the_received_arm_keeps_the_bias(
    wrong_hazard: tuple[Problem, np.ndarray],
) -> None:
    problem, dlam = wrong_hazard
    result = walk(problem, dlam, cutoff=1e-7, max_steps=20_000, tilt="factual")
    np.testing.assert_allclose(result.evaluation.score, 0.0, atol=1e-6)
    initial = np.abs(_bias(result.initial.psi))
    assert np.max(np.abs(_bias(result.evaluation.psi))[initial > 1e-3]) > 1e-3


def test_freezing_the_other_cause_keeps_the_bias(
    wrong_hazard: tuple[Problem, np.ndarray], monkeypatch: pytest.MonkeyPatch
) -> None:
    problem, dlam = wrong_hazard
    original = _targeting._direction

    def frozen(problem: Problem, evaluation: Any, v: np.ndarray) -> np.ndarray:
        out = original(problem, evaluation, v)
        out[:, 1] = 0.0
        return out

    monkeypatch.setattr(_targeting, "_direction", frozen)
    result = walk(problem, dlam, cutoff=1e-7, max_steps=2_000)
    assert np.max(np.abs(_bias(result.evaluation.psi))) > 1e-3


def test_a_held_pair_that_still_reads_below_the_floor_halves_the_step(
    wrong_hazard: tuple[Problem, np.ndarray], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A tilt of zero can round a held pair's survival down by an ulp.  Here every trial pushes
    # the lowest pair down, held or not, so holding it cannot help: the walk must halve, where
    # retrying the same step would loop forever.
    import signal

    problem, dlam = wrong_hazard
    lowest = np.unravel_index(np.argmin(evaluate(problem, dlam).survival[..., -1]), (2, problem.n))
    original = _targeting.tilt_jumps

    def rounding(current: np.ndarray, increment: np.ndarray) -> np.ndarray:
        out = original(current, increment)
        out[lowest[0], :, lowest[1]] *= 1.01
        return out

    monkeypatch.setattr(_targeting, "WALK_SURVIVAL_FLOOR", 1.0)
    monkeypatch.setattr(_targeting, "tilt_jumps", rounding)
    signal.signal(signal.SIGALRM, lambda *_: pytest.fail("the walk did not stop"))
    signal.alarm(60)
    try:
        result = walk(problem, dlam, cutoff=1e-7, max_steps=20_000)
    finally:
        signal.alarm(0)
    assert result.halvings > 0


def test_the_true_hazards_take_no_step_under_any_censoring_model() -> None:
    problem, dlam = _problem(tie_rule=False)
    assert walk(problem, dlam, cutoff=1e-12).steps == 0


# ------------------------------------------------------------------- check 4


@pytest.mark.parametrize("norm", ["covariance", "variance", "euclidean"])
def test_the_path_derivative_of_the_log_likelihood_is_the_score_norm(
    wrong_hazard: tuple[Problem, np.ndarray], norm: str
) -> None:
    problem, dlam = wrong_hazard
    evaluation = evaluate(problem, dlam)
    sigma = np.cov(evaluation.influence, rowvar=False, bias=True)
    metric = _targeting._metric(sigma, norm)  # type: ignore[arg-type]
    x = evaluation.score
    length = float(np.sqrt(x @ metric @ x))
    direction = _targeting._direction(problem, evaluation, metric @ x / length)
    factual = np.stack([(problem.arm == a) for a in (0, 1)])[:, None, :, None]

    def loglik(step: float) -> float:
        # The multinomial likelihood of each cell's jumps at the received arm.
        jumps = (_targeting.tilt_jumps(dlam, step * direction) * factual).sum(axis=0)
        none = 1.0 - jumps.sum(axis=0)
        stay = problem.at_risk - problem.events.sum(axis=0)
        return (
            float(
                np.sum(problem.events * np.log(np.where(problem.events > 0, jumps, 1.0)))
                + np.sum(stay * np.log(none))
            )
            / problem.n
        )

    h = 1e-6
    derivative = (loglik(h) - loglik(-h)) / (2 * h)
    assert derivative == pytest.approx(length, rel=1e-6)


# ------------------------------------------------------------------- check 5


def test_incidences_and_survival_add_to_one_along_the_walk(
    wrong_hazard: tuple[Problem, np.ndarray],
) -> None:
    problem, dlam = wrong_hazard
    result = walk(problem, dlam, cutoff=1e-7, max_steps=50)
    survival, incidence = _targeting.curves(result.dlam)
    np.testing.assert_allclose(survival + incidence.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(np.diff(incidence, axis=-1) >= -1e-15)


# ------------------------------------------------------------- check 6: Cox


def _cox_sample(n: int = 400, seed: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    A = rng.integers(0, 2, n)
    L = rng.normal(size=(n, 2))
    T = rng.exponential(1 / np.exp(0.5 * A - 0.3 * L[:, 0]))
    C = rng.exponential(2.0, n)
    time = np.round(np.minimum(T, C), 2)  # ties
    return np.column_stack([A, L]), time, (T <= C).astype(float)


def test_the_cox_fit_solves_its_score_with_the_exact_hessian() -> None:
    X, time, event = _cox_sample()
    model = CoxHazard("main").fit(X, time, event)
    Z = _design(X, "main")
    sets = _risk_sets(time, event)
    _, grad, info = _loglik_parts(model.coef_, Z, sets)
    assert np.max(np.abs(grad)) < 1e-9
    step = 1e-6
    numeric = np.array(
        [
            (
                _loglik_parts(model.coef_ + step * e, Z, sets)[1]
                - _loglik_parts(model.coef_ - step * e, Z, sets)[1]
            )
            / (2 * step)
            for e in np.eye(3)
        ]
    )
    np.testing.assert_allclose(-numeric, info, atol=1e-6)


def test_at_zero_coefficients_the_baseline_is_nelson_aalen_and_the_score_is_log_rank() -> None:
    X, time, event = _cox_sample()
    pooled = StratifiedNelsonAalen(by=()).fit(X, time, event)
    nearly_zero = CoxHazard("treatment", penalty=1e12).fit(X, time, event)
    grid = np.linspace(0.0, 3.0, 13)
    np.testing.assert_allclose(
        nearly_zero.cumulative_hazard(X[:2], grid), pooled.cumulative_hazard(X[:2], grid), atol=1e-9
    )
    sets = _risk_sets(time, event)
    _, grad, _ = _loglik_parts(np.zeros(1), X[:, :1], sets)
    observed = event[X[:, 0] == 1].sum()
    expected = sum(
        d * np.mean(X[time >= t, 0]) for t, d in zip(sets.times, sets.deaths, strict=True)
    )
    assert grad[0] == pytest.approx(observed - expected, rel=1e-12)


def test_a_hand_computed_tied_example() -> None:
    # Two events tie at t = 1 among four units at risk; one more at t = 2 among two.
    X = np.array([[0.0], [1.0], [0.0], [1.0]])
    time = np.array([1.0, 1.0, 2.0, 3.0])
    event = np.array([1.0, 1.0, 1.0, 0.0])
    model = CoxHazard("treatment").fit(X, time, event)
    beta = model.coef_[0]
    # Breslow: score = sum over events of x - (2 * S1/S0 at t=1) - (S1/S0 at t=2) = 0
    s0_1, s1_1 = 2 + 2 * np.exp(beta), 2 * np.exp(beta)
    s0_2, s1_2 = 1 + np.exp(beta), np.exp(beta)
    assert 1 - 2 * s1_1 / s0_1 - s1_2 / s0_2 == pytest.approx(0.0, abs=1e-10)
    jumps = model.cumulative_hazard(np.array([[0.0]]), np.array([1.0, 2.0]))[0]
    assert jumps[0] == pytest.approx(2 / s0_1, rel=1e-10)
    assert jumps[1] - jumps[0] == pytest.approx(1 / s0_2, rel=1e-10)


def test_the_selector_compares_like_with_like_and_refuses_a_mixed_library() -> None:
    X, time, event = _cox_sample()
    folds = np.arange(time.size) % 5
    selection = select_hazard(
        [CoxHazard("treatment"), CoxHazard("main"), CoxHazard("interaction")], X, time, event, folds
    )
    assert selection.chosen_name in {"CoxHazard('main')", "CoxHazard('interaction')"}
    assert np.isfinite(partial_log_likelihood(selection.model, X, time, event))
    with pytest.raises(CapabilityError, match="same risk sets"):
        select_hazard([CoxHazard("main"), StratifiedNelsonAalen((0,))], X, time, event, folds)


def test_a_member_with_a_non_finite_loss_never_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    from cleverly.survival import cox

    X, time, event = _cox_sample()
    folds = np.arange(time.size) % 5
    library = [CoxHazard("treatment"), CoxHazard("main"), CoxHazard("interaction")]
    original = cox.partial_log_likelihood

    def broken(model: Any, *args: Any) -> float:
        return float("nan") if model.terms != "treatment" else original(model, *args)

    monkeypatch.setattr(cox, "partial_log_likelihood", broken)
    assert select_hazard(library, X, time, event, folds).chosen_name == "CoxHazard('treatment')"
    monkeypatch.setattr(cox, "partial_log_likelihood", lambda *args: float("nan"))
    with pytest.raises(DataError, match="non-finite cross-validated loss"):
        select_hazard(library, X, time, event, folds)


def test_a_newton_step_that_fails_every_halving_is_not_taken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Away from zero the likelihood is not finite, as when a risk set's weights underflow at a
    # monotone-likelihood boundary: the fit stays at its last finite coefficients and warns.
    from cleverly.survival import cox

    X, time, event = _cox_sample()
    original = cox._loglik_parts

    def overflowing(beta: np.ndarray, *args: Any) -> tuple[float, np.ndarray, np.ndarray]:
        loglik, grad, info = original(beta, *args)
        return (loglik if not np.any(beta) else float("nan")), grad, info

    monkeypatch.setattr(cox, "_loglik_parts", overflowing)
    with pytest.warns(ConvergenceWarning, match="monotone-likelihood boundary"):
        model = CoxHazard("main").fit(X, time, event)
    assert not model.converged_
    np.testing.assert_array_equal(model.coef_, 0.0)


# --------------------------------------------------------- end to end and refusals


def _sample(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X3 = rng.normal(size=n)
    Z = rng.binomial(1, 1 / (1 + np.exp(-X3)))
    h1 = np.exp(-1.5 + 0.5 * X3 - Z)
    h2 = np.exp(-2.0 - 0.5 * X3 - 0.5 * Z)
    T = rng.exponential(1 / (h1 + h2))
    cause = np.where(rng.uniform(size=n) < h1 / (h1 + h2), 1, 2)
    C = rng.exponential(1 / 0.15, n)
    return pd.DataFrame(
        {"time": np.minimum(T, C), "event": np.where(T <= C, cause, 0), "Z": Z, "X3": X3}
    )


def _design_continuous(**extra: Any) -> TimeToEvent:
    return TimeToEvent(
        time="time", event="event", treatment="Z", baseline=["X3"], grid="continuous", **extra
    )


def test_the_study_route_reports_every_risk_and_contrast() -> None:
    study = CausalStudy(_sample(), design=_design_continuous(causes={1: "relapse", 2: "death"}))
    effect = study.identify(
        RegimeContrast({"treated": 1, "control": 0}, reference="control", horizons=[0.5, 1.0])
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = effect.estimate(random_state=0, cross_fit=False)
    assert "cif_regimen[treated, death @ t=1]" in result
    assert "ate_regimen[treated vs control, relapse @ t=0.5]" in result
    assert result.record.converged
    assert not result.record.cross_fitted
    assert 0.0 <= result["cif_regimen[control, relapse @ t=1]"].psi <= 1.0


def test_a_polars_frame_comes_back_as_polars_and_the_result_saves(tmp_path: Any) -> None:
    pl = pytest.importorskip("polars")
    import cleverly

    study = CausalStudy(pl.from_pandas(_sample(300)), design=_design_continuous())
    effect = study.identify(RegimeMean({"1": 1, "0": 0}, horizons=[1.0]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = effect.estimate(random_state=0, cross_fit=False)
    frame = result.to_frame()
    assert isinstance(frame, pl.DataFrame)
    assert list(frame["estimand"]) == list(result)
    loaded = cleverly.load(result.save(tmp_path / "fit.joblib"))
    assert {k: v.psi for k, v in loaded.items()} == {k: v.psi for k, v in result.items()}
    assert loaded.backend == "polars"


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ({"cluster": "X3"}, "cluster"),
        ({"eliminate": ("death",), "causes": {1: "relapse", 2: "death"}}, "eliminate"),
        ({"continuous_treatment": True}, "continuous_treatment"),
    ],
)
def test_the_design_refuses_what_has_no_continuous_form(extra: dict[str, Any], match: str) -> None:
    with pytest.raises(CapabilityError, match=match):
        _design_continuous(**extra)


def test_other_methods_estimands_and_settings_refuse() -> None:
    study = CausalStudy(_sample(), design=_design_continuous())
    effect = study.identify(RegimeMean({"1": 1, "0": 0}, horizons=[1.0]))
    with pytest.raises(CapabilityError, match="only the one-step TMLE has a continuous-time"):
        effect.estimate(method="gcomp")
    with pytest.raises(MethodConfigurationError, match="q_bounds"):
        effect.estimate(q_bounds=(0.1, 0.9))
    with pytest.raises(CapabilityError, match="two static arms"):
        study.identify(RegimeMean({"1": 1}, horizons=[1.0]))
    with pytest.raises(DataError, match="horizons"):
        study.identify(RegimeMean({"1": 1, "0": 0}))


def test_a_flexible_treatment_learner_needs_cross_fitting() -> None:
    from sklearn.ensemble import RandomForestClassifier

    with pytest.raises(CapabilityError, match="cross_fit=True"):
        ContinuousSurvivalTMLE([1.0], cross_fit=False, treatment_learner=RandomForestClassifier())


def test_a_target_before_any_event_of_a_cause_in_an_arm_is_refused() -> None:
    frame = _sample()
    early = float(frame["time"][frame["event"] > 0].min()) / 2
    with pytest.raises(CapabilityError, match="no event of cause"):
        ContinuousSurvivalTMLE([early], cross_fit=False).fit(
            frame, time="time", event="event", treatment="Z", baseline=["X3"]
        )


# ------------------------------------------------- the engine's own assembly (review items)


def _exact_propensity(learner, L, A, train, predict, folds, random_state):  # type: ignore[no-untyped-def]
    """The law's own treatment probability: the arm share at each covariate value."""
    p = np.array([np.mean(A[L[:, 0] == w] == 1) for w in L[:, 0]])
    return p[predict], None


def test_the_engine_assembles_the_exact_influence_curve(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_fit_arrays`` itself, not a hand-built problem: tie shift, S_c(u-), pi and weights.

    With saturated hazards and the law's own treatment probability the engine's influence
    curve is the Gateaux derivative.  Dropping the censoring tie shift, reading S_c(u) for
    S_c(u-), or swapping the arms of pi each moves it far past the tolerance.
    """
    from cleverly.survival import engine

    data = _law()
    frame = data["frame"]
    monkeypatch.setattr(engine, "_fit_propensity", _exact_propensity)
    fit = ContinuousSurvivalTMLE(
        list(HORIZONS),
        cross_fit=False,
        hazard_library=[StratifiedNelsonAalen((0, 1))],
        censoring_library=[StratifiedNelsonAalen((0, 1))],
        tol=1e-12,
        random_state=0,
        causes={1: "1", 2: "2"},
        arms={"0": 0, "1": 1},
        simultaneous=False,
    ).fit(frame, time="time", event="event", treatment="A", baseline=["W"])
    error = max(
        np.max(
            np.abs(
                fit[f"cif_regimen[{a}, {j} @ t={h}]"].influence_curve[data["first"]]
                - _gateaux(a, h, j)
            )
        )
        for a, j, h in TARGETS
    )
    assert error < 1e-10


def test_cross_fitting_trains_every_nuisance_without_the_held_out_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cleverly.survival import engine

    frame = _sample(300)
    sizes: list[int] = []
    original = engine.select_hazard

    def recording(library, X, time, event, folds):  # type: ignore[no-untyped-def]
        sizes.append(len(time))
        return original(library, X, time, event, folds)

    monkeypatch.setattr(engine, "select_hazard", recording)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ContinuousSurvivalTMLE([1.0], n_folds=3, learner_folds=3, random_state=0).fit(
            frame, time="time", event="event", treatment="Z", baseline=["X3"]
        )
    assert fit.record.cross_fitted and fit.n_folds == 3
    # Three folds, two causes and the censoring hazard: nine selections, each on two folds.
    assert len(sizes) == 9
    assert max(sizes) < 300 and min(sizes) >= 190
    assert all(np.isfinite(e.psi) and e.std_error > 0 for e in fit.values())


def test_a_walk_that_stops_early_warns_and_says_so() -> None:
    from cleverly.exceptions import ConvergenceWarning

    with pytest.warns(ConvergenceWarning, match="stopped after 1 steps"):
        fit = ContinuousSurvivalTMLE(
            [0.5, 1.0], cross_fit=False, max_iter=1, tol=1e-12, random_state=0
        ).fit(_sample(), time="time", event="event", treatment="Z", baseline=["X3"])
    assert not fit.record.converged and fit.record.steps == 1
    assert "continuous_not_converged" in fit.summary().note_codes


def test_a_floor_raised_until_it_binds_is_counted() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ContinuousSurvivalTMLE([1.0], cross_fit=False, g_bounds=0.6, random_state=0).fit(
            _sample(), time="time", event="event", treatment="Z", baseline=["X3"]
        )
    assert 0.0 < fit.record.bounded_share < 1.0
    assert 0.0 < fit.record.last_time_floor_share <= 1.0


def test_a_target_past_one_arms_follow_up_is_refused() -> None:
    rng = np.random.default_rng(1)
    n = 400
    X3 = rng.normal(size=n)
    Z = rng.binomial(1, 0.5, n)
    T = rng.exponential(1 / np.exp(-2 + 0.3 * X3 - 0.5 * Z))
    C = np.where(Z == 0, 3.0, 20.0)
    frame = pd.DataFrame(
        {"time": np.minimum(T, C), "event": (T <= C).astype(int), "Z": Z, "X3": X3}
    )
    with pytest.raises(CapabilityError, match="followed only to"):
        ContinuousSurvivalTMLE([2.0, 6.0], cross_fit=False).fit(
            frame, time="time", event="event", treatment="Z", baseline=["X3"]
        )


@pytest.mark.parametrize(
    ("jump", "match"), [(0.5, "combined jump of one or more"), (0.45, "survival reaches zero")]
)
def test_an_initial_fit_that_breaks_the_product_limit_is_a_data_failure(
    jump: float, match: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Both checks read the fitted hazards of one draw, so they raise DataError, which a study
    # records as a failed replicate; a CapabilityError would stop the whole run.
    from cleverly.survival import engine

    monkeypatch.setattr(engine, "_jumps", lambda cumulative: np.full(cumulative.shape, jump))
    with pytest.raises(DataError, match=match):
        ContinuousSurvivalTMLE([2.0], cross_fit=False).fit(
            _sample(), time="time", event="event", treatment="Z", baseline=["X3"]
        )


def test_a_declared_cause_with_no_event_is_refused() -> None:
    frame = _sample()
    frame["event"] = np.where(frame["event"] == 2, 0, frame["event"])
    with pytest.raises(CapabilityError, match="have no event"):
        ContinuousSurvivalTMLE([1.0], cross_fit=False, causes={1: "relapse", 2: "death"}).fit(
            frame, time="time", event="event", treatment="Z", baseline=["X3"]
        )


def test_the_study_route_answers_the_setup_questions_and_reads_the_band_default() -> None:
    from cleverly.sensitivity import evalue

    study = CausalStudy(_sample(300), design=_design_continuous())
    effect = study.identify(RegimeContrast({"t": 1, "c": 0}, reference="c", horizons=[1.0]))
    assert "continuous follow-up" in str(study.summary())
    assert "result.record.bounded_share" in str(effect.summary())
    methods = {record.name: record.available for record in effect.available_methods()}
    assert [name for name, ok in methods.items() if ok] == ["tmle"]
    with pytest.raises(MethodConfigurationError, match="upper bound is 1"):
        effect.estimate(g_bounds=(0.05, 0.95))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = effect.estimate(cross_fit=False, random_state=0)
    assert fit.simultaneous is not None
    with pytest.raises(CapabilityError, match="two arms' risks at one cause"):
        evalue(fit)


def test_the_initial_estimates_are_the_plug_in_before_any_step() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ContinuousSurvivalTMLE([0.5, 1.0], cross_fit=False, tol=10.0, random_state=0).fit(
            _sample(), time="time", event="event", treatment="Z", baseline=["X3"]
        )
    assert fit.record.steps == 0
    initial = fit.initial_estimates
    assert set(initial) == {name for name in fit if not name.startswith("rr_regimen")}
    assert all(abs(initial[name] - fit[name].psi) < 1e-12 for name in initial)


@pytest.mark.parametrize("seed", [0, 1])
def test_pairs_that_reach_the_survival_floor_are_held_and_the_walk_still_converges(
    seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The floor is raised to the initial fit's smallest survival at the last target, so a step
    # that lowers that unit's survival meets it at once.  Rejecting such steps left the walk
    # halving to its minimum step unconverged (10 to 14 steps, 34 to 38 halvings); holding those
    # pairs for the step lets the others bring every score under the cutoff.
    from tests.studies import continuous_survival as study

    monkeypatch.setattr(_targeting, "WALK_SURVIVAL_FLOOR", 1.0)
    frame = study.sample(600, seed, law_name="dependent_censoring")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = study.fit_with(frame, study.engine(cross_fit=False, hazard_wrong=True))
    assert fit.record.frozen_share > 0.0
    assert fit.record.converged


def test_the_support_reads_the_measured_configuration_and_times() -> None:
    frame = _sample(1_000, seed=4)
    observed = frame.loc[frame["event"] > 0, "time"].to_numpy()
    early, middle, late = (float(np.quantile(observed, q)) for q in (0.25, 0.5, 0.75))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ContinuousSurvivalTMLE(
            [early, middle, late], simultaneous=True, n_multiplier=200, random_state=0
        ).fit(frame, time="time", event="event", treatment="Z", baseline=["X3"])
    labels = {name: estimate.support.label for name, estimate in fit.items()}
    for name, label in labels.items():
        gated = not name.startswith("rr_regimen") and f"t={early:g}]" not in name
        assert label == ("supported" if gated else "experimental"), name
    early_name = next(name for name in fit if f"t={early:g}]" in name)
    assert "coverage fell to 0.88" in fit[early_name].support.reason
    assert fit.simultaneous is not None
    assert fit.simultaneous.support.label == "experimental"
    in_sample = ContinuousSurvivalTMLE([middle], cross_fit=False, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        refit = in_sample.fit(frame, time="time", event="event", treatment="Z", baseline=["X3"])
    assert all(estimate.support.label == "experimental" for estimate in refit.values())
    assert all("in-sample" in estimate.support.reason for estimate in refit.values())
    changed = ContinuousSurvivalTMLE([middle], step_size=0.05, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stepped = changed.fit(frame, time="time", event="event", treatment="Z", baseline=["X3"])
    assert all("step_size" in estimate.support.reason for estimate in stepped.values())
    small = ContinuousSurvivalTMLE([middle], n_folds=5, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        half = small.fit(
            frame.iloc[:600], time="time", event="event", treatment="Z", baseline=["X3"]
        )
    assert all("n_folds" in estimate.support.reason for estimate in half.values())
    assert all("600 units" in estimate.support.reason for estimate in half.values())
