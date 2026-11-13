"""The continuous-time result's functionals (X6 part (c)): restricted means, ratios and the curve.

The exact law is the two-cause ``PointLaw`` that ``test_continuous_survival.py`` reads: events at
times 1, 2 and 3, so a restricted mean time lost is an exact weighted sum of the law's risks,
and its efficient influence function is the same sum of their complex-step Gateaux derivatives.
"""

from __future__ import annotations

import json
import warnings
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly._refusal import RefusalReason
from cleverly.exceptions import CapabilityError
from cleverly.sensitivity import evalue, evalue_from_rr
from cleverly.survival import ContinuousSurvivalTMLE, StratifiedNelsonAalen, _targeting
from cleverly.survival._targeting import (
    Evaluation,
    Problem,
    evaluate,
    point_weights,
    restricted_weights,
    walk,
)
from cleverly.survival.result import (
    MEASURED_DEFAULTS,
    OFF_GRID_SHARE,
    continuous_survival_support,
    risk_names,
)
from tests.unit.test_continuous_survival import (
    _arrays,
    _exact_propensity,
    _gateaux,
    _jumps,
    _law,
    _sample,
    _truth,
)

#: The restriction times read on the exact law.  At 2.5 the integral splits the last step.
TAUS = (2.5, 3.0)


def _lost_weights(tau: float) -> dict[int, float]:
    """``{time: weight}`` of the law's risks in :math:`\\int_0^\\tau F`, events at 1, 2, 3."""
    return {h: max(0.0, min(h + 1.0, tau) - h) for h in (1, 2, 3) if h <= tau}


def _rmtl_truth(arm: int, cause: int, tau: float) -> float:
    return sum(w * _truth(arm, h, cause) for h, w in _lost_weights(tau).items())


def _rmtl_gateaux(arm: int, cause: int, tau: float) -> np.ndarray:
    return sum(w * _gateaux(arm, h, cause) for h, w in _lost_weights(tau).items())  # type: ignore[return-value]


def _problem(
    horizons: tuple[float, ...], taus: tuple[float, ...], *, hazard_by: tuple[int, ...] = (0, 1)
) -> tuple[Problem, np.ndarray]:
    """The exact-law problem with point rows at ``horizons`` and restricted rows at ``taus``."""
    W, A, T, D = _arrays()
    X = np.column_stack([A, W]).astype(float)
    grid = np.unique(T)
    dlam = np.zeros((2, 2, T.size, grid.size))
    for j in range(2):
        model = StratifiedNelsonAalen(by=hazard_by).fit(X, T, (j + 1 == D).astype(float))
        dlam[:, j] = _jumps(model, X, grid)
    censoring = StratifiedNelsonAalen(by=(0, 1)).fit(
        X, np.where(D > 0, T - 0.25, T), (D == 0).astype(float)
    )
    retention = np.cumprod(1 - _jumps(censoring, X, grid), axis=-1)
    retention = np.concatenate([np.ones((2, T.size, 1)), retention[..., :-1]], axis=-1)
    p1 = np.array([np.mean(A[w == W] == 1) for w in W])
    pi = np.stack([1 - p1, p1])
    at_risk = (T[:, None] >= grid[None, :]).astype(float)
    events = np.stack(
        [((T[:, None] == grid[None, :]) & (D[:, None] == j + 1)).astype(float) for j in range(2)]
    )
    index = np.searchsorted(grid, horizons, side="right") - 1
    rows = np.vstack(
        [point_weights(index, grid.size), *[restricted_weights(grid, t)[None] for t in taus]]
    )
    return Problem(A, at_risk, events, 1.0 / (pi[:, :, None] * retention), rows), dlam


def _target(problem: Problem, arm: int, cause: int, row: int) -> int:
    return (arm * problem.n_causes + cause) * problem.n_rows + row


# --------------------------------------------------------------- T-c1: the curve


def test_the_restricted_weights_integrate_the_step_function_exactly() -> None:
    grid = np.array([0.5, 1.0, 2.0, 4.0])
    values = np.array([0.1, 0.3, 0.4, 0.7])
    for tau, expected in [(0.4, 0.0), (1.5, 0.1 * 0.5 + 0.3 * 0.5), (5.0, 0.05 + 0.3 + 0.8 + 0.7)]:
        assert restricted_weights(grid, tau) @ values == pytest.approx(expected, abs=1e-15)


def test_the_restricted_mean_curve_is_the_gateaux_derivative() -> None:
    problem, dlam = _problem((1.0, 2.0, 3.0), TAUS)
    evaluation = evaluate(problem, dlam)
    first = _law()["first"]
    for a in (0, 1):
        for j in (1, 2):
            for r, tau in enumerate(TAUS):
                p = _target(problem, a, j - 1, 3 + r)
                assert evaluation.psi[p] == pytest.approx(_rmtl_truth(a, j, tau), abs=1e-14)
                error = np.max(np.abs(evaluation.influence[first, p] - _rmtl_gateaux(a, j, tau)))
                assert error < 1e-12
    # The split step is a real witness: the weight at time 2 is a half, not one or zero.
    assert _lost_weights(2.5) == {1: 1.0, 2: 0.5}
    assert abs(_rmtl_truth(1, 1, 2.5) - _rmtl_truth(1, 1, 3.0)) > 1e-3


def test_the_engine_assembles_the_restricted_means_and_the_rmst(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_fit_arrays`` with ``restricted_horizons``: grid weights, indexing, RMTL and RMST."""
    from cleverly.survival import engine

    data = _law()
    monkeypatch.setattr(engine, "_fit_propensity", _exact_propensity)
    fit = ContinuousSurvivalTMLE(
        [1.0, 2.0, 3.0],
        restricted_horizons=TAUS,
        cross_fit=False,
        hazard_library=[StratifiedNelsonAalen((0, 1))],
        censoring_library=[StratifiedNelsonAalen((0, 1))],
        tol=1e-12,
        random_state=0,
        causes={1: "1", 2: "2"},
        arms={"0": 0, "1": 1},
    ).fit(data["frame"], time="time", event="event", treatment="A", baseline=["W"])
    first = data["first"]
    for tau in TAUS:
        for a in (0, 1):
            total = np.zeros(first.size)
            for j in (1, 2):
                lost = fit.rmtl(str(a), tau, str(j))
                assert lost.psi == pytest.approx(_rmtl_truth(a, j, tau), abs=1e-12)
                assert lost.bounds == (0.0, tau)
                gateaux = _rmtl_gateaux(a, j, tau)
                assert np.max(np.abs(lost.influence_curve[first] - gateaux)) < 1e-10
                total += gateaux
            # T-c9: RMST = tau - sum of the RMTLs, with the curve by linearity.
            survival = fit.rmst(str(a), tau)
            truth = tau - sum(_rmtl_truth(a, j, tau) for j in (1, 2))
            assert survival.psi == pytest.approx(truth, abs=1e-12)
            assert np.max(np.abs(survival.influence_curve[first] + total)) < 1e-10
        gain = fit.rmst("1", tau, versus="0")
        assert gain.name == f"rmst_regimen[1 vs 0 @ t={tau:g}]"
        assert gain.psi == pytest.approx(fit.rmst("1", tau).psi - fit.rmst("0", tau).psi)
        back = fit.rmst("0", tau, versus="1")
        assert back.name == f"rmst_regimen[0 vs 1 @ t={tau:g}]"
        assert back.psi == pytest.approx(-gain.psi, abs=1e-15)
        np.testing.assert_array_equal(back.influence_curve, -gain.influence_curve)
        lost_gap = fit.rmtl("1", tau, "2", versus="0")
        assert lost_gap.name == f"rmtl_regimen[1 vs 0, 2 @ t={tau:g}]"
        assert lost_gap.bounds == (-tau, tau)


# ------------------------------------------------- T-c2: the direction of a weighted target


def _brute_direction(
    problem: Problem, evaluation: Evaluation, v: np.ndarray, *, drop_own_f: bool = False
) -> np.ndarray:
    """``sum_p v_p sum_m w_m 1{u <= u_m} c^(m)_l(u) / (pi S_c)``, point by point."""
    J, Q = problem.n_causes, problem.n_rows
    n, M = problem.at_risk.shape
    out = np.zeros((2, J, n, M))
    for a in (0, 1):
        S = evaluation.survival[a]
        for j in range(J):
            F = evaluation.incidence[a, j]
            for q in range(Q):
                p = (a * J + j) * Q + q
                for m in np.flatnonzero(problem.target_weights[q]):
                    w = v[p] * problem.target_weights[q, m]
                    upto = np.arange(M) <= m
                    shared = (F[:, [m]] - (0.0 if drop_own_f else F)) / S
                    for l in range(J):
                        own = 1.0 if l == j else 0.0
                        out[a, l] += w * upto[None, :] * (own - shared)
        out[a] *= problem.weight[a][None]
    return out


def test_the_weighted_direction_is_the_sum_over_grid_points() -> None:
    problem, dlam = _problem((3.0,), TAUS, hazard_by=(0,))
    evaluation = evaluate(problem, dlam)
    v = np.random.default_rng(0).normal(size=problem.n_targets)
    direction = _targeting._direction(problem, evaluation, v)
    np.testing.assert_allclose(direction, _brute_direction(problem, evaluation, v), atol=1e-12)
    # Mutation: dropping W(u) F_j(u) from the reverse sum moves the direction far.
    mutated = _brute_direction(problem, evaluation, v, drop_own_f=True)
    assert np.max(np.abs(mutated - direction)) > 1e-2


# --------------------------------------------------------------- T-c3: the walk


@pytest.fixture(scope="module")
def wrong_restricted() -> tuple[Problem, np.ndarray]:
    # One target time (3) and RMTL(2.5) = F(1) + F(2) / 2, which no risk at 3 spans.
    return _problem((3.0,), (2.5,), hazard_by=(0,))


def _rmtl_targets(problem: Problem) -> list[int]:
    return [_target(problem, a, j, 1) for a in (0, 1) for j in (0, 1)]


def _rmtl_bias(problem: Problem, psi: np.ndarray) -> np.ndarray:
    return np.array(
        [
            psi[_target(problem, a, j, 1)] - _rmtl_truth(a, j + 1, 2.5)
            for a in (0, 1)
            for j in (0, 1)
        ]
    )


def test_the_walk_solves_the_restricted_mean_score_and_removes_its_bias(
    wrong_restricted: tuple[Problem, np.ndarray],
) -> None:
    problem, dlam = wrong_restricted
    result = walk(problem, dlam, cutoff=1e-7, max_steps=20_000)
    assert result.converged
    assert np.all(result.standardized_score[_rmtl_targets(problem)] <= 1e-7)
    initial = np.abs(_rmtl_bias(problem, result.initial.psi))
    assert initial.max() > 1e-3
    assert np.max(np.abs(_rmtl_bias(problem, result.evaluation.psi))) < 1e-5


def test_a_restricted_mean_left_out_of_the_direction_stays_unsolved(
    wrong_restricted: tuple[Problem, np.ndarray], monkeypatch: pytest.MonkeyPatch
) -> None:
    problem, dlam = wrong_restricted
    original = _targeting._direction
    left_out = _rmtl_targets(problem)

    def without(problem: Problem, evaluation: Any, v: np.ndarray) -> np.ndarray:
        kept = v.copy()
        kept[left_out] = 0.0
        return original(problem, evaluation, kept)

    monkeypatch.setattr(_targeting, "_direction", without)
    result = walk(problem, dlam, cutoff=1e-7, max_steps=2_000)
    assert not result.converged
    assert np.max(result.standardized_score[left_out]) > 1e-3
    assert np.max(np.abs(_rmtl_bias(problem, result.evaluation.psi))) > 1e-3


# ----------------------------------------------------- end-to-end fits on a sample


@cache
def _fit(restricted: tuple[float, ...] = (1.5,), causes: bool = True) -> Any:
    frame = _sample()
    if not causes:
        frame = frame.assign(event=(frame["event"] > 0).astype(int))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ContinuousSurvivalTMLE(
            [0.5, 1.0],
            restricted_horizons=restricted,
            cross_fit=False,
            random_state=0,
            simultaneous=True,
            n_multiplier=200,
            causes={1: "relapse", 2: "death"} if causes else None,
            arms={"control": 0, "treated": 1},
            reference="control",
        ).fit(frame, time="time", event="event", treatment="Z", baseline=["X3"])


def test_the_fit_reports_and_targets_every_restricted_mean() -> None:
    fit = _fit()
    names = [
        "rmtl_regimen[control, relapse @ t=1.5]",
        "rmtl_regimen[treated, death @ t=1.5]",
        "rmtl_regimen[treated vs control, relapse @ t=1.5]",
        "rmst_regimen[control @ t=1.5]",
        "rmst_regimen[treated vs control @ t=1.5]",
    ]
    assert all(name in fit for name in names)
    assert fit.record.converged
    assert len(fit.record.standardized_score) == 2 * 2 * 3
    assert max(fit.record.standardized_score) <= fit.record.cutoff
    # The band covers the risks and their differences only.
    assert set(fit.simultaneous.bands) == set(risk_names(fit.arms, fit.causes, fit.horizons))
    # Every estimate reads experimental: restricted_horizons is not a measured setting.
    assert all(e.support.label == "experimental" for e in fit.values())
    assert "restricted mean" in fit[names[0]].support.reason
    assert "restricted_horizons" in fit["cif_regimen[treated, death @ t=1]"].support.reason


def test_the_initial_estimates_skip_the_restricted_rows() -> None:
    fit = _fit()
    initial = fit.initial_estimates
    assert set(initial) == set(risk_names(fit.arms, fit.causes, fit.horizons))
    assert all(0.0 <= initial[name] <= 1.0 for name in initial if "ate_" not in name)


def test_the_support_reads_a_restricted_mean_as_unmeasured() -> None:
    measured = dict(MEASURED_DEFAULTS)
    common = {"event_share": 0.6, "n": 1_000, "causes": 2, "cross_fitted": True}
    plain = continuous_survival_support(
        scale="level", default_nuisances=True, settings=measured, **common
    )
    assert plain.label == "supported"
    restricted = continuous_survival_support(
        scale="level", default_nuisances=True, settings=measured, restricted=True, **common
    )
    assert restricted.label == "experimental"
    assert "restricted mean" in restricted.reason
    moved = continuous_survival_support(
        scale="level",
        default_nuisances=True,
        settings={**measured, "restricted_horizons": (2.0,)},
        **common,
    )
    assert moved.label == "experimental" and "restricted_horizons" in moved.reason


# ----------------------------------------------------------------- T-c4: refusal C1


def test_a_restricted_mean_the_fit_did_not_target_is_refused() -> None:
    fit = _fit(())
    with pytest.raises(CapabilityError, match=r"rmtl\(\) reads a restricted mean") as caught:
        fit.rmtl("treated", 1.5, "relapse")
    message = str(caught.value)
    assert "about 40% of random times" in message
    assert "restricted_horizons=(1.5,)" in message
    assert caught.value.refusal.reason is RefusalReason.API_REDIRECTION
    with pytest.raises(CapabilityError, match=r"rmst\(\) reads a restricted mean"):
        _fit().rmst("treated", 2.0)


def test_the_refusal_quotes_the_registered_off_grid_share() -> None:
    from pathlib import Path

    readings = pd.read_csv(
        Path(__file__).parents[1] / "canonical/continuous_survival/targeting-readings.csv"
    )
    assert round(float(readings["mean_off_grid_share"].iloc[0]), 2) == OFF_GRID_SHARE


def test_restricted_means_need_a_cause_and_known_arms() -> None:
    fit = _fit()
    with pytest.raises(ValueError, match="name one with cause="):
        fit.rmtl("treated", 1.5)
    with pytest.raises(ValueError, match="unknown cause"):
        fit.rmtl("treated", 1.5, "stroke")
    with pytest.raises(ValueError, match="unknown arm"):
        fit.rmst("placebo", 1.5)
    with pytest.raises(ValueError, match="same arm"):
        fit.rmst("treated", 1.5, versus="treated")
    single = _fit(causes=False)
    assert single.rmtl("treated", 1.5).name == "rmtl_regimen[treated @ t=1.5]"
    with pytest.raises(ValueError, match="one event type"):
        single.rmtl("treated", 1.5, "1")


def test_the_engine_refuses_restricted_horizons_it_cannot_target() -> None:
    with pytest.raises(ValueError, match="positive finite"):
        ContinuousSurvivalTMLE([1.0], restricted_horizons=(-1.0,))
    with pytest.raises(ValueError, match="strictly increasing"):
        ContinuousSurvivalTMLE([1.0], restricted_horizons=(2.0, 1.0))
    frame = _sample()
    followed = float(frame.loc[frame["Z"] == 0, "time"].max())
    with pytest.raises(CapabilityError, match="before the last restricted horizon"):
        ContinuousSurvivalTMLE([1.0], restricted_horizons=(followed + 1.0,), cross_fit=False).fit(
            frame, time="time", event="event", treatment="Z", baseline=["X3"]
        )
    early = float(frame["time"][frame["event"] > 0].min()) / 2
    with pytest.raises(CapabilityError, match="first restricted horizon"):
        ContinuousSurvivalTMLE([1.0], restricted_horizons=(early,), cross_fit=False).fit(
            frame, time="time", event="event", treatment="Z", baseline=["X3"]
        )


# ----------------------------------------------------------- T-c5: ratio and refusals


def test_the_ratio_is_the_stored_log_ratio() -> None:
    fit = _fit()
    top, bottom = "cif_regimen[treated, death @ t=1]", "cif_regimen[control, death @ t=1]"
    rr = fit.ratio(top, bottom)
    stored = fit["rr_regimen[treated vs control, death @ t=1]"]
    assert rr.name == stored.name
    assert rr.psi == pytest.approx(stored.psi, rel=1e-14)
    assert rr.log_psi == pytest.approx(np.log(fit[top].psi / fit[bottom].psi), rel=1e-14)
    np.testing.assert_allclose(rr.ci, stored.ci, rtol=1e-12)
    assert rr.support == stored.support
    assert rr.support.label == "experimental"
    inverse = fit.ratio(bottom, top)
    assert inverse.name == "rr_regimen[control vs treated, death @ t=1]"
    assert inverse.psi == pytest.approx(1 / rr.psi)
    odds = fit.ratio(top, bottom, kind="or")
    a, b = fit[top].psi, fit[bottom].psi
    assert odds.psi == pytest.approx(a / (1 - a) / (b / (1 - b)))
    assert odds.name.startswith("or_regimen[")


@pytest.mark.parametrize(
    ("numerator", "denominator"),
    [
        ("cif_regimen[treated, death @ t=1]", "cif_regimen[treated, death @ t=0.5]"),
        ("cif_regimen[treated, death @ t=1]", "cif_regimen[control, relapse @ t=1]"),
        ("cif_regimen[treated, death @ t=1]", "cif_regimen[treated, death @ t=1]"),
        ("ate_regimen[treated vs control, death @ t=1]", "cif_regimen[control, death @ t=1]"),
        ("rmtl_regimen[treated, death @ t=1.5]", "rmtl_regimen[control, death @ t=1.5]"),
    ],
)
def test_a_ratio_of_anything_but_the_two_arms_risks_is_refused(
    numerator: str, denominator: str
) -> None:
    with pytest.raises(CapabilityError, match="use contrast") as caught:
        _fit().ratio(numerator, denominator)
    assert caught.value.refusal.reason is RefusalReason.API_REDIRECTION


def test_a_risk_between_target_times_is_refused() -> None:
    fit = _fit()
    with pytest.raises(CapabilityError, match=r"0\.7 is not a target time") as caught:
        fit.ratio("cif_regimen[treated, death @ t=0.7]", "cif_regimen[control, death @ t=0.7]")
    assert caught.value.refusal.reason is RefusalReason.API_REDIRECTION
    with pytest.raises(CapabilityError, match=r"0\.7 is not a target time"):
        fit.curve(times=[0.5, 0.7])


def test_the_survival_view_needs_one_event_type() -> None:
    with pytest.raises(CapabilityError, match="not event-free survival"):
        _fit().ratio(
            "cif_regimen[treated, death @ t=1]",
            "cif_regimen[control, death @ t=1]",
            view="survival",
        )
    with pytest.raises(CapabilityError, match="not event-free survival"):
        _fit().curve(scale="survival")
    single = _fit(causes=False)
    top, bottom = "risk_regimen[treated @ t=1]", "risk_regimen[control @ t=1]"
    ratio = single.ratio(top, bottom, view="survival")
    assert ratio.name == "survival_rr_regimen[treated vs control @ t=1]"
    assert ratio.psi == pytest.approx((1 - single[top].psi) / (1 - single[bottom].psi))


# ------------------------------------------------------------------ the curve


def test_the_curve_reads_the_targeted_points_and_the_band() -> None:
    fit = _fit()
    curve = fit.curve()
    assert len(curve) == 2 * 2 * 2 + 2 * 2
    assert list(curve["parameter"]) == risk_names(fit.arms, fit.causes, fit.horizons)
    for _, row in curve.iterrows():
        estimate = fit[row["parameter"]]
        assert row["psi"] == estimate.psi
        assert (row["ci_lower"], row["ci_upper"]) == estimate.ci
        assert (row["band_lower"], row["band_upper"]) == fit.simultaneous.bands[row["parameter"]]
    assert set(curve["scale"]) == {"level", "difference"}
    assert list(fit.curve(times=[1.0])["time"]) == [1.0] * 6


def test_the_survival_curve_mirrors_levels_and_negates_differences() -> None:
    fit = _fit(causes=False)
    risk, survival = fit.curve(), fit.curve(scale="survival")
    level = risk["scale"] == "level"
    np.testing.assert_allclose(survival["psi"][level], 1 - risk["psi"][level])
    np.testing.assert_allclose(survival["ci_lower"][level], 1 - risk["ci_upper"][level])
    np.testing.assert_allclose(survival["psi"][~level], -risk["psi"][~level])
    np.testing.assert_allclose(survival["ci_upper"][~level], -risk["ci_lower"][~level])
    np.testing.assert_allclose(survival["band_lower"][~level], -risk["band_upper"][~level])
    assert survival["estimand"].iloc[0] == "survival_regimen[control @ t=0.5]"
    assert all(name in fit for name in survival["parameter"])


# ------------------------------------------------------------ T-c6: contrast


def test_a_nonlinear_contrast_is_the_delta_method_by_finite_differences() -> None:
    fit = _fit()
    names = ["cif_regimen[treated, death @ t=1]", "rmst_regimen[treated @ t=1.5]"]

    def function(x: np.ndarray) -> float:
        return float(np.log(x[0]) * x[1])

    derived = fit.contrast(function, names, name="log risk times rmst", scale="level")
    point = np.array([fit[name].psi for name in names])
    h = 1e-6
    gradient = np.array(
        [(function(point + h * e) - function(point - h * e)) / (2 * h) for e in np.eye(2)]
    )
    curve = np.column_stack([fit[name].influence_curve for name in names]) @ gradient
    std_err = float(np.sqrt(np.var(curve, ddof=1) / fit.n))
    assert derived.psi == pytest.approx(function(point), rel=1e-14)
    assert derived.std_error == pytest.approx(std_err, rel=1e-6)
    low, high = derived.ci
    assert (high - low) / 2 == pytest.approx(1.959964 * std_err, rel=1e-5)
    assert derived.support.label == "experimental"
    assert "smooth function" in derived.support.reason


# --------------------------------------------------------------- T-c7: to_dict


def test_to_dict_round_trips_through_json() -> None:
    fit = _fit()
    record = fit.to_dict()
    assert json.loads(json.dumps(record)) == record
    assert set(record["estimates"]) == set(fit)
    name = "rmst_regimen[treated vs control @ t=1.5]"
    assert record["estimates"][name]["psi"] == fit[name].psi
    assert record["settings"]["restricted_horizons"] == [1.5]
    assert record["record"]["converged"] is True


# --------------------------------------------------------------- T-c8: E-value


def test_the_evalue_reads_the_ratio_of_the_two_arms_risks() -> None:
    fit = _fit()
    top, bottom = "cif_regimen[treated, death @ t=1]", "cif_regimen[control, death @ t=1]"
    result = evalue(fit, (top, bottom))
    rr = fit.ratio(top, bottom)
    assert result.estimand == rr.name
    assert result.risk_ratio == pytest.approx(rr.psi)
    assert result.point == pytest.approx(evalue_from_rr(rr.psi))
    above = rr.psi >= 1.0
    nearer = rr.ci[0] if above else rr.ci[1]
    crosses = rr.ci[0] <= 1.0 <= rr.ci[1]
    assert result.limit == pytest.approx(1.0 if crosses else evalue_from_rr(nearer))
    assert result.level == pytest.approx(0.95)
    assert result.limit_reading is not None and not result.limit_reading.licensed
    assert "competing cause" in result.note


@pytest.mark.parametrize(
    "estimand",
    [
        None,
        "cif_regimen[treated, death @ t=1]",
        ("cif_regimen[treated, death @ t=1]", "cif_regimen[treated, death @ t=0.5]"),
    ],
)
def test_the_evalue_refuses_anything_but_the_pair(estimand: Any) -> None:
    with pytest.raises(CapabilityError, match="cif_regimen\\[treated, death @ t=2\\]"):
        evalue(_fit(), estimand)
