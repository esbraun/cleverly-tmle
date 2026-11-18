"""Covariate balance over time (X103): the cells against laws whose balance is known.

The witnesses hand :func:`~cleverly.sensitivity.balance_time.longitudinal_cells` and
:func:`~cleverly.sensitivity.balance_time.continuous_cells` the **true** probabilities of a
simulated law, so no learner enters, and each one pairs the shipped weight with a deliberate
mutation that drops or changes one factor. A check whose term vanishes at the truth would pass
with the factor missing; each mutation here moves the expected difference, so the test fails
when the factor is wrong.

Thresholds sit between the pass and mutation values with a margin of at least four Monte Carlo
standard deviations on each side. Sizing runs (n = 400,000 for the population value, 30 to 40
replicates at the test's n for the spread), standardized mean differences:

======  =========================================  ============  ===========================
test    quantity, law                              shipped       mutation
======  =========================================  ============  ===========================
2       D1 of L2 across A2 within A1, law H        -0.002 (.011) no strata: 0.332 (.008)
3       D3 of L2 across A2 within A1, law L        0.007 (.041)  drop g2: 1.18 (.017)
4       C1 balance of L1 within A1, law L          -0.016 (.040) drop the C1 factor: -0.98
5a      2a of L2 across A1, law L', truth 0.726    0.735 (.011)  drop c1: 0.841 (.008)
5b      2a of L3 across A1, law L                  0.773 (.013)  cumulative: 0.583 (.017)
10a     joint, W across A, law K                   -0.014 (.019) drop S_c: -0.165 (.006)
10b     censoring arm 1, law K                     -0.015 (.021) S_c without W: -0.39 (.003)
10c     censoring arm 0, law K                     0.003 (.002)  S_c without W: -0.166 (.002)
======  =========================================  ============  ===========================

Pre-exposure factors (a ``g_s`` or ``c_s`` fixed before the compared treatment) scale both
groups alike, so no large-n witness can see them; :func:`test_hand_computed_cells` pins them
exactly.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from cleverly.exceptions import CapabilityError, DataError
from cleverly.longitudinal.data import LongitudinalData
from cleverly.longitudinal.sequential import Mechanism
from cleverly.sensitivity import BalanceOverTime, covariate_balance
from cleverly.sensitivity import balance_time as bt

_OPEN = (1e-9, 1.0)


def _value(rows: list[bt.TimeBalanceRow], **match: Any) -> float:
    """The size-weighted average of the weighted SMD over the strata of the matching rows."""
    picked = [r for r in rows if all(getattr(r, k) == v for k, v in match.items())]
    assert picked, match
    (group,) = bt.averages(picked, over="all")
    return float(group["smd_weighted"])


def _unweighted(rows: list[bt.TimeBalanceRow], **match: Any) -> float:
    picked = [r for r in rows if all(getattr(r, k) == v for k, v in match.items())]
    (group,) = bt.averages(picked, over="all")
    return float(group["smd_unweighted"])


def _mechanism(treatment: list[np.ndarray], censoring: list[np.ndarray] | None = None) -> Mechanism:
    return Mechanism(
        treatment=tuple({} for _ in treatment),
        censoring=tuple({} for _ in treatment),
        treatment_observed=tuple(treatment),
        censoring_observed=tuple(censoring or ()),
    )


def _matrix(p1: np.ndarray) -> np.ndarray:
    """A two-arm probability matrix from the probability of arm 1."""
    return np.column_stack([1.0 - p1, p1])


# ----------------------------------------------------------------------------- the laws


def _law_l(n: int, seed: int, *, strong_dropout: bool = False) -> tuple[LongitudinalData, dict]:
    """Law L (or L' with ``strong_dropout``): three nodes, dropout after node 1 only."""
    rng = np.random.default_rng(seed)
    l1 = rng.normal(size=n)
    p1 = expit(0.8 * l1)
    a1 = rng.binomial(1, p1)
    c1 = expit(0.5 + 1.5 * a1 - 1.5 * l1) if strong_dropout else expit(1.0 + a1 - l1)
    kept = rng.binomial(1, c1).astype(bool)
    l2 = rng.normal(0.8 * a1 + 0.5 * l1)
    p2 = expit(0.3 * a1 + 1.2 * l2)
    a2 = rng.binomial(1, p2)
    l3 = rng.normal(0.4 * a1 + 1.2 * a2 + 0.5 * l2)
    a3 = rng.binomial(1, 0.5, n)
    y = rng.normal(l3)
    gone = ~kept
    frame = pd.DataFrame(
        {
            "W": rng.normal(size=n),
            "L1": l1,
            "A1": a1,
            "C1": kept.astype(float),
            "L2": np.where(gone, np.nan, l2),
            "A2": np.where(gone, np.nan, a2),
            "C2": np.where(gone, np.nan, 1.0),
            "L3": np.where(gone, np.nan, l3),
            "A3": np.where(gone, np.nan, a3),
            "C3": np.where(gone, np.nan, 1.0),
            "Y": np.where(gone, np.nan, y),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = LongitudinalData.from_frame(
            frame,
            outcome="Y",
            treatment=["A1", "A2", "A3"],
            baseline=["W"],
            time_varying=[["L1"], ["L2"], ["L3"]],
            censoring=["C1", "C2", "C3"],
        )
    truth = {"p1": p1, "p2": p2, "c1": c1, "n": n}
    return data, truth


def _law_l_mechanism(data: LongitudinalData, truth: dict, **change: np.ndarray) -> Mechanism:
    n = truth["n"]
    p1 = change.get("p1", truth["p1"])
    p2 = change.get("p2", truth["p2"])
    c1 = change.get("c1", truth["c1"])
    return _mechanism(
        [_matrix(p1), _matrix(p2), _matrix(np.full(n, 0.5))],
        [c1, np.ones(n), np.ones(n)],
    )


@pytest.fixture(scope="module")
def law_l() -> tuple[LongitudinalData, dict]:
    return _law_l(50_000, 20261114)


def test_history_stratification_witness() -> None:
    """Law H: A2 depends on A1 only, so within A1 the covariate L2 is balanced across A2."""
    rng = np.random.default_rng(20261115)
    n = 50_000
    a1 = rng.binomial(1, 0.5, n)
    l2 = rng.normal(a1.astype(float))
    p2 = expit(-0.5 + 1.5 * a1)
    a2 = rng.binomial(1, p2)
    frame = pd.DataFrame(
        {"W": rng.normal(size=n), "A1": a1, "L2": l2, "A2": a2, "Y": rng.normal(size=n)}
    )
    data = LongitudinalData.from_frame(
        frame, outcome="Y", treatment=["A1", "A2"], baseline=["W"], time_varying=[[], ["L2"]]
    )
    mechanism = _mechanism([_matrix(np.full(n, 0.5)), _matrix(p2)])
    rows, _, _ = bt.longitudinal_cells(data, mechanism, _OPEN)
    shipped = _unweighted(rows, diagnostic="confounding", exposure="A2", covariate="L2")
    assert abs(shipped) < 0.1, shipped
    # Mutation: no history strata.
    original = bt._stratum_codes

    def flat(matrix: np.ndarray, rows_: np.ndarray) -> np.ndarray:
        return original(matrix[:, :0], rows_)

    try:
        bt._stratum_codes = flat  # type: ignore[assignment]
        rows, _, _ = bt.longitudinal_cells(data, mechanism, _OPEN)
    finally:
        bt._stratum_codes = original  # type: ignore[assignment]
    mutated = _unweighted(rows, diagnostic="confounding", exposure="A2", covariate="L2")
    assert mutated > 0.2, mutated


def test_confounding_weight_witness(law_l: tuple[LongitudinalData, dict]) -> None:
    """Diagnostic 3: the node-2 treatment factor balances L2 across A2 within A1."""
    data, truth = law_l
    rows, _, _ = bt.longitudinal_cells(data, _law_l_mechanism(data, truth), _OPEN)
    match = {"diagnostic": "confounding", "exposure": "A2", "covariate": "L2"}
    assert abs(_value(rows, **match)) < 0.2
    assert _unweighted(rows, **match) > 0.6
    flat = np.full(truth["n"], 0.5)  # drops g2: a constant factor cancels within each group
    rows, _, _ = bt.longitudinal_cells(data, _law_l_mechanism(data, truth, p2=flat), _OPEN)
    assert _value(rows, **match) > 0.6


def test_censoring_diagnostic_witness(law_l: tuple[LongitudinalData, dict]) -> None:
    """Censoring as an exposure: the C1 factor balances L1 across leaving and staying."""
    data, truth = law_l
    rows, _, _ = bt.longitudinal_cells(data, _law_l_mechanism(data, truth), _OPEN)
    match = {"diagnostic": "censoring", "exposure_node": 1, "covariate": "L1"}
    assert abs(_value(rows, **match)) < 0.2
    assert _unweighted(rows, **match) > 0.5  # censored units have larger L1
    flat = np.full(truth["n"], 0.5)
    rows, _, _ = bt.longitudinal_cells(data, _law_l_mechanism(data, truth, c1=flat), _OPEN)
    assert _value(rows, **match) > 0.5


def test_feedback_retention_witness() -> None:
    """Diagnostic 2a: the retention after A1 enters the feedback weight (law L')."""
    data, truth = _law_l(100_000, 20261116, strong_dropout=True)
    match = {"diagnostic": "feedback", "exposure": "A1", "covariate": "L2"}
    rows, _, _ = bt.longitudinal_cells(data, _law_l_mechanism(data, truth), _OPEN)
    assert abs(_value(rows, **match) - 0.726) < 0.053
    ones = np.ones(truth["n"])
    rows, _, _ = bt.longitudinal_cells(data, _law_l_mechanism(data, truth, c1=ones), _OPEN)
    assert abs(_value(rows, **match) - 0.726) >= 0.053


def test_feedback_weight_is_not_cumulative(law_l: tuple[LongitudinalData, dict]) -> None:
    """Diagnostic 2a reweights the earlier treatment alone, not the treatments after it."""
    data, truth = law_l
    match = {"diagnostic": "feedback", "exposure": "A1", "covariate": "L3"}
    rows, _, _ = bt.longitudinal_cells(data, _law_l_mechanism(data, truth), _OPEN)
    assert _value(rows, **match) > 0.68
    # Mutation: fold the node-2 factor into the node-1 factor of each unit's own arm.
    g2 = np.where(np.nan_to_num(data.treatment[:, 1]) == 1, truth["p2"], 1 - truth["p2"])
    a1 = data.treatment[:, 0].astype(int)
    matrix = _matrix(truth["p1"])
    matrix[np.arange(truth["n"]), a1] *= np.where(np.isfinite(data.treatment[:, 1]), g2, 1.0)
    mechanism = _law_l_mechanism(data, truth)
    mutated = Mechanism(
        treatment=mechanism.treatment,
        censoring=mechanism.censoring,
        treatment_observed=(matrix, *mechanism.treatment_observed[1:]),
        censoring_observed=mechanism.censoring_observed,
    )
    rows, _, _ = bt.longitudinal_cells(data, mutated, _OPEN)
    assert _value(rows, **match) < 0.68


# ------------------------------------------------------------------------ continuous time


def _law_k(n: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    p1 = expit(0.5 * w)
    a = rng.binomial(1, p1)
    event = rng.exponential(1 / (0.5 * np.exp(0.3 * a + 0.8 * w)))
    rate = lambda arm: 0.4 * np.exp(0.8 * w + 0.8 * arm * w)  # noqa: E731
    censor = rng.exponential(1 / rate(a))
    time = np.minimum(event, censor)
    code = (event <= censor).astype(float)
    horizon = 1.0
    before = np.stack([np.exp(-rate(0) * horizon), np.exp(-rate(1) * horizon)])[..., None]
    return {
        "time": time,
        "event": code,
        "arm": a,
        "baseline": w[:, None],
        "p1": p1,
        "retention_before": before,
        "retention_at_time": np.exp(-rate(a) * time),
        "horizon": horizon,
    }


def _continuous(law: dict[str, Any], **change: Any) -> list[bt.TimeBalanceRow]:
    values = {**law, **change}
    rows, _ = bt.continuous_cells(
        time=values["time"],
        event=values["event"],
        arm=values["arm"],
        baseline=values["baseline"],
        names=("W",),
        arm_labels=("0", "1"),
        treatment="A",
        p1=values["p1"],
        retention_before=values["retention_before"],
        retention_at_time=values["retention_at_time"],
        weight_floor=1e-9,
        horizons=(values["horizon"],),
    )
    return rows


@pytest.fixture(scope="module")
def law_k() -> dict[str, Any]:
    return _law_k(200_000, 20261117)


def test_continuous_joint_witness(law_k: dict[str, Any]) -> None:
    """Among units whose status at t is known, 1/(pi S_c) balances W across the arms."""
    rows = _continuous(law_k)
    assert abs(_value(rows, diagnostic="joint")) < 0.08
    assert _unweighted(rows, diagnostic="joint") > 0.2
    rows = _continuous(
        law_k,
        retention_before=np.ones_like(law_k["retention_before"]),
        retention_at_time=np.ones_like(law_k["retention_at_time"]),
    )
    assert _value(rows, diagnostic="joint") < -0.12


def test_continuous_censoring_witness(law_k: dict[str, Any]) -> None:
    """Within each arm, 1/S_c makes the status-known units look like the whole arm."""
    rows = _continuous(law_k)
    arm1 = _value(rows, diagnostic="censoring", levels="known vs all, 1")
    arm0 = _value(rows, diagnostic="censoring", levels="known vs all, 0")
    assert abs(arm1) < 0.15 and abs(arm0) < 0.08
    # Mutation: a censoring model that ignores W, one constant hazard per arm.
    time, arm, event = law_k["time"], law_k["arm"], law_k["event"]
    rates = np.array([((event == 0) & (arm == a)).sum() / time[arm == a].sum() for a in (0, 1)])
    rows = _continuous(
        law_k,
        retention_before=np.broadcast_to(
            np.exp(-rates[:, None, None] * law_k["horizon"]), law_k["retention_before"].shape
        ).copy(),
        retention_at_time=np.exp(-rates[arm] * time),
    )
    assert _value(rows, diagnostic="censoring", levels="known vs all, 1") < -0.25
    assert _value(rows, diagnostic="censoring", levels="known vs all, 0") < -0.12


# ------------------------------------------------------------------- exact hand calculation


def test_hand_computed_cells() -> None:
    """Ten units, two nodes: every cell, spread, size and average against a hand calculation."""
    frame = pd.DataFrame(
        {
            "W": [0.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 0.0, 3.0, 1.0],
            "A1": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "C1": [1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
            "L2": [1.0, 2.0, 0.0, np.nan, 3.0, 2.0, 4.0, 1.0, np.nan, 5.0],
            "A2": [0, 1, 0, np.nan, 1, 1, 0, 1, np.nan, 1],
            "C2": [1, 1, 1, np.nan, 1, 1, 1, 1, np.nan, 1],
            "Y": [0.0, 1.0, 0.0, np.nan, 1.0, 1.0, 0.0, 1.0, np.nan, 1.0],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = LongitudinalData.from_frame(
            frame,
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W"],
            time_varying=[[], ["L2"]],
            censoring=["C1", "C2"],
        )
    p1 = np.array([0.4, 0.5, 0.5, 0.2, 0.25, 0.5, 0.8, 0.5, 0.5, 0.5])
    p2 = np.array([0.5, 0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.25, 0.5, 0.75])
    c1 = np.array([0.8, 0.5, 1.0, 0.5, 0.5, 0.5, 0.5, 1.0, 0.5, 0.8])
    mechanism = _mechanism([_matrix(p1), _matrix(p2)], [c1, np.ones(10)])
    rows, _, share = bt.longitudinal_cells(data, mechanism, (0.05, 1.0))
    # Confounding at A2, stratum A1 = 0, covariate W: rows 0, 1, 2, 4 (row 3 censored).
    (cell,) = [
        r
        for r in rows
        if r.diagnostic == "confounding"
        and r.exposure == "A2"
        and r.history == "A1 = 0"
        and r.covariate == "W"
    ]
    # A2 = 1: rows 1 (W=1) and 4 (W=4); A2 = 0: rows 0 (W=0) and 2 (W=2).
    assert cell.mean_group == pytest.approx(2.5) and cell.mean_rest == pytest.approx(1.0)
    # Weights 1/(g1 g2 c1): row 1: 1/(0.5*0.25*0.5)=16; row 4: 1/(0.75*0.5*0.5)=16/3;
    # row 0: 1/(0.6*0.5*0.8)=1/0.24; row 2: 1/(0.5*0.5*1.0)=4.
    w1, w4, w0, w2 = 16.0, 16.0 / 3.0, 1 / 0.24, 4.0
    assert cell.mean_group_weighted == pytest.approx((w1 * 1 + w4 * 4) / (w1 + w4))
    assert cell.mean_rest_weighted == pytest.approx((w0 * 0 + w2 * 2) / (w0 + w2))
    # Spread: pooled sample variances, (4.5 + 2) / 2.
    assert cell.spread == pytest.approx(np.sqrt(3.25))
    assert cell.smd_unweighted == pytest.approx(1.5 / np.sqrt(3.25))
    assert cell.person_time_unweighted == pytest.approx(4.0)
    # Stabilized: numerators P_n(A1 = 0) = 0.5, P_n(A2 = a | A1 = 0) = 0.5 among four,
    # and the marginal retention after node 1, 8 of 10.
    stab = 0.5 * 0.5 * 0.8
    assert cell.person_time_weighted == pytest.approx(stab * (w1 + w4 + w0 + w2))
    # Stabilizing never moves a cell mean: the numerators are constant within each group.
    assert cell.difference_weighted == pytest.approx(
        cell.mean_group_weighted - cell.mean_rest_weighted
    )
    # Censoring at node 1, stratum A1 = 0, covariate W: censored row 3 (W=3) against rows
    # 0, 1, 2, 4; weights 1/(g1 * (1 - c1)) and 1/(g1 * c1), unbounded here.
    (cens,) = [
        r
        for r in rows
        if r.diagnostic == "censoring"
        and r.exposure_node == 1
        and r.history == "A1 = 0"
        and r.covariate == "W"
    ]
    assert cens.mean_group == pytest.approx(3.0)
    assert cens.mean_rest == pytest.approx(7.0 / 4.0)
    weights = np.array([1 / (0.6 * 0.8), 1 / (0.5 * 0.5), 1 / (0.5 * 1.0), 1 / (0.75 * 0.5)])
    assert cens.mean_rest_weighted == pytest.approx(np.average([0, 1, 2, 4], weights=weights))
    # The bound: no product falls below 0.05 here.
    assert share == 0.0
    rows, _, share = bt.longitudinal_cells(data, mechanism, (0.2, 1.0))
    assert share > 0.0
    (bounded,) = [
        r
        for r in rows
        if r.diagnostic == "confounding"
        and r.exposure == "A2"
        and r.history == "A1 = 0"
        and r.covariate == "W"
    ]
    # Row 1's product 0.0625 and row 4's 0.1875 rise to 0.2: equal weights in that group.
    assert bounded.mean_group_weighted == pytest.approx(2.5)
    # Averages: the history average weights each stratum by its size.
    (average,) = bt.averages(
        [
            r
            for r in rows
            if r.diagnostic == "confounding" and r.exposure == "A2" and r.covariate == "W"
        ],
        over="history",
    )
    strata = [
        r
        for r in rows
        if r.diagnostic == "confounding" and r.exposure == "A2" and r.covariate == "W"
    ]
    # A stratum whose spread is undefined (one row in a group) leaves the average.
    finite = [r for r in strata if np.isfinite(r.smd_weighted)]
    assert len(finite) < len(strata)
    expected = sum(r.smd_weighted * r.person_time_weighted for r in finite) / sum(
        r.person_time_weighted for r in finite
    )
    assert average["smd_weighted"] == pytest.approx(expected)
    assert average["smd_min"] == min(r.smd_weighted for r in strata if np.isfinite(r.smd_weighted))


# --------------------------------------------------------------------- fits and refusals


def _fit(name: str) -> Any:
    from tests.unit import _capability_sweep_support as sweep

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return getattr(sweep, name)()


@pytest.mark.parametrize(
    "name",
    [
        "fit_hazard_multinode",
        "fit_hazard_time_varying",
        "fit_hazard_mtp",
        "fit_hazard_msm",
        "fit_continuous_survival",
        "fit_ltmle",
        "fit_sdr",
        "fit_ipsi",
        "fit_ipw",
        "fit_survival_ctmle",
        "fit_survival_drtmle",
        "fit_learned_regimen",
        "fit_longitudinal_msm",
        "fit_competing",
        "fit_repeated",
        "fit_stochastic_categorical",
        "fit_categorical_mtp",
        "fit_eliminated",
        "fit_separable",
    ],
)
def test_every_family_builds_a_report(name: str) -> None:
    balance = covariate_balance(_fit(name))
    assert isinstance(balance, BalanceOverTime)
    assert balance.rows and "confounding" in balance.diagnostics
    first = [r for r in balance.rows if r.diagnostic == "confounding" and r.exposure_node == 1]
    # Node 1 has one stratum, so each covariate's rows count every unit at risk there once
    # per comparison.
    covariate = first[0].covariate
    contrasts = {r.contrast for r in first}
    units = [r.n_units for r in first if r.covariate == covariate]
    if len(contrasts) == 1:
        assert units == [balance.n]
    else:
        # Each comparison holds its arm and the reference arm; together they cover everyone,
        # the reference arm once per comparison.
        assert all(count < balance.n for count in units) and sum(units) > balance.n
    frame = balance.to_frame()
    assert list(frame.columns) == list(bt.ROW_COLUMNS)
    assert len(balance.average()) > 0
    assert str(balance.summary())


def test_survival_fit_omits_feedback() -> None:
    balance = covariate_balance(_fit("fit_hazard_multinode"))
    assert dict(balance.omitted)["feedback"] == bt.FEEDBACK_SURVIVAL_REASON
    with pytest.raises(CapabilityError, match="mixes its effect on the covariate"):
        balance.average(diagnostic="feedback")
    from cleverly.plotting import balance as plot

    with pytest.raises(CapabilityError, match="mixes its effect on the covariate"):
        plot(balance, diagnostic="feedback")


def test_eliminated_causes_pair_each_indicator_with_its_factor() -> None:
    """On a fit that eliminates a cause, censoring and the eliminated cause are two exposures."""
    result = _fit("fit_eliminated")
    data, mechanism = result.data, result.mechanism
    assert data.eliminated and mechanism.censoring_factor_observed
    rows, _, _ = bt.longitudinal_cells(data, mechanism, tuple(result.config.g_bounds))
    exposures = {r.exposure for r in rows if r.diagnostic == "censoring"}
    assert any(e.startswith("censoring at") for e in exposures)
    assert any(e.startswith("eliminated causes at") for e in exposures)
    # Mutation: read the censoring node with the composite factor (censoring times the
    # competing factor) instead of the censoring factor alone.
    mutated = Mechanism(
        treatment=mechanism.treatment,
        censoring=mechanism.censoring,
        treatment_observed=mechanism.treatment_observed,
        censoring_observed=mechanism.censoring_observed,
        competing_observed=mechanism.competing_observed,
        censoring_factor_observed=mechanism.censoring_observed,
    )
    changed, _, _ = bt.longitudinal_cells(data, mutated, tuple(result.config.g_bounds))

    def weighted(items: list[bt.TimeBalanceRow]) -> list[float]:
        return [
            r.mean_group_weighted
            for r in items
            if r.diagnostic == "censoring" and r.exposure.startswith("censoring at")
        ]

    gaps = np.abs(np.asarray(weighted(rows)) - np.asarray(weighted(changed)))
    assert np.nanmax(gaps) > 1e-3


def test_collaborative_fit_carries_its_reading() -> None:
    balance = covariate_balance(_fit("fit_survival_ctmle"))
    assert balance.notes and "collaborative selection" in balance.notes[0]
    assert "collaborative selection" in str(balance.summary())


def test_held_treatment_has_one_exposure() -> None:
    balance = covariate_balance(_fit("fit_hazard_msm"))
    exposures = {r.exposure for r in balance.rows if r.diagnostic == "confounding"}
    assert len(exposures) == 1
    assert {r.exposure_node for r in balance.rows if r.diagnostic == "censoring"}


def test_refusals() -> None:
    with pytest.raises(DataError, match="is a continuous treatment"):
        covariate_balance(_fit("fit_ltmle_mtp_censored"))
    result = _fit("fit_hazard_multinode")
    with pytest.raises(ValueError, match="estimand= chooses the point-treatment weights"):
        covariate_balance(result, estimand="ate")
    with pytest.raises(ValueError, match="threshold must be positive"):
        covariate_balance(result, threshold=0.0)
    continuous = _fit("fit_continuous_survival")
    import dataclasses

    hand_built = dataclasses.replace(continuous, data=None)
    with pytest.raises(CapabilityError, match="built by hand and holds none"):
        covariate_balance(hand_built)


def test_point_keywords_refused_on_a_point_fit() -> None:
    from sklearn.linear_model import LinearRegression, LogisticRegression

    from cleverly import ATE, CausalStudy, PointTreatment
    from cleverly.datasets import make_linear_ate
    from cleverly.plotting import balance as plot

    frame, _ = make_linear_ate(n=200, seed=0)
    study = CausalStudy(
        frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2"))
    )
    point = study.identify(ATE()).estimate(
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        cross_fit=False,
        random_state=0,
    )
    with pytest.raises(ValueError, match="applies to the balance of a fit over time"):
        plot(point, view="panels")


def test_engine_stores_retention_at_each_units_time() -> None:
    """The record's retention at a unit's own time is the retention read at its grid index."""
    result = _fit("fit_continuous_survival")
    record, data = result.record, result.data
    time = np.asarray(data.time, dtype=float)
    # A unit followed past the last target time reads the retention just before the grid's
    # end, which for this fit is the last target time's stored value when no observed time
    # falls between the last target time and the grid's end.
    last = max(result.horizons)
    beyond = time > last
    grid_end = float(np.max(time[time <= last]))
    if grid_end == last:
        pytest.skip("an observed time equals the last target time")
    stored = record.retention_before[record.arm[beyond], np.flatnonzero(beyond), -1]
    # retention_before[-1] is S_c(last^-), the product over observed times before `last`;
    # retention_at_time for these units is S_c(grid_end^-), which excludes the jump at
    # grid_end itself, so it is at least as large.
    assert np.all(record.retention_at_time[beyond] >= stored - 1e-12)
    assert np.all((record.retention_at_time > 0) & (record.retention_at_time <= 1))


def test_retention_at_time_is_read_just_before_each_units_own_time() -> None:
    """At a target time equal to a unit's own time, both stored retentions are S_c(t^-).

    The target times are two censored units' own times, so each of those units has a censoring
    jump at its own time: a record that read S_c(T) inclusive of that jump, rather than just
    before it, differs from ``retention_before`` there and fails.
    """
    from cleverly.survival import ContinuousSurvivalTMLE, CoxHazard

    rng = np.random.default_rng(0)
    n = 300
    w = rng.normal(size=n)
    a = rng.binomial(1, 1 / (1 + np.exp(-w)))
    hazard = np.exp(-1.5 + 0.5 * w - a)
    t = rng.exponential(1 / hazard)
    c = rng.exponential(1 / 0.3, n)
    time = np.minimum(t, c)
    event = (t <= c).astype(int)
    censored = np.sort(time[event == 0])
    targets = [float(censored[len(censored) // 3]), float(censored[2 * len(censored) // 3])]
    frame = pd.DataFrame({"time": time, "event": event, "A": a, "W": w})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ContinuousSurvivalTMLE(
            targets,
            cross_fit=False,
            hazard_library=[CoxHazard("main")],
            censoring_library=[CoxHazard("main")],
            random_state=0,
        ).fit(frame, time="time", event="event", treatment="A", baseline=["W"])
    record = result.record
    checked = 0
    for k, target in enumerate(targets):
        units = np.flatnonzero(time == target)
        assert units.size
        for i in units:
            arm = int(record.arm[i])
            assert record.retention_at_time[i] == pytest.approx(
                record.retention_before[arm, i, k], rel=1e-12, abs=1e-15
            )
            assert record.retention_at_time[i] < 1.0  # earlier censoring jumps are in it
            checked += 1
    assert checked >= 2


def test_the_bound_leaves_each_units_own_factor_out() -> None:
    """A binding bound clips the earlier path only, so correct models still balance.

    Law L' with strong dropout makes ``1 - c_1`` small for many units. Clipping the whole
    product with the unit's own censoring outcome inside it moves the censoring row's weighted
    difference to about 0.35 at ``g_bounds=(0.05, 1)``; bounding the earlier path alone keeps it
    near zero, and the feedback row near its truth.
    """
    data, truth = _law_l(50_000, 20261114, strong_dropout=True)
    bounds = (0.05, 1.0)
    rows, _, _ = bt.longitudinal_cells(data, _law_l_mechanism(data, truth), bounds)
    censoring = {"diagnostic": "censoring", "exposure_node": 1, "covariate": "L1"}
    assert abs(_value(rows, **censoring)) < 0.1
    feedback = {"diagnostic": "feedback", "exposure": "A1", "covariate": "L2"}
    assert abs(_value(rows, **feedback) - 0.726) < 0.03
