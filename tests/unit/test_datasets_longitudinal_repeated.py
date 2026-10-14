"""The repeated-outcome longitudinal generator and the enumeration its truths come from.

Every truth an accuracy claim about :func:`make_longitudinal_repeated` reads is a number
:func:`repeated_truth` produced.  These tests compare that number against a forward
simulation written here from the declared coefficients, so a wrong enumeration and a wrong
generator cannot agree by sharing one helper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from cleverly.datasets import (
    REPEATED_RULE_LABEL,
    make_longitudinal_repeated,
    repeated_truth,
)

COLUMNS = ["W", "A1", "C1", "Y1", "L2", "A2", "C2", "Y2", "L3", "A3", "C3", "Y3"]
PLANS = ("never", "always", REPEATED_RULE_LABEL)


def _outcome(
    w: np.ndarray,
    a: np.ndarray,
    l: np.ndarray,
    y: np.ndarray,
    y2: np.ndarray,
    l1: np.ndarray,
) -> np.ndarray:
    """``P(Y_t = 1 | W, A_t, L_t, Y_{t-1}, Y_{t-2}, L_{t-1})``, from the declared coefficients."""
    return expit(-1.6 + 0.3 * w - 0.7 * a + 0.4 * l + 0.3 * y - 1.1 * a * y + 1.8 * y2 + 0.2 * l1)


def _covariate(w: np.ndarray, a: np.ndarray, y: np.ndarray, l1: np.ndarray) -> np.ndarray:
    """``P(L_t = 1 | W, A_{t-1}, Y_{t-1}, L_{t-1})``, from the declared coefficients."""
    return expit(-0.4 + 0.3 * w - 2.0 * a + 1.4 * y + 0.9 * l1)


def _counterfactual_draw(plan: str, n: int, seed: int) -> np.ndarray:
    """``(Y_1^d, Y_2^d, Y_3^d)`` by forward simulation under ``plan``, censoring removed."""
    rng = np.random.default_rng(seed)
    w = rng.integers(0, 3, size=n).astype(float)
    y_prev = np.zeros(n)
    y_prev2 = np.zeros(n)
    a_prev = np.zeros(n)
    l_prev = np.zeros(n)
    outcomes = []
    for time in range(1, 4):
        if time == 1:
            l_t = np.zeros(n)
        else:
            l_t = rng.binomial(1, _covariate(w, a_prev, y_prev, l_prev)).astype(float)
        if plan == "never":
            a_t = np.zeros(n)
        elif plan == "always" or time == 1:
            a_t = np.ones(n)
        else:
            a_t = y_prev.copy()
        y_t = rng.binomial(1, _outcome(w, a_t, l_t, y_prev, y_prev2, l_prev)).astype(float)
        outcomes.append(y_t)
        y_prev2, y_prev, a_prev, l_prev = y_prev, y_t, a_t, l_t
    return np.column_stack(outcomes)


def test_the_frame_has_the_declared_columns_and_truth_keys() -> None:
    frame, truth = make_longitudinal_repeated(n=50, seed=1, backend="pandas")
    assert list(frame.columns) == COLUMNS
    means = {f"ey_regimen[{plan} @ t={h}]" for plan in PLANS for h in (1, 2, 3)}
    contrasts = {f"ate_regimen[{plan} vs never @ t={h}]" for plan in PLANS[1:] for h in (1, 2, 3)}
    assert set(truth) == means | contrasts
    for plan in PLANS[1:]:
        for h in (1, 2, 3):
            assert truth[f"ate_regimen[{plan} vs never @ t={h}]"] == pytest.approx(
                truth[f"ey_regimen[{plan} @ t={h}]"] - truth[f"ey_regimen[never @ t={h}]"]
            )


def test_without_censoring_every_node_is_observed() -> None:
    frame, _ = make_longitudinal_repeated(n=200, seed=2, censoring=False, backend="pandas")
    assert list(frame.columns) == [name for name in COLUMNS if not name.startswith("C")]
    assert not frame.isna().any().any()


def test_censoring_is_monotone_and_masks_exactly_the_later_nodes() -> None:
    frame, _ = make_longitudinal_repeated(n=4_000, seed=3, backend="pandas")
    alive = pd.Series(True, index=frame.index)
    for time in range(1, 4):
        nodes = ([f"L{time}"] if time > 1 else []) + [f"A{time}", f"C{time}"]
        for node in nodes:
            assert frame.loc[alive, node].notna().all(), node
            assert frame.loc[~alive, node].isna().all(), node
        observed = alive & (frame[f"C{time}"] == 1.0)
        assert frame.loc[observed, f"Y{time}"].notna().all()
        assert frame.loc[~observed, f"Y{time}"].isna().all()
        alive = observed
    # Every node censors some units, and some units reach the end.
    assert (frame[["C1", "C2", "C3"]] == 0.0).any().all()
    assert frame["Y3"].notna().mean() > 0.4


def test_the_outcome_returns_to_zero_in_every_cell() -> None:
    w, a, l, y2, l1 = np.meshgrid(
        [0.0, 1.0, 2.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], indexing="ij"
    )
    persistence = _outcome(w, a, l, np.ones_like(w), y2, l1)
    assert float(np.min(1.0 - persistence)) >= 0.15
    # The bound is tight: the cell with every positive term on returns with probability 0.154.
    assert float(np.min(1.0 - persistence)) == pytest.approx(0.1545, abs=1e-3)
    # The empirical return rate agrees: the outcome is not absorbing.
    frame, _ = make_longitudinal_repeated(n=20_000, seed=4, censoring=False, backend="pandas")
    after_one = frame.loc[frame["Y1"] == 1.0, "Y2"]
    assert float((after_one == 0.0).mean()) >= 0.3


def test_the_baseline_covariate_takes_three_equally_likely_values() -> None:
    frame, _ = make_longitudinal_repeated(n=90_000, seed=6, censoring=False, backend="pandas")
    shares = frame["W"].value_counts(normalize=True).sort_index()
    assert list(shares.index) == [0.0, 1.0, 2.0]
    np.testing.assert_allclose(shares.to_numpy(), 1.0 / 3.0, atol=4.5 * np.sqrt(2 / 9 / 9e4))


def _cell_frequencies(
    frame: pd.DataFrame, target: str, keys: list[str], expected: pd.Series, at_risk: pd.Series
) -> None:
    """Every cell of ``keys`` with 3,000 rows or more reads its declared probability."""
    sample = frame.loc[at_risk]
    grouped = sample.groupby(keys)[target]
    checked = 0
    for cell, column in grouped:
        if len(column) < 3_000:
            continue
        probability = float(expected.loc[column.index].iloc[0])
        spread = np.sqrt(probability * (1.0 - probability) / len(column))
        assert float(column.mean()) == pytest.approx(probability, abs=5.0 * spread), (target, cell)
        checked += 1
    assert checked >= 6, (target, checked)


def test_the_generator_samples_the_declared_second_order_law() -> None:
    """``L_3``, ``A_3``, ``C_3`` and ``Y_3`` read two nodes back, cell by cell."""
    frame, _ = make_longitudinal_repeated(n=600_000, seed=7, censoring=True, backend="pandas")
    w, y1, l2, a2, y2 = (frame[name] for name in ("W", "Y1", "L2", "A2", "Y2"))
    # ``L_3`` among units still observed at node 3.
    alive = frame["L3"].notna()
    expected_l = pd.Series(_covariate(w, a2, y2, l2), index=frame.index)
    _cell_frequencies(frame, "L3", ["W", "A2", "Y2", "L2"], expected_l, alive)
    expected_a = pd.Series(
        expit(-1.2 + 0.2 * w + 0.5 * frame["L3"] - 0.2 * y2 + 0.7 * a2 + 1.6 * y1),
        index=frame.index,
    )
    _cell_frequencies(frame, "A3", ["W", "L3", "Y2", "A2", "Y1"], expected_a, alive)
    expected_c = pd.Series(
        expit(3.6 + 0.3 * frame["L3"] - 0.2 * y2 + 0.3 * frame["A3"] - 1.4 * y1),
        index=frame.index,
    )
    _cell_frequencies(frame, "C3", ["L3", "Y2", "A3", "Y1"], expected_c, alive)
    # ``Y_3`` among the units observed through node 3.
    observed = frame["Y3"].notna()
    expected_y = pd.Series(
        _outcome(w, frame["A3"], frame["L3"], y2, y1, l2),
        index=frame.index,
    )
    _cell_frequencies(frame, "Y3", ["W", "A3", "L3", "Y2", "Y1", "L2"], expected_y, observed)


def test_the_second_order_terms_are_live_in_the_generator() -> None:
    """A nonzero witness: dropping ``Y_{t-2}`` or ``L_{t-1}`` moves the outcome probability."""
    w, a, l, y, y2, l1 = (np.array(value) for value in (1.0, 0.0, 1.0, 0.0, 1.0, 1.0))
    full = float(_outcome(w, a, l, y, y2, l1))
    assert full - float(_outcome(w, a, l, y, 0.0 * y2, l1)) > 0.2
    assert full - float(_outcome(w, a, l, y, y2, 0.0 * l1)) > 0.02


@pytest.mark.parametrize("plan", PLANS)
def test_the_enumeration_matches_a_forward_simulation(plan: str) -> None:
    n = 400_000
    draws = _counterfactual_draw(plan, n, seed=20263202)
    for horizon in (1, 2, 3):
        column = draws[:, horizon - 1]
        error = float(np.std(column) / np.sqrt(n))
        assert repeated_truth(plan, horizon) == pytest.approx(
            float(np.mean(column)), abs=4.5 * error
        ), (plan, horizon)


def test_the_generator_samples_the_declared_observed_law() -> None:
    """The observed first-node law, enumerated by hand, against a large uncensored draw."""
    frame, _ = make_longitudinal_repeated(n=300_000, seed=5, censoring=False, backend="pandas")
    expected = 0.0
    for w in (0.0, 1.0, 2.0):
        p_treat = float(expit(0.2 + 0.2 * w))
        for a, p_a in ((0.0, 1.0 - p_treat), (1.0, p_treat)):
            zero = np.array(0.0)
            expected += (
                p_a * float(_outcome(np.array(w), np.array(a), zero, zero, zero, zero)) / 3.0
            )
    observed = float(frame["Y1"].mean())
    assert observed == pytest.approx(expected, abs=4.5 * np.sqrt(expected * (1 - expected) / 3e5))


def test_the_rule_duplicates_always_at_the_first_node_and_departs_later() -> None:
    assert repeated_truth(REPEATED_RULE_LABEL, 1) == repeated_truth("always", 1)
    assert abs(repeated_truth(REPEATED_RULE_LABEL, 2) - repeated_truth("always", 2)) > 0.05


def test_the_enumeration_refuses_an_undeclared_plan_or_horizon() -> None:
    with pytest.raises(ValueError, match="horizon"):
        repeated_truth("never", 4)
    with pytest.raises(ValueError, match="plan"):
        repeated_truth("sometimes", 2)
