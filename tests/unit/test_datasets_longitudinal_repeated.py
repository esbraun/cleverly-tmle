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


def _outcome(w: np.ndarray, a: np.ndarray, l: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``P(Y_t = 1 | W, A_t, L_t, Y_{t-1})``, written from the declared coefficients."""
    return expit(-1.6 + 0.5 * w + 0.7 * a + 0.4 * l + 1.4 * y - 0.8 * a * y)


def _counterfactual_draw(plan: str, n: int, seed: int) -> np.ndarray:
    """``(Y_1^d, Y_2^d, Y_3^d)`` by forward simulation under ``plan``, censoring removed."""
    rng = np.random.default_rng(seed)
    w = rng.binomial(1, 0.5, size=n).astype(float)
    y_prev = np.zeros(n)
    a_prev = np.zeros(n)
    outcomes = []
    for time in range(1, 4):
        if time == 1:
            l_t = np.zeros(n)
        else:
            l_t = rng.binomial(1, expit(-0.3 + 0.3 * w - 0.5 * a_prev + 0.3 * y_prev)).astype(float)
        if plan == "never":
            a_t = np.zeros(n)
        elif plan == "always" or time == 1:
            a_t = np.ones(n)
        else:
            a_t = y_prev.copy()
        y_t = rng.binomial(1, _outcome(w, a_t, l_t, y_prev)).astype(float)
        outcomes.append(y_t)
        y_prev, a_prev = y_t, a_t
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
    w, a, l = np.meshgrid([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
    persistence = _outcome(w, a, l, np.ones_like(w))
    assert float(np.min(1.0 - persistence)) >= 0.33
    # The empirical return rate agrees: the outcome is not absorbing.
    frame, _ = make_longitudinal_repeated(n=20_000, seed=4, censoring=False, backend="pandas")
    after_one = frame.loc[frame["Y1"] == 1.0, "Y2"]
    assert float((after_one == 0.0).mean()) >= 0.3


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
    for w in (0.0, 1.0):
        p_treat = float(expit(-0.2 + 0.6 * w))
        for a, p_a in ((0.0, 1.0 - p_treat), (1.0, p_treat)):
            expected += 0.5 * p_a * float(_outcome(np.array(w), np.array(a), 0.0, 0.0))
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
