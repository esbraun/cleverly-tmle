"""Delayed entry in the continuous-time hazard learners (X6 part (e), T-e13).

A row with ``entry=e`` and time ``t`` is at risk on ``(e, t]``.  On six rows the Breslow jumps,
the Nelson-Aalen jumps and the log partial likelihood equal a hand computation over those risk
sets, and the same fit read with every row entered at zero (``entry=None``) is the shipped fit.
The mutation drops ``entry`` from the risk sets, and the hand check then fails, so the check
can see the term.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly.survival import CoxHazard, StratifiedNelsonAalen, cox, select_hazard
from cleverly.survival.cox import partial_log_likelihood

#: Six rows, two of them entering late: (entry, time, event, A, L).
ROWS = np.array(
    [
        [0.0, 1.0, 1.0, 0.0, 1.2],
        [0.0, 3.0, 0.0, 1.0, -0.5],
        [1.5, 2.0, 1.0, 1.0, 0.1],
        [0.0, 2.0, 1.0, 0.0, 0.8],
        [2.5, 4.0, 1.0, 0.0, -0.3],
        [1.0, 4.0, 0.0, 1.0, 0.4],
    ]
)
ENTRY, TIME, EVENT = ROWS[:, 0], ROWS[:, 1], ROWS[:, 2]
X = ROWS[:, 3:]


def _risk(entry: np.ndarray, time: np.ndarray, t: float) -> np.ndarray:
    return (entry < t) & (t <= time)


def _hand_loglik(eta: np.ndarray, entry: np.ndarray) -> float:
    total = 0.0
    for t in np.unique(TIME[EVENT > 0]):
        dead = (t == TIME) & (EVENT > 0)
        total += float(
            eta[dead].sum() - dead.sum() * np.log(np.exp(eta[_risk(entry, TIME, t)]).sum())
        )
    return total


def _hand_jumps(eta: np.ndarray, entry: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """``(n, M)`` each row's Breslow hazard jump at each grid time."""
    out = np.zeros((eta.size, grid.size))
    for m, t in enumerate(grid):
        dead = (t == TIME) & (EVENT > 0)
        if dead.any():
            out[:, m] = dead.sum() * np.exp(eta) / np.exp(eta[_risk(entry, TIME, t)]).sum()
    return out


def _jumps(model: object, grid: np.ndarray) -> np.ndarray:
    cumulative = model.cumulative_hazard(X, grid)  # type: ignore[attr-defined]
    return np.diff(np.concatenate([np.zeros((X.shape[0], 1)), cumulative], axis=1), axis=1)


GRID = np.unique(TIME)


def test_cox_breslow_jumps_and_partial_likelihood_match_the_hand_computation() -> None:
    model = CoxHazard("main").fit(X, TIME, EVENT, entry=ENTRY)
    eta = model.linear_predictor(X)
    np.testing.assert_allclose(_jumps(model, GRID), _hand_jumps(eta, ENTRY, GRID), atol=1e-12)
    assert partial_log_likelihood(model, X, TIME, EVENT, ENTRY) == pytest.approx(
        _hand_loglik(eta, ENTRY), abs=1e-12
    )
    # The fit maximizes the delayed-entry likelihood: no coordinate step raises it.
    for column in range(X.shape[1]):
        for sign in (-1.0, 1.0):
            moved = eta + sign * 1e-4 * (X[:, column] - X[:, column].mean())
            assert _hand_loglik(moved, ENTRY) <= _hand_loglik(eta, ENTRY) + 1e-12


def test_nelson_aalen_jumps_match_the_hand_counts() -> None:
    model = StratifiedNelsonAalen(by=(0,)).fit(X, TIME, EVENT, entry=ENTRY)
    expected = np.zeros((X.shape[0], GRID.size))
    for i in range(X.shape[0]):
        cell = X[:, 0] == X[i, 0]
        for m, t in enumerate(GRID):
            dead = cell & (t == TIME) & (EVENT > 0)
            if dead.any():
                expected[i, m] = dead.sum() / (cell & _risk(ENTRY, TIME, t)).sum()
    np.testing.assert_array_equal(_jumps(model, GRID), expected)


def test_select_hazard_passes_entry_to_every_fit() -> None:
    folds = np.array([0, 1, 0, 1, 0, 1])
    library = (CoxHazard("treatment"), CoxHazard("main"))
    chosen = select_hazard(library, X, TIME, EVENT, folds, entry=ENTRY)
    refit = CoxHazard(chosen.model.terms).fit(X, TIME, EVENT, entry=ENTRY)
    np.testing.assert_array_equal(chosen.model.coef_, refit.coef_)
    ignored = select_hazard(library, X, TIME, EVENT, folds)
    assert ignored.losses != chosen.losses


def test_entry_none_and_entry_at_zero_are_the_shipped_fit_bit_for_bit() -> None:
    shipped = CoxHazard("main").fit(X, TIME, EVENT)
    zero = CoxHazard("main").fit(X, TIME, EVENT, entry=np.zeros(X.shape[0]))
    np.testing.assert_array_equal(shipped.coef_, zero.coef_)
    np.testing.assert_array_equal(shipped.baseline_jumps_, zero.baseline_jumps_)
    assert partial_log_likelihood(shipped, X, TIME, EVENT) == partial_log_likelihood(
        shipped, X, TIME, EVENT, np.zeros(X.shape[0])
    )
    na = StratifiedNelsonAalen(by=(0,)).fit(X, TIME, EVENT)
    na_zero = StratifiedNelsonAalen(by=(0,)).fit(X, TIME, EVENT, entry=np.zeros(X.shape[0]))
    np.testing.assert_array_equal(_jumps(na, GRID), _jumps(na_zero, GRID))


def test_mutation_ignoring_entry_fails_the_hand_check(monkeypatch: pytest.MonkeyPatch) -> None:
    shipped = cox._risk_sets
    monkeypatch.setattr(cox, "_risk_sets", lambda time, event, entry=None: shipped(time, event))
    model = CoxHazard("main").fit(X, TIME, EVENT, entry=ENTRY)
    eta = model.linear_predictor(X)
    assert np.max(np.abs(_jumps(model, GRID) - _hand_jumps(eta, ENTRY, GRID))) > 1e-3
