r"""Finite-support remainder identities for multi-arm TMLE, OAT and DR-TMLE.

The exact multi-arm fit makes every collaborative correction vanish.  This module instead
evaluates the estimating equations at nuisance functions that are wrong on purpose, where
the armwise product remainder and both DR-TMLE projections are observable.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly.fluctuation.iterative import InitialFit
from cleverly.fluctuation.submodel import submodel_for
from cleverly.inference.influence import counterfactual_means
from tests import discrete_law_multi as law

ARMS = tuple(float(arm) for arm in range(law.K))
WRONG_G = np.array(
    [
        [0.40, 0.35, 0.25],
        [0.35, 0.35, 0.30],
        [0.30, 0.30, 0.40],
    ]
)
WRONG_Q = law.Q + np.array(
    [
        [0.12, -0.18, 0.10],
        [-0.10, 0.16, -0.20],
        [0.14, -0.12, -0.15],
    ]
)


def _rows() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = law.frame(labelled=False)
    return (
        frame["W"].to_numpy().astype(int),
        frame["A"].to_numpy(dtype=float),
        frame["Y"].to_numpy(dtype=float),
    )


def _plain(g_hat: np.ndarray, q_hat: np.ndarray) -> dict[float, tuple[float, np.ndarray]]:
    covariate, treatment, outcome = _rows()
    treatment_index = treatment.astype(int)
    initial = InitialFit(
        observed=q_hat[covariate, treatment_index],
        arms={arm: q_hat[covariate, int(arm)] for arm in ARMS},
    )
    submodel = submodel_for("mean", treatment, g_hat[covariate], arms=ARMS, reference=0.0)
    means = counterfactual_means(outcome, initial, submodel, np.ones(law.N))
    return {arm: (mean.psi, np.asarray(mean.influence_curve)) for arm, mean in means.items()}


def _product(g_hat: np.ndarray, q_hat: np.ndarray, arm: float) -> float:
    column = int(arm)
    return float(
        np.sum(
            law.P_W
            * (g_hat[:, column] - law.G[:, column])
            / g_hat[:, column]
            * (q_hat[:, column] - law.Q[:, column])
        )
    )


def _extra(g_hat: np.ndarray, q_hat: np.ndarray, arm: float) -> tuple[np.ndarray, np.ndarray]:
    """The two saturated armwise corrections, independently of library reduction code."""
    covariate, treatment, outcome = _rows()
    column = int(arm)
    mechanism = g_hat[covariate, column]
    indicator = (treatment == arm).astype(float)
    qr = law.Q[:, column] - q_hat[:, column]
    gr1 = law.G[:, column]
    gr2 = (law.G[:, column] - g_hat[:, column]) / g_hat[:, column]
    d_g = qr[covariate] / mechanism * (indicator - mechanism)
    d_q = indicator * gr2[covariate] / gr1[covariate] * (outcome - q_hat[covariate, column])
    return d_g, d_q


def _expansion(
    g_hat: np.ndarray,
    q_hat: np.ndarray,
    arm: float,
    *,
    guard: tuple[str, ...] = (),
    sign: float = -1.0,
) -> float:
    psi, curve = _plain(g_hat, q_hat)[arm]
    d_g, d_q = _extra(g_hat, q_hat, arm)
    if "Q" in guard:
        curve = curve + sign * d_g
    if "g" in guard:
        curve = curve + sign * d_q
    truth = float(np.sum(law.P_W * law.Q[:, int(arm)]))
    return psi - truth + float(np.mean(curve))


@pytest.mark.parametrize("arm", ARMS)
class TestTheMultiArmProductRemainder:
    def test_matches_the_closed_form(self, arm: float) -> None:
        actual = _expansion(WRONG_G, WRONG_Q, arm)
        assert actual == pytest.approx(_product(WRONG_G, WRONG_Q, arm), abs=1e-12)
        assert abs(actual) > 1e-3

    def test_either_correct_nuisance_kills_it(self, arm: float) -> None:
        assert _expansion(law.G, WRONG_Q, arm) == pytest.approx(0.0, abs=1e-12)
        assert _expansion(WRONG_G, law.Q, arm) == pytest.approx(0.0, abs=1e-12)

    def test_each_saturated_drtmle_guard_removes_it(self, arm: float) -> None:
        assert _expansion(WRONG_G, WRONG_Q, arm, guard=("Q",)) == pytest.approx(0.0, abs=1e-12)
        assert _expansion(WRONG_G, WRONG_Q, arm, guard=("g",)) == pytest.approx(0.0, abs=1e-12)

    def test_two_guards_leave_minus_the_plain_remainder(self, arm: float) -> None:
        plain = _expansion(WRONG_G, WRONG_Q, arm)
        both = _expansion(WRONG_G, WRONG_Q, arm, guard=("Q", "g"))
        assert both == pytest.approx(-plain, abs=1e-12)

    def test_the_corrections_are_subtracted_not_added(self, arm: float) -> None:
        minus = _expansion(WRONG_G, WRONG_Q, arm, guard=("Q",), sign=-1.0)
        plus = _expansion(WRONG_G, WRONG_Q, arm, guard=("Q",), sign=1.0)
        assert abs(minus - plus) > 1e-3


def _oat_projection(q_hat: np.ndarray) -> np.ndarray:
    """Population ``P(A | Qbar-vector)`` for a saturated OAT learner."""
    _, inverse = np.unique(np.round(q_hat, 12), axis=0, return_inverse=True)
    out = np.empty_like(law.G)
    for group in np.unique(inverse):
        rows = inverse == group
        weights = law.P_W[rows]
        out[rows] = np.average(law.G[rows], axis=0, weights=weights)
    return out


def test_oat_generated_regressor_can_leave_a_first_order_remainder() -> None:
    """A coarsened Qbar design makes OAT's mechanism coarser than ``g_0``.

    This is the concrete boundary behind OAT's documented loss of the mechanism-only
    robustness leg: its g is generated from Qbar and cannot be repaired independently.
    """
    coarse = WRONG_Q.copy()
    coarse[1] = coarse[0]
    mechanism = _oat_projection(coarse)
    assert not np.allclose(mechanism, law.G)
    remainders = np.array([_expansion(mechanism, coarse, arm) for arm in ARMS])
    assert np.max(np.abs(remainders)) > 1e-3
    for arm, actual in zip(ARMS, remainders, strict=True):
        assert actual == pytest.approx(_product(mechanism, coarse, arm), abs=1e-12)


def test_halving_both_multi_arm_errors_quarters_the_remainder() -> None:
    def at(scale: float) -> float:
        g_hat = law.G + scale * (WRONG_G - law.G)
        q_hat = law.Q + scale * (WRONG_Q - law.Q)
        return max(abs(_expansion(g_hat, q_hat, arm)) for arm in ARMS)

    assert at(0.005) / at(0.0025) == pytest.approx(4.0, abs=0.05)
