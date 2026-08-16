r"""Finite-support remainder identities for multi-arm TMLE, OAT and DR-TMLE.

The exact multi-arm fit makes every collaborative correction vanish.  This module instead
evaluates the estimating equations at nuisance functions that are wrong on purpose, where
the armwise product remainder and both DR-TMLE projections are observable.

The corrections are taken from the shipped
:func:`~cleverly.inference.influence.reduced_correction_parts` rather than rebuilt here, so
a sign, an indicator or a guard mapping that is wrong in the library moves these numbers.
The longhand derivation is kept as an independent oracle and compared against it in
:func:`test_the_library_corrections_are_the_longhand_ones`.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly.estimators.reduced import ReducedSet
from cleverly.fluctuation.iterative import InitialFit
from cleverly.fluctuation.submodel import submodel_for
from cleverly.inference.influence import counterfactual_means, reduced_correction_parts
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
#: Wide enough that neither the mechanism nor ``gr1`` is clipped: every quantity below
#: divides by the untruncated array the longhand derivation divides by.
INERT_BOUNDS = (1e-12, 1.0 - 1e-12)


def _rows() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = law.frame(labelled=False)
    return (
        frame["W"].to_numpy().astype(int),
        frame["A"].to_numpy(dtype=float),
        frame["Y"].to_numpy(dtype=float),
    )


def _initial(q_hat: np.ndarray) -> InitialFit:
    covariate, treatment, _ = _rows()
    return InitialFit(
        observed=q_hat[covariate, treatment.astype(int)],
        arms={arm: q_hat[covariate, int(arm)] for arm in ARMS},
    )


def _plain(g_hat: np.ndarray, q_hat: np.ndarray) -> dict[float, tuple[float, np.ndarray]]:
    covariate, treatment, outcome = _rows()
    submodel = submodel_for("mean", treatment, g_hat[covariate], arms=ARMS, reference=0.0)
    means = counterfactual_means(outcome, _initial(q_hat), submodel, np.ones(law.N))
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


def _reduced(g_hat: np.ndarray, q_hat: np.ndarray) -> ReducedSet:
    """The three reduced regressions this finite law admits *exactly*, row by row.

    ``Qbar`` and ``g`` are functions of ``W`` alone here, so each reduction is its own
    conditional expectation and no fit stands between the law and these arrays.  That is
    what lets the library's correction code be evaluated at a known answer.
    """
    covariate, _, _ = _rows()
    return ReducedSet(
        qr=(law.Q - q_hat)[covariate],
        gr1=law.G[covariate],
        gr2=((law.G - g_hat) / g_hat)[covariate],
        arms=ARMS,
        g_bounds=INERT_BOUNDS,
    )


def _corrections(
    g_hat: np.ndarray, q_hat: np.ndarray, guard: tuple[str, ...]
) -> dict[float, np.ndarray]:
    """What the shipped DR-TMLE code subtracts under ``guard``, one array per arm.

    Three arms, so this is :func:`~cleverly.inference.influence.reduced_correction_parts`'
    multi-arm branch and not the binary one the existing remainder module reaches.
    """
    covariate, treatment, outcome = _rows()
    parts = reduced_correction_parts(
        outcome,
        _initial(q_hat),
        treatment,
        _reduced(g_hat, q_hat),
        g_hat[covariate],
        bounds=INERT_BOUNDS,
        guard=guard,
    )
    return dict(parts.total())


def _expansion(
    g_hat: np.ndarray,
    q_hat: np.ndarray,
    arm: float,
    *,
    guard: tuple[str, ...] = (),
    sign: float = -1.0,
) -> float:
    psi, curve = _plain(g_hat, q_hat)[arm]
    if guard:
        curve = curve + sign * _corrections(g_hat, q_hat, guard)[arm]
    truth = float(np.sum(law.P_W * law.Q[:, int(arm)]))
    return psi - truth + float(np.mean(curve))


@pytest.mark.parametrize("arm", ARMS)
def test_the_library_corrections_are_the_longhand_ones(arm: float) -> None:
    """What pins the reduction code itself, rather than only the algebra around it.

    Both terms are nonzero at this pair of wrong nuisances, so a flipped sign, a dropped
    indicator, a mechanism read at the wrong arm or a swapped guard mapping in
    :func:`~cleverly.inference.influence.reduced_correction_parts` moves one of these
    arrays away from the longhand derivation and fails here.  The remainder identities
    below all run through the same code path, so they inherit the same reach.
    """
    d_g, d_q = _extra(WRONG_G, WRONG_Q, arm)
    assert min(np.max(np.abs(d_g)), np.max(np.abs(d_q))) > 1e-3
    # `"Q"` is the guard that adds equation (9) and so contributes `d_g`; `"g"` adds (10)
    # and contributes `d_q`.  Asserting the mapping, not just the two arrays.
    np.testing.assert_allclose(_corrections(WRONG_G, WRONG_Q, ("Q",))[arm], d_g, atol=1e-14)
    np.testing.assert_allclose(_corrections(WRONG_G, WRONG_Q, ("g",))[arm], d_q, atol=1e-14)
    np.testing.assert_allclose(
        _corrections(WRONG_G, WRONG_Q, ("Q", "g"))[arm], d_g + d_q, atol=1e-14
    )


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
