r"""Is the *incremental* estimating equation's remainder second-order -- and in which errors?

The counterpart of :mod:`tests.unit.test_remainder` for
:math:`\Psi(\delta) = E_W[m(W)]`, :math:`m = (\delta g \bar Q_1 + (1-g)\bar Q_0)/D_\delta`.
Expanding the plug-in at a pair of nuisance guesses,

.. math::

    \Psi(\bar Q, \hat g) - \Psi(P_0) + P_0 D^*(\hat g, \bar Q) = R_2 ,

and this module exists because :math:`R_2` here is **not** the shape every other module in
this family checks.  Using the exact Möbius identity
:math:`m(g_0, \bar Q) - m(\hat g, \bar Q)
= \delta(\bar Q_1 - \bar Q_0)(g_0 - \hat g)/(D_0 \hat D)`,

.. math::

    R_2 = \underbrace{(\delta - 1)\,\delta\,
            E_W\!\left[\frac{(g_0 - \hat g)^2 (\bar Q_{0,1} - \bar Q_{0,0})}
                            {D_0 \hat D^2}\right]}_{\text{pure } (\hat g - g_0)^2}
        + \underbrace{(\delta - 1)\,
            E_W\!\left[\frac{g_0 - \hat g}{\hat D}
              \bigl\{\hat q (\bar Q_{0,1} - \bar Q_1)
                   + (1 - \hat q)(\bar Q_{0,0} - \bar Q_0)\bigr\}\right]}_{\text{cross term}}

Both terms carry :math:`(g_0 - \hat g)` as a factor, and only the second carries a
:math:`\bar Q` error.  So the guarantee is one-sided:

* a consistent mechanism kills the remainder **whatever** :math:`\bar Q` does;
* a consistent :math:`\bar Q` does **not**, and no accuracy in it can -- the squared term
  survives.

That is what "not doubly robust" means here, stated as an equality rather than as an
absence, and it is the reverse of what every sibling module asserts.  At
:math:`\delta = 1` both terms carry the factor :math:`(\delta - 1) = 0`, so the remainder
is identically zero for any nuisances at all -- the same identity that makes
:math:`\Psi(1) = E[Y]`.

Nothing here runs the targeting step.  The remainder is a property of the estimating
equation, evaluated at nuisances that are wrong on purpose.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from cleverly.fluctuation.iterative import InitialFit
from cleverly.fluctuation.submodel import submodel_for
from cleverly.inference.influence import ipsi_means
from cleverly.interventions import IPSISet
from tests import discrete_law as law

#: The same deliberately wrong nuisances the sibling modules use, so they disagree about
#: nothing except which functional they expand.
WRONG_G = np.array([0.55, 0.35, 0.45])
WRONG_Q = law.Q + np.array([[0.10, -0.15], [-0.20, 0.10], [0.05, 0.20]])

NAMES = tuple(law.IPSI_DELTAS)
DELTAS = tuple(law.IPSI_DELTAS.values())

#: The tilts that actually depend on ``g``.  ``delta = 1`` has a zero remainder by
#: construction and is checked on its own, not swept up with the others.
TILTED = tuple(name for name, delta in law.IPSI_DELTAS.items() if delta != 1.0)


def _tilt_set(covariate: np.ndarray, g_hat: np.ndarray) -> IPSISet:
    """An :class:`IPSISet` at an arbitrary mechanism, without going through a fit."""
    one = g_hat[covariate]
    values, weights, derivative = [], [], []
    for delta in DELTAS:
        d = delta * one + (1.0 - one)
        values.append(np.column_stack([(1.0 - one) / d, delta * one / d]))
        weights.append(np.column_stack([1.0 / d, np.full_like(d, delta) / d]))
        derivative.append(delta / d**2)
    return IPSISet(
        NAMES,
        DELTAS,
        np.stack(values, axis=2),
        np.stack(weights, axis=2),
        np.column_stack(derivative),
        one,
    )


def _expansion(g_hat: np.ndarray, q_hat: np.ndarray) -> dict[str, float]:
    """``R_2`` per tilt, with the plug-in and ``P_0 D*`` both taken from the library."""
    frame = law.frame()
    covariate = frame["W"].to_numpy().astype(int)
    treatment = frame["A"].to_numpy(dtype=float)
    outcome = frame["Y"].to_numpy(dtype=float)

    at_one, at_zero = q_hat[covariate, 1], q_hat[covariate, 0]
    initial = InitialFit(
        observed=np.where(treatment == 1.0, at_one, at_zero),
        arms={1.0: at_one, 0.0: at_zero},
    )
    tilts = _tilt_set(covariate, g_hat)
    submodel = submodel_for(
        "ipsi",
        treatment,
        np.column_stack([1.0 - tilts.propensity, tilts.propensity]),
        incremental=tilts.weights,
    )
    means = ipsi_means(outcome, initial, submodel, tilts, treatment, np.ones(law.N))
    # The sample realises the law exactly, so the sample mean of the influence curve *is*
    # P_0 D*.
    return {
        name: means[float(index)].psi
        - law.TRUTH[f"ey_ipsi[{name}]"]
        + float(np.mean(means[float(index)].influence_curve))
        for index, name in enumerate(NAMES)
    }


def _closed_form(g_hat: np.ndarray, q_hat: np.ndarray) -> dict[str, float]:
    """The two terms of the module docstring, written longhand."""
    g0, q0 = law.G, law.Q
    out: dict[str, float] = {}
    for name, delta in law.IPSI_DELTAS.items():
        d0 = delta * g0 + (1.0 - g0)
        dh = delta * g_hat + (1.0 - g_hat)
        squared = (
            (delta - 1.0)
            * delta
            * np.sum(law.P_W * (g0 - g_hat) ** 2 * (q0[:, 1] - q0[:, 0]) / (d0 * dh**2))
        )
        q_tilt = delta * g_hat / dh
        cross = (delta - 1.0) * np.sum(
            law.P_W
            * (g0 - g_hat)
            / dh
            * (q_tilt * (q0[:, 1] - q_hat[:, 1]) + (1.0 - q_tilt) * (q0[:, 0] - q_hat[:, 0]))
        )
        out[name] = float(squared + cross)
    return out


class TestTheRemainderIsSecondOrder:
    @pytest.mark.parametrize("name", NAMES)
    def test_matches_the_closed_form(self, name: str) -> None:
        actual = _expansion(WRONG_G, WRONG_Q)[name]
        assert actual == pytest.approx(_closed_form(WRONG_G, WRONG_Q)[name], abs=1e-12)

    @pytest.mark.parametrize("name", TILTED)
    def test_the_misspecification_is_not_too_mild_to_test_anything(self, name: str) -> None:
        assert abs(_expansion(WRONG_G, WRONG_Q)[name]) > 1e-3

    @pytest.mark.parametrize("name", NAMES)
    def test_vanishes_when_both_are_right(self, name: str) -> None:
        assert _expansion(law.G, law.Q)[name] == pytest.approx(0.0, abs=1e-12)


class TestTheGuaranteeIsOneSided:
    """The whole content of "not doubly robust", asserted in both directions."""

    @pytest.mark.parametrize("name", NAMES)
    def test_a_consistent_mechanism_kills_it_whatever_qbar_does(self, name: str) -> None:
        assert _expansion(law.G, WRONG_Q)[name] == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("name", TILTED)
    def test_a_consistent_qbar_does_not(self, name: str) -> None:
        """The reverse of every sibling module, and the reason for the ``dr_condition``."""
        remainder = _expansion(WRONG_G, law.Q)[name]
        assert abs(remainder) > 1e-3, "an inconsistent mechanism is not rescued by a right Qbar"

    @pytest.mark.parametrize("name", TILTED)
    def test_what_survives_is_exactly_the_squared_mechanism_error(self, name: str) -> None:
        delta = law.IPSI_DELTAS[name]
        d0 = delta * law.G + (1.0 - law.G)
        dh = delta * WRONG_G + (1.0 - WRONG_G)
        expected = (
            (delta - 1.0)
            * delta
            * float(
                np.sum(
                    law.P_W * (law.G - WRONG_G) ** 2 * (law.Q[:, 1] - law.Q[:, 0]) / (d0 * dh**2)
                )
            )
        )
        assert _expansion(WRONG_G, law.Q)[name] == pytest.approx(expected, abs=1e-12)

    @pytest.mark.parametrize("name", TILTED)
    def test_it_is_second_order_in_the_mechanism_error(self, name: str) -> None:
        """Halving the error quarters the remainder -- which is what root-n rests on.

        The ratio is checked *in the limit* rather than at one step, because ``Dhat``
        moves with ``ghat`` too, so the exact quadratic only shows through as the error
        shrinks.  Measured here: 3.79 -> 3.99 for ``delta=2`` and 4.23 -> 4.01 for
        ``delta=0.5`` as the error goes 0.08 -> 0.0025, approaching 4 from either side.
        Asserting 4 at a single coarse step would be asserting an accident.
        """
        errors = [0.08, 0.04, 0.02, 0.01, 0.005, 0.0025]
        remainders = [abs(_expansion(law.G + eps, law.Q)[name]) for eps in errors]
        ratios = [coarse / fine for coarse, fine in pairwise(remainders)]
        assert ratios[-1] == pytest.approx(4.0, rel=0.02)
        # ... and it is getting there, rather than sitting near 4 by coincidence.
        assert abs(ratios[-1] - 4.0) < abs(ratios[0] - 4.0)


class TestTheNaturalCourseHasNoRemainderAtAll:
    """``delta = 1`` carries the factor ``(delta - 1)``, so nothing survives."""

    @pytest.mark.parametrize(
        ("g_hat", "q_hat"),
        [(WRONG_G, WRONG_Q), (law.G, WRONG_Q), (WRONG_G, law.Q), (law.G, law.Q)],
    )
    def test_it_vanishes_for_any_nuisances(self, g_hat: np.ndarray, q_hat: np.ndarray) -> None:
        assert _expansion(g_hat, q_hat)["natural course"] == pytest.approx(0.0, abs=1e-12)
