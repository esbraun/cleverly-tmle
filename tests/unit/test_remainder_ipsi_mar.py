r"""What has to be right for an incremental fit when outcomes are missing?

:mod:`tests.unit.test_remainder_ipsi` shows the incremental remainder is one-sided but
*complete*: every term carries :math:`(\hat g - g_0)`, so a consistent mechanism kills it
whatever :math:`\bar Q` does, and no accuracy in :math:`\bar Q` rescues an inconsistent
:math:`\hat g`.  :mod:`tests.unit.test_remainder_mar` shows that a third nuisance in the
clever covariate replaces "either one right" with "``Qbar`` right, or the *product*
``g * pi`` right".  This module is what happens when both hold at once, and the answer is
not the union of the two guarantees -- it is stricter than either.

Expanding the plug-in at :math:`(\hat g, \hat\pi, \bar Q)` and writing
:math:`\rho(a, W) = g_0(a \mid W)\pi_0(a, W) / \hat g(a \mid W)\hat\pi(a, W)` for the error
in the combined denominator,

.. math::

    R_2 = \underbrace{(\delta - 1)\,\delta\,
            E_W\!\left[\frac{(g_0 - \hat g)^2\,
                             (\bar Q_{0,1} - \bar Q_{0,0})}{D_0 \hat D^2}\right]
          }_{\text{pure }(\hat g - g_0)^2,\ \pi\text{-free}}
        + \underbrace{E_W\!\left[\sum_a \hat q_\delta(a \mid W)\,(\rho(a, W) - 1)\,
              \bigl(\bar Q_0(a, W) - \bar Q(a, W)\bigr)\right]
          }_{\text{product-mechanism error} \times \bar Q\text{ error}}
        + \underbrace{\delta\,E_W\!\left[\frac{(g_0 - \hat g)\,
              \bigl((\hat Q_1 - \hat Q_0) - (\bar Q_{0,1} - \bar Q_{0,0})\bigr)}{\hat D^2}
              \right]}_{\text{propensity error} \times \bar Q\text{ error}}

At :math:`\hat\pi = \pi_0` this collapses term for term onto the closed form the
no-missingness module asserts, which is the sense in which the derivation really is the
same one with an extra factor.  What it is *not* is the same guarantee:

    :math:`R_2 = 0` iff :math:`\hat g` is right **and** one of :math:`\hat\pi`,
    :math:`\bar Q` is.

Two consequences reverse a sibling module each, and both are asserted below rather than
left to a reader to notice.

* A consistent mechanism no longer suffices.  With :math:`\hat g = g_0` the squared term
  and the third term both vanish, but the middle one survives on
  :math:`(\pi_0/\hat\pi - 1)` -- so :math:`\hat g` right is necessary and not sufficient,
  where without missingness it was sufficient outright.
* The two mechanisms cannot trade off.  ``test_remainder_mar`` finds nuisances that are
  wrong everywhere whose products :math:`\hat g\hat\pi` are right, and the remainder is
  exactly zero.  Here the same construction leaves the squared term standing, because
  :math:`\hat g` is inside the estimand and nothing done to :math:`\hat\pi` can reach it.

Everything is an exact finite sum on :mod:`tests.discrete_law_mar`, checked at ``1e-12``
rather than inferred from a simulation.  :math:`\Psi` and the closed form are longhand
while :math:`D^*` is the library's, and the targeting step is never run -- so nothing here
can be satisfied by a fluctuation that merely converged.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly.fluctuation.iterative import InitialFit
from cleverly.fluctuation.submodel import submodel_for
from cleverly.inference.influence import ipsi_means
from cleverly.interventions import IPSISet
from tests import discrete_law_mar as law

#: The same deliberately wrong nuisances the sibling modules use, so the three disagree
#: about nothing except which functional they expand.
WRONG_G = np.array([0.55, 0.35, 0.45])
WRONG_PI = law.PI + np.array([[0.30, -0.15], [-0.20, 0.25], [-0.35, 0.10]])
WRONG_Q = law.Q + np.array([[0.10, -0.15], [-0.20, 0.10], [0.05, 0.20]])

NAMES = tuple(law.IPSI_DELTAS)
DELTAS = tuple(law.IPSI_DELTAS.values())

#: The tilts that actually depend on ``g``.  ``delta = 1`` no longer has a zero remainder
#: for free -- see :class:`TestTheNaturalCourseIsNoLongerFree` -- but the squared term
#: still carries ``(delta - 1)``, so it is the cases away from one that exercise it.
TILTED = tuple(name for name, delta in law.IPSI_DELTAS.items() if delta != 1.0)


def _cancelling(scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Nuisances wrong everywhere whose products ``g * pi`` are both right.

    Lifted from :mod:`tests.unit.test_remainder_mar`, where it drives the remainder to
    exactly zero.  Here it must not, and that difference is the point.
    """
    g_hat = scale * law.G
    pi_hat = np.column_stack(
        [
            (1.0 - law.G) * law.PI[:, 0] / (1.0 - g_hat),
            law.G * law.PI[:, 1] / g_hat,
        ]
    )
    return g_hat, pi_hat


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


def _expansion(g_hat: np.ndarray, pi_hat: np.ndarray, q_hat: np.ndarray) -> dict[str, float]:
    """``R_2`` per tilt, with the plug-in and ``P_0 D*`` both taken from the library."""
    frame = law.frame()
    covariate = frame["W"].to_numpy().astype(int)
    treatment = frame["A"].to_numpy(dtype=float)
    observed = frame["Delta"].to_numpy(dtype=float) == 1.0
    # The library zero-fills the outcome at unobserved rows; the Delta factor is what keeps
    # those entries out of the arithmetic, so mirror it rather than passing NaN.
    outcome = np.nan_to_num(frame["Y"].to_numpy(dtype=float))

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
        missingness=np.column_stack([pi_hat[covariate, 0], pi_hat[covariate, 1]]),
    )
    means = ipsi_means(outcome, initial, submodel, tilts, treatment, np.ones(law.N), observed)
    # The sample realises the law exactly, so the sample mean of the influence curve *is*
    # P_0 D*.
    return {
        name: means[float(index)].psi
        - law.TRUTH[f"ey_ipsi[{name}]"]
        + float(np.mean(means[float(index)].influence_curve))
        for index, name in enumerate(NAMES)
    }


def _closed_form(g_hat: np.ndarray, pi_hat: np.ndarray, q_hat: np.ndarray) -> dict[str, float]:
    """The three terms of the module docstring, written longhand."""
    g0, pi0, q0 = law.G, law.PI, law.Q
    # (3, 2) of g(a | w), arms in column order, for both the truth and the guess.
    g0_arm = np.column_stack([1.0 - g0, g0])
    g_hat_arm = np.column_stack([1.0 - g_hat, g_hat])
    rho = g0_arm * pi0 / (g_hat_arm * pi_hat)

    out: dict[str, float] = {}
    for name, delta in law.IPSI_DELTAS.items():
        d0 = delta * g0 + (1.0 - g0)
        dh = delta * g_hat + (1.0 - g_hat)
        # q_delta(a | w) at the *guessed* mechanism, arms in column order.
        q_delta = np.column_stack([(1.0 - g_hat) / dh, delta * g_hat / dh])

        squared = (
            (delta - 1.0)
            * delta
            * np.sum(law.P_W * (g0 - g_hat) ** 2 * (q0[:, 1] - q0[:, 0]) / (d0 * dh**2))
        )
        product = np.sum(law.P_W[:, None] * q_delta * (rho - 1.0) * (q0 - q_hat))
        blip = (q_hat[:, 1] - q_hat[:, 0]) - (q0[:, 1] - q0[:, 0])
        propensity = delta * np.sum(law.P_W * (g0 - g_hat) * blip / dh**2)
        out[name] = float(squared + product + propensity)
    return out


class TestTheRemainderIsSecondOrder:
    @pytest.mark.parametrize("name", NAMES)
    def test_matches_the_closed_form(self, name: str) -> None:
        actual = _expansion(WRONG_G, WRONG_PI, WRONG_Q)[name]
        assert actual == pytest.approx(_closed_form(WRONG_G, WRONG_PI, WRONG_Q)[name], abs=1e-12)

    @pytest.mark.parametrize("name", NAMES)
    def test_the_misspecification_is_not_too_mild_to_test_anything(self, name: str) -> None:
        # Every tilt, unlike the no-missingness module: with pi wrong the natural course
        # has a remainder too, which is the whole of TestTheNaturalCourseIsNoLongerFree.
        assert abs(_expansion(WRONG_G, WRONG_PI, WRONG_Q)[name]) > 1e-3

    @pytest.mark.parametrize("name", NAMES)
    def test_vanishes_when_everything_is_right(self, name: str) -> None:
        assert _expansion(law.G, law.PI, law.Q)[name] == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("name", NAMES)
    def test_the_pi_free_term_really_is_pi_free(self, name: str) -> None:
        """With ``Qbar`` right, nothing the missingness model does can move ``R_2``.

        The middle and third terms both carry a ``Qbar`` error, so what is left is the
        squared mechanism term -- and it contains no ``pi`` at all.  Two guesses at the
        missingness model that differ everywhere must give the identical remainder, which
        is a sharper statement than either one matching the closed form.
        """
        right = _expansion(WRONG_G, law.PI, law.Q)[name]
        wrong = _expansion(WRONG_G, WRONG_PI, law.Q)[name]
        assert right == pytest.approx(wrong, abs=1e-12)


class TestTheGuaranteeIsStricterThanEitherSibling:
    """``R_2 = 0`` iff ``g`` is right and one of ``pi``, ``Qbar`` is."""

    @pytest.mark.parametrize("name", NAMES)
    def test_a_consistent_mechanism_and_missingness_model_kill_it(self, name: str) -> None:
        assert _expansion(law.G, law.PI, WRONG_Q)[name] == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("name", NAMES)
    def test_a_consistent_mechanism_and_outcome_regression_kill_it(self, name: str) -> None:
        assert _expansion(law.G, WRONG_PI, law.Q)[name] == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("name", NAMES)
    def test_a_consistent_mechanism_alone_does_not(self, name: str) -> None:
        """The reversal.  Without missingness this is exactly the case that vanishes."""
        assert abs(_expansion(law.G, WRONG_PI, WRONG_Q)[name]) > 1e-3

    @pytest.mark.parametrize("name", TILTED)
    def test_a_consistent_outcome_regression_alone_does_not(self, name: str) -> None:
        assert abs(_expansion(WRONG_G, law.PI, law.Q)[name]) > 1e-3

    @pytest.mark.parametrize("name", TILTED)
    def test_what_survives_a_right_qbar_is_exactly_the_squared_mechanism_error(
        self, name: str
    ) -> None:
        delta = law.IPSI_DELTAS[name]
        d0 = delta * law.G + (1.0 - law.G)
        dh = delta * WRONG_G + (1.0 - WRONG_G)
        expected = (
            (delta - 1.0)
            * delta
            * np.sum(law.P_W * (law.G - WRONG_G) ** 2 * (law.Q[:, 1] - law.Q[:, 0]) / (d0 * dh**2))
        )
        assert _expansion(WRONG_G, WRONG_PI, law.Q)[name] == pytest.approx(expected, abs=1e-12)


class TestTheMechanismsCannotTradeOff:
    """The other reversal: ``test_remainder_mar``'s cancellation does not carry over.

    There, inflating ``ghat`` and deflating ``pihat`` by the same factor leaves both clever
    covariates unchanged and the remainder exactly zero -- the classical statement that only
    the product has to be right.  Here ``ghat`` is inside ``q_delta``, so the same nuisances
    leave the squared term standing and the remainder is not zero for any scale but one.
    """

    SCALES = (0.85, 1.15)

    @pytest.mark.parametrize("scale", SCALES)
    def test_the_construction_really_does_leave_the_product_right(self, scale: float) -> None:
        g_hat, pi_hat = _cancelling(scale)
        g0_arm = np.column_stack([1.0 - law.G, law.G])
        g_hat_arm = np.column_stack([1.0 - g_hat, g_hat])
        np.testing.assert_allclose(g_hat_arm * pi_hat, g0_arm * law.PI, atol=1e-12, rtol=0)
        assert np.max(np.abs(g_hat - law.G)) > 1e-2, "and neither factor is right on its own"

    @pytest.mark.parametrize("scale", SCALES)
    @pytest.mark.parametrize("name", TILTED)
    def test_but_the_remainder_does_not_vanish(self, scale: float, name: str) -> None:
        """Not "large" -- *exactly the squared term*, which is the sharper statement.

        A magnitude bar would be the wrong assertion here.  The surviving term is second
        order in a 15% mechanism error, so it is only 8e-4 to 9e-4 on this law; what makes
        the point is that ``test_remainder_mar`` gets exactly zero from these same
        nuisances while this gets a number the closed form predicts to the last bit, nine
        orders above the tolerance that "vanishes" is asserted at everywhere else here.
        """
        g_hat, pi_hat = _cancelling(scale)
        actual = _expansion(g_hat, pi_hat, law.Q)[name]

        delta = law.IPSI_DELTAS[name]
        d0 = delta * law.G + (1.0 - law.G)
        dh = delta * g_hat + (1.0 - g_hat)
        squared = float(
            (delta - 1.0)
            * delta
            * np.sum(law.P_W * (law.G - g_hat) ** 2 * (law.Q[:, 1] - law.Q[:, 0]) / (d0 * dh**2))
        )
        assert actual == pytest.approx(squared, abs=1e-12)
        assert abs(actual) / 1e-12 > 1e8, "and it is not zero in any sense this suite uses"


class TestTheNaturalCourseIsNoLongerFree:
    r"""At :math:`\delta = 1` the remainder was identically zero.  With ``delta=`` it is not.

    Without missingness both terms carry :math:`(\delta - 1)`, so ``psi(1)`` is ``mean(Y)``
    for *any* nuisances at all and the remainder cannot be anything but zero.  Under MAR
    ``psi(1)`` is the missing-data mean, which is a genuine estimation problem: the middle
    term of the expansion has no :math:`(\delta - 1)` factor, so a wrong missingness model
    leaves a first-order error behind.  The tilt has stopped being the thing that costs
    something.
    """

    NAME = "natural course"

    def test_the_squared_term_still_vanishes(self) -> None:
        # (delta - 1) = 0 kills it, exactly as it does without missingness -- so whatever
        # survives below is the middle term and not a leak from this one.
        assert _closed_form(WRONG_G, law.PI, law.Q)[self.NAME] == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize(
        "nuisances",
        [
            (law.G, law.PI, WRONG_Q),
            (WRONG_G, law.PI, WRONG_Q),
            (WRONG_G, WRONG_PI, law.Q),
        ],
        ids=["q wrong", "g and q wrong", "g and pi wrong"],
    )
    def test_it_still_vanishes_where_the_missingness_model_is_right_or_qbar_is(
        self, nuisances: tuple
    ) -> None:
        assert _expansion(*nuisances)[self.NAME] == pytest.approx(0.0, abs=1e-12)

    def test_but_a_wrong_missingness_model_leaves_a_first_order_error(self) -> None:
        assert abs(_expansion(law.G, WRONG_PI, WRONG_Q)[self.NAME]) > 1e-2
