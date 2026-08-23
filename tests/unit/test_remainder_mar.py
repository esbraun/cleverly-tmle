r"""What does double robustness *mean* when outcomes are missing?

:mod:`tests.unit.test_remainder` shows that the remainder of the von Mises expansion is a
product of two nuisance errors, so the estimator is consistent when either the propensity
or the outcome regression is right.  With a third nuisance in the clever covariate the
obvious generalisation -- "consistent if any one of the three is right" -- is *false*, and
the point of this module is to say what replaces it.

Working the expansion through for :math:`EY_1` gives

.. math::

    R_2 = \int \left(\frac{g_0(W)\,\pi_0(1, W)}{\hat g(W)\,\hat\pi(1, W)} - 1\right)
                \bigl(\bar Q_0(1, W) - \bar Q(1, W)\bigr)\, dP_0 ,

still a product of two factors, but the left one is the error in the **product**
:math:`\hat g\,\hat\pi`, not in either factor separately.  So the guarantee is:

    consistent if :math:`\bar Q` is right, **or** if the product :math:`g\,\pi` is right.

Two consequences the tests below pin down, because neither is what a reader would guess.
Getting the propensity exactly right buys nothing on its own -- with a wrong missingness
model the remainder stays first-order, and the classical "IPW half" of double robustness
is simply not available.  And errors in the two mechanisms can *cancel*: inflate
:math:`\hat g` by a factor and deflate :math:`\hat\pi` by the same one, and the remainder
is exactly zero although both nuisances are wrong everywhere.

The same machinery settles what nuisance truncation does, which is otherwise easy to
state loosely.  Bounding :math:`\pi` away from zero cannot move the target: the plug-in
:math:`\int \bar Q(1, w)\,dP_n(w)` does not contain :math:`\pi` at all, and
:math:`\Psi(P_0)` is a functional of the law.  What it moves is :math:`R_2`, by exactly
the closed form above evaluated at the truncated value.  Truncation buys variance and
pays in second-order bias; it does not quietly redefine the estimand.

Everything is an exact finite sum on :mod:`tests.discrete_law_mar`, so this is checked
deterministically at ``1e-12`` rather than inferred from a simulation.  As in the parent
module, :math:`\Psi` and the closed form for :math:`R_2` are written out longhand while
:math:`D^*` is the library's -- which is the claim under test -- and the targeting step is
never run, so nothing here can be satisfied by a fluctuation that merely converged.

Unlike the parent module, this one covers ``att`` and ``atc`` as well.  Their remainders
have a different shape -- the estimand conditions on a random event, so the influence
curve carries a centring term -- and no test in the suite has ever checked them, with or
without missingness.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly.fluctuation.iterative import InitialFit
from cleverly.fluctuation.submodel import submodel_for
from cleverly.inference.influence import atc_estimate, att_estimate
from tests import discrete_law_mar as law
from tests.conftest import binary_means

#: A propensity that is wrong at every covariate value, and stays inside ``(0, 1)``.
WRONG_G = np.array([0.55, 0.35, 0.45])

#: A missingness mechanism wrong in both arms, with mixed signs so that errors cannot
#: cancel into a spurious pass, and comfortably inside ``(0, 1]``.
WRONG_PI = law.PI + np.array([[0.30, -0.15], [-0.20, 0.25], [-0.35, 0.10]])

#: An outcome regression that is wrong in both arms.
WRONG_Q = law.Q + np.array([[0.10, -0.15], [-0.20, 0.10], [0.05, 0.20]])


def _cancelling(scale: float) -> tuple[np.ndarray, np.ndarray]:
    r"""Nuisances wrong everywhere whose products :math:`\hat g\hat\pi` are both right.

    Inflate the propensity by ``scale`` and deflate the treated arm's observation
    probability by the same factor; the control arm's is then whatever makes
    :math:`(1 - \hat g)\hat\pi_0` come out at :math:`(1 - g_0)\pi_0`.  Both clever
    covariates are unchanged even though neither nuisance is.
    """
    g_hat = scale * law.G
    pi_hat = np.column_stack(
        [
            (1.0 - law.G) * law.PI[:, 0] / (1.0 - g_hat),
            law.G * law.PI[:, 1] / g_hat,
        ]
    )
    return g_hat, pi_hat


def _fit(g_hat: np.ndarray, pi_hat: np.ndarray, q_hat: np.ndarray, group: str):
    """The library's plug-in and influence curve at nuisances it did not fit.

    No targeting step: ``InitialFit`` holds ``Qbar`` as handed to it, and the clever
    covariate is built from ``g_hat`` and ``pi_hat``.  Because the sample realises the law
    exactly, the sample mean of the influence curve *is* :math:`P_0 D^*`.
    """
    frame = law.frame()
    covariate = frame["W"].to_numpy().astype(int)
    treatment = frame["A"].to_numpy(dtype=float)
    observed = frame["Delta"].to_numpy(dtype=float) == 1.0
    # The library zero-fills the outcome at unobserved rows; the Delta factor is what
    # keeps those entries out of the arithmetic, so mirror it rather than passing NaN.
    outcome = np.nan_to_num(frame["Y"].to_numpy(dtype=float))

    at_one, at_zero = q_hat[covariate, 1], q_hat[covariate, 0]
    initial = InitialFit(
        observed=np.where(treatment == 1.0, at_one, at_zero),
        arms={1.0: at_one, 0.0: at_zero},
    )
    missingness = np.column_stack([pi_hat[covariate, 0], pi_hat[covariate, 1]])
    submodel = submodel_for(
        group,
        treatment,
        g_hat[covariate],
        arm_fractions=float(treatment.mean()),
        missingness=missingness,
    )
    weights = np.ones(law.N)
    if group == "mean":
        return binary_means(outcome, initial, submodel, weights, observed)
    estimate = att_estimate if group == "att" else atc_estimate
    # One entry per non-reference arm, which on this binary law is arm 1 alone.
    effect = estimate(outcome, initial, submodel, treatment, weights, observed)[1.0]
    return effect.psi, effect.influence_curve


def _expansion(g_hat: np.ndarray, pi_hat: np.ndarray, q_hat: np.ndarray) -> dict[str, float]:
    """``R_2`` for every estimand at the given nuisance guesses."""
    psi_one, ic_one, psi_zero, ic_zero = _fit(g_hat, pi_hat, q_hat, "mean")
    one = psi_one - law.TRUTH["ey1"] + float(np.mean(ic_one))
    zero = psi_zero - law.TRUTH["ey0"] + float(np.mean(ic_zero))

    out = {"ey1": one, "ey0": zero, "ate": one - zero}
    for group in ("att", "atc"):
        psi, ic = _fit(g_hat, pi_hat, q_hat, group)
        out[group] = psi - law.TRUTH[group] + float(np.mean(ic))
    return out


def _exact_remainder(g_hat: np.ndarray, pi_hat: np.ndarray, q_hat: np.ndarray) -> dict[str, float]:
    """The remainder as theory says it must be: an exact signed sum carrying both errors.

    The left factor of each product is an error in a *combined* denominator -- ``g * pi``
    for the counterfactual means, and for the conditional effects a term that mixes the
    propensity odds with the observation probability.  Each is written so that it is
    visibly zero when the mechanisms are right, which is the property being claimed.
    """
    error_one = law.Q[:, 1] - q_hat[:, 1]
    error_zero = law.Q[:, 0] - q_hat[:, 0]
    g_zero, g_hat_zero = 1.0 - law.G, 1.0 - g_hat

    # --- counterfactual means: the error in the product g * pi ------------------
    ratio_one = law.G * law.PI[:, 1] / (g_hat * pi_hat[:, 1])
    ratio_zero = g_zero * law.PI[:, 0] / (g_hat_zero * pi_hat[:, 0])
    one = float(np.sum(law.P_W * (ratio_one - 1.0) * error_one))
    zero = float(np.sum(law.P_W * (ratio_zero - 1.0) * error_zero))

    # --- ATT / ATC: the plug-in already carries the true arm shares, so the left
    # factor compares the reweighted mechanism against the arm indicator instead.
    treated_share = float(np.sum(law.P_W * law.G))
    att = float(
        np.sum(
            law.P_W
            * (
                law.G * (law.PI[:, 1] / pi_hat[:, 1] - 1.0) * error_one
                + (law.G - g_zero * law.PI[:, 0] * (g_hat / g_hat_zero) / pi_hat[:, 0]) * error_zero
            )
        )
        / treated_share
    )
    atc = float(
        np.sum(
            law.P_W
            * (
                (law.G * law.PI[:, 1] * (g_hat_zero / g_hat) / pi_hat[:, 1] - g_zero) * error_one
                + g_zero * (1.0 - law.PI[:, 0] / pi_hat[:, 0]) * error_zero
            )
        )
        / (1.0 - treated_share)
    )
    return {"ey1": one, "ey0": zero, "ate": one - zero, "att": att, "atc": atc}


ESTIMANDS = ("ey1", "ey0", "ate", "att", "atc")


class TestTheRemainderCarriesBothNuisanceErrors:
    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_matches_the_closed_form(self, name: str) -> None:
        # All three nuisances wrong, so every factor is active and nothing is zero by
        # accident.
        actual = _expansion(WRONG_G, WRONG_PI, WRONG_Q)[name]
        assert actual == pytest.approx(
            _exact_remainder(WRONG_G, WRONG_PI, WRONG_Q)[name], abs=1e-12
        )
        assert abs(actual) > 1e-3, "the misspecification is too mild to test anything"

    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_vanishes_when_the_outcome_regression_is_right(self, name: str) -> None:
        # Both mechanisms arbitrarily wrong.  This is the half the outcome model supplies,
        # and missingness does not change it: under MAR a correctly specified regression
        # of Y on (A, W) among the complete cases already identifies the estimand.
        assert _expansion(WRONG_G, WRONG_PI, law.Q)[name] == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_vanishes_when_both_mechanisms_are_right(self, name: str) -> None:
        # Qbar arbitrarily wrong.  This is the half inverse-probability weighting supplies
        # -- and it now takes *both* mechanisms, not the propensity alone.
        assert _expansion(law.G, law.PI, WRONG_Q)[name] == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_vanishes_when_everything_is_right(self, name: str) -> None:
        assert _expansion(law.G, law.PI, law.Q)[name] == pytest.approx(0.0, abs=1e-12)


class TestItIsTheProductOfTheMechanismsThatHasToBeRight:
    """Not "any one of the three nuisances" -- the two mechanisms stand or fall together."""

    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_a_right_propensity_does_not_rescue_a_wrong_missingness_model(self, name: str) -> None:
        # The generalisation a reader would guess, shown false.  With g exactly right and
        # only pi wrong the remainder is first-order and the estimate is inconsistent --
        # there is no third half of double robustness.
        remainder = _expansion(law.G, WRONG_PI, WRONG_Q)[name]
        assert abs(remainder) > 1e-3, remainder

    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_a_right_missingness_model_does_not_rescue_a_wrong_propensity(self, name: str) -> None:
        remainder = _expansion(WRONG_G, law.PI, WRONG_Q)[name]
        assert abs(remainder) > 1e-3, remainder

    @pytest.mark.parametrize("name", ("ey1", "ey0", "ate"))
    def test_errors_in_the_two_mechanisms_can_cancel_exactly(self, name: str) -> None:
        """Both nuisances wrong everywhere, both clever covariates exactly right.

        This is the sharpest statement of what the condition actually is.  Only the
        product ``g * pi`` enters the estimating equation, so a fit that overstates
        treatment assignment and understates observation by matching factors solves the
        same equation the truth does -- and the remainder is zero to machine precision
        although not one nuisance value is correct.
        """
        g_hat, pi_hat = _cancelling(1.2)
        # Not a token perturbation: the closest any of the fifteen nuisance values gets
        # to its true value is 0.038, on a probability scale.
        assert np.min(np.abs(g_hat - law.G)) > 0.03
        assert np.min(np.abs(pi_hat - law.PI)) > 0.03
        assert np.all(pi_hat > 0.0) and np.all(pi_hat <= 1.0)
        assert _expansion(g_hat, pi_hat, WRONG_Q)[name] == pytest.approx(0.0, abs=1e-12)


class TestTruncationRegularisesRatherThanRetargets:
    r"""Bounding :math:`\pi` away from zero is a bias-variance trade, not a new estimand."""

    BOUND = 0.4  # binds on the two cells where pi is 0.25 and 0.30

    def _truncated(self) -> np.ndarray:
        bounded = np.maximum(law.PI, self.BOUND)
        assert not np.allclose(bounded, law.PI), "the bound has to actually bind"
        return bounded

    def test_the_plug_in_does_not_move(self) -> None:
        # The substitution estimator is the average of the targeted predictions over the
        # sample's covariates.  No observation probability appears in it, so no bound on
        # one can shift it -- with or without truncation, the same number.
        untruncated = _fit(law.G, law.PI, WRONG_Q, "mean")
        truncated = _fit(law.G, self._truncated(), WRONG_Q, "mean")
        assert truncated[0] == pytest.approx(untruncated[0], abs=1e-15)
        assert truncated[2] == pytest.approx(untruncated[2], abs=1e-15)

    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_what_moves_is_the_second_order_remainder(self, name: str) -> None:
        # And it moves to exactly the closed form at the truncated value: the cost of
        # truncating is a remainder computed at a mechanism that is now wrong on purpose,
        # priced by the same remainder formula as any other misspecification.
        bounded = self._truncated()
        actual = _expansion(law.G, bounded, WRONG_Q)[name]
        assert actual == pytest.approx(_exact_remainder(law.G, bounded, WRONG_Q)[name], abs=1e-12)
        assert abs(actual) > 1e-3, "the bound is not binding hard enough to test anything"

    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_a_bound_that_does_not_bind_costs_nothing(self, name: str) -> None:
        # The default nuisance_bound is 0.01 and the smallest pi on this law is 0.25, so
        # the estimator runs on the unmodified mechanism -- which is the premise the
        # Gateaux module's exactness rests on.
        loose = np.maximum(law.PI, 0.01)
        assert np.allclose(loose, law.PI)
        assert _expansion(law.G, loose, WRONG_Q)[name] == pytest.approx(0.0, abs=1e-12)


class TestTheRemainderIsSecondOrder:
    """Shrinking every nuisance error by ``t`` has to shrink the remainder by ``t^2``.

    The closed form already implies it, but stating it as a rate is what connects the
    algebra to the condition the estimator needs: :math:`R_2 = o_P(n^{-1/2})` under a
    *product* rate across the nuisances, not a rate on any one of them alone.
    """

    @staticmethod
    def _at(t: float) -> float:
        g_hat = law.G + t * (WRONG_G - law.G)
        pi_hat = law.PI + t * (WRONG_PI - law.PI)
        q_hat = law.Q + t * (WRONG_Q - law.Q)
        return _expansion(g_hat, pi_hat, q_hat)["ate"]

    def test_halving_the_nuisance_error_quarters_the_remainder(self) -> None:
        # Not exactly 4: the 1/(g_hat * pi_hat) factor moves with t too, so the ratio
        # approaches 4 from one side as t shrinks.
        assert self._at(0.005) / self._at(0.0025) == pytest.approx(4.0, abs=0.05)

    def test_the_remainder_is_negligible_beside_a_first_order_error(self) -> None:
        # At a 1% nuisance error the remainder is smaller by two orders of magnitude,
        # which is the practical content of "second-order".
        assert abs(self._at(0.01)) < 0.01 * abs(self._at(1.0))
