r"""What does double robustness mean for a controlled direct effect?

:mod:`tests.unit.test_remainder` shows the remainder of the von Mises expansion is a
product of two nuisance errors, so the ordinary estimator is consistent when either the
propensity or the outcome regression is right.  :mod:`tests.unit.test_remainder_mar` shows
that a third nuisance in the clever covariate does *not* add a third half to the guarantee:
what has to be right is the **product** of the mechanisms.  A controlled direct effect adds
a fourth nuisance, and this module says what that costs.

Working the expansion through for :math:`\Psi_{1,z}` gives

.. math::

    R_2 = \int \left(\frac{g_0(W)\, q_{z,0}(1, W)\, \pi_0(1, W)}
                          {\hat g(W)\, \hat q_z(1, W)\, \hat\pi(1, W)} - 1\right)
                \bigl(\bar Q_0(1, z, W) - \bar Q(1, z, W)\bigr)\, dP_0 ,

still a product of two factors, but now the left one is the error in a **three-way**
product.  So the guarantee, exactly as :mod:`cleverly.estimators.direct_effect` states it:

    consistent if :math:`\bar Q` is right, **or** if the product
    :math:`g\, q_z\, \pi` is right.

The practical reading is the opposite of the reassuring one.  Adding an intermediate
variable does not give the analyst a third chance to be right; it adds a third factor that
must *all* be right together, so the mechanism half of double robustness gets harder to
earn, not easier.  Two consequences are pinned down below, because neither is what a reader
would guess: a correct propensity buys nothing on its own when the intermediate mechanism
is wrong, and errors across the three mechanisms can cancel exactly, leaving a remainder of
zero when not one of the twelve nuisance values is correct.

The arithmetic here is deliberately not re-derived from scratch.  :math:`q_z` enters every
one of the three submodel builders in precisely the position :math:`\pi` occupies -- see
:func:`~cleverly.fluctuation.submodel.mean_submodel` and its two conditional-effect
siblings, where ``pz0``/``pz1`` multiply ``pi0``/``pi1`` and nothing else -- so the closed
forms are the missing-outcome ones under the substitution
:math:`\pi_a \mapsto \pi_a\, q_z(a, \cdot)`.  Writing them that way is itself the claim:
if the library ever moved :math:`q_z` somewhere else in the covariate, these tests would
stop agreeing with it.

Everything is an exact finite sum on :mod:`tests.discrete_law_cde`, so this is checked
deterministically at ``1e-12`` rather than inferred from a simulation.  :math:`\Psi` and the
closed form for :math:`R_2` are written out longhand while :math:`D^*` is the library's --
which is the claim under test -- and the targeting step is never run, so nothing here can
be satisfied by a fluctuation that merely converged.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly.fluctuation.iterative import InitialFit
from cleverly.fluctuation.submodel import submodel_for
from cleverly.inference.influence import atc_estimate, att_estimate, counterfactual_means
from tests import discrete_law_cde as law

ESTIMANDS = ("ey1", "ey0", "ate", "att", "atc")

#: A propensity wrong at every covariate value, and strictly inside ``(0, 1)``.
WRONG_G = np.array([0.55, 0.35, 0.45])

#: An intermediate mechanism wrong in both arms, with mixed signs so errors cannot cancel
#: into a spurious pass.  Stated as ``P(Z = 1 | A, W)``; the level-specific density is
#: taken from it the same way the library takes it from the fitted values.
WRONG_QZ = law.QZ + np.array([[-0.30, 0.15], [-0.25, 0.35], [0.40, 0.30]])

#: A missingness mechanism wrong in both arms.
WRONG_PI = law.PI + np.array([[0.30, -0.15], [-0.20, 0.25], [-0.35, 0.10]])

#: An outcome regression wrong in both arms at both levels.
WRONG_QBAR = law.QBAR + np.array(
    [
        [[0.10, -0.15], [-0.20, 0.10]],
        [[0.05, 0.20], [0.15, -0.10]],
        [[-0.15, 0.10], [0.20, 0.15]],
    ]
)


def _density(qz: np.ndarray, level: int) -> np.ndarray:
    """``q_z(a, w)`` for the targeted level, from ``P(Z = 1 | A = a, W = w)``."""
    return qz if level == 1 else 1.0 - qz


def _cancelling(scale: float, level: int) -> tuple[np.ndarray, np.ndarray]:
    r"""Nuisances wrong everywhere whose products :math:`\hat g\,\hat q_z` are both right.

    Scale the propensity and undo it in the intermediate density, arm by arm.  Both clever
    covariates come out unchanged even though neither nuisance is anywhere near correct --
    which is the sharpest available statement that only the product enters the estimating
    equation.
    """
    g_hat = scale * law.G
    density = _density(law.QZ, level)
    hat = np.column_stack(
        [
            (1.0 - law.G) * density[:, 0] / (1.0 - g_hat),
            law.G * density[:, 1] / g_hat,
        ]
    )
    # Returned on the P(Z = 1 | A, W) scale the library's nuisance uses, so the caller can
    # hand it straight through _density() like any other guess.
    return g_hat, (hat if level == 1 else 1.0 - hat)


def _fit(
    g_hat: np.ndarray,
    qz_hat: np.ndarray,
    pi_hat: np.ndarray,
    q_hat: np.ndarray,
    group: str,
    level: int,
):
    """The library's plug-in and influence curve at nuisances it did not fit.

    No targeting step: ``InitialFit`` holds ``Qbar`` as handed to it, and the clever
    covariate is built from the four guesses.  Because the sample realises the law exactly,
    the sample mean of the influence curve *is* :math:`P_0 D^*`.
    """
    frame = law.frame()
    covariate = frame["W"].to_numpy().astype(int)
    treatment = frame["A"].to_numpy(dtype=float)
    intermediate = frame["Z"].to_numpy().astype(int)
    observed = frame["Delta"].to_numpy(dtype=float) == 1.0
    # The library zero-fills the outcome at unobserved rows; the Delta factor is what keeps
    # those entries out of the arithmetic, so mirror it rather than passing NaN.
    outcome = np.nan_to_num(frame["Y"].to_numpy(dtype=float))

    at_one = q_hat[covariate, 1, level]
    at_zero = q_hat[covariate, 0, level]
    initial = InitialFit(
        # The residual is nonzero only where the row is at the targeted level, so the
        # prediction at the row's own (A, Z) and at (A, z) agree wherever it matters.
        observed=q_hat[covariate, treatment.astype(int), intermediate],
        at_one=at_one,
        at_zero=at_zero,
    )
    density = _density(qz_hat, level)
    submodel = submodel_for(
        group,
        treatment,
        g_hat[covariate],
        treated_fraction=float(treatment.mean()),
        missingness=np.column_stack([pi_hat[covariate, 0], pi_hat[covariate, 1]]),
        intermediate_density=np.column_stack([density[covariate, 0], density[covariate, 1]]),
        selection=(intermediate == level).astype(float),
    )
    weights = np.ones(law.N)
    if group == "mean":
        return counterfactual_means(outcome, initial, submodel, weights, observed)
    estimate = att_estimate if group == "att" else atc_estimate
    return estimate(outcome, initial, submodel, treatment, weights, observed)


def _expansion(
    g_hat: np.ndarray,
    qz_hat: np.ndarray,
    pi_hat: np.ndarray,
    q_hat: np.ndarray,
    level: int,
) -> dict[str, float]:
    """``R_2`` for every estimand at the given nuisance guesses."""
    psi_one, ic_one, psi_zero, ic_zero = _fit(g_hat, qz_hat, pi_hat, q_hat, "mean", level)
    one = psi_one - law.TRUTH[level]["ey1"] + float(np.mean(ic_one))
    zero = psi_zero - law.TRUTH[level]["ey0"] + float(np.mean(ic_zero))

    out = {"ey1": one, "ey0": zero, "ate": one - zero}
    for group in ("att", "atc"):
        psi, ic = _fit(g_hat, qz_hat, pi_hat, q_hat, group, level)
        out[group] = psi - law.TRUTH[level][group] + float(np.mean(ic))
    return out


def _product_form(
    g_hat: np.ndarray,
    qz_hat: np.ndarray,
    pi_hat: np.ndarray,
    q_hat: np.ndarray,
    level: int,
) -> dict[str, float]:
    """The remainder as theory says it must be: a product of two nuisance errors.

    The mechanism factor is the error in the three-way product.  Because ``q_z`` sits
    exactly where ``pi`` sits in every submodel, folding the two into one effective
    observation probability turns these into the missing-outcome formulas verbatim -- so
    what is written here is the missing-outcome closed form under
    ``pi_a -> pi_a * q_z(a, .)``, and agreement with the library is the claim that the
    substitution is the whole of the difference.
    """
    truth = law.QBAR[:, :, level]
    error_one = truth[:, 1] - q_hat[:, 1, level]
    error_zero = truth[:, 0] - q_hat[:, 0, level]
    g_zero, g_hat_zero = 1.0 - law.G, 1.0 - g_hat

    # The effective mechanism: observation probability times intermediate density.
    mech = law.PI * _density(law.QZ, level)
    mech_hat = pi_hat * _density(qz_hat, level)

    # --- counterfactual means: the error in the product g * q_z * pi -------------
    ratio_one = law.G * mech[:, 1] / (g_hat * mech_hat[:, 1])
    ratio_zero = g_zero * mech[:, 0] / (g_hat_zero * mech_hat[:, 0])
    one = float(np.sum(law.P_W * (ratio_one - 1.0) * error_one))
    zero = float(np.sum(law.P_W * (ratio_zero - 1.0) * error_zero))

    # --- ATT / ATC: the plug-in already carries the true arm shares, so the left
    # factor compares the reweighted mechanism against the arm indicator instead.
    treated_share = float(np.sum(law.P_W * law.G))
    att = float(
        np.sum(
            law.P_W
            * (
                law.G * (mech[:, 1] / mech_hat[:, 1] - 1.0) * error_one
                + (law.G - g_zero * mech[:, 0] * (g_hat / g_hat_zero) / mech_hat[:, 0]) * error_zero
            )
        )
        / treated_share
    )
    atc = float(
        np.sum(
            law.P_W
            * (
                (law.G * mech[:, 1] * (g_hat_zero / g_hat) / mech_hat[:, 1] - g_zero) * error_one
                + g_zero * (1.0 - mech[:, 0] / mech_hat[:, 0]) * error_zero
            )
        )
        / (1.0 - treated_share)
    )
    return {"ey1": one, "ey0": zero, "ate": one - zero, "att": att, "atc": atc}


CASES = [(name, level) for level in law.LEVELS for name in ESTIMANDS]


class TestTheRemainderIsAProductOfNuisanceErrors:
    @pytest.mark.parametrize(("name", "level"), CASES)
    def test_matches_the_closed_form(self, name: str, level: int) -> None:
        # All four nuisances wrong, so every factor is active and nothing is zero by
        # accident.
        actual = _expansion(WRONG_G, WRONG_QZ, WRONG_PI, WRONG_QBAR, level)[name]
        expected = _product_form(WRONG_G, WRONG_QZ, WRONG_PI, WRONG_QBAR, level)[name]
        assert actual == pytest.approx(expected, abs=1e-12)
        assert abs(actual) > 1e-3, "the misspecification is too mild to test anything"

    @pytest.mark.parametrize(("name", "level"), CASES)
    def test_vanishes_when_the_outcome_regression_is_right(self, name: str, level: int) -> None:
        # All three mechanisms arbitrarily wrong.  This is the half the outcome model
        # supplies, and it is untouched by the intermediate: a correctly specified
        # regression of Y on (A, Z, W) among the complete cases already identifies the
        # parameter, whatever the mechanisms do.
        remainder = _expansion(WRONG_G, WRONG_QZ, WRONG_PI, law.QBAR, level)[name]
        assert remainder == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize(("name", "level"), CASES)
    def test_vanishes_when_all_three_mechanisms_are_right(self, name: str, level: int) -> None:
        # Qbar arbitrarily wrong.  This is the half inverse-probability weighting supplies
        # -- and it now takes all three mechanisms together.
        remainder = _expansion(law.G, law.QZ, law.PI, WRONG_QBAR, level)[name]
        assert remainder == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize(("name", "level"), CASES)
    def test_vanishes_when_everything_is_right(self, name: str, level: int) -> None:
        assert _expansion(law.G, law.QZ, law.PI, law.QBAR, level)[name] == pytest.approx(
            0.0, abs=1e-12
        )


class TestItIsTheProductOfTheMechanismsThatHasToBeRight:
    """Not "any one of the four nuisances" -- the three mechanisms stand or fall together."""

    @pytest.mark.parametrize(("name", "level"), CASES)
    def test_a_right_propensity_does_not_rescue_a_wrong_intermediate_model(
        self, name: str, level: int
    ) -> None:
        # The generalisation a reader would guess, shown false.  With g exactly right and
        # only q_z wrong the remainder is first-order and the estimate is inconsistent.
        remainder = _expansion(law.G, WRONG_QZ, law.PI, WRONG_QBAR, level)[name]
        assert abs(remainder) > 1e-3, remainder

    @pytest.mark.parametrize(("name", "level"), CASES)
    def test_a_right_intermediate_model_does_not_rescue_a_wrong_propensity(
        self, name: str, level: int
    ) -> None:
        remainder = _expansion(WRONG_G, law.QZ, law.PI, WRONG_QBAR, level)[name]
        assert abs(remainder) > 1e-3, remainder

    @pytest.mark.parametrize(("name", "level"), CASES)
    def test_two_right_mechanisms_do_not_rescue_a_wrong_third(self, name: str, level: int) -> None:
        # The sharpest form of the point: g and q_z both exactly right, only the
        # observation probability wrong, and the guarantee is still gone.
        remainder = _expansion(law.G, law.QZ, WRONG_PI, WRONG_QBAR, level)[name]
        assert abs(remainder) > 1e-3, remainder

    @pytest.mark.parametrize(
        ("name", "level"), [(n, z) for z in law.LEVELS for n in ("ey1", "ey0", "ate")]
    )
    def test_errors_in_the_mechanisms_can_cancel_exactly(self, name: str, level: int) -> None:
        """Both nuisances wrong everywhere, both clever covariates exactly right.

        Only the product enters the estimating equation, so a fit that understates
        treatment assignment and overstates the intermediate density by matching factors
        solves the same equation the truth does -- and the remainder is zero to machine
        precision although not one nuisance value is correct.
        """
        g_hat, qz_hat = _cancelling(0.85, level)
        assert np.min(np.abs(g_hat - law.G)) > 0.03
        assert np.min(np.abs(qz_hat - law.QZ)) > 0.01
        assert np.all(qz_hat > 0.0) and np.all(qz_hat < 1.0)
        remainder = _expansion(g_hat, qz_hat, law.PI, WRONG_QBAR, level)[name]
        assert remainder == pytest.approx(0.0, abs=1e-12)


class TestTruncationRegularisesRatherThanRetargets:
    r"""Bounding :math:`q_z` away from zero is a bias-variance trade, not a new estimand.

    The counterpart of :class:`tests.unit.test_remainder_mar.TestTruncationRegularisesRatherThanRetargets`
    for the mechanism a controlled direct effect adds -- and the one where the trade is
    most easily mis-stated, because :mod:`cleverly.estimators.direct_effect` notes that
    positivity in ``q_z`` is *asymmetric* in ``z``: the covariate divides by the density of
    the level being targeted and by its complement at the other.
    """

    BOUND = 0.4  # binds on the cells where the targeted density is 0.25

    def _truncated(self, level: int) -> np.ndarray:
        """``P(Z = 1 | A, W)`` after bounding the *targeted* density from below."""
        density = np.maximum(_density(law.QZ, level), self.BOUND)
        assert not np.allclose(density, _density(law.QZ, level)), "the bound has to bind"
        return density if level == 1 else 1.0 - density

    @pytest.mark.parametrize("level", law.LEVELS)
    def test_the_plug_in_does_not_move(self, level: int) -> None:
        # The substitution estimator is the average of the targeted predictions over the
        # sample's covariates.  No intermediate density appears in it, so no bound on one
        # can shift it -- with or without truncation, the same number.
        untruncated = _fit(law.G, law.QZ, law.PI, WRONG_QBAR, "mean", level)
        truncated = _fit(law.G, self._truncated(level), law.PI, WRONG_QBAR, "mean", level)
        assert truncated[0] == pytest.approx(untruncated[0], abs=1e-15)
        assert truncated[2] == pytest.approx(untruncated[2], abs=1e-15)

    @pytest.mark.parametrize(("name", "level"), CASES)
    def test_what_moves_is_the_second_order_remainder(self, name: str, level: int) -> None:
        # And it moves to exactly the closed form at the truncated value: the cost of
        # truncating is a remainder computed at a mechanism that is now wrong on purpose,
        # priced by the same product formula as any other misspecification.
        bounded = self._truncated(level)
        actual = _expansion(law.G, bounded, law.PI, WRONG_QBAR, level)[name]
        expected = _product_form(law.G, bounded, law.PI, WRONG_QBAR, level)[name]
        assert actual == pytest.approx(expected, abs=1e-12)
        assert abs(actual) > 1e-3, "the bound is not binding hard enough to test anything"

    @pytest.mark.parametrize(("name", "level"), CASES)
    def test_a_bound_that_does_not_bind_costs_nothing(self, name: str, level: int) -> None:
        # The default nuisance_bound is 0.01 and the smallest density on this law is 0.25,
        # so the estimator runs on the unmodified mechanism -- which is the premise the
        # Gateaux module's exactness rests on.
        density = np.maximum(_density(law.QZ, level), 0.01)
        assert np.allclose(density, _density(law.QZ, level))
        loose = density if level == 1 else 1.0 - density
        assert _expansion(law.G, loose, law.PI, WRONG_QBAR, level)[name] == pytest.approx(
            0.0, abs=1e-12
        )


class TestTheRemainderIsSecondOrder:
    """Shrinking every nuisance error by ``t`` has to shrink the remainder by ``t^2``.

    The closed form already implies it, but stating it as a rate is what connects the
    algebra to the condition the estimator needs: :math:`R_2 = o_P(n^{-1/2})` under a
    *product* rate across the nuisances, not a rate on any one of them alone.  With three
    mechanisms in the product this is the condition that gets harder, not the theorem.
    """

    @staticmethod
    def _at(t: float, level: int) -> float:
        return _expansion(
            law.G + t * (WRONG_G - law.G),
            law.QZ + t * (WRONG_QZ - law.QZ),
            law.PI + t * (WRONG_PI - law.PI),
            law.QBAR + t * (WRONG_QBAR - law.QBAR),
            level,
        )["ate"]

    @pytest.mark.parametrize("level", law.LEVELS)
    def test_halving_the_nuisance_error_quarters_the_remainder(self, level: int) -> None:
        # Not exactly 4: the 1/(g_hat * q_hat * pi_hat) factor moves with t too, so the
        # ratio approaches 4 from one side as t shrinks.
        assert self._at(0.005, level) / self._at(0.0025, level) == pytest.approx(4.0, abs=0.05)

    @pytest.mark.parametrize("level", law.LEVELS)
    def test_the_remainder_is_negligible_beside_a_first_order_error(self, level: int) -> None:
        # At a 1% nuisance error the remainder is smaller by two orders of magnitude,
        # which is the practical content of "second-order".
        assert abs(self._at(0.01, level)) < 0.01 * abs(self._at(1.0, level))
