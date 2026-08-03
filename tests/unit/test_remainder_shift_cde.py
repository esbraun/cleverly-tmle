r"""What does double robustness *mean* for a shift whose outcomes go missing?

:mod:`tests.unit.test_remainder` shows the remainder of the von Mises expansion is a
product of two nuisance errors, so the estimator is consistent when either the mechanism
or the outcome regression is right.  :mod:`tests.unit.test_remainder_mar` shows that with
a third nuisance the obvious generalisation -- "consistent if any one of the three is
right" -- is false, and that what replaces it is a statement about the *product*.  This
module makes the same statement on the shift axis, where the mechanism is a density ratio
rather than a propensity and the arithmetic is therefore not the same.

Working the expansion through for :math:`\Psi_d = E[\bar Q(d(A, W), W)]` gives

.. math::

    R_2 = \sum_{w, a} P_0(w)\,
          \Bigl[\,g_0(a \mid w)\,\frac{\pi_0(a, w)}{\hat\pi(a, w)}\,\hat h(a, w)
                 \;-\; g^d_0(a \mid w)\Bigr]\;
          \bigl(\bar Q_0(a, w) - \hat{\bar Q}(a, w)\bigr),

still a product of two factors.  The right one is the error in :math:`\bar Q`.  The left
one is the error in the **product** of the density ratio and the observation probability:
it is zero when :math:`\hat h = g^d_0 / g_0` and :math:`\hat\pi = \pi_0`, and -- the point
of the module -- it is *also* zero at pairs where neither is.  So the guarantee is:

    consistent if :math:`\bar Q` is right, **or** if the product
    :math:`\hat h \cdot \hat\pi` is right,

which is what ``_SHIFT_ID.dr_condition`` says and what the three classes below pin.  Two
consequences neither of which a reader would guess.  Getting the *density* exactly right
buys nothing on its own: with a wrong missingness model the remainder stays first order,
and the "IPW half" of double robustness is simply not available.  And errors in the two
mechanisms can cancel -- inflate the ratio and deflate :math:`\pi` to match, and the
remainder is exactly zero although both nuisances are wrong everywhere.

Everything is an exact finite sum on :mod:`tests.discrete_law_shift_cde`, so this is
checked deterministically at ``1e-12`` rather than inferred from a simulation.  As in the
parent modules :math:`\Psi` and the closed form for :math:`R_2` are written out longhand
while :math:`D^*` is the library's -- which is the claim under test -- and the targeting
step is never run, so nothing here can be satisfied by a fluctuation that merely
converged.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import tests.discrete_law_shift_cde as law
from cleverly.data import CausalData
from cleverly.fluctuation.iterative import InitialFit
from cleverly.fluctuation.submodel import submodel_for
from cleverly.inference.influence import shift_means
from cleverly.interventions import Shift, ShiftSet
from cleverly.learners.density import ConditionalDensity

SHIFTS = (
    Shift(0.0, cap=law.CAP, name="natural course"),
    Shift(law.DELTA, cap=law.CAP, name="+1"),
)

#: The policy every assertion below is taken on, and its code.
POLICY, CODE = "+1", 1.0

#: A conditional density wrong at every covariate value, with rows that still sum to one.
#: Chosen so that :func:`_cancelling`'s solution is comfortably interior -- every entry it
#: produces lands between 0.4 and 0.65 -- rather than merely admissible.
WRONG_G = np.array(
    [
        [0.60, 0.15, 0.10, 0.15],
        [0.10, 0.15, 0.15, 0.60],
        [0.40, 0.20, 0.10, 0.30],
    ]
)

#: What :func:`_cancelling` leaves at the lowest dose, where the ``+1`` policy's covariate
#: is identically zero -- nothing can be shifted *to* the lowest dose, so ``pi`` there
#: multiplies a zero and the equation does not constrain it.  Set to a deliberately wrong
#: value so the cancelling mechanism is wrong at every dose and not only where it matters.
_UNCONSTRAINED_PI = 0.4

#: A missingness mechanism wrong at every dose, mixed in sign so errors cannot cancel into
#: a spurious pass, and comfortably inside ``(0, 1]``.
WRONG_PI = law.PI_EXACT + np.array(
    [
        [0.20, -0.15, -0.25, 0.20],
        [-0.30, 0.15, 0.30, -0.20],
        [0.25, 0.30, -0.15, 0.15],
    ]
)

#: An outcome regression wrong at every dose.
WRONG_Q = law.QBAR_MARGINAL_EXACT + np.array(
    [
        [0.10, -0.15, 0.05, -0.10],
        [-0.20, 0.10, -0.05, 0.15],
        [0.05, 0.20, -0.10, 0.10],
    ]
)


def _induced(density: np.ndarray, cap: float) -> np.ndarray:
    """``g^d(b | w)`` for an arbitrary density -- the mass the policy moves onto each dose.

    The law exports this for its *own* density only; a remainder needs it for a guess, and
    writing it here keeps the longhand independent of what the library computes.
    """
    out = np.zeros_like(density)
    for source, target in enumerate(law.SHIFT_MAPS[cap]):
        out[:, target] += density[:, source]
    return out


def _cancelling(density: np.ndarray) -> np.ndarray:
    r"""A :math:`\hat\pi` making the product right while both nuisances are wrong.

    Solves the left factor of :math:`R_2` for :math:`\hat\pi` at the given wrong density:
    the bracket vanishes when
    :math:`\hat\pi = \pi_0\, \hat h / h_0`.  Every entry lands inside ``(0, 1)`` for
    :data:`WRONG_G`, which is checked as a premise rather than assumed.

    The lowest dose is the one place the equation says nothing: :math:`g^d_0` is zero
    there, because under a ``+1`` policy no unit can be *shifted* onto the lowest dose, so
    the covariate is zero and :math:`\hat\pi` multiplies it.  That entry takes
    :data:`_UNCONSTRAINED_PI` instead -- which is itself a small claim worth making, and
    the remainder coming out zero anyway is what makes it.
    """
    ratio_hat = _induced(density, law.CAP) / density
    ratio_true = _induced(law.G_EXACT, law.CAP) / law.G_EXACT
    free = ratio_true == 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        solved = law.PI_EXACT * ratio_hat / ratio_true
    return np.where(free, _UNCONSTRAINED_PI, solved)


def _fit(density: np.ndarray, pi_hat: np.ndarray, q_hat: np.ndarray) -> tuple[float, np.ndarray]:
    """The library's plug-in and influence curve at nuisances it did not fit.

    No targeting step: ``InitialFit`` holds ``Qbar`` as handed to it and the covariate is
    built from ``density`` and ``pi_hat``.  Because the sample realises the law exactly,
    the sample mean of the influence curve *is* ``P_0 D*``.
    """
    frame = law.frame()
    covariate = frame["W"].to_numpy().astype(int)
    dose = frame["A"].to_numpy(dtype=float)
    observed = frame["Delta"].to_numpy(dtype=float) == 1.0
    outcome = np.nan_to_num(frame["Y"].to_numpy(dtype=float))
    index = np.rint(dose).astype(int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = CausalData.from_arrays(
            outcome,
            dose,
            covariate.reshape(-1, 1).astype(float),
            treatment_kind="continuous",
            delta=observed.astype(float),
        )
        shifts = ShiftSet.evaluate(SHIFTS, data, ConditionalDensity(density[covariate], law.EDGES))

    maps = [np.asarray(law.POLICIES[name]) for name in shifts.labels.values()]
    at = np.column_stack([index] + [mapping[index] for mapping in maps])
    initial = InitialFit(
        q_hat[covariate, index],
        {
            float(code): q_hat[covariate, mapping[index]]
            for code, mapping in zip(shifts.codes, maps, strict=True)
        },
    )
    submodel = submodel_for(
        "mtp",
        dose,
        np.zeros((dose.size, 0)),
        arms=(),
        shifts=shifts.design,
        missingness=pi_hat[covariate[:, None], at],
    )
    mean = shift_means(outcome, initial, submodel, np.ones(dose.size), observed)[CODE]
    return float(mean.psi), np.asarray(mean.influence_curve)


def _remainder(density: np.ndarray, pi_hat: np.ndarray, q_hat: np.ndarray) -> float:
    """``psi_hat - Psi_0 + P_0 D*``, from the library's own two pieces."""
    psi, curve = _fit(density, pi_hat, q_hat)
    return psi - law.TRUTH[None][f"ey_shift[{POLICY}]"] + float(np.mean(curve))


def _product_form(density: np.ndarray, pi_hat: np.ndarray, q_hat: np.ndarray) -> float:
    """``R_2`` as theory says it must be, written out from the constants alone."""
    marginal = law.PROBS.sum(axis=(1, 2, 3))
    ratio_hat = _induced(density, law.CAP) / density
    left = law.G_EXACT * (law.PI_EXACT / pi_hat) * ratio_hat - _induced(law.G_EXACT, law.CAP)
    error = law.QBAR_MARGINAL_EXACT - q_hat
    return float(np.sum(marginal[:, None] * left * error))


class TestThePremisesHold:
    def test_the_wrong_nuisances_really_are_wrong_and_admissible(self) -> None:
        np.testing.assert_allclose(WRONG_G.sum(axis=1), 1.0, rtol=0, atol=1e-12)
        assert np.min(WRONG_G) > 0.0
        for guess, truth in ((WRONG_PI, law.PI_EXACT), (WRONG_Q, law.QBAR_MARGINAL_EXACT)):
            assert np.all(np.abs(guess - truth) > 1e-3), "every entry must be wrong"
            assert np.min(guess) > 0.0 and np.max(guess) < 1.0

    def test_the_cancelling_mechanism_is_a_probability(self) -> None:
        pi_hat = _cancelling(WRONG_G)
        assert np.min(pi_hat) > 0.0 and np.max(pi_hat) < 1.0
        assert np.all(np.abs(pi_hat - law.PI_EXACT) > 1e-3), "wrong at every dose, not merely most"


class TestTheRemainderIsAProductOfTwoErrors:
    def test_it_matches_the_closed_form(self) -> None:
        observed = _remainder(WRONG_G, WRONG_PI, WRONG_Q)
        assert observed == pytest.approx(_product_form(WRONG_G, WRONG_PI, WRONG_Q), abs=1e-12)
        assert abs(observed) > 1e-3, "the guesses must leave a remainder worth measuring"

    def test_a_right_outcome_regression_kills_it_whatever_the_mechanisms_are(self) -> None:
        assert _remainder(WRONG_G, WRONG_PI, law.QBAR_MARGINAL_EXACT) == pytest.approx(
            0.0, abs=1e-12
        )

    def test_right_mechanisms_kill_it_whatever_the_regression_is(self) -> None:
        assert _remainder(law.G_EXACT, law.PI_EXACT, WRONG_Q) == pytest.approx(0.0, abs=1e-12)


class TestWhatTheThirdNuisanceCosts:
    """The two statements that make this different from ordinary double robustness."""

    def test_a_right_density_alone_buys_nothing(self) -> None:
        # The classical "IPW half": with the mechanism right the remainder should vanish.
        # It does not, because the mechanism half here is the product with pi.
        remainder = _remainder(law.G_EXACT, WRONG_PI, WRONG_Q)
        assert abs(remainder) > 1e-3
        assert remainder == pytest.approx(_product_form(law.G_EXACT, WRONG_PI, WRONG_Q), abs=1e-12)

    def test_a_right_missingness_model_alone_buys_nothing_either(self) -> None:
        remainder = _remainder(WRONG_G, law.PI_EXACT, WRONG_Q)
        assert abs(remainder) > 1e-3

    def test_two_wrong_mechanisms_whose_product_is_right_kill_it(self) -> None:
        # Both nuisances wrong everywhere, remainder exactly zero: the guarantee really is
        # about the product and not about either factor.
        assert _remainder(WRONG_G, _cancelling(WRONG_G), WRONG_Q) == pytest.approx(0.0, abs=1e-12)


class TestTruncationMovesTheRemainderAndNotTheTarget:
    def test_bounding_the_mechanism_leaves_the_plug_in_alone(self) -> None:
        """``Psi`` is a functional of the law; ``nuisance_bound=`` is not in it.

        The plug-in averages ``Qbar`` at the assigned dose and contains no mechanism at
        all, so truncation cannot move it -- what it moves is ``R_2``, by exactly the
        closed form evaluated at the truncated value.  Truncation buys variance and pays
        in second-order bias; it does not quietly redefine the estimand.
        """
        loose, _ = _fit(law.G_EXACT, law.PI_EXACT, WRONG_Q)
        truncated = np.clip(law.PI_EXACT, 0.6, 1.0)
        tight, _ = _fit(law.G_EXACT, truncated, WRONG_Q)
        assert tight == pytest.approx(loose, abs=1e-12)
        assert np.any(truncated != law.PI_EXACT), "the bound must actually bind"

    def test_but_it_does_move_the_remainder(self) -> None:
        truncated = np.clip(law.PI_EXACT, 0.6, 1.0)
        remainder = _remainder(law.G_EXACT, truncated, WRONG_Q)
        assert abs(remainder) > 1e-3
        assert remainder == pytest.approx(_product_form(law.G_EXACT, truncated, WRONG_Q), abs=1e-12)
