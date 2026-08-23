r"""Is the *regime* estimating equation's remainder second-order?

The counterpart of :mod:`tests.unit.test_remainder` for
:math:`\Psi_r(P) = E_W \sum_a g^\star_r(a \mid W)\,\bar Q(a, W)`.  Expanding the plug-in
at a pair of nuisance guesses :math:`(\hat g, \bar Q)`,

.. math::

    \Psi_r(\bar Q) - \Psi_r(P_0) + P_0 D_r^*(\hat g, \bar Q) = R_2 ,
    \qquad
    R_2 = \sum_w P(w) \sum_a g^\star_r(a \mid w)\,
          \frac{\hat g(a \mid w) - g_0(a \mid w)}{\hat g(a \mid w)}\,
          \bigl(\bar Q(a, w) - \bar Q_0(a, w)\bigr),

which is again a *product* of the two nuisance errors -- so the regime estimands are
doubly robust on exactly the terms the arm-indexed ones are, and for the same reason.
The regime enters as a weight on the product and cannot rescue either factor: a rule that
concentrates on the arms where :math:`\hat g` is worst has a *larger* remainder, not a
different kind of one.

Nothing here runs the targeting step.  The remainder is a property of the estimating
equation, evaluated at nuisances that are wrong on purpose, so a fluctuation that merely
converged cannot satisfy these assertions.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly.fluctuation.iterative import InitialFit
from cleverly.fluctuation.submodel import submodel_for
from cleverly.inference.influence import regime_means
from tests import discrete_law as law

#: The same deliberately wrong nuisances :mod:`tests.unit.test_remainder` uses, so the two
#: modules disagree about nothing except which functional they expand.
WRONG_G = np.array([0.55, 0.35, 0.45])
WRONG_Q = law.Q + np.array([[0.10, -0.15], [-0.20, 0.10], [0.05, 0.20]])

#: Regime codes follow the order the densities are declared in.
NAMES = tuple(law.REGIMES)


def _densities(covariate: np.ndarray) -> np.ndarray:
    """``(n, K, R)`` regime densities for the sample, from the oracle's declaration."""
    return np.stack([law.REGIMES[name][covariate] for name in NAMES], axis=2)


def _expansion(g_hat: np.ndarray, q_hat: np.ndarray) -> dict[str, float]:
    """``R_2`` per regime, with the plug-in and ``P_0 D*`` both taken from the library."""
    frame = law.frame()
    covariate = frame["W"].to_numpy().astype(int)
    treatment = frame["A"].to_numpy(dtype=float)
    outcome = frame["Y"].to_numpy(dtype=float)

    at_one, at_zero = q_hat[covariate, 1], q_hat[covariate, 0]
    initial = InitialFit(
        observed=np.where(treatment == 1.0, at_one, at_zero),
        arms={1.0: at_one, 0.0: at_zero},
    )
    star = _densities(covariate)
    submodel = submodel_for("regime", treatment, g_hat[covariate], regimes=star)
    means = regime_means(outcome, initial, submodel, star, np.ones(law.N))
    # The sample realises the law exactly, so the sample mean of the influence curve *is*
    # P_0 D*.
    return {
        name: means[float(index)].psi
        - law.TRUTH[f"ey_regime[{name}]"]
        + float(np.mean(means[float(index)].influence_curve))
        for index, name in enumerate(NAMES)
    }


def _exact_remainder(g_hat: np.ndarray, q_hat: np.ndarray) -> dict[str, float]:
    """The remainder as theory says it must be: an exact signed sum carrying both errors."""
    mechanism = np.column_stack([1.0 - g_hat, g_hat])
    truth = np.column_stack([1.0 - law.G, law.G])
    factor = (mechanism - truth) / mechanism
    return {
        name: float(np.sum(law.P_W * (law.REGIMES[name] * factor * (q_hat - law.Q)).sum(axis=1)))
        for name in NAMES
    }


class TestTheRemainderCarriesBothNuisanceErrors:
    @pytest.mark.parametrize("name", NAMES)
    def test_matches_the_closed_form(self, name: str) -> None:
        actual = _expansion(WRONG_G, WRONG_Q)[name]
        assert actual == pytest.approx(_exact_remainder(WRONG_G, WRONG_Q)[name], abs=1e-12)
        assert abs(actual) > 1e-3, "the misspecification is too mild to test anything"

    @pytest.mark.parametrize("name", NAMES)
    def test_vanishes_when_the_mechanism_is_right(self, name: str) -> None:
        assert _expansion(law.G, WRONG_Q)[name] == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("name", NAMES)
    def test_vanishes_when_the_outcome_regression_is_right(self, name: str) -> None:
        assert _expansion(WRONG_G, law.Q)[name] == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("name", NAMES)
    def test_vanishes_when_both_are_right(self, name: str) -> None:
        assert _expansion(law.G, law.Q)[name] == pytest.approx(0.0, abs=1e-12)


class TestTheRegimeWeightsTheRemainderRatherThanChangingIt:
    def test_a_static_regime_reproduces_the_arm_remainder(self) -> None:
        """``always 0`` must have exactly the ``ey0`` remainder, which is checked elsewhere."""
        expected = -float(
            np.sum(law.P_W * (WRONG_G - law.G) / (1.0 - WRONG_G) * (WRONG_Q[:, 0] - law.Q[:, 0]))
        )
        assert _expansion(WRONG_G, WRONG_Q)["never"] == pytest.approx(expected, abs=1e-12)

    def test_a_regime_concentrating_on_the_worst_arm_has_the_larger_remainder(self) -> None:
        """The regime is a weight on the product, so where it puts its mass matters.

        Not a restatement of the closed form: it is the practical consequence a reader
        needs, that a rule can make the second-order bias worse without making it
        first-order.
        """
        remainders = _expansion(WRONG_G, WRONG_Q)
        contributions = {
            name: abs(value) for name, value in _exact_remainder(WRONG_G, WRONG_Q).items()
        }
        assert contributions == pytest.approx({k: abs(v) for k, v in remainders.items()}, abs=1e-12)
        # The tilt spreads its mass over both arms, so its remainder sits between the two
        # deterministic regimes' rather than outside them.
        low, high = sorted((remainders["never"], remainders["rule"]))
        assert low <= remainders["tilt"] <= high
