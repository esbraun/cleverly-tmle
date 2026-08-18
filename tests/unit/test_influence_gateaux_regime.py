r"""Is the regime influence curve the efficient influence function of the regime parameter?

The same question :mod:`tests.unit.test_influence_gateaux` asks of the arm-indexed
estimands, asked of

.. math::

    \Psi_r(P) = E_W \sum_a g^\star_r(a \mid W)\, \bar Q(a, W)

and answered the same way: differentiate a longhand statement of :math:`\Psi_r` along the
contamination path by complex step, and compare against what the estimator reports.  No
clever covariate, no submodel and no library code enters the derivation.

Three regimes are covered rather than one, and the choice is not decoration.  A *static*
regime cannot distinguish a mixture over the arms from a lookup of one column; a
*deterministic rule* cannot distinguish either from code that reads ``g*`` only through
its support.  The stochastic regime, degenerate nowhere, is the one that forces the
mixture to be a mixture.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly.estimators import TMLE
from tests import discrete_law as law
from tests import regimes as reg
from tests.conftest import OracleOutcome, OracleTreatment

MEANS = tuple(law.PER_ARM_NAMES["ey_regime"])
CONTRASTS = tuple(law.PER_ARM_NAMES["ate_regime"])
ESTIMANDS = MEANS + CONTRASTS


@pytest.fixture(scope="module")
def exact_fit():
    """A regime fit on the discrete law with oracle nuisances; see the arm-indexed module."""
    dgp = law.DiscreteLaw()
    estimator = TMLE(
        outcome_learner=OracleOutcome(dgp),
        treatment_learner=OracleTreatment(dgp),
        cross_fit=False,
        interventions=reg.interventions(),
        estimands="all",
        simultaneous=False,
        random_state=0,
    )
    return estimator.fit(law.frame(), outcome="Y", treatment="A").single()


class TestTheOracleAndTheLibraryDescribeTheSameRegimes:
    """The join between :data:`law.REGIMES` and :mod:`tests.regimes`.

    Everything below compares a library estimate against an oracle keyed by regime label.
    If the two sides disagreed about what ``"rule"`` assigns, every later assertion would
    be comparing two different parameters and could pass while both were wrong.
    """

    def test_each_intervention_reproduces_the_declared_density(self) -> None:
        from cleverly.data import CausalData

        data = CausalData.from_frame(law.frame(), outcome="Y", treatment="A", covariates=["W"])
        levels = np.rint(data.covariates[:, 0]).astype(int)
        for intervention in reg.interventions():
            expected = law.REGIMES[intervention.name][levels]
            np.testing.assert_array_equal(intervention.density(data), expected)

    def test_the_regimes_are_of_three_different_kinds(self) -> None:
        never, rule, tilt = (law.REGIMES[name] for name in ("never", "rule", "tilt"))
        assert np.array_equal(never, np.tile(never[0], (3, 1))), "static: same row everywhere"
        assert not np.array_equal(rule, np.tile(rule[0], (3, 1))), "the rule must look at W"
        assert np.all((rule == 0.0) | (rule == 1.0)), "the rule must be deterministic"
        assert np.all((tilt > 0.0) & (tilt < 1.0)), "the tilt must be degenerate nowhere"


class TestThePremisesHold:
    def test_the_gateaux_derivative_has_mean_zero(self) -> None:
        # An influence function is centred by construction; a failure here would indict
        # the numerical derivative rather than the library.
        for name in ESTIMANDS:
            assert float((law.PROBS.reshape(-1) * law.eif(name)).sum()) == pytest.approx(
                0.0, abs=1e-12
            )

    def test_targeting_has_nothing_left_to_do(self, exact_fit) -> None:
        for fluctuation in exact_fit.fluctuations.values():
            assert np.max(np.abs(fluctuation.epsilon)) == pytest.approx(0.0, abs=1e-12)


class TestTheInfluenceCurveIsTheEIF:
    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_matches_the_numerical_gateaux_derivative(self, exact_fit, name: str) -> None:
        reported = np.asarray(exact_fit.estimates[name].influence_curve)[law.first_row_of()]
        np.testing.assert_allclose(reported, law.eif(name), atol=1e-12, rtol=0)

    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_the_point_estimate_is_the_functional(self, exact_fit, name: str) -> None:
        assert exact_fit.estimates[name].psi == pytest.approx(law.TRUTH[name], abs=1e-12)

    def test_the_contrast_influence_curve_is_the_difference_of_the_means(self, exact_fit) -> None:
        """Exact, not approximate: the contrast is built from the two curves.

        Cheap, deterministic, and it fails on the mistake a simulation is worst at
        catching -- a contrast taken against the wrong reference regime.
        """
        for contrast in CONTRASTS:
            left, right = contrast[len("ate_regime[") : -1].split(" vs ")
            expected = (
                exact_fit.estimates[f"ey_regime[{left}]"].influence_curve
                - exact_fit.estimates[f"ey_regime[{right}]"].influence_curve
            )
            np.testing.assert_allclose(
                exact_fit.estimates[contrast].influence_curve, expected, atol=1e-14, rtol=0
            )

    def test_a_static_regime_reproduces_the_arm_parameter(self, exact_fit) -> None:
        """``always 0`` is ``E[Y^0]``, which the arm-indexed oracle already knows."""
        assert exact_fit.estimates["ey_regime[never]"].psi == pytest.approx(
            law.TRUTH["ey0"], abs=1e-12
        )
        np.testing.assert_allclose(
            np.asarray(exact_fit.estimates["ey_regime[never]"].influence_curve)[law.first_row_of()],
            law.eif("ey0"),
            atol=1e-12,
            rtol=0,
        )

    def test_a_wrong_density_ratio_would_be_caught(self) -> None:
        """The negative control: the comparison has teeth.

        Written out at the support points, the EIF of a regime mean is
        ``g*(a | w) / g(w) * (y - Qbar(a, w)) + sum_a g*(a | w) Qbar(a, w) - psi``.
        Scaling the weight by 1.05 -- an error targeting and inference would *share*, so
        one the score check cannot see -- has to move it far outside the ``1e-12`` window
        the assertions above use.
        """
        star = law.REGIMES["tilt"]
        truth = law.TRUTH["ey_regime[tilt]"]

        def hand_written(weight: float) -> np.ndarray:
            values = []
            for w, a, y in law.SUPPORT:
                g = law.G[w] if a == 1 else 1.0 - law.G[w]
                clever = weight * star[w, a] / g
                mixture = float((star[w] * law.Q[w]).sum())
                values.append(clever * (y - law.Q[w, a]) + mixture - truth)
            return np.array(values)

        np.testing.assert_allclose(
            hand_written(1.0), law.eif("ey_regime[tilt]"), atol=1e-12, rtol=0
        )
        assert np.max(np.abs(hand_written(1.05) - law.eif("ey_regime[tilt]"))) > 1e-2
