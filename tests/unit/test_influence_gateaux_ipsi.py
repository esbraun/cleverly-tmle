r"""Is the incremental influence curve the efficient influence function of the tilt?

The same question the arm-, regime- and shift-indexed modules ask, asked of

.. math::

    \Psi(\delta) = E_W\!\left[\frac{\delta g \bar Q(1, W) + (1 - g)\bar Q(0, W)}
                                   {\delta g + 1 - g}\right]

and answered the same way: differentiate a longhand statement of :math:`\Psi(\delta)`
along the contamination path by complex step, and compare against what the estimator
reports.  Here the derivative passes through :math:`g` as well as through
:math:`\bar Q`, which is exactly the term under test -- no ``REGIMES`` entry can exercise
it, because a regime's density is a constant of the law.

Three deltas rather than one.  A sign error in :math:`\partial m/\partial g` survives on
one side of one, so there is a tilt above and a tilt below; and :math:`\delta = 1` is the
natural course, where the influence curve collapses to :math:`Y - \Psi` **row by row**
whatever the nuisances are.  That last one is the sharpest test in the suite: it fails if
the extra term is dropped, if it carries the wrong sign, if the mechanism fluctuation is
not run, and if the alternation exits with one equation still open.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly import TMLE
from tests import discrete_law as law
from tests import incrementals as inc
from tests.conftest import OracleOutcome, OracleTreatment

MEANS = tuple(law.PER_ARM_NAMES["ey_ipsi"])
CONTRASTS = tuple(law.PER_ARM_NAMES["ate_ipsi"])
ESTIMANDS = MEANS + CONTRASTS


@pytest.fixture(scope="module")
def exact_fit():
    """An incremental fit on the discrete law with oracle nuisances.

    Both epsilons are exactly zero here, and for two separate reasons.  ``Qbar`` is exact,
    so the outcome score is already solved; and within each ``w`` the sample contains
    exactly ``N_w g(w)`` treated rows, so ``sum H_g(W)(A - g(W))`` is already zero too.
    The reported curve is therefore the EIF at ``P0`` rather than an estimate of it.
    """
    dgp = law.DiscreteLaw()
    estimator = TMLE(
        outcome_learner=OracleOutcome(dgp),
        treatment_learner=OracleTreatment(dgp),
        cross_fit=False,
        incremental=inc.interventions(),
        estimands="all",
        simultaneous=False,
        random_state=0,
    )
    return estimator.fit(law.frame(), outcome="Y", treatment="A").single()


class TestTheOracleAndTheLibraryDescribeTheSameTilts:
    """The join between :data:`law.IPSI_DELTAS` and :mod:`tests.incrementals`.

    Without it every assertion below could compare two different parameters and pass
    while both were wrong.
    """

    def test_each_declared_tilt_carries_the_oracle_multiplier(self) -> None:
        declared = {item.name: item.delta for item in inc.interventions()}
        assert declared == law.IPSI_DELTAS

    def test_the_evaluated_density_is_the_odds_tilt_of_the_oracle_mechanism(
        self, exact_fit
    ) -> None:
        tilts = exact_fit.nuisance.incremental
        levels = np.rint(exact_fit.data.covariates[:, 0]).astype(int)
        g = law.G_EXACT[levels]
        np.testing.assert_allclose(tilts.propensity, g, atol=1e-12, rtol=0)
        for index, delta in enumerate(law.IPSI_DELTAS.values()):
            expected = delta * g / (delta * g + 1.0 - g)
            np.testing.assert_allclose(tilts.values[:, 1, index], expected, atol=1e-12, rtol=0)

    def test_the_deltas_straddle_one(self) -> None:
        values = list(law.IPSI_DELTAS.values())
        assert min(values) < 1.0 < max(values), "a sign error in dm/dg survives one side"
        assert 1.0 in values, "the natural course is what pins the row-wise identity"


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
            assert fluctuation.mechanism is not None, "the ipsi group must target g too"
            assert np.max(np.abs(fluctuation.mechanism.epsilon)) == pytest.approx(0.0, abs=1e-9)


class TestTheInfluenceCurveIsTheEIF:
    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_matches_the_numerical_gateaux_derivative(self, exact_fit, name: str) -> None:
        reported = np.asarray(exact_fit.estimates[name].influence_curve)[law.first_row_of()]
        np.testing.assert_allclose(reported, law.eif(name), atol=1e-12, rtol=0)

    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_the_point_estimate_is_the_functional(self, exact_fit, name: str) -> None:
        assert exact_fit.estimates[name].psi == pytest.approx(law.TRUTH[name], abs=1e-12)

    @pytest.mark.parametrize("name", CONTRASTS)
    def test_a_contrast_curve_is_the_difference_of_the_curves(self, exact_fit, name: str) -> None:
        left, right = name[len("ate_ipsi[") : -1].split(" vs ")
        difference = (
            exact_fit.estimates[f"ey_ipsi[{left}]"].influence_curve
            - exact_fit.estimates[f"ey_ipsi[{right}]"].influence_curve
        )
        np.testing.assert_allclose(
            exact_fit.estimates[name].influence_curve, difference, atol=1e-14, rtol=0
        )


class TestTheNaturalCourseIsAnIdentity:
    """At ``delta = 1`` the tilt is the mechanism, and the whole curve collapses.

    Row by row, and for *any* nuisances: ``m - Qbar(A,W) + (Qbar(1,W) - Qbar(0,W))(A - g)``
    is identically zero when ``D = 1``, so the influence curve is ``Y - Psi`` and the
    estimate is ``mean(Y)``.  Nothing about the fit is assumed here, which is what makes
    this the canary for the alternation: it was a 6e-5 discrepancy in exactly this
    quantity that caught the loop testing a stale mechanism score.
    """

    NAME = "ey_ipsi[natural course]"

    def test_the_estimate_is_the_sample_mean(self, exact_fit) -> None:
        assert exact_fit.estimates[self.NAME].psi == pytest.approx(
            float(np.mean(exact_fit.data.outcome)), abs=1e-12
        )

    def test_the_influence_curve_is_the_outcome_centred(self, exact_fit) -> None:
        estimate = exact_fit.estimates[self.NAME]
        expected = np.asarray(exact_fit.data.outcome, dtype=float) - estimate.psi
        np.testing.assert_allclose(estimate.influence_curve, expected, atol=1e-12, rtol=0)


class TestItIsNotTheStochasticRegimeThatInducesIt:
    """The negative control, on the terms the shift axis already sets.

    A :class:`~cleverly.interventions.Stochastic` regime at the density ``q_delta``
    evaluated at the *true* mechanism is a legitimate known regime on this law, and it has
    the same mean and, entry for entry, the same clever covariate.  Its influence curve is
    this one without the ``dm/dg`` term -- and the difference is not a wash: the extra
    term is mean zero given ``W`` and orthogonal to both halves of the regime curve, so

        Var(D_ipsi) = Var(D_regime) + Var(extra)

    exactly.  Delegating one to the other would therefore report a standard error that is
    too *small*, always, and only a test of this shape would catch it.
    """

    DELTA = 2.0
    NAME = "ey_ipsi[odds x2]"

    @staticmethod
    def _regime_curve(delta: float) -> np.ndarray:
        """The oracle EIF of the *regime* at the induced density, written longhand."""
        p_w = law.PROBS.sum(axis=(1, 2))
        p_wa = law.PROBS.sum(axis=2)
        g = p_wa[:, 1] / p_w
        q = law.PROBS[:, :, 1] / p_wa
        d = delta * g + 1.0 - g
        density = np.column_stack([(1.0 - g) / d, delta * g / d])
        mixture = (density * q).sum(axis=1)
        psi = float((p_w * mixture).sum())
        curve = []
        for w, a, y in law.SUPPORT:
            h = density[w, a] / (g[w] if a == 1 else 1.0 - g[w])
            curve.append(h * (y - q[w, a]) + mixture[w] - psi)
        return np.array(curve)

    @staticmethod
    def _extra_term(delta: float) -> np.ndarray:
        p_w = law.PROBS.sum(axis=(1, 2))
        p_wa = law.PROBS.sum(axis=2)
        g = p_wa[:, 1] / p_w
        q = law.PROBS[:, :, 1] / p_wa
        d = delta * g + 1.0 - g
        return np.array(
            [delta * (q[w, 1] - q[w, 0]) / d[w] ** 2 * (a - g[w]) for w, a, _ in law.SUPPORT]
        )

    def test_the_two_means_agree_exactly(self) -> None:
        p_w = law.PROBS.sum(axis=(1, 2))
        p_wa = law.PROBS.sum(axis=2)
        g = p_wa[:, 1] / p_w
        q = law.PROBS[:, :, 1] / p_wa
        d = self.DELTA * g + 1.0 - g
        induced = float((p_w * (self.DELTA * g * q[:, 1] + (1.0 - g) * q[:, 0]) / d).sum())
        assert induced == pytest.approx(law.TRUTH[self.NAME], abs=1e-12)

    def test_the_two_curves_do_not(self) -> None:
        gap = np.max(np.abs(law.eif(self.NAME) - self._regime_curve(self.DELTA)))
        assert gap > 1e-2, (
            "the regime curve is missing dm/dg and must differ by far more than 1e-12"
        )

    def test_the_gap_is_exactly_the_missing_term(self) -> None:
        np.testing.assert_allclose(
            law.eif(self.NAME) - self._regime_curve(self.DELTA),
            self._extra_term(self.DELTA),
            atol=1e-12,
            rtol=0,
        )

    def test_the_variance_decomposition_is_exact(self) -> None:
        """Orthogonal, so the naive regime treatment understates the standard error."""
        p = law.PROBS.reshape(-1)
        full, regime, extra = (
            law.eif(self.NAME),
            self._regime_curve(self.DELTA),
            self._extra_term(self.DELTA),
        )

        def variance(values: np.ndarray) -> float:
            return float((p * values**2).sum() - ((p * values).sum()) ** 2)

        assert variance(full) == pytest.approx(variance(regime) + variance(extra), abs=1e-12)
        assert variance(regime) < variance(full), "dropping the term would shrink the SE"


class TestTheNegativeControls:
    """Each plausible way of building the curve wrong, shown to move it past ``1e-2``.

    Four orders past the window the real assertions use, so a control that fails is a
    control that was never testing anything.
    """

    NAME = "ey_ipsi[odds x2]"
    DELTA = 2.0

    def _mutated(self, *, scale: float = 1.0, drop: bool = False, sign: float = 1.0) -> np.ndarray:
        p_w = law.PROBS.sum(axis=(1, 2))
        p_wa = law.PROBS.sum(axis=2)
        g = p_wa[:, 1] / p_w
        q = law.PROBS[:, :, 1] / p_wa
        d = self.DELTA * g + 1.0 - g
        density = np.column_stack([(1.0 - g) / d, self.DELTA * g / d])
        mixture = (density * q).sum(axis=1)
        psi = float((p_w * mixture).sum())
        out = []
        for w, a, y in law.SUPPORT:
            h = density[w, a] / (g[w] if a == 1 else 1.0 - g[w])
            extra = 0.0 if drop else sign * scale * self.DELTA * (q[w, 1] - q[w, 0]) / d[w] ** 2
            out.append(h * (y - q[w, a]) + extra * (a - g[w]) + mixture[w] - psi)
        return np.array(out)

    def test_the_unmutated_control_reproduces_the_oracle(self) -> None:
        # Without this the three below could pass because the longhand is simply wrong.
        np.testing.assert_allclose(self._mutated(), law.eif(self.NAME), atol=1e-12, rtol=0)

    @pytest.mark.parametrize(
        ("kwargs", "why"),
        [
            ({"drop": True}, "the dm/dg term omitted -- the regime curve"),
            # 10% rather than the 5% the other modules use: the term is worth about 0.19
            # on this law, so a 5% error moves the curve by 9.6e-3 and lands just under
            # the bar. The bar is what is calibrated here, not the mutation -- and a
            # control that squeaked past would be worth less than one sized to clear it.
            ({"scale": 1.1}, "the dm/dg term mis-scaled by 10%"),
            ({"sign": -1.0}, "the dm/dg term with the wrong sign"),
        ],
    )
    def test_each_mutation_moves_the_curve(self, kwargs: dict, why: str) -> None:
        gap = np.max(np.abs(self._mutated(**kwargs) - law.eif(self.NAME)))
        assert gap > 1e-2, why

    def test_a_five_percent_error_is_still_far_past_the_assertion_window(self) -> None:
        """What the 1e-2 bar costs in sensitivity, measured rather than assumed."""
        gap = np.max(np.abs(self._mutated(scale=1.05) - law.eif(self.NAME)))
        assert 1e-3 < gap < 1e-2, "just under the bar, which is why the control uses 10%"
        assert gap / 1e-12 > 1e9, "and nine orders past what the real assertions allow"
