r"""Is the incremental influence curve still the EIF when outcomes are missing?

:mod:`tests.unit.test_influence_gateaux_ipsi` asks this on a law where every outcome is
recorded.  The same question on :mod:`tests.discrete_law_mar`, where roughly half are not,
is the one that was left unanswered: ``incremental=`` used to refuse ``delta=`` outright,
on the grounds that a further mechanism in the outcome half of the clever covariate would
be a different derivation and that no oracle law covered it.  The second half of that was
simply false -- this law is such an oracle -- and with it in hand the first half turns out
to be false too.

What the composition is, and why it is not a guess:

.. math::

    D^*(O) = \frac{\Delta}{\pi(A, W)}\,\frac{q_\delta(A \mid W)}{g(A \mid W)}
                \bigl(Y - \bar Q(A, W)\bigr)
           + \frac{\delta}{D_\delta^2}\bigl(\bar Q(1, W) - \bar Q(0, W)\bigr)(A - g(W))
           + \sum_a q_\delta(a \mid W)\bar Q(a, W) - \Psi(\delta)

-- the ordinary incremental curve with :math:`\pi` dividing the residual term, and
Kennedy's :math:`\partial m/\partial g` term **untouched**.  The asymmetry has a reason:
:math:`q_\delta` is a functional of :math:`P(A \mid W)`, and :math:`A` and :math:`W` are
recorded for every row however much of :math:`Y` is not, so the intervention is defined on
a fully observed sub-law.  Only :math:`\bar Q` is reached by the missingness.

Every assertion below is against a complex-step Gateaux derivative of a longhand
:math:`\Psi(\delta)` on an 18-cell law the library never touches, so "the derivation is the
same one with an extra factor" is checked rather than argued.  The negative controls in
:class:`TestTheNegativeControls` are what stop that from being vacuous: dividing the
mechanism term by :math:`\pi` as well, or reading :math:`\pi` off ``W`` alone, are the two
ways a plausible implementation gets this wrong, and both move the curve four orders past
the window the real assertions use.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly import TMLE
from tests import discrete_law_mar as law
from tests import incrementals as inc
from tests.conftest import OracleMissingness, OracleOutcome, OracleTreatment

MEANS = tuple(law.PER_ARM_NAMES["ey_ipsi"])
CONTRASTS = tuple(law.PER_ARM_NAMES["ate_ipsi"])
ESTIMANDS = MEANS + CONTRASTS

#: Rows whose outcome was never recorded.  Six of the eighteen support points, and the
#: ones no test in the incremental family has ever reached.
UNOBSERVED = tuple(i for i, (_, _, k) in enumerate(law.SUPPORT) if k == law.UNOBSERVED)


@pytest.fixture(scope="module")
def exact_fit():
    """An incremental fit on the MAR law with all three oracle nuisances.

    Both epsilons are zero for the reasons the no-missingness fixture gives, and the
    missingness model being exact is what keeps the first of them zero: within a ``(w, a)``
    cell the clever covariate is constant, so the observed outcomes average to exactly
    ``Qbar(a, w)`` and the weighted score is already solved.  The reported curve is the EIF
    at ``P0`` rather than an estimate of it.
    """
    dgp = law.DiscreteLaw()
    estimator = TMLE(
        outcome_learner=OracleOutcome(dgp),
        treatment_learner=OracleTreatment(dgp),
        missingness_learner=OracleMissingness(dgp),
        cross_fit=False,
        incremental=inc.interventions(law.IPSI_DELTAS),
        estimands="all",
        simultaneous=False,
        random_state=0,
    )
    return estimator.fit(
        law.frame(), outcome="Y", treatment="A", covariates=["W"], delta="Delta"
    ).single()


def _parts(delta: float):
    """``(p_w, g, q, pi, d, mixture, psi)`` of the law, read off the cell counts."""
    p = law.PROBS
    p_w = p.sum(axis=(1, 2))
    p_wa = p.sum(axis=2)
    observed = p[:, :, law.OBSERVED_ZERO] + p[:, :, law.OBSERVED_ONE]
    q = p[:, :, law.OBSERVED_ONE] / observed
    pi = observed / p_wa
    g = p_wa[:, 1] / p_w
    d = delta * g + (1.0 - g)
    mixture = (delta * g * q[:, 1] + (1.0 - g) * q[:, 0]) / d
    return p_w, g, q, pi, d, mixture, float((p_w * mixture).sum())


class TestTheOracleAndTheLibraryDescribeTheSameTilts:
    """The join between :data:`law.IPSI_DELTAS` and :mod:`tests.incrementals`.

    Restated in this law rather than imported from the parent one, so the join has to be
    asserted again here: without it every comparison below could be between two different
    parameters and pass while both were wrong.
    """

    def test_each_declared_tilt_carries_the_oracle_multiplier(self) -> None:
        declared = {item.name: item.delta for item in inc.interventions(law.IPSI_DELTAS)}
        assert declared == law.IPSI_DELTAS

    def test_the_mechanism_is_fitted_on_every_row_not_the_complete_cases(self, exact_fit) -> None:
        """``A`` is recorded for everyone, so ``g`` is not a missing-data problem.

        The distinction has teeth on this law: :data:`law.PI` depends on ``A``, so the
        complete cases carry a different treated fraction and a mechanism fitted on them
        would be wrong at every ``w`` -- and it would be wrong *inside the estimand*, since
        ``q_delta`` is built out of ``g``.
        """
        tilts = exact_fit.nuisance.incremental
        levels = np.rint(exact_fit.data.covariates[:, 0]).astype(int)
        np.testing.assert_allclose(tilts.propensity, law.G_EXACT[levels], atol=1e-12, rtol=0)

        complete = law.PROBS[:, :, law.OBSERVED_ZERO] + law.PROBS[:, :, law.OBSERVED_ONE]
        among_observed = complete[:, 1] / complete.sum(axis=1)
        assert np.max(np.abs(among_observed - law.G_EXACT)) > 1e-2, (
            "the two mechanisms must differ, or this test asserts nothing"
        )

    def test_the_evaluated_density_is_the_odds_tilt_of_the_oracle_mechanism(
        self, exact_fit
    ) -> None:
        tilts = exact_fit.nuisance.incremental
        levels = np.rint(exact_fit.data.covariates[:, 0]).astype(int)
        g = law.G_EXACT[levels]
        for index, delta in enumerate(law.IPSI_DELTAS.values()):
            expected = delta * g / (delta * g + 1.0 - g)
            np.testing.assert_allclose(tilts.values[:, 1, index], expected, atol=1e-12, rtol=0)


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

    def test_the_missingness_is_severe_enough_to_be_testing_something(self) -> None:
        p = law.PROBS
        recorded = float((p[:, :, law.OBSERVED_ZERO] + p[:, :, law.OBSERVED_ONE]).sum())
        assert 0.4 < recorded < 0.6, "roughly half the sample has no outcome"
        assert np.max(np.abs(law.PI[:, 1] - law.PI[:, 0])) > 0.05, (
            "pi must depend on the arm, or dividing by pi(W) alone would go unnoticed"
        )


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

    @pytest.mark.parametrize("name", MEANS)
    def test_an_unrecorded_outcome_contributes_the_two_terms_that_survive(
        self, exact_fit, name: str
    ) -> None:
        """Where ``Delta = 0`` the residual term cannot contribute, but the other two do.

        This is the part of the claim the incremental family has never reached, and it is
        where the two halves of the curve come apart: the plug-in term and the
        ``dm/dg`` term are both functionals of a sub-law that is fully observed, so a row
        with no outcome still carries them. A curve that zeroed everything at ``Delta = 0``
        -- the obvious way to "handle" missingness -- would fail here and nowhere else.
        """
        delta = law.IPSI_DELTAS[name[len("ey_ipsi[") : -1]]
        _, g, q, _, d, mixture, psi = _parts(delta)
        longhand = [
            delta / d[w] ** 2 * (q[w, 1] - q[w, 0]) * (a - g[w]) + mixture[w] - psi
            for w, a, _ in (law.SUPPORT[i] for i in UNOBSERVED)
        ]
        reported = np.asarray(exact_fit.estimates[name].influence_curve)[law.first_row_of()]
        np.testing.assert_allclose(reported[list(UNOBSERVED)], longhand, atol=1e-12, rtol=0)
        assert np.max(np.abs(longhand)) > 1e-2, "and they are not all zero anyway"


class TestTheNaturalCourseIsStillAnIdentity:
    r"""At ``delta = 1`` the curve collapses -- but to the *missing-data* mean, not ``Y``.

    Without missingness ``psi(1) = mean(Y)`` and the curve is ``Y - psi`` row by row,
    whatever the nuisances; ``tests/unit/test_influence_gateaux_ipsi.py`` keeps that as the
    canary that catches an alternation exiting with one score equation still open.  With
    ``delta=`` the same algebra runs -- ``D = 1``, so the ``dm/dg`` term and the plug-in
    collapse into ``Qbar(A, W)`` exactly as they do there -- and what is left is

    .. math::

        \frac{\Delta}{\pi(A, W)}\bigl(Y - \bar Q(A, W)\bigr) + \bar Q(A, W) - \Psi ,

    the textbook influence curve for :math:`E[Y]` under missingness at random.  So the
    canary survives in the sharper form, and the thing it must **not** be equal to is the
    complete-case mean, which on this law sits about 0.115 away.
    """

    NAME = "ey_ipsi[natural course]"

    @pytest.mark.parametrize("name", MEANS)
    def test_the_estimate_is_not_what_a_complete_case_analysis_converges_to(
        self, exact_fit, name: str
    ) -> None:
        """Where a fit that dropped the incomplete rows would land, and how far away it is.

        The bar is 5e-3 rather than the 1e-2 the sibling modules use, and the reason is
        worth recording rather than hiding behind a looser constant.  The gaps here are
        0.0066 at the natural course, 0.0293 at ``odds x2`` and 0.0091 at ``odds x0.5``:
        the natural course is the *weakest* case for this control, because at ``delta = 1``
        the estimand is ``E[Y]`` and the two arms' missingness biases partly cancel in the
        marginal.  ``ey1`` on this law is off by 0.115 by comparison.  Nine orders past the
        1e-12 window the real assertions use is what makes it a control at all, and that is
        asserted rather than eyeballed.
        """
        complete_case = float(law.observed_only_functional(law.PROBS, name))
        gap = abs(exact_fit.estimates[name].psi - complete_case)
        assert gap > 5e-3, (
            "psi is the MAR-identified functional; a fit that renormalised onto the "
            "recorded outcomes would land here instead"
        )
        assert gap / 1e-12 > 1e9

    def test_the_estimate_is_the_missing_data_mean(self, exact_fit) -> None:
        _, _, _, _, _, _, psi = _parts(1.0)
        assert exact_fit.estimates[self.NAME].psi == pytest.approx(psi, abs=1e-12)

    def test_the_curve_is_the_mar_influence_curve_for_the_outcome_mean(self, exact_fit) -> None:
        _, _, q, pi, _, _, psi = _parts(1.0)
        expected = []
        for w, a, k in law.SUPPORT:
            recorded = k != law.UNOBSERVED
            y = 1.0 if k == law.OBSERVED_ONE else 0.0
            residual = (y - q[w, a]) / pi[w, a] if recorded else 0.0
            expected.append(residual + q[w, a] - psi)
        reported = np.asarray(exact_fit.estimates[self.NAME].influence_curve)[law.first_row_of()]
        np.testing.assert_allclose(reported, expected, atol=1e-12, rtol=0)


class TestTheNegativeControls:
    """The two ways a plausible implementation composes the mechanisms wrong.

    Both are mutations of a longhand curve that is first shown to reproduce the oracle
    exactly, so a control that fails is a control that was never testing anything.
    """

    NAME = "ey_ipsi[odds x2]"
    DELTA = 2.0

    def _mutated(self, *, drop_pi: bool = False, pi_on_w: bool = False, pi_on_mech: bool = False):
        _, g, q, pi, d, mixture, psi = _parts(self.DELTA)
        if drop_pi:
            pi = np.ones_like(pi)
        elif pi_on_w:
            # P(Delta = 1 | W) rather than P(Delta = 1 | A, W): the missingness model
            # fitted without the treatment in its design, which is a one-word slip.
            marginal = (pi * np.column_stack([1.0 - g, g])).sum(axis=1)
            pi = np.column_stack([marginal, marginal])
        out = []
        for w, a, k in law.SUPPORT:
            h = (self.DELTA if a == 1 else 1.0) / d[w]
            y = 1.0 if k == law.OBSERVED_ONE else 0.0
            residual = h / pi[w, a] * (y - q[w, a]) if k != law.UNOBSERVED else 0.0
            extra = self.DELTA / d[w] ** 2 * (q[w, 1] - q[w, 0]) * (a - g[w])
            out.append(residual + (extra / pi[w, a] if pi_on_mech else extra) + mixture[w] - psi)
        return np.array(out)

    def test_the_unmutated_control_reproduces_the_oracle(self) -> None:
        np.testing.assert_allclose(self._mutated(), law.eif(self.NAME), atol=1e-12, rtol=0)

    @pytest.mark.parametrize(
        ("kwargs", "why"),
        [
            ({"drop_pi": True}, "pi omitted -- the no-missingness curve on missing data"),
            ({"pi_on_w": True}, "pi read off W alone, with the arm dropped from its design"),
            ({"pi_on_mech": True}, "pi dividing the dm/dg term, which it must not"),
        ],
    )
    def test_each_mutation_moves_the_curve(self, kwargs: dict, why: str) -> None:
        gap = np.max(np.abs(self._mutated(**kwargs) - law.eif(self.NAME)))
        assert gap > 1e-2, why
