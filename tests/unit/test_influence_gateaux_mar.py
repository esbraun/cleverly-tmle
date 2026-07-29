r"""Is the influence curve right when outcomes are *missing*?

:mod:`tests.unit.test_influence_gateaux` establishes that the library's influence curve is
the efficient influence function, from the definition, on a law it can hold in its hand.
It says nothing about the ``delta=`` path: its law has no :math:`\Delta` dimension, so the
clever covariate it checks has no :math:`1/\pi` factor, its plug-in has no rows with a
missing outcome to decide what to do with, and its fluctuation is not restricted to
anything.

This module makes the same argument on :mod:`tests.discrete_law_mar`, whose support *is*
the observed-data support -- ``(W, A, Delta)`` always, ``Y`` only when ``Delta = 1``.  The
Gateaux derivative of the identification formula at each of its eighteen points is the
efficient influence function of the observed-data model, computed by complex step from a
longhand statement of :math:`\Psi` that shares no code with the library.  Whatever the
estimator reports has to equal it, at ``1e-12``.

Six of those eighteen points have ``Delta = 0``.  There the residual term cannot
contribute at all, so the derivative is :math:`\bar Q(a, W) - \psi` exactly -- which is
worth stating as its own claim, because it is the one place where "the plug-in averages
over everybody, the residual only over the complete cases" becomes visible in a single
number.  An estimator that quietly conditioned on being observed would get every one of
those six wrong.

The negative controls at the bottom are the point of the module as much as the assertions
are.  Each takes one of the ways this could plausibly have been built wrong -- omit
:math:`1/\pi`; average the plug-in over the complete cases; regress ``Delta * Y`` on
``(A, W)`` instead of ``Y`` among the observed -- and shows it moves the answer by more
than ``1e-2``, four orders of magnitude above the window the real assertions use.  A
mistake of any of those kinds solves its own score equation to machine precision and
passes ``score_check()``, so this is where it would have to be caught.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from cleverly import TMLE
from tests import discrete_law_mar as law
from tests.conftest import OracleMissingness, OracleOutcome, OracleTreatment

ESTIMANDS = ("ey1", "ey0", "ate", "att", "atc", "rr", "or")


@pytest.fixture(scope="module")
def exact_fit():
    """TMLE on the MAR law with all three oracle nuisances.

    ``cross_fit=False`` because there is nothing to cross-fit: the oracles do not learn
    from the data, and out-of-fold prediction would only add fold bookkeeping to a fit
    whose answer is already determined.
    """
    dgp = law.DiscreteLaw()
    estimator = TMLE(
        outcome_learner=OracleOutcome(dgp),
        treatment_learner=OracleTreatment(dgp),
        missingness_learner=OracleMissingness(dgp),
        cross_fit=False,
        estimands="all",
        simultaneous=False,
        random_state=0,
    )
    return estimator.fit(law.frame(), outcome="Y", treatment="A", covariates=["W"], delta="Delta")


class TestTheSampleRealisesTheLaw:
    """The premises the rest of the module rests on, asserted rather than assumed."""

    def test_the_empirical_conditional_law_is_the_true_one(self) -> None:
        frame = law.frame()
        assert len(frame) == law.N
        # Roughly half the sample has no outcome, so this is not a token amount of
        # missingness that the estimator could ignore and still look right.
        assert frame["Delta"].mean() == pytest.approx(0.478, abs=1e-15)
        for w in range(3):
            rows = frame["W"] == w
            assert frame.loc[rows, "A"].mean() == pytest.approx(law.G[w], abs=1e-15)
            for a in range(2):
                arm = rows & (frame["A"] == a)
                assert frame.loc[arm, "Delta"].mean() == pytest.approx(law.PI[w, a], abs=1e-15)
                seen = arm & (frame["Delta"] == 1.0)
                assert frame.loc[seen, "Y"].mean() == pytest.approx(law.Q[w, a], abs=1e-15)

    def test_the_missingness_mechanism_depends_on_both_arm_and_covariate(self) -> None:
        # Both dependencies are load-bearing.  Without the covariate dependence the
        # complete cases would carry the right marginal of W and the plug-in control
        # below would have nothing to detect; without the arm dependence the two columns
        # of the missingness nuisance would be interchangeable.
        assert law.PI[:, 0].std() > 0.2 and law.PI[:, 1].std() > 0.2
        assert np.max(np.abs(law.PI[:, 1] - law.PI[:, 0])) > 0.05

    def test_the_gateaux_derivative_has_mean_zero(self) -> None:
        # An influence function is centred by construction.  If this failed, the
        # numerical derivative -- not the library -- would be the thing that is wrong.
        for name in ESTIMANDS:
            assert float((law.PROBS.reshape(-1) * law.eif(name)).sum()) == pytest.approx(
                0.0, abs=1e-12
            )

    def test_targeting_has_nothing_left_to_do(self, exact_fit) -> None:
        # Within a (w, a) cell the clever covariate is constant and the observed outcomes
        # average to exactly Qbar(a, w), so the score is already zero at epsilon = 0.
        # This is what makes the reported influence curve the EIF at P_0 rather than an
        # estimate of it -- and it holds only because the fluctuation is restricted to
        # the rows with an outcome.
        for fluctuation in exact_fit.fluctuations.values():
            assert np.max(np.abs(fluctuation.epsilon)) == pytest.approx(0.0, abs=1e-12)

    def test_no_bound_binds(self, exact_fit) -> None:
        # The law is built so that g, pi and Qbar all sit well inside their truncation
        # windows.  If one bound started to bite, the estimator would be solving a
        # different score equation and the assertions below would be testing that.
        nuisance = exact_fit.nuisance
        assert float(np.min(nuisance.propensity)) > 0.2
        assert float(np.max(nuisance.propensity)) < 0.8
        assert nuisance.missingness is not None
        assert float(np.min(nuisance.missingness)) > 10.0 * exact_fit.config.missingness_bound


class TestTheInfluenceCurveIsTheEIF:
    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_matches_the_numerical_gateaux_derivative(self, exact_fit, name: str) -> None:
        estimate = exact_fit.estimates[name]
        reported = np.asarray(estimate.influence_curve)[law.first_row_of()]
        np.testing.assert_allclose(reported, law.eif(name), atol=1e-12, rtol=0)

    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_the_point_estimate_is_the_functional(self, exact_fit, name: str) -> None:
        estimate = exact_fit.estimates[name]
        psi = estimate.log_psi if estimate.scale == "ratio" else estimate.psi
        assert psi == pytest.approx(law.TRUTH[name], abs=1e-12)

    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_the_unobserved_rows_carry_no_residual_term(self, exact_fit, name: str) -> None:
        """At ``Delta = 0`` the influence curve is the plug-in term alone.

        Nothing about the row's outcome is known, so the only thing it can contribute is
        how its covariates move the average of the targeted predictions.  Reading that
        off the reported curve is the sharpest single statement that the ``Delta`` factor
        is in the right place: it is exactly zero residual, not a small one.
        """
        reported = np.asarray(exact_fit.estimates[name].influence_curve)[law.first_row_of()]
        unobserved = [i for i, (_, _, k) in enumerate(law.SUPPORT) if k == law.UNOBSERVED]
        # Within a (w, a) pair the two arms of an unobserved cell differ only through the
        # plug-in term, which for the mean estimands does not depend on the realised arm.
        expected = law.eif(name)[unobserved]
        np.testing.assert_allclose(reported[unobserved], expected, atol=1e-12, rtol=0)
        if name in ("ey1", "ey0"):
            arm = 1 if name == "ey1" else 0
            longhand = [law.Q[law.SUPPORT[i][0], arm] - law.TRUTH[name] for i in unobserved]
            np.testing.assert_allclose(reported[unobserved], longhand, atol=1e-12, rtol=0)

    def test_the_ate_influence_curve_is_the_difference_of_the_two_means(self, exact_fit) -> None:
        ate = np.asarray(exact_fit.estimates["ate"].influence_curve)
        one = np.asarray(exact_fit.estimates["ey1"].influence_curve)
        zero = np.asarray(exact_fit.estimates["ey0"].influence_curve)
        np.testing.assert_allclose(ate, one - zero, atol=1e-12, rtol=0)

    def test_the_score_equation_is_solved(self, exact_fit) -> None:
        assert exact_fit.validation.score_check().passed


def _ey1_influence(
    *,
    missingness: np.ndarray,
    centre: np.ndarray,
    psi: float,
) -> np.ndarray:
    r"""``EY1``'s influence curve at the support points, written out by hand.

    .. math::

        \frac{\mathbb 1\{a = 1\}\,\Delta}{g(w)\,\pi(a, w)}\bigl(y - \bar Q(a, w)\bigr)
        + \bar Q(1, w) - \psi

    Parameterised by the three things a wrong implementation would get wrong -- the
    observation probability in the denominator, the regression the residual is taken
    against and the plug-in is centred on, and the value it is centred at -- so that each
    can be perturbed on its own below.
    """
    values = []
    for w, a, k in law.SUPPORT:
        if k == law.UNOBSERVED:
            values.append(centre[w, 1] - psi)
            continue
        clever = (a == 1) / (law.G[w] * missingness[w, a])
        values.append(clever * (float(k) - centre[w, a]) + centre[w, 1] - psi)
    return np.array(values)


class TestAWrongConstructionWouldBeCaught:
    """The negative controls: each assertion above has teeth only if these hold."""

    def test_the_longhand_influence_curve_reproduces_the_derivative(self) -> None:
        # The baseline the three controls are perturbations of.  If this drifted, the
        # controls would be measuring the drift rather than the mistake.
        np.testing.assert_allclose(
            _ey1_influence(missingness=law.PI, centre=law.Q, psi=law.TRUTH["ey1"]),
            law.eif("ey1"),
            atol=1e-12,
            rtol=0,
        )

    def test_dropping_the_observation_probability_would_be_caught(self) -> None:
        """``1/pi`` omitted from the clever covariate.

        The mistake targeting and inference would *share*, so ``score_check()`` cannot
        see it: the fluctuation would solve the equation it was given, to machine
        precision, and report the curve that defines it.
        """
        wrong = _ey1_influence(missingness=np.ones_like(law.PI), centre=law.Q, psi=law.TRUTH["ey1"])
        assert np.max(np.abs(wrong - law.eif("ey1"))) > 1e-2

    def test_centring_the_plug_in_on_the_complete_cases_would_be_caught(self) -> None:
        """The plug-in averaged over ``P_n(W | Delta = 1)`` rather than ``P_n(W)``.

        This is the substantive missing-data mistake -- it is what a complete-case
        analysis *is* -- and on this law it is worth 0.115, because the mechanism depends
        on ``W`` and ``Qbar(1, .)`` moves the other way.
        """
        observed_only = float(law.observed_only_functional(law.PROBS, "ey1"))
        assert abs(observed_only - law.TRUTH["ey1"]) > 1e-2
        wrong = _ey1_influence(missingness=law.PI, centre=law.Q, psi=observed_only)
        assert np.max(np.abs(wrong - law.eif("ey1"))) > 1e-2

    def test_regressing_the_masked_outcome_would_be_caught(self) -> None:
        """``Qbar`` fitted on all rows with the missing outcomes filled in as zero.

        That regression estimates ``E[Delta * Y | A, W] = pi * Qbar``, not
        ``E[Y | A, W, Delta = 1]``.  The library avoids it by fitting the outcome model
        under ``fit_mask=data.observed`` while still predicting everywhere; this is what
        the alternative would have cost.
        """
        wrong = _ey1_influence(missingness=law.PI, centre=law.PI * law.Q, psi=law.TRUTH["ey1"])
        assert np.max(np.abs(wrong - law.eif("ey1"))) > 1e-2


class TestTheOutcomeRegressionUsesOnlyTheCompleteCases:
    """The one thing the oracle fit above is structurally blind to.

    Handing the estimator an oracle outcome model is what makes every assertion in this
    module exact, but it also means those assertions cannot see *which rows* the outcome
    model was trained on: an oracle ignores its training data by construction.  Deleting
    ``fit_mask=data.observed`` from :func:`~cleverly.estimators._nuisance.fit_nuisances`
    leaves all of them passing.

    The mistake it would let through is priced above -- regressing the zero-filled
    outcome on every row estimates ``E[Delta * Y | A, W] = pi * Qbar``, and the resulting
    influence curve is wrong by more than ``1e-2``.  So the restriction is worth
    asserting directly, with a learner that does look at its data.
    """

    LEARNER_KWARGS: ClassVar[dict[str, float]] = {"C": 1e6, "max_iter": 2000}

    @staticmethod
    def _reference(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``Qbar(1, .)`` and ``Qbar(0, .)`` from a regression fitted on ``rows``."""
        from sklearn.linear_model import LogisticRegression

        frame = law.frame()
        w = frame["W"].to_numpy(dtype=float).reshape(-1, 1)
        a = frame["A"].to_numpy(dtype=float).reshape(-1, 1)
        # The library zero-fills the outcome at unobserved rows before handing it to the
        # learner, so an all-rows fit sees zeros there -- which is the whole point.
        y = np.nan_to_num(frame["Y"].to_numpy(dtype=float))
        design = np.hstack([a, w])
        model = LogisticRegression(
            **TestTheOutcomeRegressionUsesOnlyTheCompleteCases.LEARNER_KWARGS
        )
        model.fit(design[rows], y[rows])
        at_one = model.predict_proba(np.hstack([np.ones_like(a), w]))[:, 1]
        at_zero = model.predict_proba(np.hstack([np.zeros_like(a), w]))[:, 1]
        return at_one, at_zero

    @pytest.fixture(scope="class")
    def learned_fit(self):
        from sklearn.linear_model import LogisticRegression

        dgp = law.DiscreteLaw()
        estimator = TMLE(
            outcome_learner=LogisticRegression(**self.LEARNER_KWARGS),
            treatment_learner=OracleTreatment(dgp),
            missingness_learner=OracleMissingness(dgp),
            cross_fit=False,
            estimands=("ate",),
            simultaneous=False,
            random_state=0,
        )
        return estimator.fit(
            law.frame(), outcome="Y", treatment="A", covariates=["W"], delta="Delta"
        )

    def test_it_matches_a_regression_fitted_on_the_observed_rows(self, learned_fit) -> None:
        observed = law.frame()["Delta"].to_numpy() == 1.0
        at_one, at_zero = self._reference(observed)
        np.testing.assert_allclose(learned_fit.nuisance.outcome.at_one, at_one, atol=1e-8, rtol=0)
        np.testing.assert_allclose(learned_fit.nuisance.outcome.at_zero, at_zero, atol=1e-8, rtol=0)

    def test_it_does_not_match_a_regression_fitted_on_every_row(self, learned_fit) -> None:
        # The negative half: the two fits have to be far enough apart that the assertion
        # above is discriminating and not merely satisfied by both.
        every = np.ones(law.N, dtype=bool)
        at_one, _ = self._reference(every)
        assert np.max(np.abs(learned_fit.nuisance.outcome.at_one - at_one)) > 1e-2
