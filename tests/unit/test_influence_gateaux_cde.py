r"""Is the influence curve right when ``Z`` is *intervened on*?

:mod:`tests.unit.test_influence_gateaux` establishes that the library's influence curve is
the efficient influence function, from the definition, on a law it can hold in its hand;
:mod:`tests.unit.test_influence_gateaux_mar` extends that to missing outcomes.  Neither
reaches the ``intermediate=`` path, and that is the path where the gap mattered most.

:mod:`cleverly.estimators.direct_effect` derives the controlled direct effect's influence
function by argument: a longitudinal parameter with two intervention nodes would carry a
third, sequential-regression term, and that term is claimed to vanish identically because
nothing is measured between :math:`A` and :math:`Z`.  The claim is exactly the kind that
cannot be checked by simulation.  A two-term influence function that *should* have had
three terms still solves its own score equation to machine precision, still passes
``score_check()``, and -- because the estimator would use the same wrong covariate for
targeting and for inference -- still produces confident, self-consistent, wrong intervals.
Until this module existed, the library said so about itself.

The argument is made here on :mod:`tests.discrete_law_cde`, whose support is the
observed-data support ``(w, a, z, k)``.  The Gateaux derivative of the identification
formula at each of its thirty-six points is the efficient influence function of the
observed-data model, computed by complex step from a longhand statement of :math:`\Psi`
that shares no code with the library.  Whatever the estimator reports has to equal it, at
``1e-12``.

Thirty of those thirty-six points carry no residual at all: the residual lives only where
the row was in the targeted arm, took the targeted level, *and* had its outcome recorded.
Reading that off the reported curve is the sharpest single statement that all three
indicators sit where the derivation puts them, and it is checked here as its own claim.

The negative controls at the bottom are the point of the module as much as the assertions
are.  Each takes one of the ways a controlled direct effect is plausibly built wrong --
omit the :math:`q_z` factor, so the estimator is quietly computing a total effect; use the
*other* level's density, the polarity mistake this package has already made once in its
diagnostics; use the marginal :math:`P(Z = z)` where the conditional :math:`q_z(a, W)` is
needed; average the plug-in over the ``Z = z`` stratum rather than over everybody -- and
shows it moves the answer by far more than the window the real assertions use.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly.estimators import TMLE
from tests import discrete_law_cde as law
from tests.conftest import (
    OracleDirectOutcome,
    OracleIntermediate,
    OracleMissingness,
    OracleTreatment,
)

ESTIMANDS = ("ey1", "ey0", "ate", "att", "atc", "rr", "or")
CASES = [(name, level) for level in law.LEVELS for name in ESTIMANDS]


@pytest.fixture(scope="module")
def exact_fit():
    """TMLE on the CDE law with all four oracle nuisances, one result per level.

    ``cross_fit=False`` because there is nothing to cross-fit: the oracles do not learn
    from the data, and out-of-fold prediction would only add fold bookkeeping to a fit
    whose answer is already determined.
    """
    dgp = law.DiscreteLaw()
    estimator = TMLE(
        outcome_learner=OracleDirectOutcome(dgp),
        treatment_learner=OracleTreatment(dgp),
        missingness_learner=OracleMissingness(dgp),
        intermediate_learner=OracleIntermediate(dgp),
        cross_fit=False,
        estimands="all",
        simultaneous=False,
        random_state=0,
    )
    return estimator.fit(
        law.frame(),
        outcome="Y",
        treatment="A",
        covariates=["W"],
        delta="Delta",
        intermediate="Z",
    )


class TestTheSampleRealisesTheLaw:
    """The premises the rest of the module rests on, asserted rather than assumed."""

    def test_the_empirical_conditional_law_is_the_true_one(self) -> None:
        frame = law.frame()
        assert len(frame) == law.N
        # Roughly two rows in five have no outcome, so this is not a token amount of
        # missingness the estimator could ignore and still look right.
        assert frame["Delta"].mean() == pytest.approx(0.578125, abs=1e-15)
        for w in range(3):
            rows = frame["W"] == w
            assert frame.loc[rows, "A"].mean() == pytest.approx(law.G[w], abs=1e-15)
            for a in range(2):
                arm = rows & (frame["A"] == a)
                assert frame.loc[arm, "Z"].mean() == pytest.approx(law.QZ[w, a], abs=1e-15)
                assert frame.loc[arm, "Delta"].mean() == pytest.approx(law.PI[w, a], abs=1e-15)
                for z in law.LEVELS:
                    cell = arm & (frame["Z"] == z) & (frame["Delta"] == 1.0)
                    assert frame.loc[cell, "Y"].mean() == pytest.approx(
                        law.QBAR[w, a, z], abs=1e-15
                    )

    def test_the_missingness_mechanism_does_not_depend_on_the_intermediate(self) -> None:
        # Assumption 5 of the derivation, encoded in the law rather than assumed of it.
        # If it failed here, the estimator's missingness model -- which conditions on
        # (A, W) only -- would be misspecified and the oracle would not be an oracle.
        frame = law.frame()
        for w in range(3):
            for a in range(2):
                arm = (frame["W"] == w) & (frame["A"] == a)
                at_zero = frame.loc[arm & (frame["Z"] == 0), "Delta"].mean()
                at_one = frame.loc[arm & (frame["Z"] == 1), "Delta"].mean()
                assert at_zero == pytest.approx(at_one, abs=1e-15)

    def test_the_intermediate_mechanism_depends_on_both_arm_and_covariate(self) -> None:
        # Both dependencies are load-bearing.  Without the covariate dependence,
        # intervening on Z would not reweight anything and the Z-stratified control would
        # have nothing to detect; without the arm dependence the two columns of the
        # intermediate nuisance would be interchangeable.
        assert np.max(np.abs(law.QZ[:, 1] - law.QZ[:, 0])) >= 0.5
        assert law.QZ[:, 0].max() - law.QZ[:, 0].min() >= 0.5

    def test_the_outcome_carries_a_genuine_interaction(self) -> None:
        # Without an A-by-Z interaction the controlled direct effect is the same parameter
        # at both levels, and every level-specific claim below would pass vacuously.
        interaction = (
            law.QBAR[:, 1, 1] - law.QBAR[:, 1, 0] - (law.QBAR[:, 0, 1] - law.QBAR[:, 0, 0])
        )
        assert np.max(np.abs(interaction)) >= 0.5

    @pytest.mark.parametrize(("name", "level"), CASES)
    def test_the_gateaux_derivative_has_mean_zero(self, name: str, level: int) -> None:
        # An influence function is centred by construction.  If this failed, the numerical
        # derivative -- not the library -- would be the thing that is wrong.
        centred = float((law.PROBS.reshape(-1) * law.eif(name, level)).sum())
        assert centred == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("level", law.LEVELS)
    def test_targeting_has_nothing_left_to_do(self, exact_fit, level: int) -> None:
        # Within a (w, a, z) cell the clever covariate is constant and the observed
        # outcomes average to exactly Qbar(a, z, w), so the score is already zero at
        # epsilon = 0.  This is what makes the reported influence curve the EIF at P_0
        # rather than an estimate of it.
        for fluctuation in exact_fit[float(level)].fluctuations.values():
            assert np.max(np.abs(fluctuation.epsilon)) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("level", law.LEVELS)
    def test_no_bound_binds(self, exact_fit, level: int) -> None:
        # The law is built so that g, q_z, pi and Qbar all sit well inside their truncation
        # windows.  If one bound started to bite, the estimator would be solving a
        # different score equation and the assertions below would be testing that.
        nuisance = exact_fit[float(level)].nuisance
        assert float(np.min(nuisance.propensity.arm(1.0))) > 0.2
        assert float(np.max(nuisance.propensity.arm(1.0))) < 0.8
        assert nuisance.missingness is not None
        assert nuisance.intermediate is not None
        bound = exact_fit[float(level)].config.missingness_bound
        assert float(np.min(nuisance.missingness)) > 10.0 * bound
        density = nuisance.intermediate_density(float(level), bound)
        assert density is not None
        assert float(np.min(density)) > 10.0 * bound


class TestTheInfluenceCurveIsTheEIF:
    """The claim :mod:`cleverly.estimators.direct_effect` could previously only argue."""

    @pytest.mark.parametrize(("name", "level"), CASES)
    def test_matches_the_numerical_gateaux_derivative(
        self, exact_fit, name: str, level: int
    ) -> None:
        estimate = exact_fit[float(level)].estimates[name]
        reported = np.asarray(estimate.influence_curve)[law.first_row_of()]
        np.testing.assert_allclose(reported, law.eif(name, level), atol=1e-12, rtol=0)

    @pytest.mark.parametrize(("name", "level"), CASES)
    def test_the_point_estimate_is_the_functional(self, exact_fit, name: str, level: int) -> None:
        estimate = exact_fit[float(level)].estimates[name]
        psi = estimate.log_psi if estimate.scale == "ratio" else estimate.psi
        assert psi == pytest.approx(law.TRUTH[level][name], abs=1e-12)

    @pytest.mark.parametrize(("name", "level"), CASES)
    def test_the_untargeted_rows_carry_no_residual_term(
        self, exact_fit, name: str, level: int
    ) -> None:
        """Only the targeted arm, at the targeted level, with an outcome, has a residual.

        Thirty of the thirty-six support points fail at least one of those three
        conditions, and at every one of them the influence curve has to be the plug-in
        term alone -- exactly zero residual, not a small one.  This is the single
        assertion that places all three indicators at once: drop the ``Z`` indicator and
        the rows at the other level acquire a residual; drop ``Delta`` and the unobserved
        rows do.
        """
        reported = np.asarray(exact_fit[float(level)].estimates[name].influence_curve)
        reported = reported[law.first_row_of()]
        quiet = [
            i for i, (_, _, z, k) in enumerate(law.SUPPORT) if k == law.UNOBSERVED or z != level
        ]
        np.testing.assert_allclose(reported[quiet], law.eif(name, level)[quiet], atol=1e-12, rtol=0)
        if name in ("ey1", "ey0"):
            arm = 1 if name == "ey1" else 0
            longhand = [
                law.QBAR[law.SUPPORT[i][0], arm, level] - law.TRUTH[level][name] for i in quiet
            ]
            np.testing.assert_allclose(reported[quiet], longhand, atol=1e-12, rtol=0)

    @pytest.mark.parametrize("level", law.LEVELS)
    def test_the_ate_influence_curve_is_the_difference_of_the_two_means(
        self, exact_fit, level: int
    ) -> None:
        estimates = exact_fit[float(level)].estimates
        ate = np.asarray(estimates["ate"].influence_curve)
        one = np.asarray(estimates["ey1"].influence_curve)
        zero = np.asarray(estimates["ey0"].influence_curve)
        np.testing.assert_allclose(ate, one - zero, atol=1e-12, rtol=0)

    @pytest.mark.parametrize("level", law.LEVELS)
    def test_the_score_equation_is_solved(self, exact_fit, level: int) -> None:
        assert exact_fit[float(level)].validation.score_check().passed


class TestTheTwoLevelsAreDifferentParameters:
    """ "A controlled direct effect is a different parameter for each ``z``."

    :mod:`cleverly.estimators.direct_effect` warns that an additive learner produces an
    identical *initial* contrast at both levels, leaving all the ``z``-dependence to the
    targeting step.  On this law the two levels do not merely differ, they have opposite
    signs -- so an estimator that confused them would not be slightly off, it would report
    a harm as a benefit.
    """

    def test_the_controlled_direct_effects_have_opposite_signs(self, exact_fit) -> None:
        at_zero = exact_fit[0.0].estimates["ate"].psi
        at_one = exact_fit[1.0].estimates["ate"].psi
        assert at_zero == pytest.approx(-0.1875, abs=1e-12)
        assert at_one == pytest.approx(0.3125, abs=1e-12)
        assert at_zero < 0.0 < at_one

    @pytest.mark.parametrize("level", law.LEVELS)
    def test_neither_level_is_the_total_effect(self, exact_fit, level: int) -> None:
        """Intervening on ``Z`` is not the same as marginalising it away.

        The total effect is what an estimator that dropped the ``q_z`` factor and ignored
        the level when predicting the outcome would converge to.  It is a perfectly
        respectable parameter -- it is just not the one that was asked for.
        """
        reported = exact_fit[float(level)].estimates["ate"].psi
        assert abs(reported - law.TOTAL_EFFECT["ate"]) > 0.2


class TestTheContrastsSeparate:
    """``att > ate > atc``, strictly, at both levels.

    A constant-effect law cannot distinguish a correct ``att`` from one that conditions on
    the wrong arm or inverts the propensity-odds factor: every contrast is the same number,
    so every arrangement of them passes.  Here :math:`\\tau_z(w)` is correlated with
    ``g(w)``, the treated are drawn disproportionately from the covariate values with the
    larger effect, and the three contrasts separate by at least 0.07.
    """

    @pytest.mark.parametrize("level", law.LEVELS)
    def test_the_truth_is_strictly_ordered(self, level: int) -> None:
        att, ate, atc = law.ordering(level)
        assert att > ate > atc
        assert min(att - ate, ate - atc) > 0.07

    @pytest.mark.parametrize("level", law.LEVELS)
    def test_the_estimator_recovers_the_ordering(self, exact_fit, level: int) -> None:
        estimates = exact_fit[float(level)].estimates
        att, ate, atc = (estimates[name].psi for name in ("att", "ate", "atc"))
        assert att > ate > atc
        expected = law.ordering(level)
        np.testing.assert_allclose([att, ate, atc], expected, atol=1e-12, rtol=0)

    @pytest.mark.parametrize("level", law.LEVELS)
    def test_the_ate_lies_between_the_two_conditional_effects(self, exact_fit, level: int) -> None:
        # ATE is a P(A = 1)-weighted average of ATT and ATC, so this is an identity rather
        # than a coincidence -- and it fails immediately if the two shares are swapped.
        estimates = exact_fit[float(level)].estimates
        share = float(law.frame()["A"].mean())
        blended = share * estimates["att"].psi + (1.0 - share) * estimates["atc"].psi
        assert blended == pytest.approx(estimates["ate"].psi, abs=1e-12)


def _ey1_influence(
    level: int,
    *,
    density: np.ndarray | None = None,
    missingness: np.ndarray | None = None,
    centre: np.ndarray | None = None,
    psi: float | None = None,
) -> np.ndarray:
    r"""``Psi_{1,z}``'s influence curve at the support points, written out by hand.

    .. math::

        \frac{\mathbb 1\{a = 1\}\,\mathbb 1\{z' = z\}\,\Delta}
             {g(w)\, q_z(1, w)\, \pi(1, w)}
        \bigl(y - \bar Q(1, z, w)\bigr)
        + \bar Q(1, z, w) - \Psi_{1,z}

    Parameterised by the four things a wrong implementation would get wrong -- the
    intermediate density in the denominator, the observation probability beside it, the
    regression the residual is taken against, and the value the plug-in is centred at --
    so that each can be perturbed on its own below.
    """
    dens = density if density is not None else (law.QZ if level == 1 else 1.0 - law.QZ)
    miss = missingness if missingness is not None else law.PI
    cen = centre if centre is not None else law.QBAR[:, :, level]
    base = law.TRUTH[level]["ey1"] if psi is None else psi
    values = []
    for w, a, z, k in law.SUPPORT:
        plug = cen[w, 1] - base
        if k == law.UNOBSERVED or z != level or a != 1:
            values.append(plug)
            continue
        clever = 1.0 / (law.G[w] * dens[w, 1] * miss[w, 1])
        values.append(clever * (float(k) - cen[w, 1]) + plug)
    return np.array(values)


class TestAWrongConstructionWouldBeCaught:
    """The negative controls: the assertions above have teeth only if these hold."""

    @pytest.mark.parametrize("level", law.LEVELS)
    def test_the_longhand_influence_curve_reproduces_the_derivative(self, level: int) -> None:
        # The baseline the four controls are perturbations of.  If this drifted, the
        # controls would be measuring the drift rather than the mistake.
        np.testing.assert_allclose(_ey1_influence(level), law.eif("ey1", level), atol=1e-12, rtol=0)

    @pytest.mark.parametrize("level", law.LEVELS)
    def test_dropping_the_intermediate_density_would_be_caught(self, level: int) -> None:
        """``1/q_z`` omitted from the clever covariate.

        This is the mistake that turns a controlled direct effect into something close to
        a total effect, and it is the one the two-term derivation is most easily misread
        into: the covariate looks exactly like the ordinary point-treatment one.  Targeting
        and inference would share it, so ``score_check()`` cannot see it.
        """
        wrong = _ey1_influence(level, density=np.ones_like(law.QZ))
        assert np.max(np.abs(wrong - law.eif("ey1", level))) > 1e-2

    @pytest.mark.parametrize("level", law.LEVELS)
    def test_using_the_other_levels_density_would_be_caught(self, level: int) -> None:
        """``q_{1-z}`` where ``q_z`` is needed -- the polarity mistake.

        Not hypothetical: this package shipped exactly this inversion in three of its
        controlled-direct-effect diagnostics, and
        :func:`cleverly.estimators.direct_effect.check_level` exists because an
        unrecognised level was silently treated as ``z = 0``.  Here the same slip is
        checked where it would do real damage -- in the estimating equation.
        """
        other = law.QZ if level == 0 else 1.0 - law.QZ
        wrong = _ey1_influence(level, density=other)
        assert np.max(np.abs(wrong - law.eif("ey1", level))) > 1e-2

    @pytest.mark.parametrize("level", law.LEVELS)
    def test_using_the_marginal_probability_of_the_level_would_be_caught(self, level: int) -> None:
        """``P(Z = z)`` in place of ``q_z(a, W)``.

        The plausible shortcut when the intermediate mechanism looks weakly related to the
        covariates.  It is not a bias-free simplification: the density is a
        Radon--Nikodym derivative, and replacing it with its average reweights the wrong
        rows.
        """
        marginal = float(law.PROBS[:, :, level, :].sum())
        wrong = _ey1_influence(level, density=np.full_like(law.QZ, marginal))
        assert np.max(np.abs(wrong - law.eif("ey1", level))) > 1e-2

    @pytest.mark.parametrize("level", law.LEVELS)
    def test_averaging_the_plug_in_over_the_stratum_would_be_caught(self, level: int) -> None:
        """The plug-in averaged over ``P_n(W | Z = z)`` rather than ``P_n(W)``.

        Conditioning on the subpopulation that happened to receive ``Z = z`` instead of
        intervening to set it -- the substantive mistake a controlled direct effect exists
        to avoid, and the one that survives every internal consistency check.
        """
        stratified = float(law.z_stratified_functional(law.PROBS, "ey1", level))
        assert abs(stratified - law.TRUTH[level]["ey1"]) > 1e-2
        wrong = _ey1_influence(level, psi=stratified)
        assert np.max(np.abs(wrong - law.eif("ey1", level))) > 1e-2

    @pytest.mark.parametrize("level", law.LEVELS)
    def test_dropping_the_observation_probability_would_be_caught(self, level: int) -> None:
        """``1/pi`` omitted, leaving a two-way product where a three-way one belongs.

        The reason this law keeps its ``Delta`` dimension: with ``pi = 1`` everywhere the
        controlled direct effect's clever covariate would be indistinguishable from a
        two-nuisance one, and the double-robustness statement in
        :mod:`cleverly.estimators.direct_effect` -- ``Qbar`` right *or* the product
        ``g q_z pi`` right -- would be untested in its third factor.
        """
        wrong = _ey1_influence(level, missingness=np.ones_like(law.PI))
        assert np.max(np.abs(wrong - law.eif("ey1", level))) > 1e-2
