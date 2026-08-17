r"""The multi-arm influence curves, against a numerically differentiated oracle.

The three-armed counterpart of :mod:`tests.unit.test_influence_gateaux`, and it carries
the same burden of proof: whatever the estimator reports as an influence curve has to
equal the Gateaux derivative of an independently written functional, computed by complex
step on a law the sample realises exactly, to ``1e-12``.

Why a *separate* law rather than more parameters on the binary one.  Two arms cannot
distinguish an implementation that genuinely keys everything by arm from one that has
two columns and happens to call them ``0`` and ``1``: every "arm 0 / arm 1" confusion,
every ``1 - g`` that should be ``g_a``, every design column that should be an indicator
block, is invisible at ``K = 2``.  Three arms is the smallest number where those become
distinguishable, and the labels here are strings that sort into a *different* order than
they were written in (``"low", "mid", "high"`` sorts to ``"high", "low", "mid"``), so a
helper that assumed arm codes and arm positions coincide fails rather than passes.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly.estimators import TMLE
from tests import discrete_law_multi as law

#: Oracle names, on the law's own arm indices.  ``law.reported_name`` maps each to the
#: name the library reports, which is in the analyst's labels.
ORACLE_NAMES = (
    "ey[0]",
    "ey[1]",
    "ey[2]",
    "ate[1 vs 0]",
    "ate[2 vs 0]",
    "att[1 vs 0]",
    "att[2 vs 0]",
    "atc[1 vs 0]",
    "atc[2 vs 0]",
)

#: The conditional effects alone, for the checks that are about *them* rather than about
#: every estimand: they are the ones with a second fluctuation and their own population.
CONDITIONAL_NAMES = ("att[1 vs 0]", "att[2 vs 0]", "atc[1 vs 0]", "atc[2 vs 0]")


@pytest.fixture(scope="module")
def exact_fit():
    """TMLE on the three-armed law with oracle nuisances, so ``P_n`` and ``Qbar`` are exact.

    ``cross_fit=False`` because there is nothing to cross-fit: the oracle does not learn
    from the data.  ``reference="low"`` so the contrasts are against arm 0 of the law's
    own indexing, which is what :func:`law.functional` writes ``ate[a vs 0]`` for.

    The conditional effects are asked for explicitly: they are defined at every arm count
    but stay out of a multi-arm default report, so that adding them moved no existing
    fit's report.  Three fluctuations come out of this one fit -- ``mean``, ``att``,
    ``atc`` -- which is the point: they are three score equations, not one.
    """
    estimator = TMLE(
        outcome_learner=law.OracleMultiOutcome(),
        treatment_learner=law.OracleMultiTreatment(),
        cross_fit=False,
        estimands=("ey", "ate", "att", "atc"),
        reference="low",
        simultaneous=False,
        random_state=0,
    )
    return estimator.fit(law.frame(), outcome="Y", treatment="A").single()


class TestTheSampleRealisesTheLaw:
    """The premises the rest of the module rests on, asserted rather than assumed."""

    def test_the_empirical_conditional_law_is_the_true_one(self) -> None:
        frame = law.frame(labelled=False)
        assert len(frame) == law.N
        for w in range(3):
            rows = frame["W"] == w
            for a in range(law.K):
                arm = rows & (frame["A"] == a)
                assert arm.sum() / rows.sum() == pytest.approx(law.G[w, a], abs=1e-15)
                assert frame.loc[arm, "Y"].mean() == pytest.approx(law.Q[w, a], abs=1e-15)

    def test_the_gateaux_derivative_has_mean_zero(self) -> None:
        # An influence function is centred by construction.  If this failed, the
        # numerical derivative -- not the library -- would be the thing that is wrong.
        for name in ORACLE_NAMES:
            assert float((law.PROBS.reshape(-1) * law.eif(name)).sum()) == pytest.approx(
                0.0, abs=1e-12
            )

    @pytest.mark.parametrize(
        ("group", "dimension"),
        [("mean", law.K), ("att", law.K - 1), ("atc", law.K - 1)],
    )
    def test_targeting_has_nothing_left_to_do(self, exact_fit, group: str, dimension: int) -> None:
        # The initial fit is exactly correct in the sample, so each fluctuation's score is
        # already zero at epsilon = 0 across all its columns.  This is what makes the
        # reported influence curve the EIF at P_0 rather than an estimate of it.
        #
        # The dimensions say what each group is: one column per *arm* for the means, one
        # per non-reference arm for the conditional effects, which is a contrast each.
        fluctuation = exact_fit.fluctuations[group]
        assert fluctuation.epsilon.shape == (dimension,)
        assert np.max(np.abs(fluctuation.epsilon)) == pytest.approx(0.0, abs=1e-12)

    def test_no_bound_binds(self, exact_fit) -> None:
        # The per-arm truncation is deliberately not renormalised, so a law where it bound
        # would make every comparison below a comparison against a different estimand.
        propensity = exact_fit.nuisance.propensity
        lower, upper = exact_fit.config.g_bounds
        assert float(np.min(propensity.values)) > lower
        assert float(np.max(propensity.values)) < upper
        assert exact_fit.sensitivity.positivity().simplex_deviation == pytest.approx(0.0)


class TestTheInfluenceCurveIsTheEIF:
    @pytest.mark.parametrize("oracle", ORACLE_NAMES)
    def test_matches_the_numerical_gateaux_derivative(self, exact_fit, oracle: str) -> None:
        estimate = exact_fit.estimates[law.reported_name(oracle)]
        reported = np.asarray(estimate.influence_curve)
        # One representative row per support point: the influence curve is constant
        # within a cell, because the law's support is finite and the fit is exact.
        cells = law.cell_of_row()
        per_cell = np.array(
            [reported[np.flatnonzero(cells == point)[0]] for point in range(len(law.SUPPORT))]
        )
        np.testing.assert_allclose(per_cell, law.eif(oracle), atol=1e-12, rtol=0)

    @pytest.mark.parametrize("oracle", ORACLE_NAMES)
    def test_the_point_estimate_is_the_functional(self, exact_fit, oracle: str) -> None:
        # P_n is P_0 and the fluctuation is null, so the plug-in is the target parameter.
        estimate = exact_fit.estimates[law.reported_name(oracle)]
        assert estimate.psi == pytest.approx(float(law.functional(law.PROBS, oracle)), abs=1e-12)

    def test_the_influence_curve_is_constant_within_a_cell(self, exact_fit) -> None:
        """The premise of the comparison above, checked rather than assumed.

        The law has finite support and the fit is exact, so two rows with the same
        ``(W, A, Y)`` must receive the same influence-curve value.  If they did not, the
        one representative row picked per cell would be an arbitrary choice.
        """
        cells = law.cell_of_row()
        for oracle in ORACLE_NAMES:
            values = np.asarray(exact_fit.estimates[law.reported_name(oracle)].influence_curve)
            for point in range(len(law.SUPPORT)):
                rows = values[cells == point]
                assert np.ptp(rows) == pytest.approx(0.0, abs=1e-15)

    def test_a_wrong_arm_denominator_would_be_caught(self) -> None:
        """The negative control: the comparison has teeth, and it is arm-specific.

        Written out at the support points, the EIF of ``E[Y(a)]`` is
        ``1{A=a}/g(a|w) * (y - Qbar(a, w)) + Qbar(a, w) - psi``.  Using the *wrong arm's*
        propensity in that denominator -- the error a two-armed implementation makes when
        it reaches for ``1 - g`` -- has to move it far outside the ``1e-12`` window.
        """
        truth = float(law.functional(law.PROBS, "ey[1]"))

        def hand_written(denominator_arm: int) -> np.ndarray:
            values = []
            for w, a, y in law.SUPPORT:
                clever = (a == 1) / law.G[w, denominator_arm]
                values.append(clever * (y - law.Q[w, 1]) + law.Q[w, 1] - truth)
            return np.array(values)

        np.testing.assert_allclose(hand_written(1), law.eif("ey[1]"), atol=1e-12, rtol=0)
        for wrong in (0, 2):
            assert np.max(np.abs(hand_written(wrong) - law.eif("ey[1]"))) > 1e-2

    def test_an_inverted_odds_or_a_shared_population_would_be_caught(self) -> None:
        """The negative controls for the conditional effects, at three arms.

        Written out at the support points, the EIF of ``E[Y(a) - Y(r) | A = a]`` is

            h_a(A, w) (y - Qbar(A, w)) + 1{A = a}/P(A = a) (Qbar(a, w) - Qbar(r, w) - psi)

        with ``h_a = 1{A=a}/P(A=a) - 1{A=r} (g_a/g_r)/P(A=a)``.  Two ways to build that
        wrong survive a two-armed law and not this one: inverting the odds to ``g_r/g_a``
        -- which at ``K = 2`` merely rescales a single column, and here mixes the arms --
        and giving every contrast the *reference* arm's conditioning share, which is the
        ATC's population rather than the ATT's and is identical when there is one of each.
        """
        arm, reference = 1, 0
        truth = float(law.functional(law.PROBS, "att[1 vs 0]"))
        p_wa = law.PROBS.sum(axis=2)
        share = p_wa[:, arm].sum()
        share_reference = p_wa[:, reference].sum()

        def hand_written(*, invert_odds: bool = False, wrong_share: bool = False) -> np.ndarray:
            values = []
            for w, a, y in law.SUPPORT:
                odds = (
                    law.G[w, reference] / law.G[w, arm]
                    if invert_odds
                    else law.G[w, arm] / law.G[w, reference]
                )
                denominator = share_reference if wrong_share else share
                clever = ((a == arm) - (a == reference) * odds) / denominator
                contrast = law.Q[w, arm] - law.Q[w, reference]
                values.append(
                    clever * (y - law.Q[w, a]) + (a == arm) / denominator * (contrast - truth)
                )
            return np.array(values)

        np.testing.assert_allclose(hand_written(), law.eif("att[1 vs 0]"), atol=1e-12, rtol=0)
        for broken in (
            hand_written(invert_odds=True),
            hand_written(wrong_share=True),
        ):
            assert np.max(np.abs(broken - law.eif("att[1 vs 0]"))) > 1e-2

    def test_the_att_and_atc_are_different_parameters_here(self) -> None:
        """A premise of the two tests above: the law separates the populations.

        On a law where the effect happened to be constant every conditional effect would
        equal the ATE and equal each other, and the comparisons would pass on that
        coincidence rather than on the arithmetic.
        """
        values = [float(law.functional(law.PROBS, name)) for name in CONDITIONAL_NAMES]
        ate = [float(law.functional(law.PROBS, f"ate[{a} vs 0]")) for a in (1, 2)]
        assert len(set(np.round(values + ate, 6))) == len(values) + len(ate)


class TestArmsAreKeyedNotCounted:
    """Checks that only fail if arms are addressed by *code* rather than by position."""

    def test_the_reported_labels_are_the_analysts_own(self, exact_fit) -> None:
        assert set(exact_fit.estimates) == {
            "ey[low]",
            "ey[mid]",
            "ey[high]",
            "ate[mid vs low]",
            "ate[high vs low]",
            "att[mid vs low]",
            "att[high vs low]",
            "atc[mid vs low]",
            "atc[high vs low]",
        }

    def test_a_conditional_effect_averages_over_its_own_arm(self, exact_fit) -> None:
        """The plug-in, read off the targeted predictions rather than off the report.

        Cheap, exact, and it fails if the conditioning population were the whole sample
        (which would make it the ATE) or the reference arm's (which would make the ATT the
        ATC).  The Gateaux comparison covers the curve; this covers the number.
        """
        targeted = exact_fit.fluctuations["att"].targeted
        treatment = exact_fit.data.treatment
        reference = exact_fit.config.reference_arm
        for arm in exact_fit.data.arm_codes:
            if arm == reference:
                continue
            contrast = targeted.arms[arm] - targeted.arms[reference]
            label = exact_fit.data.arm_label(arm)
            expected = float(np.mean(contrast[treatment == arm]))
            reported = exact_fit.estimates[f"att[{label} vs {exact_fit.data.arm_label(reference)}]"]
            assert reported.psi == pytest.approx(expected, abs=1e-12)

    def test_the_labels_sort_differently_from_how_they_were_written(self) -> None:
        # The property that gives this module its teeth: arm *code* 0 is "high", not
        # "low", so any helper equating code with position reports the wrong arm.
        assert law.SORTED_LABELS == ("high", "low", "mid")
        assert law.ARM_OF_CODE != (0, 1, 2)

    def test_the_contrast_is_the_difference_of_the_two_means_exactly(self, exact_fit) -> None:
        """``IC_ate == IC_ey[a] - IC_ey[ref]`` to the last bit, not approximately.

        Cheap, deterministic, and it fails immediately if the contrast were rebuilt from
        the targeted predictions instead of from the means it is a contrast of.
        """
        reference = exact_fit.estimates["ey[low]"]
        for arm in ("mid", "high"):
            contrast = exact_fit.estimates[f"ate[{arm} vs low]"]
            expected = exact_fit.estimates[f"ey[{arm}]"].influence_curve - reference.influence_curve
            np.testing.assert_array_equal(contrast.influence_curve, expected)

    def test_the_delta_method_reproduces_a_named_contrast(self, exact_fit) -> None:
        """Any contrast beyond the reported ones comes free from the joint influence curve.

        This is the pay-off of reporting ``K`` means with a joint covariance: ``mid``
        against ``high`` was never estimated as a named parameter, and the delta method
        recovers it from the two means.
        """
        derived = exact_fit.contrast(lambda psi: psi[0] - psi[1], ["ey[mid]", "ey[high]"])
        expected = float(law.functional(law.PROBS, "ate[1 vs 2]"))
        assert derived.psi == pytest.approx(expected, abs=1e-12)
        np.testing.assert_allclose(
            np.asarray(derived.influence_curve),
            np.asarray(exact_fit.estimates["ey[mid]"].influence_curve)
            - np.asarray(exact_fit.estimates["ey[high]"].influence_curve),
            atol=1e-9,
            rtol=0,
        )
