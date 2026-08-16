"""The machinery a third treatment arm exercises, checked exactly rather than statistically.

:mod:`tests.unit.test_influence_gateaux_multi` establishes that the multi-arm influence
curves are the right functions.  This module covers the pieces around them -- the clever
covariate, the score equation, the fold splits, the truncation policy -- with checks that
are deterministic and cost milliseconds, in preference to simulation studies that would
cost minutes and fail on a bad seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleverly import TMLE
from cleverly.data import CausalData
from cleverly.estimators._nuisance import Propensity, UnfittedPropensity
from cleverly.exceptions import DataError
from cleverly.fluctuation.submodel import atc_submodel, att_submodel, mean_submodel
from cleverly.learners.crossfit import make_folds
from tests import discrete_law_multi as law


def _three_arm_frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n, 2))
    linear = np.column_stack([np.zeros(n), 0.8 * w[:, 0], -0.7 * w[:, 1]])
    probability = np.exp(linear)
    probability /= probability.sum(axis=1, keepdims=True)
    arm = np.array([rng.choice(3, p=row) for row in probability], dtype=float)
    outcome = 0.5 * arm + w[:, 0] - 0.3 * w[:, 1] + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"W1": w[:, 0], "W2": w[:, 1], "A": arm, "Y": outcome})


class TestTheCleverCovariate:
    def test_one_column_per_arm_with_disjoint_support(self) -> None:
        treatment = np.array([0.0, 1.0, 2.0, 2.0, 1.0, 0.0])
        propensity = np.tile([0.5, 0.3, 0.2], (6, 1))
        submodel = mean_submodel(treatment, propensity, arms=(0.0, 1.0, 2.0))

        assert submodel.dim == 3
        assert submodel.names == ("h0", "h1", "h2")
        assert submodel.arm_columns == {0.0: 0, 1.0: 1, 2.0: 2}
        # Each row is in exactly one arm, so exactly one column is non-zero.
        assert np.array_equal((submodel.observed != 0).sum(axis=1), np.ones(6))
        for position, arm in enumerate((0.0, 1.0, 2.0)):
            expected = (treatment == arm) / propensity[:, position]
            np.testing.assert_allclose(submodel.observed[:, position], expected)

    def test_the_conditional_covariate_is_one_odds_column_per_non_reference_arm(self) -> None:
        """``att`` at three arms, written out against the formula rather than the code.

        Two things here that two arms cannot show: the reference arm loads *every*
        column -- it is the arm each contrast is taken against, so it belongs to none of
        them and appears in all -- and each contrast carries its own conditioning share.
        """
        treatment = np.array([0.0, 1.0, 2.0, 2.0, 1.0, 0.0])
        propensity = np.tile([0.5, 0.3, 0.2], (6, 1))
        shares = np.array([1 / 3, 1 / 3, 1 / 3])
        submodel = att_submodel(
            treatment,
            propensity,
            arms=(0.0, 1.0, 2.0),
            arm_fractions=shares,
            reference=0.0,
        )

        assert submodel.dim == 2
        assert submodel.names == ("h_att[1]", "h_att[2]")
        assert submodel.arm_columns == {}
        assert submodel.contrast_columns == {1.0: 0, 2.0: 1}
        for column, arm in enumerate((1.0, 2.0)):
            odds = propensity[:, int(arm)] / propensity[:, 0]
            expected = (treatment == arm) / shares[int(arm)] - (treatment == 0.0) * odds / shares[
                int(arm)
            ]
            np.testing.assert_allclose(submodel.observed[:, column], expected)
            # The conditioning arm needs no reweighting; the reference is reweighted to
            # resemble it, in that contrast's column and nowhere else.
            np.testing.assert_allclose(submodel.arms[arm][:, column], 1.0 / shares[int(arm)])
            assert np.count_nonzero(submodel.arms[arm][:, 1 - column]) == 0
            np.testing.assert_allclose(submodel.arms[0.0][:, column], -odds / shares[int(arm)])

    def test_the_atc_conditions_every_column_on_the_reference_arm(self) -> None:
        """The mirror: one population, ``A = ref``, shared by every contrast."""
        treatment = np.array([0.0, 1.0, 2.0, 2.0, 1.0, 0.0])
        propensity = np.tile([0.5, 0.3, 0.2], (6, 1))
        shares = np.array([0.5, 0.2, 0.3])
        submodel = atc_submodel(
            treatment, propensity, arms=(0.0, 1.0, 2.0), arm_fractions=shares, reference=0.0
        )

        assert submodel.contrast_columns == {1.0: 0, 2.0: 1}
        for column, arm in enumerate((1.0, 2.0)):
            odds = propensity[:, 0] / propensity[:, int(arm)]
            expected = ((treatment == arm) * odds - (treatment == 0.0)) / shares[0]
            np.testing.assert_allclose(submodel.observed[:, column], expected)
            np.testing.assert_allclose(submodel.arms[0.0][:, column], -1.0 / shares[0])

    def test_the_reference_is_the_one_the_fit_declared(self) -> None:
        """Not the lowest arm when ``reference=`` says otherwise."""
        treatment = np.array([0.0, 1.0, 2.0])
        propensity = np.tile([0.5, 0.3, 0.2], (3, 1))
        submodel = att_submodel(
            treatment,
            propensity,
            arms=(0.0, 1.0, 2.0),
            arm_fractions=np.array([1 / 3, 1 / 3, 1 / 3]),
            reference=2.0,
        )
        assert submodel.contrast_columns == {0.0: 0, 1.0: 1}
        # Arm 2 is now the one every column reweights, and it has no contrast of its own.
        assert np.count_nonzero(submodel.arms[2.0]) == submodel.arms[2.0].size

    def test_an_arms_covariate_is_its_inverse_propensity_in_its_own_column(self) -> None:
        propensity = np.tile([0.5, 0.3, 0.2], (4, 1))
        submodel = mean_submodel(np.zeros(4), propensity, arms=(0.0, 1.0, 2.0))
        for position, arm in enumerate((0.0, 1.0, 2.0)):
            block = submodel.arms[arm]
            assert block.shape == (4, 3)
            np.testing.assert_allclose(block[:, position], 1.0 / propensity[:, position])
            # Every other column is zero: arm a's covariate targets arm a alone.
            others = [j for j in range(3) if j != position]
            assert np.count_nonzero(block[:, others]) == 0

    def test_two_arms_are_unchanged_by_the_general_form(self) -> None:
        """The K-column builder must reproduce the two-column one bit for bit."""
        rng = np.random.default_rng(0)
        treatment = (rng.random(200) < 0.5).astype(float)
        g1 = 0.2 + 0.6 * rng.random(200)

        from_vector = mean_submodel(treatment, g1)
        from_matrix = mean_submodel(treatment, np.column_stack([1.0 - g1, g1]), arms=(0.0, 1.0))
        assert from_vector.observed.tobytes() == from_matrix.observed.tobytes()
        for arm in (0.0, 1.0):
            assert from_vector.arms[arm].tobytes() == from_matrix.arms[arm].tobytes()

    def test_a_vector_propensity_is_refused_when_there_are_more_than_two_arms(self) -> None:
        with pytest.raises(ValueError, match=r"supply the \(n, K\) mechanism"):
            mean_submodel(np.zeros(4), np.full(4, 0.3), arms=(0.0, 1.0, 2.0))


class TestTheScoreEquation:
    def test_targeting_solves_every_column(self) -> None:
        """``P_n D*_a = 0`` for all K arms at once, not just the first two.

        The exact check the plug-in rests on: if only some columns were solved, the
        remaining arms' estimates would not be asymptotically linear, and no simulation
        would say which.
        """
        result = (
            TMLE(
                outcome_learner="glm",
                treatment_learner="glm",
                n_folds=5,
                random_state=0,
                simultaneous=False,
                estimands=("ey", "ate"),
            )
            .fit(_three_arm_frame(), outcome="Y", treatment="A")
            .single()
        )
        fluctuation = result.fluctuations["mean"]
        assert fluctuation.epsilon.shape == (3,)
        assert fluctuation.score.shape == (3,)
        assert fluctuation.converged
        # Relative to the scale each column's score could have taken.
        np.testing.assert_allclose(
            fluctuation.score / fluctuation.score_scale, np.zeros(3), atol=1e-8
        )

    def test_the_score_check_reports_one_z_per_arm(self) -> None:
        result = (
            TMLE(
                outcome_learner="glm",
                treatment_learner="glm",
                n_folds=5,
                random_state=0,
                simultaneous=False,
                estimands=("ey",),
            )
            .fit(_three_arm_frame(), outcome="Y", treatment="A")
            .single()
        )
        check = result.validation.score_check()
        assert check.passed
        assert len(result.fluctuations["mean"].score) == 3


class TestFoldsAndLeakage:
    def test_stratified_folds_keep_every_arm_in_every_fold(self) -> None:
        rng = np.random.default_rng(0)
        treatment = rng.integers(0, 3, 300).astype(float)
        folds = make_folds(300, 5, stratify=treatment, random_state=0)
        for _, test in folds:
            assert set(np.unique(treatment[test])) == {0.0, 1.0, 2.0}

    def test_every_row_gets_exactly_one_out_of_fold_prediction(self) -> None:
        rng = np.random.default_rng(1)
        treatment = rng.integers(0, 3, 300).astype(float)
        folds = make_folds(300, 5, stratify=treatment, random_state=0)
        seen = np.concatenate([test for _, test in folds])
        assert np.array_equal(np.sort(seen), np.arange(300))

    def test_training_and_validation_rows_are_disjoint(self) -> None:
        rng = np.random.default_rng(2)
        treatment = rng.integers(0, 3, 300).astype(float)
        for train, test in make_folds(300, 5, stratify=treatment, random_state=0):
            assert set(train.tolist()).isdisjoint(test.tolist())


class TestTruncation:
    def test_two_arms_keep_the_complement_form(self) -> None:
        """Asymmetric bounds must bound ``g1`` and complement, not clip both columns.

        The two are different, and the complement form is what every binary regression
        fixture was built against.
        """
        g1 = np.array([0.005, 0.2, 0.5, 0.8, 0.99])
        propensity = Propensity(np.column_stack([1.0 - g1, g1]), (0.0, 1.0))
        bounded = propensity.bounded((0.01, 0.95))
        np.testing.assert_allclose(bounded[:, 1], np.clip(g1, 0.01, 0.95))
        np.testing.assert_allclose(bounded[:, 0], 1.0 - np.clip(g1, 0.01, 0.95))
        # The simplex survives exactly, which is why a binary fit reports no deviation.
        np.testing.assert_array_equal(bounded.sum(axis=1), np.ones(5))

    def test_two_arms_off_the_simplex_are_clipped_column_by_column(self) -> None:
        """``simplex=False`` gets the multi-arm rule, because the complement is not one.

        The joint arm-and-observation mechanism a missing-outcome DR-TMLE divides by is
        ``g_a(W) pi_a(W)``: two columns that are each a probability and are *not* each
        other's complement.  Taking arm 0 as ``1 - g_1 pi_1`` would hand it the probability
        of *not* being treated-and-observed, which is a different and much larger number.
        """
        joint = np.array([[0.005, 0.30], [0.35, 0.30], [0.02, 0.97]])
        mechanism = Propensity(joint, (0.0, 1.0), simplex=False)
        bounded = mechanism.bounded((0.01, 0.95))
        np.testing.assert_array_equal(bounded, np.clip(joint, 0.01, 0.95))
        # The mutation control: on the same array the complement rule is a materially
        # different denominator, so the assertion above is not one the old code passed too.
        complement = 1.0 - np.clip(joint[:, 1], 0.01, 0.95)
        assert np.max(np.abs(bounded[:, 0] - complement)) > 0.6

    def test_a_two_arm_mechanism_off_the_simplex_must_say_so(self) -> None:
        """The flag is checked, not trusted, because forgetting it is silent otherwise.

        Without this the joint would take the complement branch and the fit would return a
        finite, plausible, wrong number rather than an error.
        """
        joint = np.column_stack([np.array([0.30, 0.35]), np.array([0.28, 0.31])])
        with pytest.raises(ValueError, match="must sum to one"):
            Propensity(joint, (0.0, 1.0))
        Propensity(joint, (0.0, 1.0), simplex=False)  # named, and so allowed

    def test_the_simplex_check_spares_the_cases_that_cannot_meet_it(self) -> None:
        """Three arms are deliberately off the simplex, and an unfitted one is all NaN."""
        Propensity(np.full((4, 3), 0.2), (0.0, 1.0, 2.0))
        UnfittedPropensity(np.full((4, 2), np.nan), (0.0, 1.0))

    def test_more_arms_are_clipped_per_arm_and_not_renormalised(self) -> None:
        """Every arm is floored, and the row is left off the simplex deliberately.

        Renormalising would divide the floored column back down and could push it under
        the floor again -- which would defeat the only thing the floor is for, bounding
        ``1 / g_a``.
        """
        values = np.array([[0.001, 0.199, 0.800], [1 / 3, 1 / 3, 1 / 3]])
        propensity = Propensity(values, (0.0, 1.0, 2.0))
        bounded = propensity.bounded((0.01, 0.99))
        assert bounded.min() >= 0.01
        assert bounded[0].sum() > 1.0  # floored upward, and left there
        np.testing.assert_allclose(bounded[1], values[1])  # untouched where nothing binds

    def test_the_deviation_from_the_simplex_is_reported(self) -> None:
        result = (
            TMLE(
                outcome_learner="glm",
                treatment_learner="glm",
                n_folds=5,
                g_bounds=0.35,  # binds hard on three arms averaging a third each
                random_state=0,
                simultaneous=False,
                estimands=("ey",),
            )
            .fit(_three_arm_frame(), outcome="Y", treatment="A")
            .single()
        )
        report = result.sensitivity.positivity()
        assert report.simplex_deviation > 0.0
        # The plug-in is an average of targeted predictions and contains no mechanism, so
        # a bound that binds this hard still leaves the estimates finite and ordered.
        assert all(np.isfinite(est.psi) for est in result.estimates.values())


class TestWhatIsRefused:
    @pytest.mark.parametrize("estimand", ["ey1", "ey0"])
    def test_binary_only_estimands_are_refused(self, estimand: str) -> None:
        with pytest.raises(ValueError, match="binary treatment only"):
            TMLE(
                outcome_learner="glm",
                treatment_learner="glm",
                estimands=(estimand,),
                random_state=0,
            ).fit(_three_arm_frame(), outcome="Y", treatment="A")

    def test_selector_ctmle_uses_one_joint_multinomial_path(self) -> None:
        from cleverly import CTMLE

        result = (
            CTMLE(
                outcome_learner="glm",
                treatment_learner="glm",
                strategy="discrete",
                candidates=((), ("W1",)),
                selection_folds=2,
                learner_folds=2,
                random_state=0,
                estimands=("ate",),
            )
            .fit(_three_arm_frame(), outcome="Y", treatment="A")
            .single()
        )
        assert len(result.extra["ctmle"].target_names) == 2
        assert result.nuisance.propensity.values.shape[1] == 3

    # The omitted-variable bound and the MNAR tilt were refused here too, and are not
    # any more: both are one parameter at a time and so generalise to one per contrast.
    # tests/unit/test_sensitivity_multi_arm.py is where they are checked, against the
    # closed-form nu^2 of each contrast's own Riesz representer.

    def test_an_unknown_reference_names_the_levels_it_could_have_been(self) -> None:
        with pytest.raises(DataError, match="is not a level of A"):
            TMLE(
                outcome_learner="glm",
                treatment_learner="glm",
                reference="nope",
                random_state=0,
                estimands=("ate",),
            ).fit(law.frame(), outcome="Y", treatment="A", covariates=["W"])


class TestTheReference:
    def test_it_chooses_which_contrasts_are_reported(self) -> None:
        frame = law.frame()
        estimates = {}
        for reference in ("low", "high"):
            result = (
                TMLE(
                    outcome_learner=law.OracleMultiOutcome(),
                    treatment_learner=law.OracleMultiTreatment(),
                    cross_fit=False,
                    estimands=("ate",),
                    reference=reference,
                    simultaneous=False,
                    random_state=0,
                )
                .fit(frame, outcome="Y", treatment="A", covariates=["W"])
                .single()
            )
            estimates[reference] = result.estimates

        assert set(estimates["low"]) == {"ate[high vs low]", "ate[mid vs low]"}
        assert set(estimates["high"]) == {"ate[low vs high]", "ate[mid vs high]"}
        # The same underlying means, so the two reports must reconcile exactly.
        assert estimates["low"]["ate[high vs low]"].psi == pytest.approx(
            -estimates["high"]["ate[low vs high]"].psi, abs=1e-12
        )

    def test_it_defaults_to_the_lowest_arm(self) -> None:
        data = CausalData.from_frame(law.frame(), outcome="Y", treatment="A", covariates=["W"])
        estimator = TMLE(estimands=("ate",))
        assert estimator._reference_arm(data) == data.arm_codes[0]
        assert data.arm_label(estimator._reference_arm(data)) == "high"


class TestTheConditionalEffects:
    """What a multi-arm ``att`` / ``atc`` reports, and when.

    The numbers are checked against the oracle in
    :mod:`tests.unit.test_influence_gateaux_multi`; this is about the report.
    """

    @staticmethod
    def _fit(**overrides):  # type: ignore[no-untyped-def]
        return (
            TMLE(
                outcome_learner="glm",
                treatment_learner="glm",
                n_folds=5,
                learner_folds=3,
                random_state=0,
                simultaneous=False,
                **overrides,
            )
            .fit(_three_arm_frame(), outcome="Y", treatment="A")
            .single()
        )

    def test_they_are_not_in_the_default_report(self) -> None:
        """Opt-in, so an existing multi-arm fit reports exactly what it always did.

        They are ``2(K - 1)`` further parameters behind two further fluctuations, and a
        default that grew to include them would move every multi-arm fit's simultaneous
        bands -- which are computed across whatever is reported.
        """
        assert set(self._fit().estimates) == {
            "ey[0.0]",
            "ey[1.0]",
            "ey[2.0]",
            "ate[1.0 vs 0.0]",
            "ate[2.0 vs 0.0]",
        }

    def test_asking_for_them_reports_one_per_non_reference_arm(self) -> None:
        result = self._fit(estimands=("att", "atc"))
        assert set(result.estimates) == {
            "att[1.0 vs 0.0]",
            "att[2.0 vs 0.0]",
            "atc[1.0 vs 0.0]",
            "atc[2.0 vs 0.0]",
        }
        # One fluctuation per group, each with a column per contrast rather than per arm.
        for group in ("att", "atc"):
            assert result.fluctuations[group].epsilon.shape == (2,)
            assert result.fluctuations[group].converged

    def test_they_follow_the_declared_reference(self) -> None:
        result = self._fit(estimands=("att",), reference=2.0)
        assert set(result.estimates) == {"att[0.0 vs 2.0]", "att[1.0 vs 2.0]"}

    def test_all_includes_them_where_the_default_does_not(self) -> None:
        assert set(self._fit(estimands="all").estimates) >= {"att[1.0 vs 0.0]", "atc[2.0 vs 0.0]"}
