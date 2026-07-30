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
from cleverly.estimators._nuisance import Propensity
from cleverly.exceptions import DataError
from cleverly.fluctuation.submodel import mean_submodel
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
    @pytest.mark.parametrize("estimand", ["att", "atc", "ey1", "ey0"])
    def test_binary_only_estimands_are_refused(self, estimand: str) -> None:
        with pytest.raises(ValueError, match="binary treatment only"):
            TMLE(
                outcome_learner="glm",
                treatment_learner="glm",
                estimands=(estimand,),
                random_state=0,
            ).fit(_three_arm_frame(), outcome="Y", treatment="A")

    def test_ctmle_is_refused(self) -> None:
        from cleverly import CTMLE

        with pytest.raises(ValueError, match="binary treatment only"):
            CTMLE(
                outcome_learner="glm",
                treatment_learner="glm",
                random_state=0,
                estimands=("ate",),
            ).fit(_three_arm_frame(), outcome="Y", treatment="A")

    def test_the_omitted_variable_bound_is_refused(self) -> None:
        result = (
            TMLE(
                outcome_learner="glm",
                treatment_learner="glm",
                n_folds=5,
                random_state=0,
                simultaneous=False,
                estimands=("ate",),
            )
            .fit(_three_arm_frame(), outcome="Y", treatment="A")
            .single()
        )
        with pytest.raises(ValueError, match="derived for a binary treatment"):
            result.sensitivity.omitted_variable(estimand="ate")

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
