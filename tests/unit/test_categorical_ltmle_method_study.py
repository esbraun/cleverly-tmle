"""Structural gates for the categorical longitudinal evidence studies."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from cleverly.learners.crossfit import Folds
from tests import discrete_law_longitudinal_multivalue as law
from tests.studies import canonical_categorical_ltmle as ordinary
from tests.studies import canonical_categorical_ltmle_crossfit as crossfit
from tests.studies import categorical_longitudinal_common as common


def test_each_categorical_outer_fold_recovers_the_exact_law_and_gateaux_curve() -> None:
    """Every training complement and held-out fold contains one complete oracle law."""
    base = law.frame()
    frame = pd.concat([base] * common.N_FOLDS, ignore_index=True)
    folds = Folds(np.repeat(np.arange(common.N_FOLDS), len(base)), common.N_FOLDS)
    with patch("cleverly.longitudinal.estimator.make_folds", return_value=folds):
        result = common.fit(frame, cross_fit=True, configuration="both_correct")

    for name in common.CONTRASTS.values():
        assert result[name].psi == pytest.approx(law.TRUTH[name], abs=1e-12)
        expected = np.tile(np.repeat(law.eif_at(law.PROBS, name), law.COUNTS), common.N_FOLDS)
        np.testing.assert_allclose(result.influence_curves[name], expected, atol=2e-12, rtol=0.0)


def test_primary_crossfit_training_and_validation_sets_retain_all_three_arms() -> None:
    """No primary fold loses a categorical treatment level at either node."""
    frame, _ = crossfit.draw_scenario(common.SCENARIO, crossfit.PRIMARY_N, 0)
    result = crossfit.fit_cleverly(frame)
    expected = set(law.ARM_LABELS)
    for training, validation in result.folds:
        for column in ("A1", "A2"):
            assert set(frame.iloc[training][column]) == expected
            assert set(frame.iloc[validation][column]) == expected


def test_a_held_out_outcome_cannot_enter_its_categorical_recursion() -> None:
    """Each categorical prediction and update uses the row's training complement."""
    frame, _ = crossfit.draw_scenario(common.SCENARIO, 500, 1)
    original = crossfit.fit_cleverly(frame)
    changed = frame.copy()
    row = 0
    changed.loc[row, "Y"] = 1.0 - changed.loc[row, "Y"]
    perturbed = crossfit.fit_cleverly(changed)

    np.testing.assert_array_equal(original.folds.assignment, perturbed.folds.assignment)
    for label in common.REGIMENS:
        for left, right in zip(
            original.fits[label].steps, perturbed.fits[label].steps, strict=True
        ):
            assert left.initial[row] == right.initial[row]
            assert left.targeted[row] == right.targeted[row]


def test_serialized_folds_are_the_fitted_assignments_for_both_constructions() -> None:
    """The R payload carries the fitted fold instead of reconstructing it from a seed."""
    ordinary_sample, _, _ = common._replicate(ordinary.STUDY, False, common.SCENARIO, 0, 500)
    assert set(ordinary_sample["fold"]) == {0}

    frame = pd.concat([law.frame()] * common.N_FOLDS, ignore_index=True)
    truth = dict(law.TRUTH)
    planted = Folds(np.repeat(np.arange(common.N_FOLDS), len(law.frame())), 5)
    with (
        patch(
            "tests.studies.categorical_longitudinal_common.draw_for", return_value=(frame, truth)
        ),
        patch("cleverly.longitudinal.estimator.make_folds", return_value=planted),
    ):
        sample, truths, _ = common._replicate(crossfit.STUDY, True, common.SCENARIO, 0, len(frame))
    np.testing.assert_array_equal(sample["fold"].to_numpy(), planted.assignment)
    assert {row["estimand"] for row in truths} == set(truth)


def test_scrambled_raw_codes_do_not_match_the_semantic_arm_order() -> None:
    """The study would detect a raw-code or sorted-label substitution."""
    assert tuple(sorted(law.ARM_LABELS)) != law.ARM_LABELS
    assert tuple(law.ARM_LABELS[index] for index in range(3)) == ("standard", "high", "low")
    assert common.LEVELS == ("high", "low", "standard")
