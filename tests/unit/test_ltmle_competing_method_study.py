"""Structural gates for the two competing-risk LTMLE evidence studies."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from cleverly.learners.crossfit import Folds
from cleverly.longitudinal import LongitudinalData
from tests import discrete_law_competing as law
from tests.studies import canonical_ltmle_competing as ordinary
from tests.studies import canonical_ltmle_competing_crossfit as crossfit
from tests.studies import ltmle_competing_crossfit_properties as crossfit_properties
from tests.studies import ltmle_competing_properties as properties


def test_the_primary_studies_report_every_unique_parameter_once() -> None:
    """The dynamic plan's duplicate first horizon stays out of repeated sampling."""
    assert len(ordinary.ESTIMANDS) == 16
    assert len(set(ordinary.ESTIMANDS)) == len(ordinary.ESTIMANDS)
    assert crossfit.ESTIMANDS == ordinary.ESTIMANDS

    for cause in law.CAUSES:
        dynamic = law.TRUTH[f"cif_regimen[continue_if_l2, {cause} @ t=1]"]
        always = law.TRUTH[f"cif_regimen[always, {cause} @ t=1]"]
        assert dynamic == pytest.approx(always, abs=1e-15)
        assert f"cif_regimen[continue_if_l2, {cause} @ t=1]" not in ordinary.ESTIMANDS


def test_the_ordinary_study_reserves_an_independent_resampling_namespace() -> None:
    assert ordinary.STUDY.resampling_seed == 20260828
    assert ordinary.STUDY.resampling_seed != crossfit.STUDY.seed


@pytest.mark.parametrize("study", [ordinary, crossfit], ids=("ordinary", "cross-fitted"))
def test_each_study_declares_the_two_cause_properties(study: Any) -> None:
    assert study.PROPERTY_LABELS == ("relapse_dynamic_t2", "death_static_t2")
    cells = study.STUDY.property_cells
    assert set(cells["competing_risk_recursion_necessity"]) == {
        "relapse_always_t2__all_cause",
        "relapse_always_t2__cause_specific_control",
        "death_always_t2__all_cause",
        "death_always_t2__cause_specific_control",
    }
    assert ("crossfit_overfitting" in cells) is (study is crossfit)


def test_the_null_is_exact_and_keeps_both_competing_causes() -> None:
    assert properties.NULL_PROBS.sum() == pytest.approx(1.0, abs=1e-15)
    assert all(value == pytest.approx(0.0, abs=1e-14) for value in properties.NULL_TRUTH.values())
    assert np.all(properties.NULL_H1 > 0.0)
    assert np.all(properties.NULL_H2 > 0.0)
    assert np.all(properties.NULL_H1.sum(axis=0) < 1.0)
    assert np.all(properties.NULL_H2.sum(axis=0) < 1.0)


def test_each_power_control_has_a_material_cause_specific_effect() -> None:
    assert properties.POWER_TRUTH["relapse_dynamic_t2"] > 0.10
    assert properties.POWER_TRUTH["death_static_t2"] < -0.10


def test_the_overfitting_panel_has_two_balanced_absorbing_causes() -> None:
    frame, truth = crossfit_properties._balanced_competing_panel(4_000, 21)
    assert abs(truth) > 0.01
    for node in (1, 2):
        both = (frame[f"R{node}"] == 1.0) & (frame[f"D{node}"] == 1.0)
        assert not bool(both.any())
        assert int((frame[f"R{node}"] == 1.0).sum()) > 100
        assert int((frame[f"D{node}"] == 1.0).sum()) > 100
    first_event = (frame["R1"] == 1.0) | (frame["D1"] == 1.0)
    assert frame.loc[first_event, ["L2", "A2", "C2"]].isna().all().all()
    assert (frame.loc[frame["R1"] == 1.0, "R2"] == 1.0).all()
    assert (frame.loc[frame["D1"] == 1.0, "D2"] == 1.0).all()


def test_the_known_mechanisms_read_the_declared_history_design() -> None:
    frame = ordinary.draw_from_seed(ordinary.SCENARIO, 600, 19)[0]
    data = LongitudinalData.from_frame(
        frame,
        outcome=law.outcome_columns(),
        treatment=["A1", "A2"],
        baseline=["W"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
    )
    treatment = ordinary.KnownCompetingMechanism("treatment").fit(None, None)
    censoring = ordinary.KnownCompetingMechanism("censoring").fit(None, None)
    for time in (1, 2):
        history = data.history_design(time)
        index = np.rint(np.nan_to_num(history)).astype(int)
        expected_g = (
            law.G1[index[:, 0]] if time == 1 else law.G2[index[:, 0], index[:, 2], index[:, 1]]
        )
        np.testing.assert_allclose(treatment.predict_proba(history)[:, 1], expected_g)

        current = data.history_design(time, include_current=True)
        index = np.rint(np.nan_to_num(current)).astype(int)
        expected_c = (
            law.C1[index[:, 0], index[:, 1]]
            if time == 1
            else law.C2[index[:, 0], index[:, 2], index[:, 1], index[:, 3]]
        )
        np.testing.assert_allclose(censoring.predict_proba(current)[:, 1], expected_c)


def test_the_all_cause_recursion_agrees_at_the_law_and_the_mutation_does_not() -> None:
    """The control must fail only when the other cause leaves the risk set."""
    frame = law.frame()
    result = properties.fit(frame, "both_correct")
    for cause in law.CAUSES:
        name = f"cif_regimen[always, {cause} @ t=2]"
        correct = properties.untargeted(frame, "always", cause, 2, "both_correct", result.folds)
        wrong = properties.untargeted(
            frame,
            "always",
            cause,
            2,
            "both_correct",
            result.folds,
            cause_specific_survival=True,
        )
        assert correct == pytest.approx(law.TRUTH[name], abs=1e-12)
        assert abs(wrong - law.TRUTH[name]) > 0.02


def test_the_cross_fitted_r_payload_uses_the_realized_outer_folds() -> None:
    frame, truth = crossfit.draw_scenario(crossfit.SCENARIO, 1_000, 0)
    planted = Folds(np.repeat(np.arange(5), len(frame) // 5), 5)
    with patch("cleverly.longitudinal.estimator.make_folds", return_value=planted):
        sample, written_truth, rows = crossfit._replicate((crossfit.SCENARIO, 0, len(frame)))

    np.testing.assert_array_equal(sample["fold"].to_numpy(), planted.assignment)
    assert {row["estimand"] for row in written_truth} == set(truth)
    assert {row["estimand"] for row in rows} == set(crossfit.ESTIMANDS)


@pytest.mark.parametrize("study", [ordinary, crossfit], ids=("ordinary", "cross-fitted"))
def test_the_frozen_r_study_matches_beyond_its_acceptance_margin(study: Any) -> None:
    path = study.STUDY.artifact("replicates.csv.gz")
    if not path.exists():
        pytest.skip("the new study artifacts have not been generated yet")
    rows = pd.read_csv(path)
    paired = rows.pivot(
        index=["scenario", "replicate", "estimand"],
        columns="implementation",
        values=["estimate", "std_error"],
    )
    for column, tolerance in {"estimate": 5e-6, "std_error": 5e-7}.items():
        difference = paired[(column, study.STUDY.implementation)] - paired[(column, "lmtp")]
        assert np.max(np.abs(difference)) < tolerance

    movement = (
        rows.assign(targeting_movement=(rows["estimate"] - rows["initial_estimate"]).abs())
        .groupby("implementation")["targeting_movement"]
        .mean()
    )
    assert (movement > 0.005).all(), movement
