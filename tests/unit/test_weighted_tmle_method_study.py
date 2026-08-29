"""Focused contracts for the weighted point-treatment evidence study."""

from __future__ import annotations

import numpy as np
import pytest

from tests.studies import canonical_weighted_tmle as study
from tests.studies import weighted_point_common as law
from tests.studies import weighted_tmle_properties as properties


def test_the_registered_design_matches_the_declared_plan() -> None:
    assert study.STUDY.replicates == 800
    assert study.STUDY.n == 2_000
    assert study.STUDY.scenarios == {study.SCENARIO: ("ey0", "ey1", "ate", "rr", "or")}
    assert study.STUDY.reference == "tmle-r-weighted"
    assert properties.DOUBLE_ROBUST_REPLICATES == 1_200
    assert properties.DOUBLE_ROBUST_N == 2_000
    assert properties.RATE_REPLICATES == 800
    assert properties.RATE_SIZES == (500, 2_000, 8_000)
    assert properties.CALIBRATION_REPLICATES == 2_400
    assert properties.CALIBRATION_N == 2_000
    assert properties.NULL_REPLICATES == 800
    assert properties.NULL_N == 1_000
    assert properties.NECESSITY_REPLICATES == 1_200
    assert properties.NECESSITY_N == 2_000
    assert properties.WEIGHT_DISPLACEMENT == 0.50


def test_inverse_selection_weights_recover_the_population_law_exactly() -> None:
    tilted = law.SELECTED_P_W * law.OBSERVATION_WEIGHTS
    tilted /= tilted.sum()
    np.testing.assert_allclose(tilted, law.P_W, rtol=0, atol=1e-15)
    assert law.population_truth()["ate"] == pytest.approx(0.33)
    assert law.selected_truth()["ate"] == pytest.approx(0.5222222222222223)
    assert abs(law.population_truth()["ate"] - law.selected_truth()["ate"]) > 0.15


def test_the_sampler_draws_the_requested_size_directly_from_the_selected_law() -> None:
    frame = law.sample_selected(law.Q, 731, 19)
    assert len(frame) == 731
    levels = frame["W"].to_numpy(dtype=int)
    np.testing.assert_allclose(frame["obs_weight"], law.OBSERVATION_WEIGHTS[levels])
    assert set(frame.columns) == {"Y", "A", "W", "obs_weight"}


def test_the_exact_weighted_eif_is_centered_under_the_sampling_law() -> None:
    probabilities = law.selected_probabilities().reshape(-1)
    curve = law.weighted_ate_eif()
    assert float(probabilities @ curve) == pytest.approx(0.0, abs=1e-14)
    assert law.weighted_ate_efficiency_sd() == pytest.approx(
        float(np.sqrt(probabilities @ np.square(curve)))
    )


def test_primary_rows_preserve_native_ratio_inference_and_r_inputs() -> None:
    samples, truths, estimates = study.draw_and_fit(replicates=1, n=600, n_jobs=1)
    assert {"qn0", "qn1", "gn1", "obs_weight"}.issubset(samples)
    assert set(truths["estimand"]) == set(study.ESTIMANDS)
    by_name = estimates.set_index("estimand")
    assert by_name.loc["rr", "inference_scale"] == "log"
    assert by_name.loc["or", "inference_scale"] == "log"
    assert by_name.loc["ate", "inference_scale"] == "identity"
    assert by_name.loc["rr", "inference_estimate"] == pytest.approx(
        np.log(by_name.loc["rr", "estimate"])
    )
    assert by_name.loc["or", "inference_estimate"] == pytest.approx(
        np.log(by_name.loc["or", "estimate"])
    )


def test_the_omitted_weight_control_recovers_only_the_selected_target() -> None:
    weighted, omitted = properties.fit_replication(
        ("weight_necessity", "weighted", 0, 4_000, 1, 641, "both_correct")
    )
    assert weighted["cell"] == "ate__weighted"
    assert omitted["cell"] == "ate__omitted_control"
    assert weighted["truth"] == omitted["truth"] == pytest.approx(properties.TRUTH)
    assert abs(weighted["estimate"] - properties.TRUTH) < 0.08
    assert abs(omitted["estimate"] - properties.SELECTED_TRUTH) < 0.08
    assert abs(omitted["estimate"] - properties.TRUTH) > 0.10


def test_the_targeting_control_removes_only_the_fluctuation() -> None:
    targeted, untargeted = properties.fit_replication(
        ("targeting_necessity", "targeted", 0, 4_000, 1, 877, "treatment_correct")
    )
    assert targeted["cell"] == "ate__targeted"
    assert untargeted["cell"] == "ate__untargeted"
    assert targeted["truth"] == untargeted["truth"] == pytest.approx(properties.TRUTH)
    assert targeted["std_error"] == untargeted["std_error"]
    assert abs(targeted["estimate"] - properties.TRUTH) < 0.08
    assert abs(untargeted["estimate"] - properties.TRUTH) > 0.20
