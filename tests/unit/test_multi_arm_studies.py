"""Cheap structural gates for the registered multi-arm evidence family."""

from __future__ import annotations

import numpy as np
import pytest

from tests.studies import (
    canonical_multi_arm_ctmle_oat,
    canonical_multi_arm_ctmle_selector,
    canonical_multi_arm_drtmle,
    canonical_multi_arm_tmle,
    multi_arm_common,
)


def test_the_shared_law_exercises_label_code_order() -> None:
    assert tuple(sorted(("low", "medium", "high"))) == multi_arm_common.LABELS
    assert multi_arm_common.law().labels != multi_arm_common.LABELS


def test_the_oracle_treatment_columns_follow_estimator_codes_not_source_order() -> None:
    process = multi_arm_common.law()
    latent = np.array(((0.4, -0.8, 0.2), (-0.5, 0.7, -0.1)))
    expected = process.probabilities(latent)
    reordered = (
        multi_arm_common.OracleMultiTreatment(process).fit(latent, [0, 1]).predict_proba(latent)
    )
    source_columns = np.column_stack(
        [expected[:, process.labels.index(label)] for label in multi_arm_common.LABELS]
    )
    np.testing.assert_allclose(reordered, source_columns)
    assert not np.allclose(reordered, expected)


def test_the_primary_fit_has_a_nonzero_targeting_witness() -> None:
    frame, _ = canonical_multi_arm_tmle.draw_from_seed(
        canonical_multi_arm_tmle.SCENARIO, 1000, 7331
    )
    result = canonical_multi_arm_tmle.fit_cleverly(frame, canonical_multi_arm_tmle.SCENARIO)
    initial = result.repeats[0].nuisance.outcome.arms
    displacements = [
        abs(result[f"ey[{label}]"].psi - float(np.mean(initial[float(code)])))
        for code, label in enumerate(multi_arm_common.LABELS)
    ]
    assert max(displacements) > 1e-6


def test_every_derived_ratio_has_the_declared_arm_mean_truth() -> None:
    truth = multi_arm_common.truth_for(multi_arm_common.law())
    for label in ("low", "medium"):
        mean = truth[f"ey[{label}]"]
        reference = truth["ey[high]"]
        assert truth[f"rr[{label} vs high]"] == pytest.approx(mean / reference)
        assert truth[f"or[{label} vs high]"] == pytest.approx(
            (mean / (1 - mean)) / (reference / (1 - reference))
        )


def test_each_materially_different_method_has_its_own_row() -> None:
    records = (
        canonical_multi_arm_tmle.STUDY,
        canonical_multi_arm_drtmle.STUDY,
        canonical_multi_arm_ctmle_oat.STUDY,
        canonical_multi_arm_ctmle_selector.STUDY,
    )
    assert len({record.slug for record in records}) == 4
    assert all(set(record.estimands) == set(multi_arm_common.ALL_ESTIMANDS) for record in records)


def test_the_selector_does_not_claim_a_binary_only_r_comparator() -> None:
    assert canonical_multi_arm_ctmle_selector.STUDY.reference is None
    assert canonical_multi_arm_tmle.STUDY.reference == "tmle3-multi-arm"
    assert canonical_multi_arm_drtmle.STUDY.reference == "drtmle-r-multi-arm"
    assert canonical_multi_arm_ctmle_oat.STUDY.reference == "ctmle3-multi-arm-oat"
