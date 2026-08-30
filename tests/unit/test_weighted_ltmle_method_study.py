"""Focused contracts for the two weighted longitudinal evidence rows."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from cleverly.datasets import make_longitudinal
from cleverly.learners.crossfit import Folds
from tests.studies import canonical_weighted_ltmle as ordinary
from tests.studies import canonical_weighted_ltmle_crossfit as crossfit
from tests.studies import weighted_longitudinal_properties_common as properties


def test_the_selected_sampler_returns_exactly_the_declared_size() -> None:
    first, first_truth = ordinary.draw_from_seed(ordinary.common.SCENARIO, 2_000, 17)
    second, second_truth = ordinary.draw_from_seed(ordinary.common.SCENARIO, 2_000, 17)

    assert len(first) == 2_000
    assert first.equals(second)
    assert first_truth == second_truth
    expected = np.where(first["W1"].to_numpy() > 0.0, 1.0 / 0.3, 1.0 / 0.9)
    np.testing.assert_allclose(first["obs_weight"], expected, atol=0.0, rtol=0.0)


def test_the_two_rows_have_distinct_seed_streams_and_declared_counts() -> None:
    assert ordinary.STUDY.seed != crossfit.STUDY.seed
    assert ordinary.STUDY.replicates == crossfit.STUDY.replicates == 800
    assert ordinary.STUDY.n == crossfit.STUDY.n == 2_000
    assert ordinary.STUDY.estimands == crossfit.STUDY.estimands
    assert ordinary.STUDY.property_cells == crossfit.STUDY.property_cells


def test_a_primary_fit_retains_the_fixed_weights() -> None:
    frame, _ = ordinary.draw_from_seed(ordinary.common.SCENARIO, 500, 31)
    result = ordinary.fit_cleverly(frame)
    expected = frame["obs_weight"].to_numpy(dtype=float).copy()
    expected = expected / expected.mean()

    for fitted in result.fits.values():
        np.testing.assert_allclose(fitted.obs_weights, expected, atol=0.0, rtol=0.0)


def test_the_cross_fitted_payload_serializes_the_fitted_fold_assignment() -> None:
    n = 500
    planted = Folds(np.repeat(np.arange(5), n // 5), 5)
    with patch("cleverly.longitudinal.estimator.make_folds", return_value=planted):
        sample, _, _ = crossfit._replicate((crossfit.common.SCENARIO, 0, n))
    np.testing.assert_array_equal(sample["fold"], planted.assignment)


def test_the_finite_property_law_is_exactly_untilted_by_its_weights() -> None:
    weighted = properties.SELECTED_PROBS * properties.OBS_WEIGHTS
    weighted /= weighted.sum()
    np.testing.assert_allclose(weighted, properties.law.PROBS, atol=1e-15, rtol=0.0)
    assert set(properties.property_cells()) == {
        "double_robustness",
        "root_n_and_efficiency",
        "root_n_rate",
        "interval_calibration",
        "type_i_error",
        "power",
        "targeting_necessity",
        "weight_necessity",
        "learner_weight_necessity",
    }


def test_the_learner_weight_law_untilts_and_moves_each_sequential_nuisance() -> None:
    selected = properties.law.PROBS * properties.LEARNER_SELECTION
    selected /= selected.sum()
    recovered = selected * properties.LEARNER_OBS_WEIGHTS
    recovered /= recovered.sum()
    np.testing.assert_allclose(recovered, properties.law.PROBS, atol=1e-15, rtol=0.0)

    def largest_conditional_shift(node: int) -> float:
        histories: dict[tuple[object, ...], list[int]] = {}
        for index, point in enumerate(properties.law.SUPPORT):
            if point[node] is not None:
                histories.setdefault(point[:node], []).append(index)

        shifts = []
        for indices in histories.values():
            source_total = float(properties.law.PROBS[indices].sum())
            selected_total = float(selected[indices].sum())
            source_mean = (
                sum(
                    properties.law.PROBS[index] * float(properties.law.SUPPORT[index][node])
                    for index in indices
                )
                / source_total
            )
            selected_mean = (
                sum(
                    selected[index] * float(properties.law.SUPPORT[index][node])
                    for index in indices
                )
                / selected_total
            )
            shifts.append(abs(source_mean - selected_mean))
        return max(shifts)

    # A1, C1, L2, A2, C2, and Y are the six sequential conditional families.
    assert all(largest_conditional_shift(node) > 0.01 for node in range(1, 7))


def test_the_lmtp_weight_adapter_excludes_its_auxiliary_column() -> None:
    root = Path(__file__).resolve().parents[2]
    learner = (root / "tests/canonical/lmtp_weighted_glm_adapter.R").read_text()
    fold_adapter = (root / "tests/canonical/lmtp_crossfit_adapter.R").read_text()
    runner = (root / "tests/canonical/weighted_lmtp_ltmle/run_study.R").read_text()

    assert 'LMTP_AUX_WEIGHT <- "obs_weight_aux"' in learner
    assert "setdiff(names(X), weight_column)" in learner
    assert "obsWeights = design$weights" in learner
    assert "weights = weights" in fold_adapter
    assert "weights = frame$obs_weight" in runner
    assert 'learners_outcome = "SL.weighted.glm"' in runner
    assert 'baseline = c("W1", "W2", "obs_weight_aux")' in runner


def test_the_optional_lmtp_weight_boundary_rejects_bad_values_in_r_source() -> None:
    root = Path(__file__).resolve().parents[2]
    adapter = (root / "tests/canonical/lmtp_crossfit_adapter.R").read_text()
    assert "weights = NULL" in adapter
    assert 'stop("weights have the wrong length")' in adapter
    assert 'stop("weights must be finite positive numbers")' in adapter
    assert adapter.count("weights = weights") == 1


def test_property_efficiency_bounds_are_finite_and_nonzero() -> None:
    assert properties.EFFICIENCY_SD.keys() == properties.CONTRASTS.keys()
    assert all(np.isfinite(value) and value > 0.0 for value in properties.EFFICIENCY_SD.values())


@pytest.mark.parametrize("module", [ordinary, crossfit], ids=("ordinary", "cross-fitted"))
def test_primary_truth_keeps_the_existing_end_of_study_target(module: object) -> None:
    _, truth = module.draw_from_seed(module.common.SCENARIO, 100, 9)  # type: ignore[attr-defined]
    _, existing = make_longitudinal(n=1, seed=0, censoring=True, backend="pandas")
    assert truth == {name: existing[name] for name in module.common.ESTIMANDS}  # type: ignore[attr-defined]
