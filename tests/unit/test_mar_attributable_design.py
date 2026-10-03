"""The declared design of the missing-outcome attributable-effect study, recomputed.

Each control of :mod:`tests.studies.mar_attributable_properties` is sized from an exact
large-sample limit on L1.  This module recomputes the limits and the standardized
distances the module docstring states, before any run reads them.  The band cells' designs
live in ``tests/unit/test_simultaneous_cell_design.py`` beside every other joint cell.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests import discrete_law_mar as mar
from tests.studies import canonical_mar_attributable as study
from tests.studies import mar_attributable_properties as properties

N = properties.ROBUSTNESS_N
TRUTH = study.TRUTHS[study.BINARY]


def _per_replication_sd(name: str) -> float:
    return study.EFFICIENCY_SD[name] / np.sqrt(N)


def test_the_stack_covariance_is_the_gateaux_covariance() -> None:
    """The closed form of ``stack_covariance`` against the law's own derivative."""
    covariance = study.stack_covariance(study.L1)
    for i, left in enumerate(study.BINARY_NAMES):
        for j, right in enumerate(study.BINARY_NAMES):
            exact = float(np.sum(mar.PROBS.reshape(-1) * mar.eif(left) * mar.eif(right)))
            assert covariance[i, j] == pytest.approx(exact, abs=1e-12)


def test_the_declared_truths_and_bounds() -> None:
    assert (
        pytest.approx(
            {"ey_obs": 0.494, "ey0": 0.380, "par": 0.114, "paf": 0.2307692307692308}, abs=1e-12
        )
        == TRUTH
    )
    three = study.TRUTHS[study.THREE_ARM]
    assert three["ey_obs"] == pytest.approx(0.533, abs=1e-12)
    assert three["ey[low]"] == pytest.approx(0.380, abs=1e-12)
    assert three["par[low]"] == pytest.approx(0.153, abs=1e-12)
    assert three["paf[low]"] == pytest.approx(0.2870544, abs=1e-7)
    assert study.EFFICIENCY_SD["par"] == pytest.approx(0.67358, abs=1e-5)
    assert study.EFFICIENCY_SD["paf"] == pytest.approx(1.43736, abs=1e-5)


@pytest.mark.parametrize("configuration", ["both_correct", "outcome_correct", "mechanisms_correct"])
def test_each_positive_robustness_cell_is_unbiased_in_the_limit(configuration: str) -> None:
    limit = study.stack_limit(study.L1, **properties.CONFIGURATIONS[configuration])
    for name in study.BINARY_NAMES:
        assert limit[name] == pytest.approx(TRUTH[name], abs=1e-9)


@pytest.mark.parametrize(
    ("configuration", "bias", "standardized"),
    [
        ("treatment_wrong", -0.1354, -8.99),
        ("observation_wrong", -0.1968, -13.07),
        ("product_only", -0.1243, -8.25),
    ],
)
def test_each_robustness_control_is_displaced_by_many_bias_margins(
    configuration: str, bias: float, standardized: float
) -> None:
    limit = study.stack_limit(study.L1, **properties.CONFIGURATIONS[configuration])
    displacement = limit["par"] - TRUTH["par"]
    assert displacement == pytest.approx(bias, abs=5e-5)
    assert displacement / _per_replication_sd("par") == pytest.approx(standardized, abs=0.01)
    # At least four of the shared standardized-bias margins of 0.25.
    assert (
        abs(displacement) / _per_replication_sd("par") >= 4 * study.STUDY.margins.standardized_bias
    )


def test_the_product_only_cell_restores_the_reference_product_only() -> None:
    """The reference arm's limit is exact; only the natural course carries the bias."""
    tables = properties.CONFIGURATIONS["product_only"]
    np.testing.assert_allclose(
        tables["g"][:, 0] * tables["pi"][:, 0], (1.0 - mar.G) * mar.PI[:, 0], rtol=0, atol=1e-15
    )
    assert np.min(np.abs(tables["pi"] - mar.PI)) > 0.04
    limit = study.stack_limit(study.L1, **tables)
    assert limit["ey0"] == pytest.approx(TRUTH["ey0"], abs=1e-9)
    assert abs(limit["ey_obs"] - TRUTH["ey_obs"]) > 0.1


def test_the_treatment_wrong_cell_biases_the_reference_path_only() -> None:
    limit = study.stack_limit(study.L1, **properties.CONFIGURATIONS["treatment_wrong"])
    assert limit["ey_obs"] == pytest.approx(TRUTH["ey_obs"], abs=1e-9)
    assert abs(limit["ey0"] - TRUTH["ey0"]) > 0.1


@pytest.mark.parametrize(("name", "ratio"), [("par", 1.9507), ("paf", 1.7152)])
def test_each_inflated_se_control_clears_the_calibration_band(name: str, ratio: float) -> None:
    covariance = study.stack_covariance(study.L1)
    observed, reference = covariance[0, 0], covariance[1, 1]
    psi_obs, psi_ref = TRUTH["ey_obs"], TRUTH["ey0"]
    independent = (
        np.sqrt(observed + reference)
        if name == "par"
        else np.sqrt(reference / psi_obs**2 + observed * psi_ref**2 / psi_obs**4)
    )
    limit = independent / study.EFFICIENCY_SD[name]
    assert limit == pytest.approx(ratio, abs=1e-4)
    assert limit > study.STUDY.margins.calibration_se_ratio[1] + 0.5


@pytest.mark.parametrize(
    ("name", "displacement", "standardized"),
    [
        ("par", -0.029481, -1.957),
        ("paf", -0.057379, -1.785),
    ],
)
def test_each_complete_case_control_is_displaced(
    name: str, displacement: float, standardized: float
) -> None:
    observed = float(mar.observed_only_functional(mar.PROBS, "ey_obs"))
    reference = float(mar.observed_only_functional(mar.PROBS, "ey0"))
    limit = observed - reference if name == "par" else 1.0 - reference / observed
    moved = limit - TRUTH[name]
    assert moved == pytest.approx(displacement, abs=1e-6)
    assert moved / _per_replication_sd(name) == pytest.approx(standardized, abs=1e-3)
    assert abs(moved) / _per_replication_sd(name) > properties.MISSINGNESS_DISPLACEMENT


def test_the_untargeted_control_is_displaced() -> None:
    wrong = properties.WRONG_Q
    psi_obs = float(study.L1.p_w @ (study.L1.g * wrong).sum(axis=1))
    psi_ref = float(study.L1.p_w @ wrong[:, 0])
    moved = (psi_obs - psi_ref) - TRUTH["par"]
    assert moved == pytest.approx(-0.228, abs=1e-9)
    assert abs(moved) / _per_replication_sd("par") > 15.0
