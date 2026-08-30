"""Focused contracts for the learned weighted point-treatment study."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from tests.studies import canonical_learned_weighted_tmle as study
from tests.studies import learned_weighted_point_common as law
from tests.studies import learned_weighted_tmle_properties as properties

DECLARED_BUDGETS: dict[str, tuple[int, int]] = {
    "interval_calibration": (
        properties.CALIBRATION_N,
        properties.CALIBRATION_REPLICATES,
    ),
    "type_i_error": (properties.NULL_N, properties.NULL_REPLICATES),
    "power": (properties.POWER_N, properties.POWER_REPLICATES),
    "learner_weight_necessity": (
        properties.NECESSITY_N,
        properties.NECESSITY_REPLICATES,
    ),
}


def test_declared_budgets_match_the_committed_property_cells() -> None:
    published = pd.read_csv(study.STUDY.artifact("properties.csv"))
    for family, (n, replicates) in DECLARED_BUDGETS.items():
        cells = published.loc[published["property"] == family]
        assert not cells.empty
        assert set(cells["n"]) == {n}
        assert set(cells["replicates"]) == {replicates}

    rate = published.loc[published["property"] == "root_n_and_efficiency"]
    assert set(rate["n"]) == set(properties.RATE_SIZES)
    assert set(rate["replicates"]) == {properties.RATE_REPLICATES}


def test_both_nuisance_learners_consume_the_declared_weights(monkeypatch: Any) -> None:
    """The positive fit sends the density-ratio weights to both learner classes."""
    frame = law.sample_selected(300, 841)
    expected = frame["obs_weight"].to_numpy(dtype=float)
    expected = expected / expected.mean()
    linear_weights: list[np.ndarray] = []
    logistic_weights: list[np.ndarray] = []
    linear_fit = LinearRegression.fit
    logistic_fit = LogisticRegression.fit

    def record_linear(
        self: LinearRegression, X: Any, y: Any, sample_weight: Any = None
    ) -> LinearRegression:
        linear_weights.append(np.asarray(sample_weight, dtype=float))
        return linear_fit(self, X, y, sample_weight=sample_weight)

    def record_logistic(
        self: LogisticRegression, X: Any, y: Any, sample_weight: Any = None
    ) -> LogisticRegression:
        logistic_weights.append(np.asarray(sample_weight, dtype=float))
        return logistic_fit(self, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(LinearRegression, "fit", record_linear)
    monkeypatch.setattr(LogisticRegression, "fit", record_logistic)
    study.fit_cleverly(frame, estimands=("ate",))

    assert linear_weights
    assert logistic_weights
    assert any(np.allclose(weights, expected, rtol=1e-15, atol=0.0) for weights in linear_weights)
    assert any(np.allclose(weights, expected, rtol=1e-15, atol=0.0) for weights in logistic_weights)


def test_selected_density_weights_recover_the_uniform_target_law() -> None:
    grid = np.linspace(-1.0, 1.0, 10_001)
    recovered = law.selected_density(grid) * law.inverse_selection_weight(grid)
    np.testing.assert_allclose(recovered, 0.5, rtol=0, atol=1e-15)
    assert law.truths() == {"ey0": 0.5, "ey1": 1.5, "ate": 1.0}
    assert law.SELECTED_W1_MEAN == 0.25
    assert law.SELECTED_ATE == 1.5


def test_the_efficiency_bound_matches_a_monte_carlo_of_the_influence_curve() -> None:
    """Rebuild the declared curve from the sampler, and reject the unnormalized rival.

    The witness is built from :func:`law.sample_selected` and :func:`law.outcome_mean`, not
    from the closed form, so it can disagree with the closed form.
    """
    frame = law.sample_selected(1_000_000, 20_260_830)
    treatment = frame["A"].to_numpy(dtype=float)
    covariate = frame["W1"].to_numpy(dtype=float)
    weight = frame["obs_weight"].to_numpy(dtype=float)
    residual = frame["Y"].to_numpy(dtype=float) - law.outcome_mean(
        covariate, frame["W2"].to_numpy(dtype=float), treatment, effect=law.TARGET_ATE
    )
    probability = law.TREATMENT_PROBABILITY
    clever = treatment / probability - (1.0 - treatment) / (1.0 - probability)
    blip = law.TARGET_ATE + law.EFFECT_MODIFICATION * covariate

    bound = law.weighted_ate_efficiency_sd()
    inside = weight * (clever * residual + blip - law.TARGET_ATE)
    np.testing.assert_allclose(np.sqrt(np.mean(inside**2)), bound, rtol=0.02, atol=0.0)

    # The deliberate-mutation control.  ``h * (cc * (Y - Q) + b) - psi`` is the unnormalized
    # Horvitz-Thompson gradient, which centers outside the weight.  ``cleverly`` averages the
    # targeted predictions with weights, so its functional is ``E_sel[h * b] / E_sel[h]`` and
    # its gradient centers inside the weight.  This arm must miss the declared bound, or the
    # assertion above would pass for either convention.
    outside = weight * (clever * residual + blip) - law.TARGET_ATE
    assert abs(np.sqrt(np.mean(outside**2)) / bound - 1.0) > 0.05


#: The standard deviation the unnormalized Horvitz-Thompson gradient gives on this law.
UNNORMALIZED_EFFICIENCY_SD = 2.452519778942014


def test_the_reported_standard_error_scales_to_the_declared_bound() -> None:
    """The fitted standard error resolves the declared bound against its unnormalized rival."""
    n = 20_000
    result = study.fit_cleverly(law.sample_selected(n, 20_260_830), estimands=("ate",))["ate"]
    scaled = float(result.std_error) * math.sqrt(n)

    np.testing.assert_allclose(scaled, law.weighted_ate_efficiency_sd(), rtol=0.05, atol=0.0)
    assert abs(scaled / UNNORMALIZED_EFFICIENCY_SD - 1.0) > 0.05


def test_learner_weight_control_moves_only_the_untargeted_plugin() -> None:
    """The weighted target repairs the control after its nuisance plug-in moves."""
    rows = properties.fit_replication(("learner_weight_necessity", "paired", 0, 4_000, 1, 841))
    estimates = {row["cell"]: float(row["estimate"]) for row in rows}

    assert abs(estimates["ate__weighted_plugin"] - law.TARGET_ATE) < 0.10
    assert abs(estimates["ate__unweighted_plugin_control"] - law.SELECTED_ATE) < 0.10
    assert abs(estimates["ate__weighted_targeted"] - law.TARGET_ATE) < 0.10
    assert abs(estimates["ate__unweighted_targeted"] - law.TARGET_ATE) < 0.10
