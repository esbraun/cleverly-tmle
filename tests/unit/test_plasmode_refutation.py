"""The point plasmode of X82 plan 5.6: its law, its exact truth, and its readings."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from cleverly import ATE
from cleverly.exceptions import CapabilityError
from cleverly.validation import DEFAULT_TESTS, GaussianAdjustmentOutcome, GaussianNoise, refute
from cleverly.validation._point_generated import PointTarget
from cleverly.validation._readings import EquivalenceBands, band_readings
from cleverly.validation.refute import _PlasmodeLaw
from tests.unit.test_generated_outcome_refutation import _linear_ate_fit

#: A strong covariate signal against unit noise, so a law that holds W fixed visibly drops
#: the covariate variance from the spread of the draws.
PROCESS = GaussianAdjustmentOutcome(
    effect=1.0, adjustment_scale=3.0, noise=GaussianNoise(mean=0.5, standard_deviation=1.0)
)

TARGET = PointTarget("difference", "all", 1.0, 0.0)


@pytest.fixture(scope="module")
def result() -> Any:
    return _linear_ate_fit(ATE())


def _mean_readings(law: _PlasmodeLaw, data: Any, *, draws: int, hold: bool) -> Any:
    """Read the stub ``mean(Y*)`` with its i.i.d. standard error over ``draws`` plasmodes.

    ``hold=True`` is the mutation: it keeps every unit's ``W`` instead of resampling it.
    """
    n = data.n
    estimates, errors = [], []
    for seed in range(draws):
        if hold:
            rng = np.random.default_rng(seed)
            arm = rng.random(n) < law.treated
            treatment = np.where(arm, law.arm_codes[1], law.arm_codes[0])
            y = (
                law.adjustment(PROCESS, np.arange(n))
                + PROCESS.effect * treatment
                + PROCESS.noise.draw(rng, n)
            )
        else:
            y = np.asarray(law.draw(data, PROCESS, seed=seed, target=TARGET).outcome)
        estimates.append(float(np.mean(y)))
        errors.append(float(np.std(y, ddof=1) / np.sqrt(n)))
    # The exact truth is a finite sum over the n support points: the adjustment averages to
    # 0 on the original data, and the arm term averages the frozen treatment model.
    truth = float(
        np.mean(law.adjustment(PROCESS, np.arange(n))) + np.mean(law.treated) + PROCESS.noise.mean
    )
    return band_readings(estimates, errors, truth, EquivalenceBands())


class TestPlasmodeLaw:
    def test_the_truth_is_the_finite_sum_over_the_support(self, result: Any) -> None:
        law = _PlasmodeLaw.fit(result)
        support = law.adjustment(PROCESS, np.arange(result.data.n))
        assert abs(float(np.mean(support))) < 1e-12
        # A row drawn twice gets the same adjustment: the function is frozen on the data.
        rows = np.array([3, 3, 7])
        values = law.adjustment(PROCESS, rows)
        assert values[0] == values[1] == support[3] and values[2] == support[7]

    def test_a_stub_with_the_iid_standard_error_passes_every_band(self, result: Any) -> None:
        law = _PlasmodeLaw.fit(result)
        readings = _mean_readings(law, result.data, draws=400, hold=False)
        assert all(reading.inside for reading in readings), readings

    def test_a_law_that_holds_w_fails_the_standard_error_ratio(self, result: Any) -> None:
        law = _PlasmodeLaw.fit(result)
        _, ratio, _ = _mean_readings(law, result.data, draws=400, hold=True)
        assert not ratio.inside and ratio.estimate > 1.15

    def test_the_treatment_model_respects_the_fit_bounds(self, result: Any) -> None:
        law = _PlasmodeLaw.fit(result)
        lower, upper = result.config.g_bounds
        assert np.all(law.treated >= lower) and np.all(law.treated <= upper)


class TestPlasmodeRefutation:
    def test_it_is_opt_in(self) -> None:
        assert "plasmode" not in DEFAULT_TESTS

    def test_a_real_fit_reads_all_three_bands_at_l2(self, result: Any) -> None:
        test = refute(result, tests=("plasmode",), n_replicates=400, random_state=2)["plasmode"]
        assert [reading.name for reading in test.readings] == ["bias", "se_ratio", "coverage"]
        assert test.rung == "L2" and test.passed is True
        assert "resampled covariates" in test.scope_sentence

    def test_below_the_calibrated_draws_it_prints_at_l1(self, result: Any) -> None:
        test = refute(result, tests=("plasmode",), n_replicates=20, random_state=2)["plasmode"]
        assert test.rung == "L1" and test.passed is None

    def test_a_clustered_fit_is_refused(self, result: Any) -> None:
        clustered = replace(
            result, data=replace(result.data, cluster=np.arange(result.data.n) // 2)
        )
        with pytest.raises(CapabilityError, match="clusters"):
            refute(clustered, tests=("plasmode",), n_replicates=20, random_state=2)


class TestPlasmodeScope:
    def test_a_continuous_treatment_is_refused_before_any_refit(self) -> None:
        """The law draws arms. A continuous treatment has no arm target."""
        from types import SimpleNamespace

        from tests.unit.test_generated_outcome_refutation import _eligible_result

        result = _eligible_result()
        result.data = SimpleNamespace(**vars(result.data))
        result.data.n = len(result.data.outcome)
        result.data.is_continuous_treatment = True
        result.data.has_missing_outcome = False
        result.data.has_intermediate = False
        with pytest.raises(CapabilityError, match="continuous treatment"):
            refute(result, tests=("plasmode",), n_replicates=10)
        assert result.estimator.calls == []

    def test_a_separable_treatment_model_stays_inside_its_clip(self, result: Any) -> None:
        from types import SimpleNamespace

        separable = np.asarray(result.data.covariates[:, 0] > 0.0, dtype=float)
        stand_in = SimpleNamespace(
            data=replace(result.data, treatment=separable),
            config=SimpleNamespace(g_bounds=None),
        )
        law = _PlasmodeLaw.fit(stand_in)
        assert law.bounds == (0.01, 0.99)
        assert law.treated.min() == pytest.approx(0.01)
        assert law.treated.max() == pytest.approx(0.99)

    def test_the_scope_sentence_names_the_clip(self, result: Any) -> None:
        test = refute(result, tests=("plasmode",), n_replicates=20, random_state=2)["plasmode"]
        lower, upper = result.config.g_bounds
        assert f"clipped to [{lower:g}, {upper:g}]" in test.scope_sentence

    def test_a_weighted_fit_is_refused(self, result: Any) -> None:
        weighted = replace(result, data=replace(result.data, weights_name="w"))
        with pytest.raises(CapabilityError, match="weights"):
            refute(weighted, tests=("plasmode",), n_replicates=20, random_state=2)

    def test_the_adjustment_is_the_process_function_with_a_frozen_standardization(
        self, result: Any
    ) -> None:
        law = _PlasmodeLaw.fit(result)
        rows = np.arange(result.data.n)
        np.testing.assert_allclose(
            law.adjustment(PROCESS, rows), PROCESS.adjustment(result.data.covariates)
        )
