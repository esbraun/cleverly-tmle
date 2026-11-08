"""The ``separable-effects`` declaration: truths, limits, witnesses and cells.

The truths are checked against a fit on the law's exact sample, and each robustness cell's
limit against the same exact sample.  Nothing here reads a study run or the pilot.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from tests.studies import separable_effects as study
from tests.studies import separable_effects_law as law
from tests.studies import separable_effects_pilot as pilot
from tests.studies import separable_effects_properties as properties
from tests.studies.evidence.seeds import replicate_seed, stream_seed

#: The robustness cells' size, in efficient standard errors of one fit.
SCALE = study.EFFICIENCY_SD[properties.ROBUSTNESS_ESTIMAND] / np.sqrt(properties.ROBUSTNESS_N)


def _quiet(function: Any, *args: Any, **kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


@pytest.fixture(scope="module")
def limits() -> dict[str, float]:
    """Each robustness cell's estimate on the exact sample."""
    frame = law.exact_frame()
    return {
        cell: float(
            _quiet(properties.robustness_fit, frame, cell)[properties.ROBUSTNESS_ESTIMAND].psi
        )
        for cell in properties.ROBUSTNESS_CELLS
    }


class TestTheTruths:
    def test_a_fit_on_the_exact_sample_returns_identification_8(self) -> None:
        result = _quiet(study.fit, law.exact_frame(), study.IN_SAMPLE)
        for name in study.ESTIMANDS[study.IN_SAMPLE]:
            assert result[name].psi == pytest.approx(study.TRUTH[study.IN_SAMPLE][name], abs=1e-10)

    def test_a_cross_fitted_fit_on_the_exact_sample_lands_near_the_truth(self) -> None:
        """Each fold of the exact sample is not exact; the largest miss is 0.13 SE at 16k."""
        result = _quiet(study.fit, law.exact_frame(), study.CROSS_FITTED)
        for name in study.ESTIMANDS[study.CROSS_FITTED]:
            scale = study.EFFICIENCY_SD[name] / np.sqrt(study.PRIMARY_N)
            miss = result[name].psi - study.TRUTH[study.CROSS_FITTED][name]
            assert abs(miss) < 0.25 * scale, name

    def test_each_separable_risk_sits_away_from_both_static_risks(self) -> None:
        """The witness of the decomposition: several efficient SEs from either arm at 16k."""
        for label in law.SEPARABLE_ARMS:
            for cause in law.CAUSES:
                name = f"cif_regimen[{label}, {cause} @ t=3]"
                scale = study.EFFICIENCY_SD[name] / np.sqrt(study.PRIMARY_N)
                for arm in law.STATIC:
                    static = law.TRUTH[f"cif_regimen[{arm}, {cause} @ t=3]"]
                    assert abs(law.TRUTH[name] - static) > 3 * scale

    def test_no_reported_contrast_is_degenerate(self) -> None:
        assert min(study.EFFICIENCY_SD.values()) > 0.1

    def test_the_bounds_never_bind(self) -> None:
        # g >= 0.25 and each node keeps at least half its units uncensored, so the product
        # through three nodes is at least 0.25 * 0.5**3, far above the bound 1e-3.
        assert np.min(1.0 / law.exact_inverse(law.exact_frame())) >= 0.25 * 0.5**3
        assert study.G_BOUNDS[0] < 0.25 * 0.5**3


class TestTheRobustnessCells:
    def test_a_wrong_factor_with_the_exact_mechanism_has_the_truth_as_limit(
        self, limits: dict[str, float]
    ) -> None:
        truth = law.TRUTH[properties.ROBUSTNESS_ESTIMAND]
        for cell, spec in properties.ROBUSTNESS_CELLS.items():
            if spec is not None and spec[2] == "positive":
                assert limits[cell] == pytest.approx(truth, abs=1e-8), cell

    def test_the_wrong_mechanism_cell_lands_near_the_truth(self, limits: dict[str, float]) -> None:
        """Cross-fitted on the exact sample, so only a small miss is checked (see the truths)."""
        miss = (
            limits["sep10_relapse_t3__mechanism_wrong"] - law.TRUTH[properties.ROBUSTNESS_ESTIMAND]
        )
        assert abs(miss) < 0.25 * SCALE

    def test_in_sample_the_wrong_mechanism_changes_no_estimate(self) -> None:
        """Saturated in-sample hazards already solve every score, so the cell is cross-fitted."""
        from sklearn.dummy import DummyClassifier

        frame = law.sample(4_000, 11)
        wrong = _quiet(study.fit, frame, study.IN_SAMPLE, treatment_learner=DummyClassifier())
        right = _quiet(study.fit, frame, study.IN_SAMPLE)
        name = properties.ROBUSTNESS_ESTIMAND
        assert wrong[name].psi == pytest.approx(right[name].psi, abs=1e-12)

    def test_cross_fitted_the_wrong_mechanism_moves_the_estimate(self) -> None:
        """The nonzero witness: out of fold the treatment model changes the targeting."""
        frame = law.sample(4_000, 11)
        wrong = _quiet(properties.robustness_fit, frame, "sep10_relapse_t3__mechanism_wrong")
        right = _quiet(study.fit, frame, study.CROSS_FITTED)
        name = properties.ROBUSTNESS_ESTIMAND
        assert abs(wrong[name].psi - right[name].psi) > 1e-6

    def test_both_factors_wrong_misses_by_more_than_one_and_a_half_ses(
        self, limits: dict[str, float]
    ) -> None:
        truth = law.TRUTH[properties.ROBUSTNESS_ESTIMAND]
        miss = limits["sep10_relapse_t3__both_hazards_wrong"] - truth
        assert abs(miss) > 1.5 * SCALE

    @pytest.mark.parametrize(
        "cell",
        ["sep10_relapse_t3__event_hazard_wrong", "sep10_relapse_t3__competing_hazard_wrong"],
    )
    def test_the_targeting_step_moves_a_wrong_start_by_more_than_one_se(self, cell: str) -> None:
        """The untargeted control: the shifted start sits away from the truth."""
        shift_d, shift_y, _ = properties.ROBUSTNESS_CELLS[cell]  # type: ignore[misc]
        frame = law.exact_frame()
        inputs, start = law.engine_inputs(frame)
        zero = np.zeros_like(start.beta_d)
        shifted = properties.SeparableState(
            start.beta_d + (law.logit_shift(frame, law.SHIFT_D) if shift_d else zero),
            start.beta_y + (law.logit_shift(frame, law.SHIFT_Y) if shift_y else zero),
        )
        before = properties.influence(shifted, inputs, [properties.TARGET])[0][0]
        assert abs(before - law.TRUTH[properties.ROBUSTNESS_ESTIMAND]) > SCALE

    def test_one_target_cannot_pin_a_factor(self) -> None:
        """Design R2: fewer targets than a factor's (W, node) cells at a level."""
        targets = [properties.TARGET]
        assert len(targets) < 2 * len(law.HORIZONS)

    def test_the_engine_cells_read_the_exact_mechanism(self) -> None:
        frame = law.sample(400, 0)
        inputs, _ = law.engine_inputs(frame)
        np.testing.assert_allclose(inputs.inverse, law.exact_inverse(frame))


class TestTheFailures:
    def test_a_converged_fit_passes_and_a_stalled_one_is_recorded(self) -> None:
        from cleverly.validation._failures import recordable

        result = _quiet(study.fit, law.sample(4_000, 3), study.IN_SAMPLE)
        assert study.require_converged(result) is result
        stalled = _quiet(properties.engine_fit, law.sample(4_000, 3), True, True)
        object.__setattr__(stalled.record, "converged", False)
        object.__setattr__(stalled.record, "failure", "newton_stalled")
        with pytest.raises(Exception, match="newton_stalled") as caught:
            study.require_converged(stalled)
        assert recordable(caught.value)


class TestTheCells:
    def test_every_published_cell_is_declared(self) -> None:
        assert set(study.PROPERTY_CELLS["double_robustness"]) == set(properties.ROBUSTNESS_CELLS)
        declared = {cell for _, cell, _, _ in properties.declared_truths()}
        for family in study.PROPERTY_CELLS.values():
            assert set(family) <= declared

    def test_the_record_lists_every_study_specific_module(self) -> None:
        from tests.unit.test_study_provenance import _required_from

        required = _required_from(study.STUDY.runner_module, study.STUDY.properties_module)
        assert required <= set(study.STUDY.modules)

    def test_the_record_declares_every_cell_the_module_writes(self) -> None:
        written = {(family, cell) for family, cell, _, _ in properties.declared_truths()}
        declared = {
            (family, cell) for family, cells in study.STUDY.property_cells.items() for cell in cells
        }
        assert written == declared

    def test_each_key_names_the_cells_it_fills(self) -> None:
        names = {name for key in properties._keys(2) for name in properties._cells_of(key)}
        assert names == {
            f"{family}/{cell}"
            for family, cells in study.STUDY.property_cells.items()
            for cell in cells
        }

    def test_the_sample_seeds_are_distinct(self) -> None:
        seeds = [
            replicate_seed(study.STUDY, scenario, r)
            for scenario in study.SCENARIOS
            for r in range(50)
        ]
        seeds += [
            stream_seed(study.STUDY, "property_sample", family, label, r)
            for family, label in {(k[0], k[1]) for k in properties._keys(1)}
            for r in range(50)
        ]
        seeds += [
            stream_seed(study.STUDY, "pilot", label, r) for label in pilot.PILOTS for r in range(50)
        ]
        assert len(set(seeds)) == len(seeds)

    def test_the_seed_streams_are_this_studys_own(self) -> None:
        from tests.studies.evidence.registry import registered

        others = [record for record in registered() if record.slug != study.STUDY.slug]
        assert study.SEED not in {record.seed for record in others}
        assert study.RESAMPLING_SEED not in {record.resampling_seed for record in others}
