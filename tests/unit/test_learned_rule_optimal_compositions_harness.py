"""The harness, the rows and the property path of ``learned-rule-optimal-compositions``.

Small draws only: these tests check what each row reads and that every declared cell is written
and summarized, not any repeated-sampling property.
"""

from __future__ import annotations

import warnings
from functools import cache

import numpy as np
import pandas as pd
import pytest

from tests.studies import _learned_rule_optimal_compositions_law as law
from tests.studies import learned_rule_optimal_compositions as study
from tests.studies import learned_rule_optimal_compositions_properties as properties

pytestmark = pytest.mark.xdist_group("learned_rule_optimal_compositions_harness")

base = law.base
N = 600


@cache
def _measured(setting: str) -> law.Measured:
    return law.measured(law.draw(setting, N, 7), setting, 11)


@cache
def _property_rows() -> pd.DataFrame:
    return properties.generate_property_rows(n_jobs=1, record=base.smoke_record(study.STUDY), cap=2)


class TestTheLaws:
    def test_the_rm30_settings_draw_rm30s_sample(self) -> None:
        for setting in ("strata", "known_regimes", "in_sample"):
            pd.testing.assert_frame_equal(
                law.draw(setting, 50, 3), base.draw("non_exceptional", 50, 3)
            )

    def test_the_weighted_draw_carries_the_weight(self) -> None:
        frame = law.draw("weighted", 200, 3)
        expected = 1.0 + frame["A"] * frame["Y"]
        np.testing.assert_array_equal(frame["w"].to_numpy(), expected.to_numpy())

    def test_only_the_weighted_setting_passes_the_weighted_mechanism(self) -> None:
        frame = law.draw("weighted", 200, 3)
        weighted = law.known_probabilities(frame, "weighted")
        design = law.known_probabilities(frame, "strata")
        np.testing.assert_allclose(weighted.sum(axis=1), 1.0)
        assert (weighted[:, 1] > design[:, 1]).all()
        np.testing.assert_array_equal(design, law.optimal.known_probabilities(frame))

    def test_the_intermediate_draw_is_the_intermediate_laws(self) -> None:
        pd.testing.assert_frame_equal(law.draw("intermediate", 50, 3), law.zlaw.draw(50, 3))

    def test_the_contrast_truth_is_the_value_minus_the_static_value(self) -> None:
        """The static value is the contrast law's trapezoid of a smooth integrand (1e-9)."""
        limits = law.label_limits()
        for arm in (1, 0):
            assert limits[f"contrast_{arm}"]["truth"] == pytest.approx(
                limits["strata_all"]["truth"] - law.contrast.static_value(float(arm), law.LAW),
                abs=1e-8,
            )

    def test_a_rule_of_nobody_has_the_untreated_value(self) -> None:
        nobody = {w2: np.zeros(base.GRID.size, dtype=bool) for w2 in base.W2_LEVELS}
        values = law.rule_values("known_regimes", nobody)
        assert values["contrast_0"] == pytest.approx(0.0, abs=1e-12)


class TestTheFits:
    @pytest.mark.parametrize("setting", law.SETTINGS)
    def test_every_row_is_checked_and_finite(self, setting: str) -> None:
        fitted = _measured(setting)
        for label in law.PARAMETERS[setting]:
            columns = law.harness_columns(fitted, label)
            assert columns["rule_rows_checked"] == N
            assert np.isfinite(fitted.estimate(label).psi)
            assert columns["regret"] == pytest.approx(
                law.truth(label) - columns["rule_value"], abs=1e-15
            )

    @pytest.mark.parametrize("setting", law.SETTINGS)
    def test_every_fit_uses_the_known_mechanism(self, setting: str) -> None:
        for result in _measured(setting).results.values():
            assert "known_mechanism" in result.config.support_key.qualifiers

    @pytest.mark.parametrize("setting", law.SETTINGS)
    def test_every_fit_reads_its_registered_row(self, setting: str) -> None:
        """The registered row ``T55.<setting>`` covers every estimate of the study's own fit."""
        for result in _measured(setting).results.values():
            for estimate in result.estimates.values():
                assert estimate.support.label == "supported", setting
                assert estimate.support.row == f"T55.{setting}", setting

    def test_a_fit_with_other_parameters_stops(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(law.REPORTED, "weighted", ("something else",))
        with pytest.raises(base.HarnessMismatch, match="reports"):
            law.measured(law.draw("weighted", N, 7), "weighted", 11)

    def test_a_refit_rule_that_differs_stops(self, monkeypatch: pytest.MonkeyPatch) -> None:
        refit = base.fold_rule

        def flipped(*args: object, **kwargs: object) -> object:
            rule = refit(*args, **kwargs)  # type: ignore[arg-type]
            return lambda covariates: ~rule(covariates)

        monkeypatch.setattr(base, "fold_rule", flipped)
        with pytest.raises(base.HarnessMismatch, match="differs"):
            law.measured(law.draw("strata", N, 7), "strata", 11)


class TestThePrimaryRows:
    def test_each_setting_writes_one_row_per_parameter(self) -> None:
        record = base.smoke_record(study.STUDY)
        for setting in law.SETTINGS:
            frame = law.draw(setting, N, 5)
            rows = study._primary(record, (setting, 0, N), frame)
            assert [row["estimand"] for row in rows] == list(study.SCENARIOS[setting])
            for row in rows:
                label = {v: k for k, v in law.PARAMETERS[setting].items()}[row["estimand"]]
                assert row["truth"] == law.truth(label)

    def test_the_draw_carries_each_truth(self) -> None:
        _, truths = study.draw_from_seed("strata", 50, 1)
        assert truths == {
            name: law.truth(label) for label, name in law.PARAMETERS["strata"].items()
        }


class TestThePropertyPath:
    def test_every_declared_cell_is_written(self) -> None:
        rows = _property_rows()
        for family, cells in study.PROPERTY_CELLS.items():
            assert set(rows.loc[rows["property"] == family, "cell"]) == set(cells)

    def test_each_target_control_reads_its_truth(self) -> None:
        rows = _property_rows()
        control = rows.loc[rows["property"] == study.TARGET_FAMILY]
        truths = {
            f"weighted__{study.TARGET_CONTROL}": law.unweighted_truth(),
            f"weighted__{study.RULE_CONTROL}": law.unweighted_rule_truth(),
        }
        assert set(control["cell"]) == set(truths)
        for cell, truth in truths.items():
            assert (control.loc[control["cell"] == cell, "truth"] == truth).all()

    def test_the_unnormalised_control_halves_the_stratum_se(self) -> None:
        rows = _property_rows()
        for label in study.STRATUM_LABELS:
            positive = rows.loc[rows["cell"] == f"{label}__declared"].set_index("replicate")
            control = rows.loc[rows["cell"] == f"{label}__{study.UNNORMALISED}"].set_index(
                "replicate"
            )
            ratio = control["std_error"] / positive["std_error"]
            assert ((ratio > 0.4) & (ratio < 0.6)).all(), ratio
            assert (control["estimate"] == positive["estimate"]).all()

    def test_the_summary_publishes_every_cell(self) -> None:
        # Two draws per cell: a bootstrap resample can repeat one draw and divide by zero.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            summary = properties.summarize_properties(_property_rows())
        published = set(zip(summary["property"], summary["cell"], strict=True))
        for family, cells in study.PROPERTY_CELLS.items():
            assert {(family, cell) for cell in cells} <= published
        assert {"mean_regret", "rule_value_standardized_bias"} <= set(summary.columns)

    def test_the_initial_estimate_is_finite_for_every_label(self) -> None:
        for setting in law.SETTINGS:
            fitted = _measured(setting)
            for label in law.PARAMETERS[setting]:
                initial = law.untargeted_estimate(fitted, label)
                estimate = fitted.estimate(label)
                assert abs(initial - estimate.psi) < 4 * estimate.std_error, (label, initial)

    def test_the_target_verdict_reads_the_declared_floor(self) -> None:
        summary = pd.DataFrame(
            {
                "property": [study.TARGET_FAMILY] * 2,
                "cell": ["a", "b"],
                "role": ["control"] * 2,
                "coverage_ci_upper": [0.919, 0.921],
                "passed": [False, False],
            }
        )
        properties.target_verdicts(summary, floor=0.92)
        assert summary["passed"].tolist() == [True, False]
