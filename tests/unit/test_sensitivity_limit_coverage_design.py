"""The pre-run design test of ``ovb-limit-coverage`` and ``evalue-limit-coverage`` (X82 SG).

``X82-SG-plan.md`` rev 2, section 2.2, declares five checks before the run:

* the closed-form truth of every O row is the package's bound on the law itself, which the
  weighted exact frame is;
* no O cell is degenerate: each bound's half-width is above 0.02;
* each O bound reports the cell's declared rule as its only feature;
* one sampled draw per cell records the saturated-GLM configuration that ``cells.csv`` carries;
* one replicate of every cell runs through ``draw_and_fit`` and gives finite rows.

The MSM Gram term has no study control (section 5.1).  Its witness is
``TestTheGramMatrixCurve.test_the_curve_is_the_gateaux_derivative`` in
``tests/unit/test_omitted_variable_axes.py``.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from tests.studies import evalue_limit_coverage as evalue_study
from tests.studies import ovb_limit_coverage as ovb_study
from tests.studies import sensitivity_limit_laws as laws
from tests.studies import stratified_law

#: The optimizer of the saturated logistic fits stops near 1e-7 in each coefficient, so the
#: exact-frame bound matches its closed form to about 1e-6 (largest gap measured: 1.2e-6).
EXACT_TOLERANCE = 1e-5

#: The smallest half-width an O bound may have (section 2.2).
DEGENERACY_FLOOR = 0.02

#: The fingerprint each O nuisance must carry.
SATURATED = "PolynomialFeatures(include_bias=False, interaction_only=True)"


@pytest.fixture(autouse=True)
def _quiet() -> None:
    # The O5 dose takes four values; the package warns that a declared continuous treatment
    # with so few values could be categorical.  The declaration is deliberate.
    warnings.filterwarnings("ignore", message="A was declared continuous")


class TestTheLaws:
    @pytest.mark.parametrize("version", laws.VERSIONS)
    def test_each_exact_frame_is_a_probability_law(self, version: str) -> None:
        frame = laws.exact_frame(version)
        assert frame["weight"].sum() == pytest.approx(1.0, abs=1e-12)
        assert (frame["weight"] > 0).all()

    def test_the_shift_frame_is_a_probability_law(self) -> None:
        frame = laws.shift_exact_frame()
        assert frame["weight"].sum() == pytest.approx(1.0, abs=1e-12)

    def test_the_response_mask_moves_sigma2(self) -> None:
        # The O1 control reads sigma^2 over the observed rows only, which divides the masked
        # value by the observed share.  It is a defect only if that share is well below 1.
        share = sum(
            laws.P_W[w] * (laws.G1[w] * laws.PI[w, 1] + (1.0 - laws.G1[w]) * laws.PI[w, 0])
            for w in range(3)
        )
        frame = laws.exact_frame("response")
        assert frame.loc[frame["D"] == 1.0, "weight"].sum() == pytest.approx(share, abs=1e-12)
        assert share < 0.8


@pytest.mark.parametrize("cell", list(ovb_study.CELLS))
class TestTheOCells:
    def test_the_truth_is_the_package_bound_on_the_law(self, cell: str) -> None:
        fitted = ovb_study.fit(cell, ovb_study._frame(cell), weights="weight")
        truth = ovb_study.truths(cell)
        for label in ovb_study.CELLS[cell].parameters:
            values = ovb_study.bound_values(cell, fitted, label)
            for end in ovb_study.ENDS:
                assert values[end][0] == pytest.approx(
                    truth[f"{label}_{end}"], abs=EXACT_TOLERANCE
                ), (cell, label, end)

    def test_no_bound_is_degenerate(self, cell: str) -> None:
        truth = ovb_study.truths(cell)
        for label in ovb_study.CELLS[cell].parameters:
            half = (truth[f"{label}_upper"] - truth[f"{label}_lower"]) / 2.0
            assert half > DEGENERACY_FLOOR, (cell, label, half)

    def test_a_sampled_bound_reports_the_declared_rule(self, cell: str) -> None:
        frame, _ = ovb_study.draw_from_seed(cell, ovb_study.PRIMARY_N, seed=11)
        fitted = ovb_study.fit(cell, frame)
        for label in ovb_study.CELLS[cell].parameters:
            # bound_values raises unless the features are exactly (rule,).
            values = ovb_study.bound_values(cell, fitted, label)
            for end in ovb_study.ENDS:
                estimate, std_error = values[end]
                assert math.isfinite(estimate)
                assert std_error > 0.0


class TestTheECells:
    def test_the_stratum_truths_are_the_law_ratios(self) -> None:
        truth = evalue_study.truths("E3")
        for name, stratum in zip(evalue_study.STRATUM_RATIOS, stratified_law.STRATA, strict=True):
            one = stratified_law.truth(stratified_law.stratum_name("ey[1]", stratum))
            zero = stratified_law.truth(stratified_law.stratum_name("ey[0]", stratum))
            assert truth[name] == pytest.approx(one / zero, rel=1e-12)
            # A null ratio would make the E-value limit trivially 1.
            assert abs(math.log(truth[name])) > 0.05

    def test_the_survival_truth_is_not_null(self) -> None:
        (truth,) = evalue_study.truths("E4").values()
        assert abs(math.log(truth)) > 0.05

    @pytest.mark.parametrize("scenario", list(evalue_study.SCENARIOS))
    def test_each_row_reads_a_log_scale_interval(self, scenario: str) -> None:
        frame, _ = evalue_study.draw_from_seed(scenario, evalue_study.PRIMARY_N, seed=11)
        estimates = evalue_study.estimates(scenario, frame)
        assert tuple(estimates) == evalue_study.SCENARIOS[scenario]
        for estimate in estimates.values():
            assert estimate.log_psi is not None
            assert math.isclose(math.exp(estimate.log_psi), estimate.psi, rel_tol=1e-9)


class TestTheConfiguration:
    def test_every_o_row_records_the_saturated_glms(self) -> None:
        cells = ovb_study.cell_configurations()
        expected = {
            (cell, label) for cell, spec in ovb_study.CELLS.items() for label in spec.parameters
        }
        assert set(zip(cells["scenario"], cells["parameter"], strict=True)) == expected
        for _, row in cells.iterrows():
            learners = row["learners"].split(" | ")
            assert len(learners) >= 2
            assert all(SATURATED in learner for learner in learners)
            assert row["cross_fitting"] == "off"
            assert row["method"] == "TMLE"
            assert row["n"] == ovb_study.PRIMARY_N

    def test_every_e_row_records_its_registered_fit(self) -> None:
        cells = evalue_study.cell_configurations()
        assert list(cells["parameter"]) == [
            row for rows in evalue_study.SCENARIOS.values() for row in rows
        ]
        assert set(cells["cross_fitting"]) == {"off"}
        assert dict(zip(cells["scenario"], cells["method"], strict=True)) == {
            "E3": "TMLE",
            "E4": "LTMLE",
        }


@pytest.mark.parametrize("study", [ovb_study, evalue_study], ids=["ovb", "evalue"])
def test_one_replicate_of_every_cell_gives_finite_rows(study: object) -> None:
    phase = study.draw_and_fit(replicates=1, n=study.PRIMARY_N, n_jobs=1)  # type: ignore[attr-defined]
    rows = phase.rows
    declared = {
        (scenario, row)
        for scenario, rows_of in study.SCENARIOS.items()  # type: ignore[attr-defined]
        for row in rows_of
    }
    assert set(zip(rows["scenario"], rows["estimand"], strict=True)) == declared
    assert np.isfinite(rows["estimate"]).all()
    assert (rows["std_error"] > 0).all()
