"""The design of ``projection-methods`` (P1 PR 3 plan rev 2, section 5), without a run.

Every number plan 5.1 computed for law P is recomputed here by quadrature, and the arms, cells,
controls and readings are checked against the plan's tables.  The pilot rule, the point-reading
fallback and the ``pilot_2`` selection are checked on stated ratios until the pilot has run.  The
summary step is run on synthetic rows of every declared cell, and the cost gate reads the
committed smoke timings.
"""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.special import logit

from tests.studies import point_reading, repeated_sizing
from tests.studies import projection_methods as study
from tests.studies import projection_methods_law as law
from tests.studies import projection_methods_pilot as pilot
from tests.studies import projection_methods_properties as properties
from tests.studies.evidence.property_verdicts import DIAGNOSTIC_ROLE, ONE_WRONG_FAMILY
from tests.studies.evidence.registry import ROOT, registered

MARGINS = study.STUDY.margins


@cache
def numbers() -> dict[str, Any]:
    return study.design_numbers()


def table() -> dict[str, Any]:
    return numbers()["law"]


# ------------------------------------------------------------------------ seeds and record


def test_the_study_is_declared_and_not_registered() -> None:
    assert study.SLUG not in {record.slug for record in registered()}


def test_the_seeds_are_unique_against_every_registered_study_and_pilot() -> None:
    records = registered()
    taken = {r.seed for r in records} | {r.resampling_seed for r in records}
    pilots = set()
    for path in (ROOT / "tests").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"^PILOT_SEED\s*=\s*([0-9_]+)", text, flags=re.MULTILINE):
            pilots.add(int(match.group(1).replace("_", "")))
    assert len(pilots) > 5
    for seed in (study.SEED, study.RESAMPLING_SEED):
        assert seed not in taken
        assert seed not in pilots
    assert study.SEED == 20261531
    assert study.SEED != study.RESAMPLING_SEED


def test_each_stream_has_its_own_samples_and_folds_per_group() -> None:
    groups = [group.label for group in study.GROUPS]
    seen: set[int] = set()
    for stream in properties.STREAMS:
        samples = {properties.sample_seed(g, r, stream) for g in groups for r in range(100)}
        folds = {properties.fold_seed(g, r, stream) for g in groups for r in range(100)}
        assert len(samples) == len(folds) == 100 * len(groups)
        assert not (samples | folds) & seen
        assert not samples & folds
        seen |= samples | folds
    assert set(properties.STREAMS) == {"property", "pilot", "pilot_2", "smoke"}


def test_the_primary_is_the_dr_known_intercept_at_2400() -> None:
    assert study.STUDY.scenarios == {"dr_known_4k": ("cate_projection[(intercept)]",)}
    cell = study.calibration_cell(study.GROUP_BY_LABEL["dr_known_4k"], "(intercept)")
    summary = pd.DataFrame(
        {"cell": [cell], "covered": [int(0.95 * pilot.REPLICATES)], "replicates": pilot.REPLICATES}
    )
    assert pilot.primary_replicates(summary) <= study.PRIMARY_REPLICATES == 2_400
    assert study.PRIMARY_COVERAGE_FLOOR == 0.90


# --------------------------------------------------------------------------- arms and cells


def test_the_groups_are_the_plans() -> None:
    sizes = {g.label: (g.n, g.replicates, g.method, g.outcome, g.mechanism) for g in study.GROUPS}
    assert sizes == {
        "dr_known_4k": (4_000, study.DECLARED_R, "drtmle", "wrong", "known"),
        "dr_both_4k": (4_000, 2_400, "drtmle", "correct", "glm"),
        "dr_slow_1k": (1_000, 1_000, "drtmle", "wrong", "slow"),
        "dr_slow_4k": (4_000, 1_000, "drtmle", "wrong", "slow"),
        "dr_slow_16k": (16_000, 1_000, "drtmle", "wrong", "slow"),
        "dr_glm_4k": (4_000, 2_400, "drtmle", "wrong", "glm"),
        "ctmle_points_4k": (4_000, 2_400, "ctmle", "correct", "per_arm"),
    }
    assert study.DECLARED_R >= study.BUDGETED_R == 3_600


def test_the_arms_hold_one_coefficient_each() -> None:
    """Plan 6.1: the intercept and the slope can sit at different SE ratios."""
    assert numbers()["calibration_arms"] == {"__dr_known_intercept": 1, "__dr_known_slope": 1}
    assert study.pilot_groups() == ("dr_known_4k",)


def test_every_cell_is_declared_once_and_the_record_reads_them() -> None:
    cells = [cell for names in study.PROPERTY_CELLS.values() for cell in names]
    assert len(cells) == len(set(cells))
    assert study.STUDY.property_cells == study.PROPERTY_CELLS
    assert properties.PROPERTY_CELLS is study.PROPERTY_CELLS
    assert set(study.PROPERTY_CELLS) == {study.CALIBRATION, study.BIAS, ONE_WRONG_FAMILY, "spread"}
    positives = [c for c in study.PROPERTY_CELLS[study.BIAS] if not c.startswith("wrong_q")]
    assert positives == [
        "dr_known_4k_intercept",
        "dr_known_4k_W1",
        "dr_both_4k_intercept",
        "dr_both_4k_W1",
        "ctmle_points_4k_intercept",
        "ctmle_points_4k_W1",
    ]
    # Eight gated bias rows (plan 5.2): six bias cells and the two dr_glm cells' S1 clause.
    one_wrong = study.PROPERTY_CELLS[ONE_WRONG_FAMILY]
    assert len(positives) + sum(c.startswith("dr_glm__") for c in one_wrong) == 8
    assert sorted(one_wrong) == sorted(
        [
            *(f"dr_slow_n{n}__{t}__first_order" for n in law.LADDER for t in ("intercept", "W1")),
            "dr_slow__intercept__growth",
            "dr_slow__W1__growth",
            "dr_glm__intercept__conservative",
            "dr_glm__W1__valid",
        ]
    )
    for cell in one_wrong:
        assert len(cell.split("__")) == 3, cell
    assert study.PROPERTY_CELLS["spread"] == (
        "dr_known_4k_intercept__d_star_reading",
        "dr_known_4k_W1__d_star_reading",
        "dr_both_4k_intercept__plugin_reading",
        "dr_both_4k_W1__plugin_reading",
        "ctmle_points_4k_intercept__plugin_reading",
        "ctmle_points_4k_W1__plugin_reading",
    )
    assert "dr_known_4k_intercept__shrunken_se_control" in study.PROPERTY_CELLS[study.CALIBRATION]


def test_the_truths_name_every_reported_coefficient() -> None:
    for group in study.GROUPS:
        assert {group.name(t) for t in group.terms} == set(study.truth(group))


# ------------------------------------------------------------------------ plan 5.1 numbers


def test_the_law_p_readings_are_the_plans() -> None:
    t = table()
    assert t["beta"] == pytest.approx((0.07302, 0.08141), abs=1e-5)
    np.testing.assert_allclose(t["gram"], np.eye(2), atol=1e-10)
    assert t["wrong_limit"] == pytest.approx((-0.2473, 0.6678, -0.4731), abs=1e-4)
    assert t["efficient_sd"] == pytest.approx((0.0128, 0.0136), abs=1e-4)
    assert t["dr_known_sd"] == pytest.approx((0.0157, 0.0159), abs=1e-4)
    assert t["d_star_sd"] == pytest.approx((0.0167, 0.0177), abs=1e-4)
    assert t["plugin_displacement"] == pytest.approx((5.35, -4.95), abs=0.01)
    assert t["conditioning_norms"][1] == pytest.approx((0.259, 0.221), abs=1e-3)
    assert t["conditioning_norms"][0] == pytest.approx((0.282, 0.243), abs=1e-3)
    assert t["conservative_ratio"] == pytest.approx((1.19, 1.00), abs=0.005)
    assert t["correlation"] == pytest.approx(0.002, abs=0.001)
    assert law.quadrature_error() < 1e-8


def test_the_ladder_is_the_plans_and_each_other_offset_vanishes_on_a_term() -> None:
    ladder = table()["ladder"]
    np.testing.assert_allclose(
        ladder["product"], [[-0.71, -0.54], [-0.91, -0.69], [-1.15, -0.88]], atol=0.005
    )
    # The near-symmetries of plan 5.1: each single offset barely moves one term.
    assert np.abs(ladder["w2"][:, 1]).max() < 0.05
    assert np.abs(ladder["w1w2"][:, 0]).max() < 0.02
    assert np.abs(ladder["w1"][:, 1]).max() < 0.03


def test_every_declared_witness_clears_its_floor_before_the_pilot() -> None:
    """Plan 6.2: 0.4 SD for the ladder at n = 1,000 and 3 SD for the wrong-Qbar witness."""
    assert np.abs(table()["ladder"]["product"][0]).min() > 0.4
    assert np.abs(table()["plugin_displacement"]).min() > 3.0


def test_the_declared_signs_are_the_conditioning_terms() -> None:
    ladder = table()["ladder"]["product"]
    for j, term in enumerate(("intercept", "W1")):
        signs = set(np.sign(ladder[:, j]).astype(int))
        assert signs == {study.PREDICTED_SIGNS[f"dr_slow__{term}"]}
        # Growth: the magnitude rises along the ladder.
        assert np.all(np.diff(np.abs(ladder[:, j])) > 0)


# ------------------------------------------------------------- the gates and their powers


def test_every_ladder_rung_and_growth_cell_detects() -> None:
    """Plan 5.3: each rung fails the +-0.25 bound with probability above ``1 - 1e-9``."""
    ladder = table()["ladder"]["product"]
    replicates = study.LADDER_R
    smallest = float(np.abs(ladder).min())
    assert smallest == pytest.approx(0.54, abs=0.005)
    assert repeated_sizing.bias_discriminated_probability(MARGINS, 0.52, replicates) > 1 - 1e-8
    spread = float(np.sqrt(2.0 / replicates))  # two independent standardized biases
    z = float(stats.norm.ppf(0.5 + MARGINS.confidence_level / 2.0))
    growth = np.abs(ladder[-1]) - np.abs(ladder[0])
    assert growth == pytest.approx((0.44, 0.34), abs=0.01)
    assert spread == pytest.approx(0.045, abs=0.001)
    for value in growth:
        assert float(stats.norm.sf((z * spread - value) / spread)) > 0.999


def test_the_wrong_outcome_witness_fails_bias_on_both_terms() -> None:
    for gap in table()["plugin_displacement"]:
        probability = repeated_sizing.bias_discriminated_probability(
            MARGINS, abs(float(gap)), study.BUDGETED_R
        )
        assert probability > 1 - 1e-9


def test_the_bias_rows_pass_at_the_design_bias() -> None:
    """Plan 5.2: 0.999999 per row at 0.10 and 0.990 at 0.15; over 8 rows 0.99999 and 0.922."""
    at = np.asarray(
        repeated_sizing.bias_equivalent_probability(MARGINS, np.array([0.10, 0.15]), study.BIAS_R)
    )
    assert at[0] > 0.999999
    assert at[1] == pytest.approx(0.990, abs=0.001)
    assert at[0] ** 8 > 0.99999
    assert at[1] ** 8 == pytest.approx(0.922, abs=0.002)


def test_the_shrunken_control_lies_below_the_band() -> None:
    for replicates in (500, study.BUDGETED_R):
        probability = repeated_sizing.calibration_control_probability(
            MARGINS, replicates, study.SHRUNKEN_FACTOR
        )
        assert probability > 0.999999


def test_the_dr_glm_clauses_pass_at_their_planning_figures() -> None:
    """Plan 5.3: the intercept conservative, the slope valid, at R = 2,400."""
    predicted = table()["conservative_ratio"]
    for j, term in enumerate(study.TERMS):
        ratio = study.PLANNING_RATIO[term]
        assert ratio == pytest.approx(predicted[j], abs=0.005)
        coverage = 2.0 * stats.norm.cdf(1.959964 * ratio) - 1.0
        assert study.PLANNING_COVERAGE[term] == pytest.approx(coverage, abs=0.001)
        floor = repeated_sizing.coverage_floor_pass_probability(
            MARGINS, study.BIAS_R, study.PLANNING_COVERAGE[term]
        )
        assert floor > 0.9999
    assert study.DR_GLM_KINDS == {"(intercept)": "conservative", "W1": "valid"}
    assert study.conservative_pass_probability(study.PLANNING_RATIO["(intercept)"], 2_400) > 0.999
    assert study.sanity_pass_probability(study.PLANNING_RATIO["W1"], 2_400) > 0.9999
    # The spread the clause probabilities read (plan 5.3: 0.014 to 0.017 at R = 2,400).
    for ratio, expected in ((1.0, 0.0146), (1.19, 0.0174)):
        assert ratio * np.sqrt(0.51 / 2_400) == pytest.approx(expected, abs=5e-4)


def test_the_d_star_reading_is_published_not_gated() -> None:
    """Plan 5.3: 1.063 and 1.113; the slope lies above the band, the intercept inside."""
    ratios = table()["d_star_ratio"]
    assert ratios == pytest.approx((1.063, 1.113), abs=0.001)
    low, high = MARGINS.calibration_se_ratio
    assert low < ratios[0] < high < ratios[1]
    assert "d_star_reading" in study.READING_CLAIMS


def test_the_family_passes_at_the_budget_with_correct_intervals() -> None:
    single = float(repeated_sizing.calibration_pass(MARGINS, study.BUDGETED_R, np.array(1.0)))
    assert single ** sum(numbers()["calibration_arms"].values()) > 0.95


# --------------------------------------------------------------------------- the sizing


def _ratios(intercept: float, slope: float, *, one_arm: bool = False) -> dict[str, float]:
    group = study.GROUP_BY_LABEL["dr_known_4k"]
    cells = [study.calibration_cell(group, t) for t in study.TERMS]
    if one_arm:
        cells = [cell.rsplit("__", 1)[0] + "__dr_known" for cell in cells]
    return dict(zip(cells, (intercept, slope), strict=True))


def _correlations(rho: float) -> repeated_sizing.Correlations:
    def correlations(labels: Any) -> np.ndarray:
        k = len(labels)
        return np.full((k, k), rho) + (1.0 - rho) * np.eye(k)

    return correlations


@pytest.mark.parametrize(
    ("intercept", "slope", "expected"),
    [(1.00, 1.00, 3_600), (1.00, 0.98, 5_600), (1.00, 0.97, 10_800)],
)
def test_one_row_per_arm_sizes_the_plans_table(
    intercept: float, slope: float, expected: int
) -> None:
    replicates = repeated_sizing.calibration_replicates(
        MARGINS, _ratios(intercept, slope), pilot.REPLICATES, _correlations(0.002)
    )
    assert replicates == expected


def test_both_rows_in_one_arm_would_leave_no_budget() -> None:
    """Plan 6.1: ratios 1.00 and 0.98 in one arm read a spread of 0.054 and no budget."""
    ratios = _ratios(1.00, 0.98, one_arm=True)
    groups = repeated_sizing.calibration_groups(ratios, pilot.REPLICATES, _correlations(0.002))
    assert repeated_sizing.heterogeneity_bound(groups) == pytest.approx(0.054, abs=0.001)
    with pytest.raises(ValueError, match="no budget"):
        repeated_sizing.calibration_replicates(
            MARGINS, ratios, pilot.REPLICATES, _correlations(0.002)
        )


# ------------------------------------------------------------------- the pilot and fallback


def _pilot_frame(ratios: Any, se: float = 0.01, replicates: int = pilot.REPLICATES) -> pd.DataFrame:
    cells = pilot.positive_cells()
    return pd.DataFrame(
        {
            "cell": list(cells),
            "group": [study.group_of_cell(cell) for cell in cells],
            "arm": [pilot.arm_of(cell) for cell in cells],
            "replicates": replicates,
            "se_ratio": ratios,
            "se_ratio_se": se,
            "covered": int(0.95 * pilot.REPLICATES),
        }
    )


NO_PAIRS = pd.DataFrame(columns=["first", "second", "correlation"])

#: Both pilots' draws: the fallback runs only after ``pilot_2`` (plan 5.0).
BOTH = pilot.REPLICATES + pilot.EXTENSION


def test_the_pilot_rule_sizes_r_by_x32a_on_correct_ratios() -> None:
    assert pilot.declared_replicates(_pilot_frame(1.0), NO_PAIRS) == (study.BUDGETED_R, "x32a")
    if not pilot.PILOT_CSV.exists():
        pytest.skip("the pilot has not run; the declaration commit adds it (plan 5.0)")
    summary = pd.read_csv(pilot.PILOT_CSV)
    computed, _ = pilot.declared_replicates(summary, pd.read_csv(pilot.CORRELATIONS_CSV))
    assert computed == study.DECLARED_R == study.R


def test_inside_the_absorb_band_x32a_finds_a_budget_at_the_declared_pilot() -> None:
    """With one row per arm, no pilot point the fallback may absorb leaves X32a without a budget.

    The fallback therefore answers only a pilot whose ratios a diagnosis has to explain first.
    """
    low, high = point_reading.ABSORB_BAND
    for ratios in ([low, low], [low, high], [high, high]):
        replicates, rule = pilot.declared_replicates(_pilot_frame(ratios), NO_PAIRS)
        assert rule == "x32a"
        assert replicates <= repeated_sizing.MAXIMUM_REPLICATES


def _no_budget(monkeypatch: pytest.MonkeyPatch, message: str = pilot.NO_BUDGET) -> None:
    def refuse(summary: pd.DataFrame, pairs: pd.DataFrame) -> int:
        raise ValueError(message)

    monkeypatch.setattr(pilot, "x32a_replicates", refuse)


def test_the_point_reading_fallback_runs_where_x32a_cannot(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_budget(monkeypatch)
    frame = _pilot_frame([0.975, 1.025], replicates=BOTH)
    replicates, rule = pilot.declared_replicates(frame, NO_PAIRS)
    assert rule == "point_reading"
    ratios = dict(zip(frame["cell"], frame["se_ratio"], strict=True))
    errors = dict(zip(frame["cell"], frame["se_ratio_se"], strict=True))
    found = point_reading.point_reading_replicates(
        MARGINS, ratios, errors, target=study.POINT_READING_TARGET
    )
    assert replicates == max(found, study.BUDGETED_R)
    assert found % repeated_sizing.BUDGET_STEP == 0
    assert point_reading.point_reading_pass(MARGINS, ratios, found) >= 0.95
    below = found - repeated_sizing.BUDGET_STEP
    if below > 0:
        assert point_reading.point_reading_pass(MARGINS, ratios, below) < 0.95


def test_the_point_reading_fallback_refuses_a_defect_suspect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_budget(monkeypatch)
    frame = _pilot_frame([0.95, 1.05], replicates=BOTH)
    with pytest.raises(ValueError, match="outside"):
        pilot.declared_replicates(frame, NO_PAIRS, diagnosed=pilot.positive_cells())
    tight = _pilot_frame([0.975, 1.025], se=0.002, replicates=BOTH)
    with pytest.raises(ValueError, match="no diagnosis"):
        pilot.declared_replicates(tight, NO_PAIRS)
    diagnosed = pilot.declared_replicates(tight, NO_PAIRS, diagnosed=pilot.positive_cells())
    assert diagnosed[1] == "point_reading"
    assert point_reading.excludes_one(1.02, 0.002)
    assert point_reading.excludes_one(0.98, 0.002)


def test_pilot_2_draws_each_arm_whose_plug_in_pass_is_low() -> None:
    assert pilot.pilot_2_arms(_pilot_frame(1.0)) == ()
    assert pilot.pilot_2_arms(_pilot_frame([1.0, 0.95])) == ("__dr_known_slope",)


def test_the_fallback_answers_only_the_no_budget_case_after_pilot_2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing extension stops the declaration, and so does another sizing error."""
    _no_budget(monkeypatch)
    low = _pilot_frame([1.0, 0.945])
    assert pilot.pilot_2_arms(low) == ("__dr_known_slope",)
    with pytest.raises(ValueError, match="only after pilot_2"):
        pilot.declared_replicates(low, NO_PAIRS)
    _no_budget(monkeypatch, "the pilot correlations are not finite")
    with pytest.raises(ValueError) as caught:
        pilot.declared_replicates(_pilot_frame([0.975, 1.025], replicates=BOTH), NO_PAIRS)
    assert pilot.NO_BUDGET not in str(caught.value)
    assert "pilot_2" not in str(caught.value)


# --------------------------------------------------------------------------- the summary


def synthetic_rows(seed: int = 0, replicates: int = 400) -> pd.DataFrame:
    """Rows of every declared cell: correct positives, detecting controls, a growing ladder."""
    rng = np.random.default_rng(seed)
    rows = []
    for family, cells in study.PROPERTY_CELLS.items():
        for cell in cells:
            if cell in study.GROWTH_CELLS:
                continue
            role, shift, se = "positive", 0.0, 1.0
            if family == study.BIAS and cell.startswith("wrong_q"):
                role, shift = "control", 3.0
            if cell.endswith("__shrunken_se_control"):
                role, se = "control", 0.7
            if family == "spread":
                role = DIAGNOSTIC_ROLE
            if cell.endswith("__first_order"):
                size = int(cell.split("__")[0].rsplit("_n", 1)[1])
                role, shift = "control", -0.5 * (size / 1_000) ** (1.0 / 3.0)
            if cell.endswith("__conservative"):
                se = 1.2
            estimate = rng.normal(shift, 1.0, replicates)
            covered = np.abs(estimate) <= 1.96 * se
            rows += [
                {
                    "property": family,
                    "cell": cell,
                    "role": role,
                    "replicate": r,
                    "n": 4_000,
                    "requested_replicates": replicates,
                    "failed_replicates": 0,
                    "truth": 0.0,
                    "estimate": float(estimate[r]),
                    "std_error": se,
                    "covered": int(covered[r]),
                    "rejected": int(not covered[r]),
                }
                for r in range(replicates)
            ]
    return pd.DataFrame(rows).loc[:, list(properties.REPLICATE_COLUMNS)]


def test_the_summary_reads_every_declared_cell() -> None:
    summary = properties.summarize_properties(synthetic_rows())
    assert set(summary["cell"]) == {c for cells in study.PROPERTY_CELLS.values() for c in cells}
    diagnostic = summary.loc[summary["role"] == DIAGNOSTIC_ROLE]
    assert set(diagnostic["cell"]) == set(study.PROPERTY_CELLS["spread"])
    assert diagnostic["passed"].all()
    controls = summary.loc[summary["role"] == "control"]
    assert controls["passed"].all(), controls.loc[~controls["passed"].astype(bool), "cell"].tolist()
    conservative = summary.loc[summary["cell"] == "dr_glm__intercept__conservative"]
    assert bool(conservative["passed"].iloc[0])


# ---------------------------------------------------------------------- the cost and files


def test_the_smoke_cost_stays_inside_the_cost_gate() -> None:
    timings = pd.read_csv(study.SMOKE_TIMINGS)
    assert set(timings["group"]) == {group.label for group in study.GROUPS}
    for row in timings.itertuples(index=False):
        declared = study.GROUP_BY_LABEL[row.group].seconds
        assert row.seconds_per_fit <= study.COST_GATE * declared, row.group
    assert numbers()["cpu_hours"] == pytest.approx(study.run_cpu_hours())
    assert study.run_cpu_hours() == pytest.approx(2.28, abs=0.05)
    assert study.run_cpu_hours(10_800) == pytest.approx(3.38, abs=0.05)
    assert study.pilot_cpu_hours() == pytest.approx(0.49, abs=0.02)


def test_the_declaration_names_every_file_a_run_writes() -> None:
    from tests.canonical.declared_run import RECORD_FILES
    from tests.canonical.regenerate import ARTIFACT_NAMES, PROPERTY_FAILURE_FILES

    assert set(study.RUN_FILES) == {*ARTIFACT_NAMES, *RECORD_FILES, *study.STUDY.property_artifacts}
    assert set(study.CONDITIONAL_RUN_FILES) == set(PROPERTY_FAILURE_FILES)
    assert (Path(__file__).parents[1] / "canonical/projection_methods/regenerate.py").exists()


# ------------------------------------------------------------------------- the fitted pieces

#: The status each group's coefficients must carry: the interval ships only with the mechanism
#: declared (plan 1.5, 5.2).
STATUSES = {
    "dr_known_4k": "influence_curve",
    "dr_both_4k": "reduced_regression_plugin",
    "dr_slow_1k": "reduced_regression_plugin",
    "dr_glm_4k": "reduced_regression_plugin",
    "ctmle_points_4k": "generated_design_plugin",
}


@pytest.mark.parametrize("label", sorted(STATUSES))
def test_each_group_fits_with_its_declared_status(label: str) -> None:
    group = study.GROUP_BY_LABEL[label]
    frame, _ = study.draw_from_seed(label, 1_000, 3)
    result = study.fit(group, frame, 3)
    assert {e.inference for e in result.estimates.values()} == {STATUSES[label]}
    assert list(result.estimates) == [group.name(t) for t in group.terms]


@cache
def _known_fit() -> Any:
    group = study.GROUP_BY_LABEL["dr_known_4k"]
    frame, _ = study.draw_from_seed(group.label, 2_000, 7)
    return frame, study.fit(group, frame, 7)


def test_the_d_star_reading_adds_back_a_live_correction() -> None:
    """``D*`` alone differs from the reported curve by the mechanism correction, row by row."""
    _, result = _known_fit()
    reported = np.column_stack(
        [np.asarray(result.estimates[f"cate_projection[{t}]"].influence_curve) for t in study.TERMS]
    )
    added = study.d_star_curves(result) - reported
    assert np.abs(added).max(axis=0).min() > 1e-3
    np.testing.assert_allclose(study.d_star_curves(result).mean(axis=0), 0.0, atol=1e-6)
    ses = study.d_star_std_errors(result)
    assert np.all(ses > 0) and np.all(np.isfinite(ses))


def test_the_wrong_q_witness_reads_the_initial_wrong_regression() -> None:
    frame, result = _known_fit()
    learner = study.outcome_learner(study.GROUP_BY_LABEL["dr_known_4k"])
    learner.fit(frame[["A", "W1", "W2"]].to_numpy(), frame["Y"].to_numpy())
    arms = {
        a: learner.predict_proba(frame[["W1", "W2"]].assign(A=a)[["A", "W1", "W2"]].to_numpy())[
            :, 1
        ]
        for a in (0.0, 1.0)
    }
    phi = np.column_stack([np.ones(len(frame)), frame["W1"].to_numpy()])
    expected = np.linalg.lstsq(phi, arms[1.0] - arms[0.0], rcond=None)[0]
    np.testing.assert_allclose(study.plugin_initial(result), expected, atol=1e-8)


def test_the_slow_mechanism_is_the_glm_plus_the_declared_offset() -> None:
    frame, _ = study.draw_from_seed("dr_slow_4k", 1_000, 5)
    design = frame[["W1", "W2"]].to_numpy()
    slow = study.SlowMechanism(4_000).fit(design, frame["A"].to_numpy())
    glm = study.law_l.OffsetLogistic(columns=(0, 1)).fit(design, frame["A"].to_numpy())
    offset = 4_000 ** (-1.0 / 3.0) * design[:, 1] * (1.0 + design[:, 0])
    np.testing.assert_allclose(
        logit(slow.predict_proba(design)[:, 1]),
        logit(glm.predict_proba(design)[:, 1]) + offset,
        atol=1e-9,
    )
    assert study.treatment_learner(study.GROUP_BY_LABEL["dr_slow_16k"]).n == 16_000


def test_the_correct_outcome_learner_holds_law_p() -> None:
    frame, _ = study.draw_from_seed("dr_both_4k", 20_000, 9)
    learner = study.CorrectOutcome().fit(frame[["A", "W1", "W2"]].to_numpy(), frame["Y"])
    truth = law.qbar0(frame["A"], frame["W1"], frame["W2"])
    fitted = learner.predict_proba(frame[["A", "W1", "W2"]].to_numpy())[:, 1]
    assert float(np.max(np.abs(fitted - truth))) < 0.05
