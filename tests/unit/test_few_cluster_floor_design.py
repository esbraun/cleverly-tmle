"""Design checks of the few-cluster floor studies (X54 parts (a), (b) and (c)).

Every declared number of ``few-cluster-floor-tmle`` and ``few-cluster-floor-cross-fitted-ltmle``
is recomputed here: the published cells against the declaration, the seed streams, the sizing
figures the declaration quotes, and a deliberate-mutation control for each declared rule (plan
X54 rev 3, section 4.6). Each mutation test computes the declared reading and a broken one on
the same synthetic rows and asserts that they disagree, so a rule that silently drops its
component fails here.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from tests.studies import few_cluster_floor as floor
from tests.studies import few_cluster_floor_ltmle as study_b
from tests.studies import few_cluster_floor_ltmle_properties as properties_b
from tests.studies import few_cluster_floor_tmle as study_a
from tests.studies import few_cluster_floor_tmle_properties as properties_a
from tests.studies.evidence.registry import registered

STUDIES = (
    (study_a, properties_a),
    (study_b, properties_b),
)


# ------------------------------------------------------------------------------ declaration


@pytest.mark.parametrize(("study", "properties"), STUDIES)
def test_published_cells_equal_the_declaration(study, properties):
    declared = tuple(cell.cell for cell in properties.declared_cells())
    assert declared == study.PUBLISHED_CELLS
    assert study.STUDY.property_cells == {floor.FAMILY: study.PUBLISHED_CELLS}
    grouped = [cell for group in properties.PROPERTY_GROUPS[floor.FAMILY] for cell in group]
    assert sorted(grouped) == sorted(declared)
    roles = {cell.cell: cell.role for cell in properties.declared_cells()}
    for group in properties.PROPERTY_GROUPS[floor.FAMILY]:
        kinds = sorted(roles[cell] for cell in group)
        assert kinds.count("positive") == 1 and kinds.count("control") == 1


@pytest.mark.parametrize(("study", "properties"), STUDIES)
def test_declared_record(study, properties):
    record = study.STUDY
    assert record.publication_policy == "reporting"
    assert record.reference is None
    assert properties.FAILURE_BOUND == floor.FAILURE_BOUND
    assert record.failure_bound == {"cleverly": floor.FAILURE_BOUND}
    # Each study's seeds are its own.
    for other in registered():
        if other.slug != record.slug:
            assert {other.seed, other.resampling_seed}.isdisjoint(
                {record.seed, record.resampling_seed}
            )


def test_the_two_studies_draw_distinct_streams():
    a = {study_a.draw_seed(law, j, r) for law in study_a.SIZE_LAWS for j in (4, 10) for r in (0, 1)}
    pilot = {
        study_a.draw_seed(law, j, r, "pilot")
        for law in study_a.SIZE_LAWS
        for j in (4, 10)
        for r in (0, 1)
    }
    b = {study_b.draw_seed(law, j, r) for law in study_b.SIZE_LAWS for j in (4, 10) for r in (0, 1)}
    assert len(a) == len(pilot) == len(b) == 8
    assert a.isdisjoint(pilot) and a.isdisjoint(b)
    assert len({study_a.SEED, study_a.RESAMPLING_SEED, study_b.SEED, study_b.RESAMPLING_SEED}) == 4


def test_the_ladders_and_part_c_counts():
    assert study_a.POINT_LADDER == (4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 39)
    assert (*range(4, 21), 30, 39) == study_a.LTMLE_LADDER
    assert (*range(4, 21), 30, 39) == study_b.LADDER
    # M4: (C) on a cross-fitted fit only where every one of min(5, J) folds holds 2 clusters.
    assert study_a.jackknife_counts("tmle_stacked_known") == (10, 15, 20, 30, 39)
    assert study_a.jackknife_counts("tmle_in_sample_known") == study_a.POINT_LADDER
    assert study_a.jackknife_counts("ltmle_in_sample") == study_a.LTMLE_LADDER
    assert study_a.jackknife_counts("drtmle_known") == ()
    assert study_a.jackknife_counts("tmle_fold_evaluated_known") == ()
    assert study_b.jackknife_counts() == (*range(10, 21), 30, 39)
    # Each validation fold of a fold-evaluated fit holds at least 2 clusters.
    for clusters in study_a.POINT_LADDER:
        assert clusters // study_a.folds("tmle_fold_evaluated_known", clusters) >= 2
    # B4: every point family but DR-TMLE runs both mechanism configurations.
    for family in ("tmle_in_sample", "tmle_stacked", "tmle_fold_evaluated", "tmle_stacked_d5"):
        assert {fit for fit, owner in study_a.FITS.items() if owner == family} == {
            f"{family}_known",
            f"{family}_estimated",
        }


def test_wang_construction():
    assert floor.wang_parameters(10, 2) == (10 / 8, 8)
    assert floor.wang_parameters(10, 5) == (2.0, 5)
    assert floor.wang_parameters(5, 5) is None
    assert floor.wang_parameters(4, 2) == (2.0, 2)


# ------------------------------------------------------------------------------- sizing


def test_sizing_reproduces_the_plan_figures():
    # Plan 5.1: G1 at R = 4,000 and 6,000 by withheld-and-lost share.
    expected = {0.0: 1.0, 0.025: 1.0, 0.04: 0.492, 0.05: 0.019}
    for share, probability in expected.items():
        assert floor.g1_pass_probability(0.95 * (1 - share), 4_000) == pytest.approx(
            probability, abs=0.002
        )
    assert floor.g1_pass_probability(0.94 * 0.975, 4_000) == pytest.approx(0.841, abs=0.002)
    assert floor.g1_pass_probability(0.95 * 0.96, 6_000) == pytest.approx(0.717, abs=0.002)
    # Section 5, M5: the family product at 0.925 over 38 and 44 cells.
    assert floor.g1_pass_probability(0.925, 4_000) ** 38 == pytest.approx(0.954, abs=0.002)
    assert floor.g1_pass_probability(0.925, 4_000) ** 44 == pytest.approx(0.948, abs=0.002)
    assert floor.g1_pass_probability(0.925, 6_000) ** 44 == pytest.approx(1.0, abs=0.002)
    # The predictive per-cell G1 figures (Jeffreys posterior of a 400-draw pilot).
    for hits, value in ((370, 0.80), (372, 0.88), (376, 0.98)):
        assert floor.predictive_g1(hits, 400, 4_000) == pytest.approx(value, abs=0.015)
    # Plan 5.7: clause (i) for an equal candidate over 36 and 44 cells.
    for cells, figures in ((36, (0.999, 0.935, 0.578)), (44, (0.999, 0.921, 0.512))):
        for discordance, value in zip((0.04, 0.06, 0.08), figures, strict=True):
            assert floor.admissibility_probability(discordance, cells) == pytest.approx(
                value, abs=0.002
            )


def test_required_replicates_rises_in_steps_to_the_cap():
    calibrated = {
        "coverage": 0.945,
        "standardized_bias": 0.0,
        "ratio": 0.6,
        "ratio_sd": 0.01,
        "se_ratio": 1.0,
        "se_ratio_sd": 0.01,
        "pilot_draws": 400,
    }
    assert floor.required_replicates([calibrated] * 10) == 4_000
    marginal = dict(calibrated, coverage=0.921)
    replicates = floor.required_replicates([marginal] * 40)
    assert replicates % floor.R_STEP == 0 and 4_000 < replicates <= floor.R_CAP


def test_expected_floor():
    coverage = {4: [0.91, 0.93], 5: [0.93, 0.94], 6: [0.95, 0.95], 10: [0.95, 0.95]}
    assert floor.expected_floor(coverage, anchor=10) == 5
    coverage[6] = [0.92, 0.95]
    assert floor.expected_floor(coverage, anchor=10) == 10


# --------------------------------------------------------------------- synthetic rows


def _rows(cell: str, covered: list[int], withheld: list[int] | None = None) -> pd.DataFrame:
    withheld = withheld or [0] * len(covered)
    return pd.DataFrame(
        {
            "property": floor.FAMILY,
            "cell": cell,
            "replicate": np.arange(len(covered)),
            "covered": covered,
            "withheld": withheld,
            "half_width": 1.0,
        }
    )


def _pair(n: int, candidate_only: int, reference_only: int) -> floor.PairedCounts:
    return floor.PairedCounts(n, candidate_only, reference_only)


def _cell(candidate, clusters, pair, *, hits=1900, shown=2000, length=1.0, high=1.0, law="a"):
    return floor.CandidateCell(
        candidate=candidate,
        configuration="fit",
        law=law,
        clusters=clusters,
        pair=pair,
        hits=hits,
        shown=shown,
        length=length,
        length_high=high,
    )


# -------------------------------------------------------------------- mutation controls


def test_floor_needs_every_configuration():
    """B4: a count passes only when both mechanism configurations pass."""
    verdicts = {
        ("x_known", "law", 6): True,
        ("x_estimated", "law", 6): False,
        ("x_known", "law", 10): True,
        ("x_estimated", "law", 10): True,
    }
    declared = floor.family_floor(
        floor.count_passes(verdicts, ["x_known", "x_estimated"]), anchor=10
    )
    mutated = floor.family_floor(floor.count_passes(verdicts, ["x_known"]), anchor=10)
    assert declared == 10
    assert mutated == 6
    with pytest.raises(ValueError, match="lacks configurations"):
        floor.count_passes({("x_known", "law", 6): True}, ["x_known", "x_estimated"])


def test_floor_rule():
    passes = {4: False, 5: True, 6: True, 10: True, 15: True}
    assert floor.family_floor(passes, anchor=10) == 5
    # A failing anchor keeps the old floor; a gap keeps the floor above it.
    assert floor.family_floor({4: True, 5: True, 10: False}, anchor=10) == 10
    assert floor.family_floor({4: True, 5: False, 6: True, 10: True}, anchor=10) == 6


def test_g1_counts_withheld_rows_as_misses():
    """G1 reads covered hits over every requested draw; dropping withheld rows inflates it."""
    rows = _rows("c", [1] * 900 + [0] * 100, [0] * 900 + [1] * 60 + [0] * 40)
    rows["requested_replicates"] = 1_000
    rows["trigger"] = np.where(rows["withheld"] == 1, "constant_cell", "")
    rows["std_error"] = 1.0
    rows["estimate"] = 0.0
    summary = pd.DataFrame({"property": [floor.FAMILY], "cell": ["fit__law__j6__t_reference"]})
    for column in floor.READING_COLUMNS:
        summary[column] = np.nan
    rows["cell"] = "fit__law__j6__t_reference"
    floor.floor_readings(summary, rows, study_a.STUDY)
    declared = float(summary.loc[0, "all_draw_coverage"])
    mutated = float(rows.loc[rows["withheld"] == 0, "covered"].mean())
    assert declared == pytest.approx(0.90)
    assert mutated == pytest.approx(900 / 940)
    assert summary.loc[0, "withheld_by_trigger"] == "constant_cell=60"
    assert float(summary.loc[0, "all_draw_ci_lower"]) < floor.G1_FLOOR


def test_noninferiority_scores_withheld_candidate_draws_as_misses():
    """A candidate that withholds 6% of draws is not non-inferior; scored on its shown fits it is."""
    n = 2_000
    reference = _rows("a", [1] * 1_900 + [0] * 100)
    withheld = [0] * n
    covered = [1] * 1_900 + [0] * 100
    for i in range(0, 120):
        withheld[i] = 1
        covered[i] = 0
    candidate = _rows("c", covered, withheld)
    declared = floor.paired_counts(candidate, reference, draws=n)
    assert floor.difference_interval([declared]).low < floor.PER_CELL_MARGIN
    shown = candidate.loc[candidate["withheld"] == 0]
    mutated = floor.paired_counts(shown, reference.loc[shown.index], draws=n)
    assert floor.difference_interval([mutated]).low > floor.PER_CELL_MARGIN


def test_part_c_reads_one_draw_set():
    """(A) and the candidate are read on the same first 2,000 draws (M3)."""
    reference = _rows("a", [1] * 2_000 + [0] * 2_000)
    candidate = _rows("c", [1] * 2_000)
    declared = floor.paired_counts(candidate, reference, draws=2_000)
    assert declared.difference == 0.0
    # The mutation: (A) on its full 4,000 draws, the candidate on its 2,000.
    mutated = reference["covered"].mean() - candidate["covered"].mean()
    assert mutated == pytest.approx(-0.5)


def test_admissibility_is_paired_not_absolute():
    """A candidate with an absolute G1 pass but a paired loss of 3 points is not admissible."""
    loss = [_cell(floor.JACKKNIFE_ARM, j, _pair(2_000, 0, 80), hits=1_840) for j in (10, 15, 20)]
    assert not floor.admissibility(loss, floor.JACKKNIFE_ARM, start=10)
    mutated = all(
        floor.clopper_pearson(c.hits, c.pair.draws, confidence_level=0.99).low >= floor.G1_FLOOR
        for c in loss
    )
    assert mutated


def test_admissibility_needs_the_pooled_clause():
    """A systematic loss of 1.5 points passes every per-cell clause and fails the pooled one."""
    cells = [
        _cell(floor.WANG_ARM, j, _pair(2_000, 10, 40), law=law)
        for j in range(10, 40)
        for law in ("a", "b")
    ]
    per_cell = all(floor.difference_interval([c.pair]).low > floor.PER_CELL_MARGIN for c in cells)
    assert per_cell
    assert not floor.admissibility(cells, floor.WANG_ARM, start=10)


def test_reference_decision_order():
    def family(c_hits: int, c_high: float) -> list[floor.CandidateCell]:
        out = []
        for j in (10, 15, 20, 30):
            out.append(_cell(floor.REFERENCE_ARM, j, _pair(2_000, 0, 0), hits=1_960))
            out.append(_cell(floor.WANG_ARM, j, _pair(2_000, 20, 0), hits=1_980))
            out.append(
                _cell(
                    floor.JACKKNIFE_ARM, j, _pair(2_000, 0, 0), hits=c_hits, length=0.9, high=c_high
                )
            )
        return out

    # A is not calibrated (0.98 shown coverage), B over-covers, C is calibrated at every count.
    decision = floor.reference_decision(family(1_900, 0.95), floor_a=10, jackknife_start=10)
    assert decision.winner == floor.JACKKNIFE_ARM
    assert decision.calibrated_counts[floor.JACKKNIFE_ARM] == 4
    # Ties go to A when no candidate leads by 2 counts and none is shorter.
    tied = floor.reference_decision(family(1_960, 1.05), floor_a=10, jackknife_start=10)
    assert tied.winner == floor.REFERENCE_ARM
    # A refused (C) below its start is not admissible there.
    late = floor.reference_decision(family(1_900, 0.95), floor_a=10, jackknife_start=40)
    assert late.admissible[floor.JACKKNIFE_ARM] is None


def test_candidate_c_absent_cells_read_as_misses():
    reference = _rows("ref", [1] * 50)
    absent = reference.iloc[0:0]
    counts = floor.paired_counts(absent, reference, draws=50)
    assert counts.reference_only == 50 and counts.candidate_only == 0


def test_rows_of_a_withheld_arm_are_misses():
    reading = floor.Reading(0.1, float("nan"), float("nan"), 2, "wang_undefined")
    row = floor.arm_row(
        reading, cell="c", role="diagnostic", replicate=0, n=40, requested=10, truth=0.1, clusters=4
    )
    assert (row["covered"], row["rejected"], row["withheld"]) == (0, 0, 1)
    shown = dataclasses.replace(reading, std_error=0.05, critical=2.0, trigger="")
    row = floor.arm_row(
        shown, cell="c", role="diagnostic", replicate=0, n=40, requested=10, truth=0.1, clusters=4
    )
    assert (row["covered"], row["withheld"], row["half_width"]) == (1, 0, pytest.approx(0.1))
