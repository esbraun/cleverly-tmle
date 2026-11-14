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
import types

import numpy as np
import pandas as pd
import pytest

from cleverly._inference_status import CLUSTER_INTERVAL_FLOORS, FLOOR_FAMILY_NAMES
from cleverly.exceptions import CapabilityError, DataError
from tests.canonical import property_differential as differential
from tests.studies import few_cluster_floor as floor
from tests.studies import few_cluster_floor_ltmle as study_b
from tests.studies import few_cluster_floor_ltmle_properties as properties_b
from tests.studies import few_cluster_floor_tmle as study_a
from tests.studies import few_cluster_floor_tmle_properties as properties_a
from tests.studies.evidence import property_failures, red_cells
from tests.studies.evidence.inference import clopper_pearson
from tests.studies.evidence.manifest import read_rows
from tests.studies.evidence.registry import registered
from tests.studies.evidence.replication import recording

STUDIES = (
    (study_a, properties_a),
    (study_b, properties_b),
)


# ------------------------------------------------------------------------------ declaration


@pytest.mark.parametrize(("study", "properties"), STUDIES)
def test_published_cells_equal_the_declaration(study, properties):
    declared = properties.declared_cells()
    by_family = {
        family: tuple(cell.cell for cell in declared if cell.property == family)
        for family in (floor.FAMILY, floor.CANDIDATE_FAMILY)
    }
    assert by_family[floor.FAMILY] == study.PUBLISHED_CELLS
    assert by_family[floor.CANDIDATE_FAMILY] == study.CANDIDATE_CELLS
    assert study.STUDY.property_cells == {
        floor.FAMILY: study.PUBLISHED_CELLS,
        floor.CANDIDATE_FAMILY: study.CANDIDATE_CELLS,
        floor.POOLED_FAMILY: study.POOLED_CELLS,
    }
    roles = {(cell.property, cell.cell): cell.role for cell in declared}
    for family in (floor.FAMILY, floor.CANDIDATE_FAMILY):
        groups = properties.PROPERTY_GROUPS[family]
        assert sorted(cell for group in groups for cell in group) == sorted(by_family[family])
        for group in groups:
            kinds = sorted(roles[(family, cell)] for cell in group)
            assert kinds.count("positive") == 1 and kinds.count("control") == 1
    # Each pair is a candidate and (A)'s copy on the same draws, read on the first 2,000.
    for positive, control in properties.PROPERTY_GROUPS[floor.CANDIDATE_FAMILY]:
        assert floor.positive_of(control) == positive
    assert {cell.replicates for cell in declared if cell.property == floor.CANDIDATE_FAMILY} == {
        floor.CANDIDATE_DRAWS
    }
    assert properties.PROPERTY_GROUPS[floor.POOLED_FAMILY] == tuple(
        (name,) for name in study.POOLED_CELLS
    )
    # The part (c) families read the module's rule; the floor family the registered one.
    for family in (floor.CANDIDATE_FAMILY, floor.POOLED_FAMILY):
        assert property_failures.rule_for(family, properties) is properties.PROPERTY_RULES[0]
    assert property_failures.rule_for(floor.FAMILY, properties) not in properties.PROPERTY_RULES


@pytest.mark.parametrize(("study", "properties"), STUDIES)
def test_declared_record(study, properties):
    """B2: the primary bound is 0.01, the property bound the harness's cap of 0.05."""
    record = study.STUDY
    assert record.publication_policy == "reporting"
    assert record.reference is None
    assert properties.FAILURE_BOUND == 0.05 == property_failures.MAXIMUM_BOUND
    assert record.failure_bound == {"cleverly": 0.01}
    # Each study's seeds are its own.
    for other in registered():
        if other.slug != record.slug:
            assert {other.seed, other.resampling_seed}.isdisjoint(
                {record.seed, record.resampling_seed}
            )


def test_the_red_owner_is_a_roadmap_owner():
    """B3: every declared reading is held by an owner the roadmap's table lists."""
    assert floor.RED_OWNER == "X54-cluster-floor"
    assert floor.RED_OWNER in red_cells.owner_ids()
    roadmap = red_cells.ROADMAP.read_text()
    for reading in floor.READINGS:
        assert f"`{reading}`" in roadmap
    assert floor.NO_DEFECT_DIAGNOSES == {}


@pytest.mark.parametrize(("study", "properties"), STUDIES)
def test_the_differential_reads_a_declared_subset_of_whole_groups(study, properties):
    """B4: both modules are differential subjects at a declared size, on whole groups."""
    module = properties.__name__
    assert module in differential.SYNTHETIC_MODULES
    assert properties.DIFFERENTIAL_BUDGET == 200
    cells = properties.differential_cells()
    for family in (floor.FAMILY, floor.CANDIDATE_FAMILY):
        for group in properties.PROPERTY_GROUPS[family]:
            inside = {f"{family}/{cell}" in cells for cell in group}
            assert len(inside) == 1, group
    families = {name.split("/")[0] for name in cells}
    assert families == {floor.FAMILY, floor.CANDIDATE_FAMILY}
    bound = differential.read_bindings()
    clauses = set(bound.loc[bound["subject"] == module, "clause"])
    assert len(clauses) == 6
    # The cap applies to rows, keeping each cell's shape.
    rows = pd.DataFrame(
        {
            "property": floor.FAMILY,
            "cell": sorted(name.split("/")[1] for name in cells if name.startswith(floor.FAMILY))[
                0
            ],
            "replicate": range(300),
            "requested_replicates": 4_000,
        }
    )
    kept = differential.declared_subset(properties, rows)
    assert len(kept) == 200 and set(kept["requested_replicates"]) == {200}


def test_study_families_map_to_the_package_floors():
    """m7, M2: each floor family is a package floor family; d5 is reported, not a family."""
    names = {**study_a.ANCHORS, study_b.FIT: study_b.ANCHOR}
    assert set(floor.PACKAGE_FAMILIES) == set(names)
    assert set(floor.PACKAGE_FAMILIES.values()) == set(FLOOR_FAMILY_NAMES)
    assert set(floor.PACKAGE_FAMILIES.values()) == set(CLUSTER_INTERVAL_FLOORS)
    for family, anchor in names.items():
        assert CLUSTER_INTERVAL_FLOORS[floor.PACKAGE_FAMILIES[family]] == anchor
    assert "tmle_stacked_d5" not in study_a.ANCHORS
    assert {study_a.FITS[fit] for fit in study_a.D5_FITS} == {"tmle_stacked"}
    assert not set(study_a.floor_fits("tmle_stacked")) & study_a.D5_FITS
    assert set(study_a.floor_fits(study_a.D5_REPORTED)) == study_a.D5_FITS
    assert study_a.D5_FITS.issubset(study_a.configurations())


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
    for family in ("tmle_in_sample", "tmle_stacked", "tmle_fold_evaluated"):
        assert set(study_a.floor_fits(family)) == {f"{family}_known", f"{family}_estimated"}
    assert study_a.jackknife_counts("tmle_stacked_d5_known") == (10, 15, 20, 30, 39)


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


def _rows(
    cell: str, covered: list[int], withheld: list[int] | None = None, family: str = floor.FAMILY
) -> pd.DataFrame:
    withheld = withheld or [0] * len(covered)
    return pd.DataFrame(
        {
            "property": family,
            "cell": cell,
            "replicate": np.arange(len(covered)),
            "covered": covered,
            "withheld": withheld,
            "half_width": 1.0,
            "requested_replicates": len(covered),
        }
    )


def _pair(n: int, candidate_only: int, reference_only: int) -> floor.PairedCounts:
    return floor.PairedCounts(n, candidate_only, reference_only)


def _cell(
    candidate,
    clusters,
    pair,
    *,
    hits=1900,
    shown=2000,
    length=1.0,
    high=1.0,
    law="a",
    gain=0.0,
    configuration="fit",
):
    return floor.CandidateCell(
        candidate=candidate,
        configuration=configuration,
        law=law,
        clusters=clusters,
        pair=pair,
        hits=hits,
        shown=shown,
        length=length,
        length_high=high,
        gain_low=gain,
    )


def _cheap(record):
    margins = dataclasses.replace(record.margins, bootstrap_replicates=200)
    return dataclasses.replace(record, margins=margins)


# ------------------------------------------------------------------- B1: decoupled keys

#: Draws of ``equal10`` at 4 clusters on a ``"unit"`` stream that no study reads: replicate 7
#: has a constant treated arm (and an out-of-range in-sample LTMLE estimate), replicate 6 a
#: shown main fit whose jackknife loses a refit.
CONSTANT_ARM, FAILED_REFIT = 7, 6


def _unit_frame(replicate: int) -> pd.DataFrame:
    return study_a.draw("equal10", 4, study_a.draw_seed("equal10", 4, replicate, "unit"))


@pytest.mark.parametrize(
    ("fit", "replicate", "main", "jackknife"),
    [
        ("tmle_in_sample_known", CONSTANT_ARM, "constant_arm_cell", "constant_arm_cell"),
        ("tmle_in_sample_known", FAILED_REFIT, "", "jackknife_refit_plugin"),
        ("ltmle_in_sample", CONSTANT_ARM, "out_of_range", "out_of_range"),
    ],
)
def test_a_withheld_or_failed_candidate_never_costs_the_reference_rows(
    fit, replicate, main, jackknife
):
    """B1: (A) and its control keep the status the main fit reports; (C) reads its own miss."""
    frame = _unit_frame(replicate)
    rows = study_a.fit_rows(fit, "equal10", 4, replicate, frame, budget=8, stream="unit")
    by_cell = {(row["property"], row["cell"]): row for row in rows}
    for arm in (floor.REFERENCE_ARM, "iid_t_control"):
        row = by_cell[(floor.FAMILY, study_a.cell_name(fit, "equal10", 4, arm))]
        assert row["trigger"] == main and row["withheld"] == int(bool(main))
    assert all(row["cell"].rsplit("__", 1)[1] != floor.JACKKNIFE_ARM for row in rows)
    (row,) = study_a.jackknife_rows(fit, "equal10", 4, replicate, frame, budget=8, stream="unit")
    assert row["property"] == floor.CANDIDATE_FAMILY
    assert row["trigger"] == jackknife and row["withheld"] == 1 and row["covered"] == 0
    assert np.isfinite(row["estimate"])
    # The (C) key fills its own cell only, so its loss costs no (A) row.
    key = ("equal10", 4, replicate, fit + study_a.JACKKNIFE_SUFFIX, 8, "unit")
    assert properties_a._cells_of(key) == [
        f"{floor.CANDIDATE_FAMILY}/{study_a.cell_name(fit, 'equal10', 4, floor.JACKKNIFE_ARM)}"
    ]


def test_an_error_in_the_candidate_is_its_own_miss(monkeypatch):
    """B1 mutation: a (C) fit that raises is a withheld row; re-raising would lose the key."""
    frame = _unit_frame(FAILED_REFIT)

    def broken(*args, **kwargs):
        raise RuntimeError("quasibinomial IRLS did not converge")

    monkeypatch.setattr(study_a, "jackknife_estimate", broken)
    (row,) = study_a.jackknife_rows("tmle_in_sample_known", "equal10", 4, FAILED_REFIT, frame)
    assert row["trigger"] == "jackknife_error"
    assert row["jackknife_error"].startswith("RuntimeError: quasibinomial")

    def reraising(fit, frame, clusters, estimate):
        try:
            return study_a.jackknife_estimate(fit, frame, clusters)
        except (CapabilityError, DataError):  # the pre-fix catch
            return floor.refused_jackknife(float(estimate.psi), clusters)

    monkeypatch.setattr(study_a, "candidate_c", reraising)
    with pytest.raises(RuntimeError, match="did not converge"):
        study_a.jackknife_rows("tmle_in_sample_known", "equal10", 4, FAILED_REFIT, frame)


def test_the_package_glm_refit_never_raises_out_of_the_key(monkeypatch):
    """B1: any exception of the package-GLM refit is recorded on (A)'s row."""
    frame = _unit_frame(FAILED_REFIT)
    original = study_a.fit_ltmle_estimate

    def failing(frame, **overrides):
        if isinstance(overrides.get("outcome_learner"), study_a.GLM):
            raise RuntimeError("information matrix is singular")
        return original(frame, **overrides)

    monkeypatch.setattr(study_a, "fit_ltmle_estimate", failing)
    rows = study_a.fit_rows("ltmle_in_sample", "equal10", 4, FAILED_REFIT, frame)
    assert rows[0]["package_glm_raised"] == 1
    assert rows[0]["package_glm_error"] == "RuntimeError: information matrix is singular"


# ------------------------------------------------------------------------ m1, m2, m6


class _Estimate:
    def __init__(self, status: str) -> None:
        self.inference = status
        self.supplies_inference = status in {"influence_curve", "few_cluster_plugin"}


def test_every_withheld_status_has_its_trigger():
    """m1: each status maps exactly; the old substring rule labelled a constant arm "other"."""
    for status, trigger in floor.STATUS_TRIGGERS.items():
        assert floor.trigger_of(_Estimate(status)) == trigger
        assert trigger in floor.TRIGGERS
    assert floor.trigger_of(_Estimate("few_cluster_plugin")) == ""
    assert floor.trigger_of(_Estimate("trimmed_ratio_plugin")) == "other"
    mutated = "constant_cell" in "constant_arm_cell_plugin"
    assert not mutated


def test_sizing_reads_requested_draws():
    """m2: a lost draw is a miss, so hits and the pilot size read the requested draws."""
    positive = {
        "all_draw_coverage": 360 / 400,
        "requested_draws": 400.0,
        "replicates": 380.0,
        "standardized_bias": 0.0,
        "se_ratio": 1.0,
        "se_ratio_ci_lower": 0.9,
        "se_ratio_ci_upper": 1.1,
    }
    control = {
        "paired_se_ratio": 0.6,
        "paired_se_ratio_ci_lower": 0.55,
        "paired_se_ratio_ci_upper": 0.65,
    }
    inputs = floor.sizing_inputs(positive, control)
    assert inputs["hits"] == pytest.approx(360.0) and inputs["pilot_draws"] == 400.0
    mutated = positive["all_draw_coverage"] * positive["replicates"]
    assert mutated == pytest.approx(342.0)


def test_the_jackknife_is_split_by_the_package_glm():
    """m6: (C)'s shown coverage is published apart on draws where the package GLM raised."""
    jackknife = _rows("f__l__j4__cluster_jackknife", [1, 0, 1, 1], family=floor.CANDIDATE_FAMILY)
    reference = _rows("f__l__j4__reference_for_cluster_jackknife", [1, 1, 1, 1])
    reference["package_glm_raised"] = [0, 1, 1, 0]
    split = floor.jackknife_package_split(jackknife, reference)
    assert split == {"shown_coverage_package_raised": 0.5, "shown_coverage_package_ran": 1.0}


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
    rows["trigger"] = np.where(rows["withheld"] == 1, "constant_arm_cell", "")
    rows["std_error"] = 1.0
    rows["estimate"] = 0.0
    rows["failed_refits"] = 0
    summary = pd.DataFrame({"property": [floor.FAMILY], "cell": ["fit__law__j6__t_reference"]})
    for column in floor.READING_COLUMNS:
        summary[column] = np.nan
    rows["cell"] = "fit__law__j6__t_reference"
    floor.floor_readings(summary, rows, study_a.STUDY)
    declared = float(summary.loc[0, "all_draw_coverage"])
    mutated = float(rows.loc[rows["withheld"] == 0, "covered"].mean())
    assert declared == pytest.approx(0.90)
    assert mutated == pytest.approx(900 / 940)
    assert summary.loc[0, "withheld_by_trigger"] == "constant_arm_cell=60"
    assert float(summary.loc[0, "all_draw_ci_lower"]) < floor.G1_FLOOR


def _stub_estimate(replicate: int, clusters: int, **overrides):
    """A deterministic shown estimate: an IC whose IID SE sits well below the cluster SE."""
    rng = np.random.default_rng(replicate)
    record = types.SimpleNamespace(failed_refits=(), jackknife_variance=0.01)
    return types.SimpleNamespace(
        psi=float(rng.normal(0.0, 0.1)),
        plugin_std_error=0.1,
        reference_df=clusters - 2,
        supplies_inference=True,
        inference="influence_curve",
        influence_curve=rng.normal(0.0, 0.5, size=clusters * study_b.CLUSTER_SIZE),
        jackknife=record if "cluster_variance" in overrides else None,
    )


def test_lost_draws_are_misses_under_the_accounting(monkeypatch):
    """m5 (plan 4.4): two lost main fits cost (A) and its copies, read as misses by the
    accounting; (C)'s own key keeps its rows, and the S7 fill reads (A)'s lost draws covered."""
    lost = {3, 4}
    sizes, clusters, budget = "equal40", 10, 40

    def estimate(frame, clusters, **overrides):
        replicate = int(frame)
        if replicate in lost and not overrides:
            raise np.linalg.LinAlgError("singular matrix")
        return _stub_estimate(replicate, clusters, **overrides)

    monkeypatch.setattr(study_b, "fit_estimate", estimate)
    monkeypatch.setattr(properties_b, "_draw", lambda key: (key[2], {study_b.TARGET: 0.0}))
    every = properties_b.keys
    monkeypatch.setattr(
        properties_b,
        "keys",
        lambda budget, stream: [k for k in every(budget, stream) if k[:2] == (sizes, clusters)],
    )
    with recording("cleverly", witness=properties_b.PROPERTY_WITNESS) as recorded:
        rows = properties_b.generate_property_rows(n_jobs=1, budget=budget)
    records = property_failures.frame(recorded)
    rows = property_failures.stamp(rows, records)
    record = _cheap(study_b.STUDY)
    summary = properties_b.summarize_properties(rows)
    accounting = property_failures.property_accounting(
        rows, records, summary, record=record, module=properties_b
    )
    positive = study_b.cell_name(sizes, clusters, floor.REFERENCE_ARM)
    reference = summary.loc[summary["cell"] == positive].iloc[0]
    assert (reference["replicates"], reference["failed_replicates"]) == (budget - 2, 2)
    jackknife = study_b.cell_name(sizes, clusters, floor.JACKKNIFE_ARM)
    assert (rows["cell"] == jackknife).sum() == budget
    legs = accounting.loc[accounting["cell"] == positive].set_index("leg")
    hits = int(rows.loc[rows["cell"] == positive, "covered"].sum())
    worst = clopper_pearson(hits, budget, confidence_level=0.99).low
    assert legs.loc["coverage floor", "worst_case"] == pytest.approx(worst - 0.90, abs=1e-9)
    gain = accounting.loc[
        (accounting["property"] == floor.CANDIDATE_FAMILY)
        & (accounting["leg"] == "candidate coverage gain over the reference")
    ]
    assert len(gain)


# ---------------------------------------------------------------- M1: the candidate family


def _pair_rows(name: str, candidate: list[int], reference: list[int]) -> pd.DataFrame:
    fit, law, count, arm = name.split("__")
    control = "__".join((fit, law, count, floor.reference_for(arm)))
    out = pd.concat(
        [
            _rows(name, candidate, family=floor.CANDIDATE_FAMILY).assign(role="positive"),
            _rows(control, reference, family=floor.CANDIDATE_FAMILY).assign(role="control"),
        ],
        ignore_index=True,
    )
    return out


def _paired_lists(n: int, candidate_only: int, reference_only: int) -> tuple[list[int], list[int]]:
    both = n - 100 - candidate_only - reference_only
    candidate = [1] * both + [1] * candidate_only + [0] * reference_only + [0] * 100
    reference = [1] * both + [0] * candidate_only + [1] * reference_only + [0] * 100
    return candidate, reference


def _candidate_summary(rows: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    cells = rows[["property", "cell", "role"]].drop_duplicates().reset_index(drop=True)
    summary = cells.assign(passed=False, property_passed=None)
    for column in floor.CANDIDATE_COLUMNS:
        summary[column] = np.nan
    groups = [
        (positive, control)
        for positive in cells.loc[cells["role"] == "positive", "cell"]
        for control in [floor.positive_of(positive).replace("__cluster", "__reference_for_cluster")]
    ]
    return summary, groups


def test_clause_i_scores_withheld_candidate_draws_as_misses():
    """A candidate that withholds 6% of draws is not non-inferior; on its shown fits it is."""
    n = 2_000
    candidate = [1] * 1_900 + [0] * 100
    withheld = [0] * n
    for i in range(120):
        withheld[i], candidate[i] = 1, 0
    reference = [1] * 1_900 + [0] * 100
    rows = _pair_rows("f__l__j10__cluster_jackknife", candidate, reference)
    rows.loc[rows["role"] == "positive", "withheld"] = withheld
    summary, groups = _candidate_summary(rows)
    record = _cheap(study_a.STUDY)
    floor.candidate_verdicts(summary, rows, record, groups)
    assert not summary["property_passed"].astype(bool).any()
    shown = rows.loc[rows["withheld"] == 0]
    kept = shown.loc[shown["role"] == "positive", "replicate"]
    mutated = rows.loc[rows["replicate"].isin(kept)]
    summary, groups = _candidate_summary(mutated)
    floor.candidate_verdicts(summary, mutated, record, groups)
    assert summary["property_passed"].astype(bool).all()


def test_the_pooled_clause_catches_a_systematic_loss():
    """A 1.5-point loss in every cell passes clause (i) in each and fails the pooled clause."""
    frames = []
    for count in range(10, 40):
        candidate, reference = _paired_lists(2_000, 10, 40)
        frames.append(_pair_rows(f"cfg__l__j{count}__wang_inflated", candidate, reference))
    rows = pd.concat(frames, ignore_index=True)
    summary, _ = _candidate_summary(rows)
    groups = [
        (f"cfg__l__j{count}__wang_inflated", f"cfg__l__j{count}__reference_for_wang_inflated")
        for count in range(10, 40)
    ]
    record = _cheap(study_a.STUDY)
    floor.candidate_verdicts(summary, rows, record, groups)
    assert summary["property_passed"].astype(bool).all()
    (pooled,) = floor.pooled_rows(summary, rows, record, {("cfg", floor.WANG_ARM): 10})
    assert pooled["pooled_cells"] == 30.0
    assert not pooled["passed"]
    assert pooled["coverage_gain_ci_lower"] < floor.POOLED_MARGIN
    # The pooled start: above every count, the row reads no cell and cannot pass.
    (late,) = floor.pooled_rows(summary, rows, record, {("cfg", floor.WANG_ARM): 40})
    assert late["pooled_cells"] == 0.0 and not late["passed"]


def test_admissibility_reads_both_verdict_clauses():
    """Clause (i) per cell and clause (ii) per configuration; either failing refuses."""
    cells = [
        _cell(floor.WANG_ARM, j, _pair(2_000, 0, 0), configuration=c, gain=-0.02)
        for j in (10, 15)
        for c in ("known", "estimated")
    ]
    pooled = {("known", floor.WANG_ARM): True, ("estimated", floor.WANG_ARM): True}
    assert floor.admissibility(cells, floor.WANG_ARM, start=10, pooled=pooled) == (True, "")
    # m9: the worse configuration decides; the mutation pooling across both would pass.
    worse = {**pooled, ("estimated", floor.WANG_ARM): False}
    ok, reason = floor.admissibility(cells, floor.WANG_ARM, start=10, pooled=worse)
    assert not ok and reason == "clause (ii) fails in estimated"
    local = [*cells[:-1], dataclasses.replace(cells[-1], gain_low=-0.031)]
    ok, reason = floor.admissibility(local, floor.WANG_ARM, start=10, pooled=pooled)
    assert not ok and reason == "clause (i) fails at estimated/a/J15"
    # A cell below the start is not read.
    early = [*cells, _cell(floor.WANG_ARM, 4, _pair(2_000, 0, 0), gain=-0.5)]
    assert floor.admissibility(early, floor.WANG_ARM, start=10, pooled=pooled)[0]


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

    pooled = {("fit", floor.WANG_ARM): True, ("fit", floor.JACKKNIFE_ARM): True}
    # A is not calibrated (0.98 shown coverage), B over-covers, C is calibrated at every count.
    decision = floor.reference_decision(
        family(1_900, 0.95), floor_a=10, jackknife_start=10, pooled=pooled
    )
    assert decision.winner == floor.JACKKNIFE_ARM
    assert decision.calibrated_counts[floor.JACKKNIFE_ARM] == 4
    # Ties go to A when no candidate leads by 2 counts and none is shorter.
    tied = floor.reference_decision(
        family(1_960, 1.05), floor_a=10, jackknife_start=10, pooled=pooled
    )
    assert tied.winner == floor.REFERENCE_ARM
    # A refused (C) below its start is not admissible there, with the reason recorded.
    late = floor.reference_decision(
        family(1_900, 0.95), floor_a=10, jackknife_start=40, pooled=pooled
    )
    assert late.admissible[floor.JACKKNIFE_ARM] is None
    assert late.reasons[floor.JACKKNIFE_ARM] == "no cluster_jackknife cell at or above 40 clusters"


def test_part_c_reads_one_draw_set():
    """(A) and the candidate are read on the same first 2,000 draws (M3)."""
    reference = _rows("a", [1] * 2_000 + [0] * 2_000)
    candidate = _rows("c", [1] * 2_000)
    declared = floor.paired_counts(candidate, reference, draws=2_000)
    assert declared.difference == 0.0
    # The mutation: (A) on its full 4,000 draws, the candidate on its 2,000.
    mutated = reference["covered"].mean() - candidate["covered"].mean()
    assert mutated == pytest.approx(-0.5)


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


# --------------------------------------------------------------- M3, B3: the declared outcome


def _summary_row(cell: str, passed: bool, **values) -> dict:
    row = {
        "cell": cell,
        "passed": passed,
        "requested_draws": 4_000.0,
        "replicates": 4_000.0,
        "withheld_draws": 0.0,
        "shown_ci_lower": 0.90,
        "shown_ci_upper": 0.93,
        "se_ratio_ci_lower": 0.95,
        "se_ratio_ci_upper": 1.05,
    }
    return {**row, **values}


def _family_cells(red: dict[int, str] | None = None, ladder=(4, 5, 6, 10, 15, 20)):
    """Two configurations, one law; ``red`` maps a count to the arm that fails there."""
    red = red or {}
    out = {}
    for configuration in ("x_known", "x_estimated"):
        for count in ladder:
            arm = red.get(count) if configuration == "x_known" else None
            positive = _summary_row(f"{configuration}__l__j{count}__t_reference", arm != "positive")
            control = _summary_row(f"{configuration}__l__j{count}__iid_t_control", arm != "control")
            out[(configuration, "l", count)] = (positive, control)
    return out


def test_the_route_reads_a_red_above_the_anchor_as_a_defect_suspect():
    """M3: a red at 15 keeps the floor and names the count; it is no below-floor red."""
    cells = _family_cells({15: "positive"})
    route = floor.declared_route("x", cells, anchor=10)
    assert (route.floor, route.route) == (10, "defect suspect: 15")
    assert route.defect_suspects == (f"{floor.FAMILY}/x_known__l__j15__t_reference",)
    mutated = floor.declared_route(
        "x", {key: value for key, value in cells.items() if key[2] <= 10}, anchor=10
    )
    assert mutated.route == floor.LOWERED and mutated.floor == 4


def test_the_route_reads_a_red_at_the_anchor_and_its_diagnosis():
    """M3, B3: a red at the anchor is a defect suspect until a diagnosis records no defect."""
    cells = _family_cells({10: "positive"})
    name = f"{floor.FAMILY}/x_known__l__j10__t_reference"
    route = floor.declared_route("x", cells, anchor=10)
    assert route.route == "defect suspect: 10" and route.owners[name] is None
    diagnosed = floor.declared_route("x", cells, anchor=10, diagnoses={name: "no defect"})
    assert diagnosed.owners[name] == (floor.RED_OWNER, floor.CHANCE_READING)
    # The mutation that treats the anchor as a lower count reads a below-floor red instead.
    mutated = floor.declared_route("x", cells, anchor=15)
    assert mutated.route == "kept: 10 fails"
    assert mutated.owners[name] == (floor.RED_OWNER, floor.FINITE_SAMPLE_READING)


def test_below_floor_readings():
    """B3: availability needs the withheld-and-lost share and a calibrated shown interval."""
    cells = _family_cells({5: "positive", 4: "control"})
    route = floor.declared_route("x", cells, anchor=10)
    assert (route.floor, route.route) == (6, floor.LOWERED)
    assert route.owners[f"{floor.FAMILY}/x_known__l__j4__iid_t_control"] == (
        floor.RED_OWNER,
        floor.CONTROL_READING,
    )
    assert route.owners[f"{floor.FAMILY}/x_known__l__j5__t_reference"] == (
        floor.RED_OWNER,
        floor.FINITE_SAMPLE_READING,
    )
    available = _summary_row(
        "p",
        False,
        withheld_draws=100.0,
        replicates=3_900.0,
        shown_ci_lower=0.94,
        shown_ci_upper=0.96,
    )
    assert floor.red_reading(available) == floor.AVAILABILITY_READING
    # Mutations: a share of 2.5%, or a shown interval that misses 0.95, is finite-sample.
    assert floor.red_reading({**available, "replicates": 4_000.0}) == floor.FINITE_SAMPLE_READING
    shifted = {**available, "shown_ci_upper": 0.94}
    assert floor.red_reading(shifted) == floor.FINITE_SAMPLE_READING


def _decision(winner: str, jackknife: int | None = 10) -> floor.Decision:
    reasons = {} if jackknife is not None else {floor.JACKKNIFE_ARM: "clause (i) fails at x"}
    return floor.Decision(
        {floor.REFERENCE_ARM: 10, floor.WANG_ARM: 10, floor.JACKKNIFE_ARM: jackknife},
        {},
        winner,
        reasons,
    )


def _routes() -> dict[str, floor.Route]:
    return {
        family: floor.Route(family, 6, floor.LOWERED, {})
        for family in ("tmle_in_sample", "tmle_stacked", "ltmle_in_sample", "ltmle_cross_fitted")
    }


def test_wang_is_adopted_only_when_it_wins_every_family():
    """M3, rule 3: (B) losing one family of four is not adopted anywhere."""
    routes = _routes()
    every = {family: _decision(floor.WANG_ARM) for family in routes}
    assert all(outcome.wang_adopted for outcome in floor.outcomes(routes, every).values())
    one_lost = {**every, "ltmle_cross_fitted": _decision(floor.REFERENCE_ARM)}
    outcomes = floor.outcomes(routes, one_lost)
    assert not any(outcome.wang_adopted for outcome in outcomes.values())
    mutated = any(decision.winner == floor.WANG_ARM for decision in one_lost.values())
    assert mutated


def test_the_jackknife_status_per_family():
    """M3, rules 4 to 6: default with its floor, option, or refused with the measured reason."""
    routes = _routes()
    decisions = {
        "tmle_in_sample": _decision(floor.JACKKNIFE_ARM),
        "tmle_stacked": _decision(floor.REFERENCE_ARM),
        "ltmle_in_sample": _decision(floor.REFERENCE_ARM, jackknife=None),
        "ltmle_cross_fitted": _decision(floor.REFERENCE_ARM),
    }
    outcomes = floor.outcomes(routes, decisions)
    assert (outcomes["tmle_in_sample"].jackknife, outcomes["tmle_in_sample"].floor) == (
        "default",
        10,
    )
    assert outcomes["tmle_stacked"].jackknife == "option"
    assert outcomes["ltmle_in_sample"].jackknife == "refused: clause (i) fails at x"
    assert outcomes["ltmle_in_sample"].floor == 6
    # The mutation that reads the winner alone ships the inadmissible (C) as an option.
    mutated = "option" if decisions["ltmle_in_sample"].winner != floor.JACKKNIFE_ARM else "default"
    assert mutated != outcomes["ltmle_in_sample"].jackknife


def test_candidate_reds_are_held():
    summary = pd.DataFrame(
        {
            "property": [floor.CANDIDATE_FAMILY, floor.POOLED_FAMILY, floor.FAMILY],
            "cell": ["a", "b", "c"],
            "property_passed": [False, False, False],
        }
    )
    assert floor.candidate_owners(summary) == {
        f"{floor.CANDIDATE_FAMILY}/a": (floor.RED_OWNER, floor.CANDIDATE_READING),
        f"{floor.POOLED_FAMILY}/b": (floor.RED_OWNER, floor.CANDIDATE_READING),
    }


# ------------------------------------------------------------ after the pilot and the run


def _committed(study, name: str):
    path = study.STUDY.artifacts / name
    return path if path.exists() else None


@pytest.mark.parametrize(("study", "properties"), STUDIES)
def test_the_declaration_reads_the_committed_pilot(study, properties):
    """M4: once the pilot is committed, R per family, the predicted reds and their owners, and
    (C)'s shortfalls are those it reads. Before it, the declaration holds the plan's R."""
    path = _committed(study, "pilot-summary.csv")
    replicates = study.PROPERTY_REPLICATES
    replicates = replicates if isinstance(replicates, dict) else {study.FIT: replicates}
    if path is None:
        assert set(replicates.values()) == {4_000}
        assert not study.PILOT_PREDICTED_REDS and not study.PILOT_SHORTFALLS
        assert not study.JACKKNIFE_SHORTFALLS
        return
    summary = pd.read_csv(path, keep_default_na=False, na_values=[""])
    sized = properties.sizing(summary)
    sized = sized if isinstance(sized, dict) else {study.FIT: sized}
    for family, result in sized.items():
        assert replicates[family] == result.replicates
        if result.plugin < floor.FAMILY_PASS_TARGET:
            assert result.replicates == floor.R_CAP and family in study.PILOT_SHORTFALLS
    table = pd.read_csv(study.STUDY.artifacts / "pilot-sizing.csv", keep_default_na=False)
    reds = table.loc[table["kind"] == "predicted red"]
    assert dict(zip(reds["cell"], reds["reading"], strict=True)) == study.PILOT_PREDICTED_REDS
    assert set(study.PILOT_PREDICTED_REDS.values()) <= set(floor.READINGS)
    candidates = pd.read_csv(study.STUDY.artifacts / "pilot-candidates.csv")
    shortfalls = set(candidates.loc[candidates["shortfall"].astype(bool), "family"])
    assert shortfalls == study.JACKKNIFE_SHORTFALLS


def test_the_committed_runs_reproduce_the_shipped_floors():
    """M1: once both runs are committed, the floors, the decisions and the outcome recomputed
    from their rows are the shipped ``CLUSTER_INTERVAL_FLOORS`` and (C) status."""
    paths = [_committed(study, "property-replicates.parquet") for study in (study_a, study_b)]
    if not all(paths):
        assert not any(paths), "commit both runs together: rule 3 reads them jointly"
        return
    rows_a, rows_b = (read_rows(path) for path in paths)
    summary_a = properties_a.summarize_properties(rows_a)
    summary_b = properties_b.summarize_properties(rows_b)
    outcomes = floor.declared_outcome(summary_a, rows_a, summary_b, rows_b)
    for outcome in outcomes.values():
        assert CLUSTER_INTERVAL_FLOORS[outcome.family] == outcome.floor
        shipped = floor.SHIPPED_JACKKNIFE.get(outcome.family, "not compared")
        assert outcome.jackknife.split(":")[0] == shipped
