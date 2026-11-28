"""Design checks of ``few-cluster-arm-means-cross-fitted-ltmle`` (X54 part (k), plan rev 2.4,
4 and 5.3).

The declaration against its published cells, seeds, ladder and gates; the pilot reading,
recomputed from the committed pilot tables once they exist; and the floor and sizing on
constructed summaries, both laws gating clause 1.
"""

from __future__ import annotations

import pandas as pd

from cleverly._inference_status import CLUSTER_INTERVAL_FLOORS
from tests.canonical import property_differential as differential
from tests.studies import few_cluster_arm_means as study_k
from tests.studies import few_cluster_arm_means_ltmle as study
from tests.studies import few_cluster_arm_means_ltmle_properties as properties
from tests.studies import few_cluster_floor as floor
from tests.studies import few_cluster_floor_ltmle as source
from tests.studies import few_cluster_floor_pilot as pilot
from tests.studies import few_cluster_floor_stacked as study_j
from tests.studies.evidence import property_failures
from tests.studies.evidence.registry import registered


def test_published_cells_equal_the_declaration():
    declared = properties.declared_cells()
    assert tuple(cell.cell for cell in declared) == study.PUBLISHED_CELLS
    assert study.STUDY.property_cells == {floor.FAMILY: study.PUBLISHED_CELLS}
    groups = properties.PROPERTY_GROUPS[floor.FAMILY]
    assert tuple(cell for group in groups for cell in group) == study.PUBLISHED_CELLS
    truths = {cell.dgp.name: cell.dgp.truth() for cell in declared}
    for cell in declared:
        estimand, sizes, clusters, arm = study.parse(cell.cell)
        assert study.cell_name(estimand, sizes, clusters, arm) == cell.cell
        assert cell.n == 40 * clusters and cell.estimand == estimand
    # The law's truths do not depend on the count.
    for sizes in study.SIZE_LAWS:
        assert len({tuple(v.items()) for k, v in truths.items() if k.startswith(sizes)}) == 1


def test_declared_record():
    record = study.STUDY
    assert record.slug == "few-cluster-arm-means-cross-fitted-ltmle"
    assert record.publication_policy == "reporting" and record.reference is None
    assert record.failure_bound == {"cleverly": floor.PRIMARY_FAILURE_BOUND}
    assert properties.FAILURE_BOUND == property_failures.MAXIMUM_BOUND
    assert (record.seed, record.resampling_seed) == (20265461, 20265462)
    assert (study.PRIMARY_ESTIMAND, study.PRIMARY_LAW, study.PRIMARY_CLUSTERS) == (
        "ey_regimen[always]",
        "equal40",
        10,
    )
    assert record.replicates == 1_000 and record.n == 400
    for other in (study_j.STUDY, study_k.STUDY, *registered()):
        assert {other.seed, other.resampling_seed}.isdisjoint({record.seed, record.resampling_seed})


def test_the_study_draws_its_own_streams():
    keys = [(law, j, r) for law in study.SIZE_LAWS for j in (4, 10) for r in (0, 1)]
    mine = {study.draw_seed(*key) for key in keys}
    pilots = {study.draw_seed(*key, "pilot") for key in keys}
    theirs = {source.draw_seed(*key) for key in keys}
    assert len(mine) == len(pilots) == len(keys)
    assert mine.isdisjoint(pilots) and mine.isdisjoint(theirs)


def test_the_ladder_laws_and_gates():
    assert (*range(4, 21), 30) == study.LADDER
    assert study.ANCHOR == 20 and study.FLOOR_FAMILY == "ltmle_cross_fitted"
    assert properties.contrast_floor() == CLUSTER_INTERVAL_FLOORS["ltmle_cross_fitted"]
    assert study.SIZE_LAWS == ("equal40", "unequal40") == study.CLAUSE_ONE_LAWS
    for estimand in study.ESTIMANDS:
        roles = dict(study.arms(estimand))
        gated = estimand in study.ARM_MEANS
        assert roles[floor.REFERENCE_ARM] == (floor.POSITIVE if gated else floor.DIAGNOSTIC)
        assert roles["iid_t_control"] == (floor.CONTROL if gated else floor.DIAGNOSTIC)
    assert study.ARM_MEANS == ("ey_regimen[never]", "ey_regimen[always]")
    assert study.CONTRAST == source.TARGET


def test_the_differential_reads_a_declared_subset_of_whole_groups():
    assert properties.__name__ in differential.SYNTHETIC_MODULES
    assert properties.DIFFERENTIAL_BUDGET == 200
    cells = properties.differential_cells()
    for group in properties.PROPERTY_GROUPS[floor.FAMILY]:
        assert len({f"{floor.FAMILY}/{cell}" in cells for cell in group}) == 1, group
    bound = differential.read_bindings()
    assert (bound["subject"] == properties.__name__).any()


def test_the_declaration_reads_the_committed_pilot():
    """Once the pilot is committed: R, the predicted floor and reds, the shortfall, the
    primary's G1 pass probability and the budget are those the sizing code reads off
    ``pilot-summary.csv``. Before it, R = 4,000 and no prediction."""
    path = study.STUDY.artifacts / "pilot-summary.csv"
    if not path.exists():
        assert study.PROPERTY_REPLICATES == 4_000
        assert not study.PILOT_PREDICTED_REDS and not study.PILOT_SHORTFALLS
        assert study.PILOT_PREDICTED_FLOOR is None and study.PILOT_PRIMARY_PASS is None
        return
    summary = pd.read_csv(path, keep_default_na=False, na_values=[""])
    result = properties.family_sizing(summary)
    assert result["replicates"] == study.PROPERTY_REPLICATES
    assert result["predicted_floor"] == study.PILOT_PREDICTED_FLOOR
    assert result["reds"] == study.PILOT_PREDICTED_REDS
    assert (set() if result["target_met"] else {study.FLOOR_FAMILY}) == study.PILOT_SHORTFALLS
    assert properties.primary_pass(summary) == study.PILOT_PRIMARY_PASS
    assert study.PILOT_PRIMARY_PASS >= 0.90
    timing = pd.read_csv(study.STUDY.artifacts / "pilot-timing.csv")
    total = timing.loc[timing["fit"] == "total at the sized R", "projected_cpu_hours"]
    assert float(total.iloc[0]) <= pilot.BUDGET_TOLERANCE * study.DECLARED_CPU_HOURS


def _summary(red: frozenset[str] = frozenset(), weak: frozenset[str] = frozenset()):
    rows = []
    for sizes, clusters in study.grid():
        for estimand in study.ARM_MEANS:
            for arm, role in study.arms(estimand)[:2]:
                name = study.cell_name(estimand, sizes, clusters, arm)
                row = {
                    "property": floor.FAMILY,
                    "cell": name,
                    "role": role,
                    "passed": name not in red and name not in weak,
                    "requested_draws": 400.0,
                    "replicates": 400.0,
                    "withheld_draws": 0.0,
                    "all_draw_coverage": 0.95,
                    "standardized_bias": 0.0,
                    "shown_ci_lower": 0.93,
                    "shown_ci_upper": 0.97,
                    "se_ratio": 1.0,
                    "shown_se_ratio": 1.0,
                    "se_ratio_ci_lower": 0.95,
                    "se_ratio_ci_upper": 1.05,
                }
                if role == floor.CONTROL:
                    high = 0.9 if name in weak else 0.6
                    row.update(
                        paired_se_ratio=high - 0.05,
                        paired_se_ratio_ci_lower=high - 0.1,
                        paired_se_ratio_ci_upper=high,
                        paired_positive_se_ratio_ci_upper=1.05,
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def test_the_floor_gates_clause_one_on_both_laws():
    """Every count passing gives the contrast floor 4; a control failing clause 1 alone at 20 on
    either law turns that count red and raises the floor to 30, held below it; a red at 30
    leaves no certified count."""
    found = properties.route(_summary())
    assert (found.floor, found.route) == (4, "lowered: 4")
    for sizes in study.SIZE_LAWS:
        weak = study.cell_name("ey_regimen[never]", sizes, 20, "iid_t_control")
        found = properties.route(_summary(weak=frozenset({weak})))
        assert (found.floor, found.route) == (30, "raised: 30")
        assert found.owners == {f"{floor.FAMILY}/{weak}": (study.RED_OWNER, floor.CONTROL_READING)}
    top = study.cell_name("ey_regimen[always]", "unequal40", 30, floor.REFERENCE_ARM)
    assert properties.route(_summary(frozenset({top}))).floor == 40


def test_the_pilot_sizing_reads_both_clauses_on_both_laws():
    weak = study.cell_name("ey_regimen[always]", "equal40", 12, "iid_t_control")
    result = properties.family_sizing(_summary(weak=frozenset({weak})))
    assert result["reds"] == {f"{floor.FAMILY}/{weak}": floor.CONTROL_READING}
    assert result["predicted_floor"] == 13 and result["expected_floor"] == 4
    # The weak control lies above the expected floor, so M5 sizes it and stops at the cap.
    assert result["replicates"] == floor.R_CAP and not result["target_met"]
    assert properties.family_sizing(_summary())["replicates"] == 4_000
    table = properties.pilot_sizing_table(_summary())
    assert table["kind"].tolist() == ["family", "primary"]
    assert float(table.loc[1, "g1_pass"]) == floor.g1_pass_probability(0.95, 1_000)
