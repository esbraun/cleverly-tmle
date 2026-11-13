"""The declaration of the ``band-t-reference`` study, before its pilot and run (Fix-2 plan 5).

The sizing rule reproduces the plan's planning numbers.  The clause formulas, the seed streams,
the cells and the row reader are checked here, each with a control that fails when the
component it guards is wrong: the nonzero witness, the df agreement with the package, the
expansion gate, the constant-cell sentinel and the paired-gain rule.  One replication of each
fit shape runs through the declared rows.  Nothing here starts the pilot or the run.
"""

from __future__ import annotations

import functools
import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from cleverly._support import registered_rows
from cleverly._support_rows import T_BAND_CELLS, UNMAPPED_BAND_CELLS
from cleverly.inference import multiplier
from tests.studies import band_t_reference as study
from tests.studies import band_t_reference_held_out as held_out
from tests.studies import band_t_reference_held_out_properties as held_out_properties
from tests.studies import band_t_reference_properties as properties
from tests.studies import default_band_properties
from tests.studies.evidence import simultaneous as joint
from tests.studies.evidence.property_verdicts import reference_gain_verdict
from tests.studies.evidence.red_cells import REPORTING_ROWS
from tests.studies.evidence.registry import registered
from tests.unit.test_study_provenance import _required_from

pytestmark = pytest.mark.xdist_group("band_t_reference_design")


@functools.cache
def planned() -> tuple[int, float]:
    return study.declared_replicates()


# ------------------------------------------------------------------------------- sizing


def test_the_planning_values_size_the_study_at_5200() -> None:
    """Plan 5.3: the three red shapes with their magnitude clauses bind, at 5,200 (0.958)."""
    replicates, probability = planned()
    assert replicates == study.PLANNED_REPLICATES == 5_200
    assert probability == pytest.approx(0.958, abs=0.002)
    # No pilot is committed yet, so R is the planning value and DECLARED_REPLICATES is unset.
    assert study.read_pilot() is None and study.DECLARED_REPLICATES is None
    assert study.REPLICATES == study.PLANNED_REPLICATES
    assert held_out.REPLICATES == study.REPLICATES


def test_a_committed_pilot_fixes_the_declared_replicates() -> None:
    """Once the pilot is committed, DECLARED_REPLICATES must be the rule on it."""
    pilot = study.read_pilot()
    if pilot is None:
        pytest.skip("the pilot has not run")
    assert study.declared_replicates(pilot)[0] == study.DECLARED_REPLICATES


def test_the_licensing_family_alone_sizes_lower() -> None:
    """The eleven licensing cells, pooled with every clause, read 3,600 (0.960) [guess]."""
    coverages, sizes, clauses = study._sizing_inputs(study.LICENSING, None)
    assert len(coverages) == 11 and len(clauses) == 9
    replicates, probability = study.size(coverages, sizes, clauses=clauses)
    assert replicates == 3_600
    assert probability == pytest.approx(0.960, abs=0.002)


def test_the_clauses_raise_the_red_family_above_the_cells_alone() -> None:
    """Control: the same red family without its clauses sizes lower, so the clauses bind."""
    coverages, sizes, _ = study._sizing_inputs(study.RED_CELLS, None)
    replicates, _ = study.size(coverages, sizes)
    assert replicates is not None and replicates < planned()[0]


# ---------------------------------------------------------------------- clause formulas


def test_the_magnitude_threshold_is_the_declared_formula() -> None:
    gain, sd, replicates = 0.019, math.sqrt(0.019 * 0.981), 5_200
    sigma_p, sigma_r = sd / math.sqrt(800), sd / math.sqrt(replicates)
    expected = min(0.003, gain - 2.326348 * sigma_p - (2.575829 + 2.326348) * sigma_r)
    assert study.magnitude_threshold(gain, sd, replicates) == pytest.approx(expected, rel=1e-5)
    assert study.magnitude_threshold(0.5, 0.01, replicates) == study.MAGNITUDE_CEILING


def test_a_zero_gain_carries_no_witness_and_a_red_cell_keeps_the_floor() -> None:
    """A ``t`` band with ``nu = inf`` gains 0: no witness is selected for it, and a red cell's
    threshold never falls below the witness floor."""
    red = study.CELL_BY_LABEL["strata"]
    witness = study.CELL_BY_LABEL["strata_n8000"]
    assert not study.carries_witness(0.0, 0.0, 5_200)
    assert study.clause_threshold(witness, 0.0, 0.0, 5_200) is None
    assert study.clause_threshold(red, 0.0, 0.0, 5_200) == joint.SCALE_GAIN_FLOOR
    assert study.clause_threshold(study.CELL_BY_LABEL["point_msm"], 0.05, 0.2, 5_200) is None
    assert study.carries_witness(0.03, math.sqrt(0.03 * 0.97), 5_200)
    assert not study.carries_witness(0.01, math.sqrt(0.01 * 0.99), 5_200)


# ---------------------------------------------------------------------- cells and seeds


def test_the_cells_are_the_plan_cells() -> None:
    labels = {cell.label for cell in study.CELLS}
    assert labels == {
        "categorical_ltmle",
        "cde_z0",
        "strata",
        "categorical_ltmle_n8000",
        "cde_z0_n8000",
        "strata_n8000",
        "categorical_ltmle_crossfit",
        "crossfit_strata",
        "six_regimens",
        "ordinary_binary",
        "point_msm",
    }
    declared = {cell.cell for cell in properties.declared_cells()}
    assert declared == set(study.PROPERTY_CELLS[joint.FAMILY])
    assert not any("strata_n1000" in name for name in declared)


def test_the_held_out_cell_is_its_own_reporting_study() -> None:
    """B1: a red held-out verdict must block nothing in band-t-reference."""
    cell = study.HELD_OUT_CELL
    assert (cell,) == held_out.CELLS
    assert cell.group == study.HELD_OUT and cell.rows == () and cell not in study.LICENSING
    assert {arm.suffix for arm in cell.arms} >= {"floor_2_diagnostic", "floor_10_diagnostic"}
    assert held_out.STUDY.publication_policy == "reporting"
    assert study.STUDY.publication_policy == "gated"
    names = {item.cell for item in held_out_properties.declared_cells()}
    assert names == set(held_out.PROPERTY_CELLS[joint.FAMILY])
    assert {
        f"strata_n1000__{suffix}" for suffix in (joint.T_BAND, joint.REFERENCE_CONTROL)
    } <= names
    assert not any(key.slug in {study.SLUG, held_out.SLUG} for key in REPORTING_ROWS)
    unmapped = f"band_t_reference_held_out:{joint.FAMILY}/strata_n1000__t_band"
    assert unmapped in UNMAPPED_BAND_CELLS
    assert held_out.STUDY.artifacts.name == "band_t_reference_held_out"


def test_each_licensed_row_exists_and_the_mapping_waits_for_the_run() -> None:
    rows = {row.id for row in registered_rows()}
    for cell in study.LICENSING:
        assert cell.rows and set(cell.rows) <= rows, cell.label
    assert dict(T_BAND_CELLS) == {}


RECORDS = {
    "main": (study, "tests.studies.band_t_reference_properties"),
    "held_out": (
        held_out,
        "tests.studies.band_t_reference_held_out_properties",
    ),
}


def test_the_studies_are_declared_and_not_registered() -> None:
    slugs = {record.slug for record in registered()}
    assert study.SLUG not in slugs and held_out.SLUG not in slugs


@pytest.mark.parametrize("which", sorted(RECORDS))
def test_each_record_lists_every_module_its_entry_points_reach(which: str) -> None:
    """M1: the set test_study_provenance requires of a registered study, checked here before
    registration."""
    module, properties_module = RECORDS[which]
    record = module.STUDY
    assert record.runner_module == module.__name__
    assert record.properties_module == properties_module
    missing = _required_from(record.runner_module, record.properties_module) - set(record.modules)
    assert missing == frozenset()
    assert len(set(record.modules)) == len(record.modules)


def test_the_seeds_are_unique() -> None:
    """No registered study shares a seed or resampling seed with either declared study."""
    taken = {
        seed
        for record in registered()
        for seed in (record.seed, record.resampling_seed)
        if seed is not None
    }
    ours = [study.SEED, study.RESAMPLING_SEED, held_out.SEED, held_out.RESAMPLING_SEED]
    assert len(set(ours)) == 4
    assert taken.isdisjoint(ours)


def test_the_streams_are_the_study_own() -> None:
    for label in ("strata", "cde_z0", "categorical_ltmle"):
        run = properties.seed(label, 0)
        assert run != properties.seed(label, 0, pilot=True)
        assert run != default_band_properties._seed(label, 0)
        assert run == properties.payloads((label,), 1)[0][0][3]
    label = study.HELD_OUT_CELL.label
    mine = properties.seed(label, 0, declaration=held_out.DECLARATION)
    assert mine != properties.seed(label, 0, pilot=True, declaration=held_out.DECLARATION)
    assert mine != properties.seed(label, 0)
    assert held_out.PILOT_STREAM != study.PILOT_STREAM


# --------------------------------------------------------------------- one replication


def _declaration(label: str) -> study.Declaration:
    return held_out.DECLARATION if label == study.HELD_OUT_CELL.label else study.DECLARATION


@functools.cache
def _one(label: str) -> tuple[Any, dict[str, float], pd.DataFrame]:
    cell = study.CELL_BY_LABEL[label]
    shape = properties.shape(cell.source)
    seed = properties.seed(label, 0, pilot=True, declaration=_declaration(label))
    frame, truth = shape.draw(cell.n, seed)
    result, full = properties.fit_cell(cell, frame, truth)
    return result, full, frame


@pytest.mark.parametrize("label", ["strata_n1000", "ordinary_binary", "cde_z0", "point_msm"])
def test_one_replication_fills_its_declared_cells(label: str) -> None:
    key = properties.payloads((label,), 1, pilot=True, declaration=_declaration(label))[0]
    rows, record = properties.fit_replication(key, properties._draw(key))
    assert {f"{row['property']}/{row['cell']}" for row in rows} == set(properties._cells_of(key))
    assert record["min_df"] >= joint.T_DF_FLOOR
    assert record["max_relative_widening"] > 0.0
    assert len(str(record["critical_values"]).split(";")) == len(str(record["df"]).split(";"))


def _shift(estimate: Any, deviation: float) -> float:
    """The truth whose range-logit standardized deviation from ``estimate`` is ``deviation``."""
    low, high = (float(value) for value in estimate.bounds)
    psi, se = float(estimate.psi), float(estimate.plugin_std_error)
    center = math.log((psi - low) / (high - psi))
    spread = se * (high - low) / ((psi - low) * (high - psi))
    return low + (high - low) * float(expit(center + deviation * spread))


def _between(label: str) -> tuple[Any, dict[str, float], str]:
    """A truth with one coordinate between ``c`` and its ``c_j``, the rest at the estimate."""
    result, _, _ = _one(label)
    band = result.simultaneous
    names = tuple(result.estimates)
    kish = {name: joint.study_kish(result[name].influence_curve) for name in names}
    target = min(names, key=lambda name: kish[name])
    c = float(band.critical_value)
    c_j = joint.t_quantile(c, max(kish[target], joint.T_DF_FLOOR))
    truth = {name: float(result[name].psi) for name in names}
    truth[target] = _shift(result[target], (c + c_j) / 2.0)
    return result, truth, target


def _rows(result: Any, truth: dict[str, float], constant: tuple[str, ...] = ()) -> dict[str, Any]:
    rows, _ = joint.t_band_rows(
        result,
        truth,
        tuple(result.estimates),
        label="strata_n1000",
        replicate=0,
        n=1_000,
        requested=1,
        pointwise_critical=1.96,
        constant_cells=constant,
        arms=(),
    )
    return {str(row["cell"]).split("__", 1)[1]: row for row in rows}


def test_a_deviation_between_c_and_c_j_is_the_nonzero_witness() -> None:
    """The ``t`` band covers and the normal band misses: a gain of one on this replicate."""
    result, truth, _ = _between("strata_n1000")
    rows = _rows(result, truth)
    assert rows[joint.T_BAND]["covered"] == 1
    assert rows[joint.REFERENCE_CONTROL]["covered"] == 0


def test_a_package_df_off_the_study_df_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Control: a study df of ``inf`` (the normal band) disagrees with the package's df."""
    result, truth, _ = _between("strata_n1000")
    monkeypatch.setattr(joint, "study_kish", lambda curve, cluster=None: math.inf)
    with pytest.raises(AssertionError, match="package df"):
        _rows(result, truth)


def test_a_package_c_j_off_the_study_c_j_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """M2 control: one package ``c_j`` replaced by ``c`` disagrees with the row module's."""
    result, truth, _ = _between("strata_n1000")
    original = multiplier.t_critical_values

    def one_replaced(critical: float, curves: Any, cluster: Any = None) -> Any:
        nus, values = original(critical, curves, cluster)
        values = np.array(values, dtype=float)
        values[0] = critical
        return nus, values

    monkeypatch.setattr(multiplier, "t_critical_values", one_replaced)
    with pytest.raises(AssertionError, match="package c_j"):
        _rows(result, truth)


def test_a_t_band_that_widens_nothing_fails_the_deterministic_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M3 control: a package and a row module that both read ``c_j = c`` agree with each other
    and still fail, because some Kish count is finite."""
    result, truth, _ = _between("strata_n1000")
    original = multiplier.t_critical_values

    def collapsed(critical: float, curves: Any, cluster: Any = None) -> Any:
        nus, values = original(critical, curves, cluster)
        return nus, np.full(len(values), critical)

    monkeypatch.setattr(multiplier, "t_critical_values", collapsed)
    monkeypatch.setattr(joint, "t_quantile", lambda critical, nu: float(critical))
    # The expansion gate would also fire at nu >= 50; switched off to isolate the witness.
    monkeypatch.setattr(joint, "check_expansion", lambda critical, nus, values: None)
    with pytest.raises(AssertionError, match="does not widen"):
        _rows(result, truth)


def test_the_widening_witness_holds_and_skips_an_all_infinite_family() -> None:
    joint.check_widening(2.5, {"a": 12.0, "b": math.inf}, {"a": 2.9, "b": 2.5}, "x")
    joint.check_widening(2.5, {"a": math.inf}, {"a": 2.5}, "x")
    with pytest.raises(AssertionError, match="does not widen"):
        joint.check_widening(2.5, {"a": 12.0}, {"a": 2.5}, "x")


def test_a_constant_cell_refuses_every_band_row() -> None:
    """The family rule (2.5): the sentinel, not covered, whatever band the fit shows."""
    result, _, _ = _one("strata_n1000")
    rows = _rows(result, {name: float(result[name].psi) for name in result.estimates}, ("A=1|V=2",))
    for suffix in (joint.T_BAND, joint.REFERENCE_CONTROL):
        assert rows[suffix]["covered"] == 0
        assert rows[suffix]["estimate"] == pytest.approx(
            rows[suffix]["std_error"] + joint.REFUSED_MARGIN
        )
    assert rows[joint.CONTROL]["covered"] == 1


def test_the_constant_arm_copy_reads_each_shape() -> None:
    frame = pd.DataFrame(
        {
            "V": [0, 0, 0, 0, 1, 1, 1, 1],
            "A": [0, 0, 1, 1, 0, 0, 1, 1],
            "Y": [0, 1, 0, 1, 0, 1, 1, 1],
        }
    )
    assert properties.constant_arm_cells("strata", frame) == ("V=1|A=1",)
    # An empty stratum-by-arm cell is constant too: the copy reads the full V x A product.
    empty = frame.loc[~((frame["V"] == 1) & (frame["A"] == 1))]
    assert properties.constant_arm_cells("strata", empty) == ("V=1|A=1",)
    assert properties.constant_arm_cells("ordinary_binary", frame) == ()
    frame.loc[frame["A"] == 1, "Y"] = 1
    assert properties.constant_arm_cells("ordinary_binary", frame) == ("A=1",)
    assert properties.constant_arm_cells("six_regimens", frame) == ()
    cde = pd.DataFrame(
        {
            "A": [0, 0, 1, 1, 1],
            "Z": [0, 0, 0, 0, 1],
            "Delta": [1, 1, 1, 0, 1],
            "Y": [0, 1, 1, np.nan, 0],
        }
    )
    assert properties.constant_arm_cells("cde_z0", cde) == ("A=1",)


def test_the_constant_arm_copy_agrees_with_fix3() -> None:
    """Plan 5.1: once Fix-3's PR 1 adds ``constant_outcome_cells``, the copy must agree."""
    from cleverly.inference import influence

    if not hasattr(influence, "constant_outcome_cells"):
        pytest.skip("Fix-3's constant_outcome_cells is not on this branch")
    raise AssertionError("write the agreement fixtures for each shape")  # pragma: no cover


# ---------------------------------------------------------------- expansion and verdict


def test_the_expansion_gate_holds_and_fails_a_wrong_quantile() -> None:
    nus = {"a": 50.0, "b": 400.0, "c": 6.0}
    c = 2.8
    values = {name: joint.t_quantile(c, nu) for name, nu in nus.items()}
    joint.check_expansion(c, nus, values)
    wrong = dict(values, b=c * (1.0 + 1.5 * (c**2 + 1.0) / (4.0 * 400.0)))
    with pytest.raises(AssertionError, match="departs"):
        joint.check_expansion(c, nus, wrong)


def test_the_t_statistic_is_inside_c_exactly_when_the_deviation_is_inside_c_j() -> None:
    c = 2.7
    for nu in (5.0, 12.0, 300.0, math.inf):
        c_j = joint.t_quantile(c, nu)
        assert joint.t_statistic(c_j * (1 - 1e-9), nu) < c < joint.t_statistic(c_j * (1 + 1e-9), nu)


def _gain_rows(gains: np.ndarray) -> pd.DataFrame:
    replicates = np.arange(len(gains))
    band = pd.DataFrame(
        {"cell": "x__t_band", "replicate": replicates, "covered": gains.astype(int)}
    )
    control = pd.DataFrame(
        {"cell": "x__normal_reference_control", "replicate": replicates, "covered": 0}
    )
    frame = pd.concat([band, control], ignore_index=True)
    frame["property"] = joint.FAMILY
    return frame


def test_the_reference_verdict_reads_the_witness_and_the_magnitude() -> None:
    rng = np.random.default_rng(3)
    rows = _gain_rows((rng.uniform(size=4_000) < 0.02).astype(float))
    cell = "x__normal_reference_control"
    low, _, passed = reference_gain_verdict(rows, study.STUDY, cell, joint.SCALE_GAIN_FLOOR)
    assert passed and low > joint.SCALE_GAIN_FLOOR
    assert not reference_gain_verdict(rows, study.STUDY, cell, 0.05)[2]
    zero = _gain_rows(np.zeros(4_000))
    assert not reference_gain_verdict(zero, study.STUDY, cell, joint.SCALE_GAIN_FLOOR)[2]
    assert reference_gain_verdict(zero, study.STUDY, cell, None)[2]
