"""The declaration of the ``band-t-reference`` study, and the reading of its run (Fix-2).

The sizing rule reproduces the committed R from the committed pilot, on coverage alone (plan
rev 3.3).  The clause formulas, the seed streams, the cells and the row reader are checked here,
each with a control that fails when the component it guards is wrong: the nonzero witness, the
df agreement with the package, the expansion gate, the constant-cell sentinel, the joint rule
and the paired-gain reading, which is reported and gates nothing.  One replication of each fit
shape runs through the declared rows.  Nothing here starts the run.  The tables the study pages
quote beside the generated ones are rebuilt here from the committed artifacts.
"""

from __future__ import annotations

import dataclasses
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
from tests.studies.evidence import property_verdicts
from tests.studies.evidence import simultaneous as joint
from tests.studies.evidence.property_verdicts import reference_gain_verdict
from tests.studies.evidence.red_cells import REPORTING_ROWS
from tests.studies.evidence.registry import registered
from tests.unit.test_study_provenance import _required_from

pytestmark = pytest.mark.xdist_group("band_t_reference_design")


# ------------------------------------------------------------------------------- sizing


@functools.cache
def declared() -> tuple[int, float]:
    pilot = study.read_pilot()
    assert pilot is not None
    return study.declared_replicates(pilot)


#: The Monte Carlo tolerance of a family pass probability.  The sizing draws 40,000 families, so
#: its standard error near 0.95 is about 0.0011, and a numpy release that changes the binomial
#: sampler moves the red family at 5,200 (0.94998 here) across the 0.95 target.
SIZING_TOLERANCE = 0.003


def _sizes_at(inputs: tuple[list[float], list[int]], replicates: int) -> None:
    """``replicates`` is the rule's R within the tolerance: it passes and the step below fails."""
    coverages, sizes = inputs
    target = study.SIZING_TARGET
    assert study.family_pass(coverages, sizes, replicates) >= target - SIZING_TOLERANCE
    below = replicates - study.SIZING_STEP
    assert below <= 0 or study.family_pass(coverages, sizes, below) < target + SIZING_TOLERANCE


def test_the_committed_pilot_fixes_the_declared_replicates() -> None:
    """Rev 3.3: the larger of the two pooled coverage rules on the pilot, 5,600 (0.957)."""
    pilot = study.read_pilot()
    assert pilot is not None
    assert study.DECLARED_REPLICATES == study.REPLICATES == 5_600
    red = study._sizing_inputs(study.RED_CELLS, pilot)
    licensing = study._sizing_inputs(study.LICENSING, pilot)
    assert study.family_pass(*red, study.REPLICATES) == pytest.approx(0.957, abs=SIZING_TOLERANCE)
    assert study.family_pass(*licensing, study.REPLICATES) >= study.SIZING_TARGET
    _sizes_at(red, study.REPLICATES)
    # The rule itself may land one step lower where the sampler puts 5,200 above the target.
    assert abs(declared()[0] - study.REPLICATES) <= study.SIZING_STEP
    assert held_out.REPLICATES == study.REPLICATES
    assert study.DECLARATION.replicates == held_out.DECLARATION.replicates == study.REPLICATES


def test_the_red_family_decides_r_and_the_licensing_family_sizes_lower() -> None:
    """The three red shapes need 5,600 (0.94998 at 5,200); the eleven licensing cells 2,400."""
    pilot = study.read_pilot()
    red = study._sizing_inputs(study.RED_CELLS, pilot)
    licensing = study._sizing_inputs(study.LICENSING, pilot)
    assert len(licensing[0]) == 11 and len(red[0]) == 3
    _sizes_at(red, 5_600)
    _sizes_at(licensing, 2_400)


def test_a_committed_r_off_the_rule_fails_the_sizing_check() -> None:
    """Control: R one step too high, or one step too low, fails :func:`_sizes_at`."""
    pilot = study.read_pilot()
    red = study._sizing_inputs(study.RED_CELLS, pilot)
    with pytest.raises(AssertionError):
        _sizes_at(red, 6_400)
    with pytest.raises(AssertionError):
        _sizes_at(red, 4_400)


def test_a_family_that_cannot_reach_the_target_refuses_to_size() -> None:
    """Control: a coverage at the rule's edge sizes no R below the ceiling."""
    assert study.size([0.921, 0.921, 0.921], [800, 800, 800], ceiling=1_600)[0] is None


def test_the_committed_thresholds_are_the_formulas_on_the_pilot() -> None:
    """The declared gain values are still computed, and the committed ones match them."""
    assert properties.declared_thresholds(study.DECLARATION) == study.REFERENCE_THRESHOLDS
    assert properties.declared_thresholds(held_out.DECLARATION) == held_out.REFERENCE_THRESHOLDS
    assert set(study.REFERENCE_THRESHOLDS) == {cell.label for cell in study.CELLS}
    pilot = study.read_pilot()
    assert pilot is not None
    witness_only = study.witness_only_red_cells(pilot, study.REPLICATES)
    assert witness_only == [cell.label for cell in study.RED_CELLS]


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


def test_each_licensed_row_exists_and_the_mapping_holds_every_passing_cell() -> None:
    rows = {row.id for row in registered_rows()}
    mapped = {cell for cells in T_BAND_CELLS.values() for cell in cells}
    for cell in study.LICENSING:
        assert cell.rows and set(cell.rows) <= rows, cell.label
        assert f"band_t_reference:{joint.FAMILY}/{cell.label}__t_band" in mapped, cell.label


RECORDS = {
    "main": (study, "tests.studies.band_t_reference_properties"),
    "held_out": (
        held_out,
        "tests.studies.band_t_reference_held_out_properties",
    ),
}


def test_both_studies_are_registered() -> None:
    records = {record.slug: record for record in registered()}
    assert records[study.SLUG] is study.STUDY
    assert records[held_out.SLUG] is held_out.STUDY


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
    """No other registered study shares a seed or resampling seed with either study."""
    taken = {
        seed
        for record in registered()
        if record.slug not in {study.SLUG, held_out.SLUG}
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
    """Plan 5.1: the copy agrees with Fix-3's ``constant_outcome_cells`` on each shape."""
    from cleverly.inference import influence

    def package(source: str, frame: pd.DataFrame) -> tuple[str, ...]:
        y, a = frame["Y"].to_numpy(), frame["A"].to_numpy()
        if source == "strata":
            cells = []
            for v in sorted(frame["V"].unique()):
                rows = (frame["V"] == v).to_numpy()
                arms = influence.constant_outcome_cells(y, a, rows)
                cells += [f"V={v}|A={arm}" for arm in sorted(arms)]
            return tuple(cells)
        if source == "cde_z0":
            observed = (frame["Delta"] == 1).to_numpy()
            arms = influence.constant_outcome_cells(
                y, a, (frame["Z"] == 0).to_numpy(), observed=observed
            )
            return tuple(f"A={arm}" for arm in sorted(arms))
        arms = influence.constant_outcome_cells(y, a, np.ones(len(frame), dtype=bool))
        return tuple(f"A={arm}" for arm in sorted(arms))

    strata = pd.DataFrame(
        {
            "V": [0, 0, 0, 0, 1, 1, 1, 1],
            "A": [0, 0, 1, 1, 0, 0, 1, 1],
            "Y": [0, 1, 0, 1, 0, 1, 1, 1],
        }
    )
    varied = strata.assign(Y=[0, 1, 0, 1, 0, 1, 0, 1])
    constant = strata.assign(Y=[0, 1, 1, 1, 0, 1, 0, 1])
    cde = pd.DataFrame(
        {
            "A": [0, 0, 1, 1, 1, 1],
            "Z": [0, 0, 0, 0, 1, 1],
            "Delta": [1, 1, 1, 0, 1, 1],
            "Y": [0, 1, 1, np.nan, 0, 1],
        }
    )
    cases = [
        ("strata", strata, ("V=1|A=1",)),
        ("strata", varied, ()),
        ("ordinary_binary", varied, ()),
        ("ordinary_binary", constant, ()),
        ("ordinary_binary", strata.assign(Y=[0, 1, 1, 1, 0, 1, 1, 1]), ("A=1",)),
        ("cde_z0", cde, ("A=1",)),
        ("cde_z0", cde.assign(Y=[0, 1, 1, 0, 0, 1], Delta=[1, 1, 1, 1, 1, 1]), ()),
    ]
    for source, frame, expected in cases:
        assert properties.constant_arm_cells(source, frame) == expected, source
        assert package(source, frame) == expected, source
    # One deliberate difference: the package skips an arm with no qualifying row, while the
    # copy counts an empty stratum-by-arm cell, because the family still reports a contrast
    # of that arm and Fix-3's fit refuses it on the empty cell.
    empty = strata.loc[~((strata["V"] == 1) & (strata["A"] == 1))]
    assert properties.constant_arm_cells("strata", empty) == ("V=1|A=1",)
    assert package("strata", empty) == ()


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


# ------------------------------------------------------------- the gated and reported reads


def _family_rows(band: float, *, gain: float = 0.0, replicates: int = 1_200) -> pd.DataFrame:
    """Every declared cell of ``ordinary_binary`` over ``replicates`` copies of one replicate.

    The ``t`` band covers on a ``band`` share of replicates, the normal reference on a share
    ``gain`` lower (each covered replicate of the reference is covered by the band), and the
    pointwise control on half.
    """
    cell = study.CELL_BY_LABEL["ordinary_binary"]
    key = properties.payloads((cell.label,), 1, pilot=True)[0]
    rows, _ = properties.fit_replication(key, properties._draw(key))
    base = pd.DataFrame(rows)
    index = np.arange(replicates)
    t_covered = (index % 1_000) < round(band * 1_000)
    normal_covered = (index % 1_000) < round((band - gain) * 1_000)
    frames = []
    for replicate in index:
        frame = base.copy()
        frame["replicate"] = replicate
        frame["requested_replicates"] = replicates
        suffix = frame["cell"].str.split("__", n=1).str[1]
        frame.loc[suffix == joint.T_BAND, "covered"] = int(t_covered[replicate])
        frame.loc[suffix == joint.REFERENCE_CONTROL, "covered"] = int(normal_covered[replicate])
        frame.loc[suffix == "pointwise_joint_control", "covered"] = int(replicate % 2)
        frames.append(frame)
    return properties._ordered(pd.concat(frames, ignore_index=True))


def _declared_with_a_value() -> study.Declaration:
    """``ordinary_binary`` alone, given the witness value its group never declares."""
    cell = study.CELL_BY_LABEL["ordinary_binary"]
    return dataclasses.replace(
        study.DECLARATION,
        cells=(cell,),
        replicates=1_200,
        reference_thresholds={cell.label: joint.SCALE_GAIN_FLOOR},
    )


def test_a_zero_gain_is_reported_and_gates_nothing() -> None:
    """Rev 3.3: a reference control below its declared value passes; the reading says so."""
    declaration = _declared_with_a_value()
    summary = properties.summarize(declaration, _family_rows(0.95)).set_index("cell")
    assert bool(summary.loc[f"ordinary_binary__{joint.T_BAND}", "passed"])
    assert bool(summary.loc[f"ordinary_binary__{joint.REFERENCE_CONTROL}", "passed"])
    assert bool(summary.loc["ordinary_binary__pointwise_joint_control", "passed"])
    readings = properties.reference_readings(declaration, summary.reset_index())
    assert readings["gain_ci_lower"].iloc[0] == 0.0
    assert not bool(readings["reached"].iloc[0])
    gained = properties.summarize(declaration, _family_rows(0.95, gain=0.05)).reset_index()
    assert bool(properties.reference_readings(declaration, gained)["reached"].iloc[0])


def test_the_joint_rule_still_gates_the_t_band_and_the_pointwise_control() -> None:
    """Mutation control: a t band at 0.85, or a control at full coverage, fails its verdict."""
    declaration = _declared_with_a_value()
    low = properties.summarize(declaration, _family_rows(0.85)).set_index("cell")
    assert not bool(low.loc[f"ordinary_binary__{joint.T_BAND}", "passed"])
    rows = _family_rows(0.95)
    rows.loc[rows["cell"] == "ordinary_binary__pointwise_joint_control", "covered"] = 1
    full = properties.summarize(declaration, rows).set_index("cell")
    assert not bool(full.loc["ordinary_binary__pointwise_joint_control", "passed"])


@pytest.mark.parametrize("module", [properties, held_out_properties])
def test_the_failure_accounting_gates_coverage_and_not_the_gain(module: Any) -> None:
    """The declared rules hold the joint rule's clauses and no reference-gain clause."""
    (rule,) = module.PROPERTY_RULES
    statistics = {clause.statistic for clause in rule.clauses}
    assert "S2" in statistics and "S7r" not in statistics
    plain = property_verdicts.simultaneous_rule()
    assert [clause.statistic for clause in rule.clauses] == [
        clause.statistic for clause in plain.clauses
    ]


@pytest.mark.parametrize("module", [study, held_out], ids=["main", "held_out"])
def test_the_configuration_names_no_key_the_record_owns(module: Any) -> None:
    """``write_manifest`` refuses a configuration key the record owns, after the fits ran."""
    from tests.studies.evidence.manifest import RECORD_CONFIGURATION_KEYS

    assert not RECORD_CONFIGURATION_KEYS & set(module.CONFIGURATION)


# ----------------------------------------------------------------- the published reading


def _page_table(path: str, first: str) -> list[list[str]]:
    """The rows of the pipe table on ``path`` whose header starts with ``first``."""
    from tests.studies.evidence.registry import ROOT

    lines = (ROOT / path).read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"| {first} |"))
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows


def _replicates(record: Any) -> pd.DataFrame:
    return pd.read_csv(record.artifact("t-band-replicates.csv"), low_memory=False)


def test_the_main_result_table_is_the_committed_reading() -> None:
    """The Result table of band-t-reference.md: coverage, gain, declared value, df and floor."""
    summary = pd.read_csv(study.STUDY.artifact("properties.csv")).set_index("cell")
    readings = properties.reference_readings(study.DECLARATION, summary.reset_index())
    readings = readings.set_index("label")
    replicates = _replicates(study.STUDY).groupby("label")
    rows = _page_table(
        "docs/technical-reference/method-evidence/band-t-reference.md", "cell | `t` band"
    )
    assert [row[0].strip("`") for row in rows] == [cell.label for cell in study.CELLS]
    for row in rows:
        label = row[0].strip("`")
        t_band = summary.loc[f"{label}__{joint.T_BAND}"]
        control = summary.loc[f"{label}__{joint.REFERENCE_CONTROL}"]
        assert row[1] == f"{t_band.coverage:.4f}", label
        assert row[2] == f"{control.coverage:.4f}", label
        low, high = readings.loc[label, ["gain_ci_lower", "gain_ci_upper"]]
        assert row[3] == f"{low:.4f} to {high:.4f}", label
        declared_value = study.REFERENCE_THRESHOLDS[label]
        assert row[4] == ("none" if declared_value is None else f"{declared_value:.4f}"), label
        assert row[5] == f"{replicates.get_group(label).min_df.median():.1f}", label
        assert row[6] == f"{100 * replicates.get_group(label).floor_bound.mean():.1f}%", label
        # Every gain interval lies above zero, as the page says, and none gates.
        assert low > 0.0 and bool(control.passed), label


def test_the_held_out_counts_are_the_committed_reading() -> None:
    """The counts band-t-reference-held-out.md quotes from t-band-replicates.csv."""
    replicates = _replicates(held_out.STUDY)
    summary = pd.read_csv(held_out.STUDY.artifact("properties.csv")).set_index("cell")
    assert len(replicates) == 5_600
    assert int(replicates.constant_cells.notna().sum()) == 54
    assert f"{100 * replicates.floor_bound.mean():.1f}" == "56.9"
    assert replicates.min_df.median() == 5.0
    coverage = summary.coverage
    assert (
        f"{coverage['strata_n1000__floor_2_diagnostic']:.4f}"
        == f"{coverage['strata_n1000__floor_10_diagnostic']:.4f}"
        == f"{coverage['strata_n1000__t_band']:.4f}"
        == "0.9543"
    )
    assert bool(summary.passed.all())


def test_the_main_page_says_the_gains_gate_nothing() -> None:
    from tests.studies.evidence.registry import ROOT

    text = (ROOT / study.STUDY.document).read_text(encoding="utf-8")
    assert "**The paired gains gate nothing.**" in text
