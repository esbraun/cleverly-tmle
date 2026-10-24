"""The printed summaries of the sensitivity analyses (X89 T4).

The contract test runs items 1 to 7 over the fixtures in
``tests/unit/summary_fixtures/sensitivity.py``. This file adds what the contract cannot
see: a literal check for each note no compact fixture fires, mutation controls for the
fidelity readers, the plain wording of each rule that withholds limits, and the new return
objects' pickling, caching and frame access.
"""

from __future__ import annotations

import dataclasses
import functools
import pickle
import warnings
from typing import Any

import numpy as np
import pytest

from cleverly._assessment_cache import _cache_key
from cleverly._summary import Summary, _render, vocabulary
from cleverly.sensitivity import MissingnessTilt, RobustnessValue, SensitivityContour
from cleverly.sensitivity.simulated_confounding import SEPARATION_METHOD
from cleverly.validation._interpretation import Resolution
from cleverly.validation.simulation import ReplicationFailure
from tests.unit.summary_fixtures import _sensitivity_readers as read
from tests.unit.summary_fixtures import collect

TEXTS = vocabulary.NOTE_TEXTS
FIXTURES = {key: case for key, case in collect().items() if key.startswith("sensitivity:")}


@functools.cache
def built(name: str) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return FIXTURES[f"sensitivity:{name}"]()


# ----------------------------------------------------------------- note coverage


#: Every note code the sensitivity summaries print.
T4_CODES = {
    "sensitivity_assumption",
    "no_inference",
    "no_inference_sensitivity",
    "limits_withheld",
    "limits_descriptive",
    "rv_limit",
    "alignment_stated",
    "joint_strength",
    "consistent_mechanism",
    "strength_clipped",
    "grid_thinned",
    "approximate_scale",
    "censoring_conditions",
    "per_arm_conditions",
    "bound_truncated",
    "tilt_diagnostic",
    "association_reading",
    "population_moves",
    "incremental_reading",
    "learned_rule_reading",
    "observation_reads",
    "bounded_rows",
    "redrawn_folds",
    "cells_failed",
    "calibration_nan",
}

#: Codes a literal check fires: in this file, or, for the E-value conditions and the two
#: estimand readings, in the test files that build those fits.
LITERAL_CODES = {
    "no_inference_sensitivity",  # test_a_diagnostic_status_marks_the_limits
    "strength_clipped",  # test_a_clipped_benchmark_drops_the_pointer
    "bound_truncated",  # test_a_truncated_ratio_bound_says_so
    "observation_reads",  # test_each_surface_condition_prints_when_it_holds
    "bounded_rows",
    "redrawn_folds",
    "cells_failed",  # test_a_failed_cell_prints_its_error
    "calibration_nan",  # test_a_separated_calibration_prints_its_note
    "censoring_conditions",  # test_evalue_part_g.py, the time-to-event ratio
    "per_arm_conditions",  # test_evalue_part_g.py, the controlled direct effect
    "incremental_reading",  # test_simulated_confounding_off_axis.py
    "learned_rule_reading",  # test_simulated_confounding_learned_rule.py
}


def test_every_sensitivity_note_fires() -> None:
    fired = {code for name in FIXTURES for code in built(name.split(":")[1]).summary().note_codes}
    assert fired | LITERAL_CODES >= T4_CODES, T4_CODES - fired - LITERAL_CODES
    assert set(vocabulary.NOTE_CODES) >= T4_CODES


def test_every_header_label_is_shared() -> None:
    for name in FIXTURES:
        summary = built(name.split(":")[1]).summary()
        assert {label for label, _ in summary.header} <= vocabulary.HEADER_LABELS, name


def test_a_short_summary_leaves_the_symbols_to_the_full_one() -> None:
    for name in FIXTURES:
        short = str(built(name.split(":")[1]).summary())
        for symbol in vocabulary.SHORT_DETAIL_SYMBOLS:
            assert symbol not in short, (name, symbol)
    full = str(built("bounds").summary(detail="full"))
    assert "sigma^2" in full and "nu^2" in full and "Riesz" in full


# ----------------------------------------------------------------- the reading of each object


def test_the_bound_states_its_verdict_and_its_limits() -> None:
    bounds = built("bounds")
    summary = bounds.summary()
    rv = f"{bounds.robustness_value:.3f}"
    assert summary.reading == TEXTS["verdict_rv_reached"].format(rv=rv, null="0", alignment="")
    assert summary.note("sensitivity_assumption").text == " ".join(
        (TEXTS["assumption_bounds"], TEXTS["assumption_bounds_limits"].format(level="95%"))
    )
    assert ("Conclusion", "range excludes 0") in summary.header


def test_a_stated_alignment_enters_the_verdict() -> None:
    bounds = built("bounds_aligned")
    reading = bounds.summary().reading
    assert reading.endswith("could not.") and " at alignment 0.500;" in reading
    assert "alignment_stated" not in built("bounds").summary().note_codes


def test_an_unreachable_value_says_no_strength_reaches_the_null() -> None:
    value = built("robustness_unreachable")
    assert value["rv"] is None
    summary = value.summary()
    assert summary.reading == TEXTS["verdict_rv_unreached"].format(
        null="0", alignment=" at alignment 0.000"
    )
    assert summary.table("other").rows[0][1].text == "none below 1"


def test_a_collaborative_point_bound_names_its_assumption_plainly() -> None:
    bounds = built("bounds_consistent")
    summary = bounds.summary()
    assert summary.note("consistent_mechanism").text == TEXTS["consistent_mechanism"]
    assert summary.note("limits_withheld").text == TEXTS["limits_withheld_consistent"].format(
        alternative=bounds.limit_refusal.alternative
    )
    assumptions = bounds.summary(detail="full").section("Assumptions").lines
    assert any(line.startswith("assume_consistent_mechanism=True") for line in assumptions)


def test_a_zero_bias_scale_withholds_the_limits_for_that_reason() -> None:
    bounds = dataclasses.replace(built("bounds"), max_bias=0.0)
    summary = bounds.summary()
    assert summary.note("limits_withheld").text == TEXTS["limits_withheld_zero"]
    assert "rv_limit" not in summary.note_codes
    assert len(summary.table("Robustness value").rows) == 1


def test_a_diagnostic_status_marks_the_limits() -> None:
    bounds = dataclasses.replace(built("bounds"), inference="working_mechanism_plugin")
    summary = bounds.summary()
    item = summary.note("no_inference_sensitivity")
    assert item.kind == "warning"
    assert summary.table("other").columns[-1] == vocabulary.SPREAD_LABELS["plugin_interval"]
    assert summary.table("Robustness value").rows[1][0].text == "plug-in limit (diagnostic)"
    assert summary.note("rv_limit").text == TEXTS["rv_limit_plugin"].format(null="0")
    # A diagnostic spread makes no coverage claim, so the limits sentence goes.
    limits = TEXTS["assumption_bounds_limits"].format(level="95%")
    assert limits not in summary.note("sensitivity_assumption").text
    assert limits in built("bounds").summary().note("sensitivity_assumption").text
    plain = built("bounds").summary().note("rv_limit").text
    assert plain == TEXTS["rv_limit"].format(limit="95% limit", level="95%", null="0")
    assert "no_inference_sensitivity" not in built("bounds").summary().note_codes


def test_a_clipped_benchmark_drops_the_pointer() -> None:
    record = built("benchmark")
    assert TEXTS["benchmark_pointer"] in record.summary().note("sensitivity_assumption").text
    clipped = dataclasses.replace(record, nu2_long=2.5 * record.nu2_short, cf_d=1.0)
    assert clipped.gain_d == pytest.approx(1.5)
    summary = clipped.summary()
    assert TEXTS["benchmark_pointer"] not in summary.note("sensitivity_assumption").text
    assert summary.note("strength_clipped").text == TEXTS["strength_clipped_d"].format(gain="1.5")


def test_a_truncated_ratio_bound_says_so() -> None:
    value = dataclasses.replace(built("evalue_converted"), truncated_bound=-0.125)
    item = value.summary().note("bound_truncated")
    assert item.text == TEXTS["bound_truncated"].format(value="-0.125")
    assert "bound_truncated" not in built("evalue_converted").summary().note_codes
    # Below one the table shows the upper limit, which the truncation leaves alone.
    below = dataclasses.replace(value, risk_ratio=1.0 / value.risk_ratio)
    item = below.summary().note("bound_truncated")
    assert item.text == TEXTS["bound_truncated_upper"].format(value="-0.125")


def test_an_estimate_at_the_null_reads_zero() -> None:
    bounds = dataclasses.replace(built("bounds"), robustness_value=0.0)
    assert bounds.summary().reading == TEXTS["verdict_rv_at_null"].format(null="0")


def test_the_evalue_conversion_prints_plain_and_formal() -> None:
    value = built("evalue_converted")
    assert [kind for kind, _ in value.note_parts] == ["conversion_smd"]
    summary = value.summary()
    assert summary.note("approximate_scale").text == TEXTS["evalue_conversion_smd"]
    assert value.note in value.summary(detail="full").section("Assumptions").lines
    exact = built("evalue_ratio")
    assert exact.note_parts == () and "approximate_scale" not in exact.summary().note_codes


def test_the_marginal_breakdown_brackets_the_grid_value() -> None:
    bounds = built("marginal")
    moved = dataclasses.replace(bounds, null=float(bounds.rows[1].lower) + 0.0)
    rows = {
        row[0].text: row
        for row in moved.summary().table(f"Where the range reaches {moved.null:g}").rows
    }
    first = moved.breakdown
    assert first is not None
    grid = sorted(row.gamma for row in moved.rows)
    earlier = [value for value in grid if value < first]
    assert rows["range"][1].value == first
    assert rows["range"][2].value == (max(earlier) if earlier else None)


def test_the_marginal_coverage_sentence_needs_a_licence() -> None:
    bounds = built("marginal")
    assert bounds.limit_descriptive
    text = bounds.summary().note("sensitivity_assumption").text
    assert TEXTS["marginal_interval"].format(level="95%") not in text
    reading = Resolution(rung="L1", licensed=True, sentence="A covering study.")
    licensed = dataclasses.replace(bounds, limit_reading=reading)
    assert not licensed.limit_descriptive
    text = licensed.summary().note("sensitivity_assumption").text
    assert text.endswith(TEXTS["marginal_interval"].format(level="95%"))


def test_the_tilt_states_the_reused_standard_error() -> None:
    tilt = built("tilt")
    text = tilt.summary().note("sensitivity_assumption").text
    direction = TEXTS["tilt_direction"] + " " + TEXTS["tilt_interval"]
    assert text == TEXTS["assumption_tilt"].format(arm_clause="", direction=direction)
    assert "reuses the main analysis's standard error" in text
    ratio = built("tilt_ratio").summary()
    assert ratio.note("tilt_diagnostic").text == TEXTS["tilt_diagnostic"].format(
        names="rr", scale=", on the log scale"
    )
    # The difference keeps its interval off a gamma of zero, so the sentence stays.
    assert TEXTS["tilt_interval"] in ratio.note("sensitivity_assumption").text
    # Mutation: no estimand of a withheld fit prints an interval, so the sentence goes.
    withheld = built("tilt_withheld").summary().note("sensitivity_assumption").text
    assert TEXTS["tilt_interval"] not in withheld
    assert withheld.endswith(TEXTS["tilt_direction"])


def test_an_arm_factor_turns_the_tilt_direction() -> None:
    from tests.unit.summary_fixtures.sensitivity import _missing

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tilt = _missing().sensitivity.missingness(gamma=(0.0, 1.0), arm_gamma={0: 0.0, 1: -1.0})
    text = tilt.summary().note("sensitivity_assumption").text
    assert TEXTS["tilt_direction_arms"] in text
    assert TEXTS["tilt_direction"] not in text
    # A negative factor moves that arm's missing outcomes down at a positive gamma.
    frame = tilt.to_frame()
    ey1 = frame[(frame["estimand"] == "ey1")].set_index("gamma")["psi"]
    assert ey1[1.0] < ey1[0.0]


def test_each_surface_condition_prints_when_it_holds() -> None:
    surface = built("surface")
    plain = surface.summary().note_codes
    for code in ("observation_reads", "bounded_rows", "redrawn_folds", "population_moves"):
        assert code not in plain
    edited = dataclasses.replace(
        surface, observation_reads=("W1",), bounded_share=0.25, refit_folds="redrawn"
    )
    summary = edited.summary()
    assert summary.note("observation_reads").text == TEXTS["observation_reads"].format(reads="W1")
    assert summary.note("bounded_rows").text == TEXTS["bounded_rows"].format(share="0.250")
    assert summary.note("redrawn_folds").text == TEXTS["redrawn_folds"]


@pytest.mark.parametrize(
    ("changes", "law"),
    [
        ({}, "law_binary"),
        ({"treatment_family": "continuous"}, "law_continuous"),
        ({"treatment_family": "vector"}, "law_vector"),
        ({"treatment_family": "longitudinal"}, "law_longitudinal"),
        ({"bounded_outcome_law": "logit"}, "law_outcome_logit"),
        ({"bounded_outcome_law": "normal_score"}, "law_outcome_normal_score"),
        ({"outcome_family": "binomial"}, "law_outcome_binomial"),
    ],
)
def test_each_surface_law_prints_in_plain_words(changes: dict[str, Any], law: str) -> None:
    surface = dataclasses.replace(built("surface"), **changes)
    assert TEXTS[law] in surface.summary().note("sensitivity_assumption").text


def test_a_failed_cell_prints_its_error() -> None:
    surface = built("surface")
    failure = ReplicationFailure(replicate=2, seed=0, error_type="RuntimeError", message="boom")
    cells = list(surface.cells)
    cells[2] = dataclasses.replace(cells[2], estimate=None, displacement=None, failure=failure)
    edited = dataclasses.replace(surface, cells=tuple(cells))
    summary = edited.summary()
    assert summary.note("cells_failed").text == TEXTS["cells_failed"].format(failed=1, total=4)
    row = summary.table("Movement").rows[2]
    assert [cell.text for cell in row[-2:]] == ["failed", "RuntimeError"]
    read.surface(edited, str(summary))


def test_a_separated_calibration_prints_its_note() -> None:
    surface = built("surface")
    first, *rest = surface.calibrations
    separated = dataclasses.replace(first, strength=float("nan"), method=SEPARATION_METHOD)
    summary = dataclasses.replace(surface, calibrations=(separated, *rest)).summary()
    assert summary.note("calibration_nan").text.startswith("Calibration: 1 of 2 strengths are nan")


# ----------------------------------------------------------------- mutation controls


def _swapping(monkeypatch: pytest.MonkeyPatch, title: str | None, columns: bool) -> None:
    """Patch the renderer to swap two rows, or two value columns, of one table."""
    original = _render._table_lines

    def swapped(table: Any, width: int) -> Any:
        if table.title == title and len(table.rows) >= 1:
            if columns:
                rows = tuple((row[0], row[2], row[1], *row[3:]) for row in table.rows)
            else:
                first, second, *others = table.rows
                rows = ((first[0], *second[1:]), (second[0], *first[1:]), *others)
            table = dataclasses.replace(table, rows=rows)
        return original(table, width)

    monkeypatch.setattr(_render, "_table_lines", swapped)


@pytest.mark.parametrize(
    ("name", "title", "columns", "reader"),
    [
        ("bounds", None, True, read.bounds),
        ("bounds", "Robustness value", False, read.bounds),
        ("benchmark", "With and without the benchmark covariates", True, read.benchmark),
        ("elements", None, False, read.elements),
        ("evalue_ratio", None, False, read.evalue),
        ("marginal", None, False, read.marginal),
        ("surface", "Movement", False, read.surface),
    ],
)
def test_a_renderer_that_swaps_values_fails_the_reader(
    monkeypatch: pytest.MonkeyPatch, name: str, title: str | None, columns: bool, reader: Any
) -> None:
    obj = built(name)
    reader(obj, str(obj.summary()))
    _swapping(monkeypatch, title, columns)
    with pytest.raises(AssertionError):
        reader(obj, str(obj.summary()))


def test_a_reader_fails_when_a_table_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    obj = built("bounds")
    original = Summary.build

    def dropped(*args: Any, **kwargs: Any) -> Summary:
        kwargs["tables"] = list(kwargs["tables"])[:1]
        return original(*args, **kwargs)

    monkeypatch.setattr(Summary, "build", staticmethod(dropped))
    with pytest.raises(AssertionError, match="no table titled"):
        read.bounds(obj, str(obj.summary()))


# ----------------------------------------------------------------- the new return objects


def test_the_robustness_value_is_a_mapping_that_pickles() -> None:
    value = built("robustness")
    assert isinstance(value, dict) and value["rv"] == pytest.approx(value.get("rv"))
    copy = pickle.loads(pickle.dumps(value))
    assert isinstance(copy, RobustnessValue) and copy == value
    assert (copy.estimand, copy.psi, copy.rho, copy.level) == (
        value.estimand,
        value.psi,
        value.rho,
        value.level,
    )
    assert str(copy.summary()) == str(value.summary())


def test_the_withheld_robustness_value_keeps_its_refusal() -> None:
    from tests.unit.summary_fixtures.sensitivity import _forest

    bounds = built("bounds_plugin")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = _forest().sensitivity.robustness_value(nu2_estimator="plugin")
    assert value.refusal == bounds.limit_refusal
    assert pickle.loads(pickle.dumps(value)).refusal == value.refusal
    withheld = value.summary().note("limits_withheld").text
    assert withheld == bounds.summary().note("limits_withheld").text


@pytest.mark.parametrize("name", ["contour", "tilt"])
def test_a_rows_object_gives_its_frame_and_pickles(name: str) -> None:
    obj = built(name)
    frame = obj.to_frame()
    assert list(frame.columns) == list(obj.columns)
    assert len(obj) == len(frame)
    first = obj.columns[0]
    assert list(obj[first]) == list(frame[first]) == list(obj.column(first))
    copy = pickle.loads(pickle.dumps(obj))
    assert copy == obj and type(copy) is type(obj)
    assert isinstance(obj, (SensitivityContour, MissingnessTilt))


def test_the_tilt_stays_a_tilt_through_the_cache_and_the_report() -> None:
    from tests.unit.summary_fixtures.sensitivity import _missing

    result = _missing()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = result.sensitivity.missingness()
        second = result.sensitivity.missingness()
        report = result.sensitivity.run_all(include_retargets=True).report("missingness")
    assert isinstance(first, MissingnessTilt) and isinstance(second, MissingnessTilt)
    assert first == second
    assert isinstance(report, MissingnessTilt)


def test_two_benchmarks_with_one_repr_key_apart() -> None:
    record = built("benchmark")
    other = dataclasses.replace(record, psi_short=record.psi_short + 1e-9)
    assert repr(other) == repr(record)
    assert _cache_key("operation", (other,), {}) != _cache_key("operation", (record,), {})
    assert _cache_key("operation", (dataclasses.replace(record),), {}) == _cache_key(
        "operation", (record,), {}
    )


def test_the_components_have_a_short_repr() -> None:
    surface = built("surface")
    for item in (surface.cells[0], surface.calibrations[0], built("marginal").rows[0]):
        text = repr(item)
        assert "\n" not in text and len(text) <= 100
        assert text.startswith(type(item).__name__ + "(")


def test_the_tipping_gamma_stays_a_number() -> None:
    from tests.unit.summary_fixtures.sensitivity import _missing

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = _missing().sensitivity.tipping_gamma()
    assert value is None or isinstance(value, float)
    assert value is None or np.isfinite(value)


def test_a_full_population_surface_names_no_stratum() -> None:
    reading = built("surface").summary().note("association_reading").text
    assert "within the selected baseline stratum" not in reading
    assert TEXTS["association_defined"].format(intermediate="", stratum="") in reading
