"""The influence report and the weight report print like every other check (X89 T3).

The contract test (``test_summary_contract.py``) checks width, notes, the repr and every
printed number of each fixture in ``summary_fixtures/spread.py``. This module checks what is
particular to these two reports: that each reading and note fires where it should and only
there, that the numbers in the readings and the full detail equal the report's, that the
result's own call records the fit's warning, and that the number readers fail a renderer
that moves numbers between rows or columns.
"""

from __future__ import annotations

import dataclasses
import functools
import warnings
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly._summary import SummaryTable, _render, vocabulary
from cleverly.data.weighting import CONCENTRATED_DESIGN_EFFECT
from cleverly.datasets import make_linear_ate
from cleverly.estimators import TMLE
from cleverly.inference.sparse import SPARSE_INFLUENCE_THRESHOLD

from .summary_fixtures import spread
from .summary_fixtures._check_fidelity import read_influence, read_weights

REPORTS = dict(spread.FIXTURES)
INFLUENCE = sorted(name for name in REPORTS if name.startswith("influence_"))
WEIGHTS = sorted(name for name in REPORTS if name.startswith("weights_"))


@functools.cache
def _report(name: str) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return REPORTS[name]()


def _reader(name: str) -> Any:
    return read_influence if name in INFLUENCE else read_weights


def test_every_note_of_the_two_reports_fires() -> None:
    fired = {code for name in REPORTS for code in _report(name).summary().note_codes}
    assert set(vocabulary.T3_SPREAD_NOTE_CODES) <= fired, (
        set(vocabulary.T3_SPREAD_NOTE_CODES) - fired
    )


def test_every_reading_fires() -> None:
    readings = {str(_report(name).summary().reading) for name in REPORTS}
    threshold = f"{SPARSE_INFLUENCE_THRESHOLD:g}"
    expected = {
        vocabulary.INFLUENCE_READINGS["none"].format(threshold=threshold),
        vocabulary.INFLUENCE_READINGS["empty"],
        vocabulary.WEIGHT_READINGS["constant"],
        vocabulary.WEIGHT_READINGS["dominant"].format(limit=CONCENTRATED_DESIGN_EFFECT),
    }
    assert expected <= readings, expected - readings
    stems = ("ey_regimen[always]: the variance estimate", "each variance estimate")
    for stem in stems:
        assert any(stem in reading for reading in readings), (stem, readings)


# ----------------------------------------------------------------- influence


def test_a_flagged_estimate_is_named_with_its_relative_sd() -> None:
    report = _report("influence_flagged")
    assert report.sparse == ("ey_regimen[always]",)
    row = report["ey_regimen[always]"]
    reading = report.summary().reading
    assert reading == vocabulary.INFLUENCE_READINGS["one"].format(
        names="ey_regimen[always]",
        value=f"{row.variance_rel_sd:.3f}",
        threshold=f"{SPARSE_INFLUENCE_THRESHOLD:g}",
    )
    # Witness: the clean estimate is not named.
    assert "ey_regimen[never]" not in str(reading)


def test_several_flagged_estimates_give_the_largest_relative_sd() -> None:
    report = _report("influence_fit")
    assert len(report.sparse) >= 2
    largest = max(report[name].variance_rel_sd for name in report.sparse)
    assert f"relative SD up to {largest:.3f}" in str(report.summary().reading)


def test_the_warning_pointer_prints_only_after_the_fit_warned() -> None:
    assert "sparse_warning_pointer" in _report("influence_flagged").summary().note_codes
    assert "sparse_warning_pointer" not in _report("influence_clean").summary().note_codes
    unwarned = dataclasses.replace(_report("influence_flagged"), warned=False)
    assert "sparse_warning_pointer" not in unwarned.summary().note_codes


def test_the_results_own_call_records_whether_the_fit_warned() -> None:
    longitudinal = spread._longitudinal()
    categories = {record.category for record in longitudinal.fit_warnings}
    assert "SparseInfluenceWarning" in categories
    assert longitudinal.diagnostics.influence().warned
    frame = make_linear_ate(n=300, seed=1)[0]
    point = (
        TMLE(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            cross_fit=False,
        )
        .fit(frame, outcome="Y", treatment="A")
        .single()
    )
    assert "SparseInfluenceWarning" not in {r.category for r in point.fit_warnings}
    assert not point.diagnostics.influence().warned


def test_a_clustered_report_counts_clusters() -> None:
    summary = _report("influence_clustered").summary()
    assert dict(summary.header)[vocabulary.CLUSTERS] == "60"
    assert "cluster_unit" in summary.note_codes
    assert summary.tables[0].columns[1] == "effective clusters"
    assert "per cluster" in summary.note("influence_terms").text
    unclustered = _report("influence_clean").summary()
    assert "cluster_unit" not in unclustered.note_codes
    assert unclustered.tables[0].columns[1] == "effective rows"


def test_the_terms_note_reads_the_reports_level() -> None:
    assert (
        "for the 90% interval"
        in _report("influence_bootstrap").summary().note("influence_terms").text
    )
    assert (
        "for the 95% interval" in _report("influence_clean").summary().note("influence_terms").text
    )


def test_the_full_detail_gives_each_kurtosis_and_the_bootstrap_shape() -> None:
    report = _report("influence_bootstrap")
    full = report.summary(detail="full")
    row = report.rows[0]
    assert full.section(vocabulary.INFLUENCE_KURTOSIS).lines == (f"ate: {row.kurtosis:.2f}",)
    assert full.section(vocabulary.INFLUENCE_BOOTSTRAP).lines == (
        f"ate: bias correction {row.bca_z0:.3f}, acceleration {row.bca_acceleration:.4f}",
    )
    # Witness: a report without a bootstrap has no bootstrap section.
    titles = [section.title for section in _report("influence_clean").summary("full").sections]
    assert vocabulary.INFLUENCE_BOOTSTRAP not in titles
    assert vocabulary.INFLUENCE_KURTOSIS in titles


def test_the_influence_repr_names_one_estimate_or_counts_several() -> None:
    single = _report("influence_bootstrap")
    assert repr(single) == f"InfluenceReport(ate, effective rows={single.rows[0].kish:.1f})"
    several = _report("influence_flagged")
    fewest = min(row.kish for row in several.rows)
    assert repr(several) == (
        f"InfluenceReport(2 estimates, fewest effective rows={fewest:.1f}, flagged=1)"
    )
    assert repr(_report("influence_empty")) == "InfluenceReport(no estimates)"


# ----------------------------------------------------------------- weights


def test_fixed_and_estimated_weights_get_one_note_each() -> None:
    fixed = _report("weights_fixed").summary().note_codes
    estimated = _report("weights_estimated").summary().note_codes
    assert "weights_fixed" in fixed and "weights_estimated" not in fixed
    assert "weights_estimated" in estimated and "weights_fixed" not in estimated


def test_the_zero_weight_note_counts_the_rows() -> None:
    summary = _report("weights_fixed").summary()
    assert summary.note("zero_weights").text == vocabulary.NOTE_TEXTS["zero_weights"].format(
        count=4
    )
    assert "zero_weights" not in _report("weights_estimated").summary().note_codes


def test_the_dominance_reading_follows_the_design_effect() -> None:
    dominant = _report("weights_dominant")
    assert dominant.design_effect > CONCENTRATED_DESIGN_EFFECT
    assert dominant.summary().reading is not None
    moderate = _report("weights_fixed")
    assert moderate.design_effect <= CONCENTRATED_DESIGN_EFFECT
    assert moderate.summary().reading is None


def test_constant_weights_print_the_table_and_no_notes() -> None:
    report = _report("weights_constant")
    summary = report.summary()
    assert not report.is_weighted
    assert summary.reading == vocabulary.WEIGHT_READINGS["constant"]
    assert summary.notes == ()
    assert summary.tables[0].rows[0][1].text == "1.00"


def test_the_full_detail_states_the_estimand() -> None:
    lines = _report("weights_fixed").summary(detail="full").section(vocabulary.WEIGHT_ESTIMAND)
    assert lines.lines == vocabulary.WEIGHT_ESTIMAND_LINES
    assert "dP_w" not in _report("weights_fixed").summary()


def test_the_weight_repr_gives_the_effective_rows() -> None:
    report = _report("weights_fixed")
    assert repr(report) == f"WeightReport(w, effective rows={report.effective_n:.1f} of 500)"


# ----------------------------------------------------------------- mutation controls

_ORIGINAL = _render._table_lines


def _swap_rows(table: SummaryTable, width: int) -> Any:
    """The renderer with the numbers of the first two rows of every table swapped."""
    if len(table.rows) >= 2:
        first, second, *rest = table.rows
        rows = ((first[0], *second[1:]), (second[0], *first[1:]), *rest)
        table = dataclasses.replace(table, rows=tuple(rows))
    return _ORIGINAL(table, width)


def _differ(table: SummaryTable) -> bool:
    if len(table.rows) < 2:
        return False
    first, second = table.rows[:2]
    # Values, not texts: 1.00 and 1 print differently and read back the same.
    return [cell.value for cell in first[1:]] != [cell.value for cell in second[1:]]


#: The fixtures where a row swap moves a printed number.
SWAPPABLE = sorted(
    name for name in REPORTS if any(_differ(table) for table in _report(name).summary().tables)
)


@pytest.mark.parametrize("name", SWAPPABLE)
def test_a_renderer_that_swaps_two_rows_fails_the_reader(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _report(name)
    _reader(name)(report, str(report.summary()))
    monkeypatch.setattr(_render, "_table_lines", _swap_rows)
    with pytest.raises(AssertionError):
        _reader(name)(report, str(report.summary()))


def test_the_row_swap_reaches_both_reports() -> None:
    """Witness: at least one fixture of each report moves a number under the swap."""
    assert set(SWAPPABLE) & set(INFLUENCE), SWAPPABLE
    assert set(SWAPPABLE) & set(WEIGHTS), SWAPPABLE


@pytest.mark.parametrize("name", sorted(set(INFLUENCE) - {"influence_empty"}))
def test_a_renderer_that_swaps_two_number_columns_fails_the_reader(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _report(name)
    _reader(name)(report, str(report.summary()))

    def swapped(table: SummaryTable, width: int) -> Any:
        i, j = table.columns.index("skewness"), table.columns.index("SE uncertainty")
        rows = []
        for row in table.rows:
            cells = list(row)
            cells[i], cells[j] = cells[j], cells[i]
            rows.append(tuple(cells))
        return _ORIGINAL(dataclasses.replace(table, rows=tuple(rows)), width)

    monkeypatch.setattr(_render, "_table_lines", swapped)
    with pytest.raises(AssertionError):
        _reader(name)(report, str(report.summary()))


def test_a_renderer_that_drops_the_flag_fails_the_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    report = _report("influence_flagged")

    def unflagged(table: SummaryTable, width: int) -> Any:
        rows = tuple((*row[:-1], dataclasses.replace(row[-1], text="-")) for row in table.rows)
        return _ORIGINAL(dataclasses.replace(table, rows=rows), width)

    monkeypatch.setattr(_render, "_table_lines", unflagged)
    with pytest.raises(AssertionError):
        read_influence(report, str(report.summary()))


def test_a_weight_report_with_its_ratios_swapped_fails_the_reader() -> None:
    report = _report("weights_dominant")
    read_weights(report, str(report.summary()))
    swapped = dataclasses.replace(report, ratio=(report.ratio[1], report.ratio[0]))
    with pytest.raises(AssertionError):
        read_weights(report, str(swapped.summary()))


def test_the_weight_reader_reads_the_effective_rows() -> None:
    report = _report("weights_fixed")
    shifted = dataclasses.replace(report, effective_n=report.effective_n + 5.0)
    with pytest.raises(AssertionError):
        read_weights(report, str(shifted.summary()))
    assert np.isfinite(report.effective_n)
