"""``truncation_curve()`` returns a curve that prints in words and reads like its frame (X89 T3).

The contract test checks width, notes, house style and every printed number of each fixture
in ``summary_fixtures/curve.py``. This module checks the object: frame access, the frame the
combined report keeps, the reading in standard errors, the notes, and that the reader fails a
renderer that moves a row.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import pickle
from typing import Any

import pandas as pd
import pytest

from cleverly import TruncationCurve
from cleverly._summary import SummaryTable, _render, vocabulary

from .summary_fixtures import curve as fixtures
from .summary_fixtures.battery import _fit

CASES = fixtures.FIXTURES


@functools.cache
def _curve(name: str) -> Any:
    return CASES[name].build()


def test_the_curve_reads_like_its_frame() -> None:
    curve = _curve("curve_point")
    frame = curve.to_frame()
    assert isinstance(curve, TruncationCurve)
    assert isinstance(frame, pd.DataFrame)
    assert curve.frame is frame
    assert list(curve.columns) == list(frame.columns)
    assert len(curve) == len(frame) and curve.shape == frame.shape
    assert list(curve) == list(frame)
    pd.testing.assert_series_equal(curve["psi"], frame["psi"])
    assert (
        curve.loc[curve["is_fitted_bound"], "psi"].tolist()
        == frame.loc[frame["is_fitted_bound"], "psi"].tolist()
    )


def test_two_curves_compare_by_identity() -> None:
    curve = _curve("curve_point")
    assert curve == curve
    assert curve != TruncationCurve(curve.to_frame())


def test_a_copy_and_a_pickle_stay_curves() -> None:
    curve = _curve("curve_point")
    for restored in (copy.deepcopy(curve), pickle.loads(pickle.dumps(curve))):
        assert isinstance(restored, TruncationCurve)
        pd.testing.assert_frame_equal(restored.to_frame(), curve.to_frame())
    with pytest.raises(AttributeError):
        curve.__missing_dunder__  # noqa: B018


def test_the_reading_scales_the_largest_change_by_the_fitted_standard_error() -> None:
    curve = _curve("curve_capped")
    frame = curve.to_frame()
    index = frame["delta_from_fitted"].abs().idxmax()
    change = abs(frame.loc[index, "delta_from_fitted"])
    assert change > 0
    fitted = frame.loc[frame["is_fitted_bound"], "std_err"].iloc[0]
    assert curve.summary().reading == (
        f"Across these caps the estimate moves by at most {change:.3g} "
        f"({change / fitted:.2g} std. errors)."
    )


def test_several_estimands_name_the_one_that_moves_most() -> None:
    curve = _curve("curve_estimands")
    frame = curve.to_frame()
    name = frame.loc[frame["delta_from_fitted"].abs().idxmax(), "estimand"]
    assert curve.summary().reading.startswith(f"Across these caps {name} moves the most: by ")
    assert curve.summary().tables[0].columns[0] == "estimand"


def test_a_diagnostic_spread_prints_the_change_alone() -> None:
    curve = _curve("curve_collaborative")
    assert "plugin_std_err" in curve.to_frame().columns
    summary = curve.summary()
    assert "std. errors" not in summary.reading
    assert "no_spread" in summary.note_codes


def test_a_longitudinal_curve_counts_score_cells() -> None:
    summary = _curve("curve_longitudinal").summary()
    assert "cells_capped" in summary.note_codes
    assert "units_capped" not in summary.note_codes
    assert "no_spread" in summary.note_codes
    assert "units_capped" in _curve("curve_point").summary().note_codes


def test_the_fitted_limits_are_marked_and_an_explicit_grid_may_omit_them() -> None:
    marked = [row[0].text for row in _curve("curve_point").summary().tables[0].rows]
    assert sum(text.endswith("*") for text in marked) == 1
    explicit = _fit().diagnostics.truncation_curve([0.02, 0.05])
    summary = explicit.summary()
    assert "fitted_limits" not in summary.note_codes
    assert not any(row[0].text.endswith("*") for row in summary.tables[0].rows)


def test_the_full_detail_lists_every_pair() -> None:
    curve = _curve("curve_point")
    lines = curve.summary("full").section(vocabulary.CURVE_LIMITS_TRIED).lines
    assert len(lines) == len(curve)
    assert sum(line.endswith("(the fitted limits)") for line in lines) == 1


def test_the_combined_report_keeps_the_frame_and_returns_the_curve() -> None:
    report = _fit().diagnostics.run_all(include_retargets=True)
    item = report["truncation_curve"]
    assert isinstance(item.report, TruncationCurve)
    assert isinstance(report.report("truncation_curve"), TruncationCurve)
    assert isinstance(report.reports()["truncation_curve"], TruncationCurve)
    [row] = [row for row in report.summary().tables[0].rows if row[0].value == "truncation_curve"]
    assert row[2].text.startswith("Across these caps the estimate moves by at most")


def test_the_repr_is_one_line() -> None:
    assert repr(_curve("curve_point")) == "TruncationCurve(8 rows, ate)"
    assert repr(_curve("curve_estimands")).endswith(", ...)")
    assert _curve("curve_point")._repr_html_().startswith("<")


# ----------------------------------------------------------------- mutation control

_ORIGINAL = _render._table_lines


def _swap_rows(table: SummaryTable, width: int) -> Any:
    """The renderer with the first and last rows' cells after the first swapped."""
    first, *middle, last = table.rows
    rows = ((first[0], *last[1:]), *middle, (last[0], *first[1:]))
    return _ORIGINAL(dataclasses.replace(table, rows=rows), width)


@pytest.mark.parametrize("name", ["curve_capped", "curve_longitudinal"])
def test_a_renderer_that_moves_a_row_fails_the_reader(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    curve = _curve(name)
    fixtures.read_curve(curve, str(curve.summary()))
    monkeypatch.setattr(_render, "_table_lines", _swap_rows)
    with pytest.raises(AssertionError):
        fixtures.read_curve(curve, str(curve.summary()))
