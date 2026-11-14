"""The balance-over-time plot (X103): what each marker draws and when the plot refuses.

Each structural check reads the artists by their gid and compares them with the report's own
``average()`` and ``to_frame()``, so a plot that drew the wrong statistic, the wrong sign, or
the wrong strata fails here.
"""

from __future__ import annotations

from typing import Any

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cleverly import plotting
from cleverly.exceptions import CapabilityError
from cleverly.sensitivity import BalanceOverTime
from tests.unit._plotting_checks import assert_style, assert_text_inside
from tests.unit.summary_fixtures import balance as point_fixtures
from tests.unit.summary_fixtures import balance_time as fixtures
from tests.unit.test_plotting_core import assert_no_note_text, assert_short_title

REPORTS = {
    "longitudinal": fixtures.longitudinal,
    "survival": fixtures.survival,
    "continuous": fixtures.continuous,
    "collaborative": fixtures.collaborative,
}


@pytest.fixture(autouse=True)
def _close() -> Any:
    yield
    plt.close("all")


def _offsets(fig: Any, gid: str) -> np.ndarray:
    found = [
        np.asarray(artist.get_offsets())
        for axis in fig.axes
        for artist in axis.collections
        if artist.get_gid() == gid
    ]
    return np.concatenate(found) if found else np.empty((0, 2))


def _cases() -> list[tuple[str, str, str]]:
    cases = []
    for family, build in REPORTS.items():
        for diagnostic in build().diagnostics:
            cases.append((family, diagnostic, "summary"))
            if family != "continuous":
                cases.append((family, diagnostic, "panels"))
    return cases


@pytest.mark.parametrize(("family", "diagnostic", "view"), _cases())
def test_every_family_view_and_check_renders_in_the_house_style(
    family: str, diagnostic: str, view: str
) -> None:
    fig = REPORTS[family]().plot(diagnostic=diagnostic, view=view)
    assert_style(fig)
    assert_text_inside(fig)
    assert_short_title(fig)
    assert_no_note_text(fig)
    assert plotting.notes(fig), "the plot records its notes"


@pytest.mark.parametrize("family", list(REPORTS))
def test_the_summary_markers_are_the_absolute_history_averages(family: str) -> None:
    balance = REPORTS[family]()
    for diagnostic in balance.diagnostics:
        fig = balance.plot(diagnostic=diagnostic, max_covariates=100)
        frame = balance.average(over="history", diagnostic=diagnostic)
        before = np.sort(_offsets(fig, "cleverly-balance-before")[:, 0])
        after = np.sort(_offsets(fig, "cleverly-balance-after")[:, 0])
        expected_before = np.sort(np.abs(np.asarray(frame["smd_unweighted"], dtype=float)))
        expected_after = np.sort(np.abs(np.asarray(frame["smd_weighted"], dtype=float)))
        np.testing.assert_allclose(before, expected_before, rtol=1e-12, equal_nan=True)
        np.testing.assert_allclose(after, expected_after, rtol=1e-12, equal_nan=True)
        plt.close(fig)


def test_the_summary_whisker_spans_the_strata_after_weighting() -> None:
    balance = fixtures.longitudinal()
    fig = balance.plot(diagnostic="confounding")
    spans = [
        tuple(sorted(segment[:, 0]))
        for axis in fig.axes
        for artist in axis.collections
        if artist.get_gid() == "cleverly-balance-strata"
        for segment in artist.get_segments()
    ]
    expected = []
    for (exposure, covariate), rows in _by_cell(balance, "confounding").items():
        values = [abs(r.smd_weighted) for r in rows if np.isfinite(r.smd_weighted)]
        if len(values) > 1:
            expected.append((min(values), max(values)))
    assert len(spans) == len(expected)
    np.testing.assert_allclose(np.asarray(sorted(spans)), np.asarray(sorted(expected)))


def _by_cell(balance: BalanceOverTime, diagnostic: str) -> dict[tuple[str, str], list[Any]]:
    cells: dict[tuple[str, str], list[Any]] = {}
    for row in balance.rows_of(diagnostic):
        cells.setdefault((row.exposure, row.covariate), []).append(row)
    return cells


@pytest.mark.parametrize("diagnostic", ["confounding", "feedback", "censoring"])
def test_the_panels_draw_each_stratum_signed_and_each_average(diagnostic: str) -> None:
    balance = fixtures.longitudinal()
    fig = balance.plot(diagnostic=diagnostic, view="panels")
    rows = balance.rows_of(diagnostic)
    dots = np.sort(_offsets(fig, "cleverly-balance-strata-dots")[:, 0])
    np.testing.assert_allclose(dots, np.sort([r.smd_weighted for r in rows]), equal_nan=True)
    averages = np.sort(_offsets(fig, "cleverly-balance-average")[:, 0])
    frame = balance.average(over="history", diagnostic=diagnostic)
    np.testing.assert_allclose(
        averages, np.sort(np.asarray(frame["smd_weighted"], dtype=float)), equal_nan=True
    )


def test_nodes_keeps_only_the_named_time_points() -> None:
    balance = fixtures.longitudinal()
    fig = balance.plot(diagnostic="confounding", view="panels", nodes=[2])
    rows = [r for r in balance.rows_of("confounding") if r.exposure_node == 2]
    dots = _offsets(fig, "cleverly-balance-strata-dots")
    assert len(dots) == len(rows)
    with pytest.raises(ValueError, match="nodes= names"):
        balance.plot(diagnostic="confounding", view="panels", nodes=[9])


def test_a_result_and_its_report_draw_the_same_figure() -> None:
    from tests.unit._capability_sweep_support import fit_ltmle

    from_result = plotting.balance(fit_ltmle(), diagnostic="confounding")
    from_report = plotting.balance(fixtures.longitudinal(), diagnostic="confounding")
    for gid in ("cleverly-balance-before", "cleverly-balance-after"):
        np.testing.assert_allclose(_offsets(from_result, gid), _offsets(from_report, gid))


def test_a_continuous_fit_has_no_panels_view() -> None:
    with pytest.raises(ValueError, match="view='summary'"):
        fixtures.continuous().plot(view="panels")


def test_an_omitted_check_is_refused_with_its_reason() -> None:
    with pytest.raises(CapabilityError, match="no feedback balance"):
        fixtures.survival().plot(diagnostic="feedback")


def test_an_unknown_view_or_check_is_refused() -> None:
    with pytest.raises(ValueError, match="view must be"):
        fixtures.longitudinal().plot(view="grid")
    with pytest.raises(ValueError, match="diagnostic must be"):
        fixtures.longitudinal().plot(diagnostic="overlap")
    with pytest.raises(ValueError, match="has no 'joint' rows"):
        fixtures.longitudinal().plot(diagnostic="joint")


@pytest.mark.parametrize(
    "keyword", [{"diagnostic": "confounding"}, {"view": "panels"}, {"nodes": [1]}]
)
def test_a_point_treatment_balance_refuses_the_over_time_keywords(
    keyword: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="balance of a fit over time"):
        plotting.balance(point_fixtures.two_arms(), **keyword)


def test_the_feedback_panels_put_covariate_times_down_the_side() -> None:
    fig = fixtures.longitudinal().plot(diagnostic="feedback", view="panels")
    titles = [axis.get_title(loc="left") for axis in fig.axes[1:] if axis.get_title(loc="left")]
    assert titles and all("covariates at time" in title for title in titles), titles
