"""The confounding plots: the contour of V13 and the simulated-confounding surface of V14.

Structural tests read the drawn artists and compare them with the object's own numbers. Each
honesty rule that applies has a mutation control: a plot, a label or an object altered so that
the assertion would fail if the plot were wrong.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import matplotlib
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cleverly import ATE, CausalStudy, ControlledDirectEffect, PointTreatment
from cleverly._summary.vocabulary import NOTE_TEXTS as TEXTS
from cleverly.datasets import make_cde, make_linear_ate
from cleverly.plotting import PALETTE, confounding
from cleverly.plotting import _confounding as module
from cleverly.plotting._core import note_text, subtitle_text
from cleverly.sensitivity import (
    ConfounderStrengthGrid,
    SensitivityContour,
    SimulatedConfoundingResult,
    benchmark,
    contour_data,
    robustness_value,
    simulated_confounding,
)
from cleverly.validation.simulation import ReplicationFailure
from tests.unit.test_plotting_core import assert_no_note_text

pytestmark = pytest.mark.xdist_group("plotting_confounding")


@pytest.fixture(autouse=True)
def _close_figures() -> Any:
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def fit() -> Any:
    frame, _ = make_linear_ate(n=300, seed=1)
    design = PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4"))
    return (
        CausalStudy(frame, design=design)
        .identify(ATE())
        .estimate(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            cross_fit=False,
            random_state=0,
        )
    )


@pytest.fixture(scope="module")
def negative_fit() -> Any:
    """The same data with the outcome negated, so the estimate lies below the null."""
    frame, _ = make_linear_ate(n=300, seed=1)
    frame["Y"] = -frame["Y"]
    design = PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4"))
    return (
        CausalStudy(frame, design=design)
        .identify(ATE())
        .estimate(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            cross_fit=False,
            random_state=0,
        )
    )


@pytest.fixture(scope="module")
def grid_data(fit: Any) -> SensitivityContour:
    # reaches past the robustness value, so the null contour is on the map
    return contour_data(fit, "ate", grid_size=26, grid_bounds=(0.6, 0.6))


@pytest.fixture(scope="module")
def rv(fit: Any) -> Any:
    return robustness_value(fit, "ate")


@pytest.fixture(scope="module")
def w2(fit: Any) -> Any:
    return benchmark(fit, ["W2"])


@pytest.fixture(scope="module")
def surface_result(fit: Any) -> SimulatedConfoundingResult:
    grid = ConfounderStrengthGrid(treatment=(0.0, 0.1, 0.2, 0.3), outcome=(0.0, 0.5, 1.0))
    return simulated_confounding(
        fit, "ate", grid=grid, random_state=0, benchmark_covariates=("W2", "W3")
    )


def _by_gid(ax: Any, gid: str) -> list[Any]:
    found = [artist for artist in [*ax.collections, *ax.lines, *ax.patches, *ax.texts]
             if artist.get_gid() == gid]  # fmt: skip
    return found


def _vertices(contour_set: Any) -> np.ndarray[Any, Any]:
    segments = [seg for level in contour_set.allsegs for seg in level if len(seg)]
    return np.vstack(segments)


def _interpolate(data: Any, points: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    from scipy.interpolate import RegularGridInterpolator

    d, y, z = module._grid(data)
    return RegularGridInterpolator((y, d), z)(points[:, ::-1])


# ----------------------------------------------------------------------- V13


class TestContour:
    def test_title_subtitle_and_axes(self, grid_data: Any, rv: Any) -> None:
        fig = grid_data.plot(robustness=rv)
        ax = fig.axes[0]
        assert ax.get_title(loc="left") == "Sensitivity to hidden confounding"
        subtitle = subtitle_text(ax)
        assert subtitle is not None
        assert "Lower bound of the average effect (" in subtitle
        assert "(ate)" not in subtitle  # no library word on the figure
        assert "\u03c1 = 1" in subtitle
        assert f"point robustness value {rv['rv']:.3g}" in subtitle
        assert ax.get_xlabel() == "Treatment-side strength, cf_d"
        assert ax.get_ylabel() == "Outcome-side strength, cf_y"
        d, y, _ = module._grid(grid_data)
        assert ax.get_xlim() == (d[0], d[-1])
        assert ax.get_ylim() == (y[0], y[-1])
        assert fig.get_facecolor()[3] == 1.0

    def test_the_bold_contour_sits_at_the_null(self, grid_data: Any) -> None:
        ax = grid_data.plot().axes[0]
        (bold,) = _by_gid(ax, "cleverly-null-contour")
        values = _interpolate(grid_data, _vertices(bold))
        assert np.allclose(values, 0.0, atol=2e-2)
        assert bold.get_linewidths()[0] > 2.0

    def test_a_shifted_null_moves_the_bold_contour(self, grid_data: Any) -> None:
        # mutation control: the assertion above would fail if the contour sat at another level
        ax = grid_data.plot(null=0.5).axes[0]
        (bold,) = _by_gid(ax, "cleverly-null-contour")
        values = _interpolate(grid_data, _vertices(bold))
        assert not np.allclose(values, 0.0, atol=2e-2)
        assert np.allclose(values, 0.5, atol=2e-2)

    def test_the_robustness_value_sits_on_the_diagonal_and_on_the_contour(
        self, grid_data: Any, rv: Any
    ) -> None:
        ax = grid_data.plot(robustness=rv).axes[0]
        (marker,) = _by_gid(ax, "cleverly-robustness-value")
        x, y = marker.get_xydata()[0]
        assert x == y == pytest.approx(rv["rv"])
        value = _interpolate(grid_data, np.array([[x, y]]))[0]
        assert abs(value) < 2e-2 * np.ptp(grid_data.column("value"))
        (diagonal,) = _by_gid(ax, "cleverly-diagonal")
        assert np.allclose(diagonal.get_xdata(), diagonal.get_ydata())

    def test_the_robustness_value_must_match_the_contours_kind(
        self, fit: Any, grid_data: Any, rv: Any
    ) -> None:
        # the other end of the range, the other null, the other alignment and the other
        # estimate each refuse; the matching value draws
        upper = contour_data(fit, "ate", grid_size=6, bound="upper")
        with pytest.raises(ValueError, match="only the lower bound can reach 0"):
            upper.plot(robustness=rv)
        with pytest.raises(ValueError, match="rho"):
            contour_data(fit, "ate", grid_size=6, rho=0.5).plot(robustness=rv)
        with pytest.raises(ValueError, match="null"):
            grid_data.plot(robustness=rv, null=0.3)
        shifted = robustness_value(fit, "ate", null_hypothesis=0.5)
        with pytest.raises(ValueError, match="differs"):
            grid_data.plot(robustness=shifted, null=0.0)
        other = replace(grid_data, psi=grid_data.psi + 0.5)
        with pytest.raises(ValueError, match="different estimates"):
            other.plot(robustness=rv)
        assert _by_gid(grid_data.plot(robustness=rv).axes[0], "cleverly-robustness-value")

    def test_the_colour_bar_names_the_quantity(self, grid_data: Any) -> None:
        fig = grid_data.plot()
        bar = _by_gid_axes(fig, "cleverly-colorbar")
        assert bar.get_ylabel() == "Bias-adjusted lower bound, no sampling error"
        assert "no sampling error" in bar.get_ylabel()
        upper = module.contour(
            replace(grid_data, bound="upper"), null=5.0
        )  # the label follows the end the cells hold
        assert "upper bound" in _by_gid_axes(upper, "cleverly-colorbar").get_ylabel()

    def test_a_label_without_the_quantity_would_fail(
        self, grid_data: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # mutation control for rule 9
        monkeypatch.setattr(module, "COLORBAR_LABEL", "Bias-adjusted {bound} bound")
        label = _by_gid_axes(grid_data.plot(), "cleverly-colorbar").get_ylabel()
        assert "no sampling error" not in label

    def test_no_confidence_limit_contour_is_drawn(self, grid_data: Any, rv: Any) -> None:
        fig = grid_data.plot(robustness=rv)
        ax = fig.axes[0]
        contour_sets = {
            artist.get_gid() for artist in ax.collections if str(artist.get_gid()).startswith("cleverly-contour")
            or artist.get_gid() == "cleverly-null-contour"
        }  # fmt: skip
        assert contour_sets <= {
            "cleverly-contour-fill",
            "cleverly-contour-lines",
            "cleverly-null-contour",
        }
        assert "confidence limit is drawn" in note_text(fig)
        assert "not on this grid" in note_text(fig)
        assert len(_by_gid(ax, "cleverly-robustness-value")) == 1  # the point value only

    def test_benchmarks_are_marked_at_their_gains(self, w2: Any, grid_data: Any) -> None:
        fig = grid_data.plot(benchmarks=w2)
        ax = fig.axes[0]
        (point,) = [line.get_xydata()[0] for line in _by_gid(ax, "cleverly-benchmark")]
        assert np.allclose(point, (w2.cf_d, w2.cf_y))
        text = note_text(fig)
        # the marks are read at the map's rho, and the benchmark's own rho is named
        assert "read at this map's \u03c1 = 1" in text and f"W2 {w2.rho:.2g}" in text
        assert "heuristic" not in text  # no multiple, so no scaling rule to state

    def test_multiples_state_their_rule_and_stay_in_range(self, w2: Any, grid_data: Any) -> None:
        fig = grid_data.plot(benchmarks=w2, multiples=(1.0, 2.0, 40.0))
        ax = fig.axes[0]
        points = sorted(tuple(line.get_xydata()[0]) for line in _by_gid(ax, "cleverly-benchmark"))
        expected = sorted((k * w2.cf_d, k * w2.cf_y) for k in (1.0, 2.0))
        assert np.allclose(points, expected)
        text = note_text(fig)
        assert "scaling heuristic of this plot, not a bound" in text
        # 40 times the outcome gain is not a share, so it is refused rather than placed
        assert 40 * w2.cf_y > 1.0
        assert "Not marked" in text and "W2 \u00d740" in text
        assert "beyond the grid" not in text

    def test_a_multiple_off_the_grid_but_in_range_is_named(self, w2: Any, fit: Any) -> None:
        small = contour_data(fit, "ate", grid_size=8, grid_bounds=(0.6, 1.5 * w2.cf_y))
        text = note_text(small.plot(benchmarks=w2, multiples=(1.0, 2.0)))
        assert "Benchmarks beyond the grid: W2 \u00d72" in text

    def test_a_clipped_gain_is_never_scaled(self, w2: Any, grid_data: Any) -> None:
        clipped = replace(w2, cf_y=1.0)
        fig = grid_data.plot(benchmarks=clipped, multiples=(0.5,))
        assert not _by_gid(fig.axes[0], "cleverly-benchmark")
        assert "W2 \u00d70.5" in note_text(fig) and "clipped" in note_text(fig)
        # control: the unclipped record at the same multiple is drawn
        assert _by_gid(
            grid_data.plot(benchmarks=w2, multiples=(0.5,)).axes[0], "cleverly-benchmark"
        )

    def test_benchmark_labels_stay_inside_the_axes(
        self, w2: Any, grid_data: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        corners = [
            replace(w2, covariates=("top right",), cf_d=0.58, cf_y=0.59),
            replace(w2, covariates=("top left",), cf_d=0.01, cf_y=0.595),
            replace(w2, covariates=("right",), cf_d=0.59, cf_y=0.2),
        ]

        def outside(fig: Any) -> list[str]:
            fig.canvas.draw()
            ax = fig.axes[0]
            box = ax.get_window_extent()
            found = []
            for text in _by_gid(ax, "cleverly-benchmark-label"):
                extent = text.get_window_extent()
                if extent.x0 < box.x0 - 1 or extent.x1 > box.x1 + 1 or extent.y1 > box.y1 + 1:
                    found.append(text.get_text())
            return found

        assert outside(grid_data.plot(benchmarks=corners)) == []
        # mutation control: always offsetting up and right runs the labels off the axes
        monkeypatch.setattr(
            module, "_offset", lambda fx, fy, *, right, up: ((7.0, 5.0), "left", "bottom")
        )
        assert outside(grid_data.plot(benchmarks=corners))

    def test_a_benchmark_of_another_fit_is_refused(self, w2: Any, grid_data: Any) -> None:
        with pytest.raises(ValueError, match="another estimate"):
            grid_data.plot(benchmarks=[replace(w2, psi_long=w2.psi_long + 0.3)])
        with pytest.raises(ValueError, match="Riesz second moment"):
            grid_data.plot(benchmarks=[replace(w2, nu2_long=w2.nu2_long * 1.2)])
        assert grid_data.plot(benchmarks=[w2])  # control: the matching record draws

    def test_a_benchmark_of_another_estimand_is_refused(self, w2: Any, grid_data: Any) -> None:
        record = replace(w2, estimand="att")
        with pytest.raises(ValueError, match="benchmark is for"):
            grid_data.plot(benchmarks=[record])

    def test_the_colour_map_follows_the_cells(self, fit: Any, grid_data: Any) -> None:
        spans = _by_gid(grid_data.plot().axes[0], "cleverly-contour-fill")[0]
        assert spans.get_cmap().name == "cleverly-diverging"
        small = contour_data(fit, "ate", grid_size=6, grid_bounds=(0.05, 0.05))
        ax = small.plot().axes[0]
        assert _by_gid(ax, "cleverly-contour-fill")[0].get_cmap().name == "cleverly-sequential_r"
        assert not _by_gid(ax, "cleverly-null-contour")
        assert "No cell on this grid reaches 0" in note_text(ax.figure)
        assert "lower bound stays above 0" in note_text(ax.figure)

    def test_the_diverging_ends_use_palette_slots_one_and_two(self) -> None:
        cmap = module._colormap(True, dark_low=False)
        assert matplotlib.colors.to_hex(cmap(1.0)) == PALETTE[0]
        assert matplotlib.colors.to_hex(cmap(0.0)) == PALETTE[1]

    def test_a_constant_grid_refuses(self, grid_data: Any) -> None:
        flat = replace(
            grid_data,
            rows=tuple(
                (name, tuple([0.5] * len(values)) if name == "value" else values)
                for name, values in grid_data.rows
            ),
        )
        with pytest.raises(ValueError, match="same value"):
            flat.plot()

    def test_a_ragged_grid_refuses(self, grid_data: Any) -> None:
        ragged = replace(
            grid_data, rows=tuple((name, values[:-1]) for name, values in grid_data.rows)
        )
        with pytest.raises(ValueError, match="full cf_d by cf_y grid"):
            ragged.plot()

    def test_png_render(self, grid_data: Any, rv: Any, tmp_path: Any) -> None:
        path = tmp_path / "v13.png"
        fig = grid_data.plot(robustness=rv)
        fig.savefig(path, dpi=60)
        assert path.stat().st_size > 5_000
        assert_no_note_text(fig)

    def test_dispatch(self, grid_data: Any) -> None:
        fig = confounding(grid_data)
        assert fig.axes[0].get_title(loc="left") == "Sensitivity to hidden confounding"
        with pytest.raises(TypeError, match="SensitivityContour"):
            confounding(object())

    def test_a_supplied_axes_is_drawn_on(self, grid_data: Any) -> None:
        fig, ax = plt.subplots()
        assert grid_data.plot(ax=ax) is fig
        assert _by_gid(ax, "cleverly-null-contour")

    def test_the_contour_records_what_its_cells_read(
        self, fit: Any, grid_data: Any, rv: Any
    ) -> None:
        assert grid_data.scale == fit["ate"].scale == "difference"
        assert grid_data.nu2_estimator == "doubly_robust"
        assert grid_data.max_bias == pytest.approx(rv["max_bias"], rel=1e-12)
        assert grid_data.n_repeats == 1 and grid_data.assumption is None

    def test_the_end_that_moves_away_from_the_null_refuses(
        self, negative_fit: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # the blocker: a negative estimate's lower bound never reaches 0, so its map would
        # say the estimate holds while the upper bound reaches 0 inside the grid
        lower = contour_data(negative_fit, "ate", grid_size=12, grid_bounds=(0.6, 0.6))
        with pytest.raises(ValueError, match="only the upper bound can reach 0"):
            lower.plot()
        negative_rv = robustness_value(negative_fit, "ate")
        with pytest.raises(ValueError, match="bound='upper'"):
            lower.plot(robustness=negative_rv)
        upper = contour_data(
            negative_fit, "ate", grid_size=12, grid_bounds=(0.6, 0.6), bound="upper"
        )
        ax = upper.plot(robustness=negative_rv).axes[0]
        assert _by_gid(ax, "cleverly-null-contour")
        assert "reaches 0" not in note_text(ax.figure)
        # mutation control: without the side check the wrong end draws the false claim
        monkeypatch.setattr(module, "_check_side", lambda data, null: None)
        assert "No cell on this grid reaches 0" in note_text(lower.plot())

    def test_the_bold_line_is_named_by_the_scale(self, grid_data: Any) -> None:
        def label(fig: Any) -> list[str]:
            return [t.get_text() for t in fig.axes[0].texts if t.get_gid() == "cleverly-null-label"]

        assert label(grid_data.plot()) == ["No effect"]
        assert label(grid_data.plot(null=0.5)) == ["Lower bound = 0.5"]
        # a level has no no-effect value: it needs a null, and its line is never "No effect"
        level = replace(grid_data, estimand="ey1", scale="level")
        with pytest.raises(ValueError, match="no no-effect value"):
            level.plot()
        assert label(level.plot(null=0.0)) == ["Lower bound = 0"]
        # mutation control: the scale, not the name, decides
        named = replace(grid_data, estimand="ey1", scale="difference")
        assert label(named.plot()) == ["No effect"]

    def test_the_strengths_are_defined_in_the_packages_words(self, grid_data: Any) -> None:
        text = note_text(grid_data.plot())
        for phrase in (
            "the share of the outcome variation left after adjustment that",
            "the share of the average squared weight that only they would add",
        ):
            assert phrase in TEXTS["assumption_bounds"]
            assert phrase in text
        assert "not a partial R\u00b2 with the treatment" in text
        assert "partial R" not in grid_data.plot().axes[0].get_xlabel()

    def test_a_robustness_value_from_other_settings_is_refused(
        self, fit: Any, grid_data: Any, rv: Any
    ) -> None:
        plugin = robustness_value(fit, "ate", nu2_estimator="plugin")
        with pytest.raises(ValueError, match="nu2_estimator='plugin'"):
            grid_data.plot(robustness=plugin)
        with pytest.raises(ValueError, match="bias scale"):
            replace(grid_data, max_bias=grid_data.max_bias * 1.1).plot(robustness=rv)
        # control: both built with the plug-in estimator agree
        plugin_grid = contour_data(
            fit, "ate", grid_size=26, grid_bounds=(0.6, 0.6), nu2_estimator="plugin"
        )
        assert _by_gid(plugin_grid.plot(robustness=plugin).axes[0], "cleverly-robustness-value")

    def test_the_diverging_scale_is_symmetric_about_the_null(self) -> None:
        cmap, norm, diverging = module._colour_scale(-0.4, 2.0, 0.0, ratio=False, dark_low=False)
        full = module._colormap(True, dark_low=False)
        assert diverging
        # -0.4 lies a fifth as far from the null as 2.0, so it is a fifth as saturated
        assert np.allclose(cmap(norm(-0.4)), full(0.4), atol=0.01)
        assert np.allclose(cmap(norm(2.0)), full(1.0), atol=0.01)
        # mutation control: a two-slope norm would paint -0.4 as deep as 2.0
        assert not np.allclose(cmap(norm(-0.4)), full(0.0), atol=0.05)

    def test_repeats_and_a_consistent_mechanism_are_stated(self, grid_data: Any) -> None:
        plain = note_text(grid_data.plot())
        assert "median over" not in plain and TEXTS["consistent_mechanism"][:40] not in plain
        flagged = replace(grid_data, n_repeats=3, assumption="the sentence")
        text = note_text(flagged.plot())
        assert "median over the fit's 3 draws" in text
        assert " ".join(TEXTS["consistent_mechanism"].split()) in text
        assert flagged.summary().note("consistent_mechanism")
        with pytest.raises(KeyError):
            grid_data.summary().note("consistent_mechanism")


def _by_gid_axes(fig: Any, gid: str) -> Any:
    (found,) = [axis for axis in fig.axes if axis.get_gid() == gid]
    return found


# ----------------------------------------------------------------------- V14


def _heat(ax: Any) -> np.ndarray[Any, Any]:
    (mesh,) = _by_gid(ax, "cleverly-heatmap")
    return np.ma.getdata(mesh.get_array()).reshape(mesh.get_array().shape)


def _crossing_surface(result: SimulatedConfoundingResult) -> SimulatedConfoundingResult:
    """The same grid with estimates that cross zero between the second and third column."""
    cells = tuple(
        replace(
            cell, estimate=0.8 - 1.0 * (cell.treatment_strength / 0.1) + 0.1 * cell.outcome_strength
        )
        for cell in result.cells
    )
    return replace(result, cells=cells)


class TestSurface:
    def test_heat_map_equals_the_cells(self, surface_result: Any) -> None:
        fig = surface_result.plot()
        ax = fig.axes[0]
        assert (
            " ".join(ax.get_title(loc="left").split()) == "Estimate under a simulated common cause"
        )
        heat = _heat(ax)
        assert heat.shape == (3, 4)  # outcome strengths (rows) by treatment strengths (columns)
        for cell in surface_result.cells:
            row = (0.0, 0.5, 1.0).index(cell.outcome_strength)
            column = (0.0, 0.1, 0.2, 0.3).index(cell.treatment_strength)
            assert heat[row, column] == pytest.approx(cell.estimate)
        assert [t.get_text() for t in ax.get_xticklabels()] == ["0", "0.1", "0.2", "0.3"]
        assert [t.get_text() for t in ax.get_yticklabels()] == ["0", "0.5", "1"]
        assert "\n" not in ax.get_title(loc="left")  # the title uses the full width
        xlabel, ylabel = (" ".join(t.split()) for t in (ax.get_xlabel(), ax.get_ylabel()))
        assert xlabel == (
            "Share of units whose treatment is flipped (those highest on the simulated cause)"
        )
        assert ylabel == "Outcome shift per SD of the simulated cause, sign reversed"

    def test_the_note_says_it_is_qualitative(
        self, surface_result: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        text = note_text(surface_result.plot())
        assert "Qualitative: not a bound or an interval" in text
        # mutation control for rule 9: a note without the statement would fail the check above
        monkeypatch.setattr(module, "QUALITATIVE_NOTE", "")
        assert "not a bound or an interval" not in note_text(surface_result.plot())

    def test_no_interval_artist_is_drawn(self, surface_result: Any) -> None:
        ax = surface_result.plot().axes[0]
        gids = {artist.get_gid() for artist in [*ax.collections, *ax.patches, *ax.lines]}
        assert gids <= {
            "cleverly-heatmap",
            "cleverly-null-crossing",
            "cleverly-failed-cell",
            "cleverly-calibration",
        }

    def test_the_null_crossing_is_outlined_on_the_cell_edges(self, surface_result: Any) -> None:
        crossing = _crossing_surface(surface_result)
        ax = crossing.plot().axes[0]
        (outline,) = _by_gid(ax, "cleverly-null-crossing")
        heat = _heat(ax)
        assert (heat[:, 0] > 0).all() and (heat[:, 1] < 0).all()  # the sign changes after 0
        edge = 0.5  # equal cells: the edge between the first and second declared values
        # estimates at x = 0 and 0.1 are 0.8 and -0.2 (+0.1y): the sign flips between them
        # whenever 0.8 + 0.1y > 0 > -0.2 + 0.1y, so every row is outlined at the 0 / 0.1 edge
        xs = {
            round(float(seg[0][0]), 9) for seg in outline.get_segments() if seg[0][0] == seg[1][0]
        }
        assert xs == {edge}
        assert len(outline.get_segments()) == 3
        assert "crosses 0" in note_text(ax.figure)

    def test_no_outline_without_a_crossing(self, surface_result: Any) -> None:
        # mutation control: an uncrossed surface draws no outline and says so
        fig = surface_result.plot()
        assert not _by_gid(fig.axes[0], "cleverly-null-crossing")
        assert "no outline" in note_text(fig)

    def test_a_level_has_no_outline(self, surface_result: Any) -> None:
        crossing = _crossing_surface(surface_result)
        level = replace(crossing, scale="level")
        fig = level.plot()
        assert not _by_gid(fig.axes[0], "cleverly-null-crossing")
        assert "outline" not in note_text(fig)
        # mutation control: the scale decides, not the name, so a mean-like name on a
        # difference is still outlined
        named = replace(crossing, estimand="ey1")
        assert _by_gid(named.plot().axes[0], "cleverly-null-crossing")

    def test_a_ratio_is_outlined_at_one_on_a_log_colour_scale(self, surface_result: Any) -> None:
        ratio = replace(
            surface_result,
            scale="ratio",
            movement_scale="log_ratio",
            estimand="rr",
            cells=tuple(
                replace(cell, estimate=float(np.exp(0.15 - 3.0 * cell.treatment_strength)))
                for cell in surface_result.cells
            ),
        )
        fig = ratio.plot()
        ax = fig.axes[0]
        (outline,) = _by_gid(ax, "cleverly-null-crossing")
        assert {round(float(seg[0][0]), 9) for seg in outline.get_segments()} == {0.5}
        (mesh,) = _by_gid(ax, "cleverly-heatmap")
        assert isinstance(mesh.norm, matplotlib.colors.LogNorm)
        assert 1.0 in list(_by_gid_axes(fig, "cleverly-colorbar").get_yticks())
        assert "crosses 1" in note_text(fig)
        # control: the same cells on a difference scale cross nothing at 0
        difference = replace(ratio, scale="difference").plot()
        assert not _by_gid(difference.axes[0], "cleverly-null-crossing")

    def test_the_colour_bar_marks_the_null(self, surface_result: Any) -> None:
        fig = _crossing_surface(surface_result).plot()
        assert 0.0 in list(_by_gid_axes(fig, "cleverly-colorbar").get_yticks())

    def test_diverging_scale_only_when_the_cells_straddle_the_null(
        self, surface_result: Any
    ) -> None:
        assert (
            _by_gid(surface_result.plot().axes[0], "cleverly-heatmap")[0]
            .get_cmap()
            .name.startswith("cleverly-sequential")
        )
        straddle = _crossing_surface(surface_result)
        assert (
            _by_gid(straddle.plot().axes[0], "cleverly-heatmap")[0].get_cmap().name
            == "cleverly-diverging"
        )

    def test_failed_cells_are_hatched_and_counted(self, surface_result: Any) -> None:
        failure = ReplicationFailure(0, 0, "ValueError", "boom")
        cells = list(surface_result.cells)
        cells[4] = replace(cells[4], estimate=None, displacement=None, failure=failure)
        broken = replace(surface_result, cells=tuple(cells))
        fig = broken.plot()
        ax = fig.axes[0]
        (hatched,) = _by_gid(ax, "cleverly-failed-cell")
        assert hatched.get_hatch()
        assert np.isnan(_heat(ax)[1, 1])  # the failed cell holds no colour
        text = note_text(fig)
        assert "Hatched: a cell that failed to refit." in text
        assert text.count("1 of 12") == 1  # the summary's count, once
        # control: the intact surface draws no hatch and no count
        intact = surface_result.plot()
        assert not _by_gid(intact.axes[0], "cleverly-failed-cell")
        assert "Hatched" not in note_text(intact)

    def test_calibrations_are_marked_only_on_a_matching_scale(self, surface_result: Any) -> None:
        fig = surface_result.plot()
        ax = fig.axes[0]
        marks = _by_gid(ax, "cleverly-calibration")
        treatment = [row for row in surface_result.calibrations if row.role == "treatment"]
        columns = [line for line in marks if line.get_marker() == "^"]
        assert len(columns) == len(treatment) == 2  # binary treatment: a flip share
        declared = [0.0, 0.1, 0.2, 0.3]
        assert sorted(float(line.get_xdata()[0]) for line in columns) == pytest.approx(
            sorted(module._position(row.strength, declared) for row in treatment)
        )
        names = sorted(
            t.get_text() for t in ax.texts if t.get_gid() == "cleverly-calibration-label"
        )
        # the Gaussian outcome law subtracts its strength, so a row sits at minus its coefficient
        outcome = {row.covariate: -row.strength for row in surface_result.calibrations
                   if row.role == "outcome"}  # fmt: skip
        inside = {name: value for name, value in outcome.items() if 0.0 <= value <= 1.0}
        assert inside  # W3's coefficient is negative, so it lands on the axis
        assert names == sorted(["W2", "W3", *inside])
        rows = [line for line in marks if line.get_marker() == ">"]
        assert [float(line.get_ydata()[0]) for line in rows] == pytest.approx(
            [module._position(value, [0.0, 0.5, 1.0]) for value in inside.values()]
        )
        text = note_text(fig)
        assert "(sign reversed)" in text and "Beyond the axes" in text

    def test_an_unsigned_block_strength_is_not_marked(self, surface_result: Any) -> None:
        # mutation control: the same rows as unsigned block strengths have no signed position
        blocks = replace(
            surface_result,
            calibrations=tuple(
                replace(row, method="block multiple correlation times target SD (unsigned)")
                if row.family == "gaussian"
                else row
                for row in surface_result.calibrations
            ),
        )
        fig = blocks.plot()
        assert all(
            line.get_marker() == "^" for line in _by_gid(fig.axes[0], "cleverly-calibration")
        )
        assert "W3 (outcome, unsigned block strength" in note_text(fig)

    def test_a_scale_mismatch_marks_nothing(self, surface_result: Any) -> None:
        # mutation control: a continuous treatment axis is a signed coefficient, which a
        # share of predictions changed is not, so the same rows are listed and never drawn
        continuous = replace(surface_result, treatment_family="continuous")
        fig = continuous.plot()
        marks = _by_gid(fig.axes[0], "cleverly-calibration")
        assert not [line for line in marks if line.get_marker() == "^"]
        assert "W2 (treatment, share of predictions changed" in note_text(fig)
        assert fig.axes[0].get_xlabel() == "Dose shift per SD of the simulated cause"

    def test_png_render(self, surface_result: Any, tmp_path: Any) -> None:
        path = tmp_path / "v14.png"
        fig = surface_result.plot()
        fig.savefig(path, dpi=60)
        assert path.stat().st_size > 5_000
        assert_no_note_text(fig)

    def test_dispatch(self, surface_result: Any) -> None:
        assert (
            confounding(surface_result).axes[0].get_title(loc="left")
            == "Estimate under a simulated common cause"
        )
        with pytest.raises(TypeError, match="SimulatedConfoundingResult"):
            confounding(1)


class TestSurfaceWords:
    def test_each_axis_label_follows_its_law(self, surface_result: Any) -> None:
        def xlabel(result: Any) -> str:
            return " ".join(result.plot().axes[0].get_xlabel().split())

        def ylabel(result: Any) -> str:
            return " ".join(result.plot().axes[0].get_ylabel().split())

        tail = "those highest on the simulated cause"
        assert tail in TEXTS["law_binary"] and tail in xlabel(surface_result)
        categorical = replace(
            surface_result, treatment_family="categorical", arm_pair=("low", "high")
        )
        assert "arms low and high" in TEXTS["law_categorical"].format(a="low", b="high")
        assert "arms low and high whose arms are swapped" in xlabel(categorical)
        assert "per standard deviation" in TEXTS["law_continuous"]
        assert xlabel(replace(surface_result, treatment_family="continuous")).startswith(
            "Dose shift per SD"
        )
        assert "with the sign reversed" in TEXTS["law_outcome_gaussian"]
        assert ylabel(surface_result).endswith("sign reversed")
        binomial = replace(surface_result, outcome_family="binomial", calibrations=())
        assert tail in TEXTS["law_outcome_binomial"]
        assert ylabel(binomial) == f"Share of units whose outcome is flipped ({tail})"
        bounded = replace(surface_result, bounded_outcome_law="logit", calibrations=())
        assert "not SD units" in ylabel(bounded)

    def test_the_law_is_in_the_notes(self, surface_result: Any) -> None:
        text = note_text(surface_result.plot())
        assert TEXTS["law_binary"] in text and TEXTS["law_outcome_gaussian"] in text
        # control: a binomial outcome reads its own law, not the Gaussian one
        binomial = note_text(
            replace(surface_result, outcome_family="binomial", calibrations=()).plot()
        )
        assert TEXTS["law_outcome_gaussian"] not in binomial

    def test_an_uneven_grid_draws_equal_cells(self, surface_result: Any) -> None:
        spread = {0.0: 0.0, 0.1: 0.01, 0.2: 0.02, 0.3: 0.4}
        uneven = replace(
            surface_result,
            cells=tuple(
                replace(cell, treatment_strength=spread[cell.treatment_strength])
                for cell in surface_result.cells
            ),
            calibrations=(),
        )
        fig = uneven.plot()
        ax = fig.axes[0]
        (mesh,) = _by_gid(ax, "cleverly-heatmap")
        widths = np.diff(mesh.get_coordinates()[0, :, 0])
        assert np.allclose(widths, 1.0)
        assert [t.get_text() for t in ax.get_xticklabels()] == ["0", "0.01", "0.02", "0.4"]
        assert "not to scale" in note_text(fig)
        # control: the evenly spaced grid says nothing about scale
        assert "not to scale" not in note_text(surface_result.plot())
        # a value between declared strengths sits between their cells
        assert module._position(0.015, [0.0, 0.01, 0.02, 0.4]) == pytest.approx(1.5)
        assert module._position(0.5, [0.0, 0.01, 0.02, 0.4]) is None

    def test_repeats_are_stated(self, surface_result: Any) -> None:
        assert "cross-fitting draws" not in note_text(surface_result.plot())
        text = note_text(replace(surface_result, n_repeats=3).plot())
        assert "median over the fit's 3 cross-fitting draws" in text


@pytest.fixture(scope="module")
def cde_surface() -> SimulatedConfoundingResult:
    frame, _ = make_cde(n=400, seed=3)
    design = PointTreatment(
        outcome="Y", treatment="A", adjustment=("W1", "W2", "W3"), intermediate="Z"
    )
    result = (
        CausalStudy(frame, design=design)
        .identify(ControlledDirectEffect(intermediate=1.0))
        .estimate(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            cross_fit=False,
            random_state=2,
            simultaneous=False,
        )
    )
    grid = ConfounderStrengthGrid(
        treatment=(0.0,), intermediate=(0.0, 0.1, 0.25), outcome=(0.0, 0.3, 0.6)
    )
    return simulated_confounding(result, "ate", grid=grid, random_state=1)


def test_a_controlled_direct_effect_moves_the_intermediate(cde_surface: Any) -> None:
    fig = cde_surface.plot()
    panels = [axis for axis in fig.axes if _by_gid(axis, "cleverly-heatmap")]
    assert len(panels) == 1  # the treatment is held, so it is not an axis
    ax = panels[0]
    assert " ".join(ax.get_xlabel().split()) == (
        "Share of units whose Z is flipped (those highest on the simulated cause)"
    )
    assert [t.get_text() for t in ax.get_xticklabels()] == ["0", "0.1", "0.25"]
    subtitle = subtitle_text(ax) or ""
    assert "unperturbed estimate" in subtitle
    assert "common cause of the intermediate and the outcome" in note_text(fig)
    assert "the target sets Z to 1" in subtitle
    assert "treatment is flipped" not in note_text(fig)
    heat = _heat(ax)
    for cell in cde_surface.cells:
        row = (0.0, 0.3, 0.6).index(cell.outcome_strength)
        column = (0.0, 0.1, 0.25).index(cell.intermediate_strength)
        assert heat[row, column] == pytest.approx(cell.estimate)


def _vector(result: SimulatedConfoundingResult) -> SimulatedConfoundingResult:
    """The surface re-read as a two-component vector treatment with a third axis of 4 values."""
    grid = ConfounderStrengthGrid(
        treatment={"F": (0.0, 0.2), "D": (0.0, 0.5, 1.0)}, outcome=(0.0, 0.2, 0.4)
    )
    cells = []
    for f, d, y in (
        (f, d, y) for f in (0.0, 0.2) for d in (0.0, 0.5, 1.0) for y in (0.0, 0.2, 0.4)
    ):
        cells.append(
            replace(
                result.cells[0],
                treatment_strength=(f, d),
                outcome_strength=y,
                estimate=1.0 + f - 0.3 * d + y,
                displacement=0.0,
            )
        )
    return replace(
        result,
        grid=grid,
        cells=tuple(cells),
        treatment_family="vector",
        treatment_components=("F", "D"),
        component_kinds=("binary", "dose"),
        calibrations=(),
    )


class TestSmallMultiples:
    def test_an_extra_axis_becomes_one_panel_per_declared_value(self, surface_result: Any) -> None:
        vector = _vector(surface_result)
        fig = vector.plot()
        panels = [axis for axis in fig.axes if _by_gid(axis, "cleverly-heatmap")]
        assert len(panels) == 3  # D at 0, 0.5, 1; the first axis F and the outcome are the plot's
        assert [axis.get_title() for axis in panels] == [
            "D shift per SD = 0",
            "D shift per SD = 0.5",
            "D shift per SD = 1",
        ]
        assert panels[0].get_xlabel() == "Share of F flipped"
        for axis, d in zip(panels, (0.0, 0.5, 1.0), strict=True):
            heat = _heat(axis)
            assert heat.shape == (3, 2)
            assert heat[2, 1] == pytest.approx(1.0 + 0.2 - 0.3 * d + 0.4)
        # one colour scale across panels, and one bar
        norms = {
            (m.norm.vmin, m.norm.vmax) for axis in panels for m in _by_gid(axis, "cleverly-heatmap")
        }
        assert len(norms) == 1
        assert len([axis for axis in fig.axes if axis.get_gid() == "cleverly-colorbar"]) == 1

    def test_calibrations_are_marked_on_every_panel(self, surface_result: Any) -> None:
        row = replace(surface_result.calibrations[0], component="F", family="binomial")
        vector = replace(_vector(surface_result), calibrations=(replace(row, strength=0.1),))
        fig = vector.plot()
        panels = [axis for axis in fig.axes if _by_gid(axis, "cleverly-heatmap")]
        for axis in panels:
            (mark,) = _by_gid(axis, "cleverly-calibration")
            assert float(mark.get_xdata()[0]) == pytest.approx(0.5)  # halfway from 0 to 0.2
        # control: a row for the dose component D is not on F's axis
        dose = replace(_vector(surface_result), calibrations=(replace(row, component="D"),))
        assert not any(_by_gid(axis, "cleverly-calibration") for axis in dose.plot().axes)

    def test_at_fixes_the_extra_axis(self, surface_result: Any) -> None:
        vector = _vector(surface_result)
        fig = vector.plot(at={"D": 0.5})
        panels = [axis for axis in fig.axes if _by_gid(axis, "cleverly-heatmap")]
        assert len(panels) == 1
        assert _heat(panels[0])[0, 0] == pytest.approx(1.0 - 0.15)
        assert len(vector.plot(at={"D": [0.0, 1.0]}).axes) < len(vector.plot().axes)

    def test_a_grid_with_too_many_panels_refuses(
        self, surface_result: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        vector = _vector(surface_result)
        monkeypatch.setattr(module, "MAX_PANELS", 2)
        with pytest.raises(ValueError, match="pass at="):
            vector.plot()
        assert vector.plot(at={"D": [0.0, 1.0]})  # two panels fit

    def test_at_must_name_an_extra_axis_and_a_declared_value(self, surface_result: Any) -> None:
        vector = _vector(surface_result)
        with pytest.raises(ValueError, match="not extra axes"):
            vector.plot(at={"F": 0.0})
        with pytest.raises(ValueError, match="not a declared value"):
            vector.plot(at={"D": 0.7})

    def test_axes_cannot_be_given_for_several_panels(self, surface_result: Any) -> None:
        _, ax = plt.subplots()
        with pytest.raises(ValueError, match="several panels"):
            _vector(surface_result).plot(ax=ax)

    def test_panel_figure_renders(self, surface_result: Any, tmp_path: Any) -> None:
        path = tmp_path / "panels.png"
        fig = _vector(surface_result).plot()
        fig.savefig(path, dpi=60)
        assert path.stat().st_size > 5_000
        assert_no_note_text(fig)

    def test_no_estimate_refuses(self, surface_result: Any) -> None:
        failure = ReplicationFailure(0, 0, "ValueError", "boom")
        cells = tuple(
            replace(cell, estimate=None, displacement=None, failure=failure)
            for cell in surface_result.cells
        )
        with pytest.raises(ValueError, match="no cell"):
            replace(surface_result, cells=cells).plot()


class TestAxisWords:
    """xlabel= and ylabel= name the two strengths in a page's own words."""

    def test_the_contour_takes_its_axis_words(self, grid_data: Any) -> None:
        ax = confounding(grid_data, xlabel="Weight side", ylabel="Score side").axes[0]
        assert ax.get_xlabel() == "Weight side" and ax.get_ylabel() == "Score side"
        # the control: the defaults stay when the words are omitted
        plain = confounding(grid_data).axes[0]
        assert plain.get_xlabel() == "Treatment-side strength, cf_d"
        assert plain.get_ylabel() == "Outcome-side strength, cf_y"

    def test_the_surface_takes_its_axis_words(self, surface_result: Any) -> None:
        fig = confounding(surface_result, xlabel="Weight side", ylabel="Score side")
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Weight side" and ax.get_ylabel() == "Score side"
        # The colour bar keeps its own label: it is not a panel's axis.
        bars = [axis for axis in fig.axes if axis.get_gid() == "cleverly-colorbar"]
        assert bars and all(bar.get_ylabel() != "Score side" for bar in bars)
        plain = confounding(surface_result).axes[0]
        assert plain.get_xlabel() != "Weight side" and plain.get_ylabel() != "Score side"

    def test_every_panel_of_small_multiples_takes_them(self, surface_result: Any) -> None:
        fig = _vector(surface_result).plot(xlabel="Weight side", ylabel="Score side")
        panels = [axis for axis in fig.axes if _by_gid(axis, "cleverly-heatmap")]
        assert len(panels) == 3
        assert {axis.get_xlabel() for axis in panels} <= {"Weight side", ""}
        assert any(axis.get_xlabel() == "Weight side" for axis in panels)
        assert panels[0].get_ylabel() == "Score side"


# ----------------------------------------------------------------------- benchmarks on the map


def _level_labels(ax: Any) -> list[Any]:
    """The contour level labels: the texts no gid names."""
    return [text for text in ax.texts if text.get_gid() is None and text.get_text().strip()]


def _overlaps(fig: Any) -> list[str]:
    """Each level label whose box overlaps a benchmark's name or triangle."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax = fig.axes[0]
    names = [t.get_window_extent(renderer) for t in _by_gid(ax, "cleverly-benchmark-label")]
    marks = [
        line.get_transform().transform(line.get_xydata())[0]
        for line in _by_gid(ax, "cleverly-benchmark")
    ]
    found = []
    for text in _level_labels(ax):
        box = text.get_window_extent(renderer)
        if any(box.overlaps(name) for name in names) or any(
            box.x0 - 5 <= x <= box.x1 + 5 and box.y0 - 5 <= y <= box.y1 + 5 for x, y in marks
        ):
            found.append(text.get_text())
    return found


class TestBenchmarksOnTheMap:
    def test_a_benchmark_on_the_bottom_edge_keeps_its_whole_triangle(
        self, w2: Any, grid_data: Any
    ) -> None:
        edge = replace(w2, covariates=("no outcome gain",), cf_d=0.3, cf_y=0.0)
        ax = grid_data.plot(benchmarks=[edge, w2]).axes[0]
        triangles = {
            tuple(line.get_xydata()[0]): line for line in _by_gid(ax, "cleverly-benchmark")
        }
        on_edge = triangles[(0.3, 0.0)]
        # The triangle sits on the axis's lower limit, where clipping would cut it in half.
        assert ax.get_ylim()[0] == pytest.approx(0.0)
        assert on_edge.get_clip_on() is False
        # Control: every triangle is drawn the same way, not only the one on the edge.
        assert all(line.get_clip_on() is False for line in triangles.values())

    def test_a_level_label_keeps_off_a_benchmark_and_its_name(
        self, w2: Any, grid_data: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Put a benchmark where a level label would go on the bare map.
        bare = grid_data.plot().axes[0]
        x, y = _level_labels(bare)[0].get_position()
        onto = replace(w2, covariates=("W2, W3, W4",), cf_d=float(x), cf_y=float(y))
        fig = grid_data.plot(benchmarks=[onto])
        assert [
            tuple(line.get_xydata()[0]) for line in _by_gid(fig.axes[0], "cleverly-benchmark")
        ] == [pytest.approx((x, y))]
        assert _overlaps(fig) == []
        # Every level is still labelled.
        assert len(_level_labels(fig.axes[0])) == len(_level_labels(bare))
        # Mutation control: with no zone around the marks the label lands on the benchmark.
        monkeypatch.setattr(module, "_MARK_ZONE", (0.0, 0.0))
        assert _overlaps(grid_data.plot(benchmarks=[onto]))


def _segments(points: list[tuple[float, float]]) -> Any:
    """A contour set's shape, one level of one segment through ``points``."""
    from types import SimpleNamespace

    return SimpleNamespace(allsegs=[[np.asarray(points, dtype=float)]])


class TestLabelsAvoidTheMarks:
    #: A level line across the map, through its middle.
    LINE = tuple((0.1 * i, 0.6) for i in range(1, 10))

    def test_a_level_label_keeps_off_the_robustness_value(self) -> None:
        bare = module._label_points(_segments(list(self.LINE)), 1.0, 1.0)
        assert bare == [pytest.approx((0.6, 0.6))]
        # The robustness value sits where the label went: the label moves out of its zone.
        (moved,) = module._label_points(_segments(list(self.LINE)), 1.0, 1.0, [(0.6, 0.6)])
        assert abs(moved[0] - 0.6) >= module._MARK_ZONE[0]
        # Mutation control: with no zone the label stays on the mark.
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(module, "_MARK_ZONE", (0.0, 0.0))
            assert module._label_points(_segments(list(self.LINE)), 1.0, 1.0, [(0.6, 0.6)]) == bare

    def test_the_no_effect_label_keeps_off_benchmarks_and_the_robustness_value(self) -> None:
        bold = _segments(list(self.LINE))
        middle = module._null_label(bold, None, 1.0, 1.0)
        # A benchmark on the label's spot moves it out of the benchmark's zone.
        moved = module._null_label(bold, None, 1.0, 1.0, [middle])
        assert abs(moved[0] - middle[0]) >= module._MARK_ZONE[0]
        # With the robustness value too, the label keeps a fifth of the map from it.
        both = module._null_label(bold, 0.2, 1.0, 1.0, [middle])
        assert abs(both[0] - middle[0]) >= module._MARK_ZONE[0]
        assert np.hypot(both[0] - 0.2, both[1] - 0.2) > 0.2
        # Control: without marks the label takes the middle of the line.
        assert middle == pytest.approx((0.5, 0.6))


def test_calibration_labels_of_close_marks_go_to_opposite_sides() -> None:
    fig, axis = plt.subplots()
    placed, _ = module._mark_calibrations(
        axis,
        [("first", 0.19), ("second", 0.2)],
        [],
        [0.0, 0.1, 0.2, 0.4],
        [0.0, 1.0],
        name_x=True,
        name_y=False,
        reversed_y=False,
    )
    labels = {
        text.get_text(): text.get_horizontalalignment()
        for text in axis.texts
        if text.get_gid() == module.GID_CALIBRATION_LABEL
    }
    assert len(placed) == 2
    assert labels == {"first": "right", "second": "left"}
    far = module._label_sides([0.0, 2.0, None])
    assert far == [0, 0, 0]
    plt.close(fig)
