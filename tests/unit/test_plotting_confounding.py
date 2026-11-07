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

from cleverly import ATE, CausalStudy, PointTreatment
from cleverly.datasets import make_linear_ate
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
def grid_data(fit: Any) -> SensitivityContour:
    # reaches past the robustness value, so the null contour is on the map
    return contour_data(fit, "ate", grid_size=26, grid_bounds=(0.6, 0.6))


@pytest.fixture(scope="module")
def rv(fit: Any) -> Any:
    return robustness_value(fit, "ate")


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
        assert ax.get_title(loc="left") == "How strong would hidden confounding need to be?"
        subtitle = subtitle_text(ax)
        assert subtitle is not None
        assert "Lower bound of ate" in subtitle and "\u03c1 = 1" in subtitle
        assert f"{rv['rv']:.3g}" in subtitle
        assert "cf_d" in ax.get_xlabel() and "cf_y" in ax.get_ylabel()
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
        with pytest.raises(ValueError, match="lower bound reaches the null"):
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
            replace(grid_data, bound="upper"), null=-5.0
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

    def test_benchmarks_are_marked_at_their_multiples(self, fit: Any, grid_data: Any) -> None:
        record = benchmark(fit, ["W2"])
        fig = grid_data.plot(benchmarks=record, multiples=(1.0, 2.0, 40.0))
        ax = fig.axes[0]
        points = sorted(tuple(line.get_xydata()[0]) for line in _by_gid(ax, "cleverly-benchmark"))
        expected = sorted((k * record.cf_d, k * record.cf_y) for k in (1.0, 2.0))
        assert np.allclose(points, expected)
        assert "beyond the grid" in note_text(fig)
        assert "W2 \u00d740" in note_text(fig)

    def test_a_benchmark_of_another_estimand_is_refused(self, fit: Any, grid_data: Any) -> None:
        record = replace(benchmark(fit, ["W2"]), estimand="att")
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
        grid_data.plot(robustness=rv).savefig(path, dpi=60)
        assert path.stat().st_size > 5_000

    def test_dispatch(self, grid_data: Any) -> None:
        fig = confounding(grid_data)
        assert fig.axes[0].get_title(loc="left").startswith("How strong")
        with pytest.raises(TypeError, match="SensitivityContour"):
            confounding(object())

    def test_a_supplied_axes_is_drawn_on(self, grid_data: Any) -> None:
        fig, ax = plt.subplots()
        assert grid_data.plot(ax=ax) is fig
        assert _by_gid(ax, "cleverly-null-contour")


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
            " ".join(ax.get_title(loc="left").split())
            == "How far does the estimate move under a simulated common cause?"
        )
        heat = _heat(ax)
        assert heat.shape == (3, 4)  # outcome strengths (rows) by treatment strengths (columns)
        for cell in surface_result.cells:
            row = (0.0, 0.5, 1.0).index(cell.outcome_strength)
            column = (0.0, 0.1, 0.2, 0.3).index(cell.treatment_strength)
            assert heat[row, column] == pytest.approx(cell.estimate)
        assert [t.get_text() for t in ax.get_xticklabels()] == ["0", "0.1", "0.2", "0.3"]
        assert [t.get_text() for t in ax.get_yticklabels()] == ["0", "0.5", "1"]
        assert "flipped" in ax.get_xlabel()
        assert "signed coefficient" in ax.get_ylabel().replace("\n", " ")

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
        edge = (0.0 + 0.1) / 2.0  # midpoint between declared 0.0 and 0.1
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
        level = replace(surface_result, estimand="ey1")
        fig = level.plot()
        assert not _by_gid(fig.axes[0], "cleverly-null-crossing")
        assert "outline" not in note_text(fig)

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
        assert "Hatched cells (1 of 12 drawn)" in note_text(fig)
        # control: the intact surface draws no hatch and no count
        intact = surface_result.plot()
        assert not _by_gid(intact.axes[0], "cleverly-failed-cell")
        assert "Hatched" not in note_text(intact)

    def test_calibrations_are_marked_only_on_a_matching_scale(self, surface_result: Any) -> None:
        fig = surface_result.plot()
        ax = fig.axes[0]
        marks = _by_gid(ax, "cleverly-calibration")
        treatment = [row for row in surface_result.calibrations if row.role == "treatment"]
        assert len(marks) == len(treatment) == 2  # binary treatment: flip probability scale
        assert sorted(float(line.get_xdata()[0]) for line in marks) == pytest.approx(
            sorted(row.strength for row in treatment)
        )
        text = note_text(fig)
        assert "another scale than an axis" in text  # the Gaussian outcome rows are listed
        assert "W2 (outcome, signed coefficient" in text

    def test_a_scale_mismatch_marks_nothing(self, surface_result: Any) -> None:
        # mutation control: a continuous treatment axis is a signed coefficient, which a
        # share of predictions changed is not, so the same rows are listed and never drawn
        continuous = replace(surface_result, treatment_family="continuous")
        fig = continuous.plot()
        assert not _by_gid(fig.axes[0], "cleverly-calibration")
        assert "W2 (treatment, share of predictions changed" in note_text(fig)

    def test_png_render(self, surface_result: Any, tmp_path: Any) -> None:
        path = tmp_path / "v14.png"
        surface_result.plot().savefig(path, dpi=60)
        assert path.stat().st_size > 5_000

    def test_dispatch(self, surface_result: Any) -> None:
        assert confounding(surface_result).axes[0].get_title(loc="left").startswith("How far")
        with pytest.raises(TypeError, match="SimulatedConfoundingResult"):
            confounding(1)


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
        calibrations=(),
    )


class TestSmallMultiples:
    def test_an_extra_axis_becomes_one_panel_per_declared_value(self, surface_result: Any) -> None:
        vector = _vector(surface_result)
        fig = vector.plot()
        panels = [axis for axis in fig.axes if _by_gid(axis, "cleverly-heatmap")]
        assert len(panels) == 3  # D at 0, 0.5, 1; the first axis F and the outcome are the plot's
        assert [axis.get_title() for axis in panels] == ["D = 0", "D = 0.5", "D = 1"]
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
        _vector(surface_result).plot().savefig(path, dpi=60)
        assert path.stat().st_size > 5_000

    def test_no_estimate_refuses(self, surface_result: Any) -> None:
        failure = ReplicationFailure(0, 0, "ValueError", "boom")
        cells = tuple(
            replace(cell, estimate=None, displacement=None, failure=failure)
            for cell in surface_result.cells
        )
        with pytest.raises(ValueError, match="no cell"):
            replace(surface_result, cells=cells).plot()


def test_the_estimate_scale_marks_a_ratio_null_at_one(surface_result: Any) -> None:
    ratio = replace(surface_result, movement_scale="log_ratio", estimand="rr")
    assert module._null_of(ratio) == 1.0
    assert module._null_of(surface_result) == 0.0
    assert module._null_of(replace(surface_result, estimand="ey1")) is None
