"""X92 part 1: the plotting core, the forest (V1), the sensitivity family (V12) and overlap (V9).

The tests read artists, never pixels: the drawn data equal the object's numbers, the texts
match, the reference lines sit at their values. Each honesty rule has a test and a mutation
control, a deliberately wrong plot that the same check must reject.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex
from matplotlib.text import Text
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import ATE, CausalStudy, PointTreatment
from cleverly._inference_status import status_record
from cleverly._support import Support
from cleverly.datasets import make_missing_outcome
from cleverly.estimators import TMLE
from cleverly.plotting import (
    PALETTE,
    _core,
    forest,
    notes,
    overlap,
    sensitivity,
    style,
)
from cleverly.plotting import _forest as forest_module
from cleverly.sensitivity import marginal_sensitivity_bounds

pytestmark = pytest.mark.xdist_group("plotting_core")

WITHHELD = "estimated_mechanism_plugin"


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def binary_fit() -> Any:
    rng = np.random.default_rng(0)
    w = rng.normal(size=600)
    a = rng.binomial(1, 1 / (1 + np.exp(-1.5 * w)))
    y = rng.binomial(1, 1 / (1 + np.exp(-(w + a - 0.5))))
    frame = pd.DataFrame({"W": w, "A": a, "Y": y})
    return (
        TMLE(
            outcome_learner=LogisticRegression(),
            treatment_learner=LogisticRegression(),
            random_state=0,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W"])
        .single()
    )


@pytest.fixture(scope="module")
def missing_fit() -> Any:
    frame, _ = make_missing_outcome(n=400, seed=61)
    design = PointTreatment(
        outcome="Y", treatment="A", adjustment=["W1", "W2", "W3"], missingness="Delta"
    )
    return (
        CausalStudy(frame, design=design)
        .identify(ATE())
        .estimate(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            cross_fit=False,
            random_state=4,
        )
    )


def _interval_segments(ax: Any) -> list[np.ndarray]:
    segments = []
    for collection in ax.collections:
        if collection.get_gid() == "cleverly-interval":
            segments.extend(np.asarray(path) for path in collection.get_segments())
    return segments


def _points(ax: Any) -> list[tuple[float, float]]:
    return [tuple(line.get_xydata()[0]) for line in ax.lines if line.get_gid() == "cleverly-point"]


def _lines(ax: Any, gid: str) -> list[Any]:
    return [line for line in ax.lines if line.get_gid() == gid]


def _title(fig: Any) -> str:
    return fig.axes[0].get_title(loc="left")


def _key(ax: Any) -> list[str]:
    legend = ax.get_legend()
    return [] if legend is None else [text.get_text() for text in legend.get_texts()]


# ----------------------------------------------------------------------- style


def test_palette_is_the_validated_constant_in_order() -> None:
    report = (Path(_core.__file__).parent / "palette-validation.txt").read_text(encoding="utf-8")
    assert PALETTE == ("#2a78d6", "#eb6834", "#1baf7a")
    assert "ALL CHECKS PASS" in report
    assert "surface #ffffff" in report
    for color in PALETTE:
        assert color in report


def test_style_context_restores_the_callers_settings() -> None:
    before = dict(matplotlib.rcParams)
    with style():
        _, ax = plt.subplots()
        (line,) = ax.plot([0, 1], [0, 1])
        assert to_hex(line.get_color()) == PALETTE[0]
    assert dict(matplotlib.rcParams) == before


def test_a_plot_leaves_the_callers_settings_alone(binary_fit: Any) -> None:
    before = dict(matplotlib.rcParams)
    binary_fit.plot()
    assert dict(matplotlib.rcParams) == before


#: The longest default title a plot writes: a short description, so a report can carry it.
SHORT_TITLE = 50


def assert_short_title(fig: Any) -> None:
    """The figure's title is a short description, not a question.

    The title is the bold text on the figure (a figure-level header) or the left title of
    the first panel. A reader's own title can be anything; the defaults are short.
    """
    texts = [t.get_text() for t in fig.findobj(Text) if t.get_gid() == _core.GID_TITLE]
    title = (texts or [fig.axes[0].get_title(loc="left")])[0]
    assert title, "the figure has no title"
    assert len(title) <= SHORT_TITLE, f"the title is longer than {SHORT_TITLE}: {title!r}"
    assert not title.rstrip().endswith("?"), f"the title is a question: {title!r}"


def assert_no_note_text(fig: Any) -> None:
    """No text the figure draws carries a note the plot recorded.

    Notes live in :func:`cleverly.plotting.notes` and the plot's docstring, never on the
    figure. Each note is compared by its first 40 characters, its words joined by spaces.
    """
    drawn = [
        " ".join(text.get_text().split())
        for text in fig.findobj(Text)
        if text.get_visible() and text.get_text().strip()
    ]
    for note in notes(fig):
        head = " ".join(note.split())[:40]
        for text in drawn:
            assert head not in text, f"the figure draws its note {head!r}"


def _style_checks(fig: Any) -> None:
    """Every figure: opaque white face, a title, labelled axes, at most three palette colours.

    Every coloured (not grey) artist must wear a palette colour, so an off-palette series
    fails too, and an axes needs a y label or row labels in its ticks. No figure draws a
    note.
    """
    assert_no_note_text(fig)
    assert to_hex(fig.get_facecolor(), keep_alpha=True) == "#ffffffff"
    assert _title(fig)
    assert_short_title(fig)
    positions = [tuple(np.round(ax.get_position().bounds, 6)) for ax in fig.axes]
    assert len(positions) == len(set(positions)), "two axes overlay each other (a twin axis)"
    fig.canvas.draw()
    for ax in fig.axes:
        assert ax.get_xlabel() or ax.get_shared_x_axes().get_siblings(ax) != [ax]
        ticks = [text.get_text() for text in ax.get_yticklabels()]
        assert ax.get_ylabel() or any(ticks), "an axes with neither a y label nor row labels"
        used = set()
        for artist in [*ax.lines, *ax.collections, *ax.patches]:
            for color in np.atleast_2d(_colors(artist)):
                if color.size == 4 and color[3] == 0.0:
                    continue
                hexed = to_hex(color)
                if _is_grey(hexed):
                    continue
                assert hexed in PALETTE, f"{hexed} is not a palette colour"
                used.add(hexed)
        assert len(used) <= _core.MAX_SERIES


def _is_grey(color: str) -> bool:
    import colorsys

    red, green, blue = matplotlib.colors.to_rgb(color)
    return colorsys.rgb_to_hsv(red, green, blue)[1] < 0.2


def _colors(artist: Any) -> Any:
    if hasattr(artist, "get_edgecolor") and hasattr(artist, "get_facecolor"):
        faces = np.atleast_2d(artist.get_facecolor())
        edges = np.atleast_2d(artist.get_edgecolor())
        return np.vstack([faces, edges]) if faces.size and edges.size else faces
    return np.atleast_2d(matplotlib.colors.to_rgba(artist.get_color()))


def test_the_style_checks_reject_a_cluttered_figure() -> None:
    """Control: eight coloured series, off-palette colours and no y label must fail."""
    fig, ax = plt.subplots()
    ax.set_title("A title", loc="left")
    ax.set_xlabel("x")
    ax.set_yticks([])
    for index in range(8):
        ax.plot([0, 1], [index, index + 1], color=f"C{index}")
    with pytest.raises(AssertionError):
        _style_checks(fig)


def test_every_figure_passes_the_style_checks(binary_fit: Any, missing_fit: Any) -> None:
    figures = [
        binary_fit.plot(),
        overlap(binary_fit),
        binary_fit.diagnostics.truncation_curve().plot(),
        marginal_sensitivity_bounds(binary_fit, gamma=(1.0, 2.0)).plot(),
        missing_fit.sensitivity.missingness(gamma=[-1.0, 0.0, 1.0]).plot(),
    ]
    for fig in figures:
        _style_checks(fig)


def test_every_figure_renders(binary_fit: Any, tmp_path: Path) -> None:
    for index, fig in enumerate(
        [
            binary_fit.plot(),
            overlap(binary_fit),
            binary_fit.diagnostics.truncation_curve().plot(estimand="rr"),
        ]
    ):
        path = tmp_path / f"figure-{index}.png"
        fig.savefig(path, dpi=60)
        assert path.stat().st_size > 1000


def test_importing_cleverly_does_not_import_matplotlib() -> None:
    code = (
        "import sys, cleverly, cleverly.plotting, cleverly.estimators; "
        "assert 'matplotlib' not in sys.modules, sorted(m for m in sys.modules if 'matplot' in m)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_missing_matplotlib_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def blocked(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("matplotlib"):
            raise ImportError("no matplotlib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match=r'pip install "cleverly\[plot\]"'):
        _core.pyplot()


# ----------------------------------------------------------------------- forest (V1)


def test_forest_draws_each_estimate_and_its_interval(binary_fit: Any) -> None:
    fig = binary_fit.plot()
    scales = {"difference": ["ate", "att", "atc"], "ratio": ["rr", "or"], "level": ["ey1", "ey0"]}
    assert len(fig.axes) == 3
    for ax, (scale, names) in zip(fig.axes, scales.items(), strict=True):
        labels = [text.get_text() for text in ax.get_yticklabels()]
        assert labels == names
        points = sorted(_points(ax), key=lambda point: -point[1])
        segments = sorted(_interval_segments(ax), key=lambda seg: -seg[0, 1])
        for name, (x, _), segment in zip(names, points, segments, strict=True):
            estimate = binary_fit[name]
            assert x == pytest.approx(estimate.psi)
            assert segment[:, 0] == pytest.approx(estimate.ci)
        if scale == "ratio":
            assert ax.get_xscale() == "log"
        nulls = [line.get_xdata()[0] for line in _lines(ax, "cleverly-reference")]
        assert nulls == ({"difference": [0.0], "ratio": [1.0], "level": []}[scale])
    assert _title(fig) == forest_module.TITLE
    assert _key(fig.axes[0]) == ["95% confidence interval", "No effect"]


def test_forest_numbers_match_the_printed_summary(binary_fit: Any) -> None:
    fig = binary_fit.plot()
    ax = fig.axes[0]
    texts = [text.get_text() for text in ax.texts if text.get_gid() == "cleverly-number"]
    summary = binary_fit.summary()
    (table,) = [table for table in summary.tables if table.kind == "estimates"]
    printed = {
        row[0].text: (row[1].text, row[3].text)
        for row in table.rows
        if row[0].text in ("ate", "att", "atc")
    }
    for name, text in zip(("ate", "att", "atc"), texts, strict=True):
        estimate, interval = printed[name]
        assert text == f"{estimate} {interval}"


def test_forest_compares_several_fits_with_their_labels(binary_fit: Any, missing_fit: Any) -> None:
    fig = forest(binary_fit["ate"], missing_fit, labels=["Binary", "Missing outcomes"])
    labels = [text.get_text() for text in fig.axes[0].get_yticklabels()]
    assert labels == ["Binary", "Missing outcomes"]


def test_forest_refuses_a_callers_axes_for_several_scales(binary_fit: Any) -> None:
    _, ax = plt.subplots()
    with pytest.raises(ValueError, match="more than one scale"):
        forest(binary_fit, ax=ax)
    fig = forest(binary_fit["ate"], ax=ax)
    assert fig is ax.figure


def _assert_no_interval_with_reason(fig: Any, status: str) -> None:
    """Honesty rule 1: a withheld estimate draws its point, no interval, and the reason."""
    ax = fig.axes[0]
    assert len(_points(ax)) == 1
    assert _interval_segments(ax) == []
    assert status_record(status).plain_reason.split(".")[0] in _core.note_text(fig)


def test_withheld_inference_draws_the_point_and_the_reason(binary_fit: Any) -> None:
    withheld = dataclasses.replace(binary_fit["ate"], inference=WITHHELD)
    _assert_no_interval_with_reason(forest(withheld), WITHHELD)


def test_withheld_inference_control_fails_when_the_interval_is_drawn(
    binary_fit: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    withheld = dataclasses.replace(binary_fit["ate"], inference=WITHHELD)
    real = _core.estimate_interval

    def ignores_status(estimate: Any, name: str | None = None) -> Any:
        found = real(estimate, name)
        low, high = estimate.plugin_interval
        return dataclasses.replace(found, lower=low, upper=high, notes=())

    monkeypatch.setattr(_core, "estimate_interval", ignores_status)
    with pytest.raises(AssertionError):
        _assert_no_interval_with_reason(forest(withheld), WITHHELD)


@pytest.fixture(scope="module")
def continuous_fit() -> Any:
    rng = np.random.default_rng(2)
    w = rng.normal(size=300)
    a = rng.binomial(1, 1 / (1 + np.exp(-w)))
    y = w + a + rng.lognormal(0.0, 1.2, size=300)
    frame = pd.DataFrame({"W": w, "A": a, "Y": y})
    return (
        TMLE(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(),
            cross_fit=False,
            random_state=0,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W"])
        .single()
    )


def _assert_asymmetric_bar(estimate: Any) -> None:
    """The bar is the shipped interval, and its two arms differ in length."""
    ax = forest(estimate).axes[0]
    (segment,) = _interval_segments(ax)
    low, high = sorted(segment[:, 0])
    assert (low, high) == pytest.approx(estimate.ci, rel=1e-12)
    below, above = estimate.psi - low, high - estimate.psi
    assert below > 0.0 and above > 0.0
    assert abs(above - below) > 0.02 * (above + below)


@pytest.mark.parametrize(
    ("fixture", "name", "rule"),
    [
        ("binary_fit", "ey1", "range_logit"),
        ("binary_fit", "ate", "mover"),
        ("continuous_fit", "ate", "skew_t"),
    ],
)
def test_an_asymmetric_interval_draws_an_asymmetric_bar(
    fixture: str, name: str, rule: str, request: pytest.FixtureRequest
) -> None:
    estimate = request.getfixturevalue(fixture)[name]
    assert estimate.interval_scale == rule
    _assert_asymmetric_bar(estimate)


def test_a_symmetric_bar_fails_the_asymmetry_witness(
    continuous_fit: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation control: a bar rebuilt as ``psi +/- z * se`` is rejected."""
    real = _core.estimate_interval

    def rebuilt(estimate: Any, name: str | None = None) -> Any:
        found = real(estimate, name)
        half = 1.959963984540054 * estimate.std_error
        return dataclasses.replace(found, lower=estimate.psi - half, upper=estimate.psi + half)

    monkeypatch.setattr(_core, "estimate_interval", rebuilt)
    with pytest.raises(AssertionError):
        _assert_asymmetric_bar(continuous_fit["ate"])


@pytest.mark.parametrize("status", ["constant_arm_cell_plugin", "out_of_range_plugin"])
def test_a_withheld_interval_status_draws_no_bar(binary_fit: Any, status: str) -> None:
    withheld = dataclasses.replace(binary_fit["ey1"], inference=status)
    assert all(np.isnan(withheld.plugin_interval))
    _assert_no_interval_with_reason(forest(withheld), status)


def test_an_experimental_estimate_says_so(binary_fit: Any) -> None:
    support = Support("experimental", "No registered study covers this configuration.")
    experimental = dataclasses.replace(binary_fit["ate"], support=support)
    assert "experimental" in _core.note_text(forest(experimental))
    # Control: the supported estimate carries no such note.
    supported = dataclasses.replace(binary_fit["ate"], support=Support("supported", "Studied."))
    assert "experimental" not in _core.note_text(forest(supported))


# ----------------------------------------------------------------------- sensitivity (V12)


def test_truncation_plot_draws_the_curve_and_the_capped_share(binary_fit: Any) -> None:
    curve = binary_fit.diagnostics.truncation_curve()
    frame = curve.to_frame()
    rows = frame[frame["estimand"] == "ate"]
    fig = curve.plot()
    top, bottom = fig.axes
    points = sorted(_points(top))
    assert [x for x, _ in points] == pytest.approx(sorted(rows["bound"]))
    assert [y for _, y in points] == pytest.approx(list(rows.sort_values("bound")["psi"]))
    segments = sorted(_interval_segments(top), key=lambda seg: seg[0, 0])
    assert [seg[0, 1] for seg in segments] == pytest.approx(
        list(rows.sort_values("bound")["ci_lower"])
    )
    (capped,) = [line for line in bottom.lines if line.get_gid() == "cleverly-capped"]
    assert capped.get_ydata() == pytest.approx(100.0 * rows["truncated_fraction"].to_numpy())
    (fitted,) = _lines(top, "cleverly-marker")
    assert fitted.get_xdata()[0] == pytest.approx(rows["fitted_lower_bound"].iloc[0])
    assert [line.get_ydata()[0] for line in _lines(top, "cleverly-reference")] == [0.0]
    assert top.get_xscale() == "log"


def test_truncation_plot_of_a_ratio_uses_a_log_axis_and_null_one(binary_fit: Any) -> None:
    fig = binary_fit.diagnostics.truncation_curve().plot(estimand="rr")
    top = fig.axes[0]
    assert top.get_yscale() == "log"
    assert [line.get_ydata()[0] for line in _lines(top, "cleverly-reference")] == [1.0]


def test_truncation_plot_refuses_a_curve_without_its_scale(binary_fit: Any) -> None:
    curve = binary_fit.diagnostics.truncation_curve()
    bare = dataclasses.replace(curve, frame=curve.to_frame().drop(columns=["scale"]))
    with pytest.raises(ValueError, match="does not record its scale"):
        bare.plot()


def test_truncation_plot_labels_the_fits_interval_level() -> None:
    rng = np.random.default_rng(3)
    w = rng.normal(size=400)
    a = rng.binomial(1, 1 / (1 + np.exp(-w)))
    y = rng.binomial(1, 1 / (1 + np.exp(-(w + a - 0.5))))
    fit = (
        TMLE(
            outcome_learner=LogisticRegression(),
            treatment_learner=LogisticRegression(),
            alpha_sig=0.1,
            random_state=0,
        )
        .fit(pd.DataFrame({"W": w, "A": a, "Y": y}), outcome="Y", treatment="A", covariates=["W"])
        .single()
    )
    curve = fit.diagnostics.truncation_curve()
    assert "90% confidence interval" in _key(curve.plot().axes[0])
    # Control: the label follows the recorded level, so a curve claiming 95% says 95%.
    frame = curve.to_frame().assign(level=0.95)
    keyed = _key(dataclasses.replace(curve, frame=frame).plot().axes[0])
    assert "95% confidence interval" in keyed
    assert "90% confidence interval" not in keyed


def test_truncation_plot_of_an_experimental_fit_says_so(binary_fit: Any) -> None:
    from tests.unit._capability_sweep_support import fit_ctmle_oat_shared

    curve = fit_ctmle_oat_shared().diagnostics.truncation_curve()
    assert "ate: experimental." in _core.note_text(curve.plot())
    # Control: a supported fit's curve carries no such note.
    assert "experimental" not in _core.note_text(binary_fit.diagnostics.truncation_curve().plot())


def test_truncation_curve_from_the_combined_report_plots(binary_fit: Any) -> None:
    combined = binary_fit.diagnostics.run_all(include_retargets=True)
    fig = combined["truncation_curve"].report.plot()
    direct = binary_fit.diagnostics.truncation_curve().plot()
    assert np.asarray(sorted(_points(fig.axes[0]))) == pytest.approx(
        np.asarray(sorted(_points(direct.axes[0])))
    )


def test_truncation_plot_draws_no_interval_without_inference(
    binary_fit: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    curve = binary_fit.diagnostics.truncation_curve()
    frame = curve.to_frame().rename(
        columns={
            "std_err": "plugin_std_err",
            "ci_lower": "plugin_interval_lower",
            "ci_upper": "plugin_interval_upper",
        }
    )
    fig = dataclasses.replace(curve, frame=frame).plot()
    assert _interval_segments(fig.axes[0]) == []
    assert "No interval is drawn" in _core.note_text(fig)
    # Control: the curve with its interval columns draws one interval per limit.
    assert len(_interval_segments(curve.plot().axes[0])) == int((frame["estimand"] == "ate").sum())


def test_marginal_plot_draws_the_range_and_its_interval(binary_fit: Any) -> None:
    bounds = marginal_sensitivity_bounds(binary_fit, gamma=(1.0, 1.5, 3.0))
    fig = bounds.plot()
    ax = fig.axes[0]
    ranges = [
        np.asarray(segment)
        for collection in ax.collections
        if collection.get_gid() == "cleverly-range"
        for segment in collection.get_segments()
    ]
    for row, segment in zip(bounds.rows, ranges, strict=True):
        assert segment[:, 0] == pytest.approx([row.gamma, row.gamma])
        assert segment[:, 1] == pytest.approx([row.lower, row.upper])
    for row, segment in zip(bounds.rows, _interval_segments(ax), strict=True):
        assert segment[:, 1] == pytest.approx([row.lower_limit, row.upper_limit])
    assert bounds.breakdown is not None
    assert bounds.limit_descriptive
    assert f"gamma = {bounds.breakdown:g}" in (_core.subtitle_text(ax) or "")
    assert "descriptive" in _core.note_text(fig)


def _odds_bounds(rows: list[tuple[float, float, float, float, float]]) -> Any:
    """Bounds of a difference over ``(gamma, lower, upper, lower_limit, upper_limit)`` rows."""
    from cleverly.sensitivity.marginal import MarginalSensitivityBounds, MarginalSensitivityRow

    return MarginalSensitivityBounds(
        rows=tuple(
            MarginalSensitivityRow(g, "ate", lo, hi, 0.01, 0.01, lo_lim, hi_lim)
            for g, lo, hi, lo_lim, hi_lim in rows
        ),
        estimand="ate",
        level=0.95,
        null=0.0,
        outcome_kind="binary",
    )


def test_marginal_subtitle_names_where_the_range_and_where_its_interval_include_zero() -> None:
    """The point-treatment page's case: the range never includes zero, the interval does."""
    bounds = _odds_bounds(
        [
            (1.0, 0.10, 0.10, 0.06, 0.14),
            (2.0, 0.06, 0.14, 0.03, 0.17),
            (3.0, 0.04, 0.16, 0.01, 0.19),
            (4.0, 0.03, 0.17, -0.01, 0.20),
        ]
    )
    assert bounds.breakdown is None
    assert bounds.limit_breakdown == 4.0
    subtitle = _core.subtitle_text(bounds.plot().axes[0]) or ""
    flat = " ".join(subtitle.split())
    assert "range clear of no effect up to gamma = 4" in flat
    assert "its 95% interval reaches it at gamma = 4" in flat
    assert "no gamma on this grid reaches no effect" not in flat
    # Control: when the range itself includes zero, both breakdowns are named with their gamma.
    reached = _odds_bounds(
        [
            (1.0, 0.10, 0.10, 0.06, 0.14),
            (2.0, 0.02, 0.18, -0.01, 0.21),
            (3.0, -0.02, 0.22, -0.05, 0.25),
        ]
    )
    flat = " ".join((_core.subtitle_text(reached.plot().axes[0]) or "").split())
    assert "range reaches no effect at gamma = 3, its 95% interval at gamma = 2" in flat
    # Control: neither reaches zero.
    never = _odds_bounds([(1.0, 0.10, 0.10, 0.06, 0.14), (2.0, 0.06, 0.14, 0.03, 0.17)])
    flat = " ".join((_core.subtitle_text(never.plot().axes[0]) or "").split())
    assert "range and its 95% interval clear of no effect up to gamma = 2" in flat


def test_marginal_plot_control_fails_without_the_warnings(
    binary_fit: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cleverly.plotting import _sensitivity

    bounds = marginal_sensitivity_bounds(binary_fit, gamma=(1.0, 1.5, 3.0))
    monkeypatch.setattr(_sensitivity, "_warnings", lambda obj: [])
    assert "descriptive" not in _core.note_text(bounds.plot())


def test_outcome_label_names_the_effect_on_the_sensitivity_axes(
    binary_fit: Any, missing_fit: Any
) -> None:
    sweep = binary_fit.diagnostics.truncation_curve().plot(outcome_label="Readmission")
    assert sweep.axes[0].get_ylabel() == "Average effect on readmission"
    ratio = binary_fit.diagnostics.truncation_curve().plot(estimand="rr", outcome_label="Y")
    assert ratio.axes[0].get_ylabel() == "Risk ratio of Y"
    bounds = marginal_sensitivity_bounds(binary_fit, gamma=(1.0, 2.0))
    assert bounds.plot(outcome_label="Readmission").axes[0].get_ylabel() == (
        "Range of the average effect on readmission"
    )
    tilt = missing_fit.sensitivity.missingness(gamma=[-1.0, 0.0, 1.0])
    assert tilt.plot(outcome_label="Score").axes[0].get_ylabel().endswith(" on score")
    assert _core.estimand_axis_label("ey1", "Order share") == "Order share treated"
    assert _core.estimand_axis_label("ate", None) == "Average effect"
    # Control: a qualified alias keeps its name, since the words alone would not say which.
    assert _core.estimand_words("ate[high vs low]") == "Average effect (ate[high vs low])"


def test_units_are_added_to_the_sensitivity_axes(binary_fit: Any, missing_fit: Any) -> None:
    sweep = binary_fit.diagnostics.truncation_curve().plot(outcome_label="Readmission", units="%")
    assert sweep.axes[0].get_ylabel() == "Average effect on readmission (%)"
    bounds = marginal_sensitivity_bounds(binary_fit, gamma=(1.0, 2.0))
    assert bounds.plot(units="points").axes[0].get_ylabel().endswith(" (points)")
    tilt = missing_fit.sensitivity.missingness(gamma=[-1.0, 0.0, 1.0])
    assert tilt.plot(units="points").axes[0].get_ylabel().endswith(" (points)")
    # Control: without units nothing is appended.
    assert tilt.plot().axes[0].get_ylabel() == "Average effect"


def test_the_marginal_null_line_sits_clear_of_the_bottom_edge() -> None:
    bounds = _odds_bounds([(1.0, 0.4, 0.4, 0.3, 0.5), (2.0, 0.05, 0.9, 0.0, 1.0)])
    ax = bounds.plot().axes[0]
    low, high = ax.get_ylim()
    assert low < 0.0 < high
    assert 0.0 - low > 0.02 * (high - low), "the null line is not on the spine"


def test_the_truncation_cap_is_on_the_axis_and_the_limits_used_in_the_subtitle(
    binary_fit: Any,
) -> None:
    fig = binary_fit.diagnostics.truncation_curve().plot()
    (raw,) = [t.get_text() for t in fig.axes[0].texts if t.get_gid() == _core.GID_SUBTITLE]
    assert "\n" not in raw, "the subtitle wraps"
    assert "the fit used [" in raw
    assert "[limit, 1 - limit]" in fig.axes[-1].get_xlabel()


def test_missingness_plot_marks_the_main_analysis(missing_fit: Any) -> None:
    tilt = missing_fit.sensitivity.missingness(gamma=[-1.0, 0.0, 1.0])
    fig = tilt.plot()
    ax = fig.axes[0]
    drawn = np.asarray(sorted(_points(ax)))
    expected = np.asarray(sorted(zip(tilt.column("gamma"), tilt.column("psi"), strict=True)))
    assert drawn == pytest.approx(expected)
    (main,) = _lines(ax, "cleverly-marker")
    assert main.get_xdata()[0] == 0.0
    # CDR-001: the ATE's tilted estimate is re-targeted with its own influence curve, so
    # every point carries its own interval and is filled.
    assert sorted(segment[0, 0] for segment in _interval_segments(ax)) == [-1.0, 0.0, 1.0]
    assert all(_filled(ax))
    assert "Hollow points" not in _core.note_text(fig)


def test_missingness_plot_leaves_a_regime_mean_hollow_off_mar() -> None:
    from cleverly.interventions import Rule, Static

    frame, _ = make_missing_outcome(n=400, seed=61)
    rule = Rule(lambda w: np.where(w["W1"] >= 0, 1, 0), name="rule", rule_kind="known")
    fit = (
        TMLE(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            missingness_learner=LogisticRegression(max_iter=1000),
            cross_fit=False,
            interventions=[Static(0, name="ref"), rule],
            estimands=("ey_regime",),
            random_state=4,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2", "W3"], delta="Delta")
        .single()
    )
    tilt = fit.sensitivity.missingness(gamma=[-1.0, 0.0, 1.0])
    name = "ey_regime[rule]"
    fig = tilt.plot(estimand=name)
    ax = fig.axes[0]
    # A regime mean keeps the plug-in spread off gamma = 0: only the main analysis has an
    # interval, and the moved points are hollow.
    (segment,) = _interval_segments(ax)
    assert segment[0, 0] == 0.0
    assert _filled(ax) == [False, True, False]
    assert "Hollow points have no interval" in _core.note_text(fig)


def test_missingness_plot_of_a_ratio_uses_a_log_axis_and_null_one() -> None:
    from tests.unit._capability_sweep_support import fit_rr_missing

    fit = fit_rr_missing()
    ratio = next(name for name in fit.estimates if str(fit[name].scale) == "ratio")
    fig = fit.sensitivity.missingness(gamma=[-0.5, 0.0, 0.5]).plot(estimand=ratio)
    ax = fig.axes[0]
    assert ax.get_yscale() == "log"
    assert [line.get_ydata()[0] for line in _lines(ax, "cleverly-reference")] == [1.0]


def test_missingness_plot_of_a_difference_draws_null_zero(missing_fit: Any) -> None:
    fig = missing_fit.sensitivity.missingness(gamma=[-1.0, 0.0, 1.0]).plot()
    ax = fig.axes[0]
    assert ax.get_yscale() == "linear"
    assert [line.get_ydata()[0] for line in _lines(ax, "cleverly-reference")] == [0.0]


def _filled(ax: Any) -> list[bool]:
    return [
        to_hex(line.get_markerfacecolor()) != _core.FACE
        for line in ax.lines
        if line.get_gid() == "cleverly-point"
    ]


def test_missingness_plot_draws_no_interval_when_the_tilt_withholds_it(missing_fit: Any) -> None:
    tilt = missing_fit.sensitivity.missingness(gamma=[-1.0, 0.0, 1.0])
    withheld = dataclasses.replace(tilt, statuses=(("ate", WITHHELD),))
    fig = withheld.plot()
    assert _interval_segments(fig.axes[0]) == []
    assert status_record(WITHHELD).plain_reason.split(".")[0] in _core.note_text(fig)
    assert not any(_filled(fig.axes[0]))
    # Control: the tilt as computed draws every interval with a filled point.
    computed = tilt.plot().axes[0]
    assert len(_interval_segments(computed)) == 3
    assert all(_filled(computed))


def test_sensitivity_refuses_other_objects() -> None:
    with pytest.raises(TypeError, match="sensitivity"):
        sensitivity(object())


# ----------------------------------------------------------------------- overlap (V9)


def test_overlap_mirrors_the_unbounded_propensity_by_arm(binary_fit: Any) -> None:
    fig = overlap(binary_fit)
    ax = fig.axes[0]
    above = [patch for patch in ax.patches if patch.get_gid() == "cleverly-overlap-above"]
    below = [patch for patch in ax.patches if patch.get_gid() == "cleverly-overlap-below"]
    treated = np.asarray(binary_fit.data.treatment) == 1.0
    raw = binary_fit.repeats[0].nuisance.propensity.arm(1.0)
    counts, _ = np.histogram(raw[treated], bins=np.linspace(0.0, 1.0, 41))
    assert [patch.get_height() for patch in above] == pytest.approx(counts / treated.size)
    assert sum(patch.get_height() for patch in below) == pytest.approx(-(~treated).mean())
    bounds = binary_fit.config.g_bounds
    conditional = binary_fit.config.g_bounds_conditional
    markers = sorted(line.get_xdata()[0] for line in _lines(ax, "cleverly-marker"))
    assert markers == pytest.approx(sorted([*bounds, *conditional]))
    assert _key(ax)[-2:] == [
        "Probability limits used for every estimate but the effect among the treated and "
        "the untreated",
        "Probability limits used for the effect among the treated and the untreated",
    ]
    shares = [text.get_text() for text in ax.texts if text.get_gid() == "cleverly-tail-share"]
    below_share = float(np.mean(raw < bounds[0]))
    assert shares[0] == f"{100 * below_share:.1f}% below"
    assert ax.get_xlabel() == "Predicted probability of A = 1"


def test_overlap_refuses_a_continuous_treatment() -> None:
    from cleverly.exceptions import DataError
    from tests.unit._capability_sweep_support import fit_shift

    with pytest.raises(DataError, match="continuous"):
        overlap(fit_shift())


def test_overlap_draws_one_panel_per_arm_on_a_shared_scale() -> None:
    from tests.unit._capability_sweep_support import fit_multi_arm

    fit = fit_multi_arm()
    fig = overlap(fit)
    arms = fit.repeats[0].nuisance.propensity.arms
    assert len(fig.axes) == len(arms)
    assert len({tuple(np.round(ax.get_ylim(), 12)) for ax in fig.axes}) == 1
    words = " ".join(text.get_text() for ax in fig.axes for text in ax.texts)
    assert "every other arm" in words
    assert "Each panel compares one arm with the rest" in _core.note_text(fig)


def test_overlap_weights_the_bars_and_says_so() -> None:
    from tests.unit._capability_sweep_support import fit_weighted

    fit = fit_weighted()
    ax = overlap(fit).axes[0]
    weights = np.asarray(fit.data.weights, dtype=float)
    treated = np.asarray(fit.data.treatment) == 1.0
    raw = fit.repeats[0].nuisance.propensity.arm(1.0)
    edges = np.linspace(0.0, 1.0, 41)
    counts, _ = np.histogram(raw[treated], bins=edges, weights=weights[treated])
    above = [p.get_height() for p in ax.patches if p.get_gid() == "cleverly-overlap-above"]
    assert above == pytest.approx(counts / weights.sum())
    assert "weighted by the data's weights" in (_core.subtitle_text(ax) or "")
    # Control: the unweighted shares differ, so the check can see the weights.
    plain, _ = np.histogram(raw[treated], bins=edges)
    assert above != pytest.approx(plain / treated.size)


def test_overlap_of_an_att_fit_draws_only_the_conditional_limits(binary_fit: Any) -> None:
    from cleverly import ATT
    from cleverly.datasets import make_linear_ate

    frame, _ = make_linear_ate(n=300, seed=2)
    design = PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4"))
    fit = (
        CausalStudy(frame, design=design)
        .identify(ATT())
        .estimate(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            cross_fit=False,
            random_state=0,
        )
    )
    ax = overlap(fit).axes[0]
    markers = sorted(line.get_xdata()[0] for line in _lines(ax, "cleverly-marker"))
    assert markers == pytest.approx(sorted(fit.config.g_bounds_conditional))
    assert any("among the treated" in entry or "the fit used" in entry for entry in _key(ax))


def test_overlap_says_when_the_predictions_are_in_sample(binary_fit: Any, missing_fit: Any) -> None:
    assert "Out-of-fold" in (_core.subtitle_text(overlap(binary_fit).axes[0]) or "")
    assert "In-sample" in (_core.subtitle_text(overlap(missing_fit).axes[0]) or "")


def test_overlap_draws_a_longitudinal_fit_per_time_point() -> None:
    from tests.unit._capability_sweep_support import fit_ltmle

    result = fit_ltmle()
    fig = overlap(result)
    titles = [axis.get_title(loc="left") for axis in fig.axes]
    times = range(1, result.data.n_times + 1)
    assert titles == [f"{label}, Time point {t}" for label in ("always", "never") for t in times]
    _style_checks(fig)


# ----------------------------------------------------------------------- forest, more cases


def test_forest_draws_the_attributable_fraction_against_zero() -> None:
    from tests.unit._capability_sweep_support import fit_attributable_missing

    fig = forest(fit_attributable_missing())
    for ax in fig.axes:
        assert [line.get_xdata()[0] for line in _lines(ax, "cleverly-reference")] == [0.0]
    assert [text.get_text() for text in fig.axes[-1].get_yticklabels()] == ["paf"]


def test_forest_draws_a_longitudinal_fit() -> None:
    from tests.unit._capability_sweep_support import fit_ltmle

    fit = fit_ltmle()
    ax = forest(fit).axes[0]
    labels = [text.get_text() for text in ax.get_yticklabels()]
    assert labels == list(fit.estimates)
    assert len(_points(ax)) == len(labels)


def test_forest_refuses_mixed_levels_in_one_panel(binary_fit: Any) -> None:
    narrow = dataclasses.replace(binary_fit["ate"], alpha=0.1)
    with pytest.raises(ValueError, match="different confidence levels"):
        forest(binary_fit["ate"], narrow)
    assert "90% confidence interval" in _key(forest(narrow).axes[0])


def test_forest_subtitle_promises_no_table() -> None:
    from tests.unit._capability_sweep_support import fit_multi_arm

    subtitle = _core.subtitle_text(forest(fit_multi_arm()).axes[0]) or ""
    assert subtitle
    assert "defined below" not in subtitle


def test_forest_names_shared_notes_once() -> None:
    from tests.unit._capability_sweep_support import fit_weighted

    text = _core.note_text(forest(fit_weighted()))
    assert text.count("experimental") == 1
    assert "ate, att, atc" in text


def test_forest_numbers_share_the_summarys_decimals_in_every_panel(binary_fit: Any) -> None:
    fig = binary_fit.plot()
    (table,) = [table for table in binary_fit.summary().tables if table.kind == "estimates"]
    printed = {row[0].text: f"{row[1].text} {row[3].text}" for row in table.rows}
    for ax in fig.axes:
        names = [text.get_text() for text in ax.get_yticklabels()]
        texts = [text.get_text() for text in ax.texts if text.get_gid() == "cleverly-number"]
        for name, text in zip(names, texts, strict=True):
            assert text == printed[name]


def test_forest_skips_the_null_a_transform_cannot_reach(binary_fit: Any) -> None:
    class LogOnly:
        name = "log"

        @staticmethod
        def forward(value: float) -> float:
            with np.errstate(divide="ignore"):
                return float(np.log(value))

        @staticmethod
        def inverse(value: float) -> float:
            return float(np.exp(value))

    estimate = binary_fit["ey1"]
    logged = dataclasses.replace(estimate, scale="difference", transform=LogOnly())
    assert _lines(forest(logged, include_null=True).axes[0], "cleverly-reference") == []
    # Control: without the transform the difference null is drawn. A mean near 0.65 lies far
    # from 0, so the axis reaches the null only when asked to.
    plain = dataclasses.replace(estimate, scale="difference")
    assert len(_lines(forest(plain, include_null=True).axes[0], "cleverly-reference")) == 1


def test_notes_are_recorded_and_never_drawn(binary_fit: Any) -> None:
    withheld = dataclasses.replace(binary_fit["ate"], inference=WITHHELD)
    figures = (
        forest(withheld),
        overlap(binary_fit),
        binary_fit.diagnostics.truncation_curve().plot(),
    )
    for fig in figures:
        assert notes(fig), "the plot recorded no notes"
        assert_no_note_text(fig)


def test_the_note_check_rejects_a_drawn_note(binary_fit: Any) -> None:
    """Control: a recorded note drawn under the axes must fail the check."""
    fig = forest(dataclasses.replace(binary_fit["ate"], inference=WITHHELD))
    fig.text(0.01, 0.01, notes(fig)[0], fontsize=8)
    with pytest.raises(AssertionError, match="draws its note"):
        assert_no_note_text(fig)


def test_notes_drop_repeats_and_blank_lines() -> None:
    fig, ax = plt.subplots()
    _core.add_notes(ax, ["first", "", "second"])
    _core.add_notes(fig, ["second", "third"])
    assert notes(fig) == ("first", "second", "third")
    assert _core.note_text(fig) == "first second third"
    assert notes(plt.figure()) == ()


def _nearest_row_reason(qualifiers: str) -> Support:
    """An experimental support whose nearest validated row differs on ``qualifiers``.

    The text is the one :func:`cleverly._support.support_for` writes.
    """
    return Support(
        "experimental",
        "No registered study covers this configuration. Nearest row: G46.tmle.categorical_k3 "
        f"(long-horizon-categorical-ltmle), differing on qualifiers ({qualifiers}).",
    )


def test_forest_note_names_known_mechanisms_in_words_not_column_ids(binary_fit: Any) -> None:
    """A longitudinal fit's qualifiers are column names; ids such as A1 must not leave gaps."""
    support = _nearest_row_reason("A1 and A2 and A3 and C1 and C2 and offer@1 and offer@2")
    note = _core.note_text(forest(dataclasses.replace(binary_fit["ate"], support=support)))
    assert "and and" not in note
    assert "in and" not in note
    assert "offer@1" not in note
    assert "differs in which treatment and censoring mechanisms are declared known." in note
    # Control: the qualifiers of a point-treatment fit keep their own words.
    support = _nearest_row_reason("known_mechanism and strata")
    note = _core.note_text(forest(dataclasses.replace(binary_fit["ate"], support=support)))
    assert "differs in a known treatment mechanism and baseline strata." in note


def test_forest_subtitle_keeps_a_column_name_as_written() -> None:
    """The summary's question can open with the outcome column, which is not capitalised."""
    from cleverly.datasets import make_longitudinal
    from cleverly.longitudinal import LTMLE

    frame, _ = make_longitudinal(n=200, seed=0)
    frame = frame.rename(columns={"Y": "full_price"})
    fit = LTMLE(
        {"always": 1, "never": 0},
        n_folds=1,
        random_state=0,
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=500),
        censoring_learner=LogisticRegression(max_iter=500),
    ).fit(
        frame,
        outcome="full_price",
        treatment=["A1", "A2"],
        baseline=["W1", "W2"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
    )
    assert fit.summary().facts["Question"].startswith("full_price ")
    assert _core.subtitle_text(forest(fit).axes[0]) == "full_price under each plan"
    # Control: a question that opens with an English word still starts with a capital.
    from tests.unit._capability_sweep_support import fit_multi_arm

    multi = fit_multi_arm()
    question = multi.summary().facts["Question"]
    assert question[:1].islower()
    subtitle = _core.subtitle_text(forest(multi).axes[0]) or ""
    assert subtitle[:1].isupper()


# ----------------------------------------------------------------------- truth=


def _truth_marks(ax: Any) -> list[tuple[float, float]]:
    return [tuple(line.get_xydata()[0]) for line in _lines(ax, _core.GID_TRUTH)]


def test_forest_truth_draws_a_cross_on_each_row_and_names_it_in_the_key(binary_fit: Any) -> None:
    names = list(binary_fit.estimates)
    truth = {name: 0.1 * (i + 1) for i, name in enumerate(names)}
    fig = forest(binary_fit, truth=truth)
    marks = [mark for ax in fig.axes for mark in _truth_marks(ax)]
    rows = {
        text.get_text(): position
        for ax in fig.axes
        for text, position in zip(ax.get_yticklabels(), ax.get_yticks(), strict=True)
    }
    assert sorted(marks) == sorted((truth[name], rows[name]) for name in names)
    assert _core.TRUTH_LABEL in _core.key_labels(fig)
    # Its own style: an ink cross, not the grey dashed no-effect line and not a palette colour.
    (mark, *_) = [line for ax in fig.axes for line in _lines(ax, _core.GID_TRUTH)]
    (null, *_) = [line for ax in fig.axes for line in _lines(ax, _core.GID_REFERENCE)]
    assert mark.get_marker() == "x" and mark.get_linestyle() == "None"
    assert to_hex(mark.get_color()) == _core.INK != to_hex(null.get_color())
    assert to_hex(mark.get_color()) not in PALETTE
    _style_checks(fig)


def test_forest_key_lists_the_true_value_only_when_truth_is_given(binary_fit: Any) -> None:
    """Control: the entry and the crosses come from truth=, not from every forest."""
    fig = forest(binary_fit)
    assert _core.TRUTH_LABEL not in _core.key_labels(fig)
    assert not any(_lines(ax, _core.GID_TRUTH) for ax in fig.axes)
    fig = forest(binary_fit["ate"], truth=0.2)
    assert _core.TRUTH_LABEL in _core.key_labels(fig)
    assert _truth_marks(fig.axes[0]) == [(0.2, 0.0)]


def test_forest_scalar_truth_needs_one_estimand_and_a_mapping_its_names(binary_fit: Any) -> None:
    from cleverly._refusal import RefusalReason
    from cleverly.exceptions import CapabilityError

    with pytest.raises(CapabilityError, match="one number") as refused:
        forest(binary_fit, truth=0.2)
    assert refused.value.refusal.reason is RefusalReason.INVALID_DECLARATION
    with pytest.raises(CapabilityError, match="no row of this plot"):
        forest(binary_fit, truth={"atee": 0.2})
    with pytest.raises(CapabilityError, match="not a finite number"):
        forest(binary_fit["ate"], truth=float("nan"))
    # Two fits of one estimand share the scalar; a mapping leaves unnamed rows bare.
    fig = forest(binary_fit["ate"], binary_fit["ate"], labels=["a", "b"], truth=0.2)
    assert len(_truth_marks(fig.axes[0])) == 2
    fig = forest(binary_fit, truth={"ate": 0.2})
    assert sum(len(_truth_marks(ax)) for ax in fig.axes) == 1


@pytest.mark.parametrize(
    ("treated", "untreated"), [("offered", "not offered"), ("an offer", "no offer")]
)
def test_overlap_arm_labels_stand_alone(binary_fit: Any, treated: str, untreated: str) -> None:
    """A plain label ("offered") and a noun phrase ("an offer") both read as written."""
    fig = overlap(binary_fit, labels={1: treated, 0: untreated})
    ax = fig.axes[0]
    groups = [t.get_text() for t in ax.texts if t.get_gid() == _core.GID_SERIES_LABEL]
    assert groups == [treated, untreated]
    assert ax.get_xlabel() == f"Predicted probability of being in the group: {treated}"
    assert not any("Units with" in text for text in [*groups, ax.get_xlabel()])
    # Control: without labels the arm is named by its column and level.
    plain = overlap(binary_fit).axes[0]
    groups = [t.get_text() for t in plain.texts if t.get_gid() == _core.GID_SERIES_LABEL]
    assert groups == ["Units with A = 1", "Units with A = 0"]
    assert plain.get_xlabel() == "Predicted probability of A = 1"


def _key_column_count(fig: Any, ax: Any) -> int:
    """How many columns the drawn key has, read from where its texts sit."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    key = ax.get_legend()
    return len({round(text.get_window_extent(renderer).x0) for text in key.get_texts()})


def test_the_key_takes_as_many_columns_as_fit_over_its_axes() -> None:
    from cleverly.plotting import curve
    from tests.unit.test_plotting_curves import full

    fig = curve(full())
    top = fig.axes[0]
    assert len(_key(top)) == 3
    assert _key_column_count(fig, top) == 3
    # Control: the same labels over a narrow axes drop to fewer columns.
    _, ax = plt.subplots(figsize=(1.5, 2.0))
    labels = _key(top)
    assert _core._key_columns(ax, labels) == 1
    _, ax = plt.subplots(figsize=(12.0, 2.0))
    assert _core._key_columns(ax, labels) == 3


def test_the_forest_key_keeps_clear_of_its_number_column(binary_fit: Any) -> None:
    fig = forest(binary_fit["ate"], truth=0.2)
    ax = fig.axes[0]
    assert len(_key(ax)) == 3
    assert _key_column_count(fig, ax) == 2


def test_forest_truth_refuses_an_empty_mapping_and_a_boolean(binary_fit: Any) -> None:
    from cleverly.exceptions import CapabilityError

    with pytest.raises(CapabilityError, match="empty mapping"):
        forest(binary_fit, truth={})
    with pytest.raises(CapabilityError, match="boolean"):
        forest(binary_fit["ate"], truth=True)
    with pytest.raises(CapabilityError, match="boolean"):
        forest(binary_fit, truth={"ate": np.bool_(False)})
    # Control: the number 1 is a true value.
    assert _truth_marks(forest(binary_fit["ate"], truth=1).axes[0]) == [(1.0, 0.0)]


def test_forest_names_the_outcome_in_the_words_given(binary_fit: Any) -> None:
    fig = forest(binary_fit, outcome_label="orders in month 3", units="orders")
    difference, ratio, level = fig.axes
    # The words name the whole quantity: no "mean" is inserted before them.
    assert difference.get_xlabel() == "Difference in orders in month 3 (orders)"
    assert level.get_xlabel() == "orders in month 3 (orders)"
    assert ratio.get_xlabel() == "Ratio (log scale)"
    # the control: without the words the axis names the outcome column
    plain = forest(binary_fit).axes[0].get_xlabel()
    assert plain == f"Difference in mean {binary_fit.data.outcome_name}"


# ----------------------------------------------------------------------- forest, the null's reach


def _narrow(estimate: Any, psi: float, *, shrink: float = 0.04) -> Any:
    """``estimate`` moved to ``psi`` with a narrower interval."""
    moved = dataclasses.replace(estimate, psi=psi, variance=estimate.variance * shrink)
    if estimate.scale == "ratio":
        moved = dataclasses.replace(moved, log_psi=float(np.log(psi)))
    return moved


def _xlim(fig: Any) -> tuple[float, float]:
    low, high = fig.axes[0].get_xlim()
    return float(low), float(high)


def test_forest_leaves_a_far_null_off_the_axis_and_says_where(binary_fit: Any) -> None:
    ate = binary_fit["ate"]
    near = [_narrow(ate, psi) for psi in (0.13, 0.15, 0.16, 0.18)]
    fig = forest(*near, labels=["a", "b", "c", "d"])
    ax = fig.axes[0]
    assert _lines(ax, "cleverly-reference") == []
    assert "No effect" not in _key(ax)
    assert _xlim(fig)[0] > 0.0
    note = _core.note_text(fig)
    assert "No effect (0) lies off the axis to the left" in note
    # The mirror image lies off to the right.
    far = forest(*[_narrow(ate, -psi) for psi in (0.13, 0.15)], labels=["a", "b"])
    assert _xlim(far)[1] < 0.0
    assert "off the axis to the right" in _core.note_text(far)


def test_forest_draws_a_null_near_the_intervals(binary_fit: Any) -> None:
    """Control: intervals near 0 keep the line, and say nothing about an axis."""
    fig = forest(_narrow(binary_fit["ate"], 0.05), binary_fit["att"], labels=["a", "b"])
    ax = fig.axes[0]
    assert [line.get_xdata()[0] for line in _lines(ax, "cleverly-reference")] == [0.0]
    low, high = _xlim(fig)
    assert low <= 0.0 <= high
    assert "off the axis" not in _core.note_text(fig)


def test_include_null_overrides_the_automatic_rule(binary_fit: Any) -> None:
    near = [_narrow(binary_fit["ate"], psi) for psi in (0.13, 0.18)]
    shown = forest(*near, labels=["a", "b"], include_null=True)
    assert len(_lines(shown.axes[0], "cleverly-reference")) == 1
    assert _xlim(shown)[0] <= 0.0
    assert "off the axis" not in _core.note_text(shown)
    # include_null=False drops a null the automatic rule would keep, and still says so.
    kept = forest(binary_fit["ate"], include_null=False)
    assert _lines(kept.axes[0], "cleverly-reference") == []
    assert "No effect (0) lies off the axis to the left" in _core.note_text(kept)
    # A null inside the intervals is always drawn: no axis can leave it out.
    across = dataclasses.replace(binary_fit["ate"], psi=0.0)
    assert len(_lines(forest(across, include_null=False).axes[0], "cleverly-reference")) == 1


def test_a_far_ratio_null_is_measured_on_the_log_axis(binary_fit: Any) -> None:
    ratio = _narrow(binary_fit["rr"], 3.0, shrink=0.25)
    fig = forest(ratio)
    assert _lines(fig.axes[0], "cleverly-reference") == []
    assert "No effect (1) lies off the axis to the left" in _core.note_text(fig)
    # Control: the fit's own risk ratio, near 1 on the log axis, keeps the line at 1.
    own = forest(binary_fit["rr"])
    assert [line.get_xdata()[0] for line in _lines(own.axes[0], "cleverly-reference")] == [1.0]


def test_the_off_axis_note_names_its_panel_when_there_are_several(binary_fit: Any) -> None:
    fig = forest(_narrow(binary_fit["ate"], 0.15), _narrow(binary_fit["rr"], 3.0, shrink=0.25))
    note = _core.note_text(fig)
    assert "On the difference panel, no effect (0) lies off the axis to the left" in note
    assert "On the ratio panel, no effect (1) lies off the axis to the left" in note


def test_a_true_value_near_the_null_brings_it_back(binary_fit: Any) -> None:
    """The truth counts as drawn data, so a cross near 0 keeps the line in reach."""
    near = _narrow(binary_fit["ate"], 0.15)
    assert _lines(forest(near).axes[0], "cleverly-reference") == []
    fig = forest(near, truth=0.02)
    assert len(_lines(fig.axes[0], "cleverly-reference")) == 1


def test_forest_title_is_short_and_descriptive() -> None:
    assert len(forest_module.TITLE) <= SHORT_TITLE and not forest_module.TITLE.endswith("?")


def test_overlap_reads_the_treatment_words_beside_arm_labels(binary_fit: Any) -> None:
    labels = {1: "Smoking mothers", 0: "Non-smoking mothers"}
    worded = overlap(binary_fit, labels=labels, treatment_label="smoking").axes[0]
    # one panel: the treatment's words name the probability, and the bars name the groups
    assert worded.get_xlabel() == "Predicted probability of smoking"
    plain = overlap(binary_fit, labels=labels).axes[0]
    assert plain.get_xlabel() == "Predicted probability of being in the group: smoking mothers"
    # The group names in the figure keep the caller's capitals.
    names = [t.get_text() for t in plain.texts if t.get_gid() == _core.GID_SERIES_LABEL]
    assert names == ["Smoking mothers", "Non-smoking mothers"]
    # Control: without labels the treatment words take the column's place.
    unlabelled = overlap(binary_fit, treatment_label="smoking").axes[0]
    assert unlabelled.get_xlabel() == "Predicted probability of smoking = 1"


def test_overlap_key_names_the_limits_as_probability_limits(binary_fit: Any) -> None:
    keys = _key(overlap(binary_fit).axes[0])
    limits = [key for key in keys if "limits" in key.lower()]
    assert limits and all(key.startswith("Probability limits") for key in limits)
    # The estimand each pair belongs to stays named: the ATT and ATC use their own pair.
    assert "Probability limits used for the effect among the treated and the untreated" in keys


def test_overlap_key_names_the_marginal_pair_by_what_it_serves(
    binary_fit: Any, missing_fit: Any
) -> None:
    # The marginal pair serves the arm means and ratios too, not the average effect alone.
    assert {"ey1", "ey0", "rr", "or"} <= set(binary_fit.estimates)
    keys = _key(overlap(binary_fit).axes[0])
    assert not any("for the average effect" in key for key in keys)
    # Control: a fit with no ATT or ATC has one pair, which every estimate used.
    assert not {"att", "atc"} & set(missing_fit.estimates)
    assert [key for key in _key(overlap(missing_fit).axes[0]) if "limits" in key] == [
        "Probability limits the fit used"
    ]


# ----------------------------------------------------------------------- forest, rows with no interval


def test_forest_keeps_the_null_beside_a_row_with_no_interval(binary_fit: Any) -> None:
    withheld = dataclasses.replace(binary_fit["ate"], psi=0.01, inference=WITHHELD)
    unknown = dataclasses.replace(binary_fit["ate"], psi=0.5, variance=float("inf"))
    assert not all(np.isfinite(unknown.ci))
    for row in (withheld, unknown):
        fig = forest(row)
        assert len(_lines(fig.axes[0], "cleverly-reference")) == 1
        assert "off the axis" not in _core.note_text(fig)
    # Beside a drawn narrow interval far from 0 the row without one still keeps the null.
    near = _narrow(binary_fit["ate"], 0.15)
    assert len(_lines(forest(near, withheld, labels=["a", "b"]).axes[0], "cleverly-reference"))
    # Control: the drawn narrow interval alone drops it.
    assert _lines(forest(near).axes[0], "cleverly-reference") == []


def test_a_null_left_off_by_hand_never_claims_an_interval_that_is_missing(
    binary_fit: Any,
) -> None:
    withheld = dataclasses.replace(binary_fit["ate"], psi=0.15, inference=WITHHELD)
    note = _core.note_text(forest(withheld, include_null=False))
    assert "lies off the axis to the left, so its line is not drawn." in note
    assert "no interval here reaches it" not in note
    # Control: rows that all have an interval say none reaches the null.
    drawn = _core.note_text(forest(_narrow(binary_fit["ate"], 0.15), include_null=False))
    assert "no interval here reaches it" in drawn


def test_a_null_left_off_by_hand_is_really_off_the_axis(
    binary_fit: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    ate = binary_fit["ate"]
    # An interval close to 0, about (0.003, 0.188): the autoscale margin would reach 0.
    close = dataclasses.replace(
        ate, psi=0.0955, variance=ate.variance * (0.0473 / ate.std_error) ** 2
    )
    low, high = close.ci
    assert 0.0 < low < 0.01 and high == pytest.approx(0.188, abs=0.002)
    fig = forest(close, include_null=False)
    assert "off the axis to the left" in _core.note_text(fig)
    assert 0.0 < fig.axes[0].get_xlim()[0] < low
    # On a ratio the limit is kept off 1 on the log axis.
    ratio = binary_fit["rr"]
    near_one = _narrow(ratio, 1.15, shrink=0.25)
    assert near_one.ci[0] > 1.0
    fig = forest(near_one, include_null=False)
    assert 1.0 < fig.axes[0].get_xlim()[0] < near_one.ci[0]
    # Mutation control: without the clamp the autoscale margin puts the null back on.
    monkeypatch.setattr(forest_module, "_keep_off", lambda *args, **kwargs: None)
    assert forest(close, include_null=False).axes[0].get_xlim()[0] <= 0.0


# ----------------------------------------------------------------------- formatting keywords


def test_title_and_subtitle_take_a_string_or_false(binary_fit: Any) -> None:
    fig = forest(binary_fit, title="Effect of the offer", subtitle="Main analysis")
    ax = fig.axes[0]
    assert ax.get_title(loc="left") == "Effect of the offer"
    assert _core.subtitle_text(ax) == "Main analysis"
    bare = forest(binary_fit, title=False, subtitle=False).axes[0]
    assert bare.get_title(loc="left") == ""
    assert _core.subtitle_text(bare) is None
    # Control: the defaults stay when neither keyword is given.
    plain = forest(binary_fit).axes[0]
    assert plain.get_title(loc="left") == forest_module.TITLE
    assert _core.subtitle_text(plain)


def test_the_formatting_keywords_refuse_what_they_cannot_draw(binary_fit: Any) -> None:
    with pytest.raises(TypeError, match="title must be a string, False or None"):
        forest(binary_fit, title=True)
    with pytest.raises(TypeError, match="subtitle must be a string, False or None"):
        forest(binary_fit, subtitle=3)
    with pytest.raises(ValueError, match="legend must be one of"):
        forest(binary_fit, legend="left")
    with pytest.raises(ValueError, match="xlabel gives 7 labels for 1 labelled panels"):
        overlap(binary_fit, xlabel=list("abcdefg"))


def test_xlabel_and_ylabel_rename_the_axes(binary_fit: Any) -> None:
    ax = overlap(binary_fit, xlabel="Chance of treatment", ylabel="Share").axes[0]
    assert ax.get_xlabel() == "Chance of treatment"
    assert ax.get_ylabel() == "Share"
    listed = overlap(binary_fit, xlabel=["Chance of treatment"]).axes[0]
    assert listed.get_xlabel() == "Chance of treatment"
    # Control: the plot's own words stay without the keywords.
    assert overlap(binary_fit).axes[0].get_xlabel().startswith("Predicted probability")


def test_a_string_label_renames_every_panel_that_shares_it(binary_fit: Any) -> None:
    fig = binary_fit.diagnostics.truncation_curve().plot(xlabel="Limit")
    shared = [ax.get_xlabel() for ax in fig.axes if ax.axison and ax.get_xlabel()]
    assert shared and set(shared) == {"Limit"}


@pytest.mark.parametrize("place", ["right", "bottom"])
def test_the_key_can_sit_right_of_the_axes_or_under_the_figure(binary_fit: Any, place: str) -> None:
    top = forest(binary_fit)
    labels = _core.key_labels(top)
    assert labels, "the forest plot draws a key"
    moved = forest(binary_fit, legend=place)
    assert _core.key_labels(moved) == labels
    moved.canvas.draw()
    renderer = moved.canvas.get_renderer()
    axes_box = moved.axes[0].get_window_extent(renderer)
    keys = [
        legend for legend in [*moved.legends, *(ax.get_legend() for ax in moved.axes)] if legend
    ]
    assert len(keys) == 1
    box = keys[0].get_window_extent(renderer)
    if place == "right":
        assert box.x0 >= axes_box.x1 - 1.0
    else:
        assert box.y1 <= axes_box.y0 + 1.0
    # The key is part of the figure a tight save keeps.
    tight = moved.get_tightbbox(renderer)
    inches = box.transformed(moved.dpi_scale_trans.inverted())
    assert tight.x0 - 0.01 <= inches.x0 and inches.x1 <= tight.x1 + 0.01
    assert tight.y0 - 0.01 <= inches.y0 and inches.y1 <= tight.y1 + 0.01


def test_legend_false_draws_no_key(binary_fit: Any) -> None:
    assert _core.key_labels(forest(binary_fit, legend=False)) == []
    # Control: the default draws one.
    assert _core.key_labels(forest(binary_fit))


def test_figsize_sets_the_size_of_a_figure_the_plot_makes(binary_fit: Any) -> None:
    fig = overlap(binary_fit, figsize=(5.0, 3.0))
    assert tuple(fig.get_size_inches()) == (5.0, 3.0)
    # Control: the plot's own size otherwise.
    assert tuple(overlap(binary_fit).get_size_inches()) != (5.0, 3.0)


def test_a_plot_drawn_on_a_callers_axes_keeps_the_callers_figure_size(binary_fit: Any) -> None:
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    bounds = marginal_sensitivity_bounds(binary_fit, gamma=(1.0, 2.0))
    bounds.plot(ax=ax, figsize=(9.0, 9.0), title="Mine")
    assert tuple(fig.get_size_inches()) == (4.0, 3.0)
    assert ax.get_title(loc="left") == "Mine"
