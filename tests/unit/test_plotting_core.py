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

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: E402

from cleverly import ATE, CausalStudy, PointTreatment  # noqa: E402
from cleverly._inference_status import status_record  # noqa: E402
from cleverly._support import Support  # noqa: E402
from cleverly.datasets import make_missing_outcome  # noqa: E402
from cleverly.estimators import TMLE  # noqa: E402
from cleverly.plotting import PALETTE, forest, overlap, sensitivity, style  # noqa: E402
from cleverly.plotting import _core  # noqa: E402
from cleverly.sensitivity import marginal_sensitivity_bounds  # noqa: E402

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
        fig, ax = plt.subplots()
        (line,) = ax.plot([0, 1], [0, 1])
        assert to_hex(line.get_color()) == PALETTE[0]
    assert dict(matplotlib.rcParams) == before


def test_a_plot_leaves_the_callers_settings_alone(binary_fit: Any) -> None:
    before = dict(matplotlib.rcParams)
    binary_fit.plot()
    assert dict(matplotlib.rcParams) == before


def _style_checks(fig: Any) -> None:
    """Every figure: opaque white face, a title, labelled axes, at most three series colours."""
    assert to_hex(fig.get_facecolor(), keep_alpha=True) == "#ffffffff"
    assert _title(fig)
    positions = [tuple(np.round(ax.get_position().bounds, 6)) for ax in fig.axes]
    assert len(positions) == len(set(positions)), "two axes overlay each other (a twin axis)"
    for ax in fig.axes:
        assert ax.get_xlabel() or ax.get_shared_x_axes().get_siblings(ax) != [ax]
        assert ax.get_ylabel() or ax.get_yticklabels()
        used = set()
        for artist in [*ax.lines, *ax.collections, *ax.patches]:
            for color in np.atleast_2d(_colors(artist)):
                hexed = to_hex(color)
                if hexed in PALETTE:
                    used.add(hexed)
        assert len(used) <= _core.MAX_SERIES


def _colors(artist: Any) -> Any:
    if hasattr(artist, "get_edgecolor") and hasattr(artist, "get_facecolor"):
        faces = np.atleast_2d(artist.get_facecolor())
        edges = np.atleast_2d(artist.get_edgecolor())
        return np.vstack([faces, edges]) if faces.size and edges.size else faces
    return np.atleast_2d(matplotlib.colors.to_rgba(artist.get_color()))


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
    assert _title(fig) == "Estimated effects"
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


def test_truncation_plot_needs_a_recorded_scale(binary_fit: Any) -> None:
    curve = binary_fit.diagnostics.truncation_curve()
    bare = dataclasses.replace(curve, scales={})
    with pytest.raises(ValueError, match="pass scale="):
        bare.plot()
    assert bare.plot(scale="difference").axes[0].get_yscale() == "linear"


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
    subtitle = _core.subtitle_text(ax) or ""
    if bounds.breakdown is not None:
        assert f"gamma = {bounds.breakdown:g}" in subtitle
    if bounds.limit_descriptive:
        assert "descriptive" in _core.note_text(fig)


def test_missingness_plot_marks_the_main_analysis(missing_fit: Any) -> None:
    tilt = missing_fit.sensitivity.missingness(gamma=[-1.0, 0.0, 1.0])
    fig = tilt.plot()
    ax = fig.axes[0]
    drawn = np.asarray(sorted(_points(ax)))
    expected = np.asarray(sorted(zip(tilt.column("gamma"), tilt.column("psi"), strict=True)))
    assert drawn == pytest.approx(expected)
    (main,) = _lines(ax, "cleverly-marker")
    assert main.get_xdata()[0] == 0.0
    assert len(_interval_segments(ax)) == 3


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
    markers = sorted(line.get_xdata()[0] for line in _lines(ax, "cleverly-marker"))
    assert markers == pytest.approx(sorted(bounds))
    shares = [text.get_text() for text in ax.texts if text.get_gid() == "cleverly-tail-share"]
    below_share = float(np.mean(raw < bounds[0]))
    assert shares[0] == f"{100 * below_share:.1f}% below"
    assert ax.get_xlabel() == "Predicted probability of A = 1"


def test_overlap_refuses_a_continuous_treatment() -> None:
    from cleverly.exceptions import DataError
    from tests.unit._capability_sweep_support import fit_shift

    with pytest.raises(DataError, match="continuous"):
        overlap(fit_shift())
