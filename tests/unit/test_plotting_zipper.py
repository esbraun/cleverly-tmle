"""The zipper plot of a coverage study, and the interval limits its records now carry."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

import matplotlib
import numpy as np
import pytest
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import plotting
from cleverly._support import Support
from cleverly.datasets import nonlinear_dgp
from cleverly.estimators import TMLE
from cleverly.exceptions import DataError
from cleverly.inference import make_estimate
from cleverly.plotting import _core
from cleverly.plotting._strips import strip_figure
from cleverly.validation import CoverageStudy
from tests.unit._plotting_checks import assert_style, assert_text_inside

N_ROWS = 40


class _Result:
    def __init__(self, estimates: dict[str, Any]) -> None:
        self.estimates = estimates

    def __getitem__(self, name: str) -> Any:
        return self.estimates[name]


class _Estimator:
    """Builds one difference, one ratio and one t-referenced estimate from the draw's seed."""

    def fit(self, frame: Any, **kwargs: Any) -> _Result:
        rng = np.random.default_rng(frame.seed)
        curve = rng.normal(size=N_ROWS)
        curve = curve - curve.mean()
        shift = 0.25 * (frame.seed % 5 - 2)
        return _Result(
            {
                "rd": make_estimate("rd", 0.1 + shift, curve, n=N_ROWS, rule_widened=False),
                "rr": make_estimate(
                    "rr",
                    float(np.exp(0.2 + shift)),
                    curve,
                    n=N_ROWS,
                    scale="ratio",
                    log_psi=0.2 + shift,
                    rule_widened=False,
                ),
                "few": make_estimate(
                    "few", 0.1 + shift, curve, n=N_ROWS, reference_df=6, rule_widened=False
                ),
            }
        )


def _study(n_replicates: int = 12, truths: dict[str, float] | None = None) -> Any:
    truth = {"rd": 0.1, "rr": float(np.exp(0.2)), "few": 0.1, **(truths or {})}
    return CoverageStudy(
        dgp=lambda n, seed: (SimpleNamespace(seed=seed), truth),
        estimator=_Estimator,
        n=N_ROWS,
        n_replicates=n_replicates,
        seed=5,
    ).run()


@pytest.fixture(scope="module")
def study() -> Any:
    return _study()


def _close(fig: Any) -> None:
    plt.close(fig)


class TestIntervalLimitsOnTheRecords:
    def test_each_records_limits_reproduce_its_covered_flag(self, study: Any) -> None:
        assert len(study.replications) == 3 * 12
        for record in study.replications:
            assert (record.lower <= record.truth <= record.upper) == record.covered
        # the limits are not degenerate: some replications cover and some miss
        flags = {record.covered for record in study.replications}
        assert flags == {True, False}

    def test_a_difference_limit_is_the_wald_interval_by_hand(self, study: Any) -> None:
        record = next(r for r in study.replications if r.estimand == "rd")
        rng = np.random.default_rng(record.seed)
        curve = rng.normal(size=N_ROWS)
        curve = curve - curve.mean()
        half = stats.norm.ppf(1 - record.alpha / 2) * np.sqrt(np.var(curve, ddof=1) / N_ROWS)
        assert record.lower == pytest.approx(record.estimate - half)
        assert record.upper == pytest.approx(record.estimate + half)

    def test_a_ratio_limit_is_on_the_natural_scale(self, study: Any) -> None:
        record = next(r for r in study.replications if r.estimand == "rr")
        half = stats.norm.ppf(1 - record.alpha / 2) * record.std_error
        assert record.lower == pytest.approx(np.exp(record.inference_estimate - half))
        assert record.upper == pytest.approx(np.exp(record.inference_estimate + half))
        assert record.lower < record.estimate < record.upper

    def test_a_t_reference_interval_is_wider_than_the_normal_one(self, study: Any) -> None:
        wide = next(r for r in study.replications if r.estimand == "few")
        normal = next(r for r in study.replications if r.estimand == "rd" and r.seed == wide.seed)
        half = stats.t.ppf(1 - wide.alpha / 2, 6) * wide.std_error
        assert wide.upper - wide.lower == pytest.approx(2 * half)
        assert wide.upper - wide.lower > normal.upper - normal.lower

    def test_a_real_study_with_a_bootstrap_fit_records_the_wald_limits(self) -> None:
        # CoverageStudy measures the interval the estimate reports through plugin_interval,
        # the Wald body, also on a fit that ran a bootstrap: a bootstrap percentile interval
        # never reaches a record, so the records' limits are the Wald ones in every case.
        real = CoverageStudy(
            dgp=nonlinear_dgp(),
            estimator=lambda: TMLE(
                outcome_learner=LinearRegression(),
                treatment_learner=LogisticRegression(max_iter=1000),
                cross_fit=False,
                n_bootstrap=20,
                random_state=0,
            ),
            n=150,
            n_replicates=4,
            estimands=("ate",),
            seed=1,
        ).run()
        assert len(real.replications) == 4
        for record in real.replications:
            assert (record.lower <= record.truth <= record.upper) == record.covered
            half = stats.norm.ppf(1 - record.alpha / 2) * record.std_error
            assert record.lower == pytest.approx(record.estimate - half)
            assert record.upper == pytest.approx(record.estimate + half)
            assert record.support == "supported" and record.support_reason == ""
        fig = plotting.zipper(real)
        assert fig.axes[1].get_xlabel().splitlines()[0] == "Average effect"
        assert "experimental" not in _core.note_text(fig)
        _close(fig)


class TestTheZipper:
    def test_it_draws_every_interval_sorted_by_centre_from_the_true_value(self, study: Any) -> None:
        fig = plotting.zipper(study, estimand="rd")
        axis = fig.axes[1]
        segments = []
        for gid in ("cleverly-zipper-covered", "cleverly-zipper-missed"):
            for collection in axis.collections:
                if collection.get_gid() == gid:
                    segments.extend(collection.get_segments())
        records = [r for r in study.replications if r.estimand == "rd"]
        assert len(segments) == len(records)
        drawn = sorted((float(s[0][0]), float(s[1][0]), float(s[0][1])) for s in segments)
        expected = sorted((r.lower - r.truth, r.upper - r.truth) for r in records)
        assert [x for a, b, _ in drawn for x in (a, b)] == pytest.approx(
            [x for pair in expected for x in pair]
        )
        # rank order follows the centre: the height increases with the centre
        by_height = sorted(drawn, key=lambda item: item[2])
        centres = [(a + b) / 2 for a, b, _ in by_height]
        assert centres == sorted(centres)
        _close(fig)

    def test_the_true_value_line_sits_at_zero_for_a_difference_and_one_for_a_ratio(
        self, study: Any
    ) -> None:
        fig = plotting.zipper(study, estimand=["rd", "rr"])
        lines = [
            [line.get_xdata()[0] for line in axis.lines if line.get_gid() == "cleverly-reference"]
            for axis in fig.axes
            if axis.collections
        ]
        assert lines == [[0.0], [1.0]]
        ratio_axis = [axis for axis in fig.axes if axis.collections][1]
        assert ratio_axis.get_xscale() == "log"
        _close(fig)

    def test_a_ratio_is_measured_as_a_multiple_of_its_own_truth(self) -> None:
        study = _study(truths={"rr": float(np.exp(0.2)) * 1.0})
        fig = plotting.zipper(study, estimand="rr")
        axis = next(a for a in fig.axes if a.collections)
        records = [r for r in study.replications if r.estimand == "rr"]
        lows = sorted(
            float(s[0][0]) for c in axis.collections if c.get_gid() for s in c.get_segments()
        )
        assert lows == pytest.approx(sorted(r.lower / r.truth for r in records))
        _close(fig)

    def test_misses_are_coloured_and_covers_are_not(self, study: Any) -> None:
        fig = plotting.zipper(study, estimand="rd")
        axis = fig.axes[1]
        by_gid = {c.get_gid(): c for c in axis.collections if c.get_gid()}
        misses = [r for r in study.replications if r.estimand == "rd" and not r.covered]
        assert len(by_gid["cleverly-zipper-missed"].get_segments()) == len(misses)
        assert (
            matplotlib.colors.to_hex(by_gid["cleverly-zipper-missed"].get_color()[0])
            == (_core.PALETTE[1])
        )
        assert (
            matplotlib.colors.to_hex(by_gid["cleverly-zipper-covered"].get_color()[0])
            == (_core.PALETTE[0])
        )
        _close(fig)

    def test_the_subtitle_and_labels_state_coverage_and_the_target(self, study: Any) -> None:
        fig = plotting.zipper(study, estimand="rd", labels={"rd": "Risk difference"})
        head = fig.axes[0]
        assert head.get_title(loc="left") == "Interval coverage"
        records = [r for r in study.replications if r.estimand == "rd"]
        coverage = 100 * np.mean([r.covered for r in records])
        subtitle = _core.subtitle_text(head)
        assert f"Risk difference {coverage:.1f}%" in subtitle
        assert "aimed at 95.0%" in subtitle
        assert "12 replications" in subtitle
        assert fig.axes[1].get_xlabel().startswith("Risk difference")
        key = [text.get_text() for text in head.get_legend().get_texts()]
        assert key == ["Covers the true value", "Misses the true value", "True value"]
        _close(fig)

    def test_the_note_counts_the_misses_on_each_side(self, study: Any) -> None:
        fig = plotting.zipper(study, estimand="rd")
        records = [r for r in study.replications if r.estimand == "rd"]
        below = sum(r.upper < r.truth for r in records)
        above = sum(r.lower > r.truth for r in records)
        assert below == 1  # the singular, which the note words as such
        assert f"rd: 1 interval falls wholly below the true value, {above} above" in (
            _core.note_text(fig)
        )
        _close(fig)

    def test_an_unknown_estimand_is_refused(self, study: Any) -> None:
        with pytest.raises(KeyError, match="nope"):
            plotting.zipper(study, estimand="nope")

    def test_study_plot_draws_the_same_figure(self, study: Any, tmp_path: Any) -> None:
        fig = study.plot()
        assert len([a for a in fig.axes if a.collections]) == 3
        fig.savefig(tmp_path / "zipper.png", dpi=60)
        assert (tmp_path / "zipper.png").stat().st_size > 1000
        _close(fig)


class TestTheHonestyRules:
    def test_limits_that_disagree_with_the_recorded_coverage_are_refused(self, study: Any) -> None:
        flipped = tuple(
            dataclasses.replace(r, covered=not r.covered) if r.replicate == 0 else r
            for r in study.replications
        )
        mutated = dataclasses.replace(study, replications=flipped)
        with pytest.raises(DataError, match="disagree"):
            plotting.zipper(mutated, estimand="rd")
        # the control: the unmutated study draws
        _close(plotting.zipper(study, estimand="rd"))

    def test_records_without_limits_are_refused(self, study: Any) -> None:
        bare = tuple(
            dataclasses.replace(r, lower=float("nan"), upper=float("nan"))
            for r in study.replications
        )
        with pytest.raises(DataError, match="no finite interval limits"):
            plotting.zipper(dataclasses.replace(study, replications=bare))

    def test_a_diagnostic_study_says_its_intervals_are_not_confidence_intervals(self) -> None:
        class _Plugin(_Estimator):
            def fit(self, frame: Any, **kwargs: Any) -> _Result:
                result = super().fit(frame)
                rng = np.random.default_rng(frame.seed)
                curve = rng.normal(size=N_ROWS)
                result.estimates = {
                    "rd": make_estimate(
                        "rd",
                        result["rd"].psi,
                        curve - curve.mean(),
                        n=N_ROWS,
                        inference="working_mechanism_plugin",
                        rule_widened=False,
                    )
                }
                return result

        diagnostic = CoverageStudy(
            dgp=lambda n, seed: (SimpleNamespace(seed=seed), {"rd": 0.1}),
            estimator=_Plugin,
            n=N_ROWS,
            n_replicates=6,
            seed=2,
        ).run()
        fig = plotting.zipper(diagnostic)
        assert "not confidence intervals" in _core.note_text(fig)
        _close(fig)
        fig = plotting.zipper(_study(), estimand="rd")
        assert "not confidence intervals" not in _core.note_text(fig)
        _close(fig)

    def test_the_experimental_note_names_the_estimands_and_the_reason(self, study: Any) -> None:
        fig = plotting.zipper(study, estimand=["rd", "rr"])
        assert study.summaries["rd"].support == "experimental"
        notes = _core.note_text(fig)
        assert (
            "rd, rr: experimental estimates. This estimate was built outside a classified fit. "
            "The study measures how often their intervals cover the true value."
        ) in notes
        _close(fig)

    def test_a_supported_study_has_no_experimental_note(self) -> None:
        # the control for the test above: the same mock, its estimates labelled supported
        supported = Support("supported", "A registered study covers it.", ("x",))

        class _Supported(_Estimator):
            def fit(self, frame: Any, **kwargs: Any) -> _Result:
                result = super().fit(frame)
                result.estimates = {
                    name: dataclasses.replace(estimate, support=supported)
                    for name, estimate in result.estimates.items()
                }
                return result

        study = CoverageStudy(
            dgp=lambda n, seed: (SimpleNamespace(seed=seed), {"rd": 0.1}),
            estimator=_Supported,
            n=N_ROWS,
            n_replicates=6,
            estimands=("rd",),
            seed=2,
        ).run()
        assert study.summaries["rd"].support == "supported"
        fig = plotting.zipper(study)
        assert "experimental" not in _core.note_text(fig)
        _close(fig)


class TestTheLayout:
    def test_the_y_label_is_short_and_the_sorting_rule_is_in_the_note(self, study: Any) -> None:
        fig = plotting.zipper(study)
        assert fig.axes[1].get_ylabel() == "Replications (sorted)"
        assert "sorted from bottom to top by the interval's centre" in _core.note_text(fig)
        _close(fig)

    def test_many_replications_draw_thinner_lines(self, study: Any) -> None:
        def widths(source: Any) -> list[float]:
            fig = plotting.zipper(source, estimand="rd")
            found = [float(c.get_linewidth()[0]) for c in fig.axes[1].collections if c.get_gid()]
            _close(fig)
            return found

        assert widths(study) == [1.2, 1.2]
        assert widths(_study(n_replicates=400)) == pytest.approx([180 / 400] * 2)

    def test_the_misses_differ_from_the_covers_in_grey_scale(self, study: Any) -> None:
        def grey(artist: Any) -> float:
            """The luminance of the line as drawn on white, from 0 (black) to 255."""
            red, green, blue, _ = matplotlib.colors.to_rgba(artist.get_color()[0])
            alpha = artist.get_alpha()
            mixed = [alpha * value + (1.0 - alpha) for value in (red, green, blue)]
            return 255.0 * (0.299 * mixed[0] + 0.587 * mixed[1] + 0.114 * mixed[2])

        fig = plotting.zipper(study, estimand="rd")
        by_gid = {c.get_gid(): c for c in fig.axes[1].collections if c.get_gid()}
        covered, missed = by_gid["cleverly-zipper-covered"], by_gid["cleverly-zipper-missed"]
        assert grey(covered) - grey(missed) > 40
        _close(fig)

    def test_the_figures_pass_the_style_and_layout_checks(self, study: Any) -> None:
        for fig in (plotting.zipper(study), plotting.zipper(study, estimand="rd")):
            assert_style(fig)
            assert_text_inside(fig)
            _close(fig)

    def test_the_header_strip_is_sized_from_its_text(self, study: Any) -> None:
        one = plotting.zipper(study, estimand="rd")
        title = "Coverage of the risk difference intervals across the replications " * 3
        long = plotting.zipper(study, estimand="rd", title=title)
        heights = []
        for fig in (one, long):
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            heights.append(fig.axes[1].get_window_extent().height / fig.dpi)
            # no empty band: the title starts, and the panels end, near the figure's edges
            title = fig.axes[0]._left_title
            top = fig.get_figheight() - title.get_window_extent(renderer).y1 / fig.dpi
            bottom = fig.axes[-1].get_tightbbox(renderer).y0 / fig.dpi
            assert 0.0 <= top < 0.25
            assert 0.0 <= bottom < 0.25
        # the panels keep their height; a longer header makes the figure taller, not the panels
        assert heights[0] == pytest.approx(heights[1], abs=0.05)
        assert long.get_figheight() > one.get_figheight() + 0.3
        _close(one)
        _close(long)

    def test_a_chosen_size_is_kept_and_the_panels_take_what_the_header_leaves(
        self, study: Any
    ) -> None:
        fig = plotting.zipper(study, estimand="rd", figsize=(6.0, 5.0))
        assert tuple(fig.get_size_inches()) == (6.0, 5.0)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        title = fig.axes[0]._left_title
        assert title.get_window_extent(renderer).y1 / fig.dpi <= 5.0
        # Control: without figsize= the strip figure sizes itself from its text.
        plain = plotting.zipper(study, estimand="rd")
        assert tuple(plain.get_size_inches()) != (6.0, 5.0)
        _close(fig)
        _close(plain)


class TestTheChecksThemselves:
    def test_the_style_check_rejects_an_off_palette_unlabelled_figure(self) -> None:
        fig, head, (panel,) = strip_figure(1, 1, panel=(3.0, 2.0))
        head.set_title("A title", loc="left")
        panel.set_xlabel("x")
        panel.set_yticks([])
        for index in range(8):
            panel.plot([0, 1], [index, index + 1], color=f"C{index}")
        with pytest.raises(AssertionError):
            assert_style(fig)
        _close(fig)

    def test_the_inside_check_rejects_a_text_off_the_page(self, study: Any) -> None:
        fig = plotting.zipper(study, estimand="rd")
        assert_text_inside(fig)
        stray = fig.axes[-1].annotate(
            "a text that runs off the right edge",
            xy=(0.98, 0.02),
            xycoords="figure fraction",
        )
        stray.set_in_layout(False)
        with pytest.raises(AssertionError, match="runs off"):
            assert_text_inside(fig)
        _close(fig)
