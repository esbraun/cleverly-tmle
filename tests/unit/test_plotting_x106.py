"""Plot preparation shared or skipped without changing a drawn number (X106 part F).

Each speedup keeps the figure bitwise: the tests compare it with a copy of the code it replaced,
kept here, and each has a mutation control that the comparison must catch.
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Callable
from typing import Any

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import cleverly.sensitivity.positivity as positivity_module
from cleverly import plotting
from cleverly.inference import make_estimate
from cleverly.inference.sparse import influence_row
from cleverly.plotting import _curves as diagnostics_curves
from cleverly.plotting import _diagnostics as diagnostics_module
from cleverly.plotting import _weights as weights_module
from tests.unit import _capability_sweep_support as sweep
from tests.unit import test_plotting_result_matrix as matrix


@pytest.fixture(autouse=True)
def _close_figures() -> Any:
    yield
    plt.close("all")


def _quiet(build: Callable[[], Any]) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build()


# --------------------------------------------------------------------------- overlap


def _drawn(fig: Any) -> list[tuple[Any, ...]]:
    """Every panel's bar heights and edges, its y-limits and its annotation texts."""
    panels = []
    for axis in fig.axes:
        bars = tuple((patch.get_x(), patch.get_height()) for patch in axis.patches)
        texts = tuple(text.get_text() for text in axis.texts)
        panels.append((bars, axis.get_ylim(), texts))
    return panels


def _overlap_outcome(result: Any) -> Any:
    try:
        fig = plotting.overlap(result)
    except Exception as error:
        return (type(error), str(error), getattr(error, "reason", None))
    drawn = (_drawn(fig), plotting.notes(fig))
    plt.close(fig)
    return drawn


def _overlap_matches_the_full_report(result: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The figure or refusal equals the one drawn when the whole support report is built."""
    current = _overlap_outcome(result)
    refuse = positivity_module._refuse_unreportable
    report = positivity_module.positivity_report
    inside: list[bool] = []

    def whole_report(fitted: Any) -> None:
        # The code before X106 built the whole report for its refusals; the report itself
        # calls the check first, which then runs as written.
        if inside:
            refuse(fitted)
            return
        inside.append(True)
        try:
            report(fitted)
        finally:
            inside.pop()

    with monkeypatch.context() as patched:
        patched.setattr(positivity_module, "_refuse_unreportable", whole_report)
        before = _overlap_outcome(result)
    assert current == before


@pytest.mark.parametrize("kind", list(sweep.KINDS))
def test_overlap_draws_or_refuses_as_with_the_full_support_report(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _overlap_matches_the_full_report(_quiet(sweep.KINDS[kind].build), monkeypatch)


def test_overlap_reads_no_support_report(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _quiet(sweep.fit_binary)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("overlap() built the support report")

    monkeypatch.setattr(positivity_module, "positivity_report", forbidden)
    plt.close(plotting.overlap(result))


@pytest.mark.parametrize(
    "build",
    [sweep.fit_shift, sweep.fit_natural_course],
    ids=["continuous treatment", "natural course"],
)
def test_overlap_raises_the_support_reports_refusal(build: Callable[[], Any]) -> None:
    result = _quiet(build)
    with pytest.raises(Exception) as from_report:
        positivity_module.positivity_report(result)
    with pytest.raises(type(from_report.value)) as from_plot:
        plotting.overlap(result)
    assert str(from_plot.value) == str(from_report.value)
    for field in ("reason", "alternative"):
        assert getattr(from_plot.value, field, None) == getattr(from_report.value, field, None)


def test_a_refusal_dropped_from_the_check_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    # Mutation control: without its continuous-treatment branch the check lets a dose
    # through, and the figure no longer matches the report's refusal.
    result = _quiet(sweep.fit_shift)
    original = positivity_module._refuse_unreportable

    def without_dose_check(fitted: Any) -> None:
        if fitted.data.is_continuous_treatment:
            return
        original(fitted)

    monkeypatch.setattr(positivity_module, "_refuse_unreportable", without_dose_check)
    with pytest.raises(AssertionError):
        _overlap_matches_the_full_report(result, pytest.MonkeyPatch())


def test_overlap_bars_equal_two_fresh_histograms() -> None:
    # The counts computed once for the peak are the ones drawn: recompute each from the rows.
    for build in (sweep.fit_binary, sweep.fit_multi_arm, sweep.fit_weighted):
        result = _quiet(build)
        fig = plotting.overlap(result)
        propensity = result.repeats[0].nuisance.propensity
        treatment = np.asarray(result.data.treatment, dtype=float)
        recorded = np.isfinite(treatment)
        weights = np.asarray(result.data.weights, dtype=float)[recorded]
        arms = tuple(float(arm) for arm in propensity.arms)
        shown = (arms[1],) if len(arms) == 2 else arms
        edges = np.linspace(0.0, 1.0, 41)
        for axis, arm in zip(fig.axes, shown, strict=False):
            raw = np.asarray(propensity.values, dtype=float)[recorded, propensity.column_for(arm)]
            inside = treatment[recorded] == arm
            heights = np.array([patch.get_height() for patch in axis.patches])
            for sign, mask, drawn in ((1.0, inside, heights[:40]), (-1.0, ~inside, heights[40:])):
                counts, _ = np.histogram(raw[mask], bins=edges, weights=weights[mask])
                assert np.array_equal(drawn, sign * counts / float(weights.sum()))
        plt.close(fig)


# --------------------------------------------------------------------------- influence


def _kish_note(fig: Any) -> str:
    return next(note for note in plotting.notes(fig) if note.startswith("Kish effective count"))


def _curves() -> list[tuple[str, np.ndarray, Any]]:
    rng = np.random.default_rng(3)
    dominant = rng.normal(size=400)
    dominant[7] = 60.0  # one row carries most of the sum of squares
    sizes = rng.integers(1, 9, size=120)
    codes = np.repeat(np.arange(120), sizes)
    rng.shuffle(codes)
    return [
        ("rows", rng.standard_t(5, size=1000), None),
        ("dominant", dominant, None),
        ("clusters", rng.normal(size=codes.size), codes),
        ("string clusters", rng.normal(size=codes.size), np.array([f"c{c:03d}" for c in codes])),
    ]


@pytest.mark.parametrize("label, curve, cluster", _curves(), ids=lambda value: str(value)[:16])
def test_influence_reports_the_kish_count_of_influence_row(
    label: str, curve: np.ndarray, cluster: Any
) -> None:
    estimate = make_estimate(
        "ate",
        0.2,
        curve,
        n=curve.size,
        rule_widened=False,
        cluster=None if cluster is None else np.unique(cluster, return_inverse=True)[1],
    )
    fig = plotting.influence(estimate, cluster=cluster)
    fraction, *_, found = diagnostics_module._concentration(curve, cluster)
    kish = influence_row("ate", curve, cluster=cluster).kish
    assert found == kish
    assert _kish_note(fig) == f"Kish effective count: {kish:.0f} of {fraction.size - 1} " + (
        "rows." if cluster is None else "clusters."
    )


@pytest.mark.parametrize("scale", [1e-82, 1e78])
def test_influence_raises_where_the_kish_count_is_not_finite(scale: float) -> None:
    curve = np.random.default_rng(3).normal(size=200) * scale
    with np.errstate(all="ignore"):  # the estimate's own moments over- or underflow too
        estimate = make_estimate("ate", 0.2, curve, n=200, rule_widened=False)
        with pytest.raises((ZeroDivisionError, OverflowError)):
            influence_row("ate", curve)
    with pytest.raises(ValueError, match="Kish count"):
        plotting.influence(estimate)


def test_influence_draws_at_a_small_but_finite_scale() -> None:
    curve = np.random.default_rng(3).normal(size=200) * 1e-75
    estimate = make_estimate("ate", 0.2, curve, n=200, rule_widened=False)
    kish = influence_row("ate", curve).kish
    assert (
        _kish_note(plotting.influence(estimate)) == f"Kish effective count: {kish:.0f} of 200 rows."
    )


# --------------------------------------------------------------------------- balance


def _shown_before_x106(summary: Any, max_covariates: int) -> tuple[list[str], int]:
    """The selection as written before X106, which rescanned every row per covariate."""
    covariates = list(dict.fromkeys(row.covariate for row in summary.rows))

    def largest(name: str, attribute: str) -> float:
        values = [
            abs(getattr(row, attribute))
            for row in summary.rows
            if row.covariate == name and np.isfinite(getattr(row, attribute))
        ]
        return max(values, default=0.0)

    before = {name: largest(name, "smd_unweighted") for name in covariates}
    after = {name: largest(name, "smd_weighted") for name in covariates}
    kept = [name for name in covariates if after[name] > summary.threshold]
    if covariates:
        kept.append(max(covariates, key=lambda name: after[name]))
    ranked = sorted(covariates, key=lambda name: -max(before[name], after[name]))
    for name in ranked:
        if len(set(kept)) >= max_covariates:
            break
        kept.append(name)
    chosen = set(kept)
    order = sorted(
        (name for name in covariates if name in chosen),
        key=lambda name: (-before[name], covariates.index(name)),
    )
    return order, len(covariates) - len(order)


class _Row:
    def __init__(self, covariate: str, before: float, after: float) -> None:
        self.covariate = covariate
        self.smd_unweighted = before
        self.smd_weighted = after


class _Summary:
    def __init__(self, rows: list[_Row], threshold: float = 0.1) -> None:
        self.rows = rows
        self.threshold = threshold


def _summaries() -> dict[str, _Summary]:
    rng = np.random.default_rng(11)
    ties = [_Row(f"t{i}", 0.3, 0.05) for i in range(6)] + [_Row("t2", -0.3, 0.2)]
    odd = [
        _Row("nan", np.nan, np.nan),
        _Row("inf", np.inf, 0.4),
        _Row("neg", -0.5, -0.02),
        _Row("zero", -0.0, 0.0),
        _Row("nan", 0.2, np.nan),
    ]
    levels = [_Row(f"c{i // 3}", *rng.normal(scale=0.2, size=2)) for i in range(60)]
    panels = [_Row(f"p{i % 25}", *rng.normal(scale=0.15, size=2)) for i in range(100)]
    flagged = [_Row(f"f{i}", 0.5, 0.3 + 0.01 * (i % 3)) for i in range(40)]
    calm = [_Row(f"q{i}", 0.01 * (i % 4), 0.001) for i in range(40)]
    return {
        "ties": _Summary(ties),
        "non-finite": _Summary(odd),
        "levels": _Summary(levels),
        "panels": _Summary(panels),
        "all flagged": _Summary(flagged),
        "none flagged": _Summary(calm),
    }


@pytest.mark.parametrize("name", list(_summaries()))
@pytest.mark.parametrize("cap", [1, 3, 30, 1000])
def test_balance_selection_matches_the_rescanning_one(name: str, cap: int) -> None:
    summary = _summaries()[name]
    assert weights_module._shown_covariates(summary, cap) == _shown_before_x106(summary, cap)


def test_a_last_maximum_tie_rule_is_caught() -> None:
    # Mutation control: with the largest after-weighting differences tied and none flagged,
    # a selection that keeps the last of them shows another covariate, and the comparison
    # with the rescanning selection sees it.
    summary = _Summary([_Row(f"t{i}", 0.01 * i, 0.05) for i in range(5)])

    def last_maximum(rows: _Summary, cap: int) -> tuple[list[str], int]:
        reversed_rows = _Summary(list(reversed(rows.rows)), rows.threshold)
        shown, _ = _shown_before_x106(reversed_rows, cap)
        return shown, len(rows.rows) - len(shown)

    assert weights_module._shown_covariates(summary, 1) == (["t0"], 4)
    assert last_maximum(summary, 1) != _shown_before_x106(summary, 1)


def test_balance_selection_on_a_fit_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    from cleverly.sensitivity import covariate_balance

    summary = covariate_balance(_quiet(sweep.fit_multi_arm))
    for cap in (1, 2, 30):
        assert weights_module._shown_covariates(summary, cap) == _shown_before_x106(summary, cap)


# --------------------------------------------------------------------------- nuisance report


def _counting(monkeypatch: pytest.MonkeyPatch, module: Any, name: str) -> list[int]:
    calls: list[int] = []
    original = getattr(module, name)

    def counted(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, counted)
    return calls


def _both_plots(result: Any) -> None:
    for plot in (plotting.calibration, plotting.learners):
        # A refusal still reads the report first.
        with contextlib.suppress(ValueError, TypeError):
            plt.close(plot(result))
    result.diagnostics.nuisance_models()


def _builders() -> list[tuple[str, Callable[[], Any], str, str]]:
    return [
        ("point", matrix._point_tmle, "cleverly.validation.nuisance", "nuisance_diagnostics"),
        ("longitudinal", matrix._ltmle_repeats, "cleverly.assessment", "_longitudinal_nuisances"),
        (
            "continuous survival",
            matrix._continuous_survival,
            "cleverly.survival.diagnostics",
            "nuisance_report",
        ),
    ]


@pytest.mark.parametrize("family, build, module, name", _builders(), ids=lambda v: str(v)[:20])
def test_calibration_and_learners_build_one_nuisance_report(
    family: str, build: Callable[[], Any], module: str, name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    result = _quiet(build)
    calls = _counting(monkeypatch, importlib.import_module(module), name)
    _both_plots(result)
    assert len(calls) == 1
    # Control: a result with its cache emptied between the plots builds it again.
    result.assessment_cache.clear()
    _both_plots(result)
    assert len(calls) == 2


# --------------------------------------------------------------------------- display limits


def _texts(fig: Any) -> list[str]:
    """Every text the figure draws: title, subtitle, labels, annotations and tick labels."""
    found = [text.get_text() for text in fig.texts]
    for axis in fig.axes:
        found.extend(text.get_text() for text in axis.texts)
        found.extend([axis.get_xlabel(), axis.get_ylabel(), axis.get_title(loc="left")])
    return found


def _same_but_one_note(limited: Any, plain: Any, prefix: str) -> None:
    """The display contract: every drawn text and note equal, plus one display note."""
    assert _texts(limited) == _texts(plain)
    added = [note for note in plotting.notes(limited) if note not in plotting.notes(plain)]
    assert len(added) == 1 and added[0].startswith(prefix), added
    assert [note for note in plotting.notes(limited) if note not in added] == list(
        plotting.notes(plain)
    )


def _influence_line(fig: Any) -> np.ndarray:
    line = next(line for line in fig.axes[0].lines if line.get_gid() == "cleverly-influence-curve")
    return np.asarray(line.get_xydata())


def _short_total_curve() -> np.ndarray:
    """A curve whose cumulative share ends a rounding error below 1."""
    for seed in range(200):
        curve = np.random.default_rng(seed).standard_t(5, size=1000)
        fraction, share, *_ = diagnostics_module._concentration(curve)
        if fraction[-1] + share[-1] < 2.0:
            return curve
    raise AssertionError("no seed gives a total below 2")


def _influence_fixtures() -> dict[str, tuple[np.ndarray, Any]]:
    rng = np.random.default_rng(5)
    dominant = rng.normal(size=1000)
    dominant[3] = 400.0  # one row carries about 90% of the sum of squares
    codes = np.repeat(np.arange(300), 4)
    return {
        "n=50": (rng.standard_t(5, size=50), None),
        "n=1,000": (rng.standard_t(5, size=1000), None),
        "n=100,000": (rng.standard_t(5, size=100_000), None),
        "total below 2": (_short_total_curve(), None),
        "dominant": (dominant, None),
        "clusters": (rng.normal(size=codes.size), codes),
    }


def _bare(curve: np.ndarray, cluster: Any) -> Any:
    return make_estimate(
        "ate",
        0.2,
        curve,
        n=curve.size,
        rule_widened=False,
        cluster=cluster,
    )


@pytest.mark.parametrize("name", list(_influence_fixtures()))
def test_a_thinned_influence_line_stays_within_its_bound(name: str) -> None:
    curve, cluster = _influence_fixtures()[name]
    fraction, share, *_ = diagnostics_module._concentration(curve, cluster)
    plain = plotting.influence(_bare(curve, cluster), cluster=cluster)
    limited = plotting.influence(_bare(curve, cluster), cluster=cluster, max_points=16)
    drawn = _influence_line(limited)
    assert 3 <= len(drawn) <= 16 < fraction.size
    gap = np.max(np.abs(share - np.interp(fraction, drawn[:, 0], drawn[:, 1])))
    assert gap <= 2.0 / (16 - 4)
    # Both ends and the top-share point are exact vertices of the drawn line.
    top = max(1, int(np.ceil(0.01 * (fraction.size - 1) - 1e-12)))
    for index in (0, top, fraction.size - 1):
        assert [fraction[index], share[index]] in drawn.tolist()
    _same_but_one_note(limited, plain, "Line drawn through")


def test_influence_draws_every_point_by_default_or_when_the_limit_is_not_below() -> None:
    curve, _ = _influence_fixtures()["n=1,000"]
    fraction, share, *_ = diagnostics_module._concentration(curve)
    for options in ({}, {"max_points": fraction.size}):
        fig = plotting.influence(_bare(curve, None), **options)
        assert np.array_equal(_influence_line(fig), np.column_stack([fraction, share]))
        assert not any(note.startswith("Line drawn") for note in plotting.notes(fig))


def test_a_selector_that_drops_the_top_point_is_caught() -> None:
    # Mutation control: on the dominant curve the evenly spread points miss the top-share
    # point at 16, so a selector without it no longer draws it exactly.
    curve, _ = _influence_fixtures()["dominant"]
    fraction, share, *_ = diagnostics_module._concentration(curve)
    count = fraction.size - 1
    top = max(1, int(np.ceil(0.01 * count - 1e-12)))
    spread = np.minimum(
        np.searchsorted(fraction + share, np.linspace(0.0, 2.0, 16 - 3), side="left"), count
    )
    assert top not in spread
    without_top = np.unique(np.concatenate([spread, [0, count]]))
    assert top in diagnostics_module._line_points(fraction, share, top, 16)
    assert top not in without_top


@pytest.mark.parametrize("value", [True, 15, 2.5, "16"])
def test_influence_refuses_a_bad_point_limit(value: Any) -> None:
    curve, _ = _influence_fixtures()["n=50"]
    with pytest.raises(ValueError, match="max_points"):
        plotting.influence(_bare(curve, None), max_points=value)


# --------------------------------------------------------------------------- rug


def _stride_before_x106(values: np.ndarray) -> np.ndarray:
    if values.size > 20000:
        return values[:: int(np.ceil(values.size / 20000))]
    return values


def test_the_rug_limit_ignores_row_order() -> None:
    values = np.random.default_rng(2).uniform(size=5000)
    shuffled = np.random.default_rng(3).permutation(values)
    picked = diagnostics_curves._rug_values(values, 11)
    assert np.array_equal(picked, diagnostics_curves._rug_values(shuffled, 11))
    # The 0th, 10th, ..., 100th percentiles by rank.
    ordered = np.sort(values)
    assert np.array_equal(picked, ordered[np.rint(np.linspace(0, 4999, 11)).astype(int)])


def test_a_row_order_stride_is_caught() -> None:
    # Mutation control: a stride over rows sorted then blocked picks other values when the
    # same rows come in another order.
    values = np.sort(np.random.default_rng(4).uniform(size=50_000))
    blocked = np.concatenate([values[1::2], values[::2]])
    assert not np.array_equal(_stride_before_x106(values), _stride_before_x106(blocked))
    assert np.array_equal(
        diagnostics_curves._rug_values(values, 2000),
        diagnostics_curves._rug_values(blocked, 2000),
    )


@pytest.mark.parametrize("size", [100, 20000, 20001, 65_432])
def test_the_default_rug_is_the_old_stride(size: int) -> None:
    values = np.random.default_rng(size).normal(size=size)
    assert np.array_equal(diagnostics_curves._rug_values(values, None), _stride_before_x106(values))


def test_the_rug_notes_say_what_was_drawn() -> None:
    note = diagnostics_curves._rug_note
    assert note([500], None, "W1") is None
    assert note([30_000], None, "W1").startswith("Rug: at most 20,000 ticks per panel")
    assert note([500, 900], 300, "W1") == (
        "Rug: at most 300 of each panel's rows' W1 values, evenly spaced in rank "
        "(900 rows in the largest panel)."
    )
    assert note([500], 600, "W1") is None
    assert note([500], 0, "W1") == "No rug is drawn."
    assert note([500], 1, "W1") == "Rug: each panel's smallest W1 value only."


@pytest.fixture(scope="module")
def effect_curve() -> Any:
    return _quiet(matrix._effect_curve)


def _rug_segments(fig: Any) -> list[int]:
    return [
        len(item.get_segments())
        for axis in fig.axes
        for item in axis.collections
        if item.get_gid() == "cleverly-rug"
    ]


def test_a_limited_rug_changes_only_the_rug(effect_curve: Any) -> None:
    plain = plotting.curve(effect_curve)
    limited = plotting.curve(effect_curve, max_rug_points=7)
    assert _rug_segments(limited) == [7] and _rug_segments(plain)[0] > 7
    curve_of = [np.asarray(line.get_xydata()) for line in plain.axes[0].lines]
    assert all(
        np.array_equal(a, np.asarray(b.get_xydata()))
        for a, b in zip(curve_of, limited.axes[0].lines, strict=True)
    )
    _same_but_one_note(limited, plain, "Rug: at most 7")
    assert _rug_segments(plotting.curve(effect_curve, max_rug_points=0)) == []


@pytest.mark.parametrize("value", [True, -1, 1.5])
def test_curve_refuses_a_bad_rug_limit(effect_curve: Any, value: Any) -> None:
    with pytest.raises(ValueError, match="max_rug_points"):
        plotting.curve(effect_curve, max_rug_points=value)


# --------------------------------------------------------------------------- zipper


@pytest.fixture(scope="module")
def coverage_study() -> Any:
    from tests.unit.test_plotting_zipper import _study

    return _study(n_replicates=60)


def _zipper_heights(axis: Any) -> np.ndarray:
    heights = [
        np.asarray(item.get_segments())[:, 0, 1]
        for item in axis.collections
        if str(item.get_gid()).startswith("cleverly-zipper-")
    ]
    return np.sort(np.concatenate(heights))


def test_a_limited_zipper_keeps_the_coverage_and_both_ends(coverage_study: Any) -> None:
    plain = plotting.zipper(coverage_study, estimand="rd")
    limited = plotting.zipper(coverage_study, estimand="rd", max_intervals=7)
    heights = _zipper_heights(limited.axes[1])
    assert heights.size == 7 and heights[0] == 1.0 and heights[-1] == 60.0
    assert np.array_equal(heights, np.rint(np.linspace(0, 59, 7)) + 1.0)
    assert limited.axes[1].get_ylim() == plain.axes[1].get_ylim()
    _same_but_one_note(limited, plain, "Drawn: at most 7 intervals per panel")


def test_evenly_spaced_ranks_keep_the_miss_share_within_the_runs_bound() -> None:
    # A property of the selection rule the zipper notes describe: evenly spaced ranks miss the study's miss share by at most the runs of misses and
    # covers in rank order over the drawn count.
    rng = np.random.default_rng(9)
    for _ in range(200):
        size = int(rng.integers(20, 3000))
        hit = rng.uniform(size=size) > rng.uniform(0.02, 0.5)
        hit = hit[np.argsort(rng.normal(size=size) + 3 * (~hit) * rng.choice([-1, 1], size))]
        limit = int(rng.integers(2, size))
        kept = np.unique(np.rint(np.linspace(0, size - 1, limit)).astype(int))
        runs = 1 + int(np.sum(hit[1:] != hit[:-1]))
        assert abs(np.mean(~hit[kept]) - np.mean(~hit)) <= runs / kept.size


@pytest.mark.parametrize("value", [True, 1, 3.0])
def test_zipper_refuses_a_bad_interval_limit(coverage_study: Any, value: Any) -> None:
    with pytest.raises(ValueError, match="max_intervals"):
        plotting.zipper(coverage_study, max_intervals=value)


# --------------------------------------------------------------------------- balance pages


@pytest.fixture(scope="module")
def wide_balance() -> Any:
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    from cleverly import ATE, CausalStudy, PointTreatment
    from cleverly.sensitivity import covariate_balance

    rng = np.random.default_rng(12)
    n, p = 400, 13
    w = rng.normal(size=(n, p))
    a = rng.binomial(1, 1 / (1 + np.exp(-(0.8 * w[:, 0] + 0.6 * w[:, 1]))))
    y = rng.binomial(1, 1 / (1 + np.exp(-(0.5 * a + w[:, 0]))))
    frame = pd.DataFrame(w, columns=[f"W{i}" for i in range(p)]).assign(A=a, Y=y)
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y", treatment="A", adjustment=tuple(f"W{i}" for i in range(p))
        ),
    )
    fit = _quiet(
        lambda: study.identify(ATE()).estimate(
            outcome_learner=LogisticRegression(C=0.05),
            treatment_learner=LogisticRegression(C=0.05),
            cross_fit=False,
            random_state=0,
        )
    )
    return covariate_balance(fit)


def _balance_rows(fig: Any) -> list[str]:
    return [
        label.get_text()
        for axis in fig.axes
        if any(item.get_gid() == "cleverly-balance-after" for item in axis.collections)
        for label in axis.get_yticklabels()
        if label.get_text()
    ]


def test_balance_pages_cover_every_covariate_flagged_first(wide_balance: Any) -> None:
    covariates = list(dict.fromkeys(row.covariate for row in wide_balance.rows))
    flagged = {
        row.covariate for row in wide_balance.rows if abs(row.smd_weighted) > wide_balance.threshold
    }
    assert flagged, "the fixture needs a covariate above the threshold after weighting"
    pages = [
        _balance_rows(plotting.balance(wide_balance, max_covariates=4, page=k))
        for k in (1, 2, 3, 4)
    ]
    assert sorted(name for page in pages for name in page) == sorted(covariates)
    position = {name: k for k, page in enumerate(pages) for name in page}
    assert max(position[name] for name in flagged) <= min(
        position[name] for name in covariates if name not in flagged
    )
    with pytest.raises(ValueError, match="at most 4"):
        plotting.balance(wide_balance, max_covariates=4, page=5)


def test_a_balance_page_replaces_the_soft_cap_note(wide_balance: Any) -> None:
    plain = plotting.balance(wide_balance, max_covariates=4)
    paged = plotting.balance(wide_balance, max_covariates=4, page=2)
    soft = "covariates are not drawn. Every covariate above the threshold"
    assert any(soft in note for note in plotting.notes(plain))
    assert not any(soft in note for note in plotting.notes(paged))
    page_notes = [note for note in plotting.notes(paged) if note.startswith("Page 2 of 4")]
    assert len(page_notes) == 1
    others = [note for note in plotting.notes(plain) if soft not in note]
    assert [note for note in plotting.notes(paged) if note not in page_notes] == others


def test_balance_without_a_page_draws_the_old_selection_in_order(wide_balance: Any) -> None:
    shown, _ = _shown_before_x106(wide_balance, 4)
    assert _balance_rows(plotting.balance(wide_balance, max_covariates=4)) == shown


def _paged_before_flags(summary: Any, max_covariates: int, page: int) -> list[str]:
    """A ranking that ignores the threshold: every covariate by its larger difference."""
    before, after = weights_module._largest_differences(summary)
    covariates = list(before)
    ranked = sorted(covariates, key=lambda name: -max(before[name], after[name]))
    on_page = ranked[(page - 1) * max_covariates : page * max_covariates]
    return sorted(on_page, key=lambda name: (-before[name], covariates.index(name)))


def test_a_page_ranks_flagged_covariates_first() -> None:
    # The unflagged covariate has the largest difference before weighting, so a ranking that
    # ignores the threshold puts it first; the page must put the flagged one first.
    summary = _Summary([_Row("big", 0.9, 0.05), _Row("flag", 0.2, 0.3), _Row("calm", 0.1, 0.01)])
    shown, pages, flagged = weights_module._paged_covariates(summary, 1, 1)
    assert (shown, pages, flagged) == (["flag"], 3, 1)
    # Mutation control: the threshold-blind ranking draws another covariate on page 1.
    assert _paged_before_flags(summary, 1, 1) != shown
    assert weights_module._paged_covariates(summary, 1, 2)[0] == ["big"]


@pytest.mark.parametrize("value", [True, 0, 1.0])
def test_balance_refuses_a_bad_page(wide_balance: Any, value: Any) -> None:
    with pytest.raises(ValueError, match="page"):
        plotting.balance(wide_balance, page=value)


def test_balance_over_time_pages_each_panel() -> None:
    from tests.unit.summary_fixtures import balance_time

    report = balance_time.longitudinal()
    plain = plotting.balance(report, diagnostic="confounding", max_covariates=1)
    pages = []
    page = 1
    while True:
        try:
            fig = plotting.balance(report, diagnostic="confounding", max_covariates=1, page=page)
        except ValueError as error:
            assert "the number of pages" in str(error)
            break
        pages.append(fig)
        page += 1
    assert len(pages) >= 2
    for fig in pages:
        for axis in fig.axes:
            if any(item.get_gid() == "cleverly-balance-after" for item in axis.collections):
                assert len([t for t in axis.get_yticklabels() if t.get_text()]) <= 1
    assert any(note.startswith("Page 1 of") for note in plotting.notes(pages[0]))
    assert not any("Every covariate above the threshold" in n for n in plotting.notes(pages[0]))
    del plain
    with pytest.raises(ValueError, match="view='panels'"):
        plotting.balance(report, diagnostic="confounding", view="panels", page=1)


def _panel_rows(fig: Any) -> dict[str, list[str]]:
    """Each over-time panel's drawn covariates, keyed by the panel's title."""
    found: dict[str, list[str]] = {}
    for axis in fig.axes:
        title = axis.get_title(loc="left")
        if not title or title == "Covariate balance over time":
            continue
        found[title] = [label.get_text() for label in axis.get_yticklabels() if label.get_text()]
    return found


@pytest.mark.parametrize("family", ["longitudinal", "survival"])
def test_balance_over_time_pages_cover_each_panel_and_keep_the_other_notes(family: str) -> None:
    from tests.unit.summary_fixtures import balance_time

    report = getattr(balance_time, family)()
    everything = _panel_rows(plotting.balance(report, diagnostic="confounding", max_covariates=99))
    plain = plotting.notes(plotting.balance(report, diagnostic="confounding", max_covariates=1))
    soft = "Every covariate above the threshold"
    seen: dict[str, list[str]] = {title: [] for title in everything}
    emptied = 0
    page = 1
    while True:
        try:
            fig = plotting.balance(report, diagnostic="confounding", max_covariates=1, page=page)
        except ValueError:
            break
        notes = plotting.notes(fig)
        page_note = [note for note in notes if note.startswith(f"Page {page} of ")]
        assert len(page_note) == 1
        assert [note for note in notes if note not in page_note] == [
            note for note in plain if soft not in note
        ]
        for title, rows in _panel_rows(fig).items():
            seen[title].extend(rows)
            if not rows:
                emptied += 1
                assert title in page_note[0]
                axis = next(a for a in fig.axes if a.get_title(loc="left") == title)
                assert any(t.get_gid() == "cleverly-balance-empty-page" for t in axis.texts)
        page += 1
    assert {title: sorted(rows) for title, rows in seen.items()} == {
        title: sorted(rows) for title, rows in everything.items()
    }
    if family == "longitudinal":
        assert emptied, "the fixture's panels hold different counts, so some page is empty"


def test_the_default_rug_note_appears_past_the_cap(
    effect_curve: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    plain = plotting.curve(effect_curve)
    size = _rug_segments(plain)[0]
    monkeypatch.setattr(diagnostics_curves, "_RUG_CAP", size // 4)
    capped = plotting.curve(effect_curve)
    assert _rug_segments(capped)[0] < size
    _same_but_one_note(capped, plain, f"Rug: at most {size // 4:,} ticks per panel")


def _record(call: Callable[[], Any]) -> tuple[Any, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig = call()
    return fig, [str(item.message) for item in caught]


@pytest.mark.parametrize(
    "plot, options",
    [
        ("influence", {"max_points": 16}),
        ("balance", {"max_covariates": 1, "page": 1}),
    ],
)
def test_a_display_limit_leaves_the_result_and_its_warnings_alone(
    plot: str, options: dict[str, Any]
) -> None:
    result = _quiet(sweep.fit_binary)
    estimates = {
        name: (e.psi, e.variance, e.influence_curve.tobytes())
        for name, e in result.estimates.items()
    }
    _, plain_warnings = _record(lambda: getattr(plotting, plot)(result))
    keys = set(result.assessment_cache)
    _, limited_warnings = _record(lambda: getattr(plotting, plot)(result, **options))
    assert limited_warnings == plain_warnings
    assert set(result.assessment_cache) == keys
    assert {
        name: (e.psi, e.variance, e.influence_curve.tobytes())
        for name, e in result.estimates.items()
    } == estimates
