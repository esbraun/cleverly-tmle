"""The result curves of ``cleverly.plotting.curve`` (V2 to V6) and the working-model accessor.

Structural tests read the drawn artists (line data, collection vertices, texts), never pixels.
Each honesty rule has a test with a mutation control: a fit or a call that must change what is
drawn, or a wrong computation that the same assertion must reject.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from functools import cache
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import (
    CausalStudy,
    ConditionalContrast,
    EPMethod,
    IncrementalMean,
    ModifiedTreatmentPolicyEffect,
    MSMProjection,
    PointTreatment,
    RegimeMean,
)
from cleverly._support import Support
from cleverly.bases import Additive, Cosine, Indicators, Linear, Modifiers, Sieve
from cleverly.datasets import (
    make_linear_ate,
    make_longitudinal,
    make_longitudinal_competing,
    make_longitudinal_repeated,
    make_multi_arm,
    make_shift_dose,
)
from cleverly.exceptions import CapabilityError
from cleverly.interventions import Incremental, Scale, Shift, Static
from cleverly.longitudinal import LTMLE, LongitudinalData, Repeated, SequentialGComputation
from cleverly.msm import MSM, working_model_means
from cleverly.plotting import _core, curve
from cleverly.plotting._curves import _Point, _select
from cleverly.validation.longitudinal import _longitudinal_stagewise

# ---------------------------------------------------------------------------- fixtures


def _quiet(build: Any, *args: Any, **kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build(*args, **kwargs)


def _panel(
    n: int,
    seed: int,
    nodes: int = 4,
    hazard: tuple[float, float] = (0.12, 0.2),
    effects: tuple[float, float] = (0.06, 0.03),
) -> LongitudinalData:
    """A baseline-treated time-to-event panel with ``nodes`` intervals."""
    rng = np.random.default_rng(seed)
    a = rng.binomial(1, 0.5, n)
    w = rng.binomial(1, 0.5, n)
    time = np.full(n, nodes)
    event = np.zeros(n, dtype=int)
    for i in range(n):
        for k in range(1, nodes + 1):
            base = hazard[1] if k != 2 else hazard[0]
            if rng.uniform() < base + effects[0] * a[i] + effects[1] * w[i]:
                time[i], event[i] = k, 1
                break
    frame = pd.DataFrame({"W1": w, "A": a, "time": time, "event": event})
    return LongitudinalData.from_time_to_event(
        frame,
        time="time",
        event="event",
        treatment="A",
        baseline=["W1"],
        grid=list(range(1, nodes + 1)),
    )


def _ltmle(data: Any = None, *, cls: Any = LTMLE, **kwargs: Any) -> Any:
    estimator = cls(
        {"treated": 1, "control": 0},
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=500),
        censoring_learner=LogisticRegression(max_iter=500),
        **kwargs,
    )
    return _quiet(estimator.fit, _panel(400, 1) if data is None else data)


@cache
def full() -> Any:
    return _ltmle(n_folds=2, random_state=0)


@cache
def subset() -> Any:
    return _ltmle(n_folds=2, random_state=0, horizons=[2, 4])


@cache
def boot() -> Any:
    return _ltmle(_panel(200, 1), n_folds=2, random_state=0, n_bootstrap=20)


@cache
def isotonic() -> Any:
    data = _panel(120, 6, hazard=(0.02, 0.2), effects=(0.05, 0.0))
    return _ltmle(data, n_folds=2, random_state=0, isotonic=True)


@cache
def withheld() -> Any:
    return _ltmle(_panel(200, 1), cls=SequentialGComputation)


@cache
def competing() -> Any:
    frame, _ = make_longitudinal_competing(n=300, seed=0)
    estimator = LTMLE(
        {"always": 1, "never": 0},
        n_folds=2,
        random_state=0,
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=500),
        censoring_learner=LogisticRegression(max_iter=500),
    )
    return _quiet(
        estimator.fit,
        frame,
        outcome={"relapse": ["R1", "R2"], "death": ["D1", "D2"]},
        treatment=["A1", "A2"],
        baseline=["W1", "W2"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
    )


@cache
def repeated() -> Any:
    frame, _ = make_longitudinal_repeated(n=300, seed=0)
    estimator = LTMLE(
        {"always": 1, "never": 0},
        n_folds=2,
        random_state=0,
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=500),
        censoring_learner=LogisticRegression(max_iter=500),
    )
    return _quiet(
        estimator.fit,
        frame,
        outcome=Repeated(["Y1", "Y2", "Y3"]),
        treatment=["A1", "A2", "A3"],
        baseline=["W"],
        time_varying=[[], ["L2"], ["L3"]],
        censoring=["C1", "C2", "C3"],
    )


@cache
def end_of_study() -> Any:
    frame, _ = make_longitudinal(n=200, seed=0)
    estimator = LTMLE(
        {"always": 1, "never": 0},
        n_folds=2,
        random_state=0,
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=500),
        censoring_learner=LogisticRegression(max_iter=500),
    )
    return _quiet(
        estimator.fit,
        frame,
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["W1", "W2"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
    )


_ESTIMATE: dict[str, Any] = {
    "outcome_learner": LinearRegression(),
    "treatment_learner": LogisticRegression(max_iter=1000),
    "cross_fit": False,
    "random_state": 0,
}


@cache
def _multi() -> Any:
    frame, _ = make_multi_arm(n=400, seed=0)
    frame["A"] = frame["A"].map({"low": 0, "medium": 1, "high": 2}).astype(float)
    return CausalStudy(
        frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3"))
    )


@cache
def level_means() -> Any:
    regimes = (Static(0, name="a0"), Static(1, name="a1"), Static(2, name="a2"))
    return _quiet(_multi().identify(RegimeMean(regimes)).estimate, **_ESTIMATE)


@cache
def msm_fit() -> Any:
    return _quiet(_multi().identify(MSMProjection(MSM.linear())).estimate, **_ESTIMATE)


@cache
def _binary_study() -> Any:
    rng = np.random.default_rng(0)
    a = rng.integers(0, 3, 400).astype(float)
    frame = pd.DataFrame({"A": a, "W": rng.normal(size=400)})
    frame["Y"] = rng.binomial(1, expit(-0.5 + 0.6 * a + 0.3 * frame["W"])).astype(float)
    return CausalStudy(frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",)))


_BINARY: dict[str, Any] = {
    "outcome_learner": LogisticRegression(max_iter=1000),
    "treatment_learner": LogisticRegression(max_iter=1000),
    "cross_fit": False,
    "random_state": 0,
}


def msm_binary(link: str, modifiers: tuple[str, ...] = ()) -> Any:
    model = MSM.linear(link=link, modifiers=modifiers)  # type: ignore[arg-type]
    return _quiet(_binary_study().identify(MSMProjection(model)).estimate, **_BINARY)


@cache
def _dose_study() -> Any:
    frame, _ = make_shift_dose(n=300, seed=1)
    return CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W1", "W2", "W3"),
            treatment_kind="continuous",
        ),
    )


def policy_fit(policies: list[Any]) -> Any:
    return _quiet(
        _dose_study().identify(ModifiedTreatmentPolicyEffect(policies=policies)).estimate,
        **_ESTIMATE,
    )


@cache
def shifts() -> Any:
    return policy_fit([Shift(d, cap=3.0, name=f"shift {d:g}") for d in (0.0, 0.25, 0.5, 0.75)])


@cache
def mixed_family() -> Any:
    return policy_fit([Shift(0.0, cap=3.0, name="s0"), Scale(1.1, cap=3.0, name="x1.1")])


@cache
def ipsi() -> Any:
    frame, _ = make_linear_ate(n=300, seed=0)
    study = CausalStudy(
        frame,
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4")),
    )
    odds = [Incremental(d, name=f"odds x{d:g}") for d in (0.5, 1.0, 2.0, 4.0)]
    return _quiet(study.identify(IncrementalMean(interventions=odds)).estimate, **_ESTIMATE)


@cache
def _modifier_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 500
    w, z = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
    g = rng.choice(["f", "m"], n)
    a = rng.binomial(1, 0.5, n)
    y = rng.binomial(1, 0.3 + 0.2 * a * (w > 0) + 0.1 * (g == "m") + 0.05 * z)
    return pd.DataFrame({"Y": y, "A": a, "W": w, "Z": z, "G": g})


_MODS = Modifiers(["W", "Z", "G"], domains={"W": (-1, 1), "Z": (-1, 1)}, levels={"G": ("f", "m")})
_CURVE_FIT: dict[str, Any] = {
    "outcome_learner": LogisticRegression(),
    "treatment_learner": LogisticRegression(),
    "cross_fit": False,
    "random_state": 0,
}


def _effect(model: Any, modifiers: Modifiers = _MODS, **kwargs: Any) -> Any:
    study = CausalStudy(
        _modifier_frame(),
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W", "Z", "G")),
    )
    contrast = ConditionalContrast(modifiers, model=model, **kwargs)
    return _quiet(study.identify(contrast).estimate, **_CURVE_FIT).effect_curve


@cache
def cate() -> Any:
    grid = pd.DataFrame({"W": np.linspace(-0.8, 0.8, 5), "Z": 0.0, "G": "f"})
    return _quiet(_effect(Linear(["W", "Z"])), grid=grid, random_state=0)


@cache
def cate_by() -> Any:
    return _quiet(_effect(Additive(Linear(["W"]), Indicators("G"))))


@cache
def log_ratio_curve() -> Any:
    return _quiet(_effect(Linear(["W"]), scale="log_ratio"))


@cache
def ep_curve() -> Any:
    one = Modifiers(["W"], domains={"W": (-1, 1)})
    study = CausalStudy(
        _modifier_frame(), design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",))
    )
    method = EPMethod(sieve=Sieve(Cosine(["W"], size=3), size=2)).with_overrides(
        outcome_learner=LogisticRegression(),
        treatment_learner=LogisticRegression(),
        random_state=0,
    )
    return _quiet(study.identify(ConditionalContrast(one)).estimate, method)


# ---------------------------------------------------------------------------- readers


@pytest.fixture(autouse=True)
def _close_figures() -> Any:
    yield
    plt.close("all")


def lines(ax: Any, gid: str) -> list[Any]:
    return [line for line in ax.lines if line.get_gid() == gid]


def collections(ax: Any, gid: str) -> list[Any]:
    return [item for item in ax.collections if item.get_gid() == gid]


def filled(ax: Any, gid: str) -> list[Any]:
    """The filled polygons (ribbons) drawn under ``gid``."""
    return [c for c in collections(ax, gid) if isinstance(c, matplotlib.collections.PolyCollection)]


def segments(ax: Any, gid: str) -> np.ndarray:
    """Every ``(x, low, high)`` of the vertical segments drawn under ``gid``."""
    found = [
        (seg[0][0], seg[0][1], seg[1][1])
        for item in collections(ax, gid)
        for seg in item.get_segments()
    ]
    return np.array(sorted(found))


def texts(fig: Any, gid: str) -> list[str]:
    return [t.get_text() for ax in fig.axes for t in ax.texts if t.get_gid() == gid]


def key_texts(ax: Any) -> list[str]:
    legend = ax.get_legend()
    return [] if legend is None else [t.get_text() for t in legend.get_texts()]


def frame_rows(result: Any, scale: str, regimen: str, *, level: bool = True) -> pd.DataFrame:
    table = result.curve(scale=scale)
    table = table[(table["regimen"] == regimen) & ((table["scale"] == "level") == level)]
    return table.sort_values("time")


def assert_style(fig: Any) -> None:
    """The style contract every figure meets."""
    assert tuple(fig.get_facecolor()) == (1.0, 1.0, 1.0, 1.0)
    positions = [tuple(np.round(ax.get_position().bounds, 6)) for ax in fig.axes]
    assert len(set(positions)) == len(positions)
    assert fig.axes[0].get_title(loc="left")
    for ax in fig.axes:
        captioned = any(t.get_gid() == "cleverly-count-caption" for t in ax.texts)
        assert ax.get_ylabel() or captioned
    assert fig.axes[-1].get_xlabel()
    for ax in fig.axes:
        colours = {
            line.get_color()
            for line in ax.lines
            if line.get_gid() in ("cleverly-curve", "cleverly-point")
        }
        assert len(colours & set(_core.PALETTE)) <= _core.MAX_SERIES


# ---------------------------------------------------------------------------- V2


class TestOverTime:
    def test_levels_and_contrast_are_the_curve_the_fit_reports(self) -> None:
        result = full()
        fig = curve(result)
        top, diff, counts = fig.axes
        for regimen in ("treated", "control"):
            expected = frame_rows(result, "risk", regimen)
            drawn = [
                line
                for line in lines(top, "cleverly-curve")
                if np.allclose(line.get_xydata()[:, 1], expected["psi"].to_numpy())
            ]
            assert len(drawn) == 1
            assert drawn[0].get_xydata()[:, 0].tolist() == expected["time"].tolist()
        contrast = frame_rows(result, "risk", "control vs treated", level=False)
        (line,) = lines(diff, "cleverly-curve")
        np.testing.assert_allclose(line.get_xydata()[:, 1], contrast["psi"].to_numpy())
        assert fig.axes[0].get_title(loc="left") == "Risk over time"
        assert top.get_ylabel() == "Cumulative risk"
        assert diff.get_ylabel() == "Difference in risk"
        assert counts.get_xlabel() == "Follow-up time"
        assert_style(fig)

    def test_the_reference_line_is_zero_and_the_key_names_each_interval_kind(self) -> None:
        fig = curve(full())
        top, diff, _ = fig.axes
        (null,) = [ln for ln in diff.lines if ln.get_gid() == "cleverly-reference"]
        assert list(null.get_ydata()) == [0.0, 0.0]
        assert key_texts(top) == [
            "95% pointwise interval",
            "95% simultaneous band (multiplier)",
            "No difference",
        ] or set(key_texts(top)) == {
            "95% pointwise interval",
            "95% simultaneous band (multiplier)",
            "No difference",
        }
        assert top.get_legend().get_texts()

    def test_the_count_row_reads_followers_from_the_diagnostics(self) -> None:
        result = full()
        fig = curve(result)
        counts = fig.axes[-1]
        table = _longitudinal_stagewise(result)
        expected = {(r.regimen, r.time): r.n_followed for r in table.rows if r.horizon == r.time}
        drawn = sorted(
            int(t.get_text().replace(",", ""))
            for t in counts.texts
            if t.get_gid() == "cleverly-count"
        )
        assert drawn == sorted(expected.values())
        assert "Following the regimen, uncensored" in texts(fig, "cleverly-count-caption")
        assert [t.get_text() for t in counts.get_yticklabels()] == ["treated", "control"]

    def test_survival_levels_and_contrasts_come_from_the_fits_own_view(self) -> None:
        """Rule 5: the drawn contrast is the fit's negated risk difference, never ``1 - x``."""
        result = full()
        fig = curve(result, scale="survival")
        top, diff, _ = fig.axes
        survival = frame_rows(result, "survival", "treated")
        risk = frame_rows(result, "risk", "treated")
        drawn_levels = [line.get_xydata()[:, 1] for line in lines(top, "cleverly-curve")]
        assert any(np.allclose(y, survival["psi"].to_numpy()) for y in drawn_levels)
        contrast_risk = frame_rows(result, "risk", "control vs treated", level=False)
        (line,) = lines(diff, "cleverly-curve")
        y = line.get_xydata()[:, 1]
        np.testing.assert_allclose(y, -contrast_risk["psi"].to_numpy())
        # Mutation control: the wrong map for a contrast, 1 - x, is not what is drawn, and
        # the assertion above would reject it.
        wrong = 1.0 - contrast_risk["psi"].to_numpy()
        assert not np.allclose(y, wrong)
        assert not np.allclose(np.asarray(drawn_levels), risk["psi"].to_numpy())
        assert fig.axes[0].get_title(loc="left") == "Survival over time"

    def test_the_survival_band_mirrors_the_risk_band_about_the_survival_estimate(self) -> None:
        result = full()
        fig = curve(result, scale="survival")
        top = fig.axes[0]
        ribbons = collections(top, "cleverly-band")
        assert len(ribbons) == 2
        bands = result.simultaneous.bands
        for regimen in ("treated", "control"):
            risk = frame_rows(result, "risk", regimen)
            low = np.array([bands[p][0] for p in risk["parameter"]])
            high = np.array([bands[p][1] for p in risk["parameter"]])
            survival_low = 1.0 - high
            survival_high = 1.0 - low
            found = False
            for ribbon in ribbons:
                values = np.unique(np.round(ribbon.get_paths()[0].vertices[:, 1], 9))
                if np.all(np.isin(np.round(survival_low, 9), values)) and np.all(
                    np.isin(np.round(survival_high, 9), values)
                ):
                    found = True
            assert found

    def test_step_ribbons_when_the_fit_reports_every_node(self) -> None:
        result = full()
        top = curve(result).axes[0]
        ribbons = collections(top, "cleverly-ribbon")
        assert len(ribbons) == 2
        risk = frame_rows(result, "risk", "treated")
        expected = np.unique(np.round(np.r_[risk["ci_lower"], risk["ci_upper"]], 9))
        found = [
            np.all(np.isin(expected, np.unique(np.round(r.get_paths()[0].vertices[:, 1], 9))))
            for r in ribbons
        ]
        assert any(found)
        # The ribbon is constant between nodes: the outline has a vertical edge at each node.
        vertices = ribbons[0].get_paths()[0].vertices
        assert {1.0, 2.0, 3.0, 4.0} <= set(np.round(vertices[:, 0], 9))

    def test_a_declared_subset_of_horizons_draws_points_and_brackets_only(self) -> None:
        """Rule 2's exception: no step artist between declared horizons."""
        result = subset()
        fig = curve(result)
        top = fig.axes[0]
        assert not filled(top, "cleverly-ribbon")
        assert not filled(top, "cleverly-band")
        assert not any(ln.get_drawstyle().startswith("steps") for ln in top.lines)
        assert not lines(top, "cleverly-curve")
        points = {tuple(np.round(ln.get_xydata()[:, 0], 9)) for ln in lines(top, "cleverly-point")}
        assert points == {(2.0, 4.0)}
        bars = segments(top, "cleverly-interval")
        assert sorted(set(bars[:, 0])) == [2.0, 4.0]
        brackets = segments(top, "cleverly-band")
        assert sorted(set(brackets[:, 0])) == [2.0, 4.0]
        assert "horizons 2, 4" in _core.note_text(fig)
        # Mutation control: the fit that reports every node does draw the step artists.
        every = curve(full()).axes[0]
        assert filled(every, "cleverly-ribbon") and filled(every, "cleverly-band")
        assert lines(every, "cleverly-curve")

    def test_the_band_is_the_one_the_fit_computed_and_wider_than_the_interval(self) -> None:
        result = full()
        fig = curve(result)
        bands = result.simultaneous.bands
        top = fig.axes[0]
        (outer, _) = sorted(
            collections(top, "cleverly-band"),
            key=lambda c: c.get_paths()[0].vertices[:, 1].max(),
            reverse=True,
        )
        high = max(bands[f"risk_regimen[treated @ t={t}]"][1] for t in (1, 2, 3, 4))
        assert outer.get_paths()[0].vertices[:, 1].max() == pytest.approx(high)
        for regimen in ("treated", "control"):
            risk = frame_rows(result, "risk", regimen)
            for parameter, upper in zip(risk["parameter"], risk["ci_upper"], strict=True):
                assert bands[parameter][1] > upper

    def test_no_band_is_drawn_for_a_fit_without_one(self) -> None:
        """Rule 2: a pointwise interval is never widened into a band."""
        result = full()
        bare = replace(result, simultaneous=None)
        top = curve(bare).axes[0]
        assert not collections(top, "cleverly-band")
        assert collections(top, "cleverly-ribbon")
        assert "95% pointwise interval" in key_texts(top)
        assert "95% simultaneous band (multiplier)" not in key_texts(top)
        # Mutation control: the fit with the band draws one.
        assert collections(curve(result).axes[0], "cleverly-band")

    def test_the_band_kind_and_the_other_band_are_named(self) -> None:
        result = boot()
        both = curve(result)
        assert "95% simultaneous band (multiplier)" in key_texts(both.axes[0])
        assert "bootstrap sup-t" in _core.note_text(both)
        chosen = curve(result, band="bootstrap")
        assert "95% simultaneous band (bootstrap sup-t)" in key_texts(chosen.axes[0])
        assert "band='multiplier'" in _core.note_text(chosen)
        only = curve(replace(result, simultaneous=None))
        assert "95% simultaneous band (bootstrap sup-t)" in key_texts(only.axes[0])
        with pytest.raises(ValueError, match="no bootstrap band"):
            curve(full(), band="bootstrap")
        assert not collections(curve(result, band="none").axes[0], "cleverly-band")

    def test_an_experimental_band_says_so_and_a_supported_one_does_not(self) -> None:
        """Rule 3, with the band's own support."""
        result = full()
        assert result.simultaneous.support.label == "experimental"
        noted = _core.note_text(curve(result))
        assert "The simultaneous band is experimental" in noted
        assert "The curve is experimental" in noted
        supported = Support("supported", "A registered study covers it.", ("study",), None)
        calm = replace(result, simultaneous=replace(result.simultaneous, support=supported))
        assert "simultaneous band is experimental" not in _core.note_text(curve(calm))

    def test_withheld_inference_draws_points_with_no_interval_and_says_why(self) -> None:
        """Rule 1."""
        result = withheld()
        assert not any(e.supplies_inference for e in result.estimates.values())
        fig = curve(result)
        top = fig.axes[0]
        assert not collections(top, "cleverly-ribbon") and not collections(top, "cleverly-band")
        assert not segments(top, "cleverly-interval").size
        assert "No interval is drawn" in _core.note_text(fig)
        # Mutation control: the fit with inference draws the ribbon.
        assert collections(curve(full()).axes[0], "cleverly-ribbon")

    def test_isotonic_moves_draw_hollow_raw_markers_beside_the_projected_curve(self) -> None:
        """Rule 6."""
        result = isotonic()
        moves = result.isotonic_moves
        assert moves
        fig = curve(result)
        top = fig.axes[0]
        raw = lines(top, "cleverly-raw")
        assert raw
        drawn = sorted((float(x), float(y)) for line in raw for x, y in line.get_xydata().tolist())
        expected = sorted(
            (float(h), float(result.scaler.unscale_level(before))) for _, _, h, before, _ in moves
        )
        np.testing.assert_allclose(drawn, expected)
        assert all(line.get_markerfacecolor() == _core.FACE for line in raw)
        assert "Hollow markers" in _core.note_text(fig)
        # The projected curve is the fit's, and differs from the raw values.
        projected = sorted((float(h), float(after)) for _, _, h, _, after in moves)
        assert not np.allclose([y for _, y in expected], [y for _, y in projected], atol=1e-12)
        # Mutation controls: a fit with no move draws none, and the survival view mirrors.
        assert not lines(curve(full()).axes[0], "cleverly-raw")
        survival = curve(result, scale="survival").axes[0]
        assert lines(survival, "cleverly-raw")

    def test_a_refused_view_raises_the_fits_own_reason(self) -> None:
        """Rule 7."""
        result = competing()
        with pytest.raises(CapabilityError) as plotted:
            curve(result, scale="survival", cause="relapse")
        with pytest.raises(CapabilityError) as direct:
            result.curve(scale="survival")
        assert str(plotted.value) == str(direct.value)
        # Mutation control: a view the fit supports does draw, for one cause at a time.
        fig = curve(result, cause="relapse")
        assert "(relapse)" in (fig.axes[0].get_title(loc="left") + _subtitle(fig))
        with pytest.raises(ValueError, match="pass cause="):
            curve(result)

    def test_a_repeated_outcome_is_a_mean_over_time(self) -> None:
        result = repeated()
        fig = curve(result, units="proportion")
        top = fig.axes[0]
        assert fig.axes[0].get_title(loc="left") == "Mean outcome over time"
        assert "Mean" in top.get_ylabel()
        expected = frame_rows(result, "mean", "always")
        assert any(
            np.allclose(line.get_xydata()[:, 1], expected["psi"].to_numpy())
            for line in lines(top, "cleverly-curve")
        )

    def test_too_many_series_for_one_axes_are_refused(self) -> None:
        rows = [
            _Point(f"p{i}", name, 1.0, 1, 0.1, None, None, True, None)
            for i, name in enumerate("abcd")
        ]
        with pytest.raises(ValueError, match="at most 3 coloured series"):
            _select(rows, None)
        chosen, _ = _select(rows, ["a", "b"])
        assert chosen == ["a", "b"]
        with pytest.raises(ValueError, match="unknown regimen"):
            _select(rows, ["z"])

    def test_plot_is_the_curve_for_a_curve_fit_and_the_forest_otherwise(self) -> None:
        fig = full().plot()
        assert fig.axes[0].get_title(loc="left") == "Risk over time"
        forest = end_of_study().plot()
        assert forest.axes[0].get_title(loc="left") == "Estimated effects"

    def test_nothing_is_cached_on_the_result(self) -> None:
        result = full()
        before = dict(result.assessment_cache)
        curve(result)
        assert dict(result.assessment_cache) == before

    def test_a_png_renders(self, tmp_path: Any) -> None:
        path = tmp_path / "over_time.png"
        for name, fig in (
            ("risk", curve(full())),
            ("survival", curve(full(), scale="survival")),
            ("subset", curve(subset())),
        ):
            fig.savefig(tmp_path / f"{name}.png", dpi=60)
            assert (tmp_path / f"{name}.png").stat().st_size > 1000
        assert not path.exists()


def _subtitle(fig: Any) -> str:
    return _core.subtitle_text(fig.axes[0]) or ""


# ---------------------------------------------------------------------------- V3


class TestRestricted:
    def test_points_are_the_fits_restricted_means_at_each_horizon(self) -> None:
        result = full()
        fig = curve(result, measure="rmst")
        top, diff = fig.axes
        horizons = [2, 3, 4]
        for regimen in ("treated", "control"):
            expected = [result.rmst(regimen, h).psi for h in horizons]
            found = [
                line
                for line in lines(top, "cleverly-guide")
                if np.allclose(line.get_xydata()[:, 1], expected)
            ]
            assert len(found) == 1
        versus = [result.rmst("control", h, versus="treated").psi for h in horizons]
        (guide,) = lines(diff, "cleverly-guide")
        np.testing.assert_allclose(guide.get_xydata()[:, 1], versus)
        assert top.get_title(loc="left") == "Restricted mean survival up to each horizon"
        assert "Restricted mean survival time" in top.get_ylabel().replace("\n", " ")
        assert_style(fig)

    def test_only_pointwise_intervals_are_drawn_and_no_band(self) -> None:
        result = full()
        fig = curve(result, measure="rmst")
        top = fig.axes[0]
        assert not collections(top, "cleverly-ribbon") and not collections(top, "cleverly-band")
        bars = segments(top, "cleverly-interval")
        expected = result.rmst("treated", 3)
        low, high = expected.ci
        assert any(np.isclose(row[1], low) and np.isclose(row[2], high) for row in bars)
        assert "computes no band across horizons" in _core.note_text(fig)
        assert "95% pointwise interval" in key_texts(top)

    def test_a_time_lost_plot_renders(self) -> None:
        fig = curve(full(), measure="rmtl")
        assert fig.axes[0].get_title(loc="left") == "Restricted mean time lost up to each horizon"

    def test_a_fit_that_omits_a_horizon_refuses_with_the_fits_own_reason(self) -> None:
        result = subset()
        with pytest.raises(ValueError) as plotted:
            curve(result, measure="rmst")
        with pytest.raises(ValueError) as direct:
            result.rmst("treated", 3)
        assert "Refit with horizons=None" in str(plotted.value)
        assert "needs the risk at every horizon" in str(direct.value)

    def test_an_unknown_measure_is_refused(self) -> None:
        with pytest.raises(ValueError, match="measure must be"):
            curve(full(), measure="median")


# ---------------------------------------------------------------------------- V4


def _hand_covariance(fit: Any) -> np.ndarray:
    names = list(fit.estimates)
    curves = np.column_stack([np.asarray(fit.estimates[n].influence_curve) for n in names])
    return np.cov(curves, rowvar=False, ddof=1) / curves.shape[0]


class TestWorkingModelMeans:
    def test_an_identity_model_is_the_line_with_the_hand_interval(self) -> None:
        fit = msm_binary("identity")
        fitted = working_model_means(fit)
        beta = np.array([fit.estimates[n].psi for n in fit.estimates])
        sigma = _hand_covariance(fit)
        levels = np.array([0.0, 1.0, 2.0])
        design = np.column_stack([np.ones(3), levels])
        eta = design @ beta
        se = np.sqrt(np.einsum("jp,pq,jq->j", design, sigma, design))
        z = norm.ppf(0.975)
        assert fitted.levels == (0.0, 1.0, 2.0)
        np.testing.assert_allclose(fitted.estimate, eta, rtol=1e-10)
        np.testing.assert_allclose(fitted.lower, eta - z * se, rtol=1e-8)
        np.testing.assert_allclose(fitted.upper, eta + z * se, rtol=1e-8)
        # The first coefficient's own standard error is the intercept row's.
        assert fitted.link_std_error[0] == pytest.approx(
            fit.estimates["msm[(intercept)]"].std_error
        )

    def test_a_logit_model_maps_the_link_scale_interval_through_the_inverse_link(self) -> None:
        fit = msm_binary("logit")
        fitted = working_model_means(fit)
        beta = np.array([fit.estimates[n].psi for n in fit.estimates])
        sigma = _hand_covariance(fit)
        design = np.column_stack([np.ones(3), np.array([0.0, 1.0, 2.0])])
        eta = design @ beta
        se = np.sqrt(np.einsum("jp,pq,jq->j", design, sigma, design))
        z = norm.ppf(0.975)
        np.testing.assert_allclose(fitted.estimate, expit(eta), rtol=1e-10)
        np.testing.assert_allclose(fitted.lower, expit(eta - z * se), rtol=1e-8)
        np.testing.assert_allclose(fitted.upper, expit(eta + z * se), rtol=1e-8)
        # Mutation control: an interval built on the mean scale (delta method) differs, so the
        # assertion above would reject it.
        mean = expit(eta)
        naive_low = mean - z * mean * (1.0 - mean) * se
        assert not np.allclose(fitted.lower, naive_low, rtol=1e-8)
        assert np.all(fitted.lower > 0.0) and np.all(fitted.upper < 1.0)

    def test_the_level_follows_alpha(self) -> None:
        fit = msm_binary("identity")
        wide = working_model_means(fit, alpha=0.01)
        narrow = working_model_means(fit, alpha=0.10)
        assert np.all(wide.upper - wide.lower > narrow.upper - narrow.lower)
        assert wide.alpha == 0.01
        assert set(wide.to_frame().columns) == {"level", "estimate", "ci_lower", "ci_upper"}

    def test_a_design_that_depends_on_modifiers_is_refused(self) -> None:
        fit = msm_binary("identity", modifiers=("W",))
        with pytest.raises(CapabilityError, match="modifier"):
            working_model_means(fit)

    def test_a_fit_without_a_working_model_is_refused(self) -> None:
        with pytest.raises(CapabilityError, match="no working model"):
            working_model_means(level_means())


class TestLevelMeans:
    def test_points_are_the_per_level_means_and_the_model_is_read_back(self) -> None:
        means, msm = level_means(), msm_fit()
        fig = curve(means, msm=msm, treatment_label="Dose level")
        ax = fig.axes[0]
        points = lines(ax, "cleverly-point")
        xy = sorted((float(ln.get_xdata()[0]), float(ln.get_ydata()[0])) for ln in points)
        fitted = working_model_means(msm)
        expected = sorted(
            [(x, means.estimates[f"ey_regime[a{int(x)}]"].psi) for x in (0, 1, 2)]
            + list(zip((0.0, 1.0, 2.0), fitted.estimate, strict=True))
        )
        np.testing.assert_allclose(xy, expected)
        bars = segments(ax, "cleverly-interval")
        model = [(x, lo, hi) for x, lo, hi in bars if np.isclose(lo, fitted.lower[int(x)])]
        assert len(model) == 3
        for x, _lo, hi in model:
            assert hi == pytest.approx(fitted.upper[int(x)])
        assert ax.get_xlabel() == "Dose level"
        assert ax.get_title(loc="left") == "How does the mean outcome change with the dose?"
        assert "95% interval of the working model" in key_texts(ax)
        assert "95% confidence interval" in key_texts(ax)
        assert_style(fig)

    def test_the_fit_of_the_working_model_can_be_the_first_argument(self) -> None:
        fig = curve(msm_fit(), means=level_means())
        assert fig.axes[0].get_title(loc="left").startswith("How does the mean outcome")
        with pytest.raises(ValueError, match="pass means="):
            curve(msm_fit())

    def test_the_means_alone_draw_no_model(self) -> None:
        ax = curve(level_means()).axes[0]
        assert "Working model" not in texts(ax.figure, "cleverly-series-label")
        assert len(lines(ax, "cleverly-point")) == 3

    def test_a_working_model_with_modifiers_is_refused(self) -> None:
        fit = msm_binary("identity", modifiers=("W",))
        with pytest.raises(CapabilityError):
            curve(level_means(), msm=fit)

    def test_levels_without_a_number_are_refused(self) -> None:
        with pytest.raises(ValueError, match="levels="):
            curve(level_means(), levels={"a0": "low", "a1": 1, "a2": 2})

    def test_the_band_marks_are_the_fits_bands_at_its_levels(self) -> None:
        means = level_means()
        ax = curve(means).axes[0]
        marks = segments(ax, "cleverly-band")
        assert len(marks) == 3
        for x, lo, hi in marks:
            bound = means.simultaneous.bands[f"ey_regime[a{int(x)}]"]
            assert (lo, hi) == pytest.approx(bound)
        # Mutation control: without the fit's band nothing is marked.
        bare = replace(means, simultaneous=None)
        assert not collections(curve(bare).axes[0], "cleverly-band")


# ---------------------------------------------------------------------------- V5


class TestPolicies:
    def test_points_are_the_effects_at_each_shift_with_a_null_line(self) -> None:
        result = shifts()
        fig = curve(result)
        ax = fig.axes[0]
        xy = sorted(
            (float(ln.get_xdata()[0]), float(ln.get_ydata()[0]))
            for ln in lines(ax, "cleverly-point")
        )
        expected = sorted(
            (float(key.value.split()[1]), result.estimates[alias].psi)
            for alias, key in result.parameter_keys.items()
        )
        np.testing.assert_allclose(xy, expected)
        (null,) = [ln for ln in ax.lines if ln.get_gid() == "cleverly-reference"]
        assert list(null.get_ydata()) == [0.0, 0.0]
        assert ax.get_title(loc="left") == "How does the effect change as the policy shifts?"
        assert "compared with shift 0" in _subtitle(fig)
        assert "capped at 3.0" in _subtitle(fig)
        assert ax.get_xlabel().startswith("Shift added to the treatment")
        assert_style(fig)

    def test_the_joint_band_is_marked_at_the_policies_of_that_one_fit(self) -> None:
        """Rule 2: bracket marks at the policies, never a ribbon."""
        result = shifts()
        ax = curve(result).axes[0]
        marks = segments(ax, "cleverly-band")
        assert len(marks) == 3
        assert not any(isinstance(c, matplotlib.collections.PolyCollection) for c in ax.collections)
        for (_x, lo, hi), (alias, _key) in zip(
            marks, sorted(result.parameter_keys.items(), key=lambda kv: kv[1].value), strict=True
        ):
            assert (lo, hi) == pytest.approx(result.simultaneous.bands[alias])
        assert "95% simultaneous band (multiplier)" in key_texts(ax)
        assert "experimental" in _core.note_text(ax.figure)
        # Mutation control: with no band on the fit nothing is marked.
        assert not collections(curve(replace(result, simultaneous=None)).axes[0], "cleverly-band")

    def test_incremental_odds_multipliers_use_a_log_axis(self) -> None:
        result = ipsi()
        fig = curve(result)
        ax = fig.axes[0]
        assert ax.get_xscale() == "log"
        assert [t.get_text() for t in ax.get_xticklabels()] == ["0.5", "1", "2", "4"]
        assert ax.get_title(loc="left") == "How does the mean outcome change as the policy shifts?"
        xs = sorted(float(ln.get_xdata()[0]) for ln in lines(ax, "cleverly-point"))
        assert xs == [0.5, 1.0, 2.0, 4.0]
        assert not [ln for ln in ax.lines if ln.get_gid() == "cleverly-reference"]

    def test_a_mixed_family_falls_back_to_the_forest(self) -> None:
        fig = curve(mixed_family())
        assert fig.axes[0].get_title(loc="left") == "Estimated effects"

    def test_an_object_that_is_not_a_result_is_refused(self) -> None:
        with pytest.raises(TypeError, match="curve\\(\\) draws"):
            curve(object())


# ---------------------------------------------------------------------------- V6


class TestEffectCurve:
    def test_the_curve_and_its_interval_are_the_fits_predictions(self) -> None:
        result = cate()
        fig = curve(result, modifier="W", at={"Z": 0.0})
        ax = fig.axes[0]
        (line,) = lines(ax, "cleverly-curve")
        x, y = line.get_xydata().T
        frame = pd.DataFrame({"W": x, "Z": 0.0})
        np.testing.assert_allclose(y, result.predict(frame), rtol=1e-10)
        table = result.predict(frame, interval=True)
        (ribbon,) = collections(ax, "cleverly-ribbon")
        values = ribbon.get_paths()[0].vertices[:, 1]
        assert values.min() == pytest.approx(table["ci_lower"].min())
        assert values.max() == pytest.approx(table["ci_upper"].max())
        assert ax.get_title(loc="left") == "Conditional effect by W"
        assert ax.get_xlabel() == "W"
        assert "Effect on Y" in ax.get_ylabel()
        assert_style(fig)

    def test_the_reference_lines_are_no_effect_and_the_average_of_the_curve(self) -> None:
        result = cate()
        ax = curve(result, modifier="W", at={"Z": 0.0}).axes[0]
        (null,) = [ln for ln in ax.lines if ln.get_gid() == "cleverly-reference"]
        assert list(null.get_ydata()) == [0.0, 0.0]
        (average,) = [ln for ln in ax.lines if ln.get_gid() == "cleverly-average"]
        assert average.get_ydata()[0] == pytest.approx(float(np.mean(result.in_sample)))
        assert "Average of the curve" in key_texts(ax)
        assert "No effect" in key_texts(ax)

    def test_the_band_is_marked_at_its_grid_points_never_as_a_ribbon(self) -> None:
        """Rule 2."""
        result = cate()
        ax = curve(result, modifier="W", at={"Z": 0.0}).axes[0]
        marks = segments(ax, "cleverly-band")
        assert marks[:, 0].tolist() == pytest.approx(result.band.grid["W"].tolist())
        np.testing.assert_allclose(marks[:, 1], result.band.lower)
        np.testing.assert_allclose(marks[:, 2], result.band.upper)
        assert len(collections(ax, "cleverly-ribbon")) == 1
        assert "95% simultaneous band (multiplier)" in key_texts(ax)
        note = _core.note_text(ax.figure)
        assert "No registered study measures its joint coverage" in note
        assert "Brackets mark it at those points only" in note
        # Mutation control: the same curve without a band marks nothing.
        bare = replace(result, band=None)
        assert not collections(curve(bare, modifier="W", at={"Z": 0.0}).axes[0], "cleverly-band")

    def test_a_band_at_other_held_values_is_not_drawn_and_the_note_says_so(self) -> None:
        result = cate()
        fig = curve(result, modifier="W", at={"Z": 0.5})
        assert not collections(fig.axes[0], "cleverly-band")
        assert "other values of the held modifiers" in _core.note_text(fig)

    def test_the_rug_is_one_tick_per_fitted_row(self) -> None:
        result = cate()
        ax = curve(result, modifier="W", at={"Z": 0.0}).axes[0]
        (rug,) = collections(ax, "cleverly-rug")
        ticks = np.array([seg[0][0] for seg in rug.get_segments()])
        np.testing.assert_allclose(np.sort(ticks), np.sort(_modifier_frame()["W"].to_numpy()))

    def test_other_modifiers_the_curve_reads_must_be_held(self) -> None:
        with pytest.raises(ValueError, match=r"hold each at a value with at="):
            curve(cate(), modifier="W")
        with pytest.raises(ValueError, match="pass modifier="):
            curve(cate())
        with pytest.raises(ValueError, match="not modifiers"):
            curve(cate(), modifier="W", at={"Z": 0.0, "nope": 1.0})
        with pytest.raises(ValueError, match="continuous modifier"):
            curve(cate(), modifier="G", at={"Z": 0.0})

    def test_a_categorical_second_modifier_gives_small_multiples(self) -> None:
        result = cate_by()
        fig = curve(result, modifier="W", by="G")
        assert len(fig.axes) == 2
        first, second = fig.axes
        assert first.get_ylim() == pytest.approx(second.get_ylim())
        for panel, level in zip(fig.axes, ("f", "m"), strict=True):
            (own,) = lines(panel, "cleverly-curve")
            frame = pd.DataFrame({"W": own.get_xdata(), "G": level})
            np.testing.assert_allclose(own.get_ydata(), result.predict(frame), rtol=1e-10)
            context = lines(panel, "cleverly-context")
            assert len(context) == 1 and context[0].get_color() == _core.CONTEXT
        assert texts(fig, "cleverly-series-label") == ["G = f", "G = m"]
        with pytest.raises(ValueError, match="own figure"):
            curve(result, modifier="W", by="G", ax=plt.subplots()[1])
        with pytest.raises(ValueError, match="not a categorical modifier"):
            curve(result, modifier="W", by="Z")

    def test_a_log_ratio_curve_is_drawn_as_a_ratio_on_a_log_axis_around_one(self) -> None:
        result = log_ratio_curve()
        ax = curve(result, modifier="W").axes[0]
        assert ax.get_yscale() == "log"
        (null,) = [ln for ln in ax.lines if ln.get_gid() == "cleverly-reference"]
        assert list(null.get_ydata()) == [1.0, 1.0]
        (line,) = lines(ax, "cleverly-curve")
        x, y = line.get_xydata().T
        np.testing.assert_allclose(y, np.exp(result.predict({"W": x})))
        assert "risk ratio" in ax.get_ylabel()

    def test_an_ep_curve_has_no_interval_and_says_so(self) -> None:
        """Rule 1."""
        result = ep_curve()
        fig = curve(result)
        ax = fig.axes[0]
        assert not collections(ax, "cleverly-ribbon")
        assert not collections(ax, "cleverly-band")
        (line,) = lines(ax, "cleverly-curve")
        x, y = line.get_xydata().T
        np.testing.assert_allclose(y, result.predict({"W": x}))
        assert "carries no interval" in _core.note_text(fig)
        # Mutation control: the projection curve draws its ribbon.
        assert collections(curve(cate(), modifier="W", at={"Z": 0.0}).axes[0], "cleverly-ribbon")

    def test_an_experimental_curve_says_so(self) -> None:
        """Rule 3."""
        result = cate()
        assert result.status == "experimental"
        fig = curve(result, modifier="W", at={"Z": 0.0})
        assert "experimental" in _core.note_text(fig)
        calm = replace(result, status="supported")
        assert "marks the curve experimental" not in _core.note_text(
            curve(calm, modifier="W", at={"Z": 0.0})
        )

    def test_plot_is_the_curve(self) -> None:
        fig = ep_curve().plot()
        assert fig.axes[0].get_title(loc="left") == "Conditional effect by W"

    def test_a_png_renders(self, tmp_path: Any) -> None:
        for name, fig in (
            ("cate", curve(cate(), modifier="W", at={"Z": 0.0})),
            ("by", curve(cate_by(), modifier="W", by="G")),
            ("ep", curve(ep_curve())),
            ("means", curve(level_means(), msm=msm_fit())),
            ("shifts", curve(shifts())),
            ("rmst", curve(full(), measure="rmst")),
        ):
            fig.savefig(tmp_path / f"{name}.png", dpi=60)
            assert (tmp_path / f"{name}.png").stat().st_size > 1000
