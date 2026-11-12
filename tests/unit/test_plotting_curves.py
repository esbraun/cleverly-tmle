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
    ATE,
    CausalStudy,
    ConditionalContrast,
    CounterfactualMean,
    CrossFitting,
    EPMethod,
    IncrementalMean,
    LongitudinalTreatment,
    ModelSpec,
    ModifiedTreatmentPolicyEffect,
    MSMProjection,
    PointTreatment,
    RegimeContrast,
    RegimeMean,
    Runtime,
    TMLEMethod,
)
from cleverly._refusal import RefusalReason
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
from cleverly.inference.influence import minimum_reference_df
from cleverly.interventions import Incremental, Scale, Shift, Static
from cleverly.longitudinal import LTMLE, LongitudinalData, Repeated, SequentialGComputation
from cleverly.msm import MSM, working_model_means
from cleverly.plotting import _core, _curves, curve
from cleverly.plotting import _forest as forest_module
from cleverly.plotting._curves import _Point, _select
from cleverly.validation.longitudinal import _longitudinal_stagewise
from tests.unit.test_plotting_core import assert_no_note_text

pytestmark = pytest.mark.xdist_group("plotting_curves")

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
def regime_contrast() -> Any:
    """A repeated outcome fitted as a regime contrast: differences at each week, no levels."""
    frame, _ = make_longitudinal_repeated(n=300, seed=0)
    design = LongitudinalTreatment(
        outcome=Repeated(["Y1", "Y2", "Y3"]),
        treatment=("A1", "A2", "A3"),
        baseline=("W",),
        time_varying=((), ("L2",), ("L3",)),
        censoring=("C1", "C2", "C3"),
    )
    logistic = LogisticRegression(max_iter=500)
    method = TMLEMethod(
        models=ModelSpec(
            outcome_learner=logistic,
            pseudo_learner=LinearRegression(),
            treatment_learner=logistic,
            censoring_learner=logistic,
        ),
        cross_fitting=CrossFitting(enabled=False),
        runtime=Runtime(random_state=0, n_jobs=1),
    )
    plans = {"never": (0, 0, 0), "always": (1, 1, 1), "first": (1, 0, 0)}
    effect = CausalStudy(frame, design=design).identify(RegimeContrast(plans, reference="never"))
    return _quiet(effect.estimate, method=method)


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


#: The number each labelled arm stands for, as a notebook declares it.
CONTACTS = {"low": 1.0, "medium": 2.0, "high": 6.0}


@cache
def _labelled() -> Any:
    """Three arms labelled by name, not by number."""
    frame, _ = make_multi_arm(n=400, seed=0)
    return CausalStudy(
        frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3"))
    )


@cache
def arm_means() -> Any:
    """A fit of per-arm means: keys on the ``"arm"`` axis, ``ey[low]`` and so on."""
    return _quiet(_labelled().identify(CounterfactualMean()).estimate, **_ESTIMATE)


def _contacts_design(arm: Any, data: Any) -> np.ndarray:
    return np.column_stack([np.ones(len(data)), np.full(len(data), CONTACTS[arm])])


@cache
def labelled_msm() -> Any:
    model = MSM(design=_contacts_design, terms=("(intercept)", "contacts"), design_kind="known")
    return _quiet(_labelled().identify(MSMProjection(model)).estimate, **_ESTIMATE)


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
def msm_clustered() -> Any:
    rng = np.random.default_rng(0)
    a = rng.integers(0, 3, 400).astype(float)
    frame = pd.DataFrame({"A": a, "W": rng.normal(size=400), "cl": rng.integers(0, 8, 400)})
    frame["Y"] = rng.binomial(1, expit(-0.5 + 0.6 * a + 0.3 * frame["W"])).astype(float)
    design = PointTreatment(outcome="Y", treatment="A", adjustment=("W",), cluster="cl")
    study = CausalStudy(frame, design=design)
    return _quiet(study.identify(MSMProjection(MSM.linear())).estimate, **_BINARY)


@cache
def msm_repeats() -> Any:
    settings = {**_BINARY, "cross_fit": True, "n_folds": 3, "repeats": 3, "simultaneous": False}
    return _quiet(_binary_study().identify(MSMProjection(MSM.linear())).estimate, **settings)


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


def policy_fit(policies: list[Any], reference: str | None = None) -> Any:
    estimand = (
        ModifiedTreatmentPolicyEffect(policies=policies)
        if reference is None
        else ModifiedTreatmentPolicyEffect(policies=policies, reference=reference)
    )
    return _quiet(_dose_study().identify(estimand).estimate, **_ESTIMATE)


@cache
def middle_reference() -> Any:
    return policy_fit([Shift(d, cap=3.0, name=f"s{d:g}") for d in (0.0, 0.5, 1.0)], "s0.5")


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


def _effect(
    model: Any, modifiers: Modifiers = _MODS, weights: str | None = None, **kwargs: Any
) -> Any:
    frame = _modifier_frame().assign(wt=np.random.default_rng(3).uniform(0.2, 3.0, 500))
    design = PointTreatment(outcome="Y", treatment="A", adjustment=("W", "Z", "G"), weights=weights)
    study = CausalStudy(frame, design=design)
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
def no_intercept_curve() -> Any:
    return _quiet(_effect(Linear(["W"], intercept=False)))


@cache
def intercept_curve() -> Any:
    return _quiet(_effect(Linear(["W"])))


@cache
def weighted_curve() -> Any:
    return _quiet(_effect(Linear(["W"]), weights="wt"))


@cache
def multi_arm_curve() -> Any:
    frame, _ = make_multi_arm(n=400, seed=0)
    study = CausalStudy(
        frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3"))
    )
    contrast = ConditionalContrast(
        Modifiers(["W1"]), model=Linear(["W1"]), treatment="high", reference="low"
    )
    return _quiet(study.identify(contrast).estimate, **_ESTIMATE).effect_curve()


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
    """Every ``(x, low, high)`` of the vertical bars drawn under ``gid``."""
    found = [
        (seg[0][0], seg[0][1], seg[1][1])
        for item in collections(ax, gid)
        if isinstance(item, matplotlib.collections.LineCollection)
        for seg in item.get_segments()
    ]
    return np.array(sorted(found))


def bars(ax: Any) -> np.ndarray:
    """``(x, low, high)`` of each pointwise interval drawn as a bar."""
    return segments(ax, _core.GID_INTERVAL)


def band_marks(ax: Any) -> np.ndarray:
    """``(x, low, high)`` of each simultaneous-band bracket pair (:func:`_core.brackets`)."""
    found = []
    for line in lines(ax, _core.GID_BAND):
        x, y = line.get_xdata(), line.get_ydata()
        half = len(x) // 2
        found.extend(zip(x[:half], y[:half], y[half:], strict=True))
    return np.array(sorted(found))


def ribbons(ax: Any, gid: str) -> list[Any]:
    """The ribbons (filled polygons) drawn under ``gid``."""
    return filled(ax, gid)


def _last_node_visible(ax: Any, x: float, low: float, high: float, gid: str) -> bool:
    """Whether the limits ``low`` and ``high`` at the last node ``x`` have a visible mark.

    A bar or a bracket pair at ``x``, or a ribbon that reaches past ``x`` at those limits.
    A ribbon that stops at ``x`` gives the last node zero width and fails.
    """
    for item in collections(ax, gid):
        if isinstance(item, matplotlib.collections.LineCollection):
            for seg in item.get_segments():
                ends = sorted([seg[0][1], seg[1][1]])
                if np.isclose(seg[0][0], x) and np.allclose(ends, [low, high]):
                    return True
        elif isinstance(item, matplotlib.collections.PolyCollection):
            for path in item.get_paths():
                beyond = path.vertices[path.vertices[:, 0] > x + 1e-9]
                if (
                    beyond.size
                    and np.isclose(beyond[:, 1], low).any()
                    and np.isclose(beyond[:, 1], high).any()
                ):
                    return True
    for line in lines(ax, gid):
        xs, ys = line.get_xdata(), line.get_ydata()
        half = len(xs) // 2
        for at, lo, hi in zip(xs[:half], ys[:half], ys[half:], strict=True):
            if np.isclose(at, x) and np.allclose([lo, hi], [low, high]):
                return True
    return False


def texts(fig: Any, gid: str) -> list[str]:
    return [t.get_text() for ax in fig.axes for t in ax.texts if t.get_gid() == gid]


def key_texts(ax: Any) -> list[str]:
    legend = ax.get_legend()
    return [] if legend is None else [t.get_text() for t in legend.get_texts()]


def frame_rows(result: Any, scale: str, regimen: str, *, level: bool = True) -> pd.DataFrame:
    table = result.curve(scale=scale)
    table = table[(table["regimen"] == regimen) & ((table["scale"] == "level") == level)]
    return table.sort_values("time")


def _is_grey(color: str) -> bool:
    import colorsys

    red, green, blue = matplotlib.colors.to_rgb(color)
    return colorsys.rgb_to_hsv(red, green, blue)[1] < 0.2


def _colors(artist: Any) -> Any:
    if hasattr(artist, "get_edgecolor") and hasattr(artist, "get_facecolor"):
        faces = np.atleast_2d(artist.get_facecolor())
        edges = np.atleast_2d(artist.get_edgecolor())
        return np.vstack([faces, edges]) if faces.size and edges.size else faces
    found = [matplotlib.colors.to_rgba(artist.get_color())]
    for getter in ("get_markeredgecolor", "get_markerfacecolor"):
        value = getattr(artist, getter, lambda: "none")()
        if not (isinstance(value, str) and value == "none"):
            found.append(matplotlib.colors.to_rgba(value))
    return np.array(found)


def assert_style(fig: Any) -> None:
    """The style contract every figure meets, read from every artist.

    White face, a title, no overlaid axes, a y label or row labels on every axes, an x label
    on the bottom axes, and every coloured artist (lines, markers, collections, patches) in
    a palette colour, at most three per axes. No figure draws a note.
    """
    assert_no_note_text(fig)
    assert matplotlib.colors.to_hex(fig.get_facecolor(), keep_alpha=True) == "#ffffffff"
    positions = [tuple(np.round(ax.get_position().bounds, 6)) for ax in fig.axes]
    assert len(set(positions)) == len(positions), "two axes overlay each other"
    assert fig.axes[0].get_title(loc="left")
    fig.canvas.draw()
    for ax in fig.axes:
        ticks = [t.get_text() for t in ax.get_yticklabels()]
        assert ax.get_ylabel() or any(ticks), "an axes with neither a y label nor row labels"
        assert ax.get_xlabel() or ax.get_shared_x_axes().get_siblings(ax) != [ax]
        used = set()
        for artist in [*ax.lines, *ax.collections, *ax.patches]:
            for color in np.atleast_2d(_colors(artist)):
                if color.size == 4 and color[3] == 0.0:
                    continue
                hexed = matplotlib.colors.to_hex(color)
                if _is_grey(hexed):
                    continue
                assert hexed in _core.PALETTE, f"{hexed} is not a palette colour"
                used.add(hexed)
        assert len(used) <= _core.MAX_SERIES
    assert fig.axes[-1].get_xlabel()


def assert_text_inside(fig: Any) -> None:
    """Every text the figure writes (title, subtitle, key, labels, notes) lies on the page."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    found = []
    for ax in fig.axes:
        found.extend(ax.texts)
        found.extend([ax.title, ax._left_title, ax.xaxis.label, ax.yaxis.label])
        if ax.get_legend() is not None:
            found.extend(ax.get_legend().get_texts())
    for text in found:
        if not text.get_visible() or not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer)
        assert box.x0 >= fig.bbox.x0 - 1 and box.x1 <= fig.bbox.x1 + 1, text.get_text()
        assert box.y0 >= fig.bbox.y0 - 1 and box.y1 <= fig.bbox.y1 + 1, text.get_text()


def all_figures() -> list[Any]:
    return [
        curve(full()),
        curve(full(), scale="survival"),
        curve(subset()),
        curve(isotonic()),
        curve(withheld()),
        curve(full(), measure="rmst", units="time units"),
        curve(full(), measure="rmtl"),
        curve(level_means(), msm=msm_fit(), treatment_label="Dose level"),
        curve(shifts()),
        curve(middle_reference()),
        curve(ipsi()),
        curve(cate(), modifier="W", at={"Z": 0.0}),
        curve(cate_by(), modifier="W", by="G"),
        curve(log_ratio_curve(), modifier="W"),
        curve(ep_curve()),
        curve(mixed_family()),
        curve(multi_arm_curve()),
        curve(repeated()),
        curve(regime_contrast()),
    ]


class TestStyle:
    def test_every_figure_passes_the_style_checks_with_its_text_on_the_page(self) -> None:
        for fig in all_figures():
            assert_style(fig)
            assert_text_inside(fig)
            plt.close(fig)

    def test_the_style_checks_reject_an_off_palette_or_cluttered_figure(self) -> None:
        """Control: the same checks fail on a figure that breaks the contract."""
        fig = curve(full())
        fig.axes[0].plot([1, 2], [0.1, 0.2], color="#1f77b4")
        with pytest.raises(AssertionError, match="not a palette colour"):
            assert_style(fig)
        fig = curve(full())
        fig.axes[0].set_ylabel("")
        fig.axes[0].set_yticks([])
        with pytest.raises(AssertionError, match="neither a y label"):
            assert_style(fig)

    def test_the_text_check_rejects_a_label_off_the_page(self) -> None:
        """Control: an end label pushed past the right edge fails the same check."""
        fig = curve(level_means(), msm=msm_fit())
        ax = fig.axes[0]
        # Freeze the layout first: constrained layout would otherwise shrink the axes until
        # the stray text fits. Axes coordinates keep the control off the data limits.
        fig.canvas.draw()
        fig.set_layout_engine("none")
        ax.text(1.6, 0.5, "a label off the page", transform=ax.transAxes)
        with pytest.raises(AssertionError, match="off the page"):
            assert_text_inside(fig)


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
            "95% simultaneous band",
            "No difference",
        ] or set(key_texts(top)) == {
            "95% pointwise interval",
            "95% simultaneous band",
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

    def test_the_survival_band_mirrors_the_risk_band_in_both_panels(self) -> None:
        """Rule 5: a level's band is [1 - hi, 1 - lo]; a contrast's is [-hi, -lo]."""
        result = full()
        fig = curve(result, scale="survival")
        top, diff, _ = fig.axes
        bands = result.simultaneous.bands

        def drawn(ax: Any) -> list[set[float]]:
            return [
                set(np.round(r.get_paths()[0].vertices[:, 1], 9))
                for r in ribbons(ax, _core.GID_BAND)
            ]

        for regimen in ("treated", "control"):
            risk = frame_rows(result, "risk", regimen)
            limits = np.array([bands[p] for p in risk["parameter"]])
            expected = set(np.round(np.r_[1.0 - limits[:, 1], 1.0 - limits[:, 0]], 9))
            assert expected in drawn(top)
        contrast = frame_rows(result, "risk", "control vs treated", level=False)
        limits = np.array([bands[p] for p in contrast["parameter"]])
        expected = set(np.round(np.r_[-limits[:, 1], -limits[:, 0]], 9))
        (ribbon,) = drawn(diff)
        assert ribbon == expected
        # The last node's band is the bracket pair, mirrored the same way.
        marks = band_marks(diff)
        assert marks[-1, 0] == 4.0
        np.testing.assert_allclose(marks[-1, 1:], [-limits[-1, 1], -limits[-1, 0]])
        # Mutation control: the wrong map for a contrast, 1 - x, is not what is drawn.
        wrong = set(np.round(np.r_[1.0 - limits[:, 1], 1.0 - limits[:, 0]], 9))
        assert ribbon != wrong

    def test_step_ribbons_when_the_fit_reports_every_node(self) -> None:
        result = full()
        top = curve(result).axes[0]
        found = ribbons(top, _core.GID_INTERVAL)
        assert len(found) == 2
        risk = frame_rows(result, "risk", "treated")
        expected = set(np.round(np.r_[risk["ci_lower"], risk["ci_upper"]], 9))
        drawn = [set(np.round(r.get_paths()[0].vertices[:, 1], 9)) for r in found]
        assert expected in drawn
        # The ribbon is constant between nodes: the outline has a vertical edge at each node.
        vertices = found[0].get_paths()[0].vertices
        assert {1.0, 2.0, 3.0, 4.0} <= set(np.round(vertices[:, 0], 9))

    def test_the_last_node_shows_its_interval_and_band(self) -> None:
        """A step ribbon has no width at the last node; a bar and brackets mark it there."""
        result = full()
        fig = curve(result)
        top, diff, _ = fig.axes
        for regimen in ("treated", "control"):
            last = frame_rows(result, "risk", regimen).iloc[-1]
            band = result.simultaneous.bands[last["parameter"]]
            assert _last_node_visible(
                top, 4.0, last["ci_lower"], last["ci_upper"], _core.GID_INTERVAL
            )
            assert _last_node_visible(top, 4.0, *band, _core.GID_BAND)
        last = frame_rows(result, "risk", "control vs treated", level=False).iloc[-1]
        assert _last_node_visible(diff, 4.0, last["ci_lower"], last["ci_upper"], _core.GID_INTERVAL)
        assert "the interval is drawn as a bar and the band as brackets" in _core.note_text(fig)

    def test_the_last_node_check_rejects_a_bare_step_ribbon(self, monkeypatch: Any) -> None:
        """Control: the plain ``fill_between(step="post")`` ribbon fails the same check."""
        result = full()
        last = frame_rows(result, "risk", "treated").iloc[-1]
        monkeypatch.setattr(_core, "brackets", lambda *args, **kwargs: None)
        top = curve(result).axes[0]
        assert not _last_node_visible(
            top, 4.0, last["ci_lower"], last["ci_upper"], _core.GID_INTERVAL
        )

    def test_the_core_step_ribbon_marks_its_last_node(self) -> None:
        _, ax = plt.subplots()
        _core.ribbon(
            ax, [1, 2, 3], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6], color="#2a78d6", label="x", step=True
        )
        assert _last_node_visible(ax, 3.0, 0.3, 0.6, _core.GID_INTERVAL)
        _core.ribbon(
            ax,
            [1, 2, 3],
            [0.0, 0.1, 0.2],
            [0.7, 0.8, 0.9],
            color="#2a78d6",
            label="y",
            band=True,
            step=True,
        )
        assert _last_node_visible(ax, 3.0, 0.2, 0.9, _core.GID_BAND)
        # Control: a ribbon over a smooth grid is not a step, and adds no mark.
        _, plain = plt.subplots()
        _core.ribbon(plain, [1, 2, 3], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6], color="#2a78d6", label="x")
        assert not bars(plain).size

    def test_a_missing_interval_leaves_a_gap_and_says_so(self) -> None:
        """A node with no finite interval drops its own step only, and its marker is hollow."""
        result = full()
        name = "risk_regimen[treated @ t=3]"
        broken = replace(result.estimates[name], variance=float("nan"))
        assert broken.supplies_inference
        gap = replace(result, estimates={**result.estimates, name: broken})
        fig = curve(gap)
        top = fig.axes[0]
        risk = frame_rows(result, "risk", "treated")
        held = risk.iloc[1]  # t = 2, which holds until t = 3
        outlines = [r.get_paths() for r in ribbons(top, _core.GID_INTERVAL)]
        covered = [
            path.vertices
            for paths in outlines
            for path in paths
            if np.isclose(path.vertices[:, 1], held["ci_upper"]).any()
        ]
        assert covered and max(v[:, 0].max() for v in covered) == pytest.approx(3.0)
        hollow = [
            tuple(line.get_xydata()[0])
            for line in lines(top, _core.GID_POINT)
            if line.get_markerfacecolor() == _core.FACE
        ]
        assert [x for x, _ in hollow] == [3.0]
        assert "no finite interval at time 3" in _core.note_text(fig)
        # Mutation control: the intact fit draws no hollow marker and no gap note.
        intact = curve(result)
        assert not [
            line
            for line in lines(intact.axes[0], _core.GID_POINT)
            if line.get_markerfacecolor() == _core.FACE
        ]
        assert "no finite interval" not in _core.note_text(intact)

    def test_a_declared_subset_of_horizons_draws_points_and_brackets_only(self) -> None:
        """Rule 2's exception: no step artist between declared horizons."""
        result = subset()
        fig = curve(result)
        top = fig.axes[0]
        assert not ribbons(top, _core.GID_INTERVAL)
        assert not ribbons(top, _core.GID_BAND)
        assert not any(ln.get_drawstyle().startswith("steps") for ln in top.lines)
        assert not lines(top, "cleverly-curve")
        points = sorted(float(ln.get_xdata()[0]) for ln in lines(top, _core.GID_POINT))
        assert points == [2.0, 2.0, 4.0, 4.0]
        assert sorted(set(bars(top)[:, 0])) == [2.0, 4.0]
        assert sorted(set(band_marks(top)[:, 0])) == [2.0, 4.0]
        assert "horizons 2, 4" in _core.note_text(fig)
        # Mutation control: the fit that reports every node does draw the step artists.
        every = curve(full()).axes[0]
        assert ribbons(every, _core.GID_INTERVAL) and ribbons(every, _core.GID_BAND)
        assert lines(every, "cleverly-curve")

    def test_the_band_is_the_one_the_fit_computed_and_wider_than_the_interval(self) -> None:
        result = full()
        fig = curve(result)
        bands = result.simultaneous.bands
        top = fig.axes[0]
        (outer, _) = sorted(
            ribbons(top, _core.GID_BAND),
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
        assert not ribbons(top, _core.GID_BAND) and not band_marks(top).size
        assert ribbons(top, _core.GID_INTERVAL)
        assert "95% pointwise interval" in key_texts(top)
        assert "95% simultaneous band" not in key_texts(top)
        # Mutation control: the fit with the band draws one.
        assert ribbons(curve(result).axes[0], _core.GID_BAND)

    def test_the_band_kind_and_the_other_band_are_named(self) -> None:
        result = boot()
        both = curve(result)
        assert "95% simultaneous band" in key_texts(both.axes[0])
        assert "band drawn is the multiplier band" in _core.note_text(both)
        assert "also holds a bootstrap sup-t" in _core.note_text(both)
        chosen = curve(result, band="bootstrap")
        assert "95% simultaneous band" in key_texts(chosen.axes[0])
        assert "band drawn is the bootstrap sup-t band" in _core.note_text(chosen)
        assert "band='multiplier'" in _core.note_text(chosen)
        only = curve(replace(result, simultaneous=None))
        assert "band drawn is the bootstrap sup-t band" in _core.note_text(only)
        # The kind is a note, never a word in the key.
        assert not any("sup-t" in text for text in key_texts(chosen.axes[0]))
        with pytest.raises(ValueError, match="no bootstrap band"):
            curve(full(), band="bootstrap")
        assert not ribbons(curve(result, band="none").axes[0], _core.GID_BAND)

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
        assert not collections(top, _core.GID_INTERVAL) and not collections(top, _core.GID_BAND)
        assert "No interval is drawn" in _core.note_text(fig)
        # Rule 1's marker: a point without an interval is hollow, as in the forest plot.
        points = lines(top, _core.GID_POINT)
        assert points and all(line.get_markerfacecolor() == _core.FACE for line in points)
        # Mutation control: the fit with inference draws the ribbon and filled points.
        every = curve(full()).axes[0]
        assert ribbons(every, _core.GID_INTERVAL)
        assert all(
            line.get_markerfacecolor() != _core.FACE for line in lines(every, _core.GID_POINT)
        )

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

    def test_the_isotonic_note_appears_only_with_its_markers(self) -> None:
        result = isotonic()
        treated_only = tuple(move for move in result.isotonic_moves if move[0] == "treated")
        assert treated_only
        repeat = replace(result.repeats[0], isotonic_moves=treated_only)
        moved = replace(result, repeats=(repeat, *result.repeats[1:]))
        quiet = curve(moved, regimens=["control"])
        assert not lines(quiet.axes[0], "cleverly-raw")
        assert "Hollow markers" not in _core.note_text(quiet)
        # Mutation control: the regimen the projection moved draws both.
        shown = curve(moved, regimens=["treated"])
        assert lines(shown.axes[0], "cleverly-raw")
        assert "Hollow markers" in _core.note_text(shown)

    def test_a_fit_of_several_repeats_marks_no_raw_values_and_says_why(self) -> None:
        single = isotonic()
        combined = replace(single, repeats=(single.repeats[0], single.repeats[0]))
        assert combined.n_repeats == 2 and combined.isotonic_moves
        fig = curve(combined)
        assert not lines(fig.axes[0], "cleverly-raw")
        assert "one of the 2 repeats" in _core.note_text(fig)
        assert "Hollow markers" not in _core.note_text(fig)
        # Mutation control: the one-repeat fit marks them.
        assert lines(curve(single).axes[0], "cleverly-raw")

    def test_a_regimen_keeps_its_colour_across_the_curve_and_the_restricted_means(self) -> None:
        """Plan 4.2: one colour per regimen across every plot of one result."""
        result = full()
        order = [key[0] for key in result._levels()]
        # The fit's own order is not alphabetical, so a sorted order would swap colours.
        assert list(dict.fromkeys(order)) != sorted(set(order))
        found = []
        for fig in (curve(result), curve(result, measure="rmst"), curve(result, measure="rmtl")):
            found.append(_label_colours(fig.axes[0]))
        assert found[0] == found[1] == found[2]
        assert set(found[0]) == {"treated", "control"}
        # Mutation control: colouring the regimens in sorted order gives another map.
        ordered = sorted(found[0])
        sorted_map = {name: _core.PALETTE[i] for i, name in enumerate(ordered)}
        assert sorted_map != found[0]

    def test_plot_of_a_fit_without_a_curve_refuses_curve_options(self) -> None:
        with pytest.raises(ValueError, match="has no curve"):
            end_of_study().plot(scale="survival")
        with pytest.raises(ValueError, match="measure"):
            end_of_study().plot(measure="rmst")
        # Mutation control: forest options draw the forest.
        fig = end_of_study().plot(title="Effects at the end of the study")
        assert fig.axes[0].get_title(loc="left") == "Effects at the end of the study"

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

    def test_a_fit_of_contrasts_alone_draws_its_difference_panel(self) -> None:
        """A regime contrast reports no level, so the figure is the difference panel alone."""
        result = regime_contrast()
        assert not any(estimate.scale == "level" for estimate in result.estimates.values())
        fig = curve(result)
        diff, counts = fig.axes
        assert diff.get_title(loc="left") == "Difference in mean outcome over time"
        assert diff.get_ylabel() == "Difference in mean outcome"
        for contrast in ("always vs never", "first vs never"):
            expected = frame_rows(result, "mean", contrast, level=False)
            drawn = [
                line
                for line in lines(diff, "cleverly-curve")
                if np.allclose(line.get_xydata()[:, 1], expected["psi"].to_numpy())
            ]
            assert len(drawn) == 1
            # The fit's own pointwise limits are drawn, as ribbons over its weekly steps.
            low = expected["ci_lower"].to_numpy()
            high = expected["ci_upper"].to_numpy()
            assert _last_node_visible(diff, 3.0, low[-1], high[-1], _core.GID_INTERVAL)
        assert len(lines(diff, "cleverly-curve")) == 2
        assert [t.get_text() for t in counts.get_yticklabels()] == ["never", "always", "first"]
        assert "only the differences between regimens" in _core.note_text(fig)
        assert "No difference" in key_texts(diff)
        assert_style(fig)
        assert_text_inside(fig)

    def test_a_fit_of_contrasts_alone_selects_regimens_in_pairs(self) -> None:
        result = regime_contrast()
        fig = curve(result, regimens=["always", "never"])
        diff, counts = fig.axes
        assert len(lines(diff, "cleverly-curve")) == 1
        assert [t.get_text() for t in counts.get_yticklabels()] == ["always", "never"]
        with pytest.raises(ValueError, match="selects no contrast of this fit"):
            curve(result, regimens=["always", "first"])
        with pytest.raises(ValueError, match="unknown regimen"):
            curve(result, regimens=["sometimes"])

    def test_a_fit_with_levels_keeps_its_level_panel_and_no_contrast_only_note(self) -> None:
        """Control: the contrast-only layout is chosen by the fit, not applied to every fit."""
        fig = curve(repeated())
        assert len(fig.axes) == 3
        assert fig.axes[0].get_title(loc="left") == "Mean outcome over time"
        assert "only the differences" not in _core.note_text(fig)

    def test_series_at_the_last_time_point_are_dodged_apart_but_keep_their_times(self) -> None:
        """Several regimens' marks at one time point sit side by side, wider than a bracket.

        The shift is a display transform only: the drawn data keep the fit's own times.
        """
        result = full()
        fig = curve(result)
        top = fig.axes[0]
        fig.canvas.draw()
        last = float(frame_rows(result, "risk", "treated")["time"].max())
        points = [ln for ln in lines(top, _core.GID_POINT) if ln.get_xdata()[0] == last]
        assert len(points) == 2
        assert all(float(ln.get_xdata()[0]) == last for ln in points)
        screen = sorted(float(ln.get_transform().transform(ln.get_xydata())[0][0]) for ln in points)
        bracket = 12.0 * fig.dpi / 72.0
        assert screen[1] - screen[0] > bracket
        # The band brackets and the bars at that node are dodged with their series.
        for gid in (_core.GID_BAND, _core.GID_INTERVAL):
            xs = set()
            marks = [
                c
                for c in collections(top, gid)
                if isinstance(c, matplotlib.collections.LineCollection)
            ]
            for artist in [*lines(top, gid), *marks]:
                data = (
                    artist.get_xydata()
                    if hasattr(artist, "get_xydata")
                    else np.vstack(artist.get_segments())
                )
                at_last = data[np.isclose(data[:, 0], last)]
                if at_last.size:
                    shown = artist.get_transform().transform(at_last)
                    xs.add(round(float(shown[0, 0]), 3))
            assert len(xs) == 2, gid
            assert max(xs) - min(xs) > bracket
        # The fit's frame, which the plot reads, still reports the true time point.
        assert last in set(result.curve(scale="risk")["time"])

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
        assert forest.axes[0].get_title(loc="left") == forest_module.TITLE

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


def _label_colours(ax: Any) -> dict[str, str]:
    """Each direct end label's text and the colour of its swatch."""
    swatches = [t for t in ax.texts if t.get_gid() == _core.GID_SWATCH]
    names = [t for t in ax.texts if t.get_gid() == _core.GID_SERIES_LABEL]
    return {
        name.get_text(): matplotlib.colors.to_hex(swatch.get_color())
        for swatch, name in zip(swatches, names, strict=True)
    }


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

    def test_a_clustered_fit_reads_the_t_reference_of_its_coefficients(self) -> None:
        """Plan 5.3: the fit's own reference distribution, here t on few clusters."""
        fit = msm_clustered()
        fitted = working_model_means(fit)
        intercept = fit.estimates["msm[(intercept)]"]
        assert intercept.reference_df is not None
        assert fitted.reference_df == minimum_reference_df(list(fit.estimates.values()))
        # At level 0 the identity model's mean is the intercept, so the limits are its CI.
        np.testing.assert_allclose([fitted.lower[0], fitted.upper[0]], intercept.ci, rtol=1e-12)
        # Mutation control: the normal quantile gives narrower limits, which the check rejects.
        z = norm.ppf(0.975)
        normal = intercept.psi - z * fitted.link_std_error[0]
        assert not np.isclose(normal, fitted.lower[0], rtol=1e-6)

    def test_a_fit_of_several_repeats_is_refused_with_its_own_reason(self) -> None:
        fit = msm_repeats()
        assert fit.n_repeats == 3
        with pytest.raises(CapabilityError, match="joint covariance of its coefficients") as read:
            working_model_means(fit)
        assert "covariance()" not in str(read.value)
        with pytest.raises(CapabilityError, match="joint covariance of its coefficients"):
            curve(level_means(), msm=fit)
        # Mutation control: the reason is not the one covariance() gives.
        with pytest.raises(Exception) as direct:
            fit.covariance()
        assert str(direct.value) != str(read.value)


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
        model = [(x, lo, hi) for x, lo, hi in bars(ax) if np.isclose(lo, fitted.lower[int(x)])]
        assert len(model) == 3
        for x, _lo, hi in model:
            assert hi == pytest.approx(fitted.upper[int(x)])
        assert ax.get_xlabel() == "Dose level"
        assert ax.get_title(loc="left") == ("Mean outcome by treatment level")
        # Plan 4.2: the interval kind is "pointwise" beside a band, and the series are
        # named by direct labels at their ends, not by a second key.
        assert key_texts(ax) == ["95% pointwise interval", "95% simultaneous band"]
        assert set(_label_colours(ax)) == {"Mean at each level", "Working model"}
        assert_style(fig)

    def test_the_guide_line_runs_through_the_dodged_markers(self) -> None:
        fig = curve(level_means(), msm=msm_fit())
        ax = fig.axes[0]
        fig.canvas.draw()
        guides = lines(ax, "cleverly-guide")
        points = lines(ax, _core.GID_POINT)
        assert len(guides) == 2
        for guide in guides:
            on_screen = guide.get_transform().transform(guide.get_xydata())
            colour = guide.get_color()
            markers = [p for p in points if p.get_color() == colour]
            centres = np.vstack([p.get_transform().transform(p.get_xydata()) for p in markers])
            np.testing.assert_allclose(np.sort(on_screen, 0), np.sort(centres, 0), atol=1e-6)
            # Mutation control: the same line drawn at the undodged x misses the markers.
            plain = ax.transData.transform(guide.get_xydata())
            assert np.abs(plain[:, 0] - on_screen[:, 0]).max() > 1.0

    def test_a_working_model_of_other_levels_or_data_is_refused(self) -> None:
        two = (Static(0, name="a0"), Static(1, name="a1"))
        fewer = _quiet(_multi().identify(RegimeMean(two)).estimate, **_ESTIMATE)
        with pytest.raises(ValueError, match="report no mean at"):
            curve(fewer, msm=msm_fit())
        with pytest.raises(ValueError, match="different data"):
            curve(level_means(), msm=msm_binary("identity"))
        # Mutation control: the matching pair draws.
        assert curve(level_means(), msm=msm_fit()).axes[0].get_title(loc="left")

    def test_the_fit_of_the_working_model_can_be_the_first_argument(self) -> None:
        fig = curve(msm_fit(), means=level_means())
        assert fig.axes[0].get_title(loc="left").startswith("Mean outcome by")
        with pytest.raises(ValueError, match="pass means="):
            curve(msm_fit())

    def test_the_means_alone_draw_no_model(self) -> None:
        ax = curve(level_means()).axes[0]
        assert "Working model" not in texts(ax.figure, _core.GID_SERIES_LABEL)
        assert len(lines(ax, _core.GID_POINT)) == 3
        assert "95% pointwise interval" in key_texts(ax)
        # Without a band the bars are each estimate's own confidence interval.
        bare = curve(replace(level_means(), simultaneous=None)).axes[0]
        assert key_texts(bare) == ["95% confidence interval"]

    def test_a_working_model_with_modifiers_is_refused(self) -> None:
        fit = msm_binary("identity", modifiers=("W",))
        with pytest.raises(CapabilityError):
            curve(level_means(), msm=fit)

    def test_a_fit_of_per_arm_means_draws_against_its_working_model(self) -> None:
        """CounterfactualMean keys its means by arm; levels= places means and model alike."""
        means, msm = arm_means(), labelled_msm()
        assert {key.axis for key in means.parameter_keys.values()} == {"arm"}
        fitted = working_model_means(msm)
        for fig in (
            curve(means, msm=msm, levels=CONTACTS),
            curve(msm, means=means, levels=CONTACTS),
        ):
            ax = fig.axes[0]
            xy = sorted(
                (float(ln.get_xdata()[0]), float(ln.get_ydata()[0]))
                for ln in lines(ax, _core.GID_POINT)
            )
            expected = sorted(
                [(CONTACTS[arm], means.estimates[f"ey[{arm}]"].psi) for arm in CONTACTS]
                + [
                    (CONTACTS[str(label)], float(value))
                    for label, value in zip(fitted.levels, fitted.estimate, strict=True)
                ]
            )
            np.testing.assert_allclose(xy, expected)
            assert list(ax.get_xticks()) == [1.0, 2.0, 6.0]
            assert_style(fig)
        # Mutation control: another mapping moves both series, so the model's places are
        # read from levels= and not from its labels' order.
        moved = {"low": 1.0, "medium": 2.0, "high": 3.0}
        ax = curve(means, msm=msm, levels=moved).axes[0]
        assert max(float(ln.get_xdata()[0]) for ln in lines(ax, _core.GID_POINT)) == 3.0

    def test_labelled_arms_without_levels_are_refused_in_words(self) -> None:
        with pytest.raises(CapabilityError, match="is not a number") as refused:
            curve(arm_means(), msm=labelled_msm())
        assert refused.value.refusal.reason is RefusalReason.INVALID_DECLARATION
        assert "levels=" in str(refused.value)

    def test_levels_that_name_other_arms_or_only_some_are_refused(self) -> None:
        """A key that matches nothing, or a partial mapping, is refused, not drawn half-mapped."""
        cases = [
            ({"low": 1, "medium": 2, "top": 6}, r"names \['top'\].*leaves out \['high'\]"),
            ({"low": 1, "medium": 2}, r"leaves out \['high'\]"),
            ({**CONTACTS, "extra": 9}, r"names \['extra'\], which this fit does not report"),
        ]
        for levels, words in cases:
            for call in (
                lambda levels=levels: curve(labelled_msm(), means=arm_means(), levels=levels),
                lambda levels=levels: curve(arm_means(), levels=levels),
            ):
                with pytest.raises(CapabilityError, match=words) as refused:
                    call()
                assert refused.value.refusal.reason is RefusalReason.INVALID_DECLARATION
        # A regime fit: the regimes' names are the keys.
        with pytest.raises(CapabilityError, match=r"names \['a9'\]"):
            curve(level_means(), levels={"a0": 0, "a1": 1, "a2": 2, "a9": 9})
        # Control: the full mapping draws.
        assert curve(arm_means(), levels=CONTACTS).axes[0].get_title(loc="left")

    def test_numeric_arms_need_no_levels_and_an_effect_on_the_arm_axis_is_not_a_mean(
        self,
    ) -> None:
        frame, _ = make_linear_ate(n=200, seed=0)
        study = CausalStudy(
            frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1",))
        )
        means = _quiet(study.identify(CounterfactualMean()).estimate, **_ESTIMATE)
        ax = curve(means).axes[0]
        assert sorted(float(ln.get_xdata()[0]) for ln in lines(ax, _core.GID_POINT)) == [0.0, 1.0]
        # Control: an average effect also sits on the arm axis, and curve() does not take it
        # for per-level means.
        effect = _quiet(study.identify(ATE()).estimate, **_ESTIMATE)
        assert {key.axis for key in effect.parameter_keys.values()} == {"arm"}
        with pytest.raises(TypeError, match="per-level means"):
            curve(effect)

    def test_levels_without_a_number_are_refused(self) -> None:
        with pytest.raises(ValueError, match="levels="):
            curve(level_means(), levels={"a0": "low", "a1": 1, "a2": 2})

    def test_the_band_marks_are_the_fits_bands_at_its_levels(self) -> None:
        means = level_means()
        ax = curve(means).axes[0]
        marks = band_marks(ax)
        assert len(marks) == 3
        for x, lo, hi in marks:
            bound = means.simultaneous.bands[f"ey_regime[a{int(x)}]"]
            assert (lo, hi) == pytest.approx(bound)
        # Mutation control: without the fit's band nothing is marked.
        bare = replace(means, simultaneous=None)
        assert not band_marks(curve(bare).axes[0]).size


# ---------------------------------------------------------------------------- V5


class TestPolicies:
    def test_points_are_the_effects_at_each_shift_with_a_null_line(self) -> None:
        result = shifts()
        fig = curve(result)
        ax = fig.axes[0]
        xy = sorted(
            (float(ln.get_xdata()[0]), float(ln.get_ydata()[0]))
            for ln in lines(ax, _core.GID_POINT)
        )
        expected = sorted(
            (float(key.value.split()[1]), result.estimates[alias].psi)
            for alias, key in result.parameter_keys.items()
        )
        np.testing.assert_allclose(xy, expected)
        (anchor,) = lines(ax, "cleverly-reference-policy")
        assert tuple(anchor.get_xydata()[0]) == (0.0, 0.0)
        (null,) = [ln for ln in ax.lines if ln.get_gid() == "cleverly-reference"]
        assert list(null.get_ydata()) == [0.0, 0.0]
        assert ax.get_title(loc="left") == "Effect by policy"
        assert "compared with shift 0" in _subtitle(fig)
        assert "capped at 3.0" in _subtitle(fig)
        assert ax.get_xlabel().startswith("Shift added to the treatment")
        assert_style(fig)

    def test_the_joint_band_is_marked_at_the_policies_of_that_one_fit(self) -> None:
        """Rule 2: bracket marks at the policies, never a ribbon."""
        result = shifts()
        ax = curve(result).axes[0]
        marks = band_marks(ax)
        assert len(marks) == 3
        assert not any(isinstance(c, matplotlib.collections.PolyCollection) for c in ax.collections)
        for (_x, lo, hi), (alias, _key) in zip(
            marks, sorted(result.parameter_keys.items(), key=lambda kv: kv[1].value), strict=True
        ):
            assert (lo, hi) == pytest.approx(result.simultaneous.bands[alias])
        assert "95% simultaneous band" in key_texts(ax)
        assert "95% pointwise interval" in key_texts(ax)
        assert "experimental" in _core.note_text(ax.figure)
        # Mutation control: with no band on the fit nothing is marked, and the bars are each
        # estimate's own confidence interval.
        bare = curve(replace(result, simultaneous=None)).axes[0]
        assert not band_marks(bare).size
        assert "95% confidence interval" in key_texts(bare)

    def test_incremental_odds_multipliers_use_a_log_axis(self) -> None:
        result = ipsi()
        fig = curve(result)
        ax = fig.axes[0]
        assert ax.get_xscale() == "log"
        assert [t.get_text() for t in ax.get_xticklabels()] == ["0.5", "1", "2", "4"]
        assert ax.get_title(loc="left") == "Mean outcome by policy"
        xs = sorted(float(ln.get_xdata()[0]) for ln in lines(ax, "cleverly-point"))
        assert xs == [0.5, 1.0, 2.0, 4.0]
        assert not [ln for ln in ax.lines if ln.get_gid() == "cleverly-reference"]

    def test_the_odds_axis_names_the_treatment_in_the_callers_words(self) -> None:
        ax = curve(ipsi(), treatment_label="the offer").axes[0]
        assert ax.get_xlabel() == "Multiplier on the odds of the offer (log scale)"
        assert "odds of treatment" not in ax.get_xlabel()
        assert "odds of A" in curve(ipsi()).axes[0].get_xlabel()

    def test_the_pointwise_bar_and_the_band_caps_sit_on_either_side_of_the_estimate(self) -> None:
        ax = curve(shifts()).axes[0]
        (bar_set,) = collections(ax, _core.GID_INTERVAL)
        (cap_line,) = lines(ax, _core.GID_BAND)
        point = lines(ax, _core.GID_POINT)[1]

        def shown(artist: Any, x: float) -> float:
            transform = artist.get_transform()
            return float(transform.transform((x, 0.0))[0])

        x = float(point.get_xdata()[0])
        centre = shown(point, x)
        assert shown(bar_set, x) < centre - 3.0, "the bar is drawn left of the marker"
        assert shown(cap_line, x) > centre + 3.0, "the caps are drawn right of it"

    def test_the_reference_policy_is_drawn_on_the_null_line_and_the_guide_runs_through_it(
        self,
    ) -> None:
        result = middle_reference()
        fig = curve(result)
        ax = fig.axes[0]
        (anchor,) = lines(ax, "cleverly-reference-policy")
        assert tuple(anchor.get_xydata()[0]) == (0.5, 0.0)
        assert anchor.get_markerfacecolor() == _core.FACE
        (guide,) = lines(ax, "cleverly-guide")
        x, y = guide.get_xydata().T
        assert x.tolist() == [0.0, 0.5, 1.0]
        assert y[1] == 0.0
        assert "reference policy s0.5" in _core.note_text(fig)
        assert [t.get_text() for t in ax.get_xticklabels()] == ["0.0", "0.5", "1.0"]
        # Mutation control: a guide joining the reported policies alone crosses zero
        # elsewhere than at the reference.
        points = sorted(tuple(ln.get_xydata()[0]) for ln in lines(ax, _core.GID_POINT))
        (x0, y0), (x1, y1) = points
        crossing = x0 - y0 * (x1 - x0) / (y1 - y0)
        assert abs(crossing - 0.5) > 0.01

    def test_a_mixed_family_falls_back_to_the_forest_and_says_why(self) -> None:
        fig = curve(mixed_family())
        assert fig.axes[0].get_title(loc="left") == forest_module.TITLE
        assert "No single parameter orders these policies" in _subtitle(fig)
        titled = curve(mixed_family(), title="Two policies", subtitle="Shift and scale")
        assert titled.axes[0].get_title(loc="left") == "Two policies"
        # The reader's subtitle wins over the one curve() passes to the forest plot.
        assert _subtitle(titled) == "Shift and scale"

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
        (ribbon,) = ribbons(ax, _core.GID_INTERVAL)
        values = ribbon.get_paths()[0].vertices[:, 1]
        assert values.min() == pytest.approx(table["ci_lower"].min())
        assert values.max() == pytest.approx(table["ci_upper"].max())
        assert ax.get_title(loc="left") == "Conditional effect by W"
        assert ax.get_xlabel() == "W"
        assert "Effect on Y" in ax.get_ylabel()
        assert_style(fig)

    def test_the_average_effect_line_is_the_mean_of_a_curve_with_a_constant(self) -> None:
        result = cate()
        ax = curve(result, modifier="W", at={"Z": 0.0}).axes[0]
        (null,) = lines(ax, _core.GID_REFERENCE)
        assert list(null.get_ydata()) == [0.0, 0.0]
        (average,) = lines(ax, "cleverly-average")
        assert average.get_ydata()[0] == pytest.approx(float(np.mean(result.in_sample)))
        assert "Average effect" in key_texts(ax)
        assert "No effect" in key_texts(ax)
        assert "equals the fit's plug-in average effect" in _core.note_text(ax.figure)

    def test_no_average_line_without_a_constant_term(self) -> None:
        result = no_intercept_curve()
        fig = curve(result)
        assert not lines(fig.axes[0], "cleverly-average")
        assert "the model has no constant term" in _core.note_text(fig)
        # Mutation control: the line the old plot drew, the mean of this curve, is far from
        # the average effect the same data give with a constant in the model.
        naive = float(np.mean(result.in_sample))
        effect = float(np.mean(intercept_curve().in_sample))
        assert abs(naive - effect) > 0.05

    def test_no_average_line_on_a_log_ratio_curve(self) -> None:
        result = log_ratio_curve()
        fig = curve(result, modifier="W")
        assert not lines(fig.axes[0], "cleverly-average")
        assert "is not the overall risk ratio" in _core.note_text(fig)
        assert "Average effect" not in key_texts(fig.axes[0])
        # Mutation control: the difference-scale curve of the same data draws the line.
        assert lines(curve(intercept_curve()).axes[0], "cleverly-average")

    def test_a_weighted_fit_draws_the_weighted_mean_and_says_so(self) -> None:
        result = weighted_curve()
        weights = np.asarray(result.fit.data.weights, dtype=float)
        fig = curve(result)
        (average,) = lines(fig.axes[0], "cleverly-average")
        expected = float(np.average(result.in_sample, weights=weights))
        assert average.get_ydata()[0] == pytest.approx(expected)
        assert "weighted by the data's weights" in _core.note_text(fig)
        # Mutation control: the unweighted mean is another number.
        assert not np.isclose(float(np.mean(result.in_sample)), expected, rtol=1e-4)

    def test_small_multiples_say_the_average_is_over_all_rows(self) -> None:
        fig = curve(cate_by(), modifier="W", by="G")
        heights = {lines(panel, "cleverly-average")[0].get_ydata()[0] for panel in fig.axes}
        assert len(heights) == 1
        assert "Average effect (all rows)" in key_texts(fig.axes[0])
        assert "drawn alike in every panel" in _core.note_text(fig)

    def test_the_interval_level_follows_the_band(self) -> None:
        """The pointwise ribbon is read at the band's level unless alpha= says otherwise."""
        grid = pd.DataFrame({"W": np.linspace(-0.8, 0.8, 5), "Z": 0.0, "G": "f"})
        result = _quiet(_effect(Linear(["W", "Z"])), grid=grid, alpha=0.1, random_state=0)
        ax = curve(result, modifier="W", at={"Z": 0.0}).axes[0]
        assert "90% pointwise interval" in key_texts(ax)
        assert "90% simultaneous band" in key_texts(ax)
        # Mutation control: an explicit level overrides it.
        chosen = curve(result, modifier="W", at={"Z": 0.0}, alpha=0.05).axes[0]
        assert "95% pointwise interval" in key_texts(chosen)

    def test_a_multi_arm_curve_names_its_arms(self) -> None:
        fig = curve(multi_arm_curve())
        assert "of high vs low" in _subtitle(fig)
        # Mutation control: a binary treatment names none.
        assert " vs " not in _subtitle(curve(intercept_curve()))

    def test_a_ratio_of_means_is_not_called_a_risk_ratio(self) -> None:
        from cleverly.plotting._curves import _effect_subtitle

        class Stub:
            fitted_method = "tmle"
            fit = None

        assert _effect_subtitle(Stub(), True, False).startswith("Conditional ratio of means")
        assert _effect_subtitle(Stub(), True, True).startswith("Conditional risk ratio")

    def test_the_band_is_marked_at_its_grid_points_never_as_a_ribbon(self) -> None:
        """Rule 2."""
        result = cate()
        ax = curve(result, modifier="W", at={"Z": 0.0}).axes[0]
        marks = band_marks(ax)
        assert marks[:, 0].tolist() == pytest.approx(result.band.grid["W"].tolist())
        np.testing.assert_allclose(marks[:, 1], result.band.lower)
        np.testing.assert_allclose(marks[:, 2], result.band.upper)
        assert len(ribbons(ax, _core.GID_INTERVAL)) == 1
        assert not ribbons(ax, _core.GID_BAND)
        assert "95% simultaneous band" in key_texts(ax)
        note = _core.note_text(ax.figure)
        assert "No registered study measures its joint coverage" in note
        assert "Brackets mark it at those points only" in note
        # Mutation control: the same curve without a band marks nothing.
        bare = replace(result, band=None)
        assert not band_marks(curve(bare, modifier="W", at={"Z": 0.0}).axes[0]).size

    def test_a_band_at_other_held_values_is_not_drawn_and_the_note_says_so(self) -> None:
        result = cate()
        fig = curve(result, modifier="W", at={"Z": 0.5})
        assert not band_marks(fig.axes[0]).size
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
        assert not ribbons(ax, _core.GID_INTERVAL)
        assert not band_marks(ax).size
        assert not lines(ax, "cleverly-average")
        (line,) = lines(ax, "cleverly-curve")
        x, y = line.get_xydata().T
        np.testing.assert_allclose(y, result.predict({"W": x}))
        assert "carries no interval" in _core.note_text(fig)
        # Mutation control: the projection curve draws its ribbon.
        assert ribbons(curve(cate(), modifier="W", at={"Z": 0.0}).axes[0], _core.GID_INTERVAL)

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


def test_plot_of_a_fit_of_several_repeats_draws_its_reported_curve() -> None:
    """The repeat witness for ``LongitudinalResult.plot``: a fit of two repeats draws the
    curve it reports, the median over its draws, and says the raw values are from one draw."""
    single = isotonic()
    combined = replace(single, repeats=(single.repeats[0], single.repeats[0]))
    assert combined.n_repeats == 2
    fig = combined.plot()
    top = fig.axes[0]
    for regimen in ("treated", "control"):
        expected = frame_rows(combined, "risk", regimen)["psi"].to_numpy()
        assert any(
            np.allclose(line.get_xydata()[:, 1], expected) for line in lines(top, "cleverly-curve")
        )
    assert "one of the 2 repeats" in _core.note_text(fig)


# ---------------------------------------------------------------------------- truth=


def truth_marks(ax: Any) -> list[tuple[float, float]]:
    """Each truth cross at its data position, read back through its own transform."""
    found = []
    for line in lines(ax, _core.GID_TRUTH):
        screen = line.get_transform().transform(line.get_xydata())
        found.append(tuple(ax.transData.inverted().transform(screen)[0]))
    return found


def point_screen(ax: Any) -> list[tuple[float, float]]:
    return [
        tuple(line.get_transform().transform(line.get_xydata())[0])
        for line in lines(ax, _core.GID_POINT)
    ]


class TestTruth:
    def test_a_mapping_by_name_puts_a_cross_beside_each_named_estimate(self) -> None:
        result = full()
        rows = result.curve(scale="risk")
        named = dict(zip(rows["parameter"], rows["psi"], strict=True))
        truth = {name: value + 0.01 for name, value in named.items()}
        fig = curve(result, truth=truth)
        top, diff, _ = fig.axes
        fig.canvas.draw()
        beside = _curves._TRUTH_NUDGE * fig.dpi / 72.0
        for ax in (top, diff):
            # Each cross sits at its estimate's screen x (the same sideways shift), or beside
            # it where the estimate carries a bar, and at the true value; none is left out and
            # none is drawn twice.
            points = point_screen(ax)
            for line in lines(ax, _core.GID_TRUTH):
                screen = line.get_transform().transform(line.get_xydata())[0]
                assert any(
                    np.isclose(screen[0], x) or np.isclose(screen[0], x + beside) for x, _ in points
                )
        drawn = sorted(v for ax in (top, diff) for _, v in truth_marks(ax))
        assert drawn == pytest.approx(sorted(truth.values()))
        assert _core.TRUTH_LABEL in key_texts(top)
        assert_style(fig)
        assert_text_inside(fig)

    def test_the_key_entry_and_the_crosses_appear_only_with_truth(self) -> None:
        """Control: a figure without truth= has neither."""
        for fig in (curve(full()), curve(shifts()), curve(level_means())):
            assert _core.TRUTH_LABEL not in _core.key_labels(fig)
            assert not any(lines(ax, _core.GID_TRUTH) for ax in fig.axes)

    def test_the_cross_check_rejects_a_mark_off_its_value(self, monkeypatch: Any) -> None:
        """Mutation control: a cross drawn away from the given value fails the same read."""
        result = shifts()
        values = [0.1, 0.2, 0.3]
        ax = curve(result, truth=values).axes[0]
        assert sorted(v for _, v in truth_marks(ax)) == pytest.approx(values)
        original = _core.truth_mark

        def shifted(ax: Any, x: float, y: float, **kwargs: Any) -> Any:
            return original(ax, x, y + 0.05, **kwargs)

        monkeypatch.setattr(_core, "truth_mark", shifted)
        ax = curve(result, truth=values).axes[0]
        assert sorted(v for _, v in truth_marks(ax)) != pytest.approx(values)

    def test_one_series_takes_values_over_its_x_levels(self) -> None:
        # The reference shift 0 is not an estimate, so the levels are the other three.
        ax = curve(shifts(), truth=[0.1, 0.2, 0.3]).axes[0]
        np.testing.assert_allclose(
            sorted(truth_marks(ax)), [(0.25, 0.1), (0.5, 0.2), (0.75, 0.3)], atol=1e-9
        )
        ax = curve(level_means(), truth={0: 1.0, 2: 3.0}).axes[0]
        np.testing.assert_allclose(sorted(truth_marks(ax)), [(0.0, 1.0), (2.0, 3.0)], atol=1e-9)
        # The working model is not a series of true values: the crosses sit with the means.
        ax = curve(level_means(), msm=msm_fit(), truth=[1.0, 2.0, 3.0]).axes[0]
        assert len(truth_marks(ax)) == 3

    def test_a_mismatched_truth_is_refused_with_its_reason(self) -> None:
        cases = [
            (shifts(), [0.1, 0.2], "has 2 values"),
            (level_means(), {5: 1.0}, "not an x level"),
            (full(), [0.1, 0.2, 0.3, 0.4], "one series"),
            (full(), {"risk_regimen[nope]": 0.1}, "does not report"),
            (shifts(), [0.1, math_nan(), 0.3], "not a finite number"),
            (shifts(), "0.1", "mapping or a sequence"),
            (cate(), [0.1, 0.2], "function of the modifier"),
            (full(), {}, "empty mapping"),
            (shifts(), {}, "empty mapping"),
            (shifts(), [0.1, True, 0.3], "boolean"),
        ]
        for result, truth, words in cases:
            extra = {"modifier": "W", "at": {"Z": 0.0}} if result is cate() else {}
            with pytest.raises(CapabilityError, match=words) as refused:
                curve(result, truth=truth, **extra)
            assert refused.value.refusal.reason is RefusalReason.INVALID_DECLARATION

    def test_a_name_the_fit_reports_but_the_plot_leaves_out_is_passed_over(self) -> None:
        result = full()
        contrast = next(n for n, e in result.estimates.items() if e.scale != "level")
        level = next(n for n, e in result.estimates.items() if e.scale == "level")
        regimen = result.parameter_index[level][0]
        fig = curve(result, regimens=[regimen], truth={contrast: 0.0, level: 0.2})
        assert [v for _, v in truth_marks(fig.axes[0])] == pytest.approx([0.2])

    def test_a_conditional_effect_curve_draws_the_true_function_as_a_line(self) -> None:
        fig = curve(cate(), modifier="W", at={"Z": 0.0}, truth=lambda w: 0.5 * w)
        ax = fig.axes[0]
        (line,) = lines(ax, _core.GID_TRUTH)
        x, y = line.get_xydata().T
        np.testing.assert_allclose(y, 0.5 * x)
        assert line.get_marker() in ("None", None, "")
        assert _core.TRUTH_LABEL in key_texts(ax)
        assert_style(fig)
        with pytest.raises(CapabilityError, match="one panel per level"):
            curve(cate_by(), modifier="W", by="G", truth=lambda w: w)

    def test_restricted_means_take_values_for_one_regimen(self) -> None:
        result = full()
        regimen = _regimen_order_of(result)[0]
        fig = curve(result, measure="rmst", regimens=[regimen], truth=[1.5, 2.0, 2.5])
        assert len(truth_marks(fig.axes[0])) == 3


def math_nan() -> float:
    return float("nan")


def _regimen_order_of(result: Any) -> list[str]:
    from cleverly.plotting._curves import _regimen_order

    return _regimen_order(result)


class TestTheKeyGlyphs:
    """A band's caps and an interval's bar read apart in the key, not by a dash's width."""

    @staticmethod
    def _glyphs(ax: Any, gid: str) -> list[Any]:
        from matplotlib.lines import Line2D

        return [line for line in ax.get_legend().findobj(Line2D) if line.get_gid() == gid]

    def test_brackets_on_a_finite_grid_draw_caps_and_a_bar_in_the_key(self) -> None:
        # the discharge-to-readmission risk curve: horizons only, a multiplier band and
        # pointwise intervals, both as marks at the horizons in the first series' colour
        top = curve(subset()).axes[0]
        assert band_marks(top).size and bars(top).size
        caps = self._glyphs(top, _core.GID_KEY_BAND)
        (bar,) = self._glyphs(top, _core.GID_KEY_INTERVAL)
        # the band's glyph is two horizontal caps, one above the other, and no stem
        assert len(caps) == 2
        assert all(np.ptp(cap.get_ydata()) == 0.0 and np.ptp(cap.get_xdata()) > 0 for cap in caps)
        assert caps[0].get_ydata()[0] != caps[1].get_ydata()[0]
        # the interval's glyph is one vertical bar spanning the caps
        assert np.ptp(bar.get_xdata()) == 0.0
        assert min(bar.get_ydata()) == min(cap.get_ydata()[0] for cap in caps)
        assert max(bar.get_ydata()) == max(cap.get_ydata()[0] for cap in caps)
        # each glyph keeps its mark's colour
        band_color = lines(top, _core.GID_BAND)[0].get_color()
        assert matplotlib.colors.to_hex(caps[0].get_color()) == matplotlib.colors.to_hex(band_color)

    def test_ribbons_keep_their_swatches(self) -> None:
        # the control: a curve that reports every node draws ribbons, whose key entries are
        # filled swatches, lighter for the band, and no bracket glyph
        top = curve(full()).axes[0]
        assert ribbons(top, _core.GID_BAND)
        assert not self._glyphs(top, _core.GID_KEY_BAND)
        assert not self._glyphs(top, _core.GID_KEY_INTERVAL)

    def test_a_figure_with_bracket_glyphs_pickles(self) -> None:
        import pickle

        fig = curve(subset())
        again = pickle.loads(pickle.dumps(fig))
        assert self._glyphs(again.axes[0], _core.GID_KEY_BAND)
        plt.close(again)

    def test_a_tuple_colour_is_read_for_the_glyph(self) -> None:
        fig, ax = plt.subplots()
        _core.brackets(ax, [1.0], [0.1], [0.9], color=(0.1, 0.2, 0.3), label="band", band=True)
        _core.brackets(ax, [1.0], [0.3], [0.7], color=(0.1, 0.2, 0.3), label="bar")
        _core.legend(ax)
        fig.canvas.draw()
        (cap, _) = self._glyphs(ax, _core.GID_KEY_BAND)
        (bar,) = self._glyphs(ax, _core.GID_KEY_INTERVAL)
        for glyph in (cap, bar):
            assert matplotlib.colors.to_hex(glyph.get_color()) == "#1a334c"

    def test_the_figure_level_key_draws_the_glyphs_too(self) -> None:
        # figure_header puts the key on the figure, through fig.legend
        fig, (left, right) = plt.subplots(1, 2)
        _core.brackets(left, [1.0], [0.1], [0.9], color=_core.PALETTE[0], label="band", band=True)
        _core.brackets(right, [1.0], [0.3], [0.7], color=_core.PALETTE[0], label="bar")
        _core.figure_header(fig, "Title", key_sources=[left, right])
        (key,) = fig.legends
        assert key.get_gid() == _core.GID_KEY
        from matplotlib.lines import Line2D

        gids = [line.get_gid() for line in key.findobj(Line2D)]
        assert gids.count(_core.GID_KEY_BAND) == 2 and gids.count(_core.GID_KEY_INTERVAL) == 1


# ---------------------------------------------------------------------------- X92 batch 3


def _mark_screen_x(ax: Any) -> list[float]:
    """The screen x of every interval bar and band cap an axes draws."""
    from matplotlib.collections import LineCollection

    found: list[float] = []
    for artist in ax.collections:
        if isinstance(artist, LineCollection) and artist.get_gid() == _core.GID_INTERVAL:
            for segment in artist.get_segments():
                found.append(float(artist.get_transform().transform(segment)[0, 0]))
    for line in lines(ax, _core.GID_BAND):
        found.extend(float(x) for x in line.get_transform().transform(line.get_xydata())[:, 0])
    return found


def _truth_clearance(fig: Any) -> float:
    """The least screen distance, in points, from a cross to any bar or bracket."""
    fig.canvas.draw()
    gaps = []
    for ax in fig.axes:
        marks = _mark_screen_x(ax)
        for line in lines(ax, _core.GID_TRUTH):
            x = float(line.get_transform().transform(line.get_xydata())[0, 0])
            if marks:
                gaps.append(min(abs(x - mark) for mark in marks) * 72.0 / fig.dpi)
    return min(gaps)


class TestTruthBesideTheBars:
    """A cross sits beside a bar and its brackets, not on them, at the estimate's own x."""

    #: A bracket cap's half width (12-point marker) and a cross's (8-point marker).
    CLEAR = 6.0 + 4.0

    @staticmethod
    def _truth(result: Any) -> dict[str, float]:
        rows = result.curve(scale="risk")
        return {name: psi + 0.01 for name, psi in zip(rows["parameter"], rows["psi"], strict=True)}

    def test_a_cross_clears_the_bars_and_brackets_of_every_series(self) -> None:
        for result in (full(), subset()):
            fig = curve(result, truth=self._truth(result))
            assert _truth_clearance(fig) >= self.CLEAR
            assert_text_inside(fig)

    def test_the_cross_keeps_the_estimates_x_in_its_data(self) -> None:
        result = subset()
        fig = curve(result, truth=self._truth(result))
        estimates = {float(x) for x in result.curve(scale="risk")["time"]}
        drawn = {
            float(line.get_xdata()[0]) for ax in fig.axes for line in lines(ax, _core.GID_TRUTH)
        }
        assert drawn <= estimates and drawn

    def test_mutation_controls_put_the_cross_back_on_a_bar(self, monkeypatch: Any) -> None:
        result = subset()
        # No nudge: the cross sits on its own bar.
        monkeypatch.setattr(_curves, "_TRUTH_NUDGE", 0.0)
        assert _truth_clearance(curve(result, truth=self._truth(result))) < 1.0
        # The nudge without the wider dodge: the cross lands on the next series' bar.
        monkeypatch.setattr(_curves, "_TRUTH_NUDGE", 11.0)
        monkeypatch.setattr(_curves, "_TRUTH_DODGE", _curves._DODGE)
        assert _truth_clearance(curve(result, truth=self._truth(result))) < self.CLEAR


class TestContrastsAloneTakeTheirPlansColours:
    def test_each_difference_has_a_palette_colour_and_the_null_stays_grey(self) -> None:
        fig = curve(regime_contrast())
        diff = fig.axes[0]
        colours = [
            matplotlib.colors.to_hex(line.get_color()) for line in lines(diff, "cleverly-curve")
        ]
        assert len(colours) == 2 and len(set(colours)) == 2
        assert all(colour in _core.PALETTE for colour in colours)
        (null,) = lines(diff, _core.GID_REFERENCE)
        assert matplotlib.colors.to_hex(null.get_color()) == _core.REFERENCE
        assert _core.REFERENCE not in colours
        # Each plan keeps its slot in the fit's regimen order (never, always, first).
        labels = _label_colours(diff)
        assert labels == {"always vs never": _core.PALETTE[1], "first vs never": _core.PALETTE[2]}
        assert_style(fig)

    def test_beside_the_regimens_curves_a_difference_takes_its_plans_colour(self) -> None:
        fig = curve(full())
        top, diff, _ = fig.axes
        curves, differences = _label_colours(top), _label_colours(diff)
        assert set(differences.values()) <= set(_core.PALETTE)
        for contrast, colour in differences.items():
            # The difference wears the colour of the plan it compares with the reference.
            assert colour == curves[contrast.split(" vs ")[0]]
        (null,) = lines(diff, _core.GID_REFERENCE)
        assert matplotlib.colors.to_hex(null.get_color()) not in differences.values()
        assert_style(fig)

    def test_differences_that_share_a_drawn_plan_stay_grey(self) -> None:
        assert _curves._drawn_plan_slots(["a vs b", "a vs c"], {"a": 0, "b": 1, "c": 2}) is None
        # A plan not drawn as a curve has no colour to lend either.
        assert _curves._drawn_plan_slots(["d vs b"], {"a": 0, "b": 1}) is None
        # Control: distinct drawn plans lend their curves' slots.
        assert _curves._drawn_plan_slots(["c vs a", "b vs a"], {"a": 0, "b": 1, "c": 2}) == {
            "c vs a": 2,
            "b vs a": 1,
        }

    def test_differences_that_share_a_plan_are_numbered_in_turn(self) -> None:
        slots = _curves._plan_slots(["a vs b", "a vs c"], ["a", "b", "c"])
        assert slots == {"a vs b": 0, "a vs c": 1}
        # Control: distinct plans keep their places in the fit's order.
        assert _curves._plan_slots(["b vs a", "c vs a"], ["a", "b", "c"]) == {
            "b vs a": 1,
            "c vs a": 2,
        }


class TestOutcomeWords:
    def test_outcome_label_names_the_quantity_without_mean(self) -> None:
        diff = curve(regime_contrast(), outcome_label="order share").axes[0]
        assert diff.get_ylabel() == "Difference in order share"
        diff = curve(regime_contrast(), outcome_label="Customers who order").axes[0]
        assert diff.get_ylabel() == "Difference in customers who order"
        # Control: without it the label is the mean of the outcome.
        assert curve(regime_contrast()).axes[0].get_ylabel() == "Difference in mean outcome"

    def test_a_short_difference_panel_breaks_before_the_phrase_only(self) -> None:
        fig = curve(repeated(), outcome_label="customers who order")
        top, diff = fig.axes[0], fig.axes[1]
        assert top.get_ylabel() == "customers who order"
        assert diff.get_ylabel() == "Difference in\ncustomers who order"

    def test_the_wrap_splits_a_phrase_only_when_it_cannot_fit(self) -> None:
        assert _curves._wrap_axis_label("Difference in", "order share", 20) == (
            "Difference in\norder share"
        )
        assert _curves._wrap_axis_label("Difference in", "order share", 26) == (
            "Difference in order share"
        )
        # Control: a phrase longer than a line is wrapped like any text.
        long = _curves._wrap_axis_label("Difference in", "share of customers who order", 20)
        assert "order" in long.split("\n")[-1] and long.count("\n") >= 2

    def test_only_a_capitalised_english_word_is_lowered(self) -> None:
        assert _core.mid_phrase("Customers who order") == "customers who order"
        for kept in ("HbA1c level", "Y", "BMI", "order share"):
            assert _core.mid_phrase(kept) == kept


class TestLevelWords:
    def test_per_level_means_are_asked_about_the_treatment_level(self) -> None:
        for fig in (curve(level_means()), curve(arm_means(), levels=CONTACTS)):
            title = fig.axes[0].get_title(loc="left")
            assert title == "Mean outcome by treatment level"
            assert "dose" not in title

    def test_a_declared_continuous_treatment_is_a_dose(self) -> None:
        from types import SimpleNamespace

        dose = SimpleNamespace(data=SimpleNamespace(is_continuous_treatment=True))
        arms = SimpleNamespace(data=SimpleNamespace(is_continuous_treatment=False))
        assert _curves._level_title(arms, dose) == "Mean outcome by dose"
        # Control: neither fit declares a dose.
        assert _curves._level_title(arms, None) == "Mean outcome by treatment level"
        assert not level_means().data.is_continuous_treatment


class TestBandOnLevelMeans:
    def test_band_none_or_false_draws_no_band(self) -> None:
        drawn = curve(msm_fit(), means=level_means())
        assert band_marks(drawn.axes[0]).size
        assert any("simultaneous band" in text for text in key_texts(drawn.axes[0]))
        for off in ("none", False):
            ax = curve(msm_fit(), means=level_means(), band=off).axes[0]
            assert not band_marks(ax).size
            assert not any("simultaneous band" in text for text in key_texts(ax))
            # The pointwise intervals stay.
            assert bars(ax).size

    def test_a_band_the_fit_does_not_hold_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no bootstrap band"):
            curve(level_means(), band="bootstrap")
        with pytest.raises(ValueError, match="band must be"):
            curve(level_means(), band="sup")
        # Control: the band it holds is drawn when named.
        assert band_marks(curve(level_means(), band="multiplier").axes[0]).size


class TestEffectCurveBand:
    def test_band_none_or_false_turns_the_curves_band_off(self) -> None:
        settings = {"modifier": "W", "at": {"Z": 0.0}}
        drawn = curve(cate(), **settings).axes[0]
        assert band_marks(drawn).size
        for off in ("none", False):
            ax = curve(cate(), band=off, **settings).axes[0]
            assert not band_marks(ax).size
            assert not any("simultaneous band" in text for text in key_texts(ax))
        # Control: naming the band it holds draws it.
        assert band_marks(curve(cate(), band="multiplier", **settings).axes[0]).size

    def test_a_band_the_curve_cannot_draw_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no bootstrap band"):
            curve(cate(), band="bootstrap", modifier="W", at={"Z": 0.0})
        assert cate_by().band is None
        with pytest.raises(ValueError, match="no multiplier band"):
            curve(cate_by(), band="multiplier", modifier="W", by="G")
        with pytest.raises(ValueError, match="band must be"):
            curve(cate(), band="sup", modifier="W", at={"Z": 0.0})
        # Control: "none" on a curve without a band has nothing to turn off, and draws.
        assert curve(cate_by(), band="none", modifier="W", by="G")


def test_an_effect_curve_puts_the_callers_words_mid_phrase() -> None:
    settings = {"modifier": "W", "at": {"Z": 0.0}}
    worded = curve(cate(), outcome_label="Customers who order", **settings).axes[0]
    assert worded.get_ylabel() == "Effect on customers who order"
    # Control: a name or symbol stays as written, and the column name is untouched.
    assert curve(cate(), outcome_label="HbA1c", **settings).axes[0].get_ylabel() == (
        "Effect on HbA1c"
    )
    assert curve(cate(), **settings).axes[0].get_ylabel() == "Effect on Y"
