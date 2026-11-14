"""Every plot on one small fit of every result family: it draws, or it refuses with a reason.

A plot that meets a result it does not draw must say so: a ``CapabilityError`` (with the
roadmap item that tracks a missing view), or a ``ValueError`` or ``TypeError`` naming what it
takes. An ``AttributeError``, ``KeyError`` or ``IndexError`` from inside a plot is a crash, and
fails here with the cell that raised it.

``EXPECTED`` pins each cell: ``"draws"``, ``"refuses"`` (a refusal with no roadmap item, such
as a fit with no bootstrap replicates) or the roadmap id the refusal names. A cell that moves
from drawing to refusing, or to another item, fails.

To add a result family, add a fitting function to ``FAMILIES`` and one entry per plot to
``EXPECTED``. To add a plot, add it to ``PLOTS`` (or to ``OTHER_INPUTS`` when it takes an object
that is not a fitted result) and pin it for every family. The meta tests fail when
``cleverly.plotting.__all__`` or ``FAMILIES`` outgrows the matrix.
"""

from __future__ import annotations

import importlib
import inspect
import linecache
import pkgutil
import re
import traceback
import warnings
from collections.abc import Callable
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.special import expit
from sklearn.linear_model import LinearRegression, LogisticRegression

import cleverly
import cleverly.plotting as plotting
from cleverly import (
    ATE,
    AdaptiveTMLEMethod,
    CausalStudy,
    CollaborativeTMLEMethod,
    ConditionalContrast,
    DRTMLEMethod,
    ModelSpec,
    PointTreatment,
    RegimeContrast,
    RegimeMean,
    Runtime,
    TimeToEvent,
    TrialATE,
    TrialWithExternalData,
)
from cleverly.bases import Linear, Modifiers
from cleverly.datasets import (
    make_linear_ate,
    make_longitudinal,
    make_longitudinal_survival,
    make_point_survival,
)
from cleverly.estimators.dose_response import DoseResponse, DoseResponseTMLE
from cleverly.exceptions import CapabilityError
from cleverly.learners import GLM
from cleverly.longitudinal import LTMLE, StructuralNestedMean
from tests.unit import _capability_sweep_support as sweep
from tests.unit.test_plotting_core import assert_short_title

#: The tests that share the module's fits run on one worker, which fits each family once.
_SHARED = pytest.mark.xdist_group("plotting_result_matrix")

#: Plots that take a fitted result.
PLOTS = (
    "forest",
    "curve",
    "influence",
    "overlap",
    "weights",
    "balance",
    "learners",
    "calibration",
    "repeats",
    "bootstrap",
)
#: Public names of ``cleverly.plotting`` that take another object, or are not plots.
OTHER_INPUTS = {
    "PALETTE": "the palette, not a plot",
    "notes": "reads a figure",
    "style": "a style context, not a plot",
    "sensitivity": "takes a sensitivity analysis or a truncation curve",
    "confounding": "takes an omitted-variable bound or a benchmark",
    "targeting": "takes a targeting curve",
    "zipper": "takes a simulation study",
}


def _logistic() -> LogisticRegression:
    return LogisticRegression(max_iter=1000)


def _point_tmle() -> Any:
    frame, _ = make_linear_ate(n=300, seed=0)
    design = PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4"))
    return (
        CausalStudy(frame, design=design)
        .identify(ATE())
        .estimate(
            outcome_learner=LinearRegression(),
            treatment_learner=_logistic(),
            n_folds=3,
            repeats=2,
            q_bounds=(-10.0, 10.0),
            random_state=0,
        )
    )


def _ltmle_repeats() -> Any:
    frame, _ = make_longitudinal_survival(n=400, seed=0)
    return LTMLE(
        {"always": 1, "never": 0},
        n_folds=2,
        repeats=3,
        outcome_learner=LinearRegression(),
        treatment_learner=_logistic(),
        censoring_learner=_logistic(),
    ).fit(
        frame,
        outcome=["Y1", "Y2"],
        treatment=["A1", "A2"],
        baseline=["W1", "W2"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
    )


def _point_bootstrap() -> Any:
    frame, _ = make_linear_ate(n=200, seed=0)
    design = PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4"))
    return (
        CausalStudy(frame, design=design)
        .identify(ATE())
        .estimate(
            outcome_learner=LinearRegression(),
            treatment_learner=_logistic(),
            cross_fit=False,
            n_bootstrap=20,
            q_bounds=(-10.0, 10.0),
            random_state=0,
        )
    )


def _point_survival() -> tuple[pd.DataFrame, TimeToEvent]:
    frame, _ = make_point_survival(n=600, seed=4, n_times=4)
    design = TimeToEvent(
        time="time", event="event", treatment="A", baseline=("W1", "W2"), grid=(1, 2, 3, 4)
    )
    return frame, design


def _time_to_event() -> Any:
    frame, design = _point_survival()
    return CausalStudy(frame, design=design).estimate(
        RegimeMean({"arm1": 1, "arm0": 0}, horizons=(1, 2, 3, 4)),
        outcome_learner=LinearRegression(),
        pseudo_learner=LinearRegression(),
        treatment_learner=_logistic(),
        cross_fit=False,
    )


def _survival_method(method: Any) -> Callable[[], Any]:
    def fit() -> Any:
        frame, design = _point_survival()
        return CausalStudy(frame, design=design).estimate(
            RegimeMean({"arm1": 1, "arm0": 0}, horizons=(2, 4)),
            method=method,
            outcome_learner=_logistic(),
            cross_fit=False,
        )

    return fit


def _continuous_survival() -> Any:
    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(size=n)
    treated = rng.binomial(1, 1 / (1 + np.exp(-x)))
    relapse = np.exp(-1.5 + 0.5 * x - treated)
    death = np.exp(-2.0 - 0.5 * x - 0.5 * treated)
    event_time = rng.exponential(1 / (relapse + death))
    cause = np.where(rng.uniform(size=n) < relapse / (relapse + death), 1, 2)
    dropout = rng.exponential(1 / 0.15, size=n)
    frame = pd.DataFrame(
        {
            "time": np.minimum(event_time, dropout),
            "event": np.where(event_time <= dropout, cause, 0),
            "treated": treated,
            "x": x,
        }
    )
    design = TimeToEvent(
        time="time",
        event="event",
        treatment="treated",
        baseline=("x",),
        grid="continuous",
        causes={1: "relapse", 2: "death"},
    )
    return CausalStudy(frame, design=design).estimate(
        RegimeContrast({"treated": 1, "control": 0}, reference="control", horizons=(1, 2)),
        random_state=0,
    )


def _adaptive() -> Any:
    rng = np.random.default_rng(1)
    n = 500
    w = rng.normal(size=n)
    trial = rng.binomial(1, 0.4, size=n)
    treated = np.where(
        trial == 1, rng.binomial(1, 0.5, size=n), rng.binomial(1, 1 / (1 + np.exp(-w)), size=n)
    )
    y = w + 0.5 * treated + rng.normal(size=n)
    frame = pd.DataFrame({"score": y, "program": treated, "risk": w, "in_trial": trial})
    design = TrialWithExternalData(
        outcome="score", treatment="program", adjustment=("risk",), source="in_trial"
    )
    models = ModelSpec(outcome_learner=LinearRegression(), treatment_learner=_logistic())
    return (
        CausalStudy(frame, design=design)
        .identify(TrialATE(population="trial"))
        .estimate(method=AdaptiveTMLEMethod(models=models, runtime=Runtime(random_state=1)))
    )


def _effect_curve() -> Any:
    rng = np.random.default_rng(0)
    n = 500
    w, z = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
    treated = rng.binomial(1, 0.5, n)
    y = rng.binomial(1, 0.3 + 0.2 * treated * (w > 0) + 0.05 * z)
    frame = pd.DataFrame({"Y": y, "A": treated, "W": w, "Z": z})
    design = PointTreatment(outcome="Y", treatment="A", adjustment=("W", "Z"))
    question = ConditionalContrast(Modifiers(["W"], domains={"W": (-1, 1)}), model=Linear(["W"]))
    fit = (
        CausalStudy(frame, design=design)
        .identify(question)
        .estimate(
            outcome_learner=_logistic(),
            treatment_learner=_logistic(),
            cross_fit=False,
            random_state=0,
        )
    )
    return fit.effect_curve()


def _dose_response() -> Any:
    rng = np.random.default_rng(0)
    n = 500
    w = rng.normal(size=n)
    dose = rng.normal(2.5 + 0.6 * w, 1.2)
    y = rng.binomial(1, expit(-1 + 0.5 * dose - 0.08 * dose * dose + 0.4 * w))
    return DoseResponseTMLE(
        DoseResponse.spanning((1.0, 4.0), 3, fractions=(0.25,)),
        outcome_learner=_logistic(),
        treatment_learner=_logistic(),
        random_state=0,
        n_folds=2,
    ).fit(pd.DataFrame({"Y": y, "A": dose, "W": w}), outcome="Y", treatment="A", covariates=["W"])


def _structural_nested_mean() -> Any:
    frame, _ = make_longitudinal(n=300, seed=0, censoring=False)
    return StructuralNestedMean(
        {"always": 1, "never": 0},
        outcome_learner=GLM(family="binomial"),
        treatment_learner=GLM(family="binomial"),
        n_folds=2,
        random_state=0,
    ).fit(
        frame,
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["W1", "W2"],
        time_varying=[[], ["L2"]],
    )


#: One small fit per result family.
FAMILIES: dict[str, Callable[[], Any]] = {
    "point_tmle_repeats": _point_tmle,
    "point_tmle_bootstrap": _point_bootstrap,
    "ltmle_repeats": _ltmle_repeats,
    "time_to_event_ltmle": _time_to_event,
    "time_to_event_ctmle": _survival_method(CollaborativeTMLEMethod()),
    "time_to_event_drtmle": _survival_method(DRTMLEMethod()),
    "continuous_survival": _continuous_survival,
    "adaptive_tmle": _adaptive,
    "effect_curve": _effect_curve,
    "dose_response": _dose_response,
    "structural_nested_mean": _structural_nested_mean,
    # A declared mechanism read in the weighted sample (Fix-16): an estimated tilt, so each
    # plot answers as it does on an estimated-mechanism fit of the same kind.
    "point_tmle_known_sample": sweep.fit_tmle_known_sample_weights,
    "ltmle_known_sample": sweep.fit_ltmle_known_sample_weights,
}

#: Public result classes that are not fits of one estimand, with the plot or fit that covers
#: them. Every other public ``*Result`` class must be the type of a ``FAMILIES`` fit.
NOT_FITS = {
    "BenchmarkResult": "an omitted-variable benchmark; confounding() draws it",
    "BootstrapResult": "a fit's replicates; bootstrap() reads it through its fit",
    "CausalResult": "a protocol, not a class with instances",
    "RefutationResult": "a refutation report of a fit",
    "SimulatedConfoundingResult": "a simulated-confounding report; sensitivity() draws it",
    "StudyResult": "a simulation study; zipper() draws it",
    "VariableImportanceResult": "one TMLEResult per candidate, each covered as a point fit",
}

#: The pinned outcome of each plot on each family: "draws", the roadmap id a refusal
#: names, or the refusal's class and first five words.
EXPECTED: dict[str, dict[str, str]] = {
    "point_tmle_repeats": {
        "forest": "draws",
        "curve": "TypeError: curve() draws a longitudinal fit,",
        "influence": "draws",
        "overlap": "draws",
        "weights": "CapabilityError: weights() draws the cumulative inverse-probability",
        "balance": "draws",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "draws",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "point_tmle_bootstrap": {
        "forest": "draws",
        "curve": "TypeError: curve() draws a longitudinal fit,",
        "influence": "draws",
        "overlap": "draws",
        "weights": "CapabilityError: weights() draws the cumulative inverse-probability",
        "balance": "draws",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "draws",
    },
    "ltmle_repeats": {
        "forest": "draws",
        "curve": "draws",
        "influence": "draws",
        "overlap": "X100",
        "weights": "draws",
        "balance": "X100",
        "learners": "X100",
        "calibration": "X100",
        "repeats": "draws",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "time_to_event_ltmle": {
        "forest": "draws",
        "curve": "draws",
        "influence": "draws",
        "overlap": "X100",
        "weights": "draws",
        "balance": "X100",
        "learners": "X100",
        "calibration": "X100",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "time_to_event_ctmle": {
        "forest": "draws",
        "curve": "draws",
        "influence": "ValueError: risk_regimen[arm1 @ t=2]: this estimate's",
        "overlap": "X100",
        "weights": "draws",
        "balance": "X100",
        "learners": "X100",
        "calibration": "X100",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "time_to_event_drtmle": {
        "forest": "draws",
        "curve": "draws",
        "influence": "draws",
        "overlap": "X100",
        "weights": "draws",
        "balance": "X100",
        "learners": "X100",
        "calibration": "X100",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "continuous_survival": {
        "forest": "draws",
        "curve": "X100",
        "influence": "draws",
        "overlap": "X100",
        "weights": "X100",
        "balance": "X100",
        "learners": "X100",
        "calibration": "CapabilityError: calibration() draws a nuisance report's",
        "repeats": "CapabilityError: repeats() draws the estimate of",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "adaptive_tmle": {
        "forest": "draws",
        "curve": "TypeError: curve() draws a longitudinal fit,",
        "influence": "draws",
        "overlap": "X100",
        "weights": "CapabilityError: weights() draws the cumulative inverse-probability",
        "balance": "CapabilityError: covariate_balance: this report reads a",
        "learners": "X100",
        "calibration": "X100",
        "repeats": "CapabilityError: repeats() draws the estimate of",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "effect_curve": {
        "forest": "TypeError: forest() takes fitted results, result",
        "curve": "draws",
        "influence": "TypeError: influence() takes a ParameterEstimate or",
        "overlap": "CapabilityError: overlap() draws a point-treatment fit's",
        "weights": "CapabilityError: weights() draws the cumulative inverse-probability",
        "balance": "CapabilityError: covariate_balance weighs the covariates by",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "CapabilityError: repeats() draws the estimate of",
        "bootstrap": "TypeError: bootstrap() takes a fitted result",
    },
    "dose_response": {
        "forest": "draws",
        "curve": "draws",
        "influence": "draws",
        "overlap": "CapabilityError: overlap() draws a point-treatment fit's",
        "weights": "CapabilityError: weights() draws the cumulative inverse-probability",
        "balance": "CapabilityError: covariate_balance weighs the covariates by",
        "learners": "TypeError: learners() takes SuperLearnerDiagnostics, a CTMLESelection,",
        "calibration": "CapabilityError: no nuisance model here reports",
        "repeats": "CapabilityError: repeats() draws the estimate of",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "structural_nested_mean": {
        "forest": "draws",
        "curve": "TypeError: curve() draws a longitudinal fit,",
        "influence": "draws",
        "overlap": "X100",
        "weights": "CapabilityError: weights() draws the cumulative inverse-probability",
        "balance": "CapabilityError: covariate_balance weighs the covariates by",
        "learners": "TypeError: learners() cannot read the nuisance",
        "calibration": "CapabilityError: no nuisance model here reports",
        "repeats": "CapabilityError: repeats() draws the estimate of",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "point_tmle_known_sample": {
        "forest": "draws",
        "curve": "draws",
        "influence": "draws",
        "overlap": "draws",
        "weights": "CapabilityError: weights() draws the cumulative inverse-probability",
        "balance": "draws",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "ltmle_known_sample": {
        "forest": "draws",
        "curve": "CapabilityError: this fit has one end-of-study",
        "influence": "draws",
        "overlap": "X100",
        "weights": "draws",
        "balance": "X100",
        "learners": "X100",
        "calibration": "X100",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
}


@pytest.fixture(scope="module")
def fits() -> dict[str, Any]:
    cache: dict[str, Any] = {}

    def get(family: str) -> Any:
        if family not in cache:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cache[family] = FAMILIES[family]()
        return cache[family]

    return get  # type: ignore[return-value]


def _raised_by_cleverly(error: BaseException) -> bool:
    """Whether a ``raise`` statement in ``cleverly``'s own source raised the error."""
    frame = traceback.extract_tb(error.__traceback__)[-1]
    in_package = "/src/cleverly/" in frame.filename.replace("\\", "/")
    line = linecache.getline(frame.filename, frame.lineno or 0).strip()
    return in_package and line.startswith("raise")


def outcome(plot: str, result: Any) -> str:
    """``"draws"``, the roadmap id a refusal names, or the refusal's class and first words.

    An exception that a ``raise`` statement of ``cleverly`` did not raise, or that is not a
    ``ValueError`` or ``TypeError`` (``CapabilityError`` is one), is a crash and propagates.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = getattr(plotting, plot)(result)
    except (ValueError, TypeError) as error:
        if not _raised_by_cleverly(error):
            raise
        if isinstance(error, CapabilityError) and error.refusal.roadmap:
            return str(error.refusal.roadmap)
        return f"{type(error).__name__}: " + " ".join(str(error).split()[:5])
    assert isinstance(fig, Figure)
    assert_short_title(fig)
    plt.close(fig)
    return "draws"


@_SHARED
@pytest.mark.parametrize("family", list(FAMILIES))
@pytest.mark.parametrize("plot", PLOTS)
def test_each_plot_draws_or_refuses_each_family(fits: Any, family: str, plot: str) -> None:
    assert outcome(plot, fits(family)) == EXPECTED[family][plot]


def test_every_public_plot_is_in_the_matrix() -> None:
    assert set(plotting.__all__) == set(PLOTS) | set(OTHER_INPUTS)


def test_every_family_pins_every_plot() -> None:
    assert set(EXPECTED) == set(FAMILIES)
    for family, cells in EXPECTED.items():
        assert set(cells) == set(PLOTS), family


def test_a_crash_is_not_a_refusal() -> None:
    """Mutation control: an exception no ``raise`` of cleverly made propagates."""

    class Broken:
        estimates: dict[str, Any] = {"psi": 0.0}  # noqa: RUF012
        repeats = (object(), object())

    with pytest.raises(AttributeError):
        outcome("repeats", Broken())


@_SHARED
@pytest.mark.parametrize(
    "crash",
    [lambda: tuple(None), lambda: float("not a number")],  # type: ignore[arg-type]
    ids=["TypeError", "ValueError"],
)
def test_a_builtin_error_inside_a_plot_is_a_crash(
    fits: Any, monkeypatch: pytest.MonkeyPatch, crash: Callable[[], Any]
) -> None:
    """Mutation control: a built-in TypeError or ValueError from inside a plot fails the cell."""
    from cleverly.plotting import _diagnostics

    def broken(result: Any, name: str | None) -> Any:
        return crash()

    monkeypatch.setattr(_diagnostics, "_estimate_of", broken)
    with pytest.raises((TypeError, ValueError)):
        outcome("repeats", fits("point_tmle_repeats"))


def _public_result_classes() -> dict[str, type]:
    found: dict[str, type] = {}
    for module in pkgutil.walk_packages(cleverly.__path__, "cleverly."):
        if any(part.startswith("_") for part in module.name.split(".")[1:]):
            continue
        loaded = importlib.import_module(module.name)
        for name in getattr(loaded, "__all__", ()):
            value = getattr(loaded, name, None)
            if inspect.isclass(value) and name.endswith("Result"):
                found[name] = value
    return found


@_SHARED
def test_every_public_result_class_has_a_family_or_a_reason(fits: Any) -> None:
    covered = {type(fits(family)).__name__ for family in FAMILIES}
    shipped = set(_public_result_classes())
    assert shipped - covered - set(NOT_FITS) == set()
    assert set(NOT_FITS) <= shipped, "a NOT_FITS entry names no shipped class"
    assert not covered & set(NOT_FITS)


@pytest.mark.parametrize("kind", list(sweep.KINDS))
def test_no_plot_crashes_on_a_capability_sweep_kind(kind: str) -> None:
    """Every fit kind of the capability sweep: each plot draws or refuses, never crashes."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sweep.KINDS[kind].build()
    for plot in PLOTS:
        outcome(plot, result)


def test_the_module_table_lists_every_plot() -> None:
    listed = set(re.findall(r"^:func:`(\w+)`", plotting.__doc__ or "", flags=re.M))
    assert listed == set(PLOTS) | {"sensitivity", "confounding", "targeting", "zipper"}


@_SHARED
def test_repeats_draws_each_longitudinal_draw(fits: Any) -> None:
    result = fits("ltmle_repeats")
    name = next(iter(result.estimates))
    wanted = [float(draw.estimates[name].psi) for draw in result.repeats]
    # The draws differ, so a reader that repeats one draw cannot pass.
    assert len(set(wanted)) == len(wanted)
    fig = plotting.repeats(result, name=name)
    drawn = [line for line in fig.axes[0].get_lines() if line.get_marker() == "o"]
    assert np.asarray(drawn[0].get_xdata(), dtype=float).tolist() == pytest.approx(wanted)
    # The reported estimate of a repeated fit is the median of its draws, drawn as a line.
    reported = float(result.estimates[name].psi)
    assert reported == pytest.approx(float(np.median(wanted)))
    flat = [
        float(np.asarray(line.get_xdata(), dtype=float)[0])
        for line in fig.axes[0].get_lines()
        if np.ptp(np.asarray(line.get_xdata(), dtype=float)) == 0
    ]
    assert any(value == pytest.approx(reported) for value in flat)
    plt.close(fig)


@_SHARED
def test_overlap_names_a_held_baseline_treatment(fits: Any) -> None:
    with pytest.raises(CapabilityError) as caught:
        plotting.overlap(fits("time_to_event_ltmle"))
    assert "per time point" not in str(caught.value)
    assert "once at baseline" in str(caught.value)


@_SHARED
def test_calibration_does_not_call_a_hazard_library_a_continuous_outcome(fits: Any) -> None:
    with pytest.raises(CapabilityError) as caught:
        plotting.calibration(fits("continuous_survival"))
    assert "continuous outcome" not in str(caught.value)
