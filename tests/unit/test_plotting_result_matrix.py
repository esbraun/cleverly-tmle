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

import re
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
from sklearn.linear_model import LinearRegression, LogisticRegression

import cleverly.plotting as plotting
from cleverly import (
    ATE,
    AdaptiveTMLEMethod,
    CausalStudy,
    CollaborativeTMLEMethod,
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
from cleverly.datasets import make_linear_ate, make_longitudinal_survival, make_point_survival
from cleverly.exceptions import CapabilityError
from cleverly.longitudinal import LTMLE

pytestmark = pytest.mark.xdist_group("plotting_result_matrix")

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
        repeats=2,
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


#: One small fit per result family.
FAMILIES: dict[str, Callable[[], Any]] = {
    "point_tmle_repeats": _point_tmle,
    "ltmle_repeats": _ltmle_repeats,
    "time_to_event_ltmle": _time_to_event,
    "time_to_event_ctmle": _survival_method(CollaborativeTMLEMethod()),
    "time_to_event_drtmle": _survival_method(DRTMLEMethod()),
    "continuous_survival": _continuous_survival,
    "adaptive_tmle": _adaptive,
}

_LONGITUDINAL = {
    "forest": "draws",
    "curve": "draws",
    "influence": "draws",
    "overlap": "X100",
    "weights": "draws",
    "balance": "X100",
    "learners": "X100",
    "calibration": "X100",
    "repeats": "refuses",
    "bootstrap": "refuses",
}

#: The pinned outcome of each plot on each family.
EXPECTED: dict[str, dict[str, str]] = {
    "point_tmle_repeats": {
        "forest": "draws",
        "curve": "refuses",
        "influence": "draws",
        "overlap": "draws",
        "weights": "refuses",
        "balance": "draws",
        "learners": "refuses",
        "calibration": "draws",
        "repeats": "draws",
        "bootstrap": "refuses",
    },
    "ltmle_repeats": {**_LONGITUDINAL, "repeats": "draws"},
    "time_to_event_ltmle": _LONGITUDINAL,
    # The collaborative search withholds the interval, so there is no uncertainty to split.
    "time_to_event_ctmle": {**_LONGITUDINAL, "influence": "refuses"},
    "time_to_event_drtmle": _LONGITUDINAL,
    "continuous_survival": {
        "forest": "draws",
        "curve": "X100",
        "influence": "draws",
        "overlap": "X100",
        "weights": "refuses",
        "balance": "X100",
        "learners": "X100",
        "calibration": "X100",
        "repeats": "refuses",
        "bootstrap": "refuses",
    },
    "adaptive_tmle": {
        "forest": "draws",
        "curve": "refuses",
        "influence": "draws",
        "overlap": "X100",
        "weights": "refuses",
        "balance": "refuses",
        "learners": "X100",
        "calibration": "X100",
        "repeats": "refuses",
        "bootstrap": "refuses",
    },
}

_ROADMAP_ID = re.compile(r"\b([XFR]\d+) in docs/roadmap\.md\b")


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


def outcome(plot: str, result: Any) -> str:
    """``"draws"``, ``"refuses"`` or the roadmap id a refusal names; a crash raises."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = getattr(plotting, plot)(result)
    except CapabilityError as error:
        found = _ROADMAP_ID.search(str(error))
        return found.group(1) if found else "refuses"
    except (ValueError, TypeError):
        return "refuses"
    assert isinstance(fig, Figure)
    plt.close(fig)
    return "draws"


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
    """Mutation control: a plot that raises AttributeError fails the matrix, not passes it."""

    class Broken:
        estimates: dict[str, Any] = {"psi": 0.0}  # noqa: RUF012
        repeats = (object(), object())

    with pytest.raises(AttributeError):
        outcome("repeats", Broken())


def test_the_module_table_lists_every_plot() -> None:
    listed = set(re.findall(r"^:func:`(\w+)`", plotting.__doc__ or "", flags=re.M))
    assert listed == set(PLOTS) | {"sensitivity", "confounding", "targeting", "zipper"}


def test_repeats_draws_each_longitudinal_draw(fits: Any) -> None:
    result = fits("ltmle_repeats")
    name = next(iter(result.estimates))
    wanted = [float(draw.estimates[name].psi) for draw in result.repeats]
    # The draws differ, so a reader that repeats one draw cannot pass.
    assert len(set(wanted)) == len(wanted)
    fig = plotting.repeats(result, name=name)
    drawn = [line for line in fig.axes[0].get_lines() if line.get_marker() == "o"]
    assert np.asarray(drawn[0].get_xdata(), dtype=float).tolist() == pytest.approx(wanted)
    plt.close(fig)


def test_overlap_names_a_held_baseline_treatment(fits: Any) -> None:
    with pytest.raises(CapabilityError) as caught:
        plotting.overlap(fits("time_to_event_ltmle"))
    assert "per time point" not in str(caught.value)
    assert "once at baseline" in str(caught.value)


def test_calibration_does_not_call_a_hazard_library_a_continuous_outcome(fits: Any) -> None:
    with pytest.raises(CapabilityError) as caught:
        plotting.calibration(fits("continuous_survival"))
    assert "continuous outcome" not in str(caught.value)
