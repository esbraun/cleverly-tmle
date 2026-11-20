"""The fixtures of the memory-guard witnesses (``test_longitudinal_memory_guard.py``).

A port of the J2d calibration's fit builders (``cases.py`` and ``tfix.py`` in the plans
repository's ``rows/X106/J2/evidence/2026-11-19-j2d/rev9/``), restricted to the cases the fast
suite fits, so the witnesses fit the panels and the settings the constants were fitted on.  The
panel has two baseline covariates (plus ``extra`` noise columns), one time-varying covariate per
node after the first, a binary treatment per node, and an end-of-study outcome or an absorbing
event per node; ``causes=2`` adds a competing event per node, drawn after the default stream.
"""

from __future__ import annotations

import gc
import tracemalloc
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.datasets import make_point_survival
from cleverly.interventions import Incremental, Shift, Stochastic
from cleverly.learners import GLM
from cleverly.longitudinal import (
    IPSI,
    LTMLE,
    SDR,
    DynamicRegimen,
    LongitudinalData,
    Separable,
    _memory_guard,
)

SEED = 20261114
DEFAULTS: dict[str, Any] = {
    "J": 1,
    "P": 2,
    "repeats": 1,
    "boot": 0,
    "B": None,
    "k": 0,
    "outcome": "glm",
}
SHIFTS = (0.0, 0.2, -0.2, 0.4)
SURVIVAL = {"ltmle_surv", "sdr_surv", "ipsi_surv", "hz_back", "hz_path", "hz_policy"}


def panel(
    n: int, n_times: int, *, survival: bool, causes: int = 1, extra: int = 0
) -> tuple[pd.DataFrame, list[list[str]]]:
    """The calibration panel (``tfix.make`` without censoring and with binary treatments)."""
    rng = np.random.default_rng(SEED)
    d: dict[str, np.ndarray] = {"W1": rng.normal(size=n), "W2": rng.normal(size=n)}
    prev = np.zeros(n)
    covariate = np.zeros(n)
    blocks: list[list[str]] = []
    lin = 0.3 * d["W1"]
    for t in range(1, n_times + 1):
        block = []
        if t >= 2:
            covariate = 0.5 * covariate + 0.3 * prev + 0.3 * d["W1"] + rng.normal(size=n)
            d[f"L{t}"] = covariate.copy()
            block.append(f"L{t}")
        a = rng.binomial(
            1, expit(0.2 * d["W1"] - 0.2 * d["W2"] + 0.3 * covariate + 0.4 * prev)
        ).astype(float)
        d[f"A{t}"] = a
        blocks.append(block)
        prev = a
        lin = lin + 0.2 * a + 0.1 * covariate
        if survival:
            d[f"Y{t}"] = rng.binomial(1, expit(-2.5 + lin * 0.5)).astype(float)
    if not survival:
        d["Y"] = rng.binomial(1, expit(-0.4 + lin * 0.5)).astype(float)
    extra_rng = np.random.default_rng([SEED, 1])
    order = ["W1", "W2"]
    for t in range(1, n_times + 1):
        if t >= 2:
            order.append(f"L{t}")
        order.append(f"A{t}")
        if survival:
            order.append(f"Y{t}")
            if causes >= 2:
                d[f"D{t}"] = (extra_rng.random(n) < 0.03).astype(float) * (1.0 - d[f"Y{t}"])
                order.append(f"D{t}")
    if not survival:
        order.append("Y")
    for k in range(extra):
        d[f"X{k + 1}"] = extra_rng.normal(size=n)
    order = [f"X{k + 1}" for k in range(extra)] + order
    frame = pd.DataFrame({name: d[name] for name in order})
    if survival:
        stop = np.zeros(n, bool)
        for t in range(1, n_times + 1):
            names = [name for name in (f"Y{t}", f"D{t}") if name in frame.columns]
            hit = np.zeros(n, bool)
            for name in names:
                hit |= frame[name].to_numpy() == 1
            hit &= ~stop
            frame.loc[hit, order[max(order.index(name) for name in names) + 1 :]] = np.nan
            stop = stop | hit
    return frame, blocks


def _logistic() -> LogisticRegression:
    return LogisticRegression(max_iter=1000)


def _settings(case: dict[str, Any]) -> dict[str, Any]:
    hazard = case["est"].startswith("hz_")
    kw: dict[str, Any] = {
        "outcome_learner": _logistic() if hazard else GLM(family="binomial"),
        "treatment_learner": _logistic(),
        "censoring_learner": _logistic(),
        "n_folds": case["folds"],
        "random_state": SEED,
        "pseudo_learner": LinearRegression(),
        "simultaneous": False,
    }
    if case["J"] > 1 and hazard:
        kw["competing_learner"] = _logistic()
    if case["outcome"] == "linear" and not hazard:
        kw["outcome_learner"] = LinearRegression()
    if hazard:
        kw["survival_model"] = "hazard"
        kw.pop("pseudo_learner")
    if case["est"] in ("hz_back", "hz_separable"):
        kw["targeting"] = "one_step"
        if case["est"] == "hz_back":
            kw["pseudo_learner"] = LinearRegression()
    if case["repeats"] > 1:
        kw["repeats"] = case["repeats"]
    if case["boot"]:
        kw["n_bootstrap"] = case["boot"]
    return kw


def _static(count: int, n_times: int) -> dict[str, Any]:
    patterns: dict[str, Any] = {
        "always": 1,
        "never": 0,
        "first": (1,) + (0,) * (n_times - 1),
        "last": (0,) * (n_times - 1) + (1,),
    }
    return dict(list(patterns.items())[:count])


def _tilts(count: int) -> dict[str, Incremental]:
    tilts = {
        "double": Incremental(2.0),
        "half": Incremental(0.5),
        "up": Incremental(1.5),
        "down": Incremental(0.75),
    }
    return dict(list(tilts.items())[:count])


def _coin(node: int, random: bool) -> Stochastic:
    def density(history: Any) -> np.ndarray:
        out = np.zeros((len(history), 2))
        if random:
            out[:] = 0.5
        else:
            out[:, 1] = 1.0
        return out

    return Stochastic(density, f"coin {node}", density_kind="known")


def _point(n: int, n_times: int, causes: int) -> LongitudinalData:
    frame, _ = make_point_survival(n=n, seed=SEED, n_times=n_times, causes=causes)
    labels: dict[str, Any] = {"causes": {1: "relapse", 2: "death"}} if causes == 2 else {}
    return LongitudinalData.from_time_to_event(
        frame,
        time="time",
        event="event",
        treatment="A",
        baseline=["W1", "W2"],
        grid=list(range(1, n_times + 1)),
        **labels,
    )


def build(case: dict[str, Any], n: int, **overrides: Any) -> tuple[Any, Callable[[], Any]]:
    """``(estimator, call)`` for one calibration case; ``call()`` runs its fit.

    A case names ``est`` (one of ``ltmle_eos``, ``ltmle_surv``, ``sdr_surv``, ``ipsi_surv``,
    ``mtp``, ``sdr_mtp``, ``hz_point``, ``hz_back``, ``hz_path``, ``hz_policy`` and
    ``hz_separable``), ``T`` nodes, ``R`` regimens, ``folds`` and ``H`` reported horizons, and
    optionally ``J`` causes, ``P`` baseline columns, ``repeats``, ``boot`` replicates, ``B``
    density bins, ``k`` the coin nodes of a policy path fit and ``outcome`` (``"linear"`` for
    a least-squares outcome regression).  ``overrides`` replace estimator settings.
    """
    case = {**DEFAULTS, **case}
    est, n_times, count, horizons = case["est"], case["T"], case["R"], case["H"]
    kw = {**_settings(case), **overrides}
    last = list(range(n_times - horizons + 1, n_times + 1))
    if est in ("hz_point", "hz_separable"):
        data = _point(n, n_times, case["J"])
        kw["horizons"] = last
        regimens: Any
        if est == "hz_separable":
            regimens = {
                "a0": 0,
                "a1": 1,
                "sep": Separable(arms={"death": 0, "relapse": 1}, first="death"),
                "pes": Separable(arms={"death": 1, "relapse": 0}, first="death"),
            }
        else:
            regimens = _static(count, 1)
        engine = LTMLE(regimens, **kw)
        return engine, lambda: engine.fit(data)
    survival = est in SURVIVAL
    frame, blocks = panel(n, n_times, survival=survival, causes=case["J"], extra=case["P"] - 2)
    nodes = range(1, n_times + 1)
    outcome: Any
    if survival and case["J"] >= 2:
        outcome = {"event": tuple(f"Y{t}" for t in nodes), "other": tuple(f"D{t}" for t in nodes)}
    else:
        outcome = [f"Y{t}" for t in nodes] if survival else "Y"
    columns: dict[str, Any] = {
        "outcome": outcome,
        "treatment": [f"A{t}" for t in nodes],
        "baseline": ["W1", "W2"] + [f"X{k + 1}" for k in range(case["P"] - 2)],
        "time_varying": [[] for _ in nodes] if est in ("hz_path", "hz_policy") else blocks,
    }
    if survival:
        kw["horizons"] = last
    cls: Any = LTMLE
    regimens = _static(count, n_times)
    if est == "sdr_surv":
        cls = SDR
    elif est == "ipsi_surv":
        cls, regimens = IPSI, _tilts(count)
    elif est in ("mtp", "sdr_mtp"):
        rng = np.random.default_rng([SEED, 2])
        prev = np.zeros(n)
        for t in nodes:
            frame[f"A{t}"] = 0.4 * frame["W1"].to_numpy() + 0.3 * prev + rng.normal(size=n)
            prev = frame[f"A{t}"].to_numpy()
        frame["Y"] = rng.binomial(
            1, expit(-0.4 + 0.2 * prev + 0.3 * frame["W1"].to_numpy())
        ).astype(float)
        regimens = {f"shift{k}": Shift(SHIFTS[k], cap=None) for k in range(count)}
        columns["continuous_treatment"] = columns["treatment"]
        if case["B"]:
            kw["density_bins"] = case["B"]
        cls = SDR if est == "sdr_mtp" else LTMLE
    elif est == "hz_policy":
        regimens = dict(_static(count, n_times))
        regimens["coin"] = DynamicRegimen("coin", tuple(_coin(t, t <= case["k"]) for t in nodes))
    engine = cls(regimens, **kw)
    return engine, lambda: engine.fit(frame, **columns)


@contextmanager
def captured_counts() -> Iterator[list[_memory_guard.FitCounts]]:
    """Record the counts of every guarded call inside the block."""
    seen: list[_memory_guard.FitCounts] = []
    real = _memory_guard.fit_counts

    def spy(*args: Any, **kwargs: Any) -> _memory_guard.FitCounts:
        counts = real(*args, **kwargs)
        seen.append(counts)
        return counts

    _memory_guard.fit_counts = spy
    try:
        yield seen
    finally:
        _memory_guard.fit_counts = real


def traced(case: dict[str, Any], n: int) -> tuple[int, _memory_guard.FitCounts]:
    """The traced peak of one fit after one untraced warm-up fit, and the guard's counts.

    The panel is built outside the traced window; the peak counts what the fit allocates
    above the bytes traced when it starts.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        build(case, n)[1]()
        _, call = build(case, n)
        with captured_counts() as seen:
            gc.collect()
            tracemalloc.start()
            try:
                start = tracemalloc.get_traced_memory()[0]
                call()
                peak = tracemalloc.get_traced_memory()[1] - start
            finally:
                tracemalloc.stop()
    return peak, seen[0]
