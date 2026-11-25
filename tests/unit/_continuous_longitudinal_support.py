"""Saturated weighted learners and fit helpers for the continuous-time longitudinal tests.

Each learner is the cell estimator on the unique rows of its design, weighted through the
private row-weight seam, so on a law passed by weight (``tests/discrete_law_continuous_longitudinal.py``)
it returns the law's own conditional: the hazard learner is the weighted Nelson-Aalen estimator
with delayed entry in each cell, the classifier the weighted cell mean of the treatment, and the
regressor the weighted cell mean of the continuation fraction.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from cleverly.survival import longitudinal
from cleverly.survival._longitudinal_data import ContinuousLongitudinalData, weighted
from cleverly.survival.longitudinal import ContinuousLongitudinalTMLE

#: The column of ``L(s_k)`` in the window design ``(A(s_k), W, L(s_k), A(s_{k-1}), k, s_k)``;
#: a ``NoL`` learner drops it.
L_COLUMN = 2
#: The column of ``L(s_k)`` in the treatment design, the window without ``A(s_k)``.
L_TREATMENT = 1


def _keys(X: np.ndarray, drop: tuple[int, ...] = ()) -> list[tuple[float, ...]]:
    X = np.delete(np.asarray(X, dtype=float), list(drop), axis=1)
    return [tuple(np.round(row, 12)) for row in X]


class CellHazard(BaseEstimator):
    """The weighted Nelson-Aalen estimator with delayed entry in each cell of the design."""

    def __init__(self, drop: tuple[int, ...] = ()) -> None:
        self.drop = drop

    def fit(
        self,
        X: np.ndarray,
        time: np.ndarray,
        event: np.ndarray,
        entry: np.ndarray | None = None,
        weight: np.ndarray | None = None,
    ) -> CellHazard:
        time = np.asarray(time, dtype=float)
        event = np.asarray(event, dtype=float)
        entry = np.zeros_like(time) if entry is None else np.asarray(entry, dtype=float)
        weight = np.ones_like(time) if weight is None else np.asarray(weight, dtype=float)
        keys = _keys(X, self.drop)
        self.cells_: dict[tuple[float, ...], tuple[np.ndarray, np.ndarray]] = {}
        for key in dict.fromkeys(keys):
            rows = np.array([k == key for k in keys])
            t, d, e, w = time[rows], event[rows], entry[rows], weight[rows]
            jumps = np.unique(t[d > 0])
            values = np.array(
                [np.sum(w * (d > 0) * (t == u)) / np.sum(w * ((e < u) & (t >= u))) for u in jumps]
            )
            self.cells_[key] = (jumps, values)
        return self

    def cumulative_hazard(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        times = np.asarray(times, dtype=float)
        out = np.zeros((len(X), times.size))
        for i, key in enumerate(_keys(X, self.drop)):
            cell = self.cells_.get(key)
            if cell is None:
                continue
            jumps, values = cell
            if jumps.size == 0:
                continue
            seen = np.searchsorted(jumps, times, side="right")
            out[i] = np.where(seen > 0, np.cumsum(values)[np.maximum(seen - 1, 0)], 0.0)
        return out


class CellClassifier(ClassifierMixin, BaseEstimator):
    """The weighted cell mean of a binary label."""

    def __init__(self, drop: tuple[int, ...] = ()) -> None:
        self.drop = drop

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> CellClassifier:
        y = np.asarray(y, dtype=float)
        w = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        keys = _keys(X, self.drop)
        self.classes_ = np.array([0, 1])
        totals: dict[tuple[float, ...], list[float]] = {}
        for key, label, weight in zip(keys, y, w, strict=True):
            cell = totals.setdefault(key, [0.0, 0.0])
            cell[0] += weight * label
            cell[1] += weight
        self.cells_ = {key: a / b for key, (a, b) in totals.items()}
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = np.array([self.cells_.get(key, 0.5) for key in _keys(X, self.drop)])
        return np.column_stack([1.0 - p, p])


class CellRegressor(RegressorMixin, BaseEstimator):
    """The weighted cell mean of a response."""

    def __init__(self, drop: tuple[int, ...] = ()) -> None:
        self.drop = drop

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> CellRegressor:
        y = np.asarray(y, dtype=float)
        w = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        totals: dict[tuple[float, ...], list[float]] = {}
        for key, value, weight in zip(_keys(X, self.drop), y, w, strict=True):
            cell = totals.setdefault(key, [0.0, 0.0])
            cell[0] += weight * value
            cell[1] += weight
        self.cells_ = {key: a / b if b > 0 else 0.0 for key, (a, b) in totals.items()}
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.cells_.get(key, 0.0) for key in _keys(X, self.drop)])


def law_data(law: Any) -> ContinuousLongitudinalData:
    """The law's paths as a container, each unit weighted by its path's probability."""
    frame = law.frame()
    data = ContinuousLongitudinalData.from_frame(
        frame.drop(columns="weight"),
        id="id",
        start="start",
        stop="stop",
        event="event",
        treatment="A",
        baseline=["W"],
        time_varying=["L"],
        decision="decision",
    )
    return weighted(data, law.unit_weights())


def saturated(
    monkeypatch: Any, horizons: tuple[float, ...], regimes: Mapping[str, Any], **kwargs: Any
) -> ContinuousLongitudinalTMLE:
    """An in-sample estimator with the saturated learners (C13 lifted for the test)."""
    monkeypatch.setattr(longitudinal, "_user_learners_in_sample", lambda estimator: None)
    settings: dict[str, Any] = {
        "cross_fit": False,
        "hazard_library": [CellHazard()],
        "visit_library": [CellHazard()],
        "censoring_library": [CellHazard()],
        "treatment_learner": CellClassifier(),
        "continuation_learner": CellRegressor(),
        "g_bounds": 1e-6,
        "tol": 1e-12,
        "max_iter": 200,
        "random_state": 0,
    }
    settings.update(kwargs)
    return ContinuousLongitudinalTMLE(horizons, regimes=regimes, **settings)


#: The learner settings of the exact-law tests: saturated, or one side omitting ``L``.
LEARNERS: dict[str, dict[str, Any]] = {
    "saturated": {},
    # Wrong intensities and continuation, right mechanism: the walk must repair the estimate.
    "nol": {
        "hazard_library": [CellHazard((L_COLUMN,))],
        "visit_library": [CellHazard((L_COLUMN,))],
        "continuation_learner": CellRegressor((L_COLUMN,)),
    },
    # Right intensities and continuation, a mechanism omitting ``L``.
    "gnol": {"treatment_learner": CellClassifier((L_TREATMENT,))},
}


def rule_when_l(frame: Any) -> Any:
    """The engine's form of the law's rule ``treat_when_l``: treat when the current L is 1."""
    return (frame["L"] == 1).astype(int)


def exact_errors(
    monkeypatch: Any,
    law: Any,
    regimes: Mapping[str, tuple[Any, Any]],
    *,
    learners: str = "saturated",
    cap: int | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """The largest error of the estimates and of their influence curves against the law.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Lifts C13 for the in-sample saturated learners.
    law : Law
        The law, passed by weight.
    regimes : mapping of str to (engine regime, law regime)
        The regimes.
    learners : {"saturated", "nol", "gnol"}
        The learner setting.
    cap : int or None
        The visit cap of the truth.
    **kwargs : Any
        Estimator settings.

    Returns
    -------
    dict of str to float
        ``psi`` and ``ic`` (the influence curve's error, meaningful under ``"saturated"``).
    """
    import warnings

    from tests.discrete_law_continuous_longitudinal import CAUSES

    settings = {"tol": 1e-10, "max_iter": 4000, **LEARNERS[learners], **kwargs}
    estimator = saturated(
        monkeypatch, law.targets, {k: v[0] for k, v in regimes.items()}, **settings
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = estimator.fit_data(law_data(law))
    targets = [(c, t) for c in CAUSES for t in law.targets]
    psi = ic = 0.0
    for label, (_, regime) in regimes.items():
        names = [f"cif_regimen[{label}, {c} @ t={t:g}]" for c, t in targets]
        got = np.array([result[name].psi for name in names])
        curve = np.column_stack([result[name].influence_curve for name in names])
        psi = max(psi, float(np.max(np.abs(got - law.truth(regime, targets, cap)))))
        ic = max(ic, float(np.max(np.abs(curve - law.gateaux(regime, targets, cap)))))
    return {"psi": psi, "ic": ic}


def _expit(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def simulate_ct(
    n: int, seed: int, *, end: float = 30.0, visit_rate: float = 0.10, visits: bool = True
) -> Any:
    """A draw of law CT (X6 plan 5.7): binary A and L at visits, two causes, censoring.

    Parameters
    ----------
    n : int
        Units.
    seed : int
        Seed.
    end : float
        The administrative end of follow-up, in days.
    visit_rate : float
        The visit intensity's base rate per day.
    visits : bool
        ``False`` draws no visit after baseline.

    Returns
    -------
    pandas.DataFrame
        ``id, start, stop, event, W1, W2, L, A``, one row per interval between visits.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(n):
        w1, w2 = int(rng.random() < 0.5), int(rng.integers(1, 7))
        l = int(rng.random() < _expit(-0.5 + 0.6 * w1))
        a = int(rng.random() < _expit(-0.3 + 0.8 * l + 0.3 * w1))
        start = t = 0.0
        while True:
            rates = np.array(
                [
                    0.010 * np.exp(-0.6 * a + 0.8 * l + 0.3 * w1),
                    0.006 * np.exp(0.2 * a + 0.5 * l - 0.3 * w1 + 0.1 * w2),
                    visit_rate * np.exp(0.4 * l - 0.3 * a) if visits else 0.0,
                    0.008 * np.exp(0.6 * l - 0.4 * a),
                ]
            )
            t = t + rng.exponential(1.0 / rates.sum())
            row = {"id": unit, "start": start, "W1": w1, "W2": w2, "L": l, "A": a}
            if t >= end:
                rows.append({**row, "stop": end, "event": 0})
                break
            kind = int(rng.choice(4, p=rates / rates.sum()))
            if kind in (0, 1):
                rows.append({**row, "stop": t, "event": kind + 1})
                break
            if kind == 3:
                rows.append({**row, "stop": t, "event": 0})
                break
            rows.append({**row, "stop": t, "event": 0})
            l_new = int(rng.random() < _expit(-0.4 + 1.2 * l - 0.8 * a + 0.4 * w1))
            a = int(rng.random() < _expit(-1.0 + 1.0 * l_new + 2.0 * a + 0.2 * w2 / 6))
            l, start = l_new, t
    return pd.DataFrame.from_records(rows)[["id", "start", "stop", "event", "W1", "W2", "L", "A"]]


#: The columns of :func:`simulate_ct` for ``fit``.
CT_COLUMNS: dict[str, Any] = {
    "id": "id",
    "start": "start",
    "stop": "stop",
    "event": "event",
    "treatment": "A",
    "baseline": ["W1", "W2"],
    "time_varying": ["L"],
}
