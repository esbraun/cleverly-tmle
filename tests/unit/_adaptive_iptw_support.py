"""Finite laws and test-side solvers for the AdaptiveIPTW exact-law tests.

The shipped estimator is checked against values this module computes in numpy from its own
inputs. Nothing here calls the library's solver, so an assertion that compares the two counts as
independent evidence.

The law has eight support points of ``W``, a binary ``A`` and a binary ``Y``. One row stands for
each ``(W, A, Y)`` cell and carries its population probability as a weight, so the weighted
sample is the law. The fixed mechanism ``G_TILDE`` differs from the true one ``G`` and ties on
two pairs of points. In each pair the two points have different ``G`` and different ``Q``, so
the regression of ``Y`` on the tied value differs from the regression on ``W``.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

#: ``P(W = w)`` over the eight support points.
P_W = np.array([0.10, 0.15, 0.10, 0.15, 0.05, 0.20, 0.10, 0.15])

#: ``g(w) = P(A = 1 | W = w)``.
G = np.array([0.20, 0.35, 0.50, 0.65, 0.80, 0.30, 0.70, 0.45])

#: A fixed fitted mechanism. Points 0 and 5 tie at 0.25, and points 2 and 6 tie at 0.60.
G_TILDE = np.array([0.25, 0.40, 0.60, 0.70, 0.75, 0.25, 0.60, 0.50])

#: ``Qbar(a, w) = P(Y = 1 | A = a, w)``, shape ``(8, 2)``. Each tied pair differs in both arms.
Q = np.array(
    [
        [0.30, 0.20],
        [0.45, 0.50],
        [0.25, 0.30],
        [0.60, 0.70],
        [0.35, 0.40],
        [0.55, 0.60],
        [0.70, 0.80],
        [0.40, 0.35],
    ]
)

SUPPORT = tuple(itertools.product(range(8), range(2), range(2)))


class TableClassifier(BaseEstimator, ClassifierMixin):
    """A fixed binary classifier: ``P(A = 1 | W = w)`` is ``table[w]`` and nothing is fitted."""

    def __init__(self, table: Sequence[float] = ()) -> None:
        self.table = table

    def fit(self, x: Any, y: Any, sample_weight: Any = None) -> TableClassifier:
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, x: Any) -> np.ndarray:
        index = np.rint(np.asarray(x, dtype=float)[:, 0]).astype(int)
        one = np.asarray(self.table, dtype=float)[index]
        return np.column_stack([1.0 - one, one])


class SpyClassifier(TableClassifier):
    """A table classifier that counts its ``fit`` calls on a class-level list."""

    calls: list[int] = []  # noqa: RUF012 - the counter is shared across clones on purpose

    def fit(self, x: Any, y: Any, sample_weight: Any = None) -> SpyClassifier:
        type(self).calls.append(int(np.asarray(x).shape[0]))
        super().fit(x, y, sample_weight)
        return self


class SaturatedRegressor(BaseEstimator, RegressorMixin):
    """The weighted mean of the target at each distinct value of the first design column."""

    def __init__(self, digits: int = 12) -> None:
        self.digits = digits

    def fit(self, x: Any, y: Any, sample_weight: Any = None) -> SaturatedRegressor:
        design = np.round(np.asarray(x, dtype=float)[:, 0], self.digits)
        target = np.asarray(y, dtype=float)
        weight = np.ones_like(target) if sample_weight is None else np.asarray(sample_weight)
        self.means_ = {
            float(value): float(
                np.sum(weight[design == value] * target[design == value])
                / np.sum(weight[design == value])
            )
            for value in np.unique(design)
        }
        self.overall_ = float(np.sum(weight * target) / np.sum(weight))
        return self

    def predict(self, x: Any) -> np.ndarray:
        design = np.round(np.asarray(x, dtype=float)[:, 0], self.digits)
        return np.array([self.means_.get(float(value), self.overall_) for value in design])


class SpyRegressor(SaturatedRegressor):
    """A saturated regressor that counts its ``fit`` calls on a class-level list."""

    calls: list[int] = []  # noqa: RUF012 - the counter is shared across clones on purpose

    def fit(self, x: Any, y: Any, sample_weight: Any = None) -> SpyRegressor:
        type(self).calls.append(int(np.asarray(x).shape[0]))
        super().fit(x, y, sample_weight)
        return self


def law_frame(
    g: Sequence[float] = G, q: np.ndarray = Q, p_w: Sequence[float] = P_W
) -> pd.DataFrame:
    """The 32-row weighted frame of the law: columns ``W``, ``A``, ``Y`` and ``wt``."""
    rows = []
    for w, a, y in SUPPORT:
        arm = g[w] if a == 1 else 1.0 - g[w]
        outcome = q[w, a] if y == 1 else 1.0 - q[w, a]
        rows.append({"W": float(w), "A": float(a), "Y": float(y), "wt": p_w[w] * arm * outcome})
    return pd.DataFrame(rows)


def expit(x: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -x))


def logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1.0 - p))


def saturated_regression(
    design: np.ndarray, target: np.ndarray, mask: np.ndarray, weight: np.ndarray
) -> np.ndarray:
    """The weighted mean of ``target`` over the ``mask`` rows at each design value, for all rows."""
    keys = np.round(design, 12)
    out = np.empty_like(target, dtype=float)
    for value in np.unique(keys):
        rows = (keys == value) & mask
        out[keys == value] = np.sum(weight[rows] * target[rows]) / np.sum(weight[rows])
    return out


def tilt_root(
    indicator: np.ndarray,
    g: np.ndarray,
    h: np.ndarray,
    weight: np.ndarray,
    bounds: tuple[float, float] | None = None,
) -> float:
    """The root ``eps`` of ``sum w h (I - clip(expit(logit g + eps h))) = 0`` by bisection."""
    lo, hi = (0.0, 1.0) if bounds is None else bounds

    def score(eps: float) -> float:
        tilted = np.clip(expit(logit(g) + eps * h), lo, hi)
        return float(np.sum(weight * h * (indicator - tilted)))

    low, high = -60.0, 60.0
    f_low, f_high = score(low), score(high)
    if f_low * f_high > 0.0:
        raise AssertionError("the tilt has no root in [-60, 60]")
    for _ in range(300):
        mid = 0.5 * (low + high)
        f_mid = score(mid)
        if f_mid * f_low > 0.0:
            low, f_low = mid, f_mid
        else:
            high = mid
    return 0.5 * (low + high)


def design_logit_regression(
    design: np.ndarray, target: np.ndarray, mask: np.ndarray, weight: np.ndarray
) -> np.ndarray:
    """The weighted logistic regression of ``target`` on ``logit(design)``, by test-side IRLS."""
    x = np.column_stack([np.ones(design.shape[0]), logit(design)])
    xm, ym, wm = x[mask], target[mask], weight[mask]
    beta = np.zeros(2)
    for _ in range(100):
        mu = expit(xm @ beta)
        variance = np.clip(mu * (1.0 - mu), 1e-12, None)
        z = xm @ beta + (ym - mu) / variance
        information = (xm * (wm * variance)[:, None]).T @ xm
        step = np.linalg.solve(information, (xm * (wm * variance)[:, None]).T @ z)
        if np.max(np.abs(step - beta)) < 1e-14:
            beta = step
            break
        beta = step
    return expit(x @ beta)


def fixed_point(
    indicator: np.ndarray,
    target: np.ndarray,
    g0: np.ndarray,
    weight: np.ndarray,
    *,
    regress: Any = saturated_regression,
    bounds: tuple[float, float] | None = None,
    refit: bool = True,
    tolerance: float = 1e-13,
    max_rounds: int = 200,
) -> dict[str, Any]:
    """The test-side targeting loop of one key.

    Each round regresses ``target`` on the current mechanism among the rows with
    ``indicator == 1``, forms ``h = Qr / g``, and moves ``g`` along ``h`` to the root of the
    score. It stops when the score of the next regression is below ``tolerance``. With
    ``refit=False`` the regression stays at its first value, which a mutation of the refit
    would use. With ``bounds`` the tilted mechanism is clipped after every step.

    Returns
    -------
    dict
        ``g`` (final mechanism), ``qr`` (final regression), ``eps`` (the steps), ``psi`` (the
        Horvitz-Thompson mean ``sum w I T / g / sum w``), ``score``, ``curve`` (the row curve
        ``I T / g - Qr (I - g) / g - psi``) and ``qr0`` (the first regression).
    """
    lo, hi = (0.0, 1.0) if bounds is None else bounds
    g = np.clip(np.asarray(g0, dtype=float), lo, hi)
    mask = indicator > 0.0
    steps: list[float] = []
    qr = regress(g, target, mask, weight)
    qr0 = qr
    for _ in range(max_rounds):
        h = qr / g
        score = float(np.sum(weight * h * (indicator - g)))
        if abs(score) < tolerance:
            break
        eps = tilt_root(indicator, g, h, weight, bounds)
        steps.append(eps)
        g = np.clip(expit(logit(g) + eps * h), lo, hi)
        if refit:
            qr = regress(g, target, mask, weight)
    h = qr / g
    psi = float(np.sum(weight * indicator * target / g) / np.sum(weight))
    return {
        "g": g,
        "qr": qr,
        "qr0": qr0,
        "eps": steps,
        "psi": psi,
        "score": float(np.sum(weight * h * (indicator - g))),
        "curve": indicator * target / g - qr * (indicator - g) / g - psi,
    }


class FunctionRegressor(BaseEstimator, RegressorMixin):
    """A fixed regression ``x -> fn(x[:, 0])``, for an exact population ``Qr``."""

    def __init__(self, fn: Any = None) -> None:
        self.fn = fn

    def fit(self, x: Any, y: Any, sample_weight: Any = None) -> FunctionRegressor:
        return self

    def predict(self, x: Any) -> np.ndarray:
        return np.asarray(self.fn(np.asarray(x, dtype=float)[:, 0]), dtype=float)


class LogitGLM(BaseEstimator, ClassifierMixin):
    """An unpenalised logistic regression on ``logit`` of the first design column."""

    def __init__(self) -> None:
        pass

    def fit(self, x: Any, y: Any, sample_weight: Any = None) -> LogitGLM:
        from sklearn.linear_model import LogisticRegression

        self.model_ = LogisticRegression(penalty=None, tol=1e-12, max_iter=2000)
        self.model_.fit(logit(np.asarray(x, dtype=float)[:, :1]), y, sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, x: Any) -> np.ndarray:
        return self.model_.predict_proba(logit(np.asarray(x, dtype=float)[:, :1]))


def fit_arm(
    frame: pd.DataFrame | None = None,
    *,
    g_tilde: Sequence[float] = G_TILDE,
    estimands: Sequence[str] | str | None = ("ey1", "ey0", "ate"),
    regressor: Any = None,
    weights: str | None = "wt",
    covariates: Sequence[str] = ("W",),
    fit: dict[str, Any] | None = None,
    **settings: Any,
) -> Any:
    """The shipped in-sample fit on the finite law with a fixed mechanism and a saturated ``Qr``."""
    from cleverly.estimators import AdaptiveIPTW

    base: dict[str, Any] = {
        "treatment_learner": TableClassifier(list(g_tilde)),
        "reduced_outcome_learner": SaturatedRegressor() if regressor is None else regressor,
        "cross_fit": False,
        "g_bounds": (1e-3, 1 - 1e-3),
        "tol": 1e-12,
        "max_iter": 200,
        "simultaneous": False,
        "random_state": 0,
        "estimands": estimands
        if estimands is None or isinstance(estimands, str)
        else list(estimands),
    }
    base.update(settings)
    return (
        AdaptiveIPTW(**base)
        .fit(
            law_frame() if frame is None else frame,
            outcome="Y",
            treatment="A",
            covariates=list(covariates),
            weights=weights,
            **(fit or {}),
        )
        .single()
    )


def arm_inputs(
    frame: pd.DataFrame, g_tilde: Sequence[float] = G_TILDE
) -> dict[int, dict[str, Any]]:
    """Per arm: the indicator, the mechanism and the target the test-side loop reads."""
    w_index = frame["W"].to_numpy(dtype=int)
    a = frame["A"].to_numpy(dtype=float)
    y = frame["Y"].to_numpy(dtype=float)
    g1 = np.asarray(g_tilde, dtype=float)[w_index]
    return {
        1: {"indicator": (a == 1).astype(float), "g": g1, "target": y},
        0: {"indicator": (a == 0).astype(float), "g": 1.0 - g1, "target": y},
    }


def mean_one(weight: np.ndarray) -> np.ndarray:
    return weight / np.mean(weight)


class CellClassifier(BaseEstimator, ClassifierMixin):
    """A classifier fitted as the weighted class frequencies in each distinct design row.

    A design row the fit never saw gets the overall class frequencies. It makes a nuisance
    that is saturated on the cells of a finite law, and it is refitted on each training fold.
    """

    def __init__(self, digits: int = 9) -> None:
        self.digits = digits

    def _key(self, row: np.ndarray) -> tuple[float, ...]:
        return tuple(np.round(row, self.digits))

    def fit(self, x: Any, y: Any, sample_weight: Any = None) -> CellClassifier:
        design = np.asarray(x, dtype=float)
        if design.ndim == 1:
            design = design[:, None]
        target = np.asarray(y)
        weight = np.ones(len(target)) if sample_weight is None else np.asarray(sample_weight)
        self.classes_ = np.unique(target)
        table: dict[tuple[float, ...], np.ndarray] = {}
        for index in range(len(target)):
            key = self._key(design[index])
            row = table.setdefault(key, np.zeros(len(self.classes_)))
            row[int(np.searchsorted(self.classes_, target[index]))] += weight[index]
        self.table_ = {key: row / row.sum() for key, row in table.items()}
        overall = np.array([np.sum(weight[target == c]) for c in self.classes_])
        self.overall_ = overall / overall.sum()
        return self

    def predict_proba(self, x: Any) -> np.ndarray:
        design = np.asarray(x, dtype=float)
        if design.ndim == 1:
            design = design[:, None]
        return np.array([self.table_.get(self._key(row), self.overall_) for row in design])


class SolveKeyLog:
    """Records every ``AdaptiveIPTW._solve_key`` call of a fit: the key, mechanism and folds."""

    def __init__(self) -> None:
        self.names: list[str] = []
        self.mechanisms: dict[str, np.ndarray] = {}
        self.assignment: np.ndarray | None = None

    def __enter__(self) -> SolveKeyLog:
        from cleverly.estimators import AdaptiveIPTW

        original = AdaptiveIPTW._solve_key
        log = self

        def spy(self: Any, data: Any, nuisance: Any, spec: Any, mechanism: Any, *rest: Any) -> Any:
            log.names.append(spec.name)
            log.mechanisms[spec.name] = np.array(mechanism, dtype=float, copy=True)
            folds = getattr(nuisance, "folds", None)
            if folds is not None and folds.n_folds > 1:
                log.assignment = np.array(folds.assignment, copy=True)
            return original(self, data, nuisance, spec, mechanism, *rest)

        self._patch = (AdaptiveIPTW, original)
        AdaptiveIPTW._solve_key = spy  # type: ignore[method-assign]
        return self

    def __exit__(self, *exc: Any) -> None:
        owner, original = self._patch
        owner._solve_key = original  # type: ignore[method-assign]


def tilt_loop_with_fold_regression(
    indicator: np.ndarray,
    target: np.ndarray,
    g0: np.ndarray,
    assignment: np.ndarray | None,
    *,
    tolerance: float = 1e-12,
    max_rounds: int = 100,
    bounds: tuple[float, float] = (0.0, 1.0),
    clip: tuple[float, float] | None = (0.0, 1.0),
) -> dict[str, Any]:
    """The test-side targeting loop with a linear reduced regression on ``[1, g]``.

    A binary outcome makes the shipped reduced regression a probability, so ``clip`` bounds the
    prediction to ``[0, 1]``. Pass ``None`` for a continuous outcome.

    With ``assignment`` the regression of each fold is fitted on the other folds' rows with
    ``indicator == 1`` and evaluated on the held-out rows, as a cross-fitted fit does. Without
    it, the regression is fitted on all rows. The tilt is pooled over all rows.
    """
    lo, hi = bounds
    g = np.clip(np.asarray(g0, dtype=float), lo, hi)
    mask = indicator > 0.0
    folds = np.zeros(len(g), dtype=int) if assignment is None else np.asarray(assignment)

    def regress(design: np.ndarray) -> np.ndarray:
        out = np.empty(len(design))
        for fold in np.unique(folds):
            held = folds == fold
            train = mask & (~held if assignment is not None else np.ones(len(g), dtype=bool))
            x = np.column_stack([np.ones(train.sum()), design[train]])
            beta, *_ = np.linalg.lstsq(x, target[train], rcond=None)
            out[held] = beta[0] + beta[1] * design[held]
        return out if clip is None else np.clip(out, clip[0], clip[1])

    weight = np.ones(len(g))
    qr = regress(g)
    qr0 = qr
    steps: list[float] = []
    for _ in range(max_rounds):
        h = qr / g
        score = float(np.mean(h * (indicator - g)))
        if abs(score) < tolerance * max(1.0, float(np.mean(np.abs(h)))):
            break
        eps = tilt_root(indicator, g, h, weight, None if bounds == (0.0, 1.0) else bounds)
        steps.append(eps)
        g = np.clip(expit(logit(g) + eps * h), lo, hi)
        qr = regress(g)
    psi = float(np.mean(indicator * target / g))
    return {"g": g, "qr": qr, "qr0": qr0, "eps": steps, "psi": psi}
