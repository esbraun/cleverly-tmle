r"""Laws, truths and learners of the ``adaptive-iptw`` study (X21, ``X21-plan.md`` 3.5).

The study fits :class:`~cleverly.estimators.AdaptiveIPTW`, whose targeted propensity moves along
:math:`H = Q_r / g`, where :math:`Q_r(g) = E(T \mid I = 1, g(W))` is the reduced regression.
This module holds what the fits read: the samplers of the three continuous laws, their exact
truths, and the learners that stand for each cell's mechanism.

=====  ========================================================================================
law    construction
=====  ========================================================================================
P      Benkeser, Carone, van der Laan and Gilbert (2017): ``W1 ~ U(-2, 2)``,
       ``W2 ~ Bern(0.5)``, ``logit g = -W1 + 2 W1 W2``, ``logit Q = 0.2 A - W1 + 2 W1 W2``.
       The reduced regression is exact (hypothesis H1: ``Q_0 = Qr_0``) for both arms
PD     Law P's covariates and mechanism, with ``logit Q(a, W) = -W1 + 2 W1 W2 + a (0.2 + W1)``.
       H1 fails for arm 1 and for the regimes ``d`` and ``s``
VL1    ``W1, W2 ~ N(0, 1)``, ``g_0 = expit(0.6 W1 + 0.3 W2)``,
       ``Q(1, W) = expit(-0.2 - W1 + 2 W2)``.  H1 fails (``L2(Q - Qr) = 0.329``)
VL0    As VL1 with ``Q(1, W) = expit(0.1 + 1.5 (0.6 W1 + 0.3 W2))``.  H1 holds
=====  ========================================================================================

The samplers here draw from the generator of their own seed, so the study does not import
another study's sampler.  The laws M, C, WT, CL, ST and MA are the shared law modules of the
registered studies; :mod:`tests.studies.adaptive_iptw_properties` builds their draws.

**The reduced learner** is :class:`DesignBasisGLM`.  It regresses the key's target on a
polynomial of ``logit`` of the key's mechanism: a logistic regression for an indicator key and
least squares for a stochastic-regime key.  The degree is the plan's basis: 1 for law P and
for the strata, 2 to 5 for the finite-support laws, 3 elsewhere.  On a law whose mechanism
takes fewer than ``degree + 1`` values, the degree stops at the number of values minus one,
which is the saturated regression.  The learner is neither a classifier nor a regressor in
the scikit-learn sense, so the refusal R19 (a classifier on a stochastic regime) does not
read it.

**The mechanism learners** differ from a library classifier in one respect: a ``rate`` learner
moves the fitted logit by a fixed multiple of :math:`n^{-1/3}`, with ``n`` the declared sample
size and not the training size, so the perturbation is the same on every fold.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial.hermite_e import hermegauss
from numpy.polynomial.legendre import leggauss
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# --------------------------------------------------------------------------------- quadrature

#: Gauss-Legendre nodes on ``(-2, 2)`` and ``(0, 2)`` for the continuous covariate of laws P and
#: PD.  The integrands are smooth on each side of 0, where the rule ``1{W1 > 0}`` jumps.
_NODES = 200


def _legendre(low: float, high: float, nodes: int = _NODES) -> tuple[np.ndarray, np.ndarray]:
    x, w = leggauss(nodes)
    return 0.5 * (high - low) * x + 0.5 * (high + low), 0.5 * (high - low) * w


def p_grid() -> pd.DataFrame:
    """Quadrature rows of the covariate law of P and PD: ``W1``, ``W2`` and the weight.

    The weights sum to one.  The grid splits ``(-2, 2)`` at 0 so that a rule on ``W1 > 0`` is
    integrated exactly.
    """
    rows = []
    for low, high in ((-2.0, 0.0), (0.0, 2.0)):
        x, w = _legendre(low, high)
        for w2 in (0.0, 1.0):
            rows.append(pd.DataFrame({"W1": x, "W2": w2, "weight": 0.5 * w / 4.0}))
    return pd.concat(rows, ignore_index=True)


def gauss_hermite_grid(nodes: int = 80) -> pd.DataFrame:
    """Quadrature rows of two independent standard normals: ``W1``, ``W2`` and the weight."""
    x, w = hermegauss(nodes)
    w = w / np.sqrt(2.0 * np.pi)
    w1, w2 = np.meshgrid(x, x, indexing="ij")
    return pd.DataFrame({"W1": w1.ravel(), "W2": w2.ravel(), "weight": np.outer(w, w).ravel()})


# ----------------------------------------------------------------------------------- law P / PD


def p_logit_g(w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """``logit P(A = 1 | W)`` of laws P and PD."""
    return -w1 + 2.0 * w1 * w2


def p_outcome_logit(a: np.ndarray | float, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """``logit Q(a, W)`` of law P: a constant effect 0.2."""
    return 0.2 * np.asarray(a, dtype=float) + p_logit_g(w1, w2)


def pd_outcome_logit(a: np.ndarray | float, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """``logit Q(a, W)`` of law PD: an effect ``0.2 + W1`` that varies with ``W1``."""
    return p_logit_g(w1, w2) + np.asarray(a, dtype=float) * (0.2 + w1)


def sample_p(n: int, seed: int, *, effect_modified: bool = False) -> pd.DataFrame:
    """Draw law P (or law PD with ``effect_modified``): ``Y``, ``A``, ``W1``, ``W2``, ``W12``.

    Parameters
    ----------
    n : int
        Rows.
    seed : int
        Seed of the generator.
    effect_modified : bool
        Draw law PD, whose outcome loads on ``A W1``.

    Returns
    -------
    pandas.DataFrame
        The columns ``Y``, ``A``, ``W1``, ``W2`` and ``W12 = W1 W2``.
    """
    rng = np.random.default_rng(seed)
    w1 = rng.uniform(-2.0, 2.0, size=n)
    w2 = rng.binomial(1, 0.5, size=n).astype(float)
    a = rng.binomial(1, expit(p_logit_g(w1, w2))).astype(float)
    logit_q = pd_outcome_logit(a, w1, w2) if effect_modified else p_outcome_logit(a, w1, w2)
    y = rng.binomial(1, expit(logit_q)).astype(float)
    return pd.DataFrame({"Y": y, "A": a, "W1": w1, "W2": w2, "W12": w1 * w2})


def true_g(frame: pd.DataFrame) -> np.ndarray:
    """``P(A = 1 | W)`` of laws P and PD at each row."""
    return np.asarray(expit(p_logit_g(frame["W1"].to_numpy(), frame["W2"].to_numpy())))


def known_matrix(frame: pd.DataFrame) -> np.ndarray:
    """The ``(n, 2)`` known mechanism ``[1 - g, g]`` of laws P and PD."""
    g1 = true_g(frame)
    return np.column_stack([1.0 - g1, g1])


def _arm_mean(grid: pd.DataFrame, a: float, outcome: Callable[..., np.ndarray]) -> float:
    q = expit(outcome(a, grid["W1"].to_numpy(), grid["W2"].to_numpy()))
    return float(np.sum(grid["weight"].to_numpy() * q))


def truth_p() -> dict[str, float]:
    """The arm means and effect of law P by quadrature."""
    grid = p_grid()
    ey0, ey1 = (_arm_mean(grid, a, p_outcome_logit) for a in (0.0, 1.0))
    return {"ey0": ey0, "ey1": ey1, "ate": ey1 - ey0, "rr": ey1 / ey0, "or": _odds(ey1, ey0)}


def _odds(left: float, right: float) -> float:
    return (left / (1.0 - left)) / (right / (1.0 - right))


#: The regimes of law PD: never treat, treat when ``W1 > 0``, and treat with ``expit(W1)``.
RULE_NAME = "d"
STOCHASTIC_NAME = "s"
NEVER_NAME = "never"


def rule_arm(frame: pd.DataFrame) -> np.ndarray:
    """The deterministic rule ``d(W) = 1{W1 > 0}`` of law PD."""
    return (frame["W1"].to_numpy() > 0.0).astype(int)


def stochastic_one(frame: pd.DataFrame) -> np.ndarray:
    """The stochastic regime ``s(1 | W) = expit(W1)`` of law PD."""
    return np.asarray(expit(frame["W1"].to_numpy()))


def truth_pd() -> dict[str, float]:
    """The arm and regime means of law PD by quadrature.

    ``ey_regime[never]`` is the arm-0 mean, ``ey_regime[d]`` the mean under the rule and
    ``ey_regime[s]`` the mean under the stochastic regime.
    """
    grid = p_grid()
    w1, w2 = grid["W1"].to_numpy(), grid["W2"].to_numpy()
    weight = grid["weight"].to_numpy()
    q0 = expit(pd_outcome_logit(0.0, w1, w2))
    q1 = expit(pd_outcome_logit(1.0, w1, w2))
    rule = (w1 > 0.0).astype(float)
    tilt = expit(w1)
    ey0, ey1 = float(np.sum(weight * q0)), float(np.sum(weight * q1))
    on_rule = float(np.sum(weight * (rule * q1 + (1.0 - rule) * q0)))
    stochastic = float(np.sum(weight * (tilt * q1 + (1.0 - tilt) * q0)))
    return {
        "ey0": ey0,
        "ey1": ey1,
        "ate": ey1 - ey0,
        "rr": ey1 / ey0,
        "or": _odds(ey1, ey0),
        f"ey_regime[{NEVER_NAME}]": ey0,
        f"ey_regime[{RULE_NAME}]": on_rule,
        f"ey_regime[{STOCHASTIC_NAME}]": stochastic,
        f"ate_regime[{RULE_NAME} vs {NEVER_NAME}]": on_rule - ey0,
        f"ate_regime[{STOCHASTIC_NAME} vs {NEVER_NAME}]": stochastic - ey0,
    }


# --------------------------------------------------------------------------------- laws VL1, VL0


VL_LAWS = ("vl1", "vl0")


def vl_g(w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """``P(A = 1 | W)`` of the validity laws."""
    return np.asarray(expit(0.6 * w1 + 0.3 * w2))


def vl_q1(law: str, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """``Q(1, W)`` of the validity law ``law``."""
    if law == "vl1":
        return np.asarray(expit(-0.2 - 1.0 * w1 + 2.0 * w2))
    if law == "vl0":
        return np.asarray(expit(0.1 + 1.5 * (0.6 * w1 + 0.3 * w2)))
    raise KeyError(law)


def sample_vl(law: str, n: int, seed: int) -> pd.DataFrame:
    """Draw a validity law: ``Y``, ``A``, ``W1`` and ``W2``.  ``Y | A = 0`` is ``Bern(0.5)``."""
    rng = np.random.default_rng(seed)
    w1, w2 = rng.standard_normal(n), rng.standard_normal(n)
    a = rng.binomial(1, vl_g(w1, w2)).astype(float)
    q = np.where(a == 1.0, vl_q1(law, w1, w2), 0.5)
    y = rng.binomial(1, q).astype(float)
    return pd.DataFrame({"Y": y, "A": a, "W1": w1, "W2": w2})


def truth_vl(law: str) -> dict[str, float]:
    """``E Y(1)`` of a validity law by 80 x 80 Gauss-Hermite quadrature."""
    grid = gauss_hermite_grid()
    q = vl_q1(law, grid["W1"].to_numpy(), grid["W2"].to_numpy())
    return {"ey1": float(np.sum(grid["weight"].to_numpy() * q))}


# ------------------------------------------------------------------------------------- learners


def _clip(p: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), floor, 1.0 - floor)


class DesignBasisGLM(BaseEstimator):
    """The reduced regression of the study: a polynomial of ``logit`` of the mechanism.

    The first design column is the key's mechanism :math:`g`.  The learner standardizes
    ``x = logit(g)`` on the training rows, expands it to a polynomial of degree
    ``min(degree, distinct - 1)`` and fits an unpenalized logistic regression when the target
    holds only 0 and 1, and least squares otherwise.  With the cap, a mechanism that takes
    ``m`` values gets the saturated regression on those ``m`` values.

    Parameters
    ----------
    degree : int
        The polynomial degree of the basis.
    """

    def __init__(self, degree: int = 1) -> None:
        self.degree = degree

    @staticmethod
    def _x(design: Any) -> np.ndarray:
        return np.asarray(logit(_clip(np.asarray(design, dtype=float)[:, 0])), dtype=float)

    def _basis(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.center_) / self.scale_
        return np.column_stack([z**k for k in range(1, self.degree_ + 1)])

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> DesignBasisGLM:
        x = self._x(X)
        target = np.asarray(y, dtype=float)
        distinct = len(np.unique(np.round(x, 10)))
        self.degree_ = max(1, min(int(self.degree), distinct - 1))
        self.center_ = float(np.mean(x))
        spread = float(np.std(x))
        self.scale_ = spread if spread > 0.0 else 1.0
        basis = self._basis(x)
        weight = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        self.binary_ = bool(np.all((target == 0.0) | (target == 1.0)))
        if self.binary_ and 0.0 < target.mean() < 1.0:
            self.model_: Any = LogisticRegression(
                C=np.inf, solver="newton-cholesky", tol=1e-12, max_iter=200
            )
            self.model_.fit(basis, target, sample_weight=weight)
        else:
            design = np.column_stack([np.ones(len(x)), basis])
            root = np.ones(len(x)) if weight is None else np.sqrt(weight)
            self.beta_, *_ = np.linalg.lstsq(design * root[:, None], target * root, rcond=None)
            self.model_ = None
            self.constant_ = float(target.mean()) if self.binary_ else None
        return self

    def predict(self, X: Any) -> np.ndarray:
        basis = self._basis(self._x(X))
        if self.model_ is not None:
            return np.asarray(self.model_.predict_proba(basis)[:, 1], dtype=float)
        if self.constant_ is not None:
            return np.full(len(basis), self.constant_)
        return np.asarray(np.column_stack([np.ones(len(basis)), basis]) @ self.beta_, dtype=float)

    def predict_proba(self, X: Any) -> np.ndarray:
        p = np.clip(self.predict(X), 0.0, 1.0)
        return np.column_stack([1.0 - p, p])


#: The columns of the study frames the mechanism learners read.
P_COLUMNS = ("W1", "W2", "W12")


class RateLogistic(BaseEstimator, ClassifierMixin):
    """An unpenalized logistic mechanism moved by a perturbation of order ``n^(-1/3)``.

    The fit is an unpenalized logistic regression on ``columns`` of the design.  The returned
    logit is ``scale * logit(g_hat) + shift * (1 + W2)``, where ``shift = 2 n^(-1/3)`` and ``n``
    is the declared sample size.  ``scale = 1`` and ``shift = 0`` is the correct fit.

    Parameters
    ----------
    columns : tuple of int or None
        The design columns the regression reads, or all of them.
    n : int
        The declared sample size, in ``shift = coefficient n^(-1/3)``.
    shift : float
        The multiple of ``n^(-1/3)`` added to the logit for ``(1 + W2)`` (design column 1).
    scale : float
        The multiple of ``n^(-1/3)`` added to one in front of the fitted logit.
    """

    def __init__(
        self,
        columns: tuple[int, ...] | None = None,
        n: int = 1,
        shift: float = 0.0,
        scale: float = 0.0,
    ) -> None:
        self.columns = columns
        self.n = n
        self.shift = shift
        self.scale = scale

    def _design(self, X: Any) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        return design if self.columns is None else design[:, list(self.columns)]

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> RateLogistic:
        self.model_ = LogisticRegression(
            C=np.inf, solver="newton-cholesky", tol=1e-12, max_iter=200
        )
        self.model_.fit(self._design(X), np.asarray(y).reshape(-1), sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        eta = self.model_.decision_function(self._design(X))
        step = float(self.n) ** (-1.0 / 3.0)
        eta = (1.0 + self.scale * step) * eta + self.shift * step * (1.0 + design[:, 1])
        p = np.asarray(expit(eta))
        return np.column_stack([1.0 - p, p])


def p_mechanism(kind: str, n: int) -> Any:
    """The treatment learner of law P or PD for one of the mechanism kinds.

    Parameters
    ----------
    kind : {"correct", "rate", "wrong", "flexible", "rate_scaled"}
        ``correct`` is the logistic fit on ``W1, W2, W1 W2``.  ``rate`` adds
        ``2 n^(-1/3) (1 + W2)`` to the logit.  ``rate_scaled`` multiplies the logit by
        ``1 + 2 n^(-1/3)``.  ``wrong`` is the main-terms fit.  ``flexible`` is gradient boosting.
    n : int
        The declared sample size.
    """
    if kind == "correct":
        return RateLogistic()
    if kind == "rate":
        return RateLogistic(n=n, shift=2.0)
    if kind == "rate_scaled":
        return RateLogistic(n=n, scale=2.0)
    if kind == "wrong":
        return RateLogistic(columns=(0, 1))
    if kind == "flexible":
        return HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.1,
            max_leaf_nodes=8,
            min_samples_leaf=40,
            early_stopping=False,
            l2_regularization=0.0,
            random_state=0,
        )
    raise KeyError(kind)


class ValidityLogistic(BaseEstimator, ClassifierMixin):
    """The validity laws' mechanism: logistic on ``(1, W1, W2)``, optionally level-set breaking.

    ``kind="lsb"`` adds ``n^(-1/3) W2`` to the fitted logit, the direction that breaks the level
    sets of the mechanism for the outcome regression of VL1.

    Parameters
    ----------
    kind : {"glm", "lsb"}
        The correct fit, or the fit with the level-set-breaking shift.
    n : int
        The declared sample size.
    """

    def __init__(self, kind: str = "glm", n: int = 1) -> None:
        self.kind = kind
        self.n = n

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> ValidityLogistic:
        self.model_ = LogisticRegression(
            C=np.inf, solver="newton-cholesky", tol=1e-12, max_iter=200
        )
        self.model_.fit(
            np.asarray(X, dtype=float), np.asarray(y).reshape(-1), sample_weight=sample_weight
        )
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        eta = self.model_.decision_function(design)
        if self.kind == "lsb":
            eta = eta + float(self.n) ** (-1.0 / 3.0) * design[:, 1]
        p = np.asarray(expit(eta))
        return np.column_stack([1.0 - p, p])


class CellFrequency(BaseEstimator, ClassifierMixin):
    """A binary classifier fitted as the weighted class frequency in each distinct design row.

    This is the maximum-likelihood fit of a logistic regression saturated on the cells of the
    design, so it is the "saturated logistic" of the finite-support laws.  A design row never
    seen in training gets the overall frequency.  Optionally the fitted logit moves by
    ``size * n^(-1/3) * shift(design)``, which is the rate perturbation of a cell; ``n`` is the
    declared sample size.

    Parameters
    ----------
    shift : callable or None
        ``shift(design)`` returns the per-row multiple of ``size * n^(-1/3)``; ``None`` is no
        perturbation.
    size : float
        The size of the perturbation.
    n : int
        The declared sample size.
    digits : int
        The rounding of a design value to a cell key.
    """

    def __init__(
        self,
        shift: Callable[[np.ndarray], np.ndarray] | None = None,
        size: float = 2.0,
        n: int = 1,
        digits: int = 9,
    ) -> None:
        self.shift = shift
        self.size = size
        self.n = n
        self.digits = digits

    def _keys(self, design: np.ndarray) -> list[tuple[float, ...]]:
        return [tuple(row) for row in np.round(design, self.digits)]

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> CellFrequency:
        design = np.asarray(X, dtype=float)
        if design.ndim == 1:
            design = design[:, None]
        target = np.asarray(y).reshape(-1)
        self.classes_ = np.unique(target)
        if len(self.classes_) != 2:
            raise ValueError("CellFrequency fits a binary target with both classes present")
        weight = np.ones(len(target)) if sample_weight is None else np.asarray(sample_weight)
        positive = (target == self.classes_[1]).astype(float)
        total: dict[tuple[float, ...], float] = {}
        ones: dict[tuple[float, ...], float] = {}
        for key, w, p in zip(self._keys(design), weight, positive, strict=True):
            total[key] = total.get(key, 0.0) + float(w)
            ones[key] = ones.get(key, 0.0) + float(w * p)
        self.table_ = {key: ones[key] / total[key] for key in total}
        self.overall_ = float(np.sum(weight * positive) / np.sum(weight))
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        if design.ndim == 1:
            design = design[:, None]
        p = np.array([self.table_.get(key, self.overall_) for key in self._keys(design)])
        if self.shift is not None:
            step = float(self.size) * float(self.n) ** (-1.0 / 3.0)
            eta = logit(np.clip(p, 1e-6, 1.0 - 1e-6)) + step * np.asarray(self.shift(design))
            p = np.asarray(expit(eta))
        return np.column_stack([1.0 - p, p])


class ShiftedMultinomial(BaseEstimator, ClassifierMixin):
    """The multinomial logistic mechanism of law M, optionally moved by ``n^(-1/3)``.

    The fit is ``LogisticRegression(C=1e6)``.  The returned logit of every class but the first
    moves by ``size * n^(-1/3) * (1 + W_1)``, with ``W_1`` the first design column.

    Parameters
    ----------
    size : float
        The size of the perturbation: 0 is the correct fit, 2 is the rate fit.
    n : int
        The declared sample size.
    """

    def __init__(self, size: float = 0.0, n: int = 1) -> None:
        self.size = size
        self.n = n

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> ShiftedMultinomial:
        self.model_ = LogisticRegression(C=1e6, max_iter=5000)
        self.model_.fit(np.asarray(X, dtype=float), np.asarray(y), sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        if self.size == 0.0:
            return np.asarray(self.model_.predict_proba(design))
        eta = np.asarray(self.model_.decision_function(design), dtype=float)
        eta[:, 1:] += self.size * float(self.n) ** (-1.0 / 3.0) * (1.0 + design[:, [0]])
        eta -= eta.max(axis=1, keepdims=True)
        weights = np.exp(eta)
        return np.asarray(weights / weights.sum(axis=1, keepdims=True))


__all__ = [
    "CellFrequency",
    "DesignBasisGLM",
    "RateLogistic",
    "ShiftedMultinomial",
    "ValidityLogistic",
    "gauss_hermite_grid",
    "known_matrix",
    "p_grid",
    "p_mechanism",
    "sample_p",
    "sample_vl",
    "truth_p",
    "truth_pd",
    "truth_vl",
]
