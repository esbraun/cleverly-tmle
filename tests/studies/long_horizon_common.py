"""Shared pieces of the X39 long-horizon studies: learners, mechanisms, fits and rows.

The three studies (``long-horizon-ltmle``, ``long-horizon-categorical-ltmle`` and
``long-horizon-sdr``) fit ``LTMLE`` and ``SDR`` over three to five decisions on the law family of
:mod:`tests.discrete_law_longitudinal_horizon`.  This module holds what they share:

* the outcome learners: the declared misspecified quasibinomial GLM, the saturated
  :class:`CellMeans` of the exact witnesses, the ladder rungs of the necessity controls
  (:class:`ColumnSubsetGLM` for the W-blind and L-blind GLMs, :class:`InterceptOnly`,
  :class:`FixedOffsetLearner`), the correctly specified :class:`OracleOffsetGLM`, and
  :class:`NodePattern`, which routes each node's fit to its own learner;
* the known treatment learner :class:`KnownHorizonMechanism` and its two-arm mutation
  :class:`HorizonComplementMechanism`;
* the declared wrong mechanisms as probability columns (:func:`wrong_mechanism`);
* :func:`fit`, one ``LTMLE`` or ``SDR`` call on a frame of the law, and
  :func:`untargeted_plugin`, the unfluctuated sequential regression.

A learner reads the node ``t`` from its design width.  A label or rule plan's outcome design at
node ``t`` is ``[W, L_2, ..., L_t]`` (``LongitudinalData.covariate_history``); a policy plan adds
one drop-first arm-indicator block per policy node up to ``t`` (``sequential.outcome_design``).
The treatment design at node ``t`` is ``[W, L_2, ..., L_t]`` and one drop-first block per earlier
arm (``LongitudinalData.history_design``).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LinearRegression

from cleverly.longitudinal import LTMLE, SDR
from tests import discrete_law_longitudinal_horizon as law_
from tests import longitudinal_horizon_plans as plans_
from tests.discrete_law_longitudinal_horizon import Law
from tests.studies.fractional_glm import QuasiBinomialGLM

#: The bound every gated cell declares: it binds on no path of any study law.
G_BOUNDS = (1e-8, 1.0)


# ----------------------------------------------------------------------------- design decoding
def outcome_node(width: int, m: int, policy_nodes: Iterable[int], K: int) -> int:
    """The node ``t`` whose outcome design has ``width`` columns.

    ``width = t + (m - 1) |{policy nodes <= t}|`` is strictly increasing in ``t``, so the node
    is unique.  A width that matches no node raises.
    """
    nodes = set(policy_nodes)
    for t in range(1, K + 1):
        if t + (m - 1) * sum(1 for s in nodes if s <= t) == width:
            return t
    raise ValueError(f"no node of a {K}-node design has outcome width {width}")


def treatment_node(width: int, m: int) -> int:
    """The node ``t`` whose treatment design has ``width = 1 + m (t - 1)`` columns."""
    t, remainder = divmod(width - 1, m)
    if remainder or t < 0:
        raise ValueError(f"treatment design width {width} matches no node at m = {m}")
    return t + 1


def _decode_arm(block: np.ndarray, m: int) -> np.ndarray:
    """The arm code of a drop-first indicator block."""
    code = np.zeros(block.shape[0], dtype=int)
    for k in range(1, m):
        code = np.where(block[:, k - 1] == 1.0, k, code)
    return code


# ----------------------------------------------------------------------------- outcome learners
class _Probabilistic(BaseEstimator):
    """``predict_proba`` and ``classes_`` from a learner's ``predict``."""

    classes_ = np.array([0.0, 1.0])

    def predict(self, X: Any) -> np.ndarray:  # pragma: no cover - overridden
        raise NotImplementedError

    def predict_proba(self, X: Any) -> np.ndarray:
        p = np.asarray(self.predict(X), dtype=float)
        return np.column_stack([1.0 - p, p])


class CellMeans(_Probabilistic):
    """The saturated regression: the weighted mean of the target in each distinct design row.

    Exact on the weighted frame of a finite law, where every support point is a row.  A design
    row unseen in training predicts the overall weighted mean.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> CellMeans:
        matrix = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float).reshape(-1)
        weight = np.ones_like(target) if sample_weight is None else np.asarray(sample_weight)
        cells, inverse = np.unique(matrix, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        total = np.bincount(inverse, weights=weight, minlength=len(cells))
        hits = np.bincount(inverse, weights=weight * target, minlength=len(cells))
        self.cells_ = {
            tuple(row): hit / mass for row, hit, mass in zip(cells, hits, total, strict=False)
        }
        self.mean_ = float(np.sum(weight * target) / np.sum(weight))
        return self

    def predict(self, X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        return np.array([self.cells_.get(tuple(row), self.mean_) for row in matrix])


class InterceptOnly(_Probabilistic):
    """The weighted mean of the target, whatever the design: rung O3 and D2."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> InterceptOnly:
        target = np.asarray(y, dtype=float).reshape(-1)
        weight = np.ones_like(target) if sample_weight is None else np.asarray(sample_weight)
        self.mean_ = float(np.sum(weight * target) / np.sum(weight))
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.full(np.asarray(X).shape[0], self.mean_)


class ColumnSubsetGLM(_Probabilistic):
    """The declared quasibinomial GLM on a subset of the design: rungs O2 (W-blind) and O4.

    ``drop`` is ``"W"`` (the W-blind GLM: every column but the first) or ``"L"`` (the L-blind
    GLM: the baseline ``W`` and the arm blocks, without ``L_2..L_t``).  The L-blind GLM reads
    the node from the width, so it takes the plan's policy nodes.
    """

    def __init__(
        self, drop: str = "W", m: int = 2, K: int = 5, policy_nodes: tuple[int, ...] = ()
    ) -> None:
        self.drop = drop
        self.m = m
        self.K = K
        self.policy_nodes = policy_nodes

    def _columns(self, width: int) -> list[int]:
        if self.drop == "W":
            return list(range(1, width))
        if self.drop == "L":
            t = outcome_node(width, self.m, self.policy_nodes, self.K)
            return [0, *range(t, width)]
        raise ValueError(f"drop must be 'W' or 'L'; got {self.drop!r}")

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> ColumnSubsetGLM:
        matrix = np.asarray(X, dtype=float)
        self.columns_ = self._columns(matrix.shape[1])
        self.glm_ = QuasiBinomialGLM().fit(matrix[:, self.columns_], y, sample_weight=sample_weight)
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.glm_.predict(np.asarray(X, dtype=float)[:, self.columns_]))


class FixedOffsetLearner(_Probabilistic):
    """Rung D3: the declared GLM with the fixed offset ``0.8 z^2``, ``z = (W - 1.5) / 1.5``.

    The offset is outside the design's span, so the fluctuation does not vanish by construction
    at node 1, where the design is ``[W]``.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> FixedOffsetLearner:
        matrix = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float).reshape(-1)
        weight = np.ones_like(target) if sample_weight is None else np.asarray(sample_weight)
        self.coef_ = _offset_irls(matrix, target, weight, self._offset(matrix))
        return self

    @staticmethod
    def _offset(matrix: np.ndarray) -> np.ndarray:
        z = (matrix[:, 0] - 1.5) / 1.5
        return 0.8 * z**2

    def predict(self, X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        design = np.column_stack([np.ones(len(matrix)), matrix])
        return np.asarray(expit(self._offset(matrix) + design @ self.coef_))


def _offset_irls(
    matrix: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    offset: np.ndarray,
    *,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> np.ndarray:
    """The quasibinomial IRLS of ``target`` on ``[1, matrix]`` with a fixed logit offset."""
    design = np.column_stack([np.ones(len(matrix)), matrix])
    coef = np.zeros(design.shape[1])
    for _ in range(max_iter):
        fitted = expit(offset + design @ coef)
        variance = np.clip(fitted * (1.0 - fitted), 1e-12, None)
        working = design @ coef + (target - fitted) / variance
        root = np.sqrt(weight * variance)
        updated = np.linalg.lstsq(design * root[:, None], working * root, rcond=None)[0]
        if np.max(np.abs(updated - coef)) <= tol:
            return np.asarray(updated)
        coef = updated
    return np.asarray(coef)


class OracleOffsetGLM(_Probabilistic):
    """A fitted, correctly specified node regression: the oracle regression as an offset.

    At node ``t`` the learner computes ``o = logit Q^A_{0,t}(a, H_t)`` per row from the oracle
    and fits ``logit E[y] = o + b0 + X b`` by quasibinomial IRLS (``link="logit"``), or, on the
    ``SDR`` transform before the horizon, ``y - Q^A_{0,t}`` on ``X`` by weighted least squares
    (``link="identity"``).  The offset is a function of one plan, so a fit takes one plan per call.
    The coefficients converge to zero, and the fit goes through ``_fit_node_regression`` with the
    node's own rows, so a wrong follower mask or wrong pseudo-outcome rows bias it.
    """

    def __init__(
        self,
        plan: str = "low",
        K: int = 5,
        m: int = 2,
        law: Law = law_.L_R,
        link: str = "logit",
    ) -> None:
        self.plan = plan
        self.K = K
        self.m = m
        self.law = law
        self.link = link

    def _oracle(self, matrix: np.ndarray) -> np.ndarray:
        K, m = self.K, self.m
        policy = law_.policy_nodes(self.plan, K)
        t = outcome_node(matrix.shape[1], m, policy, K)
        w = matrix[:, 0]
        covs = {s: matrix[:, s - 1] for s in range(2, t + 1)}
        arms: dict[int, np.ndarray] = {}
        column = t
        tree_arms = _plan_arms(self.plan, K, m, w, covs)
        for s in range(1, t + 1):
            if s in policy:
                arms[s] = _decode_arm(matrix[:, column : column + m - 1], m)
                column += m - 1
            else:
                arms[s] = tree_arms[s]
        regression = law_.oracle_regression(K, m, self.plan, self.law, t)
        return regression(w, covs, {s: arms[s] for s in range(1, t)}, arms[t])

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> OracleOffsetGLM:
        matrix = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float).reshape(-1)
        weight = np.ones_like(target) if sample_weight is None else np.asarray(sample_weight)
        oracle = self._oracle(matrix)
        if self.link == "identity":
            model = LinearRegression().fit(matrix, target - oracle, sample_weight=weight)
            self.coef_ = np.concatenate([[model.intercept_], model.coef_])
        else:
            offset = logit(np.clip(oracle, 1e-12, 1.0 - 1e-12))
            self.coef_ = _offset_irls(matrix, target, weight, offset)
        return self

    def predict(self, X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        oracle = self._oracle(matrix)
        design = np.column_stack([np.ones(len(matrix)), matrix])
        if self.link == "identity":
            return np.asarray(oracle + design @ self.coef_)
        offset = logit(np.clip(oracle, 1e-12, 1.0 - 1e-12))
        return np.asarray(expit(offset + design @ self.coef_))


def _plan_arms(
    plan: str, K: int, m: int, w: np.ndarray, covs: Mapping[int, np.ndarray]
) -> dict[int, np.ndarray]:
    """The arm a label or rule plan assigns at each node, from ``W`` and ``L`` columns.

    Policy nodes are absent from the result; their arms are read from the design blocks.
    """
    top = m - 1
    n = len(w)
    out: dict[int, np.ndarray] = {}
    for s in range(1, K + 1):
        if s in law_.policy_nodes(plan, K):
            continue
        l_s = covs.get(s, np.zeros(n))
        if plan in ("low", "never") or plan in ("mixlast", "coinlast"):
            out[s] = np.zeros(n, dtype=int)
        elif plan in ("high", "always") or plan == "taper":
            out[s] = np.full(n, top)
        elif plan == "switch":
            out[s] = np.full(n, top if s < K else 0)
        elif plan.startswith("dur"):
            out[s] = np.full(n, int(s <= int(plan[3:])))
        elif plan in ("respond", "reversed"):
            on = w >= 2 if s == 1 else np.asarray(l_s) == 1.0
            out[s] = np.where(on != (plan == "reversed"), top, 0)
        elif plan == "lagged":
            on = w >= 2 if s <= 2 else np.asarray(covs.get(s - 1, np.zeros(n))) == 1.0
            out[s] = np.where(on, top, 0)
        else:
            raise ValueError(f"no rule for plan {plan!r}")
    return out


class NodePattern(_Probabilistic):
    """One outcome learner per node, chosen by the node read from the design width.

    ``learners`` maps a node to its learner; a node it omits uses ``default``.
    """

    def __init__(
        self,
        learners: Mapping[int, Any] | None = None,
        default: Any = None,
        m: int = 2,
        K: int = 5,
        policy_nodes: tuple[int, ...] = (),
    ) -> None:
        self.learners = learners
        self.default = default
        self.m = m
        self.K = K
        self.policy_nodes = policy_nodes

    def _learner(self, width: int) -> Any:
        t = outcome_node(width, self.m, self.policy_nodes, self.K)
        chosen = dict(self.learners or {}).get(t, self.default)
        return QuasiBinomialGLM() if chosen is None else chosen

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> NodePattern:
        matrix = np.asarray(X, dtype=float)
        self.fitted_ = clone(self._learner(matrix.shape[1])).fit(
            matrix, y, sample_weight=sample_weight
        )
        return self

    def predict(self, X: Any) -> np.ndarray:
        fitted = self.fitted_
        if hasattr(fitted, "predict_proba") and not isinstance(fitted, LinearRegression):
            return np.asarray(fitted.predict_proba(np.asarray(X, dtype=float)))[:, 1]
        return np.asarray(fitted.predict(np.asarray(X, dtype=float)))


# ----------------------------------------------------------------------------- mechanisms
class KnownHorizonMechanism(BaseEstimator):
    """The law's treatment probabilities, as a treatment learner on the learner route.

    The node is read from the treatment design width ``1 + m (t - 1)``; a width it does not
    recognise raises.  The earlier arm is decoded from the last indicator block, and the law's
    arm probabilities are returned in the sorted label order ``0, ..., m - 1``.
    """

    def __init__(self, K: int = 5, m: int = 3, law: Law = law_.L_R) -> None:
        self.K = K
        self.m = m
        self.law = law

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> KnownHorizonMechanism:
        self.classes_ = np.arange(self.m, dtype=float)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        m = self.m
        t = treatment_node(matrix.shape[1], m)
        if t > self.K:
            raise ValueError(f"treatment design width {matrix.shape[1]} is past node {self.K}")
        w = matrix[:, 0]
        l_t = matrix[:, t - 1] if t > 1 else np.zeros(len(matrix))
        previous = _decode_arm(matrix[:, -(m - 1) :], m) if t > 1 else np.zeros(len(matrix), int)
        return law_.treatment_probabilities(t, m, self.law, w, l_t, previous)

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.predict_proba(X))[:, -1]


class HorizonComplementMechanism(KnownHorizonMechanism):
    """:class:`KnownHorizonMechanism` with the middle arm read as one minus the top arm.

    At each node in ``nodes`` the middle column becomes ``1 - P(A_t = m - 1)``: the two-arm
    shortcut of a three-level node.  The rows then no longer sum to one.
    """

    def __init__(
        self, K: int = 5, m: int = 3, law: Law = law_.L_R, nodes: tuple[int, ...] = ()
    ) -> None:
        super().__init__(K=K, m=m, law=law)
        self.nodes = nodes

    def predict_proba(self, X: Any) -> np.ndarray:
        out = np.array(super().predict_proba(X))
        t = treatment_node(np.asarray(X).shape[1], self.m)
        if t in set(self.nodes):
            out[:, 1] = 1.0 - out[:, self.m - 1]
        return out


def wrong_mechanism(frame: pd.DataFrame, K: int, m: int, kind: str) -> pd.DataFrame:
    """``frame`` with its ``g{t}_{k}`` and ``pc{t}`` columns replaced by a declared wrong mechanism.

    ``kind``: ``"constant"`` (per-node marginal arm and retention probabilities), ``"wblind"``
    (treatment given ``(L_t, A_{t-1})`` and retention given ``(L_t, A_{t-1}, A_t)``, the law's
    projections with ``W`` marginalized), ``"wblind_retention_one"`` (W-blind treatment, retention
    one), ``"confounder_blind"`` (treatment given ``A_{t-1}`` only, retention one),
    ``"retention_one"`` (the law's treatment, retention one).  Probabilities are mass-weighted on
    a weighted frame (column ``w``) and frequencies otherwise.
    """
    out = frame.copy()
    weight = np.asarray(frame["w"], dtype=float) if "w" in frame else np.ones(len(frame))
    for t in range(1, K + 1):
        observed = frame[f"A{t}"].notna().to_numpy()
        arm = np.nan_to_num(frame[f"A{t}"].to_numpy(float)).astype(int)
        l_t = np.nan_to_num(frame[f"L{t}"].to_numpy(float)) if t > 1 else np.zeros(len(frame))
        prev = (
            np.nan_to_num(frame[f"A{t - 1}"].to_numpy(float)).astype(int)
            if t > 1
            else np.zeros(len(frame), int)
        )
        stays = np.nan_to_num(frame[f"C{t}"].to_numpy(float))
        if kind == "retention_one":
            out.loc[observed, f"pc{t}"] = 1.0
            continue
        if kind == "constant":
            key = np.zeros(len(frame), int)
        elif kind in ("wblind", "wblind_retention_one"):
            key = l_t.astype(int) * 10 + prev
        elif kind == "confounder_blind":
            key = prev
        else:
            raise ValueError(f"unknown wrong mechanism {kind!r}")
        for value in np.unique(key[observed]):
            cell = observed & (key == value)
            mass = weight[cell].sum()
            for k in range(m):
                out.loc[cell, f"g{t}_{k}"] = weight[cell & (arm == k)].sum() / mass
        if kind in ("wblind_retention_one", "confounder_blind"):
            out.loc[observed, f"pc{t}"] = 1.0
            continue
        rkey = key * 10 + arm if kind == "wblind" else key
        for value in np.unique(rkey[observed]):
            cell = observed & (rkey == value)
            out.loc[cell, f"pc{t}"] = np.sum(weight[cell] * stays[cell]) / weight[cell].sum()
    return out


# ----------------------------------------------------------------------------- fits
def fit(
    frame: pd.DataFrame,
    K: int,
    m: int,
    plans: Sequence[str],
    *,
    estimator: str = "LTMLE",
    outcome_learner: Any = None,
    pseudo_learner: Any = None,
    treatment_learner: Any = None,
    censoring_learner: Any = None,
    probabilities: bool | str = True,
    n_folds: int = 1,
    g_bounds: tuple[float, float] = G_BOUNDS,
    weights: str | None = None,
    random_state: int = 0,
    regimens: Mapping[str, Any] | None = None,
    **extra: Any,
) -> Any:
    """One ``LTMLE`` or ``SDR`` call; the first plan is the reference.

    ``probabilities=True`` passes the law's columns for treatment and retention;
    ``"censoring"`` passes the retention columns only (the treatment then comes from
    ``treatment_learner``); ``False`` passes none.
    """
    cls = {"LTMLE": LTMLE, "SDR": SDR}[estimator]
    outcome = QuasiBinomialGLM() if outcome_learner is None else outcome_learner
    if pseudo_learner is None:
        pseudo_learner = clone(outcome) if estimator == "LTMLE" else LinearRegression()
    columns = law_.probability_columns(K, m)
    route: dict[str, Any] = {}
    if probabilities is True:
        route = columns
    elif probabilities == "censoring":
        route = {"censoring_probabilities": columns["censoring_probabilities"]}
    learners = {
        name: value
        for name, value in (
            ("treatment_learner", treatment_learner),
            ("censoring_learner", censoring_learner),
        )
        if value is not None
    }
    specs = dict(regimens) if regimens is not None else plans_.regimens(plans, K, m)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # A working model reports coefficients, so it takes no reference.
        if "msm" not in extra:
            extra["reference"] = next(iter(specs))
        model = cls(
            specs,
            outcome_learner=outcome,
            pseudo_learner=pseudo_learner,
            n_folds=n_folds,
            g_bounds=g_bounds,
            simultaneous=False,
            random_state=random_state,
            **learners,
            **extra,
        )
        return model.fit(frame, weights=weights, **law_.roles(K, m), **route)


def per_unit_curve(result: Any, name: str, weight: np.ndarray) -> np.ndarray:
    """A weighted fit's reported curve per unit: the curve divided by ``n w_i / sum w``."""
    curve = np.asarray(result[name].influence_curve, dtype=float)
    scale = len(weight) * weight / np.sum(weight)
    return curve / scale


# ----------------------------------------------------------------------------- the plug-in
def _design(frame: pd.DataFrame, t: int, m: int, policy: Iterable[int], arm: Any = None) -> Any:
    n = len(frame)
    columns = [np.nan_to_num(frame["W"].to_numpy(float))]
    columns += [np.nan_to_num(frame[f"L{s}"].to_numpy(float)) for s in range(2, t + 1)]
    for s in sorted(set(policy)):
        if s > t:
            continue
        codes = (
            np.full(n, arm)
            if s == t and arm is not None
            else np.nan_to_num(frame[f"A{s}"].to_numpy(float))
        )
        columns += [(codes == k).astype(float) for k in range(1, m)]
    return np.column_stack(columns)


def plan_density(frame: pd.DataFrame, plan: str, K: int, m: int, t: int) -> np.ndarray:
    """``(n, m)`` intervention density of ``plan`` at node ``t`` on the rows of ``frame``."""
    spec = plans_.regimen(plan, K, m)
    n = len(frame)
    if isinstance(spec, int):
        node: Any = spec
    else:
        node = spec.plan[t - 1]
    if isinstance(node, (int, np.integer, float)):
        out = np.zeros((n, m))
        out[:, int(node)] = 1.0
        return out
    if hasattr(node, "density_fn"):
        history = frame.copy()
        return np.asarray(node.density_fn(history), dtype=float)
    arms = np.asarray(node(frame), dtype=int)
    out = np.zeros((n, m))
    out[np.arange(n), arms] = 1.0
    return out


def untargeted_plugin(
    frame: pd.DataFrame,
    plan: str,
    K: int,
    m: int,
    learner: Any = None,
    weights: np.ndarray | None = None,
) -> float:
    """The unfluctuated sequential regression of ``plan``: every node fit, no update."""
    n = len(frame)
    weight = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    learner = QuasiBinomialGLM() if learner is None else learner
    policy = law_.policy_nodes(plan, K)
    follow = np.ones(n, dtype=bool)
    risk: dict[int, np.ndarray] = {}
    fol: dict[int, np.ndarray] = {}
    density: dict[int, np.ndarray] = {}
    for t in range(1, K + 1):
        observed = frame[f"A{t}"].notna().to_numpy() & follow
        risk[t] = observed
        density[t] = plan_density(frame, plan, K, m, t)
        arm = np.nan_to_num(frame[f"A{t}"].to_numpy(float)).astype(int)
        positive = density[t][np.arange(n), arm] > 0
        fol[t] = observed & positive & (np.nan_to_num(frame[f"C{t}"].to_numpy(float)) == 1.0)
        follow = fol[t]
    target = np.nan_to_num(frame["Y"].to_numpy(float))
    for t in range(K, 0, -1):
        rows = fol[t]
        X = _design(frame, t, m, policy)
        model = clone(learner).fit(X[rows], target[rows], sample_weight=weight[rows])
        if t in policy:
            by_arm = np.column_stack(
                [_predict(model, _design(frame, t, m, policy, arm=a)) for a in range(m)]
            )
            carried = np.sum(density[t] * by_arm, axis=1)
        else:
            carried = _predict(model, X)
        target = np.where(risk[t], carried, 0.0)
    return float(np.sum(weight * target) / np.sum(weight))


def _predict(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    return np.asarray(model.predict(X))


def outcome_rung(rung: str, plan: str, K: int, m: int) -> Any:
    """The outcome-learner ladder: O1 the declared GLM, O2 W-blind, O3 intercept, O4 L-blind."""
    policy = tuple(sorted(law_.policy_nodes(plan, K)))
    if rung == "O1":
        return QuasiBinomialGLM()
    if rung == "O2":
        return ColumnSubsetGLM("W", m, K, policy)
    if rung == "O3":
        return InterceptOnly()
    if rung == "O4":
        return ColumnSubsetGLM("L", m, K, policy)
    raise ValueError(f"unknown rung {rung!r}")


def independent_se(values: np.ndarray) -> float:
    """The standard deviation of a column of replicate estimates."""
    values = np.asarray(values, dtype=float)
    return float(np.std(values, ddof=1)) if values.size > 1 else math.nan
