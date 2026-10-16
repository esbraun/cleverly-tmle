"""The exact sampling SD of a long-horizon estimator on a frame of the K-node law.

The single-fold ``LTMLE`` of :mod:`cleverly.longitudinal.sequential`, its unfluctuated plug-in
and the single-fold ``SDR`` recursion of :mod:`cleverly.longitudinal.sdr` are re-implemented here
in numpy for the declared learners, so that each one is an M-estimator of a stacked parameter

    theta = (beta_K, eps_K, ..., beta_1, eps_1, psi)

with one estimating function per node regression, per fluctuation and for the estimate.  Its
influence function is ``IF_i = -M^{-1} phi_i(theta)`` with ``M = E[d phi / d theta]``.  On the
weighted exact frame (:func:`tests.discrete_law_longitudinal_horizon.weighted_frame`) the frame
mean is the law's expectation, so ``M`` (by central differences of the frame mean) and the
standard deviation of ``IF`` are exact.  That SD is the sampling SD of the estimator: it keeps
the outcome-fit terms that the reported curve omits when the mechanism is wrong or the fit is
untargeted, and it is what the shipped bias verdict divides by.

The node learners are the declared ladder: ``"glm"`` the declared quasibinomial GLM on
``[W, L_2..L_t]`` and the policy arm blocks, ``"wblind"`` without ``W``, ``"lblind"`` without
``L``, ``"int"`` the intercept, and ``"offset"`` the GLM with the oracle offset
(:class:`tests.studies.long_horizon_common.OracleOffsetGLM`: logit scale in a quasibinomial
fit, additive in the least-squares fit of the ``SDR`` transform).  The mechanism is read from
the frame's ``g{t}_{k}`` and ``pc{t}`` columns, from one frame per node when a pattern needs it.

``tests/unit/test_long_horizon_sampling_sd.py`` checks every estimate and every reported curve
against ``cleverly`` on the frame.  This module imports nothing from ``cleverly``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from tests import discrete_law_longitudinal_horizon as law_
from tests.discrete_law_longitudinal_horizon import Law

#: The node learners this module re-implements.
LEARNERS = ("glm", "wblind", "lblind", "int", "offset")
#: The estimators this module re-implements.
KINDS = ("ltmle", "plugin", "sdr")


def _logit(x: np.ndarray) -> np.ndarray:
    return np.asarray(logit(np.clip(x, 1e-15, 1.0 - 1e-15)))


def _column(frame: pd.DataFrame, name: str) -> np.ndarray:
    return np.nan_to_num(frame[name].to_numpy(dtype=float))


@dataclass(frozen=True)
class _Columns:
    """The frame's columns as arrays; a node a row never reached reads zero."""

    n: int
    p: np.ndarray
    W: np.ndarray
    L: dict[int, np.ndarray]
    A: dict[int, np.ndarray]
    observed: dict[int, np.ndarray]
    C: dict[int, np.ndarray]
    Y: np.ndarray


def _columns(frame: pd.DataFrame, K: int) -> _Columns:
    n = len(frame)
    weight = frame["w"].to_numpy(dtype=float) if "w" in frame else np.ones(n)
    L = {t: _column(frame, f"L{t}") for t in range(2, K + 1)}
    L[1] = np.zeros(n)
    A = {t: _column(frame, f"A{t}").astype(int) for t in range(1, K + 1)}
    A[0] = np.zeros(n, dtype=int)
    return _Columns(
        n=n,
        p=weight / weight.sum(),
        W=_column(frame, "W"),
        L=L,
        A=A,
        observed={t: frame[f"A{t}"].notna().to_numpy() for t in range(1, K + 1)},
        C={t: _column(frame, f"C{t}") for t in range(1, K + 1)},
        Y=_column(frame, "Y"),
    )


def _mechanism(frame: pd.DataFrame, t: int, m: int) -> tuple[np.ndarray, np.ndarray]:
    g = np.column_stack(
        [np.nan_to_num(frame[f"g{t}_{k}"].to_numpy(dtype=float), nan=1.0) for k in range(m)]
    )
    return g, np.nan_to_num(frame[f"pc{t}"].to_numpy(dtype=float), nan=1.0)


def _irls(X: np.ndarray, z: np.ndarray, weight: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """The weighted quasibinomial IRLS of ``z`` on ``X`` with a fixed logit offset."""
    beta = np.zeros(X.shape[1])
    if not np.any(offset):
        mean = np.clip(np.average(z, weights=weight), 1e-8, 1.0 - 1e-8)
        beta[0] = np.log(mean / (1.0 - mean))
    for _ in range(200):
        fitted = expit(X @ beta + offset)
        variance = np.clip(fitted * (1.0 - fitted), 1e-12, None)
        step = np.linalg.solve(
            (X * (weight * variance)[:, None]).T @ X, X.T @ (weight * (z - fitted))
        )
        beta = beta + step
        if np.max(np.abs(step)) < 1e-13:
            break
    return beta


def _solve_epsilon(offset: np.ndarray, z: np.ndarray, weight: np.ndarray) -> float:
    """The intercept fluctuation: ``sum weight (z - expit(offset + eps)) = 0``."""
    eps = 0.0
    for _ in range(200):
        fitted = expit(offset + eps)
        step = np.sum(weight * (z - fitted)) / np.sum(weight * fitted * (1.0 - fitted))
        eps += float(step)
        if abs(step) < 1e-14:
            break
    return eps


class Sequential:
    """One estimator of one plan mean on a frame of the K-node law.

    Parameters
    ----------
    frame : pandas.DataFrame
        A frame with the columns of :func:`~tests.discrete_law_longitudinal_horizon.draw`; a
        column ``w`` weights the rows.
    K : int
        Decision nodes.
    m : int
        Arms.
    plan : str
        The plan, a name of :func:`~tests.discrete_law_longitudinal_horizon.plan`.
    law : Law
        The law of the oracle offset; it does not enter any other learner.
    kind : str
        ``"ltmle"``, ``"plugin"`` (no fluctuation) or ``"sdr"``.
    learners : str or Mapping
        One learner for every node, or ``{node: learner}``.
    mechanism : pandas.DataFrame or Mapping or None
        The frame whose ``g{t}_{k}`` and ``pc{t}`` columns the ratios read, or
        ``{node: frame}``; ``None`` reads ``frame``.
    skip : Iterable
        ``LTMLE`` nodes whose fluctuation is pinned at zero.
    pseudo : str
        ``"targeted"`` (declared) or ``"untargeted"``: node ``t - 1`` regresses node ``t``'s
        unfluctuated value.
    plan_nodes : Iterable or None
        The nodes of a mutated plan (``lagged``, ``mix_noprev``).
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        K: int,
        m: int,
        plan: str,
        law: Law = law_.L_R,
        *,
        kind: str = "ltmle",
        learners: str | Mapping[int, str] = "glm",
        mechanism: pd.DataFrame | Mapping[int, pd.DataFrame] | None = None,
        skip: Iterable[int] = (),
        pseudo: str = "targeted",
        plan_nodes: Iterable[int] | None = None,
    ) -> None:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}; got {kind!r}")
        self.K, self.m, self.kind, self.plan = K, m, kind, plan
        self.learners = (
            dict.fromkeys(range(1, K + 1), learners)
            if isinstance(learners, str)
            else {t: learners.get(t, "glm") for t in range(1, K + 1)}
        )
        unknown = set(self.learners.values()) - set(LEARNERS)
        if unknown:
            raise ValueError(f"unknown learners {sorted(unknown)}")
        self.skip, self.pseudo = set(skip), pseudo
        cols = self.columns = _columns(frame, K)
        n, rows = cols.n, np.arange(cols.n)
        self.policy = law_.policy_nodes(plan, K)
        tree = law_.histories(K, m)
        density = law_.plan(plan, K, m, plan_nodes)
        index = {
            t: law_.history_index(t, m, cols.W, {s: cols.L[s] for s in range(2, t + 1)}, cols.A)
            for t in range(1, K + 1)
        }
        self.q = {t: density(t, tree)[index[t]] for t in range(1, K + 1)}
        frames: Mapping[int, pd.DataFrame]
        if mechanism is None or isinstance(mechanism, pd.DataFrame):
            frames = dict.fromkeys(range(1, K + 1), frame if mechanism is None else mechanism)
        else:
            frames = {t: mechanism.get(t, frame) for t in range(1, K + 1)}
        self.risk: dict[int, np.ndarray] = {}
        self.follow: dict[int, np.ndarray] = {}
        self.pi: dict[int, np.ndarray] = {}
        self.pk: dict[int, np.ndarray] = {}
        risk = np.ones(n, dtype=bool)
        for t in range(1, K + 1):
            self.risk[t] = risk & cols.observed[t]
            assigned = self.q[t][rows, cols.A[t]]
            self.follow[t] = self.risk[t] & (assigned > 0) & (cols.C[t] == 1)
            g, pc = _mechanism(frames[t], t, m)
            self.pi[t] = np.where(self.risk[t], assigned, 1.0)
            self.pk[t] = np.where(self.risk[t], g[rows, cols.A[t]] * pc, 1.0)
            risk = self.follow[t]
        numerator, denominator = np.ones(n), np.ones(n)
        self.H: dict[int, np.ndarray] = {}
        for t in range(1, K + 1):
            numerator, denominator = numerator * self.pi[t], denominator * self.pk[t]
            self.H[t] = np.where(self.follow[t], numerator / denominator, 0.0)
        self.linear = {t: kind == "sdr" and t < K for t in range(1, K + 1)}
        self.X: dict[int, np.ndarray] = {}
        self.Xa: dict[int, list[np.ndarray]] = {}
        self.Xo: dict[int, np.ndarray] = {}
        self.offset: dict[int, np.ndarray] = {}
        self.offset_by_arm: dict[int, np.ndarray] = {}
        oracle = law_.gformula(K, m, plan, law) if "offset" in self.learners.values() else None
        for t in range(1, K + 1):
            self.X[t] = self._design(t)
            if t in self.policy:
                self.Xa[t] = [self._design(t, arm) for arm in range(m)]
                self.Xo[t] = np.stack(self.Xa[t])[cols.A[t], rows]
            if self.learners[t] == "offset":
                assert oracle is not None
                by_arm = oracle.qa[t][index[t]]
                self.offset_by_arm[t] = by_arm
                arm = cols.A[t] if t in self.policy else np.argmax(self.q[t], axis=1)
                self.offset[t] = by_arm[rows, arm]
        self.dims: list[tuple[str, int, int]] = []
        for t in range(K, 0, -1):
            self.dims.append(("beta", t, self.X[t].shape[1]))
            if kind == "ltmle":
                self.dims.append(("eps", t, 1))
        self.dims.append(("psi", 0, 1))

    # ------------------------------------------------------------------------- designs
    def _design(self, t: int, arm: int | None = None) -> np.ndarray:
        cols, learner = self.columns, self.learners[t]
        out = [np.ones(cols.n)]
        if learner in ("glm", "lblind", "offset"):
            out.append(cols.W)
        if learner in ("glm", "wblind", "offset"):
            out += [cols.L[s] for s in range(2, t + 1)]
        if learner != "int":
            for s in sorted(self.policy):
                if s > t:
                    continue
                codes = np.full(cols.n, arm) if s == t and arm is not None else cols.A[s]
                out += [(codes == k).astype(float) for k in range(1, self.m)]
        return np.column_stack(out)

    def _observed_design(self, t: int) -> np.ndarray:
        return self.Xo.get(t, self.X[t])

    def _predict(self, t: int, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """The node-``t`` regression at the observed arm, and at every arm on a policy node."""
        linear, has_offset = self.linear[t], t in self.offset

        def value(X: np.ndarray, offset: np.ndarray | None) -> np.ndarray:
            eta = X @ beta
            if linear:
                return eta + (offset if offset is not None else 0.0)
            return np.asarray(expit(eta + (_logit(offset) if offset is not None else 0.0)))

        if t in self.policy:
            by_arm = np.column_stack(
                [
                    value(self.Xa[t][a], self.offset_by_arm[t][:, a] if has_offset else None)
                    for a in range(self.m)
                ]
            )
            return by_arm[np.arange(self.columns.n), self.columns.A[t]], by_arm
        return value(self.X[t], self.offset.get(t)), None

    def _fit_beta(self, t: int, z: np.ndarray) -> np.ndarray:
        rows = self.follow[t]
        X, target, weight = self._observed_design(t)[rows], z[rows], self.columns.p[rows]
        offset = self.offset[t][rows] if t in self.offset else np.zeros(int(rows.sum()))
        if self.linear[t]:
            weighted = X * weight[:, None]
            return np.asarray(np.linalg.solve(weighted.T @ X, weighted.T @ (target - offset)))
        return _irls(X, target, weight, _logit(offset) if t in self.offset else offset)

    # ------------------------------------------------------------------------- the recursion
    def _run(self, theta: np.ndarray | None) -> np.ndarray:
        """Fit when ``theta`` is ``None``; else the per-row estimating functions at ``theta``."""
        cols, K = self.columns, self.K
        given = None if theta is None else self._unpack(theta)
        functions: list[np.ndarray] = []
        parameters: list[np.ndarray] = []
        z, z_initial = cols.Y.copy(), cols.Y.copy()
        values: dict[int, np.ndarray] = {K + 1: cols.Y.copy()}
        observed: dict[int, np.ndarray] = {}
        parts: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        epsilon: dict[int, float] = {}
        for t in range(K, 0, -1):
            rows = self.follow[t]
            target = z if (self.kind == "sdr" or self.pseudo == "targeted") else z_initial
            beta = self._fit_beta(t, target) if given is None else given[("beta", t)]
            at_arm, by_arm = self._predict(t, beta)
            functions.append(
                np.where(rows[:, None], self._observed_design(t) * (target - at_arm)[:, None], 0.0)
            )
            parameters.append(np.atleast_1d(beta))
            if self.kind == "sdr":
                values[t] = at_arm if by_arm is None else np.sum(self.q[t] * by_arm, axis=1)
                observed[t] = at_arm
                numerator, denominator = np.ones(cols.n), np.ones(cols.n)
                transform = values[t].copy()
                for s in range(t, K + 1):
                    numerator, denominator = numerator * self.pi[s], denominator * self.pk[s]
                    ratio = np.where(self.follow[s], numerator / denominator, 0.0)
                    transform = transform + ratio * (values[s + 1] - observed[s])
                z = np.where(self.risk[t], transform, 0.0)
                continue
            eps = 0.0
            if self.kind == "ltmle":
                if given is not None:
                    eps = float(given[("eps", t)][0])
                elif t not in self.skip:
                    eps = _solve_epsilon(
                        _logit(at_arm[rows]), target[rows], cols.p[rows] * self.H[t][rows]
                    )
                targeted = np.asarray(expit(_logit(at_arm) + eps))
                if t in self.skip:
                    functions.append(np.full((cols.n, 1), eps))
                else:
                    functions.append(np.where(rows, self.H[t] * (target - targeted), 0.0)[:, None])
                parameters.append(np.array([eps]))
            else:
                targeted = at_arm
            epsilon[t] = eps
            parts[t] = (target, targeted)
            if by_arm is not None:
                carried = np.sum(self.q[t] * expit(_logit(by_arm) + eps), axis=1)
                carried_initial = np.sum(self.q[t] * by_arm, axis=1)
            else:
                carried, carried_initial = targeted, at_arm
            z = np.where(self.risk[t], carried, 0.0)
            z_initial = np.where(self.risk[t], carried_initial, 0.0)
        psi = float(np.sum(cols.p * z)) if given is None else float(given[("psi", 0)][0])
        functions.append((z - psi)[:, None])
        parameters.append(np.array([psi]))
        if given is None:
            self.theta = np.concatenate(parameters)
            self.psi = psi
            self.epsilon = epsilon
            curve = z - psi
            if self.kind != "sdr":
                for t in range(1, K + 1):
                    target, targeted = parts[t]
                    curve = curve + np.where(self.follow[t], self.H[t] * (target - targeted), 0.0)
            self.curve = curve
        return np.hstack(functions)

    def _unpack(self, theta: np.ndarray) -> dict[tuple[str, int], np.ndarray]:
        out, at = {}, 0
        for kind, t, width in self.dims:
            out[(kind, t)] = theta[at : at + width]
            at += width
        return out

    def fit(self) -> Sequential:
        """Fit every node; sets ``psi``, ``theta``, ``epsilon`` and the reported ``curve``."""
        self._run(None)
        return self

    def phi(self, theta: np.ndarray) -> np.ndarray:
        """``(rows, len(theta))`` estimating functions at ``theta``."""
        return self._run(theta)

    def influence(self, step: float = 1e-6) -> np.ndarray:
        """The per-row influence function of ``psi``, ``-M^{-1} phi_i`` at the fit."""
        theta, p = self.theta, self.columns.p
        size = theta.size
        jacobian = np.zeros((size, size))
        for j in range(size):
            h = step * max(1.0, abs(theta[j]))
            up, down = theta.copy(), theta.copy()
            up[j] += h
            down[j] -= h
            jacobian[:, j] = (p @ self.phi(up) - p @ self.phi(down)) / (2.0 * h)
        functions = self.phi(theta)
        #: The largest frame mean of an estimating function at the fit: zero up to the solvers.
        self.mean_function = float(np.max(np.abs(p @ functions)))
        self.IF = -np.linalg.solve(jacobian, functions.T).T[:, -1]
        return self.IF


# ----------------------------------------------------------------------------- readings
def sd(weight: np.ndarray, values: np.ndarray) -> float:
    """The SD of a mean-zero per-row function under ``weight``."""
    p = np.asarray(weight, float) / np.sum(weight)
    return float(np.sqrt(np.sum(p * np.asarray(values) ** 2)))


def kappa(weight: np.ndarray, values: np.ndarray) -> float:
    """``E D^4 / (E D^2)^2`` under ``weight``."""
    p = np.asarray(weight, float) / np.sum(weight)
    variance = np.sum(p * np.asarray(values) ** 2)
    return float(np.sum(p * np.asarray(values) ** 4) / variance**2)


def n_kappa(value: float) -> int:
    """The smallest n with ``sqrt((kappa - 1) / n) <= C_KAPPA``."""
    return int(np.ceil((value - 1.0) / law_.C_KAPPA**2))


@dataclass(frozen=True)
class Reading:
    """The exact reading of one mean or one contrast of two fits with the same settings.

    Parameters
    ----------
    estimate : float
        The estimate on the frame.
    truth : float
        The oracle value.
    sigma : float
        The sampling SD per unit, from the stacked influence function.
    sigma_curve : float
        The SD of the reported curve per unit.
    kappa : float
        The tail ratio of the sampling influence function.
    kappa_curve : float
        The tail ratio of the reported curve.
    mean_function : float
        The largest frame mean of an estimating function at the fit.
    """

    estimate: float
    truth: float
    sigma: float
    sigma_curve: float
    kappa: float
    kappa_curve: float
    mean_function: float

    @property
    def bias(self) -> float:
        """The estimate minus the truth."""
        return self.estimate - self.truth

    def d(self, n: int) -> float:
        """The displacement ``|bias| / (sigma / sqrt n)``."""
        return abs(self.bias) / (self.sigma / np.sqrt(n))


def reading(
    frame: pd.DataFrame,
    K: int,
    m: int,
    first: str,
    second: str | None = None,
    law: Law = law_.L_R,
    *,
    truth_first: str | None = None,
    **settings: Any,
) -> Reading:
    """The exact reading of ``psi(first)``, or of ``psi(first) - psi(second)``.

    ``truth_first`` names the plan whose truth the estimate of ``first`` targets, when the fit
    is a mutation of another plan (a policy fitted with a wrong density targets the declared
    policy).
    """
    weight = frame["w"].to_numpy(dtype=float) if "w" in frame else np.ones(len(frame))
    a = Sequential(frame, K, m, first, law, **settings).fit()
    influence, curve = a.influence(), a.curve
    estimate = a.psi
    truth = law_.truth(K, m, truth_first or first, law)
    mean_function = a.mean_function
    if second is not None:
        b = Sequential(frame, K, m, second, law, **settings).fit()
        influence = influence - b.influence()
        curve = curve - b.curve
        estimate -= b.psi
        truth -= law_.truth(K, m, second, law)
        mean_function = max(mean_function, b.mean_function)
    return Reading(
        estimate=float(estimate),
        truth=float(truth),
        sigma=sd(weight, influence),
        sigma_curve=sd(weight, curve),
        kappa=kappa(weight, influence),
        kappa_curve=kappa(weight, curve),
        mean_function=mean_function,
    )


def complement(frame: pd.DataFrame, nodes: Iterable[int]) -> pd.DataFrame:
    """``frame`` with ``g{t}_1`` read as one minus ``g{t}_2`` at ``nodes`` (the two-arm shortcut)."""
    out = frame.copy()
    for t in nodes:
        out[f"g{t}_1"] = 1.0 - out[f"g{t}_2"]
    return out
