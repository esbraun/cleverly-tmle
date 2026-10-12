# A probe, not a registered study. `cleverly` at 572501b8.
#
# The investigation of 2026-10-08 ran this script against `cleverly` at commit 572501b8. It is
# committed so that the numbers RM40 cites from `drdr_summary.log` have a command and seeds.
# Edits from the copy that ran: this comment block, and the script names in the usage text.
#
# | item | value |
# | --- | --- |
# | limits | `python reviews/rm40/drdr_sim.py limits` (2e6 draws, seed 12345). It writes `_drdr_limits.npz` beside the script. That file is not committed; this command regenerates it |
# | runs | `python reviews/rm40/drdr_sim.py run <setting> <n> <R> <out.csv> --procs 4 [--start <first rep>]`, with the settings, n and R of each row of `drdr_summary.log` |
# | seeds | replicate `rep` of setting `s` at size `n` uses `1_000_003 * (index(s) + 1) + 7919 * n + rep`, with `index` the position in `SETTINGS` |
# | top-up runs | `qwrong_gglm` n = 16,000: 400 reps, then 600 with `--start 400`. `qwrong_gslow` n = 16,000: 300, then 300 with `--start 300`. `qwrong_gknown` n = 4,000: 400, then 600 with `--start 400` |
# | analysis | `python reviews/rm40/drdr_analyze.py`, which reads `_drdr_runs/*.csv` beside it. The raw CSVs are not committed |
# | threads | one BLAS thread per process (`OMP_NUM_THREADS=1`) |
#
"""DRTMLE doubly robust interval check (verifier probe, not evidence).

Own code. Calls the shipped ``cleverly.DRTMLE`` from origin/main (572501b8) on a law where the
misspecified outcome regression's residual is NOT a function of the true propensity score
(and, for the mirror case, where the misspecified mechanism's residual is not a function of the
true outcome regression).

Law ``L``: W1, W2 iid N(0, 1);
    g0(W)    = expit(0.6 W1 + 0.3 W2)
    Q0(a, W) = expit(-0.2 + 0.5 a - 1.0 W1 + 2.0 W2)
Targets: ey[1], ey[0], ate.

Settings (outcome learner / treatment learner):
    qwrong_gglm   : logistic GLM on (A, W1)           / logistic GLM on (W1, W2)  (correct, parametric)
    qwrong_gslow  : logistic GLM on (A, W1)           / correct GLM + logit offset n^(-1/3) W2
                    (a model of a flexible learner whose bias is n^(-1/3) in a level-set-breaking direction)
    qwrong_ghgb   : logistic GLM on (A, W1)           / HistGradientBoostingClassifier (fixed tuning)
    qwrong_gknown : logistic GLM on (A, W1)           / the true g0 (H2 control)
    qglm_gwrong   : logistic GLM on (A, W1, W2) (correct) / logistic GLM on (W1) (misspecified)
    qslow_gwrong  : correct Q GLM + logit offset n^(-1/3) W1 / logistic GLM on (W1)

Every fit: DRTMLE(guard=("Q","g"), reduction="univariate", cross_fit=False), reduced
regressions = cubic polynomials in the raw scalar design (least squares for Qr and gr2,
logistic for gr1). A first draft used B-splines in logit; their linear extrapolation made
Qr/g explode at small g and was abandoned (see the report).

Per replicate, the script also records an in-sample proxy of the conditioning remainder of the
outcome-misspecification branch (Benkeser et al. 2017, Appendix A, R_3), using the true law:
    R3 = mean_i[ (Q0_1 - Qbar_n,1 - Qr0_1)(W_i) * (g0 - g_n)(W_i) / g0(W_i) ]   (arm 1)
where Qr0_1 = E[Q0_1 - Qbar_lim,1 | g0(W)] by Gauss-Hermite quadrature along the g0 level set.
sqrt(n) * R3 / SD is the predicted first-order standardized bias that Theorem 1 assumes away.

Usage:
    python reviews/rm40/drdr_sim.py limits
    python reviews/rm40/drdr_sim.py time
    python reviews/rm40/drdr_sim.py run <setting> <n> <R> <out.csv> [--procs 4]
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
import warnings
from multiprocessing import Pool
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import SplineTransformer

A_G = np.array([0.6, 0.3])
B_Q = (-0.2, 0.5, -1.0, 2.0)  # intercept, A, W1, W2
G_BOUNDS = (0.01, 0.99)
SETTINGS = ("qwrong_gglm", "qwrong_gslow", "qwrong_ghgb", "qwrong_gknown", "qglm_gwrong", "qslow_gwrong",
            "qwrong_ghgbcf", "qslow2_gwrong2", "qwrong_gglmcf")


def g0(w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    return expit(A_G[0] * w1 + A_G[1] * w2)


def q0(a: Any, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    return expit(B_Q[0] + B_Q[1] * a + B_Q[2] * w1 + B_Q[3] * w2)


def draw(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal(n)
    w2 = rng.standard_normal(n)
    a = rng.binomial(1, g0(w1, w2)).astype(float)
    y = rng.binomial(1, q0(a, w1, w2)).astype(float)
    return pd.DataFrame({"Y": y, "A": a, "W1": w1, "W2": w2})


# ------------------------------------------------------------------ learners
def _logistic(x: np.ndarray, y: np.ndarray, offset: np.ndarray | None = None) -> np.ndarray:
    """Unpenalized logistic MLE by Newton, with an optional fixed offset."""
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(X.shape[1])
    off = 0.0 if offset is None else offset
    for _ in range(100):
        p = expit(X @ beta + off)
        h = (X * (p * (1 - p))[:, None]).T @ X
        step = np.linalg.solve(h + 1e-12 * np.eye(len(beta)), X.T @ (y - p))
        beta += step
        if np.max(np.abs(step)) < 1e-11:
            break
    return beta


class Glm(BaseEstimator, ClassifierMixin):
    """Logistic GLM on chosen design columns, with an optional logit offset kappa * col."""

    def __init__(self, columns: tuple[int, ...] = (), offset_col: int = -1, offset_coef: float = 0.0):
        self.columns = columns
        self.offset_col = offset_col
        self.offset_coef = offset_coef

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> "Glm":
        X = np.asarray(X, dtype=float)
        self.beta_ = _logistic(X[:, list(self.columns)], np.asarray(y, dtype=float))
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        eta = self.beta_[0] + X[:, list(self.columns)] @ self.beta_[1:]
        if self.offset_col >= 0:
            eta = eta + self.offset_coef * X[:, self.offset_col]
        p = expit(eta)
        return np.column_stack([1 - p, p])


class KnownG(BaseEstimator, ClassifierMixin):
    """Returns the true g0 from the design [W1, W2]."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> "KnownG":
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        p = g0(X[:, 0], X[:, 1])
        return np.column_stack([1 - p, p])


def _basis() -> SplineTransformer:
    return SplineTransformer(n_knots=5, degree=3, extrapolation="linear", include_bias=False)


def _lg(x: Any) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(len(x), -1)[:, :1]
    return logit(np.clip(x, 1e-6, 1 - 1e-6))


def _poly(X: Any) -> np.ndarray:
    """(1, x, x^2, x^3) of the raw scalar design, which lies in [0, 1]: no extrapolation."""
    x = np.asarray(X, dtype=float).reshape(len(X), -1)[:, 0]
    return np.column_stack([np.ones_like(x), x, x**2, x**3])


class SplineReg(BaseEstimator, RegressorMixin):
    """Cubic polynomial least squares in the raw scalar design (name kept from the first draft)."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> "SplineReg":
        B = _poly(X)
        y = np.asarray(y, float)
        sw = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        r = np.sqrt(sw)
        self.coef_ = np.linalg.lstsq(B * r[:, None], y * r, rcond=None)[0]
        return self

    def predict(self, X: Any) -> np.ndarray:
        return _poly(X) @ self.coef_


class SplineClf(BaseEstimator, ClassifierMixin):
    """Cubic polynomial logistic regression in the raw scalar design (Newton IRLS, tiny ridge)."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> "SplineClf":
        B = _poly(X)
        y = np.asarray(y, float)
        sw = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        beta = np.zeros(B.shape[1])
        beta[0] = logit(np.clip(np.average(y, weights=sw), 1e-6, 1 - 1e-6))
        for _ in range(100):
            p = expit(B @ beta)
            h = (B * (sw * p * (1 - p))[:, None]).T @ B + 1e-8 * np.eye(B.shape[1])
            step = np.linalg.solve(h, B.T @ (sw * (y - p)))
            beta += step
            if np.max(np.abs(step)) < 1e-10:
                break
        self.coef_ = beta
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        p = expit(_poly(X) @ self.coef_)
        return np.column_stack([1 - p, p])


def learners(setting: str, n: int) -> tuple[Any, Any]:
    # outcome design is [A, W1, W2]; treatment design is [W1, W2]
    slow = n ** (-1.0 / 3.0)
    q_wrong = Glm(columns=(0, 1))
    q_right = Glm(columns=(0, 1, 2))
    q_slow = Glm(columns=(0, 1, 2), offset_col=1, offset_coef=slow)
    g_right = Glm(columns=(0, 1))
    g_wrong = Glm(columns=(0,))
    g_slow = Glm(columns=(0, 1), offset_col=1, offset_coef=slow)
    g_hgb = HistGradientBoostingClassifier(
        max_iter=100, learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=20, early_stopping=False, random_state=0
    )
    return {
        "qwrong_gglm": (q_wrong, g_right),
        "qwrong_gslow": (q_wrong, g_slow),
        "qwrong_ghgb": (q_wrong, g_hgb),
        "qwrong_gknown": (q_wrong, KnownG()),
        "qglm_gwrong": (q_right, g_wrong),
        "qslow_gwrong": (q_slow, g_wrong),
        # second round (after the first results): cross-fitted HGB, a stronger mirror witness,
        # and the parametric case under the shipped 10-fold cross-fitting default
        "qwrong_ghgbcf": (q_wrong, g_hgb),
        "qslow2_gwrong2": (QPerp(slow), Glm(columns=(1,))),
        "qwrong_gglmcf": (q_wrong, g_right),
    }[setting]


class QPerp(BaseEstimator, ClassifierMixin):
    """Correct outcome GLM on (A, W1, W2) plus a logit offset 3 n^(-1/3) (2 W1 + W2)/sqrt(5):
    a slow learner whose error breaks the level sets of Q0 (index -W1 + 2 W2)."""

    def __init__(self, slow: float = 0.0):
        self.slow = slow

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> "QPerp":
        X = np.asarray(X, dtype=float)
        self.beta_ = _logistic(X[:, :3], np.asarray(y, dtype=float))
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        eta = self.beta_[0] + X[:, :3] @ self.beta_[1:] + 3.0 * self.slow * (2 * X[:, 1] + X[:, 2]) / np.sqrt(5.0)
        p = expit(eta)
        return np.column_stack([1 - p, p])


# ------------------------------------------------------------------ population quantities
def _gh(nodes: int = 120) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    return x, w / np.sqrt(2 * np.pi)


def truth() -> dict[str, float]:
    x, w = _gh()
    W1, W2 = np.meshgrid(x, x, indexing="ij")
    M = np.outer(w, w)
    e1 = float((M * q0(1.0, W1, W2)).sum())
    e0 = float((M * q0(0.0, W1, W2)).sum())
    return {"ey[1]": e1, "ey[0]": e0, "ate": e1 - e0}


def limits(big: int = 2_000_000, seed: int = 12345) -> dict[str, np.ndarray]:
    """Large-sample limits of the misspecified GLMs."""
    d = draw(big, seed)
    w1, w2, a, y = d.W1.to_numpy(), d.W2.to_numpy(), d.A.to_numpy(), d.Y.to_numpy()
    bq = _logistic(np.column_stack([a, w1]), y)  # Q wrong: (1, A, W1)
    bg = _logistic(w1[:, None], a)  # g wrong: (1, W1)
    return {"q_wrong": bq, "g_wrong": bg}


def qr0_arm(beta_q_wrong: np.ndarray, arm: float, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """Qr0_a(W) = E[Q0(a, W) - Qbar_lim(a, W) | g0(W)] at sample points, by quadrature along the level set."""
    s = np.linalg.norm(A_G)
    u_dir = A_G / s
    v_dir = np.array([-u_dir[1], u_dir[0]])
    u = w1 * u_dir[0] + w2 * u_dir[1]
    x, wt = _gh()
    W1 = u[:, None] * u_dir[0] + x[None, :] * v_dir[0]
    W2 = u[:, None] * u_dir[1] + x[None, :] * v_dir[1]
    resid = q0(arm, W1, W2) - expit(beta_q_wrong[0] + beta_q_wrong[1] * arm + beta_q_wrong[2] * W1)
    return (resid * wt[None, :]).sum(axis=1)


# ------------------------------------------------------------------ one replicate
_LIM: dict[str, np.ndarray] | None = None
_TRUTH: dict[str, float] | None = None


def _init() -> None:
    global _LIM, _TRUTH
    _LIM = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_drdr_limits.npz"))
    _LIM = {k: _LIM[k] for k in _LIM.files}
    _TRUTH = truth()
    warnings.simplefilter("ignore")


def one(args: tuple[str, int, int]) -> list[dict[str, Any]]:
    from cleverly.estimators import DRTMLE

    setting, n, rep = args
    seed = 1_000_003 * (SETTINGS.index(setting) + 1) + 7919 * n + rep
    d = draw(n, seed)
    outcome, treatment = learners(setting, n)
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    try:
        res = (
            DRTMLE(
                outcome_learner=outcome,
                treatment_learner=treatment,
                reduced_outcome_learner=SplineReg(),
                reduced_treatment_learner=SplineClf(),
                cross_fit=setting.endswith("cf"),
                estimands=("ey", "ate"),
                simultaneous=False,
                g_bounds=G_BOUNDS,
                guard=("Q", "g"),
                reduction="univariate",
                random_state=0,
            )
            .fit(d, outcome="Y", treatment="A", covariates=["W1", "W2"])
            .single()
        )
    except Exception as exc:  # noqa: BLE001
        return [{"setting": setting, "n": n, "rep": rep, "estimand": "error", "error": repr(exc)[:300]}]
    secs = time.perf_counter() - t0
    red = res.repeats[0].fluctuations["mean"].reduction
    conv = (bool(red.converged), str(red.exit_reason), int(red.n_outer))
    w1, w2, a = d.W1.to_numpy(), d.W2.to_numpy(), d.A.to_numpy()
    # in-sample R3 proxy (outcome-misspecification branch), arm 1 and arm 0
    nuis = res.repeats[0].nuisance
    gn1 = np.asarray(nuis.propensity.arm(1.0), dtype=float)
    gt1 = g0(w1, w2)
    r3 = {}
    if setting.startswith("qwrong"):
        bq = _LIM["q_wrong"]
        qn1 = np.asarray(nuis.outcome.arms[1.0], dtype=float)
        qn0 = np.asarray(nuis.outcome.arms[0.0], dtype=float)
        f1 = q0(1.0, w1, w2) - qn1 - qr0_arm(bq, 1.0, w1, w2)
        f0 = q0(0.0, w1, w2) - qn0 - qr0_arm(bq, 0.0, w1, w2)
        r3["ey[1]"] = float(np.mean(f1 * (gt1 - gn1) / gt1))
        r3["ey[0]"] = float(np.mean(f0 * ((1 - gt1) - (1 - gn1)) / (1 - gt1)))
        r3["ate"] = r3["ey[1]"] - r3["ey[0]"]
    # projected SE for the parametric-g case: residual of the curve on the GLM score columns
    X = np.column_stack([np.ones(n), w1, w2])
    S = X * (a - gn1)[:, None]
    for name in ("ey[1]", "ey[0]", "ate"):
        est = res.estimates[name]
        lo, hi = est.ci
        row = {
            "setting": setting,
            "n": n,
            "rep": rep,
            "estimand": name,
            "truth": _TRUTH[name],
            "psi": float(est.psi),
            "se": float(est.std_error),
            "lo": float(lo),
            "hi": float(hi),
            "covered": int(lo <= _TRUTH[name] <= hi),
            "r3": r3.get(name, np.nan),
            "secs": secs,
            "converged": conv[0],
            "exit": conv[1],
            "n_outer": conv[2],
        }
        ic = getattr(est, "influence_curve", None)
        if ic is None:
            ic = getattr(est, "ic", None)
        if ic is not None:
            ic = np.asarray(ic, dtype=float)
            if ic.shape == (n,):
                coef, *_ = np.linalg.lstsq(S, ic - ic.mean(), rcond=None)
                resid = ic - ic.mean() - S @ coef
                row["se_ic"] = float(np.std(ic) / np.sqrt(n))
                row["se_proj"] = float(np.std(resid) / np.sqrt(n))
        rows.append(row)
    return rows


def main() -> None:
    mode = sys.argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    if mode == "limits":
        lim = limits()
        np.savez(os.path.join(here, "_drdr_limits.npz"), **lim)
        print({k: v.tolist() for k, v in lim.items()}, truth())
        return
    if mode == "time":
        _init()
        for setting in SETTINGS:
            for n in (1000, 4000, 16000):
                out = one((setting, n, 0))
                print(setting, n, out[0].get("secs"), out[0].get("error", ""), {k: out[2].get(k) for k in ("psi", "se", "se_ic", "se_proj", "r3")}, flush=True)
        return
    setting, n, R, out = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 4
    start = int(sys.argv[sys.argv.index("--start") + 1]) if "--start" in sys.argv else 0
    t0 = time.time()
    with Pool(procs, initializer=_init) as pool:
        rows = [r for chunk in pool.imap_unordered(one, [(setting, n, i) for i in range(start, start + R)], chunksize=4) for r in chunk]
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"{setting} n={n} R={R} done in {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
