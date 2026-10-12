# A probe, not a registered study. `cleverly` at 572501b8.
#
# The linear-remainder audit of 2026-10-08 ran this script against `cleverly` at commit
# 572501b8. It is committed so that the numbers RM40 cites from `lra_summary.log` have a
# command and seeds. Edits from the copy that ran: this comment block, the script names in the
# usage text, and the removal of an assertion on the name of the audit's temporary worktree.
#
# | item | value |
# | --- | --- |
# | outcome limit | `q_limit`: 2e6 draws, seed 4242 (logged `Qbar_lim: [0.0267, 0.2936, -0.6687]`) |
# | runs | `python reviews/rm40/lra_mar_sim.py run <setting> <n> <R> <out.csv>`, one process, one thread: `pislow` at (n, R) = (1,000, 300), (4,000, 200) and (16,000, 60); `piglm` and `piknown` at (4,000, 150) |
# | seeds | replicate `rep` of setting `s` at size `n` uses `7_000_003 * (index(s) + 1) + 7919 * n + rep`, with `SETTINGS = ("pislow", "piglm", "piknown")` |
# | constant | `KAPPA = 2`, the logit offset `2 n^(-1/3) W2` of `pislow` |
# | analysis | `python reviews/rm40/lra_analyze.py`, which reads `_lra_runs/*.csv` beside it. The raw CSVs are not committed |
#
"""Linear-remainder audit, simulation 1: DRTMLE randomized missing-outcome path (Diaz & van der Laan 2017).

Own code (auditor probe, not evidence). Calls the shipped ``cleverly.DRTMLE`` at origin/main
572501b8 from a temporary detached worktree; ``cleverly.__file__`` was asserted in the copy that ran
(removed in this committed copy).

Question. Diaz & van der Laan (2017), Appendix A.1, proof of Theorem 1, write the drift term in the
outcome-misspecified branch with ``e_hat_0 = E[Y - m1 | A=1, M=1, g0(W), g_hat(W)]`` and then
replace it by ``e_0 = E[Y - m1 | A=1, M=1, g0(W)]``, calling the difference (21) ``O_P(||g_hat - g0||^2)``.
By the X21 identity the difference is linear in ``g_hat - g0`` when the error of ``g_hat`` breaks
the level sets of ``g0`` and ``m0 - m1`` is not a function of ``g0``.

Law M (randomized trial, MAR outcome):
    W1, W2 iid N(0, 1);  A ~ Bern(0.5), known;
    pi0(a, W) = P(Delta = 1 | A = a, W) = expit(0.4 + 0.2 a + 0.6 W1 + 0.3 W2);
    Q0(a, W)  = expit(-0.2 + 0.5 a - 1.0 W1 + 2.0 W2);  Y observed when Delta = 1.
Outcome learner: logistic GLM on (A, W1) -- wrong; Q0 - Qbar is not a function of pi0.
Missingness learners (design [A, W1, W2]):
    pislow  : correct GLM plus logit offset KAPPA * n^(-1/3) * W2, KAPPA = 2 (level-set-breaking slow error)
    piglm   : correct GLM (parametric; theory predicts a conservative interval)
    piknown : the true pi0 (H2 control; the term is identically 0)
Reduced regressions: cubic polynomials in the scalar design (as in _drdr_sim.py).

Per replicate it records the in-sample proxy of the linear term, arm a:
    T_a = mean_i[ (Q0 - Qbar_n - e0)(a, W_i) * (pi0 - pi_n)(a, W_i) / pi_n(a, W_i) ]
with e0(a, W) = E[Q0(a, W) - Qbar_lim(a, W) | pi0(a, W)] by Gauss-Hermite quadrature along the level
set of the index 0.6 W1 + 0.3 W2. The drift enters the estimator with a minus sign (Diaz eq. 4/14).

Usage: python reviews/rm40/lra_mar_sim.py time | run <setting> <n> <R> <out.csv>
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import sys
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

IDX = np.array([0.6, 0.3])
PI = (0.4, 0.2)  # intercept, A
BQ = (-0.2, 0.5, -1.0, 2.0)
SETTINGS = ("pislow", "piglm", "piknown")
KAPPA = 2.0


def pi0(a: Any, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    return expit(PI[0] + PI[1] * a + IDX[0] * w1 + IDX[1] * w2)


def q0(a: Any, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    return expit(BQ[0] + BQ[1] * a + BQ[2] * w1 + BQ[3] * w2)


def draw(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w1, w2 = rng.standard_normal(n), rng.standard_normal(n)
    a = rng.binomial(1, 0.5, n).astype(float)
    d = rng.binomial(1, pi0(a, w1, w2)).astype(float)
    y = rng.binomial(1, q0(a, w1, w2)).astype(float)
    y = np.where(d == 1, y, np.nan)
    return pd.DataFrame({"Y": y, "A": a, "W1": w1, "W2": w2, "Delta": d})


def _logistic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(X.shape[1])
    for _ in range(100):
        p = expit(X @ beta)
        h = (X * (p * (1 - p))[:, None]).T @ X
        step = np.linalg.solve(h + 1e-12 * np.eye(len(beta)), X.T @ (y - p))
        beta += step
        if np.max(np.abs(step)) < 1e-11:
            break
    return beta


class Glm(BaseEstimator, ClassifierMixin):
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


class KnownPi(BaseEstimator, ClassifierMixin):
    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> "KnownPi":
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        p = pi0(X[:, 0], X[:, 1], X[:, 2])
        return np.column_stack([1 - p, p])


def _poly(X: Any) -> np.ndarray:
    x = np.asarray(X, dtype=float).reshape(len(X), -1)[:, 0]
    return np.column_stack([np.ones_like(x), x, x**2, x**3])


class PolyReg(BaseEstimator, RegressorMixin):
    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> "PolyReg":
        B = _poly(X)
        y = np.asarray(y, float)
        sw = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        r = np.sqrt(sw)
        self.coef_ = np.linalg.lstsq(B * r[:, None], y * r, rcond=None)[0]
        return self

    def predict(self, X: Any) -> np.ndarray:
        return _poly(X) @ self.coef_


class PolyClf(BaseEstimator, ClassifierMixin):
    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> "PolyClf":
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


def missingness_learner(setting: str, n: int) -> Any:
    slow = KAPPA * n ** (-1.0 / 3.0)
    return {
        "pislow": Glm(columns=(0, 1, 2), offset_col=2, offset_coef=slow),
        "piglm": Glm(columns=(0, 1, 2)),
        "piknown": KnownPi(),
    }[setting]


def _gh(nodes: int = 120) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    return x, w / np.sqrt(2 * np.pi)


def truth() -> dict[str, float]:
    x, w = _gh()
    W1, W2 = np.meshgrid(x, x, indexing="ij")
    M = np.outer(w, w)
    e1, e0 = float((M * q0(1.0, W1, W2)).sum()), float((M * q0(0.0, W1, W2)).sum())
    return {"ey[1]": e1, "ey[0]": e0, "ate": e1 - e0}


def q_limit(big: int = 2_000_000, seed: int = 4242) -> np.ndarray:
    d = draw(big, seed)
    obs = d.Delta.to_numpy() == 1
    return _logistic(np.column_stack([d.A.to_numpy()[obs], d.W1.to_numpy()[obs]]), d.Y.to_numpy()[obs])


def e0_arm(bq: np.ndarray, arm: float, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    s = np.linalg.norm(IDX)
    u_dir = IDX / s
    v_dir = np.array([-u_dir[1], u_dir[0]])
    u = w1 * u_dir[0] + w2 * u_dir[1]
    x, wt = _gh()
    W1 = u[:, None] * u_dir[0] + x[None, :] * v_dir[0]
    W2 = u[:, None] * u_dir[1] + x[None, :] * v_dir[1]
    resid = q0(arm, W1, W2) - expit(bq[0] + bq[1] * arm + bq[2] * W1)
    return (resid * wt[None, :]).sum(axis=1)


BQ_LIM: np.ndarray | None = None
TRUTH: dict[str, float] | None = None


def one(setting: str, n: int, rep: int) -> list[dict[str, Any]]:
    from cleverly.estimators import DRTMLE

    seed = 7_000_003 * (SETTINGS.index(setting) + 1) + 7919 * n + rep
    d = draw(n, seed)
    t0 = time.perf_counter()
    try:
        res = (
            DRTMLE(
                randomized=True,
                cross_fit=False,
                outcome_learner=Glm(columns=(0, 1)),
                missingness_learner=missingness_learner(setting, n),
                reduced_outcome_learner=PolyReg(),
                reduced_treatment_learner=PolyClf(),
                estimands=("ey", "ate"),
                simultaneous=False,
                g_bounds=(0.01, 0.99),
                guard=("Q", "g"),
                random_state=0,
            )
            .fit(
                d,
                outcome="Y",
                treatment="A",
                covariates=["W1", "W2"],
                delta="Delta",
                treatment_probabilities=np.full(n, 0.5),
            )
            .single()
        )
    except Exception as exc:  # noqa: BLE001
        return [{"setting": setting, "n": n, "rep": rep, "estimand": "error", "error": repr(exc)[:300]}]
    secs = time.perf_counter() - t0
    nuis = res.repeats[0].nuisance
    w1, w2 = d.W1.to_numpy(), d.W2.to_numpy()
    miss = np.asarray(nuis.missingness, dtype=float)
    term = {}
    for j, arm in enumerate(nuis.arms):
        qn = np.asarray(nuis.outcome.arms[arm], dtype=float)
        f = q0(arm, w1, w2) - qn - e0_arm(BQ_LIM, arm, w1, w2)
        pn = miss[:, j]
        term[f"ey[{int(arm)}]"] = float(np.mean(f * (pi0(arm, w1, w2) - pn) / pn))
    term["ate"] = term["ey[1]"] - term["ey[0]"]
    rows = []
    for name in ("ey[1]", "ey[0]", "ate"):
        est = res.estimates[name]
        lo, hi = est.ci
        rows.append(
            {
                "setting": setting,
                "n": n,
                "rep": rep,
                "estimand": name,
                "truth": TRUTH[name],
                "psi": float(est.psi),
                "se": float(est.std_error),
                "covered": int(lo <= TRUTH[name] <= hi),
                "term": term[name],
                "secs": secs,
                "status": str(getattr(est, "inference", "")),
            }
        )
    return rows


def init() -> None:
    global BQ_LIM, TRUTH
    import cleverly

    here = os.path.normcase(os.path.abspath(cleverly.__file__))
    BQ_LIM = q_limit()
    TRUTH = truth()
    warnings.simplefilter("ignore")
    print("cleverly:", here, "Qbar_lim:", BQ_LIM.round(4).tolist(), "truth:", TRUTH, flush=True)


def main() -> None:
    init()
    if sys.argv[1] == "time":
        for setting in SETTINGS:
            for n in (1000, 4000, 16000):
                out = one(setting, n, 0)
                print(setting, n, round(out[0].get("secs", -1), 2), out[0].get("error", ""), out[0].get("status"),
                      {k: out[2].get(k) for k in ("psi", "se", "term")}, flush=True)
        return
    setting, n, R, out = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    t0 = time.time()
    rows = [r for rep in range(R) for r in one(setting, n, rep)]
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"{setting} n={n} R={R} done in {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
