# A probe, not a registered study. RM40b step 0b, the gradient-boosting diagnosis.
#
# `drdr_summary.log` row `qwrong_ghgb` shows an in-sample DRTMLE ATE bias of -1.31, -2.28 and
# -2.62 SD at n = 1,000, 4,000 and 16,000, with the linear R_3 proxy near 0. This script runs the
# bounded probe of RM40 plan section 5.11 on fresh draws of law L. Law, learners, HGB tuning,
# reduced learners and the R_3 proxy come from `drdr_sim.py`; the seeds, the HGB cache, the
# extra fits and the diagnostics are new.
#
# | item | value |
# | --- | --- |
# | command | `python reviews/rm40/hgb_diagnosis.py run <n> <R> --procs 2` for (n, R) = (1000, 400), (4000, 400), (16000, 200); then `python reviews/rm40/hgb_diagnosis.py analyze > reviews/rm40/hgb_diagnosis.log` |
# | seeds | replicate `rep` at size `n` draws with `13_000_019 + 7919 * n + rep`, rep = 0 .. R-1. `drdr_sim.py` uses `1_000_003 * (k + 1) + ...`, k <= 8, so no seed repeats. The evaluation sample (20,000 rows) uses seed 777_001; the GLM limit uses 2e6 draws with seed 12345, as in `drdr_sim.py limits` |
# | fits per replicate | one draw, three fits on it: `dr_qwrong` (DRTMLE, outcome GLM on (A, W1)), `tmle_qwrong` (TMLE, the same GLM), `dr_qright` (DRTMLE, outcome GLM on (A, W1, W2)). All use the `g_hgb` tuning of `drdr_sim.py`, in sample (`cross_fit=False`), `g_bounds=(0.01, 0.99)`. DRTMLE: `guard=("Q", "g")`, univariate reduction, cubic-polynomial reduced learners |
# | HGB cache | HGB is deterministic (`random_state=0`, no early stopping), so the three fits share one HGB fit per draw. The script checks that the three initial propensities agree |
# | output | `_hgb_diagnosis/n<n>.csv` and `_hgb_diagnosis/limit.npz` beside this script (not committed); the analysis is `hgb_diagnosis.log` |
# | threads | one thread per process (`OMP_NUM_THREADS=1`) |
#
# Definitions (arm 1; arm 0 replaces g by 1 - g; the ATE is arm 1 minus arm 0):
#
# | symbol | definition |
# | --- | --- |
# | g_n | the HGB propensity of the fit, clipped into (0.01, 0.99) |
# | Qbar | the limit of the wrong outcome GLM on (1, A, W1) (2e6 draws) |
# | Qr0 | `Q̄_{0,r}(w) = E_0[Q̄_0(1, W) - Qbar(1, W) | g_0(W) = g_0(w)]`, by quadrature along the g_0 level set (`drdr_sim.qr0_arm`) |
# | norm, train / fresh | `sqrt(mean[(g_n - g_0)^2])` over the n training rows / over the 20,000-row evaluation sample |
# | QP, train / fresh | `mean[(g_n - g_0)^2 * Qr0 / g_0^2]` over the training rows / the evaluation sample. Sign: the remainder `(g_0 - g_n)/g_n * Qr0` is the linear part `(g_0 - g_n)/g_0 * Qr0`, which the DR equation removes, plus `(g_0 - g_n)^2/(g_0 g_n) * Qr0 ~ +(g_n - g_0)^2 Qr0/g_0^2`, so QP predicts `psi_n - psi_0` with a plus sign |
# | R3, train / fresh | `mean[(Q̄_0 - Q̄_n - Qr0) * (g_0 - g_n)/g_0]`, Q̄_n the fitted wrong GLM (`drdr_sim.py`'s proxy) |
# | standardized | every bias and proxy is divided by the Monte Carlo SD of `psi_n` in its cell; for `dr_qwrong` that SD is the one the rule uses |
#
# Extra diagnostics (labelled X; they decide nothing). With the targeted fits `Q̄*` and `g*` of
# each estimator, the solved equation `P_n[A/g* (Y - Q̄*)] = 0` gives the exact split
# `psi_n - psi_0 = O + W`, `O = P_n[Q̄* + A/g_0 (Y - Q̄*)] - psi_0` (an AIPW with the true g_0),
# `W = P_n[A (1/g* - 1/g_0)(Y - Q̄*)]`. W splits into `Wnoise = P_n[A (1/g* - 1/g_0)(Y - Q̄_0)]`
# and `Wrem = P_n[A (1/g* - 1/g_0)(Q̄_0 - Q̄*)]`. `Wown = P_n[(A - g_0)(1/g* - 1/g_0)(Q̄_0 - Q̄*)]`
# is the part of Wrem that uses each row's own treatment (an empirical-process term `(P_n - P_0)`
# of a g-dependent curve; zero mean when g* is independent of the row's A). `Rfresh` is the
# initial-fit remainder on the evaluation sample: `P_0[(g_0/g_n - 1)(Q̄_0 - Q̄_n)]` for TMLE and
# `P_0[(g_0/g_n - 1)(Q̄_0 - Q̄_n - Qr0)]` for DRTMLE.
# O splits exactly into `P_n[D_0]` (the oracle curve at (Q̄_0, g_0), mean 0) and
# `Oq = -P_n[(A - g_0)/g_0 (Q̄* - Q̄_0)]`, which has a nonzero mean only when Q̄* depends on the
# row's own treatment. `Oqfl = -P_n[(A - g_0)/g_0 (Q̄* - Q̄_n)]` is the part of Oq from the
# targeting step (Q̄_n the initial outcome fit).
#
# Decision rule, fixed before the run (RM40 plan section 5.11):
#
# | outcome | criterion | consequence |
# | --- | --- | --- |
# | A | at n = 4,000, for the `dr_qwrong` ATE: the training-row QP prediction `mean(QP_train)/SD` has the sign of the measured bias/SD `b`, and `abs(pred - b) <= max(0.30 * abs(b), 2 * MCSE)` with `MCSE = sqrt(1/R + Var(QP_train/SD)/R)`; AND the `dr_qright` ATE 99% interval `b +/- 2.576/sqrt(R)` lies inside [-0.25, 0.25] | record in the RM40b PR body: the quadratic proxy explains the HGB bias |
# | B | otherwise | stop and report a defect suspect for a separate row |
#
# The training-row QP is primary because the estimator is in sample and averages over its own
# rows. The fresh-sample QP and the same test at n = 1,000 and 16,000 are reported beside the rule
# and decide nothing. For `dr_qright`, Qr0 is 0 (Q̄_n is consistent), so its QP is 0 by definition.
"""Diagnose the in-sample HGB bias of the RM40 probe (DRTMLE, wrong outcome GLM, HGB propensity)."""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import hashlib
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import drdr_sim as sim  # noqa: E402

OUT = HERE / "_hgb_diagnosis"
SEED_BASE = 13_000_019
EVAL_N = 20_000
EVAL_SEED = 777_001
CELLS = ("dr_qwrong", "tmle_qwrong", "dr_qright")
ESTIMANDS = ("ey[1]", "ey[0]", "ate")
LO, HI = sim.G_BOUNDS


def script_sha() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


# ------------------------------------------------------------------ HGB with a per-draw cache
_CACHE: dict[str, HistGradientBoostingClassifier] = {}
_FITS = {"hgb": 0}


def _hgb() -> HistGradientBoostingClassifier:
    # the exact `g_hgb` tuning of drdr_sim.learners
    return sim.learners("qwrong_ghgb", 1000)[1]


class CachedHGB(BaseEstimator, ClassifierMixin):
    """`drdr_sim`'s HGB; refits only on (X, y) it has not seen. Deterministic, so sharing is exact."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> "CachedHGB":
        # cleverly passes sample weights; they go to HGB unchanged, as in drdr_sim.py, and into the key
        X = np.ascontiguousarray(X, dtype=float)
        y = np.ascontiguousarray(y, dtype=float)
        sw = None if sample_weight is None else np.ascontiguousarray(sample_weight, dtype=float)
        key = hashlib.sha1(X.tobytes() + y.tobytes() + (b"" if sw is None else sw.tobytes())).hexdigest()
        if key not in _CACHE:
            _CACHE.clear()
            _CACHE[key] = _hgb().fit(X, y, sample_weight=sw)
            _FITS["hgb"] += 1
        self.model_ = _CACHE[key]
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.model_.predict_proba(np.asarray(X, dtype=float))


# ------------------------------------------------------------------ process state
_S: dict[str, Any] = {}


def limit_q_wrong() -> np.ndarray:
    path = OUT / "limit.npz"
    if not path.exists():
        OUT.mkdir(exist_ok=True)
        np.savez(path, **sim.limits())
    return np.load(path)["q_wrong"]


def init() -> None:
    warnings.simplefilter("ignore")
    bq = limit_q_wrong()
    ev = sim.draw(EVAL_N, EVAL_SEED)
    w1, w2 = ev.W1.to_numpy(), ev.W2.to_numpy()
    _S.update(
        bq=bq,
        truth=sim.truth(),
        ew1=w1,
        ew2=w2,
        eg0=sim.g0(w1, w2),
        eqr={1.0: sim.qr0_arm(bq, 1.0, w1, w2), 0.0: sim.qr0_arm(bq, 0.0, w1, w2)},
        eq0={1.0: sim.q0(1.0, w1, w2), 0.0: sim.q0(0.0, w1, w2)},
    )


def _arms(g1: np.ndarray, g01: np.ndarray, arm: float) -> tuple[np.ndarray, np.ndarray]:
    return (g1, g01) if arm == 1.0 else (1.0 - g1, 1.0 - g01)


def _combine(per_arm: dict[float, float]) -> dict[str, float]:
    return {"ey[1]": per_arm[1.0], "ey[0]": per_arm[0.0], "ate": per_arm[1.0] - per_arm[0.0]}


def _glm_q(beta: np.ndarray, cols: tuple[int, ...], arm: float, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    design = {0: np.full_like(w1, arm), 1: w1, 2: w2}
    eta = beta[0] + sum(b * design[c] for b, c in zip(beta[1:], cols))
    return sim.expit(eta)


def one(payload: tuple[int, int]) -> list[dict[str, Any]]:
    from cleverly.estimators import DRTMLE, TMLE

    n, rep = payload
    seed = SEED_BASE + 7919 * n + rep
    fits0 = _FITS["hgb"]
    d = sim.draw(n, seed)
    w1, w2, a, y = (d[c].to_numpy() for c in ("W1", "W2", "A", "Y"))
    g01 = sim.g0(w1, w2)
    q0 = {1.0: sim.q0(1.0, w1, w2), 0.0: sim.q0(0.0, w1, w2)}
    bq = _S["bq"]
    qr = {1.0: sim.qr0_arm(bq, 1.0, w1, w2), 0.0: sim.qr0_arm(bq, 0.0, w1, w2)}
    common = dict(
        treatment_learner=CachedHGB(),
        cross_fit=False,
        estimands=("ey", "ate"),
        simultaneous=False,
        g_bounds=sim.G_BOUNDS,
        random_state=0,
    )
    dr_extra = dict(
        reduced_outcome_learner=sim.SplineReg(),
        reduced_treatment_learner=sim.SplineClf(),
        guard=("Q", "g"),
        reduction="univariate",
    )
    specs = {
        "dr_qwrong": (DRTMLE, (0, 1), dr_extra),
        "tmle_qwrong": (TMLE, (0, 1), {}),
        "dr_qright": (DRTMLE, (0, 1, 2), dr_extra),
    }
    rows: list[dict[str, Any]] = []
    g_ref: np.ndarray | None = None
    diag: dict[str, Any] = {}
    for cell, (cls, cols, extra) in specs.items():
        t0 = time.perf_counter()
        try:
            res = (
                cls(outcome_learner=sim.Glm(columns=cols), **common, **extra)
                .fit(d, outcome="Y", treatment="A", covariates=["W1", "W2"])
                .single()
            )
        except Exception as exc:  # noqa: BLE001
            rows.append({"n": n, "rep": rep, "cell": cell, "estimand": "error", "error": repr(exc)[:300]})
            continue
        secs = time.perf_counter() - t0
        r0 = res.repeats[0]
        gn1 = np.clip(np.asarray(r0.nuisance.propensity.arm(1.0), dtype=float), LO, HI)
        if g_ref is None:
            g_ref = gn1
            model = _CACHE[next(iter(_CACHE))]
            eg1 = np.clip(model.predict_proba(np.column_stack([_S["ew1"], _S["ew2"]]))[:, 1], LO, HI)
            diag["norm_train"] = float(np.sqrt(np.mean((gn1 - g01) ** 2)))
            diag["norm_fresh"] = float(np.sqrt(np.mean((eg1 - _S["eg0"]) ** 2)))
            qp_t, qp_f = {}, {}
            for arm in (1.0, 0.0):
                gn, gt = _arms(gn1, g01, arm)
                egn, egt = _arms(eg1, _S["eg0"], arm)
                qp_t[arm] = float(np.mean((gn - gt) ** 2 * qr[arm] / gt**2))
                qp_f[arm] = float(np.mean((egn - egt) ** 2 * _S["eqr"][arm] / egt**2))
            diag["qp_train"], diag["qp_fresh"] = _combine(qp_t), _combine(qp_f)
            diag["eg1"] = eg1
        g_gap = float(np.max(np.abs(gn1 - g_ref)))
        fl = r0.fluctuations["mean"]
        qstar = {arm: np.asarray(fl.targeted.arms[arm], dtype=float) for arm in (1.0, 0.0)}
        if cls is DRTMLE:
            gstar1 = np.asarray(fl.mechanism.propensity, dtype=float).reshape(-1)
            red = fl.reduction
            conv = (bool(red.converged), str(red.exit_reason))
        else:
            gstar1 = gn1
            conv = (bool(fl.converged), "")
        # the fitted outcome GLM, refitted here by the same Newton MLE, for the R3 and Rfresh terms
        beta = sim._logistic(np.column_stack([a, w1, w2])[:, list(cols)], y)
        eg1 = diag["eg1"]
        x: dict[str, dict[float, float]] = {k: {} for k in ("W", "Wnoise", "Wrem", "Wown", "Oq", "Oqfl", "r3t", "r3f", "Rfresh", "solve")}
        for arm in (1.0, 0.0):
            ind = a if arm == 1.0 else 1.0 - a
            gs, gt = _arms(gstar1, g01, arm)
            gn, _ = _arms(gn1, g01, arm)
            dw = 1.0 / gs - 1.0 / gt
            x["W"][arm] = float(np.mean(ind * dw * (y - qstar[arm])))
            x["Wnoise"][arm] = float(np.mean(ind * dw * (y - q0[arm])))
            x["Wrem"][arm] = float(np.mean(ind * dw * (q0[arm] - qstar[arm])))
            x["Wown"][arm] = float(np.mean((ind - gt) * dw * (q0[arm] - qstar[arm])))
            x["solve"][arm] = float(np.mean(ind / gs * (y - qstar[arm])))
            qinit = np.asarray(r0.nuisance.outcome.arms[arm], dtype=float)
            x["Oq"][arm] = float(np.mean(-(ind - gt) / gt * (qstar[arm] - q0[arm])))
            x["Oqfl"][arm] = float(np.mean(-(ind - gt) / gt * (qstar[arm] - qinit)))
            qn = _glm_q(beta, cols, arm, w1, w2)
            x["r3t"][arm] = float(np.mean((q0[arm] - qn - qr[arm]) * (gt - gn) / gt))
            egn, egt = _arms(eg1, _S["eg0"], arm)
            eqn = _glm_q(beta, cols, arm, _S["ew1"], _S["ew2"])
            eresid = _S["eq0"][arm] - eqn
            x["r3f"][arm] = float(np.mean((eresid - _S["eqr"][arm]) * (egt - egn) / egt))
            if cls is DRTMLE:
                eresid = eresid - (_S["eqr"][arm] if cols == (0, 1) else 0.0)
            x["Rfresh"][arm] = float(np.mean((egt / egn - 1.0) * eresid))
        comb = {k: _combine(v) for k, v in x.items()}
        for name in ESTIMANDS:
            est = res.estimates[name]
            lo, hi = est.ci
            truth = _S["truth"][name]
            psi = float(est.psi)
            rows.append(
                {
                    "n": n,
                    "rep": rep,
                    "cell": cell,
                    "estimand": name,
                    "truth": truth,
                    "psi": psi,
                    "se": float(est.std_error),
                    "covered": int(lo <= truth <= hi),
                    "secs": secs,
                    "converged": conv[0],
                    "exit": conv[1],
                    "g_gap": g_gap,
                    "norm_train": diag["norm_train"],
                    "norm_fresh": diag["norm_fresh"],
                    "qp_train": diag["qp_train"][name] if cols == (0, 1) else 0.0,
                    "qp_fresh": diag["qp_fresh"][name] if cols == (0, 1) else 0.0,
                    "r3_train": comb["r3t"][name],
                    "r3_fresh": comb["r3f"][name],
                    "x_O": psi - truth - comb["W"][name],
                    "x_W": comb["W"][name],
                    "x_Wnoise": comb["Wnoise"][name],
                    "x_Wrem": comb["Wrem"][name],
                    "x_Wown": comb["Wown"][name],
                    "x_Oq": comb["Oq"][name],
                    "x_Oqfl": comb["Oqfl"][name],
                    "x_Rfresh": comb["Rfresh"][name],
                    "x_solve": comb["solve"][name],
                    "hgb_fits": _FITS["hgb"] - fits0,
                }
            )
    return rows


def run(n: int, reps: int, procs: int) -> None:
    OUT.mkdir(exist_ok=True)
    limit_q_wrong()
    sha = script_sha()
    t0 = time.time()
    with Pool(procs, initializer=init) as pool:
        rows = [r for chunk in pool.imap(one, [(n, i) for i in range(reps)], chunksize=4) for r in chunk]
    frame = pd.DataFrame(rows)
    frame["sha256"] = sha
    frame["wall"] = time.time() - t0
    frame.to_csv(OUT / f"n{n}.csv", index=False)
    print(f"n={n} R={reps} done in {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ analysis
def analyze() -> None:
    sha = script_sha()
    print(f"sha256 {sha}  reviews/rm40/hgb_diagnosis.py")
    frames = [pd.read_csv(p) for p in sorted(OUT.glob("n*.csv"))]
    frame = pd.concat(frames, ignore_index=True)
    shas = set(frame["sha256"].dropna())
    print(f"run CSVs written by sha256 {sorted(shas)}; match: {shas == {sha}}")
    errors = frame.loc[frame["estimand"] == "error"]
    print(f"errors: {len(errors)}" + ("" if errors.empty else f" first: {errors['error'].iloc[0]}"))
    frame = frame.loc[frame["estimand"] != "error"].copy()
    for n, grp in frame.groupby("n"):
        wall = grp["wall"].iloc[0]
        print(
            f"n={n}: HGB fits per replicate, max = {grp['hgb_fits'].max():.0f} "
            f"(cached across the 3 fits); max |g_n gap| between fits = {grp['g_gap'].max():.2e}; "
            f"max |P_n[A/g* (Y - Q*)]| = {grp['x_solve'].abs().max():.1e}; wall {wall:.0f}s"
        )
    print()
    print("Per cell, n and estimand. Every bias and proxy is divided by the Monte Carlo SD of psi in the cell.")
    print("mcse of bias/SD = 1/sqrt(R). norm = sqrt(mean (g_n - g_0)^2). QP, R3: see the header.")
    hdr = (
        f"{'cell':12s} {'n':>6s} {'est':6s} {'R':>4s} {'nonconv':>7s} {'bias/SD':>8s} {'mcse':>6s} {'SEratio':>7s} "
        f"{'cover':>6s} {'normTr':>7s} {'normFr':>7s} {'QPtr/SD':>8s} {'QPfr/SD':>8s} {'R3tr/SD':>8s} {'R3fr/SD':>8s}"
    )
    xhdr = f"{'cell':12s} {'n':>6s} {'est':6s} {'X:O/SD':>7s} {'W/SD':>7s} {'Wnoise':>7s} {'Wrem':>7s} {'Wown':>7s} {'Rfresh':>7s} {'Oq':>7s} {'Oqfl':>7s}"
    lines, xlines = [hdr], [xhdr]
    stats: dict[tuple[str, int, str], dict[str, float]] = {}
    for (cell, n, name), g in frame.groupby(["cell", "n", "estimand"], sort=True):
        r = len(g)
        sd = float(g["psi"].std(ddof=1))
        b = float((g["psi"] - g["truth"]).mean()) / sd
        st = {
            "R": r,
            "b": b,
            "mcse": 1.0 / np.sqrt(r),
            "qp_tr": float(g["qp_train"].mean()) / sd,
            "qp_tr_sd": float((g["qp_train"] / sd).std(ddof=1)),
            "qp_fr": float(g["qp_fresh"].mean()) / sd,
            "qp_fr_sd": float((g["qp_fresh"] / sd).std(ddof=1)),
        }
        stats[(cell, n, name)] = st
        nonconv = int((~g["converged"].astype(bool)).sum())
        lines.append(
            f"{cell:12s} {n:6d} {name:6s} {r:4d} {nonconv:7d} {b:+8.3f} {st['mcse']:6.3f} "
            f"{float(g['se'].mean()) / sd:7.3f} {float(g['covered'].mean()):6.3f} "
            f"{float(g['norm_train'].mean()):7.4f} {float(g['norm_fresh'].mean()):7.4f} "
            f"{st['qp_tr']:+8.3f} {st['qp_fr']:+8.3f} "
            f"{float(g['r3_train'].mean()) / sd:+8.3f} {float(g['r3_fresh'].mean()) / sd:+8.3f}"
        )
        xlines.append(
            f"{cell:12s} {n:6d} {name:6s} "
            + " ".join(f"{float(g[c].mean()) / sd:+7.3f}" for c in ("x_O", "x_W", "x_Wnoise", "x_Wrem", "x_Wown", "x_Rfresh", "x_Oq", "x_Oqfl"))
        )
    print("\n".join(lines))
    print()
    print("X: extra diagnostics (decide nothing). psi_n - psi_0 = O + W exactly; W = Wnoise + Wrem; Wown is the")
    print("own-row part of Wrem; Rfresh is the initial-fit remainder on the evaluation sample. O = P_n[D_0] + Oq,")
    print("P_n[D_0] the oracle curve (mean exactly 0); Oqfl is the part of Oq from the targeting step Q* - Q_n. All / SD.")
    print("\n".join(xlines))
    print()
    print("Decision rule (dr_qwrong ATE vs training-row QP; dr_qright ATE 99% interval inside +/-0.25):")
    verdict = None
    for n in sorted(frame["n"].unique()):
        if ("dr_qwrong", n, "ate") not in stats:
            continue
        s = stats[("dr_qwrong", n, "ate")]
        c = stats[("dr_qright", n, "ate")]
        out = []
        for tag in ("tr", "fr"):
            pred = s[f"qp_{tag}"]
            mcse = np.sqrt(1.0 / s["R"] + s[f"qp_{tag}_sd"] ** 2 / s["R"])
            tol = max(0.30 * abs(s["b"]), 2.0 * mcse)
            ok = np.sign(pred) == np.sign(s["b"]) and abs(pred - s["b"]) <= tol
            out.append((tag, pred, tol, ok))
        clo, chi = c["b"] - 2.576 * c["mcse"], c["b"] + 2.576 * c["mcse"]
        right_ok = -0.25 <= clo and chi <= 0.25
        a_here = out[0][3] and right_ok
        print(
            f"n={n:6d}: measured {s['b']:+.3f}; QP train pred {out[0][1]:+.3f} (tol {out[0][2]:.3f}, pass {out[0][3]}); "
            f"QP fresh pred {out[1][1]:+.3f} (pass {out[1][3]}); dr_qright 99% [{clo:+.3f}, {chi:+.3f}] inside: {right_ok}; "
            f"A-criterion: {a_here}" + ("  <- decides" if n == 4000 else "  (reported only)")
        )
        if n == 4000:
            verdict = "A" if a_here else "B"
    print(f"outcome: {verdict}")


def main() -> None:
    if sys.argv[1] == "run":
        procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 2
        run(int(sys.argv[2]), int(sys.argv[3]), procs)
    elif sys.argv[1] == "time":
        init()
        for n in (1000, 4000, 16000):
            t0 = time.time()
            rows = one((n, 9999))
            print(n, f"{time.time() - t0:.2f}s", [(r["cell"], r["estimand"], round(r["psi"] - r["truth"], 4), r["g_gap"]) for r in rows if r["estimand"] == "ate"], flush=True)
    else:
        analyze()


if __name__ == "__main__":
    main()
