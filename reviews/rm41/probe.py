# A probe, not a registered study. RM41, round 1: is the in-sample HGB bias a code defect?
#
# Law L, learners and reduced learners come from `reviews/rm40/drdr_sim.py` (wrong outcome GLM on
# (A, W1), HGB propensity at the `g_hgb` tuning, cubic-polynomial reduced learners,
# `g_bounds=(0.01, 0.99)`, `guard=("Q", "g")`, univariate reduction).
#
# | cell | what it fits |
# | --- | --- |
# | `dr_in` | `cleverly.DRTMLE`, in sample (`cross_fit=False`) |
# | `dr_in_bk` | the same, `update_order="benkeser"` (the published six-step order) |
# | `tmle_in` | `cleverly.TMLE`, in sample |
# | `own_in` | `own_drtmle` below, on `dr_in`'s initial `Q_n` and `g_n` |
# | `dr_cf`, `tmle_cf` | the same two estimators, `cross_fit=True` (the shipped 10-fold split) |
# | `own_ext` | `own_drtmle` with `g_n` from an HGB fitted on an independent draw of the same size |
# | `own_ext_or` | `own_drtmle` at `own_ext`'s `g_n`, with the reduced regression `Q_r` replaced by its population version at the estimated index (an oracle `Q_r`) |
# | `own_ext_g0` | `own_drtmle` at `own_ext`'s `g_n`, with `Q_r` regressed on the true `g_0` instead of `g_n` |
#
# `own_drtmle` is an independent transcription of the recursion of Benkeser et al. (2017,
# Section 3.2) for one arm: an initial outcome step along `1/g`, then rounds of (fit `Q_r` on `g`,
# tilt `g` along `Q_r/g`, fit `g_{r,1}` and `g_{r,2}` on `Q̄`, tilt `Q̄` along `(1/g, g_{r,2}/g_{r,1})`)
# until the three empirical means are below 1e-9, at most 500 rounds, the mechanism kept in `g_bounds` after each tilt. It shares no code with
# `cleverly`.
#
# Proxies, on a 200,000-row evaluation draw, for each fitted mechanism `g_n` (arm a):
#
# | proxy | definition |
# | --- | --- |
# | `lin` | `P_0[(g_0 - g_n)/g_n · r]`, `r = Q̄_0(a, W) - Q̄_n(a, W)`: the remainder of plain TMLE |
# | `dr` | `P_0[(g_0 - g_n)/g_n · (r - Q_r^lim(g_n))]`, `Q_r^lim` the cubic least-squares projection of `r` on `g_n` with weights `g_0` (the population target of the `A = a` regression): the remainder that DR-TMLE leaves at the estimated index |
#
# Seeds: replicate `rep` at size `n` draws with `41_000_011 + 7919 * n + rep`; the independent
# mechanism draw adds 500_000; the evaluation draw uses 41_777_001.
#
# Usage: `python reviews/rm41/probe.py run <n> <R> --procs 4`, then
# `python reviews/rm41/probe.py analyze > reviews/rm41/probe.log`.
"""RM41 round-1 probe: separate a code defect from a property of DR-TMLE under an HGB mechanism."""

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
from scipy.special import expit, logit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rm40"))

import drdr_sim as sim  # noqa: E402

OUT = HERE / "_probe"
SEED_BASE = 41_000_011
EXT_OFFSET = 500_000
EVAL_N = 200_000
EVAL_SEED = 41_777_001
LO, HI = sim.G_BOUNDS
ESTIMANDS = ("ey[1]", "ey[0]", "ate")


def script_sha() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


# ------------------------------------------------------------------ the independent DR-TMLE
def _poly(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones_like(x), x, x**2, x**3])


def _ls(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    b = _poly(x)
    r = np.ones(len(y)) if w is None else np.sqrt(w)
    return np.linalg.lstsq(b * r[:, None], y * r, rcond=None)[0]


def _logistic_poly(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    b = _poly(x)
    beta = np.zeros(4)
    beta[0] = logit(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    for _ in range(100):
        p = expit(b @ beta)
        step = np.linalg.solve((b * (p * (1 - p))[:, None]).T @ b + 1e-8 * np.eye(4), b.T @ (y - p))
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta


def _tilt(y: np.ndarray, offset: np.ndarray, cov: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
    """Logistic MLE of the coefficients of `cov` with a fixed offset (no intercept); returns eps."""
    sel = slice(None) if rows is None else rows
    yy, oo, cc = y[sel], offset[sel], cov[sel]
    eps = np.zeros(cc.shape[1])
    for _ in range(200):
        p = expit(oo + cc @ eps)
        h = (cc * (p * (1 - p))[:, None]).T @ cc + 1e-12 * np.eye(cc.shape[1])
        step = np.linalg.solve(h, cc.T @ (yy - p))
        eps += step
        if np.max(np.abs(step)) < 1e-12:
            break
    return eps


def own_drtmle(
    y: np.ndarray,
    ind: np.ndarray,
    qn: np.ndarray,
    gn: np.ndarray,
    qr_index: np.ndarray | None = None,
    qr_oracle: Any = None,
) -> tuple[float, float, int]:
    """One arm. `ind` is 1{A = a}; `qn` is Q̄_n(a, W); `gn` is g_n(a|W). Returns (psi, max score, rounds).

    `qr_index` replaces the regressor of Q_r (default: the current g); `qr_oracle(g)` replaces the
    fitted Q_r by a given function of the current g.
    """
    q = np.clip(qn, 1e-6, 1 - 1e-6)
    g = np.clip(gn, LO, HI)
    treated = ind == 1
    eps = _tilt(y, logit(q), (1.0 / g)[:, None], treated)
    q = expit(logit(q) + eps[0] / g)
    worst = np.inf
    rounds = 0
    for rounds in range(1, 501):
        idx = g if qr_index is None else qr_index
        if qr_oracle is not None:
            qr = qr_oracle(g)
        else:
            qr = _poly(idx) @ _ls(idx[treated], (y - q)[treated])
        e = _tilt(ind, logit(g), (qr / g)[:, None])
        g = np.clip(expit(logit(g) + e[0] * qr / g), LO, HI)
        gr1 = np.clip(expit(_poly(q) @ _logistic_poly(q, ind)), LO, 1.0)
        gr2 = _poly(q) @ _ls(q, (ind - g) / g)
        h = np.column_stack([1.0 / g, gr2 / gr1])
        eps = _tilt(y, logit(q), h, treated)
        q = np.clip(expit(logit(q) + h @ eps), 1e-9, 1 - 1e-9)
        if qr_oracle is not None:
            qr = qr_oracle(g)
        else:
            idx = g if qr_index is None else qr_index
            qr = _poly(idx) @ _ls(idx[treated], (y - q)[treated])
        s = [
            np.mean(ind / g * (y - q)),
            np.mean(ind * gr2 / gr1 * (y - q)),
            np.mean(qr / g * (ind - g)),
        ]
        worst = float(np.max(np.abs(s)))
        if worst < 1e-9:
            break
    return float(np.mean(q)), worst, rounds


# ------------------------------------------------------------------ process state
_S: dict[str, Any] = {}


def init() -> None:
    warnings.simplefilter("ignore")
    ev = sim.draw(EVAL_N, EVAL_SEED)
    w1, w2 = ev.W1.to_numpy(), ev.W2.to_numpy()
    _S.update(
        truth=sim.truth(),
        w=np.column_stack([w1, w2]),
        g0=sim.g0(w1, w2),
        q0={1.0: sim.q0(1.0, w1, w2), 0.0: sim.q0(0.0, w1, w2)},
    )


def _hgb() -> Any:
    return sim.learners("qwrong_ghgb", 1000)[1]


def _glm_q(beta: np.ndarray, arm: float, w1: np.ndarray) -> np.ndarray:
    return expit(beta[0] + beta[1] * arm + beta[2] * w1)


def proxies(model: Any, beta: np.ndarray) -> dict[str, dict[float, float]]:
    """`lin` and `dr` on the evaluation draw for a fitted HGB and outcome GLM coefficients."""
    w = _S["w"]
    g1 = np.clip(model.predict_proba(w)[:, 1], LO, HI)
    out: dict[str, dict[float, float]] = {"lin": {}, "dr": {}}
    for arm in (1.0, 0.0):
        gn = g1 if arm == 1.0 else 1.0 - g1
        gt = _S["g0"] if arm == 1.0 else 1.0 - _S["g0"]
        r = _S["q0"][arm] - _glm_q(beta, arm, w[:, 0])
        out["lin"][arm] = float(np.mean((gt - gn) / gn * r))
        qr = _poly(gn) @ _ls(gn, r, gt)
        out["dr"][arm] = float(np.mean((gt - gn) / gn * (r - qr)))
    return out


def oracle_qr(model: Any, beta: np.ndarray, arm: float) -> Any:
    """The population Q_r at the estimated index: E_0[r | g_n] among A = a, as a cubic in g_n."""
    w = _S["w"]
    g1 = np.clip(model.predict_proba(w)[:, 1], LO, HI)
    gn = g1 if arm == 1.0 else 1.0 - g1
    gt = _S["g0"] if arm == 1.0 else 1.0 - _S["g0"]
    coef = _ls(gn, _S["q0"][arm] - _glm_q(beta, arm, w[:, 0]), gt)
    return lambda g: _poly(g) @ coef


def _combine(per_arm: dict[float, float]) -> dict[str, float]:
    return {"ey[1]": per_arm[1.0], "ey[0]": per_arm[0.0], "ate": per_arm[1.0] - per_arm[0.0]}


def one(payload: tuple[int, int]) -> list[dict[str, Any]]:
    from cleverly.estimators import DRTMLE, TMLE

    n, rep = payload
    seed = SEED_BASE + 7919 * n + rep
    d = sim.draw(n, seed)
    w1, w2, a, y = (d[c].to_numpy() for c in ("W1", "W2", "A", "Y"))
    g01 = sim.g0(w1, w2)
    beta = sim._logistic(np.column_stack([a, w1]), y)
    rows: list[dict[str, Any]] = []

    def record(cell: str, per_arm: dict[float, float], extra: dict[str, Any]) -> None:
        for name, psi in _combine(per_arm).items():
            rows.append({"n": n, "rep": rep, "cell": cell, "estimand": name, "psi": psi,
                         "truth": _S["truth"][name], **extra})

    common = dict(treatment_learner=_hgb(), outcome_learner=sim.Glm(columns=(0, 1)), estimands=("ey", "ate"),
                  simultaneous=False, g_bounds=sim.G_BOUNDS, random_state=0)
    dr_extra = dict(reduced_outcome_learner=sim.SplineReg(), reduced_treatment_learner=sim.SplineClf(),
                    guard=("Q", "g"), reduction="univariate")
    gn_in: np.ndarray | None = None
    qn_in: dict[float, np.ndarray] = {}
    for cell, cls, cf, extra in (("dr_in", DRTMLE, False, dr_extra), ("tmle_in", TMLE, False, {}),
                                 ("dr_in_bk", DRTMLE, False, {**dr_extra, "update_order": "benkeser"}),
                                 ("dr_cf", DRTMLE, True, dr_extra), ("tmle_cf", TMLE, True, {})):
        try:
            res = cls(cross_fit=cf, **common, **extra).fit(
                d, outcome="Y", treatment="A", covariates=["W1", "W2"]).single()
        except Exception as exc:  # noqa: BLE001
            rows.append({"n": n, "rep": rep, "cell": cell, "estimand": "error", "error": repr(exc)[:300]})
            continue
        r0 = res.repeats[0]
        if cell == "dr_in":
            gn_in = np.clip(np.asarray(r0.nuisance.propensity.arm(1.0), dtype=float), LO, HI)
            qn_in = {arm: np.asarray(r0.nuisance.outcome.arms[arm], dtype=float) for arm in (1.0, 0.0)}
        for name in ESTIMANDS:
            est = res.estimates[name]
            lo, hi = est.ci
            rows.append({"n": n, "rep": rep, "cell": cell, "estimand": name, "psi": float(est.psi),
                         "truth": _S["truth"][name], "se": float(est.std_error),
                         "covered": int(lo <= _S["truth"][name] <= hi)})

    # in-sample HGB, refitted here (deterministic), for its proxies
    model_in = _hgb().fit(np.column_stack([w1, w2]), a)
    px_in = {k: _combine(v) for k, v in proxies(model_in, beta).items()}
    gtrain = np.clip(model_in.predict_proba(np.column_stack([w1, w2]))[:, 1], LO, HI)
    gap = float(np.max(np.abs(gtrain - gn_in))) if gn_in is not None else np.nan
    if gn_in is not None:
        per, worst = {}, 0.0
        for arm in (1.0, 0.0):
            ind = a if arm == 1.0 else 1.0 - a
            g = gn_in if arm == 1.0 else 1.0 - gn_in
            per[arm], s, _ = own_drtmle(y, ind, qn_in[arm], g)
            worst = max(worst, s)
        record("own_in", per, {"score": worst})

    # an independent mechanism: HGB on a separate draw of the same law and size
    dx = sim.draw(n, seed + EXT_OFFSET)
    model_ext = _hgb().fit(dx[["W1", "W2"]].to_numpy(), dx["A"].to_numpy())
    g_ext = np.clip(model_ext.predict_proba(np.column_stack([w1, w2]))[:, 1], LO, HI)
    px_ext = {k: _combine(v) for k, v in proxies(model_ext, beta).items()}
    for cell in ("own_ext", "own_ext_or", "own_ext_g0"):
        per, worst = {}, 0.0
        for arm in (1.0, 0.0):
            ind = a if arm == 1.0 else 1.0 - a
            g = g_ext if arm == 1.0 else 1.0 - g_ext
            qn = _glm_q(beta, arm, w1)
            kw: dict[str, Any] = {}
            if cell == "own_ext_or":
                kw["qr_oracle"] = oracle_qr(model_ext, beta, arm)
            if cell == "own_ext_g0":
                kw["qr_index"] = g01 if arm == 1.0 else 1.0 - g01
            per[arm], s, _ = own_drtmle(y, ind, qn, g, **kw)
            worst = max(worst, s)
        record(cell, per, {"score": worst})

    for r in rows:
        if r["estimand"] == "error":
            continue
        r["gap"] = gap
        for k in ("lin", "dr"):
            r[f"in_{k}"] = px_in[k][r["estimand"]]
            r[f"ext_{k}"] = px_ext[k][r["estimand"]]
    return rows


def run(n: int, reps: int, procs: int) -> None:
    OUT.mkdir(exist_ok=True)
    sha = script_sha()
    t0 = time.time()
    with Pool(procs, initializer=init) as pool:
        rows = [r for chunk in pool.imap(one, [(n, i) for i in range(reps)], chunksize=2) for r in chunk]
    frame = pd.DataFrame(rows)
    frame["sha256"] = sha
    frame["wall"] = time.time() - t0
    frame.to_csv(OUT / f"n{n}.csv", index=False)
    print(f"n={n} R={reps} done in {time.time() - t0:.0f}s", flush=True)


def analyze() -> None:
    sha = script_sha()
    print(f"sha256 {sha}  reviews/rm41/probe.py")
    frame = pd.concat([pd.read_csv(p) for p in sorted(OUT.glob("n*.csv"))], ignore_index=True)
    print(f"run CSVs written by this script: {set(frame['sha256'].dropna()) == {sha}}")
    errors = frame.loc[frame["estimand"] == "error"]
    print(f"errors: {len(errors)}" + ("" if errors.empty else f"; first: {errors['error'].iloc[0]}"))
    frame = frame.loc[frame["estimand"] != "error"].copy()
    print(f"max |g_n gap| between cleverly's in-sample mechanism and the refit: {frame['gap'].max():.2e}")
    if "score" in frame:
        print(f"own_drtmle: worst final score {frame['score'].max():.1e}")
    print()
    print("bias/SD: mean(psi - truth) / SD(psi) in the cell; mcse 1/sqrt(R). Proxies / the same SD.")
    print(f"{'cell':11s} {'n':>6s} {'est':6s} {'R':>4s} {'bias/SD':>8s} {'mcse':>6s} {'cover':>6s} "
          f"{'in_lin':>7s} {'in_dr':>7s} {'ext_lin':>8s} {'ext_dr':>7s}")
    for (cell, n, name), g in frame.groupby(["cell", "n", "estimand"], sort=True):
        r = len(g)
        sd = float(g["psi"].std(ddof=1))
        b = float((g["psi"] - g["truth"]).mean()) / sd
        cov = float(g["covered"].mean()) if "covered" in g and g["covered"].notna().any() else np.nan
        print(f"{cell:11s} {n:6d} {name:6s} {r:4d} {b:+8.3f} {1 / np.sqrt(r):6.3f} {cov:6.3f} "
              + " ".join(f"{float(g[c].mean()) / sd:+7.3f}" for c in ("in_lin", "in_dr", "ext_lin", "ext_dr")))


def main() -> None:
    if sys.argv[1] == "run":
        procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 4
        run(int(sys.argv[2]), int(sys.argv[3]), procs)
    elif sys.argv[1] == "time":
        init()
        t0 = time.time()
        rows = one((int(sys.argv[2]), 9999))
        print(f"{time.time() - t0:.1f}s")
        for r in rows:
            if r.get("estimand") in ("ate", "error"):
                print(r["cell"], r.get("psi", 0) - r.get("truth", 0), r.get("error", ""))
    else:
        analyze()


if __name__ == "__main__":
    main()
