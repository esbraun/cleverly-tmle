# A probe, not a registered study. RM41: is the DR-TMLE bias under an HGB mechanism a code defect?
#
# Law L, learners and reduced learners come from `reviews/rm40/drdr_sim.py`: a wrong outcome GLM
# on (A, W1), an HGB propensity at the `g_hgb` tuning, cubic-polynomial reduced learners,
# `g_bounds=(0.01, 0.99)`, `guard=("Q", "g")` and the univariate reduction.
#
# | cell | what it fits |
# | --- | --- |
# | `dr_in` | `cleverly.DRTMLE`, in sample (`cross_fit=False`) |
# | `dr_in_bk` | the same, `update_order="benkeser"` (the published six-step order) |
# | `tmle_in` | `cleverly.TMLE`, in sample |
# | `dr_cf`, `tmle_cf` | `DRTMLE` and `TMLE` with `cross_fit=True` (the shipped 10-fold split) |
# | `own_in` | `own_drtmle` below, at `dr_in`'s initial `Q̄_n` and `g_n` |
# | `own_in_tmle` | `own_in` without either correction: plain TMLE |
# | `own_in_gonly` | `own_in` with the mechanism tilt (equation 9) and without equation 10 |
# | `own_in_qonly` | `own_in` with equation 10 and without the mechanism tilt |
# | `own_in_g0` | `own_in` with `Q_r` regressed on the true `g_0` instead of `g_n` |
# | `own_ext` | `own_drtmle` with `g_n` from an HGB fitted on an independent draw of the same size |
# | `own_ext_tmle` | `own_ext` without either correction |
# | `own_ext_or` | `own_ext` with `Q_r` replaced by its population version at the estimated index: |
# | | the weighted cubic least-squares fit of `Q̄_0 - Q̄_n` on `g_n` over the evaluation draw |
# | `own_ext_g0` | `own_ext` with `Q_r` regressed on the true `g_0` instead of `g_n` |
# | `dr_ext`, `dr_ext_bk` | `cleverly.DRTMLE` in sample at `own_ext`'s mechanism (a prefit learner), both orders |
# | `pop_ext`, `pop_ext_g0`, `pop_ext_tmle` | `population_drtmle`: the limit of `own_ext`, `own_ext_g0` and |
# | | `own_ext_tmle` at that draw's `g_n` and `Q̄_n`, on the first 60,000 evaluation rows |
#
# `own_drtmle` is an independent solver of equations (8) to (10) of Benkeser et al. (2017) for one
# arm. It shares no code with `cleverly`, and its route differs from both of `cleverly`'s orders:
# after one initial outcome step along `1/g`, each round fits `Q_r` on `g`, tilts `g` along `Q_r/g`,
# fits `g_{r,1}` and `g_{r,2}` on `Q̄`, and tilts `Q̄` along `(1/g, g_{r,2}/g_{r,1})` in one joint
# step. It tilts each arm's mechanism separately, keeps it in `g_bounds` after each tilt, bounds
# `g_{r,1}` below only, and has no closing pass. It stops when the three empirical means are below
# 1e-9, or after 500 rounds. Its fits count only below `SOLVED`; `analyze` also prints a robust
# reading over all of them. `population_drtmle` runs the same route with `Y` replaced by
# `Q̄_0(a, W)` and `1{A = a}` by `g_0(a|W)`, so it gives the limit at a fixed `g_n` and `Q̄_n` (at most 200 rounds).
#
# Weight means: for the in-sample HGB, `mean(1{A=a}/g_n - 1)` on its own training rows and on a
# 200,000-row evaluation draw. Both are zero in mean at `g_0`.
#
# Seeds: replicate `rep` at size `n` draws with `41_000_011 + 7919 * n + rep`; the independent
# mechanism draw adds 500_000; the evaluation draw uses 41_777_001.
#
# Usage: `python reviews/rm41/probe.py run <n> 200 --procs 4` for n = 1000, 4000 and 16000, then
# `python reviews/rm41/probe.py analyze > reviews/rm41/probe.log`.
# The run CSVs go to `reviews/rm41/_probe/` and are not committed.
"""RM41 probe: separate a code defect from a property of DR-TMLE under an HGB mechanism."""

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
from sklearn.base import BaseEstimator, ClassifierMixin

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
#: an `own_drtmle` fit counts only when every score it solves is below this
SOLVED = 1e-8
#: evaluation rows used by the population-limit cells
POP_N = 60_000


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
    corrections: tuple[str, ...] = ("g", "Q"),
) -> tuple[float, float, int]:
    """One arm. `ind` is 1{A = a}; `qn` is Q̄_n(a, W); `gn` is g_n(a|W). Returns (psi, max score, rounds).

    `qr_index` replaces the regressor of Q_r (default: the current g); `qr_oracle(g)` replaces the
    fitted Q_r by a given function of the current g. `corrections` drops the mechanism tilt
    (`"g"`, equation 9) or the second outcome covariate (`"Q"`, equation 10) when absent.
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
        if "g" in corrections:
            e = _tilt(ind, logit(g), (qr / g)[:, None])
            g = np.clip(expit(logit(g) + e[0] * qr / g), LO, HI)
        gr1 = np.clip(expit(_poly(q) @ _logistic_poly(q, ind)), LO, 1.0)
        gr2 = _poly(q) @ _ls(q, (ind - g) / g)
        h = np.column_stack([1.0 / g, gr2 / gr1]) if "Q" in corrections else (1.0 / g)[:, None]
        eps = _tilt(y, logit(q), h, treated)
        q = np.clip(expit(logit(q) + h @ eps), 1e-9, 1 - 1e-9)
        if qr_oracle is not None:
            qr = qr_oracle(g)
        else:
            idx = g if qr_index is None else qr_index
            qr = _poly(idx) @ _ls(idx[treated], (y - q)[treated])
        s = [np.mean(ind / g * (y - q))]
        if "Q" in corrections:
            s.append(np.mean(ind * gr2 / gr1 * (y - q)))
        if "g" in corrections:
            s.append(np.mean(qr / g * (ind - g)))
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
        a=ev.A.to_numpy(),
        g0=sim.g0(w1, w2),
        q0={1.0: sim.q0(1.0, w1, w2), 0.0: sim.q0(0.0, w1, w2)},
    )


def _hgb() -> Any:
    return sim.learners("qwrong_ghgb", 1000)[1]


def _glm_q(beta: np.ndarray, arm: float, w1: np.ndarray) -> np.ndarray:
    return expit(beta[0] + beta[1] * arm + beta[2] * w1)


def weight_means(model: Any, w: np.ndarray, a: np.ndarray) -> dict[str, float]:
    """`mean(1{A = a}/g_n - 1)` per arm, on the given rows (zero in mean at the truth)."""
    g1 = np.clip(model.predict_proba(w)[:, 1], LO, HI)
    return {
        "w1": float(np.mean(a / g1 - 1.0)),
        "w0": float(np.mean((1.0 - a) / (1.0 - g1) - 1.0)),
    }


def oracle_qr(model: Any, beta: np.ndarray, arm: float) -> Any:
    """The population Q_r at the estimated index: E_0[r | g_n] among A = a, as a cubic in g_n."""
    w = _S["w"]
    g1 = np.clip(model.predict_proba(w)[:, 1], LO, HI)
    gn = g1 if arm == 1.0 else 1.0 - g1
    gt = _S["g0"] if arm == 1.0 else 1.0 - _S["g0"]
    coef = _ls(gn, _S["q0"][arm] - _glm_q(beta, arm, w[:, 0]), gt)
    return lambda g: _poly(g) @ coef


_PREFIT: dict[str, Any] = {}


class PrefitMechanism(BaseEstimator, ClassifierMixin):
    """A treatment learner whose `fit` keeps the HGB in `_PREFIT["model"]`, fitted elsewhere."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> "PrefitMechanism":
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return _PREFIT["model"].predict_proba(np.asarray(X, dtype=float))


def _wtilt(y: np.ndarray, offset: np.ndarray, cov: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted logistic MLE of the coefficients of `cov` with a fixed offset; `y` may be fractional."""
    eps = np.zeros(cov.shape[1])
    for _ in range(200):
        p = expit(offset + cov @ eps)
        h = (cov * (w * p * (1 - p))[:, None]).T @ cov + 1e-12 * np.eye(cov.shape[1])
        step = np.linalg.solve(h, cov.T @ (w * (y - p)))
        eps += step
        if np.max(np.abs(step)) < 1e-12:
            break
    return eps


def population_drtmle(
    q0a: np.ndarray,
    g0a: np.ndarray,
    qn: np.ndarray,
    gn: np.ndarray,
    qr_index: np.ndarray | None = None,
    corrections: tuple[str, ...] = ("g", "Q"),
) -> tuple[float, float]:
    """`own_drtmle`'s population limit at fixed `Q̄_n` and `g_n` over evaluation rows W.

    Every score and reduced regression replaces `Y` by `Q̄_0(a, W)` and `1{A = a}` by `g_0(a|W)`, so
    no row sees its own treatment or outcome. Returns (psi - psi_0 on these rows, worst score).
    """
    q = np.clip(qn, 1e-6, 1 - 1e-6)
    g = np.clip(gn, LO, HI)
    e = _wtilt(q0a, logit(q), (1 / g)[:, None], g0a)
    q = expit(logit(q) + e[0] / g)
    worst = np.inf
    for _ in range(200):
        idx = g if qr_index is None else qr_index
        qr = _poly(idx) @ _ls(idx, q0a - q, g0a)
        if "g" in corrections:
            e = _wtilt(g0a, logit(g), (qr / g)[:, None], np.ones_like(g))
            g = np.clip(expit(logit(g) + e[0] * qr / g), LO, HI)
        gr1 = np.clip(expit(_poly(q) @ _logistic_poly(q, g0a)), LO, 1.0)
        gr2 = _poly(q) @ _ls(q, (g0a - g) / g)
        h = np.column_stack([1 / g, gr2 / gr1]) if "Q" in corrections else (1 / g)[:, None]
        e = _wtilt(q0a, logit(q), h, g0a)
        q = np.clip(expit(logit(q) + h @ e), 1e-9, 1 - 1e-9)
        idx = g if qr_index is None else qr_index
        qr = _poly(idx) @ _ls(idx, q0a - q, g0a)
        sc = [np.mean(g0a / g * (q0a - q))]
        if "Q" in corrections:
            sc.append(np.mean(g0a * gr2 / gr1 * (q0a - q)))
        if "g" in corrections:
            sc.append(np.mean(qr / g * (g0a - g)))
        worst = float(np.max(np.abs(sc)))
        if worst < 1e-9:
            break
    return float(np.mean(q) - np.mean(q0a)), worst


def _combine(per_arm: dict[float, float]) -> dict[str, float]:
    return {"ey[1]": per_arm[1.0], "ey[0]": per_arm[0.0], "ate": per_arm[1.0] - per_arm[0.0]}


#: in-sample `own_drtmle` cells at `dr_in`'s initial fits
IN_VARIANTS: dict[str, dict[str, Any]] = {
    "own_in": {},
    "own_in_tmle": {"corrections": ()},
    "own_in_gonly": {"corrections": ("g",)},
    "own_in_qonly": {"corrections": ("Q",)},
    "own_in_g0": {"qr_index": "g0"},
}


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

    # in-sample HGB, refitted here (deterministic), for its weight means
    model_in = _hgb().fit(np.column_stack([w1, w2]), a)
    wm_in = weight_means(model_in, np.column_stack([w1, w2]), a)
    wm_fresh = weight_means(model_in, _S["w"], _S["a"])
    gtrain = np.clip(model_in.predict_proba(np.column_stack([w1, w2]))[:, 1], LO, HI)
    gap = float(np.max(np.abs(gtrain - gn_in))) if gn_in is not None else np.nan
    if gn_in is not None:
        for cell, kw in IN_VARIANTS.items():
            per, worst = {}, 0.0
            for arm in (1.0, 0.0):
                ind = a if arm == 1.0 else 1.0 - a
                g = gn_in if arm == 1.0 else 1.0 - gn_in
                k = dict(kw)
                if k.get("qr_index") == "g0":
                    k["qr_index"] = g01 if arm == 1.0 else 1.0 - g01
                per[arm], s, _ = own_drtmle(y, ind, qn_in[arm], g, **k)
                worst = max(worst, s)
            record(cell, per, {"score": worst})

    # an independent mechanism: HGB on a separate draw of the same law and size
    dx = sim.draw(n, seed + EXT_OFFSET)
    model_ext = _hgb().fit(dx[["W1", "W2"]].to_numpy(), dx["A"].to_numpy())
    g_ext = np.clip(model_ext.predict_proba(np.column_stack([w1, w2]))[:, 1], LO, HI)
    for cell in ("own_ext", "own_ext_or", "own_ext_g0", "own_ext_tmle"):
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
            if cell == "own_ext_tmle":
                kw["corrections"] = ()
            per[arm], s, _ = own_drtmle(y, ind, qn, g, **kw)
            worst = max(worst, s)
        record(cell, per, {"score": worst})

    # cleverly at the same independent mechanism: paired with own_ext
    _PREFIT["model"] = model_ext
    for cell, order in (("dr_ext", "drtmle"), ("dr_ext_bk", "benkeser")):
        try:
            res = DRTMLE(
                cross_fit=False, **{**common, "treatment_learner": PrefitMechanism()}, **dr_extra,
                update_order=order,
            ).fit(d, outcome="Y", treatment="A", covariates=["W1", "W2"]).single()
        except Exception as exc:  # noqa: BLE001
            rows.append({"n": n, "rep": rep, "cell": cell, "estimand": "error", "error": repr(exc)[:300]})
            continue
        for name in ESTIMANDS:
            rows.append({"n": n, "rep": rep, "cell": cell, "estimand": name,
                         "psi": float(res.estimates[name].psi), "truth": _S["truth"][name]})

    # the population limit at the independent mechanism and this draw's outcome GLM
    w_ev = _S["w"][:POP_N]
    g_ev = np.clip(model_ext.predict_proba(w_ev)[:, 1], LO, HI)
    g0_ev = _S["g0"][:POP_N]
    for cell, kw in (("pop_ext", {}), ("pop_ext_g0", {"qr_index": "g0"}), ("pop_ext_tmle", {"corrections": ()})):
        per, worst = {}, 0.0
        for arm in (1.0, 0.0):
            g0a = g0_ev if arm == 1.0 else 1.0 - g0_ev
            k = dict(kw)
            if k.get("qr_index") == "g0":
                k["qr_index"] = g0a
            b, s = population_drtmle(
                _S["q0"][arm][:POP_N], g0a, _glm_q(beta, arm, w_ev[:, 0]),
                g_ev if arm == 1.0 else 1.0 - g_ev, **k,
            )
            per[arm] = b + (_S["truth"]["ey[1]"] if arm == 1.0 else _S["truth"]["ey[0]"])
            worst = max(worst, s)
        record(cell, per, {"score": worst})

    for r in rows:
        if r["estimand"] == "error":
            continue
        r["gap"] = gap
        for arm in ("w1", "w0"):
            r[f"{arm}_train"] = wm_in[arm]
            r[f"{arm}_fresh"] = wm_fresh[arm]
    return rows


def _head() -> str:
    import subprocess

    import cleverly

    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    return f"{head} {Path(cleverly.__file__).parent}"


def run(n: int, reps: int, procs: int) -> None:
    OUT.mkdir(exist_ok=True)
    sha = script_sha()
    t0 = time.time()
    with Pool(procs, initializer=init) as pool:
        rows = [r for chunk in pool.imap(one, [(n, i) for i in range(reps)], chunksize=2) for r in chunk]
    frame = pd.DataFrame(rows)
    frame["sha256"] = sha
    frame["head"] = _head()
    frame["wall"] = time.time() - t0
    frame.to_csv(OUT / f"n{n}.csv", index=False)
    print(f"n={n} R={reps} done in {time.time() - t0:.0f}s", flush=True)


def _sd(frame: pd.DataFrame, cell: str, n: int) -> float:
    g = frame.loc[(frame["cell"] == cell) & (frame["n"] == n) & (frame["estimand"] == "ate")]
    return float(g["psi"].std(ddof=1))


def _paired(frame: pd.DataFrame, a: str, b: str, unit: str) -> None:
    """Per-replicate `a - b` on the ATE, over the replicates kept in both, in SDs of `unit`."""
    for n, g in frame.loc[frame["estimand"] == "ate"].groupby("n"):
        x = g.loc[g["cell"] == a].set_index("rep")["psi"]
        y = g.loc[g["cell"] == b].set_index("rep")["psi"]
        d = (x - y).dropna() / _sd(frame, unit, n)
        print(f"  {a:>13s} - {b:13s} n={n:6d}: pairs {d.size:3d}; mean {d.mean():+.3f} (mcse "
              f"{d.std(ddof=1) / np.sqrt(d.size):.3f}); SD {d.std(ddof=1):.3f}; mean |d| {d.abs().mean():.3f}")


def analyze() -> None:
    sha = script_sha()
    print(f"sha256 {sha}  reviews/rm41/probe.py")
    frame = pd.concat([pd.read_csv(p) for p in sorted(OUT.glob("n*.csv"))], ignore_index=True)
    print(f"run CSVs written by this script: {set(frame['sha256'].dropna()) == {sha}}")
    print(f"commit and cleverly path of the run: {sorted(set(frame['head'].dropna()))}")
    errors = frame.loc[frame["estimand"] == "error"]
    print(f"errors: {len(errors)}" + ("" if errors.empty else f"; first: {errors['error'].iloc[0]}"))
    frame = frame.loc[frame["estimand"] != "error"].copy()
    print(f"max |g_n gap| between cleverly's in-sample mechanism and the refit: {frame['gap'].max():.2e}")
    tm = frame.loc[frame["cell"].isin(["tmle_in", "own_in_tmle"])].pivot_table(
        index=["n", "rep", "estimand"], columns="cell", values="psi"
    )
    print(f"max |own_in_tmle - tmle_in| over {len(tm)} estimates: {(tm['own_in_tmle'] - tm['tmle_in']).abs().max():.1e}")
    pop = frame["cell"].str.startswith("pop")
    print(f"population-limit fits with a worst score above 1e-6: {int((frame.loc[pop, 'score'] > 1e-6).sum() / 3)} "
          f"of {int(pop.sum() / 3)} (largest {frame.loc[pop, 'score'].max():.1e}); all are kept")
    own = frame["cell"].str.startswith("own")
    print()
    print("Robust reading over all replicates of each own_drtmle cell (ATE): median error / SD(dr_in), and IQR.")
    for (cell, n), g in frame.loc[own & (frame["estimand"] == "ate")].groupby(["cell", "n"]):
        e = (g["psi"] - g["truth"]) / _sd(frame, "dr_in", n)
        q1, q3 = e.quantile([0.25, 0.75])
        print(f"  {cell:13s} n={n:6d}: R {len(e)}; median {e.median():+.3f}; IQR [{q1:+.3f}, {q3:+.3f}]")
    solved = ~own | (frame["score"] < SOLVED)
    print(f"own_drtmle fits kept (every score below {SOLVED:g}): {int(solved[own].sum() / 3)} of {int(own.sum() / 3)}")
    frame = frame.loc[solved]
    print()
    print("bias/SD: mean(psi - truth) / SD(psi) over the kept fits of the cell; mcse 1/sqrt(R).")
    print("rn_bias: sqrt(n) * mean(psi - truth). cover: interval coverage, with its mcse.")
    print("pop_* cells are population limits: their SD is not a sampling SD, so they are read below.")
    print(f"{'cell':13s} {'n':>6s} {'est':6s} {'R':>4s} {'bias/SD':>8s} {'mcse':>6s} {'rn_bias':>8s} {'cover':>6s} {'mcse':>6s}")
    for (cell, n, name), g in frame.groupby(["cell", "n", "estimand"], sort=True):
        if cell.startswith("pop"):
            continue
        r = len(g)
        err = g["psi"] - g["truth"]
        sd = float(g["psi"].std(ddof=1))
        cov = float(g["covered"].mean()) if "covered" in g and g["covered"].notna().any() else np.nan
        cmc = np.sqrt(cov * (1 - cov) / r) if np.isfinite(cov) else np.nan
        print(f"{cell:13s} {n:6d} {name:6s} {r:4d} {float(err.mean()) / sd:+8.3f} {1 / np.sqrt(r):6.3f} "
              f"{np.sqrt(n) * float(err.mean()):+8.3f} {cov:6.3f} {cmc:6.3f}")
    print()
    print("Population limits at the independent mechanism (ATE): mean(psi_lim - psi_0) / SD(own_ext), with")
    print("the mcse of that mean, beside the observed own_ext cells on the same replicates.")
    for n in sorted(frame["n"].unique()):
        unit = _sd(frame, "own_ext", n)
        parts = []
        for cell in ("pop_ext", "pop_ext_g0", "pop_ext_tmle", "own_ext", "own_ext_g0", "own_ext_tmle"):
            g = frame.loc[(frame["cell"] == cell) & (frame["n"] == n) & (frame["estimand"] == "ate")]
            e = (g["psi"] - g["truth"]) / unit
            parts.append(f"{cell} {e.mean():+.3f} ({e.std(ddof=1) / np.sqrt(len(e)):.3f})")
        print(f"  n={n:6d}: " + "; ".join(parts))
    print()
    print("Paired differences on the ATE (same draw, same initial fits).")
    print("cleverly against own_drtmle at one independent mechanism, in SDs of own_ext:")
    for a, b in (("dr_ext", "own_ext"), ("dr_ext_bk", "own_ext"), ("dr_ext", "dr_ext_bk")):
        _paired(frame, a, b, "own_ext")
    print("cleverly against own_drtmle in sample, and cleverly's two orders, in SDs of dr_in:")
    for a, b in (("own_in", "dr_in"), ("dr_in_bk", "dr_in")):
        _paired(frame, a, b, "dr_in")
    print("own-row term (in sample minus independent mechanism) and index effect (g_n minus g_0), in SDs of dr_in:")
    for a, b in (("own_in", "own_ext"), ("own_in_g0", "own_ext_g0"), ("own_in_tmle", "own_ext_tmle"),
                 ("own_in", "own_in_g0"), ("own_ext", "own_ext_g0")):
        _paired(frame, a, b, "dr_in")
    _weights(frame)


def _weights(frame: pd.DataFrame) -> None:
    print()
    print("In-sample HGB: mean(1{A=a}/g_n - 1) on its own training rows and on the 200,000-row")
    print("evaluation draw (both zero in mean at g_0). Mean over replicates; sd = their spread.")
    one_row = frame.loc[(frame["cell"] == "dr_in") & (frame["estimand"] == "ate")]
    for n, g in one_row.groupby("n"):
        cols = ("w1_train", "w1_fresh", "w0_train", "w0_fresh")
        print(f"n={n:6d}: " + "; ".join(f"{c} {g[c].mean():+.4f} (sd {g[c].std(ddof=1):.4f})" for c in cols))


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
