# A probe, not a registered study. RM40b, the drift-attribution probe (RM40 plan section 6, D1, D2).
#
# It predicts the DR-TMLE ATE bias that the conditioning term produces in the two archived Tier-2
# cells (`g-drift`, `q-drift` of `benchmarks/drtmle_tier2.py` at tag
# `drtmle-validation-archive-2026-08`). It then reads the archived measured bias and applies the
# decision rule below, which was written before the first run.
#
# | item | value |
# | --- | --- |
# | command | `python reviews/rm40/drift_attribution.py > reviews/rm40/drift_attribution.log` (one process, one BLAS thread, under 1 GB) |
# | reads | at the tag, through `git show`: `benchmarks/drtmle_tier2.py` (imported verbatim; `KernelMechanism`, `KernelOutcome`, `_smooth_one`, `bandwidth`, `wrong_mechanism_coefficients`), `src/cleverly/datasets/synthetic.py` (the `linear_dgp` block), `docs/drtmle/investigation-log.md`, `docs/drtmle/coverage-study.md`, `docs/drtmle/construction-contrasts.md` (cited lines are checked for the quoted numbers), `evidence/f5-terminal/20260807T171525-audit.jsonl` and `20260807T152658-selection.jsonl` (per-replicate DRTMLE rows) |
# | seeds | quadrature: scrambled Sobol, `seed=40_001` (primary, 2^20 points) and `seed=40_002` (quadrature-error check). Fit `r` of kind `k` at size `n` draws from `numpy.random.default_rng([20261012, k, n, r])`, `k` = 1 (mechanism backfit, 4n/5 rows), 2 (outcome backfit, 4n/5 rows), 3 (outcome backfit, n rows, variance check) |
# | replicates | mechanism backfit 2,000 / 1,200 / 600 fits at n = 600 / 1,200 / 2,400; outcome backfit 1,000 / 600 / 400; variance check 600 / 400 / 300. Monte Carlo errors of the predictions are batch-means errors over 20 batches |
# | log | the first line is the sha256 of this file as it ran |
# | pilot | one `--scale 0.04` run tested the code before the committed run. After it, the fixed-h limit was corrected to the archived double smoothing (two diagnostic columns and the note below) and one citation line number was fixed. The rule and the primary predictions were not changed |
#
# Notation (0-indexed covariates, as the code's column indices). W0..W3 iid N(0, 1),
# g0 = expit(s), s = 0.3 W0 - 0.2 W1 + 0.1 W2, Q0(a, W) = 2 + 1.5 a + W0 + 0.5 W1 - 0.8 W2 + 0.4 W3,
# Y = Q0(A, W) + N(0, 1). All quantities are on the raw outcome scale. The archived learners
# fit the outcome scaled by `q_bounds = (-8, 14)`; both smoothers and OLS are linear in the
# target, so the raw scale gives the same fits, and no clip is active at these values.
#
# g-drift (outcome per-arm subset OLS: arm 1 on W1, W2, W3; arm 0 on W1, W2; mechanism the
# archived additive Nadaraya-Watson backfit, h_n = 1.15 n^-0.125, trained on 4n/5 rows because
# the archived settings cross-fit with 5 folds). Per arm, with a limit mechanism ghat:
#   B1 = P0[(g0 - ghat)/ghat * F],  F = Q0 - Qbar - Qr,  Qr(t) = E[g0 (Q0 - Qbar) | ghat = t] / E[g0 | ghat = t]
# Qbar is the g0-weighted (arm 0: (1 - g0)-weighted) OLS limit on the kept columns. It is
# computed by quadrature, because under A = a the covariates are tilted and the dropped term
# does not simply vanish. Qr is a weighted degree-9 polynomial in ghat (checked against
# 256-bin conditional means). Arm 0 is the mirror with 1 - g, 1 - ghat. ghat = E ghat_n, the mean
# of the archived `KernelMechanism` fits; each additive component is evaluated on a grid of
# 1,201 points on [-6, 6] and interpolated. The split is
#   (a) = P0[(g0 - g_add)/ghat * F]  (the n -> infinity additive-projection error),
#   (b) = P0[(g_add - ghat)/ghat * F] (the smoothing bias), (a) + (b) = B1,
# with g_add = sum_j E[g0 | W_j] - 3 E[g0], by 1-D Gauss-Hermite quadrature. Predicted ATE bias:
# B_ATE = B1 - B0. This omits second-order terms: the targeting step's shift of Qbar, the
# random part of ghat_n, and the estimation error of the reductions. The first two are
# reported as sensitivity rows; the archived reductions are `glm`, so a linear-in-ghat Qr is a
# third sensitivity row.
#
# q-drift (outcome the archived additive kernel backfit per arm; mechanism a logistic GLM on
# W1, W2, limit gbar from the archived `REFERENCE_N = 200,000` fit). Per arm 1, with
# r = (g0 - gbar)/gbar and h = r - g0 E[r | Q0] / E[g0 | Q0] (so E[h | Q0] = 0), the leftover is
# P0[h (Q0 - Qhat_n)]. Arm 0: r0 = (gbar - g0)/(1 - gbar), h0 = r0 - (1 - g0) E[r0 | Q0(0)] / E[1 - g0 | Q0(0)].
# The conditional expectations given Q0 are exact 2-D Gauss-Hermite integrals, because
# (s, logit gbar) given the linear index of Q0 is bivariate normal.
#  - Deterministic part: M_a = P0[h_a (Q0_a - E Qhat_n,a)]. Two versions: the analytic limit
#    (affine, a constant plus beta.w/(1 + h^2)^2, a function of Q0, so M_a = 0 and the computed
#    value is quadrature error), and E Qhat_n,a from the mean of the archived `KernelOutcome`
#    fits (primary for the rule). Predicted ATE bias M_1 - M_0. The factor is 1/(1 + h^2)^2,
#    not the plan's 1/(1 + h^2): the archived `_Additive._evaluate` smooths the stored
#    training-point components a second time, so the fixed-h limit applies the smoother twice.
#    The same holds for the mechanism backfit (the g_h,inf column of step 2). Any multiple of
#    beta.w is a function of Q0, so the plan's conclusion is unchanged.
#  - Variance part. For a linear additive smoother on the A = a rows,
#    P0[h (Qhat_n - E Qhat_n)] ~ P_n[A (Y - Q0) k1], k1 = Pi(h/g0), the projection onto additive
#    functions in L2(g0 dP) (arm 0: (1 - A), 1 - g0). The reported curve at the limit nuisances
#    is the closed form D_rep = A (Y - Q0) a1 - (1 - A)(Y - Q0) a0 + const, a1 = 1/gbar - E[r|Q0]/E[g0|Q0]
#    = 1/g0 + h/g0 (D* - D_Q - D_g of `cleverly.inference.influence.reduced_corrections` at the
#    tag, with Qr = 0). The estimator's curve is D_rep - (extra), extra = A (Y - Q0) k1 - (1 - A)(Y - Q0) k0.
#    Predicted SE ratio = sd(D_rep) / sd(D_rep - extra). Pi uses Hermite polynomials to degree 8
#    per covariate (degree 5 as a check). A Monte Carlo check with the finite-h archived
#    smoother (fits on n rows, no cross-fitting) is printed beside it.
#  - N, the points of the archived 10.4-point shortfall that the predicted ratio alone accounts
#    for: N = 100 (0.95 - (2 Phi(1.96 ratio) - 1)). At the archived ratio 0.903, N = 2.7.
#
# Archived bias at n = 2,400 (raw ATE units; the root-n bias is sqrt(2400) times it):
#  - q-drift, source S1 (primary): C3c batch A, root-n bias 1.44 (investigation-log.md:1607-1608),
#    empirical sd 0.0443 (:1625), 250 replicates (:1533). MC SE = 0.0443 / sqrt(250).
#    Source S2: the F5 `glm-pooled` arm (the shipped construction, `glm` reductions), both
#    cohorts pooled, computed from the per-replicate rows.
#  - g-drift: the docs section 6 names state no C3c bias at n = 2,400 (deviation). Source S1
#    (primary): the F5 `glm-pooled` arm, both cohorts pooled. Source S2: the C3c bias implied by
#    the Wald identity cov = Phi(1.96 r - b) - Phi(-1.96 r - b) at the batch A coverage 0.780
#    (investigation-log.md:1559) and SE ratio 1.157 (coverage-study.md:942), times the F5 pooled
#    empirical sd. Its MC SE comes from the binomial error of the coverage (250 replicates).
#  - If S1 and S2 select different outcomes, the outcome is inconclusive.
#
# Decision rule, fixed before the run (RM40 plan section 6). Bias only, at n = 2,400, applied per
# cell to the predicted ATE bias (g-drift: B_ATE = (a) + (b); q-drift: the deterministic mirror
# part M_1 - M_0 with the averaged archived smoother). "MC SE" is the archived bias's MC SE.
#
# | outcome | criterion | docs text |
# | --- | --- | --- |
# | confirmed | the predicted ATE bias has the archived sign and abs(predicted - archived) <= max(0.30 abs(archived), 2 MC SE) | g-drift names the dominant part. If (b) dominates: "The shortfall is consistent with the smoothing bias of the mechanism fit, which adds a conditioning term of order `h_n²`. The fit also converges to an additive approximation of `g_0`, outside Theorem 1; that error is smaller." If (a) dominates: the same two sentences in the other order. q-drift (the plan gives no wording; this probe's): "The shortfall is consistent with the mirror conditioning term of the outcome smoother." |
# | refuted | (abs(predicted) < abs(archived)/3, or the predicted bias has the opposite sign) and the predicted bias lies outside archived +/- 2 MC SE | "The shortfall is not attributed. The mechanism error does not account for it." |
# | inconclusive | otherwise | "The shortfall is not attributed." |
#
# The SE-ratio predictions are variance findings, reported beside the table. For g-drift
# (archived 1.157, a long SE) a variance match never confirms an attribution. For q-drift
# (archived 0.903, a short SE) the predicted ratio is a partial account, stated as "the variance
# part accounts for about N of the 10.4 points of shortfall". If the predicted ratio is 1 or
# more, N <= 0 and the variance part accounts for none of it. The bias rule decides the rest.
"""RM40b drift-attribution probe for the archived DR-TMLE Tier-2 cells (non-registered)."""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.optimize import brentq
from scipy.stats import norm, qmc
from sklearn.linear_model import LogisticRegression

TAG = "drtmle-validation-archive-2026-08"
REPO = Path(__file__).resolve().parents[2]
SIZES = (600, 1200, 2400)
GAMMA = np.array([0.3, -0.2, 0.1, 0.0])
BETA = np.array([1.0, 0.5, -0.8, 0.4])
EFFECT = 1.5
KEPT = {1: (1, 2, 3), 0: (1, 2)}  # subset OLS columns per arm (archived WRONG_OUTCOME_COLUMNS)
GRID = np.linspace(-6.0, 6.0, 1201)
SEED_BASE = 20261012
QUAD_SEEDS = (40_001, 40_002)
REPS_G = {600: 2000, 1200: 1200, 2400: 600}
REPS_Q = {600: 1000, 1200: 600, 2400: 400}
REPS_V = {600: 600, 1200: 400, 2400: 300}
BATCHES = 20
SHORTFALL_POINTS = 10.4


def say(text: str = "") -> None:
    print(text, flush=True)


def git_show(path: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), "show", f"{TAG}:{path}"],
        check=True, capture_output=True, text=True,
    ).stdout  # fmt: skip


def expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def g0(w: np.ndarray) -> np.ndarray:
    return expit(w @ GAMMA)


def q0(w: np.ndarray, a: float | np.ndarray) -> np.ndarray:
    return 2.0 + EFFECT * a + w @ BETA


def sample(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The tag's `DGP.sample` for `linear_dgp`: latent, then A, then the outcome noise."""
    w = rng.normal(size=(n, 4))
    a = rng.binomial(1, np.clip(g0(w), 1e-9, 1.0 - 1e-9)).astype(float)
    y = np.where(a == 1.0, q0(w, 1.0), q0(w, 0.0)) + rng.normal(scale=1.0, size=n)
    return w, a, y


def sobol_normal(log2: int, seed: int) -> np.ndarray:
    u = qmc.Sobol(d=4, scramble=True, seed=seed).random_base2(log2)
    return norm.ppf(np.clip(u, 1e-12, 1.0 - 1e-12))


def gh(k: int) -> tuple[np.ndarray, np.ndarray]:
    x, wt = hermegauss(k)
    return x, wt / np.sqrt(2.0 * np.pi)


def wls(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    xw = x * w[:, None]
    return np.linalg.solve(x.T @ xw, xw.T @ y)


def additive(curves: np.ndarray, intercept: float, w: np.ndarray) -> np.ndarray:
    """intercept + sum_j curve_j(w_j), each component interpolated on GRID."""
    out = np.full(w.shape[0], intercept)
    for j in range(curves.shape[0]):
        out += np.interp(w[:, j], GRID, curves[j])
    return out


def load_archived() -> object:
    tmp = Path(tempfile.mkdtemp(prefix="rm40b_"))
    path = tmp / "drtmle_tier2_archived.py"
    path.write_text(git_show("benchmarks/drtmle_tier2.py"), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("drtmle_tier2_archived", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ step 1: the law


def step1(t2: object) -> tuple[np.ndarray, float]:
    say("## Step 1. linear_dgp re-implemented and checked")
    tag_src = git_show("src/cleverly/datasets/synthetic.py")
    head_src = (REPO / "src/cleverly/datasets/synthetic.py").read_text(encoding="utf-8")

    def block(src: str) -> str:
        start = src.index("def linear_dgp(")
        return src[start : src.index("return DGP(", start)]

    assert block(tag_src) == block(head_src), "linear_dgp differs between the tag and HEAD"
    say("linear_dgp's propensity and outcome_mean source block: identical at the tag and at HEAD")
    from cleverly.datasets import linear_dgp

    dgp = linear_dgp()
    probe = np.random.default_rng(7).normal(size=(1000, 4))
    assert np.allclose(dgp.propensity(probe), g0(probe), rtol=0, atol=1e-15)
    for a in (0.0, 1.0):
        assert np.allclose(dgp.outcome_mean(probe, a, None), q0(probe, a), rtol=0, atol=1e-12)
    n_ref, seed_ref = t2.REFERENCE_N, t2.REFERENCE_SEED
    w, a, y = sample(n_ref, np.random.default_rng(seed_ref))
    frame, _ = dgp.sample(n_ref, seed=seed_ref)
    names = list(dgp.covariate_names)
    assert np.array_equal(frame[names].to_numpy(), w) and np.array_equal(frame["A"].to_numpy(), a)
    assert np.allclose(frame["Y"].to_numpy(), y, rtol=0, atol=1e-12)
    mine = LogisticRegression(C=1e6, max_iter=1000).fit(w[:, [1, 2]], a)
    theta, icpt = t2.wrong_mechanism_coefficients()
    assert np.allclose(mine.coef_.ravel(), theta, rtol=0, atol=1e-10)
    assert abs(float(mine.intercept_[0]) - icpt) < 1e-10
    say(f"REFERENCE_N = {n_ref:,}, REFERENCE_SEED = {seed_ref}: re-implemented sampler reproduces "
        "the cleverly draw exactly (W, A, Y)")  # fmt: skip
    say(f"wrong_mechanism_coefficients (archived, code columns (1, 2)): theta = "
        f"({theta[0]:+.8f}, {theta[1]:+.8f}), intercept = {icpt:+.8f}; re-implemented fit "
        "agrees to 1e-10")  # fmt: skip
    say()
    return np.asarray(theta, dtype=float), float(icpt)


# ------------------------------------------------------------------ step 2: g-drift


def g_add_curves(h: float | None) -> tuple[np.ndarray, float]:
    """Additive components of g0's L2 additive projection (h None), or the backfit's fixed-h limit."""
    x, wt = gh(80)
    var = GAMMA**2
    mean_g = float(np.sum(wt * expit(np.sqrt(var.sum()) * x)))
    curves = np.zeros((4, GRID.size))
    for j in range(4):
        rest = var.sum() - var[j]
        if h is None:
            mu, tau2 = GAMMA[j] * GRID, rest
        else:
            # The archived prediction smooths twice (fit, then `_evaluate`): V2 | t is normal
            # with mean t/(1 + h^2)^2 and variance h^2/(1 + h^2) (1 + 1/(1 + h^2)^2).
            shrink = 1.0 + h * h
            mu = GAMMA[j] * GRID / shrink**2
            tau2 = rest + var[j] * h * h / shrink * (1.0 + 1.0 / shrink**2)
        curves[j] = expit(mu[:, None] + np.sqrt(tau2) * x[None, :]) @ wt - mean_g
    return curves, mean_g


def qbar(w: np.ndarray, arm: int) -> np.ndarray:
    weight = g0(w) if arm == 1 else 1.0 - g0(w)
    x = np.column_stack([np.ones(w.shape[0]), w[:, list(KEPT[arm])]])
    coef = wls(x, q0(w, float(arm)), weight)
    return x @ coef


def qr_fit(ghat: np.ndarray, target: np.ndarray, weight: np.ndarray, how: str) -> np.ndarray:
    """E[target | ghat] under `weight`: polynomial (degree 9 or 1) or 256 equal-count bins."""
    if how == "bins":
        order = np.argsort(ghat)
        edges = np.linspace(0, ghat.size, 257).astype(int)
        centers, means = np.empty(256), np.empty(256)
        for k in range(256):
            idx = order[edges[k] : edges[k + 1]]
            wk = weight[idx]
            centers[k] = np.sum(wk * ghat[idx]) / wk.sum()
            means[k] = np.sum(wk * target[idx]) / wk.sum()
        return np.interp(ghat, centers, means)
    degree = {"poly9": 9, "poly5": 5, "linear": 1}[how]
    z = (ghat - ghat.mean()) / ghat.std()
    x = np.vander(z, degree + 1, increasing=True)
    return x @ wls(x, target, weight)


class GDriftLaw:
    """Fixed quadrature quantities of the g-drift cell."""

    def __init__(self, w: np.ndarray) -> None:
        self.w = w
        self.g = g0(w)
        self.d = {arm: q0(w, float(arm)) - qbar(w, arm) for arm in (1, 0)}
        self.qbar = {arm: qbar(w, arm) for arm in (1, 0)}
        add, mean_g = g_add_curves(None)
        self.g_add = additive(add, mean_g, w)

    def leftover(self, ghat: np.ndarray, how: str = "poly9", targeted: bool = False) -> dict:
        out = {}
        for arm in (1, 0):
            g_true = self.g if arm == 1 else 1.0 - self.g
            g_lim = ghat if arm == 1 else 1.0 - ghat
            g_mid = self.g_add if arm == 1 else 1.0 - self.g_add
            d = self.d[arm]
            if targeted:
                d = d - self._shift(arm, g_true, g_lim)
            f = d - qr_fit(g_lim, d, g_true, how)
            a_part = float(np.mean((g_true - g_mid) / g_lim * f))
            b_part = float(np.mean((g_mid - g_lim) / g_lim * f))
            out[arm] = (a_part, b_part)
        return {
            "a1": out[1][0], "b1": out[1][1], "a0": out[0][0], "b0": out[0][1],
            "a": out[1][0] - out[0][0], "b": out[1][1] - out[0][1],
            "ate": out[1][0] + out[1][1] - out[0][0] - out[0][1],
        }  # fmt: skip

    def _shift(self, arm: int, g_true: np.ndarray, g_lim: np.ndarray) -> np.ndarray:
        """Q* - Qbar for a logistic fluctuation of the scaled Qbar along 1/ghat (TMLE step)."""
        lo, width = -8.0, 22.0
        base = (self.qbar[arm] - lo) / width
        logit = np.log(base / (1.0 - base))
        target = (q0(self.w, float(arm)) - lo) / width

        def score(eps: float) -> float:
            return float(np.mean(g_true / g_lim * (target - expit(logit + eps / g_lim))))

        eps = brentq(score, -1.0, 1.0)
        return width * (expit(logit + eps / g_lim) - base)


def mechanism_fits(t2: object, n: int, reps: int) -> tuple[np.ndarray, np.ndarray]:
    m = n - n // 5
    curves = np.empty((reps, 4, GRID.size))
    icpt = np.empty(reps)
    for r in range(reps):
        w, a, _ = sample(m, np.random.default_rng([SEED_BASE, 1, n, r]))
        st = t2.KernelMechanism(n).fit(w, a).state_
        icpt[r] = st["intercept"]
        for j in range(4):
            curves[r, j] = t2._smooth_one(
                st["design"][:, j], st["components"][j], st["weight"], GRID, st["h"]
            )
    return curves, icpt


def batch_se(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def step2(t2: object, wq: np.ndarray, wq2: np.ndarray, sub: np.ndarray, scale: float) -> dict:
    say("## Step 2. g-drift: E ghat_n, the additive split, and the predicted ATE bias")
    law, law2, law_sub = GDriftLaw(wq), GDriftLaw(wq2), GDriftLaw(sub)
    gq = law.g
    say(f"g0 on the quadrature: mean {gq.mean():.6f}, sd {gq.std():.6f}; "
        f"||g_add - g0||_2 = {np.sqrt(np.mean((law.g_add - gq) ** 2)):.3e} (the n -> infinity error)")  # fmt: skip
    for arm in (1, 0):
        say(f"arm {arm}: Qbar weighted-OLS limit; E[Q0 - Qbar] = {law.d[arm].mean():+.5f}, "
            f"E[w_a (Q0 - Qbar)] = {np.mean((gq if arm else 1 - gq) * law.d[arm]):+.1e}")  # fmt: skip
    results = {}
    say()
    say("| n | fits | h_n | train rows | ||E ghat_n - g0|| | ||E ghat_n - g_add|| | ||E ghat_n - g_h,inf|| | min, max of E ghat_n |")
    say("| --- | --- | --- | --- | --- | --- | --- | --- |")
    rows = []
    for n in SIZES:
        reps = max(BATCHES * 2, int(round(REPS_G[n] * scale)))
        t0 = time.time()
        curves, icpt = mechanism_fits(t2, n, reps)
        secs = time.time() - t0
        mean_curves, mean_icpt = curves.mean(axis=0), float(icpt.mean())
        ghat = additive(mean_curves, mean_icpt, wq)
        hn = t2.bandwidth(n)
        hlim, mg = g_add_curves(hn)
        g_hinf = additive(hlim, mg, wq)
        rows.append(
            f"| {n:,} | {reps} | {hn:.4f} | {n - n // 5:,} | "
            f"{np.sqrt(np.mean((ghat - gq) ** 2)):.3e} | {np.sqrt(np.mean((ghat - law.g_add) ** 2)):.3e} | "
            f"{np.sqrt(np.mean((ghat - g_hinf) ** 2)):.3e} | {ghat.min():.4f}, {ghat.max():.4f} |"
        )
        primary = law.leftover(ghat)
        ghat2 = additive(mean_curves, mean_icpt, wq2)
        quad2 = law2.leftover(ghat2)
        bins = law.leftover(ghat, "bins")
        poly5 = law.leftover(ghat, "poly5")
        linear = law.leftover(ghat, "linear")
        targeted = law.leftover(ghat, "poly9", targeted=True)
        hinf = law.leftover(g_hinf)
        # Monte Carlo error: predictions at batch means, on a 2^17 sub-quadrature.
        per_batch = np.array_split(np.arange(reps), BATCHES)
        batch_vals = []
        for idx in per_batch:
            gb = additive(curves[idx].mean(axis=0), float(icpt[idx].mean()), sub)
            batch_vals.append(law_sub.leftover(gb))
        mc = {key: batch_se(np.array([b[key] for b in batch_vals])) for key in ("a", "b", "ate")}
        # Mean over fits of the leftover with a linear-in-ghat Qr (includes the random part).
        per_fit = np.array(
            [law_sub.leftover(additive(curves[r], float(icpt[r]), sub), "linear")["ate"] for r in range(reps)]
        )
        at_mean_sub = law_sub.leftover(additive(mean_curves, mean_icpt, sub), "linear")["ate"]
        results[n] = {
            "primary": primary, "quad2": quad2, "bins": bins, "poly5": poly5, "linear": linear,
            "targeted": targeted, "hinf": hinf, "mc": mc, "per_fit_mean": float(per_fit.mean()),
            "per_fit_se": float(per_fit.std(ddof=1) / np.sqrt(reps)), "at_mean_sub": at_mean_sub,
            "reps": reps, "secs": secs, "hn": hn,
        }  # fmt: skip
        del curves
    for row in rows:
        say(row)
    say("g_h,inf is the fixed-h population limit of the backfit (each E[g0 | W_j] component smoothed twice, "
        "as the archived fit-then-evaluate does); E ghat_n approaches g_add as n grows.")  # fmt: skip
    say()
    say("Per arm (raw ATE units, at ghat = E ghat_n, Qr degree-9 polynomial):")
    say("| n | (a) arm 1 | (b) arm 1 | (a) arm 0 | (b) arm 0 |")
    say("| --- | --- | --- | --- | --- |")
    for n in SIZES:
        p = results[n]["primary"]
        say(f"| {n:,} | {p['a1']:+.3e} | {p['b1']:+.3e} | {p['a0']:+.3e} | {p['b0']:+.3e} |")
    say()
    say("ATE (arm 1 minus arm 0), root-n scaled, sqrt(n) x leftover, with batch-means MC error:")
    say("| n | sqrt(n) (a) | sqrt(n) (b) | sqrt(n) B_ATE | raw B_ATE | MC SE of sqrt(n) B_ATE |")
    say("| --- | --- | --- | --- | --- | --- |")
    for n in SIZES:
        p, mc, rn = results[n]["primary"], results[n]["mc"], np.sqrt(n)
        say(f"| {n:,} | {rn * p['a']:+.4f} | {rn * p['b']:+.4f} | {rn * p['ate']:+.4f} | "
            f"{p['ate']:+.3e} | {rn * mc['ate']:.4f} |")  # fmt: skip
    say()
    say("Error of the conditional-expectation step and sensitivity rows, sqrt(n) B_ATE:")
    say("| n | primary | 2nd scramble | Qr 256 bins | Qr degree 5 | Qr linear (glm limit) | with TMLE shift of Qbar | at g_h,inf | mean over fits: random part of ghat_n included (linear Qr, 2^17) | same Qr at E ghat_n (2^17) |")
    say("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for n in SIZES:
        r, rn = results[n], np.sqrt(n)
        say(f"| {n:,} | {rn * r['primary']['ate']:+.4f} | {rn * r['quad2']['ate']:+.4f} | "
            f"{rn * r['bins']['ate']:+.4f} | {rn * r['poly5']['ate']:+.4f} | {rn * r['linear']['ate']:+.4f} | "
            f"{rn * r['targeted']['ate']:+.4f} | {rn * r['hinf']['ate']:+.4f} | "
            f"{rn * r['per_fit_mean']:+.4f} +/- {rn * r['per_fit_se']:.4f} | {rn * r['at_mean_sub']:+.4f} |")  # fmt: skip
    say("(fits: " + ", ".join(f"n = {n:,}: {results[n]['reps']} in {results[n]['secs']:.0f} s" for n in SIZES) + ")")
    say()
    return results


# ------------------------------------------------------------------ step 3: q-drift


class QDriftLaw:
    """r, h and the conditional expectations given Q0's linear index L = beta . W."""

    def __init__(self, theta: np.ndarray, icpt: float) -> None:
        self.theta, self.icpt = theta, icpt
        tvec = np.array([0.0, theta[0], theta[1], 0.0])
        cov = np.array([[GAMMA @ GAMMA, GAMMA @ tvec], [GAMMA @ tvec, tvec @ tvec]])
        c = np.array([GAMMA @ BETA, tvec @ BETA])
        var_l = BETA @ BETA
        cond = cov - np.outer(c, c) / var_l
        chol = np.linalg.cholesky(cond)
        x, wt = gh(48)
        z1, z2 = np.meshgrid(x, x, indexing="ij")
        ww = np.outer(wt, wt).ravel()
        zz = np.vstack([z1.ravel(), z2.ravel()])
        dev = chol @ zz
        self.lgrid = np.linspace(-9.0, 9.0, 1801)
        mean = np.outer(self.lgrid, c / var_l)
        s = mean[:, :1] + dev[0][None, :]
        sb = icpt + mean[:, 1:] + dev[1][None, :]
        gt, gb = expit(s), expit(sb)
        self.e = {
            "r1": ((gt - gb) / gb) @ ww, "g1": gt @ ww,
            "r0": ((gb - gt) / (1.0 - gb)) @ ww, "g0": (1.0 - gt) @ ww,
        }  # fmt: skip

    def gbar(self, w: np.ndarray) -> np.ndarray:
        return expit(self.icpt + w[:, [1, 2]] @ self.theta)

    def parts(self, w: np.ndarray) -> dict:
        g, gb, ell = g0(w), self.gbar(w), w @ BETA
        cond = {k: np.interp(ell, self.lgrid, v) for k, v in self.e.items()}
        r1, r0 = (g - gb) / gb, (gb - g) / (1.0 - gb)
        h1 = r1 - g * cond["r1"] / cond["g1"]
        h0 = r0 - (1.0 - g) * cond["r0"] / cond["g0"]
        a1_direct = 1.0 / gb - cond["r1"] / cond["g1"]
        a0_direct = 1.0 / (1.0 - gb) - cond["r0"] / cond["g0"]
        return {"g": g, "gb": gb, "ell": ell, "h1": h1, "h0": h0, "r1": r1, "r0": r0,
                "a1": 1.0 / g + h1 / g, "a0": 1.0 / (1.0 - g) + h0 / (1.0 - g),
                "a1_direct": a1_direct, "a0_direct": a0_direct}  # fmt: skip

    def glm_h(self, w: np.ndarray, p: dict) -> dict:
        """h with the archived `glm` reductions' limits (gr2 linear, gr1 logistic, in Q0)."""
        out = {}
        x = np.column_stack([np.ones(w.shape[0]), p["ell"]])
        for arm in (1, 0):
            target_r = p["r1"] if arm == 1 else p["r0"]
            prob = p["g"] if arm == 1 else 1.0 - p["g"]
            gr2 = x @ np.linalg.lstsq(x, target_r, rcond=None)[0]
            beta = np.zeros(2)
            for _ in range(50):
                mu = expit(x @ beta)
                step = np.linalg.solve((x * (mu * (1 - mu))[:, None]).T @ x, x.T @ (prob - mu))
                beta += step
                if np.max(np.abs(step)) < 1e-13:
                    break
            gr1 = expit(x @ beta)
            out[arm] = target_r - prob * gr2 / gr1
        return out


def hermite_basis(w: np.ndarray, degree: int) -> np.ndarray:
    cols = [np.ones(w.shape[0])]
    for j in range(w.shape[1]):
        prev, cur = np.ones(w.shape[0]), w[:, j].copy()
        cols.append(cur)
        for k in range(2, degree + 1):
            prev, cur = cur, w[:, j] * cur - (k - 1) * prev
            cols.append(cur / np.sqrt(float(np.prod(np.arange(1, k + 1)))))
    return np.column_stack(cols)


def additive_projection(w: np.ndarray, target: np.ndarray, weight: np.ndarray, degree: int) -> np.ndarray:
    xtx, xty = 0.0, 0.0
    for start in range(0, w.shape[0], 65536):
        sl = slice(start, start + 65536)
        x = hermite_basis(w[sl], degree)
        xw = x * weight[sl, None]
        xtx = xtx + x.T @ xw
        xty = xty + xw.T @ target[sl]
    coef = np.linalg.solve(xtx, xty)
    out = np.empty(w.shape[0])
    for start in range(0, w.shape[0], 65536):
        sl = slice(start, start + 65536)
        out[sl] = hermite_basis(w[sl], degree) @ coef
    return out


def outcome_curves(t2: object, n: int, rows: int, rng: np.random.Generator) -> tuple[dict, dict, tuple]:
    w, a, y = sample(rows, rng)
    learner = t2.KernelOutcome(n).fit(np.column_stack([a, w]), y)
    curves, icpts = {}, {}
    for arm in (1, 0):
        st = learner.states_[float(arm)]
        icpts[arm] = st["intercept"]
        curves[arm] = np.array(
            [t2._smooth_one(st["design"][:, j], st["components"][j], st["weight"], GRID, st["h"]) for j in range(4)]
        )
    return curves, icpts, (w, a, y)


def step3(t2: object, theta: np.ndarray, icpt: float, wq: np.ndarray, wq2: np.ndarray,
          sub: np.ndarray, scale: float) -> dict:  # fmt: skip
    say("## Step 3. q-drift: the mirror term's deterministic part and its variance part")
    law = QDriftLaw(theta, icpt)
    p, p2, ps = law.parts(wq), law.parts(wq2), law.parts(sub)
    err_a = max(np.max(np.abs(p["a1"] - p["a1_direct"])), np.max(np.abs(p["a0"] - p["a0_direct"])))
    assert err_a < 1e-9, err_a
    say(f"identity a = 1/g + h/g (arm 1; arm 0 mirrored) holds to {err_a:.1e}")
    for arm in (1, 0):
        h = p[f"h{arm}"]
        say(f"arm {arm}: E[h] = {h.mean():+.2e}, E[h L] = {np.mean(h * p['ell']):+.2e}, "
            f"E[h L^2] = {np.mean(h * p['ell'] ** 2):+.2e}, sd(h) = {h.std():.4f} "
            f"(2nd scramble: E[h L] = {np.mean(p2[f'h{arm}'] * p2['ell']):+.2e})")  # fmt: skip
    say()
    results = {"law": law}
    say("Deterministic part, sqrt(n) (M_1 - M_0):")
    say("| n | h_n | analytic affine limit (quadrature error) | 2nd scramble | averaged archived smoother | MC SE | glm-reduction h | ||E Qhat_n,1 - affine limit|| | fits |")
    say("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    glm = law.glm_h(wq, p)
    for n in SIZES:
        hn, rn = t2.bandwidth(n), np.sqrt(n)
        shrink = 1.0 - 1.0 / (1.0 + hn * hn) ** 2

        def affine_m(pp: dict) -> float:
            # Q0 - E Qhat = const + L (1 - 1/(1 + h^2)^2); the constant is arbitrary because E[h] = 0.
            return float(np.mean(pp["h1"] * (0.7 + shrink * pp["ell"])) - np.mean(pp["h0"] * (0.3 + shrink * pp["ell"])))

        reps = max(BATCHES * 2, int(round(REPS_Q[n] * scale)))
        m = n - n // 5
        store = {arm: np.empty((reps, 4, GRID.size)) for arm in (1, 0)}
        icp = {arm: np.empty(reps) for arm in (1, 0)}
        t0 = time.time()
        for r in range(reps):
            curves, icpts, _ = outcome_curves(t2, n, m, np.random.default_rng([SEED_BASE, 2, n, r]))
            for arm in (1, 0):
                store[arm][r], icp[arm][r] = curves[arm], icpts[arm]
        secs = time.time() - t0

        def mirror(pp: dict, ww: np.ndarray, idx: np.ndarray | None = None, hh: dict | None = None) -> float:
            total = 0.0
            for arm, sign in ((1, 1.0), (0, -1.0)):
                sel = slice(None) if idx is None else idx
                eq = additive(store[arm][sel].mean(axis=0), float(icp[arm][sel].mean()), ww)
                h = pp[f"h{arm}"] if hh is None else hh[arm]
                total += sign * float(np.mean(h * (q0(ww, float(arm)) - eq)))
            return total

        main = mirror(p, wq)
        batch = np.array([mirror(ps, sub, idx) for idx in np.array_split(np.arange(reps), BATCHES)])
        eq1 = additive(store[1].mean(axis=0), float(icp[1].mean()), wq)
        aff1 = q0(wq, 1.0) - (np.mean(q0(wq, 1.0) - eq1) + shrink * p["ell"])
        dist = float(np.sqrt(np.mean((eq1 - aff1) ** 2)))
        results[n] = {"affine": affine_m(p), "affine2": affine_m(p2), "main": main,
                      "mc": batch_se(batch), "glm": mirror(p, wq, hh=glm), "reps": reps, "secs": secs}  # fmt: skip
        r_ = results[n]
        say(f"| {n:,} | {hn:.4f} | {rn * r_['affine']:+.2e} | {rn * r_['affine2']:+.2e} | {rn * main:+.4f} | "
            f"{rn * r_['mc']:.4f} | {rn * r_['glm']:+.4f} | {dist:.4f} | {reps} in {secs:.0f} s |")  # fmt: skip
        del store
    say("(the affine constant is arbitrary because E[h] = 0; the archived smoother is trained on 4n/5 rows per fold, per arm on the A = a rows)")
    say()

    # Variance part.
    say("Variance part (raw scale, Var(Y | A, W) = 1):")
    g = p["g"]
    k = {}
    for degree in (5, 8):
        k[degree] = {
            1: additive_projection(wq, p["h1"] / g, g, degree),
            0: additive_projection(wq, p["h0"] / (1.0 - g), 1.0 - g, degree),
        }
    var_rep = float(np.mean(g * p["a1"] ** 2) + np.mean((1.0 - g) * p["a0"] ** 2))
    ratios = {}
    for degree in (5, 8):
        kk = k[degree]
        var_true = float(np.mean(g * (p["a1"] - kk[1]) ** 2) + np.mean((1.0 - g) * (p["a0"] - kk[0]) ** 2))
        ratios[degree] = np.sqrt(var_rep / var_true)
    ek = {arm: float(np.mean(k[8][arm])) for arm in (1, 0)}
    ratio = ratios[8]
    coverage = 2.0 * norm.cdf(1.96 * ratio) - 1.0
    n_points = 100.0 * (0.95 - coverage)
    say(f"Var(D_rep) = {var_rep:.5f}; projection ||k1||^2_g0 = {np.mean(g * k[8][1] ** 2):.5f}, "
        f"||k0||^2_(1-g0) = {np.mean((1 - g) * k[8][0] ** 2):.5f}, E[k1] = {ek[1]:+.5f}, E[k0] = {ek[0]:+.5f}")  # fmt: skip
    say(f"predicted SE ratio sd(D_rep)/sd(D_rep - extra): {ratio:.4f} (Hermite degree 8); "
        f"{ratios[5]:.4f} (degree 5)")  # fmt: skip
    say(f"implied coverage from the ratio alone: 2 Phi(1.96 x {ratio:.4f}) - 1 = {coverage:.4f}; "
        f"N = 100 (0.95 - {coverage:.4f}) = {n_points:+.2f} points of the {SHORTFALL_POINTS} "
        f"(at the archived 0.903: {100 * (0.95 - (2 * norm.cdf(1.96 * 0.903) - 1)):.2f})")  # fmt: skip
    # Monte Carlo check with the finite-h archived smoother, n rows, no cross-fitting.
    say()
    say("Monte Carlo check of the variance part with the archived finite-h smoother (one fit on n rows, no cross-fitting):")
    say("T1 = P_n D_rep (true residuals), T2 = sum over arms of +/- P0[h_a (Q0_a - Qhat_a)] (2^17 sub-quadrature).")
    say("| n | fits | sd(T1) sqrt(n) | sd(T1 + T2) sqrt(n) | ratio sd(T1)/sd(T1 + T2) | bootstrap SE | projection prediction |")
    say("| --- | --- | --- | --- | --- | --- | --- |")
    sim = {}
    for n in SIZES:
        reps = max(BATCHES * 2, int(round(REPS_V[n] * scale)))
        t1, t2v = np.empty(reps), np.empty(reps)
        t0 = time.time()
        for r in range(reps):
            curves, icpts, (w, a, y) = outcome_curves(t2, n, n, np.random.default_rng([SEED_BASE, 3, n, r]))
            pp = law.parts(w)
            resid = y - q0(w, a)
            t1[r] = np.mean(a * resid * pp["a1"] - (1.0 - a) * resid * pp["a0"])
            val = 0.0
            for arm, sign in ((1, 1.0), (0, -1.0)):
                qhat = additive(curves[arm], icpts[arm], sub)
                val += sign * float(np.mean(ps[f"h{arm}"] * (q0(sub, float(arm)) - qhat)))
            t2v[r] = val
        secs = time.time() - t0
        tot = t1 + t2v
        est = float(np.std(t1, ddof=1) / np.std(tot, ddof=1))
        boot_rng = np.random.default_rng([SEED_BASE, 4, n])
        boots = []
        for _ in range(400):
            idx = boot_rng.integers(0, reps, reps)
            boots.append(np.std(t1[idx], ddof=1) / np.std(tot[idx], ddof=1))
        sim[n] = (est, float(np.std(boots, ddof=1)))
        say(f"| {n:,} | {reps} ({secs:.0f} s) | {np.sqrt(n) * np.std(t1, ddof=1):.4f} | "
            f"{np.sqrt(n) * np.std(tot, ddof=1):.4f} | {est:.4f} | {sim[n][1]:.4f} | {ratio:.4f} |")  # fmt: skip
    say(f"(sqrt(Var(D_rep)) = {np.sqrt(var_rep):.4f} is the large-n value of sd(T1) sqrt(n))")
    say()
    results.update({"ratio": ratio, "ratio5": ratios[5], "coverage": coverage, "n_points": n_points, "sim": sim})
    return results


# ------------------------------------------------------------------ step 4: archived values


def check_line(text: str, line: int, *needles: str) -> str:
    content = text.splitlines()[line - 1]
    for needle in needles:
        assert needle in content, (line, needle, content)
    return content


def f5_rows() -> dict:
    out: dict = {}
    for name in ("20260807T171525-audit.jsonl", "20260807T152658-selection.jsonl"):
        for raw in git_show(f"evidence/f5-terminal/{name}").splitlines():
            row = json.loads(raw)
            if row["arm"] == "glm-pooled" and row["estimand"] == "ate":
                out.setdefault((row["cell"], row["n"]), []).append(row)
    return out


def wald_bias(coverage: float, ratio: float) -> float:
    return brentq(lambda b: norm.cdf(1.96 * ratio - b) - norm.cdf(-1.96 * ratio - b) - coverage, 0.0, 6.0)


def step4() -> dict:
    say("## Step 4. The archived measurements (DRTMLE, ATE)")
    il = git_show("docs/drtmle/investigation-log.md")
    cs = git_show("docs/drtmle/coverage-study.md")
    cc = git_show("docs/drtmle/construction-contrasts.md")
    cites = [
        ("investigation-log.md", il, 1533, ("250",)),
        ("investigation-log.md", il, 1554, ("0.792 / 0.820",)),
        ("investigation-log.md", il, 1555, ("0.824 / 0.880",)),
        ("investigation-log.md", il, 1556, ("0.844 / 0.848",)),
        ("investigation-log.md", il, 1557, ("0.776 / 0.808",)),
        ("investigation-log.md", il, 1558, ("0.800 / 0.844",)),
        ("investigation-log.md", il, 1559, ("0.780 / 0.784",)),
        ("investigation-log.md", il, 1583, ("0.89", "0.815")),
        ("investigation-log.md", il, 1584, ("0.80", "0.824")),
        ("investigation-log.md", il, 1585, ("0.67", "0.844")),
        ("investigation-log.md", il, 1608, ("2.01 / 1.69 / 1.44",)),
        ("investigation-log.md", il, 1625, ("0.0443",)),
        ("investigation-log.md", il, 1626, ("0.0400", "0.903")),
        ("investigation-log.md", il, 1658, ("+0.1260",)),
        ("investigation-log.md", il, 1659, ("0.4472",)),
        ("coverage-study.md", cs, 942, ("0.903", "1.157", "1.156")),
        ("coverage-study.md", cs, 943, ("0.844", "0.780", "0.784")),
        ("construction-contrasts.md", cc, 363, ("+0.06987", "0.05487")),
    ]
    say(f"Cited lines at {TAG} (each checked for the quoted numbers):")
    for name, text, line, needles in cites:
        say(f"- {name}:{line}: {check_line(text, line, *needles).strip()[:150]}")
    say()
    sizes = SIZES
    q_rootn_bias = {600: 2.01, 1200: 1.69, 2400: 1.44}
    q_std_bias = {600: 0.89, 1200: 0.80, 2400: 0.67}
    say("C3c (250 replicates per cell and size, two batches):")
    say("| cell | n | root-n bias (A) | bias/sd (A) | coverage A / B | SE ratio A / B | coverage MC SE |")
    say("| --- | --- | --- | --- | --- | --- | --- |")
    cov = {("q-drift", 600): (0.792, 0.820), ("q-drift", 1200): (0.824, 0.880), ("q-drift", 2400): (0.844, 0.848),
           ("g-drift", 600): (0.776, 0.808), ("g-drift", 1200): (0.800, 0.844), ("g-drift", 2400): (0.780, 0.784)}  # fmt: skip
    for cell in ("q-drift", "g-drift"):
        for n in sizes:
            c = cov[(cell, n)]
            se_c = np.sqrt(c[0] * (1 - c[0]) / 250)
            if cell == "q-drift":
                rb, sb = f"{q_rootn_bias[n]:+.2f}", f"{q_std_bias[n]:.2f}"
                ratio = "0.903 / 0.903" if n == 2400 else "not stated"
            else:
                rb = "+3.09 (raw +0.1260)" if n == 600 else "not stated"
                sb = "not stated"
                ratio = "1.157 / 1.156" if n == 2400 else "not stated"
            say(f"| {cell} | {n:,} | {rb} | {sb} | {c[0]:.3f} / {c[1]:.3f} | {ratio} | {se_c:.3f} |")
    say()
    f5 = f5_rows()
    say("F5 terminal, `glm-pooled` arm (shipped construction, `glm` reductions), both cohorts pooled, from the per-replicate rows:")
    say("| cell | n | draws | raw bias | MC SE | root-n bias | empirical sd | SE ratio | coverage |")
    say("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    f5_stats = {}
    for cell in ("q-drift", "g-drift"):
        for n in (600, 2400):
            rows = f5[(cell, n)]
            err = np.array([r["psi"] - r["truth"] for r in rows])
            se = np.array([r["std_error"] for r in rows])
            sd = float(err.std(ddof=1))
            stats = {"bias": float(err.mean()), "mcse": sd / np.sqrt(err.size), "sd": sd, "draws": err.size}
            f5_stats[(cell, n)] = stats
            say(f"| {cell} | {n:,} | {err.size} | {stats['bias']:+.5f} | {stats['mcse']:.5f} | "
                f"{np.sqrt(n) * stats['bias']:+.3f} | {sd:.5f} | {se.mean() / sd:.3f} | "
                f"{np.mean(np.abs(err) <= 1.96 * se):.3f} |")  # fmt: skip
    say("F4 audit, `cleverly` arm (construction-contrasts.md:363): g-drift n = 2,400 bias +0.06987, se 0.05487 (no MC SE stated; a reading only).")
    say()
    n = 2400
    rn = np.sqrt(n)
    archived = {}
    # q-drift
    q_s1 = {"bias": 1.44 / rn, "mcse": 0.0443 / np.sqrt(250), "label": "S1 C3c batch A (docs)"}
    q_s2 = {"bias": f5_stats[("q-drift", n)]["bias"], "mcse": f5_stats[("q-drift", n)]["mcse"], "label": "S2 F5 glm-pooled"}
    # g-drift
    sd_f5 = f5_stats[("g-drift", n)]["sd"]
    b_std = wald_bias(0.780, 1.157)
    r_ = 1.96 * 1.157
    slope = norm.pdf(r_ - b_std) - norm.pdf(r_ + b_std)
    se_b = np.sqrt(0.780 * 0.220 / 250) / slope
    g_s1 = {"bias": f5_stats[("g-drift", n)]["bias"], "mcse": f5_stats[("g-drift", n)]["mcse"], "label": "S1 F5 glm-pooled"}
    g_s2 = {"bias": b_std * sd_f5, "mcse": se_b * sd_f5, "label": "S2 C3c Wald-implied (batch A) x F5 sd"}
    b_std_b = wald_bias(0.784, 1.156)
    say(f"C3c g-drift Wald-implied standardized bias at n = 2,400: batch A {b_std:.3f} +/- {se_b:.3f}, "
        f"batch B {b_std_b:.3f}; times the F5 pooled sd {sd_f5:.5f} gives raw {b_std * sd_f5:+.5f} +/- {se_b * sd_f5:.5f}.")  # fmt: skip
    say(f"C3c q-drift check: bias/sd 0.67 x sd 0.0443 = {0.67 * 0.0443:+.5f} against root-n 1.44 / sqrt(2400) = {1.44 / rn:+.5f}.")
    say()
    archived["q-drift"] = (q_s1, q_s2)
    archived["g-drift"] = (g_s1, g_s2)
    return archived


# ------------------------------------------------------------------ the rule


def rule(pred: float, arch: float, mcse: float) -> str:
    same_sign = np.sign(pred) == np.sign(arch)
    if same_sign and abs(pred - arch) <= max(0.30 * abs(arch), 2.0 * mcse):
        return "confirmed"
    outside = abs(pred - arch) > 2.0 * mcse
    if (abs(pred) < abs(arch) / 3.0 or not same_sign) and outside:
        return "refuted"
    return "inconclusive"


SENTENCE = {
    "refuted": "The shortfall is not attributed. The mechanism error does not account for it.",
    "inconclusive": "The shortfall is not attributed.",
    ("confirmed", "b"): "The shortfall is consistent with the smoothing bias of the mechanism fit, which adds a "
    "conditioning term of order `h_n²`. The fit also converges to an additive approximation of `g_0`, outside "
    "Theorem 1; that error is smaller.",
    ("confirmed", "a"): "The fit converges to an additive approximation of `g_0`, outside Theorem 1, and the "
    "shortfall is consistent with that error. The smoothing bias of the mechanism fit also adds a conditioning "
    "term of order `h_n²`; that error is smaller.",
    ("confirmed", "q"): "The shortfall is consistent with the mirror conditioning term of the outcome smoother.",
}


def decide(g: dict, q: dict, archived: dict) -> None:
    say("## Decision (rule in the header, n = 2,400, raw ATE bias; root-n in brackets)")
    rn = np.sqrt(2400)
    gp = g[2400]["primary"]
    preds = {"g-drift": (gp["ate"], g[2400]["mc"]["ate"]), "q-drift": (q[2400]["main"], q[2400]["mc"])}
    say("| cell | predicted (MC SE) | source | archived (MC SE) | |pred - arch| | max(0.30 |arch|, 2 MC SE) | |arch|/3 | outcome |")
    say("| --- | --- | --- | --- | --- | --- | --- | --- |")
    outcomes = {}
    for cell in ("g-drift", "q-drift"):
        pred, pmc = preds[cell]
        verdicts = []
        for src in archived[cell]:
            v = rule(pred, src["bias"], src["mcse"])
            verdicts.append(v)
            say(f"| {cell} | {pred:+.5f} ({pmc:.5f}) [{rn * pred:+.3f}] | {src['label']} | {src['bias']:+.5f} "
                f"({src['mcse']:.5f}) [{rn * src['bias']:+.3f}] | {abs(pred - src['bias']):.5f} | "
                f"{max(0.30 * abs(src['bias']), 2 * src['mcse']):.5f} | {abs(src['bias']) / 3:.5f} | {v} |")  # fmt: skip
        outcomes[cell] = verdicts[0] if len(set(verdicts)) == 1 else "inconclusive"
    say()
    for cell in ("g-drift", "q-drift"):
        outcome = outcomes[cell]
        if outcome == "confirmed":
            if cell == "g-drift":
                key = ("confirmed", "b" if abs(gp["b"]) >= abs(gp["a"]) else "a")
            else:
                key = ("confirmed", "q")
            sentence = SENTENCE[key]
        else:
            sentence = SENTENCE[outcome]
        say(f"**{cell}: {outcome}.** Selected docs sentence: \"{sentence}\"")
    if gp["a"] or gp["b"]:
        dom = "(b) smoothing bias" if abs(gp["b"]) >= abs(gp["a"]) else "(a) additive-projection error"
        say(f"g-drift dominant part of the prediction: {dom} (sqrt(n) (a) = {rn * gp['a']:+.4f}, sqrt(n) (b) = {rn * gp['b']:+.4f}).")
    pf, pfse = g[2400].get("per_fit_mean"), g[2400].get("per_fit_se")
    if pf is not None:
        alt = [rule(pf, src["bias"], src["mcse"]) for src in archived["g-drift"]]
        say(f"Sensitivity, not the rule: the g-drift mean over fits (random part of ghat_n included, linear Qr) is "
            f"{pf:+.5f} +/- {pfse:.5f} [{rn * pf:+.3f}]; against S1 / S2 the rule would read {alt[0]} / {alt[1]}.")  # fmt: skip
    say()
    say("Variance findings (decide nothing about attribution):")
    ratio, npts = q["ratio"], q["n_points"]
    if ratio < 1.0:
        say(f"- q-drift: predicted SE ratio {ratio:.3f} against the archived 0.903. The variance part accounts for "
            f"about {npts:.1f} of the {SHORTFALL_POINTS} points of shortfall.")  # fmt: skip
    else:
        say(f"- q-drift: predicted SE ratio {ratio:.3f} against the archived 0.903. The predicted ratio is 1 or more, "
            "so the variance part accounts for none of the shortfall (N <= 0).")  # fmt: skip
    say("- g-drift: archived SE ratio 1.157 (a long SE). This probe makes no g-drift SE-ratio prediction; a long SE cannot cause a shortfall.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=float, default=1.0, help="replicate multiplier (pilot runs only)")
    args = parser.parse_args()
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    say(f"sha256 {digest}  reviews/rm40/drift_attribution.py")
    say(f"# RM40b drift-attribution probe. Tag {TAG}. numpy {np.__version__}, python {sys.version.split()[0]}.")
    if args.scale != 1.0:
        say(f"# PILOT: replicate scale {args.scale}; not the committed run")
    say()
    start = time.time()
    t2 = load_archived()
    theta, icpt = step1(t2)
    wq = sobol_normal(20, QUAD_SEEDS[0])
    wq2 = sobol_normal(20, QUAD_SEEDS[1])
    sub = wq[: 2**17]
    g = step2(t2, wq, wq2, sub, args.scale)
    q = step3(t2, theta, icpt, wq, wq2, sub, args.scale)
    archived = step4()
    decide(g, q, archived)
    say()
    say(f"runtime {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
