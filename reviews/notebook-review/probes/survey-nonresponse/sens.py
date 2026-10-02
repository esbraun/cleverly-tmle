"""Probe the missingness tilt: (1) bias of the tilted plug-in from a misspecified Q*,
(2) dependence of tipping gamma on n through the data-dependent min-max scale,
(3) composite death + Delta: does the declared MAR hold?"""
import warnings

import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import ATE, CausalStudy, CrossFitting, ModelSpec, PointTreatment, Runtime, TMLEMethod
from cleverly.datasets import make_missing_outcome
from cleverly.sensitivity.missingness import missingness_tilt, tipping_gamma

warnings.filterwarnings("ignore")
cov = ("W1", "W2", "W3")
AG = {0: 0.0, 1: -1.0}

rng = np.random.default_rng(999)
M = 2_000_000
W = rng.normal(size=(M, 3))
w1, w2, w3 = W.T
Q = lambda a: 1.0 + 1.2 * a + 0.9 * w1 + 0.6 * w2 - 0.4 * w3 + (1.1 * np.tanh(1.5 * w1) + 0.8 * w2**2 - 0.9 * a * w1)
PI = lambda a: expit(1.2 + 0.6 * a - 1.4 * w1 + 0.3 * w3)


def true_tilted_ate(gamma, lo, hi):
    """Population pattern-mixture ATE on the same scaled logit tilt (arm 1 shifted by -gamma)."""
    q1 = np.clip((Q(1) - lo) / (hi - lo), 1e-6, 1 - 1e-6)
    q0 = (Q(0) - lo) / (hi - lo)
    p1 = PI(1)
    full1 = p1 * q1 + (1 - p1) * expit(logit(q1) - gamma)
    return float(np.mean(full1 - q0)) * (hi - lo)


def method(seed):
    return TMLEMethod(
        models=ModelSpec(outcome_learner=LinearRegression(n_jobs=1),
                         treatment_learner=LogisticRegression(max_iter=1000, random_state=seed),
                         missingness_learner=LogisticRegression(max_iter=1000, random_state=seed)),
        cross_fitting=CrossFitting(enabled=False), runtime=Runtime(random_state=seed, n_jobs=1))


def fit(n, seed):
    f, _ = make_missing_outcome(n=n, seed=seed, strength=2.0)
    s = CausalStudy(f, design=PointTreatment(outcome="Y", treatment="A", adjustment=cov, missingness="Delta"))
    return s.identify(ATE(reference=0)).estimate(method=method(seed))


print("(1) tilted plug-in vs population tilted ATE on the same scale")
diffs = {1.0: [], 2.0: []}
for seed in range(2000, 2040):
    r = fit(4000, seed)
    sc = r.nuisance.scaler
    curve = missingness_tilt(r, [1.0, 2.0], estimands=["ate"], arm_gamma=AG)
    for g, psi in zip(curve["gamma"], curve["psi"]):
        t = true_tilted_ate(g, sc.lower, sc.upper)
        diffs[g].append(psi - t)
    if seed == 2000:
        print("  seed 2000", list(curve["psi"]), [true_tilted_ate(g, sc.lower, sc.upper) for g in (1.0, 2.0)])
for g, d in diffs.items():
    d = np.asarray(d)
    print(f"  gamma={g}: mean(est - pop tilted)={d.mean():.4f}  sd={d.std(ddof=1):.4f}  mcse={d.std(ddof=1)/np.sqrt(len(d)):.4f}")
print("  seed 71 scale: pop tilted ATE at gamma=1,2:",
      true_tilted_ate(1.0, -6.56, 13.49), true_tilted_ate(2.0, -6.56, 13.49))
# population tipping gamma on the seed-71 scale
from scipy.optimize import brentq
print("  population tipping gamma on seed-71 scale:", brentq(lambda g: true_tilted_ate(g, -6.56, 13.49), 0.1, 5))

print("(2) tipping gamma and scale range vs n")
for n in (1000, 4000, 16000, 64000):
    tg, rg = [], []
    for seed in range(3000, 3010):
        r = fit(n, seed)
        sc = r.nuisance.scaler
        tg.append(tipping_gamma(r, arm_gamma=AG))
        rg.append(sc.upper - sc.lower)
    print(f"  n={n}: mean range={np.mean(rg):.2f}  mean tipping gamma={np.mean(tg):.3f}  "
          f"sd={np.std(tg, ddof=1):.3f}  pop tipping on mean range={brentq(lambda g: true_tilted_ate(g, -np.mean(rg)/2+1.5, np.mean(rg)/2+1.5), 0.05, 8):.3f}")

print("(3) composite death scored worst and counted as observed")
pD = lambda a: expit(-3.5 + 0.8 * w1 - 0.5 * a)  # death before day 30
WORST = -7.0
for a in (0, 1):
    d = pD(a)
    pi = PI(a)
    truth = np.mean(d * WORST + (1 - d) * Q(a))
    # identification functional the notebook declares: E_W[E(Y | A=a, Delta=1, W)]
    num = d * WORST + (1 - d) * pi * Q(a)
    den = d + (1 - d) * pi
    ident = np.mean(num / den)
    print(f"  a={a}: P(death)={d.mean():.4f}  true E[Y^a]={truth:.4f}  declared functional={ident:.4f}  bias={ident-truth:.4f}")
true_ate = np.mean(pD(1) * WORST + (1 - pD(1)) * Q(1)) - np.mean(pD(0) * WORST + (1 - pD(0)) * Q(0))
id_ate = np.mean((pD(1) * WORST + (1 - pD(1)) * PI(1) * Q(1)) / (pD(1) + (1 - pD(1)) * PI(1))) - \
    np.mean((pD(0) * WORST + (1 - pD(0)) * PI(0) * Q(0)) / (pD(0) + (1 - pD(0)) * PI(0)))
print(f"  ATE truth={true_ate:.4f}  declared functional={id_ate:.4f}  bias={id_ate-true_ate:.4f}")
