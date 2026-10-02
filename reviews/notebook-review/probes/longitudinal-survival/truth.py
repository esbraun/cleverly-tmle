"""Independent truths from re-implemented structural equations (no library truth helper).

Large-n direct counterfactual simulation, plus an independent Gauss-Hermite quadrature of
the death-as-censoring (controlled-direct-effect) g-formula functional.
"""

import json

import numpy as np
from scipy.special import expit

N = 4_000_000
rng = np.random.default_rng(20261001)


def h1(w1, w2, a1):
    return expit(-1.1 - 0.7 * a1 + 0.35 * w1 - 0.25 * w2)


def h2(w1, w2, l2, a1, a2):
    return expit(-1.15 - 0.25 * a1 - 0.8 * a2 + 0.4 * l2 + 0.3 * w1 - 0.2 * w2 + 0.5 * np.tanh(l2))


def s1(w1, w2, a1):  # relapse share
    return expit(0.15 + 1.1 * a1 + 0.3 * w1 - 0.2 * w2)


def s2(w1, l2, a1, a2):
    return expit(0.1 + 0.5 * a1 + 0.9 * a2 - 0.25 * l2 + 0.2 * w1)


out = {}
w1 = rng.standard_normal(N)
w2 = rng.standard_normal(N)
noise = rng.standard_normal(N)
u_ev1, u_ev2, u_c1, u_c2 = (rng.random(N) for _ in range(4))
for label, a in (("always", 1.0), ("never", 0.0)):
    # survival law: counterfactual event times, censoring intervened away
    ev1 = u_ev1 < h1(w1, w2, a)
    l2 = 0.6 * w1 + 0.9 * a + noise
    ev2 = u_ev2 < h2(w1, w2, l2, a, a)
    out[f"surv {label} t1"] = ev1.mean()
    out[f"surv {label} t2"] = (ev1 | (~ev1 & ev2)).mean()
    rel1 = ev1 & (u_c1 < s1(w1, w2, a))
    dth1 = ev1 & ~rel1
    rel2 = ~ev1 & ev2 & (u_c2 < s2(w1, l2, a, a))
    dth2 = ~ev1 & ev2 & ~rel2
    out[f"cif relapse {label} t1"] = rel1.mean()
    out[f"cif relapse {label} t2"] = (rel1 | rel2).mean()
    out[f"cif death {label} t1"] = dth1.mean()
    out[f"cif death {label} t2"] = (dth1 | dth2).mean()
for k in ("surv {} t1", "surv {} t2", "cif relapse {} t1", "cif relapse {} t2",
          "cif death {} t1", "cif death {} t2"):
    out["diff " + k.format("a-n")] = out[k.format("always")] - out[k.format("never")]
mc_se = {k: float(np.sqrt(v * (1 - v) / N)) for k, v in out.items() if not k.startswith("diff")}

# Death-as-censoring functional, Young et al. ordering (death D_k precedes readmission Y_k):
# lambda_k = P(R_k = 1 | D_k = 0, history) = h s / (1 - h (1 - s)).
pts, wts = np.polynomial.hermite_e.hermegauss(60)
wts = wts / np.sqrt(2 * np.pi)
W1, W2, E = np.meshgrid(pts, pts, pts, indexing="ij")
M = wts[:, None, None] * wts[None, :, None] * wts[None, None, :]
for label, a in (("always", 1.0), ("never", 0.0)):
    H1 = h1(W1, W2, a)
    S1 = s1(W1, W2, a)
    L2 = 0.6 * W1 + 0.9 * a + E
    H2 = h2(W1, W2, L2, a, a)
    S2 = s2(W1, L2, a, a)
    lam1 = H1 * S1 / (1 - H1 * (1 - S1))
    lam2 = H2 * S2 / (1 - H2 * (1 - S2))
    out[f"cde_dfirst {label} t1"] = float(np.sum(M * lam1))
    out[f"cde_dfirst {label} t2"] = float(np.sum(M * (lam1 + (1 - lam1) * lam2)))
    # alternative ordering: readmission before death within an interval (cause-specific hazard)
    c1 = H1 * S1
    c2 = H2 * S2
    out[f"cde_rfirst {label} t2"] = float(np.sum(M * (c1 + (1 - c1) * c2)))
    # quadrature cross-check of total-effect CIF
    out[f"quad cif relapse {label} t2"] = float(np.sum(M * (H1 * S1 + (1 - H1) * H2 * S2)))
    out[f"quad surv {label} t2"] = float(np.sum(M * (H1 + (1 - H1) * H2)))
for k in ("cde_dfirst {} t1", "cde_dfirst {} t2", "cde_rfirst {} t2"):
    out["diff " + k.format("a-n")] = out[k.format("always")] - out[k.format("never")]

out = {k: float(v) for k, v in out.items()}
for k, v in out.items():
    print(f"{k:32s} {v:.4f}" + (f"  (MC se {mc_se[k]:.5f})" if k in mc_se else ""))
json.dump(out, open(".tmp/notebook-review/longitudinal-survival/truth.json", "w"), indent=1)
