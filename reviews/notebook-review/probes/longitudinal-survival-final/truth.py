"""Recompute the truths of docs/examples/longitudinal-survival.ipynb without cleverly.

The structural equations are read from ``make_longitudinal_survival`` and
``make_longitudinal_competing`` in ``src/cleverly/datasets/longitudinal.py`` and written out
here. ``W1, W2, U ~ N(0, 1)`` are independent, and ``L2 = 0.6 W1 + 0.9 A1 + U``.

- all-cause hazard, period 1: ``logit h1 = -1.1 - 0.7 A1 + 0.35 W1 - 0.25 W2``
- all-cause hazard, period 2: ``logit h2 = -1.15 - 0.25 A1 - 0.8 A2 + 0.4 L2 + 0.3 W1 - 0.2 W2
  + 0.5 tanh(L2)``
- readmission share of an event, period 1: ``logit s1 = 0.15 + 1.1 A1 + 0.3 W1 - 0.2 W2``
- readmission share of an event, period 2: ``logit s2 = 0.1 + 0.5 A1 + 0.9 A2 - 0.25 L2 + 0.2 W1``

Censoring (loss of tracking, disenrollment) does not enter a counterfactual quantity under
"every patient stays observed". The death-as-censoring functional is the g-formula with death
as a censoring node placed before readmission in each period (Young et al. 2020, ordering
``(D_k, Y_k)``). Its readmission hazard among those alive is ``h s / (1 - h (1 - s))``. The
readmission-first ordering uses the cause-specific hazard ``h s`` instead.

Two methods: a 60-point product Gauss-Hermite rule, and a 4,000,000-unit direct
counterfactual simulation of the plan-exit and cause-specific risks.

Usage: ``python truth.py > truth.log``.
"""

import numpy as np
from scipy.special import expit


def h1(w1, w2, a):
    return expit(-1.1 - 0.7 * a + 0.35 * w1 - 0.25 * w2)


def h2(w1, w2, l2, a1, a2):
    return expit(
        -1.15 - 0.25 * a1 - 0.8 * a2 + 0.4 * l2 + 0.3 * w1 - 0.2 * w2 + 0.5 * np.tanh(l2)
    )


def s1(w1, w2, a):
    return expit(0.15 + 1.1 * a + 0.3 * w1 - 0.2 * w2)


def s2(w1, l2, a1, a2):
    return expit(0.1 + 0.5 * a1 + 0.9 * a2 - 0.25 * l2 + 0.2 * w1)


points, weights = np.polynomial.hermite_e.hermegauss(60)
weights = weights / np.sqrt(2.0 * np.pi)
W1, W2, U = np.meshgrid(points, points, points, indexing="ij")
MASS = weights[:, None, None] * weights[None, :, None] * weights[None, None, :]

quad = {}
for plan, a in (("always", 1.0), ("never", 0.0)):
    H1, S1 = h1(W1, W2, a), s1(W1, W2, a)
    L2 = 0.6 * W1 + 0.9 * a + U
    H2, S2 = h2(W1, W2, L2, a, a), s2(W1, L2, a, a)
    quad[f"exit {plan} t=1"] = np.sum(MASS * H1)
    quad[f"exit {plan} t=2"] = np.sum(MASS * (H1 + (1 - H1) * H2))
    quad[f"readmission {plan} t=1"] = np.sum(MASS * H1 * S1)
    quad[f"readmission {plan} t=2"] = np.sum(MASS * (H1 * S1 + (1 - H1) * H2 * S2))
    quad[f"death {plan} t=1"] = np.sum(MASS * H1 * (1 - S1))
    quad[f"death {plan} t=2"] = np.sum(MASS * (H1 * (1 - S1) + (1 - H1) * H2 * (1 - S2)))
    lam1 = H1 * S1 / (1 - H1 * (1 - S1))
    lam2 = H2 * S2 / (1 - H2 * (1 - S2))
    quad[f"death-first {plan} t=1"] = np.sum(MASS * lam1)
    quad[f"death-first {plan} t=2"] = np.sum(MASS * (lam1 + (1 - lam1) * lam2))
    c1, c2 = H1 * S1, H2 * S2
    quad[f"readmission-first {plan} t=2"] = np.sum(MASS * (c1 + (1 - c1) * c2))
for stem in ("exit", "readmission", "death", "death-first"):
    for horizon in (1, 2):
        quad[f"{stem} always-never t={horizon}"] = (
            quad[f"{stem} always t={horizon}"] - quad[f"{stem} never t={horizon}"]
        )
quad["readmission-first always-never t=2"] = (
    quad["readmission-first always t=2"] - quad["readmission-first never t=2"]
)
for plan in ("always", "never"):
    for horizon in (1, 2):
        quad[f"event-free {plan} t={horizon}"] = 1 - (
            quad[f"readmission {plan} t={horizon}"] + quad[f"death {plan} t={horizon}"]
        )

N = 4_000_000
rng = np.random.default_rng(20261002)
w1, w2, u = rng.standard_normal((3, N))
e1, e2, c1u, c2u = rng.random((4, N))
sim = {}
for plan, a in (("always", 1.0), ("never", 0.0)):
    ev1 = e1 < h1(w1, w2, a)
    l2 = 0.6 * w1 + 0.9 * a + u
    ev2 = ~ev1 & (e2 < h2(w1, w2, l2, a, a))
    rel1 = ev1 & (c1u < s1(w1, w2, a))
    rel2 = ev2 & (c2u < s2(w1, l2, a, a))
    sim[f"exit {plan} t=1"] = ev1.mean()
    sim[f"exit {plan} t=2"] = (ev1 | ev2).mean()
    sim[f"readmission {plan} t=2"] = (rel1 | rel2).mean()
    sim[f"death {plan} t=2"] = ((ev1 & ~rel1) | (ev2 & ~rel2)).mean()

print("quadrature, 60 nodes per axis")
for key, value in quad.items():
    print(f"  {key:40s} {value: .6f}")
print(f"simulation, {N} units (Monte Carlo SE at most 0.00025)")
for key, value in sim.items():
    print(f"  {key:40s} {value: .6f}   quadrature {quad[key]: .6f}")
