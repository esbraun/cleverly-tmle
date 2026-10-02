"""Recompute the printed truths of cross-fitting.ipynb from the structural equations.

The equations are typed here from ``nonlinear_dgp``, ``nonlinear_bounded_dgp`` and
``clustered_dgp(family="binomial")`` in ``src/cleverly/datasets/synthetic.py``.  This script
does not import ``cleverly``, so it is an independent check of ``navigation_data(...)[1]`` and
``make_clustered(..., family="binomial")[1]``.

Usage: ``python truth.py``.  Output: ``truth.log``.

- The navigation law: Monte Carlo over 10^7 standard-normal draws of W1 to W4, in 20 chunks.
- The team law: W1, W2 and the shared team factor u are independent standard normals.  The
  ATE integrates over all three.  A team's mean effect integrates over W1 and W2 at its own u;
  the share of teams with a negative mean effect is P(effect(u) < 0) under u ~ N(0, 1).
"""

import numpy as np
from scipy.special import expit
from scipy.stats import norm

CHUNKS, SIZE = 20, 500_000


def propensity(w):
    u = 0.6 * w[:, 0] - 0.4 * w[:, 1] ** 2 + 0.5 * w[:, 1] * w[:, 2] + 0.3 * (w[:, 3] > 0)
    return 0.05 + 0.90 * expit(u)


def outcome_mean(w, a):
    baseline = (
        -0.5
        + 0.8 * np.sin(1.5 * w[:, 0])
        + 0.5 * np.tanh(w[:, 1] ** 2 - 1.0)
        - 0.4 * np.tanh(w[:, 2] * w[:, 3])
        + 0.3 * np.tanh(np.abs(w[:, 3]))
    )
    effect = 0.9 + 0.4 * np.tanh(w[:, 0]) - 0.3 * (w[:, 1] > 0)
    return expit(baseline + effect * a)


def team_mean(w1, w2, u, a):
    return expit(-0.4 + 0.8 * a + 0.5 * w1 + 0.3 * w2 + 0.6 * u + 8.0 * a * u)


rng = np.random.default_rng(20261001)
rows = []
for _ in range(CHUNKS):
    w = rng.standard_normal((SIZE, 4))
    g = propensity(w)
    m1, m0 = outcome_mean(w, 1.0), outcome_mean(w, 0.0)
    t = rng.standard_normal((SIZE, 3))
    effect = team_mean(t[:, 0], t[:, 1], t[:, 2], 1.0) - team_mean(t[:, 0], t[:, 1], t[:, 2], 0.0)
    rows.append(
        {
            "navigation ate": (m1 - m0).mean(),
            "navigation nu2": (1.0 / g + 1.0 / (1.0 - g)).mean(),
            "navigation share g < 0.10": (g < 0.10).mean(),
            "team ate": effect.mean(),
        }
    )
print("Monte Carlo, 20 chunks of 500000")
for key in rows[0]:
    values = np.array([row[key] for row in rows])
    se = values.std(ddof=1) / np.sqrt(len(values))
    print(f"  {key:28s} {values.mean():.6f}  (MC SE {se:.1e})")

# A team's mean effect as a function of its shared factor u, by Gauss-Hermite quadrature over
# W1 and W2, then the share of u ~ N(0, 1) where that effect is negative.
nodes, weights = np.polynomial.hermite_e.hermegauss(80)
weights = weights / weights.sum()
w1, w2 = np.meshgrid(nodes, nodes, indexing="ij")
pair = np.outer(weights, weights)


def team_effect(u):
    return float(np.sum(pair * (team_mean(w1, w2, u, 1.0) - team_mean(w1, w2, u, 0.0))))


grid = np.linspace(-1.0, 1.0, 200_001)
values = np.array([team_effect(u) for u in grid[::1000]])
crossing = grid[::1000][np.argmax(values >= 0.0)]
low, high = crossing - 0.01, crossing + 0.01
for _ in range(60):
    mid = 0.5 * (low + high)
    low, high = (mid, high) if team_effect(mid) < 0.0 else (low, mid)
root = 0.5 * (low + high)
print()
print("Team law, quadrature over W1 and W2 at each value of the shared team factor u")
print(f"  team mean effect is negative for u below {root:.4f}")
print(f"  share of teams with a negative mean effect, P(u < {root:.4f}): {norm.cdf(root):.4f}")
for q in (0.05, 0.5, 0.95):
    print(f"  team mean effect at the {q:.0%} quantile of u: {team_effect(norm.ppf(q)):+.4f}")
