"""Recompute the true ATE, ATT, and ATC of ``make_instrument`` from its structural equations.

The equations are typed here from ``instrument_dgp`` without importing ``cleverly``: three
independent standard-normal covariates, ``P(A = 1 | W) = expit(0.8 W1 + 1.5 W2)``, and
``E[Y | A, W] = 1 + A + 1.5 W1 + 0.8 W3``.  The script draws 10^6 rows, evaluates both
counterfactual means on every row, and averages them over all rows, the treated, and the
controls.  It also reports the mean and SD of the observed score, and the population ``nu2`` of
the ATE representer, ``E[1 / (g (1 - g))]``, for two adjustment sets: all three covariates (the
Monte Carlo mean over the drawn rows), and the design-based set ``(W1, W3)``, whose propensity
``E[expit(0.8 W1 + 1.5 W2) | W1]`` the script integrates over ``W2`` by Gauss-Hermite
quadrature.  Output: ``truth.log``.
"""

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(20261002)
n = 1_000_000
w = rng.standard_normal((n, 3))
g = 1.0 / (1.0 + np.exp(-(0.8 * w[:, 0] + 1.5 * w[:, 1])))
a = rng.random(n) < g


def mean(treatment: float) -> np.ndarray:
    return 1.0 + treatment + 1.5 * w[:, 0] + 0.8 * w[:, 2]


effect = mean(1.0) - mean(0.0)
score = mean(a.astype(float)) + rng.standard_normal(n)
lines = [
    f"rows: {n}",
    f"ate: {effect.mean():.6f}",
    f"att: {effect[a].mean():.6f}",
    f"atc: {effect[~a].mean():.6f}",
    f"observed score mean {score.mean():.3f}, SD {score.std():.3f}, "
    f"min {score.min():.2f}, max {score.max():.2f}",
]
nodes, weights = np.polynomial.hermite_e.hermegauss(80)
weights = weights / weights.sum()
design_g = np.array([(weights / (1.0 + np.exp(-(0.8 * x + 1.5 * nodes)))).sum() for x in nodes])
lines.append(f"population nu2, all three covariates: {(1.0 / (g * (1.0 - g))).mean():.3f}")
design_nu2 = (weights / (design_g * (1.0 - design_g))).sum()
lines.append(f"population nu2, design-based set (W1, W3): {design_nu2:.3f}")
(HERE / "truth.log").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
print("\n".join(lines))
