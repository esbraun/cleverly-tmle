"""Recompute the printed truths of dr-tmle.ipynb from the structural equations.

The equations are typed here from ``nonlinear_dgp`` and ``nonlinear_bounded_dgp`` in
``src/cleverly/datasets/synthetic.py``.  This script does not import ``cleverly``, so it is an
independent check of ``navigation_data(...)[1]``.

Usage: ``python truth.py``.  Output: ``truth.log``.  Monte Carlo over 10^7 standard-normal draws
of W1 to W4, in 20 chunks.

It also reports the range of the true propensity, the share of the law below the page's
automatic truncation bound 5 / (sqrt(2000) ln 2000) = 0.01471, and the population limit of the
page's main-effects logistic assignment model (fitted by Newton iterations on one chunk), with the
mean absolute distance between that limit and the true propensity.
"""

import numpy as np
from scipy.special import expit

CHUNKS, SIZE = 20, 500_000
BOUND = 5.0 / (np.sqrt(2000.0) * np.log(2000.0))


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


def main_effects_limit(w, g):
    """Population logistic regression of A on (1, W1..W4), weighting by the true propensity."""
    x = np.column_stack([np.ones(len(w)), w])
    beta = np.zeros(x.shape[1])
    for _ in range(25):
        p = expit(x @ beta)
        gradient = x.T @ (g - p)
        hessian = (x * (p * (1 - p))[:, None]).T @ x
        beta += np.linalg.solve(hessian, gradient)
    return beta


rng = np.random.default_rng(20261002)
rows = []
beta = None
for chunk in range(CHUNKS):
    w = rng.standard_normal((SIZE, 4))
    g = propensity(w)
    if beta is None:
        beta = main_effects_limit(w, g)
    limit = expit(np.column_stack([np.ones(len(w)), w]) @ beta)
    m1, m0 = outcome_mean(w, 1.0), outcome_mean(w, 0.0)
    rows.append(
        {
            "ey1": m1.mean(),
            "ey0": m0.mean(),
            "ate": (m1 - m0).mean(),
            "nu2 = E[1/g + 1/(1-g)]": (1.0 / g + 1.0 / (1.0 - g)).mean(),
            "share g < bound": (g < BOUND).mean(),
            "E|g* - g|": np.abs(limit - g).mean(),
            "min g": g.min(),
            "max g": g.max(),
        }
    )
print(f"Monte Carlo, {CHUNKS} chunks of {SIZE}; truncation bound {BOUND:.5f}")
for key in rows[0]:
    values = np.array([row[key] for row in rows])
    if key == "min g":
        print(f"  {key:28s} {values.min():.6f}")
    elif key == "max g":
        print(f"  {key:28s} {values.max():.6f}")
    else:
        se = values.std(ddof=1) / np.sqrt(len(values))
        print(f"  {key:28s} {values.mean():.6f}  (MC SE {se:.1e})")
print("main-effects logistic limit (intercept, W1..W4):", np.round(beta, 3).tolist())
