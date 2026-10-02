"""Recompute the printed truths of point-treatment-tmle.ipynb from the structural equations.

The equations are typed here from the docstrings of ``nonlinear_dgp`` and
``nonlinear_bounded_dgp`` in ``src/cleverly/datasets/synthetic.py``.  This script does not import
``cleverly``, so it is an independent check of ``navigation_data(...)[1]``.

Usage: ``python truth.py``.  Output: ``truth.log``.  Monte Carlo over 10^7 standard-normal draws
of W1 to W4, in 20 chunks of 5 x 10^5, with the chunk-to-chunk standard error.
"""

import numpy as np
from scipy.special import expit

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


rng = np.random.default_rng(20261001)
rows = []
for _ in range(CHUNKS):
    w = rng.standard_normal((SIZE, 4))
    g = propensity(w)
    m1, m0 = outcome_mean(w, 1.0), outcome_mean(w, 0.0)
    rows.append(
        {
            "ey1": m1.mean(),
            "ey0": m0.mean(),
            "ate": (m1 - m0).mean(),
            "att": np.average(m1 - m0, weights=g),
            "atc": np.average(m1 - m0, weights=1.0 - g),
            "nu2": (1.0 / g + 1.0 / (1.0 - g)).mean(),
            "share g < 0.10": (g < 0.10).mean(),
            "share g > 0.90": (g > 0.90).mean(),
            "share g < 0.20": (g < 0.20).mean(),
            "min g": g.min(),
            "max g": g.max(),
        }
    )
for key in rows[0]:
    values = np.array([row[key] for row in rows])
    if key.startswith("min"):
        print(f"{key:16s} {values.min():.6f}")
    elif key.startswith("max"):
        print(f"{key:16s} {values.max():.6f}")
    else:
        se = values.std(ddof=1) / np.sqrt(len(values))
        print(f"{key:16s} {values.mean():.6f}  (MC SE {se:.1e})")
