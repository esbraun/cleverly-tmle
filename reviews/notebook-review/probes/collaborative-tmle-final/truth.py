"""Recompute the true ATE, ATT, and ATC of ``make_instrument`` from its structural equations.

The equations are typed here from ``instrument_dgp`` without importing ``cleverly``: three
independent standard-normal covariates, ``P(A = 1 | W) = expit(0.8 W1 + 1.5 W2)``, and
``E[Y | A, W] = 1 + A + 1.5 W1 + 0.8 W3``.  The script draws 10^6 rows, evaluates both
counterfactual means on every row, and averages them over all rows, the treated, and the
controls.  It also reports the mean and SD of the observed score.  Output: ``truth.log``.
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
(HERE / "truth.log").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
print("\n".join(lines))
