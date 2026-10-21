"""Population nu2 of the page's law, for the reading of Step 10.  Not a registered study.

nu2 = E[1/g(W) + 1/(1 - g(W))], the second moment of the ATE Riesz representer, by Monte Carlo
with 4,000,000 draws of W.  The script reports it for the full adjustment set, with g clipped at
the page's default bounds, and for the set without staffing (g averaged over staffing by
Gauss-Hermite quadrature).
Run: ``python reviews/notebook-review/probes/collaborative-tmle-oat/truth.py``.
"""

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import expit

rng = np.random.default_rng(20261015)
readiness, staffing = rng.standard_normal((2, 4_000_000))
g = expit(0.5 * readiness + 1.6 * staffing)
clipped = np.clip(g, 0.01765, 0.9823)
nodes, weights = hermegauss(80)
weights = weights / weights.sum()
without = expit(0.5 * readiness[:, None] + 1.6 * nodes[None, :]) @ weights
for label, value in (
    ("full set", g),
    ("full set, clipped at the page bounds", clipped),
    ("without staffing", without),
):
    nu2 = np.mean(1.0 / value + 1.0 / (1.0 - value))
    print(f"{label}: nu2 = {nu2:.3f}")
