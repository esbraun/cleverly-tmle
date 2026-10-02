"""Independent 10^7-draw Monte Carlo of the navigation_data ATE from the structural equations."""
import numpy as np
from scipy.special import expit

rng = np.random.default_rng(20261001)
total = 0
sums = np.zeros(2)
sq = 0.0
inv_g = []
for _ in range(10):
    w = rng.normal(size=(1_000_000, 4))
    base = (
        -0.5
        + 0.8 * np.sin(1.5 * w[:, 0])
        + 0.5 * np.tanh(w[:, 1] ** 2 - 1.0)
        - 0.4 * np.tanh(w[:, 2] * w[:, 3])
        + 0.3 * np.tanh(np.abs(w[:, 3]))
    )
    eff = 0.9 + 0.4 * np.tanh(w[:, 0]) - 0.3 * (w[:, 1] > 0)
    d = expit(base + eff) - expit(base)
    sums += [d.sum(), 0]
    sq += (d ** 2).sum()
    total += len(d)
    g = expit(0.6 * w[:, 0] - 0.4 * w[:, 1] ** 2 + 0.5 * w[:, 1] * w[:, 2] + 0.3 * (w[:, 3] > 0))
    inv_g.append(((g < 0.0114) | (g > 1 - 0.0114)).mean())
mean = sums[0] / total
sd = np.sqrt(sq / total - mean ** 2)
print(f"ATE MC = {mean:.6f}  MC SE = {sd / np.sqrt(total):.2e}  (n={total})")
print(f"P(g < 0.0114 or g > 0.9886) under the true g = {np.mean(inv_g):.4%}")
