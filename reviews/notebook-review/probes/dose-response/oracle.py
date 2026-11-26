"""Oracle standard error of each pair: the SD of the efficient influence curve under the true law.

D = w(A) {Y - Q(A, W)} / g(A | W) + int_D w(a) Q(a, W) da - theta_h(a0), with the true outcome
regression Q and the true conditional density g of the page's generator: the engine's score is
N(m(W), s^2) with s the jitter SD (0.8, or 1.0 with JITTER=1.0) and the discount is 40 / (1 + exp(-score)), so
g(a | W) = phi((logit(a / 40) - m(W)) / 0.8) / 0.8 * 40 / (a (40 - a)).
Monte Carlo over 2,000,000 rows (seed 20261127); prints SD(D) / sqrt(4,000) per pair, the SE a
fit of 4,000 rows should report, beside the sweep's mean reported SE and the SD of its estimates.

Run: python reviews/notebook-review/probes/dose-response/oracle.py > oracle.log
     JITTER=1.0 python reviews/notebook-review/probes/dose-response/oracle.py > oracle-jitter1.log
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from sweep import CURVE, JITTER, SWEEP, renewal_chance

HERE = Path(__file__).parent
N = 2_000_000

rng = np.random.default_rng(20261127)
usage = rng.normal(size=N)
tenure = rng.normal(size=N)
mean = -0.3 - 0.45 * usage - 0.25 * tenure
score = mean + rng.normal(scale=JITTER, size=N)
dose = 40.0 / (1.0 + np.exp(-score))
q_obs = renewal_chance(dose, usage, tenure)
y = rng.binomial(1, q_obs)
density = stats.norm.pdf((score - mean) / JITTER) / JITTER * 40.0 / (dose * (40.0 - dose))
grid = np.linspace(*CURVE.support, 601)
sweep = pd.read_csv(SWEEP)
rows = []
for h, a0 in CURVE.pairs:
    weight_grid = CURVE.weight(h, a0, grid)
    keep = weight_grid != 0.0
    integral = np.zeros(N)
    for a, wa in zip(grid[keep], weight_grid[keep], strict=True):
        integral += wa * renewal_chance(a, usage, tenure)
    integral *= grid[1] - grid[0]
    w_obs = CURVE.weight(h, a0, dose)
    influence = w_obs * (y - q_obs) / density + integral
    cell = sweep[(sweep["bandwidth"] == h) & (sweep["dose"] == a0)]
    rows.append(
        {
            "bandwidth": h,
            "dose": a0,
            "oracle_se": influence.std() / np.sqrt(4_000),
            "mean_reported_se": cell["se"].mean(),
            "sd_of_estimates": cell["estimate"].std(ddof=1),
        }
    )
print(f"draws in sweep.csv: {sweep['seed'].nunique()}")
print(pd.DataFrame(rows).round(4).to_string(index=False))
