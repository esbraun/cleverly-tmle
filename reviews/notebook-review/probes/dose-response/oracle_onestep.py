"""Coverage of the one-step estimate built from the true nuisances, at the page's n = 4,000.

For each draw the estimate is the mean of int_D w(a) Q(a, W) da + w(A) {Y - Q(A, W)} / g(A | W),
with the true outcome regression Q and the true density g of the engine (see oracle.py), and its
SE is the SD of those terms over sqrt(n). No model is fitted, so a shortfall here is the sample
size meeting a heavy-tailed influence curve, not a fitted model. 2,000 draws, seeds 40000-41999.

Run: python reviews/notebook-review/probes/dose-response/oracle_onestep.py > oracle_onestep.log
     JITTER=1.0 python reviews/notebook-review/probes/dose-response/oracle_onestep.py > oracle_onestep-jitter1.log
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from sweep import CURVE, JITTER, renewal_chance

HERE = Path(__file__).parent
truth = pd.read_csv(HERE / "truth.csv").set_index(["bandwidth", "dose"])["true"]
grid = np.linspace(*CURVE.support, 601)
step = grid[1] - grid[0]
PAIRS = [(2.5, 5.0), (5.0, 5.0), (5.0, 10.0), (5.0, 15.0), (5.0, 20.0)]
covers = {pair: [] for pair in PAIRS}
ratios = {pair: ([], []) for pair in PAIRS}
for seed in range(40000, 42000):
    rng = np.random.default_rng(seed)
    usage = rng.normal(size=4_000)
    tenure = rng.normal(size=4_000)
    mean = -0.3 - 0.45 * usage - 0.25 * tenure
    score = mean + rng.normal(scale=JITTER, size=4_000)
    dose = 40.0 / (1.0 + np.exp(-score))
    q_obs = renewal_chance(dose, usage, tenure)
    y = rng.binomial(1, q_obs)
    density = stats.norm.pdf((score - mean) / JITTER) / JITTER * 40.0 / (dose * (40.0 - dose))
    for h, a0 in PAIRS:
        weight_grid = CURVE.weight(h, a0, grid)
        keep = weight_grid != 0.0
        integral = step * sum(
            wa * renewal_chance(a, usage, tenure)
            for a, wa in zip(grid[keep], weight_grid[keep], strict=True)
        )
        terms = integral + CURVE.weight(h, a0, dose) * (y - q_obs) / density
        estimate, se = terms.mean(), terms.std(ddof=1) / np.sqrt(4_000)
        covers[(h, a0)].append(abs(estimate - truth[(h, a0)]) <= 1.959964 * se)
        ratios[(h, a0)][0].append(se)
        ratios[(h, a0)][1].append(estimate)
print(f"jitter SD {JITTER:g}; draws 2000")
for pair in PAIRS:
    ses, estimates = ratios[pair]
    print(
        f"h={pair[0]:g} dose={pair[1]:g}: coverage {np.mean(covers[pair]):.3f}, "
        f"mean SE / SD of estimates {np.mean(ses) / np.std(estimates, ddof=1):.3f}"
    )
