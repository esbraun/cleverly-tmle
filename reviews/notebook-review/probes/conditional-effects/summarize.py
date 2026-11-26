"""Summarize ``sweep.csv`` into the repeated-draw numbers docs/examples/conditional-effects.ipynb quotes.

Usage: ``python summarize.py``.  Output: ``summary.log``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
rows = pd.read_csv(HERE / "sweep.csv")
n = len(rows)
lines = []


def share(mask) -> str:
    p = float(np.mean(mask))
    return f"{p:.3f} (MC SE {np.sqrt(p * (1 - p) / n):.3f}; {int(np.sum(mask))} of {n})"


lines.append(f"seeds {rows['seed'].min()} to {rows['seed'].max()}, {n} draws")
lines.append("")
lines.append("TMLE projection on Cosine(['age'], 3): coverage of the shape's own truth")
for name in ("intercept", "cos1", "cos2", "cos3"):
    lines.append(f"  {name} interval covers: {share(rows[f'{name}_covers'])}")
for age in range(20, 81, 10):
    lines.append(f"  pointwise interval at age {age} covers: {share(rows[f'point_covers_{age}'])}")
lines.append(f"  band covers all seven ages: {share(rows['band_covers'])}")
lines.append(f"  heterogeneity test p < 0.0001: {share(rows['heterogeneity_p'] < 1e-4)}")
lines.append("")
lines.append("EP curve at the declared schedule")
lines.append(f"  label supported: {share(rows['ep_status'] == 'supported')}")
q = rows["ep_rmse"].quantile([0.05, 0.5, 0.95]).round(3).tolist()
lines.append(f"  RMSE over ages 18-80, 5% / 50% / 95%: {q}")
lines.append("")
lines.append("Held-out rule score")
q = rows["cutoff"].quantile([0.05, 0.5, 0.95]).tolist()
lines.append(f"  cutoff (last age before the first-half curve first falls to the cost), 5% / 50% / 95%: {q}")
lines.append(f"  old rule (oldest age above the cost) at 80: {share(rows['old_cutoff'] == 80)}")
lines.append(f"  net-gain interval covers the true net gain: {share(rows['net_covers'])}")
bias = rows["net"] - rows["true_net"]
lines.append(f"  net-gain bias {bias.mean():+.4f} (MC SE {bias.std(ddof=1) / np.sqrt(n):.4f})")
(HERE / "summary.log").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
