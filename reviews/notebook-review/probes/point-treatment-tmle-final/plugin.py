"""Seed sweep of the Step 7 g-computation plug-in of docs/examples/point-treatment-tmle.ipynb.

Each seed ``s`` draws ``navigation_data(n=3000, seed=s)`` and computes the notebook's plug-in:
a main-terms linear regression of the score on the offer and the four covariates, averaged
over the rows at each arm.  The seeds are the ones ``sweep.py`` uses (6000 to 6199), so the
TMLE rows of ``summary.log`` and this plug-in describe the same draws.

Usage: ``python plugin.py``.  Output: ``plugin.log``.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

from cleverly.datasets import navigation_data

HERE = Path(__file__).resolve().parent
COVARIATES = ["discharge_risk", "prior_utilization", "medication_burden", "age"]
SEEDS = range(6000, 6200)


def plug_in(seed: int) -> tuple[float, float]:
    frame, truth = navigation_data(n=3_000, seed=seed)
    features = frame[["transition_navigation", *COVARIATES]]
    model = LinearRegression().fit(features, frame["transition_score"])
    offered = model.predict(features.assign(transition_navigation=1.0))
    not_offered = model.predict(features.assign(transition_navigation=0.0))
    return float(np.mean(offered - not_offered)), truth["ate"]


def main() -> None:
    rows = np.array([plug_in(seed) for seed in SEEDS])
    errors = rows[:, 0] - rows[:, 1]
    count = len(errors)
    lines = [
        f"seeds {SEEDS.start} to {SEEDS.stop - 1}, {count} draws",
        f"plug-in, linear Q: bias {errors.mean():+.4f} (MC SE {errors.std(ddof=1) / np.sqrt(count):.4f})",
        f"  empirical SD {rows[:, 0].std(ddof=1):.4f}; mean |miss| {np.abs(errors).mean():.4f}",
        f"  error below zero: {np.mean(errors < 0):.3f} ({int(np.sum(errors < 0))} of {count})",
    ]
    (HERE / "plugin.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
