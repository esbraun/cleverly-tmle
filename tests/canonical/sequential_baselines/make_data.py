"""Write ``data.csv``, the panel that R ``ltmle`` and cleverly both fit for the W9 fixture.

Run from the repository root: ``python -m tests.canonical.sequential_baselines.make_data``.
The law is nonsaturated, so the parity covers the GLM fits, not only the cell means.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

N = 1_000
SEED = 20261014


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def draw(n: int = N, seed: int = SEED) -> pd.DataFrame:
    """Two nodes, two baseline covariates, a continuous L2, censoring, a binary outcome."""
    rng = np.random.default_rng(seed)
    w1 = rng.normal(size=n)
    w2 = rng.binomial(1, 0.4, n).astype(float)
    a1 = rng.binomial(1, _expit(0.1 + 0.6 * w1 - 0.4 * w2))
    c1 = rng.binomial(1, _expit(2.2 + 0.3 * w1 - 0.3 * a1))
    l2 = 0.5 * w1 - 0.4 * a1 + 0.3 * w2 + rng.normal(size=n)
    a2 = rng.binomial(1, _expit(-0.2 + 0.5 * l2 + 0.8 * a1))
    c2 = rng.binomial(1, _expit(2.0 + 0.3 * l2 - 0.2 * a2))
    y = rng.binomial(1, _expit(-0.3 + 0.4 * w1 + 0.3 * w2 + 0.5 * l2 - 0.6 * a2 - 0.3 * a1))
    stays = c1 == 1
    observed = stays & (c2 == 1)
    return pd.DataFrame(
        {
            "W1": np.round(w1, 6),
            "W2": w2,
            "A1": a1.astype(float),
            "C1": c1.astype(float),
            "L2": np.where(stays, np.round(l2, 6), np.nan),
            "A2": np.where(stays, a2, np.nan),
            "C2": np.where(stays, c2, np.nan),
            "Y": np.where(observed, y, np.nan),
        }
    )


if __name__ == "__main__":
    draw().to_csv(Path(__file__).with_name("data.csv"), index=False, float_format="%.6f")
