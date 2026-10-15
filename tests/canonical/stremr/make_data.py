"""Write ``long.csv``, the long panel that R ``stremr`` and cleverly both fit.

Run from the repository root: ``python -m tests.canonical.stremr.make_data``.

Two decision times, a survival event after each decision, and no censoring.  ``stremr``
orders a row as ``L, C, A, Y`` and cleverly as ``L, A, C, Y``, so the two factor a censoring
node differently.  Without censoring the factorizations agree, and the points can match
exactly.  ``Lprev`` and ``Aprev`` give ``stremr``'s per-row formulas the history that
cleverly's node-2 design holds.  They are zero at the first time, where both fits alias them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

N = 2_000
SEED = 20261015


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def draw(n: int = N, seed: int = SEED) -> pd.DataFrame:
    """One row per unit and decision time, until the unit's event."""
    rng = np.random.default_rng(seed)
    w = np.round(rng.normal(size=n), 6)
    l1 = np.round(0.5 * w + rng.normal(size=n), 6)
    a1 = rng.binomial(1, _expit(0.1 + 0.5 * w + 0.4 * l1))
    y1 = rng.binomial(1, _expit(-1.6 + 0.4 * w + 0.4 * l1 - 0.5 * a1))
    l2 = np.round(0.4 * w + 0.3 * l1 - 0.4 * a1 + rng.normal(size=n), 6)
    a2 = rng.binomial(1, _expit(-0.2 + 0.3 * w + 0.5 * l2 + 0.7 * a1))
    y2 = rng.binomial(1, _expit(-1.3 + 0.3 * w + 0.5 * l2 - 0.6 * a2 - 0.2 * a1))
    first = pd.DataFrame(
        {"ID": np.arange(n), "t": 0, "W": w, "L": l1, "Lprev": 0.0, "Aprev": 0.0, "A": a1, "Y": y1}
    )
    second = pd.DataFrame(
        {"ID": np.arange(n), "t": 1, "W": w, "L": l2, "Lprev": l1, "Aprev": a1, "A": a2, "Y": y2}
    )[y1 == 0]
    long = pd.concat([first, second]).sort_values(["ID", "t"]).reset_index(drop=True)
    long["C"] = 0
    return long


if __name__ == "__main__":
    draw().to_csv(Path(__file__).with_name("long.csv"), index=False, float_format="%.6f")
