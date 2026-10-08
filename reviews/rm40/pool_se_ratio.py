# A probe, not a registered study. It reads the committed `drdr_summary.log` only.
#
# Usage: `python reviews/rm40/pool_se_ratio.py`
#
# It pools the in-sample ATE SE ratio of the `qwrong_gglm` setting (outcome regression wrong,
# treatment mechanism a correctly specified logistic GLM) over n = 1,000, 4,000 and 16,000. The
# pool is inverse-variance weighted with the log's Monte Carlo standard errors. The log prints
# three decimals, so the pooled standard error carries that rounding.
"""Pool the in-sample ATE SE ratio of the parametric-mechanism cells of the RM40 probe."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LOG = Path(__file__).resolve().parent / "drdr_summary.log"

#: The log's columns, in order. Each data row has one whitespace-free token per column.
COLUMNS = (
    "setting", "n", "estimand", "R", "errors", "nonconv", "bias/SD", "bias/SD mcse",
    "SE ratio (mean)", "SE ratio (median)", "SE ratio mcse", "proj SE ratio", "coverage",
    "cov mcse", "R3/SD", "secs/fit",
)  # fmt: skip


def table() -> pd.DataFrame:
    """The summary table, without its comment header."""
    lines = [line for line in LOG.read_text(encoding="utf-8").splitlines() if not line.startswith("#")]
    header, body = lines[0], [line.split() for line in lines[1:] if line.strip()]
    # The header's own spacing is irregular, so it is checked rather than parsed.
    assert header.split()[:3] == ["setting", "n", "estimand"], header
    assert all(len(row) == len(COLUMNS) for row in body)
    return pd.DataFrame(body, columns=list(COLUMNS))


def main() -> None:
    frame = table()
    cells = frame[(frame["setting"] == "qwrong_gglm") & (frame["estimand"] == "ate")]
    ratio = cells["SE ratio (mean)"].astype(float).to_numpy()
    mcse = cells["SE ratio mcse"].astype(float).to_numpy()
    weights = 1.0 / mcse**2
    pooled = float((weights * ratio).sum() / weights.sum())
    se = float(1.0 / np.sqrt(weights.sum()))
    for n, r, m in zip(cells["n"], ratio, mcse, strict=True):
        print(f"n = {n:>6}: SE ratio {r:.3f} (MC SE {m:.3f})")
    print(f"pooled (inverse-variance): {pooled:.4f} +/- {se:.4f}")
    print(f"one-sided lower 90% bound: {pooled - 1.2816 * se:.4f}")


if __name__ == "__main__":
    main()
