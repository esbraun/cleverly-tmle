"""Summarize ``sweep.csv`` into ``summary.log``: the repeated-draw numbers the page quotes.

Usage: ``python summarize.py``.  Coverage MC SE is ``sqrt(p (1 - p) / draws)``; bias MC SE is the
standard deviation of the estimates over ``sqrt(draws)``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from cleverly.datasets import exact_readmission_truth

HERE = Path(__file__).resolve().parent
TRUTH = exact_readmission_truth((1.5, 3.0), restricted_horizons=(3.0,))
DEATH_FREE = exact_readmission_truth((1.5, 3.0), death_prevented=True)


def main() -> None:
    rows = pd.read_csv(HERE / "sweep.csv")
    draws = len(rows)
    lines = [f"draws: {draws}, seeds {rows['seed'].min()}..{rows['seed'].max()}"]
    names = [c[: -len(":psi")] for c in rows.columns if c.endswith(":psi")]
    for name in names:
        fit, estimand = name.split(":", 1)
        truth = (DEATH_FREE if fit == "one" else TRUTH)[estimand]
        psi, se = rows[f"{name}:psi"], rows[f"{name}:std_err"]
        covers = ((rows[f"{name}:ci_lower"] <= truth) & (truth <= rows[f"{name}:ci_upper"])).mean()
        sd = psi.std(ddof=1)
        lines.append(
            f"{fit} {estimand}: truth {truth:.4f}, bias {np.mean(psi - truth):.4f} (MC SE "
            f"{sd / np.sqrt(draws):.4f}), emp SD {sd:.4f}, mean SE {se.mean():.4f}, "
            f"SE/SD {se.mean() / sd:.2f}, coverage {covers:.3f} "
            f"(MC SE {np.sqrt(covers * (1 - covers) / draws):.3f})"
        )
    simple = rows["simple_readmission_difference"]
    lines.append(
        f"simple readmission difference by month 3: mean {simple.mean():.4f}, SD "
        f"{simple.std(ddof=1):.4f}, truth "
        f"{TRUTH['ate_regimen[offered vs not offered, readmission @ t=3]']:.4f}"
    )
    (HERE / "summary.log").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
