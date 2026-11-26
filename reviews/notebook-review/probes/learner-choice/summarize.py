"""Summarize ``sweep.csv`` into ``summary.log``: the repeated-draw numbers the page quotes.

Usage: ``python summarize.py``.  Coverage MC SE is ``sqrt(p (1 - p) / draws)``; bias MC SE is the
standard deviation of the estimates over ``sqrt(draws)``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
COVARIATES = ("discharge_risk", "prior_utilization", "medication_burden", "age")


def main() -> None:
    rows = pd.read_csv(HERE / "sweep.csv")
    draws = len(rows)
    lines = [f"draws: {draws}, seeds {rows['seed'].min()}..{rows['seed'].max()}"]
    truth = rows["truth"]
    for fit in ("declared", "discrete", "unscreened", "screened"):
        psi, se, cov = rows[f"{fit}_psi"], rows[f"{fit}_se"], rows[f"{fit}_covers"].mean()
        sd = psi.std(ddof=1)
        lines.append(
            f"{fit}: bias {np.mean(psi - truth):.4f} (MC SE {sd / np.sqrt(draws):.4f}), "
            f"emp SD {sd:.4f}, mean SE {se.mean():.4f}, SE/SD {se.mean() / sd:.2f}, "
            f"coverage {cov:.3f} (MC SE {np.sqrt(cov * (1 - cov) / draws):.3f}), "
            f"draws with any row at the cap {(rows[f'{fit}_at_cap'] > 0).mean():.3f}"
        )
    diff = rows["screened_minus_unscreened"]
    lines.append(
        f"screened minus unscreened: mean {diff.mean():.4f} (MC SE "
        f"{diff.std(ddof=1) / np.sqrt(draws):.4f}), SD {diff.std(ddof=1):.4f}"
    )
    lines.append(
        "screened SE / unscreened SE: median "
        f"{(rows['screened_se'] / rows['unscreened_se']).median():.3f}"
    )
    for name in COVARIATES:
        lines.append(f"screen keeps {name}: {rows[f'kept_{name}'].mean():.3f}")
    for name in ("medication_burden", "age"):
        lines.append(f"draws where the screen drops {name}: {(~rows[f'kept_{name}']).sum()}")
    for fit in ("declared", "screened"):
        lines.append(
            f"draws where the {fit} interval contains the truth: {rows[f'{fit}_covers'].sum()}"
        )
    both = (~rows["kept_medication_burden"] & ~rows["kept_age"]).mean()
    lines.append(f"screen drops both medication_burden and age: {both:.3f}")
    lines.append(f"kept columns: median {rows['kept_count'].median():.0f}")
    for fit in ("declared", "discrete"):
        for model in ("propensity", "outcome"):
            counts = rows[f"{fit}_{model}_top"].value_counts(normalize=True).round(3)
            lines.append(f"{fit} {model} top candidate shares: {counts.to_dict()}")
    text = "\n".join(lines) + "\n"
    (HERE / "summary.log").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
