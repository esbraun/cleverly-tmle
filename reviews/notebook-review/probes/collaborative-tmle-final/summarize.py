"""Summarize ``sweep.csv`` into ``summary.log``, the repeated-draw numbers the page quotes.

Usage: ``python summarize.py``.  Every line names its configuration.  "SE/SD" is the mean
reported (or plug-in) standard error over the empirical SD of the estimate.  MC SE is the Monte
Carlo standard error of a mean.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
rows = pd.read_csv(HERE / "sweep.csv", float_precision="round_trip", keep_default_na=False)
n = len(rows)
truth = rows["truth"].iloc[0]
lines = [f"draws: {n}, seeds {rows['seed'].min()} to {rows['seed'].max()}, truth {truth}"]


def share(mask: pd.Series) -> str:
    return f"{int(mask.sum())} of {n} ({mask.mean():.3f})"


def spread(label: str, estimate: str, se: str | None = None, covers: str | None = None) -> None:
    error = rows[estimate] - truth
    sd = rows[estimate].std(ddof=1)
    text = (
        f"{label}: bias {error.mean():+.4f} (MC SE {sd / np.sqrt(n):.4f}), empirical SD {sd:.4f}, "
        f"RMSE {np.sqrt((error**2).mean()):.4f}"
    )
    if se is not None:
        text += f", mean SE {rows[se].mean():.4f}, SE/SD {rows[se].mean() / sd:.3f}"
    if covers is not None:
        covered = rows[covers].astype(bool)
        text += f", coverage {covered.mean():.3f} ({int(covered.sum())} of {n})"
    lines.append(text)


def selection(label: str, column: str) -> None:
    chosen = rows[column].astype(str)
    sets = chosen.map(lambda value: frozenset(filter(None, value.split("|"))))
    lines.append(f"{label} selected g:")
    lines.append(f"  empty: {share(sets.map(len) == 0)}")
    lines.append(f"  social_support alone: {share(sets == frozenset({'social_support'}))}")
    for name in ("baseline_readiness", "queue_lottery_draw", "social_support"):
        lines.append(f"  contains {name}: {share(sets.map(lambda s, k=name: k in s))}")
        lines.append(f"  leaves out {name}: {share(sets.map(lambda s, k=name: k not in s))}")
    for value, count in chosen.value_counts().items():
        lines.append(f"  set '{value}': {count}")


lines.append("")
lines.append("linear Q (Steps 6 to 8)")
selection("C-TMLE", "c_selected")
lines.append(f"  |cv risk k=1 - k=0| < 1e-3: {share(rows['c_gap10'].abs() < 1e-3)}")
lines.append(
    f"  first greedy step is social_support: {share(rows['c_first_step'] == 'social_support')}"
)
lines.append(
    f"  in-sample risk rises somewhere on the path: {share(rows['c_risk_rises'].astype(bool))}"
)
spread("plain TMLE", "p_psi", "p_se", "p_covers")
spread("C-TMLE (plug-in diagnostic)", "c_psi", "c_pse", "c_plugin_covers")
spread(
    "design-based plain TMLE, without queue_lottery_draw (the reported fit)",
    "p_ni_psi",
    "p_ni_se",
    "p_ni_covers",
)
inside = (rows["c_psi"] >= rows["p_ni_low"]) & (rows["c_psi"] <= rows["p_ni_high"])
lines.append(f"  C-TMLE estimate inside the design-based interval: {share(inside)}")
lines.append(
    f"  empirical SD ratio, C-TMLE over design-based: "
    f"{rows['c_psi'].std(ddof=1) / rows['p_ni_psi'].std(ddof=1):.3f}"
)
for key, label in (("p", "all-three plain TMLE"), ("p_ni", "design-based plain TMLE")):
    lines.append(
        f"  {label} omitted-variable rows, mean (min to max): "
        + ", ".join(
            f"{name} {rows[f'{key}_{name}'].mean():.3f} "
            f"({rows[f'{key}_{name}'].min():.3f} to {rows[f'{key}_{name}'].max():.3f})"
            for name in ("nu2", "sigma2", "rv", "rva")
        )
    )
lines.append(
    f"  robustness value larger on the design-based fit: {share(rows['p_ni_rv'] > rows['p_rv'])}"
)
lines.append(f"  nu2 smaller on the design-based fit: {share(rows['p_ni_nu2'] < rows['p_nu2'])}")
spread("regression coefficient (HC0)", "ols", "hc0")
lines.append(
    f"  HC0 SE over the empirical SD of the C-TMLE estimate: "
    f"{rows['hc0'].mean() / rows['c_psi'].std(ddof=1):.3f}"
)
lines.append(f"  plug-in SE below HC0 SE: {share(rows['c_pse'] < rows['hc0'])}")
lines.append(f"  plug-in SE below plain SE: {share(rows['c_pse'] < rows['p_se'])}")
lines.append(f"  |regression - C-TMLE| < 5e-4: {share((rows['ols'] - rows['c_psi']).abs() < 5e-4)}")
closer = (rows["c_psi"] - truth).abs() < (rows["p_psi"] - truth).abs()
lines.append(f"  C-TMLE closer to the truth than plain TMLE: {share(closer)}")
lines.append(f"  plain g share below 0.1 above 0.05: {share(rows['p_below'] > 0.05)}")
lines.append(f"  plain g share above 0.9 above 0.05: {share(rows['p_above'] > 0.05)}")
lines.append(
    f"  empirical SD ratio, C-TMLE over plain: "
    f"{rows['c_psi'].std(ddof=1) / rows['p_psi'].std(ddof=1):.3f}"
)

lines.append("")
lines.append("constant Q (Step 9)")
selection("C-TMLE", "wc_selected")
spread("plain TMLE", "wp_psi", "wp_se", "wp_covers")
spread("C-TMLE (plug-in diagnostic)", "wc_psi", "wc_pse", "wc_plugin_covers")
large = (rows["wc_psi"] - truth).abs()
lines.append(
    f"  largest |C-TMLE error|: {large.max():.3f} (seed {rows.loc[large.idxmax(), 'seed']})"
)
lines.append(
    f"  median |error|, C-TMLE {large.median():.4f}, plain {(rows['wp_psi'] - truth).abs().median():.4f}"
)
lines.append(
    f"  empirical SD ratio, C-TMLE over plain: "
    f"{rows['wc_psi'].std(ddof=1) / rows['wp_psi'].std(ddof=1):.3f}"
)
lines.append(f"  plug-in over plain SE below 0.5: {share(rows['wc_pse'] / rows['wp_se'] < 0.5)}")
lines.append(f"  mean seconds per seed: {rows['seconds'].mean():.2f}")

(HERE / "summary.log").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
print("\n".join(lines))
