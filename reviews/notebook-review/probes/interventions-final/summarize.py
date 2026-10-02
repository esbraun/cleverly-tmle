"""Summarize ``sweep_binary.csv`` and ``sweep_dose.csv`` into ``summary.log``.

The truths are the package's quadrature values that ``sweep_binary.log`` prints for the binary
law, and the per-draw truth columns of ``sweep_dose.csv`` for the dose law.  ``truth.py``
recomputes both independently.

For each configuration and contrast the table gives the coverage of the 95% interval, the mean
error (estimate minus truth) with its Monte Carlo standard error, the empirical SD of the
estimate, the mean reported SE, and their ratio.

Usage: ``python summarize.py > summary.log``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SCREEN_TRUTH = 0.1054659
IPSI_TRUTH = 0.0241378


def table(frame: pd.DataFrame, label: str, truth) -> str:
    truth = frame[f"{label}_truth"] if truth is None else truth
    error = frame[f"{label}_psi"] - truth
    covered = (frame[f"{label}_lo"] <= truth) & (truth <= frame[f"{label}_hi"])
    sd = frame[f"{label}_psi"].std(ddof=1)
    se = frame[f"{label}_se"].mean()
    return (
        f"| {label} | {int(covered.sum())}/{len(frame)} = {covered.mean():.3f} | "
        f"{error.mean():+.5f} ({sd / np.sqrt(len(frame)):.5f}) | {sd:.5f} | {se:.5f} | "
        f"{se / sd:.2f} |"
    )


HEADER = "| contrast | coverage | mean error (MC SE) | empirical SD | mean SE | SE / SD |\n" + (
    "| --- | --- | --- | --- | --- | --- |"
)

binary = pd.read_csv(HERE / "sweep_binary.csv")
print("## Regime and incremental axes (sweep_binary.csv)")
print(f"seeds {binary.seed.min()} to {binary.seed.max()}, n = 3000, three folds")
for learner, group in binary.groupby("learner"):
    print(f"\n### treatment learner: {learner} ({len(group)} draws)\n")
    print(HEADER)
    print(table(group, "all", group["ate_truth"]))
    print(table(group, "screen", SCREEN_TRUTH))
    print(table(group, "ipsi", IPSI_TRUTH))
    flagged = group["regime_attention"].fillna("").str.contains("nuisance_models")
    print(f"\nnuisance_models in needs attention: {int(flagged.sum())}/{len(group)}")
    q = group["cal_slope"].quantile([0.05, 0.5, 0.95])
    print(f"calibration slope 5%, 50%, 95%: {q[0.05]:.3f}, {q[0.5]:.3f}, {q[0.95]:.3f}")
    q = group["all_min_g"].quantile([0.05, 0.5, 0.95])
    print(f"offer-to-all min g 5%, 50%, 95%: {q[0.05]:.2e}, {q[0.5]:.2e}, {q[0.95]:.2e}")
    below = (group["all_min_g"] < 0.05).mean()
    print(f"draws whose fitted offer-to-all min g is below the law's floor 0.05: {below:.3f}")
    ipsi_above = (group["ipsi_psi"] > IPSI_TRUTH).mean()
    print(f"draws with the incremental estimate above its truth: {ipsi_above:.3f}")
    print(f"warnings raised: {group['warning_kinds'].fillna('none').value_counts().to_dict()}")

dose = pd.read_csv(HERE / "sweep_dose.csv")
print("\n## Modified-treatment-policy axis (sweep_dose.csv)")
print(f"seeds {dose.seed.min()} to {dose.seed.max()}, n = 3000, in sample")
for configuration, group in dose.groupby("configuration"):
    print(f"\n### configuration: {configuration} ({len(group)} draws)\n")
    print(HEADER)
    for label in ("capped", "uncapped", "one", "gap"):
        print(table(group, label, None))
    width = (group["one_hi"] - group["one_lo"]) / (group["uncapped_hi"] - group["uncapped_lo"])
    print(
        f"\n+1.0 width / +0.5 uncapped width: mean {width.mean():.2f}, "
        f"above 1.8 on {(width > 1.8).mean():.3f}"
    )
    print(f"gap interval excludes zero: {(group['gap_lo'] > 0).mean():.3f}")
    print(f"gap estimate positive: {(group['gap_psi'] > 0).mean():.3f}")
    print(
        f"gap SE below half of each contrast SE: "
        f"{(group['gap_se'] < 0.5 * group[['capped_se', 'uncapped_se']].min(axis=1)).mean():.3f}"
    )
    print(
        f"mean estimated ESS share: +0.5 uncapped {group['uncapped_ess'].mean():.3f}, "
        f"+1.0 {group['one_ess'].mean():.3f}"
    )
    below = (group["one_ess"] < np.exp(-1.0)).mean()
    print(f"+1.0 estimated ESS share below exp(-1): {below:.3f}")
    print(f"PositivityWarnings per fit: {group['positivity_warnings'].value_counts().to_dict()}")
    if "one_mean_ratio" in group and group["one_mean_ratio"].notna().all():
        ratios = {label: group[f"{label}_mean_ratio"] for label in ("capped", "uncapped", "one")}
        means = ", ".join(
            f"{label} {value.mean():.3f} ({value.min():.3f} to {value.max():.3f})"
            for label, value in ratios.items()
        )
        print(f"mean density ratio at the observed dose, mean (range): {means}")
    attention = group["attention"].fillna("").ne("").sum()
    print(f"fits with a needs-attention item: {attention}/{len(group)}")
