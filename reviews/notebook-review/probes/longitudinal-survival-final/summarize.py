"""Summarize sweep.csv and sweep-confirm.csv for the readings of longitudinal-survival.ipynb.

Usage: ``python summarize.py > summary.log``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FUNCTIONAL = 0.264627 - 0.304906  # truth.py
TOTAL = 0.246694 - 0.244344  # truth.py


def count(mask):
    return f"{int(mask.sum())} of {len(mask)} ({mask.mean():.3f})"


def load(name):
    return pd.read_csv(HERE / name, float_precision="round_trip")


def report(name, s=None):
    s = load(name) if s is None else s
    print(f"=== {name}: seeds {s['seed'].min()}-{s['seed'].max()}, {len(s)} draws")
    print("Step 3: naive understates the benefit (population < naive < 0)")
    print("  t=1", count((-0.108207 < s["naive_t1"]) & (s["naive_t1"] < 0)))
    print("  t=2", count((-0.138029 < s["naive_t2"]) & (s["naive_t2"] < 0)))
    print("Step 6-7: retention fit")
    for col in [c for c in s.columns if c.startswith("exit_") and c.endswith("_cov")]:
        print(f"  coverage {col[:-4]:22s} {s[col].mean():.3f}")
    for h in (1, 2):
        sd = s[f"exit_diff_t{h}"].std(ddof=1)
        print(
            f"  diff t={h}: mean SE / SD {s[f'exit_diff_t{h}_se'].mean() / sd:.3f}, "
            f"bias {s[f'exit_diff_t{h}'].mean() - (-0.108207 if h == 1 else -0.138029):+.4f}"
        )
    print("  60-day reduction larger:", count(s["exit_diff_t2"] < s["exit_diff_t1"]))
    print("  both differences negative:", count((s["exit_diff_t1"] < 0) & (s["exit_diff_t2"] < 0)))
    print("  simultaneous bands cover all four:", count(s["bands_all_cover"]))
    print("Step 8-9: competing fit with disenrollment censoring")
    print(f"  disenrolled p1 range {s['disenrolled_p1'].min()}-{s['disenrolled_p1'].max()}")
    for col in [c for c in s.columns if (c.startswith("cif") or c.startswith("free")) and c.endswith("_cov")]:
        print(f"  coverage {col[:-4]:28s} {s[col].mean():.3f}")
    for c in ("readmission", "death"):
        for h in (1, 2):
            sd = s[f"cifd_{c}_t{h}"].std(ddof=1)
            print(f"  diff {c} t={h}: mean SE / SD {s[f'cifd_{c}_t{h}_se'].mean() / sd:.3f}")
    print("  misses per draw (of 12): mean", round(s["misses"].mean(), 3))
    print("  draws with no miss:", count(s["misses"] == 0))
    print("  draws with at least one miss:", count(s["misses"] >= 1))
    print("  readmission diff t=1 negative:", count(s["cifd_readmission_t1"] < 0))
    print(
        "  readmission diff t=2 above t=1:",
        count(s["cifd_readmission_t2"] > s["cifd_readmission_t1"]),
    )
    print("  readmission diff t=2 positive:", count(s["cifd_readmission_t2"] > 0))
    print("  death diff larger reduction at t=2:", count(s["cifd_death_t2"] < s["cifd_death_t1"]))
    print("  death diff t=1 negative and t=2 negative:", count((s["cifd_death_t1"] < 0) & (s["cifd_death_t2"] < 0)))
    print("  max incidence-total excess is 0:", count(s["excess_max"] == 0.0))
    print("  largest |epsilon| row, top 4:")
    print("   ", s["eps_max_row"].value_counts().head(4).to_dict())
    print("Step 10: death as censoring, K_k = C_k (1 - D_k)")
    sd = s["el_diff_t2"].std(ddof=1)
    print(
        f"  diff t=2: mean {s['el_diff_t2'].mean():.4f} (functional {FUNCTIONAL:.4f}, "
        f"bias {s['el_diff_t2'].mean() - FUNCTIONAL:+.4f}, MC SE {sd / np.sqrt(len(s)):.4f}), "
        f"mean SE / SD {s['el_diff_t2_se'].mean() / sd:.3f}"
    )
    print("  CI covers the functional:", count(s["el_diff_t2_cov_functional"]))
    print("  CI covers the total effect:", count(s["el_diff_t2_cov_total"]))
    print("  never level covers functional:", count(s["el_never_t2_cov"]))
    print("  always level covers functional:", count(s["el_always_t2_cov"]))
    gap = s["el_diff_t2"] - s["cifd_readmission_t2"]
    print(f"  paired gap (censored minus competing): mean {gap.mean():.4f}, max {gap.max():.4f}")
    print("  censored diff below competing diff:", count(gap < 0))
    print("  censored diff below competing diff - 0.01:", count(gap < -0.01))
    print(
        "  raises never more than always, always raised:",
        count((s["raised_never"] > s["raised_always"]) & (s["raised_always"] > 0)),
    )
    print("Step 11: support")
    print(f"  min ESS ratio range {s['exit_min_ess_ratio'].min():.3f}-{s['exit_min_ess_ratio'].max():.3f}")
    print("  weakest row never/2:", count(s["exit_weakest_row"] == "never/2"))
    print("  no truncation, both fits:", count((s["exit_max_truncated"] == 0) & (s["event_max_truncated"] == 0)))
    print("  all converged, both fits:", count(s["exit_all_converged"] & s["event_all_converged"]))
    print("Step 12: covariate-drop benchmark, 60-day difference, move in SE")
    moves = s[["move_age", "move_baseline_readiness", "move_identified_needs"]]
    for col in moves:
        print(f"  {col}: mean {moves[col].mean():.2f}, positive {count(moves[col] > 0)}")
    print("  all three positive:", count((moves > 0).all(axis=1)))
    print("  largest move by column:", moves.idxmax(axis=1).value_counts().to_dict())
    print(f"  seconds per seed: mean {s['secs'].mean():.2f}")
    print()


for name in ("sweep.csv", "sweep-confirm.csv"):
    report(name)
report("pooled", pd.concat([load("sweep.csv"), load("sweep-confirm.csv")], ignore_index=True))
