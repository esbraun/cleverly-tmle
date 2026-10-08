# A probe, not a registered study. `cleverly` at 572501b8.
#
# The investigation of 2026-10-08 ran this script against `cleverly` at commit 572501b8. It is
# committed so that the numbers RM40 cites from `drdr_summary.log` have a command and seeds.
# Edits from the copy that ran: this comment block, and the script names in the usage text.
#
# | item | value |
# | --- | --- |
# | limits | `python reviews/rm40/drdr_sim.py limits` (2e6 draws, seed 12345). It writes `_drdr_limits.npz` beside the script. That file is not committed; this command regenerates it |
# | runs | `python reviews/rm40/drdr_sim.py run <setting> <n> <R> <out.csv> --procs 4 [--start <first rep>]`, with the settings, n and R of each row of `drdr_summary.log` |
# | seeds | replicate `rep` of setting `s` at size `n` uses `1_000_003 * (index(s) + 1) + 7919 * n + rep`, with `index` the position in `SETTINGS` |
# | top-up runs | `qwrong_gglm` n = 16,000: 400 reps, then 600 with `--start 400`. `qwrong_gslow` n = 16,000: 300, then 300 with `--start 300`. `qwrong_gknown` n = 4,000: 400, then 600 with `--start 400` |
# | analysis | `python reviews/rm40/drdr_analyze.py`, which reads `_drdr_runs/*.csv` beside it. The raw CSVs are not committed |
# | threads | one BLAS thread per process (`OMP_NUM_THREADS=1`) |
#
"""Summarize _drdr_runs/*.csv (verifier probe)."""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
rows = []
frames = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(HERE, "_drdr_runs", "*.csv")))]
alld = pd.concat(frames, ignore_index=True)
errors = alld[alld.estimand == "error"]
alld = alld[alld.estimand != "error"]
assert not alld.duplicated(["setting", "n", "rep", "estimand"]).any(), "seed overlap"
for (setting, n, est), g in alld.groupby(["setting", "n", "estimand"]):
    R = len(g)
    bias = g.psi.mean() - g.truth.iloc[0]
    sd = g.psi.std(ddof=1)
    rows.append(
        {
            "setting": setting,
            "n": n,
            "estimand": est,
            "R": R,
            "errors": int(((errors.setting == setting) & (errors.n == n)).sum()),
            "nonconv": int((~g.converged.astype(bool)).sum()),
            "bias/SD": bias / sd,
            "bias/SD mcse": 1 / np.sqrt(R),
            "SE ratio (mean)": g.se.mean() / sd,
            "SE ratio (median)": g.se.median() / sd,
            "SE ratio mcse": (g.se.mean() / sd) / np.sqrt(2 * (R - 1)),
            "proj SE ratio": g.se_proj.mean() / sd,
            "coverage": g.covered.mean(),
            "cov mcse": np.sqrt(0.95 * 0.05 / R),
            "R3/SD": (g.r3.mean() / sd) if g.r3.notna().any() else np.nan,
            "secs/fit": g.secs.mean(),
        }
    )
out = pd.DataFrame(rows)
order = {"qwrong_gglm": 0, "qwrong_gglmcf": 1, "qwrong_gslow": 2, "qwrong_ghgb": 3, "qwrong_ghgbcf": 4, "qwrong_gknown": 5,
         "qglm_gwrong": 6, "qslow_gwrong": 7, "qslow2_gwrong2": 8}
out["o"] = out.setting.map(order)
out = out.sort_values(["o", "estimand", "n"]).drop(columns="o")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
