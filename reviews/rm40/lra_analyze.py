# A probe, not a registered study. `cleverly` at 572501b8.
#
# The linear-remainder audit of 2026-10-08 ran this script against `cleverly` at commit
# 572501b8. It is committed so that the numbers RM40 cites from `lra_summary.log` have a
# command and seeds. Edits from the copy that ran: this comment block, the script names in the
# usage text, and the removal of an assertion on the name of the audit's temporary worktree.
#
# | item | value |
# | --- | --- |
# | outcome limit | `q_limit`: 2e6 draws, seed 4242 (logged `Qbar_lim: [0.0267, 0.2936, -0.6687]`) |
# | runs | `python reviews/rm40/lra_mar_sim.py run <setting> <n> <R> <out.csv>`, one process, one thread: `pislow` at (n, R) = (1,000, 300), (4,000, 200) and (16,000, 60); `piglm` and `piknown` at (4,000, 150) |
# | seeds | replicate `rep` of setting `s` at size `n` uses `7_000_003 * (index(s) + 1) + 7919 * n + rep`, with `SETTINGS = ("pislow", "piglm", "piknown")` |
# | constant | `KAPPA = 2`, the logit offset `2 n^(-1/3) W2` of `pislow` |
# | analysis | `python reviews/rm40/lra_analyze.py`, which reads `_lra_runs/*.csv` beside it. The raw CSVs are not committed |
#
"""Summarize _lra_runs/*.csv: bias/SD, mean term/SD, SE ratio, coverage, per setting x n x estimand."""
import glob
import os

import numpy as np
import pandas as pd

here = os.path.dirname(os.path.abspath(__file__))
frames = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(here, "_lra_runs", "*.csv")))]
df = pd.concat(frames, ignore_index=True)
err = df[df.estimand == "error"]
if len(err):
    print("errors:", len(err), err.error.head(3).tolist())
df = df[df.estimand != "error"]
rows = []
for (setting, n, est), g in df.groupby(["setting", "n", "estimand"], sort=True):
    sd = g.psi.std(ddof=1)
    R = len(g)
    cov = g.covered.mean()
    rows.append(
        {
            "setting": setting,
            "n": n,
            "estimand": est,
            "R": R,
            "bias/SD": (g.psi - g.truth).mean() / sd,
            "MCSE(bias/SD)": 1 / np.sqrt(R),
            "term/SD": g.term.mean() / sd,
            "SE ratio": g.se.mean() / sd,
            "coverage": cov,
            "MCSE(cov)": np.sqrt(cov * (1 - cov) / R),
            "status": ",".join(sorted(set(map(str, g.status)))),
            "secs/fit": g.secs.mean(),
        }
    )
out = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(out.round(3).to_string(index=False))
