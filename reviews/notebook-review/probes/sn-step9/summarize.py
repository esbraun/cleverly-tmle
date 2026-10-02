"""Summarize probe-<tag>.csv files: coverage, bias, empirical SD, mean SE per construction."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
TRUTH = 0.186224486509527
out = []
for tag in sys.argv[1:]:
    f = pd.read_csv(HERE / f"probe-{tag}.csv", float_precision="round_trip")
    out.append(f"== {tag}: n {f['n'].iloc[0]}, draws {len(f)}, seeds {f.seed.min()}-{f.seed.max()}")
    for lab in ("lib_cv", "lib_in", "own_cv", "own_in", "oracle"):
        psi, se = f[f"{lab}_psi"], f[f"{lab}_se"]
        cov = ((psi - 1.959964 * se <= TRUTH) & (TRUTH <= psi + 1.959964 * se))
        c, m = cov.mean(), len(f)
        h = 1.96 * np.sqrt(c * (1 - c) / m)
        sd = psi.std(ddof=1)
        out.append(f"  {lab:7s} cover {cov.sum()}/{m} = {c:.3f} ({c-h:.3f}-{c+h:.3f}); bias {psi.mean()-TRUTH:+.5f} (MC SE {sd/np.sqrt(m):.5f}); "
                   f"emp SD {sd:.5f}; mean SE {se.mean():.5f}; SE/SD {se.mean()/sd:.3f}; t-SD {((psi-TRUTH)/se).std(ddof=1):.3f}")
    out.append(f"  oracle EIF at truth: mean SE {f['eif_se'].mean():.5f}")
    for lab in ("lib_cv", "lib_in"):
        out.append(f"  {lab}: max |IC - recomputed D| {f[f'{lab}_ic_maxdiff'].max():.2e}; reported SE / recomputed SE mean {(f[f'{lab}_se']/f[f'{lab}_recomp_se']).mean():.6f}; min g*pi median {f[f'{lab}_min_gpi'].median():.4f}")
    out.append(f"  corr(lib_cv psi, own_cv psi) {np.corrcoef(f.lib_cv_psi, f.own_cv_psi)[0,1]:.3f}; corr(lib_cv, oracle) {np.corrcoef(f.lib_cv_psi, f.oracle_psi)[0,1]:.3f}")
txt = "\n".join(out)
print(txt)
