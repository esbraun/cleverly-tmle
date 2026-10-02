"""Oracle-only coverage at n = 4000 over many draws of make_missing_outcome_binary.

Per seed: the oracle TMLE (true g, pi, Q; the pooled two-coefficient fluctuation of probe.py) and
the oracle EIF mean ``truth + mean(EIF at truth)`` with SE sd(EIF)/sqrt(n), a pure-CLT
construction with no estimation.  Also prints Var(EIF) from one draw of 4,000,000 rows.
Usage: ``python oracle.py <n> <first seed> <count> <workers>``; writes ``oracle-<n>.csv``.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from probe import TRUTH, expit, target

HERE = Path(__file__).resolve().parent


def parts(n, seed):
    from cleverly.datasets import make_missing_outcome_binary
    fr, _ = make_missing_outcome_binary(n=n, seed=seed)
    w = fr[["W1", "W2", "W3"]].to_numpy()
    a = fr["A"].to_numpy().astype(float)
    d = fr["Delta"].to_numpy().astype(float)
    y = np.nan_to_num(fr["Y"].to_numpy().astype(float))
    q1 = expit(0.3 + 0.7 * w[:, 0] - 0.5 * w[:, 1] + 0.4 * w[:, 2])
    q0 = expit(-0.6 + 0.7 * w[:, 0] - 0.5 * w[:, 1] + 0.4 * w[:, 2])
    g1 = expit(0.4 * w[:, 0] - 0.3 * w[:, 1] + 0.2 * w[:, 2])
    p1 = expit(1.5 - 0.9 * w[:, 0] + 0.4 * w[:, 1])
    p0 = expit(1.0 - 0.9 * w[:, 0] + 0.4 * w[:, 1])
    return y, a, d, q1, q0, g1, p1, p0


def one(args):
    n, seed = args
    y, a, d, q1, q0, g1, p1, p0 = parts(n, seed)
    psi, se = target(y, a, d, q1, q0, g1, p1, p0)
    eif = a * d / (g1 * p1) * (y - q1) - (1 - a) * d / ((1 - g1) * p0) * (y - q0) + q1 - q0 - TRUTH
    return {"seed": seed, "oracle_psi": psi, "oracle_se": se,
            "eif_psi": TRUTH + eif.mean(), "eif_se": eif.std() / np.sqrt(n)}


if __name__ == "__main__":
    import pandas as pd
    n, first, count, workers = (int(x) for x in sys.argv[1:5])
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one, [(n, s) for s in range(first, first + count)], chunksize=50))
    f = pd.DataFrame(rows)
    f.to_csv(HERE / f"oracle-{n}.csv", index=False, lineterminator="\n")
    y, a, d, q1, q0, g1, p1, p0 = parts(4_000_000, 7)
    eif = a * d / (g1 * p1) * (y - q1) - (1 - a) * d / ((1 - g1) * p0) * (y - q0) + q1 - q0 - TRUTH
    sd_true = eif.std()
    lines = [f"Var(EIF) from 4,000,000 rows: SD {sd_true:.5f}; SD at n={n}: {sd_true/np.sqrt(n):.6f}; "
             f"EIF skewness {(((eif-eif.mean())/sd_true)**3).mean():.2f}, kurtosis {(((eif-eif.mean())/sd_true)**4).mean():.1f}"]
    for lab in ("oracle", "eif"):
        c = (f[lab + "_psi"] - 1.959964 * f[lab + "_se"] <= TRUTH) & (TRUTH <= f[lab + "_psi"] + 1.959964 * f[lab + "_se"])
        m = len(f); p = c.mean(); h = 1.96 * np.sqrt(p * (1 - p) / m)
        lines.append(f"{lab}: n {n}, draws {m} (seeds {first}-{first+count-1}): cover {c.sum()} ({p:.4f}; {p-h:.4f}-{p+h:.4f}); "
                     f"emp SD {f[lab+'_psi'].std():.6f}; mean SE {f[lab+'_se'].mean():.6f}")
        blocks = [c.iloc[i:i + 500].sum() for i in range(0, m, 500)]
        lines.append(f"  covers per consecutive 500-seed block: min {min(blocks)}, max {max(blocks)}, "
                     f"blocks <= 461: {sum(b <= 461 for b in blocks)} of {len(blocks)}")
    txt = "\n".join(lines)
    (HERE / f"oracle-{n}.log").write_text(txt + "\n", encoding="utf-8", newline="\n")
    print(txt)
