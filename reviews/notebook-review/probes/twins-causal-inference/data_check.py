"""Independent data checks for the TWINS notebook (no cleverly)."""
import numpy as np
import pandas as pd

D = "C:/Users/erics/Documents/Projects/cleverly-tmle/.claude/worktrees/bridge-cse_01BbvRGwt6wTxoguDudAmLTi/.tmp/notebook-review/twins-causal-inference/"
X = pd.read_csv(D + "twin_pairs_X_3years_samesex.csv")
T = pd.read_csv(D + "twin_pairs_T_3years_samesex.csv")
Y = pd.read_csv(D + "twin_pairs_Y_3years_samesex.csv")
print("rows", len(X), len(T), len(Y))
print("T NaN", T[["dbirwt_0", "dbirwt_1"]].isna().sum().to_dict())
print("Y NaN", Y[["mort_0", "mort_1"]].isna().sum().to_dict())
print("Y values", sorted(pd.unique(Y[["mort_0", "mort_1"]].to_numpy().ravel())))
print("twin0 lighter strictly:", (T.dbirwt_0 < T.dbirwt_1).mean(), "equal:", (T.dbirwt_0 == T.dbirwt_1).sum())
print("weight range", T.min().to_dict(), T.max().to_dict())
print("both < 2000g pairs:", ((T.dbirwt_0 < 2000) & (T.dbirwt_1 < 2000)).sum())
print("csex unique", X["csex"].unique()[:10])
print("gestat10 NaN share", X["gestat10"].isna().mean())
for c in ["mager8", "meduc6", "mrace", "dmar", "mplbir_reg", "data_year", "diabetes", "chyper", "phyper", "preterm", "tobacco", "alcohol"]:
    print(c, "NaN", int(X[c].isna().sum()), "levels", sorted(X[c].dropna().unique().tolist())[:12])

# Full-file child-level quantities
lbw0 = (T.dbirwt_0 < 2500).astype(int); lbw1 = (T.dbirwt_1 < 2500).astype(int)
A = np.r_[lbw0, lbw1]; Yc = np.r_[Y.mort_0, Y.mort_1]
print("FULL: LBW share %.4f mort %.4f crude RD %.4f" % (A.mean(), Yc.mean(), Yc[A == 1].mean() - Yc[A == 0].mean()))
disc = lbw0 != lbw1
print("FULL: discordant pairs", disc.sum(), "of", len(T), "; any disc where heavier is LBW and lighter not:", int(((lbw1 == 1) & (lbw0 == 0)).sum()))
print("FULL: within-discordant RD %.4f" % (Y.mort_0[disc] - Y.mort_1[disc]).mean())

# Notebook sample, recomputed
SEED = 2026
rows = X.sample(n=6000, random_state=SEED).index.sort_values()
t, y = T.loc[rows], Y.loc[rows]
a = np.r_[(t.dbirwt_0 < 2500).astype(int), (t.dbirwt_1 < 2500).astype(int)]
yy = np.r_[y.mort_0, y.mort_1]
d = (t.dbirwt_0 < 2500) != (t.dbirwt_1 < 2500)
print("SAMPLE: LBW %.4f mort %.4f r0 %.4f r1 %.4f crude %.4f disc %d paired %.4f" % (
    a.mean(), yy.mean(), yy[a == 0].mean(), yy[a == 1].mean(), yy[a == 1].mean() - yy[a == 0].mean(),
    d.sum(), (y.mort_0[d] - y.mort_1[d]).mean()))
# Sweep of the paired stress test over pair-sample seeds
res = []
rng = np.random.default_rng(0)
for s in range(200):
    rows = X.sample(n=6000, random_state=s).index
    t, y = T.loc[rows], Y.loc[rows]
    d = ((t.dbirwt_0 < 2500) != (t.dbirwt_1 < 2500)).to_numpy()
    diff = (y.mort_0 - y.mort_1).to_numpy()[d]
    boots = rng.choice(diff, size=(4000, len(diff))).mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    res.append((len(diff), diff.mean(), lo > 0))
r = np.array(res, float)
print("SWEEP 200 pair samples: n_disc mean %.0f range %d-%d; paired RD mean %.4f sd %.4f; boot CI excludes 0 in %.2f" % (
    r[:, 0].mean(), r[:, 0].min(), r[:, 0].max(), r[:, 1].mean(), r[:, 1].std(), r[:, 2].mean()))
