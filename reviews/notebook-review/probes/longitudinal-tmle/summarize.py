import numpy as np
import pandas as pd

T, R, RA = 0.36154, 0.32114, -0.04041
d = pd.read_csv(".tmp/notebook-review/longitudinal-tmle/sweep.csv")
n = len(d)
print("seeds", n, d.seed.min(), d.seed.max())


def frac(x):
    return f"{x.mean():.3f} ({int(x.sum())}/{len(x)})"


print("coverage always-vs-never (clustered CI)", frac((d.lo <= T) & (T <= d.hi)))
z = 1.959964
print("coverage with iid SE", frac((d.psi - z * d.se_iid <= T) & (T <= d.psi + z * d.se_iid)))
print("mean bias", round((d.psi - T).mean(), 4), "mcse", round((d.psi - T).std() / np.sqrt(n), 4))
print("empirical SD psi", round(d.psi.std(), 4), "mean clustered SE", round(d.se.mean(), 4),
      "mean iid SE", round(d.se_iid.mean(), 4))
print("coverage rule-vs-never", frac((d.rule_lo <= R) & (R <= d.rule_hi)),
      "bias", round((d.rule_psi - R).mean(), 4), "SD", round(d.rule_psi.std(), 4), "meanSE", round(d.rule_se.mean(), 4))
print("coverage rule-vs-always", frac((d.rva_lo <= RA) & (RA <= d.rva_hi)),
      "bias", round((d.rva_psi - RA).mean(), 4))
print("rule-vs-always CI excludes zero", frac(d.rva_hi < 0))
print("|seq - truth| < SE", frac((d.psi - T).abs() < d.se))
print("crude > truth", frac(d.crude > T), "mean crude-T", round((d.crude - T).mean(), 3))
ab = d.adj_psi - T
bb = d.base_psi - T
print("adjusted below truth", frac(ab < 0), "mean", round(ab.mean(), 4), "range", round(ab.min(), 3), round(ab.max(), 3))
print("adjusted CI excludes truth (upper<T)", frac(d.adj_hi < T))
print("baseline above truth", frac(bb > 0), "mean", round(bb.mean(), 4), "range", round(bb.min(), 3), round(bb.max(), 3))
print("baseline CI excludes truth (lower>T)", frac(d.base_lo > T))
print("opposite directions", frac((ab < 0) & (bb > 0)))
print("eng gap >0.5", frac(d.eng_gap > 0.5), "share engaged>not", frac(d.share_engaged > d.share_not))
print("max weight: max", round(d.max_weight.max(), 1), "median", round(d.max_weight.median(), 1), "<100", frac(d.max_weight < 100))
print("share truncated at 0.01 is 0", frac(d.share_trunc_max == 0))
print("min eff ratio: min", round(d.min_eff_ratio.min(), 3), "median", round(d.min_eff_ratio.median(), 3))
print("curve move in SE: max", round(d.curve_move_se.max(), 3), "median", round(d.curve_move_se.median(), 3), "<0.3", frac(d.curve_move_se < 0.3))
print("cal slope range", round(d.cal_min.min(), 4), round(d.cal_max.max(), 4))
print("reg slope range", round(d.reg_min.min(), 4), round(d.reg_max.max(), 4))
print("lowest auc role counts", d.lowest_auc_role.value_counts().to_dict(), "lowest auc max", round(d.lowest_auc.max(), 3))
print("scores pass", frac(d.scores_pass))
print("rule share t2 range", round(d.rule_share_t2.min(), 3), round(d.rule_share_t2.max(), 3))
