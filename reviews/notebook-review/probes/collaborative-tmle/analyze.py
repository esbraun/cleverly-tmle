import sys

import numpy as np
import pandas as pd

d = pd.read_csv(sys.argv[1], keep_default_na=False)
n = len(d)
print("seeds", n)
z = 1.959963984540054
print("\n== correct-Q C-TMLE selection ==")
print(d["c_sel"].replace("", "(empty)").value_counts())
print("share selecting W1 (confounder):", d["c_sel"].str.contains("W1").mean())
print("share selecting W2 (instrument):", d["c_sel"].str.contains("W2").mean())
print("share selecting exactly W3:", (d["c_sel"] == "W3").mean())
print("share selecting empty:", (d["c_sel"] == "").mean())
print("share where W1 and W2 both excluded:", (~d["c_sel"].str.contains("W1") & ~d["c_sel"].str.contains("W2")).mean())
print("first path step:", d["c_path"].str.split(";").str[1].value_counts().to_dict())
print("|gap k1-k0| < 1e-3:", (d["c_gap10"].abs() < 1e-3).mean(), " gap<0:", (d["c_gap10"] < 0).mean())
print("train risk rises somewhere:", d["c_train_rises"].mean())
print("\n== constant-Q C-TMLE selection ==")
print(d["wc_sel"].replace("", "(empty)").value_counts())
print("share W1 kept:", d["wc_sel"].str.contains("W1").mean(), " W2 left out:", (~d["wc_sel"].str.contains("W2")).mean(),
      " W3 kept:", d["wc_sel"].str.contains("W3").mean())
print("\n== point estimates, truth 1 ==")
for col in ("c_psi", "p_psi", "wp_psi", "wc_psi", "ols"):
    x = d[col] - 1
    print(f"{col:7s} bias {x.mean():+.4f} (mcse {x.std()/np.sqrt(n):.4f})  emp sd {x.std():.4f}  rmse {np.sqrt((x**2).mean()):.4f}")
print("\n== spreads ==")
for est, se in (("c_psi", "c_pse"), ("p_psi", "p_se"), ("wp_psi", "wp_se"), ("wc_psi", "wc_pse"), ("c_psi", "hc0"), ("ols", "hc0")):
    cov = (np.abs(d[est] - 1) <= z * d[se]).mean()
    print(f"{est:7s} with {se:6s}: mean se {d[se].mean():.4f}, emp sd {d[est].std():.4f}, ratio {d[se].mean()/d[est].std():.3f}, coverage {cov:.3f}")
print("\nplug-in < plain se:", (d["c_pse"] < d["p_se"]).mean())
print("weak plug-in < 0.5 * weak plain se:", (d["wc_pse"] < 0.5 * d["wp_se"]).mean(), " ratio mean", (d["wc_pse"] / d["wp_se"]).mean())
print("c plug-in < hc0:", (d["c_pse"] < d["hc0"]).mean(), "  hc0 > 1.1 plug-in:", (d["hc0"] > 1.1 * d["c_pse"]).mean())
print("|ols - c_psi| < 5e-4:", (np.abs(d["ols"] - d["c_psi"]) < 5e-4).mean(), " when sel==W3:", (np.abs(d["ols"] - d["c_psi"])[d["c_sel"] == "W3"] < 5e-4).mean())
print("|c - truth| < |plain - truth|:", (np.abs(d["c_psi"] - 1) < np.abs(d["p_psi"] - 1)).mean())
print("plain AUC range", d["plain_auc"].min(), d["plain_auc"].max())
print("c tails both zero:", ((d["c_below"] == 0) & (d["c_above"] == 0)).mean(), " c ESS>0.99:", (d["c_ess_t"] > 0.99).mean())
print("plain tails >0.05 both:", ((d["p_below"] > 0.05) & (d["p_above"] > 0.05)).mean())
sub = d[d["c_sel"].str.contains("W2")]
print("\nwhen C-TMLE (correct Q) includes W2: n", len(sub), " mean plug-in se", sub["c_pse"].mean(), " c tails", sub["c_below"].mean(), sub["c_above"].mean())
