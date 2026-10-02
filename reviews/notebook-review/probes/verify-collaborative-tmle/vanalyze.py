import sys
import numpy as np
import pandas as pd

d = pd.read_csv(sys.argv[1], keep_default_na=False)
n = len(d)
print("seeds", n)
for lab in ("lin", "const"):
    sel = d[f"{lab}_sel"]
    print(f"\n[{lab}] selected-set shares:")
    print(sel.value_counts(normalize=True).round(3).to_string())
    print(" contains W1:", round(sel.str.contains("W1").mean(), 3),
          " contains W2:", round(sel.str.contains("W2").mean(), 3),
          " contains W3:", round(sel.str.contains("W3").mean(), 3),
          " empty:", round((sel == "").mean(), 3))
    print(" |gap10|<1e-3:", round((d[f"{lab}_gap10"].abs() < 1e-3).mean(), 3))
    c, p = d[f"{lab}_c_psi"], d[f"{lab}_plain_psi"]
    for name, psi, se in (("C", c, d[f"{lab}_c_plugse"]), ("plain", p, d[f"{lab}_plain_se"])):
        sd = psi.std(ddof=1)
        cov = ((psi - 1.96 * se <= 1) & (1 <= psi + 1.96 * se)).mean()
        print(f" {name}: bias {psi.mean()-1:+.4f} (mcse {sd/np.sqrt(n):.4f}) empSD {sd:.4f} meanSE {se.mean():.4f} ratio {se.mean()/sd:.3f} cov {cov:.3f} rmse {np.sqrt(((psi-1)**2).mean()):.4f}")
    print(" C plugse < plain se share:", round((d[f"{lab}_c_plugse"] < d[f"{lab}_plain_se"]).mean(), 3))
    print(" ratio plug/plain mean:", round((d[f"{lab}_c_plugse"] / d[f"{lab}_plain_se"]).mean(), 3))
c = d["lin_c_psi"]
print("\nHC0 vs C-TMLE(lin): ratio", round(d.hc0.mean() / c.std(ddof=1), 3),
      "cov", round(((c - 1.96 * d.hc0 <= 1) & (1 <= c + 1.96 * d.hc0)).mean(), 3))
print("plugse < hc0 share:", round((d.lin_c_plugse < d.hc0).mean(), 3))
print("|ols - C| < 5e-4:", round(((d.ols - c).abs() < 5e-4).mean(), 3))
