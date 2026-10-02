import sys
import numpy as np, pandas as pd
d = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "probe_ic.csv")
order = [c for c in ("insample_boost","xfit_boost","xfit_boost_notrim","xfit_rboost","xfit_oracle") if c in set(d.config)] + sorted(set(d.config) - {"insample_boost","xfit_boost","xfit_boost_notrim","xfit_rboost","xfit_oracle"})
for lab in ("one", "unc"):
    print(f"== {lab} ({'+1.0' if lab=='one' else '+0.5 uncapped'} vs current practice), seeds={d.seed.nunique()}")
    print(f"{'config':18s} {'cover':>5s} {'bias':>8s} {'SD':>7s} {'meanSE':>7s} {'SE/SD':>6s} | {'SDos':>6s} {'SEos':>6s} {'SDos/SD':>7s} | {'SEtrueh':>7s} {'SEeff':>6s} | {'move tmle/os':>12s} | {'Eh':>5s} {'Eh2':>6s} {'Eh2true':>7s} {'maxh':>6s} {'top1%':>5s} | icdiff")
    for cfg in order:
        g = d[d.config == cfg]
        sd = g[f"{lab}_psi"].std(ddof=1); se = g[f"{lab}_se"].mean()
        sdos = g[f"{lab}_onestep"].std(ddof=1); seos = g[f"{lab}_se_os"].mean()
        mv = np.median(g[f"{lab}_tmle_move"] / g[f"{lab}_os_move"])
        print(f"{cfg:18s} {g[f'{lab}_cover'].mean():5.3f} {(g[f'{lab}_psi']-g[f'{lab}_truth']).mean():+8.4f} {sd:7.4f} {se:7.4f} {se/sd:6.2f} | {sdos:6.4f} {seos:6.4f} {sdos/sd:7.2f} | {g[f'{lab}_se_trueh'].mean():7.4f} {g[f'{lab}_se_eff'].mean():6.4f} | {mv:12.3f} | {g[f'{lab}_mean_h'].mean():5.2f} {g[f'{lab}_mean_h2'].median():6.2f} {g[f'{lab}_mean_htrue2'].median():7.2f} {g[f'{lab}_max_h'].median():6.1f} {g[f'{lab}_ic_top1pct_share'].median():5.2f} | {g[f'{lab}_ic_maxdiff'].max():.1e}")
    g = d[d.config.str.startswith("xfit")]
    cols = [c for c in d.columns if c.startswith(f"{lab}_f") and c.endswith("_mean_h")]
    print("  per-fold E_n[h] (xfit_boost): mean", d.loc[d.config=="xfit_boost", cols].stack().mean().round(3),
          "range", d.loc[d.config=="xfit_boost", cols].stack().min().round(3), d.loc[d.config=="xfit_boost", cols].stack().max().round(3))
    print("  per-fold E_n[h] (xfit_oracle): mean", d.loc[d.config=="xfit_oracle", cols].stack().mean().round(3))
    # one-step bias/coverage using its own SE
    for cfg in order:
        g = d[d.config == cfg]
        lo = g[f"{lab}_onestep"] - 1.96*g[f"{lab}_se_os"]; hi = g[f"{lab}_onestep"] + 1.96*g[f"{lab}_se_os"]
        cov = ((lo <= g[f"{lab}_truth"]) & (g[f"{lab}_truth"] <= hi)).mean()
        print(f"  one-step {cfg:18s} bias {(g[f'{lab}_onestep']-g[f'{lab}_truth']).mean():+.4f} SD {g[f'{lab}_onestep'].std():.4f} cover {cov:.3f}  plug-in SD {g[f'{lab}_plugin'].std():.4f} bias {(g[f'{lab}_plugin']-g[f'{lab}_truth']).mean():+.4f}")
