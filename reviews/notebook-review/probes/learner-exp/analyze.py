import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "rows.csv"), float_precision="round_trip")
print("rows", len(df), "errors", df["error"].notna().sum())
if df["error"].notna().any():
    print(df.loc[df["error"].notna(), ["config", "learner", "n", "seed", "error"]].head(10).to_string())
ok = df[df["psi"].notna()]
out = []
for (config, n, learner), g in ok.groupby(["config", "n", "learner"]):
    k = len(g)
    cov = g["covers"].mean()
    sd = g["psi"].std(ddof=1)
    out.append(
        dict(
            config=config, n=n, learner=learner, seeds=k,
            bias=g["psi"].mean() - g["truth"].iloc[0],
            bias_mcse=sd / np.sqrt(k),
            emp_sd=sd, mean_se=g["se"].mean(), se_over_sd=g["se"].mean() / sd,
            coverage=cov, cov_mcse=np.sqrt(cov * (1 - cov) / k),
            trunc_any=(g["trunc_count"] > 0).mean(), trunc_mean=g["trunc_frac"].mean(),
            min_g_med=g["min_g"].median(), cal_slope_med=g["cal_slope"].median(),
            sens_ok=g["sens_ok"].mean(), fit_s_med=g["fit_s"].median(), fit_s_max=g["fit_s"].max(),
        )
    )
t = pd.DataFrame(out)
pd.set_option("display.width", 250)
print(t.round(4).to_string(index=False))
t.to_csv(os.path.join(HERE, "summary.csv"), index=False)
print("sens errors:", df["sens_error"].value_counts(dropna=False).to_dict())
print("g_lower values:", df["g_lower"].round(5).value_counts().head().to_dict())
