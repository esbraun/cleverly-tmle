import pandas as pd, numpy as np
d = pd.read_csv("probe_ic.csv")
cols = ["one_kappa","one_tmle_move","one_os_move","one_se_trueh","one_se_trueh_init","one_resid_init_sd","one_resid_star_sd","one_resid_init_top_sd","one_resid_star_top_sd","one_qmove_top1pct","one_qmove_rest","one_mse_h","one_var_resid","one_var_resid_trueh","one_var_plug","q_rmse","one_mean_h_at_d"]
pd.set_option("display.width", 250)
print(d.groupby("config")[cols].median().round(4).T.to_string())
# relation: TMLE move vs kappa * one-step move
for cfg, g in d.groupby("config"):
    pred = g.one_kappa * g.one_os_move
    print(f"{cfg:24s} corr(tmle_move, kappa*os_move) {np.corrcoef(g.one_tmle_move, pred)[0,1]:.2f}; SD(psi - plugin) {g.one_tmle_move.std():.4f} SD(os_move) {g.one_os_move.std():.4f}")
