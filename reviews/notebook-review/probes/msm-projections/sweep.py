"""Seed sweep for docs/examples/msm-projections.ipynb at n=3000.

Truth is recomputed independently from the structural equations (not via the library truth).
"""
import os
import sys
import time
import warnings

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

import cleverly
from cleverly import ATE, CausalStudy, CounterfactualMean, CrossFitting, ModelSpec, MSMProjection, PointTreatment, Runtime, TMLEMethod
from cleverly.datasets import make_multi_arm
from cleverly.msm import MSM

warnings.filterwarnings("ignore")
assert "bridge-cse" in cleverly.__file__, cleverly.__file__

ARMS = ("low", "medium", "high")
C = {"low": 1.0, "medium": 2.0, "high": 6.0}
contacts = np.array([C[a] for a in ARMS])
design = np.column_stack([np.ones(3), contacts])

# Independent truth: E[Y(a)] = 0.6*step[a] + E[W1 - 0.5 W2 + 0.2 W3], W iid N(0,1) -> 0.6*step.
step = np.array([0.0, 1.0, 2.4])
pop = 0.6 * step
# Monte Carlo cross-check with fresh rng and own structural equations.
rng = np.random.default_rng(12345)
W = rng.normal(size=(2_000_000, 3))
mc = np.array([np.mean(0.6 * s + W[:, 0] - 0.5 * W[:, 1] + 0.2 * W[:, 2]) for s in step])
proj = np.linalg.lstsq(design, pop, rcond=None)[0]
# arm marginal probabilities P(A=a) from softmax of logits
logits = np.column_stack([np.zeros(len(W)), 0.8 * W[:, 0] - 0.4 * W[:, 1], -0.5 * W[:, 0] + 0.8 * W[:, 1]])
p = np.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
pa = p.mean(0)
share_proj = np.linalg.solve(design.T @ (pa[:, None] * design), design.T @ (pa * pop))
hat = design @ np.linalg.pinv(design)
pop_miss = (hat @ pop - pop)[1]
print("truth means", pop, "mc", mc.round(4))
print("projection", proj, "contrast slope", (-2 * pop[0] - pop[1] + 3 * pop[2]) / 14)
print("P(A=a)", pa.round(4), "share-weight projection with P(A=a)", share_proj)
print("population miss at medium", pop_miss)
# Naive slopes for comparison
print("min g by arm over 2e6 draws", p.min(0))
sys.stdout.flush()

seeds = [int(s) for s in sys.argv[1:]] or list(range(1, 41))
rows = []
for seed in seeds:
    t0 = time.time()
    frame, truth = make_multi_arm(n=3_000, seed=seed)
    frame = frame.rename(columns={"Y": "y", "A": "cadence", "W1": "risk", "W2": "age", "W3": "need"})
    study = CausalStudy(frame, design=PointTreatment(outcome="y", treatment="cadence", adjustment=("risk", "age", "need")))
    method = TMLEMethod(
        models=ModelSpec(outcome_learner=LinearRegression(), treatment_learner=LogisticRegression(max_iter=1000, random_state=61)),
        cross_fitting=CrossFitting(enabled=False),
        runtime=Runtime(random_state=61, n_jobs=1),
    )
    arm = study.identify(CounterfactualMean()).estimate(method=method)

    def cd(a, data):
        return np.column_stack([np.ones(len(data)), np.full(len(data), C[a])])

    trend = study.identify(MSMProjection(MSM(design=cd, terms=("i", "s"), design_kind="known"))).estimate(method=method)
    ate = study.identify(ATE(reference="low")).estimate(method=method)
    means = np.array([arm[f"ey[{a}]"].psi for a in ARMS])
    obs = frame.groupby("cadence")[["y", "risk", "age"]].mean()
    shares = frame["cadence"].value_counts(normalize=True).loc[list(ARMS)].to_numpy()
    sp = np.linalg.solve(design.T @ (shares[:, None] * design), design.T @ (shares * pop))
    mw = hat[1] - np.eye(3)[1]
    mis = arm.contrast(lambda m: float(mw @ m), [f"ey[{a}]" for a in ARMS], name="m", gradient=lambda m: mw)
    line = design @ np.array([trend["msm[i]"].psi, trend["msm[s]"].psi])
    step7 = line[1] - means[1]
    a = trend.assess(include_retargets=True)
    sup = a.report("support")
    curve = a.report("truncation_curve")
    sc = curve.loc[curve["estimand"] == "msm[s]"]
    rv = {k: ate.sensitivity.robustness_value(estimand=k) for k in ("ate[medium vs low]", "ate[high vs low]")}
    ess = {k: v["ratio"] for k, v in sup.effective_sample_size.items()}
    # simultaneous band coverage: read bands from summary text is fragile; use pointwise joint coverage instead
    si, ss = trend["msm[i]"], trend["msm[s]"]
    rows.append(dict(
        seed=seed,
        slope=ss.psi, slope_se=ss.std_error,
        slope_cov=ss.ci[0] <= proj[1] <= ss.ci[1],
        int_cov=si.ci[0] <= proj[0] <= si.ci[1],
        both_cov=(ss.ci[0] <= proj[1] <= ss.ci[1]) and (si.ci[0] <= proj[0] <= si.ci[1]),
        slope_cov_share_target=ss.ci[0] <= sp[1] <= ss.ci[1],
        arms_all_cov=all(arm[f"ey[{x}]"].ci[0] <= t <= arm[f"ey[{x}]"].ci[1] for x, t in zip(ARMS, pop)),
        mis_psi=mis.psi, mis_excl0=mis.ci[1] < 0, mis_cov=mis.ci[0] <= pop_miss <= mis.ci[1],
        step7=step7, mis_minus_step7=mis.psi - step7,
        obs_med_gt_high=obs.loc["medium", "y"] > obs.loc["high", "y"],
        risk_med_gt_high=obs.loc["medium", "risk"] > obs.loc["high", "risk"],
        age_high_gt_med=obs.loc["high", "age"] > obs.loc["medium", "age"],
        attention=",".join(sorted(i.name for i in a.attention)),
        trunc_frac=sup.truncated["fraction"], severity=sup.severity,
        narrowest=min(ess, key=ess.get),
        curve_max_abs_delta=float(sc["delta_from_fitted"].abs().max()),
        curve_min=float(sc["psi"].min()), curve_max=float(sc["psi"].max()),
        rv_med=rv["ate[medium vs low]"]["rv"], rv_high=rv["ate[high vs low]"]["rv"],
        share_slope_diff=sp[1] - proj[1],
        band_cov=all(trend.simultaneous.bands[k][0] <= t <= trend.simultaneous.bands[k][1] for k, t in (("msm[i]", proj[0]), ("msm[s]", proj[1]))),
        crit=trend.simultaneous.critical_value,
        cov_low=arm["ey[low]"].ci[0] <= pop[0] <= arm["ey[low]"].ci[1],
        cov_med=arm["ey[medium]"].ci[0] <= pop[1] <= arm["ey[medium]"].ci[1],
        cov_high=arm["ey[high]"].ci[0] <= pop[2] <= arm["ey[high]"].ci[1],
        psi_low=means[0], psi_med=means[1], psi_high=means[2],
        se_low=arm["ey[low]"].std_error, se_med=arm["ey[medium]"].std_error, se_high=arm["ey[high]"].std_error,
        int_psi=si.psi, int_se=si.std_error,
        ccmax_arm=max(ARMS, key=lambda x: C[x] / np.clip(np.asarray(trend.nuisance.propensity.values), *sup.bounds)[frame["cadence"].to_numpy() == x, list(study.data.treatment_levels).index(x)].min()),
        ccmax_matches=bool(np.isclose(sup.clever_covariate_max["msm"], max(C[x] / np.clip(np.asarray(trend.nuisance.propensity.values), *sup.bounds)[frame["cadence"].to_numpy() == x, list(study.data.treatment_levels).index(x)].min() for x in ARMS))),
        secs=time.time() - t0,
    ))
    print(rows[-1], flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(os.path.dirname(__file__), "sweep.csv"), index=False)
print()
print(df.drop(columns=["attention", "severity", "narrowest"]).describe().T.to_string())
for c in ["slope_cov", "int_cov", "both_cov", "slope_cov_share_target", "arms_all_cov", "mis_excl0", "mis_cov",
          "obs_med_gt_high", "risk_med_gt_high", "age_high_gt_med"]:
    print(c, df[c].mean())
print("attention", df["attention"].value_counts().to_dict())
print("severity", df["severity"].value_counts().to_dict())
print("trunc>0.01", (df["trunc_frac"] > 0.01).mean())
print("ccmax_arm", df["ccmax_arm"].value_counts().to_dict(), "ccmax_matches", df["ccmax_matches"].mean())
for c in ["band_cov","cov_low","cov_med","cov_high"]:
    print(c, df[c].mean())
for k,i in (("low",0),("med",1),("high",2)):
    print(k, "bias", df["psi_"+k].mean()-pop[i], "sd", df["psi_"+k].std(), "meanSE", df["se_"+k].mean())
print("intercept bias", df["int_psi"].mean()-proj[0], "sd", df["int_psi"].std(), "meanSE", df["int_se"].mean())
print("narrowest", df["narrowest"].value_counts().to_dict())
print("rv_med<rv_high", (df["rv_med"] < df["rv_high"]).mean())
print("|mis-step7|<0.0005", (df["mis_minus_step7"].abs() < 0.0005).mean(), "max", df["mis_minus_step7"].abs().max())
print("|share diff|/slope_se mean", (df["share_slope_diff"].abs() / df["slope_se"]).mean())
print("empirical SD slope", df["slope"].std(), "mean SE", df["slope_se"].mean(), "bias", df["slope"].mean() - proj[1])
