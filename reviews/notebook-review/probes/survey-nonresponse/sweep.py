"""Seed sweep of the survey-nonresponse notebook's claims at n=4000."""
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LinearRegression, LogisticRegression

import cleverly
from cleverly import (ATE, CausalStudy, CrossFitting, ModelSpec, OddsRatio, PointTreatment,
                      RiskRatio, Runtime, TMLEMethod)
from cleverly.datasets import make_missing_outcome, make_missing_outcome_binary

assert "bridge-cse" in cleverly.__file__, cleverly.__file__
warnings.filterwarnings("ignore")

RESP_POP = 1.4284  # independent MC, truth.py
TRUTH = 1.2
names = {"Y": "Y", "A": "A", "W1": "W1", "W2": "W2", "W3": "W3", "Delta": "D"}
cov = ("W1", "W2", "W3")


def method(seed):
    return TMLEMethod(
        models=ModelSpec(
            outcome_learner=LinearRegression(n_jobs=1),
            treatment_learner=LogisticRegression(max_iter=1000, random_state=seed),
            missingness_learner=LogisticRegression(max_iter=1000, random_state=seed),
        ),
        cross_fitting=CrossFitting(enabled=False),
        runtime=Runtime(random_state=seed, n_jobs=1),
    )


def cc(frame, m):
    kept = frame[frame["Delta"] == 1].drop(columns=["Delta"])
    d = PointTreatment(outcome="Y", treatment="A", adjustment=cov)
    return CausalStudy(kept, design=d).identify(ATE(reference=0)).estimate(method=m)["ate"], kept


def cov_(p, t):
    return p.ci[0] <= t <= p.ci[1]


seeds = [int(s) for s in sys.argv[1:]] or list(range(100, 140))
rows = []
for seed in seeds:
    t0 = time.time()
    m = method(seed)
    frame, truth = make_missing_outcome(n=4000, seed=seed, strength=2.0)
    r = {"seed": seed}
    r["resp_rate"] = frame["Delta"].mean()
    byr = frame.groupby("Delta")["W1"].mean()
    r["w1_resp"], r["w1_non"] = byr.loc[1.0], byr.loc[0.0]
    rb = frame.groupby("A")["Delta"].mean()
    r["resp_a0"], r["resp_a1"] = rb.loc[0.0], rb.loc[1.0]
    obs = frame[frame["Delta"] == 1]
    sm = obs.groupby("A")["Y"].mean()
    r["unadj"] = sm.loc[1.0] - sm.loc[0.0]
    study = CausalStudy(frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=cov,
                                                     missingness="Delta"))
    full = study.identify(ATE(reference=0)).estimate(method=m)
    f = full["ate"]
    r["full_psi"], r["full_se"], r["full_cov"] = f.psi, f.std_error, cov_(f, TRUTH)
    c, kept = cc(frame, m)
    w1 = kept["W1"].to_numpy()
    resp_eff = float(np.mean(1.2 - 0.9 * w1))
    r["cc_psi"], r["cc_se"] = c.psi, c.std_error
    r["cc_cov_truth"] = cov_(c, TRUTH)
    r["cc_cov_resp_sample"] = cov_(c, resp_eff)
    r["cc_cov_resp_pop"] = cov_(c, RESP_POP)
    r["resp_eff"] = resp_eff
    r["shift_most"] = (resp_eff - TRUTH) > (c.psi - resp_eff)
    r["dist_se"] = (c.psi - resp_eff) / c.std_error
    r["cc_gt_truth"] = c.psi > TRUTH
    # mild law
    mf, mt = make_missing_outcome(n=4000, seed=seed, strength=1.0)
    mc, _ = cc(mf, m)
    r["mild_psi"], r["mild_cov"] = mc.psi, cov_(mc, TRUTH)
    # sensitivity: tipping gamma via assess, as in notebook
    a = full.assess(include_retargets=True, arguments={
        "missingness": {"gamma": (-2.0, -1.0, 0.0, 1.0, 2.0), "arm_gamma": {0: 0.0, 1: -1.0}},
        "tipping_gamma": {"arm_gamma": {0: 0.0, 1: -1.0}}})
    r["attention"] = len(tuple(a.attention))
    tg = a.report("tipping_gamma")
    r["tipping"] = tg
    sc = full.nuisance.scaler
    r["range"] = sc.upper - sc.lower
    nuis = a.report("nuisance_models").summary()
    r["nuis_ok"] = "nuisance fits look reasonable" in nuis
    # actual mean shift of unobserved arm-1 means at the tipping gamma (fitted Q*, scaled)
    if tg is not None:
        pk = expit(tg / 2)
        r["largest_shift_units"] = (2 * pk - 1) * r["range"]
    # binary law, 5-fold
    bf, bt = make_missing_outcome_binary(n=4000, seed=seed + 1)
    bstudy = CausalStudy(bf, design=PointTreatment(outcome="Y", treatment="A", adjustment=cov,
                                                    missingness="Delta"))
    bm = TMLEMethod(models=ModelSpec(
        outcome_learner=LogisticRegression(max_iter=1000, random_state=seed),
        treatment_learner=LogisticRegression(max_iter=1000, random_state=seed),
        missingness_learner=LogisticRegression(max_iter=1000, random_state=seed)),
        cross_fitting=CrossFitting(n_folds=5), runtime=Runtime(random_state=seed, n_jobs=1))
    bp = {}
    for est, key in ((ATE(reference=0), "ate"), (RiskRatio(reference=0), "rr"),
                     (OddsRatio(reference=0), "or")):
        p = bstudy.identify(est).estimate(method=bm)[key]
        bp[key] = p
        r[f"box_{key}_cov"] = cov_(p, bt[key])
    r["box_or_gt_rr"] = bp["or"].psi > bp["rr"].psi
    r["box_rr_asym"] = (bp["rr"].ci[1] - bp["rr"].psi) > (bp["rr"].psi - bp["rr"].ci[0])
    r["secs"] = time.time() - t0
    rows.append(r)
    print(r, flush=True)

df = pd.DataFrame(rows)
df.to_csv(f".tmp/notebook-review/survey-nonresponse/sweep_{seeds[0]}.csv", index=False)
print(df.describe().T.to_string())
print(df.select_dtypes(bool).mean().to_string())
