"""Why does the ordinary package TMLE differ from the hand-built TMLE at pair-sample seed 1?"""
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "C:/Users/erics/Documents/Projects/cleverly-tmle/.claude/worktrees/bridge-cse_01BbvRGwt6wTxoguDudAmLTi/.tmp/notebook-review/twins-causal-inference")
from point_sweep import build, ex, lg
from cleverly import ATE, CausalStudy, CrossFitting, Inference, ModelSpec, PointTreatment, Runtime, TMLEMethod

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
pdat, ADJ = build(seed)
sl = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=seed))
W = pdat.loc[:, ADJ]; A = pdat["low_birth_weight"].to_numpy(); Y = pdat["mortality_1y"].to_numpy()
des = pd.concat([pdat[["low_birth_weight"]], W], axis=1)
q = sl().fit(des, Y)
q1 = q.predict_proba(des.assign(low_birth_weight=1))[:, 1]; q0 = q.predict_proba(des.assign(low_birth_weight=0))[:, 1]
g = sl().fit(W, A).predict_proba(W)[:, 1]
print("hand g range", g.min(), g.max(), " q1 range", q1.min(), q1.max(), " q0 range", q0.min(), q0.max())
print("hand gcomp", q1.mean() - q0.mean())
study = CausalStudy(pdat, design=PointTreatment(outcome="mortality_1y", treatment="low_birth_weight", adjustment=ADJ, cluster="pair_id", outcome_family="binomial"))
eff = study.identify(ATE(reference=0))
m = TMLEMethod(models=ModelSpec(outcome_learner=sl(), treatment_learner=sl()), cross_fitting=CrossFitting(enabled=False),
               inference=Inference(alpha=0.05, simultaneous=False), runtime=Runtime(random_state=seed, n_jobs=1))
r = eff.estimate(method=m)
print("pkg ate", r["ate"].psi)
nu = r.nuisance
print([a for a in dir(nu) if not a.startswith("_")])
gp = np.asarray(nu.propensity.arm(1.0))
print("pkg g range", gp.min(), gp.max(), " max |g - hand g|", np.abs(gp - g).max())
for name in dir(nu):
    if "outcome" in name or name.startswith("q"):
        print(name, type(getattr(nu, name)))
try:
    o = nu.outcome
    print([a for a in dir(o) if not a.startswith("_")])
    for lv in (0.0, 1.0):
        qa = np.asarray(o.arm(lv)) if hasattr(o, "arm") else None
        if qa is not None:
            ref = q1 if lv == 1.0 else q0
            print("arm", lv, "pkg initial Q mean", qa.mean(), "hand", ref.mean(), "max abs diff", np.abs(qa - ref).max())
except Exception as e:
    print("outcome access failed", repr(e))
print(r.summary())
