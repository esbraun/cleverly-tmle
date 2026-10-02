"""Calibration of the Step 9 binary fit (5-fold stacked CV-TMLE, correct logistic models)
and of an in-sample variant, at n=4000 over many seeds."""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from cleverly import ATE, CausalStudy, CrossFitting, ModelSpec, PointTreatment, Runtime, TMLEMethod
from cleverly.datasets import make_missing_outcome_binary

warnings.filterwarnings("ignore")
TRUTH = 0.18624  # independent MC (truth.py): 0.5626 - 0.3764
cov = ("W1", "W2", "W3")
start, stop = int(sys.argv[1]), int(sys.argv[2])
rows = []
for seed in range(start, stop):
    f, t = make_missing_outcome_binary(n=4000, seed=seed)
    s = CausalStudy(f, design=PointTreatment(outcome="Y", treatment="A", adjustment=cov, missingness="Delta"))
    eff = s.identify(ATE(reference=0))
    r = {"seed": seed, "truth": t["ate"]}
    for label, cf in (("cv5", CrossFitting(n_folds=5)), ("insample", CrossFitting(enabled=False))):
        m = TMLEMethod(models=ModelSpec(
            outcome_learner=LogisticRegression(max_iter=1000, random_state=seed),
            treatment_learner=LogisticRegression(max_iter=1000, random_state=seed),
            missingness_learner=LogisticRegression(max_iter=1000, random_state=seed)),
            cross_fitting=cf, runtime=Runtime(random_state=seed, n_jobs=1))
        p = eff.estimate(method=m)["ate"]
        r[f"{label}_psi"], r[f"{label}_se"] = p.psi, p.std_error
        r[f"{label}_cov"] = p.ci[0] <= TRUTH <= p.ci[1]
    rows.append(r)
d = pd.DataFrame(rows)
d.to_csv(f".tmp/notebook-review/survey-nonresponse/boxcal_{start}.csv", index=False)
print("truth from library (for reference only)", d.truth.iloc[0])
for label in ("cv5", "insample"):
    psi, se = d[f"{label}_psi"], d[f"{label}_se"]
    print(f"{label}: n={len(d)} bias={psi.mean()-TRUTH:.5f} (mcse {psi.std()/np.sqrt(len(d)):.5f}) "
          f"empSD={psi.std():.5f} meanSE={se.mean():.5f} ratio={se.mean()/psi.std():.3f} "
          f"coverage={d[f'{label}_cov'].mean():.3f}")
