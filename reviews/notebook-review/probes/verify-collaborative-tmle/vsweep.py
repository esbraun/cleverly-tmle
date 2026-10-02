"""Independent verifier sweep for collaborative-tmle (seeds 1000+)."""
import os
import sys
import time

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

import cleverly
from cleverly import ATE, CausalStudy, CollaborativeTMLEMethod, CrossFitting, ModelSpec, PointTreatment, Runtime, TMLEMethod
from cleverly.datasets import make_instrument

assert "bridge-cse" in cleverly.__file__, cleverly.__file__
print(cleverly.__file__, flush=True)

start, stop = int(sys.argv[1]), int(sys.argv[2])
out = sys.argv[3]
cov = ["W1", "W2", "W3"]
rows = []
t0 = time.time()
for seed in range(start, stop):
    frame, truth = make_instrument(n=2_000, seed=seed)
    effect = CausalStudy(frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=tuple(cov))).identify(ATE(reference=0))
    rt = Runtime(random_state=44, n_jobs=1)
    folds = CrossFitting(enabled=False)
    row = {"seed": seed}
    for label, qlearner in (("lin", LinearRegression(n_jobs=1)), ("const", DummyRegressor())):
        models = ModelSpec(outcome_learner=qlearner, treatment_learner=LogisticRegression(max_iter=1000, random_state=44))
        plain = effect.estimate(method=TMLEMethod(models=models, cross_fitting=folds, runtime=rt))["ate"]
        cres = effect.estimate(method=CollaborativeTMLEMethod(models=models, cross_fitting=folds, runtime=rt, strategy="greedy", selection_folds=3, selection_inner_folds=2))
        c = cres["ate"]
        sel = cres.diagnostics.nuisance_models().selection
        row[f"{label}_plain_psi"] = plain.psi
        row[f"{label}_plain_se"] = plain.std_error
        row[f"{label}_c_psi"] = c.psi
        row[f"{label}_c_plugse"] = c.plugin_std_error
        row[f"{label}_sel"] = "+".join(sorted(sel.selected_covariates))
        row[f"{label}_gap10"] = sel.cv_risk[1] - sel.cv_risk[0]
    # OLS / HC0
    A, Y = frame["A"].to_numpy(), frame["Y"].to_numpy()
    X = frame[cov].to_numpy()
    full = np.column_stack([np.ones(len(A)), A, X])
    beta, *_ = np.linalg.lstsq(full, Y, rcond=None)
    resid = Y - full @ beta
    W = np.column_stack([np.ones(len(A)), X])
    ga, *_ = np.linalg.lstsq(W, A, rcond=None)
    ap = A - W @ ga
    row["ols"] = (ap * Y).sum() / (ap ** 2).sum()
    row["hc0"] = np.sqrt((ap ** 2 * resid ** 2).sum()) / (ap ** 2).sum()
    rows.append(row)
    if seed % 10 == 0:
        print(seed, round(time.time() - t0, 1), flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)
pd.DataFrame(rows).to_csv(out, index=False)
print("done", time.time() - t0)
