"""Mechanism probe for Step 8: remove only the instrument from g (fixed candidate, no search)
and check that the spread of the estimator falls back. Correct linear Q, in sample, n=2000."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import ATE, CausalStudy, CollaborativeTMLEMethod, CrossFitting, ModelSpec, PointTreatment, Runtime, TMLEMethod
from cleverly.datasets import make_instrument

rows = []
for seed in range(150):
    frame, truth = make_instrument(n=2_000, seed=seed)
    effect = CausalStudy(frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3"))).identify(ATE(reference=0))
    models = ModelSpec(outcome_learner=LinearRegression(n_jobs=1), treatment_learner=LogisticRegression(max_iter=1000, random_state=44))
    common = dict(models=models, cross_fitting=CrossFitting(enabled=False), runtime=Runtime(random_state=44, n_jobs=1))
    row = {"seed": seed}
    for label, cand in (("g_W1W3", ("W1", "W3")), ("g_W1W2W3", ("W1", "W2", "W3")), ("g_W1", ("W1",))):
        r = effect.estimate(method=CollaborativeTMLEMethod(**common, strategy="discrete", candidates=(cand,), selection_folds=3))
        row[label] = r["ate"].psi
        row[label + "_pse"] = r["ate"].plugin_std_error
    row["plain"] = effect.estimate(method=TMLEMethod(**common))["ate"].psi
    rows.append(row)
d = pd.DataFrame(rows)
d.to_csv(".tmp/notebook-review/collaborative-tmle/mechanism.csv", index=False)
for c in ("plain", "g_W1W2W3", "g_W1W3", "g_W1"):
    x = d[c] - 1
    extra = f" mean plug-in se {d[c + '_pse'].mean():.4f}" if c + "_pse" in d else ""
    print(f"{c:9s} bias {x.mean():+.4f} sd {x.std():.4f} rmse {np.sqrt((x**2).mean()):.4f}{extra}")
