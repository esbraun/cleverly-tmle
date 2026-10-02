"""Test whether an arm-specific (two clever covariate) hand TMLE at the package bound
reproduces the package fit on seeds where the notebook assertion fails."""
import sys
import warnings

import numpy as np
import pandas as pd

from common import build, expit, logit, slog
from cleverly import (ATE, CausalStudy, CrossFitting, Inference, ModelSpec, PointTreatment,
                      Runtime, TMLEMethod)

warnings.simplefilter("ignore")
for seed in [int(s) for s in sys.argv[1:]]:
    d, adj = build(seed)
    n = len(d)
    lo = 5 / (np.sqrt(n) * np.log(n))
    W = d.loc[:, adj]
    A = d["low_birth_weight"].to_numpy()
    y = d["mortality_1y"].to_numpy()
    X = pd.concat([d[["low_birth_weight"]], W], axis=1)
    qm = slog(seed).fit(X, y)
    qo = qm.predict_proba(X)[:, 1]
    q1 = qm.predict_proba(X.assign(low_birth_weight=1))[:, 1]
    q0 = qm.predict_proba(X.assign(low_birth_weight=0))[:, 1]
    g = np.clip(slog(seed).fit(W, A).predict_proba(W)[:, 1], lo, 1 - lo)
    H = np.column_stack([A / g, (1 - A) / (1 - g)])
    eps = np.zeros(2)
    for _ in range(100):
        qs = expit(logit(qo) + H @ eps)
        grad = H.T @ (y - qs) / n
        hess = (H * (qs * (1 - qs))[:, None]).T @ H / n
        step = np.linalg.solve(hess, grad)
        eps += step
        if np.abs(step).max() < 1e-13:
            break
    two = expit(logit(q1) + eps[0] / g).mean() - expit(logit(q0) + eps[1] / (1 - g)).mean()
    eff = CausalStudy(d, design=PointTreatment(outcome="mortality_1y", treatment="low_birth_weight",
                                               adjustment=adj, cluster="pair_id",
                                               outcome_family="binomial")).identify(ATE(reference=0))
    m = TMLEMethod(models=ModelSpec(outcome_learner=slog(seed), treatment_learner=slog(seed)),
                   cross_fitting=CrossFitting(enabled=False),
                   inference=Inference(alpha=0.05, simultaneous=False),
                   runtime=Runtime(random_state=seed, n_jobs=1))
    est = eff.estimate(method=m)["ate"]
    gcomp = q1.mean() - q0.mean()
    print(f"seed {seed}: pkg {est.psi:.10f} two-cov hand {two:.10f} gap {est.psi - two:.3e} "
          f"shift {two - gcomp:.3e} se {est.std_error:.4f}", flush=True)
