"""Probe: does slope under-coverage shrink with n? Args: n seed_start seed_end."""
import os
import sys
import warnings

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CausalStudy, CrossFitting, ModelSpec, MSMProjection, PointTreatment, Runtime, TMLEMethod
from cleverly.datasets import make_multi_arm
from cleverly.msm import MSM

warnings.filterwarnings("ignore")
C = {"low": 1.0, "medium": 2.0, "high": 6.0}
pop = np.array([0.0, 0.6, 1.44])
design = np.column_stack([np.ones(3), [1.0, 2.0, 6.0]])
proj = np.linalg.lstsq(design, pop, rcond=None)[0]
n, a, b = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
est, se, cov = [], [], []
for seed in range(a, b):
    frame, _ = make_multi_arm(n=n, seed=seed)
    study = CausalStudy(frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3")))
    method = TMLEMethod(
        models=ModelSpec(outcome_learner=LinearRegression(), treatment_learner=LogisticRegression(max_iter=1000, random_state=61)),
        cross_fitting=CrossFitting(enabled=False),
        runtime=Runtime(random_state=61, n_jobs=1),
    )
    r = study.identify(MSMProjection(MSM(design=lambda arm, d: np.column_stack([np.ones(len(d)), np.full(len(d), C[arm])]), terms=("i", "s"), design_kind="known"))).estimate(method=method)
    s = r["msm[s]"]
    est.append(s.psi); se.append(s.std_error); cov.append(s.ci[0] <= proj[1] <= s.ci[1])
est, se = np.array(est), np.array(se)
print(f"n={n} reps={len(est)} coverage={np.mean(cov):.4f} SEratio={se.mean()/est.std(ddof=1):.4f} bias/sd={(est.mean()-proj[1])/est.std(ddof=1):.3f}")
