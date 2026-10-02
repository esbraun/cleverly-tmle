import numpy as np, cleverly
assert "bridge-cse" in cleverly.__file__
from sklearn.linear_model import LinearRegression, LogisticRegression
from cleverly import ATE, CausalStudy, PointTreatment, CrossFitting, ModelSpec, Runtime, TMLEMethod
from cleverly.datasets import make_missing_outcome
from cleverly.sensitivity import tipping_gamma
f, t = make_missing_outcome(n=4000, seed=71, strength=2.0)
m = TMLEMethod(models=ModelSpec(outcome_learner=LinearRegression(n_jobs=1),
    treatment_learner=LogisticRegression(max_iter=1000, random_state=71),
    missingness_learner=LogisticRegression(max_iter=1000, random_state=71)),
    cross_fitting=CrossFitting(enabled=False), runtime=Runtime(random_state=71, n_jobs=1))
r = CausalStudy(f, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1","W2","W3"), missingness="Delta")).identify(ATE(reference=0)).estimate(method=m)
print("use_ci", tipping_gamma(r, "ate", arm_gamma={0: 0.0, 1: -1.0}, use_ci=True))
print("sd observed Y", np.nanstd(f.Y), "arm1", np.nanstd(f.Y[f.A==1]))
