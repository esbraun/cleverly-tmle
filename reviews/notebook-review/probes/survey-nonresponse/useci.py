import warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression, LogisticRegression
from cleverly import ATE, CausalStudy, CrossFitting, ModelSpec, PointTreatment, Runtime, TMLEMethod
from cleverly.datasets import make_missing_outcome
from cleverly.sensitivity.missingness import tipping_gamma, missingness_tilt
f, _ = make_missing_outcome(n=4000, seed=71, strength=2.0)
m = TMLEMethod(models=ModelSpec(outcome_learner=LinearRegression(n_jobs=1), treatment_learner=LogisticRegression(max_iter=1000, random_state=71), missingness_learner=LogisticRegression(max_iter=1000, random_state=71)), cross_fitting=CrossFitting(enabled=False), runtime=Runtime(random_state=71, n_jobs=1))
r = CausalStudy(f, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1","W2","W3"), missingness="Delta")).identify(ATE(reference=0)).estimate(method=m)
ag = {0: 0.0, 1: -1.0}
g = tipping_gamma(r, arm_gamma=ag, use_ci=True)
print("use_ci tipping", g)
print(missingness_tilt(r, [g], estimands=["ate"], arm_gamma=ag)[["psi","ci_lower","ci_upper"]])
print("point tipping", tipping_gamma(r, arm_gamma=ag))
