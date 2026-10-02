import warnings; warnings.filterwarnings("ignore")
import cleverly, numpy as np
print(cleverly.__file__)
from sklearn.linear_model import LogisticRegression
from cleverly import ATE, CausalStudy, CrossFitting, ModelSpec, PointTreatment, Runtime, TMLEMethod
from cleverly.datasets import make_missing_outcome_binary
f, t = make_missing_outcome_binary(n=4000, seed=73)
print(f.head()); print(t)
st = CausalStudy(f, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1","W2","W3"), missingness="Delta"))
m = TMLEMethod(models=ModelSpec(outcome_learner=LogisticRegression(max_iter=1000), treatment_learner=LogisticRegression(max_iter=1000), missingness_learner=LogisticRegression(max_iter=1000)), cross_fitting=CrossFitting(n_folds=5), runtime=Runtime(random_state=1, n_jobs=1))
r = st.identify(ATE(reference=0)).estimate(method=m)
p = r["ate"]
print(type(r), [a for a in dir(r) if not a.startswith("__")])
print(type(p), [a for a in dir(p) if not a.startswith("__")])
print(p.psi, p.std_error, p.ci)
nu = r.nuisance
print(type(nu), [a for a in dir(nu) if not a.startswith("__")])
for a in ['arms','bounded_missingness','bounded_propensity','missingness','missingness_at_realised_arm','outcome','outcome_by_level','propensity','targeting_outcome','folds','scaler']:
    v = getattr(nu,a)
    print(a, type(v), getattr(v,'shape',None) if not isinstance(v,dict) else {k:getattr(x,'shape',x) for k,x in v.items()})
print('fluct', r.fluctuations)
print('ic', type(r.influence_curves), getattr(r.influence_curves,'keys',lambda:None)())
print(p.influence_curve[:5], p.variance, p.n, p.covariance_rule, p.scale)
print(r.estimates.keys() if hasattr(r.estimates,'keys') else r.estimates)
