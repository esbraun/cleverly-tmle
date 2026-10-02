import time
import cleverly
print(cleverly.__file__)
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from cleverly import ATE, CausalStudy, PointTreatment, CrossFitting, Inference, ModelSpec, Runtime, Targeting, TMLEMethod
from cleverly.datasets import navigation_data

frame, truth = navigation_data(n=3000, seed=21)
print(truth)
cov = ("discharge_risk", "prior_utilization", "medication_burden", "age")
effect = CausalStudy(frame, design=PointTreatment(outcome="transition_score", treatment="transition_navigation", adjustment=cov)).identify(ATE(reference=0))
m = TMLEMethod(models=ModelSpec(outcome_learner=HistGradientBoostingRegressor(random_state=21), treatment_learner=HistGradientBoostingClassifier(random_state=21)),
               cross_fitting=CrossFitting(n_folds=5), targeting=Targeting(q_bounds=(0.0, 1.0)), inference=Inference(alpha=0.05), runtime=Runtime(random_state=21, n_jobs=1))
t = time.perf_counter()
r = effect.estimate(method=m)
print("fit s", time.perf_counter() - t)
e = r["ate"]
print(e.psi, e.std_error, e.ci)
print([a for a in dir(r) if not a.startswith("_")])
t = time.perf_counter()
diag = r.diagnostics.run_all()
print("diag s", time.perf_counter() - t)
s = diag.report("support")
print(type(s), [a for a in dir(s) if not a.startswith("_")])
print(s.truncated, s.bounds, s.propensity_quantiles)
nm = diag.report("nuisance_models")
print(type(nm), [a for a in dir(nm) if not a.startswith("_")])
print(nm.summary())
print([a for a in dir(r.diagnostics) if not a.startswith("_")])
try:
    r.sensitivity.robustness_value()
except Exception as ex:
    print(type(ex).__name__, str(ex)[:200])
