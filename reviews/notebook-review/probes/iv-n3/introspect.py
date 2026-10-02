import warnings
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from cleverly import (CausalStudy, CrossFitting, ModelSpec, ModifiedTreatmentPolicyEffect,
                      PointTreatment, Runtime, Targeting, TMLEMethod)
from cleverly.datasets import make_shift_dose
from cleverly.interventions import Shift
SHIFTS = ((0.0, None, "current practice"), (0.5, 5.0, "+0.5 capped at 5"),
          (0.5, None, "+0.5 uncapped"), (1.0, None, "+1.0 uncapped"))
seed = 9000
frame, truth = make_shift_dose(n=3000, seed=seed, shifts=SHIFTS)
effect = CausalStudy(frame, design=PointTreatment(outcome="Y", treatment="A",
    adjustment=("W1", "W2", "W3"), treatment_kind="continuous")).identify(
    ModifiedTreatmentPolicyEffect(tuple(Shift(d, cap=c, name=n) for d, c, n in SHIFTS)))
method = TMLEMethod(models=ModelSpec(outcome_learner=HistGradientBoostingRegressor(random_state=seed),
    treatment_learner=HistGradientBoostingClassifier(random_state=seed), density_bins=40),
    cross_fitting=CrossFitting(n_folds=3), targeting=Targeting(q_bounds=(-30.0, 40.0)),
    runtime=Runtime(random_state=seed, n_jobs=1))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    res = effect.estimate(method=method)
print(type(res)); print([a for a in dir(res) if not a.startswith("__")])
import numpy as np
nu = res.nuisance
print("fluct keys", list(res.fluctuations))
print("estimates", list(res.estimates))
f = list(res.fluctuations.values())[0]
print("eps", f.epsilon, "method", f.method, "score", f.score, "names", f.names)
print("targeted obs range", f.targeted.observed[:5], list(f.targeted.arms))
print("initial obs", nu.outcome.observed[:5], list(nu.outcome.arms))
print("scaler", nu.scaler)
print("folds", type(nu.folds), [ (len(tr), len(te)) for tr, te in nu.folds])
ss = nu.shifts
print("ceiling", ss.ceiling, "trimmed", ss.trimmed)
print("Y", np.asarray(frame["Y"])[:5])
p = res.estimates["ate_shift[+1.0 uncapped vs current practice]"]
print(p.psi, p.std_error, p.influence_curve[:5], np.std(p.influence_curve)/np.sqrt(3000))
print(res.estimates.keys())
