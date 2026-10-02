import warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression, LogisticRegression
from cleverly import CausalStudy, CrossFitting, LongitudinalTreatment, ModelSpec, RegimeMean, Runtime, TMLEMethod
from cleverly.datasets import make_longitudinal_competing
m = TMLEMethod(models=ModelSpec(outcome_learner=LogisticRegression(max_iter=1000), pseudo_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000), censoring_learner=LogisticRegression(max_iter=1000)),
    cross_fitting=CrossFitting(enabled=False), runtime=Runtime(random_state=41, n_jobs=1))
g, t = make_longitudinal_competing(n=4000, seed=53, censoring=True)
print("censored by t1:", int((g.C1==0).sum()), " by t2:", int((g.C2==0).sum()))
r = CausalStudy(g, design=LongitudinalTreatment(outcome={"relapse": ("R1","R2"), "death": ("D1","D2")},
    censoring=("C1","C2"), treatment=("A1","A2"), baseline=("W1","W2"), time_varying=((),("L2",)))).identify(
    RegimeMean({"always":1,"never":0}, reference="never", horizons=(1,2))).estimate(method=m)
for k,e in r.estimates.items(): print(k, round(e.psi,4), round(e.std_error,4), "truth", round(t[k],4))
