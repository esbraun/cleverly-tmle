import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from cleverly import CausalStudy, CrossFitting, LongitudinalTreatment, ModelSpec, RegimeMean, Runtime, TMLEMethod
from cleverly.datasets import make_longitudinal_competing
m = TMLEMethod(models=ModelSpec(outcome_learner=LogisticRegression(max_iter=1000), pseudo_learner=LinearRegression(), treatment_learner=LogisticRegression(max_iter=1000), censoring_learner=LogisticRegression(max_iter=1000)), cross_fitting=CrossFitting(enabled=False), runtime=Runtime(random_state=41, n_jobs=1))
nodes = dict(treatment=("A1", "A2"), baseline=("W1", "W2"), time_varying=((), ("L2",)))
est = RegimeMean({"always": 1, "never": 0}, reference="never", horizons=(1, 2))
out = []
for seed in range(3000, 3030):
    df, _ = make_longitudinal_competing(n=4000, seed=seed, censoring=True)
    ok1 = df["C1"].eq(1)
    k1 = df["C1"] * (1 - df["D1"].fillna(0))
    through1 = ok1 & df["D1"].eq(0) & df["R1"].eq(0)
    k2 = (df["C2"] * (1 - df["D2"].fillna(0))).where(through1)
    dc = df.assign(K1=k1, K2=k2, R1=df["R1"].where(k1.eq(1)), R2=df["R2"].where(k2.eq(1) | (k1.eq(1) & df["R1"].eq(1)))).drop(columns=["D1", "D2", "C1", "C2"])
    r = CausalStudy(dc, design=LongitudinalTreatment(outcome=("R1", "R2"), censoring=("K1", "K2"), **nodes)).identify(est).estimate(method=m)
    out.append(r["risk_regimen[always @ t=2]"].psi - r["risk_regimen[never @ t=2]"].psi)
print("elim+enrollment censoring diff mean", round(float(np.mean(out)), 4), "sd", round(float(np.std(out)), 4), "functional -0.0403")
