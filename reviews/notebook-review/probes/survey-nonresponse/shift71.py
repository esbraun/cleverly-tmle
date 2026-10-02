"""Seed 71: the actual mean shift at the tipping gamma, versus the printed upper bound."""
import warnings

import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import ATE, CausalStudy, CrossFitting, ModelSpec, PointTreatment, Runtime, TMLEMethod
from cleverly.datasets import make_missing_outcome
from cleverly.sensitivity.missingness import tipping_gamma

warnings.filterwarnings("ignore")
f, _ = make_missing_outcome(n=4000, seed=71, strength=2.0)
m = TMLEMethod(models=ModelSpec(outcome_learner=LinearRegression(n_jobs=1),
                                treatment_learner=LogisticRegression(max_iter=1000, random_state=71),
                                missingness_learner=LogisticRegression(max_iter=1000, random_state=71)),
               cross_fitting=CrossFitting(enabled=False), runtime=Runtime(random_state=71, n_jobs=1))
r = CausalStudy(f, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3"),
                                         missingness="Delta")).identify(ATE(reference=0)).estimate(method=m)
tg = tipping_gamma(r, arm_gamma={0: 0.0, 1: -1.0})
rep = r.repeats[0]
sc = rep.nuisance.scaler
pi = rep.nuisance.bounded_missingness(r.config.missingness_bound)
arms = rep.nuisance.arms
q1 = rep.fluctuations[next(iter(rep.fluctuations))].targeted.arms[1.0]
p1 = pi[:, arms.index(1.0)]
shift = q1 - expit(logit(q1) - tg)
rng = sc.upper - sc.lower
print("tipping", tg, "range", rng)
print("pointwise shift (score units): mean", shift.mean() * rng, "max", shift.max() * rng,
      "weighted by P(nonresponse|A=1,W)", np.sum((1 - p1) * shift) / np.sum(1 - p1) * rng)
print("ATE move", np.mean((1 - p1) * shift) * rng, "E[1-pi1]", np.mean(1 - p1))
print("fitted Q*(1,W) scaled quantiles", np.quantile(q1, [0, .05, .5, .95, 1]))
y = f["Y"].dropna()
print("observed score SD", y.std(), "navigation-arm observed SD", f.loc[f.A == 1, "Y"].std())
