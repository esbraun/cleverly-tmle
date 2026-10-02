"""Probe: does cleverly refuse a cross-fitted continuous-outcome fit without q_bounds, under
both fluctuations? And independent truth check by direct counterfactual simulation."""
import warnings

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LinearRegression, LogisticRegression

import cleverly
from cleverly import ATE, CausalStudy, CollaborativeTMLEMethod, CrossFitting, ModelSpec, PointTreatment, Runtime, TMLEMethod, Targeting
from cleverly.datasets import make_instrument

print(cleverly.__file__)

# Independent truth: structural equations re-implemented from the generator docstring/code
rng = np.random.default_rng(12345)
N = 1_000_000
W = rng.normal(size=(N, 3))
eps = rng.normal(size=N)
y1 = 1 + 1 + 1.5 * W[:, 0] + 0.8 * W[:, 2] + eps
y0 = 1 + 0 + 1.5 * W[:, 0] + 0.8 * W[:, 2] + eps
g = expit(0.8 * W[:, 0] + 1.5 * W[:, 1])
a = rng.binomial(1, g)
print("MC ATE", (y1 - y0).mean(), "ATT", (y1 - y0)[a == 1].mean())
yobs = np.where(a == 1, y1, y0)
print("P(A=1)", a.mean(), "Y mean", yobs.mean(), "Y sd", yobs.std())
# Asymptotic SDs at n=2000
n = 2000
p = a.mean()
lo, hi = 0.01471, 0.9853
gt = np.clip(g, lo, hi)
print("EIC sd, full g (truncated):", np.sqrt(np.mean(1 / gt + 1 / (1 - gt)) / n))
print("EIC sd, constant g (plug-in at intercept):", np.sqrt((1 / p + 1 / (1 - p)) / n))
# OLS coefficient sd: sigma^2 / (n E[Var(A | W_linear-proj)])
X = np.column_stack([np.ones(N), W])
beta = np.linalg.lstsq(X, a, rcond=None)[0]
resid = a - X @ beta
print("OLS coef sd (homoskedastic, sigma=1):", np.sqrt(1 / (n * np.mean(resid**2))))
# Unadjusted difference expected
print("unadjusted expected", yobs[a == 1].mean() - yobs[a == 0].mean())

frame, truth = make_instrument(n=2000, seed=44)
study = CausalStudy(frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3")))
effect = study.identify(ATE(reference=0))
models = ModelSpec(outcome_learner=LinearRegression(n_jobs=1), treatment_learner=LogisticRegression(max_iter=1000, random_state=44))
for fluct in ("logistic", "linear"):
    for method_cls in (TMLEMethod, CollaborativeTMLEMethod):
        kw = dict(models=models, cross_fitting=CrossFitting(n_folds=5), runtime=Runtime(random_state=44, n_jobs=1), targeting=Targeting(fluctuation=fluct))
        if method_cls is CollaborativeTMLEMethod:
            kw.update(selection_folds=3, selection_inner_folds=2)
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                r = effect.estimate(method=method_cls(**kw))
            print(fluct, method_cls.__name__, "ACCEPTED psi", r["ate"].psi, [str(x.message)[:120] for x in w])
        except Exception as e:  # noqa: BLE001
            print(fluct, method_cls.__name__, "REFUSED", type(e).__name__, str(e)[:300])
# In-sample, linear fluctuation: does the estimate differ?
for fluct in ("logistic", "linear"):
    r = effect.estimate(method=CollaborativeTMLEMethod(models=models, cross_fitting=CrossFitting(enabled=False), runtime=Runtime(random_state=44, n_jobs=1), targeting=Targeting(fluctuation=fluct), selection_folds=3, selection_inner_folds=2))
    print("in-sample", fluct, r["ate"].psi, r.diagnostics.nuisance_models().selection.selected_covariates)
