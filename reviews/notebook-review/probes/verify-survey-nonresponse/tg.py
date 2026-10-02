import sys, time, numpy as np, cleverly
assert "bridge-cse" in cleverly.__file__
from sklearn.linear_model import LinearRegression, LogisticRegression
from cleverly import ATE, CausalStudy, PointTreatment, CrossFitting, ModelSpec, Runtime, TMLEMethod
from cleverly.datasets import make_missing_outcome
from cleverly.sensitivity import tipping_gamma
def run(seed, n, qb=None):
    f, t = make_missing_outcome(n=n, seed=seed, strength=2.0)
    kw = {}
    m = TMLEMethod(models=ModelSpec(outcome_learner=LinearRegression(n_jobs=1),
        treatment_learner=LogisticRegression(max_iter=1000, random_state=seed),
        missingness_learner=LogisticRegression(max_iter=1000, random_state=seed)),
        cross_fitting=CrossFitting(enabled=False), runtime=Runtime(random_state=seed, n_jobs=1))
    r = CausalStudy(f, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1","W2","W3"), missingness="Delta")).identify(ATE(reference=0)).estimate(method=m)
    tg = tipping_gamma(r, "ate", arm_gamma={0: 0.0, 1: -1.0})
    return r["ate"].psi, tg, r.nuisance.scaler.range
mode = sys.argv[1]
if mode == "71":
    print(run(71, 4000))
elif mode == "seeds":
    out = [run(s, 4000) for s in range(5000, 5060)]
    tg = np.array([o[1] for o in out]); rg = np.array([o[2] for o in out])
    print("n=4000 60 seeds tg mean %.3f median %.3f p5 %.3f p95 %.3f min %.3f max %.3f frac>=1.306 %.3f" % (tg.mean(), np.median(tg), *np.quantile(tg,[.05,.95]), tg.min(), tg.max(), (tg>=1.306).mean()))
    print("range mean %.2f min %.2f max %.2f corr(tg,range) %.2f" % (rg.mean(), rg.min(), rg.max(), np.corrcoef(tg, rg)[0,1]))
elif mode == "n":
    for n in (1000, 4000, 16000, 64000):
        out = [run(s, n) for s in range(6000, 6008)]
        print(n, "mean tg %.3f mean range %.2f" % (np.mean([o[1] for o in out]), np.mean([o[2] for o in out])), flush=True)
