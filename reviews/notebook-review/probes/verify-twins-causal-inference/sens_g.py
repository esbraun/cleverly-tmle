"""Run notebook sections 7 (flexible fit), 8 (assess), 9 (sensitivity) at a given seed."""
import sys
import traceback
import warnings

from sklearn.ensemble import HistGradientBoostingClassifier

from common import build, slog
from cleverly import (ATE, CausalStudy, CrossFitting, Inference, ModelSpec, PointTreatment,
                      Runtime, SuperLearner, TMLEMethod)

warnings.simplefilter("ignore")
seed = int(sys.argv[1])
d, adj = build(seed)


def sl():
    return SuperLearner(
        library=[("logistic", slog(seed)),
                 ("boosting", HistGradientBoostingClassifier(max_iter=100, min_samples_leaf=30,
                                                             random_state=seed))],
        task="classification", n_folds=3, random_state=seed, n_jobs=1)


eff = CausalStudy(d, design=PointTreatment(outcome="mortality_1y", treatment="low_birth_weight",
                                           adjustment=adj, cluster="pair_id",
                                           outcome_family="binomial")).identify(ATE(reference=0))
m = TMLEMethod(models=ModelSpec(outcome_learner=sl(), treatment_learner=sl()),
               cross_fitting=CrossFitting(n_folds=3, learner_folds=3),
               inference=Inference(alpha=0.05, simultaneous=False),
               runtime=Runtime(random_state=seed, n_jobs=1))
res = eff.estimate(method=m)
print("seed", seed, "psi", res["ate"].psi, flush=True)
import numpy as np; g=np.asarray(res.nuisance.propensity.arm(1.0)); A=d["low_birth_weight"].to_numpy(); print("cf g range", g.min(), g.max(), "treated min", g[A==1].min(), "control max", g[A==0].max()); sys.exit(0)
    "omitted_confounding": {"estimand": "ate", "cf_y": 0.05, "cf_d": 0.05, "rho": 1.0}})
frame = assessment.to_frame()
print(frame[frame.astype(str).apply(lambda r: "omitted" in " ".join(r), axis=1)].to_string())
for label, call in [
    ("report", lambda: assessment.report("omitted_confounding").summary()),
    ("benchmark", lambda: res.sensitivity.benchmark(covariates=("preterm", "tobacco"),
                                                    estimand="ate", random_state=seed).summary()),
]:
    try:
        print(label, "OK:\n", call())
    except Exception as exc:  # noqa: BLE001
        print(label, "RAISED", type(exc).__name__, str(exc)[:400])
