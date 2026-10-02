import sys
import warnings

import numpy as np
import pandas as pd

from common import build, hand_tmle, slog
from cleverly import (ATE, CausalStudy, CrossFitting, Inference, ModelSpec, PointTreatment,
                      Runtime, TMLEMethod)

warnings.simplefilter("ignore")
seeds = [int(s) for s in sys.argv[1:]] or [2026] + list(range(30))
rows = []
for seed in seeds:
    d, adj = build(seed)
    n = len(d)
    pkg_lo = 5 / (np.sqrt(n) * np.log(n))
    h01 = hand_tmle(d, adj, seed, 0.01)
    hpk = hand_tmle(d, adj, seed, pkg_lo)
    eff = CausalStudy(d, design=PointTreatment(outcome="mortality_1y", treatment="low_birth_weight",
                                               adjustment=adj, cluster="pair_id",
                                               outcome_family="binomial")).identify(ATE(reference=0))
    m = TMLEMethod(models=ModelSpec(outcome_learner=slog(seed), treatment_learner=slog(seed)),
                   cross_fitting=CrossFitting(enabled=False),
                   inference=Inference(alpha=0.05, simultaneous=False),
                   runtime=Runtime(random_state=seed, n_jobs=1))
    psi = eff.estimate(method=m)["ate"].psi
    shift = h01["tmle"] - h01["gcomp"]
    r = dict(seed=seed, gmin=h01["gmin"], gmax=h01["gmax"], n_out01=h01["n_out"],
             gap01=psi - h01["tmle"], shift01=shift, ratio01=abs(psi - h01["tmle"]) / abs(shift),
             gappk=psi - hpk["tmle"], shiftpk=hpk["tmle"] - hpk["gcomp"],
             ratiopk=abs(psi - hpk["tmle"]) / abs(hpk["tmle"] - hpk["gcomp"]),
             sfin=h01["sfin"], s0=h01["s0"], eps=h01["eps"])
    rows.append(r)
    print(r, flush=True)
df = pd.DataFrame(rows)
df.to_csv(f"sweep_gap_{seeds[0]}.csv", index=False, float_format="%.17g")
print("assert fails (0.01 clip):", int((df.ratio01 >= 0.1).sum()), "of", len(df))
print("assert fails (pkg bound):", int((df.ratiopk >= 0.1).sum()), "of", len(df))
print("final score exactly 0:", int((df.sfin == 0).sum()))
