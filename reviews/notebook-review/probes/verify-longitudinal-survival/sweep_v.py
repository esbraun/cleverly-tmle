"""Verifier sweep: death-as-censoring fit vs the functional, and a censored competing fit."""
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
import cleverly
from cleverly import CausalStudy, CrossFitting, LongitudinalTreatment, ModelSpec, RegimeMean, Runtime, TMLEMethod
from cleverly.datasets import make_longitudinal_competing

warnings.filterwarnings("ignore")
assert "bridge-cse" in cleverly.__file__, cleverly.__file__
F_NEVER, F_ALWAYS = 0.3049, 0.2646  # death_first functional (functional.py)
T_NEVER, T_ALWAYS = 0.2443, 0.2467  # total effect readmission t=2

method = TMLEMethod(
    models=ModelSpec(
        outcome_learner=LogisticRegression(max_iter=1000, random_state=41),
        pseudo_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000, random_state=41),
        censoring_learner=LogisticRegression(max_iter=1000, random_state=41),
    ),
    cross_fitting=CrossFitting(enabled=False),
    runtime=Runtime(random_state=41, n_jobs=1),
)
nodes = dict(treatment=("A1", "A2"), baseline=("W1", "W2"), time_varying=((), ("L2",)))
est = RegimeMean({"always": 1, "never": 0}, reference="never", horizons=(1, 2))


def diff(res, a, b):
    return res.contrast(lambda p: p[0] - p[1], [a, b], name="d")


rows = []
seeds = range(int(sys.argv[1]), int(sys.argv[2]))
for seed in seeds:
    df, truth = make_longitudinal_competing(n=4000, seed=seed, censoring=False)
    comp = CausalStudy(df, design=LongitudinalTreatment(outcome={"readmission": ("R1", "R2"), "death": ("D1", "D2")}, **nodes)).identify(est).estimate(method=method)
    alive1 = df["D1"].eq(0) & df["R1"].eq(0)
    dc = df.assign(
        alive1=1.0 - df["D1"],
        alive2=(1.0 - df["D2"]).where(alive1),
        R1=df["R1"].where(df["D1"].eq(0)),
        R2=df["R2"].where(df["D2"].eq(0)),
    ).drop(columns=["D1", "D2"])
    elim = CausalStudy(dc, design=LongitudinalTreatment(outcome=("R1", "R2"), censoring=("alive1", "alive2"), **nodes)).identify(est).estimate(method=method)
    ed = diff(elim, "risk_regimen[always @ t=2]", "risk_regimen[never @ t=2]")
    cd = diff(comp, "cif_regimen[always, readmission @ t=2]", "cif_regimen[never, readmission @ t=2]")
    # censored competing fit (LS-01 fix)
    dfc, truthc = make_longitudinal_competing(n=4000, seed=seed, censoring=True)
    cc = CausalStudy(dfc, design=LongitudinalTreatment(outcome={"readmission": ("R1", "R2"), "death": ("D1", "D2")}, censoring=("C1", "C2"), **nodes)).identify(est).estimate(method=method)
    ccd = diff(cc, "cif_regimen[always, death @ t=2]", "cif_regimen[never, death @ t=2]")
    nd = cc["cif_regimen[never, death @ t=2]"]
    rows.append(dict(
        seed=seed,
        elim_never=elim["risk_regimen[never @ t=2]"].psi, elim_never_se=elim["risk_regimen[never @ t=2]"].std_error,
        elim_always=elim["risk_regimen[always @ t=2]"].psi,
        elim_diff=ed.psi, elim_diff_se=ed.std_error, elim_lo=ed.ci[0], elim_hi=ed.ci[1],
        comp_diff=cd.psi,
        cc_death_diff=ccd.psi, cc_death_diff_se=ccd.std_error, cc_lo=ccd.ci[0], cc_hi=ccd.ci[1],
        cc_never_death=nd.psi, cc_never_death_se=nd.std_error,
        cens1=int(dfc["C1"].eq(0).sum()),
    ))
    print(seed, flush=True)
pd.DataFrame(rows).to_csv(f".tmp/notebook-review/verify-longitudinal-survival/sweep_{sys.argv[1]}.csv", index=False)
