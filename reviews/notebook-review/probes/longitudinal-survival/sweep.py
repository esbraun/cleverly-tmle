"""Seed sweep of the notebook's three fits at its sample size (n=4000)."""

import json
import sys
import time
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

import cleverly
from cleverly import CausalStudy, CrossFitting, LongitudinalTreatment, ModelSpec, RegimeMean, Runtime, TMLEMethod
from cleverly.datasets import (
    longitudinal_navigation_protocol,
    make_longitudinal_competing,
    make_longitudinal_survival,
)

warnings.filterwarnings("ignore")
assert "bridge-cse" in cleverly.__file__, cleverly.__file__
T = json.load(open(".tmp/notebook-review/longitudinal-survival/truth.json"))

seeds = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else list(range(1000, 1040))
outpath = sys.argv[2] if len(sys.argv) > 2 else ".tmp/notebook-review/longitudinal-survival/sweep.csv"

sequential = TMLEMethod(
    models=ModelSpec(
        outcome_learner=LogisticRegression(max_iter=1000, random_state=41),
        pseudo_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000, random_state=41),
        censoring_learner=LogisticRegression(max_iter=1000, random_state=41),
    ),
    cross_fitting=CrossFitting(enabled=False),
    runtime=Runtime(random_state=41, n_jobs=1),
)
nodes = {
    "treatment": ("navigation_p1", "navigation_p2"),
    "baseline": ("age", "baseline_readiness"),
    "time_varying": ((), ("identified_needs",)),
}
plans = {"always": 1, "never": 0}
levels = RegimeMean(plans, reference="never", horizons=(1, 2))
program = longitudinal_navigation_protocol()


def diff(result, a, b):
    return result.contrast(lambda psi: psi[0] - psi[1], [a, b], name=f"{a} - {b}")


rows = []
for seed in seeds:
    t0 = time.time()
    rec = {"seed": seed}
    # ---- retention fit (clustered, censored)
    f, tr = make_longitudinal_survival(n=4_000, seed=seed, cluster_size=20)
    f = f.rename(columns={"W1": "age", "W2": "baseline_readiness", "A1": "navigation_p1",
                          "C1": "tracked_p1", "Y1": "plan_exit_p1", "L2": "identified_needs",
                          "A2": "navigation_p2", "C2": "tracked_p2", "Y2": "plan_exit_p2",
                          "id": "navigator_team"})
    # naive follower comparison exactly as the notebook
    first = {"always": f["navigation_p1"].eq(1), "never": f["navigation_p1"].eq(0)}
    second = {"always": ~f["navigation_p2"].eq(0), "never": ~f["navigation_p2"].eq(1)}
    fol = {1: first, 2: {p: first[p] & second[p] for p in first}}
    for h in (1, 2):
        y = f[f"plan_exit_p{h}"]
        sh = {p: float(y[r & y.notna()].mean()) for p, r in fol[h].items()}
        rec[f"naive_t{h}"] = sh["always"] - sh["never"]
    study = CausalStudy(f, design=LongitudinalTreatment(
        outcome=("plan_exit_p1", "plan_exit_p2"), censoring=("tracked_p1", "tracked_p2"),
        cluster="navigator_team", **nodes), protocol=program)
    r = study.identify(levels).estimate(method=sequential)
    for p in plans:
        for h in (1, 2):
            e = r[f"risk_regimen[{p} @ t={h}]"]
            rec[f"exit_{p}_t{h}"], rec[f"exit_{p}_t{h}_se"] = e.psi, e.std_error
            lo, hi = e.ci
            rec[f"exit_{p}_t{h}_cov"] = lo <= T[f"surv {p} t{h}"] <= hi
    for h in (1, 2):
        d = diff(r, f"risk_regimen[always @ t={h}]", f"risk_regimen[never @ t={h}]")
        rec[f"exit_diff_t{h}"], rec[f"exit_diff_t{h}_se"] = d.psi, d.std_error
        lo, hi = d.ci
        rec[f"exit_diff_t{h}_cov"] = lo <= T[f"diff surv a-n t{h}"] <= hi
    if r.simultaneous is not None:
        rec["band_all_cover"] = all(
            lo <= T[f"surv {n.split('[')[1].split(' ')[0]} t{n[-2]}"] <= hi
            for n, (lo, hi) in r.simultaneous.bands.items()
        )
    sup = r.assess().report("support").to_frame()
    rec["exit_min_ess_ratio"] = float((sup["effective_n"] / sup["n_followed"]).min())
    rec["exit_max_trunc"] = float(sup["share_truncated"].max())
    # ---- competing fit (no censoring)
    g, _ = make_longitudinal_competing(n=4_000, seed=seed + 1, censoring=False)
    g = g.rename(columns={"W1": "age", "W2": "baseline_readiness", "A1": "navigation_p1",
                          "L2": "identified_needs", "A2": "navigation_p2", "D1": "death_p1",
                          "D2": "death_p2", "R1": "readmission_p1", "R2": "readmission_p2"})
    est = CausalStudy(g, design=LongitudinalTreatment(
        outcome={"readmission": ("readmission_p1", "readmission_p2"),
                 "death": ("death_p1", "death_p2")}, **nodes), protocol=program
    ).identify(levels).estimate(method=sequential)
    key = {"readmission": "relapse", "death": "death"}
    for p in plans:
        for c in ("readmission", "death"):
            for h in (1, 2):
                e = est[f"cif_regimen[{p}, {c} @ t={h}]"]
                tv = T[f"cif {key[c]} {p} t{h}"]
                rec[f"cif_{p}_{c}_t{h}"], rec[f"cif_{p}_{c}_t{h}_se"] = e.psi, e.std_error
                lo, hi = e.ci
                rec[f"cif_{p}_{c}_t{h}_cov"] = lo <= tv <= hi
    for c in ("readmission", "death"):
        for h in (1, 2):
            d = diff(est, f"cif_regimen[always, {c} @ t={h}]", f"cif_regimen[never, {c} @ t={h}]")
            rec[f"cifd_{c}_t{h}"], rec[f"cifd_{c}_t{h}_se"] = d.psi, d.std_error
            lo, hi = d.ci
            rec[f"cifd_{c}_t{h}_cov"] = lo <= T[f"diff cif {key[c]} a-n t{h}"] <= hi
    tot = est.incidence_total()
    rec["excess_max"] = float(tot["excess"].max())
    esup = est.assess().report("support").to_frame()
    i = esup["epsilon"].abs().idxmax()
    rec["eps_max_row"] = f"{esup.loc[i, 'regimen']}/{esup.loc[i, 'cause']}/{esup.loc[i, 'horizon']}/{esup.loc[i, 'time']}"
    # ---- death-as-censoring fit, exactly as the notebook recodes it
    alive1 = g["death_p1"].eq(0) & g["readmission_p1"].eq(0)
    dac = g.assign(
        alive_p1=1.0 - g["death_p1"],
        alive_p2=(1.0 - g["death_p2"]).where(alive1),
        readmission_p1=g["readmission_p1"].where(g["death_p1"].eq(0)),
        readmission_p2=g["readmission_p2"].where(g["death_p2"].eq(0)),
    )
    el = CausalStudy(dac, design=LongitudinalTreatment(
        outcome=("readmission_p1", "readmission_p2"), censoring=("alive_p1", "alive_p2"), **nodes),
        protocol=program).identify(levels).estimate(method=sequential)
    for p in plans:
        for h in (1, 2):
            e = el[f"risk_regimen[{p} @ t={h}]"]
            rec[f"el_{p}_t{h}"], rec[f"el_{p}_t{h}_se"] = e.psi, e.std_error
            lo, hi = e.ci
            rec[f"el_{p}_t{h}_cov_cde"] = lo <= T[f"cde_dfirst {p} t{h}"] <= hi
    d = diff(el, "risk_regimen[always @ t=2]", "risk_regimen[never @ t=2]")
    rec["el_diff_t2"], rec["el_diff_t2_se"] = d.psi, d.std_error
    lo, hi = d.ci
    rec["el_diff_t2_cov_cde"] = lo <= T["diff cde_dfirst a-n t2"] <= hi
    rec["el_diff_t2_cov_total"] = lo <= T["diff cif relapse a-n t2"] <= hi
    rec["secs"] = time.time() - t0
    rows.append(rec)
    print(seed, f"{rec['secs']:.1f}s", flush=True)
    pd.DataFrame(rows).to_csv(outpath, index=False)
