"""Seed sweep of the notebook's cells at n=2000 (the notebook's own configuration)."""
import sys
import time
from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import ATE, CausalStudy, CollaborativeTMLEMethod, CrossFitting, ModelSpec, PointTreatment, Runtime, TMLEMethod
from cleverly.datasets import make_instrument

seeds = range(int(sys.argv[1]), int(sys.argv[2]))
rows = []
t0 = time.time()
for seed in seeds:
    frame, truth = make_instrument(n=2_000, seed=seed)
    study = CausalStudy(frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3")))
    effect = study.identify(ATE(reference=0))
    models = ModelSpec(outcome_learner=LinearRegression(n_jobs=1), treatment_learner=LogisticRegression(max_iter=1000, random_state=44))
    folds = CrossFitting(enabled=False)
    runtime = Runtime(random_state=44, n_jobs=1)
    cm = CollaborativeTMLEMethod(models=models, cross_fitting=folds, runtime=runtime, strategy="greedy", selection_folds=3, selection_inner_folds=2)
    pm = TMLEMethod(models=models, cross_fitting=folds, runtime=runtime)
    coll = effect.estimate(method=cm)
    plain = effect.estimate(method=pm)
    weak = ModelSpec(outcome_learner=DummyRegressor(), treatment_learner=LogisticRegression(max_iter=1000, random_state=44))
    wplain = effect.estimate(method=replace(pm, models=weak))
    wcoll = effect.estimate(method=replace(cm, models=weak))
    sel = coll.diagnostics.nuisance_models().selection
    wsel = wcoll.diagnostics.nuisance_models().selection
    # OLS + HC0 via FWL, as in the notebook
    cov = ["W1", "W2", "W3"]
    X = np.column_stack([np.ones(len(frame)), frame[cov].to_numpy()])
    a = frame["A"].to_numpy(); y = frame["Y"].to_numpy()
    full = np.column_stack([X, a])
    res = y - full @ np.linalg.lstsq(full, y, rcond=None)[0]
    ap = a - X @ np.linalg.lstsq(X, a, rcond=None)[0]
    coef = (ap * y).sum() / (ap**2).sum()
    hc0 = np.sqrt((ap**2 * res**2).sum()) / (ap**2).sum()
    psup = plain.diagnostics.support(); csup = coll.diagnostics.support()
    plain_auc = plain.diagnostics.nuisance_models()["propensity"].metrics["auc"]
    rows.append(dict(
        seed=seed, truth=truth["ate"],
        c_psi=coll["ate"].psi, c_pse=coll["ate"].plugin_std_error,
        c_sel="|".join(sel.selected_covariates), c_path=";".join("|".join(p) for p in sel.path),
        c_gap10=sel.cv_risk[1] - sel.cv_risk[0], c_cvrisk=";".join(f"{v:.6f}" for v in sel.cv_risk),
        c_train_rises=any(l > e for e, l in zip(sel.train_risk, sel.train_risk[1:])),
        p_psi=plain["ate"].psi, p_se=plain["ate"].std_error,
        wp_psi=wplain["ate"].psi, wp_se=wplain["ate"].std_error,
        wc_psi=wcoll["ate"].psi, wc_pse=wcoll["ate"].plugin_std_error, wc_sel="|".join(wsel.selected_covariates),
        ols=coef, hc0=hc0, plain_auc=plain_auc,
        p_below=psup.tail_mass[0.1]["below"], p_above=psup.tail_mass[0.1]["above"],
        c_below=csup.tail_mass[0.1]["below"], c_above=csup.tail_mass[0.1]["above"],
        c_ess_t=csup.effective_sample_size["treated"]["ratio"],
    ))
    print(seed, round(time.time() - t0, 1), rows[-1]["c_sel"], rows[-1]["wc_sel"], flush=True)
pd.DataFrame(rows).to_csv(f".tmp/notebook-review/collaborative-tmle/sweep_{seeds.start}_{seeds.stop}.csv", index=False)
