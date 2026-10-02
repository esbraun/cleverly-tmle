"""Sweep the point-treatment sections over the pair-sample seed (learner seeds follow it)."""
import re
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cleverly import ATE, CausalStudy, CrossFitting, Inference, ModelSpec, PointTreatment, Runtime, SuperLearner, TMLEMethod

D = "C:/Users/erics/Documents/Projects/cleverly-tmle/.claude/worktrees/bridge-cse_01BbvRGwt6wTxoguDudAmLTi/.tmp/notebook-review/twins-causal-inference/"
RAW = {k: pd.read_csv(D + f"twin_pairs_{k}_3years_samesex.csv") for k in ("X", "T", "Y")}
N_PAIRS = 6000
shared = ["mager8", "meduc6", "mrace", "dmar", "mplbir_reg", "data_year", "diabetes", "chyper", "phyper", "preterm", "tobacco", "alcohol"]
CATEGORICAL = ["mager8", "meduc6", "mrace", "mplbir_reg", "data_year"]
BINARY = ["dmar", "diabetes", "chyper", "phyper", "preterm", "tobacco", "alcohol"]


def build(seed):
    rows = RAW["X"].sample(n=N_PAIRS, random_state=seed).index.sort_values().to_numpy()
    X, T, Y = (RAW[k].loc[rows].reset_index(drop=True) for k in ("X", "T", "Y"))
    children = []
    for twin in (0, 1):
        c = X[shared].copy(); c["pair_id"] = np.arange(N_PAIRS); c["twin"] = twin
        c["birth_weight_g"] = T[f"dbirwt_{twin}"]; c["mortality_1y"] = Y[f"mort_{twin}"].astype(int)
        children.append(c)
    tw = pd.concat(children, ignore_index=True).sort_values(["pair_id", "twin"]).reset_index(drop=True)
    tw["low_birth_weight"] = (tw["birth_weight_g"] < 2500).astype(int)
    pdat = tw.copy(); adj = []; pats = set()
    for col in BINARY:
        miss = pdat[col].isna(); sig = miss.to_numpy().tobytes()
        if miss.any() and sig not in pats:
            pdat[f"{col}_missing"] = miss.astype(float); adj.append(f"{col}_missing"); pats.add(sig)
        pdat[col] = pdat[col].fillna(0).astype(float); adj.append(col)
    enc = pd.get_dummies(pdat[CATEGORICAL].fillna(-1).astype("category"), prefix=CATEGORICAL, dtype=float)
    pdat = pd.concat([pdat, enc], axis=1); adj.extend(enc.columns.tolist())
    return pdat, tuple(adj)


def ex(v): return 1 / (1 + np.exp(-v))
def lg(p):
    p = np.clip(p, 1e-12, 1 - 1e-12); return np.log(p / (1 - p))


def num(text, pat):
    m = re.search(pat, text); return float(m.group(1)) if m else np.nan


def run(seed):
    pdat, ADJ = build(seed)
    sl = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=seed))
    W = pdat.loc[:, ADJ]; A = pdat["low_birth_weight"].to_numpy(); Yv = pdat["mortality_1y"].to_numpy()
    des = pd.concat([pdat[["low_birth_weight"]], W], axis=1)
    q = sl().fit(des, Yv); qo = q.predict_proba(des)[:, 1]
    q1 = q.predict_proba(des.assign(low_birth_weight=1))[:, 1]; q0 = q.predict_proba(des.assign(low_birth_weight=0))[:, 1]
    gcomp = q1.mean() - q0.mean()
    g = np.clip(sl().fit(W, A).predict_proba(W)[:, 1], 0.01, 0.99)
    ipw = np.mean(A * Yv / g) - np.mean((1 - A) * Yv / (1 - g))
    H = A / g - (1 - A) / (1 - g); s0 = np.mean(H * (Yv - qo)); eps = 0.0
    for _ in range(50):
        qs = ex(lg(qo) + eps * H); st = np.mean(H * (Yv - qs)) / np.mean(H**2 * qs * (1 - qs)); eps += st
        if abs(st) < 1e-12: break
    tm = ex(lg(q1) + eps / g).mean() - ex(lg(q0) - eps / (1 - g)).mean()
    fs = np.mean(H * (Yv - ex(lg(qo) + eps * H)))
    study = CausalStudy(pdat, design=PointTreatment(outcome="mortality_1y", treatment="low_birth_weight", adjustment=ADJ,
                                                    cluster="pair_id", outcome_family="binomial"))
    eff = study.identify(ATE(reference=0))
    inf = Inference(alpha=0.05, simultaneous=False); rt = Runtime(random_state=seed, n_jobs=1)
    ordm = TMLEMethod(models=ModelSpec(outcome_learner=sl(), treatment_learner=sl()), cross_fitting=CrossFitting(enabled=False), inference=inf, runtime=rt)
    o = eff.estimate(method=ordm)["ate"]
    csl = lambda: SuperLearner(library=[("logistic", sl()), ("boosting", HistGradientBoostingClassifier(max_iter=100, min_samples_leaf=30, random_state=seed))],
                               task="classification", n_folds=3, random_state=seed, n_jobs=1)
    flex = TMLEMethod(models=ModelSpec(outcome_learner=csl(), treatment_learner=csl()), cross_fitting=CrossFitting(n_folds=3, learner_folds=3), inference=inf, runtime=rt)
    fr = eff.estimate(method=flex); m = fr["ate"]
    asm = fr.assess(random_state=seed, arguments={"omitted_confounding": {"estimand": "ate", "cf_y": 0.05, "cf_d": 0.05, "rho": 1.0}})
    att = tuple(i.name for i in asm.attention)
    nm = asm.report("nuisance_models").summary()
    try:
        om = asm.report("omitted_confounding").summary()
    except Exception as exc:
        om = "OMITTED_ERROR " + repr(exc)[:200]
        print("omitted row:", asm.to_frame().query("operation == 'omitted_confounding'").to_dict("records") if "operation" in asm.to_frame() else asm.to_frame().head(20).to_string())
    try:
        bm = fr.sensitivity.benchmark(covariates=("preterm", "tobacco"), estimand="ate", random_state=seed).summary()
    except Exception as exc:
        bm = "BENCHMARK_ERROR " + type(exc).__name__
    sup = asm.report("support").summary()
    return dict(seed=seed, gcomp=gcomp, ipw=ipw, hand_tmle=tm, s0=s0, final_score=fs, eps=eps, pkg=o.psi,
                gap_over_shift=abs(o.psi - tm) / abs(tm - gcomp), move_over_s0=(tm - gcomp) / s0,
                cv=m.psi, lo=m.ci[0], hi=m.ci[1], att=";".join(att),
                flag_prop="propensity: the out-of-fold" in nm, flag_out="outcome: the out-of-fold" in nm, bench_error=bm.startswith("BENCHMARK_ERROR"), omitted_error=om.startswith("OMITTED_ERROR"),
                sign_survives="sign of the effect survives" in om, rv=num(om, r"RV\s+= (\S+):"),
                cf_y=num(bm, r"implied cf_y = (\S+),"), cf_d=num(bm, r"cf_d = (\S+),"),
                gmin=num(sup, r"overall\s+(\S+)"), trunc=num(sup, r"truncated: (\d+) unit"))


if __name__ == "__main__":
    seeds = list(range(int(sys.argv[1]), int(sys.argv[2])))
    recs = []
    for s in seeds:
        t0 = time.time(); r = run(s); r["secs"] = time.time() - t0; recs.append(r); print(r); sys.stdout.flush()
        pd.DataFrame(recs).to_csv(D + f"point_sweep_{sys.argv[1]}.csv", index=False)
    df = pd.DataFrame(recs); sw = df[df.seed != 2026] if len(df) > 1 else df
    print("N sweep", len(sw))
    for c in ["flag_prop", "flag_out", "sign_survives"]:
        print(c, sw[c].mean())
    print("att==nuisance_models only", (sw.att == "nuisance_models").mean())
    print("gap/shift max", sw.gap_over_shift.max(), " move/s0 range", sw.move_over_s0.min(), sw.move_over_s0.max())
    print("cf_y==0 share", (sw.cf_y.round(4) == 0).mean(), " 0.05/cf_d range", (0.05 / sw.cf_d).min(), (0.05 / sw.cf_d).max(), "in (3.5,4.5):", ((0.05 / sw.cf_d > 3.5) & (0.05 / sw.cf_d < 4.5)).mean())
    print("cv psi mean %.4f sd %.4f ; gcomp mean %.4f ipw %.4f hand %.4f" % (sw.cv.mean(), sw.cv.std(), sw.gcomp.mean(), sw.ipw.mean(), sw.hand_tmle.mean()))
    print("final score exactly 0 share", (sw.final_score == 0).mean())
    print("truncated units any", (sw.trunc > 0).mean(), " min g min", sw.gmin.min())
