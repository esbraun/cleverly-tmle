"""Independent re-implementation of notebook sections 2, 6, 7 for a given seed."""
import os
import sys

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import cleverly

assert "bridge-cse_01BbvRGwt6wTxoguDudAmLTi" in cleverly.__file__, cleverly.__file__

DATA = os.path.join(os.path.dirname(__file__), "..", "twins-causal-inference")
RAW = {
    k: pd.read_csv(os.path.join(DATA, f"twin_pairs_{k}_3years_samesex.csv"))
    for k in ("X", "T", "Y")
}
N_PAIRS = 6000
SHARED = ["mager8", "meduc6", "mrace", "dmar", "mplbir_reg", "data_year", "diabetes", "chyper",
          "phyper", "preterm", "tobacco", "alcohol"]
CATEGORICAL = ["mager8", "meduc6", "mrace", "mplbir_reg", "data_year"]
BINARY = ["dmar", "diabetes", "chyper", "phyper", "preterm", "tobacco", "alcohol"]


def build(seed):
    rows = RAW["X"].sample(n=N_PAIRS, random_state=seed).index.sort_values().to_numpy()
    X = RAW["X"].loc[rows].reset_index(drop=True)
    T = RAW["T"].loc[rows].reset_index(drop=True)
    Y = RAW["Y"].loc[rows].reset_index(drop=True)
    kids = []
    for tw in (0, 1):
        c = X[SHARED].copy()
        c["pair_id"] = np.arange(N_PAIRS)
        c["twin"] = tw
        c["birth_weight_g"] = T[f"dbirwt_{tw}"]
        c["mortality_1y"] = Y[f"mort_{tw}"].astype(int)
        kids.append(c)
    d = pd.concat(kids, ignore_index=True).sort_values(["pair_id", "twin"]).reset_index(drop=True)
    d["low_birth_weight"] = (d["birth_weight_g"] < 2500).astype(int)
    adj, pats = [], set()
    for col in BINARY:
        m = d[col].isna()
        sig = m.to_numpy().tobytes()
        if m.any() and sig not in pats:
            d[f"{col}_missing"] = m.astype(float)
            adj.append(f"{col}_missing")
            pats.add(sig)
        d[col] = d[col].fillna(0).astype(float)
        adj.append(col)
    enc = pd.get_dummies(d[CATEGORICAL].fillna(-1).astype("category"), prefix=CATEGORICAL,
                         dtype=float)
    d = pd.concat([d, enc], axis=1)
    adj.extend(enc.columns.tolist())
    return d, tuple(adj)


def slog(seed):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=seed))


def expit(v):
    return 1 / (1 + np.exp(-v))


def logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def hand_tmle(d, adj, seed, lo):
    W = d.loc[:, adj]
    A = d["low_birth_weight"].to_numpy()
    y = d["mortality_1y"].to_numpy()
    X = pd.concat([d[["low_birth_weight"]], W], axis=1)
    qm = slog(seed).fit(X, y)
    qo = qm.predict_proba(X)[:, 1]
    q1 = qm.predict_proba(X.assign(low_birth_weight=1))[:, 1]
    q0 = qm.predict_proba(X.assign(low_birth_weight=0))[:, 1]
    gm = slog(seed).fit(W, A)
    graw = gm.predict_proba(W)[:, 1]
    g1 = np.clip(graw, lo, 1 - lo)
    H = A / g1 - (1 - A) / (1 - g1)
    s0 = np.mean(H * (y - qo))
    eps = 0.0
    for _ in range(50):
        qs = expit(logit(qo) + eps * H)
        step = np.mean(H * (y - qs)) / np.mean(H**2 * qs * (1 - qs))
        eps += step
        if abs(step) < 1e-12:
            break
    qs = expit(logit(qo) + eps * H)
    tm = expit(logit(q1) + eps / g1).mean() - expit(logit(q0) - eps / (1 - g1)).mean()
    fs = np.mean(H * (y - qs))
    return dict(gcomp=q1.mean() - q0.mean(), tmle=tm, eps=eps, s0=s0, sfin=fs,
                gmin=graw.min(), gmax=graw.max(),
                n_out=int(((graw < lo) | (graw > 1 - lo)).sum()))
