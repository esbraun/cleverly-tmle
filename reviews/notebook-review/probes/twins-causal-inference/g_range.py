"""In-sample hand-built propensity range per pair-sample seed: when does the 0.01 clip bind?"""
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "C:/Users/erics/Documents/Projects/cleverly-tmle/.claude/worktrees/bridge-cse_01BbvRGwt6wTxoguDudAmLTi/.tmp/notebook-review/twins-causal-inference")
from point_sweep import build

rows = []
for seed in [2026] + list(range(30)):
    pdat, ADJ = build(seed)
    W = pdat.loc[:, ADJ]; A = pdat["low_birth_weight"].to_numpy()
    g = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=seed)).fit(W, A).predict_proba(W)[:, 1]
    bound = 5 / (np.sqrt(len(g)) * np.log(len(g)))
    rows.append(dict(seed=seed, gmin=g.min(), gmax=g.max(), n_outside_01=int(((g < 0.01) | (g > 0.99)).sum()),
                     n_outside_pkg=int(((g < bound) | (g > 1 - bound)).sum())))
    print(rows[-1]); sys.stdout.flush()
df = pd.DataFrame(rows)
sw = df[df.seed != 2026]
print("share of 30 seeds where the 0.01 clip binds:", (sw.n_outside_01 > 0).mean(), " where package bound binds:", (sw.n_outside_pkg > 0).mean())
