"""IV-N3 hypothesis 2: are the held-out ratios from the right fold, at the right dose,
and is the targeted score solved with the same (trimmed) covariate the IC reads?

Refits the pooled-hazard density per fold with an independently written long expansion and
a fresh HistGradientBoostingClassifier, and compares bin probabilities with the library's.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

import cleverly
from cleverly import (
    CausalStudy,
    CrossFitting,
    ModelSpec,
    ModifiedTreatmentPolicyEffect,
    PointTreatment,
    Runtime,
    Targeting,
    TMLEMethod,
)
from cleverly.datasets import make_shift_dose
from cleverly.interventions import Shift

assert Path(cleverly.__file__).resolve().is_relative_to(ROOT / "src"), cleverly.__file__
SHIFTS = (
    (0.0, None, "current practice"),
    (0.5, 5.0, "+0.5 capped at 5"),
    (0.5, None, "+0.5 uncapped"),
    (1.0, None, "+1.0 uncapped"),
)
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 9000
frame, truth = make_shift_dose(n=3000, seed=seed, shifts=SHIFTS)
a = np.asarray(frame["A"], dtype=float)
y = np.asarray(frame["Y"], dtype=float)
w = np.column_stack([np.asarray(frame[c], dtype=float) for c in ("W1", "W2", "W3")])
effect = CausalStudy(
    frame,
    design=PointTreatment(
        outcome="Y", treatment="A", adjustment=("W1", "W2", "W3"), treatment_kind="continuous"
    ),
).identify(ModifiedTreatmentPolicyEffect(tuple(Shift(d, cap=c, name=n) for d, c, n in SHIFTS)))
method = TMLEMethod(
    models=ModelSpec(
        outcome_learner=HistGradientBoostingRegressor(random_state=seed),
        treatment_learner=HistGradientBoostingClassifier(random_state=seed),
        density_bins=40,
    ),
    cross_fitting=CrossFitting(n_folds=3),
    targeting=Targeting(q_bounds=(-30.0, 40.0)),
    runtime=Runtime(random_state=seed, n_jobs=1),
)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    res = effect.estimate(method=method)
nu = res.nuisance
dens = nu.density
edges = np.asarray(dens.edges)
B = edges.size - 1
H = B - 1
bins = np.clip(np.digitize(a, edges) - 1, 0, B - 1)


def design_rows(wv, bidx):
    ind = (bidx[:, None] == np.arange(1, H)[None, :]).astype(float)
    return np.column_stack([wv, bidx.astype(float), ind])


def refit(train):
    X, t = [], []
    for i in train:
        for b in range(min(bins[i], H - 1) + 1):
            X.append((i, b))
            t.append(1.0 if b == bins[i] else 0.0)
    rows = np.array([r for r, _ in X])
    bidx = np.array([b for _, b in X])
    model = HistGradientBoostingClassifier(random_state=seed)
    model.fit(design_rows(w[rows], bidx), np.array(t), sample_weight=np.ones(len(t)))
    return model


def probs(model, test):
    haz = np.column_stack(
        [model.predict_proba(design_rows(w[test], np.full(test.size, b)))[:, 1] for b in range(H)]
    )
    haz = np.clip(haz, 1e-12, 1 - 1e-12)
    surv = np.cumprod(1 - haz, axis=1)
    p = np.empty((test.size, B))
    p[:, 0] = haz[:, 0]
    p[:, 1:H] = haz[:, 1:] * surv[:, :-1]
    p[:, H] = surv[:, -1]
    return p


folds = list(nu.folds)
models = [refit(train) for train, _ in folds]
print(f"seed {seed}: bins {B}, folds {[len(t) for _, t in folds]}")
for v, (train, test) in enumerate(folds):
    lib = dens.bin_probabilities[test]
    for u, m in enumerate(models):
        mine = probs(m, test)
        tag = "own fold" if u == v else "other fold"
        print(
            f"  test fold {v} vs model {u} ({tag}): max|dP| {np.max(np.abs(mine - lib)):.2e}, "
            f"corr log P {np.corrcoef(np.log(mine).ravel(), np.log(lib).ravel())[0, 1]:.4f}"
        )

# ratio and ratio_at from an independent lookup into the library's bin probabilities
P = dens.bin_probabilities
width = np.diff(edges)


def g(x):
    b = np.clip(np.digitize(x, edges) - 1, 0, B - 1)
    out = (x < edges[0]) | (x > edges[-1])
    return np.where(out, 0.0, P[np.arange(x.size), b] / width[b])


ss = nu.shifts
for r, (delta, cap, name) in enumerate(SHIFTS):
    if cap is not None:
        continue
    den = g(a)
    h = np.where(den > 0, g(a - delta) / np.where(den > 0, den, 1), 0.0)
    dd = a + delta
    den_d = g(dd)
    h_d = np.where(den_d > 0, g(dd - delta) / np.where(den_d > 0, den_d, 1), 0.0)
    print(
        f"  {name}: max|ratio - lookup| {np.max(np.abs(ss.ratio[:, r] - h)):.2e}; "
        f"max|ratio_at[:, r, r] - lookup at a+delta| {np.max(np.abs(ss.ratio_at[:, r, r] - h_d)):.2e}"
    )

# targeted score with the trimmed covariate, on the scaled outcome
fl = res.fluctuations["mtp"]
sc = nu.scaler
ys = (y - sc.lower) / (sc.upper - sc.lower)
resid = ys - np.asarray(fl.targeted.observed)
trimmed = ss.design[:, 0, :]
raw = ss.ratio
print("  score with trimmed h:", np.mean(trimmed * resid[:, None], axis=0))
print("  score with raw h    :", np.mean(raw * resid[:, None], axis=0))
print("  ceiling", ss.ceiling, "trimmed rows", ss.trimmed)
