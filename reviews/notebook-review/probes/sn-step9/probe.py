"""Coverage probe for Step 9 of docs/examples/survey-nonresponse.ipynb.

Per seed s and sample size n, draws ``make_missing_outcome_binary(n, seed=s)`` and records:

* ``lib_cv``: the library fit of Step 9 (5 folds, sklearn LogisticRegression nuisances), ATE;
  its psi, reported SE, and an SE recomputed from the EIF built from the library's own
  cross-fitted g, pi and targeted Q* (``recomp_se``), plus the max |reported IC - recomputed D|.
* ``lib_in``: the same library fit in sample (``CrossFitting(enabled=False)``).
* ``own_cv`` / ``own_in``: an independent TMLE written here (unpenalized IRLS logistic
  nuisances, the pooled two-coefficient logistic fluctuation on respondents), cross-fitted with
  5 folds and in sample.
* ``oracle``: the same independent targeting with the true g, pi and Q.

Usage: ``python probe.py <n> <first seed> <count> <workers> <tag>``; writes ``probe-<tag>.csv``.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TRUTH = 0.186224486509527  # truth.py of survey-nonresponse-final: ey1 - ey0


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    return np.log(p / (1.0 - p))


def irls(x, y, offset=None, iters=50):
    """Unpenalized logistic regression by Newton; ``x`` includes any intercept column."""
    off = np.zeros(len(y)) if offset is None else offset
    beta = np.zeros(x.shape[1])
    for _ in range(iters):
        p = expit(x @ beta + off)
        w = p * (1 - p)
        grad = x.T @ (y - p)
        hess = (x * w[:, None]).T @ x
        step = np.linalg.solve(hess, grad)
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta


def design(w, a=None):
    cols = [np.ones(len(w))]
    if a is not None:
        cols.append(a)
    cols += [w[:, 0], w[:, 1], w[:, 2]]
    return np.column_stack(cols)


def target(y, a, d, q1, q0, g1, p1, p0):
    """Pooled logistic fluctuation, one coefficient per arm, fitted on respondents."""
    g0 = 1 - g1
    h1 = a * d / (g1 * p1)
    h0 = (1 - a) * d / (g0 * p0)
    r = d == 1
    qa = np.where(a == 1, q1, q0)
    eps = irls(np.column_stack([h0[r], h1[r]]), y[r], offset=logit(qa[r]))
    q1s = expit(logit(q1) + eps[1] / (g1 * p1))
    q0s = expit(logit(q0) + eps[0] / (g0 * p0))
    yy = np.where(r, y, 0.0)
    psi = q1s.mean() - q0s.mean()
    ic = h1 * (yy - q1s) - h0 * (yy - q0s) + q1s - q0s - psi
    return psi, ic.std() / np.sqrt(len(y))


def own_nuisance(w, a, d, y, train, test):
    """Fit Q, g, pi on ``train`` (Q on respondents) and predict on ``test``."""
    resp = train & (d == 1)
    bq = irls(design(w[resp], a[resp]), y[resp])
    bg = irls(design(w[train]), a[train])
    bp = irls(design(w[train], a[train]), d[train])
    wt = w[test]
    one, zero = np.ones(len(wt)), np.zeros(len(wt))
    return (
        expit(design(wt, one) @ bq),
        expit(design(wt, zero) @ bq),
        expit(design(wt) @ bg),
        expit(design(wt, one) @ bp),
        expit(design(wt, zero) @ bp),
    )


def one(args):
    n, seed = args
    warnings.filterwarnings("ignore")
    from sklearn.linear_model import LogisticRegression

    from cleverly import ATE, CausalStudy, CrossFitting, ModelSpec, PointTreatment, Runtime
    from cleverly import TMLEMethod
    from cleverly.datasets import make_missing_outcome_binary

    frame, _ = make_missing_outcome_binary(n=n, seed=seed)
    w = frame[["W1", "W2", "W3"]].to_numpy()
    a = frame["A"].to_numpy().astype(float)
    d = frame["Delta"].to_numpy().astype(float)
    y = np.nan_to_num(frame["Y"].to_numpy().astype(float))
    row = {"n": n, "seed": seed}
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y", treatment="A", adjustment=("W1", "W2", "W3"), missingness="Delta"
        ),
    )
    for label, cf in (("lib_cv", CrossFitting(n_folds=5)), ("lib_in", CrossFitting(enabled=False))):
        method = TMLEMethod(
            models=ModelSpec(
                outcome_learner=LogisticRegression(max_iter=1000, random_state=seed),
                treatment_learner=LogisticRegression(max_iter=1000, random_state=seed),
                missingness_learner=LogisticRegression(max_iter=1000, random_state=seed),
            ),
            cross_fitting=cf,
            runtime=Runtime(random_state=seed, n_jobs=1),
        )
        fit = study.identify(ATE(reference=0)).estimate(method=method)
        p = fit["ate"]
        row[f"{label}_psi"], row[f"{label}_se"] = p.psi, p.std_error
        nu = fit.nuisance
        g1 = nu.propensity.values[:, 1]
        pi = nu.missingness
        tq = fit.fluctuations["mean"].targeted.arms
        q1s, q0s = tq[1.0], tq[0.0]
        psi = q1s.mean() - q0s.mean()
        dd = (
            a * d / (g1 * pi[:, 1]) * (y - q1s)
            - (1 - a) * d / ((1 - g1) * pi[:, 0]) * (y - q0s)
            + q1s
            - q0s
            - psi
        )
        row[f"{label}_recomp_psi"] = psi
        row[f"{label}_recomp_se"] = dd.std() / np.sqrt(n)
        row[f"{label}_ic_maxdiff"] = float(np.max(np.abs(np.asarray(p.influence_curve) - dd)))
        row[f"{label}_min_gpi"] = float(
            min(np.min(g1 * pi[:, 1]), np.min((1 - g1) * pi[:, 0]))
        )
    # independent implementation, cross-fitted and in sample
    rng = np.random.default_rng(seed + 10_000_000)
    fold = rng.permutation(np.arange(n) % 5)
    preds = [np.empty(n) for _ in range(5)]
    for v in range(5):
        test = fold == v
        for arr, val in zip(preds, own_nuisance(w, a, d, y, ~test, test)):
            arr[test] = val
    row["own_cv_psi"], row["own_cv_se"] = target(y, a, d, *preds)
    allrows = np.ones(n, bool)
    row["own_in_psi"], row["own_in_se"] = target(
        y, a, d, *own_nuisance(w, a, d, y, allrows, allrows)
    )
    # oracle nuisances, same targeting
    q1 = expit(-0.6 + 0.9 + 0.7 * w[:, 0] - 0.5 * w[:, 1] + 0.4 * w[:, 2])
    q0 = expit(-0.6 + 0.7 * w[:, 0] - 0.5 * w[:, 1] + 0.4 * w[:, 2])
    g1 = expit(0.4 * w[:, 0] - 0.3 * w[:, 1] + 0.2 * w[:, 2])
    p1 = expit(1.5 - 0.9 * w[:, 0] + 0.4 * w[:, 1])
    p0 = expit(1.0 - 0.9 * w[:, 0] + 0.4 * w[:, 1])
    row["oracle_psi"], row["oracle_se"] = target(y, a, d, q1, q0, g1, p1, p0)
    # the oracle EIF at the truth (no estimation at all)
    eif = (
        a * d / (g1 * p1) * (y - q1) - (1 - a) * d / ((1 - g1) * p0) * (y - q0) + q1 - q0 - TRUTH
    )
    row["eif_se"] = eif.std() / np.sqrt(n)
    return row


def main():
    import cleverly
    import pandas as pd

    n, first, count, workers, tag = sys.argv[1:6]
    n, first, count, workers = int(n), int(first), int(count), int(workers)
    src = Path(cleverly.__file__).resolve()
    assert str(HERE.parents[3]) in str(src), src
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one, [(n, s) for s in range(first, first + count)]))
    pd.DataFrame(rows).to_csv(HERE / f"probe-{tag}.csv", index=False, lineterminator="\n")
    print(f"done {tag}: cleverly from {src}")


if __name__ == "__main__":
    main()
