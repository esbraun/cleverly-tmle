"""Coverage probe for the Step 6 fit of docs/examples/survey-nonresponse.ipynb.

Per seed ``s``, draws ``make_missing_outcome(n=4000, seed=s, strength=2.0)`` and records:

* ``lib``: the Step 6 library fit (in-sample TMLE, linear outcome regression, main-effects
  logistic treatment and response models, every ``random_state`` equal to ``s``), as in
  ``probes/survey-nonresponse-final/sweep.py``.
* ``oracle``: the one-step estimator with the true Q, g and pi of ``missing_outcome_dgp(2.0)``:
  ``mean(Q1 - Q0) + mean(D)``, SE ``sd(IC)/sqrt(n)``.  No nuisance is estimated.

Usage: ``python probe.py <first seed> <count> <workers> <tag>``; writes ``probe-<tag>.csv``.
Every worker is single-threaded (``OMP_NUM_THREADS=1``, ``n_jobs=1``).
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
TRUE_ATE = 1.2  # probes/survey-nonresponse-final/truth.log, strength 2
COVARIATES = ("discharge_risk", "age", "prior_utilization")
NAMES = {
    "Y": "transition_score",
    "A": "transition_navigation",
    "W1": "discharge_risk",
    "W2": "age",
    "W3": "prior_utilization",
    "Delta": "responded",
}


def oracle(frame):
    from cleverly.datasets import missing_outcome_dgp

    law = missing_outcome_dgp(strength=2.0)
    w = frame[["W1", "W2", "W3"]].to_numpy()
    a = frame["A"].to_numpy().astype(float)
    d = frame["Delta"].to_numpy().astype(float)
    y = np.nan_to_num(frame["Y"].to_numpy().astype(float))
    q1 = law.outcome_mean(w, 1.0, None)
    q0 = law.outcome_mean(w, 0.0, None)
    g1 = law.propensity(w)
    p1 = law.missingness(w, 1.0)
    p0 = law.missingness(w, 0.0)
    ic = a * d / (g1 * p1) * (y - q1) - (1 - a) * d / ((1 - g1) * p0) * (y - q0) + q1 - q0
    psi = ic.mean()
    return psi, (ic - psi).std() / np.sqrt(len(y))


def one_seed(seed: int) -> dict:
    warnings.filterwarnings("ignore")
    from sklearn.linear_model import LinearRegression, LogisticRegression

    from cleverly import (
        ATE,
        CausalStudy,
        CrossFitting,
        ModelSpec,
        PointTreatment,
        Runtime,
        TMLEMethod,
    )
    from cleverly.datasets import make_missing_outcome

    raw, _ = make_missing_outcome(n=4000, seed=seed, strength=2.0)
    o_psi, o_se = oracle(raw)
    method = TMLEMethod(
        models=ModelSpec(
            outcome_learner=LinearRegression(n_jobs=1),
            treatment_learner=LogisticRegression(max_iter=1000, random_state=seed),
            missingness_learner=LogisticRegression(max_iter=1000, random_state=seed),
        ),
        cross_fitting=CrossFitting(enabled=False),
        runtime=Runtime(random_state=seed, n_jobs=1),
    )
    study = CausalStudy(
        raw.rename(columns=NAMES),
        design=PointTreatment(
            outcome="transition_score",
            treatment="transition_navigation",
            adjustment=COVARIATES,
            missingness="responded",
        ),
    )
    point = study.identify(ATE(reference=0)).estimate(method=method)["ate"]
    return {
        "seed": seed,
        "lib_psi": point.psi,
        "lib_se": point.std_error,
        "lib_covers": bool(point.ci[0] <= TRUE_ATE <= point.ci[1]),
        "oracle_psi": o_psi,
        "oracle_se": o_se,
        "oracle_covers": bool(abs(o_psi - TRUE_ATE) <= 1.959964 * o_se),
    }


if __name__ == "__main__":
    import cleverly
    import pandas as pd

    first, count, workers = (int(x) for x in sys.argv[1:4])
    tag = sys.argv[4]
    source = Path(cleverly.__file__).resolve()
    assert str(HERE.parents[3]) in str(source), source
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one_seed, range(first, first + count), chunksize=10))
    pd.DataFrame(rows).to_csv(HERE / f"probe-{tag}.csv", index=False, lineterminator="\n")
    print(f"cleverly from {source}; seeds {first}-{first + count - 1}; wrote probe-{tag}.csv")
