"""Seed sweep of the fits that docs/examples/survey-nonresponse.ipynb shows.

Each seed ``s`` draws ``make_missing_outcome(n=4000, seed=s, strength=2.0)`` and runs the
notebook's Step 6 fit (in-sample TMLE with a linear outcome regression and main-effects logistic
treatment and response models), its Step 7 complete-case fit, its Step 11 tipping gamma (point
and ``use_ci=True``) with the navigation-arm tilt ``arm_gamma={0: 0.0, 1: -1.0}``, and its
Step 8 complete-case fit on ``strength=1.0``.  It also draws
``make_missing_outcome_binary(n=4000, seed=s + 1)`` and runs the Step 9 five-fold fits of
``ate``, ``rr``, and ``or``.  Every ``random_state`` is ``s``; the notebook's code is used with
71 replaced by ``s`` (72 by ``s + 1`` for the data of Step 9).  Truths come from ``truth.py``.

Usage: ``python sweep.py <first seed> <count> <workers>``.  The page cites
``python sweep.py 1000 500 12``.  Outputs: ``sweep.csv`` (one row per seed) and ``sweep.log``.
Every worker is single-threaded (``OMP_NUM_THREADS=1``, ``n_jobs=1``).
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRUE_ATE = 1.2  # truth.py, both strengths
RESPONDENT_TARGET = 1.427936  # truth.py, strength 2
BOX_TRUTH = {"ate": 0.186224, "rr": 1.494714, "or": 2.131168}  # truth.py
ARM_GAMMA = {0: 0.0, 1: -1.0}
COVARIATES = ("discharge_risk", "age", "prior_utilization")
NAMES = {
    "Y": "transition_score",
    "A": "transition_navigation",
    "W1": "discharge_risk",
    "W2": "age",
    "W3": "prior_utilization",
    "Delta": "responded",
}


def one_seed(seed: int) -> dict:
    warnings.filterwarnings("ignore")
    import numpy as np
    from scipy.special import expit
    from sklearn.linear_model import LinearRegression, LogisticRegression

    from cleverly import (
        ATE,
        CausalStudy,
        CrossFitting,
        ModelSpec,
        OddsRatio,
        PointTreatment,
        RiskRatio,
        Runtime,
        TMLEMethod,
    )
    from cleverly.datasets import make_missing_outcome, make_missing_outcome_binary

    start = time.time()
    method = TMLEMethod(
        models=ModelSpec(
            outcome_learner=LinearRegression(n_jobs=1),
            treatment_learner=LogisticRegression(max_iter=1000, random_state=seed),
            missingness_learner=LogisticRegression(max_iter=1000, random_state=seed),
        ),
        cross_fitting=CrossFitting(enabled=False),
        runtime=Runtime(random_state=seed, n_jobs=1),
    )

    def covers(point, value):
        return bool(point.ci[0] <= value <= point.ci[1])

    def complete_case(data):
        kept = data[data["responded"] == 1].drop(columns=["responded"])
        design = PointTreatment(
            outcome="transition_score", treatment="transition_navigation", adjustment=COVARIATES
        )
        fit = CausalStudy(kept, design=design).identify(ATE(reference=0)).estimate(method=method)
        return fit["ate"], kept

    frame, _ = make_missing_outcome(n=4000, seed=seed, strength=2.0)
    frame = frame.rename(columns=NAMES)
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="transition_score",
            treatment="transition_navigation",
            adjustment=COVARIATES,
            missingness="responded",
        ),
    )
    full = study.identify(ATE(reference=0)).estimate(method=method)
    point = full["ate"]
    row = {
        "seed": seed,
        "full_psi": point.psi,
        "full_se": point.std_error,
        "full_covers": covers(point, TRUE_ATE),
    }
    cc, kept = complete_case(frame)
    sample_target = float(np.mean(1.2 - 0.9 * kept["discharge_risk"].to_numpy()))
    row.update(
        cc_psi=cc.psi,
        cc_se=cc.std_error,
        cc_covers_truth=covers(cc, TRUE_ATE),
        cc_covers_sample_target=covers(cc, sample_target),
        cc_covers_population_target=covers(cc, RESPONDENT_TARGET),
        sample_target=sample_target,
        cc_distance_se=(cc.psi - sample_target) / cc.std_error,
        shift_is_most=(sample_target - TRUE_ATE) > (cc.psi - sample_target),
    )
    scaler = full.nuisance.scaler
    score_range = scaler.upper - scaler.lower
    arm_sd = float(
        frame.loc[
            (frame["responded"] == 1) & (frame["transition_navigation"] == 1), "transition_score"
        ].std()
    )
    for label, use_ci in (("point", False), ("ci", True)):
        gamma = full.sensitivity.tipping_gamma(arm_gamma=ARM_GAMMA, use_ci=use_ci)
        row[f"tip_{label}"] = np.nan if gamma is None else float(gamma)
        if gamma is not None:
            shift = (2.0 * expit(gamma / 2.0) - 1.0) * score_range
            row[f"tip_{label}_units"] = shift
            row[f"tip_{label}_sd_ratio"] = shift / arm_sd
    row["score_range"] = score_range
    row["arm_sd"] = arm_sd
    mild_frame, _ = make_missing_outcome(n=4000, seed=seed, strength=1.0)
    mild, _ = complete_case(mild_frame.rename(columns=NAMES))
    row["mild_psi"] = mild.psi
    row["mild_covers"] = covers(mild, TRUE_ATE)

    box_frame, _ = make_missing_outcome_binary(n=4000, seed=seed + 1)
    box_frame = box_frame.rename(columns={**NAMES, "Y": "top_box"})
    box_study = CausalStudy(
        box_frame,
        design=PointTreatment(
            outcome="top_box",
            treatment="transition_navigation",
            adjustment=COVARIATES,
            missingness="responded",
        ),
    )
    box_method = TMLEMethod(
        models=ModelSpec(
            outcome_learner=LogisticRegression(max_iter=1000, random_state=seed),
            treatment_learner=LogisticRegression(max_iter=1000, random_state=seed),
            missingness_learner=LogisticRegression(max_iter=1000, random_state=seed),
        ),
        cross_fitting=CrossFitting(n_folds=5),
        runtime=Runtime(random_state=seed, n_jobs=1),
    )
    for estimand, key in (
        (ATE(reference=0), "ate"),
        (RiskRatio(reference=0), "rr"),
        (OddsRatio(reference=0), "or"),
    ):
        box = box_study.identify(estimand).estimate(method=box_method)[key]
        row[f"box_{key}_psi"] = box.psi
        row[f"box_{key}_covers"] = covers(box, BOX_TRUTH[key])
    row["secs"] = time.time() - start
    return row


def main() -> None:
    import cleverly
    import pandas as pd

    first, count, workers = (int(value) for value in sys.argv[1:4])
    source = Path(cleverly.__file__).resolve()
    assert str(HERE.parents[3]) in str(source), source
    seeds = list(range(first, first + count))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one_seed, seeds))
    frame = pd.DataFrame(rows).sort_values("seed")
    frame.to_csv(HERE / "sweep.csv", index=False, lineterminator="\n")
    log = (
        f"cleverly {cleverly.__version__} from {source}\n"
        f"seeds {first} to {first + count - 1}, {workers} workers, "
        f"total {frame['secs'].sum():.0f} worker-seconds\n"
    )
    (HERE / "sweep.log").write_text(log, encoding="utf-8", newline="\n")
    print(log)


if __name__ == "__main__":
    main()
