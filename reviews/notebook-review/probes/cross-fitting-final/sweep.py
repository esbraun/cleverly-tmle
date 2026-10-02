"""Seed sweep of the in-sample and cross-fitted fits that docs/examples/cross-fitting.ipynb shows.

Each seed ``s`` draws ``navigation_data(n=3000, seed=s)`` and runs the notebook's Step 5 fit
(cross-fitted) and its Step 6 fit (in-sample), its Step 11 assessment, and its Step 13
sensitivity calls, with every ``random_state`` set to ``s``.  The code of each fit is the
notebook's, with ``34`` replaced by ``s``.

Usage: ``python sweep.py <first seed> <count> <workers>``.  The page cites
``python sweep.py 7000 200 12``.  Outputs: ``sweep.csv`` (one row per seed) and ``sweep.log``,
which ``summarize.py`` reads.  Every worker is single-threaded (``OMP_NUM_THREADS=1``,
``n_jobs=1``).
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
COVARIATES = ("discharge_risk", "prior_utilization", "medication_burden", "age")


def one_seed(seed: int) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    from cleverly import (
        ATE,
        CausalStudy,
        CrossFitting,
        ModelSpec,
        PointTreatment,
        Runtime,
        Targeting,
        TMLEMethod,
    )
    from cleverly.datasets import navigation_data

    start = time.perf_counter()
    frame, truth = navigation_data(n=3_000, seed=seed)
    ate = truth["ate"]
    effect = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="transition_score",
            treatment="transition_navigation",
            adjustment=COVARIATES,
        ),
    ).identify(ATE(reference=0))
    flexible = ModelSpec(
        outcome_learner=HistGradientBoostingRegressor(random_state=seed),
        treatment_learner=HistGradientBoostingClassifier(
            max_depth=2,
            learning_rate=0.05,
            max_iter=200,
            l2_regularization=1.0,
            random_state=seed,
        ),
    )
    declared_support = Targeting(q_bounds=(0.0, 1.0))
    row: dict = {"seed": seed}
    fits = {}
    for label, folds in (
        ("cf", CrossFitting(n_folds=5)),
        ("is", CrossFitting(enabled=False)),
    ):
        fitted = effect.estimate(
            method=TMLEMethod(
                models=flexible,
                cross_fitting=folds,
                targeting=declared_support,
                runtime=Runtime(random_state=seed, n_jobs=1),
            )
        )
        fits[label] = fitted
        point = fitted["ate"]
        row |= {
            f"{label}_psi": point.psi,
            f"{label}_se": point.std_error,
            f"{label}_covers": point.ci[0] <= ate <= point.ci[1],
            f"{label}_miss_in_se": abs(ate - point.psi) / point.std_error,
        }

    # Step 11.
    cross_fitted = fits["cf"]
    assessment = cross_fitted.assess()
    row["attention"] = "|".join(item.name for item in assessment.attention)
    nuisance = assessment.report("nuisance_models")
    row["cal_slope"] = nuisance["propensity"].metrics["calibration_slope"]
    support = assessment.report("support")
    row["truncated"] = support.truncated["count"]
    row["min_g"] = support.propensity_quantiles["overall"][0.0]
    row["share_g_below_0.1"] = support.tail_mass[0.1]["below"]
    ess = support.effective_sample_size
    row["cf_ess_treated"] = ess["treated"]["ratio"]
    row["cf_ess_control"] = ess["control"]["ratio"]
    in_ess = fits["is"].diagnostics.support().effective_sample_size
    row["is_ess_min"] = min(arm["ratio"] for arm in in_ess.values())

    # Step 13.
    sensitivity = cross_fitted.sensitivity
    robustness = sensitivity.robustness_value()
    row["rv"] = robustness["rv"]
    row["rva"] = robustness["rva"]
    elements = sensitivity.elements(estimand="ate")
    row["nu2"] = elements.nu2
    bounds = sensitivity.omitted_confounding()
    row |= {
        "bound_lower": bounds.lower,
        "bound_upper": bounds.upper,
        "limit_lower": bounds.ci_lower,
        "limit_upper": bounds.ci_upper,
    }
    row["seconds"] = time.perf_counter() - start
    return row


def main() -> None:
    first, count, workers = (int(value) for value in sys.argv[1:4])
    import cleverly

    assert Path(cleverly.__file__).resolve().is_relative_to(HERE.parents[3] / "src")
    seeds = range(first, first + count)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one_seed, seeds))
    frame = pd.DataFrame(rows).sort_values("seed")
    frame.to_csv(HERE / "sweep.csv", index=False, float_format="%.10g")
    print(f"{len(frame)} seeds, {first} to {first + count - 1}; cleverly {cleverly.__file__}")
    print(f"median seconds per seed: {np.median(frame['seconds']):.1f}")


if __name__ == "__main__":
    main()
