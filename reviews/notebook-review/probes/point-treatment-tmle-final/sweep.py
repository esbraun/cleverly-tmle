"""Seed sweep of the configurations that docs/examples/point-treatment-tmle.ipynb shows.

Each seed ``s`` draws ``navigation_data(n=3000, seed=s)`` and runs the notebook's Step 6 fit,
its Step 8 fits, its Step 9 assessment and truncation curve, its Step 10 sensitivity calls (all
four benchmarks and the tipping search), and the "Who gains" lesson's ATT, ATC and counterfactual
mean fits, with every ``random_state`` set to ``s``.  The code of each fit is the notebook's,
with ``21`` replaced by ``s``.

Usage: ``python sweep.py <first seed> <count> <workers>``.  The page cites
``python sweep.py 6000 200 4``.  Outputs: ``sweep.csv`` (one row per seed) and ``sweep.log``,
which ``summarize.py`` reads.  Every worker is single-threaded (``OMP_NUM_THREADS=1``,
``n_jobs=1``).
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
COVARIATES = ("discharge_risk", "prior_utilization", "medication_burden", "age")


def one_seed(seed: int) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, LogisticRegression

    from cleverly import (
        ATC,
        ATE,
        ATT,
        CausalStudy,
        CounterfactualMean,
        CrossFitting,
        Inference,
        ModelSpec,
        PointTreatment,
        Runtime,
        Targeting,
        TMLEMethod,
    )
    from cleverly.datasets import navigation_data

    def shallow_booster():
        return HistGradientBoostingClassifier(
            max_depth=2,
            learning_rate=0.05,
            max_iter=200,
            l2_regularization=1.0,
            random_state=seed,
        )

    start = time.perf_counter()
    frame, truth = navigation_data(n=3_000, seed=seed)
    ate = truth["ate"]
    row: dict = {"seed": seed}
    by_arm = frame.groupby("transition_navigation")["transition_score"].mean()
    unadjusted = by_arm.loc[1.0] - by_arm.loc[0.0]
    row["unadjusted"] = unadjusted
    row["unadjusted_closer_to_att"] = abs(unadjusted - truth["att"]) < abs(unadjusted - ate)

    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="transition_score",
            treatment="transition_navigation",
            adjustment=COVARIATES,
        ),
    )
    effect = study.identify(ATE(reference=0))
    flexible = TMLEMethod(
        models=ModelSpec(
            outcome_learner=HistGradientBoostingRegressor(random_state=seed),
            treatment_learner=shallow_booster(),
        ),
        cross_fitting=CrossFitting(n_folds=5),
        targeting=Targeting(q_bounds=(0.0, 1.0)),
        inference=Inference(alpha=0.05),
        runtime=Runtime(random_state=seed, n_jobs=1),
    )
    result = effect.estimate(method=flexible)
    point = result["ate"]
    row |= {
        "psi": point.psi,
        "se": point.std_error,
        "ci_low": point.ci[0],
        "ci_high": point.ci[1],
        "covers": point.ci[0] <= ate <= point.ci[1],
    }

    # Step 8.
    learners = {
        "flex_q_lin_g": (
            HistGradientBoostingRegressor(random_state=seed),
            LogisticRegression(max_iter=1000, random_state=seed),
        ),
        "lin_q_flex_g": (LinearRegression(n_jobs=1), shallow_booster()),
        "both_linear": (
            LinearRegression(n_jobs=1),
            LogisticRegression(max_iter=1000, random_state=seed),
        ),
    }
    for label, (q, g) in learners.items():
        method = replace(flexible, models=ModelSpec(outcome_learner=q, treatment_learner=g))
        fitted = effect.estimate(method=method)["ate"]
        row[f"{label}_psi"] = fitted.psi
        row[f"{label}_covers"] = fitted.ci[0] <= ate <= fitted.ci[1]

    # Step 9.
    assessment = result.assess()
    row["attention"] = "|".join(item.name for item in assessment.attention)
    nuisance = assessment.report("nuisance_models")
    (propensity,) = (m for m in nuisance.models if m.name == "propensity")
    row["cal_slope"] = propensity.metrics["calibration_slope"]
    support = assessment.report("support")
    row["trunc_default"] = support.truncated["fraction"]
    curve = result.diagnostics.truncation_curve()
    for bound in (0.05, 0.1, 0.2):
        at = curve.loc[(curve["bound"] - bound).abs().idxmin()]
        row[f"trunc_{bound}"] = at["truncated_fraction"]
        row[f"psi_move_{bound}"] = at["psi"] - point.psi

    # Step 10.
    sensitivity = result.sensitivity
    robustness = sensitivity.robustness_value()
    row["rv"] = robustness["rv"]
    row["rva"] = robustness["rva"]
    row["nu2"] = sensitivity.elements(estimand="ate").nu2
    records = {name: sensitivity.benchmark(covariates=(name,)) for name in COVARIATES}
    for name, record in records.items():
        row[f"bench_{name}_cf_y"] = record.cf_y
        row[f"bench_{name}_cf_d"] = record.cf_d
    strong = records["discharge_risk"]
    row["strong_cf_y"] = strong.cf_y
    row["strong_gain_y"] = strong.sigma2_short / strong.sigma2_long - 1.0
    row["strong_cf_d"] = strong.cf_d
    strong_bounds = sensitivity.omitted_confounding(benchmark=strong)
    row["strong_bias"] = strong_bounds.bias
    row["strong_limit_lower"] = strong_bounds.ci_lower
    benchmark = records["medication_burden"]
    row["med_cf_y"] = benchmark.cf_y
    row["med_cf_d"] = benchmark.cf_d
    row["med_rho"] = benchmark.rho
    bounds = sensitivity.omitted_confounding(benchmark=benchmark)
    # The page's tipping search: the weight-side strength at which the one-sided lower limit
    # reaches zero for cf_y = 1, by bisection from the largest benchmarked cf_d.
    max_cf_d = max(record.cf_d for record in records.values())
    row["max_bench_cf_d"] = max_cf_d
    row["max_bench_name"] = max(records, key=lambda name: records[name].cf_d)
    low, high = max_cf_d, 0.99
    for _ in range(40):
        middle = (low + high) / 2
        if sensitivity.omitted_confounding(cf_y=1.0, cf_d=middle).ci_lower > 0:
            low = middle
        else:
            high = middle
    row["tipping_cf_d"] = (low + high) / 2
    row |= {
        "bound_lower": bounds.lower,
        "bound_upper": bounds.upper,
        "limit_lower": bounds.ci_lower,
        "limit_upper": bounds.ci_upper,
    }

    # "Who gains": the conditional effects and both counterfactual means on the same fit.
    for estimand in (ATT(), ATC(), CounterfactualMean()):
        fitted = study.identify(estimand).estimate(method=flexible)
        influence = fitted.diagnostics.influence().to_frame().set_index("name")
        for key in fitted.parameter_keys:
            value = fitted[key]
            target = truth[key.replace("[", "").replace("]", "")]
            label = key.replace("[", "").replace("]", "")
            row |= {
                f"{label}_psi": value.psi,
                f"{label}_se": value.std_error,
                f"{label}_ci_low": value.ci[0],
                f"{label}_ci_high": value.ci[1],
                f"{label}_covers": value.ci[0] <= target <= value.ci[1],
                f"{label}_df": value.effective_df,
                f"{label}_kish": influence.loc[key, "kish"],
                f"{label}_variance_rel_sd": influence.loc[key, "variance_rel_sd"],
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
    import sklearn

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE, check=True
    ).stdout.strip()
    print(f"{len(frame)} seeds, {first} to {first + count - 1}; {workers} workers")
    print(f"commit {commit}; cleverly {cleverly.__version__}")
    print(f"numpy {np.__version__}; pandas {pd.__version__}; scikit-learn {sklearn.__version__}")
    print(f"median seconds per seed: {np.median(frame['seconds']):.1f}")


if __name__ == "__main__":
    main()
