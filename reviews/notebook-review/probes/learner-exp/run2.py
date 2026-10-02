"""Strong-positivity navigation law: g_nav = 0.05 + 0.90 * expit(...).  Point-treatment config only."""
import csv
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.special import expit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "rows_strong.csv")
LOG = os.path.join(HERE, "run2.log")
from run import FIELDS, make_learner  # noqa: E402


def g_nav(w):
    return 0.05 + 0.90 * expit(
        0.6 * w[:, 0] - 0.4 * w[:, 1] ** 2 + 0.5 * w[:, 1] * w[:, 2] + 0.3 * (w[:, 3] > 0)
    )


def strong_navigation_data(n, seed):
    from dataclasses import replace

    from cleverly.datasets.navigation import _PROGRAM_COLUMNS
    from cleverly.datasets.synthetic import _make, nonlinear_bounded_dgp

    dgp = replace(nonlinear_bounded_dgp(), propensity=g_nav)
    frame, truth = _make(dgp, n, seed, None)
    return frame.rename(columns=_PROGRAM_COLUMNS), truth


def learner_for(name, rs):
    if name == "e2_logit_poly2":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler

        return make_pipeline(
            StandardScaler(), PolynomialFeatures(2), LogisticRegression(C=1e6, max_iter=5000, random_state=rs)
        )
    return make_learner(name, rs)


def one(task):
    learner, n, seed = task
    warnings.simplefilter("ignore")
    from sklearn.ensemble import HistGradientBoostingRegressor

    from cleverly import (
        ATE, CausalStudy, CrossFitting, Inference, ModelSpec, PointTreatment, Runtime,
        Targeting, TMLEMethod,
    )

    rs = 21
    row = dict(config="point_strong", learner=learner, n=n, seed=seed)
    try:
        frame, truth = strong_navigation_data(n, seed)
        row["truth"] = truth["ate"]
        effect = CausalStudy(
            frame,
            design=PointTreatment(
                outcome="transition_score",
                treatment="transition_navigation",
                adjustment=("discharge_risk", "prior_utilization", "medication_burden", "age"),
            ),
        ).identify(ATE(reference=0))
        method = TMLEMethod(
            models=ModelSpec(
                outcome_learner=HistGradientBoostingRegressor(random_state=rs),
                treatment_learner=learner_for(learner, rs),
            ),
            cross_fitting=CrossFitting(n_folds=5),
            targeting=Targeting(q_bounds=(0.0, 1.0)),
            inference=Inference(alpha=0.05),
            runtime=Runtime(random_state=rs, n_jobs=1),
        )
        t0 = time.perf_counter()
        result = effect.estimate(method=method)
        row["fit_s"] = time.perf_counter() - t0
        est = result["ate"]
        lo, hi = est.ci
        row.update(psi=est.psi, se=est.std_error, lo=lo, hi=hi, covers=int(lo <= truth["ate"] <= hi))
        try:
            diag = result.diagnostics.run_all()
            sup = diag.report("support")
            row["g_lower"] = sup.bounds[0]
            row["trunc_frac"] = sup.truncated["fraction"]
            row["trunc_count"] = sup.truncated["count"]
            row["min_g"] = sup.propensity_quantiles["overall"][0.0]
            tf = diag.report("nuisance_models").to_frame()
            sel = tf[tf["model"] == "propensity"] if "model" in tf.columns else tf[tf.iloc[:, 0].astype(str).str.contains("propensity")]
            if len(sel):
                row["cal_slope"] = float(sel["cal_slope" if "cal_slope" in tf.columns else "calibration_slope"].iloc[0])
        except Exception as exc:
            row["error"] = f"diag:{type(exc).__name__}:{exc}"[:200]
        try:
            rv = result.sensitivity.robustness_value()
            row["sens_ok"] = 1
            row["sens_rv"] = rv["rv"]
        except Exception as exc:
            row["sens_ok"] = 0
            row["sens_error"] = type(exc).__name__
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}:{exc}"[:300]
    return row


def main():
    learners = ["a_hgb_default", "b_hgb_regularised", "c_hgb_early_stop", "e2_logit_poly2"]
    tasks = [(l, 3000, s) for s in range(5000, 5200) for l in learners]
    start = time.time()
    with open(OUT, "w", newline="") as fh, open(LOG, "a") as log:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=12) as pool:
            futures = [pool.submit(one, t) for t in tasks]
            for i, fut in enumerate(as_completed(futures), 1):
                writer.writerow(fut.result())
                fh.flush()
                if i % 50 == 0 or i == len(futures):
                    log.write(f"{time.time() - start:7.0f}s  {i}/{len(futures)}\n")
                    log.flush()


if __name__ == "__main__":
    main()
