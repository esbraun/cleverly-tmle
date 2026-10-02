"""Treatment-learner coverage sweep on navigation_data.  Run with OMP/BLAS threads = 1."""
import csv
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "rows.csv")
LOG = os.path.join(HERE, "run.log")

FIELDS = [
    "config", "learner", "n", "seed", "psi", "se", "lo", "hi", "covers", "truth",
    "g_lower", "trunc_frac", "trunc_count", "min_g", "cal_slope",
    "sens_ok", "sens_rv", "sens_error", "fit_s", "error",
]

CONFIG_SEED = {"point": 21, "cross": 34}


def make_learner(name, rs):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer

    sys.path.insert(0, HERE)
    from feats import true_terms

    if name == "a_hgb_default":
        return HistGradientBoostingClassifier(random_state=rs)
    if name == "b_hgb_regularised":
        return HistGradientBoostingClassifier(
            max_depth=2, learning_rate=0.05, max_iter=200, l2_regularization=1.0, random_state=rs
        )
    if name == "c_hgb_early_stop":
        return HistGradientBoostingClassifier(
            early_stopping=True, validation_fraction=0.2, n_iter_no_change=10, random_state=rs
        )
    if name == "d_logit_main":
        return LogisticRegression(max_iter=1000, random_state=rs)
    if name == "e_logit_true":
        return make_pipeline(
            FunctionTransformer(true_terms), LogisticRegression(C=1e6, max_iter=1000, random_state=rs)
        )
    raise ValueError(name)


def one(task):
    config, learner, n, seed = task
    warnings.simplefilter("ignore")
    from sklearn.ensemble import HistGradientBoostingRegressor

    from cleverly import (
        ATE, CausalStudy, CrossFitting, Inference, ModelSpec, PointTreatment, Runtime,
        Targeting, TMLEMethod,
    )
    from cleverly.datasets import navigation_data

    rs = CONFIG_SEED[config]
    row = dict(config=config, learner=learner, n=n, seed=seed)
    try:
        frame, truth = navigation_data(n=n, seed=seed)
        row["truth"] = truth["ate"]
        effect = CausalStudy(
            frame,
            design=PointTreatment(
                outcome="transition_score",
                treatment="transition_navigation",
                adjustment=("discharge_risk", "prior_utilization", "medication_burden", "age"),
            ),
        ).identify(ATE(reference=0))
        models = ModelSpec(
            outcome_learner=HistGradientBoostingRegressor(random_state=rs),
            treatment_learner=make_learner(learner, rs),
        )
        if config == "point":
            method = TMLEMethod(
                models=models,
                cross_fitting=CrossFitting(n_folds=5),
                targeting=Targeting(q_bounds=(0.0, 1.0)),
                inference=Inference(alpha=0.05),
                runtime=Runtime(random_state=rs, n_jobs=1),
            )
        else:
            method = TMLEMethod(
                models=models,
                cross_fitting=CrossFitting(n_folds=5),
                targeting=Targeting(q_bounds=(0.0, 1.0)),
                runtime=Runtime(random_state=rs, n_jobs=1),
            )
        t0 = time.perf_counter()
        result = effect.estimate(method=method)
        row["fit_s"] = time.perf_counter() - t0
        est = result["ate"]
        lo, hi = est.ci
        row.update(psi=est.psi, se=est.std_error, lo=lo, hi=hi,
                   covers=int(lo <= truth["ate"] <= hi))
        try:
            diag = result.diagnostics.run_all()
            sup = diag.report("support")
            row["g_lower"] = sup.bounds[0]
            row["trunc_frac"] = sup.truncated["fraction"]
            row["trunc_count"] = sup.truncated["count"]
            row["min_g"] = sup.propensity_quantiles["overall"][0.0]
            nm = diag.report("nuisance_models")
            tf = nm.to_frame()
            sel = tf[tf.iloc[:, 0].astype(str).str.contains("propensity")] if "model" not in tf.columns else tf[tf["model"] == "propensity"]
            if "cal_slope" in tf.columns and len(sel):
                row["cal_slope"] = float(sel["cal_slope"].iloc[0])
            elif "calibration_slope" in tf.columns and len(sel):
                row["cal_slope"] = float(sel["calibration_slope"].iloc[0])
        except Exception as exc:  # diagnostics are optional
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
    learners = ["a_hgb_default", "b_hgb_regularised", "c_hgb_early_stop", "d_logit_main", "e_logit_true"]
    seeds = range(5000, 5300)
    tasks = []
    # Interleave so that partial results are balanced across cells.
    for seed in seeds:
        for config in ("point", "cross"):
            for learner in learners:
                tasks.append((config, learner, 3000, seed))
    for seed in range(5000, 5100):
        for learner in ("b_hgb_regularised", "e_logit_true"):
            tasks.append(("point", learner, 12000, seed))
    done = set()
    if os.path.exists(OUT):
        with open(OUT, newline="") as fh:
            for r in csv.DictReader(fh):
                done.add((r["config"], r["learner"], int(r["n"]), int(r["seed"])))
    tasks = [t for t in tasks if t not in done]
    new = not os.path.exists(OUT)
    start = time.time()
    with open(OUT, "a", newline="") as fh, open(LOG, "a") as log:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
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
