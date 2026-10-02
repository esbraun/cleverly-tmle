"""Seed sweep of the fits that docs/examples/longitudinal-tmle.ipynb shows.

Each seed ``s`` draws ``make_longitudinal(n=8000, seed=s, cluster_size=20)``, renames the
columns as the notebook does, and runs the notebook's code with 41 replaced by ``s``: the
day-seven shares within discharge strata (Step 3), the clustered in-sample fit (Step 6), the
same fit without ``cluster=``, the two point-treatment shortcuts (Step 7), the rule fit and its
contrast with ``always`` (Step 8), the assessment with the truncation curve (Step 9), and the
three covariate-drop refits (Step 10). Truths come from ``truth.py``.

Usage: ``python sweep.py <first seed> <count> <workers>``. The page cites
``python sweep.py 1000 400 4``. Outputs: ``sweep.csv`` (one row per seed) and ``sweep.log``.
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
TARGET = 0.361570  # truth.py
RULE_TRUTH = 0.321181  # truth.py
RULE_VS_ALWAYS = -0.040389  # truth.py
RENAME = {
    "W1": "age",
    "W2": "baseline_readiness",
    "A1": "navigation_discharge",
    "C1": "tracked_day7",
    "L2": "engagement_day7",
    "A2": "navigation_day7",
    "C2": "tracked_day30",
    "Y": "transition_top_box",
    "id": "navigator_team",
}
DROPPED = ("age", "baseline_readiness", "engagement_day7")


def one_seed(seed: int) -> dict:
    warnings.filterwarnings("ignore")
    from dataclasses import replace

    from sklearn.linear_model import LinearRegression, LogisticRegression

    from cleverly import (
        ATE,
        CausalStudy,
        CrossFitting,
        LongitudinalTreatment,
        ModelSpec,
        PointTreatment,
        RegimeContrast,
        Runtime,
        TMLEMethod,
    )
    from cleverly.datasets import make_longitudinal
    from cleverly.longitudinal import DynamicRegimen

    frame, _ = make_longitudinal(n=8_000, seed=seed, cluster_size=20)
    frame = frame.rename(columns=RENAME)
    row: dict = {"seed": seed}

    tracked = frame[frame["tracked_day7"] == 1]
    by = tracked.groupby("navigation_discharge")["engagement_day7"].mean()
    row["eng_gap"] = by[1.0] - by[0.0]
    engaged = tracked["engagement_day7"] > 0
    shares = tracked.groupby(["navigation_discharge", engaged])["navigation_day7"].mean().unstack()
    for stratum in (0, 1):
        row[f"share_a1{stratum}_engaged"] = shares.loc[float(stratum), True]
        row[f"share_a1{stratum}_not"] = shares.loc[float(stratum), False]
    pooled = tracked.groupby(engaged)["navigation_day7"].mean()
    row["pooled_gap"] = pooled[True] - pooled[False]

    logistic = LogisticRegression(max_iter=1000, random_state=seed)
    sequential = TMLEMethod(
        models=ModelSpec(
            outcome_learner=logistic,
            pseudo_learner=LinearRegression(),
            treatment_learner=logistic,
            censoring_learner=logistic,
        ),
        cross_fitting=CrossFitting(enabled=False),
        runtime=Runtime(random_state=seed, n_jobs=1),
    )
    design = LongitudinalTreatment(
        outcome="transition_top_box",
        treatment=("navigation_discharge", "navigation_day7"),
        baseline=("age", "baseline_readiness"),
        time_varying=((), ("engagement_day7",)),
        censoring=("tracked_day7", "tracked_day30"),
        cluster="navigator_team",
    )
    study = CausalStudy(frame, design=design)
    plan = RegimeContrast({"always": 1, "never": 0}, reference="never")
    result = study.identify(plan).estimate(method=sequential)
    est = result["ate_regimen[always vs never]"]
    row.update(psi=est.psi, se=est.std_error, lo=est.ci[0], hi=est.ci[1])

    unclustered = replace(design, cluster=None)
    iid = (
        CausalStudy(frame.drop(columns=["navigator_team"]), design=unclustered)
        .identify(plan)
        .estimate(method=sequential)["ate_regimen[always vs never]"]
    )
    row.update(psi_iid=iid.psi, se_iid=iid.std_error)

    rule_result = study.identify(
        RegimeContrast(
            {
                "always": 1,
                "never": 0,
                "continue if engaged": DynamicRegimen(
                    "continue if engaged",
                    (1, lambda h: (h["engagement_day7"] > 0).astype(float)),
                    rule_kind="known",
                ),
            },
            reference="never",
        )
    ).estimate(method=sequential)
    rp = rule_result["ate_regimen[continue if engaged vs never]"]
    row.update(rule_psi=rp.psi, rule_lo=rp.ci[0], rule_hi=rp.ci[1], rule_se=rp.std_error)
    rva = rule_result.contrast(
        lambda v: v[0] - v[1],
        ["ate_regimen[continue if engaged vs never]", "ate_regimen[always vs never]"],
        name="rva",
    )
    row.update(rva_psi=rva.psi, rva_lo=rva.ci[0], rva_hi=rva.ci[1])
    rs = rule_result.diagnostics.support().to_frame().set_index(["regimen", "time"])
    row["rule_share_t2"] = rs.loc[("continue if engaged", 2), "share_assigned_1"]

    observed = frame[(frame["tracked_day7"] == 1) & (frame["tracked_day30"] == 1)]
    consistent = observed[observed["navigation_discharge"] == observed["navigation_day7"]].copy()
    consistent = consistent.rename(columns={"navigation_discharge": "navigation_throughout"})
    point_method = replace(
        sequential, models=ModelSpec(outcome_learner=logistic, treatment_learner=logistic)
    )

    def naive(adjustment):
        naive_study = CausalStudy(
            consistent,
            design=PointTreatment(
                outcome="transition_top_box",
                treatment="navigation_throughout",
                adjustment=adjustment,
                cluster="navigator_team",
            ),
        )
        return naive_study.identify(ATE(reference=0)).estimate(method=point_method)["ate"]

    adjusted = naive(("age", "baseline_readiness", "engagement_day7"))
    baseline_only = naive(("age", "baseline_readiness"))
    row.update(
        adj_psi=adjusted.psi,
        adj_hi=adjusted.ci[1],
        base_psi=baseline_only.psi,
        base_lo=baseline_only.ci[0],
    )

    assessment = result.assess(
        include_refits=True, arguments={"truncation_curve": {"bounds": [0.01, 0.05, 0.1]}}
    )
    support = assessment.report("support").to_frame()
    row["max_weight"] = support["max_weight"].max()
    row["share_trunc_max"] = support["share_truncated"].max()
    row["min_eff_ratio"] = (support["effective_n"] / support["n_followed"]).min()
    curve = assessment.report("truncation_curve")
    row["curve_move_se"] = float(curve["delta_from_fitted"].abs().max()) / est.std_error
    nuisance = assessment.report("nuisance_models").to_frame()
    binary = nuisance["calibration_slope"].notna()
    row["cal_min"] = nuisance.loc[binary, "calibration_slope"].min()
    row["cal_max"] = nuisance.loc[binary, "calibration_slope"].max()
    row["reg_min"] = nuisance.loc[~binary, "regression_slope"].min()
    row["reg_max"] = nuisance.loc[~binary, "regression_slope"].max()
    censoring = nuisance[nuisance["role"] == "censoring"].set_index("time")["auc"]
    row["auc_censoring_1"] = censoring[1]
    row["auc_censoring_2"] = censoring[2]
    row["auc_min_other"] = nuisance.loc[nuisance["role"] != "censoring", "auc"].min()
    scores = assessment.report("score_equations").to_frame()
    row["scores_pass"] = bool(scores["passed"].all())

    for dropped in DROPPED:
        reduced = replace(
            design,
            baseline=tuple(c for c in design.baseline if c != dropped),
            time_varying=tuple(
                tuple(c for c in node if c != dropped) for node in design.time_varying
            ),
        )
        refit = (
            CausalStudy(frame, design=reduced)
            .identify(plan)
            .estimate(method=sequential)["ate_regimen[always vs never]"]
        )
        row[f"move_{dropped}"] = (refit.psi - est.psi) / est.std_error
    return row


if __name__ == "__main__":
    import pandas as pd

    import cleverly

    first, count, workers = (int(value) for value in sys.argv[1:4])
    src = Path(cleverly.__file__).resolve()
    assert (HERE.parents[3] / "src") in src.parents, src
    start = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one_seed, range(first, first + count)))
    frame = pd.DataFrame(rows).sort_values("seed")
    frame.to_csv(HERE / "sweep.csv", index=False, lineterminator="\n")
    with open(HERE / "sweep.log", "w", encoding="utf-8", newline="\n") as log:
        log.write(f"cleverly {cleverly.__version__} from {src}\n")
        log.write(f"seeds {first} to {first + count - 1}, {workers} workers\n")
        log.write(f"wall time {time.time() - start:.0f} s\n")
