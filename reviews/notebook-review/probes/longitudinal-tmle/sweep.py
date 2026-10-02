"""Seed sweep of every comparative/property claim in longitudinal-tmle.ipynb.

Runs the notebook's own code path at n=8000, cluster_size=20, for each seed.
Usage: python sweep.py START STOP OUT.csv
"""

import sys
import time
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

import cleverly
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

warnings.filterwarnings("ignore")
assert "bridge-cse" in cleverly.__file__, cleverly.__file__

# Independently recomputed truths (truth_and_mechanism.log, 1e7 RB Monte Carlo)
TARGET = 0.36154
RULE_TRUTH = 0.32114
RULE_VS_ALWAYS = -0.04041

RENAME = {
    "W1": "age", "W2": "baseline_readiness", "A1": "navigation_discharge",
    "C1": "tracked_day7", "L2": "engagement_day7", "A2": "navigation_day7",
    "C2": "tracked_day30", "Y": "transition_top_box", "id": "navigator_team",
}


def one(seed, with_assess=True):
    frame, truth = make_longitudinal(n=8_000, seed=seed, cluster_size=20)
    frame = frame.rename(columns=RENAME)
    row = {"seed": seed}
    tracked = frame[frame["tracked_day7"] == 1]
    by = tracked.groupby("navigation_discharge")["engagement_day7"].mean()
    row["eng_gap"] = by[1.0] - by[0.0]
    sh = tracked.groupby(tracked["engagement_day7"] > 0)["navigation_day7"].mean()
    row["share_engaged"], row["share_not"] = sh[True], sh[False]
    complete = frame.dropna(subset=["transition_top_box"])
    arms = complete.groupby(["navigation_discharge", "navigation_day7"])["transition_top_box"].mean()
    row["crude"] = arms.loc[(1.0, 1.0)] - arms.loc[(0.0, 0.0)]

    logistic = LogisticRegression(max_iter=1000, random_state=41)
    sequential = TMLEMethod(
        models=ModelSpec(outcome_learner=logistic, pseudo_learner=LinearRegression(),
                         treatment_learner=logistic, censoring_learner=logistic),
        cross_fitting=CrossFitting(enabled=False),
        runtime=Runtime(random_state=41, n_jobs=1),
    )

    def design(cluster):
        return LongitudinalTreatment(
            outcome="transition_top_box",
            treatment=("navigation_discharge", "navigation_day7"),
            baseline=("age", "baseline_readiness"),
            time_varying=((), ("engagement_day7",)),
            censoring=("tracked_day7", "tracked_day30"),
            **({"cluster": "navigator_team"} if cluster else {}),
        )

    study = CausalStudy(frame, design=design(True))
    plan = RegimeContrast({"always": 1, "never": 0}, reference="never")
    result = study.identify(plan).estimate(method=sequential)
    est = result["ate_regimen[always vs never]"]
    row.update(psi=est.psi, se=est.std_error, lo=est.ci[0], hi=est.ci[1])

    # unclustered SE on the same data
    study_iid = CausalStudy(frame.drop(columns=["navigator_team"]), design=design(False))
    est_iid = study_iid.identify(plan).estimate(method=sequential)["ate_regimen[always vs never]"]
    row["se_iid"] = est_iid.std_error
    row["psi_iid"] = est_iid.psi

    rule_result = study.identify(
        RegimeContrast(
            {"always": 1, "never": 0,
             "continue if engaged": DynamicRegimen(
                 "continue if engaged",
                 (1, lambda h: (h["engagement_day7"] > 0).astype(float)),
                 rule_kind="known")},
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

    # failure mode
    observed = frame[(frame["tracked_day7"] == 1) & (frame["tracked_day30"] == 1)]
    consistent = observed[observed["navigation_discharge"] == observed["navigation_day7"]].copy()
    consistent = consistent.rename(columns={"navigation_discharge": "navigation_throughout"})
    point_method = replace(sequential, models=ModelSpec(outcome_learner=logistic, treatment_learner=logistic))

    def naive(adj):
        s = CausalStudy(consistent, design=PointTreatment(
            outcome="transition_top_box", treatment="navigation_throughout",
            adjustment=adj, cluster="navigator_team"))
        return s.identify(ATE(reference=0)).estimate(method=point_method)["ate"]

    a = naive(("age", "baseline_readiness", "engagement_day7"))
    b = naive(("age", "baseline_readiness"))
    row.update(adj_psi=a.psi, adj_hi=a.ci[1], base_psi=b.psi, base_lo=b.ci[0])

    if with_assess:
        assessment = result.assess(include_refits=True,
                                   arguments={"truncation_curve": {"bounds": [0.01, 0.05, 0.1]}})
        sup = assessment.report("support").to_frame()
        row["max_weight"] = sup["max_weight"].max()
        row["share_trunc_max"] = sup["share_truncated"].max()
        row["min_eff_ratio"] = (sup["effective_n"] / sup["n_followed"]).min()
        curve = assessment.report("truncation_curve")
        row["curve_move_se"] = float(curve["delta_from_fitted"].abs().max()) / est.std_error
        row["trunc_cells_01"] = int(curve["truncated_score_cells"].iloc[-1])
        nu = assessment.report("nuisance_models").to_frame()
        binary = nu["calibration_slope"].notna()
        row["cal_min"] = nu.loc[binary, "calibration_slope"].min()
        row["cal_max"] = nu.loc[binary, "calibration_slope"].max()
        row["reg_min"] = nu.loc[~binary, "regression_slope"].min()
        row["reg_max"] = nu.loc[~binary, "regression_slope"].max()
        low = nu.loc[nu["auc"].idxmin()]
        row["lowest_auc_role"] = f"{low['role']}@{low['time']}"
        row["lowest_auc"] = low["auc"]
        sc = assessment.report("score_equations").to_frame()
        row["scores_pass"] = bool(sc["passed"].all())
    return row


if __name__ == "__main__":
    start, stop, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    rows = []
    for seed in range(start, stop):
        t = time.time()
        rows.append(one(seed))
        print(seed, f"{time.time() - t:.1f}s", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)
