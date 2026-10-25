"""Seed sweep of the schedules-and-intensity page's two fits, 400 draws each at the page's size.

Run from the repository root:
    python reviews/notebook-review/probes/schedules-and-intensity/sweep.py \
        > reviews/notebook-review/probes/schedules-and-intensity/sweep.log

Each draw refits the page's configurations with the page's learners: TMLE of the two odds
changes of the contact schedules (linear outcome model, multinomial logistic treatment model,
in sample, outcome range declared as 0 to 1), and TMLE of the three dose plans (logistic outcome
model on scaled degree-2 terms, a logistic classifier for the density ratio, in sample). It also
records the difference "both raises minus the two separate raises added up". The CSV holds every
estimate.
"""

from __future__ import annotations

import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from cleverly import (
    CausalStudy,
    CrossFitting,
    IncrementalEffect,
    ModelSpec,
    ModifiedTreatmentPolicyEffect,
    PointTreatment,
    Runtime,
    Targeting,
    TMLEMethod,
)
from cleverly.datasets import make_contact_schedules, make_navigation_doses
from cleverly.interventions import Incremental, JointPolicy, Shift

N = 3_000
DRAWS = 400
FIRST_SEED = 5_000
HERE = Path(__file__).parent
TILTS = {
    "current practice": {"low": 1.0},
    "medium and high doubled": {"medium": 2.0, "high": 2.0},
    "high doubled": {"high": 2.0},
}
HOURS = "ate_policy[more navigator time vs current practice]"
MINUTES = "ate_policy[more pharmacist time vs current practice]"
BOTH = "ate_policy[both vs current practice]"


def tilt_fit(seed: int) -> dict[str, float]:
    frame, truth = make_contact_schedules(n=N, seed=seed, tilts=TILTS)
    design = PointTreatment(
        outcome="transition_score",
        treatment="cadence",
        adjustment=("discharge_risk", "age", "baseline_support_need"),
    )
    effect = CausalStudy(frame, design=design).identify(
        IncrementalEffect(tuple(Incremental(f, name=n) for n, f in TILTS.items()))
    )
    result = effect.estimate(
        method=TMLEMethod(
            models=ModelSpec(
                outcome_learner=LinearRegression(),
                treatment_learner=LogisticRegression(max_iter=1000),
            ),
            cross_fitting=CrossFitting(enabled=False),
            targeting=Targeting(q_bounds=(0.0, 1.0)),
            runtime=Runtime(random_state=seed, n_jobs=1),
        )
    )
    out: dict[str, float] = {}
    for name in list(TILTS)[1:]:
        key = f"ate_ipsi[{name} vs current practice]"
        point = result[key]
        out[f"{name}|estimate"] = point.psi
        out[f"{name}|se"] = point.std_error
        out[f"{name}|truth"] = truth[key]
        out[f"{name}|covered"] = float(point.ci[0] <= truth[key] <= point.ci[1])
        out[f"{name}|supported"] = float(point.support.label == "supported")
    return out


def dose_fit(seed: int) -> dict[str, float]:
    frame, truth = make_navigation_doses(n=N, seed=seed)
    doses = ["navigator_hours", "pharmacist_minutes"]
    design = PointTreatment(
        outcome="transition_success",
        treatment=doses,
        continuous_treatment=doses,
        adjustment=("discharge_risk", "age", "medication_burden"),
    )
    hours, minutes = Shift(0.5, cap=7.0), Shift(5.0, cap=70.0)
    plans = [
        JointPolicy("current practice", components={}, policy_kind="known"),
        JointPolicy("more navigator time", components={doses[0]: hours}, policy_kind="known"),
        JointPolicy("more pharmacist time", components={doses[1]: minutes}, policy_kind="known"),
        JointPolicy("both", components={doses[0]: hours, doses[1]: minutes}, policy_kind="known"),
    ]
    effect = CausalStudy(frame, design=design).identify(
        ModifiedTreatmentPolicyEffect(plans, reference="current practice")
    )
    result = effect.estimate(
        method=TMLEMethod(
            models=ModelSpec(
                outcome_learner=make_pipeline(
                    StandardScaler(), PolynomialFeatures(2), LogisticRegression(max_iter=1000)
                ),
                treatment_learner=LogisticRegression(max_iter=1000),
                ratio="classifier",
            ),
            cross_fitting=CrossFitting(enabled=False),
            runtime=Runtime(random_state=seed, n_jobs=1),
        )
    )
    extra = result.contrast(lambda p: p[0] - p[1] - p[2], [BOTH, HOURS, MINUTES])
    out: dict[str, float] = {}
    rows = {
        "more navigator time": (result[HOURS], truth[HOURS]),
        "more pharmacist time": (result[MINUTES], truth[MINUTES]),
        "both": (result[BOTH], truth[BOTH]),
        "together minus added up": (extra, truth[BOTH] - truth[HOURS] - truth[MINUTES]),
    }
    for name, (point, value) in rows.items():
        out[f"{name}|estimate"] = point.psi
        out[f"{name}|se"] = point.std_error
        out[f"{name}|truth"] = value
        out[f"{name}|covered"] = float(point.ci[0] <= value <= point.ci[1])
        out[f"{name}|supported"] = float(point.support.label == "supported")
    out["added up below both's interval"] = float(
        result[HOURS].psi + result[MINUTES].psi < result[BOTH].ci[0]
    )
    return out


def one(seed: int) -> dict[str, float]:
    warnings.simplefilter("ignore")
    return {"seed": float(seed), **tilt_fit(seed), **dose_fit(seed)}


def main() -> None:
    seeds = range(FIRST_SEED, FIRST_SEED + DRAWS)
    with Pool(4) as pool:
        rows = pool.map(one, seeds)
    table = pd.DataFrame(rows)
    table.to_csv(HERE / "sweep.csv", index=False)
    print(f"n = {N}; draws = {DRAWS}; seeds {FIRST_SEED} to {FIRST_SEED + DRAWS - 1}")
    names = [*list(TILTS)[1:], "more navigator time", "more pharmacist time", "both"]
    for name in [*names, "together minus added up"]:
        error = table[f"{name}|estimate"] - table[f"{name}|truth"]
        ratio = table[f"{name}|se"].mean() / table[f"{name}|estimate"].std(ddof=1)
        covered = int(table[f"{name}|covered"].sum())
        supported = int(table[f"{name}|supported"].sum())
        print(
            f"{name}: mean error {error.mean():.5f}; SE/SD {ratio:.2f}; "
            f"covered {covered} of {DRAWS}; supported {supported} of {DRAWS}"
        )
    below = int(table["added up below both's interval"].sum())
    print(f"added-up estimate below the lower limit of both's interval: {below} of {DRAWS}")
    mean_extra = float(np.mean(table["together minus added up|estimate"]))
    print(f"mean estimate, together minus added up: {mean_extra:.4f}")


if __name__ == "__main__":
    main()
