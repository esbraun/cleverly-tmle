"""Seed sweep of the fits that docs/examples/subgroup-effects.ipynb shows.

Each seed ``s`` draws ``make_home_support(n=8000, seed=s)`` and runs every fit of the notebook
with the notebook's settings (logistic learners, ``outcome_by_arm=True``, ``random_state=0``):
the projection on the group indicators with its band and test, the projection on the indicators
plus a straight line in age with its nine-point band, the omitted-variable bound on the
"alone" difference, the refutation checks of that difference, the direct effect with the 7-day
visit held at 0 and at 1, the age slope inside each group (``strata=``), and the log risk ratio
by group.  It records, under a named column, each relation the notebook's readings state, so
``summarize.py`` can give the share of draws on which it holds.

The notebook shows seed 100, the first seed of the swept range, chosen by that rule before any
fit was read.

Usage: ``python sweep.py <first seed> <count> <workers>``.  The notebook cites
``python sweep.py 100 100 4``.  Output: ``sweep.csv`` (one row per seed), which
``summarize.py`` reads.  Every worker is single-threaded (``OMP_NUM_THREADS=1``).
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
ALONE = "cate_projection[home_support[alone]]"
GROUP_TERMS = {
    "family": "(intercept)",
    "alone": "home_support[alone]",
    "carer": "home_support[paid carer]",
}
AGES = (60, 75, 90)


def _covers(table, prefix, truth, target="cate_projection"):
    """Whether each coefficient's interval covers its true value, by short name."""
    rows = table.set_index("estimand")
    out = {}
    for short, term in GROUP_TERMS.items():
        value = truth[f"{prefix}:{target}[{term}]"]
        row = rows.loc[f"{target}[{term}]"]
        out[short] = bool(row["ci_lower"] <= value <= row["ci_upper"])
    return out


def one(seed):
    warnings.simplefilter("ignore")
    from cleverly import CausalStudy, ConditionalContrast, ControlledDirectEffect, PointTreatment
    from cleverly.bases import Additive, Indicators, Linear, Modifiers
    from cleverly.datasets import HOME_SUPPORT_LEVELS, home_support_effect, make_home_support

    levels = list(HOME_SUPPORT_LEVELS)
    learners = {
        "outcome_learner": LogisticRegression(max_iter=1000),
        "treatment_learner": LogisticRegression(max_iter=1000),
    }
    frame, truth = make_home_support(n=8000, seed=seed)
    design = PointTreatment(
        outcome="home_30",
        treatment="navigation",
        adjustment=("home_support", "age", "discharge_risk"),
    )
    study = CausalStudy(frame, design=design)
    by_group = Modifiers(["home_support"], levels={"home_support": HOME_SUPPORT_LEVELS})
    row = {"seed": seed}

    # Why a simple comparison misleads: the treated-minus-untreated difference in each group.
    means = frame.groupby(["home_support", "navigation"])["home_30"].mean().unstack()
    for short, level in zip(GROUP_TERMS, levels):
        naive = float(means.loc[level, 1] - means.loc[level, 0])
        row[f"naive_{short}"] = naive
        row[f"naive_gap_{short}"] = truth[f"group_effect[{level}]"] - naive

    # The projection on the group indicators, its band over the three groups, and the test.
    fit = study.identify(ConditionalContrast(by_group, model=Indicators("home_support"))).estimate(
        outcome_by_arm=True, random_state=0, **learners
    )
    table = fit.to_frame()
    for short, covered in _covers(table, "indicators", truth).items():
        row[f"indicators_covers_{short}"] = covered
    row["indicators_all_supported"] = bool((table["support"] == "supported").all())
    curve = fit.effect_curve(grid={"home_support": levels}, random_state=0)
    effects = np.array([truth[f"group_effect[{level}]"] for level in levels])
    point = curve.predict({"home_support": levels}, interval=True)
    row["group_points_cover"] = bool(
        np.all((point["ci_lower"] <= effects) & (effects <= point["ci_upper"]))
    )
    row["group_band_covers"] = bool(
        np.all((curve.band.lower <= effects) & (effects <= curve.band.upper))
    )
    row["group_band_label"] = curve.band.support.label
    row["heterogeneity_p"] = float(fit.heterogeneity_test().p_value)
    for short, level in zip(GROUP_TERMS, levels):
        row[f"error_{short}"] = (
            float(point["estimate"].iloc[levels.index(level)]) - truth[f"group_effect[{level}]"]
        )

    # The indicators plus a straight line in age, and its band over nine points.
    by_group_age = Modifiers(
        ["home_support", "age"],
        levels={"home_support": HOME_SUPPORT_LEVELS},
        domains={"age": (50, 90)},
    )
    shape = Additive(Indicators("home_support"), Linear(["age"]))
    with_age = study.identify(ConditionalContrast(by_group_age, model=shape)).estimate(
        outcome_by_arm=True, random_state=0, **learners
    )
    rows = with_age.to_frame().set_index("estimand")
    for short, covered in _covers(with_age.to_frame(), "additive", truth).items():
        row[f"additive_covers_{short}"] = covered
    slope = rows.loc["cate_projection[age]"]
    row["age_slope_covers"] = bool(slope["ci_lower"] <= truth["age_slope"] <= slope["ci_upper"])
    row["age_slope_excludes_zero"] = bool(slope["ci_lower"] > 0 or slope["ci_upper"] < 0)
    grid = {
        "home_support": [level for level in levels for _ in AGES],
        "age": [float(age) for _ in levels for age in AGES],
    }
    at_points = home_support_effect(grid["home_support"], grid["age"])
    age_curve = with_age.effect_curve(grid=grid, random_state=0)
    band = age_curve.band
    row["age_band_covers"] = bool(np.all((band.lower <= at_points) & (at_points <= band.upper)))
    pointwise = age_curve.predict(grid, interval=True)
    row["age_points_cover"] = int(
        np.sum((pointwise["ci_lower"] <= at_points) & (at_points <= pointwise["ci_upper"]))
    )
    row["age_heterogeneity_p"] = float(with_age.heterogeneity_test().p_value)

    # The omitted-variable bound on the "alone" difference.
    robustness = fit.sensitivity.robustness_value(estimand=ALONE)
    bounds = fit.sensitivity.omitted_confounding(estimand=ALONE, cf_y=0.03, cf_d=0.03)
    row["alone_rv"] = float(robustness["rv"])
    row["alone_rva"] = float(robustness["rva"])
    row["alone_bound_lower_above_zero"] = bool(bounds.lower > 0)

    # Try to break the result: placebo, random common cause and subset on the difference.
    refutation = fit.diagnostics.refute(estimand=ALONE, random_state=0).to_frame().set_index("test")
    original = float(refutation["original"].iloc[0])
    placebo = refutation.loc["placebo"]
    row["placebo_within_2sd"] = bool(abs(placebo["refuted_mean"]) <= 2 * placebo["refuted_sd"])
    for test in ("random_common_cause", "subset"):
        row[f"{test}_within_001"] = bool(
            abs(refutation.loc[test, "refuted_mean"] - original) <= 0.01
        )

    # The direct effect with the 7-day visit held at 0 and at 1.
    held_study = CausalStudy(frame, design=replace(design, intermediate="pcp_visit_7d"))
    for z in (0, 1):
        direct = held_study.identify(
            ControlledDirectEffect(
                intermediate=float(z),
                contrast=ConditionalContrast(by_group, model=Indicators("home_support")),
            )
        ).estimate(
            outcome_by_arm=True,
            random_state=0,
            intermediate_learner=LogisticRegression(max_iter=1000),
            **learners,
        )
        for short, covered in _covers(direct.to_frame(), f"direct[visit={z}]", truth).items():
            row[f"direct{z}_covers_{short}"] = covered
        held = np.array([truth[f"direct_effect[visit={z}][{level}]"] for level in levels])
        direct_curve = direct.effect_curve(grid={"home_support": levels}, random_state=0)
        direct_point = direct_curve.predict({"home_support": levels}, interval=True)
        row[f"direct{z}_points_cover"] = bool(
            np.all((direct_point["ci_lower"] <= held) & (held <= direct_point["ci_upper"]))
        )
        row[f"direct{z}_band_covers"] = bool(
            np.all((direct_curve.band.lower <= held) & (held <= direct_curve.band.upper))
        )

    # The age slope inside each group.
    by_age = Modifiers(["age"], domains={"age": (50, 90)})
    strata = (
        CausalStudy(frame, design=replace(design, strata=("home_support",)))
        .identify(ConditionalContrast(by_age, model=Linear(["age"])))
        .estimate(outcome_by_arm=True, random_state=0, **learners)
    )
    rows = strata.to_frame().set_index("estimand")
    for short, level in zip(GROUP_TERMS, levels):
        slope = rows.loc[f"cate_projection[age][home_support='{level}']"]
        row[f"strata_slope_covers_{short}"] = bool(
            slope["ci_lower"] <= truth["age_slope"] <= slope["ci_upper"]
        )

    # The log risk ratio by group.
    ratio = study.identify(
        ConditionalContrast(by_group, scale="log_ratio", model=Indicators("home_support"))
    ).estimate(outcome_by_arm=True, random_state=0, **learners)
    for short, covered in _covers(
        ratio.to_frame(), "log_ratio", truth, "log_crr_projection"
    ).items():
        row[f"ratio_covers_{short}"] = covered
    ratio_curve = ratio.effect_curve(grid={"home_support": levels}, random_state=0)
    logs = np.log([truth[f"risk_ratio[{level}]"] for level in levels])
    ratio_point = ratio_curve.predict({"home_support": levels}, interval=True)
    row["ratio_points_cover"] = bool(
        np.all((ratio_point["ci_lower"] <= logs) & (logs <= ratio_point["ci_upper"]))
    )
    return row


def main():
    first, count, workers = (int(value) for value in sys.argv[1:4])
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(one, range(first, first + count)))
    pd.DataFrame(rows).to_csv(HERE / "sweep.csv", index=False)


if __name__ == "__main__":
    main()
