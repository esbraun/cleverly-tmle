"""Review probe of docs/examples/treatment-rules.ipynb: the page's fits over 400 seeds.

The page reads one simulated sample. This probe redraws it 400 times at the page's size and
refits it with the page's settings, so the readings can say what holds across samples rather
than on one draw.

- The learned rule: the fold-evaluated estimate, its interval coverage of the true value of the
  fold rules, its standard error against the spread of the error, and the gap between that true
  value and the value of the best rule.
- The in-sample score: a rule learned on all rows and scored on the same rows, with the page's
  logistic model and with its random forest, against the true value of that rule.
- The lottery: the in-sample fit of the lottery against offering navigation to no one, with its
  interval coverage.

Usage: ``python probe.py > probe.log``. The page quotes ``probe.log``. A review probe is not a
registered study.
"""

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from cleverly import (
    CausalStudy,
    CrossFitting,
    LearnedRuleValue,
    ModelSpec,
    PointTreatment,
    RegimeMean,
    Runtime,
    TMLEMethod,
)
from cleverly.datasets import learned_rule_truth, make_treatment_rules, treatment_rule_value
from cleverly.interventions import Static, Stochastic
from cleverly.learners import thread_limit

DRAWS = 400
SEEDS = range(30_000, 30_000 + DRAWS)

OUTCOME_MODEL = make_pipeline(
    PolynomialFeatures(2, interaction_only=True, include_bias=False),
    LogisticRegression(C=1e6, max_iter=5000),
)
METHOD = TMLEMethod(
    models=ModelSpec(
        outcome_learner=OUTCOME_MODEL,
        treatment_learner=LogisticRegression(C=1e6, max_iter=1000),
    ),
    cross_fitting=CrossFitting(n_folds=10, fold_evaluation=True),
    runtime=Runtime(random_state=13, n_jobs=1),
)
IN_SAMPLE = replace(METHOD, cross_fitting=CrossFitting(enabled=False))
FOREST = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=17, n_jobs=1)
DESIGN = PointTreatment(outcome="home_30", treatment="offer", adjustment=("need", "lives_alone"))
COLUMNS = ["offer", "need", "lives_alone"]


def lottery_chances(data):
    offer = np.where(data["need"] > 0, 0.5, 0.0)
    return np.column_stack([1 - offer, offer])


PLANS = (
    Static(0, name="offer to none"),
    Stochastic(lottery_chances, name="lottery", density_kind="known"),
)
TRUE_LOTTERY = treatment_rule_value(lambda data: lottery_chances(data)[:, 1])


def in_sample_gap(frame, model):
    """The in-sample score of the rule learned on all rows, minus the true value of that rule."""
    with thread_limit():
        fitted = clone(model).fit(frame[COLUMNS], frame["home_30"])

        def chance(data, offer):
            return fitted.predict_proba(
                data[["need", "lives_alone"]].assign(offer=float(offer))[COLUMNS]
            )[:, 1]

        def rule(data):
            return (chance(data, 1) > chance(data, 0)).astype(float)

        offered = rule(frame)
        score = float(np.where(offered == 1, chance(frame, 1), chance(frame, 0)).mean())
        return score - treatment_rule_value(rule)


def one_draw(seed):
    frame, truth = make_treatment_rules(n=2_000, seed=seed)
    study = CausalStudy(frame, design=DESIGN)
    learned = study.identify(LearnedRuleValue()).estimate(method=METHOD)
    value = learned["ey_learned_rule[learned rule]"]
    target = learned_rule_truth(learned, OUTCOME_MODEL)
    policies = study.identify(RegimeMean(PLANS, reference="offer to none")).estimate(
        method=IN_SAMPLE
    )
    names = ["ey_regime[lottery]", "ey_regime[offer to none]"]
    gain = policies.contrast(lambda psi: psi[0] - psi[1], names, name="lottery minus none")
    true_gain = TRUE_LOTTERY - truth["ey_none"]
    row = {
        "rule_error": value.psi - target,
        "rule_se": value.std_error,
        "rule_covers": value.ci[0] <= target <= value.ci[1],
        "target_gap": truth["ey_best_rule"] - target,
        "logistic_gap": in_sample_gap(frame, OUTCOME_MODEL),
        "forest_gap": in_sample_gap(frame, FOREST),
    }
    row["lottery_error"] = gain.psi - true_gain
    row["lottery_se"] = gain.std_error
    row["lottery_covers"] = gain.ci[0] <= true_gain <= gain.ci[1]
    return row


def main():
    start = time.time()
    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(one_draw, SEEDS))
    column = {key: np.array([row[key] for row in rows], dtype=float) for key in rows[0]}
    print(f"draws {DRAWS}, seeds {SEEDS.start} to {SEEDS.stop - 1}, n = 2,000")
    error, se = column["rule_error"], column["rule_se"]
    print(
        f"learned rule: mean error {error.mean():.4f}, SD {error.std(ddof=1):.4f}, "
        f"mean SE {se.mean():.4f}, SE over SD {se.mean() / error.std(ddof=1):.4f}, "
        f"coverage {column['rule_covers'].mean():.4f}"
    )
    gap = column["target_gap"]
    print(
        f"best rule minus true value of the fold rules: mean {gap.mean():.4f}, "
        f"largest {gap.max():.4f}"
    )
    for label in ("logistic", "forest"):
        gaps = column[f"{label}_gap"]
        print(
            f"in-sample score minus true value, {label}: mean {gaps.mean():.4f}, "
            f"SD {gaps.std(ddof=1):.4f}, share above zero {np.mean(gaps > 0):.4f}"
        )
    error, se = column["lottery_error"], column["lottery_se"]
    print(
        f"lottery minus none: mean error {error.mean():.4f}, SD {error.std(ddof=1):.4f}, "
        f"mean SE {se.mean():.4f}, SE over SD {se.mean() / error.std(ddof=1):.4f}, "
        f"coverage {column['lottery_covers'].mean():.4f}"
    )
    print(f"elapsed {time.time() - start:.0f} seconds")


if __name__ == "__main__":
    main()
