"""Seed sweep of the fits that docs/examples/model-based-over-time.ipynb shows.

Each seed ``s`` draws ``make_recovery_score(4_000, seed=s)`` and runs the page's fits: the
parametric g-formula of the plans ``never`` and ``always`` with symptom models that hold every
product of two record columns (the right models), the same fit with
main-terms symptom models (the wrong models), the cross-fitted structural nested mean model with
``lives_alone`` in the blips of the first two offers, its model check against LTMLE, and the crude
comparison of the patients who followed each plan. Truths come from ``make_recovery_score``.

Usage: ``python sweep.py <first seed> <count> <workers>``. The page cites
``python sweep.py 3100 400 4``. The notebook draws seed 3100, the first of that range, by a rule
fixed before the page ran with it, so its fits are the first row of ``sweep.csv``. Outputs:
``sweep.csv`` (one row per seed) and ``sweep.log`` (the summary).
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
N = 4_000
OFFERS = ("offer_discharge", "offer_day10", "offer_day20")
PLANS = {"never": (0, 0, 0), "always": (1, 1, 1)}
LEVELS = ("ey_regimen[never]", "ey_regimen[always]")
CONTRAST = "ate_regimen[always vs never]"
BLIPS = (
    "blip[offer_discharge:1.0:intercept]",
    "blip[offer_discharge:1.0:lives_alone]",
    "blip[offer_day10:1.0:intercept]",
    "blip[offer_day10:1.0:lives_alone]",
    "blip[offer_day20:1.0:intercept]",
)


def one_seed(seed: int) -> dict:
    warnings.filterwarnings("ignore")
    from cleverly import (
        CausalStudy,
        LongitudinalTreatment,
        ModelSpec,
        ParametricGFormulaMethod,
        RegimeMean,
        Runtime,
    )
    from cleverly.datasets import RECOVERY_BOUNDS, make_recovery_score
    from cleverly.learners import GLM
    from cleverly.longitudinal import snmm

    frame, truth = make_recovery_score(N, seed=seed)
    row: dict = {"seed": seed}

    followed = {}
    for name, plan in PLANS.items():
        mask = (frame[list(OFFERS)] == list(plan)).all(axis=1)
        followed[name] = frame.loc[mask, "recovery_day30"].mean()
    row["crude"] = followed["always"] - followed["never"]

    design = LongitudinalTreatment(
        outcome="recovery_day30",
        treatment=OFFERS,
        baseline=("lives_alone", "discharge_risk"),
        time_varying=((), ("symptoms_day10",), ("symptoms_day20",)),
    )
    question = CausalStudy(frame, design=design).identify(RegimeMean(PLANS, reference="never"))
    models = ModelSpec(outcome_learner=GLM(family="gaussian"), treatment_learner=GLM())
    for tag, learner in (("gf", GLM(interactions=2)), ("wrong", None)):
        fit = question.estimate(
            method=ParametricGFormulaMethod(
                models=models,
                covariate_learner=learner,
                n_draws=10,
                runtime=Runtime(random_state=seed, n_jobs=1),
            )
        )
        for key in LEVELS:
            estimate = fit[key]
            row[f"{tag}:{key}"] = estimate.psi
            row[f"{tag}:{key}:covers"] = estimate.ci[0] <= truth[key] <= estimate.ci[1]
        contrast = fit.contrast(lambda v: v[1] - v[0], list(LEVELS), name=CONTRAST)
        row[f"{tag}:{CONTRAST}"] = contrast.psi
        row[f"{tag}:{CONTRAST}:covers"] = contrast.ci[0] <= truth[CONTRAST] <= contrast.ci[1]
        if tag == "gf":
            shares = fit.simulation.monte_carlo_share
            for key in (*LEVELS, CONTRAST):
                row[f"gf:{key}:mc_share"] = shares.get(key, float("nan"))

    structural = snmm(
        frame,
        outcome="recovery_day30",
        treatment=list(OFFERS),
        baseline=["lives_alone", "discharge_risk"],
        time_varying=[[], ["symptoms_day10"], ["symptoms_day20"]],
        regimens=PLANS,
        reference="never",
        blip_features={1: ["lives_alone"], 2: ["lives_alone"]},
        outcome_learner=GLM(family="gaussian"),
        treatment_learner=GLM(family="binomial"),
        n_folds=5,
        random_state=seed,
    )
    for key in (*BLIPS, *LEVELS, CONTRAST):
        estimate = structural[key]
        row[f"snmm:{key}"] = estimate.psi
        row[f"snmm:{key}:covers"] = estimate.ci[0] <= truth[key] <= estimate.ci[1]
    # The effect of each offer by living situation: the intercept, plus the lives_alone term.
    for column in OFFERS:
        names = [f"blip[{column}:1.0:intercept]"]
        if column != "offer_day20":
            names.append(f"blip[{column}:1.0:lives_alone]")
        for alone in (0, 1):
            effect = structural.contrast(lambda v, a=alone: v[0] + a * sum(v[1:]), names)
            true = truth[names[0]] + alone * sum(truth[name] for name in names[1:])
            key = f"effect:{column}:alone={alone}"
            row[f"snmm:{key}"] = effect.psi
            row[f"snmm:{key}:covers"] = effect.ci[0] <= true <= effect.ci[1]
            row[f"truth:{key}"] = true
    check = structural.check_model(q_bounds=RECOVERY_BOUNDS)
    row["check:pvalue"] = float(check.pvalue)
    return row


def main() -> None:
    import pandas as pd

    first, count, workers = (int(value) for value in sys.argv[1:4])
    start = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one_seed, range(first, first + count)))
    frame = pd.DataFrame(rows).sort_values("seed")
    frame.to_csv(HERE / "sweep.csv", index=False)

    from cleverly.datasets import make_recovery_score

    _, truth = make_recovery_score(10, seed=0)
    lines = [f"seeds {first} to {first + count - 1}, n = {N}, {time.time() - start:.0f} s"]
    lines.append(f"crude: mean {frame['crude'].mean():.3f}, truth {truth[CONTRAST]:.3f}")
    for column in frame.columns:
        if column.endswith(":covers"):
            key = column.removesuffix(":covers")
            name = key.split(":", 1)[1]
            target = truth[name] if name in truth else frame[f"truth:{name}"].iloc[0]
            bias = frame[key].mean() - target
            sd = frame[key].std(ddof=1)
            lines.append(
                f"{key}: covers {int(frame[column].sum())} of {count}, "
                f"mean error {bias:.4f} (Monte Carlo SE {sd / count**0.5:.4f})"
            )
    for column in frame.columns:
        if column.endswith(":mc_share"):
            lines.append(
                f"{column}: median {frame[column].median():.4f}, max {frame[column].max():.4f}"
            )
    rejects = int((frame["check:pvalue"] < 0.05).sum())
    lines.append(f"model check: p below 0.05 on {rejects} of {count}")
    (HERE / "sweep.log").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
