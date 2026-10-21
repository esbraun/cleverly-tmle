"""Review probe for docs/examples/collaborative-tmle.ipynb: the page's three fits over many draws.

Not a registered study.  Each draw samples ``binary_active`` of the registered study
``ctmle-oat-per-arm-parametric`` at n = 1,500 (``tests/studies/oat_per_arm_laws.py``), and fits
the three estimators of the page with the page's settings:

* ``tmle``: TMLE with ParametricOutcome() and an unpenalized main-terms logistic g on W1, W2, W3;
* ``oat``: per-arm outcome-adaptive C-TMLE with ParametricOutcome() and ParametricMechanism(3);
* ``gcomp``: the g-computation plug-in of ParametricOutcome(), the untargeted outcome fit;
* ``tmle_main`` and ``oat_main``: the first two fits with ParametricOutcome("main_terms"), which
  cannot represent the arm-specific outcome index of the law.

Run from the repository root:
``PYTHONPATH=src:. python reviews/notebook-review/probes/collaborative-tmle-oat/sweep.py 2000``.
It writes ``draws.csv`` beside itself and prints ``summary.log`` to stdout.
The committed log was written with ``-W ignore``.
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from cleverly import (
    ATE,
    CausalStudy,
    CollaborativeTMLEMethod,
    CrossFitting,
    Inference,
    ModelSpec,
    PointTreatment,
    Runtime,
    TMLEMethod,
)
from cleverly.learners import ParametricMechanism, ParametricOutcome
from tests.studies.oat_per_arm_laws import BINARY_ACTIVE

HERE = Path(__file__).resolve().parent
N = 1_500
FIRST_SEED = 100_000


def one(seed: int) -> dict[str, float]:
    sample, _ = BINARY_ACTIVE.sample(N, seed)
    effect = CausalStudy(
        sample, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3"))
    ).identify(ATE(reference=0))
    common = {
        "cross_fitting": CrossFitting(enabled=False),
        "runtime": Runtime(random_state=seed, n_jobs=1),
        "inference": Inference(simultaneous=False),
    }
    tmle = effect.estimate(
        method=TMLEMethod(
            models=ModelSpec(
                outcome_learner=ParametricOutcome(),
                treatment_learner=LogisticRegression(C=np.inf, max_iter=1000),
            ),
            **common,
        )
    )
    oat = effect.estimate(
        method=CollaborativeTMLEMethod(
            models=ModelSpec(
                outcome_learner=ParametricOutcome(),
                treatment_learner=ParametricMechanism(degree=3),
            ),
            strategy="oat",
            **common,
        )
    )
    main = ParametricOutcome(features="main_terms")
    tmle_main = effect.estimate(
        method=TMLEMethod(
            models=ModelSpec(
                outcome_learner=main,
                treatment_learner=LogisticRegression(C=np.inf, max_iter=1000),
            ),
            **common,
        )
    )
    oat_main = effect.estimate(
        method=CollaborativeTMLEMethod(
            models=ModelSpec(outcome_learner=main, treatment_learner=ParametricMechanism(degree=3)),
            strategy="oat",
            **common,
        )
    )
    covariates = sample[["W1", "W2", "W3"]].to_numpy()
    outcome = ParametricOutcome().fit(
        np.column_stack([sample["A"].to_numpy(), covariates]), sample["Y"].to_numpy()
    )
    arm_means = [
        outcome.predict(np.column_stack([np.full(N, arm), covariates])).mean() for arm in (0, 1)
    ]
    row: dict[str, float] = {"seed": seed, "gcomp_psi": arm_means[1] - arm_means[0]}
    row["gcomp_status"] = "point only"
    fits = (("tmle", tmle), ("oat", oat), ("tmle_main", tmle_main), ("oat_main", oat_main))
    for name, result in fits:
        point = result["ate"]
        row[f"{name}_psi"] = point.psi
        row[f"{name}_status"] = point.inference
        if point.inference == "influence_curve":
            low, high = point.ci
            row[f"{name}_se"] = point.std_error
            row[f"{name}_low"], row[f"{name}_high"] = low, high
    for name, result in (("tmle", tmle), ("oat", oat)):
        support = result.diagnostics.support()
        row[f"{name}_tail"] = support.tail_mass[0.1]["below"] + support.tail_mass[0.1]["above"]
        row[f"{name}_clever_max"] = support.clever_covariate_max["mean"]
    return row


def main(draws: int) -> None:
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(one, range(FIRST_SEED, FIRST_SEED + draws), chunksize=10))
    frame = pd.DataFrame(rows)
    frame.to_csv(HERE / "draws.csv", index=False)
    truth = BINARY_ACTIVE.truth()["ate"]
    print(f"law binary_active, n = {N}, draws = {draws}, true ate = {truth:.6f}")
    for name in ("tmle", "oat", "gcomp", "tmle_main", "oat_main"):
        psi = frame[f"{name}_psi"]
        line = f"{name:9s} bias {psi.mean() - truth:+.5f}  empirical sd {psi.std(ddof=1):.5f}"
        statuses = frame[f"{name}_status"].value_counts().to_dict()
        line += f"  status {statuses}"
        if f"{name}_se" in frame:
            se = frame[f"{name}_se"]
            cover = ((frame[f"{name}_low"] <= truth) & (truth <= frame[f"{name}_high"])).sum()
            line += f"  mean se / sd {se.mean() / psi.std(ddof=1):.4f}  covered {cover} of {draws}"
        print(line)
    ratio = frame["oat_psi"].std(ddof=1) / frame["tmle_psi"].std(ddof=1)
    print(f"oat sd / tmle sd {ratio:.4f}")
    narrower = (frame["oat_se"] < frame["tmle_se"]).sum()
    print(f"oat se below tmle se on {narrower} of {draws} draws")
    for name in ("tmle", "oat"):
        print(
            f"{name:6s} mean share of g outside (0.1, 0.9) {frame[f'{name}_tail'].mean():.4f}; "
            f"median max clever covariate {frame[f'{name}_clever_max'].median():.2f}"
        )
    gap = (frame["oat_psi"] - frame["tmle_psi"]).abs()
    print(f"|oat - tmle| median {gap.median():.4f}, 95th percentile {gap.quantile(0.95):.4f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
