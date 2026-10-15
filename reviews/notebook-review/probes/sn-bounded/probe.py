"""Seed sweep of the Step 6 and Step 7 fits of docs/examples/survey-nonresponse.ipynb.

Run from the repository root: ``python reviews/notebook-review/probes/sn-bounded/probe.py``.
It writes ``sweep.csv`` and ``summary.log`` beside this file.  The fits copy the notebook's
learners, folds, and bounds; only the seed changes.
"""

from __future__ import annotations

import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from cleverly import (
    ATE,
    CausalStudy,
    CrossFitting,
    ModelSpec,
    PointTreatment,
    Runtime,
    Targeting,
    TMLEMethod,
)
from cleverly.datasets import make_missing_outcome_bounded, missing_outcome_bounded_dgp

HERE = Path(__file__).resolve().parent
SEEDS = range(1000, 1500)
COVARIATES = ("W1", "W2", "W3")
ARM_GAMMA = {0: 0.0, 1: -1.0}


def target() -> tuple[float, float]:
    """The ATE and the complete-case target E[tau(W) | Delta = 1], by quadrature."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(40)
    weights = weights / weights.sum()
    grid = np.stack(np.meshgrid(nodes, nodes, nodes, indexing="ij"), -1).reshape(-1, 3)
    mass = np.prod(np.stack(np.meshgrid(weights, weights, weights, indexing="ij"), -1), -1)
    mass = mass.reshape(-1)
    law = missing_outcome_bounded_dgp()
    g = law.propensity(grid)
    respond = g * law.missingness(grid, 1.0) + (1 - g) * law.missingness(grid, 0.0)
    effect = law.outcome_mean(grid, 1.0, None) - law.outcome_mean(grid, 0.0, None)
    return float((effect * mass).sum()), float(
        (effect * respond * mass).sum() / (respond * mass).sum()
    )


def method(seed: int) -> TMLEMethod:
    return TMLEMethod(
        models=ModelSpec(
            outcome_learner=HistGradientBoostingRegressor(
                max_iter=200,
                learning_rate=0.05,
                max_depth=3,
                early_stopping=True,
                random_state=seed,
            ),
            treatment_learner=LogisticRegression(max_iter=1000, random_state=seed),
            missingness_learner=LogisticRegression(max_iter=1000, random_state=seed),
        ),
        cross_fitting=CrossFitting(n_folds=5),
        targeting=Targeting(q_bounds=(0.0, 1.0)),
        runtime=Runtime(random_state=seed, n_jobs=1),
    )


def one(seed: int) -> dict[str, float | int | bool | None]:
    ate, cc_target = target()
    frame, _ = make_missing_outcome_bounded(n=4_000, seed=seed)
    fit_method = method(seed)
    design = PointTreatment(outcome="Y", treatment="A", adjustment=COVARIATES, missingness="Delta")
    full = CausalStudy(frame, design=design).identify(ATE(reference=0)).estimate(method=fit_method)
    kept = frame[frame["Delta"] == 1].drop(columns=["Delta"])
    cc_design = PointTreatment(outcome="Y", treatment="A", adjustment=COVARIATES)
    cc = CausalStudy(kept, design=cc_design).identify(ATE(reference=0)).estimate(method=fit_method)
    point, cc_point = full["ate"], cc["ate"]
    return {
        "seed": seed,
        "psi": point.psi,
        "se": point.std_error,
        "covers": bool(point.ci[0] <= ate <= point.ci[1]),
        "cc_psi": cc_point.psi,
        "cc_se": cc_point.std_error,
        "cc_covers_ate": bool(cc_point.ci[0] <= ate <= cc_point.ci[1]),
        "cc_covers_target": bool(cc_point.ci[0] <= cc_target <= cc_point.ci[1]),
        "tip_ci": full.sensitivity.tipping_gamma(arm_gamma=ARM_GAMMA, use_ci=True),
        "tip_point": full.sensitivity.tipping_gamma(arm_gamma=ARM_GAMMA),
    }


def main() -> None:
    jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    with ProcessPoolExecutor(jobs) as pool:
        rows = list(pool.map(one, SEEDS))
    with (HERE / "sweep.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    ate, cc_target = target()
    n = len(rows)

    def rate(key: str) -> str:
        k = sum(bool(row[key]) for row in rows)
        p = k / n
        half = 1.959964 * np.sqrt(p * (1 - p) / n)
        return f"{k} of {n} ({p:.3f}; Wald 95% {p - half:.3f} to {p + half:.3f})"

    psi = np.array([row["psi"] for row in rows])
    se = np.array([row["se"] for row in rows])
    cc = np.array([row["cc_psi"] for row in rows])
    tips = np.array([np.nan if row["tip_ci"] is None else row["tip_ci"] for row in rows])
    lines = [
        f"draws: {n}, seeds {SEEDS.start} to {SEEDS.stop - 1}",
        f"truth: ATE {ate:.6f}, complete-case target {cc_target:.6f} (quadrature)",
        f"Step 6: covers ATE {rate('covers')}; bias {psi.mean() - ate:+.5f} "
        f"(MC SE {psi.std(ddof=1) / np.sqrt(n):.5f}); SE/SD {se.mean() / psi.std(ddof=1):.3f}",
        f"Step 7: covers ATE {rate('cc_covers_ate')}; covers target {rate('cc_covers_target')}; "
        f"mean minus target {cc.mean() - cc_target:+.5f} (MC SE {cc.std(ddof=1) / np.sqrt(n):.5f})",
        f"point tipping gamma found on {sum(row['tip_point'] is not None for row in rows)} draws",
        f"interval tipping gamma: found on {int(np.isfinite(tips).sum())} draws; 5% "
        f"{np.nanpercentile(tips, 5):.4f}, 95% {np.nanpercentile(tips, 95):.4f}",
    ]
    (HERE / "summary.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
