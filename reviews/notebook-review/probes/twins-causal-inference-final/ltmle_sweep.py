"""Sweep the section 11 semi-synthetic LTMLE over synthetic draws.

Usage: python ltmle_sweep.py <output.csv> <draws>

The baseline rows are the notebook's pair sample (seed 2026). Draw k uses the generator seed
2026 + 1 + k, so draw 0 is the notebook's draw. Each draw refits the notebook's LTMLE and records
whether its interval contains the exact truth and the naive observed-regime contrast.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd

SEED = 2026


def _baseline():
    import common

    rows = common.RAW["X"].sample(n=common.N_PAIRS, random_state=SEED).index.sort_values()
    X = common.RAW["X"].loc[rows].reset_index(drop=True)
    w1 = X["dmar"].fillna(0).to_numpy(float)
    w2 = np.maximum(
        X["diabetes"].fillna(0).to_numpy(float), X["chyper"].fillna(0).to_numpy(float)
    )
    return w1, w2


def expit(value):
    return 1.0 / (1.0 + np.exp(-value))


def truth(w2):
    """Always-minus-never risk, by enumerating L2 for each baseline row."""

    def risk(a1, a2):
        p_l2 = expit(-0.7 + 0.9 * a1 + 0.8 * w2)
        total = 0.0
        for l2, weight in ((0, 1 - p_l2), (1, p_l2)):
            total = total + weight * expit(-2.2 - 0.35 * a1 - 0.55 * a2 + 0.9 * l2 + 0.5 * w2)
        return np.mean(total)

    return risk(1, 1) - risk(0, 0)


def _one(draw: int) -> dict:
    from sklearn.linear_model import LinearRegression, LogisticRegression

    from cleverly import (
        CausalStudy,
        CrossFitting,
        Inference,
        LongitudinalTreatment,
        ModelSpec,
        RegimeContrast,
        Runtime,
        TMLEMethod,
    )

    w1, w2 = _baseline()
    target = truth(w2)
    rng = np.random.default_rng(SEED + 1 + draw)
    a1 = rng.binomial(1, expit(-0.2 + 0.8 * w1 - 0.6 * w2))
    l2 = rng.binomial(1, expit(-0.7 + 0.9 * a1 + 0.8 * w2))
    a2 = rng.binomial(1, expit(-0.3 + 0.5 * a1 + l2 + 0.4 * w1))
    y = rng.binomial(1, expit(-2.2 - 0.35 * a1 - 0.55 * a2 + 0.9 * l2 + 0.5 * w2))
    semi = pd.DataFrame(
        {"sim_W1": w1, "sim_W2": w2, "sim_A1": a1, "sim_L2": l2, "sim_A2": a2, "sim_Y": y}
    )
    study = CausalStudy(
        semi,
        design=LongitudinalTreatment(
            outcome="sim_Y",
            treatment=("sim_A1", "sim_A2"),
            baseline=("sim_W1", "sim_W2"),
            time_varying=((), ("sim_L2",)),
            outcome_family="binomial",
        ),
    )
    effect = study.identify(RegimeContrast({"always": 1, "never": 0}, reference="never"))
    method = TMLEMethod(
        models=ModelSpec(
            outcome_learner=LogisticRegression(max_iter=3_000, random_state=SEED),
            pseudo_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=3_000, random_state=SEED),
        ),
        cross_fitting=CrossFitting(n_folds=3, learner_folds=3),
        inference=Inference(alpha=0.05, simultaneous=False),
        runtime=Runtime(random_state=SEED, n_jobs=1),
    )
    estimate = effect.estimate(method=method)["ate_regimen[always vs never]"]
    risks = semi.groupby(["sim_A1", "sim_A2"])["sim_Y"].mean()
    naive = risks.loc[(1, 1)] - risks.loc[(0, 0)]
    low, high = estimate.ci
    return {
        "draw": draw,
        "truth": target,
        "psi": estimate.psi,
        "se": estimate.std_error,
        "ci_low": low,
        "ci_high": high,
        "naive": naive,
        "covers_truth": bool(low <= target <= high),
        "covers_naive": bool(low <= naive <= high),
    }


def main() -> None:
    output, draws = sys.argv[1], int(sys.argv[2])
    with ProcessPoolExecutor(max_workers=10) as pool:
        frame = pd.DataFrame(list(pool.map(_one, range(draws))))
    frame.to_csv(output, index=False, lineterminator="\n")
    print(frame.to_string(index=False))
    print()
    for label, part in (("first 40", frame.head(40)), (f"all {draws}", frame)):
        print(
            f"{label}: interval contains the truth {int(part.covers_truth.sum())}/{len(part)}; "
            f"contains the naive contrast {int(part.covers_naive.sum())}/{len(part)}; "
            f"mean naive - truth {(part.naive - part.truth).mean():.4f}; "
            f"mean se {part.se.mean():.4f}"
        )


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
