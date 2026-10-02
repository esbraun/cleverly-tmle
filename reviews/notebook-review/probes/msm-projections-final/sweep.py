"""Seed sweep of the msm-projections page configuration (n = 3000, in-sample linear learners).

Each draw refits the page's fits on ``make_multi_arm(n=3000, seed=s)``:

- the per-contact trend (Step 7), with the fitted multinomial g and with the true g;
- the per-step trend (Step 8);
- the 1:10:1 fixed-weight trend (Step 10);
- the arm means and the ``contrast`` miss at medium (Steps 6 and 9).

Truths come from the structural equations (``truth.py``), not from the library.

Run: .venv/Scripts/python.exe reviews/notebook-review/probes/msm-projections-final/sweep.py 30000 33000 10
Writes sweep.csv beside this script and prints the summary.
"""

import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
WORKTREE = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))

ARMS = ("low", "medium", "high")
CONTACTS = {"low": 1.0, "medium": 2.0, "high": 6.0}
STEPS = {"low": 0.0, "medium": 1.0, "high": 2.0}
FIXED = {"low": 1.0, "medium": 10.0, "high": 1.0}
MEANS = np.array([0.0, 0.6, 1.44])
X_CONTACT = np.column_stack([np.ones(3), [1.0, 2.0, 6.0]])
X_STEP = np.column_stack([np.ones(3), [0.0, 1.0, 2.0]])


def _projection(design, weights):
    return np.linalg.solve(design.T @ (weights[:, None] * design), design.T @ (weights * MEANS))


UNIFORM = _projection(X_CONTACT, np.ones(3))  # slope 0.265714
FIXED_TARGET = _projection(X_CONTACT, np.array([1.0, 10.0, 1.0]))  # slope 0.24
STEP_TARGET = _projection(X_STEP, np.ones(3))  # slope 0.72
POP_MISS = (X_CONTACT @ UNIFORM - MEANS)[1]
MISFIT_WEIGHTS = (X_CONTACT @ np.linalg.pinv(X_CONTACT))[1] - np.eye(3)[1]


class TrueMechanism(BaseEstimator):
    """The law's softmax arm probabilities, from (W1, W2) = the first two adjustment columns."""

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        w = np.asarray(X, dtype=float)
        logits = np.column_stack(
            [np.zeros(len(w)), 0.8 * w[:, 0] - 0.4 * w[:, 1], -0.5 * w[:, 0] + 0.8 * w[:, 1]]
        )
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        columns = []
        for label in self.classes_:
            key = label if isinstance(label, str) else sorted(ARMS)[int(label)]
            columns.append(ARMS.index(key))
        return p[:, columns]


def _covers(estimate, value):
    low, high = estimate.ci
    return bool(low <= value <= high)


def run(seed):
    warnings.filterwarnings("ignore")
    import cleverly
    from cleverly import (
        CausalStudy,
        CounterfactualMean,
        CrossFitting,
        ModelSpec,
        MSMProjection,
        PointTreatment,
        Runtime,
        TMLEMethod,
    )
    from cleverly.datasets import make_multi_arm
    from cleverly.msm import MSM

    assert os.path.abspath(cleverly.__file__).startswith(os.path.join(WORKTREE, "src"))

    frame, _ = make_multi_arm(n=3_000, seed=seed)
    study = CausalStudy(
        frame,
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3")),
    )

    def method(treatment_learner):
        return TMLEMethod(
            models=ModelSpec(outcome_learner=LinearRegression(), treatment_learner=treatment_learner),
            cross_fitting=CrossFitting(enabled=False),
            runtime=Runtime(random_state=61, n_jobs=1),
        )

    fitted = method(LogisticRegression(max_iter=1000, random_state=61))
    oracle = method(TrueMechanism())

    def msm(coding, weights=None):
        def design(arm, data):
            return np.column_stack([np.ones(len(data)), np.full(len(data), coding[arm])])

        if weights is None:
            return MSM(design=design, terms=("i", "s"), design_kind="known")
        return MSM(
            design=design,
            terms=("i", "s"),
            design_kind="known",
            weights=lambda arm, data: np.full(len(data), weights[arm]),
            weights_kind="known",
        )

    trend = study.identify(MSMProjection(msm(CONTACTS))).estimate(method=fitted)
    trend_oracle = study.identify(MSMProjection(msm(CONTACTS))).estimate(method=oracle)
    step = study.identify(MSMProjection(msm(STEPS))).estimate(method=fitted)
    weighted = study.identify(MSMProjection(msm(CONTACTS, FIXED))).estimate(method=fitted)
    arm = study.identify(CounterfactualMean()).estimate(method=fitted)

    means = np.array([arm[f"ey[{a}]"].psi for a in ARMS])
    miss = arm.contrast(
        lambda m: float(MISFIT_WEIGHTS @ m),
        [f"ey[{a}]" for a in ARMS],
        name="miss",
        gradient=lambda m: MISFIT_WEIGHTS,
    )
    line = X_CONTACT @ np.array([trend["msm[i]"].psi, trend["msm[s]"].psi])
    shares = frame["A"].value_counts(normalize=True).loc[list(ARMS)].to_numpy()
    share_slope = np.linalg.solve(
        X_CONTACT.T @ (shares[:, None] * X_CONTACT), X_CONTACT.T @ (shares * MEANS)
    )[1]
    slope = trend["msm[s]"]
    return {
        "seed": seed,
        "slope": slope.psi,
        "slope_se": slope.std_error,
        "slope_cover": _covers(slope, UNIFORM[1]),
        "intercept_cover": _covers(trend["msm[i]"], UNIFORM[0]),
        "oracle_slope": trend_oracle["msm[s]"].psi,
        "oracle_slope_se": trend_oracle["msm[s]"].std_error,
        "oracle_slope_cover": _covers(trend_oracle["msm[s]"], UNIFORM[1]),
        "step_slope": step["msm[s]"].psi,
        "step_slope_se": step["msm[s]"].std_error,
        "step_slope_cover": _covers(step["msm[s]"], STEP_TARGET[1]),
        "weighted_slope": weighted["msm[s]"].psi,
        "weighted_cover": _covers(weighted["msm[s]"], FIXED_TARGET[1]),
        "uniform_excludes_fixed": not _covers(slope, FIXED_TARGET[1]),
        "weighted_excludes_uniform": not _covers(weighted["msm[s]"], UNIFORM[1]),
        "share_shift_over_se": abs(share_slope - UNIFORM[1]) / slope.std_error,
        "arms_cover": all(_covers(arm[f"ey[{a}]"], m) for a, m in zip(ARMS, MEANS, strict=True)),
        "miss_excludes_zero": miss.ci[1] < 0.0,
        "miss_covers_population": _covers(miss, POP_MISS),
        "miss_minus_step7": miss.psi - (line[1] - means[1]),
        "min_fitted_g": float(np.asarray(trend.nuisance.propensity.values).min()),
    }


def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def summarize(df):
    n = len(df)
    print(f"draws: {n}, seeds {df['seed'].min()} to {df['seed'].max()}")
    for column in (
        "slope_cover",
        "oracle_slope_cover",
        "intercept_cover",
        "step_slope_cover",
        "weighted_cover",
        "uniform_excludes_fixed",
        "weighted_excludes_uniform",
        "arms_cover",
        "miss_excludes_zero",
        "miss_covers_population",
    ):
        k = int(df[column].sum())
        low, high = wilson(k, n)
        print(f"{column}: {k}/{n} = {k / n:.4f} (Wilson 95% {low:.4f} to {high:.4f})")
    for label, psi, se, target in (
        ("per-contact slope, fitted g", "slope", "slope_se", UNIFORM[1]),
        ("per-contact slope, true g", "oracle_slope", "oracle_slope_se", UNIFORM[1]),
        ("per-step slope, fitted g", "step_slope", "step_slope_se", STEP_TARGET[1]),
    ):
        print(
            f"{label}: mean SE / empirical SD {df[se].mean() / df[psi].std():.3f}, "
            f"bias {df[psi].mean() - target:+.5f}"
        )
    print(
        "share-weight shift below one slope SE:",
        f"{(df['share_shift_over_se'] < 1).mean():.4f}",
        f"(median ratio {df['share_shift_over_se'].median():.3f})",
    )
    gap = df["miss_minus_step7"].abs()
    print(
        "|contrast miss - Step 7 miss|:",
        f"median {gap.median():.5f}, share below 0.0005 {(gap < 5e-4).mean():.4f},",
        f"share below 0.001 {(gap < 1e-3).mean():.4f}, max {gap.max():.5f}",
    )
    for start in range(0, n, 1000):
        block = df.iloc[start : start + 1000]
        print(
            f"block seeds {block['seed'].min()}-{block['seed'].max()}: slope cover "
            f"{block['slope_cover'].mean():.4f}, true-g slope cover {block['oracle_slope_cover'].mean():.4f}"
        )


if __name__ == "__main__":
    first, last, workers = (int(value) for value in sys.argv[1:4])
    with Pool(workers) as pool:
        rows = pool.map(run, range(first, last), chunksize=20)
    df = pd.DataFrame(rows).sort_values("seed")
    df.to_csv(os.path.join(HERE, "sweep.csv"), index=False, lineterminator="\n")
    summarize(df)
