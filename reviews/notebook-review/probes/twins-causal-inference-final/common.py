"""The TWINS notebook pipeline, sections 2 and 6 to 9, for one pair-sample seed.

The code mirrors `docs/examples/twins-causal-inference.ipynb` after change N10. The notebook uses
one seed, `SEED`, for the pair sample, the learners, the folds, and the assessment, so a sweep
over pair-sample seeds sets all of them together. The three CSV files are read from the pinned
CEVAE commit; set `TWINS_DATA` to a directory that holds local copies to skip the download.
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import re
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import cleverly
from cleverly import (
    ATE,
    CausalStudy,
    CrossFitting,
    Inference,
    ModelSpec,
    PointTreatment,
    Runtime,
    SuperLearner,
    TMLEMethod,
)

WORKTREE = Path(__file__).resolve().parents[4]
assert Path(cleverly.__file__).resolve().is_relative_to(WORKTREE / "src"), cleverly.__file__

COMMIT = "9081f863e24ce21bd34c8d6a41bf0edc7d1b65dd"
BASE_URL = f"https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/{COMMIT}/datasets/TWINS"
FILES = {
    "X": "twin_pairs_X_3years_samesex.csv",
    "T": "twin_pairs_T_3years_samesex.csv",
    "Y": "twin_pairs_Y_3years_samesex.csv",
}
N_PAIRS = 6_000
SHARED = [
    "mager8", "meduc6", "mrace", "dmar", "mplbir_reg", "data_year",
    "diabetes", "chyper", "phyper", "preterm", "tobacco", "alcohol",
]
CATEGORICAL = ["mager8", "meduc6", "mrace", "mplbir_reg", "data_year"]
BINARY = ["dmar", "diabetes", "chyper", "phyper", "preterm", "tobacco", "alcohol"]
Q_BOUND = 0.0005  # the package shrinks Q into [0.0005, 0.9995] before targeting


def load_raw() -> dict[str, pd.DataFrame]:
    local = os.environ.get("TWINS_DATA")
    base = local if local else BASE_URL
    return {name: pd.read_csv(f"{base}/{filename}") for name, filename in FILES.items()}


RAW = load_raw()


def build(seed: int) -> tuple[pd.DataFrame, tuple[str, ...]]:
    pair_rows = RAW["X"].sample(n=N_PAIRS, random_state=seed).index.sort_values().to_numpy()
    X = RAW["X"].loc[pair_rows].reset_index(drop=True)
    T = RAW["T"].loc[pair_rows].reset_index(drop=True)
    Y = RAW["Y"].loc[pair_rows].reset_index(drop=True)
    children = []
    for twin in (0, 1):
        child = X[SHARED].copy()
        child["pair_id"] = np.arange(N_PAIRS)
        child["twin"] = twin
        child["birth_weight_g"] = T[f"dbirwt_{twin}"]
        child["mortality_1y"] = Y[f"mort_{twin}"].astype(int)
        children.append(child)
    data = (
        pd.concat(children, ignore_index=True)
        .sort_values(["pair_id", "twin"])
        .reset_index(drop=True)
    )
    data["low_birth_weight"] = (data["birth_weight_g"] < 2_500).astype(int)
    adjustment: list[str] = []
    patterns: set[bytes] = set()
    for column in BINARY:
        missing = data[column].isna()
        signature = missing.to_numpy().tobytes()
        if missing.any() and signature not in patterns:
            data[f"{column}_missing"] = missing.astype(float)
            adjustment.append(f"{column}_missing")
            patterns.add(signature)
        data[column] = data[column].fillna(0).astype(float)
        adjustment.append(column)
    encoded = pd.get_dummies(
        data[CATEGORICAL].fillna(-1).astype("category"), prefix=CATEGORICAL, dtype=float
    )
    data = pd.concat([data, encoded], axis=1)
    adjustment.extend(encoded.columns.tolist())
    return data, tuple(adjustment)


def scaled_logistic(seed: int):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3_000, random_state=seed))


BOOSTERS = {
    # the stored notebook's booster before N10
    "current": lambda seed: HistGradientBoostingClassifier(
        max_iter=100, min_samples_leaf=30, random_state=seed
    ),
    # learner (b) of plan.md: shallow and regularized
    "shallow": lambda seed: HistGradientBoostingClassifier(
        max_depth=2, learning_rate=0.05, max_iter=200, l2_regularization=1.0, random_state=seed
    ),
    # shallow, regularized, and early-stopped on a validation split
    "shallow_early": lambda seed: HistGradientBoostingClassifier(
        max_depth=2,
        learning_rate=0.05,
        max_iter=200,
        l2_regularization=1.0,
        min_samples_leaf=100,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=10,
        random_state=seed,
    ),
}


def expit(value):
    return 1.0 / (1.0 + np.exp(-value))


def logit(probability):
    clipped = np.clip(probability, 1e-12, 1 - 1e-12)
    return np.log(clipped / (1 - clipped))


def hand_tmle(data, adjustment, seed, shrink_q=True):
    """The section 6 hand-built TMLE with two arm-specific clever covariates."""
    W = data.loc[:, adjustment]
    A = data["low_birth_weight"].to_numpy()
    outcome = data["mortality_1y"].to_numpy()
    n = len(data)
    design = pd.concat([data[["low_birth_weight"]], W], axis=1)
    q_model = scaled_logistic(seed).fit(design, outcome)
    q_obs = q_model.predict_proba(design)[:, 1]
    q1 = q_model.predict_proba(design.assign(low_birth_weight=1))[:, 1]
    q0 = q_model.predict_proba(design.assign(low_birth_weight=0))[:, 1]
    gcomp = q1.mean() - q0.mean()
    g_bound = 5 / (np.sqrt(n) * np.log(n))
    g_raw = scaled_logistic(seed).fit(W, A).predict_proba(W)[:, 1]
    g1 = np.clip(g_raw, g_bound, 1 - g_bound)
    ipw = np.mean(A * outcome / g1) - np.mean((1 - A) * outcome / (1 - g1))
    if shrink_q:
        q_obs, q1, q0 = (np.clip(v, Q_BOUND, 1 - Q_BOUND) for v in (q_obs, q1, q0))
    H = np.column_stack([A / g1, (1 - A) / (1 - g1)])
    initial_scores = H.T @ (outcome - q_obs) / n
    epsilon = np.zeros(2)
    for _ in range(50):
        q_star_obs = expit(logit(q_obs) + H @ epsilon)
        gradient = H.T @ (outcome - q_star_obs) / n
        hessian = (H * (q_star_obs * (1 - q_star_obs))[:, None]).T @ H / n
        step = np.linalg.solve(hessian, gradient)
        epsilon += step
        if np.abs(step).max() < 1e-12:
            break
    q_star_obs = expit(logit(q_obs) + H @ epsilon)
    q1_star = expit(logit(q1) + epsilon[0] / g1)
    q0_star = expit(logit(q0) + epsilon[1] / (1 - g1))
    final_scores = H.T @ (outcome - q_star_obs) / n
    return {
        "gcomp": gcomp,
        "ipw": ipw,
        "hand_tmle": q1_star.mean() - q0_star.mean(),
        "eps1": epsilon[0],
        "eps0": epsilon[1],
        "score1_before": initial_scores[0],
        "score0_before": initial_scores[1],
        "max_score_after": float(np.abs(final_scores).max()),
        "g_raw_min": g_raw.min(),
        "g_raw_max": g_raw.max(),
        "q_below_bound": int((np.minimum(q1, q0) <= Q_BOUND).sum()),
    }


def study_effect(data, adjustment):
    return CausalStudy(
        data,
        design=PointTreatment(
            outcome="mortality_1y",
            treatment="low_birth_weight",
            adjustment=adjustment,
            cluster="pair_id",
            outcome_family="binomial",
        ),
    ).identify(ATE(reference=0))


def super_learner(seed, booster):
    return SuperLearner(
        library=[("logistic", scaled_logistic(seed)), ("boosting", BOOSTERS[booster](seed))],
        task="classification",
        n_folds=3,
        random_state=seed,
        n_jobs=1,
    )


def _float(pattern, text):
    match = re.search(pattern, text)
    return float(match.group(1)) if match else np.nan


def run(seed: int, booster: str = "shallow", sensitivity: bool = True) -> dict:
    warnings.simplefilter("ignore")
    data, adjustment = build(seed)
    record: dict = {"seed": seed, "booster": booster}
    record.update(hand_tmle(data, adjustment, seed))
    effect = study_effect(data, adjustment)
    inference = Inference(alpha=0.05, simultaneous=False)
    runtime = Runtime(random_state=seed, n_jobs=1)
    ordinary = effect.estimate(
        method=TMLEMethod(
            models=ModelSpec(
                outcome_learner=scaled_logistic(seed), treatment_learner=scaled_logistic(seed)
            ),
            cross_fitting=CrossFitting(enabled=False),
            inference=inference,
            runtime=runtime,
        )
    )["ate"]
    record["package_tmle"] = ordinary.psi
    record["package_se"] = ordinary.std_error
    record["gap"] = ordinary.psi - record["hand_tmle"]
    record["gap_over_se"] = abs(record["gap"]) / ordinary.std_error
    if not sensitivity:
        return record
    try:
        flexible = effect.estimate(
            method=TMLEMethod(
                models=ModelSpec(
                    outcome_learner=super_learner(seed, booster),
                    treatment_learner=super_learner(seed, booster),
                ),
                cross_fitting=CrossFitting(n_folds=3, learner_folds=3),
                inference=inference,
                runtime=runtime,
            )
        )
        estimate = flexible["ate"]
        record["flexible_psi"] = estimate.psi
        record["flexible_se"] = estimate.std_error
        g_hat = np.asarray(flexible.nuisance.propensity.arm(1.0))
        record["cf_g_min"] = g_hat.min()
        record["cf_g_max"] = g_hat.max()
        assessment = flexible.assess(
            random_state=seed,
            arguments={
                "omitted_confounding": {"estimand": "ate", "cf_y": 0.05, "cf_d": 0.05, "rho": 1.0}
            },
        )
        summary = assessment.summary()
        record["nu2"] = _float(r"nu2=(-?[0-9.]+)", summary)
        record["attention"] = ";".join(item.name for item in assessment.attention)
        nuisance = assessment.report("nuisance_models").summary()
        for model in ("propensity", "outcome"):
            line = next(
                (row for row in nuisance.splitlines() if row.startswith(f"{model} ")), ""
            ).split()
            record[f"{model}_auc"] = float(line[1]) if len(line) > 6 else np.nan
            record[f"{model}_cal_slope"] = float(line[6]) if len(line) > 6 else np.nan
            weight = re.search(rf"{model}: super learner weights logistic=([0-9.]+)", nuisance)
            record[f"{model}_logistic_weight"] = float(weight.group(1)) if weight else np.nan
        record["calibration_flags"] = nuisance.count("more extreme than the observed rates")
        omitted = assessment.report("omitted_confounding").summary()
        record["rv"] = _float(r"robustness value RV\s+=\s+([0-9.]+)", omitted)
        record["sign_survives"] = "the sign of the effect survives" in omitted
        benchmark = flexible.sensitivity.benchmark(
            covariates=("preterm", "tobacco"), estimand="ate", random_state=seed
        ).summary()
        record["bench_cf_y"] = _float(r"implied cf_y = (-?[0-9.]+)", benchmark)
        record["bench_cf_d"] = _float(r"cf_d = (-?[0-9.]+), rho", benchmark)
        record["sensitivity_ran"] = True
        record["error"] = ""
    except Exception as exc:  # noqa: BLE001 - the sweep records every failure
        record["sensitivity_ran"] = False
        record["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        record["traceback"] = traceback.format_exc()[-1500:]
    return record
