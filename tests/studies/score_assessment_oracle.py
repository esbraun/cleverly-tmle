"""The oracle reference of the ``score-assessment`` study (X97 plan, rev 2.2).

The single-score risk difference against the best constant carries a second-order term from
estimating the constant: the plug-in of a minimum sits below the true minimum by about
``Var(R_n'(c_0)) / (2 R''(c_0))``, which is ``O(1/n)``.  On the log-ratio scale it is large next
to the row's own spread, so it shows as a positive standardized bias and an SE ratio above 1.
This module measures both without a learner, and without the library's estimator code, so the
prediction it records does not share code with the estimator it judges.

For every row of :data:`ROWS` it draws :data:`REPLICATES` samples of the group's law at the
group's ``n`` from the named stream ``stream_seed(STUDY, "oracle_reference", label, block)``,
computes the one-step estimate from first principles (AIPW pseudo-outcomes, then the risk
functional) and its influence-curve SE, and records the standardized bias, the SE ratio and the
Wald coverage with their Monte Carlo SEs.

- The log-ratio calibration groups use the law's own arm regressions and mechanism (nuisance
  ``"law"``): the reference the SE-ratio rows of plan rev 2.2 (B2) and the red-cell route (S4)
  read.  The rows at ``n = 2,000`` are the n rule's (``N_RULE_GROUPS``); those at 4,000 are
  the run's.
- The two DR groups use the declared wrong outcome regression and a logistic mechanism fitted
  on each draw, the limits of the design's own nuisances (nuisance ``"declared_q_fitted_g"``):
  the check of each bias row's analytic second-order bias (B1).

The committed summary is ``score_assessment_oracle.csv`` beside this module, written by::

    python -m tests.studies.score_assessment_oracle --jobs 4
"""

from __future__ import annotations

import argparse
import math
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit

from tests.studies import conditional_projection as cp
from tests.studies import score_assessment as study
from tests.studies.evidence.seeds import stream_seed

#: The stream the reference draws from.
STREAM = "oracle_reference"

#: Draws of each group.
REPLICATES = 40_000

#: Draws per seeded block, the unit of parallel work.
BLOCK = 1_000

#: The rows the reference records, by group.
ROWS: dict[str, tuple[str, ...]] = {
    "single_crr_2k": ("S_risk_difference",),
    "paired_crr_2k": ("S_risk_difference", "SB_risk_difference", "paired"),
    "single_crr_4k": ("S_risk_difference",),
    "paired_crr_4k": ("S_risk_difference", "SB_risk_difference", "paired"),
    "paired_dr_4k": study.DR_ROWS,
    "paired_crr_dr_4k": study.DR_ROWS,
}

COLUMNS = (
    "group",
    "label",
    "n",
    "nuisance",
    "replicates",
    "standardized_bias",
    "standardized_bias_se",
    "se_ratio",
    "se_ratio_se",
    "coverage",
    "sd",
)


def _logistic_mechanism(w: np.ndarray, a: np.ndarray) -> np.ndarray:
    """``g(1 | W)`` from an unpenalised main-terms logistic regression, by Newton's method."""
    x = np.column_stack([np.ones(len(a)), w])
    beta = np.zeros(x.shape[1])
    for _ in range(50):
        p = expit(x @ beta)
        step = np.linalg.solve((x * (p * (1.0 - p))[:, None]).T @ x, x.T @ (a - p))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return np.asarray(expit(x @ beta))


def _softplus(theta: np.ndarray) -> np.ndarray:
    return np.asarray(np.logaddexp(0.0, theta))


def _one_draw(group: study.Group, frame: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """``label -> (estimate, influence-curve SE)`` of the one-step on one draw."""
    w = frame.loc[:, ["W1", "W2", "W3"]].to_numpy(dtype=float)
    a = frame["A"].to_numpy(dtype=float)
    y = frame["Y"].to_numpy(dtype=float)
    n = len(y)
    if group.learner == "declared":
        q1 = expit(cp.outcome_logit(w, 1) + cp.declared_error("noweight", w, 1))
        q0 = expit(cp.outcome_logit(w, 0) + cp.declared_error("noweight", w, 0))
        g = _logistic_mechanism(w, a)
    else:
        q1, q0 = study.arm_means(group.law, w)
        g = study.mechanism(group.law, w)
    d1 = q1 + a * (y - q1) / g
    d0 = q0 + (1.0 - a) * (y - q0) / (1.0 - g)
    pair = study.SCORES[group.kind]
    s1 = pair.first(w)
    s2 = pair.second(w) if pair.second is not None else None

    def summary(curve: np.ndarray, value: float) -> tuple[float, float]:
        return value, float(np.std(curve, ddof=1) / math.sqrt(n))

    out: dict[str, tuple[float, float]] = {}
    if group.scale == "log_ratio":
        c = math.log(float(d1.mean()) / float(d0.mean()))

        def augmented(theta: np.ndarray) -> np.ndarray:
            return np.asarray(d1 * _softplus(-theta) + d0 * _softplus(theta))

        at_c = augmented(np.full(n, c))
        for column, s in ((study.FIRST, s1), (study.SECOND, s2)):
            if s is None:
                continue
            at_s = augmented(s)
            curve = (at_s - at_s.mean()) - (at_c - at_c.mean())
            out[f"{column}_risk_difference"] = summary(curve, float(at_s.mean() - at_c.mean()))
        if s2 is not None:
            difference = augmented(s1) - augmented(s2)
            out["paired"] = summary(difference, float(difference.mean()))
        return out
    d = d1 - d0
    ate = float(d.mean())
    for column, s in ((study.FIRST, s1), (study.SECOND, s2)):
        if s is None:
            continue
        loss = s**2 - 2.0 * s * d
        curve = (loss - loss.mean()) + 2.0 * ate * (d - ate)
        out[f"{column}_risk_difference"] = summary(curve, float(loss.mean() + ate**2))
    if s2 is not None:
        difference = (s1**2 - s2**2) - 2.0 * (s1 - s2) * d
        out["paired"] = summary(difference, float(difference.mean()))
    return out


def _block(key: tuple[str, int]) -> dict[str, np.ndarray]:
    label, block = key
    group = study.PILOT_GROUP_BY_LABEL[label]
    rng = np.random.default_rng(stream_seed(study.STUDY, STREAM, label, block))
    values: dict[str, list[tuple[float, float]]] = {row: [] for row in ROWS[label]}
    for _ in range(BLOCK):
        draw = _one_draw(group, study.sample(group.law, group.n, rng))
        for row in values:
            values[row].append(draw[row])
    return {row: np.asarray(pairs) for row, pairs in values.items()}


def reference(*, jobs: int, replicates: int = REPLICATES) -> pd.DataFrame:
    """Simulate every group of :data:`ROWS` and summarise each row."""
    rows = []
    blocks = replicates // BLOCK
    with Pool(jobs) as pool:
        for label, labels in ROWS.items():
            group = study.PILOT_GROUP_BY_LABEL[label]
            truths = study.truth(group.law, group.scale, group.kind)
            parts = pool.map(_block, [(label, b) for b in range(blocks)])
            for row in labels:
                pairs = np.concatenate([part[row] for part in parts])
                estimate, se = pairs[:, 0], pairs[:, 1]
                sd = float(estimate.std(ddof=1))
                ratio = float(se.mean() / sd)
                kurtosis = float(stats.kurtosis(estimate, fisher=False))
                rows.append(
                    {
                        "group": label,
                        "label": row,
                        "n": group.n,
                        "nuisance": "declared_q_fitted_g" if group.learner == "declared" else "law",
                        "replicates": len(estimate),
                        "standardized_bias": float((estimate.mean() - truths[row]) / sd),
                        "standardized_bias_se": 1.0 / math.sqrt(len(estimate)),
                        "se_ratio": ratio,
                        "se_ratio_se": ratio * math.sqrt((kurtosis - 1.0) / (4.0 * len(estimate))),
                        "coverage": float(
                            np.mean(np.abs(estimate - truths[row]) <= study.CRITICAL * se)
                        ),
                        "sd": sd,
                    }
                )
            print(f"{label}: {blocks * BLOCK} draws", flush=True)
    return pd.DataFrame(rows, columns=list(COLUMNS))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    arguments = parser.parse_args()
    summary = reference(jobs=arguments.jobs, replicates=arguments.replicates)
    summary.to_csv(study.ORACLE_CSV, index=False, float_format="%.6g")
    print(f"wrote {study.ORACLE_CSV}", flush=True)


if __name__ == "__main__":
    main()
