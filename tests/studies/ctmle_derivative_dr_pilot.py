"""The declared pilot that sizes the ``outcome_correct`` law of the ``ctmle-derivative`` study.

The first run published a red ``double_robustness/both_wrong`` control.  Its law put the term
the main-terms lasso cannot read on ``W1 W2``.  ``W1`` is an instrument, so the term barely
moves the outcome: the ordinary TMLE's both-wrong limit bias was about -0.004 and C-TMLE0's
about +0.006, both on the 0.0057 margin at ``n = 2,000``.  The control could not separate by
construction, which is a design bug under the roadmap's red-cell rule.

The redesigned law is the primary law plus ``beta W2 |W2|`` in the treatment logit.  ``W2`` is
a confounder, so the misspecification moves the outcome, and the positive cell gets harder.
The pilot assesses every value of :data:`CURVATURE_GRID`.  ``beta`` is the smallest value
that passes all five checks below.

1. The main-terms projection's KL gap is at least :data:`KL_FLOOR`, twice the primary law's
   ceiling, so the lasso target is wrong.  KL measures how wrong ``g`` is, not how much the
   error moves the estimate: the first run's law had a KL gap of 0.046 and moved neither
   estimator past the margin.  Checks 3 to 5 measure the bias the error causes.
2. The share of mechanism values outside ``G_BOUNDS`` is at most :data:`DR_TRUNCATION_CEILING`.
3. The paired ordinary TMLE's limit bias is at least :data:`TMLE_LIMIT_FLOOR` sigma.
4. C-TMLE0's limit bias, read off a straight-line trend over the grid, is at least
   :data:`CTMLE_LIMIT_FLOOR` sigma.
5. At study conditions, :data:`PILOT_REPLICATES` draws give a both-wrong control pass
   probability and an ``outcome_correct`` equivalence pass probability of at least
   :data:`PASS_PROBABILITY` each.  Each probability reads the cell's bias off a weighted
   quadratic over the grid.  Its error combines the quadratic's standard error, inflated by
   the lack of fit, with the study's own noise.  The point value at the same coefficient, with
   the pilot's error and the study's noise, must also reach :data:`POINT_PASS_PROBABILITY`.

Checks 4 and 5 read trends because single grid points are noisy.  The point pass
probabilities of the first search were 0.69, 0.07, 0.73, 0.99, 0.61 and 0.99 on a bias that
rises with ``beta``.  A rule that stops at the first noisy pass picks a value whose estimate is
high by chance.  A straight line was tried first and rejected: its lack of fit was
``chi^2 = 20.5`` on 6 degrees of freedom, and it left the study's noise out, so it predicted
pass probabilities near 1 where a fresh pilot measured about 0.83.  ``dr-pilot.csv`` keeps
the point values beside the trend values.

``sigma`` is the paired TMLE's standard error at ``n = 10^6`` rescaled to the study's
``n = 2,000``.  The limit fits use the declared lasso pair of ``penalties.csv`` and a constant
outcome regression.  The pilot draws its own seed stream, disjoint from every study cell.

Run ``python -m tests.studies.ctmle_derivative_dr_pilot`` to rewrite ``dr-pilot.csv`` and
``dr-pilot-replicates.csv``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LogisticRegression

from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import ctmle_derivative_laws as laws
from tests.studies.canonical_ctmle_derivative import (
    ARTIFACTS,
    PRIMARY_N,
    STUDY,
    declared_pair,
    estimator,
    paired_tmle,
)
from tests.studies.evidence.seeds import stream_seed

#: The candidate coefficients of ``W2 |W2|``, searched in order.
CURVATURE_GRID = tuple(round(0.45 + 0.05 * step, 2) for step in range(8))
KL_FLOOR = 2.0 * laws.PRIMARY_KL_CEILING
#: Excludes laws with gross practical positivity problems.  Check 5 guards the positive cell's
#: pass probability directly, so this ceiling does not have to.  A first ceiling of 0.05 left no
#: grid value passing: C-TMLE0's limit bias reached 0.40 sigma only at a share of 0.052.
DR_TRUNCATION_CEILING = 0.10
TMLE_LIMIT_FLOOR = 0.5
CTMLE_LIMIT_FLOOR = 0.40
PASS_PROBABILITY = 0.95
POINT_PASS_PROBABILITY = 0.90
PILOT_REPLICATES = 400
LIMIT_N = 1_000_000
LIMIT_SEEDS = (20263201, 20263202, 20263203)
#: The study's double-robustness budget and margin, which the pass probabilities use.
DR_REPLICATES = 1_200
BIAS_MARGIN_SD = STUDY.margins.standardized_bias
LEVEL_Z = float(norm.ppf(1.0 - (1.0 - STUDY.margins.confidence_level) / 2.0))

PILOT = ARTIFACTS / "dr-pilot.csv"
PILOT_REPLICATE_ROWS = ARTIFACTS / "dr-pilot-replicates.csv"


def law_for(curvature: float) -> laws.DerivativeLaw:
    """The primary law plus ``curvature W2 |W2|`` in the treatment logit."""
    return replace(laws.PRIMARY, name="outcome_correct", curvature=curvature)


def first_run_mechanism(w: np.ndarray) -> np.ndarray:
    """``P(A = 1 | W)`` of the first run's law: the primary score plus ``0.8 W1 W2``."""
    return np.asarray(expit(w @ np.asarray(laws._SCORE[0]) + 0.8 * w[:, 0] * w[:, 1]))


def projection_kl(mechanism: Callable[[np.ndarray], np.ndarray]) -> float:
    """The KL gap of the main-terms logistic projection of ``P(A = 1 | W)`` (check 1)."""
    rng = np.random.default_rng(20263003)
    w = rng.standard_normal((100_000, 4))
    g = mechanism(w)
    a = (rng.uniform(size=g.size) < g).astype(float)
    p = LogisticRegression(C=1e6, max_iter=2000, tol=1e-10).fit(w, a).predict_proba(w)[:, 1]
    return float(np.mean(g * np.log(g / p) + (1 - g) * np.log((1 - g) / (1 - p))))


def truncation_share(law: laws.DerivativeLaw) -> float:
    """The share of mechanism values outside ``G_BOUNDS`` (check 2)."""
    g = law.mechanism(np.random.default_rng(20263004).standard_normal((200_000, 4)))[:, 1]
    return float(np.mean((g < laws.G_BOUNDS[0]) | (g > laws.G_BOUNDS[1])))


def _limit(payload: tuple[float, int]) -> dict[str, float]:
    curvature, seed = payload
    frame = law_for(curvature).sample(LIMIT_N, seed)
    result = (
        estimator(lambdas=declared_pair(), outcome=DummyRegressor())
        .fit(frame, outcome="Y", treatment="A", covariates=list(laws.COVARIATES))
        .single()
    )
    plain = paired_tmle(result)["ate"]
    return {
        "ctmle": float(result.estimates["ate"].psi),
        "tmle": float(plain.psi),
        "sigma": float(plain.std_error) * np.sqrt(LIMIT_N / PRIMARY_N),
    }


def limit_biases(curvature: float, *, n_jobs: int = 1) -> dict[str, float]:
    """Checks 3 and 4: both limit biases, averaged over :data:`LIMIT_SEEDS`, in sigma."""
    rows = pd.DataFrame(
        map_parallel(_limit, [((curvature, seed),) for seed in LIMIT_SEEDS], n_jobs=n_jobs)
    )
    truth = law_for(curvature).truth()["ate"]
    sigma = float(rows["sigma"].mean())
    return {
        "ctmle_limit_bias": float(rows["ctmle"].mean() - truth),
        "tmle_limit_bias": float(rows["tmle"].mean() - truth),
        "sigma": sigma,
    }


def pilot_seed(curvature: float, cell: str, replicate: int) -> int:
    """The pilot's own stream: no study cell draws a ``dr_pilot`` label."""
    return stream_seed(STUDY, "dr_pilot", f"{curvature:.2f}", cell, replicate)


def _pilot_fit(payload: tuple[float, str, int]) -> dict[str, Any]:
    curvature, cell, replicate = payload
    law = law_for(curvature)
    frame = law.sample(PRIMARY_N, pilot_seed(curvature, cell, replicate))
    outcome = DummyRegressor() if cell == "both_wrong" else None
    result = (
        estimator(lambdas=None, outcome=outcome)
        .fit(frame, outcome="Y", treatment="A", covariates=list(laws.COVARIATES))
        .single()
    )
    estimate = result.estimates["ate"]
    plain = paired_tmle(result)["ate"]
    return {
        "curvature": curvature,
        "cell": cell,
        "replicate": replicate,
        "estimate": float(estimate.psi),
        "std_error": float(estimate.std_error),
        "tmle_estimate": float(plain.psi),
    }


def pilot_rows(
    curvature: float, *, replicates: int = PILOT_REPLICATES, n_jobs: int = 1
) -> pd.DataFrame:
    """Check 5's draws: both cells at study conditions."""
    payloads = [
        ((curvature, cell, replicate),)
        for cell in ("both_wrong", "outcome_correct")
        for replicate in range(replicates)
    ]
    return pd.DataFrame(map_parallel(_pilot_fit, payloads, n_jobs=n_jobs))


def pass_probabilities(rows: pd.DataFrame, truth: float) -> dict[str, float]:
    """The pilot's estimate of each cell's pass probability at :data:`DR_REPLICATES`.

    A control passes when its bias interval lies outside the margin, so its bias must exceed
    ``margin + z sd / sqrt(R)``.  A positive cell passes when its interval lies inside, so
    ``|bias|`` must stay below ``margin - z sd / sqrt(R)``.  The margin is
    :data:`BIAS_MARGIN_SD` empirical SDs.
    """
    out: dict[str, float] = {}
    for cell in ("both_wrong", "outcome_correct"):
        values = rows.loc[rows["cell"] == cell, "estimate"].to_numpy(float)
        bias = float(values.mean() - truth)
        sd = float(values.std(ddof=1))
        error = sd / np.sqrt(values.size)
        half = LEVEL_Z * sd / np.sqrt(DR_REPLICATES)
        if cell == "both_wrong":
            probability = norm.cdf((abs(bias) - BIAS_MARGIN_SD * sd - half) / error)
        else:
            probability = norm.cdf((BIAS_MARGIN_SD * sd - half - abs(bias)) / error)
        out[f"{cell}_bias"] = bias
        out[f"{cell}_sd"] = sd
        out[f"{cell}_pass_probability"] = float(probability)
    return out


def assess(curvature: float, *, n_jobs: int = 1) -> tuple[dict[str, Any], pd.DataFrame]:
    """Every point value at one candidate coefficient, and checks 1 to 3."""
    law = law_for(curvature)
    limits = limit_biases(curvature, n_jobs=n_jobs)
    rows = pilot_rows(curvature, n_jobs=n_jobs)
    record: dict[str, Any] = {
        "curvature": curvature,
        "kl": projection_kl(lambda w: law.mechanism(w)[:, 1]),
        "first_run_kl": projection_kl(first_run_mechanism),
        "truncation": truncation_share(law),
        **limits,
        **pass_probabilities(rows, law.truth()["ate"]),
    }
    sigma = limits["sigma"]
    record["check_1"] = bool(record["kl"] >= KL_FLOOR)
    record["check_2"] = bool(record["truncation"] <= DR_TRUNCATION_CEILING)
    record["check_3"] = bool(abs(limits["tmle_limit_bias"]) >= TMLE_LIMIT_FLOOR * sigma)
    return record, rows


def _weighted_quadratic(
    x: np.ndarray, y: np.ndarray, error: np.ndarray
) -> Callable[[float], tuple[float, float]]:
    """The weighted least-squares quadratic of ``y`` on ``x``: its value and standard error.

    The covariance is inflated by ``chi^2 / df`` when that ratio exceeds 1, so a lack of fit
    widens the error rather than hiding behind the Monte Carlo weights.
    """
    design = np.column_stack([np.ones_like(x), x, x**2])
    weights = 1.0 / error**2
    covariance = np.linalg.inv(design.T @ (design * weights[:, None]))
    coefficients = covariance @ design.T @ (weights * y)
    residual = y - design @ coefficients
    dof = x.size - design.shape[1]
    covariance = covariance * max(1.0, float(np.sum(weights * residual**2)) / dof)

    def curve(point: float) -> tuple[float, float]:
        row = np.array([1.0, point, point**2])
        return float(row @ coefficients), float(np.sqrt(row @ covariance @ row))

    return curve


def _pass_z(cell: str, bias: float, sd: float, error: float) -> float:
    """The z value of a cell's pass at :data:`DR_REPLICATES`, with ``error`` on the bias.

    A control passes when its study bias interval lies outside the margin, and a positive
    cell when it lies inside.  ``error`` counts the pilot's error on the bias and the study's
    own Monte Carlo noise ``sd / sqrt(R)``.
    """
    half = LEVEL_Z * sd / np.sqrt(DR_REPLICATES)
    if cell == "both_wrong":
        return (abs(bias) - BIAS_MARGIN_SD * sd - half) / error
    return (BIAS_MARGIN_SD * sd - half - abs(bias)) / error


def select(summary: pd.DataFrame) -> pd.DataFrame:
    """Add the trend values, checks 4 and 5, and ``admissible`` to the assessed grid.

    Check 4 reads unweighted lines of C-TMLE0's limit bias and of sigma.  Check 5 reads, per
    cell, a quadratic of the bias weighted by its Monte Carlo error ``sd / sqrt(400)`` and
    inflated by its lack of fit, and a line of the SD.  Each predictive pass probability
    puts the trend's error and the study's own noise ``sd / sqrt(R)`` in its denominator, and
    must reach :data:`PASS_PROBABILITY`.  The point pass probability at the same value, with
    the pilot's own error and the study's noise, must reach :data:`POINT_PASS_PROBABILITY`.
    """
    out = summary.copy()
    x = out["curvature"].to_numpy(float)
    ctmle = np.polyfit(x, out["ctmle_limit_bias"].to_numpy(float), 1)
    sigma = np.polyfit(x, out["sigma"].to_numpy(float), 1)
    out["trend_ctmle_limit_bias"] = np.polyval(ctmle, x)
    out["trend_sigma"] = np.polyval(sigma, x)
    passes = np.ones(x.size, dtype=bool)
    for cell in ("both_wrong", "outcome_correct"):
        sd = out[f"{cell}_sd"].to_numpy(float)
        point_bias = out[f"{cell}_bias"].to_numpy(float)
        bias = _weighted_quadratic(x, point_bias, sd / np.sqrt(PILOT_REPLICATES))
        spread = np.polyfit(x, sd, 1)
        values, probabilities, points = [], [], []
        for index, value in enumerate(x):
            mean, error = bias(value)
            width = float(np.polyval(spread, value))
            noise = width / np.sqrt(DR_REPLICATES)
            values.append(mean)
            probabilities.append(
                float(norm.cdf(_pass_z(cell, mean, width, float(np.hypot(error, noise)))))
            )
            own = float(
                np.hypot(sd[index] / np.sqrt(PILOT_REPLICATES), sd[index] / np.sqrt(DR_REPLICATES))
            )
            points.append(float(norm.cdf(_pass_z(cell, point_bias[index], sd[index], own))))
        out[f"trend_{cell}_bias"] = values
        out[f"trend_{cell}_pass_probability"] = probabilities
        out[f"point_{cell}_pass_probability"] = points
        passes &= (np.asarray(probabilities) >= PASS_PROBABILITY) & (
            np.asarray(points) >= POINT_PASS_PROBABILITY
        )
    out["check_4"] = out["trend_ctmle_limit_bias"].abs() >= CTMLE_LIMIT_FLOOR * out["trend_sigma"]
    out["check_5"] = passes
    out["admissible"] = out[[f"check_{index}" for index in range(1, 6)]].all(axis=1)
    return out


def search(*, n_jobs: int = STUDY_JOBS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assess every coefficient of the grid, then apply :func:`select`."""
    records, frames = [], []
    for curvature in CURVATURE_GRID:
        record, rows = assess(curvature, n_jobs=n_jobs)
        records.append(record)
        frames.append(rows)
        print(record, flush=True)
    return select(pd.DataFrame(records)), pd.concat(frames, ignore_index=True)


def chosen_curvature() -> float:
    """The coefficient the committed pilot chose."""
    frame = pd.read_csv(PILOT, float_precision="round_trip")
    passed = frame.loc[frame["admissible"].astype(bool), "curvature"]
    if passed.empty:
        raise ValueError("no coefficient of the committed pilot passes every check")
    return float(passed.min())


if __name__ == "__main__":  # pragma: no cover - the pre-declaration step, run by hand
    parser = argparse.ArgumentParser(description="Size the outcome_correct law.")
    parser.add_argument("--jobs", type=int, default=STUDY_JOBS)
    arguments = parser.parse_args()
    summary, replicate_rows = search(n_jobs=arguments.jobs)
    summary.to_csv(PILOT, index=False, float_format="%.17g", lineterminator="\n")
    replicate_rows.to_csv(
        PILOT_REPLICATE_ROWS, index=False, float_format="%.17g", lineterminator="\n"
    )
    print(summary.to_string())
