"""The pilot of ``refuter-calibration-continuous-survival``, on its own seed stream.

It draws no declared sample: every draw, fit and refutation of a pilot trial takes
``trial_seed(PILOT_STREAM, label, group, r)``, a stream no declared cell reads. It writes one
row per trial, estimand and reading to ``refuter_calibration_continuous_survival_pilot.csv.gz``
beside this module, with the error class of a trial that raised. Run it as
``python -m tests.studies.refuter_calibration_continuous_survival_pilot --jobs N``.

=========================  =====================================================================
group                      what it sizes
=========================  =====================================================================
``size``                   the eight size readings of a trial: each cell's rejection rate and
                           the correlation between cells, for the predictive family pass of the
                           design test
``power``                  the stale-row placebo on the power law: the within-trial spread of
                           the placebo refits and the spread of their trial means, which the
                           design test reads to compute the per-trial power of
                           :data:`~tests.studies.refuter_calibration_continuous_survival.STALE_SHARE`
``dependent_censoring``,   the reporting cells: that they run, and their time per trial
``imbalanced``
=========================  =====================================================================
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from cleverly.utils.parallel import map_parallel
from tests.studies import refuter_calibration_continuous_survival as study

OUTPUT = Path(__file__).with_name("refuter_calibration_continuous_survival_pilot.csv.gz")

#: Trials per pilot group.
PILOTS: dict[str, int] = {"size": 60, "power": 12, "dependent_censoring": 4, "imbalanced": 4}


def _one(group: str, trial: int) -> list[dict[str, Any]]:
    frame = study.draw_trial(group, trial, stream=study.PILOT_STREAM)
    base = {"group": group, "trial": trial}
    start = time.perf_counter()
    try:
        outcome, result, reports = study.trial_readings(
            group, frame, trial, stream=study.PILOT_STREAM
        )
    except Exception as error:  # a raise is a pilot finding, recorded with its class
        return [{**base, "refusal": type(error).__name__, "message": str(error)[:300]}]
    seconds = time.perf_counter() - start
    rows = []
    for key, name in study.ESTIMANDS.items():
        estimate = result[name]
        configuration = outcome.configurations[key]
        report = reports[key]
        placebo = np.asarray(report["placebo"].values, dtype=float)
        extra: dict[str, Any] = {
            "estimand": key,
            "psi": float(estimate.psi),
            "std_error": float(estimate.std_error),
            "placebo_mean": float(placebo.mean()),
            "placebo_sd": float(placebo.std(ddof=1)),
            "event_share": configuration["event_share"],
            "min_event_count": configuration["min_event_count"],
            "inference": configuration["inference"],
            "fits": outcome.fits,
            "seconds": seconds,
            "refusal": "",
        }
        for reading, value in outcome.readings.items():
            cell, kind = reading.split("__", 1)
            if cell == key:
                rows.append(
                    {
                        **base,
                        **extra,
                        "reading": kind,
                        "statistic": value.statistic,
                        "pvalue": value.pvalue,
                        "rejected": int(value.rejected),
                    }
                )
    return rows


def run(n_jobs: int, groups: list[str] | None = None, first: int = 0) -> pd.DataFrame:
    """Run trials ``first`` onwards of ``groups`` (every group by default); keep earlier rows."""
    chosen = groups or list(PILOTS)
    jobs = [(group, r) for group in chosen for r in range(first, PILOTS[group])]
    outcomes = map_parallel(_one, jobs, n_jobs=n_jobs)
    frame = pd.DataFrame([row for rows in outcomes for row in rows])
    if OUTPUT.exists():
        kept = pd.read_csv(OUTPUT, low_memory=False)
        keep = ~kept["group"].isin(chosen) | (kept["trial"] < first)
        frame = pd.concat([kept.loc[keep], frame], ignore_index=True)
    frame.to_csv(OUTPUT, index=False)
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--groups", nargs="*", default=None)
    parser.add_argument("--first", type=int, default=0)
    arguments = parser.parse_args()
    run(arguments.jobs, arguments.groups, arguments.first)


# ---- the design readings -------------------------------------------------------------------


def read(path: Path = OUTPUT) -> pd.DataFrame:
    """The committed pilot rows."""
    return pd.read_csv(path, low_memory=False, float_precision="round_trip")


def signed_scores(rows: pd.DataFrame) -> pd.DataFrame:
    """Each size reading of each pilot trial as a signed normal score, one column per cell.

    A placebo ``t`` maps through ``t_19`` and a subset ``rho`` through ``chi^2_19`` of
    ``19 rho^2``, each then through the standard normal quantile, so a reading rejects at 0.05
    exactly when its score's absolute value exceeds ``z_0.975``.
    """
    df = study.REFITS - 1
    size = rows.loc[(rows["group"] == "size") & rows["refusal"].fillna("").eq("")]
    size = size.loc[size["reading"].isin(study.SIZE_READINGS)].copy()
    placebo = size["reading"] == "placebo"
    probability = np.where(
        placebo,
        stats.t.cdf(size["statistic"], df),
        stats.chi2.cdf(df * size["statistic"] ** 2, df),
    )
    size["score"] = stats.norm.ppf(np.clip(probability, 1e-15, 1.0 - 1e-15))
    size["cell"] = size["estimand"] + "__" + size["reading"]
    table = size.pivot(index="trial", columns="cell", values="score")
    return table.loc[:, list(study.PROPERTY_CELLS[study.SIZE_FAMILY])]


def count_ceiling(trials: int, ceiling: float, level: float = 0.99) -> int:
    """The largest rejection count of ``trials`` whose exact upper endpoint is at most ``ceiling``."""
    from tests.studies.evidence.inference import clopper_pearson

    count = 0
    while clopper_pearson(count + 1, trials, confidence_level=level).high <= ceiling:
        count += 1
    return count


def family_pass(
    scores: pd.DataFrame,
    *,
    trials: int = study.SIZE_TRIALS,
    max_rejections: int = study.SIZE_MAX_REJECTIONS,
    draws: int = 4_000,
    seed: int = 2026111403,
) -> float:
    """The predictive probability that every size cell passes its count rule.

    Each cell's rejection rate has the Jeffreys posterior ``Beta(k + 1/2, m - k + 1/2)`` of
    its ``k`` pilot rejections in ``m`` trials. The posteriors are coupled, and the trials of
    the declared run are drawn, through a Gaussian copula with the correlation of the pilot's
    signed scores (:func:`signed_scores`): one draw of the rates sets each cell's threshold
    ``c`` with ``P(|Z| > c) = p``, and ``trials`` correlated normal vectors give each cell's
    count. The probability is the share of draws in which no count exceeds ``max_rejections``.
    """
    rng = np.random.default_rng(seed)
    values = scores.to_numpy(dtype=float)
    m, cells = values.shape
    critical = stats.norm.ppf(1.0 - study.LEVEL / 2.0)
    k = (np.abs(values) > critical).sum(axis=0)
    correlation = np.corrcoef(values, rowvar=False)
    # The nearest correlation matrix with eigenvalues at least 1e-6, for the Cholesky factor.
    eigenvalues, vectors = np.linalg.eigh(correlation)
    repaired = vectors @ np.diag(np.maximum(eigenvalues, 1e-6)) @ vectors.T
    scale = np.sqrt(np.diag(repaired))
    factor = np.linalg.cholesky(repaired / np.outer(scale, scale))
    passed = 0
    for _ in range(draws):
        coupling = stats.norm.cdf(factor @ rng.standard_normal(cells))
        rates = stats.beta.ppf(coupling, k + 0.5, m - k + 0.5)
        thresholds = stats.norm.ppf(1.0 - rates / 2.0)
        drawn = rng.standard_normal((trials, cells)) @ factor.T
        counts = (np.abs(drawn) > thresholds).sum(axis=0)
        passed += bool(np.all(counts <= max_rejections))
    return passed / draws


def stale_power(rows: pd.DataFrame, share: float = study.STALE_SHARE) -> dict[str, float]:
    """The per-trial power of the stale-row placebo on each estimand of the power law.

    The trial mean of the placebo refits is the exact shift of :func:`~tests.studies.refuter_calibration_continuous_survival.stale_shift`
    plus a trial effect of SD ``tau``; the refits scatter about it with SD ``s``. The pilot's
    power trials give ``s`` (the root mean square of the within-trial SD) and ``tau`` (from the
    spread of the trial means less ``s^2 / R``). Given the trial effect the ``t`` is noncentral
    ``t_19`` with noncentrality ``sqrt(R) (shift + tau z) / s``, and the power averages over
    ``z`` by Gauss-Hermite quadrature.
    """
    nodes, weights = np.polynomial.hermite_e.hermegauss(40)
    weights = weights / weights.sum()
    refits = study.REFITS
    critical = stats.t.ppf(1.0 - study.LEVEL / 2.0, refits - 1)
    shift = study.stale_shift(share)
    power = rows.loc[(rows["group"] == "power") & rows["refusal"].fillna("").eq("")]
    out = {}
    for key in study.ESTIMANDS:
        chosen = power.loc[power["estimand"] == key].drop_duplicates("trial")
        within = float(np.sqrt(np.mean(chosen["placebo_sd"] ** 2)))
        between = float(chosen["placebo_mean"].var(ddof=1)) - within**2 / refits
        tau = float(np.sqrt(max(between, 0.0)))
        # The two-sided power is even in the noncentrality. Past 8 it is 1 to seven digits,
        # and scipy's noncentral cdf returns nan near 9.5; the noncentrality is clipped at 8.
        noncentrality = np.minimum(
            np.abs(np.sqrt(refits) * (abs(shift[key]) + tau * nodes) / within), 8.0
        )
        each = stats.nct.sf(critical, refits - 1, noncentrality) + stats.nct.cdf(
            -critical, refits - 1, noncentrality
        )
        out[key] = float(weights @ each)
    return out
