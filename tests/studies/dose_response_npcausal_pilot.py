r"""The sizing pilot of ``dose-response-npcausal`` (X93 S2), on its own seed streams.

It draws no declared sample: draw ``r`` takes ``stream_seed(STUDY, "pilot", "primary", r)``, a
stream no declared cell reads.  Each draw is fitted twice: by the declared subject fit, and by
:func:`ctseff_sketch`, a Python transcription of ``npcausal::ctseff`` at commit ``56a5ac1``.
R cannot run where this pilot ran (its package downloads are blocked there), so the transcription
stands in for the reference's spread, its correlation with the subject and its standard-error
calibration, which the paired legs read.  It is not the comparator, and nothing published reads
it.  It differs from ``ctseff`` in three declared places:

* the point estimate is the exact local-linear fit at each dose, where ``ctseff`` interpolates
  a binned fit (``KernSmooth::locpoly`` on 401 points);
* ``smooth.spline`` is :func:`scipy.interpolate.make_smoothing_spline` with its GCV penalty;
* ``density`` is an exact Gaussian kernel sum with R's ``bw.nrd0`` bandwidth, on R's 512-point
  grid from ``min - 3 bw`` to ``max + 3 bw``, where R bins and convolves by FFT.

It keeps ``ctseff``'s pairing of the two density grids: the density of the standardized
residuals is ``approx(density(all)$x, density(observed)$y, xout = all)``, the ``x`` grid of
every row's density with the ``y`` values of the observed rows' density.

It writes one row per draw, implementation and estimand to ``dose_response_npcausal_pilot.csv.gz``
beside this module.  Run it as ``python -m tests.studies.dose_response_npcausal_pilot --jobs N``.
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import make_smoothing_spline
from scipy.stats import norm

from cleverly.utils.parallel import map_parallel
from tests.studies import dose_response_npcausal as study
from tests.studies import dose_response_npcausal_law as law
from tests.studies.evidence.seeds import stream_seed

OUTPUT = Path(__file__).with_name("dose_response_npcausal_pilot.csv.gz")
DRAWS = 1_000
SKETCH = "ctseff-sketch"


def _interactions(columns: np.ndarray) -> np.ndarray:
    """``[1, x, x_i x_j (i < j)]``: R's ``glm(Y ~ .^2)`` design."""
    p = columns.shape[1]
    products = [columns[:, i] * columns[:, j] for i in range(p) for j in range(i + 1, p)]
    return np.column_stack([np.ones(len(columns)), columns, *products])


def _ols(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(design, target, rcond=None)[0]


def _nrd0(x: np.ndarray) -> float:
    spread = min(float(np.std(x, ddof=1)), float(np.subtract(*np.percentile(x, [75, 25]))) / 1.34)
    return 0.9 * spread * len(x) ** -0.2


def _density_grid(x: np.ndarray) -> tuple[np.ndarray, float]:
    bw = _nrd0(x)
    return np.linspace(x.min() - 3.0 * bw, x.max() + 3.0 * bw, 512), bw


def _kde(points: np.ndarray, data: np.ndarray, bw: float) -> np.ndarray:
    out = np.empty(len(points))
    for start in range(0, len(points), 64):
        block = points[start : start + 64]
        out[start : start + 64] = norm.pdf((block[:, None] - data[None, :]) / bw).mean(axis=1) / bw
    return out


def ctseff_sketch(frame: pd.DataFrame, bandwidth: float, doses: tuple[float, ...]) -> pd.DataFrame:
    """``ctseff``'s estimate and standard error at ``doses``, transcribed (see the module docstring)."""
    w = frame[list(study.COVARIATES)].to_numpy()
    a = frame["A"].to_numpy()
    y = frame["Y"].to_numpy()
    n = len(a)
    grid = study.npcausal_grid(a)
    k = len(grid)
    design_w = _interactions(w)
    pi = design_w @ _ols(design_w, a)
    pi2 = design_w @ _ols(design_w, (a - pi) ** 2)
    observed = _interactions(np.column_stack([w, a]))
    coefficients = _ols(observed, y)
    mu = observed @ coefficients
    mu_grid = np.column_stack(
        [_interactions(np.column_stack([w, np.full(n, value)])) @ coefficients for value in grid]
    )
    scale = np.sqrt(pi2)
    standard = np.concatenate(
        [(a - pi) / scale, ((grid[None, :] - pi[:, None]) / scale[:, None]).T.ravel()]
    )
    x_all, _ = _density_grid(standard)
    x_obs, bw_obs = _density_grid(standard[:n])
    y_obs = _kde(x_obs, standard[:n], bw_obs)
    pi_hat = np.interp(standard, x_all, y_obs)
    pi_obs = pi_hat[:n] / scale
    pi_grid = (pi_hat[n:].reshape(k, n).T) / scale[:, None]
    marginal = pi_grid.mean(axis=0)
    m_grid = mu_grid.mean(axis=0)
    varpi = make_smoothing_spline(grid, marginal)(a)
    m_obs = make_smoothing_spline(grid, m_grid)(a)
    pseudo = (y - mu) / (pi_obs / varpi) + m_obs
    steps = np.diff(grid)
    rows = []
    for dose in doses:
        u = (a - dose) / bandwidth
        weight = norm.pdf(u) / bandwidth
        design = np.column_stack([np.ones(n), u])
        beta = np.linalg.solve(design.T @ (design * weight[:, None]), design.T @ (weight * pseudo))
        dh = np.array(
            [
                [weight.mean(), (weight * u).mean()],
                [(weight * u).mean(), (weight * u * u).mean()],
            ]
        )
        g = (grid - dose) / bandwidth
        kernel_grid = norm.pdf(g) / bandwidth
        integrand1 = kernel_grid[None, :] * (mu_grid - m_grid[None, :]) * marginal[None, :]
        integrand2 = g[None, :] * integrand1
        int1 = (steps[None, :] * (integrand1[:, 1:] + integrand1[:, :-1]) / 2.0).sum(axis=1)
        int2 = (steps[None, :] * (integrand2[:, 1:] + integrand2[:, :-1]) / 2.0).sum(axis=1)
        residual = pseudo - beta[0] - beta[1] * u
        scores = np.linalg.solve(
            dh, np.vstack([weight * residual + int1, u * weight * residual + int2])
        )
        se = float(np.sqrt(np.cov(scores)[0, 0]))
        rows.append({"dose": dose, "estimate": float(beta[0]), "se": se / np.sqrt(n)})
    return pd.DataFrame(rows)


def _one(replicate: int) -> list[dict[str, Any]]:
    frame = law.sample(study.PRIMARY_N, stream_seed(study.STUDY, "pilot", "primary", replicate))
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result, initial = study.fit_with_initials(frame)
    except Exception as error:  # a refusal is a pilot finding, recorded with its class
        rows.append(
            {
                "replicate": replicate,
                "implementation": study.IMPLEMENTATION,
                "refusal": type(error).__name__,
            }
        )
    else:
        seconds = time.perf_counter() - start
        for name in study.ESTIMANDS:
            estimate = result[name]
            low, high = estimate.ci
            truth = study.TRUTH[name]
            rows.append(
                {
                    "replicate": replicate,
                    "implementation": study.IMPLEMENTATION,
                    "estimand": name,
                    "truth": truth,
                    "estimate": float(estimate.psi),
                    "initial_estimate": float(initial[name]),
                    "std_error": float(estimate.std_error),
                    "covered": int(low <= truth <= high),
                    "seconds": seconds,
                    "refusal": "",
                }
            )
    for h in study.BANDWIDTHS:
        start = time.perf_counter()
        sketch = ctseff_sketch(frame, study.NPCAUSAL_BANDWIDTHS[h], study.DOSES)
        seconds = time.perf_counter() - start
        for row in sketch.itertuples(index=False):
            name = next(
                key for key, (a0, hh, _) in study.PAIRS.items() if a0 == row.dose and hh == h
            )
            truth = study.TRUTH[name]
            low, high = row.estimate - 1.96 * row.se, row.estimate + 1.96 * row.se
            rows.append(
                {
                    "replicate": replicate,
                    "implementation": SKETCH,
                    "estimand": name,
                    "truth": truth,
                    "estimate": float(row.estimate),
                    "initial_estimate": float("nan"),
                    "std_error": float(row.se),
                    "covered": int(low <= truth <= high),
                    "seconds": seconds,
                    "refusal": "",
                }
            )
    return rows


def run(n_jobs: int, draws: int = DRAWS) -> pd.DataFrame:
    """Fit ``draws`` pilot draws and write every row to :data:`OUTPUT`."""
    outcomes = map_parallel(_one, [(r,) for r in range(draws)], n_jobs=n_jobs)
    frame = pd.DataFrame([row for rows in outcomes for row in rows])
    frame.to_csv(OUTPUT, index=False)
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--draws", type=int, default=DRAWS)
    arguments = parser.parse_args()
    run(arguments.jobs, arguments.draws)
