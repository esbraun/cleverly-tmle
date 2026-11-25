r"""The numpy emulation of hte3's EP learners, and the pilot of the comparison (plan 4.4).

**The emulation.**  hte3's two debiasing steps at
:data:`~tests.studies.ep_hte3.HTE3_COMMIT`, with the runner-local sieve final stage
(``Lrnr_sieve_ls`` in ``tests/canonical/hte3_ep/hte3_ep_runner.R``), written out in numpy:

- :func:`truncate_propensity` is hte3's ``truncate_propensity(truncation_method =
  "adaptive")`` (``R/utils.R:1156-1195``), which ``resolve_contrast_nuisances`` applies to
  each supplied propensity column.  Its two cutoffs come from R's ``optimize``, transcribed in
  :func:`brent_fmin` (R's ``Brent_fmin``, ``src/library/stats/src/optimize.c``), at
  ``optim``'s ``reltol``.
- :func:`sieve_basis` is ``Sieve::sieve_preprocess(X, basisN = k d + 1, interaction_order = 1,
  type = "cosine")`` of Sieve 2.1: each column rescaled by its training 1% and 99% quantiles
  (R's type 7), clipped to ``[0, 1]``, then the constant and ``cos(j pi x_c)`` for
  ``j = 1..k`` ordered by ``j`` and then by column (``create_index_matrix``).  Prediction
  reuses the training quantiles.
- :func:`cate_ep` is ``compute_cate_ep_dr_pseudo_outcome``: ridge of ``Y`` on
  ``(2A - 1) phi`` with offset ``mu_n(A, W)``, weights ``1 / pi_n(A | W)`` and lambda
  ``1e-8`` without an intercept, then least squares of ``mu_1^* - mu_0^*`` on the sieve.
- :func:`crr_ep` is ``Lrnr_crr_EP``'s ``get_pseudo_data``: a weighted logistic ridge per arm
  on the sieve with offset ``logit mu_n``, then the quasibinomial fit of
  ``mu_1^* / (mu_1^* + mu_0^*)`` with weights ``mu_1^* + mu_0^*``, read on the logit scale
  through ``bounded_qlogis``.

glmnet solves the ridge steps by coordinate descent to its default ``thresh = 1e-7``; the
emulation solves them exactly.  The R smoke reads how far apart that leaves the two curves
(``EMULATION_TOLERANCE``, ``1e-6`` per point), and the design test reads its committed rows.

**The pilot.**  A pilot, not a result: 30 replicates of every reading cell on the
``hte3_pilot`` stream of the parent's record (``stream_seed(parent.STUDY, STREAM,
group, r)``), disjoint from every draw the parent made and from the declared run's.  Each
parent group is drawn once per replicate and refitted once per learner its cells need; M1 runs
on E8's draws and M2 on C6's.  It writes :data:`PILOT_CSV`, one row per cell and statistic,
which :func:`tests.studies.ep_hte3.design_numbers` reads.  The pilot is written by::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \\
    NUMEXPR_NUM_THREADS=1 python -m tests.studies.ep_hte3_pilot --raw <scratch dir> --jobs J

Each replicate's records are JSON lines in ``<scratch dir>/<parent>__<group>.jsonl``; a rerun
skips what is there, and every run rewrites the CSV from all of it.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

from tests.studies import ep_hte3 as study

STREAM = study.PILOT_STREAM
REPLICATES = 30
PILOT_CSV = study.PILOT_CSV
COLUMNS = ("cell", "statistic", "replicates", "mean", "sd")

#: hte3's probability and positivity bounds (``bound_probability``, ``bound_positive``).
EPS = 1e-8
#: The ridge penalty of both glmnet debiasing fits.
RIDGE = 1e-8
#: ``optim``'s default ``reltol``, which ``method = "Brent"`` passes to ``optimize`` as ``tol``.
BRENT_TOL = math.sqrt(np.finfo(float).eps)


# ---------------------------------------------------------------- hte3's propensity truncation


def brent_fmin(f: Any, lower: float, upper: float, tol: float = BRENT_TOL) -> float:
    """R's ``Brent_fmin``: the minimizer of ``f`` on ``[lower, upper]`` that ``optimize`` returns."""
    c = (3.0 - math.sqrt(5.0)) * 0.5
    eps = math.sqrt(np.finfo(float).eps)
    a, b = lower, upper
    v = a + c * (b - a)
    w = x = v
    d = e = 0.0
    fx = f(x)
    fv = fw = fx
    tol3 = tol / 3.0
    while True:
        xm = (a + b) * 0.5
        tol1 = eps * abs(x) + tol3
        t2 = tol1 * 2.0
        if abs(x - xm) <= t2 - (b - a) * 0.5:
            break
        p = q = r = 0.0
        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = (q - r) * 2.0
            if q > 0.0:
                p = -p
            else:
                q = -q
            r = e
            e = d
        if abs(p) >= abs(q * 0.5 * r) or p <= q * (a - x) or p >= q * (b - x):
            e = (b - x) if x < xm else (a - x)
            d = c * e
        else:
            d = p / q
            u = x + d
            if u - a < t2 or b - u < t2:
                d = tol1 if x < xm else -tol1
        if abs(d) >= tol1:
            u = x + d
        elif d > 0.0:
            u = x + tol1
        else:
            u = x - tol1
        fu = f(u)
        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, fv = w, fw
                w, fw = u, fu
            elif fu <= fv or v in (x, w):
                v, fv = u, fu
    return x


def bound_probability(x: np.ndarray, eps: float = EPS) -> np.ndarray:
    return np.minimum(np.maximum(x, eps), 1.0 - eps)


def truncate_propensity(pi: np.ndarray, indicator: np.ndarray) -> np.ndarray:
    """hte3's adaptive truncation of one propensity column (``R/utils.R:1156-1195``)."""
    n = indicator.size
    treated = indicator.astype(float)
    pi = bound_probability(np.asarray(pi, dtype=float))
    lower = max(1.0 / max(n, 2), 1e-8)
    upper = min(0.5, 1.0 - lower)
    star = pi.copy()
    if lower < upper:

        def low_risk(cutoff: float) -> float:
            bounded = np.maximum(pi, cutoff)
            return float(np.mean(np.where(treated == 1, 1.0 / bounded, 0.0) ** 2 - 2.0 / bounded))

        star = np.maximum(star, brent_fmin(low_risk, lower, upper))
        first = star.copy()

        def high_risk(cutoff: float) -> float:
            bounded = np.minimum(first, 1.0 - cutoff)
            alpha = np.where(treated == 0, 1.0 / (1.0 - bounded), 0.0)
            return float(np.mean(alpha**2 - 2.0 / (1.0 - bounded)))

        star = np.minimum(star, 1.0 - brent_fmin(high_risk, lower, upper))
    star = np.maximum(star, lower)
    return np.minimum(star, 1.0 - lower)


# ---------------------------------------------------------------- Sieve's cosine basis


def sieve_index(dimension: int, k: int) -> np.ndarray:
    """Sieve 2.1's index matrix at ``basisN = k d + 1``, interaction order 1 (1-based)."""
    rows = [np.ones(dimension, dtype=int)]
    for frequency in range(2, k + 2):
        for column in range(dimension):
            row = np.ones(dimension, dtype=int)
            row[column] = frequency
            rows.append(row)
    return np.vstack(rows)


def sieve_norm(x: np.ndarray) -> np.ndarray:
    """The training quantiles Sieve's ``normalize_X`` rescales by: rows 0.01 and 0.99."""
    return np.quantile(np.asarray(x, dtype=float), [0.01, 0.99], axis=0)


def sieve_basis(x: np.ndarray, k: int, norm: np.ndarray) -> np.ndarray:
    """``Sieve::sieve_preprocess(...)$Phi`` with the given normalization."""
    x = np.asarray(x, dtype=float)
    scaled = np.clip((x - norm[0]) / (norm[1] - norm[0]), 0.0, 1.0)
    index = sieve_index(x.shape[1], k)
    phi = np.ones((x.shape[0], index.shape[0]))
    for j, row in enumerate(index):
        for column, value in enumerate(row):
            if value > 1:
                phi[:, j] *= np.cos((value - 1) * np.pi * scaled[:, column])
    return phi


# ---------------------------------------------------------------- the two EP learners


def _weights(export: study.Export) -> np.ndarray:
    treated = export.a == 1.0
    pi1 = truncate_propensity(export.pi1, treated)
    pi0 = truncate_propensity(export.pi0, ~treated)
    return np.where(treated, 1.0 / pi1, 1.0 / pi0)


def _logistic_ridge(
    design: np.ndarray, y: np.ndarray, offset: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """glmnet's binomial ridge objective at lambda :data:`RIDGE`, no intercept, by Newton."""
    w = weights / weights.sum()
    beta = np.zeros(design.shape[1])
    for _ in range(100):
        p = expit(offset + design @ beta)
        gradient = design.T @ (w * (p - y)) + RIDGE * beta
        hessian = design.T @ ((w * p * (1.0 - p))[:, None] * design) + RIDGE * np.eye(beta.size)
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if np.max(np.abs(step)) < 1e-13:
            break
    return beta


def _quasibinomial(design: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``glm.fit(family = quasibinomial(), intercept = FALSE)`` by Newton from zero."""
    beta = np.zeros(design.shape[1])
    for _ in range(100):
        p = expit(design @ beta)
        gradient = design.T @ (weights * (p - y))
        hessian = design.T @ ((weights * p * (1.0 - p))[:, None] * design)
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if np.max(np.abs(step)) < 1e-13:
            break
    return beta


def cate_ep(export: study.Export, points: np.ndarray) -> np.ndarray:
    """hte3's ``Lrnr_cate_EP(targeting_style = "dr")`` with the sieve final stage, at ``points``."""
    norm = sieve_norm(export.w)
    phi = sieve_basis(export.w, export.k, norm)
    treated = export.a == 1.0
    sign = np.where(treated, 1.0, -1.0)
    offset = np.where(treated, export.mu1, export.mu0)
    w = _weights(export)
    w = w / w.sum()
    design = phi * sign[:, None]
    gram = design.T @ (w[:, None] * design) + RIDGE * np.eye(phi.shape[1])
    beta = np.linalg.solve(gram, design.T @ (w * (export.y - offset)))
    pseudo = export.mu1 - export.mu0 + 2.0 * (phi @ beta)
    final = np.linalg.lstsq(phi, pseudo, rcond=None)[0]
    return np.asarray(sieve_basis(points, export.k, norm) @ final)


def crr_ep(export: study.Export, points: np.ndarray) -> np.ndarray:
    """hte3's ``Lrnr_crr_EP`` with the quasibinomial sieve final stage: the log CRR at ``points``."""
    norm = sieve_norm(export.w)
    phi = sieve_basis(export.w, export.k, norm)
    treated = export.a == 1.0
    mu1 = bound_probability(export.mu1)
    mu0 = bound_probability(export.mu0)
    offset = np.log(np.where(treated, mu1, mu0)) - np.log1p(-np.where(treated, mu1, mu0))
    design = np.hstack([phi * treated[:, None], phi * (~treated)[:, None]])
    beta = _logistic_ridge(design, export.y, offset, _weights(export))
    p = phi.shape[1]
    star1 = bound_probability(expit(np.log(mu1 / (1.0 - mu1)) + phi @ beta[:p]))
    star0 = bound_probability(expit(np.log(mu0 / (1.0 - mu0)) + phi @ beta[p:]))
    weights = np.maximum(star1 + star0, EPS)
    share = bound_probability(star1 / weights)
    final = _quasibinomial(phi, share, weights)
    probability = bound_probability(expit(sieve_basis(points, export.k, norm) @ final))
    return np.asarray(np.log(probability) - np.log1p(-probability))


def emulate(cell: study.Cell, export: study.Export, points: np.ndarray) -> np.ndarray:
    """hte3's EP curve of one cell's export at ``points``."""
    return cate_ep(export, points) if cell.scale == "difference" else crr_ep(export, points)


# ---------------------------------------------------------------- the pilot


def pilot_groups() -> list[tuple[str, str]]:
    """Each ``(parent, group)`` the reading cells draw, in cell order."""
    seen: list[tuple[str, str]] = []
    for cell in study.READING_CELLS:
        key = (cell.parent, cell.group)
        if key not in seen:
            seen.append(key)
    return seen


def cells_of(parent: str, group: str) -> list[study.Cell]:
    return [cell for cell in study.READING_CELLS if (cell.parent, cell.group) == (parent, group)]


def one(task: tuple[str, str, int]) -> list[dict[str, Any]]:
    """Every cell's record on one pilot draw of one parent group."""
    parent, group, replicate = task
    cells = cells_of(parent, group)
    frame = study.draw(cells[0], replicate, STREAM)
    points = study.ise_points(cells[0])
    records = []
    for cell in cells:
        export = study.python_phase(cell, frame, replicate)
        start = time.process_time()
        theta = emulate(cell, export, points)
        emulation_seconds = time.process_time() - start
        row = study.reduce(cell, export, theta)
        values = {
            "d": row["log_ise_cleverly"] - row["log_ise_hte3"],
            "log_ratio_cleverly": row["log_ise_cleverly"] - row["log_ise_oracle"],
            "log_ratio_hte3": row["log_ise_hte3"] - row["log_ise_oracle"],
            "t_minus_hte3": row["log_ise_t"] - row["log_ise_hte3"],
            "curve_distance": row["curve_distance"],
            "seconds": export.seconds,
            "emulation_seconds": emulation_seconds,
        }
        records.append({"cell": cell.label, "replicate": replicate, "values": values})
        for mutation in study.MUTATION_CELLS:
            if mutation.mutation_of != cell.label:
                continue
            mutated = export.mutated(mutation.label)
            theta_m = emulate(mutation, mutated, points)
            row_m = study.reduce(mutation, mutated, theta_m)
            records.append(
                {
                    "cell": mutation.label,
                    "replicate": replicate,
                    "values": {"log_ratio_hte3": row_m["log_ise_hte3"] - row_m["log_ise_oracle"]},
                }
            )
    return records


def raw_path(raw: Path, parent: str, group: str) -> Path:
    return raw / f"{parent}__{group}.jsonl"


def summarize(raw: Path) -> pd.DataFrame:
    """One row per cell and statistic; ``correlation`` holds the correlation of the two
    oracle comparisons in ``mean`` (its ``sd`` is empty)."""
    by_cell: dict[str, list[dict[str, float]]] = {}
    for path in sorted(raw.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            by_cell.setdefault(record["cell"], []).append(record["values"])
    out = []
    for cell in study.CELLS:
        if cell.label not in by_cell:
            continue
        frame = pd.DataFrame(by_cell[cell.label])
        for column in frame.columns:
            values = frame[column].dropna()
            out.append(
                (cell.label, column, len(values), float(values.mean()), float(values.std(ddof=1)))
            )
        if {"log_ratio_cleverly", "log_ratio_hte3"} <= set(frame.columns):
            correlation = float(frame["log_ratio_cleverly"].corr(frame["log_ratio_hte3"]))
            out.append((cell.label, "correlation", len(frame), correlation, math.nan))
    return pd.DataFrame(out, columns=list(COLUMNS))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--groups", nargs="*", default=None, help="parent__group labels")
    parser.add_argument("--csv", type=Path, default=PILOT_CSV)
    arguments = parser.parse_args()
    arguments.raw.mkdir(parents=True, exist_ok=True)
    tasks = []
    for parent, group in pilot_groups():
        if arguments.groups and f"{parent}__{group}" not in arguments.groups:
            continue
        path = raw_path(arguments.raw, parent, group)
        done = set()
        if path.exists():
            done = {json.loads(line)["replicate"] for line in path.read_text().splitlines()}
        tasks += [(parent, group, r) for r in range(arguments.replicates) if r not in done]
    # The d = 20 groups first, so the pool's tail is short.
    tasks.sort(key=lambda task: -len(study.law_of(cells_of(task[0], task[1])[0]).columns))
    started = time.time()
    with Pool(arguments.jobs, maxtasksperchild=4) as pool:
        for records in pool.imap_unordered(one, tasks):
            first = study.CELL_BY_LABEL[records[0]["cell"]]
            with raw_path(arguments.raw, first.parent, first.group).open("a") as handle:
                handle.write("".join(json.dumps(record) + "\n" for record in records))
            seconds = sum(r["values"].get("seconds", 0.0) for r in records)
            print(first.parent, first.group, records[0]["replicate"], round(seconds, 1), flush=True)
    summary = summarize(arguments.raw)
    summary.to_csv(arguments.csv, index=False, float_format="%.6g")
    print(f"wall seconds: {time.time() - started:.0f}", flush=True)
    if arguments.csv == PILOT_CSV:
        numbers = study.design_numbers()
        print(f"replicates: {numbers['replicates']}", flush=True)
        print(f"reported: {sorted(numbers['reported'])}", flush=True)
        print(f"family pass: {numbers['family']:.3f}", flush=True)
        print(f"M1 miss probability: {numbers['mutations']['M1']:.2e}", flush=True)
        print(
            f"declared run CPU-hours (Python phase): {numbers['cost']['cpu_hours']:.1f}", flush=True
        )


if __name__ == "__main__":
    main()
