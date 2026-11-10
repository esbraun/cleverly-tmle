"""The oracle-score design computation of the ``targeting-curves`` study (review finding 2).

A design computation, not a result: it measures the half-sample bootstrap's own
finite-sample SE ratio for every declared calibration row, with no nuisance fit.  Each
replicate draws the row's cell law at a size ``n``, builds the doubly robust score from the
**true** outcome regression and mechanism,

.. math:: \\Gamma = \\bar Q_1 - \\bar Q_0 + A (Y - \\bar Q_1)/g - (1 - A)(Y - \\bar Q_0)/(1 - g),

ranks the cell's score, and reads each quantity's point and its half-sample SE exactly as the
shipped estimator does (:func:`cleverly.targeting_curves.half_sample_draws`).  The ratio of
the mean SE to the SD of the errors is then the bootstrap's own calibration, which no amount of
nuisance accuracy can move.  The study's fits carry it into every calibration row, so a ratio
the bootstrap cannot reach at the declared ``n`` is declared before the pilot rather than read
off a red row: the summary sets
:data:`~tests.studies.targeting_curves.REPORTING_CELLS` and
:data:`~tests.studies.targeting_curves.R_MAIN`, which the design test recomputes from it.

The replicate counts were set after looking.  A first pass of 6,000 replicates per cell read
cell ``H``'s AUTOC at 0.999 +- 0.009 against the first review's 0.970 +- 0.011 on its own
seeds, a gap of about 2.7 SE whose cause is not identified (the review's same-draw oracle at
R = 1,000 read 0.988), and cell ``B``'s TOC(0.9) at 1.038.  ``H`` and ``B`` were then extended,
and after the second review every other cell too (:data:`CELL_REPLICATES`), so that no cell's
count depends on which way it read.  Every replicate of a cell enters its summary row; the CSV
holds one row per quantity.

Cell ``S`` ranks by ``W1``, the limit of its training refit on ``H``'s law (the projection of
``tau = 1 + W1`` on an additive basis), whose truths are closed form: ``TOC(q) = 1 - q``,
AUTOC ``1/2`` and Qini ``1/6``.

The committed summary is ``targeting_curves_oracle.csv`` beside this module, written by::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \\
    NUMEXPR_NUM_THREADS=1 python -m tests.studies.targeting_curves_oracle --jobs "$(nproc --all)"
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from cleverly import targeting_curves as tc
from tests.studies import targeting_curves as study
from tests.studies.evidence.seeds import stream_seed

#: The stream the computation draws from, apart from the pilot's and the run's.
STREAM = "targeting_oracle"

#: Replicates per (cell, n), and half-sample draws per replicate.  250 draws bias the mean SE
#: low by about ``1 / (4 * 250) = 0.1%``, which the ratios carry; 6,000 replicates fix a
#: ratio's own SE near 0.009.
REPLICATES = 6_000
DRAWS = 250

#: Each cell at its declared size is measured on more replicates than :data:`REPLICATES`, as
#: a run of the same stream (replicates ``0`` to ``R - 1``): ``H`` on 30,000, every other cell
#: on 18,000.  The module docstring records the order in which these were set.  The larger
#: sizes of cell ``H`` keep :data:`REPLICATES`.
CELL_REPLICATES: Mapping[tuple[str, int], int] = {
    ("H", 2_000): 30_000,
    **{(label, 2_000): 18_000 for label in ("B", "K", "U", "N", "D", "S")},
}

#: The sizes beyond its declared ``n`` at which cell ``H``'s rows are remeasured, to see
#: whether a departure from 1 closes with ``n``.  ``H`` stands for every continuous-score cell: the
#: rank term that drives the shortfall depends on the score's law, not on the outcome's.
LARGER = (4_000, 8_000)

#: The committed summary.
ORACLE_CSV = Path(__file__).with_name("targeting_curves_oracle.csv")

#: The group-effect edges of cell ``H`` (``groups=5``).
EDGES = (0.2, 0.4, 0.6, 0.8, 1.0)

#: Cell ``S``'s score limit and its closed-form truths.
S_TRUTH = {
    **{study.row_key("toc", q): 1.0 - q for q in study.GRID if q < 1.0},
    "autoc": 0.5,
    "qini": 1.0 / 6.0,
}


def _law(label: str) -> study.Law:
    return study.LAWS["H" if label == "S" else study.GROUP_BY_LABEL[label].law]


def truths(label: str) -> dict[str, float]:
    """The population truth of every calibration row of a cell under its score limit."""
    if label == "S":
        return dict(S_TRUTH)
    out = study.truth(_law(label))
    if label == "H":
        out.update(study.paired_truth())
    return out


def _rows(label: str, n: int) -> tuple[str, ...]:
    """The rows measured at ``n``: every calibration row of the cell, at any size."""
    del n
    return study.GROUP_BY_LABEL[label].rows


def _draw(label: str, n: int, seed: int, paired: bool) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """The scores and the oracle doubly robust score of one replicate."""
    law = _law(label)
    rng = np.random.default_rng(seed)
    w = rng.uniform(-1.0, 1.0, (n, 3))
    g = study.propensity(w)
    a = rng.binomial(1, g)
    if law.binary:
        q1, q0 = study.binary_mean(w, 1), study.binary_mean(w, 0)
        y = rng.binomial(1, np.where(a == 1, q1, q0)).astype(float)
    else:
        q0 = law.base(w)
        q1 = q0 + law.tau(w)
        y = q0 + a * law.tau(w) + law.sigma * rng.normal(size=n)
    gamma = q1 - q0 + a / g * (y - q1) - (1 - a) / (1 - g) * (y - q0)
    if label == "S":
        scores = {"S": w[:, 0].copy()}
    else:
        scores = {"S": law.score_of(w)}
        if paired:
            scores["S_B"] = w[:, 0] + study.C_SECOND * w[:, 2]
    return scores, gamma


def one(key: tuple[str, int, int]) -> dict[str, tuple[float, float]]:
    """Each calibration row's point and half-sample SE in one replicate."""
    label, n, replicate = key
    seed = stream_seed(study.STUDY, STREAM, f"{label}:{n}", replicate)
    rows = _rows(label, n)
    paired = any(name.startswith(f"{study.DIFFERENCE}:") for name in rows)
    scores, gamma = _draw(label, n, seed, paired)
    edges = EDGES if any(name.startswith("group_effect") for name in rows) else ()
    evaluation = tc.TargetingEvaluation(
        scores=scores,
        targeted=gamma,
        one_step=gamma,
        mass=np.ones(n),
        units=np.arange(n),
        grid=tc.DEFAULT_GRID,
        edges=edges,
    )
    draws = tc.half_sample_draws(evaluation, n_half_samples=DRAWS, random_state=replicate)
    points = {
        name: tc._point_values(evaluation.layout(name), gamma, np.ones(n), tc.DEFAULT_GRID, edges)
        for name in scores
    }
    out = {}
    for name in rows:
        key_ = study._parse(name)
        if name.startswith(f"{study.DIFFERENCE}:"):
            point = points["S"][key_] - points["S_B"][key_]
            spread = draws["S"][key_] - draws["S_B"][key_]
        else:
            point, spread = points["S"][key_], draws["S"][key_]
        out[name] = (float(point), float(np.std(spread, ddof=1)))
    return out


def summarize(label: str, n: int, records: list[Mapping[str, tuple[float, float]]]) -> list[dict]:
    """One row per calibration quantity: the SE ratio with its SE, coverage, bias, kurtosis."""
    truth = truths(label)
    rows = []
    for name in records[0]:
        estimate = np.array([record[name][0] for record in records])
        se = np.array([record[name][1] for record in records])
        errors = estimate - truth[name]
        sd = float(np.std(errors, ddof=1))
        ratio = float(np.mean(se) / sd)
        kurtosis = float(stats.kurtosis(errors, fisher=False))
        cv = float(np.std(se, ddof=1) / np.mean(se))
        ratio_se = ratio * math.sqrt(cv**2 / len(errors) + (kurtosis - 1.0) / (4.0 * len(errors)))
        rows.append(
            {
                "group": label,
                "n": n,
                "name": name,
                "replicates": len(records),
                "se_ratio": ratio,
                "se_ratio_se": ratio_se,
                "coverage": float(np.mean(np.abs(errors) <= study.CRITICAL * se)),
                "standardized_bias": float(np.mean(errors) / sd),
                "kurtosis": kurtosis,
            }
        )
    return rows


def plan(replicates: int = REPLICATES) -> list[tuple[str, int, int]]:
    """Every ``(cell, n, R)``: each calibration cell at its declared ``n``, then cell ``H`` at
    :data:`LARGER`, at ``replicates`` each unless :data:`CELL_REPLICATES` names more."""
    out = [(group.label, group.n) for group in study.GROUPS if group.rows]
    out += [("H", n) for n in LARGER]
    return [(label, n, CELL_REPLICATES.get((label, n), replicates)) for label, n in out]


def run(
    jobs: int, replicates: int = REPLICATES, cells: list[tuple[str, int, int]] | None = None
) -> pd.DataFrame:
    """Measure every ``(cell, n, R)`` of :func:`plan`."""
    rows: list[dict[str, Any]] = []
    with Pool(jobs) as pool:
        for label, n, count in cells or plan(replicates):
            records = pool.map(one, [(label, n, r) for r in range(count)], chunksize=50)
            rows += summarize(label, n, records)
            print(label, n, flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--output", type=Path, default=ORACLE_CSV)
    args = parser.parse_args()
    frame = run(args.jobs, args.replicates)
    frame.to_csv(args.output, index=False, float_format="%.6g")


if __name__ == "__main__":
    main()
