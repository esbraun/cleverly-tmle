r"""The pilot and the sizing of ``few-cluster-fold-variance``, run before the declaration.

Each cell of the study is fitted on :data:`PILOT_REPLICATES` draws of its own ``"pilot"`` seed
stream, which no property or primary draw reads, through the study's own fit and arms
(:func:`tests.studies.few_cluster_fold_variance.draw_arms`). A cell whose plug-in pass
probability at its sized replication count is below 0.90 is extended to
:data:`EXTENDED_REPLICATES` draws on the same stream before the declaration (the calibration
rule of the plans guide, Fix-5 plan 6.6). The rows are committed as
``tests/canonical/few_cluster_fold_variance/pilot.csv.gz``.

The sizing functions come from plan X54's ``diag/sizing.py`` (rev 2, ``variance_ratio_sim``,
``band_pass``):

* :func:`variance_ratio_pass` is the closed form of the plan's simulation: the log variance ratio
  is normal with variance :math:`r/R + (2 + \kappa)/(R-1)`, where :math:`r` is the relative
  variance of the squared SE and :math:`\kappa` the excess kurtosis of the estimate, and the
  interval is the same delta-method interval. :func:`variance_ratio_sim`, the plan's simulation,
  is its witness in the design test.
* :func:`band_pass` is the exact probability that the Clopper-Pearson 99% interval of coverage
  lies inside the band.

Run with ``python -m tests.studies.few_cluster_fold_variance_pilot --output <file.csv.gz>``;
``--extend`` adds the extension draws of the named cells to an existing file.
"""

from __future__ import annotations

import argparse
import functools
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from scipy import stats

from cleverly.exceptions import CleverlyError
from tests.studies import few_cluster_fold_variance as study
from tests.studies.evidence.seeds import stream_seed

PILOT_REPLICATES: Final[int] = 400
EXTENDED_REPLICATES: Final[int] = 1_600
CONFIDENCE: Final[float] = study.STUDY.margins.confidence_level
VARIANCE_BAND: Final[tuple[float, float]] = (
    study.STUDY.margins.calibration_se_ratio[0] ** 2,
    study.STUDY.margins.calibration_se_ratio[1] ** 2,
)
COVERAGE_BAND: Final[tuple[float, float]] = study.STUDY.margins.calibration_coverage
NOMINAL: Final[float] = 1.0 - study.STUDY.margins.alpha
#: The declared replications of every cell (plan 5.3, rev 3.2).
REPLICATES: Final[int] = 8_000
#: A positive cell whose plug-in pass probability is below this is a predicted red.
CELL_FLOOR: Final[float] = 0.90
#: G1: the exact 99% lower end of all-draw coverage must be at least this.
G1_FLOOR: Final[float] = 0.90
#: The ceiling of the median paired IID/cluster SE ratio's 99% upper end.
IID_MEDIAN_CEILING: Final[float] = 0.95

ARM_COLUMNS: Final[tuple[str, ...]] = ("psi", "std_error", "withheld", "covered")


def pilot_seed(cell: str, replicate: int) -> int:
    """The seed of one pilot draw: the study's ``"pilot"`` stream."""
    return int(stream_seed(study.STUDY, "pilot", study.FAMILY, cell, replicate))


def draw_row(cell: study.Cell, replicate: int) -> dict[str, Any]:
    """One pilot draw's row: each arm's point, SE, ``withheld`` and ``covered``."""
    truth = study.law_truth(cell.law)[study.TARGETS[cell.law]]
    rng = np.random.default_rng(pilot_seed(cell.name, replicate))
    frame = study.draw(cell.law, cell.clusters, rng)
    row: dict[str, Any] = {"cell": cell.name, "replicate": replicate, "lost": ""}
    try:
        entry = study.draw_arms(frame, cell, truth)
    except (CleverlyError, ValueError, np.linalg.LinAlgError) as error:
        row["lost"] = f"{type(error).__name__}: {str(error)[:80]}"
        for arm in study.ARMS:
            row.update({f"{arm}_psi": np.nan, f"{arm}_std_error": np.nan})
            row.update({f"{arm}_withheld": "", f"{arm}_covered": 0})
        return row
    for arm in study.ARMS:
        for column in ARM_COLUMNS:
            row[f"{arm}_{column}"] = entry[arm][column]
    return row


def _task(job: tuple[str, int]) -> dict[str, Any]:
    name, replicate = job
    return draw_row(study.CELL_BY_NAME[name], replicate)


def run(jobs: Sequence[tuple[str, int]], workers: int) -> pd.DataFrame:
    """Every pilot draw of ``jobs``, in a process pool."""
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_task, jobs, chunksize=20))
    return pd.DataFrame(rows).sort_values(["cell", "replicate"], ignore_index=True)


def read(path: Path) -> pd.DataFrame:
    """The committed pilot rows, with any arm the pilot predates derived (:func:`derive_known_deflated`)."""
    frame = pd.read_csv(path)
    frame["lost"] = frame["lost"].fillna("")
    if "known_deflated_control_std_error" not in frame:
        frame = derive_known_deflated(frame)
    for arm in study.ARMS:
        frame[f"{arm}_withheld"] = frame[f"{arm}_withheld"].fillna("")
    return frame


def derive_known_deflated(frame: pd.DataFrame) -> pd.DataFrame:
    """``known_deflated_control`` from ``known_propensity``, as :func:`study.draw_arms` builds it.

    The arm was added after the pilot ran (plan rev 3.2). It is the known-propensity fit's point
    and SE times ``sqrt(DEFLATION)`` on the same ``t_{J-V}`` reference, so it is derived from the
    committed columns exactly: no draw is refitted.
    """
    out = frame.copy()
    factor = float(np.sqrt(study.DEFLATION))
    out["known_deflated_control_psi"] = out["known_propensity_psi"]
    out["known_deflated_control_std_error"] = out["known_propensity_std_error"] * factor
    out["known_deflated_control_withheld"] = out["known_propensity_withheld"]
    covered = np.zeros(len(out), dtype=int)
    for cell in study.CELLS:
        rows = (out["cell"] == cell.name).to_numpy()
        truth = study.law_truth(cell.law)[study.TARGETS[cell.law]]
        half = study.t_critical(cell.df) * out.loc[rows, "known_deflated_control_std_error"]
        error = (out.loc[rows, "known_deflated_control_psi"] - truth).abs()
        shown = out.loc[rows, "known_propensity_withheld"].fillna("") == ""
        covered[rows] = ((error <= half) & shown).astype(int).to_numpy()
    out["known_deflated_control_covered"] = covered
    return out


# ------------------------------------------------------------------------------ the readings


def shown(rows: pd.DataFrame, cell: str, arm: str = "t_reference") -> pd.DataFrame:
    """The kept draws of one cell on which ``arm`` shows its interval."""
    group = rows.loc[rows["cell"] == cell]
    return group.loc[(group["lost"] == "") & (group[f"{arm}_withheld"] == "")]


def ratio_of(group: pd.DataFrame, arm: str) -> float:
    """The variance ratio of one arm over the given draws."""
    se = group[f"{arm}_std_error"].to_numpy(dtype=float)
    return float(np.mean(se**2) / np.var(group[f"{arm}_psi"].to_numpy(dtype=float), ddof=1))


def shapes(rows: pd.DataFrame, cell: str, arm: str = "t_reference") -> dict[str, float]:
    """The pilot readings of one cell and positive arm that the sizing reads.

    ``relvar`` is the relative variance of the arm's squared SE and ``kurtosis`` the excess
    kurtosis of its estimate (0 when negative, as the plan's simulation reads it). ``ratio`` is
    the observed variance ratio, ``trimmed_ratio`` the same ratio without the largest 5% of
    squared SEs and ``largest_se`` the largest SE in SDs of the estimate (the tail readings of
    plan 2.4 item 2). ``coverage`` is the shown-fit coverage, ``all_draw`` the coverage over all
    draws with withheld and lost fits as misses (G1), ``draws`` the shown draws, and
    ``withheld`` and ``lost`` the shares of all draws. The control readings (``deflated_ratio``,
    ``pooled_ratio``, ``iid_median`` and ``iid_mean``, the median and the ratio of means of the
    paired IID/cluster SE ratio) read the estimated-propensity fit whatever ``arm`` is.
    """
    group = rows.loc[rows["cell"] == cell]
    kept = shown(rows, cell, arm)
    se = kept[f"{arm}_std_error"].to_numpy(dtype=float)
    se2 = se**2
    psi = kept[f"{arm}_psi"].to_numpy(dtype=float)
    spread = float(np.var(psi, ddof=1))
    trimmed = np.sort(se2)[: int(np.floor(0.95 * se2.size))]
    reference = shown(rows, cell)
    iid = reference["iid_t_control_std_error"].to_numpy(dtype=float)
    fold = reference["t_reference_std_error"].to_numpy(dtype=float)
    return {
        "training": float(study.CELL_BY_NAME[cell].training),
        "draws": float(len(kept)),
        "relvar": float(np.var(se2, ddof=1) / np.mean(se2) ** 2),
        "kurtosis": max(float(stats.kurtosis(psi)), 0.0),
        "ratio": float(np.mean(se2) / spread),
        "trimmed_ratio": float(np.mean(trimmed) / spread),
        "largest_se": float(se.max() / np.sqrt(spread)),
        "coverage": float(kept[f"{arm}_covered"].mean()),
        "all_draw": float(kept[f"{arm}_covered"].sum() / len(group)),
        "withheld": float((group[f"{arm}_withheld"] != "").mean()),
        "lost": float((group["lost"] != "").mean()),
        "pooled_ratio": ratio_of(reference, "pooled_centered"),
        "deflated_ratio": ratio_of(reference, "deflated_control"),
        "iid_median": float(np.median(iid / fold)),
        "iid_mean": float(iid.mean() / fold.mean()),
    }


def readings_of(rows: pd.DataFrame, arm: str) -> dict[str, dict[str, float]]:
    """:func:`shapes` of every cell for one positive arm."""
    return {cell.name: shapes(rows, cell.name, arm) for cell in study.CELLS}


# -------------------------------------------------------------------------------- the sizing


def _z() -> float:
    return float(stats.norm.ppf(1.0 - (1.0 - CONFIDENCE) / 2.0))


def log_ratio_sd(relvar: float, kurtosis: float, replicates: int) -> float:
    """The SD of the log variance ratio at ``replicates`` draws (the plan's delta method)."""
    return float(np.sqrt(relvar / replicates + (2.0 + kurtosis) / (replicates - 1)))


def variance_lower_pass(
    relvar: float, kurtosis: float, replicates: int, ratio: float = 1.0
) -> float:
    """P(the 99% variance-ratio interval's lower end is at least 0.8649) at a true ``ratio``."""
    s = log_ratio_sd(relvar, kurtosis, replicates)
    return float(stats.norm.cdf((np.log(ratio) - np.log(VARIANCE_BAND[0])) / s - _z()))


def below_band_power(relvar: float, kurtosis: float, replicates: int, ratio: float) -> float:
    """P(the 99% upper end lies below the band's lower end) at a true ``ratio``."""
    s = log_ratio_sd(relvar, kurtosis, replicates)
    return float(stats.norm.cdf((np.log(VARIANCE_BAND[0]) - _z() * s - np.log(ratio)) / s))


def variance_ratio_sim(
    k_df: float,
    kurt_se: float,
    replicates: int,
    kurt_psi: float = 0.0,
    true_ratio: float = 1.0,
    sims: int = 1_000,
    seed: int = 1,
) -> float:
    """Plan X54 ``diag/sizing.py:variance_ratio_sim``, read one-sided: the simulated pass rate.

    The estimate has excess kurtosis ``kurt_psi`` (a scaled Student t); the squared SE has
    relative variance ``2 / k_df + kurt_se``. Vectorised by blocks of draws; the interval is the
    plan's, with the sample kurtosis of the estimate in its log-ratio SE, and a draw passes when
    its lower end is at least the band's lower end (rev 3.2).
    """
    rng = np.random.default_rng(seed)
    k_eff = 1.0 / (1.0 / k_df + kurt_se / 2.0)
    nu = None if kurt_psi <= 0 else 4.0 + 6.0 / kurt_psi
    z, passed = _z(), 0
    for start in range(0, sims, 100):
        b = min(100, sims - start)
        if nu is None:
            psi = rng.standard_normal((b, replicates))
        else:
            psi = rng.standard_t(nu, (b, replicates)) / np.sqrt(nu / (nu - 2.0))
        s2 = true_ratio * rng.chisquare(k_eff, (b, replicates)) / k_eff
        mean = s2.mean(axis=1)
        ratio = mean / psi.var(axis=1, ddof=1)
        excess = stats.kurtosis(psi, axis=1)
        se = np.sqrt(
            s2.var(axis=1, ddof=1) / mean**2 / replicates + (2.0 + excess) / (replicates - 1)
        )
        passed += int(np.sum(ratio * np.exp(-z * se) >= VARIANCE_BAND[0]))
    return passed / sims


def _exact_ends(replicates: int) -> tuple[np.ndarray, np.ndarray]:
    k = np.arange(replicates + 1)
    tail = (1.0 - CONFIDENCE) / 2.0
    lower = np.where(k > 0, stats.beta.ppf(tail, np.maximum(k, 1), replicates - k + 1), 0.0)
    upper = np.where(
        k < replicates, stats.beta.ppf(1.0 - tail, k + 1, np.maximum(replicates - k, 1)), 1.0
    )
    return lower, upper


@functools.cache
def band_pass(coverage: float, replicates: int) -> float:
    """P(the exact 99% coverage interval of ``replicates`` draws lies inside the band)."""
    lower, upper = _exact_ends(replicates)
    ok = (lower >= COVERAGE_BAND[0]) & (upper <= COVERAGE_BAND[1])
    k = np.arange(replicates + 1)
    return float(stats.binom.pmf(k, replicates, coverage)[ok].sum())


@functools.cache
def g1_pass(all_draw: float, replicates: int) -> float:
    """P(the exact 99% lower end of all-draw coverage is at least 0.90) (G1)."""
    lower, _ = _exact_ends(replicates)
    k = np.arange(replicates + 1)
    return float(stats.binom.pmf(k, replicates, all_draw)[lower >= G1_FLOOR].sum())


def cell_pass(
    reading: Mapping[str, float],
    replicates: int,
    *,
    ratio: float = 1.0,
    coverage: float = NOMINAL,
    all_draw: float | None = None,
) -> float:
    """P(a positive cell passes its three gates) at ``replicates`` requested draws.

    The shown draws are the requested draws less the pilot's withheld and lost shares; G1 reads
    every requested draw. By default the cell is exactly calibrated: variance ratio 1, shown
    coverage 0.95 and an all-draw coverage of 0.95 times the shown share. The gates read the
    same draws, and their product is the figure the plan declares.
    """
    shown_share = 1.0 - reading["withheld"] - reading["lost"]
    draws = max(int(replicates * shown_share), 2)
    every = coverage * shown_share if all_draw is None else all_draw
    return (
        variance_lower_pass(reading["relvar"], reading["kurtosis"], draws, ratio)
        * band_pass(round(min(coverage, 0.9999), 4), draws)
        * g1_pass(round(every, 4), replicates)
    )


def plug_in_pass(reading: Mapping[str, float], replicates: int = REPLICATES) -> float:
    """The cell pass probability at the pilot's own variance ratio and coverages."""
    return cell_pass(
        reading,
        replicates,
        ratio=reading["ratio"],
        coverage=reading["coverage"],
        all_draw=reading["all_draw"],
    )


def floor_product(
    readings: Mapping[str, Mapping[str, float]], floor: int, replicates: int = REPLICATES
) -> float:
    """The plug-in pass probability of every cell with at least ``floor`` training clusters."""
    return float(
        np.prod([plug_in_pass(r, replicates) for r in readings.values() if r["training"] >= floor])
    )


def predicted_red(
    readings: Mapping[str, Mapping[str, float]], replicates: int = REPLICATES
) -> frozenset[str]:
    """The cells whose plug-in pass probability is below 0.90: predicted reds."""
    return frozenset(
        cell for cell, r in readings.items() if plug_in_pass(r, replicates) < CELL_FLOOR
    )


def predicted_floor(
    readings: Mapping[str, Mapping[str, float]], replicates: int = REPLICATES
) -> int | None:
    """The smallest training count with no predicted red at or above it (plan 5.3's rule)."""
    red = predicted_red(readings, replicates)
    for floor in study.TRAINING_LADDER:
        if not any(readings[c]["training"] >= floor for c in red):
            return floor
    return None


def iid_median_power(rows: pd.DataFrame, cell: str, replicates: int, *, boot: int = 2_000) -> float:
    """P(the median paired IID/cluster SE ratio's 99% upper end is below 0.95), plug-in.

    The median's SD is the pilot's bootstrap SD scaled by the square root of the draw ratio.
    """
    kept = shown(rows, cell)
    ratio = kept["iid_t_control_std_error"].to_numpy(dtype=float) / kept[
        "t_reference_std_error"
    ].to_numpy(dtype=float)
    rng = np.random.default_rng(7)
    index = rng.integers(0, ratio.size, size=(boot, ratio.size))
    sd = float(np.std(np.median(ratio[index], axis=1), ddof=1)) * np.sqrt(ratio.size / replicates)
    point = float(np.median(ratio))
    return float(stats.norm.cdf((IID_MEDIAN_CEILING - _z() * sd - point) / sd))


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - the pilot's driver
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--extend", nargs="*", default=None)
    parser.add_argument("--refit-lost", action="store_true")
    args = parser.parse_args(argv)
    if args.refit_lost:
        # The draws a refusal lost, refitted on their own seeds after a refusal changed; every
        # other draw's fit is unchanged, because the refusal is a check made before any learner.
        existing = read(args.output)
        lost = existing["lost"] != ""
        jobs = list(
            zip(
                existing.loc[lost, "cell"], existing.loc[lost, "replicate"].astype(int), strict=True
            )
        )
        refitted = run(jobs, args.workers)
        combined = pd.concat([existing.loc[~lost], refitted], ignore_index=True)
        combined.sort_values(["cell", "replicate"], ignore_index=True).to_csv(
            args.output, index=False
        )
        return
    if args.extend is None:
        jobs = [(cell.name, r) for cell in study.CELLS for r in range(PILOT_REPLICATES)]
        run(jobs, args.workers).to_csv(args.output, index=False)
        return
    existing = read(args.output)
    jobs = [(name, r) for name in args.extend for r in range(PILOT_REPLICATES, EXTENDED_REPLICATES)]
    added = run(jobs, args.workers)
    combined = pd.concat([existing, added], ignore_index=True)
    combined.sort_values(["cell", "replicate"], ignore_index=True).to_csv(args.output, index=False)


if __name__ == "__main__":  # pragma: no cover
    main()
