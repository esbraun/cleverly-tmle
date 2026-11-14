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
#: The replication ladder of plan 5.3: multiples of 400 from 4,000 to 8,000.
LADDER: Final[tuple[int, ...]] = tuple(range(4_000, 8_001, 400))
CELL_FLOOR: Final[float] = 0.90
FAMILY_FLOOR: Final[float] = 0.95

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
    """The committed pilot rows."""
    frame = pd.read_csv(path)
    frame["lost"] = frame["lost"].fillna("")
    for arm in study.ARMS:
        frame[f"{arm}_withheld"] = frame[f"{arm}_withheld"].fillna("")
    return frame


# ------------------------------------------------------------------------------ the readings


def shown(rows: pd.DataFrame, cell: str, arm: str = "t_reference") -> pd.DataFrame:
    """The kept draws of one cell on which ``arm`` shows its interval."""
    group = rows.loc[rows["cell"] == cell]
    return group.loc[(group["lost"] == "") & (group[f"{arm}_withheld"] == "")]


def ratio_of(group: pd.DataFrame, arm: str) -> float:
    """The variance ratio of one arm over the given draws."""
    se = group[f"{arm}_std_error"].to_numpy(dtype=float)
    return float(np.mean(se**2) / np.var(group[f"{arm}_psi"].to_numpy(dtype=float), ddof=1))


def shapes(rows: pd.DataFrame, cell: str) -> dict[str, float]:
    """The pilot readings of one cell that the sizing reads.

    ``relvar`` is the relative variance of the positive arm's squared SE, ``kurtosis`` the
    excess kurtosis of its estimate (0 when negative, as the plan's simulation reads it),
    ``ratio`` the observed variance ratio, ``coverage`` the shown-fit coverage of the positive
    arm, ``draws`` its shown draws, ``withheld`` and ``lost`` the shares of all draws, and the
    other arms' variance ratios and the IID control's paired SE ratio beside them.
    """
    group = rows.loc[rows["cell"] == cell]
    kept = shown(rows, cell)
    se2 = kept["t_reference_std_error"].to_numpy(dtype=float) ** 2
    psi = kept["t_reference_psi"].to_numpy(dtype=float)
    known = shown(rows, cell, "known_propensity")
    return {
        "draws": float(len(kept)),
        "relvar": float(np.var(se2, ddof=1) / np.mean(se2) ** 2),
        "kurtosis": max(float(stats.kurtosis(psi)), 0.0),
        "ratio": ratio_of(kept, "t_reference"),
        "median_ratio": float(np.median(se2) / np.var(psi, ddof=1)),
        "coverage": float(kept["t_reference_covered"].mean()),
        "withheld": float((group["t_reference_withheld"] != "").mean()),
        "lost": float((group["lost"] != "").mean()),
        "pooled_ratio": ratio_of(kept, "pooled_centered"),
        "deflated_ratio": ratio_of(kept, "deflated_control"),
        "known_ratio": ratio_of(known, "known_propensity"),
        "known_coverage": float(known["known_propensity_covered"].mean()),
        "iid_paired": float(
            kept["iid_t_control_std_error"].mean() / kept["t_reference_std_error"].mean()
        ),
    }


# -------------------------------------------------------------------------------- the sizing


def _z() -> float:
    return float(stats.norm.ppf(1.0 - (1.0 - CONFIDENCE) / 2.0))


def log_ratio_sd(relvar: float, kurtosis: float, replicates: int) -> float:
    """The SD of the log variance ratio at ``replicates`` draws (the plan's delta method)."""
    return float(np.sqrt(relvar / replicates + (2.0 + kurtosis) / (replicates - 1)))


def variance_ratio_pass(
    relvar: float, kurtosis: float, replicates: int, ratio: float = 1.0
) -> float:
    """P(the 99% variance-ratio interval lies inside the band) at a true ``ratio``."""
    s = log_ratio_sd(relvar, kurtosis, replicates)
    low, high = np.log(VARIANCE_BAND)
    centre, z = float(np.log(ratio)), _z()
    upper = stats.norm.cdf((high - z * s - centre) / s)
    lower = stats.norm.cdf((low + z * s - centre) / s)
    return float(max(upper - lower, 0.0))


def below_band_power(relvar: float, kurtosis: float, replicates: int, ratio: float) -> float:
    """P(the 99% upper end lies below the band's lower end) at a true ``ratio``."""
    s = log_ratio_sd(relvar, kurtosis, replicates)
    return float(stats.norm.cdf((np.log(VARIANCE_BAND[0]) - _z() * s - np.log(ratio)) / s))


def above_band_power(relvar: float, kurtosis: float, replicates: int, ratio: float) -> float:
    """P(the 99% lower end lies above the band's upper end) at a true ``ratio``."""
    s = log_ratio_sd(relvar, kurtosis, replicates)
    return float(stats.norm.cdf((np.log(ratio) - _z() * s - np.log(VARIANCE_BAND[1])) / s))


def variance_ratio_sim(
    k_df: float,
    kurt_se: float,
    replicates: int,
    kurt_psi: float = 0.0,
    true_ratio: float = 1.0,
    sims: int = 1_000,
    seed: int = 1,
) -> float:
    """Plan X54 ``diag/sizing.py:variance_ratio_sim``: the simulated pass probability.

    The estimate has excess kurtosis ``kurt_psi`` (a scaled Student t); the squared SE has
    relative variance ``2 / k_df + kurt_se``. Vectorised by blocks of draws; the interval is the
    plan's, with the sample kurtosis of the estimate in its log-ratio SE.
    """
    rng = np.random.default_rng(seed)
    k_eff = 1.0 / (1.0 / k_df + kurt_se / 2.0)
    nu = None if kurt_psi <= 0 else 4.0 + 6.0 / kurt_psi
    z, inside = _z(), 0
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
        low, high = ratio * np.exp(-z * se), ratio * np.exp(z * se)
        inside += int(np.sum((low >= VARIANCE_BAND[0]) & (high <= VARIANCE_BAND[1])))
    return inside / sims


@functools.cache
def band_pass(coverage: float, replicates: int) -> float:
    """P(the exact 99% coverage interval of ``replicates`` draws lies inside the band)."""
    k = np.arange(replicates + 1)
    tail = (1.0 - CONFIDENCE) / 2.0
    lower = np.where(k > 0, stats.beta.ppf(tail, np.maximum(k, 1), replicates - k + 1), 0.0)
    upper = np.where(
        k < replicates, stats.beta.ppf(1.0 - tail, k + 1, np.maximum(replicates - k, 1)), 1.0
    )
    ok = (lower >= COVERAGE_BAND[0]) & (upper <= COVERAGE_BAND[1])
    return float(stats.binom.pmf(k, replicates, coverage)[ok].sum())


def cell_pass(
    reading: Mapping[str, float], replicates: int, *, ratio: float = 1.0, coverage: float = NOMINAL
) -> float:
    """P(the positive arm passes both clauses) at ``replicates`` requested draws.

    The shown draws are the requested draws less the pilot's withheld and lost shares. The two
    clauses read the same draws; their product is the figure the plan declares.
    """
    draws = max(int(replicates * (1.0 - reading["withheld"] - reading["lost"])), 2)
    return variance_ratio_pass(reading["relvar"], reading["kurtosis"], draws, ratio) * band_pass(
        coverage, draws
    )


def shortfall(readings: Mapping[str, Mapping[str, float]]) -> frozenset[str]:
    """The cells that cannot reach a pass probability of 0.90 at 8,000 even when calibrated.

    Their pilot spread of the squared SE leaves the variance-ratio interval wider than the band
    at any affordable R. Each is recorded as a predicted red at 8,000 before the declaration.
    """
    return frozenset(
        cell for cell, reading in readings.items() if cell_pass(reading, LADDER[-1]) < CELL_FLOOR
    )


def declared_replicates(readings: Mapping[str, Mapping[str, float]]) -> dict[str, int]:
    """Each cell's R under exact calibration (plan 5.3).

    A cell of :func:`shortfall` takes 8,000. Every other cell takes the smallest ladder value
    with a cell pass probability of at least 0.90; then, while the product over those cells is
    below 0.95, the one with the lowest pass probability steps up the ladder, to 8,000 at most.
    """
    short = shortfall(readings)
    sized: dict[str, int] = dict.fromkeys(short, LADDER[-1])
    for cell, reading in readings.items():
        if cell not in short:
            sized[cell] = next(r for r in LADDER if cell_pass(reading, r) >= CELL_FLOOR)
    kept = [cell for cell in readings if cell not in short]

    def family(values: Mapping[str, int]) -> float:
        return float(np.prod([cell_pass(readings[c], values[c]) for c in kept]))

    while family(sized) < FAMILY_FLOOR:
        open_cells = [c for c in kept if sized[c] < LADDER[-1]]
        if not open_cells:
            break
        worst = min(open_cells, key=lambda c: cell_pass(readings[c], sized[c]))
        sized[worst] = LADDER[LADDER.index(sized[worst]) + 1]
    return {cell: sized[cell] for cell in readings}


def family_pass(
    readings: Mapping[str, Mapping[str, float]],
    sized: Mapping[str, int],
    cells: Sequence[str] | None = None,
) -> float:
    """The product of the cell pass probabilities under exact calibration over ``cells``."""
    names = list(sized) if cells is None else list(cells)
    return float(np.prod([cell_pass(readings[c], sized[c]) for c in names]))


def plug_in_pass(reading: Mapping[str, float], replicates: int) -> float:
    """The cell pass probability at the pilot's observed variance ratio and coverage."""
    return cell_pass(reading, replicates, ratio=reading["ratio"], coverage=reading["coverage"])


def iid_power(rows: pd.DataFrame, cell: str, replicates: int, *, boot: int = 2_000) -> float:
    """P(the paired IID/cluster SE ratio's 99% upper end lies below 0.80) at the pilot ratio.

    The ratio's SD is the pilot's bootstrap SD scaled by the square root of the draw ratio.
    """
    kept = shown(rows, cell)
    iid = kept["iid_t_control_std_error"].to_numpy(dtype=float)
    fold = kept["t_reference_std_error"].to_numpy(dtype=float)
    rng = np.random.default_rng(7)
    index = rng.integers(0, len(kept), size=(boot, len(kept)))
    draws = iid[index].mean(axis=1) / fold[index].mean(axis=1)
    sd = float(np.std(draws, ddof=1)) * np.sqrt(len(kept) / replicates)
    point = float(iid.mean() / fold.mean())
    return float(stats.norm.cdf((0.80 - _z() * sd - point) / sd))


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - the pilot's driver
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--extend", nargs="*", default=None)
    args = parser.parse_args(argv)
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
