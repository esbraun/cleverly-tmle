r"""The pilot of ``boundary-node-ltmle``, run before the declaration on its own seed stream.

Each cell of the study is fitted on 2,000 draws of :mod:`tests.studies.boundary_node_laws`, and
each draw writes one row per target with three arms on the same fit:

* ``supplemented``: the shipped interval, the curve's variance plus the boundary-node
  supplement (:mod:`cleverly.inference.node_variance`);
* ``plugin``: the same estimate at the curve's variance alone, the interval the package withheld
  before the supplement;
* ``oracle_node``: the curve's variance plus the flagged nodes' variance at the true hazard,
  :math:`n^{-2}\sum_u(\sum_{i \in u} |\lambda_r||C_i|\sigma_i)^2` with :math:`\sigma_i^2 = h(1-h)`
  from the law, in the same Minkowski form as the supplement.

Each row records whether node 4 of the target's firing regimen was flagged (``firing``:
``always`` for ``always`` and the contrast, ``never`` for ``never``) and whether another node was
too (``multi_flag``). The fit and the arms are the study's own
(:func:`tests.studies.boundary_node_ltmle.draw_arms`). :func:`summarize` reads the coverage of
each arm on all, firing and non-firing draws, and :func:`control_power` the two clauses of the
paired control at the pilot rates.

Run with ``python -m tests.studies.boundary_node_pilot --output <file.csv.gz>``. Law A's cells
fit the iterated-mean path; law B's (``held_n200``, ``held_zero_n200``) fit the hazard path,
which targets a held baseline treatment only.
"""

from __future__ import annotations

import argparse
import zlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from scipy import stats

from cleverly.exceptions import CleverlyError
from tests.studies import boundary_node_laws as laws
from tests.studies import boundary_node_ltmle as study
from tests.studies import boundary_node_properties as properties

#: The seed of the pilot stream, used by no other study.
PILOT_SEED: Final[int] = 20261431
PILOT_REPLICATES: Final[int] = 2_000
CELLS: Final[tuple[study.Cell, ...]] = study.CELLS
TARGETS: Final[dict[str, str]] = study.TARGETS
ARMS: Final[tuple[str, ...]] = study.ARMS


def draw_seed(label: str, replicate: int) -> np.random.SeedSequence:
    """The seed of one draw: the pilot seed, the cell label and the replicate."""
    return np.random.SeedSequence([PILOT_SEED, zlib.crc32(label.encode()), replicate])


def arm_rows(
    label: str, replicate: int, entries: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Flat rows of one draw's :func:`~tests.studies.boundary_node_ltmle.draw_arms` entries."""
    rows = []
    for entry in entries:
        row: dict[str, Any] = {"lost": "", "cell": label, "replicate": replicate}
        row.update({k: v for k, v in entry.items() if k not in ARMS})
        for arm in ARMS:
            for key in ("low", "high", "std_error", "covered"):
                row[f"{arm}_{key}"] = entry[arm][key]
        rows.append(row)
    return rows


def cell_rows(cell: study.Cell, replicates: int) -> list[dict[str, Any]]:
    """Every row of one pilot cell."""
    law = laws.law_of(cell.variant)
    truth = law.truths()
    rows: list[dict[str, Any]] = []
    for replicate in range(replicates):
        rng = np.random.default_rng(draw_seed(cell.name, replicate))
        frame = law.draw(cell.n, rng, cluster_size=cell.cluster_size)
        try:
            entries = study.draw_arms(frame, cell, truth)
        except (CleverlyError, RuntimeError) as error:
            # A lost fit is a miss in every arm (the study's rules count it so).
            rows.extend(_lost_rows(cell.name, replicate, truth, error))
            continue
        rows.extend(arm_rows(cell.name, replicate, entries))
    return rows


def _lost_rows(
    label: str, replicate: int, truth: Mapping[str, float], error: Exception
) -> list[dict[str, Any]]:
    """One row per target of a fit that raised: no point, every arm a miss."""
    lost = f"{type(error).__name__}: {str(error)[:80]}"
    rows = []
    for target in TARGETS:
        row: dict[str, Any] = {
            "lost": lost,
            "cell": label,
            "replicate": replicate,
            "target": target,
            "truth": truth[target],
            "psi": np.nan,
            "withheld": "",
            "firing": False,
            "multi_flag": False,
            "supplemented_here": False,
        }
        for arm in ARMS:
            row[f"{arm}_low"] = row[f"{arm}_high"] = row[f"{arm}_std_error"] = np.nan
            row[f"{arm}_covered"] = 0
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------- readings


def exact_interval(successes: int, trials: int, level: float = 0.99) -> tuple[float, float]:
    """The Clopper-Pearson interval of a proportion."""
    tail = (1.0 - level) / 2.0
    low = 0.0 if successes == 0 else float(stats.beta.ppf(tail, successes, trials - successes + 1))
    high = (
        1.0
        if successes == trials
        else float(stats.beta.ppf(1 - tail, successes + 1, trials - successes))
    )
    return low, high


mcnemar_lower = properties.mcnemar_lower
newcombe_interval = properties.newcombe_interval


def control_power(
    rows: pd.DataFrame,
    replicates: int = 4_000,
    simulations: int = 2_000,
    seed: int = 0,
    *,
    predictive: bool = False,
) -> dict[str, float]:
    """The power of both paired clauses at ``replicates`` draws, at the pilot rates.

    Each simulated study draws the firing count, then the joint arm outcomes of the firing
    draws and the plug-in coverage of the non-firing draws. At the pilot rates these come from
    the pilot's empirical shares. With ``predictive`` each simulation first draws the rates
    from their Jeffreys posteriors (a Dirichlet over the four joint outcomes, a Beta for the
    firing share and for the non-firing coverage), so the power carries the pilot's own
    uncertainty.
    """
    rng = np.random.default_rng(seed)
    fire = rows["firing"].to_numpy(dtype=bool)
    joint = rows.loc[fire, ["supplemented_covered", "plugin_covered"]].to_numpy(dtype=bool)
    rest = rows.loc[~fire & (rows["lost"] == ""), "plugin_covered"].to_numpy(dtype=bool)
    # The four joint outcomes on firing draws: (supplemented, plug-in) covered.
    outcomes = np.array([[True, True], [True, False], [False, True], [False, False]])
    counts = np.array([np.sum(np.all(joint == outcome, axis=1)) for outcome in outcomes])
    first = second = 0
    for _ in range(simulations):
        if predictive:
            share = rng.beta(fire.sum() + 0.5, (~fire).sum() + 0.5)
            shares = rng.dirichlet(counts + 0.5)
            quiet_rate = rng.beta(rest.sum() + 0.5, (~rest).sum() + 0.5)
        else:
            share, shares, quiet_rate = fire.mean(), counts / counts.sum(), rest.mean()
        m = rng.binomial(replicates, share)
        both, gained, lost, _ = rng.multinomial(m, shares)
        first += mcnemar_lower(int(gained), int(lost), m) > 0.0
        quiet = rng.binomial(replicates - m, quiet_rate)
        _, high = newcombe_interval(int(both + lost), m, quiet, replicates - m)
        second += high < 0.0
    return {"clause_i": first / simulations, "clause_ii": second / simulations}


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    """Coverage of each arm per cell and target, on all, firing and non-firing draws."""
    out: list[dict[str, Any]] = []
    for (cell, target), group in frame.groupby(["cell", "target"], sort=False):
        row: dict[str, Any] = {"cell": cell, "target": target, "draws": len(group)}
        row["firing_share"] = float(group["firing"].mean())
        row["multi_flag_share"] = float(group["multi_flag"].mean())
        row["lost"] = int((group["lost"] != "").sum())
        row["withheld"] = int((group["withheld"] != "").sum())
        for arm in ARMS:
            covered = group[f"{arm}_covered"]
            row[f"{arm}_all"] = float(covered.mean())
            row[f"{arm}_firing"] = float(covered[group["firing"]].mean())
            row[f"{arm}_quiet"] = float(covered[~group["firing"] & (group["lost"] == "")].mean())
        fire = group[group["firing"] & group["supplemented_here"]]
        if len(fire):
            sup = (fire["supplemented_high"] - fire["supplemented_low"]).to_numpy()
            orc = (fire["oracle_node_high"] - fire["oracle_node_low"]).to_numpy()
            row["width_ratio_to_oracle"] = float(np.nanmedian(sup / orc))
        out.append(row)
    return pd.DataFrame(out)


def run(cells: Iterable[study.Cell], replicates: int) -> pd.DataFrame:
    """The pilot rows of ``cells``."""
    return pd.DataFrame([row for cell in cells for row in cell_rows(cell, replicates)])


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=PILOT_REPLICATES)
    parser.add_argument("--cells", nargs="*", default=[cell.name for cell in CELLS])
    args = parser.parse_args(list(argv) if argv is not None else None)
    chosen = [cell for cell in CELLS if cell.name in set(args.cells)]
    frame = run(chosen, args.replicates)
    frame.to_csv(args.output, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(summarize(frame).round(3).to_string(index=False))
        interior = frame[(frame["cell"] == "interior_n200") & (frame["target"] == "always")]
        if len(interior):
            print("control power at R = 4000:", control_power(interior))
            print("predictive:", control_power(interior, predictive=True))


#: The readings :func:`summarize` writes, by name, for the design test.
READINGS: Final[Mapping[str, str]] = {
    "firing_share": "share of draws on which node 4 of the always fit is flagged",
    "width_ratio_to_oracle": "median half-width ratio, supplemented to oracle_node, firing draws",
}


if __name__ == "__main__":
    main()
