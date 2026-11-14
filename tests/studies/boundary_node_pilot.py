r"""The pilot of ``boundary-node-ltmle``, run before the declaration on its own seed stream.

Each cell of the study is fitted on 400 draws of :mod:`tests.studies.boundary_node_laws`, and
each draw writes one row per target with three arms on the same fit:

* ``supplemented``: the shipped interval, the curve's variance plus the boundary-node
  supplement (:mod:`cleverly.inference.node_variance`);
* ``plugin``: the same estimate at the curve's variance alone, the interval the package withheld
  before the supplement;
* ``oracle_node``: the curve's variance plus the flagged nodes' variance at the true hazard,
  :math:`n^{-2}\sum_u(\sum_{i \in u} |\lambda_r||C_i|\sigma_i)^2` with :math:`\sigma_i^2 = h(1-h)`
  from the law, in the same Minkowski form as the supplement.

Each row records whether node 4 of the ``always`` fit was flagged (``firing``) and whether any
other node was (``other_flag``). :func:`summarize` reads the coverage of each arm on all, firing
and non-firing draws, and the two clauses of the paired control at the pilot rates.

Run with ``python -m tests.studies.boundary_node_pilot --output <file.csv.gz>``. The path of the
plan, the hazard targeting path, is not fitted: it targets a held baseline treatment only, and
this law has three treatment decisions.
"""

from __future__ import annotations

import argparse
import dataclasses
import re
import warnings
import zlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from scipy import stats

from cleverly.exceptions import CleverlyError
from cleverly.longitudinal import LTMLE, LongitudinalData
from tests.studies import boundary_node_laws as laws
from tests.studies.fractional_glm import QuasiBinomialGLM

#: The seed of the pilot stream, used by no other study.
PILOT_SEED: Final[int] = 20261431
PILOT_REPLICATES: Final[int] = 400
#: The cells: ``(name, law variant, n, folds, cluster size)``.
CELLS: Final[tuple[tuple[str, str, int, int, int | None], ...]] = (
    ("interior_n100", "interior", 100, 1, None),
    ("interior_n200", "interior", 200, 1, None),
    ("interior_n400", "interior", 400, 1, None),
    ("interior_n200_crossfit", "interior", 200, 5, None),
    ("clustered_j40", "clustered", 400, 1, 10),
    ("zero_n200", "zero", 200, 1, None),
)
TARGETS: Final[dict[str, str]] = {
    "always": "risk_regimen[always @ t=4]",
    "never": "risk_regimen[never @ t=4]",
    "contrast": "ate_regimen[always vs never @ t=4]",
}
ARMS: Final[tuple[str, ...]] = ("supplemented", "plugin", "oracle_node")
ALPHA: Final[float] = 0.05


def draw_seed(label: str, replicate: int) -> np.random.SeedSequence:
    """The seed of one draw: the pilot seed, the cell label and the replicate."""
    return np.random.SeedSequence([PILOT_SEED, zlib.crc32(label.encode()), replicate])


def fit(frame: pd.DataFrame, *, n_folds: int, clustered: bool) -> Any:
    """The iterated-mean LTMLE of the study: GLM main terms, the known mechanisms."""
    data = LongitudinalData.from_frame(
        frame,
        outcome=[f"Y{t}" for t in range(1, laws.NODES + 1)],
        treatment=[f"A{t}" for t in range(1, laws.NODES + 1)],
        baseline=["W"],
        treatment_probabilities=frame[[f"p{t}" for t in range(1, laws.NODES + 1)]].to_numpy(),
        id="cluster" if clustered else None,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LTMLE(
            laws.REGIMENS,
            reference="never",
            outcome_learner=QuasiBinomialGLM(),
            pseudo_learner=QuasiBinomialGLM(),
            n_folds=n_folds,
            simultaneous=False,
            random_state=0,
            g_bounds=(1e-8, 1.0),
        ).fit(data)


def _fit_key(level: str) -> str:
    """``risk_regimen[always @ t=4]`` -> ``always @ t=4``, the key of :attr:`fits`."""
    match = re.search(r"\[(.*)\]", level)
    assert match is not None
    return match.group(1)


def oracle_node_variance(result: Any, name: str, law: laws.Law, frame: pd.DataFrame) -> float:
    """The flagged nodes' variance of ``name`` at the true hazard (module docstring)."""
    estimate = result[name]
    if estimate.node is None:
        return 0.0
    loadings = estimate.node.loadings
    weights = np.abs(np.asarray(estimate.node.gradient, dtype=float))
    units = (
        np.arange(len(frame))
        if result.data.cluster is None
        else np.unique(np.asarray(result.data.cluster), return_inverse=True)[1]
    )
    obs = np.asarray(result.data.weights, dtype=float)
    per_node: dict[int, np.ndarray] = {}
    for record in loadings.nodes:
        lam = weights[loadings.levels.index(record.level)]
        fit_ = result.fits[_fit_key(record.level)]
        step = next(step for step in fit_.steps if step.time == record.node)
        rows = np.asarray(step.trained_on, dtype=bool) & (obs > 0.0)
        regimen = laws.REGIMENS[_fit_key(record.level).split(" @ ")[0]]
        spread = laws.oracle_spread(law, frame, record.node, regimen)
        term = lam * np.abs(obs * step.clever) * spread * rows
        per_node[record.node] = per_node.get(record.node, 0.0) + np.bincount(
            units, weights=term, minlength=units.max() + 1
        )
    stacked = np.array(list(per_node.values()))
    total = np.sum(stacked.sum(axis=0) ** 2) if loadings.clustered else np.sum(stacked**2)
    return float(total) / float(loadings.n) ** 2


def _interval(estimate: Any, variance: float | None) -> tuple[float, float]:
    """The estimate's interval, or the same construction at another variance."""
    if variance is None:
        return estimate.ci if estimate.supplies_inference else (np.nan, np.nan)
    moved = dataclasses.replace(estimate, node=None, variance=variance)
    return moved.plugin_interval if estimate.supplies_inference else (np.nan, np.nan)


def cell_rows(cell: tuple[str, str, int, int, int | None], replicates: int) -> list[dict[str, Any]]:
    """Every row of one pilot cell."""
    label, variant, n, n_folds, cluster_size = cell
    law = laws.law_of(variant)
    truth = law.truths()
    rows: list[dict[str, Any]] = []
    for replicate in range(replicates):
        rng = np.random.default_rng(draw_seed(label, replicate))
        frame = law.draw(n, rng, cluster_size=cluster_size)
        try:
            result = fit(frame, n_folds=n_folds, clustered=cluster_size is not None)
        except (CleverlyError, RuntimeError) as error:
            # A lost fit is a miss in every arm (the study's G1 counts it so).
            rows.extend(_lost_rows(label, replicate, truth, error))
            continue
        always = result[TARGETS["always"]]
        flagged = set() if always.node is None else {r.node for r in always.node.loadings.nodes}
        firing = 4 in flagged
        others = {
            r.node
            for key in TARGETS.values()
            if result[key].node is not None
            for r in result[key].node.loadings.nodes
            if r.node != 4
        }
        for target, name in TARGETS.items():
            estimate = result[name]
            curve = (
                float(estimate.node.influence_variance)
                if estimate.node is not None
                else float(estimate.variance)
            )
            oracle = oracle_node_variance(result, name, law, frame)
            intervals = {
                "supplemented": _interval(estimate, None),
                "plugin": _interval(estimate, curve),
                "oracle_node": _interval(estimate, curve + oracle),
            }
            row: dict[str, Any] = {
                "lost": "",
                "cell": label,
                "replicate": replicate,
                "target": target,
                "truth": truth[target],
                "psi": float(estimate.psi),
                "inference": str(estimate.inference),
                "firing": firing,
                "other_flag": bool(others),
                "supplemented_here": estimate.node is not None,
                "curve_variance": curve,
                "node_variance": 0.0 if estimate.node is None else estimate.node.node_variance,
                "oracle_variance": oracle,
            }
            for arm, (low, high) in intervals.items():
                row[f"{arm}_low"], row[f"{arm}_high"] = low, high
                row[f"{arm}_covered"] = bool(low <= truth[target] <= high)
            rows.append(row)
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
            "inference": "lost",
            "firing": False,
            "other_flag": False,
            "supplemented_here": False,
            "curve_variance": np.nan,
            "node_variance": np.nan,
            "oracle_variance": np.nan,
        }
        for arm in ARMS:
            row[f"{arm}_low"] = row[f"{arm}_high"] = np.nan
            row[f"{arm}_covered"] = False
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


def mcnemar_lower(gained: int, lost: int, trials: int, level: float = 0.99) -> float:
    """The lower end of the paired difference in coverage, exact on the discordant draws.

    The discordant draws are ``gained`` (supplemented covers, plug-in misses) and ``lost``;
    the share :math:`\\theta` of ``gained`` among them has a Clopper-Pearson interval, and the
    difference is :math:`(2\\theta - 1)(\\text{gained} + \\text{lost}) / \\text{trials}`.
    """
    discordant = gained + lost
    if discordant == 0:
        return 0.0
    low, _ = exact_interval(gained, discordant, level)
    return (2.0 * low - 1.0) * discordant / trials


def newcombe_interval(
    x1: int, n1: int, x2: int, n2: int, level: float = 0.99
) -> tuple[float, float]:
    """Newcombe's hybrid score interval of :math:`p_1 - p_2` (two independent binomials)."""
    z = float(stats.norm.ppf(1.0 - (1.0 - level) / 2.0))

    def wilson(x: int, n: int) -> tuple[float, float, float]:
        p = x / n
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return p, centre - half, centre + half

    p1, l1, u1 = wilson(x1, n1)
    p2, l2, u2 = wilson(x2, n2)
    d = p1 - p2
    return (
        d - float(np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)),
        d + float(np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)),
    )


def control_power(
    rows: pd.DataFrame, replicates: int = 4_000, simulations: int = 2_000, seed: int = 0
) -> dict[str, float]:
    """The power of both paired clauses at ``replicates`` draws, at the pilot rates.

    Each simulated study draws the firing count, then the joint arm outcomes of each firing
    draw from the pilot's empirical joint distribution and the plug-in coverage of each
    non-firing draw from its pilot rate.
    """
    rng = np.random.default_rng(seed)
    fire = rows["firing"].to_numpy(dtype=bool)
    joint = rows.loc[fire, ["supplemented_covered", "plugin_covered"]].to_numpy(dtype=bool)
    rest = rows.loc[~fire & (rows["lost"] == ""), "plugin_covered"].to_numpy(dtype=bool)
    share = fire.mean()
    first = second = 0
    for _ in range(simulations):
        m = rng.binomial(replicates, share)
        pick = joint[rng.integers(0, len(joint), m)]
        gained = int(np.sum(pick[:, 0] & ~pick[:, 1]))
        lost = int(np.sum(~pick[:, 0] & pick[:, 1]))
        first += mcnemar_lower(gained, lost, m) > 0.0
        quiet = rng.binomial(replicates - m, rest.mean())
        low, high = newcombe_interval(int(pick[:, 1].sum()), m, quiet, replicates - m)
        second += high < 0.0 or low > 0.0
    return {"clause_i": first / simulations, "clause_ii": second / simulations}


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    """Coverage of each arm per cell and target, on all, firing and non-firing draws."""
    out: list[dict[str, Any]] = []
    for (cell, target), group in frame.groupby(["cell", "target"], sort=False):
        row: dict[str, Any] = {"cell": cell, "target": target, "draws": len(group)}
        row["firing_share"] = float(group["firing"].mean())
        row["other_flag_share"] = float(group["other_flag"].mean())
        row["lost"] = int((group["lost"] != "").sum())
        row["withheld"] = int(
            ((group["inference"] != "influence_curve") & (group["lost"] == "")).sum()
        )
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


def run(cells: Iterable[tuple[str, str, int, int, int | None]], replicates: int) -> pd.DataFrame:
    """The pilot rows of ``cells``."""
    return pd.DataFrame([row for cell in cells for row in cell_rows(cell, replicates)])


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=PILOT_REPLICATES)
    parser.add_argument("--cells", nargs="*", default=[cell[0] for cell in CELLS])
    args = parser.parse_args(list(argv) if argv is not None else None)
    chosen = [cell for cell in CELLS if cell[0] in set(args.cells)]
    frame = run(chosen, args.replicates)
    frame.to_csv(args.output, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(summarize(frame).round(3).to_string(index=False))
        interior = frame[(frame["cell"] == "interior_n200") & (frame["target"] == "always")]
        if len(interior):
            print("control power at R = 4000:", control_power(interior))


#: The readings :func:`summarize` writes, by name, for the design test.
READINGS: Final[Mapping[str, str]] = {
    "firing_share": "share of draws on which node 4 of the always fit is flagged",
    "width_ratio_to_oracle": "median half-width ratio, supplemented to oracle_node, firing draws",
}


if __name__ == "__main__":
    main()
