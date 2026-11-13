"""The pre-read and the tolerance check of ``isotonic-binding`` (Fix-8 plan sections 5.0 and 5.8).

Every pilot draw comes from :data:`PILOT_SEED`, a stream no study draws from, and is fitted by
:func:`tests.studies._isotonic_binding_law.fit` exactly as the property phase fits it.

* ``run`` fits pilot draws of a law at the pinned tolerance (``1e-13``) and writes one row per
  draw, regimen and horizon: the projected and unprojected levels, the plug-in SE, whether the
  estimate reports ``std_error``, and the logistic fits that stopped at ``max_iter``.
* ``tolerance`` fits pilot draws of ``LG2`` at ``1e-13`` and at ``1e-10`` and writes each
  level's move at both.  The committed table :data:`TOLERANCE` holds these rows for draws 0 to
  299, and the design test applies the tolerance rule of plan 5.8 to it.
* ``run --plan-seeds`` fits the plan's own 2,400 pre-read samples (:data:`PLAN_SEEDS`).
* ``summarize`` writes :data:`SUMMARY`: per law, regimen and horizon, the pooled draws and
  their blocks, the coverage of
  the projected and unprojected intervals and their covered counts on the draws that report an
  SE, the SE ratio (mean SE over the SD of the projected level), the standardized bias, the SE
  share, the binding and material-move shares, the move distribution, the share of draws where
  the two intervals disagree, both coverages on moved draws, and the non-converged fits.

The per-draw rows of ``run`` are not committed.  Run
``python -m tests.studies.isotonic_binding_pilot run --law LG2 --start S --draws D --out FILE``
per block, ``... run --plan-seeds --out FILE`` once, ``... tolerance --draws 300`` once, and
``... summarize FILE...``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from tests.studies import _isotonic_binding_law as law

#: The pre-read draws of ``LG2`` (plan 5.0 asks for at least 2,000), of the larger-move law,
#: and of the tolerance check (plan 5.8).
DRAWS = 4_800
LARGE_MOVE_DRAWS = 1_600
TOLERANCE_DRAWS = 300
N = 2_000
#: The plan's 2,400 pre-read draws of ``LG2`` (Fix-8 ``pilots/isotonic_continuous_pilot.py``,
#: seeds 10000-11199 and 20000-21199, each drawn by ``numpy.random.default_rng(seed)`` in the
#: node order of :func:`tests.studies._isotonic_binding_law.draw`).  The pre-read pools them with
#: the :data:`DRAWS` fresh draws, as plan 5.8 (rev 2.2.1) asks.
PLAN_SEEDS = (*range(10_000, 11_200), *range(20_000, 21_200))
#: The pilot's own seed stream.
PILOT_SEED = 20261131
#: A level "binds" when it moves by more than this many SE, and moves materially at this many.
BINDING_SE = 1e-6
MATERIAL_SE = 0.1
CRITICAL = float(norm.ppf(0.975))
SUMMARY = Path(__file__).with_name("isotonic_binding_pilot.csv")
TOLERANCE = Path(__file__).with_name("isotonic_binding_tolerance.csv")


def pilot_seed(draw: int, law_name: str = law.LAW) -> int:
    """The seed of one pilot draw of one law."""
    key = (draw, 0 if law_name == law.LAW else 1)
    return int(np.random.SeedSequence(entropy=PILOT_SEED, spawn_key=key).generate_state(1)[0])


def records(draw: int, law_name: str = law.LAW, *, block: str = "pilot") -> list[dict[str, Any]]:
    """One pilot row per regimen and horizon of one draw.

    A ``plan`` draw is the plan's own pre-read sample, whose seed is the draw id.
    """
    seed = draw if block == "plan" else pilot_seed(draw, law_name)
    frame = law.draw(N, seed, law_name)
    result, stalled = law.fit_counted(frame)
    return [
        {
            "law": law_name,
            "block": block,
            "regimen": regimen,
            "horizon": horizon,
            "n": N,
            "draw": draw,
            "truth": law.TRUTH[law.name(regimen, horizon)],
            "projected": level.projected,
            "unprojected": level.unprojected,
            "std_error": level.std_error,
            "has_se": int(level.has_se),
            "stalled": stalled,
        }
        for (regimen, horizon), level in law.levels(result).items()
    ]


def tolerance_records(draw: int) -> list[dict[str, Any]]:
    """Each level's move of one ``LG2`` draw at the pinned and the refit tolerance."""
    frame = law.draw(N, pilot_seed(draw))
    fits = {tol: law.fit_counted(frame, tol) for tol in (law.LOGISTIC_TOL, law.REFIT_TOL)}
    pinned, refit = (law.levels(fits[tol][0]) for tol in (law.LOGISTIC_TOL, law.REFIT_TOL))
    return [
        {
            "draw": draw,
            "regimen": regimen,
            "horizon": horizon,
            "move_pinned": pinned[key].move,
            "move_refit": refit[key].move,
            "stalled_pinned": fits[law.LOGISTIC_TOL][1],
            "stalled_refit": fits[law.REFIT_TOL][1],
        }
        for key in pinned
        for regimen, horizon in [key]
    ]


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """One summary row per law, regimen and horizon."""
    out = []
    for (law_name, regimen, horizon), group in rows.groupby(["law", "regimen", "horizon"]):
        blocks = group["block"].value_counts()
        se = group["std_error"].to_numpy()
        truth = float(group["truth"].iloc[0])
        projected = group["projected"].to_numpy()
        unprojected = group["unprojected"].to_numpy()
        move = np.abs(projected - unprojected) / se
        covered = np.abs(projected - truth) <= CRITICAL * se
        covered_before = np.abs(unprojected - truth) <= CRITICAL * se
        with_se = group["has_se"].to_numpy(dtype=bool)
        moved = move > BINDING_SE
        sd = float(np.std(projected[with_se], ddof=1))
        out.append(
            {
                "law": law_name,
                "regimen": regimen,
                "horizon": int(horizon),
                "n": int(group["n"].iloc[0]),
                "draws": len(group),
                "pilot_draws": int(blocks.get("pilot", 0)),
                "plan_draws": int(blocks.get("plan", 0)),
                "truth": truth,
                "se_share": float(with_se.mean()),
                "covered": int(covered[with_se].sum()),
                "covered_unprojected": int(covered_before[with_se].sum()),
                "coverage": float(covered[with_se].mean()),
                "coverage_unprojected": float(covered_before[with_se].mean()),
                "disagree_share": float(np.mean(covered[with_se] != covered_before[with_se])),
                "sd": sd,
                "mean_se": float(se[with_se].mean()),
                "se_ratio": float(se[with_se].mean()) / sd,
                "bias_over_sd": float(np.mean(projected[with_se]) - truth) / sd,
                "binding_share": float(moved.mean()),
                "material_share": float(np.mean(move >= MATERIAL_SE)),
                "mean_move_moved": float(move[moved].mean()) if moved.any() else 0.0,
                "max_move": float(move.max()),
                "coverage_moved": float(covered[moved & with_se].mean()) if moved.any() else np.nan,
                "coverage_unprojected_moved": (
                    float(covered_before[moved & with_se].mean()) if moved.any() else np.nan
                ),
                "stalled": int(group.drop_duplicates("draw")["stalled"].sum()),
            }
        )
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("--law", choices=(law.LAW, law.LARGE_MOVE_LAW), default=law.LAW)
    runner.add_argument("--start", type=int, default=0)
    runner.add_argument("--draws", type=int, default=0)
    runner.add_argument("--plan-seeds", action="store_true")
    runner.add_argument("--out", type=Path, required=True)
    tolerance = commands.add_parser("tolerance")
    tolerance.add_argument("--draws", type=int, default=TOLERANCE_DRAWS)
    summary = commands.add_parser("summarize")
    summary.add_argument("files", type=Path, nargs="+")
    arguments = parser.parse_args(argv)
    if arguments.command == "run":
        if arguments.plan_seeds:
            rows = [row for seed in PLAN_SEEDS for row in records(seed, law.LAW, block="plan")]
        else:
            draws = range(arguments.start, arguments.start + arguments.draws)
            rows = [row for draw in draws for row in records(draw, arguments.law)]
        pd.DataFrame(rows).to_parquet(arguments.out, index=False)
        return
    if arguments.command == "tolerance":
        rows = [row for draw in range(arguments.draws) for row in tolerance_records(draw)]
        pd.DataFrame(rows).to_csv(TOLERANCE, index=False, float_format="%.10g")
        return
    rows = pd.concat([pd.read_parquet(path) for path in arguments.files], ignore_index=True)
    rows = rows.drop_duplicates(["law", "block", "regimen", "horizon", "draw"])
    summarize(rows).to_csv(SUMMARY, index=False, float_format="%.6g")


if __name__ == "__main__":
    main()
