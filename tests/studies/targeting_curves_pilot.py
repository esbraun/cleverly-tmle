"""The pilot of the ``targeting-curves`` study (X92 PR-5 plan, section 10, "Pilot").

A pilot, not a result: it fixes what the declaration cannot compute in closed form before the
run.  Each group is fitted ``R`` times (800 by default) on draws from the named stream
``stream_seed(STUDY, "targeting_pilot", label, r)``, which is disjoint from the run's
``property_sample`` streams and from its primary ``SEED``.  For every calibration row it
records the ratio of the mean reported SE to the empirical SD of the errors with its
bootstrap SE, the SE's coefficient of variation, the errors' kurtosis and the Wald coverage
(the inputs of :func:`tests.studies.targeting_curves.calibration_row_pass`); for every SE
control the ratio of its mean SE to the same SD; for every point control its standardized
shift; the band's joint coverage and rejection rate; and the seconds per fit.

The design test reads it before the run, and never after a verdict, for three checks whose
consequences are declared now:

- the in-sample control's displacement on cell ``S``: below detection 0.90 the curve moves to
  :data:`~tests.studies.targeting_curves.SPLIT_FALLBACK_BASIS`, on record;
- each calibration row's paired gap: on every replicate the pilot also reads the half-sample
  SE of the oracle score (true nuisances) on the same rows and ranking
  (:func:`~tests.studies.targeting_curves._same_draw_oracle`, about 0.09 s), and the gap is the
  fitted ratio minus that same-draw oracle ratio.  Each cell's gap, pooled over its positives,
  shifts its rows' oracle ratios (:func:`~tests.studies.targeting_curves.design_ratios`), and
  ``R_MAIN`` and ``REPORTING_CELLS`` are recomputed from the shifted ratios by the declared
  rules (:func:`~tests.studies.targeting_curves.declare_design`, capped at ``R_CAP``).  If
  the cap does not reach the floor, the cell with the lowest gap is a defect suspect,
  diagnosed before the run, and none of its rows moves to reporting.  A row whose gap departs
  from its cell's by a Bonferroni z over the 34 positives is a defect suspect the same way;
- the kurtosis: above 4, ``R`` moves to the smallest multiple of 400 meeting both floors at
  that kurtosis.

Cell ``K``'s control side is computed, not piloted
(:func:`~tests.studies.targeting_curves.k_untargeted_ratio`).

The committed summary is ``targeting_curves_pilot.csv`` beside this module, written by::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \\
    NUMEXPR_NUM_THREADS=1 python -m tests.studies.targeting_curves_pilot \\
        --raw <scratch dir> --jobs 4 [--groups ...] [--replicates R] [--summarize]

Each group's records go to ``<scratch dir>/<label>.json``; ``--summarize`` reads them and
writes the CSV.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from tests.studies import targeting_curves as study
from tests.studies.evidence.seeds import stream_seed

#: The stream the pilot draws from.
STREAM = "targeting_pilot"

#: Replications of each group.
REPLICATES = study.PILOT_REPLICATES

#: Bootstrap resamples of each ratio's standard error.
BOOTSTRAP = 400

PILOT_CSV = study.PILOT_CSV

COLUMNS = (
    "group",
    "name",
    "kind",
    "replicates",
    "failed",
    "mean_se",
    "se_ratio",
    "se_ratio_se",
    "oracle_se_ratio",
    "gap_se",
    "se_cv",
    "kurtosis",
    "coverage",
    "standardized_bias",
    "seconds",
    "sobol_error",
)


def _one(key: tuple[str, int]) -> dict[str, Any]:
    label, replicate = key
    group = study.GROUP_BY_LABEL[label]
    seed = stream_seed(study.STUDY, STREAM, label, replicate)
    frame, truths = study.draw_from_seed(label, group.n, seed)
    started = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = study.fit_group(group, frame, random_state=replicate, oracle=True)
    except Exception as error:
        return {"replicate": replicate, "error": repr(error)[:300]}
    truths = {**truths, **fitted.get("truths", {})}
    return {
        "replicate": replicate,
        "seconds": time.perf_counter() - started,
        "truth": {name: float(truths[name]) for name in group.rows},
        "estimate": {name: float(fitted["rows"][name]["estimate"]) for name in group.rows},
        "se": {name: float(fitted["rows"][name]["std_error"]) for name in group.rows},
        "oracle_estimate": {name: fitted["oracle"][name][0] for name in group.rows},
        "oracle_se": {name: fitted["oracle"][name][1] for name in group.rows},
        "controls": fitted["controls"],
        "control_truth": {
            name: float(truths[name])
            for names in group.controls.values()
            for name in names
            if name in truths
        },
        "control_estimate": {
            name: float(fitted["rows"][name]["estimate"])
            for names in group.controls.values()
            for name in names
            if name in fitted["rows"]
        },
        "band": fitted.get("band"),
    }


def run(labels: tuple[str, ...], raw: Path, *, jobs: int, replicates: int) -> None:
    """Fit replicates ``0`` to ``replicates - 1`` of each group's pilot into ``raw``."""
    raw.mkdir(parents=True, exist_ok=True)
    with Pool(jobs) as pool:
        for label in labels:
            records = pool.map(_one, [(label, r) for r in range(replicates)], chunksize=4)
            (raw / f"{label}.json").write_text(json.dumps(records))


def _ratio(se: np.ndarray, errors: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    ratio = float(np.mean(se) / np.std(errors, ddof=1))
    draws = []
    for _ in range(BOOTSTRAP):
        index = rng.integers(0, len(errors), len(errors))
        draws.append(np.mean(se[index]) / np.std(errors[index], ddof=1))
    return ratio, float(np.nanstd(draws, ddof=1))


def _paired_gap(
    se: np.ndarray,
    errors: np.ndarray,
    oracle_se: np.ndarray,
    oracle_errors: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """The same-draw oracle ratio, and the bootstrap SE of the gap between the two ratios."""
    oracle = float(np.mean(oracle_se) / np.std(oracle_errors, ddof=1))
    gaps = []
    for _ in range(BOOTSTRAP):
        index = rng.integers(0, len(errors), len(errors))
        gaps.append(
            np.mean(se[index]) / np.std(errors[index], ddof=1)
            - np.mean(oracle_se[index]) / np.std(oracle_errors[index], ddof=1)
        )
    return oracle, float(np.nanstd(gaps, ddof=1))


def summarize(raw: Path) -> pd.DataFrame:
    """One row per calibration quantity, SE control and point control of every group."""
    rng = np.random.default_rng(study.SEED)
    out = []
    for group in study.GROUPS:
        path = raw / f"{group.label}.json"
        if not path.exists():
            continue
        records = json.loads(path.read_text())
        kept = [record for record in records if "error" not in record]
        common = {
            "group": group.label,
            "replicates": len(kept),
            "failed": len(records) - len(kept),
            "seconds": float(np.mean([record["seconds"] for record in kept])),
        }
        for name in group.rows:
            truth = np.array([record["truth"][name] for record in kept])
            estimate = np.array([record["estimate"][name] for record in kept])
            se = np.array([record["se"][name] for record in kept])
            errors = estimate - truth
            ratio, ratio_se = _ratio(se, errors, rng)
            oracle_se = np.array([record["oracle_se"][name] for record in kept])
            oracle_errors = np.array([record["oracle_estimate"][name] for record in kept]) - truth
            oracle_ratio, gap_se = _paired_gap(se, errors, oracle_se, oracle_errors, rng)
            out.append(
                {
                    **common,
                    "name": name,
                    "kind": group.positive,
                    "mean_se": float(np.mean(se)),
                    "se_ratio": ratio,
                    "se_ratio_se": ratio_se,
                    "oracle_se_ratio": oracle_ratio,
                    "gap_se": gap_se,
                    "se_cv": float(np.std(se, ddof=1) / np.mean(se)),
                    "kurtosis": float(stats.kurtosis(errors, fisher=False)),
                    "coverage": float(np.mean(np.abs(errors) <= study.CRITICAL * se)),
                    "standardized_bias": float(np.mean(errors) / np.std(errors, ddof=1)),
                }
            )
        for control, names in group.controls.items():
            if control == "pointwise_joint":
                continue
            for name in names:
                values = np.array([record["controls"][control][name] for record in kept])
                if control in study.SE_MUTATIONS:
                    truth = np.array([record["truth"][name] for record in kept])
                    estimate = np.array([record["estimate"][name] for record in kept])
                    ratio, ratio_se = _ratio(values, estimate - truth, rng)
                    row = {"se_ratio": ratio, "se_ratio_se": ratio_se}
                else:
                    key = f"truth:{name}"
                    truth = np.array(
                        [
                            record["controls"][control].get(key, record["control_truth"].get(name))
                            for record in kept
                        ],
                        dtype=float,
                    )
                    errors = values - truth
                    row = {"standardized_bias": float(np.mean(errors) / np.std(errors, ddof=1))}
                out.append({**common, "name": name, "kind": control, **row})
        if group.band is not None:
            covered = np.array([record["band"]["covered"] for record in kept])
            rejected = np.array([record["band"]["rejected"] for record in kept])
            pointwise = np.array([record["band"]["pointwise_covered"] for record in kept])
            for name, values in (
                ("band_covered", covered),
                ("band_rejected", rejected),
                ("band_pointwise_covered", pointwise),
            ):
                out.append(
                    {**common, "name": name, "kind": "band", "coverage": float(values.mean())}
                )
    frame = pd.DataFrame(out)
    for column in COLUMNS:
        if column not in frame:
            frame[column] = math.nan
    return frame.loc[:, list(COLUMNS)]


#: Replicates and points of cell ``S``'s truth-accuracy check (plan section 10).
SOBOL_CHECK_REPLICATES = 20
SOBOL_CHECK_POINTS = 2**22
#: Seed of the check's Sobol set, apart from the truth set and the design's.
SOBOL_CHECK_SEED = 20261112


def sobol_check(replicates: int = SOBOL_CHECK_REPLICATES) -> pd.DataFrame:
    """The largest difference over ``replicates`` fitted ``S`` curves between the declared
    ``2^20``-point truth and a ``2^22``-point one, per quantity.

    The fits are those of the pilot's own stream (replicates ``0`` to ``replicates - 1``).
    The design test requires each difference to be at most 1% of the row's pilot SE.
    """
    from scipy.stats import qmc

    group = study.GROUP_BY_LABEL["S"]
    law = study.law_of(group)
    fine = 2.0 * qmc.Sobol(3, scramble=True, seed=SOBOL_CHECK_SEED).random(SOBOL_CHECK_POINTS) - 1.0
    worst = dict.fromkeys(group.rows, 0.0)
    for replicate in range(replicates):
        seed = stream_seed(study.STUDY, STREAM, "S", replicate)
        frame, _ = study.draw_from_seed("S", group.n, seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            effect = study._study(group, frame).identify(
                study.ConditionalContrast(study.SPLIT_MODIFIERS, model=study.SPLIT_BASIS)
            )
            fitted = effect.estimate(
                **study.learners(group, law), random_state=replicate
            ).effect_curve()
            curve = fitted.diagnostics.targeting_curve(n_half_samples=2, random_state=replicate)
        coarse = study.split_truth(curve.score_curve.predict)
        dense = study.split_truth(curve.score_curve.predict, points=fine)
        for name in group.rows:
            worst[name] = max(worst[name], abs(coarse[name] - dense[name]))
    return pd.DataFrame(
        [
            {
                "group": "S",
                "name": name,
                "kind": "sobol_error",
                "replicates": replicates,
                "sobol_error": value,
            }
            for name, value in worst.items()
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--groups", nargs="*", default=[group.label for group in study.GROUPS])
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        frame = pd.concat([summarize(args.raw), sobol_check()], ignore_index=True)
        frame.loc[:, list(COLUMNS)].to_csv(PILOT_CSV, index=False, float_format="%.6g")
        return
    run(tuple(args.groups), args.raw, jobs=args.jobs, replicates=args.replicates)


if __name__ == "__main__":
    main()
