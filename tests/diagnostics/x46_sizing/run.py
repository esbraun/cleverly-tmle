"""The sizing probe of ``longitudinal-repeated-crossfit``: the X46 estimator on fresh draws.

:mod:`tests.unit.test_longitudinal_repeated_crossfit_design` reads every pass probability of
the study's declaration off this probe (``X46-plan.md`` 5.4).  The probe fits the study's own
declared fits (:func:`tests.studies.longitudinal_repeated_crossfit.fit_scenario`, three draws)
on seeds outside every declared stream, so it cannot change a verdict.  Per sample and name it
records the median point and standard error, draw 0's point and standard error, and the
displacement share :math:`\\operatorname{median}_r(\\psi_r - \\tilde\\psi)^2 / \\tilde\\sigma^2`.
Per sample and fit it records whether the shipped band and the pointwise intervals cover the
whole family.

=================  ==========================================================================
set                what it reads
=================  ==========================================================================
``d@8000``         law D at the declared size, ``LTMLE`` and ``SDR`` on the same sample
``i@40000``        law I at the declared size
``s@8000``         law S, horizons 1 to 3
``c@8000``         law C, 800 clusters of 10
``m@8000``         law M with the centred duration term
``<law>@2000``     each law but I at n = 2,000, where the rule must fail
``stability_d``    one fixed law-D sample, paired three-draw and first-draw fits per trial
``stability_i``    one fixed law-I sample, the same pairing
``i@80000``        law I at n = 80,000, after the 40,000 reading failed the sizing rule
``i@70000``        law I at n = 70,000, the next smaller size the 80,000 projection admits
``stability_i@``   the law-I stability pairing on one fixed sample of 80,000
``80000``
=================  ==========================================================================

Replication ``r`` of set ``k`` draws the seed ``SEED_BASE + 100_000 k + r``.  Each set runs
in saved chunks of :data:`CHUNK` samples under ``--work``, so a stopped run resumes at its
last finished chunk.  ``--assemble`` writes ``rows.csv.gz``, ``bands.csv.gz``,
``stability.csv.gz`` and ``reading.csv`` beside this file.

    python -m tests.diagnostics.x46_sizing.run --work <dir> --jobs 4
    python -m tests.diagnostics.x46_sizing.run --work <dir> --set i@40000 --jobs 4
    python -m tests.diagnostics.x46_sizing.run --work <dir> --assemble
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cleverly.learners.crossfit import CrossFitPlan
from cleverly.utils.parallel import map_parallel
from tests.studies import longitudinal_repeated_crossfit as study
from tests.studies.evidence.manifest import write_csv

HERE = Path(__file__).resolve().parent
SEED_BASE = 9_460_000
CHUNK = 100
STABILITY_TRIALS = 120
BOOTSTRAP = 4_000

#: ``set: (index, kind, scenarios, n, samples)``.  The index fixes the seed stream.
SETS: dict[str, tuple[int, str, tuple[str, ...], int, int]] = {
    "d@8000": (0, "law", ("law_d_ltmle", "law_d_sdr"), 8_000, 2_000),
    "i@40000": (1, "law", ("law_i",), 40_000, 600),
    "s@8000": (2, "law", ("law_s",), 8_000, 2_000),
    "c@8000": (3, "law", ("law_c",), 8_000, 1_600),
    "m@8000": (4, "law", ("law_m",), 8_000, 4_000),
    "d@2000": (5, "law", ("law_d_ltmle", "law_d_sdr"), 2_000, 500),
    "s@2000": (6, "law", ("law_s",), 2_000, 500),
    "c@2000": (7, "law", ("law_c",), 2_000, 500),
    "m@2000": (8, "law", ("law_m",), 2_000, 500),
    "stability_d": (9, "stability", ("law_d_ltmle", "law_d_sdr"), 8_000, STABILITY_TRIALS),
    "stability_i": (10, "stability", ("law_i",), 40_000, STABILITY_TRIALS),
    # Law I at the size the 40,000 reading failed to support (``X46-progress.md``, stage 2).
    "i@80000": (11, "law", ("law_i",), 80_000, 1_200),
    "stability_i@80000": (12, "stability", ("law_i",), 80_000, STABILITY_TRIALS),
    # The 80,000 reading passed; 70,000 is the next smaller size its projection admits.
    "i@70000": (13, "law", ("law_i",), 70_000, 1_200),
    "stability_i@70000": (14, "stability", ("law_i",), 70_000, STABILITY_TRIALS),
}

#: The names each stability set reads, by scenario.
STABILITY_NAMES = {
    "law_d_ltmle": tuple(study.D_CONTRASTS.values()),
    "law_d_sdr": tuple(study.D_CONTRASTS.values()),
    "law_i": (study.I_CONTRASTS["contrast_x2"], f"ey_regimen[{study.I_REFERENCE}]"),
}


def seed(index: int, replicate: int) -> int:
    return SEED_BASE + 100_000 * index + replicate


def _truth(scenario: str) -> dict[str, float]:
    return study.I_TRUTH if scenario == "law_i" else study.TRUTHS[scenario]


def law_sample(name: str, replicate: int) -> list[dict[str, Any]]:
    """Every row one sample of a law set publishes, estimates and bands."""
    index, _, scenarios, n, _ = SETS[name]
    frame = study.sample(study.SCENARIO_LAW[scenarios[0]], n, seed(index, replicate))
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        start = time.process_time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = study.fit_scenario(scenario, frame, simultaneous=True)
        cpu = time.process_time() - start
        truth = _truth(scenario)
        first = result.repeats[0].estimates
        for key, estimate in study.estimates(scenario, result).items():
            psi = float(estimate.inference_value)
            ratio = key == study.D_RATIO
            if ratio:
                # A derived ratio is not stored per draw; its draw value is the log ratio of
                # the draw's two levels.
                draws = np.array(
                    [
                        np.log(
                            r.estimates["ey_regimen[always]"].psi
                            / r.estimates["ey_regimen[never]"].psi
                        )
                        for r in result.repeats
                    ]
                )
            else:
                draws = np.array([float(r.estimates[key].psi) for r in result.repeats])
            rows.append(
                {
                    "kind": "estimate",
                    "set": name,
                    "scenario": scenario,
                    "replicate": replicate,
                    "name": key,
                    "truth": float(np.log(truth[key]) if ratio else truth[key]),
                    "psi_med": psi,
                    "se_med": float(estimate.std_error),
                    "psi0": float(draws[0]) if ratio else float(first[key].psi),
                    "se0": float(first[key].std_error) if key in first else np.nan,
                    "share": float(np.median((draws - psi) ** 2) / estimate.std_error**2),
                    "cpu": cpu,
                }
            )
        bands = result.simultaneous
        names = sorted(bands.bands)
        rows.append(
            {
                "kind": "band",
                "set": name,
                "scenario": scenario,
                "replicate": replicate,
                "name": "|".join(names),
                "band_covered": int(
                    all(bands.bands[k][0] <= truth[k] <= bands.bands[k][1] for k in names)
                ),
                "pointwise_covered": int(
                    all(result[k].ci[0] <= truth[k] <= result[k].ci[1] for k in names)
                ),
                "critical": float(bands.critical_value),
                "cpu": cpu,
            }
        )
    return rows


def stability_trial(name: str, trial: int) -> list[dict[str, Any]]:
    """One fold-seed trial on the set's fixed sample: three draws, and the first alone."""
    index, _, scenarios, n, _ = SETS[name]
    frame = study.sample(study.SCENARIO_LAW[scenarios[0]], n, seed(index, 99_999))
    base = seed(index, trial)
    first = CrossFitPlan(n_folds=study.N_FOLDS, random_state=base, repeats=study.REPEATS).seeds()[0]
    assert first is not None
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            repeated = study.fit_scenario(scenario, frame, random_state=base)
            single = study.fit_scenario(scenario, frame, repeats=1, random_state=first)
        for key in STABILITY_NAMES[scenario]:
            if float(repeated.repeats[0].estimates[key].psi) != float(single[key].psi):
                raise RuntimeError(f"{scenario}: the one-draw fit is not the first draw")
            rows.append(
                {
                    "set": name,
                    "scenario": scenario,
                    "trial": trial,
                    "name": key,
                    "three": float(repeated[key].psi),
                    "one": float(single[key].psi),
                }
            )
    return rows


def run_set(name: str, work: Path, jobs: int) -> None:
    _, kind, _, _, samples = SETS[name]
    folder = work / name.replace("@", "_")
    folder.mkdir(parents=True, exist_ok=True)
    worker = law_sample if kind == "law" else stability_trial
    for start in range(0, samples, CHUNK):
        path = folder / f"chunk_{start:05d}.csv.gz"
        if path.exists():
            continue
        stop = min(start + CHUNK, samples)
        began = time.time()
        parts = map_parallel(worker, [(name, r) for r in range(start, stop)], n_jobs=jobs)
        frame = pd.DataFrame([row for part in parts for row in part])
        write_csv(frame, path, compression={"method": "gzip", "mtime": 0})
        print(f"{name}: samples {start}-{stop - 1} in {time.time() - began:.0f}s", flush=True)


def _collect(work: Path, name: str) -> pd.DataFrame:
    folder = work / name.replace("@", "_")
    paths = sorted(folder.glob("chunk_*.csv.gz"))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(p, float_precision="round_trip") for p in paths])


def _bootstrap_se(statistic: Any, size: int, rng: np.random.Generator) -> float:
    draws = [statistic(rng.integers(0, size, size)) for _ in range(BOOTSTRAP)]
    return float(np.std(draws, ddof=1))


def reading(rows: pd.DataFrame, bands: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    """One row per set, scenario, name and reading."""
    rng = np.random.default_rng(46)
    records: list[dict[str, Any]] = []
    for (name, scenario, key), group in rows.groupby(["set", "scenario", "name"], sort=True):
        truth = float(group["truth"].iloc[0])
        replicates = len(group)
        for which, (psi, se) in {
            "median": ("psi_med", "se_med"),
            "draw_0": ("psi0", "se0"),
        }.items():
            estimate = group[psi].to_numpy(dtype=float)
            error = group[se].to_numpy(dtype=float)
            if not np.isfinite(error).all():
                continue
            spread = float(np.std(estimate, ddof=1))
            records.append(
                {
                    "set": name,
                    "scenario": scenario,
                    "name": key,
                    "reading": which,
                    "n": int(name.split("@")[1]),
                    "replicates": replicates,
                    "standardized_bias": float((estimate.mean() - truth) / spread),
                    "mcse": float(1.0 / np.sqrt(replicates)),
                    "se_ratio": float(error.mean() / spread),
                    "coverage": float(np.mean(np.abs(estimate - truth) <= 1.959964 * error)),
                    "share": float(group["share"].mean()) if which == "median" else np.nan,
                    "value": np.nan,
                    "value_se": np.nan,
                }
            )
        if np.isfinite(group["se0"]).all():
            median = group[["psi_med", "se_med"]].to_numpy(dtype=float)
            first = group[["psi0", "se0"]].to_numpy(dtype=float)

            def increment(
                index: np.ndarray, m: np.ndarray = median, f: np.ndarray = first
            ) -> float:
                return float(
                    m[index, 1].mean() / m[index, 0].std(ddof=1)
                    - f[index, 1].mean() / f[index, 0].std(ddof=1)
                )

            every = np.arange(replicates)
            records.append(
                {
                    "set": name,
                    "scenario": scenario,
                    "name": key,
                    "reading": "increment",
                    "n": int(name.split("@")[1]),
                    "replicates": replicates,
                    "value": increment(every),
                    "value_se": _bootstrap_se(increment, replicates, rng),
                    "share": float(median[:, 1].mean() / first[:, 1].mean() - 1.0),
                }
            )
    for (name, scenario), group in bands.groupby(["set", "scenario"], sort=True):
        for which in ("band_covered", "pointwise_covered"):
            records.append(
                {
                    "set": name,
                    "scenario": scenario,
                    "name": "band",
                    "reading": which,
                    "n": int(name.split("@")[1]),
                    "replicates": len(group),
                    "value": float(group[which].mean()),
                    "value_se": float(group[which].std(ddof=1) / np.sqrt(len(group))),
                }
            )
    for (name, scenario, key), group in stability.groupby(["set", "scenario", "name"], sort=True):
        pair = group[["three", "one"]].to_numpy(dtype=float)
        one_sd = float(pair[:, 1].std(ddof=1))

        def ratio(index: np.ndarray, p: np.ndarray = pair) -> float:
            return float(p[index, 0].std(ddof=1) / p[index, 1].std(ddof=1))

        records.append(
            {
                "set": name,
                "scenario": scenario,
                "name": key,
                "reading": "stability",
                "n": SETS[name][3],
                "replicates": len(group),
                "value": ratio(np.arange(len(group))) if one_sd > 0 else np.nan,
                "value_se": _bootstrap_se(ratio, len(group), rng) if one_sd > 0 else np.nan,
                "share": one_sd,
            }
        )
    return pd.DataFrame(records)


def complete(work: Path, name: str) -> bool:
    """Whether every chunk of set ``name`` is saved under ``work``."""
    folder = work / name.replace("@", "_")
    return len(list(folder.glob("chunk_*.csv.gz"))) * CHUNK >= SETS[name][4]


def assemble(work: Path) -> None:
    """Write the frames and the reading of every complete set; a set still running is left out."""
    laws = [name for name, spec in SETS.items() if spec[1] == "law" and complete(work, name)]
    frames = pd.concat([_collect(work, name) for name in laws], ignore_index=True)
    rows = frames.loc[frames["kind"] == "estimate"].drop(
        columns=["kind", "band_covered", "pointwise_covered", "critical"]
    )
    bands = frames.loc[
        frames["kind"] == "band",
        [
            "set",
            "scenario",
            "replicate",
            "name",
            "band_covered",
            "pointwise_covered",
            "critical",
            "cpu",
        ],
    ]
    stability = pd.concat(
        [
            _collect(work, name)
            for name, spec in SETS.items()
            if spec[1] == "stability" and complete(work, name)
        ],
        ignore_index=True,
    )
    archive = {"method": "gzip", "mtime": 0}
    write_csv(rows, HERE / "rows.csv.gz", compression=archive)
    write_csv(bands, HERE / "bands.csv.gz", compression=archive)
    write_csv(stability, HERE / "stability.csv.gz", compression=archive)
    write_csv(reading(rows, bands, stability), HERE / "reading.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--set", action="append", choices=sorted(SETS))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--assemble", action="store_true")
    arguments = parser.parse_args()
    if arguments.assemble:
        assemble(arguments.work)
        return
    for name in arguments.set or list(SETS):
        run_set(name, arguments.work, arguments.jobs)


if __name__ == "__main__":
    main()
