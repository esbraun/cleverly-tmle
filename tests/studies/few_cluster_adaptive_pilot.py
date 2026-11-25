r"""The pilot of ``few-cluster-adaptive-learners``, run before the declaration on its own stream.

Every draw cell (law and cluster count) is fitted on :data:`PILOT_DRAWS` draws by every fit of
the study: the default-library fits, their GLM baselines and both controls on L1 and L2, and
both ``LTMLE`` fits on L3. The rows are the study's own
(:func:`tests.studies.few_cluster_adaptive_properties.fit_rows`), with each key's wall and CPU
seconds; a fit that raises writes one row per arm with ``lost`` set and ``covered = 0``. The
draws come from :data:`PILOT_SEED`, a stream no other study reads.

The pilot is 400 draws per cell: at 400 draws a
coverage reading has an SE of about 0.011 and a log variance ratio about 0.06, enough to tell a
coverage of 0.95 from 0.90 and a variance ratio of 0.52 from 1. Budget, from the measured CPU
seconds per fit (stacked default library 17.8 to 23.3, in sample 3.8 to 5.9, cross-fitted
``LTMLE`` about 26, the stacked control about 0.5, every GLM fit under 0.1): about 14 CPU-h
stacked, 3 in sample, 3 for L3 and under 1 for the rest, so about 20 CPU-h, and 24 if the
``LTMLE`` fit takes the 62 s a shared machine read. The committed run, on a shared 4-core
machine, took 26.9 CPU-h (``pilot.json``), with every stacked fit and the ``LTMLE`` fit within
1.5 times these figures.

Run from the repository root, with every BLAS and OpenMP thread variable at 1::

    python -m tests.studies.few_cluster_adaptive_pilot --output <directory> --jobs <cores>

The run writes one checkpoint file per chunk of :data:`CHUNK_DRAWS` draws of one fit under
``<directory>/chunks/`` as each chunk finishes, and skips every chunk already written, so the
same command resumes a stopped run. When every chunk is present it writes
``<directory>/pilot.csv.gz``, the file the design test reads from
``tests/canonical/few_cluster_adaptive_learners/pilot.csv.gz``, and ``pilot.json`` with the
commit, the draw count and the CPU seconds.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import zlib
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from scipy import stats

from tests.studies import few_cluster_adaptive_learners as study
from tests.studies import few_cluster_adaptive_properties as properties

#: The seed of the pilot stream, used by no other study.
PILOT_SEED: Final[int] = 20264583
#: Draws per cell and fit.
PILOT_DRAWS: Final[int] = 400
#: Draws per checkpoint chunk: one file per ``(law, J, fit)`` and block of this many draws.
CHUNK_DRAWS: Final[int] = 20
#: The fits by cost, heaviest first, so the run's tail is short fits.
_HEAVY: Final[dict[str, int]] = {"tmle_stacked": 0, "ltmle_crossfit": 0, "tmle_in_sample": 1}

_Chunk = tuple[str, int, str, int]


def draw_seed(law: str, clusters: int, replicate: int) -> np.random.SeedSequence:
    """The seed of one pilot draw: the pilot seed, the cell label and the replicate."""
    label = f"{law}/J{clusters}"
    return np.random.SeedSequence([PILOT_SEED, zlib.crc32(label.encode()), replicate])


def pilot_keys() -> list[tuple[str, int, int, str]]:
    """Every ``(law, J, replicate, fit)`` the pilot fits."""
    return [
        (law, clusters, r, fit)
        for law, clusters in study.DRAW_CELLS
        for fit in study.fits_of(law)
        for r in range(PILOT_DRAWS)
    ]


def pilot_chunks() -> list[_Chunk]:
    """Every ``(law, J, fit, first replicate)`` chunk, the expensive fits first."""
    chunks = [
        (law, clusters, fit, start)
        for law, clusters in study.DRAW_CELLS
        for fit in study.fits_of(law)
        for start in range(0, PILOT_DRAWS, CHUNK_DRAWS)
    ]
    return sorted(chunks, key=lambda c: (_HEAVY.get(c[2], 2), -c[1], c[0], c[2], c[3]))


def chunk_path(directory: Path, chunk: _Chunk) -> Path:
    """The checkpoint file of one chunk."""
    law, clusters, fit, start = chunk
    return directory / "chunks" / f"{law}__j{clusters}__{fit}__{start:04d}.csv.gz"


def _frame(law: str, clusters: int, replicate: int) -> pd.DataFrame:
    seed = int(draw_seed(law, clusters, replicate).generate_state(1)[0])
    return study.draw(law, clusters, seed)


def key_rows(key: tuple[str, int, int, str]) -> list[dict[str, Any]]:
    """The rows of one pilot key, with its wall and CPU seconds; a raising fit writes lost rows."""
    law, clusters, replicate, fit = key
    start, cpu = time.perf_counter(), time.process_time()
    frame = _frame(law, clusters, replicate)
    try:
        rows = properties.fit_rows(
            fit, law, clusters, replicate, study.fit_one(fit, frame), PILOT_DRAWS
        )
        lost = ""
    except Exception as error:
        lost = f"{type(error).__name__}: {str(error)[:80]}"
        rows = [
            {
                "property": study.FAMILY,
                "cell": study.cell_name(fit, law, clusters, arm),
                "role": role,
                "replicate": replicate,
                "n": study.CLUSTER_SIZE[law] * clusters,
                "truth": study.law_truth(law),
                "estimate": np.nan,
                "std_error": np.nan,
                "covered": 0,
                "rejected": 0,
                "withheld": 0,
                "sl_single": np.nan,
            }
            for arm, role in study.ARMS[fit]
        ]
    seconds, cpu_seconds = time.perf_counter() - start, time.process_time() - cpu
    return [{**row, "lost": lost, "seconds": seconds, "cpu_seconds": cpu_seconds} for row in rows]


def run_chunk(directory: Path, chunk: _Chunk) -> Path:
    """Fit one chunk and write its checkpoint atomically."""
    law, clusters, fit, start = chunk
    stop = min(start + CHUNK_DRAWS, PILOT_DRAWS)
    rows = [row for r in range(start, stop) for row in key_rows((law, clusters, r, fit))]
    path = chunk_path(directory, chunk)
    partial = path.with_name(path.name.removesuffix(".csv.gz") + ".partial.csv.gz")
    pd.DataFrame(rows).to_csv(partial, index=False, float_format="%.17g", compression="gzip")
    os.replace(partial, path)
    return path


def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run(directory: Path, jobs: int) -> pd.DataFrame:
    """Fit every chunk not yet checkpointed, then gather every pilot row.

    Parameters
    ----------
    directory : pathlib.Path
        The output directory; its ``chunks/`` folder holds the checkpoints.
    jobs : int
        Worker processes.

    Returns
    -------
    pandas.DataFrame
        Every pilot row, sorted by cell and replicate.
    """
    (directory / "chunks").mkdir(parents=True, exist_ok=True)
    todo = [chunk for chunk in pilot_chunks() if not chunk_path(directory, chunk).exists()]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(run_chunk, directory, chunk) for chunk in todo]
        for done, future in enumerate(as_completed(futures), start=1):
            print(f"{done}/{len(todo)} chunks: {future.result().name}", flush=True)
    frames = [pd.read_csv(chunk_path(directory, chunk)) for chunk in pilot_chunks()]
    rows = pd.concat(frames, ignore_index=True)
    return rows.sort_values(["cell", "replicate"], ignore_index=True)


# ---------------------------------------------------------------- the readings of the pilot

#: The two-sided normal quantile of the declared 99% intervals.
Z_99: Final[float] = float(stats.norm.ppf(0.995))


def _cp_lower(hits: np.ndarray, trials: int) -> np.ndarray:
    hits = np.asarray(hits)
    safe = np.clip(hits, 1, None)
    low = stats.beta.ppf(0.005, safe, trials - safe + 1)
    return np.where(hits > 0, low, 0.0)


def _cp_upper(hits: np.ndarray, trials: int) -> np.ndarray:
    hits = np.asarray(hits)
    safe = np.clip(hits, None, trials - 1)
    high = stats.beta.ppf(0.995, safe + 1, trials - safe)
    return np.where(hits < trials, high, 1.0)


def _predictive(hits: int, trials: int, study_trials: int) -> np.ndarray:
    """The predictive law of the study's hit count: beta-binomial under the Jeffreys prior.

    Exact, so a pass probability carries no Monte Carlo error of its own (a 400-draw posterior
    sample moved the primary's figure across 0.95).
    """
    k = np.arange(study_trials + 1)
    return np.asarray(stats.betabinom.pmf(k, study_trials, hits + 0.5, trials - hits + 0.5))


def lower_end_pass(hits: int, trials: int, study_trials: int, floor: float) -> float:
    """Predictive P(exact 99% lower end of a rate over ``study_trials`` at least ``floor``)."""
    clears = _cp_lower(np.arange(study_trials + 1), study_trials) >= floor
    return float(_predictive(hits, trials, study_trials)[clears].sum())


def upper_end_fails(hits: int, trials: int, study_trials: int, ceiling: float) -> float:
    """Predictive P(exact 99% upper end of a rate over ``study_trials`` below ``ceiling``)."""
    clears = _cp_upper(np.arange(study_trials + 1), study_trials) < ceiling
    return float(_predictive(hits, trials, study_trials)[clears].sum())


def normal_pass(estimate: float, pilot_se: float, study_se: float, threshold: float) -> float:
    """Predictive P(study estimate minus ``Z_99`` study SEs at least ``threshold``).

    The true value is normal about the pilot estimate with the pilot SE, and the study estimate
    is normal about the true value with the study SE.
    """
    spread = float(np.hypot(pilot_se, study_se))
    return float(stats.norm.cdf((estimate - threshold - Z_99 * study_se) / spread))


def log_variance_ratio(estimate: np.ndarray, std_error: np.ndarray) -> tuple[float, float]:
    """The log of mean(SE^2)/Var(estimate) and its delta-method SE over these draws."""
    estimate = np.asarray(estimate, dtype=float)
    s2 = np.asarray(std_error, dtype=float) ** 2
    n = estimate.size
    centred = (estimate - estimate.mean()) ** 2
    variance = centred.sum() / (n - 1)
    # Influence of log(mean s2) - log(var psi), per draw.
    terms = s2 / s2.mean() - centred / variance
    return float(np.log(s2.mean() / variance)), float(terms.std(ddof=1) / np.sqrt(n))


def _lost(frame: pd.DataFrame) -> pd.Series:
    """Whether each row's fit raised: a non-empty ``lost`` text."""
    return frame["lost"].notna() & (frame["lost"].astype(str) != "")


def pilot_readings(
    rows: pd.DataFrame, study_draws: int = study.PROPERTY_REPLICATES
) -> pd.DataFrame:
    """The pilot reading and predictive pass probability of each declared clause, by cell.

    One row per ``t_reference`` cell (gated or reported) and per control. A lost fit reads a
    miss in coverage, as the declared rule's worst case does. Columns: the pilot statistic,
    ``p_<clause>`` the predictive probability that the clause passes at ``study_draws`` draws,
    and ``p_passed`` the product over a gated row's clauses (absent on a reported row).
    """
    out = []
    groups = dict(tuple(rows.groupby("cell", sort=False)))
    for cell, group in groups.items():
        fit, law, clusters, arm = properties.parse(str(cell))
        role = str(group["role"].iloc[0])
        if role == "diagnostic" and arm != "t_reference":
            continue
        n = len(group)
        lost = int(_lost(group).sum())
        covered = int((group["covered"] > 0).sum())
        record: dict[str, Any] = {
            "cell": cell,
            "fit": fit,
            "law": law,
            "clusters": clusters,
            "arm": arm,
            "role": role,
            "draws": n,
            "lost": lost,
            "coverage": covered / n,
            "seconds": float(group["seconds"].mean()),
            "cpu_seconds": float(group["cpu_seconds"].mean())
            if "cpu_seconds" in group
            else float("nan"),
        }
        kept = group.loc[~_lost(group)]
        shown = kept.loc[kept["withheld"] < 0.5]
        record["withheld"] = float(1 - len(shown) / max(len(kept), 1))
        if arm == "t_reference":
            record["p_g1"] = lower_end_pass(covered, n, study_draws, 0.90)
            shown_hits = int((shown["covered"] > 0).sum())
            record["shown_coverage"] = shown_hits / max(len(shown), 1)
            shown_study = round(study_draws * len(shown) / n)
            record["p_shown"] = lower_end_pass(
                shown_hits, len(shown), shown_study, properties.SHOWN_COVERAGE_FLOOR
            )
            errors = (kept["estimate"] - kept["truth"]).to_numpy(dtype=float)
            bias = float(errors.mean() / errors.std(ddof=1))
            record["standardized_bias"] = bias
            pilot_se, study_se = 1 / np.sqrt(len(errors)), 1 / np.sqrt(study_draws)
            record["p_g2"] = normal_pass(-abs(bias), pilot_se, study_se, -0.25) - (
                1 - normal_pass(abs(bias), pilot_se, study_se, -0.25)
            )
            ratio, ratio_se = log_variance_ratio(shown["estimate"], shown["std_error"])
            record["variance_ratio"] = float(np.exp(ratio))
            record["p_variance_ratio"] = normal_pass(
                ratio,
                ratio_se,
                ratio_se * np.sqrt(len(shown) / shown_study),
                float(np.log(properties.VARIANCE_RATIO_FLOOR)),
            )
            single = group["sl_single"].to_numpy(dtype=float)
            single = single[np.isfinite(single)]
            record["sl_single_share"] = float(single.mean()) if single.size else float("nan")
            # G1 and the shown-coverage clause read one hit count when nothing is withheld, and
            # a shown lower end of 0.92 then implies an all-draw one of 0.90: their joint
            # probability is the shown clause's, on the one predictive hit law. With intervals
            # withheld the two counts differ, and the product is kept as a lower bound.
            if len(shown) == n:
                record["p_coverage"] = lower_end_pass(
                    covered,
                    n,
                    study_draws,
                    max(study.STUDY.margins.coverage_floor, properties.SHOWN_COVERAGE_FLOOR),
                )
            else:
                record["p_coverage"] = record["p_g1"] * record["p_shown"]
            passes = ["p_coverage", "p_g2", "p_variance_ratio"]
            glm = groups.get(properties.glm_cell(str(cell)))
            if glm is not None:
                # The paired difference is published on every default-library row (L1, L2, L3);
                # its clause is gated on the stacked L1 rows only.
                paired = group.merge(glm, on="replicate", suffixes=("", "_glm"))
                gained = int(((paired["covered"] > 0) & (paired["covered_glm"] <= 0)).sum())
                dropped = int(((paired["covered"] <= 0) & (paired["covered_glm"] > 0)).sum())
                trials = len(paired)
                diff = (gained - dropped) / trials
                discord = (gained + dropped) / trials
                unit = float(np.sqrt(max(discord - diff**2, 1e-12)))
                record["noninferiority_difference"] = diff
                record["p_noninferiority"] = normal_pass(
                    diff,
                    unit / np.sqrt(trials),
                    unit / np.sqrt(study_draws),
                    properties.NONINFERIORITY_MARGIN,
                )
                if properties.noninferiority_gated(str(cell)):
                    passes.append("p_noninferiority")
            if role == "positive":
                record["p_passed"] = float(np.prod([record[p] for p in passes]))
        elif arm == "iid_t_control":
            positive = groups[study.cell_name(fit, law, clusters, "t_reference")]
            paired = kept.merge(positive, on="replicate", suffixes=("", "_t"))
            ratio = paired["std_error"] / paired["std_error_t"]
            log_ratio = float(np.log(paired["std_error"].mean() / paired["std_error_t"].mean()))
            per = paired["std_error"] / paired["std_error"].mean() - paired["std_error_t"] / (
                paired["std_error_t"].mean()
            )
            se = float(per.std(ddof=1) / np.sqrt(len(paired)))
            record["paired_se_ratio"] = float(np.exp(log_ratio))
            record["median_draw_ratio"] = float(ratio.median())
            first = 1 - normal_pass(
                log_ratio,
                se,
                se * np.sqrt(len(paired) / study_draws),
                float(np.log(properties.FEW_CLUSTER_IID_CONTROL_CEILING)),
            )
            t_kept = positive.loc[~_lost(positive)]
            errors = (t_kept["estimate"] - t_kept["truth"]).to_numpy(dtype=float)
            se_ratio, se_ratio_se = _log_se_ratio(errors, t_kept["std_error"].to_numpy())
            record["positive_se_ratio"] = float(np.exp(se_ratio))
            second = 1 - normal_pass(
                se_ratio,
                se_ratio_se,
                se_ratio_se * np.sqrt(len(errors) / study_draws),
                float(np.log(STUDY_SANITY_CEILING)),
            )
            record["p_passed"] = first * second
        else:
            record["p_passed"] = upper_end_fails(
                covered, n, study_draws, properties.MUST_FAIL_CEILING
            )
        out.append(record)
    return pd.DataFrame(out)


#: The upper end of the SE-ratio sanity band, the IID control's second clause.
STUDY_SANITY_CEILING: Final[float] = study.STUDY.margins.se_ratio_sanity[1]


def _log_se_ratio(errors: np.ndarray, std_error: np.ndarray) -> tuple[float, float]:
    """The log of mean(SE)/SD(error) and its delta-method SE over these draws."""
    errors = np.asarray(errors, dtype=float)
    se = np.asarray(std_error, dtype=float)
    centred = (errors - errors.mean()) ** 2
    variance = centred.sum() / (errors.size - 1)
    terms = se / se.mean() - 0.5 * centred / variance
    return float(np.log(se.mean() / np.sqrt(variance))), float(
        terms.std(ddof=1) / np.sqrt(errors.size)
    )


#: The coverage a cell at target reads (the nominal level of every declared interval).
NOMINAL: Final[float] = 1.0 - study.STUDY.margins.alpha


def at_target_pass(
    rows: pd.DataFrame, cell: str, study_draws: int = study.PROPERTY_REPLICATES
) -> dict[str, float]:
    """Each clause's pass probability for a positive cell at target, at ``study_draws`` draws.

    At target the variance ratio is 1, coverage is nominal and the paired difference to the GLM
    arm is 0. The spreads are the pilot's: the log variance ratio's delta-method SE
    and the paired discordant share, each scaled to ``study_draws``. A figure below 0.90 raises
    the cell to 4,800 draws. The non-inferiority figure is ``nan`` on a row that does
    not carry the clause.
    """
    groups = dict(tuple(rows.groupby("cell", sort=False)))
    group = groups[cell]
    k = np.arange(study_draws + 1)
    at_nominal = stats.binom.pmf(k, study_draws, NOMINAL)
    lower = _cp_lower(k, study_draws)
    _, ratio_se = log_variance_ratio(group["estimate"], group["std_error"])
    spread = ratio_se * np.sqrt(len(group) / study_draws)
    floor = -float(np.log(properties.VARIANCE_RATIO_FLOOR))
    out = {
        "g1": float(at_nominal[lower >= study.STUDY.margins.coverage_floor].sum()),
        "shown": float(at_nominal[lower >= properties.SHOWN_COVERAGE_FLOOR].sum()),
        "variance_ratio": float(stats.norm.cdf((floor - Z_99 * spread) / spread)),
        "noninferiority": float("nan"),
    }
    if properties.noninferiority_gated(cell):
        counts = properties.paired_counts(group, groups[properties.glm_cell(cell)])
        discord = (counts["gained"] + counts["lost"]) / counts["trials"]
        unit = float(np.sqrt(discord / study_draws))
        margin = -properties.NONINFERIORITY_MARGIN
        out["noninferiority"] = float(stats.norm.cdf((margin - Z_99 * unit) / unit))
    return out


def plug_in_pass(
    rows: pd.DataFrame, cell: str, study_draws: int = study.PROPERTY_REPLICATES
) -> float:
    """A positive cell's pass probability at ``study_draws`` with the pilot readings as true.

    The product of G1, shown coverage, the variance ratio, G2 and, on a row that carries it, the
    paired non-inferiority clause, each at its pilot value without the pilot's own error. The
    declared rule reads :func:`pilot_readings`, which carries that error; this figure is
    published beside it.
    """
    groups = dict(tuple(rows.groupby("cell", sort=False)))
    group = groups[cell]
    n = len(group)
    k = np.arange(study_draws + 1)
    covered = float((group["covered"] > 0).mean())
    law = stats.binom.pmf(k, study_draws, covered)
    lower = _cp_lower(k, study_draws)
    g1 = float(law[lower >= study.STUDY.margins.coverage_floor].sum())
    shown = float(law[lower >= properties.SHOWN_COVERAGE_FLOOR].sum())
    ratio, ratio_se = log_variance_ratio(group["estimate"], group["std_error"])
    spread = ratio_se * np.sqrt(n / study_draws)
    floor = float(np.log(properties.VARIANCE_RATIO_FLOOR))
    variance = float(stats.norm.cdf((ratio - floor - Z_99 * spread) / spread))
    errors = (group["estimate"] - group["truth"]).to_numpy(dtype=float)
    bias, unit = float(errors.mean() / errors.std(ddof=1)), 1 / np.sqrt(study_draws)
    margin = study.STUDY.margins.standardized_bias
    g2 = float(
        stats.norm.cdf((margin - bias) / unit - Z_99)
        - stats.norm.cdf((-margin - bias) / unit + Z_99)
    )
    out = g1 * shown * variance * g2
    if properties.noninferiority_gated(cell):
        counts = properties.paired_counts(group, groups[properties.glm_cell(cell)])
        gained, dropped = counts["gained"] / n, counts["lost"] / n
        diff = gained - dropped
        sd = float(np.sqrt(gained + dropped - diff**2) / np.sqrt(study_draws))
        out *= float(stats.norm.cdf((diff - properties.NONINFERIORITY_MARGIN - Z_99 * sd) / sd))
    return float(out)


#: The pilot cell of the primary's configuration (L1, 20 clusters, stacked default library).
PRIMARY_CELL: Final[str] = study.cell_name(
    "tmle_stacked", "equal10", study.PRIMARY_CLUSTERS, "t_reference"
)
#: The arm-pooled rule's step, smallest primary and family-wise target: a primary has at least
#: 1,000 draws, and the smallest multiple of 400 with a family-wise pass of at least 0.95.
PRIMARY_STEP: Final[int] = 400
PRIMARY_MINIMUM: Final[int] = 1_000
FAMILY_TARGET: Final[float] = 0.95


def primary_pass(rows: pd.DataFrame, replicates: int) -> dict[str, float]:
    """The primary's predictive pass probability at ``replicates``, leg by leg.

    The primary is one row (the ATE of one scenario) with three legs
    (:func:`tests.studies.evidence.performance.independent_performance_tests`): the exact 99%
    coverage lower end at least 0.90, the 99% bias interval inside 0.25 SD, and the 99% SE-ratio
    interval inside the sanity band. Each is predicted from :data:`PRIMARY_CELL` with the
    pilot's own error: coverage beta-binomial, the log SE ratio and the standardized bias
    normal. ``family`` is the product, the family-wise figure of the one-row family.
    """
    group = rows.loc[rows["cell"] == PRIMARY_CELL]
    n = len(group)
    hits = int((group["covered"] > 0).sum())
    margins = study.STUDY.margins
    g1 = lower_end_pass(hits, n, replicates, margins.coverage_floor)
    errors = (group["estimate"] - group["truth"]).to_numpy(dtype=float)
    ratio, ratio_se = _log_se_ratio(errors, group["std_error"].to_numpy(dtype=float))
    study_se = ratio_se * np.sqrt(n / replicates)
    low, high = (float(np.log(v)) for v in margins.se_ratio_sanity)
    se_ratio = normal_pass(ratio, ratio_se, study_se, low) - (
        1 - normal_pass(-ratio, ratio_se, study_se, -high)
    )
    bias = float(errors.mean() / errors.std(ddof=1))
    pilot_se, run_se = 1 / np.sqrt(n), 1 / np.sqrt(replicates)
    margin = -margins.standardized_bias
    g2 = normal_pass(-abs(bias), pilot_se, run_se, margin) - (
        1 - normal_pass(abs(bias), pilot_se, run_se, margin)
    )
    return {"g1": g1, "se_ratio": se_ratio, "bias": g2, "family": g1 * se_ratio * g2}


def primary_replicates(rows: pd.DataFrame, cap: int = 8_000) -> int:
    """The smallest multiple of :data:`PRIMARY_STEP`, at least :data:`PRIMARY_MINIMUM`, whose
    family-wise :func:`primary_pass` reaches :data:`FAMILY_TARGET`."""
    start = -(-PRIMARY_MINIMUM // PRIMARY_STEP) * PRIMARY_STEP
    for replicates in range(start, cap + 1, PRIMARY_STEP):
        if primary_pass(rows, replicates)["family"] >= FAMILY_TARGET:
            return replicates
    raise ValueError(f"no primary count up to {cap} reaches {FAMILY_TARGET}")


def cpu_seconds(rows: pd.DataFrame) -> dict[tuple[str, str, int], float]:
    """The mean CPU seconds of one fit, by ``(fit, law, J)``, over the pilot's keys."""
    parts = rows["cell"].str.split("__")
    keys = rows.assign(fit=parts.str[0], law=parts.str[1], J=parts.str[2].str[1:].astype(int))
    keys = keys.drop_duplicates(["fit", "law", "J", "replicate"])
    means = keys.groupby(["fit", "law", "J"])["cpu_seconds"].mean()
    return {(str(f), str(law), int(j)): float(v) for (f, law, j), v in means.items()}


def projected_cpu_hours(
    rows: pd.DataFrame,
    property_draws: int = study.PROPERTY_REPLICATES,
    primary_draws: int = study.PRIMARY_REPLICATES,
) -> dict[str, float]:
    """The declared run's CPU hours at the pilot's per-fit CPU times, by group, and the total.

    The primary refits the stacked L1 fit at 20 clusters. The summary's bootstraps are not
    counted (under 1 CPU-h).
    """
    seconds = cpu_seconds(rows)
    groups: dict[str, float] = {}
    for (fit, _, _), value in seconds.items():
        group = fit if fit in {"tmle_stacked", "tmle_in_sample", "ltmle_crossfit"} else "other"
        groups[group] = groups.get(group, 0.0) + value * property_draws / 3600
    groups["primary"] = seconds[("tmle_stacked", "equal10", study.PRIMARY_CLUSTERS)] * (
        primary_draws / 3600
    )
    groups["total"] = sum(groups.values())
    return groups


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="the output directory")
    parser.add_argument("--jobs", type=int, default=1)
    arguments = parser.parse_args(argv)
    rows = run(arguments.output, arguments.jobs)
    rows.to_csv(arguments.output / "pilot.csv.gz", index=False, float_format="%.17g")
    parts = rows["cell"].str.split("__")
    keys = rows.assign(fit=parts.str[0], law=parts.str[1], J=parts.str[2]).drop_duplicates(
        ["fit", "law", "J", "replicate"]
    )
    record = {
        "commit": _commit(),
        "pilot_seed": PILOT_SEED,
        "draws": PILOT_DRAWS,
        "rows": len(rows),
        "lost_rows": int((rows["lost"].notna() & (rows["lost"].astype(str) != "")).sum()),
        "cpu_seconds": float(keys["cpu_seconds"].sum()),
    }
    (arguments.output / "pilot.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
