r"""The pilot of ``few-cluster-adaptive-learners``, run before the declaration on its own stream.

Each draw cell (law and cluster count) is fitted on :data:`PILOT_DRAWS` draws by every fit of
the study except the stacked default-library ``TMLE``, which is fitted on the first
:data:`PILOT_STACKED_DRAWS` of them, because it costs about four times the in-sample fit. The
L3 cell fits both of its fits on :data:`PILOT_LTMLE_DRAWS` draws. The rows are the study's own
(:func:`tests.studies.few_cluster_adaptive_properties.fit_rows`); a fit that raises writes one
row per arm with ``lost`` set and ``covered = 0``.

The pilot budget was about 1.5 CPU-hours (the row thread's cap for this pilot), far below the
plan's 400 draws per cell (22 CPU-hours). Its readings therefore enter the design test as
plug-in pass probabilities with wide uncertainty, and the declaration says so
(``tests/canonical/few_cluster_adaptive_learners/regenerate.py``).

Run with ``python -m tests.studies.few_cluster_adaptive_pilot --output <file.csv.gz> --jobs J``.
"""

from __future__ import annotations

import argparse
import time
import zlib
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from scipy import stats

from tests.studies import few_cluster_adaptive_learners as study
from tests.studies import few_cluster_adaptive_properties as properties

#: The seed of the pilot stream, used by no other study.
PILOT_SEED: Final[int] = 20264583
PILOT_DRAWS: Final[int] = 60
PILOT_STACKED_DRAWS: Final[int] = 20
PILOT_LTMLE_DRAWS: Final[int] = 20


def draw_seed(law: str, clusters: int, replicate: int) -> np.random.SeedSequence:
    """The seed of one pilot draw: the pilot seed, the cell label and the replicate."""
    label = f"{law}/J{clusters}"
    return np.random.SeedSequence([PILOT_SEED, zlib.crc32(label.encode()), replicate])


def pilot_keys() -> list[tuple[str, int, int, str]]:
    """Every ``(law, J, replicate, fit)`` the pilot fits, the expensive fits first."""
    out = []
    for law, clusters in study.DRAW_CELLS:
        if law == study.LTMLE_LAW:
            out += [
                (law, clusters, r, fit)
                for r in range(PILOT_LTMLE_DRAWS)
                for fit in study.LTMLE_FITS
            ]
            continue
        for r in range(PILOT_DRAWS):
            for fit in study.POINT_FITS:
                if fit == "tmle_stacked" and r >= PILOT_STACKED_DRAWS:
                    continue
                out.append((law, clusters, r, fit))
    heavy = {"tmle_stacked": 0, "ltmle_crossfit": 0, "tmle_in_sample": 1}
    return sorted(out, key=lambda key: (heavy.get(key[3], 2), -key[1]))


def _frame(law: str, clusters: int, replicate: int) -> pd.DataFrame:
    seed = int(draw_seed(law, clusters, replicate).generate_state(1)[0])
    return study.draw(law, clusters, seed)


def key_rows(key: tuple[str, int, int, str]) -> list[dict[str, Any]]:
    """The rows of one pilot key, with its wall time; a raising fit writes lost rows."""
    law, clusters, replicate, fit = key
    start = time.perf_counter()
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
    seconds = time.perf_counter() - start
    return [{**row, "lost": lost, "seconds": seconds} for row in rows]


def run(jobs: int) -> pd.DataFrame:
    """Every pilot row."""
    keys = pilot_keys()
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        chunks = list(pool.map(key_rows, keys, chunksize=1))
    rows = pd.DataFrame([row for chunk in chunks for row in chunk])
    return rows.sort_values(["cell", "replicate"], ignore_index=True)


# ---------------------------------------------------------------- the readings of the pilot

#: Posterior draws per predictive pass probability, and their seed.
POSTERIOR_DRAWS: Final[int] = 400
POSTERIOR_SEED: Final[int] = 0
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


def _rates(hits: int, trials: int, rng: np.random.Generator) -> np.ndarray:
    """Posterior draws of a rate under the Jeffreys prior."""
    return rng.beta(hits + 0.5, trials - hits + 0.5, POSTERIOR_DRAWS)


def lower_end_pass(hits: int, trials: int, study_trials: int, floor: float, seed: int = 0) -> float:
    """Predictive P(exact 99% lower end of a rate over ``study_trials`` at least ``floor``)."""
    k = np.arange(study_trials + 1)
    clears = _cp_lower(k, study_trials) >= floor
    rates = _rates(hits, trials, np.random.default_rng(seed))
    return float(np.mean([stats.binom.pmf(k, study_trials, p)[clears].sum() for p in rates]))


def upper_end_fails(
    hits: int, trials: int, study_trials: int, ceiling: float, seed: int = 0
) -> float:
    """Predictive P(exact 99% upper end of a rate over ``study_trials`` below ``ceiling``)."""
    k = np.arange(study_trials + 1)
    clears = _cp_upper(k, study_trials) < ceiling
    rates = _rates(hits, trials, np.random.default_rng(seed))
    return float(np.mean([stats.binom.pmf(k, study_trials, p)[clears].sum() for p in rates]))


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

    One row per positive and per control. A lost fit reads a miss in coverage, as the declared
    rule's worst case does. Columns: the pilot statistic, and ``p_<clause>`` the predictive
    probability that the clause passes at ``study_draws`` draws.
    """
    out = []
    groups = dict(tuple(rows.groupby("cell", sort=False)))
    for cell, group in groups.items():
        fit, law, clusters, arm = properties.parse(str(cell))
        role = str(group["role"].iloc[0])
        if role == "diagnostic":
            continue
        n = len(group)
        lost = int(_lost(group).sum())
        seed = zlib.crc32(str(cell).encode())
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
        }
        kept = group.loc[~_lost(group)]
        shown = kept.loc[kept["withheld"] < 0.5]
        record["withheld"] = float(1 - len(shown) / max(len(kept), 1))
        if arm == "t_reference":
            record["p_g1"] = lower_end_pass(covered, n, study_draws, 0.90, seed)
            shown_hits = int((shown["covered"] > 0).sum())
            record["shown_coverage"] = shown_hits / max(len(shown), 1)
            shown_study = round(study_draws * len(shown) / n)
            record["p_shown"] = lower_end_pass(
                shown_hits, len(shown), shown_study, properties.SHOWN_COVERAGE_FLOOR, seed
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
            passes = ["p_g1", "p_shown", "p_g2", "p_variance_ratio"]
            if properties.noninferiority_gated(str(cell)):
                glm = groups.get(properties.glm_cell(str(cell)))
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
                passes.append("p_noninferiority")
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
                covered, n, study_draws, properties.MUST_FAIL_CEILING, seed
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


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    arguments = parser.parse_args(argv)
    rows = run(arguments.jobs)
    rows.to_csv(arguments.output, index=False, float_format="%.17g")


if __name__ == "__main__":
    main()
