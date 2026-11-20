"""The pilot of the two ``continuous-survival-bootstrap`` records, on its own seed streams.

It draws no declared sample.  Draw ``r`` of a kind takes ``stream_seed(STUDY, "pilot", kind, r)``
on the in-sample record, a stream no declared cell of either record reads, and seeds that fit's folds and bootstrap with the same value.  The
driver writes one file beside this module, ``continuous_survival_bootstrap_pilot.csv.gz``, with a
``kind`` column:

* ``in_sample``: one bootstrap fit of ``b1``'s configuration with
  :data:`~tests.studies.continuous_survival_bootstrap.PILOT_BOOTSTRAP` refits, one row per target
  with the Wald, percentile and BCa readings of
  :func:`~tests.studies.continuous_survival_bootstrap.interval_readings`, or one row with the error
  class when the fit raised (a withheld bootstrap included);
* ``cross_fitted``: the same for the cross-fitted record
  (:mod:`tests.studies.continuous_survival_bootstrap_cross_fitted`), for its failures and seconds;
* ``warp``: the warp-speed screen of Giacomini, Politis and White (2013): one fit of ``b1``'s
  configuration and one bootstrap refit of it per draw.  It never gates (plan 5.5).

Every row holds all 18 targets; the in-sample record gates the 12 at the median and upper quartile.
The committed rows give these readings of X6 plan rev 5, section 5.5, which each record's
regeneration script checks before a declared run (``regenerate.pilot_refusals``):

* **the count** (:func:`at_target_count`): the family-wise rule at the design point, the smallest
  multiple of 400 at which every gated calibration row of ``b1`` passes together with probability at
  least 0.95 at SE ratio 1 and coverage 0.95.  A pilot of 400 leaves each row's SE ratio with a
  predictive SD near half the band, so the rule takes ``guide/studies.md``'s at-target branch;
* **each gated row's plug-in pass** (:func:`calibration_checks`) at the declared count, from its pilot
  SE ratio and coverage: a row below 0.90 asks for a committed diagnosis before the declaration;
* **the failures** (:func:`failure_refusal`): the posterior predictive probability, under a
  Jeffreys prior, that a declared cell loses more fits than its bound allows, at most 0.10 for the
  gated in-sample cell; the cross-fitted record publishes its reading;
* **the shrunk-SE control's power** (:func:`shrunken_power`), and **seconds per fit**
  (:func:`seconds`) for the budget.

Run it as ``python -m tests.studies.continuous_survival_bootstrap_pilot --jobs N [--kinds ...]
[--draws D] [--resume]``.  It writes the file after every block of :data:`BLOCK` draws, so
``--resume`` continues each kind from its first draw the file does not hold, on the same streams.
"""

from __future__ import annotations

import argparse
import time
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import betabinom

from cleverly.utils.parallel import map_parallel
from tests.studies import continuous_survival as registered
from tests.studies import continuous_survival_bootstrap as study
from tests.studies import continuous_survival_bootstrap_cross_fitted as cross_fitted
from tests.studies import continuous_survival_law as law
from tests.studies import repeated_sizing as sizing
from tests.studies.evidence.seeds import stream_seed

OUTPUT = Path(__file__).with_name("continuous_survival_bootstrap_pilot.csv.gz")
#: ``kind -> (cell, declared draws)``; the warp-speed screen is no declared cell.
KINDS: dict[str, tuple[str, int]] = {
    "in_sample": (study.GATED_CELL, study.PILOT_REPLICATES),
    "cross_fitted": (cross_fitted.CELL, cross_fitted.PILOT_REPLICATES),
    "warp": (study.GATED_CELL, 4_000),
}
#: ``kind -> (record module, declared replicates)`` of the two bootstrap kinds.
RECORDS: dict[str, tuple[Any, int]] = {
    "in_sample": (study, study.PRIMARY_REPLICATES),
    "cross_fitted": (cross_fitted, cross_fitted.PRIMARY_REPLICATES),
}
#: Whether a kind's failure prediction refuses its record's declared run.  The cross-fitted
#: record's failure bound gates nothing (``reported_bounds_gate=False``), so its reading is
#: published, not a refusal.
FAILURES_REFUSE: dict[str, bool] = {"in_sample": True, "cross_fitted": False}
BLOCK = 20
#: The plug-in pass probability below which ``guide/studies.md`` asks for a committed diagnosis.
PLUG_IN_PASS = 0.90
#: The largest posterior predictive probability of exceeding a failure bound that may proceed.
FAILURE_EXCEEDANCE = 0.10
DIAGNOSIS = study.STUDY.artifacts / "pilot-diagnosis.md"


def pilot_seed(kind: str, replicate: int) -> int:
    """The seed of pilot draw ``replicate`` of ``kind``."""
    return stream_seed(study.STUDY, "pilot", kind, replicate)


def _bootstrap(kind: str, replicate: int) -> list[dict[str, Any]]:
    cell, _ = KINDS[kind]
    record = RECORDS[kind][0]
    seed = pilot_seed(kind, replicate)
    frame = registered.sample(study.PRIMARY_N, seed)
    head = {"kind": kind, "label": cell, "replicate": replicate, "n": study.PRIMARY_N}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start = time.perf_counter()
            result = study.bootstrap_fit(
                frame,
                cross_fit=record.CELLS[cell],
                n_bootstrap=study.PILOT_BOOTSTRAP,
                seed=seed,
            )
            seconds = time.perf_counter() - start
    except Exception as error:  # a raise is a pilot finding, recorded with its class
        return [{**head, "refusal": type(error).__name__, "message": str(error)[:200]}]
    readings = study.interval_readings(result, law.TRUTH, study.ALL_ROWS)
    return [
        {
            **head,
            "estimand": name,
            "truth": float(law.TRUTH[name]),
            **reading,
            "failed_refits": float(result.bootstrap.n_failed),
            "refusal": "",
            "seconds": seconds,
        }
        for name, reading in readings.items()
    ]


def _warp(replicate: int) -> list[dict[str, Any]]:
    """One draw of the warp-speed screen: the fit and one bootstrap refit of it."""
    seed = pilot_seed("warp", replicate)
    frame = registered.sample(study.PRIMARY_N, seed)
    head = {"kind": "warp", "label": "b1_in_sample", "replicate": replicate, "n": study.PRIMARY_N}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # The engine takes at least two refits; the screen reads the first.
            result = registered.fit_with(
                frame, study.engine(cross_fit=False, n_bootstrap=2, seed=seed)
            )
    except Exception as error:  # a raise is a pilot finding, recorded with its class
        return [{**head, "refusal": type(error).__name__, "message": str(error)[:200]}]
    if result.bootstrap is None or result.bootstrap.n_failed:
        return [{**head, "refusal": "BootstrapFailure", "message": ""}]
    return [
        {
            **head,
            "estimand": name,
            "truth": float(law.TRUTH[name]),
            "estimate": float(result[name].psi),
            "bootstrap_draw": float(result.bootstrap.draws[name][0]),
            "refusal": "",
        }
        for name in study.ALL_ROWS
    ]


def _one(kind: str, replicate: int) -> list[dict[str, Any]]:
    return _warp(replicate) if kind == "warp" else _bootstrap(kind, replicate)


def run(
    n_jobs: int,
    kinds: Sequence[str] = ("in_sample", "cross_fitted"),
    *,
    resume: bool = False,
    draws: int | None = None,
) -> pd.DataFrame:
    """Run each kind's draws, writing the file after every block.

    With ``resume`` each kind starts at its first draw the file does not hold and every other row
    is kept; without it, the file is replaced.  ``draws`` replaces the kind's declared count.
    """
    frame = pd.read_csv(OUTPUT, low_memory=False) if resume and OUTPUT.exists() else pd.DataFrame()
    for kind in kinds:
        count = draws if draws is not None else KINDS[kind][1]
        held = frame.loc[frame["kind"] == kind] if not frame.empty else frame
        first = int(held["replicate"].max()) + 1 if not held.empty else 0
        if not held.empty and set(held["replicate"].astype(int)) != set(range(first)):
            raise SystemExit(f"{OUTPUT.name} does not hold draws 0 to {first - 1} of {kind}")
        for start in range(first, count, BLOCK):
            keys = [(kind, r) for r in range(start, min(start + BLOCK, count))]
            outcomes = map_parallel(_one, keys, n_jobs=n_jobs)
            block = pd.DataFrame([row for rows in outcomes for row in rows])
            frame = pd.concat([frame, block], ignore_index=True)
            frame.to_csv(OUTPUT, index=False)
            print(f"{kind}: {min(start + BLOCK, count)} of {count}", flush=True)
    return frame


# -------------------------------------------------------------------------- readings


def read() -> pd.DataFrame:
    """The committed pilot rows, or an empty frame before the pilot is committed."""
    return pd.read_csv(OUTPUT, low_memory=False) if OUTPUT.exists() else pd.DataFrame()


def of(rows: pd.DataFrame, kind: str) -> pd.DataFrame:
    """The rows of one kind."""
    return rows.loc[rows["kind"] == kind] if not rows.empty else rows


def kept_rows(rows: pd.DataFrame, kind: str) -> pd.DataFrame:
    """The rows of ``kind``'s fits that did not raise."""
    chosen = of(rows, kind)
    return chosen.loc[chosen["refusal"].fillna("").eq("")]


def draws(rows: pd.DataFrame, kind: str) -> int:
    """The draws ``kind`` holds, raised or not."""
    chosen = of(rows, kind)
    return int(chosen["replicate"].nunique()) if not chosen.empty else 0


def failed_draws(rows: pd.DataFrame, kind: str) -> int:
    """The draws of ``kind`` whose fit raised."""
    chosen = of(rows, kind)
    if chosen.empty:
        return 0
    return int(chosen.loc[chosen["refusal"].fillna("").ne(""), "replicate"].nunique())


def at_target_count(rows: int = len(study.ROWS)) -> int:
    """The smallest multiple of 400 at which ``rows`` calibration rows pass together with 0.95.

    Each row at SE ratio 1 and coverage 0.95 (``calibration_pass``); rows taken as independent,
    which is conservative for rows that share fits.
    """
    replicates = sizing.BUDGET_STEP
    while (
        float(sizing.calibration_pass(study.MARGINS, replicates, np.array([1.0]), 0.95)[0]) ** rows
        < sizing.FAMILY_WISE_PASS
    ):
        replicates += sizing.BUDGET_STEP
    return replicates


def coverage_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Per bootstrap kind and target: the three intervals' coverage and the two SE ratios."""
    records = []
    for kind in ("in_sample", "cross_fitted"):
        kept = kept_rows(rows, kind)
        if kept.empty:
            continue
        for estimand, group in kept.groupby("estimand", sort=False):
            spread = float(group["estimate"].std(ddof=1))
            records.append(
                {
                    "kind": kind,
                    "estimand": estimand,
                    "fits": len(group),
                    "percentile_coverage": float(group["percentile_covered"].mean()),
                    "bca_coverage": float(group["bca_covered"].mean()),
                    "wald_coverage": float(group["wald_covered"].mean()),
                    "bootstrap_se_ratio": float(group["bootstrap_std_error"].mean() / spread),
                    "wald_se_ratio": float(group["wald_std_error"].mean() / spread),
                    "standardized_bias": float(
                        (group["estimate"] - group["truth"]).mean() / spread
                    ),
                }
            )
    return pd.DataFrame(records)


def calibration_checks(rows: pd.DataFrame, replicates: int) -> pd.DataFrame:
    """Each gated ``b1`` row's pilot SE ratio and coverage, and its plug-in pass at ``replicates``."""
    table = coverage_table(rows)
    table = table.loc[(table["kind"] == "in_sample") & table["estimand"].isin(study.ROWS)]
    passes = [
        float(
            sizing.calibration_pass(
                study.MARGINS, replicates, np.array([ratio]), min(max(coverage, 1e-6), 1 - 1e-6)
            )[0]
        )
        for ratio, coverage in zip(
            table["bootstrap_se_ratio"], table["percentile_coverage"], strict=True
        )
    ]
    return table.assign(plug_in_pass=passes).reset_index(drop=True)


def shrunken_power(rows: pd.DataFrame, replicates: int) -> dict[str, float]:
    """The probability that each row's shrunk-SE control leaves the band at ``replicates``."""
    table = calibration_checks(rows, replicates)
    return {
        str(name): sizing.calibration_control_probability(
            study.MARGINS, replicates, study.SHRUNKEN_SE_FACTOR * float(ratio)
        )
        for name, ratio in zip(table["estimand"], table["bootstrap_se_ratio"], strict=True)
    }


def predicted_exceedance(failed: int, pilot_draws: int, replicates: int, allowance: int) -> float:
    """The posterior predictive probability that more than ``allowance`` of ``replicates`` fail.

    Jeffreys prior on the failure rate, so the count of the declared run is beta-binomial.
    """
    return float(betabinom(replicates, failed + 0.5, pilot_draws - failed + 0.5).sf(allowance))


def failure_exceedance(rows: pd.DataFrame, kind: str) -> float | None:
    """The predictive probability that ``kind``'s declared run breaches its failure bound.

    ``None`` when the pilot holds no draw of ``kind``.
    """
    held = draws(rows, kind)
    if not held:
        return None
    record, replicates = RECORDS[kind]
    allowance = int(np.floor(record.FAILURE_BOUND * replicates))
    return predicted_exceedance(failed_draws(rows, kind), held, replicates, allowance)


def failure_refusal(rows: pd.DataFrame, kind: str) -> list[str]:
    """A refusal when ``kind``'s pilot is missing, or predicts a breach of a bound that gates."""
    probability = failure_exceedance(rows, kind)
    if probability is None:
        return [f"the pilot holds no draws of {kind}; run its pilot"]
    if not FAILURES_REFUSE[kind] or probability <= FAILURE_EXCEEDANCE:
        return []
    record, replicates = RECORDS[kind]
    allowance = int(np.floor(record.FAILURE_BOUND * replicates))
    return [
        f"{kind} lost {failed_draws(rows, kind)} of {draws(rows, kind):,} pilot fits, and the "
        f"declared run of {replicates:,} loses more than the {allowance} its failure bound "
        f"allows with predictive probability {probability:.3f}, above {FAILURE_EXCEEDANCE:.2f}"
    ]


def seconds(rows: pd.DataFrame) -> dict[str, float]:
    """The median seconds per bootstrap fit of each kind, and per single fit within it."""
    out: dict[str, float] = {}
    for kind in ("in_sample", "cross_fitted"):
        fits = kept_rows(rows, kind).drop_duplicates("replicate")
        if not fits.empty:
            median = float(fits["seconds"].median())
            out[kind] = median
            out[f"{kind}_per_fit"] = median / (study.PILOT_BOOTSTRAP + 1)
    return out


def budget_cpu_hours(rows: pd.DataFrame) -> dict[str, float]:
    """Each record's declared CPU hours at its count and :data:`N_BOOTSTRAP` refits, from the pilot."""
    per_fit = seconds(rows)
    return {
        KINDS[kind][0]: replicates * (study.N_BOOTSTRAP + 1) * per_fit[f"{kind}_per_fit"] / 3_600.0
        for kind, (_, replicates) in RECORDS.items()
        if f"{kind}_per_fit" in per_fit
    }


def warp_coverage(rows: pd.DataFrame) -> pd.DataFrame:
    """The warp-speed screen: each target's coverage under the pooled bootstrap quantiles.

    Every draw's interval is ``[estimate + q(alpha/2), estimate + q(1 - alpha/2)]`` with ``q``
    the quantiles of the root ``draw - estimate`` pooled over all draws: the percentile form of
    the pooled interval.  It estimates the coverage of the interval built from the expected
    bootstrap distribution, and cannot see the draw-to-draw variation of the width
    (``rows/X6/sources.md`` section 5); a screen, never a gate.
    """
    kept = kept_rows(rows, "warp")
    alpha = study.STUDY.margins.alpha
    records = []
    for estimand, group in kept.groupby("estimand", sort=False):
        root = group["bootstrap_draw"] - group["estimate"]
        low, high = np.quantile(root, [alpha / 2.0, 1.0 - alpha / 2.0])
        truth = group["truth"]
        covered = (group["estimate"] + low <= truth) & (truth <= group["estimate"] + high)
        records.append({"estimand": estimand, "draws": len(group), "coverage": covered.mean()})
    return pd.DataFrame(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--kinds", nargs="*", default=["in_sample", "cross_fitted"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--draws", type=int, default=None)
    arguments = parser.parse_args()
    run(arguments.jobs, arguments.kinds, resume=arguments.resume, draws=arguments.draws)
