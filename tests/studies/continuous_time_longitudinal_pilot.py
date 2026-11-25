r"""The pilot and the pre-declaration readings of ``continuous-time-longitudinal`` (plan 5.7).

It draws no declared sample.  Draw ``r`` of a pilot cell takes
``stream_seed(STUDY, "pilot", cell, r)``; the truths take the stream ``truth``, the plug-in limit
the stream ``plugin_limit``.  The driver writes one file beside this module,
``continuous_time_longitudinal_pilot.csv.gz``, with a ``kind`` column:

* ``primary``: one fit of a pilot cell (:data:`CELLS`: ``day_cf``, ``cont_cf``,
  ``mechanism_correct`` and ``outcome_correct``, :data:`DRAWS` each), one row per target with the
  targeted estimate, its SE, coverage and the untargeted plug-in, the support readings
  (``variance_rel_sd``, ``bounded_share``, ``followers_at_last``, ``past_cap_share``) and the
  visit-cap readings (``cap``, ``reference_cap`` and ``capped_truth``, review S3), or one row with
  the error class when the fit raised a recordable error (any other error stops the pilot);
* ``truth``: the Monte Carlo risks of each law and regime, one row per cause and time, from
  :data:`~tests.studies.continuous_time_longitudinal_law.TRUTH_UNITS` units in chunks
  (:func:`truth_seeds`); the law module's ``TRUTH_RISKS`` must equal them;
* ``truth_capped`` and ``truth_panel``: the capped risks :math:`\psi^{(K)}` for every cap in
  ``law.CAPS`` (stream ``truth_capped``) and the risk the X70 panel identifies (stream
  ``truth_panel``), on as many units; ``CAPPED_RISKS`` and ``PANEL_RISKS`` must equal them;
* ``feasibility``: the truth-only feasibility readings of a draw of :data:`FEASIBILITY_N`
  CT-day units (``law.feasibility``);
* ``plugin_limit``: one draw of :data:`PLUGIN_N` CT-day units fitted once with
  ``mechanism_correct``'s configuration (:data:`PLUGIN_FOLDS` folds), its untargeted plug-in at
  each day-30 risk of always and never against the truth, and the SD of the true-weight IPW
  estimator at :math:`n = 1{,}000` on the same draw (``law.ipw_sd``).

The committed rows give these readings, which ``regenerate.pilot_refusals`` checks before a
declared run:

* **the anchored family pass** (:func:`anchored_family_pass`): the counts are declared, 3,200
  primary and 2,400 ``mechanism_correct`` fits (Eric's choice, 2026-11-25), and the product of
  the 60 gated rows' plug-in pass probabilities (:func:`row_pass`, rows taken as independent) at
  those counts must reach ``FAMILY_WISE_PASS`` 0.95, each row read at its group's pooled
  coverage, SE ratio and standardized bias (:func:`pooled_table`: ``cont_cf`` static, ``day_cf``
  static, ``day_cf`` rule, ``mechanism_correct``).  The product with each row's own bias and the
  pooled coverage and SE ratio (:func:`per_row_family_pass`, :func:`own_passes`), the row-level
  product (:func:`family_pass`) and every rung's passes (:func:`anchored_passes`) are reports;
  no rung is chosen from them, and the plan's move of rows with coverage below 0.925 to
  reported is not applied;
* **the at-target family pass** (:func:`at_target_pass`): both calibration rows at SE ratio 1
  and coverage 0.95 at the primary count (0.995 for the pair at 3,200), and each calibration
  row's plug-in pass at the declared count (:func:`calibration_checks`): a row below 0.90
  extends its cell's pilot to :data:`EXTENSION_DRAWS`, and only a confirmed departure
  (:func:`calibration_departure`) needs a committed diagnosis;
* **the plug-in separation and the IPW-SD witness** (:func:`plugin_separation`): the largest
  bias of the limit over the true-weight IPW SD at day 30 for always or never, at least 0.5;
  the pilot's own ``mechanism_correct`` plug-in bias gives the control's power at the declared
  count (:func:`control_power`);
* **feasibility** (:func:`feasibility_refusals`), the **shrunk-SE power** at a true SE ratio of
  1 (:func:`shrunken_power`; at the pilot's ratio a report, :func:`shrunken_power_at_pilot`), the
  **failure prediction** (:func:`failure_refusals`) and the
  **seconds per fit and peak memory** (:func:`seconds`, :func:`budget`).

Run it as ``python -m tests.studies.continuous_time_longitudinal_pilot --jobs N [--kinds ...]``
with every BLAS thread variable at 1.  It writes the file after every block of :data:`BLOCK`
draws, so ``--resume`` continues each cell from its first draw the file does not hold.
"""

from __future__ import annotations

import argparse
import itertools
import resource
import time
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import betabinom, norm

from cleverly.inference.sparse import influence_row
from cleverly.utils.parallel import map_parallel
from cleverly.validation._failures import recordable
from tests.studies import continuous_time_longitudinal as study
from tests.studies import continuous_time_longitudinal_law as laws
from tests.studies import repeated_sizing as sizing
from tests.studies.evidence.inference import clopper_pearson
from tests.studies.evidence.performance import se_ratio_interval
from tests.studies.evidence.seeds import stream_seed

OUTPUT = Path(__file__).with_name("continuous_time_longitudinal_pilot.csv.gz")
DRAWS = study.PILOT_REPLICATES
#: A calibration cell whose row passes below :data:`PLUG_IN_PASS` extends to this many draws.
EXTENSION_DRAWS = 800
BLOCK = 50
PLUG_IN_PASS = 0.90
#: The plug-in separation the declaration needs (review S8), in true-weight IPW SDs.
SEPARATION = 0.5
FAILURE_EXCEEDANCE = 0.10
FEASIBILITY_N = 1_000_000
PLUGIN_N = 200_000
#: The folds of the plug-in limit fit: a user learner is refused in sample (C13), and two folds
#: of 100,000 units reach the misspecified side's limit as ten would.
PLUGIN_FOLDS = 2
DIAGNOSIS = study.STUDY.artifacts / "pilot-diagnosis.md"

#: ``pilot cell -> (law, regimes, configuration)``.
CELLS: dict[str, tuple[str, tuple[str, ...], str]] = {
    "day_cf": ("ct_day", tuple(laws.REGIMES), "default"),
    "cont_cf": ("ct", tuple(laws.STATIC), "default"),
    "mechanism_correct": ("ct_day", tuple(laws.STATIC), "mechanism_correct"),
    "outcome_correct": ("ct_day", tuple(laws.STATIC), "outcome_correct"),
}
#: ``pilot cell -> the gated rows it sizes`` (``outcome_correct`` sizes none).
GATED: dict[str, tuple[str, ...]] = {
    "day_cf": study.DAY_ROWS,
    "cont_cf": study.CONT_ROWS,
    "mechanism_correct": study.ROBUST_ROWS,
}


def pilot_seed(cell: str, replicate: int) -> int:
    """The seed of pilot draw ``replicate`` of a cell."""
    return stream_seed(study.STUDY, "pilot", cell, replicate)


def truth_seeds(law: str, regime: str) -> list[int]:
    """The seeds of the truth chunks of one law and regime, from the stream ``truth``."""
    chunks = laws.TRUTH_UNITS // laws.TRUTH_CHUNK
    return [stream_seed(study.STUDY, "truth", law, regime, chunk) for chunk in range(chunks)]


def capped_seeds(law: str, regime: str, cap: int) -> list[int]:
    r"""The seeds of the chunks of :math:`\psi^{(K)}`, from the stream ``truth_capped``."""
    chunks = laws.TRUTH_UNITS // laws.TRUTH_CHUNK
    return [
        stream_seed(study.STUDY, "truth_capped", law, regime, cap, chunk) for chunk in range(chunks)
    ]


def panel_seeds(regime: str) -> list[int]:
    """The seeds of the chunks of the panel's risk, from the stream ``truth_panel``."""
    chunks = laws.TRUTH_UNITS // laws.TRUTH_CHUNK
    return [stream_seed(study.STUDY, "truth_panel", regime, chunk) for chunk in range(chunks)]


def plugin_seed() -> int:
    """The seed of the plug-in limit's draw."""
    return stream_seed(study.STUDY, "plugin_limit")


def feasibility_seed() -> int:
    """The seed of the feasibility draw, on the stream ``truth``."""
    return stream_seed(study.STUDY, "truth", "feasibility")


def _peak_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)


def _primary(cell: str, replicate: int) -> list[dict[str, Any]]:
    law, regimes, configuration = CELLS[cell]
    truth = laws.TRUTHS[law]
    frame = laws.SAMPLERS[law](study.PRIMARY_N, pilot_seed(cell, replicate))
    head = {"kind": "primary", "label": cell, "replicate": replicate, "n": study.PRIMARY_N}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start = time.perf_counter()
            result = study.fit_with(frame, study.engine(regimes, configuration))
            seconds = time.perf_counter() - start
    except Exception as error:
        # A recordable failure (a DataError refusal of the draw, a numerical failure) is a pilot
        # finding, recorded with its class; any other error is a defect and stops the pilot.
        if not recordable(error):
            raise
        return [{**head, "refusal": type(error).__name__}]
    initial = result.initial_estimates
    peak = _peak_mb()
    rows = []
    for estimand in laws.targets(regimes):
        estimate = result[estimand]
        low, high = estimate.ci
        value = truth[estimand]
        rows.append(
            {
                **head,
                "estimand": estimand,
                "truth": float(value),
                "estimate": float(estimate.psi),
                "std_error": float(estimate.std_error),
                "covered": int(low <= value <= high),
                "initial_estimate": float(initial[estimand]),
                "refusal": "",
                "seconds": seconds,
                "peak_mb": peak,
                "converged": float(result.record.converged),
                "bounded_share": float(result.record.bounded_share),
                "variance_rel_sd": float(
                    influence_row(estimand, estimate.influence_curve).variance_rel_sd
                ),
                **study._regime_readings(result, estimand),
                **study.cap_readings(result, estimand, law),
            }
        )
    return rows


def _truth_chunk(
    law: str, regime: str, seed: int, cap: int | None = None, panel: bool = False
) -> list[dict[str, Any]]:
    resets = laws.PANEL_RESETS if panel else ()
    table = laws.monte_carlo_risks(law, regime, [seed], cap=cap, resets=resets)
    label = f"panel/{regime}" if panel else f"{law}/{regime}" + ("" if cap is None else f"/{cap}")
    return [
        {
            "label": label,
            "replicate": seed,
            "estimand": f"{cause}@{t:g}",
            "estimate": value,
            "n": units,
        }
        for (cause, t), (value, units) in table.items()
    ]


def _averaged(keys: list[tuple[Any, ...]], kind: str, n_jobs: int) -> pd.DataFrame:
    chunks = pd.DataFrame(
        [row for rows in map_parallel(_truth_chunk, keys, n_jobs=n_jobs) for row in rows]
    )
    out = (
        chunks.groupby(["label", "estimand"], sort=True)
        .agg(estimate=("estimate", "mean"), n=("n", "sum"))
        .reset_index()
    )
    out["kind"] = kind
    out["replicate"] = 0
    return out


def truth_rows(n_jobs: int) -> pd.DataFrame:
    """The Monte Carlo risks of every law and regime, averaged over the chunks."""
    keys = [
        (law, regime, seed)
        for law in laws.SAMPLERS
        for regime in laws.REGIMES
        for seed in truth_seeds(law, regime)
    ]
    return _averaged(keys, "truth", n_jobs)


def capped_rows(n_jobs: int) -> pd.DataFrame:
    r""":math:`\psi^{(K)}` of every law, regime and cap in ``law.CAPS`` (review S1)."""
    keys = [
        (law, regime, seed, cap)
        for law in laws.SAMPLERS
        for regime in laws.REGIMES
        for cap in laws.CAPS
        for seed in capped_seeds(law, regime, cap)
    ]
    return _averaged(keys, "truth_capped", n_jobs)


def panel_rows(n_jobs: int) -> pd.DataFrame:
    """The risk the X70 panel identifies, per static regime (review S5)."""
    keys = [
        ("ct_day", regime, seed, None, True)
        for regime in laws.STATIC
        for seed in panel_seeds(regime)
    ]
    return _averaged(keys, "truth_panel", n_jobs)


def _risk_table(rows: pd.DataFrame, kind: str) -> dict[str, dict[tuple[int, float], float]]:
    """``label -> (cause, time) -> risk`` of one kind of truth rows."""
    out: dict[str, dict[tuple[int, float], float]] = {}
    for _, row in rows.loc[rows["kind"] == kind].iterrows():
        cause, t = str(row["estimand"]).split("@")
        # A risk on 10^7 units is a multiple of 10^-7; rounding drops the mean's float noise.
        out.setdefault(str(row["label"]), {})[(int(cause), float(t))] = round(
            float(row["estimate"]), 7
        )
    return out


def capped_table(
    rows: pd.DataFrame,
) -> dict[str, dict[str, dict[int, dict[tuple[int, float], float]]]]:
    """``law -> regime -> cap -> (cause, time) -> risk`` of the committed capped rows."""
    out: dict[str, dict[str, dict[int, dict[tuple[int, float], float]]]] = {}
    for label, table in _risk_table(rows, "truth_capped").items():
        law, regime, cap = label.split("/")
        out.setdefault(law, {}).setdefault(regime, {})[int(cap)] = table
    return out


def panel_table(rows: pd.DataFrame) -> dict[str, dict[tuple[int, float], float]]:
    """``regime -> (cause, time) -> risk`` of the committed panel rows."""
    return {label.split("/")[1]: table for label, table in _risk_table(rows, "truth_panel").items()}


def truth_table(rows: pd.DataFrame) -> dict[str, dict[str, dict[tuple[int, float], float]]]:
    """``law -> regime -> (cause, time) -> risk`` of the committed truth rows."""
    out: dict[str, dict[str, dict[tuple[int, float], float]]] = {}
    for label, table in _risk_table(rows, "truth").items():
        law, regime = label.split("/")
        out.setdefault(law, {})[regime] = table
    return out


def feasibility_rows() -> pd.DataFrame:
    """The truth-only feasibility readings on a draw of :data:`FEASIBILITY_N` CT-day units."""
    paths = laws.day_paths(laws.sample_day(FEASIBILITY_N, feasibility_seed()))
    table = laws.feasibility(paths)
    long = table.melt(id_vars="regime", var_name="estimand", value_name="estimate")
    long["kind"] = "feasibility"
    long["label"] = long.pop("regime")
    long["replicate"] = 0
    long["n"] = FEASIBILITY_N
    return long


def plugin_rows(n: int = PLUGIN_N) -> pd.DataFrame:
    """The plug-in limit of ``mechanism_correct``'s outcome side and the IPW witness."""
    frame = laws.sample_day(n, plugin_seed())
    paths = laws.day_paths(frame)
    subject = study.engine(tuple(laws.STATIC), "mechanism_correct")
    subject.n_folds = PLUGIN_FOLDS
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = study.fit_with(frame, subject)
    seconds = time.perf_counter() - start
    truth = laws.TRUTHS["ct_day"]
    records = []
    for name in study.ROBUST_ROWS:
        limit = float(result.initial_estimates[name])
        sd = laws.ipw_sd(paths, name, study.PRIMARY_N)
        records.append(
            {
                "kind": "plugin_limit",
                "label": "mechanism_correct",
                "replicate": 0,
                "n": n,
                "estimand": name,
                "estimate": limit,
                "truth": float(truth[name]),
                "std_error": sd,
                "seconds": seconds,
                "peak_mb": _peak_mb(),
            }
        )
    return pd.DataFrame.from_records(records)


def _one(cell: str, replicate: int) -> list[dict[str, Any]]:
    return _primary(cell, replicate)


#: The kinds the driver runs.
KINDS = ("truth", "truth_capped", "truth_panel", "feasibility", "plugin_limit", "primary")


def run(
    n_jobs: int,
    kinds: Sequence[str] = KINDS,
    *,
    resume: bool = False,
    draws: int | None = None,
    cells: Sequence[str] = tuple(CELLS),
    plugin_n: int = PLUGIN_N,
) -> pd.DataFrame:
    """Run each kind, writing the file after every block.

    With ``resume`` each cell starts at its first draw the file does not hold and every other
    row is kept; a rerun of ``truth``, ``feasibility`` or ``plugin_limit`` replaces that kind's
    rows.  ``draws`` replaces :data:`DRAWS` (``--draws 800 --cells cont_cf --resume`` is the
    declared extension).
    """
    frame = pd.read_csv(OUTPUT, low_memory=False) if resume and OUTPUT.exists() else pd.DataFrame()

    def replace(kind: str, rows: pd.DataFrame) -> pd.DataFrame:
        kept = frame.loc[frame["kind"] != kind] if not frame.empty else frame
        out = pd.concat([kept, rows], ignore_index=True)
        out.to_csv(OUTPUT, index=False)
        return out

    for kind in kinds:
        if kind == "truth":
            frame = replace("truth", truth_rows(n_jobs))
            print("truth: written", flush=True)
        elif kind == "truth_capped":
            frame = replace("truth_capped", capped_rows(n_jobs))
            print("truth_capped: written", flush=True)
        elif kind == "truth_panel":
            frame = replace("truth_panel", panel_rows(n_jobs))
            print("truth_panel: written", flush=True)
        elif kind == "feasibility":
            frame = replace("feasibility", feasibility_rows())
            print("feasibility: written", flush=True)
        elif kind == "plugin_limit":
            frame = replace("plugin_limit", plugin_rows(plugin_n))
            print("plugin_limit: written", flush=True)
        elif kind == "primary":
            count = DRAWS if draws is None else draws
            for cell in cells:
                held = of(frame, cell) if not frame.empty else frame
                first = int(held["replicate"].max()) + 1 if not held.empty else 0
                if not held.empty and set(held["replicate"].astype(int)) != set(range(first)):
                    raise SystemExit(
                        f"{OUTPUT.name} does not hold draws 0 to {first - 1} of {cell}"
                    )
                for start in range(first, count, BLOCK):
                    keys = [(cell, r) for r in range(start, min(start + BLOCK, count))]
                    outcomes = map_parallel(_one, keys, n_jobs=n_jobs)
                    block = pd.DataFrame([row for rows in outcomes for row in rows])
                    frame = pd.concat([frame, block], ignore_index=True)
                    frame.to_csv(OUTPUT, index=False)
                    print(f"primary {cell}: {min(start + BLOCK, count)} of {count}", flush=True)
        else:
            raise SystemExit(f"unknown kind {kind!r}; the kinds are {KINDS}")
    return frame


# -------------------------------------------------------------------------- readings


def read() -> pd.DataFrame:
    """The committed pilot rows, or an empty frame before the pilot is committed."""
    return pd.read_csv(OUTPUT, low_memory=False) if OUTPUT.exists() else pd.DataFrame()


def of(rows: pd.DataFrame, cell: str) -> pd.DataFrame:
    """The primary rows of one pilot cell."""
    return rows.loc[(rows["kind"] == "primary") & (rows["label"] == cell)]


def kept_rows(rows: pd.DataFrame, cell: str) -> pd.DataFrame:
    """The rows of a cell's fits that did not raise."""
    chosen = of(rows, cell)
    return chosen.loc[chosen["refusal"].fillna("").eq("")]


def draws(rows: pd.DataFrame, cell: str) -> int:
    """The draws a cell holds, raised or not."""
    return int(of(rows, cell)["replicate"].nunique()) if not rows.empty else 0


def failed_draws(rows: pd.DataFrame, cell: str) -> int:
    """The draws of a cell whose fit raised."""
    chosen = of(rows, cell)
    return int(chosen.loc[chosen["refusal"].fillna("").ne(""), "replicate"].nunique())


def coverage_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Each pilot row's coverage, SE ratio, standardized bias and plug-in standardized bias."""
    records = []
    for cell in CELLS:
        kept = kept_rows(rows, cell)
        for estimand, group in kept.groupby("estimand", sort=False):
            error = group["estimate"] - group["truth"]
            spread = float(group["estimate"].std(ddof=1))
            initial = group["initial_estimate"] - group["truth"]
            records.append(
                {
                    "cell": cell,
                    "estimand": estimand,
                    "gated": estimand in GATED.get(cell, ()),
                    "fits": len(group),
                    "coverage": float(group["covered"].mean()),
                    "se_ratio": float(group["std_error"].mean() / spread),
                    "standardized_bias": float(error.mean() / spread),
                    "plug_in_standardized_bias": float(
                        initial.mean() / group["initial_estimate"].std(ddof=1)
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def se_band_pass(replicates: int, ratio: float) -> float:
    """P(the 99% interval of an SE ratio lies inside the primary band (0.8, 1.2)).

    The normal reading of ``repeated_sizing.calibration_pass``'s SE half, on the primary band.
    """
    low, high = study.MARGINS.se_ratio_sanity
    z = float(norm.ppf(0.5 + study.MARGINS.confidence_level / 2.0))
    spread = ratio / np.sqrt(2.0 * (replicates - 1))
    half = z * spread
    return float(
        max(norm.cdf((high - half - ratio) / spread) - norm.cdf((low + half - ratio) / spread), 0.0)
    )


def row_pass(replicates: int, coverage: float, ratio: float, bias: float) -> float:
    """A gated row's plug-in pass probability at ``replicates``: coverage, SE ratio and bias.

    Coverage above 0.99 is conservative, not failed, so only the floor clause reads coverage.
    """
    margins = study.MARGINS
    return (
        sizing.coverage_floor_pass_probability(margins, replicates, min(coverage, 1.0))
        * se_band_pass(replicates, ratio)
        * float(sizing.bias_equivalent_probability(margins, bias, replicates))
    )


def family_pass(rows: pd.DataFrame, primary: int, mechanism: int) -> float:
    """The product of the 60 gated rows' plug-in pass probabilities at the two counts."""
    table = coverage_table(rows)
    product = 1.0
    for _, row in table.loc[table["gated"]].iterrows():
        count = mechanism if row["cell"] == "mechanism_correct" else primary
        product *= row_pass(count, row["coverage"], row["se_ratio"], row["standardized_bias"])
    return product


def seconds(rows: pd.DataFrame) -> dict[str, float]:
    """The median seconds per fit of each pilot cell, on its first :data:`DRAWS` draws."""
    out = {}
    for cell in CELLS:
        fits = kept_rows(rows, cell).drop_duplicates("replicate")
        fits = fits.loc[fits["replicate"] < DRAWS]
        out[cell] = float(fits["seconds"].median()) if not fits.empty else np.nan
    return out


def peak_mb(rows: pd.DataFrame) -> dict[str, float]:
    """The largest worker peak memory a pilot cell's fits recorded, in MB."""
    return {cell: float(kept_rows(rows, cell)["peak_mb"].max()) for cell in CELLS}


def cost(rows: pd.DataFrame, primary: int, mechanism: int) -> float:
    """The CPU-hours of the declared run at the two counts, at the pilot's seconds per fit.

    ``outcome_correct`` runs at the reported count at its pilot's seconds; ``both_wrong`` and the
    X70 pairing, which the pilot does not run, cost the smoke's measured seconds.
    """
    return sum(budget(rows, primary, mechanism).values())


#: The X70 pairing's seconds per fit, from the smoke (``pilot-diagnosis.md``).
SMOKE_X70_SECONDS = 51.0
#: ``both_wrong``'s seconds per fit, from the smoke.
SMOKE_BOTH_WRONG_SECONDS = 2.4


#: The groups of the anchored reading: the static and rule rows of ``day_cf``, ``cont_cf`` and
#: ``mechanism_correct``.
GROUPS = ("cont_cf static", "day_cf static", "day_cf rule", "mechanism_correct")
_STATIC_ROWS = frozenset(laws.targets(tuple(laws.STATIC)))


def group_of(cell: str, estimand: str) -> str:
    """The anchored group of a gated row: its cell, and for ``day_cf`` static or rule."""
    if cell == "mechanism_correct":
        return cell
    return f"{cell} {'static' if estimand in _STATIC_ROWS else 'rule'}"


def pooled_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Each group's gated rows, mean coverage, mean SE ratio and mean standardized bias."""
    table = coverage_table(rows)
    gated = table.loc[table["gated"]].copy()
    gated["group"] = [
        group_of(cell, name) for cell, name in zip(gated["cell"], gated["estimand"], strict=True)
    ]
    pooled = gated.groupby("group", sort=False).agg(
        rows=("estimand", "size"),
        coverage=("coverage", "mean"),
        se_ratio=("se_ratio", "mean"),
        standardized_bias=("standardized_bias", "mean"),
    )
    return pooled.reindex(list(GROUPS)).reset_index()


def anchored_family_pass(rows: pd.DataFrame, primary: int, mechanism: int) -> float:
    """The product of the 60 gated rows' plug-in pass probabilities, each read at its group's
    pooled coverage, SE ratio and standardized bias (the anchored reading).

    A 400-draw row reads its coverage with an SE near 0.011 and its SE ratio with a relative SD
    near 0.035, so a product of 60 row-level point readings is pessimistic.  No earlier study
    pooled readings across rows: the ``continuous-survival-bootstrap`` pilot anchored each row's
    SE ratio to an external study row by row and read no bias.  Averaging signed biases lets
    opposite biases cancel within a group, so :func:`per_row_family_pass`, which reads each row's
    own bias, is reported beside it.
    """
    pooled = pooled_table(rows).set_index("group")
    product = 1.0
    for group, row in pooled.iterrows():
        count = mechanism if group == "mechanism_correct" else primary
        product *= row_pass(
            count, row["coverage"], row["se_ratio"], row["standardized_bias"]
        ) ** int(row["rows"])
    return float(product)


def own_passes(rows: pd.DataFrame, primary: int, mechanism: int) -> pd.DataFrame:
    """Each gated row's plug-in pass probability at the two counts, read at its group's pooled
    coverage and SE ratio and at its own pilot standardized bias (the per-row bias reading).

    Coverage and the SE ratio plausibly share a group value up to pilot noise; bias is a property
    of each estimand, and a signed group mean lets opposite biases cancel.
    """
    table = coverage_table(rows)
    gated = table.loc[table["gated"]].reset_index(drop=True)
    pooled = pooled_table(rows).set_index("group")
    records = []
    for cell, name, bias in zip(
        gated["cell"], gated["estimand"], gated["standardized_bias"], strict=True
    ):
        group = pooled.loc[group_of(cell, name)]
        count = mechanism if cell == "mechanism_correct" else primary
        records.append(
            {
                "cell": cell,
                "estimand": name,
                "standardized_bias": float(bias),
                "own_pass": row_pass(count, group["coverage"], group["se_ratio"], float(bias)),
            }
        )
    return pd.DataFrame.from_records(records)


def per_row_family_pass(rows: pd.DataFrame, primary: int, mechanism: int) -> float:
    """The product of the 60 gated rows' :func:`own_passes`: a report beside the anchored pass."""
    return float(np.prod(own_passes(rows, primary, mechanism)["own_pass"].to_numpy()))


def anchored_passes(rows: pd.DataFrame) -> pd.DataFrame:
    """The anchored, per-row bias and row-level family pass and the cost of every pair of rungs:
    a report.

    The counts are declared (Eric's choice, 2026-11-25), so no rung is chosen from it.
    """
    records = [
        {
            "primary": primary,
            "mechanism": mechanism,
            "anchored": anchored_family_pass(rows, primary, mechanism),
            "per_row_bias": per_row_family_pass(rows, primary, mechanism),
            "row_level": family_pass(rows, primary, mechanism),
            "cpu_hours": cost(rows, primary, mechanism),
        }
        for primary, mechanism in itertools.product(study.PRIMARY_RUNGS, study.MECHANISM_RUNGS)
        if mechanism <= primary
    ]
    return pd.DataFrame.from_records(records)


def at_target_pass(replicates: int) -> float:
    """Both calibration rows passing together at SE ratio 1 and coverage 0.95."""
    one = float(sizing.calibration_pass(study.MARGINS, replicates, np.array([1.0]), 0.95)[0])
    return one ** len(study.CALIBRATION_CELLS)


def calibration_checks(rows: pd.DataFrame, replicates: int) -> pd.DataFrame:
    """Each calibration row's pilot SE ratio and coverage, and its plug-in pass at the count."""
    table = coverage_table(rows)
    records = []
    for cell in study.CALIBRATION_CELLS:
        row = table.loc[(table["cell"] == cell) & (table["estimand"] == study.CALIBRATION_ROW)]
        ratio = float(row["se_ratio"].iloc[0])
        passed = float(sizing.calibration_pass(study.MARGINS, replicates, np.array([ratio]))[0])
        records.append(
            {
                "cell": cell,
                "se_ratio": ratio,
                "coverage": float(row["coverage"].iloc[0]),
                "plug_in_pass": passed,
            }
        )
    return pd.DataFrame.from_records(records)


def shrunken_power(replicates: int) -> float:
    """The probability that a shrunk-SE control leaves the band below at ``replicates``, at a
    true SE ratio of 1 (plan 5's preface: the shrunk control is detected at target).

    It reads no pilot row: a 400-draw SE ratio has a relative SD near 0.035, so a power read at
    the pilot's point ratio would refuse a calibrated estimator by chance (review B1).
    """
    return float(
        sizing.calibration_control_probability(
            study.MARGINS, replicates, study.SHRUNKEN_SE_FACTOR * 1.0
        )
    )


def shrunken_power_at_pilot(rows: pd.DataFrame, replicates: int) -> dict[str, float]:
    """The same probability at each calibration row's pilot SE ratio: a report, never a refusal.

    Above a ratio near 1.01 the band reading falls below 0.999; the declared run then reads a
    conservative row's control against the row's own interval (``readings.calibration``).
    """
    checks = calibration_checks(rows, replicates)
    return {
        cell: float(
            sizing.calibration_control_probability(
                study.MARGINS, replicates, study.SHRUNKEN_SE_FACTOR * ratio
            )
        )
        for cell, ratio in zip(checks["cell"], checks["se_ratio"], strict=True)
    }


def calibration_departure(rows: pd.DataFrame, cell: str) -> bool:
    """Whether a calibration row's pilot departs from calibration beyond chance (review S4).

    True when the 99% interval of the pilot's SE ratio excludes 1 or the 99% Clopper-Pearson
    interval of its coverage excludes 0.95.  Only such a row needs a committed diagnosis after
    the extension; any other row below the plug-in pass of 0.90 is a chance reading.
    """
    chosen = kept_rows(rows, cell)
    chosen = chosen.loc[chosen["estimand"] == study.CALIBRATION_ROW]
    values = chosen["estimate"].to_numpy(dtype=float)
    errors = chosen["std_error"].to_numpy(dtype=float)
    ratio = se_ratio_interval(
        values,
        errors,
        record=study.STUDY,
        seed=stream_seed(study.STUDY, "pilot", "calibration", cell),
    )
    covered = chosen["covered"].to_numpy(dtype=int)
    coverage = clopper_pearson(
        int(covered.sum()), int(covered.size), confidence_level=study.MARGINS.confidence_level
    )
    return bool(not (ratio.low <= 1.0 <= ratio.high) or not (coverage.low <= 0.95 <= coverage.high))


def plugin_separation(rows: pd.DataFrame) -> pd.DataFrame:
    """The plug-in limit's bias over the true-weight IPW SD at :math:`n = 1{,}000`, per row."""
    chosen = rows.loc[rows["kind"] == "plugin_limit"].copy()
    chosen["separation"] = (chosen["estimate"] - chosen["truth"]).abs() / chosen["std_error"]
    chosen["control_row"] = chosen["estimand"].isin(study.PLUG_IN_ROWS)
    return chosen.loc[
        :, ["estimand", "estimate", "truth", "std_error", "separation", "control_row"]
    ]


def separated(rows: pd.DataFrame) -> bool | None:
    """Whether some day-30 risk of always or never separates by 0.5 SD; ``None`` before the run."""
    table = plugin_separation(rows)
    if table.empty:
        return None
    return bool(table.loc[table["control_row"], "separation"].max() >= SEPARATION)


def control_power(rows: pd.DataFrame, replicates: int) -> float:
    """The probability that the plug-in control fires at ``replicates``, at the pilot's biases.

    It fires when any one of its rows does, so the largest row's probability bounds it below.
    """
    table = coverage_table(rows)
    chosen = table.loc[
        (table["cell"] == "mechanism_correct") & table["estimand"].isin(study.PLUG_IN_ROWS)
    ]
    return max(
        sizing.bias_discriminated_probability(study.MARGINS, bias, replicates)
        for bias in chosen["plug_in_standardized_bias"]
    )


def feasibility_refusals(rows: pd.DataFrame) -> list[str]:
    """Each feasibility reading outside the plan's bounds (plan 5.7)."""
    table = rows.loc[rows["kind"] == "feasibility"]
    if table.empty:
        return ["the pilot holds no feasibility rows; run --kinds feasibility"]
    out = []
    for _, row in table.iterrows():
        name, value = str(row["estimand"]), float(row["estimate"])
        if name.startswith("events_") and value < laws.FEASIBLE_EVENTS:
            out.append(f"{row['label']} expects {value:.1f} {name}, below {laws.FEASIBLE_EVENTS:g}")
        if name == "weight_p99" and value > laws.FEASIBLE_WEIGHT:
            out.append(f"{row['label']}'s 99th weight percentile is {value:.1f}, above 20")
    return out


def predicted_exceedance(failed: int, pilot_draws: int, replicates: int, allowance: int) -> float:
    """The posterior predictive probability that more than ``allowance`` of ``replicates`` fail."""
    law = betabinom(replicates, failed + 0.5, pilot_draws - failed + 0.5)
    return float(law.sf(allowance))


def failure_refusals(rows: pd.DataFrame, primary: int, mechanism: int) -> list[str]:
    """One refusal per gated cell whose predicted failure bound exceeds the threshold."""
    out = []
    for cell, count in (
        ("day_cf", primary),
        ("cont_cf", primary),
        ("mechanism_correct", mechanism),
    ):
        held = draws(rows, cell)
        if not held:
            out.append(f"the pilot holds no draws of {cell}; run its pilot")
            continue
        allowance = int(np.floor(study.FAILURE_BOUND * count))
        failed = failed_draws(rows, cell)
        probability = predicted_exceedance(failed, held, count, allowance)
        if probability > FAILURE_EXCEEDANCE:
            out.append(
                f"{cell} lost {failed} of {held} pilot fits, and the declared run of {count:,} "
                f"loses more than {allowance} with predictive probability {probability:.3f}; "
                "diagnose and fix the refusal and rerun the cell's pilot"
            )
    return out


def budget(rows: pd.DataFrame, primary: int, mechanism: int) -> dict[str, float]:
    """The declared run's CPU-hours per item at the pilot's seconds per fit."""
    per = seconds(rows)
    return {
        "day_cf": primary * per["day_cf"] / 3_600,
        "cont_cf": primary * per["cont_cf"] / 3_600,
        "mechanism_correct": mechanism * per["mechanism_correct"] / 3_600,
        "outcome_correct": study.REPORTED_REPLICATES * per["outcome_correct"] / 3_600,
        "both_wrong": study.REPORTED_REPLICATES * SMOKE_BOTH_WRONG_SECONDS / 3_600,
        "ltmle_daily": study.PAIRED_REPLICATES * SMOKE_X70_SECONDS / 3_600,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--kinds", nargs="*", default=list(KINDS))
    parser.add_argument("--cells", nargs="*", default=list(CELLS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--draws", type=int, default=None)
    arguments = parser.parse_args()
    run(
        arguments.jobs,
        arguments.kinds,
        resume=arguments.resume,
        draws=arguments.draws,
        cells=arguments.cells,
    )
